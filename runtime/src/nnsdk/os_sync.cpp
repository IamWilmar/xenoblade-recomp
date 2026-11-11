// Las primitivas de nn::os que el SDK implementa dentro de la memoria del propio objeto, sin tabla aparte: el
// juego las declara con inicializadores constexpr (todo a 0 ya es válido), las crea a millares y nunca las
// finaliza (SdkMutex no tiene Finalize). Se respeta el layout del SDK 11.4 y el estado vive en el objeto:
//   SdkMutexType           u32 cs                       (InternalCriticalSection)
//   SdkRecursiveMutexType  u32 cs; u32 count
//   LightEventType         u32 state; u8 auto_clear; u8 initialized; u32 (sin uso)
// Donde Horizon usa ArbitrateLock / WaitForAddress / SignalToAddress sobre esas palabras, aquí se usa futex.

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "nnsdk/os.h"
#include "nnsdk/types.h"

namespace {

int futex_wait(uint32_t* word, uint32_t expected, const timespec* relative) {
    return syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, expected, relative, nullptr, 0) == 0 ? 0 : errno;
}
void futex_wake(uint32_t* word, int count) { syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, count, nullptr, nullptr, 0); }

uint32_t load(const uint32_t* word) { return __atomic_load_n(word, __ATOMIC_ACQUIRE); }
bool cas(uint32_t* word, uint32_t expected, uint32_t desired) {
    return __atomic_compare_exchange_n(word, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

[[noreturn]] void misuse(const char* what) {
    printf("[nnsdk] %s\n", what);
    fflush(stdout);
    abort();
}

// --- InternalCriticalSection ---
// Como en Horizon: 0 = libre; si no, el identificador del dueño (allí el handle del hilo, aquí su tid, que en
// Linux cabe de sobra en 30 bits) con el bit 30 puesto si puede haber alguien esperando.
constexpr uint32_t kWaiters = 0x40000000;

uint32_t self_tag() { return static_cast<uint32_t>(gettid()); }

bool cs_is_locked_by_current_thread(const uint32_t* cs) { return (load(cs) & ~kWaiters) == self_tag(); }

void cs_enter(uint32_t* cs) {
    const uint32_t me = self_tag();
    if (cas(cs, 0, me)) return;
    for (;;) {
        uint32_t v = load(cs);
        if (v == 0) {
            // Quien entra tras una espera deja el bit puesto: puede que quede otro esperando.
            if (cas(cs, 0, me | kWaiters)) return;
            continue;
        }
        if (!(v & kWaiters)) {
            if (!cas(cs, v, v | kWaiters)) continue;
            v |= kWaiters;
        }
        futex_wait(cs, v, nullptr);
    }
}

void cs_leave(uint32_t* cs) {
    if (__atomic_exchange_n(cs, 0, __ATOMIC_RELEASE) & kWaiters) futex_wake(cs, 1);
}

// --- LightEvent ---
// Estados del SDK: 0 sin señalar, 1 sin señalar con alguien esperando, 2 señalado. Señalar con esperas pendientes
// despierta a todos; en AutoClear se quedan la señal los que ganen el 2 -> 0 (uno) y los demás vuelven a dormir,
// que es lo que hace Horizon al despertar a uno solo. En ManualClear el que despierta vuelve aunque otro haya
// limpiado ya (un Signal + Clear seguido libera a los que esperaban, como en Horizon).
enum : uint32_t { kNotSignaled = 0, kWaiting = 1, kSignaled = 2 };

struct LightEvent {
    uint32_t state;
    uint8_t auto_clear;
    uint8_t initialized;
    uint32_t reserved;
};

thread_local int g_light_wake_err = -1;
thread_local uint32_t g_light_wake_state = 0;

// end: plazo absoluto en ns de CLOCK_MONOTONIC, o -1 sin plazo. false si venció.
bool light_wait(LightEvent* e, int64_t end) {
    for (;;) {
        uint32_t s = load(&e->state);
        if (s == kSignaled) {
            if (!e->auto_clear) return true;
            if (cas(&e->state, kSignaled, kNotSignaled)) return true;
            continue;
        }
        if (s == kNotSignaled && !cas(&e->state, kNotSignaled, kWaiting)) continue;
        timespec rel{}, *timeout = nullptr;
        if (end >= 0) {
            const int64_t left = end - now_ns();
            if (left <= 0) return false;
            rel = {static_cast<time_t>(left / 1'000'000'000), static_cast<long>(left % 1'000'000'000)};
            timeout = &rel;
        }
        const int err = futex_wait(&e->state, kWaiting, timeout);
        g_light_wake_err = err;  // DEPURACIÓN: por qué volvió la última espera (lo registra WaitLightEvent)
        g_light_wake_state = load(&e->state);
        if (err == 0 && !e->auto_clear) return true;  // despertado por un Signal
        if (err == ETIMEDOUT) return false;
    }
}

}  // namespace

namespace nn::os {

struct SdkMutexType {
    void Initialize();
    void Lock();
    void Unlock();
    uint32_t cs;
};

struct SdkRecursiveMutexType {
    void Initialize();
    void Lock();
    void Unlock();
    bool IsLockedByCurrentThread() const;
    uint32_t cs;
    uint32_t count;
};

struct LightEventType;

void SdkMutexType::Initialize() { __atomic_store_n(&cs, 0, __ATOMIC_RELEASE); }
void SdkMutexType::Lock() {
    if (cs_is_locked_by_current_thread(&cs)) misuse("SdkMutex::Lock: el hilo ya lo tiene (no es recursivo)");
    cs_enter(&cs);
}
void SdkMutexType::Unlock() {
    if (!cs_is_locked_by_current_thread(&cs)) misuse("SdkMutex::Unlock: el hilo no lo tiene");
    cs_leave(&cs);
}

void SdkRecursiveMutexType::Initialize() {
    __atomic_store_n(&cs, 0, __ATOMIC_RELEASE);
    count = 0;
}
void SdkRecursiveMutexType::Lock() {
    if (!cs_is_locked_by_current_thread(&cs)) cs_enter(&cs);
    count++;
}
void SdkRecursiveMutexType::Unlock() {
    if (!cs_is_locked_by_current_thread(&cs)) misuse("SdkRecursiveMutex::Unlock: el hilo no lo tiene");
    if (--count == 0) cs_leave(&cs);
}
bool SdkRecursiveMutexType::IsLockedByCurrentThread() const { return cs_is_locked_by_current_thread(&cs); }

void InitializeLightEvent(LightEventType* event, bool signaled, EventClearMode mode) {
    auto* e = reinterpret_cast<LightEvent*>(event);
    e->auto_clear = mode == EventClearMode_AutoClear;
    e->reserved = 0;
    e->initialized = 1;
    __atomic_store_n(&e->state, signaled ? kSignaled : kNotSignaled, __ATOMIC_RELEASE);
}
void FinalizeLightEvent(LightEventType* event) { reinterpret_cast<LightEvent*>(event)->initialized = 0; }

void SignalLightEvent(LightEventType* event) {
    nnsdk::log_signal("SignalLightEvent", event, __builtin_return_address(0));
    auto* e = reinterpret_cast<LightEvent*>(event);
    if (__atomic_exchange_n(&e->state, kSignaled, __ATOMIC_ACQ_REL) == kWaiting) futex_wake(&e->state, INT_MAX);
}
void ClearLightEvent(LightEventType* event) { cas(&reinterpret_cast<LightEvent*>(event)->state, kSignaled, kNotSignaled); }
void WaitLightEvent(LightEventType* event) {
    const int64_t start = nnsdk::wait_clock_ns();
    g_light_wake_err = -1;
    light_wait(reinterpret_cast<LightEvent*>(event), -1);
    if (nnsdk::wait_clock_ns() - start > 150'000'000) {
        auto* e = reinterpret_cast<LightEvent*>(event);
        char what[96];
        snprintf(what, sizeof what, "WaitLightEvent[%s err %d estado %u]", e->auto_clear ? "auto" : "manual",
                 g_light_wake_err, g_light_wake_state);
        nnsdk::log_slow_wait(what, event, start, __builtin_return_address(0));
    }
}
bool TimedWaitLightEvent(LightEventType* event, TimeSpan timeout) {
    const int64_t now = now_ns();
    const int64_t end = timeout.ns >= INT64_MAX - now ? -1 : now + (timeout.ns > 0 ? timeout.ns : 0);
    const bool ok = light_wait(reinterpret_cast<LightEvent*>(event), end);
    if (!ok) nnsdk::log_slow_wait("TimedWaitLightEvent caducado", event, now, __builtin_return_address(0));
    return ok;
}

}  // namespace nn::os
