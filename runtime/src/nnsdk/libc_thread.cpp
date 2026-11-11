// pthreads de la musl de Nintendo. Por debajo son objetos de nn::os: pthread_mutex_t empieza por un MutexType y
// pthread_cond_t por un ConditionVariableType (la musl pone 0x20 bytes a cero y llama a nn::os::Initialize*),
// pthread_t es el ThreadType* del hilo y pthread_rwlock_t es un ReaderWriterLockType. Mutex y condiciones van
// igual, sobre nuestro nn::os (os.cpp), que crea el estado al vuelo si el objeto viene de un inicializador
// estático. Los atributos son de 4 bytes (los de bionic, de 8: no se pueden pasar tal cual).
// Lo llaman librerías de terceros (BoringSSL, gRPC…), no el motor; pthread_once va directo a bionic.
// Aquí no se llama a ningún pthread_* de bionic: con el mismo nombre nos llamaríamos a nosotros mismos.

#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "nnsdk/os.h"
#include "nnsdk/types.h"

using namespace nn;

namespace nn::os {
void InitializeMutex(MutexType* m, bool recursive, int lock_level);
void InitializeConditionVariable(ConditionVariableType* c);
void BroadcastConditionVariable(ConditionVariableType* c);
ThreadType* GetCurrentThread();
}  // namespace nn::os

namespace {

os::MutexType* mutex(pthread_mutex_t* m) { return reinterpret_cast<os::MutexType*>(m); }
os::ConditionVariableType* cond(pthread_cond_t* c) { return reinterpret_cast<os::ConditionVariableType*>(c); }
uint32_t* attr_word(void* a) { return static_cast<uint32_t*>(a); }
uint32_t attr_value(const void* a) { return *static_cast<const uint32_t*>(a); }

// Tope de una espera: os.cpp suma el plazo al reloj, y un plazo "infinito" ({LONG_MAX, 0}) se desbordaría.
constexpr int64_t kMaxWaitNs = int64_t{1} << 60;  // ~36 años

int64_t ns(const timespec& ts) {
    if (ts.tv_sec >= kMaxWaitNs / 1'000'000'000) return kMaxWaitNs;
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

// Lo que falta hasta un plazo absoluto. En la Switch todos los relojes de musl (clock_gettime(0), time,
// gettimeofday, system_clock) cuentan desde el arranque; aquí son el de pared, como en Linux (libc.cpp, bionic,
// cxx.cpp), así que el plazo se mide contra él. Uno que no puede ser una fecha (antes de 2001) salió de un reloj
// de arranque (ticks de nn::os, steady_clock): se mide contra el monotónico.
int64_t ns_until(const timespec& deadline) {
    if (deadline.tv_sec < 1'000'000'000) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return ns(deadline) - std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    }
    timespec now;
    timespec_get(&now, TIME_UTC);
    return ns(deadline) - ns(now);
}

// ReaderWriterLockType: el que tiene la escritura puede volver a entrar, no puede pedir lectura (EDEADLK) y unlock
// suelta la escritura si la tiene este hilo y si no una lectura. El juego no mezcla estos con la API de nn::os:
// estado propio por dirección, creado al vuelo (los de BoringSSL son estáticos, sin pthread_rwlock_init).
struct RwLock {
    std::shared_mutex m;
    std::atomic<pid_t> writer{0};
    int depth = 0;  // entradas del escritor; solo lo toca él
};

std::mutex g_rw_m;
std::unordered_map<const void*, std::unique_ptr<RwLock>> g_rw;

RwLock& rwlock(const void* key) {
    std::lock_guard lock(g_rw_m);
    auto& slot = g_rw[key];
    if (!slot) slot = std::make_unique<RwLock>();
    return *slot;
}

}  // namespace

extern "C" {

pthread_t nngame_pthread_self() { return reinterpret_cast<pthread_t>(os::GetCurrentThread()); }

// --- Mutex: el tipo va en los bits 0-1; solo normal (0) y recursivo (1), con ERRORCHECK la de Nintendo da EINVAL ---

int nngame_pthread_mutexattr_init(pthread_mutexattr_t* a) {
    *attr_word(a) = 0;
    return 0;
}
int nngame_pthread_mutexattr_destroy(pthread_mutexattr_t*) { return 0; }
int nngame_pthread_mutexattr_settype(pthread_mutexattr_t* a, int type) {
    if (static_cast<unsigned>(type) > PTHREAD_MUTEX_RECURSIVE) return EINVAL;
    *attr_word(a) = (*attr_word(a) & ~3u) | type;
    return 0;
}

// Recursivo solo si el atributo entero vale 1, como la de Nintendo.
int nngame_pthread_mutex_init(pthread_mutex_t* m, const pthread_mutexattr_t* a) {
    memset(m, 0, 0x20);
    os::InitializeMutex(mutex(m), a && attr_value(a) == 1, 0);
    return 0;
}
int nngame_pthread_mutex_destroy(pthread_mutex_t* m) {
    os::FinalizeMutex(mutex(m));
    return 0;
}
int nngame_pthread_mutex_lock(pthread_mutex_t* m) {
    os::LockMutex(mutex(m));
    return 0;
}
int nngame_pthread_mutex_unlock(pthread_mutex_t* m) {
    os::UnlockMutex(mutex(m));
    return 0;
}

// --- Variables de condición (el atributo no se mira: nn::os no tiene reloj ni pshared) ---

int nngame_pthread_condattr_init(pthread_condattr_t* a) {
    *attr_word(a) = 0;
    return 0;
}
int nngame_pthread_cond_init(pthread_cond_t* c, const pthread_condattr_t*) {
    memset(c, 0, 0x20);
    os::InitializeConditionVariable(cond(c));
    return 0;
}
int nngame_pthread_cond_destroy(pthread_cond_t* c) {
    os::FinalizeConditionVariable(cond(c));
    return 0;
}
int nngame_pthread_cond_signal(pthread_cond_t* c) {
    os::SignalConditionVariable(cond(c));
    return 0;
}
int nngame_pthread_cond_broadcast(pthread_cond_t* c) {
    os::BroadcastConditionVariable(cond(c));
    return 0;
}
int nngame_pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) {
    os::WaitConditionVariable(cond(c), mutex(m));
    return 0;
}
// Sin plazo, la de Nintendo espera 0 ns.
int nngame_pthread_cond_timedwait(pthread_cond_t* c, pthread_mutex_t* m, const timespec* deadline) {
    int64_t left = deadline ? std::clamp<int64_t>(ns_until(*deadline), 0, kMaxWaitNs) : 0;
    auto status = os::TimedWaitConditionVariable(cond(c), mutex(m), TimeSpan{left});
    return status == os::ConditionVariableStatus_NoTimeout ? 0 : ETIMEDOUT;
}

// --- RW locks ---

int nngame_pthread_rwlock_init(pthread_rwlock_t* l, const pthread_rwlockattr_t*) {
    if (!l) return EINVAL;
    std::lock_guard lock(g_rw_m);
    g_rw[l] = std::make_unique<RwLock>();
    return 0;
}
int nngame_pthread_rwlock_destroy(pthread_rwlock_t* l) {
    if (!l) return EINVAL;
    std::lock_guard lock(g_rw_m);
    g_rw.erase(l);
    return 0;
}
int nngame_pthread_rwlock_rdlock(pthread_rwlock_t* l) {
    if (!l) return EINVAL;
    RwLock& rw = rwlock(l);
    if (rw.writer.load(std::memory_order_relaxed) == gettid()) return EDEADLK;
    rw.m.lock_shared();
    return 0;
}
int nngame_pthread_rwlock_wrlock(pthread_rwlock_t* l) {
    if (!l) return EINVAL;
    RwLock& rw = rwlock(l);
    pid_t me = gettid();
    if (rw.writer.load(std::memory_order_relaxed) != me) {
        rw.m.lock();
        rw.writer.store(me, std::memory_order_relaxed);
    }
    rw.depth++;
    return 0;
}
int nngame_pthread_rwlock_unlock(pthread_rwlock_t* l) {
    if (!l) return EINVAL;
    RwLock& rw = rwlock(l);
    if (rw.writer.load(std::memory_order_relaxed) != gettid()) {
        rw.m.unlock_shared();
    } else if (--rw.depth == 0) {
        rw.writer.store(0, std::memory_order_relaxed);
        rw.m.unlock();
    }
    return 0;
}

}  // extern "C"
