// nn::os (y su API en C, nnos*) sobre pthreads. Cada objeto del juego se asocia por su
// dirección a un objeto nuestro; si el juego usa uno sin inicializarlo (inicializador
// estático), se crea al vuelo con valores seguros (mutex recursivo, evento manual).
// Aparte: los objetos que el SDK implementa en su propia memoria (SdkMutex, LightEvent) en os_sync.cpp, las
// fibras en os_fiber.cpp y el heap de la aplicación en os_memory.cpp.

#include <dirent.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/os.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

using namespace nn;
using namespace nn::os;

namespace nn::os {
struct TimerEventType;
struct MessageQueueType;
}  // namespace nn::os

namespace nvn {
void note_cpu_flush(uint64_t address, uint64_t size);  // nvn/commands.h
}  // namespace nvn

namespace {

constexpr int64_t kForever = INT64_MAX;

int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

timespec deadline_after(int64_t ns) {
    int64_t t = now_ns() + (ns > 0 ? ns : 0);
    return {static_cast<time_t>(t / 1'000'000'000), static_cast<long>(t % 1'000'000'000)};
}

// Plazo absoluto (CLOCK_MONOTONIC) a partir de un TimeSpan, sin desbordar con los "infinitos" del juego.
int64_t deadline_of(TimeSpan t) {
    const int64_t now = now_ns();
    return t.ns >= kForever - now ? kForever : now + std::max<int64_t>(t.ns, 0);
}

// Objeto del juego (por dirección) -> objeto nuestro. 64 trozos para no serializar todo.
template <typename T>
class Registry {
public:
    T& get(const void* key) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        auto& slot = s.map[key];
        if (!slot) slot = std::make_unique<T>();
        return *slot;
    }
    template <typename... Args>
    T& reset(const void* key, Args&&... args) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        auto& slot = s.map[key];
        slot = std::make_unique<T>(std::forward<Args>(args)...);
        return *slot;
    }
    void erase(const void* key) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        s.map.erase(key);
    }
    template <typename Fn>
    void for_each(Fn fn) {
        for (Shard& s : shards_) {
            std::lock_guard lock(s.m);
            for (auto& [key, value] : s.map) fn(*value);
        }
    }

private:
    struct Shard {
        std::mutex m;
        std::unordered_map<const void*, std::unique_ptr<T>> map;
    };
    Shard& shard(const void* key) { return shards_[(reinterpret_cast<uintptr_t>(key) >> 4) % 64]; }
    Shard shards_[64];
};

struct HostMutex {
    pthread_mutex_t m;
    pid_t owner = 0;  // tid de quien lo tiene por LockMutex (IsMutexLockedByCurrentThread); se lee con atómicos
    int depth = 0;    // anidamiento de ese dueño, solo con m tomado
    explicit HostMutex(bool recursive = true) {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_settype(&a, recursive ? PTHREAD_MUTEX_RECURSIVE : PTHREAD_MUTEX_NORMAL);
        pthread_mutex_init(&m, &a);
    }
};

struct HostCond {
    pthread_cond_t c;
    HostCond() {
        pthread_condattr_t a;
        pthread_condattr_init(&a);
        pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
        pthread_cond_init(&c, &a);
    }
    // false si venció el plazo.
    bool wait(pthread_mutex_t* m, int64_t timeout_ns = -1) {
        if (timeout_ns < 0) return pthread_cond_wait(&c, m), true;
        timespec ts = deadline_after(timeout_ns);
        return pthread_cond_timedwait(&c, m, &ts) != ETIMEDOUT;
    }
    // Hasta el plazo absoluto end (kForever = sin plazo); false si ya venció.
    bool wait_until(pthread_mutex_t* m, int64_t end) {
        if (end == kForever) return wait(m), true;
        const int64_t left = end - now_ns();
        return left > 0 && wait(m, left);
    }
};

// La variable de condición que ve el juego. En Horizon (svcWaitProcessWideKeyAtomic) una espera solo vuelve por un
// Signal o Broadcast hecho después de empezarla, o por plazo; pthread_cond_wait puede volver sin nada. MH Rise espera
// una sola vez sin volver a mirar la condición en el relevo de frames entre GameThread y el render (main+0x792dbe0 y
// main+0x7924064, contadores comparados con !=): un despertar de más adelantaba un frame a uno de los dos y acababan
// esperándose el uno al otro (se colgaba al salir el aviso de crear partida). Aquí cada espera se apunta y lee el
// número de secuencia antes de soltar el mutex del juego; Signal da un permiso a quien esté apuntado (Broadcast, uno
// a cada uno) y cambia la secuencia, así que ni se pierde un aviso ni se vuelve sin él.
struct GameCond {
    std::mutex state;
    uint32_t waiters = 0, permits = 0;
    std::atomic<uint32_t> seq{0};

    void signal(bool all) {
        {
            std::lock_guard lock(state);
            if (permits >= waiters) return;
            permits = all ? waiters : permits + 1;
            seq.fetch_add(1, std::memory_order_release);
        }
        syscall(SYS_futex, &seq, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
    }
    // m es el mutex del juego, tomado. false si venció el plazo (end: kForever o absoluto en ns monotónicos).
    bool wait(pthread_mutex_t* m, int64_t end) {
        uint32_t s;
        {
            std::lock_guard lock(state);
            waiters++;
            s = seq.load(std::memory_order_acquire);
        }
        pthread_mutex_unlock(m);
        bool got = false;
        for (;;) {
            {
                std::lock_guard lock(state);
                if (permits) {
                    permits--, waiters--, got = true;
                    break;
                }
            }
            timespec rel{}, *timeout = nullptr;
            if (end != kForever) {
                const int64_t left = end - now_ns();
                if (left <= 0) {
                    std::lock_guard lock(state);
                    waiters--;
                    if (permits > waiters) permits = waiters;  // el permiso que le tocaba no se lo queda otro
                    break;
                }
                rel = {static_cast<time_t>(left / 1'000'000'000), static_cast<long>(left % 1'000'000'000)};
                timeout = &rel;
            }
            syscall(SYS_futex, &seq, FUTEX_WAIT_PRIVATE, s, timeout, nullptr, 0);
            s = seq.load(std::memory_order_acquire);
        }
        pthread_mutex_lock(m);
        return got;
    }
};

// Cualquier evento señalado despierta a quien espere en WaitAny. Con el reloj monotónico, como los plazos de
// deadline_after: con PTHREAD_COND_INITIALIZER (tiempo real) el plazo quedaba en el pasado, cada espera
// vencía al momento y el hilo del vídeo gastaba un núcleo entero.
HostCond g_any_cond;
pthread_mutex_t g_any_pm = PTHREAD_MUTEX_INITIALIZER;
std::atomic<uint64_t> g_any_gen{0};

void notify_any() {
    pthread_mutex_lock(&g_any_pm);
    g_any_gen++;
    pthread_cond_broadcast(&g_any_cond.c);
    pthread_mutex_unlock(&g_any_pm);
}

struct HostEvent {
    HostMutex lock{false};
    HostCond cond;
    bool signaled = false;
    bool auto_clear = false;

    void signal() {
        pthread_mutex_lock(&lock.m);
        signaled = true;
        auto_clear ? pthread_cond_signal(&cond.c) : pthread_cond_broadcast(&cond.c);
        pthread_mutex_unlock(&lock.m);
        notify_any();
    }
    bool wait(int64_t timeout_ns = -1) {
        int64_t end = timeout_ns < 0 ? -1 : now_ns() + timeout_ns;
        pthread_mutex_lock(&lock.m);
        while (!signaled) {
            if (end < 0) {
                cond.wait(&lock.m);
                continue;
            }
            // Ojo: con plazo vencido el resto sale negativo, y para HostCond negativo es "sin plazo".
            int64_t left = end - now_ns();
            if (left <= 0 || !cond.wait(&lock.m, left)) break;
        }
        bool got = signaled;
        if (got && auto_clear) signaled = false;
        pthread_mutex_unlock(&lock.m);
        return got;
    }
    bool try_take() { return wait(0); }
    // Para MultiWait: mira sin consumir.
    bool is_signaled() {
        pthread_mutex_lock(&lock.m);
        bool s = signaled;
        pthread_mutex_unlock(&lock.m);
        return s;
    }
    void clear() {
        pthread_mutex_lock(&lock.m);
        signaled = false;
        pthread_mutex_unlock(&lock.m);
    }
};

// TimerEvent como el del SDK: no hay hilo que lo dispare, se recalcula al mirarlo (Wait, MultiWait) con el
// mismo criterio que UpdateSignalStateAndRecalcNextTimeToWakeupUnsafe: dispara si next < ahora; el periódico
// salta los disparos que se haya perdido sin acumularlos.
struct HostTimer {
    HostMutex lock{false};
    HostCond cond;
    bool signaled = false, auto_clear = false;
    enum State : uint8_t { Stop, OneShot, Periodic } state = Stop;
    int64_t next = 0, first = 0, interval = 0;  // ns de CLOCK_MONOTONIC (first y next absolutos)

    // Con lock.
    bool update(int64_t now) {
        if (state == Stop || next >= now) return false;
        signaled = true;
        if (state == OneShot) {
            state = Stop;
            next = 0;
            return true;
        }
        next = next > kForever - interval ? kForever : next + interval;
        if (next < now) {
            const int64_t k = (now - first) / interval + 1;
            next = k > (kForever - first) / interval ? kForever : first + k * interval;
        }
        return true;
    }
    int64_t wake_time() const { return state == Stop ? kForever : next; }
};

struct HostSemaphore {
    HostMutex lock{false};
    HostCond cond;
    int count = 0, max = 1;
};

// Lector-escritor con preferencia al escritor, como el del SDK: el dueño de la escritura puede volver a
// entrar y además leer; los lectores nuevos esperan si hay un escritor esperando.
struct HostRwLock {
    HostMutex lock{false};
    HostCond readers_cv, writers_cv;
    int readers = 0;          // lecturas en curso (también las del dueño de la escritura)
    int writes = 0;           // anidamiento de la escritura del dueño
    pid_t owner = 0;          // tid del que tiene la escritura
    int waiting_writers = 0;

    bool can_read(pid_t me) const { return owner == me || (writes == 0 && owner == 0 && waiting_writers == 0); }
    bool can_write(pid_t me) const { return owner == me || (owner == 0 && readers == 0); }
    void take_write(pid_t me) {
        owner = me;
        writes++;
    }
    // Con lock: nadie escribe ni lee; primero un escritor, si no todos los lectores.
    void wake_next() {
        if (waiting_writers > 0)
            pthread_cond_signal(&writers_cv.c);
        else
            pthread_cond_broadcast(&readers_cv.c);
    }
};

// Cola de mensajes sobre el buffer que da el juego (circular, como en el SDK).
struct HostMessageQueue {
    HostMutex lock{false};
    HostCond not_full, not_empty;
    uintptr_t* buffer = nullptr;
    size_t capacity = 0, count = 0, head = 0;

    // Con lock; false si no cabe.
    bool push(uintptr_t data, bool front) {
        if (count >= capacity) return false;
        if (front) {
            head = (head + capacity - 1) % capacity;
            buffer[head] = data;
        } else {
            buffer[(head + count) % capacity] = data;
        }
        count++;
        pthread_cond_signal(&not_empty.c);
        return true;
    }
    bool pop(uintptr_t* out) {
        if (count == 0) return false;
        if (out) *out = buffer[head];
        head = (head + 1) % capacity;
        count--;
        pthread_cond_signal(&not_full.c);
        return true;
    }
};

struct HostThread {
    ThreadType* type = nullptr;
    ThreadFunction fn = nullptr;
    void* arg = nullptr;
    size_t stack_size = 0;
    int priority = 16, ideal_core = 0;
    uint64_t core_mask = 0;  // 0: solo el núcleo ideal
    const char* name = "";
    char name_buffer[32] = {};  // SetThreadName copia el nombre (el juego puede pasar uno temporal)
    pthread_t pthread{};
    pid_t tid = 0;
    bool started = false, joined = false;
};

struct HostMultiWait {
    std::mutex m;
    std::vector<MultiWaitHolderType*> holders;
};

struct HostHolder {
    enum Kind : uint8_t { None, Event, SystemEvent, Thread, Timer } kind = None;
    const void* object = nullptr;
    MultiWaitType* owner = nullptr;
    uintptr_t user_data = 0;
};

Registry<HostMutex> g_mutexes;
Registry<GameCond> g_conds;
Registry<HostEvent> g_events;
Registry<HostTimer> g_timers;
Registry<HostSemaphore> g_semaphores;
Registry<HostRwLock> g_rwlocks;
Registry<HostMessageQueue> g_queues;
Registry<HostThread> g_threads;
Registry<HostMultiWait> g_multiwaits;
Registry<HostHolder> g_holders;

std::mutex g_regions_m;
std::unordered_map<uintptr_t, size_t> g_regions;

thread_local ThreadType* t_current = nullptr;

// Byte de estado dentro del ThreadType del juego. Hay código que lo lee en línea (cabeceras del
// SDK) en vez de preguntar a nn::os: el device de E/S de Wwise guarda su ThreadType en +0x10 y,
// tras StartThread, da el hilo por válido si el byte +0x50 del device (ThreadType+0x40) no es 0.
// En el SDK 11.4 de MH Rise sigue en +0x40 (MultiWaitHolderOfThread::IsSignaled compara ese byte con 4).
constexpr size_t kThreadStateOffset = 0x40;
enum ThreadState : uint8_t { NotInitialized = 0, Initialized = 1, Started = 3, Terminated = 4 };

void set_state(ThreadType* type, ThreadState state) {
    reinterpret_cast<volatile uint8_t*>(type)[kThreadStateOffset] = state;
}
// Lo que el SDK deja en el ThreadType al crear el hilo y el juego lee en línea: el handle del hilo en el kernel
// (+0x1b0, u32) y su id (+0x1b8, lo que devuelve GetThreadId). MH Rise guarda el handle de un hilo concreto y con
// él decide si un lector de archivo avanza su posición (main+0x6c3f0fc): con todos a 0 ningún hilo la avanzaba y
// el lector de los .pak se quedaba saltando hacia atrás para siempre. Cada hilo lleva uno distinto y no nulo.
constexpr size_t kThreadHandleOffset = 0x1b0, kThreadIdOffset = 0x1b8;
std::atomic<uint32_t> g_next_handle{0x8001};
void set_handle(ThreadType* type) {
    reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint8_t*>(type) + kThreadHandleOffset)[0] = g_next_handle++;
}
void set_thread_id(ThreadType* type, uint64_t id) {
    reinterpret_cast<volatile uint64_t*>(reinterpret_cast<uint8_t*>(type) + kThreadIdOffset)[0] = id;
}
ThreadState get_state(const ThreadType* type) {
    return static_cast<ThreadState>(reinterpret_cast<const volatile uint8_t*>(type)[kThreadStateOffset]);
}

// Valores especiales del núcleo ideal (CreateThread, SetThreadCoreMask); -1 (DontCare) se queda: cualquiera de
// la máscara.
constexpr int kIdealCoreUseDefault = -2, kIdealCoreNoUpdate = -3;
constexpr int kDefaultCore = 0;  // el núcleo por defecto de una aplicación (el del hilo principal)

// --- TLS de nn::os (TlsSlot) ---
// No es el thread_local ELF del juego (eso lo lleva el loader): son 32 valores por hilo, 16 para la aplicación
// (AllocateTlsSlot, del 0 hacia arriba) y 16 para el SDK (SdkAllocateTlsSlot, del 31 hacia abajo), como el
// TlsManager del SDK. Un slot nuevo vale 0 en todos los hilos. Al terminar un hilo se llama al destructor de
// cada slot ocupado (aunque el valor sea 0, como el SDK) y hasta cuatro pasadas más con los valores que esos
// destructores vuelvan a poner.
constexpr int kTlsSlotCount = 32, kAppTlsSlotCount = 16;

// La tabla de cada hilo se crea en el primer SetTlsValue (hasta entonces todo vale 0) y vive en el heap: la
// destruye su clave de pthread al salir el hilo, sin depender del orden en que se libera el TLS ELF.
struct TlsTable {
    uintptr_t values[kTlsSlotCount] = {};
    bool finished = false;  // ya corrieron los destructores
};
thread_local TlsTable* t_tls = nullptr;

std::mutex g_tls_m;
TlsDestructor g_tls_destructors[kTlsSlotCount];  // != nullptr: slot ocupado
bool g_tls_is_app[kTlsSlotCount];                // lo dio AllocateTlsSlot (cuenta para el límite de 16)
int g_tls_app_count = 0;
std::vector<TlsTable*> g_tls_threads;            // las tablas vivas: un slot nuevo se pone a 0 en todas
pthread_key_t g_tls_key;
pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;

void tls_noop_destructor(uintptr_t) {}

void invoke_tls_destructors(TlsTable& t) {
    if (t.finished) return;
    t.finished = true;
    TlsDestructor dtors[kTlsSlotCount];
    {
        std::lock_guard lock(g_tls_m);
        std::copy(std::begin(g_tls_destructors), std::end(g_tls_destructors), dtors);
    }
    for (int i = 0; i < kTlsSlotCount; i++) {
        if (dtors[i]) dtors[i](t.values[i]);
        t.values[i] = 0;
    }
    for (int pass = 0; pass < 4; pass++) {
        bool any = false;
        for (int i = 0; i < kTlsSlotCount; i++) {
            if (!t.values[i]) continue;
            any = true;
            if (dtors[i]) dtors[i](t.values[i]);
            t.values[i] = 0;
        }
        if (!any) break;
    }
}

// Al acabar la función de un hilo de nn::os (thread_entry).
void finish_thread_tls() {
    TlsTable zero;  // nunca puso nada: los destructores se llaman igual, con 0
    invoke_tls_destructors(t_tls ? *t_tls : zero);
}

// Al salir el hilo (también los que no creó nn::os, como el principal).
void tls_thread_exit(void* p) {
    auto* t = static_cast<TlsTable*>(p);
    invoke_tls_destructors(*t);
    {
        std::lock_guard lock(g_tls_m);
        std::erase(g_tls_threads, t);
    }
    // Si otro destructor de pthread vuelve a poner un valor, se crea otra tabla (bionic libera el TLS después).
    if (t_tls == t) t_tls = nullptr;
    delete t;
}

TlsTable& tls_table() {
    if (!t_tls) {
        pthread_once(&g_tls_once, [] { pthread_key_create(&g_tls_key, tls_thread_exit); });
        t_tls = new TlsTable;
        {
            std::lock_guard lock(g_tls_m);
            g_tls_threads.push_back(t_tls);
        }
        pthread_setspecific(g_tls_key, t_tls);
    }
    return *t_tls;
}

Result allocate_tls_slot(TlsSlot* slot, TlsDestructor destructor, bool sdk) {
    std::lock_guard lock(g_tls_m);
    if (!sdk && g_tls_app_count >= kAppTlsSlotCount) return Result{0x1203};  // ResultOutOfResource
    int found = -1;
    if (sdk) {
        for (int i = kTlsSlotCount - 1; i >= 0 && found < 0; i--)
            if (!g_tls_destructors[i]) found = i;
    } else {
        for (int i = 0; i < kTlsSlotCount && found < 0; i++)
            if (!g_tls_destructors[i]) found = i;
    }
    if (found < 0) return Result{0x1203};
    g_tls_destructors[found] = destructor ? destructor : tls_noop_destructor;
    g_tls_is_app[found] = !sdk;
    if (!sdk) g_tls_app_count++;
    for (TlsTable* t : g_tls_threads) t->values[found] = 0;
    slot->inner = static_cast<uint32_t>(found);
    return ResultSuccess;
}

// --- Afinidad de los hilos críticos ---
// El frame es una cadena: el hilo principal del juego (envía y presenta), MainDraw (graba los comandos) y
// NvnSubmit (los reproduce en Vulkan). Sin afinidad, el planificador los podía dejar en los núcleos pequeños:
// se limitan a los grandes (los de frecuencia máxima por encima de la mínima; en el SD865, cpu4-7) con una
// máscara y no un núcleo fijo, porque core_ctl puede aislar alguno. Los demás hilos no se tocan, salvo que un
// hilo nuevo hereda la máscara de quien lo crea: los de nn::os que no son críticos vuelven a la completa al
// arrancar, y los demás en la revisión de cada segundo. Android rehace la afinidad al mover el proceso de
// cpuset (segundo plano y vuelta): refresh_critical_threads la vuelve a poner. Interruptor sin_afinidad:
// máscara completa para los hilos críticos; lo mira refresh_critical_threads como mucho una vez por segundo,
// desde el hilo que presenta, y lo aplica a todos (sched_setaffinity admite el tid de otro hilo del proceso).
struct CpuMasks {
    cpu_set_t big, all;
    bool usable = false;  // hay núcleos de al menos dos frecuencias máximas
};

uint64_t mask_bits(const cpu_set_t& set) {  // los 64 primeros, para el log
    uint64_t bits = 0;
    for (int cpu = 0; cpu < 64; cpu++)
        if (CPU_ISSET(cpu, &set)) bits |= uint64_t{1} << cpu;
    return bits;
}

const CpuMasks& cpu_masks() {
    static const CpuMasks masks = [] {
        CpuMasks m;
        CPU_ZERO(&m.big);
        CPU_ZERO(&m.all);
        const long n = std::min<long>(sysconf(_SC_NPROCESSORS_CONF), 64);
        std::vector<unsigned long> freq(n > 0 ? n : 0, 0);
        unsigned long lowest = ~0ul;
        for (int cpu = 0; cpu < n; cpu++) {
            CPU_SET(cpu, &m.all);
            char path[80];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%lu", &freq[cpu]) != 1) freq[cpu] = 0;
                fclose(f);
            }
            if (freq[cpu] && freq[cpu] < lowest) lowest = freq[cpu];
        }
        for (int cpu = 0; cpu < n; cpu++)
            if (freq[cpu] && freq[cpu] > lowest) CPU_SET(cpu, &m.big);
        m.usable = CPU_COUNT(&m.big) > 0;
        printf("[nnsdk] afinidad: núcleos grandes %#llx de %#llx%s\n", static_cast<unsigned long long>(mask_bits(m.big)),
               static_cast<unsigned long long>(mask_bits(m.all)), m.usable ? "" : " (todos iguales o sin datos: no se usa)");
        return m;
    }();
    return masks;
}

bool affinity_off() { return NVN_FLAG("sin_afinidad"); }

struct CriticalThread {
    pid_t tid;
    char name[16];
};
std::mutex g_critical_m;
std::vector<CriticalThread> g_critical;
std::atomic<bool> g_any_critical{false};
thread_local bool t_critical = false;

// Pone la máscara si no la tiene ya; false si el hilo ya no existe. Si el sistema no deja, se queda como está.
bool apply_mask(const CriticalThread& t, const cpu_set_t& want, bool refreshing) {
    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(t.tid, sizeof now, &now) == 0 && CPU_EQUAL(&now, &want)) return true;
    if (sched_setaffinity(t.tid, sizeof want, &want) != 0) {
        if (errno == ESRCH) return false;
        static std::atomic<bool> failed_logged{false};
        if (!failed_logged.exchange(true))
            printf("[nnsdk] afinidad: no se pudo poner la de '%s' (%s); se deja como está\n", t.name, strerror(errno));
        return true;
    }
    static std::atomic<int> lost_logged{0};  // DEPURACIÓN: ¿quién nos la quita?
    if (refreshing && lost_logged.fetch_add(1) < 5)
        printf("[nnsdk] afinidad de '%s' perdida (era %#llx): se vuelve a poner\n", t.name,
               static_cast<unsigned long long>(mask_bits(now)));
    return true;
}

void pin_current_thread() {
    const CpuMasks& m = cpu_masks();
    if (!m.usable || t_critical) return;
    t_critical = true;
    CriticalThread t{gettid(), {}};
    pthread_getname_np(pthread_self(), t.name, sizeof t.name);
    const bool off = affinity_off();
    std::lock_guard lock(g_critical_m);
    apply_mask(t, off ? m.all : m.big, false);
    g_critical.push_back(t);
    g_any_critical = true;
    printf("[nnsdk] hilo crítico '%s' (tid %d): %s\n", t.name, t.tid, off ? "sin afinidad (sin_afinidad)" : "núcleos grandes");
}

void unpin_current_thread() {  // el hilo termina: su tid se puede reutilizar
    if (!t_critical) return;
    t_critical = false;
    const pid_t tid = gettid();
    std::lock_guard lock(g_critical_m);
    std::erase_if(g_critical, [&](const CriticalThread& t) { return t.tid == tid; });
}

// Un hilo nuevo hereda la máscara de quien lo crea: si fue un hilo crítico, se le devuelve la completa.
void release_inherited_affinity() {
    if (!g_any_critical) return;  // nadie tiene máscara: nada que heredar
    const CpuMasks& m = cpu_masks();
    if (!m.usable) return;
    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(0, sizeof now, &now) == 0 && !CPU_EQUAL(&now, &m.all)) sched_setaffinity(0, sizeof m.all, &m.all);
}

// Lo mismo para los hilos que no son de nn::os (del driver de Vulkan, de AAudio, el que guarda la caché de
// pipelines) si los crea un hilo crítico: una vez por segundo, los que no son críticos y tienen justo la máscara
// de los grandes (la heredada) vuelven a la completa. Con g_critical_m.
void release_inherited_masks(const CpuMasks& m) {
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return;
    while (const dirent* e = readdir(dir)) {
        const pid_t tid = static_cast<pid_t>(atoi(e->d_name));
        if (tid <= 0 || std::any_of(g_critical.begin(), g_critical.end(), [&](const CriticalThread& t) { return t.tid == tid; }))
            continue;
        cpu_set_t now;
        CPU_ZERO(&now);
        if (sched_getaffinity(tid, sizeof now, &now) == 0 && CPU_EQUAL(&now, &m.big))
            sched_setaffinity(tid, sizeof m.all, &m.all);
    }
    closedir(dir);
}

bool critical_name(const char* name) { return strcmp(name, "MainDraw") == 0; }

void set_pthread_name(pthread_t thread, const char* name) {
    char buf[16];  // Linux admite 15 caracteres: con más, el nombre no se pone (ManaSwitchDecoderWatchThread)
    snprintf(buf, sizeof buf, "%s", name && name[0] ? name : "nn::os");
    pthread_setname_np(thread, buf);
}

void* thread_entry(void* p) {
    auto* t = static_cast<HostThread*>(p);
    t_current = t->type;
    t->tid = gettid();
    set_thread_id(t->type, t->tid);
    set_pthread_name(pthread_self(), t->name);
    if (critical_name(t->name))
        pin_current_thread();
    else
        release_inherited_affinity();
    nnsdk::on_thread_start();
    t->fn(t->arg);
    // Como el SDK al acabar la función del hilo: destructores de los TlsSlot (con el TLS del juego aún vivo),
    // estado Terminated (lo miran el código en línea y MultiWaitHolderOfThread) y aviso a los MultiWait.
    finish_thread_tls();
    unpin_current_thread();
    set_state(t->type, Terminated);
    notify_any();
    return nullptr;
}

// Hilos que no creó nn::os (el principal): se les da un ThreadType propio.
ThreadType* current_thread() {
    if (!t_current) {
        auto* fake = static_cast<ThreadType*>(calloc(1, 0x400));
        HostThread& t = g_threads.reset(fake);
        t.type = fake;
        t.pthread = pthread_self();
        t.tid = gettid();
        t.started = true;
        t.name = "MainThread";
        set_handle(fake);
        set_thread_id(fake, t.tid);
        set_state(fake, Started);
        t_current = fake;
    }
    return t_current;
}

Result create_thread(ThreadType* type, ThreadFunction fn, void* arg, size_t stack_size, int priority, int core) {
    HostThread& t = g_threads.reset(type);
    t.type = type;
    t.fn = fn;
    t.arg = arg;
    t.stack_size = stack_size;
    t.priority = priority;
    t.ideal_core = core == kIdealCoreUseDefault || core == kIdealCoreNoUpdate ? kDefaultCore : core;
    set_handle(type);
    set_state(type, Initialized);
    return ResultSuccess;
}

// El núcleo en el que "está" el hilo. En la Switch un hilo solo corre en los núcleos de su máscara (por defecto,
// solo su núcleo ideal), y el juego usa el número como bit de una máscara (1 << núcleo, guardado en un TlsSlot
// por cada worker): se contesta dentro de la máscara del hilo y de los núcleos 0-2 de la aplicación, nunca el
// núcleo real de la Flip (0-7).
int current_core(const HostThread& t) {
    if (t.ideal_core >= 0) return t.ideal_core;
    const uint64_t mask = (t.core_mask ? t.core_mask : 0b111) & 0b111;
    const int count = __builtin_popcountll(mask);
    if (count == 0) return kDefaultCore;
    int k = sched_getcpu();
    k = (k < 0 ? 0 : k) % count;
    for (int core = 0; core < 3; core++)
        if ((mask >> core & 1) && k-- == 0) return core;
    return kDefaultCore;
}

// --- MultiWait ---
// Como el SDK, WaitAny solo mira si el objeto está señalado y no lo consume (tampoco un Event AutoClear): quien
// recibe el holder llama luego a TryWait/Wait/Clear. Devuelve el primero señalado en orden de enlace.
bool holder_signaled(const HostHolder& h, int64_t now, int64_t* wake) {
    switch (h.kind) {
        case HostHolder::Event:
        case HostHolder::SystemEvent:
            return g_events.get(h.object).is_signaled();
        case HostHolder::Thread:
            return get_state(static_cast<const ThreadType*>(h.object)) == Terminated;
        case HostHolder::Timer: {
            HostTimer& t = g_timers.get(h.object);
            pthread_mutex_lock(&t.lock.m);
            t.update(now);
            const bool s = t.signaled;
            *wake = std::min(*wake, t.wake_time());
            pthread_mutex_unlock(&t.lock.m);
            return s;
        }
        case HostHolder::None:
            break;
    }
    return false;
}

// end: plazo absoluto (kForever = sin plazo; ya vencido = solo mirar).
MultiWaitHolderType* wait_any(MultiWaitType* w, int64_t end) {
    HostMultiWait& mw = g_multiwaits.get(w);
    for (;;) {
        const uint64_t gen = g_any_gen.load();
        const int64_t now = now_ns();
        int64_t wake = end;  // el próximo disparo de un temporizador también despierta
        {
            std::lock_guard lock(mw.m);
            for (MultiWaitHolderType* h : mw.holders)
                if (holder_signaled(g_holders.get(h), now, &wake)) return h;
        }
        if (now >= end) return nullptr;
        pthread_mutex_lock(&g_any_pm);
        if (g_any_gen.load() == gen) {
            // Como red, nunca más de 100 ms seguidos: todo lo que señala pasa por notify_any, pero no cuesta nada.
            timespec ts = deadline_after(std::min<int64_t>(wake - now, 100'000'000));
            pthread_cond_timedwait(&g_any_cond.c, &g_any_pm, &ts);
        }
        pthread_mutex_unlock(&g_any_pm);
    }
}

void init_holder(MultiWaitHolderType* h, HostHolder::Kind kind, const void* object) {
    HostHolder& holder = g_holders.reset(h);
    holder.kind = kind;
    holder.object = object;
}

// --- Semáforos ---

bool acquire_semaphore(SemaphoreType* s, int64_t end) {
    HostSemaphore& sem = g_semaphores.get(s);
    pthread_mutex_lock(&sem.lock.m);
    while (sem.count == 0)
        if (!sem.cond.wait_until(&sem.lock.m, end)) break;
    const bool got = sem.count > 0;
    if (got) sem.count--;
    pthread_mutex_unlock(&sem.lock.m);
    return got;
}

// --- Colas de mensajes ---

bool send_message(MessageQueueType* q, uintptr_t data, bool front, bool block) {
    HostMessageQueue& mq = g_queues.get(q);
    pthread_mutex_lock(&mq.lock.m);
    bool sent;
    while (!(sent = mq.push(data, front)) && block && mq.capacity) mq.not_full.wait(&mq.lock.m);
    pthread_mutex_unlock(&mq.lock.m);
    return sent;
}

bool receive_message(uintptr_t* out, MessageQueueType* q, bool block) {
    HostMessageQueue& mq = g_queues.get(q);
    pthread_mutex_lock(&mq.lock.m);
    bool got;
    while (!(got = mq.pop(out)) && block) mq.not_empty.wait(&mq.lock.m);
    pthread_mutex_unlock(&mq.lock.m);
    return got;
}

// Lo que da una consola de venta: el g_CommandLineParameter del sdk vale argc 1, argv {"", nullptr} y nadie lo
// cambia (SetHostArgc/SetHostArgv no tienen llamadas). nnMain pasa argv[0] a UTF-16 y se lo da al parser de
// argumentos con argc: con argc 0 le daba un array sin rellenar y se caía en main+0x6bddcdc.
char g_argv0[] = "";
char* g_argv[] = {g_argv0, nullptr};

}  // namespace

namespace nnsdk {

int64_t wait_clock_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec;
}

// DEPURACIÓN (con <datos>/registro): las esperas de más de 150 ms, con el objeto y quién la llamó (dirección
// absoluta: restarle la base de main que sale al principio del log). Para ver qué espera se come un frame.
// Objetos con esperas largas recientes: quién los señala lo registra log_signal.
std::atomic<const void*> g_hot_objects[8];
std::atomic<uint32_t> g_hot_next{0};

void log_signal(const char* what, const void* object, const void* caller) {
    for (auto& hot : g_hot_objects) {
        if (hot.load(std::memory_order_relaxed) != object) continue;
        static std::atomic<int> logged{0};
        if (logged++ > 20000) return;
        char name[32] = "?";
        pthread_getname_np(pthread_self(), name, sizeof name);
        printf("[os] %s %p desde '%s' (llamada desde %p) a %.1f ms\n", what, object, name, caller, wait_clock_ns() / 1e6);
        return;
    }
}

void log_slow_wait(const char* what, const void* object, int64_t start_ns, const void* caller) {
    const int64_t end_ns = wait_clock_ns(), ms = (end_ns - start_ns) / 1'000'000;
    if (ms < 150 || !NVN_FLAG("registro")) return;
    // Solo los objetos en los que espera GameThread (el que despierta primero en cada ciclo lento de MH Rise).
    char waiter[32] = "?";
    pthread_getname_np(pthread_self(), waiter, sizeof waiter);
    bool known = false;
    for (auto& hot : g_hot_objects) known |= hot.load(std::memory_order_relaxed) == object;
    if (!known && strcmp(waiter, "GameThread") == 0) g_hot_objects[g_hot_next++ % 8].store(object, std::memory_order_relaxed);
    static std::atomic<int> logged{0};
    if (logged++ > 400) return;
    char name[32] = "?";
    pthread_getname_np(pthread_self(), name, sizeof name);
    printf("[os] %s %p: %lld ms en '%s' (llamada desde %p, despierta a %.1f ms)\n", what, object,
           static_cast<long long>(ms), name, caller, end_ns / 1e6);
}

}  // namespace nnsdk

namespace nn::os {

// --- Mutex, variables de condición, RW lock ---

void InitializeMutex(MutexType* m, bool recursive, int) { g_mutexes.reset(m, recursive); }
void FinalizeMutex(MutexType* m) { g_mutexes.erase(m); }
// El dueño se apunta al tomarlo y se borra al soltar la última vez. Durante la espera de una variable de condición
// sigue apuntado: ese hilo no puede preguntar mientras espera y para los demás no cambia nada.
static void note_locked(HostMutex& h) {
    if (h.depth++ == 0) __atomic_store_n(&h.owner, gettid(), __ATOMIC_RELAXED);
}
void LockMutex(MutexType* m) {
    HostMutex& h = g_mutexes.get(m);
    pthread_mutex_lock(&h.m);
    note_locked(h);
}
bool TryLockMutex(MutexType* m) {
    HostMutex& h = g_mutexes.get(m);
    if (pthread_mutex_trylock(&h.m) != 0) return false;
    note_locked(h);
    return true;
}
void UnlockMutex(MutexType* m) {
    HostMutex& h = g_mutexes.get(m);
    if (h.depth > 0 && --h.depth == 0) __atomic_store_n(&h.owner, 0, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&h.m);
}
bool IsMutexLockedByCurrentThread(const MutexType* m) {
    return __atomic_load_n(&g_mutexes.get(m).owner, __ATOMIC_RELAXED) == gettid();
}

void InitializeConditionVariable(ConditionVariableType* c) { g_conds.reset(c); }
void FinalizeConditionVariable(ConditionVariableType* c) { g_conds.erase(c); }
void SignalConditionVariable(ConditionVariableType* c) {
    nnsdk::log_signal("SignalConditionVariable", c, __builtin_return_address(0));
    g_conds.get(c).signal(false);
}
void BroadcastConditionVariable(ConditionVariableType* c) {
    nnsdk::log_signal("BroadcastConditionVariable", c, __builtin_return_address(0));
    g_conds.get(c).signal(true);
}
void WaitConditionVariable(ConditionVariableType* c, MutexType* m) {
    const int64_t start = nnsdk::wait_clock_ns();
    g_conds.get(c).wait(&g_mutexes.get(m).m, kForever);
    nnsdk::log_slow_wait("WaitConditionVariable", c, start, __builtin_return_address(0));
}
ConditionVariableStatus TimedWaitConditionVariable(ConditionVariableType* c, MutexType* m, TimeSpan t) {
    const int64_t start = nnsdk::wait_clock_ns();
    const bool ok = g_conds.get(c).wait(&g_mutexes.get(m).m, deadline_of(t));
    if (!ok) nnsdk::log_slow_wait("TimedWaitConditionVariable caducado", c, start, __builtin_return_address(0));
    return ok ? ConditionVariableStatus_NoTimeout : ConditionVariableStatus_Timeout;
}

void InitializeReaderWriterLock(ReaderWriterLockType* l) { g_rwlocks.reset(l); }
void FinalizeReaderWriterLock(ReaderWriterLockType* l) { g_rwlocks.erase(l); }

void AcquireReadLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    const pid_t me = gettid();
    pthread_mutex_lock(&rw.lock.m);
    while (!rw.can_read(me)) rw.readers_cv.wait(&rw.lock.m);
    rw.readers++;
    pthread_mutex_unlock(&rw.lock.m);
}
bool TryAcquireReadLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    pthread_mutex_lock(&rw.lock.m);
    const bool got = rw.can_read(gettid());
    if (got) rw.readers++;
    pthread_mutex_unlock(&rw.lock.m);
    return got;
}
void ReleaseReadLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    pthread_mutex_lock(&rw.lock.m);
    if (rw.readers > 0) rw.readers--;
    if (rw.readers == 0) {
        // El dueño que ya soltó la escritura pero seguía leyendo la suelta del todo.
        if (rw.owner && rw.writes == 0) rw.owner = 0;
        if (!rw.owner) rw.wake_next();
    }
    pthread_mutex_unlock(&rw.lock.m);
}

void AcquireWriteLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    const pid_t me = gettid();
    pthread_mutex_lock(&rw.lock.m);
    if (!rw.can_write(me)) {
        rw.waiting_writers++;
        while (!rw.can_write(me)) rw.writers_cv.wait(&rw.lock.m);
        rw.waiting_writers--;
    }
    rw.take_write(me);
    pthread_mutex_unlock(&rw.lock.m);
}
bool TryAcquireWriteLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    const pid_t me = gettid();
    pthread_mutex_lock(&rw.lock.m);
    const bool got = rw.can_write(me);
    if (got) rw.take_write(me);
    pthread_mutex_unlock(&rw.lock.m);
    return got;
}
void ReleaseWriteLock(ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    pthread_mutex_lock(&rw.lock.m);
    if (rw.writes > 0) rw.writes--;
    // Si el dueño aún lee, sigue siéndolo hasta su último ReleaseReadLock.
    if (rw.writes == 0 && rw.readers == 0) {
        rw.owner = 0;
        rw.wake_next();
    }
    pthread_mutex_unlock(&rw.lock.m);
}

// Lo tiene alguien para leer (cualquier hilo, como en el SDK).
bool IsReadLockHeld(const ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    pthread_mutex_lock(&rw.lock.m);
    const bool held = rw.readers > 0;
    pthread_mutex_unlock(&rw.lock.m);
    return held;
}
bool IsWriteLockHeldByCurrentThread(const ReaderWriterLockType* l) {
    HostRwLock& rw = g_rwlocks.get(l);
    pthread_mutex_lock(&rw.lock.m);
    const bool held = rw.owner == gettid() && rw.writes > 0;
    pthread_mutex_unlock(&rw.lock.m);
    return held;
}

// La caché de datos de la CPU ya es coherente con la GPU y los dispositivos de Android, pero lo que la GPU escribió vive
// en overlays (executor.cpp) que no ven una reescritura de la CPU con los mismos bytes: el vaciado dice que manda la CPU.
void FlushDataCache(const void* address, size_t size) { nvn::note_cpu_flush(reinterpret_cast<uintptr_t>(address), size); }

// --- Eventos, eventos de sistema, semáforos ---

void InitializeEvent(EventType* e, bool signaled, EventClearMode mode) {
    HostEvent& ev = g_events.reset(e);
    ev.signaled = signaled;
    ev.auto_clear = mode == EventClearMode_AutoClear;
}
void FinalizeEvent(EventType* e) { g_events.erase(e); }
void SignalEvent(EventType* e) { g_events.get(e).signal(); }
void WaitEvent(EventType* e) {
    const int64_t start = nnsdk::wait_clock_ns();
    g_events.get(e).wait();
    nnsdk::log_slow_wait("WaitEvent", e, start, __builtin_return_address(0));
}
bool TryWaitEvent(EventType* e) { return g_events.get(e).try_take(); }
bool TimedWaitEvent(EventType* e, TimeSpan t) {
    const int64_t start = nnsdk::wait_clock_ns();
    const bool ok = g_events.get(e).wait(t.ns);
    if (!ok) nnsdk::log_slow_wait("TimedWaitEvent caducado", e, start, __builtin_return_address(0));
    return ok;
}
void ClearEvent(EventType* e) { g_events.get(e).clear(); }

void DestroySystemEvent(SystemEventType* e) { g_events.erase(e); }
void WaitSystemEvent(SystemEventType* e) {
    const int64_t start = nnsdk::wait_clock_ns();
    g_events.get(e).wait();
    nnsdk::log_slow_wait("WaitSystemEvent", e, start, __builtin_return_address(0));
}
bool TryWaitSystemEvent(SystemEventType* e) { return g_events.get(e).try_take(); }
bool TimedWaitSystemEvent(SystemEventType* e, TimeSpan t) { return g_events.get(e).wait(t.ns); }
void ClearSystemEvent(SystemEventType* e) { g_events.get(e).clear(); }

void InitializeSemaphore(SemaphoreType* s, int initial, int max) {
    HostSemaphore& sem = g_semaphores.reset(s);
    sem.count = initial;
    sem.max = max;
}
void FinalizeSemaphore(SemaphoreType* s) { g_semaphores.erase(s); }
void AcquireSemaphore(SemaphoreType* s) {
    const int64_t start = nnsdk::wait_clock_ns();
    acquire_semaphore(s, kForever);
    nnsdk::log_slow_wait("AcquireSemaphore", s, start, __builtin_return_address(0));
}
bool TimedAcquireSemaphore(SemaphoreType* s, TimeSpan t) {
    const int64_t start = nnsdk::wait_clock_ns();
    const bool ok = acquire_semaphore(s, deadline_of(t));
    if (!ok) nnsdk::log_slow_wait("TimedAcquireSemaphore caducado", s, start, __builtin_return_address(0));
    return ok;
}
void ReleaseSemaphore(SemaphoreType* s, int count) {
    HostSemaphore& sem = g_semaphores.get(s);
    pthread_mutex_lock(&sem.lock.m);
    // El SDK aborta si se pasa del máximo; aquí se queda en el máximo.
    sem.count = std::min(sem.count + std::max(count, 0), sem.max);
    count > 1 ? pthread_cond_broadcast(&sem.cond.c) : pthread_cond_signal(&sem.cond.c);
    pthread_mutex_unlock(&sem.lock.m);
}
void ReleaseSemaphore(SemaphoreType* s) { ReleaseSemaphore(s, 1); }

// --- Temporizadores ---

void InitializeTimerEvent(TimerEventType* e, EventClearMode mode) {
    g_timers.reset(e).auto_clear = mode == EventClearMode_AutoClear;
}
void FinalizeTimerEvent(TimerEventType* e) { g_timers.erase(e); }

void StartOneShotTimerEvent(TimerEventType* e, TimeSpan first) {
    HostTimer& t = g_timers.get(e);
    pthread_mutex_lock(&t.lock.m);
    t.next = t.first = deadline_of(first);
    t.interval = 0;
    t.state = HostTimer::OneShot;
    pthread_cond_broadcast(&t.cond.c);
    pthread_mutex_unlock(&t.lock.m);
    notify_any();  // los MultiWait recalculan cuándo despertar
}
void StartPeriodicTimerEvent(TimerEventType* e, TimeSpan first, TimeSpan interval) {
    HostTimer& t = g_timers.get(e);
    pthread_mutex_lock(&t.lock.m);
    t.next = t.first = deadline_of(first);
    t.interval = std::max<int64_t>(interval.ns, 1);
    t.state = HostTimer::Periodic;
    pthread_cond_broadcast(&t.cond.c);
    pthread_mutex_unlock(&t.lock.m);
    notify_any();
}
void StopTimerEvent(TimerEventType* e) {
    HostTimer& t = g_timers.get(e);
    pthread_mutex_lock(&t.lock.m);
    t.state = HostTimer::Stop;
    t.next = 0;
    pthread_cond_broadcast(&t.cond.c);
    pthread_mutex_unlock(&t.lock.m);
    notify_any();
}
void WaitTimerEvent(TimerEventType* e) {
    HostTimer& t = g_timers.get(e);
    pthread_mutex_lock(&t.lock.m);
    for (;;) {
        t.update(now_ns());
        if (t.signaled) {
            if (t.auto_clear) t.signaled = false;
            break;
        }
        t.cond.wait_until(&t.lock.m, t.wake_time());
    }
    pthread_mutex_unlock(&t.lock.m);
}

// --- Colas de mensajes ---

void InitializeMessageQueue(MessageQueueType* q, uintptr_t* buffer, size_t count) {
    HostMessageQueue& mq = g_queues.reset(q);
    mq.buffer = buffer;
    mq.capacity = buffer ? count : 0;
}
void FinalizeMessageQueue(MessageQueueType* q) { g_queues.erase(q); }
void SendMessageQueue(MessageQueueType* q, uintptr_t data) { send_message(q, data, false, true); }
bool TrySendMessageQueue(MessageQueueType* q, uintptr_t data) { return send_message(q, data, false, false); }
void JamMessageQueue(MessageQueueType* q, uintptr_t data) { send_message(q, data, true, true); }
bool TryJamMessageQueue(MessageQueueType* q, uintptr_t data) { return send_message(q, data, true, false); }
void ReceiveMessageQueue(uintptr_t* out, MessageQueueType* q) {
    const int64_t start = nnsdk::wait_clock_ns();
    receive_message(out, q, true);
    nnsdk::log_slow_wait("ReceiveMessageQueue", q, start, __builtin_return_address(0));
}
bool TryReceiveMessageQueue(uintptr_t* out, MessageQueueType* q) { return receive_message(out, q, false); }

// --- MultiWait ---

void InitializeMultiWait(MultiWaitType* w) { g_multiwaits.reset(w); }
void FinalizeMultiWait(MultiWaitType* w) { g_multiwaits.erase(w); }
void InitializeMultiWaitHolder(MultiWaitHolderType* h, EventType* e) { init_holder(h, HostHolder::Event, e); }
void InitializeMultiWaitHolder(MultiWaitHolderType* h, SystemEventType* e) { init_holder(h, HostHolder::SystemEvent, e); }
void InitializeMultiWaitHolder(MultiWaitHolderType* h, ThreadType* t) { init_holder(h, HostHolder::Thread, t); }
void InitializeMultiWaitHolder(MultiWaitHolderType* h, TimerEventType* e) { init_holder(h, HostHolder::Timer, e); }
void FinalizeMultiWaitHolder(MultiWaitHolderType* h) { g_holders.erase(h); }
void LinkMultiWaitHolder(MultiWaitType* w, MultiWaitHolderType* h) {
    HostMultiWait& mw = g_multiwaits.get(w);
    std::lock_guard lock(mw.m);
    mw.holders.push_back(h);
    g_holders.get(h).owner = w;
}
void UnlinkMultiWaitHolder(MultiWaitHolderType* h) {
    HostHolder& holder = g_holders.get(h);
    if (!holder.owner) return;
    HostMultiWait& mw = g_multiwaits.get(holder.owner);
    std::lock_guard lock(mw.m);
    std::erase(mw.holders, h);
    holder.owner = nullptr;
}
void SetMultiWaitHolderUserData(MultiWaitHolderType* h, uintptr_t data) { g_holders.get(h).user_data = data; }
uintptr_t GetMultiWaitHolderUserData(const MultiWaitHolderType* h) { return g_holders.get(h).user_data; }
MultiWaitHolderType* WaitAny(MultiWaitType* w) {
    const int64_t start = nnsdk::wait_clock_ns();
    MultiWaitHolderType* h = wait_any(w, kForever);
    nnsdk::log_slow_wait("WaitAny", w, start, __builtin_return_address(0));
    return h;
}
MultiWaitHolderType* TryWaitAny(MultiWaitType* w) { return wait_any(w, 0); }
MultiWaitHolderType* TimedWaitAny(MultiWaitType* w, TimeSpan t) {
    const int64_t start = nnsdk::wait_clock_ns();
    MultiWaitHolderType* h = wait_any(w, deadline_of(t));
    if (!h) nnsdk::log_slow_wait("TimedWaitAny caducado", w, start, __builtin_return_address(0));
    return h;
}

// --- Hilos ---

Result CreateThread(ThreadType* t, ThreadFunction fn, void* arg, void*, size_t stack_size, int priority) {
    return create_thread(t, fn, arg, stack_size, priority, kIdealCoreUseDefault);
}
Result CreateThread(ThreadType* t, ThreadFunction fn, void* arg, void*, size_t stack_size, int priority,
                    int core) {
    return create_thread(t, fn, arg, stack_size, priority, core);
}

// La pila que da el juego no se usa: bionic reserva la suya del mismo tamaño.
void StartThread(ThreadType* type) {
    HostThread& t = g_threads.get(type);
    // Ya corre: el vídeo reintenta Configure + StartThread si su Start falla, y un segundo pthread_create
    // sobre el mismo HostThread dejaba el primero sin join (su pila perdida) y dos bucles en un decoder.
    if (t.started && !t.joined) return;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, t.stack_size < 256 * 1024 ? 256 * 1024 : t.stack_size);
    set_state(type, Started);
    t.started = pthread_create(&t.pthread, &attr, thread_entry, &t) == 0;
    pthread_attr_destroy(&attr);
    printf("[nnsdk] hilo '%s' fn=%p arg=%p pila=%zu KB núcleo %d%s\n", t.name, reinterpret_cast<void*>(t.fn), t.arg,
           t.stack_size / 1024, t.ideal_core, t.started ? "" : " NO SE PUDO CREAR");
}

void WaitThread(ThreadType* type) {
    HostThread& t = g_threads.get(type);
    if (t.started && !t.joined) {
        pthread_join(t.pthread, nullptr);
        t.joined = true;
    }
}

void DestroyThread(ThreadType* type) {
    WaitThread(type);
    g_threads.erase(type);
    set_state(type, NotInitialized);
}

ThreadType* GetCurrentThread() { return current_thread(); }
void YieldThread() { sched_yield(); }
void SleepThread(TimeSpan t) {
    // DEPURACIÓN (registro): quién duerme 20 ms o más de una vez.
    if (t.ns >= 20'000'000 && NVN_FLAG("registro")) {
        static std::atomic<int> logged{0};
        char name[32] = "?";
        pthread_getname_np(pthread_self(), name, sizeof name);
        if (nvn::first_times(logged, 300))
            printf("[os] SleepThread %lld ms en '%s' (llamada desde %p)\n", static_cast<long long>(t.ns / 1'000'000), name,
                   __builtin_return_address(0));
    }
    timespec ts{static_cast<time_t>(t.ns / 1'000'000'000), static_cast<long>(t.ns % 1'000'000'000)};
    nanosleep(&ts, nullptr);
}

int ChangeThreadPriority(ThreadType* type, int priority) {
    HostThread& t = g_threads.get(type);
    int old = t.priority;
    t.priority = priority;
    return old;
}
// Las prioridades no se pasan a Linux (sin root no se puede subir), así que la actual es la base.
int GetThreadPriority(const ThreadType* type) { return g_threads.get(type).priority; }
int GetThreadCurrentPriority(const ThreadType* type) { return g_threads.get(type).priority; }

void SetThreadCoreMask(ThreadType* type, int ideal_core, uint64_t mask) {
    HostThread& t = g_threads.get(type);
    if (ideal_core != kIdealCoreNoUpdate) t.ideal_core = ideal_core == kIdealCoreUseDefault ? kDefaultCore : ideal_core;
    t.core_mask = mask;
}
void GetThreadCoreMask(int* ideal_core, uint64_t* mask, const ThreadType* type) {
    HostThread& t = g_threads.get(type);
    if (ideal_core) *ideal_core = t.ideal_core;
    if (mask) *mask = t.core_mask ? t.core_mask : t.ideal_core >= 0 ? uint64_t{1} << t.ideal_core : 0b111;
}
uint64_t GetThreadAvailableCoreMask() { return 0b111; }  // las aplicaciones de Switch usan los núcleos 0-2
int GetCurrentCoreNumber() { return current_core(g_threads.get(current_thread())); }
uint64_t GetThreadId(const ThreadType* type) { return g_threads.get(type).tid; }

void SetThreadName(ThreadType* type, const char* name) {
    HostThread& t = g_threads.get(type);
    snprintf(t.name_buffer, sizeof t.name_buffer, "%s", name ? name : "");
    t.name = t.name_buffer;
    if (t.started && !t.joined) set_pthread_name(t.pthread, t.name);
}
void SetThreadNamePointer(ThreadType* type, const char* name) {
    HostThread& t = g_threads.get(type);
    t.name = name ? name : "";
    if (t.started && !t.joined) set_pthread_name(t.pthread, t.name);
}
const char* GetThreadNamePointer(const ThreadType* type) { return g_threads.get(type).name; }

int GetHostArgc() { return 1; }
char** GetHostArgv() { return g_argv; }

// --- TLS ---

Result AllocateTlsSlot(TlsSlot* slot, TlsDestructor destructor) { return allocate_tls_slot(slot, destructor, false); }
Result SdkAllocateTlsSlot(TlsSlot* slot, TlsDestructor destructor) { return allocate_tls_slot(slot, destructor, true); }
void FreeTlsSlot(TlsSlot slot) {
    if (slot.inner >= kTlsSlotCount) return;
    std::lock_guard lock(g_tls_m);
    if (!g_tls_destructors[slot.inner]) return;
    g_tls_destructors[slot.inner] = nullptr;
    if (g_tls_is_app[slot.inner]) g_tls_app_count--;
    g_tls_is_app[slot.inner] = false;
}
uintptr_t GetTlsValue(TlsSlot slot) {
    const TlsTable* t = t_tls;
    return t && slot.inner < kTlsSlotCount ? t->values[slot.inner] : 0;
}
void SetTlsValue(TlsSlot slot, uintptr_t value) {
    if (slot.inner < kTlsSlotCount) tls_table().values[slot.inner] = value;
}

// --- Tiempo: el contador virtual de ARM, igual que en la Switch (19.2 MHz en ambas) ---

Tick GetSystemTick() {
    uint64_t v;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return Tick{static_cast<int64_t>(v)};
}
// El SDK lee CNTPCT_EL0 entre barreras; en Android EL0 solo puede leer el virtual (mismo ritmo).
Tick GetSystemTickOrdered() {
    uint64_t v;
    asm volatile("dsb ish; isb; mrs %0, cntvct_el0; isb" : "=r"(v)::"memory");
    return Tick{static_cast<int64_t>(v)};
}
int64_t GetSystemTickFrequency() {
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return static_cast<int64_t>(f);
}
TimeSpan ConvertToTimeSpan(Tick tick) {
    __int128 ns = static_cast<__int128>(tick.value) * 1'000'000'000 / GetSystemTickFrequency();
    return TimeSpan{static_cast<int64_t>(ns)};
}
// Como TickManager::ConvertToTick: segundos enteros más la fracción redondeada hacia arriba, simétrico para
// negativos, y el mínimo de int64 tal cual.
Tick ConvertToTick(TimeSpan t) {
    if (t.ns == INT64_MIN) return Tick{INT64_MIN};
    const int64_t freq = GetSystemTickFrequency();
    const int64_t abs = t.ns < 0 ? -t.ns : t.ns;
    const int64_t sec = abs / 1'000'000'000, rem = abs % 1'000'000'000;
    const int64_t ticks = sec * freq + (rem * freq + 999'999'999) / 1'000'000'000;
    return Tick{t.ns < 0 ? -ticks : ticks};
}

// --- Memoria virtual ---

Result AllocateAddressRegion(uintptr_t* out, size_t size) {
    void* p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return Result{0x1801};
    std::lock_guard lock(g_regions_m);
    g_regions[reinterpret_cast<uintptr_t>(p)] = size;
    *out = reinterpret_cast<uintptr_t>(p);
    printf("[nnsdk] AllocateAddressRegion %p +%zu KB\n", p, size / 1024);
    return ResultSuccess;
}
Result FreeAddressRegion(uintptr_t address) {
    std::lock_guard lock(g_regions_m);
    auto it = g_regions.find(address);
    if (it == g_regions.end()) return Result{0x1801};
    munmap(reinterpret_cast<void*>(address), it->second);
    g_regions.erase(it);
    return ResultSuccess;
}
Result AllocateMemoryPages(uintptr_t address, size_t size) {
    printf("[nnsdk] AllocateMemoryPages 0x%lx +%zu KB\n", address, size / 1024);
    return mprotect(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE) == 0 ? ResultSuccess
                                                                                         : Result{0x1801};
}
Result FreeMemoryPages(uintptr_t address, size_t size) {
    printf("[nnsdk] FreeMemoryPages 0x%lx +%zu KB\n", address, size / 1024);
    madvise(reinterpret_cast<void*>(address), size, MADV_DONTNEED);
    mprotect(reinterpret_cast<void*>(address), size, PROT_NONE);
    return ResultSuccess;
}

void GenerateRandomBytes(void* out, size_t size) { arc4random_buf(out, size); }

}  // namespace nn::os

void nnsdk::signal_system_event(SystemEventType* event) { g_events.get(event).signal(); }

void nnsdk::create_system_event(SystemEventType* event, bool auto_clear, bool signaled) {
    HostEvent& ev = g_events.reset(event);
    ev.auto_clear = auto_clear;
    ev.signaled = signaled;
}

void nnsdk::thread_totals(size_t* stack_bytes, int* count) {
    size_t stack = 0;
    int n = 0;
    g_threads.for_each([&](HostThread& t) {
        if (t.joined || !t.fn) return;  // sin fn: el ThreadType propio de un hilo que no creó nn::os
        stack += t.stack_size;
        n++;
    });
    *stack_bytes = stack;
    *count = n;
}

// Declaradas en nvn/commands.h: los hilos críticos que no crea nn::os (NvnSubmit, el principal) se apuntan solos.
namespace nvn {

void pin_critical_thread() { pin_current_thread(); }

void pin_background_thread() {
    const CpuMasks& m = cpu_masks();
    if (!m.usable || affinity_off()) return;
    cpu_set_t little;
    CPU_ZERO(&little);
    for (int cpu = 0; cpu < 64; cpu++)
        if (CPU_ISSET(cpu, &m.all) && !CPU_ISSET(cpu, &m.big)) CPU_SET(cpu, &little);
    if (CPU_COUNT(&little) > 0) sched_setaffinity(0, sizeof little, &little);
}

void refresh_critical_threads() {
    static std::atomic<int64_t> next{0};
    const int64_t now = now_ns();
    if (now < next.load(std::memory_order_relaxed)) return;
    next.store(now + 1'000'000'000, std::memory_order_relaxed);
    const CpuMasks& m = cpu_masks();
    if (!m.usable || !g_any_critical) return;
    const bool off = affinity_off();
    static bool applied_off = false;  // con g_critical_m
    std::lock_guard lock(g_critical_m);
    const bool flag_changed = off != applied_off;  // entonces no es que se haya perdido
    applied_off = off;
    std::erase_if(g_critical, [&](const CriticalThread& t) { return !apply_mask(t, off ? m.all : m.big, !flag_changed); });
    release_inherited_masks(m);
}

}  // namespace nvn

// --- API en C (nnos*): mismos objetos, misma implementación ---

extern "C" {

// Para el watchdog del loader: los hilos del juego vivos.
int nnsdk_list_threads(pthread_t* out, int max) {
    int n = 0;
    g_threads.for_each([&](HostThread& t) {
        if (t.started && !t.joined && n < max && !pthread_equal(t.pthread, pthread_self())) out[n++] = t.pthread;
    });
    return n;
}

uint32_t nnosCreateThread(ThreadType* t, ThreadFunction fn, void* arg, void* stack, size_t size, int prio) {
    return CreateThread(t, fn, arg, stack, size, prio).value;
}
void nnosDestroyThread(ThreadType* t) { DestroyThread(t); }
void nnosStartThread(ThreadType* t) { StartThread(t); }
void nnosWaitThread(ThreadType* t) { WaitThread(t); }
ThreadType* nnosGetCurrentThread() { return GetCurrentThread(); }
int nnosGetThreadPriority(const ThreadType* t) { return GetThreadPriority(t); }
void nnosSetThreadNamePointer(ThreadType* t, const char* name) { SetThreadNamePointer(t, name); }
void nnosGetThreadCoreMask(int* ideal_core, uint64_t* mask, const ThreadType* t) { GetThreadCoreMask(ideal_core, mask, t); }
void nnosSetThreadCoreMask(ThreadType* t, int ideal_core, uint64_t mask) { SetThreadCoreMask(t, ideal_core, mask); }
void nnosSleepThread(int64_t ns) { SleepThread(TimeSpan{ns}); }
int64_t nnosGetSystemTick() { return GetSystemTick().value; }
int64_t nnosGetSystemTickFrequency() { return GetSystemTickFrequency(); }
int64_t nnosConvertTickToNanoSeconds(int64_t tick) { return ConvertToTimeSpan(Tick{tick}).ns; }

void nnosInitializeMutex(MutexType* m, bool recursive, int level) { InitializeMutex(m, recursive, level); }
void nnosFinalizeMutex(MutexType* m) { FinalizeMutex(m); }
void nnosLockMutex(MutexType* m) { LockMutex(m); }
void nnosUnlockMutex(MutexType* m) { UnlockMutex(m); }

void nnosInitializeEvent(EventType* e, bool signaled, EventClearMode mode) { InitializeEvent(e, signaled, mode); }
void nnosFinalizeEvent(EventType* e) { FinalizeEvent(e); }
void nnosSignalEvent(EventType* e) { SignalEvent(e); }
void nnosWaitEvent(EventType* e) { WaitEvent(e); }

}  // extern "C"
