// nn::os (y su API en C, nnos*) sobre pthreads. Cada objeto del juego se asocia por su
// dirección a un objeto nuestro; si el juego usa uno sin inicializarlo (inicializador
// estático), se crea al vuelo con valores seguros (mutex recursivo, evento manual).
// Aparte: los objetos que el SDK implementa en su propia memoria (SdkMutex, LightEvent) en os_sync.cpp, las
// fibras en os_fiber.cpp y el heap de la aplicación en os_memory.cpp.

#include <dirent.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
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

// Interruptor sin_cache_registro: Registry::get siempre por el cerrojo del trozo y la tabla, como antes (ver la caché).
bool registry_cache_off() { return NVN_FLAG("sin_cache_registro"); }

// Objeto del juego (por dirección) -> objeto nuestro. 64 trozos para no serializar todo.
// Con kCacheSets > 0, cada hilo recuerda además los últimos objetos que buscó (kCacheSets conjuntos de dos, en su
// TLS): get no toma el cerrojo del trozo ni busca en la tabla si la clave está ahí. En el primer combate de Xenoblade 3
// el hilo principal espera a sus trabajos en un bucle que toma y suelta varios mutex por vuelta (main+0x1280e74) y el
// 39 % de su tiempo era este get (el cerrojo del trozo y la búsqueda en la tabla). La clave sigue siendo la dirección
// y no se escribe nada en la memoria del juego: lo que el juego pasa como MutexType no siempre lo es (cxx.cpp monta
// aquí el std::mutex del juego) y puede venir sin inicializar o copiado de otro sitio.
// Una entrada vale mientras nadie haya rehecho o borrado nada en su trozo: reset y erase suben la época del trozo
// con su cerrojo tomado, antes de tocar la tabla, y get compara la época que guardó (leída con ese mismo cerrojo al
// buscar) con la de ahora. Si coinciden, la tabla sigue diciendo lo mismo para esa clave (solo reset y erase cambian
// lo que hay en una clave que ya existe), así que devuelve justo lo que devolvería buscando. Las épocas empiezan en 1:
// una caché recién creada (todo a cero) no acierta nunca.
template <typename T, int kCacheSets = 0>
class Registry {
public:
    T& get(const void* key) {
        const size_t index = shard_index(key);
        if constexpr (kCacheSets > 0) {
            if (T* hit = cached(key, index)) return *hit;
        }
        Shard& s = shards_[index];
        std::lock_guard lock(s.m);
        auto& slot = s.map[key];
        if (!slot) slot = std::make_unique<T>();
        if constexpr (kCacheSets > 0) remember(key, slot.get(), index);
        return *slot;
    }
    template <typename... Args>
    T& reset(const void* key, Args&&... args) {
        const size_t index = shard_index(key);
        Shard& s = shards_[index];
        std::lock_guard lock(s.m);
        epochs_[index].fetch_add(1, std::memory_order_release);  // lo que los hilos recuerden de este trozo ya no vale
        auto& slot = s.map[key];
        slot = std::make_unique<T>(std::forward<Args>(args)...);
        return *slot;
    }
    void erase(const void* key) {
        const size_t index = shard_index(key);
        Shard& s = shards_[index];
        std::lock_guard lock(s.m);
        epochs_[index].fetch_add(1, std::memory_order_release);
        s.map.erase(key);
    }
    // Para el log (registro): búsquedas resueltas por la caché del hilo y por la tabla desde la última vez. Los
    // aciertos se suman de 1024 en 1024 por hilo (un contador compartido por búsqueda costaría más que la búsqueda).
    void take_cache_stats(uint64_t* hits, uint64_t* misses) {
        *hits = cache_hits_.exchange(0, std::memory_order_relaxed);
        *misses = cache_misses_.exchange(0, std::memory_order_relaxed);
    }
    // fn(objeto) solo si la clave existe, y con su trozo tomado: nadie lo puede borrar ni rehacer mientras corre. Para
    // quien toca un objeto del juego desde un hilo nuestro sin saber si el juego ya lo ha destruido.
    template <typename Fn>
    bool with_existing(const void* key, Fn fn) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        auto it = s.map.find(key);
        if (it == s.map.end()) return false;
        fn(*it->second);
        return true;
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
    static size_t shard_index(const void* key) { return (reinterpret_cast<uintptr_t>(key) >> 4) % 64; }
    Shard& shard(const void* key) { return shards_[shard_index(key)]; }
    Shard shards_[64];

    // La caché de cada hilo (una Registry por tipo: la caché va con el tipo). Sin constructor ni destructor: nace a
    // cero con el TLS del hilo y no hay nada que soltar cuando el hilo acaba.
    struct CacheEntry {
        const void* key;
        T* value;
        uint64_t epoch;
    };
    struct Cache {
        CacheEntry sets[kCacheSets > 0 ? kCacheSets : 1][2];
        uint32_t hits;  // aciertos aún sin sumar a cache_hits_
    };
    static Cache& cache() {
        static thread_local Cache c;
        return c;
    }
    static CacheEntry* cache_set(Cache& c, const void* key) {
        static_assert(kCacheSets <= 0 || (kCacheSets & (kCacheSets - 1)) == 0, "potencia de dos");
        const uintptr_t k = reinterpret_cast<uintptr_t>(key);
        return c.sets[((k >> 4) ^ (k >> 11) ^ (k >> 19)) & static_cast<uintptr_t>(kCacheSets - 1)];
    }
    T* cached(const void* key, size_t index) {
        if (registry_cache_off()) return nullptr;
        Cache& c = cache();
        const CacheEntry* set = cache_set(c, key);
        const uint64_t epoch = epochs_[index].load(std::memory_order_acquire);
        for (int way = 0; way < 2; way++) {
            if (set[way].key != key || set[way].epoch != epoch) continue;
            if (++c.hits == 1024) {
                cache_hits_.fetch_add(1024, std::memory_order_relaxed);
                c.hits = 0;
            }
            return set[way].value;
        }
        return nullptr;
    }
    // Con el cerrojo del trozo: la época que se guarda es la de lo que se acaba de leer de la tabla.
    void remember(const void* key, T* value, size_t index) {
        if (registry_cache_off()) return;
        cache_misses_.fetch_add(1, std::memory_order_relaxed);
        CacheEntry* set = cache_set(cache(), key);
        if (set[0].key != key) set[1] = set[0];  // el más antiguo de los dos se va; nunca la misma clave dos veces
        set[0] = {key, value, epochs_[index].load(std::memory_order_relaxed)};
    }
    // Aparte de los trozos: solo se escriben en reset y erase, y así leerlas no comparte línea de caché con el
    // cerrojo de un trozo que otro hilo esté tomando.
    std::atomic<uint64_t> epochs_[64] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::atomic<uint64_t> cache_hits_{0}, cache_misses_{0};
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

    // signal y set devuelven si el evento no estaba ya señalado (YieldThread: quien señala algo que alguien consumió
    // está moviendo a otro hilo, no esperando a ciegas).
    bool signal() {
        const bool changed = set();
        notify_any();
        return changed;
    }
    // Sin avisar a WaitAny (notify_any toma su propio mutex: quien llame con otro cerrojo tomado lo hace después).
    bool set() {
        pthread_mutex_lock(&lock.m);
        const bool changed = !signaled;
        signaled = true;
        auto_clear ? pthread_cond_signal(&cond.c) : pthread_cond_broadcast(&cond.c);
        pthread_mutex_unlock(&lock.m);
        return changed;
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
    struct SuspendState* suspend = nullptr;  // el del propio hilo (thread_local), para SuspendThread
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

Registry<HostMutex, 64> g_mutexes;  // con caché por hilo (ver Registry): los que se usan en bucles
Registry<GameCond, 16> g_conds;
Registry<HostEvent, 16> g_events;
Registry<HostTimer> g_timers;
Registry<HostSemaphore> g_semaphores;
Registry<HostRwLock> g_rwlocks;
Registry<HostMessageQueue> g_queues;
Registry<HostThread> g_threads;
Registry<HostMultiWait> g_multiwaits;
Registry<HostHolder> g_holders;

std::mutex g_regions_m;
std::unordered_map<uintptr_t, size_t> g_regions;
std::unordered_map<uintptr_t, size_t> g_regions_rw;  // las que se mapearon ya escribibles (AllocateAddressRegion)
std::unordered_map<uintptr_t, std::pair<uintptr_t, size_t>> g_regions_map;  // región -> lo mapeado de verdad (munmap)
// Las de g_regions_rw, a la vista de nnsdk::copy_readable, que las mira sin g_regions_m (una vez por uniform buffer de cada
// draw, desde NvnSubmit). Cada hueco: base 0 si está libre; end se escribe antes que base. Quien copia se apunta en
// g_readable_readers antes de mirar los huecos, y quien retira una región pone su base a 0 y luego espera a que no quede
// nadie apuntado (unpublish_readable): o el que copia ve la base a 0, o el que retira le ve apuntado y le espera. Hace falta
// el orden total de las cargas y escrituras atómicas por defecto (seq_cst): no relajarlas.
constexpr int kReadableRegions = 4;
std::atomic<uintptr_t> g_readable_base[kReadableRegions], g_readable_end[kReadableRegions];
std::atomic<int> g_readable_readers{0};
void publish_readable(uintptr_t base, size_t size) {  // con g_regions_m
    for (int k = 0; k < kReadableRegions; k++)
        if (!g_readable_base[k].load()) {
            g_readable_end[k].store(base + size);
            g_readable_base[k].store(base);
            return;
        }
}
// Retira lo publicado que toque [address, address + size). Al volver nadie lo está copiando ni lo copiará sin pasar por el
// kernel: ya se puede desmapear o dejar sin permiso de lectura. true si había algo.
bool unpublish_readable(uintptr_t address, size_t size) {  // con g_regions_m
    bool any = false;
    for (int k = 0; k < kReadableRegions; k++) {
        const uintptr_t base = g_readable_base[k].load();
        if (base && address < g_readable_end[k].load() && address + size > base) g_readable_base[k].store(0), any = true;
    }
    while (any && g_readable_readers.load() != 0) sched_yield();
    return any;
}

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
    // El núcleo rápido (los de la frecuencia máxima más alta; en el SD865, cpu7) y los grandes sin él (cpu4-6), para
    // los hilos que giran (ver note_spinning_thread). has_prime: hay grandes que no son el rápido.
    cpu_set_t prime, gold;
    bool has_prime = false;
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
        CPU_ZERO(&m.prime);
        CPU_ZERO(&m.gold);
        const unsigned long highest = n > 0 ? *std::max_element(freq.begin(), freq.end()) : 0;
        for (int cpu = 0; cpu < n; cpu++)
            if (CPU_ISSET(cpu, &m.big)) CPU_SET(cpu, freq[cpu] == highest ? &m.prime : &m.gold);
        m.has_prime = m.usable && CPU_COUNT(&m.gold) > 0;
        if (m.has_prime)
            printf("[nnsdk] afinidad: núcleo rápido %#llx; los hilos que giran en YieldThread, a %#llx\n",
                   static_cast<unsigned long long>(mask_bits(m.prime)), static_cast<unsigned long long>(mask_bits(m.gold)));
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

// --- Hilos que giran ---
// Un hilo del juego que espera dando vueltas con YieldThread gasta un núcleo entero sin hacer nada y, con la máscara
// completa, el planificador lo pone donde le parece: en el primer combate de Xenoblade 3 el hilo principal (91 % de un
// núcleo esperando a sus trabajos, nice -10 como NvnSubmit) salía en el núcleo rápido en la mitad de las muestras. No
// se lo repartía con NvnSubmit a la vez (en el perfil coinciden en un núcleo en 100 ms de 10 000): se pasaba al rápido
// en cuanto NvnSubmit paraba entre dos frames, y en 195 de sus 217 arranques NvnSubmit lo encontraba ocupado y empezaba
// el frame en uno de 2,4 GHz en vez de 2,8 (el 43 % de su tiempo; los dos grupos iban a su frecuencia máxima).
// El hilo que gira (YieldThread lo avisa aquí, como mucho una vez por segundo) se queda en los núcleos grandes
// sin el rápido: ni le quita el rápido a NvnSubmit ni baja a los pequeños cuando el freno de YieldThread lo deja casi
// siempre dormido (su trabajo de verdad entre espera y espera sí está en el camino del frame). No se toca a los
// críticos (tienen su máscara y refresh_critical_threads la repondría). Si lleva kSpinForgetNs sin girar, vuelve a la
// máscara completa. Interruptores: sin_apartar_giro (máscara completa, como antes) y sin_afinidad; los dos se miran
// como mucho una vez por segundo, en refresh_spinning_threads.
struct SpinningThread {
    CriticalThread thread;
    int64_t last_ns;  // la última vez que avisó de que gira
};
constexpr int64_t kSpinForgetNs = 5'000'000'000;
std::vector<SpinningThread> g_spinning;  // con g_critical_m
std::atomic<uint32_t> g_spinning_count{0};  // para el log
thread_local bool t_spinning = false;  // este hilo está (o estuvo) en g_spinning

bool spin_apart_off() { return NVN_FLAG("sin_apartar_giro") || affinity_off(); }

// Desde YieldThread, en el hilo que gira.
void note_spinning_thread() {
    const CpuMasks& m = cpu_masks();
    if (!m.has_prime || t_critical) return;
    const pid_t tid = gettid();
    const bool off = spin_apart_off();
    std::lock_guard lock(g_critical_m);
    const auto mine = [&](const SpinningThread& t) { return t.thread.tid == tid; };
    auto it = std::find_if(g_spinning.begin(), g_spinning.end(), mine);
    if (it != g_spinning.end()) {
        it->last_ns = now_ns();  // la máscara la mantiene refresh_spinning_threads
        return;
    }
    SpinningThread t{{tid, {}}, now_ns()};
    pthread_getname_np(pthread_self(), t.thread.name, sizeof t.thread.name);
    if (!off) apply_mask(t.thread, m.gold, false);
    g_spinning.push_back(t);
    g_spinning_count = static_cast<uint32_t>(g_spinning.size());
    t_spinning = true;
    g_any_critical = true;  // ya hay una máscara que un hilo nuevo puede heredar (release_inherited_affinity)
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 16))
        printf("[nnsdk] hilo '%s' (tid %d) gira en YieldThread: %s\n", t.thread.name, tid,
               off ? "se deja donde está (sin_apartar_giro o sin_afinidad)" : "núcleos grandes sin el rápido");
}

// Una vez por segundo, desde el hilo que presenta: repone la máscara de los que giran (Android la rehace al cambiar
// el proceso de cpuset), aplica los interruptores y devuelve la completa a los que ya no giran o ya no existen.
void refresh_spinning_threads(const CpuMasks& m, int64_t now) {
    if (!m.has_prime) return;
    const bool off = spin_apart_off();
    std::lock_guard lock(g_critical_m);
    std::erase_if(g_spinning, [&](const SpinningThread& t) {
        // Si después se hizo crítico (el que presenta), manda la máscara de los críticos.
        const auto same = [&](const CriticalThread& c) { return c.tid == t.thread.tid; };
        if (std::any_of(g_critical.begin(), g_critical.end(), same)) return true;
        const bool stale = now - t.last_ns > kSpinForgetNs;
        return !apply_mask(t.thread, off || stale ? m.all : m.gold, false) || stale;
    });
    g_spinning_count = static_cast<uint32_t>(g_spinning.size());
}

// Lo que YieldThread lleva de cada hilo para saber si gira (allí está el porqué), en unidades del contador de
// GetSystemTick. busy lo ponen GetSystemTick y las funciones de nn::os con las que un hilo mueve a otro.
struct SpinState {
    uint64_t last_end = 0;     // al salir del último YieldThread (lo que duerme o cede no cuenta como trabajo)
    uint64_t tight_since = 0;  // desde cuándo cada YieldThread llega a menos de kSpinGapNs del anterior
    uint64_t blind_since = 0;  // lo mismo, y además sin busy entre medias
    uint64_t next_note = 0;    // cuándo volver a avisar a note_spinning_thread
    uint32_t calls = 0;        // llamadas aún sin sumar a g_yield_calls
    bool busy = false;         // desde el último YieldThread leyó el reloj o señaló algo: no espera a ciegas
};
thread_local SpinState t_spin;

// Cuentas de YieldThread para el log (las suma YieldThread: las llamadas, de 256 en 256 por hilo).
std::atomic<uint64_t> g_yield_calls{0}, g_yield_naps{0};

// DEPURACIÓN (con <datos>/registro), cada ~10 s desde refresh_critical_threads: si la caché por hilo de los registros,
// el freno de YieldThread y el apartado de los hilos que giran están actuando y cuánto.
void log_thread_stats(int64_t now) {
    static int64_t next = 0;  // solo el hilo que presenta
    if (now < next) return;
    const bool first = next == 0;
    next = now + 10'000'000'000;
    uint64_t mh, mm, eh, em, ch, cm;
    g_mutexes.take_cache_stats(&mh, &mm);
    g_events.take_cache_stats(&eh, &em);
    g_conds.take_cache_stats(&ch, &cm);
    const uint64_t yields = g_yield_calls.exchange(0, std::memory_order_relaxed);
    const uint64_t naps = g_yield_naps.exchange(0, std::memory_order_relaxed);
    if (first || !NVN_FLAG("registro")) return;
    const auto ull = [](uint64_t v) { return static_cast<unsigned long long>(v); };
    printf("[os] hilos, en 10 s: mutex %llu por la caché del hilo y %llu por el registro (eventos %llu y %llu, variables de "
           "condición %llu y %llu); YieldThread %llu, de ellos %llu con siesta; %u hilos que giran apartados del núcleo "
           "rápido%s\n",
           ull(mh), ull(mm), ull(eh), ull(em), ull(ch), ull(cm), ull(yields), ull(naps),
           g_spinning_count.load(std::memory_order_relaxed), spin_apart_off() ? " (sin apartar: interruptor)" : "");
}

void unpin_current_thread() {  // el hilo termina: su tid se puede reutilizar
    if (t_spinning) {
        t_spinning = false;
        const pid_t tid = gettid();
        std::lock_guard lock(g_critical_m);
        std::erase_if(g_spinning, [&](const SpinningThread& t) { return t.thread.tid == tid; });
    }
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
// de los grandes (la heredada) vuelven a la completa. Con g_critical_m. Igual con la de los hilos que giran (grandes
// sin el rápido) en un hilo que no es uno de ellos.
void release_inherited_masks(const CpuMasks& m) {
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return;
    while (const dirent* e = readdir(dir)) {
        const pid_t tid = static_cast<pid_t>(atoi(e->d_name));
        if (tid <= 0 || std::any_of(g_critical.begin(), g_critical.end(), [&](const CriticalThread& t) { return t.tid == tid; }))
            continue;
        cpu_set_t now;
        CPU_ZERO(&now);
        if (sched_getaffinity(tid, sizeof now, &now) != 0) continue;
        const bool from_spinning = m.has_prime && CPU_EQUAL(&now, &m.gold) &&
            std::none_of(g_spinning.begin(), g_spinning.end(), [&](const SpinningThread& t) { return t.thread.tid == tid; });
        if (CPU_EQUAL(&now, &m.big) || from_spinning) sched_setaffinity(tid, sizeof m.all, &m.all);
    }
    closedir(dir);
}

bool critical_name(const char* name) { return strcmp(name, "MainDraw") == 0; }

void set_pthread_name(pthread_t thread, const char* name) {
    char buf[16];  // Linux admite 15 caracteres: con más, el nombre no se pone (ManaSwitchDecoderWatchThread)
    snprintf(buf, sizeof buf, "%s", name && name[0] ? name : "nn::os");
    pthread_setname_np(thread, buf);
}

// SuspendThread / GetThreadContext (el GC de IL2CPP para el mundo así para buscar punteros en registros y pilas):
// una señal al hilo, que guarda sus registros, avisa y se queda esperando en su manejador hasta el ResumeThread.
// El estado es thread_local y se toca al empezar cada hilo: en un manejador de señal no se puede reservar el TLS
// dinámico de libnnsdk ni tomar el cerrojo de g_threads.
struct SuspendState {
    std::atomic<int> count{0};
    sem_t ack, resume;
    uint64_t x[31] = {}, sp = 0, pc = 0, pstate = 0;
    SuspendState() {
        sem_init(&ack, 0, 0);
        sem_init(&resume, 0, 0);
    }
};
thread_local SuspendState t_suspend;
constexpr int kSuspendSignal = SIGPWR;

void suspend_handler(int, siginfo_t*, void* context) {
    const int saved_errno = errno;
    SuspendState& s = t_suspend;
    const mcontext_t& m = static_cast<ucontext_t*>(context)->uc_mcontext;
    memcpy(s.x, m.regs, sizeof s.x);
    s.sp = m.sp;
    s.pc = m.pc;
    s.pstate = m.pstate;
    sem_post(&s.ack);
    while (sem_wait(&s.resume) != 0) {
    }
    errno = saved_errno;
}

void install_suspend_handler() {
    static std::once_flag once;
    std::call_once(once, [] {
        struct sigaction sa = {};
        sa.sa_sigaction = suspend_handler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigfillset(&sa.sa_mask);
        sigaction(kSuspendSignal, &sa, nullptr);
    });
}

// El stack del ThreadType (+0x50 dirección, +0x58 tamaño), que en la consola rellena CreateThread con la pila del juego.
// El GC de IL2CPP (Disco Elysium, main+0x1176fa0) busca punteros en todo ese rango de cada hilo parado, no desde su
// sp: los hilos corren en pilas de bionic, así que se apunta ahí la de verdad. Sin esto el GC no veía ninguna pila
// (la del principal quedaba a 0) y recogía objetos vivos.
constexpr size_t kThreadStackOffset = 0x50, kThreadStackSizeOffset = 0x58;
void record_stack(ThreadType* type) {
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) return;
    void* stack = nullptr;
    size_t size = 0, guard = 0;
    pthread_attr_getstack(&attr, &stack, &size);
    pthread_attr_getguardsize(&attr, &guard);
    pthread_attr_destroy(&attr);
    // bionic cuenta la página de guarda (PROT_NONE) dentro de la pila: el GC la leería entera.
    if (guard < size) {
        stack = static_cast<uint8_t*>(stack) + guard;
        size -= guard;
    }
    static std::atomic<int> logged{0};
    if (logged++ < 4) printf("[os] pila de %p para el GC: %p +%zu KB (guarda %zu)\n", static_cast<void*>(type), stack, size / 1024, guard);
    auto* bytes = reinterpret_cast<uint8_t*>(type);
    memcpy(bytes + kThreadStackOffset, &stack, sizeof stack);
    memcpy(bytes + kThreadStackSizeOffset, &size, sizeof size);
}

void* thread_entry(void* p) {
    auto* t = static_cast<HostThread*>(p);
    t_current = t->type;
    t->suspend = &t_suspend;
    record_stack(t->type);
    t->tid = gettid();
    set_thread_id(t->type, t->tid);
    set_pthread_name(pthread_self(), t->name);
    if (critical_name(t->name))
        pin_current_thread();
    else
        release_inherited_affinity();
    nnsdk::on_thread_start();
    t->fn(t->arg);
    nnsdk::run_thread_atexit();  // los thread_local del juego con destructor, antes de soltar nada del hilo
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
        t.suspend = &t_suspend;
        record_stack(fake);
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
    if (sent) t_spin.busy = true;  // mueve a otro hilo: no espera a ciegas (ver YieldThread)
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
    t_spin.busy = true;  // mueve a otro hilo: no espera a ciegas (ver YieldThread)
    g_conds.get(c).signal(false);
}
void BroadcastConditionVariable(ConditionVariableType* c) {
    nnsdk::log_signal("BroadcastConditionVariable", c, __builtin_return_address(0));
    t_spin.busy = true;
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
void SignalEvent(EventType* e) {
    if (g_events.get(e).signal()) t_spin.busy = true;  // alguien había consumido el anterior (ver YieldThread)
}
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
    t_spin.busy = true;  // mueve a otro hilo: no espera a ciegas (ver YieldThread)
    HostSemaphore& sem = g_semaphores.get(s);
    pthread_mutex_lock(&sem.lock.m);
    // El SDK aborta si se pasa del máximo; aquí se queda en el máximo.
    if (sem.count + std::max(count, 0) > sem.max) {  // DEPURACIÓN: un aviso perdido así cuelga a quien espere
        static std::atomic<int> logged{0};
        if (logged++ < 16)
            printf("[os] ReleaseSemaphore %p por encima del máximo (%d + %d > %d), desde %p\n", static_cast<void*>(s),
                   sem.count, count, sem.max, __builtin_return_address(0));
    }
    sem.count = std::min(sem.count + std::max(count, 0), sem.max);
    count > 1 ? pthread_cond_broadcast(&sem.cond.c) : pthread_cond_signal(&sem.cond.c);
    pthread_mutex_unlock(&sem.lock.m);
}
void ReleaseSemaphore(SemaphoreType* s) { ReleaseSemaphore(s, 1); }
// Catherine: Full Body la mira antes de esperar en el semáforo de su hilo de carga.
int GetCurrentSemaphoreCount(const SemaphoreType* s) {
    HostSemaphore& sem = g_semaphores.get(const_cast<SemaphoreType*>(s));
    pthread_mutex_lock(&sem.lock.m);
    const int count = sem.count;
    pthread_mutex_unlock(&sem.lock.m);
    return count;
}

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

bool TryWaitThread(ThreadType* type) {
    if (get_state(type) != Terminated) return false;
    WaitThread(type);
    return true;
}

struct ThreadContextInfoAarch64 {
    uint64_t r[29], fp, lr, sp, pc;
    uint32_t pstate, padding;
    uint8_t v[32][16];
    uint32_t fpcr, fpsr;
};

// Devuelve cuántas veces estaba suspendido antes.
int SuspendThread(ThreadType* type) {
    install_suspend_handler();
    HostThread& t = g_threads.get(type);
    static std::atomic<int> logged{0};
    if (logged++ < 8)
        printf("[os] SuspendThread '%s' desde %p\n", t.name, __builtin_return_address(0));
    SuspendState* s = t.suspend;
    if (!s || !t.started || type == current_thread()) return 0;
    const int before = s->count.fetch_add(1);
    if (before == 0) {
        pthread_kill(t.pthread, kSuspendSignal);
        while (sem_wait(&s->ack) != 0) {
        }
    }
    return before;
}
int ResumeThread(ThreadType* type) {
    SuspendState* s = g_threads.get(type).suspend;
    if (!s) return 0;
    int before = s->count.load();
    while (before > 0 && !s->count.compare_exchange_weak(before, before - 1)) {
    }
    if (before == 1) sem_post(&s->resume);
    return before;
}
int GetThreadSuspendCount(const ThreadType* type) {
    SuspendState* s = g_threads.get(const_cast<ThreadType*>(type)).suspend;
    return s ? s->count.load() : 0;
}
// Solo los registros enteros (hasta pstate, 0x110 bytes): el GC solo busca punteros ahí y en la pila, y si el
// ThreadContextInfo del juego fuera más corto que el de Horizon, escribir los v[] le pisaría la pila.
Result GetThreadContext(ThreadContextInfoAarch64* out, const ThreadType* type) {
    SuspendState* s = g_threads.get(const_cast<ThreadType*>(type)).suspend;
    static std::atomic<int> logged{0};
    if (logged++ < 8)
        printf("[os] GetThreadContext de '%s' (%p) suspendido %d, desde %p\n", g_threads.get(const_cast<ThreadType*>(type)).name,
               static_cast<const void*>(type), s ? s->count.load() : -1, __builtin_return_address(0));
    memset(out, 0, offsetof(ThreadContextInfoAarch64, v));
    if (!s) return ResultSuccess;
    memcpy(out->r, s->x, sizeof out->r);
    out->fp = s->x[29];
    out->lr = s->x[30];
    out->sp = s->sp;
    out->pc = s->pc;
    out->pstate = static_cast<uint32_t>(s->pstate);
    return ResultSuccess;
}

Result CreateSystemEvent(SystemEventType* e, EventClearMode mode, bool) {
    nnsdk::create_system_event(e, mode == EventClearMode_AutoClear, false);
    return ResultSuccess;
}
void SignalSystemEvent(SystemEventType* e) {
    t_spin.busy = true;  // mueve a otro hilo: no espera a ciegas (ver YieldThread)
    nnsdk::signal_system_event(e);
}

bool TryAcquireSemaphore(SemaphoreType* s) { return acquire_semaphore(s, 0); }

// Memoria de direcciones virtuales (VammManager del SDK): existe cuando el main.npdm del juego reserva "system
// resource" para sus tablas de páginas (Final Fantasy Resonance: 0x1400000, 20 MB). Unreal Engine la exige: si no
// está, su arranque llama a exit(1) (main+0x13820). Con ella reserva 62 GB de direcciones (AllocateAddressRegion) y
// compromete y suelta páginas a demanda (AllocateMemoryPages / FreeMemoryPages).
constexpr size_t kSystemResourceSize = 0x1400000;
std::atomic<size_t> g_va_committed{0};
std::atomic<size_t> g_va_allocated_total{0}, g_va_freed_total{0};  // DEPURACIÓN
bool IsVirtualAddressMemoryEnabled() { return true; }

// Como VammManager::GetVirtualAddressMemoryResourceUsage (sdk+0x146930): lo asignado es el system resource del
// proceso (svcGetInfo 16) y lo usado, lo que el kernel lleva gastado de él (svcGetInfo 17) más 512 KB, con tope en lo
// asignado. Aquí lo gastado se estima como las tablas de las páginas comprometidas (8 bytes por página de 4 KB).
struct VirtualAddressMemoryResourceUsage {
    size_t assignedSize;
    size_t usedSize;
};
VirtualAddressMemoryResourceUsage GetVirtualAddressMemoryResourceUsage() {
    const size_t used = (512u << 10) + g_va_committed.load(std::memory_order_relaxed) / 512;
    return {kSystemResourceSize, std::min(used, kSystemResourceSize)};
}

// El pseudo-handle del proceso actual, como en el SDK (sdk+0x140e90).
uint32_t GetCurrentProcessHandle() { return 0xFFFF8001; }

enum MemoryPermission { MemoryPermission_None = 0, MemoryPermission_ReadOnly = 1, MemoryPermission_ReadWrite = 3 };
void SetMemoryPermission(uintptr_t address, size_t size, MemoryPermission permission) {
    const int prot = permission == MemoryPermission_ReadWrite ? PROT_READ | PROT_WRITE
                     : permission == MemoryPermission_ReadOnly ? PROT_READ
                                                               : PROT_NONE;
    if (prot == PROT_NONE) {
        // Dentro de una región que copy_readable da por legible: deja de serlo entera y para siempre (no se lleva la cuenta
        // de qué páginas vuelven a tener permiso). Lo que se lea de ella vuelve a ir por el kernel.
        std::lock_guard lock(g_regions_m);
        if (unpublish_readable(address, size))
            printf("[nnsdk] SetMemoryPermission 0x%lx +%zu KB sin lectura: su región ya no se copia sin el kernel\n", address,
                   size / 1024);
    }
    mprotect(reinterpret_cast<void*>(address), size, prot);
}
namespace {
// --- YieldThread sin quemar un núcleo ---
// El hilo principal de Xenoblade 3 espera a que acaben sus trabajos dando vueltas (main+0x1280e74: mira las colas con
// sus mutex, despierta a los workers y YieldThread; en el primer combate, casi todo el frame, mientras NvnSubmit
// reproduce). sched_yield no duerme a nadie: con un núcleo para él solo vuelve al momento, y el hilo gastaba el
// 91 % de un núcleo grande (calor: el SoC a 85 °C) y la mitad del tiempo el núcleo rápido de NvnSubmit. Cuando un
// hilo lleva kSpinNs dando vueltas a ciegas, cada YieldThread duerme kSpinNapNs en vez de ceder: "a ciegas" es con
// menos de kSpinGapNs de trabajo propio entre un YieldThread y el siguiente y sin que entre medias (SpinState::busy):
// - haya leído el reloj. El limitador de frames del juego (main+0x124fb14) y su spinlock con plazo (main+0xe93888)
//   también esperan con YieldThread, pero mirando GetSystemTick en cada vuelta (el limitador, hasta el 99,7 % del
//   periodo): esos no se frenan (siguen con sched_yield), así que el instante en que acaba cada frame no cambia.
// - haya señalado un evento que no lo estaba ya (o una variable de condición, un semáforo o una cola). El bucle de
//   espera del juego despierta en cada vuelta a los workers con trabajo; si un worker consume esos avisos, el que
//   espera lo está moviendo y sigue a toda velocidad. Si nadie los consume (el worker está ocupado o bloqueado: en la
//   espera larga del combate ningún worker corre), el evento sigue señalado mientras se duerme.
// Lo peor que añade el freno: una espera que ya llevaba kSpinNs sin que pasara nada se entera de su final (o da su
// siguiente aviso) una siesta tarde: 30 µs más la holgura del temporizador de Linux, 50 µs para un hilo en primer
// plano, ~0,1 ms; una vez por espera, no por vuelta. Interruptor sin_freno_yield: siempre sched_yield, como antes.
constexpr int64_t kSpinGapNs = 20'000, kSpinNs = 500'000, kSpinNapNs = 30'000;
constexpr int64_t kSpinApartNs = 2'000'000;  // girando (a ciegas o no) este rato seguido: note_spinning_thread

// El contador de GetSystemTick, sin barrera (aquí solo se miden ratos de decenas de microsegundos).
uint64_t spin_ticks() {
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
uint64_t spin_ticks_of(int64_t ns) {
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f * static_cast<uint64_t>(ns) / 1'000'000'000;
}
}  // namespace

void YieldThread() {
    SpinState& s = t_spin;
    const uint64_t now = spin_ticks();
    const bool tight = s.last_end != 0 && now - s.last_end < spin_ticks_of(kSpinGapNs);
    if (!tight) s.tight_since = now;
    if (!tight || s.busy) s.blind_since = now;
    s.busy = false;
    if (++s.calls == 256) {
        g_yield_calls.fetch_add(256, std::memory_order_relaxed);
        s.calls = 0;
    }
    if (now - s.tight_since >= spin_ticks_of(kSpinApartNs) && now >= s.next_note) {
        s.next_note = now + spin_ticks_of(1'000'000'000);
        note_spinning_thread();
    }
    if (now - s.blind_since >= spin_ticks_of(kSpinNs) && !NVN_FLAG("sin_freno_yield")) {
        g_yield_naps.fetch_add(1, std::memory_order_relaxed);
        const timespec nap{0, static_cast<long>(kSpinNapNs)};
        nanosleep(&nap, nullptr);
    } else {
        sched_yield();
    }
    s.last_end = spin_ticks();
}
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
// Hilos que aún puede crear el proceso (la aplicación tiene ~96 en la consola); aquí no hay tope real.
int GetFreeThreadCount() { return 64; }
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
    t_spin.busy = true;  // quien espera mirando el reloj no espera a ciegas (ver YieldThread)
    return Tick{static_cast<int64_t>(v)};
}
// El SDK lee CNTPCT_EL0 entre barreras; en Android EL0 solo puede leer el virtual (mismo ritmo).
Tick GetSystemTickOrdered() {
    uint64_t v;
    asm volatile("dsb ish; isb; mrs %0, cntvct_el0; isb" : "=r"(v)::"memory");
    t_spin.busy = true;
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

// La región se mapea ya legible y escribible, sin reservar memoria (MAP_NORESERVE): comprometer páginas no cuesta
// nada y soltarlas es un MADV_DONTNEED (vuelven a cero al tocarlas, como las páginas nuevas de la consola). Con
// PROT_NONE y un mprotect por llamada, el allocator de Unreal (miles de bloques de 64 KB que van y vienen) partiría
// la región en decenas de miles de VMA y Linux deja 65 530 por proceso.
// Alineada como en la consola (VammManager::AllocateAddressRegion, sdk+0x145e00): a 2 MB las regiones de 2 MB o más y
// a 64 KB las menores. Unreal reparte su región en bloques cuya alineación da por hecha (MallocBinned3: "alignment was
// too large for OS" si un bloque no sale alineado). Se mapea de más y se devuelve la primera dirección alineada.
Result AllocateAddressRegion(uintptr_t* out, size_t size) {
    const size_t align = size >= (size_t{2} << 20) ? size_t{2} << 20 : size_t{64} << 10;
    const size_t mapped = size + align;
    bool rw = true;
    void* p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        rw = false;
        p = mmap(nullptr, mapped, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    }
    if (p == MAP_FAILED) {
        printf("[nnsdk] AllocateAddressRegion +%zu MB: %s\n", size >> 20, strerror(errno));
        return Result{0x1003};
    }
    const uintptr_t base = (reinterpret_cast<uintptr_t>(p) + align - 1) & ~(align - 1);
    std::lock_guard lock(g_regions_m);
    g_regions[base] = size;
    g_regions_map[base] = {reinterpret_cast<uintptr_t>(p), mapped};
    if (rw) g_regions_rw[base] = size;
    if (rw) publish_readable(base, size);  // para copy_readable
    *out = base;
    printf("[nnsdk] AllocateAddressRegion 0x%lx +%zu MB%s\n", base, size >> 20,
           rw ? "" : " (PROT_NONE: se compromete con mprotect)");
    return ResultSuccess;
}
Result FreeAddressRegion(uintptr_t address) {
    std::lock_guard lock(g_regions_m);
    auto it = g_regions.find(address);
    if (it == g_regions.end()) return Result{0x1801};
    const auto map = g_regions_map[address];
    unpublish_readable(address, it->second);  // antes de desmapear: que nadie esté copiando de ella (copy_readable)
    munmap(reinterpret_cast<void*>(map.first), map.second);
    g_regions.erase(it);
    g_regions_map.erase(address);
    g_regions_rw.erase(address);
    return ResultSuccess;
}
namespace {
// ¿Cae la dirección en una región mapeada ya escribible? (pocas regiones: Unreal reserva una).
bool region_rw(uintptr_t address) {
    std::lock_guard lock(g_regions_m);
    for (const auto& [base, size] : g_regions_rw)
        if (address >= base && address < base + size) return true;
    return false;
}
std::atomic<uint32_t> g_va_calls{0};
}  // namespace
// os_memory.cpp (mismo layout).
struct MemoryInfo {
    size_t total_available_memory_size;
    size_t total_used_memory_size;
    size_t total_memory_heap_size;
    size_t allocated_memory_heap_size;
    size_t program_size;
    size_t total_thread_stack_size;
    int thread_count;
};
void QueryMemoryInfo(MemoryInfo* info);

Result AllocateMemoryPages(uintptr_t address, size_t size) {
    if (g_va_calls.fetch_add(1, std::memory_order_relaxed) < 8)
        printf("[nnsdk] AllocateMemoryPages 0x%lx +%zu KB\n", address, size / 1024);
    // Como en la consola, no se compromete más memoria física de la que tiene la aplicación (QueryMemoryInfo: total
    // disponible menos lo usado). Xenoblade 3 mide así su presupuesto al arrancar (main+0x12434b0): compromete de MB en
    // MB hasta que falla, apunta cuánto consiguió y lo suelta. Sin límite se llevaba los 60 GB de su región y se quedaba
    // sin presupuesto apuntado. <datos>/sin_limite_va: como antes.
    {
        MemoryInfo info{};
        QueryMemoryInfo(&info);
        static const bool sin_limite = NVN_FLAG("sin_limite_va");
        if (!sin_limite && info.total_used_memory_size + size > info.total_available_memory_size) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
                printf("[nnsdk] AllocateMemoryPages 0x%lx +%zu KB: sin memoria (usados %zu MB de %zu MB)\n", address,
                       size / 1024, info.total_used_memory_size >> 20, info.total_available_memory_size >> 20);
            return Result{0x1003};
        }
    }
    if (!region_rw(address) && mprotect(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE) != 0) {
        printf("[nnsdk] AllocateMemoryPages 0x%lx +%zu KB: %s\n", address, size / 1024, strerror(errno));
        return Result{0x1003};  // os, 8: sin memoria (Unreal lo trata como tal, main+0x27f0c)
    }
    const size_t now = g_va_committed.fetch_add(size, std::memory_order_relaxed) + size;
    g_va_allocated_total.fetch_add(size, std::memory_order_relaxed);
    // DEPURACIÓN: cada 128 MB más que el máximo anterior, y en cada petición de 8 MB o más (las 40 primeras), quién
    // compromete la memoria (pila por los frame pointers del juego). Unreal lleva su propia cuenta contra lo que
    // dio QueryMemoryInfo al arrancar y aborta cuando se le acaba.
    static std::atomic<size_t> mark{0};
    static std::atomic<int> big{0};
    size_t m = mark.load(std::memory_order_relaxed);
    const bool crossed = now >= m + (128u << 20) && mark.compare_exchange_strong(m, now & ~size_t{(128u << 20) - 1});
    if (crossed || (size >= (8u << 20) && big.fetch_add(1, std::memory_order_relaxed) < 40)) {
        char bt[320];
        int n = 0;
        const auto& h = nnsdk::host();
        const auto base = reinterpret_cast<uintptr_t>(h.main_base);
        auto* fp = static_cast<uintptr_t*>(__builtin_frame_address(0));
        for (int i = 0; i < 20 && fp && n < 290; i++) {
            const uintptr_t ret = fp[1];
            if (ret >= base && ret < base + h.main_size)
                n += snprintf(bt + n, sizeof bt - n, " %llx", static_cast<unsigned long long>(ret - base));
            auto* next = reinterpret_cast<uintptr_t*>(fp[0]);
            if (next <= fp || reinterpret_cast<uintptr_t>(next) - reinterpret_cast<uintptr_t>(fp) > (8u << 20)) break;
            fp = next;
        }
        bt[n] = 0;
        char name[32] = "?";
        pthread_getname_np(pthread_self(), name, sizeof name);
        printf("[nnsdk] memoria virtual: %zu MB comprometidos (pedidos %zu MB, soltados %zu MB en total); ahora +%zu KB en "
               "'%s'; pila%s\n",
               now >> 20, g_va_allocated_total.load() >> 20, g_va_freed_total.load() >> 20, size >> 10, name, bt);
    }
    return ResultSuccess;
}
Result FreeMemoryPages(uintptr_t address, size_t size) {
    g_va_freed_total.fetch_add(size, std::memory_order_relaxed);
    madvise(reinterpret_cast<void*>(address), size, MADV_DONTNEED);
    if (!region_rw(address)) mprotect(reinterpret_cast<void*>(address), size, PROT_NONE);
    size_t before = g_va_committed.load(std::memory_order_relaxed);
    while (!g_va_committed.compare_exchange_weak(before, before > size ? before - size : 0, std::memory_order_relaxed)) {}
    return ResultSuccess;
}

void GenerateRandomBytes(void* out, size_t size) { arc4random_buf(out, size); }

}  // namespace nn::os

void nnsdk::signal_system_event(SystemEventType* event) { g_events.get(event).signal(); }

// El AudioRenderer señala el evento del juego cada 5 ms desde un hilo suyo, y el reproductor de vídeo de Unity lo
// destruye con el renderer aún en marcha (main+0x397c70, DestroySystemEvent antes de StopAudioRenderer). Con
// signal_system_event se podía señalar un HostEvent a medio destruir, y si no, dejarlo creado otra vez en esa dirección.
bool nnsdk::signal_system_event_if_alive(SystemEventType* event) {
    if (!g_events.with_existing(event, [](HostEvent& ev) { ev.set(); })) return false;
    notify_any();
    return true;
}
bool nnsdk::clear_system_event_if_alive(SystemEventType* event) {
    return g_events.with_existing(event, [](HostEvent& ev) { ev.clear(); });
}

void nnsdk::create_system_event(SystemEventType* event, bool auto_clear, bool signaled) {
    HostEvent& ev = g_events.reset(event);
    ev.auto_clear = auto_clear;
    ev.signaled = signaled;
}

size_t nnsdk::va_committed() { return nn::os::g_va_committed.load(std::memory_order_relaxed); }

bool nnsdk::copy_readable(void* dst, const void* src, size_t size) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(src);
    g_readable_readers.fetch_add(1);  // antes de mirar los huecos (ver g_readable_base)
    bool ok = false;
    for (int k = 0; k < kReadableRegions && !ok; k++) {
        const uintptr_t base = g_readable_base[k].load();
        if (!base || a < base) continue;
        const uintptr_t end = g_readable_end[k].load();
        ok = a < end && size <= end - a;
    }
    if (ok) memcpy(dst, src, size);
    g_readable_readers.fetch_sub(1);
    return ok;
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
    log_thread_stats(now);
    refresh_spinning_threads(m, now);  // los hilos que giran en YieldThread: su máscara y sus interruptores
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
