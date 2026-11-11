// nvnQueue* y nvnSync*. Enviar comandos los graba y envía a Vulkan al momento (executor.cpp), sin
// esperar a la GPU; un NVNsync recuerda el último envío al hacer FenceSync y esperar el sync es esperar
// a ese envío. Adquirir textura de la ventana es rotar.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"

namespace {

// NVNqueueBuilder: 64 bytes opacos (Xenoblade lo tiene en la pila de main+0x706e70, de sp+0x10 a sp+0x50); guardamos aquí
// lo que se pone, para los getters.
struct QueueBuilder {
    void* device;
    int flags;
    int command_memory, compute_memory, control_memory;
    int flush_threshold;
    int priority;  // NVNqueuePriority: LOW 0, NORMAL 1, HIGH 2
    void* queue_memory;
    size_t queue_memory_size;
};
static_assert(sizeof(QueueBuilder) <= 64);
constexpr int kQueuePriorityNormal = 1;

struct Queue {};

// NVNevent: un valor de 32 bits en un memory pool (pool + offset del builder) que escriben la CPU (nvnEventSignal) y la cola
// (SignalEvent) y por el que la cola puede esperar (WaitEvent).
struct Event {
    uint32_t* value = nullptr;
    void* pool = nullptr;
    int64_t offset = 0;
};
nvn::Objects<Event> g_events;
struct Sync {
    uint64_t submit = 0;  // envío que lo señala (0: señalado desde el principio)
    // solapar_frame: lo puso nvnWindowAcquireTexture ("textura libre"): basta con que el ejecutor haya reproducido el envío,
    // sin esperar a la GPU. Un FenceSync lo quita.
    bool replay_only = false;
};

// NVNsyncWaitResult.
constexpr int kSyncAlreadySignaled = 0, kSyncConditionSatisfied = 1, kSyncTimeoutExpired = 2;
constexpr int kWindowAcquireSuccess = 0;

nvn::Objects<Queue> g_queues;
nvn::Objects<Sync> g_syncs;

}  // namespace

extern "C" {

void nvnQueueBuilderSetDefaults(QueueBuilder* b) {
    memset(b, 0, sizeof *b);
    b->priority = kQueuePriorityNormal;
}
void nvnQueueBuilderSetDevice(QueueBuilder* b, void* device) { b->device = device; }
void nvnQueueBuilderSetFlags(QueueBuilder* b, int flags) { b->flags = flags; }
void nvnQueueBuilderSetCommandMemorySize(QueueBuilder* b, size_t size) { b->command_memory = size; }
void nvnQueueBuilderSetComputeMemorySize(QueueBuilder* b, size_t size) { b->compute_memory = size; }
void nvnQueueBuilderSetControlMemorySize(QueueBuilder* b, size_t size) { b->control_memory = size; }
// El umbral a partir del cual NVN manda solo lo grabado a la GPU: aquí cada SubmitCommands ya lo manda entero (execute).
void nvnQueueBuilderSetCommandFlushThreshold(QueueBuilder* b, size_t threshold) { b->flush_threshold = static_cast<int>(threshold); }
// Lo que necesita la cola en su memoria: en NVN, su memoria de comandos, de compute y de control, en páginas de 4 KB. La
// nuestra no la usa (lo grabado va en las listas), pero se pide lo mismo que pediría NVN para lo que dice el builder.
size_t nvnQueueBuilderGetQueueMemorySize(const QueueBuilder* b) {
    auto page = [](int size) { return (static_cast<size_t>(std::max(size, 0)) + 4095) & ~size_t{4095}; };
    return std::max<size_t>(page(b->command_memory) + page(b->compute_memory) + page(b->control_memory), 4096);
}
void nvnQueueBuilderSetQueueMemory(QueueBuilder* b, void* memory, size_t size) {
    b->queue_memory = memory;
    b->queue_memory_size = size;
}
void nvnQueueBuilderSetQueuePriority(QueueBuilder* b, int priority) { b->priority = priority; }
int nvnQueueBuilderGetQueuePriority(const QueueBuilder* b) { return b->priority; }
void* nvnQueueBuilderGetDevice(const QueueBuilder* b) { return b->device; }
int nvnQueueBuilderGetFlags(const QueueBuilder* b) { return b->flags; }
// Los tamaños: NVN_TRUE si el builder fija uno (si no, la cola usa el de por defecto y se devuelve 0).
nvn::Boolean nvnQueueBuilderGetCommandMemorySize(const QueueBuilder* b, int* size) {
    if (size) *size = b->command_memory;
    return b->command_memory != 0;
}
nvn::Boolean nvnQueueBuilderGetComputeMemorySize(const QueueBuilder* b, int* size) {
    if (size) *size = b->compute_memory;
    return b->compute_memory != 0;
}
nvn::Boolean nvnQueueBuilderGetControlMemorySize(const QueueBuilder* b, int* size) {
    if (size) *size = b->control_memory;
    return b->control_memory != 0;
}
nvn::Boolean nvnQueueBuilderGetCommandFlushThreshold(const QueueBuilder* b, int* threshold) {
    if (threshold) *threshold = b->flush_threshold;
    return b->flush_threshold != 0;
}
// La memoria que se le dio con SetQueueMemory.
void* nvnQueueBuilderGetMemory(const QueueBuilder* b) { return b->queue_memory; }
size_t nvnQueueBuilderGetMemorySize(const QueueBuilder* b) { return b->queue_memory_size; }

nvn::Boolean nvnQueueInitialize(void* queue, const QueueBuilder*) {
    g_queues.create(queue);
    return 1;
}
void nvnQueueFinalize(void* queue) { g_queues.destroy(queue); }

void nvnQueueSubmitCommands(void* queue, int count, const uint64_t* handles) {
    if (!g_queues.get(queue)) return;
    std::vector<const nvn::CommandList*> lists;
    bool capture = false;
    for (int k = 0; k < count; k++)
        if (handles[k]) {
            lists.push_back(reinterpret_cast<const nvn::CommandList*>(handles[k]));
            capture = capture || lists.back()->cpu_capture;
        }
    nvn::execute(lists);
    // Una lista pinta la textura de captura que la CPU leerá de su memoria unos ticks después sin ningún sync (ver
    // CommandList::cpu_capture): la reproducción y la GPU se esperan aquí. Una parada por captura (al guardar la partida).
    // <datos>/captura_sin_espera lo quita.
    if (capture && !NVN_FLAG("captura_sin_espera")) {
        const auto t0 = std::chrono::steady_clock::now();
        nvn::wait_idle();
        static int logged = 0;
        if (nvn::first_times(logged, 8) || NVN_FLAG("diag_miniatura"))
            NVN_LOG("captura: espera a la reproducción y a la GPU, %.1f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}
// Mandar a la GPU lo enviado: SubmitCommands ya lo pasa a la cola de envíos, que lo reproduce en cuanto puede.
void nvnQueueFlush(void*) {}
void nvnQueueFinish(void*) {
    static int logged = 0;  // DEPURACIÓN (miniatura de la partida)
    if (nvn::first_times(logged, 20) || NVN_FLAG("diag_miniatura")) NVN_LOG("nvnQueueFinish (envío %llu)\n", static_cast<unsigned long long>(nvn::last_submit()));
    nvn::wait_idle();
}
void nvnQueueFenceSync(void*, void* sync, int, int) { nvn::sync_signal_at(sync, nvn::last_submit()); }
// Espera de la GPU a un sync: un NVNsync solo lo señala un FenceSync de esta misma cola (no hay más colas ni syncs de
// otras APIs), así que lo enviado antes ya va antes en el orden. NVN_TRUE: la espera quedó puesta.
nvn::Boolean nvnQueueWaitSync(void*, const void*) { return 1; }

void nvnQueuePresentTexture(void*, void* window, int index) {
    // El hilo principal del juego lo crea el loader, no nn::os, y no tiene nombre del juego: se le reconoce por
    // ser el que presenta (y envía los comandos). Es crítico como MainDraw y NvnSubmit.
    thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        nvn::pin_critical_thread();
    }
    nvn::refresh_critical_threads();
    if (nvn::Window* w = nvn::g_windows.get(window)) nvn::queue_present(*w, index);
}

// NVNeventBuilder: dónde vive el valor del evento (pool + offset). Xenoblade lo lee de vuelta con GetMemoryOffset;
// los eventos en sí (Signal/Wait en la cola) no los usa.
struct EventBuilderDesc {
    void* pool;
    int64_t offset;
};
void nvnEventBuilderSetDefaults(EventBuilderDesc* b) { *b = {}; }
void nvnEventBuilderSetStorage(EventBuilderDesc* b, void* pool, int64_t offset) {
    b->pool = pool;
    b->offset = offset;
}
void* nvnEventBuilderGetStorage(const EventBuilderDesc* b, int64_t* offset) {
    if (offset) *offset = b->offset;
    return b->pool;
}
void* nvnEventBuilderGetMemoryPool(const EventBuilderDesc* b) { return b->pool; }
int64_t nvnEventBuilderGetMemoryOffset(const EventBuilderDesc* b) { return b->offset; }

nvn::Boolean nvnEventInitialize(void* event, const EventBuilderDesc* b) {
    nvn::MemoryPool* pool = b ? nvn::g_memory_pools.get(b->pool) : nullptr;
    if (!pool || !pool->storage || b->offset < 0 || static_cast<size_t>(b->offset) + 4 > pool->size) {
        NVN_LOG("evento %p sobre un memory pool desconocido %p (+%lld)\n", event, b ? b->pool : nullptr,
                static_cast<long long>(b ? b->offset : 0));
        return 0;
    }
    Event& e = g_events.create(event);
    e.value = reinterpret_cast<uint32_t*>(pool->storage + b->offset);
    e.pool = b->pool;
    e.offset = b->offset;
    return 1;
}
void nvnEventFinalize(void* event) { g_events.destroy(event); }
uint32_t nvnEventGetValue(const void* event) {
    Event* e = g_events.get(event);
    return e ? __atomic_load_n(e->value, __ATOMIC_ACQUIRE) : 0;
}
// NVNeventSignalMode: WRITE 0, ADD 1. Desde la CPU, al momento.
void nvnEventSignal(void* event, int mode, uint32_t value) {
    if (Event* e = g_events.get(event)) nvn::signal_event_value(e->value, mode, value);
}
void* nvnEventGetMemoryPool(const void* event) {
    Event* e = g_events.get(event);
    return e ? e->pool : nullptr;
}
int64_t nvnEventGetMemoryOffset(const void* event) {
    Event* e = g_events.get(event);
    return e ? e->offset : 0;
}

nvn::Boolean nvnSyncInitialize(void* sync, void*) {
    g_syncs.create(sync);
    return 1;
}
void nvnSyncFinalize(void* sync) { g_syncs.destroy(sync); }
// DEPURACIÓN (con <datos>/registro): cuánto espera el juego a sus syncs por segundo y cuántas esperas caducan.
void count_wait(int64_t ns, bool timed_out, uint64_t timeout_ns) {
    static std::mutex m;
    static int64_t window_start = 0, waited = 0;
    static int waits = 0, timeouts = 0;
    static uint64_t last_timeout = 0;
    if (!NVN_FLAG("registro")) return;
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard lock(m);
    waited += ns, waits++, timeouts += timed_out;
    if (timed_out) last_timeout = timeout_ns;
    if (!window_start) window_start = now;
    if (now - window_start < 1'000'000'000) return;
    NVN_LOG("esperas de syncs: %.0f ms en %d (%d caducadas, plazo %.1f ms)\n", waited / 1e6, waits, timeouts,
            last_timeout / 1e6);
    window_start = now, waited = 0, waits = timeouts = 0;
}
int nvnSyncWait(const void* sync, uint64_t timeout_ns) {
    static int logged = 0;  // DEPURACIÓN (miniatura de la partida)
    if (NVN_FLAG("diag_miniatura") && nvn::first_times(logged, 2000))
        NVN_LOG("nvnSyncWait %p plazo %llu ns (envío %llu)\n", sync, static_cast<unsigned long long>(timeout_ns),
                static_cast<unsigned long long>(nvn::last_submit()));
    Sync* s = g_syncs.get(sync);
    if (!s || !s->submit) return kSyncAlreadySignaled;
    const uint64_t submit = s->submit;
    // Con devoluciones de render targets a la memoria del juego pendientes (miniatura de la partida: el juego lee la textura
    // desde la CPU unos ticks después, sin ningún sync propio), el sync de la ventana espera también a la GPU.
    const bool forced = s->replay_only && nvn::cpu_rt_pending() > 0;
    if (forced && NVN_FLAG("diag_miniatura")) NVN_LOG("nvnSyncWait: espera a la GPU forzada por %d devoluciones pendientes\n", nvn::cpu_rt_pending());
    auto wait = s->replay_only && !forced ? nvn::wait_replayed : nvn::wait_submit;
    if (wait(submit, 0)) return kSyncAlreadySignaled;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = wait(submit, timeout_ns);
    count_wait((std::chrono::steady_clock::now() - start).count(), !ok, timeout_ns);
    return ok ? kSyncConditionSatisfied : kSyncTimeoutExpired;
}

// La siguiente textura de la ventana, rotando por las activas (SetNumActiveTextures; 0: todas). NVNwindowAcquireTextureResult
// y NVNqueueAcquireTextureResult: SUCCESS 0, NATIVE_ERROR 1.
static int acquire_window_texture(void* window, int* index) {
    nvn::Window* w = nvn::g_windows.get(window);
    if (!w || w->textures.empty() || !index) return 1;
    const int count = static_cast<int>(w->textures.size());
    const int active = w->num_active > 0 ? std::min(w->num_active, count) : count;
    if (w->next_texture >= active) w->next_texture = 0;
    *index = w->next_texture;
    w->next_texture = (w->next_texture + 1) % active;
    return kWindowAcquireSuccess;
}
// El sync (textureAvailableSync) se señala cuando la textura está libre. Antes se ignoraba y Xenoblade, que le pasa el mismo
// sync al que acaba de hacer FenceSync (grlib::CGLibDisplay::renderWindow, main+0x728360: FenceSync, PresentTexture y
// AcquireTexture) y lo espera al empezar el frame siguiente (CGLibDisplay::swapChain, main+0x72c180, desde
// ml::DevGraphNx::renderSyncSys), esperaba a que la GPU acabara el frame: el siguiente no se grababa ni se traducía
// hasta entonces, y en el campo la GPU se quedaba ~8,3 ms por frame esperando al primer envío (36 ms por frame con 28 de
// GPU). Activo: la textura está libre cuando el ejecutor ha reproducido todo lo enviado (en la GPU, lo de después ya va
// detrás en la cola, y el presentar la copia en orden); la GPU sigue con ese frame mientras el juego graba el siguiente.
// Medido en Colonia 9: de 26,9 a 30,1 fps. <datos>/sin_solapar_frame (en vivo): como antes (el sync se queda con su
// FenceSync y el juego espera a la GPU).
int nvnWindowAcquireTexture(void* window, void* sync, int* index) {
    const int r = acquire_window_texture(window, index);
    if (sync && !NVN_FLAG("sin_solapar_frame")) {
        Sync* s = g_syncs.get(sync);
        if (!s) s = &g_syncs.create(sync);
        s->submit = nvn::last_submit();
        s->replay_only = true;
    }
    return r;
}
// Como el de la ventana, pero la espera a que la textura esté libre la hace la cola: aquí no hay que esperar (presentar
// copia la textura al swapchain en orden con los envíos).
int nvnQueueAcquireTexture(void*, void* window, int* index) { return acquire_window_texture(window, index); }

// NVNqueueGetErrorResult: 0 = GPU_NO_ERROR (los errores de Vulkan van al log, no al juego).
int nvnQueueGetError(void*, void*) { return 0; }
// Memoria propia de la cola (la de SetQueueMemory) usada desde el último reinicio: la nuestra no usa nada de ella (lo
// grabado vive en las listas y los envíos en la cola de envíos), así que lo usado es 0 de verdad y no hay qué reiniciar.
size_t nvnQueueGetTotalCommandMemoryUsed(void*) { return 0; }
size_t nvnQueueGetTotalComputeMemoryUsed(void*) { return 0; }
size_t nvnQueueGetTotalControlMemoryUsed(void*) { return 0; }
void nvnQueueResetMemoryUsageCounts(void*) {}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnQueueSetDebugLabel(void*, const char*) {}
void nvnSyncSetDebugLabel(void*, const char*) {}
// Interoperabilidad con OpenGL y con los syncs de NvRm (el driver de Tegra): en Android no hay ni un contexto GL del juego
// ni NvRm, así que no se pueden crear ni importar. NVN_FALSE (o nulo): no se hizo.
nvn::Boolean nvnSyncCreateGLSync(const void*, uint64_t*) { return 0; }
nvn::Boolean nvnSyncInitializeFromFencedGLSync(void*, void*, const void*, int*) { return 0; }
nvn::Boolean nvnSyncInitializeFromNvRmSyncNVX(void*, void*, const void*) { return 0; }
nvn::Boolean nvnSyncGetNvRmSyncNVX(const void*, void*) { return 0; }

}  // extern "C"

void nvn::signal_event_value(uint32_t* value, int mode, uint32_t v) {
    if (!value) return;
    if (mode == 1)
        __atomic_fetch_add(value, v, __ATOMIC_ACQ_REL);
    else
        __atomic_store_n(value, v, __ATOMIC_RELEASE);
}
uint32_t* nvn::event_value(const void* event) {
    Event* e = g_events.get(event);
    return e ? e->value : nullptr;
}

void nvn::sync_signal_at(void* sync, uint64_t submit) {
    Sync* s = g_syncs.get(sync);
    if (!s) s = &g_syncs.create(sync);
    s->submit = submit;
    s->replay_only = false;
}

NVN_PROC(nvnQueueBuilderSetDefaults);
NVN_PROC(nvnQueueBuilderSetDevice);
NVN_PROC(nvnQueueBuilderSetFlags);
NVN_PROC(nvnQueueBuilderSetCommandMemorySize);
NVN_PROC(nvnQueueBuilderSetComputeMemorySize);
NVN_PROC(nvnQueueBuilderSetControlMemorySize);
NVN_PROC(nvnQueueBuilderSetCommandFlushThreshold);
NVN_PROC(nvnQueueBuilderGetQueueMemorySize);
NVN_PROC(nvnQueueBuilderSetQueueMemory);
NVN_PROC(nvnQueueInitialize);
NVN_PROC(nvnQueueFinalize);
NVN_PROC(nvnQueueSubmitCommands);
NVN_PROC(nvnQueueFlush);
NVN_PROC(nvnQueueFinish);
NVN_PROC(nvnQueueFenceSync);
NVN_PROC(nvnQueueWaitSync);
NVN_PROC(nvnQueuePresentTexture);
NVN_PROC(nvnSyncInitialize);
NVN_PROC(nvnEventBuilderSetDefaults);
NVN_PROC(nvnEventBuilderSetStorage);
NVN_PROC(nvnEventBuilderGetStorage);
NVN_PROC(nvnEventBuilderGetMemoryPool);
NVN_PROC(nvnEventBuilderGetMemoryOffset);
NVN_PROC(nvnSyncFinalize);
NVN_PROC(nvnSyncWait);
NVN_PROC(nvnWindowAcquireTexture);
NVN_PROC(nvnQueueGetError);
NVN_PROC(nvnQueueSetDebugLabel);
NVN_PROC(nvnSyncSetDebugLabel);
NVN_PROC(nvnQueueBuilderSetQueuePriority);
NVN_PROC(nvnQueueBuilderGetQueuePriority);
NVN_PROC(nvnQueueBuilderGetDevice);
NVN_PROC(nvnQueueBuilderGetFlags);
NVN_PROC(nvnQueueBuilderGetCommandMemorySize);
NVN_PROC(nvnQueueBuilderGetComputeMemorySize);
NVN_PROC(nvnQueueBuilderGetControlMemorySize);
NVN_PROC(nvnQueueBuilderGetCommandFlushThreshold);
NVN_PROC(nvnQueueBuilderGetMemory);
NVN_PROC(nvnQueueBuilderGetMemorySize);
NVN_PROC(nvnQueueAcquireTexture);
NVN_PROC(nvnQueueGetTotalCommandMemoryUsed);
NVN_PROC(nvnQueueGetTotalComputeMemoryUsed);
NVN_PROC(nvnQueueGetTotalControlMemoryUsed);
NVN_PROC(nvnQueueResetMemoryUsageCounts);
NVN_PROC(nvnEventInitialize);
NVN_PROC(nvnEventFinalize);
NVN_PROC(nvnEventGetValue);
NVN_PROC(nvnEventSignal);
NVN_PROC(nvnEventGetMemoryPool);
NVN_PROC(nvnEventGetMemoryOffset);
NVN_PROC(nvnSyncCreateGLSync);
NVN_PROC(nvnSyncInitializeFromFencedGLSync);
NVN_PROC(nvnSyncInitializeFromNvRmSyncNVX);
NVN_PROC(nvnSyncGetNvRmSyncNVX);
