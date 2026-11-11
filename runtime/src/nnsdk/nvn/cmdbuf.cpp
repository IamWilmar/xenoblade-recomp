// nvnCommandBuffer*: graba los comandos en una CommandList propia (commands.h). La memoria de
// comandos y de control que aporta el juego no se usa; el handle que recibe es el puntero a la lista.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"

using namespace nvn;

namespace {

// Las listas grabadas viven hasta que el juego recicla la memoria de control en la que (en NVN) estaría
// su handle: al volver a dar esa memoria con AddControlMemory, sus handles dejan de ser válidos y se
// liberan. Antes no se liberaban nunca: ~4 MB/s de fuga y Android acababa matando el juego.
struct Recorded {
    const uint8_t* control;  // memoria de control vigente al grabarla
    CommandList* list;
};
struct CommandBuffer {
    CommandList* recording = nullptr;
    const uint8_t* control = nullptr;
    size_t control_size = 0;
    std::vector<Recorded> lists;
    size_t last_size = 0;  // comandos de su última grabación: qué lista del pool le viene bien
    // La memoria de comandos que da el juego (AddCommandMemory) no se usa, pero se lleva la cuenta como si se usara: 16
    // bytes por comando, como GetCommandMemoryUsed. command_base: comandos grabados en su vida al darla.
    size_t command_size = 0, command_base = 0;
    size_t recorded_total = 0;  // comandos de todas sus grabaciones terminadas
    // SetMemoryCallback*: el juego da más memoria cuando se le acaba. Aquí nunca se acaba (lo grabado va en nuestras
    // listas), así que se guarda para los getters y no se llama.
    void* memory_callback = nullptr;
    void* memory_callback_data = nullptr;
    bool command_callback_enabled = true;
    // SetCopyRowStride / SetCopyImageStride: zancadas del buffer en las copias entre buffer y textura (0: compacto).
    int64_t copy_row_stride = 0, copy_image_stride = 0;
};
std::atomic<int64_t> g_live_lists{0};

Objects<CommandBuffer> g_command_buffers;

// --- Búsqueda del command buffer ---
// record() se llama 33-40k veces por frame y cada vez buscaba el command buffer (mutex + hash). Cada hilo
// recuerda el último que usó; g_generation cambia al inicializar o finalizar cualquiera, así que una dirección
// reutilizada nunca encuentra el objeto viejo. Interruptor sin_cache_cb (se mira en BeginRecording): buscarlo
// cada vez, como antes.
std::atomic<uint64_t> g_generation{1};
std::atomic<bool> g_cache_enabled{true};
struct LastBuffer {
    const void* key = nullptr;
    CommandBuffer* cb = nullptr;
    uint64_t generation = 0;
};
thread_local LastBuffer t_last;

CommandBuffer* find(const void* cmd) {
    if (!g_cache_enabled.load(std::memory_order_relaxed)) return g_command_buffers.get(cmd);
    const uint64_t generation = g_generation.load(std::memory_order_acquire);
    if (t_last.key == cmd && t_last.generation == generation) return t_last.cb;
    CommandBuffer* cb = g_command_buffers.get(cmd);
    t_last = {cmd, cb, generation};
    return cb;
}

CommandList* recording(void* cmd) {
    CommandBuffer* cb = find(cmd);
    return cb ? cb->recording : nullptr;
}

// --- Listas recicladas ---
// Una lista grande (~35k comandos, 11-14 MB) crecía desde vacía en cada BeginRecording: realloc por
// duplicación, fallos de página y munmap al borrarla, ~5 % de la CPU. Las que la cola ya reprodujo vuelven
// aquí (recycle_command_list) y BeginRecording reutiliza su memoria: clear() conserva la capacidad y, con
// Command trivial, no recorre los comandos. Cada command buffer pide una lista del tamaño de su última
// grabación; si aun así se llena, se pasa a una mayor del pool (grow) en vez de realojar. Se guardan como mucho
// kPoolLists listas y kPoolBytes; lo que sobra se borra al grabar, fuera de cualquier mutex de la cola.
// Interruptor sin_reciclar (se mira en BeginRecording): una lista nueva en cada grabación, que crece sola, y
// delete al soltarla, como antes.
constexpr size_t kPoolLists = 64, kPoolBytes = size_t{64} << 20;
std::mutex g_pool_mutex;
std::vector<CommandList*> g_pool;
size_t g_pool_bytes = 0;  // capacidad de las listas del pool
std::atomic<bool> g_recycle{true};

size_t capacity_bytes(const CommandList& list) { return list.commands.capacity() * sizeof(Command); }

// La lista del pool que mejor le viene a una grabación de unos `hint` comandos: la más pequeña en la que
// caben o, si no cabe en ninguna, la más grande. nullptr si el pool está vacío.
CommandList* take_list(size_t hint) {
    // ¿Le viene mejor una lista de capacidad a que una de capacidad b?
    auto better = [hint](size_t a, size_t b) {
        if ((a >= hint) != (b >= hint)) return a >= hint;  // gana la que tiene sitio
        return a >= hint ? a < b : a > b;                 // la más justa; sin sitio, la más grande
    };
    std::vector<CommandList*> excess;
    CommandList* best = nullptr;
    {
        std::lock_guard lock(g_pool_mutex);
        if (!g_pool.empty()) {
            size_t pick = 0;
            for (size_t k = 1; k < g_pool.size(); k++)
                if (better(g_pool[k]->commands.capacity(), g_pool[pick]->commands.capacity())) pick = k;
            best = g_pool[pick];
            g_pool[pick] = g_pool.back();
            g_pool.pop_back();
            g_pool_bytes -= capacity_bytes(*best);
        }
        // Lo que pase del tope: primero las más pequeñas (por número), luego las más grandes (por memoria).
        while (g_pool.size() > kPoolLists || g_pool_bytes > kPoolBytes) {
            const bool by_count = g_pool.size() > kPoolLists;
            size_t victim = 0;
            for (size_t k = 1; k < g_pool.size(); k++) {
                const size_t cap = g_pool[k]->commands.capacity(), victim_cap = g_pool[victim]->commands.capacity();
                if (by_count ? cap < victim_cap : cap > victim_cap) victim = k;
            }
            excess.push_back(g_pool[victim]);
            g_pool_bytes -= capacity_bytes(*g_pool[victim]);
            g_pool[victim] = g_pool.back();
            g_pool.pop_back();
        }
    }
    for (CommandList* list : excess) delete list;
    return best;
}

void drain_pool() {
    std::vector<CommandList*> lists;
    {
        std::lock_guard lock(g_pool_mutex);
        lists.swap(g_pool);
        g_pool_bytes = 0;
    }
    for (CommandList* list : lists) delete list;
}

CommandList* new_list(size_t hint) {
    if (!g_recycle.load(std::memory_order_relaxed)) {
        drain_pool();  // lo que volviera mientras se apagaba
        return new CommandList;
    }
    if (CommandList* list = take_list(hint)) {
        list->commands.clear();
        list->spilled.clear();
        list->reset_inline();
        return list;
    }
    return new CommandList;  // pool vacío (al arrancar): crece sola
}

// La lista que se graba está llena: se pasa a una mayor del pool (su memoria ya está en uso: sin fallos de
// página ni munmap) copiando lo grabado, que es trivial de copiar. Si no hay ninguna mayor, el vector crece
// como siempre. La lista (el handle) es la misma: solo cambia su memoria.
void grow(CommandList& list) {
    if (!g_recycle.load(std::memory_order_relaxed)) return;
    const size_t size = list.commands.size();
    CommandList* bigger = take_list(2 * size + 64);
    if (!bigger) return;
    if (bigger->commands.capacity() > size) {
        bigger->commands.clear();
        bigger->commands.insert(bigger->commands.end(), list.commands.begin(), list.commands.end());
        std::swap(list.commands, bigger->commands);  // `bigger` se queda la memoria pequeña
    }
    bigger->spilled.clear();  // lo de su grabación anterior (la de `list` sigue en `list`)
    recycle_command_list(bigger);
}

// Se construye directamente en la lista (sin un Command temporal de 344 bytes que luego se mueve).
template <typename T>
void add(CommandList& list, T&& c) {
    if (list.commands.size() == list.commands.capacity()) [[unlikely]]
        grow(list);
    list.commands.emplace_back(std::in_place_type<std::decay_t<T>>, std::forward<T>(c));
}
template <typename T>
void record(void* cmd, T&& c) {
    if (CommandList* list = recording(cmd)) add(*list, std::forward<T>(c));
}

// Copia count elementos a la SmallVector del comando (lo que no quepa, a memoria de la lista).
using RangeList = decltype(cmd::BindUniformBuffers::ranges);
static_assert(std::is_same_v<RangeList, decltype(cmd::BindVertexBuffers::ranges)>);
using AttribList = decltype(cmd::BindVertexAttribs::attribs);
using StreamList = decltype(cmd::BindVertexStreams::streams);
template <typename V, typename T>
V items(CommandList& list, const T* p, int count) {
    const size_t n = count > 0 && p ? static_cast<size_t>(count) : 0;
    static int logged = 0;  // DEPURACIÓN: si salen a menudo, la capacidad de dentro se queda corta
    if (n > V::inline_capacity() && logged < 5) {
        logged++;
        NVN_LOG("lista de %zu elementos de %zu bytes fuera del comando (caben %zu)\n", n, sizeof(T),
                V::inline_capacity());
    }
    return V(p, n, [&](size_t bytes) { return list.spill(bytes); });
}

TextureViewDesc view_of(const void* view) {
    return view ? *static_cast<const TextureViewDesc*>(view) : TextureViewDesc{};
}

// Borrado de una región de una textura: el color son 4 valores de 32 bits (float, int o uint) que se guardan tal cual.
void clear_texture(void* cmd, const void* texture, const void* view, const CopyRegion* region, const void* color, int mask) {
    if (!texture || !region || !color) return;
    cmd::ClearTexture c{};
    c.texture = g_textures.get(texture);
    c.view = view_of(view);
    c.region = *region;
    memcpy(c.color, color, sizeof c.color);
    c.mask = mask;
    if (c.texture) record(cmd, c);
}

}  // namespace

void nvn::recycle_command_list(CommandList* list) {
    if (!list) return;
    if (!g_recycle.load(std::memory_order_relaxed) || list->commands.capacity() == 0) {  // sin memoria: no sirve
        delete list;
        return;
    }
    std::lock_guard lock(g_pool_mutex);
    g_pool.push_back(list);
    g_pool_bytes += capacity_bytes(*list);
}

extern "C" {

nvn::Boolean nvnCommandBufferInitialize(void* cmd, void*) {
    g_command_buffers.create(cmd);
    g_generation.fetch_add(1, std::memory_order_acq_rel);  // tras crearlo: lo que se recordara de esta dirección ya no vale
    return 1;
}
void nvnCommandBufferFinalize(void* cmd) {
    if (CommandBuffer* cb = g_command_buffers.get(cmd)) {
        for (Recorded& r : cb->lists) release_command_list(r.list);
        g_live_lists -= static_cast<int64_t>(cb->lists.size());
        recycle_command_list(cb->recording);  // a medio grabar: nunca llegó a la cola
        cb->recording = nullptr;
    }
    g_command_buffers.destroy(cmd);
    g_generation.fetch_add(1, std::memory_order_acq_rel);  // tras destruirlo: ninguna búsqueda posterior lo recuerda
}
// Un trozo nuevo de memoria de comandos: lo usado se cuenta desde aquí (ver CommandBuffer::command_size).
void nvnCommandBufferAddCommandMemory(void* cmd, void*, int64_t, size_t size) {
    CommandBuffer* cb = g_command_buffers.get(cmd);
    if (!cb) return;
    cb->command_size = size;
    cb->command_base = cb->recorded_total + (cb->recording ? cb->recording->commands.size() : 0);
}
void nvnCommandBufferAddControlMemory(void* cmd, void* memory, size_t size) {
    CommandBuffer* cb = g_command_buffers.get(cmd);
    if (!cb) return;
    const auto* begin = static_cast<const uint8_t*>(memory);
    size_t freed = 0;
    for (size_t k = 0; k < cb->lists.size();) {
        const uint8_t* c = cb->lists[k].control;
        if (c && c >= begin && c < begin + size) {
            release_command_list(cb->lists[k].list);
            cb->lists[k] = cb->lists.back();
            cb->lists.pop_back();
            freed++;
        } else {
            k++;
        }
    }
    g_live_lists -= static_cast<int64_t>(freed);
    cb->control = begin;
    cb->control_size = size;
}
// Lo que queda del último trozo de memoria de comandos y de control, con la misma cuenta que Get*MemoryUsed (16 bytes por
// comando, 64 por lista). Nada se graba en esa memoria, así que quedarse sin ella no para la grabación.
size_t nvnCommandBufferGetCommandMemoryFree(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    if (!cb) return 0;
    const size_t total = cb->recorded_total + (cb->recording ? cb->recording->commands.size() : 0);
    const size_t used = (total - std::min(cb->command_base, total)) * 16;
    return cb->command_size > used ? cb->command_size - used : 0;
}
size_t nvnCommandBufferGetControlMemoryFree(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    if (!cb) return 0;
    size_t lists = cb->recording ? 1 : 0;
    for (const Recorded& r : cb->lists) lists += r.control == cb->control;
    const size_t used = lists * 64;
    return cb->control_size > used ? cb->control_size - used : 0;
}
size_t nvnCommandBufferGetCommandMemorySize(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    return cb ? cb->command_size : 0;
}
size_t nvnCommandBufferGetControlMemorySize(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    return cb ? cb->control_size : 0;
}
// Memoria usada desde que el juego dio la de control: 16 bytes de comandos por comando grabado y 64 de control por
// lista, 0 si no hay nada grabado. Xenoblade (grlib::CGLibCmdBuffer::end, main+0x70b220) guarda las dos tras
// EndRecording y solo llama (CallCommands) o envía la lista si alguna no es 0: con 0 fijo no dibujaba nada (pantalla
// negra a 30 fps con audio).
size_t nvnCommandBufferGetCommandMemoryUsed(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    if (!cb) return 0;
    size_t n = cb->recording ? cb->recording->commands.size() : 0;
    for (const Recorded& r : cb->lists) n += r.list->commands.size();
    return n * 16;
}
size_t nvnCommandBufferGetControlMemoryUsed(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    if (!cb || nvnCommandBufferGetCommandMemoryUsed(cmd) == 0) return 0;
    return (cb->lists.size() + (cb->recording ? 1 : 0)) * 64;
}
void nvnCommandBufferSetMemoryCallback(void* cmd, void* callback) {
    if (CommandBuffer* cb = g_command_buffers.get(cmd)) cb->memory_callback = callback;
}
void nvnCommandBufferSetMemoryCallbackData(void* cmd, void* data) {
    if (CommandBuffer* cb = g_command_buffers.get(cmd)) cb->memory_callback_data = data;
}
void* nvnCommandBufferGetMemoryCallback(const void* cmd) {
    const CommandBuffer* cb = g_command_buffers.get(cmd);
    return cb ? cb->memory_callback : nullptr;
}
void* nvnCommandBufferGetMemoryCallbackData(const void* cmd) {
    const CommandBuffer* cb = g_command_buffers.get(cmd);
    return cb ? cb->memory_callback_data : nullptr;
}
void nvnCommandBufferSetCommandMemoryCallbackEnabled(void* cmd, nvn::Boolean enabled) {
    if (CommandBuffer* cb = g_command_buffers.get(cmd)) cb->command_callback_enabled = enabled != 0;
}

void nvnCommandBufferBeginRecording(void* cmd) {
    // Interruptores: una vez por grabación, no por comando.
    const bool recycle = !NVN_FLAG("sin_reciclar");
    g_recycle.store(recycle, std::memory_order_relaxed);
    g_cache_enabled.store(!NVN_FLAG("sin_cache_cb"), std::memory_order_relaxed);
    if (CommandBuffer* cb = find(cmd)) {
        recycle_command_list(cb->recording);  // una grabación sin EndRecording: nunca llegó a la cola
        cb->recording = new_list(cb->last_size);
    }
}
uint64_t nvnCommandBufferEndRecording(void* cmd) {
    CommandBuffer* cb = find(cmd);
    if (!cb || !cb->recording) return 0;
    CommandList* list = cb->recording;
    cb->recording = nullptr;
    cb->last_size = list->commands.size();
    cb->recorded_total += list->commands.size();
    cb->lists.push_back({cb->control, list});
    int64_t live = ++g_live_lists;
    static int64_t next_report = 1000;
    if (live >= next_report) {  // DEPURACIÓN: si esto crece sin parar, hay listas que no se liberan
        NVN_LOG("%lld listas de comandos vivas\n", static_cast<long long>(live));
        next_report *= 2;
    }
    return reinterpret_cast<uint64_t>(list);
}

// --- Render targets y borrados ---

void nvnCommandBufferSetRenderTargets(void* cmd, int count, void* const* colors, const void* const* color_views,
                                      void* depth, const void* depth_view) {
    cmd::SetRenderTargets c{};
    c.count = count > 8 ? 8 : count;
    for (int k = 0; k < c.count; k++) {
        c.colors[k] = colors && colors[k] ? g_textures.get(colors[k]) : nullptr;
        c.color_views[k] = view_of(color_views ? color_views[k] : nullptr);
    }
    c.depth = depth ? g_textures.get(depth) : nullptr;
    c.depth_view = view_of(depth_view);
    for (int k = 0; k < c.count; k++) {
        const Texture* t = c.colors[k];
        if (t && t->desc.flags == nvn::kTextureFlagImage && t->desc.format == 0x25 && t->desc.levels == 1 &&
            t->desc.target == nvn::kTarget2D) {
            if (CommandList* list = recording(cmd)) list->cpu_capture = true;
            static int logged = 0;
            if (nvn::first_times(logged, 8) || NVN_FLAG("diag_miniatura"))
                NVN_LOG("captura grabada: render target %dx%d flags 0x%x storage %p\n", t->desc.width, t->desc.height,
                        t->desc.flags, static_cast<const void*>(t->storage));
        }
    }
    static int logged = 0;  // DEPURACIÓN: ¿se pierde algún target de un MRT?
    if (count > 1 && logged < 20) {
        logged++;
        std::string s;
        for (int k = 0; k < count && k < 8; k++) {
            char b[64];
            snprintf(b, sizeof b, " %p->%s", colors ? colors[k] : nullptr, c.colors[k] ? "ok" : "NULO");
            s += b;
        }
        NVN_LOG("SetRenderTargets %d:%s depth %p->%s\n", count, s.c_str(), depth, c.depth ? "ok" : "NULO");
    }
    record(cmd, c);
}
void nvnCommandBufferClearColor(void* cmd, int index, const float* color, int mask) {
    cmd::ClearColor c{index, {}, mask};
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
// Borrados de targets enteros: los 4 valores de 32 bits van tal cual en el float[4] del comando y el ejecutor los copia
// sin convertir a VkClearColorValue (que es una unión: int32/uint32 leen los mismos bits).
void nvnCommandBufferClearColori(void* cmd, int index, const int* color, int mask) {
    cmd::ClearColor c{index, {}, mask};
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
void nvnCommandBufferClearColorui(void* cmd, int index, const uint32_t* color, int mask) {
    cmd::ClearColor c{index, {}, mask};
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
void nvnCommandBufferClearDepthStencil(void* cmd, float depth, nvn::Boolean depth_mask, int stencil, int mask) {
    record(cmd, cmd::ClearDepthStencil{depth, depth_mask != 0, stencil, mask});
}

// --- Viewport, scissor, profundidad, stencil ---

void nvnCommandBufferSetViewport(void* cmd, int x, int y, int w, int h) {
    record(cmd, cmd::Viewport{0, float(x), float(y), float(w), float(h)});
}
void nvnCommandBufferSetViewports(void* cmd, int first, int count, const float* rects) {
    for (int k = 0; k < count; k++)
        record(cmd, cmd::Viewport{first + k, rects[4 * k], rects[4 * k + 1], rects[4 * k + 2], rects[4 * k + 3]});
}
void nvnCommandBufferSetScissor(void* cmd, int x, int y, int w, int h) { record(cmd, cmd::Scissor{0, x, y, w, h}); }
void nvnCommandBufferSetScissors(void* cmd, int first, int count, const int* rects) {
    for (int k = 0; k < count; k++)
        record(cmd, cmd::Scissor{first + k, rects[4 * k], rects[4 * k + 1], rects[4 * k + 2], rects[4 * k + 3]});
}
void nvnCommandBufferSetDepthRange(void* cmd, float n, float f) { record(cmd, cmd::DepthRange{n, f}); }
// Un viewport: solo cuenta el 0 (como SetViewports).
void nvnCommandBufferSetDepthRanges(void* cmd, int first, int count, const float* ranges) {
    if (first == 0 && count > 0 && ranges) record(cmd, cmd::DepthRange{ranges[0], ranges[1]});
}
// Reinicio de primitiva (MH Rise lo apaga al empezar cada lista). Vulkan solo admite el índice de todo unos.
void nvnCommandBufferSetPrimitiveRestart(void* cmd, nvn::Boolean enable, int index) {
    record(cmd, cmd::PrimitiveRestart{enable != 0, static_cast<uint32_t>(index)});
}
// Rasterización conservadora: el Adreno no la tiene en Vulkan. MH Rise la apaga al empezar cada lista; si alguna vez
// la enciende, se dibuja normal (se avisa).
void nvnCommandBufferSetConservativeRasterEnable(void*, nvn::Boolean enable) {
    static bool logged = false;
    if (enable && !logged) {
        logged = true;
        NVN_LOG("rasterización conservadora pedida: sin soporte, se dibuja normal\n");
    }
}
// La dilatación de la rasterización conservadora: sin rasterización conservadora (arriba), no hay qué dilatar.
void nvnCommandBufferSetConservativeRasterDilate(void*, float) {}
// Optimizaciones internas de Maxwell sin equivalente en Vulkan (el driver del Adreno decide solo): la caché por tiles y su
// tamaño, el descarte por stencil del ZCull, la precisión subpíxel del rasterizador, la memoria de scratch de los shaders
// (Vulkan la reserva sola) y guardar/restaurar el ZCull (el Adreno no tiene ZCull que guardar).
void nvnCommandBufferSetTiledCacheAction(void*, int) {}
void nvnCommandBufferSetTiledCacheTileSize(void*, int, int) {}
void nvnCommandBufferSetStencilCullCriteria(void*, int, int, int) {}
void nvnCommandBufferSetSubpixelPrecisionBias(void*, int, int) {}
void nvnCommandBufferSetShaderScratchMemory(void*, const void*, int64_t, size_t) {}
void nvnCommandBufferSaveZCullData(void*, uint64_t, size_t) {}
void nvnCommandBufferRestoreZCullData(void*, uint64_t, size_t) {}

// Estado de rasterización que en Vulkan va dentro de la pipeline. El ejecutor no lo emula: haría falta un campo más en
// PipelineKey (y cambiar el formato de pipelines_estado.bin y de la caché de pipelines), y el ancho de línea y los límites
// de profundidad, además, características que no se piden al crear el dispositivo (wideLines, depthBounds). Lo de por
// defecto es lo que ya hace; si el juego pide otra cosa se avisa una vez por función.
static void warn_raster_state(std::atomic<int>& logged, const char* what) {
    if (nvn::first_times(logged, 1)) NVN_LOG("%s pedido: el ejecutor no lo emula\n", what);
}
void nvnCommandBufferSetPointSize(void*, float size) {
    static std::atomic<int> logged{0};
    if (size != 1.0f) warn_raster_state(logged, "SetPointSize distinto de 1");
}
void nvnCommandBufferSetLineWidth(void*, float width) {
    static std::atomic<int> logged{0};
    if (width != 1.0f) warn_raster_state(logged, "SetLineWidth distinto de 1");
}
void nvnCommandBufferSetDepthBounds(void*, nvn::Boolean enable, float, float) {
    static std::atomic<int> logged{0};
    if (enable) warn_raster_state(logged, "SetDepthBounds");
}
void nvnCommandBufferSetDepthClamp(void*, nvn::Boolean enable) {
    static std::atomic<int> logged{0};
    if (enable) warn_raster_state(logged, "SetDepthClamp");
}
// Factor, unidades y límite del desplazamiento de polígonos (los que activa PolygonState::offset_enables).
void nvnCommandBufferSetPolygonOffsetClamp(void*, float factor, float units, float clamp) {
    static std::atomic<int> logged{0};
    if (factor != 0.0f || units != 0.0f || clamp != 0.0f) warn_raster_state(logged, "SetPolygonOffsetClamp");
}
// Referencia del alpha test de ColorState: Vulkan no tiene alpha test fijo y el traductor no lo añade a los shaders.
void nvnCommandBufferSetAlphaRef(void*, float) {}
// Teselación: los niveles por defecto (sin tess control shader) y los vértices por parche (patchControlPoints de la
// pipeline). Por defecto 3 vértices por parche, como en OpenGL.
void nvnCommandBufferSetInnerTessellationLevels(void*, const float*) {
    static std::atomic<int> logged{0};
    warn_raster_state(logged, "SetInnerTessellationLevels");
}
void nvnCommandBufferSetOuterTessellationLevels(void*, const float*) {
    static std::atomic<int> logged{0};
    warn_raster_state(logged, "SetOuterTessellationLevels");
}
void nvnCommandBufferSetPatchSize(void*, int size) {
    static std::atomic<int> logged{0};
    if (size != 3) warn_raster_state(logged, "SetPatchSize distinto de 3");
}
// Subrutinas de GLSL: el traductor no las tiene (ver nvnProgramSetSubroutineLinkage).
void nvnCommandBufferSetProgramSubroutines(void*, void*, int, int, int count, const int*) {
    static std::atomic<int> logged{0};
    if (count > 0) warn_raster_state(logged, "SetProgramSubroutines");
}
// Tabla de modulación de cobertura (muestras mixtas de NVIDIA, VK_NV_framebuffer_mixed_samples): el Adreno no la tiene.
void nvnCommandBufferBindCoverageModulationTable(void*, const float*) {
    static std::atomic<int> logged{0};
    warn_raster_state(logged, "BindCoverageModulationTable");
}
// Las muestras que escribe cada fragmento. El ejecutor no tiene pSampleMask en PipelineKey: una máscara que no deja ninguna
// muestra del pase se dibuja con el rasterizador descartado (el vertex shader corre, no sale ningún fragmento); una
// parcial con multisample se avisa.
void nvnCommandBufferSetSampleMask(void* cmd, int mask) { record(cmd, cmd::SampleMask{static_cast<uint32_t>(mask)}); }
void nvnCommandBufferSetStencilRef(void* cmd, int faces, int ref) {
    record(cmd, cmd::Stencil{cmd::Stencil::Ref, faces, ref});
}
void nvnCommandBufferSetStencilMask(void* cmd, int faces, int mask) {
    record(cmd, cmd::Stencil{cmd::Stencil::Mask, faces, mask});
}
void nvnCommandBufferSetStencilValueMask(void* cmd, int faces, int mask) {
    record(cmd, cmd::Stencil{cmd::Stencil::ValueMask, faces, mask});
}

// --- Objetos de estado (se copian: el juego puede reutilizar el struct) ---

void nvnCommandBufferBindBlendState(void* cmd, const BlendState* s) { record(cmd, *s); }
void nvnCommandBufferBindColorState(void* cmd, const ColorState* s) { record(cmd, *s); }
void nvnCommandBufferBindChannelMaskState(void* cmd, const ChannelMaskState* s) { record(cmd, *s); }
void nvnCommandBufferBindDepthStencilState(void* cmd, const DepthStencilState* s) { record(cmd, *s); }
void nvnCommandBufferBindPolygonState(void* cmd, const PolygonState* s) { record(cmd, *s); }
void nvnCommandBufferBindMultisampleState(void* cmd, const MultisampleState* s) { record(cmd, *s); }
void nvnCommandBufferBindVertexAttribState(void* cmd, int count, const VertexAttribState* attribs) {
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexAttribs{items<AttribList>(*list, attribs, count)});
}
void nvnCommandBufferBindVertexStreamState(void* cmd, int count, const VertexStreamState* streams) {
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexStreams{items<StreamList>(*list, streams, count)});
}

// --- Programas, buffers, texturas ---

void nvnCommandBufferBindProgram(void* cmd, const void* program, int stages) {
    record(cmd, cmd::BindProgram{program ? find_program(program) : nullptr, stages});
}
void nvnCommandBufferBindVertexBuffers(void* cmd, int first, int count, const BufferRange* ranges) {
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindVertexBuffers{first, items<RangeList>(*list, ranges, count)});
}
void nvnCommandBufferBindVertexBuffer(void* cmd, int index, uint64_t address, size_t size) {
    const BufferRange range{address, size};
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexBuffers{index, items<RangeList>(*list, &range, 1)});
}
void nvnCommandBufferBindUniformBuffers(void* cmd, int stage, int first, int count, const BufferRange* ranges) {
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindUniformBuffers{stage, first, items<RangeList>(*list, ranges, count)});
}
void nvnCommandBufferBindUniformBuffer(void* cmd, int stage, int index, uint64_t address, size_t size) {
    const BufferRange range{address, size};
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindUniformBuffers{stage, index, items<RangeList>(*list, &range, 1)});
}
void nvnCommandBufferBindStorageBuffer(void* cmd, int stage, int index, uint64_t address, size_t size) {
    record(cmd, cmd::BindStorageBuffer{stage, index, {address, size}});
}
void nvnCommandBufferBindStorageBuffers(void* cmd, int stage, int first, int count, const BufferRange* ranges) {
    for (int k = 0; k < count; k++) record(cmd, cmd::BindStorageBuffer{stage, first + k, ranges[k]});
}
void nvnCommandBufferBindSeparateTexture(void* cmd, int stage, int index, uint64_t handle) {
    record(cmd, cmd::BindTexture{stage, index, handle, false});
}
void nvnCommandBufferBindTexture(void* cmd, int stage, int index, uint64_t handle) {
    static int logged = 0;  // DEPURACIÓN: ¿índices fuera de 0..31?
    if (index >= 32 && nvn::first_times(logged, 20)) NVN_LOG("BindTexture etapa %d índice %d\n", stage, index);
    record(cmd, cmd::BindTexture{stage, index, handle, true});
}
void nvnCommandBufferBindSeparateSampler(void* cmd, int stage, int index, uint64_t handle) {
    record(cmd, cmd::BindSampler{stage, index, handle});
}
void nvnCommandBufferBindTextures(void* cmd, int stage, int first, int count, const uint64_t* handles) {
    for (int k = 0; k < count; k++) nvnCommandBufferBindTexture(cmd, stage, first + k, handles[k]);
}
void nvnCommandBufferBindSeparateTextures(void* cmd, int stage, int first, int count, const uint64_t* handles) {
    for (int k = 0; k < count; k++) nvnCommandBufferBindSeparateTexture(cmd, stage, first + k, handles[k]);
}
void nvnCommandBufferBindSeparateSamplers(void* cmd, int stage, int first, int count, const uint64_t* handles) {
    for (int k = 0; k < count; k++) nvnCommandBufferBindSeparateSampler(cmd, stage, first + k, handles[k]);
}
void nvnCommandBufferBindImage(void* cmd, int stage, int index, uint64_t handle) {
    record(cmd, cmd::BindImage{stage, index, handle});
}
void nvnCommandBufferBindImages(void* cmd, int stage, int first, int count, const uint64_t* handles) {
    for (int k = 0; k < count && handles; k++) record(cmd, cmd::BindImage{stage, first + k, handles[k]});
}
void nvnCommandBufferSetTexturePool(void* cmd, const void* pool) {
    record(cmd, cmd::SetPools{g_texture_pools.get(pool), nullptr});
}
void nvnCommandBufferSetSamplerPool(void* cmd, const void* pool) {
    record(cmd, cmd::SetPools{nullptr, g_sampler_pools.get(pool)});
}

// --- Draws y compute ---

void nvnCommandBufferDrawArrays(void* cmd, int primitive, int first, int count) {
    record(cmd, cmd::Draw{primitive, -1, count, first, 0, 0, 0, 1});
}
void nvnCommandBufferDrawElements(void* cmd, int primitive, int type, int count, uint64_t indices) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, 0, 0, 1});
}
void nvnCommandBufferDrawElementsBaseVertex(void* cmd, int primitive, int type, int count, uint64_t indices,
                                            int base_vertex) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, base_vertex, 0, 1});
}
void nvnCommandBufferDrawArraysInstanced(void* cmd, int primitive, int first, int count, int base_instance,
                                         int instances) {
    record(cmd, cmd::Draw{primitive, -1, count, first, 0, 0, base_instance, instances});
}
void nvnCommandBufferDrawElementsInstanced(void* cmd, int primitive, int type, int count, uint64_t indices,
                                           int base_vertex, int base_instance, int instances) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, base_vertex, base_instance, instances});
}
void nvnCommandBufferDispatchCompute(void* cmd, int x, int y, int z) { record(cmd, cmd::Dispatch{x, y, z}); }
// Indirectos: los parámetros los escribe a menudo la propia GPU (un compute anterior), así que se leen
// de memoria al ejecutar, no al grabar.
void nvnCommandBufferDispatchComputeIndirect(void* cmd, uint64_t indirect) {
    record(cmd, cmd::Dispatch{0, 0, 0, indirect});
}
void nvnCommandBufferDrawArraysIndirect(void* cmd, int primitive, uint64_t indirect) {
    record(cmd, cmd::Draw{primitive, -1, 0, 0, 0, 0, 0, 1, indirect});
}
void nvnCommandBufferDrawElementsIndirect(void* cmd, int primitive, int type, uint64_t indices, uint64_t indirect) {
    record(cmd, cmd::Draw{primitive, type, 0, 0, indices, 0, 0, 1, indirect});
}
// Multi-draw: hasta max_draws draws indirectos, cada uno a stride bytes del anterior, y su número en count. En MH Rise
// los escribe el compute que descarta lo que no se ve (el mundo entero se dibuja así).
void nvnCommandBufferMultiDrawElementsIndirectCount(void* cmd, int primitive, int type, uint64_t indices, uint64_t indirect,
                                                    uint64_t count, int max_draws, ptrdiff_t stride) {
    cmd::Draw d{primitive, type, 0, 0, indices, 0, 0, 1, indirect};
    d.draw_count = count;
    d.max_draws = max_draws;
    d.stride = static_cast<int>(stride);
    if (count && indirect && max_draws > 0) record(cmd, d);
}
void nvnCommandBufferMultiDrawArraysIndirectCount(void* cmd, int primitive, uint64_t indirect, uint64_t count, int max_draws,
                                                  ptrdiff_t stride) {
    cmd::Draw d{primitive, -1, 0, 0, 0, 0, 0, 1, indirect};
    d.draw_count = count;
    d.max_draws = max_draws;
    d.stride = static_cast<int>(stride);
    if (count && indirect && max_draws > 0) record(cmd, d);
}

// --- Copias, barreras, contadores ---

void nvnCommandBufferCopyBufferToBuffer(void* cmd, uint64_t src, uint64_t dst, size_t size, int) {
    record(cmd, cmd::CopyBuffer{src, dst, size});
}
void nvnCommandBufferCopyTextureToTexture(void* cmd, const void* src, const void* src_view, const CopyRegion* src_region,
                                          const void* dst, const void* dst_view, const CopyRegion* dst_region, int) {
    record(cmd, cmd::CopyTexture{g_textures.get(src), view_of(src_view), *src_region, g_textures.get(dst),
                                 view_of(dst_view), *dst_region});
}
void nvnCommandBufferCopyBufferToTexture(void* cmd, uint64_t src, const void* dst, const void* dst_view,
                                         const CopyRegion* region, int) {
    const CommandBuffer* cb = find(cmd);
    record(cmd, cmd::CopyBufferToTexture{src, g_textures.get(dst), view_of(dst_view), *region, cb ? cb->copy_row_stride : 0,
                                         cb ? cb->copy_image_stride : 0});
}
void nvnCommandBufferCopyTextureToBuffer(void* cmd, const void* src, const void* src_view, const CopyRegion* region,
                                         uint64_t dst, int) {
    const CommandBuffer* cb = find(cmd);
    Texture* t = g_textures.get(src);
    if (!t || !region || !dst) return;
    static int logged = 0;  // DEPURACIÓN (miniatura de la partida): quién copia texturas a buffers
    if (nvn::first_times(logged, 8) || NVN_FLAG("diag_miniatura"))
        NVN_LOG("CopyTextureToBuffer grabado: textura %dx%d formato 0x%x flags 0x%x región %d,%d %dx%dx%d -> %#llx zancadas %lld/%lld\n",
                t->desc.width, t->desc.height, t->desc.format, t->desc.flags, region->x, region->y, region->width,
                region->height, region->depth, static_cast<unsigned long long>(dst),
                static_cast<long long>(cb ? cb->copy_row_stride : 0), static_cast<long long>(cb ? cb->copy_image_stride : 0));
    record(cmd, cmd::CopyTextureToBuffer{t, view_of(src_view), *region, dst, cb ? cb->copy_row_stride : 0,
                                         cb ? cb->copy_image_stride : 0});
}
// Zancadas del buffer para las copias que se graben después (0: filas y capas compactas).
void nvnCommandBufferSetCopyRowStride(void* cmd, ptrdiff_t stride) {
    if (CommandBuffer* cb = find(cmd)) cb->copy_row_stride = stride;
}
void nvnCommandBufferSetCopyImageStride(void* cmd, ptrdiff_t stride) {
    if (CommandBuffer* cb = find(cmd)) cb->copy_image_stride = stride;
}
ptrdiff_t nvnCommandBufferGetCopyRowStride(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    return cb ? cb->copy_row_stride : 0;
}
ptrdiff_t nvnCommandBufferGetCopyImageStride(const void* cmd) {
    const CommandBuffer* cb = find(cmd);
    return cb ? cb->copy_image_stride : 0;
}
// Downsample: resolver un multisample (o reducir) src en dst. Los Tiled* hacen lo mismo con la caché por tiles de Maxwell.
static void downsample(void* cmd, const void* src, const void* src_view, const void* dst, const void* dst_view) {
    Texture* s = g_textures.get(src);
    Texture* d = g_textures.get(dst);
    if (s && d) record(cmd, cmd::Downsample{s, view_of(src_view), d, view_of(dst_view)});
}
void nvnCommandBufferDownsample(void* cmd, const void* src, const void* dst) { downsample(cmd, src, nullptr, dst, nullptr); }
void nvnCommandBufferDownsampleTextureView(void* cmd, const void* src, const void* src_view, const void* dst,
                                           const void* dst_view) {
    downsample(cmd, src, src_view, dst, dst_view);
}
void nvnCommandBufferTiledDownsample(void* cmd, const void* src, const void* dst) { downsample(cmd, src, nullptr, dst, nullptr); }
void nvnCommandBufferTiledDownsampleTextureView(void* cmd, const void* src, const void* src_view, const void* dst,
                                                const void* dst_view) {
    downsample(cmd, src, src_view, dst, dst_view);
}
// Descartar un target: su contenido queda indefinido. Un borrado es un contenido válido de "indefinido" y le dice al
// ejecutor que no hace falta cargarlo (como en el port de MHGU).
void nvnCommandBufferDiscardColor(void* cmd, int index) { record(cmd, cmd::ClearColor{index, {0, 0, 0, 0}, 0xF}); }
void nvnCommandBufferDiscardDepthStencil(void* cmd) { record(cmd, cmd::ClearDepthStencil{1.0f, true, 0, 0xFF}); }
// DrawTexture (NV_draw_texture: un rectángulo con una textura, sin programa): el dispositivo dice que no lo tiene
// (GetInteger SUPPORTS_DRAW_TEXTURE = 0) y el ejecutor no tiene una pipeline para eso. Se avisa.
void nvnCommandBufferDrawTexture(void*, uint64_t, const void*, const void*) {
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 4)) NVN_LOG("DrawTexture sin soporte: no se dibuja\n");
}
void nvnCommandBufferBarrier(void* cmd, int bits) { record(cmd, cmd::Barrier{bits}); }
void nvnCommandBufferSetBlendColor(void* cmd, const float* color) {
    cmd::BlendColor c;
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
void nvnCommandBufferSetRenderEnable(void* cmd, nvn::Boolean enable) { record(cmd, cmd::RenderEnable{enable != 0}); }
// Descarte de rasterización: no sale ningún fragmento. Va como un RenderEnable con modo kRasterizerDiscard (estado
// aparte en el ejecutor: no toca el render enable). Xenoblade lo pone alrededor de su transform feedback.
void nvnCommandBufferSetRasterizerDiscard(void* cmd, nvn::Boolean discard) {
    record(cmd, cmd::RenderEnable{discard != 0, cmd::kRasterizerDiscard});
}
// Condicional a un informe de contador (oclusión): se dibuja siempre (los contadores de muestras ya miden de verdad,
// ver ResetCounter, pero el condicional no los mira todavía).
void nvnCommandBufferSetRenderEnableConditional(void* cmd, int mode, uint64_t counters) {
    record(cmd, cmd::RenderEnable{true, mode, counters});
}
void nvnCommandBufferReportValue(void* cmd, uint32_t value, uint64_t address) {
    record(cmd, cmd::ReportValue{value, address});
}
// Pone a cero un contador de la GPU. Xenoblade lo hace con SAMPLES_PASSED antes de cada medida (el histograma de la
// autoexposición, las consultas de oclusión, nubes, IBL): el ejecutor empieza ahí una consulta de oclusión de Vulkan.
void nvnCommandBufferResetCounter(void* cmd, int type) {
    cmd::ReportCounter r{type, 0};
    r.reset = true;
    record(cmd, std::move(r));
}

// Transform feedback: el Adreno no tiene VK_EXT_transform_feedback. Se emula: los vertex shaders de captura (escriben
// genéricos y no la posición) guardan además sus salidas en un SSBO (nvnshader::kXfbBank) que el ejecutor enlaza al
// buffer de captura 0 mientras está activo. Xenoblade deforma así con el viento la hierba alta del campo (draws de puntos
// con el rasterizador descartado) y la dibuja leyendo lo capturado como stream de vértices; sin captura leía lo último
// que dejó el skinning en ese buffer (matrices) y salían pinchos negros. DrawTransformFeedback sigue sin hacer nada.
static void warn_transform_feedback(const char* what) {
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 8) NVN_LOG("transform feedback sin soporte: %s\n", what);
}
void nvnCommandBufferBindTransformFeedbackBuffers(void* cmd, int first, int count, const BufferRange* ranges) {
    for (int k = 0; k < count; k++)
        record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::Bind, first + k, ranges[k]});
}
void nvnCommandBufferBindTransformFeedbackBuffer(void* cmd, int index, uint64_t address, size_t size) {
    record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::Bind, index, {address, size}});
}
void nvnCommandBufferBeginTransformFeedback(void* cmd, uint64_t) {
    record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::Begin, 0, {}});
}
void nvnCommandBufferEndTransformFeedback(void* cmd, uint64_t) {
    record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::End, 0, {}});
}
void nvnCommandBufferPauseTransformFeedback(void* cmd, uint64_t) {
    record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::Pause, 0, {}});
}
void nvnCommandBufferResumeTransformFeedback(void* cmd, uint64_t) {
    record(cmd, cmd::TransformFeedback{cmd::TransformFeedback::Resume, 0, {}});
}
void nvnCommandBufferDrawTransformFeedback(void*, int, uint64_t) { warn_transform_feedback("DrawTransformFeedback"); }

// Swizzle de viewport (NV_viewport_swizzle, para dibujar las caras de un cubo de una vez): Vulkan no lo tiene en el
// Adreno. La identidad (+X, +Y, +Z, +W = 0, 2, 4, 6) no cambia nada; otra se avisa.
void nvnCommandBufferSetViewportSwizzles(void*, int first, int count, const int* swizzles) {
    static std::atomic<int> logged{0};
    for (int k = 0; k < count; k++) {
        const int* s = swizzles + 4 * k;
        if ((s[0] != 0 || s[1] != 2 || s[2] != 4 || s[3] != 6) && logged.fetch_add(1) < 8)
            NVN_LOG("swizzle de viewport %d sin soporte: %d %d %d %d\n", first + k, s[0], s[1], s[2], s[3]);
    }
}

// Resolver la profundidad multisample del target: sin MSAA en el port, no hay nada que resolver.
void nvnCommandBufferResolveDepthBuffer(void*) {
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 4) NVN_LOG("ResolveDepthBuffer sin efecto\n");
}
// Compresión de color por reducción (optimización interna de Maxwell): nada que hacer.
void nvnCommandBufferSetColorReductionEnable(void*, nvn::Boolean) {}
void nvnCommandBufferSetColorReductionThresholds(void*, int, const void*) {}

// CallCommands: las listas ya grabadas (sus handles son CommandList*) se copian en la que se graba. Los datos en línea
// y lo que no cupo en los comandos siguen siendo de la lista llamada: el juego la mantiene viva mientras la use (la
// misma regla que en NVN para la memoria de comandos de un handle llamado).
void nvnCommandBufferCallCommands(void* cmd, int count, const uint64_t* handles) {
    CommandList* list = recording(cmd);
    if (!list || !handles) return;
    for (int k = 0; k < count; k++) {
        const auto* called = reinterpret_cast<const CommandList*>(handles[k]);
        if (!called) continue;
        if (called->cpu_capture) {  // la lista llamada pinta la textura de captura: la que la llama, también
            list->cpu_capture = true;
            if (NVN_FLAG("diag_miniatura")) NVN_LOG("captura llamada desde otra lista\n");
        }
        for (const Command& c : called->commands) {
            if (list->commands.size() == list->commands.capacity()) [[unlikely]]
                grow(*list);
            list->commands.push_back(c);
        }
    }
}
// CopyCommands copia los comandos de los handles en la lista (CallCommands salta a ellos): con nuestras listas, lo mismo.
void nvnCommandBufferCopyCommands(void* cmd, int count, const uint64_t* handles) {
    nvnCommandBufferCallCommands(cmd, count, handles);
}
void nvnCommandBufferClearBuffer(void* cmd, uint64_t address, size_t size, uint32_t value) {
    record(cmd, cmd::ClearBuffer{address, size, value});
}
void nvnCommandBufferFenceSync(void* cmd, void* sync, int, int) { record(cmd, cmd::FenceSync{sync}); }
// La cola espera a un sync: solo los señala un FenceSync de esta misma cola, que va antes en el orden (como
// nvnQueueWaitSync), así que no hay nada que esperar.
void nvnCommandBufferWaitSync(void*, const void*) {}
// Eventos: la señal escribe o suma el valor al reproducir, en su sitio del orden (como ReportValue); la espera para la
// reproducción hasta que el valor cumple la condición (lo cambia la CPU con nvnEventSignal o una señal anterior).
void nvnCommandBufferSignalEvent(void* cmd, const void* event, int mode, int /*location*/, int /*flags*/, uint32_t value) {
    uint32_t* p = nvn::event_value(event);
    if (p) record(cmd, cmd::Event{p, mode, value, false});
}
void nvnCommandBufferWaitEvent(void* cmd, const void* event, int mode, uint32_t value) {
    uint32_t* p = nvn::event_value(event);
    if (p) record(cmd, cmd::Event{p, mode, value, true});
}
// Constantes en línea (las de cada draw en MH Rise): los bytes se copian ahora a la lista y se escriben en
// buffer + offset al reproducir, entre los comandos de antes y los de después (ver el ejecutor).
void nvnCommandBufferUpdateUniformBuffer(void* cmd, uint64_t buffer, size_t buffer_size, int64_t offset, size_t size,
                                         const void* data) {
    CommandList* list = recording(cmd);
    if (!list || !buffer || !data || !size) return;
    if (offset < 0 || static_cast<size_t>(offset) > buffer_size || size > buffer_size - static_cast<size_t>(offset)) {
        static int logged = 0;
        if (nvn::first_times(logged, 8))
            NVN_LOG("UpdateUniformBuffer fuera del buffer: +%lld (%zu bytes) en %zu\n", static_cast<long long>(offset), size,
                    buffer_size);
        if (offset < 0 || static_cast<size_t>(offset) >= buffer_size) return;
        size = buffer_size - static_cast<size_t>(offset);
    }
    void* copy = list->inline_data(size);
    memcpy(copy, data, size);
    add(*list, cmd::UpdateBuffer{buffer + static_cast<uint64_t>(offset), static_cast<uint32_t>(size), copy});
}
void nvnCommandBufferClearTexture(void* cmd, const void* texture, const void* view, const CopyRegion* region,
                                  const float* color, int mask) {
    clear_texture(cmd, texture, view, region, color, mask);
}
void nvnCommandBufferClearTexturei(void* cmd, const void* texture, const void* view, const CopyRegion* region,
                                   const int* color, int mask) {
    clear_texture(cmd, texture, view, region, color, mask);
}
void nvnCommandBufferClearTextureui(void* cmd, const void* texture, const void* view, const CopyRegion* region,
                                    const uint32_t* color, int mask) {
    clear_texture(cmd, texture, view, region, color, mask);
}

// --- Depuración: marcas y grupos. Son para herramientas de depuración de GPU; sin VK_EXT_debug_utils en el Adreno no hay
// quien las lea. ---

nvn::Boolean nvnCommandBufferIsRecording(const void* cmd) { return recording(const_cast<void*>(cmd)) != nullptr; }
void nvnCommandBufferSetDebugLabel(void*, const char*) {}
void nvnCommandBufferInsertDebugMarker(void*, const char*) {}
void nvnCommandBufferInsertDebugMarkerStatic(void*, uint32_t, const char*) {}
void nvnCommandBufferInsertDebugMarkerDynamic(void*, uint32_t, const char*) {}
void nvnCommandBufferPushDebugGroup(void*, const char*) {}
void nvnCommandBufferPushDebugGroupStatic(void*, uint32_t, const char*) {}
void nvnCommandBufferPushDebugGroupDynamic(void*, uint32_t, const char*) {}
void nvnCommandBufferPopDebugGroup(void*) {}
void nvnCommandBufferPopDebugGroupId(void*, uint32_t) {}
void nvnCommandBufferReportCounter(void* cmd, int type, uint64_t address) {
    record(cmd, cmd::ReportCounter{type, address});
}

}  // extern "C"

NVN_PROC(nvnCommandBufferInitialize);
NVN_PROC(nvnCommandBufferFinalize);
NVN_PROC(nvnCommandBufferAddCommandMemory);
NVN_PROC(nvnCommandBufferAddControlMemory);
NVN_PROC(nvnCommandBufferGetCommandMemoryFree);
NVN_PROC(nvnCommandBufferGetControlMemoryFree);
NVN_PROC(nvnCommandBufferGetCommandMemoryUsed);
NVN_PROC(nvnCommandBufferGetControlMemoryUsed);
NVN_PROC(nvnCommandBufferSetMemoryCallback);
NVN_PROC(nvnCommandBufferSetMemoryCallbackData);
NVN_PROC(nvnCommandBufferBeginRecording);
NVN_PROC(nvnCommandBufferEndRecording);
NVN_PROC(nvnCommandBufferSetRenderTargets);
NVN_PROC(nvnCommandBufferClearColor);
NVN_PROC(nvnCommandBufferClearColori);
NVN_PROC(nvnCommandBufferClearColorui);
NVN_PROC(nvnCommandBufferBindTextures);
NVN_PROC(nvnCommandBufferBindSeparateTextures);
NVN_PROC(nvnCommandBufferBindSeparateSamplers);
NVN_PROC(nvnCommandBufferSetRasterizerDiscard);
NVN_PROC(nvnCommandBufferBindTransformFeedbackBuffers);
NVN_PROC(nvnCommandBufferBindTransformFeedbackBuffer);
NVN_PROC(nvnCommandBufferBeginTransformFeedback);
NVN_PROC(nvnCommandBufferEndTransformFeedback);
NVN_PROC(nvnCommandBufferPauseTransformFeedback);
NVN_PROC(nvnCommandBufferResumeTransformFeedback);
NVN_PROC(nvnCommandBufferDrawTransformFeedback);
NVN_PROC(nvnCommandBufferSetViewportSwizzles);
NVN_PROC(nvnCommandBufferResolveDepthBuffer);
NVN_PROC(nvnCommandBufferSetColorReductionEnable);
NVN_PROC(nvnCommandBufferCallCommands);
NVN_PROC(nvnCommandBufferClearDepthStencil);
NVN_PROC(nvnCommandBufferSetViewport);
NVN_PROC(nvnCommandBufferSetViewports);
NVN_PROC(nvnCommandBufferSetScissor);
NVN_PROC(nvnCommandBufferSetScissors);
NVN_PROC(nvnCommandBufferSetDepthRange);
NVN_PROC(nvnCommandBufferSetDepthRanges);
NVN_PROC(nvnCommandBufferSetPrimitiveRestart);
NVN_PROC(nvnCommandBufferSetConservativeRasterEnable);
NVN_PROC(nvnCommandBufferSetConservativeRasterDilate);
NVN_PROC(nvnCommandBufferSetPointSize);
NVN_PROC(nvnCommandBufferSetLineWidth);
NVN_PROC(nvnCommandBufferSetTiledCacheAction);
NVN_PROC(nvnCommandBufferSetTiledCacheTileSize);
NVN_PROC(nvnCommandBufferSetStencilCullCriteria);
NVN_PROC(nvnCommandBufferSetSubpixelPrecisionBias);
NVN_PROC(nvnCommandBufferSetDepthBounds);
NVN_PROC(nvnCommandBufferSetDepthClamp);
NVN_PROC(nvnCommandBufferSetPolygonOffsetClamp);
NVN_PROC(nvnCommandBufferSetShaderScratchMemory);
NVN_PROC(nvnCommandBufferSaveZCullData);
NVN_PROC(nvnCommandBufferRestoreZCullData);
NVN_PROC(nvnCommandBufferSetStencilRef);
NVN_PROC(nvnCommandBufferSetStencilMask);
NVN_PROC(nvnCommandBufferSetStencilValueMask);
NVN_PROC(nvnCommandBufferBindBlendState);
NVN_PROC(nvnCommandBufferBindColorState);
NVN_PROC(nvnCommandBufferBindChannelMaskState);
NVN_PROC(nvnCommandBufferBindDepthStencilState);
NVN_PROC(nvnCommandBufferBindPolygonState);
NVN_PROC(nvnCommandBufferBindMultisampleState);
NVN_PROC(nvnCommandBufferBindVertexAttribState);
NVN_PROC(nvnCommandBufferBindVertexStreamState);
NVN_PROC(nvnCommandBufferBindProgram);
NVN_PROC(nvnCommandBufferBindVertexBuffers);
NVN_PROC(nvnCommandBufferBindVertexBuffer);
NVN_PROC(nvnCommandBufferBindUniformBuffers);
NVN_PROC(nvnCommandBufferBindUniformBuffer);
NVN_PROC(nvnCommandBufferBindStorageBuffer);
NVN_PROC(nvnCommandBufferBindStorageBuffers);
NVN_PROC(nvnCommandBufferBindSeparateTexture);
NVN_PROC(nvnCommandBufferBindTexture);
NVN_PROC(nvnCommandBufferBindSeparateSampler);
NVN_PROC(nvnCommandBufferBindImage);
NVN_PROC(nvnCommandBufferBindImages);
NVN_PROC(nvnCommandBufferSetTexturePool);
NVN_PROC(nvnCommandBufferSetSamplerPool);
NVN_PROC(nvnCommandBufferDrawArrays);
NVN_PROC(nvnCommandBufferDrawElements);
NVN_PROC(nvnCommandBufferDrawElementsBaseVertex);
NVN_PROC(nvnCommandBufferDrawArraysInstanced);
NVN_PROC(nvnCommandBufferDrawElementsInstanced);
NVN_PROC(nvnCommandBufferDispatchCompute);
NVN_PROC(nvnCommandBufferDispatchComputeIndirect);
NVN_PROC(nvnCommandBufferDrawArraysIndirect);
NVN_PROC(nvnCommandBufferDrawElementsIndirect);
NVN_PROC(nvnCommandBufferMultiDrawElementsIndirectCount);
NVN_PROC(nvnCommandBufferMultiDrawArraysIndirectCount);
NVN_PROC(nvnCommandBufferSetBlendColor);
NVN_PROC(nvnCommandBufferSetRenderEnable);
NVN_PROC(nvnCommandBufferSetRenderEnableConditional);
NVN_PROC(nvnCommandBufferReportValue);
NVN_PROC(nvnCommandBufferResetCounter);
NVN_PROC(nvnCommandBufferClearBuffer);
NVN_PROC(nvnCommandBufferFenceSync);
NVN_PROC(nvnCommandBufferUpdateUniformBuffer);
NVN_PROC(nvnCommandBufferClearTexture);
NVN_PROC(nvnCommandBufferClearTexturei);
NVN_PROC(nvnCommandBufferClearTextureui);
NVN_PROC(nvnCommandBufferIsRecording);
NVN_PROC(nvnCommandBufferSetDebugLabel);
NVN_PROC(nvnCommandBufferInsertDebugMarker);
NVN_PROC(nvnCommandBufferInsertDebugMarkerStatic);
NVN_PROC(nvnCommandBufferInsertDebugMarkerDynamic);
NVN_PROC(nvnCommandBufferPushDebugGroup);
NVN_PROC(nvnCommandBufferPushDebugGroupStatic);
NVN_PROC(nvnCommandBufferPushDebugGroupDynamic);
NVN_PROC(nvnCommandBufferPopDebugGroup);
NVN_PROC(nvnCommandBufferPopDebugGroupId);
NVN_PROC(nvnCommandBufferCopyBufferToBuffer);
NVN_PROC(nvnCommandBufferCopyTextureToTexture);
NVN_PROC(nvnCommandBufferCopyBufferToTexture);
NVN_PROC(nvnCommandBufferBarrier);
NVN_PROC(nvnCommandBufferReportCounter);
NVN_PROC(nvnCommandBufferGetCommandMemorySize);
NVN_PROC(nvnCommandBufferGetControlMemorySize);
NVN_PROC(nvnCommandBufferGetMemoryCallback);
NVN_PROC(nvnCommandBufferGetMemoryCallbackData);
NVN_PROC(nvnCommandBufferSetCommandMemoryCallbackEnabled);
NVN_PROC(nvnCommandBufferSetAlphaRef);
NVN_PROC(nvnCommandBufferSetInnerTessellationLevels);
NVN_PROC(nvnCommandBufferSetOuterTessellationLevels);
NVN_PROC(nvnCommandBufferSetPatchSize);
NVN_PROC(nvnCommandBufferSetProgramSubroutines);
NVN_PROC(nvnCommandBufferBindCoverageModulationTable);
NVN_PROC(nvnCommandBufferSetSampleMask);
NVN_PROC(nvnCommandBufferSetColorReductionThresholds);
NVN_PROC(nvnCommandBufferCopyCommands);
NVN_PROC(nvnCommandBufferWaitSync);
NVN_PROC(nvnCommandBufferSignalEvent);
NVN_PROC(nvnCommandBufferWaitEvent);
NVN_PROC(nvnCommandBufferCopyTextureToBuffer);
NVN_PROC(nvnCommandBufferSetCopyRowStride);
NVN_PROC(nvnCommandBufferSetCopyImageStride);
NVN_PROC(nvnCommandBufferGetCopyRowStride);
NVN_PROC(nvnCommandBufferGetCopyImageStride);
NVN_PROC(nvnCommandBufferDownsample);
NVN_PROC(nvnCommandBufferDownsampleTextureView);
NVN_PROC(nvnCommandBufferTiledDownsample);
NVN_PROC(nvnCommandBufferTiledDownsampleTextureView);
NVN_PROC(nvnCommandBufferDiscardColor);
NVN_PROC(nvnCommandBufferDiscardDepthStencil);
NVN_PROC(nvnCommandBufferDrawTexture);
