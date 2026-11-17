#pragma once
// Comandos que graba un NVNcommandBuffer. Se guardan tal cual (con copias de los objetos de estado
// y de las vistas, que el juego puede reutilizar); el ejecutor los reproduce al enviar a la cola, con
// el estado de la cola persistiendo entre listas, como en NVN.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>
#include <vector>

#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/state.h"

namespace nvn {

struct BufferRange {
    uint64_t address;
    uint64_t size;
};
struct CopyRegion {
    int x, y, z, width, height, depth;
};

// Lista corta de T guardada dentro del propio comando (antes un std::vector: ~13k reservas de memoria por
// frame, sobre todo BindUniformBuffers). Si no caben N, los elementos van a memoria de la lista de comandos
// (CommandList::spill), que vive lo mismo que el comando. Sin destructor ni reservas propias: Command es
// trivial de copiar y de destruir, y vaciar una lista no tiene que recorrerla. Copiar una SmallVector copia el
// puntero a esa memoria: la copia no debe vivir más que su lista (el ejecutor no guarda comandos de estos).
template <typename T, uint32_t N>
class SmallVector {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    SmallVector() = default;
    // spill(bytes) da memoria que vive lo mismo que el comando; solo se pide si n > N.
    template <typename Spill>
    SmallVector(const T* items, size_t n, Spill&& spill) : size_(static_cast<uint32_t>(n)) {
        T* out = inline_;
        if (n > N) out = external_ = static_cast<T*>(spill(n * sizeof(T)));
        if (n) memcpy(out, items, n * sizeof(T));
    }
    static constexpr size_t inline_capacity() { return N; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T* data() const { return size_ <= N ? inline_ : external_; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size_; }
    const T& operator[](size_t k) const { return data()[k]; }
    // El estado del ejecutor guarda std::vector (`st_.attribs = v.attribs`): se copia a un vector del hilo y la
    // asignación reutiliza la capacidad del destino, sin reservar memoria. La referencia vale hasta la siguiente
    // conversión de este mismo tipo en el mismo hilo: no se guarda.
    operator const std::vector<T>&() const {
        thread_local std::vector<T> scratch;
        scratch.assign(begin(), end());
        return scratch;
    }

private:
    uint32_t size_ = 0;
    union {
        T inline_[N];
        T* external_;
    };
};

namespace cmd {

struct SetRenderTargets {
    int count;
    Texture* colors[8];
    TextureViewDesc color_views[8];
    Texture* depth;
    TextureViewDesc depth_view;
};
struct ClearColor {
    int index;
    float color[4];
    int mask;
};
struct ClearDepthStencil {
    float depth;
    bool depth_mask;
    int stencil, stencil_mask;
};
struct Viewport {
    int index;
    float x, y, w, h;
};
struct Scissor {
    int index;
    int x, y, w, h;
};
struct DepthRange {
    float n, f;
};
struct BindProgram {
    Program* program;
    int stages;
};
// Capacidades dentro del comando: cada struct de estos queda en ≤ 128 bytes (lo que ocupa CopyTexture), para
// que no sean ellos los que fijen el tamaño de Command si SetRenderTargets deja algún día de hacerlo.
struct BindVertexBuffers {
    int first;
    SmallVector<BufferRange, 6> ranges;
};
struct BindUniformBuffers {
    int stage, first;
    SmallVector<BufferRange, 6> ranges;
};
struct BindStorageBuffer {
    int stage, index;
    BufferRange range;
};
struct BindTexture {  // separada o combinada (sampler en los 32 bits altos)
    int stage, index;
    uint64_t handle;
    bool combined;
};
struct BindSampler {
    int stage, index;
    uint64_t handle;
};
struct BindVertexAttribs {
    SmallVector<VertexAttribState, 16> attribs;
};
struct BindVertexStreams {
    SmallVector<VertexStreamState, 8> streams;
};
struct SetPools {
    DescriptorPool* textures;
    DescriptorPool* samplers;
};
struct Stencil {
    enum What { Ref, Mask, ValueMask } what;
    int faces, value;
};
struct Draw {
    int primitive;
    int index_type;  // -1: sin índices
    int count, first;
    uint64_t index_address;
    int base_vertex, base_instance, instances;
    uint64_t indirect = 0;  // si no es 0, los parámetros están en esta dirección (se leen al ejecutar)
    // Multi-draw (si no es 0): el número de draws está en esta dirección (hasta max_draws), con los parámetros de
    // cada uno a stride bytes del anterior desde indirect.
    uint64_t draw_count = 0;
    int max_draws = 0, stride = 0;
};
struct Dispatch {
    int x, y, z;
    uint64_t indirect = 0;  // {x, y, z} en esta dirección
};
struct BlendColor {
    float color[4];
};
// mode kRasterizerDiscard: no es el render enable sino SetRasterizerDiscard (enable = descartar).
constexpr int kRasterizerDiscard = -2;
struct RenderEnable {
    bool enable;
    int mode = -1;             // condicional: NVNconditionalRenderMode (-1: no lo es)
    uint64_t counters = 0;     // condicional: dirección de los dos contadores que compara
};
struct ReportValue {
    uint32_t value;
    uint64_t address;
};
struct ClearBuffer {
    uint64_t address, size;
    uint32_t value;
};
struct FenceSync {
    void* sync;
};
struct CopyBuffer {
    uint64_t src, dst, size;
};
struct CopyTexture {
    Texture* src;
    TextureViewDesc src_view;
    CopyRegion src_region;
    Texture* dst;
    TextureViewDesc dst_view;
    CopyRegion dst_region;
};
struct CopyBufferToTexture {
    uint64_t src;
    Texture* dst;
    TextureViewDesc dst_view;
    CopyRegion region;
    // SetCopyRowStride / SetCopyImageStride del command buffer al grabar: las zancadas del buffer (0: compacto).
    int64_t row_stride = 0, image_stride = 0;
    // DIAGNÓSTICO (reproducción en diferido): los bytes del buffer que la copia va a leer y la huella de sus extremos al
    // grabar (edge_print; span 0: sin tomar). La cola lee el buffer al reproducir: si para entonces el juego ha reutilizado
    // esa memoria, la textura se queda para siempre con los texels de otra. Unity (Disco Elysium) sube así sus texturas sin
    // comprimir (main+0x95be60, con un buffer temporal que suelta nada más grabar la copia, main+0x95beb8) y no reutiliza
    // esa memoria hasta 3 presentaciones después (main+0x9915e8), así que no debería pasar: el ejecutor lo cuenta.
    uint32_t span = 0;
    uint64_t print = 0;
    // DIAGNÓSTICO (Castlevania, ruido de los poderes a 0xCD): en las copias grandes, una muestra de todo el buffer (1 byte
    // de cada 64) y cuántos de esos bytes valen 0xCD, al grabar.
    uint64_t sample_print = 0;
    uint32_t sample_cd = 0, samples = 0;
};
// Muestra de todo el buffer: 1 byte de cada 64 (huella FNV y cuántos valen 0xCD).
inline void sample_stats(const void* p, size_t size, uint64_t& print, uint32_t& cd, uint32_t& n) {
    const auto* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    cd = 0, n = 0;
    for (size_t k = 0; k < size; k += 64, n++) {
        h = (h ^ b[k]) * 0x100000001b3ull;
        cd += b[k] == 0xCD;
    }
    print = h;
}
// Huella de los extremos de [p, p + size): hasta 64 bytes del principio y 64 del final (lo que basta para ver que un buffer
// temporal de subida ya es de otra textura sin recorrerlo entero al grabar y al reproducir).
inline uint64_t edge_print(const void* p, size_t size) {
    const auto* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull ^ size;
    const size_t head = size < 64 ? size : 64;
    for (size_t k = 0; k < head; k++) h = (h ^ b[k]) * 0x100000001b3ull;
    const size_t tail = size - head < 64 ? size - head : 64;
    for (size_t k = size - tail; k < size; k++) h = (h ^ b[k]) * 0x100000001b3ull;
    return h ? h : 1;
}
// nvnCommandBufferCopyTextureToBuffer: la región de la textura a un buffer (memoria del juego), con las zancadas del
// command buffer. Al reproducir, desde la memoria de la textura (como ReadTexels).
struct CopyTextureToBuffer {
    Texture* src;
    TextureViewDesc src_view;
    CopyRegion region;
    uint64_t dst;
    int64_t row_stride, image_stride;
};
// nvnCommandBufferDownsample* (los Tiled* dan lo mismo): un multisample resuelto en una textura de una muestra, o una
// textura reducida (filtrada) a otra más pequeña. El nivel y la capa, los de cada vista.
struct Downsample {
    Texture* src;
    TextureViewDesc src_view;
    Texture* dst;
    TextureViewDesc dst_view;
};
// nvnCommandBufferSignalEvent / WaitEvent: el valor de 32 bits de un NVNevent (memoria del juego).
struct Event {
    uint32_t* value;
    int mode;  // señal: NVNeventSignalMode (WRITE 0, ADD 1); espera: NVNeventWaitMode (EQUAL 0, GEQUAL_WRAP 1)
    uint32_t v;
    bool wait;
};
// nvnCommandBufferSetSampleMask: las muestras que puede escribir cada fragmento (bit por muestra).
struct SampleMask {
    uint32_t mask;
};
struct Barrier {
    int bits;
};
// reset: es nvnCommandBufferResetCounter (address 0): pone a cero ese contador de la GPU.
struct ReportCounter {
    int type;
    uint64_t address;
    bool reset = false;
};
// Imagen (storage image) del handle de nvnDeviceGetImageHandle en la ranura `index` de la etapa.
struct BindImage {
    int stage, index;
    uint64_t handle;
};
// nvnCommandBufferClearTexture{,i,ui}: una región de una textura a un color (float, int o uint según el formato).
struct ClearTexture {
    Texture* texture;
    TextureViewDesc view;
    CopyRegion region;
    uint32_t color[4];  // los bits tal cual: VkClearColorValue los lee como float, int o uint según el formato
    int mask;           // NVNclearColorMask: R 1, G 2, B 4, A 8
};
// nvnCommandBufferUpdateUniformBuffer: `size` bytes de `data` (copiados al grabar, en la lista) a `address` al
// reproducir, en su sitio del orden.
struct UpdateBuffer {
    uint64_t address;
    uint32_t size;
    const void* data;
};
struct PrimitiveRestart {
    bool enable;
    uint32_t index;
};
// Transform feedback (emulado con un SSBO, ver nvnshader::kXfbBank): buffers de captura y Begin/End/Pause/Resume.
struct TransformFeedback {
    enum Op { Bind, Begin, End, Pause, Resume, Draw } op;
    int index;          // Bind: buffer de captura (0..3); Draw: la primitiva
    BufferRange range;  // Bind
    uint64_t control = 0;  // Begin/End/Pause/Resume/Draw: el buffer de control del juego (dónde va la cuenta)
};

}  // namespace cmd

using Command = std::variant<cmd::SetRenderTargets, cmd::ClearColor, cmd::ClearDepthStencil, cmd::Viewport,
                             cmd::Scissor, cmd::DepthRange, cmd::BindProgram, cmd::BindVertexBuffers,
                             cmd::BindUniformBuffers, cmd::BindStorageBuffer, cmd::BindTexture, cmd::BindSampler,
                             BlendState, ColorState, ChannelMaskState, DepthStencilState, PolygonState,
                             MultisampleState, cmd::BindVertexAttribs, cmd::BindVertexStreams, cmd::SetPools,
                             cmd::Stencil, cmd::Draw, cmd::Dispatch, cmd::CopyBuffer, cmd::CopyTexture,
                             cmd::CopyBufferToTexture, cmd::Barrier, cmd::ReportCounter, cmd::BlendColor,
                             cmd::RenderEnable, cmd::ReportValue, cmd::ClearBuffer, cmd::BindImage, cmd::ClearTexture,
                             cmd::UpdateBuffer, cmd::PrimitiveRestart, cmd::CopyTextureToBuffer, cmd::Downsample,
                             cmd::Event, cmd::SampleMask, cmd::FenceSync, cmd::TransformFeedback>;

// Las listas se reciclan (cmdbuf.cpp) y vaciarlas no recorre sus ~35k comandos: que siga siendo así.
static_assert(std::is_trivially_destructible_v<Command> && std::is_trivially_copyable_v<Command>);
// Las listas cortas no hacen crecer Command (lo fija SetRenderTargets).
static_assert(sizeof(cmd::BindVertexBuffers) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindUniformBuffers) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindVertexAttribs) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindVertexStreams) <= sizeof(cmd::SetRenderTargets));

struct CommandList {
    std::vector<Command> commands;
    // Los elementos de las SmallVector que no caben en su comando: viven lo mismo que la lista.
    std::vector<std::unique_ptr<uint64_t[]>> spilled;
    void* spill(size_t bytes) { return spilled.emplace_back(new uint64_t[(bytes + 7) / 8]).get(); }
    // Datos en línea (UpdateUniformBuffer, uno o más por draw): trozos de 64 KB que se llenan seguidos (un `new` por
    // actualización eran miles por frame) y que la lista conserva al reciclarse (reset_inline); lo que no cabe en un
    // trozo, aparte (spill).
    static constexpr size_t kInlineChunk = 64 << 10;
    std::vector<std::unique_ptr<uint64_t[]>> inline_chunks;
    size_t inline_chunk = 0, inline_used = 0;
    void* inline_data(size_t bytes) {
        bytes = (bytes + 15) & ~size_t{15};
        if (bytes > kInlineChunk) return spill(bytes);
        if (inline_chunk < inline_chunks.size() && inline_used + bytes > kInlineChunk) inline_chunk++, inline_used = 0;
        if (inline_chunk == inline_chunks.size()) inline_chunks.emplace_back(new uint64_t[kInlineChunk / 8]);
        void* p = reinterpret_cast<uint8_t*>(inline_chunks[inline_chunk].get()) + inline_used;
        inline_used += bytes;
        return p;
    }
    // Para la siguiente grabación: los trozos se reutilizan (como mucho 1 MB guardado por lista).
    void reset_inline() {
        if (inline_chunks.size() > 16) inline_chunks.resize(16);
        inline_chunk = inline_used = 0;
        cpu_capture = false;
    }
    // Pinta en una textura que la CPU va a leer de su memoria (Xenoblade DE: la captura de la miniatura de la partida,
    // la única con flags IMAGE a secas): al enviarla, la cola se reproduce y se espera a la GPU ahí mismo, para que la
    // devolución (Executor::readback_1x1_targets) esté en la memoria del juego antes de que la lea.
    bool cpu_capture = false;
    // Reproducción en diferido: las texturas que sus comandos nombran por puntero (una entrada por mención) y si la lista ya
    // se envió alguna vez. Al enviarla, cada una apunta el ticket (Texture::last_ticket) y deja de contar en open_refs; si
    // se recicla sin enviarse, solo lo segundo (note_list_dropped).
    // submitted también cuando sus menciones pasaron a la lista que la llamó (nvnCommandBufferCallCommands copia sus
    // comandos en ella): desde entonces quien las cuenta, y quien les pondrá el ticket, es esa.
    std::vector<Texture*> texture_refs;
    bool submitted = false;
};

// Reproducción en diferido. El juego destruye o reinicia AL MOMENTO (nvnTextureFinalize, nvnMemoryPoolFinalize…) cosas que
// los comandos ya enviados necesitan cuando la cola los reproduce, que aquí es más tarde y en otro hilo. En la consola eso
// no pasa: el juego solo las suelta cuando la GPU ya ejecutó esos comandos. Disco Elysium (Unity + AmplifyTexture): cada
// página del fondo se sube a una textura temporal de 136x136, se graba su copia a la caché y la temporal se finaliza; con
// la copia aún sin reproducir se quedaba sin origen y la capa de la caché seguía con su relleno (teselas negras en media
// habitación). replay_barrier espera a que la cola haya reproducido hasta ese envío (0: todo lo enviado hasta ahora); no
// espera a la GPU. No hace nada desde el propio hilo de la cola ni en modo síncrono.
void replay_barrier(uint64_t ticket, const char* what);
// El último envío por el que este hilo esperó a la GPU con un NVNsync del juego (nvnSyncWait con plazo y señalado); 0 si
// nunca. Un hilo así libera recursos cuando la GPU ha acabado el envío que los usaba (Unreal Engine 5: el
// RHIInterruptThread espera el sync de cada envío y luego finaliza sus pools, buffers y texturas), así que lo que
// finaliza no lo nombra ningún envío posterior y su barrera solo tiene que cubrir hasta ese: con la barrera general
// (todo lo encolado) cada nvnMemoryPoolFinalize esperaba a que la cola reprodujera el frame que el juego acababa de
// enviar, el hilo soltaba menos envíos de los que el juego creaba (iba 1500 envíos por detrás al minuto y medio) y
// Unreal agotaba su presupuesto de memoria (3 GB comprometidos, aborto en su aviso de memoria agotada).
void note_thread_completion(uint64_t ticket);
uint64_t thread_completion_ticket();
uint64_t replayed_ticket();  // el último envío del juego ya reproducido
// Los PresentTexture del juego (su número de frame), para saber cuántos frames después de usar algo lo suelta.
uint32_t game_frame();
void note_game_present();
// El intervalo de presentación en curso (el de la ventana, o 1 con <datos>/60fps): lo apunta present() y lo lee la guarda
// del FSR (a intervalo 1 el frame tiene 16,7 ms).
void set_present_interval(int interval);
int present_interval();
// ¿Acabó ya el frame `frame` del juego (presentó) y la cola reprodujo todo lo que envió hasta entonces?
bool game_frame_replayed(uint32_t frame, uint64_t replayed);
// Una lista grabada que se recicla sin haberse enviado: sus texturas dejan de contarla (Texture::open_refs).
void note_list_dropped(CommandList& list);
// Para el log de ritmo (window.cpp), desde la última vez: barreras pedidas, las que esperaron y cuánto; y texturas
// finalizadas o reiniciadas (texture.cpp): con un envío suyo aún sin reproducir, con comandos grabados sin enviar, y cuántos
// frames del juego después del último comando que las nombraba (0, 1, 2, 3, 4 o más).
void take_barrier_stats(uint64_t& calls, uint64_t& waits, uint64_t& wait_ns, uint64_t& max_ns);
void take_teardown_stats(uint64_t& total, uint64_t& pending, uint64_t& unsent, uint64_t& limbo, uint64_t& copied_bytes,
                         uint64_t frames_after[5]);
// Texturas finalizadas por el juego que comandos sin reproducir aún nombran (texture.cpp, "limbo"): la cola las finaliza de
// verdad tras reproducir el último envío que las nombra. Cuántas hay ahora.
void reap_limbo_textures(uint64_t replayed);
size_t limbo_textures();
// Más cuentas para el log de ritmo, desde la última vez. Del limbo (texture.cpp): cuántas texturas y cuántos bytes (imágenes
// y copias de texels) llegó a tener, las que soltaron sus válvulas (por edad y por tope de memoria) y las texturas
// finalizadas recién creadas, sin imagen y sin que ningún comando las nombre (las que limbo_recientes conservaría). De la
// cola (executor.cpp): cuántos envíos llegó a tener pendientes y cuántos CopyBufferToTexture encontraron su buffer
// cambiado al reproducirse (cmd::CopyBufferToTexture::print). kept_total no es del tramo: las texturas finalizadas del todo
// en lo que va de ejecución, cuyos objetos se conservan para siempre (los pools de texturas y la cola pueden nombrarlos).
void take_limbo_stats(uint64_t& max_count, uint64_t& max_bytes, uint64_t& forced_age, uint64_t& forced_budget, uint64_t& young,
                      uint64_t& kept_total);
void take_queue_stats(uint64_t& max_pending, uint64_t& stale_uploads);

// Ejecuta en Vulkan las listas que el juego envía a la cola (executor.cpp).
void execute(const std::vector<const CommandList*>& lists);
// Presentar en orden con los envíos (la cola de envíos asíncrona) y soltar una lista que el juego recicla.
void queue_present(Window& window, int index);
void release_command_list(CommandList* list);
// Una lista que ya no se va a reproducir (la cola ya la soltó): vuelve al pool de cmdbuf.cpp para que la
// próxima grabación reutilice su memoria. Barato: se puede llamar con el mutex de la cola tomado (no vacía ni
// libera nada ahí; con el interruptor sin_reciclar la borra, como antes).
void recycle_command_list(CommandList* list);
// Afinidad (os.cpp): el hilo que llama pasa a los núcleos grandes junto con los demás hilos críticos (el
// principal del juego, MainDraw y NvnSubmit). refresh_critical_threads, una vez por frame desde el hilo que
// presenta: como mucho una vez por segundo mira el interruptor sin_afinidad, rehace las máscaras que se hayan
// perdido o cambiado y devuelve la completa a los hilos no críticos que heredaron la de los grandes.
void pin_critical_thread();
// Los hilos de trabajo de fondo (precarga de shaders, pipelines en segundo plano) a los núcleos pequeños: su CPU no se
// la quita a los críticos ni calienta los grandes (el SoC caliente baja la GPU). sin_afinidad: como estén.
void pin_background_thread();
void refresh_critical_threads();
// Copia la textura de la ventana a la imagen del swapchain (la llama window.cpp al presentar).
// Con FSR 1 (fsr.cpp) si el formato lo permite; si no, un blit lineal.
bool blit_to_swapchain(Texture& texture, const VkRect2D& crop, VkImage swapchain_image, VkFormat format, VkExtent2D extent,
                       VkCommandBuffer cmd);
// Se presentó un frame (window.cpp): las texturas que escribe la CPU se revisan una vez por frame.
void next_frame();
// Para la resolución dinámica (window.cpp): la GPU del último frame del juego que ya se leyó (sus timestamps de
// ReportCounter), con un número que sube con cada uno (true si hay uno más nuevo que `serial`), y el ejecutor del último
// frame presentado (la suma de sus envíos).
bool game_gpu_frame(uint64_t& serial, int64_t& ns);
uint64_t last_frame_run_ns();
// La GPU ocupada entre los dos sellos del último frame del juego que se pudo medir (Executor::busy_end) y el tramo entre
// ellos, con un número que sube con cada uno (true si hay uno más nuevo que `serial`). El número se publica el último:
// quien lo ve nuevo lee los dos valores de ese frame o de uno posterior.
inline bool game_gpu_busy(uint64_t& serial, int64_t& busy_ns, int64_t& span_ns) {
    const uint64_t now = g_drop_stats.busy_serial.load(std::memory_order_acquire);
    if (now == serial) return false;
    serial = now;
    busy_ns = static_cast<int64_t>(g_drop_stats.busy_last_ns.load(std::memory_order_relaxed));
    span_ns = static_cast<int64_t>(g_drop_stats.busy_span_last_ns.load(std::memory_order_relaxed));
    return true;
}
// DEPURACIÓN: traza del frame siguiente pedida desde el código (el detector de destellos de window.cpp); full: con los
// render targets, como <datos>/trazar con "completa".
void request_trace(bool full);
// DEPURACIÓN: en ciertos frames vuelca a la SD la textura presentada y los render targets del frame.
void debug_dump_frame(Texture* presented);
// Envíos asíncronos (executor.cpp): número del último envío, espera a que termine uno (false si vence
// el plazo; no espera a que acabe la reproducción de otro) y espera a que terminen todos.
uint64_t last_submit();
// nn::os::FlushDataCache: la CPU acaba de escribir [address, address+size) para la GPU. Va al siguiente envío (en orden
// con los anteriores): los overlays que lo tocan toman la memoria del juego aunque los bytes no cambien (Xenoblade pone a 0
// el desplazamiento del quad de las películas en cada frame y un draw lo vuelve a 1e5; la huella no veía el 0).
void note_cpu_flush(uint64_t address, uint64_t size);
// Un NVNsync queda señalado al terminar ese envío (queue.cpp; FenceSync de la cola y del command buffer).
void sync_signal_at(void* sync, uint64_t submit);
// NVNevent (queue.cpp): el valor de 32 bits de un evento (nullptr si no está inicializado) y cómo lo cambia una señal
// (NVNeventSignalMode: WRITE 0 lo escribe, ADD 1 lo suma), de la CPU o de la cola al reproducir SignalEvent.
uint32_t* event_value(const void* event);
void signal_event_value(uint32_t* value, int mode, uint32_t v);
bool wait_submit(uint64_t submit, uint64_t timeout_ns);
// solapar_frame (queue.cpp, el sync de nvnWindowAcquireTexture): espera solo a que el ejecutor haya reproducido ese envío
// (lo que leía de la memoria del juego ya está copiado), no a que la GPU lo termine. false si vence el plazo.
bool wait_replayed(uint64_t submit, uint64_t timeout_ns);
// Devoluciones de render targets que lee la CPU (miniatura de la partida) grabadas y aún sin entregar: la espera del sync de
// la ventana tiene que esperar entonces a la GPU, no solo a la reproducción.
int cpu_rt_pending();
void wait_idle();
// La VkQueue no admite accesos concurrentes: todo vkQueueSubmit/vkQueuePresentKHR pasa por este mutex (solo
// la llamada: no se espera nada con él). Va después de executor_mutex() si se toman los dos.
std::mutex& vk_queue_mutex();
// Lo que el juego suelta (la imagen de una textura, un sampler) puede seguir en un envío pendiente: se
// destruye cuando la GPU termina el último envío hecho hasta ahora; con la imagen caen sus vistas y
// framebuffers. Quien toca un objeto que el ejecutor puede estar leyendo toma executor_mutex() (el mismo
// que serializa la ejecución de los comandos).
std::recursive_mutex& executor_mutex();
// Perfil de cargas: lo que un hilo del juego esperó a executor_mutex() (el ejecutor lo tiene durante todo un envío).
void note_executor_lock_wait(uint64_t ns);
// Toma executor_mutex() con `lock` (sin tomar) y, si estaba ocupado, apunta la espera.
// ceder_cerrojo: cuántos hilos del juego esperan ahora a executor_mutex() (el ejecutor lo suelta entre pases si hay alguno).
void note_executor_lock_waiting(int delta);
inline void lock_executor(std::unique_lock<std::recursive_mutex>& lock) {
    if (lock.try_lock()) return;
    const auto t0 = std::chrono::steady_clock::now();
    note_executor_lock_waiting(1);
    lock.lock();
    note_executor_lock_waiting(-1);
    note_executor_lock_wait(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
}
void retire_texture_gpu(Texture& t);  // deja la textura sin imagen
void forget_texture_storage(Texture& t);  // su memoria del juego deja de ser suya (antes de cambiar Texture::storage)
void retire_sampler(VkSampler sampler);

}  // namespace nvn
