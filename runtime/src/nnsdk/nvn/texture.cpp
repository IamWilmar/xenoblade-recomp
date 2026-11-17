// nvnTextureBuilder*, nvnTexture*, nvnTextureView*, nvnTexturePool*, nvnSamplerBuilder*, nvnSampler*,
// nvnSamplerPool* y los handles del dispositivo. Las imágenes de Vulkan las crea el ejecutor cuando
// hacen falta; aquí solo se guarda la descripción y dónde están los texels en memoria del juego.

#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

using nvn::SamplerDesc;
using nvn::TextureDesc;
using nvn::TextureViewDesc;

namespace {

constexpr size_t kStorageAlignment = 512;

nvn::Objects<nvn::Sampler> g_samplers;

nvn::TextureLayout layout_of(const TextureDesc& d) {
    const nvn::FormatInfo* f = nvn::format_info(d.format);
    uint32_t bpb = f && f->bytes ? f->bytes : 4, block_w = f ? f->block : 1, block_h = f ? f->block_height() : 1;
    bool array = d.target == nvn::kTarget1DArray || d.target == nvn::kTarget2DArray ||
                 d.target == nvn::kTarget2DMultisampleArray || d.target == nvn::kTargetCubeArray;
    bool cube = d.target == nvn::kTargetCube || d.target == nvn::kTargetCubeArray;
    bool is_3d = d.target == nvn::kTarget3D;
    // Un cubemap tiene 6 caras aunque el juego deje depth en 1 (NieR lo hace); un array de cubos, depth
    // caras (múltiplo de 6). Con 1 sola capa, GetStorageSize salía de una cara y los datos de las otras
    // 5 se escribían sobre lo que hubiera detrás (el mar del prólogo salía magenta: ASTC inválido).
    uint32_t layers = d.target == nvn::kTargetCube ? 6 : array || cube ? std::max(d.depth, 1) : 1;
    uint32_t depth = is_3d ? std::max(d.depth, 1) : 1;
    uint32_t height = d.target == nvn::kTarget1D || d.target == nvn::kTarget1DArray ? 1 : d.height;
    uint32_t samples = std::max(d.samples, 1);
    // NVNpackagedTextureLayout (SetPackagedTextureLayout): byte 0 = alto del bloque y byte 1 = alineación del pitch, en
    // log2 de GOBs. Las texturas sparse de MH Rise traen 03 04 (8 GOBs de alto, pitch a 16: tiles de 64 KB) y el juego
    // escribe sus mips ya swizzleados así, en los offsets que le da GetViewOffset: hay que usar los mismos.
    // <datos>/sin_layout_empaquetado (al arrancar: las texturas se crean al cargar): el bloque calculado, como antes.
    const bool packaged = d.has_packaged_layout && !(d.flags & nvn::kTextureFlagLinear) && !is_3d &&
                          d.packaged_layout[0] <= 5 && d.packaged_layout[1] <= 5 && !NVN_FLAG("sin_layout_empaquetado");
    return nvn::compute_layout(d.width * (samples > 1 ? 2 : 1), height * (samples > 2 ? 2 : 1), depth, layers,
                               d.levels, bpb * (samples > 4 ? 2 : 1), block_w, block_h, d.flags & nvn::kTextureFlagLinear,
                               d.stride, is_3d, packaged ? d.packaged_layout[0] : -1, packaged ? d.packaged_layout[1] : 0);
}

// NVNstorageClass: el "kind" de Maxwell de las páginas que pide la textura. MH Rise solo lo pasa a su asignador de
// memoria de texturas (main+0x77055e8, main+0x7711088) y de ahí a nvnMemoryPoolMapVirtual, que aquí no lo usa: basta
// con que sea estable. Los de Maxwell: pitch (lineal) y block-linear genérico (GENERIC_16BX2). Sin compresión, el "raw"
// (sin el bit de compresión) es el mismo.
constexpr int kStorageClassPitch = 0x00, kStorageClassBlockLinear = 0xfe;
int storage_class_of(const TextureDesc& d) {
    return (d.flags & nvn::kTextureFlagLinear) ? kStorageClassPitch : kStorageClassBlockLinear;
}

// Nivel y capa que nombra una vista (lo que no fija, el primero).
int view_level(const TextureViewDesc* v) { return v && (v->flags & nvn::kViewLevels) ? v->base_level : 0; }
int view_layer(const TextureViewDesc* v) { return v && (v->flags & nvn::kViewLayers) ? v->base_layer : 0; }

// Lo que tocan WriteTexels/ReadTexels: un nivel, unas capas y una caja en bloques dentro del nivel. La región va en
// texels; en los arrays su z (y en los 1D array su y) elige las capas, como en las copias de NVN. false si no cabe.
struct TexelRegion {
    const nvn::LevelLayout* level;
    uint32_t level_index, layer, layers, bpb;
    nvn::BlockBox box;
    size_t row_pitch, image_pitch;  // de la memoria lineal del juego (compacta si no da otros)
};
bool texel_region(const nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* r, int64_t row_stride,
                  int64_t image_stride, TexelRegion& out) {
    const nvn::TextureLayout& L = t.layout;
    const int level = view_level(view);
    if (!r || !t.storage || level < 0 || level >= static_cast<int>(L.levels.size()) || r->width <= 0 || r->height <= 0)
        return false;
    const int target = t.desc.target;
    const bool array1d = target == nvn::kTarget1DArray;
    const bool layered = target == nvn::kTarget2DArray || target == nvn::kTarget2DMultisampleArray ||
                         target == nvn::kTargetCube || target == nvn::kTargetCubeArray;
    const nvn::LevelLayout& lv = L.levels[level];
    out.level = &lv;
    out.level_index = level;
    out.bpb = L.bytes_per_block;
    int first = 0, count = 1;
    int x = r->x, y = r->y, z = r->z, w = r->width, h = r->height, d = std::max(r->depth, 1);
    if (array1d) first = y, count = h, y = 0, h = 1;
    if (layered) first = z, count = d, z = 0, d = 1;
    first += view_layer(view);
    if (x < 0 || y < 0 || z < 0 || first < 0 || first + count > static_cast<int>(L.layers)) return false;
    // Recortada al nivel: el juego no debería pasarse, y escribir fuera pisaría el siguiente nivel.
    w = std::min<int>(w, static_cast<int>(lv.width) - x), h = std::min<int>(h, static_cast<int>(lv.height) - y);
    d = std::min<int>(d, static_cast<int>(std::max(lv.depth, 1u)) - z);
    if (w <= 0 || h <= 0 || d <= 0) return false;
    out.layer = static_cast<uint32_t>(first), out.layers = static_cast<uint32_t>(count);
    const uint32_t bw = L.block_w, bh = L.block_h;
    out.box = {x / bw, y / bh, static_cast<uint32_t>(z), (x % bw + w + bw - 1) / bw, (y % bh + h + bh - 1) / bh,
               static_cast<uint32_t>(d)};
    out.box.width = std::min(out.box.width, lv.width_blocks - out.box.x);
    out.box.height = std::min(out.box.height, lv.height_blocks - out.box.y);
    out.row_pitch = row_stride > 0 ? static_cast<size_t>(row_stride) : size_t{out.box.width} * out.bpb;
    out.image_pitch = image_stride > 0 ? static_cast<size_t>(image_stride) : out.row_pitch * out.box.height;
    return true;
}

// Una capa o un corte de lo que tocan: dónde empieza el nivel en la memoria de la textura.
uint8_t* level_storage(const nvn::Texture& t, const TexelRegion& r, uint32_t layer) {
    return t.storage + size_t{layer} * t.layout.layer_size + r.level->offset;
}

// Lineal con stride: filas tal cual. Block-linear: la caja (ver layout.h).
void write_texels(nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* region, const void* data,
                  int64_t row_stride, int64_t image_stride) {
    TexelRegion r;
    if (!data || !texel_region(t, view, region, row_stride, image_stride, r)) {
        static int logged = 0;
        if (nvn::first_times(logged, 8))
            NVN_LOG("WriteTexels fuera de la textura %dx%dx%d (%d,%d,%d %dx%dx%d) o sin memoria\n", t.desc.width,
                    t.desc.height, t.desc.depth, region ? region->x : 0, region ? region->y : 0, region ? region->z : 0,
                    region ? region->width : 0, region ? region->height : 0, region ? region->depth : 0);
        return;
    }
    const auto* src = static_cast<const uint8_t*>(data);
    for (uint32_t k = 0; k < r.layers; k++) {
        uint8_t* level = level_storage(t, r, r.layer + k);
        const uint8_t* in = src + k * r.image_pitch;
        if (t.layout.linear) {
            for (uint32_t y = 0; y < r.box.height; y++)
                memcpy(level + size_t{r.box.y + y} * t.layout.stride + size_t{r.box.x} * r.bpb, in + y * r.row_pitch,
                       size_t{r.box.width} * r.bpb);
        } else {
            nvn::swizzle_region(*r.level, r.bpb, r.box, in, r.row_pitch, r.image_pitch, level);
        }
    }
    // La imagen se rehace desde la memoria al usarla (solo esta caja si la GPU ya escribió el resto).
    std::lock_guard lock(nvn::texel_writes_mutex());
    t.texel_writes.push_back({r.level_index, r.layer, r.layers, r.box});
    nvn::g_texel_writes++;
}

// Las escrituras pendientes de una textura que se reinicia o se finaliza: fuera, y fuera de la cuenta (con ella a 0 el
// ejecutor ni toma el mutex).
void drop_texel_writes(nvn::Texture& t) {
    std::lock_guard lock(nvn::texel_writes_mutex());
    nvn::g_texel_writes -= static_cast<uint32_t>(t.texel_writes.size());
    t.texel_writes.clear();
}

void read_texels(const nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data,
                 int64_t row_stride, int64_t image_stride) {
    TexelRegion r;
    if (!data || !texel_region(t, view, region, row_stride, image_stride, r)) return;
    // Lo que escribió la GPU vive en la imagen de Vulkan, no en la memoria: se lee lo de la CPU.
    static int logged = 0;
    if (t.gpu_written && nvn::first_times(logged, 4))
        NVN_LOG("ReadTexels de una textura %dx%d escrita por la GPU: se lee su memoria (sin lo de la GPU)\n", t.desc.width,
                t.desc.height);
    auto* dst = static_cast<uint8_t*>(data);
    for (uint32_t k = 0; k < r.layers; k++) {
        const uint8_t* level = level_storage(t, r, r.layer + k);
        uint8_t* out = dst + k * r.image_pitch;
        if (t.layout.linear) {
            for (uint32_t y = 0; y < r.box.height; y++)
                memcpy(out + y * r.row_pitch, level + size_t{r.box.y + y} * t.layout.stride + size_t{r.box.x} * r.bpb,
                       size_t{r.box.width} * r.bpb);
        } else {
            nvn::deswizzle_region(*r.level, r.bpb, r.box, level, out, r.row_pitch, r.image_pitch);
        }
    }
}

// NVNtextureSparseTileLayout: niveles con tiles enteros y tamaño del tile en texels (siguen 4 int reservados, que no se
// tocan). Un tile son 64 KB de la disposición que usamos: bloques de 1 GOB de ancho (64 bytes) y block_height_gobs de alto
// (x block_depth), 128 GOBs por tile. Con las texturas sparse de MH Rise (bloques de 8 GOBs, pitch a 16) sale 1024 bytes x
// 64 filas, lo mismo que sus tiles.
struct SparseTileLayout {
    int num_tiled_levels, tile_width, tile_height, tile_depth;
};
void sparse_tile_layout(const nvn::TextureLayout& L, bool is_3d, SparseTileLayout* out) {
    if (!out) return;
    *out = {};
    if (L.linear || L.levels.empty()) return;  // las lineales no van por tiles
    const nvn::LevelLayout& l0 = L.levels[0];
    const uint32_t gobs_high = std::max(l0.block_height_gobs, 1u), deep = is_3d ? std::max(l0.block_depth, 1u) : 1u;
    const uint32_t width_bytes = 64 * std::max(128 / (gobs_high * deep), 1u);
    out->tile_width = static_cast<int>(std::max(width_bytes / std::max(L.bytes_per_block, 1u), 1u) * L.block_w);
    out->tile_height = static_cast<int>(8 * gobs_high * L.block_h);
    out->tile_depth = static_cast<int>(deep);
    for (const nvn::LevelLayout& l : L.levels) {
        if (l.width < static_cast<uint32_t>(out->tile_width) || l.height < static_cast<uint32_t>(out->tile_height) ||
            std::max(l.depth, 1u) < deep)
            break;
        out->num_tiled_levels++;
    }
}

// Getters del builder y del sampler (la misma descripción).
void sampler_min_mag(const SamplerDesc& d, int* min, int* mag) {
    if (min) *min = d.min_filter;
    if (mag) *mag = d.mag_filter;
}
void sampler_wrap(const SamplerDesc& d, int* s, int* t, int* r) {
    if (s) *s = d.wrap[0];
    if (t) *t = d.wrap[1];
    if (r) *r = d.wrap[2];
}
void sampler_lod_clamp(const SamplerDesc& d, float* min, float* max) {
    if (min) *min = d.lod_min;
    if (max) *max = d.lod_max;
}
void sampler_compare(const SamplerDesc& d, int* mode, int* func) {
    if (mode) *mode = d.compare_mode;
    if (func) *func = d.compare_func;
}
template <typename T>
void border_int(const SamplerDesc& d, T* color) {
    for (int k = 0; k < 4; k++) color[k] = static_cast<T>(d.border[k]);
}
const SamplerDesc& sampler_desc(const void* sampler) {
    static const SamplerDesc none{};
    nvn::Sampler* s = g_samplers.get(sampler);
    return s ? s->desc : none;
}

}  // namespace

void nvn::write_texels_to(Texture& t, const TextureViewDesc* view, const CopyRegion* region, const void* data,
                          int64_t row_stride, int64_t image_stride) {
    write_texels(t, view, region, data, row_stride, image_stride);
}
void nvn::read_texels_from(const Texture& t, const TextureViewDesc* view, const CopyRegion* region, void* data,
                           int64_t row_stride, int64_t image_stride) {
    read_texels(t, view, region, data, row_stride, image_stride);
}
bool nvn::texel_region_info(const Texture& t, const TextureViewDesc* view, const CopyRegion* region, int64_t row_stride,
                            int64_t image_stride, TexelRegionInfo& out) {
    TexelRegion r;
    if (!texel_region(t, view, region, row_stride, image_stride, r)) return false;
    out.level = r.level_index, out.layer = r.layer, out.layers = r.layers, out.bpb = r.bpb;
    out.x = r.box.x, out.y = r.box.y, out.z = r.box.z;
    out.width = r.box.width, out.height = r.box.height, out.depth = r.box.depth;
    out.row_pitch = r.row_pitch, out.image_pitch = r.image_pitch;
    return true;
}
size_t nvn::texel_span(const Texture& t, const TextureViewDesc* view, const CopyRegion* region, int64_t row_stride,
                       int64_t image_stride) {
    TexelRegion r;
    if (!texel_region(t, view, region, row_stride, image_stride, r)) return 0;
    const size_t slices = std::max(r.layers, r.box.depth);
    return (slices - 1) * r.image_pitch + (r.box.height - 1) * r.row_pitch + size_t{r.box.width} * r.bpb;
}

nvn::Objects<nvn::Texture> nvn::g_textures;
std::atomic<uint32_t> nvn::g_texel_writes{0};

namespace {
// Reproducción en diferido: cuándo suelta el juego sus texturas respecto a los comandos que las nombran (para el log).
struct TeardownStats {
    std::atomic<uint64_t> total{0}, pending{0}, unsent{0}, limbo{0}, copied_bytes{0};
    std::atomic<uint64_t> frames_after[5]{};  // frames del juego entre el último comando grabado que la nombra y soltarla
    // Del limbo (take_limbo_stats): lo máximo que llegó a tener, lo que soltaron sus válvulas y las finalizadas recién
    // creadas que no pasaron por él.
    std::atomic<uint64_t> max_count{0}, max_bytes{0}, forced_age{0}, forced_budget{0}, young{0};
    // Texturas finalizadas del todo en lo que va de ejecución (no se pone a cero): sus objetos no se borran nunca.
    std::atomic<uint64_t> kept{0};
    std::atomic<uint64_t> buffers{0};  // texturas de tipo BUFFER finalizadas y borradas
};
TeardownStats g_teardown;

// ¿Quedan comandos que la nombran sin reproducir? unsent: grabados en listas que el juego aún no ha enviado; pending: en un
// envío que la cola aún no ha reproducido. Apunta las cuentas del log.
struct Use {
    bool unsent, pending;
};
Use pending_use(const nvn::Texture& t, const char* what) {
    g_teardown.total.fetch_add(1, std::memory_order_relaxed);
    const uint64_t last = __atomic_load_n(&t.last_ticket, __ATOMIC_ACQUIRE);
    const int open = __atomic_load_n(&t.open_refs, __ATOMIC_RELAXED);
    if (last || open > 0) {
        const uint32_t frames = nvn::game_frame() - __atomic_load_n(&t.record_frame, __ATOMIC_RELAXED);
        g_teardown.frames_after[std::min<uint32_t>(frames, 4)].fetch_add(1, std::memory_order_relaxed);
    }
    const Use use{open > 0, last > nvn::replayed_ticket()};
    if (use.pending) g_teardown.pending.fetch_add(1, std::memory_order_relaxed);
    if (use.unsent) {
        g_teardown.unsent.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<int> logged{0};
        if (nvn::first_times(logged, 8))
            NVN_LOG("%s de una textura %dx%dx%d formato 0x%x con %d comandos grabados sin enviar que la nombran\n", what,
                    t.desc.width, t.desc.height, t.desc.depth, t.desc.format, open);
    }
    return use;
}

// El final de nvnTextureFinalize, con executor_mutex tomado: la imagen se libera, pero el objeto queda vivo (sin imagen: se
// muestrea la de relleno) por si algún pool de texturas aún lo nombra.
void finish_finalize(std::unique_ptr<nvn::Texture> t) {
    nvn::release_texture_gpu(*t);
    drop_texel_writes(*t);
    // Un buffer registrado como textura no lo nombra nadie: las entradas del pool tienen su copia (buffer_objects) y los
    // comandos no los llevan por puntero. Se borra. Conservados, Unreal Engine 5 dejaba 32 000 objetos por segundo (mil
    // por frame): 1,2 GB en cuatro minutos y Android mataba el juego por memoria en la cinemática inicial.
    if (t->desc.target == nvn::kTargetBuffer && !NVN_FLAG("buffers_por_objeto")) {
        g_teardown.buffers.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    t->serial = nvn::next_texture_serial();
    t->storage = nullptr;
    t->vk_failed = true;
    static std::mutex m;
    static std::vector<std::unique_ptr<nvn::Texture>> finalized;
    std::lock_guard lock(m);
    finalized.push_back(std::move(t));
    g_teardown.kept.fetch_add(1, std::memory_order_relaxed);
}

// Limbo: texturas que el juego ya finalizó y que comandos grabados antes aún nombran por puntero (su copia a otra textura,
// el pase que las pinta). En la consola esos comandos llevan la dirección de sus texels y funcionan mientras el juego no
// reutilice esa memoria; aquí necesitan el objeto y su imagen. Siguen enteras hasta que la cola reproduce el último envío
// que las nombra (reap_limbo_textures) y entonces se finalizan de verdad. Disco Elysium: la textura temporal de cada página
// de AmplifyTexture (WriteTexels, copia a la caché, Finalize un frame después con la copia aún en la cola: media
// habitación con teselas negras) y render targets finalizados en el mismo frame en que se pintan, antes de enviar la lista.
// `texels`: copia de su memoria cuando le quedaban texels de la CPU sin subir (el juego puede liberar o reutilizar la suya
// al volver de Finalize; Unity la guarda aún 3 presentaciones, main+0x9920a8 y main+0x9915e8, pero no se cuenta con
// ello); la textura lee de la copia.
// Además, cualquier textura con imagen pasa por aquí aunque ningún comando la nombre por puntero: los draws la muestrean
// por su id del pool de texturas, que no se puede sellar al grabar, y Unity finaliza texturas en el mismo frame en que las
// usa, antes de enviar la lista. Esas esperan a que acabe ese frame del juego y la cola lo reproduzca (`frame`).
// <datos>/limbo_solo_nombradas (en vivo): solo las que nombra un comando sin reproducir.
struct Limbo {
    std::unique_ptr<nvn::Texture> texture;
    std::unique_ptr<uint8_t[]> texels;
    uint32_t frame = 0;     // frame del juego en que se finalizó
    uint64_t ticket = 0;    // el último envío del juego encolado entonces
    size_t copy_bytes = 0;  // lo que ocupa texels
};
std::mutex g_limbo_mutex;
std::vector<Limbo> g_limbo;
std::atomic<size_t> g_limbo_count{0};
constexpr size_t kLimboCopyMax = size_t{64} << 20;
// Válvulas del limbo (reap_limbo_textures). Edad: una lista grabada que nombra la textura y no se ha enviado ni reciclado en
// tantos frames ya no se va a enviar (Unity recicla la memoria de control de sus listas cada 4 frames: el anillo de
// main+0x98dfd4, `and x9, x9, #0x3`); sin esto, una lista que el juego no recicla nunca dejaba su textura (y su imagen)
// en el limbo para siempre. Tope: lo que esperan las del limbo es a que acabe el frame del juego en que se finalizaron, y
// sin PresentTexture (una carga que no presenta) no acaba: todo lo que el juego suelte mientras, con sus imágenes y las
// copias de sus texels, se quedaría aquí hasta el siguiente frame, con la memoria del sistema tan justa como va (el
// lowmemorykiller mata el proceso en primer plano). Pasado el tope se sueltan, las más antiguas primero, las que ya solo
// esperan al final del frame y tras las que se ha reproducido algún envío posterior (la lista que se grababa al
// finalizarlas ya salió: Unity envía la suya en cada vaciado). <datos>/limbo_sin_valvulas (en vivo): sin ninguna de las dos.
constexpr uint32_t kLimboMaxFrames = 16;
constexpr uint64_t kLimboBudget = uint64_t{256} << 20;

// ¿Se puede leer [p, p + size) como memoria de un pool vivo? El juego puede finalizar el pool (y liberar su memoria) antes
// que la textura: Unity no lo hace (main+0x991f1c y main+0x992094: nvnTextureFinalize y después el bloque de memoria, que
// en el segundo caso además espera 3 presentaciones en su lista de borrado diferido), pero copiar de memoria ya devuelta
// tiraría el juego.
bool pool_memory(const nvn::Texture& t, const uint8_t* p, size_t size) {
    const nvn::MemoryPool* pool = nvn::g_memory_pools.get(t.desc.pool);
    return pool && pool->storage && p >= pool->storage && size <= pool->size &&
           static_cast<size_t>(p - pool->storage) <= pool->size - size;
}

// Con executor_mutex tomado (la cola no está reproduciendo nada). named: algún comando sin reproducir la nombra por puntero
// (puede ser el origen de una copia: sus texels de la CPU tienen que llegar a subirse). ticket: el último envío encolado.
void enter_limbo(std::unique_ptr<nvn::Texture> t, bool named, uint64_t ticket) {
    Limbo l;
    l.frame = nvn::game_frame();
    l.ticket = ticket;
    bool pending_writes = false;
    {
        std::lock_guard lock(nvn::texel_writes_mutex());
        pending_writes = !t->texel_writes.empty();
    }
    // Texels de la CPU que la imagen aún no tiene y que lo pendiente puede necesitar: los de una textura nombrada que no
    // se ha subido nunca (el origen de una copia) y las cajas de WriteTexels sin subir de cualquiera (los draws que la
    // muestrean por id esperan verlas).
    const bool cpu_texels = pending_writes || (named && !t->uploaded && !t->gpu_written);
    const size_t size = t->layout.total_size();
    if (t->storage && cpu_texels && size && size <= kLimboCopyMax && pool_memory(*t, t->storage, size)) {
        l.texels.reset(new uint8_t[size]);
        memcpy(l.texels.get(), t->storage, size);
        nvn::forget_texture_storage(*t);
        t->storage = l.texels.get();
        l.copy_bytes = size;
        g_teardown.copied_bytes.fetch_add(size, std::memory_order_relaxed);
    } else {
        // Sin copia, las cajas de WriteTexels que quedaran por subir se leerían de la memoria del juego cuando ya no es
        // suya (Executor::apply_texel_writes lee storage): fuera.
        if (pending_writes) drop_texel_writes(*t);
        // Ya subida (o demasiado grande para copiarla): su memoria no se vuelve a mirar. La de un render target se deja:
        // no se lee (la imagen manda) y es la clave de sus alias (Executor::alias_to_target), que se van al retirarla.
        if (!t->gpu_written) {
            nvn::forget_texture_storage(*t);
            t->storage = nullptr;
        }
    }
    t->watched = false, t->no_watch = true;  // la vigilancia es sobre la memoria del juego
    t->limbo = true;
    l.texture = std::move(t);
    g_teardown.limbo.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(g_limbo_mutex);
    g_limbo.push_back(std::move(l));
    g_limbo_count.store(g_limbo.size(), std::memory_order_release);
    nvn::note_max(g_teardown.max_count, g_limbo.size());
}
}  // namespace

// Tras reproducir un envío (el hilo de la cola, sin cerrojos tomados): las del limbo que ya no nombra nada pendiente.
void nvn::reap_limbo_textures(uint64_t replayed) {
    if (!g_limbo_count.load(std::memory_order_acquire)) return;
    const bool valves = !NVN_FLAG("limbo_sin_valvulas");
    const uint32_t now = nvn::game_frame();
    std::vector<Limbo> ready;
    {
        std::lock_guard lock(g_limbo_mutex);
        // Sin comandos pendientes que la nombren: open_refs antes que last_ticket (quien envía la lista pone el ticket y
        // después deja de contarla: SubmitQueue::submit).
        auto quiet = [&](const nvn::Texture& t) {
            return __atomic_load_n(&t.open_refs, __ATOMIC_ACQUIRE) <= 0 &&
                   __atomic_load_n(&t.last_ticket, __ATOMIC_ACQUIRE) <= replayed;
        };
        auto take = [&](size_t k) {
            ready.push_back(std::move(g_limbo[k]));
            g_limbo[k] = std::move(g_limbo.back());
            g_limbo.pop_back();
        };
        uint64_t bytes = 0;  // lo que se queda: imágenes (las crea y las suelta este hilo) y copias de texels
        for (size_t k = 0; k < g_limbo.size();) {
            const Limbo& l = g_limbo[k];
            const nvn::Texture& t = *l.texture;
            const bool sent = __atomic_load_n(&t.last_ticket, __ATOMIC_ACQUIRE) <= replayed;
            if (quiet(t) && nvn::game_frame_replayed(l.frame, replayed)) {
                take(k);
            } else if (valves && sent && now - l.frame > kLimboMaxFrames) {
                g_teardown.forced_age.fetch_add(1, std::memory_order_relaxed);
                take(k);
            } else {
                bytes += l.copy_bytes + t.gpu_bytes;
                k++;
            }
        }
        nvn::note_max(g_teardown.max_bytes, bytes);
        for (size_t k = 0; valves && bytes > kLimboBudget && k < g_limbo.size();) {
            const Limbo& l = g_limbo[k];
            if (quiet(*l.texture) && l.ticket < replayed) {
                bytes -= l.copy_bytes + l.texture->gpu_bytes;
                g_teardown.forced_budget.fetch_add(1, std::memory_order_relaxed);
                take(k);
            } else {
                k++;
            }
        }
        g_limbo_count.store(g_limbo.size(), std::memory_order_release);
    }
    if (ready.empty()) return;
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex());
    for (Limbo& l : ready) finish_finalize(std::move(l.texture));  // l.texels se libera con `ready`
}
size_t nvn::limbo_textures() { return g_limbo_count.load(std::memory_order_relaxed); }

void nvn::take_limbo_stats(uint64_t& max_count, uint64_t& max_bytes, uint64_t& forced_age, uint64_t& forced_budget,
                           uint64_t& young, uint64_t& kept_total) {
    max_count = g_teardown.max_count.exchange(0, std::memory_order_relaxed);
    max_bytes = g_teardown.max_bytes.exchange(0, std::memory_order_relaxed);
    forced_age = g_teardown.forced_age.exchange(0, std::memory_order_relaxed);
    forced_budget = g_teardown.forced_budget.exchange(0, std::memory_order_relaxed);
    young = g_teardown.young.exchange(0, std::memory_order_relaxed);
    kept_total = g_teardown.kept.load(std::memory_order_relaxed);
}

// Para el log de ritmo (window.cpp): desde la última vez.
void nvn::take_teardown_stats(uint64_t& total, uint64_t& pending, uint64_t& unsent, uint64_t& limbo, uint64_t& copied_bytes,
                              uint64_t frames_after[5]) {
    total = g_teardown.total.exchange(0, std::memory_order_relaxed);
    pending = g_teardown.pending.exchange(0, std::memory_order_relaxed);
    unsent = g_teardown.unsent.exchange(0, std::memory_order_relaxed);
    limbo = g_teardown.limbo.exchange(0, std::memory_order_relaxed);
    copied_bytes = g_teardown.copied_bytes.exchange(0, std::memory_order_relaxed);
    for (int k = 0; k < 5; k++) frames_after[k] = g_teardown.frames_after[k].exchange(0, std::memory_order_relaxed);
}

std::mutex& nvn::texel_writes_mutex() {
    static std::mutex m;
    return m;
}
std::atomic<uint64_t> nvn::g_texture_gpu_bytes{0}, nvn::g_texture_gpu_images{0};

uint64_t nvn::next_texture_serial() {
    static std::atomic<uint64_t> serial{1};
    return serial++;
}

// Destruirla ya podía pillar a la GPU leyéndola en un envío pendiente: la retira el ejecutor.
void nvn::release_texture_gpu(Texture& t) { nvn::retire_texture_gpu(t); }
nvn::Objects<nvn::DescriptorPool> nvn::g_texture_pools, nvn::g_sampler_pools;

nvn::Sampler* nvn::find_sampler(const void* sampler) { return g_samplers.get(sampler); }

extern "C" {

// --- Builder de texturas ---

void nvnTextureBuilderSetDefaults(TextureDesc* b) {
    memset(b, 0, sizeof *b);
    b->target = nvn::kTarget2D;
    b->width = b->height = b->depth = b->levels = 1;
    b->swizzle[0] = 2, b->swizzle[1] = 3, b->swizzle[2] = 4, b->swizzle[3] = 5;  // R G B A
}
void nvnTextureBuilderSetDevice(TextureDesc* b, void* device) { b->device = device; }
void nvnTextureBuilderSetFlags(TextureDesc* b, int flags) { b->flags = flags; }
void nvnTextureBuilderSetTarget(TextureDesc* b, int target) { b->target = target; }
void nvnTextureBuilderSetFormat(TextureDesc* b, int format) { b->format = format; }
void nvnTextureBuilderSetLevels(TextureDesc* b, int levels) { b->levels = levels; }
void nvnTextureBuilderSetSamples(TextureDesc* b, int samples) { b->samples = samples; }
void nvnTextureBuilderSetStride(TextureDesc* b, int64_t stride) { b->stride = static_cast<int>(stride); }
void nvnTextureBuilderSetWidth(TextureDesc* b, int w) { b->width = w; }
void nvnTextureBuilderSetHeight(TextureDesc* b, int h) { b->height = h; }
void nvnTextureBuilderSetDepth(TextureDesc* b, int d) { b->depth = d; }
void nvnTextureBuilderSetSize1D(TextureDesc* b, int w) { b->width = w, b->height = b->depth = 1; }
void nvnTextureBuilderSetSize2D(TextureDesc* b, int w, int h) { b->width = w, b->height = h, b->depth = 1; }
void nvnTextureBuilderSetSize3D(TextureDesc* b, int w, int h, int d) { b->width = w, b->height = h, b->depth = d; }
void nvnTextureBuilderSetStorage(TextureDesc* b, void* pool, int64_t offset) {
    b->pool = pool;
    b->offset = offset;
}
// El juego reserva esto y escribe ahí los texels en disposición Tegra: tiene que ser el tamaño real.
size_t nvnTextureBuilderGetStorageSize(const TextureDesc* b) {
    size_t size = (layout_of(*b).total_size() + kStorageAlignment - 1) & ~(kStorageAlignment - 1);
    static int logged_big = 0;
    if (size > (4u << 20) && nvn::first_times(logged_big, 40))
        NVN_LOG("GetStorageSize %zu KB para %dx%dx%d formato 0x%x target %d niveles %d flags 0x%x\n", size >> 10, b->width,
                b->height, b->depth, b->format, b->target, b->levels, b->flags);
    if (size > (64u << 20))
        NVN_LOG("tamaño de textura enorme: %zu MB para %dx%dx%d formato 0x%x target %d niveles %d muestras %d flags 0x%x\n",
                size >> 20, b->width, b->height, b->depth, b->format, b->target, b->levels, b->samples, b->flags);
    return size;
}
// La alineación que pide la memoria de una textura: 512 bytes para todas. Es la de la Switch para las block-linear y
// más que la de las lineales; las sparse de MH Rise funcionan con ella (MapVirtual va por páginas de 4 KB del heap).
// Xenoblade la pide en cada reserva de texturas (55 sitios): mejor no moverla.
// Las lineales (flags LINEAR 0x10 o LINEAR_RENDER_TARGET 0x40) piden menos: Catherine: Full Body reserva cada textura
// con su propio allocator (bloques de 0x80 en 0x80, main+0x2ec080), le da a NVN el offset del bloque redondeado a esta
// alineación (main+0x3139d0) y luego escribe los texels desde el PRINCIPIO del bloque (su "lock", main+0x308480, sin
// Flush). Con 512 su atlas de glifos (1024x1024 RGBA8 lineal) quedaba 0x80 bytes por delante de donde el juego pinta:
// cada letra salía 32 píxeles corrida (la de la casilla de al lado). En la consola no pasa, así que la alineación de
// una lineal divide a 0x80; se da la del paso de fila (LINEAR_TEXTURE_STRIDE_ALIGNMENT, 32).
size_t nvnTextureBuilderGetStorageAlignment(const TextureDesc* b) {
    constexpr int kLinear = 0x10, kLinearRenderTarget = 0x40;
    if (b && (b->flags & (kLinear | kLinearRenderTarget))) return 32;
    return kStorageAlignment;
}
// Lo que se muestrea de una textura de profundidad+stencil si la vista no lo dice (MH Rise lo pone a 0, DEPTH).
void nvnTextureBuilderSetDepthStencilMode(TextureDesc* b, int mode) { b->depth_stencil_mode = static_cast<uint8_t>(mode); }
// La disposición del empaquetador (NVNpackagedTextureLayout, 8 bytes opacos): se guarda y se registra con lo que
// calculamos nosotros, para ver si coinciden.
void nvnTextureBuilderSetPackagedTextureLayout(TextureDesc* b, const uint8_t* layout) {
    b->has_packaged_layout = layout != nullptr;
    if (layout) memcpy(b->packaged_layout, layout, 8);
}
const void* nvnTextureBuilderGetPackagedTextureLayout(const TextureDesc* b) {
    return b->has_packaged_layout ? b->packaged_layout : nullptr;
}
void nvnTextureBuilderSetPackagedTextureData(TextureDesc* b, const void* data) { b->packaged_data = data; }
const void* nvnTextureBuilderGetPackagedTextureData(const TextureDesc* b) { return b->packaged_data; }
void nvnTextureBuilderSetSwizzle(TextureDesc* b, int r, int g, int bl, int a) {
    {  // Los swizzles distintos de RGBA que pide el juego (Unity: ARGB32 -> 3 4 5 2, el _VTPageTableDesc de AmplifyTexture)
        static std::mutex m;
        static int seen[16][4], logged = 0;
        std::lock_guard<std::mutex> lock(m);
        bool fresh = !(r == 2 && g == 3 && bl == 4 && a == 5);
        for (int k = 0; k < std::min(logged, 16) && fresh; k++)
            if (seen[k][0] == r && seen[k][1] == g && seen[k][2] == bl && seen[k][3] == a) fresh = false;
        if (fresh && logged < 16) {
            seen[logged][0] = r, seen[logged][1] = g, seen[logged][2] = bl, seen[logged][3] = a;
            logged++;
            NVN_LOG("swizzle de builder: %d %d %d %d (formato 0x%x %dx%d)\n", r, g, bl, a, b->format, b->width, b->height);
        }
    }
    b->swizzle[0] = r, b->swizzle[1] = g, b->swizzle[2] = bl, b->swizzle[3] = a;
}
int nvnTextureBuilderGetStorageClass(const TextureDesc* b) { return storage_class_of(*b); }
int nvnTextureBuilderGetRawStorageClass(const TextureDesc* b) { return storage_class_of(*b); }
// Nombre de una textura de OpenGL de la que tomar la memoria (interoperabilidad con GL): en Android no hay GL del juego;
// se guarda para el getter y Initialize sigue usando el storage del builder.
void nvnTextureBuilderSetGLTextureName(TextureDesc* b, uint32_t name) { b->gl_texture_name = name; }
uint32_t nvnTextureBuilderGetGLTextureName(const TextureDesc* b) { return b->gl_texture_name; }
void nvnTextureBuilderGetSparseTileLayout(const TextureDesc* b, void* out) {
    sparse_tile_layout(layout_of(*b), b->target == nvn::kTarget3D, static_cast<SparseTileLayout*>(out));
}
// Sin ZCull (la GPU lo hace a su manera): nada que reservar.
size_t nvnTextureBuilderGetZCullStorageSize(const TextureDesc*) { return 0; }
void* nvnTextureBuilderGetDevice(const TextureDesc* b) { return b->device; }
int nvnTextureBuilderGetFlags(const TextureDesc* b) { return b->flags; }
int nvnTextureBuilderGetTarget(const TextureDesc* b) { return b->target; }
int nvnTextureBuilderGetWidth(const TextureDesc* b) { return b->width; }
int nvnTextureBuilderGetHeight(const TextureDesc* b) { return b->height; }
int nvnTextureBuilderGetDepth(const TextureDesc* b) { return b->depth; }
int nvnTextureBuilderGetLevels(const TextureDesc* b) { return b->levels; }
int nvnTextureBuilderGetFormat(const TextureDesc* b) { return b->format; }
int nvnTextureBuilderGetSamples(const TextureDesc* b) { return b->samples; }
int64_t nvnTextureBuilderGetStride(const TextureDesc* b) { return b->stride; }
void* nvnTextureBuilderGetMemoryPool(const TextureDesc* b) { return b->pool; }
int64_t nvnTextureBuilderGetMemoryOffset(const TextureDesc* b) { return b->offset; }
int nvnTextureBuilderGetDepthStencilMode(const TextureDesc* b) { return b->depth_stencil_mode; }
void nvnTextureBuilderGetSwizzle(const TextureDesc* b, int* r, int* g, int* bl, int* a) {
    if (r) *r = b->swizzle[0];
    if (g) *g = b->swizzle[1];
    if (bl) *bl = b->swizzle[2];
    if (a) *a = b->swizzle[3];
}

// --- Texturas ---

nvn::Boolean nvnTextureInitialize(void* texture, const TextureDesc* b) {
    // Reinicializar sin Finalize: el mismo objeto (los comandos grabados y los pools de texturas guardan su
    // dirección), sin la imagen anterior.
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::Texture* existing = nvn::g_textures.get(texture);
    if (existing) {
        // Lo ya enviado que la nombra, primero (el objeto es el mismo: no puede ir al limbo como en Finalize).
        if (pending_use(*existing, "nvnTextureInitialize sobre una textura viva").pending)
            nvn::replay_barrier(0, "nvnTextureInitialize sobre una textura viva");
        nvn::lock_executor(lock);  // el ejecutor puede estar leyendo este objeto
        nvn::release_texture_gpu(*existing);
        drop_texel_writes(*existing);
        // Lo que llevan las listas ya grabadas que nombran este objeto sigue valiendo para el nuevo (es el mismo puntero):
        // a cero, cada lista sin enviar que lo nombraba dejaba open_refs en negativo al enviarse (y para siempre), y la
        // siguiente vez que se finalizara con una copia grabada sin enviar no se vería (la copia, sin origen).
        const uint64_t last_ticket = __atomic_load_n(&existing->last_ticket, __ATOMIC_ACQUIRE);
        const int32_t open_refs = __atomic_load_n(&existing->open_refs, __ATOMIC_ACQUIRE);
        const uint32_t record_frame = __atomic_load_n(&existing->record_frame, __ATOMIC_RELAXED);
        *existing = nvn::Texture{};
        __atomic_store_n(&existing->last_ticket, last_ticket, __ATOMIC_RELEASE);
        __atomic_store_n(&existing->open_refs, open_refs, __ATOMIC_RELEASE);
        __atomic_store_n(&existing->record_frame, record_frame, __ATOMIC_RELAXED);
    }
    nvn::Texture& t = existing ? *existing : nvn::g_textures.create(texture);
    t.serial = nvn::next_texture_serial();
    t.init_frame = nvn::game_frame();
    {  // Marca en el NVNtexture del juego (el driver de verdad también lo rellena): si el juego copia el struct, la copia
       // lleva al original (cmdbuf.cpp texture_of).
        const uint64_t tag[2] = {nvn::kTextureTag, reinterpret_cast<uint64_t>(texture)};
        memcpy(texture, tag, sizeof tag);
    }
    t.desc = *b;
    if (!nvn::format_info(b->format) || !nvn::format_info(b->format)->bytes)
        NVN_LOG("textura con formato 0x%x sin mapear: %dx%dx%d target %d niveles %d flags 0x%x\n", b->format, b->width,
                b->height, b->depth, b->target, b->levels, b->flags);
    t.layout = layout_of(*b);
    if (b->target == nvn::kTargetBuffer) {
        // Un buffer visto como textura (texel buffer). Unity lo registra con un NVNtexture temporal en la pila que finaliza
        // nada más registrarlo en el pool de texturas (main+0x993780 Initialize, main+0x99379c y main+0x9937ac
        // RegisterTexture y RegisterImage, main+0x9937bc Finalize): en NVN el descriptor del pool sigue valiendo sin el
        // objeto. Aquí la entrada del pool se queda con su copia (nvnTexturePoolRegisterTexture) y el ejecutor lo enlaza
        // como texel buffer (texel_binding).
        static int logged = 0;
        if (nvn::first_times(logged, 4))
            NVN_LOG("textura de tipo BUFFER (un buffer registrado como textura): %d texels formato 0x%x flags 0x%x\n",
                    b->width, b->format, b->flags);
    }
    if (b->flags & 0x20) {  // DEPURACIÓN: texturas sparse (streaming): ¿cómo las crea y re-crea?
        static int logged = 0;
        if (nvn::first_times(logged, 200))
            NVN_LOG("textura sparse %p%s: %dx%d formato 0x%x niveles %d pool %p +0x%llx -> %p\n", texture,
                    existing ? " (re-init)" : "", b->width, b->height, b->format, b->levels, b->pool,
                    static_cast<unsigned long long>(b->offset), static_cast<void*>(nvn::g_memory_pools.get(b->pool) ?
                    nvn::g_memory_pools.get(b->pool)->storage + b->offset : nullptr));
    }
    if (b->has_packaged_layout) {  // DEPURACIÓN: qué dice el empaquetador frente a nuestros bloques
        static int logged = 0;
        if (nvn::first_times(logged, 400)) {
            const auto& l0 = t.layout.levels.empty() ? nvn::LevelLayout{} : t.layout.levels[0];
            const uint8_t* p = b->packaged_layout;
            NVN_LOG("layout empaquetado %02x %02x %02x %02x %02x %02x %02x %02x: %dx%dx%d target %d formato 0x%x niveles %d "
                    "flags 0x%x; nuestro bloque %ux%u GOBs, pitch %u GOBs, %zu KB, %s\n", p[0], p[1], p[2], p[3], p[4], p[5],
                    p[6], p[7], b->width, b->height, b->depth, b->target, b->format, b->levels, b->flags,
                    l0.block_height_gobs, l0.block_depth, l0.row_gobs, t.layout.total_size() >> 10,
                    t.layout.linear ? "lineal" : "block-linear");
        }
    }
    if (nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool)) {
        t.storage = pool->storage + b->offset;
        t.cpu_storage = t.storage;
    } else {  // DEPURACIÓN (miniatura de la partida): texturas cuyo pool no conocemos (quedan sin memoria de CPU)
        static int logged = 0;
        if (nvn::first_times(logged, 16) || NVN_FLAG("diag_miniatura"))
            NVN_LOG("textura %p sin pool conocido: pool %p offset %lld, %dx%dx%d target %d formato 0x%x niveles %d flags 0x%x\n",
                    texture, b->pool, static_cast<long long>(b->offset), b->width, b->height, b->depth, b->target, b->format,
                    b->levels, b->flags);
    }
    // DEPURACIÓN (<datos>/log_texturas): las texturas grandes al inicializarse (la del vídeo de Xenoblade: ¿qué memoria usa?).
    if (b->width * b->height >= 256 * 256 && NVN_FLAG("log_texturas")) {
        static int logged = 0;
        if (nvn::first_times(logged, 400))
            NVN_LOG("textura nueva %p: %dx%dx%d target %d formato 0x%x niveles %d flags 0x%x stride %d storage %p\n", texture,
                    b->width, b->height, b->depth, b->target, b->format, b->levels, b->flags, b->stride,
                    static_cast<void*>(t.storage));
    }
    // DEPURACIÓN: dos texturas sobre la misma memoria (alias). En la Switch comparten texels; aquí cada una tiene su
    // imagen de Vulkan, así que lo que la GPU pinte en una no se ve en la otra.
    if (t.storage && b->target != nvn::kTargetBuffer) {  // (los buffers no: mil por frame, y no tienen imagen)
        static std::mutex m;
        static std::unordered_map<const uint8_t*, std::pair<const void*, int>> by_storage;  // storage -> (objeto, formato)
        std::lock_guard lock(m);
        auto [it, inserted] = by_storage.try_emplace(t.storage, texture, b->format);
        if (!inserted && it->second.first != texture) {
            static int logged = 0;
            if (nvn::first_times(logged, 40))
                NVN_LOG("alias de textura: %p (%dx%d formato 0x%x ds %d) sobre la memoria %p de %p (formato 0x%x)\n", texture,
                        b->width, b->height, b->format, b->depth_stencil_mode, static_cast<void*>(t.storage),
                        it->second.first, it->second.second);
            it->second = {texture, b->format};
        }
    }
    // Datos empaquetados: ya vienen con la disposición de la textura en memoria (la que calcula layout_of con el layout
    // del empaquetador); se copian enteros a su sitio en el pool, como hace el driver al inicializarla.
    if (b->packaged_data && t.storage) {
        const size_t size = t.layout.total_size();
        memcpy(t.storage, b->packaged_data, size);
        static int logged = 0;
        if (nvn::first_times(logged, 20))
            NVN_LOG("textura %p con datos empaquetados: %dx%d formato 0x%x niveles %d, %zu KB copiados\n", texture,
                    b->width, b->height, b->format, b->levels, size >> 10);
    }
    if ((b->width == 1600 && b->height == 904) || (b->width == 800 && b->height == 452))  // DEPURACIÓN: vídeo
        NVN_LOG("textura de vídeo %p: %dx%d formato 0x%x flags 0x%x stride %d pool %p +0x%llx -> %p\n", texture, b->width,
                b->height, b->format, b->flags, b->stride, b->pool, static_cast<unsigned long long>(b->offset),
                static_cast<void*>(t.storage));
    return 1;
}
void nvnTextureFinalize(void* texture) {
    if (const nvn::Texture* t = nvn::g_textures.get(texture); t && t->desc.flags == nvn::kTextureFlagImage) {  // DEPURACIÓN
        static int logged = 0;
        if (nvn::first_times(logged, 8) || NVN_FLAG("diag_miniatura"))
            NVN_LOG("finaliza textura %dx%d formato 0x%x flags 0x%x storage %p\n", t->desc.width, t->desc.height, t->desc.format,
                    t->desc.flags, static_cast<const void*>(t->storage));
    }
    // Con comandos que aún la nombran sin reproducir (enviados o no), al limbo: sigue entera hasta que se reproducen.
    // <datos>/sin_limbo_texturas (en vivo): como antes, se suelta ya (esos comandos se reproducen sin ella).
    nvn::Texture* live = nvn::g_textures.get(texture);
    if (!live) return;
    // Un buffer registrado como textura: sin imagen ni estado en el ejecutor, y nadie lo nombra (ver finish_finalize). Se
    // borra sin tomar el cerrojo del ejecutor: Unreal Engine 5 finaliza mil por frame y cada uno esperaba al envío en curso.
    if (live->desc.target == nvn::kTargetBuffer && !NVN_FLAG("buffers_por_objeto")) {
        nvn::g_textures.take(texture);
        g_teardown.buffers.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const Use use = pending_use(*live, "nvnTextureFinalize");
    const uint64_t ticket = nvn::last_submit();  // antes del cerrojo del ejecutor (toma el de la cola)
    std::unique_lock<std::recursive_mutex> executor_lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(executor_lock);
    std::unique_ptr<nvn::Texture> t = nvn::g_textures.take(texture);
    if (!t) return;
    bool named = use.unsent || use.pending;
    // Recién creada, sin imagen y sin que ningún comando la nombre: si algún draw ya grabado la muestrea por su id, la cola
    // aún no ha llegado a él (por eso no tiene imagen) y se quedaría con la de relleno. En la consola ese draw funciona:
    // Unity finaliza el objeto al momento (main+0x992094) pero no devuelve su memoria ni su id del pool de texturas hasta 3
    // presentaciones después (los apunta en main+0x9920a8 con la marca 'NTRE' y los suelta main+0x9915e8, que avanza una
    // de sus 3 listas por PresentTexture: main+0x94c81c). Se cuentan siempre (log de ritmo), y con
    // <datos>/limbo_recientes (en vivo, apagado por defecto) van al limbo con sus texels copiados, como las nombradas.
    // (Las de tipo BUFFER no cuentan: son las temporales con que Unity registra un buffer en el pool, ver Initialize.)
    if (!named && !t->image && !t->gpu_written && t->storage && t->desc.target != nvn::kTargetBuffer &&
        nvn::game_frame() - t->init_frame <= 1) {
        g_teardown.young.fetch_add(1, std::memory_order_relaxed);
        if (NVN_FLAG("limbo_recientes")) named = true;
    }
    if ((named || (t->image && !NVN_FLAG("limbo_solo_nombradas"))) && !NVN_FLAG("sin_limbo_texturas"))
        enter_limbo(std::move(t), named, ticket);
    else
        finish_finalize(std::move(t));
}

#define TEXTURE_GETTER(name, field)                          \
    int nvnTextureGet##name(const void* t) {                \
        nvn::Texture* tex = nvn::texture_lookup(t);         \
        return tex ? tex->desc.field : 0;                   \
    }
TEXTURE_GETTER(Width, width)
TEXTURE_GETTER(Height, height)
TEXTURE_GETTER(Depth, depth)
TEXTURE_GETTER(Format, format)
TEXTURE_GETTER(Target, target)
TEXTURE_GETTER(Levels, levels)
TEXTURE_GETTER(Flags, flags)
TEXTURE_GETTER(Samples, samples)
TEXTURE_GETTER(DepthStencilMode, depth_stencil_mode)
#undef TEXTURE_GETTER
int nvnTextureGetStorageClass(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? storage_class_of(tex->desc) : kStorageClassBlockLinear;
}
int nvnTextureGetRawStorageClass(const void* t) { return nvnTextureGetStorageClass(t); }
size_t nvnTextureGetZCullStorageSize(const void*) { return 0; }
void* nvnTextureGetMemoryPool(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? tex->desc.pool : nullptr;
}
int64_t nvnTextureGetMemoryOffset(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? tex->desc.offset : 0;
}
// Dirección de GPU = de CPU: la de sus texels.
uint64_t nvnTextureGetTextureAddress(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? reinterpret_cast<uint64_t>(tex->storage) : 0;
}
void nvnTextureGetSwizzle(const void* t, int* r, int* g, int* b, int* a) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    const uint8_t identity[4] = {2, 3, 4, 5};
    const uint8_t* s = tex ? tex->desc.swizzle : identity;
    if (r) *r = s[0];
    if (g) *g = s[1];
    if (b) *b = s[2];
    if (a) *a = s[3];
}
int nvnTextureGetDebugID(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? static_cast<int>(tex->serial) : 0;
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnTextureSetDebugLabel(void*, const char*) {}
void nvnTextureGetSparseTileLayout(const void* t, void* out) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    if (!tex) {
        if (out) *static_cast<SparseTileLayout*>(out) = {};
        return;
    }
    sparse_tile_layout(tex->layout, tex->desc.target == nvn::kTarget3D, static_cast<SparseTileLayout*>(out));
}
// La superficie de NvRm (el driver de Tegra) de la textura: en Android no existe NvRm; no hay nada que dar y la salida
// se deja como está.
void nvnTextureGetNvRmSurfaceNVX(const void*, void*) {}
// Dos texturas iguales: la misma memoria con la misma descripción (lo que el hardware vería igual).
nvn::Boolean nvnTextureCompare(const void* a, const void* b) {
    if (a == b) return 1;
    const nvn::Texture* x = nvn::texture_lookup(a);
    const nvn::Texture* y = nvn::texture_lookup(b);
    if (!x || !y) return 0;
    const TextureDesc &p = x->desc, &q = y->desc;
    return x->storage == y->storage && p.target == q.target && p.format == q.format && p.flags == q.flags &&
           p.width == q.width && p.height == q.height && p.depth == q.depth && p.levels == q.levels &&
           p.samples == q.samples && p.stride == q.stride && p.depth_stencil_mode == q.depth_stencil_mode &&
           memcmp(p.swizzle, q.swizzle, sizeof p.swizzle) == 0;
}
// Dónde empieza en la memoria de la textura el primer nivel y la primera capa de la vista: capa · tamaño de capa +
// offset del nivel, con la misma disposición que GetStorageSize y la subida del ejecutor. MH Rise recorre cada
// (capa, nivel) con vistas de uno y copia en su staging, para cada uno, lo que va de su offset al del siguiente (el
// último, hasta GetStorageSize): con 0 copiaba la textura entera en cada nivel y se salía del staging
// (main+0x771da34 -> main+0x7723388).
int64_t nvnTextureGetViewOffset(const void* t, const TextureViewDesc* view) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    if (!tex || tex->layout.levels.empty()) return 0;
    const nvn::TextureLayout& L = tex->layout;
    const size_t level = std::min<size_t>(std::max(view_level(view), 0), L.levels.size() - 1);
    // Las 3D no tienen capas (su profundidad va dentro de cada nivel).
    const size_t layer = tex->desc.target == nvn::kTarget3D ? 0 : std::min<size_t>(std::max(view_layer(view), 0), L.layers - 1);
    return static_cast<int64_t>(layer * L.layer_size + L.levels[level].offset);
}

// --- Texels desde la CPU: en la Switch van directos a la memoria de la textura (swizzleados si es block-linear) ---

void nvnTextureWriteTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, const void* data) {
    if (nvn::Texture* tex = nvn::texture_lookup(t)) write_texels(*tex, view, region, data, 0, 0);
}
void nvnTextureWriteTexelsStrided(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region,
                                  const void* data, int64_t row_stride, int64_t image_stride) {
    if (nvn::Texture* tex = nvn::texture_lookup(t)) write_texels(*tex, view, region, data, row_stride, image_stride);
}
void nvnTextureReadTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data) {
    if (nvn::Texture* tex = nvn::texture_lookup(t)) read_texels(*tex, view, region, data, 0, 0);
}
void nvnTextureReadTexelsStrided(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data,
                                 int64_t row_stride, int64_t image_stride) {
    if (nvn::Texture* tex = nvn::texture_lookup(t)) read_texels(*tex, view, region, data, row_stride, image_stride);
}
// La CPU escribió directamente esos texels en la memoria de la textura (en su disposición): como tras WriteTexels, la
// caja se sube a la imagen al usarla (también si la GPU ya la había escrito: solo esa caja).
void nvnTextureFlushTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    TexelRegion r;
    if (!tex || !texel_region(*tex, view, region, 0, 0, r)) return;
    std::lock_guard lock(nvn::texel_writes_mutex());
    tex->texel_writes.push_back({r.level_index, r.layer, r.layers, r.box});
    nvn::g_texel_writes++;
}
// La CPU va a leer la memoria de la textura: lo que pinta la GPU se queda en la imagen de Vulkan y no vuelve a esa memoria
// (como en ReadTexels), así que no hay nada que traer.
void nvnTextureInvalidateTexels(const void*, const TextureViewDesc*, const nvn::CopyRegion*) {}
// El juego escribe por CPU en texturas lineales (los frames de vídeo) con este stride.
int64_t nvnTextureGetStride(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex && tex->layout.linear ? tex->layout.stride : 0;
}
size_t nvnTextureGetStorageSize(const void* t) {
    nvn::Texture* tex = nvn::texture_lookup(t);
    return tex ? tex->layout.total_size() : 0;
}

// --- Vistas ---

void nvnTextureViewSetDefaults(TextureViewDesc* v) { memset(v, 0, sizeof *v); }
void nvnTextureViewSetLevels(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLevels;
    v->base_level = base, v->levels = count;
}
void nvnTextureViewSetLayers(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLayers;
    v->base_layer = base, v->layers = count;
}
void nvnTextureViewSetFormat(TextureViewDesc* v, int format) {
    v->flags |= nvn::kViewFormat;
    v->format = format;
}
void nvnTextureViewSetTarget(TextureViewDesc* v, int target) {
    v->flags |= nvn::kViewTarget;
    v->target = target;
}
void nvnTextureViewSetDepthStencilMode(TextureViewDesc* v, int mode) {
    v->flags |= nvn::kViewDepthStencil;
    v->depth_stencil_mode = mode;
}
void nvnTextureViewSetSwizzle(TextureViewDesc* v, int r, int g, int b, int a) {
    static int logged = 0;  // DEPURACIÓN: qué combinaciones usa el juego
    static int seen[16][4];
    bool fresh = true;
    for (int k = 0; k < std::min(logged, 16); k++)
        if (seen[k][0] == r && seen[k][1] == g && seen[k][2] == b && seen[k][3] == a) fresh = false;
    if (fresh && logged < 16) {
        seen[logged][0] = r, seen[logged][1] = g, seen[logged][2] = b, seen[logged][3] = a;
        logged++;
        NVN_LOG("swizzle de vista: %d %d %d %d\n", r, g, b, a);
    }
    v->flags |= nvn::kViewSwizzle;
    v->swizzle[0] = r, v->swizzle[1] = g, v->swizzle[2] = b, v->swizzle[3] = a;
}

// Dos vistas iguales: fijan los mismos campos con los mismos valores (lo que no fijan no cuenta).
nvn::Boolean nvnTextureViewCompare(const TextureViewDesc* a, const TextureViewDesc* b) {
    if (a == b) return 1;
    if (!a || !b || a->flags != b->flags) return 0;
    const uint8_t f = a->flags;
    if ((f & nvn::kViewLevels) && (a->base_level != b->base_level || a->levels != b->levels)) return 0;
    if ((f & nvn::kViewLayers) && (a->base_layer != b->base_layer || a->layers != b->layers)) return 0;
    if ((f & nvn::kViewFormat) && a->format != b->format) return 0;
    if ((f & nvn::kViewTarget) && a->target != b->target) return 0;
    if ((f & nvn::kViewDepthStencil) && a->depth_stencil_mode != b->depth_stencil_mode) return 0;
    if ((f & nvn::kViewSwizzle) && memcmp(a->swizzle, b->swizzle, sizeof a->swizzle) != 0) return 0;
    return 1;
}

// Getters de la vista: NVN_TRUE si la vista fija ese campo (si no, no tocan la salida).
nvn::Boolean nvnTextureViewGetLevels(const TextureViewDesc* v, int* base, int* count) {
    if (!(v->flags & nvn::kViewLevels)) return 0;
    if (base) *base = v->base_level;
    if (count) *count = v->levels;
    return 1;
}
nvn::Boolean nvnTextureViewGetLayers(const TextureViewDesc* v, int* base, int* count) {
    if (!(v->flags & nvn::kViewLayers)) return 0;
    if (base) *base = v->base_layer;
    if (count) *count = v->layers;
    return 1;
}
nvn::Boolean nvnTextureViewGetFormat(const TextureViewDesc* v, int* format) {
    if (!(v->flags & nvn::kViewFormat)) return 0;
    if (format) *format = v->format;
    return 1;
}
nvn::Boolean nvnTextureViewGetTarget(const TextureViewDesc* v, int* target) {
    if (!(v->flags & nvn::kViewTarget)) return 0;
    if (target) *target = v->target;
    return 1;
}
nvn::Boolean nvnTextureViewGetDepthStencilMode(const TextureViewDesc* v, int* mode) {
    if (!(v->flags & nvn::kViewDepthStencil)) return 0;
    if (mode) *mode = v->depth_stencil_mode;
    return 1;
}
nvn::Boolean nvnTextureViewGetSwizzle(const TextureViewDesc* v, int* r, int* g, int* b, int* a) {
    if (!(v->flags & nvn::kViewSwizzle)) return 0;
    if (r) *r = v->swizzle[0];
    if (g) *g = v->swizzle[1];
    if (b) *b = v->swizzle[2];
    if (a) *a = v->swizzle[3];
    return 1;
}

// --- Pools de descriptores ---

nvn::Boolean nvnTexturePoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    nvn::DescriptorPool& p = nvn::g_texture_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.textures.resize(count);
    p.buffer_objects.resize(count);
    return 1;
}
// Los pools de descriptores no se borran: la cola guarda su puntero (cmd::SetPools en las listas grabadas y, entre un envío
// y el siguiente, st_.texture_pool y st_.sampler_pool en el ejecutor) y lo sigue usando al reproducir lo que quede. Unity
// crea uno de cada al arrancar (main+0x3e4fb8) y solo los finaliza al cerrar (main+0x3e510c y main+0x3e5124), así que
// conservarlos no cuesta nada; borrados, un draw pendiente leía memoria liberada.
static void keep_descriptor_pool(std::unique_ptr<nvn::DescriptorPool> p) {
    if (!p) return;
    static std::mutex m;
    static std::vector<std::unique_ptr<nvn::DescriptorPool>> finalized;
    std::lock_guard lock(m);
    finalized.push_back(std::move(p));
}
void nvnTexturePoolFinalize(void* pool) { keep_descriptor_pool(nvn::g_texture_pools.take(pool)); }
void nvnTexturePoolRegisterTexture(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    if (!p || id < 0 || id >= p->count) return;
    nvn::Texture* registered = nvn::texture_lookup(texture);
    if (registered && registered->desc.target == nvn::kTargetBuffer && !NVN_FLAG("buffers_por_objeto")) {
        // Un buffer visto como textura: la entrada se queda con su propia copia (ver DescriptorPool::buffer_objects).
        // Como una textura ya finalizada: sin imagen ni memoria que subir; el ejecutor lee desc y cpu_storage (texel_range).
        // <datos>/buffers_por_objeto (en vivo): como antes, la entrada apunta al objeto del juego.
        std::unique_ptr<nvn::Texture>& own = p->buffer_objects[id];
        if (!own) own = std::make_unique<nvn::Texture>();
        own->desc = registered->desc;
        own->serial = registered->serial;
        own->init_frame = registered->init_frame;
        own->storage = nullptr;
        own->cpu_storage = registered->storage ? registered->storage : registered->cpu_storage;
        own->vk_failed = true;
        registered = own.get();
    }
    p->textures[id] = {registered, view ? *view : TextureViewDesc{}};
    {  // DEPURACIÓN: ¿limita el juego los niveles al registrar (streaming)?
        static int logged = 0, with_levels = 0, total = 0;
        total++;
        const nvn::Texture* t = nvn::texture_lookup(texture);
        if (view && (view->flags & nvn::kViewLevels) && t && t->desc.levels > 1) {
            with_levels++;
            if (nvn::first_times(logged, 30))
                NVN_LOG("registro con niveles: textura %llu %dx%d formato 0x%x niveles %d -> vista base %d cuenta %d (id %d)\n",
                        static_cast<unsigned long long>(t->serial), t->desc.width, t->desc.height, t->desc.format, t->desc.levels,
                        view->base_level, view->levels, id);
        }
        if (total % 5000 == 0) NVN_LOG("registros de texturas: %d, con niveles %d\n", total, with_levels);
    }
}
void nvnTexturePoolRegisterImage(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    nvnTexturePoolRegisterTexture(pool, id, texture, view);
}

nvn::Boolean nvnSamplerPoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    nvn::DescriptorPool& p = nvn::g_sampler_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.samplers.resize(count);
    return 1;
}
void nvnSamplerPoolFinalize(void* pool) { keep_descriptor_pool(nvn::g_sampler_pools.take(pool)); }
void nvnSamplerPoolRegisterSampler(void* pool, int id, const void* sampler) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    if (!p || id < 0 || id >= p->count) return;
    p->samplers[id] = g_samplers.get(sampler);
}
nvn::Boolean nvnSamplerInitialize(void* sampler, const SamplerDesc* b);
// Un sampler sin objeto del juego (solo el builder): uno nuestro por ranura, con la ranura del pool como clave (memoria
// nuestra, no choca con objetos del juego). Reinicializarlo retira su VkSampler como nvnSamplerInitialize.
void nvnSamplerPoolRegisterSamplerBuilder(void* pool, int id, const SamplerDesc* b) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    if (!p || !b || id < 0 || id >= p->count) return;
    void* key = &p->samplers[id];
    nvnSamplerInitialize(key, b);
    p->samplers[id] = g_samplers.get(key);
}
void* nvnTexturePoolGetMemoryPool(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->memory_pool : nullptr;
}
int64_t nvnTexturePoolGetMemoryOffset(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->offset : 0;
}
int nvnTexturePoolGetSize(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->count : 0;
}
void* nvnSamplerPoolGetMemoryPool(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->memory_pool : nullptr;
}
int64_t nvnSamplerPoolGetMemoryOffset(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->offset : 0;
}
int nvnSamplerPoolGetSize(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->count : 0;
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnTexturePoolSetDebugLabel(void*, const char*) {}
void nvnSamplerPoolSetDebugLabel(void*, const char*) {}

// --- Handles: separados = id; combinado = sampler << 32 | textura (los resuelve el ejecutor) ---

uint64_t nvnDeviceGetSeparateTextureHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }
uint64_t nvnDeviceGetSeparateSamplerHandle(const void*, int sampler_id) { return static_cast<uint32_t>(sampler_id); }
uint64_t nvnDeviceGetTextureHandle(const void*, int texture_id, int sampler_id) {
    return uint64_t{static_cast<uint32_t>(sampler_id)} << 32 | static_cast<uint32_t>(texture_id);
}
uint64_t nvnDeviceGetImageHandle(const void*, int image_id) { return static_cast<uint32_t>(image_id); }
uint64_t nvnDeviceGetTexelFetchHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }

// --- Samplers ---

void nvnSamplerBuilderSetDefaults(SamplerDesc* b) {
    memset(b, 0, sizeof *b);
    b->min_filter = 4;  // NEAREST_MIPMAP_LINEAR (el de OpenGL)
    b->mag_filter = 1;  // LINEAR
    b->wrap[0] = b->wrap[1] = b->wrap[2] = 1;  // REPEAT
    b->max_anisotropy = 1.0f;
    b->lod_min = -1000.0f;
    b->lod_max = 1000.0f;
    b->compare_func = 8;  // ALWAYS
}
void nvnSamplerBuilderSetDevice(SamplerDesc* b, void* device) { b->device = device; }
void nvnSamplerBuilderSetMinMagFilter(SamplerDesc* b, int min, int mag) { b->min_filter = min, b->mag_filter = mag; }
void nvnSamplerBuilderSetWrapMode(SamplerDesc* b, int s, int t, int r) { b->wrap[0] = s, b->wrap[1] = t, b->wrap[2] = r; }
void nvnSamplerBuilderSetLodClamp(SamplerDesc* b, float min, float max) {
    b->lod_min = min, b->lod_max = max;
    static int logged = 0;  // DEPURACIÓN
    if (min > 0 && nvn::first_times(logged, 20)) NVN_LOG("sampler con LOD mínimo %.2f (máximo %.2f)\n", min, max);
}
void nvnSamplerBuilderSetLodBias(SamplerDesc* b, float bias) { b->lod_bias = bias; }
void nvnSamplerBuilderSetCompare(SamplerDesc* b, int mode, int func) { b->compare_mode = mode, b->compare_func = func; }
void nvnSamplerBuilderSetBorderColor(SamplerDesc* b, const float* color) { memcpy(b->border, color, sizeof b->border); }
void nvnSamplerBuilderSetMaxAnisotropy(SamplerDesc* b, float a) { b->max_anisotropy = a; }
void nvnSamplerBuilderSetReductionFilter(SamplerDesc* b, int filter) { b->reduction = filter; }
// Bordes enteros, guardados como float: el ejecutor solo mira si son negro, blanco o transparente (0 y 1).
void nvnSamplerBuilderSetBorderColori(SamplerDesc* b, const int* color) {
    for (int k = 0; k < 4; k++) b->border[k] = static_cast<float>(color[k]);
}
void nvnSamplerBuilderSetBorderColorui(SamplerDesc* b, const uint32_t* color) {
    for (int k = 0; k < 4; k++) b->border[k] = static_cast<float>(color[k]);
}

void nvnSamplerBuilderGetMinMagFilter(const SamplerDesc* b, int* min, int* mag) { sampler_min_mag(*b, min, mag); }
void nvnSamplerBuilderGetWrapMode(const SamplerDesc* b, int* s, int* t, int* r) { sampler_wrap(*b, s, t, r); }
void nvnSamplerBuilderGetLodClamp(const SamplerDesc* b, float* min, float* max) { sampler_lod_clamp(*b, min, max); }
float nvnSamplerBuilderGetLodBias(const SamplerDesc* b) { return b->lod_bias; }
void nvnSamplerBuilderGetCompare(const SamplerDesc* b, int* mode, int* func) { sampler_compare(*b, mode, func); }
void nvnSamplerBuilderGetBorderColor(const SamplerDesc* b, float* color) { memcpy(color, b->border, sizeof b->border); }
void nvnSamplerBuilderGetBorderColori(const SamplerDesc* b, int* color) { border_int(*b, color); }
void nvnSamplerBuilderGetBorderColorui(const SamplerDesc* b, uint32_t* color) { border_int(*b, color); }
float nvnSamplerBuilderGetMaxAnisotropy(const SamplerDesc* b) { return b->max_anisotropy; }
int nvnSamplerBuilderGetReductionFilter(const SamplerDesc* b) { return b->reduction; }
void* nvnSamplerBuilderGetDevice(const SamplerDesc* b) { return b->device; }
// LOD snap de Maxwell (redondear el LOD al nivel entero si está a menos de esto): Vulkan no lo tiene; se guarda.
void nvnSamplerBuilderSetLodSnap(SamplerDesc* b, float snap) { b->lod_snap = snap; }
float nvnSamplerBuilderGetLodSnap(const SamplerDesc* b) { return b->lod_snap; }

void nvnSamplerGetMinMagFilter(const void* s, int* min, int* mag) { sampler_min_mag(sampler_desc(s), min, mag); }
void nvnSamplerGetWrapMode(const void* s, int* ws, int* wt, int* wr) { sampler_wrap(sampler_desc(s), ws, wt, wr); }
void nvnSamplerGetLodClamp(const void* s, float* min, float* max) { sampler_lod_clamp(sampler_desc(s), min, max); }
float nvnSamplerGetLodBias(const void* s) { return sampler_desc(s).lod_bias; }
void nvnSamplerGetCompare(const void* s, int* mode, int* func) { sampler_compare(sampler_desc(s), mode, func); }
void nvnSamplerGetBorderColor(const void* s, float* color) { memcpy(color, sampler_desc(s).border, 16); }
void nvnSamplerGetBorderColori(const void* s, int* color) { border_int(sampler_desc(s), color); }
void nvnSamplerGetBorderColorui(const void* s, uint32_t* color) { border_int(sampler_desc(s), color); }
float nvnSamplerGetMaxAnisotropy(const void* s) { return sampler_desc(s).max_anisotropy; }
int nvnSamplerGetReductionFilter(const void* s) { return sampler_desc(s).reduction; }
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnSamplerSetDebugLabel(void*, const char*) {}
uint64_t nvnSamplerGetDebugID(const void* sampler) {
    nvn::Sampler* s = g_samplers.get(sampler);
    return s ? s->debug_id : 0;
}
// Dos samplers iguales: la misma descripción (sin contar el dispositivo).
nvn::Boolean nvnSamplerCompare(const void* a, const void* b) {
    if (a == b) return 1;
    const SamplerDesc& x = sampler_desc(a);
    const SamplerDesc& y = sampler_desc(b);
    if (!g_samplers.get(a) || !g_samplers.get(b)) return 0;
    return x.min_filter == y.min_filter && x.mag_filter == y.mag_filter && memcmp(x.wrap, y.wrap, sizeof x.wrap) == 0 &&
           x.compare_mode == y.compare_mode && x.compare_func == y.compare_func && x.reduction == y.reduction &&
           x.max_anisotropy == y.max_anisotropy && x.lod_min == y.lod_min && x.lod_max == y.lod_max &&
           x.lod_bias == y.lod_bias && memcmp(x.border, y.border, sizeof x.border) == 0 && x.lod_snap == y.lod_snap;
}

// Como las texturas, el objeto no se borra nunca: los pools de samplers guardan su dirección (borrarlo
// dejaba al ejecutor escribiendo en memoria liberada). Su VkSampler se retira hasta que la GPU acaba.
nvn::Boolean nvnSamplerInitialize(void* sampler, const SamplerDesc* b) {
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(lock);
    nvn::Sampler* s = g_samplers.get(sampler);
    if (!s) s = &g_samplers.create(sampler);
    if (s->vk) nvn::retire_sampler(s->vk);
    if (s->vk_af) nvn::retire_sampler(s->vk_af);
    *s = nvn::Sampler{};
    s->desc = *b;
    static std::atomic<uint64_t> next_id{1};
    s->debug_id = next_id++;
    return 1;
}
void nvnSamplerFinalize(void* sampler) {
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(lock);
    if (nvn::Sampler* s = g_samplers.get(sampler)) {
        if (s->vk) nvn::retire_sampler(s->vk);
        if (s->vk_af) nvn::retire_sampler(s->vk_af);
        s->vk = s->vk_af = VK_NULL_HANDLE;
    }
}

}  // extern "C"

NVN_PROC(nvnTextureBuilderSetDefaults);
NVN_PROC(nvnTextureBuilderSetDevice);
NVN_PROC(nvnTextureBuilderSetFlags);
NVN_PROC(nvnTextureBuilderSetTarget);
NVN_PROC(nvnTextureBuilderSetFormat);
NVN_PROC(nvnTextureBuilderSetLevels);
NVN_PROC(nvnTextureBuilderSetSamples);
NVN_PROC(nvnTextureBuilderSetStride);
NVN_PROC(nvnTextureBuilderSetWidth);
NVN_PROC(nvnTextureBuilderSetHeight);
NVN_PROC(nvnTextureBuilderSetDepth);
NVN_PROC(nvnTextureBuilderSetSize1D);
NVN_PROC(nvnTextureBuilderSetSize2D);
NVN_PROC(nvnTextureBuilderSetSize3D);
NVN_PROC(nvnTextureBuilderSetStorage);
NVN_PROC(nvnTextureBuilderGetStorageSize);
NVN_PROC(nvnTextureBuilderGetStorageAlignment);
NVN_PROC(nvnTextureInitialize);
NVN_PROC(nvnTextureFinalize);
NVN_PROC(nvnTextureGetWidth);
NVN_PROC(nvnTextureGetHeight);
NVN_PROC(nvnTextureGetDepth);
NVN_PROC(nvnTextureGetFormat);
NVN_PROC(nvnTextureGetTarget);
NVN_PROC(nvnTextureGetLevels);
NVN_PROC(nvnTextureGetFlags);
NVN_PROC(nvnTextureGetStorageSize);
NVN_PROC(nvnTextureGetStride);
NVN_PROC(nvnTextureGetSamples);
NVN_PROC(nvnTextureGetDepthStencilMode);
NVN_PROC(nvnTextureGetStorageClass);
NVN_PROC(nvnTextureGetRawStorageClass);
NVN_PROC(nvnTextureGetZCullStorageSize);
NVN_PROC(nvnTextureGetMemoryPool);
NVN_PROC(nvnTextureGetMemoryOffset);
NVN_PROC(nvnTextureGetTextureAddress);
NVN_PROC(nvnTextureGetSwizzle);
NVN_PROC(nvnTextureGetDebugID);
NVN_PROC(nvnTextureSetDebugLabel);
NVN_PROC(nvnTextureGetViewOffset);
NVN_PROC(nvnTextureWriteTexels);
NVN_PROC(nvnTextureWriteTexelsStrided);
NVN_PROC(nvnTextureReadTexels);
NVN_PROC(nvnTextureReadTexelsStrided);
NVN_PROC(nvnTextureFlushTexels);
NVN_PROC(nvnTextureInvalidateTexels);
NVN_PROC(nvnTextureBuilderSetDepthStencilMode);
NVN_PROC(nvnTextureBuilderSetSwizzle);
NVN_PROC(nvnTextureBuilderSetPackagedTextureLayout);
NVN_PROC(nvnTextureBuilderGetPackagedTextureLayout);
NVN_PROC(nvnTextureBuilderSetPackagedTextureData);
NVN_PROC(nvnTextureBuilderGetPackagedTextureData);
NVN_PROC(nvnTextureBuilderGetStorageClass);
NVN_PROC(nvnTextureBuilderGetRawStorageClass);
NVN_PROC(nvnTextureBuilderGetZCullStorageSize);
NVN_PROC(nvnTextureBuilderGetDevice);
NVN_PROC(nvnTextureBuilderGetFlags);
NVN_PROC(nvnTextureBuilderGetTarget);
NVN_PROC(nvnTextureBuilderGetWidth);
NVN_PROC(nvnTextureBuilderGetHeight);
NVN_PROC(nvnTextureBuilderGetDepth);
NVN_PROC(nvnTextureBuilderGetLevels);
NVN_PROC(nvnTextureBuilderGetFormat);
NVN_PROC(nvnTextureBuilderGetSamples);
NVN_PROC(nvnTextureBuilderGetStride);
NVN_PROC(nvnTextureBuilderGetMemoryPool);
NVN_PROC(nvnTextureBuilderGetMemoryOffset);
NVN_PROC(nvnTextureBuilderGetDepthStencilMode);
NVN_PROC(nvnTextureBuilderGetSwizzle);
NVN_PROC(nvnTextureViewSetDefaults);
NVN_PROC(nvnTextureViewSetLevels);
NVN_PROC(nvnTextureViewSetLayers);
NVN_PROC(nvnTextureViewSetFormat);
NVN_PROC(nvnTextureViewSetTarget);
NVN_PROC(nvnTextureViewSetDepthStencilMode);
NVN_PROC(nvnTextureViewSetSwizzle);
NVN_PROC(nvnTextureViewGetLevels);
NVN_PROC(nvnTextureViewGetLayers);
NVN_PROC(nvnTextureViewGetFormat);
NVN_PROC(nvnTextureViewGetTarget);
NVN_PROC(nvnTextureViewGetDepthStencilMode);
NVN_PROC(nvnTextureViewGetSwizzle);
NVN_PROC(nvnTexturePoolInitialize);
NVN_PROC(nvnTexturePoolFinalize);
NVN_PROC(nvnTexturePoolRegisterTexture);
NVN_PROC(nvnTexturePoolRegisterImage);
NVN_PROC(nvnSamplerPoolInitialize);
NVN_PROC(nvnSamplerPoolFinalize);
NVN_PROC(nvnSamplerPoolRegisterSampler);
NVN_PROC(nvnSamplerPoolRegisterSamplerBuilder);
NVN_PROC(nvnTexturePoolGetMemoryPool);
NVN_PROC(nvnTexturePoolGetMemoryOffset);
NVN_PROC(nvnTexturePoolGetSize);
NVN_PROC(nvnSamplerPoolGetMemoryPool);
NVN_PROC(nvnSamplerPoolGetMemoryOffset);
NVN_PROC(nvnSamplerPoolGetSize);
NVN_PROC(nvnTexturePoolSetDebugLabel);
NVN_PROC(nvnSamplerPoolSetDebugLabel);
NVN_PROC(nvnDeviceGetSeparateTextureHandle);
NVN_PROC(nvnDeviceGetSeparateSamplerHandle);
NVN_PROC(nvnDeviceGetTextureHandle);
NVN_PROC(nvnDeviceGetImageHandle);
NVN_PROC(nvnDeviceGetTexelFetchHandle);
NVN_PROC(nvnSamplerBuilderSetDefaults);
NVN_PROC(nvnSamplerBuilderSetDevice);
NVN_PROC(nvnSamplerBuilderSetMinMagFilter);
NVN_PROC(nvnSamplerBuilderSetWrapMode);
NVN_PROC(nvnSamplerBuilderSetLodClamp);
NVN_PROC(nvnSamplerBuilderSetLodBias);
NVN_PROC(nvnSamplerBuilderSetCompare);
NVN_PROC(nvnSamplerBuilderSetBorderColor);
NVN_PROC(nvnSamplerBuilderSetMaxAnisotropy);
NVN_PROC(nvnSamplerBuilderSetReductionFilter);
NVN_PROC(nvnSamplerBuilderSetBorderColori);
NVN_PROC(nvnSamplerBuilderSetBorderColorui);
NVN_PROC(nvnSamplerBuilderGetMinMagFilter);
NVN_PROC(nvnSamplerBuilderGetWrapMode);
NVN_PROC(nvnSamplerBuilderGetLodClamp);
NVN_PROC(nvnSamplerBuilderGetLodBias);
NVN_PROC(nvnSamplerBuilderGetCompare);
NVN_PROC(nvnSamplerBuilderGetBorderColor);
NVN_PROC(nvnSamplerBuilderGetBorderColori);
NVN_PROC(nvnSamplerBuilderGetBorderColorui);
NVN_PROC(nvnSamplerBuilderGetMaxAnisotropy);
NVN_PROC(nvnSamplerBuilderGetReductionFilter);
NVN_PROC(nvnSamplerBuilderGetDevice);
NVN_PROC(nvnSamplerGetMinMagFilter);
NVN_PROC(nvnSamplerGetWrapMode);
NVN_PROC(nvnSamplerGetLodClamp);
NVN_PROC(nvnSamplerGetLodBias);
NVN_PROC(nvnSamplerGetCompare);
NVN_PROC(nvnSamplerGetBorderColor);
NVN_PROC(nvnSamplerGetBorderColori);
NVN_PROC(nvnSamplerGetBorderColorui);
NVN_PROC(nvnSamplerGetMaxAnisotropy);
NVN_PROC(nvnSamplerGetReductionFilter);
NVN_PROC(nvnSamplerSetDebugLabel);
NVN_PROC(nvnSamplerInitialize);
NVN_PROC(nvnSamplerFinalize);
NVN_PROC(nvnTextureBuilderSetGLTextureName);
NVN_PROC(nvnTextureBuilderGetGLTextureName);
NVN_PROC(nvnTextureBuilderGetSparseTileLayout);
NVN_PROC(nvnTextureGetSparseTileLayout);
NVN_PROC(nvnTextureGetNvRmSurfaceNVX);
NVN_PROC(nvnTextureCompare);
NVN_PROC(nvnTextureViewCompare);
NVN_PROC(nvnSamplerBuilderSetLodSnap);
NVN_PROC(nvnSamplerBuilderGetLodSnap);
NVN_PROC(nvnSamplerGetDebugID);
NVN_PROC(nvnSamplerCompare);
