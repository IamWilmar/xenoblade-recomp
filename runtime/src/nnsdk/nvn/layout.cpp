#include "nnsdk/nvn/layout.h"

#include <algorithm>
#include <cstring>

#include "nnsdk/nvn/flags.h"

namespace nvn {

namespace {

constexpr uint32_t kGobWidth = 64, kGobHeight = 8, kGobSize = kGobWidth * kGobHeight;

uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
size_t align(size_t v, size_t a) { return (v + a - 1) / a * a; }

uint32_t pow2_ceil(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Alto del bloque del nivel 0 cuando la textura no trae NVNpackagedTextureLayout: la regla del NVN real (la misma que
// usa el empaquetador de Monolith para las texturas pre-swizzleadas de los .wilay, y la de block_height_mip0 de
// tegra_swizzle): con h = filas de bloques, h + h/2 >= 128 -> 16 GOBs, >= 64 -> 8, >= 32 -> 4, >= 16 -> 2, si no 1.
// Antes: la potencia de 2 que cubría el alto (pow2_ceil(ceil(h / 8))), que da el doble en h = 9-10, 17-21, 33-42 y
// 65-85 (en BC7: 33-40, 65-84, 129-168 y 257-340 px de alto). Con el doble, los GOBs de la fila se leen en otro
// orden y la imagen sale con las columnas de 16 px pares comprimidas a la izquierda y las impares perdidas: el
// glifo y el anillo fino del emblema de carga (mnu002_cont01: 80x80 y 144x144 BC7) salían aplastados y
// desplazados arriba-izquierda, y los rótulos de zona (mnu050_cont01, 760x40 y 1024x296) ilegibles. Verificado
// descodificando las texturas del romfs con las dos reglas: solo con esta salen limpias. Los tamaños y offsets
// de los niveles no cambian con la regla (la alineación de filas absorbe la diferencia), así que la memoria que
// pide el juego es la misma. <datos>/bloque_alto_pot2 vuelve a la regla antigua (para las texturas creadas
// después de tocarlo). Las 3D siguen con la antigua: no hay medida de qué hace NVN con ellas.
uint32_t base_block_height(uint32_t height_blocks, bool is_3d) {
    if (is_3d || NVN_FLAG("bloque_alto_pot2")) return std::min<uint32_t>(16, pow2_ceil(div_ceil(height_blocks, kGobHeight)));
    const uint32_t h = height_blocks + height_blocks / 2;
    return h >= 128 ? 16 : h >= 64 ? 8 : h >= 32 ? 4 : h >= 16 ? 2 : 1;
}

uint32_t level_block_height(uint32_t base, uint32_t height_blocks) {
    uint32_t bh = base;
    while (bh > 1 && height_blocks <= (bh * kGobHeight) / 2) bh /= 2;
    return bh;
}

uint32_t level_block_depth(uint32_t base, uint32_t depth) {
    uint32_t bd = base;
    while (bd > 1 && depth <= bd / 2) bd /= 2;
    return bd;
}

}  // namespace

TextureLayout compute_layout(uint32_t width, uint32_t height, uint32_t depth, uint32_t layers, uint32_t levels,
                             uint32_t bytes_per_block, uint32_t block_w, uint32_t block_h, bool linear, uint32_t stride,
                             bool is_3d, int block_height_log2, uint32_t tile_width_log2) {
    TextureLayout t;
    t.linear = linear;
    t.bytes_per_block = bytes_per_block;
    t.block_w = block_w;
    t.block_h = block_h;
    t.layers = std::max(layers, 1u);
    levels = std::max(levels, 1u);
    width = std::max(width, 1u), height = std::max(height, 1u), depth = std::max(depth, 1u);

    if (linear) {
        uint32_t w_blocks = div_ceil(width, block_w), h_blocks = div_ceil(height, block_h);
        t.stride = stride ? stride : static_cast<uint32_t>(align(w_blocks * bytes_per_block, 128));
        t.levels.push_back({0, size_t{t.stride} * h_blocks, width, height, 1, w_blocks, h_blocks, 1, 1});
        t.layer_size = t.levels[0].size;
        return t;
    }

    // 3D: como mucho 16 GOBs por bloque, primero en profundidad. Medido en un LUT de color de MH Rise de 32x32x32
    // RGBA8: 1 GOB de alto y 16 de fondo (con 4x32, los cortes se mezclaban y la imagen salía psicodélica).
    uint32_t base_bd = is_3d ? std::min<uint32_t>(16, pow2_ceil(depth)) : 1;
    uint32_t base_bh = std::min<uint32_t>(base_block_height(div_ceil(height, block_h), is_3d), std::max<uint32_t>(16 / base_bd, 1));
    // Texturas empaquetadas (las sparse del streaming de MH Rise traen "03 04 00 00 07 00 01 00"): bloque de 8 GOBs de
    // alto y pitch alineado a 16 GOBs, o sea tiles de 16x8 GOBs = 64 KB, la página sparse. Con el alto calculado (16 en
    // 1024x1024) los niveles 0 y 1 salían revueltos (bandas repetidas y columnas de GOBs cambiadas): las manchas negras
    // y los parches de las caras al acercarse. Medido sobre la memoria del juego: con 8 GOBs la cara de Hinoa (BC1), su
    // mapa de normales (BC7) y las pestañas (BC4) salen limpias; el kimono en ASTC 6x6 necesita además el pitch de 48
    // GOBs en el nivel 0 (43 usados + 5 de relleno, se ve en las páginas a cero) y de 32 en el 1.
    if (block_height_log2 >= 0) base_bh = 1u << std::min(block_height_log2, 5);
    const uint32_t tile_w = 1u << std::min<uint32_t>(tile_width_log2, 5);  // en GOBs
    size_t offset = 0;
    for (uint32_t l = 0; l < levels; l++) {
        uint32_t w = std::max(width >> l, 1u), h = std::max(height >> l, 1u), d = is_3d ? std::max(depth >> l, 1u) : 1;
        uint32_t w_blocks = div_ceil(w, block_w), h_blocks = div_ceil(h, block_h);
        uint32_t bh = level_block_height(base_bh, h_blocks);
        uint32_t bd = is_3d ? level_block_depth(base_bd, d) : 1;
        uint32_t row_gobs = div_ceil(w_blocks * bytes_per_block, kGobWidth);
        // El pitch va alineado al tile salvo en los niveles más pequeños que un tile en ancho o alto (la cola de mips),
        // como en el SizeCalculator de Ryujinx / NumGobs de Eden (con el alto de bloque del nivel 0).
        const bool small = w_blocks <= (kGobWidth / std::max(bytes_per_block, 1u)) * tile_w || h_blocks <= base_bh * kGobHeight ||
                           d < base_bd;
        if (tile_w > 1 && !small) row_gobs = static_cast<uint32_t>(align(row_gobs, tile_w));
        size_t row_bytes = size_t{row_gobs} * kGobWidth;
        size_t rows = align(h_blocks, kGobHeight * bh);
        size_t size = row_bytes * rows * align(d, bd);
        t.levels.push_back({offset, size, w, h, d, w_blocks, h_blocks, bh, bd, row_gobs});
        offset += size;
    }
    // Cada capa empieza alineada al bloque del nivel 0 (con tiles, al tile: también con una sola capa, como el
    // AlignLayerSize de Eden; es lo que ocupa la textura sparse).
    if (tile_w > 1)
        t.layer_size = align(offset, size_t{kGobSize} * tile_w * base_bh * base_bd);
    else
        t.layer_size = t.layers > 1 ? align(offset, size_t{kGobSize} * base_bh * base_bd) : offset;
    return t;
}

// Dirección block-linear (Tegra): dentro de un GOB de 64x8 bytes los 512 bytes se ordenan en
// sectores de 16x2; los GOBs se apilan en vertical (y luego en profundidad, en las 3D) dentro del bloque
// y los bloques van por filas y luego por cortes.
namespace {

struct Swizzle {
    uint32_t row_bytes, gobs_per_row, bh, bd;
    size_t block_size, block_row_size, slice_size;
    explicit Swizzle(const LevelLayout& level, uint32_t bpb)
        : row_bytes(level.width_blocks * bpb),
          gobs_per_row(level.row_gobs ? level.row_gobs : div_ceil(row_bytes, kGobWidth)),
          bh(level.block_height_gobs),
          bd(std::max(level.block_depth, 1u)),
          block_size(size_t{kGobSize} * bh * bd),
          block_row_size(block_size * gobs_per_row),
          slice_size(block_row_size * div_ceil(level.height_blocks, kGobHeight * bh)) {}
    // Dónde empieza la fila (y, z): su GOB en x = 0 más la fila dentro del GOB. Para no dividir en cada sector.
    size_t row(uint32_t y, uint32_t z) const {
        const uint32_t in_y = y % kGobHeight;
        return size_t{z / bd} * slice_size + size_t{y / (kGobHeight * bh)} * block_row_size +
               (size_t{z % bd} * bh + (y / kGobHeight) % bh) * kGobSize + (in_y / 2) * 64 + (in_y % 2) * 16;
    }
    // Lo que añade el byte x (múltiplo de 16) a su fila: su GOB en x y su sector (fórmula de la Tegra para in_x).
    size_t column(uint32_t x) const {
        const uint32_t in_x = x % kGobWidth;
        return size_t{x / kGobWidth} * block_size + (in_x / 32) * 256 + ((in_x % 32) / 16) * 32;
    }
};

// La caja de bloques entre la disposición Tegra (tiled) y la lineal, por trozos de sector: el mismo recorrido en los
// dos sentidos, así WriteTexels y ReadTexels son exactamente inversas (y coinciden con deswizzle_level).
template <bool kToTiled>
void copy_box(const LevelLayout& level, uint32_t bpb, const BlockBox& box, uint8_t* tiled, uint8_t* linear,
              size_t row_pitch, size_t slice_pitch) {
    const Swizzle s(level, bpb);
    const uint32_t x0 = box.x * bpb, x1 = (box.x + box.width) * bpb;
    for (uint32_t z = 0; z < box.depth; z++)
        for (uint32_t y = 0; y < box.height; y++) {
            uint8_t* line = linear + z * slice_pitch + y * row_pitch;
            uint8_t* tiled_row = tiled + s.row(box.y + y, box.z + z);
            for (uint32_t x = x0; x < x1;) {
                const uint32_t n = std::min<uint32_t>(16 - x % 16, x1 - x);  // hasta el final del sector
                uint8_t* t = tiled_row + s.column(x & ~15u) + x % 16;
                if (kToTiled)
                    std::memcpy(t, line + (x - x0), n);
                else
                    std::memcpy(line + (x - x0), t, n);
                x += n;
            }
        }
}

}  // namespace

namespace {
// 16 bytes con dos cargas de 8 (ldp/stp, sin llamar a memcpy); true si eran todo ceros.
inline bool copy16(uint8_t* dst, const uint8_t* src) {
    uint64_t a, b;
    std::memcpy(&a, src, 8);
    std::memcpy(&b, src + 8, 8);
    std::memcpy(dst, &a, 8);
    std::memcpy(dst + 8, &b, 8);
    return (a | b) == 0;
}
}  // namespace

size_t count_zero_chunks(const uint8_t* p, size_t size) {
    size_t zeros = 0;
    for (size_t k = 0; k + 16 <= size; k += 16) {
        uint64_t a, b;
        std::memcpy(&a, p + k, 8);
        std::memcpy(&b, p + k + 8, 8);
        zeros += (a | b) == 0;
    }
    return zeros;
}

// Por GOBs enteros: cada fila de 64 bytes de un GOB son sus 4 sectores de 16 en +0, +32, +256 y +288 (column() con
// in_x = 0, 16, 32, 48), cuatro copias fijas de 16 bytes; solo la cola de la fila (menos de un GOB) va trozo a trozo.
// Antes cada trozo de 16 bytes era un memcpy de tamaño variable (una llamada) con su column(): al cargar el selector de
// personaje se subían 47 MB de texturas en un envío (3 millones de llamadas).
void deswizzle_level(const LevelLayout& level, uint32_t bpb, const uint8_t* src, uint8_t* dst, size_t* zero_chunks) {
    deswizzle_rows(level, bpb, src, dst, 0, size_t{std::max(level.depth, 1u)} * level.height_blocks, zero_chunks);
}

void deswizzle_rows(const LevelLayout& level, uint32_t bpb, const uint8_t* src, uint8_t* dst, size_t row_begin,
                    size_t row_end, size_t* zero_chunks) {
    const Swizzle s(level, bpb);
    const uint32_t full = s.row_bytes / kGobWidth;  // GOBs enteros de cada fila
    size_t zeros = 0;
    if (!level.height_blocks) return;
    row_end = std::min(row_end, size_t{std::max(level.depth, 1u)} * level.height_blocks);
    for (size_t r = row_begin; r < row_end; r++) {
        const uint32_t z = static_cast<uint32_t>(r / level.height_blocks), y = static_cast<uint32_t>(r % level.height_blocks);
        uint8_t* out = dst + r * s.row_bytes;  // (z * height_blocks + y) filas
        const uint8_t* in = src + s.row(y, z);
        const uint8_t* gob = in;
        for (uint32_t g = 0; g < full; g++, gob += s.block_size, out += kGobWidth) {
            zeros += copy16(out, gob);
            zeros += copy16(out + 16, gob + 32);
            zeros += copy16(out + 32, gob + 256);
            zeros += copy16(out + 48, gob + 288);
        }
        for (uint32_t x = full * kGobWidth; x < s.row_bytes; x += 16, out += 16) {
            const uint32_t n = std::min<uint32_t>(16, s.row_bytes - x);
            if (n == 16)
                zeros += copy16(out, in + s.column(x));
            else
                std::memcpy(out, in + s.column(x), n);
        }
    }
    if (zero_chunks) *zero_chunks += zeros;
}

void deswizzle_level_v1(const LevelLayout& level, uint32_t bpb, const uint8_t* src, uint8_t* dst) {
    const Swizzle s(level, bpb);
    // Las 3D, corte a corte (antes solo salía el primero aunque la subida copiaba la profundidad entera).
    for (uint32_t z = 0; z < std::max(level.depth, 1u); z++)
        for (uint32_t y = 0; y < level.height_blocks; y++) {
            uint8_t* out = dst + (size_t{z} * level.height_blocks + y) * s.row_bytes;
            const uint8_t* in = src + s.row(y, z);
            for (uint32_t x = 0; x < s.row_bytes; x += 16)
                std::memcpy(out + x, in + s.column(x), std::min<uint32_t>(16, s.row_bytes - x));
        }
}

void swizzle_region(const LevelLayout& level, uint32_t bpb, const BlockBox& box, const uint8_t* src, size_t row_pitch,
                    size_t slice_pitch, uint8_t* tiled) {
    copy_box<true>(level, bpb, box, tiled, const_cast<uint8_t*>(src), row_pitch, slice_pitch);
}

void deswizzle_region(const LevelLayout& level, uint32_t bpb, const BlockBox& box, const uint8_t* tiled, uint8_t* dst,
                      size_t row_pitch, size_t slice_pitch) {
    copy_box<false>(level, bpb, box, const_cast<uint8_t*>(tiled), dst, row_pitch, slice_pitch);
}

}  // namespace nvn
