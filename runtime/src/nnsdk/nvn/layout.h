#pragma once
// Disposición de texturas en memoria de la GPU Tegra X1 (la que asume el juego al escribir sus
// texturas ya "swizzleadas", tal como vienen en sus archivos): block-linear con GOBs de 64 bytes x 8
// filas y bloques de 1..32 GOBs de alto, o lineal con stride (flag LINEAR).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nvn {

struct LevelLayout {
    size_t offset;               // desde el inicio de la capa
    size_t size;
    uint32_t width, height, depth;          // en texels
    uint32_t width_blocks, height_blocks;   // en bloques del formato (4x4 en BCn, 6x6 en ASTC 6x6…)
    uint32_t block_height_gobs;             // alto del bloque block-linear, en GOBs
    uint32_t block_depth;
    // GOBs por fila de bloques (el pitch): lo justo para el ancho, o alineado al ancho de tile en las texturas
    // empaquetadas (sparse de MH Rise: 16 GOBs, tiles de 64 KB). 0 = lo justo.
    uint32_t row_gobs = 0;
};

struct TextureLayout {
    bool linear = false;
    uint32_t stride = 0;  // solo lineal
    uint32_t bytes_per_block = 4, block_w = 1, block_h = 1;
    std::vector<LevelLayout> levels;
    size_t layer_size = 0;  // tamaño de una capa (todas sus mips), alineado al bloque
    uint32_t layers = 1;
    size_t total_size() const { return layer_size * layers; }
};

// block_height_log2 >= 0: el alto de bloque del nivel 0 lo fija la textura (NVNpackagedTextureLayout, byte 0) en vez de
// salir del alto; tile_width_log2: alineación del pitch en GOBs (byte 1; el tile_width_spacing del TIC de Maxwell).
TextureLayout compute_layout(uint32_t width, uint32_t height, uint32_t depth, uint32_t layers, uint32_t levels,
                             uint32_t bytes_per_block, uint32_t block_w, uint32_t block_h, bool linear, uint32_t stride,
                             bool is_3d, int block_height_log2 = -1, uint32_t tile_width_log2 = 0);

// Copia un nivel de una capa de block-linear (src) a lineal compacto (dst, filas de width_blocks, cortes de
// height_blocks filas). zero_chunks: si no es nulo, suma los trozos de 16 bytes copiados enteros que son todo ceros
// (con 16 bytes por bloque, los bloques a cero), mirados en src al copiarlos: dst es el staging (memoria de la GPU sin
// caché de la CPU), leerlo después cuesta lo que una lectura de la RAM por cada carga.
void deswizzle_level(const LevelLayout& level, uint32_t bytes_per_block, const uint8_t* src, uint8_t* dst,
                     size_t* zero_chunks = nullptr);
// Lo mismo para las filas [row_begin, row_end) del nivel (fila r: corte r / height_blocks, fila r % height_blocks): trozos
// independientes, para repartir un nivel grande entre hilos.
void deswizzle_rows(const LevelLayout& level, uint32_t bytes_per_block, const uint8_t* src, uint8_t* dst, size_t row_begin,
                    size_t row_end, size_t* zero_chunks = nullptr);
// La de antes (un memcpy de tamaño variable por cada trozo de 16 bytes), para <datos>/subida_v1.
void deswizzle_level_v1(const LevelLayout& level, uint32_t bytes_per_block, const uint8_t* src, uint8_t* dst);
// Trozos de 16 bytes todo ceros en [p, p + size) (size múltiplo de 16).
size_t count_zero_chunks(const uint8_t* p, size_t size);

// Una caja de un nivel, en bloques del formato (texels si no está comprimido).
struct BlockBox {
    uint32_t x, y, z, width, height, depth;
};
// nvnTextureWriteTexels / ReadTexels: la caja entre memoria lineal (filas de row_pitch bytes, cortes de slice_pitch) y
// el nivel en block-linear (tiled apunta al inicio del nivel). Una es la inversa exacta de la otra y usan las mismas
// direcciones que deswizzle_level.
void swizzle_region(const LevelLayout& level, uint32_t bytes_per_block, const BlockBox& box, const uint8_t* src,
                    size_t row_pitch, size_t slice_pitch, uint8_t* tiled);
void deswizzle_region(const LevelLayout& level, uint32_t bytes_per_block, const BlockBox& box, const uint8_t* tiled,
                      uint8_t* dst, size_t row_pitch, size_t slice_pitch);

}  // namespace nvn
