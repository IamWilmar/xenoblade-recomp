#pragma once
// Objetos de estado de NVN (blend, color, máscara de canales, profundidad/stencil, polígono,
// multisample, atributos y streams de vértice). Viven dentro de structs del juego cuyo tamaño sale de
// las direcciones de la traza (p. ej. tres BlendState seguidos a 8 bytes): aquí se codifican en esos
// bytes con nuestro propio formato. Los valores de los enums siguen el orden estilo OpenGL de NVN; los
// que el juego usa al arrancar lo confirman (ONE/ZERO, LEQUAL, ALWAYS, BACK, FILL, REPEAT…).

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn {

struct BlendState {  // 8 bytes
    // Render target (0-7) y, en los bits altos del mismo byte, el blend avanzado: NVNblendAdvancedOverlap (UNCORRELATED 0,
    // DISJOINT 1, CONJOINT 2) y los dos booleanos negados (NVN los pone a TRUE por defecto). Así lo de por defecto son ceros,
    // como antes: los bytes van tal cual a PipelineKey y a pipelines_estado.bin.
    uint8_t target : 3, advanced_overlap : 2, advanced_src_not_premultiplied : 1, advanced_dst_not_normalized : 1, : 1;
    uint8_t src_rgb, dst_rgb, src_alpha, dst_alpha;  // NVNblendFunc (ZERO = 1, ONE = 2…)
    uint8_t eq_rgb, eq_alpha;                        // NVNblendEquation (ADD = 1…)
    uint8_t advanced_mode;                           // NVNblendAdvancedMode (NONE 0); era el relleno, siempre a 0
};
struct ColorState {  // 4 bytes
    uint8_t blend_enable;  // bit por render target
    uint8_t logic_op;
    uint8_t alpha_test;    // NVNalphaFunc (NEVER = 1 … ALWAYS = 8)
    uint8_t pad;
};
struct ChannelMaskState {  // 4 bytes: 4 bits (RGBA) por render target
    uint32_t masks;
};
struct DepthStencilState {  // 8 bytes; funciones NVNdepthFunc/NVNstencilFunc (NEVER = 1 … ALWAYS = 8)
    uint64_t depth_test : 1, depth_write : 1, depth_func : 4, stencil_test : 1;
    uint64_t front_func : 4, back_func : 4;
    uint64_t front_fail : 4, front_zfail : 4, front_zpass : 4;  // NVNstencilOp (KEEP = 1 …)
    uint64_t back_fail : 4, back_zfail : 4, back_zpass : 4;
};
struct PolygonState {  // 4 bytes
    uint8_t cull_face;     // NVNface: NONE 0, FRONT 1, BACK 2, FRONT_AND_BACK 3
    uint8_t front_face;    // CW 0, CCW 1
    uint8_t polygon_mode;  // POINT 0, LINE 1, FILL 2
    uint8_t offset_enables;
};
struct MultisampleState {  // 24 bytes, como en NVN (Xenoblade los pone seguidos: main+0x7276f4 en +0x30, el siguiente
                           // objeto en +0x48). La modulación de cobertura, el color de cobertura (muestras mixtas de
                           // NVIDIA) y las posiciones programables de las muestras no tienen equivalente en Vulkan en el
                           // Adreno: se guardan para que los getters digan lo que se puso.
    uint8_t enable : 1, sample_locations_enable : 1, sample_locations_grid_enable : 1, : 5;
    uint8_t samples, alpha_to_coverage, alpha_to_coverage_dither;
    uint8_t raster_samples, coverage_modulation, coverage_to_color, coverage_to_color_output;
    // SetSampleLocations: las 16 posiciones de Maxwell (rejilla x muestras = 16), x en los bits 0-3 e y en los 4-7, en
    // dieciseisavos de píxel.
    uint8_t sample_locations[16];
};
struct VertexAttribState {  // 4 bytes
    uint8_t format;   // NVNformat
    uint8_t stream;
    uint16_t offset;
};
struct VertexStreamState {  // 8 bytes
    uint32_t stride;
    uint32_t divisor;
};

static_assert(sizeof(BlendState) == 8 && sizeof(ColorState) == 4 && sizeof(ChannelMaskState) == 4 &&
              sizeof(DepthStencilState) == 8 && sizeof(PolygonState) == 4 && sizeof(VertexAttribState) == 4 &&
              sizeof(VertexStreamState) == 8 && sizeof(MultisampleState) == 24);

VkBlendFactor vk_blend_factor(int nvn);
VkBlendOp vk_blend_op(int nvn);
VkCompareOp vk_compare_op(int nvn);
VkStencilOp vk_stencil_op(int nvn);
VkCullModeFlags vk_cull_mode(int nvn_face);
VkPolygonMode vk_polygon_mode(int nvn);
VkPrimitiveTopology vk_topology(int nvn_primitive);
VkIndexType vk_index_type(int nvn);
uint32_t index_size(int nvn);

}  // namespace nvn
