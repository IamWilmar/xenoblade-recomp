// Funciones de los objetos de estado de NVN y su traducción a Vulkan (ver state.h).

#include "nnsdk/nvn/state.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#include "nnsdk/nvn/nvn.h"

namespace nvn {

VkBlendFactor vk_blend_factor(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO;
        case 2: return VK_BLEND_FACTOR_ONE;
        case 3: return VK_BLEND_FACTOR_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: return VK_BLEND_FACTOR_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 16: return VK_BLEND_FACTOR_SRC1_COLOR;
        case 17: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case 18: return VK_BLEND_FACTOR_SRC1_ALPHA;
        case 19: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp vk_blend_op(int e) {
    switch (e) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

VkCompareOp vk_compare_op(int f) {
    switch (f) {
        case 1: return VK_COMPARE_OP_NEVER;
        case 2: return VK_COMPARE_OP_LESS;
        case 3: return VK_COMPARE_OP_EQUAL;
        case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case 5: return VK_COMPARE_OP_GREATER;
        case 6: return VK_COMPARE_OP_NOT_EQUAL;
        case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        default: return VK_COMPARE_OP_ALWAYS;
    }
}

VkStencilOp vk_stencil_op(int o) {
    switch (o) {
        case 2: return VK_STENCIL_OP_ZERO;
        case 3: return VK_STENCIL_OP_REPLACE;
        case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 6: return VK_STENCIL_OP_INVERT;
        case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

VkCullModeFlags vk_cull_mode(int face) {
    switch (face) {
        case 1: return VK_CULL_MODE_FRONT_BIT;
        case 2: return VK_CULL_MODE_BACK_BIT;
        case 3: return VK_CULL_MODE_FRONT_AND_BACK;
        default: return VK_CULL_MODE_NONE;
    }
}

VkPolygonMode vk_polygon_mode(int m) {
    return m == 0 ? VK_POLYGON_MODE_POINT : m == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
}

// NVNdrawPrimitive: orden de OpenGL. Los quads y polígonos no existen en Vulkan (se avisa al dibujar).
VkPrimitiveTopology vk_topology(int p) {
    switch (p) {
        case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        case 10: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
        case 11: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        case 12: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
        case 13: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        case 14: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

// NVNindexType: UNSIGNED_BYTE 0, UNSIGNED_SHORT 1, UNSIGNED_INT 2.
VkIndexType vk_index_type(int t) { return t == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16; }
uint32_t index_size(int t) { return t == 0 ? 1 : t == 1 ? 2 : 4; }

}  // namespace nvn

using namespace nvn;

extern "C" {

// --- Blend ---
void nvnBlendStateSetDefaults(BlendState* s) {
    *s = {};
    s->src_rgb = s->src_alpha = 2;  // ONE
    s->dst_rgb = s->dst_alpha = 1;  // ZERO
    s->eq_rgb = s->eq_alpha = 1;    // ADD
}
void nvnBlendStateSetBlendTarget(BlendState* s, int target) { s->target = static_cast<uint8_t>(target); }
void nvnBlendStateSetBlendFunc(BlendState* s, int src_rgb, int dst_rgb, int src_alpha, int dst_alpha) {
    s->src_rgb = src_rgb, s->dst_rgb = dst_rgb, s->src_alpha = src_alpha, s->dst_alpha = dst_alpha;
}
void nvnBlendStateSetBlendEquation(BlendState* s, int rgb, int alpha) { s->eq_rgb = rgb, s->eq_alpha = alpha; }
int nvnBlendStateGetBlendTarget(const BlendState* s) { return s->target; }
void nvnBlendStateGetBlendFunc(const BlendState* s, int* src_rgb, int* dst_rgb, int* src_alpha, int* dst_alpha) {
    if (src_rgb) *src_rgb = s->src_rgb;
    if (dst_rgb) *dst_rgb = s->dst_rgb;
    if (src_alpha) *src_alpha = s->src_alpha;
    if (dst_alpha) *dst_alpha = s->dst_alpha;
}
void nvnBlendStateGetBlendEquation(const BlendState* s, int* rgb, int* alpha) {
    if (rgb) *rgb = s->eq_rgb;
    if (alpha) *alpha = s->eq_alpha;
}
// Blend avanzado (NV_blend_equation_advanced): el dispositivo dice que no lo tiene (GetInteger SUPPORTS_ADVANCED_BLEND_MODES
// = 0) y el ejecutor mezcla con las ecuaciones de siempre. Se guarda para los getters y se avisa si alguien lo pide.
void nvnBlendStateSetAdvancedMode(BlendState* s, int mode) {
    static std::atomic<int> logged{0};
    if (mode && first_times(logged, 4)) NVN_LOG("blend avanzado %d pedido: sin soporte, se mezcla normal\n", mode);
    s->advanced_mode = static_cast<uint8_t>(mode);
}
void nvnBlendStateSetAdvancedOverlap(BlendState* s, int overlap) { s->advanced_overlap = static_cast<uint8_t>(overlap); }
void nvnBlendStateSetAdvancedPremultipliedSrc(BlendState* s, nvn::Boolean premultiplied) {
    s->advanced_src_not_premultiplied = !premultiplied;
}
void nvnBlendStateSetAdvancedNormalizedDst(BlendState* s, nvn::Boolean normalized) {
    s->advanced_dst_not_normalized = !normalized;
}
int nvnBlendStateGetAdvancedMode(const BlendState* s) { return s->advanced_mode; }
int nvnBlendStateGetAdvancedOverlap(const BlendState* s) { return s->advanced_overlap; }
nvn::Boolean nvnBlendStateGetAdvancedPremultipliedSrc(const BlendState* s) { return !s->advanced_src_not_premultiplied; }
nvn::Boolean nvnBlendStateGetAdvancedNormalizedDst(const BlendState* s) { return !s->advanced_dst_not_normalized; }

// --- Color ---
void nvnColorStateSetDefaults(ColorState* s) {
    *s = {};
    s->logic_op = 3;    // NVN_LOGIC_OP_COPY: sin operación lógica (el 0 es CLEAR)
    s->alpha_test = 8;  // ALWAYS
}
void nvnColorStateSetBlendEnable(ColorState* s, int index, bool enable) {
    if (enable)
        s->blend_enable |= 1u << index;
    else
        s->blend_enable &= ~(1u << index);
}
void nvnColorStateSetLogicOp(ColorState* s, int op) { s->logic_op = op; }
void nvnColorStateSetAlphaTest(ColorState* s, int func) { s->alpha_test = func; }
nvn::Boolean nvnColorStateGetBlendEnable(const ColorState* s, int index) { return (s->blend_enable >> index) & 1; }
int nvnColorStateGetLogicOp(const ColorState* s) { return s->logic_op; }
int nvnColorStateGetAlphaTest(const ColorState* s) { return s->alpha_test; }

// --- Máscara de canales ---
void nvnChannelMaskStateSetDefaults(ChannelMaskState* s) { s->masks = 0xFFFFFFFF; }
void nvnChannelMaskStateSetChannelMask(ChannelMaskState* s, int index, bool r, bool g, bool b, bool a) {
    uint32_t bits = (r ? 1 : 0) | (g ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
    s->masks = (s->masks & ~(0xFu << (4 * index))) | bits << (4 * index);
}
void nvnChannelMaskStateGetChannelMask(const ChannelMaskState* s, int index, nvn::Boolean* r, nvn::Boolean* g,
                                       nvn::Boolean* b, nvn::Boolean* a) {
    const uint32_t bits = s->masks >> (4 * index);
    if (r) *r = bits & 1;
    if (g) *g = (bits >> 1) & 1;
    if (b) *b = (bits >> 2) & 1;
    if (a) *a = (bits >> 3) & 1;
}

// --- Profundidad y stencil ---
void nvnDepthStencilStateSetDefaults(DepthStencilState* s) {
    *s = {};
    s->depth_func = 2;                   // LESS
    s->front_func = s->back_func = 8;    // ALWAYS
    s->front_fail = s->front_zfail = s->front_zpass = 1;  // KEEP
    s->back_fail = s->back_zfail = s->back_zpass = 1;
}
void nvnDepthStencilStateSetDepthTestEnable(DepthStencilState* s, bool e) { s->depth_test = e; }
void nvnDepthStencilStateSetDepthWriteEnable(DepthStencilState* s, bool e) { s->depth_write = e; }
void nvnDepthStencilStateSetDepthFunc(DepthStencilState* s, int f) { s->depth_func = f; }
void nvnDepthStencilStateSetStencilTestEnable(DepthStencilState* s, bool e) { s->stencil_test = e; }
void nvnDepthStencilStateSetStencilFunc(DepthStencilState* s, int faces, int f) {
    if (faces & 1) s->front_func = f;
    if (faces & 2) s->back_func = f;
}
void nvnDepthStencilStateSetStencilOp(DepthStencilState* s, int faces, int fail, int zfail, int zpass) {
    if (faces & 1) s->front_fail = fail, s->front_zfail = zfail, s->front_zpass = zpass;
    if (faces & 2) s->back_fail = fail, s->back_zfail = zfail, s->back_zpass = zpass;
}
nvn::Boolean nvnDepthStencilStateGetDepthTestEnable(const DepthStencilState* s) { return s->depth_test; }
nvn::Boolean nvnDepthStencilStateGetDepthWriteEnable(const DepthStencilState* s) { return s->depth_write; }
int nvnDepthStencilStateGetDepthFunc(const DepthStencilState* s) { return s->depth_func; }
nvn::Boolean nvnDepthStencilStateGetStencilTestEnable(const DepthStencilState* s) { return s->stencil_test; }
// NVNface: FRONT 1, BACK 2 (con las dos, lo de la cara delantera).
int nvnDepthStencilStateGetStencilFunc(const DepthStencilState* s, int face) {
    return face == 2 ? s->back_func : s->front_func;
}
void nvnDepthStencilStateGetStencilOp(const DepthStencilState* s, int face, int* fail, int* zfail, int* zpass) {
    const bool back = face == 2;
    if (fail) *fail = back ? s->back_fail : s->front_fail;
    if (zfail) *zfail = back ? s->back_zfail : s->front_zfail;
    if (zpass) *zpass = back ? s->back_zpass : s->front_zpass;
}

// --- Polígono ---
void nvnPolygonStateSetDefaults(PolygonState* s) {
    *s = {};
    s->polygon_mode = 2;  // FILL
    s->front_face = 1;    // CCW
}
void nvnPolygonStateSetCullFace(PolygonState* s, int face) { s->cull_face = face; }
void nvnPolygonStateSetFrontFace(PolygonState* s, int face) { s->front_face = face; }
void nvnPolygonStateSetPolygonMode(PolygonState* s, int mode) { s->polygon_mode = mode; }
void nvnPolygonStateSetPolygonOffsetEnables(PolygonState* s, int enables) { s->offset_enables = enables; }
int nvnPolygonStateGetCullFace(const PolygonState* s) { return s->cull_face; }
int nvnPolygonStateGetFrontFace(const PolygonState* s) { return s->front_face; }
int nvnPolygonStateGetPolygonMode(const PolygonState* s) { return s->polygon_mode; }
int nvnPolygonStateGetPolygonOffsetEnables(const PolygonState* s) { return s->offset_enables; }

// --- Multisample ---
void nvnMultisampleStateSetDefaults(MultisampleState* s) {
    *s = {};
    s->enable = 1;  // NVN: multisample activo, 0 muestras (sin MSAA), dither de alpha-to-coverage activo
    s->samples = 1;
    s->alpha_to_coverage_dither = 1;
    memset(s->sample_locations, 0x88, sizeof s->sample_locations);  // posiciones programables apagadas; en el centro
}
void nvnMultisampleStateSetMultisampleEnable(MultisampleState* s, bool e) { s->enable = e; }
void nvnMultisampleStateSetSamples(MultisampleState* s, int samples) { s->samples = samples; }
void nvnMultisampleStateSetAlphaToCoverageEnable(MultisampleState* s, bool e) { s->alpha_to_coverage = e; }
void nvnMultisampleStateSetAlphaToCoverageDither(MultisampleState* s, bool e) { s->alpha_to_coverage_dither = e; }
void nvnMultisampleStateSetRasterSamples(MultisampleState* s, int samples) { s->raster_samples = samples; }
void nvnMultisampleStateSetCoverageModulationMode(MultisampleState* s, int mode) { s->coverage_modulation = mode; }
void nvnMultisampleStateSetCoverageToColorEnable(MultisampleState* s, bool e) { s->coverage_to_color = e; }
void nvnMultisampleStateSetCoverageToColorOutput(MultisampleState* s, int output) { s->coverage_to_color_output = output; }
nvn::Boolean nvnMultisampleStateGetMultisampleEnable(const MultisampleState* s) { return s->enable; }
int nvnMultisampleStateGetSamples(const MultisampleState* s) { return s->samples; }
nvn::Boolean nvnMultisampleStateGetAlphaToCoverageEnable(const MultisampleState* s) { return s->alpha_to_coverage; }
nvn::Boolean nvnMultisampleStateGetAlphaToCoverageDither(const MultisampleState* s) { return s->alpha_to_coverage_dither; }
int nvnMultisampleStateGetRasterSamples(const MultisampleState* s) { return s->raster_samples; }
int nvnMultisampleStateGetCoverageModulationMode(const MultisampleState* s) { return s->coverage_modulation; }
nvn::Boolean nvnMultisampleStateGetCoverageToColorEnable(const MultisampleState* s) { return s->coverage_to_color; }
int nvnMultisampleStateGetCoverageToColorOutput(const MultisampleState* s) { return s->coverage_to_color_output; }
// Posiciones programables de las muestras (NV_sample_locations): Vulkan las tiene con VK_EXT_sample_locations, que el
// Adreno no ofrece; se guardan (ver MultisampleState) y se avisa si se encienden.
void nvnMultisampleStateSetSampleLocationsEnable(MultisampleState* s, nvn::Boolean enable) {
    static std::atomic<int> logged{0};
    if (enable && first_times(logged, 2)) NVN_LOG("posiciones programables de muestras pedidas: sin soporte\n");
    s->sample_locations_enable = enable != 0;
}
nvn::Boolean nvnMultisampleStateGetSampleLocationsEnable(const MultisampleState* s) { return s->sample_locations_enable; }
void nvnMultisampleStateSetSampleLocationsGridEnable(MultisampleState* s, nvn::Boolean enable) {
    s->sample_locations_grid_enable = enable != 0;
}
nvn::Boolean nvnMultisampleStateGetSampleLocationsGridEnable(const MultisampleState* s) {
    return s->sample_locations_grid_enable;
}
// La rejilla de píxeles en la que se repiten las posiciones: Maxwell tiene 16, así que ancho x alto x muestras = 16
// (1 muestra: 4x4, 2: 4x2, 4: 2x2, 8: 2x1, 16: 1x1). Cuentan las de rasterización si las hay (muestras mixtas).
void nvnMultisampleStateGetSampleLocationsGrid(const MultisampleState* s, int* width, int* height) {
    const int samples = std::max<int>({s->samples, s->raster_samples, 1});
    int w = 4, h = 4;
    for (int n = 1; n < samples && w * h > 1; n *= 2) (w > h ? w : h) /= 2;
    if (width) *width = w;
    if (height) *height = h;
}
// values: pares (x, y) en [0, 1) desde la posición start; se guardan en dieciseisavos de píxel, como el hardware.
void nvnMultisampleStateSetSampleLocations(MultisampleState* s, int start, int count, const float* values) {
    if (!values) return;
    auto sixteenths = [](float v) { return static_cast<uint8_t>(std::clamp(static_cast<int>(v * 16.0f), 0, 15)); };
    for (int k = 0; k < count; k++) {
        const int i = start + k;
        if (i < 0 || i >= static_cast<int>(sizeof s->sample_locations)) continue;
        s->sample_locations[i] = static_cast<uint8_t>(sixteenths(values[2 * k]) | sixteenths(values[2 * k + 1]) << 4);
    }
}

// --- Vértices ---
void nvnVertexAttribStateSetDefaults(VertexAttribState* s) { *s = {}; }
void nvnVertexAttribStateSetFormat(VertexAttribState* s, int format, ptrdiff_t offset) {
    s->format = format;
    s->offset = static_cast<uint16_t>(offset);
}
void nvnVertexAttribStateSetStreamIndex(VertexAttribState* s, int stream) { s->stream = stream; }
// Los getters de estado no son de adorno: el renderer de MH Rise compara el layout nuevo con el enlazado a través de
// ellos (main+0x77178b0 para los streams) y solo vuelve a enlazar si difieren. Con GetStride/GetDivisor como stubs a 0
// (hasta 3d8197d) todos los layouts con el mismo número de streams eran "iguales" y las cabezas con skinning en compute
// (la del cazador) heredaban las zancadas del cuerpo anterior. El juego también pide estos dos por nombre: mejor el
// valor de verdad que un 0 si alguna ruta (diálogos, vista de equipo) los usa para lo mismo con los atributos.
void nvnVertexAttribStateGetFormat(const VertexAttribState* s, int* format, ptrdiff_t* offset) {
    if (format) *format = s->format;
    if (offset) *offset = s->offset;
}
int nvnVertexAttribStateGetStreamIndex(const VertexAttribState* s) { return s->stream; }
void nvnVertexStreamStateSetDefaults(VertexStreamState* s) { *s = {}; }
void nvnVertexStreamStateSetStride(VertexStreamState* s, ptrdiff_t stride) { s->stride = static_cast<uint32_t>(stride); }
void nvnVertexStreamStateSetDivisor(VertexStreamState* s, int divisor) { s->divisor = divisor; }
ptrdiff_t nvnVertexStreamStateGetStride(const VertexStreamState* s) { return s->stride; }
int nvnVertexStreamStateGetDivisor(const VertexStreamState* s) { return static_cast<int>(s->divisor); }

}  // extern "C"

NVN_PROC(nvnBlendStateSetDefaults);
NVN_PROC(nvnBlendStateSetBlendTarget);
NVN_PROC(nvnBlendStateSetBlendFunc);
NVN_PROC(nvnBlendStateSetBlendEquation);
NVN_PROC(nvnColorStateSetDefaults);
NVN_PROC(nvnColorStateSetBlendEnable);
NVN_PROC(nvnColorStateSetLogicOp);
NVN_PROC(nvnColorStateSetAlphaTest);
NVN_PROC(nvnChannelMaskStateSetDefaults);
NVN_PROC(nvnChannelMaskStateSetChannelMask);
NVN_PROC(nvnDepthStencilStateSetDefaults);
NVN_PROC(nvnDepthStencilStateSetDepthTestEnable);
NVN_PROC(nvnDepthStencilStateSetDepthWriteEnable);
NVN_PROC(nvnDepthStencilStateSetDepthFunc);
NVN_PROC(nvnDepthStencilStateSetStencilTestEnable);
NVN_PROC(nvnDepthStencilStateSetStencilFunc);
NVN_PROC(nvnDepthStencilStateSetStencilOp);
NVN_PROC(nvnPolygonStateSetDefaults);
NVN_PROC(nvnPolygonStateSetCullFace);
NVN_PROC(nvnPolygonStateSetFrontFace);
NVN_PROC(nvnPolygonStateSetPolygonMode);
NVN_PROC(nvnPolygonStateSetPolygonOffsetEnables);
NVN_PROC(nvnMultisampleStateSetDefaults);
NVN_PROC(nvnMultisampleStateSetMultisampleEnable);
NVN_PROC(nvnMultisampleStateSetSamples);
NVN_PROC(nvnMultisampleStateSetAlphaToCoverageEnable);
NVN_PROC(nvnBlendStateGetBlendTarget);
NVN_PROC(nvnBlendStateGetBlendFunc);
NVN_PROC(nvnBlendStateGetBlendEquation);
NVN_PROC(nvnColorStateGetBlendEnable);
NVN_PROC(nvnColorStateGetLogicOp);
NVN_PROC(nvnColorStateGetAlphaTest);
NVN_PROC(nvnChannelMaskStateGetChannelMask);
NVN_PROC(nvnDepthStencilStateGetDepthTestEnable);
NVN_PROC(nvnDepthStencilStateGetDepthWriteEnable);
NVN_PROC(nvnDepthStencilStateGetDepthFunc);
NVN_PROC(nvnDepthStencilStateGetStencilTestEnable);
NVN_PROC(nvnDepthStencilStateGetStencilFunc);
NVN_PROC(nvnDepthStencilStateGetStencilOp);
NVN_PROC(nvnPolygonStateGetCullFace);
NVN_PROC(nvnPolygonStateGetFrontFace);
NVN_PROC(nvnPolygonStateGetPolygonMode);
NVN_PROC(nvnPolygonStateGetPolygonOffsetEnables);
NVN_PROC(nvnMultisampleStateSetAlphaToCoverageDither);
NVN_PROC(nvnMultisampleStateSetRasterSamples);
NVN_PROC(nvnMultisampleStateSetCoverageModulationMode);
NVN_PROC(nvnMultisampleStateSetCoverageToColorEnable);
NVN_PROC(nvnMultisampleStateSetCoverageToColorOutput);
NVN_PROC(nvnMultisampleStateGetMultisampleEnable);
NVN_PROC(nvnMultisampleStateGetSamples);
NVN_PROC(nvnMultisampleStateGetAlphaToCoverageEnable);
NVN_PROC(nvnMultisampleStateGetAlphaToCoverageDither);
NVN_PROC(nvnMultisampleStateGetRasterSamples);
NVN_PROC(nvnMultisampleStateGetCoverageModulationMode);
NVN_PROC(nvnMultisampleStateGetCoverageToColorEnable);
NVN_PROC(nvnMultisampleStateGetCoverageToColorOutput);
NVN_PROC(nvnVertexAttribStateSetDefaults);
NVN_PROC(nvnVertexAttribStateSetFormat);
NVN_PROC(nvnVertexAttribStateSetStreamIndex);
NVN_PROC(nvnVertexAttribStateGetFormat);
NVN_PROC(nvnVertexAttribStateGetStreamIndex);
NVN_PROC(nvnVertexStreamStateSetDefaults);
NVN_PROC(nvnVertexStreamStateSetStride);
NVN_PROC(nvnVertexStreamStateSetDivisor);
NVN_PROC(nvnVertexStreamStateGetStride);
NVN_PROC(nvnVertexStreamStateGetDivisor);
NVN_PROC(nvnBlendStateSetAdvancedMode);
NVN_PROC(nvnBlendStateSetAdvancedOverlap);
NVN_PROC(nvnBlendStateSetAdvancedPremultipliedSrc);
NVN_PROC(nvnBlendStateSetAdvancedNormalizedDst);
NVN_PROC(nvnBlendStateGetAdvancedMode);
NVN_PROC(nvnBlendStateGetAdvancedOverlap);
NVN_PROC(nvnBlendStateGetAdvancedPremultipliedSrc);
NVN_PROC(nvnBlendStateGetAdvancedNormalizedDst);
NVN_PROC(nvnMultisampleStateSetSampleLocationsEnable);
NVN_PROC(nvnMultisampleStateGetSampleLocationsEnable);
NVN_PROC(nvnMultisampleStateSetSampleLocationsGridEnable);
NVN_PROC(nvnMultisampleStateGetSampleLocationsGridEnable);
NVN_PROC(nvnMultisampleStateGetSampleLocationsGrid);
NVN_PROC(nvnMultisampleStateSetSampleLocations);
