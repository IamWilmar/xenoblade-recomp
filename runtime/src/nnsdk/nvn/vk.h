#pragma once
// Contexto Vulkan único detrás de NVN: instancia, GPU, dispositivo y cola gráfica.

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

namespace nvn {

struct VkContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceMemoryProperties memory{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    // VK_QCOM_render_pass_store_ops activa: los render passes pueden usar VK_ATTACHMENT_STORE_OP_NONE.
    bool store_op_none = false;
    // VK_GOOGLE_display_timing activa: se puede pedir a qué hora se muestra cada imagen (LSFG: el real 16,7 ms
    // después del generado, para que el compositor no tire uno de los dos).
    bool display_timing = false;
    bool push_descriptor = false;  // VK_KHR_push_descriptor activa (el escalado FSR, fsr.cpp)
    bool shader_float16 = false;  // VK_KHR_shader_float16_int8 encendida: los shaders FP16 de LSFG (lsfg/)
    // VK_EXT_descriptor_indexing con shaderSampledImageArrayNonUniformIndexing: los arrays bindless (decals, sondas) se
    // indexan por píxel con NonUniform (nvnshader::Options::nonuniform_indexing).
    bool nonuniform_sampled = false;
    // VK_EXT_custom_border_color (sin formato) y VK_EXT_sampler_filter_minmax: bordes exactos y reducción MIN/MAX.
    bool custom_border_color = false, filter_minmax = false;
    // VK_KHR_sampler_mirror_clamp_to_edge activa: VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE es válido (los wraps
    // MIRROR_CLAMP, MIRROR_CLAMP_TO_EDGE y MIRROR_CLAMP_TO_BORDER de NVN). Sin ella se aproximan con CLAMP_TO_EDGE.
    bool mirror_clamp_to_edge = false;
    // logicOp (feature del núcleo): las operaciones lógicas de nvnColorStateSetLogicOp (Xenoblade: OR en las máscaras de
    // luces por píxel, RG32UI). Sin ella las pipelines salen sin operación lógica, como antes.
    bool logic_op = false;
    uint32_t max_custom_border_samplers = 0;
    // VK_KHR_draw_indirect_count: los multi-draw de NVN con el número de draws en la GPU (nulos sin la extensión).
    PFN_vkCmdDrawIndexedIndirectCountKHR draw_indexed_indirect_count = nullptr;
    PFN_vkCmdDrawIndirectCountKHR draw_indirect_count = nullptr;
};

// Se crea en la primera llamada; nullptr si Vulkan no arranca.
VkContext* vk();

const char* vk_result_name(VkResult r);

}  // namespace nvn
