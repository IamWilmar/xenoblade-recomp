#pragma once
// Memoria de dispositivo para las imágenes: bloques grandes repartidos entre muchas texturas. Una
// reserva de Vulkan por textura agotaba el driver del Adreno hacia las 1.100 texturas (vkAllocateMemory
// fallaba con ~300 MB en uso y el driver acababa cayéndose), así que solo las enormes van aparte.

#include "nnsdk/nvn/vk.h"  // define la plataforma antes de vulkan.h

namespace nvn {

struct GpuAllocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, size = 0;
    int block = -1;  // -1: reserva propia (memory es suya)
    uint32_t type = 0;
};

// Reserva para esos requisitos en ese tipo de memoria; false si el driver no da más.
bool gpu_alloc(const VkMemoryRequirements& req, uint32_t memory_type, GpuAllocation& out);
// Devuelve el rango (o la reserva propia) y deja `a` vacío.
void gpu_free(GpuAllocation& a);
// DEPURACIÓN/perfil: bloques y reservas propias vivas, y bytes de los bloques.
void gpu_memory_stats(size_t* blocks, size_t* dedicated, VkDeviceSize* block_bytes);

}  // namespace nvn
