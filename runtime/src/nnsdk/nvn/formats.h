#pragma once
// NVNformat -> VkFormat y bytes por bloque. El orden del enum es el de NVN (estilo OpenGL);
// lo confirman los dos formatos que NieR registra para borrado rápido: 0x25 (RGBA8, el de
// la ventana de 1280x720) y 0x3f (R11G11B10F, el del buffer HDR).

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn {

struct FormatInfo {
    VkFormat vk;
    uint8_t bytes;   // por píxel, o por bloque si block > 1
    uint8_t block;   // ancho del bloque comprimido (1 = sin comprimir)
    const char* name;
    uint8_t block_h = 0;  // alto del bloque si no es cuadrado (ASTC 5x4, 8x6…); 0 = igual al ancho
    // Qué componente del VkFormat lleva cada canal NVN R,G,B,A (selector NVN: R 2, G 3, B 4, A 5, ONE 1) cuando el VkFormat
    // no reproduce el orden en memoria. Los empaquetados de 16 bits de NVN llevan el primer componente en los bits bajos
    // (RGBA4 = A4B4G4R4 de Vulkan: medido en PC079402_BODY_SHY, una textura gris con el alfa en el nibble alto, bits 12-15;
    // VK_FORMAT_R4G4B4A4_UNORM_PACK16 lee R de esos bits). {0} = identidad. El ejecutor lo compone con el swizzle de la vista.
    uint8_t swz[4] = {0, 0, 0, 0};
    bool has_swizzle() const { return swz[0] != 0; }
    uint32_t block_height() const { return block_h ? block_h : block; }
};

// nullptr si no lo conocemos todavía (se registra en el log).
const FormatInfo* format_info(int nvn_format);

}  // namespace nvn
