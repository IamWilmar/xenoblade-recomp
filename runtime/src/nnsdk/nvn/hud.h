#pragma once
// Contador de FPS arriba a la izquierda: solo el número de frames en pantalla por segundo (con los generados de
// LSFG), pequeño y en color (verde a su ritmo, amarillo corto, rojo mal). Lo dibuja la CPU en un buffer pequeño que se
// copia tal cual sobre la imagen del swapchain, después del escalado del frame. <datos>/sin_fps lo oculta en vivo.

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn::hud {

// window.cpp: se presentó un frame real (en el ritmo del juego, steady_clock en ns).
void real_frame(int64_t now_ns);
// lsfg/framegen.cpp: se presentó un frame generado.
void generated_frame();
// Copia el HUD sobre la imagen del swapchain, que está en TRANSFER_DST_OPTIMAL (entre el blit y la barrera a
// PRESENT_SRC). format es el del swapchain: solo se dibuja si es RGBA8 o BGRA8.
void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format);

}  // namespace nvn::hud
