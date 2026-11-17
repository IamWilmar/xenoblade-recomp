#pragma once
// Escalado de la imagen del juego a la pantalla con FSR 1 de AMD (EASU reconstruye de 1280x720 a 1920x1080, RCAS añade
// nitidez), en vez del blit lineal: el mismo coste que un efecto de posproceso pequeño (~1 ms) y bastante más nítido.
// <datos>/sin_fsr vuelve al blit de siempre en vivo.

#include "nnsdk/nvn/vk.h"

namespace nvn::fsr {

// Escala src (vista muestreable, en GENERAL, de tamaño src_extent) a dst, una imagen del swapchain en
// TRANSFER_DST_OPTIMAL (dst_extent, formato dst_format). false si no puede (formato, recursos, push descriptors): quien
// llama hace el blit de siempre.
// viewport: la zona de src (desde 0,0) que se escala; src_extent, el tamaño de la textura entera.
// easu = false: sin EASU, una sola pasada de RCAS sobre la entrada escalada con el filtro lineal (fsr_rcas_up.comp): menos
// de la mitad de GPU que el FSR entero y casi la misma nitidez (los bordes, algo menos limpios). Para los frames en los que
// el entero no cabe (Castlevania a 60 fps: 9-11 ms de juego y ~7 del FSR entero no caben en 16,7).
bool record(VkCommandBuffer cmd, VkImageView src, VkExtent2D src_extent, VkExtent2D viewport, VkImage dst,
            VkFormat dst_format, VkExtent2D dst_extent, bool easu = true);

// El blit lineal de siempre para una textura de ventana sRGB, sin perder su codificación. vkCmdBlitImage desde la imagen
// sRGB la decodifica al leerla y un swapchain UNORM no vuelve a codificar (la pantalla recibía valores lineales). Aquí se
// copian sus bytes tal cual (vkCmdCopyImage no convierte entre formatos compatibles) a una imagen propia del tamaño de la
// textura (src_extent) y formato raw_format (el UNORM del sRGB de src), y se escala esa zona a dst: lo mismo que haría el
// blit si la textura de la ventana fuera UNORM. src en GENERAL; dst, del swapchain (no sRGB), en TRANSFER_DST_OPTIMAL.
// false si no puede: quien llama hace el blit directo.
bool blit_encoded(VkCommandBuffer cmd, VkImage src, VkFormat raw_format, VkExtent2D src_extent, VkRect2D rect, VkImage dst,
                  VkExtent2D dst_extent);

}  // namespace nvn::fsr
