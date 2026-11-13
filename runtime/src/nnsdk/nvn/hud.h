#pragma once
// Contador de FPS arriba a la izquierda: solo el número de frames en pantalla por segundo (con los generados de
// LSFG), pequeño y en color (verde a su ritmo, amarillo corto, rojo mal). Lo dibuja la CPU en un buffer pequeño que se
// copia tal cual sobre la imagen del swapchain, después del escalado del frame. <datos>/sin_fps lo oculta en vivo.
// Panel de ajustes: un cajón de un tercio de la pantalla que sale del borde derecho, con una fila por ajuste (hoy, el
// interruptor del contador de FPS, que queda guardado con ese mismo fichero). Se abre con el gesto de "atrás" del sistema
// desde el borde derecho (el de la flecha) o deslizando un dedo de derecha a izquierda desde la mitad derecha de la
// pantalla; un toque en una fila cambia su ajuste, y un toque fuera, deslizar hacia la derecha, el mismo gesto del borde
// u 8 s sin tocarlo lo esconden. Los toques y el gesto los deja el loader en NnsdkInput (host.h).
// En las apps con pantalla de inicio (NIER_SETUP en el entorno) hay una segunda fila, "Cambiar carpeta": la app arranca el
// juego directamente si ya tiene su carpeta, y con ese interruptor (el fichero <datos>/elegir_carpeta) la pantalla de inicio
// vuelve a salir la próxima vez que se abra.
// En los ports con modo TV (NIER_TV_MODE en el entorno) hay otra, "Modo TV" (el fichero <datos>/modo_tv): el juego cree
// estar en la base (más resolución, mando Pro) desde la próxima vez que se abra.

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
