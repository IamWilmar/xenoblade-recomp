// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
// Generación de frames LSFG 2x (Lossless Scaling, con la DLL del jugador) en el presentador. El juego sigue
// presentando 30 frames reales por segundo; entre cada par se presenta uno generado y la pantalla de 60 Hz
// muestra 60. Apagado por defecto: <datos>/lsfg lo enciende en vivo y quitarlo lo apaga.

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn {

struct Texture;
struct Window;

namespace lsfg {

// Imágenes del swapchain: 4 si hay una Lossless.dll (el generado y el real van seguidos a FIFO y el segundo
// no debe esperar a que la pantalla suelte una), 3 si no (como siempre).
uint32_t swapchain_images(const VkSurfaceCapabilitiesKHR& caps);

// nvn::present, tras el ritmo de 33 ms y antes de presentar el frame real: si LSFG está encendido y listo,
// copia la textura de la ventana, calcula el frame intermedio entre el real anterior y este y lo presenta.
// El real lo presenta después window.cpp como siempre. Toma executor_mutex() y vk_queue_mutex(), en ese orden, por su cuenta.
// Tiempo de GPU que hay que reservarle a LSFG en cada frame real (ns; 0 si está apagado): executor.cpp se lo suma al
// tiempo de GPU que mide la resolución dinámica del juego, para que esta deje sitio al frame generado.
int64_t reserved_ns();
// Devuelve cuándo presentó el generado (steady_clock en ns, CLOCK_MONOTONIC), o 0 si no hubo; en *id, su presentID
// de VK_GOOGLE_display_timing (impar).
int64_t present_generated(Window& w, Texture* t, uint32_t* id = nullptr);

}  // namespace lsfg
}  // namespace nvn
