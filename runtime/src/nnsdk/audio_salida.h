#pragma once
// Lo que la salida de AAudio del proceso (audio.cpp) deja usar a otros módulos de nn::audio. Hoy solo el
// AudioRenderer (audio_renderer.cpp), que no abre stream propio: suma su mezcla a la del stream único, que ya está en
// marcha (regla del proyecto: la salida tiene que arrancar al instante).

#include <cstddef>
#include <cstdint>

namespace nnsdk::audio {

constexpr int kSalidaHz = 48000, kSalidaCanales = 2;  // el stream único: PCM16 estéreo a 48 kHz

// Una fuente más del stream, además de las colas de los AudioOut. Las dos funciones corren con el cerrojo del stream
// tomado, así que no pueden volver a entrar en audio.cpp ni tardar: sin reservas de memoria, sin log, sin ficheros.
struct FuenteExtra {
    // En el callback de AAudio, con `salida` ya mezclada de los AudioOut: suma lo suyo, saturando.
    void (*mezclar)(int16_t* salida, size_t frames);
    // Reloj de reserva: AAudio no llama (aún no ha arrancado, no se pudo abrir, se desconectó) y han pasado `frames`:
    // la fuente avanza como si hubieran sonado, para que el juego siga recibiendo sus buffers a tiempo real.
    void (*descartar)(size_t frames);
};

// Abre (una vez, en segundo plano) el stream de AAudio del proceso.
void preparar_salida();
// `fuente` vive para siempre (un objeto estático); nullptr la quita.
void fijar_fuente_extra(const FuenteExtra* fuente);

}  // namespace nnsdk::audio
