#pragma once
// Una voz del AudioRenderer por software (audio_renderer.cpp): su cola de WaveBuffer, el conversor de frecuencia y la
// mezcla a la salida del dispositivo. Sin dependencias de AAudio ni de nn::os, para probarla fuera del dispositivo
// (tools/tests/audio_renderer_prueba.cpp).
//
// Modela lo que hace con una voz PcmInt16 el servidor de audio de la consola, que en este SDK viene dentro del propio
// módulo sdk (direcciones "sdk+0x…" = work/scope/sdk.elf):
//  - Estados Started/Stopped/Paused (server::VoiceInfo::UpdatePlayState, sdk+0x687e8). Al parar, lo que hubiera en cola
//    se da por consumido y posición, historia y fracción vuelven a cero (UpdateParametersForCommandGeneration,
//    sdk+0x68f88); en pausa se conserva todo.
//  - Un buffer se consume cuando la lectura pasa de su final; el acumulado (VoiceState::waveBuffersConsumed) es lo que
//    el cliente compara en GetReleasedWaveBuffer con los que ya ha devuelto (VoiceInfo+0x1b8 frente a +0x30,
//    sdk+0x46b80).
//  - PCM16 intercalado con los offsets en frames: muestra = buffer[(inicio + frame) * canales + canal] (DecodePcm16,
//    sdk+0x6d570).
//  - El volumen de la voz por el de mezcla, con rampa lineal de un frame de audio (5 ms) cuando cambia, y sin rampa en
//    una voz recién adquirida. Al cortar una voz que sonaba, su último valor se deja caer exponencialmente en vez de
//    saltar a cero (depop: ApplyDepop, sdk+0x6ddf0).
// La conversión va directa de la frecuencia de la voz a la del dispositivo con interpolación cúbica de 4 puntos (la
// consola convierte primero a la frecuencia del renderer, 32 o 48 kHz, con un polifásico de 4 taps, y su sink sube
// luego a 48 kHz). Con la misma frecuencia a los dos lados la interpolación devuelve la muestra tal cual.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nnsdk::audio {

constexpr int kVozCanalesMax = 6;    // canales de una voz (los VoiceChannelResource de un VoiceInfo, sdk+0x47304)
constexpr int kVozOndasMax = 4;      // VoiceType::WaveBufferCountMax (sdk+0xa36d0c)
constexpr int kRampaFrames = 240;    // un frame de audio de la consola a 48 kHz
// Caída del depop por muestra a 48 kHz: 0x7B29 en Q15, la constante del comando DepopForMixBuffers.
constexpr float kDepopCaida = 0.962189f;

// Un WaveBuffer ya en el mezclador. Los datos siguen en la memoria del juego (su memory pool), como en la consola: se
// leen al tocar y dejan de mirarse en cuanto el buffer se da por consumido.
struct OndaVoz {
    const int16_t* datos = nullptr;
    int32_t inicio = 0, fin = 0;  // frames [inicio, fin)
    bool bucle = false;           // WaveBuffer::loop: se repite hasta que paran la voz
};

class VozMezcla {
public:
    enum Estado : uint8_t { kSonando, kParada, kPausada };

    void reiniciar() { *this = VozMezcla{}; }

    // `ratio`: frames de la voz por frame de salida (frecuencia de la voz por su tono, entre la del dispositivo).
    void configurar(int canales, double ratio) {
        canales_ = canales < 1 ? 1 : canales > kVozCanalesMax ? kVozCanalesMax : canales;
        if (!(ratio > 0)) ratio = 1;
        if (ratio > 16) ratio = 16;
        paso_ = static_cast<uint64_t>(ratio * double(kUno) + 0.5);
        if (!paso_) paso_ = 1;
    }

    // false si la cola está llena (el cliente no deja pasar de 4: AppendWaveBuffer, sdk+0x46a58).
    bool encolar(const OndaVoz& o) {
        if (n_ == kVozOndasMax) return false;
        cola_[(cabeza_ + n_) % kVozOndasMax] = o;
        if (n_++ == 0) pos_ = o.inicio;
        saltar_vacios();
        return true;
    }

    // `depop` (2 valores, se le suma): lo que sonaba en el momento de cortar la voz.
    void fijar_estado(Estado nuevo, float* depop) {
        if (nuevo == estado_) return;
        if (nuevo == kParada) vaciar();
        cortar(depop);
        estado_ = nuevo;
    }

    // Toda la cola se da por consumida y la lectura vuelve al principio, sin cambiar de estado: lo que hace el servidor
    // al parar la voz, y lo que toca cuando la memoria de alguno de sus buffers deja de valer (su memory pool se
    // suelta con la voz parada o en pausa). Entera y no solo ese buffer: el juego los recoge por orden.
    void vaciar() {
        consumidos_ += static_cast<uint32_t>(n_);
        n_ = 0, cabeza_ = 0, pos_ = 0, vivos_ = 0;
        fase_ = kUno;
        std::memset(h_, 0, sizeof h_);
    }

    // Cuánto de cada canal de la voz va a cada canal de la salida. Con `rampa`, se llega en kRampaFrames.
    void fijar_ganancias(const float g[kVozCanalesMax][2], bool rampa) {
        bool cambia = false;
        for (int c = 0; c < kVozCanalesMax; c++)
            for (int k = 0; k < 2; k++) cambia |= g[c][k] != objetivo_[c][k];
        if (!cambia && rampa) return;
        std::memcpy(objetivo_, g, sizeof objetivo_);
        if (!rampa) {
            std::memcpy(g_, g, sizeof g_);
            rampa_ = 0;
            return;
        }
        for (int c = 0; c < kVozCanalesMax; c++)
            for (int k = 0; k < 2; k++) paso_g_[c][k] = (objetivo_[c][k] - g_[c][k]) / kRampaFrames;
        rampa_ = kRampaFrames;
    }

    // Suma a `salida` (estéreo intercalado, en unidades de PCM16) `frames` frames de la voz.
    void mezclar(float* salida, size_t frames) {
        if (estado_ != kSonando || (n_ == 0 && vivos_ == 0)) return;
        for (size_t i = 0; i < frames; i++) {
            while (fase_ >= kUno) {
                avanzar();
                fase_ -= kUno;
            }
            const float t = static_cast<float>(fase_) * (1.0f / float(kUno));
            float izq = 0, der = 0;
            for (int c = 0; c < canales_; c++) {
                const float s = interpolar(h_[c], t);
                izq += s * g_[c][0];
                der += s * g_[c][1];
            }
            salida[2 * i] += izq;
            salida[2 * i + 1] += der;
            ultimo_[0] = izq, ultimo_[1] = der;
            if (rampa_) {
                const bool fin = --rampa_ == 0;
                for (int c = 0; c < canales_; c++)
                    for (int k = 0; k < 2; k++) g_[c][k] = fin ? objetivo_[c][k] : g_[c][k] + paso_g_[c][k];
            }
            fase_ += paso_;
        }
        if (n_ == 0 && vivos_ == 0) ultimo_[0] = ultimo_[1] = 0;  // se acabó sola: nada que cortar
    }

    // El reloj de reserva: pasan `frames` frames de salida sin sonar.
    void descartar(size_t frames) {
        if (estado_ != kSonando || (n_ == 0 && vivos_ == 0)) return;
        fase_ += paso_ * frames;
        while (fase_ >= kUno) {
            fase_ -= kUno;
            if (n_ == 0 && vivos_ == 0) {  // ya no queda nada: el resto del tiempo no cuenta
                fase_ &= kUno - 1;
                break;
            }
            avanzar();
        }
        if (rampa_) {
            std::memcpy(g_, objetivo_, sizeof g_);
            rampa_ = 0;
        }
        ultimo_[0] = ultimo_[1] = 0;
    }

    // La voz deja de existir (ReleaseVoiceSlot, o el renderer se para o se cierra con ella sonando).
    void cortar(float* depop) {
        if (estado_ == kSonando && depop) depop[0] += ultimo_[0], depop[1] += ultimo_[1];
        ultimo_[0] = ultimo_[1] = 0;
    }

    Estado estado() const { return estado_; }
    uint32_t consumidos() const { return consumidos_; }  // acumulado desde reiniciar()
    // Frames de sus WaveBuffer que la voz ha tocado desde reiniciar(), a su propia frecuencia (lo que el servidor de
    // audio devuelve como playedSampleCount): el reloj con el que CRI Atom sabe por dónde va cada sonido.
    uint64_t tocados() const { return tocados_; }
    int en_cola() const { return n_; }
    // ¿Algún buffer en cola dentro de [base, base + bytes)? Mientras lo haya, su memory pool no se puede soltar.
    bool usa(const void* base, size_t bytes) const {
        const auto lo = reinterpret_cast<uintptr_t>(base), hi = lo + bytes;
        for (int i = 0; i < n_; i++) {
            const auto p = reinterpret_cast<uintptr_t>(cola_[(cabeza_ + i) % kVozOndasMax].datos);
            if (p >= lo && p < hi) return true;
        }
        return false;
    }

private:
    static constexpr uint64_t kUno = uint64_t{1} << 32;  // un frame de la voz en la fase (32.32)

    // Hermite de 4 puntos entre y[1] y y[2]; en t = 0 da y[1] exacto.
    static float interpolar(const float* y, float t) {
        const float c1 = 0.5f * (y[2] - y[0]);
        const float c2 = y[0] - 2.5f * y[1] + 2.0f * y[2] - 0.5f * y[3];
        const float c3 = 0.5f * (y[3] - y[0]) + 1.5f * (y[1] - y[2]);
        return ((c3 * t + c2) * t + c1) * t + y[1];
    }

    // Los buffers sin datos se dan por tocados en cuanto llegan a la cabeza.
    void saltar_vacios() {
        while (n_ > 0) {
            const OndaVoz& o = cola_[cabeza_];
            if (o.datos && o.fin > o.inicio) break;
            sacar();
        }
    }
    void sacar() {
        cabeza_ = (cabeza_ + 1) % kVozOndasMax;
        consumidos_++;
        if (--n_ > 0) pos_ = cola_[cabeza_].inicio;
    }

    // Un frame más de la voz a la historia: el siguiente de la cola o, si no hay, silencio.
    void avanzar() {
        for (int c = 0; c < canales_; c++) h_[c][0] = h_[c][1], h_[c][1] = h_[c][2], h_[c][2] = h_[c][3];
        if (n_ == 0) {
            for (int c = 0; c < canales_; c++) h_[c][3] = 0;
            if (vivos_ > 0) vivos_--;
            return;
        }
        const OndaVoz& o = cola_[cabeza_];
        const int16_t* frame = o.datos + static_cast<ptrdiff_t>(pos_) * canales_;
        for (int c = 0; c < canales_; c++) h_[c][3] = frame[c];
        vivos_ = 3;
        tocados_++;
        if (++pos_ >= o.fin) {
            if (o.bucle) {
                pos_ = o.inicio;
            } else {
                sacar();
                saltar_vacios();
            }
        }
    }

    OndaVoz cola_[kVozOndasMax];
    int cabeza_ = 0, n_ = 0;
    int32_t pos_ = 0;            // frame del buffer de la cabeza que toca leer
    uint32_t consumidos_ = 0;
    uint64_t tocados_ = 0;
    int canales_ = 1;
    uint64_t paso_ = kUno, fase_ = kUno;  // con fase_ >= kUno toca traer otro frame antes de interpolar
    float h_[kVozCanalesMax][4] = {};     // los 4 últimos frames de la voz, por canal
    int vivos_ = 0;                       // frames de la historia que aún no son silencio de relleno
    float g_[kVozCanalesMax][2] = {}, objetivo_[kVozCanalesMax][2] = {}, paso_g_[kVozCanalesMax][2] = {};
    int rampa_ = 0;
    float ultimo_[2] = {};  // lo último que sumó a la salida
    Estado estado_ = kParada;
};

}  // namespace nnsdk::audio
