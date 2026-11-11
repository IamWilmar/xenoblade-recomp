#pragma once
// Cola continua de la salida de audio (audio.cpp). Sin dependencias de AAudio ni de nn::os, para probarla fuera
// del dispositivo (tools/audio_cola_prueba.cpp).
//
// El sink de Wwise de MH Rise cierra y vuelve a abrir su AudioOut muchas veces durante el juego (Term + Init del
// dispositivo: Stop, Close, Open, 4 Append de silencio de cebado, Start). Con la cola de antes, cada reapertura
// tiraba el audio ya mezclado que quedaba en cola (hasta 4 buffers, 85 ms) y tocaba en su lugar los 4 buffers de
// silencio del cebado: con reaperturas frecuentes no se oía casi nada, y solo sonaba seguido cuando el juego se
// atascaba (sin mensajes nuevos para Wwise no hay reaperturas).
//
// Aquí el PCM de cada buffer se copia al encolarlo y la cola es una sola para todo el proceso:
//  - un buffer se devuelve al juego cuando la lectura (el callback de AAudio, o el reloj de reserva) pasa de su
//    final: el ritmo al que se devuelven sigue siendo el reloj del audio de Wwise;
//  - al parar una salida, sus buffers sin tocar se devuelven ya (como en la Switch), pero su audio se queda en la
//    cola y se sigue oyendo ("huérfano");
//  - los buffers de cebado (silencio encolado antes de Start) se cubren con ese audio huérfano: se devuelven al
//    ritmo de la lectura como si sonaran, pero suena lo que quedaba de la salida anterior. La latencia y el reloj
//    de Wwise no cambian; deja de haber 85 ms de silencio y 85 ms perdidos por reapertura.
// Todo con el lock de quien la usa (SharedStream::m_ en audio.cpp).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <vector>

namespace nnsdk::audio {

class ColaContinua {
public:
    // `continuidad` false: como la cola de antes (Stop tira el audio no tocado y el cebado suena como silencio).
    ColaContinua(size_t capacidad_frames, int canales) : cap_(capacidad_frames), canales_(canales) {
        pcm_.resize(cap_ * canales_);
    }

    // Encola un buffer del juego de `frames` frames. `cebado`: silencio encolado antes de arrancar la salida.
    void encolar(void* out, void* buffer, const int16_t* datos, size_t frames, bool cebado, bool continuidad) {
        uint64_t fin;
        if (cebado && continuidad && v_ < w_) {
            // Lo cubre el audio huérfano que queda en la cola; si no llega, el resto es silencio.
            v_ = std::max(v_, r_);
            fin = v_ + frames;
            if (fin > w_) escribir(nullptr, size_t(fin - w_));
            v_ = fin;
            cubiertos_ += 1;
        } else {
            escribir(datos, frames);
            fin = w_;
            v_ = w_;
        }
        marcas_.push_back({out, buffer, fin});
    }

    // Lo que pide el callback: copia hasta `frames` frames a `destino` y devuelve cuántos había (el resto es
    // silencio). Los buffers que se acaban de tocar quedan listos para `liberar`.
    size_t leer(int16_t* destino, size_t frames) {
        size_t n = std::min<uint64_t>(frames, w_ - r_);
        copiar_de(r_, destino, n);
        r_ += n;
        if (frames > n) memset(destino + n * canales_, 0, (frames - n) * canales_ * sizeof(int16_t));
        return n;
    }

    // Avance del reloj de reserva (AAudio no llama): se descarta como si se hubiera tocado.
    size_t descartar(size_t frames) {
        size_t n = std::min<uint64_t>(frames, w_ - r_);
        r_ += n;
        return n;
    }

    // Devuelve (f(out, buffer)) los buffers ya tocados enteros.
    template <class F>
    void liberar(F&& f) {
        while (!marcas_.empty() && marcas_.front().fin <= r_) {
            f(marcas_.front().out, marcas_.front().buffer);
            marcas_.pop_front();
        }
    }

    // Stop o Close de una salida: sus buffers pendientes se devuelven ya con f(out, buffer). Con continuidad, su
    // audio se queda en la cola; sin ella se tira lo que no se ha tocado (si no hay otra salida detrás).
    template <class F>
    void parar(const void* out, bool continuidad, F&& f) {
        uint64_t fin_otros = r_;
        for (auto it = marcas_.begin(); it != marcas_.end();) {
            if (it->out == out) {
                f(it->out, it->buffer);
                it = marcas_.erase(it);
            } else {
                fin_otros = std::max(fin_otros, it->fin);
                ++it;
            }
        }
        if (!continuidad) w_ = std::max(r_, fin_otros);
        v_ = r_;  // el cebado de la próxima salida empieza a contar desde lo que suena ahora
    }

    size_t en_cola() const { return size_t(w_ - r_); }     // frames por tocar
    size_t marcas() const { return marcas_.size(); }       // buffers del juego sin devolver
    uint64_t leidos() const { return r_; }
    uint64_t perdidos() const { return perdidos_; }        // frames tirados por desbordar la cola
    uint64_t cubiertos() const { return cubiertos_; }      // buffers de cebado cubiertos con audio huérfano

private:
    struct Marca {
        void* out;
        void* buffer;
        uint64_t fin;  // posición de la cola (en frames) en la que acaba
    };

    // Escribe al final (datos null: silencio). Si no cabe, se tira lo más antiguo (y se devuelve lo que tapaba).
    void escribir(const int16_t* datos, size_t frames) {
        if (frames > cap_) {  // no cabe ni con la cola vacía: se queda el final
            size_t salta = frames - cap_;
            if (datos) datos += salta * canales_;
            perdidos_ += salta;
            w_ += salta;
            r_ = std::max(r_, w_);
            frames = cap_;
        }
        if (w_ - r_ + frames > cap_) {
            uint64_t sobra = w_ - r_ + frames - cap_;
            r_ += sobra;
            perdidos_ += sobra;
        }
        size_t hecho = 0;
        while (hecho < frames) {
            size_t i = size_t((w_ + hecho) % cap_), n = std::min(frames - hecho, cap_ - i);
            int16_t* d = pcm_.data() + i * canales_;
            if (datos)
                memcpy(d, datos + hecho * canales_, n * canales_ * sizeof(int16_t));
            else
                memset(d, 0, n * canales_ * sizeof(int16_t));
            hecho += n;
        }
        w_ += frames;
    }
    void copiar_de(uint64_t pos, int16_t* destino, size_t frames) const {
        size_t hecho = 0;
        while (hecho < frames) {
            size_t i = size_t((pos + hecho) % cap_), n = std::min(frames - hecho, cap_ - i);
            memcpy(destino + hecho * canales_, pcm_.data() + i * canales_, n * canales_ * sizeof(int16_t));
            hecho += n;
        }
    }

    size_t cap_;
    int canales_;
    std::vector<int16_t> pcm_;
    std::deque<Marca> marcas_;
    uint64_t r_ = 0, w_ = 0;  // lectura y escritura, en frames desde el principio
    uint64_t v_ = 0;          // hasta dónde llega el cebado cubierto con audio huérfano
    uint64_t perdidos_ = 0, cubiertos_ = 0;
};

// Reloj de reserva: si AAudio deja de llamar (arranque del stream, desconexión, el hilo del callback sin CPU),
// la cola avanza con el reloj monotónico para que Wwise siga viendo buffers devueltos a su ritmo y no dé la
// salida por muerta. Mientras AAudio llama, no hace nada.
struct RelojDeReserva {
    int64_t umbral_ns = 40'000'000;  // AAudio en marcha: callbacks cada 2 ms; con 40 ms sin ninguno, toma el relevo
    int64_t ultimo_ns = 0;           // hasta dónde se ha contado (mientras AAudio llama, su último callback)
    double resto = 0;

    // Frames que tocaría descartar ahora. `ultimo_callback_ns` 0 = AAudio aún no ha llamado nunca. Al tomar el
    // relevo cuenta desde el último callback: el hueco entero, no solo lo que pase desde que se nota.
    size_t frames(int64_t ahora_ns, int64_t ultimo_callback_ns, int hz) {
        if (ultimo_callback_ns && ahora_ns - ultimo_callback_ns < umbral_ns) {
            ultimo_ns = ultimo_callback_ns;  // AAudio lleva el ritmo
            resto = 0;
            return 0;
        }
        if (!ultimo_ns) ultimo_ns = ahora_ns;  // nunca ha habido ritmo: se empieza a contar ahora
        if (ahora_ns <= ultimo_ns) return 0;
        double f = double(ahora_ns - ultimo_ns) * hz / 1e9 + resto;
        ultimo_ns = ahora_ns;
        size_t n = size_t(f);
        resto = f - double(n);
        return n;
    }
};

}  // namespace nnsdk::audio
