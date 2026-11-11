// Prueba fuera del dispositivo de la cola de audio (runtime/src/nnsdk/audio_cola.h) con un modelo del sink de
// Wwise de MH Rise: 4 buffers de 1024 frames, cebado de silencio, Start, sondeo cada 5 ms y reaperturas
// (Stop + Close + Open + cebado + Start) en los instantes que se den. AAudio: callback de 96 frames cada 2 ms,
// con huecos opcionales. Compara la cola de antes (continuidad off) con la continua.
//   c++ -std=c++20 -O2 -I runtime/src tools/audio_cola_prueba.cpp -o /tmp/prueba && /tmp/prueba
// Mide: % del tiempo con audio de Wwise, frames que Wwise mezcló y no sonaron, saltos en lo que suena, ritmo del
// reloj de Wwise (frames mezclados / tiempo real) y el mayor tiempo que Wwise pasa sin que se le devuelva nada.

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#include "nnsdk/audio_cola.h"

using nnsdk::audio::ColaContinua;
using nnsdk::audio::RelojDeReserva;

constexpr int kHz = 48000, kCh = 2, kFramesBuf = 1024, kBufs = 4, kBurst = 96;
constexpr int64_t kMs = 1'000'000;

struct Out {
    bool started = false;
    std::deque<int> released;  // índices de buffer
};

struct Escenario {
    const char* nombre;
    int64_t duracion_ns;
    std::vector<int64_t> reaperturas;           // instantes (ns) de Term+Init del dispositivo
    std::vector<std::pair<int64_t, int64_t>> sin_callback;  // tramos sin callbacks de AAudio
    int64_t aaudio_desde_ns = 0;                // AAudio no llama antes de esto
};

struct Resultado {
    double audible = 0, perdidos_ms = 0, ritmo = 0, max_sin_devolver_ms = 0, max_cola_ms = 0;
    long saltos = 0, reaperturas = 0;
};

Resultado simular(const Escenario& e, bool continuidad) {
    ColaContinua cola(kHz / 4, kCh);
    RelojDeReserva reloj;
    std::vector<std::vector<int16_t>> mem(kBufs, std::vector<int16_t>(kFramesBuf * kCh));
    Out* out = nullptr;
    std::vector<Out*> vivos;
    int queued = 0;               // lo que el sink cree que tiene en cola
    int64_t siguiente_muestra = 0;  // contador de lo que mezcla Wwise
    int64_t mezclados = 0, sonados_wwise = 0, frames_tot = 0;
    int64_t ultima_sonada = 0;
    int64_t last_cb = 0, ultima_devolucion = 0;
    double max_sin_devolver = 0;
    Resultado res;

    auto liberar = [&] {
        cola.liberar([&](void* o, void* b) {
            static_cast<Out*>(o)->released.push_back(int(reinterpret_cast<intptr_t>(b)) - 1);
        });
    };
    auto encolar = [&](int i, bool silencio, int64_t t) {
        if (silencio) {
            std::fill(mem[i].begin(), mem[i].end(), 0);
        } else {
            for (int f = 0; f < kFramesBuf; f++) {
                // izquierda: contador % 32767 + 1 (nunca 0: "hay audio"); derecha: contador / 32767
                mem[i][f * 2] = int16_t(siguiente_muestra % 32767 + 1);
                mem[i][f * 2 + 1] = int16_t(siguiente_muestra / 32767);
                siguiente_muestra++;
            }
            mezclados += kFramesBuf;
        }
        bool cebado = !out->started && silencio;
        cola.encolar(out, reinterpret_cast<void*>(intptr_t(i + 1)), mem[i].data(), kFramesBuf, cebado, continuidad);
        liberar();
        queued++;
        (void)t;
    };
    auto init = [&](int64_t t) {
        out = new Out;
        vivos.push_back(out);
        queued = 0;
        for (int i = 0; i < kBufs; i++) encolar(i, true, t);
        out->started = true;
        ultima_devolucion = t;
    };
    auto term = [&] {
        out->started = false;
        cola.parar(out, continuidad, [&](void* o, void* b) {
            static_cast<Out*>(o)->released.push_back(int(reinterpret_cast<intptr_t>(b)) - 1);
        });
        out = nullptr;
        res.reaperturas++;
    };
    auto sondeo = [&](int64_t t) {
        if (size_t n = continuidad ? reloj.frames(t, last_cb, kHz) : 0) {  // audio_v1: sin reloj de reserva
            cola.descartar(n);
            liberar();
        }
        while (!out->released.empty()) {
            int i = out->released.front();
            out->released.pop_front();
            queued--;
            max_sin_devolver = std::max(max_sin_devolver, (t - ultima_devolucion) / 1e6);
            ultima_devolucion = t;
            encolar(i, false, t);
        }
    };

    init(0);
    size_t ri = 0;
    std::vector<int16_t> dest(kBurst * kCh);
    for (int64_t t = 0; t < e.duracion_ns; t += 100'000) {  // pasos de 0.1 ms
        bool cb_ok = t >= e.aaudio_desde_ns;
        for (auto [a, b] : e.sin_callback)
            if (t >= a && t < b) cb_ok = false;
        if (cb_ok && t % (2 * kMs) == 0) {
            last_cb = t;
            res.max_cola_ms = std::max(res.max_cola_ms, cola.en_cola() * 1000.0 / kHz);
            size_t n = cola.leer(dest.data(), kBurst);
            liberar();
            frames_tot += kBurst;
            for (size_t f = 0; f < n; f++) {
                if (!dest[f * 2]) continue;
                const int64_t c = int64_t(dest[f * 2 + 1]) * 32767 + dest[f * 2] - 1;
                if (sonados_wwise && c != ultima_sonada + 1) res.saltos++;
                ultima_sonada = c;
                sonados_wwise++;
            }
        } else if (!cb_ok && t % (2 * kMs) == 0) {
            frames_tot += kBurst;  // el tiempo pasa igual (silencio en el altavoz)
        }
        while (ri < e.reaperturas.size() && e.reaperturas[ri] <= t) {
            sondeo(t);
            term();
            init(t);
            ri++;
        }
        if (t % (5 * kMs) == 0) sondeo(t);
    }
    res.audible = 100.0 * sonados_wwise / std::max<int64_t>(1, frames_tot);
    res.perdidos_ms = (mezclados - sonados_wwise - int64_t(cola.en_cola())) * 1000.0 / kHz;
    res.ritmo = mezclados / (e.duracion_ns / 1e9) / kHz;
    res.max_sin_devolver_ms = max_sin_devolver;
    for (Out* o : vivos) delete o;
    return res;
}

int main() {
    std::vector<Escenario> es;
    // 1) El arranque del log de hoy (log.txt, 164948595.4 = Start 0): reaperturas a +9.1 ms, 4 seguidas a +9.4..9.6,
    //    +28.4, +28.5, +60.8, +60.9, +76.9, +93.0, +109.0; luego estable.
    {
        Escenario e{"arranque (log de hoy)", 1000 * kMs, {}, {}, 0};
        for (double ms : {9.1, 9.4, 9.5, 9.5, 9.6, 28.4, 28.5, 60.8, 60.9, 76.9, 93.0, 109.0})
            e.reaperturas.push_back(int64_t(ms * kMs));
        es.push_back(e);
    }
    for (int cada : {16, 50, 100, 250, 500, 1000, 3000}) {
        static char nombres[10][64];
        static int k = 0;
        snprintf(nombres[k], 64, "reapertura cada %d ms", cada);
        Escenario e{nombres[k++], 20'000 * kMs, {}, {}, 0};
        for (int64_t t = cada * kMs; t < e.duracion_ns; t += cada * kMs) e.reaperturas.push_back(t);
        es.push_back(e);
    }
    {
        Escenario e{"reaperturas al azar (media 300 ms)", 20'000 * kMs, {}, {}, 0};
        std::mt19937 rng(7);
        std::exponential_distribution<double> d(1.0 / 300);
        for (double t = d(rng); t < 20'000; t += d(rng)) e.reaperturas.push_back(int64_t(t * kMs));
        es.push_back(e);
    }
    es.push_back({"sin reaperturas", 20'000 * kMs, {}, {}, 0});
    es.push_back({"AAudio se para 150 ms (a los 2 s)", 5'000 * kMs, {}, {{2000 * kMs, 2150 * kMs}}, 0});
    es.push_back({"AAudio empieza a llamar a los 65 ms", 2'000 * kMs, {}, {}, 65 * kMs});

    printf("%-36s | %-52s | %-52s\n", "escenario", "cola de antes (audio_v1)", "cola continua");
    printf("%-36s | %6s %8s %6s %6s %8s %7s | %6s %8s %6s %6s %8s %7s\n", "", "audio%", "perdidos", "saltos", "ritmo",
           "sin dev.", "cola", "audio%", "perdidos", "saltos", "ritmo", "sin dev.", "cola");
    for (auto& e : es) {
        Resultado a = simular(e, false), b = simular(e, true);
        printf("%-36s | %5.1f%% %6.0fms %6ld %6.3f %6.1fms %5.0fms | %5.1f%% %6.0fms %6ld %6.3f %6.1fms %5.0fms\n",
               e.nombre, a.audible, a.perdidos_ms, a.saltos, a.ritmo, a.max_sin_devolver_ms, a.max_cola_ms, b.audible,
               b.perdidos_ms, b.saltos, b.ritmo, b.max_sin_devolver_ms, b.max_cola_ms);
    }
}
