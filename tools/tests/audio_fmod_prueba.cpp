// Prueba fuera del dispositivo de la mitad AudioOut de nn::audio (runtime/src/nnsdk/audio.cpp) con lo que hace el FMOD
// de Unity en Disco Elysium: se compila el .cpp tal cual, con un AAudio de mentira (falsos/aaudio/AAudio.h) y los
// SystemEvent de falsos/eventos.h, y se le llama con la secuencia de la salida "FMOD NX Output" sacada de main.elf
// (main+0x3776e0 abrir, main+0x1184d50 preparar los buffers, main+0x1184fdc el hilo, main+0x1184f04 cerrar), con los
// tamaños de sus structs.
//
//   c++ -std=c++20 -O1 -g -fsanitize=address,undefined -I tools/tests/falsos -I runtime/src -I tools/tests \
//       tools/tests/audio_fmod_prueba.cpp -o /tmp/audio_fmod_prueba && /tmp/audio_fmod_prueba
//   (y otra vez con -fsanitize=thread)
//
// Lo que comprueba: que InitializeAudioOutParameter no pisa lo que hay detrás de sus 8 bytes (el fallo que tiraba
// FMOD); que hay una salida y FMOD recibe 48000 Hz / PCM16 / 2 canales; que su hilo, que solo despierta con el evento
// de la salida, mezcla al ritmo de AAudio sin perder ni repetir un frame; que si AAudio deja de llamar sigue recibiendo
// sus buffers a tiempo real (reloj de reserva); y que cerrar y volver a abrir (suspender y volver) no se cuelga.

#include "falsos/eventos.h"

#include "../../runtime/src/nnsdk/audio.cpp"

#include <cinttypes>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

namespace {

int g_fallos = 0;
#define COMPROBAR(cond)                                                           \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fallos++;                                                           \
        }                                                                         \
    } while (0)

using namespace nn::audio;
using Reloj = std::chrono::steady_clock;

constexpr uint64_t kCentinela = 0x00000076'12345678;  // un puntero de la pila de FMOD, como el outputrate que se rompía
constexpr uint32_t kDsp = 1024, kBuffers = 4;         // dspbufferlength y dspnumbuffers de Unity

// --- El trozo del objeto de fmod_output_nx.cpp que toca nn::audio (this+0x1b8..0x484) ---
struct SalidaFmod {
    std::atomic<bool> abierta{false};  // this+0x1b8
    alignas(8) uint8_t audio_out[0x168];  // this+0x1c0: lo que ocupa un nn::audio::AudioOut del SDK (sdk+0x54664)
    AudioOutBuffer buffers[4];         // this+0x328, 40 bytes cada uno (main+0x1184e14)
    alignas(8) uint8_t evento_buffers[0x30];  // this+0x3d0
    alignas(8) uint8_t evento_fin[0x30];      // this+0x400
    std::thread hilo;                  // this+0x430
    void* crudo[4] = {};               // this+0x438
    void* alineado[4] = {};            // this+0x458
    uint32_t bytes_buffer = 0;         // this+0x478
    uint32_t n_buffers = 0;            // this+0x47c
    uint32_t dsp = 0;                  // this+0x480
    // lo que mezcla "FMOD": un contador de frames, para ver luego qué ha sonado
    std::atomic<uint64_t> frame{0}, mezclados{0};

    AudioOut* out() { return reinterpret_cast<AudioOut*>(audio_out); }
    nn::os::SystemEvent* ev() { return reinterpret_cast<nn::os::SystemEvent*>(evento_buffers); }
};
static_assert(sizeof(AudioOutBuffer) == 40);

// main+0x3776e0, la función de Unity que abre la salida por defecto.
bool abrir_unity(AudioOut* out, nn::os::SystemEvent* evento) {
    struct {
        AudioOutInfo salidas[2];  // sp..sp+0x200
        uint8_t parametro[8];     // x29-0x28
        uint64_t x22;             // x29-0x20: el x22 de quien llama, guardado al entrar
    } pila;
    static_assert(sizeof pila.salidas == 0x200);
    pila.x22 = kCentinela;
    const int n = ListAudioOuts(pila.salidas, 2);
    auto* parametro = reinterpret_cast<AudioOutParameter*>(pila.parametro);
    InitializeAudioOutParameter(parametro);
    COMPROBAR(pila.x22 == kCentinela);  // con 12 bytes de parámetro volvía como 0x76'00000000
    const uint64_t como_el_sdk = 0xCAFE000000000000;  // sdk+0x3ac60
    COMPROBAR(std::memcmp(pila.parametro, &como_el_sdk, 8) == 0);
    parametro->sample_rate = 0;    // stur wzr, [x29, #-0x28]
    parametro->channel_count = 2;  // sturh w22, [x29, #-0x24]
    if (n < 1) return false;
    COMPROBAR(std::strcmp(pila.salidas[0].name, "DeviceOut") == 0);
    const nn::Result r = evento ? OpenDefaultAudioOut(out, evento, *parametro) : OpenDefaultAudioOut(out, *parametro);
    COMPROBAR(r.value == 0);  // si no, el juego aborta (main+0x377784)
    COMPROBAR(pila.x22 == kCentinela);
    return GetAudioOutChannelCount(out) > 0;
}

// main+0x1184fdc, "FMOD audio out thread".
void hilo_fmod(SalidaFmod* s) {
    falsos::esperar(s->evento_buffers, -1);
    while (!falsos::esperar(s->evento_fin, 0)) {
        if (s->abierta) {
            while (AudioOutBuffer* b = GetReleasedAudioOutBuffer(s->out())) {
                auto* pcm = static_cast<int16_t*>(GetAudioOutBufferDataPointer(b));
                for (uint32_t f = 0; f < s->dsp; f++) {  // el mezclador de FMOD: aquí, un contador que nunca vale 0
                    const uint64_t n = s->frame++;
                    pcm[2 * f] = static_cast<int16_t>(n % 32767 + 1);
                    pcm[2 * f + 1] = static_cast<int16_t>(n / 32767);
                }
                s->mezclados++;
                COMPROBAR(AppendAudioOutBuffer(s->out(), b));
            }
        }
        falsos::esperar(s->evento_buffers, -1);
    }
}

// main+0x1184d50 (y antes main+0x1184b40, que guarda dspbufferlength y el número de buffers).
bool abrir(SalidaFmod& s, int* outputrate, int* outputformat) {
    s.dsp = kDsp, s.n_buffers = std::min(kBuffers, 4u);
    if (s.abierta) return true;
    const bool ok = abrir_unity(s.out(), s.ev());
    s.abierta = ok;
    if (!ok) return false;
    const int formato = GetAudioOutSampleFormat(s.out());
    const int canales = GetAudioOutChannelCount(s.out());
    const int hz = GetAudioOutSampleRate(s.out());
    if (outputrate) *outputrate = hz;
    const auto bytes_muestra = static_cast<uint32_t>(GetSampleByteSize(SampleFormat{formato}));
    if (outputformat) *outputformat = 2;  // FMOD_SOUND_FORMAT_PCM16
    s.bytes_buffer = bytes_muestra * static_cast<uint32_t>(canales) * s.dsp;
    const uint32_t redondeado = (s.bytes_buffer + 0xfff) & ~0xfffu;
    for (uint32_t i = 0; i < s.n_buffers; i++) {
        s.crudo[i] = calloc(1, redondeado + 0x1000);
        s.alineado[i] = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(s.crudo[i]) + 0xfff) & ~uintptr_t{0xfff});
        SetAudioOutBufferInfo(&s.buffers[i], s.alineado[i], redondeado, s.bytes_buffer);
        COMPROBAR(AppendAudioOutBuffer(s.out(), &s.buffers[i]));
    }
    COMPROBAR(StartAudioOut(s.out()).value == 0);  // si no, el juego aborta (main+0x1184e88)
    nnsdk::create_system_event(reinterpret_cast<nn::os::SystemEventType*>(s.evento_fin), false, false);
    s.hilo = std::thread(hilo_fmod, &s);
    return true;
}

// main+0x1184f04. false si el hilo no sale (el cierre se quedaría colgado en DestroyThread).
bool cerrar(SalidaFmod& s) {
    if (!s.abierta) return true;
    s.abierta = false;
    nnsdk::signal_system_event(reinterpret_cast<nn::os::SystemEventType*>(s.evento_fin));
    if (GetAudioOutState(s.out()) == 0) StopAudioOut(s.out());
    auto fin = std::async(std::launch::async, [&] { s.hilo.join(); });
    const bool salio = fin.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    if (!salio) {
        // para no dejar la prueba colgada: se le despierta a mano
        nnsdk::signal_system_event(reinterpret_cast<nn::os::SystemEventType*>(s.evento_buffers));
        fin.wait();
    }
    CloseAudioOut(s.out());
    falsos::destruir(s.evento_buffers);
    falsos::destruir(s.evento_fin);
    for (uint32_t i = 0; i < s.n_buffers; i++) {
        free(s.crudo[i]);
        s.crudo[i] = s.alineado[i] = nullptr;
    }
    return salio;
}

// --- El dispositivo: un hilo que hace de AAudio, 96 frames cada 2 ms mientras `llamar` ---
struct Dispositivo {
    std::atomic<bool> llamar{true}, fin{false};
    std::mutex m;
    std::vector<int16_t> sonado;  // todo lo que ha salido
    std::atomic<int64_t> hueco_max_ns{0};
    std::thread hilo;

    void arrancar() {
        hilo = std::thread([this] {
            int16_t rafaga[96 * 2];
            auto antes = Reloj::now();
            while (!fin) {
                if (llamar && falsos::aaudio_rafaga(rafaga, 96)) {
                    const auto ahora = Reloj::now();
                    const int64_t hueco = std::chrono::duration_cast<std::chrono::nanoseconds>(ahora - antes).count();
                    if (hueco > hueco_max_ns) hueco_max_ns = hueco;
                    antes = ahora;
                    std::lock_guard lock(m);
                    sonado.insert(sonado.end(), rafaga, rafaga + 96 * 2);
                } else {
                    antes = Reloj::now();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    void parar() {
        fin = true;
        hilo.join();
    }
    size_t frames() {
        std::lock_guard lock(m);
        return sonado.size() / 2;
    }
};

void dormir(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// De lo que ha sonado, los frames de FMOD (los del contador): cuántos, y cuántas veces el contador no sigue al anterior.
struct Continuidad {
    uint64_t frames = 0, saltos = 0, primero = 0, ultimo = 0;
};
Continuidad continuidad_de(const std::vector<int16_t>& sonado) {
    Continuidad c;
    bool hay = false;
    for (size_t i = 0; i + 1 < sonado.size(); i += 2) {
        if (sonado[i] == 0) continue;  // silencio: el cebado o un hueco
        const uint64_t n = static_cast<uint64_t>(sonado[i + 1]) * 32767 + static_cast<uint64_t>(sonado[i] - 1);
        if (hay && n != c.ultimo + 1) c.saltos++;
        if (!hay) c.primero = n;
        hay = true;
        c.ultimo = n;
        c.frames++;
    }
    return c;
}

void prueba_fmod() {
    Dispositivo disp;
    auto s = std::make_unique<SalidaFmod>();
    int outputrate = 0, outputformat = 0;

    // 1. Init de FMOD con AAudio todavía sin llamar: la salida se abre igual y FMOD ve lo que espera.
    disp.llamar = false;
    disp.arrancar();
    COMPROBAR(abrir(*s, &outputrate, &outputformat));
    COMPROBAR(outputrate == 48000 && outputformat == 2);
    COMPROBAR(s->bytes_buffer == 4096);  // 2 bytes * 2 canales * 1024 frames
    COMPROBAR(GetAudioOutState(s->out()) == 0);
    // Sin AAudio, el reloj de reserva devuelve los buffers a tiempo real y FMOD mezcla (antes se quedaba esperando).
    dormir(400);
    const uint64_t sin_aaudio = s->mezclados;
    COMPROBAR(sin_aaudio >= 8 && sin_aaudio <= 24);  // 400 ms son 18,75 buffers; la reserva entra a los 40 ms
    std::printf("  sin AAudio: %" PRIu64 " buffers mezclados en 400 ms (a tiempo real serían 18,75)\n", sin_aaudio);

    // 2. AAudio llama: lo que suena es lo que mezcla FMOD, seguido.
    {
        std::lock_guard lock(disp.m);
        disp.sonado.clear();
    }
    disp.hueco_max_ns = 0;
    disp.llamar = true;
    dormir(700);
    disp.llamar = false;
    dormir(10);
    std::vector<int16_t> sonado;
    {
        std::lock_guard lock(disp.m);
        sonado = disp.sonado;
    }
    const Continuidad c = continuidad_de(sonado);
    const double hueco_ms = disp.hueco_max_ns / 1e6;
    std::printf("  con AAudio: %zu frames tocados, %" PRIu64 " de FMOD (del %" PRIu64 " al %" PRIu64 "), %" PRIu64
                " saltos; hueco máximo entre ráfagas %.1f ms\n",
                sonado.size() / 2, c.frames, c.primero, c.ultimo, c.saltos, hueco_ms);
    COMPROBAR(c.frames > 4096);
    if (hueco_ms < 35) {
        // Ni un frame perdido ni repetido, y casi todo lo tocado es de FMOD (el resto: lo que tardó AAudio en "arrancar").
        COMPROBAR(c.saltos == 0);
        COMPROBAR(c.frames + 2 * 1024 >= sonado.size() / 2);
    } else {
        std::printf("  (la máquina paró el hilo del dispositivo %.0f ms: no se puede exigir continuidad)\n", hueco_ms);
    }
    // FMOD va al paso del dispositivo: lo mezclado desde que llama AAudio es lo tocado, con sus 4 buffers por delante.
    const uint64_t mezclado_ahora = s->frame;
    COMPROBAR(mezclado_ahora >= c.ultimo && mezclado_ahora <= c.ultimo + 1 + 5 * 1024);

    // 3. AAudio se queda parado (cambio de ruta, su hilo sin CPU): FMOD sigue recibiendo buffers a tiempo real.
    const uint64_t antes = s->mezclados;
    dormir(400);
    const uint64_t parado = s->mezclados - antes;
    COMPROBAR(parado >= 8 && parado <= 24);
    std::printf("  AAudio parado: %" PRIu64 " buffers mezclados en 400 ms\n", parado);
    disp.llamar = true;
    dormir(100);

    // 4. Suspender y volver, tres veces: cerrar no se cuelga y la salida vuelve a sonar.
    for (int vuelta = 0; vuelta < 3; vuelta++) {
        COMPROBAR(cerrar(*s));
        COMPROBAR(!falsos::existe(s->evento_buffers) && !falsos::existe(s->evento_fin));
        const uint64_t antes_de_abrir = s->mezclados;
        COMPROBAR(abrir(*s, nullptr, nullptr));  // al volver, sin outputrate ni outputformat (main+0x1184c20)
        dormir(150);
        COMPROBAR(s->mezclados > antes_de_abrir + 2);
    }
    COMPROBAR(falsos::g_creados_al_senalar == 0);
    COMPROBAR(falsos::g_abiertos == 1);  // un solo stream de AAudio para todo: el de siempre

    // 5. El cierre en el peor momento: todos los buffers ya devueltos y el aviso ya consumido por el hilo, que está
    //    entre mirar `abierta` y volver a esperar. StopAudioOut tiene que señalar igual.
    COMPROBAR(cerrar(*s));
    {
        auto t = std::make_unique<SalidaFmod>();
        t->dsp = kDsp, t->n_buffers = 4;
        COMPROBAR(abrir_unity(t->out(), t->ev()));
        std::vector<int16_t> pcm(4 * kDsp * 2, 0);
        for (uint32_t i = 0; i < 4; i++) {
            SetAudioOutBufferInfo(&t->buffers[i], pcm.data() + i * kDsp * 2, 4096, 4096);
            COMPROBAR(AppendAudioOutBuffer(t->out(), &t->buffers[i]));
        }
        COMPROBAR(StartAudioOut(t->out()).value == 0);
        dormir(200);  // el dispositivo se lleva los cuatro
        int devueltos = 0;
        while (GetReleasedAudioOutBuffer(t->out())) devueltos++;
        COMPROBAR(devueltos == 4);
        falsos::esperar(t->evento_buffers, 0);  // el hilo ya había consumido el aviso
        COMPROBAR(!falsos::esperar(t->evento_buffers, 0));
        StopAudioOut(t->out());
        COMPROBAR(falsos::esperar(t->evento_buffers, 0));  // y aun así despierta para ver su evento de fin
        COMPROBAR(GetAudioOutState(t->out()) == 1);
        CloseAudioOut(t->out());
        falsos::destruir(t->evento_buffers);
    }
    disp.parar();
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    prueba_fmod();
    if (g_fallos) {
        std::printf("audio_fmod_prueba: %d FALLOS\n", g_fallos);
        return 1;
    }
    std::printf("audio_fmod_prueba: todo bien\n");
    return 0;
}
