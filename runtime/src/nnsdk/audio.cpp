// nn::audio. Wwise (el motor de sonido de NieR) pregunta primero por el dispositivo de salida: con
// 0 canales se queda en modo suspendido esperando a que se conecte uno, y el juego no avanza
// (el hilo principal espera a que Wwise arranque). La Flip tiene salida estéreo.
// Aquí están el dispositivo, AudioOut y el stream de AAudio del proceso; el AudioRenderer (voces, mixes, sinks,
// memory pools), en audio_renderer.cpp, que suena por este mismo stream (audio_salida.h).

#include <aaudio/AAudio.h>
#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "nnsdk/audio_cola.h"
#include "nnsdk/audio_salida.h"
#include "nnsdk/os.h"
#include "nnsdk/types.h"
#include "nnsdk/nvn/flags.h"

namespace nn::os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}

namespace nn::audio {

namespace {
void preparar_salida();  // abre AAudio en segundo plano (más abajo)
}

struct AudioDeviceName {
    char name[256];
};

int GetActiveAudioDeviceChannelCountForOutput() {
    preparar_salida();  // Wwise lo pregunta al arrancar, mucho antes de abrir su AudioOut
    return 2;
}

Result GetActiveAudioDeviceName(AudioDeviceName* out) {
    strncpy(out->name, "AudioStereoJackOutput", sizeof out->name);
    return ResultSuccess;
}

// Al suscribirse por primera vez se notifica el estado actual (hay salida estéreo): Wwise, que arranca
// suspendido esperando dispositivo, sale de ese modo con este aviso. Solo la primera: el Wwise de MH Rise toma cada
// aviso por un cambio de dispositivo, cierra y vuelve a abrir la salida y se vuelve a suscribir; avisando siempre,
// reabría AAudio cada ~250 ms, el reloj del audio se paraba en cada reapertura y con él la carga de bancos de
// sonido de la que espera todo el motor (la advertencia de salud iba a 4 fps).
// El evento es AutoClear, como el del SDK: Wwise (main+0x2603df0, en cada buffer que mezcla) y el RE Engine
// (main+0x7a41510, en su actualización) lo miran con TryWaitSystemEvent y nunca lo limpian. Sin crearlo se quedaba
// ManualClear: una vez señalado, TryWait daba true para siempre y Wwise rehacía el cambio de dispositivo
// (SetPanningRule y el recálculo de los listeners) en cada buffer. Y devuelve Result: antes devolvía lo que
// quedase en w0 (0 la primera vez, 1 las demás) y Wwise lo mira. Interruptor audio_aviso_v1: como antes.
Result AcquireAudioDeviceNotificationForOutput(os::SystemEvent* event) {
    preparar_salida();
    static std::atomic<bool> notified{false};
    auto* e = reinterpret_cast<os::SystemEventType*>(event);
    const bool first = !notified.exchange(true);
    // Dos llamadas por arranque (Wwise, main+0x2603d34; el RE Engine, main+0x7a415b4/0x7a416e8): quién va primero decide
    // quién recibe el aviso inicial. Con el de antes (ManualClear para siempre), el primero lo veía en cada consulta: Wwise
    // rehace SetPanningRule por buffer y el RE Engine manda su orden 20 (main+0x7a4b300) en cada frame.
    char name[16] = "?";
    pthread_getname_np(pthread_self(), name, sizeof name);
    printf("[audio] AcquireAudioDeviceNotificationForOutput %p en '%s' (desde %p): %s\n", static_cast<void*>(event), name,
           __builtin_return_address(0), first ? "primera, con aviso" : "sin aviso");
    if (NVN_FLAG("audio_aviso_v1")) {
        if (!first) return Result{1};
        nnsdk::signal_system_event(e);
        return ResultSuccess;
    }
    nnsdk::create_system_event(e, /*auto_clear=*/true, /*signaled=*/first);
    return ResultSuccess;
}

// --- AudioOut ---
// El sink de Wwise para Switch (main+0x2605544 Init, +0x26057b8 Term, +0x26059ec IsDataNeeded) abre una salida,
// reparte 4 buffers de 4 KB (1024 frames, 21.3 ms), los rellena de silencio, los encola y arranca; luego solo
// vuelve a mezclar cuando GetReleasedAudioOutBuffer le devuelve alguno: el ritmo al que se liberan es el reloj
// del audio (y de lo que se sincroniza con él, como los vídeos). Aquí se copian a una cola única para todo el
// proceso (audio_cola.h) que toca AAudio; un buffer se libera cuando la lectura pasa de su final.
//
// En Disco Elysium quien lo usa es el FMOD de Unity, con su salida "FMOD NX Output" (fmod_output_nx.cpp); es todo el
// sonido del juego. Lo que hace (main = work/scope/main.elf, sdk = work/scope/sdk.elf, el SDK de verdad):
//  - Init (main+0x1184b40) guarda dspbufferlength (this+0x480) y min(dspnumbuffers, 4) (this+0x47c) y abre
//    (main+0x1184d50): una función de Unity (main+0x3776e0) hace ListAudioOuts(2 entradas), InitializeAudioOutParameter,
//    pone sampleRate = 0 y channelCount = 2 y llama a OpenDefaultAudioOut con un SystemEvent (this+0x3d0); si
//    ListAudioOuts da menos de 1 no abre nada y el init de FMOD falla (devuelve 0x3c, main+0x1184ba0): sin salida.
//  - Con la salida abierta devuelve *outputrate = GetAudioOutSampleRate y *outputformat = PCM16, y calcula el buffer:
//    GetSampleByteSize(formato) * GetAudioOutChannelCount * dspbufferlength bytes (this+0x478). Reserva los buffers a
//    cero y alineados a 4096, SetAudioOutBufferInfo + AppendAudioOutBuffer de cada uno, StartAudioOut (aborta si falla)
//    y crea el hilo "FMOD audio out thread". Con los ajustes de audio del juego (el AudioManager de Unity, en
//    romfs/Data/globalgamemanagers+0x5928: estéreo, frecuencia del sistema, DSPBufferSize 1024) son buffers de 1024
//    frames, 4096 bytes y 21,3 ms.
//  - Ese hilo (main+0x1184fdc) no sondea: WaitSystemEvent del evento de la salida y, por cada buffer que le da
//    GetReleasedAudioOutBuffer, mezcla dspbufferlength frames en GetAudioOutBufferDataPointer y lo vuelve a encolar.
//    Sale cuando ve señalado su evento de fin (this+0x400), que solo mira al despertar del otro.
//  - Cerrar (main+0x1184f04, también al suspender; al volver abre otra vez): señala el evento de fin, StopAudioOut si
//    GetAudioOutState es 0 (Started), espera al hilo, CloseAudioOut y destruye los dos eventos.

// 8 bytes: InitializeAudioOutParameter del SDK escribe un solo u64, 0xCAFE000000000000 (sdk+0x3ac60): frecuencia y
// canales a 0 ("los del dispositivo") y una marca en los 16 bits altos. Y 8 son los que le reserva Unity en la pila
// (main+0x377714: x29-0x28, con el x22 de quien llama guardado justo detrás, en x29-0x20). Con los 12 de antes (una
// marca de 32 bits en +8) el memset pisaba la mitad baja de ese x22, que en la salida de FMOD es el puntero
// `outputrate` (main+0x1184d78): volvía como 0x76'00000000 y FMOD se caía al escribir en él (main+0x1184dc0).
struct AudioOutParameter {
    int32_t sample_rate;     // 0: la del dispositivo
    uint16_t channel_count;  // 0: los del dispositivo
    uint16_t marca;
};
static_assert(sizeof(AudioOutParameter) == 8);
struct AudioOutBuffer {  // 40 bytes en el juego (stride de su array); el formato interno es nuestro
    void* next;
    void* buffer;
    size_t buffer_size;
    size_t data_size;
    size_t reserved;
};
static_assert(sizeof(AudioOutBuffer) == 40);
struct AudioOut {
    uint32_t handle;  // el sink lee esta palabra como "abierto" y la pone a 0 tras cerrar
};

namespace {

constexpr int kSampleRate = 48000, kStreamChannels = 2;
constexpr size_t kColaFrames = kSampleRate / 4;  // 250 ms: el sink tiene como mucho 4 buffers de 21 ms en cola
constexpr int64_t kInformeNs = 5'000'000'000;

int64_t ahora_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

// Interruptor audio_v1: cola de antes (Stop tira el audio ya mezclado sin tocar, el cebado suena como silencio y
// sin reloj de reserva).
bool continuidad() { return !NVN_FLAG("audio_v1"); }

bool silencio(const void* datos, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(datos);
    for (size_t i = 0; i < bytes; i++)
        if (p[i]) return false;
    return true;
}

class HostAudioOut {
public:
    const void* key = nullptr;  // el AudioOut del juego: su cola de mezcla (la conserva si lo cierra y lo reabre)
    int channels = 2;
    bool started = false;
    std::deque<AudioOutBuffer*> released;  // con el lock de SharedStream
    os::SystemEventType* event = nullptr;  // el de OpenDefaultAudioOut con evento (FMOD de Unity): se señala al liberar
};

void release_buffer(void* o, void* b) {
    auto* out = static_cast<HostAudioOut*>(o);
    out->released.push_back(static_cast<AudioOutBuffer*>(b));
    if (out->event) nnsdk::signal_system_event(out->event);
}

// Una sola salida de AAudio para todo el proceso, siempre en marcha: toca la cola (lo que mezcló Wwise o
// silencio). Se abre en segundo plano al primer uso de nn::audio (Wwise pregunta por el dispositivo al arrancar):
// abrir y arrancar AAudio tarda ~65 ms y antes lo hacía el primer OpenDefaultAudioOut, dentro del hilo de audio
// de Wwise. Mientras no hay AAudio, o si deja de llamar (desconexión, su hilo sin CPU), el reloj de reserva
// libera los buffers a tiempo real: Wwise nunca ve una salida parada.
class SharedStream {
public:
    // Nunca se destruye: su hilo y el callback de AAudio siguen hasta que muere el proceso, también mientras corren los
    // destructores estáticos de la salida.
    static SharedStream& get() {
        static SharedStream* s = new SharedStream;
        return *s;
    }

    void attach(HostAudioOut* out) {
        std::lock_guard lock(m_);
        cola(out);
        starts_++;
    }
    // Stop o Close: los buffers pendientes de `out` se le devuelven ya; su audio sigue en la cola (continuidad).
    void detach(HostAudioOut* out, bool closing) {
        const bool cont = continuidad();  // fuera del lock: mira un fichero de la SD (como mucho una vez por segundo)
        std::lock_guard lock(m_);
        cola(out).parar(out, cont, release_buffer);
        (closing ? closes_ : stops_)++;
    }
    void opened() {
        std::lock_guard lock(m_);
        opens_++;
    }
    void append(HostAudioOut* out, AudioOutBuffer* b) {
        // Una salida de 6 canales (5.1: CRI Atom en Final Fantasy Resonance la abre así siempre) se baja a estéreo como
        // hace la consola en portátil: frontales tal cual, y central y traseros a -3 dB en cada lado; el LFE no va a
        // los altavoces. En el propio buffer del juego (el estéreo ocupa su primer tercio y el juego lo rellena entero
        // cada vez): sin esto sus muestras se leían como estéreo, el triple de largo y con los canales cruzados.
        const int channels = out->channels > 0 ? out->channels : kStreamChannels;
        const size_t frames = b->data_size / (static_cast<size_t>(channels) * sizeof(int16_t));
        if (channels == 6 && b->buffer) {
            auto* pcm = static_cast<int16_t*>(b->buffer);
            for (size_t i = 0; i < frames; i++) {
                const int16_t* in = pcm + 6 * i;
                const float centre = 0.7071f * in[2];
                const float left = in[0] + centre + 0.7071f * in[4], right = in[1] + centre + 0.7071f * in[5];
                pcm[2 * i] = static_cast<int16_t>(std::clamp(left, -32768.0f, 32767.0f));
                pcm[2 * i + 1] = static_cast<int16_t>(std::clamp(right, -32768.0f, 32767.0f));
            }
        }
        // El silencio se mira en el cebado (antes de arrancar) y, para decirlo una vez en el log, hasta el primer buffer
        // con sonido: es la prueba de que el mezclador del juego produce algo, sin tener que activar `registro`.
        const bool visto = sonido_visto_.load(std::memory_order_relaxed);
        const bool mudo = (!out->started || !visto) && silencio(b->buffer, frames * kStreamChannels * sizeof(int16_t));
        const bool cebado = !out->started && mudo, cont = continuidad();
        uint64_t mudos = 0;
        bool primero = false;
        {
            std::lock_guard lock(m_);
            cola(out).encolar(out, b, static_cast<const int16_t*>(b->buffer), frames, cebado, cont);
            liberar_locked();
            appended_++;
            if (cebado) cebados_++;
            if (!visto && mudo) mudos_++;
            if (!visto && !mudo && !sonido_visto_.exchange(true)) primero = true, mudos = mudos_;
        }
        if (primero)
            printf("[audio] primer buffer con sonido del juego (%zu frames; antes, %llu en silencio)\n", frames,
                   ull(mudos));
    }
    AudioOutBuffer* pop_released(HostAudioOut* out) {
        AudioOutBuffer* b = nullptr;
        char informe[640];
        informe[0] = 0;
        const bool cont = continuidad();
        {
            std::lock_guard lock(m_);
            const int64_t t = ahora_ns();
            if (cont) avanzar_reserva_locked(t);  // audio_v1: sin reloj de reserva
            if (!out->released.empty()) {
                b = out->released.front();
                out->released.pop_front();
                released_++;
            }
            if (!next_report_ns_) next_report_ns_ = t + kInformeNs;
            if (t >= next_report_ns_) informe_locked(t, informe, sizeof informe);
        }
        if (informe[0] && NVN_FLAG("registro")) fputs(informe, stdout);
        if (dead_.exchange(false)) std::thread([this] { reopen(); }).detach();
        return b;
    }
    // Otra fuente además de las colas (el AudioRenderer): audio_salida.h.
    void fuente(const nnsdk::audio::FuenteExtra* f) { fuente_.store(f, std::memory_order_release); }
    // Una salida más (o menos) cuyo juego espera un evento en vez de sondear.
    void con_evento(int delta) { con_evento_.fetch_add(delta, std::memory_order_relaxed); }

private:
    SharedStream() {
        std::thread([this] {
            open();
            vigilar();
        }).detach();
    }
    // El reloj de reserva solo avanzaba cuando el juego sondeaba (GetReleasedAudioOutBuffer): vale para el sink de
    // Wwise, que pregunta cada pocos ms. El FMOD de Unity no sondea: su hilo (main+0x1184fdc) duerme en
    // WaitSystemEvent hasta que se le devuelve un buffer, y el reproductor de vídeo espera igual el evento de su
    // AudioRenderer. Sin AAudio llamando (aún no ha arrancado, no se pudo abrir, se desconectó) nadie les devolvía
    // nada y su mezclador se quedaba parado para siempre. Este hilo (el mismo que abrió el stream) mira cada 20 ms si
    // AAudio ha dejado de llamar y, si hay quien dependa de un evento, mueve el reloj él. También reabre el stream
    // desconectado, que antes solo se reabría desde el sondeo del juego.
    void vigilar() {
#if defined(__ANDROID__) || defined(__linux__)
        pthread_setname_np(pthread_self(), "AudioSalida");
#endif
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (dead_.exchange(false)) {
                reopen();
                continue;
            }
            if (con_evento_.load(std::memory_order_relaxed) <= 0 && !fuente_.load(std::memory_order_relaxed)) continue;
            const int64_t t = ahora_ns(), last = last_callback_ns_.load(std::memory_order_relaxed);
            if (last && t - last < kUmbralMinimoNs) continue;  // AAudio lleva el ritmo: ni se toma el cerrojo
            if (!continuidad()) continue;                      // audio_v1: sin reloj de reserva
            std::lock_guard lock(m_);
            avanzar_reserva_locked(ahora_ns());
        }
    }
    // Con m_: si AAudio no lleva el ritmo, pasan los frames que tocaban desde la última vez (colas y fuente extra).
    void avanzar_reserva_locked(int64_t t) {
        const size_t n = reloj_.frames(t, last_callback_ns_.load(std::memory_order_relaxed), kSampleRate);
        if (!n) return;
        size_t hecho = 0;
        for (auto& [k, c] : colas_) hecho = std::max(hecho, c->descartar(n));
        reserva_frames_ += hecho;
        liberar_locked();
        if (const auto* f = fuente_.load(std::memory_order_acquire)) f->descartar(n);
    }
    void open() {
        AAudioStreamBuilder* b = nullptr;
        if (AAudio_createStreamBuilder(&b) != AAUDIO_OK) return;
        AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
        AAudioStreamBuilder_setChannelCount(b, kStreamChannels);
        AAudioStreamBuilder_setSampleRate(b, kSampleRate);
        AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
        AAudioStreamBuilder_setDataCallback(b, &SharedStream::callback, this);
        AAudioStreamBuilder_setErrorCallback(b, &SharedStream::on_error, this);
        AAudioStream* s = nullptr;
        const int64_t t0 = ahora_ns();
        aaudio_result_t r = AAudioStreamBuilder_openStream(b, &s);
        AAudioStreamBuilder_delete(b);
        if (r != AAUDIO_OK) {
            printf("[audio] AAudio openStream: %s (sigue el reloj de reserva, sin sonido)\n",
                   AAudio_convertResultToText(r));
            return;
        }
        r = AAudioStream_requestStart(s);
        {
            // Con la ruta de baja latencia (MMAP, ráfagas de 2 ms) 40 ms sin callback es un atasco; con una ruta de
            // ráfagas largas (Bluetooth) el umbral crece para no tirar audio que AAudio sí iba a pedir.
            const int64_t periodo = std::max(AAudioStream_getFramesPerBurst(s), AAudioStream_getBufferSizeInFrames(s));
            std::lock_guard lock(m_);
            reloj_.umbral_ns = std::max<int64_t>(40'000'000, 3 * periodo * 1'000'000'000 / kSampleRate);
        }
        printf("[audio] salida AAudio %d Hz, %d canales, ráfaga %d frames, búfer %d frames (una para todo el "
               "proceso, abierta en segundo plano en %.1f ms; requestStart %s)\n",
               AAudioStream_getSampleRate(s), AAudioStream_getChannelCount(s), AAudioStream_getFramesPerBurst(s),
               AAudioStream_getBufferSizeInFrames(s), (ahora_ns() - t0) / 1e6, AAudio_convertResultToText(r));
        stream_.store(s);
    }
    // Desconexión (cambio de ruta, el servicio de audio reiniciado): AAudio no deja cerrar desde su callback de
    // error; lo reabre un hilo aparte, y entretanto lleva el ritmo el reloj de reserva.
    void reopen() {
        AAudioStream* s = nullptr;
        {
            // informe_locked usa el stream con m_ tomado: que no se cierre a medias. El cierre va fuera del lock (espera a
            // que acabe el callback, que también toma m_).
            std::lock_guard lock(m_);
            s = stream_.exchange(nullptr);
        }
        if (s) {
            AAudioStream_requestStop(s);
            AAudioStream_close(s);
        }
        printf("[audio] AAudio desconectado: se reabre\n");
        open();
    }
    static void on_error(AAudioStream*, void* user, aaudio_result_t error) {
        auto* self = static_cast<SharedStream*>(user);
        self->last_error_ = error;
        self->dead_ = true;
    }
    static aaudio_data_callback_result_t callback(AAudioStream*, void* user, void* audio, int32_t frames);

    void liberar_locked() {
        for (auto& [k, c] : colas_)
            c->liberar(release_buffer);
    }
    // Una cola por AudioOut del juego, mezcladas en el callback: Xenoblade toca a la vez el de Wwise y el del reproductor
    // de vídeo (AudioOutRenderer). Con la cola única de MH Rise los dos se encolaban uno tras otro: cada uno sonaba a
    // medias y el vídeo, que sigue el reloj de su audio, no pasaba del primer frame.
    nnsdk::audio::ColaContinua& cola(HostAudioOut* out) {
        auto& c = colas_[out->key ? out->key : out];
        if (!c) c = std::make_unique<nnsdk::audio::ColaContinua>(kColaFrames, kStreamChannels);
        return *c;
    }
    size_t en_cola_locked() const {
        size_t n = 0;
        for (const auto& [k, c] : colas_) n = std::max(n, c->en_cola());
        return n;
    }
    uint64_t sumar_locked(uint64_t (nnsdk::audio::ColaContinua::*f)() const) const {
        uint64_t n = 0;
        for (const auto& [k, c] : colas_) n += ((*c).*f)();
        return n;
    }
    // DEPURACIÓN (registro): cada 5 s de reloj (no por salida: antes el informe solo salía si una salida duraba
    // 5 s, y durante el juego no salía nunca).
    void informe_locked(int64_t t, char* out, size_t size) {
        const double s = (t - next_report_ns_ + kInformeNs) / 1e9;
        AAudioStream* st = stream_.load();
        snprintf(out, size,
                 "[audio] %.1f s: %llu Open, %llu Start, %llu Stop, %llu Close; %llu buffers encolados (%llu de "
                 "cebado, %llu cubiertos con audio de la salida anterior), %llu devueltos; con datos %.0f%% de %.1f s "
                 "tocados, %llu callbacks sin datos con la salida en marcha, hueco máx. entre callbacks %.1f ms; en "
                 "cola %.0f ms, reloj de reserva %.0f ms, perdidos %.0f ms; nivel máx. %s; AAudio %s, xruns %d%s\n",
                 s, ull(opens_), ull(starts_), ull(stops_), ull(closes_), ull(appended_), ull(cebados_),
                 ull(sumar_locked(&nnsdk::audio::ColaContinua::cubiertos) - cubiertos_antes_), ull(released_),
                 frames_ ? 100.0 * frames_datos_ / frames_ : 0.0, frames_ / double(kSampleRate), ull(huecos_),
                 max_gap_ns_.exchange(0) / 1e6, en_cola_locked() * 1000.0 / kSampleRate,
                 reserva_frames_ * 1000.0 / kSampleRate,
                 (sumar_locked(&nnsdk::audio::ColaContinua::perdidos) - perdidos_antes_) * 1000.0 / kSampleRate,
                 nivel_texto(pico_).c_str(), st ? AAudio_convertStreamStateToText(AAudioStream_getState(st)) : "sin abrir",
                 st ? AAudioStream_getXRunCount(st) : 0, last_error_ ? ", hubo error" : "");
        next_report_ns_ = t + kInformeNs;
        opens_ = starts_ = stops_ = closes_ = appended_ = cebados_ = released_ = 0;
        frames_ = frames_datos_ = huecos_ = reserva_frames_ = 0;
        pico_ = 0;
        cubiertos_antes_ = sumar_locked(&nnsdk::audio::ColaContinua::cubiertos);
        perdidos_antes_ = sumar_locked(&nnsdk::audio::ColaContinua::perdidos);
    }
    static unsigned long long ull(uint64_t v) { return static_cast<unsigned long long>(v); }
    // Pico de lo que sale (la mezcla de todas las salidas) en dBFS: "silencio" si todo fue 0 (Wwise mandando ceros).
    static std::string nivel_texto(int32_t pico) {
        if (!pico) return "silencio";
        char b[24];
        snprintf(b, sizeof b, "%.0f dBFS", 20.0 * std::log10(pico / 32768.0));
        return b;
    }

    static constexpr int64_t kUmbralMinimoNs = 40'000'000;  // el umbral más corto del reloj de reserva (audio_cola.h)
    std::atomic<AAudioStream*> stream_{nullptr};
    std::atomic<const nnsdk::audio::FuenteExtra*> fuente_{nullptr};
    std::atomic<int> con_evento_{0};
    std::atomic<bool> sonido_visto_{false};
    uint64_t mudos_ = 0;  // con m_: buffers en silencio encolados antes del primero con sonido
    std::atomic<bool> dead_{false};
    std::atomic<aaudio_result_t> last_error_{0};
    std::atomic<int64_t> last_callback_ns_{0}, max_gap_ns_{0};
    std::mutex m_;
    std::unordered_map<const void*, std::unique_ptr<nnsdk::audio::ColaContinua>> colas_;
    std::vector<int16_t> mezcla_;  // lo que da una cola en el callback, antes de sumarlo
    nnsdk::audio::RelojDeReserva reloj_;
    // DEPURACIÓN: contadores del informe (con m_).
    int64_t next_report_ns_ = 0;
    uint64_t opens_ = 0, starts_ = 0, stops_ = 0, closes_ = 0, appended_ = 0, cebados_ = 0, released_ = 0;
    uint64_t frames_ = 0, frames_datos_ = 0, huecos_ = 0, reserva_frames_ = 0, cubiertos_antes_ = 0,
             perdidos_antes_ = 0;
    int32_t pico_ = 0;  // DEPURACIÓN: máximo |muestra| de la mezcla en el periodo del informe
};

aaudio_data_callback_result_t SharedStream::callback(AAudioStream*, void* user, void* audio, int32_t frames) {
    auto* self = static_cast<SharedStream*>(user);
    const int64_t t = ahora_ns();
    const int64_t before = self->last_callback_ns_.exchange(t, std::memory_order_relaxed);
    if (before && t - before > self->max_gap_ns_.load(std::memory_order_relaxed))
        self->max_gap_ns_.store(t - before, std::memory_order_relaxed);
    std::lock_guard lock(self->m_);
    auto* out = static_cast<int16_t*>(audio);
    const size_t muestras = size_t(frames) * kStreamChannels;
    std::memset(out, 0, muestras * sizeof(int16_t));
    if (self->mezcla_.size() < muestras) self->mezcla_.resize(muestras);
    size_t n = 0;
    for (auto& [k, c] : self->colas_) {
        const size_t leidos = c->leer(self->mezcla_.data(), size_t(frames));
        n = std::max(n, leidos);
        for (size_t i = 0; i < leidos * kStreamChannels; i++)
            out[i] = static_cast<int16_t>(std::clamp<int32_t>(int32_t{out[i]} + self->mezcla_[i], -32768, 32767));
    }
    if (const auto* f = self->fuente_.load(std::memory_order_acquire)) f->mezclar(out, size_t(frames));
    for (size_t i = 0; i < muestras; i++) self->pico_ = std::max<int32_t>(self->pico_, std::abs(int32_t{out[i]}));
    self->liberar_locked();
    self->frames_ += frames;
    self->frames_datos_ += n;
    if (n < size_t(frames) && n > 0) self->huecos_++;
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

void preparar_salida() { SharedStream::get(); }

std::mutex g_outs_mutex;
std::unordered_map<const AudioOut*, std::unique_ptr<HostAudioOut>> g_outs;
uint32_t g_next_handle = 1;

HostAudioOut* host(const AudioOut* out) {
    std::lock_guard lock(g_outs_mutex);
    auto it = g_outs.find(out);
    return it == g_outs.end() ? nullptr : it->second.get();
}

// DEPURACIÓN (registro): direcciones de retorno por la cadena de x29, dentro de la pila del hilo.
int x29_chain(const void** pcs, int max) {
#if !defined(__ANDROID__) && !defined(__linux__)
    (void)pcs, (void)max;
    return 0;  // la prueba fuera del dispositivo (tools/tests/audio_fmod_prueba.cpp): sin pthread_getattr_np
#else
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) return 0;
    void* base = nullptr;
    size_t size = 0;
    pthread_attr_getstack(&attr, &base, &size);
    pthread_attr_destroy(&attr);
    const auto lo = reinterpret_cast<uintptr_t>(base), hi = lo + size;
    auto fp = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
    int n = 0;
    while (n < max && fp >= lo && fp + 16 <= hi && !(fp & 7)) {
        const auto* frame = reinterpret_cast<const uintptr_t*>(fp);
        if (!frame[1]) break;
        pcs[n++] = reinterpret_cast<const void*>(frame[1]);
        if (frame[0] <= fp) break;
        fp = frame[0];
    }
    return n;
#endif
}

// DEPURACIÓN (registro): vida de las salidas (Open/Start/Stop/Close) con hilo y pila, para ver qué camino de
// Wwise las cierra y reabre (ReplaceOutput main+0x259c26c, Suspend/Wakeup main+0x259cd7c, reintento de un
// dispositivo caído main+0x259a4d4...; restar "main en 0x…" del principio del log). Append y Released solo los
// primeros, que son cientos por segundo.
void trace_out(const char* what, const AudioOut* out, const void* caller) {
    if (!NVN_FLAG("registro")) return;
    const bool vida = what[0] == 'O' || what[0] == 'S' || what[0] == 'C';
    static std::atomic<int> n_vida{0}, n_cola{0};
    if ((vida ? n_vida : n_cola)++ >= (vida ? 800 : 64)) return;
    const int64_t t = ahora_ns();
    if (!vida) {
        printf("[audio] %s %p a %.1f ms (desde %p)\n", what, static_cast<const void*>(out), t / 1e6, caller);
        return;
    }
    const void* pcs[8] = {};
    const int depth = x29_chain(pcs, 8);
    char name[16] = "?";
    pthread_getname_np(pthread_self(), name, sizeof name);
    char stack[8 * 20] = "";
    for (int i = 0, used = 0; i < depth && used < int(sizeof stack); i++)
        used += snprintf(stack + used, sizeof stack - used, " %p", pcs[i]);
    printf("[audio] %s %p a %.1f ms en '%s' (desde %p; pila%s)\n", what, static_cast<const void*>(out), t / 1e6, name,
           caller, stack);
}

}  // namespace

void InitializeAudioOutParameter(AudioOutParameter* p) {
    preparar_salida();
    *p = {0, 0, 0xCAFE};
}

// 256 bytes, el nombre de la salida: ListAudioOuts del SDK le pide al servicio `count << 8` bytes (sdk+0x5436c) y Unity
// reserva 0x200 para dos (main+0x3776f4). Unity solo mira cuántas hay (main+0x37771c); el nombre lo pone en la consola
// el servicio de audio y no viene en el volcado: "DeviceOut" es el que documenta switchbrew. Con 0, que es lo que se
// devolvía mientras InitializeAudioOutParameter rompía la pila de Unity (ver AudioOutParameter), FMOD no abre su
// salida y el juego no suena. Interruptor sin_audio: ninguna salida, como entonces (FMOD lo pregunta al arrancar y al
// volver de una suspensión).
struct AudioOutInfo {
    char name[256];
};
int ListAudioOuts(AudioOutInfo* out, int count) {
    preparar_salida();  // lo primero que pregunta FMOD: que AAudio vaya abriéndose
    if (!out || count < 1) return 0;
    static std::atomic<int> avisos{0};
    if (NVN_FLAG("sin_audio")) {
        if (nvn::first_times(avisos, 4)) printf("[audio] ListAudioOuts: ninguna salida (interruptor sin_audio)\n");
        return 0;
    }
    memset(&out[0], 0, sizeof out[0]);
    strcpy(out[0].name, "DeviceOut");
    if (nvn::first_times(avisos, 4)) printf("[audio] ListAudioOuts: 1 salida (\"DeviceOut\")\n");
    return 1;
}

// Nunca falla: si AAudio aún no está (o no se pudo abrir), la cola avanza con el reloj de reserva.
Result OpenDefaultAudioOut(AudioOut* out, const AudioOutParameter& p) {
    trace_out("Open", out, __builtin_return_address(0));
    if (p.sample_rate && p.sample_rate != kSampleRate) printf("[audio] AudioOut pide %d Hz\n", p.sample_rate);
    auto h = std::make_unique<HostAudioOut>();
    h->key = out;
    h->channels = p.channel_count ? p.channel_count : 2;
    if (h->channels != kStreamChannels)
        printf("[audio] AudioOut de %d canales: %s\n", h->channels,
               h->channels == 6 ? "se baja a estéreo (central y traseros a -3 dB)" : "SIN SOPORTE, se lee como estéreo");
    SharedStream::get().opened();
    std::lock_guard lock(g_outs_mutex);
    out->handle = g_next_handle++;
    g_outs[out] = std::move(h);
    return ResultSuccess;
}

// La de FMOD (Unity): con un SystemEvent que se señala cada vez que vuelve un buffer. AutoClear, como el que engancha
// el SDK (AttachReadableHandleToSystemEvent con EventClearMode 1, sdk+0x547b0).
Result OpenDefaultAudioOut(AudioOut* out, os::SystemEvent* event, const AudioOutParameter& p) {
    const Result r = OpenDefaultAudioOut(out, p);
    auto* e = reinterpret_cast<os::SystemEventType*>(event);
    nnsdk::create_system_event(e, true, false);
    int canales = 0;
    {
        std::lock_guard lock(g_outs_mutex);
        if (auto it = g_outs.find(out); it != g_outs.end()) {
            it->second->event = e;
            canales = it->second->channels;
        }
    }
    SharedStream::get().con_evento(+1);
    static std::atomic<int> avisos{0};
    if (nvn::first_times(avisos, 16))
        printf("[audio] AudioOut %p abierto con evento de buffers %p: %d Hz, %d canales, PCM16 (piden %d Hz y %d "
               "canales)\n",
               static_cast<void*>(out), static_cast<void*>(e), kSampleRate, canales, p.sample_rate, p.channel_count);
    return r;
}

int GetAudioOutChannelCount(const AudioOut* out) {
    HostAudioOut* h = host(out);
    return h ? h->channels : 2;
}

void SetAudioOutBufferInfo(AudioOutBuffer* b, void* buffer, size_t buffer_size, size_t data_size) {
    b->next = nullptr;
    b->buffer = buffer;
    b->buffer_size = buffer_size;
    b->data_size = data_size;
}

void* GetAudioOutBufferDataPointer(const AudioOutBuffer* b) { return b->buffer; }
size_t GetAudioOutBufferDataSize(const AudioOutBuffer* b) { return b->data_size; }

// El AudioOut de la consola siempre toca PCM de 16 bits a 48 kHz (lo que da InitializeAudioOutParameter).
// SampleFormat: Invalid 0, PcmInt8 1, PcmInt16 2, PcmInt24 3, PcmInt32 4, PcmFloat 5, Adpcm 6.
int GetAudioOutSampleRate(const AudioOut*) { return kSampleRate; }
// El nombre del dispositivo abierto: el de serie de la consola (Unreal lo guarda como nombre de su salida).
const char* GetAudioOutName(const AudioOut*) { return "MainAudioOut"; }
enum SampleFormat : int {};
SampleFormat GetAudioOutSampleFormat(const AudioOut*) { return SampleFormat{2}; }
size_t GetSampleByteSize(SampleFormat format) {
    switch (format) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    case 4:
    case 5: return 4;
    default: return 0;
    }
}

int GetAudioOutState(const AudioOut* out) {  // AudioOutState: Started 0, Stopped 1
    HostAudioOut* h = host(out);
    return h && h->started ? 0 : 1;
}

Result StartAudioOut(AudioOut* out) {
    trace_out("Start", out, __builtin_return_address(0));
    if (HostAudioOut* h = host(out)) {
        h->started = true;
        SharedStream::get().attach(h);
    }
    return ResultSuccess;
}

void StopAudioOut(AudioOut* out) {
    trace_out("Stop", out, __builtin_return_address(0));
    if (HostAudioOut* h = host(out)) {
        h->started = false;
        SharedStream::get().detach(h, false);
        // El hilo de FMOD solo mira su evento de fin al despertar del de la salida (main+0x1184ff8..0x1185064), y quien
        // cierra espera a ese hilo justo después de este Stop (main+0x1184f48..0x1184f54). detach señala el evento al
        // devolver los buffers pendientes, pero si no quedaba ninguno (la cola se había vaciado y el hilo ya había
        // consumido el aviso) nadie lo despertaba y el cierre se quedaba esperándolo para siempre: se señala siempre.
        if (h->event) nnsdk::signal_system_event(h->event);
    }
}

void CloseAudioOut(AudioOut* out) {
    trace_out("Close", out, __builtin_return_address(0));
    std::unique_ptr<HostAudioOut> h;
    {
        std::lock_guard lock(g_outs_mutex);
        auto it = g_outs.find(out);
        if (it == g_outs.end()) return;
        h = std::move(it->second);
        g_outs.erase(it);
    }
    SharedStream::get().detach(h.get(), true);  // sin marcas suyas en la cola antes de destruirla
    if (h->event) SharedStream::get().con_evento(-1);
}

bool AppendAudioOutBuffer(AudioOut* out, AudioOutBuffer* b) {
    trace_out("Append", out, __builtin_return_address(0));
    HostAudioOut* h = host(out);
    if (!h) return false;
    SharedStream::get().append(h, b);
    return true;
}

AudioOutBuffer* GetReleasedAudioOutBuffer(AudioOut* out) {
    HostAudioOut* h = host(out);
    AudioOutBuffer* b = h ? SharedStream::get().pop_released(h) : nullptr;
    trace_out(b ? "Released (uno)" : "Released (ninguno)", out, __builtin_return_address(0));
    return b;
}

// AudioIn (micrófono). La Flip no le da micrófono al juego: ListAudioIns dice que no hay ninguno y abrir uno falla,
// que es lo que ve el juego en la consola sin micrófono conectado. InitializeAudioInParameter escribe 8 bytes como el
// de AudioOut (escribir de más ya rompió la pila de Unity una vez).
struct AudioIn;
struct AudioInInfo;
struct AudioInParameter {
    int32_t sample_rate;
    uint16_t channel_count;
    uint16_t marca;
};
static_assert(sizeof(AudioInParameter) == 8);
struct AudioInBuffer {  // mismo formato que AudioOutBuffer en el SDK (tipo propio: va en el mangling)
    void* next;
    void* buffer;
    size_t buffer_size;
    size_t data_size;
    size_t reserved;
};

constexpr Result kAudioInNotFound{153 | 1 << 9};

void InitializeAudioInParameter(AudioInParameter* p) {
    p->sample_rate = 0;
    p->channel_count = 0;
    p->marca = 0xCAFE;
}
int ListAudioIns(AudioInInfo*, int) {
    static int avisos = 0;
    if (nvn::first_times(avisos, 2)) printf("[audio] ListAudioIns: ningún micrófono\n");
    return 0;
}
Result OpenAudioIn(AudioIn*, const char* name, const AudioInParameter&) {
    printf("[audio] OpenAudioIn(%s): sin micrófono\n", name ? name : "?");
    return kAudioInNotFound;
}
Result StartAudioIn(AudioIn*) { return kAudioInNotFound; }
void StopAudioIn(AudioIn*) {}
void CloseAudioIn(AudioIn*) {}
bool AppendAudioInBuffer(AudioIn*, AudioInBuffer*) { return false; }
AudioInBuffer* GetReleasedAudioInBuffer(AudioIn*) { return nullptr; }
int GetAudioInSampleRate(const AudioIn*) { return kSampleRate; }
int GetAudioInChannelCount(const AudioIn*) { return 1; }
SampleFormat GetAudioInSampleFormat(const AudioIn*) { return SampleFormat{2}; }
void SetAudioInBufferInfo(AudioInBuffer* b, void* buffer, size_t buffer_size, size_t data_size) {
    b->buffer = buffer;
    b->buffer_size = buffer_size;
    b->data_size = data_size;
}
void* GetAudioInBufferDataPointer(const AudioInBuffer* b) { return b->buffer; }

}  // namespace nn::audio

// Lo que usa el AudioRenderer del stream del proceso (audio_salida.h).
void nnsdk::audio::preparar_salida() { nn::audio::SharedStream::get(); }
void nnsdk::audio::fijar_fuente_extra(const FuenteExtra* fuente) { nn::audio::SharedStream::get().fuente(fuente); }

// --- Resampler --------------------------------------------------------------------------------------------------------
// El remuestreador de PCM16 del SDK. El mezclador de audio de Unreal Engine (NintendoAudioMixer, main+0x5921a84) pasa
// por él lo que mezcla cuando su frecuencia no es la de la salida. Interfaz como la del SDK (sdk+0x2634e0..0x263690):
//  - el juego da el buffer de trabajo: 0x260 bytes por canal (GetRequiredBufferSizeForResampler);
//  - ResamplerType guarda el buffer, su tamaño, las dos frecuencias y los canales;
//  - las cuentas van en frames (muestras por canal) y los buffers intercalados;
//  - GetResamplerOutputSampleCount es una cota para reservar: ceil(salida/entrada · n) + 2;
//  - ProcessResamplerBuffer da los frames que escribe y falla (audio, desc 4) si no caben en el buffer de salida.
// El del SDK es un polifásico en coma fija; este interpola con Hermite de 4 puntos y posición 32.32, con dos frames de
// retardo para no necesitar muestras del bloque siguiente. A igual frecuencia copia la entrada (retardada).
namespace nn::audio {

struct ResamplerType {
    void* buffer;
    size_t size;
    int32_t input_rate, output_rate, channels;
};

namespace {

constexpr size_t kResamplerBytesPerChannel = 0x260;
struct ResamplerChannel {  // dentro del buffer del juego, uno por canal
    uint64_t position;     // del primer canal: posición (32.32) de la próxima salida respecto del próximo bloque
    float history[3];      // los 3 últimos frames del bloque anterior
};
static_assert(sizeof(ResamplerChannel) <= kResamplerBytesPerChannel);

ResamplerChannel* resampler_channel(const ResamplerType* r, int c) {
    return reinterpret_cast<ResamplerChannel*>(static_cast<uint8_t*>(r->buffer) + c * kResamplerBytesPerChannel);
}
bool resampler_ok(const ResamplerType* r) {
    return r && r->buffer && r->channels > 0 && r->input_rate > 0 && r->output_rate > 0 &&
           r->size >= kResamplerBytesPerChannel * static_cast<size_t>(r->channels);
}

}  // namespace

size_t GetRequiredBufferSizeForResampler(int channels) {
    return kResamplerBytesPerChannel * static_cast<size_t>(channels > 0 ? channels : 0);
}

Result InitializeResampler(ResamplerType* r, void* buffer, size_t size, int input_rate, int output_rate, int channels) {
    r->buffer = buffer;
    r->size = size;
    r->input_rate = input_rate;
    r->output_rate = output_rate;
    r->channels = channels;
    if (resampler_ok(r)) memset(buffer, 0, kResamplerBytesPerChannel * static_cast<size_t>(channels));
    printf("[audio] resampler: %d -> %d Hz, %d canales\n", input_rate, output_rate, channels);
    return ResultSuccess;
}

int GetResamplerOutputSampleCount(const ResamplerType* r, int input_frames) {
    if (input_frames < 1 || !r || r->input_rate <= 0) return 0;
    return static_cast<int>(std::ceil(static_cast<float>(r->output_rate) / static_cast<float>(r->input_rate) *
                                      static_cast<float>(input_frames))) + 2;
}

Result ProcessResamplerBuffer(ResamplerType* r, int* output_frames, int16_t* output, size_t output_size,
                              const int16_t* input, int input_frames) {
    if (output_frames) *output_frames = 0;
    if (!resampler_ok(r) || !output || !input || input_frames <= 0) return ResultSuccess;
    constexpr uint64_t kOne = uint64_t{1} << 32;
    const uint64_t step = (static_cast<uint64_t>(r->input_rate) << 32) / static_cast<uint64_t>(r->output_rate);
    const uint64_t end = static_cast<uint64_t>(input_frames) << 32;
    const uint64_t start = resampler_channel(r, 0)->position;
    const uint64_t count = start < end && step ? (end - start + step - 1) / step : 0;
    if (output_frames) *output_frames = static_cast<int>(count);
    if (count * static_cast<uint64_t>(r->channels) * sizeof(int16_t) > output_size) return Result{153 | (4 << 9)};
    const int channels = r->channels;
    for (int c = 0; c < channels; c++) {
        ResamplerChannel* state = resampler_channel(r, c);
        // La señal con su historia: s(-3), s(-2), s(-1) son los del bloque anterior.
        auto sample = [&](int64_t k) -> float {
            return k < 0 ? state->history[k + 3] : static_cast<float>(input[k * channels + c]);
        };
        uint64_t position = start;
        for (uint64_t n = 0; n < count; n++, position += step) {
            const int64_t i = static_cast<int64_t>(position >> 32);
            const float t = static_cast<float>(position & (kOne - 1)) * (1.0f / 4294967296.0f);
            const float y0 = sample(i - 3), y1 = sample(i - 2), y2 = sample(i - 1), y3 = sample(i);
            const float c1 = 0.5f * (y2 - y0);
            const float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
            const float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
            const float v = ((c3 * t + c2) * t + c1) * t + y1;
            output[n * channels + c] = static_cast<int16_t>(std::lrintf(std::clamp(v, -32768.0f, 32767.0f)));
        }
        float history[3];
        for (int k = 0; k < 3; k++) history[k] = sample(static_cast<int64_t>(input_frames) - 3 + k);
        memcpy(state->history, history, sizeof history);
    }
    resampler_channel(r, 0)->position = start + count * step - end;
    return ResultSuccess;
}

}  // namespace nn::audio
