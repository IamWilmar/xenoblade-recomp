// nn::audio, la parte del AudioRenderer (voces, final mix, device sink, memory pools). La otra mitad, AudioOut, está en
// audio.cpp.
//
// Quién lo usa en Disco Elysium: solo el reproductor de vídeo de Unity para Switch ("SwitchVideoPlayer", el
// MediaDecoderPlayer de los ejemplos de Nintendo), para la pista de audio de un VideoClip. Todas las llamadas de main a
// estas funciones están entre main+0x397420 y main+0x39aa6c. El sonido del juego (FMOD) no pasa por aquí: su salida
// "FMOD NX Output" (fmod_output_nx.cpp, main+0x1184b40) usa AudioOut. Y los dos VideoClip que he encontrado en la RomFS
// (Data/sharedassets1.resource, 128x128, y Data/sharedassets2.resource, 1280x720) solo llevan pista de vídeo: con
// ellos el reproductor no llega a preparar el audio y nada de esto se ejecuta.
//
// Lo que hace el reproductor, y por tanto lo que aquí tiene que funcionar:
//  - Preparar (main+0x399c3c), solo si el vídeo trae audio (main+0x399b40): InitializeAudioRendererParameter y encima
//    sampleRate = el de la pista, sampleCount = sampleRate / 200, mixBufferCount = los canales de la pista sin pasar de
//    los del modo de altavoces de Unity ni de 8 (y 2 si la pista es mono, main+0x399c8c..0x399cb4), voiceCount = los
//    canales de la pista, sinkCount = 1 (constante de main+0x3f4edb0). IsValidAudioRendererParameter (si da false el
//    vídeo se queda sin audio), los dos tamaños de work buffer (memalign de 4096), OpenAudioRenderer con un SystemEvent,
//    InitializeAudioRendererConfig, AcquireFinalMix, AddDeviceSink("MainAudioOut") con las entradas 0, 1…, un
//    RequestUpdate y StartAudioRenderer. Luego una voz PcmInt16 (formato 2) con todos los canales, SetVoiceVolume,
//    SetVoiceDestination al final mix, cuatro memory pools de 32 KB (AcquireMemoryPool + RequestAttachMemoryPool) con su
//    WaveBuffer cada uno, SetVoicePlayState(Play = 0), SetVoiceMixVolume(0.5, canal i -> buffer i) y otro RequestUpdate.
//  - Tocar (hilo "AudioPlayerThread", main+0x3971f0): espera el SystemEvent del renderer con TimedWaitSystemEvent de
//    100 ms (main+0x39733c) y, si no llega, vuelve a esperar: sin ese evento solo avanza cuando el propio reproductor
//    se lo pide por un contador suyo (this+0xfc0, main+0x3972b4). Con él, saca el PCM del decodificador, recoge con
//    GetReleasedWaveBuffer los buffers ya tocados (los compara por dirección con los suyos, main+0x397428), copia el
//    PCM a un pool libre con CopyMemoryPoolData, pone endSampleOffset = bytes / 2 / canales, AppendWaveBuffer y
//    RequestUpdateAudioRenderer. Sin buffer libre repite RequestUpdate + 16,7 ms de espera.
//  - Cerrar (main+0x397b6c): RequestDetachMemoryPool de los cuatro, SetVoicePlayState(Stop = 1) y, mientras alguno siga
//    IsMemoryPoolAttached, WaitSystemEvent + RequestUpdate. Después DestroySystemEvent (¡con el renderer todavía en
//    marcha!), ReleaseMemoryPool, ReleaseVoiceSlot, ReleaseFinalMix, StopAudioRenderer si GetAudioRendererState no es 1,
//    RemoveDeviceSink y CloseAudioRenderer.
//
// La semántica de cada función sale del SDK de verdad, que viene en el volcado (work/scope/sdk.elf, "sdk+0x…"; se lee
// con tools/sdk_dis.py) y en esta versión trae también el servidor de audio (nn::audio::server). Como en el SDK, lo
// que el juego toca (AudioRendererConfig) es un estado suyo que vive en el work buffer que él reserva, y
// RequestUpdateAudioRenderer es el único punto en que ese estado pasa al mezclador y vuelven de él los buffers
// consumidos y el estado de los pools. El mezclador (un Renderer por handle) vive en nuestro heap y lo toca el callback
// de AAudio (audio.cpp) a través de nnsdk::audio::FuenteExtra: nada de lo que hace el juego fuera de RequestUpdate,
// Start/Stop/Close y los Release le llega.
//
// Interruptor sin_audio_renderer: IsValidAudioRendererParameter da false y el reproductor sigue sin audio, como cuando
// todo esto eran stubs.

#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "nnsdk/audio_salida.h"
#include "nnsdk/audio_voz.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/os.h"
#include "nnsdk/types.h"

namespace nn::os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}

namespace nn::audio {

using nnsdk::audio::kSalidaHz;
using nnsdk::audio::kVozCanalesMax;
using nnsdk::audio::kVozOndasMax;
using nnsdk::audio::OndaVoz;
using nnsdk::audio::VozMezcla;

// --- Lo que el juego reserva y lee -------------------------------------------------------------------------------

// 0x38 bytes: InitializeAudioRendererParameter los pone todos (sdk+0x3af10) y el reproductor le reserva justo eso en
// la pila (main+0x399c64: sp+0x38, con d8 guardado en sp+0x70).
struct AudioRendererParameter {
    int32_t sampleRate;
    int32_t sampleCount;
    int32_t mixBufferCount;
    int32_t subMixCount;
    int32_t voiceCount;
    int32_t sinkCount;
    int32_t effectCount;
    int32_t performanceFrameCount;
    bool isVoiceDropEnabled;
    int32_t splitterCount;
    int32_t splitterSendChannelCount;
    int32_t renderingDevice;  // 0: el DSP; 1: la CPU
    int32_t executionMode;    // 0: automático; 1: manual
    uint32_t _magic;          // la revisión del SDK
};
static_assert(sizeof(AudioRendererParameter) == 0x38 && offsetof(AudioRendererParameter, splitterCount) == 0x24 &&
              offsetof(AudioRendererParameter, _magic) == 0x34);

// Dos punteros en el SDK (el objeto del servicio y el SystemEvent propio del modo manual, sdk+0x4c39c); el juego lo
// guarda en 16 bytes (main+0x399f4c: this+0xcb0) y lo pasa por valor en x0:x1 (main+0x399fac).
struct AudioRendererHandle {
    uint64_t id;
    uint64_t marca;
};
static_assert(sizeof(AudioRendererHandle) == 16);

namespace {
struct Configuracion;
struct InfoVoz;
struct InfoMix;
struct InfoSumidero;
struct InfoPool;
}  // namespace

// 0x78 bytes: los 15 punteros que escribe InitializeAudioRendererConfig (sdk+0x3b478..0x3b4bc), y lo que le reserva
// el reproductor (this+0xcc0..0xd38). Aquí solo hace falta uno, al estado que va en el work buffer.
struct AudioRendererConfig {
    uint64_t marca;
    Configuracion* c;
    uint64_t sin_uso[13];
};
static_assert(sizeof(AudioRendererConfig) == 0x78);

// Los cuatro son un puntero a su estado dentro del config, como en el SDK (sdk+0x46288, sdk+0x43378, sdk+0x4453c,
// sdk+0x42850); el reproductor reserva 8 bytes para cada uno (this+0xd48, +0xd40, +0xd38 y +0x40 de cada ranura).
struct VoiceType {
    enum PlayState : int { PlayState_Play = 0, PlayState_Stop = 1, PlayState_Pause = 2 };
    InfoVoz* info;
};
struct FinalMixType {
    InfoMix* info;
};
struct DeviceSinkType {
    InfoSumidero* info;
};
struct MemoryPoolType {
    InfoPool* info;
};

// AppendWaveBuffer copia estos campos (sdk+0x46aa4..0x46aec); los tres últimos solo con MemberOptions, que el juego no
// usa. El reproductor rellena hasta contextSize (main+0x39a0ac..0x39a0d4) y deja 0x40 bytes por ranura.
struct WaveBuffer {
    const void* buffer;
    size_t size;
    int32_t startSampleOffset;  // en frames: la muestra es buffer[(start + frame) * canales + canal] (sdk+0x6d5b0)
    int32_t endSampleOffset;
    bool loop;
    bool isEndOfStream;
    const void* pContext;
    size_t contextSize;
    int32_t loopStartSampleOffset;
    int32_t loopEndSampleOffset;
    int32_t loopCount;
};
static_assert(sizeof(WaveBuffer) == 0x40 && offsetof(WaveBuffer, pContext) == 0x20);

enum SampleFormat : int {};  // Invalid 0, PcmInt8 1, PcmInt16 2, PcmInt24 3, PcmInt32 4, PcmFloat 5, Adpcm 6

namespace {

constexpr int kPcmInt16 = 2;
constexpr uint32_t kRevision = 0x38564552;  // "REV8": common::GetCurrentRevision (sdk+0x482f0)
// Lo que dan AddDeviceSink sin sink libre (sdk+0x44550) y OpenAudioRenderer con dos ya abiertos (sdk+0x4bf54).
constexpr Result kSinRecursos{0xa99};
constexpr int kMixBuffersVoz = 24;          // los mixVolume de cada canal de una voz (sdk+0x46344: 0x60 bytes)
constexpr int kSumideroEntradas = 6;        // AddDeviceSink copia como mucho 6 (sdk+0x444c0)
constexpr int32_t kSinDestino = 0x7fffffff; // VoiceInfo+0x98 hasta SetVoiceDestination (sdk+0x462e0)
constexpr int32_t kMixFinal = 0;            // el id del final mix (sdk+0x43360)

// Estados de un memory pool en el cliente (MemoryPoolInfo+0x10). Los pasos de RequestAttach/RequestDetach son los de
// MemoryPoolInfo::Attach (sdk+0x42b30, tabla 0xa36c24) y ::Detach (sdk+0x42c2c).
enum EstadoPool : int32_t {
    kPoolNuevo = 1,
    kPoolPideSoltar = 2,
    kPoolSuelto = 3,
    kPoolPideEnganchar = 4,
    kPoolEnganchado = 5,
    kPoolLiberado = 6,
};

// --- El estado del config, dentro del work buffer del juego (lo que el SDK llama VoiceInfo, MixInfo…) -------------

struct InfoOnda {
    const WaveBuffer* del_juego;  // lo que devuelve GetReleasedWaveBuffer: el juego los reconoce por su dirección
    const void* datos;
    size_t bytes;
    int32_t inicio, fin;
    bool bucle;
};

struct InfoVoz {
    bool en_uso;
    uint8_t canales;
    int32_t hz;
    int32_t indice;
    uint32_t generacion;  // cambia en cada AcquireVoiceSlot: el mezclador reinicia su voz (el isNew del SDK)
    // SetVoiceVolume lo puede llamar otro hilo que el que actualiza (en el reproductor, el del juego: main+0x39aa6c).
    std::atomic<float> volumen, tono;
    std::atomic<uint8_t> estado;  // VoiceType::PlayState
    int32_t destino;
    float mezcla[kVozCanalesMax][kMixBuffersVoz];  // SetVoiceMixVolume(canal, buffer del mix)
    // El anillo de WaveBuffer del cliente (VoiceInfo+0x10/+0x7c/+0x80/+0x30/+0x1b8/+0x1c8).
    InfoOnda ondas[kVozOndasMax];
    uint32_t adjuntados;  // el n-ésimo va a ondas[n % 4]
    uint32_t enviados;    // de ellos, los que ya tiene el mezclador
    uint32_t devueltos;   // los que ya ha recogido GetReleasedWaveBuffer
    uint32_t consumidos;  // los que el mezclador había tocado en el último RequestUpdate
    int64_t tocados;      // frames que la voz había tocado en el último RequestUpdate (GetVoicePlayedSampleCount)
};

struct InfoMix {
    bool en_uso;
    int32_t buffers;
    float volumen;
};

struct InfoSumidero {
    bool en_uso;
    int32_t entradas;
    int8_t entrada[kSumideroEntradas];  // el buffer del final mix que alimenta cada canal del dispositivo
};

struct InfoPool {
    bool en_uso;
    int32_t estado;
    void* direccion;
    size_t bytes;
};

constexpr uint64_t kMarcaConfig = 0x31676663'6f647561;  // "audocfg1"
constexpr uint64_t kMarcaHandle = 0x316e6572'6f647561;  // "audoren1"

struct Configuracion {
    uint64_t marca;
    int32_t hz, n_voces, n_pools, n_sumideros, mix_buffers;
    int32_t voces_usadas, canales_usados;
    uint32_t generacion;
    uint64_t renderer;  // el del último RequestUpdate: los Release* le quitan ya lo suyo al mezclador
    InfoMix mix_final;
    InfoVoz* voces;
    InfoPool* pools;
    InfoSumidero* sumideros;
};

size_t alinear(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// Las cuentas del config salen del parámetro como en el SDK: voiceCount voces y voiceCount canales entre todas
// (VoiceInfoManager::Initialize, sdk+0x46e28 y sdk+0x46fbc), y effectCount + 4 * voiceCount memory pools
// (MemoryPoolManager::GetWorkBufferSize, sdk+0x42d74).
int pools_de(const AudioRendererParameter& p) { return p.effectCount + 4 * p.voiceCount; }

size_t bytes_config(const AudioRendererParameter& p) {
    return alinear(sizeof(Configuracion), 16) + alinear(sizeof(InfoVoz) * size_t(p.voiceCount), 16) +
           alinear(sizeof(InfoPool) * size_t(pools_de(p)), 16) + alinear(sizeof(InfoSumidero) * size_t(p.sinkCount), 16);
}

Configuracion* config_de(const AudioRendererConfig* config) {
    return config && config->marca == kMarcaConfig ? config->c : nullptr;
}

// --- El mezclador ------------------------------------------------------------------------------------------------

struct VozRenderer {
    VozMezcla mezcla;
    uint32_t generacion = 0;
    uint32_t base = 0;  // los WaveBuffer que la voz del config ya tenía consumidos cuando llegó a este renderer
    bool activa = false;
};

struct Renderer {
    uint64_t id = 0;
    bool en_marcha = false;
    os::SystemEventType* evento = nullptr;
    int hz = kSalidaHz;
    std::vector<VozRenderer> voces;
    float depop[2] = {};
    // DEPURACIÓN (registro): lo del informe.
    uint64_t frames = 0, actualizaciones = 0, ondas = 0;
    float pico = 0;

    // Con g_m, en el callback de AAudio: suma `frames` frames a `bloque` (estéreo intercalado).
    void mezclar(float* bloque, size_t n) {
        for (VozRenderer& v : voces)
            if (v.activa) v.mezcla.mezclar(bloque, n);
        if (depop[0] != 0 || depop[1] != 0) {
            for (size_t i = 0; i < n; i++)
                for (int k = 0; k < 2; k++) bloque[2 * i + k] += depop[k] *= nnsdk::audio::kDepopCaida;
            for (float& d : depop)
                if (std::fabs(d) < 0.5f) d = 0;
        }
        for (size_t i = 0; i < 2 * n; i++) pico = std::max(pico, std::fabs(bloque[i]));
        frames += n;
    }
    void descartar(size_t n) {
        for (VozRenderer& v : voces)
            if (v.activa) v.mezcla.descartar(n);
        depop[0] = depop[1] = 0;
    }
};

// Uno para todo: el callback de AAudio lo toma en cada ráfaga (solo si hay algún renderer en marcha) y el juego en
// RequestUpdate, Start/Stop/Close y los Acquire/Release, que son cuentas cortas. Con él tomado no se escribe en el
// log, no se mira ningún interruptor ni se llama a audio.cpp.
// Ni él ni la tabla se destruyen nunca: el hilo del evento y el callback de AAudio siguen hasta que muere el proceso,
// también mientras corren los destructores estáticos de la salida.
std::mutex& g_m = *new std::mutex;
constexpr int kRenderersMax = 2;  // los que deja abrir la consola por proceso (sdk+0x4bf58)
auto& g_renderers = *new std::array<std::unique_ptr<Renderer>, kRenderersMax>;
std::atomic<int> g_en_marcha{0};
uint64_t g_serie = 0;

Renderer* buscar(uint64_t id) {  // con g_m
    Renderer* r = id ? g_renderers[id % kRenderersMax].get() : nullptr;
    return r && r->id == id ? r : nullptr;
}
Renderer* buscar(const AudioRendererHandle& h) { return h.marca == kMarcaHandle ? buscar(h.id) : nullptr; }

constexpr size_t kBloque = 256;

void fuente_mezclar(int16_t* salida, size_t frames) {
    if (g_en_marcha.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard lock(g_m);
    float bloque[kBloque * 2];
    for (size_t hecho = 0; hecho < frames;) {
        const size_t n = std::min(frames - hecho, kBloque);
        std::memset(bloque, 0, n * 2 * sizeof(float));
        for (auto& r : g_renderers)
            if (r && r->en_marcha) r->mezclar(bloque, n);
        int16_t* out = salida + hecho * 2;
        for (size_t i = 0; i < 2 * n; i++) {
            if (bloque[i] == 0) continue;
            out[i] = static_cast<int16_t>(std::clamp<long>(out[i] + std::lrintf(bloque[i]), -32768, 32767));
        }
        hecho += n;
    }
}

void fuente_descartar(size_t frames) {
    if (g_en_marcha.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard lock(g_m);
    for (auto& r : g_renderers)
        if (r && r->en_marcha) r->descartar(frames);
}

constexpr nnsdk::audio::FuenteExtra kFuente{fuente_mezclar, fuente_descartar};

// --- El evento del renderer --------------------------------------------------------------------------------------
// El servidor señala el SystemEvent de OpenAudioRenderer una vez por frame de audio (sampleCount muestras: 5 ms a 32 y
// a 48 kHz) y solo con el renderer en marcha (AudioRenderSystem::SendCommandToDsp, sdk+0x5a108..0x5a244); el evento es
// AutoClear (sdk+0x4c3b0). El hilo del reproductor no hace nada hasta que llega (main+0x39733c), así que aquí lo da un
// hilo propio cada 5 ms mientras haya algún renderer en marcha, y duerme sin gastar nada si no lo hay (en Disco Elysium,
// casi siempre). El juego destruye el evento antes de parar el renderer (main+0x397c70): se señala solo si sigue vivo.

constexpr auto kPeriodo = std::chrono::milliseconds(5);
constexpr auto kInforme = std::chrono::seconds(5);
std::condition_variable& g_tic_cv = *new std::condition_variable;
bool g_tic_creado = false;  // con g_m

void hilo_tic() {
#if defined(__ANDROID__) || defined(__linux__)
    pthread_setname_np(pthread_self(), "AudioRenderer");
#endif
    using reloj = std::chrono::steady_clock;
    auto siguiente = reloj::now(), informe = siguiente + kInforme;
    for (;;) {
        struct {
            os::SystemEventType* evento;
            uint64_t id;
        } avisos[kRenderersMax];
        int n = 0;
        char texto[kRenderersMax][200];
        int n_texto = 0;
        const bool registro = NVN_FLAG("registro");  // fuera del cerrojo
        {
            std::unique_lock lock(g_m);
            if (g_en_marcha.load() == 0) {
                g_tic_cv.wait(lock, [] { return g_en_marcha.load() > 0; });
                siguiente = reloj::now();
                informe = siguiente + kInforme;
            }
            const bool toca_informe = siguiente >= informe;
            for (auto& r : g_renderers) {
                if (!r || !r->en_marcha) continue;
                if (r->evento) avisos[n++] = {r->evento, r->id};
                if (!toca_informe) continue;
                if (registro) {
                    int sonando = 0, en_cola = 0;
                    for (const VozRenderer& v : r->voces) {
                        if (!v.activa) continue;
                        sonando += v.mezcla.estado() == VozMezcla::kSonando;
                        en_cola += v.mezcla.en_cola();
                    }
                    snprintf(texto[n_texto++], sizeof texto[0],
                             "[audio] renderer %llu: %.1f s tocados, %llu RequestUpdate, %llu WaveBuffer, %d voces "
                             "sonando con %d buffers en cola, pico %.0f%s\n",
                             static_cast<unsigned long long>(r->id >> 8), double(r->frames) / kSalidaHz,
                             static_cast<unsigned long long>(r->actualizaciones),
                             static_cast<unsigned long long>(r->ondas), sonando, en_cola, double(r->pico),
                             r->evento ? "" : " (el juego ya destruyó su evento)");
                }
                r->frames = r->actualizaciones = r->ondas = 0;
                r->pico = 0;
            }
            if (toca_informe) informe = siguiente + kInforme;
        }
        for (int i = 0; i < n_texto; i++) fputs(texto[i], stdout);
        for (int i = 0; i < n; i++) {
            if (nnsdk::signal_system_event_if_alive(avisos[i].evento)) continue;
            std::lock_guard lock(g_m);  // el juego lo destruyó: esa dirección ya no es nuestra
            if (Renderer* r = buscar(avisos[i].id); r && r->evento == avisos[i].evento) r->evento = nullptr;
        }
        siguiente += kPeriodo;
        if (const auto ahora = reloj::now(); siguiente < ahora) siguiente = ahora + kPeriodo;  // sin ráfagas de atrasos
        std::this_thread::sleep_until(siguiente);
    }
}

// Con g_m: un renderer más (o menos) en marcha.
void contar_en_marcha(int delta) {
    g_en_marcha.fetch_add(delta);
    if (delta > 0) {
        if (!g_tic_creado) {
            g_tic_creado = true;
            std::thread(hilo_tic).detach();
        }
        g_tic_cv.notify_all();
    }
}

// --- RequestUpdate: el config pasa al mezclador ---------------------------------------------------------------------

// Cuánto de cada canal de la voz llega a cada canal del dispositivo: volumen de la voz por el de mezcla hacia cada
// buffer del final mix, por el volumen del mix, y de ahí por las entradas del device sink (la entrada i es el buffer
// del mix que alimenta el canal i del dispositivo: server::CommandGenerator::GenerateDeviceSinkCommand, sdk+0x5f214).
// El reproductor nunca pasa de 2 entradas con el dispositivo en estéreo (main+0x399c9c). Con 6 (5.1) se baja a
// estéreo con frontal 1, central 0.707, LFE 0.251 y trasero 0.707: son los que usa yuzu para el dispositivo, no salen
// del volcado (en la consola esa bajada la hace el dispositivo de salida, fuera del SDK).
void ganancias(const Configuracion& c, const InfoVoz& v, float g[kVozCanalesMax][2]) {
    std::memset(g, 0, sizeof(float) * kVozCanalesMax * 2);
    if (v.destino != kMixFinal || !c.mix_final.en_uso) return;
    float ruta[kMixBuffersVoz][2] = {};
    bool hay = false;
    for (int i = 0; i < c.n_sumideros; i++) {
        const InfoSumidero& s = c.sumideros[i];
        if (!s.en_uso) continue;
        static constexpr float k51[kSumideroEntradas][2] = {{1, 0}, {0, 1}, {0.707f, 0.707f},
                                                            {0.251f, 0.251f}, {0.707f, 0}, {0, 0.707f}};
        for (int e = 0; e < s.entradas && e < kSumideroEntradas; e++) {
            const int d = s.entrada[e];
            if (d < 0 || d >= kMixBuffersVoz || d >= c.mix_final.buffers) continue;
            hay = true;
            if (s.entradas == 1) {
                ruta[d][0] += 1, ruta[d][1] += 1;
            } else if (s.entradas == kSumideroEntradas) {
                ruta[d][0] += k51[e][0], ruta[d][1] += k51[e][1];
            } else if (e < 2) {
                ruta[d][e] += 1;
            }
        }
    }
    if (!hay) return;
    const float vol = v.volumen.load(std::memory_order_relaxed) * c.mix_final.volumen;
    for (int canal = 0; canal < v.canales; canal++)
        for (int d = 0; d < kMixBuffersVoz && d < c.mix_final.buffers; d++) {
            const float m = v.mezcla[canal][d] * vol;
            g[canal][0] += m * ruta[d][0];
            g[canal][1] += m * ruta[d][1];
        }
}

// Con g_m. El orden es el del servidor (AudioRenderSystem::Update: UpdateMemoryPools en sdk+0x59af0, UpdateVoices en
// sdk+0x59b1c), así que un pool cuya voz se para en esta misma actualización se suelta en la siguiente, como en la
// consola.
void actualizar(Renderer& r, Configuracion& c) {
    c.renderer = r.id;
    r.actualizaciones++;
    for (int i = 0; i < c.n_pools; i++) {
        InfoPool& p = c.pools[i];
        if (!p.en_uso) continue;
        if (p.estado == kPoolPideEnganchar) {
            p.estado = kPoolEnganchado;
        } else if (p.estado == kPoolPideSoltar) {
            // PoolMapper::Update no lo desmapea mientras lo esté leyendo una voz (MemoryPoolInfo::IsUsed, que marcan
            // los comandos de las voces que suenan): se queda pidiéndolo. Las voces paradas o en pausa no lo retienen,
            // pero aquí dejan de apuntar a esa memoria, que el juego puede liberar en cuanto lo vea suelto.
            const bool usado = std::any_of(r.voces.begin(), r.voces.end(), [&](const VozRenderer& v) {
                return v.activa && v.mezcla.estado() == VozMezcla::kSonando && v.mezcla.usa(p.direccion, p.bytes);
            });
            if (usado) continue;
            for (VozRenderer& v : r.voces)
                if (v.activa && v.mezcla.usa(p.direccion, p.bytes)) v.mezcla.vaciar();
            p.estado = kPoolSuelto;
        }
    }
    const int n = std::min<int>(c.n_voces, static_cast<int>(r.voces.size()));
    for (int i = 0; i < n; i++) {
        InfoVoz& iv = c.voces[i];
        VozRenderer& vr = r.voces[i];
        if (!iv.en_uso) {
            if (vr.activa) {
                vr.mezcla.cortar(r.depop);
                vr.mezcla.reiniciar();
                vr.activa = false;
            }
            continue;
        }
        const bool nueva = !vr.activa || vr.generacion != iv.generacion;
        if (nueva) {
            if (vr.activa) vr.mezcla.cortar(r.depop);
            vr.mezcla.reiniciar();
            vr.generacion = iv.generacion;
            vr.activa = true;
            // Recién adquirida, las dos cuentas valen 0. Si el config viene de otro renderer (uno que se cerró), la
            // voz sigue aquí con lo que aquel no llegó a tocar.
            vr.base = iv.consumidos;
            iv.enviados = iv.consumidos;
        }
        const float tono = iv.tono.load(std::memory_order_relaxed);
        vr.mezcla.configurar(iv.canales, double(iv.hz) * double(tono) / kSalidaHz);
        // Los WaveBuffer nuevos, antes que el estado: los que lleguen junto con un Stop vuelven también al juego.
        for (; iv.enviados != iv.adjuntados; iv.enviados++) {
            const InfoOnda& o = iv.ondas[iv.enviados % kVozOndasMax];
            OndaVoz onda;
            // Lo que no cabe en su buffer no se lee (server::VoiceInfo::UpdateWaveBuffer rechaza el buffer entero).
            const int64_t caben = static_cast<int64_t>(o.bytes / (sizeof(int16_t) * iv.canales));
            onda.datos = static_cast<const int16_t*>(o.datos);
            onda.inicio = std::max(o.inicio, 0);
            onda.fin = static_cast<int32_t>(std::min<int64_t>(o.fin, caben));
            onda.bucle = o.bucle;
            vr.mezcla.encolar(onda);
            r.ondas++;
        }
        float g[kVozCanalesMax][2];
        ganancias(c, iv, g);
        vr.mezcla.fijar_ganancias(g, !nueva);
        switch (iv.estado.load(std::memory_order_relaxed)) {
        case VoiceType::PlayState_Play: vr.mezcla.fijar_estado(VozMezcla::kSonando, r.depop); break;
        case VoiceType::PlayState_Pause: vr.mezcla.fijar_estado(VozMezcla::kPausada, r.depop); break;
        default: vr.mezcla.fijar_estado(VozMezcla::kParada, r.depop); break;
        }
        iv.consumidos = vr.base + vr.mezcla.consumidos();
        iv.tocados = static_cast<int64_t>(vr.mezcla.tocados());
    }
}

// Con g_m: lo que el mezclador tuviera de esta voz deja de sonar y de mirar la memoria del juego ya, sin esperar al
// próximo RequestUpdate (el juego puede liberar esa memoria en cuanto vuelve de ReleaseVoiceSlot).
void quitar_voz(Configuracion& c, const InfoVoz& iv) {
    Renderer* r = buscar(c.renderer);
    if (!r || iv.indice < 0 || iv.indice >= static_cast<int>(r->voces.size())) return;
    VozRenderer& vr = r->voces[iv.indice];
    if (!vr.activa || vr.generacion != iv.generacion) return;
    vr.mezcla.cortar(r->depop);
    vr.mezcla.reiniciar();
    vr.activa = false;
}

InfoVoz* voz_de(const VoiceType* v) { return v && v->info && v->info->en_uso ? v->info : nullptr; }
InfoPool* pool_de(const MemoryPoolType* p) { return p && p->info && p->info->en_uso ? p->info : nullptr; }

}  // namespace

// --- Parámetro y tamaños ----------------------------------------------------------------------------------------------

// Los valores por defecto del SDK (sdk+0x3af24..0x3af48): 48 kHz, 240 muestras por frame, un mix buffer, una voz y
// un sink; lo demás a cero y la revisión en _magic.
void InitializeAudioRendererParameter(AudioRendererParameter* p) {
    std::memset(p, 0, sizeof *p);
    p->sampleRate = 48000;
    p->sampleCount = 240;
    p->mixBufferCount = 1;
    p->voiceCount = 1;
    p->sinkCount = 1;
    p->_magic = kRevision;
}

// Las mismas comprobaciones que el SDK (sdk+0x3adc0..0x3aebc). La que importa aquí: la frecuencia del renderer solo
// puede ser 32000 o 48000 con su sampleCount de 5 ms; el reproductor pasa la de la pista de audio del vídeo, y con
// cualquier otra (44100) se queda sin audio en la consola igual que aquí.
bool IsValidAudioRendererParameter(const AudioRendererParameter& p) {
    if (NVN_FLAG("sin_audio_renderer")) return false;
    if (p.renderingDevice != p.executionMode || (p.renderingDevice != 0 && p.renderingDevice != 1)) return false;
    if (p._magic != kRevision) return false;
    if (p.sampleRate != 48000 && p.sampleRate != 32000) return false;
    if (p.sampleCount != p.sampleRate / 200) return false;
    return p.mixBufferCount >= 1 && p.mixBufferCount <= 256 && p.subMixCount >= 0 && p.subMixCount <= 256 &&
           p.voiceCount >= 1 && p.voiceCount <= 1024 && p.sinkCount >= 1 && p.sinkCount <= 256 && p.effectCount >= 0 &&
           p.effectCount <= 256;
}

// En la consola es la memoria de transferencia del servicio de audio (comandos del DSP, estado de las voces…). Aquí
// el mezclador vive en nuestro heap y no la toca: una página, para que el juego siga su camino normal (la reserva con
// memalign(4096) y se la pasa a OpenAudioRenderer, main+0x399d04 y main+0x399f48).
size_t GetAudioRendererWorkBufferSize(const AudioRendererParameter&) { return 0x1000; }

// Lo que ocupa el estado del config (el de InitializeAudioRendererConfig), en páginas como el del SDK (sdk+0x3b074).
size_t GetAudioRendererConfigWorkBufferSize(const AudioRendererParameter& p) {
    if (p.voiceCount < 1 || p.voiceCount > 1024 || p.sinkCount < 0 || p.sinkCount > 256 || p.effectCount < 0 ||
        p.effectCount > 256)
        return 0x1000;
    return alinear(bytes_config(p) + 16, 0x1000);
}

// --- Renderer -----------------------------------------------------------------------------------------------------------

Result OpenAudioRenderer(AudioRendererHandle* out, os::SystemEvent* event, const AudioRendererParameter& p, void*,
                         size_t) {
    *out = {};
    if (!IsValidAudioRendererParameter(p)) {
        // El servicio de la consola daría su propio error; el reproductor ni mira este resultado (main+0x399f5c) y ya
        // ha comprobado el parámetro antes.
        printf("[audio] OpenAudioRenderer con un parámetro no válido (%d Hz, %d muestras, %d voces): no se abre\n",
               p.sampleRate, p.sampleCount, p.voiceCount);
        return kSinRecursos;
    }
    // El stream de AAudio ya suele estar en marcha (lo abre el primer uso de nn::audio, al arrancar FMOD); si no, las
    // voces avanzan con el reloj de reserva de audio.cpp hasta que lo esté.
    nnsdk::audio::preparar_salida();
    nnsdk::audio::fijar_fuente_extra(&kFuente);
    auto* e = reinterpret_cast<os::SystemEventType*>(event);
    if (e) nnsdk::create_system_event(e, /*auto_clear=*/true, /*signaled=*/false);
    auto r = std::make_unique<Renderer>();
    r->hz = p.sampleRate;
    r->evento = e;
    r->voces.resize(static_cast<size_t>(p.voiceCount));
    uint64_t id = 0;
    {
        std::lock_guard lock(g_m);
        for (int i = 0; i < kRenderersMax && !id; i++) {
            const uint64_t candidato = (++g_serie << 8) | static_cast<uint64_t>(i);
            if (g_renderers[candidato % kRenderersMax]) continue;
            id = r->id = candidato;
            g_renderers[candidato % kRenderersMax] = std::move(r);
        }
    }
    if (!id) {
        printf("[audio] OpenAudioRenderer: ya hay %d abiertos, que es el máximo\n", kRenderersMax);
        return kSinRecursos;
    }
    *out = {id, kMarcaHandle};
    printf("[audio] AudioRenderer %llu abierto: %d Hz, %d muestras por frame, %d voces, %d mix buffers, %d sinks; evento "
           "%p cada 5 ms mientras esté en marcha\n",
           static_cast<unsigned long long>(id >> 8), p.sampleRate, p.sampleCount, p.voiceCount, p.mixBufferCount,
           p.sinkCount, static_cast<void*>(e));
    return ResultSuccess;
}

// AudioRenderSystem::Start (sdk+0x59a0c): activo, y desde el siguiente frame se señala el evento.
Result StartAudioRenderer(AudioRendererHandle handle) {
    bool hecho = false;
    {
        std::lock_guard lock(g_m);
        if (Renderer* r = buscar(handle); r && !r->en_marcha) {
            r->en_marcha = true;
            r->depop[0] = r->depop[1] = 0;
            contar_en_marcha(+1);
            hecho = true;
        }
    }
    if (hecho) printf("[audio] AudioRenderer %llu en marcha\n", static_cast<unsigned long long>(handle.id >> 8));
    return ResultSuccess;
}

// AudioRenderSystem::Stop (sdk+0x599ac): deja de renderizar y de señalar el evento; las voces conservan su estado.
void StopAudioRenderer(AudioRendererHandle handle) {
    bool hecho = false;
    {
        std::lock_guard lock(g_m);
        if (Renderer* r = buscar(handle); r && r->en_marcha) {
            r->en_marcha = false;
            contar_en_marcha(-1);
            hecho = true;
        }
    }
    if (hecho) printf("[audio] AudioRenderer %llu parado\n", static_cast<unsigned long long>(handle.id >> 8));
}

void CloseAudioRenderer(AudioRendererHandle handle) {
    std::unique_ptr<Renderer> r;
    {
        std::lock_guard lock(g_m);
        if (!buscar(handle)) return;
        r = std::move(g_renderers[handle.id % kRenderersMax]);
        if (r->en_marcha) contar_en_marcha(-1);
    }
    printf("[audio] AudioRenderer %llu cerrado\n", static_cast<unsigned long long>(handle.id >> 8));
}

// AudioRendererState: Started 0, Stopped 1 (el reproductor lo para si no es 1, main+0x397cd4).
int GetAudioRendererState(AudioRendererHandle handle) {
    std::lock_guard lock(g_m);
    const Renderer* r = buscar(handle);
    return r && r->en_marcha ? 0 : 1;
}

Result RequestUpdateAudioRenderer(AudioRendererHandle handle, const AudioRendererConfig* config) {
    Configuracion* c = config_de(config);
    if (!c) return ResultSuccess;
    os::SystemEventType* evento = nullptr;
    {
        std::lock_guard lock(g_m);
        if (Renderer* r = buscar(handle)) {
            actualizar(*r, *c);
            evento = r->evento;
        }
    }
    // El servidor limpia el evento al acabar cada actualización (AudioRenderSystem::Update, sdk+0x59c24): quien espera
    // después de actualizar espera al frame de audio siguiente, no a uno que ya pasó.
    if (evento) nnsdk::clear_system_event_if_alive(evento);
    return ResultSuccess;
}

// --- Config ---------------------------------------------------------------------------------------------------------------

void InitializeAudioRendererConfig(AudioRendererConfig* config, const AudioRendererParameter& p, void* buffer,
                                   size_t size) {
    std::memset(config, 0, sizeof *config);
    const auto base = alinear(reinterpret_cast<uintptr_t>(buffer), 16);
    const size_t hueco = base - reinterpret_cast<uintptr_t>(buffer);
    if (!buffer || p.voiceCount < 1 || p.voiceCount > 1024 || p.sinkCount < 0 || p.sinkCount > 256 ||
        p.effectCount < 0 || p.effectCount > 256 || size < hueco + bytes_config(p)) {
        printf("[audio] InitializeAudioRendererConfig: work buffer %p de %zu bytes no vale para %d voces (hacen falta %zu): "
               "el config se queda vacío\n",
               buffer, size, p.voiceCount, GetAudioRendererConfigWorkBufferSize(p));
        return;
    }
    auto* cursor = reinterpret_cast<uint8_t*>(base);
    auto* c = new (cursor) Configuracion{};
    cursor += alinear(sizeof(Configuracion), 16);
    c->voces = reinterpret_cast<InfoVoz*>(cursor);
    for (int i = 0; i < p.voiceCount; i++) new (&c->voces[i]) InfoVoz{};
    cursor += alinear(sizeof(InfoVoz) * size_t(p.voiceCount), 16);
    c->pools = reinterpret_cast<InfoPool*>(cursor);
    for (int i = 0; i < pools_de(p); i++) new (&c->pools[i]) InfoPool{};
    cursor += alinear(sizeof(InfoPool) * size_t(pools_de(p)), 16);
    c->sumideros = reinterpret_cast<InfoSumidero*>(cursor);
    for (int i = 0; i < p.sinkCount; i++) new (&c->sumideros[i]) InfoSumidero{};
    c->hz = p.sampleRate;
    c->n_voces = p.voiceCount;
    c->n_pools = pools_de(p);
    c->n_sumideros = p.sinkCount;
    c->mix_buffers = p.mixBufferCount;
    c->marca = kMarcaConfig;
    config->c = c;
    config->marca = kMarcaConfig;
}

// MixManager::Acquire (sdk+0x4331c): uno solo, y con no más buffers que los del parámetro. Volumen 1 (sdk+0x4268c).
bool AcquireFinalMix(AudioRendererConfig* config, FinalMixType* out, int buffer_count) {
    out->info = nullptr;
    Configuracion* c = config_de(config);
    if (!c) return false;
    std::lock_guard lock(g_m);
    if (c->mix_final.en_uso || buffer_count < 1 || buffer_count > c->mix_buffers) return false;
    c->mix_final = {true, buffer_count, 1.0f};
    out->info = &c->mix_final;
    return true;
}

void ReleaseFinalMix(AudioRendererConfig* config, FinalMixType* mix) {
    Configuracion* c = config_de(config);
    if (!c || mix->info != &c->mix_final) return;
    std::lock_guard lock(g_m);
    c->mix_final = {};
    mix->info = nullptr;
}

// El sink queda unido al final mix sin más (el SDK ni mira ese argumento, sdk+0x44484). `input` son los buffers del
// mix, uno por canal del dispositivo; el reproductor pasa 0, 1… (main+0x399f34) y el nombre "MainAudioOut".
Result AddDeviceSink(AudioRendererConfig* config, DeviceSinkType* out, FinalMixType*, const int8_t* input,
                     int input_count, const char* name) {
    out->info = nullptr;
    Configuracion* c = config_de(config);
    if (!c) return kSinRecursos;
    InfoSumidero* s = nullptr;
    {
        std::lock_guard lock(g_m);
        for (int i = 0; i < c->n_sumideros && !s; i++)
            if (!c->sumideros[i].en_uso) s = &c->sumideros[i];
        if (!s) return kSinRecursos;
        *s = {};
        s->en_uso = true;
        s->entradas = std::clamp(input_count, 0, kSumideroEntradas);
        for (int i = 0; i < s->entradas; i++) s->entrada[i] = input[i];
    }
    out->info = s;
    printf("[audio] device sink \"%s\" con %d entradas%s\n", name ? name : "", input_count,
           input_count == kSumideroEntradas ? " (5.1: se baja a estéreo)" : "");
    return ResultSuccess;
}

// SinkManager::Free (sdk+0x44ac4) lo deja libre y no toca el DeviceSinkType del juego.
void RemoveDeviceSink(AudioRendererConfig* config, DeviceSinkType* sink, FinalMixType*) {
    Configuracion* c = config_de(config);
    if (!c || !sink->info) return;
    std::lock_guard lock(g_m);
    if (sink->info >= c->sumideros && sink->info < c->sumideros + c->n_sumideros) sink->info->en_uso = false;
}

// --- Voces ----------------------------------------------------------------------------------------------------------------

// detail::AcquireVoiceSlot (sdk+0x46240) y VoiceInfoManager::AcquireVoiceInfo (sdk+0x47208): hace falta una voz libre
// y tantos canales libres como pida, de los voiceCount que hay para todas. La voz nace parada (PlayState_Stop,
// sdk+0x462e8), con volumen y tono 1 (sdk+0x462d4), sin destino (sdk+0x462e0) y con toda su mezcla a 0 (sdk+0x46344).
// Solo PcmInt16, que es lo que pide el reproductor (main+0x399fec); de los otros formatos del SDK (Adpcm, PcmFloat)
// no hay quien los use en este juego.
bool AcquireVoiceSlot(AudioRendererConfig* config, VoiceType* out, int sample_rate, int channel_count,
                      SampleFormat format, int, const void*, size_t) {
    out->info = nullptr;
    Configuracion* c = config_de(config);
    if (!c) return false;
    if (format != kPcmInt16 || channel_count < 1 || channel_count > kVozCanalesMax || sample_rate <= 0) {
        printf("[audio] AcquireVoiceSlot: %d Hz, %d canales, formato %d: sin soporte (solo PcmInt16 de 1 a 6 canales)\n",
               sample_rate, channel_count, static_cast<int>(format));
        return false;
    }
    InfoVoz* v = nullptr;
    {
        std::lock_guard lock(g_m);
        if (c->voces_usadas >= c->n_voces || c->n_voces - c->canales_usados < channel_count) return false;
        for (int i = 0; i < c->n_voces && !v; i++)
            if (!c->voces[i].en_uso) v = &c->voces[i];
        if (!v) return false;
        const int indice = static_cast<int>(v - c->voces);
        new (v) InfoVoz{};
        v->en_uso = true;
        v->canales = static_cast<uint8_t>(channel_count);
        v->hz = sample_rate;
        v->indice = indice;
        v->generacion = ++c->generacion;
        v->volumen = 1.0f;
        v->tono = 1.0f;
        v->estado = VoiceType::PlayState_Stop;
        v->destino = kSinDestino;
        c->voces_usadas++;
        c->canales_usados += channel_count;
    }
    out->info = v;
    printf("[audio] voz del renderer: PcmInt16, %d Hz, %d canales\n", sample_rate, channel_count);
    return true;
}

// sdk+0x46630: la voz vuelve a estar libre y el VoiceType del juego queda a null.
void ReleaseVoiceSlot(AudioRendererConfig* config, VoiceType* voice) {
    Configuracion* c = config_de(config);
    InfoVoz* v = voz_de(voice);
    voice->info = nullptr;
    if (!c || !v || v < c->voces || v >= c->voces + c->n_voces) return;
    std::lock_guard lock(g_m);
    quitar_voz(*c, *v);
    c->voces_usadas--;
    c->canales_usados -= v->canales;
    v->en_uso = false;
}

// sdk+0x46834: un float en el VoiceInfo.
void SetVoiceVolume(VoiceType* voice, float volume) {
    if (InfoVoz* v = voz_de(voice)) v->volumen.store(volume, std::memory_order_relaxed);
}

// sdk+0x4656c: el id del mix de destino (el del final mix es 0).
void SetVoiceDestination(AudioRendererConfig*, VoiceType* voice, FinalMixType* mix) {
    if (InfoVoz* v = voz_de(voice)) v->destino = mix && mix->info ? kMixFinal : kSinDestino;
}

// sdk+0x4689c: mixVolume[destino] del canal `origen` de la voz; el mix no se mira. El reproductor recorre los buffers
// del mix con origen = destino (main+0x39a1f0), también con una pista mono (dos buffers): ese origen 1 no existe.
void SetVoiceMixVolume(VoiceType* voice, FinalMixType*, float volume, int source, int destination) {
    InfoVoz* v = voz_de(voice);
    if (!v || source < 0 || source >= v->canales || destination < 0 || destination >= kMixBuffersVoz) return;
    v->mezcla[source][destination] = volume;
}

// sdk+0x46730: un byte en el VoiceInfo; el servidor lo aplica en la siguiente actualización.
void SetVoicePlayState(VoiceType* voice, VoiceType::PlayState state) {
    if (InfoVoz* v = voz_de(voice)) v->estado.store(static_cast<uint8_t>(state), std::memory_order_relaxed);
}

// El estado que pidió el juego (el VoiceInfo del cliente guarda el byte tal cual) y los frames que la voz llevaba
// tocados en el último RequestUpdate: CRI Atom (Final Fantasy Resonance) lleva con esa cuenta la posición de cada
// sonido y la sincronía de sus vídeos. Una voz recién adquirida empieza de 0.
VoiceType::PlayState GetVoicePlayState(const VoiceType* voice) {
    const InfoVoz* v = voz_de(voice);
    return v ? static_cast<VoiceType::PlayState>(v->estado.load(std::memory_order_relaxed)) : VoiceType::PlayState_Stop;
}
int64_t GetVoicePlayedSampleCount(const VoiceType* voice) {
    const InfoVoz* v = voz_de(voice);
    return v ? v->tocados : 0;
}

// sdk+0x46a50: como mucho 4 sin recoger; el buffer se apunta en el anillo y sale hacia el servidor en el siguiente
// RequestUpdate. loop sin MemberOptions = repetir entero y sin fin (sdk+0x46b14).
bool AppendWaveBuffer(VoiceType* voice, const WaveBuffer* wave) {
    InfoVoz* v = voz_de(voice);
    if (!v || !wave || v->adjuntados - v->devueltos >= kVozOndasMax) return false;
    InfoOnda& o = v->ondas[v->adjuntados % kVozOndasMax];
    o = {wave, wave->buffer, wave->size, wave->startSampleOffset, wave->endSampleOffset, wave->loop};
    v->adjuntados++;
    return true;
}

// sdk+0x46ba8: el siguiente del anillo si el servidor ya lo dio por consumido en el último RequestUpdate.
const WaveBuffer* GetReleasedWaveBuffer(VoiceType* voice) {
    InfoVoz* v = voz_de(voice);
    if (!v || v->consumidos == v->devueltos) return nullptr;
    return v->ondas[v->devueltos++ % kVozOndasMax].del_juego;
}

// --- Memory pools -----------------------------------------------------------------------------------------------------------
// En la consola son la memoria del juego que el proceso de audio puede leer; aquí toda lo es, y un WaveBuffer suena
// esté o no en un pool. Lo que el juego observa es el estado, y el reproductor espera a verlos sueltos para cerrar.

// MemoryPoolManager::Acquire (sdk+0x42f48): sin dirección o sin tamaño, o sin sitio libre, no hay pool.
bool AcquireMemoryPool(AudioRendererConfig* config, MemoryPoolType* out, void* address, size_t size) {
    out->info = nullptr;
    Configuracion* c = config_de(config);
    if (!c || !address || !size) return false;
    std::lock_guard lock(g_m);
    for (int i = 0; i < c->n_pools; i++) {
        if (c->pools[i].en_uso) continue;
        c->pools[i] = {true, kPoolNuevo, address, size};
        out->info = &c->pools[i];
        return true;
    }
    return false;
}

void ReleaseMemoryPool(AudioRendererConfig* config, MemoryPoolType* pool) {
    InfoPool* p = pool_de(pool);
    pool->info = nullptr;
    if (!config_de(config) || !p) return;
    std::lock_guard lock(g_m);
    *p = {false, kPoolLiberado, nullptr, 0};
}

// MemoryPoolInfo::Attach (sdk+0x42b30): nuevo o suelto pasa a pedirlo; si estaba pidiendo soltarse, vuelve a enganchado.
bool RequestAttachMemoryPool(MemoryPoolType* pool) {
    InfoPool* p = pool_de(pool);
    if (!p) return false;
    switch (p->estado) {
    case kPoolNuevo:
    case kPoolSuelto: p->estado = kPoolPideEnganchar; return true;
    case kPoolPideSoltar: p->estado = kPoolEnganchado; return true;
    default: return false;
    }
}

// MemoryPoolInfo::Detach (sdk+0x42c2c): enganchado pasa a pedirlo; si aún estaba pidiendo engancharse, queda suelto.
bool RequestDetachMemoryPool(MemoryPoolType* pool) {
    InfoPool* p = pool_de(pool);
    if (!p) return false;
    switch (p->estado) {
    case kPoolEnganchado: p->estado = kPoolPideSoltar; return true;
    case kPoolPideEnganchar: p->estado = kPoolSuelto; return true;
    default: return false;
    }
}

// MemoryPoolInfo::IsAttached (sdk+0x42c68, máscara 0b1101 sobre estado - 2): también mientras pide engancharse o
// soltarse. Por eso el reproductor, tras RequestDetach, sigue viéndolo enganchado hasta que un RequestUpdate lo suelta.
bool IsMemoryPoolAttached(const MemoryPoolType* pool) {
    const InfoPool* p = pool_de(pool);
    return p && (p->estado == kPoolPideSoltar || p->estado == kPoolPideEnganchar || p->estado == kPoolEnganchado);
}

void* GetMemoryPoolAddress(const MemoryPoolType* pool) {
    const InfoPool* p = pool_de(pool);
    return p ? p->direccion : nullptr;
}

// Un memcpy también en el SDK (sdk+0x429f0).
void* CopyMemoryPoolData(const MemoryPoolType*, void* destination, const void* source, size_t size) {
    return std::memcpy(destination, source, size);
}

}  // namespace nn::audio
