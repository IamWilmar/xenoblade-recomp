// Prueba fuera del dispositivo del AudioRenderer (runtime/src/nnsdk/audio_renderer.cpp + audio_voz.h): se compila el
// .cpp tal cual y se le llama con la secuencia exacta del reproductor de vídeo de Unity (main+0x399c3c preparar,
// main+0x3971f0 tocar, main+0x397b6c cerrar), con sus mismos structs y huecos. AAudio es la propia prueba, que llama a
// la fuente que el renderer registra en la salida (audio_salida.h); los SystemEvent, los de falsos/eventos.h.
//
//   c++ -std=c++20 -O1 -g -fsanitize=address,undefined -I runtime/src -I tools/tests \
//       tools/tests/audio_renderer_prueba.cpp -o /tmp/audio_renderer_prueba && /tmp/audio_renderer_prueba
//   (y otra vez con -fsanitize=thread para la parte de hilos)

#include "falsos/eventos.h"

#include "../../runtime/src/nnsdk/audio_renderer.cpp"

#include <cinttypes>
#include <cstdlib>
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

// --- La salida de audio.cpp, de mentira ---
const nnsdk::audio::FuenteExtra* g_fuente = nullptr;
int g_preparada = 0;

}  // namespace

void nnsdk::audio::preparar_salida() { g_preparada++; }
void nnsdk::audio::fijar_fuente_extra(const FuenteExtra* f) { g_fuente = f; }

namespace {

using namespace nn::audio;

// Lo que haría el callback de AAudio: `frames` frames de la mezcla del renderer sobre silencio.
void rafaga(std::vector<int16_t>& salida, size_t frames) {
    const size_t antes = salida.size();
    salida.resize(antes + frames * 2, 0);
    if (g_fuente) g_fuente->mezclar(salida.data() + antes, frames);
}

// --- El trozo del reproductor de Unity que toca nn::audio, con su mismo reparto (this+0xcb0..0xed8) ---
struct Ranura {  // 0x60 bytes: main+0x39a050 avanza de 0x60 en 0x60
    WaveBuffer onda;      // +0x00
    MemoryPoolType pool;  // +0x40
    int64_t canales;      // +0x48
    void* direccion;      // +0x50
    size_t bytes;         // +0x58
};
static_assert(sizeof(Ranura) == 0x60);

struct Reproductor {
    uint8_t evento[0x48];        // this+0x9f0: nn::os::SystemEvent
    AudioRendererHandle handle;  // this+0xcb0
    AudioRendererConfig config;  // this+0xcc0
    DeviceSinkType sink;         // this+0xd38
    FinalMixType mix;            // this+0xd40
    VoiceType voz;               // this+0xd48
    float volumen;               // this+0xd50
    Ranura ranuras[4];           // this+0xd58
    uint64_t centinela;
    // lo que en el juego son vectores
    std::vector<uint8_t> trabajo, trabajo_config;
    std::vector<Ranura*> libres;  // this+0xed8
    int hz = 0, canales = 0;
    int actualizaciones_al_cerrar = 0;
};
static_assert(offsetof(Reproductor, config) - offsetof(Reproductor, handle) == 0xcc0 - 0xcb0);
static_assert(offsetof(Reproductor, sink) - offsetof(Reproductor, handle) == 0xd38 - 0xcb0);
static_assert(offsetof(Reproductor, voz) - offsetof(Reproductor, handle) == 0xd48 - 0xcb0);
static_assert(offsetof(Reproductor, ranuras) - offsetof(Reproductor, handle) == 0xd58 - 0xcb0);

constexpr uint64_t kCentinela = 0x1122334455667788;
constexpr size_t kPool = 0x8000;  // memalign(0x1000, 0x8000), main+0x399054

nn::os::SystemEvent* evento_de(Reproductor& p) { return reinterpret_cast<nn::os::SystemEvent*>(p.evento); }

// main+0x399c3c. `canales_dispositivo`: lo que da el AudioManager de Unity (2 en estéreo).
bool preparar(Reproductor& p, int hz, int canales, int canales_dispositivo = 2) {
    p.hz = hz, p.canales = canales, p.volumen = 1.0f, p.centinela = kCentinela;
    struct {
        AudioRendererParameter parametro;  // sp+0x38, 0x38 bytes
        uint64_t d8;                       // sp+0x70
    } pila;
    pila.d8 = kCentinela;
    InitializeAudioRendererParameter(&pila.parametro);
    COMPROBAR(pila.d8 == kCentinela);
    AudioRendererParameter& par = pila.parametro;
    const int mezcla = canales_dispositivo > 1 && canales == 1 ? 2 : std::min(std::min(canales_dispositivo, 8), canales);
    par.sampleRate = hz;
    par.sampleCount = hz / 200;
    par.mixBufferCount = mezcla;
    par.subMixCount = 0;
    par.voiceCount = canales;
    par.sinkCount = 1;  // main+0x3f4edb0: {1, 0}
    par.effectCount = 0;
    par.performanceFrameCount = 0;
    if (!IsValidAudioRendererParameter(par)) return false;  // "An audio renderer parameter is not valid."
    const size_t n_trabajo = GetAudioRendererWorkBufferSize(par);
    p.trabajo.assign(n_trabajo, 0);
    const size_t n_config = GetAudioRendererConfigWorkBufferSize(par);
    COMPROBAR(n_config % 0x1000 == 0 && n_config >= 0x1000);
    p.trabajo_config.assign(n_config + 16, 0xEE);  // 16 de más con otro valor: no se puede pasar del tamaño pedido
    int8_t indices[8];
    for (int i = 0; i < mezcla; i++) indices[i] = static_cast<int8_t>(i);
    OpenAudioRenderer(&p.handle, evento_de(p), par, p.trabajo.data(), n_trabajo);
    InitializeAudioRendererConfig(&p.config, par, p.trabajo_config.data(), n_config);
    for (size_t i = n_config; i < n_config + 16; i++) COMPROBAR(p.trabajo_config[i] == 0xEE);
    AcquireFinalMix(&p.config, &p.mix, mezcla);
    AddDeviceSink(&p.config, &p.sink, &p.mix, indices, mezcla, "MainAudioOut");
    RequestUpdateAudioRenderer(p.handle, &p.config);
    if (StartAudioRenderer(p.handle).value != 0) return false;  // "Failed to start an audio renderer."
    if (!AcquireVoiceSlot(&p.config, &p.voz, hz, canales, SampleFormat{2}, 0, nullptr, 0))
        return false;  // "Failed to acquire a voice slot."
    SetVoiceVolume(&p.voz, p.volumen);
    SetVoiceDestination(&p.config, &p.voz, &p.mix);
    p.libres.clear();
    for (Ranura& r : p.ranuras) {
        r.canales = canales;
        r.direccion = aligned_alloc(0x1000, kPool);
        r.bytes = kPool;
        std::memset(r.direccion, 0, r.bytes);
        COMPROBAR(AcquireMemoryPool(&p.config, &r.pool, r.direccion, r.bytes));
        COMPROBAR(RequestAttachMemoryPool(&r.pool));
        r.onda.buffer = r.direccion;
        r.onda.startSampleOffset = 0;
        r.onda.loop = false, r.onda.isEndOfStream = false;
        r.onda.size = r.bytes;
        r.onda.endSampleOffset = static_cast<int32_t>(r.bytes >> 1);
        r.onda.pContext = nullptr, r.onda.contextSize = 0;
        p.libres.push_back(&r);
    }
    SetVoicePlayState(&p.voz, VoiceType::PlayState_Play);
    for (int i = 0; i < mezcla; i++) SetVoiceMixVolume(&p.voz, &p.mix, 0.5f, i, indices[i]);
    RequestUpdateAudioRenderer(p.handle, &p.config);
    COMPROBAR(p.centinela == kCentinela);
    return true;
}

// Lo que hace main+0x39741c..0x397664 con un buffer de PCM decodificado: recoge los liberados y, si hay ranura y el
// buffer trae algo (main+0x397504), lo encola. Devuelve si lo ha encolado.
bool tocar(Reproductor& p, const int16_t* pcm, size_t bytes) {
    while (const WaveBuffer* w = GetReleasedWaveBuffer(&p.voz)) {
        Ranura* de = nullptr;
        for (Ranura& r : p.ranuras)
            if (w == &r.onda) de = &r;
        COMPROBAR(de != nullptr);  // el juego solo reconoce los suyos, por dirección
        if (de) p.libres.push_back(de);
    }
    if (bytes < 1 || p.libres.empty()) {
        RequestUpdateAudioRenderer(p.handle, &p.config);
        return false;
    }
    Ranura* r = p.libres.front();
    p.libres.erase(p.libres.begin());
    const size_t n = std::min(r->bytes, bytes);
    CopyMemoryPoolData(&r->pool, r->direccion, pcm, n);
    r->onda.endSampleOffset = static_cast<int32_t>((n >> 1) / static_cast<size_t>(r->canales));
    COMPROBAR(AppendWaveBuffer(&p.voz, &r->onda));
    RequestUpdateAudioRenderer(p.handle, &p.config);
    return true;
}

// main+0x397b6c.
void cerrar(Reproductor& p) {
    for (Ranura& r : p.ranuras) RequestDetachMemoryPool(&r.pool);
    SetVoicePlayState(&p.voz, VoiceType::PlayState_Stop);
    auto alguno = [&] {
        bool a = false;
        for (Ranura& r : p.ranuras) a |= IsMemoryPoolAttached(&r.pool);
        return a;
    };
    p.actualizaciones_al_cerrar = 0;
    while (alguno()) {
        if (!falsos::esperar(p.evento, 2000)) {  // WaitSystemEvent: en el juego, sin plazo
            COMPROBAR(!"el evento del renderer no llega al cerrar");
            break;
        }
        RequestUpdateAudioRenderer(p.handle, &p.config);
        p.actualizaciones_al_cerrar++;
    }
    falsos::destruir(p.evento);  // DestroySystemEvent, con el renderer aún en marcha
    for (Ranura& r : p.ranuras) ReleaseMemoryPool(&p.config, &r.pool);
    p.libres.clear();
    ReleaseVoiceSlot(&p.config, &p.voz);
    ReleaseFinalMix(&p.config, &p.mix);
    if (GetAudioRendererState(p.handle) != 1) StopAudioRenderer(p.handle);
    RemoveDeviceSink(&p.config, &p.sink, &p.mix);
    CloseAudioRenderer(p.handle);
    for (Ranura& r : p.ranuras) {
        std::memset(r.direccion, 0xDD, r.bytes);  // el juego la libera: nadie puede seguir leyéndola
        free(r.direccion);
        r.direccion = nullptr;
    }
    COMPROBAR(p.centinela == kCentinela);
}

// Valores pares (la mezcla del reproductor es 0.5: sale exacto).
int16_t muestra(size_t frame, int canal) {
    return static_cast<int16_t>(canal == 0 ? 2 * static_cast<int>(frame % 9001) - 9000 : 6000 - 4 * static_cast<int>(frame % 3001));
}

void prueba_parametro() {
    struct {
        AudioRendererParameter p;
        uint64_t detras;
    } pila;
    pila.detras = kCentinela;
    std::memset(&pila.p, 0xAA, sizeof pila.p);
    InitializeAudioRendererParameter(&pila.p);
    COMPROBAR(pila.detras == kCentinela);
    // Lo que deja el SDK (sdk+0x3af24..0x3af48), byte a byte.
    const uint32_t esperado[14] = {48000, 240, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0x38564552};
    COMPROBAR(std::memcmp(&pila.p, esperado, sizeof esperado) == 0);
    COMPROBAR(IsValidAudioRendererParameter(pila.p));
    AudioRendererParameter p = pila.p;
    p.sampleRate = 44100, p.sampleCount = 220;  // un vídeo con audio a 44,1 kHz: sin audio, como en la consola
    COMPROBAR(!IsValidAudioRendererParameter(p));
    p.sampleRate = 32000, p.sampleCount = 160;
    COMPROBAR(IsValidAudioRendererParameter(p));
    p.sampleCount = 240;
    COMPROBAR(!IsValidAudioRendererParameter(p));
    p = pila.p, p._magic = 0;
    COMPROBAR(!IsValidAudioRendererParameter(p));
    p = pila.p, p.voiceCount = 1025;
    COMPROBAR(!IsValidAudioRendererParameter(p));
    p = pila.p, p.executionMode = 1;
    COMPROBAR(!IsValidAudioRendererParameter(p));
    p = pila.p, p.mixBufferCount = 0;
    COMPROBAR(!IsValidAudioRendererParameter(p));
}

// Un vídeo estéreo a 48 kHz de principio a fin: lo que sale es lo que entra por 0.5, sin perder ni repetir un frame.
void prueba_reproductor_48k() {
    auto p = std::make_unique<Reproductor>();
    COMPROBAR(preparar(*p, 48000, 2));
    COMPROBAR(g_preparada > 0 && g_fuente != nullptr);
    COMPROBAR(falsos::existe(p->evento));
    COMPROBAR(GetAudioRendererState(p->handle) == 0);
    for (Ranura& r : p->ranuras) COMPROBAR(IsMemoryPoolAttached(&r.pool));

    constexpr size_t kFramesBuffer = 1024, kBuffers = 60;  // lo que da el decodificador de AAC
    std::vector<int16_t> salida, pcm(kFramesBuffer * 2);
    rafaga(salida, 96);  // antes del primer buffer: silencio
    for (int16_t s : salida) COMPROBAR(s == 0);
    salida.clear();
    size_t encolados = 0, max_en_poder = 0;
    while (encolados < kBuffers) {
        for (size_t f = 0; f < kFramesBuffer; f++) {
            pcm[2 * f] = muestra(encolados * kFramesBuffer + f, 0);
            pcm[2 * f + 1] = muestra(encolados * kFramesBuffer + f, 1);
        }
        if (tocar(*p, pcm.data(), pcm.size() * sizeof(int16_t))) encolados++;
        max_en_poder = std::max(max_en_poder, 4 - p->libres.size());
        rafaga(salida, 96);
    }
    COMPROBAR(max_en_poder == 4);  // el reproductor llena sus cuatro ranuras y espera
    for (int i = 0; i < 2000 && p->libres.size() < 4; i++) {  // lo que queda en cola
        tocar(*p, pcm.data(), 0);  // sin datos: solo recoge y actualiza (main+0x39751c)
        rafaga(salida, 96);
    }
    COMPROBAR(p->libres.size() == 4);
    rafaga(salida, 96);
    const size_t total = kBuffers * kFramesBuffer;
    COMPROBAR(salida.size() / 2 >= total + 2);
    size_t malos = 0;
    for (size_t k = 0; k < salida.size() / 2; k++) {
        // dos frames de retraso: la interpolación de 4 puntos
        const int izq = k >= 2 && k - 2 < total ? muestra(k - 2, 0) / 2 : 0;
        const int der = k >= 2 && k - 2 < total ? muestra(k - 2, 1) / 2 : 0;
        if (salida[2 * k] != izq || salida[2 * k + 1] != der) malos++;
    }
    COMPROBAR(malos == 0);
    if (malos) std::fprintf(stderr, "  %zu frames distintos de %zu\n", malos, salida.size() / 2);

    // El volumen del vídeo (main+0x39aa48): la mitad, y llega con una rampa de 5 ms, sin salto.
    p->volumen = 0.5f;
    SetVoiceVolume(&p->voz, p->volumen);
    for (auto& s : pcm) s = 8000;
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * sizeof(int16_t)));
    salida.clear();
    rafaga(salida, 600);
    int salto = 0;
    for (size_t k = 3; k < 600; k++) salto = std::max(salto, std::abs(salida[2 * k] - salida[2 * k - 2]));
    COMPROBAR(salida[2 * 2] > 3950 && salida[2 * 2] <= 4000);  // empieza en el volumen de antes (8000 * 1.0 * 0.5)
    COMPROBAR(salida[2 * 400] == 2000);                        // y acaba en el nuevo (8000 * 0.5 * 0.5)
    COMPROBAR(salto > 0 && salto <= 2000 / 240 + 2);           // bajando poco a poco, en 240 frames
    COMPROBAR(salida[2 * 239] > 2000 && salida[2 * 245] == 2000);
    cerrar(*p);
    // En la primera actualización los pools con un buffer en cola siguen en uso (la voz se para en esa misma); en la
    // segunda ya no. Con la cola vacía, una.
    COMPROBAR(p->actualizaciones_al_cerrar >= 1 && p->actualizaciones_al_cerrar <= 2);
    COMPROBAR(p->voz.info == nullptr && p->mix.info == nullptr);
    COMPROBAR(!falsos::existe(p->evento));  // nadie lo vuelve a crear al señalarlo
    salida.clear();
    rafaga(salida, 480);  // el renderer ya no existe: silencio, y sin leer la memoria que el juego ha liberado
    for (int16_t s : salida) COMPROBAR(s == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    COMPROBAR(!falsos::existe(p->evento));
}

// El evento del renderer: cada 5 ms con el renderer en marcha, y nunca sobre un evento que el juego ha destruido.
void prueba_evento() {
    auto p = std::make_unique<Reproductor>();
    COMPROBAR(preparar(*p, 48000, 2));
    const long antes = falsos::senales(p->evento);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const long en_200ms = falsos::senales(p->evento) - antes;
    COMPROBAR(en_200ms >= 25 && en_200ms <= 45);  // 40 a 5 ms; el planificador del Mac puede retrasar alguno
    if (en_200ms < 25 || en_200ms > 45) std::fprintf(stderr, "  %ld señales en 200 ms\n", en_200ms);
    // El hilo del reproductor: TimedWaitSystemEvent(100 ms) (main+0x39733c). AutoClear: una espera por señal.
    COMPROBAR(falsos::esperar(p->evento, 100));
    // Una actualización limpia el evento (sdk+0x59c24): la espera de después es la del frame siguiente.
    std::this_thread::sleep_for(std::chrono::milliseconds(12));
    const auto antes_de_actualizar = std::chrono::steady_clock::now();
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(falsos::esperar(p->evento, 100));
    COMPROBAR(std::chrono::steady_clock::now() - antes_de_actualizar < std::chrono::milliseconds(60));
    StopAudioRenderer(p->handle);
    COMPROBAR(GetAudioRendererState(p->handle) == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    falsos::esperar(p->evento, 0);  // la que quedara
    const long parado = falsos::senales(p->evento);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    COMPROBAR(falsos::senales(p->evento) == parado);  // parado no se señala (sdk+0x5a150)
    COMPROBAR(StartAudioRenderer(p->handle).value == 0);
    COMPROBAR(falsos::esperar(p->evento, 100));
    cerrar(*p);
    COMPROBAR(falsos::g_creados_al_senalar == 0);
}

// Una voz a otra frecuencia que la salida: un seno de 1 kHz sale como un seno de 1 kHz.
void prueba_conversion(int hz, double tolerancia) {
    VozMezcla v;
    v.configurar(1, double(hz) / kSalidaHz);
    float g[kVozCanalesMax][2] = {};
    g[0][0] = 1, g[0][1] = 0.25f;
    v.fijar_ganancias(g, false);
    const double w = 2 * M_PI * 1000.0;
    std::vector<int16_t> pcm(static_cast<size_t>(hz));  // un segundo
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = static_cast<int16_t>(std::lrint(10000 * std::sin(w * i / hz)));
    // en cuatro buffers seguidos, para pasar por las costuras
    for (int b = 0; b < 4; b++) {
        OndaVoz o;
        o.datos = pcm.data();
        o.inicio = static_cast<int32_t>(pcm.size() * b / 4);
        o.fin = static_cast<int32_t>(pcm.size() * (b + 1) / 4);
        COMPROBAR(v.encolar(o));
    }
    OndaVoz quinto;
    quinto.datos = pcm.data(), quinto.fin = 10;
    COMPROBAR(!v.encolar(quinto));  // como mucho 4
    float depop[2] = {};
    v.fijar_estado(VozMezcla::kSonando, depop);
    std::vector<float> salida(static_cast<size_t>(kSalidaHz) * 2, 0.0f);
    v.mezclar(salida.data(), kSalidaHz * 9 / 10);
    COMPROBAR(v.consumidos() == 3);  // a los 0.9 s ha pasado de tres buffers y está en el cuarto
    v.mezclar(salida.data() + size_t(kSalidaHz) * 9 / 10 * 2, kSalidaHz / 10);
    COMPROBAR(v.consumidos() == 4);
    // El retraso son 2 frames de la voz; se compara lejos de los extremos.
    const double retraso = 2.0 / hz;
    double error = 0, canal_derecho = 0;
    for (int k = 480; k < kSalidaHz - 960; k++) {
        const double ideal = 10000 * std::sin(w * (double(k) / kSalidaHz - retraso));
        error = std::max(error, std::fabs(salida[2 * k] - ideal));
        canal_derecho = std::max(canal_derecho, std::fabs(salida[2 * k + 1] - salida[2 * k] * 0.25));
    }
    COMPROBAR(error <= tolerancia);
    COMPROBAR(canal_derecho < 0.01);
    if (error > tolerancia) std::fprintf(stderr, "  %d Hz: error máximo %.1f de 10000\n", hz, error);
    std::printf("  conversión %d -> %d Hz: error máximo %.2f de 10000 (%.1f dB)\n", hz, kSalidaHz, error,
                20 * std::log10(std::max(error, 1e-9) / 10000));
}

// Estados de la voz y de los memory pools, como el servidor del SDK.
void prueba_estados() {
    auto p = std::make_unique<Reproductor>();
    COMPROBAR(preparar(*p, 48000, 2));
    std::vector<int16_t> pcm(1024 * 2, 5000), salida;
    // Los pasos de MemoryPoolInfo::Attach/Detach/IsAttached (sdk+0x42b30, +0x42c2c, +0x42c68).
    MemoryPoolType suelto{};
    COMPROBAR(!IsMemoryPoolAttached(&suelto) && !RequestAttachMemoryPool(&suelto) && !RequestDetachMemoryPool(&suelto));
    alignas(4096) static uint8_t memoria[4096];
    MemoryPoolType pool{};
    COMPROBAR(!AcquireMemoryPool(&p->config, &pool, nullptr, 4096) && !AcquireMemoryPool(&p->config, &pool, memoria, 0));
    COMPROBAR(AcquireMemoryPool(&p->config, &pool, memoria, sizeof memoria));
    COMPROBAR(!IsMemoryPoolAttached(&pool));          // nuevo
    COMPROBAR(!RequestDetachMemoryPool(&pool));       // nuevo no se puede soltar
    COMPROBAR(RequestAttachMemoryPool(&pool));
    COMPROBAR(IsMemoryPoolAttached(&pool));           // pidiéndolo ya cuenta como enganchado
    COMPROBAR(!RequestAttachMemoryPool(&pool));
    COMPROBAR(RequestDetachMemoryPool(&pool));        // se arrepiente antes de la actualización: suelto
    COMPROBAR(!IsMemoryPoolAttached(&pool));
    COMPROBAR(RequestAttachMemoryPool(&pool));
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(IsMemoryPoolAttached(&pool));           // enganchado
    COMPROBAR(RequestDetachMemoryPool(&pool));
    COMPROBAR(IsMemoryPoolAttached(&pool));           // hasta que una actualización lo suelte
    COMPROBAR(RequestAttachMemoryPool(&pool));        // se arrepiente: enganchado sin más
    COMPROBAR(RequestDetachMemoryPool(&pool));
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(!IsMemoryPoolAttached(&pool));          // nadie lo usaba: suelto
    ReleaseMemoryPool(&p->config, &pool);
    COMPROBAR(pool.info == nullptr);
    // 8 pools para 2 voces (effectCount + 4 * voiceCount): 4 del reproductor, quedan 4.
    MemoryPoolType mas[5]{};
    for (int i = 0; i < 4; i++) COMPROBAR(AcquireMemoryPool(&p->config, &mas[i], memoria, sizeof memoria));
    COMPROBAR(!AcquireMemoryPool(&p->config, &mas[4], memoria, sizeof memoria));
    for (int i = 0; i < 4; i++) ReleaseMemoryPool(&p->config, &mas[i]);

    // Un pool con un buffer sonando no se suelta; al parar la voz, sí, y los buffers vuelven todos.
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    COMPROBAR(!tocar(*p, pcm.data(), pcm.size() * 2));  // sin ranura
    COMPROBAR(!AppendWaveBuffer(&p->voz, &p->ranuras[0].onda));  // y el SDK tampoco deja un quinto (sdk+0x46a58)
    rafaga(salida, 96);
    COMPROBAR(salida[2 * 50] == 2500);
    COMPROBAR(RequestDetachMemoryPool(&p->ranuras[0].pool));
    RequestUpdateAudioRenderer(p->handle, &p->config);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(IsMemoryPoolAttached(&p->ranuras[0].pool));  // sigue sonando desde ahí
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
    COMPROBAR(RequestAttachMemoryPool(&p->ranuras[0].pool));  // se arrepiente
    // Pausa: no avanza ni suelta nada; lo que sonaba cae sin salto (depop).
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Pause);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    salida.clear();
    rafaga(salida, 480);
    COMPROBAR(salida[0] > 2000 && salida[0] < 2500 && salida[2 * 479] == 0);
    for (size_t k = 1; k < 480; k++) COMPROBAR(salida[2 * k] <= salida[2 * k - 2]);
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Play);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    salida.clear();
    rafaga(salida, 96);
    COMPROBAR(salida[2 * 50] == 2500);  // sigue donde estaba
    // Stop: todo lo encolado vuelve, por orden, y el pool se suelta en la siguiente actualización.
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Stop);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    for (int i = 0; i < 4; i++) COMPROBAR(GetReleasedWaveBuffer(&p->voz) == &p->ranuras[i].onda);
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
    COMPROBAR(RequestDetachMemoryPool(&p->ranuras[0].pool));
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(!IsMemoryPoolAttached(&p->ranuras[0].pool));
    COMPROBAR(RequestAttachMemoryPool(&p->ranuras[0].pool));
    for (Ranura& r : p->ranuras) p->libres.push_back(&r);

    // Parada, los buffers esperan; al volver a Play suenan desde el principio.
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    salida.clear();
    rafaga(salida, 96);
    for (int16_t s : salida) COMPROBAR(s == 0 || std::abs(s) < 2500);  // solo la cola del depop
    rafaga(salida, 480);                                               // que se acaba en unos ms
    COMPROBAR(salida[salida.size() - 2] == 0);
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Play);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    salida.clear();
    rafaga(salida, 96);
    COMPROBAR(salida[2 * 50] == 2500);

    // El reloj de reserva (AAudio no llama): pasa el tiempo y los buffers vuelven sin sonar.
    g_fuente->descartar(2000);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) != nullptr);
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
    p->libres.clear();
    for (Ranura& r : p->ranuras) p->libres.push_back(&r);

    // En pausa la voz no retiene su pool (el servidor solo lo hace mientras lee de él): se suelta, y como esa memoria
    // ya no se puede leer, la cola entera vuelve al juego.
    Ranura* primera = p->libres.front();
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Pause);
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
    COMPROBAR(RequestDetachMemoryPool(&primera->pool));
    RequestUpdateAudioRenderer(p->handle, &p->config);
    COMPROBAR(!IsMemoryPoolAttached(&primera->pool));
    int devueltos = 0;
    while (GetReleasedWaveBuffer(&p->voz)) devueltos++;
    COMPROBAR(devueltos == 2);
    p->libres.clear();
    for (Ranura& r : p->ranuras) p->libres.push_back(&r);
    COMPROBAR(RequestAttachMemoryPool(&primera->pool));
    SetVoicePlayState(&p->voz, VoiceType::PlayState_Play);
    RequestUpdateAudioRenderer(p->handle, &p->config);

    // Límites del config: una voz estéreo gasta los dos canales de voiceCount = 2; un solo final mix.
    VoiceType otra{};
    COMPROBAR(!AcquireVoiceSlot(&p->config, &otra, 48000, 1, SampleFormat{2}, 0, nullptr, 0));
    FinalMixType otro_mix{};
    COMPROBAR(!AcquireFinalMix(&p->config, &otro_mix, 2));
    DeviceSinkType otro_sink{};
    const int8_t entradas[2] = {0, 1};
    COMPROBAR(AddDeviceSink(&p->config, &otro_sink, &p->mix, entradas, 2, "MainAudioOut").value == 0xa99);
    // Con structs a cero (lo que queda tras Release) no pasa nada.
    VoiceType nula{};
    SetVoiceVolume(&nula, 1), SetVoicePlayState(&nula, VoiceType::PlayState_Play);
    SetVoiceMixVolume(&nula, &p->mix, 1, 0, 0), SetVoiceDestination(&p->config, &nula, &p->mix);
    COMPROBAR(!AppendWaveBuffer(&nula, &p->ranuras[0].onda) && GetReleasedWaveBuffer(&nula) == nullptr);
    ReleaseVoiceSlot(&p->config, &nula);
    AudioRendererConfig vacio{};
    COMPROBAR(!AcquireVoiceSlot(&vacio, &otra, 48000, 2, SampleFormat{2}, 0, nullptr, 0));
    COMPROBAR(RequestUpdateAudioRenderer(p->handle, &vacio).value == 0);
    COMPROBAR(RequestUpdateAudioRenderer(AudioRendererHandle{}, &p->config).value == 0);
    // Dos renderers por proceso, como la consola (sdk+0x4bf58): el tercero da 0xa99.
    {
        AudioRendererParameter par;
        InitializeAudioRendererParameter(&par);
        AudioRendererHandle segundo{}, tercero{};
        COMPROBAR(OpenAudioRenderer(&segundo, nullptr, par, nullptr, 0).value == 0);
        COMPROBAR(OpenAudioRenderer(&tercero, nullptr, par, nullptr, 0).value == 0xa99);
        COMPROBAR(GetAudioRendererState(segundo) == 1 && GetAudioRendererState(tercero) == 1);
        CloseAudioRenderer(tercero);  // nada que cerrar
        CloseAudioRenderer(segundo);
        CloseAudioRenderer(segundo);  // dos veces: la segunda no hace nada
        COMPROBAR(GetAudioRendererState(p->handle) == 0);
    }
    // El renderer se cierra con buffers sin tocar y el mismo config sigue con otro: los buffers llegan al nuevo y
    // vuelven todos, por orden.
    {
        while (GetReleasedWaveBuffer(&p->voz)) {
        }
        p->libres.clear();
        for (Ranura& r : p->ranuras) p->libres.push_back(&r);
        SetVoicePlayState(&p->voz, VoiceType::PlayState_Play);
        COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
        COMPROBAR(tocar(*p, pcm.data(), pcm.size() * 2));
        rafaga(salida, 1024 + 96);  // el primero entero
        RequestUpdateAudioRenderer(p->handle, &p->config);
        COMPROBAR(GetReleasedWaveBuffer(&p->voz) == &p->ranuras[0].onda);
        CloseAudioRenderer(p->handle);
        AudioRendererParameter par;
        InitializeAudioRendererParameter(&par);
        par.mixBufferCount = 2, par.voiceCount = 2;
        falsos::destruir(p->evento);
        COMPROBAR(OpenAudioRenderer(&p->handle, evento_de(*p), par, nullptr, 0).value == 0);
        COMPROBAR(StartAudioRenderer(p->handle).value == 0);
        RequestUpdateAudioRenderer(p->handle, &p->config);
        COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
        salida.clear();
        rafaga(salida, 1024 + 96);
        COMPROBAR(salida[2 * 500] == 2500);  // el segundo suena en el renderer nuevo
        RequestUpdateAudioRenderer(p->handle, &p->config);
        COMPROBAR(GetReleasedWaveBuffer(&p->voz) == &p->ranuras[1].onda);
        COMPROBAR(GetReleasedWaveBuffer(&p->voz) == nullptr);
        p->libres.clear();
        for (Ranura& r : p->ranuras) p->libres.push_back(&r);
    }
    // Otros formatos: no hay voz (el reproductor diría "Failed to acquire a voice slot").
    ReleaseVoiceSlot(&p->config, &p->voz);
    COMPROBAR(!AcquireVoiceSlot(&p->config, &p->voz, 48000, 2, SampleFormat{6}, 0, nullptr, 0));
    COMPROBAR(AcquireVoiceSlot(&p->config, &p->voz, 48000, 2, SampleFormat{2}, 0, nullptr, 0));
    cerrar(*p);
}

// El hilo del reproductor, el de AAudio y el del juego cambiando el volumen, a la vez (para -fsanitize=thread).
void prueba_hilos() {
    auto p = std::make_unique<Reproductor>();
    COMPROBAR(preparar(*p, 48000, 2));
    std::atomic<bool> fin{false};
    std::atomic<long> encolados{0}, frames{0};
    std::thread aaudio([&] {
        std::vector<int16_t> salida;
        while (!fin) {
            salida.clear();
            rafaga(salida, 96);
            frames += 96;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
    std::thread volumen([&] {
        float v = 0.1f;
        while (!fin) {
            SetVoiceVolume(&p->voz, v);
            v = v > 0.9f ? 0.1f : v + 0.1f;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::thread reproductor([&] {
        std::vector<int16_t> pcm(1024 * 2, 1000);
        while (!fin) {
            if (!falsos::esperar(p->evento, 100)) continue;
            if (tocar(*p, pcm.data(), pcm.size() * 2)) encolados++;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    fin = true;
    aaudio.join(), volumen.join(), reproductor.join();
    COMPROBAR(encolados > 4);
    // Lo encolado va al paso de lo que toca "AAudio": ni se acumula ni se pierde (como mucho 4 buffers por delante).
    COMPROBAR(encolados * 1024 <= frames + 5 * 1024);
    std::printf("  hilos: %ld buffers encolados, %ld frames tocados\n", encolados.load(), frames.load());
    cerrar(*p);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    prueba_parametro();
    prueba_reproductor_48k();
    prueba_evento();
    prueba_conversion(32000, 60);   // 1 kHz a 32 kHz: 32 muestras por ciclo
    prueba_conversion(44100, 30);
    prueba_conversion(48000, 0.51);  // misma frecuencia: la muestra tal cual (solo el redondeo del seno de referencia)
    prueba_estados();
    prueba_hilos();
    if (g_fallos) {
        std::printf("audio_renderer_prueba: %d FALLOS\n", g_fallos);
        return 1;
    }
    std::printf("audio_renderer_prueba: todo bien\n");
    return 0;
}
