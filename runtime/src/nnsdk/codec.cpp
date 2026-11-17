// nn::codec::HardwareOpusDecoder. El juego solo importa el constructor: el resto va por vtable, y
// el orden de los métodos virtuales es el del vtable que exporta el sdk (tools/vtable.py), así el
// compilador genera uno idéntico. El juego reserva 64 bytes para el objeto (CAkOpusDecoder::Init, main+0xaef460).
//
// Con él decodifica Wwise todo lo que es Opus NX (códec 0x3039 en los .wem): las voces (en.pck, jp.pck), la música
// (BGM.pck), los streams de se.pck y parte de los efectos de SE.bnk. El main no trae otro decodificador Opus: los
// CAkSrc{File,Bank}OpusNX pasan los paquetes a CAkOpusDecodingMgr, y su hilo 'AK::OpusDecodingThread' llama aquí
// (CAkOpusDecoder::PerformDecoding, main+0xaf0690: Initialize por vtable+0x20, DecodeInterleaved por vtable+0x48).
// Cada paquete trae la cabecera de Nintendo (tamaño y rango final, big-endian) y luego el paquete Opus normal: se
// decodifica con libopus (third_party/opus) y su estado va en el work buffer que reserva el juego.
// Antes daba silencio de la duración correcta: el reloj de Wwise avanzaba, pero sin voces ni música (de lo de Wwise
// solo podían sonar los efectos en PCM de los bancos). Interruptor opus_silencio: como antes.

#include <opus.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/types.h"

namespace {

uint32_t be32(const uint8_t* p) { return uint32_t{p[0]} << 24 | uint32_t{p[1]} << 16 | uint32_t{p[2]} << 8 | p[3]; }

// Muestras por canal de un paquete Opus (RFC 6716, 3.1): duración del frame por la config del TOC
// y número de frames por el código del TOC. Para el silencio de reserva (interruptor o paquete que libopus rechaza).
int opus_samples(const uint8_t* packet, size_t size, int sample_rate) {
    if (size == 0) return 0;
    int config = packet[0] >> 3;
    int tenths_ms;  // duración de un frame en décimas de ms
    if (config < 12)
        tenths_ms = (const int[]){100, 200, 400, 600}[config & 3];
    else if (config < 16)
        tenths_ms = (config & 1) ? 200 : 100;
    else
        tenths_ms = (const int[]){25, 50, 100, 200}[config & 3];
    int code = packet[0] & 3;
    int frames = code == 0 ? 1 : code < 3 ? 2 : (size > 1 ? packet[1] & 0x3F : 0);
    return frames * tenths_ms * sample_rate / 10000;
}

// DEPURACIÓN (registro): informe cada 5 s. Solo decodifica el hilo 'AK::OpusDecodingThread'; atómicos por si acaso.
constexpr int64_t kInformeNs = 5'000'000'000;
std::atomic<uint64_t> g_paquetes{0}, g_muestras{0}, g_errores{0}, g_rango_distinto{0}, g_inicializados{0},
    g_silenciados{0};
std::atomic<int32_t> g_pico{0};
std::atomic<int> g_ultimo_error{0};
std::atomic<int64_t> g_siguiente_ns{0};

int64_t ahora_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

void anotar_pico(const int16_t* pcm, size_t muestras) {
    int32_t pico = 0;
    for (size_t i = 0; i < muestras; i++) pico = std::max<int32_t>(pico, std::abs(int32_t{pcm[i]}));
    int32_t antes = g_pico.load(std::memory_order_relaxed);
    while (pico > antes && !g_pico.compare_exchange_weak(antes, pico, std::memory_order_relaxed)) {
    }
}

void informe_si_toca() {
    const int64_t t = ahora_ns();
    int64_t sig = g_siguiente_ns.load(std::memory_order_relaxed);
    if (!sig) {
        g_siguiente_ns.compare_exchange_strong(sig, t + kInformeNs);
        return;
    }
    if (t < sig || !g_siguiente_ns.compare_exchange_strong(sig, t + kInformeNs)) return;
    const double s = (t - sig + kInformeNs) / 1e9;
    const uint64_t paquetes = g_paquetes.exchange(0), muestras = g_muestras.exchange(0), errores = g_errores.exchange(0),
                   rango = g_rango_distinto.exchange(0), inicializados = g_inicializados.exchange(0),
                   silenciados = g_silenciados.exchange(0);
    const int32_t pico = g_pico.exchange(0);
    if (!NVN_FLAG("registro")) return;
    char nivel[24] = "silencio";
    if (pico) snprintf(nivel, sizeof nivel, "%.0f dBFS", 20.0 * std::log10(pico / 32768.0));
    // "muestras" son por canal y suman todas las voces: 10 s en 5 s = dos streams a la vez.
    printf("[opus] %.1f s: %llu paquetes (%.1f s de audio), nivel máx. %s; %llu inicializaciones, %llu errores de "
           "libopus (último %s), %llu con rango final distinto, %llu en silencio por el interruptor\n",
           s, static_cast<unsigned long long>(paquetes), muestras / 48000.0, nivel,
           static_cast<unsigned long long>(inicializados), static_cast<unsigned long long>(errores),
           errores ? opus_strerror(g_ultimo_error.load()) : "ninguno", static_cast<unsigned long long>(rango),
           static_cast<unsigned long long>(silenciados));
}

}  // namespace

namespace nn::codec {

class HardwareOpusDecoder {
public:
    explicit HardwareOpusDecoder(int);
    virtual ~HardwareOpusDecoder();
    virtual size_t GetWorkBufferSize(int sample_rate, int channels) const;
    virtual bool IsInitialized() const;
    virtual Result Initialize(int sample_rate, int channels, void* work, size_t work_size);
    virtual Result Finalize();
    virtual int GetSampleRate() const;
    virtual int GetChannelCount() const;
    virtual Result Reset();
    virtual Result DecodeInterleaved(size_t* consumed, int* samples, int16_t* out, size_t out_size, const void* in,
                                     size_t in_size);
    // La memoria del objeto la reserva el juego: el destructor con delete no la libera.
    static void operator delete(void*) {}

private:
    void liberar();

    int sample_rate_ = 0, channels_ = 0;
    bool initialized_ = false;
    bool propio_ = false;  // el estado de libopus va en memoria nuestra (work buffer del juego demasiado pequeño)
    OpusDecoder* dec_ = nullptr;  // nullptr: libopus no aceptó el formato y se decodifica silencio
};
static_assert(sizeof(HardwareOpusDecoder) <= 64, "el juego reserva 64 bytes para el objeto");

HardwareOpusDecoder::HardwareOpusDecoder(int) {}
HardwareOpusDecoder::~HardwareOpusDecoder() { liberar(); }

void HardwareOpusDecoder::liberar() {
    if (propio_) free(dec_);
    dec_ = nullptr;
    propio_ = false;
}

// Wwise reserva con esto el work buffer de cada decodificador (64 KB por decodificador en la Flip, visto en memoria).
size_t HardwareOpusDecoder::GetWorkBufferSize(int, int channels) const {
    return std::max<size_t>(64 * 1024, static_cast<size_t>(opus_decoder_get_size(channels == 1 ? 1 : 2)));
}

bool HardwareOpusDecoder::IsInitialized() const { return initialized_; }

Result HardwareOpusDecoder::Initialize(int sample_rate, int channels, void* work, size_t work_size) {
    liberar();
    sample_rate_ = sample_rate;
    channels_ = channels;
    initialized_ = true;
    g_inicializados++;
    if (channels < 1 || channels > 2) {  // el HardwareOpusDecoder solo admite mono y estéreo
        printf("[opus] Initialize con %d canales: silencio\n", channels);
        return ResultSuccess;
    }
    const int tam = opus_decoder_get_size(channels);
    void* mem = work;
    if (!work || work_size < static_cast<size_t>(tam)) {
        mem = malloc(static_cast<size_t>(tam));
        propio_ = mem != nullptr;
    }
    dec_ = static_cast<OpusDecoder*>(mem);
    const int err = dec_ ? opus_decoder_init(dec_, sample_rate, channels) : OPUS_ALLOC_FAIL;
    static std::atomic<bool> dicho{false};
    if (err != OPUS_OK) {
        printf("[opus] opus_decoder_init(%d Hz, %d canales): %s; ese stream sonará en silencio\n", sample_rate, channels,
               opus_strerror(err));
        liberar();
    } else if (!dicho.exchange(true)) {
        printf("[opus] %s: primer decodificador %d Hz, %d canales, estado de %d bytes %s (%zu bytes)\n",
               opus_get_version_string(), sample_rate, channels, tam,
               propio_ ? "en memoria propia: el work buffer del juego no llega" : "en el work buffer del juego",
               work_size);
    }
    return ResultSuccess;
}

Result HardwareOpusDecoder::Finalize() {
    liberar();
    initialized_ = false;
    return ResultSuccess;
}

int HardwareOpusDecoder::GetSampleRate() const { return sample_rate_; }
int HardwareOpusDecoder::GetChannelCount() const { return channels_; }

Result HardwareOpusDecoder::Reset() {
    if (dec_) opus_decoder_ctl(dec_, OPUS_RESET_STATE);
    return ResultSuccess;
}

Result HardwareOpusDecoder::DecodeInterleaved(size_t* consumed, int* samples, int16_t* out, size_t out_size,
                                              const void* in, size_t in_size) {
    auto* data = static_cast<const uint8_t*>(in);
    size_t packet_size = in_size >= 8 ? be32(data) : 0;
    const uint8_t* packet = data + 8;
    size_t used = 8 + packet_size;
    const bool cabecera = in_size >= 8 && used <= in_size;
    if (!cabecera) {  // sin cabecera de Nintendo: el paquete entero
        packet = data;
        packet_size = in_size;
        used = in_size;
    }
    const int ch = channels_ == 1 ? 1 : 2;
    const int capacidad = static_cast<int>(out_size / (ch * sizeof(int16_t)));  // muestras por canal que caben
    const bool silenciar = NVN_FLAG("opus_silencio");
    int n = -1;
    if (dec_ && !silenciar && packet_size)
        n = opus_decode(dec_, packet, static_cast<opus_int32>(packet_size), out, capacidad, 0);
    if (n >= 0) {
        // El rango final del decodificador debe coincidir con el que el codificador dejó en la cabecera: si no, el
        // paquete se ha leído mal (desfase en el stream, cabecera mal interpretada).
        opus_uint32 rango = 0;
        if (cabecera && opus_decoder_ctl(dec_, OPUS_GET_FINAL_RANGE(&rango)) == OPUS_OK && rango != be32(data + 4))
            g_rango_distinto++;
    } else {
        // Silencio de la duración del paquete, como antes: el reloj de Wwise sigue igual aunque este paquete no suene.
        if (silenciar)
            g_silenciados++;
        else if (dec_ && packet_size) {
            g_errores++;
            g_ultimo_error = n;
        }
        n = std::min(opus_samples(packet, packet_size, sample_rate_), capacidad);
        memset(out, 0, static_cast<size_t>(n) * ch * sizeof(int16_t));
    }
    anotar_pico(out, static_cast<size_t>(n) * ch);
    g_paquetes++;
    g_muestras += static_cast<uint64_t>(n);
    *consumed = used;
    *samples = n;
    informe_si_toca();
    return ResultSuccess;
}

}  // namespace nn::codec

// API en C de nn::codec (FMOD 2.03 de Castlevania: Belmont's Curse decodifica así el Opus de sus bancos). Hay
// cuatro tipos (OpusDecoderType, HardwareOpusDecoderType y sus variantes MultiStream) con la misma interfaz. El juego
// reserva los structs y el work buffer con los tamaños que le damos, pero el tamaño de los structs no lo sabemos: el
// estado de libopus va en memoria nuestra y se busca por la dirección del struct. Paquetes: los del SDK traen
// delante 8 bytes (tamaño y rango final, big-endian); sin esa cabecera se toma el paquete entero, como en la clase.
#include <opus_multistream.h>

#include <mutex>
#include <unordered_map>

namespace {

struct OpusState {
    int sample_rate = 0, channels = 0, streams = 0, stereo = 0;
    OpusDecoder* dec = nullptr;
    OpusMSDecoder* ms = nullptr;
};

std::mutex g_opus_m;
std::unordered_map<const void*, OpusState*> g_opus;

OpusState* opus_state(const void* type) {
    std::lock_guard lock(g_opus_m);
    auto it = g_opus.find(type);
    return it == g_opus.end() ? nullptr : it->second;
}

void opus_finalize(const void* type) {
    OpusState* s = nullptr;
    {
        std::lock_guard lock(g_opus_m);
        auto it = g_opus.find(type);
        if (it == g_opus.end()) return;
        s = it->second;
        g_opus.erase(it);
    }
    if (s->dec) opus_decoder_destroy(s->dec);
    if (s->ms) opus_multistream_decoder_destroy(s->ms);
    delete s;
}

nn::Result opus_init(const void* type, int sample_rate, int channels, int streams, int stereo,
                     const uint8_t* mapping) {
    opus_finalize(type);
    auto* s = new OpusState{sample_rate, channels, streams, stereo};
    int err = OPUS_OK;
    if (streams > 0) {
        uint8_t identidad[255];
        if (!mapping) {
            for (int i = 0; i < 255; i++) identidad[i] = static_cast<uint8_t>(i);
            mapping = identidad;
        }
        s->ms = opus_multistream_decoder_create(sample_rate, channels, streams, stereo, mapping, &err);
    } else {
        s->dec = opus_decoder_create(sample_rate, channels, &err);
    }
    g_inicializados++;
    static std::atomic<int> avisos{0};
    if (err != OPUS_OK) {
        printf("[opus] init %d Hz, %d canales, %d streams (%d estéreo): %s; sonará en silencio\n", sample_rate,
               channels, streams, stereo, opus_strerror(err));
    } else if (avisos++ < 4) {
        printf("[opus] decodificador %s: %d Hz, %d canales, %d streams (%d estéreo)\n",
               streams > 0 ? "multistream" : "simple", sample_rate, channels, streams, stereo);
    }
    std::lock_guard lock(g_opus_m);
    g_opus[type] = s;
    return nn::ResultSuccess;
}

nn::Result opus_decode_interleaved(const void* type, size_t* consumed, int* samples, int16_t* out, size_t out_size,
                                   const void* in, size_t in_size) {
    OpusState* s = opus_state(type);
    auto* data = static_cast<const uint8_t*>(in);
    size_t packet_size = in_size >= 8 ? be32(data) : 0;
    const uint8_t* packet = data + 8;
    size_t used = 8 + packet_size;
    const bool cabecera = in_size >= 8 && used <= in_size;
    if (!cabecera) {
        packet = data;
        packet_size = in_size;
        used = in_size;
    }
    const int ch = s ? std::max(1, s->channels) : 2;
    const int capacidad = static_cast<int>(out_size / (static_cast<size_t>(ch) * sizeof(int16_t)));
    const bool silenciar = NVN_FLAG("opus_silencio");
    int n = -1;
    if (s && !silenciar && packet_size) {
        if (s->dec)
            n = opus_decode(s->dec, packet, static_cast<opus_int32>(packet_size), out, capacidad, 0);
        else if (s->ms)
            n = opus_multistream_decode(s->ms, packet, static_cast<opus_int32>(packet_size), out, capacidad, 0);
    }
    if (n >= 0) {
        opus_uint32 rango = 0;
        int r = OPUS_UNIMPLEMENTED;
        if (cabecera && s->dec) r = opus_decoder_ctl(s->dec, OPUS_GET_FINAL_RANGE(&rango));
        if (cabecera && s->ms) r = opus_multistream_decoder_ctl(s->ms, OPUS_GET_FINAL_RANGE(&rango));
        if (r == OPUS_OK && rango != be32(data + 4)) g_rango_distinto++;
    } else {
        if (silenciar)
            g_silenciados++;
        else if (s && packet_size) {
            g_errores++;
            g_ultimo_error = n;
        }
        n = std::min(opus_samples(packet, packet_size, s ? s->sample_rate : 48000), capacidad);
        memset(out, 0, static_cast<size_t>(std::max(n, 0)) * ch * sizeof(int16_t));
    }
    anotar_pico(out, static_cast<size_t>(std::max(n, 0)) * ch);
    g_paquetes++;
    g_muestras += static_cast<uint64_t>(std::max(n, 0));
    *consumed = used;
    *samples = n;
    informe_si_toca();
    return nn::ResultSuccess;
}

nn::Result opus_reset(const void* type) {
    if (OpusState* s = opus_state(type)) {
        if (s->dec) opus_decoder_ctl(s->dec, OPUS_RESET_STATE);
        if (s->ms) opus_multistream_decoder_ctl(s->ms, OPUS_RESET_STATE);
    }
    return nn::ResultSuccess;
}

int opus_field(const void* type, int OpusState::*field) {
    OpusState* s = opus_state(type);
    return s ? s->*field : 0;
}

// El estado va en memoria nuestra: el work buffer solo tiene que existir (algunos juegos comprueban que no sea 0).
size_t opus_work_size(int channels) { return 4096 + 256 * static_cast<size_t>(std::max(channels, 1)); }

}  // namespace

namespace nn::codec {

struct OpusDecoderType;
struct HardwareOpusDecoderType;
struct OpusMultiStreamDecoderType;
struct HardwareOpusMultiStreamDecoderType;

// Simple, por software y por "hardware" (aquí es lo mismo).
size_t GetOpusDecoderWorkBufferSize(int, int channels) { return opus_work_size(channels); }
size_t GetHardwareOpusDecoderWorkBufferSize(int, int channels) { return opus_work_size(channels); }
size_t GetHardwareOpusDecoderWorkBufferSize(int, int channels, int) { return opus_work_size(channels); }
Result InitializeOpusDecoder(OpusDecoderType* t, int sample_rate, int channels, void*, size_t) {
    return opus_init(t, sample_rate, channels, 0, 0, nullptr);
}
Result InitializeHardwareOpusDecoder(HardwareOpusDecoderType* t, int sample_rate, int channels, void*, size_t, int) {
    return opus_init(t, sample_rate, channels, 0, 0, nullptr);
}
Result FinalizeOpusDecoder(OpusDecoderType* t) { return opus_finalize(t), ResultSuccess; }
Result FinalizeHardwareOpusDecoder(HardwareOpusDecoderType* t) { return opus_finalize(t), ResultSuccess; }
Result ResetOpusDecoder(OpusDecoderType* t) { return opus_reset(t); }
Result ResetHardwareOpusDecoder(HardwareOpusDecoderType* t) { return opus_reset(t); }
int GetOpusDecoderSampleRate(const OpusDecoderType* t) { return opus_field(t, &OpusState::sample_rate); }
int GetHardwareOpusDecoderSampleRate(const HardwareOpusDecoderType* t) {
    return opus_field(t, &OpusState::sample_rate);
}
int GetOpusDecoderChannelCount(const OpusDecoderType* t) { return opus_field(t, &OpusState::channels); }
int GetHardwareOpusDecoderChannelCount(const HardwareOpusDecoderType* t) { return opus_field(t, &OpusState::channels); }
Result DecodeOpusInterleaved(OpusDecoderType* t, size_t* consumed, int* samples, int16_t* out, size_t out_size,
                             const void* in, size_t in_size) {
    return opus_decode_interleaved(t, consumed, samples, out, out_size, in, in_size);
}
Result DecodeOpusInterleavedWithHardware(HardwareOpusDecoderType* t, size_t* consumed, int* samples, int16_t* out,
                                         size_t out_size, const void* in, size_t in_size) {
    return opus_decode_interleaved(t, consumed, samples, out, out_size, in, in_size);
}

// MultiStream (más de 2 canales: música o ambiente envolvente).
size_t GetOpusMultiStreamDecoderWorkBufferSize(int, int channels, int, int) { return opus_work_size(channels); }
size_t GetHardwareOpusMultiStreamDecoderWorkBufferSize(int, int channels, int, int) {
    return opus_work_size(channels);
}
size_t GetHardwareOpusMultiStreamDecoderWorkBufferSize(int, int channels, int, int, int) {
    return opus_work_size(channels);
}
Result InitializeOpusMultiStreamDecoder(OpusMultiStreamDecoderType* t, int sample_rate, int channels, int streams,
                                        int stereo, const uint8_t* mapping, void*, size_t) {
    return opus_init(t, sample_rate, channels, streams, stereo, mapping);
}
Result InitializeHardwareOpusMultiStreamDecoder(HardwareOpusMultiStreamDecoderType* t, int sample_rate, int channels,
                                                int streams, int stereo, const uint8_t* mapping, void*, size_t, int) {
    return opus_init(t, sample_rate, channels, streams, stereo, mapping);
}
Result FinalizeOpusMultiStreamDecoder(OpusMultiStreamDecoderType* t) { return opus_finalize(t), ResultSuccess; }
Result FinalizeHardwareOpusMultiStreamDecoder(HardwareOpusMultiStreamDecoderType* t) {
    return opus_finalize(t), ResultSuccess;
}
Result ResetOpusMultiStreamDecoder(OpusMultiStreamDecoderType* t) { return opus_reset(t); }
Result ResetHardwareOpusMultiStreamDecoder(HardwareOpusMultiStreamDecoderType* t) { return opus_reset(t); }
int GetOpusMultiStreamDecoderSampleRate(const OpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::sample_rate);
}
int GetHardwareOpusMultiStreamDecoderSampleRate(const HardwareOpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::sample_rate);
}
int GetOpusMultiStreamDecoderChannelCount(const OpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::channels);
}
// Cuánto tardó el decoder de hardware en el último paquete, en microsegundos (estadística del juego). Aquí se
// decodifica por software en la misma llamada: no hay tiempo aparte que dar.
int64_t GetHardwareOpusMultiStreamDecoderProcessingTime(const HardwareOpusMultiStreamDecoderType*) { return 0; }
int GetHardwareOpusMultiStreamDecoderChannelCount(const HardwareOpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::channels);
}
int GetOpusMultiStreamDecoderTotalStreamCount(const OpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::streams);
}
int GetHardwareOpusMultiStreamDecoderTotalStreamCount(const HardwareOpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::streams);
}
int GetOpusMultiStreamDecoderStereoStreamCount(const OpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::stereo);
}
int GetHardwareOpusMultiStreamDecoderStereoStreamCount(const HardwareOpusMultiStreamDecoderType* t) {
    return opus_field(t, &OpusState::stereo);
}
Result DecodeOpusMultiStreamInterleaved(OpusMultiStreamDecoderType* t, size_t* consumed, int* samples, int16_t* out,
                                        size_t out_size, const void* in, size_t in_size) {
    return opus_decode_interleaved(t, consumed, samples, out, out_size, in, in_size);
}
Result DecodeOpusMultiStreamInterleavedWithHardware(HardwareOpusMultiStreamDecoderType* t, size_t* consumed,
                                                    int* samples, int16_t* out, size_t out_size, const void* in,
                                                    size_t in_size) {
    return opus_decode_interleaved(t, consumed, samples, out, out_size, in, in_size);
}

}  // namespace nn::codec
