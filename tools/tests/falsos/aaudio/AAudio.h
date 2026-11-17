#pragma once
// AAudio de mentira para compilar runtime/src/nnsdk/audio.cpp fuera del dispositivo (tools/tests/audio_fmod_prueba.cpp).
// Solo lo que usa audio.cpp. El stream no suena: guarda el callback de datos y la prueba lo llama cuando quiere
// (falsos::aaudio_rafaga), o deja de llamarlo para simular un stream que no arranca o se queda parado.

#include <atomic>
#include <cstdint>

using aaudio_result_t = int32_t;
using aaudio_data_callback_result_t = int32_t;
using aaudio_format_t = int32_t;
using aaudio_stream_state_t = int32_t;

constexpr aaudio_result_t AAUDIO_OK = 0;
constexpr aaudio_format_t AAUDIO_FORMAT_PCM_I16 = 1;
constexpr int32_t AAUDIO_PERFORMANCE_MODE_LOW_LATENCY = 12;
constexpr aaudio_data_callback_result_t AAUDIO_CALLBACK_RESULT_CONTINUE = 0;

struct AAudioStream;
using AAudioStream_dataCallback = aaudio_data_callback_result_t (*)(AAudioStream*, void* user, void* audio, int32_t frames);
using AAudioStream_errorCallback = void (*)(AAudioStream*, void* user, aaudio_result_t error);

struct AAudioStream {
    AAudioStream_dataCallback datos = nullptr;
    AAudioStream_errorCallback error = nullptr;
    void* user = nullptr;
    bool en_marcha = false;
};
struct AAudioStreamBuilder {
    AAudioStream plantilla;
};

namespace falsos {
inline std::atomic<AAudioStream*> g_stream{nullptr};  // el último abierto y arrancado
inline std::atomic<int> g_abiertos{0};
// Lo que haría el hilo de AAudio: pide `frames` frames al callback. false si aún no hay stream en marcha.
inline bool aaudio_rafaga(int16_t* salida, int32_t frames) {
    AAudioStream* s = g_stream.load();
    if (!s || !s->datos) return false;
    s->datos(s, s->user, salida, frames);
    return true;
}
}  // namespace falsos

inline aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** b) {
    *b = new AAudioStreamBuilder;
    return AAUDIO_OK;
}
inline void AAudioStreamBuilder_setFormat(AAudioStreamBuilder*, aaudio_format_t) {}
inline void AAudioStreamBuilder_setChannelCount(AAudioStreamBuilder*, int32_t) {}
inline void AAudioStreamBuilder_setSampleRate(AAudioStreamBuilder*, int32_t) {}
inline void AAudioStreamBuilder_setPerformanceMode(AAudioStreamBuilder*, int32_t) {}
inline void AAudioStreamBuilder_setDataCallback(AAudioStreamBuilder* b, AAudioStream_dataCallback f, void* user) {
    b->plantilla.datos = f;
    b->plantilla.user = user;
}
inline void AAudioStreamBuilder_setErrorCallback(AAudioStreamBuilder* b, AAudioStream_errorCallback f, void*) {
    b->plantilla.error = f;
}
inline aaudio_result_t AAudioStreamBuilder_openStream(AAudioStreamBuilder* b, AAudioStream** s) {
    *s = new AAudioStream(b->plantilla);
    falsos::g_abiertos++;
    return AAUDIO_OK;
}
inline void AAudioStreamBuilder_delete(AAudioStreamBuilder* b) { delete b; }
inline aaudio_result_t AAudioStream_requestStart(AAudioStream* s) {
    s->en_marcha = true;
    falsos::g_stream = s;
    return AAUDIO_OK;
}
inline aaudio_result_t AAudioStream_requestStop(AAudioStream* s) {
    s->en_marcha = false;
    return AAUDIO_OK;
}
inline aaudio_result_t AAudioStream_close(AAudioStream* s) {
    AAudioStream* esperado = s;
    falsos::g_stream.compare_exchange_strong(esperado, nullptr);
    return AAUDIO_OK;  // no se libera: la prueba puede estar a media ráfaga
}
inline int32_t AAudioStream_getFramesPerBurst(AAudioStream*) { return 96; }
inline int32_t AAudioStream_getBufferSizeInFrames(AAudioStream*) { return 768; }
inline int32_t AAudioStream_getSampleRate(AAudioStream*) { return 48000; }
inline int32_t AAudioStream_getChannelCount(AAudioStream*) { return 2; }
inline aaudio_stream_state_t AAudioStream_getState(AAudioStream*) { return 4; }
inline int32_t AAudioStream_getXRunCount(AAudioStream*) { return 0; }
inline const char* AAudio_convertResultToText(aaudio_result_t r) { return r == AAUDIO_OK ? "AAUDIO_OK" : "error"; }
inline const char* AAudio_convertStreamStateToText(aaudio_stream_state_t) { return "STARTED"; }
