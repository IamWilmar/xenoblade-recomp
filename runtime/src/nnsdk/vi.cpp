// nn::vi (pantalla y capa) y nv:: (arranque del driver). En Android solo hay una ventana:
// la de la APK, que es lo que devuelve GetNativeWindow y luego recibe NVN.

#include <time.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "nnsdk/os.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::os {
Tick GetSystemTick();
}

namespace nn::vi {

struct Display;
struct Layer;

namespace {
char g_display, g_layer;
}

void Initialize() {}
void Finalize() {}

// El evento de vsync de la pantalla: un hilo lo señala a 60 Hz (Unity espera en él para su ritmo de frames).
Result GetDisplayVsyncEvent(os::SystemEventType* event, Display*) {
    static std::mutex m;
    static std::vector<os::SystemEventType*> events;
    nnsdk::create_system_event(event, true, false);
    std::lock_guard lock(m);
    events.push_back(event);
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            timespec next;
            clock_gettime(CLOCK_MONOTONIC, &next);
            for (;;) {
                next.tv_nsec += 16'666'667;
                if (next.tv_nsec >= 1'000'000'000) {
                    next.tv_nsec -= 1'000'000'000;
                    next.tv_sec++;
                }
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
                std::lock_guard lock(m);
                for (os::SystemEventType* e : events) nnsdk::signal_system_event(e);
            }
        }).detach();
    });
    return ResultSuccess;
}

Result OpenDefaultDisplay(Display** out) {
    *out = reinterpret_cast<Display*>(&g_display);
    return ResultSuccess;
}

void CloseDisplay(Display*) {}

Result CreateLayer(Layer** out, Display*) {
    *out = reinterpret_cast<Layer*>(&g_layer);
    return ResultSuccess;
}
// La ventana es de la APK y sigue viva hasta que Android la destruya.
void DestroyLayer(Layer*) {}
// El escalado de la capa a la pantalla: aquí lo hace el presentador (FSR o blit a la ventana de Android).
enum ScalingMode : int {};
Result SetLayerScalingMode(Layer*, ScalingMode) { return ResultSuccess; }

// --- Historial de frames ----------------------------------------------------------------------------------------------
// Unreal Engine sigue con él cuándo llega a pantalla cada frame que presenta (su "flip tracking", main+0x144b3e0):
// antes de presentar pide el número del último frame (el suyo será el siguiente) y un hilo, en cada vsync, lista el
// historial y avisa del frame más nuevo que ya se ha visto. El SDK (sdk+0x4510a0) guarda los 8 últimos y da, por cada
// uno, 40 bytes: estado (1 en cola, 2 ya presentado), número de frame, tick en que se encoló y dos veces el tick en
// que se presentó (0 mientras sigue en cola). Sin buffer dice cuántos puede haber: 8.
struct FrameInfo {
    uint64_t state;
    uint64_t number;
    int64_t queued_tick;
    int64_t present_begin_tick;
    int64_t present_end_tick;
};
static_assert(sizeof(FrameInfo) == 0x28);

namespace {
constexpr int kHistory = 8;
std::mutex g_frames_m;
FrameInfo g_frames[kHistory];  // anillo: el frame n está en n % 8
uint64_t g_latest = 0;         // número del último frame encolado (el primero es el 1)
}  // namespace

Result GetLatestFrameNumber(uint64_t* out, const Layer*) {
    std::lock_guard lock(g_frames_m);
    *out = g_latest;
    return g_latest ? ResultSuccess : Result{0x1672};  // como el SDK: aún no hay ninguno
}

int ListFrameHistory(FrameInfo* out, int count, const Layer*) {
    if (!out || count <= 0) return kHistory;
    std::lock_guard lock(g_frames_m);
    int n = 0;
    for (uint64_t f = g_latest; f > 0 && g_latest - f < kHistory && n < count; f--) out[n++] = g_frames[f % kHistory];
    return n;
}

// Sin ventana (terminal) se devuelve error: el juego no tiene dónde presentar.
Result GetNativeWindow(void** out, Layer*) {
    *out = nnsdk::host().native_window;
    return *out ? ResultSuccess : Result{0x272};
}

}  // namespace nn::vi

void nnsdk::frame_queued() {
    const int64_t now = nn::os::GetSystemTick().value;
    std::lock_guard lock(nn::vi::g_frames_m);
    // Un frame que lleva más de 3 por detrás sin salir (no había ventana) se da por presentado.
    if (nn::vi::g_latest >= 3) {
        nn::vi::FrameInfo& old = nn::vi::g_frames[(nn::vi::g_latest - 3) % nn::vi::kHistory];
        if (old.state == 1) old.state = 2, old.present_begin_tick = old.present_end_tick = now;
    }
    nn::vi::g_frames[++nn::vi::g_latest % nn::vi::kHistory] = {1, nn::vi::g_latest, now, 0, 0};
}

void nnsdk::frame_presented() {
    const int64_t now = nn::os::GetSystemTick().value;
    std::lock_guard lock(nn::vi::g_frames_m);
    // Las imágenes salen en el orden en que se encolaron: la que sale es la más vieja que sigue en cola.
    for (uint64_t f = nn::vi::g_latest >= nn::vi::kHistory ? nn::vi::g_latest - nn::vi::kHistory + 1 : 1;
         f <= nn::vi::g_latest; f++) {
        nn::vi::FrameInfo& info = nn::vi::g_frames[f % nn::vi::kHistory];
        if (info.state != 1) continue;
        info.state = 2;
        info.present_begin_tick = info.present_end_tick = now;
        return;
    }
}

namespace nv {

void InitializeGraphics(void*, size_t) {}
void FinalizeGraphics() {}
void SetGraphicsAllocator(void* (*)(size_t, size_t, void*), void (*)(void*, void*), void* (*)(void*, size_t, void*),
                          void*) {}
void SetGraphicsDevtoolsAllocator(void* (*)(size_t, size_t, void*), void (*)(void*, void*),
                                  void* (*)(void*, size_t, void*), void*) {}

}  // namespace nv
