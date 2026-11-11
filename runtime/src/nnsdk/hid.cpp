// nn::hid: la Flip es una consola portátil, así que el juego ve un solo mando, en modo portátil (NpadIdType Handheld =
// 0x20, estilo Handheld) con los botones y sticks que traduce el loader (host.h). MH Rise recorre No1..No8 y Handheld
// con GetNpadStyleSet y crea un mando por cada id conectado (main+0x6f8fa30): con el portátil como único conectado, la
// Flip es el jugador 1 y no salta el applet de "conecta un mando".
//
// Lo que la Flip no tiene responde como una Switch sin ello: ratón y teclado USB desconectados (estados con los
// atributos a 0), pantalla táctil sin toques ni gestos, y los sensores de movimiento del portátil quietos. La vibración
// acepta handles y valores pero no tiene motor; el grafo de VibrationNode está en hid_vibration.cpp.

#include <time.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::util {

// BitFlagSet<32, NpadStyleTag> del SDK: un uint32 envuelto, se pasa y se devuelve en un registro.
template <int N, typename Tag>
struct BitFlagSet {
    uint32_t storage[(N + 31) / 32];
};

namespace neon {
struct Vector4fType {
    float v[4];
};
}  // namespace neon

}  // namespace nn::util

namespace nn::hid {

using NpadIdType = uint32_t;
struct NpadStyleTag;
using NpadStyleSet = util::BitFlagSet<32, NpadStyleTag>;

struct NpadFullKeyState;
struct NpadHandheldState;
struct NpadJoyDualState;
struct NpadJoyLeftState;
struct NpadJoyRightState;
struct GestureState;

enum NpadJoyHoldType : int {};
enum NpadHandheldActivationMode : int {};
enum NpadJoyDeviceType : int {};
enum NpadJoyAssignmentMode : int { NpadJoyAssignmentMode_Dual = 0, NpadJoyAssignmentMode_Single = 1 };
enum GyroscopeZeroDriftMode : int {};

struct SixAxisSensorHandle {
    uint32_t storage;
};
struct VibrationDeviceHandle {
    uint32_t storage;
};
struct VibrationDeviceInfo {
    uint32_t device_type;  // 1: actuador lineal (Joy-Con, Pro), 2: motor excéntrico (mando de GameCube)
    uint32_t position;     // 0: ninguna, 1: izquierda, 2: derecha
};

// Colores del mando (nn::util::Color4u8: r, g, b, a).
struct NpadControllerColor {
    uint8_t main[4];
    uint8_t sub[4];
};

struct ControllerSupportArg {
    uint8_t storage[0x430];
    void SetDefault();
};
struct ControllerSupportResultInfo {
    int8_t player_count;
    char pad[3];
    NpadIdType selected_id;
};

struct MouseState {
    int64_t sampling_number;
    int32_t x, y, delta_x, delta_y, wheel_delta_x, wheel_delta_y;
    uint32_t buttons;
    uint32_t attributes;  // bit 1: conectado (main+0x6fa2ae8 lo mira antes que nada)
};
static_assert(sizeof(MouseState) == 0x28);

struct KeyboardState {
    int64_t sampling_number;
    uint32_t modifiers;
    uint32_t attributes;  // bit 0: conectado
    uint64_t keys[4];
};
static_assert(sizeof(KeyboardState) == 0x30);

struct TouchState {
    int64_t delta_time;
    uint32_t attributes;
    int32_t finger_id;
    int32_t x, y, diameter_x, diameter_y, rotation_angle;
    int32_t reserved;
};
template <size_t N>
struct TouchScreenState {
    int64_t sampling_number;
    int32_t count;
    int32_t reserved;
    TouchState touches[N];
};
static_assert(sizeof(TouchScreenState<16>) == 0x290);

struct SixAxisSensorState {
    int64_t delta_time;  // ns
    int64_t sampling_number;
    float acceleration[3];      // en G
    float angular_velocity[3];  // en vueltas por segundo
    float angle[3];             // en vueltas
    float direction[9];         // ejes x, y, z del mando en el sistema del mundo
    uint32_t attributes;        // bit 0: conectado, bit 1: interpolado
    uint32_t reserved;
    void GetQuaternion(util::neon::Vector4fType* out) const;
};
static_assert(sizeof(SixAxisSensorState) == 0x60);

}  // namespace nn::hid

using namespace nn;
using namespace nn::hid;

namespace {

constexpr NpadIdType kNpadHandheld = 0x20;
constexpr uint32_t kStyleFullKey = 1u << 0, kStyleHandheld = 1u << 1, kStyleJoyDual = 1u << 2, kStyleJoyLeft = 1u << 3,
                   kStyleJoyRight = 1u << 4, kStyleGc = 1u << 5;
// Atributos del estado: conectado, con cable, mitad izquierda y derecha conectadas y con cable.
constexpr uint32_t kHandheldAttributes = 0x3F;

// Errores del módulo hid (202) que el SDK deja pasar al juego en vez de abortar.
constexpr Result kResultNpadIsDualJoyCon{202 | 601 << 9};
constexpr Result kResultNpadIsSameType{202 | 602 << 9};
constexpr Result kResultNpadControllerNotConnected{202 | 604 << 9};

// Estado común de todos los estilos (NpadFullKeyState, NpadHandheldState, NpadJoyDualState…): 40 bytes.
struct NpadState {
    int64_t sampling_number;
    uint64_t buttons;
    int32_t lx, ly, rx, ry;
    uint32_t attributes;
    uint32_t reserved;
};
static_assert(sizeof(NpadState) == 0x28);

std::atomic<int64_t> g_sampling{0};

bool handheld_connected() { return nnsdk::host().input != nullptr; }

uint32_t style_of(NpadIdType id) { return id == kNpadHandheld && handheld_connected() ? kStyleHandheld : 0; }

// El estado de un estilo solo trae datos si es el que tiene el mando: GetNpadState(NpadFullKeyState*, Handheld) da en
// una Switch un mando desconectado, y así no se cuenta dos veces la misma pulsación si el juego lee varios estilos.
void fill_state(void* state, NpadIdType id, uint32_t style) {
    static std::atomic<int> total{0};  // DEPURACIÓN: cada 100 lecturas, la última y lo que hay en la entrada
    if (++total % 100 == 0) {
        NnsdkInput* in = nnsdk::host().input;
        printf("[hid] %d lecturas de mando (última id %#x estilo %#x), entrada %p botones %#llx retenidos %#llx\n",
               total.load(), id, style, static_cast<void*>(in), in ? static_cast<unsigned long long>(in->buttons.load()) : 0ull,
               in ? static_cast<unsigned long long>(in->latched.load()) : 0ull);
    }
    static std::atomic<int> calls{0};  // DEPURACIÓN: qué mandos y estilos lee el juego mientras hay una pulsación
    if (NnsdkInput* in = nnsdk::host().input; in && (in->latched.load() || in->buttons.load()) && nvn::first_times(calls, 40))
        printf("[hid] GetNpadState id %#x estilo %#x (conectado %#x) con botones %#llx\n", id, style, style_of(id),
               static_cast<unsigned long long>(in->buttons.load() | in->latched.load()));
    auto* out = static_cast<NpadState*>(state);
    memset(out, 0, sizeof *out);
    out->sampling_number = ++g_sampling;
    NnsdkInput* in = nnsdk::host().input;
    if (style_of(id) != style || !in) return;
    uint64_t buttons = in->buttons.load();
    if (uint64_t latched = in->latched.load()) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        if (int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec < in->latched_until_ns.load())
            buttons |= latched;
        else
            in->latched.fetch_and(~latched);
    }
    out->buttons = buttons;
    static uint64_t last = 0;  // DEPURACIÓN: qué ve el juego cuando cambian los botones
    if (buttons != last) {
        static int logged = 0;
        if (nvn::first_times(logged, 60)) printf("[hid] botones 0x%llx\n", static_cast<unsigned long long>(buttons));
        last = buttons;
    }
    out->lx = in->sticks[0], out->ly = in->sticks[1], out->rx = in->sticks[2], out->ry = in->sticks[3];
    out->attributes = kHandheldAttributes;
}

// Número de muestra de lo que no tiene estado propio (táctil, ratón, teclado, sensores): avanza al ritmo del muestreo
// del hid de la Switch, cada 5 ms, y no con cada lectura, para que dos lecturas en el mismo frame no parezcan dos
// muestras nuevas.
int64_t sample_clock() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec) / 5'000'000;
}

bool valid_id(NpadIdType id) { return id < 8 || id == 0x10 || id == kNpadHandheld; }

// Handle de sensor o de vibración como lo arma nn::hid::utility: tipo (índice del estilo + 3), id del mando e índice
// del dispositivo (0 izquierda, 1 derecha, 2 el único del mando).
uint32_t make_handle(uint32_t type, NpadIdType id, uint32_t index) { return type | (id & 0xFF) << 8 | index << 16; }

// Modo de asignación de los Joy-Con de No1..No8 (el portátil siempre va en dual): solo se guarda para devolverlo.
std::atomic<int> g_assignment[8] = {};

}  // namespace

namespace nn::hid {

void InitializeNpad() {}
void InitializeTouchScreen() {}
void InitializeGesture() {}
void InitializeMouse() {}
void InitializeKeyboard() {}

void SetSupportedNpadIdType(const NpadIdType* ids, size_t count) {
    char list[128] = "";  // DEPURACIÓN: qué mandos acepta el juego en cada momento
    for (size_t k = 0, n = 0; k < count && n < sizeof list - 8; k++) n += snprintf(list + n, sizeof list - n, " %#x", ids[k]);
    printf("[hid] SetSupportedNpadIdType:%s\n", list);
}
// El juego pide 0x1f (todos los estilos de Joy-Con y Pro) y, con el agarre en horizontal, solo Joy-Con sueltos (0x18).
// No se hace caso: en una Switch eso desconectaría el portátil, y la Flip no tiene otro mando que ofrecer.
void SetSupportedNpadStyleSet(NpadStyleSet styles) { printf("[hid] SetSupportedNpadStyleSet %#x\n", styles.storage[0]); }
void SetNpadJoyHoldType(NpadJoyHoldType) {}
void SetNpadHandheldActivationMode(NpadHandheldActivationMode) {}

// Solo hay mando en el modo portátil.
NpadStyleSet GetNpadStyleSet(const NpadIdType& id) {
    static std::atomic<int> calls{0};  // DEPURACIÓN: qué ids pregunta el juego (cada 500 llamadas, la última)
    if (++calls % 500 == 0)
        printf("[hid] %d GetNpadStyleSet (último id %#x, desde %p)\n", calls.load(), id, __builtin_return_address(0));
    return {{style_of(id)}};
}

void GetNpadState(NpadFullKeyState* out, const NpadIdType& id) { fill_state(out, id, kStyleFullKey); }
void GetNpadState(NpadHandheldState* out, const NpadIdType& id) { fill_state(out, id, kStyleHandheld); }
void GetNpadState(NpadJoyDualState* out, const NpadIdType& id) { fill_state(out, id, kStyleJoyDual); }
void GetNpadState(NpadJoyLeftState* out, const NpadIdType& id) { fill_state(out, id, kStyleJoyLeft); }
void GetNpadState(NpadJoyRightState* out, const NpadIdType& id) { fill_state(out, id, kStyleJoyRight); }

// El juego solo desconecta No1..No8 (al portátil no lo toca, main+0x6f910a4), que nunca están conectados.
void DisconnectNpad(const NpadIdType&) {}

// Asignación de Joy-Con (menú de mandos del juego): no hay Joy-Con sueltos que juntar, separar ni intercambiar.
NpadJoyAssignmentMode GetNpadJoyAssignment(const NpadIdType& id) {
    return id < 8 ? static_cast<NpadJoyAssignmentMode>(g_assignment[id].load()) : NpadJoyAssignmentMode_Dual;
}
void SetNpadJoyAssignmentModeSingle(const NpadIdType& id) {
    if (id < 8) g_assignment[id] = NpadJoyAssignmentMode_Single;
}
void SetNpadJoyAssignmentModeSingle(const NpadIdType& id, NpadJoyDeviceType) { SetNpadJoyAssignmentModeSingle(id); }
void SetNpadJoyAssignmentModeDual(const NpadIdType& id) {
    if (id < 8) g_assignment[id] = NpadJoyAssignmentMode_Dual;
}
void SwapNpadAssignment(const NpadIdType&, const NpadIdType&) {}
void StartLrAssignmentMode() {}
void StopLrAssignmentMode() {}

// Solo se juntan dos Joy-Con sueltos de distinto lado; con el portátil y huecos vacíos el hid contesta "son del mismo
// tipo" (los dos vacíos) o "ya es un par" (el portátil), que el juego trata como que no se pudo (main+0x6f8ff20).
Result MergeSingleJoyAsDualJoy(const NpadIdType& a, const NpadIdType& b) {
    return style_of(a) == style_of(b) ? kResultNpadIsSameType : kResultNpadIsDualJoyCon;
}

// Las luces de jugador de los Joy-Con de cada hueco; el portátil no tiene.
uint8_t GetPlayerLedPattern(const NpadIdType& id) {
    static const uint8_t kPatterns[8] = {0b0001, 0b0011, 0b0111, 0b1111, 0b1001, 0b0101, 0b1101, 0b0110};
    return id < 8 ? kPatterns[id] : 0;
}

// Colores del portátil: Joy-Con grises (cuerpo 828282, botones 0F0F0F). Un hueco sin mando da el error de "no hay
// mando" que el SDK devuelve en vez de abortar.
Result GetNpadControllerColor(NpadControllerColor* out, const NpadIdType& id) {
    if (!style_of(id)) return kResultNpadControllerNotConnected;
    *out = {{0x82, 0x82, 0x82, 0xFF}, {0x0F, 0x0F, 0x0F, 0xFF}};
    return ResultSuccess;
}
Result GetNpadControllerColor(NpadControllerColor* left, NpadControllerColor* right, const NpadIdType& id) {
    if (Result r = GetNpadControllerColor(left, id); r.value) return r;
    *right = *left;
    return ResultSuccess;
}

// Applet de selección de mando: el portátil ya está conectado como jugador 1.
Result ShowControllerSupport(ControllerSupportResultInfo* out, const ControllerSupportArg&) {
    if (out) {
        out->player_count = 1;
        out->selected_id = kNpadHandheld;
    }
    return ResultSuccess;
}

// Lo mismo que el SDK: de 0 a 4 jugadores, quitar la conexión a otro y justificar a la izquierda, Joy-Con en dual
// permitido; el resto (colores y textos de cada jugador) a cero.
void ControllerSupportArg::SetDefault() {
    storage[0] = 0;
    storage[1] = 4;
    storage[2] = 1;
    storage[3] = 1;
    storage[4] = 1;
    memset(storage + 5, 0, sizeof storage - 5);
}

// --- Pantalla táctil, gestos, ratón y teclado: presentes pero sin nada que contar ---

// Una sola muestra, sin toques: el juego recorre las muestras más nuevas que la última que vio (main+0x7c6c44c) y se
// queda con el número de la primera.
template <size_t N>
int GetTouchScreenStates(TouchScreenState<N>* out, int count);
template <>
int GetTouchScreenStates<16>(TouchScreenState<16>* out, int count) {
    if (count < 1) return 0;
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
    return 1;
}

// Sin toques no hay gestos: ninguna muestra, y el juego no procesa nada (main+0x7c6c380).
int GetGestureStates(GestureState*, int) { return 0; }

void GetMouseState(MouseState* out) {
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
}
int GetMouseStates(MouseState* out, int count) {
    if (count < 1) return 0;
    GetMouseState(out);
    return 1;
}

void GetKeyboardState(KeyboardState* out) {
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
}
int GetKeyboardStates(KeyboardState* out, int count) {
    if (count < 1) return 0;
    GetKeyboardState(out);
    return 1;
}

// --- Sensores de movimiento: los del portátil, siempre en reposo ---

int GetSixAxisSensorHandles(SixAxisSensorHandle* out, int count, const NpadIdType& id, NpadStyleSet style_set) {
    const uint32_t style = style_set.storage[0];
    if (count < 1 || !valid_id(id)) return 0;
    if (style & (kStyleFullKey | kStyleGc)) {
        out[0].storage = make_handle(3, id, 2);
        return 1;
    }
    if (style & kStyleHandheld) {
        out[0].storage = make_handle(4, id, 2);
        return 1;
    }
    if (style & kStyleJoyDual) {
        out[0].storage = make_handle(5, id, 0);
        if (count < 2) return 1;
        out[1].storage = make_handle(5, id, 1);
        return 2;
    }
    if (style & kStyleJoyLeft) {
        out[0].storage = make_handle(6, id, 0);
        return 1;
    }
    if (style & kStyleJoyRight) {
        out[0].storage = make_handle(7, id, 1);
        return 1;
    }
    return 0;
}

void StartSixAxisSensor(const SixAxisSensorHandle&) {}
void StopSixAxisSensor(const SixAxisSensorHandle&) {}
void SetGyroscopeZeroDriftMode(const SixAxisSensorHandle&, const GyroscopeZeroDriftMode&) {}

// En reposo y plano: solo la gravedad (-1 G en z), sin giro y con los ejes del mando alineados con los del mundo. Así
// GetQuaternion da la identidad y nada que use el giroscopio se mueve solo.
void GetSixAxisSensorState(SixAxisSensorState* out, const SixAxisSensorHandle& handle) {
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
    out->delta_time = 5'000'000;
    if ((handle.storage >> 8 & 0xFF) != kNpadHandheld || !handheld_connected()) return;
    out->acceleration[2] = -1.0f;
    out->direction[0] = out->direction[4] = out->direction[8] = 1.0f;
    out->attributes = 1;
}

// La misma cuenta que el SDK: las cuatro estimaciones de (x, y, z, w) a partir de la matriz de dirección, se queda la
// de la componente mayor (con >=: en un empate gana la última) y se normaliza.
void SixAxisSensorState::GetQuaternion(util::neon::Vector4fType* out) const {
    const float* d = direction;
    const float t[4] = {1 + d[0] - d[4] - d[8], 1 - d[0] + d[4] - d[8], 1 - d[0] - d[4] + d[8], 1 + d[0] + d[4] + d[8]};
    const float candidates[4][4] = {
        {t[0], d[1] + d[3], d[6] + d[2], d[5] - d[7]},
        {d[1] + d[3], t[1], d[5] + d[7], d[6] - d[2]},
        {d[6] + d[2], d[5] + d[7], t[2], d[1] - d[3]},
        {d[5] - d[7], d[6] - d[2], d[1] - d[3], t[3]},
    };
    int best = 0;
    for (int k = 1; k < 4; k++)
        if (t[k] >= t[best]) best = k;
    const float* q = candidates[best];
    const float norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    const float scale = norm2 != 0.0f ? 1.0f / std::sqrt(norm2) : 0.0f;
    for (int k = 0; k < 4; k++) out->v[k] = q[k] * scale;
}

// --- Vibración: dispositivos válidos, sin motor detrás ---

// Como el SDK: un estilo cada vez; Pro, portátil y Joy-Con en dual tienen dos actuadores (izquierdo y derecho).
int GetVibrationDeviceHandles(VibrationDeviceHandle* out, int count, const NpadIdType& id, NpadStyleSet style_set) {
    const uint32_t style = style_set.storage[0];
    if (count < 1 || !valid_id(id) || __builtin_popcount(style) != 1) return 0;
    uint32_t type;
    if (style & kStyleFullKey)
        type = 3;
    else if (style & kStyleHandheld)
        type = 4;
    else if (style & kStyleJoyDual)
        type = 5;
    else if (style & kStyleJoyLeft) {
        out[0].storage = make_handle(6, id, 0);
        return 1;
    } else if (style & kStyleJoyRight) {
        out[0].storage = make_handle(7, id, 1);
        return 1;
    } else if (style & kStyleGc) {
        out[0].storage = make_handle(8, id, 0);
        return 1;
    } else
        return 0;
    out[0].storage = make_handle(type, id, 0);
    if (count < 2) return 1;
    out[1].storage = make_handle(type, id, 1);
    return 2;
}

void GetVibrationDeviceInfo(VibrationDeviceInfo* out, const VibrationDeviceHandle& handle) {
    const uint32_t type = handle.storage & 0xFF, index = handle.storage >> 16 & 0xFF;
    *out = {};
    if (type >= 3 && type <= 7) {
        out->device_type = 1;
        out->position = index == 0 ? 1 : 2;
    } else if (type == 8) {
        out->device_type = 2;
    }
}

void InitializeVibrationDevice(const VibrationDeviceHandle&) {}

}  // namespace nn::hid
