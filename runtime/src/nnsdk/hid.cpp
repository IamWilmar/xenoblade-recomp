// nn::hid: la Flip es una consola portátil, así que el juego ve un solo mando, en modo portátil (NpadIdType Handheld =
// 0x20, estilo Handheld) con los botones y sticks que traduce el loader (host.h). MH Rise recorre No1..No8 y Handheld
// con GetNpadStyleSet y crea un mando por cada id conectado (main+0x6f8fa30): con el portátil como único conectado, la
// Flip es el jugador 1 y no salta el applet de "conecta un mando".
//
// Lo que la Flip no tiene responde como una Switch sin ello: ratón y teclado USB desconectados (estados con los
// atributos a 0), pantalla táctil sin toques ni gestos, y los sensores de movimiento del portátil quietos. La vibración
// de los dos actuadores del portátil va al único motor de la consola (el vibrador de Android, que mueve el loader:
// host.h); el grafo de VibrationNode y los ficheros .bnvib están en hid_vibration.cpp.
//
// Disco Elysium llega aquí por tres caminos (work/scope/main.elf): el gestor de entrada clásico de Unity
// (main+0x916cd8, cada frame: GetNpadStyleSet de sus 9 ids y GetNpadStates de los estilos que tenga cada uno), el
// plugin de C# NintendoSDK (envoltorios en main+0x155b960..0x155c600) que usa InControl
// (InControl.NintendoSwitchInputDevice, main+0x2b08730..0x2b0b780), y el sistema de entrada nuevo de Unity
// (main+0x91a3ec), que este juego no llega a construir (en los logs de la consola solo hay una configuración nativa
// de mandos, la del clásico, y la de InControl). El plugin trae todo nn::hid aunque el juego use una parte: lo que
// nadie llama está hecho como lo hace el sdk del juego (work/scope/sdk.elf), y lo que se puede comparar con él se
// compara en tools/tests/sdk_puro_test.cpp.

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/os.h"
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
// Cuándo cuenta el portátil como mando: con los dos Joy-Con puestos, con uno, o siempre. Lo que el servicio hid deja
// leer de vuelta son esos tres valores (sdk+0x215440).
enum NpadHandheldActivationMode : int {
    NpadHandheldActivationMode_Dual = 0,
    NpadHandheldActivationMode_Single = 1,
    NpadHandheldActivationMode_None = 2,
};
// Cada cuánto habla la consola con los mandos inalámbricos (sdk+0x215584: 0..3 tal cual).
enum NpadCommunicationMode : int {
    NpadCommunicationMode_5ms = 0,
    NpadCommunicationMode_10ms = 1,
    NpadCommunicationMode_15ms = 2,
    NpadCommunicationMode_Default = 3,
};
enum NpadJoyDeviceType : int {};
enum NpadJoyAssignmentMode : int { NpadJoyAssignmentMode_Dual = 0, NpadJoyAssignmentMode_Single = 1 };
enum GyroscopeZeroDriftMode : int {
    GyroscopeZeroDriftMode_Loose = 0,
    GyroscopeZeroDriftMode_Standard = 1,
    GyroscopeZeroDriftMode_Tight = 2,
};

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
// Amplitud (0..1) y frecuencia (Hz) de las dos bandas de un actuador lineal. El plugin la pasa en s0..s3 y su
// envoltorio la baja a la pila (main+0x155c4b8).
struct VibrationValue {
    float amplitude_low, frequency_low, amplitude_high, frequency_high;
};

// Colores del mando (nn::util::Color4u8: r, g, b, a).
struct NpadControllerColor {
    uint8_t main[4];
    uint8_t sub[4];
};

// Lo que se le pasa al applet de mandos, 0x430 bytes en el sdk del juego (los que manda al applet, sdk+0x244808):
// +0 jugadores mínimos, +1 máximos, +2 quitar la conexión a otro, +3 justificar a la izquierda, +4 permitir Joy-Con en
// dual, +5 un solo jugador, +6 colores propios, +7 los 8 colores, +0x27 textos propios, +0x28 los 8 textos de 129
// bytes. El plugin de C# es de un SDK anterior y solo arma 0x21c (4 jugadores, main+0x1cd2b74): de él no se lee más
// allá de los primeros bytes.
struct ControllerSupportArg {
    uint8_t storage[0x430];
    void SetDefault();
};
constexpr size_t kExplainTextOffset = 0x28, kExplainTextSize = 0x81;
struct ControllerSupportResultInfo {
    int8_t player_count;
    char pad[3];
    NpadIdType selected_id;
};
struct ControllerFirmwareUpdateArg {
    bool enable_force_update;
    uint8_t reserved[3];
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
// El mando Pro: conectado y con cable (va por USB o Bluetooth, pero así no pide batería ni emparejar).
constexpr uint32_t kProAttributes = 0x3;

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

// Dónde está el mando de la Flip: el portátil, o en modo TV (nnsdk::tv_mode) un mando Pro en No1, que es lo que ve el
// juego en una consola en la base (el portátil no existe ahí). Xenoblade acepta 0, 1, 2 y Handheld con estilos 0x7.
NpadIdType pad_id() { return nnsdk::tv_mode() ? 0 : kNpadHandheld; }
uint32_t pad_style() { return nnsdk::tv_mode() ? kStyleFullKey : kStyleHandheld; }
// Tipo de los handles de sensor y vibración del mando (índice del estilo + 3): 3 el Pro, 4 el portátil.
uint32_t pad_handle_type() { return nnsdk::tv_mode() ? 3 : 4; }

uint32_t style_of(NpadIdType id) { return id == pad_id() && handheld_connected() ? pad_style() : 0; }

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
    // DEPURACIÓN (<datos>/traza_sticks, en vivo): los dos sticks pulsados a la vez (L3+R3) piden la traza completa del frame
    // siguiente, para efectos que duran poco (se disparan jugando). Como mucho una cada 20 s; el juego se para mientras se
    // vuelca. Sin el fichero no hace nada: pulsar los dos sticks es un gesto normal de juego.
    constexpr uint64_t kBothSticks = (1 << 4) | (1 << 5);
    static int64_t last_trace_ns = 0;
    if ((buttons & kBothSticks) == kBothSticks && NVN_FLAG("traza_sticks")) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const int64_t now = int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec;
        if (now - last_trace_ns > 20'000'000'000) {
            last_trace_ns = now;
            printf("[hid] L3+R3: traza completa del frame siguiente\n");
            nvn::request_trace(true);
        }
    }
    static uint64_t last = 0;  // DEPURACIÓN: qué ve el juego cuando cambian los botones
    if (buttons != last) {
        static int logged = 0;
        if (nvn::first_times(logged, 60)) printf("[hid] botones 0x%llx\n", static_cast<unsigned long long>(buttons));
        last = buttons;
    }
    out->lx = in->sticks[0], out->ly = in->sticks[1], out->rx = in->sticks[2], out->ry = in->sticks[3];
    out->attributes = nnsdk::tv_mode() ? kProAttributes : kHandheldAttributes;
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

// Lo que el juego le deja dicho al servicio hid y puede leer de vuelta. El portátil se activa con los dos Joy-Con y
// la comunicación va en su modo por defecto hasta que el juego diga otra cosa.
std::atomic<uint32_t> g_supported_styles{0};
std::atomic<int> g_handheld_activation{NpadHandheldActivationMode_Dual};
std::atomic<int> g_communication_mode{NpadCommunicationMode_Default};

// Historial de un estilo (hasta 16 muestras, la más nueva primero): aquí solo la actual, con su número de muestra
// nuevo. Unity recorre las que sean más nuevas que la última que vio (main+0x917804) y con una le basta.
int fill_states(void* out, int count, NpadIdType id, uint32_t style) {
    if (count < 1) return 0;
    fill_state(out, id, style);
    return 1;
}

// El applet de mandos sale cancelado si el jugador lo cierra sin completar lo que pide el juego (sdk+0x2447b0:
// 0x183aca; con +0x200, estilo no admitido).
constexpr Result kResultControllerSupportCanceled{202 | 3101 << 9};

// El applet de mandos, sin applet: la Flip solo tiene su mando, ya conectado como portátil, así que responde lo que
// respondería la consola cuando el jugador confirma con él. Si el juego exige más de un jugador no hay con qué
// completarlo y la única salida del applet es cancelar. Del argumento solo se miran los primeros bytes (ver
// ControllerSupportArg).
Result controller_support(ControllerSupportResultInfo* out, const ControllerSupportArg& arg) {
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 8))
        printf("[hid] ShowControllerSupport: de %u a %u jugadores, un solo jugador %u\n", arg.storage[0], arg.storage[1],
               arg.storage[5]);
    if (arg.storage[0] > 1) return kResultControllerSupportCanceled;
    if (out) {
        out->player_count = 1;
        out->selected_id = pad_id();
    }
    return ResultSuccess;
}

// --- Sensores de movimiento: lo que el servicio hid guarda por sensor ---

// La fusión del acelerómetro con el giroscopio (encendida), sus dos parámetros y la corrección de deriva del
// giroscopio (la normal). Los valores de fábrica no están en el sdk del juego (los pone el servicio): son los que
// documentan los emuladores, 0,03 y 0,4.
struct SixAxisSettings {
    bool fusion = true;
    float revise_power = 0.03f, revise_range = 0.4f;
    int drift_mode = GyroscopeZeroDriftMode_Standard;
};
std::mutex g_six_axis_m;
std::unordered_map<uint32_t, SixAxisSettings> g_six_axis;

template <typename F>
auto with_six_axis(const SixAxisSensorHandle& handle, F&& f) {
    std::lock_guard lock(g_six_axis_m);
    return f(g_six_axis[handle.storage]);
}

// --- Vibración: los dos actuadores del portátil sobre el único motor de la consola ---

// Lo que vale un actuador que no vibra: amplitud 0 a 160 y 320 Hz (VibrationValue.Make del plugin, main+0x1ce3280).
constexpr VibrationValue kNoVibration{0.0f, 160.0f, 0.0f, 320.0f};
std::mutex g_vibration_m;
VibrationValue g_actuators[2] = {kNoVibration, kNoVibration};  // izquierdo y derecho del portátil

// <datos>/sin_vibracion: como la opción de la consola "vibración de los mandos: no". El loader tampoco arranca su
// hilo del vibrador si el fichero está al abrir el juego.
bool vibration_off() { return NVN_FLAG("sin_vibracion"); }

// Qué actuador del portátil nombra un handle (ver GetVibrationDeviceHandles), o -1 si es de un hueco sin mando.
int actuator_of(const VibrationDeviceHandle& handle) {
    const uint32_t type = handle.storage & 0xFF, id = handle.storage >> 8 & 0xFF, index = handle.storage >> 16 & 0xFF;
    return type == pad_handle_type() && id == pad_id() && index < 2 && handheld_connected() ? static_cast<int>(index) : -1;
}

// La consola tiene un solo motor para los dos actuadores: va con el que más pida. Las dos bandas de un actuador
// mueven la misma masa, así que su fuerza es la suma de las amplitudes (hasta 1). La amplitud de Android sale de la
// raíz: el juego manda toques de 0,02 a 0,14 (InControl da a cada banda entre 0 y 0,08 según un seno y un coseno de
// 30 Hz, main+0x2b0a0a0) y en lineal se quedarían en 6-35 de 255, por debajo de lo que arranca un motor de móvil.
// Sin probar en la consola: si se nota poco o demasiado, es esta curva.
void drive_motor() {
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
    NnsdkInput* in = nnsdk::host().input;
    if (!in) return;
    auto unit = [](float a) { return a > 0.0f ? (a < 1.0f ? a : 1.0f) : 0.0f; };  // un NaN cuenta como 0
    auto strength = [&](const VibrationValue& v) {
        const float sum = unit(v.amplitude_low) + unit(v.amplitude_high);
        return sum < 1.0f ? sum : 1.0f;
    };
    const float s = std::fmax(strength(g_actuators[0]), strength(g_actuators[1]));
    const long scaled = std::lround(255.0f * std::sqrt(s));
    const uint32_t amplitude = s > 0.0f ? static_cast<uint32_t>(scaled < 1 ? 1 : scaled > 255 ? 255 : scaled) : 0;
    if (in->rumble.exchange(amplitude) == amplitude) return;
    in->rumble_serial.fetch_add(1);
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&in->rumble_serial), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 24)) printf("[hid] vibración: motor a %u/255\n", amplitude);
}

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
// No se hace caso: en una Switch eso desconectaría el portátil, y la Flip no tiene otro mando que ofrecer. Se guarda
// para devolverlo: Disco Elysium pide 7 (Pro, portátil y Joy-Con en dual) desde Unity (main+0x916b28) y desde
// InControl (main+0x2b0b864).
void SetSupportedNpadStyleSet(NpadStyleSet styles) {
    g_supported_styles = styles.storage[0];
    printf("[hid] SetSupportedNpadStyleSet %#x\n", styles.storage[0]);
}
NpadStyleSet GetSupportedNpadStyleSet() { return {{g_supported_styles.load()}}; }
void SetNpadJoyHoldType(NpadJoyHoldType) {}

// Cuándo se activa el portátil y el ritmo de la comunicación con los mandos: la Flip es siempre un portátil entero
// y no tiene mandos inalámbricos, así que no cambian nada; se guardan porque el juego los puede leer de vuelta
// (InControl deja el portátil en Dual, main+0x2b0b870).
void SetNpadHandheldActivationMode(NpadHandheldActivationMode mode) { g_handheld_activation = mode; }
NpadHandheldActivationMode GetNpadHandheldActivationMode() {
    return static_cast<NpadHandheldActivationMode>(g_handheld_activation.load());
}
void SetNpadCommunicationMode(NpadCommunicationMode mode) { g_communication_mode = mode; }
NpadCommunicationMode GetNpadCommunicationMode() {
    return static_cast<NpadCommunicationMode>(g_communication_mode.load());
}

// El evento que el servicio hid señala cuando cambia el estilo de un hueco (se conecta o se va un mando). Nace
// señalado: el sistema de entrada nuevo de Unity solo mira GetNpadStyleSet cuando lo ve señalado (main+0x919c64), y
// sin ese primer aviso nunca vería el mando que ya estaba conectado al arrancar. Después no vuelve a señalarse: el
// mando de la Flip ni se va ni cambia de estilo. El plugin de C# lo pide con AutoClear (main+0x155ba00) y Unity con
// ManualClear (main+0x91a5a8).
void BindNpadStyleSetUpdateEvent(const NpadIdType& id, os::SystemEventType* event, os::EventClearMode mode) {
    nnsdk::create_system_event(event, mode == os::EventClearMode_AutoClear, /*signaled=*/true);
    printf("[hid] BindNpadStyleSetUpdateEvent id %#x evento %p (%s)\n", id, static_cast<void*>(event),
           mode == os::EventClearMode_AutoClear ? "AutoClear" : "ManualClear");
}

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

// Unity (Disco Elysium) lee cada frame el historial (16 muestras de 0x28 bytes) de cada estilo que GetNpadStyleSet le
// dio para el hueco: una función por estilo, cada una con su bit (main+0x917774 Pro, +0x917888 dual, +0x91799c
// portátil, +0x917ab0 y +0x917bc4 Joy-Con sueltos). Devuelve cuántas escribió. Como el único estilo que hay es el
// portátil, las otras cuatro solo las alcanza quien pregunte sin mirar el estilo, y le dan lo mismo que GetNpadState
// por un estilo que el mando no tiene: una muestra nueva, sin botones y sin el atributo de conectado.
int GetNpadStates(NpadFullKeyState* out, int count, const NpadIdType& id) { return fill_states(out, count, id, kStyleFullKey); }
int GetNpadStates(NpadHandheldState* out, int count, const NpadIdType& id) { return fill_states(out, count, id, kStyleHandheld); }
int GetNpadStates(NpadJoyDualState* out, int count, const NpadIdType& id) { return fill_states(out, count, id, kStyleJoyDual); }
int GetNpadStates(NpadJoyLeftState* out, int count, const NpadIdType& id) { return fill_states(out, count, id, kStyleJoyLeft); }
int GetNpadStates(NpadJoyRightState* out, int count, const NpadIdType& id) { return fill_states(out, count, id, kStyleJoyRight); }

// Agarre: horizontal (los Joy-Con en horizontal no aplican al portátil, que es lo único que hay).
NpadJoyHoldType GetNpadJoyHoldType() { return static_cast<NpadJoyHoldType>(1); }

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
// La variante que además dice a qué hueco se fue el otro Joy-Con del par (true si lo movió, sdk+0x222324). Aquí nunca
// hay un par que separar: no se mueve ninguno. Qué deja el servicio en *out en ese caso no se sabe; se deja el mismo
// hueco para que no quede basura.
bool SetNpadJoyAssignmentModeSingle(NpadIdType* out, const NpadIdType& id, NpadJoyDeviceType) {
    SetNpadJoyAssignmentModeSingle(id);
    if (out) *out = id;
    return false;
}
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

// Applet de selección de mando: el portátil ya está conectado como jugador 1. InControl solo lo abre si un hueco
// trae el estilo 0x20 o un par de Joy-Con al que le falta una mitad (main+0x2b0a290 y +0x2b0a378), que aquí no pasa;
// el juego lo prepara para un jugador (NXapiControl.Awake, main+0x1abaf80).
Result ShowControllerSupport(const ControllerSupportArg& arg) { return controller_support(nullptr, arg); }
Result ShowControllerSupport(ControllerSupportResultInfo* out, const ControllerSupportArg& arg) {
    return controller_support(out, arg);
}

// El texto que el applet pone junto a un jugador. Como el sdk (sdk+0x244bf0): hasta 128 caracteres y el terminador
// en +0x28 + 129·jugador para No1..No8, y nada para el portátil ni otros ids. Solo se guarda: no hay applet que lo
// enseñe.
void SetExplainText(ControllerSupportArg* arg, const char* text, const NpadIdType& id) {
    if (!arg || !text || id >= 8) return;
    char* dst = reinterpret_cast<char*>(arg->storage) + kExplainTextOffset + id * kExplainTextSize;
    size_t n = 0;
    for (; n < kExplainTextSize - 1 && text[n]; n++) dst[n] = text[n];
    dst[n] = 0;
}

// La pantalla que enseña a ponerse la correa de los Joy-Con: la Flip no tiene Joy-Con ni correa, así que no hay nada
// que enseñar y vuelve como cuando el jugador la cierra.
Result ShowControllerStrapGuide() { return ResultSuccess; }

// El applet que actualiza el firmware de los mandos conectados: sin Joy-Con ni mando Pro no hay nada que actualizar
// y vuelve con éxito (el único fallo que saca el sdk, 202-3201, es que el applet no acabe bien: sdk+0x244980).
Result ShowControllerFirmwareUpdate(const ControllerFirmwareUpdateArg&) { return ResultSuccess; }

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
// Unity (Disco Elysium) pide todas las instancias de 1 a 16 toques.
template <size_t N>
int GetTouchScreenStates(TouchScreenState<N>* out, int count) {
    if (count < 1) return 0;
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
    return 1;
}
template <size_t N>
void GetTouchScreenState(TouchScreenState<N>* out) {
    GetTouchScreenStates(out, 1);
}
#define NN_HID_TOUCH(n)                                                       \
    template int GetTouchScreenStates<n>(TouchScreenState<n>*, int); \
    template void GetTouchScreenState<n>(TouchScreenState<n>*);
NN_HID_TOUCH(1) NN_HID_TOUCH(2) NN_HID_TOUCH(3) NN_HID_TOUCH(4) NN_HID_TOUCH(5) NN_HID_TOUCH(6) NN_HID_TOUCH(7)
NN_HID_TOUCH(8) NN_HID_TOUCH(9) NN_HID_TOUCH(10) NN_HID_TOUCH(11) NN_HID_TOUCH(12) NN_HID_TOUCH(13) NN_HID_TOUCH(14)
NN_HID_TOUCH(15) NN_HID_TOUCH(16)
#undef NN_HID_TOUCH

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

// Con varios estilos a la vez manda el primero de la lista del sdk (sdk+0x22a4fc): Pro, portátil, dual, izquierdo,
// derecho y, el último, el mando de GameCube, que usa el handle del Pro.
int GetSixAxisSensorHandles(SixAxisSensorHandle* out, int count, const NpadIdType& id, NpadStyleSet style_set) {
    const uint32_t style = style_set.storage[0];
    if (count < 1 || !valid_id(id)) return 0;
    if (style & kStyleFullKey) {
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
    if (style & kStyleGc) {
        out[0].storage = make_handle(3, id, 2);
        return 1;
    }
    return 0;
}

void StartSixAxisSensor(const SixAxisSensorHandle&) {}
void StopSixAxisSensor(const SixAxisSensorHandle&) {}

// Ajustes del sensor: no hay sensor al que aplicarlos (el estado es siempre el de reposo), pero el servicio los
// guarda por handle y el juego los puede leer de vuelta.
void SetGyroscopeZeroDriftMode(const SixAxisSensorHandle& handle, const GyroscopeZeroDriftMode& mode) {
    with_six_axis(handle, [&](SixAxisSettings& s) { s.drift_mode = mode; });
}
GyroscopeZeroDriftMode GetGyroscopeZeroDriftMode(const SixAxisSensorHandle& handle) {
    return with_six_axis(handle, [](SixAxisSettings& s) { return static_cast<GyroscopeZeroDriftMode>(s.drift_mode); });
}
void EnableSixAxisSensorFusion(const SixAxisSensorHandle& handle, bool enable) {
    with_six_axis(handle, [&](SixAxisSettings& s) { s.fusion = enable; });
}
bool IsSixAxisSensorFusionEnabled(const SixAxisSensorHandle& handle) {
    return with_six_axis(handle, [](SixAxisSettings& s) { return s.fusion; });
}
void SetSixAxisSensorFusionParameters(const SixAxisSensorHandle& handle, float revise_power, float revise_range) {
    with_six_axis(handle, [&](SixAxisSettings& s) { s.revise_power = revise_power, s.revise_range = revise_range; });
}
void GetSixAxisSensorFusionParameters(float* revise_power, float* revise_range, const SixAxisSensorHandle& handle) {
    const SixAxisSettings s = with_six_axis(handle, [](SixAxisSettings& v) { return v; });
    if (revise_power) *revise_power = s.revise_power;
    if (revise_range) *revise_range = s.revise_range;
}
void ResetSixAxisSensorFusionParameters(const SixAxisSensorHandle& handle) {
    with_six_axis(handle, [](SixAxisSettings& s) {
        const SixAxisSettings factory;
        s.revise_power = factory.revise_power, s.revise_range = factory.revise_range;
    });
}

// Sin sensor nada se mueve: siempre en reposo, que es lo que el juego miraría para recalibrar.
bool IsSixAxisSensorAtRest(const SixAxisSensorHandle&) { return true; }

// El aviso de "hay firmware nuevo para el sensor de este mando": no hay mando externo que actualizar.
bool IsFirmwareUpdateAvailableForSixAxisSensor(const SixAxisSensorHandle&) { return false; }

// En reposo y plano: solo la gravedad (-1 G en z), sin giro y con los ejes del mando alineados con los del mundo. Así
// GetQuaternion da la identidad y nada que use el giroscopio se mueve solo.
void GetSixAxisSensorState(SixAxisSensorState* out, const SixAxisSensorHandle& handle) {
    memset(out, 0, sizeof *out);
    out->sampling_number = sample_clock();
    out->delta_time = 5'000'000;
    if ((handle.storage >> 8 & 0xFF) != pad_id() || !handheld_connected()) return;
    out->acceleration[2] = -1.0f;
    out->direction[0] = out->direction[4] = out->direction[8] = 1.0f;
    out->attributes = 1;
}

// El historial del sensor (muestras de 0x60 bytes, la más nueva primero): solo la actual, como el del mando. El
// sistema de entrada nuevo de Unity pide 16 y se queda con el número de muestra más alto (main+0x919b50).
int GetSixAxisSensorStates(SixAxisSensorState* out, int count, const SixAxisSensorHandle& handle) {
    if (count < 1) return 0;
    GetSixAxisSensorState(out, handle);
    return 1;
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

// --- Vibración ---

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

// El valor se queda puesto en el actuador hasta el siguiente: para parar, el juego manda uno a cero
// (CrossPlatformInputManager.ResetRumbleController, main+0x16eb940). Disco Elysium vibra en un solo sitio: al cambiar
// de objetivo andando con el stick, 0,1 en los dos motores durante 0,2 s (CharacterAnalogueControl.
// DoChangedTargetEffects, main+0x3227e14 → CrossPlatformInputManager.RumbleController, main+0x16eb770, que lo baja a
// 0,08 y programa la vuelta a cero), por InControl.NintendoSwitchInputDevice.Vibrate (main+0x2b0a0a0) y el envoltorio
// del plugin (main+0x155c4c0). Lo que va a un hueco sin mando no tiene quién lo reciba.
void SendVibrationValue(const VibrationDeviceHandle& handle, const VibrationValue& value) {
    const int actuator = actuator_of(handle);
    if (actuator < 0) return;
    std::lock_guard lock(g_vibration_m);
    g_actuators[actuator] = value;
    if (vibration_off()) g_actuators[actuator].amplitude_low = g_actuators[actuator].amplitude_high = 0.0f;
    drive_motor();
}

// Lo que el actuador está haciendo de verdad: lo último que se le mandó (con la vibración apagada, quieto).
void GetActualVibrationValue(VibrationValue* out, const VibrationDeviceHandle& handle) {
    const int actuator = actuator_of(handle);
    std::lock_guard lock(g_vibration_m);
    *out = actuator < 0 ? kNoVibration : g_actuators[actuator];
}

// La opción de la consola "vibración de los mandos".
bool IsVibrationPermitted() { return !vibration_off(); }

}  // namespace nn::hid
