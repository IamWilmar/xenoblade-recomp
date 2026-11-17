// nn::irsensor: la cámara de infrarrojos del Joy-Con derecho. La Flip no tiene, así que responde como una consola
// cuyo mando integrado no la lleva (una Switch Lite): la cámara del portátil está "sin cámara" y la de los demás
// huecos "sin mando"; los procesadores de imagen se pueden lanzar y parar (lo que se pide queda apuntado aunque no haya
// cámara que lo atienda), y leer de ellos no da muestras sino el error del SDK para ese estado, nunca un éxito vacío.
//
// Disco Elysium no la usa. Sus 24 imports son del plugin de C# NintendoSDK, que trae todo el SDK: los envoltorios
// (main+0x1559fa0..0x155a5b4 y main+0x155c610..0x155c7e4) solo los llaman los métodos nn.irsensor.* del propio plugin
// (main+0x1cd0ba0..0x1cd7d20), y a esos no los llama ningún otro método del juego (todos los bl de main.dis). Por eso
// está hecho a partir del sdk del juego (work/scope/sdk.elf), que resuelve dentro del proceso casi todo lo que aquí
// importa: los handles, las configuraciones por defecto, el estado por defecto que deja aunque la lectura falle, el
// error según el estado de la cámara (tabla en sdk+0xa83fc8) y las cuentas sobre los momentos. Lo que en la consola
// decide el servicio irs, y por tanto no sale del sdk, se dice en cada sitio.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::irsensor {

// El jugador (0..7 para No1..No8, 8 el portátil, 9 "otros") y el tipo de dispositivo (0, un mando), como los arma el
// sdk (sdk+0x2b04e0). Se devuelve por valor, en w0 (main+0x155c724).
struct IrCameraHandle {
    uint8_t player_number;
    uint8_t device_type;
    uint8_t reserved[2];
};
static_assert(sizeof(IrCameraHandle) == 4);

enum IrCameraStatus : int {
    IrCameraStatus_Available = 0,
    IrCameraStatus_Unsupported = 1,  // el mando no lleva cámara
    IrCameraStatus_Unconnected = 2,  // no hay mando
};
enum ImageProcessorStatus : int { ImageProcessorStatus_Stopped = 0, ImageProcessorStatus_Running = 1 };

struct Rect {
    int16_t x, y, width, height;
};

// Lo común a todos los procesadores: exposición en ns, a qué apuntan los LED, ganancia y si la imagen va en negativo.
struct IrCameraConfig {
    int64_t exposure_time_ns;
    int32_t light_target;
    int32_t gain;
    bool is_negative_image_used;
};
static_assert(sizeof(IrCameraConfig) == 0x18);

struct MomentProcessorConfig {
    IrCameraConfig camera;
    Rect window_of_interest;
    int32_t preprocess;
    int32_t preprocess_intensity_threshold;
};
static_assert(sizeof(MomentProcessorConfig) == 0x28);

struct ClusteringProcessorConfig {
    IrCameraConfig camera;
    Rect window_of_interest;
    int32_t object_pixel_count_min;
    int32_t object_pixel_count_max;
    int32_t object_intensity_min;
    bool is_external_light_filter_enabled;
};
static_assert(sizeof(ClusteringProcessorConfig) == 0x30);

struct ImageTransferProcessorConfig {
    IrCameraConfig camera;
    int32_t format;
};
static_assert(sizeof(ImageTransferProcessorConfig) == 0x20);

struct ImageTransferProcessorExConfig {
    IrCameraConfig camera;
    int32_t orig_format;
    int32_t trimming_format;
    int16_t trimming_start_x, trimming_start_y;
    bool is_external_light_filter_enabled;
};
static_assert(sizeof(ImageTransferProcessorExConfig) == 0x28);

// El modo del análisis de manos (1..4: silueta, imagen, las dos, silueta sola).
struct HandAnalysisConfig {
    int32_t mode;
};

// El procesador de momentos parte su ventana en 6 filas de 8 bloques y da de cada uno la intensidad media y el
// centroide.
struct MomentStatistic {
    float average_intensity;
    float centroid_x, centroid_y;
};
constexpr int kMomentRows = 6, kMomentColumns = 8;
struct MomentProcessorState {
    int64_t sampling_number;
    int64_t time_stamp_ns;
    int32_t ambient_noise_level;
    int32_t reserved;
    MomentStatistic blocks[kMomentRows * kMomentColumns];
};
static_assert(sizeof(MomentProcessorState) == 0x258);

// Solo la cabecera: lo único que se escribe sin cámara (detrás van hasta 16 objetos que nadie rellena).
struct ClusteringProcessorState {
    int64_t sampling_number;
    int64_t time_stamp_ns;
    int8_t object_count;
    int32_t ambient_noise_level;
};
static_assert(sizeof(ClusteringProcessorState) == 0x18);

struct ImageTransferProcessorState;
struct HandAnalysisSilhouetteState;
struct HandAnalysisImageState;

}  // namespace nn::irsensor

using namespace nn;
using namespace nn::irsensor;

namespace {

constexpr int kPlayers = 10;  // No1..No8, el portátil y "otros"
constexpr int kHandheldPlayer = 8;
constexpr int kAmbientNoiseUnknown = 3;  // IrCameraAmbientNoiseLevel: lo que pone el sdk en un estado sin muestra

// Errores del módulo irsensor (205) como los reparte el sdk: el de leer con la cámara en cada estado (sdk+0xa83fc8:
// 0x140cd disponible pero sin muestras aún, 0xdecd sin cámara, 0xdccd sin mando) y el de leer de un procesador que no
// es el que está lanzado (sdk+0x2b10ac). El de modo incorrecto del análisis de manos es el 1101 (main+0x1cd7a80).
constexpr Result kResultUnconnected{205 | 110 << 9};
constexpr Result kResultUnsupported{205 | 111 << 9};
constexpr Result kResultNotReady{205 | 160 << 9};
constexpr Result kResultHandAnalysisModeIncorrect{205 | 1101 << 9};

// Qué procesador se lanzó para una cámara, con los números que usa la sesión del sdk (sesión+0x88), y lo que guarda de
// su configuración para el estado por defecto (sdk+0x2b41e0): si la imagen va en negativo y la ventana.
enum Mode : int { kNone = 0, kMoment = 1, kClustering = 2, kImageTransfer = 3, kTeraPlugin = 5 };
struct Camera {
    int mode = kNone;
    bool negative = false;
    Rect window{};
};
std::mutex g_m;
Camera g_cameras[kPlayers];

// nullptr con un handle que GetIrCameraHandle no da.
Camera* camera_of(const IrCameraHandle& handle) {
    return handle.player_number < kPlayers ? &g_cameras[handle.player_number] : nullptr;
}

// La Flip es el portátil y su mando no lleva cámara; en los demás huecos no hay mando.
IrCameraStatus status_of(const IrCameraHandle& handle) {
    return handle.player_number == kHandheldPlayer && nnsdk::host().input ? IrCameraStatus_Unsupported
                                                                           : IrCameraStatus_Unconnected;
}

// Lo que devuelve el sdk cuando el procesador lanzado no tiene muestras: depende de por qué (sdk+0x2b1404).
Result no_data(const IrCameraHandle& handle) {
    return status_of(handle) == IrCameraStatus_Unsupported ? kResultUnsupported : kResultUnconnected;
}

void run(const IrCameraHandle& handle, Mode mode, const IrCameraConfig* config, const Rect* window, const char* what) {
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 8))
        printf("[irsensor] %s en el jugador %u: apuntado, pero no hay cámara de infrarrojos\n", what, handle.player_number);
    std::lock_guard lock(g_m);
    Camera* camera = camera_of(handle);
    if (!camera) return;
    camera->mode = mode;
    if (config) camera->negative = config->is_negative_image_used;
    if (window) camera->window = *window;
}

// El estado de momentos que el sdk deja en la primera posición antes de mirar nada más, y que se queda si la lectura
// falla (sdk+0x2b4874): muestra 0, ruido ambiente desconocido y los 48 bloques a cero; con la imagen en negativo, cada
// bloque blanco (255) con el centroide en su centro.
void default_moment_state(MomentProcessorState* state, const Camera* camera) {
    state->sampling_number = 0;
    state->time_stamp_ns = 0;
    state->ambient_noise_level = kAmbientNoiseUnknown;
    const bool negative = camera && camera->negative;
    const int block_width = camera ? camera->window.width / kMomentColumns : 0;
    const int block_height = camera ? camera->window.height / kMomentRows : 0;
    for (int row = 0; row < kMomentRows; row++) {
        for (int column = 0; column < kMomentColumns; column++) {
            MomentStatistic& block = state->blocks[row * kMomentColumns + column];
            block.average_intensity = negative ? 255.0f : 0.0f;
            block.centroid_x = negative ? float((2 * column + 1) * block_width) * 0.5f : 0.0f;
            block.centroid_y = negative ? float((2 * row + 1) * block_height) * 0.5f : 0.0f;
        }
    }
}

// Lo mismo para el de agrupaciones (sdk+0x2b5ec8): ningún objeto.
void default_clustering_state(ClusteringProcessorState* state) {
    state->sampling_number = 0;
    state->time_stamp_ns = 0;
    state->object_count = 0;
    state->ambient_noise_level = kAmbientNoiseUnknown;
}

// Leer de un procesador: sin el suyo lanzado, "aún no hay nada" (sdk+0x2b111c); con él lanzado, que no haya muestras
// es culpa de la cámara.
Result read_result(const IrCameraHandle& handle, Mode mode) {
    std::lock_guard lock(g_m);
    const Camera* camera = camera_of(handle);
    return camera && camera->mode == mode ? no_data(handle) : kResultNotReady;
}

}  // namespace

namespace nn::irsensor {

// --- La cámara ---

// Como el sdk (sdk+0x2b04e0): el jugador del hueco, 8 para el portátil y 9 para "otros".
IrCameraHandle GetIrCameraHandle(const uint32_t& id) {
    return {static_cast<uint8_t>(id < 8 ? id : id == 0x10 ? 9 : kHandheldPlayer), 0, {}};
}

// En la consola enciende la sesión con el servicio; aquí no hay nada que encender. Se anota porque sería la primera
// señal de que algo del juego usa la cámara.
void Initialize(const IrCameraHandle& handle) {
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 8))
        printf("[irsensor] Initialize del jugador %u: la Flip no tiene cámara de infrarrojos (estado %d)\n",
               handle.player_number, status_of(handle));
}

// Como el sdk (sdk+0x2b06d0), para lo que estuviera lanzado.
void Finalize(const IrCameraHandle& handle) {
    std::lock_guard lock(g_m);
    if (Camera* camera = camera_of(handle)) camera->mode = kNone;
}

IrCameraStatus GetIrCameraStatus(const IrCameraHandle& handle) { return status_of(handle); }

// Si el Joy-Con necesita firmware nuevo para la cámara. Sin cámara o sin mando el sdk no llega a preguntarlo: sale
// con el error del estado y no toca *necessary (sdk+0x2b0b5c).
Result CheckFirmwareUpdateNecessity(bool*, const IrCameraHandle& handle) { return no_data(handle); }

// --- Los procesadores de imagen ---

// En marcha desde que se lanza uno hasta que se para. En la consola lo dice el servicio (el sdk lo deduce de un
// estado interno de la memoria compartida, sdk+0x2b0a00); el sdk solo necesita que tras parar acabe leyéndose
// "parado", que espera hasta 5 s en StopImageProcessor (sdk+0x2b081c).
ImageProcessorStatus GetImageProcessorStatus(const IrCameraHandle& handle) {
    std::lock_guard lock(g_m);
    const Camera* camera = camera_of(handle);
    return camera && camera->mode != kNone ? ImageProcessorStatus_Running : ImageProcessorStatus_Stopped;
}

void StopImageProcessor(const IrCameraHandle& handle) {
    std::lock_guard lock(g_m);
    if (Camera* camera = camera_of(handle)) camera->mode = kNone;
}

// Momentos. Por defecto (sdk+0x2b6850): 300 µs de exposición, todos los LED, ganancia 8, imagen normal, la ventana
// entera (320x240) y recorte de lo que no llegue a 80 de intensidad. Como el sdk, no toca el relleno del struct.
void GetMomentProcessorDefaultConfig(MomentProcessorConfig* config) {
    config->camera.exposure_time_ns = 300'000;
    config->camera.light_target = 0;
    config->camera.gain = 8;
    config->camera.is_negative_image_used = false;
    config->window_of_interest = {0, 0, 320, 240};
    config->preprocess = 1;
    config->preprocess_intensity_threshold = 80;
}

void RunMomentProcessor(const IrCameraHandle& handle, const MomentProcessorConfig& config) {
    run(handle, kMoment, &config.camera, &config.window_of_interest, "RunMomentProcessor");
}

// Hasta 5 estados, el más nuevo primero. Como el sdk (sdk+0x2b10c0): cuenta a 0 y estado por defecto en el primero
// antes de nada.
Result GetMomentProcessorStates(MomentProcessorState* states, int* count, int, const IrCameraHandle& handle) {
    if (count) *count = 0;
    if (states) {
        std::lock_guard lock(g_m);
        default_moment_state(states, camera_of(handle));
    }
    return read_result(handle, kMoment);
}
Result GetMomentProcessorState(MomentProcessorState* state, const IrCameraHandle& handle) {
    return GetMomentProcessorStates(state, nullptr, 1, handle);
}

// Junta los bloques de un rectángulo de la rejilla de momentos en uno: intensidad media y centroide ponderado por la
// "masa" de cada bloque (intensidad por área). La misma cuenta que el sdk (sdk+0x2b6910), en su mismo orden y con sus
// mismas precisiones (la masa en float, las sumas en double); el sdk no mira si el rectángulo se sale de la rejilla
// (leería fuera del estado): aquí lo que se sale no cuenta.
MomentStatistic CalculateMomentRegionStatistic(const MomentProcessorState* state, const Rect& window, int start_row,
                                               int start_column, int row_count, int column_count) {
    const int block_width = window.width / kMomentColumns, block_height = window.height / kMomentRows;
    double mass = 0.0, moment_x = 0.0, moment_y = 0.0;
    if (row_count >= 1 && column_count >= 1) {
        const float width = static_cast<float>(block_width), height = static_cast<float>(block_height);
        for (int64_t row = start_row; row < int64_t{start_row} + row_count; row++) {
            for (int64_t column = start_column; column < int64_t{start_column} + column_count; column++) {
                if (row < 0 || row >= kMomentRows || column < 0 || column >= kMomentColumns) continue;
                const MomentStatistic& block = state->blocks[row * kMomentColumns + column];
                const float block_mass = block.average_intensity * width * height;
                moment_x += static_cast<double>(block.centroid_x * block_mass);
                mass += static_cast<double>(block_mass);
                moment_y += static_cast<double>(block.centroid_y * block_mass);
            }
        }
    }
    const int32_t area = static_cast<int32_t>(static_cast<uint32_t>(column_count) * static_cast<uint32_t>(row_count) *
                                              static_cast<uint32_t>(block_width) * static_cast<uint32_t>(block_height));
    MomentStatistic out;
    out.average_intensity = static_cast<float>(mass / static_cast<double>(area));
    out.centroid_x = mass == 0.0 ? 0.0f : static_cast<float>(moment_x / mass);
    out.centroid_y = mass == 0.0 ? 0.0f : static_cast<float>(moment_y / mass);
    return out;
}

// Agrupaciones (objetos brillantes). Por defecto (sdk+0x2b6a40): 200 µs, todos los LED, ganancia 2, imagen normal, la
// ventana entera, objetos de 3 a 76800 píxeles con intensidad de 150 o más y el filtro de luz exterior puesto.
void GetClusteringProcessorDefaultConfig(ClusteringProcessorConfig* config) {
    config->camera.exposure_time_ns = 200'000;
    config->camera.light_target = 0;
    config->camera.gain = 2;
    config->camera.is_negative_image_used = false;
    config->window_of_interest = {0, 0, 320, 240};
    config->object_pixel_count_min = 3;
    config->object_pixel_count_max = 320 * 240;
    config->object_intensity_min = 150;
    config->is_external_light_filter_enabled = true;
}

void RunClusteringProcessor(const IrCameraHandle& handle, const ClusteringProcessorConfig& config) {
    run(handle, kClustering, &config.camera, &config.window_of_interest, "RunClusteringProcessor");
}

Result GetClusteringProcessorStates(ClusteringProcessorState* states, int* count, int, const IrCameraHandle& handle) {
    if (count) *count = 0;
    if (states) default_clustering_state(states);
    return read_result(handle, kClustering);
}
Result GetClusteringProcessorState(ClusteringProcessorState* state, const IrCameraHandle& handle) {
    return GetClusteringProcessorStates(state, nullptr, 1, handle);
}

// La imagen entera. Por defecto (sdk+0x2b6b10 y +0x2b6b30): 300 µs, todos los LED, ganancia 8, imagen normal y
// 320x240; la variante con recorte, sin recorte desde (0, 0) y, como el sdk, sin tocar su filtro de luz exterior.
void GetImageTransferProcessorDefaultConfig(ImageTransferProcessorConfig* config) {
    config->camera.exposure_time_ns = 300'000;
    config->camera.light_target = 0;
    config->camera.gain = 8;
    config->camera.is_negative_image_used = false;
    config->format = 0;
}
void GetImageTransferProcessorDefaultConfig(ImageTransferProcessorExConfig* config) {
    config->camera.exposure_time_ns = 300'000;
    config->camera.light_target = 0;
    config->camera.gain = 8;
    config->camera.is_negative_image_used = false;
    config->orig_format = 0;
    config->trimming_format = 0;
    config->trimming_start_x = 0;
    config->trimming_start_y = 0;
}

// El búfer de trabajo es memoria del juego que la consola presta al servicio para ir dejando la imagen: sin cámara
// nadie escribe en él.
void RunImageTransferProcessor(const IrCameraHandle& handle, const ImageTransferProcessorConfig& config, void*, size_t) {
    run(handle, kImageTransfer, &config.camera, nullptr, "RunImageTransferProcessor");
}
void RunImageTransferProcessor(const IrCameraHandle& handle, const ImageTransferProcessorExConfig& config, void*, size_t) {
    run(handle, kImageTransfer, &config.camera, nullptr, "RunImageTransferProcessor");
}

// Aquí el sdk cambia cualquier fallo, también el de no tenerlo lanzado, por el del estado de la cámara (sdk+0x2b1d58),
// y no escribe ni el estado ni la imagen.
Result GetImageTransferProcessorState(ImageTransferProcessorState*, void*, size_t, const IrCameraHandle& handle) {
    return no_data(handle);
}

// Análisis de manos (el "TeraPlugin" del sdk, sdk+0x2b6c10). Si se puede lanzar lo dice el servicio; como los demás,
// se apunta. Con un modo que no sea 1..4 el sdk aborta: aquí, el error de modo incorrecto.
Result RunHandAnalysis(const IrCameraHandle& handle, const HandAnalysisConfig& config) {
    if (config.mode < 1 || config.mode > 4) return kResultHandAnalysisModeIncorrect;
    run(handle, kTeraPlugin, nullptr, nullptr, "RunHandAnalysis");
    return ResultSuccess;
}

// El sdk lee estados crudos del plugin y los descodifica sobre el propio búfer de salida (sdk+0x2b7230 y +0x2b7340):
// sin muestras no hay nada que descodificar y el búfer se queda como estaba.
Result GetHandAnalysisSilhouetteState(HandAnalysisSilhouetteState*, int* count, int, int64_t, const IrCameraHandle& handle) {
    if (count) *count = 0;
    return read_result(handle, kTeraPlugin);
}
Result GetHandAnalysisImageState(HandAnalysisImageState*, int* count, int, int64_t, const IrCameraHandle& handle) {
    if (count) *count = 0;
    return read_result(handle, kTeraPlugin);
}

}  // namespace nn::irsensor
