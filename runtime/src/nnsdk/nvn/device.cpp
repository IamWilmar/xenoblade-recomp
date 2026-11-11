// nvnDevice*: el NVNdevice del juego es la puerta al contexto Vulkan (vk.cpp).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/vk.h"

namespace {

// NVNdeviceBuilder: opaco y reservado por el juego; solo usamos sus primeros bytes.
struct DeviceBuilder {
    int flags;
};

struct Device {
    int flags = 0;
    int window_origin_mode = 0;
    int depth_mode = 0;
};

nvn::Objects<Device> g_devices;

// nvnDeviceGetInteger: los valores de NVNdeviceInfo de la Switch. La tabla de NieR se dedujo del uso de cada uno; Xenoblade
// pide además la versión, la memoria mínima de comandos de la cola y las características (nn::gfx GetDeviceFeature,
// main+0xb2f120), y usa COUNTER_ALIGNMENT (10) como tamaño de medio registro de su tabla de timestamps de GPU
// (grlib::CGLibBuffOcclusion, 0x2000 × 2 × esto): con 256 en vez de 16 la reserva de 4 MB fallaba, la tabla quedaba en 0
// y CGLibGpuTimeStamp::end leía el contador en 0x808. Las características que el port no tiene dicen 0.
int device_info(int id) {
    switch (id) {
        case 0: return 53;          // API_MAJOR_VERSION
        case 1: return 311;         // API_MINOR_VERSION
        case 4: return 256;         // UNIFORM_BUFFER_ALIGNMENT (NieR: alineación del código de shaders, main+0x9c8834)
        case 10: return 16;         // COUNTER_ALIGNMENT (= tamaño de NVNcounterData)
        case 13: return 4;          // INDIRECT_DRAW_ALIGNMENT
        case 15: return 32;         // TEXTURE_DESCRIPTOR_SIZE: × (reservados + 10240) en NieR (main+0x9ca0d4)
        case 16: return 32;         // SAMPLER_DESCRIPTOR_SIZE
        case 17: return 256;        // RESERVED_TEXTURE_DESCRIPTORS al principio del pool
        case 18: return 256;        // RESERVED_SAMPLER_DESCRIPTORS
        case 19: return 4096;       // COMMAND_BUFFER_COMMAND_ALIGNMENT
        case 20: return 4096;       // COMMAND_BUFFER_CONTROL_ALIGNMENT (NieR redondea con esto un bloque de 512 KB)
        case 40: return 1;          // SUPPORTS_MIN_MAX_FILTERING (VK_EXT_sampler_filter_minmax)
        case 41: return 0;          // SUPPORTS_STENCIL8_FORMAT
        case 42: return 1;          // SUPPORTS_ASTC_FORMATS
        case 50: return 0;          // SUPPORTS_CONSERVATIVE_RASTER: el Adreno no la tiene
        case 56: return 32;         // LINEAR_TEXTURE_STRIDE_ALIGNMENT: el juego usa max(ancho·bpp, esto) (NieR main+0x9c6014)
        case 59: return 1;          // SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES (memory.cpp: lo no mapeado lee ceros)
        case 68: return 1;          // SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES (texturas por teselas de 64 KB, como MH Rise)
        case 78: return 1024;       // SHADER_CODE_MEMORY_POOL_PADDING_SIZE: relleno detrás del código de shaders
        case 81: return 0x10000;    // QUEUE_COMMAND_MEMORY_MIN_SIZE: Xenoblade pide el doble para su cola
        default: return 0;          // 66 67 69 71-76 87 90 (passthrough GS, swizzle de viewport, blend avanzado, DrawTexture…)
    }
}

}  // namespace

namespace nvn {
int g_window_origin_mode = 0;
int g_depth_mode = 0;
}  // namespace nvn
using nvn::g_depth_mode;
using nvn::g_window_origin_mode;

extern "C" {

void nvnDeviceBuilderSetDefaults(DeviceBuilder* b) { b->flags = 0; }
void nvnDeviceBuilderSetFlags(DeviceBuilder* b, int flags) { b->flags = flags; }

nvn::Boolean nvnDeviceInitialize(void* device, const DeviceBuilder* b) {
    if (!nvn::vk()) return 0;
    g_devices.create(device).flags = b->flags;
    NVN_LOG("dispositivo %p listo (flags 0x%x)\n", device, b->flags);
    return 1;
}

void nvnDeviceFinalize(void* device) { g_devices.destroy(device); }

void nvnDeviceGetInteger(const void*, int id, int* value) {
    *value = device_info(id);
    static uint64_t logged[2] = {};  // una línea por id (los de NVN caben en 128)
    if (id >= 0 && id < 128 && !(logged[id / 64] & 1ull << (id % 64))) {
        logged[id / 64] |= 1ull << (id % 64);
        NVN_LOG("GetInteger(%d) = %d\n", id, *value);
    }
}

void nvnDeviceSetWindowOriginMode(void* device, int mode) {
    NVN_LOG("WindowOriginMode %d\n", mode);
    g_window_origin_mode = mode;
    if (Device* d = g_devices.get(device)) d->window_origin_mode = mode;
}

void nvnDeviceSetDepthMode(void* device, int mode) {
    NVN_LOG("DepthMode %d\n", mode);
    g_depth_mode = mode;
    if (Device* d = g_devices.get(device)) d->depth_mode = mode;
}

int nvnDeviceGetWindowOriginMode(const void*) { return g_window_origin_mode; }
int nvnDeviceGetDepthMode(const void*) { return g_depth_mode; }
int nvnDeviceBuilderGetFlags(const DeviceBuilder* b) { return b->flags; }

// Los colores de borrado rápido son una optimización del hardware de la Switch (una tabla de colores que el ZBC borra
// sin escribir la memoria): en Vulkan el driver decide solo cómo borra. Se contesta que quedó registrado (NVN_TRUE), que
// es lo que el juego espera; NVN_FALSE solo diría que la tabla está llena, y aquí no hay tabla que llenar.
nvn::Boolean nvnDeviceRegisterFastClearColor(void*, const float*, int) { return 1; }
nvn::Boolean nvnDeviceRegisterFastClearColori(void*, const int*, int) { return 1; }
nvn::Boolean nvnDeviceRegisterFastClearColorui(void*, const uint32_t*, int) { return 1; }
nvn::Boolean nvnDeviceRegisterFastClearDepth(void*, float) { return 1; }

// NVNcounterData = {valor, timestamp}; el timestamp va en ticks de 614,4 MHz (los que escribe ReportCounter en el
// ejecutor: x625/384 a ns, como hace NieR en main+0x9bbb08).
uint64_t nvnDeviceGetTimestampInNanoseconds(const void*, const uint64_t* counter_data) {
    return counter_data ? counter_data[1] / 384 * 625 + counter_data[1] % 384 * 625 / 384 : 0;
}
uint64_t nvnDeviceGetCurrentTimestampInNanoseconds(const void*) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnDeviceSetDebugLabel(void*, const char*) {}

// Finalizaciones diferidas (NVN_DEVICE_FLAG_DEFERRED_FINALIZE): aquí lo que se finaliza ya se retira solo cuando la GPU
// acaba el último envío que lo usa (retire_texture_gpu, retire_sampler), así que no queda nada pendiente de aplicar.
void nvnDeviceApplyDeferredFinalizes(void*, int) {}

// Las listas grabadas (los handles) viven hasta que el juego recicla la memoria de control donde estarían (AddControlMemory,
// cmdbuf.cpp), como su memoria en NVN; finalizar el handle no la libera, y soltar la lista antes dejaría colgando lo que
// CallCommands copió de ella a otras listas aún sin reproducir. No queda nada más que finalizar.
void nvnDeviceFinalizeCommandHandle(void*, uint64_t) {}

// Un id por nombre de dominio (InsertDebugMarkerStatic/PushDebugGroupStatic): el mismo nombre da el mismo id.
int nvnDeviceGenerateDebugDomainId(const void*, const char* name) {
    static std::mutex m;
    static std::unordered_map<std::string, int> ids;
    std::lock_guard lock(m);
    auto [it, inserted] = ids.try_emplace(name ? name : "", static_cast<int>(ids.size()) + 1);
    return it->second;
}

// Callbacks de la capa de depuración de NVN. Aquí no hay capa de depuración (no se valida nada que pudiera avisarlos):
// se guardan (enable FALSE quita el que se puso) y nadie los llama.
void nvnDeviceInstallDebugCallback(void*, void* callback, void* data, nvn::Boolean enable) {
    static std::mutex m;
    static std::vector<std::pair<void*, void*>> callbacks;
    std::lock_guard lock(m);
    auto it = std::find_if(callbacks.begin(), callbacks.end(), [&](const auto& c) { return c.first == callback; });
    if (enable && it == callbacks.end()) callbacks.emplace_back(callback, data);
    else if (enable) it->second = data;
    else if (it != callbacks.end()) callbacks.erase(it);
}

// ¿Hay un depurador enganchado al proceso? En Android, el TracerPid de /proc/self/status (gdb, lldb).
nvn::Boolean nvnDeviceIsExternalDebuggerAttached(const void*) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[128];
    int tracer = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "TracerPid: %d", &tracer) == 1) break;
    fclose(f);
    return tracer != 0;
}

// Caché de shaders intermedios del compilador en línea (GLSLC): aquí los shaders llegan compilados y se traducen a
// SPIR-V con su propia caché (spirv_cache): no hay nada que dimensionar.
void nvnDeviceSetIntermediateShaderCache(void*, int) {}

// La base de datos de objetos solo la llena la capa de depuración de NVN; sin ella (aquí no la hay) está vacía y el
// recorrido no llama al callback ninguna vez.
void nvnDeviceWalkDebugDatabase(const void*, int, void*, void*) {}

}  // extern "C"

NVN_PROC(nvnDeviceBuilderSetDefaults);
NVN_PROC(nvnDeviceBuilderSetFlags);
NVN_PROC(nvnDeviceInitialize);
NVN_PROC(nvnDeviceFinalize);
NVN_PROC(nvnDeviceGetInteger);
NVN_PROC(nvnDeviceSetWindowOriginMode);
NVN_PROC(nvnDeviceSetDepthMode);
NVN_PROC(nvnDeviceRegisterFastClearColor);
NVN_PROC(nvnDeviceRegisterFastClearColori);
NVN_PROC(nvnDeviceRegisterFastClearColorui);
NVN_PROC(nvnDeviceRegisterFastClearDepth);
NVN_PROC(nvnDeviceGetWindowOriginMode);
NVN_PROC(nvnDeviceGetDepthMode);
NVN_PROC(nvnDeviceBuilderGetFlags);
NVN_PROC(nvnDeviceGetTimestampInNanoseconds);
NVN_PROC(nvnDeviceGetCurrentTimestampInNanoseconds);
NVN_PROC(nvnDeviceSetDebugLabel);
NVN_PROC(nvnDeviceApplyDeferredFinalizes);
NVN_PROC(nvnDeviceFinalizeCommandHandle);
NVN_PROC(nvnDeviceGenerateDebugDomainId);
NVN_PROC(nvnDeviceInstallDebugCallback);
NVN_PROC(nvnDeviceIsExternalDebuggerAttached);
NVN_PROC(nvnDeviceSetIntermediateShaderCache);
NVN_PROC(nvnDeviceWalkDebugDatabase);
