// nvnWindow*: la ventana NVN es un swapchain de Vulkan sobre la ventana de la APK. Las texturas
// de la ventana del juego (1280x720) son nuestras; al presentar se copian a la imagen del swapchain.
// Primer paso: se presenta el último ClearColor que el juego hizo sobre la textura.

#include <android/native_window.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/hud.h"
#include "nnsdk/nvn/lsfg/framegen.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/runtime.h"

namespace {

// NVNwindowBuilder: 64 bytes opacos; guardamos aquí lo que se pone, para los getters.
struct WindowBuilder {
    void* device;
    void* native_window;
    int present_interval;
    int texture_count;
    void* const* textures;
    int num_active;  // SetNumActiveTextures (0: todas)
};
static_assert(sizeof(WindowBuilder) <= 64);

// Varios juegos de command buffer, fence y semáforos: con uno, presentar el frame N esperaba a que la GPU
// acabara el N-1 (CPU y GPU por turnos: la GPU al 66 % y 22 fps en la fábrica).
constexpr int kPresentSlots = 3;
struct PresentSlot {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
};
struct Presenter {
    VkCommandPool pool = VK_NULL_HANDLE;
    PresentSlot slots[kPresentSlots];
    int next = 0;
};

Presenter g_presenter;
// Un present cada vez (g_presenter, su pool y el ritmo). No es el del ejecutor: esperar al fence de hace
// kPresentSlots frames o a una imagen del swapchain no para a los hilos del juego ni a una reproducción.
std::mutex g_present_mutex;

bool create_swapchain(nvn::Window& w) {
    nvn::VkContext* c = nvn::vk();
    if (!c) return false;
    VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    sci.window = static_cast<ANativeWindow*>(w.native_window);
    if (VkResult r = vkCreateAndroidSurfaceKHR(c->instance, &sci, nullptr, &w.surface); r != VK_SUCCESS) {
        NVN_LOG("vkCreateAndroidSurfaceKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->gpu, w.surface, &caps);
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, formats.data());
    VkSurfaceFormatKHR format = formats[0];
    for (const VkSurfaceFormatKHR& f : formats)
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) format = f;

    // La pantalla de la Flip es vertical de fábrica (1080x1920) y Android la usa girada: el surface
    // da el tamaño ya en horizontal (1920x1080) con una rotación de 90°. Pretransformar obligaría a
    // dibujar girado (el blit no puede), así que se pide sin transformar y la gira el compositor.
    w.extent = caps.currentExtent;
    VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
    if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    NVN_LOG("swapchain %ux%u (transformación actual 0x%x, usada 0x%x)\n", w.extent.width, w.extent.height,
            caps.currentTransform, transform);
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = w.surface;
    info.minImageCount = std::max(nvn::lsfg::swapchain_images(caps), caps.minImageCount);  // 4 con LSFG posible
    info.imageFormat = format.format;
    w.format = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = w.extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.preTransform = transform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & info.compositeAlpha)) info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    if (VkResult r = vkCreateSwapchainKHR(c->device, &info, nullptr, &w.swapchain); r != VK_SUCCESS) {
        NVN_LOG("vkCreateSwapchainKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, nullptr);
    w.images.resize(count);
    vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, w.images.data());

    if (!g_presenter.pool) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = c->queue_family;
        vkCreateCommandPool(c->device, &pci, nullptr, &g_presenter.pool);
        for (PresentSlot& slot : g_presenter.slots) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = g_presenter.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(c->device, &ai, &slot.cmd);
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            vkCreateSemaphore(c->device, &si, nullptr, &slot.acquired);
            vkCreateSemaphore(c->device, &si, nullptr, &slot.rendered);
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(c->device, &fi, nullptr, &slot.done);
        }
    }
    NVN_LOG("swapchain %ux%u, %u imágenes, formato %d\n", w.extent.width, w.extent.height, count, format.format);
    return true;
}

// --- Detector de destellos (DEPURACIÓN) ---
// <datos>/captura_destello (en vivo): el fuego del Aknosom salía como un disco amarillo plano que tapaba media pantalla
// durante ~0,5 s (5-11), demasiado poco para pedir la traza a mano. Cada frame se copian 16x9 texels de la textura de la
// ventana (dentro de su recorte) a un buffer de la CPU; al volver a ese juego de present (kPresentSlots frames después, con
// su fence ya esperado) se miran: con un sexto o más de amarillo saturado se pide la traza completa del frame siguiente
// (request_trace). Como mucho una cada 20 s y 4 por partida.
constexpr int kProbeW = 16, kProbeH = 9;
struct FlashProbe {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const uint8_t* mapped = nullptr;
    bool pending = false;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
FlashProbe g_probes[kPresentSlots];

uint32_t texel_bytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        default: return 0;
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * man / 16777216.0f;
    if (exp == 31) return 65504.0f;
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + man / 1024.0f, static_cast<int>(exp) - 15);
}

bool make_probe(FlashProbe& p) {
    nvn::VkContext* c = nvn::vk();
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = kProbeW * kProbeH * 8;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(c->device, &bi, nullptr, &p.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, p.buffer, &req);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t k = 0; k < c->memory.memoryTypeCount && type == UINT32_MAX; k++)
        if ((req.memoryTypeBits >> k & 1) && (c->memory.memoryTypes[k].propertyFlags & want) == want) type = k;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    void* mapped = nullptr;
    if (type == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &p.memory) != VK_SUCCESS ||
        vkBindBufferMemory(c->device, p.buffer, p.memory, 0) != VK_SUCCESS ||
        vkMapMemory(c->device, p.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        if (p.memory) vkFreeMemory(c->device, p.memory, nullptr);
        vkDestroyBuffer(c->device, p.buffer, nullptr);
        p = {};
        return false;
    }
    p.mapped = static_cast<const uint8_t*>(mapped);
    return true;
}

// Tras esperar el fence del juego de present: lo que copió hace kPresentSlots frames.
void check_probe(FlashProbe& p) {
    if (!p.pending) return;
    p.pending = false;
    const uint32_t bytes = texel_bytes(p.format);
    // Umbrales (en vivo, <datos>/destello_umbral: "R G Bmáx R-B sextos"): un destello cálido y muy claro. Los primeros
    // (230 190 200 40 y un sexto) no saltaron con uno casi blanco del 5-11; por defecto ahora 220 180 235 20.
    struct Thresholds {
        float r = 220, g = 180, b = 235, rb = 20;
        int sixths = 1;
    };
    static Thresholds t;
    static auto read_at = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    if (std::chrono::steady_clock::now() - read_at > std::chrono::seconds(5)) {
        read_at = std::chrono::steady_clock::now();
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/destello_umbral").c_str(), "r")) {
            Thresholds n;
            if (fscanf(f, "%f %f %f %f %d", &n.r, &n.g, &n.b, &n.rb, &n.sixths) == 5) t = n;
            fclose(f);
        }
    }
    int yellow = 0;
    for (int k = 0; k < kProbeW * kProbeH; k++) {
        const uint8_t* q = p.mapped + k * bytes;
        float r = 0, g = 0, b = 0;
        switch (p.format) {
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB: r = q[2], g = q[1], b = q[0]; break;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: {
                uint32_t v;
                memcpy(&v, q, 4);
                r = (v & 0x3ff) / 4.0f, g = (v >> 10 & 0x3ff) / 4.0f, b = (v >> 20 & 0x3ff) / 4.0f;
                break;
            }
            case VK_FORMAT_R16G16B16A16_SFLOAT: {
                uint16_t h[3];
                memcpy(h, q, 6);
                r = std::min(half_to_float(h[0]), 1.0f) * 255, g = std::min(half_to_float(h[1]), 1.0f) * 255,
                b = std::min(half_to_float(h[2]), 1.0f) * 255;
                break;
            }
            default: r = q[0], g = q[1], b = q[2]; break;
        }
        yellow += r >= t.r && g >= t.g && b <= t.b && r - b >= t.rb;
    }
    static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    static int taken = 0;
    const auto now = std::chrono::steady_clock::now();
    if (yellow * 6 >= kProbeW * kProbeH * t.sixths && taken < 4 && now - last >= std::chrono::seconds(20)) {
        last = now;
        taken++;
        NVN_LOG("detector de destellos: %d de %d muestras en amarillo saturado: traza completa del frame siguiente\n", yellow,
                kProbeW * kProbeH);
        nvn::request_trace(true);
    }
}

// Graba la copia de las muestras de la textura de la ventana (en GENERAL, como la lee el blit).
void record_probe(FlashProbe& p, VkCommandBuffer cmd, const nvn::Texture& t, const VkRect2D& crop) {
    const uint32_t bytes = texel_bytes(t.vk_format);
    if (!bytes || !t.image || (!p.buffer && !make_probe(p))) return;
    const uint32_t w = crop.extent.width ? crop.extent.width : t.desc.width;
    const uint32_t h = crop.extent.height ? crop.extent.height : t.desc.height;
    VkBufferImageCopy regions[kProbeW * kProbeH] = {};
    for (int y = 0; y < kProbeH; y++)
        for (int x = 0; x < kProbeW; x++) {
            VkBufferImageCopy& r = regions[y * kProbeW + x];
            r.bufferOffset = static_cast<VkDeviceSize>(y * kProbeW + x) * bytes;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageOffset = {crop.offset.x + static_cast<int32_t>((2 * x + 1) * w / (2 * kProbeW)),
                             crop.offset.y + static_cast<int32_t>((2 * y + 1) * h / (2 * kProbeH)), 0};
            r.imageExtent = {1, 1, 1};
        }
    vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, p.buffer, kProbeW * kProbeH, regions);
    p.pending = true;
    p.format = t.vk_format;
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
}

}  // namespace

nvn::Objects<nvn::Window> nvn::g_windows;

namespace {
// DEPURACIÓN: <datos>/lsfg_margen con unos milisegundos (p. ej. "4"): margen fijo entre el generado y el real, en vez
// del que se ajusta solo. Se lee como mucho una vez por segundo; -1 si no hay fichero.
int64_t live_margin_ns() {
    static int64_t value = -1;
    static std::chrono::steady_clock::time_point next{};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next) {
        next = now + std::chrono::seconds(1);
        const char* data = getenv("NIER_DATA");
        double ms = -1;
        if (FILE* f = fopen((std::string(data ? data : ".") + "/lsfg_margen").c_str(), "r")) {
            if (fscanf(f, "%lf", &ms) != 1) ms = -1;
            fclose(f);
        }
        const int64_t v = ms < 0 ? -1 : static_cast<int64_t>(ms * 1e6);
        if (v != value) {
            if (v < 0) NVN_LOG("lsfg: margen del real tras el generado: automático\n");
            else NVN_LOG("lsfg: margen del real tras el generado fijo en %.1f ms\n", ms);
        }
        value = v;
    }
    return value;
}
// El real tiene que salir un refresco después de que salga el generado, y el generado sale cuando la GPU lo acaba
// (detrás del frame del juego): desde que se manda hasta que se ve pasan L ms, que dependen de la carga. Se mide L con
// los tiempos de presentación pasados (VK_GOOGLE_display_timing) y el real se pide para "mandado + 16,7 + L - 1"; si
// un generado no llega a verse (el compositor lo tiró), se sube L.
class LsfgPacer {
public:
    int64_t margin_ns(VkDevice device, VkSwapchainKHR swapchain) {
        if (const int64_t fixed = live_margin_ns(); fixed >= 0) return fixed;
        if (!get_past_) get_past_ = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                            vkGetDeviceProcAddr(device, "vkGetPastPresentationTimingGOOGLE"));
        if (get_past_) {
            uint32_t n = 0;
            get_past_(device, swapchain, &n, nullptr);
            n = std::min<uint32_t>(n, 16);
            VkPastPresentationTimingGOOGLE past[16];
            if (n && get_past_(device, swapchain, &n, past) >= 0)
                for (uint32_t k = 0; k < n; k++) note(past[k]);
        }
        return std::max<int64_t>(latency_ns_ - 1000000, 2000000);
    }
    void queued_generated(uint32_t id, int64_t ns) { sent_[id % kRing] = {id, ns}; }

private:
    static constexpr uint32_t kRing = 64;
    struct Sent {
        uint32_t id = 0;
        int64_t ns = 0;
    };
    void note(const VkPastPresentationTimingGOOGLE& t) {
        if (!(t.presentID & 1)) {  // un real: ¿se vio el generado de delante?
            const uint32_t g = t.presentID - 1;
            if (g > last_seen_g_ && g > 1) {
                latency_ns_ = std::min<int64_t>(latency_ns_ + 4000000, 45000000);  // no salió: más margen
                if (dropped_++ < 5) NVN_LOG("lsfg: el generado %u no se vio; margen %.1f ms\n", g, latency_ns_ / 1e6);
            }
            return;
        }
        const Sent& s = sent_[t.presentID % kRing];
        if (s.id != t.presentID || !t.actualPresentTime) return;
        last_seen_g_ = t.presentID;
        const int64_t l = static_cast<int64_t>(t.actualPresentTime) - s.ns;
        if (l > 0 && l < 100000000) latency_ns_ += (l - latency_ns_) / 8;  // media móvil
    }
    PFN_vkGetPastPresentationTimingGOOGLE get_past_ = nullptr;
    Sent sent_[kRing];
    uint32_t last_seen_g_ = 0, dropped_ = 0;
    int64_t latency_ns_ = 12000000;
};
LsfgPacer g_lsfg_pacer;

// El reloj de MH Rise (16.0.1), en el objeto de via.Application (App = *(main+0x12950348)). Su bucle principal
// (main+0x6b047c0) mide cada vuelta con GetSystemTick, ejecuta el frame (main+0x6b03fa4) y luego ESPERA (sleep de 1 ms,
// y girando los últimos 2 ms) hasta que hayan pasado App+0x430 / App+0x434 frames de 60 Hz: base 60 y MaxFps 30 en
// juego (main+0x3600f04 pone MaxFps 60 u 30 y el intervalo 1 o 2 de la ventana a la vez; las cargas van a 60). El paso es
// variable (App+0x441, isFixedFrameRate, a 0): dt = lo medido en frames de 60 Hz (como mucho 8, App+0x438; más de 50,
// App+0x43c, vuelve a 1) en App+0x41c, y en segundos en App+0x424; App+0x420 = ms de la vuelta anterior. O sea: entre
// 7,5 y 60 fps el juego va a su velocidad (no a cámara lenta como NieR), pero forzar el intervalo 1 al presentar no
// basta para 60: el limitador del juego seguía en 30. Con <datos>/60fps se le pone MaxFps 60 en cada present (el juego lo
// vuelve a poner en 30 al cambiar de escena) y al quitarlo se le devuelve lo suyo.
constexpr size_t kAppPointer = 0x12950348;      // via.Application* (dato de main)
constexpr size_t kLimiterLoad = 0x6b04828;      // ldr s1, [x19, #0x434]: el MaxFps que lee el limitador
constexpr uint32_t kLimiterLoadCode = 0xBD443661;
struct GameClock {  // App+0x41c..0x434
    float dt60, frame_ms, dt, unused[2], base, max_fps;
};
static_assert(sizeof(GameClock) == 0x434 + 4 - 0x41c);

GameClock* game_clock() {
    static const uint8_t* base = nnsdk::host().main_base;
    static const bool known = [] {
        const bool ok = base && nnsdk::host().main_size > kAppPointer + 8 &&
                        *reinterpret_cast<const uint32_t*>(base + kLimiterLoad) == kLimiterLoadCode;
        NVN_LOG("reloj del juego: %s\n", ok ? "MH Rise 16.0.1 (App en main+0x12950348)" : "no es el main conocido: sin tocar MaxFps");
        return ok;
    }();
    if (!known) return nullptr;
    const uint64_t app = *reinterpret_cast<const volatile uint64_t*>(base + kAppPointer);
    if (!app) return nullptr;  // aún sin via.Application
    auto* c = reinterpret_cast<GameClock*>(app + 0x41c);
    return c->base == 60.0f ? c : nullptr;
}

// --- Resolución dinámica (MH Rise 16.0.1) ---
// via.render.Renderer (*(main+0x12950538)) guarda en +0x1a78 su ImageQualityRate. La actualización de RenderOutput
// (main+0x7936a24) lo multiplica cada frame por su ImageQuality (en portátil la escena sale a 960x540) y layer.Scene recorta
// a eso su región (ImageQualityRegion) y la escala él a la salida: el resto del port no cambia. El juego solo lo toca al
// cargar (main+0x5389b50: 0,1 mientras carga, 1,0 al acabar). Su propia resolución dinámica (snow.DynamicResolutionManager)
// escribe en RenderOutput y aquí no baja de 960x540; esto multiplica encima. Cambiarlo cuesta un frame sin la historia del
// TAA: por pasos (anchos múltiplos de 64: 960, 896, 832, 768, 704) y con esperas. En combate la GPU del frame del juego
// pasaba de 33 ms (al 90-99 % y a su reloj máximo): 25 y 21 fps.
// Baja un paso cuando la media de los últimos 8 frames pasa del 91 % del presupuesto (33,3 ms a 30 fps) o dos de ellos
// pasan del 100 %, como pronto 0,5 s después del cambio anterior y solo si el ejecutor cabe (si el frame lo limita la CPU,
// bajar la resolución no lo arregla). Sube tras 3 s sin cambios si el peor de esos 90 frames, escalado por los píxeles
// del paso de arriba, cabe en el 78 % (con 2 s y el 80 % subía y bajaba 7 veces en 10 s al entrar en la misión).
// <datos>/sin_resolucion_dinamica (en vivo): de vuelta a 1,0 (lo del juego), y quieta.
constexpr size_t kRendererPointer = 0x12950538;  // via.render.Renderer* (dato de main)
constexpr size_t kRateField = 0x1a78;
constexpr size_t kRateSetter = 0x78a0ea0;  // set_ImageQualityRate: adrp/ldr del Renderer y str s0, [x8, #0x1a78]
constexpr uint32_t kRateSetterCode[3] = {0x90058588, 0xF9429D08, 0xBD1A7900};
constexpr size_t kRateLoad = 0x7936e20;  // ldr s1, [x8, #0x1a78] en la actualización de RenderOutput
constexpr uint32_t kRateLoadCode = 0xBD5A7901;
constexpr float kRates[] = {1.0f, 0.93334f, 0.86667f, 0.80001f, 0.73334f};  // un poco por encima: floor no pierde un píxel
constexpr int kLevels = static_cast<int>(std::size(kRates));

struct DynamicResolution {
    int level = 0;
    int since_change = 1000;  // frames de GPU desde el último cambio
    int64_t samples[90] = {};
    int count = 0, next = 0;
    uint64_t serial = 0;
    uint64_t changes = 0;   // para el log de ritmo
    bool written = false;   // se escribió algo distinto de 1,0
    int logged = 0;
};
DynamicResolution g_dynres;

volatile float* game_quality_rate() {
    static const uint8_t* base = nnsdk::host().main_base;
    static const bool known = [] {
        const auto* code = reinterpret_cast<const uint32_t*>(base + kRateSetter);
        const bool ok = base && nnsdk::host().main_size > kRendererPointer + 8 && code[0] == kRateSetterCode[0] &&
                        code[1] == kRateSetterCode[1] && code[2] == kRateSetterCode[2] &&
                        *reinterpret_cast<const uint32_t*>(base + kRateLoad) == kRateLoadCode;
        NVN_LOG("resolución dinámica: %s\n", ok ? "MH Rise 16.0.1 (Renderer en main+0x12950538, ImageQualityRate en +0x1a78)"
                                                : "no es el main conocido: sin tocar la resolución");
        return ok;
    }();
    if (!known) return nullptr;
    const uint64_t renderer = *reinterpret_cast<const volatile uint64_t*>(base + kRendererPointer);
    return renderer ? reinterpret_cast<volatile float*>(renderer + kRateField) : nullptr;
}

// Desde el hilo que presenta, una vez por frame. interval: el de presentación (presupuesto = interval x 16,7 ms).
void dynamic_resolution(int interval) {
    DynamicResolution& d = g_dynres;
    volatile float* rate = game_quality_rate();
    if (!rate) return;
    const float now = *rate;
    if (now < 0.2f) {  // cargando (0,1): es del juego; al acabar pone 1,0 y se vuelve a empezar desde arriba
        d.level = 0;
        d.count = d.next = 0;
        d.since_change = 1000;
        return;
    }
    if (NVN_FLAG("sin_resolucion_dinamica")) {
        if (d.written) *rate = 1.0f;
        d.written = false;
        d.level = 0;
        return;
    }
    int64_t gpu = 0;
    const bool fresh = nvn::game_gpu_frame(d.serial, gpu);
    if (fresh && gpu > 0) {
        d.samples[d.next] = gpu;
        d.next = (d.next + 1) % 90;
        d.count = std::min(d.count + 1, 90);
        d.since_change++;
        const double budget = interval * 16.6667e6;
        auto last = [&](int k) { return d.samples[(d.next - 1 - k + 180) % 90]; };
        int level = d.level;
        if (d.count >= 8 && d.since_change >= 15 && level + 1 < kLevels) {
            double sum = 0;
            int over = 0;
            for (int k = 0; k < 8; k++) sum += last(k), over += last(k) > budget;
            const bool cpu_fits = nvn::last_frame_run_ns() < 0.85 * budget;
            if ((sum / 8 > 0.91 * budget || over >= 2) && cpu_fits) level++;
        }
        if (level == d.level && d.count >= 90 && d.since_change >= 90 && level > 0) {
            int64_t worst = 0;
            for (int k = 0; k < 90; k++) worst = std::max(worst, last(k));
            const double up = static_cast<double>(kRates[level - 1]) / kRates[level];
            if (worst * up * up < 0.78 * budget) level--;
        }
        if (level != d.level) {
            if (d.logged++ < 40)
                NVN_LOG("resolución dinámica: x%.3f -> x%.3f (escena ~%dx%d), GPU del juego %.1f ms\n", kRates[d.level],
                        kRates[level], static_cast<int>(960 * kRates[level]), static_cast<int>(540 * kRates[level]), gpu / 1e6);
            d.level = level;
            d.since_change = 0;
            d.changes++;
        }
    }
    const float want = kRates[d.level];
    if (now != want) {
        *rate = want;
        d.written = d.written || want != 1.0f;
    }
}

// Registro de ritmo (siempre, sin <datos>/registro): cada 10 s, si hubo algún frame de más de 40 ms (por debajo de 30 fps,
// con el intervalo 2 del juego), una línea con por qué pudo ser: la GPU del frame del juego (media y peor), el ejecutor
// por frame (y lo que esperó a la GPU), las pipelines que se crearon en NvnSubmit con el frame parado, las que fueron al
// hilo de segundo plano (y los draws que no se dibujaron esperándolas) y las traducciones en NvnSubmit y en la precarga.
// Sin frames lentos, una cada 60 s. Con fflush: si el juego se cierra, no se pierde lo último (va por la tubería, no a la
// SD desde aquí).
void log_pace(const nvn::Window& w) {
    using clock = std::chrono::steady_clock;
    static clock::time_point window_start = clock::now(), last_present = window_start, last_line = window_start;
    static uint64_t frames = 0, slow = 0, worst_ns = 0;
    const clock::time_point now = clock::now();
    const uint64_t gap = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_present).count());
    last_present = now;
    frames++;
    if (gap > 40'000'000) slow++;
    worst_ns = std::max(worst_ns, gap);
    const double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs < 10.0) return;
    nvn::DropStats& d = nvn::g_drop_stats;
    auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
    const uint64_t gpu_sum = take(d.gpu_sum_ns), gpu_max = take(d.gpu_max_ns), gpu_frames = take(d.gpu_frames);
    const uint64_t gap_sum = take(d.gap_sum_ns), gap_frames = take(d.gap_frames);
    const uint64_t run = take(d.run_ns), run_max = take(d.run_max_ns), wait = take(d.wait_ns);
    const uint64_t live = take(d.pipelines_live), live_ns = take(d.pipelines_live_ns);
    const uint64_t async = take(d.pipelines_async), skipped = take(d.draws_skipped);
    const uint64_t translated = take(d.translated_live), translated_ns = take(d.translated_live_ns);
    const uint64_t prefetched = take(d.prefetched), prefetch_ns = take(d.prefetch_ns);
    std::string slowest;
    uint64_t slowest_ns = 0;
    {
        std::lock_guard lock(d.slow_mutex);
        slowest_ns = std::exchange(d.slow_ns, 0);
        slowest = std::move(d.slow_summary);
        d.slow_summary.clear();
    }
    if (slow || now - last_line >= std::chrono::seconds(60)) {
        last_line = now;
        NVN_LOG("ritmo: %.1f fps, %llu frames de más de 40 ms (el peor %.0f ms), intervalo %d, resolución x%.3f (%llu cambios); "
                "GPU del juego %.1f ms (peor "
                "%.1f, hueco hasta el siguiente %.1f); ejecutor %.1f ms/frame (peor %.1f, %.1f esperando a la GPU); "
                "pipelines en NvnSubmit %llu (%.0f ms), "
                "en segundo plano %llu (%llu draws esperándolas); traducciones en NvnSubmit %llu (%.0f ms), precargadas %llu "
                "(%.0f ms)\n",
                frames / secs, static_cast<unsigned long long>(slow), worst_ns / 1e6, w.present_interval, kRates[g_dynres.level],
                static_cast<unsigned long long>(std::exchange(g_dynres.changes, 0)),
                gpu_frames ? gpu_sum / 1e6 / gpu_frames : 0.0, gpu_max / 1e6, gap_frames ? gap_sum / 1e6 / gap_frames : 0.0,
                run / 1e6 / std::max<uint64_t>(frames, 1),
                run_max / 1e6, wait / 1e6 / std::max<uint64_t>(frames, 1), static_cast<unsigned long long>(live), live_ns / 1e6,
                static_cast<unsigned long long>(async), static_cast<unsigned long long>(skipped),
                static_cast<unsigned long long>(translated), translated_ns / 1e6, static_cast<unsigned long long>(prefetched),
                prefetch_ns / 1e6);
        if (slowest_ns) NVN_LOG("ritmo: el envío más lento, %.0f ms: %s\n", slowest_ns / 1e6, slowest.c_str());
        fflush(stdout);
    }
    window_start = now;
    frames = slow = worst_ns = 0;
}

// <datos>/60fps: MaxFps 60 mientras esté (lo que el juego tenía, de vuelta al quitarlo). Desde el hilo que presenta.
void game_max_fps(bool fps60) {
    static float wanted = 30.0f;  // lo último que puso el juego
    static bool forced = false;
    GameClock* c = game_clock();
    if (!c) return;
    volatile float& max_fps = c->max_fps;
    const float now = max_fps;
    if (fps60) {
        if (now != 60.0f) {
            wanted = now;
            max_fps = 60.0f;
        }
        if (!forced) NVN_LOG("reloj del juego: MaxFps %.0f -> 60 (<datos>/60fps)\n", now);
        forced = true;
    } else if (forced) {
        if (now == 60.0f && wanted != 60.0f) max_fps = wanted;
        NVN_LOG("reloj del juego: MaxFps de vuelta a %.0f\n", wanted);
        forced = false;
    }
}
}  // namespace

// La ventana de Android cambió (la app se fue a segundo plano o volvió; ver NnsdkWindow): se suelta el swapchain y la
// superficie de la vieja y, si hay una nueva, se crean en ella. Antes, al volver del escritorio la pantalla se quedaba
// en negro (se seguía presentando en la ventana destruida).
void follow_native_window(nvn::Window& w) {
    NnsdkWindow* state = nnsdk::host().window_state;
    if (!state) return;
    const uint32_t serial = state->serial.load(std::memory_order_acquire);
    if (serial == state->done.load(std::memory_order_relaxed)) return;
    void* now = state->window.load(std::memory_order_acquire);
    nvn::VkContext* c = nvn::vk();
    if (now != w.native_window || !w.swapchain) {
        {
            std::lock_guard queue_lock(nvn::vk_queue_mutex());
            vkQueueWaitIdle(c->queue);  // lo presentado en la vieja, terminado antes de soltarla
        }
        if (w.swapchain) vkDestroySwapchainKHR(c->device, w.swapchain, nullptr);
        if (w.surface) vkDestroySurfaceKHR(c->instance, w.surface, nullptr);
        w.swapchain = VK_NULL_HANDLE;
        w.surface = VK_NULL_HANDLE;
        w.images.clear();
        w.native_window = now;
        if (now) {
            const bool ok = create_swapchain(w);
            NVN_LOG("ventana nueva: swapchain %s\n", ok ? "creado" : "sin crear");
        } else {
            NVN_LOG("sin ventana (segundo plano): no se presenta\n");
        }
    }
    state->done.store(serial, std::memory_order_release);
}

void nvn::present(Window& w, int index) {
    VkContext* c = vk();
    if (!c || index < 0 || index >= static_cast<int>(w.textures.size())) return;
    std::lock_guard present_lock(g_present_mutex);
    follow_native_window(w);
    Texture* t = g_textures.get(w.textures[index]);
    // Intervalo de presentación de NVN (NieR pide 2: 30 fps, como en la Switch). FIFO solo garantiza
    // uno por vsync, así que el resto lo pone este ritmo. El reloj del juego mide tiempo real (pasos de 1/60 s,
    // entre 1 y 2 por frame): a 60 va a su velocidad, y por debajo de 30 a cámara lenta (eso es NieR; MH Rise mide su
    // paso, ver GameClock). Con <datos>/60fps se fuerza el intervalo 1 (y, en MH Rise, MaxFps 60). Sin el lock del
    // ejecutor: dormido con él, los hilos del juego que esperaban un sync se quedaban parados hasta 33 ms.
    {
        using clock = std::chrono::steady_clock;
        static clock::time_point next = clock::now();
        static int logged_interval = 0;
        const bool fps60 = NVN_FLAG("60fps");
        game_max_fps(fps60);
        const int interval = fps60 ? 1 : std::max(w.present_interval, 1);
        dynamic_resolution(interval);
        if (interval != logged_interval) {
            NVN_LOG("ritmo de presentación: intervalo %d%s\n", interval, fps60 ? " (forzado por <datos>/60fps)" : "");
            logged_interval = interval;
        }
        const auto period = std::chrono::microseconds(16667) * interval;
        // A intervalo 2 se despierta 2 ms antes (FIFO pone el resto). A 1, en `next`: un frame de menos de 16,7 ms
        // le cuenta al juego como uno entero y lo aceleraría un poco.
        const auto early = std::chrono::milliseconds(interval == 1 ? 0 : 2);
        auto now = clock::now();
        if (now < next - early) std::this_thread::sleep_until(next - early);
        next = std::max(next + period, clock::now() + period / 2);
    }
    if (!w.swapchain) {  // sin ventana (en segundo plano): el juego sigue a su ritmo, sin presentar
        std::lock_guard executor_lock(executor_mutex());
        next_frame();
        return;
    }
    hud::real_frame(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    // LSFG 2x encendido: antes del real, el frame intermedio (lsfg/framegen.cpp).
    uint32_t generated_id = 0;
    const int64_t generated_ns = lsfg::present_generated(w, t, &generated_id);
    if (generated_ns) g_lsfg_pacer.queued_generated(generated_id, generated_ns);
    PresentSlot& slot = g_presenter.slots[g_presenter.next];
    g_presenter.next = (g_presenter.next + 1) % kPresentSlots;

    vkWaitForFences(c->device, 1, &slot.done, VK_TRUE, UINT64_MAX);  // el de hace kPresentSlots frames
    FlashProbe& probe = g_probes[&slot - g_presenter.slots];
    // Las muestras de cualquier juego de present ya terminado, no solo las de este (el de hace kPresentSlots frames): el
    // del frame anterior suele estarlo ya, y así la traza cae 2 frames después del destello y no 4 (el 5-11 llegó tarde:
    // en el frame trazado el cazador ya estaba en el suelo).
    for (int k = 0; k < kPresentSlots; k++)
        if (&g_probes[k] == &probe || vkGetFenceStatus(c->device, g_presenter.slots[k].done) == VK_SUCCESS)
            check_probe(g_probes[k]);
    uint32_t image = 0;
    VkResult r = vkAcquireNextImageKHR(c->device, w.swapchain, UINT64_MAX, slot.acquired, VK_NULL_HANDLE, &image);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        NVN_LOG("vkAcquireNextImageKHR: %s\n", vk_result_name(r));
        return;  // el fence sigue señalado (se reinicia después): este contexto no se queda esperando para siempre
    }
    vkResetFences(c->device, 1, &slot.done);

    // El blit lee la imagen de la textura: desde grabarlo hasta enviarlo, con el lock del ejecutor. Así, si el
    // juego la retira, se libera tras un envío del ejecutor posterior a este (va detrás en la cola).
    std::unique_lock executor_lock(executor_mutex());
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(slot.cmd, 0);
    vkBeginCommandBuffer(slot.cmd, &bi);
    {  // lo que pintaron los envíos anteriores (en vuelo, misma cola), visible para el blit
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                             &mb, 0, nullptr, 0, nullptr);
    }
    barrier(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue color{};
    if (t && t->cleared) memcpy(color.float32, t->clear_color, sizeof color.float32);
    // Lo que el juego dibujó en la textura de la ventana, escalado a la pantalla; si aún no hay
    // imagen (no se ha dibujado nada), su último color de borrado.
    if (!t || !blit_to_swapchain(*t, w.crop, w.images[image], w.format, w.extent, slot.cmd)) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    if (t && NVN_FLAG("captura_destello")) record_probe(probe, slot.cmd, *t, w.crop);
    hud::record(slot.cmd, w.images[image], w.extent, w.format);  // FPS y gráfica, encima del frame
    barrier(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(slot.cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &slot.acquired;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &slot.rendered;
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        vkQueueSubmit(c->queue, 1, &si, slot.done);
    }
    executor_lock.unlock();

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &slot.rendered;
    pi.swapchainCount = 1;
    pi.pSwapchains = &w.swapchain;
    pi.pImageIndices = &image;
    // Con un generado delante, el real no antes de un refresco después: con los dos listos en el mismo refresco,
    // el compositor de Android se queda con el más nuevo y tira el otro (en pantalla seguían saliendo 30).
    // <datos>/lsfg_margen (ms) añade margen en vivo por si el generado aún no está en pantalla a esa hora.
    VkPresentTimeGOOGLE when{};
    VkPresentTimesInfoGOOGLE times{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
    if (generated_ns && c->display_timing) {
        when.presentID = generated_id + 1;  // par: el real detrás del generado generated_id
        when.desiredPresentTime =
            static_cast<uint64_t>(generated_ns + 16666667 + g_lsfg_pacer.margin_ns(c->device, w.swapchain));
        times.swapchainCount = 1;
        times.pTimes = &when;
        pi.pNext = &times;
    }
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        vkQueuePresentKHR(c->queue, &pi);
    }
    executor_lock.lock();  // el frame del ejecutor y sus volcados
    debug_dump_frame(t);
    next_frame();
    executor_lock.unlock();

    w.presented++;
    log_pace(w);
    // Rendimiento: frames por segundo cada ~5 s.
    static auto window_start = std::chrono::steady_clock::now();
    static uint64_t window_frames = 0;
    window_frames++;
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs >= 5.0) {
        if (NVN_FLAG("registro")) {
            NVN_LOG("%.1f fps (frame %llu)\n", window_frames / secs, static_cast<unsigned long long>(w.presented));
            // El reloj del juego: su MaxFps, lo que midió de la última vuelta y el paso que dio (a 60 fps, ~16,7 ms y 1).
            if (const GameClock* c = game_clock())
                NVN_LOG("reloj del juego: MaxFps %.0f, vuelta %.1f ms, paso %.2f frames de 60 Hz (%.4f s)\n", c->max_fps,
                        c->frame_ms, c->dt * 60.0f, c->dt);
        }
        window_start = now;
        window_frames = 0;
    }
}

extern "C" {

void nvnWindowBuilderSetDefaults(WindowBuilder* b) {
    memset(b, 0, sizeof *b);
    b->present_interval = 1;
}
void nvnWindowBuilderSetDevice(WindowBuilder* b, void* device) { b->device = device; }
void nvnWindowBuilderSetNativeWindow(WindowBuilder* b, void* native_window) { b->native_window = native_window; }
void nvnWindowBuilderSetPresentInterval(WindowBuilder* b, int interval) { b->present_interval = interval; }
void nvnWindowBuilderSetTextures(WindowBuilder* b, int count, void* const* textures) {
    b->texture_count = count;
    b->textures = textures;
}
void nvnWindowBuilderSetNumActiveTextures(WindowBuilder* b, int count) { b->num_active = count; }
void* nvnWindowBuilderGetDevice(const WindowBuilder* b) { return b->device; }
void* nvnWindowBuilderGetNativeWindow(const WindowBuilder* b) { return b->native_window; }
int nvnWindowBuilderGetPresentInterval(const WindowBuilder* b) { return b->present_interval; }
int nvnWindowBuilderGetNumTextures(const WindowBuilder* b) { return b->texture_count; }
// Sin fijar, las activas son todas.
int nvnWindowBuilderGetNumActiveTextures(const WindowBuilder* b) {
    return b->num_active > 0 ? std::min(b->num_active, b->texture_count) : b->texture_count;
}
const void* nvnWindowBuilderGetTexture(const WindowBuilder* b, int index) {
    return b->textures && index >= 0 && index < b->texture_count ? b->textures[index] : nullptr;
}

nvn::Boolean nvnWindowInitialize(void* window, const WindowBuilder* b) {
    nvn::Window& w = nvn::g_windows.create(window);
    w.native_window = b->native_window;
    w.present_interval = b->present_interval;
    NVN_LOG("ventana con intervalo de presentación %d\n", w.present_interval);
    w.textures.assign(b->textures, b->textures + b->texture_count);
    w.num_active = b->num_active;
    return create_swapchain(w) ? 1 : 0;
}
void nvnWindowFinalize(void* window) { nvn::g_windows.destroy(window); }
int nvnWindowGetPresentInterval(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? w->present_interval : 1;
}
void nvnWindowSetPresentInterval(void* window, int interval) {
    static int last = -1;
    if (interval != last) NVN_LOG("intervalo de presentación %d\n", interval), last = interval;
    if (nvn::Window* w = nvn::g_windows.get(window)) w->present_interval = interval;
}
int nvnWindowGetNumTextures(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? static_cast<int>(w->textures.size()) : 0;
}
void nvnWindowSetCrop(void* window, int x, int y, int w, int h) {
    nvn::Window* win = nvn::g_windows.get(window);
    if (!win) return;
    static VkRect2D logged{};  // DEPURACIÓN: cada recorte nuevo
    const VkRect2D crop{{x, y}, {static_cast<uint32_t>(std::max(w, 0)), static_cast<uint32_t>(std::max(h, 0))}};
    if (memcmp(&crop, &logged, sizeof crop) != 0) NVN_LOG("recorte de la ventana %d,%d %dx%d\n", x, y, w, h), logged = crop;
    win->crop = crop;
}
// NVNrectangle: x, y, ancho, alto.
void nvnWindowGetCrop(const void* window, int* rect) {
    const nvn::Window* w = nvn::g_windows.get(window);
    rect[0] = w ? w->crop.offset.x : 0, rect[1] = w ? w->crop.offset.y : 0;
    rect[2] = w ? static_cast<int>(w->crop.extent.width) : 0, rect[3] = w ? static_cast<int>(w->crop.extent.height) : 0;
}
void* nvnWindowGetNativeWindow(const void* window) {
    const nvn::Window* w = nvn::g_windows.get(window);
    return w ? w->native_window : nullptr;
}
// Las texturas por las que rota AcquireTexture (las primeras count; el resto se quedan sin usar).
int nvnWindowGetNumActiveTextures(const void* window) {
    const nvn::Window* w = nvn::g_windows.get(window);
    if (!w) return 0;
    const int count = static_cast<int>(w->textures.size());
    return w->num_active > 0 ? std::min(w->num_active, count) : count;
}
void nvnWindowSetNumActiveTextures(void* window, int count) {
    if (nvn::Window* w = nvn::g_windows.get(window)) {
        NVN_LOG("ventana con %d texturas activas de %zu\n", count, w->textures.size());
        w->num_active = count;
    }
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnWindowSetDebugLabel(void*, const char*) {}

}  // extern "C"

NVN_PROC(nvnWindowBuilderSetDefaults);
NVN_PROC(nvnWindowBuilderSetDevice);
NVN_PROC(nvnWindowBuilderSetNativeWindow);
NVN_PROC(nvnWindowBuilderSetPresentInterval);
NVN_PROC(nvnWindowBuilderSetTextures);
NVN_PROC(nvnWindowInitialize);
NVN_PROC(nvnWindowFinalize);
NVN_PROC(nvnWindowGetPresentInterval);
NVN_PROC(nvnWindowSetPresentInterval);
NVN_PROC(nvnWindowGetNumTextures);
NVN_PROC(nvnWindowSetCrop);
NVN_PROC(nvnWindowGetCrop);
NVN_PROC(nvnWindowSetDebugLabel);
NVN_PROC(nvnWindowBuilderSetNumActiveTextures);
NVN_PROC(nvnWindowBuilderGetDevice);
NVN_PROC(nvnWindowBuilderGetNativeWindow);
NVN_PROC(nvnWindowBuilderGetPresentInterval);
NVN_PROC(nvnWindowBuilderGetNumTextures);
NVN_PROC(nvnWindowBuilderGetNumActiveTextures);
NVN_PROC(nvnWindowBuilderGetTexture);
NVN_PROC(nvnWindowGetNativeWindow);
NVN_PROC(nvnWindowGetNumActiveTextures);
NVN_PROC(nvnWindowSetNumActiveTextures);
