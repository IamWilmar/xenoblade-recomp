// nvnWindow*: la ventana NVN es un swapchain de Vulkan sobre la ventana de la APK. Las texturas
// de la ventana del juego (1280x720) son nuestras; al presentar se copian a la imagen del swapchain.
// Primer paso: se presenta el último ClearColor que el juego hizo sobre la textura.

#include <android/native_window.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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

// vkQueuePresentKHR fuera del hilo de la cola (NvnSubmit). En el Adreno la llamada no vuelve hasta que la GPU ha terminado
// lo que tiene delante (Castlevania: 6 ms de cada frame, con 11 de reproducción: 17,5 ms por frame, 57 fps). Este hilo
// espera al fence del blit (sin el cerrojo de la VkQueue) y presenta, mientras NvnSubmit reproduce el frame siguiente.
// Nunca hay más de una pendiente y nunca se adquiere otra imagen con una pendiente: present() espera a la anterior (drain)
// antes de tocar el swapchain, así que el swapchain y los juegos de present (PresentSlot) se usan igual que en línea.
// <datos>/presentar_en_cola (en vivo): como antes, en el hilo de la cola.
class PresentThread {
public:
    void push(VkSwapchainKHR swapchain, uint32_t image, VkSemaphore rendered, VkFence done) {
        std::unique_lock lock(m_);
        if (!started_) {
            started_ = true;
            std::thread([this] { loop(); }).detach();
        }
        done_cv_.wait(lock, [&] { return !pending_; });
        job_ = {swapchain, image, rendered, done};
        pending_ = true;
        cv_.notify_one();
    }
    // Espera a que la presentación pendiente (si la hay) haya salido. Devuelve lo esperado en ns.
    int64_t drain() {
        std::unique_lock lock(m_);
        if (!pending_) return 0;
        const auto t0 = std::chrono::steady_clock::now();
        done_cv_.wait(lock, [&] { return !pending_; });
        return (std::chrono::steady_clock::now() - t0).count();
    }
    // DEPURACIÓN (diag_syncs): lo que pasó esperando al fence y dentro de vkQueuePresentKHR desde la última consulta, en ns.
    void take_stats(int64_t* fence_ns, int64_t* present_ns) {
        std::lock_guard lock(m_);
        *fence_ns = fence_ns_, *present_ns = present_ns_;
        fence_ns_ = present_ns_ = 0;
    }

private:
    struct Job {
        VkSwapchainKHR swapchain;
        uint32_t image;
        VkSemaphore rendered;
        VkFence done;
    };
    void loop() {
        pthread_setname_np(pthread_self(), "NvnPresent");
        std::unique_lock lock(m_);
        for (;;) {
            cv_.wait(lock, [&] { return pending_; });
            const Job job = job_;
            lock.unlock();
            nvn::VkContext* c = nvn::vk();
            const auto t0 = std::chrono::steady_clock::now();
            vkWaitForFences(c->device, 1, &job.done, VK_TRUE, UINT64_MAX);
            const auto t1 = std::chrono::steady_clock::now();
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &job.rendered;
            pi.swapchainCount = 1;
            pi.pSwapchains = &job.swapchain;
            pi.pImageIndices = &job.image;
            {
                std::lock_guard queue_lock(nvn::vk_queue_mutex());
                vkQueuePresentKHR(c->queue, &pi);
            }
            nnsdk::frame_presented();
            const auto t2 = std::chrono::steady_clock::now();
            lock.lock();
            fence_ns_ += (t1 - t0).count();
            present_ns_ += (t2 - t1).count();
            pending_ = false;
            done_cv_.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    Job job_{};
    bool pending_ = false, started_ = false;
    int64_t fence_ns_ = 0, present_ns_ = 0;
};
PresentThread g_present_thread;

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
    // También origen de copias si el surface lo deja: el reporte del panel copia el frame (hud.cpp).
    info.imageUsage |= caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    nvn::hud::swapchain_usage(info.imageUsage);
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
// Detector de barras (<datos>/captura_barras, en vivo; Xenoblade 3): en los primeros frames tras algunos cortes de la intro
// media pantalla sale en franjas verticales negras o más oscuras, 4 de cada 8 teselas de las 48 de su desenfoque (unos 8
// frames: imposible pedir la traza a mano). Las muestras van una por tesela (48 columnas); si en bastantes filas los 12
// bloques de 4 columnas alternan claro/oscuro con la misma fase, se pide la traza completa del frame siguiente.
constexpr int kProbeW = 48, kProbeH = 18;
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
    static float lum[kProbeW * kProbeH];
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
        lum[k] = r + g + b;
    }
    if (NVN_FLAG("captura_barras")) {
        int rows[2] = {};  // filas que alternan, por fase
        for (int y = 0; y < kProbeH; y++) {
            float block[kProbeW / 4];
            for (int k = 0; k < kProbeW / 4; k++)
                block[k] = lum[y * kProbeW + 4 * k] + lum[y * kProbeW + 4 * k + 1] + lum[y * kProbeW + 4 * k + 2] +
                           lum[y * kProbeW + 4 * k + 3];
            int agree[2] = {};
            for (int k = 1; k + 1 < kProbeW / 4; k++) {
                const float around = (block[k - 1] + block[k + 1]) / 2;
                if (std::fabs(block[k] - around) < 0.04f * (block[k] + around) + 6) continue;  // sin diferencia clara
                agree[((block[k] > around) ^ (k & 1)) ? 1 : 0]++;
            }
            if (agree[0] >= 9) rows[0]++;
            if (agree[1] >= 9) rows[1]++;
        }
        static auto last_bars = std::chrono::steady_clock::now() - std::chrono::seconds(60);
        static int taken_bars = 0;
        const auto now_bars = std::chrono::steady_clock::now();
        if (std::max(rows[0], rows[1]) >= 7 && taken_bars < 6 && now_bars - last_bars >= std::chrono::seconds(4)) {
            last_bars = now_bars;
            taken_bars++;
            NVN_LOG("detector de barras: %d de %d filas alternan claro/oscuro cada 4 teselas (fase %d): traza completa del "
                    "frame siguiente\n", std::max(rows[0], rows[1]), kProbeH, rows[1] > rows[0]);
            nvn::request_trace(true);
        }
    }
    static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    static int taken = 0;
    const auto now = std::chrono::steady_clock::now();
    if (NVN_FLAG("captura_destello") && yellow * 6 >= kProbeW * kProbeH * t.sixths && taken < 4 &&
        now - last >= std::chrono::seconds(20)) {  // las muestras también se copian con captura_barras
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
    static VkBufferImageCopy regions[kProbeW * kProbeH];  // solo el hilo que presenta
    memset(regions, 0, sizeof regions);
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

// --- Resolución dinámica de Xenoblade DE 1.1.2 (ml::DrCalcGpuCost, main+0x73a5a0) ---
// El juego la decide solo cada frame (DrMan::beginSetup → DrCalcGpuCost::update): coste = GPU del frame (de sus
// timestamps, que aquí escribe la GPU) en % del presupuesto + 3. Con 94 o más en ~3 frames baja un nivel; tras 90 frames
// por debajo de 89,3 sube uno (dos por debajo de 83,66). Hay seis niveles entre la escala máxima (+0x10) y la mínima
// (+0x18), cada una multiplicada por DrPixlPostParm+0x10ac cuando DrMan+0x37b0 es 0. En portátil la escena iba de
// 784x440 hacia abajo dentro de su objetivo de 1280x720, y en el campo se quedaba en el techo con la GPU a 26 ms de 33.
// Aquí solo se cambian el techo y el suelo de la X (la escala de la resolución; la Y es otra cosa y no se toca), cada
// frame por si el mapa cambia el multiplicador (en Bionis' Leg es 0,75, así que el techo del juego ya era 960x540; en
// la Colonia 9 era 784x440): portátil 0,75-0,50 del objetivo (960x540 a 640x360) en todos los mapas, modo TV 0,60-0,34
// (1152x648 a 653x367, sin probar en combate). El nivel lo sigue eligiendo el juego con su coste. Medido el 15-11 en
// Maguel Road: la GPU del frame apenas depende de la resolución (25 ms a 640x360, 20-31 ms a 768-960 según el momento)
// y 1088x612 ya no cabe (31-35 ms, 28 fps). Con el FSR sumado al coste (reservar_fsr) se iba al suelo y ahí se quedaba.
// <datos>/sin_techo_xc (en vivo): los valores del juego. <datos>/techo_xc con "techo suelo" (fracciones del objetivo)
// para probar otros; se relee cada 5 s.
constexpr size_t kDrManGot = 0x13f1098;  // la entrada de la GOT con la dirección de la variable ml::DrMan* (DrMan::get)
constexpr size_t kDrManGet = 0x75b330;   // adrp x8; ldr x8, [x8, #0x98]; ldr x0, [x8]; ret
constexpr uint32_t kDrManGetCode[4] = {0xd00064a8, 0xf9404d08, 0xf9400100, 0xd65f03c0};
constexpr size_t kGpuCostMultLoad = 0x73a850;  // mov w8, #0x10ac; ldr d1, [x20, x8] en DrCalcGpuCost::update
constexpr uint32_t kGpuCostMultCode[2] = {0x52821588, 0xfc686a81};
constexpr size_t kDrManGpuCost = 0x990, kDrManPostParm = 0x1a0, kDrManRawScale = 0x37b0, kPostParmMult = 0x10ac;

struct XcResolution {
    float game_max = 0, game_min = 0;  // los del juego, para devolverlos con sin_techo_xc
    bool saved = false, written = false;
    float ceiling = 0, floor = 0;  // fracciones del objetivo (1280 de ancho en portátil, 1920 en modo TV)
    uint64_t frames = 0;
    bool bad_logged = false;
};
XcResolution g_xcres;

void xc_read_range(XcResolution& x) {
    const bool tv = nnsdk::tv_mode();
    x.ceiling = tv ? 0.60f : 0.75f;
    x.floor = tv ? 0.34f : 0.50f;
    const char* env = getenv("NIER_DATA");
    if (FILE* f = fopen((std::string(env ? env : "/sdcard/NieR") + "/techo_xc").c_str(), "r")) {
        float c = 0, s = 0;
        if (fscanf(f, "%f %f", &c, &s) == 2 && c > 0.2f && c <= 1.0f && s > 0.15f && s <= c) x.ceiling = c, x.floor = s;
        fclose(f);
    }
}

// Desde el hilo que presenta, una vez por frame.
void xc_resolution() {
    static const uint8_t* base = nnsdk::host().main_base;
    static const bool known = [] {
        bool ok = base && nnsdk::host().main_size > kDrManGot + 8;
        for (int k = 0; ok && k < 4; k++) ok = reinterpret_cast<const uint32_t*>(base + kDrManGet)[k] == kDrManGetCode[k];
        for (int k = 0; ok && k < 2; k++)
            ok = reinterpret_cast<const uint32_t*>(base + kGpuCostMultLoad)[k] == kGpuCostMultCode[k];
        NVN_LOG("resolución de Xenoblade: %s\n", ok ? "1.1.2 (ml::DrCalcGpuCost en DrMan+0x990): techo y suelo propios"
                                                     : "no es el main conocido: la del juego");
        return ok;
    }();
    if (!known) return;
    XcResolution& x = g_xcres;
    const uint64_t var = *reinterpret_cast<const volatile uint64_t*>(base + kDrManGot);
    const uint64_t man = var ? *reinterpret_cast<const volatile uint64_t*>(var) : 0;
    const uint64_t cost = man ? *reinterpret_cast<const volatile uint64_t*>(man + kDrManGpuCost) : 0;
    if (!cost) return;
    volatile float* f = reinterpret_cast<volatile float*>(cost);  // f[4] +0x10 máx X, f[6] +0x18 mín X, f[8] +0x20 escala X
    if (!x.saved) {
        x.game_max = f[4], x.game_min = f[6];
        x.saved = true;
        xc_read_range(x);
        NVN_LOG("resolución de Xenoblade: el juego trae máx %.3f mín %.3f; aquí %.2f-%.2f del objetivo\n", x.game_max,
                x.game_min, x.ceiling, x.floor);
    }
    if (!(x.game_max > 0.3f && x.game_max < 1.5f && x.game_min > 0.1f && x.game_min <= x.game_max)) {
        if (!x.bad_logged) NVN_LOG("resolución de Xenoblade: valores raros (máx %.3f mín %.3f): sin tocar\n", x.game_max, x.game_min);
        x.bad_logged = true;
        return;
    }
    if (NVN_FLAG("sin_techo_xc")) {
        if (x.written) f[4] = x.game_max, f[6] = x.game_min;
        x.written = false;
        return;
    }
    const bool raw = *reinterpret_cast<const volatile uint32_t*>(man + kDrManRawScale) != 0;
    const uint64_t post = *reinterpret_cast<const volatile uint64_t*>(man + kDrManPostParm);
    const float mult = raw ? 1.0f : post ? *reinterpret_cast<const volatile float*>(post + kPostParmMult) : 0.0f;
    if (!(mult > 0.05f && mult < 2.0f)) return;
    if (++x.frames % 150 == 0) {
        xc_read_range(x);
        const int level = *reinterpret_cast<const volatile int32_t*>(cost + 0x34);
        const int width = nnsdk::tv_mode() ? 1920 : 1280, height = nnsdk::tv_mode() ? 1080 : 720;
        NVN_LOG("resolución de Xenoblade: nivel %d, escena %dx%d (escala %.2f, multiplicador %.3f%s), rango %.2f-%.2f\n", level,
                static_cast<int>(width * f[8]), static_cast<int>(height * f[8]), f[8], mult, raw ? " sin aplicar" : "",
                x.ceiling, x.floor);
    }
    f[4] = x.ceiling / mult;
    f[6] = x.floor / mult;
    x.written = true;
}

// --- Resolución dinámica (Xenoblade 3 2.2.0) ---
// El juego regula su resolución con un objeto de 0x4c bytes, D = *(R + 0xaa0), con R = *(main+0x1ba3970) su renderer: un
// nivel entero en D+0x40 que su regulador (main+0xeb6d44, una vez por frame) mueve con la GPU que mide entre sus dos
// TIMESTAMP. Cada nivel de 1 a 5 quita un 6 % de escala (del máximo al mínimo de su lib_nx.bin, 1,0 y 0,70) y cada nivel
// negativo, solo en portátil con el reescalado temporal, sube 0,025 la proporción entre la escena y la salida del
// reescalado (0,53). En portátil, con base 1280x720: escena 776x440 / 744x424 / 712x400 (niveles -3, -2, -1), 680x384
// (0), 640x360, 600x344, 560x320, 520x296 y 480x272 (5), con el reescalado sacando 1280x720 hasta el nivel 0 y de ahí
// bajando hasta 896x504. En los eventos (sin reescalado temporal, niveles 0..5): 960x544, 896x504, 848x480, 784x440,
// 736x416, 672x376. En modo TV (0..5): de 960x544 a 672x384, con salida de 1920x1080 a 1344x760.
// Aquí esa medida no le sirve: el tramo entre sus dos sellos mide al ejecutor (la GPU espera parada al envío siguiente)
// y, cuando el juego va dos frames por delante de la cola, lee el sello de comienzo de un frame con el de final de otro.
// Se quedaba en 480x272 en el campo con la GPU ocupada 20-24 ms de 33, y eso es lo borroso.
// No hace falta parchear código: D+0x48 es una retención del propio juego. Con D+0x48 = 1 y D+0 = 1 (activada), su
// regulador se salta toda la decisión (main+0xeb6e3c..0xeb6e48) y solo recalcula la escala a partir del nivel. El nivel
// lo ponemos nosotros, una vez por frame presentado, y lo demás (tamaños, recrear los render targets, el reescalado) lo
// sigue haciendo el juego igual que cuando el nivel lo mueve él. Nadie más escribe D+0x40, +0x44 ni +0x48 (el
// constructor deja la retención a 0). Solo se escriben esos tres campos: la retención, el nivel y, a 0, su contador.
// Con D+0 = 0 (lo apaga código del juego, main+0xed8fdc) o E+0x2096 = 0 el juego usa su escala fija y pone el nivel a 0
// en cada frame: ahí no se escribe nada (un nivel nuestro solo haría bailar la proporción y recrear los targets).
// Límites: el nivel nunca sale de -3..5 en portátil con reescalado temporal ni de 0..5 en lo demás (por encima de 5 el
// juego lee fuera de una tabla de 6 floats, main+0xf55de4; por debajo de -3, o negativo en modo TV, saldría una
// proporción que el juego no usa nunca y una escena mayor que los 960x544 con los que dimensiona sus targets).
//
// El regulador (Xc3Regulator), con una muestra por frame del juego: la GPU ocupada entre sus dos sellos (lo que la GPU
// trabajó), el tramo entre ellos (lo que tardó el frame en la GPU, esperas al ejecutor incluidas: el frame llega a 30 fps
// mientras el tramo quepa en el presupuesto menos ~1 ms de presentar; en los logs, tramo + hueco = 33,3 ms siempre) y el
// ejecutor del último frame. Presupuesto = intervalo del juego (main+0x1b42db0: 2) x 16,67 ms. Primero 30 fps, luego
// nitidez:
//  - BAJA (nivel + 1) como pronto 1 s después del cambio anterior (cada cambio le hace recrear unos 60 render targets):
//    deprisa, con los últimos 8 frames, si la GPU ocupada media pasa del presupuesto o si 6 de los 8 tramos pasan del
//    97 % (32,3 ms: no llegan) con media por encima del 103 %, y dos niveles de golpe por encima del 110 %; con 1 s (30
//    frames), si la GPU ocupada media pasa del 94 % (31,3 ms) o 20 tramos no llegan; con 3 s en el mismo nivel, si no
//    llegan 18 de 90. Un frame que no llega es un vsync de más a 60 Hz: se ve. Las reglas de "no llegan" piden además
//    que el ejecutor medio quepa en el 94 %: si el frame lo pierde la CPU, bajar la resolución no lo arregla.
//    La de 18 de 90 cuenta solo los tramos que no llegan con la GPU ocupada por encima del 94 % (31,3 ms). En el primer
//    combate (traza_1080-1630) saltaba con la ocupada media en 26,9-29,6 ms y el tramo 2-4,5 ms por encima (la GPU
//    parada esperando al ejecutor, que va a 20 ms/frame), sin que se perdieran frames: a nivel 3 y 4 el ritmo daba 30,1
//    fps con 2-8 frames de más de 40 ms cada 10 s, los mismos que a nivel 5 fijo en el campo (3-11). Bajar no los quitaba:
//    de 3 a 4 cada 10-20 s, una y otra vez. Un frame con la ocupada por debajo del 94 % tuvo al menos 2 ms de GPU libres.
//    Con <datos>/sin_tardanza_gpu, todos los tramos, como antes.
//  - SUBE (nivel - 1) con 3 s desde el último cambio (10 s tras una bajada; el doble, hasta 30 s, si la bajada deshizo
//    una subida de hace menos de 20 s; de vuelta a 3 s tras 60 s sin bajar), mirando todo lo medido desde ese cambio
//    hasta 10 s: lo previsto con el nivel de arriba tiene que caber, ocupada media + coste < 85 % (28,3 ms), tramo medio
//    + coste < 87 % (29,0 ms) y menos de 1 de cada 20 frames por encima del presupuesto en GPU o del 97 % en tramo con
//    ese coste sumado. Coste = 0,30 x ocupada x (píxeles de arriba / píxeles de ahora - 1): en los logs la GPU ocupada
//    depende poco de los píxeles de la escena (unos 3 ms más con un 58 % más, entre un quinto y un cuarto), así que 0,30
//    queda del lado prudente. Medido después en modo TV (campo, niveles fijos: 24,2 ms a 672x384, 27,6 a 792x448, 30,3 a
//    848x480, 33,5 a 960x544) la parte que escala es 0,35-0,55: de 736x416 a 792x448 se prevén 27,1 ms y son 27,6. Ese
//    medio milisegundo cabe en el margen del 85 %, y la subida que aun así no aguante la para la memoria de abajo.
//  - Memoria por nivel: bajar de un nivel a menos de 30 s de haber subido a él es una subida que no aguantó, y el nivel
//    queda vetado 60 s (el doble cada vez que vuelva a fallar, hasta 4 min; si aguanta 30 s, de vuelta a 60) salvo que la
//    ocupada media del nivel de abajo quede por debajo de la que había al subir menos el coste previsto del nivel (como
//    poco 1 ms): la escena pesa menos que cuando falló. Así no rehace unos 60 render targets cada 10-20 s para volver a
//    bajar. Las bajadas siguen igual de rápidas, y tras una bajada la espera para subir es la de siempre (10 s), sin
//    doblarla: lo largo es solo para el nivel que falló. Se olvida al empezar, al cambiar de modo y tras una carga.
//    Con <datos>/sin_memoria_resolucion, sin memoria y con la espera doblada de antes; con <datos>/sin_regulador_nuevo,
//    ni memoria ni la cuenta nueva de los 18 de 90: el regulador de antes entero.
//  - Tras una carga (8 frames seguidos con la GPU ocupada menos de 3 ms: en las cargas son 0,5-1,2) y al empezar, el nivel
//    2 (600x344) como mucho, y a medir de nuevo: el juego se iba a 776x440 durante la carga y luego bajaba tres veces a la
//    vista. Al cambiar de modo (evento, base) el nivel se recorta al rango nuevo y también se mide de nuevo.
//  - Los frames de más de 1,5 presupuestos cuentan como 1,5: un tirón (una pipeline, un fichero) no es carga de la escena.
// Con cargas inventadas (ruido de 1 a 3,5 ms, tirones, carga, campo, combate y vuelta) sube a un nivel por cada 3 s hasta
// quedarse con un 2-6 % de tramos justos y un cambio cada 15-60 s según el ruido; en el combate pesado de los logs baja al
// mínimo en 1-2 s.
// Todos los números son de partida, sacados de dos logs del dispositivo: falta medir el coste real por nivel (con
// resolucion_fija en n-3, n0 y n5 en el mismo sitio del campo y en un combate, mirando "ocupada" en el ritmo).
// En el combate pesado de los logs el juego ya no llegaba a 30 con 480x272 (ocupada 28-33 ms, ejecutor 26-31): ahí este
// regulador también baja al mínimo; quien prefiera nitidez a esos 2-3 fps tiene resolucion_suelo.
// Interruptores (en vivo; lo de dentro se lee como mucho una vez por segundo):
//   <datos>/resolucion_fija: sin regulador, el nivel cuya escena queda más cerca de ese factor x 960x544 (la escena
//     completa del juego): 1.0 es lo más nítido del modo (776x440 en el campo, 960x544 en los eventos), 0.75 son 712x400
//     en el campo, 0.5 son 480x272; vacío, 1.0. O un nivel del juego con una n delante: n-3, n0, n5.
//   <datos>/resolucion_suelo: el regulador no baja de ahí (mismo formato; vacío o ilegible, sin suelo).
//   <datos>/sin_resolucion_propia: se quita la retención y el juego vuelve a su regla, sin tocar nada más.
//
// Suelo extra, por debajo del nivel 5 y solo en modo TV con el reescalado temporal. En el primer combate grande
// (will_ulula_1943) el regulador se queda en el 5 (672x384, lo mínimo del juego en TV) con la GPU ocupada 30-34 ms de 33,3:
// la GPU sola ya no cabe y no queda escalón. El juego saca la escala de cada frame de su nivel y de los límites de D:
// escala = trunc((máx - (máx - mín)/5 x L) x 100)/100, recortada a [mín, máx] (main+0xeb6ee4). Con el nivel retenido en
// 5 eso es el mínimo, sea cual sea el máximo: basta con escribir el mínimo (D+0x24, +0x28) para bajar la escala. Solo el
// mínimo, y no los dos como en las notas de ingeniería inversa (máx = mín = F): con el nivel en 5 cualquier estado a
// medias entre dos escrituras da también el mínimo, y el máximo del juego (1,0) no se toca nunca. Los escalones son el
// mínimo del juego menos 0,05 cada uno hasta el suelo: con 0,70 y el suelo por defecto 0,55, escala 0,65, 0,60 y 0,55,
// escena 624x352, 576x328 y 528x304 (salida 1248x704, 1152x648 y 1056x600), y el regulador los ve como niveles 6, 7 y 8.
// En portátil sin reescalado temporal el juego multiplica los límites por 0,75 y en los eventos la base es otra: ahí no se
// escribe nada (sin comprobar que la misma cuenta valga). Entrar y moverse entre los escalones solo por la GPU: las
// reglas de "los frames no llegan" (que también saltan con el ejecutor justo) no bajan del 5, y para subir no se mira el
// tramo, solo la GPU ocupada (con el coste previsto del escalón de arriba) y los frames que no llegan con la GPU justa;
// la espera, el intervalo mínimo entre cambios y la memoria por nivel son las de siempre. El mínimo del juego se lee y se
// guarda antes de la primera escritura y se devuelve al salir de los escalones, al cambiar de modo, con el juego en
// escala fija, con resolucion_fija o resolucion_suelo, al quitar la retención y al poner sin_suelo_extra; si el juego
// rehace D, ese D no se toca (el viejo ya no es suyo) y se empieza de cero. Nadie más que el cargador de lib_nx.bin
// escribe esos campos (una vez, al iniciar el renderer).
//   <datos>/sin_suelo_extra: sin escalones por debajo del nivel 5 (y el mínimo del juego devuelto): el regulador de antes.
//   <datos>/suelo_extra_minimo: la escala más baja permitida (dentro, un número de 0,40 a 0,69; sin él, 0,55). En modo TV
//     la escala coincide con el factor de 960x544 (0,55 son 528x304).
constexpr size_t kXc3Renderer = 0x1ba3970;  // R: el renderer (dato de main, en .bss; lo devuelve main+0xed2ff4)
constexpr size_t kXc3Upscale = 0x1b2590d;   // u8: reescalado temporal en uso (lo lee main+0xe98f4c; los eventos lo apagan)
constexpr size_t kXc3Interval = 0x1b42db0;  // i32: el intervalo del juego (2: 30 fps; <= 0 cuenta como 2)
constexpr size_t kXc3Docked = 0x49c0;       // R+0x49c0, i32: 0 portátil, 1 en la base (el juego lo reescribe cada frame)
// D = *(R + 0xaa0): el objeto de la resolución dinámica.
constexpr size_t kXc3D = 0xaa0;
constexpr size_t kXc3DEnabled = 0x00;                                             // u8
constexpr size_t kXc3DMaxX = 0x1c, kXc3DMaxY = 0x20, kXc3DMinX = 0x24, kXc3DMinY = 0x28;  // float: límites de lib_nx.bin
constexpr size_t kXc3DLevel = 0x40, kXc3DCount = 0x44;                            // i32: nivel y contador de histéresis
constexpr size_t kXc3DHold = 0x48;                                                // u8: la retención
// E = *(R + 0x1a0): los ajustes de la vista 0 (lo que el regulador del juego recibe en x1). Solo se leen.
constexpr size_t kXc3E = 0x1a0;
constexpr size_t kXc3EAllowed = 0x2096;                              // u8: la vista admite escala
constexpr size_t kXc3EScaleX = 0x20a4, kXc3EScaleY = 0x20a8;         // float: la escala aplicada
constexpr size_t kXc3EUpscale = 0x20b0;                              // u8: reescalado temporal de la vista
constexpr size_t kXc3ERatioDocked = 0x20b4, kXc3ERatioHandheld = 0x20b8;  // float: proporción escena/salida (0,5 y 0,53)
constexpr size_t kXc3ERatioNow = 0x20c0;                             // float: la proporción en uso
constexpr size_t kXc3EMulX = 0x20c8, kXc3EMulY = 0x20cc;             // float: 0,75 en portátil sin reescalado temporal
struct Xc3Word {
    size_t offset;
    uint32_t code;
};
// Las instrucciones por las que se llega a cada cosa que se lee o se escribe: si alguna no es esta, no es el main conocido.
constexpr Xc3Word kXc3Signature[] = {
    {0xed2ff4, 0xF0006488}, {0xed2ff8, 0xF946FD08}, {0xed2ffc, 0xF9400100},  // el getter de R (adrp, ldr, ldr)
    {0xed4c34, 0xF9455260}, {0xed4c38, 0xF940D261}, {0xed4c3c, 0x97FF8842},  // x0 = R->+0xaa0, x1 = R->+0x1a0, bl regulador
    {0xeb6e3c, 0x39412269}, {0xeb6e40, 0x39400268}, {0xeb6e44, 0x34000069}, {0xeb6e48, 0x350003A8},  // la retención
    {0xeb6ee4, 0xB9404268},                                                  // el nivel que lee el cálculo de la escala
    {0xf54d40, 0xF9455009}, {0xf54d44, 0xB9404129},                          // y el que lee la proporción
};
constexpr int kXc3Worst = 5;   // el nivel más borroso del juego
constexpr int kXc3ExtraMax = 4;  // escalones del suelo extra como mucho (niveles 6..9 del regulador; nunca del juego)
constexpr int kXc3Start = 2;   // lo más nítido con lo que se empieza, y tras una carga

const uint8_t* xc3_main() {
    static const uint8_t* const main = []() -> const uint8_t* {
        const uint8_t* base = nnsdk::host().main_base;
        bool ok = base && nnsdk::host().main_size > kXc3Renderer + 8;
        if (ok)
            for (const Xc3Word& w : kXc3Signature) ok = ok && *reinterpret_cast<const uint32_t*>(base + w.offset) == w.code;
        NVN_LOG("resolución (Xenoblade 3): %s\n",
                ok ? "main 2.2.0 reconocido (renderer en main+0x1ba3970, nivel en +0xaa0 -> +0x40, retención en +0x48)"
                   : "no es el main de Xenoblade 3 2.2.0: sin tocar su resolución");
        return ok ? base : nullptr;
    }();
    return main;
}

template <class T>
T xc3_peek(uint64_t address) {
    return *reinterpret_cast<const volatile T*>(address);
}
template <class T>
void xc3_poke(uint64_t address, T value) {
    *reinterpret_cast<volatile T*>(address) = value;
}

// Lo que decide los tamaños, leído del juego en cada frame.
struct Xc3Mode {
    bool docked = false;  // R+0x49c0 == 1
    bool flag = false;    // reescalado temporal en uso (el global)
    bool view = false;    // y el de la vista: con él la escena es una proporción de la salida del reescalado
    float max_x = 1.0f, max_y = 1.0f, min_x = 0.7f, min_y = 0.7f;
    float ratio = 0.53f;  // la proporción base del modo
    float mul_x = 0.75f, mul_y = 0.75f;
    // El nivel más nítido: los negativos solo los usa el juego en portátil con el reescalado temporal.
    int sharpest() const { return !docked && flag && view ? -3 : 0; }
    int key() const { return (docked ? 1 : 0) | (flag ? 2 : 0) | (view ? 4 : 0); }
};
struct Xc3Size {
    int scene_w = 0, scene_h = 0, out_w = 0, out_h = 0;
};
int round_up8(int v) { return v + (-v & 7); }

// main+0xf54f30: la salida es la base por la escala aplicada y la escena, la salida por la proporción; todo truncado y
// subido al múltiplo de 8. La base es 1920x1080 solo en la base con el reescalado temporal.
Xc3Size xc3_sizes(const Xc3Mode& m, float fx, float fy, float ratio) {
#pragma clang fp contract(off)
    const float base_w = m.docked && m.flag ? 1920.0f : 1280.0f, base_h = m.docked && m.flag ? 1080.0f : 720.0f;
    Xc3Size s;
    s.out_w = round_up8(static_cast<int>(fx * base_w));
    s.out_h = round_up8(static_cast<int>(fy * base_h));
    s.scene_w = round_up8(static_cast<int>(ratio * static_cast<float>(s.out_w)));
    s.scene_h = round_up8(static_cast<int>(ratio * static_cast<float>(s.out_h)));
    return s;
}

// Los tamaños que el juego saca de un nivel: la misma aritmética en float de 32 bits y en el mismo orden (sin fundir
// multiplicación y resta: el juego trunca a centésimas y un redondeo distinto cambia 0,88 por 0,87). Reproduce todos los
// tamaños vistos en los logs del dispositivo.
Xc3Size xc3_size(const Xc3Mode& m, int level) {
#pragma clang fp contract(off)
    const float steps = static_cast<float>(std::max(level, 0));
    auto applied = [&](float hi, float lo, float mul, float base) {
#pragma clang fp contract(off)
        if (!m.docked && !m.flag) hi = mul * hi, lo = mul * lo;  // main+0xeb6f6c: portátil sin reescalado temporal
        const float step = (hi - lo) / 5.0f;                     // main+0xeb6ee4: la escala del nivel, en centésimas
        float s = static_cast<float>(static_cast<int>((hi - step * steps) * 100.0f)) * 0.01f;
        if (lo > s)
            s = lo;
        else if (!(hi >= s))
            s = hi;
        const float q = static_cast<float>(static_cast<int>(s * 100.0f)) * 0.01f;  // main+0xf54c3c: la aplicada
        const float w = static_cast<float>(round_up8(static_cast<int>(q * base)));
        return std::min((w + 0.1f) / base, 1.0f);
    };
    const bool big = m.docked && m.flag;
    const float fx = applied(m.max_x, m.min_x, m.mul_x, big ? 1920.0f : 1280.0f);
    const float fy = applied(m.max_y, m.min_y, m.mul_y, big ? 1080.0f : 720.0f);
    const float ratio = m.view ? m.ratio + static_cast<float>(std::max(-level, 0)) * 0.025f : 1.0f;  // main+0xf54d44
    return xc3_sizes(m, fx, fy, ratio);
}

// Los escalones del suelo extra (ver arriba): el mínimo del juego menos 0,05 cada uno, sin bajar de `lowest`. El nivel 6
// del regulador es scale[0], el 7 scale[1]...
struct Xc3Extra {
    int count = 0;
    float scale[kXc3ExtraMax] = {};
};
Xc3Extra xc3_extra_steps(float game_min, double lowest) {
    Xc3Extra x;
    const long top = std::lround(static_cast<double>(game_min) * 100.0), bottom = std::lround(lowest * 100.0);
    for (long h = top - 5; h >= bottom && x.count < kXc3ExtraMax; h -= 5) x.scale[x.count++] = static_cast<float>(h) / 100.0f;
    return x;
}
// Los tamaños de un nivel del regulador, también los del suelo extra: el juego en el nivel 5 con el mínimo escrito (el
// máximo sigue siendo el suyo).
Xc3Size xc3_size_extra(const Xc3Mode& m, const Xc3Extra& extra, int level) {
    if (level <= kXc3Worst || extra.count == 0) return xc3_size(m, std::min(level, kXc3Worst));
    Xc3Mode low = m;
    low.min_x = low.min_y = extra.scale[std::min(level - kXc3Worst, extra.count) - 1];
    return xc3_size(low, kXc3Worst);
}

// Lo escrito dentro de resolucion_fija o resolucion_suelo: un factor de la escena completa (960x544) o, con una n delante,
// un nivel del juego.
struct Xc3Request {
    bool valid = false, is_level = false;
    double value = 1.0;
    bool operator==(const Xc3Request& o) const { return valid == o.valid && is_level == o.is_level && value == o.value; }
};
// El nivel del modo cuya escena queda más cerca de lo pedido (a igual distancia, el más nítido).
int xc3_level_for(const Xc3Mode& m, const Xc3Request& q) {
    if (q.is_level) return std::clamp(static_cast<int>(q.value), m.sharpest(), kXc3Worst);
    int best = kXc3Worst;
    double best_error = 1e9;
    for (int level = m.sharpest(); level <= kXc3Worst; level++) {
        const double error = std::fabs(xc3_size(m, level).scene_w / 960.0 - q.value);
        if (error < best_error - 1e-6) best = level, best_error = error;
    }
    return best;
}
// Un interruptor con un número dentro. Se lee como mucho una vez por segundo y solo con el interruptor puesto: es un
// fopen en la SD desde el hilo que presenta.
class Xc3Setting {
public:
    Xc3Setting(const char* name, bool empty_is_one) : name_(name), empty_is_one_(empty_is_one) {}
    const Xc3Request& get() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_) return value_;
        next_ = now + std::chrono::seconds(1);
        Xc3Request r;
        char text[32] = "";
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/" + name_).c_str(), "r")) {
            const size_t n = fread(text, 1, sizeof text - 1, f);
            text[n] = 0;
            fclose(f);
        }
        const char* p = text;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        char* end = nullptr;
        if (*p == 'n' || *p == 'N') {
            const long level = strtol(p + 1, &end, 10);
            if (end != p + 1 && level >= -3 && level <= kXc3Worst) r = {true, true, static_cast<double>(level)};
        } else if (*p) {
            std::string number(p);
            std::replace(number.begin(), number.end(), ',', '.');  // 0,75 también vale
            const double factor = strtod(number.c_str(), &end);
            if (end != number.c_str() && factor > 0.0 && factor <= 4.0) r = {true, false, factor};
        } else if (empty_is_one_) {
            r = {true, false, 1.0};
        }
        if (!(r == value_) && nvn::first_times(logged_, 40)) {
            if (!r.valid)
                NVN_LOG("resolución (Xenoblade 3): %s no se entiende (\"%.20s\"): un factor como 0.75 o un nivel como n2\n",
                        name_, p);
            else if (r.is_level)
                NVN_LOG("resolución (Xenoblade 3): %s pide el nivel %d\n", name_, static_cast<int>(r.value));
            else
                NVN_LOG("resolución (Xenoblade 3): %s pide el factor %.3f (x 960x544)\n", name_, r.value);
        }
        value_ = r;
        return value_;
    }

private:
    const char* name_;
    bool empty_is_one_;
    Xc3Request value_;
    std::chrono::steady_clock::time_point next_{};
    int logged_ = 0;
};

// Una muestra por frame del juego, en ns.
struct Xc3Sample {
    double busy = 0, span = 0, run = 0;  // GPU ocupada entre los sellos, el tramo entre ellos y el ejecutor del frame
};
// El regulador: solo decide el nivel (sin memoria del juego, para poder probarlo aparte con cargas inventadas).
struct Xc3Regulator {
    static constexpr int kRing = 300;           // 10 s
    static constexpr int kMinGap = 30;          // frames entre dos cambios: 1 s
    static constexpr int kUpWait = 90, kUpWaitAfterDown = 300, kUpWaitMax = 900;  // frames sin cambios antes de subir
    static constexpr double kLightNs = 3.0e6;   // por debajo, carga o pantalla en negro
    int level = kXc3Start;
    Xc3Sample ring[kRing];
    int count = 0, next = 0;
    int since_change = 0;                       // frames medidos desde el último cambio, o desde que se empezó de nuevo
    int since_up = 1 << 20, since_down = 1 << 20;
    int up_wait = kUpWait;
    int light = 0;
    bool loading = false;
    // Para el log del cambio: por qué, y las medias con las que se decidió.
    const char* reason = "";
    double seen_busy = 0, seen_span = 0, seen_run = 0;
    // La memoria por nivel (índice nivel + 3): la última subida a ese nivel que no aguantó.
    struct Failed {
        int left = 0;     // frames medidos que le quedan al veto (0: se puede subir)
        int wait = 0;     // el último veto, que se dobla si la subida vuelve a fallar
        double busy = 0;  // la GPU ocupada media del nivel de abajo con la que se subió
    };
    static constexpr int kFailWindow = 900;                   // bajar a menos de 30 s de subir: la subida no aguantó
    static constexpr int kFailFirst = 1800, kFailMax = 7200;  // el veto: 60 s, y doblando hasta 4 min
    static constexpr int kNoLevel = 99;
    Failed failed[kXc3Worst + 4 + kXc3ExtraMax];
    int raised_to = kNoLevel;  // el nivel de la última subida (since_up cuenta desde ella), mientras no se baje
    double raised_busy = 0;    // y la ocupada media con la que se decidió
    // Para el log del cambio: el veto que deja la bajada de este frame.
    bool marked = false;
    int marked_level = 0, marked_wait = 0;
    double marked_busy = 0;
    // Para el ritmo (los pone a cero): frames en que la subida cabía y la paró un veto, y frames en que los 18 de 90 de
    // antes habrían bajado con tramos que no llegan sin la GPU justa.
    uint64_t vetoed = 0, cpu_late = 0;

    // A medir de nuevo: lo anterior era de otra escena o de otro modo, y lo que se aprendió por nivel, también.
    void restart() {
        count = next = 0;
        since_change = 0;
        for (Failed& f : failed) f = {};
        raised_to = kNoLevel;
    }
    // Una muestra más. sharpest y worst: el rango permitido ahora; up_pixels: píxeles de la escena del nivel de arriba
    // entre los del actual. memory: la memoria por nivel (sin ella, la espera doblada de antes); gpu_late: los 18 de 90
    // cuentan solo los tramos con la GPU justa. gpu_from: desde ese nivel hacia abajo (el suelo extra) solo cuenta la GPU:
    // para bajar de él, solo las reglas de la GPU ocupada y la de los tramos con la GPU justa, y para subir a él desde
    // abajo no se mira el tramo. Con gpu_from >= worst no cambia nada. Deja el nivel nuevo en `level`.
    void step(const Xc3Sample& in, double budget, int sharpest, int worst, double up_pixels, bool memory, bool gpu_late,
              int gpu_from) {
        marked = false;
        level = std::clamp(level, sharpest, worst);
        if (in.busy < kLightNs) {
            if (light < 8 && ++light == 8) {  // una carga: el cambio no se ve, y lo que venga puede pesar más
                loading = true;
                level = std::clamp(std::max(level, kXc3Start), sharpest, worst);
                reason = "carga";
            }
        } else {
            if (loading) restart();
            light = 0;
            loading = false;
        }
        if (loading) return;
        const double cap = 1.5 * budget;  // un tirón de 100 ms (una pipeline, un fichero) no es carga de la escena
        ring[next] = {std::min(in.busy, cap), std::min(in.span, cap), std::min(in.run, cap)};
        next = (next + 1) % kRing;
        count = std::min(count + 1, kRing);
        since_change = std::min(since_change + 1, 1 << 20);
        since_up = std::min(since_up + 1, 1 << 20);
        since_down = std::min(since_down + 1, 1 << 20);
        if (since_down == 1800) up_wait = kUpWait;  // 60 s sin bajar: las subidas vuelven a su espera normal
        for (Failed& f : failed) f.left = std::max(f.left - 1, 0);
        // Una subida que aguanta 30 s: si ese nivel vuelve a fallar más adelante, es otra escena; el veto, desde 60 s.
        if (since_up == kFailWindow && level == raised_to) failed[level + 3] = {};
        const double tight = 0.97 * budget;
        auto last = [&](int k) -> const Xc3Sample& { return ring[(next - 1 - k + 2 * kRing) % kRing]; };
        // De los últimos n frames: las medias, cuántos pasan del presupuesto en GPU ocupada y cuántos tramos no llegan
        // (con `extra` sumado: lo que costaría el nivel de arriba), y de esos, en late_gpu, los que además tenían la GPU
        // ocupada por encima del 94 %.
        int late_gpu = 0;
        auto sums = [&](int n, double extra, Xc3Sample& mean, int& over, int& late) {
            mean = {};
            over = late = late_gpu = 0;
            for (int k = 0; k < n; k++) {
                const Xc3Sample& s = last(k);
                mean.busy += s.busy, mean.span += s.span, mean.run += s.run;
                over += s.busy + extra > budget;
                const bool missed = s.span + extra > tight;
                late += missed;
                late_gpu += missed && s.busy + extra > 0.94 * budget;
            }
            mean.busy /= n, mean.span /= n, mean.run /= n;
        };
        Xc3Sample mean;
        int over = 0, late = 0;
        if (count >= 8 && since_change >= kMinGap && level < worst) {
            // Deprisa, con 8 frames: la GPU sola no cabe, o casi ninguno llega y por bastante.
            sums(8, 0.0, mean, over, late);
            const bool gpu_only = level >= gpu_from;  // de aquí abajo es el suelo extra: solo la GPU lo justifica
            int down = 0;
            if (mean.busy > budget ||
                (!gpu_only && late >= 6 && mean.span > 1.03 * budget && mean.run < 0.94 * budget)) {
                down = mean.busy > 1.10 * budget || (!gpu_only && mean.span > 1.10 * budget) ? 2 : 1;
                // Los dos de golpe por el tramo no entran en el suelo extra: hasta gpu_from, como mucho.
                if (!(mean.busy > 1.10 * budget) && level + down > gpu_from) down = std::max(gpu_from - level, 1);
                reason = mean.busy > budget ? "la GPU no cabe" : "los frames no llegan";
            }
            if (!down && count >= 30) {  // Con 1 s: dos de cada tres no llegan, o la GPU va justa de continuo.
                sums(30, 0.0, mean, over, late);
                if (mean.busy > 0.94 * budget || (!gpu_only && late >= 20 && mean.run < 0.94 * budget)) {
                    down = 1;
                    reason = mean.busy > 0.94 * budget ? "la GPU va justa" : "dos de cada tres frames no llegan";
                }
            }
            if (!down && count >= 90 && since_change >= 90) {  // Con 3 s en este nivel: uno de cada cinco no llega.
                sums(90, 0.0, mean, over, late);
                // Con la GPU justa: un tramo que no llega con la GPU ocupada por debajo del 94 % se fue esperando al
                // ejecutor, y eso no lo arregla bajar (ver arriba).
                if ((gpu_late || gpu_only ? late_gpu : late) >= 18 && mean.run < 0.94 * budget) {
                    down = 1;
                    reason = "uno de cada cinco frames no llega";
                } else if (late >= 18 && mean.run < 0.94 * budget) {
                    cpu_late++;
                }
            }
            if (down) {
                if (memory && since_up < kFailWindow && level == raised_to) {  // la subida no aguantó: ese nivel, vetado
                    Failed& f = failed[level + 3];
                    f.wait = f.wait ? std::min(2 * f.wait, kFailMax) : kFailFirst;
                    f.left = f.wait;
                    f.busy = raised_busy;
                    marked = true, marked_level = level, marked_wait = f.wait, marked_busy = raised_busy;
                }
                raised_to = kNoLevel;
                level = std::min(level + down, worst);
                if (memory) {
                    up_wait = kUpWaitAfterDown;  // 10 s; lo largo es el veto, y solo para el nivel que no aguantó
                } else {
                    // La subida no cabía: la próxima, más tarde. Y tras cualquier bajada, 10 s antes de volver a probar.
                    up_wait = std::min(since_up < 600 ? up_wait * 2 : up_wait, kUpWaitMax);
                    up_wait = std::max(up_wait, kUpWaitAfterDown);
                }
                since_change = since_down = 0;
                seen_busy = mean.busy, seen_span = mean.span, seen_run = mean.run;
                return;
            }
        }
        // Subir, con lo medido desde el último cambio (hasta 10 s): justo después de una subida o de empezar son 3 s y se
        // sube deprisa; con el nivel asentado, o tras una bajada, tienen que caber los 10 s enteros.
        const int n = std::min({count, since_change, static_cast<int>(kRing)});
        if (n >= kUpWait && since_change >= up_wait && level > sharpest) {
            sums(n, 0.0, mean, over, late);
            const double cost = 0.30 * mean.busy * (up_pixels - 1.0);
            // Dentro del suelo extra no se mira el tramo: se entró por la GPU, y un tramo largo por el ejecutor no lo
            // arregla quedarse abajo.
            const bool gpu_only = level > gpu_from;
            if (mean.busy + cost < 0.85 * budget && (gpu_only || mean.span + cost < 0.87 * budget)) {
                Xc3Sample unused;
                sums(n, cost, unused, over, late);
                if (gpu_only) late = late_gpu;
                // Un nivel que no aguantó hace poco: solo si la escena pesa ahora menos que cuando se subió a él, por lo
                // que cuesta el nivel (como poco 1 ms); si no, a esperar a que pase el veto.
                const Failed& f = failed[level - 1 + 3];
                const bool held = memory && f.left > 0;
                const bool lighter = mean.busy + std::max(cost, 0.03 * budget) < f.busy;
                if (over * 20 < n && late * 20 < n) {
                    if (held && !lighter) {
                        vetoed++;
                    } else {
                        level--;
                        reason = held ? "hay sitio: la escena pesa menos que cuando no aguantó" : "hay sitio";
                        raised_to = level, raised_busy = mean.busy;
                        since_change = since_up = 0;
                        seen_busy = mean.busy, seen_span = mean.span, seen_run = mean.run;
                    }
                }
            }
        }
    }
};

enum class Xc3State { kUnknown, kWaiting, kOwn, kFixed, kGame, kPaused, kOdd };
struct Xc3Resolution {
    Xc3Regulator reg;
    Xc3State state = Xc3State::kUnknown;
    bool active = false;         // el nivel es nuestro (se escribió en el frame anterior)
    int mode = -1;               // Xc3Mode::key del frame anterior
    uint64_t wrote_d = 0;        // el D en el que se escribió en el frame anterior con la retención ya puesta de antes
    int written = 0;             // y el nivel
    uint64_t busy_serial = 0, gpu_serial = 0;
    int unmeasured = 0;          // frames presentados seguidos sin medida de GPU ocupada
    uint64_t changes = 0;        // para el ritmo
    int shown_level = 0;         // el nivel en vigor (el del juego, si la regla es suya)
    Xc3Size live;                // lo que el juego tiene aplicado
    int logged = 0, logged_state = 0, logged_odd = 0, logged_moved = 0, logged_span = 0;
    // El suelo extra: el D cuyo mínimo (D+0x24, +0x28) es nuestro (0: ninguno), el mínimo del juego leído antes de la
    // primera escritura y lo escrito en los dos campos.
    uint64_t extra_d = 0;
    float game_min_x = 0, game_min_y = 0, extra_written = 0;
    float extra_scale = 0;                         // la escala del escalón en vigor (0: ninguno), para el ritmo
    uint64_t extra_frames = 0, extra_restores = 0;  // para el ritmo
    int logged_extra = 0, logged_extra_floor = 0;
};
Xc3Resolution g_xc3;

const char* xc3_mode_name(const Xc3Mode& m) {
    return m.docked ? (m.flag && m.view ? "modo TV" : "modo TV sin reescalado temporal")
                    : (m.flag && m.view ? "portátil" : "portátil sin reescalado temporal");
}

// Desde el hilo que presenta, una vez por frame.
void xc3_resolution() {
    Xc3Resolution& x = g_xc3;
    const uint8_t* main = xc3_main();
    if (!main) return;
    const auto base = reinterpret_cast<uint64_t>(main);
    const uint64_t wrote_d = std::exchange(x.wrote_d, 0);
    const bool was_active = std::exchange(x.active, false);
    auto note_state = [&](Xc3State s, const char* text) {
        if (s != x.state && nvn::first_times(x.logged_state, 40)) NVN_LOG("resolución (Xenoblade 3): %s\n", text);
        x.state = s;
    };
    auto pointer = [](uint64_t p) { return p >= 0x10000 && p < (1ull << 48) && !(p & 3); };  // un puntero de usuario, alineado
    const uint64_t r = xc3_peek<uint64_t>(base + kXc3Renderer);
    const uint64_t d = pointer(r) ? xc3_peek<uint64_t>(r + kXc3D) : 0;
    const uint64_t e = pointer(r) ? xc3_peek<uint64_t>(r + kXc3E) : 0;
    if (!pointer(d) || !pointer(e)) {  // aún sin renderer
        x.state = Xc3State::kWaiting;
        return;
    }
    Xc3Mode m;
    m.docked = xc3_peek<int32_t>(r + kXc3Docked) == 1;
    m.flag = xc3_peek<uint8_t>(base + kXc3Upscale) != 0;
    m.view = xc3_peek<uint8_t>(e + kXc3EUpscale) != 0;
    m.max_x = xc3_peek<float>(d + kXc3DMaxX), m.max_y = xc3_peek<float>(d + kXc3DMaxY);
    m.min_x = xc3_peek<float>(d + kXc3DMinX), m.min_y = xc3_peek<float>(d + kXc3DMinY);
    m.ratio = xc3_peek<float>(e + (m.docked ? kXc3ERatioDocked : kXc3ERatioHandheld));
    m.mul_x = xc3_peek<float>(e + kXc3EMulX), m.mul_y = xc3_peek<float>(e + kXc3EMulY);
    // El suelo extra: si el mínimo de este D es el que escribimos, el del juego es el guardado (para los tamaños de cada
    // nivel y para la comprobación de abajo). Otro D (el juego rehízo el objeto): el viejo ya no es suyo ni nuestro, no
    // se toca. Lo de este D distinto de lo escrito: lo reescribió el juego, y lo suyo es lo nuevo.
    if (x.extra_d && x.extra_d != d) {
        if (nvn::first_times(x.logged_extra, 40))
            NVN_LOG("resolución (Xenoblade 3): suelo extra: el juego rehízo su objeto; el nuevo, con su mínimo\n");
        x.extra_d = 0, x.extra_scale = 0;
    } else if (x.extra_d) {
        if (m.min_x == x.extra_written && m.min_y == x.extra_written) {
            m.min_x = x.game_min_x, m.min_y = x.game_min_y;
        } else {
            if (nvn::first_times(x.logged_extra, 40))
                NVN_LOG("resolución (Xenoblade 3): suelo extra: el juego reescribió su mínimo (%.3f/%.3f, escrito %.3f)\n",
                        m.min_x, m.min_y, x.extra_written);
            // Lo que aún sea nuestro, de vuelta al del juego (sin dejar un eje a medias).
            if (m.min_x == x.extra_written) xc3_poke<float>(d + kXc3DMinX, m.min_x = x.game_min_x);
            if (m.min_y == x.extra_written) xc3_poke<float>(d + kXc3DMinY, m.min_y = x.game_min_y);
            x.extra_d = 0, x.extra_scale = 0;
        }
    }
    // Devuelve el mínimo del juego si el de este D es nuestro (antes de soltar la retención: con el nivel en 5 y su
    // mínimo, el juego queda donde lo dejaría él).
    auto extra_release = [&](const char* why) {
        if (!x.extra_d) return;
        if (xc3_peek<float>(d + kXc3DMinX) == x.extra_written) xc3_poke<float>(d + kXc3DMinX, x.game_min_x);
        if (xc3_peek<float>(d + kXc3DMinY) == x.extra_written) xc3_poke<float>(d + kXc3DMinY, x.game_min_y);
        if (nvn::first_times(x.logged_extra, 40))
            NVN_LOG("resolución (Xenoblade 3): suelo extra: devuelto el mínimo del juego %.3f/%.3f (%s)\n", x.game_min_x,
                    x.game_min_y, why);
        x.extra_d = 0, x.extra_scale = 0;
        x.extra_restores++;
    };
    const int game_level = xc3_peek<int32_t>(d + kXc3DLevel);
    const uint8_t enabled = xc3_peek<uint8_t>(d + kXc3DEnabled), hold = xc3_peek<uint8_t>(d + kXc3DHold);
    // Antes de escribir nada, que lo que hay tenga la forma de lo conocido (los límites de lib_nx.bin son 1,0 y 0,70; el
    // juego usa los niveles -3..5, y 6 por un despiste suyo). Si no, ni una escritura: no es el objeto que creemos.
    auto unit = [](float v) { return v > 0.3f && v <= 1.0f; };
    if (!unit(m.max_x) || !unit(m.max_y) || !unit(m.min_x) || !unit(m.min_y) || m.min_x > m.max_x || m.min_y > m.max_y ||
        !unit(m.ratio) || !unit(m.mul_x) || !unit(m.mul_y) || game_level < -3 || game_level > 6 || enabled > 1 || hold > 1) {
        if (nvn::first_times(x.logged_odd, 3))
            NVN_LOG("resolución (Xenoblade 3): lo que hay en el objeto no cuadra (límites %.3f/%.3f..%.3f/%.3f, proporción "
                    "%.3f, nivel %d, activada %u, retención %u): sin tocarlo\n",
                    m.max_x, m.max_y, m.min_x, m.min_y, m.ratio, game_level, enabled, hold);
        x.state = Xc3State::kOdd;
        return;
    }
    {  // lo que el juego tiene aplicado ahora mismo, para el ritmo: es la prueba de que el nivel escrito llega a la escena
        const float fx = xc3_peek<float>(e + kXc3EScaleX), fy = xc3_peek<float>(e + kXc3EScaleY);
        const float ratio = xc3_peek<float>(e + kXc3ERatioNow);
        x.live = unit(fx) && unit(fy) && unit(ratio) ? xc3_sizes(m, fx, fy, ratio) : Xc3Size{};
    }
    x.shown_level = game_level;
    if (NVN_FLAG("sin_resolucion_propia")) {
        extra_release("sin_resolucion_propia");
        if (hold) xc3_poke<uint8_t>(d + kXc3DHold, 0);  // una vez: después se lee 0 y ya no se escribe nada
        note_state(Xc3State::kGame, "sin_resolucion_propia: retención quitada, el juego vuelve a su regla");
        return;
    }
    if (!enabled || !xc3_peek<uint8_t>(e + kXc3EAllowed)) {
        extra_release("el juego tiene la escala fija");
        note_state(Xc3State::kPaused, "el juego tiene su resolución dinámica apagada (escala fija): en pausa, sin escribir");
        return;
    }
    static Xc3Setting fixed_setting("resolucion_fija", true), floor_setting("resolucion_suelo", false);
    const int sharpest = m.sharpest();
    const bool fixed = NVN_FLAG("resolucion_fija");
    int worst = kXc3Worst;
    bool user_floor = false;
    if (!fixed && NVN_FLAG("resolucion_suelo")) {
        const Xc3Request& q = floor_setting.get();
        if (q.valid) worst = xc3_level_for(m, q), user_floor = true;
    }
    // El suelo extra: escalones por debajo del 5 solo en modo TV con el reescalado temporal, con el regulador propio y
    // sin un suelo pedido.
    Xc3Extra extra;
    const char* no_extra = "vuelta al nivel 5";  // por qué no hay escalón, para el log de la devolución
    if (NVN_FLAG("sin_suelo_extra")) {
        no_extra = "sin_suelo_extra";
    } else if (!(m.docked && m.flag && m.view)) {
        no_extra = "fuera del modo TV con reescalado temporal";
    } else if (fixed) {
        no_extra = "resolucion_fija";
    } else if (user_floor) {
        no_extra = "resolucion_suelo";
    } else {
        static Xc3Setting lowest_setting("suelo_extra_minimo", false);
        double lowest = 0.55;
        if (NVN_FLAG("suelo_extra_minimo")) {
            const Xc3Request& q = lowest_setting.get();
            if (q.valid && !q.is_level && q.value >= 0.40 && q.value < 0.70)
                lowest = q.value;
            else if (q.valid && nvn::first_times(x.logged_extra_floor, 5))
                NVN_LOG("resolución (Xenoblade 3): suelo_extra_minimo pide una escala de 0,40 a 0,69: se queda en 0,55\n");
        }
        extra = xc3_extra_steps(std::min(m.min_x, m.min_y), lowest);
        worst = kXc3Worst + extra.count;
    }
    Xc3Regulator& reg = x.reg;
    int64_t interval = xc3_peek<int32_t>(base + kXc3Interval);
    if (interval <= 0 || interval > 4) interval = 2;
    const double budget = static_cast<double>(interval) * 16.6667e6;
    const int before = was_active ? reg.level : game_level;
    const char* why = "";
    if (!was_active) {  // al empezar, o al volver de una pausa o de la regla del juego: prudente y a medir de nuevo
        reg.level = std::clamp(std::max(game_level, kXc3Start), sharpest, worst);
        reg.restart();
        why = "se toma el nivel";
    } else if (m.key() != x.mode) {
        reg.level = std::clamp(reg.level, sharpest, worst);
        reg.restart();
        why = "cambio de modo";
    } else if (wrote_d == d && game_level != x.written && nvn::first_times(x.logged_moved, 5)) {
        // No debería pasar: con la retención puesta el juego no toca el nivel. Si sale, la retención no es lo que creemos.
        NVN_LOG("resolución (Xenoblade 3): el juego movió el nivel con la retención puesta (escrito %d, leído %d)\n", x.written,
                game_level);
    }
    x.mode = m.key();
    if (fixed) {
        reg.level = xc3_level_for(m, fixed_setting.get());
        reg.restart();
        if (!*why) why = "resolucion_fija";
    } else {
        // La GPU ocupada del último frame medido. Sin esa medida (sin_gpu_ocupada, o frames sin todas sus marcas) durante
        // 2 s, el tramo entre los sellos hace de las dos cosas: mide de más, así que como mucho se queda corto de nitidez.
        int64_t busy = 0, span = 0;
        bool fresh = nvn::game_gpu_busy(x.busy_serial, busy, span);
        if (fresh) {
            x.unmeasured = 0;
        } else if (x.unmeasured < 60) {
            x.unmeasured++;
        } else if (nvn::game_gpu_frame(x.gpu_serial, span) && span > 0) {
            busy = span;
            fresh = true;
            if (nvn::first_times(x.logged_span, 1))
                NVN_LOG("resolución (Xenoblade 3): sin medida de GPU ocupada: el regulador usa el tramo entre los sellos\n");
        }
        if (fresh && busy > 0) {
            const int level = std::clamp(reg.level, sharpest, worst);
            double up_pixels = 1.0;
            if (level > sharpest) {
                const Xc3Size now = xc3_size_extra(m, extra, level), up = xc3_size_extra(m, extra, level - 1);
                up_pixels = static_cast<double>(up.scene_w) * up.scene_h / std::max(now.scene_w * now.scene_h, 1);
            }
            const Xc3Sample sample{static_cast<double>(busy), static_cast<double>(std::max(span, busy)),
                                   static_cast<double>(nvn::last_frame_run_ns())};
            reg.reason = "el suelo";  // si step no decide nada y aun así cambia, es que lo recortó el rango
            // sin_regulador_nuevo: el de antes entero (sin memoria por nivel y con los 18 de 90 contando todos los tramos).
            const bool old_policy = NVN_FLAG("sin_regulador_nuevo");
            reg.step(sample, budget, sharpest, worst, up_pixels, !old_policy && !NVN_FLAG("sin_memoria_resolucion"),
                     !old_policy && !NVN_FLAG("sin_tardanza_gpu"), kXc3Worst);
            if (!*why) why = reg.reason;
        } else {
            reg.level = std::clamp(reg.level, sharpest, worst);
            if (!*why) why = "el suelo";
        }
    }
    // Pase lo que pase arriba, dentro de lo que el juego admite (y de los escalones del suelo extra de este frame).
    const int level = std::clamp(reg.level, sharpest, kXc3Worst + extra.count);
    reg.level = level;
    if (level != before) {
        x.changes++;
        if (nvn::first_times(x.logged, 120)) {
            const Xc3Size s = xc3_size_extra(m, extra, level);
            char veto[192] = "";
            if (reg.marked)
                snprintf(veto, sizeof veto,
                         "; la subida al %d no aguantó: vetado %d s, o antes con la ocupada del %d por debajo de %.1f ms "
                         "menos su coste",
                         reg.marked_level, reg.marked_wait / 30, reg.marked_level + 1, reg.marked_busy / 1e6);
            if (level > kXc3Worst) {
                const size_t used = strlen(veto);
                snprintf(veto + used, sizeof veto - used, "; suelo extra: escala %.2f (el mínimo del juego, %.2f)",
                         extra.scale[level - kXc3Worst - 1], std::min(m.min_x, m.min_y));
            }
            NVN_LOG("resolución (Xenoblade 3): nivel %d -> %d (%s), escena %dx%d (x%.2f) y salida %dx%d, %s; GPU ocupada %.1f ms, "
                    "tramo %.1f, ejecutor %.1f (presupuesto %.1f); próxima subida tras %d frames%s\n",
                    before, level, why, s.scene_w, s.scene_h, s.scene_w / 960.0, s.out_w, s.out_h, xc3_mode_name(m),
                    reg.seen_busy / 1e6, reg.seen_span / 1e6, reg.seen_run / 1e6, budget / 1e6, reg.up_wait, veto);
        }
    }
    note_state(fixed ? Xc3State::kFixed : Xc3State::kOwn,
               fixed ? "resolucion_fija: nivel fijo, sin regulador" : "regulador propio (retención del juego puesta)");
    // Las tres únicas escrituras: la retención, el nivel y el contador de histéresis del juego a 0 (para cuando se le
    // devuelva). Cada una, un store alineado en un campo que el juego solo lee mientras está retenido. En cada frame
    // aunque no cambie: si el juego rehiciera el objeto, se vuelve a poner sin más.
    xc3_poke<uint8_t>(d + kXc3DHold, 1);
    xc3_poke<int32_t>(d + kXc3DLevel, std::min(level, kXc3Worst));
    xc3_poke<int32_t>(d + kXc3DCount, 0);
    // El suelo extra: el nivel del juego se queda en 5 y baja el mínimo, que con el 5 es la escala. Lo del juego se
    // guarda antes de la primera escritura en este D (aquí m.min_* es suyo: lo nuestro se cambió arriba por lo guardado).
    if (level > kXc3Worst) {
        const float f = extra.scale[level - kXc3Worst - 1];
        if (x.extra_d != d) x.extra_d = d, x.game_min_x = m.min_x, x.game_min_y = m.min_y;
        xc3_poke<float>(d + kXc3DMinX, f);
        xc3_poke<float>(d + kXc3DMinY, f);
        x.extra_written = f, x.extra_scale = f;
        x.extra_frames++;
    } else {
        extra_release(no_extra);
    }
    // Para el aviso de "el juego movió el nivel": solo si la retención ya estaba puesta al llegar. En el frame en que se
    // pone, el regulador del juego puede ir por la mitad de su decisión (leyó la retención a 0) y guardar su nivel encima
    // del nuestro una vez: es inocuo (un nivel suyo, y se reescribe en el frame siguiente) y no dice nada de la retención.
    x.wrote_d = hold ? d : 0;
    x.written = std::min(level, kXc3Worst);  // lo escrito en D+0x40 (el suelo extra es el 5 del juego)
    x.shown_level = level;
    x.active = true;
}

// La resolución para la línea de ritmo: en Xenoblade 3, el nivel, la escena que el juego tiene aplicada (y su factor de
// 960x544) y quién manda; en MH Rise (o sin juego conocido), su factor.
void resolution_text(char* out, size_t size) {
    Xc3Resolution& x = g_xc3;
    if (x.state == Xc3State::kUnknown) {
        snprintf(out, size, "x%.3f (%llu cambios)", kRates[g_dynres.level],
                 static_cast<unsigned long long>(std::exchange(g_dynres.changes, 0)));
        return;
    }
    static const char* const kNames[] = {"", "esperando al renderer", "regulador propio", "fija", "regla del juego",
                                         "apagada por el juego", "objeto desconocido"};
    // Y del regulador: frames con la subida parada por un veto y frames en que los 18 de 90 de antes habrían bajado con
    // tramos que no llegan sin la GPU justa (la prueba de que la memoria y la cuenta nueva actúan).
    // Y del suelo extra: la escala del escalón en vigor (0: ninguno), los frames en él y las devoluciones del mínimo.
    snprintf(out, size, "nivel %d x%.2f, escena %dx%d, %s (%llu cambios; subida vetada %llu frames, tarde sin GPU justa "
             "%llu; suelo extra %.2f, %llu frames, %llu devoluciones)",
             x.shown_level, x.live.scene_w / 960.0, x.live.scene_w, x.live.scene_h, kNames[static_cast<int>(x.state)],
             static_cast<unsigned long long>(std::exchange(x.changes, 0)),
             static_cast<unsigned long long>(std::exchange(x.reg.vetoed, 0)),
             static_cast<unsigned long long>(std::exchange(x.reg.cpu_late, 0)), x.extra_scale,
             static_cast<unsigned long long>(std::exchange(x.extra_frames, 0)),
             static_cast<unsigned long long>(std::exchange(x.extra_restores, 0)));
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
    // GPU ocupada entre los dos sellos del frame del juego (lo que de "GPU del juego" es trabajo y no espera al ejecutor)
    // y, con sello_ocupado, lo que se le dijo al juego.
    const uint64_t busy_sum = take(d.busy_sum_ns), busy_max = take(d.busy_max_ns), busy_frames = take(d.busy_frames);
    const uint64_t told_sum = take(d.told_sum_ns), told_frames = take(d.told_frames);
    char busy_text[112] = "";
    if (busy_frames) {
        const int n = snprintf(busy_text, sizeof busy_text, ", ocupada %.1f (peor %.1f)", busy_sum / 1e6 / busy_frames,
                               busy_max / 1e6);
        if (told_frames)
            snprintf(busy_text + n, sizeof busy_text - n, ", al juego %.1f en %llu frames", told_sum / 1e6 / told_frames,
                     static_cast<unsigned long long>(told_frames));
    }
    const uint64_t run = take(d.run_ns), run_max = take(d.run_max_ns), wait = take(d.wait_ns);
    const uint64_t live = take(d.pipelines_live), live_ns = take(d.pipelines_live_ns);
    const uint64_t async = take(d.pipelines_async), skipped = take(d.draws_skipped);
    const uint64_t known = take(d.pipelines_known), known_ns = take(d.pipelines_known_ns);
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
        char resolution[256];
        resolution_text(resolution, sizeof resolution);
        NVN_LOG("ritmo: %.1f fps, %llu frames de más de 40 ms (el peor %.0f ms), intervalo %d, resolución %s; "
                "GPU del juego %.1f ms (peor "
                "%.1f, hueco hasta el siguiente %.1f)%s; ejecutor %.1f ms/frame (peor %.1f, %.1f esperando a la GPU); "
                "pipelines en NvnSubmit %llu (%.0f ms), "
                "en segundo plano %llu (%llu draws esperándolas), ya apuntadas sin esperar %llu (%.0f ms); "
                "traducciones en NvnSubmit %llu (%.0f ms), precargadas %llu (%.0f ms)\n",
                frames / secs, static_cast<unsigned long long>(slow), worst_ns / 1e6, w.present_interval, resolution,
                gpu_frames ? gpu_sum / 1e6 / gpu_frames : 0.0, gpu_max / 1e6, gap_frames ? gap_sum / 1e6 / gap_frames : 0.0,
                busy_text,
                run / 1e6 / std::max<uint64_t>(frames, 1),
                run_max / 1e6, wait / 1e6 / std::max<uint64_t>(frames, 1), static_cast<unsigned long long>(live), live_ns / 1e6,
                static_cast<unsigned long long>(async), static_cast<unsigned long long>(skipped),
                static_cast<unsigned long long>(known), known_ns / 1e6, static_cast<unsigned long long>(translated), translated_ns / 1e6, static_cast<unsigned long long>(prefetched),
                prefetch_ns / 1e6);
        if (slowest_ns) NVN_LOG("ritmo: el envío más lento, %.0f ms: %s\n", slowest_ns / 1e6, slowest.c_str());
        fflush(stdout);
    }
    {  // Reproducción en diferido (commands.h): solo si en estos 10 s el juego soltó algo con la cola por detrás
        uint64_t calls, waits, wait_ns, max_ns, total, pending, unsent, limbo, copied, after[5];
        nvn::take_barrier_stats(calls, waits, wait_ns, max_ns);
        nvn::take_teardown_stats(total, pending, unsent, limbo, copied, after);
        if (waits || limbo) {
            NVN_LOG("diferido: texturas finalizadas %llu, %llu al limbo (%llu con su envío sin reproducir, %llu con comandos sin "
                    "enviar; %llu KB de texels copiados; %zu siguen); frames tras su último comando 0/1/2/3/4+: "
                    "%llu/%llu/%llu/%llu/%llu; barreras %llu, %llu esperaron (%.1f ms en total, la peor %.1f)\n",
                    static_cast<unsigned long long>(total), static_cast<unsigned long long>(limbo),
                    static_cast<unsigned long long>(pending), static_cast<unsigned long long>(unsent),
                    static_cast<unsigned long long>(copied >> 10), nvn::limbo_textures(),
                    static_cast<unsigned long long>(after[0]), static_cast<unsigned long long>(after[1]),
                    static_cast<unsigned long long>(after[2]), static_cast<unsigned long long>(after[3]),
                    static_cast<unsigned long long>(after[4]), static_cast<unsigned long long>(calls),
                    static_cast<unsigned long long>(waits), wait_ns / 1e6, max_ns / 1e6);
            fflush(stdout);
        }
        // Lo que no debería pasar o conviene vigilar: cuánto llegó a guardar el limbo (sus válvulas saltan a los 256 MB o
        // a los 16 frames), lo que las válvulas soltaron, texturas finalizadas recién creadas sin pasar por el limbo
        // (<datos>/limbo_recientes las conserva), subidas cuyo buffer cambió antes de reproducirse y lo que se llegó a
        // acumular en la cola. Y los objetos de las texturas ya finalizadas, que no se borran nunca (unos 0,7 KB cada uno:
        // AmplifyTexture crea y suelta una por página): cada vez que pasan otros 32768 (unos 23 MB). Solo si hay algo que contar.
        uint64_t limbo_max, limbo_bytes, forced_age, forced_budget, young, kept, queue_max, stale;
        nvn::take_limbo_stats(limbo_max, limbo_bytes, forced_age, forced_budget, young, kept);
        nvn::take_queue_stats(queue_max, stale);
        static uint64_t kept_logged = 0;
        const bool kept_step = kept >> 15 != kept_logged >> 15;
        if (forced_age || forced_budget || young || stale || limbo_bytes > (64ull << 20) || queue_max > 64 || kept_step) {
            kept_logged = kept;
            NVN_LOG("diferido (vigilar): limbo hasta %llu texturas y %.1f MB; soltadas por edad %llu y por tope %llu; "
                    "finalizadas recién creadas sin imagen %llu; subidas con el buffer cambiado %llu; cola hasta %llu envíos; "
                    "objetos de texturas finalizadas conservados %llu (~%.1f MB)\n",
                    static_cast<unsigned long long>(limbo_max), limbo_bytes / 1048576.0,
                    static_cast<unsigned long long>(forced_age), static_cast<unsigned long long>(forced_budget),
                    static_cast<unsigned long long>(young), static_cast<unsigned long long>(stale),
                    static_cast<unsigned long long>(queue_max), static_cast<unsigned long long>(kept), kept * 0.7 / 1024.0);
            fflush(stdout);
        }
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
        g_present_thread.drain();  // la que quedara pendiente en el swapchain viejo
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
    // DEPURACIÓN (<datos>/diag_syncs): en qué se va el tiempo de presentar (ritmo, fence del hueco, adquirir, grabar y enviar,
    // vkQueuePresentKHR, resto), media de 120 frames.
    static int64_t diag_ns[6], diag_drain_ns = 0;
    static int diag_frames = 0;
    auto diag_last = std::chrono::steady_clock::now();
    auto diag_mark = [&](int k) {
        const auto now = std::chrono::steady_clock::now();
        diag_ns[k] += (now - diag_last).count();
        diag_last = now;
    };
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
        set_present_interval(interval);
        dynamic_resolution(interval);
        xc_resolution();
        xc3_resolution();
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
    diag_mark(0);
    if (!w.swapchain) {  // sin ventana (en segundo plano): el juego sigue a su ritmo, sin presentar
        std::lock_guard executor_lock(executor_mutex());
        next_frame();
        return;
    }
    hud::real_frame(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    // La presentación anterior, si sigue en su hilo (PresentThread): fuera antes de tocar el swapchain.
    diag_drain_ns += g_present_thread.drain();
    // LSFG 2x encendido: antes del real, el frame intermedio (lsfg/framegen.cpp).
    uint32_t generated_id = 0;
    const int64_t generated_ns = lsfg::present_generated(w, t, &generated_id);
    if (generated_ns) g_lsfg_pacer.queued_generated(generated_id, generated_ns);
    PresentSlot& slot = g_presenter.slots[g_presenter.next];
    g_presenter.next = (g_presenter.next + 1) % kPresentSlots;

    vkWaitForFences(c->device, 1, &slot.done, VK_TRUE, UINT64_MAX);  // el de hace kPresentSlots frames
    diag_mark(1);
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
    diag_mark(2);

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
    // Unity (Disco Elysium) dibuja la composición final en otro NVNtexture sobre la misma memoria que la textura de la
    // ventana (en la Switch es la misma imagen): si la de la ventana no tiene imagen propia, la que comparte su memoria
    // y su tamaño.
    if (t && !t->image && t->storage) {
        const uint8_t* storage = t->storage;
        const uint32_t width = t->desc.width, height = t->desc.height;
        if (Texture* alias = g_textures.find_if([&](const void*, const Texture& o) {
                return &o != t && o.image && o.storage == storage && o.desc.width == width && o.desc.height == height;
            })) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                NVN_LOG("present: la textura de la ventana no tiene imagen; se usa la que comparte su memoria (%p)\n",
                        static_cast<const void*>(storage));
            }
            t = alias;
        } else {
            static int logged = 0;
            if (logged++ < 2) {  // DEPURACIÓN: con qué texturas se podría confundir
                NVN_LOG("present: ventana %p storage %p cpu %p %ux%u sin imagen; candidatas con imagen:\n",
                        static_cast<const void*>(t), static_cast<const void*>(storage),
                        static_cast<const void*>(t->cpu_storage), width, height);
                g_textures.find_if([&](const void* key, const Texture& o) {
                    if (o.image && o.desc.width >= 1280 && o.desc.height >= 720)
                        NVN_LOG("  %p storage %p %ux%u formato %#x serial %llu\n", key,
                                static_cast<const void*>(o.storage), o.desc.width, o.desc.height, o.desc.format,
                                static_cast<unsigned long long>(o.serial));
                    return false;
                });
            }
        }
    } else if (t && !t->image) {
        static int logged = 0;
        if (logged++ < 2)
            NVN_LOG("present: ventana %p sin imagen ni storage (cpu %p)\n", static_cast<const void*>(t),
                    static_cast<const void*>(t->cpu_storage));
    }
    const bool blitted = t && blit_to_swapchain(*t, w.crop, w.images[image], w.format, w.extent, slot.cmd);
    {  // DEPURACIÓN (Disco Elysium, pantalla negra): qué textura se presenta y si llega a la pantalla, cada 300 frames
        static int presents = 0;
        if (presents++ % 300 == 0)
            NVN_LOG("present %d: textura %d %s %ux%u formato %#x imagen %s, %s\n", presents, index, t ? "sí" : "no",
                    t ? t->desc.width : 0, t ? t->desc.height : 0, t ? t->desc.format : 0,
                    t && t->image ? "sí" : "no", blitted ? "copiada" : "borrado con su color");
    }
    if (!blitted) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    if (t && (NVN_FLAG("captura_destello") || NVN_FLAG("captura_barras"))) record_probe(probe, slot.cmd, *t, w.crop);
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
    diag_mark(3);

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
    if (!generated_ns && !NVN_FLAG("presentar_en_cola")) {
        g_present_thread.push(w.swapchain, image, slot.rendered, slot.done);  // sin generado delante: en su hilo
    } else {
        std::lock_guard queue_lock(vk_queue_mutex());
        vkQueuePresentKHR(c->queue, &pi);
        nnsdk::frame_presented();
    }
    diag_mark(4);
    executor_lock.lock();  // el frame del ejecutor y sus volcados
    debug_dump_frame(t);
    next_frame();
    executor_lock.unlock();
    diag_mark(5);
    if (++diag_frames == 120) {
        int64_t thread_fence = 0, thread_present = 0;
        g_present_thread.take_stats(&thread_fence, &thread_present);
        if (NVN_FLAG("diag_syncs"))
            NVN_LOG("[sync] presentar, por frame: ritmo %.2f ms, esperar a la presentación anterior %.2f, eso más el fence del "
                    "hueco %.2f, adquirir %.2f, grabar y enviar %.2f, vkQueuePresentKHR o pasarla a su hilo %.2f, volcados y "
                    "perfil %.2f; en su hilo: fence del blit %.2f, vkQueuePresentKHR %.2f\n",
                    diag_ns[0] / 120e6, diag_drain_ns / 120e6, diag_ns[1] / 120e6, diag_ns[2] / 120e6, diag_ns[3] / 120e6,
                    diag_ns[4] / 120e6, diag_ns[5] / 120e6, thread_fence / 120e6, thread_present / 120e6);
        diag_drain_ns = 0;
        memset(diag_ns, 0, sizeof diag_ns);
        diag_frames = 0;
    }

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
void nvnWindowFinalize(void* window) {
    // La cola puede tener aún una presentación de esta ventana (guarda el puntero al Window): primero, que salga. Unity
    // solo la finaliza al cerrar o al rehacer la ventana (main+0x95d094), nunca en un frame normal.
    if (nvn::g_windows.get(window)) {
        nvn::wait_idle();
        g_present_thread.drain();
    }
    nvn::g_windows.destroy(window);
}
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
