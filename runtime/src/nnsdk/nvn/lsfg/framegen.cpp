// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// LSFG 2x en el presentador: nuestra versión de frame_gen.cpp y frame_gen_pacer.cpp de Eden (la copia del frame
// a un anillo de 2 entradas, el calentamiento, el apagado por carga con reintentos 5/15/30/60 s).
//
// Por frame real N, con LSFG encendido y listo (todo en el hilo que presenta, bajo vk_queue_mutex()):
//   1. un envío propio: copia exacta de la textura de la ventana a input(N), las pasadas compartidas de N y,
//      con la historia ya llena, las de generación: G(N-1, N) en la imagen de salida de la cadena (1280x720);
//   2. si hay G: blit de la salida a una imagen del swapchain y vkQueuePresentKHR;
//   3. window.cpp presenta N como siempre (blit de la textura del juego), justo después, sin dormir.
// FIFO pone uno por vsync: G y luego N, dos por cada 33,3 ms. El ritmo del juego no cambia (una llamada a
// nvn::present por frame real), así que su lógica sigue a 30.
//
// Interruptores (flags.h): <datos>/lsfg enciende; <datos>/lsfg_sin_guarda quita la guarda de carga;
// <datos>/lsfg_flujo_100 usa el flujo óptico a resolución completa (por defecto 0.5); <datos>/lsfg_fp32 fuerza
// los shaders FP32 (se mira al compilar, una vez por sesión).

#include "nnsdk/nvn/lsfg/framegen.h"

#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/fsr.h"
#include "nnsdk/nvn/hud.h"
#include "nnsdk/nvn/lsfg/chain.h"
#include "nnsdk/nvn/lsfg/dll.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

namespace nvn::lsfg {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kSlots = 3;                        // envíos de LSFG en vuelo: uno por frame real
constexpr uint64_t kSlotWaitNs = 250'000'000;    // el de hace 3 frames sin acabar en 250 ms: este frame sin LSFG
constexpr uint64_t kAcquireNs = 100'000'000;
constexpr uint64_t kRetireWaitNs = 50'000'000;   // al apagar: espera acotada antes de soltar la cadena
constexpr uint64_t kWarmFrames = 3;              // se genera desde el 4.º frame seguido: beta lee 3 de historia
constexpr double kHitchMs = 100.0;               // un hueco así (carga, tirón): no se interpola a través de él
constexpr double kTooFastMs = 25.0;              // el juego presenta a más de ~40 fps: no cabe un frame generado
constexpr double kLateMs = 38.0;                 // un frame real tarde (33,3 ms + 14 %)
constexpr int kLateRun = 3;                      // tantos seguidos tarde: salta la guarda
constexpr double kMeanLimitMs = 34.5;            // o la media de 30 frames por debajo de 29 fps
constexpr int kMeanWindow = 30;
// Esperas cortas: con la resolución dinámica del juego haciendo sitio (executor.cpp le suma el coste de LSFG al tiempo
// de GPU), un salto suele ser una zona pesada pasajera; esperar hasta un minuto dejaba 30 fps en zonas que ya cabían.
constexpr int kBackoffS[] = {2, 3, 5, 5};
constexpr auto kStableForget = std::chrono::seconds(60);  // un minuto generando sin saltar: el reintento vuelve a 5 s

double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

std::string data_path(const char* name) {
    const char* data = getenv("NIER_DATA");
    return std::string(data ? data : ".") + "/" + name;
}

struct Slot {
    VkCommandBuffer cmd = VK_NULL_HANDLE, blit = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
    bool pending = false, timed = false;
    const Chain* chain = nullptr;  // la cadena que usa este envío (no se suelta hasta que acabe)
};

class FrameGen {
public:
    uint32_t swapchain_images(const VkSurfaceCapabilitiesKHR& caps);
    void present(Window& w, Texture* t);
    uint64_t generated() const { return total_generated_; }  // desde el arranque (las estadísticas se reinician)
    // Lo que le cuesta a la GPU un frame generado (media móvil, ns) mientras LSFG está encendido; 0 apagado.
    int64_t reserved_ns() const { return enabled_ ? std::max<int64_t>(cost_ns_.load(std::memory_order_relaxed), 6000000) : 0; }
    int64_t last_present_ns() const { return last_present_ns_; }
    uint32_t last_id() const { return last_id_; }

private:
    uint32_t last_id_ = 0;
    std::atomic<int64_t> cost_ns_{0};
    uint64_t total_generated_ = 0;
    int64_t last_present_ns_ = 0;
    void disable(const std::string& why);
    bool setup();
    void collect(int index);
    void poll_slots();
    void retire();
    void reap(bool wait);
    void start_job(VkExtent2D extent, VkFormat format, float flow);
    void poll_job();
    bool compile(std::string* why, bool* missing);
    bool guard(double dt, Clock::time_point now, bool enabled);
    void log_stats(Clock::time_point now);
    bool record_lsfg(int index, const Texture& t, uint64_t n, bool generate);
    // La zona de la textura de la ventana que va a la pantalla (nvnWindowSetCrop; ancho 0: entera). Disco Elysium pinta
    // 1280x720 en la esquina de texturas de 1920x1080: la cadena trabaja solo con esa zona (2,25 veces menos que la textura
    // entera) y el frame generado muestra lo mismo que el real (antes salía la textura entera: la imagen encogida en una
    // esquina, un frame sí y otro no). Se fija en present() antes de grabar.
    VkRect2D src_{};
    bool record_blit(Slot& s, const Window& w, uint32_t image);

    enum class State { Off, Waiting, Ready, Failed };
    State state_ = State::Off;  // Waiting: compilando o montando la cadena en el hilo de fondo
    bool enabled_ = false, missing_dll_ = false, few_images_logged_ = false;

    // Hilo de fondo: compila (la primera vez) y monta una cadena; el hilo que presenta recoge el resultado.
    struct Job {
        std::mutex m;
        bool busy = false, done = false, fatal = false, missing = false;
        Chain* chain = nullptr;  // sin unique_ptr: un resultado sin recoger no se destruye al salir del proceso
        std::string why;
        double ms = 0;
    } job_;
    Pipelines pipelines_;     // los escribe solo el hilo de fondo, antes de la primera cadena
    bool compiled_ = false;   // ídem; se lee con el trabajo ya terminado
    const char* variant_ = "";

    Chain* chain_ = nullptr;    // la que se usa
    Chain* retired_ = nullptr;  // apagada o reconstruida: se suelta cuando la GPU acaba sus envíos
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    uint64_t timestamp_mask_ = 0;
    Slot slots_[kSlots];
    int next_ = 0;
    bool set_up_ = false;

    uint64_t warm_ = 0;  // frames seguidos copiados a la cadena (0: hay que volver a calentar)
    Clock::time_point last_{}, hold_until_{}, guard_until_{}, resumed_{}, settle_until_{};
    double ema_ = 0;
    int late_run_ = 0, window_n_ = 0, backoff_ = 0;
    double window_sum_ = 0;
    bool pace_logged_ = false;

    // Contadores de los ~5 s del log.
    Clock::time_point stats_start_{};
    uint32_t real_ = 0, generated_ = 0, warming_ = 0, skip_guard_ = 0, skip_hold_ = 0, skip_pace_ = 0, skip_busy_ = 0,
             skip_image_ = 0, trips_ = 0;
    double gpu_sum_ = 0, gpu_max_ = 0;
    uint32_t gpu_n_ = 0;
    bool announced_ = false;
};

FrameGen g_framegen;

uint32_t FrameGen::swapchain_images(const VkSurfaceCapabilitiesKHR& caps) {
    uint32_t n = 3;
    const std::string dll = find_dll();
    if (!dll.empty()) {
        n = 4;
        NVN_LOG("lsfg: hay %s: swapchain de 4 imágenes (el frame generado y el real van seguidos)\n", dll.c_str());
    }
    return caps.maxImageCount ? std::min(n, caps.maxImageCount) : n;
}

// Un error de Vulkan o de la DLL: sin LSFG hasta reiniciar el juego. Lo que hubiera en vuelo se queda (no se
// sabe si la GPU lo sigue leyendo); el juego sigue con el camino de siempre.
void FrameGen::disable(const std::string& why) {
    if (state_ == State::Failed) return;
    state_ = State::Failed;
    NVN_LOG("lsfg: desactivado en esta sesión: %s\n", why.c_str());
}

bool FrameGen::setup() {
    if (set_up_) return true;
    VkContext* c = vk();
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->queue_family;
    if (VkResult r = vkCreateCommandPool(c->device, &pci, nullptr, &pool_); r != VK_SUCCESS)
        return disable(std::string("vkCreateCommandPool: ") + vk_result_name(r)), false;
    for (Slot& s : slots_) {
        VkCommandBuffer cmds[2];
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 2;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, cmds) != VK_SUCCESS ||
            vkCreateSemaphore(c->device, &si, nullptr, &s.acquired) != VK_SUCCESS ||
            vkCreateSemaphore(c->device, &si, nullptr, &s.rendered) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, nullptr, &s.done) != VK_SUCCESS)
            return disable("sin memoria para los envíos"), false;
        s.cmd = cmds[0];
        s.blit = cmds[1];
    }
    // Tiempo de GPU de LSFG: dos timestamps por envío.
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &count, families.data());
    const uint32_t bits = c->queue_family < count ? families[c->queue_family].timestampValidBits : 0;
    if (bits) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2 * kSlots;
        if (vkCreateQueryPool(c->device, &qi, nullptr, &queries_) != VK_SUCCESS) queries_ = VK_NULL_HANDLE;
        timestamp_mask_ = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    }
    if (!queries_) NVN_LOG("lsfg: sin timestamps: no se medirá su tiempo de GPU\n");
    set_up_ = true;
    return true;
}

// Un envío que ya acabó: su tiempo de GPU a las cuentas y el hueco libre.
void FrameGen::collect(int index) {
    Slot& s = slots_[index];
    if (s.timed && queries_) {
        uint64_t ts[2] = {};
        if (vkGetQueryPoolResults(vk()->device, queries_, 2 * index, 2, sizeof ts, ts, sizeof ts[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const double gpu = static_cast<double>((ts[1] - ts[0]) & timestamp_mask_) *
                               vk()->props.limits.timestampPeriod / 1e6;
            gpu_sum_ += gpu;
            gpu_max_ = std::max(gpu_max_, gpu);
            gpu_n_++;
            const int64_t ns = static_cast<int64_t>(gpu * 1e6), old = cost_ns_.load(std::memory_order_relaxed);
            cost_ns_.store(old ? old + (ns - old) / 8 : ns, std::memory_order_relaxed);  // media móvil
        }
    }
    s.pending = s.timed = false;
    s.chain = nullptr;
}

void FrameGen::poll_slots() {
    for (int i = 0; i < kSlots; i++) {
        if (!slots_[i].pending) continue;
        VkResult r = vkGetFenceStatus(vk()->device, slots_[i].done);
        if (r == VK_SUCCESS) collect(i);
        else if (r != VK_NOT_READY) return disable(std::string("vkGetFenceStatus: ") + vk_result_name(r));
    }
}

// La cadena deja de usarse; se destruye en cuanto la GPU acabe lo que la lee (espera acotada aquí y, si no
// basta, se vuelve a mirar en cada frame sin bloquear).
void FrameGen::retire() {
    if (retired_) reap(true);
    if (retired_) return;  // la anterior sigue en la GPU: esta se queda puesta (sin usarse) hasta el frame siguiente
    retired_ = chain_;
    chain_ = nullptr;
    warm_ = 0;
    reap(true);
}

void FrameGen::reap(bool wait) {
    if (!retired_) return;
    VkFence fences[kSlots];
    uint32_t n = 0;
    for (Slot& s : slots_)
        if (s.pending && s.chain == retired_) fences[n++] = s.done;
    if (n) {
        VkResult r = vkWaitForFences(vk()->device, n, fences, VK_TRUE, wait ? kRetireWaitNs : 0);
        if (r == VK_TIMEOUT) return;
        if (r != VK_SUCCESS) return disable(std::string("vkWaitForFences: ") + vk_result_name(r));
        for (int i = 0; i < kSlots; i++)
            if (slots_[i].pending && slots_[i].chain == retired_) collect(i);
    }
    delete retired_;
    retired_ = nullptr;
    NVN_LOG("lsfg: cadena liberada\n");
}

void FrameGen::start_job(VkExtent2D extent, VkFormat format, float flow) {
    {
        std::lock_guard lock(job_.m);
        if (job_.busy) return;
        job_.busy = true;
        job_.done = false;
    }
    const bool need_compile = !compiled_;
    state_ = State::Waiting;
    if (need_compile) NVN_LOG("lsfg: compilando los shaders en segundo plano (hasta entonces, sin generar)\n");
    std::thread([this, extent, format, flow, need_compile] {
        pthread_setname_np(pthread_self(), "NvnLsfg");
        const Clock::time_point t0 = Clock::now();
        std::string why;
        bool fatal = false, missing = false;
        std::unique_ptr<Chain> chain;
        if (need_compile && !compile(&why, &missing)) {
            fatal = !missing;
        } else {
            chain = std::make_unique<Chain>();
            if (!chain->create(pipelines_, extent, format, flow, &why)) chain.reset();
        }
        std::lock_guard lock(job_.m);
        job_.fatal = fatal;
        job_.missing = missing;
        job_.chain = chain.release();
        job_.why = why;
        job_.ms = ms_between(t0, Clock::now());
        job_.done = true;
    }).detach();  // vive lo que tarde en compilar; `this` es global
}

void FrameGen::poll_job() {
    std::unique_ptr<Chain> chain;
    std::string why;
    bool fatal, missing;
    double took;
    {
        std::lock_guard lock(job_.m);
        if (!job_.done) return;
        job_.done = job_.busy = false;
        chain.reset(job_.chain);
        job_.chain = nullptr;
        why = job_.why;
        fatal = job_.fatal;
        missing = job_.missing;
        took = job_.ms;
    }
    if (missing) {  // sin DLL: se reintenta al apagar y volver a encender
        NVN_LOG("lsfg: %s; se reintenta al quitar y volver a poner <datos>/lsfg\n", why.c_str());
        missing_dll_ = true;
        state_ = State::Off;
        return;
    }
    if (fatal) return disable(why);
    if (!chain) return disable("no se pudo montar la cadena: " + why);
    NVN_LOG("lsfg: cadena %ux%u (formato %d), flujo %.2f, %s: %zu imágenes, %.1f MB, %u descriptor sets; %.0f ms\n",
            chain->extent().width, chain->extent().height, chain->format(), chain->flow(), variant_, chain->images(),
            chain->bytes() / 1048576.0, chain->sets(), took);
    chain_ = chain.release();
    state_ = State::Ready;
    warm_ = 0;
}

// En el hilo de fondo: la DLL, las auditorías y los 25 pipelines. FP16 si el dispositivo tiene shaderFloat16; si
// el driver no lo acepta, FP32. Una marca en <datos> antes de crear cada juego: si el driver tira el proceso
// compilando, la vez siguiente se salta ese juego en vez de caer otra vez.
bool FrameGen::compile(std::string* why, bool* missing) {
    VkContext* c = vk();
    const std::string path = find_dll();
    if (path.empty()) {
        *missing = true;
        *why = "no encuentro Lossless.dll (ni <datos>/lsfg/Lossless.dll ni /storage/emulated/0/Download/Lossless.dll)";
        return false;
    }
    Modules fp16, fp32;
    size_t bytes = 0;
    const Clock::time_point t0 = Clock::now();
    if (DllStatus s = load_shaders(path, fp16, fp32, &bytes); s != DllStatus::Ok) {
        *why = path + ": " + dll_status_name(s);
        return false;
    }
    NVN_LOG("lsfg: %s (%zu KB): %zu shaders FP16 y %zu FP32 en %.0f ms\n", path.c_str(), bytes >> 10, fp16.size(),
            fp32.size(), ms_between(t0, Clock::now()));

    const std::string crumb16 = data_path("lsfg_compilando_fp16"), crumb32 = data_path("lsfg_compilando_fp32");
    auto build = [&](const Modules& modules, bool float16, const std::string& crumb, std::string* reason) {
        if (FILE* f = fopen(crumb.c_str(), "w")) fclose(f);
        const Clock::time_point t1 = Clock::now();
        const bool ok = pipelines_.create(modules, float16, reason);
        unlink(crumb.c_str());
        if (ok)
            NVN_LOG("lsfg: %zu pipelines %s en %.0f ms\n", pipelines_.size(), float16 ? "FP16" : "FP32",
                    ms_between(t1, Clock::now()));
        return ok;
    };
    std::string reason16;
    const bool force32 = NVN_FLAG("lsfg_fp32");
    if (fp16.empty() || !c->shader_float16 || force32) {
        NVN_LOG("lsfg: uso FP32 (%s)\n", force32 ? "lo pide <datos>/lsfg_fp32"
                                         : fp16.empty() ? "a la DLL le faltan shaders FP16"
                                                        : "el dispositivo no tiene shaderFloat16");
    } else if (access(crumb16.c_str(), F_OK) == 0) {
        NVN_LOG("lsfg: la última vez el proceso murió creando los pipelines FP16 (queda %s): uso FP32\n", crumb16.c_str());
    } else if (build(fp16, true, crumb16, &reason16)) {
        compiled_ = true;
        variant_ = "FP16";
        return true;
    } else {
        NVN_LOG("lsfg: FP16 no vale (%s): pruebo FP32\n", reason16.c_str());
    }
    if (fp32.empty()) {
        *why = "a la DLL le faltan shaders FP32";
        return false;
    }
    if (access(crumb32.c_str(), F_OK) == 0) {
        *why = "la última vez el proceso murió creando los pipelines FP32; borra " + crumb32 + " para reintentar";
        return false;
    }
    std::string reason32;
    if (!build(fp32, false, crumb32, &reason32)) {
        *why = "FP32: " + reason32;
        return false;
    }
    compiled_ = true;
    variant_ = "FP32";
    return true;
}

// La guarda de carga: si con LSFG los frames reales llegan tarde (la GPU no cabe frame real + LSFG en 33,3 ms),
// se deja de generar un rato y se reintenta, cada vez más tarde. true: se puede generar este frame.
bool FrameGen::guard(double dt, Clock::time_point now, bool enabled) {
    if (!enabled) {
        late_run_ = window_n_ = 0;
        window_sum_ = 0;
        return true;
    }
    if (now < guard_until_) return false;
    if (backoff_ && now - resumed_ > kStableForget) backoff_ = 0;
    if (!warm_) return true;  // el intervalo de este frame no incluye trabajo de LSFG
    // Tras un tirón (una carga: frames de más de 100 ms) el juego tarda unos segundos en volver a su ritmo, sin culpa de
    // LSFG: esos frames tarde no cuentan (si no, la guarda saltaba en cada carga y llegaba a pararlo un minuto).
    if (dt > kHitchMs) settle_until_ = now + std::chrono::seconds(8);
    if (now < settle_until_) {
        late_run_ = window_n_ = 0;
        window_sum_ = 0;
        return true;
    }
    late_run_ = dt > kLateMs ? late_run_ + 1 : 0;
    window_sum_ += dt;
    window_n_++;
    const double mean = window_sum_ / window_n_;
    const bool slow_mean = window_n_ >= kMeanWindow && mean > kMeanLimitMs;
    if (window_n_ >= kMeanWindow) {
        window_n_ = 0;
        window_sum_ = 0;
    }
    if (late_run_ < kLateRun && !slow_mean) return true;
    const int wait = kBackoffS[std::min<int>(backoff_, std::size(kBackoffS) - 1)];
    if (slow_mean)
        NVN_LOG("lsfg: guarda: %.1f fps reales en el último segundo: sin generar %d s\n", 1000.0 / mean, wait);
    else
        NVN_LOG("lsfg: guarda: %d frames reales seguidos tarde (el último %.1f ms): sin generar %d s\n", late_run_, dt, wait);
    guard_until_ = resumed_ = now + std::chrono::seconds(wait);
    backoff_++;
    trips_++;
    late_run_ = window_n_ = 0;
    window_sum_ = 0;
    return false;
}

void FrameGen::log_stats(Clock::time_point now) {
    real_++;
    const double secs = ms_between(stats_start_, now) / 1000.0;
    if (secs < 5.0) return;
    if (NVN_FLAG("registro")) {  // DEPURACIÓN
        if (gpu_n_)
            NVN_LOG("lsfg: %.1f fps reales, %u generados; sin generar: %u calentando, %u guarda (%u saltos), %u tirón, "
                    "%u ritmo, %u GPU ocupada, %u sin imagen; GPU %.2f ms de media (máx %.2f)\n",
                    real_ / secs, generated_, warming_, skip_guard_, trips_, skip_hold_, skip_pace_, skip_busy_, skip_image_,
                    gpu_sum_ / gpu_n_, gpu_max_);
        else
            NVN_LOG("lsfg: %.1f fps reales, %u generados; sin generar: %u calentando, %u guarda (%u saltos), %u tirón, "
                    "%u ritmo, %u GPU ocupada, %u sin imagen\n",
                    real_ / secs, generated_, warming_, skip_guard_, trips_, skip_hold_, skip_pace_, skip_busy_, skip_image_);
    }
    stats_start_ = now;
    real_ = generated_ = warming_ = skip_guard_ = skip_hold_ = skip_pace_ = skip_busy_ = skip_image_ = trips_ = 0;
    gpu_sum_ = gpu_max_ = 0;
    gpu_n_ = 0;
}

bool FrameGen::record_lsfg(int index, const Texture& t, uint64_t n, bool generate) {
    Slot& s = slots_[index];
    VkCommandBuffer cmd = s.cmd;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd, 0);
    if (VkResult r = vkBeginCommandBuffer(cmd, &bi); r != VK_SUCCESS)
        return disable(std::string("vkBeginCommandBuffer: ") + vk_result_name(r)), false;
    {  // después del frame del juego (en vuelo, misma cola) y de lo que leyeron el LSFG y los blits anteriores
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);
    }
    if (queries_) {  // el primero, cuando acaba todo lo anterior (el frame del juego): mide solo lo nuestro
        vkCmdResetQueryPool(cmd, queries_, 2 * index, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2 * index);
    }
    if (!chain_->initialized()) chain_->record_init(cmd);
    const Image& in = chain_->input(n);
    if (in.extent.width == src_.extent.width && in.extent.height == src_.extent.height) {  // la zona que se ve, tal cual
        VkImageCopy region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffset = {src_.offset.x, src_.offset.y, 0};
        region.extent = {in.extent.width, in.extent.height, 1};
        vkCmdCopyImage(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, in.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    } else {  // lsfg_media: la cadena es más pequeña que la zona que se ve
        VkImageBlit region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[0] = {src_.offset.x, src_.offset.y, 0};
        region.srcOffsets[1] = {src_.offset.x + static_cast<int32_t>(src_.extent.width),
                                src_.offset.y + static_cast<int32_t>(src_.extent.height), 1};
        region.dstOffsets[1] = {static_cast<int32_t>(in.extent.width), static_cast<int32_t>(in.extent.height), 1};
        vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, in.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_LINEAR);
    }
    {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    if (n >= 1) chain_->record_shared(cmd, n);
    if (generate) {
        chain_->record_generate(cmd, n);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // la salida, lista para el blit
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    if (queries_) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2 * index + 1);
    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS)
        return disable(std::string("vkEndCommandBuffer: ") + vk_result_name(r)), false;
    return true;
}

// La salida de la cadena (1280x720) escalada a la imagen del swapchain, igual que window.cpp hace con el real.
bool FrameGen::record_blit(Slot& s, const Window& w, uint32_t image) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(s.blit, 0);
    if (VkResult r = vkBeginCommandBuffer(s.blit, &bi); r != VK_SUCCESS)
        return disable(std::string("vkBeginCommandBuffer: ") + vk_result_name(r)), false;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = w.images[image];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(s.blit, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &b);
    const Image& out = chain_->output();
    // Con FSR, como el frame real (window.cpp): si no, el generado salía más blando y la nitidez parpadeaba a 30 Hz.
    if (!fsr::record(s.blit, out.view, out.extent, out.extent, w.images[image], w.format, w.extent)) {
        VkImageBlit region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[1] = {static_cast<int32_t>(out.extent.width), static_cast<int32_t>(out.extent.height), 1};
        region.dstOffsets[1] = {static_cast<int32_t>(w.extent.width), static_cast<int32_t>(w.extent.height), 1};
        vkCmdBlitImage(s.blit, out.image, VK_IMAGE_LAYOUT_GENERAL, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                       &region, VK_FILTER_LINEAR);
    }
    hud::record(s.blit, w.images[image], w.extent, w.format);  // el HUD también en los generados
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(s.blit, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    if (VkResult r = vkEndCommandBuffer(s.blit); r != VK_SUCCESS)
        return disable(std::string("vkEndCommandBuffer: ") + vk_result_name(r)), false;
    return true;
}

void FrameGen::present(Window& w, Texture* t) {
    const bool on = NVN_FLAG("lsfg");
    const Clock::time_point now = Clock::now();
    const double dt = last_ == Clock::time_point{} ? 0.0 : ms_between(last_, now);
    last_ = now;
    if (state_ == State::Failed || (!on && !enabled_ && !chain_ && !retired_)) return;  // apagado: casi gratis
    VkContext* c = vk();
    if (!c || !w.swapchain) return;
    // Primero el del ejecutor (lee t->image, que el juego puede rehacer) y luego el de la cola: el mismo orden que
    // el resto (ejecutor -> cola), sin cruces.
    std::lock_guard executor_lock(executor_mutex());
    std::lock_guard lock(vk_queue_mutex());
    poll_slots();
    poll_job();
    reap(false);
    if (state_ == State::Failed) return;

    if (!on) {
        if (enabled_) NVN_LOG("lsfg: apagado\n");
        enabled_ = missing_dll_ = announced_ = false;
        if (chain_) retire();
        return;
    }
    // Más barato (para que quepa en las zonas pesadas): <datos>/lsfg_flujo_25 (flujo óptico a un cuarto) y
    // <datos>/lsfg_media (el frame generado a 3/4 de resolución, 960x540; el real sigue a la suya).
    // Flujo a un cuarto por defecto: medido en las Ruinas, 3,5 ms por frame generado frente a 6,5 a la mitad, y así cabe
    // con el frame real en zonas abiertas. lsfg_flujo_50 / lsfg_flujo_100 para comparar.
    const float flow = NVN_FLAG("lsfg_flujo_100") ? 1.0f : NVN_FLAG("lsfg_flujo_50") ? 0.5f : 0.25f;
    const bool reduced = NVN_FLAG("lsfg_media");
    const bool guarded = !NVN_FLAG("lsfg_sin_guarda");
    if (!enabled_) {
        enabled_ = true;
        stats_start_ = now;
        NVN_LOG("lsfg: encendido: 2x, modo rendimiento, flujo %.2f, SDR, ui_threshold 0.5, guarda %s\n", flow,
                guarded ? "sí" : "no");
    }
    log_stats(now);
    if (missing_dll_ || !setup()) return;

    // La cadena es para la textura de la ventana de ahora y el flujo pedido; si cambia, se monta otra en el hilo
    // de fondo (la primera vez, con los pipelines) y hasta entonces se presenta como siempre.
    if (!t || !t->image || t->vk_format == VK_FORMAT_UNDEFINED || t->desc.width <= 0 || t->desc.height <= 0) {
        warm_ = 0;
        return;
    }
    {  // la zona que se ve: el recorte de la ventana si lo hay y cabe en la textura; si no, la textura entera
        const VkRect2D& c = w.crop;
        const bool cropped = c.extent.width && c.extent.height && c.offset.x >= 0 && c.offset.y >= 0 &&
                             c.offset.x + c.extent.width <= static_cast<uint32_t>(t->desc.width) &&
                             c.offset.y + c.extent.height <= static_cast<uint32_t>(t->desc.height) &&
                             !NVN_FLAG("lsfg_textura_entera");
        src_ = cropped ? c : VkRect2D{{0, 0}, {static_cast<uint32_t>(t->desc.width), static_cast<uint32_t>(t->desc.height)}};
    }
    const VkExtent2D extent{src_.extent.width * (reduced ? 3 : 4) / 4, src_.extent.height * (reduced ? 3 : 4) / 4};
    if (!chain_ || chain_->extent().width != extent.width || chain_->extent().height != extent.height ||
        chain_->format() != t->vk_format || chain_->flow() != flow) {
        if (chain_) retire();
        if (!chain_ && !retired_) start_job(extent, t->vk_format, flow);
        warm_ = 0;
        return;
    }
    if (w.images.size() < 4) {  // la DLL apareció después de crear el swapchain: el segundo present podría esperar
        if (!few_images_logged_)
            NVN_LOG("lsfg: el swapchain tiene %zu imágenes (no había Lossless.dll al arrancar): reinicia el juego\n",
                    w.images.size());
        few_images_logged_ = true;
        return;
    }

    // ¿Toca? Solo con el juego a ~30 fps (intervalo 2), sin tirones recientes y si la guarda lo deja.
    if (dt > 0) ema_ = ema_ > 0 ? ema_ + (dt - ema_) * 0.1 : dt;
    const bool pace_ok = w.present_interval == 2 && ema_ >= kTooFastMs;
    if (pace_ok == pace_logged_) {
        pace_logged_ = !pace_ok;
        if (!pace_ok)
            NVN_LOG("lsfg: el juego no va a 30 fps (intervalo %d, %.1f ms por frame): sin generar\n", w.present_interval, ema_);
    }
    if (!pace_ok) {
        skip_pace_++;
        warm_ = 0;
        return;
    }
    if (dt > kHitchMs) hold_until_ = now + std::chrono::seconds(1);
    if (now < hold_until_) {
        skip_hold_++;
        warm_ = 0;
        return;
    }
    if (!guard(dt, now, guarded)) {
        skip_guard_++;
        warm_ = 0;
        return;
    }

    const int index = next_;
    Slot& s = slots_[index];
    if (s.pending) {  // el envío de hace 3 frames reales
        VkResult r = vkWaitForFences(c->device, 1, &s.done, VK_TRUE, kSlotWaitNs);
        if (r == VK_TIMEOUT) {
            skip_busy_++;
            warm_ = 0;
            return;
        }
        if (r != VK_SUCCESS) return disable(std::string("vkWaitForFences: ") + vk_result_name(r));
        collect(index);
    }
    const uint64_t n = warm_;
    bool generate = false;
    uint32_t image = 0;
    if (n >= kWarmFrames) {
        VkResult r = vkAcquireNextImageKHR(c->device, w.swapchain, kAcquireNs, s.acquired, VK_NULL_HANDLE, &image);
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) generate = true;
        else if (r == VK_TIMEOUT || r == VK_NOT_READY || r == VK_ERROR_OUT_OF_DATE_KHR) skip_image_++;
        else return disable(std::string("vkAcquireNextImageKHR: ") + vk_result_name(r));
    } else {
        warming_++;
    }
    // Sin imagen se graba igual la copia y lo compartido: la historia sigue llena para el frame siguiente.
    if (!record_lsfg(index, *t, n, generate) || (generate && !record_blit(s, w, image))) return;

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si[2] = {{VK_STRUCTURE_TYPE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SUBMIT_INFO}};
    si[0].commandBufferCount = 1;
    si[0].pCommandBuffers = &s.cmd;
    si[1].waitSemaphoreCount = 1;  // solo el blit espera a la imagen; la copia y el compute no
    si[1].pWaitSemaphores = &s.acquired;
    si[1].pWaitDstStageMask = &wait_stage;
    si[1].commandBufferCount = 1;
    si[1].pCommandBuffers = &s.blit;
    si[1].signalSemaphoreCount = 1;
    si[1].pSignalSemaphores = &s.rendered;
    vkResetFences(c->device, 1, &s.done);
    if (VkResult r = vkQueueSubmit(c->queue, generate ? 2 : 1, si, s.done); r != VK_SUCCESS)
        return disable(std::string("vkQueueSubmit: ") + vk_result_name(r));
    s.pending = true;
    s.timed = queries_ != VK_NULL_HANDLE;
    s.chain = chain_;
    next_ = (next_ + 1) % kSlots;
    warm_++;
    if (!generate) return;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &s.rendered;
    pi.swapchainCount = 1;
    pi.pSwapchains = &w.swapchain;
    pi.pImageIndices = &image;
    // presentID impar (2k+1): window.cpp pregunta luego cuándo salió en pantalla (VK_GOOGLE_display_timing).
    VkPresentTimeGOOGLE when{static_cast<uint32_t>(2 * total_generated_ + 1), 0};
    VkPresentTimesInfoGOOGLE times{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
    if (c->display_timing) {
        times.swapchainCount = 1;
        times.pTimes = &when;
        pi.pNext = &times;
    }
    const int64_t queued_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    VkResult r = vkQueuePresentKHR(c->queue, &pi);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR)
        return disable(std::string("vkQueuePresentKHR: ") + vk_result_name(r));
    generated_++;
    hud::generated_frame();
    last_id_ = when.presentID;
    total_generated_++;
    last_present_ns_ = queued_ns;
    if (!announced_) NVN_LOG("lsfg: generando: un frame intermedio entre cada par de frames reales\n");
    announced_ = true;
}

}  // namespace

uint32_t swapchain_images(const VkSurfaceCapabilitiesKHR& caps) { return g_framegen.swapchain_images(caps); }

int64_t reserved_ns() { return g_framegen.reserved_ns(); }

int64_t present_generated(Window& w, Texture* t, uint32_t* id) {
    const uint64_t before = g_framegen.generated();
    g_framegen.present(w, t);
    if (g_framegen.generated() == before) return 0;
    if (id) *id = g_framegen.last_id();
    return g_framegen.last_present_ns();
}

}  // namespace nvn::lsfg
