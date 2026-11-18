// HUD de rendimiento (ver hud.h). Todo en el hilo que presenta: sin locks.

#include "nnsdk/nvn/hud.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <dlfcn.h>
#include <jni.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/system_properties.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nnsdk/host.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"

namespace nvn::hud {
namespace {

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

constexpr uint32_t kW = 44, kH = 26;       // en píxeles del swapchain (1920x1080): solo el número, que no estorbe
constexpr int32_t kX = 8, kY = 8;          // esquina superior izquierda
constexpr uint32_t kRing = 12;             // buffers: más que presentaciones en vuelo (3 reales + 3 de LSFG)
// El panel de ajustes (ver hud.h): un cajón que sale del borde derecho, de un tercio del ancho de la pantalla y toda su
// altura, con una fila por ajuste.
constexpr int64_t kPanelSlideNs = 180'000'000;    // lo que tarda en salir o esconderse
constexpr int64_t kPanelIdleNs = 8'000'000'000;   // sin tocarlo, se esconde solo
constexpr int kPanelHeader = 150, kPanelRow = 132;  // alto de la cabecera y de cada fila, en píxeles

// Fuente de 5x7: cada fila, 5 bits (el alto a la izquierda).
struct Glyph {
    char c;
    uint8_t rows[7];
};
constexpr Glyph kFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}}, {'%', {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}}, {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}}, {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}}, {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}}, {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}}, {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}}, {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}}, {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}}, {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}}, {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}}, {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}}, {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}}, {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}}, {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
};

struct Rgba {
    uint8_t r, g, b, a;
};
constexpr Rgba kBack{12, 12, 14, 255};
constexpr Rgba kGood{80, 210, 110, 255}, kWarn{235, 195, 60, 255}, kBad{235, 80, 70, 255};
constexpr Rgba kEdge{58, 58, 66, 255}, kText{232, 232, 238, 255}, kOff{74, 74, 84, 255}, kKnob{240, 240, 244, 255};
constexpr Rgba kDim{150, 150, 160, 255}, kLine{34, 34, 40, 255};

// Los textos del panel, en español si el sistema está en español y en inglés si no.
bool spanish() {
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = "";
    if (__system_property_get("persist.sys.locale", value) <= 0) __system_property_get("ro.product.locale", value);
    return value[0] == 'e' && value[1] == 's';
#else
    return true;
#endif
}

// El último toque de la pantalla, como lo deja el loader en NnsdkInput (host.h).
struct Touch {
    uint32_t serial = 0, state = 0;  // state: 1 posado, 0 levantado, 2 cancelado
    int sx = 0, sy = 0, x = 0, y = 0;  // dónde se posó y por dónde va (o dónde se levantó), en píxeles de la ventana
};
bool read_touch(Touch* t) {
    NnsdkInput* in = nnsdk::host().input;
    if (!in) return false;
    t->serial = in->touch_serial.load(std::memory_order_acquire);
    t->state = in->touch_state.load(std::memory_order_acquire);
    const uint64_t start = in->touch_start.load(std::memory_order_acquire), last = in->touch_last.load(std::memory_order_acquire);
    t->sx = static_cast<int32_t>(start & 0xffffffffu), t->sy = static_cast<int32_t>(start >> 32);
    t->x = static_cast<int32_t>(last & 0xffffffffu), t->y = static_cast<int32_t>(last >> 32);
    return t->serial == in->touch_serial.load(std::memory_order_acquire);  // si cambió a medias, al frame siguiente
}

// ---------------------------------------------------------------------------------------------------------------------
// Solo en las APKs publicadas (io.github.iamwilmar.*): "Guardar reporte" y los avisos de versión. Ninguna de las dos manda
// nada a un servidor nuestro: el reporte se queda en la carpeta del juego y quien juega decide si lo adjunta a un issue;
// el aviso solo pregunta a la API pública de GitHub (sin clave) cuál es la última release.

const std::string& package_name() {
    static const std::string name = [] {
        char buf[256] = {};
        if (FILE* f = fopen("/proc/self/cmdline", "r")) {
            fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
        }
        return std::string(buf);
    }();
    return name;
}
bool published_app() {
    static const bool published = package_name().rfind("io.github.iamwilmar.", 0) == 0;
    return published;
}
// IamWilmar/<juego>-recomp, del paquete io.github.iamwilmar.<juego>port.
std::string github_repo() {
    std::string name = package_name().substr(package_name().rfind('.') + 1);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, "port") == 0) name.resize(name.size() - 4);
    return "IamWilmar/" + name + "-recomp";
}
std::string data_dir() {
    const char* d = getenv("NIER_DATA");
    return d ? d : ".";
}
// El final de un fichero (hasta `max` bytes), leído hasta el final: los de /proc dicen tamaño 0.
std::string read_tail(const std::string& path, size_t max) {
    std::string s;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return s;
    if (fseek(f, 0, SEEK_END) == 0) {
        const long size = ftell(f);
        fseek(f, size > static_cast<long>(max) ? size - static_cast<long>(max) : 0, SEEK_SET);
    }
    char buf[64 * 1024];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) {
        s.append(buf, n);
        if (s.size() > 2 * max) s.erase(0, s.size() - max);
    }
    if (s.size() > max) s.erase(0, s.size() - max);
    fclose(f);
    return s;
}
bool write_file(const std::string& path, const void* data, size_t size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(data, 1, size, f) == size;
    return fclose(f) == 0 && ok;
}
std::string property(const char* name) {
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = "";
    __system_property_get(name, value);
    return value;
#else
    (void)name;
    return "";
#endif
}
// Los hilos de estas filas van en los núcleos pequeños (0-3) y con prioridad baja: el juego no los nota.
void background_thread(const char* name) {
    pthread_setname_np(pthread_self(), name);
#ifdef __ANDROID__
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    for (int k = 0; k < 4; k++) CPU_SET(k, &cpus);
    sched_setaffinity(0, sizeof cpus, &cpus);
    setpriority(PRIO_PROCESS, 0, 10);
#endif
}

// PNG RGB sin comprimir (bloques "stored" de deflate): sin depender de zlib, que no todos los ports enlazan.
std::vector<uint8_t> png_rgb(const uint8_t* rgb, uint32_t width, uint32_t height) {
    static const auto crc_table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    const auto crc = [&](const uint8_t* p, size_t n) {
        uint32_t c = 0xffffffffu;
        for (size_t k = 0; k < n; k++) c = crc_table[(c ^ p[k]) & 0xff] ^ (c >> 8);
        return c ^ 0xffffffffu;
    };
    const auto be32 = [](std::vector<uint8_t>& v, uint32_t x) {
        for (int k = 3; k >= 0; k--) v.push_back(static_cast<uint8_t>(x >> (8 * k)));
    };
    std::vector<uint8_t> raw;
    raw.reserve(size_t{height} * (width * 3 + 1));
    for (uint32_t y = 0; y < height; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb + size_t{y} * width * 3, rgb + size_t{y + 1} * width * 3);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (size_t at = 0; at < raw.size();) {
        const size_t n = std::min<size_t>(65535, raw.size() - at);
        z.push_back(at + n == raw.size() ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n)), z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n)), z.push_back(static_cast<uint8_t>(~n >> 8));
        for (size_t k = 0; k < n; k++) a = (a + raw[at + k]) % 65521, b = (b + a) % 65521;
        z.insert(z.end(), raw.begin() + static_cast<long>(at), raw.begin() + static_cast<long>(at + n));
        at += n;
    }
    be32(z, b << 16 | a);
    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    const auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        be32(png, static_cast<uint32_t>(data.size()));
        const size_t start = png.size();
        png.insert(png.end(), type, type + 4);
        png.insert(png.end(), data.begin(), data.end());
        be32(png, crc(png.data() + start, data.size() + 4));
    };
    std::vector<uint8_t> ihdr;
    be32(ihdr, width);
    be32(ihdr, height);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return png;
}

// "1.4.0" > "1.3.2"? (con o sin la v delante)
bool newer_version(const std::string& latest, const std::string& current) {
    const auto parts = [](const std::string& v) {
        std::vector<long> out;
        const char* p = v.c_str();
        if (*p == 'v' || *p == 'V') p++;
        while (*p) {
            char* end = nullptr;
            out.push_back(strtol(p, &end, 10));
            if (end == p) break;
            p = *end == '.' ? end + 1 : end;
            if (*end != '.') break;
        }
        return out;
    };
    const std::vector<long> l = parts(latest), c = parts(current);
    for (size_t k = 0; k < std::max(l.size(), c.size()); k++) {
        const long x = k < l.size() ? l[k] : 0, y = k < c.size() ? c[k] : 0;
        if (x != y) return x > y;
    }
    return false;
}

// El valor de texto de "clave" en un JSON (sin escapes: basta para tag_name).
std::string json_string(const std::string& json, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return "";
    p = json.find('"', json.find(':', p + needle.size()) + 1);
    const size_t end = p == std::string::npos ? p : json.find('"', p + 1);
    return end == std::string::npos ? "" : json.substr(p + 1, end - p - 1);
}

#ifdef __ANDROID__
JavaVM* java_vm() {
    static JavaVM* const vm = []() -> JavaVM* {
        using Fn = jint (*)(JavaVM**, jsize, jsize*);
        void* lib = dlopen("libnativehelper.so", RTLD_NOW);
        auto fn = lib ? reinterpret_cast<Fn>(dlsym(lib, "JNI_GetCreatedJavaVMs")) : nullptr;
        JavaVM* v = nullptr;
        jsize n = 0;
        return fn && fn(&v, 1, &n) == JNI_OK && n > 0 ? v : nullptr;
    }();
    return vm;
}
// El hilo, enganchado a la VM mientras vive esto (y desenganchado al final si lo enganchó él).
struct JavaThread {
    JNIEnv* env = nullptr;
    bool attached = false;
    JavaThread() {
        JavaVM* vm = java_vm();
        if (!vm) return;
        if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
            JavaVMAttachArgs args{JNI_VERSION_1_6, "hud", nullptr};
            if (vm->AttachCurrentThread(&env, &args) == JNI_OK) attached = true;
            else env = nullptr;
        }
        if (env && env->PushLocalFrame(64) != JNI_OK) env->ExceptionClear();
    }
    ~JavaThread() {
        if (!env) return;
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->PopLocalFrame(nullptr);
        if (attached) java_vm()->DetachCurrentThread();
    }
    bool failed() {
        if (!env->ExceptionCheck()) return false;
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    // La Application del proceso (un Context): la APK no tiene más Java propio aquí.
    jobject application() {
        jclass thread = env->FindClass("android/app/ActivityThread");
        if (failed() || !thread) return nullptr;
        jmethodID current = env->GetStaticMethodID(thread, "currentApplication", "()Landroid/app/Application;");
        if (failed() || !current) return nullptr;
        jobject app = env->CallStaticObjectMethod(thread, current);
        return failed() ? nullptr : app;
    }
    // versionName de la APK.
    std::string app_version() {
        jobject app = application();
        if (!app) return "";
        jclass context = env->FindClass("android/content/Context");
        jobject pm = env->CallObjectMethod(app, env->GetMethodID(context, "getPackageManager", "()Landroid/content/pm/PackageManager;"));
        if (failed() || !pm) return "";
        jclass pm_class = env->FindClass("android/content/pm/PackageManager");
        jobject info = env->CallObjectMethod(
            pm, env->GetMethodID(pm_class, "getPackageInfo", "(Ljava/lang/String;I)Landroid/content/pm/PackageInfo;"),
            env->NewStringUTF(package_name().c_str()), 0);
        if (failed() || !info) return "";
        jclass info_class = env->FindClass("android/content/pm/PackageInfo");
        auto name = static_cast<jstring>(env->GetObjectField(info, env->GetFieldID(info_class, "versionName", "Ljava/lang/String;")));
        if (failed() || !name) return "";
        const char* c = env->GetStringUTFChars(name, nullptr);
        std::string version = c ? c : "";
        if (c) env->ReleaseStringUTFChars(name, c);
        return version;
    }
    // GET de una URL HTTPS; devuelve el cuerpo (hasta 512 KB) y deja el código HTTP en `status` (0 si no contestó).
    std::string get(const std::string& url, int* status) {
        *status = 0;
        std::string body;
        jclass url_class = env->FindClass("java/net/URL");
        jobject u = env->NewObject(url_class, env->GetMethodID(url_class, "<init>", "(Ljava/lang/String;)V"), env->NewStringUTF(url.c_str()));
        if (failed() || !u) return body;
        jobject connection = env->CallObjectMethod(u, env->GetMethodID(url_class, "openConnection", "()Ljava/net/URLConnection;"));
        if (failed() || !connection) return body;
        jclass http = env->FindClass("java/net/HttpURLConnection");
        env->CallVoidMethod(connection, env->GetMethodID(http, "setConnectTimeout", "(I)V"), 10000);
        env->CallVoidMethod(connection, env->GetMethodID(http, "setReadTimeout", "(I)V"), 10000);
        env->CallVoidMethod(connection, env->GetMethodID(http, "setRequestProperty", "(Ljava/lang/String;Ljava/lang/String;)V"),
                            env->NewStringUTF("Accept"), env->NewStringUTF("application/vnd.github+json"));
        if (failed()) return body;
        *status = env->CallIntMethod(connection, env->GetMethodID(http, "getResponseCode", "()I"));
        if (failed()) {
            *status = 0;
            return body;
        }
        if (*status == 200) {
            jobject in = env->CallObjectMethod(connection, env->GetMethodID(http, "getInputStream", "()Ljava/io/InputStream;"));
            if (!failed() && in) {
                jclass in_class = env->FindClass("java/io/InputStream");
                jmethodID read = env->GetMethodID(in_class, "read", "([BII)I");
                jbyteArray buf = env->NewByteArray(16384);
                for (jint n; body.size() < 512 * 1024 && (n = env->CallIntMethod(in, read, buf, 0, 16384)) >= 0 && !failed();) {
                    const size_t at = body.size();
                    body.resize(at + static_cast<size_t>(n));
                    env->GetByteArrayRegion(buf, 0, n, reinterpret_cast<jbyte*>(body.data() + at));
                }
                failed();
                env->CallVoidMethod(in, env->GetMethodID(in_class, "close", "()V"));
                failed();
            }
        }
        env->CallVoidMethod(connection, env->GetMethodID(http, "disconnect", "()V"));
        failed();
        return body;
    }
    // Abre una URL en el navegador.
    void open(const std::string& url) {
        jobject app = application();
        if (!app) return;
        jclass uri = env->FindClass("android/net/Uri");
        jobject parsed = env->CallStaticObjectMethod(uri, env->GetStaticMethodID(uri, "parse", "(Ljava/lang/String;)Landroid/net/Uri;"),
                                                     env->NewStringUTF(url.c_str()));
        if (failed() || !parsed) return;
        jclass intent_class = env->FindClass("android/content/Intent");
        jobject intent = env->NewObject(intent_class, env->GetMethodID(intent_class, "<init>", "(Ljava/lang/String;Landroid/net/Uri;)V"),
                                        env->NewStringUTF("android.intent.action.VIEW"), parsed);
        if (failed() || !intent) return;
        env->CallObjectMethod(intent, env->GetMethodID(intent_class, "addFlags", "(I)Landroid/content/Intent;"), 0x10000000);  // NEW_TASK
        jclass context = env->FindClass("android/content/Context");
        env->CallVoidMethod(app, env->GetMethodID(context, "startActivity", "(Landroid/content/Intent;)V"), intent);
        failed();
    }
};
#endif

class Hud {
public:
    void real_frame(int64_t now) {
        if (last_real_ && now > last_real_) {
            intervals_.push_back((now - last_real_) / 1e6);
            if (intervals_.size() > 300) intervals_.pop_front();
        }
        last_real_ = now;
        reals_.push_back(now);
        trim(reals_, now);
    }
    void generated_frame() {
        generated_.push_back(last_real_);
        trim(generated_, last_real_);
    }
    void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
        // El contador se apaga con <datos>/sin_fps (en vivo, como siempre) o desde el panel, que pone y quita ese mismo
        // fichero: lo elegido vale también para la próxima vez que se abra el juego.
        if (const bool file = NVN_FLAG("sin_fps"); !flag_seen_ || file != flag_) flag_seen_ = true, flag_ = hidden_ = file;
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        gestures(extent, now);
        if (failed_) return;
        if (const uint32_t s = published_serial_.load(std::memory_order_acquire); s != published_seen_)
            published_seen_ = s, panel_dirty_ = true;
        if (published_app()) published_frame(now);
        const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
        if (!bgra && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R8G8B8A8_SRGB) return;
        if (capture_wanted_) capture_wanted_ = false, record_capture(cmd, image, extent, bgra);  // antes de dibujar nada encima
        if (!hidden_ && extent.width >= kX + kW && extent.height >= kY + kH && (!buffers_.empty() || create(buffers_, kW, kH))) {
            Buffer& b = buffers_[next_];
            next_ = (next_ + 1) % kRing;
            draw(reinterpret_cast<Rgba*>(b.data), bgra);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset = {kX, kY, 0};
            region.imageExtent = {kW, kH, 1};
            vkCmdCopyBufferToImage(cmd, b.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        record_notice(cmd, image, extent, bgra, now);
        record_panel(cmd, image, extent, bgra, now);
    }
    void set_swapchain_usage(uint32_t usage) { swapchain_src_ = (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0; }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* data = nullptr;
    };
    static void trim(std::deque<int64_t>& q, int64_t now) {
        while (!q.empty() && now - q.front() > 1000000000) q.pop_front();
    }
    // El panel: cuánto asoma (0 a 1) en este instante, según cuándo se abrió o se cerró.
    float panel_shown(int64_t now) const {
        const float t = std::clamp(static_cast<float>(now - panel_at_) / kPanelSlideNs, 0.0f, 1.0f);
        const float eased = 1.0f - (1.0f - t) * (1.0f - t);
        return panel_open_ ? panel_from_ + (1.0f - panel_from_) * eased : panel_from_ * (1.0f - eased);
    }
    void set_panel(bool open, int64_t now) {
        if (open == panel_open_) return;
        panel_from_ = panel_shown(now);  // desde donde esté: abrirlo a medio esconderse no da un salto
        panel_open_ = open;
        panel_at_ = now;
        touched_at_ = now;
        if (open) {  // por si los ficheros cambiaron desde fuera
            const bool marked = has_setup() && access(data_file("elegir_carpeta").c_str(), F_OK) == 0;
            if (marked != choose_folder_) choose_folder_ = marked, panel_dirty_ = true;
            const bool tv = has_tv() && access(data_file("modo_tv").c_str(), F_OK) == 0;
            if (tv != tv_) tv_ = tv, panel_dirty_ = true;
            // El frame de ahora, sin el panel encima, es el que va en "Guardar reporte" (mientras se guarda uno, no).
            if (published_app() && swapchain_src_ && report_state_ != kSaving) capture_wanted_ = true;
        }
    }
    // Las filas del panel, en orden: el contador de FPS siempre; el modo TV si el port lo tiene; cambiar de carpeta si
    // la app tiene pantalla de inicio; y en las APKs publicadas, guardar reporte, los avisos de versión y, si hay una
    // versión nueva, descargarla.
    enum RowId { kRowFps, kRowTv, kRowFolder, kRowReport, kRowUpdates, kRowDownload };
    int rows(RowId* out) const {
        int n = 0;
        out[n++] = kRowFps;
        if (has_tv()) out[n++] = kRowTv;
        if (has_setup()) out[n++] = kRowFolder;
        if (published_app()) {
            out[n++] = kRowReport;
            out[n++] = kRowUpdates;
            if (update_state_ == kNewer) out[n++] = kRowDownload;
        }
        return n;
    }
    // ¿El port tiene modo TV (la consola en la base)? El loader pone NIER_TV_MODE en los que lo traen.
    static bool has_tv() {
        static const bool tv = getenv("NIER_TV_MODE") != nullptr;
        return tv;
    }
    // <datos>/<name>.
    static std::string data_file(const char* name) {
        const char* data = getenv("NIER_DATA");
        return std::string(data ? data : ".") + "/" + name;
    }
    // ¿La app tiene pantalla de inicio (la que pide la carpeta del juego)? El loader pone NIER_SETUP cuando la carpeta sale
    // de ella. Solo entonces tiene sentido la fila "Cambiar carpeta".
    static bool has_setup() {
        static const bool setup = getenv("NIER_SETUP") != nullptr;
        return setup;
    }
    // Los gestos, una vez por frame presentado.
    // Abrir: el gesto de "atrás" del sistema desde el borde derecho (el de la flecha; Android se queda con ese toque y al
    // juego solo le llega la tecla, que el loader convierte en NnsdkInput::hud_gesture), o un dedo que se posa en la mitad
    // derecha de la pantalla y va hacia la izquierda al menos un octavo de su ancho.
    // Con el panel fuera: un toque en una fila cambia su ajuste; un toque fuera del panel, deslizar hacia la derecha, el
    // mismo gesto del borde o un rato sin tocarlo lo esconden.
    void gestures(VkExtent2D extent, int64_t now) {
        const int w = static_cast<int>(extent.width);
        const int panel_w = w / 3;
        if (NnsdkInput* in = nnsdk::host().input) {
            const uint64_t gesture = in->hud_gesture.load(std::memory_order_acquire);
            if (!gesture_seen_) gesture_seen_ = true, gesture_ = gesture;  // lo de antes de empezar no cuenta
            if (gesture != gesture_) {
                gesture_ = gesture;
                // Si ese mismo toque ya lo abrió deslizando (el sistema tardó en quedárselo), no lo cierra.
                if (!(panel_open_ && opened_by_ == static_cast<uint32_t>(gesture >> 32))) set_panel(!panel_open_, now);
            }
        }
        Touch t;
        if (!read_touch(&t)) return;
        if (!touch_seen_) touch_seen_ = true, touch_handled_ = t.serial;
        if (t.serial != touch_handled_) {
            const int dx = t.x - t.sx, dy = t.y - t.sy;
            const bool ended = t.state != 1;
            if (t.state == 2) {
                touch_handled_ = t.serial;
            } else if (!panel_open_) {
                if (t.sx > w / 2 && dx < -w / 8 && std::abs(dy) * 2 < -dx) {
                    set_panel(true, now);
                    opened_by_ = t.serial;
                    touch_handled_ = t.serial;  // al levantar este mismo dedo no pasa nada más
                } else if (ended) {
                    touch_handled_ = t.serial;
                }
            } else if (dx > w / 12 && std::abs(dy) * 2 < dx) {
                set_panel(false, now);
                touch_handled_ = t.serial;
            } else if (ended) {
                if (std::abs(dx) < 32 && std::abs(dy) < 32) {
                    if (t.x < w - panel_w) {
                        set_panel(false, now);
                    } else {
                        touched_at_ = now;
                        const int row = t.y >= kPanelHeader ? (t.y - kPanelHeader) / kPanelRow : -1;
                        RowId ids[8];
                        if (row >= 0 && row < rows(ids)) {
                            if (ids[row] == kRowFps) set_hidden(!hidden_);
                            else if (ids[row] == kRowTv) set_tv(!tv_);
                            else if (ids[row] == kRowFolder) set_choose_folder(!choose_folder_);
                            else if (ids[row] == kRowReport) start_report();
                            else if (ids[row] == kRowUpdates) set_update_notices(!update_notices_);
                            else if (ids[row] == kRowDownload) open_release_page();
                        }
                    }
                }
                touch_handled_ = t.serial;
            }
        }
        if (panel_open_ && now - touched_at_ > kPanelIdleNs) set_panel(false, now);
    }
    void set_hidden(bool hidden) {
        hidden_ = hidden;
        panel_dirty_ = true;
        const std::string file = data_file("sin_fps");
        if (!hidden) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: contador de FPS %s desde el panel\n", hidden ? "apagado" : "encendido");
    }
    // "Cambiar carpeta": con el fichero <datos>/elegir_carpeta, la pantalla de inicio de la app (que normalmente arranca el
    // juego directamente) vuelve a salir la próxima vez que se abra, para elegir otra carpeta; ella misma borra el fichero.
    void set_choose_folder(bool on) {
        choose_folder_ = on;
        panel_dirty_ = true;
        const std::string file = data_file("elegir_carpeta");
        if (!on) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: cambiar de carpeta al abrir la app: %s\n", on ? "sí" : "no");
    }
    // "Modo TV": con el fichero <datos>/modo_tv el juego cree estar en la base (más resolución, mando Pro) desde la próxima
    // vez que se abra; el modo se decide al arrancar (nnsdk::tv_mode).
    void set_tv(bool on) {
        tv_ = on;
        panel_dirty_ = true;
        const std::string file = data_file("modo_tv");
        if (!on) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: modo TV al abrir la app: %s\n", on ? "sí" : "no");
    }
    void record_panel(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, bool bgra, int64_t now) {
        const uint32_t w = extent.width / 3, h = extent.height;
        if (w < 360 || h < static_cast<uint32_t>(kPanelHeader + kPanelRow)) return;
        const uint32_t visible = static_cast<uint32_t>(panel_shown(now) * w + 0.5f);
        if (!visible) return;
        // Dos buffers del tamaño del panel: se redibuja en el otro solo cuando cambia lo que enseña, así las copias que
        // la GPU aún tenga pendientes siguen leyendo uno que nadie toca (entre dos cambios pasa al menos un toque).
        if (panel_buffers_.empty() || panel_w_ != w || panel_h_ != h) {
            if (!panel_buffers_.empty() || !create(panel_buffers_, w, h, 2)) return;  // otro tamaño: no pasa (el swapchain no cambia)
            panel_w_ = w, panel_h_ = h, panel_dirty_ = true;
        }
        if (hidden_ != panel_drawn_hidden_) panel_dirty_ = true;  // lo apagaron con el fichero, desde fuera
        if (panel_dirty_) {
            panel_dirty_ = false;
            panel_drawn_hidden_ = hidden_;
            panel_next_ ^= 1;
            draw_panel(reinterpret_cast<Rgba*>(panel_buffers_[panel_next_].data), bgra);
        }
        // Asoma por el borde derecho: las `visible` primeras columnas del panel, pegadas al borde.
        VkBufferImageCopy region{};
        region.bufferRowLength = w;
        region.bufferImageHeight = h;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {static_cast<int32_t>(extent.width - visible), 0, 0};
        region.imageExtent = {visible, h, 1};
        vkCmdCopyBufferToImage(cmd, panel_buffers_[panel_next_].buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    // --- Las filas de las APKs publicadas (ver published_app()).
    enum ReportState { kIdle, kSaving, kSaved, kFailed };
    enum UpdateState { kUnchecked, kChecking, kUpToDate, kNewer, kCheckFailed };
    void published_changed() { published_serial_.fetch_add(1, std::memory_order_release); }
    // Una vez por frame: los avisos se cargan al principio y la consulta a GitHub sale a los 8 s de juego.
    void published_frame(int64_t now) {
        if (!first_frame_) {
            first_frame_ = now;
            update_notices_ = access(data_file("sin_avisos_version").c_str(), F_OK) != 0;
        }
        if (!check_started_ && update_notices_ && now - first_frame_ > 8'000'000'000) {
            check_started_ = true;
            start_release_check();
        }
        const int state = update_state_;
        if (state != notice_state_) {
            notice_state_ = state;
            if (state == kNewer) notice_until_ = now + 10'000'000'000;
        }
    }
    void published_texts(RowId id, bool es, std::string* name, std::string* hint) const {
        std::string latest, current;
        {
            std::lock_guard<std::mutex> lock(versions_mutex_);
            latest = latest_, current = current_;
        }
        if (id == kRowReport) {
            *name = es ? "GUARDAR REPORTE" : "SAVE REPORT";
            const int state = report_state_;
            *hint = state == kSaving  ? (es ? "GUARDANDO..." : "SAVING...")
                    : state == kSaved ? (es ? "GUARDADO EN REPORTS" : "SAVED IN REPORTS")
                    : state == kFailed ? (es ? "NO SE PUDO GUARDAR" : "COULD NOT SAVE")
                                       : (es ? "LOG Y CAPTURA A LA CARPETA" : "LOG AND SCREENSHOT TO FOLDER");
        } else if (id == kRowUpdates) {
            *name = es ? "AVISOS DE VERSION" : "UPDATE NOTICES";
            const int state = update_state_;
            if (!update_notices_) *hint = es ? "APAGADOS" : "OFF";
            else if (state == kChecking) *hint = es ? "MIRANDO GITHUB..." : "CHECKING GITHUB...";
            else if (state == kUpToDate) *hint = (es ? "AL DIA: V" : "UP TO DATE: V") + current;
            else if (state == kNewer) *hint = "V" + latest + (es ? " EN GITHUB" : " ON GITHUB");
            else if (state == kCheckFailed) *hint = es ? "SIN CONEXION" : "NO CONNECTION";
            else *hint = es ? "MIRA GITHUB AL ABRIR" : "CHECKS GITHUB AT START";
        } else {
            *name = (es ? "DESCARGAR V" : "GET V") + latest;
            *hint = es ? "ABRE LA PAGINA EN GITHUB" : "OPENS THE GITHUB PAGE";
        }
    }
    // "Avisos de versión": con el fichero <datos>/sin_avisos_version, apagados (no se pregunta a GitHub).
    void set_update_notices(bool on) {
        update_notices_ = on;
        panel_dirty_ = true;
        const std::string file = data_file("sin_avisos_version");
        if (on) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: avisos de versión %s\n", on ? "encendidos" : "apagados");
        if (on && !check_started_ && first_frame_) check_started_ = true, start_release_check();
    }
    // La última release del repo del port, en la API pública de GitHub (sin clave), contra la versión de la APK.
    void start_release_check() {
        update_state_ = kChecking;
        published_changed();
#ifdef __ANDROID__
        std::thread([this] {
            background_thread("avisos");
            JavaThread java;
            int status = 0;
            const std::string current = java.env ? java.app_version() : "";
            const std::string body =
                java.env ? java.get("https://api.github.com/repos/" + github_repo() + "/releases/latest", &status) : "";
            std::string latest = json_string(body, "tag_name");
            if (!latest.empty() && (latest[0] == 'v' || latest[0] == 'V')) latest.erase(0, 1);
            {
                std::lock_guard<std::mutex> lock(versions_mutex_);
                latest_ = latest, current_ = current;
            }
            update_state_ = status != 200 || latest.empty() || current.empty() ? kCheckFailed
                            : newer_version(latest, current)                   ? kNewer
                                                                               : kUpToDate;
            NVN_LOG("hud: esta versión %s; la última de %s en GitHub, %s (HTTP %d)\n", current.c_str(), github_repo().c_str(),
                    latest.empty() ? "?" : latest.c_str(), status);
            published_changed();
        }).detach();
#else
        update_state_ = kCheckFailed;
#endif
    }
    void open_release_page() {
#ifdef __ANDROID__
        std::thread([] {
            background_thread("avisos");
            JavaThread java;
            if (java.env) java.open("https://github.com/" + github_repo() + "/releases/latest");
        }).detach();
#endif
    }
    // La captura del reporte: una copia del frame entero a un buffer que lee la CPU.
    bool create_capture(VkDeviceSize size) {
        VkContext* c = vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(c->device, &bi, nullptr, &capture_.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c->device, capture_.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX)
            ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &capture_.memory) != VK_SUCCESS ||
            vkBindBufferMemory(c->device, capture_.buffer, capture_.memory, 0) != VK_SUCCESS ||
            vkMapMemory(c->device, capture_.memory, 0, VK_WHOLE_SIZE, 0, &capture_.data) != VK_SUCCESS) {
            NVN_LOG("hud: sin buffer para la captura del reporte\n");
            if (capture_.memory) vkFreeMemory(c->device, capture_.memory, nullptr);
            vkDestroyBuffer(c->device, capture_.buffer, nullptr);
            capture_ = Buffer{};
            return false;
        }
        capture_coherent_ = c->memory.memoryTypes[ai.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        capture_size_ = size;
        return true;
    }
    void record_capture(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, bool bgra) {
        const VkDeviceSize size = VkDeviceSize{extent.width} * extent.height * 4;
        if ((capture_.buffer && capture_size_ != size) || (!capture_.buffer && !create_capture(size))) return;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_.buffer, 1, &region);
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = capture_.buffer;
        bb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                             nullptr, 1, &bb, 1, &b);
        capture_w_ = extent.width, capture_h_ = extent.height, capture_bgra_ = bgra;
        capture_at_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    // "Guardar reporte": <datos>/reports/<fecha>/ con info.txt, log.txt, log_previous.txt, frames.txt y screenshot.png
    // (el frame de cuando se abrió el panel, a media resolución). No sale de la carpeta del juego.
    void start_report() {
        if (report_state_ == kSaving) return;
        report_state_ = kSaving;
        published_changed();
        std::vector<double> frames(intervals_.begin(), intervals_.end());
        const int64_t played = first_frame_ ? last_real_ - first_frame_ : 0;
        std::thread([this, frames = std::move(frames), played] {
            background_thread("reporte");
            const time_t t = time(nullptr);
            char id[32];
            strftime(id, sizeof id, "%Y%m%d-%H%M%S", localtime(&t));
            const std::string dir = data_dir() + "/reports/" + id;
            mkdir((data_dir() + "/reports").c_str(), 0755);
            bool ok = mkdir(dir.c_str(), 0755) == 0;
            // La captura: la GPU la escribió unos frames después de pedirla.
            std::string shot = "no";
            if (const int64_t at = capture_at_; at && capture_.data) {
                for (int k = 0; k < 20; k++) {
                    const int64_t now =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (now - at > 300'000'000) break;
                    usleep(50'000);
                }
                if (!capture_coherent_) {
                    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                    r.memory = capture_.memory;
                    r.size = VK_WHOLE_SIZE;
                    vkInvalidateMappedMemoryRanges(vk()->device, 1, &r);
                }
                const uint32_t w = capture_w_ / 2, h = capture_h_ / 2, stride = capture_w_ * 4;
                const auto* src = static_cast<const uint8_t*>(capture_.data);
                std::vector<uint8_t> rgb(size_t{w} * h * 3);
                for (uint32_t y = 0; y < h; y++)
                    for (uint32_t x = 0; x < w; x++)
                        for (int ch = 0; ch < 3; ch++) {
                            const int from = capture_bgra_ ? 2 - ch : ch;
                            const uint8_t* p = src + size_t{2 * y} * stride + size_t{2 * x} * 4 + from;
                            rgb[(size_t{y} * w + x) * 3 + ch] = static_cast<uint8_t>((p[0] + p[4] + p[stride] + p[stride + 4] + 2) / 4);
                        }
                const std::vector<uint8_t> png = png_rgb(rgb.data(), w, h);
                if (write_file(dir + "/screenshot.png", png.data(), png.size())) shot = "screenshot.png (" + std::to_string(w) + "x" + std::to_string(h) + ")";
            }
            fflush(stdout);  // lo que quede en el buffer de stdio llega a log.txt
            usleep(500'000);
            const std::string log = read_tail(data_dir() + "/log.txt", 4 << 20), previous = read_tail(data_dir() + "/log_1.txt", 2 << 20);
            ok = ok && write_file(dir + "/log.txt", log.data(), log.size());
            if (!previous.empty()) write_file(dir + "/log_previous.txt", previous.data(), previous.size());
            std::string csv = "frame_ms\n";
            double sum = 0, worst = 0;
            int count = 0;
            for (size_t k = frames.size(); k-- > 0 && sum < 10'000;) sum += frames[k], worst = std::max(worst, frames[k]), count++;
            for (double ms : frames) csv += std::to_string(ms) + "\n";
            write_file(dir + "/frames.txt", csv.data(), csv.size());
            std::string version = "?";
#ifdef __ANDROID__
            {
                JavaThread java;
                if (java.env) version = java.app_version();
            }
#endif
            char when[64];
            strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S %z", localtime(&t));
            std::string info = "app: " + package_name() + " " + version + "\n" + "device: " + property("ro.product.manufacturer") + " " +
                               property("ro.product.model") + ", Android " + property("ro.build.version.release") + "\n" +
                               "time: " + when + "\n";
            char line[160];
            snprintf(line, sizeof line, "played this session: %lld:%02lld:%02lld\n", static_cast<long long>(played / 3'600'000'000'000),
                     static_cast<long long>(played / 60'000'000'000 % 60), static_cast<long long>(played / 1'000'000'000 % 60));
            info += line;
            if (count) {
                snprintf(line, sizeof line, "last %.1f s: %.1f fps average, slowest frame %.1f ms\n", sum / 1000, count * 1000 / sum, worst);
                info += line;
            }
            const std::string status = read_tail("/proc/self/status", 1 << 16), meminfo = read_tail("/proc/meminfo", 1 << 16);
            for (const auto& [text, key] : {std::pair{&status, "VmRSS"}, std::pair{&meminfo, "MemAvailable"}})
                if (const size_t p = text->find(key); p != std::string::npos) info += text->substr(p, text->find('\n', p) - p + 1);
            info += "screenshot: " + shot + "\n";
            ok = ok && write_file(dir + "/info.txt", info.data(), info.size());
            NVN_LOG("hud: reporte %s en %s\n", ok ? "guardado" : "NO guardado", dir.c_str());
            report_state_ = ok ? kSaved : kFailed;
            published_changed();
        }).detach();
    }
    // El aviso de versión nueva: una franja centrada arriba durante 10 s.
    void record_notice(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, bool bgra, int64_t now) {
        if (!notice_until_ || now > notice_until_) return;
        constexpr uint32_t nw = 1000, nh = 72;
        if (extent.width < nw + 16 || extent.height < nh + 32) return;
        if (notice_buffers_.empty() && !create(notice_buffers_, nw, nh, 2)) return;
        if (!notice_drawn_) {
            notice_drawn_ = true;
            std::string latest;
            {
                std::lock_guard<std::mutex> lock(versions_mutex_);
                latest = latest_;
            }
            const std::string t = (spanish() ? "VERSION " + latest + " EN GITHUB" : "VERSION " + latest + " ON GITHUB");
            auto* px = reinterpret_cast<Rgba*>(notice_buffers_[0].data);
            const auto put = [&](int x, int y, Rgba c) {
                if (x >= 0 && y >= 0 && x < static_cast<int>(nw) && y < static_cast<int>(nh)) px[y * nw + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
            };
            for (int y = 0; y < static_cast<int>(nh); y++)
                for (int x = 0; x < static_cast<int>(nw); x++)
                    put(x, y, x < 4 || y < 4 || x >= static_cast<int>(nw) - 4 || y >= static_cast<int>(nh) - 4 ? kGood : kBack);
            const int scale = 4;
            int x = (static_cast<int>(nw) - (static_cast<int>(t.size()) * 6 * scale - scale)) / 2;
            for (const char c : t) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == c) g = &f;
                if (g)
                    for (int r = 0; r < 7; r++)
                        for (int col = 0; col < 5; col++)
                            if (g->rows[r] & (0x10 >> col))
                                for (int dy = 0; dy < scale; dy++)
                                    for (int dx = 0; dx < scale; dx++) put(x + col * scale + dx, (nh - 7 * scale) / 2 + r * scale + dy, kText);
                x += 6 * scale;
            }
        }
        VkBufferImageCopy region{};
        region.bufferRowLength = nw;
        region.bufferImageHeight = nh;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {static_cast<int32_t>((extent.width - nw) / 2), 24, 0};
        region.imageExtent = {nw, nh, 1};
        vkCmdCopyBufferToImage(cmd, notice_buffers_[0].buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    // La cabecera y una fila por ajuste: su nombre y un interruptor (verde con el pomo a la derecha, encendido; gris con el
    // pomo a la izquierda, apagado).
    void draw_panel(Rgba* px, bool bgra) const {
        const int w = static_cast<int>(panel_w_), h = static_cast<int>(panel_h_);
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= w || y >= h) return;
            px[y * w + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int rw, int rh, Rgba c) {
            for (int y = y0; y < y0 + rh; y++)
                for (int x = x0; x < x0 + rw; x++) put(x, y, c);
        };
        const auto pill = [&](int x0, int y0, int pw, int ph, Rgba c) {  // rectángulo con los extremos redondos
            const int r = ph / 2;
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    const int cx = x < r ? r : x >= pw - r ? pw - r - 1 : x, ex = x - cx, ey = y - r;
                    if (ex * ex + ey * ey <= r * r) put(x0 + x, y0 + y, c);
                }
        };
        const auto text = [&](int x, int y, int scale, const char* s, Rgba c) {
            for (; *s; s++, x += 6 * scale) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b)) rect(x + b * scale, y + r * scale, scale, scale, c);
            }
        };
        const bool es = spanish();
        const int margin = 44;
        rect(0, 0, w, h, kBack);
        rect(0, 0, 4, h, kEdge);  // el canto que da al juego
        text(margin, (kPanelHeader - 7 * 6) / 2, 6, es ? "AJUSTES" : "SETTINGS", kText);
        rect(4, kPanelHeader - 3, w - 4, 3, kEdge);
        // Las filas de rows(): el contador de FPS, el modo TV y volver a elegir la carpeta del juego.
        struct Row {
            const char* es;
            const char* en;
            const char* hint_es;  // una línea pequeña debajo (o nullptr)
            const char* hint_en;
            bool on;
        };
        const Row all[] = {{"CONTADOR DE FPS", "FPS COUNTER", nullptr, nullptr, !hidden_},
                           {"MODO TV", "TV MODE", "AL ABRIR LA APP", "AT NEXT START", tv_},
                           {"CAMBIAR CARPETA", "CHANGE FOLDER", "AL ABRIR LA APP", "AT NEXT START", choose_folder_}};
        RowId ids[8];
        const int count = rows(ids);
        int y = kPanelHeader;
        for (int k = 0; k < count; k++) {
            if (ids[k] >= kRowReport && ids[k] <= kRowDownload) {  // las de las APKs publicadas
                std::string name, hint;
                published_texts(ids[k], es, &name, &hint);
                text(margin, y + 30, 4, name.c_str(), kText);
                text(margin, y + 30 + 7 * 4 + 14, 3, hint.c_str(), kDim);
                if (ids[k] == kRowUpdates) {  // interruptor
                    const int sw = 132, sh = 60, sx = w - sw - margin, sy = y + (kPanelRow - sh) / 2;
                    pill(sx, sy, sw, sh, update_notices_ ? kGood : kOff);
                    pill(update_notices_ ? sx + sw - sh + 5 : sx + 5, sy + 5, sh - 10, sh - 10, kKnob);
                } else {  // botón: un triángulo a la derecha
                    for (int r = 0; r < 25; r++) rect(w - margin - 12, y + kPanelRow / 2 - 12 + r, 12 - std::abs(r - 12), 1, kDim);
                }
                rect(margin, y + kPanelRow - 2, w - 2 * margin, 2, kLine);
                y += kPanelRow;
                continue;
            }
            const Row& row = all[ids[k]];
            const int sw = 132, sh = 60, sx = w - sw - margin, sy = y + (kPanelRow - sh) / 2;
            if (row.hint_es) {
                text(margin, y + 30, 4, es ? row.es : row.en, kText);
                text(margin, y + 30 + 7 * 4 + 14, 3, es ? row.hint_es : row.hint_en, kDim);
            } else {
                text(margin, y + (kPanelRow - 7 * 4) / 2, 4, es ? row.es : row.en, kText);
            }
            pill(sx, sy, sw, sh, row.on ? kGood : kOff);
            pill(row.on ? sx + sw - sh + 5 : sx + 5, sy + 5, sh - 10, sh - 10, kKnob);
            rect(margin, y + kPanelRow - 2, w - 2 * margin, 2, kLine);
            y += kPanelRow;
        }
        // Al pie, cómo se cierra.
        text(margin, h - 70, 3, es ? "TOCA FUERA PARA CERRAR" : "TAP OUTSIDE TO CLOSE", kDim);
    }
    bool create(std::vector<Buffer>& buffers, uint32_t width, uint32_t height, uint32_t count = kRing) {
        VkContext* c = vk();
        for (uint32_t k = 0; k < count; k++) {
            Buffer b;
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = VkDeviceSize{width} * height * 4;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VkMemoryRequirements req;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            if (vkCreateBuffer(c->device, &bi, nullptr, &b.buffer) != VK_SUCCESS) return fail();
            vkGetBufferMemoryRequirements(c->device, b.buffer, &req);
            ai.allocationSize = req.size;
            ai.memoryTypeIndex =
                find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &b.memory) != VK_SUCCESS ||
                vkBindBufferMemory(c->device, b.buffer, b.memory, 0) != VK_SUCCESS ||
                vkMapMemory(c->device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.data) != VK_SUCCESS)
                return fail();
            buffers.push_back(b);
        }
        return true;
    }
    bool fail() {
        NVN_LOG("hud: no se pudo crear su buffer; sin HUD\n");
        failed_ = true;
        return false;
    }

    void draw(Rgba* px, bool bgra) {
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= static_cast<int>(kW) || y >= static_cast<int>(kH)) return;
            px[y * kW + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int w, int h, Rgba c) {
            for (int y = y0; y < y0 + h; y++)
                for (int x = x0; x < x0 + w; x++) put(x, y, c);
        };
        const auto text = [&](int x, int y, int scale, const char* s, Rgba c) {
            for (; *s; s++, x += 6 * scale) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b)) rect(x + b * scale, y + r * scale, scale, scale, c);
            }
        };
        rect(0, 0, kW, kH, kBack);

        // El número cambia cuatro veces por segundo, para que se pueda leer.
        if (last_real_ - text_at_ > 250000000 || !text_at_) {
            text_at_ = last_real_;
            shown_ = static_cast<int>(reals_.size() + generated_.size());
        }
        // Verde a su ritmo (30, o 60 con LSFG), amarillo si se queda corto, rojo si va mal.
        const int target = generated_.empty() ? 30 : 60;
        const Rgba color = shown_ >= target - 1 ? kGood : shown_ >= target * 5 / 6 ? kWarn : kBad;
        char line[8];
        snprintf(line, sizeof line, "%d", std::min(shown_, 99));
        const int width = static_cast<int>(strlen(line)) * 6 * 3 - 3;
        text((static_cast<int>(kW) - width) / 2, 3, 3, line, color);
    }

    std::vector<Buffer> buffers_, panel_buffers_;
    // Las filas de las APKs publicadas: sus estados cambian en sus hilos (atómicos) y el panel se redibuja con el contador.
    std::atomic<int> report_state_{kIdle}, update_state_{kUnchecked};
    std::atomic<uint32_t> published_serial_{0};
    uint32_t published_seen_ = 0;
    mutable std::mutex versions_mutex_;
    std::string latest_, current_;                 // la última release de GitHub y la de esta APK (sin la v)
    bool update_notices_ = true, check_started_ = false;
    int64_t first_frame_ = 0, notice_until_ = 0;
    int notice_state_ = kUnchecked;
    bool notice_drawn_ = false;
    std::vector<Buffer> notice_buffers_;
    Buffer capture_{};
    VkDeviceSize capture_size_ = 0;
    uint32_t capture_w_ = 0, capture_h_ = 0;
    bool capture_bgra_ = false, capture_coherent_ = true, capture_wanted_ = false;
    std::atomic<bool> swapchain_src_{false};
    std::atomic<int64_t> capture_at_{0};
    uint32_t next_ = 0, panel_next_ = 0;
    bool failed_ = false;
    bool hidden_ = false, flag_ = false, flag_seen_ = false;  // el contador apagado; lo último que dijo <datos>/sin_fps
    bool panel_open_ = false, touch_seen_ = false, gesture_seen_ = false;
    bool panel_dirty_ = true, panel_drawn_hidden_ = false;  // hay que redibujarlo; el estado con el que se dibujó
    bool choose_folder_ = false;                   // la fila "Cambiar carpeta" (el fichero <datos>/elegir_carpeta)
    bool tv_ = false;                              // la fila "Modo TV" (el fichero <datos>/modo_tv)
    uint32_t panel_w_ = 0, panel_h_ = 0;           // el tamaño con el que se crearon sus buffers
    float panel_from_ = 0.0f;                     // lo que asomaba cuando se abrió o se cerró por última vez
    int64_t panel_at_ = 0, touched_at_ = 0;       // cuándo fue eso, y la última vez que se tocó
    uint32_t touch_handled_ = 0;                  // el toque (NnsdkInput::touch_serial) ya atendido
    uint32_t opened_by_ = 0;                      // el toque que lo abrió deslizando
    uint64_t gesture_ = 0;                        // el último NnsdkInput::hud_gesture atendido
    std::deque<double> intervals_;          // ms entre frames reales (para el 1 % más lento, si vuelve a hacer falta)
    std::deque<int64_t> reals_, generated_;  // en el último segundo
    int64_t last_real_ = 0, text_at_ = 0;
    int shown_ = 0;
};

Hud g_hud;

}  // namespace

void real_frame(int64_t now_ns) { g_hud.real_frame(now_ns); }
void generated_frame() { g_hud.generated_frame(); }
void swapchain_usage(uint32_t usage) { g_hud.set_swapchain_usage(usage); }
void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
    g_hud.record(cmd, image, extent, format);
}

}  // namespace nvn::hud
