// El cuadro de texto de Android que hace de teclado de la consola (ver teclado_android.h).
//
// El cuadro es Java (teclado/Teclado.java): un AlertDialog con un EditText en el hilo de la interfaz, con su tecla ✓
// que acepta. La APK no lleva Java propio: el dex va dentro de esta biblioteca (teclado/teclado_dex.inc, generado por
// tools/teclado_dex.sh), se carga en memoria con InMemoryDexClassLoader y su método nativo resultado() se registra
// aquí. Así también llega con una actualización que solo cambia las .so.
//
// La actividad y la VM las deja el loader en NIER_ACTIVIDAD y NIER_JAVA_VM (punteros en hexadecimal; activity->clazz
// ya es una referencia global que vive lo que la actividad).

#include "nnsdk/teclado_android.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __ANDROID__
#include <jni.h>
#include <sys/system_properties.h>

#include "nnsdk/teclado/teclado_dex.inc"
#endif

namespace nn::swkbd::android_ui {

namespace {

std::string data_dir() {
    const char* env = getenv("NIER_DATA");
    return env ? env : "/sdcard/NieR";
}
bool existe(const char* nombre) { return access((data_dir() + "/" + nombre).c_str(), F_OK) == 0; }

// -1: libre. Abierto mientras se ve. Aceptado/Cancelado/Fallo: acabado y sin recoger (estado() lo deja libre).
std::atomic<int> g_estado{-1};
std::mutex g_texto_mutex;
std::u16string g_texto;

void terminar(Estado e, const std::u16string& texto = {}) {
    {
        std::lock_guard<std::mutex> lock(g_texto_mutex);
        g_texto = texto;
    }
    g_estado.store(e);
}

#ifdef __ANDROID__

void* puntero_de(const char* variable) {
    const char* v = getenv(variable);
    return v && *v ? reinterpret_cast<void*>(strtoull(v, nullptr, 16)) : nullptr;
}
JavaVM* vm() { return static_cast<JavaVM*>(puntero_de("NIER_JAVA_VM")); }
jobject actividad() { return static_cast<jobject>(puntero_de("NIER_ACTIVIDAD")); }

// El hilo que llama (uno del juego), enganchado a la VM mientras vive esto.
struct HiloJava {
    JNIEnv* env = nullptr;
    bool enganchado = false;
    HiloJava() {
        JavaVM* v = vm();
        if (!v) return;
        if (v->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
            JavaVMAttachArgs args{JNI_VERSION_1_6, "teclado", nullptr};
            if (v->AttachCurrentThread(&env, &args) == JNI_OK) enganchado = true;
            else env = nullptr;
        }
        if (env && env->PushLocalFrame(32) != JNI_OK) env->ExceptionClear();
    }
    ~HiloJava() {
        if (!env) return;
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->PopLocalFrame(nullptr);
        if (enganchado) vm()->DetachCurrentThread();
    }
    bool fallo(const char* paso) {
        if (!env->ExceptionCheck()) return false;
        env->ExceptionDescribe();
        env->ExceptionClear();
        printf("[teclado] falla en %s\n", paso);
        return true;
    }
    jstring java(const std::u16string& s) {
        return env->NewString(reinterpret_cast<const jchar*>(s.data()), jsize(s.size()));
    }
};

// Teclado.resultado(int, String), en el hilo de la interfaz.
void JNICALL resultado(JNIEnv* env, jclass, jint e, jstring texto) {
    std::u16string t;
    if (texto) {
        const jchar* chars = env->GetStringChars(texto, nullptr);
        t.assign(reinterpret_cast<const char16_t*>(chars), size_t(env->GetStringLength(texto)));
        env->ReleaseStringChars(texto, chars);
    }
    printf("[teclado] %s (%zu caracteres)\n", e == Aceptado ? "aceptado" : e == Cancelado ? "cancelado" : "falló",
           t.size());
    terminar(e == Aceptado || e == Cancelado ? Estado(e) : Fallo, t);
}

// La clase port.Teclado cargada del dex de dentro de la biblioteca (una vez; nullptr si no se pudo).
jclass clase_teclado(HiloJava& h) {
    static std::mutex m;
    static jclass clase = nullptr;
    static bool intentado = false;
    std::lock_guard<std::mutex> lock(m);
    if (intentado) return clase;
    intentado = true;
    JNIEnv* env = h.env;
    // ART lee el dex de este búfer: una copia que no se libera nunca.
    static std::vector<unsigned char> dex(kTecladoDex, kTecladoDex + sizeof kTecladoDex);
    jobject buffer = env->NewDirectByteBuffer(dex.data(), jlong(dex.size()));
    jclass c_context = env->FindClass("android/content/Context");
    jobject padre = env->CallObjectMethod(
        actividad(), env->GetMethodID(c_context, "getClassLoader", "()Ljava/lang/ClassLoader;"));
    if (h.fallo("getClassLoader") || !buffer) return nullptr;
    jclass c_cargador = env->FindClass("dalvik/system/InMemoryDexClassLoader");
    jobject cargador = env->NewObject(
        c_cargador, env->GetMethodID(c_cargador, "<init>", "(Ljava/nio/ByteBuffer;Ljava/lang/ClassLoader;)V"), buffer,
        padre);
    if (h.fallo("InMemoryDexClassLoader") || !cargador) return nullptr;
    jclass c_loader = env->FindClass("java/lang/ClassLoader");
    auto local = static_cast<jclass>(env->CallObjectMethod(
        cargador, env->GetMethodID(c_loader, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;"),
        env->NewStringUTF("port.Teclado")));
    if (h.fallo("loadClass") || !local) return nullptr;
    const JNINativeMethod nativos[] = {
        {const_cast<char*>("resultado"), const_cast<char*>("(ILjava/lang/String;)V"), reinterpret_cast<void*>(resultado)}};
    if (env->RegisterNatives(local, nativos, 1) != JNI_OK || h.fallo("RegisterNatives")) return nullptr;
    clase = static_cast<jclass>(env->NewGlobalRef(local));
    printf("[teclado] cuadro de texto listo (%zu bytes de dex)\n", dex.size());
    return clase;
}

#endif  // __ANDROID__

}  // namespace

bool disponible() {
#ifdef __ANDROID__
    return vm() && actividad() && !existe("nombre.txt") && !existe("sin_teclado");
#else
    return false;
#endif
}

bool abrir(const Pedido& p) {
#ifdef __ANDROID__
    if (!disponible()) return false;
    int actual = g_estado.load();
    do {
        if (actual == Abierto) return false;  // ya hay uno a la vista
    } while (!g_estado.compare_exchange_weak(actual, Abierto));
    HiloJava h;
    jclass clase = h.env ? clase_teclado(h) : nullptr;
    if (!clase) {
        g_estado.store(-1);
        return false;
    }
    std::u16string inicial = p.inicial;
    if (p.max > 0 && inicial.size() > size_t(p.max)) inicial.resize(size_t(p.max));
    jmethodID abrir_java = h.env->GetStaticMethodID(
        clase, "abrir",
        "(Landroid/app/Activity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
        "Ljava/lang/String;IZZ)V");
    if (!h.fallo("abrir") && abrir_java)
        h.env->CallStaticVoidMethod(clase, abrir_java, actividad(), h.java(p.titulo), h.java(p.mensaje), h.java(p.guia),
                                    h.java(inicial), h.java(p.ok), jint(p.max), jboolean(p.clave), jboolean(p.lineas));
    if (!abrir_java || h.fallo("Teclado.abrir")) {
        g_estado.store(-1);
        return false;
    }
    return true;
#else
    (void)p;
    return false;
#endif
}

Estado estado(std::u16string* texto) {
    const int e = g_estado.load();
    if (e < 0) return Fallo;  // no había ninguno
    if (e == Abierto) return Abierto;
    if (texto) {
        std::lock_guard<std::mutex> lock(g_texto_mutex);
        *texto = g_texto;
    }
    g_estado.store(-1);
    return Estado(e);
}

void cerrar() {
#ifdef __ANDROID__
    if (g_estado.load() != Abierto) return;
    HiloJava h;
    jclass clase = h.env ? clase_teclado(h) : nullptr;
    if (!clase) return;
    jmethodID cerrar_java = h.env->GetStaticMethodID(clase, "cerrar", "(Landroid/app/Activity;)V");
    if (!h.fallo("cerrar") && cerrar_java) h.env->CallStaticVoidMethod(clase, cerrar_java, actividad());
    h.fallo("Teclado.cerrar");
#endif
}

Estado preguntar(const Pedido& pedido, std::u16string* texto) {
    if (!abrir(pedido)) return Fallo;
    Estado e;
    while ((e = estado(texto)) == Abierto) std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return e;
}

bool en_espanol() {
#ifdef __ANDROID__
    char locale[PROP_VALUE_MAX] = {};
    if (__system_property_get("persist.sys.locale", locale) <= 0) __system_property_get("ro.product.locale", locale);
    return strncmp(locale, "es", 2) == 0;
#else
    return false;
#endif
}

}  // namespace nn::swkbd::android_ui
