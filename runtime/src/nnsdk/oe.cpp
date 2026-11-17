// nn::oe: estado de la aplicación en el sistema. Lo que el juego usa como reloj tiene que avanzar:
// CRI Mana (los vídeos) mide la reproducción con GetProgramTotalActiveTime (main+0xa5154c), y con 0
// fijo el vídeo se queda en los primeros frames.
//
// La Flip es una consola en modo portátil que nunca cambia de estado (salvo el modo TV, fijo en cada sesión:
// nnsdk::tv_mode): rendimiento normal, siempre en primer plano y sin mensajes del sistema (los eventos de aviso
// existen pero no se señalan nunca). Grabación, capturas y la imagen de copyright no tienen dónde ir.

#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn {
namespace os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}
namespace time {
struct PosixTime {
    int64_t value;
};
}  // namespace time
namespace settings {
enum Language : int;
struct LanguageCode {  // como en settings.cpp
    char string[8];
    static LanguageCode Make(Language language);
};
}  // namespace settings
namespace util {
struct Uuid {  // como en util.cpp: los 16 bytes en el orden en que se escriben
    uint8_t data[16];
};
Uuid GenerateUuid();
}  // namespace util
}  // namespace nn

namespace nn::oe {

enum OperationMode : uint8_t { OperationMode_Handheld = 0, OperationMode_Console = 1 };
enum PerformanceMode : int { PerformanceMode_Normal = 0, PerformanceMode_Boost = 1 };
enum FocusState : uint8_t { FocusState_InFocus = 1, FocusState_OutOfFocus = 2, FocusState_Background = 3 };
enum CpuBoostMode : int {};
enum FocusHandlingMode : int {};
enum WindowOriginMode : int {};

struct DisplayVersion {
    char value[16];
};

namespace {

int64_t monotonic_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec;
}

const int64_t g_start_ns = monotonic_ns();

// El SystemEvent de los mensajes del sistema (en el sdk, un global que devuelve GetNotificationMessageEvent).
// os.cpp crea su evento al primer uso, sin señalar.
alignas(16) unsigned char g_message_event[0x40];

// Configuración de rendimiento por modo (la de serie del portátil: 1020 MHz de CPU, 384 de GPU).
int g_performance_config[2] = {0x00010000, 0x00020000};

// El idioma que pide el juego: el de Android traducido a los que trae MH Rise en su control.nacp (japonés,
// inglés de EE. UU., francés, alemán, italiano, español de España y de Latinoamérica, ruso, coreano y chino
// tradicional y simplificado). Se puede forzar con <datos>/idioma.txt (un código de Nintendo: "es", "es-419"…).
std::string desired_language() {
    const char* env = getenv("NIER_DATA");
    std::string path = std::string(env ? env : "/sdcard/NieR") + "/idioma.txt";
    if (FILE* f = fopen(path.c_str(), "r")) {
        char code[16] = {};
        bool ok = fscanf(f, "%15s", code) == 1;
        fclose(f);
        if (ok) return code;
    }
    char locale[PROP_VALUE_MAX] = {};
    if (__system_property_get("persist.sys.locale", locale) <= 0) __system_property_get("ro.product.locale", locale);
    std::string l = locale;  // "es-CO", "zh-Hant-TW", "en-US"…
    std::string lang = l.substr(0, l.find('-')), rest = l.size() > lang.size() ? l.substr(lang.size()) : "";
    if (lang == "ja" || lang == "fr" || lang == "de" || lang == "it" || lang == "ru" || lang == "ko") return lang;
    if (lang == "es") return rest.empty() || rest == "-ES" ? "es" : "es-419";
    if (lang == "zh")
        return rest.find("Hant") != std::string::npos || rest.find("TW") != std::string::npos ||
                       rest.find("HK") != std::string::npos || rest.find("MO") != std::string::npos
                   ? "zh-Hant"
                   : "zh-Hans";
    return "en-US";
}

}  // namespace

void Initialize() {}

// Tiempo activo desde que arrancó el programa (en la consola no cuenta el tiempo en suspensión).
TimeSpan GetProgramTotalActiveTime() { return TimeSpan{monotonic_ns() - g_start_ns}; }

// Idiomas que declara el juego (SupportedLanguageFlag del control.nacp, +0x302C). La consola nunca le da uno que no
// declare: Xenoblade 3 de cartucho europeo no trae inglés de EE. UU. y con "en-US" buscaba bdat/us y
// menu/font/bf3_us.wisty, que no existen, y se caía al montar la pantalla de título (main+0x10141ac). Se lee de
// <datos>/control.nacp (copiado del volcado); sin él se aceptan todos.
static uint32_t supported_languages() {
    const char* env = getenv("NIER_DATA");
    uint32_t mask = 0;
    if (FILE* f = fopen((std::string(env ? env : "/sdcard/NieR") + "/control.nacp").c_str(), "rb")) {
        if (fseek(f, 0x302C, SEEK_SET) != 0 || fread(&mask, 4, 1, f) != 1) mask = 0;
        fclose(f);
    }
    return mask ? mask : 0xFFFF;
}

// La regla de la consola: el idioma pedido si el juego lo tiene; si no, su pariente (inglés de EE. UU. <-> de Reino
// Unido, francés <-> de Canadá, español <-> de Latinoamérica, portugués <-> de Brasil, chino tradicional <->
// simplificado); si tampoco, el primero que declare el juego.
static std::string supported_language(const std::string& wanted) {
    static const char* const kCodes[] = {"en-US", "en-GB", "ja",    "fr",      "de",      "es-419", "es",    "it",
                                         "nl",    "fr-CA", "pt",    "ru",      "ko",      "zh-Hant", "zh-Hans", "pt-BR"};
    const uint32_t mask = supported_languages();
    auto index = [&](const std::string& c) {
        for (int k = 0; k < 16; k++)
            if (c == kCodes[k]) return k;
        return -1;
    };
    auto ok = [&](int k) { return k >= 0 && (mask >> k & 1); };
    static const int kSibling[] = {1, 0, -1, 9, -1, 6, 5, -1, -1, 3, 15, -1, -1, 14, 13, 10};
    const int k = index(wanted);
    if (ok(k)) return wanted;
    if (k >= 0 && ok(kSibling[k])) return kCodes[kSibling[k]];
    for (int j = 0; j < 16; j++)
        if (ok(j)) return kCodes[j];
    return wanted;
}

settings::LanguageCode GetDesiredLanguage() {
    static const std::string code = [] {
        const std::string pedido = desired_language();
        std::string c = supported_language(pedido);
        printf("[oe] idioma: %s%s\n", c.c_str(), c == pedido ? "" : (" (pedido " + pedido + ", el juego no lo trae)").c_str());
        return c;
    }();
    settings::LanguageCode out{};
    strncpy(out.string, code.c_str(), sizeof out.string - 1);
    return out;
}

// La versión del juego (DisplayVersion del control.nacp, +0x3060): Xenoblade 3 la pinta en el título ("Ver. 2.2.0");
// antes salía "16.0.1", que es la del firmware. Sin <datos>/control.nacp, "1.0.0".
void GetDisplayVersion(DisplayVersion* out) {
    memset(out, 0, sizeof *out);
    strcpy(out->value, "1.0.0");
    const char* env = getenv("NIER_DATA");
    if (FILE* f = fopen((std::string(env ? env : "/sdcard/NieR") + "/control.nacp").c_str(), "rb")) {
        char v[16] = {};
        if (fseek(f, 0x3060, SEEK_SET) == 0 && fread(v, 1, sizeof v - 1, f) == sizeof v - 1 && v[0])
            memcpy(out->value, v, sizeof v);
        fclose(f);
    }
}

// Portátil salvo en modo TV (nnsdk::tv_mode), que es la consola en la base: Xenoblade elige con esto su resolución
// (main importa GetOperationMode y GetPerformanceMode, no GetDefaultDisplayResolution).
OperationMode GetOperationMode() { return nnsdk::tv_mode() ? OperationMode_Console : OperationMode_Handheld; }
PerformanceMode GetPerformanceMode() { return nnsdk::tv_mode() ? PerformanceMode_Boost : PerformanceMode_Normal; }
FocusState GetCurrentFocusState() { return FocusState_InFocus; }
bool IsCpuOverclockEnabled() { return false; }

// La resolución de la pantalla: la del portátil, o la de la tele en modo TV.
void GetDefaultDisplayResolution(int* width, int* height) {
    *width = nnsdk::tv_mode() ? 1920 : 1280;
    *height = nnsdk::tv_mode() ? 1080 : 720;
}
void GetDefaultDisplayResolutionChangeEvent(os::SystemEvent*) {}

os::SystemEventType* GetNotificationMessageEvent() { return reinterpret_cast<os::SystemEventType*>(g_message_event); }
bool TryPopNotificationMessage(uint32_t*) { return false; }

int GetPerformanceConfiguration(PerformanceMode mode) { return g_performance_config[mode == PerformanceMode_Boost]; }
void SetPerformanceConfiguration(PerformanceMode mode, int config) {
    g_performance_config[mode == PerformanceMode_Boost] = config;
}
// La misma configuración para los dos modos: el sdk llama dos veces a nn::apm, con Normal y con Boost (sdk+0x341358 y
// sdk+0x3413fc). Solo admite 0x10001, 0x20003, 0x20004 y 0x92220008 (CPU a 1020 MHz y GPU a 768, 307, 384 y 460), y
// con otro valor aborta (sdk+0x341400); aquí los relojes son los de la Flip y solo se recuerda lo pedido. Unity lo tiene
// en su tabla de funciones de la plataforma (main+0x4c9d8e0 → main+0xf854e0 → main+0x376dd0); el juego cambia de
// configuración con la de dos argumentos (NXapiControl, main+0x1abb3d4).
void SetPerformanceConfiguration(int config) {
    g_performance_config[0] = g_performance_config[1] = config;
}
void SetCpuBoostMode(CpuBoostMode) {}

void SetFocusHandlingMode(FocusHandlingMode) {}
void SetResumeNotificationEnabled(bool) {}
void SetOperationModeChangedNotificationEnabled(bool) {}
void SetPerformanceModeChangedNotificationEnabled(bool) {}
void EnterExitRequestHandlingSection() {}
void LeaveExitRequestHandlingSection() {}

void FinishStartupLogo() {}
void EnableRecording() {}
void DisableRecording() {}
void MuteRecording() {}
void UnmuteRecording() {}
// Relanzar el programa (Unreal lo usa para reiniciarse: main+0x14fc8, tras PushOpenUsers). En la consola no vuelve:
// el proceso termina y el sistema abre el programa de nuevo. Aquí la app se cierra y hay que volver a abrirla.
[[noreturn]] void ExecuteProgram(int index, const void*, size_t size) {
    printf("[oe] ExecuteProgram(%d, %zu bytes de argumento): el juego pide reiniciarse; se cierra\n", index, size);
    fflush(stdout);
    exit(0);
}
void EnableScreenShot() {}
void DisableScreenShot() {}
void InitializeCopyrightFrameBuffer(void*, size_t) {}
void SetCopyrightImage(const void*, size_t, int, int, int, int, WindowOriginMode) {}
void SetCopyrightVisibility(bool) {}

// Aviso de mantenimiento de los servidores: no hay servidores.
void SendServerMaintenanceOverlayNotification(time::PosixTime, time::PosixTime) {}

// Alargar el tiempo hasta el apagado automático de pantalla (cinemáticas): en Android lo lleva la ventana; aquí solo se
// recuerda lo que pidió. Modo 0 = desactivado.
namespace {
std::atomic<bool> g_inactivity_extended{false};
}  // namespace
void SetUserInactivityDetectionTimeExtended(bool extended) { g_inactivity_extended = extended; }
void SetUserInactivityDetectionTimeExtendedUnsafe(bool extended) { g_inactivity_extended = extended; }
bool IsUserInactivityDetectionTimeExtended() { return g_inactivity_extended; }
int GetUserInactivityDetectionTimeExtensionMode() { return 0; }

// El identificador de "esta consola para esta aplicación": un UUID que el sistema da siempre igual (el sdk se lo pide
// al servicio y aborta si falla, sdk+0x342650). Unity saca de él SystemInfo.deviceUniqueIdentifier (main+0x37755c: lo
// escribe con %02x en grupos de 8-4-4-4-12 y lo guarda) y la semilla de sus GUID (main+0xc07b2c). Aquí es uno al azar
// que se crea la primera vez y se queda en <datos>/pseudo_device_id.txt (borrarlo equivale a reiniciar la consola de
// fábrica); si la carpeta no deja escribir, vale lo que dure el proceso.
void GetPseudoDeviceId(util::Uuid* out) {
    static const util::Uuid id = [] {
        const char* env = getenv("NIER_DATA");
        const std::string path = std::string(env ? env : "/sdcard/NieR") + "/pseudo_device_id.txt";
        util::Uuid uuid{};
        char text[40] = {};
        bool ok = false;
        if (FILE* f = fopen(path.c_str(), "r")) {
            ok = fread(text, 1, 32, f) == 32;
            fclose(f);
            for (int i = 0; ok && i < 16; i++) {
                unsigned byte = 0;
                ok = sscanf(text + 2 * i, "%2x", &byte) == 1;
                uuid.data[i] = static_cast<uint8_t>(byte);
            }
        }
        if (!ok) {
            uuid = util::GenerateUuid();
            for (int i = 0; i < 16; i++) snprintf(text + 2 * i, 3, "%02x", uuid.data[i]);
            if (FILE* f = fopen(path.c_str(), "w")) {
                fprintf(f, "%s\n", text);
                fclose(f);
            } else {
                printf("[oe] no se pudo guardar %s: el id de dispositivo cambiará en el próximo arranque\n", path.c_str());
            }
        }
        return uuid;
    }();
    *out = id;
}

// Si el sistema debe generar un informe de error cuando la aplicación se cae: el sdk solo le pasa el bool al sistema
// (sdk+0x343140 y sdk+0x3431a0) y no hay forma de leerlo de vuelta. Android no deja a una app decidirlo (sus
// tombstones los hace siempre el sistema) y el volcado propio del loader (loader/crash.cpp) es para depurar el port y
// conviene siempre: no hay nada que encender ni apagar. Solo los llama el plugin de Unity del SDK (nn.oe.CrashReport,
// main+0x155c950 y +0x155c960), que en Disco Elysium no tiene llamadores.
void EnableCrashReport() {}
void DisableCrashReport() {}
// SDK 12 (Xenoblade 3): avisos de capturas guardadas en el álbum; sin álbum no llega ninguno.
void EnableAlbumImageTakenNotification() {}
void DisableAlbumImageTakenNotification() {}

}  // namespace nn::oe

namespace nnsdk {

bool tv_mode() {
    static const bool on = [] {
        const char* env = getenv("NIER_DATA");
        const std::string path = std::string(env ? env : "/sdcard/NieR") + "/modo_tv";
        const bool v = access(path.c_str(), F_OK) == 0;
        printf("[oe] modo %s\n", v ? "TV (base, mando Pro en No1)" : "portátil");
        return v;
    }();
    return on;
}

}  // namespace nnsdk
