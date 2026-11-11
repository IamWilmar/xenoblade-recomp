// nn::oe: estado de la aplicación en el sistema. Lo que el juego usa como reloj tiene que avanzar:
// CRI Mana (los vídeos) mide la reproducción con GetProgramTotalActiveTime (main+0xa5154c), y con 0
// fijo el vídeo se queda en los primeros frames.
//
// La Flip es una consola en modo portátil que nunca cambia de estado: rendimiento normal, siempre en primer
// plano y sin mensajes del sistema (los eventos de aviso existen pero no se señalan nunca). Grabación, capturas
// y la imagen de copyright no tienen dónde ir.

#include <sys/system_properties.h>
#include <time.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

settings::LanguageCode GetDesiredLanguage() {
    static const std::string code = [] {
        std::string c = desired_language();
        printf("[oe] idioma: %s\n", c.c_str());
        return c;
    }();
    settings::LanguageCode out{};
    strncpy(out.string, code.c_str(), sizeof out.string - 1);
    return out;
}

void GetDisplayVersion(DisplayVersion* out) {
    memset(out, 0, sizeof *out);
    strcpy(out->value, "16.0.1");
}

OperationMode GetOperationMode() { return OperationMode_Handheld; }
PerformanceMode GetPerformanceMode() { return PerformanceMode_Normal; }
FocusState GetCurrentFocusState() { return FocusState_InFocus; }
bool IsCpuOverclockEnabled() { return false; }

// La resolución del modo portátil, como la consola en la mano.
void GetDefaultDisplayResolution(int* width, int* height) {
    *width = 1280;
    *height = 720;
}
void GetDefaultDisplayResolutionChangeEvent(os::SystemEvent*) {}

os::SystemEventType* GetNotificationMessageEvent() { return reinterpret_cast<os::SystemEventType*>(g_message_event); }
bool TryPopNotificationMessage(uint32_t*) { return false; }

int GetPerformanceConfiguration(PerformanceMode mode) { return g_performance_config[mode == PerformanceMode_Boost]; }
void SetPerformanceConfiguration(PerformanceMode mode, int config) {
    g_performance_config[mode == PerformanceMode_Boost] = config;
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

}  // namespace nn::oe
