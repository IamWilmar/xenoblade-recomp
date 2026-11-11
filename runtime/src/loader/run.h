#pragma once
// Núcleo del loader, común a la terminal (nier_loader) y a la APK (libnier.so).

#include <string>

struct RunConfig {
    std::string exefs;               // carpeta con main, sdk y main.npdm
    std::string nnsdk;               // ruta o soname de libnnsdk.so
    bool list_stubs = false;
    int watchdog_seconds = 15;       // <= 0: sin watchdog
    void* native_window = nullptr;   // ANativeWindow* en la APK; nullptr en la terminal
    struct NnsdkInput* input = nullptr;  // mando (APK)
    struct NnsdkWindow* window_state = nullptr;  // la ventana cuando cambia (APK)
};

// Carga main, lo enlaza y corre nninitStartup, constructores y nnMain. Vuelve al terminar nnMain.
int nier_run(const RunConfig& cfg);
