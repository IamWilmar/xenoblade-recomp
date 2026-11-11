#pragma once
// Lo que el loader le pasa a libnnsdk.so antes de arrancar el juego.

#include <stddef.h>
#include <stdint.h>

#include <atomic>

// Mando de la Flip, ya en términos de la Switch: lo escribe el loader con los eventos de Android y lo
// lee nn::hid (modo portátil).
struct NnsdkInput {
    std::atomic<uint64_t> buttons{0};  // bits de nn::hid::NpadButton (A 0, B 1, X 2, Y 3 …)
    std::atomic<int32_t> sticks[4] = {};  // izquierdo x, y; derecho x, y; -32767..32767, y positivo arriba
    // Pulsaciones recientes: siguen contando hasta latched_until_ns (CLOCK_MONOTONIC) aunque ya se hayan
    // soltado, para que un toque más corto que un frame no se pierda entre dos lecturas del juego.
    std::atomic<uint64_t> latched{0};
    std::atomic<int64_t> latched_until_ns{0};
};

// La ventana de Android: se destruye al irse la app a segundo plano (o al apagar la pantalla) y llega otra al volver.
// La escribe el loader (APP_CMD_TERM_WINDOW / INIT_WINDOW, serial + 1 en cada cambio); el presentador (window.cpp)
// deja de usar la vieja, crea la superficie y el swapchain en la nueva y apunta en done el serial que ya atendió.
struct NnsdkWindow {
    std::atomic<void*> window{nullptr};
    std::atomic<uint32_t> serial{0}, done{0};
};

struct NnsdkHost {
    // El juego exporta su propio allocator: lo que libnnsdk le devuelva y él libere va por aquí.
    void* (*game_malloc)(size_t);
    void (*game_free)(void*);
    void* (*game_new)(size_t);    // operator new del juego (_Znwm)
    void (*game_delete)(void*);   // operator delete del juego (_ZdlPv)
    // Se llama al principio de cada hilo que crea nn::os.
    void (*thread_start)();
    // La plantilla del TLS del juego (thread_local), de __nnmusl_init_dso: ver loader/tls.h.
    void (*tls_template)(const void* image, size_t image_size, size_t total_size, size_t align);
    // Stub con nombre que registra sus llamadas y devuelve ret (funciones NVN sin implementar).
    void* (*make_stub)(const char* name, uint64_t ret);
    // ANativeWindow* de la APK; nullptr si se corre desde la terminal.
    void* native_window;
    // Estado del mando (nullptr en la terminal: no hay mando conectado).
    NnsdkInput* input;
    // La ventana actual (nullptr en la terminal: la de native_window no cambia).
    NnsdkWindow* window_state;
    // DEPURACIÓN: el próximo acceso a la página de addr se registra (pc, lr, backtrace) y se deja pasar.
    void (*watch_writes)(uint64_t addr);
    // Dónde quedó main (base y tamaño): para leer y tocar datos del juego por su offset (ver window.cpp, MaxFps).
    const uint8_t* main_base;
    size_t main_size;
};

extern "C" void nnsdk_attach(const NnsdkHost* host);
