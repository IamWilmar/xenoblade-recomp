// Entrada de la APK (NativeActivity): espera la ventana, manda stdout a <datos>/log.txt y
// arranca el juego en su propio hilo. El hilo de android_main sigue atendiendo a Android y traduce
// el mando de la Flip a botones de la Switch.

#include <mutex>
#include <vector>
#include <android/input.h>
#include <android/keycodes.h>
#include <android/log.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "loader/crash.h"
#include "loader/run.h"
#include "nnsdk/host.h"
#include "nnsdk/rutas.h"

namespace {

// <datos>: la carpeta del juego que eligió el usuario en la pantalla de inicio (SetupActivity, files/carpeta.txt).
std::string g_data;
RunConfig g_cfg;

// El log pasa por una tubería de 1 MB que vacía un hilo propio ("LogSD"). Antes cada printf escribía en la SD (FUSE →
// MediaProvider → exFAT) desde el hilo que lo hacía, y con <datos>/registro línea a línea: ~100 escrituras por segundo
// desde el executor, el hilo del juego (las instantáneas del vigilante, 100 líneas) y el del audio. Cuando la SD se
// atascaba (guardando pipelines.bin, 4-7 s) esos hilos esperaban con ella: de 30 a 1-3 fps y el audio a saltos con el
// juego quieto. Ahora solo espera si la tubería se llena (~100 s del log normal). <datos>/log_directo: como antes.
int g_log_file = -1;  // log.txt
int g_log_pipe = -1;  // extremo de lectura de la tubería
std::atomic<uint64_t> g_log_read{0}, g_log_written{0};  // bytes que LogSD sacó de la tubería y que ya escribió

void write_all(int fd, const char* p, ssize_t n) {
    while (n > 0) {
        const ssize_t w = write(fd, p, static_cast<size_t>(n));
        if (w > 0) {
            p += w;
            n -= w;
        } else if (w < 0 && errno == EINTR) {
            continue;
        } else {
            return;  // SD llena o quitada: se pierde
        }
    }
}

void* log_writer(void*) {
    pthread_setname_np(pthread_self(), "LogSD");
    static char buffer[1 << 16];
    for (;;) {
        const ssize_t n = read(g_log_pipe, buffer, sizeof buffer);
        if (n > 0) {
            g_log_read += static_cast<uint64_t>(n);
            write_all(g_log_file, buffer, n);
            g_log_written += static_cast<uint64_t>(n);
        } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
            if (errno == EAGAIN) usleep(1000);  // drain_log_pipe la dejó sin bloqueo: el proceso está saliendo
        } else {
            return nullptr;
        }
    }
}

// Tras un crash, antes de _exit (async-signal-safe): que lo que quede en la tubería llegue al fichero. Lo normal es que
// LogSD lo vacíe en unos ms (y así queda en orden): se le esperan hasta 300 ms. Si está atascado con la SD, se vacía
// desde aquí (sin bloquear; el trozo que tenga LogSD a medias puede quedar después o perderse).
void drain_log_pipe() {
    const timespec step{0, 5 * 1000 * 1000};
    for (int k = 0, quiet = 0; k < 60; k++) {
        int pending = 0;
        const bool empty = ioctl(g_log_pipe, FIONREAD, &pending) == 0 && pending == 0 && g_log_read == g_log_written;
        if (empty && ++quiet == 2) return;  // dos veces seguidas: LogSD no tiene nada entre manos
        if (!empty) quiet = 0;
        nanosleep(&step, nullptr);
    }
    const int flags = fcntl(g_log_pipe, F_GETFL);
    if (flags >= 0) fcntl(g_log_pipe, F_SETFL, flags | O_NONBLOCK);
    char buffer[4096];
    for (int k = 0; k < 4096; k++) {  // como mucho 16 MB
        const ssize_t n = read(g_log_pipe, buffer, sizeof buffer);
        if (n <= 0) break;
        write_all(g_log_file, buffer, n);
    }
}

// stdout (y luego stderr) pasa a ser la tubería; el fichero lo escribe log_writer. false: sigue escribiendo directo.
bool start_log_pipe() {
    fflush(stdout);
    int p[2];
    if (pipe2(p, O_CLOEXEC) != 0) return false;
    fcntl(p[1], F_SETPIPE_SZ, 1 << 20);  // si no se puede, se queda en 64 KB
    const int file = dup(fileno(stdout));
    pthread_t thread;
    g_log_file = file;
    g_log_pipe = p[0];
    if (file < 0 || pthread_create(&thread, nullptr, log_writer, nullptr) != 0) {
        if (file >= 0) close(file);
        close(p[0]);
        close(p[1]);
        g_log_file = g_log_pipe = -1;
        return false;
    }
    pthread_detach(thread);
    dup2(p[1], fileno(stdout));
    close(p[1]);
    g_crash_log_drain = drain_log_pipe;
    return true;
}
NnsdkInput g_input;
uint64_t g_key_buttons = 0, g_axis_buttons = 0;

// Bits de nn::hid::NpadButton.
enum : uint64_t {
    kA = 1 << 0, kB = 1 << 1, kX = 1 << 2, kY = 1 << 3, kStickL = 1 << 4, kStickR = 1 << 5, kL = 1 << 6, kR = 1 << 7,
    kZL = 1 << 8, kZR = 1 << 9, kPlus = 1 << 10, kMinus = 1 << 11, kLeft = 1 << 12, kUp = 1 << 13, kRight = 1 << 14,
    kDown = 1 << 15, kStickLLeft = 1 << 16, kStickLUp = 1 << 17, kStickLRight = 1 << 18, kStickLDown = 1 << 19,
    kStickRLeft = 1 << 20, kStickRUp = 1 << 21, kStickRRight = 1 << 22, kStickRDown = 1 << 23,
};

// Por posición, con la disposición de Nintendo: el de abajo es B, el de la derecha A.
uint64_t button_for(int32_t keycode) {
    switch (keycode) {
        case AKEYCODE_BUTTON_A: return kB;
        case AKEYCODE_BUTTON_B: return kA;
        case AKEYCODE_BUTTON_X: return kY;
        case AKEYCODE_BUTTON_Y: return kX;
        case AKEYCODE_BUTTON_L1: return kL;
        case AKEYCODE_BUTTON_R1: return kR;
        case AKEYCODE_BUTTON_L2: return kZL;
        case AKEYCODE_BUTTON_R2: return kZR;
        case AKEYCODE_BUTTON_THUMBL: return kStickL;
        case AKEYCODE_BUTTON_THUMBR: return kStickR;
        case AKEYCODE_BUTTON_START: return kPlus;
        case AKEYCODE_BUTTON_SELECT:
        case AKEYCODE_BACK: return kMinus;
        case AKEYCODE_DPAD_UP: return kUp;
        case AKEYCODE_DPAD_DOWN: return kDown;
        case AKEYCODE_DPAD_LEFT: return kLeft;
        case AKEYCODE_DPAD_RIGHT: return kRight;
        default: return 0;
    }
}

int32_t to_stick(float v) { return static_cast<int32_t>(std::lround(std::fmax(-1.0f, std::fmin(1.0f, v)) * 32767.0f)); }

uint64_t g_auto_buttons = 0;  // DEPURACIÓN: autojuego (NIER_AUTOJUEGO)
// El jugador tocó el mando físico: el autojuego se retira para siempre (las teclas que inyecta adb vienen del
// dispositivo virtual, id < 0, y no cuentan).
std::atomic<bool> g_player_took_over{false};
void take_over(AInputEvent* event) {
    if (AInputEvent_getDeviceId(event) <= 0) return;
    // Solo un botón pulsado o un stick movido de verdad: el mando de la Flip manda eventos de ejes en reposo, y eso
    // paraba el paseo automático nada más empezar.
    const bool key = AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY && AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_DOWN;
    const bool stick = AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
                       (std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_X, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Y, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Z, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_RZ, 0)) > 0.5f);
    if ((!key && !stick) || g_player_took_over.exchange(true)) return;
    g_auto_buttons = 0;
    printf("[entrada] el jugador toma el mando: autojuego desactivado\n");
}

// DEPURACIÓN: grabación del mando. Con <datos>/grabar al arrancar, cada cambio (botones y sticks) se apunta con su
// segundo desde el arranque en <datos>/grabacion.txt ("t botones lx ly rx ry"); un hilo lo escribe cada segundo, así
// ningún evento espera a la SD. Con <datos>/reproducir al arrancar, se vuelve a tocar igual desde el arranque (para
// repetir una sesión del jugador en las pruebas). Tocar el mando físico durante la reproducción la para.
const auto g_boot = std::chrono::steady_clock::now();
bool g_recording = false, g_replaying = false;
std::mutex g_rec_mutex;
std::string g_rec_buffer;
uint64_t g_rec_last_buttons = ~0ull;
int32_t g_rec_last_sticks[4] = {};

double since_boot() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_boot).count(); }

void record_input() {
    if (!g_recording || g_replaying) return;
    std::lock_guard lock(g_rec_mutex);
    const uint64_t buttons = g_input.buttons;
    int32_t sticks[4];
    for (int k = 0; k < 4; k++) sticks[k] = g_input.sticks[k];
    if (buttons == g_rec_last_buttons && memcmp(g_rec_last_sticks, sticks, sizeof sticks) == 0) return;
    g_rec_last_buttons = buttons;
    memcpy(g_rec_last_sticks, sticks, sizeof sticks);
    char line[96];
    snprintf(line, sizeof line, "%.3f %llx %d %d %d %d\n", since_boot(), static_cast<unsigned long long>(buttons), sticks[0],
             sticks[1], sticks[2], sticks[3]);
    g_rec_buffer += line;
}

void publish() {
    g_input.buttons = g_key_buttons | g_axis_buttons | g_auto_buttons;
    record_input();
}

void* recorder(void*) {
    const std::string file = g_data + "/grabacion.txt";
    if (FILE* f = fopen(file.c_str(), "w")) fclose(f);
    for (;;) {
        sleep(1);
        std::string chunk;
        {
            std::lock_guard lock(g_rec_mutex);
            chunk.swap(g_rec_buffer);
        }
        if (chunk.empty()) continue;
        if (FILE* f = fopen(file.c_str(), "a")) {
            fwrite(chunk.data(), 1, chunk.size(), f);
            fclose(f);
        }
    }
    return nullptr;
}

void* replayer(void*) {
    struct Event {
        double t;
        unsigned long long buttons;
        int32_t sticks[4];
    };
    std::vector<Event> events;
    if (FILE* f = fopen((g_data + "/grabacion.txt").c_str(), "r")) {
        Event e{};
        while (fscanf(f, "%lf %llx %d %d %d %d", &e.t, &e.buttons, &e.sticks[0], &e.sticks[1], &e.sticks[2], &e.sticks[3]) == 6)
            events.push_back(e);
        fclose(f);
    }
    printf("[entrada] reproducción: %zu eventos (%.0f s)\n", events.size(), events.empty() ? 0.0 : events.back().t);
    for (const Event& e : events) {
        const double wait = e.t - since_boot();
        if (wait > 0) usleep(static_cast<useconds_t>(wait * 1e6));
        if (g_player_took_over) {
            printf("[entrada] reproducción parada: el jugador tocó el mando\n");
            return nullptr;
        }
        for (int k = 0; k < 4; k++) g_input.sticks[k] = e.sticks[k];
        g_auto_buttons = e.buttons;
        publish();
    }
    printf("[entrada] reproducción terminada\n");
    return nullptr;
}

// DEPURACIÓN: con NIER_AUTOJUEGO=<segundo>, desde ese segundo mantiene R (disparo de la unidad de vuelo) y
// mueve el stick izquierdo en zigzag, para que las pruebas lleguen lejos sin que el jugador muera.
void* autoplay(void*) {
    const double from = atof(getenv("NIER_AUTOJUEGO"));
    timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        usleep(50'000);
        timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        const double t = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (t < from) continue;
        if (g_player_took_over) return nullptr;
        g_auto_buttons = 1 << 7;  // R
        g_input.sticks[0] = to_stick(0.7f * static_cast<float>(std::sin(t * 1.6)));
        g_input.sticks[1] = to_stick(0.4f * static_cast<float>(std::sin(t * 0.7)));
        publish();
    }
    return nullptr;
}

// DEPURACIÓN: mientras exista <datos>/paseo, el personaje camina en zigzag y la cámara gira despacio (para medir el
// mundo abierto sin el jugador). Al quitar el fichero se sueltan los sticks; tocar el mando físico lo para del todo.
// DEPURACIÓN: <datos>/mando con "lx ly rx ry botones" (sticks de -1 a 1, botones en hex con los bits de NpadButton:
// 1 = A, 2 = B, 0x400 = +…): el mando lo pone ese fichero mientras exista (pruebas sin tocar la consola; un evento de
// entrada de adb cuenta como el jugador y pararía esto).
bool scripted_pad() {
    static const std::string file = g_data + "/mando";
    static bool active = false;
    FILE* f = fopen(file.c_str(), "r");
    if (!f) {
        if (active) {
            active = false;
            for (auto& s : g_input.sticks) s = 0;
            g_auto_buttons = 0;
            publish();
            printf("[entrada] mando por fichero: no\n");
        }
        return false;
    }
    float v[4] = {};
    unsigned long long buttons = 0;
    const int n = fscanf(f, "%f %f %f %f %llx", &v[0], &v[1], &v[2], &v[3], &buttons);
    fclose(f);
    if (!active) printf("[entrada] mando por fichero: sí\n");
    active = true;
    if (n >= 4) {
        g_input.sticks[0] = to_stick(v[0]), g_input.sticks[1] = to_stick(v[1]);
        g_input.sticks[2] = to_stick(v[2]), g_input.sticks[3] = to_stick(v[3]);
    }
    g_auto_buttons = n == 5 ? buttons : 0;
    publish();
    return true;
}

void* walk(void*) {
    const std::string file = g_data + "/paseo";
    bool active = false;
    double t = 0;
    for (;;) {
        usleep(50'000);
        if (g_player_took_over) return nullptr;
        if (scripted_pad()) continue;
        const bool want = access(file.c_str(), F_OK) == 0;
        if (want != active) {
            active = want;
            printf("[entrada] paseo automático: %s\n", active ? "sí" : "no");
            if (!active) {
                for (auto& s : g_input.sticks) s = 0;
                publish();
            }
        }
        if (!active) continue;
        t += 0.05;
        g_input.sticks[0] = to_stick(0.6f * static_cast<float>(std::sin(t * 0.5)));
        g_input.sticks[1] = to_stick(0.9f);                                          // adelante
        g_input.sticks[2] = to_stick(0.35f * static_cast<float>(std::sin(t * 0.2)));  // la cámara barre
        publish();
    }
    return nullptr;
}


int32_t on_input(android_app*, AInputEvent* event) {
    take_over(event);
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY) {
        uint64_t bit = button_for(AKeyEvent_getKeyCode(event));
        static int logged = 0;  // DEPURACIÓN: las primeras teclas que llegan (de qué dispositivo)
        if (logged++ < 12)
            printf("[entrada] tecla %d acción %d dispositivo %d fuente %#x -> %#llx (en %p)\n", AKeyEvent_getKeyCode(event),
                   AKeyEvent_getAction(event), AInputEvent_getDeviceId(event), AInputEvent_getSource(event),
                   static_cast<unsigned long long>(bit), static_cast<void*>(&g_input));
        if (!bit) return 0;
        if (AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_DOWN) {
            g_key_buttons |= bit;
            timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            g_input.latched |= bit;
            g_input.latched_until_ns = int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec + 60'000'000;  // ~4 frames
        }
        else if (AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_UP)
            g_key_buttons &= ~bit;
        publish();
        return 1;  // si no, Android convierte B en "atrás" y cierra la app
    }
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
        (AInputEvent_getSource(event) & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        auto axis = [&](int32_t a) { return AMotionEvent_getAxisValue(event, a, 0); };
        float lx = axis(AMOTION_EVENT_AXIS_X), ly = axis(AMOTION_EVENT_AXIS_Y);
        float rx = axis(AMOTION_EVENT_AXIS_Z), ry = axis(AMOTION_EVENT_AXIS_RZ);
        g_input.sticks[0] = to_stick(lx);
        g_input.sticks[1] = to_stick(-ly);  // en Android abajo es positivo
        g_input.sticks[2] = to_stick(rx);
        g_input.sticks[3] = to_stick(-ry);
        float hx = axis(AMOTION_EVENT_AXIS_HAT_X), hy = axis(AMOTION_EVENT_AXIS_HAT_Y);
        float lt = std::fmax(axis(AMOTION_EVENT_AXIS_LTRIGGER), axis(AMOTION_EVENT_AXIS_BRAKE));
        float rt = std::fmax(axis(AMOTION_EVENT_AXIS_RTRIGGER), axis(AMOTION_EVENT_AXIS_GAS));
        uint64_t b = 0;
        if (hx < -0.5f) b |= kLeft;
        if (hx > 0.5f) b |= kRight;
        if (hy < -0.5f) b |= kUp;
        if (hy > 0.5f) b |= kDown;
        if (lt > 0.5f) b |= kZL;
        if (rt > 0.5f) b |= kZR;
        if (lx < -0.5f) b |= kStickLLeft;
        if (lx > 0.5f) b |= kStickLRight;
        if (ly < -0.5f) b |= kStickLUp;
        if (ly > 0.5f) b |= kStickLDown;
        if (rx < -0.5f) b |= kStickRLeft;
        if (rx > 0.5f) b |= kStickRRight;
        if (ry < -0.5f) b |= kStickRUp;
        if (ry > 0.5f) b |= kStickRDown;
        g_axis_buttons = b;
        publish();
        return 1;
    }
    return 0;
}

void* game_main(void*) {
    int code = nier_run(g_cfg);
    __android_log_print(ANDROID_LOG_INFO, "NieR", "nier_run terminó: %d", code);
    return nullptr;
}

NnsdkWindow g_window_state;

void on_cmd(android_app* app, int32_t cmd) {
    if (cmd == APP_CMD_INIT_WINDOW && app->window) {
        g_window_state.window = app->window;
        if (!g_cfg.native_window) {  // la primera: arranca el juego
            g_cfg.native_window = app->window;
            g_window_state.serial = g_window_state.done = 1;
            pthread_t thread;
            pthread_create(&thread, nullptr, game_main, nullptr);
            pthread_detach(thread);
        } else {
            g_window_state.serial++;  // al volver de segundo plano: el presentador crea el swapchain en la nueva
        }
    } else if (cmd == APP_CMD_TERM_WINDOW && g_cfg.native_window) {
        // Android destruye la ventana en cuanto volvemos de aquí: el presentador tiene que soltarla antes (como mucho
        // 1,5 s, para no bloquear la app; presenta cada 33 ms).
        g_window_state.window = nullptr;
        const uint32_t serial = ++g_window_state.serial;
        for (int k = 0; k < 150 && g_window_state.done.load() != serial; k++) usleep(10'000);
    }
}

// Atiende a Android hasta que cierra la actividad.
[[noreturn]] void pump(android_app* app) {
    while (!app->destroyRequested) {
        android_poll_source* source = nullptr;
        if (ALooper_pollOnce(-1, nullptr, nullptr, reinterpret_cast<void**>(&source)) >= 0 && source)
            source->process(app, source);
    }
    exit(0);
}

}  // namespace

void android_main(android_app* app) {
    if (FILE* f = fopen((std::string(app->activity->internalDataPath) + "/carpeta.txt").c_str(), "r")) {
        char line[4096] = {};
        if (fgets(line, sizeof line, f)) g_data.assign(line, strcspn(line, "\r\n"));
        fclose(f);
    }
    if (g_data.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "NieR", "sin carpeta del juego: abre la app desde su icono");
        ANativeActivity_finish(app->activity);
        pump(app);
    }
    setenv("NIER_DATA", g_data.c_str(), 1);
    setenv("NIER_TRAZA_DESDE", "94", 1);  // DEPURACIÓN: también vuelca los shaders de los draws trazados desde aquí
    setenv("NIER_VOLCAR_TRAS", "", 1);  // DEPURACIÓN: RT tras esos draws
    setenv("NIER_VOLCAR_TAM", "", 1);  // DEPURACIÓN: RT tras los draws a ese tamaño
    setenv("NIER_AUTOJUEGO", "", 1);  // DEPURACIÓN: "150": disparo y zigzag automáticos desde el segundo 150
    setenv("NIER_TEX_SUST", "", 1);  // DEPURACIÓN
    setenv("NIER_VOLCAR_TEX", "", 1);  // DEPURACIÓN
    setenv("NIER_TRAZA_LIGERA", "1", 1);
    setenv("NIER_SONDA_MEMORIA", "", 1);  // DEPURACIÓN: "1": mide los límites del driver al arrancar  // DEPURACIÓN: trazas sin volcar render targets ni shaders
    std::string log = g_data + "/log.txt";
    if (freopen(log.c_str(), "w", stdout)) {
        // Por bloques: línea a línea, cada printf era una escritura en la SD. <datos>/registro o <datos>/log_lineas: línea
        // a línea (para leerlo en vivo, y lo último no se pierde si el proceso se cae).
        static char buffer[1 << 16];
        const bool verbose = access((g_data + "/registro").c_str(), F_OK) == 0 ||
                             access((g_data + "/log_lineas").c_str(), F_OK) == 0;
        setvbuf(stdout, verbose ? nullptr : buffer, verbose ? _IOLBF : _IOFBF, verbose ? 0 : sizeof buffer);
        const bool direct = access((g_data + "/log_directo").c_str(), F_OK) == 0;
        const bool piped = !direct && start_log_pipe();
        dup2(fileno(stdout), STDERR_FILENO);
        __android_log_print(ANDROID_LOG_INFO, "NieR", "log en %s (%s)", log.c_str(), piped ? "por tubería" : "directo");
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "NieR", "no puedo escribir %s: ¿falta el permiso de archivos?",
                            log.c_str());
    }
    g_cfg.exefs = rutas::get().exefs;
    g_cfg.nnsdk = "libnnsdk.so";
    // DEPURACIÓN: el vigilante interrumpe el hilo del juego cada 15 s para volcar su pila (un tirón y más de 100 líneas en
    // la SD cada vez): solo con <datos>/registro.
    g_cfg.watchdog_seconds = access((g_data + "/registro").c_str(), F_OK) == 0 ? 15 : 0;
    g_cfg.input = &g_input;
    g_cfg.window_state = &g_window_state;
    if (const char* a = getenv("NIER_AUTOJUEGO"); a && *a) {
        pthread_t thread;
        pthread_create(&thread, nullptr, autoplay, nullptr);
        pthread_detach(thread);
    }
    g_recording = access((g_data + "/grabar").c_str(), F_OK) == 0;
    g_replaying = access((g_data + "/reproducir").c_str(), F_OK) == 0;
    if (g_recording && !g_replaying) {
        pthread_t thread;
        pthread_create(&thread, nullptr, recorder, nullptr);
        pthread_detach(thread);
        printf("[entrada] grabando el mando en <datos>/grabacion.txt\n");
    }
    if (g_replaying) {
        pthread_t thread;
        pthread_create(&thread, nullptr, replayer, nullptr);
        pthread_detach(thread);
    }
    {
        pthread_t thread;
        pthread_create(&thread, nullptr, walk, nullptr);
        pthread_detach(thread);
    }

    // Como cualquier juego: la pantalla no se apaga mientras corre.
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
    app->onAppCmd = on_cmd;
    app->onInputEvent = on_input;
    pump(app);
}
