#pragma once
// Interruptores en vivo: un fichero vacío en la carpeta de datos (NIER_DATA) cambia un comportamiento sin
// reinstalar ni reiniciar el juego (`adb shell touch <datos>/<nombre>`, `rm` para volver). Sirven para
// comparar una optimización con y sin ella en la misma escena, y para apagarla si algo sale mal.
// Cada interruptor mira su fichero al usarse la primera vez; después los mira todos un hilo propio ("Interruptores")
// una vez por segundo, con un solo readdir de la carpeta de datos, y anota los cambios en el log. Antes cada uno hacía
// su access() en el hilo que lo leía: ~75 búsquedas por segundo en la SD (FUSE → MediaProvider → exFAT) desde el
// executor, el hilo del juego y el del audio, de 0,2-3 ms cada una y paradas del todo cuando la SD se atasca.

#include <dirent.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "nnsdk/nvn/nvn.h"

namespace nvn {

class LiveFlag {
public:
    explicit LiveFlag(const char* name);
    bool get() const { return value_.load(std::memory_order_relaxed); }
    const char* name() const { return name_; }
    // Desde el hilo de los interruptores.
    void set(bool v) {
        if (v != value_.exchange(v, std::memory_order_relaxed)) NVN_LOG("interruptor %s: %s\n", name_, v ? "sí" : "no");
    }

private:
    const char* name_;
    std::atomic<bool> value_{false};
};

// Los interruptores creados y el hilo que los mira. Nunca se destruye: el hilo sigue hasta que muere el proceso.
class LiveFlags {
public:
    static LiveFlags& instance() {
        static LiveFlags* flags = new LiveFlags;
        return *flags;
    }
    void add(LiveFlag* flag) {
        std::lock_guard lock(mutex_);
        flags_.push_back(flag);
        if (!started_) {
            started_ = true;
            std::thread([this] { run(); }).detach();
        }
    }

private:
    LiveFlags() = default;
    void run() {
#if defined(__ANDROID__) || defined(__linux__)
        pthread_setname_np(pthread_self(), "Interruptores");
#endif
        const char* env = getenv("NIER_DATA");
        const std::string data = env ? env : ".";
        std::unordered_set<std::string> present;
        std::vector<LiveFlag*> flags;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            DIR* dir = opendir(data.c_str());
            if (!dir) continue;  // la SD no responde: se quedan como estaban
            present.clear();
            while (const dirent* e = readdir(dir)) present.insert(e->d_name);
            closedir(dir);
            {
                std::lock_guard lock(mutex_);
                flags = flags_;
            }
            for (LiveFlag* f : flags) f->set(present.count(f->name()) != 0);
        }
    }

    std::mutex mutex_;
    std::vector<LiveFlag*> flags_;
    bool started_ = false;
};

inline LiveFlag::LiveFlag(const char* name) : name_(name) {
    const char* data = getenv("NIER_DATA");
    const bool v = access((std::string(data ? data : ".") + "/" + name_).c_str(), F_OK) == 0;
    value_.store(v, std::memory_order_relaxed);
    NVN_LOG("interruptor %s: %s\n", name_, v ? "sí" : "no");
    LiveFlags::instance().add(this);
}

}  // namespace nvn

// true si existe <NIER_DATA>/<nombre>; `nombre` es un literal (un nombre de fichero, sin '/'). Leerlo es una carga
// atómica: vale en cualquier camino, también por draw.
#define NVN_FLAG(nombre)                     \
    ([]() -> bool {                          \
        static ::nvn::LiveFlag flag(nombre); \
        return flag.get();                   \
    }())
