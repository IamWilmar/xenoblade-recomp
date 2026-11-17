#pragma once
// Los SystemEvent de nn::os que usan audio.cpp y audio_renderer.cpp, para las pruebas fuera del dispositivo: como en
// os.cpp, cada evento del juego se busca por su dirección; aquí con un mapa y una variable de condición.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>

#include "nnsdk/os.h"

namespace falsos {

struct Evento {
    std::mutex m;
    std::condition_variable cv;
    bool senalado = false, auto_clear = false;
    std::atomic<long> senales{0};
};

inline std::mutex g_eventos_m;
inline std::map<const void*, std::shared_ptr<Evento>> g_eventos;
inline std::atomic<long> g_creados_al_senalar{0};  // signal_system_event sobre un evento que no existía

inline std::shared_ptr<Evento> evento(const void* e, bool crear) {
    std::lock_guard lock(g_eventos_m);
    auto it = g_eventos.find(e);
    if (it != g_eventos.end()) return it->second;
    if (!crear) return nullptr;
    return g_eventos[e] = std::make_shared<Evento>();
}

inline void senalar(Evento& ev) {
    {
        std::lock_guard lock(ev.m);
        ev.senalado = true;
        ev.senales++;
    }
    ev.cv.notify_all();
}

// WaitSystemEvent (ms < 0), TimedWaitSystemEvent o TryWaitSystemEvent (ms == 0).
inline bool esperar(const void* e, int ms) {
    auto ev = evento(e, true);
    std::unique_lock lock(ev->m);
    if (ms < 0)
        ev->cv.wait(lock, [&] { return ev->senalado; });
    else if (!ev->cv.wait_for(lock, std::chrono::milliseconds(ms), [&] { return ev->senalado; }))
        return false;
    if (ev->auto_clear) ev->senalado = false;
    return true;
}

inline void destruir(const void* e) {  // DestroySystemEvent
    std::lock_guard lock(g_eventos_m);
    g_eventos.erase(e);
}
inline bool existe(const void* e) { return evento(e, false) != nullptr; }
inline long senales(const void* e) {
    auto ev = evento(e, false);
    return ev ? ev->senales.load() : -1;
}

}  // namespace falsos

inline void nnsdk::create_system_event(nn::os::SystemEventType* event, bool auto_clear, bool signaled) {
    auto ev = std::make_shared<falsos::Evento>();
    ev->auto_clear = auto_clear;
    ev->senalado = signaled;
    std::lock_guard lock(falsos::g_eventos_m);
    falsos::g_eventos[event] = ev;
}
inline void nnsdk::signal_system_event(nn::os::SystemEventType* event) {
    if (!falsos::existe(event)) falsos::g_creados_al_senalar++;
    falsos::senalar(*falsos::evento(event, true));
}
inline bool nnsdk::signal_system_event_if_alive(nn::os::SystemEventType* event) {
    auto ev = falsos::evento(event, false);
    if (!ev) return false;
    falsos::senalar(*ev);
    return true;
}
inline bool nnsdk::clear_system_event_if_alive(nn::os::SystemEventType* event) {
    auto ev = falsos::evento(event, false);
    if (!ev) return false;
    std::lock_guard lock(ev->m);
    ev->senalado = false;
    return true;
}
