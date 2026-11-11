#pragma once
// Infraestructura de la capa NVN: cada archivo registra sus funciones con NVN_PROC y el juego
// las obtiene por nombre (nvnBootstrapLoader / nvnDeviceGetProcAddress).
//
// No hay cabeceras ni driver de referencia (el driver no viene en el volcado): los objetos NVN
// del juego son opacos, se asocian por dirección a objetos nuestros, y los enums y tamaños se
// deducen del uso que hace NieR. Cada valor deducido lleva un comentario con la evidencia.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace nvn {

using Boolean = int;  // NVNboolean: 1 = éxito

void register_proc(const char* name, void* fn);
void* find_proc(const char* name);

struct ProcRegistrar {
    ProcRegistrar(const char* name, void* fn) { register_proc(name, fn); }
};

// Objeto del juego (por dirección) -> objeto nuestro.
template <typename T>
class Objects {
public:
    T* get(const void* key) {
        std::lock_guard lock(m_);
        auto it = map_.find(key);
        return it == map_.end() ? nullptr : it->second.get();
    }
    T& create(const void* key) {
        std::lock_guard lock(m_);
        auto& slot = map_[key];
        slot = std::make_unique<T>();
        generation_.fetch_add(1, std::memory_order_release);
        return *slot;
    }
    void destroy(const void* key) {
        std::lock_guard lock(m_);
        map_.erase(key);
        generation_.fetch_add(1, std::memory_order_release);
    }
    // Lo quita del registro y devuelve la propiedad (para quien deba mantenerlo vivo).
    std::unique_ptr<T> take(const void* key) {
        std::lock_guard lock(m_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        std::unique_ptr<T> out = std::move(it->second);
        map_.erase(it);
        generation_.fetch_add(1, std::memory_order_release);
        return out;
    }
    // Sube con cada create, destroy y take: quien recuerde lo que encontró con find_if (el ejecutor, el memory pool de una
    // dirección) sabe así si sigue valiendo.
    uint64_t generation() const { return generation_.load(std::memory_order_acquire); }
    // Primer objeto que cumple pred(clave, objeto); nullptr si ninguno.
    template <typename Pred>
    T* find_if(Pred pred) {
        std::lock_guard lock(m_);
        for (auto& [key, value] : map_)
            if (pred(key, *value)) return value.get();
        return nullptr;
    }

private:
    std::mutex m_;
    std::unordered_map<const void*, std::unique_ptr<T>> map_;
    std::atomic<uint64_t> generation_{0};
};

// Modos globales del dispositivo (device.cpp). NVNwindowOriginMode: 0 LOWER_LEFT, 1 UPPER_LEFT.
// NVNdepthMode: 0 NEAR_IS_MINUS_W (estilo OpenGL), 1 NEAR_IS_ZERO.
extern int g_window_origin_mode;
extern int g_depth_mode;

}  // namespace nvn

#define NVN_PROC(fn) static ::nvn::ProcRegistrar nvn_proc_##fn(#fn, reinterpret_cast<void*>(&fn))

#define NVN_LOG(...) printf("[nvn] " __VA_ARGS__)

namespace nvn {
// "Solo las n primeras veces": el contador se queda en n. Un int que sigue sumando se desborda tras 2^31 llamadas, se
// vuelve negativo y escribe en cada una (70 GB de log en la SD en una sesión larga de Xenoblade, a 9 fps).
inline bool first_times(int& count, int n) {
    if (count >= n) return false;
    ++count;
    return true;
}
inline bool first_times(std::atomic<int>& count, int n) {
    int c = count.load(std::memory_order_relaxed);
    while (c < n)
        if (count.compare_exchange_weak(c, c + 1, std::memory_order_relaxed)) return true;
    return false;
}
// Las n primeras veces y luego una de cada `every` (cuenta en 64 bits: no se desborda).
inline bool first_then_every(long long& count, int n, int every) {
    ++count;
    return count <= n || count % every == 0;
}
}  // namespace nvn
