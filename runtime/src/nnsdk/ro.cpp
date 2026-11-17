// nn::ro: módulos NRO que el juego carga en marcha (Unity carga así sus plugins nativos de Plugins/*.nro; en Disco
// Elysium, AmplifyTextureNative). La carga real (copiar, relocar contra main y libnnsdk, constructores) la hace el
// loader (host().load_nro); aquí solo se lleva qué nn::ro::Module del juego es cada módulo cargado.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::ro {

struct Module;
struct RegistrationInfo;

}  // namespace nn::ro

namespace {

constexpr nn::Result kNotLoaded{22 | 1001 << 9};  // módulo 22 (ro)

// nn::ro::Module del SDK: el juego lo lee y lo COPIA (Unity guarda una copia del struct y busca los símbolos con
// ella), así que el módulo cargado va dentro del struct, como en la consola, y no en una tabla por dirección.
struct ModuleView {
    void* module;            // +0x00: el del loader
    uint32_t state;          // +0x08: 1 cargado (Unity lo comprueba tras LoadModule; con 0 da la carga por fallida)
    uint32_t pad;
    const void* file;        // +0x10: la imagen del NRO
    void* buffer;            // +0x18: el buffer que dio el juego
};
constexpr uint32_t kStateLoaded = 1;

std::mutex g_m;
std::map<void*, int> g_live;              // módulos del loader vivos (para validar lo que viene en el struct)

void* loaded_of(const nn::ro::Module* m) {
    if (!m) return nullptr;
    const auto* v = reinterpret_cast<const ModuleView*>(m);
    return v->state == kStateLoaded && g_live.count(v->module) ? v->module : nullptr;
}

uint32_t rd32(const void* p, size_t off) {
    uint32_t v;
    memcpy(&v, static_cast<const uint8_t*>(p) + off, sizeof v);
    return v;
}

}  // namespace

namespace nn::ro {

Result Initialize() { return ResultSuccess; }

// El loader pone el módulo en memoria propia (bss incluido): el buffer del juego no hace falta, pero se pide lo que
// pediría el SDK (el .bss, en páginas) para que el juego reserve lo mismo que en la consola.
Result GetBufferSize(unsigned long* out, const void* image) {
    if (out) *out = image ? (rd32(image, 0x38) + 0xFFF) & ~0xFFFu : 0;
    return ResultSuccess;
}

Result RegisterModuleInfo(RegistrationInfo*, const void*) { return ResultSuccess; }
Result UnregisterModuleInfo(RegistrationInfo*) { return ResultSuccess; }
// Protege la RELRO de un módulo ya cargado: los NRO del loader ya quedan con sus permisos (protect en run.cpp).
Result ProtectRelro(const void*, const void*, const void*, const void*, const void*) { return ResultSuccess; }

void trace(const char* what, const void* module, const char* name) {
    char bt[200];
    int n = 0;
    const auto& h = nnsdk::host();
    const auto base = reinterpret_cast<uintptr_t>(h.main_base);
    auto* fp = static_cast<uintptr_t*>(__builtin_frame_address(0));
    for (int i = 0; i < 10 && fp && n < 180; i++) {
        const uintptr_t ret = fp[1];
        if (ret >= base && ret < base + h.main_size) n += snprintf(bt + n, sizeof bt - n, " %llx", static_cast<unsigned long long>(ret - base));
        auto* next = reinterpret_cast<uintptr_t*>(fp[0]);
        if (next <= fp || reinterpret_cast<uintptr_t>(next) - reinterpret_cast<uintptr_t>(fp) > (8u << 20)) break;
        fp = next;
    }
    bt[n] = 0;
    printf("[ro] %s %p %s; pila%s\n", what, module, name ? name : "", bt);
}

Result LoadModule(Module* module, const void* image, void* buffer, unsigned long, int) {
    trace("LoadModule", module, nullptr);
    if (!image || !module || !nnsdk::host().load_nro) return kNotLoaded;
    void* loaded = nnsdk::host().load_nro(image, rd32(image, 0x18));
    if (!loaded) return kNotLoaded;
    std::lock_guard lock(g_m);
    g_live[loaded]++;
    auto* v = reinterpret_cast<ModuleView*>(module);
    v->module = loaded;
    v->state = kStateLoaded;
    v->file = image;
    v->buffer = buffer;
    return ResultSuccess;
}

// Se deja en memoria: el plugin puede tener hilos o punteros vivos y descargarlo no ahorra nada que importe.
void UnloadModule(Module* module) {
    trace("UnloadModule", module, nullptr);
    std::lock_guard lock(g_m);
    if (module) reinterpret_cast<ModuleView*>(module)->state = 0;
}

Result LookupModuleSymbol(unsigned long* out, const Module* module, const char* name) {
    static int traced = 0;
    if (traced++ < 60) trace("LookupModuleSymbol", module, name);
    std::lock_guard lock(g_m);
    void* loaded = loaded_of(module);
    const uint64_t addr = loaded ? nnsdk::host().nro_symbol(loaded, name) : 0;
    if (out) *out = addr;
    if (!addr) printf("[ro] %s no está en el módulo\n", name ? name : "(null)");
    return addr ? ResultSuccess : kNotLoaded;
}

Result LookupSymbol(unsigned long* out, const char* name) {
    std::lock_guard lock(g_m);
    for (const auto& [loaded, n] : g_live)
        if (uint64_t addr = nnsdk::host().nro_symbol(loaded, name)) {
            if (out) *out = addr;
            return ResultSuccess;
        }
    if (out) *out = 0;
    return kNotLoaded;
}

}  // namespace nn::ro
