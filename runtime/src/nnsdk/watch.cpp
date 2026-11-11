// Vigilancia de escrituras (ver watch.h). Cada página de 4 KB del memfd del heap lleva una versión y su estado:
// vigilada (alguien la copió y la quiere ver cambiar) y abierta (vigilada pero escribible: pudo cambiar sin saltar).
// Una escritura en una página vigilada y cerrada salta aquí: se abre la página en esa dirección y sube su versión.
// collect() vuelve a cerrar las abiertas (en todas sus direcciones conocidas) antes de que el executor las copie.
// Un cerrojo de espera activa ordena el manejador de la señal y collect(); nadie lo tiene mucho rato ni espera dentro.

#include "nnsdk/watch.h"

#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "nnsdk/nvn/flags.h"

namespace nnsdk::watch {
namespace {

constexpr uintptr_t kPage = 4096;
constexpr uint8_t kWatched = 1, kOpen = 2, kAliased = 4;  // kAliased: algún pool virtual la ve en otra dirección
constexpr uint64_t kZeroKey = 0xFFFFFFFF00000000ull;     // página de un pool virtual sin nada mapeado

uintptr_t g_map = 0;
size_t g_size = 0;
std::atomic<uint32_t>* g_version = nullptr;
std::atomic<uint8_t>* g_state = nullptr;

struct VirtualRange {
    uintptr_t base = 0;
    size_t size = 0;
    std::atomic<uint32_t>* pages = nullptr;  // página del heap + 1 que se ve en cada página (0: ceros)
};
constexpr int kMaxVirtual = 8;
VirtualRange g_virtual[kMaxVirtual];
std::atomic<int> g_virtual_count{0};

std::atomic_flag g_lock = ATOMIC_FLAG_INIT;
struct Lock {
    Lock() {
        while (g_lock.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~Lock() { g_lock.clear(std::memory_order_release); }
};

std::atomic<uint64_t> g_faults{0}, g_protects{0}, g_protect_ns{0};
bool g_enabled = false;
bool g_handler = false;
struct sigaction g_old_segv{};

template <typename T>
T* table(size_t count) {
    void* p = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<T*>(p);
}

// La página del heap que se ve en a: >= 0, -2 si es un hueco de un pool virtual (ceros), -1 si no es del juego.
int64_t heap_page(uintptr_t a, bool* is_virtual = nullptr) {
    if (is_virtual) *is_virtual = false;
    if (a >= g_map && a < g_map + g_size) return static_cast<int64_t>((a - g_map) / kPage);
    const int n = g_virtual_count.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const VirtualRange& v = g_virtual[i];
        if (a < v.base || a >= v.base + v.size) continue;
        if (is_virtual) *is_virtual = true;
        const uint32_t e = v.pages[(a - v.base) / kPage].load(std::memory_order_relaxed);
        return e ? static_cast<int64_t>(e) - 1 : -2;
    }
    return -1;
}

uintptr_t heap_address(int64_t page) { return g_map + static_cast<uintptr_t>(page) * kPage; }

// Una escritura en memoria vigilada: se abre esa página (en esa dirección) y cuenta como cambio.
bool on_fault(uintptr_t a) {
    const int64_t h = heap_page(a);
    if (h < 0) return false;
    Lock lock;
    const uint8_t s = g_state[h].load(std::memory_order_relaxed);
    if (!(s & kWatched)) return false;
    if (mprotect(reinterpret_cast<void*>(a & ~(kPage - 1)), kPage, PROT_READ | PROT_WRITE) != 0) return false;
    g_version[h].fetch_add(1, std::memory_order_relaxed);
    g_state[h].store(s | kOpen, std::memory_order_relaxed);
    g_faults.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void on_segv(int sig, siginfo_t* info, void* ctx) {
    // El kernel de la Flip da SEGV_MAPERR (no ACCERR) al escribir en una página compartida de solo lectura: vale
    // cualquiera de los dos si la página está vigilada.
    if (info && (info->si_code == SEGV_ACCERR || info->si_code == SEGV_MAPERR) &&
        on_fault(reinterpret_cast<uintptr_t>(info->si_addr)))
        return;
    if (g_old_segv.sa_flags & SA_SIGINFO) {
        if (g_old_segv.sa_sigaction) g_old_segv.sa_sigaction(sig, info, ctx);
    } else if (g_old_segv.sa_handler == SIG_DFL) {
        signal(sig, SIG_DFL);
        raise(sig);
    } else if (g_old_segv.sa_handler != SIG_IGN) {
        g_old_segv.sa_handler(sig);
    }
}

// La primera vez que hace falta (con el manejador de crashes del loader ya puesto, que queda detrás).
void install_handler() {
    if (g_handler) return;
    g_handler = true;
    struct sigaction sa{};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_old_segv);
}

// Tramos de direcciones que cerrar, juntando los seguidos.
void add_run(std::vector<std::pair<uintptr_t, size_t>>& runs, uintptr_t a, size_t len = kPage) {
    if (!runs.empty() && runs.back().first + runs.back().second == a)
        runs.back().second += len;
    else
        runs.push_back({a, len});
}

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// Cierra (solo lectura) los tramos y cuenta llamadas y tiempo (perfil de cargas del executor).
void protect_runs(const std::vector<std::pair<uintptr_t, size_t>>& runs) {
    if (runs.empty()) return;
    const uint64_t t0 = now_ns();
    for (auto [a, len] : runs) mprotect(reinterpret_cast<void*>(a), len, PROT_READ);
    g_protect_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    g_protects.fetch_add(runs.size(), std::memory_order_relaxed);
}

}  // namespace

void init_heap(uintptr_t map, size_t size) {
    const char* data = getenv("NIER_DATA");
    g_enabled = access((std::string(data ? data : ".") + "/sin_vigilancia").c_str(), F_OK) != 0;
    if (!g_enabled) return;
    g_version = table<std::atomic<uint32_t>>(size / kPage);
    g_state = table<std::atomic<uint8_t>>(size / kPage);
    if (!g_version || !g_state) {
        g_enabled = false;
        return;
    }
    g_map = map;
    g_size = size;
}

void heap_changed(uintptr_t address, size_t size) {
    if (!g_enabled) return;
    Lock lock;
    for (uintptr_t a = address & ~(kPage - 1); a < address + size; a += kPage)
        if (const int64_t h = heap_page(a); h >= 0) g_version[h].fetch_add(1, std::memory_order_relaxed);
}

void heap_forget(uintptr_t address, size_t size) {
    if (!g_enabled) return;
    Lock lock;
    for (uintptr_t a = address & ~(kPage - 1); a < address + size; a += kPage)
        if (const int64_t h = heap_page(a); h >= 0) {
            g_state[h].store(0, std::memory_order_relaxed);
            g_version[h].fetch_add(1, std::memory_order_relaxed);
        }
}

void add_virtual_range(uintptr_t base, size_t size) {
    if (!g_enabled) return;
    Lock lock;
    const int n = g_virtual_count.load(std::memory_order_relaxed);
    if (n >= kMaxVirtual) return;
    auto* pages = table<std::atomic<uint32_t>>(size / kPage);
    if (!pages) return;
    g_virtual[n] = {base, size, pages};
    g_virtual_count.store(n + 1, std::memory_order_release);
}

void virtual_mapped(uintptr_t address, size_t size, uintptr_t heap) {
    if (!g_enabled) return;
    Lock lock;
    const int n = g_virtual_count.load(std::memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        VirtualRange& v = g_virtual[i];
        if (address < v.base || address + size > v.base + v.size) continue;
        std::vector<std::pair<uintptr_t, size_t>> close;  // las seguidas, en un solo mprotect
        for (uintptr_t off = 0; off < size; off += kPage) {
            const int64_t h = heap ? heap_page(heap + off) : -1;
            v.pages[(address + off - v.base) / kPage].store(h >= 0 ? static_cast<uint32_t>(h + 1) : 0, std::memory_order_relaxed);
            if (h < 0) continue;
            const uint8_t s = g_state[h].load(std::memory_order_relaxed);
            g_state[h].store(s | kAliased, std::memory_order_relaxed);
            // El mmap nuevo es escribible: si la página está vigilada y cerrada, aquí también se cierra.
            if ((s & kWatched) && !(s & kOpen)) add_run(close, address + off);
        }
        protect_runs(close);
        return;
    }
}

void before_kernel_write(void* p, size_t size) {
    if (!g_enabled || !g_handler || !size) return;
    const uintptr_t start = reinterpret_cast<uintptr_t>(p) & ~(kPage - 1), end = reinterpret_cast<uintptr_t>(p) + size;
    Lock lock;
    for (uintptr_t a = start; a < end; a += kPage) {
        const int64_t h = heap_page(a);
        if (h < 0) continue;
        const uint8_t s = g_state[h].load(std::memory_order_relaxed);
        if (!(s & kWatched)) continue;
        mprotect(reinterpret_cast<void*>(a), kPage, PROT_READ | PROT_WRITE);
        g_version[h].fetch_add(1, std::memory_order_relaxed);
        g_state[h].store(s | kOpen, std::memory_order_relaxed);
    }
}

bool watchable(uintptr_t address, size_t size) {
    if (!g_enabled || !size) return false;
    bool v0 = false, v1 = false;
    // El rango entero en el heap o entero en un mismo pool virtual (las páginas sin mapear valen: se leen como ceros).
    if (heap_page(address, &v0) == -1 || heap_page(address + size - 1, &v1) == -1 || v0 != v1) return false;
    return !v0 || [&] {
        for (int i = 0; i < g_virtual_count.load(std::memory_order_acquire); i++)
            if (address >= g_virtual[i].base && address + size <= g_virtual[i].base + g_virtual[i].size) return true;
        return false;
    }();
}

void collect(uintptr_t address, size_t size, std::vector<uint64_t>& keys, std::vector<std::pair<size_t, size_t>>* runs) {
    install_handler();
    const size_t n = ((address & (kPage - 1)) + size + kPage - 1) / kPage;
    if (keys.size() != n) keys.assign(n, ~0ull);
    // Direcciones que cerrar: las del rango y, aparte, las del heap que un pool virtual ve en él. Antes iban en una sola
    // lista, alternadas (página del pool, su página del heap, la siguiente del pool…), y add_run nunca juntaba nada: un
    // espejo nuevo de 24 MB en un pool virtual eran 12288 mprotect de 4 KB, con el cerrojo tomado (el manejador de la
    // señal de cualquier hilo que escribiera en memoria vigilada esperaba detrás). Perfil del 4-11: 21700 en el envío que
    // carga el selector de personaje ("724.1 tramos cerrados por frame"), 10800 en la aldea y 60600 al entrar en la
    // misión. Ahora el rango va seguido (un tramo si no tiene huecos) y lo del heap se ordena y se junta (16 páginas
    // seguidas por tile de 64 KB como poco). <datos>/vigilancia_por_pagina: como antes.
    const bool per_page = NVN_FLAG("vigilancia_por_pagina");
    std::vector<std::pair<uintptr_t, size_t>> close, close_heap;
    {
        Lock lock;
        for (size_t p = 0; p < n; p++) {
            const uintptr_t a = (address & ~(kPage - 1)) + p * kPage;
            bool is_virtual = false;
            const int64_t h = heap_page(a, &is_virtual);
            uint64_t key = kZeroKey;
            bool always = false;
            if (h >= 0) {
                const uint8_t s = g_state[h].load(std::memory_order_relaxed);
                if (!(s & kWatched) || (s & kOpen)) {
                    g_state[h].store((s | kWatched) & ~kOpen, std::memory_order_relaxed);
                    g_version[h].fetch_add(1, std::memory_order_relaxed);
                    add_run(close, a);
                    if (is_virtual) add_run(per_page ? close : close_heap, heap_address(h));
                }
                // En el heap y con alias: la otra dirección no se cierra desde aquí (no se sabe cuál es).
                always = !is_virtual && (s & kAliased);
                key = static_cast<uint64_t>(h) << 32 | g_version[h].load(std::memory_order_relaxed);
            }
            if (keys[p] == key && !always) continue;
            keys[p] = key;
            const size_t off = p * kPage - (address & (kPage - 1));
            if (runs) {
                const size_t begin = p ? off : 0, len = std::min<size_t>(size, off + kPage) - begin;
                if (!runs->empty() && runs->back().first + runs->back().second == begin)
                    runs->back().second += len;
                else
                    runs->push_back({begin, len});
            }
        }
        if (!close_heap.empty()) {
            std::sort(close_heap.begin(), close_heap.end());
            for (auto [a, len] : close_heap) add_run(close, a, len);  // tras las del rango
        }
        protect_runs(close);
    }
}

bool fully_mapped(uintptr_t address, size_t size) {
    if (!g_enabled || !size) return true;
    bool is_virtual = false;
    if (heap_page(address, &is_virtual) == -1 || !is_virtual) return true;
    for (uintptr_t a = address & ~(kPage - 1); a < address + size; a += kPage)
        if (heap_page(a) < 0) return false;
    return true;
}

uint64_t faults() { return g_faults.load(std::memory_order_relaxed); }
uint64_t protects() { return g_protects.load(std::memory_order_relaxed); }
uint64_t protect_ns() { return g_protect_ns.load(std::memory_order_relaxed); }

}  // namespace nnsdk::watch
