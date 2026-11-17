// nn::mem::StandardAllocator y el ExpHeap de nn::lmem.
//
// StandardAllocator va sobre el malloc de bionic. El juego le da una región para gestionar (su malloc: 8 MB del
// heap); se ignora: le da igual de dónde salga la memoria mientras Free la acepte, y su free() no mira rangos, va
// directo a Free. Así tampoco hay tope de 8 MB.
//
// El ExpHeap, en cambio, sí reparte de verdad la región que le dan: el juego decide a quién devolver un puntero
// mirando si cae dentro del bloque que le dio al heap (main+0x6c6b598), así que tiene que salir de ahí. Los
// registros de huecos y bloques van fuera de la región, para no tocar sus páginas.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <vector>
#include <sys/mman.h>
#include <memory>
#include <unordered_map>

#include "nnsdk/runtime.h"

namespace nn::mem {

class StandardAllocator {
public:
    StandardAllocator();
    void Initialize(void* address, size_t size);
    void Initialize(void* address, size_t size, bool enable_cache);
    void Finalize();
    void* Allocate(size_t size);
    void* Allocate(size_t size, size_t alignment);
    void* Reallocate(void* p, size_t size);
    void Free(void* p);
    size_t GetSizeOf(const void* p) const;
    size_t GetTotalFreeSize() const;
    size_t GetAllocatableSize() const;
    void Dump() const;
};

namespace {
// Tamaño de la región que dio cada allocator al iniciarse (lo pocos que hay): lo que "queda libre" para el juego. El
// malloc de bionic no lleva la cuenta por allocator, así que se dice la región entera.
std::mutex g_regions_m;
std::unordered_map<const void*, size_t> g_regions;

// Lo que el juego tiene pedido a sus StandardAllocator (su malloc entero va por aquí). En la Switch no pasa de la
// región que le dio (todo el heap); aquí se apunta para el log y para GetAllocatableSize.
std::atomic<int64_t> g_used{0};
std::atomic<int64_t> g_logged_mark{0};

uint64_t main_offset(const void* lr) {
    const auto& h = nnsdk::host();
    const auto a = reinterpret_cast<uintptr_t>(lr), base = reinterpret_cast<uintptr_t>(h.main_base);
    return a >= base && a < base + h.main_size ? a - base : a;
}

void account(void* p, ptrdiff_t old_size, size_t asked, const void* lr) {
    const int64_t now = g_used.fetch_add(static_cast<int64_t>(p ? malloc_usable_size(p) : 0) - old_size) +
                        static_cast<int64_t>(p ? malloc_usable_size(p) : 0) - old_size;
    if (asked >= (16u << 20)) {
        // Pila por los frame pointers (el juego los guarda): dónde se pidió de verdad.
        char bt[256];
        int n = 0;
        const auto& h = nnsdk::host();
        const auto base = reinterpret_cast<uintptr_t>(h.main_base);
        auto* fp = static_cast<uintptr_t*>(__builtin_frame_address(0));
        for (int i = 0; i < 12 && fp && n < 230; i++) {
            const uintptr_t ret = fp[1];
            if (ret >= base && ret < base + h.main_size) n += snprintf(bt + n, sizeof bt - n, " %llx", static_cast<unsigned long long>(ret - base));
            auto* next = reinterpret_cast<uintptr_t*>(fp[0]);
            if (next <= fp || reinterpret_cast<uintptr_t>(next) - reinterpret_cast<uintptr_t>(fp) > (8u << 20)) break;
            fp = next;
        }
        bt[n] = 0;
        printf("[mem] el juego pide %zu MB (main+0x%llx); lleva %lld MB; pila%s\n", asked >> 20,
               static_cast<unsigned long long>(main_offset(lr)), static_cast<long long>(now >> 20), bt);
    }
    // Cada 256 MB más que el máximo anterior, una línea.
    int64_t mark = g_logged_mark.load();
    while (now >= mark + (256ll << 20)) {
        if (g_logged_mark.compare_exchange_weak(mark, now & ~((256ll << 20) - 1))) {
            printf("[mem] el juego lleva %lld MB en su malloc\n", static_cast<long long>(now >> 20));
            fflush(stdout);
            break;
        }
    }
}
}  // namespace

namespace {
// Bloques grandes en la región del allocator. Lo pequeño va al malloc de bionic (rápido y sin fragmentar la región),
// pero un bloque grande tiene que estar donde el juego lo puso: en su heap (el memfd de os_memory.cpp). Catherine: Full
// Body saca de su StandardAllocator los tres memory pools de NVN (96 MB, 1 GB y 512 MB) y escribe en sus texturas
// lineales directamente, sin avisar (su atlas de glifos); fuera del heap esa memoria no se puede vigilar
// (nnsdk::watch) y los glifos nuevos no llegaban a la GPU. Reparto simple por páginas: primer hueco que quepa, huecos
// vecinos fusionados al liberar, y lo liberado se devuelve al sistema (MADV_REMOVE sobre el memfd).
constexpr size_t kBigBlock = 1 << 20, kRegionPage = 4096;
struct Region {
    uintptr_t base = 0;
    size_t size = 0;
    std::mutex m;
    std::map<uintptr_t, size_t> free_ranges;     // inicio -> bytes
    std::unordered_map<uintptr_t, size_t> used;  // inicio -> bytes
};
std::mutex g_big_m;
std::unordered_map<const void*, std::unique_ptr<Region>> g_big;  // por allocator
std::vector<Region*> g_big_list;                                  // para buscar un puntero al liberar

Region* region_of_allocator(const void* self) {
    std::lock_guard lock(g_big_m);
    auto it = g_big.find(self);
    return it != g_big.end() ? it->second.get() : nullptr;
}
Region* region_of_pointer(const void* p) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    std::lock_guard lock(g_big_m);
    for (Region* r : g_big_list)
        if (a >= r->base && a < r->base + r->size) return r;
    return nullptr;
}
void add_region(const void* self, void* address, size_t size) {
    const uintptr_t begin = (reinterpret_cast<uintptr_t>(address) + kRegionPage - 1) & ~(kRegionPage - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(address) + size) & ~(kRegionPage - 1);
    if (!address || end <= begin || end - begin < (16u << 20)) return;  // regiones pequeñas: todo a bionic
    auto r = std::make_unique<Region>();
    r->base = begin;
    r->size = end - begin;
    r->free_ranges[begin] = end - begin;
    std::lock_guard lock(g_big_m);
    g_big_list.push_back(r.get());
    g_big[self] = std::move(r);
}
void* region_allocate(Region& r, size_t size, size_t alignment) {
    size = (size + kRegionPage - 1) & ~(kRegionPage - 1);
    alignment = std::max(alignment, kRegionPage);
    std::lock_guard lock(r.m);
    for (auto it = r.free_ranges.begin(); it != r.free_ranges.end(); ++it) {
        const uintptr_t start = it->first, end = start + it->second;
        const uintptr_t at = (start + alignment - 1) & ~(alignment - 1);
        if (at < start || at + size > end) continue;
        r.free_ranges.erase(it);
        if (at > start) r.free_ranges[start] = at - start;
        if (at + size < end) r.free_ranges[at + size] = end - (at + size);
        r.used[at] = size;
        return reinterpret_cast<void*>(at);
    }
    return nullptr;
}
// 0 si p no es un bloque de la región.
size_t region_size_of(Region& r, const void* p) {
    std::lock_guard lock(r.m);
    auto it = r.used.find(reinterpret_cast<uintptr_t>(p));
    return it != r.used.end() ? it->second : 0;
}
bool region_free(Region& r, void* p) {
    const auto at = reinterpret_cast<uintptr_t>(p);
    std::lock_guard lock(r.m);
    auto it = r.used.find(at);
    if (it == r.used.end()) return false;
    size_t size = it->second;
    r.used.erase(it);
    madvise(p, size, MADV_REMOVE);  // las páginas vuelven al sistema; al tocarlas otra vez salen a cero
    uintptr_t start = at;
    auto next = r.free_ranges.lower_bound(start);
    if (next != r.free_ranges.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) {
            start = prev->first;
            size += prev->second;
            r.free_ranges.erase(prev);
        }
    }
    if (next != r.free_ranges.end() && start + size == next->first) {
        size += next->second;
        r.free_ranges.erase(next);
    }
    r.free_ranges[start] = size;
    return true;
}
// La cuenta de account() para un bloque de la región (delta: bytes que suma o resta).
void big_account(int64_t delta, size_t asked, const void* lr) {
    const int64_t now = g_used.fetch_add(delta) + delta;
    if (asked >= (16u << 20))
        printf("[mem] el juego pide %zu MB (main+0x%llx), en su región; lleva %lld MB\n", asked >> 20,
               static_cast<unsigned long long>(main_offset(lr)), static_cast<long long>(now >> 20));
}
}  // namespace

// Sin miembros: el objeto vive en memoria del juego y no se toca.
StandardAllocator::StandardAllocator() {}
void StandardAllocator::Initialize(void* address, size_t size) {
    add_region(this, address, size);
    printf("[nnsdk] StandardAllocator %p: región %p +%zu MB (lo pequeño, al malloc de bionic; los bloques de 1 MB o más, "
           "en la región)\n",
           static_cast<void*>(this), address, size >> 20);
}
void StandardAllocator::Initialize(void* address, size_t size, bool) {
    {
        std::lock_guard lock(g_regions_m);
        g_regions[this] = size;
    }
    add_region(this, address, size);
    printf("[nnsdk] StandardAllocator %p: región %p +%zu KB (lo pequeño, al malloc de bionic; los bloques de 1 MB o más, "
           "en la región)\n",
           static_cast<void*>(this), address, size / 1024);
}
void StandardAllocator::Finalize() {}

void* StandardAllocator::Allocate(size_t size) { return Allocate(size, 16); }

void* StandardAllocator::Allocate(size_t size, size_t alignment) {
    void* p = nullptr;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    if (size >= kBigBlock)
        if (Region* r = region_of_allocator(this))
            if (void* big = region_allocate(*r, size, alignment)) {
                big_account(static_cast<int64_t>(region_size_of(*r, big)), size, __builtin_return_address(0));
                return big;
            }
    if (posix_memalign(&p, alignment, size ? size : 1) != 0) return nullptr;
    account(p, 0, size, __builtin_return_address(0));
    return p;
}

// Como el SDK: tamaño "negativo" -> nullptr sin tocar nada; sin puntero es Allocate (alineado a 16); tamaño 0 libera
// y devuelve nullptr; si no cabe, nullptr y el bloque sigue como estaba (realloc hace lo mismo).
void* StandardAllocator::Reallocate(void* p, size_t size) {
    if (static_cast<ptrdiff_t>(size) < 0) return nullptr;
    if (!p) return Allocate(size, 16);
    if (Region* r = region_of_pointer(p)) {  // bloque de la región: uno nuevo (donde toque por tamaño) y se copia
        const size_t had = region_size_of(*r, p);
        if (!had) return nullptr;
        if (size == 0) {
            Free(p);
            return nullptr;
        }
        if (size <= had && size >= kBigBlock) return p;
        void* q = Allocate(size, 16);
        if (!q) return nullptr;
        memcpy(q, p, std::min(had, size));
        Free(p);
        return q;
    }
    const ptrdiff_t old = static_cast<ptrdiff_t>(malloc_usable_size(p));
    if (size == 0) {
        free(p);
        account(nullptr, old, 0, nullptr);
        return nullptr;
    }
    void* q = realloc(p, (size + 15) & ~size_t{15});
    if (q) account(q, old, size, __builtin_return_address(0));
    return q;
}

void StandardAllocator::Free(void* p) {
    if (!p) return;
    if (Region* r = region_of_pointer(p)) {
        const size_t had = region_size_of(*r, p);
        if (region_free(*r, p)) big_account(-static_cast<int64_t>(had), 0, nullptr);
        return;
    }
    account(nullptr, static_cast<ptrdiff_t>(malloc_usable_size(p)), 0, nullptr);
    free(p);
}
size_t StandardAllocator::GetSizeOf(const void* p) const {
    if (!p) return 0;
    if (Region* r = region_of_pointer(p)) return region_size_of(*r, p);
    return malloc_usable_size(p);
}
size_t StandardAllocator::GetTotalFreeSize() const {
    std::lock_guard lock(g_regions_m);
    auto it = g_regions.find(this);
    return it != g_regions.end() ? it->second : 0;
}
size_t StandardAllocator::GetAllocatableSize() const {
    static std::atomic<int> logged{0};
    const size_t r = GetTotalFreeSize();
    if (logged.fetch_add(1) < 6)
        printf("[mem] GetAllocatableSize -> %zu MB (main+0x%llx)\n", r >> 20,
               static_cast<unsigned long long>(main_offset(__builtin_return_address(0))));
    return r;
}

// Dump: en la consola imprime el estado del heap por el log de desarrollo; aquí no hay nada propio que contar.
void StandardAllocator::Dump() const {}

// NumberLineAllocator: reparte números enteros en rangos [primero, primero+n) (Catherine: Full Body numera así sus
// descriptores de textura y sampler, 0..0xF00). Los rangos libres se fusionan al liberar. Free solo recibe el
// primero: el tamaño de cada reserva se guarda aparte. El estado vive aquí, indexado por el objeto del juego.
class NumberLineAllocator {
public:
    using AllocateFunction = void* (*)(size_t, void*);
    using FreeFunction = void (*)(void*, void*);
    void Initialize(AllocateFunction, void*, FreeFunction, void*, bool);
    void Finalize();
    bool AddRange(int first, int count);
    bool Allocate(int* out, int count);
    bool Free(int first);
};

namespace {
struct NumberLine {
    std::mutex m;
    std::map<int, int> free_ranges;  // primero -> cuántos
    std::unordered_map<int, int> used;  // primero -> cuántos
};
std::mutex g_number_lines_m;
std::unordered_map<const void*, std::unique_ptr<NumberLine>> g_number_lines;
NumberLine& number_line(const void* self) {
    std::lock_guard lock(g_number_lines_m);
    auto& p = g_number_lines[self];
    if (!p) p = std::make_unique<NumberLine>();
    return *p;
}
void add_free(NumberLine& n, int first, int count) {
    auto next = n.free_ranges.lower_bound(first);
    if (next != n.free_ranges.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == first) {
            prev->second += count;
            if (next != n.free_ranges.end() && prev->first + prev->second == next->first) {
                prev->second += next->second;
                n.free_ranges.erase(next);
            }
            return;
        }
    }
    if (next != n.free_ranges.end() && first + count == next->first) {
        count += next->second;
        n.free_ranges.erase(next);
    }
    n.free_ranges[first] = count;
}
}  // namespace

void NumberLineAllocator::Initialize(AllocateFunction, void*, FreeFunction, void*, bool) {
    NumberLine& n = number_line(this);
    std::lock_guard lock(n.m);
    n.free_ranges.clear();
    n.used.clear();
}
void NumberLineAllocator::Finalize() {
    std::lock_guard lock(g_number_lines_m);
    g_number_lines.erase(this);
}
bool NumberLineAllocator::AddRange(int first, int count) {
    if (count <= 0) return false;
    NumberLine& n = number_line(this);
    std::lock_guard lock(n.m);
    add_free(n, first, count);
    return true;
}
// El primer hueco que quepa (el SDK también reparte desde abajo).
bool NumberLineAllocator::Allocate(int* out, int count) {
    if (count <= 0) return false;
    NumberLine& n = number_line(this);
    std::lock_guard lock(n.m);
    for (auto it = n.free_ranges.begin(); it != n.free_ranges.end(); ++it) {
        if (it->second < count) continue;
        const int first = it->first, left = it->second - count;
        n.free_ranges.erase(it);
        if (left) n.free_ranges[first + count] = left;
        n.used[first] = count;
        *out = first;
        return true;
    }
    static std::atomic<int> logged{0};
    if (logged++ < 4) printf("[mem] NumberLineAllocator %p sin hueco para %d\n", static_cast<void*>(this), count);
    return false;
}
bool NumberLineAllocator::Free(int first) {
    NumberLine& n = number_line(this);
    std::lock_guard lock(n.m);
    auto it = n.used.find(first);
    if (it == n.used.end()) return false;
    add_free(n, first, it->second);
    n.used.erase(it);
    return true;
}

}  // namespace nn::mem

namespace nn::lmem {

namespace detail {
struct HeapHead;
}
using HeapHandle = detail::HeapHead*;

enum AllocationMode { AllocationMode_FirstFit = 0, AllocationMode_BestFit = 1 };

}  // namespace nn::lmem

namespace {

using nn::lmem::AllocationMode;

constexpr int kOptionZeroClear = 1;  // CreateExpHeap: las asignaciones salen a 0 (2 = DebugFill, 4 = ThreadSafe)

// El handle que recibe el juego es este objeto (para él, opaco). Siempre con cerrojo, pida ThreadSafe o no.
struct ExpHeap {
    std::mutex m;
    uintptr_t begin = 0, end = 0;
    int option = 0;
    AllocationMode mode = nn::lmem::AllocationMode_FirstFit;
    std::map<uintptr_t, size_t> free;               // huecos: dirección -> tamaño, sin contiguos
    std::unordered_map<uintptr_t, size_t> used;     // bloque dado -> tamaño

    void add_free(uintptr_t address, size_t size) {
        auto next = free.lower_bound(address);
        if (next != free.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == address) {
                address = prev->first;
                size += prev->second;
                free.erase(prev);
            }
        }
        if (next != free.end() && address + size == next->first) {
            size += next->second;
            free.erase(next);
        }
        free[address] = size;
    }

    // Alineación positiva: desde abajo (el principio del hueco); negativa: desde arriba (el final del hueco), como
    // nn::lmem. FirstFit toma el primer hueco que sirva en ese sentido; BestFit el que deje menos sobrante.
    void* allocate(size_t size, int alignment) {
        const bool from_top = alignment < 0;
        size_t align = static_cast<size_t>(from_top ? -static_cast<long>(alignment) : alignment);
        if (align < 8) align = 8;
        if (align & (align - 1)) return nullptr;
        size = (size ? size + 7 : 8) & ~size_t{7};
        std::lock_guard lock(m);
        auto best = free.end();
        uintptr_t best_at = 0;
        size_t best_left = SIZE_MAX;
        auto consider = [&](std::map<uintptr_t, size_t>::iterator it) {
            const uintptr_t lo = it->first, hi = it->first + it->second;
            if (it->second < size) return false;
            const uintptr_t at = from_top ? (hi - size) & ~(align - 1) : (lo + align - 1) & ~(align - 1);
            if (at < lo || at + size > hi) return false;
            const size_t left = it->second - size;
            if (left < best_left) {
                best = it;
                best_at = at;
                best_left = left;
            }
            return mode == nn::lmem::AllocationMode_FirstFit;
        };
        if (from_top) {
            for (auto it = free.end(); it != free.begin();)
                if (consider(--it)) break;
        } else {
            for (auto it = free.begin(); it != free.end(); ++it)
                if (consider(it)) break;
        }
        if (best == free.end()) return nullptr;
        const uintptr_t lo = best->first, hi = best->first + best->second;
        free.erase(best);
        if (best_at > lo) free[lo] = best_at - lo;
        if (best_at + size < hi) free[best_at + size] = hi - (best_at + size);
        used[best_at] = size;
        void* p = reinterpret_cast<void*>(best_at);
        if (option & kOptionZeroClear) memset(p, 0, size);
        return p;
    }

    void release(void* p) {
        std::lock_guard lock(m);
        auto it = used.find(reinterpret_cast<uintptr_t>(p));
        if (it == used.end()) {
            printf("[nnsdk] FreeToExpHeap(%p): no es un bloque de este heap, se ignora\n", p);
            return;
        }
        add_free(it->first, it->second);
        used.erase(it);
    }
};

ExpHeap* exp_heap(nn::lmem::HeapHandle handle) { return reinterpret_cast<ExpHeap*>(handle); }

}  // namespace

namespace nn::lmem {

HeapHandle CreateExpHeap(void* address, size_t size, int option) {
    if (!address || size < 64) return nullptr;
    auto* heap = new ExpHeap;
    heap->begin = (reinterpret_cast<uintptr_t>(address) + 7) & ~uintptr_t{7};
    heap->end = (reinterpret_cast<uintptr_t>(address) + size) & ~uintptr_t{7};
    heap->option = option;
    heap->free[heap->begin] = heap->end - heap->begin;
    printf("[nnsdk] CreateExpHeap %p +%zu KB (opciones %#x)\n", address, size / 1024, option);
    return reinterpret_cast<HeapHandle>(heap);
}

void DestroyExpHeap(HeapHandle handle) { delete exp_heap(handle); }

AllocationMode SetExpHeapAllocationMode(HeapHandle handle, AllocationMode mode) {
    ExpHeap* heap = exp_heap(handle);
    std::lock_guard lock(heap->m);
    const AllocationMode old = heap->mode;
    heap->mode = mode;
    return old;
}

void* AllocateFromExpHeap(HeapHandle handle, size_t size, int alignment) {
    return handle ? exp_heap(handle)->allocate(size, alignment) : nullptr;
}

void FreeToExpHeap(HeapHandle handle, void* p) {
    if (handle && p) exp_heap(handle)->release(p);
}

}  // namespace nn::lmem
