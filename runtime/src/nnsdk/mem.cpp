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
#include <unordered_map>

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
};

namespace {
// Tamaño de la región que dio cada allocator al iniciarse (lo pocos que hay): lo que "queda libre" para el juego. El
// malloc de bionic no lleva la cuenta por allocator, así que se dice la región entera.
std::mutex g_regions_m;
std::unordered_map<const void*, size_t> g_regions;
}  // namespace

// Sin miembros: el objeto vive en memoria del juego y no se toca.
StandardAllocator::StandardAllocator() {}
void StandardAllocator::Initialize(void*, size_t) {}
void StandardAllocator::Initialize(void* address, size_t size, bool) {
    {
        std::lock_guard lock(g_regions_m);
        g_regions[this] = size;
    }
    printf("[nnsdk] StandardAllocator %p: región %p +%zu KB (se usa el malloc de bionic)\n", static_cast<void*>(this),
           address, size / 1024);
}
void StandardAllocator::Finalize() {}

void* StandardAllocator::Allocate(size_t size) { return Allocate(size, 16); }

void* StandardAllocator::Allocate(size_t size, size_t alignment) {
    void* p = nullptr;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    return posix_memalign(&p, alignment, size ? size : 1) == 0 ? p : nullptr;
}

// Como el SDK: tamaño "negativo" -> nullptr sin tocar nada; sin puntero es Allocate (alineado a 16); tamaño 0 libera
// y devuelve nullptr; si no cabe, nullptr y el bloque sigue como estaba (realloc hace lo mismo).
void* StandardAllocator::Reallocate(void* p, size_t size) {
    if (static_cast<ptrdiff_t>(size) < 0) return nullptr;
    if (!p) return Allocate(size, 16);
    if (size == 0) {
        free(p);
        return nullptr;
    }
    return realloc(p, (size + 15) & ~size_t{15});
}

void StandardAllocator::Free(void* p) { free(p); }
size_t StandardAllocator::GetSizeOf(const void* p) const { return p ? malloc_usable_size(p) : 0; }
size_t StandardAllocator::GetTotalFreeSize() const {
    std::lock_guard lock(g_regions_m);
    auto it = g_regions.find(this);
    return it != g_regions.end() ? it->second : 0;
}
size_t StandardAllocator::GetAllocatableSize() const { return GetTotalFreeSize(); }

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
