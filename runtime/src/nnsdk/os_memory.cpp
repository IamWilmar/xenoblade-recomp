// El heap de la aplicación (SetMemoryHeapSize / AllocateMemoryBlock) y QueryMemoryInfo. El nninitStartup de MH Rise
// pide toda la memoria libre como heap ((disponible - usada) redondeado a 2 MB), saca de ahí 8 MB para su malloc
// (StandardAllocator) y el motor se queda luego con todo lo que sobra del heap (total - asignado) en un bloque.
// En la Switch eso es la memoria física de la aplicación; aquí es memoria virtual reservada sin compromiso
// (MAP_NORESERVE): solo cuestan RAM las páginas que el juego toque, y al liberar un bloque se le devuelven a Linux.
//
// Como MemoryHeapManager del SDK: el heap crece y encoge por el final (encoger solo si esa cola está libre) y los
// bloques se dan con el mejor ajuste (el hueco más pequeño que quepa; a igualdad, el de dirección más baja), desde
// el principio del hueco. A diferencia del SDK, la lista de huecos vive fuera del heap para no tocar sus páginas.
//
// La reserva es un memfd mapeado MAP_SHARED (si no se puede, memoria anónima como antes): así unas páginas del heap se
// pueden ver también en otra dirección (alias_heap_pages), que es lo que hacen los memory pools virtuales de NVN
// (nvn/memory.cpp) con los trozos de pools físicos que el juego les mapea. Liberar memoria es entonces abrir un hueco
// en el memfd (MADV_DONTNEED no devuelve las páginas de memoria compartida), que también vacía los alias.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>

#include "nnsdk/os.h"
#include "nnsdk/types.h"
#include "nnsdk/watch.h"

namespace nnsdk {
bool alias_heap_pages(void* dst, const void* src, size_t size);
}  // namespace nnsdk

namespace nn::os {

// Layout del SDK 11.4 (DebugImplByHorizon::QueryMemoryInfo).
struct MemoryInfo {
    size_t total_available_memory_size;  // svcGetInfo TotalNonSystemMemorySize
    size_t total_used_memory_size;       // svcGetInfo UsedNonSystemMemorySize
    size_t total_memory_heap_size;
    size_t allocated_memory_heap_size;
    size_t program_size;
    size_t total_thread_stack_size;
    int thread_count;
};

}  // namespace nn::os

namespace {

constexpr size_t kMB = 1024 * 1024;
constexpr size_t kHeapUnit = 2 * kMB;  // MemoryHeapUnitSize: el heap y su base van de 2 MB en 2 MB
constexpr size_t kPage = 4096;
// La memoria de una aplicación de Switch (el pool de aplicación de una consola de 4 GB, 3285 MB).
constexpr size_t kTotalMemory = 0xCD500000;
// Lo que Horizon cuenta como usado antes del heap: las imágenes de main (125 MB de código, 72 MB de rodata, 98 MB de
// data y 4 MB de bss), sdk y subsdk0.
constexpr size_t kProgramSize = 320 * kMB;

constexpr uint32_t kResultBusy = 0x803;          // os, 4: la cola del heap no está libre para encoger
constexpr uint32_t kResultOutOfMemory = 0x1003;  // os, 8

size_t round_up(size_t n, size_t unit) { return (n + unit - 1) & ~(unit - 1); }

std::mutex g_m;
uintptr_t g_base = 0;   // reserva de kTotalMemory alineada a 2 MB, PROT_NONE salvo el heap
uintptr_t g_map = 0;    // inicio del mmap: offset 0 del memfd
int g_fd = -1;          // memfd de la reserva (-1: memoria anónima, sin alias)
size_t g_heap_size = 0;
size_t g_allocated = 0;
std::map<uintptr_t, size_t> g_free;  // huecos del heap: dirección -> tamaño, sin solapes ni contiguos

// SetMemoryAllocatorForThreadLocal: solo se apuntan.
void* (*g_tls_allocate)(size_t, size_t) = nullptr;
void (*g_tls_free)(void*, size_t) = nullptr;

bool reserve() {
    if (g_base) return true;
    const size_t size = kTotalMemory + kHeapUnit;
    // El memfd no ocupa nada hasta que se tocan sus páginas (como MAP_NORESERVE). <datos>/sin_memfd: memoria anónima,
    // como antes (sin alias: lo que MH Rise mapea en sus pools virtuales se vería a cero).
    const char* data = getenv("NIER_DATA");
    const bool memfd = access((std::string(data ? data : ".") + "/sin_memfd").c_str(), F_OK) != 0;
    int fd = memfd ? memfd_create("nnsdk-heap", MFD_CLOEXEC) : -1;
    void* p = MAP_FAILED;
    if (fd >= 0 && ftruncate(fd, static_cast<off_t>(size)) == 0)
        p = mmap(nullptr, size, PROT_NONE, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED) {
        printf("[nnsdk] heap: sin memfd (%s): memoria anónima, los pools virtuales de NVN se verán a cero\n",
               memfd ? strerror(errno) : "interruptor sin_memfd");
        if (fd >= 0) close(fd);
        fd = -1;
        p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) return false;
    }
    g_fd = fd;
    g_map = reinterpret_cast<uintptr_t>(p);
    g_base = round_up(g_map, kHeapUnit);
    if (g_fd >= 0) nnsdk::watch::init_heap(g_map, size);
    printf("[nnsdk] heap: %zu MB de memoria virtual reservados en 0x%lx%s\n", kTotalMemory / kMB, g_base,
           g_fd >= 0 ? " (memfd)" : "");
    return true;
}

// Mete [address, address + size) en los huecos, uniéndolo con los vecinos.
void add_free(uintptr_t address, size_t size) {
    auto next = g_free.lower_bound(address);
    if (next != g_free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == address) {
            address = prev->first;
            size += prev->second;
            g_free.erase(prev);
        }
    }
    if (next != g_free.end() && address + size == next->first) {
        size += next->second;
        g_free.erase(next);
    }
    g_free[address] = size;
}

void discard(uintptr_t address, size_t size) {
    nnsdk::watch::heap_changed(address, size);  // las páginas pasan a ceros sin que nadie escriba
    if (g_fd >= 0 &&
        fallocate(g_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(address - g_map), static_cast<off_t>(size)) == 0)
        return;
    madvise(reinterpret_cast<void*>(address), size, MADV_DONTNEED);
}

}  // namespace

// Para los memory pools virtuales de NVN (nvn/memory.cpp): [dst, dst + size) pasa a ver las mismas páginas que
// [src, src + size), que tiene que estar en la reserva del heap. Todo alineado a página.
bool nnsdk::alias_heap_pages(void* dst, const void* src, size_t size) {
    static const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t s = reinterpret_cast<uintptr_t>(src), d = reinterpret_cast<uintptr_t>(dst);
    std::lock_guard lock(g_m);
    if (g_fd < 0 || !size || ((s | d | size) & (page - 1)) || s < g_base || s + size > g_base + kTotalMemory) return false;
    if (mmap(dst, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_fd, static_cast<off_t>(s - g_map)) == MAP_FAILED)
        return false;
    watch::virtual_mapped(d, size, s);
    return true;
}

namespace nn::os {

void QueryMemoryInfo(MemoryInfo* info) {
    size_t stacks = 0;
    int threads = 0;
    nnsdk::thread_totals(&stacks, &threads);
    std::lock_guard lock(g_m);
    info->total_available_memory_size = kTotalMemory;
    info->total_used_memory_size = kProgramSize + g_heap_size + stacks;
    info->total_memory_heap_size = g_heap_size;
    info->allocated_memory_heap_size = g_allocated;
    info->program_size = kProgramSize;
    info->total_thread_stack_size = stacks;
    info->thread_count = threads + 1;  // más el principal, que no crea nn::os
}

Result SetMemoryHeapSize(size_t size) {
    size = round_up(size, kHeapUnit);
    std::lock_guard lock(g_m);
    if (size > kTotalMemory || !reserve()) return Result{kResultOutOfMemory};
    const uintptr_t old_end = g_base + g_heap_size, new_end = g_base + size;
    if (size > g_heap_size) {
        if (mprotect(reinterpret_cast<void*>(old_end), size - g_heap_size, PROT_READ | PROT_WRITE) != 0)
            return Result{kResultOutOfMemory};
        add_free(old_end, size - g_heap_size);
    } else if (size < g_heap_size) {
        // Solo si el último hueco cubre toda la cola que se quita.
        if (g_free.empty()) return Result{kResultBusy};
        auto last = std::prev(g_free.end());
        if (last->first + last->second != old_end || last->first > new_end) return Result{kResultBusy};
        const uintptr_t start = last->first;
        g_free.erase(last);
        if (start < new_end) g_free[start] = new_end - start;
        discard(new_end, old_end - new_end);
        nnsdk::watch::heap_forget(new_end, old_end - new_end);
        mprotect(reinterpret_cast<void*>(new_end), old_end - new_end, PROT_NONE);
    }
    printf("[nnsdk] SetMemoryHeapSize %zu MB (antes %zu MB)\n", size / kMB, g_heap_size / kMB);
    g_heap_size = size;
    return ResultSuccess;
}

Result AllocateMemoryBlock(uintptr_t* out, size_t size) {
    const size_t want = round_up(size ? size : 1, kPage);
    std::lock_guard lock(g_m);
    auto best = g_free.end();
    for (auto it = g_free.begin(); it != g_free.end(); ++it)
        if (it->second >= want && (best == g_free.end() || it->second < best->second)) best = it;
    if (best == g_free.end()) {
        printf("[nnsdk] AllocateMemoryBlock %zu KB: no cabe (heap %zu MB, asignado %zu MB)\n", size / 1024,
               g_heap_size / kMB, g_allocated / kMB);
        return Result{kResultOutOfMemory};
    }
    const uintptr_t address = best->first;
    const size_t left = best->second - want;
    g_free.erase(best);
    if (left) g_free[address + want] = left;
    g_allocated += want;
    *out = address;
    printf("[nnsdk] AllocateMemoryBlock 0x%lx +%zu KB\n", address, size / 1024);
    return ResultSuccess;
}

void FreeMemoryBlock(uintptr_t address, size_t size) {
    const size_t len = round_up(size ? size : 1, kPage);
    std::lock_guard lock(g_m);
    if (address < g_base || address + len > g_base + g_heap_size) {
        // El SDK aborta: no es memoria del heap.
        printf("[nnsdk] FreeMemoryBlock 0x%lx +%zu KB: fuera del heap, se ignora\n", address, size / 1024);
        return;
    }
    discard(address, len);
    add_free(address, len);
    g_allocated = g_allocated > len ? g_allocated - len : 0;
    printf("[nnsdk] FreeMemoryBlock 0x%lx +%zu KB\n", address, size / 1024);
}

// El thread_local ELF del juego lo monta el loader (loader/tls.cpp) sin pedir memoria al juego; solo se apunta.
void SetMemoryAllocatorForThreadLocal(void* (*allocate)(size_t, size_t), void (*deallocate)(void*, size_t)) {
    g_tls_allocate = allocate;
    g_tls_free = deallocate;
}

// La región donde el proceso deja su mensaje al morir (para el informe de error de Horizon): no hay quien la lea.
void SetDyingMessageRegion(uintptr_t, size_t) {}

}  // namespace nn::os
