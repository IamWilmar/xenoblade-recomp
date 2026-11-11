// nvnMemoryPool*: la memoria la pone el juego (SetStorage). Como dirección de GPU usamos la
// misma dirección de CPU: así cualquier dirección que el juego pase en un comando se lee directo.
//
// Pools virtuales (MH Rise los usa para la memoria de texturas: trozos de 4 MB de pools físicos que mapea seguidos en
// un pool virtual con nvnMemoryPoolMapVirtual): no tienen memoria propia. Les damos una reserva de direcciones y
// MapVirtual hace que esas direcciones vean las mismas páginas que el trozo físico (el heap del juego es un memfd
// compartido, os_memory.cpp), así la CPU, las subidas del ejecutor y las texturas creadas en el pool virtual leen lo
// que el juego escribe por el físico. Lo no mapeado se lee como ceros.

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#include "nnsdk/watch.h"
#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/objects.h"

namespace nnsdk {
// os_memory.cpp: [dst, dst + size) pasa a ver las páginas del heap que hay en src. false si no se puede (src fuera del
// heap, sin alinear a página o sin memfd); entonces dst no cambia.
bool alias_heap_pages(void* dst, const void* src, size_t size);
}  // namespace nnsdk

namespace {

// NVNmemoryPoolBuilder es opaco; guardamos aquí dentro solo estos campos (cabe sobrado).
struct MemoryPoolBuilder {
    void* device;
    void* storage;
    size_t size;
    int flags;
};

// NVNmappingRequest (40 bytes): main+0x7728f40 los rellena seguidos con zancada 0x28 y pone en +0x00 el pool físico (el
// que va dentro de cada trozo de 4 MB, en trozo+0x58), +0x08 el offset en él (0), +0x10 el offset en el virtual (base +
// i·4 MB), +0x18 el tamaño (4 MB) y en +0x20 la clase de almacenamiento (32 bits). main+0x7726b34 mapea partes de un
// trozo (offset físico múltiplo de 256, tamaño múltiplo de 64 KB). Sin pool físico, se desmapea.
struct MappingRequest {
    const void* physical_pool;
    int64_t physical_offset;
    int64_t virtual_offset;
    uint64_t size;
    int32_t storage_class;
    int32_t pad;
};
static_assert(sizeof(MappingRequest) == 40);

size_t page_size() {
    static const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return page;
}

// Ceros en [p, p + size): páginas anónimas nuevas (lo que hubiera mapeado deja de verse).
bool map_zeros(void* p, size_t size) {
    if (mmap(p, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) == MAP_FAILED)
        return false;
    nnsdk::watch::virtual_mapped(reinterpret_cast<uintptr_t>(p), size, 0);
    return true;
}

}  // namespace

nvn::Objects<nvn::MemoryPool> nvn::g_memory_pools;

extern "C" {

void nvnMemoryPoolBuilderSetDefaults(MemoryPoolBuilder* b) { memset(b, 0, sizeof *b); }
void nvnMemoryPoolBuilderSetDevice(MemoryPoolBuilder* b, void* device) { b->device = device; }
void nvnMemoryPoolBuilderSetStorage(MemoryPoolBuilder* b, void* storage, size_t size) {
    b->storage = storage;
    b->size = size;
}
void nvnMemoryPoolBuilderSetFlags(MemoryPoolBuilder* b, int flags) { b->flags = flags; }
void* nvnMemoryPoolBuilderGetMemory(const MemoryPoolBuilder* b) { return b->storage; }
size_t nvnMemoryPoolBuilderGetSize(const MemoryPoolBuilder* b) { return b->size; }
int nvnMemoryPoolBuilderGetFlags(const MemoryPoolBuilder* b) { return b->flags; }

void* nvnMemoryPoolBuilderGetDevice(const MemoryPoolBuilder* b) { return b->device; }

nvn::Boolean nvnMemoryPoolInitialize(void* pool, const MemoryPoolBuilder* b) {
    uint8_t* storage = static_cast<uint8_t*>(b->storage);
    bool reserved = false;
    // Virtual (o sin memoria, que en NVN solo vale para los virtuales): direcciones propias, a cero hasta MapVirtual.
    if ((b->flags & nvn::kPoolFlagVirtual) || (!storage && b->size)) {
        const size_t size = (b->size + page_size() - 1) & ~(page_size() - 1);
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            NVN_LOG("memory pool virtual %p de %zu MB: sin direcciones\n", pool, b->size >> 20);
            return 0;
        }
        storage = static_cast<uint8_t*>(p);
        reserved = true;
        nnsdk::watch::add_virtual_range(reinterpret_cast<uintptr_t>(p), size);
    }
    static int logged = 0;  // DEPURACIÓN: los pools (para confirmar sus flags; miniatura de la partida: cuál es el de captura)
    if (nvn::first_times(logged, 64) || NVN_FLAG("diag_miniatura"))
        NVN_LOG("memory pool %p %s: %zu KB, flags 0x%x, memoria %p%s\n", pool,
                reserved ? "virtual" : (b->flags & nvn::kPoolFlagPhysical) ? "físico" : "", b->size >> 10, b->flags,
                static_cast<void*>(storage), reserved && b->storage ? " (la del juego no se usa)" : "");
    nvn::MemoryPool& p = nvn::g_memory_pools.create(pool);
    p.storage = storage;
    p.size = b->size;
    p.flags = b->flags;
    p.reserved = reserved;
    return 1;
}
void nvnMemoryPoolFinalize(void* pool) {
    {  // DEPURACIÓN (miniatura de la partida)
        static int logged = 0;
        if (nvn::first_times(logged, 64) || NVN_FLAG("diag_miniatura")) NVN_LOG("pool finalizado %p\n", pool);
    }
    if (std::unique_ptr<nvn::MemoryPool> p = nvn::g_memory_pools.take(pool); p && p->reserved) {
        // Lo que ya está grabado puede seguir leyendo de sus direcciones (subidas de texturas): primero que acabe.
        nvn::wait_idle();
        munmap(p->storage, (p->size + page_size() - 1) & ~(page_size() - 1));
    }
}

// Cada petición hace que [virtual + offset, + size) vea [físico + offset, + size), o ceros si no hay pool físico. Si
// algo no se puede ver (no está en el heap o sin alinear a página), se deja a cero y se avisa: esas direcciones siguen
// siendo válidas y se contesta que sí (solo una petición fuera del pool virtual da NVN_FALSE).
nvn::Boolean nvnMemoryPoolMapVirtual(void* pool, int count, const MappingRequest* requests) {
    nvn::MemoryPool* v = nvn::g_memory_pools.get(pool);
    if (!v || !v->reserved || count < 0 || (count && !requests)) {
        NVN_LOG("MapVirtual sobre %p, que no es un pool virtual\n", pool);
        return 0;
    }
    static int logged = 0, failed = 0;
    bool ok = true;
    for (int k = 0; k < count; k++) {
        const MappingRequest& r = requests[k];
        if (r.virtual_offset < 0 || r.size > v->size || static_cast<uint64_t>(r.virtual_offset) > v->size - r.size) {
            NVN_LOG("MapVirtual: +0x%llx (%llu KB) fuera del pool virtual de %zu KB\n", static_cast<unsigned long long>(r.virtual_offset),
                    static_cast<unsigned long long>(r.size >> 10), v->size >> 10);
            ok = false;
            continue;
        }
        uint8_t* dst = v->storage + r.virtual_offset;
        const nvn::MemoryPool* phys = r.physical_pool ? nvn::g_memory_pools.get(r.physical_pool) : nullptr;
        bool mapped = false;
        if (!r.physical_pool) {
            mapped = map_zeros(dst, r.size);
        } else if (phys && r.physical_offset >= 0 && r.size <= phys->size &&
                   static_cast<uint64_t>(r.physical_offset) <= phys->size - r.size) {
            mapped = nnsdk::alias_heap_pages(dst, phys->storage + r.physical_offset, r.size);
        }
        if (!mapped) {
            map_zeros(dst, r.size);
            if (r.physical_pool && nvn::first_times(failed, 16))
                NVN_LOG("MapVirtual: no se puede ver %p+0x%llx (pool %s) en +0x%llx (%llu KB): se queda a cero\n",
                        phys ? static_cast<void*>(phys->storage) : nullptr, static_cast<unsigned long long>(r.physical_offset),
                        phys ? "fuera del heap o sin alinear" : "desconocido", static_cast<unsigned long long>(r.virtual_offset),
                        static_cast<unsigned long long>(r.size >> 10));
        }
        if (logged < 24) {  // DEPURACIÓN: las primeras (las del arranque: 6, 8 y 1 peticiones)
            logged++;
            NVN_LOG("MapVirtual %p: +0x%llx <- %p+0x%llx (%llu KB, clase 0x%x)%s\n", pool,
                    static_cast<unsigned long long>(r.virtual_offset), r.physical_pool,
                    static_cast<unsigned long long>(r.physical_offset), static_cast<unsigned long long>(r.size >> 10),
                    r.storage_class, mapped ? "" : " FALLO");
        }
    }
    return ok ? 1 : 0;
}

// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnMemoryPoolSetDebugLabel(void*, const char*) {}

void* nvnMemoryPoolMap(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->storage : nullptr;
}
uint64_t nvnMemoryPoolGetBufferAddress(const void* pool) { return reinterpret_cast<uint64_t>(nvnMemoryPoolMap(pool)); }
size_t nvnMemoryPoolGetSize(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->size : 0;
}
int nvnMemoryPoolGetFlags(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->flags : 0;
}
// La CPU escribió ese rango para la GPU: como nn::os::FlushDataCache (note_cpu_flush), los overlays de la GPU que lo
// tocan toman la memoria del juego en el siguiente envío.
void nvnMemoryPoolFlushMappedRange(const void* pool, int64_t offset, size_t size) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    if (!p || !p->storage || offset < 0 || static_cast<size_t>(offset) >= p->size) return;
    nvn::note_cpu_flush(reinterpret_cast<uint64_t>(p->storage) + static_cast<uint64_t>(offset),
                        std::min(size, p->size - static_cast<size_t>(offset)));
}
// La CPU va a leer lo que escribió la GPU: lo que vuelve a la memoria del juego ya está ahí al esperar el sync del envío
// (deliver_write_backs), que NVN exige antes de invalidar. No queda nada que hacer.
void nvnMemoryPoolInvalidateMappedRange(const void*, int64_t, size_t) {}

}  // extern "C"

NVN_PROC(nvnMemoryPoolBuilderSetDefaults);
NVN_PROC(nvnMemoryPoolBuilderSetDevice);
NVN_PROC(nvnMemoryPoolBuilderSetStorage);
NVN_PROC(nvnMemoryPoolBuilderSetFlags);
NVN_PROC(nvnMemoryPoolBuilderGetMemory);
NVN_PROC(nvnMemoryPoolBuilderGetSize);
NVN_PROC(nvnMemoryPoolBuilderGetFlags);
NVN_PROC(nvnMemoryPoolBuilderGetDevice);
NVN_PROC(nvnMemoryPoolInitialize);
NVN_PROC(nvnMemoryPoolMapVirtual);
NVN_PROC(nvnMemoryPoolSetDebugLabel);
NVN_PROC(nvnMemoryPoolFinalize);
NVN_PROC(nvnMemoryPoolMap);
NVN_PROC(nvnMemoryPoolGetBufferAddress);
NVN_PROC(nvnMemoryPoolGetSize);
NVN_PROC(nvnMemoryPoolGetFlags);
NVN_PROC(nvnMemoryPoolFlushMappedRange);
NVN_PROC(nvnMemoryPoolInvalidateMappedRange);
