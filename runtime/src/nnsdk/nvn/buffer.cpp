// nvnBuffer*: una ventana (offset, tamaño) sobre un memory pool. Dirección de GPU = de CPU.

#include <algorithm>
#include <atomic>
#include <cstring>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

namespace {

// NVNbufferBuilder es opaco; guardamos aquí dentro solo estos campos.
struct BufferBuilder {
    void* device;
    void* pool;
    int64_t offset;
    size_t size;
};

struct Buffer {
    uint8_t* data = nullptr;
    size_t size = 0;
    void* pool = nullptr;
    int64_t offset = 0;
    uint64_t debug_id = 0;  // GetDebugID: único en toda la ejecución (la dirección del objeto se reutiliza)
};

nvn::Objects<Buffer> g_buffers;

}  // namespace

extern "C" {

void nvnBufferBuilderSetDefaults(BufferBuilder* b) { memset(b, 0, sizeof *b); }
void nvnBufferBuilderSetDevice(BufferBuilder* b, void* device) { b->device = device; }
void nvnBufferBuilderSetStorage(BufferBuilder* b, void* pool, int64_t offset, size_t size) {
    b->pool = pool;
    b->offset = offset;
    b->size = size;
}

nvn::Boolean nvnBufferInitialize(void* buffer, const BufferBuilder* b) {
    nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool);
    if (!pool) {
        NVN_LOG("buffer %p sobre un memory pool desconocido %p\n", buffer, b->pool);
        return 0;
    }
    Buffer& buf = g_buffers.create(buffer);
    buf.data = pool->storage + b->offset;
    pool->buffers = true;  // finalizar el pool espera a que la cola reproduzca lo que lee de él (memory.cpp)
    buf.size = b->size;
    buf.pool = b->pool;
    buf.offset = b->offset;
    static std::atomic<uint64_t> next_id{1};
    buf.debug_id = next_id++;
    return 1;
}
void nvnBufferFinalize(void* buffer) { g_buffers.destroy(buffer); }

void* nvnBufferMap(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->data : nullptr;
}
uint64_t nvnBufferGetAddress(const void* buffer) { return reinterpret_cast<uint64_t>(nvnBufferMap(buffer)); }
size_t nvnBufferGetSize(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->size : 0;
}
void* nvnBufferBuilderGetDevice(const BufferBuilder* b) { return b->device; }
void* nvnBufferBuilderGetMemoryPool(const BufferBuilder* b) { return b->pool; }
int64_t nvnBufferBuilderGetMemoryOffset(const BufferBuilder* b) { return b->offset; }
size_t nvnBufferBuilderGetSize(const BufferBuilder* b) { return b->size; }
void* nvnBufferGetMemoryPool(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->pool : nullptr;
}
int64_t nvnBufferGetMemoryOffset(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->offset : 0;
}
uint64_t nvnBufferGetDebugID(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->debug_id : 0;
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnBufferSetDebugLabel(void*, const char*) {}
// La CPU escribió ese rango para la GPU: como nn::os::FlushDataCache, va al siguiente envío y los overlays de la GPU que
// lo tocan toman la memoria del juego (note_cpu_flush).
void nvnBufferFlushMappedRange(const void* buffer, int64_t offset, size_t size) {
    Buffer* b = g_buffers.get(buffer);
    if (!b || offset < 0 || static_cast<size_t>(offset) >= b->size) return;
    nvn::note_cpu_flush(reinterpret_cast<uint64_t>(b->data) + static_cast<uint64_t>(offset),
                        std::min(size, b->size - static_cast<size_t>(offset)));
}
// La CPU va a leer lo que escribió la GPU: lo que se devuelve a la memoria del juego ya está ahí al esperar el sync del
// envío (deliver_write_backs en la espera), que NVN exige antes de invalidar. No queda nada que hacer.
void nvnBufferInvalidateMappedRange(const void*, int64_t, size_t) {}

}  // extern "C"

NVN_PROC(nvnBufferBuilderSetDefaults);
NVN_PROC(nvnBufferBuilderSetDevice);
NVN_PROC(nvnBufferBuilderSetStorage);
NVN_PROC(nvnBufferInitialize);
NVN_PROC(nvnBufferFinalize);
NVN_PROC(nvnBufferMap);
NVN_PROC(nvnBufferGetAddress);
NVN_PROC(nvnBufferGetSize);
NVN_PROC(nvnBufferBuilderGetDevice);
NVN_PROC(nvnBufferBuilderGetMemoryPool);
NVN_PROC(nvnBufferBuilderGetMemoryOffset);
NVN_PROC(nvnBufferBuilderGetSize);
NVN_PROC(nvnBufferGetMemoryPool);
NVN_PROC(nvnBufferGetMemoryOffset);
NVN_PROC(nvnBufferSetDebugLabel);
NVN_PROC(nvnBufferFlushMappedRange);
NVN_PROC(nvnBufferInvalidateMappedRange);
NVN_PROC(nvnBufferGetDebugID);
