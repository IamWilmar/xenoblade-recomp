// Lo que glslc (el compilador de shaders de NVIDIA que Catherine: Full Body trae en subsdk0) importa del sdk: NvOs
// (archivos y mutex de la capa de sistema de NVIDIA) y el pegamento de su allocator.
//
// glslc_*: en la consola el sdk guarda una tabla de 4 funciones {Alloc, AllocAlign, Realloc, Free} (sdk+0xc0fb90) que
// glslcSetAllocator (subsdk0+0x56f170) rellena con sus propios trampolines hacia el allocator del juego;
// glslc_GetAllocator devuelve esa tabla y glslc_Alloc y compañía llaman a su entrada pasándole la tabla. Hasta que el
// juego pone el suyo van al malloc del juego.
//
// NvOs*: NvError 0 = éxito. Los archivos son FILE* de bionic abiertos con la traducción de rutas del runtime
// (nngame_fopen); glslc solo los usa para sus volcados de depuración. El mutex es recursivo, como el del sdk
// (nn::os::InitializeMutex(m, true, 0) en sdk+0x3cf288).

#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "nnsdk/runtime.h"

extern "C" FILE* nngame_fopen(const char* path, const char* mode);

namespace {

struct GlslcAllocator {
    void* (*alloc)(GlslcAllocator*, size_t);
    void* (*alloc_align)(GlslcAllocator*, size_t, size_t);
    void* (*realloc)(GlslcAllocator*, void*, size_t);
    void (*free)(GlslcAllocator*, void*);
};

void* default_alloc(GlslcAllocator*, size_t size) { return nnsdk::host().game_malloc(size); }
void* default_alloc_align(GlslcAllocator*, size_t size, size_t align) {
    return nnsdk::host().game_aligned_alloc ? nnsdk::host().game_aligned_alloc(align, size)
                                            : nnsdk::host().game_malloc(size);
}
// Sin el tamaño viejo no se puede mover un bloque del allocator del juego: el juego pone el suyo antes de compilar.
void* default_realloc(GlslcAllocator*, void*, size_t) {
    printf("[glslc] Realloc antes de glslcSetAllocator\n");
    return nullptr;
}
void default_free(GlslcAllocator*, void* p) { nnsdk::host().game_free(p); }

GlslcAllocator g_glslc{default_alloc, default_alloc_align, default_realloc, default_free};

constexpr uint32_t kNvSuccess = 0, kNvErrorBadParameter = 4, kNvErrorInsufficientMemory = 6,
                   kNvErrorFileOperationFailed = 0x30003, kNvErrorEndOfFile = 0x30004;
constexpr uint32_t kOpenRead = 1, kOpenWrite = 2, kOpenCreate = 4, kOpenAppend = 8;

}  // namespace

extern "C" {

GlslcAllocator* glslc_GetAllocator() { return &g_glslc; }
void* glslc_Alloc(size_t size) { return g_glslc.alloc(&g_glslc, size); }
void* glslc_AllocAlign(size_t size, size_t align) { return g_glslc.alloc_align(&g_glslc, size, align); }
void* glslc_Realloc(void* p, size_t size) { return g_glslc.realloc(&g_glslc, p, size); }
void glslc_Free(void* p) { g_glslc.free(&g_glslc, p); }

uint32_t NvOsFopen(const char* path, uint32_t flags, FILE** out) {
    if (!path || !out) return kNvErrorBadParameter;
    const char* mode = "rb";
    if (flags & kOpenAppend)
        mode = flags & kOpenRead ? "a+b" : "ab";
    else if (flags & kOpenWrite)
        mode = flags & kOpenCreate ? (flags & kOpenRead ? "w+b" : "wb") : "r+b";
    *out = nngame_fopen(path, mode);
    return *out ? kNvSuccess : kNvErrorFileOperationFailed;
}
void NvOsFclose(FILE* f) {
    if (f) fclose(f);
}
uint32_t NvOsFwrite(FILE* f, const void* p, size_t size) {
    return f && fwrite(p, 1, size, f) == size ? kNvSuccess : kNvErrorFileOperationFailed;
}
uint32_t NvOsFread(FILE* f, void* p, size_t size, size_t* read) {
    if (!f) return kNvErrorBadParameter;
    const size_t n = fread(p, 1, size, f);
    if (read) *read = n;
    if (n == size) return kNvSuccess;
    return feof(f) ? kNvErrorEndOfFile : kNvErrorFileOperationFailed;
}
uint32_t NvOsFseek(FILE* f, int64_t offset, int whence) {
    return f && fseeko(f, offset, whence) == 0 ? kNvSuccess : kNvErrorFileOperationFailed;
}
uint32_t NvOsFtell(FILE* f, uint64_t* pos) {
    if (!f || !pos) return kNvErrorBadParameter;
    const off_t p = ftello(f);
    if (p < 0) return kNvErrorFileOperationFailed;
    *pos = static_cast<uint64_t>(p);
    return kNvSuccess;
}

uint32_t NvOsMutexCreate(pthread_mutex_t** out) {
    auto* m = new (std::nothrow) pthread_mutex_t;
    if (!m) {
        *out = nullptr;
        return kNvErrorInsufficientMemory;
    }
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(m, &attr);
    pthread_mutexattr_destroy(&attr);
    *out = m;
    return kNvSuccess;
}
void NvOsMutexLock(pthread_mutex_t* m) {
    if (m) pthread_mutex_lock(m);
}
void NvOsMutexUnlock(pthread_mutex_t* m) {
    if (m) pthread_mutex_unlock(m);
}
void NvOsMutexDestroy(pthread_mutex_t* m) {
    if (!m) return;
    pthread_mutex_destroy(m);
    delete m;
}

}  // extern "C"
