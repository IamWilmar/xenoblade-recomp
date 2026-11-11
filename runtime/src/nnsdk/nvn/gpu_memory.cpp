#include "nnsdk/nvn/gpu_memory.h"

#include <cstdio>
#include <unistd.h>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/vk.h"

namespace nvn {
namespace {

constexpr VkDeviceSize kBlockSize = 64ull << 20;
constexpr VkDeviceSize kDedicatedFrom = kBlockSize / 4;  // a partir de aquí, reserva propia

struct Block {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint32_t type = 0;
    std::map<VkDeviceSize, VkDeviceSize> free;  // inicio -> tamaño de cada hueco libre
    VkDeviceSize used = 0;
};

std::mutex g_mutex;
std::vector<Block> g_blocks;  // un bloque liberado se queda con memory nulo y se reutiliza su hueco en el vector
size_t g_dedicated = 0;

VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) { return a ? (v + a - 1) / a * a : v; }

// Primer hueco de b donde cabe (size, alignment); carga el desplazamiento en *offset.
bool carve(Block& b, VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize* offset) {
    for (auto it = b.free.begin(); it != b.free.end(); ++it) {
        const VkDeviceSize start = it->first, len = it->second;
        const VkDeviceSize at = align_up(start, alignment);
        if (at + size > start + len) continue;
        b.free.erase(it);
        if (at > start) b.free[start] = at - start;
        if (at + size < start + len) b.free[at + size] = start + len - (at + size);
        b.used += size;
        *offset = at;
        return true;
    }
    return false;
}

void give_back(Block& b, VkDeviceSize offset, VkDeviceSize size) {
    b.used -= size;
    auto it = b.free.emplace(offset, size).first;
    if (auto next = std::next(it); next != b.free.end() && it->first + it->second == next->first) {
        it->second += next->second;
        b.free.erase(next);
    }
    if (it != b.free.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            b.free.erase(it);
        }
    }
}

// Un fallo deja constancia de por qué (código de Vulkan y RAM libre del sistema en ese momento).
bool allocate(VkDevice device, const VkMemoryAllocateInfo& ai, VkDeviceMemory* out) {
    VkResult r = vkAllocateMemory(device, &ai, nullptr, out);
    if (r == VK_SUCCESS) return true;
    long available_kb = -1;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "MemAvailable: %ld kB", &available_kb) == 1) break;
        fclose(f);
    }
    static int logged = 0;
    if (nvn::first_times(logged, 8))
        NVN_LOG("vkAllocateMemory de %llu MB: %s (VkResult %d), RAM disponible %ld MB\n",
                static_cast<unsigned long long>(ai.allocationSize >> 20), r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "sin memoria de dispositivo"
                : r == VK_ERROR_OUT_OF_HOST_MEMORY ? "sin memoria del host" : "error", r, available_kb / 1024);
    return false;
}

}  // namespace

bool gpu_alloc(const VkMemoryRequirements& req, uint32_t memory_type, GpuAllocation& out) {
    VkContext* c = vk();
    out = {};
    out.type = memory_type;
    out.size = req.size;
    if (req.size >= kDedicatedFrom) {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type;
        bool ok = false;
        for (int attempt = 0; attempt < 4 && !ok; attempt++) {
            ok = allocate(c->device, ai, &out.memory);
            if (!ok) usleep(40'000);
        }
        if (!ok) return false;
        std::lock_guard lock(g_mutex);
        g_dedicated++;
        return true;
    }
    g_mutex.lock();
    struct Unlock {
        ~Unlock() { g_mutex.unlock(); }
    } unlock;
    for (size_t k = 0; k < g_blocks.size(); k++) {
        Block& b = g_blocks[k];
        if (!b.memory || b.type != memory_type) continue;
        if (carve(b, req.size, req.alignment, &out.offset)) {
            out.memory = b.memory;
            out.block = static_cast<int>(k);
            return true;
        }
    }
    Block fresh;
    fresh.type = memory_type;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.memoryTypeIndex = memory_type;
    // Con poca RAM libre el driver rechaza un bloque grande sin esperar a que Android recupere memoria:
    // se prueba más pequeño y, si tampoco, se espera un poco (el lowmemorykiller cierra apps de fondo).
    const VkDeviceSize sizes[] = {kBlockSize, kBlockSize / 4};
    bool ok = false;
    for (int attempt = 0; attempt < 4 && !ok; attempt++) {
        for (VkDeviceSize size : sizes) {
            if (size < req.size + req.alignment) continue;
            ai.allocationSize = size;
            if (allocate(c->device, ai, &fresh.memory)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            g_mutex.unlock();
            usleep(40'000);
            g_mutex.lock();
        }
    }
    if (!ok) return false;
    fresh.size = ai.allocationSize;
    fresh.free[0] = fresh.size;
    size_t index = g_blocks.size();
    for (size_t k = 0; k < g_blocks.size(); k++)
        if (!g_blocks[k].memory) {
            index = k;
            break;
        }
    if (index == g_blocks.size()) g_blocks.emplace_back();
    g_blocks[index] = std::move(fresh);
    Block& b = g_blocks[index];
    if (!carve(b, req.size, req.alignment, &out.offset)) return false;  // no pasa: el bloque está vacío
    out.memory = b.memory;
    out.block = static_cast<int>(index);
    if (index % 8 == 0) NVN_LOG("memoria de texturas: bloque %zu de %llu MB\n", index, (unsigned long long)(b.size >> 20));
    return true;
}

void gpu_free(GpuAllocation& a) {
    if (!a.memory) return;
    VkContext* c = vk();
    if (a.block < 0) {
        vkFreeMemory(c->device, a.memory, nullptr);
        std::lock_guard lock(g_mutex);
        g_dedicated--;
    } else {
        std::lock_guard lock(g_mutex);
        Block& b = g_blocks[a.block];
        give_back(b, a.offset, a.size);
        // Un bloque vacío se devuelve al driver si queda otro del mismo tipo con hueco.
        if (b.used == 0) {
            size_t others = 0;
            for (const Block& o : g_blocks) others += o.memory && o.type == b.type && &o != &b;
            if (others) {
                vkFreeMemory(c->device, b.memory, nullptr);
                b = Block{};
            }
        }
    }
    a = {};
}

void gpu_memory_stats(size_t* blocks, size_t* dedicated, VkDeviceSize* block_bytes) {
    std::lock_guard lock(g_mutex);
    size_t n = 0;
    for (const Block& b : g_blocks) n += b.memory != VK_NULL_HANDLE;
    *blocks = n;
    *dedicated = g_dedicated;
    VkDeviceSize bytes = 0;
    for (const Block& b : g_blocks) bytes += b.memory ? b.size : 0;
    *block_bytes = bytes;
}

}  // namespace nvn
