// HUD de rendimiento (ver hud.h). Todo en el hilo que presenta: sin locks.

#include "nnsdk/nvn/hud.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"

namespace nvn::hud {
namespace {

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

constexpr uint32_t kW = 44, kH = 26;       // en píxeles del swapchain (1920x1080): solo el número, que no estorbe
constexpr int32_t kX = 8, kY = 8;          // esquina superior izquierda
constexpr uint32_t kRing = 12;             // buffers: más que presentaciones en vuelo (3 reales + 3 de LSFG)

// Fuente de 5x7: cada fila, 5 bits (el alto a la izquierda).
struct Glyph {
    char c;
    uint8_t rows[7];
};
constexpr Glyph kFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}}, {'%', {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}}, {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}}, {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}}, {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}}, {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}}, {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}}, {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
};

struct Rgba {
    uint8_t r, g, b, a;
};
constexpr Rgba kBack{12, 12, 14, 255};
constexpr Rgba kGood{80, 210, 110, 255}, kWarn{235, 195, 60, 255}, kBad{235, 80, 70, 255};

class Hud {
public:
    void real_frame(int64_t now) {
        if (last_real_ && now > last_real_) {
            intervals_.push_back((now - last_real_) / 1e6);
            if (intervals_.size() > 300) intervals_.pop_front();
        }
        last_real_ = now;
        reals_.push_back(now);
        trim(reals_, now);
    }
    void generated_frame() {
        generated_.push_back(last_real_);
        trim(generated_, last_real_);
    }
    void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
        if (NVN_FLAG("sin_fps") || failed_) return;
        const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
        if (!bgra && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R8G8B8A8_SRGB) return;
        if (extent.width < kX + kW || extent.height < kY + kH) return;
        if (buffers_.empty() && !create()) return;
        Buffer& b = buffers_[next_];
        next_ = (next_ + 1) % kRing;
        draw(reinterpret_cast<Rgba*>(b.data), bgra);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {kX, kY, 0};
        region.imageExtent = {kW, kH, 1};
        vkCmdCopyBufferToImage(cmd, b.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* data = nullptr;
    };
    static void trim(std::deque<int64_t>& q, int64_t now) {
        while (!q.empty() && now - q.front() > 1000000000) q.pop_front();
    }
    bool create() {
        VkContext* c = vk();
        for (uint32_t k = 0; k < kRing; k++) {
            Buffer b;
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = VkDeviceSize{kW} * kH * 4;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VkMemoryRequirements req;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            if (vkCreateBuffer(c->device, &bi, nullptr, &b.buffer) != VK_SUCCESS) return fail();
            vkGetBufferMemoryRequirements(c->device, b.buffer, &req);
            ai.allocationSize = req.size;
            ai.memoryTypeIndex =
                find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &b.memory) != VK_SUCCESS ||
                vkBindBufferMemory(c->device, b.buffer, b.memory, 0) != VK_SUCCESS ||
                vkMapMemory(c->device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.data) != VK_SUCCESS)
                return fail();
            buffers_.push_back(b);
        }
        return true;
    }
    bool fail() {
        NVN_LOG("hud: no se pudo crear su buffer; sin HUD\n");
        failed_ = true;
        return false;
    }

    void draw(Rgba* px, bool bgra) {
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= static_cast<int>(kW) || y >= static_cast<int>(kH)) return;
            px[y * kW + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int w, int h, Rgba c) {
            for (int y = y0; y < y0 + h; y++)
                for (int x = x0; x < x0 + w; x++) put(x, y, c);
        };
        const auto text = [&](int x, int y, int scale, const char* s, Rgba c) {
            for (; *s; s++, x += 6 * scale) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b)) rect(x + b * scale, y + r * scale, scale, scale, c);
            }
        };
        rect(0, 0, kW, kH, kBack);

        // El número cambia cuatro veces por segundo, para que se pueda leer.
        if (last_real_ - text_at_ > 250000000 || !text_at_) {
            text_at_ = last_real_;
            shown_ = static_cast<int>(reals_.size() + generated_.size());
        }
        // Verde a su ritmo (30, o 60 con LSFG), amarillo si se queda corto, rojo si va mal.
        const int target = generated_.empty() ? 30 : 60;
        const Rgba color = shown_ >= target - 1 ? kGood : shown_ >= target * 5 / 6 ? kWarn : kBad;
        char line[8];
        snprintf(line, sizeof line, "%d", std::min(shown_, 99));
        const int width = static_cast<int>(strlen(line)) * 6 * 3 - 3;
        text((static_cast<int>(kW) - width) / 2, 3, 3, line, color);
    }

    std::vector<Buffer> buffers_;
    uint32_t next_ = 0;
    bool failed_ = false;
    std::deque<double> intervals_;          // ms entre frames reales (para el 1 % más lento, si vuelve a hacer falta)
    std::deque<int64_t> reals_, generated_;  // en el último segundo
    int64_t last_real_ = 0, text_at_ = 0;
    int shown_ = 0;
};

Hud g_hud;

}  // namespace

void real_frame(int64_t now_ns) { g_hud.real_frame(now_ns); }
void generated_frame() { g_hud.generated_frame(); }
void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
    g_hud.record(cmd, image, extent, format);
}

}  // namespace nvn::hud
