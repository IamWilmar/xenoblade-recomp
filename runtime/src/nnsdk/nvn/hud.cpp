// HUD de rendimiento (ver hud.h). Todo en el hilo que presenta: sin locks.

#include "nnsdk/nvn/hud.h"

#include <fcntl.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "nnsdk/host.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"

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
// El panel de ajustes (ver hud.h): un cajón que sale del borde derecho, de un tercio del ancho de la pantalla y toda su
// altura, con una fila por ajuste.
constexpr int64_t kPanelSlideNs = 180'000'000;    // lo que tarda en salir o esconderse
constexpr int64_t kPanelIdleNs = 8'000'000'000;   // sin tocarlo, se esconde solo
constexpr int kPanelHeader = 150, kPanelRow = 132;  // alto de la cabecera y de cada fila, en píxeles

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
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}}, {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}}, {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}}, {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}}, {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}}, {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}}, {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}}, {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
};

struct Rgba {
    uint8_t r, g, b, a;
};
constexpr Rgba kBack{12, 12, 14, 255};
constexpr Rgba kGood{80, 210, 110, 255}, kWarn{235, 195, 60, 255}, kBad{235, 80, 70, 255};
constexpr Rgba kEdge{58, 58, 66, 255}, kText{232, 232, 238, 255}, kOff{74, 74, 84, 255}, kKnob{240, 240, 244, 255};
constexpr Rgba kDim{150, 150, 160, 255}, kLine{34, 34, 40, 255};

// Los textos del panel, en español si el sistema está en español y en inglés si no.
bool spanish() {
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = "";
    if (__system_property_get("persist.sys.locale", value) <= 0) __system_property_get("ro.product.locale", value);
    return value[0] == 'e' && value[1] == 's';
#else
    return true;
#endif
}

// El último toque de la pantalla, como lo deja el loader en NnsdkInput (host.h).
struct Touch {
    uint32_t serial = 0, state = 0;  // state: 1 posado, 0 levantado, 2 cancelado
    int sx = 0, sy = 0, x = 0, y = 0;  // dónde se posó y por dónde va (o dónde se levantó), en píxeles de la ventana
};
bool read_touch(Touch* t) {
    NnsdkInput* in = nnsdk::host().input;
    if (!in) return false;
    t->serial = in->touch_serial.load(std::memory_order_acquire);
    t->state = in->touch_state.load(std::memory_order_acquire);
    const uint64_t start = in->touch_start.load(std::memory_order_acquire), last = in->touch_last.load(std::memory_order_acquire);
    t->sx = static_cast<int32_t>(start & 0xffffffffu), t->sy = static_cast<int32_t>(start >> 32);
    t->x = static_cast<int32_t>(last & 0xffffffffu), t->y = static_cast<int32_t>(last >> 32);
    return t->serial == in->touch_serial.load(std::memory_order_acquire);  // si cambió a medias, al frame siguiente
}

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
        // El contador se apaga con <datos>/sin_fps (en vivo, como siempre) o desde el panel, que pone y quita ese mismo
        // fichero: lo elegido vale también para la próxima vez que se abra el juego.
        if (const bool file = NVN_FLAG("sin_fps"); !flag_seen_ || file != flag_) flag_seen_ = true, flag_ = hidden_ = file;
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        gestures(extent, now);
        if (failed_) return;
        const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
        if (!bgra && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R8G8B8A8_SRGB) return;
        if (!hidden_ && extent.width >= kX + kW && extent.height >= kY + kH && (!buffers_.empty() || create(buffers_, kW, kH))) {
            Buffer& b = buffers_[next_];
            next_ = (next_ + 1) % kRing;
            draw(reinterpret_cast<Rgba*>(b.data), bgra);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset = {kX, kY, 0};
            region.imageExtent = {kW, kH, 1};
            vkCmdCopyBufferToImage(cmd, b.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        record_panel(cmd, image, extent, bgra, now);
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
    // El panel: cuánto asoma (0 a 1) en este instante, según cuándo se abrió o se cerró.
    float panel_shown(int64_t now) const {
        const float t = std::clamp(static_cast<float>(now - panel_at_) / kPanelSlideNs, 0.0f, 1.0f);
        const float eased = 1.0f - (1.0f - t) * (1.0f - t);
        return panel_open_ ? panel_from_ + (1.0f - panel_from_) * eased : panel_from_ * (1.0f - eased);
    }
    void set_panel(bool open, int64_t now) {
        if (open == panel_open_) return;
        panel_from_ = panel_shown(now);  // desde donde esté: abrirlo a medio esconderse no da un salto
        panel_open_ = open;
        panel_at_ = now;
        touched_at_ = now;
        if (open && has_setup()) {  // por si el fichero cambió desde fuera
            const bool marked = access(data_file("elegir_carpeta").c_str(), F_OK) == 0;
            if (marked != choose_folder_) choose_folder_ = marked, panel_dirty_ = true;
        }
    }
    // <datos>/<name>.
    static std::string data_file(const char* name) {
        const char* data = getenv("NIER_DATA");
        return std::string(data ? data : ".") + "/" + name;
    }
    // ¿La app tiene pantalla de inicio (la que pide la carpeta del juego)? El loader pone NIER_SETUP cuando la carpeta sale
    // de ella. Solo entonces tiene sentido la fila "Cambiar carpeta".
    static bool has_setup() {
        static const bool setup = getenv("NIER_SETUP") != nullptr;
        return setup;
    }
    // Los gestos, una vez por frame presentado.
    // Abrir: el gesto de "atrás" del sistema desde el borde derecho (el de la flecha; Android se queda con ese toque y al
    // juego solo le llega la tecla, que el loader convierte en NnsdkInput::hud_gesture), o un dedo que se posa en la mitad
    // derecha de la pantalla y va hacia la izquierda al menos un octavo de su ancho.
    // Con el panel fuera: un toque en una fila cambia su ajuste; un toque fuera del panel, deslizar hacia la derecha, el
    // mismo gesto del borde o un rato sin tocarlo lo esconden.
    void gestures(VkExtent2D extent, int64_t now) {
        const int w = static_cast<int>(extent.width);
        const int panel_w = w / 3;
        if (NnsdkInput* in = nnsdk::host().input) {
            const uint64_t gesture = in->hud_gesture.load(std::memory_order_acquire);
            if (!gesture_seen_) gesture_seen_ = true, gesture_ = gesture;  // lo de antes de empezar no cuenta
            if (gesture != gesture_) {
                gesture_ = gesture;
                // Si ese mismo toque ya lo abrió deslizando (el sistema tardó en quedárselo), no lo cierra.
                if (!(panel_open_ && opened_by_ == static_cast<uint32_t>(gesture >> 32))) set_panel(!panel_open_, now);
            }
        }
        Touch t;
        if (!read_touch(&t)) return;
        if (!touch_seen_) touch_seen_ = true, touch_handled_ = t.serial;
        if (t.serial != touch_handled_) {
            const int dx = t.x - t.sx, dy = t.y - t.sy;
            const bool ended = t.state != 1;
            if (t.state == 2) {
                touch_handled_ = t.serial;
            } else if (!panel_open_) {
                if (t.sx > w / 2 && dx < -w / 8 && std::abs(dy) * 2 < -dx) {
                    set_panel(true, now);
                    opened_by_ = t.serial;
                    touch_handled_ = t.serial;  // al levantar este mismo dedo no pasa nada más
                } else if (ended) {
                    touch_handled_ = t.serial;
                }
            } else if (dx > w / 12 && std::abs(dy) * 2 < dx) {
                set_panel(false, now);
                touch_handled_ = t.serial;
            } else if (ended) {
                if (std::abs(dx) < 32 && std::abs(dy) < 32) {
                    if (t.x < w - panel_w) {
                        set_panel(false, now);
                    } else {
                        touched_at_ = now;
                        const int row = t.y >= kPanelHeader ? (t.y - kPanelHeader) / kPanelRow : -1;
                        if (row == 0) set_hidden(!hidden_);  // el contador de FPS
                        else if (row == 1 && has_setup()) set_choose_folder(!choose_folder_);  // cambiar de carpeta
                    }
                }
                touch_handled_ = t.serial;
            }
        }
        if (panel_open_ && now - touched_at_ > kPanelIdleNs) set_panel(false, now);
    }
    void set_hidden(bool hidden) {
        hidden_ = hidden;
        panel_dirty_ = true;
        const std::string file = data_file("sin_fps");
        if (!hidden) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: contador de FPS %s desde el panel\n", hidden ? "apagado" : "encendido");
    }
    // "Cambiar carpeta": con el fichero <datos>/elegir_carpeta, la pantalla de inicio de la app (que normalmente arranca el
    // juego directamente) vuelve a salir la próxima vez que se abra, para elegir otra carpeta; ella misma borra el fichero.
    void set_choose_folder(bool on) {
        choose_folder_ = on;
        panel_dirty_ = true;
        const std::string file = data_file("elegir_carpeta");
        if (!on) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: cambiar de carpeta al abrir la app: %s\n", on ? "sí" : "no");
    }
    void record_panel(VkCommandBuffer cmd, VkImage image, VkExtent2D extent, bool bgra, int64_t now) {
        const uint32_t w = extent.width / 3, h = extent.height;
        if (w < 360 || h < static_cast<uint32_t>(kPanelHeader + kPanelRow)) return;
        const uint32_t visible = static_cast<uint32_t>(panel_shown(now) * w + 0.5f);
        if (!visible) return;
        // Dos buffers del tamaño del panel: se redibuja en el otro solo cuando cambia lo que enseña, así las copias que
        // la GPU aún tenga pendientes siguen leyendo uno que nadie toca (entre dos cambios pasa al menos un toque).
        if (panel_buffers_.empty() || panel_w_ != w || panel_h_ != h) {
            if (!panel_buffers_.empty() || !create(panel_buffers_, w, h, 2)) return;  // otro tamaño: no pasa (el swapchain no cambia)
            panel_w_ = w, panel_h_ = h, panel_dirty_ = true;
        }
        if (hidden_ != panel_drawn_hidden_) panel_dirty_ = true;  // lo apagaron con el fichero, desde fuera
        if (panel_dirty_) {
            panel_dirty_ = false;
            panel_drawn_hidden_ = hidden_;
            panel_next_ ^= 1;
            draw_panel(reinterpret_cast<Rgba*>(panel_buffers_[panel_next_].data), bgra);
        }
        // Asoma por el borde derecho: las `visible` primeras columnas del panel, pegadas al borde.
        VkBufferImageCopy region{};
        region.bufferRowLength = w;
        region.bufferImageHeight = h;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {static_cast<int32_t>(extent.width - visible), 0, 0};
        region.imageExtent = {visible, h, 1};
        vkCmdCopyBufferToImage(cmd, panel_buffers_[panel_next_].buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    // La cabecera y una fila por ajuste: su nombre y un interruptor (verde con el pomo a la derecha, encendido; gris con el
    // pomo a la izquierda, apagado).
    void draw_panel(Rgba* px, bool bgra) const {
        const int w = static_cast<int>(panel_w_), h = static_cast<int>(panel_h_);
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= w || y >= h) return;
            px[y * w + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int rw, int rh, Rgba c) {
            for (int y = y0; y < y0 + rh; y++)
                for (int x = x0; x < x0 + rw; x++) put(x, y, c);
        };
        const auto pill = [&](int x0, int y0, int pw, int ph, Rgba c) {  // rectángulo con los extremos redondos
            const int r = ph / 2;
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    const int cx = x < r ? r : x >= pw - r ? pw - r - 1 : x, ex = x - cx, ey = y - r;
                    if (ex * ex + ey * ey <= r * r) put(x0 + x, y0 + y, c);
                }
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
        const bool es = spanish();
        const int margin = 44;
        rect(0, 0, w, h, kBack);
        rect(0, 0, 4, h, kEdge);  // el canto que da al juego
        text(margin, (kPanelHeader - 7 * 6) / 2, 6, es ? "AJUSTES" : "SETTINGS", kText);
        rect(4, kPanelHeader - 3, w - 4, 3, kEdge);
        // Fila 0: el contador de FPS. Fila 1 (solo con pantalla de inicio): volver a elegir la carpeta del juego.
        struct Row {
            const char* es;
            const char* en;
            const char* hint_es;  // una línea pequeña debajo (o nullptr)
            const char* hint_en;
            bool on;
        };
        const Row rows[] = {{"CONTADOR DE FPS", "FPS COUNTER", nullptr, nullptr, !hidden_},
                            {"CAMBIAR CARPETA", "CHANGE FOLDER", "AL ABRIR LA APP", "AT NEXT START", choose_folder_}};
        const int count = has_setup() ? 2 : 1;
        int y = kPanelHeader;
        for (int k = 0; k < count; k++) {
            const Row& row = rows[k];
            const int sw = 132, sh = 60, sx = w - sw - margin, sy = y + (kPanelRow - sh) / 2;
            if (row.hint_es) {
                text(margin, y + 30, 4, es ? row.es : row.en, kText);
                text(margin, y + 30 + 7 * 4 + 14, 3, es ? row.hint_es : row.hint_en, kDim);
            } else {
                text(margin, y + (kPanelRow - 7 * 4) / 2, 4, es ? row.es : row.en, kText);
            }
            pill(sx, sy, sw, sh, row.on ? kGood : kOff);
            pill(row.on ? sx + sw - sh + 5 : sx + 5, sy + 5, sh - 10, sh - 10, kKnob);
            rect(margin, y + kPanelRow - 2, w - 2 * margin, 2, kLine);
            y += kPanelRow;
        }
        // Al pie, cómo se cierra.
        text(margin, h - 70, 3, es ? "TOCA FUERA PARA CERRAR" : "TAP OUTSIDE TO CLOSE", kDim);
    }
    bool create(std::vector<Buffer>& buffers, uint32_t width, uint32_t height, uint32_t count = kRing) {
        VkContext* c = vk();
        for (uint32_t k = 0; k < count; k++) {
            Buffer b;
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = VkDeviceSize{width} * height * 4;
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
            buffers.push_back(b);
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

    std::vector<Buffer> buffers_, panel_buffers_;
    uint32_t next_ = 0, panel_next_ = 0;
    bool failed_ = false;
    bool hidden_ = false, flag_ = false, flag_seen_ = false;  // el contador apagado; lo último que dijo <datos>/sin_fps
    bool panel_open_ = false, touch_seen_ = false, gesture_seen_ = false;
    bool panel_dirty_ = true, panel_drawn_hidden_ = false;  // hay que redibujarlo; el estado con el que se dibujó
    bool choose_folder_ = false;                   // la fila "Cambiar carpeta" (el fichero <datos>/elegir_carpeta)
    uint32_t panel_w_ = 0, panel_h_ = 0;           // el tamaño con el que se crearon sus buffers
    float panel_from_ = 0.0f;                     // lo que asomaba cuando se abrió o se cerró por última vez
    int64_t panel_at_ = 0, touched_at_ = 0;       // cuándo fue eso, y la última vez que se tocó
    uint32_t touch_handled_ = 0;                  // el toque (NnsdkInput::touch_serial) ya atendido
    uint32_t opened_by_ = 0;                      // el toque que lo abrió deslizando
    uint64_t gesture_ = 0;                        // el último NnsdkInput::hud_gesture atendido
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
