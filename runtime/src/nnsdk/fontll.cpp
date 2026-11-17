// nn::fontll::ScalableFontEngine: el rasterizador de fuentes escalables del SDK (en la consola, el motor de
// Monotype dentro de sdk). Catherine: Full Body (cvFont_switch.cpp, main+0x6c8f8..0x6e804) lo usa directamente
// para su texto con dos fuentes de su RomFS: font/Catherine.ttf (TrueType) y font/FOT-CezannePro-DB.otf (OpenType
// con contornos CFF). Aquí rasteriza stb_truetype (dominio público, src/third), que entiende los dos formatos.
//
// Lo que el juego hace con él, leído en su código:
//   Initialize(trabajo de 1 MB) → LoadFont(nombre de salida [16 bytes], datos, índice 0, 16) → SetFont(nombre)
//   → SetScale(s, 0, 0, s) con s en 16.16 (26.0 al cargar; luego el tamaño de cada texto) → SetBoldWeight
//   → GetFontMetrics: lee Metrics+4 (ascendente, unidades de fuente, con signo) y Metrics+0x930 (unidades por em,
//     sin signo); su cociente por el tamaño en píxeles es la línea base.
//   CheckGlyphExist(código) → bool (bit 0).
//   AcquireGlyphmap(código, 8) → mapa de grises de 8 bits que lee así (main+0x6d5ac):
//     +0x0E alto sobre la línea base (s16), +0x10 avance en píxeles (s16), +0x1C ancho, +0x1E alto (s16), y los
//     píxeles desde +0x28, fila a fila con el ancho como paso. Lo devuelve con ReleasesGlyph.
// El resto de campos del mapa siguen el FS_GLYPHMAP de Monotype hasta donde se sabe (tamaño total en +0, izquierda en
// +0x0C, avance en 16.16 en +0x14, bytes por fila en +0x20, bits por píxel en +0x22); el juego no los lee.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "third/stb_truetype.h"

namespace nn::fontll {

struct Metrics;

class ScalableFontEngine {
public:
    ScalableFontEngine();
    ~ScalableFontEngine();
    int Initialize(void* work, uint32_t size);
    int Finalize();
    int LoadFont(char* name, const void* data, uint32_t index, uint32_t name_size);
    int SetFont(char* name);
    int SetScale(int s00, int s01, int s10, int s11);
    int SetBoldWeight(int weight);
    int GetFontMetrics(Metrics* out);
    bool CheckGlyphExist(uint32_t code);
    void* AcquireGlyphmap(uint32_t code, uint16_t map_type);
    int ReleasesGlyph(void* glyph);
};

namespace {

struct Font {
    stbtt_fontinfo info{};
    int units_per_em = 1000;
};

struct Engine {
    std::mutex m;
    std::map<std::string, std::unique_ptr<Font>> fonts;
    Font* current = nullptr;
    int scale_x = 26 << 16, scale_y = 26 << 16;  // 16.16, píxeles por em
    int bold = 0;                                 // 16.16, fracción del em
};

std::mutex g_engines_m;
std::unordered_map<const void*, std::unique_ptr<Engine>> g_engines;
int g_font_serial = 0;

Engine& engine_of(const void* self) {
    std::lock_guard lock(g_engines_m);
    auto& e = g_engines[self];
    if (!e) e = std::make_unique<Engine>();
    return *e;
}

// FS_GLYPHMAP tal como lo lee el juego (ver arriba).
struct GlyphMap {
    int32_t size;        // +0x00: bytes del mapa con sus píxeles
    int16_t embedded;    // +0x04
    int16_t pad0;        // +0x06
    int32_t pad1;        // +0x08
    int16_t lo_x;        // +0x0C: izquierda del mapa respecto al origen
    int16_t hi_y;        // +0x0E: arriba del mapa sobre la línea base
    int16_t dx;          // +0x10: avance horizontal en píxeles
    int16_t dy;          // +0x12
    int32_t i_dx;        // +0x14: avance en 16.16
    int32_t i_dy;        // +0x18
    int16_t width;       // +0x1C
    int16_t height;      // +0x1E
    int16_t bpl;         // +0x20: bytes por fila
    int16_t bits_per_pixel;  // +0x22
    int32_t type;        // +0x24: tipo de mapa pedido
    uint8_t bits[];      // +0x28
};
static_assert(offsetof(GlyphMap, hi_y) == 0x0E && offsetof(GlyphMap, dx) == 0x10 &&
              offsetof(GlyphMap, width) == 0x1C && offsetof(GlyphMap, height) == 0x1E &&
              offsetof(GlyphMap, bits) == 0x28);

// Negrita: ensancha cada fila con el máximo de una ventana de `extra` píxeles (con la fracción del último),
// lo que hace el emboldening de Monotype a efectos prácticos en tamaños de texto de interfaz.
std::vector<uint8_t> embolden(const std::vector<uint8_t>& src, int w, int h, float extra, int* out_w) {
    const int add = static_cast<int>(std::ceil(extra));
    const float frac = extra - static_cast<float>(add - 1);
    const int nw = w + add;
    std::vector<uint8_t> dst(static_cast<size_t>(nw) * h, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < nw; x++) {
            float v = 0;
            for (int k = 0; k <= add; k++) {
                const int sx = x - k;
                if (sx < 0 || sx >= w) continue;
                float c = src[static_cast<size_t>(y) * w + sx];
                if (k == add) c *= frac;
                v = std::max(v, c);
            }
            dst[static_cast<size_t>(y) * nw + x] = static_cast<uint8_t>(std::min(255.0f, v));
        }
    }
    *out_w = nw;
    return dst;
}

}  // namespace

ScalableFontEngine::ScalableFontEngine() { engine_of(this); }
ScalableFontEngine::~ScalableFontEngine() {
    std::lock_guard lock(g_engines_m);
    g_engines.erase(this);
}

int ScalableFontEngine::Initialize(void*, uint32_t) { return 0; }
int ScalableFontEngine::Finalize() {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    e.fonts.clear();
    e.current = nullptr;
    return 0;
}

// El tamaño de los datos no viene: stb_truetype lo saca de las tablas de la propia fuente.
int ScalableFontEngine::LoadFont(char* name, const void* data, uint32_t index, uint32_t name_size) {
    Engine& e = engine_of(this);
    auto font = std::make_unique<Font>();
    const auto* bytes = static_cast<const unsigned char*>(data);
    const int offset = stbtt_GetFontOffsetForIndex(bytes, static_cast<int>(index));
    if (offset < 0 || !stbtt_InitFont(&font->info, bytes, offset)) {
        printf("[fontll] LoadFont: datos de fuente no válidos en %p (índice %u)\n", data, index);
        if (name && name_size) name[0] = 0;
        return -1;
    }
    const float per_em = stbtt_ScaleForMappingEmToPixels(&font->info, 1.0f);
    font->units_per_em = per_em > 0 ? static_cast<int>(std::lround(1.0 / per_em)) : 1000;
    char buf[16];
    snprintf(buf, sizeof buf, "fuente%d", ++g_font_serial);
    if (name && name_size) {
        strncpy(name, buf, name_size - 1);
        name[name_size - 1] = 0;
    }
    int ascent = 0, descent = 0, gap = 0;
    stbtt_GetFontVMetrics(&font->info, &ascent, &descent, &gap);
    printf("[fontll] LoadFont %s: %u unidades por em, ascendente %d, descendente %d, %s\n", buf, font->units_per_em,
           ascent, descent, font->info.cff.size ? "CFF" : "TrueType");
    std::lock_guard lock(e.m);
    e.current = font.get();
    e.fonts[name && name_size ? std::string(name) : std::string(buf)] = std::move(font);
    return 0;
}

int ScalableFontEngine::SetFont(char* name) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    auto it = e.fonts.find(name ? name : "");
    if (it == e.fonts.end()) return -1;
    e.current = it->second.get();
    return 0;
}

int ScalableFontEngine::SetScale(int s00, int, int, int s11) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    e.scale_x = s00;
    e.scale_y = s11;
    return 0;
}

int ScalableFontEngine::SetBoldWeight(int weight) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    if (weight != e.bold) {
        static int logged = 0;
        if (logged++ < 8) printf("[fontll] SetBoldWeight %d\n", weight);
    }
    e.bold = weight;
    return 0;
}

int ScalableFontEngine::GetFontMetrics(Metrics* out) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    if (!e.current) return -1;
    int ascent = 0, descent = 0, gap = 0;
    stbtt_GetFontVMetrics(&e.current->info, &ascent, &descent, &gap);
    auto* m = reinterpret_cast<uint8_t*>(out);
    const int16_t a = static_cast<int16_t>(ascent), d = static_cast<int16_t>(descent), g = static_cast<int16_t>(gap);
    const uint16_t upem = static_cast<uint16_t>(e.current->units_per_em);
    memcpy(m + 0, &upem, 2);
    memcpy(m + 4, &a, 2);
    memcpy(m + 6, &d, 2);
    memcpy(m + 8, &g, 2);
    memcpy(m + 0x930, &upem, 2);
    return 0;
}

bool ScalableFontEngine::CheckGlyphExist(uint32_t code) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    return e.current && stbtt_FindGlyphIndex(&e.current->info, static_cast<int>(code)) != 0;
}

void* ScalableFontEngine::AcquireGlyphmap(uint32_t code, uint16_t map_type) {
    Engine& e = engine_of(this);
    std::lock_guard lock(e.m);
    if (!e.current) return nullptr;
    if (map_type != 8) {
        static int logged = 0;
        if (logged++ < 4) printf("[fontll] AcquireGlyphmap: tipo de mapa %u (se da el de 8 bits)\n", map_type);
    }
    const stbtt_fontinfo* info = &e.current->info;
    const int glyph = stbtt_FindGlyphIndex(info, static_cast<int>(code));
    const float px_x = e.scale_x / 65536.0f, px_y = e.scale_y / 65536.0f;
    const float sx = stbtt_ScaleForMappingEmToPixels(info, px_x), sy = stbtt_ScaleForMappingEmToPixels(info, px_y);
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    stbtt_GetGlyphBitmapBox(info, glyph, sx, sy, &x0, &y0, &x1, &y1);
    int w = std::max(0, x1 - x0), h = std::max(0, y1 - y0);
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h);
    if (w && h) stbtt_MakeGlyphBitmap(info, pixels.data(), w, h, w, sx, sy, glyph);
    int advance = 0, lsb = 0;
    stbtt_GetGlyphHMetrics(info, glyph, &advance, &lsb);
    float adv = advance * sx;
    const float extra = e.bold / 65536.0f * px_x;
    if (w && h && extra >= 0.25f) {
        pixels = embolden(pixels, w, h, extra, &w);
        adv += extra;
    }
    const size_t bytes = sizeof(GlyphMap) + pixels.size();
    auto* g = static_cast<GlyphMap*>(calloc(1, bytes));
    if (!g) return nullptr;
    g->size = static_cast<int32_t>(bytes);
    g->lo_x = static_cast<int16_t>(x0);
    g->hi_y = static_cast<int16_t>(-y0);
    g->dx = static_cast<int16_t>(std::lround(adv));
    g->i_dx = static_cast<int32_t>(std::lround(adv * 65536.0f));
    g->width = static_cast<int16_t>(w);
    g->height = static_cast<int16_t>(h);
    g->bpl = static_cast<int16_t>(w);
    g->bits_per_pixel = 8;
    g->type = map_type;
    if (!pixels.empty()) memcpy(g->bits, pixels.data(), pixels.size());
    return g;
}

int ScalableFontEngine::ReleasesGlyph(void* glyph) {
    free(glyph);
    return 0;
}

}  // namespace nn::fontll
