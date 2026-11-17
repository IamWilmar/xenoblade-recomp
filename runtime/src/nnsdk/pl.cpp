// nn::pl: las fuentes del sistema ("shared fonts"). En la consola el servicio pl:u las deja descifradas en una memoria
// compartida y el sdk da, de cada una, si ya está cargada, su dirección y su tamaño (sdk+0x359dc0, +0x359e50 y
// +0x359ff0): un archivo TrueType entero, listo para un rasterizador.
//
// Unity las usa como las fuentes del sistema operativo de su texto dinámico (lo que no viene en la fuente del juego):
// main+0x3e4390 traduce el nombre con el que las conoce a su SharedFontType, espera en un bucle de SleepThread(100 ms)
// a que GetSharedFontLoadState diga 1 (main+0x3e4478…0x3e44a0: con otra respuesta el hilo no sale nunca de ahí),
// y le pasa dirección y tamaño a FT_New_Memory_Face de su FreeType (main+0x3e65d0), que se queda con el puntero. Pide
// las seis, cada una con su nombre (main+0xb59a88…0xb5a2ac):
//   0 Standard               "Switch_Font_STD"            japonés, latino, griego y cirílico
//   1 ChineseSimple          "Switch_Font_CN_Simplified"
//   2 ChineseSimpleExtension "Switch_Font_CN_Extended"
//   3 ChineseTraditional     "Switch_Font_CN_Traditional"
//   4 Korean                 "Switch_Font_KR"
//   5 NintendoExtension      "Switch_Font_NIN_Extended"   los iconos de botones de Nintendo (zona de uso privado)
//
// Aquí cada una es la fuente de Android que cubre lo mismo, mapeada del archivo (sin copia: páginas del propio
// archivo): Roboto para la estándar, las caras SC, TC y KR de Noto Sans CJK para las de chino y coreano, y los
// símbolos de Noto para la de Nintendo, que no tiene equivalente (sus iconos de botones no existen en Android: esos
// caracteres salen sin glifo). El FreeType del juego trae los módulos truetype y cff (sus nombres, main+0x3bbeb04 y
// main+0x3c766f5), así que le valen tanto Roboto como Noto Sans CJK, que es OpenType con contornos CFF.
//
// Noto Sans CJK viene como colección (.ttc, una cara por idioma) y cada fuente de la consola es un archivo de una sola
// cara: Unity recorre todas las caras del archivo que se le da (guarda face->num_faces, main+0xb5940c), así que con la
// colección entera cada tipo traería las de todos los idiomas. Se le da la cara que toca como archivo suelto: el directorio de tablas
// de esa cara se copia al principio del mapeo privado (los offsets de una colección ya son absolutos y siguen valiendo)
// y solo esa primera página deja de ser del archivo.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>

namespace nn::pl {

enum SharedFontType : int {
    SharedFontType_Standard = 0,
    SharedFontType_ChineseSimple = 1,
    SharedFontType_ChineseSimpleExtension = 2,
    SharedFontType_ChineseTraditional = 3,
    SharedFontType_Korean = 4,
    SharedFontType_NintendoExtension = 5,
    SharedFontType_Max = 6,
};
enum SharedFontLoadState : int { SharedFontLoadState_Loading = 0, SharedFontLoadState_Loaded = 1 };

namespace {

struct Source {
    const char* path;
    int face;  // la cara dentro de una colección .ttc; en un archivo suelto no se mira
};

// El orden de las caras de NotoSansCJK-Regular.ttc es el del fonts.xml de Android: 0 japonés, 1 coreano, 2 chino
// simplificado, 3 chino tradicional.
constexpr char kCjk[] = "/system/fonts/NotoSansCJK-Regular.ttc";
constexpr Source kStandard[] = {{"/system/fonts/Roboto-Regular.ttf", 0}, {"/system/fonts/RobotoStatic-Regular.ttf", 0},
                                {"/system/fonts/DroidSans.ttf", 0}, {kCjk, 0}};
constexpr Source kChineseSimple[] = {{kCjk, 2}};
constexpr Source kChineseTraditional[] = {{kCjk, 3}};
constexpr Source kKorean[] = {{kCjk, 1}};
constexpr Source kSymbols[] = {{"/system/fonts/NotoSansSymbols-Regular-Subsetted.ttf", 0},
                               {"/system/fonts/NotoSansSymbols-Regular-Subsetted2.ttf", 0}};

struct Sources {
    const Source* first;
    size_t count;
};
constexpr Sources kSources[SharedFontType_Max] = {
    {kStandard, std::size(kStandard)},
    {kChineseSimple, std::size(kChineseSimple)},
    {kChineseSimple, std::size(kChineseSimple)},
    {kChineseTraditional, std::size(kChineseTraditional)},
    {kKorean, std::size(kKorean)},
    {kSymbols, std::size(kSymbols)},
};

struct Font {
    std::once_flag once;
    const void* data = nullptr;
    size_t size = 0;
};
Font g_fonts[SharedFontType_Max];

uint16_t be16(const uint8_t* p) { return static_cast<uint16_t>(p[0] << 8 | p[1]); }
uint32_t be32(const uint8_t* p) { return uint32_t{p[0]} << 24 | uint32_t{p[1]} << 16 | uint32_t{p[2]} << 8 | p[3]; }

// Deja en [data, data + size) (un mapeo privado del archivo, de solo lectura) una fuente de una sola cara. Un archivo
// suelto ya lo es; de una colección ("ttcf", número de caras en +8 y el offset del directorio de cada una desde +12)
// se sube al principio el directorio de la cara pedida (12 bytes y 16 por tabla: etiqueta, suma, offset absoluto y
// largo). Solo vale si ninguna tabla de esa cara empieza donde va a quedar el directorio; en las colecciones los
// directorios van todos antes que los datos.
bool keep_one_face(uint8_t* data, size_t size, int face) {
    if (size < 12) return false;
    if (memcmp(data, "ttcf", 4) == 0) {
        const uint32_t faces = be32(data + 8);
        if (faces == 0 || size < 12 + 4 * uint64_t{faces}) return false;
        const uint64_t dir = be32(data + 12 + 4 * std::min<uint32_t>(static_cast<uint32_t>(face), faces - 1));
        if (dir + 12 > size) return false;
        const uint64_t dir_size = 12 + 16 * uint64_t{be16(data + dir + 4)};
        if (dir + dir_size > size) return false;
        for (uint64_t record = dir + 12; record < dir + dir_size; record += 16) {
            const uint64_t offset = be32(data + record + 8), length = be32(data + record + 12);
            if (offset < dir_size || offset + length > size) return false;
        }
        // Solo las páginas del principio pasan a ser copia propia; el resto sigue siendo el archivo.
        const size_t span = std::min<size_t>(size, (dir_size + 0xfff) & ~size_t{0xfff});
        if (mprotect(data, span, PROT_READ | PROT_WRITE) != 0) return false;
        memmove(data, data + dir, dir_size);
        mprotect(data, span, PROT_READ);
    }
    // Lo que FreeType reconoce como sfnt: TrueType (0x00010000 o "true") u OpenType con CFF ("OTTO").
    const uint32_t version = be32(data);
    return version == 0x00010000 || version == 0x74727565 || version == 0x4F54544F;
}

bool map_font(const Source& source, Font* font) {
    const int fd = open(source.path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st {};
    void* map = MAP_FAILED;
    if (fstat(fd, &st) == 0 && st.st_size > 0)
        map = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return false;
    const size_t size = static_cast<size_t>(st.st_size);
    if (!keep_one_face(static_cast<uint8_t*>(map), size, source.face)) {
        munmap(map, size);
        return false;
    }
    font->data = map;
    font->size = size;
    return true;
}

void load(int type) {
    Font& font = g_fonts[type];
    // Si falta la de un tipo (un Android sin Noto Sans CJK), la estándar: una fuente válida, aunque sin esos glifos.
    const Sources lists[] = {kSources[type], kSources[SharedFontType_Standard]};
    for (size_t list = 0; list < (type == SharedFontType_Standard ? 1u : 2u); list++) {
        for (size_t i = 0; i < lists[list].count; i++) {
            const Source& source = lists[list].first[i];
            if (!map_font(source, &font)) continue;
            printf("[pl] fuente compartida %d: %s (cara %d), %zu KB\n", type, source.path, source.face, font.size / 1024);
            return;
        }
    }
    // Sin dirección FT_New_Memory_Face devuelve Invalid_Argument (main+0x3e6610) y Unity se queda sin esa fuente.
    printf("[pl] fuente compartida %d: Android no tiene ninguna de las previstas\n", type);
}

// Cargada al preguntar por ella la primera vez. Un tipo que no existe no tiene fuente (en la consola el sdk aborta).
const Font* font(SharedFontType type) {
    if (type < 0 || type >= SharedFontType_Max) return nullptr;
    std::call_once(g_fonts[type].once, load, static_cast<int>(type));
    return &g_fonts[type];
}

}  // namespace

// Nunca "cargando": la carga es síncrona, y con Loading Unity se queda esperando.
SharedFontLoadState GetSharedFontLoadState(SharedFontType type) {
    font(type);
    return SharedFontLoadState_Loaded;
}

const void* GetSharedFontAddress(SharedFontType type) {
    const Font* f = font(type);
    return f ? f->data : nullptr;
}

size_t GetSharedFontSize(SharedFontType type) {
    const Font* f = font(type);
    return f ? f->size : 0;
}

}  // namespace nn::pl
