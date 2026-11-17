// nn::ngc: el filtro de palabras malsonantes. Los diccionarios son un archivo del sistema de la consola que aquí no
// existe: ninguna palabra está en ellos y nada se enmascara por ellos. Lo que el filtro hace sin diccionario sí está:
// las comprobaciones de argumentos con sus resultados, la regla de la arroba de CheckProfanityWords y CountNumbers.
//
// El juego reserva el objeto (0x30 bytes, con su operator new) y solo importa el constructor, Initialize y los ajustes;
// el resto lo llama por la vtable: +0x08 el destructor que borra, +0x10 GetContentVersion, +0x28 CheckProfanityWords y
// +0x38 MaskProfanityWordsInText en UTF-16 (el plugin de Unity del SDK, main+0x155c880…0x155c8f0; MH Rise usaba las de
// char16_t, +0x30 y +0x40). La clase tiene las funciones virtuales del SDK en su orden, así la vtable que pone nuestro
// constructor tiene las mismas posiciones, y el delete va al allocator del juego. Los campos van donde los tiene el SDK
// (los escriben SetMaskMode en +0x28 y SkipAtSignCheck en +0x0c, sdk+0x31e088 y sdk+0x31e080).
//
// En Disco Elysium todo esto solo se alcanza desde el plugin (nn.ngc.*), que no tiene llamadores en el juego.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::settings {
struct LanguageCode {  // como en settings.cpp
    char string[8];
};
}  // namespace nn::settings
namespace nn::oe {
settings::LanguageCode GetDesiredLanguage();
}

namespace nn::ngc {

// Oculta salvo lo que importa main: las virtuales solo se alcanzan por la vtable.
class [[gnu::visibility("hidden")]] ProfanityFilter {
public:
    enum MaskMode : int { MaskMode_OverWrite = 0, MaskMode_ReplaceByOneCharacter = 1 };
    enum SkipMode : int { SkipMode_NotSkip = 0, SkipMode_SkipAtSignCheck = 1 };

    [[gnu::visibility("default")]] ProfanityFilter();
    virtual ~ProfanityFilter();
    virtual uint32_t GetContentVersion();
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const char** words, size_t count);
    virtual Result MaskProfanityWordsInText(int* count, char* text, uint32_t patterns);
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const uint16_t** words, size_t count);
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const char16_t** words, size_t count);
    virtual Result MaskProfanityWordsInText(int* count, uint16_t* text, uint32_t patterns);
    virtual Result MaskProfanityWordsInText(int* count, char16_t* text, uint32_t patterns);

    [[gnu::visibility("default")]] Result Initialize(void* work, unsigned long size, bool check_desired_language);
    [[gnu::visibility("default")]] Result Initialize(void* work, unsigned long size);
    [[gnu::visibility("default")]] void Finalize();
    [[gnu::visibility("default")]] void SetMaskMode(MaskMode mode);
    [[gnu::visibility("default")]] void SkipAtSignCheck(SkipMode mode);

    static void operator delete(void* p) { nnsdk::host().game_delete(p); }

private:
    bool check_desired_language_ = false;                       // +0x08
    bool initialized_ = false;                                  // +0x09
    SkipMode skip_at_sign_ = SkipMode_NotSkip;                  // +0x0c
    [[maybe_unused]] uint32_t current_pattern_ = 0;             // +0x10: el patrón de la pasada en curso del SDK
    void* work_ = nullptr;                                      // +0x18
    [[maybe_unused]] const void* contents_ = nullptr;           // +0x20: los diccionarios montados
    MaskMode mask_mode_ = MaskMode_OverWrite;                   // +0x28
};
static_assert(sizeof(ProfanityFilter) == 0x30);

namespace {

// Módulo ngc (146), los que devuelve el propio sdk antes de mirar ningún diccionario (Initialize sdk+0x31ddd4,
// CheckProfanityWords sdk+0x31ec44, MaskProfanityWordsInText sdk+0x31fe30).
constexpr Result kResultNotInitialized{146 | (1 << 9)};
constexpr Result kResultAlreadyInitialized{146 | (2 << 9)};
constexpr Result kResultInvalidPointer{146 | (3 << 9)};
constexpr Result kResultInvalidSize{146 | (4 << 9)};      // memoria de trabajo corta, o más de 16 palabras
constexpr Result kResultInvalidEncoding{146 | (5 << 9)};  // el texto no es UTF-8 / UTF-16 válido

constexpr size_t kWorkMemorySize = 0x20000;  // lo mínimo que acepta Initialize
constexpr size_t kMaxWords = 16;
constexpr size_t kWordLength = 64;  // lo que el SDK mira de cada palabra (la convierte a un búfer de 64)

// Siguiente carácter de un UTF-8 estricto (sin sobrelargos, sustitutos ni nada más allá de U+10FFFF). false si no vale.
bool next_utf8(const uint8_t*& p, uint32_t* out) {
    const uint8_t c = p[0];
    if (c < 0x80) {
        *out = c;
        p += 1;
        return true;
    }
    const int extra = c >= 0xc2 && c < 0xe0 ? 1 : c >= 0xe0 && c < 0xf0 ? 2 : c >= 0xf0 && c < 0xf5 ? 3 : 0;
    if (!extra) return false;
    uint32_t cp = c & (0x3f >> extra);
    for (int i = 1; i <= extra; i++) {
        if ((p[i] & 0xc0) != 0x80) return false;  // también corta en el 0 final
        cp = cp << 6 | (p[i] & 0x3f);
    }
    static constexpr uint32_t kMin[] = {0, 0x80, 0x800, 0x10000};
    if (cp < kMin[extra] || cp > 0x10ffff || (cp >= 0xd800 && cp < 0xe000)) return false;
    *out = cp;
    p += 1 + extra;
    return true;
}

bool valid(const char* text) {
    auto* p = reinterpret_cast<const uint8_t*>(text);
    uint32_t cp;
    while (*p)
        if (!next_utf8(p, &cp)) return false;
    return true;
}

// UTF-16: un sustituto alto sin su bajo detrás (o un bajo suelto) no vale.
template <typename Char>
bool valid(const Char* text) {
    for (; *text; text++) {
        const uint32_t c = static_cast<uint16_t>(*text);
        if (c >= 0xdc00 && c < 0xe000) return false;
        if (c >= 0xd800 && c < 0xdc00) {
            const uint32_t low = static_cast<uint16_t>(text[1]);
            if (low < 0xdc00 || low >= 0xe000) return false;
            text++;
        }
    }
    return true;
}

// La arroba, normal o de ancho completo (ProfanityFilterBase::IsIncludesAtSign, sdk+0x3223ec), en lo que el SDK mira de
// la palabra.
bool has_at_sign(const char* word) {
    auto* w = reinterpret_cast<const uint8_t*>(word);
    for (size_t i = 0; i < kWordLength && w[i]; i++)
        if (w[i] == '@' || (w[i] == 0xef && w[i + 1] == 0xbc && w[i + 2] == 0xa0)) return true;  // U+FF20 en UTF-8
    return false;
}
template <typename Char>
bool has_at_sign(const Char* word) {
    for (size_t i = 0; i < kWordLength && word[i]; i++)
        if (word[i] == 0x40 || static_cast<uint16_t>(word[i]) == 0xff20) return true;
    return false;
}

// El bit de patrón (ProfanityFilterPatternList) del idioma de la consola: la tabla de sdk+0x322580.
uint32_t language_pattern() {
    static constexpr struct {
        const char* code;
        uint32_t bit;
    } kPatterns[] = {{"ja", 1 << 0},      {"en-US", 1 << 1},  {"fr-CA", 1 << 2}, {"es-419", 1 << 3},
                     {"en-GB", 1 << 4},   {"fr", 1 << 5},     {"de", 1 << 6},    {"it", 1 << 7},
                     {"es", 1 << 8},      {"nl", 1 << 9},     {"ko", 1 << 10},   {"zh-Hans", 1 << 11},
                     {"pt", 1 << 12},     {"ru", 1 << 13},    {"pt-BR", 1 << 14}, {"zh-Hant", 1 << 15}};
    const settings::LanguageCode code = oe::GetDesiredLanguage();
    for (const auto& p : kPatterns)
        if (strncmp(code.string, p.code, sizeof code.string) == 0) return p.bit;
    return 0;
}

// Lo que CountNumbers cuenta como número (sdk+0x31dac4…0x31dbf8): las cifras ASCII y de ancho completo, superíndices y
// subíndices, números romanos y las cifras con círculo, paréntesis o punto.
bool is_number(uint32_t c) {
    static constexpr struct {
        uint16_t first, last;
    } kRanges[] = {{0x0030, 0x0039}, {0x00b2, 0x00b3}, {0x00b9, 0x00b9}, {0x2070, 0x2079}, {0x2080, 0x2089},
                   {0x2160, 0x216b}, {0x2170, 0x217b}, {0x2460, 0x249b}, {0x24ea, 0x24ea}, {0x24ff, 0x24ff},
                   {0x2776, 0x2793}, {0xff10, 0xff19}};
    for (const auto& r : kRanges)
        if (c >= r.first && c <= r.last) return true;
    return false;
}

// Sin diccionarios solo queda la regla de la arroba (CheckProfanityWordsImpl, sdk+0x31e7ec): una palabra con arroba
// cuenta como malsonante en el idioma de la consola y en los patrones pedidos (para que nadie comparta una dirección),
// salvo con SkipAtSignCheck.
template <typename Char>
Result check(bool initialized, bool at_sign_rule, uint32_t* results, uint32_t patterns, const Char** words, size_t count) {
    if (!initialized) return kResultNotInitialized;
    if (!results || !words) return kResultInvalidPointer;
    if (count > kMaxWords) return kResultInvalidSize;
    for (size_t i = 0; i < count; i++)
        if (!words[i] || !valid(words[i])) return kResultInvalidEncoding;
    const uint32_t flagged = (language_pattern() | patterns) & 0xffff;
    for (size_t i = 0; i < count; i++) results[i] = at_sign_rule && has_at_sign(words[i]) ? flagged : 0;
    return ResultSuccess;
}

// El SDK tapa las palabras del diccionario y, sin SkipAtSignCheck, lo que parece una dirección de correo (la máquina
// de estados de sdk+0x31f654); esto último no se ha calcado: aquí no se tapa nada y el texto queda como estaba.
template <typename Char>
Result mask(bool initialized, int* count, const Char* text) {
    if (!initialized) return kResultNotInitialized;
    if (!text) return kResultInvalidPointer;
    if (!valid(text)) return kResultInvalidEncoding;
    if (count) *count = 0;  // el juego pasa nullptr
    return ResultSuccess;
}

}  // namespace

// Cuántos caracteres de un texto UTF-8 son números; -1 con un puntero nulo o un texto mal codificado.
int CountNumbers(const char* text) {
    if (!text) return -1;
    auto* p = reinterpret_cast<const uint8_t*>(text);
    int count = 0;
    while (*p) {
        uint32_t cp;
        if (!next_utf8(p, &cp)) return -1;
        count += is_number(cp);
    }
    return count;
}

ProfanityFilter::ProfanityFilter() {}
ProfanityFilter::~ProfanityFilter() { Finalize(); }

// check_desired_language: mirar también el idioma que eligió la aplicación además del de la consola (aquí son el
// mismo). El SDK monta ahora los diccionarios del sistema; sin ellos queda lo demás.
Result ProfanityFilter::Initialize(void* work, unsigned long size, bool check_desired_language) {
    if (!work) return kResultInvalidPointer;
    if (size < kWorkMemorySize) return kResultInvalidSize;
    if (initialized_) return kResultAlreadyInitialized;
    work_ = work;
    skip_at_sign_ = SkipMode_NotSkip;
    mask_mode_ = MaskMode_OverWrite;
    check_desired_language_ = check_desired_language;
    initialized_ = true;
    return ResultSuccess;
}

Result ProfanityFilter::Initialize(void* work, unsigned long size) { return Initialize(work, size, true); }

void ProfanityFilter::Finalize() {
    initialized_ = false;
    work_ = nullptr;
}

// Cómo tapa MaskProfanityWordsInText lo que encuentra: un asterisco por carácter o uno solo por palabra.
void ProfanityFilter::SetMaskMode(MaskMode mode) { mask_mode_ = mode; }
// Si CheckProfanityWords deja de dar por malsonante toda palabra con arroba.
void ProfanityFilter::SkipAtSignCheck(SkipMode mode) { skip_at_sign_ = mode; }

uint32_t ProfanityFilter::GetContentVersion() { return 0; }

Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t patterns, const char** words, size_t count) {
    return check(initialized_, skip_at_sign_ == SkipMode_NotSkip, results, patterns, words, count);
}
Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t patterns, const uint16_t** words, size_t count) {
    return check(initialized_, skip_at_sign_ == SkipMode_NotSkip, results, patterns, words, count);
}
Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t patterns, const char16_t** words, size_t count) {
    return check(initialized_, skip_at_sign_ == SkipMode_NotSkip, results, patterns, words, count);
}
Result ProfanityFilter::MaskProfanityWordsInText(int* count, char* text, uint32_t) { return mask(initialized_, count, text); }
Result ProfanityFilter::MaskProfanityWordsInText(int* count, uint16_t* text, uint32_t) {
    return mask(initialized_, count, text);
}
Result ProfanityFilter::MaskProfanityWordsInText(int* count, char16_t* text, uint32_t) {
    return mask(initialized_, count, text);
}

}  // namespace nn::ngc
