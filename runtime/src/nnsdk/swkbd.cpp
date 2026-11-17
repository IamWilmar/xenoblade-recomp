// nn::swkbd: el teclado de software del sistema. No hay applet ni pantalla para él, así que cada teclado lo "rellena"
// un jugador que siempre acepta lo propuesto:
// - ShowKeyboard (nombre del cazador, del Felyne y del Canyne, main+0x6fa6984): si el juego trae texto inicial se
//   devuelve tal cual (renombrar sin cambiar nada); si no, la siguiente línea de <datos>/nombre.txt (UTF-8, una por
//   teclado, en orden; la última se repite) o "Flip". Se respetan la longitud máxima y mínima del config y, si el juego
//   puso un TextCheckCallback, se le pasa el texto igual que haría el applet; si lo rechaza se prueba "Flip" y, si
//   tampoco, el usuario "cancela".
// - InlineKeyboard (el teclado dentro del juego: el nombre del cazador al crearlo, el chat): se inicializa en el primer
//   Calc como el applet de verdad, y cada Appear se contesta en el Calc siguiente con DecidedEnter: el texto que el
//   juego puso con SetInputText o, si no puso ninguno, el mismo nombre que ShowKeyboard (nombre.txt o "Flip"). Sin
//   callback de DecidedEnter, DecidedCancel.
//
// Los setters de KeyboardConfig escriben los mismos campos que el SDK: el juego también rellena algunos a mano
// (longitud máxima, modo de contraseña… main+0x6fa5f3c) y ShowKeyboard los lee de ahí.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "nnsdk/types.h"

namespace nn::swkbd {

enum Preset : int { Preset_Default = 0, Preset_Password = 1, Preset_UserName = 2, Preset_DownloadCode = 3 };
enum TextCheckResult : int {
    TextCheckResult_Success = 0,
    TextCheckResult_ShowFailureDialog = 1,
    TextCheckResult_ShowConfirmDialog = 2,
};
// Estado del InlineKeyboard (el que devuelve Calc): sin iniciar, oculto, apareciendo, visible, desapareciendo.
enum State : int { State_None = 0, State_Hidden = 1, State_Appearing = 2, State_Shown = 3, State_Disappearing = 4 };

struct String {
    void* ptr;
    size_t size;  // capacidad del búfer en bytes
};
struct UserWord;
struct AppearArg;
struct ChangedStringArg;
struct MovedCursorArg;
// Lo que recibe el callback de DecidedEnter (main+0x6fa7adc): el texto en UTF-16 y su longitud en caracteres.
struct DecidedEnterArg {
    char16_t text[502];
    int32_t length;
};
static_assert(offsetof(DecidedEnterArg, length) == 0x3ec);
struct Rect {
    int16_t x, y, width, height;
};

using TextCheckCallback = TextCheckResult (*)(void* message, size_t* message_size, String* text);

struct KeyboardConfig {
    uint32_t keyboard_mode;
    char16_t ok_text[9];
    char16_t left_optional_symbol_key;
    char16_t right_optional_symbol_key;
    bool is_prediction_enabled;
    uint32_t invalid_char_flag;
    int32_t initial_cursor_pos;
    char16_t header_text[65];
    char16_t sub_text[129];
    char16_t guide_text[257];
    int32_t text_max_length;  // 0x3ac
    int32_t text_min_length;
    uint32_t password_mode;
    uint32_t input_form_mode;
    bool is_use_new_line;
    bool is_use_utf8;  // 0x3bd: el resultado en UTF-8 y no en UTF-16
    bool is_use_blur_background;
    int32_t initial_string_offset;  // 0x3c0: dónde está el texto inicial dentro del work buffer
    int32_t initial_string_length;
    int32_t user_dictionary_offset;
    int32_t user_dictionary_num;
    bool is_use_text_check;  // 0x3d0
    TextCheckCallback text_check_callback;
    int32_t separate_text_pos[8];  // 0x3e0: dónde parte los grupos el código de descarga
    uint32_t customized_dictionaries[24][2];
    uint8_t customized_dictionary_count;
    uint8_t reserved[0xf];
};
static_assert(sizeof(KeyboardConfig) == 0x4d0);
static_assert(offsetof(KeyboardConfig, text_max_length) == 0x3ac);
static_assert(offsetof(KeyboardConfig, initial_string_offset) == 0x3c0);
static_assert(offsetof(KeyboardConfig, text_check_callback) == 0x3d8);
static_assert(offsetof(KeyboardConfig, customized_dictionaries) == 0x400);

struct ShowKeyboardArg {
    KeyboardConfig keyboard_config;
    void* work_buf;
    size_t work_buf_size;
    void* text_check_work_buf;
    size_t text_check_work_buf_size;
    void* customized_dictionary_buf;
    size_t customized_dictionary_buf_size;
};
static_assert(sizeof(ShowKeyboardArg) == 0x500);

struct CustomizedDictionarySet {
    void* buffer;
    uint32_t size;
    uint32_t dictionaries[24][2];
    uint16_t count;
};
static_assert(offsetof(CustomizedDictionarySet, count) == 0xcc);

struct InputText {
    char16_t text[502];
    int32_t cursor_pos;
    int32_t reserved;
    void Set(const char16_t* text);
};
static_assert(sizeof(InputText) == 0x3f4);

struct UserWordInfo {
    static size_t GetRequiredWorkBufferSize(int count);
};

class InlineKeyboard {
public:
    enum WindowMode : int {};

    InlineKeyboard();
    ~InlineKeyboard();
    static size_t GetRequiredWorkBufferSize();
    bool Initialize(void* work_buffer);
    void Finalize();
    State Calc();
    void Appear(const AppearArg& arg);
    void Disappear();
    void SetInputText(const InputText& text);
    void SetCursorPos(int pos);
    void SetUserWordInfo(const UserWordInfo& info);
    void UnsetUserWordInfo();
    void SetCustomizedDictionaries(const CustomizedDictionarySet& set);
    void UnsetCustomizedDictionaries();
    void SetWindowMode(WindowMode mode);
    void SetKeytopAlpha(float alpha);
    void SetFooterAlpha(float alpha);
    void SetKeytopScale(float scale);
    void SetKeytopTranslate(float x, float y);
    void SetTouchEnabled(bool enabled);
    void SetHardwareKeyboardEnabled(bool enabled);
    void SetFinishedInitializeCallback(void (*callback)());
    void SetFinishedKeyboardCallback(void (*callback)());
    void SetChangedStringCallback(void (*callback)(const ChangedStringArg*));
    void SetMovedCursorCallback(void (*callback)(const MovedCursorArg*));
    void SetDecidedEnterCallback(void (*callback)(const DecidedEnterArg*));
    void SetDecidedCancelCallback(void (*callback)());
    void SetReleasedUserWordInfoCallback(void (*callback)());
    int GetMaxHeight();
    bool IsUsedTouchPointByKeyboard(int x, int y);
    void GetWindowSize(int* width, int* height) const;
    int GetTouchRectangles(Rect* first, Rect* second) const;
    void GetImageMemoryRequirement(size_t* size, size_t* alignment);
    bool GetImage(void* buffer, size_t size);

private:
    struct Impl;
    Impl* impl_;
    void* work_;
};
static_assert(sizeof(InlineKeyboard) == 16);

namespace {

constexpr Result kResultCanceled{159 | 1 << 9};
constexpr const char* kDefaultName = "Flip";

std::string data_dir() {
    const char* env = getenv("NIER_DATA");
    return env ? env : "/sdcard/NieR";
}

// Las líneas de <datos>/nombre.txt (sin el salto de línea final ni espacios al final; las vacías no cuentan).
std::vector<std::string> configured_names() {
    std::vector<std::string> names;
    if (FILE* f = fopen((data_dir() + "/nombre.txt").c_str(), "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (!s.empty()) names.push_back(s);
        }
        fclose(f);
    }
    return names;
}

std::u16string utf8_to_utf16(const std::string& in) {
    std::u16string out;
    for (size_t i = 0; i < in.size();) {
        const unsigned char c = in[i];
        const int extra = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
        if (extra < 0 || i + extra >= in.size()) {  // byte suelto o secuencia cortada: se salta
            i++;
            continue;
        }
        uint32_t cp = extra ? c & (0x3F >> extra) : c;
        for (int k = 1; k <= extra; k++) cp = cp << 6 | (in[i + k] & 0x3F);
        i += 1 + extra;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out += char16_t(0xD800 | cp >> 10);
            out += char16_t(0xDC00 | (cp & 0x3FF));
        } else {
            out += char16_t(cp);
        }
    }
    return out;
}

std::string utf16_to_utf8(const char16_t* in, size_t n) {
    std::string out;
    for (size_t i = 0; i < n && in[i]; i++) {
        uint32_t cp = in[i];
        if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < n && in[i + 1] >= 0xDC00 && in[i + 1] < 0xE000)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (in[++i] - 0xDC00);
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | cp >> 6);
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | cp >> 12);
            out += char(0x80 | (cp >> 6 & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | cp >> 18);
            out += char(0x80 | (cp >> 12 & 0x3F));
            out += char(0x80 | (cp >> 6 & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

size_t length16(const char16_t* s, size_t max) {
    size_t n = 0;
    while (s && n < max && s[n]) n++;
    return n;
}

// Copia como los setters del SDK: hasta capacity - 1 caracteres y el 0 final.
void copy_text(char16_t* dst, size_t capacity, const char16_t* src) {
    const size_t n = length16(src, capacity - 1);
    if (n) memcpy(dst, src, n * sizeof(char16_t));
    dst[n] = 0;
}

// SetInitialText escribe el texto en el work buffer, pero MH Rise lo llama antes de reservarlo (main+0x6fa6030, el work
// buffer llega en main+0x6fa61f0): se guarda aquí también, con el arg al que pertenece.
const ShowKeyboardArg* g_initial_arg = nullptr;
std::u16string g_initial;
size_t g_names_used = 0;  // líneas de nombre.txt ya dadas

std::u16string initial_text(const ShowKeyboardArg& arg) {
    const KeyboardConfig& config = arg.keyboard_config;
    if (config.initial_string_length <= 0) return {};
    if (g_initial_arg == &arg) return g_initial;
    const size_t end = size_t(config.initial_string_offset) + size_t(config.initial_string_length) * sizeof(char16_t);
    if (!arg.work_buf || config.initial_string_offset < 0 || end > arg.work_buf_size) return {};
    auto* text =
        reinterpret_cast<const char16_t*>(static_cast<const uint8_t*>(arg.work_buf) + config.initial_string_offset);
    return std::u16string(text, length16(text, size_t(config.initial_string_length)));
}

// Deja el texto en el String del juego, en UTF-16 o UTF-8 según el config, terminado en 0. Devuelve los bytes escritos
// sin contar el 0.
size_t write_result(String* out, const std::u16string& text, bool utf8) {
    auto* dst = static_cast<uint8_t*>(out->ptr);
    if (!dst || out->size < 2) return 0;
    if (utf8) {
        std::string s = utf16_to_utf8(text.data(), text.size());
        const size_t n = std::min(s.size(), out->size - 1);
        memcpy(dst, s.data(), n);
        dst[n] = 0;
        return n;
    }
    const size_t n = std::min(text.size(), out->size / sizeof(char16_t) - 1);
    memmove(dst, text.data(), n * sizeof(char16_t));
    reinterpret_cast<char16_t*>(dst)[n] = 0;
    return n * sizeof(char16_t);
}

}  // namespace

// --- KeyboardConfig: los mismos campos que el SDK ---

void InitializeKeyboardConfig(KeyboardConfig* config) {
    config->initial_string_offset = config->initial_string_length = 0;
    config->user_dictionary_offset = config->user_dictionary_num = 0;
    config->is_use_text_check = false;
    config->text_check_callback = nullptr;
}

// Los cuatro presets del SDK; los campos que no se nombran quedan a cero.
void MakePreset(KeyboardConfig* config, Preset preset) {
    memset(config, 0, sizeof *config);
    config->initial_cursor_pos = 1;  // al final del texto
    config->is_use_blur_background = true;
    for (int32_t& pos : config->separate_text_pos) pos = -1;
    switch (preset) {
        case Preset_Default:
            config->keyboard_mode = 2;  // completo
            config->input_form_mode = 1;
            config->is_use_new_line = true;
            break;
        case Preset_Password:
            config->keyboard_mode = 2;
            config->password_mode = 1;
            break;
        case Preset_UserName:
            config->invalid_char_flag = 0x100;
            break;
        case Preset_DownloadCode:
            config->keyboard_mode = 2;
            config->invalid_char_flag = 0x80;
            config->text_max_length = config->text_min_length = 16;
            config->input_form_mode = 2;
            config->separate_text_pos[0] = 3;
            config->separate_text_pos[1] = 7;
            config->separate_text_pos[2] = 11;
            break;
    }
}

void SetOkText(KeyboardConfig* config, const char16_t* text) { copy_text(config->ok_text, 9, text); }
void SetLeftOptionalSymbolKey(KeyboardConfig* config, char16_t key) { config->left_optional_symbol_key = key; }
void SetRightOptionalSymbolKey(KeyboardConfig* config, char16_t key) { config->right_optional_symbol_key = key; }
void SetHeaderText(KeyboardConfig* config, const char16_t* text) { copy_text(config->header_text, 65, text); }
void SetSubText(KeyboardConfig* config, const char16_t* text) { copy_text(config->sub_text, 129, text); }
void SetGuideText(KeyboardConfig* config, const char16_t* text) { copy_text(config->guide_text, 257, text); }

// Los mismos con el texto en UTF-8: el SDK lo convierte a UTF-16 sobre el mismo campo (SetOkTextUtf8 en +0x4 con sitio
// para 9, SetHeaderTextUtf8 en +0x24 para 65, SetSubTextUtf8 en +0xa6 para 129 y SetGuideTextUtf8 en +0x1a8 para 257:
// sdk+0x3a7290, +0x3a74d0, +0x3a7610 y +0x3a7740), con nullptr lo deja vacío y aborta si el texto no es UTF-8 válido o
// no cabe. Aquí lo que no vale se salta y lo que no cabe se corta: un texto de la interfaz no merece tumbar el juego.
// En Disco Elysium solo los alcanza el plugin de Unity del SDK (nn.swkbd.Swkbd, main+0x155ca40…0x155cae0), que no tiene
// llamadores.
namespace {
std::u16string from_utf8(const char* text) { return text ? utf8_to_utf16(text) : std::u16string(); }
}  // namespace

void SetOkTextUtf8(KeyboardConfig* config, const char* text) { copy_text(config->ok_text, 9, from_utf8(text).c_str()); }
void SetHeaderTextUtf8(KeyboardConfig* config, const char* text) {
    copy_text(config->header_text, 65, from_utf8(text).c_str());
}
void SetSubTextUtf8(KeyboardConfig* config, const char* text) { copy_text(config->sub_text, 129, from_utf8(text).c_str()); }
void SetGuideTextUtf8(KeyboardConfig* config, const char* text) {
    copy_text(config->guide_text, 257, from_utf8(text).c_str());
}
// La tecla de símbolo es un solo carácter UTF-16 (el SDK convierte con sitio para uno, sdk+0x3a7374): el primero del
// texto. Con un texto vacío el campo se queda como estaba, como en el SDK.
void SetLeftOptionalSymbolKeyUtf8(KeyboardConfig* config, const char* text) {
    if (const std::u16string key = from_utf8(text); !key.empty()) config->left_optional_symbol_key = key[0];
}
void SetRightOptionalSymbolKeyUtf8(KeyboardConfig* config, const char* text) {
    if (const std::u16string key = from_utf8(text); !key.empty()) config->right_optional_symbol_key = key[0];
}

// --- ShowKeyboardArg ---

size_t GetRequiredWorkBufferSize(bool use_dictionary) { return use_dictionary ? 0xd000 : 0x1000; }
// Con lista de palabras: 100 bytes por palabra tras los 0x7e8 del texto inicial, redondeado a 4 KB (máximo 5000).
size_t GetRequiredWorkBufferSize(int user_word_count) {
    if (unsigned(user_word_count) > 5000) return 0;
    if (user_word_count == 0) return 0x1000;
    return (size_t(user_word_count) * 100 + 0x17d3) & ~size_t{0xfff};
}
size_t GetRequiredStringBufferSize() { return 2004; }
size_t GetRequiredTextCheckWorkBufferSize() { return 2004; }
size_t UserWordInfo::GetRequiredWorkBufferSize(int count) {
    return unsigned(count) > 5000 ? 0 : (size_t(count) * 100 + 0x1007) & ~size_t{0xfff};
}

void SetInitialText(ShowKeyboardArg* arg, const char16_t* text) {
    KeyboardConfig& config = arg->keyboard_config;
    if (!text) {
        config.initial_string_length = 0;
        return;
    }
    const size_t n = length16(text, 500);
    config.initial_string_offset = 0x14;
    config.initial_string_length = int32_t(n);
    g_initial_arg = arg;
    g_initial.assign(text, n);
    if (arg->work_buf && 0x14 + n * sizeof(char16_t) <= arg->work_buf_size)
        memcpy(static_cast<uint8_t*>(arg->work_buf) + 0x14, text, n * sizeof(char16_t));
}

// El texto inicial en UTF-8: el SDK lo deja en el mismo sitio y en UTF-16 (sdk+0x3a7870), así que es el de arriba con
// el texto convertido. Es el que usa el teclado de Unity (TouchScreenKeyboard, main+0x3e3ad8) con el texto que ya tenía
// el campo; Unity reserva el work buffer antes (main+0x3e38e0) y lee la respuesta en UTF-16 (main+0x3e3b0c).
void SetInitialTextUtf8(ShowKeyboardArg* arg, const char* text) {
    if (!text) return SetInitialText(arg, nullptr);
    SetInitialText(arg, utf8_to_utf16(text).c_str());
}

// Las palabras del diccionario de usuario solo las usaría el applet para sugerir: se apuntan cuántas hay y dónde irían
// (tras el texto inicial), sin copiarlas.
void SetUserWordList(ShowKeyboardArg* arg, const UserWord* words, int count) {
    KeyboardConfig& config = arg->keyboard_config;
    if (!words || count <= 0) {
        config.user_dictionary_num = 0;
        return;
    }
    config.user_dictionary_num = count;
    if (arg->work_buf && arg->work_buf_size >= GetRequiredWorkBufferSize(count)) config.user_dictionary_offset = 0x7e8;
}

void SetCustomizedDictionaries(ShowKeyboardArg* arg, const CustomizedDictionarySet& set) {
    arg->customized_dictionary_buf = set.buffer;
    arg->customized_dictionary_buf_size = set.size;
    const size_t count = std::min<size_t>(set.count, 24);
    arg->keyboard_config.customized_dictionary_count = uint8_t(set.count);
    memcpy(arg->keyboard_config.customized_dictionaries, set.dictionaries, count * sizeof set.dictionaries[0]);
}

void SetTextCheckCallback(ShowKeyboardArg* arg, TextCheckCallback callback) {
    arg->keyboard_config.is_use_text_check = callback != nullptr;
    if (callback) arg->keyboard_config.text_check_callback = callback;
}

Result ShowKeyboard(String* out, const ShowKeyboardArg& arg) {
    const KeyboardConfig& config = arg.keyboard_config;
    printf("[swkbd] teclado: «%s» «%s» (de %d a %d caracteres)\n", utf16_to_utf8(config.header_text, 65).c_str(),
           utf16_to_utf8(config.guide_text, 257).c_str(), config.text_min_length, config.text_max_length);

    std::vector<std::u16string> candidates;
    if (std::u16string initial = initial_text(arg); !initial.empty()) {
        candidates.push_back(initial);
    } else {
        std::vector<std::string> names = configured_names();
        if (!names.empty()) {
            candidates.push_back(utf8_to_utf16(names[std::min(g_names_used, names.size() - 1)]));
            g_names_used++;
        }
    }
    candidates.push_back(utf8_to_utf16(kDefaultName));
    if (g_initial_arg == &arg) g_initial_arg = nullptr;

    for (std::u16string& text : candidates) {
        if (config.text_max_length > 0 && text.size() > size_t(config.text_max_length))
            text.resize(config.text_max_length);
        if (text.size() < size_t(std::max(config.text_min_length, 0))) continue;
        const size_t bytes = write_result(out, text, config.is_use_utf8);
        // Como el applet: el String lleva durante la llamada el tamaño del texto (con el 0), y el mensaje de error que
        // deje el juego va al work buffer de comprobación.
        if (config.is_use_text_check && config.text_check_callback) {
            const size_t capacity = out->size;
            out->size = bytes + (config.is_use_utf8 ? 1 : sizeof(char16_t));
            size_t message_size = 0;
            const TextCheckResult r = config.text_check_callback(arg.text_check_work_buf, &message_size, out);
            out->size = capacity;
            if (r == TextCheckResult_ShowFailureDialog) {
                const auto* message = static_cast<const char16_t*>(arg.text_check_work_buf);
                printf("[swkbd] el juego rechaza «%s»: %s\n", utf16_to_utf8(text.data(), text.size()).c_str(),
                       message ? utf16_to_utf8(message, 1002).c_str() : "");
                continue;
            }
        }
        printf("[swkbd] nombre: %s\n", utf16_to_utf8(text.data(), text.size()).c_str());
        return ResultSuccess;
    }
    write_result(out, {}, config.is_use_utf8);
    printf("[swkbd] ningún texto vale: cancelado\n");
    return kResultCanceled;
}

// --- InputText ---

void InputText::Set(const char16_t* text) {
    if (!text) {
        this->text[0] = 0;
        cursor_pos = reserved = 0;
        return;
    }
    copy_text(this->text, 501, text);
}

// --- InlineKeyboard: el estado vive en el work buffer del juego (0x6e0 bytes), como el del SDK ---

struct InlineKeyboard::Impl {
    State state = State_None;
    bool initialize_pending = true;
    bool decision_pending = false;
    bool released_user_word_pending = false;
    int32_t cursor_pos = 0;
    void (*finished_initialize)() = nullptr;
    void (*finished_keyboard)() = nullptr;
    void (*changed_string)(const ChangedStringArg*) = nullptr;
    void (*moved_cursor)(const MovedCursorArg*) = nullptr;
    void (*decided_enter)(const DecidedEnterArg*) = nullptr;
    void (*decided_cancel)() = nullptr;
    void (*released_user_word)() = nullptr;
    InputText text{};
};

InlineKeyboard::InlineKeyboard() : impl_(nullptr), work_(nullptr) {}
InlineKeyboard::~InlineKeyboard() {}
size_t InlineKeyboard::GetRequiredWorkBufferSize() { return 0x6e0; }

bool InlineKeyboard::Initialize(void* work_buffer) {
    static_assert(sizeof(Impl) <= 0x6e0);
    if (!work_buffer) return false;
    work_ = work_buffer;
    impl_ = new (work_buffer) Impl;
    return true;
}

void InlineKeyboard::Finalize() {
    if (!impl_) return;
    impl_->~Impl();
    impl_ = nullptr;
}

// Las respuestas del "applet" llegan aquí, como en el SDK: el fin de la inicialización (el juego pone los callbacks
// después de Initialize, main+0x6fa760c), la de un Appear (el jugador cancela; el juego espera a ver el teclado oculto
// otra vez para cerrar, main+0x6fa7e80) y la de UnsetUserWordInfo.
State InlineKeyboard::Calc() {
    if (!impl_) return State_None;
    Impl& k = *impl_;
    if (k.initialize_pending) {
        k.initialize_pending = false;
        k.state = State_Hidden;
        if (k.finished_initialize) k.finished_initialize();
    } else if (k.decision_pending) {
        k.decision_pending = false;
        k.state = State_Hidden;
        if (k.decided_enter) {
            std::u16string text(k.text.text, length16(k.text.text, 501));
            if (text.empty()) {
                const std::vector<std::string> names = configured_names();
                text = utf8_to_utf16(names.empty() ? kDefaultName : names[std::min(g_names_used, names.size() - 1)]);
                g_names_used++;
            }
            DecidedEnterArg arg{};
            const size_t n = std::min<size_t>(text.size(), 501);
            memcpy(arg.text, text.data(), n * sizeof(char16_t));
            arg.length = static_cast<int32_t>(n);
            printf("[swkbd] teclado en línea: «%s»\n", utf16_to_utf8(arg.text, n).c_str());
            k.decided_enter(&arg);
        } else {
            printf("[swkbd] teclado en línea: cancelado\n");
            if (k.decided_cancel) k.decided_cancel();
        }
    }
    if (k.released_user_word_pending) {
        k.released_user_word_pending = false;
        if (k.released_user_word) k.released_user_word();
    }
    return k.state;
}

void InlineKeyboard::Appear(const AppearArg&) {
    if (!impl_) return;
    impl_->state = State_Shown;
    impl_->decision_pending = true;
}

void InlineKeyboard::Disappear() {
    if (!impl_) return;
    impl_->decision_pending = false;
    if (impl_->state != State_None) impl_->state = State_Hidden;
}

void InlineKeyboard::SetInputText(const InputText& text) {
    if (impl_) impl_->text = text;
}
void InlineKeyboard::SetCursorPos(int pos) {
    if (impl_) impl_->cursor_pos = pos;
}
void InlineKeyboard::SetUserWordInfo(const UserWordInfo&) {}
// El applet avisa con ReleasedUserWordInfo cuando suelta las palabras: el juego puede esperarlo para liberarlas.
void InlineKeyboard::UnsetUserWordInfo() {
    if (impl_) impl_->released_user_word_pending = true;
}
void InlineKeyboard::SetCustomizedDictionaries(const CustomizedDictionarySet&) {}
void InlineKeyboard::UnsetCustomizedDictionaries() {}
void InlineKeyboard::SetWindowMode(WindowMode) {}
void InlineKeyboard::SetKeytopAlpha(float) {}
void InlineKeyboard::SetFooterAlpha(float) {}
void InlineKeyboard::SetKeytopScale(float) {}
void InlineKeyboard::SetKeytopTranslate(float, float) {}
void InlineKeyboard::SetTouchEnabled(bool) {}
void InlineKeyboard::SetHardwareKeyboardEnabled(bool) {}

void InlineKeyboard::SetFinishedInitializeCallback(void (*callback)()) {
    if (impl_) impl_->finished_initialize = callback;
}
void InlineKeyboard::SetFinishedKeyboardCallback(void (*callback)()) {
    if (impl_) impl_->finished_keyboard = callback;
}
void InlineKeyboard::SetChangedStringCallback(void (*callback)(const ChangedStringArg*)) {
    if (impl_) impl_->changed_string = callback;
}
void InlineKeyboard::SetMovedCursorCallback(void (*callback)(const MovedCursorArg*)) {
    if (impl_) impl_->moved_cursor = callback;
}
void InlineKeyboard::SetDecidedEnterCallback(void (*callback)(const DecidedEnterArg*)) {
    if (impl_) impl_->decided_enter = callback;
}
void InlineKeyboard::SetDecidedCancelCallback(void (*callback)()) {
    if (impl_) impl_->decided_cancel = callback;
}
void InlineKeyboard::SetReleasedUserWordInfoCallback(void (*callback)()) {
    if (impl_) impl_->released_user_word = callback;
}

// La altura del teclado completo según el SDK (la usa la interfaz del juego para dejarle sitio).
int InlineKeyboard::GetMaxHeight() { return 400; }

// Nunca se dibuja: ni ocupa la pantalla táctil ni da imagen.
bool InlineKeyboard::IsUsedTouchPointByKeyboard(int, int) { return false; }
int InlineKeyboard::GetTouchRectangles(Rect* first, Rect* second) const {
    if (first) *first = {};
    if (second) *second = {};
    return 0;
}
bool InlineKeyboard::GetImage(void*, size_t) { return false; }

// La ventana del teclado es la pantalla de la Switch, y su imagen RGBA8 de 1280x720 como la pide vi: múltiplo de
// 128 KB alineado a 4 KB. El juego crea una textura con esto (main+0x6eff9cc) aunque nunca le llegue imagen.
void InlineKeyboard::GetWindowSize(int* width, int* height) const {
    *width = 1280;
    *height = 720;
}
void InlineKeyboard::GetImageMemoryRequirement(size_t* size, size_t* alignment) {
    *size = (size_t{1280} * 720 * 4 + 0x1ffff) & ~size_t{0x1ffff};
    *alignment = 0x1000;
}

}  // namespace nn::swkbd
