// nn::util: utilidades puras del SDK (UTF-8/UTF-16, printf, StringBuilder, Base64, UUID, URI, el diccionario
// intrusivo, pmr) y su zlib (nnutilZlib_*). Lo que tiene casos límite se calca de sdk.elf (SDK 11.4): códigos de
// retorno, cuánto se escribe y cuándo se termina en 0, porque el juego mira esos resultados. Los printf van a los de
// bionic, con el formato retocado donde el formateador del SDK hace otra cosa (sdk_format). La zlib del SDK es la
// 1.2.11 renombrada; la del sistema (libz del NDK) tiene el mismo z_stream (112 bytes) y solo compara la primera cifra
// de la versión, así que se le pasa la llamada tal cual, con los zalloc/zfree del juego.

#include <stdarg.h>
#include <stdlib.h>
#include <zlib.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn {
class MemoryResource;
}

namespace nn::util {

enum ConversionResult {
    ConversionResult_Success = 0,
    ConversionResult_InsufficientLength = 1,  // no cabe en el destino
    ConversionResult_InvalidFormat = 2,       // origen mal formado (o cortado a media secuencia)
};

struct Uuid {
    uint8_t data[16];
};

namespace {

// Largo de la secuencia UTF-8 según su primer byte, como la tabla de ConvertStringUtf8ToUtf16Native: 0 en los de
// continuación, C0/C1 cuentan como 2 (luego se rechazan por sobrelargos) y F8..FF como 5..8 (luego no se aceptan).
// Importa el valor exacto: con el origen corto da InvalidFormat y con el destino lleno InsufficientLength.
unsigned utf8_sequence_length(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xc0) return 0;
    if (c < 0xe0) return 2;
    if (c < 0xf0) return 3;
    if (c < 0xf8) return 4;
    if (c < 0xfc) return 5;
    if (c < 0xfe) return 6;
    return c == 0xfe ? 7 : 8;
}

bool is_continuation(uint8_t c) { return (c & 0xc0) == 0x80; }

// Núcleo de ConvertStringUtf8ToUtf16Native: written y consumed son unidades del destino y bytes del origen ya
// tratados, también cuando falla. Los largos llegan como int y se extienden con signo (dst_len -1 = sin límite).
ConversionResult utf8_to_utf16(size_t* written, size_t* consumed, uint16_t* dst, int dst_len, const char* src,
                               int src_len) {
    const uint64_t dst_total = static_cast<int64_t>(dst_len), src_total = static_cast<int64_t>(src_len);
    uint64_t dst_left = dst_total, src_left = src_total;
    auto s = reinterpret_cast<const uint8_t*>(src);
    ConversionResult result = ConversionResult_InvalidFormat;
    if (src_len == 0) {
        *written = *consumed = 0;
        return ConversionResult_Success;
    }
    for (;;) {
        const uint8_t c = s[0];
        const unsigned n = utf8_sequence_length(c);
        if (src_left < n) break;
        if (dst_left == 0) {
            result = ConversionResult_InsufficientLength;
            break;
        }
        if (n == 1) {
            *dst++ = c;
            dst_left--;
        } else if (n == 2) {
            if (!(c & 0x1e) || !is_continuation(s[1])) break;
            *dst++ = static_cast<uint16_t>((c & 0x1f) << 6 | (s[1] & 0x3f));
            dst_left--;
        } else if (n == 3) {
            if (!is_continuation(s[1]) || !is_continuation(s[2])) break;
            const uint32_t cp = (c & 0xf) << 12 | (s[1] & 0x3f) << 6 | (s[2] & 0x3f);
            if ((cp & 0xf800) == 0 || (cp & 0xf800) == 0xd800) break;  // sobrelargo o sustituto
            *dst++ = static_cast<uint16_t>(cp);
            dst_left--;
        } else if (n == 4) {
            if (!is_continuation(s[1]) || !is_continuation(s[2]) || !is_continuation(s[3])) break;
            const uint32_t cp = ((c & 7) << 18 | (s[1] & 0x3f) << 12 | (s[2] & 0x3f) << 6 | (s[3] & 0x3f)) - 0x10000;
            if (cp >> 20) break;  // sobrelargo (da la vuelta) o más allá de U+10FFFF
            if (dst_left == 1) {
                result = ConversionResult_InsufficientLength;
                break;
            }
            dst[0] = static_cast<uint16_t>(0xd800 + (cp >> 10));
            dst[1] = static_cast<uint16_t>(0xdc00 | (cp & 0x3ff));
            dst += 2;
            dst_left -= 2;
        } else {
            break;
        }
        src_left -= n;
        if (src_left == 0) {
            *written = dst_total - dst_left;
            *consumed = src_total;
            return ConversionResult_Success;
        }
        s += n;
    }
    *written = dst_total - dst_left;
    *consumed = src_total - src_left;
    return result;
}

// Núcleo de ConvertStringUtf16NativeToUtf8, mismo convenio. Un sustituto alto mal emparejado deja escrito el primer
// byte de la secuencia de 4 antes de fallar (y lo cuenta), sin mirar si cabe: el SDK lo hace así, y el envoltorio le
// deja siempre un byte de margen.
ConversionResult utf16_to_utf8(size_t* written, size_t* consumed, char* dst_chars, int dst_len, const uint16_t* s,
                               int src_len) {
    const uint64_t dst_total = static_cast<int64_t>(dst_len), src_total = static_cast<int64_t>(src_len);
    uint64_t dst_left = dst_total, src_left = src_total;
    auto dst = reinterpret_cast<uint8_t*>(dst_chars);
    ConversionResult result = ConversionResult_InvalidFormat;
    if (src_len == 0) {
        *written = *consumed = 0;
        return ConversionResult_Success;
    }
    for (;;) {
        const uint32_t c = s[0];
        unsigned units = 1, bytes;
        if (c < 0x80) {
            if (dst_left == 0) {
                result = ConversionResult_InsufficientLength;
                break;
            }
            dst[0] = static_cast<uint8_t>(c);
            bytes = 1;
        } else if ((c & 0xf800) == 0) {
            if (dst_left < 2) {
                result = ConversionResult_InsufficientLength;
                break;
            }
            dst[0] = static_cast<uint8_t>(0xc0 | (c >> 6 & 0x1f));
            dst[1] = static_cast<uint8_t>(0x80 | (c & 0x3f));
            bytes = 2;
        } else if ((c >> 11) != 0x1b) {
            if (dst_left < 3) {
                result = ConversionResult_InsufficientLength;
                break;
            }
            dst[0] = static_cast<uint8_t>(0xe0 | c >> 12);
            dst[1] = static_cast<uint8_t>(0x80 | (c >> 6 & 0x3f));
            dst[2] = static_cast<uint8_t>(0x80 | (c & 0x3f));
            bytes = 3;
        } else {
            if (src_left == 1) {  // sustituto en la última unidad
                if ((c & 0xfc00) != 0xd800) break;
                *dst = static_cast<uint8_t>(0xf0 | (((c << 10) + 0x210000) >> 18));
                dst_left--;
                break;
            }
            const uint32_t c2 = s[1];
            if (c & 0x400) break;  // empieza por un sustituto bajo
            const uint32_t cp = (c << 10) + c2 + 0xfca02400u;  // ((c - D800) << 10) + (c2 - DC00) + 0x10000
            if ((c2 & 0xfc00) != 0xdc00) {
                *dst = static_cast<uint8_t>(0xf0 | cp >> 18);
                dst_left--;
                break;
            }
            if (dst_left < 4) {
                result = ConversionResult_InsufficientLength;
                break;
            }
            dst[0] = static_cast<uint8_t>(0xf0 | cp >> 18);
            dst[1] = static_cast<uint8_t>(0x80 | (cp >> 12 & 0x3f));
            dst[2] = static_cast<uint8_t>(0x80 | (cp >> 6 & 0x3f));
            dst[3] = static_cast<uint8_t>(0x80 | (cp & 0x3f));
            units = 2;
            bytes = 4;
        }
        dst += bytes;
        dst_left -= bytes;
        src_left -= units;
        if (src_left == 0) {
            *written = dst_total - dst_left;
            *consumed = src_total;
            return ConversionResult_Success;
        }
        s += units;
    }
    *written = dst_total - dst_left;
    *consumed = src_total - src_left;
    return result;
}

// GetCodePointCountOfUtf8String del SDK: -1 si [s, s + n) no es UTF-8 válido (sobrelargos, sustitutos y más allá de
// U+10FFFF incluidos). Manda n, no el terminador: un 0 en medio cuenta como carácter.
int code_point_count(const char* str, size_t n) {
    auto s = reinterpret_cast<const uint8_t*>(str);
    int count = 0;
    while (n) {
        const uint8_t c = s[0];
        const size_t len = c < 0x80 ? 1 : c < 0xc2 ? 0 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : c < 0xf5 ? 4 : 0;
        if (n < len || len == 0) return -1;
        bool ok = true;
        if (len == 2) {
            ok = is_continuation(s[1]);
        } else if (len == 3) {
            ok = (c != 0xe0 || (s[1] & 0x20)) && (c != 0xed || !(s[1] & 0x20)) && is_continuation(s[1]) &&
                 is_continuation(s[2]);
        } else if (len == 4) {
            ok = (c != 0xf0 || (s[1] & 0x30)) && (c != 0xf4 || !(s[1] & 0x30)) && is_continuation(s[1]) &&
                 is_continuation(s[2]) && is_continuation(s[3]);
        }
        if (!ok) return -1;
        s += len;
        n -= len;
        count++;
    }
    return count;
}

// El formateador del SDK (VFormatString) da lo mismo que el vsnprintf de bionic en todo lo que usa el juego (%s %.*s
// %u %d %08x, cabeceras HTTP) salvo en: %p (16 cifras hexadecimales sin "0x", sin precisión) y lo que no conoce (%n,
// %a, %1$s, %% o %c con modificadores; bionic aborta con %n). Eso se reescribe aquí antes de llamar a bionic. Con lo
// que no conoce, el SDK copia el texto hasta el carácter que no entiende (sin él), sigue leyendo desde ese carácter y
// se queda con los argumentos de los '*' que ya leyó: aquí cada '*' se gasta con un "%.0s" (un argumento de 8 bytes que
// no imprime nada). Quedan distintos, y no salen en el juego: %hhd (el SDK no extiende el signo), %f de más de 17
// cifras (redondea distinto), "%+.0d" con 0 (el SDK no pone el signo) y el ancho de "%*p". Comprobado contra el código
// de sdk.elf con formatos aleatorios. Devuelve format si no hay nada que cambiar.
const char* sdk_format(const char* format, std::string& rewritten) {
    bool changed = false;
    for (const char* p = format; *p;) {
        if (*p != '%') {
            rewritten += *p++;
            continue;
        }
        const char* start = p++;
        while (*p && strchr("-+ #0", *p)) p++;
        const char* width = p;
        if (*p == '*') {
            p++;
        } else {
            while (*p >= '0' && *p <= '9') p++;
        }
        const char* width_end = p;
        const bool precision = *p == '.';
        if (precision) {
            p++;
            if (*p == '*') {
                p++;
            } else {
                while (*p >= '0' && *p <= '9') p++;
            }
        }
        while (*p && strchr("hljztL", *p)) p++;
        const char c = *p;
        const int stars = (*width == '*') + (precision && width_end[1] == '*');
        if ((c && strchr("diouxXsfeEgG", c)) || (c == '%' && p == start + 1) || (c == 'c' && !precision)) {
            rewritten.append(start, ++p);
        } else if (c == 'p') {
            // '#' pone el "0x" como en el SDK. Con '*' el ancho se pierde, pero los argumentos siguen en su sitio.
            for (int i = 0; i < stars; i++) rewritten += "%.0s";
            rewritten.append(start, *width == '*' ? width : width_end).append(".16lx");
            p++;
            changed = true;
        } else {
            for (int i = 0; i < stars; i++) rewritten += "%.0s";
            for (const char* q = start; q != p; q++) {
                if (*q == '%') rewritten += '%';
                rewritten += *q;
            }
            changed = true;
        }
    }
    return changed ? rewritten.c_str() : format;
}

}  // namespace

ConversionResult ConvertStringUtf8ToUtf16Native(uint16_t* dst, int dst_len, const char* src) {
    size_t written, consumed;
    // Se reserva el sitio del 0; solo se termina la cadena si la conversión fue entera.
    ConversionResult r = utf8_to_utf16(&written, &consumed, dst, dst_len - 1, src, static_cast<int>(strlen(src)));
    if (r == ConversionResult_Success) dst[written] = 0;
    return r;
}

ConversionResult ConvertStringUtf16NativeToUtf8(char* dst, int dst_len, const uint16_t* src) {
    size_t len = 0;
    while (src && src[len]) len++;
    size_t written, consumed;
    ConversionResult r = utf16_to_utf8(&written, &consumed, dst, dst_len - 1, src, static_cast<int>(len));
    if (r == ConversionResult_Success) dst[written] = 0;
    return r;
}

bool VerifyUtf8String(const char* s, size_t size) { return code_point_count(s, size) != -1; }

// Devuelven lo que ocuparía la salida entera (sin el 0), como snprintf: el juego lo usa para saber si truncó.
int VSNPrintf(char* buffer, size_t size, const char* format, va_list args) {
    std::string rewritten;
    return vsnprintf(buffer, size, sdk_format(format, rewritten), args);
}

int SNPrintf(char* buffer, size_t size, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int n = VSNPrintf(buffer, size, format, args);
    va_end(args);
    return n;
}

void ReferSymbol(const void*) {}

// UUID versión 4 (RFC 4122): todo aleatorio salvo la versión (byte 6) y la variante (byte 8).
Uuid GenerateUuid() {
    Uuid uuid;
    arc4random_buf(uuid.data, sizeof uuid.data);
    uuid.data[6] = static_cast<uint8_t>(0x40 | (uuid.data[6] & 0x0f));
    uuid.data[8] = static_cast<uint8_t>(0x80 | (uuid.data[8] & 0x3f));
    return uuid;
}

// El objeto es del juego (lo tiene en su pila); el layout es el del SDK.
class StringBuilder {
public:
    StringBuilder(char* buffer, size_t size);
    StringBuilder& Append(const char* s);
    StringBuilder& AppendFormat(const char* format, ...);
    bool Fix();
    size_t GetLength() const;

private:
    char* buffer_;
    size_t size_;
    size_t length_;
    bool valid_;  // pasa a false en cuanto algo no cabe, y ya no vuelve
    bool fixed_;
};
static_assert(sizeof(StringBuilder) == 0x20);

StringBuilder::StringBuilder(char* buffer, size_t size)
    : buffer_(buffer), size_(size), length_(0), valid_(true), fixed_(false) {
    buffer[0] = 0;
}

StringBuilder& StringBuilder::Append(const char* s) {
    for (; *s; s++) {
        if (size_ <= length_ + 1) {
            valid_ = false;
            break;
        }
        buffer_[length_++] = *s;
    }
    buffer_[length_] = 0;
    return *this;
}

StringBuilder& StringBuilder::AppendFormat(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int n = VSNPrintf(buffer_ + length_, size_ - length_, format, args);
    va_end(args);
    length_ += n;  // con signo, como el SDK
    if (size_ <= length_) {
        length_ = size_ - 1;
        valid_ = false;
    }
    return *this;
}

bool StringBuilder::Fix() {
    buffer_[length_] = 0;
    fixed_ = true;
    return valid_;
}

size_t StringBuilder::GetLength() const { return length_; }

class Base64 {
public:
    enum Mode {
        Mode_Normal = 0,             // con CR LF cada 76 caracteres (MIME); cualquier otro valor hace lo mismo
        Mode_NormalNoLinefeed = 1,   // lo que usa el juego
        Mode_UrlSafe = 2,            // '-' y '_', sin '=' ni saltos
    };
    enum Status { Status_Success = 0, Status_BufferFull = 1, Status_BadData = 2 };
    static Status ToBase64String(char* dst, size_t size, const void* src, size_t src_size, Mode mode);
};

// El 0 final tiene que caber: si no, BufferFull (con lo que sí cupo ya escrito).
Base64::Status Base64::ToBase64String(char* dst, size_t size, const void* src, size_t src_size, Mode mode) {
    static const char kNormal[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static const char kUrlSafe[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    const char* table = mode == Mode_UrlSafe ? kUrlSafe : kNormal;
    const bool line_breaks = mode != Mode_UrlSafe && mode != Mode_NormalNoLinefeed;
    size_t pos = 0;
    int column = 0;
    auto put = [&](char c) {
        if (pos == size) return false;
        dst[pos++] = c;
        return true;
    };
    auto put_digit = [&](char c) {
        if (line_breaks && column == 76) {
            if (!put('\r') || !put('\n')) return false;
            column = 0;
        }
        column++;
        return put(c);
    };
    auto in = static_cast<const uint8_t*>(src);
    uint32_t bits = 0;
    int shift = 0;  // bits que quedan pendientes del byte anterior: 0, 2 o 4
    for (size_t i = 0; i < src_size; i++) {
        bits = bits << 8 | in[i];
        shift += 2;
        if (!put_digit(table[(bits >> shift) & 63])) return Status_BufferFull;
        if (shift == 6) {
            shift = 0;
            if (!put_digit(table[bits & 63])) return Status_BufferFull;
        }
    }
    if (shift > 0 && !put_digit(table[(bits << (6 - shift)) & 63])) return Status_BufferFull;
    if (mode != Mode_UrlSafe) {
        for (; column & 3; column++)
            if (!put('=')) return Status_BufferFull;
    }
    if (!put(0)) return Status_BufferFull;
    return Status_Success;
}

// UriParser: un URI de RFC 3986 partido en trozos que apuntan dentro de la cadena del juego. La red del juego lo usa
// para sacar host y puerto. Cada Parse* calca su homónimo del SDK, rarezas incluidas (lee el byte que sigue a la
// autoridad aunque esté vacía, '%' vale como '-' en las etiquetas de dominio...).
class UriPart {
public:
    const char* GetPointer() const;
    size_t GetLength() const;
    bool Exists() const;
    bool CompareInsensitive(const char* s, size_t size) const;
    bool ParseInteger(uint16_t* out) const;

    const char* begin;
    const char* end;
};

struct UriParts {
    UriPart scheme, authority, user_info, host, port, path, query, fragment;
};

class UriParser {
public:
    enum Option {
        Option_None = 0,
        Option_HostAsDomainNameOrIpv4 = 1,  // el host tiene que ser un nombre de dominio o una IPv4
        Option_RelativeReference = 2,       // sin esquema
    };
    UriParser();
    bool Parse(const char* uri, Option option);
    const UriParts& GetParts() const;

private:
    UriParts parts_;
    bool valid_;
};
static_assert(sizeof(UriParser) == 0x88);

namespace {

bool is_digit(uint8_t c) { return static_cast<uint8_t>(c - '0') < 10; }
bool is_alpha(uint8_t c) { return static_cast<uint8_t>((c & 0xdf) - 'A') < 26; }
bool is_hex_digit(uint8_t c) {
    return is_digit(c) || (static_cast<uint8_t>(c - 'A') <= 0x25 && (1ull << (c - 'A')) & 0x3f0000003full);
}

// Signos que admite cada trozo además de alfanuméricos y %XX (las tablas de saltos del SDK).
constexpr const char* kUserInfoChars = "!$&'()*+,-.:;=_~";
constexpr const char* kHostChars = "!$&'()*+,-.;=_~";
constexpr const char* kPathChars = "!$&'()*+,-./:;=@_~";
constexpr const char* kQueryChars = "!$&'()*+,-./:;=?@_~";

bool valid_chars(const char* p, const char* end, const char* extra) {
    int hex_left = 0;  // dígitos hexadecimales que faltan tras un '%'
    for (; p != end; p++) {
        const uint8_t c = *p;
        if (hex_left) {
            if (!is_hex_digit(c)) return false;
            hex_left--;
        } else if (c == '%') {
            hex_left = 2;
        } else if (!is_digit(c) && !is_alpha(c) && (static_cast<uint8_t>(c - 0x21) > 0x5d || !strchr(extra, c))) {
            return false;
        }
    }
    return hex_left == 0;
}

bool parse_user_info(UriParts& parts, const char* p, const char* end) {
    for (const char* q = p; q != end; q++) {
        if (*q == '@') {
            parts.user_info.end = q;
            break;
        }
    }
    if (!parts.user_info.end) return true;
    parts.user_info.begin = p;
    return parts.user_info.end <= p || valid_chars(p, parts.user_info.end, kUserInfoChars);
}

bool parse_host(UriParts& parts, const char* p, const char* end) {
    parts.host = {p, end};
    if (p == end) return true;
    const char* q = p;
    while (q != end && *q != ':') q++;
    if (q != end) parts.host.end = q;  // ahí empieza el puerto
    return q <= p || valid_chars(p, q, kHostChars);
}

// Etiqueta de dominio: 1..63 caracteres alfanuméricos o '-' (y '%': el SDK compara c | 8 con '-'), sin '-' en los
// extremos.
bool valid_label(const char* p, const char* end) {
    if (static_cast<uint32_t>(end - p - 1) > 0x3e) return false;
    if (*p == '-' || end[-1] == '-') return false;
    for (; p != end; p++) {
        const uint8_t c = *p;
        if (!is_digit(c) && !is_alpha(c) && (c | 8) != '-') return false;
    }
    return true;
}

bool parse_host_as_domain_name_or_ipv4(UriParts& parts, const char* p, const char* end) {
    if (static_cast<uint32_t>(end - p - 1) > 0xfc) return false;  // 1..253 caracteres
    parts.host = {p, end};
    const char* label = p;
    for (const char* q = p;;) {
        if (*q == '.') {
            if (!valid_label(label, q)) return false;
            label = ++q;
            if (q == end) break;  // termina en '.': la última etiqueta queda vacía
            continue;
        }
        if (*q == ':') {
            parts.host.end = end = q;
            break;
        }
        if (++q == end) break;
    }
    return valid_label(label, end);
}

// [IPv6]: dígitos hexadecimales, ':' y '.' hasta el ']'; "[]" no vale.
bool parse_host_as_ipv6(UriParts& parts, const char* p, const char* end) {
    parts.host.begin = p;
    if (p + 1 != end) {
        if (p[1] == ']') {
            parts.host.end = p + 2;
        } else {
            for (const char* q = p;;) {
                const uint8_t c = q[1];
                if (!is_digit(c) &&
                    !(static_cast<uint32_t>(c - 0x2e) <= 0x38 && (1ull << (c - 0x2e)) & 0x01f8000001f81001ull))
                    return false;
                if (q == end - 2) break;
                q++;
                if (q[1] == ']') {
                    parts.host.end = q + 2;
                    break;
                }
            }
        }
    }
    return parts.host.end && static_cast<int>(parts.host.end - p) != 2;
}

bool parse_authority(UriParts& parts, const char* p, const char* end, bool domain_name) {
    parts.authority.begin = p;
    const char* a = p;
    while (a != end && !(static_cast<uint8_t>(*a) <= 0x3f && (1ull << *a) & 0x8000800800000000ull)) a++;  // / ? #
    parts.authority.end = a;
    if (!parse_user_info(parts, p, a)) return false;
    const char* host = parts.user_info.begin ? parts.user_info.end + 1 : p;
    if (*host == '[') {
        if (!parse_host_as_ipv6(parts, host, a)) return false;
    } else if (!(domain_name ? parse_host_as_domain_name_or_ipv4(parts, host, a) : parse_host(parts, host, a))) {
        return false;
    }
    const char* host_end = parts.host.end;
    if (*host_end == ':') {
        for (const char* q = host_end + 1; q != a; q++)
            if (!is_digit(*q)) return false;
        parts.port = {host_end + 1, a};
        return true;
    }
    return (parts.port.begin ? parts.port.end : host_end) == a;
}

bool parse_path(UriParts& parts, const char* p, const char* end, bool relative) {
    parts.path.begin = p;
    const char* q = p;
    while (q != end && *q != '#' && *q != '?') q++;
    parts.path.end = q;
    if (q > p && !valid_chars(p, q, kPathChars)) return false;
    const int n = static_cast<int>(q - p);
    for (int i = 0; i < n && p[i]; i++)
        if (p[i] == '/' && i + 1 < n && p[i + 1] == '/') return false;  // "//" dentro del camino
    if (q == p || !relative) return true;
    // Referencia relativa: el primer segmento no puede llevar ':' (se confundiría con un esquema).
    for (const char* r = p; r != q && *r != '/'; r++)
        if (*r == ':') return false;
    return true;
}

bool parse_query(UriParts& parts, const char* q, const char* end) {
    if (q == end || *q != '?') return true;
    parts.query.begin = q + 1;
    const char* e = q + 1;
    while (e != end && *e != '#') e++;
    parts.query.end = e;
    return e == q + 1 || valid_chars(q + 1, e, kQueryChars);
}

bool parse_fragment(UriParts& parts, const char* f, const char* end) {
    if (f == end || *f != '#') return true;
    parts.fragment = {f + 1, end};
    return f + 1 >= end || valid_chars(f + 1, end, kQueryChars);
}

bool parse_uri(UriParts& parts, const char* uri, size_t size, int option) {
    const char* end = uri + size;
    const char* p = uri;
    if (!(option & UriParser::Option_RelativeReference)) {
        // Esquema: letra, luego alfanuméricos y "+-.", hasta ':'.
        if (static_cast<int>(size) < 2 || !is_alpha(uri[0])) return false;
        size_t i = 0;
        for (;;) {
            const uint8_t c = uri[i + 1];
            if (c == ':') break;
            if (!is_digit(c) && !is_alpha(c) && (c > 0x2e || !((1ull << c) & 0x680000000000ull))) return false;
            if (++i == size - 1) return false;
        }
        parts.scheme = {uri, uri + i + 1};
        p = uri + i + 2;
    }
    if (static_cast<int>(end - p) >= 2 && p[0] == '/' && p[1] == '/') {
        if (!parse_authority(parts, p + 2, end, option & UriParser::Option_HostAsDomainNameOrIpv4)) return false;
        p = parts.authority.end;
    }
    if (!parse_path(parts, p, end, option & UriParser::Option_RelativeReference)) return false;
    p = parts.path.end;
    if (!parse_query(parts, p, end)) return false;
    if (parts.query.begin) p = parts.query.end;
    if (!parse_fragment(parts, p, end)) return false;
    if (parts.fragment.begin) p = parts.fragment.end;
    return p == end;
}

}  // namespace

UriParser::UriParser() : parts_{}, valid_(false) {}

bool UriParser::Parse(const char* uri, Option option) {
    parts_ = {};
    valid_ = false;
    valid_ = parse_uri(parts_, uri, strnlen(uri, 0x7fffffff), option);
    return valid_;
}

const UriParts& UriParser::GetParts() const { return parts_; }

const char* UriPart::GetPointer() const { return begin; }
size_t UriPart::GetLength() const { return end - begin; }
bool UriPart::Exists() const { return begin && end; }

bool UriPart::CompareInsensitive(const char* s, size_t size) const {
    const size_t length = end - begin;
    if (size == 0) return length == 0;
    if (length != size) return false;
    auto lower = [](uint8_t c) { return static_cast<uint8_t>(static_cast<uint8_t>(c - 'A') < 26 ? c + 0x20 : c); };
    for (int i = 0; i < static_cast<int>(size); i++) {
        const uint8_t a = lower(begin[i]), b = lower(s[i]);
        if (a == 0 || a != b) return a == b;
    }
    return true;
}

// Decimal que tiene que ocupar el trozo entero y caber en 16 bits (un puerto).
bool UriPart::ParseInteger(uint16_t* out) const {
    if (!begin || !end || !is_digit(*begin)) return false;
    char* stop = nullptr;
    const unsigned long value = strtoul(begin, &stop, 10);
    if (value >> 16) return false;
    if (stop && stop != end) return false;
    *out = static_cast<uint16_t>(value);
    return true;
}

// Diccionario intrusivo (árbol Patricia) del SDK: los nodos son del juego, que inserta con código inline y nos pide
// buscar, recorrer y borrar. Cada nodo es a la vez interno (decide por ref_bit) y hoja (le llega una arista "hacia
// atrás", de un nodo con ref_bit mayor o igual). La raíz es un nodo con ref_bit -1 y la clave vacía.
struct IntrusiveDicNode {
    const char* key;  // basic_string_view<char>: {datos, tamaño}
    size_t key_size;
    int ref_bit;      // bits contados desde el final de la clave: bit b = byte size-1-b/8, bit b%8
    IntrusiveDicNode* parent;
    IntrusiveDicNode* child[2];
};
static_assert(sizeof(IntrusiveDicNode) == 0x30);

namespace detail {

class IntrusiveDicImpl {
public:
    const IntrusiveDicNode* FindIter() const;
    static const IntrusiveDicNode* MoveIter(const IntrusiveDicNode* node, int direction);
    void EraseImpl(IntrusiveDicNode* node);

private:
    IntrusiveDicNode root_;
};

}  // namespace detail

namespace {

int key_bit(const char* key, size_t size, int bit) {
    const int64_t byte = bit >> 3;  // con signo: el -1 de la raíz no cae dentro de ninguna clave
    if (size <= static_cast<uint64_t>(byte)) return 0;
    return static_cast<uint8_t>(key[size - 1 - byte]) >> (bit & 7) & 1;
}

}  // namespace

namespace detail {

// El primer elemento: el nodo que vuelve a la raíz por child[0]; desde él, un paso por child[1] y luego child[0]
// mientras se baje.
const IntrusiveDicNode* IntrusiveDicImpl::FindIter() const {
    const IntrusiveDicNode* last = &root_;
    for (const IntrusiveDicNode* n = root_.child[0]; n != &root_; n = n->child[0]) last = n;
    int bit = last->ref_bit;
    const IntrusiveDicNode* n = last->child[1];
    while (bit < n->ref_bit) {
        bit = n->ref_bit;
        n = n->child[0];
    }
    return n;
}

// Siguiente (direction 1) o anterior (0) elemento de node, que es estático: no usa el diccionario.
const IntrusiveDicNode* IntrusiveDicImpl::MoveIter(const IntrusiveDicNode* node, int direction) {
    // Baja desde el propio nodo con su clave hasta la arista que vuelve a él.
    const IntrusiveDicNode* p = node;
    for (int bit = node->ref_bit;;) {
        const IntrusiveDicNode* next = p->child[key_bit(node->key, node->key_size, bit)];
        if (bit >= next->ref_bit) break;
        bit = next->ref_bit;
        p = next;
    }
    const IntrusiveDicNode* n = p->child[direction];
    if (n == node) {
        const IntrusiveDicNode* from;
        do {
            from = p;
            p = p->parent;
            n = p->child[direction];
        } while (from == n);
    }
    int bit = n->ref_bit;
    if (p->ref_bit >= bit) return n;
    for (;;) {
        n = n->child[1 - direction];
        if (bit >= n->ref_bit) return n;
        bit = n->ref_bit;
    }
}

// Saca node del árbol y lo deja como un nodo suelto (apuntándose a sí mismo). Calca el SDK paso a paso: el orden de
// las lecturas importa cuando un hijo es el propio nodo.
void IntrusiveDicImpl::EraseImpl(IntrusiveDicNode* node) {
    const char* key = node->key;
    const size_t size = node->key_size;
    const int node_bit = node->ref_bit;
    IntrusiveDicNode* parent = node->parent;
    const int b = key_bit(key, size, node_bit);
    IntrusiveDicNode* down = node->child[b];
    IntrusiveDicNode* other = node->child[b ^ 1];
    IntrusiveDicNode* replacement = down == node ? other : down;
    parent->child[parent->child[0] == node ? 0 : 1] = replacement;
    if (node->ref_bit < replacement->ref_bit) replacement->parent = node->parent;
    node->key = nullptr;
    node->key_size = 0;
    node->ref_bit = -1;
    node->parent = node;
    node->child[0] = node->child[1] = node;

    // Si el nodo bajaba hacia su propia hoja por otro camino, se rehace ese camino con su clave hasta la arista que
    // volvía a él.
    IntrusiveDicNode* cur = down;
    if (parent->ref_bit >= cur->ref_bit) return;
    int prev_bit = node_bit, cur_bit = cur->ref_bit;
    for (;;) {
        const int b1 = key_bit(key, size, cur_bit);
        IntrusiveDicNode* next = cur->child[b1];
        IntrusiveDicNode* sibling = cur->child[b1 ^ 1];
        const int b2 = key_bit(key, size, prev_bit);
        cur->child[b2] = cur_bit < next->ref_bit ? next : sibling;
        cur->child[b2 ^ 1] = other;
        cur->ref_bit = prev_bit;
        if (prev_bit < other->ref_bit) other->parent = cur;
        if (cur->ref_bit >= next->ref_bit) return;
        prev_bit = cur_bit;
        cur_bit = next->ref_bit;
        cur = next;
        other = sibling;
    }
}

}  // namespace detail

namespace pmr {

MemoryResource* NewDeleteResource();
MemoryResource* GetDefaultResource();
MemoryResource* SetDefaultResource(MemoryResource* resource);

}  // namespace pmr

}  // namespace nn::util

namespace {

// nn::MemoryResource con la vtable del SDK {~D1, ~D0, do_allocate, do_deallocate, do_is_equal}: el juego la llama
// directamente (allocate/deallocate son inline en la cabecera). La memoria es del allocator del juego; por encima de
// 16 de alineación se reserva de más y el puntero real se guarda justo detrás del bloque, como hace el SDK.
class GameNewDelete {
public:
    virtual ~GameNewDelete() = default;

    virtual void* do_allocate(size_t bytes, size_t alignment) {
        if (alignment <= 16) return nnsdk::host().game_new(bytes);
        const size_t rounded = (bytes + 7) & ~size_t{7};
        auto raw = static_cast<char*>(nnsdk::host().game_new(alignment + rounded + 7));
        if (!raw) return nullptr;
        char* p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(alignment - 1));
        memcpy(p + rounded, &raw, sizeof raw);
        return p;
    }

    virtual void do_deallocate(void* p, size_t bytes, size_t alignment) {
        if (p && alignment > 16) memcpy(&p, static_cast<char*>(p) + ((bytes + 7) & ~size_t{7}), sizeof p);
        nnsdk::host().game_delete(p);
    }

    virtual bool do_is_equal(const GameNewDelete& other) const { return this == &other; }
};

GameNewDelete g_new_delete;
std::atomic<nn::MemoryResource*> g_default_resource{nullptr};  // nullptr = la de new/delete

}  // namespace

namespace nn::util::pmr {

MemoryResource* NewDeleteResource() { return reinterpret_cast<MemoryResource*>(&g_new_delete); }

MemoryResource* GetDefaultResource() {
    MemoryResource* r = g_default_resource.load();
    return r ? r : NewDeleteResource();
}

// Devuelve la anterior; nullptr vuelve a la de new/delete.
MemoryResource* SetDefaultResource(MemoryResource* resource) {
    MemoryResource* old = g_default_resource.exchange(resource ? resource : NewDeleteResource());
    return old ? old : NewDeleteResource();
}

}  // namespace nn::util::pmr

namespace {

struct StringView {
    const char* data;
    size_t size;
};

}  // namespace

extern "C" {

// IntrusiveDicImpl::FindImpl(basic_string_view<char, std::__1::char_traits<char>> const&) const: con su nombre
// mangleado porque std::__1 no existe en el NDK. Devuelve el nodo al que lleva la clave; el juego compara las claves.
const nn::util::IntrusiveDicNode*
_ZNK2nn4util6detail16IntrusiveDicImpl8FindImplERKNS0_17basic_string_viewIcNSt3__111char_traitsIcEEEE(
    const nn::util::IntrusiveDicNode* root, const StringView* key) {
    const nn::util::IntrusiveDicNode* n = root->child[0];
    if (root->ref_bit >= n->ref_bit) return n;
    for (int bit = n->ref_bit;;) {
        n = n->child[nn::util::key_bit(key->data, key->size, bit)];
        if (bit >= n->ref_bit) return n;
        bit = n->ref_bit;
    }
}

int nnutilZlib_deflateInit2_(z_streamp stream, int level, int method, int window_bits, int mem_level, int strategy,
                             const char* version, int stream_size) {
    return deflateInit2_(stream, level, method, window_bits, mem_level, strategy, version, stream_size);
}
int nnutilZlib_deflate(z_streamp stream, int flush) { return deflate(stream, flush); }
int nnutilZlib_deflateEnd(z_streamp stream) { return deflateEnd(stream); }

int nnutilZlib_inflateInit2_(z_streamp stream, int window_bits, const char* version, int stream_size) {
    return inflateInit2_(stream, window_bits, version, stream_size);
}
int nnutilZlib_inflate(z_streamp stream, int flush) { return inflate(stream, flush); }
int nnutilZlib_inflateEnd(z_streamp stream) { return inflateEnd(stream); }

}  // extern "C"
