// Locale, iostreams y regex_traits de la libc++ de Nintendo (std::__1, ABI v1). El juego instancia en línea sus
// stringstreams y su __put_character_sequence y lee directamente los campos de ios_base/basic_ios/basic_streambuf
// (flags +8, width +0x18, rdbuf +0x28, fill +0x90…), así que aquí esas clases se declaran con el mismo layout y el
// mismo orden de virtuales: el compilador da entonces los nombres, los D0/D1/D2 con su VTT, los thunks y las
// typeinfo que el juego importa.
//
// No se reenvía a la libc++ del NDK (std::__ndk1, mismo layout) porque su ctype<char> usa otras máscaras (unsigned
// long, 1 << n) y el juego compara en línea con las de Nintendo, que son las de glibc (unsigned short): con la tabla
// del NDK, isspace(c, loc) o [[:alpha:]]/\d de std::regex darían otra cosa. Hay una sola locale, la "C", inmortal.
//
// Las funciones que llaman a virtuales del streambuf del juego no tienen objetos con destructor ni try/catch: así no
// llevan LSDA y, si el juego lanza desde su streambuf, la unwind (la libc++abi del loader) atraviesa estos marcos sin
// pasar por la personality de la copia de libnnsdk.

#include <pthread.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <new>

#include "nnsdk/runtime.h"

namespace std {
namespace __1 {

template <class C>
struct char_traits;
template <class T>
class allocator;
template <class C, class T, class A>
class basic_string;
template <class C, class T>
class basic_streambuf;
template <class C, class T>
class basic_ios;
template <class C, class T>
class basic_istream;
template <class C, class T>
class basic_ostream;
template <class C, class T>
class basic_iostream;
template <class C>
class ctype;
template <class C>
class collate;

using streamsize = long;

// basic_string<char> (24 bytes): aquí solo se devuelve; lo rellena el append de cxx.cpp con la memoria del juego.
template <>
class basic_string<char, char_traits<char>, allocator<char>> {
  public:
    void* rep_[3] = {};
};
using string = basic_string<char, char_traits<char>, allocator<char>>;

class locale {
  public:
    class facet;
    class id;

    locale() noexcept;
    locale(const locale& other) noexcept;
    ~locale();
    const facet* use_facet(id& x) const;
    bool has_facet(id& x) const;
    string name() const;

  private:
    const void* __locale_;
};

// locale::id = {once_flag, int}: el juego solo pasa su dirección a use_facet.
class locale::id {
  public:
    unsigned long __flag_ = 0;
    int __id_ = 0;
};

// codecvt<char, char, __mbstate_t>::id con el nombre de la libc++ de Nintendo (su mbstate_t es el __mbstate_t de musl):
// el juego lo importa y aquí solo hace falta su dirección.
extern "C" locale::id _ZNSt3__17codecvtIcc11__mbstate_tE2idE;

// locale::facet: el __shared_count de libc++ ({vptr, cuenta}; virtuales D1, D0, __on_zero_shared) sin declararlo
// como base: el juego no mira la typeinfo de las facetas y las nuestras son estáticas, la cuenta no llega a cero.
class locale::facet {
  public:
    constexpr facet() = default;
    virtual ~facet();

  private:
    virtual void __on_zero_shared() noexcept;

  protected:
    long __shared_owners_ = 0;
};

// ctype_base de Nintendo: las máscaras de glibc (las devuelve también __get_classname del sdk).
struct ctype_base {
    using mask = unsigned short;
    static constexpr mask upper = 0x100, lower = 0x200, alpha = 0x400, digit = 0x800, xdigit = 0x1000,
                          space = 0x2000, print = 0x4000, blank = 0x1, cntrl = 0x2, punct = 0x4, alnum = alpha | digit,
                          graph = alnum | punct;
};

// ctype<char> = {facet, tabla +0x10, del +0x18}; el juego lee la tabla en línea (is()) y llama do_widen (slot 7) y
// do_tolower/do_toupper por la vtable.
template <>
class ctype<char> : public locale::facet, public ctype_base {
  public:
    static locale::id id;

    constexpr explicit ctype(const mask* tab) : __tab_(tab) {}
    ~ctype() override;
    virtual char do_toupper(char c) const;
    virtual const char* do_toupper(char* lo, const char* hi) const;
    virtual char do_tolower(char c) const;
    virtual const char* do_tolower(char* lo, const char* hi) const;
    virtual char do_widen(char c) const;
    virtual const char* do_widen(const char* lo, const char* hi, char* to) const;
    virtual char do_narrow(char c, char dfault) const;
    virtual const char* do_narrow(const char* lo, const char* hi, char dfault, char* to) const;

    const mask* __tab_;
    bool __del_ = false;
};
static_assert(sizeof(ctype<char>) == 0x20);

// collate<char>: la usa regex_traits (transform) para rangos y clases de equivalencia.
template <>
class collate<char> : public locale::facet {
  public:
    static locale::id id;

    constexpr collate() = default;
    ~collate() override;
    virtual int do_compare(const char* lo1, const char* hi1, const char* lo2, const char* hi2) const;
    virtual string do_transform(const char* lo, const char* hi) const;
    virtual long do_hash(const char* lo, const char* hi) const;
};

class ios_base {
  public:
    using fmtflags = unsigned int;
    using iostate = unsigned int;
    using event_callback = void (*)(int event, ios_base& base, int index);

    static constexpr fmtflags dec = 0x2, hex = 0x8, internal = 0x10, left = 0x20, oct = 0x40, right = 0x80,
                              showbase = 0x200, showpos = 0x800, skipws = 0x1000, unitbuf = 0x2000,
                              uppercase = 0x4000, adjustfield = left | right | internal, basefield = dec | oct | hex;
    static constexpr iostate goodbit = 0, badbit = 1, eofbit = 2, failbit = 4;

    virtual ~ios_base();
    void init(void* sb);
    void clear(iostate state);
    locale getloc() const;
    void setstate(iostate state) { clear(__rdstate_ | state); }
    bool good() const { return __rdstate_ == 0; }

    // Los D0 de las clases de iostream liberan objetos del juego.
    static void operator delete(void* p) noexcept { nnsdk::host().game_delete(p); }

    fmtflags __fmtflags_;
    streamsize __precision_;
    streamsize __width_;
    iostate __rdstate_;
    iostate __exceptions_;
    void* __rdbuf_;
    void* __loc_;  // aquí vive la locale (un puntero)
    event_callback* __fn_;
    int* __index_;
    size_t __event_size_;
    size_t __event_cap_;
    long* __iarray_;
    size_t __iarray_size_;
    size_t __iarray_cap_;
    void** __parray_;
    size_t __parray_size_;
    size_t __parray_cap_;

  protected:
    ios_base() {}  // como libc++: lo rellena init()
};
static_assert(sizeof(ios_base) == 0x88);

template <>
class basic_ios<char, char_traits<char>> : public ios_base {
  public:
    ~basic_ios() override;
    basic_streambuf<char, char_traits<char>>* rdbuf() const {
        return static_cast<basic_streambuf<char, char_traits<char>>*>(__rdbuf_);
    }

    basic_ostream<char, char_traits<char>>* __tie_;
    int __fill_;

  protected:
    basic_ios() {}
    void init(basic_streambuf<char, char_traits<char>>* sb) {
        ios_base::init(sb);
        __tie_ = nullptr;
        __fill_ = -1;
    }
};
static_assert(sizeof(basic_ios<char, char_traits<char>>) == 0x98);

// fpos<mbstate_t> de musl: {mbstate_t de 8 bytes, streamoff}; va y vuelve en dos registros.
struct __pos {
    long state, off;
};

// Las virtuales de la base son las de libc++: el basic_stringbuf del juego hereda imbue, setbuf, sync, showmanyc,
// xsgetn, uflow y xsputn, y las llama con sus objetos.
template <>
class basic_streambuf<char, char_traits<char>> {
  public:
    virtual ~basic_streambuf();
    static void operator delete(void* p) noexcept { nnsdk::host().game_delete(p); }

    int pubsync() { return sync(); }
    streamsize sputn(const char* s, streamsize n) { return xsputn(s, n); }

    virtual void imbue(const locale& loc);
    virtual basic_streambuf* setbuf(char* s, streamsize n);
    virtual __pos seekoff(long off, int way, unsigned which);
    virtual __pos seekpos(__pos sp, unsigned which);
    virtual int sync();
    virtual streamsize showmanyc();
    virtual streamsize xsgetn(char* s, streamsize n);
    virtual int underflow();
    virtual int uflow();
    virtual int pbackfail(int c);
    virtual streamsize xsputn(const char* s, streamsize n);
    virtual int overflow(int c);

    locale __loc_;
    char* __binp_ = nullptr;
    char* __ninp_ = nullptr;
    char* __einp_ = nullptr;
    char* __bout_ = nullptr;
    char* __nout_ = nullptr;
    char* __eout_ = nullptr;

  protected:
    basic_streambuf();
};
static_assert(sizeof(basic_streambuf<char, char_traits<char>>) == 0x40);

template <>
class basic_istream<char, char_traits<char>> : virtual public basic_ios<char, char_traits<char>> {
  public:
    class sentry;

    ~basic_istream() override;
    streamsize __gc_;
};

// El de libc++: un bool. Lo construye en línea el operator>> del juego.
class basic_istream<char, char_traits<char>>::sentry {
  public:
    explicit sentry(basic_istream& is, bool noskipws);

    bool __ok_;
};

template <>
class basic_ostream<char, char_traits<char>> : virtual public basic_ios<char, char_traits<char>> {
  public:
    class sentry;

    explicit basic_ostream(basic_streambuf<char, char_traits<char>>* sb) { this->init(sb); }
    ~basic_ostream() override;
    basic_ostream& operator<<(int n);
    basic_ostream& operator<<(unsigned int n);
    basic_ostream& operator<<(unsigned long n);
    basic_ostream& operator<<(unsigned short n);
    basic_ostream& operator<<(const void* p);
    basic_ostream& flush();

  protected:
    basic_ostream() {}
};
static_assert(sizeof(basic_ostream<char, char_traits<char>>) == 0xa0);

class basic_ostream<char, char_traits<char>>::sentry {
  public:
    explicit sentry(basic_ostream& os);
    ~sentry();

    bool __ok_;
    basic_ostream& __os_;
};

template <>
class basic_iostream<char, char_traits<char>> : public basic_istream<char, char_traits<char>>,
                                                public basic_ostream<char, char_traits<char>> {
  public:
    ~basic_iostream() override;
};

}  // namespace __1
}  // namespace std

using std::__1::ios_base;
using std::__1::locale;
using std::__1::streamsize;
using ctype_char = std::__1::ctype<char>;
using collate_char = std::__1::collate<char>;
using streambuf = std::__1::basic_streambuf<char, std::__1::char_traits<char>>;
using ostream = std::__1::basic_ostream<char, std::__1::char_traits<char>>;

// cxx.cpp: string::append(const char*, size_type), sobre la memoria del juego.
extern "C" std::__1::string* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm(
    std::__1::string* self, const char* s, size_t n);

namespace {

std::__1::string make_string(const char* p, size_t n) {
    std::__1::string s;
    if (n) _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm(&s, p, n);
    return s;
}

[[noreturn]] void fail(const char* what) {
    printf("[libc++] %s\n", what);
    fflush(stdout);
    abort();
}

// La tabla clásica de ctype<char> de Nintendo (coincide entera con la del sdk): solo ASCII, de 0x80 en adelante 0.
struct ClassicTable {
    ctype_char::mask m[256];
};
constexpr ClassicTable classic_table() {
    using b = std::__1::ctype_base;
    ClassicTable t{};
    for (int c = 0; c < 128; c++) {
        b::mask m = 0;
        if (c >= 'A' && c <= 'Z') m |= b::upper | b::alpha;
        if (c >= 'a' && c <= 'z') m |= b::lower | b::alpha;
        if (c >= '0' && c <= '9') m |= b::digit | b::xdigit;
        if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) m |= b::xdigit;
        if (c == ' ' || (c >= '\t' && c <= '\r')) m |= b::space;
        if (c == ' ' || c == '\t') m |= b::blank;
        if (c >= 0x20 && c < 0x7f) m |= b::print;
        if (c < 0x20 || c == 0x7f) m |= b::cntrl;
        if (c > 0x20 && c < 0x7f && !(m & b::alnum)) m |= b::punct;
        t.m[c] = m;
    }
    return t;
}
constexpr ClassicTable kClassic = classic_table();
static_assert(kClassic.m[' '] == 0x6001 && kClassic.m['A'] == 0x5500 && kClassic.m['a'] == 0x5600 &&
              kClassic.m['0'] == 0x5800 && kClassic.m['\t'] == 0x2003);

// La única locale: su __locale_ apunta a su nombre.
constexpr char kClassicName[] = "C";
[[clang::no_destroy]] constinit ctype_char g_ctype(kClassic.m);
[[clang::no_destroy]] constinit collate_char g_collate;

// codecvt<char, char, mbstate_t> de la locale "C": no convierte nada (always_noconv). basic_filebuf la pide al
// construirse y, con always_noconv, lee y escribe los bytes tal cual. Virtuales en el orden de libc++, tras las de facet.
class codecvt_char : public std::__1::locale::facet {
  public:
    enum result { ok, partial, error, noconv };
    constexpr codecvt_char() = default;
    ~codecvt_char() override;
    virtual result do_out(mbstate_t&, const char* from, const char*, const char*& from_next, char* to, char*,
                          char*& to_next) const;
    virtual result do_in(mbstate_t&, const char* from, const char*, const char*& from_next, char* to, char*,
                         char*& to_next) const;
    virtual result do_unshift(mbstate_t&, char* to, char*, char*& to_next) const;
    virtual int do_encoding() const noexcept;
    virtual bool do_always_noconv() const noexcept;
    virtual int do_length(mbstate_t&, const char* from, const char* end, size_t max) const;
    virtual int do_max_length() const noexcept;
};
static_assert(sizeof(codecvt_char) == 16);

codecvt_char::~codecvt_char() {}
codecvt_char::result codecvt_char::do_out(mbstate_t&, const char* from, const char*, const char*& from_next, char* to,
                                          char*, char*& to_next) const {
    from_next = from;
    to_next = to;
    return noconv;
}
codecvt_char::result codecvt_char::do_in(mbstate_t&, const char* from, const char*, const char*& from_next, char* to,
                                         char*, char*& to_next) const {
    from_next = from;
    to_next = to;
    return noconv;
}
codecvt_char::result codecvt_char::do_unshift(mbstate_t&, char* to, char*, char*& to_next) const {
    to_next = to;
    return noconv;
}
int codecvt_char::do_encoding() const noexcept { return 1; }
bool codecvt_char::do_always_noconv() const noexcept { return true; }
int codecvt_char::do_length(mbstate_t&, const char* from, const char* end, size_t max) const {
    const size_t n = static_cast<size_t>(end - from);
    return static_cast<int>(n < max ? n : max);
}
int codecvt_char::do_max_length() const noexcept { return 1; }

[[clang::no_destroy]] constinit codecvt_char g_codecvt;

char to_upper(char c) { return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c; }
char to_lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

// El sentry de basic_ostream sin objeto (ver arriba: sin LSDA). begin = constructor, end = destructor.
bool begin_output(ostream& os) {
    if (!os.good()) return false;
    if (os.__tie_) os.__tie_->flush();
    return true;
}

// libc++ además lo salta si hay una excepción en vuelo; esa cuenta la lleva la libc++abi del loader, no la nuestra.
void end_output(ostream& os) {
    if (os.__rdbuf_ && os.good() && (os.__fmtflags_ & ios_base::unitbuf) && os.rdbuf()->pubsync() == -1)
        os.setstate(ios_base::badbit);
}

// basic_ios::fill(): el relleno se fija la primera vez (widen(' ') en la locale "C"); operator<< lo pide siempre.
char fill_char(ostream& os) {
    if (os.__fill_ == -1) os.__fill_ = ' ';
    return static_cast<char>(os.__fill_);
}

// __identify_padding de libc++: dónde va el relleno según adjustfield (internal: tras el signo o el 0x).
const char* padding_point(const char* nb, const char* ne, const ios_base& iob) {
    switch (iob.__fmtflags_ & ios_base::adjustfield) {
        case ios_base::internal:
            if (nb[0] == '-' || nb[0] == '+') return nb + 1;
            if (ne - nb >= 2 && nb[0] == '0' && (nb[1] == 'x' || nb[1] == 'X')) return nb + 2;
            return nb;
        case ios_base::left:
            return ne;
        default:
            return nb;
    }
}

// __pad_and_output de libc++: [nb, np) + relleno hasta width + [np, ne); si todo sale, width vuelve a 0.
bool pad_and_output(ostream& os, const char* nb, const char* np, const char* ne, char fill) {
    streambuf* sb = os.rdbuf();
    if (!sb) return false;
    streamsize fill_count = os.__width_ > ne - nb ? os.__width_ - (ne - nb) : 0;
    if (np > nb && sb->sputn(nb, np - nb) != np - nb) return false;
    if (fill_count > 0) {
        char pad[64];
        memset(pad, fill, sizeof pad);
        for (streamsize left = fill_count; left > 0; left -= static_cast<streamsize>(sizeof pad)) {
            streamsize n = left < static_cast<streamsize>(sizeof pad) ? left : sizeof pad;
            if (sb->sputn(pad, n) != n) return false;
        }
    }
    if (ne > np && sb->sputn(np, ne - np) != ne - np) return false;
    os.__width_ = 0;
    return true;
}

// num_put<char>::do_put (long / unsigned long) en la locale "C" (sin separadores de miles): printf con las banderas
// del stream (showpos, showbase, oct/hex/uppercase) como __format_int de libc++.
ostream& put_integer(ostream& os, unsigned long v, bool is_signed) {
    if (begin_output(os)) {
        const ios_base::fmtflags flags = os.__fmtflags_, base = flags & ios_base::basefield;
        char fmt[8] = "%";
        char* f = fmt + 1;
        if ((flags & ios_base::showpos) && base != ios_base::oct && base != ios_base::hex && is_signed) *f++ = '+';
        if (flags & ios_base::showbase) *f++ = '#';
        *f++ = 'l';
        *f++ = base == ios_base::oct   ? 'o'
               : base == ios_base::hex ? (flags & ios_base::uppercase ? 'X' : 'x')
               : is_signed             ? 'd'
                                       : 'u';
        *f = 0;
        char text[32];
        int len = is_signed ? snprintf(text, sizeof text, fmt, static_cast<long>(v))
                            : snprintf(text, sizeof text, fmt, v);
        if (!pad_and_output(os, text, padding_point(text, text + len, os), text + len, fill_char(os)))
            os.setstate(ios_base::badbit | ios_base::failbit);
    }
    end_output(os);
    return os;
}

// La streambuf de std::clog: lo que el juego escribe ahí va al log del runtime, por líneas.
class LogBuf : public streambuf {
  public:
    streamsize xsputn(const char* s, streamsize n) override {
        pthread_mutex_lock(&mutex_);
        for (streamsize k = 0; k < n; k++) put(s[k]);
        pthread_mutex_unlock(&mutex_);
        return n;
    }
    int overflow(int c) override {
        if (c == -1) return 0;
        pthread_mutex_lock(&mutex_);
        put(static_cast<char>(c));
        pthread_mutex_unlock(&mutex_);
        return c;
    }
    int sync() override {
        pthread_mutex_lock(&mutex_);
        flush_line();
        pthread_mutex_unlock(&mutex_);
        return 0;
    }

  private:
    void put(char c) {
        if (c == '\n' || len_ == sizeof line_) flush_line();
        if (c != '\n') line_[len_++] = c;
    }
    void flush_line() {
        if (!len_) return;
        printf("[clog] %.*s\n", static_cast<int>(len_), line_);
        fflush(stdout);
        len_ = 0;
    }

    char line_[512];
    size_t len_ = 0;
    pthread_mutex_t mutex_ = PTHREAD_MUTEX_INITIALIZER;
};

[[clang::no_destroy]] LogBuf g_clog_buf;

}  // namespace

namespace std {
namespace __1 {

locale::locale() noexcept : __locale_(kClassicName) {}
locale::locale(const locale& other) noexcept : __locale_(other.__locale_) {}
locale::~locale() {}
string locale::name() const { return make_string(kClassicName, 1); }

// use_facet(F::id): el juego importa los id de ctype<char>, collate<char> y codecvt<char, char, mbstate_t> (Xenoblade,
// para su basic_filebuf); otra faceta sería un bad_cast (std::terminate en la consola).
bool locale::has_facet(id& x) const {
    return &x == &ctype<char>::id || &x == &collate<char>::id || &x == &_ZNSt3__17codecvtIcc11__mbstate_tE2idE;
}
const locale::facet* locale::use_facet(id& x) const {
    if (&x == &ctype<char>::id) return &g_ctype;
    if (&x == &collate<char>::id) return &g_collate;
    if (&x == &_ZNSt3__17codecvtIcc11__mbstate_tE2idE) return &g_codecvt;
    printf("[libc++] use_facet: la locale \"C\" no tiene la faceta del id %p\n", static_cast<void*>(&x));
    fflush(stdout);
    abort();
}

locale::facet::~facet() {}
void locale::facet::__on_zero_shared() noexcept {}  // las facetas son estáticas

locale::id ctype<char>::id;
locale::id collate<char>::id;
extern "C" {
locale::id _ZNSt3__17codecvtIcc11__mbstate_tE2idE;
}

ctype<char>::~ctype() {}
char ctype<char>::do_toupper(char c) const { return to_upper(c); }
const char* ctype<char>::do_toupper(char* lo, const char* hi) const {
    for (; lo != hi; ++lo) *lo = to_upper(*lo);
    return lo;
}
char ctype<char>::do_tolower(char c) const { return to_lower(c); }
const char* ctype<char>::do_tolower(char* lo, const char* hi) const {
    for (; lo != hi; ++lo) *lo = to_lower(*lo);
    return lo;
}
char ctype<char>::do_widen(char c) const { return c; }
const char* ctype<char>::do_widen(const char* lo, const char* hi, char* to) const {
    memcpy(to, lo, static_cast<size_t>(hi - lo));
    return hi;
}
char ctype<char>::do_narrow(char c, char dfault) const { return static_cast<unsigned char>(c) < 0x80 ? c : dfault; }
const char* ctype<char>::do_narrow(const char* lo, const char* hi, char dfault, char* to) const {
    for (; lo != hi; ++lo, ++to) *to = static_cast<unsigned char>(*lo) < 0x80 ? *lo : dfault;
    return hi;
}

collate<char>::~collate() {}
// Orden lexicográfico de char (sin signo en AArch64, en los dos lados).
int collate<char>::do_compare(const char* lo1, const char* hi1, const char* lo2, const char* hi2) const {
    for (; lo2 != hi2; ++lo1, ++lo2) {
        if (lo1 == hi1 || *lo1 < *lo2) return -1;
        if (*lo2 < *lo1) return 1;
    }
    return lo1 != hi1;
}
string collate<char>::do_transform(const char* lo, const char* hi) const {
    return make_string(lo, static_cast<size_t>(hi - lo));
}
// El hash de libc++ (PJW), por si el juego guarda el resultado.
long collate<char>::do_hash(const char* lo, const char* hi) const {
    size_t h = 0;
    constexpr size_t sr = 8 * sizeof(size_t) - 8, mask = size_t{0xF} << (sr + 4);
    for (const char* p = lo; p != hi; ++p) {
        h = (h << 4) + static_cast<unsigned char>(*p);
        size_t g = h & mask;
        h ^= g | (g >> sr);
    }
    return static_cast<long>(h);
}

// ~ios_base de libc++ (la de Nintendo no comprueba __loc_): callbacks de erase_event (register_callback no se
// importa, así que no hay), la locale y los arrays de iword/pword, que reservaría su malloc.
ios_base::~ios_base() {
    for (size_t k = __event_size_; k > 0; k--) __fn_[k - 1](0, *this, __index_[k - 1]);
    reinterpret_cast<locale*>(&__loc_)->~locale();
    void* arrays[] = {reinterpret_cast<void*>(__fn_), __index_, __iarray_, __parray_};
    for (void* p : arrays)
        if (p) nnsdk::host().game_free(p);
}

void ios_base::init(void* sb) {
    __rdbuf_ = sb;
    __rdstate_ = sb ? goodbit : badbit;
    __exceptions_ = goodbit;
    __fmtflags_ = skipws | dec;
    __width_ = 0;
    __precision_ = 6;
    __fn_ = nullptr;
    __index_ = nullptr;
    __event_size_ = __event_cap_ = 0;
    __iarray_ = nullptr;
    __iarray_size_ = __iarray_cap_ = 0;
    __parray_ = nullptr;
    __parray_size_ = __parray_cap_ = 0;
    ::new (&__loc_) locale;
}

// Sin rdbuf el estado siempre lleva badbit. Si el juego pidió excepciones para ese estado, libc++ lanzaría
// ios_base::failure: una nuestra no la puede capturar el juego (otra libc++abi), así que es fatal.
void ios_base::clear(iostate state) {
    __rdstate_ = __rdbuf_ ? state : state | badbit;
    if (__rdstate_ & __exceptions_) fail("ios_base::clear: ios_base::failure");
}

locale ios_base::getloc() const { return *reinterpret_cast<const locale*>(&__loc_); }

basic_ios<char, char_traits<char>>::~basic_ios() {}

basic_streambuf<char, char_traits<char>>::basic_streambuf() {}
basic_streambuf<char, char_traits<char>>::~basic_streambuf() {}
void basic_streambuf<char, char_traits<char>>::imbue(const locale&) {}
basic_streambuf<char, char_traits<char>>* basic_streambuf<char, char_traits<char>>::setbuf(char*, streamsize) {
    return this;
}
__pos basic_streambuf<char, char_traits<char>>::seekoff(long, int, unsigned) { return {0, -1}; }
__pos basic_streambuf<char, char_traits<char>>::seekpos(__pos, unsigned) { return {0, -1}; }
int basic_streambuf<char, char_traits<char>>::sync() { return 0; }
streamsize basic_streambuf<char, char_traits<char>>::showmanyc() { return 0; }
int basic_streambuf<char, char_traits<char>>::underflow() { return -1; }
int basic_streambuf<char, char_traits<char>>::pbackfail(int) { return -1; }
int basic_streambuf<char, char_traits<char>>::overflow(int) { return -1; }

int basic_streambuf<char, char_traits<char>>::uflow() {
    if (underflow() == -1) return -1;
    return static_cast<unsigned char>(*__ninp_++);
}

// Lo que haya en [ninp, einp) y luego carácter a carácter con uflow() (del streambuf del juego).
streamsize basic_streambuf<char, char_traits<char>>::xsgetn(char* s, streamsize n) {
    streamsize done = 0;
    while (done < n) {
        if (__ninp_ < __einp_) {
            streamsize chunk = __einp_ - __ninp_ < n - done ? __einp_ - __ninp_ : n - done;
            memcpy(s + done, __ninp_, static_cast<size_t>(chunk));
            __ninp_ += chunk;
            done += chunk;
        } else {
            int c = uflow();
            if (c == -1) break;
            s[done++] = static_cast<char>(c);
        }
    }
    return done;
}

// Lo que quepa en [nout, eout) y el resto carácter a carácter con overflow() (el stringbuf del juego crece ahí).
streamsize basic_streambuf<char, char_traits<char>>::xsputn(const char* s, streamsize n) {
    streamsize done = 0;
    while (done < n) {
        if (__nout_ < __eout_) {
            streamsize chunk = __eout_ - __nout_ < n - done ? __eout_ - __nout_ : n - done;
            memcpy(__nout_, s + done, static_cast<size_t>(chunk));
            __nout_ += chunk;
            done += chunk;
        } else {
            if (overflow(static_cast<unsigned char>(s[done])) == -1) break;
            done++;
        }
    }
    return done;
}

basic_istream<char, char_traits<char>>::~basic_istream() {}
basic_ostream<char, char_traits<char>>::~basic_ostream() {}
basic_iostream<char, char_traits<char>>::~basic_iostream() {}

basic_ostream<char, char_traits<char>>::sentry::sentry(basic_ostream& os) : __ok_(begin_output(os)), __os_(os) {}

// Como libc++: con el stream bien, vacía el ostream atado y, si toca (skipws y sin noskipws), salta los espacios de
// la locale "C"; si se acaba la entrada, failbit | eofbit. Sin el stream bien, failbit.
basic_istream<char, char_traits<char>>::sentry::sentry(basic_istream& is, bool noskipws) : __ok_(false) {
    if (!is.good()) {
        is.setstate(ios_base::failbit);
        return;
    }
    if (is.__tie_) is.__tie_->flush();
    if (!noskipws && (is.__fmtflags_ & ios_base::skipws)) {
        basic_streambuf<char, char_traits<char>>* sb = is.rdbuf();
        for (;;) {
            int c = sb->__ninp_ < sb->__einp_ ? static_cast<unsigned char>(*sb->__ninp_) : sb->underflow();
            if (c == -1) {
                is.setstate(ios_base::failbit | ios_base::eofbit);
                break;
            }
            if (!(kClassic.m[c] & ctype_base::space)) break;
            if (sb->__ninp_ < sb->__einp_)
                sb->__ninp_++;
            else
                sb->uflow();
        }
    }
    __ok_ = is.good();
}
basic_ostream<char, char_traits<char>>::sentry::~sentry() { end_output(__os_); }

basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::flush() {
    if (rdbuf()) {
        if (begin_output(*this) && rdbuf()->pubsync() == -1) setstate(badbit);
        end_output(*this);
    }
    return *this;
}

// libc++ pasa int como long, salvo en oct/hex, donde va como su valor sin signo; los sin signo van como unsigned long.
basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::operator<<(int n) {
    const fmtflags base = __fmtflags_ & basefield;
    const long v = base == oct || base == hex ? static_cast<long>(static_cast<unsigned int>(n)) : n;
    return put_integer(*this, static_cast<unsigned long>(v), true);
}
basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::operator<<(unsigned int n) {
    return put_integer(*this, n, false);
}
basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::operator<<(unsigned long n) {
    return put_integer(*this, n, false);
}
basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::operator<<(unsigned short n) {
    return put_integer(*this, n, false);
}

// %p de printf, con el mismo relleno que los números.
basic_ostream<char, char_traits<char>>& basic_ostream<char, char_traits<char>>::operator<<(const void* p) {
    if (begin_output(*this)) {
        char text[32];
        int len = snprintf(text, sizeof text, "%p", p);
        if (!pad_and_output(*this, text, padding_point(text, text + len, *this), text + len, fill_char(*this)))
            setstate(badbit | failbit);
    }
    end_output(*this);
    return *this;
}

// std::clog: el juego vuelca ahí algunos avisos.
[[clang::no_destroy]] basic_ostream<char, char_traits<char>> clog(&g_clog_buf);

// regex_traits<char>::lookup_classname ([[:alpha:]], \d, \w…) con las máscaras de Nintendo. \w (__regex_word, 0x80)
// sale como alnum|upper|lower|0x80; con icase, upper o lower incluyen alpha.
ctype_base::mask __get_classname(const char* s, bool icase) {
    using b = ctype_base;
    static const struct {
        const char* name;
        b::mask mask;
    } kClasses[] = {{"alnum", b::alnum}, {"alpha", b::alpha}, {"blank", b::blank},   {"cntrl", b::cntrl},
                    {"d", b::digit},     {"digit", b::digit}, {"graph", b::graph},   {"lower", b::lower},
                    {"print", b::print}, {"punct", b::punct}, {"s", b::space},       {"space", b::space},
                    {"upper", b::upper}, {"w", 0x80},         {"xdigit", b::xdigit}};
    for (const auto& c : kClasses) {
        if (strcmp(s, c.name) != 0) continue;
        if (c.mask == 0x80) return 0x80 | b::alnum | b::upper | b::lower;
        if (icase && (c.mask & (b::lower | b::upper))) return c.mask | b::alpha;
        return c.mask;
    }
    return 0;
}

// regex_traits<char>::lookup_collatename ([[.space.]]…): los nombres POSIX del juego de caracteres portable, en el
// orden de strcmp como la tabla de libc++. Devuelve el carácter como string, o vacío si no existe.
string __get_collation_name(const char* s) {
    static const struct {
        const char* name;
        char c;
    } kNames[] = {
        {"A", 'A'}, {"B", 'B'}, {"C", 'C'}, {"D", 'D'}, {"E", 'E'}, {"F", 'F'}, {"G", 'G'}, {"H", 'H'}, {"I", 'I'},
        {"J", 'J'}, {"K", 'K'}, {"L", 'L'}, {"M", 'M'}, {"N", 'N'}, {"NUL", 0}, {"O", 'O'}, {"P", 'P'}, {"Q", 'Q'},
        {"R", 'R'}, {"S", 'S'}, {"T", 'T'}, {"U", 'U'}, {"V", 'V'}, {"W", 'W'}, {"X", 'X'}, {"Y", 'Y'}, {"Z", 'Z'},
        {"a", 'a'}, {"alert", '\a'}, {"ampersand", '&'}, {"apostrophe", '\''}, {"asterisk", '*'}, {"b", 'b'},
        {"backslash", '\\'}, {"backspace", '\b'}, {"c", 'c'}, {"carriage-return", '\r'}, {"circumflex", '^'},
        {"circumflex-accent", '^'}, {"colon", ':'}, {"comma", ','}, {"commercial-at", '@'}, {"d", 'd'},
        {"dollar-sign", '$'}, {"e", 'e'}, {"eight", '8'}, {"equals-sign", '='}, {"exclamation-mark", '!'},
        {"f", 'f'}, {"five", '5'}, {"form-feed", '\f'}, {"four", '4'}, {"full-stop", '.'}, {"g", 'g'},
        {"grave-accent", '`'}, {"greater-than-sign", '>'}, {"h", 'h'}, {"hyphen", '-'}, {"hyphen-minus", '-'},
        {"i", 'i'}, {"j", 'j'}, {"k", 'k'}, {"l", 'l'}, {"left-brace", '{'}, {"left-curly-bracket", '{'},
        {"left-parenthesis", '('}, {"left-square-bracket", '['}, {"less-than-sign", '<'}, {"low-line", '_'},
        {"m", 'm'}, {"n", 'n'}, {"newline", '\n'}, {"nine", '9'}, {"number-sign", '#'}, {"o", 'o'}, {"one", '1'},
        {"p", 'p'}, {"percent-sign", '%'}, {"period", '.'}, {"plus-sign", '+'}, {"q", 'q'}, {"question-mark", '?'},
        {"quotation-mark", '"'}, {"r", 'r'}, {"reverse-solidus", '\\'}, {"right-brace", '}'},
        {"right-curly-bracket", '}'}, {"right-parenthesis", ')'}, {"right-square-bracket", ']'}, {"s", 's'},
        {"semicolon", ';'}, {"seven", '7'}, {"six", '6'}, {"slash", '/'}, {"solidus", '/'}, {"space", ' '},
        {"t", 't'}, {"tab", '\t'}, {"three", '3'}, {"tilde", '~'}, {"two", '2'}, {"u", 'u'}, {"underscore", '_'},
        {"v", 'v'}, {"vertical-line", '|'}, {"vertical-tab", '\v'}, {"w", 'w'}, {"x", 'x'}, {"y", 'y'}, {"z", 'z'},
        {"zero", '0'},
    };
    static_assert(sizeof kNames / sizeof kNames[0] == 111);
    for (const auto& n : kNames)
        if (strcmp(s, n.name) == 0) return make_string(&n.c, 1);
    return {};
}

}  // namespace __1
}  // namespace std

extern "C" {

// El '.' de std::regex: acepta un carácter que no sea \n ni \r y pasa al nodo siguiente. __state<char>: __do_ en +0,
// __current_ +0x10, __last_ +0x18, __node_ +0x50; el siguiente nodo (first()) está en +8 de este.
struct RegexState {
    int do_;
    const char* first;
    const char* current;
    const char* last;
    char vectors[0x30];  // __sub_matches_, __loop_data_
    const void* node;
};
static_assert(offsetof(RegexState, node) == 0x50);

void _ZNKSt3__123__match_any_but_newlineIcE6__execERNS_7__stateIcEE(const void* self, RegexState* s) {
    constexpr int kAcceptAndConsume = -995, kReject = -993;
    if (s->current != s->last && *s->current != '\n' && *s->current != '\r') {
        s->do_ = kAcceptAndConsume;
        s->current++;
        s->node = *reinterpret_cast<const void* const*>(static_cast<const char*>(self) + 8);
    } else {
        s->do_ = kReject;
        s->node = nullptr;
    }
}

}  // extern "C"
