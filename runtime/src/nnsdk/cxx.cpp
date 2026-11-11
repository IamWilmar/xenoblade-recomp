// La libc++ de Nintendo (std::__1) que importa el juego, salvo locale/iostreams/regex (cxx_ios.cpp). El layout es
// el de libc++ ABI v1, igual que en el NDK, pero se implementan a mano: la memoria de los strings tiene que salir del
// operator new del juego, que es quien la libera después. Se definen con su nombre mangleado porque std::__1 no
// existe en el NDK (usa std::__ndk1); lo que necesita vtable o typeinfo (shared_ptr) se declara como clase de verdad.
//
// Lo que está fuera de std::__1 (std::exception, logic_error, sus typeinfo/vtables, std::nothrow, __cxa_throw y
// compañía, _Unwind_*) no se puede definir aquí: la libc++abi/libunwind estática de libnnsdk ya define esos nombres
// (el enlazado daría "duplicate symbol") y --exclude-libs los oculta. Los tiene que dar el loader (host_symbols.c),
// que es la misma copia de libc++abi que ya le da al juego __cxa_begin_catch, la personality y las vtables de
// __cxxabiv1: el juego tiene que lanzar, capturar y comparar typeinfo con una sola copia.
//
// Las excepciones que en la Switch lanzaría la propia libc++ (out_of_range, length_error, stoul sin número…) aquí
// terminan el proceso con un mensaje: una excepción de nuestra copia no llegaría al juego (nuestra libunwind no ve sus
// FDE y él captura con la copia del loader), y el juego no tiene catch con tipo (sus 14 catch son catch (...), casi
// todos de limpiar y relanzar), así que en la consola acabaría igual en terminate.

#include <errno.h>
#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <random>
#include <typeinfo>

#include "nnsdk/os.h"
#include "nnsdk/runtime.h"

using namespace nn;

namespace {

constexpr size_t kShortMax = 22;

// basic_string<char>: largo = {capacidad reservada | 1, tamaño, datos}; corto = {tamaño << 1, 23 bytes}.
struct String {
    union {
        struct {
            size_t cap;
            size_t size;
            char* data;
        } l;
        struct {
            unsigned char size;
            char data[23];
        } s;
    };
    bool is_long() const { return s.size & 1; }
    size_t length() const { return is_long() ? l.size : s.size >> 1; }
    const char* chars() const { return is_long() ? l.data : s.data; }
    char* chars() { return is_long() ? l.data : s.data; }
    // Caracteres que caben sin realojar (sin contar el 0 final).
    size_t capacity() const { return is_long() ? (l.cap & ~size_t{1}) - 1 : kShortMax; }
};
static_assert(sizeof(String) == 24);

// Escribe [p, p+n) en str. Si str ya tenía contenido (fresh = false) libera su buffer al final,
// por si p apunta dentro de él.
void set_chars(String* str, const char* p, size_t n, bool fresh) {
    char* old = !fresh && str->is_long() ? str->l.data : nullptr;
    if (old && (str->l.cap & ~size_t{1}) > n) {
        memmove(old, p, n);
        old[n] = 0;
        str->l.size = n;
        return;
    }
    if (n <= kShortMax) {
        char tmp[kShortMax + 1];
        memcpy(tmp, p, n);
        str->s.size = static_cast<unsigned char>(n << 1);
        memcpy(str->s.data, tmp, n);
        str->s.data[n] = 0;
    } else {
        size_t alloc = (n + 16) & ~size_t{15};
        auto* d = static_cast<char*>(nnsdk::host().game_new(alloc));
        memcpy(d, p, n);
        d[n] = 0;
        str->l.cap = alloc | 1;
        str->l.size = n;
        str->l.data = d;
    }
    if (old) nnsdk::host().game_delete(old);
}

// Fija el tamaño (y el 0 final) sin tocar el buffer.
void set_length(String* str, size_t n) {
    if (str->is_long())
        str->l.size = n;
    else
        str->s.size = static_cast<unsigned char>(n << 1);
    str->chars()[n] = 0;
}

// Pasa el contenido a un buffer del juego con sitio para n caracteres (redondeado a 16 como __recommend de libc++).
void grow(String* str, size_t n) {
    const size_t len = str->length(), alloc = (n + 16) & ~size_t{15};
    auto* d = static_cast<char*>(nnsdk::host().game_new(alloc));
    memcpy(d, str->chars(), len + 1);
    if (str->is_long()) nnsdk::host().game_delete(str->l.data);
    str->l.cap = alloc | 1;
    str->l.size = len;
    str->l.data = d;
}

// Sustituye [pos, pos+n_old) por [p, p+n). Si no cabe crece como __grow_by de libc++ (al menos al doble).
// p puede apuntar dentro del propio string (s.append(s.c_str())): entonces se copia antes de mover nada.
void replace_chars(String* str, size_t pos, size_t n_old, const char* p, size_t n) {
    const char* cur = str->chars();
    const size_t len = str->length();
    if (n && p >= cur && p <= cur + len) {
        char tmp[256];
        char* copy = n <= sizeof tmp ? tmp : static_cast<char*>(malloc(n));
        memcpy(copy, p, n);
        replace_chars(str, pos, n_old, copy, n);
        if (copy != tmp) free(copy);
        return;
    }
    const size_t total = len - n_old + n, cap = str->capacity();
    if (total > cap) grow(str, std::max(total, 2 * cap));
    char* d = str->chars();
    memmove(d + pos + n, d + pos + n_old, len - pos - n_old);
    if (n) memcpy(d + pos, p, n);
    set_length(str, total);
}

// basic_string<wchar_t> (wchar_t de 4 bytes): corto = {tamaño << 1, relleno, 5 wchar_t desde +4}; largo como el de char,
// con la capacidad en elementos.
struct WString {
    union {
        struct {
            size_t cap;
            size_t size;
            wchar_t* data;
        } l;
        struct {
            unsigned char size;
            wchar_t data[5];
        } s;
    };
};
static_assert(sizeof(WString) == 24 && offsetof(WString, s.data) == 4);

[[noreturn]] void fail(const char* what) {
    printf("[libc++] %s\n", what);
    fflush(stdout);
    abort();
}

int64_t clock_ns(clockid_t clock) {
    timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

// unique_lock<mutex> = {mutex*, bool}: el mutex es el primer campo.
os::MutexType* lock_mutex(void* unique_lock) { return *static_cast<os::MutexType**>(unique_lock); }

String from_number(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
String from_number(const char* fmt, ...) {
    char buf[32];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    String r{};
    set_chars(&r, buf, static_cast<size_t>(n), true);
    return r;
}

// call_once: un mutex y una condición para todas las once_flag, como libc++.
pthread_mutex_t g_once_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t g_once_cond = PTHREAD_COND_INITIALIZER;

// Los 16 mutex de los shared_ptr atómicos (atomic_load(&sp)…). __sp_mut es opaco para el juego: solo recibe la
// referencia y llama lock/unlock, así que aquí el __sp_mut es directamente el pthread_mutex_t (a cero = el
// PTHREAD_MUTEX_INITIALIZER de bionic).
pthread_mutex_t g_sp_mutex[16] = {};

// El mt19937 de la libc++ de Nintendo: sus palabras son uint_fast32_t = 32 bits en musl (en bionic son 64, por eso
// no vale std::mt19937 del NDK).
using Mt19937 = std::mersenne_twister_engine<uint32_t, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680,
                                             15, 0xefc60000, 18, 1812433253>;
static_assert(sizeof(Mt19937) == 0x9c8);

}  // namespace

extern "C" {

// basic_string(const basic_string&)
void _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_(String* self, const String* other) {
    set_chars(self, other->chars(), other->length(), true);
}

// basic_string(const basic_string& str, size_type pos, size_type n, const allocator&): substr.
void _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_mmRKS4_(String* self, const String* str,
                                                                                    size_t pos, size_t n, const void*) {
    const size_t len = str->length();
    if (pos > len) fail("basic_string(str, pos, n): out_of_range");
    set_chars(self, str->chars() + pos, std::min(n, len - pos), true);
}

// basic_string& operator=(const basic_string&)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEaSERKS5_(String* self, const String* other) {
    if (self != other) set_chars(self, other->chars(), other->length(), false);
    return self;
}

// basic_string& operator=(char)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEaSEc(String* self, char c) {
    set_chars(self, &c, 1, false);
    return self;
}

// basic_string& assign(const char*)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6assignEPKc(String* self, const char* s) {
    set_chars(self, s, strlen(s), false);
    return self;
}

// basic_string& append(const char*) / append(const char*, size_type)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm(String* self, const char* s,
                                                                                   size_t n) {
    replace_chars(self, self->length(), 0, s, n);
    return self;
}
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKc(String* self, const char* s) {
    replace_chars(self, self->length(), 0, s, strlen(s));
    return self;
}

// basic_string& insert(size_type pos, const char*)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6insertEmPKc(String* self, size_t pos,
                                                                                   const char* s) {
    if (pos > self->length()) fail("basic_string::insert: out_of_range");
    replace_chars(self, pos, 0, s, strlen(s));
    return self;
}

void _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9push_backEc(String* self, char c) {
    replace_chars(self, self->length(), 0, &c, 1);
}

// int compare(size_type pos, size_type n1, const char* s, size_type n2) const
int _ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEmmPKcm(const String* self, size_t pos,
                                                                                    size_t n1, const char* s,
                                                                                    size_t n2) {
    size_t size = self->length();
    if (pos > size) fail("basic_string::compare: out_of_range");
    size_t len = n1 < size - pos ? n1 : size - pos;
    int r = memcmp(self->chars() + pos, s, len < n2 ? len : n2);
    if (r) return r;
    return len < n2 ? -1 : len > n2 ? 1 : 0;
}

// int compare(const char*) const
int _ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEPKc(const String* self, const char* s) {
    return _ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEmmPKcm(self, 0, SIZE_MAX, s,
                                                                                           strlen(s));
}

// operator+(const char*, const string&)
String _ZNSt3__1plIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_12basic_stringIT_T0_T1_EEPKS6_RKS9_(const char* lhs,
                                                                                                 const String* rhs) {
    String r{};
    set_chars(&r, lhs, strlen(lhs), true);
    replace_chars(&r, r.length(), 0, rhs->chars(), rhs->length());
    return r;
}

// wstring: __init(size_type n, wchar_t c) = n copias de c (wstring(n, c)).
void _ZNSt3__112basic_stringIwNS_11char_traitsIwEENS_9allocatorIwEEE6__initEmw(WString* self, size_t n, wchar_t c) {
    if (n >= 0x3ffffffffffffff0) fail("wstring: length_error");
    wchar_t* d;
    if (n < 5) {
        self->s.size = static_cast<unsigned char>(n << 1);
        d = self->s.data;
    } else {
        const size_t count = (n + 4) & ~size_t{3};  // __recommend(n) + 1: múltiplo de 16 bytes
        d = static_cast<wchar_t*>(nnsdk::host().game_new(count * sizeof(wchar_t)));
        self->l.cap = count | 1;
        self->l.size = n;
        self->l.data = d;
    }
    for (size_t k = 0; k < n; k++) d[k] = c;
    d[n] = 0;
}

void _ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv() { fail("basic_string: length_error"); }
void _ZNKSt3__121__basic_string_commonILb1EE20__throw_out_of_rangeEv() { fail("basic_string: out_of_range"); }
void _ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv() { fail("vector: length_error"); }
void _ZNKSt3__120__vector_base_commonILb1EE20__throw_out_of_rangeEv() { fail("vector: out_of_range"); }

// to_string: printf en la locale "C", como libc++.
String _ZNSt3__19to_stringEi(int v) { return from_number("%d", v); }
String _ZNSt3__19to_stringEj(unsigned v) { return from_number("%u", v); }
String _ZNSt3__19to_stringEl(long v) { return from_number("%ld", v); }
String _ZNSt3__19to_stringEm(unsigned long v) { return from_number("%lu", v); }

// unsigned long stoul(const string&, size_t* idx, int base): strtoul conservando el errno del juego.
unsigned long _ZNSt3__15stoulERKNS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEPmi(const String* str,
                                                                                              size_t* idx, int base) {
    const char* p = str->chars();
    char* end;
    int saved = errno;
    errno = 0;
    unsigned long r = strtoul(p, &end, base);
    std::swap(errno, saved);
    if (saved == ERANGE) fail("stoul: out of range");
    if (end == p) fail("stoul: no conversion");
    if (idx) *idx = static_cast<size_t>(end - p);
    return r;
}

// Primo >= n, para el número de cubetas de unordered_map.
size_t _ZNSt3__112__next_primeEm(size_t n) {
    if (n <= 2) return 2;
    for (n |= 1;; n += 2) {
        bool prime = true;
        for (size_t d = 3; d * d <= n && prime; d += 2) prime = n % d != 0;
        if (prime) return n;
    }
}

// std::sort de enteros (las instancias que la libc++ trae compiladas: int en MH Rise, el resto en Xenoblade).
void _ZNSt3__16__sortIRNS_6__lessIiiEEPiEEvT0_S5_T_(int* first, int* last, void*) { std::sort(first, last); }
void _ZNSt3__16__sortIRNS_6__lessIhhEEPhEEvT0_S5_T_(uint8_t* first, uint8_t* last, void*) { std::sort(first, last); }
void _ZNSt3__16__sortIRNS_6__lessIttEEPtEEvT0_S5_T_(uint16_t* first, uint16_t* last, void*) { std::sort(first, last); }
void _ZNSt3__16__sortIRNS_6__lessIjjEEPjEEvT0_S5_T_(uint32_t* first, uint32_t* last, void*) { std::sort(first, last); }
void _ZNSt3__16__sortIRNS_6__lessImmEEPmEEvT0_S5_T_(uint64_t* first, uint64_t* last, void*) { std::sort(first, last); }

// std::align(alignment, size, ptr, space): como libc++, nullptr y nada cambia si no cabe.
void* _ZNSt3__15alignEmmRPvRm(size_t alignment, size_t size, void** ptr, size_t* space) {
    const auto p = reinterpret_cast<uintptr_t>(*ptr);
    const uintptr_t aligned = (p + alignment - 1) & ~(alignment - 1);
    const size_t pad = aligned - p;
    if (size > *space || pad > *space - size) return nullptr;
    *space -= pad;
    *ptr = reinterpret_cast<void*>(aligned);
    return *ptr;
}

// operator new(size_t, const nothrow_t&): con el operator new del juego, para que su delete lo libere.
void* _ZnwmRKSt9nothrow_t(size_t size, const void*) { return nnsdk::host().game_new(size); }

// chrono: steady_clock en nanosegundos, system_clock en microsegundos (libc++).
int64_t _ZNSt3__16chrono12steady_clock3nowEv() { return clock_ns(CLOCK_MONOTONIC); }
int64_t _ZNSt3__16chrono12system_clock3nowEv() { return clock_ns(CLOCK_REALTIME) / 1000; }
int64_t _ZNSt3__16chrono12system_clock11from_time_tEl(long t) { return t * 1'000'000; }

// this_thread::sleep_for(const nanoseconds&)
void _ZNSt3__111this_thread9sleep_forERKNS_6chrono8durationIxNS_5ratioILl1ELl1000000000EEEEE(const int64_t* ns) {
    if (*ns <= 0) return;
    timespec ts{static_cast<time_t>(*ns / 1'000'000'000), static_cast<long>(*ns % 1'000'000'000)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
    }
}

// this_thread::get_id(): basta con que sea distinto de 0 y único por hilo vivo (el juego no crea std::thread).
pthread_t _ZNSt3__130__libcpp_thread_get_current_idEv() { return pthread_self(); }

// call_once: el juego mira en línea si la flag ya es ~0 (hecho) y solo llama aquí si no; 1 = otro hilo la está
// ejecutando. La función corre sin el mutex, como en libc++. Si lanza, la excepción (del juego) atraviesa este
// marco y la flag se queda en 1, donde libc++ la volvería a 0: no hay catch aquí para que la unwind de la copia del
// loader no tenga que pasar por la personality de la nuestra.
void _ZNSt3__111__call_onceERVmPvPFvS2_E(volatile unsigned long* flag, void* arg, void (*func)(void*)) {
    pthread_mutex_lock(&g_once_mutex);
    while (*flag == 1) pthread_cond_wait(&g_once_cond, &g_once_mutex);
    if (*flag != 0) {
        pthread_mutex_unlock(&g_once_mutex);
        return;
    }
    *flag = 1;
    pthread_mutex_unlock(&g_once_mutex);
    func(arg);
    pthread_mutex_lock(&g_once_mutex);
    __atomic_store_n(flag, ~0ul, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_once_mutex);
    pthread_cond_broadcast(&g_once_cond);
}

// __sp_mut& __get_sp_mut(const void*) y __sp_mut::lock()/unlock().
pthread_mutex_t* _ZNSt3__112__get_sp_mutEPKv(const void* p) {
    return &g_sp_mutex[(reinterpret_cast<uintptr_t>(p) >> 4) % 16];
}
void _ZNSt3__18__sp_mut4lockEv(pthread_mutex_t* m) { pthread_mutex_lock(m); }
void _ZNSt3__18__sp_mut6unlockEv(pthread_mutex_t* m) { pthread_mutex_unlock(m); }

// random_device: en la Switch es un mt19937 con la semilla por defecto (5489), sin entropía: da la misma secuencia en
// cada arranque. Se hace igual, con el mersenne_twister del NDK sobre el mismo objeto: mismo layout (624 palabras de
// 32 bits + índice), que son los 0x9c8 bytes que el juego reserva. El token se ignora, como allí.
void _ZNSt3__113random_deviceC1ERKNS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE(void* self,
                                                                                            const String*) {
    new (self) Mt19937();
}
void _ZNSt3__113random_deviceD1Ev(void*) {}
unsigned _ZNSt3__113random_deviceclEv(void* self) { return (*static_cast<Mt19937*>(self))(); }

// std::mutex y std::condition_variable: por dirección, sobre los de nn::os.
void _ZNSt3__15mutex4lockEv(os::MutexType* m) { os::LockMutex(m); }
void _ZNSt3__15mutex6unlockEv(os::MutexType* m) { os::UnlockMutex(m); }
void _ZNSt3__15mutexD1Ev(os::MutexType* m) { os::FinalizeMutex(m); }

void _ZNSt3__118condition_variable10notify_oneEv(os::ConditionVariableType* c) { os::SignalConditionVariable(c); }
void _ZNSt3__118condition_variable10notify_allEv(os::ConditionVariableType* c) { os::BroadcastConditionVariable(c); }
void _ZNSt3__118condition_variableD1Ev(os::ConditionVariableType* c) { os::FinalizeConditionVariable(c); }
void _ZNSt3__118condition_variable4waitERNS_11unique_lockINS_5mutexEEE(os::ConditionVariableType* c, void* lock) {
    os::WaitConditionVariable(c, lock_mutex(lock));
}
// __do_timed_wait(unique_lock&, time_point<system_clock, nanoseconds>): plazo absoluto en ns.
void _ZNSt3__118condition_variable15__do_timed_waitERNS_11unique_lockINS_5mutexEEENS_6chrono10time_pointINS5_12system_clockENS5_8durationIxNS_5ratioILl1ELl1000000000EEEEEEE(
    os::ConditionVariableType* c, void* lock, int64_t deadline_ns) {
    int64_t left = deadline_ns - clock_ns(CLOCK_REALTIME);
    os::TimedWaitConditionVariable(c, lock_mutex(lock), TimeSpan{left > 0 ? left : 0});
}

}  // extern "C"

// shared_ptr: el bloque de control lo instancia el juego (__shared_ptr_pointer<…>, __shared_ptr_emplace<…>) y sube y
// baja sus cuentas en línea con atómicos; aquí va la parte fuera de línea y la typeinfo de la base, con los mismos
// campos y el mismo orden de virtuales que libc++ (__release_weak llama __on_zero_shared_weak, slot 4).
namespace std {
namespace __1 {

class __shared_count {
  protected:
    long __shared_owners_;
    virtual ~__shared_count();

  private:
    virtual void __on_zero_shared() noexcept = 0;

  public:
    // Los D0 liberan objetos del juego.
    static void operator delete(void* p) noexcept { nnsdk::host().game_delete(p); }
};

class __shared_weak_count : private __shared_count {
    long __shared_weak_owners_;

  protected:
    ~__shared_weak_count() override;

  public:
    void __release_weak() noexcept;
    __shared_weak_count* lock() noexcept;
    virtual const void* __get_deleter(const type_info&) const noexcept;

  private:
    virtual void __on_zero_shared_weak() noexcept = 0;
};
static_assert(sizeof(__shared_weak_count) == 24);

__shared_count::~__shared_count() {}
__shared_weak_count::~__shared_weak_count() {}

// Como libc++: si no queda ningún weak_ptr (0) no hace falta el atómico, nadie más lo ve.
void __shared_weak_count::__release_weak() noexcept {
    if (__atomic_load_n(&__shared_weak_owners_, __ATOMIC_ACQUIRE) == 0 ||
        __atomic_add_fetch(&__shared_weak_owners_, -1, __ATOMIC_ACQ_REL) == -1)
        __on_zero_shared_weak();
}

// weak_ptr::lock(): sube la cuenta fuerte salvo que ya haya llegado a -1 (objeto destruido).
__shared_weak_count* __shared_weak_count::lock() noexcept {
    long owners = __atomic_load_n(&__shared_owners_, __ATOMIC_SEQ_CST);
    while (owners != -1)
        if (__atomic_compare_exchange_n(&__shared_owners_, &owners, owners + 1, true, __ATOMIC_SEQ_CST,
                                        __ATOMIC_SEQ_CST))
            return this;
    return nullptr;
}

// La base no tiene deleter; lo sobrescribe __shared_ptr_pointer en el juego.
const void* __shared_weak_count::__get_deleter(const type_info&) const noexcept { return nullptr; }

// bad_weak_ptr: lo lanza el juego en línea (shared_ptr(weak_ptr) caducado) con __cxa_throw y esta typeinfo/vtable;
// what() lo usa el terminate de la libc++abi.
class bad_weak_ptr : public exception {
  public:
    ~bad_weak_ptr() noexcept override;
    const char* what() const noexcept override;
    static void operator delete(void* p) noexcept { nnsdk::host().game_delete(p); }
};

bad_weak_ptr::~bad_weak_ptr() noexcept {}
const char* bad_weak_ptr::what() const noexcept { return "bad_weak_ptr"; }

}  // namespace __1
}  // namespace std
