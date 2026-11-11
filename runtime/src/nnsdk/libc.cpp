// Partes de la libc de Nintendo (musl) cuya ABI no coincide con la de bionic. Las de rutas están en
// libc_file.cpp y los pthreads en libc_thread.cpp.

#include <arpa/inet.h>
#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <cwctype>

#include "nnsdk/runtime.h"

namespace {
NnsdkHost g_host{};
}

namespace nnsdk {
const NnsdkHost& host() { return g_host; }
void on_thread_start() {
    if (g_host.thread_start) g_host.thread_start();
}
}  // namespace nnsdk

extern "C" void nnsdk_attach(const NnsdkHost* host) { g_host = *host; }

// El crt0 de cada módulo la llama desde DT_INIT con sus tablas; si no devuelve 1, el
// propio crt0 recorre el .init_array. Con 0 dejamos que lo haga él.
// En x2, x3, x6 y x7 van los límites de su TLS: .tdata [x2, x3) y .tbss [x6, x7) (lo que lee el código de MH Rise
// desde TP+16 cuadra con esto). La alineación no viene: se deduce de la dirección de .tdata, hasta 16 (con 8 o 16
// el bloque empieza en TP+16 igual).
extern "C" int __nnmusl_init_dso(uint64_t, uint64_t, uint64_t tdata, uint64_t tdata_end, uint64_t, uint64_t,
                                 uint64_t tbss, uint64_t tbss_end) {
    const bool sane = tdata && tdata <= tdata_end && tdata_end <= tbss && tbss <= tbss_end && tbss_end > tdata &&
                      tbss_end - tdata < (1u << 20);
    if (sane && g_host.tls_template) {
        const uint64_t align = std::min<uint64_t>(tdata & (~tdata + 1), 16);
        g_host.tls_template(reinterpret_cast<const void*>(tdata), tdata_end - tdata, tbss_end - tdata, align);
    } else if (tdata != tdata_end || tbss != tbss_end) {
        printf("[aviso] __nnmusl_init_dso: TLS raro (%#lx-%#lx, %#lx-%#lx): sin plantilla\n", tdata, tdata_end, tbss,
               tbss_end);
    }
    return 0;
}

// Al cerrar el proceso: los destructores de los thread_local del juego no corren aquí (el proceso muere entero).
extern "C" void __nnmusl_fini_dso() {}

extern "C" int* __errno_location() { return &errno; }

extern "C" [[noreturn]] void __assert_fail(const char* expr, const char* file, int line, const char* func) {
    fprintf(stdout, "assert del juego: %s (%s:%d, %s)\n", expr, file, line, func);
    fflush(stdout);
    abort();
}

// El juego libera con su propio free: hay que reservar con su malloc.
extern "C" char* nngame_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    auto* p = static_cast<char*>(g_host.game_malloc(n));
    return p ? static_cast<char*>(memcpy(p, s, n)) : nullptr;
}

// Valores de musl: FP_NAN 0, FP_INFINITE 1, FP_ZERO 2, FP_SUBNORMAL 3, FP_NORMAL 4. Bionic usa otros.
extern "C" int nngame___fpclassifyl(long double x) {
    switch (std::fpclassify(x)) {
        case FP_NAN: return 0;
        case FP_INFINITE: return 1;
        case FP_ZERO: return 2;
        case FP_SUBNORMAL: return 3;
        default: return 4;
    }
}

extern "C" int nngame___signbitl(long double x) { return std::signbit(x); }

namespace {

// La función de bionic que tapa una nuestra del mismo nombre (dlsym sobre libc.so, no sobre libnnsdk).
template <typename F>
F bionic(const char* name) {
    static void* libc = dlopen("libc.so", RTLD_NOW | RTLD_NOLOAD);
    return reinterpret_cast<F>(dlsym(libc, name));
}

// Con las cabeceras de Nintendo AF_INET6 vale 28 (el de nn::socket, BSD); en Linux, 10. AF_INET es 2 en los dos.
int host_family(int af) { return af == 28 ? AF_INET6 : af; }

}  // namespace

// La musl de Nintendo solo admite el reloj 0 (CLOCK_REALTIME); con otro da EINVAL, y quien lo llama (la mezcla
// de entropía de una librería de red, que también prueba CLOCK_BOOTTIME) sigue por otro lado. Allí cuenta desde
// el arranque; aquí es el de pared, como time() y gettimeofday() (bionic) y system_clock (cxx.cpp): los plazos
// de pthread_cond_timedwait se miden contra el mismo reloj.
extern "C" int nngame_clock_gettime(clockid_t clock, timespec* ts) {
    if (clock != CLOCK_REALTIME) return errno = EINVAL, -1;
    timespec_get(ts, TIME_UTC);
    return 0;
}

// C11 anexo K, como la de Nintendo: con n > smax llena smax bytes y da EOVERFLOW; errno queda siempre con el
// resultado, también 0.
extern "C" int memset_s(void* s, size_t smax, int c, size_t n) {
    if (!s) return errno = EINVAL;
    if (static_cast<int64_t>(smax) < 0) return errno = E2BIG;
    int err = static_cast<int64_t>(n) < 0 ? E2BIG : 0;
    size_t len = static_cast<int64_t>(n) < 0 ? smax : n;
    if (len > smax) len = smax, err = EOVERFLOW;
    memset(s, c, len);
    return errno = err;
}

// Las de musl: solo ASCII y su lista de espacios (bionic pregunta a ICU, que también cuenta U+00A0 o U+202F).
extern "C" int nngame_iswdigit(wint_t wc) { return wc - L'0' < 10u; }
extern "C" int nngame_iswspace(wint_t wc) {
    static const wchar_t kSpaces[] = {L' ',    L'\t',   L'\n',   L'\r',   11,      12,      0x0085, 0x2000,
                                      0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2008, 0x2009,
                                      0x200a, 0x2028, 0x2029, 0x205f, 0x3000, 0};
    return wc && wcschr(kSpaces, static_cast<wchar_t>(wc));
}

// La musl de Nintendo numera los sockets que crea ella como 0x0303xxxx (xxxx = descriptor de nn::socket) y aquí
// los traduce. El juego no crea ninguno así (no importa socket()): los de nn::socket ya son nativos y pasan.
extern "C" int _nnmusl_socket_TranslateNnmuslFileDescriptorOrPassThrough(int fd) {
    if (fd < 0) return errno = EBADF, -1;
    return fd;
}

extern "C" const char* nngame_inet_ntop(int af, const void* src, char* dst, socklen_t size) {
    static auto fn = bionic<decltype(&inet_ntop)>("inet_ntop");
    return fn(host_family(af), src, dst, size);
}
extern "C" int nngame_inet_pton(int af, const char* src, void* dst) {
    static auto fn = bionic<decltype(&inet_pton)>("inet_pton");
    return fn(host_family(af), src, dst);
}
