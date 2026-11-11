// Funciones de bionic, libm, compiler-rt y libc++abi con la misma ABI que la libc de
// Nintendo (musl, AArch64): el juego las llama directo. Lo que no coincide (errno,
// fpclassify, strdup con el malloc del juego, rutas "rom:/", pthreads sobre nn::os…) lo
// define libnnsdk.so, que tiene prioridad (libc*.cpp).
// Coinciden también struct stat, dirent, tm, lconv, ldiv_t y va_list; FILE* y DIR* son
// siempre los de bionic (fopen y opendir de libnnsdk los crean con bionic).

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <locale.h>
#include <math.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>

#include "loader/host_symbols.h"

char* strcasestr(const char*, const char*);

// compiler-rt: long double es de 128 bits en los dos lados.
void __addtf3(void), __subtf3(void), __multf3(void), __eqtf2(void), __netf2(void), __lttf2(void),
    __unordtf2(void), __extenddftf2(void), __trunctfdf2(void), __fixtfsi(void), __fixunstfsi(void),
    __floatditf(void), __floatsitf(void), __floatunsitf(void), __fixunsdfti(void), __udivti3(void);
// libc++abi / libunwind (ABI Itanium).
void _Unwind_Resume(void), __gxx_personality_v0(void), __cxa_begin_catch(void), __cxa_guard_abort(void),
    __cxa_guard_acquire(void), __cxa_guard_release(void), __cxa_pure_virtual(void), _ZSt9terminatev(void);
// Lanzar, capturar y comparar typeinfo tienen que ir por una sola copia de libc++abi: la del loader (libnnsdk
// no puede definirlas: su libc++ estática ya las trae y las esconde). Las excepciones estándar viven en std::,
// no en std::__1, así que valen las de la NDK tal cual.
void __cxa_allocate_exception(void), __cxa_free_exception(void), __cxa_throw(void), __cxa_rethrow(void),
    __cxa_end_catch(void), _Unwind_Backtrace(void), _Unwind_GetIP(void), _ZNKSt9exception4whatEv(void),
    _ZNSt9exceptionD2Ev(void), _ZNSt11logic_errorC2EPKc(void), _ZNSt11logic_errorC2ERKS_(void),
    _ZNSt12length_errorD1Ev(void), _ZNSt12out_of_rangeD1Ev(void), _ZNSt16invalid_argumentD1Ev(void),
    _ZNSt13runtime_errorC1EPKc(void), _ZNSt13runtime_errorD1Ev(void), _ZNKSt13runtime_error4whatEv(void);
extern char _ZTISt9exception[], _ZTISt12length_error[], _ZTISt12out_of_range[], _ZTISt16invalid_argument[],
    _ZTVSt12length_error[], _ZTVSt12out_of_range[], _ZTVSt16invalid_argument[], _ZSt7nothrow[];
extern char _ZTVN10__cxxabiv117__class_type_infoE[], _ZTVN10__cxxabiv120__si_class_type_infoE[],
    _ZTVN10__cxxabiv121__vmi_class_type_infoE[];

#define S(name) {#name, (void*)&name}

static const HostSymbol kSymbols[] = {
    S(__addtf3), S(__subtf3), S(__multf3), S(__eqtf2), S(__netf2), S(__lttf2), S(__unordtf2),
    S(__extenddftf2), S(__trunctfdf2), S(__fixtfsi), S(__fixunstfsi), S(__floatditf), S(__floatsitf),
    S(__floatunsitf), S(__fixunsdfti), S(__udivti3),

    S(_Unwind_Resume), S(__gxx_personality_v0), S(__cxa_begin_catch), S(__cxa_guard_abort),
    S(__cxa_guard_acquire), S(__cxa_guard_release), S(__cxa_pure_virtual), S(_ZSt9terminatev),
    S(_ZTVN10__cxxabiv117__class_type_infoE), S(_ZTVN10__cxxabiv120__si_class_type_infoE),
    S(_ZTVN10__cxxabiv121__vmi_class_type_infoE),
    S(__cxa_allocate_exception), S(__cxa_free_exception), S(__cxa_throw), S(__cxa_rethrow), S(__cxa_end_catch),
    S(_Unwind_Backtrace), S(_Unwind_GetIP), S(_ZNKSt9exception4whatEv), S(_ZNSt9exceptionD2Ev),
    S(_ZNSt11logic_errorC2EPKc), S(_ZNSt11logic_errorC2ERKS_), S(_ZNSt12length_errorD1Ev),
    S(_ZNSt12out_of_rangeD1Ev), S(_ZNSt16invalid_argumentD1Ev), S(_ZNSt13runtime_errorC1EPKc),
    S(_ZNSt13runtime_errorD1Ev), S(_ZNKSt13runtime_error4whatEv), S(_ZTISt9exception), S(_ZTISt12length_error),
    S(_ZTISt12out_of_range), S(_ZTISt16invalid_argument), S(_ZTVSt12length_error), S(_ZTVSt12out_of_range),
    S(_ZTVSt16invalid_argument), S(_ZSt7nothrow),

    S(abort), S(exit), S(getenv), S(atof), S(atoi), S(atol), S(strtod), S(strtof), S(strtol), S(strtoll),
    S(strtoul), S(strtoull), S(div), S(ldiv), S(qsort), S(bsearch), S(rand), S(setjmp), S(longjmp),

    // strerror_r: en C (sin _GNU_SOURCE) es la POSIX, que devuelve int como la de musl.
    S(memchr), S(memcmp), S(memcpy), S(memmove), S(memset), S(strcat), S(strchr), S(strcmp), S(strcpy),
    S(strcspn), S(strerror), S(strerror_r), S(strlen), S(strncat), S(strncmp), S(strncpy), S(strnlen),
    S(strrchr), S(strspn), S(strstr), S(strtok), S(strcasecmp), S(strncasecmp), S(wcslen), S(wmemcpy),
    // Sin ellas caían en un stub que devuelve 0 (tools/cobertura.py): strcasestr (7 sitios en main), wcscpy (4) y
    // wcsrtombs (40; bionic, como la musl de Nintendo, convierte con el locale "C").
    S(strcasestr), S(wcscpy), S(wcsrtombs),

    // towlower/towupper: bionic va a ICU fuera de ASCII; musl usa sus tablas de Unicode, mismo resultado.
    S(isalnum), S(isalpha), S(isprint), S(isspace), S(isupper), S(tolower), S(toupper), S(towlower),
    S(towupper), S(localeconv),

    // stdin/stdout/stderr son datos (FILE* de bionic): todo lo que reciba uno va a bionic.
    S(stdin), S(stdout), S(stderr), S(fclose), S(fread), S(fwrite), S(fseek), S(ftell), S(fseeko), S(ftello),
    S(rewind), S(feof),
    S(ferror), S(fflush), S(fgetc), S(fputc), S(fputs), S(putc), S(putchar), S(puts), S(printf), S(fprintf),
    S(vfprintf), S(sprintf), S(snprintf), S(vsprintf), S(vsnprintf), S(sscanf),

    // Descriptores y DIR* de bionic (open y opendir de libnnsdk traducen la ruta y abren con bionic).
    S(read), S(close), S(fstat), S(readdir), S(closedir), S(getpid), S(nanosleep), S(htons), S(ntohs),

    // Mismo int a 0 al empezar; bionic no serializa los inits como el cerrojo global de la musl de Nintendo.
    S(pthread_once),

    // time y gettimeofday: reloj de pared (en la Switch cuentan desde el arranque, ver clock_gettime en libc.cpp).
    S(time), S(gettimeofday), S(gmtime), S(gmtime_r), S(localtime), S(localtime_r), S(mktime), S(strftime),

    S(acos), S(acosf), S(acosh), S(asin), S(asinf), S(asinh), S(atan), S(atanf), S(atan2), S(atan2f),
    S(atanh), S(cbrt), S(cbrtf), S(cos), S(cosf), S(cosh), S(coshf), S(erf), S(erfc), S(exp), S(exp2),
    S(exp2f), S(expf), S(fmod), S(fmodf), S(frexp), S(hypot), S(hypotf), S(ldexp), S(ldexpf), S(log),
    S(log10), S(log10f), S(log2), S(log2f), S(logf), S(modf), S(modff), S(pow), S(powf), S(sin), S(sinf),
    S(sinh), S(sinhf), S(sqrt), S(sqrtf), S(tan), S(tanf), S(tanh), S(tanhf),
};

const HostSymbol* host_symbols(size_t* count) {
    *count = sizeof kSymbols / sizeof kSymbols[0];
    return kSymbols;
}
