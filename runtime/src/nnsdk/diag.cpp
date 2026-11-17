// nn::diag: aserciones y abortos del SDK. Se imprimen y se aborta, como en la consola. Aquí también los demás
// abortos del SDK (nn::detail, su API en C), las utilidades en C de nn::Result y el backtrace del hilo.

#include <pthread.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "nnsdk/types.h"

namespace nn::diag {

namespace {

// Hasta dónde llega la pila del hilo (la de bionic: os.cpp no usa la memoria que el juego le da a CreateThread). Se
// pregunta una vez por hilo.
uintptr_t thread_stack_end(uintptr_t* begin) {
    struct Stack {
        uintptr_t begin = 0, end = 0;
    };
    static thread_local const Stack stack = [] {
        Stack s;
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) == 0) {
            void* base = nullptr;
            size_t size = 0;
            pthread_attr_getstack(&attr, &base, &size);
            pthread_attr_destroy(&attr);
            s.begin = reinterpret_cast<uintptr_t>(base);
            s.end = s.begin + size;
        }
        return s;
    }();
    *begin = stack.begin;
    return stack.end;
}

// Un marco {x29 anterior, dirección de retorno} de una pila que no es la del hilo (una fibra): se lee sin caerse si
// el puntero no vale.
bool read_frame(uintptr_t fp, uintptr_t frame[2]) {
    iovec local{frame, 2 * sizeof(uintptr_t)}, remote{reinterpret_cast<void*>(fp), 2 * sizeof(uintptr_t)};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(2 * sizeof(uintptr_t));
}

}  // namespace

// Las direcciones de retorno del hilo que llama, de la más reciente a la más antigua, siguiendo la cadena de marcos de
// x29 (el juego y libnnsdk los guardan en todas sus funciones): la primera es a dónde vuelve esta llamada. Es lo que
// hace el sdk (sdk+0x1463a0: un nn::diag::Backtrace que descarta sus propios marcos y luego un Step por dirección,
// sdk+0x147820), con las mismas paradas que su Step (sdk+0x1479c0): un x29 nulo, mal alineado, fuera de la pila o que
// no sube.
//
// No es solo diagnóstico: es la pila gestionada del il2cpp del juego. os::StackTrace::WalkStack (main+0x36439c) pide
// hasta 256 direcciones y pasa cada una, de la más antigua a la más reciente, a un callback (main+0x2e6e84) que busca
// el método de C# que la contiene (main+0x31961c, solo dentro de la sección de código gestionado) y lo apunta como
// marco. De ahí salen las trazas de las excepciones y de Debug.Log y todo lo que en C# mira quién llama. Con cero
// direcciones esa pila estaba siempre vacía.
size_t GetBacktrace(uintptr_t* out, int count) {
    if (!out || count <= 0) return 0;
    auto fp = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
    uintptr_t begin = 0;
    const uintptr_t end = thread_stack_end(&begin);
    const bool own_stack = fp >= begin && fp < end;
    size_t n = 0;
    while (n < static_cast<size_t>(count) && fp && !(fp & 7)) {
        uintptr_t frame[2];
        if (own_stack) {
            if (fp + sizeof frame > end) break;
            frame[0] = reinterpret_cast<const uintptr_t*>(fp)[0];
            frame[1] = reinterpret_cast<const uintptr_t*>(fp)[1];
        } else if (!read_frame(fp, frame)) {
            break;
        }
        if (!frame[1]) break;
        out[n++] = frame[1];
        if (frame[0] <= fp) break;
        fp = frame[0];
    }
    return n;
}

enum AssertionType {};

namespace detail {

bool NotNullImpl(uintptr_t value) { return value != 0; }

[[noreturn]] void AbortImpl(const char* condition, const char* function, const char* file, int line) {
    printf("[nn::diag] abort: %s en %s (%s:%d)\n", condition, function, file, line);
    fflush(stdout);
    abort();
}

[[noreturn]] void AbortImpl(const char* condition, const char* function, const char* file, int line,
                            const Result* result, const char* format, ...) {
    printf("[nn::diag] abort: %s en %s (%s:%d), result 0x%x: ", condition, function, file, line,
           result ? result->value : 0);
    va_list args;
    va_start(args, format);
    if (format) vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
    abort();
}

// Unity 2022 (Castlevania): las formas sin texto, con y sin Result.
[[noreturn]] void AbortImpl(const Result* result) {
    printf("[nn::diag] abort, result 0x%x\n", result ? result->value : 0);
    fflush(stdout);
    abort();
}
[[noreturn]] void AbortImpl() {
    printf("[nn::diag] abort\n");
    fflush(stdout);
    abort();
}

// La forma sin texto (Unreal Engine y CRI en Final Fantasy Resonance).
[[noreturn]] void OnAssertionFailure(AssertionType type) {
    printf("[nn::diag] aserción %d (sin texto), desde %p\n", type, __builtin_return_address(0));
    fflush(stdout);
    abort();
}

[[noreturn]] void OnAssertionFailure(AssertionType type, const char* condition, const char* function,
                                     const char* file, int line) {
    printf("[nn::diag] aserción %d: %s en %s (%s:%d)\n", type, condition, function, file, line);
    fflush(stdout);
    abort();
}

}  // namespace detail
}  // namespace nn::diag

namespace nn::detail {

// El default de un switch que no debía llegar (NN_UNEXPECTED_DEFAULT).
[[noreturn]] void UnexpectedDefaultImpl(const char* function, const char* file, int line) {
    printf("[nn::detail] default inesperado en %s (%s:%d)\n", function, file, line);
    fflush(stdout);
    abort();
}

}  // namespace nn::detail

extern "C" {

[[noreturn]] void nndetailDiagAbortNoMessageImpl() {
    printf("[nn::diag] abort (C, sin mensaje)\n");
    fflush(stdout);
    abort();
}

// Como AbortImpl con formato, desde C.
[[noreturn]] void nndetailDiagAbortFormatImpl(const char* condition, const char* function, const char* file, int line,
                                              const char* format, ...) {
    printf("[nn::diag] abort: %s en %s (%s:%d): ", condition, function, file, line);
    va_list args;
    va_start(args, format);
    if (format) vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
    abort();
}

// nn::Result en C: módulo en los 9 bits bajos, descripción en los 13 siguientes; 0 es éxito.
uint32_t nnResultGetModule(uint32_t result) { return result & 0x1FF; }
uint32_t nnResultGetDescription(uint32_t result) { return (result >> 9) & 0x1FFF; }
bool nnResultIsSuccess(uint32_t result) { return result == 0; }
bool nnResultIsFailure(uint32_t result) { return result != 0; }

}  // extern "C"
