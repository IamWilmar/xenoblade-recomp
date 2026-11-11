// nn::diag: aserciones y abortos del SDK. Se imprimen y se aborta, como en la consola. Aquí también los demás
// abortos del SDK (nn::detail, su API en C) y las utilidades en C de nn::Result.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "nnsdk/types.h"

namespace nn::diag {

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
