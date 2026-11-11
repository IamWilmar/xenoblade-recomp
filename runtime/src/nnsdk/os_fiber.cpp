// Fibras de nn::os y GetCurrentStackInfo. MH Rise las usa como corrutinas: Resume guarda GetCurrentFiber() (nullptr
// si lo llama un hilo) y salta a la fibra; la fibra vuelve con SwitchToFiber(lo guardado). Una fibra puede seguir
// en otro hilo distinto del que la empezó, así que el contexto vive en la fibra y el estado por hilo (fibra actual,
// fibra que representa al hilo) en thread_local.
//
// Layout del FiberType del SDK 11.4 (InitializeFiber, SwitchToFiber), que se respeta porque es memoria del juego:
//   +0x20 u8 estado (0 sin inicializar, 1 lista, 2 corriendo, 4 terminada)   +0x21 u8 pila con alias
//   +0x28 función   +0x30 argumento   +0x38 pila original   +0x40 pila   +0x48 tamaño   +0x50 sp guardado
// Igual que FiberImplByHorizon, el contexto (registros que conserva la llamada) se guarda en la propia pila de la
// fibra y en +0x50 solo queda el sp. La pila no se reubica con guarda (el SDK lo hace si no se pide
// FiberFlag_NoStackGuard): se usa la del juego tal cual.

#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::os {
struct FiberType;
using FiberFunction = FiberType* (*)(void*);
}  // namespace nn::os

using nn::os::FiberFunction;
using nn::os::FiberType;

// Cambio de contexto: guarda x19-x30, d8-d15 en la pila actual y su sp en *save_sp; carga el sp de *load_sp y
// restaura lo que haya allí. En el primer salto a una fibra el marco lo prepara InitializeFiber: x0 = la fibra,
// x30 = fiber_entry, x29 = 0 (fin de la cadena de marcos para los backtraces). Símbolo local de este archivo.
extern "C" void fiber_switch_context(void** save_sp, void* const* load_sp);
asm(R"(
    .text
    .p2align 2
    .type fiber_switch_context, %function
fiber_switch_context:
    sub  sp, sp, #0xb0
    stp  xzr, x19, [sp, #0x00]
    stp  x20, x21, [sp, #0x10]
    stp  x22, x23, [sp, #0x20]
    stp  x24, x25, [sp, #0x30]
    stp  x26, x27, [sp, #0x40]
    stp  x28, x29, [sp, #0x50]
    stp  x30, xzr, [sp, #0x60]
    stp  d8, d9, [sp, #0x70]
    stp  d10, d11, [sp, #0x80]
    stp  d12, d13, [sp, #0x90]
    stp  d14, d15, [sp, #0xa0]
    mov  x2, sp
    str  x2, [x0]
    ldr  x2, [x1]
    mov  sp, x2
    ldp  x0, x19, [sp, #0x00]
    ldp  x20, x21, [sp, #0x10]
    ldp  x22, x23, [sp, #0x20]
    ldp  x24, x25, [sp, #0x30]
    ldp  x26, x27, [sp, #0x40]
    ldp  x28, x29, [sp, #0x50]
    ldr  x30, [sp, #0x60]
    ldp  d8, d9, [sp, #0x70]
    ldp  d10, d11, [sp, #0x80]
    ldp  d12, d13, [sp, #0x90]
    ldp  d14, d15, [sp, #0xa0]
    add  sp, sp, #0xb0
    ret
    .size fiber_switch_context, . - fiber_switch_context
)");

namespace {

constexpr size_t kFrameSize = 0xb0;
constexpr size_t kFiberSize = 0x58;  // hasta el sp guardado, inclusive
enum FiberState : uint8_t { NotInitialized = 0, Ready = 1, Running = 2, Completed = 4 };

template <typename T>
T& field(FiberType* f, size_t offset) {
    return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(f) + offset);
}
uint8_t& state_of(FiberType* f) { return field<uint8_t>(f, 0x20); }
void*& context_of(FiberType* f) { return field<void*>(f, 0x50); }

struct ThreadFibers {
    FiberType* current = nullptr;       // la que corre en este hilo (nullptr: el propio hilo)
    FiberType* thread_fiber = nullptr;  // la que representa al hilo mientras corre una fibra
};
thread_local ThreadFibers t_fibers;

// Siempre a través de esta función y después de cada cambio de contexto: la fibra puede volver en otro hilo, y el
// compilador daría por buena la dirección del thread_local que calculó antes del salto. Por lo mismo, las funciones
// que siguen tras un salto van sin protector de pila: el canario se lee del TLS de bionic (TPIDR_EL0+0x28) con el
// TP guardado al entrar, que podría ser el de un hilo que ya terminó.
[[gnu::noinline]] ThreadFibers& fibers() {
    asm volatile("" ::: "memory");
    return t_fibers;
}

[[noreturn]] void fatal(const char* what, FiberType* f) {
    printf("[nnsdk] fibras: %s (fibra %p, estado %d)\n", what, static_cast<void*>(f), f ? state_of(f) : -1);
    fflush(stdout);
    abort();
}

// Primera instrucción de cada fibra (con el marco de InitializeFiber). Si la función vuelve, devuelve la fibra a la
// que seguir; nullptr es volver al hilo que esté corriendo la fibra ahora (como CallFiberFunction y lo que le sigue
// en el SDK).
[[noreturn, clang::no_stack_protector]] void fiber_entry(FiberType* f) {
    FiberType* next = field<FiberFunction>(f, 0x28)(field<void*>(f, 0x30));
    ThreadFibers& tf = fibers();
    if (!next) next = tf.thread_fiber;
    if (!next || state_of(next) != Ready) fatal("la función de la fibra devolvió una fibra que no está lista", next);
    state_of(f) = Completed;
    state_of(next) = Running;
    tf.current = next;
    fiber_switch_context(&context_of(f), &context_of(next));
    fatal("se volvió a una fibra terminada", f);
}

}  // namespace

namespace nn::os {

void InitializeFiber(FiberType* f, FiberFunction function, void* argument, void* stack, size_t size, int) {
    field<FiberFunction>(f, 0x28) = function;
    field<void*>(f, 0x30) = argument;
    field<void*>(f, 0x38) = stack;
    field<void*>(f, 0x40) = stack;
    field<size_t>(f, 0x48) = size;
    field<uint8_t>(f, 0x21) = 0;
    const uintptr_t top = (reinterpret_cast<uintptr_t>(stack) + size) & ~uintptr_t{15};
    auto* frame = reinterpret_cast<uint64_t*>(top - kFrameSize);
    memset(frame, 0, kFrameSize);
    frame[0] = reinterpret_cast<uint64_t>(f);                    // x0
    frame[0x60 / 8] = reinterpret_cast<uint64_t>(&fiber_entry);  // x30
    context_of(f) = frame;
    state_of(f) = Ready;
}

void FinalizeFiber(FiberType* f) {
    field<uint8_t>(f, 0x21) = 0;
    state_of(f) = NotInitialized;
}

FiberType* GetCurrentFiber() { return fibers().current; }

[[clang::no_stack_protector]] void SwitchToFiber(FiberType* target) {
    ThreadFibers& tf = fibers();
    if (!target) target = tf.thread_fiber;  // nullptr: de vuelta al hilo
    if (!target || state_of(target) != Ready) fatal("SwitchToFiber a una fibra que no está lista", target);
    if (FiberType* current = tf.current) {
        state_of(current) = Ready;
        state_of(target) = Running;
        tf.current = target;
        fiber_switch_context(&context_of(current), &context_of(target));
        return;  // alguien volvió a esta fibra, quizá desde otro hilo
    }
    // El hilo deja de correr su propio código: se le da una fibra en su pila mientras tanto (como el SDK), a la que
    // se vuelve con SwitchToFiber(nullptr) o cuando una función de fibra devuelve nullptr. Solo se puede volver a
    // ella desde este mismo hilo.
    alignas(16) uint8_t self[kFiberSize] = {};
    auto* me = reinterpret_cast<FiberType*>(self);
    state_of(me) = Ready;  // como el SDK: el hilo queda "listo" mientras corre la fibra
    state_of(target) = Running;
    tf.thread_fiber = me;
    tf.current = target;
    fiber_switch_context(&context_of(me), &context_of(target));
    ThreadFibers& back = fibers();
    back.current = nullptr;
    back.thread_fiber = nullptr;
}

// La pila en la que se está: la de la fibra si corre una, si no la del hilo (la que reservó bionic para un hilo de
// nn::os, de su mismo tamaño). El juego la apunta al empezar cada worker.
void GetCurrentStackInfo(uintptr_t* out_stack, size_t* out_size) {
    uintptr_t stack = 0;
    size_t size = 0;
    if (FiberType* f = fibers().current) {
        stack = reinterpret_cast<uintptr_t>(field<void*>(f, 0x38));
        size = field<size_t>(f, 0x48);
    } else {
        thread_local uintptr_t t_stack = 0;
        thread_local size_t t_size = 0;
        if (!t_size) {
            pthread_attr_t attr;
            if (pthread_getattr_np(pthread_self(), &attr) == 0) {
                void* base = nullptr;
                pthread_attr_getstack(&attr, &base, &t_size);
                t_stack = reinterpret_cast<uintptr_t>(base);
                pthread_attr_destroy(&attr);
            }
        }
        stack = t_stack;
        size = t_size;
    }
    if (out_stack) *out_stack = stack;
    if (out_size) *out_size = size;
}

}  // namespace nn::os
