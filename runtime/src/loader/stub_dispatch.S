// Destino común de los thunks de imports sin implementar. Entra con x17 = índice
// del stub y los argumentos del juego en x0..x7; devuelve lo que diga nier_on_stub.

    .text
    .global nier_stub_dispatch
    .type nier_stub_dispatch, %function
nier_stub_dispatch:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    stp x4, x5, [sp, #48]
    stp x6, x7, [sp, #64]
    str x8, [sp, #80]
    mov x0, x17             // índice del stub
    add x1, sp, #16         // x0..x7 del juego
    mov x2, x30             // desde dónde llamó
    bl nier_on_stub
    ldp x29, x30, [sp], #96
    ret
    .size nier_stub_dispatch, .-nier_stub_dispatch
