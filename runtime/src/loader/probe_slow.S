// DEPURACIÓN: destino de las sondas (probe.cpp). Entra con x16 = número de sonda y, en la pila, lo que guardó el
// stub (x16, x17, x30 del juego). Se guarda todo menos eso (puede estar vivo cualquier registro y los flags), se
// llama a nier_on_probe(id, registros, bloque del stub) y se vuelve al stub, que ejecuta la instrucción original.

    .text
    .global nier_probe_slow
    .type nier_probe_slow, %function
nier_probe_slow:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #768
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    str x18, [sp, #128]
    stp x19, x20, [sp, #136]
    stp x21, x22, [sp, #152]
    stp x23, x24, [sp, #168]
    stp x25, x26, [sp, #184]
    stp x27, x28, [sp, #200]
    mrs x1, nzcv
    mrs x2, fpsr
    mrs x3, fpcr
    stp x1, x2, [sp, #216]
    str x3, [sp, #232]
    stp q0, q1, [sp, #240]
    stp q2, q3, [sp, #272]
    stp q4, q5, [sp, #304]
    stp q6, q7, [sp, #336]
    stp q8, q9, [sp, #368]
    stp q10, q11, [sp, #400]
    stp q12, q13, [sp, #432]
    stp q14, q15, [sp, #464]
    stp q16, q17, [sp, #496]
    stp q18, q19, [sp, #528]
    stp q20, q21, [sp, #560]
    stp q22, q23, [sp, #592]
    stp q24, q25, [sp, #624]
    stp q26, q27, [sp, #656]
    stp q28, q29, [sp, #688]
    stp q30, q31, [sp, #720]
    mov x0, x16             // número de sonda
    mov x1, sp              // x0..x15, x18, x19..x28
    add x2, x29, #16        // bloque del stub
    bl nier_on_probe
    ldp q30, q31, [sp, #720]
    ldp q28, q29, [sp, #688]
    ldp q26, q27, [sp, #656]
    ldp q24, q25, [sp, #624]
    ldp q22, q23, [sp, #592]
    ldp q20, q21, [sp, #560]
    ldp q18, q19, [sp, #528]
    ldp q16, q17, [sp, #496]
    ldp q14, q15, [sp, #464]
    ldp q12, q13, [sp, #432]
    ldp q10, q11, [sp, #400]
    ldp q8, q9, [sp, #368]
    ldp q6, q7, [sp, #336]
    ldp q4, q5, [sp, #304]
    ldp q2, q3, [sp, #272]
    ldp q0, q1, [sp, #240]
    ldp x1, x2, [sp, #216]
    ldr x3, [sp, #232]
    msr nzcv, x1
    msr fpsr, x2
    msr fpcr, x3
    ldr x18, [sp, #128]
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x2, x3, [sp, #16]
    ldp x0, x1, [sp, #0]
    add sp, sp, #768
    ldp x29, x30, [sp], #16
    ret
    .size nier_probe_slow, .-nier_probe_slow
