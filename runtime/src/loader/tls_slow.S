// Camino lento de los stubs de TLS (tls.cpp): el hilo aún no tiene bloque. Se llega desde cualquier punto del
// código del juego, no en una llamada: todo puede estar vivo (x1..x18, los 128 bits de los q, los flags), así que
// se guarda todo menos x0, que devuelve el TP del juego. x16 y x30 los guarda el veneer que llama aquí.

    .text
    .global nier_tls_slow
    .type nier_tls_slow, %function
nier_tls_slow:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #688
    stp x1, x2, [sp, #0]
    stp x3, x4, [sp, #16]
    stp x5, x6, [sp, #32]
    stp x7, x8, [sp, #48]
    stp x9, x10, [sp, #64]
    stp x11, x12, [sp, #80]
    stp x13, x14, [sp, #96]
    stp x15, x17, [sp, #112]
    str x18, [sp, #128]
    mrs x1, nzcv
    mrs x2, fpsr
    mrs x3, fpcr
    stp x1, x2, [sp, #136]
    str x3, [sp, #152]
    stp q0, q1, [sp, #160]
    stp q2, q3, [sp, #192]
    stp q4, q5, [sp, #224]
    stp q6, q7, [sp, #256]
    stp q8, q9, [sp, #288]
    stp q10, q11, [sp, #320]
    stp q12, q13, [sp, #352]
    stp q14, q15, [sp, #384]
    stp q16, q17, [sp, #416]
    stp q18, q19, [sp, #448]
    stp q20, q21, [sp, #480]
    stp q22, q23, [sp, #512]
    stp q24, q25, [sp, #544]
    stp q26, q27, [sp, #576]
    stp q28, q29, [sp, #608]
    stp q30, q31, [sp, #640]
    bl nier_tls_ensure_thread
    ldp q30, q31, [sp, #640]
    ldp q28, q29, [sp, #608]
    ldp q26, q27, [sp, #576]
    ldp q24, q25, [sp, #544]
    ldp q22, q23, [sp, #512]
    ldp q20, q21, [sp, #480]
    ldp q18, q19, [sp, #448]
    ldp q16, q17, [sp, #416]
    ldp q14, q15, [sp, #384]
    ldp q12, q13, [sp, #352]
    ldp q10, q11, [sp, #320]
    ldp q8, q9, [sp, #288]
    ldp q6, q7, [sp, #256]
    ldp q4, q5, [sp, #224]
    ldp q2, q3, [sp, #192]
    ldp q0, q1, [sp, #160]
    ldp x1, x2, [sp, #136]
    ldr x3, [sp, #152]
    msr nzcv, x1
    msr fpsr, x2
    msr fpcr, x3
    ldr x18, [sp, #128]
    ldp x15, x17, [sp, #112]
    ldp x13, x14, [sp, #96]
    ldp x11, x12, [sp, #80]
    ldp x9, x10, [sp, #64]
    ldp x7, x8, [sp, #48]
    ldp x5, x6, [sp, #32]
    ldp x3, x4, [sp, #16]
    ldp x1, x2, [sp, #0]
    add sp, sp, #688
    ldp x29, x30, [sp], #16
    ret
    .size nier_tls_slow, .-nier_tls_slow
