// setjmp / _setjmp / longjmp con el jmp_buf de la musl de Nintendo (aarch64): x19-x30, sp y d8-d15 en claro, 176
// bytes. Los de bionic no valen: su jmp_buf es más grande (pisaría el del juego) y guarda los registros cifrados con una
// cookie. El GC de IL2CPP (Boehm, GC_with_callee_saves_pushed) hace _setjmp para buscar punteros en los registros que
// conserva la llamada: con los de bionic no los veía y recogía objetos vivos (Disco Elysium caía al cargar los bundles).

    .text
    .p2align 2

    .global nngame_setjmp
    .type nngame_setjmp, %function
    .global nngame__setjmp
    .type nngame__setjmp, %function
nngame_setjmp:
nngame__setjmp:
    stp x19, x20, [x0, #0]
    stp x21, x22, [x0, #16]
    stp x23, x24, [x0, #32]
    stp x25, x26, [x0, #48]
    stp x27, x28, [x0, #64]
    stp x29, x30, [x0, #80]
    mov x2, sp
    str x2, [x0, #104]
    stp d8, d9, [x0, #112]
    stp d10, d11, [x0, #128]
    stp d12, d13, [x0, #144]
    stp d14, d15, [x0, #160]
    mov x0, #0
    ret
    .size nngame_setjmp, . - nngame_setjmp

    .global nngame_longjmp
    .type nngame_longjmp, %function
nngame_longjmp:
    ldp x19, x20, [x0, #0]
    ldp x21, x22, [x0, #16]
    ldp x23, x24, [x0, #32]
    ldp x25, x26, [x0, #48]
    ldp x27, x28, [x0, #64]
    ldp x29, x30, [x0, #80]
    ldr x2, [x0, #104]
    mov sp, x2
    ldp d8, d9, [x0, #112]
    ldp d10, d11, [x0, #128]
    ldp d12, d13, [x0, #144]
    ldp d14, d15, [x0, #160]
    cmp w1, #0
    csinc w0, w1, wzr, ne
    br x30
    .size nngame_longjmp, . - nngame_longjmp
