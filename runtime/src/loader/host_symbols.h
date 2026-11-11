#pragma once
// Tabla de símbolos del sistema que se enlazan directo al juego (la usa host_symbols.c, en C).

#include <stddef.h>

typedef struct HostSymbol {
    const char* name;
    void* addr;
} HostSymbol;

#ifdef __cplusplus
extern "C"
#endif
    const HostSymbol*
    host_symbols(size_t* count);
