#pragma once
// Enlace de los imports del juego: libnnsdk.so > bionic/compiler-rt/libc++abi > datos del sdk > stub.

#include <cstdio>
#include <string>

#include "loader/host_symbols.h"
#include "loader/nso.h"

// Abre libnnsdk.so y prepara la memoria de los stubs.
bool init_imports(const std::string& nnsdk_path, std::string& err);

// Dirección para un import de main. sdk (opcional) aporta los datos que el juego importa.
uint64_t resolve_import(const ElfSym& sym, const Module* sdk);

// Stub con nombre creado en caliente (p. ej. funciones NVN pedidas por nombre); devuelve ret.
void* make_named_stub(const char* name, uint64_t ret);

void report_imports(bool list_stubs);
void* nnsdk_symbol(const char* name);
std::string demangle(const std::string& name);

// "main+0x1234", "stub:nn::os::…", "libc.so(memcpy+0x10)" o la dirección cruda.
std::string describe_addr(uint64_t addr);
void dump_recent_stubs(FILE* out);
