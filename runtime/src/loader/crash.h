#pragma once
#include <stdint.h>
// Si el juego se cae: registros, dirección traducida a main+offset, últimos stubs y backtrace.

void install_crash_handler();

// sigaltstack es por hilo: cada hilo del juego lo llama al arrancar.
void install_thread_altstack();

// DEPURACIÓN: protege la página de addr; el primer acceso (lectura o escritura) se registra y se deja pasar.
void watch_writes(uint64_t addr);

// Lo llama el manejador antes de salir tras un crash, si no es nulo (async-signal-safe): android_main lo usa para
// pasar al fichero lo que quede en la tubería del log.
extern void (*g_crash_log_drain)();
