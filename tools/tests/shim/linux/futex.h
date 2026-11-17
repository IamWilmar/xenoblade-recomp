// Solo para compilar runtime/src/nnsdk/hid.cpp en el Mac (tools/tests/sdk_puro_test.cpp): macOS no tiene futex, y
// llamar a syscall() con un número inventado mata el proceso. En la prueba no hay hilo del loader al que despertar,
// así que el aviso no hace nada; lo que se comprueba es el valor que hid deja en NnsdkInput::rumble.
#pragma once

#include <unistd.h>

#define FUTEX_WAKE_PRIVATE 0
#define SYS_futex 0
#define syscall(...) 0L
