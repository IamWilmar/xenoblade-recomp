#pragma once
// DEPURACIÓN: sondas en el código del juego. <datos>/sondas tiene una dirección por línea ("main+0x6c50d60", y
// opcionalmente cuántas veces registrarla, 40 por defecto): al cargar, esa instrucción se cambia por un salto a un
// stub que registra x0..x7, x19..x24 y lr con el nombre del hilo, ejecuta la instrucción original y vuelve. Sirve
// para ver con qué argumentos llama el juego a sus propias funciones sin depurador.

#include <string>

#include "loader/nso.h"

namespace probe {

// Antes de tls::patch (el área de parches aún es RW; tls::patch la protege). Sin el fichero no hace nada.
bool install(Module& mod, std::string& err);

}  // namespace probe
