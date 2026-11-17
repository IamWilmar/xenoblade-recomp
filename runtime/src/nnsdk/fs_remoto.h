#pragma once
// RomFS remota (solo para el emulador del Mac): rom: y aoc: se leen por TCP de tools/romfs_remota.py en vez de una
// carpeta del dispositivo. Así la RomFS (Xenoblade 3: ~20 GB con los DLC) vive una sola vez en el Mac y no hay que
// copiarla dentro del AVD. Se activa con rutas.txt: "romfs=tcp:10.0.2.2:7878/romfs" y "aoc 4=tcp:10.0.2.2:7878/aoc/4"
// (10.0.2.2 es el Mac visto desde el emulador). En la Flip nunca se usa: allí las rutas son carpetas de la SD.
//
// Protocolo (una conexión por hilo, TCP_NODELAY): petición = op (1 byte) | largo de la ruta (u16) | ruta | offset
// (i64) | tamaño (i64). Respuestas: 'S' (stat) -> tipo (i8: -1 no existe, 0 carpeta, 1 archivo) | tamaño (i64);
// 'L' (lista) -> n (i32) y n x {tipo (i8), tamaño (i64), largo (u16), nombre}; 'R' (lee) -> n (i64, -1 error) | datos.

#include <cstdint>
#include <string>
#include <vector>

namespace fs_remoto {

struct Entrada {
    std::string nombre;
    bool carpeta;
    int64_t tamano;
};

inline bool es_remota(const std::string& ruta) { return ruta.compare(0, 4, "tcp:") == 0; }

// -1 no existe (o no hay servidor), 0 carpeta, 1 archivo.
int stat(const std::string& ruta, int64_t* tamano);
bool lista(const std::string& ruta, std::vector<Entrada>& out);
// Bytes leídos, o -1.
int64_t lee(const std::string& ruta, int64_t offset, void* buffer, size_t tamano);
// Copia del archivo en <datos>/rom_local/ (para quien necesita un descriptor de verdad: los vídeos, la libc del
// juego). Vacía si no existe.
std::string copia_local(const std::string& ruta);

}  // namespace fs_remoto
