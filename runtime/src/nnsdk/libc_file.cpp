// Funciones de archivo de la libc de Nintendo que reciben una ruta. En la Switch van por nn::fs, así que entienden
// "montaje:/..." (rom:, save: y lo que monte el juego); aquí esas rutas se traducen a la carpeta del montaje y el
// resto ("/dev/urandom") se pasa tal cual a Android. El motor lee y guarda con nn::fs: esto lo usan librerías de
// terceros (la semilla de entropía abre /dev/urandom, abseil busca tzdata, gRPC sus certificados, NPLN su
// "_npln:/npln_config.json"), que sin el archivo siguen su camino.
// struct stat, dirent (d_name en +19) y los O_* son iguales en musl y bionic (AArch64): read, close, fstat,
// readdir y closedir van directos a bionic (host_symbols.c). Bionic se llama por otros nombres (openat, stat64,
// fopen64, fdopendir): con el que definimos aquí nos llamaríamos a nosotros mismos (open64 tampoco vale: con
// _FORTIFY_SOURCE el wrapper de bionic acaba en el símbolo open).

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/rutas.h"

namespace nnsdk {
// La traducción de fs.cpp con sus montajes reales ("" si el montaje no existe). Débil: si fs.cpp no la exporta,
// aquí solo se conoce rom:, que fs.cpp pone donde diga rutas.h.
[[gnu::weak]] std::string fs_host_path(const char* path);
}  // namespace nnsdk

namespace {

int g_logged = 0;

bool has_mount(const char* path) {
    const char* colon = strchr(path, ':');
    return colon && colon != path && !memchr(path, '/', colon - path);
}

// Ruta de Android para una de la Switch; vacía si nombra un montaje que no existe.
std::string host_path(const char* path) {
    if (!has_mount(path)) return path;
    std::string p;
    if (nnsdk::fs_host_path) {
        p = nnsdk::fs_host_path(path);
    } else if (!strncmp(path, "rom:", 4)) {
        const char* rest = path + 4;
        while (*rest == '/') rest++;
        p = rutas::get().romfs + "/" + rest;
    }
    if (p.empty())
        printf("[libc] montaje desconocido: %s\n", path);
    else if (g_logged++ < 20)
        printf("[libc] %s -> %s\n", path, p.c_str());
    return p;
}

}  // namespace

extern "C" {

int nngame_open(const char* path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    std::string p = host_path(path);
    if (p.empty()) return errno = ENOENT, -1;
    return openat(AT_FDCWD, p.c_str(), flags, mode);
}

int nngame_stat(const char* path, struct stat* st) {
    std::string p = host_path(path);
    if (p.empty()) return errno = ENOENT, -1;
    return stat64(p.c_str(), reinterpret_cast<struct stat64*>(st));
}

// El DIR* es el de bionic: readdir y closedir del juego son las suyas.
DIR* nngame_opendir(const char* path) {
    std::string p = host_path(path);
    if (p.empty()) return errno = ENOENT, nullptr;
    int fd = openat(AT_FDCWD, p.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return nullptr;
    DIR* d = fdopendir(fd);
    if (!d) {
        int err = errno;
        close(fd);
        errno = err;
    }
    return d;
}

// El FILE* es el de bionic, como stdout/stderr: fread, fputs, vfprintf… del juego van directos a bionic.
FILE* nngame_fopen(const char* path, const char* mode) {
    std::string p = host_path(path);
    if (p.empty()) return errno = ENOENT, nullptr;
    return fopen64(p.c_str(), mode);
}

}  // extern "C"
