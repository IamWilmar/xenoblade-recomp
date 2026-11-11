// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Ficheros de registros de tamaño fijo con cabecera (el del precalentado de pipelines, <datos>/pipelines_estado.bin): se
// leen enteros al arrancar y después solo se añade al final, desde otro hilo y sin fsync. Un registro a medias (el proceso
// murió escribiendo) o roto acaba la lectura: se corta ahí para que lo que se añada después se pueda leer. Con otra
// cabecera (otro formato, otro traductor) se vacía. Sin dependencias del resto de la capa: lo prueba
// tools/tests/cache_files_test.cpp en el Mac.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvn::records_file {

struct Loaded {
    std::vector<uint8_t> records;  // los registros válidos, seguidos
    size_t file_bytes = 0;         // lo que tenía el fichero al abrirlo
    bool reset = false;            // la cabecera no valía (o no había fichero): se dejó solo la cabecera nueva
    bool cut = false;              // tenía un final roto: se cortó tras el último bueno
    bool ready = false;            // se puede añadir (el fichero existe con la cabecera buena)
};

constexpr size_t kMaxFile = 64u << 20;  // con más, se empieza de cero

// header: la cabecera exacta que tiene que tener. valid(const uint8_t* registro): si el registro está bien.
template <typename Valid>
Loaded load(const std::string& path, const void* header, size_t header_size, size_t record_size, Valid valid) {
    Loaded out;
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0) return out;
    std::vector<uint8_t> data;
    struct stat st {};
    if (fstat(fd, &st) == 0 && st.st_size > 0 && static_cast<uint64_t>(st.st_size) <= kMaxFile) {
        data.resize(static_cast<size_t>(st.st_size));
        size_t got = 0;
        while (got < data.size()) {
            const ssize_t r = pread(fd, data.data() + got, data.size() - got, static_cast<off_t>(got));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            got += static_cast<size_t>(r);
        }
        if (got != data.size()) data.clear();
    }
    out.file_bytes = st.st_size > 0 ? static_cast<size_t>(st.st_size) : 0;
    size_t used = data.size() >= header_size && memcmp(data.data(), header, header_size) == 0 ? header_size : 0;
    if (used) {
        while (used + record_size <= data.size() && valid(data.data() + used)) used += record_size;
        out.records.assign(data.begin() + static_cast<std::ptrdiff_t>(header_size),
                           data.begin() + static_cast<std::ptrdiff_t>(used));
        out.cut = used < out.file_bytes;
        out.ready = !out.cut || ftruncate(fd, static_cast<off_t>(used)) == 0;
    } else {
        out.reset = true;
        out.ready = ftruncate(fd, 0) == 0 &&
                    pwrite(fd, header, header_size, 0) == static_cast<ssize_t>(header_size);
    }
    close(fd);
    return out;
}

// Al final del fichero (que ya tiene que existir: load). false si no se pudo escribir todo.
inline bool append(const std::string& path, const void* data, size_t size) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
    if (fd < 0) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    bool ok = true;
    while (size && ok) {
        const ssize_t w = ::write(fd, p, size);
        if (w < 0 && errno == EINTR) continue;
        ok = w > 0;
        if (ok) p += w, size -= static_cast<size_t>(w);
    }
    close(fd);
    return ok;
}

}  // namespace nvn::records_file
