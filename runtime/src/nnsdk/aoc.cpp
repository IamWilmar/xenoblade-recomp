// nn::aoc: contenido descargable instalado. Cada carpeta <datos>/aoc/<índice> es uno: el índice del DLC es su id menos
// el de la aplicación con 0x1000 sumado (los de MH Rise van de 0x0100B04011743001 en adelante), y nn::fs::MountAddOnContent
// la monta como "aoc:" (fs.cpp). Si la pantalla de inicio encontró los DLC en el volcado del usuario, son los de
// <datos>/rutas.txt (rutas.h). Sin ninguno, el juego arranca como la base con el update (sin Sunbreak).

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nnsdk/rutas.h"

namespace {

std::vector<int> installed() {
    std::vector<int> out;
    const char* data = getenv("NIER_DATA");
    if (rutas::get().from_file) {
        for (const auto& [index, dir] : rutas::get().aoc) out.push_back(index);
    } else if (DIR* d = opendir((std::string(data ? data : ".") + "/aoc").c_str())) {
        while (const dirent* e = readdir(d)) {
            char* end = nullptr;
            const long n = strtol(e->d_name, &end, 10);
            if (end != e->d_name && *end == '\0' && n > 0 && n < 4096) out.push_back(static_cast<int>(n));
        }
        closedir(d);
    }
    std::sort(out.begin(), out.end());
    static bool logged = false;
    if (!logged) {
        logged = true;
        printf("[aoc] %zu DLC instalados en <datos>/aoc\n", out.size());
    }
    return out;
}

}  // namespace

namespace nn::aoc {

int CountAddOnContent() { return static_cast<int>(installed().size()); }

// Escribe hasta count índices a partir del offset-ésimo y devuelve cuántos escribió.
int ListAddOnContent(int* out, int offset, int count) {
    const std::vector<int> all = installed();
    int written = 0;
    for (int k = offset; out && k >= 0 && k < static_cast<int>(all.size()) && written < count; k++) out[written++] = all[k];
    return written;
}

}  // namespace nn::aoc
