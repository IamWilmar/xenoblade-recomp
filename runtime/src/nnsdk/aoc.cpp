// nn::aoc: contenido descargable instalado. Cada carpeta <datos>/aoc/<índice> es uno: el índice del DLC es su id menos
// el de la aplicación con 0x1000 sumado (los de MH Rise van de 0x0100B04011743001 en adelante), y nn::fs::MountAddOnContent
// la monta como "aoc:" (fs.cpp). Si la pantalla de inicio encontró los DLC en el volcado del usuario, son los de
// <datos>/rutas.txt (rutas.h). Sin ninguno, el juego arranca como la base con el update (sin Sunbreak).

#include <dirent.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nnsdk/os.h"
#include "nnsdk/rutas.h"

namespace nn::os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}

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

// El evento que el sistema señala cuando cambia la lista de DLC (se instala uno con el juego abierto). El sdk le
// engancha el handle del servicio en modo AutoClear (sdk+0x33230) y deja el SystemEventType como inicializado: el byte
// de estado de +0x28 a 2 (sdk+0x3471c4), que es lo que mira el destructor del juego para destruirlo
// (nn::os::SystemEvent::~SystemEvent, main+0x944db4). Aquí las carpetas de <datos>/aoc solo se miran al preguntar, así
// que el evento existe y no se señala nunca: quien lo sondea (el plugin de Unity, main+0x155aa70) ve "sin cambios".
// Solo lo pide ese plugin (main+0x155aa60), que en Disco Elysium no tiene llamadores.
void GetAddOnContentListChangedEvent(os::SystemEvent* event) {
    constexpr size_t kStateOffset = 0x28;
    constexpr uint8_t kStateInterProcessEvent = 2;
    nnsdk::create_system_event(reinterpret_cast<os::SystemEventType*>(event), /*auto_clear=*/true, /*signaled=*/false);
    reinterpret_cast<uint8_t*>(event)[kStateOffset] = kStateInterProcessEvent;
}

}  // namespace nn::aoc
