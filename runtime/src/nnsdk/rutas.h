#pragma once
// Dónde está cada parte del volcado del juego. La pantalla de inicio de la APK (SetupActivity) busca el volcado en la
// carpeta que elige el usuario y lo apunta en <datos>/rutas.txt ("exefs=", "romfs=", "aoc <índice>="); sin ese
// archivo, lo de siempre: <datos>/exefs, <datos>/romfs y <datos>/aoc/<índice>. Solo cabecera: la usan el loader y
// libnnsdk.so, cada uno con su copia (NIER_DATA tiene que estar puesta antes de la primera llamada).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace rutas {

struct Paths {
    std::string exefs, romfs;
    std::map<int, std::string> aoc;  // índice del DLC -> su RomFS; solo con rutas.txt
    bool from_file = false;
};

inline const Paths& get() {
    static const Paths paths = [] {
        Paths p;
        const char* env = getenv("NIER_DATA");
        const std::string data = env ? env : "/sdcard/NieR";
        p.exefs = data + "/exefs";
        p.romfs = data + "/romfs";
        FILE* f = fopen((data + "/rutas.txt").c_str(), "r");
        if (!f) return p;
        p.from_file = true;
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n")] = '\0';
            const char* eq = strchr(line, '=');
            if (line[0] == '#' || !eq) continue;
            const std::string key(line, eq - line), value(eq + 1);
            if (key == "exefs") {
                p.exefs = value;
            } else if (key == "romfs") {
                p.romfs = value;
            } else if (!key.compare(0, 4, "aoc ")) {
                const int index = atoi(key.c_str() + 4);
                if (index > 0 && index < 4096) p.aoc[index] = value;
            }
        }
        fclose(f);
        printf("[rutas] de rutas.txt: exefs %s, romfs %s, %zu DLC\n", p.exefs.c_str(), p.romfs.c_str(), p.aoc.size());
        return p;
    }();
    return paths;
}

}  // namespace rutas
