// nier_loader: el juego desde la terminal de la Flip (sin ventana).
// Uso: nier_loader <carpeta_exefs> [--stubs]   (NIER_WATCHDOG=<s> cambia el watchdog)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "loader/run.h"

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) {
        printf("uso: %s <carpeta_exefs> [--stubs]\n", argv[0]);
        return 2;
    }
    std::string self = argv[0];
    RunConfig cfg;
    cfg.exefs = argv[1];
    cfg.list_stubs = argc > 2 && strcmp(argv[2], "--stubs") == 0;
    cfg.nnsdk = (self.find('/') == std::string::npos ? "." : self.substr(0, self.find_last_of('/'))) + "/libnnsdk.so";
    if (const char* env = getenv("NIER_WATCHDOG")) cfg.watchdog_seconds = atoi(env);
    return nier_run(cfg);
}
