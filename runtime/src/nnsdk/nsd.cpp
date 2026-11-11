// nn::nsd: los ajustes de servidores de Nintendo. El entorno es "lp1", el de las consolas de venta, que es un
// ajuste local (el juego lo mete con snprintf en nombres de servidor sin mirar si hubo error, así que tiene que
// quedar escrito). Resolve/ResolveEx no traducen nada: no hay ajustes de nsd, y quien pregunta (libcurl y la
// librería de red del juego) corta la conexión antes de llegar al DNS.

#include <cstring>

#include "nnsdk/types.h"

namespace nn::nsd {

struct EnvironmentIdentifier {
    char value[8];
};
struct Fqdn {
    char value[256];
};

namespace {

// nsd (141) 5: la que da el servicio cuando no hay ajustes guardados (EnableForcedGlobalServerAccess del SDK la
// trata así).
constexpr Result kResultNotFound{141 | (5 << 9)};

}  // namespace

void GetEnvironmentIdentifier(EnvironmentIdentifier* out) {
    memset(out, 0, sizeof *out);
    strcpy(out->value, "lp1");
}

Result Resolve(Fqdn* out, const Fqdn&) {
    out->value[0] = 0;
    return kResultNotFound;
}

Result ResolveEx(Fqdn* out, const Fqdn&) {
    out->value[0] = 0;
    return kResultNotFound;
}

}  // namespace nn::nsd
