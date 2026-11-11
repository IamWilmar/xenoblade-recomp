// nn::nifm: la consola sin conexión, como en modo avión. Initialize va bien, la petición de red se resuelve en el
// acto como rechazada (nunca queda "en espera", que el juego sondea) y no hay IP ni proxy. El juego, al ver
// IsNetworkAvailable() a false, se queda en sus modos sin conexión; con SetLocalNetworkMode(true) (juego local por
// LDN) la respuesta es la misma: tampoco hay comunicación inalámbrica.

#include <atomic>
#include <cstdio>

#include "nnsdk/types.h"

struct in_addr;

namespace nn::socket {
struct InAddr;
}

namespace nn::nifm {

struct ProxySetting;

namespace {

// nifm (110) 300: "no hay conexión a Internet", lo que contesta el servicio a quien pide la IP o el proxy sin red.
constexpr Result kResultNoInternetConnection{110 | (300 << 9)};

}  // namespace

Result Initialize() { return ResultSuccess; }

void SubmitNetworkRequest() {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) printf("[nifm] petición de red: rechazada (sin conexión)\n");
}
void CancelNetworkRequest() {}
void SetLocalNetworkMode(bool) {}
bool IsNetworkRequestOnHold() { return false; }
bool IsNetworkAvailable() { return false; }
// En la consola enseña el applet de error de red; aquí no hay nada que el usuario pueda arreglar: false.
bool HandleNetworkRequestErrorResult() { return false; }

// Sin red no se tocan las salidas: el juego las inicializa a -1 y solo las lee si hay éxito.
Result GetCurrentPrimaryIpAddress(in_addr*) { return kResultNoInternetConnection; }
Result GetCurrentPrimaryIpAddress(socket::InAddr*) { return kResultNoInternetConnection; }
Result GetCurrentIpConfigInfo(in_addr*, in_addr*, in_addr*, in_addr*, in_addr*) { return kResultNoInternetConnection; }
Result GetCurrentProxySetting(ProxySetting*) { return kResultNoInternetConnection; }

}  // namespace nn::nifm
