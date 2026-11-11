// nn::ldn: comunicación local inalámbrica (varias consolas en la misma sala, sin internet). La Flip no
// tiene una radio que hable el protocolo de la Switch, así que la biblioteca se inicializa y la máquina
// de estados responde como una consola a la que no llega ninguna otra: buscar devuelve 0 redes, y crear
// una red o unirse a otra falla con el error del servicio, que el juego (main+0x24e891c) convierte en su
// aviso de "no se pudo conectar". El juego solo pasa por aquí si se elige el juego local.
//
// MakeIpv4Address lo llaman ~6000 constructores estáticos del juego (las direcciones constantes de cada
// unidad de compilación), antes de nnMain y de nnsdk_attach: no puede depender de nada.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "nnsdk/os.h"
#include "nnsdk/types.h"

namespace nn::ldn {

struct Ipv4Address {
    uint32_t raw;
};
struct SubnetMask {
    uint32_t raw;
};
struct NetworkInfo;
struct NetworkConfig;
struct ScanFilter;
struct SecurityConfig;
struct SecurityParameter;
struct UserConfig;
struct AddressEntry;

enum State {
    State_None = 0,
    State_Initialized = 1,
    State_AccessPoint = 2,
    State_AccessPointCreated = 3,
    State_Station = 4,
    State_StationConnected = 5,
    State_Error = 6,
};
enum DisconnectReason { DisconnectReason_None = 0 };
enum AcceptPolicy {};
enum ConnectOption {};

namespace {

// Módulo ldn (203). Sin radio no se llega a establecer ninguna red: el mismo error al crear que al unirse.
constexpr Result kResultAdvertiseDataTooLarge{203 | (10 << 9)};
constexpr Result kResultConnectionFailed{203 | (31 << 9)};
constexpr Result kResultBadState{203 | (32 << 9)};
constexpr size_t kAdvertiseDataSizeMax = 384;

std::atomic<int> g_state{State_None};
// El SystemEvent que el juego enganchó con AttachStateChangeEvent: el servicio lo señala en cada cambio
// de estado, y el juego puede estar esperándolo después de abrir o cerrar un modo.
std::atomic<os::SystemEventType*> g_state_event{nullptr};

void set_state(State s) {
    if (g_state.exchange(s) == s) return;
    if (os::SystemEventType* e = g_state_event.load()) nnsdk::signal_system_event(e);
}

bool initialized() { return g_state.load() != State_None; }

}  // namespace

Ipv4Address MakeIpv4Address(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
    return {uint32_t{a} << 24 | uint32_t{b} << 16 | uint32_t{c} << 8 | d};
}

Ipv4Address MakeBroadcastAddress(Ipv4Address address, SubnetMask mask) { return {address.raw | ~mask.raw}; }

Result Initialize() {
    if (!initialized()) set_state(State_Initialized);
    return ResultSuccess;
}

void Finalize() {
    g_state = State_None;
    g_state_event = nullptr;  // el juego lo destruye después
}

State GetState() { return static_cast<State>(g_state.load()); }

void AttachStateChangeEvent(os::SystemEventType* event) { g_state_event = event; }

// Abrir un modo vale desde cualquier estado inicializado (el servicio real pide cerrar antes el otro).
// Cerrar y desmontar responden éxito aunque no haya nada abierto: no hay nada que deshacer.
Result OpenAccessPoint() {
    if (!initialized()) return kResultBadState;
    set_state(State_AccessPoint);
    return ResultSuccess;
}

Result CloseAccessPoint() {
    int s = g_state.load();
    if (s == State_AccessPoint || s == State_AccessPointCreated) set_state(State_Initialized);
    return ResultSuccess;
}

Result OpenStation() {
    if (!initialized()) return kResultBadState;
    set_state(State_Station);
    return ResultSuccess;
}

Result CloseStation() {
    int s = g_state.load();
    if (s == State_Station || s == State_StationConnected) set_state(State_Initialized);
    return ResultSuccess;
}

// La red no llega a crearse: el estado sigue en AccessPoint, como tras un fallo real.
Result CreateNetwork(const NetworkConfig&, const SecurityConfig&, const UserConfig&) {
    return g_state.load() == State_AccessPoint ? kResultConnectionFailed : kResultBadState;
}

Result CreateNetworkPrivate(const NetworkConfig&, const SecurityConfig&, const SecurityParameter&, const UserConfig&,
                            int, const AddressEntry*) {
    return g_state.load() == State_AccessPoint ? kResultConnectionFailed : kResultBadState;
}

Result DestroyNetwork() {
    if (g_state.load() == State_AccessPointCreated) set_state(State_AccessPoint);
    return ResultSuccess;
}

Result SetAdvertiseData(const void*, size_t size) {
    return size > kAdvertiseDataSizeMax ? kResultAdvertiseDataTooLarge : ResultSuccess;
}

Result SetStationAcceptPolicy(AcceptPolicy) { return ResultSuccess; }

Result Reject(Ipv4Address) { return ResultSuccess; }  // no hay nadie conectado

// Ninguna consola cerca: 0 redes. El SDK solo escribe la cuenta si el servicio responde éxito.
Result Scan(NetworkInfo*, int* count, int, const ScanFilter&, int) {
    if (!initialized()) return kResultBadState;
    *count = 0;
    return ResultSuccess;
}

Result ScanPrivate(NetworkInfo*, int* count, int, const ScanFilter&, int) {
    if (!initialized()) return kResultBadState;
    *count = 0;
    return ResultSuccess;
}

// Unirse falla y el estado sigue en Station.
Result Connect(const NetworkInfo&, const SecurityConfig&, const UserConfig&, int, ConnectOption) {
    return g_state.load() == State_Station ? kResultConnectionFailed : kResultBadState;
}

Result ConnectPrivate(const NetworkConfig&, const SecurityConfig&, const SecurityParameter&, const UserConfig&, int,
                      ConnectOption) {
    return g_state.load() == State_Station ? kResultConnectionFailed : kResultBadState;
}

Result Disconnect() {
    if (g_state.load() == State_StationConnected) set_state(State_Station);
    return ResultSuccess;
}

DisconnectReason GetDisconnectReason() { return DisconnectReason_None; }

// Lo que solo existe dentro de una red (dirección, configuración, clave) no existe nunca: BadState, como
// el servicio fuera de AccessPointCreated / StationConnected. Las salidas quedan a 0 por si el juego no
// mira el resultado.
Result GetIpv4Address(Ipv4Address* address, SubnetMask* mask) {
    address->raw = 0;
    mask->raw = 0;
    return kResultBadState;
}

Result GetNetworkInfo(NetworkInfo*) { return kResultBadState; }
Result GetNetworkConfig(NetworkConfig*) { return kResultBadState; }
Result GetSecurityParameter(SecurityParameter*) { return kResultBadState; }

}  // namespace nn::ldn
