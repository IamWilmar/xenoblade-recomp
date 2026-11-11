// nn::nfp: los amiibo (NFC del Joy-Con derecho). La Flip no tiene lector: la biblioteca se inicializa y no
// hay ningún dispositivo. ListDevices devuelve 0 con el error del servicio cuando no hay lector
// (NfcDeviceNotFound), que el juego trata igual que una lista vacía (main+0x73439f0) sin avisar de nada;
// cualquier operación sobre un DeviceHandle, que el juego nunca llega a tener, da ese mismo error.

#include <atomic>
#include <cstdint>

#include "nnsdk/types.h"

namespace nn::nfp {

struct DeviceHandle;
struct TagInfo;
struct ModelInfo;
struct CommonInfo;
struct RegisterInfo;
struct AmiiboSettingsStartParam;
struct ApplicationAreaCreateInfo;
enum ModelType {};
enum MountTarget {};

enum State { State_None = 0, State_Init = 1 };
enum DeviceState {
    DeviceState_Init = 0,
    DeviceState_Search = 1,
    DeviceState_Active = 2,
    DeviceState_Deactive = 3,
    DeviceState_Mount = 4,
    DeviceState_Unexpected = 5,
    DeviceState_Finalized = 6,
};

namespace {

// Módulo nfp (115). 64: no hay lector NFC (el que pone detail::CheckError). 208: el applet de ajustes
// de amiibo volvió sin terminar (el valor por defecto de detail::StartAmiiboSettings).
// Ojo: 80 (NFC desactivado) haría que el juego marque el lector como apagado (main+0x7343dac).
constexpr Result kResultNfcDeviceNotFound{115 | (64 << 9)};
constexpr Result kResultAmiiboSettingsNotCompleted{115 | (208 << 9)};

std::atomic<int> g_state{State_None};

}  // namespace

void Initialize() { g_state = State_Init; }
void Finalize() { g_state = State_None; }
State GetState() { return static_cast<State>(g_state.load()); }

Result ListDevices(DeviceHandle*, int* count, int) {
    *count = 0;
    return kResultNfcDeviceNotFound;
}

// La disponibilidad del NFC no cambia nunca: el evento se queda sin señalar.
void AttachAvailabilityChangeEvent(os::SystemEventType*) {}

// El SDK devuelve Unexpected sin inicializar; con un handle que no existe no hay otra respuesta sensata.
DeviceState GetDeviceState(const DeviceHandle&) { return DeviceState_Unexpected; }

Result GetNpadId(unsigned int*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result AttachActivateEvent(os::SystemEventType*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result AttachDeactivateEvent(os::SystemEventType*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result StartDetection(const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result StopDetection(const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result Mount(const DeviceHandle&, ModelType) { return kResultNfcDeviceNotFound; }
Result Mount(const DeviceHandle&, ModelType, MountTarget) { return kResultNfcDeviceNotFound; }
Result Flush(const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result Restore(const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result GetTagInfo(TagInfo*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result GetModelInfo(ModelInfo*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result GetCommonInfo(CommonInfo*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result GetRegisterInfo(RegisterInfo*, const DeviceHandle&) { return kResultNfcDeviceNotFound; }
Result OpenApplicationArea(const DeviceHandle&, unsigned int) { return kResultNfcDeviceNotFound; }
Result CreateApplicationArea(const DeviceHandle&, const ApplicationAreaCreateInfo&) { return kResultNfcDeviceNotFound; }
Result SetApplicationArea(const DeviceHandle&, const void*, size_t) { return kResultNfcDeviceNotFound; }
Result GetApplicationArea(void*, size_t* size, const DeviceHandle&, size_t) {
    *size = 0;
    return kResultNfcDeviceNotFound;
}

// Applets de ajustes de amiibo (registrar dueño, borrar datos, restaurar): sin applet, se cierran sin hacer
// nada, como si el usuario cancelara.
Result StartNicknameAndOwnerSettings(DeviceHandle*, bool*, RegisterInfo*, const AmiiboSettingsStartParam&,
                                     const TagInfo&) {
    return kResultAmiiboSettingsNotCompleted;
}
Result StartGameDataEraser(DeviceHandle*, const AmiiboSettingsStartParam&, const TagInfo&) {
    return kResultAmiiboSettingsNotCompleted;
}
Result StartRestorer(DeviceHandle*, const AmiiboSettingsStartParam&, const TagInfo&) {
    return kResultAmiiboSettingsNotCompleted;
}

}  // namespace nn::nfp
