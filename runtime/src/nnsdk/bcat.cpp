// nn::bcat: la caché de entregas (BCAT), por donde Capcom manda las misiones de evento y las noticias.
// Responde como una consola sin conexión que nunca llegó a descargar nada: el almacenamiento se monta y
// está vacío (EnumerateDeliveryCacheDirectory da 0 carpetas: el juego se salta el bucle, main+0x6c17780),
// abrir cualquier archivo o carpeta da "no existe", y sincronizar falla antes de empezar con el error
// que da el SDK cuando nifm no tiene internet.
//
// Los objetos los reserva el juego. DeliveryCacheFile/Directory son en el SDK un puntero al objeto IPC
// (nulo = cerrado): aquí se quedan siempre en nulo y responden como cerrados. DeliveryCacheProgress
// (0x258 bytes) guarda en +0x58 el estado que lee GetStatus y detrás el resto del progreso: se mantiene
// ese layout para que el juego vea lo mismo por los accesores o leyendo el objeto.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::bcat {

struct DirectoryName {
    char value[32];
};
struct DeliveryCacheDirectoryEntry;

enum DeliveryCacheProgressStatus {
    DeliveryCacheProgressStatus_Queued = 0,
    DeliveryCacheProgressStatus_Done = 9,
};

namespace {

// Módulo bcat (122), con los mismos códigos que pone el SDK en el lado del cliente: 1 nombre no válido,
// 4 ya montado, 5 sin montar, 6 ya abierto, 7 sin abrir, 8 ninguna petición de internet aceptada.
// 2 es el "no existe" del servicio.
constexpr Result kResultInvalidArgument{122 | (1 << 9)};
constexpr Result kResultNotFound{122 | (2 << 9)};
constexpr Result kResultAlreadyMounted{122 | (4 << 9)};
constexpr Result kResultNotMounted{122 | (5 << 9)};
constexpr Result kResultAlreadyOpen{122 | (6 << 9)};
constexpr Result kResultNotOpen{122 | (7 << 9)};
constexpr Result kResultInternetRequestNotAccepted{122 | (8 << 9)};

std::atomic<bool> g_mounted{false};

}  // namespace

class DeliveryCacheFile {
public:
    DeliveryCacheFile();
    ~DeliveryCacheFile();
    Result Open(const char* path);
    Result Read(size_t* read, int64_t offset, void* buffer, size_t size);
    int64_t GetSize();
    void Close();

private:
    void* handle_;
};

class DeliveryCacheDirectory {
public:
    DeliveryCacheDirectory();
    ~DeliveryCacheDirectory();
    Result Open(const DirectoryName& name);
    Result Read(int* count, DeliveryCacheDirectoryEntry* entries, int max);

private:
    void* handle_;
};

class DeliveryCacheProgress {
public:
    DeliveryCacheProgress();
    ~DeliveryCacheProgress();
    void Update();
    DeliveryCacheProgressStatus GetStatus() const;
    Result GetResult() const;

private:
    friend Result RequestSyncDeliveryCache(DeliveryCacheProgress* progress);

    [[maybe_unused]] uint8_t mutex_[0x20];
    [[maybe_unused]] void* handle_;              // +0x20: el objeto IPC del progreso
    [[maybe_unused]] uint8_t event_[0x28];       // +0x28
    [[maybe_unused]] bool event_attached_;       // +0x50
    [[maybe_unused]] uint8_t pad_[7];
    int32_t status_;                             // +0x58: DeliveryCacheProgressImpl (0x200 bytes)
    uint32_t result_;                            // +0x5c
    [[maybe_unused]] uint8_t impl_rest_[0x1f8];  // carpeta y archivo actuales, bytes bajados…
};
static_assert(sizeof(DeliveryCacheProgress) == 0x258);

void Initialize() {}

Result MountDeliveryCacheStorage() {
    if (g_mounted.exchange(true)) return kResultAlreadyMounted;
    return ResultSuccess;
}

void UnmountDeliveryCacheStorage() { g_mounted = false; }

Result EnumerateDeliveryCacheDirectory(int* count, DirectoryName*, int) {
    if (!g_mounted) return kResultNotMounted;
    *count = 0;
    return ResultSuccess;
}

// El SDK mira nifm antes de pedir nada y devuelve 8 sin tocar el progreso. Aquí además el progreso
// queda terminado con ese error, por si el juego lo consultara igualmente: nunca se queda esperando.
Result RequestSyncDeliveryCache(DeliveryCacheProgress* progress) {
    progress->status_ = DeliveryCacheProgressStatus_Done;
    progress->result_ = kResultInternetRequestNotAccepted.value;
    return kResultInternetRequestNotAccepted;
}

DeliveryCacheProgress::DeliveryCacheProgress() { memset(this, 0, sizeof *this); }
DeliveryCacheProgress::~DeliveryCacheProgress() {}
void DeliveryCacheProgress::Update() {}  // no hay descarga que avance
DeliveryCacheProgressStatus DeliveryCacheProgress::GetStatus() const {
    return static_cast<DeliveryCacheProgressStatus>(status_);
}
Result DeliveryCacheProgress::GetResult() const { return {result_}; }

DeliveryCacheFile::DeliveryCacheFile() : handle_(nullptr) {}
DeliveryCacheFile::~DeliveryCacheFile() {}

// "carpeta/archivo", con la carpeta de 1 a 32 caracteres, como valida el SDK antes de llamar al servicio.
Result DeliveryCacheFile::Open(const char* path) {
    const char* slash = path ? strchr(path, '/') : nullptr;
    if (!slash || slash == path || slash - path > 32 || !slash[1]) return kResultInvalidArgument;
    if (handle_) return kResultAlreadyOpen;
    if (!g_mounted) return kResultNotMounted;
    return kResultNotFound;
}

Result DeliveryCacheFile::Read(size_t* read, int64_t, void*, size_t) {
    *read = 0;
    return kResultNotOpen;
}

// El SDK aborta si no está abierto; el juego solo la llama tras abrir con éxito (main+0x6c3eb1c).
int64_t DeliveryCacheFile::GetSize() { return 0; }

void DeliveryCacheFile::Close() { handle_ = nullptr; }

DeliveryCacheDirectory::DeliveryCacheDirectory() : handle_(nullptr) {}
DeliveryCacheDirectory::~DeliveryCacheDirectory() {}

Result DeliveryCacheDirectory::Open(const DirectoryName& name) {
    if (!name.value[0]) return kResultInvalidArgument;
    if (handle_) return kResultAlreadyOpen;
    if (!g_mounted) return kResultNotMounted;
    return kResultNotFound;
}

Result DeliveryCacheDirectory::Read(int* count, DeliveryCacheDirectoryEntry*, int) {
    *count = 0;
    return kResultNotOpen;
}

}  // namespace nn::bcat
