// nn::ec: la eShop (contenido descargable, consumibles). Sin internet no hay tienda: lo local (reservar la
// biblioteca, crear el accesor, llevar la lista de derechos comprados) funciona, y cada petición al servidor falla al
// momento con el error que da el SDK cuando nifm no tiene conexión, que un juego trata como "tienda no disponible".
// Los applets de la tienda se cierran sin comprar nada.
//
// Disco Elysium importa las 63 funciones y solo las alcanza el plugin de Unity del SDK (NintendoSDK.dll: sus
// envoltorios nativos están en main+0x1559f80…0x155b410 y los métodos de C# nn.ec.* que los llaman no tienen ningún
// llamador en el juego). Los objetos los reserva ese plugin con el tamaño del SDK y aquí llevan su misma disposición.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nnsdk/types.h"

namespace nn {
struct ApplicationId {
    uint64_t value;
};
namespace account {
struct Uid {
    uint64_t data[2];
};
struct UserHandle;
}  // namespace account
namespace err {
struct ErrorCode {
    uint32_t category;
    uint32_t number;
};
}  // namespace err
namespace crypto::detail {
class Sha256Impl {  // crypto.cpp: 0x78 bytes de estado
public:
    ~Sha256Impl();
    void Initialize();
    void Update(const void* data, size_t size);
    void GetHash(void* out, size_t size);

private:
    alignas(8) uint8_t storage_[0x78];
};
}  // namespace crypto::detail
}  // namespace nn

namespace nn::ec {

struct NsUid {
    uint64_t value;
};
struct ConsumableId;
struct CourseId;
struct PurchasedItemInfo;
struct ShopService {
    enum Method {};
};

namespace {

// Módulo ec (164), los que devuelve el propio sdk:
constexpr Result kResultInternetRequestNotAccepted{164 | (56 << 9)};  // lo primero que mira Request (sdk+0x151498)
constexpr Result kResultInsufficientWorkMemory{164 | (90 << 9)};      // no cabe en la memoria de trabajo (sdk+0x148328)
constexpr Result kResultNotRequested{164 | (200 << 9)};               // el AsyncResponse no tiene petición detrás
constexpr Result kResultInvalidSerializedData{164 | (300 << 9)};      // datos exportados que no cuadran (sdk+0x148354)
constexpr Result kResultInsufficientItemIdBuffer{164 | (302 << 9)};   // más artículos que sitio (sdk+0x1488c8)
constexpr Result kResultItemNotProvidable{164 | (303 << 9)};          // ese artículo no está por entregar (sdk+0x148d20)

}  // namespace

class ShopServiceAccessor {
public:
    struct PostData;
    // El juego lo reserva (0x50 bytes en el SDK: nodo de lista, la petición en +0x18, su evento). Como
    // ninguna petición sale, responde siempre como uno sin usar.
    class AsyncResponse {
    public:
        AsyncResponse();
        ~AsyncResponse();
        void Wait();
        bool TryWait();
        void Cancel();
        Result GetSize(size_t* size);
        Result Get(void* buffer, size_t size);
        err::ErrorCode GetErrorCode() const;

    private:
        uint8_t storage_[0x50];
    };

    // Lo que espera Request si no se le dice otra cosa: un minuto (sdk+0xa5fc08). El plugin lo lee por su dirección
    // (main+0x4d5ce50).
    static const TimeSpan DefaultTimeout;

    ShopServiceAccessor();
    ~ShopServiceAccessor();
    Result Initialize(const ShopService& service);
    Result Finalize();
    Result Request(AsyncResponse* response, const account::Uid& user, ShopService::Method method, const char* path,
                   const PostData& data, TimeSpan timeout);

private:
    uint8_t storage_[0x48];
};

const TimeSpan ShopServiceAccessor::DefaultTimeout{60'000'000'000};

Result InitializeForShopServiceAccessors(void*, size_t) { return ResultSuccess; }
Result FinalizeForShopServiceAccessors() { return ResultSuccess; }

ShopServiceAccessor::ShopServiceAccessor() { memset(storage_, 0, sizeof storage_); }
ShopServiceAccessor::~ShopServiceAccessor() {}
Result ShopServiceAccessor::Initialize(const ShopService&) { return ResultSuccess; }
Result ShopServiceAccessor::Finalize() { return ResultSuccess; }

Result ShopServiceAccessor::Request(AsyncResponse*, const account::Uid&, ShopService::Method, const char*,
                                    const PostData&, TimeSpan) {
    return kResultInternetRequestNotAccepted;
}

ShopServiceAccessor::AsyncResponse::AsyncResponse() { memset(storage_, 0, sizeof storage_); }
ShopServiceAccessor::AsyncResponse::~AsyncResponse() {}
// Sin petición detrás no hay nada que esperar ni que cancelar: el sdk vuelve al momento (sdk+0x151af8, sdk+0x151db8).
void ShopServiceAccessor::AsyncResponse::Wait() {}
void ShopServiceAccessor::AsyncResponse::Cancel() {}
bool ShopServiceAccessor::AsyncResponse::TryWait() { return true; }  // nada pendiente
Result ShopServiceAccessor::AsyncResponse::GetSize(size_t* size) {
    *size = 0;
    return kResultNotRequested;
}
Result ShopServiceAccessor::AsyncResponse::Get(void*, size_t) { return kResultNotRequested; }
err::ErrorCode ShopServiceAccessor::AsyncResponse::GetErrorCode() const { return {0, 0}; }  // GetInvalidErrorCode

// --- Artículos consumibles con servicio ("consumable service items") ---
//
// Lo comprado en la tienda llega como "derechos" que el juego canjea: pregunta al servidor los que hay
// (AsyncGetConsumableRightDataRequest), los consume (AsyncConsumeRightDataRequest), entrega el artículo al jugador y lo
// marca (MarkServiceProvided). La lista vive en la memoria de trabajo que da el juego y se guarda en su partida
// (ExportSerializedRightData / Initialize con esos datos). Todo eso es local y funciona; las tres peticiones al
// servidor no salen.

// El id de un artículo: texto de hasta 16 caracteres con su 0 (los arrays del juego van de 17 en 17, sdk+0x1489dc).
struct ConsumableServiceItemId {
    char value[17];
    bool IsValid() const;
};
static_assert(sizeof(ConsumableServiceItemId) == 17);

bool ConsumableServiceItemId::IsValid() const { return value[0] != 0; }  // sdk+0x14f3a0

namespace {

// Un derecho, tal cual lo guarda y lo serializa el SDK (0x68 bytes). Lo que el sdk lee de él: el id del derecho (lo
// que compara al quitarlo, sdk+0x149c24), el del artículo (GetConsumableServiceItemId, sdk+0x14f804), si ya está
// consumido en el servidor y solo falta entregarlo (IsServiceProvidableOnLocal, sdk+0x14f81c) y el texto del que sale
// la semilla de MarkServiceProvided (un UUID que el SDK genera al recibir el derecho, sdk+0x14a210).
struct RightData {
    char right_id[0x25];
    ConsumableServiceItemId item_id;  // +0x25
    uint8_t reserved_36;
    bool providable_on_local;  // +0x37
    uint8_t reserved_38[8];
    char seed_text[0x25];  // +0x40
    uint8_t reserved_65[3];
};
static_assert(sizeof(RightData) == 0x68 && offsetof(RightData, item_id) == 0x25 &&
              offsetof(RightData, providable_on_local) == 0x37 && offsetof(RightData, seed_text) == 0x40);

constexpr int kMaxRights = 100;  // el sitio que reserva Initialize: 0x28a0 bytes (sdk+0x1482e8)

// Los datos exportados: versión 1, cuántos derechos y los derechos (sdk+0x148700).
struct SerializedHeader {
    int32_t version;
    int32_t count;
};

}  // namespace

class AsyncGetConsumableRightDataRequest;
class AsyncConsumeRightDataRequest;
class AsyncRecoverRightDataRequest;

// 0x80 bytes en el SDK (los reserva el plugin, main+0x155ac70): los campos van en su sitio; donde el SDK tiene su
// StandardAllocator sobre la memoria de trabajo (+0x18) aquí no hace falta nada, la lista va al principio de esa memoria.
class ConsumableServiceItemManager {
public:
    Result Initialize(ShopServiceAccessor* accessor, const account::Uid& user, void* work, size_t work_size);
    Result Initialize(ShopServiceAccessor* accessor, const account::Uid& user, void* work, size_t work_size,
                      const void* serialized, size_t serialized_size);
    bool IsInitialized();
    bool CanQueryNewData();
    bool IsRecoveryRequired();
    bool IsConsumptionRequired();
    void SetupAsyncRequest(AsyncGetConsumableRightDataRequest* request);
    void SetupAsyncRequest(AsyncConsumeRightDataRequest* request);
    void SetupAsyncRequest(AsyncRecoverRightDataRequest* request);
    size_t GetRequiredBufferSizeToExportSerializedRightData() const;
    size_t ExportSerializedRightData(void* buffer, size_t size) const;
    int GetProvidableItemIdCount() const;
    Result GetProvidableItemIds(int* count, ConsumableServiceItemId* ids, int max);
    Result MarkServiceProvided(const ConsumableServiceItemId& id);
    Result MarkServiceProvided(size_t* seed, const ConsumableServiceItemId& id);

    // 0 nada pendiente; 1 hay derechos a medio consumir y hay que preguntar al servidor cómo quedaron; 2 hay derechos
    // nuevos por consumir (sdk+0x148554…0x148594).
    enum State : int32_t { State_Idle = 0, State_RecoveryRequired = 1, State_ConsumptionRequired = 2 };

    bool initialized_;               // +0x00
    RightData* rights_;              // +0x08
    int32_t count_;                  // +0x10
    uint8_t allocator_[0x48];        // +0x18
    ShopServiceAccessor* accessor_;  // +0x60
    account::Uid user_;              // +0x68
    State state_;                    // +0x78
};
static_assert(sizeof(ConsumableServiceItemManager) == 0x80 && offsetof(ConsumableServiceItemManager, accessor_) == 0x60 &&
              offsetof(ConsumableServiceItemManager, state_) == 0x78);

// La parte común de las tres peticiones: lo que rellena SetupAsyncRequest (sdk+0x148598) y el AsyncResponse, que el
// plugin construye en su sitio (main+0x155ad70). Las clases del SDK siguen con lo suyo detrás (qué página pidieron,
// qué derechos mandaron a consumir, si Begin salió bien: 0x80 y 0x88 bytes en total), que solo se escribe cuando Begin
// consigue mandar la petición: aquí nunca.
class ConsumableServiceItemAsyncRequestBase {
public:
    void Wait();
    void Cancel();
    bool TryWait();
    err::ErrorCode GetErrorCode() const;

    ShopServiceAccessor* accessor_;             // +0x00
    ConsumableServiceItemManager* manager_;     // +0x08
    account::Uid user_;                         // +0x10
    ShopServiceAccessor::AsyncResponse response_;  // +0x20
    bool is_setup_;                             // +0x70
};
static_assert(offsetof(ConsumableServiceItemAsyncRequestBase, response_) == 0x20 &&
              offsetof(ConsumableServiceItemAsyncRequestBase, is_setup_) == 0x70);

class AsyncGetConsumableRightDataRequest : public ConsumableServiceItemAsyncRequestBase {
public:
    Result Begin();
    Result End();
};
class AsyncConsumeRightDataRequest : public ConsumableServiceItemAsyncRequestBase {
public:
    Result Begin();
    Result End();
};
// El plugin usa para esta los Begin y End de AsyncGetConsumableRightDataRequest (main+0x155ad10): main no importa los
// suyos.
class AsyncRecoverRightDataRequest : public ConsumableServiceItemAsyncRequestBase {};

namespace {

void setup(ConsumableServiceItemManager* manager, ConsumableServiceItemAsyncRequestBase* request) {
    request->accessor_ = manager->accessor_;
    request->manager_ = manager;
    request->user_ = manager->user_;
    request->is_setup_ = true;
}

bool same_item(const RightData& right, const ConsumableServiceItemId& id) {
    return right.providable_on_local && strncmp(right.item_id.value, id.value, sizeof id.value) == 0;
}

}  // namespace

// El constructor y el destructor con su nombre exacto (C1 y D1): el objeto es memoria del juego, sin vtable.
extern "C" void _ZN2nn2ec28ConsumableServiceItemManagerC1Ev(ConsumableServiceItemManager* self) {
    memset(static_cast<void*>(self), 0, sizeof *self);
}
extern "C" void _ZN2nn2ec28ConsumableServiceItemManagerD1Ev(ConsumableServiceItemManager* self) {
    self->initialized_ = false;  // la lista está en la memoria de trabajo, que libera el juego
}

// La lista de derechos (hasta 100) va en la memoria de trabajo; si no cabe, el mismo error que el allocator del SDK
// cuando no puede reservarla.
Result ConsumableServiceItemManager::Initialize(ShopServiceAccessor* accessor, const account::Uid& user, void* work,
                                                size_t work_size) {
    const uintptr_t start = (reinterpret_cast<uintptr_t>(work) + 7) & ~uintptr_t{7};
    const size_t skipped = start - reinterpret_cast<uintptr_t>(work);
    if (!work || work_size < skipped + kMaxRights * sizeof(RightData)) return kResultInsufficientWorkMemory;
    rights_ = reinterpret_cast<RightData*>(start);
    count_ = 0;
    accessor_ = accessor;
    user_ = user;
    initialized_ = true;
    return ResultSuccess;
}

// Con lo que el juego guardó en su partida. Como el sdk (sdk+0x14833c): los datos se validan antes de tocar nada, y
// si alguno de los derechos guardados no estaba ya listo para entregar, quedó a medio consumir: hay que recuperar.
Result ConsumableServiceItemManager::Initialize(ShopServiceAccessor* accessor, const account::Uid& user, void* work,
                                                size_t work_size, const void* serialized, size_t serialized_size) {
    SerializedHeader header;
    if (!serialized || serialized_size < sizeof header) return kResultInvalidSerializedData;
    memcpy(&header, serialized, sizeof header);
    // El sdk no mira que quepan en la lista; más de 100 no los ha podido exportar él.
    if (header.version != 1 || header.count < 0 || header.count > kMaxRights ||
        sizeof header + header.count * sizeof(RightData) > serialized_size)
        return kResultInvalidSerializedData;
    if (const Result r = Initialize(accessor, user, work, work_size); r.value) return r;
    memcpy(static_cast<void*>(rights_), static_cast<const uint8_t*>(serialized) + sizeof header,
           header.count * sizeof(RightData));
    count_ = header.count;
    for (int i = 0; i < count_; i++)
        if (!rights_[i].providable_on_local) state_ = State_RecoveryRequired;
    return ResultSuccess;
}

bool ConsumableServiceItemManager::IsInitialized() { return initialized_; }
// Se puede preguntar por derechos nuevos si queda sitio en la lista y no hay una recuperación pendiente.
bool ConsumableServiceItemManager::CanQueryNewData() { return count_ < kMaxRights && state_ != State_RecoveryRequired; }
bool ConsumableServiceItemManager::IsRecoveryRequired() { return state_ == State_RecoveryRequired; }
bool ConsumableServiceItemManager::IsConsumptionRequired() { return state_ == State_ConsumptionRequired; }

void ConsumableServiceItemManager::SetupAsyncRequest(AsyncGetConsumableRightDataRequest* request) { setup(this, request); }
void ConsumableServiceItemManager::SetupAsyncRequest(AsyncConsumeRightDataRequest* request) { setup(this, request); }
void ConsumableServiceItemManager::SetupAsyncRequest(AsyncRecoverRightDataRequest* request) { setup(this, request); }

size_t ConsumableServiceItemManager::GetRequiredBufferSizeToExportSerializedRightData() const {
    return sizeof(SerializedHeader) + static_cast<size_t>(count_) * sizeof(RightData);
}

// Devuelve lo escrito. El sdk no mira el tamaño del búfer (el juego pide antes el necesario); aquí, si no cabe, no se
// escribe nada.
size_t ConsumableServiceItemManager::ExportSerializedRightData(void* buffer, size_t size) const {
    const size_t needed = GetRequiredBufferSizeToExportSerializedRightData();
    if (!buffer || size < needed) return 0;
    const SerializedHeader header{1, count_};
    memcpy(buffer, &header, sizeof header);
    if (count_) memcpy(static_cast<uint8_t*>(buffer) + sizeof header, rights_, needed - sizeof header);
    return needed;
}

// Los artículos que ya se pueden entregar al jugador: los derechos consumidos en el servidor.
int ConsumableServiceItemManager::GetProvidableItemIdCount() const {
    int n = 0;
    for (int i = 0; i < count_; i++) n += rights_[i].providable_on_local;
    return n;
}

Result ConsumableServiceItemManager::GetProvidableItemIds(int* count, ConsumableServiceItemId* ids, int max) {
    const int providable = GetProvidableItemIdCount();
    if (providable > max) return kResultInsufficientItemIdBuffer;
    int n = 0;
    for (int i = 0; i < count_; i++)
        if (rights_[i].providable_on_local) ids[n++] = rights_[i].item_id;
    *count = n;
    return ResultSuccess;
}

// El juego ya le dio el artículo al jugador: el derecho sale de la lista. La semilla son los 8 primeros bytes del
// SHA-256 del texto que el SDK guardó con el derecho (sdk+0x148d30): un número distinto por compra, para que el juego
// decida con él lo que tenga de azar la entrega.
Result ConsumableServiceItemManager::MarkServiceProvided(size_t* seed, const ConsumableServiceItemId& id) {
    int found = -1;
    for (int i = 0; i < count_ && found < 0; i++)
        if (same_item(rights_[i], id)) found = i;
    if (found < 0) return kResultItemNotProvidable;
    uint8_t hash[32];
    crypto::detail::Sha256Impl sha;
    sha.Initialize();
    sha.Update(rights_[found].seed_text, sizeof rights_[found].seed_text);
    sha.GetHash(hash, sizeof hash);
    memcpy(seed, hash, sizeof *seed);
    memmove(static_cast<void*>(rights_ + found), rights_ + found + 1, (count_ - found - 1) * sizeof(RightData));
    count_--;
    return ResultSuccess;
}

Result ConsumableServiceItemManager::MarkServiceProvided(const ConsumableServiceItemId& id) {
    size_t seed = 0;
    return MarkServiceProvided(&seed, id);
}

void ConsumableServiceItemAsyncRequestBase::Wait() { response_.Wait(); }
void ConsumableServiceItemAsyncRequestBase::Cancel() { response_.Cancel(); }
bool ConsumableServiceItemAsyncRequestBase::TryWait() { return response_.TryWait(); }
err::ErrorCode ConsumableServiceItemAsyncRequestBase::GetErrorCode() const { return response_.GetErrorCode(); }

// Las dos piden por el accesor (GET /v1/rights/my?…&status=PURCHASED y un POST con los derechos a consumir, sdk+0x149f20
// y sdk+0x14b570), y Request lo primero que hace es mirar si hay internet: sin red, el mismo error que él.
Result AsyncGetConsumableRightDataRequest::Begin() { return kResultInternetRequestNotAccepted; }
Result AsyncConsumeRightDataRequest::Begin() { return kResultInternetRequestNotAccepted; }

// End lee la respuesta de su AsyncResponse, y uno que no llegó a mandar nada no tiene (sdk+0x14a2fc, sdk+0x14bd50).
Result AsyncGetConsumableRightDataRequest::End() {
    size_t size = 0;
    return response_.GetSize(&size);
}

// Si el final de un consumo falla no se sabe qué derechos consumió el servidor: el sdk deja al gestor en "hay que
// recuperar" (sdk+0x149bbc…0x149c00).
Result AsyncConsumeRightDataRequest::End() {
    size_t size = 0;
    const Result r = response_.GetSize(&size);
    if (r.value && is_setup_ && manager_) manager_->state_ = ConsumableServiceItemManager::State_RecoveryRequired;
    return r;
}

// --- Compras hechas en la tienda mientras el juego está abierto ---

// El sdk le pide al sistema el evento que avisa de cada compra (sdk+0x14f860). Sin tienda no se compra nada, y main no
// importa ni cómo obtener ese evento ni cómo esperarlo (GetPurchasedEvent): solo puede sondear con
// PopPurchasedItemInfo, que nunca tiene nada. No hay nada que preparar.
void InitializePurchasedEvent() {}
// false: no hay ninguna compra en la cola, y no se toca *info (el resultado 2164-0400 del servicio, sdk+0x14fe90).
bool PopPurchasedItemInfo(PurchasedItemInfo*) { return false; }

// El NsUid de un artículo de la tienda escrito en decimal; 0 (ninguno) si el texto no es solo un número (sdk+0x1481a0).
NsUid MakeNsUid(const char* text) {
    if (!text) return {0};
    char* end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    return {end && end != text && *end == '\0' ? value : 0};
}

// Los applets de la tienda no devuelven nada: el usuario la "cierra" al instante. Las formas sin UserHandle son las
// mismas con el usuario sin especificar (el sdk pasa un Uid nulo, sdk+0x150ab0…0x150c44).
namespace {
void shop(const char* what) {
    printf("[ec] %s: no hay tienda\n", what);
}
}  // namespace

void ShowShopProductDetails(NsUid, const account::UserHandle&) { shop("ficha de un producto"); }
void ShowShopProductDetails(NsUid) { shop("ficha de un producto"); }
void ShowShopProductList(const NsUid*, int, const char*, const account::UserHandle&) { shop("lista de productos"); }
void ShowShopProductList(const NsUid*, int, const char*) { shop("lista de productos"); }
void ShowShopEnterCodeScene(const account::UserHandle&) { shop("canjear un código"); }
void ShowShopEnterCodeScene() { shop("canjear un código"); }
void ShowShopApplicationInformation(ApplicationId, const account::UserHandle&) { shop("ficha de la aplicación"); }
void ShowShopApplicationInformation(ApplicationId) { shop("ficha de la aplicación"); }
void ShowShopAddOnContentList(ApplicationId, const account::UserHandle&) { shop("contenido descargable"); }
void ShowShopAddOnContentList(ApplicationId) { shop("contenido descargable"); }
void ShowShopSubscriptionList(ApplicationId, const account::UserHandle&) { shop("suscripciones"); }
void ShowShopSubscriptionList(ApplicationId) { shop("suscripciones"); }
void ShowShopSubscriptionList(ApplicationId, const CourseId&, const account::UserHandle&) { shop("una suscripción"); }
void ShowShopSubscriptionList(ApplicationId, const CourseId&) { shop("una suscripción"); }
void ShowShopConsumableItemList(ApplicationId, const account::UserHandle&) { shop("consumibles"); }
void ShowShopConsumableItemList(ApplicationId) { shop("consumibles"); }
void ShowShopConsumableItemDetail(ApplicationId, const ConsumableId&, NsUid, const account::UserHandle&) {
    shop("ficha de un consumible");
}
void ShowShopConsumableItemDetail(ApplicationId, const ConsumableId&, NsUid) { shop("ficha de un consumible"); }

}  // namespace nn::ec
