// nn::ec: la eShop (contenido descargable, consumibles). Sin internet no hay tienda: lo local (reservar la
// biblioteca, crear el accesor) funciona, y cada petición al servidor falla al momento con el error que da
// el SDK cuando nifm no tiene conexión, que el juego guarda y trata como "tienda no disponible"
// (main+0x79814ec). Los applets de la tienda se cierran sin comprar nada.

#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace nn {
struct ApplicationId {
    uint64_t value;
};
namespace account {
struct Uid;
struct UserHandle;
}  // namespace account
namespace err {
struct ErrorCode {
    uint32_t category;
    uint32_t number;
};
}  // namespace err
}  // namespace nn

namespace nn::ec {

struct NsUid {
    uint64_t value;
};
struct ConsumableId;
struct CourseId;
struct ShopService {
    enum Method {};
};

namespace {

// Módulo ec (164). 56: ninguna petición de internet aceptada (lo primero que mira Request).
// 200: el AsyncResponse no tiene ninguna petición detrás.
constexpr Result kResultInternetRequestNotAccepted{164 | (56 << 9)};
constexpr Result kResultNotRequested{164 | (200 << 9)};

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
        bool TryWait();
        Result GetSize(size_t* size);
        Result Get(void* buffer, size_t size);
        err::ErrorCode GetErrorCode() const;

    private:
        uint8_t storage_[0x50];
    };

    ShopServiceAccessor();
    ~ShopServiceAccessor();
    Result Initialize(const ShopService& service);
    Result Finalize();
    Result Request(AsyncResponse* response, const account::Uid& user, ShopService::Method method, const char* path,
                   const PostData& data, TimeSpan timeout);

private:
    uint8_t storage_[0x48];
};

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
bool ShopServiceAccessor::AsyncResponse::TryWait() { return true; }  // nada pendiente
Result ShopServiceAccessor::AsyncResponse::GetSize(size_t* size) {
    *size = 0;
    return kResultNotRequested;
}
Result ShopServiceAccessor::AsyncResponse::Get(void*, size_t) { return kResultNotRequested; }
err::ErrorCode ShopServiceAccessor::AsyncResponse::GetErrorCode() const { return {0, 0}; }  // GetInvalidErrorCode

// Los applets de la tienda no devuelven nada: el usuario la "cierra" al instante.
void ShowShopProductDetails(NsUid, const account::UserHandle&) {}
void ShowShopApplicationInformation(ApplicationId, const account::UserHandle&) {}
void ShowShopAddOnContentList(ApplicationId, const account::UserHandle&) {}
void ShowShopSubscriptionList(ApplicationId, const account::UserHandle&) {}
void ShowShopSubscriptionList(ApplicationId, const CourseId&, const account::UserHandle&) {}
void ShowShopConsumableItemList(ApplicationId, const account::UserHandle&) {}
void ShowShopConsumableItemDetail(ApplicationId, const ConsumableId&, NsUid, const account::UserHandle&) {}

}  // namespace nn::ec
