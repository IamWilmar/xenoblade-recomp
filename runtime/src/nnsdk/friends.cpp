// nn::friends: la lista de amigos de la cuenta de Nintendo. Sin cuenta de red (account.cpp) no hay amigos,
// ni bloqueados, ni perfiles que descargar: las listas salen vacías y con éxito, y lo que necesita
// internet falla con el error que da el SDK cuando nifm no tiene conexión. Lo que solo se anota
// (historial de juego, sesión en línea abierta o cerrada) se acepta y se descarta.
//
// Friend y Profile los reserva el juego (arrays de 0x200 y 0x100 bytes) y lee sus campos con los
// accesores de aquí: el layout es el del SDK, para que un objeto recién construido diga lo mismo
// (cuenta 0, apodo vacío, desconectado, no válido).

#include <cstdint>
#include <cstring>
#include <new>

#include "nnsdk/types.h"

namespace nn::account {
struct Uid;
struct Nickname;
struct NetworkServiceAccountId {
    uint64_t id;
};
}  // namespace nn::account

namespace nn::friends {

struct FriendFilter;
struct InAppScreenName;
struct Url;
struct PlayHistoryRegistrationKey {
    uint8_t data[0x40];  // el juego los guarda seguidos de 0x40 en 0x40 (main+0x2508a0c)
};
enum ImageSize {};
enum PresenceStatus { PresenceStatus_Offline = 0, PresenceStatus_Online = 1, PresenceStatus_OnlinePlay = 2 };

namespace {

// Módulo friends (121). 6: nifm no tiene ninguna petición de internet aceptada (detail::
// CheckInternetRequestAccepted). 7: el AsyncContext no tiene ninguna operación en curso.
constexpr Result kResultInternetRequestNotAccepted{121 | (6 << 9)};
constexpr Result kResultNotCalled{121 | (7 << 9)};

}  // namespace

class FriendPresence {
public:
    PresenceStatus GetStatus() const;
    bool IsSamePresenceGroupApplication() const;

private:
    [[maybe_unused]] uint8_t reserved_[0x18];
    int32_t status_;               // +0x18
    bool same_group_application_;  // +0x1c
};

class Friend {
public:
    Friend();
    account::NetworkServiceAccountId GetAccountId() const;
    const account::Nickname& GetNickname() const;
    const FriendPresence& GetPresence() const;

private:
    [[maybe_unused]] uint8_t reserved_[0x10];
    account::NetworkServiceAccountId account_id_;  // +0x10
    char nickname_[0x28];                          // +0x18 (Nickname: 33 bytes)
    FriendPresence presence_;                      // +0x40
    [[maybe_unused]] uint8_t rest_[0x1a0];         // IsValid es el byte +0x128
};
static_assert(sizeof(FriendPresence) == 0x20 && sizeof(Friend) == 0x200);

class Profile {
public:
    Profile();
    account::NetworkServiceAccountId GetAccountId() const;
    Result GetProfileImageUrl(Url* out, ImageSize size) const;
    bool IsValid() const;

private:
    account::NetworkServiceAccountId account_id_;  // +0x00
    [[maybe_unused]] uint8_t reserved_[0xc8];      // nickname, imagen de perfil (+0x30)…
    bool valid_;                                   // +0xd0
    [[maybe_unused]] uint8_t rest_[0x2f];
};
static_assert(sizeof(Profile) == 0x100);

// El SDK guarda un mutex (0x20) y la operación en curso (+0x20). Aquí ninguna operación llega a
// empezar (GetProfileList falla antes), así que el contexto responde como uno sin usar.
class AsyncContext {
public:
    AsyncContext();
    ~AsyncContext();
    Result HasDone(bool* done);
    Result GetResult() const;

private:
    uint8_t storage_[0x28];
};

PresenceStatus FriendPresence::GetStatus() const { return static_cast<PresenceStatus>(status_); }
bool FriendPresence::IsSamePresenceGroupApplication() const { return same_group_application_; }

Friend::Friend() { memset(this, 0, sizeof *this); }
account::NetworkServiceAccountId Friend::GetAccountId() const { return account_id_; }
const account::Nickname& Friend::GetNickname() const { return reinterpret_cast<const account::Nickname&>(nickname_); }
const FriendPresence& Friend::GetPresence() const { return presence_; }

Profile::Profile() { memset(this, 0, sizeof *this); }
account::NetworkServiceAccountId Profile::GetAccountId() const { return account_id_; }
bool Profile::IsValid() const { return valid_; }
// La URL la arma el servicio con datos que solo trae un perfil descargado.
Result Profile::GetProfileImageUrl(Url*, ImageSize) const { return kResultInternetRequestNotAccepted; }

AsyncContext::AsyncContext() { memset(storage_, 0, sizeof storage_); }
AsyncContext::~AsyncContext() {}
Result AsyncContext::HasDone(bool*) { return kResultNotCalled; }
Result AsyncContext::GetResult() const { return kResultNotCalled; }

Result GetFriendList(int* count, Friend*, int, int, const FriendFilter&) {
    *count = 0;
    return ResultSuccess;
}

Result GetFriendList(int* count, account::NetworkServiceAccountId*, int, int, const FriendFilter&) {
    *count = 0;
    return ResultSuccess;
}

Result GetBlockedUserList(int* count, account::NetworkServiceAccountId*, int, int) {
    *count = 0;
    return ResultSuccess;
}

// Ninguna de esas cuentas es amiga: cada entrada queda como recién construida (desconectado, no válida).
// El juego no mira el resultado y lee la presencia justo después (main+0x730f8b0).
Result UpdateFriendInfo(Friend* friends, const account::NetworkServiceAccountId*, int count) {
    for (int i = 0; i < count; i++) new (&friends[i]) Friend();
    return ResultSuccess;
}

// Pedirlos al servidor necesita internet. El juego, si falla, no espera el contexto (main+0x72a7514).
Result GetProfileList(AsyncContext*, Profile*, const account::NetworkServiceAccountId*, int) {
    return kResultInternetRequestNotAccepted;
}

// La clave se genera en la consola sin red; la del juego solo viaja dentro de una sesión multijugador,
// que aquí no se llega a formar. A ceros, y AddPlayHistory la acepta y la descarta.
Result GetPlayHistoryRegistrationKey(PlayHistoryRegistrationKey* key, const account::Uid&, bool) {
    memset(key, 0, sizeof *key);
    return ResultSuccess;
}

Result AddPlayHistory(const account::Uid&, const PlayHistoryRegistrationKey&, const InAppScreenName&,
                      const InAppScreenName&) {
    return ResultSuccess;
}

Result DeclareOpenOnlinePlaySession() { return ResultSuccess; }
Result DeclareCloseOnlinePlaySession() { return ResultSuccess; }

// Applet con la ficha de otro jugador: solo se llega desde una sesión en línea. Se cierra al momento.
Result ShowUserDetailInfo(const account::Uid&, const account::NetworkServiceAccountId&, const InAppScreenName&,
                          const InAppScreenName&) {
    return ResultSuccess;
}

}  // namespace nn::friends
