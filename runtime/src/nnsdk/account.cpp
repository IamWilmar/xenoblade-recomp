// nn::account: el usuario de la consola. NieR pide el usuario preseleccionado al arrancar (sin él se
// queda en "Loading system data..." reintentando) y con su Uid monta la partida guardada. Hay un único
// usuario local; las funciones de cuenta de red (Nintendo Account) responden que no hay, y el juego
// sigue sin las funciones en línea.
//
// El apodo del usuario sale de <datos>/nombre.txt (la primera línea), o "Hunter".

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/os.h"
#include "nnsdk/types.h"

namespace nn::os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}

namespace nn::account {

struct Uid {
    uint64_t data[2];
};
struct UserHandle {
    uint64_t data[3];
};
struct Nickname {
    char name[33];
};
struct NetworkServiceAccountId {
    uint64_t id;
};
struct UserSelectionSettings;

namespace {

constexpr Uid kUser{{0x0123456789abcdefULL, 0xfedcba9876543210ULL}};
// Resultado de "no hay cuenta de red" (módulo account, 124): cualquier error sirve, el juego
// solo distingue éxito de fallo.
constexpr Result kResultNoNetworkAccount{124 | (280 << 9)};
// El usuario cerró el applet sin elegir (2124-0001): lo que da el sdk cuando el applet no devuelve
// resultado (StartPselApplet, sdk+0x1af4).
constexpr Result kResultCancelledByUser{124 | (1 << 9)};

bool same(const Uid& a, const Uid& b) { return a.data[0] == b.data[0] && a.data[1] == b.data[1]; }

std::string nickname() {
    const char* env = getenv("NIER_DATA");
    std::string path = std::string(env ? env : "/sdcard/NieR") + "/nombre.txt";
    std::string name;
    if (FILE* f = fopen(path.c_str(), "r")) {
        char line[64] = {};
        if (fgets(line, sizeof line, f)) name = line;
        fclose(f);
    }
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r' || name.back() == ' ')) name.pop_back();
    if (name.empty()) return "Hunter";
    // Como mucho 32 bytes (Nickname), sin partir un carácter UTF-8.
    size_t n = std::min<size_t>(name.size(), 32);
    while (n < name.size() && n > 0 && (static_cast<unsigned char>(name[n]) & 0xC0) == 0x80) n--;
    return name.substr(0, n);
}

}  // namespace

class AsyncContext {
public:
    AsyncContext();
    ~AsyncContext();
    Result HasDone(bool* done);
    Result GetResult();
    Result Cancel();
    Result GetSystemEvent(os::SystemEvent* event);

private:
    uint8_t storage_[0x20];  // el juego reserva el objeto; no guardamos estado
};

void Initialize() {}

bool TryOpenPreselectedUser(UserHandle* out) {
    out->data[0] = kUser.data[0];
    out->data[1] = kUser.data[1];
    out->data[2] = 1;
    return true;
}

Result OpenUser(UserHandle* out, const Uid& uid) {
    if (!same(uid, kUser)) printf("[account] OpenUser de un Uid desconocido: se abre el único usuario\n");
    return TryOpenPreselectedUser(out) ? ResultSuccess : kResultCancelledByUser;
}
void CloseUser(const UserHandle&) {}

Result GetUserId(Uid* out, const UserHandle&) {
    *out = kUser;
    return ResultSuccess;
}

Result GetUserCount(int* out) {
    *out = 1;
    return ResultSuccess;
}
Result ListAllUsers(int* count, Uid* out, int max) {
    *count = max > 0 ? 1 : 0;
    if (max > 0) out[0] = kUser;
    return ResultSuccess;
}
Result GetUserExistence(bool* out, const Uid& uid) {
    *out = same(uid, kUser);
    return ResultSuccess;
}

// El selector de usuario con un único usuario: lo elige directamente.
Result ShowUserSelector(Uid* out, const UserSelectionSettings&) {
    *out = kUser;
    return ResultSuccess;
}

Result GetNickname(Nickname* out, const Uid&) {
    static const std::string name = nickname();
    memset(out, 0, sizeof *out);
    strncpy(out->name, name.c_str(), sizeof out->name - 1);
    return ResultSuccess;
}

Result IsNetworkServiceAccountAvailable(bool* out, const UserHandle&) {
    *out = false;
    return ResultSuccess;
}
// El applet que explica que hace falta Nintendo Switch Online: el usuario lo cierra.
Result ShowLicenseRequirementsForNetworkService(const UserHandle&) { return kResultCancelledByUser; }

Result GetNetworkServiceAccountId(NetworkServiceAccountId* out, const UserHandle&) {
    out->id = 0;
    return kResultNoNetworkAccount;
}
Result EnsureNetworkServiceAccountAvailable(const UserHandle&) { return kResultNoNetworkAccount; }
Result EnsureNetworkServiceAccountIdTokenCacheAsync(AsyncContext*, const UserHandle&) { return kResultNoNetworkAccount; }
Result LoadNetworkServiceAccountIdTokenCache(unsigned long* size, char*, unsigned long, const UserHandle&) {
    *size = 0;
    return kResultNoNetworkAccount;
}

AsyncContext::AsyncContext() { memset(storage_, 0, sizeof storage_); }
AsyncContext::~AsyncContext() {}
Result AsyncContext::HasDone(bool* done) {
    *done = true;
    return ResultSuccess;
}
Result AsyncContext::GetResult() { return kResultNoNetworkAccount; }
Result AsyncContext::Cancel() { return ResultSuccess; }
Result AsyncContext::GetSystemEvent(os::SystemEvent* event) {
    nnsdk::signal_system_event(reinterpret_cast<os::SystemEventType*>(event));  // ya terminó
    return ResultSuccess;
}

}  // namespace nn::account
