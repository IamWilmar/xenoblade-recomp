// nn::account: el usuario de la consola. NieR pide el usuario preseleccionado al arrancar (sin él se
// queda en "Loading system data..." reintentando) y con su Uid monta la partida guardada. Hay un único
// usuario local; las funciones de cuenta de red (Nintendo Account) responden que no hay, y el juego
// sigue sin las funciones en línea.
//
// El apodo del usuario sale de <datos>/nombre.txt (la primera línea), o "Hunter". Su icono, de <datos>/perfil.jpg si
// existe (un JPEG de 256x256, hasta 128 KB como en la consola); si no, uno gris liso. La miniatura de la partida que
// la consola enseña en su gestor de datos se queda en <datos>/miniatura_partida.rgba (256x144 RGBA8).
//
// Disco Elysium solo llega a nn::account por el plugin de Unity del SDK (NintendoSDK.dll, main+0x155a640…0x155aa30).

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

// Los del servicio de cuentas para un Uid que no es de nadie (2124-0020) y para un búfer de otro tamaño que el que
// pide (2124-0032). El sdk del juego solo los reenvía (sdk+0x24f8, sdk+0x2dd8): los valores son los conocidos del
// servicio, que no viene en el volcado.
constexpr Result kResultInvalidUserId{124 | (20 << 9)};
constexpr Result kResultInvalidArrayLength{124 | (32 << 9)};

// La miniatura de la partida: 256x144 RGBA8 (lo que copia el plugin antes de llamar, main+0x155a810).
constexpr size_t kThumbnailBytes = 256 * 144 * 4;
// Lo que puede ocupar el icono de un usuario (nn::account::ProfileImageBytesMax).
constexpr size_t kProfileImageBytesMax = 0x20000;

// Handles abiertos del único usuario (OpenUser / TryOpenPreselectedUser menos CloseUser): lo que mira ListOpenUsers.
std::atomic<int> g_open_handles{0};

bool same(const Uid& a, const Uid& b) { return a.data[0] == b.data[0] && a.data[1] == b.data[1]; }

std::string data_dir() {
    const char* env = getenv("NIER_DATA");
    return env ? env : "/sdcard/NieR";
}

// Las dos listas de usuarios del sdk (ListOpenUsers sdk+0x1f50, ListQualifiedUsers sdk+0x26c0) hacen lo mismo con lo
// que les da el servicio: *count es cuántos hay (no cuántos caben) y el array se rellena entero, con el Uid nulo en lo
// que sobra.
Result list_users(int* count, Uid* out, int max, int present) {
    for (int i = 0; out && i < max; i++) out[i] = i < present ? kUser : Uid{};
    if (count) *count = present;
    return ResultSuccess;
}

// El icono que la consola pone a todo usuario: aquí, un JPEG de 256x256 en gris liso (416 bytes). La cabecera es la de
// Pillow con tablas optimizadas (JFIF, DQT, SOF0 de un componente, una DHT de DC con las categorías 0 y 5 y otra de AC
// con solo el fin de bloque, SOS); los datos son el primer bloque (0x9e: DC del gris 96 y fin de bloque) y dos bits a
// cero por cada uno de los otros 1023 bloques ("igual que el anterior" y fin de bloque).
constexpr uint8_t kAvatarHeader[] = {
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x10, 0x0b, 0x0c, 0x0e, 0x0c, 0x0a, 0x10, 0x0e, 0x0d, 0x0e, 0x12, 0x11, 0x10, 0x13, 0x18, 0x28, 0x1a, 0x18, 0x16, 0x16, 0x18, 0x31, 0x23,
    0x25, 0x1d, 0x28, 0x3a, 0x33, 0x3d, 0x3c, 0x39, 0x33, 0x38, 0x37, 0x40, 0x48, 0x5c, 0x4e, 0x40, 0x44, 0x57, 0x45, 0x37, 0x38, 0x50, 0x6d, 0x51,
    0x57, 0x5f, 0x62, 0x67, 0x68, 0x67, 0x3e, 0x4d, 0x71, 0x79, 0x70, 0x64, 0x78, 0x5c, 0x65, 0x67, 0x63, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x01, 0x00,
    0x01, 0x00, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x05, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00,
};
static_assert(sizeof kAvatarHeader == 157);

std::vector<uint8_t> default_avatar() {
    std::vector<uint8_t> jpeg(kAvatarHeader, kAvatarHeader + sizeof kAvatarHeader);
    jpeg.push_back(0x9e);
    jpeg.insert(jpeg.end(), 255, 0x00);
    jpeg.insert(jpeg.end(), {0x03, 0xff, 0xd9});  // los dos bits de relleno y el fin de imagen
    return jpeg;
}

// <datos>/perfil.jpg si es un JPEG que cabe en lo que admite la consola; si no, el de por defecto.
const std::vector<uint8_t>& profile_image() {
    static const std::vector<uint8_t> image = [] {
        std::vector<uint8_t> file;
        if (FILE* f = fopen((data_dir() + "/perfil.jpg").c_str(), "rb")) {
            file.resize(kProfileImageBytesMax + 1);
            file.resize(fread(file.data(), 1, file.size(), f));
            fclose(f);
            const bool jpeg = file.size() > 3 && file[0] == 0xff && file[1] == 0xd8;
            if (jpeg && file.size() <= kProfileImageBytesMax) return file;
            printf("[account] <datos>/perfil.jpg no vale (%s): se usa el icono por defecto\n",
                   jpeg ? "pasa de 128 KB" : "no es un JPEG");
        }
        return default_avatar();
    }();
    return image;
}

std::string nickname() {
    std::string path = data_dir() + "/nombre.txt";
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
    AsyncContext& operator=(AsyncContext&& other);

private:
    uint8_t storage_[0x20];  // el juego reserva el objeto; no guardamos estado
};

void Initialize() {}

bool TryOpenPreselectedUser(UserHandle* out) {
    out->data[0] = kUser.data[0];
    out->data[1] = kUser.data[1];
    out->data[2] = 1;
    g_open_handles++;
    return true;
}

Result OpenUser(UserHandle* out, const Uid& uid) {
    if (!same(uid, kUser)) printf("[account] OpenUser de un Uid desconocido: se abre el único usuario\n");
    return TryOpenPreselectedUser(out) ? ResultSuccess : kResultCancelledByUser;
}
void CloseUser(const UserHandle&) {
    int open = g_open_handles.load();
    while (open > 0 && !g_open_handles.compare_exchange_weak(open, open - 1)) {
    }
}

// Los usuarios abiertos que un programa deja al que lanza con oe::ExecuteProgram (Unreal al reiniciarse,
// main+0x14f84) y que el nuevo recoge al arrancar (main+0x15a8c). Aquí cada arranque es limpio: no hay ninguno.
void PushOpenUsers() {}
void PopOpenUsers(int* count, UserHandle*, int) {
    if (count) *count = 0;
}

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

// Los usuarios con un handle abierto en esta aplicación: el único, mientras el juego lo tenga abierto.
Result ListOpenUsers(int* count, Uid* out, int max) { return list_users(count, out, max, g_open_handles > 0 ? 1 : 0); }
// Los que pueden arrancar la aplicación (todos, sin restricciones de cuenta de red).
Result ListQualifiedUsers(int* count, Uid* out, int max) { return list_users(count, out, max, 1); }

// El último usuario que abrió alguien en la consola: lo recuerda el sistema entre arranques, y aquí no hay otro.
Result GetLastOpenedUser(Uid* out) {
    *out = kUser;
    return ResultSuccess;
}

// El icono del usuario, un JPEG. Como el sdk (sdk+0x24b0): sin búfer solo dice el tamaño; con él copia la imagen y deja
// en *size lo que ocupa (si no cabe entera, lo que quepa: los juegos reservan ProfileImageBytesMax).
Result LoadProfileImage(size_t* size, void* buffer, size_t buffer_size, const Uid& uid) {
    if (!same(uid, kUser)) return kResultInvalidUserId;
    const std::vector<uint8_t>& image = profile_image();
    if (buffer) memcpy(buffer, image.data(), std::min(buffer_size, image.size()));
    if (size) *size = image.size();
    return ResultSuccess;
}

// La miniatura que la consola enseña junto a la partida en su gestor de datos. Android no tiene nada parecido: se
// guarda tal cual junto a la carpeta de la partida, y se borra cuando el juego lo pide.
Result StoreSaveDataThumbnailImage(const Uid& uid, const void* image, size_t size) {
    if (!same(uid, kUser)) return kResultInvalidUserId;
    if (!image || size != kThumbnailBytes) return kResultInvalidArrayLength;
    const std::string path = data_dir() + "/miniatura_partida.rgba";
    FILE* f = fopen(path.c_str(), "wb");
    const bool ok = f && fwrite(image, 1, size, f) == size;
    if (f) fclose(f);
    if (!ok) printf("[account] no se pudo guardar la miniatura de la partida en %s\n", path.c_str());
    return ResultSuccess;  // la partida no depende de ella
}

Result DeleteSaveDataThumbnailImage(const Uid& uid) {
    if (!same(uid, kUser)) return kResultInvalidUserId;
    unlink((data_dir() + "/miniatura_partida.rgba").c_str());
    return ResultSuccess;
}

// El selector de usuario con un único usuario: lo elige directamente.
Result ShowUserSelector(Uid* out, const UserSelectionSettings&) {
    *out = kUser;
    return ResultSuccess;
}

// El applet para dar de alta otro usuario en la consola (sdk+0x7f0: el mismo applet de usuarios, en modo creación).
// Aquí no se pueden crear usuarios: como si lo cerraran sin crear ninguno.
Result ShowUserCreator() {
    printf("[account] ShowUserCreator: no hay más usuarios que el local\n");
    return kResultCancelledByUser;
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
// Mover un contexto (SDK 8): no guarda estado, así que basta con copiar sus bytes y dejar el otro a cero.
AsyncContext& AsyncContext::operator=(AsyncContext&& other) {
    memcpy(storage_, other.storage_, sizeof storage_);
    memset(other.storage_, 0, sizeof other.storage_);
    return *this;
}
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
