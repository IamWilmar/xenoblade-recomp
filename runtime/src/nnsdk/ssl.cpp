// nn::ssl (y su API en C, nnssl*): sin red no hay con quién negociar. Contextos y conexiones se crean (el juego,
// libcurl y su librería de red los preparan antes de conectar) y la conexión se queda con el socket que le dan,
// pero el handshake falla con ResultConnectionReset, igual que Read/Write/Peek/Poll. Nunca WouldBlock (204) ni
// Timeout (205): con esos el juego hace Poll y vuelve a intentarlo.
//
// Los objetos son del juego y tienen el tamaño del SDK (Context 8 bytes, Connection 0x30); el estado va dentro
// con la disposición del SDK: un id (0 = sin crear), el contexto, el socket en +0x10 (Destroy lo cierra: tras
// SetSocketDescriptor el juego deja de ser su dueño y pone su copia a -1) y el último error en +0x2c.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <new>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/types.h"

#define HIDDEN [[gnu::visibility("hidden")]]  // del SDK pero main no lo importa (lo usa la API en C), o propio

namespace nn::socket {
int Close(int fd);  // socket.cpp: solo cierra descriptores que dio nn::socket
}

namespace nn::ssl {

enum CaCertificateId : int32_t {};

class Context {
public:
    enum SslVersion : int32_t {};

    Context();
    ~Context();
    Result Create(SslVersion version);
    Result Destroy();
    Result GetContextId(unsigned long* id);

private:
    uint64_t id_;
};
static_assert(sizeof(Context) == 8);

class Connection {
public:
    enum IoMode : int32_t {};
    enum SessionCacheMode : int32_t {};
    enum OptionType : int32_t {};
    enum PollEvent : int32_t {};

    Connection();
    ~Connection();
    Result Create(Context* context);
    Result Destroy();
    Result SetSocketDescriptor(int fd);
    Result SetHostName(const char* name, unsigned int length);
    Result SetIoMode(IoMode mode);
    Result SetSessionCacheMode(SessionCacheMode mode);
    Result SetOption(OptionType option, bool enable);
    Result DoHandshake();
    int Read(char* buf, unsigned int size);
    Result Read(char* buf, int* out_size, unsigned int size);
    int Write(const char* buf, unsigned int size);
    Result Write(const char* buf, int* out_size, unsigned int size);
    Result Peek(char* buf, int* out_size, unsigned int size);
    Result Poll(PollEvent* out, PollEvent* in, unsigned int timeout_ms);
    Result GetLastError(Result* out);
    Result GetVerifyCertError(Result* out);
    Result GetConnectionId(unsigned long* id);

    HIDDEN Result DoHandshake(unsigned int* cert_size, unsigned int* cert_count);
    HIDDEN Result Pending(int* out);
    HIDDEN Result SetVerifyOption(uint32_t option);
    HIDDEN Result SetNextAlpnProto(const unsigned char* protos, unsigned int length);
    HIDDEN Result GetNextAlpnProto(uint32_t* state, unsigned char* buf, unsigned int* length, unsigned int size);

private:
    HIDDEN Result fail(Result r);
    HIDDEN Result io_result(const void* buf, unsigned int size) const;

    uint64_t id_;
    uint64_t context_id_;
    int32_t socket_;
    [[maybe_unused]] uint8_t reserved_[0x18];  // el resto del estado del SDK: no se usa
    Result last_error_;
};
static_assert(sizeof(Connection) == 0x30);

class BuiltInManager {
public:
    struct BuiltInCertificateInfo;

    static Result GetBuiltInCertificateBufSize(unsigned int* size, CaCertificateId* ids, unsigned int count);
    static Result GetBuiltInCertificates(BuiltInCertificateInfo** out, unsigned char* buf, unsigned int size,
                                         CaCertificateId* ids, unsigned int count);
};

namespace {

// Módulo ssl (123). Los de validación son los que devuelve el propio SDK en el mismo caso. Todos caben en los 22
// bits a los que la API en C recorta el Result.
constexpr Result kResultNoSocket{123 | (103 << 9)};
constexpr Result kResultConnectionNotCreated{123 | (109 << 9)};  // Connection sin Create
constexpr Result kResultContextNotCreated{123 | (110 << 9)};     // Context sin Create
constexpr Result kResultInvalidSize{123 | (112 << 9)};           // tamaño 0
constexpr Result kResultInvalidPointer{123 | (113 << 9)};        // puntero nulo
constexpr Result kResultInvalidCount{123 | (127 << 9)};          // 0 certificados
constexpr Result kResultConnectionReset{123 | (209 << 9)};

std::atomic<uint64_t> g_next_id{1};

}  // namespace

Result Initialize() { return ResultSuccess; }
Result Finalize() { return ResultSuccess; }

Context::Context() : id_(0) {}
Context::~Context() {}

Result Context::Create(SslVersion) {
    id_ = g_next_id++;
    return ResultSuccess;
}

Result Context::Destroy() {
    if (!id_) return kResultContextNotCreated;
    id_ = 0;
    return ResultSuccess;
}

Result Context::GetContextId(unsigned long* id) {
    if (!id) return kResultInvalidPointer;
    *id = id_;
    return ResultSuccess;
}

Connection::Connection() : id_(0), context_id_(0), socket_(-1), reserved_{}, last_error_(ResultSuccess) {}
Connection::~Connection() {}

Result Connection::Create(Context* context) {
    if (!context) return kResultInvalidPointer;
    unsigned long context_id = 0;
    if (context->GetContextId(&context_id).value || !context_id) return kResultContextNotCreated;
    id_ = g_next_id++;
    context_id_ = context_id;
    return ResultSuccess;
}

Result Connection::Destroy() {
    if (!id_) return kResultConnectionNotCreated;
    if (socket_ >= 0) socket::Close(socket_);
    socket_ = -1;
    id_ = context_id_ = 0;
    return ResultSuccess;
}

Result Connection::SetSocketDescriptor(int fd) {
    if (!id_) return kResultConnectionNotCreated;
    socket_ = fd;
    return ResultSuccess;
}

Result Connection::SetHostName(const char* name, unsigned int length) {
    if (!id_) return kResultConnectionNotCreated;
    if (!name) return kResultInvalidPointer;
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 8)) printf("[ssl] conexión a %.*s: sin red, el handshake fallará\n", static_cast<int>(length), name);
    return ResultSuccess;
}

Result Connection::SetIoMode(IoMode) { return id_ ? ResultSuccess : kResultConnectionNotCreated; }
Result Connection::SetSessionCacheMode(SessionCacheMode) { return id_ ? ResultSuccess : kResultConnectionNotCreated; }
Result Connection::SetOption(OptionType, bool) { return id_ ? ResultSuccess : kResultConnectionNotCreated; }
Result Connection::SetVerifyOption(uint32_t) { return id_ ? ResultSuccess : kResultConnectionNotCreated; }
Result Connection::SetNextAlpnProto(const unsigned char*, unsigned int) {
    return id_ ? ResultSuccess : kResultConnectionNotCreated;
}

// Sin handshake no se ha negociado nada: estado 0 y protocolo vacío.
Result Connection::GetNextAlpnProto(uint32_t* state, unsigned char*, unsigned int* length, unsigned int) {
    if (!state || !length) return kResultInvalidPointer;
    *state = 0;
    *length = 0;
    return ResultSuccess;
}

// El fallo de una operación sobre la conexión queda como último error (GetLastError lo lee de +0x2c, como el SDK).
Result Connection::fail(Result r) {
    last_error_ = r;
    return r;
}

Result Connection::io_result(const void* buf, unsigned int size) const {
    if (!id_) return kResultConnectionNotCreated;
    if (!buf) return kResultInvalidPointer;
    if (!size) return kResultInvalidSize;
    return kResultConnectionReset;
}

Result Connection::DoHandshake() {
    if (!id_) return fail(kResultConnectionNotCreated);
    return fail(socket_ < 0 ? kResultNoSocket : kResultConnectionReset);
}

Result Connection::DoHandshake(unsigned int* cert_size, unsigned int* cert_count) {
    if (cert_size) *cert_size = 0;
    if (cert_count) *cert_count = 0;
    return DoHandshake();
}

int Connection::Read(char* buf, unsigned int size) {
    fail(io_result(buf, size));
    return -1;
}

// Como el SDK: el tamaño leído es -1 si falla la operación, y no se toca si fallan los argumentos.
Result Connection::Read(char* buf, int* out_size, unsigned int size) {
    if (!out_size) return fail(kResultInvalidPointer);
    const Result r = fail(io_result(buf, size));
    if (r.value == kResultConnectionReset.value) *out_size = -1;
    return r;
}

int Connection::Write(const char* buf, unsigned int size) {
    fail(io_result(buf, size));
    return -1;
}

Result Connection::Write(const char* buf, int* out_size, unsigned int size) {
    if (!out_size) return fail(kResultInvalidPointer);
    const Result r = fail(io_result(buf, size));
    if (r.value == kResultConnectionReset.value) *out_size = -1;
    return r;
}

Result Connection::Peek(char* buf, int* out_size, unsigned int size) { return Read(buf, out_size, size); }

Result Connection::Pending(int* out) {
    if (!out) return kResultInvalidPointer;
    *out = 0;
    return fail(id_ ? kResultConnectionReset : kResultConnectionNotCreated);
}

Result Connection::Poll(PollEvent* out, PollEvent*, unsigned int) {
    if (!out) return kResultInvalidPointer;
    *out = PollEvent{};
    return fail(id_ ? kResultConnectionReset : kResultConnectionNotCreated);
}

Result Connection::GetLastError(Result* out) {
    if (!out) return kResultInvalidPointer;
    *out = last_error_;
    return ResultSuccess;
}

// El handshake no falla por el certificado: no hay error de verificación.
Result Connection::GetVerifyCertError(Result* out) {
    if (!out) return kResultInvalidPointer;
    *out = ResultSuccess;
    return ResultSuccess;
}

Result Connection::GetConnectionId(unsigned long* id) {
    if (!id) return kResultInvalidPointer;
    *id = id_;
    return ResultSuccess;
}

// No hay certificados de la consola. Quien los pide (el almacén de CAs de la librería TLS que lleva el juego)
// registra el fallo de cada uno y sigue sin ellos; tampoco le harían falta sin red.
Result BuiltInManager::GetBuiltInCertificateBufSize(unsigned int* size, CaCertificateId* ids, unsigned int) {
    if (!size || !ids) return kResultInvalidPointer;
    *size = 0;
    return kResultInvalidCount;
}

Result BuiltInManager::GetBuiltInCertificates(BuiltInCertificateInfo** out, unsigned char* buf, unsigned int size,
                                              CaCertificateId* ids, unsigned int) {
    if (!out || !buf || !ids) return kResultInvalidPointer;
    if (!size) return kResultInvalidSize;
    return kResultInvalidCount;
}

}  // namespace nn::ssl

using namespace nn;
using namespace nn::ssl;

// La API en C: los mismos objetos (el juego reserva Context y Connection, Create/Destroy los construyen y
// destruyen en su sitio). Un puntero nulo da ResultInvalidPointer.
extern "C" {

Result nnsslInitialize() { return nn::ssl::Initialize(); }
Result nnsslFinalize() { return nn::ssl::Finalize(); }

Result nnsslContextDestroy(Context* context) {
    if (!context) return kResultInvalidPointer;
    const Result r = context->Destroy();
    if (!r.value) context->~Context();
    return r;
}
Result nnsslContextGetContextId(Context* context, unsigned long* id) {
    if (!context) return kResultInvalidPointer;
    return context->GetContextId(id);
}

Result nnsslConnectionCreate(Connection* connection, Context* context) {
    if (!connection || !context) return kResultInvalidPointer;
    new (connection) Connection();
    return connection->Create(context);
}
Result nnsslConnectionDestroy(Connection* connection) {
    if (!connection) return kResultInvalidPointer;
    const Result r = connection->Destroy();
    if (!r.value) connection->~Connection();
    return r;
}

Result nnsslConnectionSetSocketDescriptor(Connection* c, int fd) {
    return c ? c->SetSocketDescriptor(fd) : kResultInvalidPointer;
}
Result nnsslConnectionSetHostName(Connection* c, const char* name, unsigned int length) {
    return c ? c->SetHostName(name, length) : kResultInvalidPointer;
}
Result nnsslConnectionSetVerifyOption(Connection* c, uint32_t option) {
    return c ? c->SetVerifyOption(option) : kResultInvalidPointer;
}
Result nnsslConnectionSetIoMode(Connection* c, uint32_t mode) {
    return c ? c->SetIoMode(static_cast<Connection::IoMode>(mode)) : kResultInvalidPointer;
}
Result nnsslConnectionSetSessionCacheMode(Connection* c, uint32_t mode) {
    return c ? c->SetSessionCacheMode(static_cast<Connection::SessionCacheMode>(mode)) : kResultInvalidPointer;
}
Result nnsslConnectionSetOption(Connection* c, uint32_t option, bool enable) {
    return c ? c->SetOption(static_cast<Connection::OptionType>(option), enable) : kResultInvalidPointer;
}
Result nnsslConnectionSetNextAlpnProto(Connection* c, const unsigned char* protos, unsigned int length) {
    return c ? c->SetNextAlpnProto(protos, length) : kResultInvalidPointer;
}
Result nnsslConnectionGetNextAlpnProto(Connection* c, uint32_t* state, unsigned char* buf, unsigned int* length,
                                       unsigned int size) {
    return c ? c->GetNextAlpnProto(state, buf, length, size) : kResultInvalidPointer;
}

Result nnsslConnectionDoHandshakeWithCertBuffer(Connection* c, unsigned int* cert_size, unsigned int* cert_count) {
    return c ? c->DoHandshake(cert_size, cert_count) : kResultInvalidPointer;
}
Result nnsslConnectionRead(Connection* c, char* buf, int* out_size, unsigned int size) {
    return c ? c->Read(buf, out_size, size) : kResultInvalidPointer;
}
Result nnsslConnectionWrite(Connection* c, const char* buf, int* out_size, unsigned int size) {
    return c ? c->Write(buf, out_size, size) : kResultInvalidPointer;
}
Result nnsslConnectionPeek(Connection* c, char* buf, int* out_size, unsigned int size) {
    return c ? c->Peek(buf, out_size, size) : kResultInvalidPointer;
}
Result nnsslConnectionPending(Connection* c, int* out) { return c ? c->Pending(out) : kResultInvalidPointer; }
Result nnsslConnectionGetVerifyCertError(Connection* c, Result* out) {
    return c ? c->GetVerifyCertError(out) : kResultInvalidPointer;
}
Result nnsslConnectionGetConnectionId(Connection* c, unsigned long* id) {
    return c ? c->GetConnectionId(id) : kResultInvalidPointer;
}

}  // extern "C"
