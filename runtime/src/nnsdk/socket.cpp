// nn::socket (y su API en C, nnsocket*): una consola sin red. Los descriptores son fds reales de bionic, así
// Poll/Select/Close valen igual para sockets y para los eventfd que el juego sí necesita sin red: el
// PhysicalSocketServer de WebRTC aborta si EventFd falla y gRPC despierta su poller con otro.
//
// Los sockets AF_INET se crean si Android deja (la APK no pide INTERNET: sin el permiso socket() falla y se
// responde ENETDOWN, como una consola sin red), pero solo hablan con 127.0.0.0/8: conectar o enviar a otra
// dirección da ENETUNREACH, aunque la Flip tenga Wi-Fi. El resolver solo entiende direcciones numéricas y
// "localhost"; ningún nombre se resuelve.
//
// Las constantes son las del servicio bsd de la consola (FreeBSD: SOL_SOCKET 0xffff, sockaddr con sa_len,
// MSG_DONTWAIT 0x80, EAI_*) salvo errno y O_NONBLOCK, que ya llegan con los valores de Linux. errno es el de
// bionic: __errno_location (libc.cpp) es el mismo para el juego.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <strings.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/watch.h"
#include "nnsdk/types.h"

// Tipos de la API vieja: el nombre global sale en el mangling (sockaddr y pollfd ya los declara bionic). Se usan
// solo por puntero: su memoria tiene la disposición de la consola, no la de bionic.
struct addrinfo;
struct hostent;

namespace nn::socket {

enum class Family : uint8_t {};  // de un byte: el SDK hace `and w0, #0xff`
enum class Type : int32_t {};
enum class Protocol : int32_t {};
enum class Level : int32_t {};
enum class Option : int32_t {};
enum class MsgFlag : int32_t {};
enum class FcntlCommand : int32_t {};
enum class EventFdFlags : int32_t {};
enum class ResolverOptionKey : uint32_t {};
enum class Errno : int32_t {};

struct Config;
struct SockAddr;

struct InAddr {
    uint32_t addr;  // orden de red
};
struct PollFd {
    int32_t fd;
    int16_t events;
    int16_t revents;
};
struct FdSet {
    uint64_t bits[1024 / 64];  // FD_SETSIZE 1024, como el SDK (manda 0x80 bytes por conjunto)
};
struct TimeVal {
    int64_t sec;
    int64_t usec;
};

// AddrInfo de la API nueva: orden de musl (ai_addr antes que ai_canonname). ai_family es Family (un byte): de
// las pistas solo se lee el primero.
struct AddrInfo {
    int32_t flags;
    int32_t family;
    int32_t socktype;
    int32_t protocol;
    uint32_t addrlen;
    void* addr;
    char* canonname;
    AddrInfo* next;
};
static_assert(offsetof(AddrInfo, addr) == 0x18 && offsetof(AddrInfo, next) == 0x28);

// hostent y HostEnt comparten disposición; el juego lee h_addr_list[0] en +0x18.
struct HostEnt {
    char* name;
    char** aliases;
    int32_t addrtype;
    int32_t length;
    InAddr** addr_list;
};

// {clave, tipo, tamaño, valor}. Tipos del SDK: 100 bool, 101 int32, 103 int64, 105 puntero.
struct ResolverOption {
    ResolverOptionKey key;
    uint32_t type;
    uint64_t size;
    union {
        bool b;
        int32_t i32;
        int64_t i64;
        const void* ptr;
    } data;
};
static_assert(sizeof(ResolverOption) == 0x18);

}  // namespace nn::socket

using namespace nn;
using namespace nn::socket;

namespace {

// sockaddr_in de la consola: longitud y familia de un byte cada una.
struct BsdSockAddrIn {
    uint8_t len;
    uint8_t family;
    uint16_t port;  // orden de red
    uint32_t addr;  // orden de red
    uint8_t zero[8];
};
static_assert(sizeof(BsdSockAddrIn) == 16);

// addrinfo de la API vieja: orden de FreeBSD (ai_canonname antes que ai_addr).
struct LegacyAddrInfo {
    int32_t flags;
    int32_t family;
    int32_t socktype;
    int32_t protocol;
    uint32_t addrlen;
    char* canonname;
    void* addr;
    LegacyAddrInfo* next;
};
static_assert(offsetof(LegacyAddrInfo, addr) == 0x20 && offsetof(LegacyAddrInfo, next) == 0x28);

// msghdr de la API en C, el de la consola (0x30 bytes): msg_controllen de 32 bits y msg_flags en +0x2c.
struct MsgHdr {
    void* name;
    uint32_t namelen;
    iovec* iov;
    int32_t iovlen;
    void* control;
    uint32_t controllen;
    int32_t flags;
};
static_assert(sizeof(MsgHdr) == 0x30 && offsetof(MsgHdr, flags) == 0x2c);

constexpr int kAfInet = 2;
constexpr int kSockNonBlock = 0x20000000, kSockCloExec = 0x10000000;  // bits altos del tipo (FreeBSD)
constexpr int kAiPassive = 0x1, kAiCanonName = 0x2;

// AiErrno de la consola (los de FreeBSD).
constexpr int kEaiFail = 4, kEaiFamily = 5, kEaiMemory = 6, kEaiNoName = 8, kEaiService = 9;

int fail(int err) {
    errno = err;
    return -1;
}

// Descriptores que ha dado nn::socket: Close solo cierra esos. Son fds del proceso, y un Close repetido (en la
// consola un EBADF inofensivo) cerraría aquí el archivo o el fd de la GPU que haya heredado el número.
constexpr int kMaxFd = 1 << 16;
std::atomic<uint64_t> g_owned[kMaxFd / 64];

int own(int fd) {
    if (fd < 0) return fd;
    if (fd >= kMaxFd) {
        ::close(fd);
        return fail(EMFILE);
    }
    g_owned[fd >> 6].fetch_or(uint64_t{1} << (fd & 63));
    return fd;
}

bool disown(int fd) {
    if (fd < 0 || fd >= kMaxFd) return false;
    const uint64_t bit = uint64_t{1} << (fd & 63);
    return g_owned[fd >> 6].fetch_and(~bit) & bit;
}

// Sin red solo existe la máquina propia: 127.0.0.0/8 y 0.0.0.0 (que Linux también trata como local).
bool is_local(uint32_t net_addr) {
    const uint32_t a = ntohl(net_addr);
    return a == INADDR_ANY || (a >> 24) == 127;
}

void log_offline(const char* what, const sockaddr_in& to) {
    static std::atomic<int> logged{0};
    if (logged++ >= 20) return;
    char ip[INET_ADDRSTRLEN] = "?";
    inet_ntop(AF_INET, &to.sin_addr, ip, sizeof ip);
    printf("[socket] %s a %s:%d: sin red\n", what, ip, ntohs(to.sin_port));
}

bool to_host_addr(const void* p, unsigned len, sockaddr_in* out) {
    const auto* sa = static_cast<const BsdSockAddrIn*>(p);
    if (!sa || len < 8) {
        errno = EINVAL;
        return false;
    }
    if (sa->family != kAfInet) {
        errno = EAFNOSUPPORT;
        return false;
    }
    *out = {};
    out->sin_family = AF_INET;
    out->sin_port = sa->port;
    out->sin_addr.s_addr = sa->addr;
    return true;
}

void from_host_addr(const sockaddr_storage& ss, void* p, unsigned* len) {
    if (!p || !len) return;
    BsdSockAddrIn sa{};
    sa.len = sizeof sa;
    if (ss.ss_family == AF_INET) {
        const auto& in = reinterpret_cast<const sockaddr_in&>(ss);
        sa.family = kAfInet;
        sa.port = in.sin_port;
        sa.addr = in.sin_addr.s_addr;
    }
    memcpy(p, &sa, std::min<unsigned>(*len, sizeof sa));
    *len = sizeof sa;
}

// MsgFlag de la consola -> Linux. OOB, PEEK y DONTROUTE coinciden.
int host_msg_flags(int f) {
    int r = f & (MSG_OOB | MSG_PEEK | MSG_DONTROUTE);
    if (f & 0x8) r |= MSG_EOR;
    if (f & 0x10) r |= MSG_TRUNC;
    if (f & 0x20) r |= MSG_CTRUNC;
    if (f & 0x40) r |= MSG_WAITALL;
    if (f & 0x80) r |= MSG_DONTWAIT;
    return r;
}

int switch_msg_flags(int f) {
    int r = f & (MSG_OOB | MSG_PEEK | MSG_DONTROUTE);
    if (f & MSG_EOR) r |= 0x8;
    if (f & MSG_TRUNC) r |= 0x10;
    if (f & MSG_CTRUNC) r |= 0x20;
    if (f & MSG_WAITALL) r |= 0x40;
    if (f & MSG_DONTWAIT) r |= 0x80;
    return r;
}

// La consola no tiene SIGPIPE; en Android un send a un socket cerrado mataría el proceso.
constexpr int kSendFlags = MSG_NOSIGNAL;

int do_socket(int family, int type, int protocol) {
    if (family != kAfInet) return fail(EAFNOSUPPORT);  // la consola solo tiene IPv4
    const int flags = SOCK_CLOEXEC | (type & kSockNonBlock ? SOCK_NONBLOCK : 0);
    const int fd = ::socket(AF_INET, (type & ~(kSockNonBlock | kSockCloExec)) | flags, protocol);
    if (fd < 0 && (errno == EACCES || errno == EPERM)) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) printf("[socket] Android no deja crear sockets (sin permiso INTERNET): ENETDOWN\n");
        return fail(ENETDOWN);
    }
    return own(fd);
}

int do_bind(int fd, const void* addr, unsigned len) {
    sockaddr_in in;
    if (!to_host_addr(addr, len, &in)) return -1;
    if (!is_local(in.sin_addr.s_addr)) return fail(EADDRNOTAVAIL);  // no hay más interfaz que lo
    return ::bind(fd, reinterpret_cast<const sockaddr*>(&in), sizeof in);
}

int do_connect(int fd, const void* addr, unsigned len) {
    sockaddr_in in;
    if (!to_host_addr(addr, len, &in)) return -1;
    if (!is_local(in.sin_addr.s_addr)) {
        log_offline("connect", in);
        return fail(ENETUNREACH);
    }
    return ::connect(fd, reinterpret_cast<const sockaddr*>(&in), sizeof in);
}

int do_accept(int fd, void* addr, unsigned* len) {
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    const int r = ::accept4(fd, reinterpret_cast<sockaddr*>(&ss), &sl, SOCK_CLOEXEC);
    if (r >= 0) from_host_addr(ss, addr, len);
    return own(r);
}

int do_sockname(int fd, void* addr, unsigned* len, bool peer) {
    if (!addr || !len) return fail(EFAULT);
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    const int r = peer ? ::getpeername(fd, reinterpret_cast<sockaddr*>(&ss), &sl)
                       : ::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &sl);
    if (r == 0) from_host_addr(ss, addr, len);
    return r;
}

ssize_t do_recvfrom(int fd, void* buf, size_t n, int flags, void* addr, unsigned* len) {
    nnsdk::watch::before_kernel_write(buf, n);
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    const bool want = addr && len;
    const ssize_t r = ::recvfrom(fd, buf, n, host_msg_flags(flags), want ? reinterpret_cast<sockaddr*>(&ss) : nullptr,
                                 want ? &sl : nullptr);
    if (r >= 0 && want) from_host_addr(ss, addr, len);
    return r;
}

ssize_t do_sendto(int fd, const void* buf, size_t n, int flags, const void* addr, unsigned len) {
    if (!addr) return ::send(fd, buf, n, host_msg_flags(flags) | kSendFlags);
    sockaddr_in in;
    if (!to_host_addr(addr, len, &in)) return -1;
    if (!is_local(in.sin_addr.s_addr)) {
        log_offline("sendto", in);
        return fail(ENETUNREACH);
    }
    return ::sendto(fd, buf, n, host_msg_flags(flags) | kSendFlags, reinterpret_cast<const sockaddr*>(&in), sizeof in);
}

// Los datos auxiliares (cmsg) tienen otra cabecera en Linux; sin red nadie los necesita: no se mandan y al recibir
// se devuelven vacíos.
ssize_t do_sendmsg(int fd, const MsgHdr* m, int flags) {
    if (!m) return fail(EINVAL);
    sockaddr_in in;
    msghdr h{};
    if (m->name) {
        if (!to_host_addr(m->name, m->namelen, &in)) return -1;
        if (!is_local(in.sin_addr.s_addr)) {
            log_offline("sendmsg", in);
            return fail(ENETUNREACH);
        }
        h.msg_name = &in;
        h.msg_namelen = sizeof in;
    }
    h.msg_iov = m->iov;
    h.msg_iovlen = m->iovlen;
    return ::sendmsg(fd, &h, host_msg_flags(flags) | kSendFlags);
}

ssize_t do_recvmsg(int fd, MsgHdr* m, int flags) {
    if (!m) return fail(EINVAL);
    sockaddr_storage ss{};
    msghdr h{};
    if (m->name) {
        h.msg_name = &ss;
        h.msg_namelen = sizeof ss;
    }
    h.msg_iov = m->iov;
    h.msg_iovlen = m->iovlen;
    const ssize_t r = ::recvmsg(fd, &h, host_msg_flags(flags));
    if (r >= 0) {
        if (m->name) from_host_addr(ss, m->name, &m->namelen);
        m->controllen = 0;
        m->flags = switch_msg_flags(h.msg_flags);
    }
    return r;
}

ssize_t do_write(int fd, const void* buf, size_t n) {
    const ssize_t r = ::send(fd, buf, n, kSendFlags);  // write() a un socket cerrado daría SIGPIPE
    return r < 0 && errno == ENOTSOCK ? ::write(fd, buf, n) : r;
}

int do_close(int fd) {
    if (!disown(fd)) return fail(EBADF);
    return ::close(fd);
}

// Bits de poll iguales en los dos salvo POLLWRBAND: 0x100 en la consola, 0x200 aquí (0x100 es POLLWRNORM).
int do_poll(pollfd* game, unsigned long n, int timeout) {
    pollfd local[16];
    std::vector<pollfd> big;
    pollfd* fds = local;
    if (n > 16) {
        big.resize(n);
        fds = big.data();
    }
    for (unsigned long i = 0; i < n; i++) {
        fds[i] = game[i];
        if (fds[i].events & 0x100) fds[i].events = static_cast<short>((fds[i].events & ~0x100) | POLLWRBAND);
    }
    const int r = ::poll(fds, n, timeout);
    if (r < 0) return r;
    for (unsigned long i = 0; i < n; i++) {
        const short rev = fds[i].revents;
        short out = rev & ~(POLLWRNORM | POLLWRBAND);
        if (rev & POLLWRNORM) out |= POLLOUT;
        if (rev & POLLWRBAND) out |= 0x100;
        game[i].revents = out;
    }
    return r;
}

// select sobre poll: el FD_SET de bionic con FORTIFY aborta con fds >= 1024 y los nuestros pueden pasar de ahí.
int do_select(int nfds, FdSet* rd, FdSet* wr, FdSet* ex, const TimeVal* tv) {
    if (nfds < 0 || nfds > 1024) return fail(EINVAL);
    auto has = [](const FdSet* s, int fd) { return s && (s->bits[fd >> 6] >> (fd & 63) & 1); };
    std::vector<pollfd> fds;
    for (int fd = 0; fd < nfds; fd++) {
        const short ev = (has(rd, fd) ? POLLIN : 0) | (has(wr, fd) ? POLLOUT : 0) | (has(ex, fd) ? POLLPRI : 0);
        if (ev) fds.push_back({fd, ev, 0});
    }
    int timeout = -1;
    if (tv) timeout = static_cast<int>(std::clamp<int64_t>(tv->sec * 1000 + (tv->usec + 999) / 1000, 0, INT_MAX));
    if (::poll(fds.data(), fds.size(), timeout) < 0) return -1;
    for (const pollfd& p : fds)
        if (p.revents & POLLNVAL) return fail(EBADF);
    for (FdSet* s : {rd, wr, ex})
        if (s) memset(s, 0, sizeof *s);
    int count = 0;
    auto set = [&count](FdSet* s, int fd) {
        s->bits[fd >> 6] |= uint64_t{1} << (fd & 63);
        count++;
    };
    for (const pollfd& p : fds) {
        if ((p.events & POLLIN) && (p.revents & (POLLIN | POLLHUP | POLLERR))) set(rd, p.fd);
        if ((p.events & POLLOUT) && (p.revents & (POLLOUT | POLLHUP | POLLERR))) set(wr, p.fd);
        if ((p.events & POLLPRI) && (p.revents & POLLPRI)) set(ex, p.fd);
    }
    return count;
}

// F_GETFD/F_SETFD (1/2) y F_GETFL/F_SETFL (3/4) tienen los números de Linux, y O_NONBLOCK llega ya como 0x800.
int do_fcntl(int fd, int cmd, int arg) {
    switch (cmd) {
        case F_GETFD:
        case F_SETFD: return ::fcntl(fd, cmd, arg);
        case F_GETFL: {
            const int f = ::fcntl(fd, F_GETFL);
            return f < 0 ? -1 : f & (O_NONBLOCK | O_ACCMODE);
        }
        case F_SETFL: {
            const int f = ::fcntl(fd, F_GETFL);
            return f < 0 ? -1 : ::fcntl(fd, F_SETFL, (f & ~O_NONBLOCK) | (arg & O_NONBLOCK));
        }
        default: return fail(EINVAL);
    }
}

// Opciones de la consola (FreeBSD) -> Linux. Las mismas estructuras de valor (int, linger, timeval de 64 bits).
struct SockOpt {
    int level, name, host_level, host_name;
};
constexpr int kSolSocket = 0xffff;
constexpr SockOpt kSockOpts[] = {
    {kSolSocket, 0x0002, SOL_SOCKET, SO_ACCEPTCONN},
    {kSolSocket, 0x0004, SOL_SOCKET, SO_REUSEADDR},
    {kSolSocket, 0x0008, SOL_SOCKET, SO_KEEPALIVE},
    {kSolSocket, 0x0010, SOL_SOCKET, SO_DONTROUTE},
    {kSolSocket, 0x0020, SOL_SOCKET, SO_BROADCAST},
    {kSolSocket, 0x0080, SOL_SOCKET, SO_LINGER},
    {kSolSocket, 0x0100, SOL_SOCKET, SO_OOBINLINE},
    {kSolSocket, 0x0200, SOL_SOCKET, SO_REUSEPORT},
    {kSolSocket, 0x1001, SOL_SOCKET, SO_SNDBUF},
    {kSolSocket, 0x1002, SOL_SOCKET, SO_RCVBUF},
    {kSolSocket, 0x1003, SOL_SOCKET, SO_SNDLOWAT},
    {kSolSocket, 0x1004, SOL_SOCKET, SO_RCVLOWAT},
    {kSolSocket, 0x1005, SOL_SOCKET, SO_SNDTIMEO},
    {kSolSocket, 0x1006, SOL_SOCKET, SO_RCVTIMEO},
    {kSolSocket, 0x1007, SOL_SOCKET, SO_ERROR},
    {kSolSocket, 0x1008, SOL_SOCKET, SO_TYPE},
    {IPPROTO_IP, 3, IPPROTO_IP, IP_TOS},
    {IPPROTO_IP, 4, IPPROTO_IP, IP_TTL},
    {IPPROTO_IP, 9, IPPROTO_IP, IP_MULTICAST_IF},
    {IPPROTO_IP, 10, IPPROTO_IP, IP_MULTICAST_TTL},
    {IPPROTO_IP, 11, IPPROTO_IP, IP_MULTICAST_LOOP},
    {IPPROTO_IP, 12, IPPROTO_IP, IP_ADD_MEMBERSHIP},
    {IPPROTO_IP, 13, IPPROTO_IP, IP_DROP_MEMBERSHIP},
    {IPPROTO_TCP, 0x001, IPPROTO_TCP, TCP_NODELAY},
    {IPPROTO_TCP, 0x002, IPPROTO_TCP, TCP_MAXSEG},
    {IPPROTO_TCP, 0x100, IPPROTO_TCP, TCP_KEEPIDLE},
    {IPPROTO_TCP, 0x200, IPPROTO_TCP, TCP_KEEPINTVL},
    {IPPROTO_TCP, 0x400, IPPROTO_TCP, TCP_KEEPCNT},
};

const SockOpt* find_option(int level, int name) {
    for (const SockOpt& o : kSockOpts)
        if (o.level == level && o.name == name) return &o;
    return nullptr;
}

int do_setsockopt(int fd, int level, int name, const void* value, unsigned len) {
    const SockOpt* o = find_option(level, name);
    if (!o) {
        // SO_NOSIGPIPE (ya se manda con MSG_NOSIGNAL), TCP_NOPUSH y las propias de Nintendo: sin equivalente que
        // cambie algo sin red. Se aceptan, como las aceptaría la consola.
        static std::atomic<int> logged{0};
        if (nvn::first_times(logged, 8)) printf("[socket] SetSockOpt(0x%x, 0x%x) ignorada\n", level, name);
        return 0;
    }
    return ::setsockopt(fd, o->host_level, o->host_name, value, len);
}

int do_getsockopt(int fd, int level, int name, void* value, unsigned* len) {
    const SockOpt* o = find_option(level, name);
    if (!o) return fail(ENOPROTOOPT);
    socklen_t l = len ? *len : 0;
    const int r = ::getsockopt(fd, o->host_level, o->host_name, value, &l);
    if (r == 0 && len) *len = l;
    return r;
}

int do_eventfd(unsigned long initval, int flags) {
    // EventFdFlags: 1 semáforo, 4 no bloqueante (los de FreeBSD).
    const int host = EFD_CLOEXEC | (flags & 1 ? EFD_SEMAPHORE : 0) | (flags & 4 ? EFD_NONBLOCK : 0);
    return own(::eventfd(static_cast<unsigned>(initval), host));
}

// Dirección de un nombre sin preguntar a nadie: numérica o localhost. Una consola sin red no resuelve más.
bool resolve_local(const char* name, uint32_t* addr) {
    if (!name) return false;
    in_addr a;
    if (inet_pton(AF_INET, name, &a) == 1) {
        *addr = a.s_addr;
        return true;
    }
    if (!strcasecmp(name, "localhost") || !strcasecmp(name, "localhost.localdomain")) {
        *addr = htonl(INADDR_LOOPBACK);
        return true;
    }
    static std::atomic<int> logged{0};
    if (nvn::first_times(logged, 20)) printf("[socket] resolver %s: sin red\n", name);
    return false;
}

// Una sola entrada (IPv4) en un bloque con su sockaddr: FreeAddrInfo libera el bloque entero.
template <typename Info>
int make_addrinfo(const char* node, const char* service, const Info* hints, Info** out) {
    if (!out) return kEaiFail;
    *out = nullptr;
    const int flags = hints ? hints->flags : 0;
    const int family = hints ? hints->family & 0xff : 0;
    if (family != 0 && family != kAfInet) return kEaiFamily;
    if (!node && !service) return kEaiNoName;
    uint32_t addr;
    if (!node)
        addr = htonl(flags & kAiPassive ? INADDR_ANY : INADDR_LOOPBACK);
    else if (!resolve_local(node, &addr))
        return kEaiNoName;
    uint16_t port = 0;
    if (service && *service) {
        char* end;
        const unsigned long p = strtoul(service, &end, 10);
        if (*end || p > 65535) return kEaiService;  // sin /etc/services: solo puertos numéricos
        port = htons(static_cast<uint16_t>(p));
    }
    struct Block {
        Info info;
        BsdSockAddrIn sa;
        char name[256];
    };
    auto* b = static_cast<Block*>(calloc(1, sizeof(Block)));
    if (!b) return kEaiMemory;
    b->sa.len = sizeof b->sa;
    b->sa.family = kAfInet;
    b->sa.port = port;
    b->sa.addr = addr;
    const int socktype = hints && hints->socktype ? hints->socktype : SOCK_STREAM;
    b->info.flags = flags;
    b->info.family = kAfInet;
    b->info.socktype = socktype;
    b->info.protocol = hints && hints->protocol ? hints->protocol : socktype == SOCK_DGRAM ? IPPROTO_UDP : IPPROTO_TCP;
    b->info.addrlen = sizeof b->sa;
    b->info.addr = &b->sa;
    if ((flags & kAiCanonName) && node) {
        snprintf(b->name, sizeof b->name, "%s", node);
        b->info.canonname = b->name;
    }
    *out = &b->info;
    return 0;
}

template <typename Info>
void free_addrinfo(Info* info) {
    while (info) {
        Info* next = info->next;
        free(info);  // es el principio de su bloque
        info = next;
    }
}

// Como en la consola, el resultado vive en un buffer del hilo hasta la siguiente llamada.
HostEnt* host_by_name(const char* name) {
    struct Entry {
        HostEnt h;
        char name[256];
        char* aliases[1];
        InAddr addr;
        InAddr* list[2];
    };
    static thread_local Entry e;
    uint32_t addr;
    if (!resolve_local(name, &addr)) return nullptr;
    snprintf(e.name, sizeof e.name, "%s", name);
    e.aliases[0] = nullptr;
    e.addr.addr = addr;
    e.list[0] = &e.addr;
    e.list[1] = nullptr;
    e.h = {e.name, e.aliases, kAfInet, sizeof(InAddr), e.list};
    return &e.h;
}

// Claves del resolver que el SDK atiende sin servicio: 10001 da un id de petición nuevo (libcurl y el juego lo usan
// luego como asa de cancelación, claves 20001 y 30002) y 0x11171..0x11176 son opciones int64 locales. El resto
// iría al servicio: se aceptan y no cambian nada sin red. ResolverSetOption no debe fallar nunca con EAGAIN:
// libcurl lo reintenta en bucle cada 10 ms.
constexpr uint32_t kKeyRequestId = 10001, kKeyLocalFirst = 0x11171, kKeyLocalLast = 0x11176;
std::atomic<int32_t> g_request_id{1};
std::atomic<int64_t> g_local_options[kKeyLocalLast - kKeyLocalFirst + 1];

int resolver_get_option(ResolverOption* out, uint32_t key) {
    if (!out) return fail(EINVAL);
    if (key == kKeyRequestId) {
        out->key = static_cast<ResolverOptionKey>(key);
        out->type = 101;
        out->size = 4;
        out->data.i32 = g_request_id++;
        return 0;
    }
    if (key >= kKeyLocalFirst && key <= kKeyLocalLast) {
        out->key = static_cast<ResolverOptionKey>(key);
        out->type = 103;
        out->size = 8;
        out->data.i64 = g_local_options[key - kKeyLocalFirst];
        return 0;
    }
    return fail(EINVAL);
}

int resolver_set_option(const ResolverOption* opt) {
    if (!opt) return fail(EINVAL);
    const uint32_t key = static_cast<uint32_t>(opt->key);
    if (key >= kKeyLocalFirst && key <= kKeyLocalLast && opt->type == 103 && opt->size == 8)
        g_local_options[key - kKeyLocalFirst] = opt->data.i64;
    return 0;
}

const char* gai_error(int code) {
    switch (code) {
        case 0: return "Success";
        case 1: return "Address family for hostname not supported";
        case 2: return "Temporary failure in name resolution";
        case 3: return "Invalid value for ai_flags";
        case 4: return "Non-recoverable failure in name resolution";
        case 5: return "ai_family not supported";
        case 6: return "Memory allocation failure";
        case 7: return "No address associated with hostname";
        case 8: return "hostname nor servname provided, or not known";
        case 9: return "servname not supported for ai_socktype";
        case 10: return "ai_socktype not supported";
        case 11: return "System error returned in errno";
        case 12: return "Invalid value for hints";
        case 13: return "Resolved protocol is unknown";
        case 14: return "Argument buffer overflow";
        default: return "Unknown error";
    }
}

}  // namespace

namespace nn::socket {

Result Initialize(const Config&) {
    printf("[socket] Initialize: consola sin red (solo 127.0.0.1)\n");
    return ResultSuccess;
}
Result Finalize() { return ResultSuccess; }

int GetLastErrno() { return errno; }
Errno GetLastError() { return static_cast<Errno>(errno); }

int Socket(int family, int type, int protocol) { return do_socket(family, type, protocol); }
int Socket(Family family, Type type, Protocol protocol) {
    return do_socket(static_cast<uint8_t>(family), static_cast<int>(type), static_cast<int>(protocol));
}

int Bind(int fd, const sockaddr* addr, unsigned int len) { return do_bind(fd, addr, len); }
int Bind(int fd, const SockAddr* addr, unsigned int len) { return do_bind(fd, addr, len); }
int Connect(int fd, const sockaddr* addr, unsigned int len) { return do_connect(fd, addr, len); }
int Connect(int fd, const SockAddr* addr, unsigned int len) { return do_connect(fd, addr, len); }
int Accept(int fd, sockaddr* addr, unsigned int* len) { return do_accept(fd, addr, len); }
int Accept(int fd, SockAddr* addr, unsigned int* len) { return do_accept(fd, addr, len); }
int Listen(int fd, int backlog) { return ::listen(fd, backlog); }
int Shutdown(int fd, int how) { return ::shutdown(fd, how); }  // SHUT_RD/WR/RDWR: 0, 1, 2 en los dos
int Close(int fd) { return do_close(fd); }

int GetSockName(int fd, sockaddr* addr, unsigned int* len) { return do_sockname(fd, addr, len, false); }
int GetSockName(int fd, SockAddr* addr, unsigned int* len) { return do_sockname(fd, addr, len, false); }
int GetPeerName(int fd, sockaddr* addr, unsigned int* len) { return do_sockname(fd, addr, len, true); }

int GetSockOpt(int fd, int level, int name, void* value, unsigned int* len) {
    return do_getsockopt(fd, level, name, value, len);
}
int SetSockOpt(int fd, int level, int name, const void* value, unsigned int len) {
    return do_setsockopt(fd, level, name, value, len);
}
int SetSockOpt(int fd, Level level, Option name, const void* value, unsigned int len) {
    return do_setsockopt(fd, static_cast<int>(level), static_cast<int>(name), value, len);
}

ssize_t Recv(int fd, void* buf, unsigned long n, int flags) { return do_recvfrom(fd, buf, n, flags, nullptr, nullptr); }
ssize_t Recv(int fd, void* buf, unsigned long n, MsgFlag flags) {
    return do_recvfrom(fd, buf, n, static_cast<int>(flags), nullptr, nullptr);
}
ssize_t RecvFrom(int fd, void* buf, unsigned long n, int flags, sockaddr* addr, unsigned int* len) {
    return do_recvfrom(fd, buf, n, flags, addr, len);
}
ssize_t RecvFrom(int fd, void* buf, unsigned long n, MsgFlag flags, SockAddr* addr, unsigned int* len) {
    return do_recvfrom(fd, buf, n, static_cast<int>(flags), addr, len);
}
ssize_t Send(int fd, const void* buf, unsigned long n, int flags) { return do_sendto(fd, buf, n, flags, nullptr, 0); }
ssize_t Send(int fd, const void* buf, unsigned long n, MsgFlag flags) {
    return do_sendto(fd, buf, n, static_cast<int>(flags), nullptr, 0);
}
ssize_t SendTo(int fd, const void* buf, unsigned long n, int flags, const sockaddr* addr, unsigned int len) {
    return do_sendto(fd, buf, n, flags, addr, len);
}
ssize_t SendTo(int fd, const void* buf, unsigned long n, MsgFlag flags, const SockAddr* addr, unsigned int len) {
    return do_sendto(fd, buf, n, static_cast<int>(flags), addr, len);
}
ssize_t Read(int fd, void* buf, unsigned long n) { return ::read(fd, buf, n); }
ssize_t Write(int fd, const void* buf, unsigned long n) { return do_write(fd, buf, n); }

int Fcntl(int fd, FcntlCommand cmd, ...) {
    va_list args;
    va_start(args, cmd);
    const int arg = va_arg(args, int);
    va_end(args);
    return do_fcntl(fd, static_cast<int>(cmd), arg);
}

int Poll(pollfd* fds, unsigned long n, int timeout) { return do_poll(fds, n, timeout); }
int Poll(PollFd* fds, unsigned long n, int timeout) {
    static_assert(sizeof(PollFd) == sizeof(pollfd));
    return do_poll(reinterpret_cast<pollfd*>(fds), n, timeout);
}
int Select(int nfds, FdSet* rd, FdSet* wr, FdSet* ex, TimeVal* timeout) { return do_select(nfds, rd, wr, ex, timeout); }

int EventFd(unsigned long initval, EventFdFlags flags) { return do_eventfd(initval, static_cast<int>(flags)); }
// Como el SDK: un read/write de 8 bytes; 0 o -1.
int EventFdRead(int fd, unsigned long* value) {
    if (!value) return -1;
    uint64_t v;
    if (::read(fd, &v, sizeof v) != sizeof v) return -1;
    *value = v;
    return 0;
}
int EventFdWrite(int fd, unsigned long value) {
    const uint64_t v = value;
    return ::write(fd, &v, sizeof v) == sizeof v ? 0 : -1;
}

// La consola solo tiene AF_INET: con otra familia EAFNOSUPPORT, como el SDK.
const char* InetNtop(int family, const void* src, char* dst, unsigned int size) {
    if (family != kAfInet) {
        errno = EAFNOSUPPORT;
        return nullptr;
    }
    return ::inet_ntop(AF_INET, src, dst, size);
}
int InetPton(int family, const char* src, void* dst) {
    if (family != kAfInet) return fail(EAFNOSUPPORT);
    return ::inet_pton(AF_INET, src, dst);
}
int InetPton(Family family, const char* src, void* dst) { return InetPton(static_cast<uint8_t>(family), src, dst); }
unsigned int InetHtonl(unsigned int v) { return htonl(v); }
unsigned short InetHtons(unsigned short v) { return htons(v); }
unsigned int InetNtohl(unsigned int v) { return ntohl(v); }
unsigned short InetNtohs(unsigned short v) { return ntohs(v); }

int GetAddrInfo(const char* node, const char* service, const addrinfo* hints, addrinfo** out) {
    return make_addrinfo(node, service, reinterpret_cast<const LegacyAddrInfo*>(hints),
                         reinterpret_cast<LegacyAddrInfo**>(out));
}
int GetAddrInfo(const char* node, const char* service, const AddrInfo* hints, AddrInfo** out) {
    return make_addrinfo(node, service, hints, out);
}
void FreeAddrInfo(addrinfo* info) { free_addrinfo(reinterpret_cast<LegacyAddrInfo*>(info)); }
void FreeAddrInfo(AddrInfo* info) { free_addrinfo(info); }

HostEnt* GetHostEntByName(const char* name, const ResolverOption*, unsigned long) { return host_by_name(name); }

int ResolverGetOption(ResolverOption* out, ResolverOptionKey key) {
    return resolver_get_option(out, static_cast<uint32_t>(key));
}
int ResolverSetOption(const ResolverOption& opt) { return resolver_set_option(&opt); }

}  // namespace nn::socket

// La API en C: las mismas funciones con la firma vieja (sockaddr, addrinfo, pollfd de la consola).
extern "C" {

int nnsocketSocket(int family, int type, int protocol) { return do_socket(family, type, protocol); }
int nnsocketBind(int fd, const sockaddr* addr, unsigned int len) { return do_bind(fd, addr, len); }
int nnsocketConnect(int fd, const sockaddr* addr, unsigned int len) { return do_connect(fd, addr, len); }
int nnsocketAccept(int fd, sockaddr* addr, unsigned int* len) { return do_accept(fd, addr, len); }
int nnsocketListen(int fd, int backlog) { return ::listen(fd, backlog); }
int nnsocketShutdown(int fd, int how) { return ::shutdown(fd, how); }
int nnsocketClose(int fd) { return do_close(fd); }

int nnsocketGetSockName(int fd, sockaddr* addr, unsigned int* len) { return do_sockname(fd, addr, len, false); }
int nnsocketGetPeerName(int fd, sockaddr* addr, unsigned int* len) { return do_sockname(fd, addr, len, true); }
int nnsocketGetSockOpt(int fd, int level, int name, void* value, unsigned int* len) {
    return do_getsockopt(fd, level, name, value, len);
}
int nnsocketSetSockOpt(int fd, int level, int name, const void* value, unsigned int len) {
    return do_setsockopt(fd, level, name, value, len);
}

ssize_t nnsocketRecv(int fd, void* buf, size_t n, int flags) { return do_recvfrom(fd, buf, n, flags, nullptr, nullptr); }
ssize_t nnsocketSend(int fd, const void* buf, size_t n, int flags) { return do_sendto(fd, buf, n, flags, nullptr, 0); }
ssize_t nnsocketRecvMsg(int fd, MsgHdr* msg, int flags) { return do_recvmsg(fd, msg, flags); }
ssize_t nnsocketSendMsg(int fd, const MsgHdr* msg, int flags) { return do_sendmsg(fd, msg, flags); }

int nnsocketFcntl(int fd, int cmd, ...) {
    va_list args;
    va_start(args, cmd);
    const int arg = va_arg(args, int);
    va_end(args);
    return do_fcntl(fd, cmd, arg);
}
int nnsocketPoll(pollfd* fds, unsigned long n, int timeout) { return do_poll(fds, n, timeout); }

int nnsocketGetLastErrno() { return errno; }
void nnsocketSetLastErrno(int value) { errno = value; }

unsigned int nnsocketInetHtonl(unsigned int v) { return htonl(v); }
unsigned short nnsocketInetHtons(unsigned short v) { return htons(v); }
unsigned int nnsocketInetNtohl(unsigned int v) { return ntohl(v); }
unsigned short nnsocketInetNtohs(unsigned short v) { return ntohs(v); }

int nnsocketGetAddrInfo(const char* node, const char* service, const addrinfo* hints, addrinfo** out) {
    return nn::socket::GetAddrInfo(node, service, hints, out);
}
void nnsocketFreeAddrInfo(addrinfo* info) { nn::socket::FreeAddrInfo(info); }
const char* nnsocketGAIStrError(int code) { return gai_error(code); }
hostent* nnsocketGetHostByNameWithOptions(const char* name, const ResolverOption*, size_t) {
    return reinterpret_cast<hostent*>(host_by_name(name));
}
int nnsocketResolverGetOption(ResolverOption* out, uint32_t key) { return resolver_get_option(out, key); }
int nnsocketResolverSetOption(const ResolverOption* opt) { return resolver_set_option(opt); }

// El juego lo usa como getifaddrs de BSD (sysctl NET_RT_IFLIST, 6 niveles): sin red no hay interfaces que dar.
// ENOENT y no ENOMEM, con el que reintenta.
int nnsocketSysctl(int*, unsigned long, void*, unsigned long*, void*, unsigned long) { return fail(ENOENT); }

}  // extern "C"
