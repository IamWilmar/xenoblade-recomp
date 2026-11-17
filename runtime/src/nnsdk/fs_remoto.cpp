// RomFS remota por TCP (ver fs_remoto.h). Solo para el emulador del Mac.

#include "nnsdk/fs_remoto.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "nnsdk/watch.h"

namespace fs_remoto {
namespace {

// "tcp:10.0.2.2:7878/romfs/data/a" -> servidor y ruta "romfs/data/a".
bool partir(const std::string& ruta, std::string& host, int& puerto, std::string& resto) {
    if (!es_remota(ruta)) return false;
    const size_t dos = ruta.find(':', 4), barra = ruta.find('/', 4);
    if (dos == std::string::npos || barra == std::string::npos || dos > barra) return false;
    host = ruta.substr(4, dos - 4);
    puerto = atoi(ruta.c_str() + dos + 1);
    resto = ruta.substr(barra + 1);
    while (!resto.empty() && resto.back() == '/') resto.pop_back();
    return true;
}

struct Conexion {
    int fd = -1;
    ~Conexion() {
        if (fd >= 0) close(fd);
    }
};
thread_local Conexion t_con;

bool conecta(const std::string& host, int puerto) {
    if (t_con.fd >= 0) return true;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(puerto));
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
        static std::once_flag aviso;
        std::call_once(aviso, [&] {
            printf("[fs remota] no hay servidor en %s:%d: %s (¿tools/romfs_remota.py?)\n", host.c_str(), puerto,
                   strerror(errno));
        });
        close(fd);
        return false;
    }
    int uno = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &uno, sizeof uno);
    int buf = 4 << 20;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
    t_con.fd = fd;
    return true;
}

void cierra() {
    if (t_con.fd >= 0) close(t_con.fd);
    t_con.fd = -1;
}

bool envia_todo(const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    while (n) {
        ssize_t k = send(t_con.fd, c, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        c += k;
        n -= k;
    }
    return true;
}

bool recibe_todo(void* p, size_t n) {
    char* c = static_cast<char*>(p);
    while (n) {
        ssize_t k = recv(t_con.fd, c, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        c += k;
        n -= k;
    }
    return true;
}

bool pide(char op, const std::string& host, int puerto, const std::string& resto, int64_t offset, int64_t tamano) {
    std::vector<char> m(1 + 2 + resto.size() + 16);
    m[0] = op;
    const uint16_t largo = static_cast<uint16_t>(resto.size());
    memcpy(&m[1], &largo, 2);
    memcpy(&m[3], resto.data(), resto.size());
    memcpy(&m[3 + resto.size()], &offset, 8);
    memcpy(&m[11 + resto.size()], &tamano, 8);
    // Un reintento con conexión nueva: el servidor pudo reiniciarse entre dos lecturas.
    for (int intento = 0; intento < 2; intento++) {
        if (!conecta(host, puerto)) return false;
        if (envia_todo(m.data(), m.size())) return true;
        cierra();
    }
    return false;
}

}  // namespace

int stat(const std::string& ruta, int64_t* tamano) {
    std::string host, resto;
    int puerto;
    if (!partir(ruta, host, puerto, resto) || !pide('S', host, puerto, resto, 0, 0)) return -1;
    int8_t tipo;
    int64_t t;
    if (!recibe_todo(&tipo, 1) || !recibe_todo(&t, 8)) {
        cierra();
        return -1;
    }
    if (tamano) *tamano = t;
    return tipo;
}

bool lista(const std::string& ruta, std::vector<Entrada>& out) {
    std::string host, resto;
    int puerto;
    if (!partir(ruta, host, puerto, resto) || !pide('L', host, puerto, resto, 0, 0)) return false;
    int32_t n;
    if (!recibe_todo(&n, 4)) {
        cierra();
        return false;
    }
    if (n < 0) return false;
    for (int32_t k = 0; k < n; k++) {
        int8_t tipo;
        int64_t t;
        uint16_t largo;
        if (!recibe_todo(&tipo, 1) || !recibe_todo(&t, 8) || !recibe_todo(&largo, 2)) {
            cierra();
            return false;
        }
        std::string nombre(largo, '\0');
        if (largo && !recibe_todo(nombre.data(), largo)) {
            cierra();
            return false;
        }
        out.push_back({nombre, tipo == 0, t});
    }
    return true;
}

int64_t lee(const std::string& ruta, int64_t offset, void* buffer, size_t tamano) {
    std::string host, resto;
    int puerto;
    if (!partir(ruta, host, puerto, resto) || !pide('R', host, puerto, resto, offset, static_cast<int64_t>(tamano)))
        return -1;
    int64_t n;
    if (!recibe_todo(&n, 8)) {
        cierra();
        return -1;
    }
    if (n < 0) return -1;
    nnsdk::watch::before_kernel_write(buffer, static_cast<size_t>(n));  // recv en memoria vigilada daría EFAULT
    if (n && !recibe_todo(buffer, static_cast<size_t>(n))) {
        cierra();
        return -1;
    }
    return n;
}

std::string copia_local(const std::string& ruta) {
    std::string host, resto;
    int puerto;
    if (!partir(ruta, host, puerto, resto)) return {};
    int64_t tamano = 0;
    if (stat(ruta, &tamano) != 1) return {};
    const char* env = getenv("NIER_DATA");
    const std::string base = std::string(env ? env : "/sdcard/NieR") + "/rom_local";
    const std::string dst = base + "/" + resto;
    struct ::stat st;
    if (::stat(dst.c_str(), &st) == 0 && st.st_size == tamano) return dst;
    for (size_t barra = dst.find('/', 1); barra != std::string::npos; barra = dst.find('/', barra + 1))
        mkdir(dst.substr(0, barra).c_str(), 0775);
    FILE* f = fopen((dst + ".tmp").c_str(), "wb");
    if (!f) return {};
    std::vector<char> buf(4 << 20);
    int64_t hecho = 0;
    while (hecho < tamano) {
        const int64_t n = lee(ruta, hecho, buf.data(), std::min<int64_t>(buf.size(), tamano - hecho));
        if (n <= 0 || fwrite(buf.data(), 1, n, f) != static_cast<size_t>(n)) break;
        hecho += n;
    }
    fclose(f);
    if (hecho != tamano || rename((dst + ".tmp").c_str(), dst.c_str()) != 0) {
        unlink((dst + ".tmp").c_str());
        return {};
    }
    printf("[fs remota] copia local de %s (%lld bytes)\n", resto.c_str(), static_cast<long long>(tamano));
    return dst;
}

}  // namespace fs_remoto
