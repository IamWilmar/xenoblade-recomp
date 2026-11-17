// nn::fs sobre carpetas de la Flip. Cada punto de montaje del juego ("rom:", "save:"…) se
// traduce a una carpeta bajo NIER_DATA (por defecto /sdcard/NieR):
//   rom:   -> romfs/      (RomFS del juego con el update aplicado; solo lectura)
//   aoc:   -> aoc/<n>/    (DLC; solo lectura)
// salvo que <datos>/rutas.txt diga otra cosa para rom: y aoc: (rutas.h).
//   save:  -> save/       (partidas; save_prueba/ si existe al abrir el juego: ver save_dir)
//   cache: -> cache/      (cache storage de la aplicación)
//
// La partida guardada de la consola es transaccional (nada cambia hasta CommitSaveData); aquí se escribe en los
// archivos de verdad, y un cierre a mitad de un guardado dejaría el archivo a medias. Por eso, antes de tocar
// por primera vez un archivo de la partida desde el último commit, se copia tal como estaba a save.bak/: ahí
// queda siempre la última versión confirmada de lo que se esté modificando (se restaura a mano).

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "nnsdk/fs_remoto.h"
#include "nnsdk/rutas.h"
#include "nnsdk/watch.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/types.h"

namespace nn {
struct ApplicationId {
    uint64_t value;
};
namespace account {
struct Uid {
    uint64_t data[2];
};
}  // namespace account

namespace fs {

struct FileHandle {
    void* handle;
};
struct DirectoryHandle {
    void* handle;
};
struct DirectoryEntry {
    char name[769];
    char reserved0[3];
    int8_t type;
    char reserved1[3];
    int64_t size;
};
struct WriteOption {
    int flags;
};
struct ReadOption {
    int flags;
};
// El juego monta rom: con un FsContext propio (main+0x236eb88) para recibir el error en vez de abortar: un
// puntero a su función de decisión. Aquí ningún error aborta; solo se guarda cuál está puesto en cada hilo.
struct FsContext {
    void* handler;
};
enum DirectoryEntryType { DirectoryEntryType_Directory = 0, DirectoryEntryType_File = 1 };

}  // namespace fs
}  // namespace nn

using namespace nn;
using namespace nn::fs;

namespace {

// Códigos de nn::fs (módulo 2): PathNotFound, PathAlreadyExists, DirectoryNotEmpty, TargetNotFound,
// WriteNotPermitted (escribir en un archivo sin modo escritura o en un montaje de solo lectura).
constexpr Result kPathNotFound{0x202}, kPathAlreadyExists{0x402}, kDirectoryNotEmpty{0x1002},
    kTargetNotFound{0x7D402}, kWriteNotPermitted{2 | 6203 << 9};
constexpr int kOpenWrite = 2, kOpenAppend = 4;
constexpr int kDirOpenDirs = 1, kDirOpenFiles = 2;
constexpr int kWriteFlush = 1;
// Xenoblade Chronicles DE (0100FF500E34A000) y el tamaño de su partida según su control.nacp: 12 MB + 12 MB de journal.
constexpr uint64_t kProgramId = 0x0100FF500E34A000;
constexpr int64_t kSaveSize = 0xC14000, kSaveJournalSize = 0xC14000;

struct Mount {
    std::string dir;
    bool read_only = false;
    bool journaled = false;  // la partida: copia de lo confirmado antes de cambiarlo
};

std::mutex g_mounts_m;
std::map<std::string, Mount> g_mounts;
int g_logged_opens = 0;

std::mutex g_save_m;
int64_t g_save_size = kSaveSize, g_save_journal = kSaveJournalSize;  // lo que haya pedido ExtendSaveData
std::set<std::string> g_backed_up;  // archivos de la partida ya copiados en este ciclo de commit

thread_local const FsContext* t_context = nullptr;
const FsContext g_default_context{};

struct File {
    int fd;
    int mode;
    std::string path;  // DEPURACIÓN: para traza_fs
    std::string remoto;  // RomFS remota (emulador): ruta "tcp:..." y su tamaño; fd = -1
    int64_t size = 0;
};
struct Directory {
    std::vector<DirectoryEntry> entries;
    size_t next = 0;
};

std::string data_dir() {
    const char* env = getenv("NIER_DATA");
    return env ? env : "/sdcard/NieR";
}

Result from_errno() {
    if (errno == ENOENT || errno == ENOTDIR) return kPathNotFound;
    if (errno == EEXIST) return kPathAlreadyExists;
    if (errno == ENOTEMPTY) return kDirectoryNotEmpty;
    return kTargetNotFound;
}

void mount(const char* name, const std::string& dir, bool read_only = false, bool journaled = false) {
    if (!fs_remoto::es_remota(dir)) mkdir(dir.c_str(), 0775);
    std::lock_guard lock(g_mounts_m);
    g_mounts[name] = Mount{dir, read_only, journaled};
    printf("[fs] %s: -> %s%s\n", name, dir.c_str(), read_only ? " (solo lectura)" : "");
}

// "rom:/data/a.dat" -> "<carpeta de rom>/data/a.dat" y su punto de montaje. Vacío si el punto de montaje no
// existe. Un nombre sin ':' es el punto de montaje entero ("save" en CommitSaveData).
std::string host_path(const char* path, Mount* out = nullptr) {
    const char* colon = strchr(path, ':');
    std::string name = colon ? std::string(path, colon - path) : path;
    std::lock_guard lock(g_mounts_m);
    auto it = g_mounts.find(name);
    if (it == g_mounts.end()) return {};
    if (out) *out = it->second;
    const char* rest = colon ? colon + 1 : "";
    while (*rest == '/') rest++;
    return it->second.dir + "/" + rest;
}

// La carpeta de la partida. DEPURACIÓN: si al abrir el juego existe <datos>/save_prueba, se usa esa (y save_prueba.bak) y
// la del jugador no se toca: las pruebas en la Flip cargan ahí una copia de su partida y el autoguardado del juego va a
// esa copia. Se mira una vez por proceso.
std::string save_dir() {
    static const std::string dir = [] {
        const std::string test = data_dir() + "/save_prueba";
        struct stat st;
        const bool use_test = stat(test.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
        if (use_test) printf("[fs] PARTIDA DE PRUEBA: %s (la del jugador, en save/, no se toca)\n", test.c_str());
        return use_test ? test : data_dir() + "/save";
    }();
    return dir;
}
std::string backup_dir() { return save_dir() + ".bak"; }

// Copia a save.bak/ la versión confirmada de un archivo de la partida antes de que el juego lo cambie (abrirlo
// para escribir o borrarlo; RenameFile no pisa destinos). Una vez por archivo y ciclo de commit. Si el juego se cerró
// a mitad de un ciclo (queda la marca .pendiente), las copias que ya hay son las buenas: no se pisan con el
// archivo quizá a medias. La copia se escribe aparte y se renombra: nunca queda una copia a medias.
void backup(const Mount& m, const std::string& host) {
    if (!m.journaled) return;
    std::lock_guard lock(g_save_m);
    if (!g_backed_up.insert(host).second) return;
    struct stat st;
    if (stat(host.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return;
    std::string rel = host.substr(m.dir.size() + 1), bak = backup_dir();
    mkdir(bak.c_str(), 0775);
    std::string marker = bak + "/.pendiente", dst = bak + "/" + rel;
    if (access(marker.c_str(), F_OK) == 0 && access(dst.c_str(), F_OK) == 0) return;
    for (size_t slash = rel.find('/'); slash != std::string::npos; slash = rel.find('/', slash + 1))
        mkdir((bak + "/" + rel.substr(0, slash)).c_str(), 0775);
    int in = open(host.c_str(), O_RDONLY | O_CLOEXEC);
    int out = open((dst + ".tmp").c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0664);
    bool ok = in >= 0 && out >= 0;
    std::vector<char> buf(1 << 20);
    for (ssize_t n; ok && (n = read(in, buf.data(), buf.size())) != 0;)
        ok = n > 0 && write(out, buf.data(), n) == n;
    ok = ok && fsync(out) == 0;
    if (in >= 0) close(in);
    if (out >= 0) close(out);
    if (ok && rename((dst + ".tmp").c_str(), dst.c_str()) == 0) {
        close(open(marker.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0664));
        printf("[fs] copia de seguridad de %s (%lld bytes)\n", rel.c_str(), static_cast<long long>(st.st_size));
    } else {
        printf("[fs] no se pudo copiar %s a save.bak: %s\n", rel.c_str(), strerror(errno));
        unlink((dst + ".tmp").c_str());
    }
}

void log_open(const char* path, bool ok) {
    if (!ok)
        printf("[fs] no existe: %s\n", path);
    else if (g_logged_opens++ < 40 || strstr(path, "Amplify") || strstr(path, "vtc2"))
        printf("[fs] abre %s\n", path);
}

}  // namespace

namespace nn::fs {

void SetAllocator(void* (*)(size_t), void (*)(void*, size_t)) {}

Result QueryMountRomCacheSize(size_t* out) {
    *out = 0;
    return ResultSuccess;
}
Result QueryMountAddOnContentCacheSize(size_t* out, int) {
    *out = 0;
    return ResultSuccess;
}

Result MountRom(const char* name, void*, size_t) {
    mount(name, rutas::get().romfs, true);
    return ResultSuccess;
}
Result MountRom(const char* name) { return MountRom(name, nullptr, 0); }
Result MountAddOnContent(const char* name, int index, void*, size_t) {
    const auto& aoc = rutas::get().aoc;
    if (const auto it = aoc.find(index); it != aoc.end()) {
        mount(name, it->second, true);
    } else {
        mkdir((data_dir() + "/aoc").c_str(), 0775);
        mount(name, data_dir() + "/aoc/" + std::to_string(index), true);
    }
    return ResultSuccess;
}
// Cache storage y temporary storage: existen solo si el control.nacp del juego los declara, y el de Final Fantasy
// Resonance no declara ninguno (CacheStorageSize, CacheStorageDataAndJournalSizeMax y TemporaryStorageSize a 0). En la
// consola montarlos o crearlos falla y Unreal sigue sin ellos (su arranque los prueba: main+0x6340 "cache",
// main+0x6450 "temp", y la lista por índices en main+0x6520): aquí igual, para que el motor vaya por el mismo camino
// que en la consola.
struct CacheStorageListHandle {
    void* handle;
};
struct CacheStorageInfo {
    int32_t index;
};
Result MountCacheStorage(const char* name) {
    printf("[fs] MountCacheStorage(%s): el juego no declara cache storage\n", name);
    return kTargetNotFound;
}
Result MountCacheStorage(const char* name, int index) {
    printf("[fs] MountCacheStorage(%s, %d): el juego no declara cache storage\n", name, index);
    return kTargetNotFound;
}
Result MountTemporaryStorage(const char* name) {
    printf("[fs] MountTemporaryStorage(%s): el juego no declara temporary storage\n", name);
    return kTargetNotFound;
}
Result CreateCacheStorage(int index, int64_t size, int64_t journal) {
    printf("[fs] CreateCacheStorage(%d, %lld, %lld): el juego no declara cache storage\n", index,
           static_cast<long long>(size), static_cast<long long>(journal));
    return kTargetNotFound;
}
Result DeleteCacheStorage(int) { return kTargetNotFound; }
Result GetCacheStorageSize(int64_t* size, int64_t* journal, int) {
    if (size) *size = 0;
    if (journal) *journal = 0;
    return kTargetNotFound;
}
// La lista de los que hay: ninguno.
Result OpenCacheStorageList(CacheStorageListHandle* out) {
    static char list;
    out->handle = &list;
    return ResultSuccess;
}
Result ReadCacheStorageList(int* count, CacheStorageInfo*, CacheStorageListHandle, int) {
    *count = 0;
    return ResultSuccess;
}
void CloseCacheStorageList(CacheStorageListHandle) {}

// RAM disk (Unity 2022 monta uno para temporales). Aquí es una carpeta que se vacía al montar, como un RAM disk
// recién creado. El buffer no se usa: se pide poco para no gastarle memoria al juego.
size_t GetRamDiskBufferSize(size_t size) {
    printf("[fs] GetRamDiskBufferSize(%zu KB): 64 KB (va a una carpeta)\n", size / 1024);
    return 64 * 1024;
}
Result MountRamDisk(const char* name, void*, size_t) {
    const std::string dir = data_dir() + "/ramdisk";
    mkdir(dir.c_str(), 0775);
    const std::string sub = dir + "/" + name;
    std::function<void(const std::string&)> vaciar = [&](const std::string& d) {
        if (DIR* dp = opendir(d.c_str())) {
            while (dirent* e = readdir(dp)) {
                if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
                const std::string p = d + "/" + e->d_name;
                if (e->d_type == DT_DIR) {
                    vaciar(p);
                    rmdir(p.c_str());
                } else {
                    unlink(p.c_str());
                }
            }
            closedir(dp);
        }
    };
    vaciar(sub);
    mount(name, sub);
    return ResultSuccess;
}

// La partida es una carpeta: existe en cuanto EnsureSaveData la crea (MH Rise mira IsSaveDataExisting y, si no
// existe, la crea antes de montarla: main+0x796b1a4).
Result EnsureSaveData(const account::Uid&) {
    mkdir(data_dir().c_str(), 0775);
    mkdir(save_dir().c_str(), 0775);
    return ResultSuccess;
}
bool IsSaveDataExisting(const account::Uid&) {
    struct stat st;
    return stat(save_dir().c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
// MH Rise busca también las partidas de otros juegos suyos (bonificaciones): aquí no hay ninguna.
bool IsSaveDataExisting(ApplicationId app, const account::Uid& uid) {
    return app.value == kProgramId && IsSaveDataExisting(uid);
}
Result MountSaveData(const char* name, const account::Uid& uid) {
    EnsureSaveData(uid);
    mount(name, save_dir(), false, true);
    return ResultSuccess;
}
Result MountSaveDataReadOnly(const char* name, ApplicationId app, const account::Uid& uid) {
    if (app.value != kProgramId || !IsSaveDataExisting(uid)) return kTargetNotFound;
    mount(name, save_dir(), true);
    return ResultSuccess;
}

// El tamaño de la partida solo crece (el juego lo amplía si lo que va a escribir no cabe: main+0x796a454).
Result GetSaveDataSize(int64_t* size, int64_t* journal, const account::Uid&) {
    std::lock_guard lock(g_save_m);
    *size = g_save_size;
    *journal = g_save_journal;
    return ResultSuccess;
}
// SDK 12 (Xenoblade 3): el máximo al que se puede ampliar. Su control.nacp no declara máximos (0): el tamaño de
// siempre.
Result GetSaveDataSizeMax(int64_t* size, int64_t* journal) {
    std::lock_guard lock(g_save_m);
    *size = std::max(g_save_size, kSaveSize);
    *journal = std::max(g_save_journal, kSaveJournalSize);
    return ResultSuccess;
}
Result ExtendSaveData(const account::Uid&, int64_t size, int64_t journal) {
    std::lock_guard lock(g_save_m);
    printf("[fs] amplía la partida a %lld + %lld bytes\n", static_cast<long long>(size),
           static_cast<long long>(journal));
    g_save_size = std::max(g_save_size, size);
    g_save_journal = std::max(g_save_journal, journal);
    return ResultSuccess;
}

// El espacio libre real de la tarjeta; en la partida, como mucho su tamaño (el juego calcula lo ocupado como
// tamaño - libre).
Result GetFreeSpaceSize(int64_t* out, const char* path) {
    Mount m;
    std::string p = host_path(path, &m);
    struct statvfs st;
    if (p.empty() || statvfs(p.c_str(), &st) != 0) return kPathNotFound;
    *out = static_cast<int64_t>(st.f_bavail) * st.f_frsize;
    if (m.journaled) {
        std::lock_guard lock(g_save_m);
        *out = std::min(*out, g_save_size);
    }
    return ResultSuccess;
}

// Confirma lo escrito: a disco, y el próximo cambio vuelve a copiar la versión (nueva) confirmada.
Result Commit(const char* name) {
    printf("[fs] commit %s\n", name);
    sync();
    Mount m;
    if (!host_path(name, &m).empty() && m.journaled) {
        std::lock_guard lock(g_save_m);
        g_backed_up.clear();
        unlink((backup_dir() + "/.pendiente").c_str());
    }
    return ResultSuccess;
}
Result CommitSaveData(const char* name) { return Commit(name); }
void Unmount(const char* name) {
    std::lock_guard lock(g_mounts_m);
    g_mounts.erase(name);
}

const FsContext* GetCurrentThreadFsContext() { return t_context ? t_context : &g_default_context; }
void SetCurrentThreadFsContext(const FsContext* context) { t_context = context; }

Result OpenFile(FileHandle* out, const char* path, int mode) {
    Mount m;
    std::string p = host_path(path, &m);
    bool writing = mode & (kOpenWrite | kOpenAppend);
    if (writing && m.read_only) return kWriteNotPermitted;
    if (fs_remoto::es_remota(p)) {
        int64_t size = 0;
        const bool ok = fs_remoto::stat(p, &size) == 1;
        log_open(path, ok);
        if (!ok) return kPathNotFound;
        out->handle = new File{-1, mode, path, p, size};
        return ResultSuccess;
    }
    if (writing && !p.empty()) backup(m, p);
    int fd = p.empty() ? -1 : open(p.c_str(), (writing ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    log_open(path, fd >= 0);
    if (fd < 0) return p.empty() ? kPathNotFound : from_errno();
    out->handle = new File{fd, mode, path};
    return ResultSuccess;
}
void CloseFile(FileHandle h) {
    auto* f = static_cast<File*>(h.handle);
    if (f->fd >= 0) close(f->fd);
    delete f;
}
// Los pak de rom: se leen a trozos al azar: pread directo, repetido si el kernel devuelve menos (señal o FUSE).
Result ReadFile(size_t* out, FileHandle h, int64_t offset, void* buffer, size_t size) {
    timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t done = 0;
    if (const auto* f = static_cast<File*>(h.handle); !f->remoto.empty()) {
        const int64_t n = fs_remoto::lee(f->remoto, offset, buffer, size);
        if (n < 0) {
            static int logged = 0;
            if (nvn::first_times(logged, 16))
                printf("[fs] ReadFile remoto de %zu bytes en %s @%#llx falló\n", size, f->path.c_str(),
                       static_cast<unsigned long long>(offset));
            return kTargetNotFound;
        }
        done = static_cast<size_t>(n);
        size = done;  // salta el bucle de pread
    }
    int fd = static_cast<File*>(h.handle)->fd;
    if (fd >= 0) nnsdk::watch::before_kernel_write(buffer, size);  // pread en memoria vigilada daría EFAULT
    while (fd >= 0 && done < size) {
        ssize_t n = pread(fd, static_cast<char*>(buffer) + done, size - done, offset + done);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {
            static int logged = 0;
            if (nvn::first_times(logged, 16))
                printf("[fs] ReadFile: pread de %zu bytes en %p falló: %s\n", size - done, static_cast<char*>(buffer) + done,
                       strerror(errno));
            return from_errno();
        }
        if (n == 0) break;  // fin del archivo
        done += n;
    }
    if (out) *out = done;
    // DEPURACIÓN: <datos>/traza_fs registra las primeras lecturas (ruta, offset, tamaño, leído y un hash del
    // contenido) para compararlas con el archivo de verdad en el Mac.
    static std::atomic<int> traced{0};
    if (NVN_FLAG("traza_fs") && nvn::first_times(traced, 3000)) {
        uint64_t hash = 1469598103934665603ull;
        for (size_t k = 0; k < done; k++) hash = (hash ^ static_cast<uint8_t*>(buffer)[k]) * 1099511628211ull;
        timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const long us = (t1.tv_sec - t0.tv_sec) * 1'000'000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
        printf("[fs] lee %s @%#llx +%#zx -> %#zx hash %016llx en %ld us\n", static_cast<File*>(h.handle)->path.c_str(),
               static_cast<unsigned long long>(offset), size, done, static_cast<unsigned long long>(hash), us);
    }
    return ResultSuccess;
}
Result ReadFile(FileHandle h, int64_t offset, void* buffer, size_t size) {
    return ReadFile(nullptr, h, offset, buffer, size);
}
Result WriteFile(FileHandle h, int64_t offset, const void* buffer, size_t size, const WriteOption& option) {
    auto* f = static_cast<File*>(h.handle);
    if (!(f->mode & (kOpenWrite | kOpenAppend))) return kWriteNotPermitted;
    size_t done = 0;
    while (done < size) {
        ssize_t n = pwrite(f->fd, static_cast<const char*>(buffer) + done, size - done, offset + done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return from_errno();
        done += n;
    }
    if (option.flags & kWriteFlush) fsync(f->fd);
    return ResultSuccess;
}
Result FlushFile(FileHandle h) {
    fsync(static_cast<File*>(h.handle)->fd);
    return ResultSuccess;
}
Result GetFileSize(int64_t* out, FileHandle h) {
    if (const auto* f = static_cast<File*>(h.handle); !f->remoto.empty()) {
        *out = f->size;
        return ResultSuccess;
    }
    struct stat st;
    if (fstat(static_cast<File*>(h.handle)->fd, &st) != 0) return from_errno();
    *out = st.st_size;
    return ResultSuccess;
}
Result SetFileSize(FileHandle h, int64_t size) {
    auto* f = static_cast<File*>(h.handle);
    if (!(f->mode & (kOpenWrite | kOpenAppend))) return kWriteNotPermitted;
    return ftruncate(f->fd, size) == 0 ? ResultSuccess : from_errno();
}

Result CreateFile(const char* path, int64_t size) {
    Mount m;
    std::string p = host_path(path, &m);
    if (m.read_only) return kWriteNotPermitted;
    printf("[fs] crea %s (%lld bytes)\n", path, static_cast<long long>(size));
    int fd = p.empty() ? -1 : open(p.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0664);
    if (fd < 0) return p.empty() ? kPathNotFound : from_errno();
    int ok = ftruncate(fd, size);
    close(fd);
    return ok == 0 ? ResultSuccess : from_errno();
}
Result DeleteFile(const char* path) {
    Mount m;
    std::string p = host_path(path, &m);
    if (m.read_only) return kWriteNotPermitted;
    if (p.empty()) return kPathNotFound;
    printf("[fs] borra %s\n", path);
    backup(m, p);
    return unlink(p.c_str()) == 0 ? ResultSuccess : from_errno();
}
Result RenameFile(const char* from, const char* to) {
    Mount m;
    std::string a = host_path(from, &m), b = host_path(to);
    if (m.read_only) return kWriteNotPermitted;
    if (a.empty() || b.empty()) return kPathNotFound;
    // Nintendo no pisa un destino que exista (rename sí): PathAlreadyExists.
    if (access(b.c_str(), F_OK) == 0) return kPathAlreadyExists;
    return rename(a.c_str(), b.c_str()) == 0 ? ResultSuccess : from_errno();
}
Result GetEntryType(DirectoryEntryType* out, const char* path) {
    std::string p = host_path(path);
    if (fs_remoto::es_remota(p)) {
        const int tipo = fs_remoto::stat(p, nullptr);
        if (tipo < 0) return kPathNotFound;
        *out = tipo == 0 ? DirectoryEntryType_Directory : DirectoryEntryType_File;
        return ResultSuccess;
    }
    struct stat st;
    if (p.empty() || stat(p.c_str(), &st) != 0) return kPathNotFound;
    *out = S_ISDIR(st.st_mode) ? DirectoryEntryType_Directory : DirectoryEntryType_File;
    return ResultSuccess;
}

Result CreateDirectory(const char* path) {
    Mount m;
    std::string p = host_path(path, &m);
    if (m.read_only) return kWriteNotPermitted;
    if (p.empty()) return kPathNotFound;
    return mkdir(p.c_str(), 0775) == 0 ? ResultSuccess : from_errno();
}
// Solo una carpeta vacía, como en la consola (lo recursivo es otra función, que el juego no usa).
Result DeleteDirectory(const char* path) {
    Mount m;
    std::string p = host_path(path, &m);
    if (m.read_only) return kWriteNotPermitted;
    if (p.empty()) return kPathNotFound;
    return rmdir(p.c_str()) == 0 ? ResultSuccess : from_errno();
}

Result OpenDirectory(DirectoryHandle* out, const char* path, int mode) {
    std::string p = host_path(path);
    if (fs_remoto::es_remota(p)) {
        std::vector<fs_remoto::Entrada> lista;
        if (!fs_remoto::lista(p, lista)) {
            printf("[fs] no existe la carpeta: %s\n", path);
            return kPathNotFound;
        }
        auto* dir = new Directory;
        for (const auto& e : lista) {
            if (!(mode & (e.carpeta ? kDirOpenDirs : kDirOpenFiles))) continue;
            DirectoryEntry entry{};
            snprintf(entry.name, sizeof entry.name, "%s", e.nombre.c_str());
            entry.type = e.carpeta ? DirectoryEntryType_Directory : DirectoryEntryType_File;
            entry.size = e.carpeta ? 0 : e.tamano;
            dir->entries.push_back(entry);
        }
        out->handle = dir;
        return ResultSuccess;
    }
    DIR* d = p.empty() ? nullptr : opendir(p.c_str());
    if (!d) {
        printf("[fs] no existe la carpeta: %s\n", path);
        return kPathNotFound;
    }
    auto* dir = new Directory;
    while (dirent* e = readdir(d)) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (stat((p + "/" + e->d_name).c_str(), &st) != 0) continue;
        bool is_dir = S_ISDIR(st.st_mode);
        if (!(mode & (is_dir ? kDirOpenDirs : kDirOpenFiles))) continue;
        DirectoryEntry entry{};
        snprintf(entry.name, sizeof entry.name, "%s", e->d_name);
        entry.type = is_dir ? DirectoryEntryType_Directory : DirectoryEntryType_File;
        entry.size = is_dir ? 0 : st.st_size;
        dir->entries.push_back(entry);
    }
    closedir(d);
    out->handle = dir;
    return ResultSuccess;
}
void CloseDirectory(DirectoryHandle h) { delete static_cast<Directory*>(h.handle); }
Result GetDirectoryEntryCount(int64_t* out, DirectoryHandle h) {
    *out = static_cast<Directory*>(h.handle)->entries.size();
    return ResultSuccess;
}
Result ReadDirectory(int64_t* out, DirectoryEntry* entries, DirectoryHandle h, int64_t max) {
    auto* dir = static_cast<Directory*>(h.handle);
    int64_t n = 0;
    while (n < max && dir->next < dir->entries.size()) entries[n++] = dir->entries[dir->next++];
    *out = n;
    return ResultSuccess;
}


// --- Lo que añade Disco Elysium (Unity) ---

bool IsMounted(const char* name) {
    std::lock_guard lock(g_mounts_m);
    return name && g_mounts.count(name);
}
Result EnableGlobalFileDataCache(void*, size_t) { return ResultSuccess; }
void DisableGlobalFileDataCache() {}
Result ReadFile(FileHandle h, int64_t offset, void* buffer, size_t size, const ReadOption&) {
    return ReadFile(h, offset, buffer, size);
}
Result ReadFile(size_t* out, FileHandle h, int64_t offset, void* buffer, size_t size, const ReadOption&) {
    return ReadFile(out, h, offset, buffer, size);
}
int GetFileOpenMode(FileHandle h) { return static_cast<File*>(h.handle)->mode; }
Result Commit(const char* const* names, int count) {
    for (int k = 0; k < count; k++)
        if (Result r = Commit(names[k]); r.value) return r;
    return ResultSuccess;
}

namespace {

// Borra lo que hay dentro de dir (y dir también si also_dir). Cada archivo pasa antes por backup, como DeleteFile.
Result remove_tree(const char* path, bool also_dir) {
    Mount m;
    const std::string root = host_path(path, &m);
    if (m.read_only) return kWriteNotPermitted;
    if (root.empty()) return kPathNotFound;
    printf("[fs] borra el árbol %s%s\n", path, also_dir ? "" : " (solo el contenido)");
    std::function<bool(const std::string&)> clean = [&](const std::string& dir) {
        DIR* d = opendir(dir.c_str());
        if (!d) return false;
        while (const dirent* e = readdir(d)) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            const std::string p = dir + "/" + e->d_name;
            struct stat st;
            if (lstat(p.c_str(), &st) != 0) continue;
            if (S_ISDIR(st.st_mode)) {
                clean(p);
                rmdir(p.c_str());
            } else {
                backup(m, p);
                unlink(p.c_str());
            }
        }
        closedir(d);
        return true;
    };
    if (!clean(root)) return from_errno();
    if (also_dir && rmdir(root.c_str()) != 0) return from_errno();
    return ResultSuccess;
}

}  // namespace

Result DeleteDirectoryRecursively(const char* path) { return remove_tree(path, true); }
Result CleanDirectoryRecursively(const char* path) { return remove_tree(path, false); }
Result RenameDirectory(const char* from, const char* to) {
    Mount m;
    std::string a = host_path(from, &m), b = host_path(to);
    if (m.read_only) return kWriteNotPermitted;
    if (a.empty() || b.empty()) return kPathNotFound;
    if (access(b.c_str(), F_OK) == 0) return kPathAlreadyExists;
    return rename(a.c_str(), b.c_str()) == 0 ? ResultSuccess : from_errno();
}

}  // namespace nn::fs

namespace nnsdk {

// Para libc_file.cpp: la libc del juego también abre rutas "montaje:/..." (librerías de terceros).
// Con la RomFS remota (emulador) devuelve una copia local del archivo: quien llama necesita un descriptor de verdad.
std::string fs_host_path(const char* path) {
    std::string p = host_path(path);
    return fs_remoto::es_remota(p) ? fs_remoto::copia_local(p) : p;
}

}  // namespace nnsdk
