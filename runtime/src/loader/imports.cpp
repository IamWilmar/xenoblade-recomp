#include "loader/imports.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <libgen.h>
#include <sys/mman.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

extern "C" void nier_stub_dispatch();

namespace {

struct Stub {
    std::string pretty;
    uint64_t ret = 0;  // lo que devuelve al juego
    std::atomic<uint64_t> calls{0};
};

// Cada stub es un thunk de 24 bytes que carga su índice en x17 y salta a nier_stub_dispatch.
// Se generan todos al arrancar: crear un stub luego solo le pone nombre, sin tocar código.
constexpr size_t kThunkSize = 24, kMaxThunks = 8192;
constexpr uint32_t kRing = 64;
constexpr uint64_t kLoggedCalls = 3;

struct Call {
    uint32_t stub;
    uint64_t lr, x0, x1, x2;
};

Stub g_stubs[kMaxThunks];
std::atomic<uint32_t> g_stub_count{0};
std::mutex g_stub_m;
uint8_t* g_thunks = nullptr;
Call g_ring[kRing];
std::atomic<uint32_t> g_ring_pos{0};
std::unordered_map<std::string, void*> g_host;
void* g_nnsdk = nullptr;
void* g_nnsdk_base = nullptr;
struct {
    int nnsdk, host, sdk_data, zero_data, weak, stub;
} g_counts{};

uint64_t make_stub(const std::string& name, uint64_t ret) {
    std::lock_guard lock(g_stub_m);
    uint32_t idx = g_stub_count.load();
    if (idx >= kMaxThunks) {
        fprintf(stdout, "demasiados stubs\n");
        abort();
    }
    g_stubs[idx].pretty = demangle(name);
    g_stubs[idx].ret = ret;
    g_stub_count.store(idx + 1, std::memory_order_release);
    return reinterpret_cast<uint64_t>(g_thunks + idx * kThunkSize);
}

bool write_thunks(std::string& err) {
    size_t len = kThunkSize * kMaxThunks;
    void* mem = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        err = std::string("mmap de stubs: ") + strerror(errno);
        return false;
    }
    g_thunks = static_cast<uint8_t*>(mem);
    uint64_t target = reinterpret_cast<uint64_t>(&nier_stub_dispatch);
    for (uint32_t idx = 0; idx < kMaxThunks; idx++) {
        auto* t = reinterpret_cast<uint32_t*>(g_thunks + idx * kThunkSize);
        t[0] = 0xD2800011 | idx << 5;  // movz x17, #idx
        t[1] = 0x58000070;             // ldr  x16, [pc, #12]
        t[2] = 0xD61F0200;             // br   x16
        t[3] = 0xD503201F;             // nop
        memcpy(&t[4], &target, sizeof target);
    }
    if (mprotect(g_thunks, len, PROT_READ | PROT_EXEC) != 0) {
        err = std::string("mprotect de stubs: ") + strerror(errno);
        return false;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(g_thunks), reinterpret_cast<char*>(g_thunks) + len);
    return true;
}

}  // namespace

void* make_named_stub(const char* name, uint64_t ret) { return reinterpret_cast<void*>(make_stub(name, ret)); }

std::string demangle(const std::string& name) {
    int status = 0;
    char* p = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
    if (status != 0 || !p) return name;
    std::string out(p);
    free(p);
    return out;
}

bool init_imports(const std::string& nnsdk_path, std::string& err) {
    if (!write_thunks(err)) return false;
    size_t n;
    const HostSymbol* hs = host_symbols(&n);
    for (size_t k = 0; k < n; k++) g_host.emplace(hs[k].name, hs[k].addr);

    g_nnsdk = dlopen(nnsdk_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    Dl_info info;
    if (!g_nnsdk || !dladdr(dlsym(g_nnsdk, "nnsdk_attach"), &info)) {
        err = std::string("libnnsdk.so: ") + dlerror();
        return false;
    }
    g_nnsdk_base = info.dli_fbase;
    return true;
}

// Solo cuenta lo que define libnnsdk.so: dlsym con handle también busca en sus dependencias (libc).
void* nnsdk_symbol(const char* name) {
    void* p = dlsym(g_nnsdk, name);
    Dl_info info;
    return p && dladdr(p, &info) && info.dli_fbase == g_nnsdk_base ? p : nullptr;
}

uint64_t resolve_import(const ElfSym& s, const Module* sdk) {
    // Las funciones de libc con comportamiento de la Switch (pthread_* sobre nn::os, open con rutas "rom:/",
    // clock_gettime solo con CLOCK_REALTIME…) se exportan como nngame_<nombre>: con su nombre de libc, las
    // llamadas de la propia libnnsdk (y de las bibliotecas que carga) las tomarían en vez de las de bionic.
    if (void* p = nnsdk_symbol(("nngame_" + s.name).c_str())) {
        g_counts.nnsdk++;
        return reinterpret_cast<uint64_t>(p);
    }
    if (void* p = nnsdk_symbol(s.name.c_str())) {
        g_counts.nnsdk++;
        return reinterpret_cast<uint64_t>(p);
    }
    if (auto it = g_host.find(s.name); it != g_host.end()) {
        g_counts.host++;
        return reinterpret_cast<uint64_t>(it->second);
    }
    if (s.bind == STB_WEAK) {
        g_counts.weak++;
        return 0;
    }
    if (s.type == STT_OBJECT) {
        // Constantes del SDK (tablas de nn::util…): se copian tal cual del módulo sdk volcado.
        const ElfSym* src = sdk ? sdk->find_export(s.name) : nullptr;
        size_t size = src && src->size ? src->size : 4096;
        void* mem = aligned_alloc(16, (size + 15) & ~size_t{15});
        memset(mem, 0, size);
        if (src) {
            memcpy(mem, sdk->base + src->value, size);
            g_counts.sdk_data++;
        } else {
            printf("[aviso] dato %s sin origen: 4 KB a cero\n", demangle(s.name).c_str());
            g_counts.zero_data++;
        }
        return reinterpret_cast<uint64_t>(mem);
    }
    g_counts.stub++;
    return make_stub(s.name, 0);
}

void report_imports(bool list_stubs) {
    printf("imports: %d libnnsdk, %d host, %d datos del sdk, %d datos a cero, %d weak a 0, %d stubs\n",
           g_counts.nnsdk, g_counts.host, g_counts.sdk_data, g_counts.zero_data, g_counts.weak, g_counts.stub);
    if (list_stubs)
        for (uint32_t k = 0; k < g_stub_count; k++) printf("  stub: %s\n", g_stubs[k].pretty.c_str());
}

std::string describe_addr(uint64_t a) {
    char buf[512];
    for (const Module* m : registered_modules()) {
        if (m->contains(a)) {
            snprintf(buf, sizeof buf, "%s+0x%lx", m->name.c_str(), a - m->addr(0));
            return buf;
        }
    }
    auto thunks = reinterpret_cast<uint64_t>(g_thunks);
    if (g_thunks && a >= thunks && a < thunks + kThunkSize * g_stub_count)
        return "stub:" + g_stubs[(a - thunks) / kThunkSize].pretty;
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(a), &info) && info.dli_fname) {
        auto start = reinterpret_cast<uint64_t>(info.dli_saddr ? info.dli_saddr : info.dli_fbase);
        std::string sym = info.dli_sname ? demangle(info.dli_sname) : "?";
        snprintf(buf, sizeof buf, "%s(%s+0x%lx)", basename(const_cast<char*>(info.dli_fname)), sym.c_str(),
                 a - start);
        return buf;
    }
    snprintf(buf, sizeof buf, "0x%lx", a);
    return buf;
}

void dump_recent_stubs(FILE* out) {
    uint32_t end = g_ring_pos.load();
    uint32_t begin = end > kRing ? end - kRing : 0;
    fprintf(out, "Últimas %u llamadas a stubs:\n", end - begin);
    for (uint32_t k = begin; k < end; k++) {
        const Call& c = g_ring[k % kRing];
        fprintf(out, "  %s  x0=%#lx x1=%#lx x2=%#lx  desde %s\n", g_stubs[c.stub].pretty.c_str(), c.x0, c.x1,
                c.x2, describe_addr(c.lr).c_str());
    }
}

// Lo llama nier_stub_dispatch: registra la llamada y devuelve el valor del stub
// (0 para los imports: nn::Result de éxito).
extern "C" uint64_t nier_on_stub(uint64_t idx, const uint64_t* args, uint64_t lr) {
    Stub& s = g_stubs[idx];
    uint64_t n = s.calls++;
    g_ring[g_ring_pos++ % kRing] = {static_cast<uint32_t>(idx), lr, args[0], args[1], args[2]};
    if (n < kLoggedCalls)
        printf("[stub] %s  x0=%#lx x1=%#lx x2=%#lx  desde %s\n", s.pretty.c_str(), args[0], args[1], args[2],
               describe_addr(lr).c_str());
    else if (n == kLoggedCalls)
        printf("[stub] %s: más llamadas, ya no las registro\n", s.pretty.c_str());
    return s.ret;
}
