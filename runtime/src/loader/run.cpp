#include "loader/run.h"

#include <dlfcn.h>
#include <vector>
#include <sys/syscall.h>
#include <dirent.h>
#include <sys/mman.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <string>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "loader/crash.h"
#include "loader/imports.h"
#include "loader/nso.h"
#include "loader/probe.h"
#include "loader/tls.h"
#include "nnsdk/host.h"

// Después de nso.h: <link.h> define STB_WEAK y STT_OBJECT como macros.
#include <link.h>

namespace {

Module g_main, g_sdk;
// Bibliotecas del juego en su ExeFS (subsdkN) que se cargan tal cual: Catherine: Full Body trae en subsdk0 glslc, el
// compilador de shaders de NVIDIA. Las del SDK que el runtime ya sustituye (multimedia: movie::) no se cargan.
std::vector<Module*> g_subsdks;
std::atomic<bool> g_done{false};

[[noreturn]] void die(const std::string& msg) {
    fprintf(stdout, "error: %s\n", msg.c_str());
    fflush(stdout);
    exit(1);
}

// MainThreadStackSize del main.npdm (META, offset 0x1C).
uint32_t npdm_stack_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    char buf[0x20] = {};
    if (!f.read(buf, sizeof buf) || memcmp(buf, "META", 4) != 0) return 0;
    uint32_t v;
    memcpy(&v, buf + 0x1C, sizeof v);
    return v;
}

template <typename Fn>
Fn game_fn(const Module& m, const char* name) {
    const ElfSym* s = m.find_export(name);
    return s ? reinterpret_cast<Fn>(m.addr(s->value)) : nullptr;
}

// Orden de arranque del SDK: nninitStartup (monta el heap) antes de los constructores, luego nnMain.
// Al empezar cada hilo del juego (el principal y los de nn::os).
void game_thread_start() {
    install_thread_altstack();
    tls::thread_start();
}

void* game_thread(void*) {
    game_thread_start();
    using VoidFn = void (*)();
    if (auto startup = game_fn<VoidFn>(g_main, "nninitStartup")) {
        printf("→ nninitStartup\n");
        startup();
    }
    if (g_main.init) {
        // El DT_INIT del crt0 de Nintendo recorre él mismo el .init_array
        // (cuando __nnmusl_init_dso devuelve 0): no hay que correrlo otra vez.
        printf("→ DT_INIT (constructores)\n");
        reinterpret_cast<VoidFn>(g_main.addr(g_main.init))();
    } else {
        auto* ctors = reinterpret_cast<uint64_t*>(g_main.addr(g_main.init_array));
        size_t count = g_main.init_array_size / sizeof(uint64_t);
        printf("→ %zu constructores\n", count);
        for (size_t k = 0; k < count; k++)
            if (ctors[k] && ctors[k] != ~uint64_t{0}) reinterpret_cast<VoidFn>(ctors[k])();
    }
    for (Module* mod : g_subsdks) {
        printf("→ constructores de %s\n", mod->name.c_str());
        if (mod->init) {
            reinterpret_cast<VoidFn>(mod->addr(mod->init))();
        } else {
            auto* ctors = reinterpret_cast<uint64_t*>(mod->addr(mod->init_array));
            for (size_t k = 0; mod->init_array && k < mod->init_array_size / sizeof(uint64_t); k++)
                if (ctors[k] && ctors[k] != ~uint64_t{0}) reinterpret_cast<VoidFn>(ctors[k])();
        }
    }
    printf("→ nnMain\n");
    game_fn<VoidFn>(g_main, "nnMain")();
    printf("nnMain volvió\n");
    g_done = true;
    return nullptr;
}

uint32_t branch_to(const void* from, const void* to, uint32_t opcode) {
    const intptr_t delta = (reinterpret_cast<intptr_t>(to) - reinterpret_cast<intptr_t>(from)) / 4;
    return opcode | (static_cast<uint32_t>(delta) & 0x3FFFFFF);
}

// Parches de código del juego, cada uno comprobando las instrucciones originales (otra versión: no se toca nada).
// Van antes de tls::patch, con el text y el área de parches aún escribibles; sus stubs, justo antes de las sondas.
void game_patches(Module& m) {
    // Parches por dirección de Disco Elysium 1.0.9 (heredados del fork): solo con su build exacto. Castlevania no
    // necesita ninguno, así que cualquier versión del juego carga sin tocar su código.
    static const uint8_t kDisco109[16] = {0xea, 0xa1, 0xed, 0xbe, 0xae, 0xf5, 0x08, 0x54,
                                          0x5c, 0xe5, 0x1e, 0x3a, 0x39, 0xad, 0x36, 0x7d};
    if (memcmp(m.build_id, kDisco109, sizeof kDisco109) != 0) return;
    auto* text = reinterpret_cast<uint32_t*>(m.base + m.seg_off[0]);
    auto* area = reinterpret_cast<uint32_t*>(m.patch_area + Module::kPatchAreaSize - Module::kProbeAreaSize - 4096);
    // Disco Elysium 1.0.9, blit de vídeo (main+0xf44150): el material de conversión YUV de su tabla estática sale nulo y
    // Material::SetTexture("_SecondTex") cae en 0x38. Sin material, el blit dice que falló (el llamador ya lo trata: el
    // vídeo corre sin verse y la escena sigue cuando acaba). Con el runtime de hoy el material (Hidden/VideoDecode) sí se
    // crea (leído en vivo: la tabla de main+0x4d5a280 tiene material en sus pases) y el vídeo del título se ve; esto queda
    // como red por si algún vídeo llega antes de que exista.
    constexpr size_t kSite = 0xf4418c / 4, kEpilogue = 0xf441d4 / 4;
    if (m.seg_size[0] > 0xf441d8 && text[kSite] == 0xF940041A && text[kSite + 1] == 0x12800008 &&
        text[kEpilogue] == 0xA9457BFD) {
        uint32_t* s = area;
        // DEPURACIÓN: <datos>/sin_blit_video: el blit de vídeo siempre falla (el vídeo no se dibuja).
        const char* data = getenv("NIER_DATA");
        const bool never = access((std::string(data ? data : ".") + "/sin_blit_video").c_str(), F_OK) == 0;
        s[0] = never ? 0x14000004 : 0xB4000080;            // b falla / cbz x0, falla (+4 instrucciones)
        s[1] = 0xF940041A;                                  // ldr x26, [x0, #8]
        s[2] = 0xB400005A;                                  // cbz x26, falla (+2)
        s[3] = branch_to(&s[3], &text[kSite + 1], 0x14000000);  // b de vuelta
        s[4] = 0x2A1F03E0;                                  // falla: mov w0, wzr
        s[5] = branch_to(&s[5], &text[kEpilogue], 0x14000000);  // b al epílogo (devuelve false)
        text[kSite] = branch_to(&text[kSite], s, 0x14000000);
        printf("[parche] blit de vídeo sin material: sale con false (main+0xf4418c)\n");
    }
    // Disco Elysium 1.0.9, aviso perdido en el semáforo ligero con tope de la subida asíncrona (contador en
    // main+0x4faf7a0, tope en +0x10, semáforo del SO en +8). Quien avisa hace "marca = 1" con un str normal y lee el
    // contador con un ldr normal para salir si ya está en el tope; ARMv8 deja que esa lectura vaya antes que la escritura
    // (el Kryo de la Flip lo hace a menudo; el A57 de la Switch casi nunca). Si el hilo principal acaba de gastar el
    // permiso viejo y aún no ve la marca, vuelve a dormir y nadie le avisa: el arranque se quedaba en negro (~1 de cada 2).
    // Entre la escritura y la lectura va un dmb ish.
    static const uint32_t kFlagStores[] = {0x6cf748 / 4, 0x6cf970 / 4, 0x6d4fd4 / 4};
    uint32_t* stub = area + 8;
    for (uint32_t site : kFlagStores) {
        if (m.seg_size[0] <= site * 4 + 16 || text[site] != 0xB9000109 /* str w9, [x8] */ ||
            (text[site + 2] & 0xFFC003FF) != 0x91000000 + (text[site + 2] & 0x1F) * 0x21 /* add xN, xN, #imm */ ||
            ((text[site + 2] >> 10) & 0xFFF) != 0x7a0) {
            printf("[parche] semáforo de la subida asíncrona: main+%#x no es el esperado, sin tocar\n", site * 4);
            continue;
        }
        stub[0] = 0xB9000109;  // str w9, [x8]
        stub[1] = 0xD5033BBF;  // dmb ish
        stub[2] = branch_to(&stub[2], &text[site + 1], 0x14000000);
        text[site] = branch_to(&text[site], stub, 0x14000000);
        stub += 4;
    }
    // Y del lado del que espera (main+0x6d196c): tras restar del contador (ldaxr/stxr) lee la marca con ldar, que sí
    // puede adelantarse al stxr. Con barrera solo en un lado, los dos pueden leer el valor viejo (store buffering):
    // el que avisa ve el contador lleno y no avisa, y el que espera no ve la marca y se duerme.
    static const uint32_t kWaitChecks[] = {0x6d1a0c / 4, 0x6d1aac / 4};
    for (uint32_t site : kWaitChecks) {
        if (m.seg_size[0] <= site * 4 + 8 || text[site] != 0x7100011F /* cmp w8, #0 */ ||
            (text[site - 1] & 0xFF000000) != 0x35000000 /* cbnz (reintento del stxr) */) {
            printf("[parche] semáforo de la subida asíncrona: main+%#x no es el esperado, sin tocar\n", site * 4);
            continue;
        }
        stub[0] = 0xD5033BBF;  // dmb ish
        stub[1] = 0x7100011F;  // cmp w8, #0
        stub[2] = branch_to(&stub[2], &text[site + 1], 0x14000000);
        text[site] = branch_to(&text[site], stub, 0x14000000);
        stub += 4;
    }
    printf("[parche] semáforo de la subida asíncrona: dmb tras las marcas (main+0x6cf748, 0x6cf970, 0x6d4fd4) y antes de "
           "mirarlas (0x6d1a0c, 0x6d1aac)\n");
}

// nn::ro::LoadModule: los NRO de la RomFS (AmplifyTextureNative en Disco Elysium). Sus imports van primero a main
// (el plugin llama a funciones del juego y de su libc) y si no, al runtime como los de main.
void* load_nro_module(const void* image, size_t size) {
    static int count = 0;
    auto* mod = new Module;
    std::string err;
    auto resolve = [](const ElfSym& s) -> uint64_t {
        if (const ElfSym* e = g_main.find_export(s.name)) return g_main.addr(e->value);
        return resolve_import(s, g_sdk.base ? &g_sdk : nullptr);
    };
    if (!load_nro(image, size, "nro" + std::to_string(count++), *mod, err) || !relocate(*mod, resolve, err) ||
        !tls::patch(*mod, err) || !protect(*mod, err)) {
        printf("[ro] %s\n", err.c_str());
        delete mod;
        return nullptr;
    }
    register_module(mod);
    printf("[ro] %s en %p (%.1f MB), %zu símbolos\n", mod->name.c_str(), mod->base, mod->size / 1048576.0,
           mod->symbols.size());
    using VoidFn = void (*)();
    if (mod->init) {
        reinterpret_cast<VoidFn>(mod->addr(mod->init))();
    } else {
        auto* ctors = reinterpret_cast<uint64_t*>(mod->addr(mod->init_array));
        for (size_t k = 0; mod->init_array && k < mod->init_array_size / sizeof(uint64_t); k++)
            if (ctors[k] && ctors[k] != ~uint64_t{0}) reinterpret_cast<VoidFn>(ctors[k])();
    }
    return mod;
}

uint64_t nro_symbol(const void* module, const char* name) {
    const auto* mod = static_cast<const Module*>(module);
    const ElfSym* s = mod ? mod->find_export(name) : nullptr;
    return s ? mod->addr(s->value) : 0;
}

}  // namespace

// libunwind (el de nuestra libc++abi, la que lanza y captura por el juego) busca las tablas de cada pc con
// dl_iterate_phdr. main no es un ELF cargado por bionic: se le añade a la lista con su text y su .eh_frame_hdr
// (la tabla de búsqueda binaria de sus ~360k FDE), o una excepción del juego no cruzaría sus propios frames.
// Oculta: solo la ve el libunwind enlazado en este mismo binario.
extern "C" __attribute__((visibility("hidden"))) int dl_iterate_phdr(int (*cb)(dl_phdr_info*, size_t, void*),
                                                                    void* data) {
    using Fn = int (*)(int (*)(dl_phdr_info*, size_t, void*), void*);
    static const Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "dl_iterate_phdr"));
    if (int r = real(cb, data)) return r;
    if (!g_main.base || !g_main.eh_frame_hdr) return 0;
    ElfW(Phdr) ph[2] = {};
    ph[0].p_type = PT_LOAD;
    ph[0].p_vaddr = g_main.seg_off[0];
    ph[0].p_memsz = g_main.seg_size[0];
    ph[1].p_type = PT_GNU_EH_FRAME;
    ph[1].p_vaddr = g_main.eh_frame_hdr;
    ph[1].p_memsz = g_main.eh_frame_hdr_end - g_main.eh_frame_hdr;
    dl_phdr_info info{};
    info.dlpi_addr = reinterpret_cast<ElfW(Addr)>(g_main.base);
    info.dlpi_name = "main";
    info.dlpi_phdr = ph;
    info.dlpi_phnum = 2;
    return cb(&info, sizeof info, data);
}

// Carga los subsdkN del ExeFS que no sean del SDK sustituido. Sus imports van al runtime como los de main; los de main
// buscan primero en sus exports (glslcCompile y compañía).
void load_subsdks(const std::string& exefs) {
    for (int n = 0; n < 10; n++) {
        const std::string path = exefs + "/subsdk" + std::to_string(n);
        if (access(path.c_str(), R_OK) != 0) continue;
        auto* mod = new Module;
        std::string err;
        if (!load_nso(path, *mod, err)) {
            printf("[subsdk] %s\n", err.c_str());
            delete mod;
            continue;
        }
        bool replaced = false;
        for (const ElfSym& sym : mod->symbols)
            if (sym.defined() && sym.name.rfind("_ZN5movie", 0) == 0) replaced = true;
        if (replaced) {
            printf("[subsdk] %s es la biblioteca multimedia del SDK: la da el runtime\n", mod->name.c_str());
            munmap(mod->patch_area, Module::kPatchAreaSize + mod->size);
            delete mod;
            continue;
        }
        // Como el rtld de la consola: primero lo que exporta main (operator new/delete del juego), luego el runtime.
        auto resolve = [](const ElfSym& sym) -> uint64_t {
            if (const ElfSym* e = g_main.find_export(sym.name)) return g_main.addr(e->value);
            return resolve_import(sym, g_sdk.base ? &g_sdk : nullptr);
        };
        if (!relocate(*mod, resolve, err) || !tls::patch(*mod, err) || !protect(*mod, err)) die(err);
        register_module(mod);
        g_subsdks.push_back(mod);
        printf("[subsdk] %s en %p (%.1f MB), %zu símbolos\n", mod->name.c_str(), mod->base, mod->size / 1048576.0,
               mod->symbols.size());
    }
}

int nier_run(const RunConfig& cfg) {
    install_crash_handler();
    std::string err;
    if (!init_imports(cfg.nnsdk, err)) die(err);
    if (!load_nso(cfg.exefs + "/main", g_main, err)) die(err);
    bool have_sdk = load_nso(cfg.exefs + "/sdk", g_sdk, err);
    if (!have_sdk) printf("[aviso] sin sdk para las constantes: %s\n", err.c_str());
    register_module(&g_main);
    printf("main en %p (%.1f MB), %zu símbolos, %zu relocaciones\n", g_main.base, g_main.size / 1048576.0,
           g_main.symbols.size(), g_main.relocs.size());

    load_subsdks(cfg.exefs);
    auto resolve = [&](const ElfSym& s) -> uint64_t {
        for (const Module* mod : g_subsdks)
            if (const ElfSym* e = mod->find_export(s.name)) return mod->addr(e->value);
        return resolve_import(s, have_sdk ? &g_sdk : nullptr);
    };
    if (!relocate(g_main, resolve, err)) die(err);
    game_patches(g_main);
    if (!probe::install(g_main, err) || !tls::patch(g_main, err) ||
        !protect(g_main, err))
        die(err);
    report_imports(cfg.list_stubs);

    NnsdkHost host{};
    host.game_malloc = game_fn<void* (*)(size_t)>(g_main, "malloc");
    host.game_free = game_fn<void (*)(void*)>(g_main, "free");
    host.game_new = game_fn<void* (*)(size_t)>(g_main, "_Znwm");
    host.game_delete = game_fn<void (*)(void*)>(g_main, "_ZdlPv");
    host.game_aligned_alloc = game_fn<void* (*)(size_t, size_t)>(g_main, "aligned_alloc");
    host.thread_start = game_thread_start;
    host.tls_template = tls::set_template;
    host.make_stub = make_named_stub;
    host.watch_writes = watch_writes;
    host.native_window = cfg.native_window;
    host.input = cfg.input;
    host.window_state = cfg.window_state;
    host.main_base = g_main.base;
    host.main_size = g_main.size;
    host.load_nro = load_nro_module;
    host.nro_symbol = nro_symbol;
    // Unreal Engine no exporta sus operator new/delete (son internos de main): van a su mismo allocator que el
    // malloc/free que sí exporta (FMemory), así que valen esos.
    if (!host.game_new) host.game_new = host.game_malloc;
    if (!host.game_delete) host.game_delete = host.game_free;
    if (!host.game_malloc || !host.game_free) die("main no exporta malloc/free");
    reinterpret_cast<void (*)(const NnsdkHost*)>(nnsdk_symbol("nnsdk_attach"))(&host);

    uint32_t stack = std::max<uint32_t>(npdm_stack_size(cfg.exefs + "/main.npdm"), 1 << 20);
    printf("hilo principal con %u KB de pila\n", stack / 1024);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stack);
    pthread_t thread;
    if (pthread_create(&thread, &attr, game_thread, nullptr) != 0) die("pthread_create");

    // Watchdog: cada watchdog_seconds vuelca dónde está el hilo principal, sin pararlo. En la
    // segunda instantánea vuelca además todos los hilos del juego, uno detrás de otro.
    auto list_threads = reinterpret_cast<int (*)(pthread_t*, int)>(nnsdk_symbol("nnsdk_list_threads"));
    // DEPURACIÓN: con el fichero <datos>/hilos se vuelcan todos los hilos del juego en ese momento (p. ej. uno que
    // gasta un núcleo entero) y se borra el fichero.
    const char* data = getenv("NIER_DATA");
    const std::string threads_request = std::string(data ? data : ".") + "/hilos";
    for (int tenths = 1, snapshots = 0; !g_done; tenths++) {
        usleep(100'000);
        if (tenths % 10 == 0 && list_threads && access(threads_request.c_str(), F_OK) == 0) {
            unlink(threads_request.c_str());
            // Todos los hilos del proceso, no solo los del juego: también los del runtime (NvnSubmit, NvnPresent, el
            // de audio…), que son los que hay que ver cuando el juego espera a la cola y la cola no avanza.
            std::vector<pid_t> tids;
            if (DIR* dir = opendir("/proc/self/task")) {
                while (dirent* e = readdir(dir))
                    if (const pid_t tid = atoi(e->d_name); tid > 0 && tid != gettid()) tids.push_back(tid);
                closedir(dir);
            }
            for (pid_t tid : tids) {
                usleep(30'000);
                syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
            }
        }
        if (cfg.watchdog_seconds <= 0 || tenths % (cfg.watchdog_seconds * 10) != 0) continue;
        pthread_kill(thread, SIGUSR2);
        if (++snapshots != 2 || !list_threads) continue;
        pthread_t others[64];
        int n = list_threads(others, 64);
        for (int k = 0; k < n; k++) {
            usleep(50'000);
            pthread_kill(others[k], SIGUSR2);
        }
    }
    pthread_join(thread, nullptr);
    return 0;
}
