#include "loader/tls.h"

#include <pthread.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

extern "C" void nier_tls_slow();

namespace tls {
namespace {

constexpr int kSlot = 2;          // TLS_SLOT_APP de bionic
constexpr size_t kTcbSize = 16;   // musl en AArch64: el bloque del módulo va detrás de 16 bytes

struct Template {
    const uint8_t* image = nullptr;
    size_t image_size = 0, total_size = 0, align = 16, offset = kTcbSize;
    bool known = false;
};

std::mutex g_m;
Template g_tpl;
std::vector<uint8_t*> g_blocks;  // bloques vivos, para darles la plantilla si llega tarde
pthread_key_t g_key;
pthread_once_t g_key_once = PTHREAD_ONCE_INIT;

uintptr_t* bionic_tp() {
    uintptr_t* tp;
    asm volatile("mrs %0, tpidr_el0" : "=r"(tp));
    return tp;
}

void fill(uint8_t* block) {
    if (g_tpl.known) memcpy(block + g_tpl.offset, g_tpl.image, g_tpl.image_size);
}

void free_block(void* p) {
    auto* block = static_cast<uint8_t*>(p);
    {
        std::lock_guard lock(g_m);
        g_blocks.erase(std::remove(g_blocks.begin(), g_blocks.end(), block), g_blocks.end());
    }
    // Si otro destructor del hilo vuelve a leer el TLS del juego, el stub le da un bloque nuevo.
    if (bionic_tp()[kSlot] == reinterpret_cast<uintptr_t>(block)) bionic_tp()[kSlot] = 0;
    free(block);
}

uintptr_t ensure() {
    uintptr_t* tp = bionic_tp();
    if (tp[kSlot]) return tp[kSlot];
    pthread_once(&g_key_once, [] { pthread_key_create(&g_key, free_block); });
    std::lock_guard lock(g_m);
    // Sin plantilla aún (el hilo principal antes de DT_INIT) se reserva de sobra; set_template la copia luego.
    const size_t align = std::max<size_t>(g_tpl.align, 16);
    const size_t size = ((g_tpl.known ? g_tpl.offset + g_tpl.total_size : 4096) + align - 1) & ~(align - 1);
    auto* block = static_cast<uint8_t*>(aligned_alloc(align, size));
    memset(block, 0, size);
    fill(block);
    g_blocks.push_back(block);
    tp[kSlot] = reinterpret_cast<uintptr_t>(block);
    pthread_setspecific(g_key, block);
    return tp[kSlot];
}

constexpr uint32_t kNop = 0xD503201F;

uint32_t branch(const void* from, const void* to, uint32_t opcode) {
    const intptr_t delta = (reinterpret_cast<intptr_t>(to) - reinterpret_cast<intptr_t>(from)) / 4;
    return opcode | (static_cast<uint32_t>(delta) & 0x3FFFFFF);
}

}  // namespace

// Lo llama nier_tls_slow (tls_slow.S), que guarda todos los registros y los flags.
extern "C" uintptr_t nier_tls_ensure_thread() { return ensure(); }

void thread_start() { ensure(); }

void set_template(const void* image, size_t image_size, size_t total_size, size_t align) {
    std::lock_guard lock(g_m);
    if (g_tpl.known) return;  // un solo módulo con TLS: main
    g_tpl.image = static_cast<const uint8_t*>(image);
    g_tpl.image_size = image_size;
    g_tpl.total_size = total_size;
    g_tpl.align = align;
    // Como musl con TLS_ABOVE_TP: 16 bytes de TCB y luego la imagen, alineada como su dirección en el módulo.
    g_tpl.offset = kTcbSize + ((reinterpret_cast<uintptr_t>(image) - kTcbSize) & (align - 1));
    g_tpl.known = true;
    printf("TLS del juego: %zu bytes (%zu de .tdata), alineado a %zu, en TP+%#zx; %zu hilos ya tenían bloque\n",
           total_size, image_size, align, g_tpl.offset, g_blocks.size());
    if (g_tpl.offset + total_size > 4096) {
        printf("[aviso] el TLS del juego no cabe en el bloque provisional de los hilos que ya existían\n");
        return;
    }
    for (uint8_t* block : g_blocks) fill(block);
}

bool patch(Module& mod, std::string& err) {
    // Stub de cada sitio (10 instrucciones) y, al principio del área, el salto largo a nier_tls_slow.
    constexpr size_t kStubSize = 40, kVeneerSize = 32;
    auto* text = reinterpret_cast<uint32_t*>(mod.base + mod.seg_off[0]);
    const size_t words = mod.seg_size[0] / 4;
    auto* veneer = reinterpret_cast<uint32_t*>(mod.patch_area);
    veneer[0] = 0xA9BF7BF0;  // stp x16, x30, [sp, #-16]!
    veneer[1] = 0x580000B0;  // ldr x16, =nier_tls_slow (en +24)
    veneer[2] = 0xD63F0200;  // blr x16
    veneer[3] = 0xA8C17BF0;  // ldp x16, x30, [sp], #16
    veneer[4] = 0xD65F03C0;  // ret
    veneer[5] = kNop;
    const uint64_t slow = reinterpret_cast<uint64_t>(&nier_tls_slow);
    memcpy(&veneer[6], &slow, sizeof slow);

    size_t count = 0;
    for (size_t k = 0; k < words; k++) {
        const uint32_t insn = text[k];
        if ((insn & 0xFFFFFFE0) != 0xD53BD040) continue;  // mrs xN, tpidr_el0
        const uint32_t n = insn & 31;
        if (n == 31) continue;
        auto* stub = reinterpret_cast<uint32_t*>(mod.patch_area + kVeneerSize + count * kStubSize);
        if (reinterpret_cast<uint8_t*>(stub + kStubSize / 4) > mod.base - Module::kProbeAreaSize) {
            err = mod.name + ": no caben los stubs de TLS";
            return false;
        }
        uint32_t* site = &text[k];
        uint32_t* back = site + 1;
        stub[0] = insn;                              // mrs xN, tpidr_el0
        stub[1] = 0xF9400800 | n << 5 | n;           // ldr xN, [xN, #16]: el TP del juego
        stub[2] = 0xB4000040 | n;                    // cbz xN, +8: el hilo aún no tiene bloque
        stub[3] = branch(&stub[3], back, 0x14000000);
        if (n == 0) {
            stub[4] = 0xF81F0FFE;                    // str x30, [sp, #-16]!
            stub[5] = branch(&stub[5], veneer, 0x94000000);
            stub[6] = 0xF84107FE;                    // ldr x30, [sp], #16
            stub[7] = branch(&stub[7], back, 0x14000000);
            stub[8] = kNop;
        } else if (n == 30) {                        // x30 es el destino: su valor viejo no hace falta
            stub[4] = 0xF81F0FE0;                    // str x0, [sp, #-16]!
            stub[5] = branch(&stub[5], veneer, 0x94000000);
            stub[6] = 0xAA0003FE;                    // mov x30, x0
            stub[7] = 0xF84107E0;                    // ldr x0, [sp], #16
            stub[8] = branch(&stub[8], back, 0x14000000);
        } else {
            stub[4] = 0xA9BF7BE0;                    // stp x0, x30, [sp, #-16]!
            stub[5] = branch(&stub[5], veneer, 0x94000000);
            stub[6] = 0xAA0003E0 | n;                // mov xN, x0
            stub[7] = 0xA8C17BE0;                    // ldp x0, x30, [sp], #16
            stub[8] = branch(&stub[8], back, 0x14000000);
        }
        stub[9] = kNop;
        const intptr_t reach = reinterpret_cast<uint8_t*>(site) - reinterpret_cast<uint8_t*>(stub);
        if (reach >= (intptr_t{1} << 27) || -reach >= (intptr_t{1} << 27)) {
            err = mod.name + ": un mrs tpidr_el0 queda a más de 128 MB de su stub";
            return false;
        }
        *site = branch(site, stub, 0x14000000);
        count++;
    }
    if (mprotect(mod.patch_area, Module::kPatchAreaSize, PROT_READ | PROT_EXEC) != 0) {
        err = std::string("mprotect del área de parches: ") + strerror(errno);
        return false;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(mod.patch_area),
                            reinterpret_cast<char*>(mod.patch_area + Module::kPatchAreaSize));
    printf("%s: %zu lecturas de tpidr_el0 van al TLS propio del juego\n", mod.name.c_str(), count);
    return true;
}

}  // namespace tls
