#include "loader/probe.h"

#include <pthread.h>
#include <time.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" void nier_probe_slow();

namespace probe {
namespace {

struct Probe {
    uint64_t offset;  // en main
    uint32_t limit;
    std::atomic<uint32_t> hits{0};
};

Probe g_probes[64];
uint32_t g_count = 0;
uint64_t g_base = 0, g_size = 0;  // main, para dar x8 como offset

// Instrucciones que dependen del pc: en el stub harían otra cosa.
bool pc_relative(uint32_t insn) {
    return (insn & 0x1F000000) == 0x10000000 ||  // adr, adrp
           (insn & 0x7C000000) == 0x14000000 ||  // b, bl
           (insn & 0xFF000010) == 0x54000000 ||  // b.cond
           (insn & 0x7E000000) == 0x34000000 ||  // cbz, cbnz
           (insn & 0x7E000000) == 0x36000000 ||  // tbz, tbnz
           (insn & 0x3B000000) == 0x18000000 ||  // ldr (literal)
           (insn & 0xFFFFFFE0) == 0xD53BD040;    // mrs tpidr_el0: ya lo reescribe tls.cpp
}

uint32_t branch(const void* from, const void* to, uint32_t opcode) {
    const intptr_t delta = (reinterpret_cast<intptr_t>(to) - reinterpret_cast<intptr_t>(from)) / 4;
    return opcode | (static_cast<uint32_t>(delta) & 0x3FFFFFF);
}

}  // namespace

// Lo llama nier_probe_slow (probe_slow.S) con todo guardado. regs: x0..x15, x18, y luego x19..x28; stub: lo que
// guardó el stub (x16, x17, x30 del juego).
extern "C" void nier_on_probe(uint64_t id, const uint64_t* regs, const uint64_t* stub) {
    if (id >= g_count) return;
    Probe& p = g_probes[id];
    const uint32_t n = p.hits++;
    if (n >= p.limit) return;
    char name[32] = "?";
    pthread_getname_np(pthread_self(), name, sizeof name);
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const double ms = ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
    const uint64_t* hi = regs + 17;  // x19..
    printf("[sonda] %.1f ms main+%#lx #%u '%s' x0=%#lx x1=%#lx x2=%#lx x3=%#lx x4=%#lx x5=%#lx x6=%#lx x7=%#lx "
           "x19=%#lx x20=%#lx x21=%#lx x22=%#lx x23=%#lx x24=%#lx lr=%#lx\n",
           ms, p.offset, n, name, regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7], hi[0], hi[1],
           hi[2], hi[3], hi[4], hi[5], stub[2]);
    const uint64_t x8 = regs[8];
    printf("[sonda]   x8=%#lx%s%#lx\n", x8, x8 >= g_base && x8 < g_base + g_size ? " main+" : " ",
           x8 >= g_base && x8 < g_base + g_size ? x8 - g_base : 0);
    if (n + 1 == p.limit) printf("[sonda] main+%#lx: más llamadas, ya no las registro\n", p.offset);
}

bool install(Module& mod, std::string& err) {
    g_base = reinterpret_cast<uint64_t>(mod.base), g_size = mod.size;
    const char* data = getenv("NIER_DATA");
    std::ifstream in(std::string(data ? data : ".") + "/sondas");
    if (!in) return true;
    // Al final del área de parches; los stubs de TLS van desde el principio (tls.cpp comprueba que no lleguen).
    uint8_t* area = mod.patch_area + Module::kPatchAreaSize - Module::kProbeAreaSize;
    auto* veneer = reinterpret_cast<uint32_t*>(area);
    veneer[0] = 0x58000051;  // ldr x17, =nier_probe_slow
    veneer[1] = 0xD61F0220;  // br x17
    const uint64_t slow = reinterpret_cast<uint64_t>(&nier_probe_slow);
    memcpy(&veneer[2], &slow, sizeof slow);
    auto* text = mod.base + mod.seg_off[0];
    std::string line;
    while (std::getline(in, line) && g_count < 64) {
        uint64_t offset = 0;
        unsigned limit = 40;
        if (sscanf(line.c_str(), "main+%lx %u", &offset, &limit) < 1) continue;
        if (offset % 4 || offset >= mod.seg_size[0]) {
            printf("[sonda] main+%#lx no está en el text\n", offset);
            continue;
        }
        auto* site = reinterpret_cast<uint32_t*>(text + offset);
        if (pc_relative(*site)) {
            printf("[sonda] main+%#lx: la instrucción %08x depende del pc, pon la sonda en otra\n", offset, *site);
            continue;
        }
        auto* stub = reinterpret_cast<uint32_t*>(area + 16 + g_count * 32);
        const uint32_t id = g_count;
        stub[0] = 0xA9BE47F0;                             // stp x16, x17, [sp, #-32]!
        stub[1] = 0xF9000BFE;                             // str x30, [sp, #16]
        stub[2] = 0xD2800010 | id << 5;                   // movz x16, #id
        stub[3] = branch(&stub[3], veneer, 0x94000000);   // bl veneer
        stub[4] = 0xF9400BFE;                             // ldr x30, [sp, #16]
        stub[5] = 0xA8C247F0;                             // ldp x16, x17, [sp], #32
        stub[6] = *site;                                  // la instrucción original
        stub[7] = branch(&stub[7], site + 1, 0x14000000);  // b de vuelta
        g_probes[id].offset = offset;
        g_probes[id].limit = limit;
        *site = branch(site, stub, 0x14000000);
        printf("[sonda] puesta en main+%#lx (%u veces)\n", offset, limit);
        g_count++;
    }
    (void)err;
    return true;
}

}  // namespace probe
