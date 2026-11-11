#include "loader/nso.h"

#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>

namespace {

constexpr int64_t DT_NULL = 0, DT_PLTRELSZ = 2, DT_HASH = 4, DT_STRTAB = 5, DT_SYMTAB = 6, DT_RELA = 7,
                  DT_RELASZ = 8, DT_INIT = 12, DT_JMPREL = 23, DT_INIT_ARRAY = 25, DT_INIT_ARRAYSZ = 27,
                  DT_GNU_HASH = 0x6FFFFEF5;
constexpr uint32_t R_AARCH64_ABS64 = 257, R_AARCH64_GLOB_DAT = 1025, R_AARCH64_JUMP_SLOT = 1026,
                   R_AARCH64_RELATIVE = 1027;
constexpr uint16_t SHN_ABS = 0xFFF1;

std::vector<const Module*> g_modules;

template <typename T>
T rd(const uint8_t* p) {
    T v;
    memcpy(&v, p, sizeof v);
    return v;
}

size_t page_up(size_t n) { return (n + 0xFFF) & ~size_t{0xFFF}; }

// Bloque LZ4 crudo, el formato de los segmentos NSO.
bool lz4_block(const uint8_t* src, size_t n, uint8_t* dst, size_t size) {
    size_t i = 0, o = 0;
    auto extend = [&](size_t& len) {
        uint8_t b;
        do {
            if (i >= n) return false;
            b = src[i++];
            len += b;
        } while (b == 255);
        return true;
    };
    while (i < n) {
        uint8_t token = src[i++];
        size_t lit = token >> 4;
        if (lit == 15 && !extend(lit)) return false;
        if (i + lit > n || o + lit > size) return false;
        memcpy(dst + o, src + i, lit);
        i += lit;
        o += lit;
        if (i >= n) break;
        if (i + 2 > n) return false;
        size_t dist = src[i] | src[i + 1] << 8;
        i += 2;
        size_t len = token & 15;
        if (len == 15 && !extend(len)) return false;
        len += 4;
        if (dist == 0 || dist > o || o + len > size) return false;
        for (size_t k = 0; k < len; k++) dst[o + k] = dst[o + k - dist];  // puede solaparse
        o += len;
    }
    return o == size;
}

size_t gnu_hash_count(const uint8_t* b, uint64_t off) {
    uint32_t nbuckets = rd<uint32_t>(b + off), symoffset = rd<uint32_t>(b + off + 4);
    uint32_t bloom_size = rd<uint32_t>(b + off + 8);
    const uint8_t* buckets = b + off + 16 + bloom_size * 8;
    uint32_t last = 0;
    for (uint32_t k = 0; k < nbuckets; k++) last = std::max(last, rd<uint32_t>(buckets + 4 * k));
    if (last < symoffset) return symoffset;
    const uint8_t* chain = buckets + 4 * nbuckets;
    while (!(rd<uint32_t>(chain + 4 * (last - symoffset)) & 1)) last++;
    return last + 1;
}

bool parse_dynamic(Module& mod, std::string& err) {
    const uint8_t* b = mod.base;
    uint32_t mod0 = rd<uint32_t>(b + 4);
    if (mod0 + 28 > mod.size || memcmp(b + mod0, "MOD0", 4) != 0) {
        err = mod.name + ": no encuentro la cabecera MOD0";
        return false;
    }
    uint64_t dynamic = mod0 + rd<int32_t>(b + mod0 + 4);
    mod.eh_frame_hdr = mod0 + rd<int32_t>(b + mod0 + 16);
    mod.eh_frame_hdr_end = mod0 + rd<int32_t>(b + mod0 + 20);

    std::map<int64_t, uint64_t> tags;  // primer valor de cada tag
    for (uint64_t off = dynamic; off + 16 <= mod.size; off += 16) {
        int64_t tag = rd<int64_t>(b + off);
        if (tag == DT_NULL) break;
        tags.emplace(tag, rd<uint64_t>(b + off + 8));
    }
    uint64_t strtab = tags[DT_STRTAB], symtab = tags[DT_SYMTAB];
    size_t count = tags.count(DT_HASH)       ? rd<uint32_t>(b + tags[DT_HASH] + 4)
                   : tags.count(DT_GNU_HASH) ? gnu_hash_count(b, tags[DT_GNU_HASH])
                                             : (strtab - symtab) / 24;
    mod.symbols.resize(count);
    for (size_t k = 0; k < count; k++) {
        const uint8_t* p = b + symtab + 24 * k;
        ElfSym& s = mod.symbols[k];
        s.name = reinterpret_cast<const char*>(b + strtab + rd<uint32_t>(p));
        s.bind = p[4] >> 4;
        s.type = p[4] & 15;
        s.shndx = rd<uint16_t>(p + 6);
        s.value = rd<uint64_t>(p + 8);
        s.size = rd<uint64_t>(p + 16);
    }
    for (auto [start, size] : {std::pair{DT_RELA, DT_RELASZ}, std::pair{DT_JMPREL, DT_PLTRELSZ}}) {
        if (!tags.count(start)) continue;
        for (uint64_t off = tags[start]; off < tags[start] + tags[size]; off += 24)
            mod.relocs.push_back({rd<uint64_t>(b + off), rd<uint64_t>(b + off + 8), rd<int64_t>(b + off + 16)});
    }
    mod.init = tags.count(DT_INIT) ? tags[DT_INIT] : 0;
    mod.init_array = tags.count(DT_INIT_ARRAY) ? tags[DT_INIT_ARRAY] : 0;
    mod.init_array_size = tags.count(DT_INIT_ARRAYSZ) ? tags[DT_INIT_ARRAYSZ] : 0;
    return true;
}

}  // namespace

const ElfSym* Module::find_export(std::string_view sym) const {
    for (const ElfSym& s : symbols)
        if (s.defined() && s.name == sym) return &s;
    return nullptr;
}

bool load_nso(const std::string& path, Module& mod, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "no puedo abrir " + path;
        return false;
    }
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (file.size() < 0x100 || memcmp(file.data(), "NSO0", 4) != 0) {
        err = path + " no es un NSO";
        return false;
    }
    const uint8_t* h = file.data();
    uint32_t flags = rd<uint32_t>(h + 0xC), file_off[3], csize[3];
    for (int s = 0; s < 3; s++) {
        file_off[s] = rd<uint32_t>(h + 0x10 + 0x10 * s);
        mod.seg_off[s] = rd<uint32_t>(h + 0x14 + 0x10 * s);
        mod.seg_size[s] = rd<uint32_t>(h + 0x18 + 0x10 * s);
        csize[s] = rd<uint32_t>(h + 0x60 + 4 * s);
    }
    mod.name = path.substr(path.find_last_of('/') + 1);
    mod.bss_size = rd<uint32_t>(h + 0x3C);
    mod.size = page_up(size_t{mod.seg_off[2]} + mod.seg_size[2] + mod.bss_size);
    void* mem = mmap(nullptr, Module::kPatchAreaSize + mod.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (mem == MAP_FAILED) {
        err = std::string("mmap: ") + strerror(errno);
        return false;
    }
    mod.patch_area = static_cast<uint8_t*>(mem);
    mod.base = mod.patch_area + Module::kPatchAreaSize;
    for (int s = 0; s < 3; s++) {
        bool compressed = flags >> s & 1;
        size_t stored = compressed ? csize[s] : mod.seg_size[s];
        if (size_t{file_off[s]} + stored > file.size()) {
            err = mod.name + ": segmento " + std::to_string(s) + " fuera del archivo";
            return false;
        }
        uint8_t* dst = mod.base + mod.seg_off[s];
        if (!compressed) {
            memcpy(dst, h + file_off[s], stored);
        } else if (!lz4_block(h + file_off[s], stored, dst, mod.seg_size[s])) {
            err = mod.name + ": LZ4 corrupto en el segmento " + std::to_string(s);
            return false;
        }
    }
    return parse_dynamic(mod, err);
}

bool relocate(Module& mod, const ImportResolver& resolve, std::string& err) {
    std::vector<uint64_t> cache(mod.symbols.size());
    std::vector<bool> done(mod.symbols.size());
    auto sym_addr = [&](uint32_t idx) {
        if (!done[idx]) {
            const ElfSym& s = mod.symbols[idx];
            cache[idx] = s.shndx == SHN_ABS ? s.value : s.defined() ? mod.addr(s.value) : resolve(s);
            done[idx] = true;
        }
        return cache[idx];
    };
    for (const Module::Rela& r : mod.relocs) {
        uint32_t type = r.info & 0xFFFFFFFF, sym = r.info >> 32;
        if (r.offset + 8 > mod.size || sym >= mod.symbols.size()) {
            err = mod.name + ": relocación fuera de rango";
            return false;
        }
        auto* where = reinterpret_cast<uint64_t*>(mod.base + r.offset);
        switch (type) {
            case R_AARCH64_RELATIVE: *where = mod.addr(r.addend); break;
            case R_AARCH64_ABS64:
            case R_AARCH64_GLOB_DAT:
            case R_AARCH64_JUMP_SLOT: *where = sym_addr(sym) + r.addend; break;
            default: err = mod.name + ": relocación no soportada " + std::to_string(type); return false;
        }
    }
    return true;
}

bool protect(const Module& mod, std::string& err) {
    const int prot[3] = {PROT_READ | PROT_EXEC, PROT_READ, PROT_READ | PROT_WRITE};
    for (int s = 0; s < 3; s++) {
        size_t len = s == 2 ? mod.size - mod.seg_off[2] : page_up(mod.seg_size[s]);
        if (len && mprotect(mod.base + mod.seg_off[s], len, prot[s]) != 0) {
            err = mod.name + ": mprotect del segmento " + std::to_string(s) + ": " + strerror(errno);
            return false;
        }
    }
    auto* text = reinterpret_cast<char*>(mod.base + mod.seg_off[0]);
    __builtin___clear_cache(text, text + mod.seg_size[0]);
    return true;
}

void register_module(const Module* mod) { g_modules.push_back(mod); }

const std::vector<const Module*>& registered_modules() { return g_modules; }
