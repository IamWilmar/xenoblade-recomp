// SPDX-License-Identifier: GPL-3.0-or-later
// Prueba en el Mac de los ficheros de caché de la capa NVN: la caché de SPIR-V (nnsdk/nvn/spirv_cache, <datos>/spirv.bin)
// y los registros del precalentado de pipelines (nnsdk/nvn/records_file.h, <datos>/pipelines_estado.bin). Ida y vuelta,
// final a medias, registro roto, otra cabecera / otro traductor.
//
//   clang++ -std=c++20 -I runtime/src -I runtime/recompiler tools/tests/cache_files_test.cpp \
//       runtime/src/nnsdk/nvn/spirv_cache.cpp -o /tmp/cache_files_test && /tmp/cache_files_test <carpeta temporal>

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nnsdk/nvn/records_file.h"
#include "nnsdk/nvn/spirv_cache.h"

namespace {

int g_failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

size_t file_size(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 ? static_cast<size_t>(st.st_size) : 0;
}

void corrupt_byte(const std::string& path, size_t offset) {
    FILE* f = std::fopen(path.c_str(), "r+b");
    std::fseek(f, static_cast<long>(offset), SEEK_SET);
    const int c = std::fgetc(f);
    std::fseek(f, static_cast<long>(offset), SEEK_SET);
    std::fputc(c ^ 0x5a, f);
    std::fclose(f);
}

// --- records_file ---

struct Rec {
    uint32_t magic;
    uint32_t n;
    uint64_t payload[8];
    uint64_t sum;
};
constexpr uint32_t kMagic = 0x54534550;
uint64_t sum_of(const Rec& r) {
    const auto* p = reinterpret_cast<const uint8_t*>(&r);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < offsetof(Rec, sum); k++) h = (h ^ p[k]) * 0x100000001b3ULL;
    return h;
}
Rec make(uint32_t n) {
    Rec r;
    std::memset(&r, 0, sizeof r);
    r.magic = kMagic;
    r.n = n;
    for (int k = 0; k < 8; k++) r.payload[k] = n * 1000003ull + k;
    r.sum = sum_of(r);
    return r;
}
bool valid(const uint8_t* p) {
    Rec r;
    std::memcpy(&r, p, sizeof r);
    return r.magic == kMagic && r.sum == sum_of(r);
}
std::vector<uint32_t> numbers(const nvn::records_file::Loaded& l) {
    std::vector<uint32_t> out;
    for (size_t k = 0; k + sizeof(Rec) <= l.records.size(); k += sizeof(Rec)) {
        Rec r;
        std::memcpy(&r, l.records.data() + k, sizeof r);
        out.push_back(r.n);
    }
    return out;
}

void test_records(const std::string& dir) {
    namespace rf = nvn::records_file;
    const std::string path = dir + "/pipelines_estado.bin";
    unlink(path.c_str());
    uint8_t header[32] = {'N', 'V', 'P', 'I', 'P', 'E', 'S', '1', 1};
    uint8_t other[32] = {'N', 'V', 'P', 'I', 'P', 'E', 'S', '1', 2};

    rf::Loaded l = rf::load(path, header, sizeof header, sizeof(Rec), valid);  // no había fichero
    CHECK(l.ready && l.reset && l.records.empty() && file_size(path) == sizeof header);

    const Rec three[3] = {make(1), make(2), make(3)};
    CHECK(rf::append(path, three, sizeof three));
    l = rf::load(path, header, sizeof header, sizeof(Rec), valid);
    CHECK(l.ready && !l.reset && !l.cut && (numbers(l) == std::vector<uint32_t>{1, 2, 3}));

    // El proceso murió a media escritura: el trozo se corta y lo que se añada después se lee.
    const Rec four = make(4);
    CHECK(rf::append(path, &four, sizeof four / 2));
    l = rf::load(path, header, sizeof header, sizeof(Rec), valid);
    CHECK(l.ready && l.cut && (numbers(l) == std::vector<uint32_t>{1, 2, 3}));
    CHECK(file_size(path) == sizeof header + 3 * sizeof(Rec));
    CHECK(rf::append(path, &four, sizeof four));
    l = rf::load(path, header, sizeof header, sizeof(Rec), valid);
    CHECK(!l.cut && (numbers(l) == std::vector<uint32_t>{1, 2, 3, 4}));

    // Un registro roto en medio: la lectura acaba ahí (y se corta).
    corrupt_byte(path, sizeof header + sizeof(Rec) + 12);
    l = rf::load(path, header, sizeof header, sizeof(Rec), valid);
    CHECK(l.cut && (numbers(l) == std::vector<uint32_t>{1}));
    CHECK(file_size(path) == sizeof header + sizeof(Rec));

    // Otra cabecera (otro formato u otro traductor): de cero.
    l = rf::load(path, other, sizeof other, sizeof(Rec), valid);
    CHECK(l.ready && l.reset && l.records.empty() && file_size(path) == sizeof other);
    l = rf::load(path, header, sizeof header, sizeof(Rec), valid);
    CHECK(l.reset && l.records.empty());
}

// --- caché de SPIR-V ---

nvnshader::Translated sample(uint32_t seed, bool brx) {
    nvnshader::Translated t;
    t.stage = brx ? nvnshader::Stage::Compute : nvnshader::Stage::Fragment;
    t.spirv = {0x07230203u, 0x00010300u, 0u, 100u + seed, 0u};
    for (uint32_t k = 0; k < 2000 + seed; k++) t.spirv.push_back(k * 2654435761u ^ seed);
    nvnshader::Resource r{nvnshader::Resource::Texture, 64 + seed, 6, 0xc83c, 3};
    r.count = 223;
    r.size_shift = 6;
    r.handle_and_mask = 0x7fff;
    r.handle_or = 0x10b00000;
    r.set = nvnshader::kBindlessSet;
    r.has_secondary = true;
    r.secondary_cbuf_index = 3;
    r.secondary_cbuf_offset = 0x40;
    t.resources.push_back(r);
    t.resources.push_back({nvnshader::Resource::UniformBuffer, 65, 1, 0, 0});
    t.constant_buffer_mask = 0x2a + seed;
    t.constant_buffer_used[1] = 0x400;
    t.constant_buffer_used[6] = 0xc900;
    t.input_generics = 0x1f ^ seed;
    t.output_generics = 0xff00;
    t.color_outputs = 7;
    t.has_brx = brx;
    t.has_bindless_set = true;
    t.conditional_barriers = brx ? 2 : 0;
    return t;
}

bool same(const nvnshader::Translated& a, const nvnshader::Translated& b) {
    if (a.stage != b.stage || a.spirv != b.spirv || a.resources.size() != b.resources.size()) return false;
    for (size_t k = 0; k < a.resources.size(); k++) {
        const nvnshader::Resource &x = a.resources[k], &y = b.resources[k];
        if (x.kind != y.kind || x.binding != y.binding || x.cbuf_index != y.cbuf_index || x.cbuf_offset != y.cbuf_offset ||
            x.texture_type != y.texture_type || x.is_depth != y.is_depth || x.is_written != y.is_written ||
            x.has_secondary != y.has_secondary || x.secondary_cbuf_index != y.secondary_cbuf_index ||
            x.secondary_cbuf_offset != y.secondary_cbuf_offset || x.shift_left != y.shift_left ||
            x.secondary_shift_left != y.secondary_shift_left || x.count != y.count || x.size_shift != y.size_shift ||
            x.handle_and_mask != y.handle_and_mask || x.handle_or != y.handle_or || x.set != y.set)
            return false;
    }
    return a.constant_buffer_mask == b.constant_buffer_mask &&
           std::memcmp(a.constant_buffer_used, b.constant_buffer_used, sizeof a.constant_buffer_used) == 0 &&
           a.input_generics == b.input_generics && a.output_generics == b.output_generics &&
           a.color_outputs == b.color_outputs && a.has_brx == b.has_brx && a.has_bindless_set == b.has_bindless_set &&
           a.conditional_barriers == b.conditional_barriers;
}

void test_spirv(const std::string& dir) {
    namespace sc = nvn::spirv_cache;
    using namespace std::chrono_literals;
    const std::string path = dir + "/spirv.bin";
    unlink(path.c_str());
    const nvnshader::Translated a = sample(1, false), b = sample(2, true);
    std::vector<uint8_t> bytes;
    sc::serialize(a, bytes);
    nvnshader::Translated back;
    CHECK(sc::deserialize(bytes.data(), bytes.size(), back) && same(a, back));
    CHECK(!sc::deserialize(bytes.data(), bytes.size() - 1, back));  // corto: no vale

    sc::Hasher h1, h2, h3;
    h1.u32(1), h1.u64(2);
    h2.u32(1), h2.u64(2);
    h3.u32(1), h3.u64(3);
    const sc::Key ka = h1.key(), kb = h3.key();
    CHECK(h1.key() == h2.key() && !(ka == kb));

    {
        sc::Cache c;
        CHECK(!c.wait_loaded(10ms));  // sin abrir
        c.open(path, 0x1234);
        CHECK(c.wait_loaded(2000ms));
        c.insert(ka, a);
        c.insert(kb, b);
        c.wait_idle();
        nvnshader::Translated got;
        CHECK(c.find(ka, got) && same(a, got));
        CHECK(c.stats().written == 2);
    }
    {
        sc::Cache c;  // otra partida, mismo traductor
        c.open(path, 0x1234);
        CHECK(c.wait_loaded(2000ms));
        nvnshader::Translated got;
        CHECK(c.find(ka, got) && same(a, got));
        CHECK(c.find(kb, got) && same(b, got));
        CHECK(c.stats().loaded == 2 && !c.stats().rebuilt);
    }
    const size_t full = file_size(path);
    truncate(path.c_str(), static_cast<off_t>(full - 7));  // el segundo registro a medias
    {
        sc::Cache c;
        c.open(path, 0x1234);
        CHECK(c.wait_loaded(2000ms));
        nvnshader::Translated got;
        CHECK(c.find(ka, got) && !c.find(kb, got));
        CHECK(c.stats().cut && c.stats().loaded == 1);
        c.insert(kb, b);  // se vuelve a añadir tras el corte
        c.wait_idle();
    }
    {
        sc::Cache c;
        c.open(path, 0x1234);
        CHECK(c.wait_loaded(2000ms));
        nvnshader::Translated got;
        CHECK(c.find(kb, got) && same(b, got) && c.stats().loaded == 2);
    }
    {
        sc::Cache c;  // otro traductor: se vacía
        c.open(path, 0x9999);
        CHECK(c.wait_loaded(2000ms));
        nvnshader::Translated got;
        CHECK(!c.find(ka, got) && c.stats().rebuilt && c.stats().loaded == 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";
    test_records(dir);
    test_spirv(dir);
    std::printf("%s (%d fallos)\n", g_failures ? "MAL" : "bien", g_failures);
    return g_failures ? 1 : 0;
}
