// SPDX-License-Identifier: GPL-3.0-or-later
// Caché de traducciones Maxwell → SPIR-V (ver spirv_cache.h).

#include "nnsdk/nvn/spirv_cache.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

#include "nnsdk/nvn/nvn.h"

namespace nvn::spirv_cache {

namespace {

constexpr char kMagic[8] = {'N', 'V', 'S', 'P', 'I', 'R', 'V', '1'};
constexpr uint32_t kFormat = 1;
constexpr uint32_t kRecordMagic = 0x52565053;  // "SPVR"
constexpr size_t kHeaderSize = 32, kRecordHeader = 32;
constexpr uint64_t kMaxFile = 128ull << 20;  // tope: con más, se deja de añadir
constexpr uint32_t kResourceWords = 18;      // campos de un Resource (el último, is_integer, era de reserva)

uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < n; k++) h = (h ^ p[k]) * 0x100000001b3ULL;
    return h;
}

struct Out {
    std::vector<uint8_t>& v;
    void put(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        v.insert(v.end(), b, b + n);
    }
    void u32(uint32_t x) { put(&x, 4); }
    void u64(uint64_t x) { put(&x, 8); }
};

struct In {
    const uint8_t* p;
    size_t n;
    bool ok = true;
    bool get(void* dst, size_t size) {
        if (!ok || size > n) return ok = false;
        memcpy(dst, p, size);
        p += size;
        n -= size;
        return true;
    }
    uint32_t u32() {
        uint32_t x = 0;
        get(&x, 4);
        return x;
    }
    uint64_t u64() {
        uint64_t x = 0;
        get(&x, 8);
        return x;
    }
};

bool read_all(int fd, uint8_t* p, size_t n, off_t offset) {
    while (n) {
        const ssize_t r = pread(fd, p, n, offset);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r, n -= static_cast<size_t>(r), offset += r;
    }
    return true;
}

bool write_all(int fd, const uint8_t* p, size_t n, off_t offset) {
    while (n) {
        const ssize_t w = pwrite(fd, p, n, offset);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        p += w, n -= static_cast<size_t>(w), offset += w;
    }
    return true;
}

}  // namespace

void Hasher::bytes(const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t a = a_, b = b_;
    for (size_t k = 0; k < size; k++) {
        a = (a ^ p[k]) * 0x100000001b3ULL;
        b = (b ^ p[k]) * 0x9E3779B97F4A7C15ULL;
        b = (b << 27) | (b >> 37);
    }
    a_ = a, b_ = b;
}

void serialize(const nvnshader::Translated& t, std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(t.spirv.size() * 4 + t.resources.size() * kResourceWords * 4 + 160);
    Out o{out};
    o.u32(static_cast<uint32_t>(t.stage));
    o.u32(static_cast<uint32_t>(t.spirv.size()));
    o.put(t.spirv.data(), t.spirv.size() * sizeof(uint32_t));
    o.u32(static_cast<uint32_t>(t.resources.size()));
    for (const nvnshader::Resource& r : t.resources) {
        const uint32_t w[kResourceWords] = {static_cast<uint32_t>(r.kind), r.binding, r.cbuf_index, r.cbuf_offset,
                                            r.texture_type, r.is_depth, r.is_written, r.has_secondary,
                                            r.secondary_cbuf_index, r.secondary_cbuf_offset, r.shift_left,
                                            r.secondary_shift_left, r.count, r.size_shift, r.handle_and_mask, r.handle_or,
                                            r.set, r.is_integer};
        o.put(w, sizeof w);
    }
    o.u32(t.constant_buffer_mask);
    o.put(t.constant_buffer_used, sizeof t.constant_buffer_used);
    o.u64(t.input_generics);
    o.u64(t.output_generics);
    o.u32(t.color_outputs);
    o.u32(t.has_brx);
    o.u32(t.has_bindless_set);
    o.u32(static_cast<uint32_t>(t.conditional_barriers));
}

bool deserialize(const uint8_t* data, size_t size, nvnshader::Translated& t) {
    t = {};
    In in{data, size};
    const uint32_t stage = in.u32();
    if (stage > static_cast<uint32_t>(nvnshader::Stage::Compute)) return false;
    t.stage = static_cast<nvnshader::Stage>(stage);
    const uint32_t words = in.u32();
    if (!in.ok || words > in.n / 4) return false;
    t.spirv.resize(words);
    in.get(t.spirv.data(), words * sizeof(uint32_t));
    const uint32_t count = in.u32();
    if (!in.ok || count > in.n / (kResourceWords * 4)) return false;
    t.resources.resize(count);
    for (nvnshader::Resource& r : t.resources) {
        uint32_t w[kResourceWords];
        in.get(w, sizeof w);
        if (w[0] > nvnshader::Resource::Image) return false;
        r.kind = static_cast<nvnshader::Resource::Kind>(w[0]);
        r.binding = w[1], r.cbuf_index = w[2], r.cbuf_offset = w[3], r.texture_type = w[4];
        r.is_depth = w[5] != 0, r.is_written = w[6] != 0, r.has_secondary = w[7] != 0;
        r.secondary_cbuf_index = w[8], r.secondary_cbuf_offset = w[9], r.shift_left = w[10], r.secondary_shift_left = w[11];
        r.count = w[12], r.size_shift = w[13], r.handle_and_mask = w[14], r.handle_or = w[15], r.set = w[16];
        r.is_integer = w[17] != 0;  // antes la palabra de reserva (0): las de un traductor anterior no se leen (huella)
    }
    t.constant_buffer_mask = in.u32();
    in.get(t.constant_buffer_used, sizeof t.constant_buffer_used);
    t.input_generics = in.u64();
    t.output_generics = in.u64();
    t.color_outputs = in.u32();
    t.has_brx = in.u32() != 0;
    t.has_bindless_set = in.u32() != 0;
    t.conditional_barriers = static_cast<int>(in.u32());
    // SPIR-V de verdad (número mágico) y nada sobrante.
    return in.ok && in.n == 0 && t.spirv.size() >= 5 && t.spirv[0] == 0x07230203;
}

Cache::~Cache() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) close(fd_);
}

void Cache::open(const std::string& path, uint64_t build) {
    std::lock_guard lock(mutex_);
    if (opened_) return;
    opened_ = true;
    path_ = path;
    build_ = build;
    thread_ = std::thread([this] {
        load();
        write_loop();
    });
}

void Cache::load() {
    const auto start = std::chrono::steady_clock::now();
    const int fd = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    std::vector<uint8_t> data;
    struct stat st {};
    if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0 && static_cast<uint64_t>(st.st_size) <= 2 * kMaxFile) {
        data.resize(static_cast<size_t>(st.st_size));
        if (!read_all(fd, data.data(), data.size(), 0)) data.clear();
    }
    uint64_t valid = 0;
    size_t loaded = 0;
    bool rebuilt = false, cut = false;
    std::unordered_map<Key, std::vector<uint8_t>, KeyHash> found;
    if (data.size() >= kHeaderSize) {
        uint32_t format = 0;
        uint64_t build = 0;
        memcpy(&format, data.data() + 8, 4);
        memcpy(&build, data.data() + 16, 8);
        if (memcmp(data.data(), kMagic, 8) == 0 && format == kFormat && build == build_) valid = kHeaderSize;
    }
    if (valid) {
        while (valid + kRecordHeader <= data.size()) {
            const uint8_t* h = data.data() + valid;
            uint32_t magic, size;
            Key key;
            uint64_t sum;
            memcpy(&magic, h, 4);
            memcpy(&size, h + 4, 4);
            memcpy(&key.a, h + 8, 8);
            memcpy(&key.b, h + 16, 8);
            memcpy(&sum, h + 24, 8);
            if (magic != kRecordMagic || size > data.size() - valid - kRecordHeader) break;
            const uint8_t* payload = h + kRecordHeader;
            if (fnv(payload, size) != sum) break;
            found[key].assign(payload, payload + size);
            loaded++;
            valid += kRecordHeader + size;
        }
        cut = valid < data.size();
        if (cut && fd >= 0 && ftruncate(fd, static_cast<off_t>(valid)) != 0) valid = 0;  // no se pudo: de cero
    }
    if (!valid && fd >= 0) {  // vacío, de otro traductor o roto: se empieza de cero
        rebuilt = st.st_size > 0;
        found.clear();
        loaded = 0;
        uint8_t header[kHeaderSize] = {};
        memcpy(header, kMagic, 8);
        memcpy(header + 8, &kFormat, 4);
        memcpy(header + 16, &build_, 8);
        if (ftruncate(fd, 0) == 0 && write_all(fd, header, sizeof header, 0)) valid = kHeaderSize;
    }
    {
        std::lock_guard lock(mutex_);
        for (auto& [key, bytes] : found) {
            on_disk_.insert(key);
            entries_.try_emplace(key, std::move(bytes));  // lo traducido mientras tanto se queda
        }
        fd_ = valid ? fd : -1;
        file_bytes_ = valid;
        stats_.loaded = loaded;
        stats_.rebuilt = rebuilt;
        stats_.cut = cut;
        stats_.file_bytes = valid;
        loaded_ = true;
    }
    if (!valid && fd >= 0) close(fd);
    cv_.notify_all();
    NVN_LOG("caché de SPIR-V: %zu traducciones leídas de %s (%llu KB%s%s) en %lld ms\n", loaded, path_.c_str(),
            static_cast<unsigned long long>(valid >> 10), rebuilt ? ", rehecha: otro traductor o rota" : "",
            cut ? ", final roto cortado" : "",
            static_cast<long long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()));
}

void Cache::write_loop() {
    std::vector<uint8_t> record;
    bool full_logged = false;
    std::unique_lock lock(mutex_);
    for (;; cv_.notify_all()) {  // wait_idle mira la cola
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) return;  // stop_
        auto [key, bytes] = std::move(queue_.front());
        queue_.pop_front();
        if (fd_ < 0 || on_disk_.count(key)) continue;
        const uint64_t size = kRecordHeader + bytes.size();
        if (file_bytes_ + size > kMaxFile) {
            if (!full_logged) NVN_LOG("caché de SPIR-V: %s llegó a %llu MB, no se añade más\n", path_.c_str(),
                                      static_cast<unsigned long long>(kMaxFile >> 20));
            full_logged = true;
            continue;
        }
        writing_ = true;
        const uint64_t offset = file_bytes_;
        const int fd = fd_;
        lock.unlock();
        record.resize(static_cast<size_t>(size));
        const uint32_t magic = kRecordMagic, length = static_cast<uint32_t>(bytes.size());
        const uint64_t sum = fnv(bytes.data(), bytes.size());
        memcpy(record.data(), &magic, 4);
        memcpy(record.data() + 4, &length, 4);
        memcpy(record.data() + 8, &key.a, 8);
        memcpy(record.data() + 16, &key.b, 8);
        memcpy(record.data() + 24, &sum, 8);
        memcpy(record.data() + kRecordHeader, bytes.data(), bytes.size());
        const bool ok = write_all(fd, record.data(), record.size(), static_cast<off_t>(offset));
        lock.lock();
        writing_ = false;
        if (ok) {
            file_bytes_ = offset + size;
            stats_.file_bytes = file_bytes_;
            stats_.written++;
            on_disk_.insert(key);
        } else {
            // A medias: lo que haya quedado se descarta al leer (suma que no cuadra); no se escribe detrás.
            NVN_LOG("caché de SPIR-V: no se pudo escribir en %s (%s); sigue solo en memoria\n", path_.c_str(),
                    strerror(errno));
            fd_ = -1;
        }
    }
}

bool Cache::find(const Key& key, nvnshader::Translated& out) {
    std::unique_lock lock(mutex_);
    if (opened_ && !loaded_ && !gave_up_ &&
        !cv_.wait_for(lock, std::chrono::milliseconds(500), [this] { return loaded_; }))
        gave_up_ = true;
    const auto it = entries_.find(key);
    if (it == entries_.end() || !deserialize(it->second.data(), it->second.size(), out)) return false;
    stats_.hits++;
    return true;
}

void Cache::insert(const Key& key, const nvnshader::Translated& t) {
    std::vector<uint8_t> bytes;
    serialize(t, bytes);
    {
        std::lock_guard lock(mutex_);
        if (entries_.count(key)) return;
        stats_.inserted++;
        if (opened_) queue_.emplace_back(key, bytes);
        entries_.emplace(key, std::move(bytes));
    }
    cv_.notify_all();
}

bool Cache::wait_loaded(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return opened_ && cv_.wait_for(lock, timeout, [this] { return loaded_; });
}

Cache::Stats Cache::stats() {
    std::lock_guard lock(mutex_);
    return stats_;
}

void Cache::wait_idle() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !opened_ || (loaded_ && queue_.empty() && !writing_); });
}

Cache& global() {
    static Cache* cache = new Cache;
    return *cache;
}

}  // namespace nvn::spirv_cache
