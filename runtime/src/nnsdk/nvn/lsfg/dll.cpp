// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

// Port de Eden (src/video_core/frame_gen/lossless_dll.cpp y lsfg_translate.cpp): el lector de .rsrc del PE y
// la renumeración de bindings, con stdio en vez de Common::FS y los dos juegos de shaders (FP16 y FP32). Sin
// caché en disco: leer y recorrer la DLL cuesta milisegundos y Eden la recorre igual en cada arranque.

#include "nnsdk/nvn/lsfg/dll.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <tuple>

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

namespace nvn::lsfg {

namespace {

constexpr uint16_t kDosMagic = 0x5A4D;
constexpr uint32_t kPeSignature = 0x00004550;
constexpr uint16_t kPe32Magic = 0x010B, kPe32PlusMagic = 0x020B;
constexpr size_t kDosLfanewOffset = 0x3C, kCoffHeaderSize = 20, kOptionalHeaderSizeOffset = 16;
constexpr size_t kSectionHeaderSize = 40, kDataDirectoryEntrySize = 8;
constexpr size_t kDataDirectoryOffsetPe32 = 96, kDataDirectoryOffsetPe32Plus = 112, kResourceDirectoryIndex = 2;
constexpr size_t kResourceDirectorySize = 16, kResourceNamedCountOffset = 12, kResourceIdCountOffset = 14;
constexpr size_t kResourceEntrySize = 8;
constexpr uint32_t kResourceSubdirectoryFlag = 0x80000000, kResourceTypeRcdata = 10;

constexpr uint32_t kSpirvMagic = 0x07230203;
constexpr size_t kSpirvHeaderWords = 5;

struct Section {
    uint32_t virtual_address, virtual_size, raw_address, raw_size;
};
struct ResourceEntry {
    uint32_t id, offset;
    bool is_directory, is_named;
};

class ImageReader {
public:
    explicit ImageReader(std::span<const uint8_t> image) : image_(image) {}
    template <typename T>
    bool read(size_t offset, T& out) const {
        if (offset > image_.size() || image_.size() - offset < sizeof(T)) return false;
        memcpy(&out, image_.data() + offset, sizeof(T));
        return true;
    }
    bool slice(size_t offset, size_t size, std::span<const uint8_t>& out) const {
        if (offset > image_.size() || image_.size() - offset < size) return false;
        out = image_.subspan(offset, size);
        return true;
    }

private:
    std::span<const uint8_t> image_;
};

std::optional<size_t> find_pe_header(const ImageReader& r) {
    uint16_t dos = 0;
    uint32_t pe = 0, signature = 0;
    if (!r.read(0, dos) || dos != kDosMagic || !r.read(kDosLfanewOffset, pe)) return std::nullopt;
    if (!r.read(pe, signature) || signature != kPeSignature) return std::nullopt;
    return pe;
}

std::optional<size_t> find_data_directory(const ImageReader& r, size_t optional_header) {
    uint16_t magic = 0;
    if (!r.read(optional_header, magic)) return std::nullopt;
    if (magic == kPe32Magic) return optional_header + kDataDirectoryOffsetPe32;
    if (magic == kPe32PlusMagic) return optional_header + kDataDirectoryOffsetPe32Plus;
    return std::nullopt;
}

bool read_sections(const ImageReader& r, size_t pe, std::vector<Section>& out) {
    uint16_t count = 0, optional_size = 0;
    if (!r.read(pe + 4 + 2, count) || !r.read(pe + 4 + kOptionalHeaderSizeOffset, optional_size)) return false;
    const size_t table = pe + 4 + kCoffHeaderSize + optional_size;
    for (size_t i = 0; i < count; i++) {
        const size_t o = table + i * kSectionHeaderSize;
        Section s{};
        if (!r.read(o + 8, s.virtual_size) || !r.read(o + 12, s.virtual_address) || !r.read(o + 16, s.raw_size) ||
            !r.read(o + 20, s.raw_address))
            return false;
        out.push_back(s);
    }
    return true;
}

std::optional<size_t> rva_to_offset(std::span<const Section> sections, uint32_t rva) {
    for (const Section& s : sections) {
        const uint32_t span = std::max(s.virtual_size, s.raw_size);
        if (span == 0 || rva < s.virtual_address) continue;
        if (rva - s.virtual_address < span) return static_cast<size_t>(s.raw_address) + (rva - s.virtual_address);
    }
    return std::nullopt;
}

bool read_entries(const ImageReader& r, size_t directory, std::vector<ResourceEntry>& out) {
    uint16_t named = 0, ids = 0;
    if (!r.read(directory + kResourceNamedCountOffset, named) || !r.read(directory + kResourceIdCountOffset, ids))
        return false;
    out.clear();
    for (size_t i = 0; i < size_t{named} + ids; i++) {
        const size_t o = directory + kResourceDirectorySize + i * kResourceEntrySize;
        uint32_t name = 0, data = 0;
        if (!r.read(o, name) || !r.read(o + 4, data)) return false;
        out.push_back({name & ~kResourceSubdirectoryFlag, data & ~kResourceSubdirectoryFlag,
                       (data & kResourceSubdirectoryFlag) != 0, (name & kResourceSubdirectoryFlag) != 0});
    }
    return true;
}

bool read_leaf(const ImageReader& r, std::span<const Section> sections, size_t leaf, std::span<const uint8_t>& out) {
    uint32_t rva = 0, size = 0;
    if (!r.read(leaf, rva) || !r.read(leaf + 4, size) || size == 0) return false;
    const std::optional<size_t> o = rva_to_offset(sections, rva);
    return o && r.slice(*o, size, out);
}

// RT_RCDATA por id numérico (el primer idioma de cada uno).
bool collect_rcdata(const ImageReader& r, std::span<const Section> sections, size_t base,
                    std::map<uint32_t, std::span<const uint8_t>>& out) {
    std::vector<ResourceEntry> types, names, languages;
    if (!read_entries(r, base, types)) return false;
    for (const ResourceEntry& type : types) {
        if (type.is_named || type.id != kResourceTypeRcdata || !type.is_directory) continue;
        if (!read_entries(r, base + type.offset, names)) return false;
        for (const ResourceEntry& name : names) {
            if (name.is_named || !name.is_directory) continue;
            if (!read_entries(r, base + name.offset, languages)) return false;
            for (const ResourceEntry& language : languages) {
                if (language.is_directory) continue;
                std::span<const uint8_t> data;
                if (!read_leaf(r, sections, base + language.offset, data)) continue;
                out.insert_or_assign(name.id, data);
                break;
            }
        }
    }
    return true;
}

bool is_spirv(std::span<const uint8_t> blob) {
    if (blob.size() < kSpirvHeaderWords * 4 || blob.size() % 4) return false;
    uint32_t magic = 0;
    memcpy(&magic, blob.data(), 4);
    return magic == kSpirvMagic;
}

// Los bindings del SPIR-V de la DLL vienen de registros de D3D; se renumeran 0..N-1 en orden (set, binding).
void renumber_bindings(std::vector<uint32_t>& words) {
    struct Slot {
        uint32_t set, binding;
        size_t literal;
    };
    std::map<uint32_t, uint32_t> sets;
    std::vector<Slot> slots;
    size_t o = kSpirvHeaderWords;
    while (o + 1 <= words.size()) {
        const uint32_t length = words[o] >> 16, opcode = words[o] & 0xffff;
        if (length == 0 || o + length > words.size()) return;
        if (opcode == 54) break;  // OpFunction: las decoraciones van antes
        if (opcode == 71 && length >= 4) {  // OpDecorate
            if (words[o + 2] == 34) sets[words[o + 1]] = words[o + 3];  // DescriptorSet
            else if (words[o + 2] == 33) slots.push_back({0, words[o + 3], o + 3});  // Binding
        }
        o += length;
    }
    for (Slot& s : slots) {
        auto hit = sets.find(words[s.literal - 2]);
        s.set = hit == sets.end() ? 0 : hit->second;
    }
    std::stable_sort(slots.begin(), slots.end(),
                     [](const Slot& a, const Slot& b) { return std::tie(a.set, a.binding) < std::tie(b.set, b.binding); });
    for (size_t i = 0; i < slots.size(); i++) words[slots[i].literal] = static_cast<uint32_t>(i);
}

std::vector<uint32_t> adopt(std::span<const uint8_t> blob) {
    if (!is_spirv(blob)) return {};
    std::vector<uint32_t> words(blob.size() / 4);
    memcpy(words.data(), blob.data(), blob.size());
    renumber_bindings(words);
    return words;
}

std::vector<uint32_t> performance_ids() {
    std::vector<uint32_t> ids{shader::kMipmaps, shader::kGenerate};
    for (uint32_t id = 280; id <= 302; id++) ids.push_back(id);
    return ids;
}

}  // namespace

const char* dll_status_name(DllStatus s) {
    switch (s) {
        case DllStatus::Ok: return "bien";
        case DllStatus::NotFound: return "no está";
        case DllStatus::Unreadable: return "no se puede leer";
        case DllStatus::NotPortableExecutable: return "no es un PE";
        case DllStatus::MissingShaders: return "faltan shaders";
    }
    return "?";
}

std::string find_dll() {
    const char* data = getenv("NIER_DATA");
    const std::string candidates[] = {std::string(data ? data : ".") + "/lsfg/Lossless.dll",
                                      "/storage/emulated/0/Download/Lossless.dll"};
    for (const std::string& path : candidates)
        if (access(path.c_str(), R_OK) == 0) return path;
    return {};
}

DllStatus load_shaders(const std::string& path, Modules& fp16, Modules& fp32, size_t* dll_bytes) {
    fp16.clear();
    fp32.clear();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return access(path.c_str(), F_OK) == 0 ? DllStatus::Unreadable : DllStatus::NotFound;
    std::vector<uint8_t> image;
    if (fseek(f, 0, SEEK_END) == 0) {
        const long size = ftell(f);
        if (size > 0 && fseek(f, 0, SEEK_SET) == 0) {
            image.resize(static_cast<size_t>(size));
            if (fread(image.data(), 1, image.size(), f) != image.size()) image.clear();
        }
    }
    fclose(f);
    if (image.empty()) return DllStatus::Unreadable;
    if (dll_bytes) *dll_bytes = image.size();

    const ImageReader r(image);
    const std::optional<size_t> pe = find_pe_header(r);
    if (!pe) return DllStatus::NotPortableExecutable;
    const std::optional<size_t> directory = find_data_directory(r, *pe + 4 + kCoffHeaderSize);
    std::vector<Section> sections;
    if (!directory || !read_sections(r, *pe, sections)) return DllStatus::NotPortableExecutable;
    uint32_t resource_rva = 0;
    if (!r.read(*directory + kResourceDirectoryIndex * kDataDirectoryEntrySize, resource_rva) || !resource_rva)
        return DllStatus::MissingShaders;
    const std::optional<size_t> base = rva_to_offset(sections, resource_rva);
    if (!base) return DllStatus::NotPortableExecutable;
    std::map<uint32_t, std::span<const uint8_t>> resources;
    if (!collect_rcdata(r, sections, *base, resources)) return DllStatus::MissingShaders;

    for (auto [offset, out] : {std::pair<uint32_t, Modules*>{shader::kOffsetFp16, &fp16}, {shader::kOffsetFp32, &fp32}}) {
        for (uint32_t id : performance_ids()) {
            auto hit = resources.find(id + offset);
            std::vector<uint32_t> words;
            if (hit != resources.end()) words = adopt(hit->second);
            if (words.empty()) {  // falta uno: el juego entero no vale
                out->clear();
                break;
            }
            out->emplace(id, std::move(words));
        }
    }
    return fp16.empty() && fp32.empty() ? DllStatus::MissingShaders : DllStatus::Ok;
}

bool audit_module(const std::vector<uint32_t>& w, bool float16, std::string* entry, std::vector<int>* bindings,
                  std::string* why) {
    char buf[96];
    auto fail = [&](std::string reason) {
        *why = std::move(reason);
        return false;
    };
    if (w.size() < kSpirvHeaderWords || w[0] != kSpirvMagic) return fail("no es SPIR-V");
    if (w[1] > 0x00010300) {  // Vulkan 1.1 sin VK_KHR_spirv_1_4: hasta SPIR-V 1.3
        snprintf(buf, sizeof buf, "SPIR-V %u.%u (el dispositivo admite hasta 1.3)", (w[1] >> 16) & 0xff, (w[1] >> 8) & 0xff);
        return fail(buf);
    }
    // Lo que el dispositivo cubre con lo que enciende vk.cpp (features básicas y shaderFloat16).
    auto capability_ok = [&](uint32_t c) {
        switch (c) {
            case 1:   // Shader
            case 25:  // ImageGatherExtended
            case 49:  // StorageImageExtendedFormats
            case 50:  // ImageQuery
            case 55:  // StorageImageReadWithoutFormat
            case 56:  // StorageImageWriteWithoutFormat
                return true;
            case 9: return float16;  // Float16
            default: return false;
        }
    };
    struct Var {
        uint32_t type, storage;
    };
    std::map<uint32_t, uint32_t> binding_of;
    std::map<uint32_t, int> type_kind;        // tipo -> VK_DESCRIPTOR_TYPE_* (-1: otra cosa)
    std::map<uint32_t, uint32_t> pointee;     // puntero -> tipo apuntado
    std::map<uint32_t, Var> variables;
    entry->clear();
    size_t o = kSpirvHeaderWords;
    while (o < w.size()) {
        const uint32_t n = w[o] >> 16, op = w[o] & 0xffff;
        if (n == 0 || o + n > w.size()) return fail("SPIR-V cortado");
        const uint32_t* a = &w[o + 1];
        // Operandos mínimos de cada instrucción que se mira (con menos, el SPIR-V está roto).
        const uint32_t need = op == 25 ? 9 : op == 15 || op == 32 || op == 59 ? 4 : op == 14 || op == 71 ? 3 : 2;
        if (n < need && (op == 10 || op == 14 || op == 15 || op == 17 || op == 25 || op == 26 || op == 30 || op == 32 ||
                         op == 59 || op == 71))
            return fail("SPIR-V roto");
        switch (op) {
            case 17:  // OpCapability
                if (!capability_ok(a[0])) {
                    snprintf(buf, sizeof buf, "capacidad %u sin soporte", a[0]);
                    return fail(buf);
                }
                break;
            case 10:  // OpExtension
                return fail("pide una extensión de SPIR-V: " + std::string(reinterpret_cast<const char*>(a)));
            case 14:  // OpMemoryModel: direccionamiento lógico, Simple o GLSL450 (no el modelo de Vulkan)
                if (a[0] != 0 || a[1] > 1) {
                    snprintf(buf, sizeof buf, "modelo de memoria %u/%u", a[0], a[1]);
                    return fail(buf);
                }
                break;
            case 15:  // OpEntryPoint
                if (a[0] == 5 && entry->empty()) *entry = reinterpret_cast<const char*>(a + 2);  // GLCompute
                break;
            case 71:  // OpDecorate
                if (n >= 4 && a[1] == 33) binding_of[a[0]] = a[2];  // Binding
                break;
            case 25:  // OpTypeImage: Sampled 1 = muestreada, 2 = storage
                type_kind[a[0]] = a[6] == 1 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                  : a[6] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                              : -1;
                break;
            case 26: type_kind[a[0]] = VK_DESCRIPTOR_TYPE_SAMPLER; break;  // OpTypeSampler
            case 30: type_kind[a[0]] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; break;  // OpTypeStruct (con Uniform)
            case 32: pointee[a[0]] = a[2]; break;  // OpTypePointer
            case 59: variables[a[1]] = {a[0], a[2]}; break;  // OpVariable
        }
        o += n;
    }
    if (entry->empty()) return fail("sin punto de entrada de compute");
    bindings->assign(binding_of.size(), -1);
    for (auto [id, binding] : binding_of) {
        auto v = variables.find(id);
        if (v == variables.end() || binding >= bindings->size()) return fail("binding raro");
        auto t = type_kind.find(pointee[v->second.type]);
        int kind = t == type_kind.end() ? -1 : t->second;
        if (kind == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && v->second.storage != 2) kind = -1;  // solo UBO (Uniform)
        if (kind < 0) return fail("un binding no es UBO, sampler ni imagen");
        (*bindings)[binding] = kind;
    }
    return true;
}

}  // namespace nvn::lsfg
