#pragma once
// Carga y relocación de módulos NSO (los ejecutables de la Switch) en un proceso Android.

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

constexpr uint8_t STB_WEAK = 2;
constexpr uint8_t STT_OBJECT = 1;

struct ElfSym {
    std::string name;
    uint8_t bind = 0, type = 0;
    uint16_t shndx = 0;
    uint64_t value = 0, size = 0;
    bool defined() const { return shndx != 0; }
};

struct Module {
    struct Rela {
        uint64_t offset, info;
        int64_t addend;
    };

    std::string name;
    uint8_t* base = nullptr;
    size_t size = 0;  // text + rodata + data + bss, en páginas
    // Justo antes de base (a menos de ±128 MB de todo el text): código propio al que saltan las instrucciones
    // reescritas al cargar (tls.cpp).
    uint8_t* patch_area = nullptr;
    static constexpr size_t kPatchAreaSize = 1 << 20;
    static constexpr size_t kProbeAreaSize = 64 << 10;  // el final del área: sondas de depuración (probe.cpp)
    uint32_t seg_off[3] = {}, seg_size[3] = {};
    uint32_t bss_size = 0;
    std::vector<ElfSym> symbols;
    std::vector<Rela> relocs;
    uint64_t init = 0, init_array = 0, init_array_size = 0;
    uint64_t eh_frame_hdr = 0, eh_frame_hdr_end = 0;

    uintptr_t addr(uint64_t off) const { return reinterpret_cast<uintptr_t>(base) + off; }
    bool contains(uintptr_t a) const { return base && a >= addr(0) && a < addr(size); }
    const ElfSym* find_export(std::string_view name) const;
};

// Descomprime los segmentos en memoria propia (RW) y lee la tabla dinámica.
bool load_nso(const std::string& path, Module& mod, std::string& err);

// Aplica las relocaciones. resolve da la dirección de cada símbolo que el módulo importa.
using ImportResolver = std::function<uint64_t(const ElfSym&)>;
bool relocate(Module& mod, const ImportResolver& resolve, std::string& err);

// Permisos finales: text R-X, rodata R--, data y bss RW-.
bool protect(const Module& mod, std::string& err);

// Módulos que el crash handler y los logs usan para traducir direcciones.
void register_module(const Module* mod);
const std::vector<const Module*>& registered_modules();
