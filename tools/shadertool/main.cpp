// SPDX-License-Identifier: GPL-3.0-or-later
// shadertool: traduce en el Mac un shader NVN volcado por nnsdk/nvn/program.cpp a SPIR-V con el
// mismo puente que usa la Flip (runtime/recompiler/nvn_shader) y lista sus recursos con su binding.
// Uso: shadertool <shader.code> [salida.spv]
//      shadertool --opt <entrada.spv> [salida.spv]: un SPIR-V por el optimizador de la Flip (nvnshader::optimize_spirv)

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>

#include "nvn_shader.h"

static void on_abort(int) {
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    _exit(134);
}

int main(int argc, char** argv) {
    signal(SIGABRT, on_abort);
    if (argc < 2 || (argc < 3 && !std::strcmp(argv[1], "--opt"))) {
        std::fprintf(stderr, "uso: %s <shader.code> [salida.spv]\n     %s --opt <entrada.spv> [salida.spv]\n", argv[0], argv[0]);
        return 2;
    }
    if (!std::strcmp(argv[1], "--opt")) {
        std::ifstream in(argv[2], std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<uint32_t> spirv(bytes.size() / 4);
        std::memcpy(spirv.data(), bytes.data(), spirv.size() * 4);
        const size_t before = spirv.size();
        std::string error;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = nvnshader::optimize_spirv(spirv, error);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (!ok) {
            std::fprintf(stderr, "%s: %s (%.1f ms)\n", argv[2], error.c_str(), ms);
            return 1;
        }
        std::printf("%s: %zu → %zu palabras, %.1f ms\n", argv[2], before, spirv.size(), ms);
        if (argc > 3) {
            std::ofstream out(argv[3], std::ios::binary);
            out.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
        }
        return 0;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    nvnshader::Translated t;
    nvnshader::Options options;
    std::string ir;
    if (getenv("IR")) options.ir = &ir;  // IR=1 shadertool …: imprime el IR de Eden
    // Como en la Flip: arrays bindless al set 1 y NonUniform (el Adreno 650 lo tiene). SIN_BINDLESS=1 / SIN_NONUNIFORM=1:
    // como antes / sin la decoración.
    options.bindless_arrays = !getenv("SIN_BINDLESS");
    options.nonuniform_indexing = !getenv("SIN_NONUNIFORM");
    // CON_BARRERA=1: deja las barreras de grupo dentro de ifs y bucles, como program.cpp con las huellas de
    // kUniformBarrierShaders o de <datos>/con_barrera (para comparar y validar ese SPIR-V).
    options.keep_conditional_barriers = getenv("CON_BARRERA") != nullptr;
    // SIN_COMPARTIDA_PRIVADA=1: la memoria compartida privada de cada invocación se queda en Workgroup (como antes).
    options.shared_to_local = !getenv("SIN_COMPARTIDA_PRIVADA");
    // QUALCOMM=1: los arreglos que program.cpp pone con el driver de Qualcomm (vendorID 0x5143): OpReturn en vez de
    // OpUnreachable tras un BRX sin destino y shuffle emulado.
    options.broken_unreachable = options.broken_subgroup_shuffle = getenv("QUALCOMM") != nullptr;
    // OR=<máscara>: la variante de pixel shader que emula la operación lógica OR en esos render targets (bit k: el k), la
    // de program.cpp (logic_or_variant) en un dispositivo sin logicOp. OR=1: las sondas IBL de Xenoblade (RG32UI en el 0).
    if (const char* mask = getenv("OR")) options.logic_or_targets = static_cast<uint32_t>(std::strtoul(mask, nullptr, 0));
    // UINT=<máscara> / SINT=<máscara>: salidas de color enteras (uvec4/ivec4) en esos render targets, la variante de
    // program.cpp (integer_variant) con un R8UI/RG32UI enlazado. ENTERAS=<banco>:<offset hex>,…: texturas declaradas
    // enteras por el draw (vista de stencil, R8UI…), p. ej. ENTERAS=2:28 para c2[0x28]. SIN_IMAGENES_ENTERAS=1: las
    // imágenes R32UI y los atómicos como antes (float).
    if (const char* mask = getenv("UINT")) options.unsigned_color_targets = static_cast<uint32_t>(std::strtoul(mask, nullptr, 0));
    if (const char* mask = getenv("SINT")) options.signed_color_targets = static_cast<uint32_t>(std::strtoul(mask, nullptr, 0));
    if (const char* list = getenv("ENTERAS"))
        for (const char* p = list; p && *p; p = std::strchr(p, ',') ? std::strchr(p, ',') + 1 : nullptr) {
            unsigned bank = 0, offset = 0;
            if (std::sscanf(p, "%u:%x", &bank, &offset) == 2) options.integer_handles.push_back({bank, offset});
        }
    options.integer_images = getenv("SIN_IMAGENES_ENTERAS") == nullptr;
    // SPIRV_OPT=1: el SPIR-V de los compute por el optimizador, como program.cpp (spirv_opt_stages); SPIRV_OPT=todas: de
    // todas las etapas.
    if (const char* opt = getenv("SPIRV_OPT"))
        options.optimize_stages = !std::strcmp(opt, "todas") ? 0x3Fu : 1u << static_cast<uint32_t>(nvnshader::Stage::Compute);
    std::vector<uint8_t> ctrl;
    if (const char* ctrl_path = getenv("CTRL")) {  // CTRL=<sección de control>: datos de compute como en la Flip
        std::ifstream cf(ctrl_path, std::ios::binary);
        ctrl.assign((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        if (ctrl.size() >= 0x708 && !getenv("SIN_C1")) {  // c1 como en program.cpp (SIN_C1=1: como antes)
            uint32_t begin, end;
            std::memcpy(&begin, &ctrl[0x700], 4);
            std::memcpy(&end, &ctrl[0x704], 4);
            if (end > begin && end - begin <= 0x10000 && end <= code.size()) {
                options.constants = code.data() + begin;
                options.constants_size = end - begin;
            }
        }
        if (ctrl.size() >= 0x758) {
            uint32_t stage = 0;  // como program.cpp: +0x714 == 5 es compute (MH Rise no lleva la cabecera 0x12345679)
            std::memcpy(&stage, &ctrl[0x714], 4);
            options.compute_hint = stage == 5;
            std::memcpy(&options.compute_code_offset, &ctrl[0x708], 4);
            std::memcpy(options.workgroup, &ctrl[0x748], 12);
            std::memcpy(&options.shared_memory, &ctrl[0x754], 4);
            for (uint32_t& w : options.workgroup) w = w ? w : 1;
        }
    }
    if (!nvnshader::translate(code.data(), code.size(), options, t)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], t.error.c_str());
        return 1;
    }
    if (t.shared_to_local) std::printf("memoria compartida privada de cada invocación -> local: %d accesos\n", t.shared_to_local);
    if (t.spirv_opt > 0) std::printf("spirv-opt: %u → %zu palabras, %.1f ms\n", t.spirv_opt_from, t.spirv.size(), t.spirv_opt_us / 1000.0);
    if (t.spirv_opt < 0) std::printf("spirv-opt no pudo: %s (%.1f ms)\n", t.spirv_opt_error.c_str(), t.spirv_opt_us / 1000.0);
    static const char* kKinds[] = {"ubo", "ssbo", "texbuf", "imgbuf", "textura", "imagen"};
    std::printf("%s: %s, %zu palabras de SPIR-V, entradas 0x%llx, colores 0x%x\n", argv[1],
                nvnshader::stage_name(t.stage), t.spirv.size(), static_cast<unsigned long long>(t.input_generics),
                t.color_outputs);
    for (const auto& r : t.resources) {
        if (r.kind == nvnshader::Resource::UniformBuffer)
            std::printf("  binding %u: ubo c%u\n", r.binding, r.cbuf_index);
        else
            std::printf("  %sbinding %u: %s%s, handle en c%u[0x%x]%s%s\n", r.set ? "set 1 " : "", r.binding, kKinds[r.kind],
                        r.is_integer ? " entera" : "",
                        r.cbuf_index, r.cbuf_offset,
                        r.count > 1 || r.handle_and_mask != ~0u || r.handle_or
                            ? [&] {
                                  char b[96];
                                  snprintf(b, sizeof b, " x%u paso %u (& 0x%x | 0x%x) tipo %u%s", r.count, 1u << r.size_shift,
                                           r.handle_and_mask, r.handle_or, r.texture_type, r.is_depth ? " comparación" : "");
                                  return std::string(b);
                              }().c_str()
                            : "",
                        r.has_secondary ? (" | c" + std::to_string(r.secondary_cbuf_index) + "[0x" +
                                           [&] { char b[16]; snprintf(b, sizeof b, "%x", r.secondary_cbuf_offset); return std::string(b); }() +
                                           "] desplazamientos " + std::to_string(r.shift_left) + "/" + std::to_string(r.secondary_shift_left)).c_str()
                                        : "");
    }
    if (!ir.empty()) std::printf("%s\n", ir.c_str());
    if (argc > 2) {
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(t.spirv.data()), t.spirv.size() * sizeof(uint32_t));
    }
    return 0;
}
