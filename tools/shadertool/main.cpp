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
    // PROFUNDIDAD=1: la profundidad de -1..1 pasada a 0..1 al final del vertex shader, como en la Flip con Xenoblade
    // (program.cpp: convert_depth_mode con g_depth_mode 0).
    options.convert_depth_mode = getenv("PROFUNDIDAD") != nullptr;
    // XFB=1: la variante de captura del vertex shader (xfb_variant); XFB_GS=1: la del geometry shader (xfb_gs_variant).
    options.xfb_capture_always = getenv("XFB") != nullptr;
    options.xfb_gs_capture = getenv("XFB_GS") != nullptr;
    // SUBPASS="banco:offset:adjunto,…": esas texturas como input attachment (feedback_variant), p. ej. "2:48:0,2:56:2".
    if (const char* list = getenv("SUBPASS")) {
        unsigned bank = 0, offset = 0, attachment = 0;
        int used = 0;
        for (const char* q = list; std::sscanf(q, "%u:%u:%u%n", &bank, &offset, &attachment, &used) == 3; q += used + (q[used] == ',')) {
            options.input_attachment_handles.push_back({bank, offset, attachment});
            if (!q[used]) break;
        }
    }
    // Como en la Flip: arrays bindless al set 1 y NonUniform (el Adreno 650 lo tiene). SIN_BINDLESS=1 / SIN_NONUNIFORM=1:
    // como antes / sin la decoración.
    options.bindless_arrays = !getenv("SIN_BINDLESS");
    options.nonuniform_indexing = !getenv("SIN_NONUNIFORM");
    // CON_BARRERA=1: deja las barreras de grupo dentro de ifs y bucles, como program.cpp con las huellas de
    // kUniformBarrierShaders o de <datos>/con_barrera (para comparar y validar ese SPIR-V).
    options.keep_conditional_barriers = getenv("CON_BARRERA") != nullptr;
    // SIN_BARRERAS_UNIFORMES=1: como antes, todas las barreras condicionales fuera (también las de flujo uniforme).
    options.uniform_barriers = getenv("SIN_BARRERAS_UNIFORMES") == nullptr;
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
    // GENERICOS=<máscara>: los atributos genéricos que escribe la etapa anterior (Options::previous_stores), como la
    // variante de program.cpp (generics_variant) de un pixel shader que lee genéricos que su vertex no escribe: solo
    // declara las entradas que le llegan y las demás las lee a cero. TOPOLOGIA=<n>: la primitiva de entrada de un geometry
    // shader (InputTopology de Eden: puntos 0, líneas 1, triángulos 3), la de geometry_variant según el draw.
    if (const char* mask = getenv("GENERICOS")) options.previous_stores = std::strtoull(mask, nullptr, 0);
    if (const char* topology = getenv("TOPOLOGIA")) options.input_topology = std::atoi(topology);
    // SUBGRUPOS=<tamaño>:<etapas>:<operaciones>: los subgrupos del dispositivo, lo que program.cpp saca de
    // VkPhysicalDeviceSubgroupProperties. La Flip (Adreno 650) es SUBGRUPOS=64:0x20:0x3f: solo en compute y sin quads.
    if (const char* subgroups = getenv("SUBGRUPOS")) {
        int size = 0, stages = 0, operations = 0;
        if (std::sscanf(subgroups, "%i:%i:%i", &size, &stages, &operations) == 3) {
            options.subgroup_size = static_cast<uint32_t>(size);
            options.subgroup_stages = static_cast<uint32_t>(stages);
            options.subgroup_operations = static_cast<uint32_t>(operations);
        }
    }
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
            // Vertex shader en dos programas (como program.cpp, vertex_b_offset): +0x748 y +0x754 son sus programas.
            uint32_t first = 0, second = 0, second_size = 0;
            std::memcpy(&first, &ctrl[0x748], 4);
            std::memcpy(&second, &ctrl[0x754], 4);
            if (ctrl.size() >= 0x75c) std::memcpy(&second_size, &ctrl[0x758], 4);
            if (stage == 0 && first == 0x30 && second > first && second_size) {
                options.vertex_b_offset = second;
                options.constants = nullptr;  // entre +0x700 y +0x704 está el segundo programa, no constantes
                options.constants_size = 0;
            }
            std::memcpy(&options.compute_code_offset, &ctrl[0x708], 4);
            if (!options.vertex_b_offset) {
                std::memcpy(options.workgroup, &ctrl[0x748], 12);
                std::memcpy(&options.shared_memory, &ctrl[0x754], 4);
            }
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
    // AUDITORIA=1: lo que el texto de arriba no dice y tools/shader_audit.py necesita (una línea por dato, clave=valor):
    // los genéricos que escribe (los de la etapa siguiente: GENERICOS), si tiene un switch (BRX), los bytes que lee de
    // cada banco de constantes y, de cada recurso, su clase, su tipo de textura (Shader::TextureType), de dónde sale su
    // handle (y el del sampler, si va aparte: banco2/offset2) y sus marcas.
    if (getenv("AUDITORIA")) {
        std::printf("auditoria: etapa=%s palabras=%zu entradas=0x%llx salidas=0x%llx colores=0x%x brx=%d barreras=%d "
                    "enteras=%u cbufs=0x%x bindless=%d compartida_a_local=%d barreras_uniformes=%d\n",
                    nvnshader::stage_name(t.stage), t.spirv.size(), static_cast<unsigned long long>(t.input_generics),
                    static_cast<unsigned long long>(t.output_generics), t.color_outputs, t.has_brx,
                    t.conditional_barriers, t.integer_textures, t.constant_buffer_mask, t.has_bindless_set,
                    t.shared_to_local, t.uniform_barriers);
        std::printf("leido:");
        for (unsigned k = 0; k < 18; k++)
            if (t.constant_buffer_used[k]) std::printf(" c%u=%u", k, t.constant_buffer_used[k]);
        std::printf("\n");
        for (const auto& r : t.resources)
            std::printf("recurso: clase=%s set=%u binding=%u banco=%u offset=0x%x tipo=%u cuenta=%u sombra=%d entera=%d "
                        "escrita=%d banco2=%d offset2=0x%x\n",
                        kKinds[r.kind], r.set, r.binding, r.cbuf_index, r.cbuf_offset, r.texture_type, r.count,
                        r.is_depth, r.is_integer, r.is_written, r.has_secondary ? static_cast<int>(r.secondary_cbuf_index) : -1,
                        r.has_secondary ? r.secondary_cbuf_offset : 0);
    }
    if (!ir.empty()) std::printf("%s\n", ir.c_str());
    if (argc > 2) {
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(t.spirv.data()), t.spirv.size() * sizeof(uint32_t));
        if (!t.layer_gs.empty()) {  // el geometry shader de paso de un vertex shader que escribe la capa
            const std::string path = std::string(argv[2]) + ".capa.spv";
            std::ofstream gs(path, std::ios::binary);
            gs.write(reinterpret_cast<const char*>(t.layer_gs.data()), t.layer_gs.size() * sizeof(uint32_t));
            std::printf("  escribe la capa: geometry shader de paso en %s (%zu palabras)\n", path.c_str(), t.layer_gs.size());
        }
    }
    return 0;
}
