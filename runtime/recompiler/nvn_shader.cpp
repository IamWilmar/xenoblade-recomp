// SPDX-License-Identifier: GPL-3.0-or-later
#include "nvn_shader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include <optional>
#include <stdexcept>

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace nvnshader {

namespace {

class NvnEnvironment final : public Shader::Environment {
public:
    // second: el segundo programa de un vertex shader doble (VertexB, en options.vertex_b_offset).
    NvnEnvironment(const uint8_t* code, size_t max_size, const Options& options, bool compute, bool second = false)
        // En compute NVN deja toda la tabla del driver en c0 (handles en 0x168 + 8·i, SSBO en 0x310 + 16·k):
        // los TEX con índice inmediato leen ahí, no en c2 como en gráficos.
        : code_(code), max_size_(max_size), texture_bound_(compute ? 0 : options.texture_bound_buffer), options_(options) {
        if (compute) {
            stage = Shader::Stage::Compute;
            start_address = options.compute_code_offset;
            return;
        }
        start_address = second ? options.vertex_b_offset : kNvnHeaderSize;
        if (start_address + sizeof(sph) > max_size) throw std::runtime_error("cabecera SPH fuera del código");
        std::memcpy(&sph, code_ + start_address, sizeof(sph));
        uint32_t sph_word0 = 0;
        std::memcpy(&sph_word0, code_ + start_address, sizeof sph_word0);
        // El tipo de la cabecera es el índice del programa en Maxwell: 0 VertexA, 1 VertexB, 2-3 teselación, 4 geometry,
        // 5 pixel. El glslc del SDK 8 (Catherine: Full Body, código de GPU 1.15) parte cada vertex shader en dos: VertexA
        // (cabecera VTG, bits 0-4 = 1, con tipo 0) y VertexB (Options::vertex_b_offset). Sin segundo programa un VTG con
        // tipo 0 se traduce como el vertex shader entero.
        const bool vertex_a = (sph_word0 & 0x1F) == 1 && ((sph_word0 >> 10) & 0xF) == 0;
        if (vertex_a && !second && options.vertex_b_offset) {
            stage = Shader::Stage::VertexA;
            return;
        }
        switch (vertex_a ? 1u : static_cast<unsigned>(sph.common0.shader_type)) {
            case 1: stage = Shader::Stage::VertexB; break;
            case 2: stage = Shader::Stage::TessellationControl; break;
            case 3: stage = Shader::Stage::TessellationEval; break;
            case 4: stage = Shader::Stage::Geometry; break;
            case 5: stage = Shader::Stage::Fragment; break;
            default: stage = Shader::Stage::Compute; break;
        }
    }

    u64 ReadInstruction(u32 address) override {
        u64 v = 0;
        if (address + sizeof v <= max_size_) std::memcpy(&v, code_ + address, sizeof v);
        return v;
    }
    // Los valores de los constant buffers solo existen al dibujar, salvo c1 (constantes del propio programa):
    // ahí están las tablas de salto de los BRX, que Eden necesita para encontrar los destinos del switch.
    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (index == 1) brx_ = true;  // Eden solo lee c1 al traducir para las tablas de un BRX (switch)
        u32 v = 0;
        if (index == 1 && options_.constants && offset + sizeof v <= options_.constants_size)
            std::memcpy(&v, options_.constants + offset, sizeof v);
        return v;
    }
    bool HasBrx() const { return brx_; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    // Un texelFetch de "1D" es de un buffer (samplerBuffer): es lo que emite el compilador de NVN para Buffer<T>.Load, y
    // la instrucción no distingue. Unreal Engine 5 lee así las tangentes, las UV y los colores de cada vértice en el
    // vertex shader (manual vertex fetch: el único atributo es la posición) y los datos de escena por instancia: como
    // textura 1D se quedaban sin enlazar (ceros: normales NaN, todo negro y sin texturas). Una 1D de verdad leída con
    // texelFetch saldría a ceros (el ejecutor lo avisa: "sin buffer enlazado"); Options::fetch_1d_is_texture lo quita.
    bool Fetch1DIsBuffer() const override { return !options_.fetch_1d_is_texture; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    // Las texturas que el shader lee como enteros (ver integer_textures): por la dirección de su handle.
    bool IsTextureIntegerAt(u32 cbuf_index, u32 cbuf_offset, u32) override {
        return integer_textures_.contains({cbuf_index, cbuf_offset});
    }
    void SetIntegerTextures(std::set<std::pair<u32, u32>> textures) { integer_textures_ = std::move(textures); }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound_; }
    // Gráficos: lo dice la cabecera (SPH), como en Eden con un GPU de verdad (más la pila CRS). Compute no
    // tiene SPH: lo que diga la sección de control o, si no, un margen (solo cuenta si el shader la usa).
    u32 LocalMemorySize() const override {
        if (stage != Shader::Stage::Compute)
            return static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        return options_.local_memory ? options_.local_memory : 1024;
    }
    u32 SharedMemorySize() const override { return options_.shared_memory; }
    std::array<u32, 3> WorkgroupSize() const override {
        return {options_.workgroup[0], options_.workgroup[1], options_.workgroup[2]};
    }
    // Parámetros del draw que el driver de NVN deja en su c0 (las macros que Eden emula): base vertex en 0x640, base
    // instance en 0x644 y el número de draw de un multi-draw en 0x648 (gl_BaseVertexARB, gl_BaseInstanceARB y
    // gl_DrawIDARB del GLSL de NVN), que salen de Vulkan (gl_BaseVertex, gl_BaseInstance, gl_DrawID). MH Rise usa 0x640
    // como VertexId − c0[0x640] (traza tf1, draws 14, 182 y 192): el índice sin base. VertexId es gl_VertexIndex, que
    // lleva el vertexOffset que usemos (desplazado a la ventana de vértices en los draws directos, el del juego en los
    // indirectos): restar gl_BaseVertex lo quita en todos los casos. Con 0x640 a cero (antes) un draw directo desplazado
    // leía el vértice índice − mínimo y un multi-draw con base vertex 1302, el 1302 + i.
    bool HasHLEMacroState() const override { return stage == Shader::Stage::VertexB; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        if (bank != 0 || stage != Shader::Stage::VertexB) return std::nullopt;
        if (offset == 0x640) return Shader::ReplaceConstant::BaseVertex;
        if (offset == 0x644) return Shader::ReplaceConstant::BaseInstance;
        if (offset == 0x648) return Shader::ReplaceConstant::DrawID;
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    const uint8_t* code_;
    bool brx_ = false;
    size_t max_size_;
    uint32_t texture_bound_;
    const Options& options_;
    std::set<std::pair<u32, u32>> integer_textures_;
};

// Texturas que el GLSL del juego declara enteras (usampler/isampler): en el código de Maxwell no se distinguen (TLD da
// los bits en los registros), pero todo lo que sale de sus lecturas se reinterpreta como u32 (BitCastU32F32). Eden, sin
// saberlo, las declara float, y una vista de stencil o de un formato entero leída como float da 0 en el Adreno: la
// máscara de materiales de Xenoblade lee así el stencil del G-buffer y sin él nada recibía luz. Solo los texelFetch
// (ImageFetch) cuyos componentes se usan todos como enteros; un solo uso como float deja la textura en float.
// Imágenes (integer_images): las que tienen atómicos o un formato entero en la instrucción (SULD.D/SUST.D: todos los
// ImageFormat de Eden menos Typeless son enteros) se declaran enteras por su dirección, así que también las demás
// instrucciones sobre la misma imagen (el SUST.P sin formato de las sondas IBL escribe el R32UI que lee el SULD R32UI).
std::set<std::pair<u32, u32>> integer_textures(Shader::IR::Program& program, bool integer_images) {
    using Shader::IR::Inst;
    using Shader::IR::Opcode;
    std::unordered_map<const Inst*, std::vector<const Inst*>> users;
    for (Shader::IR::Block* block : program.blocks)
        for (Inst& inst : block->Instructions())
            for (size_t k = 0; k < inst.NumArgs(); k++)
                if (const Shader::IR::Value arg = inst.Arg(k); !arg.IsImmediate()) users[arg.Inst()].push_back(&inst);
    // Usos de una instrucción saltando los Identity.
    auto uses_of = [&](const Inst* inst, auto&& self, std::vector<const Inst*>& out) -> void {
        for (const Inst* u : users[inst]) {
            if (u->GetOpcode() == Opcode::Identity)
                self(u, self, out);
            else
                out.push_back(u);
        }
    };
    std::unordered_map<u32, bool> by_descriptor;  // descriptor -> ¿todas sus lecturas como enteros?
    // Lo mismo para los buffers leídos con texelFetch (sus descriptores van aparte). Unreal Engine 5 lee así la rejilla de
    // luces de sus pixel shaders de iluminación (R16UI) y los índices ordenados de las partículas (R32UI): declarados
    // float, lo que sale de un formato entero es indefinido (luces que parpadean).
    std::unordered_map<u32, bool> by_buffer_descriptor;
    std::set<u32> integer_image_descriptors;
    for (Shader::IR::Block* block : program.blocks)
        for (Inst& inst : block->Instructions()) {
            if (integer_images) {
                const std::string_view name = Shader::IR::NameOf(inst.GetOpcode());
                const bool atomic = name.starts_with("ImageAtomic");
                if (atomic || inst.GetOpcode() == Opcode::ImageRead || inst.GetOpcode() == Opcode::ImageWrite) {
                    const auto info = inst.Flags<Shader::IR::TextureInstInfo>();
                    if (info.type != Shader::TextureType::Buffer &&
                        (atomic || info.image_format != Shader::ImageFormat::Typeless))
                        integer_image_descriptors.insert(info.descriptor_index);
                    continue;
                }
            }
            if (inst.GetOpcode() != Opcode::ImageFetch) continue;
            const auto info = inst.Flags<Shader::IR::TextureInstInfo>();
            const bool buffer = info.type == Shader::TextureType::Buffer;
            bool integer = true, any = false;
            std::vector<const Inst*> extracts;
            uses_of(&inst, uses_of, extracts);
            for (const Inst* e : extracts) {
                if (e->GetOpcode() != Opcode::CompositeExtractF32x4) {
                    integer = false;
                    break;
                }
                std::vector<const Inst*> consumers;
                uses_of(e, uses_of, consumers);
                for (const Inst* c : consumers) {
                    any = true;
                    if (c->GetOpcode() != Opcode::BitCastU32F32) integer = false;
                }
            }
            auto [it, inserted] = (buffer ? by_buffer_descriptor : by_descriptor).try_emplace(info.descriptor_index, integer && any);
            if (!inserted) it->second = it->second && integer && any;
        }
    std::set<std::pair<u32, u32>> out;
    const auto& descriptors = program.info.texture_descriptors;
    for (const auto& [index, integer] : by_descriptor)
        if (integer && index < descriptors.size()) out.insert({descriptors[index].cbuf_index, descriptors[index].cbuf_offset});
    const auto& buffers = program.info.texture_buffer_descriptors;
    for (const auto& [index, integer] : by_buffer_descriptor)
        if (integer && index < buffers.size()) out.insert({buffers[index].cbuf_index, buffers[index].cbuf_offset});
    const auto& images = program.info.image_descriptors;
    for (u32 index : integer_image_descriptors)
        if (index < images.size()) out.insert({images[index].cbuf_index, images[index].cbuf_offset});
    return out;
}

Stage to_stage(Shader::Stage s) {
    switch (s) {
        case Shader::Stage::VertexA:
        case Shader::Stage::VertexB: return Stage::Vertex;
        case Shader::Stage::TessellationControl: return Stage::TessControl;
        case Shader::Stage::TessellationEval: return Stage::TessEval;
        case Shader::Stage::Geometry: return Stage::Geometry;
        case Shader::Stage::Fragment: return Stage::Fragment;
        default: return Stage::Compute;
    }
}

// Mismo orden y mismo contador que EmitContext con bindings unificados y sin aliasing de
// descriptores: UBOs y SSBOs avanzan `count`, el resto uno por descriptor.
void list_resources(const Shader::Info& info, uint32_t binding, Translated& out) {
    for (const auto& d : info.constant_buffer_descriptors) {
        for (uint32_t k = 0; k < d.count; k++)
            out.resources.push_back({Resource::UniformBuffer, binding++, d.index + k, 0, 0});
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        for (uint32_t k = 0; k < d.count; k++) {
            Resource r{Resource::StorageBuffer, binding++, d.cbuf_index, d.cbuf_offset + k * 16, 0};
            r.is_written = d.is_written;
            out.resources.push_back(r);
        }
    }
    for (const auto& d : info.texture_buffer_descriptors) {
        Resource r{Resource::TextureBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0};
        r.is_integer = d.is_integer;  // usamplerBuffer: el ejecutor le da una vista de formato entero
        out.resources.push_back(r);
    }
    for (const auto& d : info.image_buffer_descriptors) {
        // Como las imágenes: si el shader lo escribe (lo que escribe se queda en la GPU) y si lo declara entero.
        Resource r{Resource::ImageBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0};
        r.is_written = d.is_written;
        r.is_integer = d.is_integer;
        out.resources.push_back(r);
    }
    for (const auto& d : info.texture_descriptors) {
        Resource r{Resource::Texture, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_depth = d.is_depth;
        r.is_integer = d.is_integer;
        r.input_attachment = static_cast<int8_t>(d.input_attachment);
        r.has_secondary = d.has_secondary;
        r.secondary_cbuf_index = d.secondary_cbuf_index;
        r.secondary_cbuf_offset = d.secondary_cbuf_offset;
        r.shift_left = d.shift_left;
        r.secondary_shift_left = d.secondary_shift_left;
        r.count = d.count;
        r.size_shift = d.size_shift;
        r.handle_and_mask = d.handle_and_mask;
        r.handle_or = d.handle_or;
        out.resources.push_back(r);
    }
    for (const auto& d : info.image_descriptors) {
        Resource r{Resource::Image, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_written = d.is_written;
        r.is_integer = d.is_integer;
        out.resources.push_back(r);
    }
}

// Los arrays de texturas (count > 1) al set 1, con bindings etapa · 16 + j en el orden de los recursos: en el SPIR-V se
// cambian sus decoraciones Binding y DescriptorSet. Así el layout del set 1 depende solo de cuántos arrays hay y de qué
// tamaño, y dos shaders que leen los mismos handles comparten el set (el ejecutor lo arma una vez por envío: 3 arrays de
// 512 y uno de 223 por pixel shader de MH Rise no se pueden escribir en cada draw).
void move_arrays_to_bindless_set(Translated& out) {
    std::vector<std::pair<uint32_t, uint32_t>> moved;  // (binding de Eden, binding en el set 1)
    const uint32_t first = static_cast<uint32_t>(out.stage) * kBindlessPerStage;
    for (Resource& r : out.resources)
        if (r.kind == Resource::Texture && r.count > 1 && moved.size() < kBindlessPerStage) {
            moved.push_back({r.binding, first + static_cast<uint32_t>(moved.size())});
            r.binding = moved.back().second;
            r.set = kBindlessSet;
        }
    if (moved.empty()) return;
    out.has_bindless_set = true;
    constexpr uint32_t kOpDecorate = 71, kBinding = 33, kDescriptorSet = 34;
    std::vector<uint32_t> targets;  // variables movidas
    std::vector<uint32_t>& w = out.spirv;
    for (int pass = 0; pass < 2; pass++)
        for (size_t k = 5; k < w.size();) {
            const uint32_t count = w[k] >> 16, op = w[k] & 0xFFFF;
            if (!count) break;
            if (op == kOpDecorate && count >= 4) {
                const uint32_t target = w[k + 1], decoration = w[k + 2];
                if (pass == 0 && decoration == kBinding)
                    for (const auto& [from, to] : moved)
                        if (w[k + 3] == from) {
                            w[k + 3] = to;
                            targets.push_back(target);
                            break;
                        }
                if (pass == 1 && decoration == kDescriptorSet &&
                    std::find(targets.begin(), targets.end(), target) != targets.end())
                    w[k + 3] = kBindlessSet;
            }
            k += count;
        }
}

// Valor de una expresión entera del IR que solo depende de LocalInvocationId y de inmediatos, para la invocación lid.
// Falso si depende de cualquier otra cosa (cbufs, memoria, phis…) o se sale de lo que se sabe evaluar.
bool eval_invocation(const Shader::IR::Value& value, const uint32_t lid[3], uint32_t& out, int depth = 0) {
    using Shader::IR::Opcode;
    if (depth > 64) return false;
    const Shader::IR::Value v = value.Resolve();
    if (v.IsImmediate()) {
        if (v.Type() != Shader::IR::Type::U32) return false;
        out = v.U32();
        return true;
    }
    const Shader::IR::Inst* inst = v.Inst();
    const Opcode op = inst->GetOpcode();
    if (op == Opcode::CompositeExtractU32x3) {
        const Shader::IR::Value vec = inst->Arg(0).Resolve(), index = inst->Arg(1).Resolve();
        if (vec.IsImmediate() || vec.Inst()->GetOpcode() != Opcode::LocalInvocationId || !index.IsImmediate() ||
            index.U32() > 2)
            return false;
        out = lid[index.U32()];
        return true;
    }
    uint32_t a = 0, b = 0, c = 0;
    const size_t n = inst->NumArgs();
    if (n < 2 || n > 3 || !eval_invocation(inst->Arg(0), lid, a, depth + 1) ||
        !eval_invocation(inst->Arg(1), lid, b, depth + 1) || (n == 3 && !eval_invocation(inst->Arg(2), lid, c, depth + 1)))
        return false;
    switch (op) {
        case Opcode::IAdd32: out = a + b; return true;
        case Opcode::ISub32: out = a - b; return true;
        case Opcode::IMul32: out = a * b; return true;
        case Opcode::BitwiseAnd32: out = a & b; return true;
        case Opcode::BitwiseOr32: out = a | b; return true;
        case Opcode::BitwiseXor32: out = a ^ b; return true;
        case Opcode::ShiftLeftLogical32: out = a << b; return b < 32;
        case Opcode::ShiftRightLogical32: out = a >> b; return b < 32;
        case Opcode::ShiftRightArithmetic32: out = static_cast<uint32_t>(static_cast<int32_t>(a) >> b); return b < 32;
        case Opcode::BitFieldUExtract:
            if (b + c > 32) return false;
            out = c == 0 ? 0 : (a >> b) & (c == 32 ? ~0u : (1u << c) - 1);
            return true;
        default: return false;
    }
}

// Memoria compartida que en realidad es privada de cada invocación: el compilador de NVN derrama registros a la memoria
// compartida (STS/LDS en tid·28 + k en el skinning de la cara de MH Rise, 793657b4f1e65631: el índice del 4.º hueso y
// las componentes que no le caben en registros). Si todos los accesos son base(invocación) + k, con la misma base para
// la invocación en todos y ventanas [base + kmin, base + kmax + 4) que no se solapan entre las invocaciones del grupo,
// ninguna lee lo que escribe otra (con o sin barreras): es memoria privada y se pasa a la memoria local de Eden
// (Private con índices constantes, que el driver deja en registros). Con esos stores a Workgroup dentro de los casos de
// su switch (BRX), el driver de Qualcomm (512.746) deja la salida de la Flip igual, bit a bit, a la de ese SPIR-V con los
// 4 pesos del skinning a 0 (los del caso 1272, que los pone a 0): suma 0, 1/0 = inf, inf·0 = NaN en todo el grupo; un
// intérprete de SPIR-V con los mismos buffers da posiciones finitas. El skinning hermano sin memoria compartida
// (6aaf54a71876a3f2, mismo switch y mismos casos) sale bien. Devuelve los accesos cambiados (0: no aplica).
int shared_to_local(Shader::IR::Program& program, const std::array<u32, 3>& workgroup) {
    using Shader::IR::Opcode;
    if (program.stage != Shader::Stage::Compute || program.shared_memory_size == 0) return 0;
    struct Access {
        Shader::IR::Inst* inst;
        Shader::IR::Value base;
        uint32_t imm;
    };
    std::vector<Access> accesses;
    bool uses_local = false;
    for (Shader::IR::Block* block : program.post_order_blocks)
        for (Shader::IR::Inst& inst : block->Instructions()) {
            const Opcode op = inst.GetOpcode();
            if (op == Opcode::LoadLocal || op == Opcode::WriteLocal) uses_local = true;
            if (op != Opcode::LoadSharedU32 && op != Opcode::WriteSharedU32) {
                if (Shader::IR::NameOf(op).find("Shared") != std::string_view::npos) return 0;  // otros tamaños, atómicos
                continue;
            }
            const Shader::IR::Value offset = inst.Arg(0).Resolve();
            Shader::IR::Value base = offset;
            uint32_t imm = 0;
            if (!offset.IsImmediate() && offset.Inst()->GetOpcode() == Opcode::IAdd32) {
                const Shader::IR::Value x = offset.Inst()->Arg(0).Resolve(), y = offset.Inst()->Arg(1).Resolve();
                if (y.IsImmediate() && y.Type() == Shader::IR::Type::U32) base = x, imm = y.U32();
                else if (x.IsImmediate() && x.Type() == Shader::IR::Type::U32) base = y, imm = x.U32();
            }
            if (imm % 4 || imm >= 4096) return 0;
            accesses.push_back({&inst, base, imm});
        }
    if (accesses.empty()) return 0;
    uint32_t lo = ~0u, hi = 0;
    for (const Access& a : accesses) lo = std::min(lo, a.imm), hi = std::max(hi, a.imm + 4);
    const uint32_t window = hi - lo;
    std::vector<uint32_t> bases;
    for (uint32_t z = 0; z < workgroup[2]; z++)
        for (uint32_t y = 0; y < workgroup[1]; y++)
            for (uint32_t x = 0; x < workgroup[0]; x++) {
                const uint32_t lid[3] = {x, y, z};
                uint32_t first = 0;
                for (size_t k = 0; k < accesses.size(); k++) {
                    uint32_t b = 0;
                    if (!eval_invocation(accesses[k].base, lid, b) || (k && b != first)) return 0;
                    first = b;
                }
                // Fuera de la memoria compartida declarada (lo que haría el original es indefinido): no se toca.
                if (uint64_t{first} + hi > program.shared_memory_size) return 0;
                bases.push_back(first);
            }
    std::sort(bases.begin(), bases.end());
    for (size_t k = 1; k < bases.size(); k++)
        if (bases[k] - bases[k - 1] < window) return 0;
    // La ventana va tras la memoria local que ya use el programa (LDL/STL), o sola (la de 1 KB por defecto no se usaba).
    const uint32_t first_word = uses_local ? (program.local_memory_size + 3) / 4 : 0;
    for (const Access& a : accesses) {
        const bool load = a.inst->GetOpcode() == Opcode::LoadSharedU32;
        a.inst->ReplaceOpcode(load ? Opcode::LoadLocal : Opcode::WriteLocal);
        a.inst->SetArg(0, Shader::IR::Value{first_word + (a.imm - lo) / 4});
    }
    program.local_memory_size = first_word * 4 + window;
    program.shared_memory_size = 0;
    program.info.uses_local_memory = true;
    return static_cast<int>(accesses.size());
}

// Transform feedback emulado (ver kXfbBank en nvn_shader.h). Un vertex shader que escribe atributos genéricos y no la
// posición solo sirve para capturar sus salidas (con el rasterizador descartado no sale nada más): antes del Epilogue
// guarda las componentes genéricas escritas, seguidas y en el orden de los atributos, en un SSBO nuevo, en
// c0[kXfbParams + 8] + (InstanceId · c0[kXfbParams] + VertexId) · c0[kXfbParams + 4] + 4 · componente. VertexId es
// gl_VertexIndex, relativo a la ventana de vértices del draw directo (empieza en 0, como el orden de captura). Solo
// programas sin control de flujo: la última escritura de cada componente es su valor final. Devuelve el stride en bytes
// (0: no es de captura). El de la hierba alta de Xenoblade (24431a60fdbc6c96) escribe los genéricos 0 y 1 (xyz):
// posición y posición anterior deformadas, 24 bytes, el stride del stream 1 de los draws que los leen.
// force (Options::xfb_capture_always, la variante de captura que pide el ejecutor cuando hay transform feedback activo
// y el shader del juego no la llevaba): también con control de flujo y aunque escriba la posición. Cada SetAttribute de
// un genérico guarda además su valor en su sitio del SSBO justo después (la última que se ejecute es la que queda), en
// vez de una sola vez antes del Epilogue. Xenoblade 3 simula así sus partículas: cada frame dibuja con
// DrawTransformFeedback lo capturado el anterior (puntos, rasterizador descartado) y captura el resultado en el otro
// buffer; con un shader que no se capturaba la cuenta se quedaba en 0 y las partículas no salían nunca.
uint32_t add_xfb_capture_forced(Shader::IR::Program& program);

uint32_t add_xfb_capture(Shader::IR::Program& program, bool force = false) {
    namespace IR = Shader::IR;
    if (program.stage != Shader::Stage::VertexB) return 0;
    if (force) return add_xfb_capture_forced(program);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list)
        if (node.type != IR::AbstractSyntaxNode::Type::Block && node.type != IR::AbstractSyntaxNode::Type::Return) return 0;
    std::array<IR::Value, 128> last{};
    std::array<bool, 128> written{};
    IR::Block* end_block = nullptr;
    IR::Block::iterator end_it;
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type != IR::AbstractSyntaxNode::Type::Block) continue;
        IR::Block* block = node.data.block;
        for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() == IR::Opcode::Epilogue) end_block = block, end_it = it;
            if (it->GetOpcode() != IR::Opcode::SetAttribute) continue;
            const IR::Attribute a = it->Arg(0).Attribute();
            if (a >= IR::Attribute::PositionX && a <= IR::Attribute::PositionW) return 0;  // rasteriza: no es de captura
            if (!IR::IsGeneric(a)) continue;
            const size_t k = static_cast<size_t>(a) - static_cast<size_t>(IR::Attribute::Generic0X);
            last[k] = it->Arg(1);
            written[k] = true;
        }
    }
    uint32_t components = 0;
    for (bool w : written) components += w;
    if (!end_block || !components) return 0;
    Shader::Info& info = program.info;
    const uint32_t binding = static_cast<uint32_t>(info.storage_buffers_descriptors.size());
    IR::IREmitter ir{*end_block, end_it};
    const IR::U32 c0 = ir.Imm32(0);
    const IR::U32 vertices = ir.GetCbuf(c0, ir.Imm32(kXfbParams));
    const IR::U32 stride = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 4));
    const IR::U32 base = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 8));
    const IR::U32 index{ir.IAdd(ir.IMul(ir.GetAttributeU32(IR::Attribute::InstanceId), vertices),
                                ir.GetAttributeU32(IR::Attribute::VertexId))};
    const IR::U32 start{ir.IAdd(base, ir.IMul(index, stride))};
    uint32_t offset = 0;
    for (size_t k = 0; k < written.size(); k++) {
        if (!written[k]) continue;
        const IR::U32 bits = ir.BitCast<IR::U32, IR::F32>(IR::F32{last[k]});
        const IR::U32 at{ir.IAdd(start, ir.Imm32(offset))};
        end_block->PrependNewInst(end_it, IR::Opcode::WriteStorage32, {IR::Value{binding}, at, bits});
        offset += 4;
    }
    // Lo que habría anotado CollectShaderInfoPass: el SSBO, c0 (hasta los parámetros) y las entradas de sistema.
    info.storage_buffers_descriptors.push_back({kXfbBank, 0, 1, true});
    info.used_storage_buffer_types |= IR::Type::U32;
    info.used_constant_buffer_types |= IR::Type::U32;
    if (!(info.constant_buffer_mask & 1u)) {
        info.constant_buffer_mask |= 1u;
        info.constant_buffer_descriptors.insert(info.constant_buffer_descriptors.begin(), Shader::ConstantBufferDescriptor{0, 1});
    }
    info.constant_buffer_used_sizes[0] = std::max<u32>(info.constant_buffer_used_sizes[0], kXfbParams + 16);
    info.loads.Set(IR::Attribute::VertexId);
    info.loads.Set(IR::Attribute::InstanceId);
    return offset;
}

uint32_t add_xfb_capture_forced(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    // Las componentes genéricas que escribe el programa (en cualquier bloque), en orden: su sitio en el vértice capturado.
    std::array<bool, 128> written{};
    std::vector<std::pair<IR::Block*, IR::Block::iterator>> sets;
    for (IR::Block* block : program.blocks)
        for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() != IR::Opcode::SetAttribute) continue;
            const IR::Attribute a = it->Arg(0).Attribute();
            if (!IR::IsGeneric(a)) continue;
            written[static_cast<size_t>(a) - static_cast<size_t>(IR::Attribute::Generic0X)] = true;
            sets.push_back({block, it});
        }
    std::array<uint32_t, 128> offset_of{};
    uint32_t offset = 0;
    for (size_t k = 0; k < written.size(); k++)
        if (written[k]) offset_of[k] = offset, offset += 4;
    if (!offset) return 0;
    Shader::Info& info = program.info;
    const uint32_t binding = static_cast<uint32_t>(info.storage_buffers_descriptors.size());
    for (auto& [block, it] : sets) {
        const IR::Attribute a = it->Arg(0).Attribute();
        const size_t k = static_cast<size_t>(a) - static_cast<size_t>(IR::Attribute::Generic0X);
        const IR::Value value = it->Arg(1);
        const auto next = std::next(it);
        IR::IREmitter ir{*block, next};
        const IR::U32 c0 = ir.Imm32(0);
        const IR::U32 vertices = ir.GetCbuf(c0, ir.Imm32(kXfbParams));
        const IR::U32 stride = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 4));
        const IR::U32 base = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 8));
        const IR::U32 index{ir.IAdd(ir.IMul(ir.GetAttributeU32(IR::Attribute::InstanceId), vertices),
                                    ir.GetAttributeU32(IR::Attribute::VertexId))};
        const IR::U32 at{ir.IAdd(ir.IAdd(base, ir.IMul(index, stride)), ir.Imm32(offset_of[k]))};
        const IR::U32 bits = ir.BitCast<IR::U32, IR::F32>(IR::F32{value});
        block->PrependNewInst(next, IR::Opcode::WriteStorage32, {IR::Value{binding}, at, bits});
    }
    info.storage_buffers_descriptors.push_back({kXfbBank, 0, 1, true});
    info.used_storage_buffer_types |= IR::Type::U32;
    info.used_constant_buffer_types |= IR::Type::U32;
    if (!(info.constant_buffer_mask & 1u)) {
        info.constant_buffer_mask |= 1u;
        info.constant_buffer_descriptors.insert(info.constant_buffer_descriptors.begin(), Shader::ConstantBufferDescriptor{0, 1});
    }
    info.constant_buffer_used_sizes[0] = std::max<u32>(info.constant_buffer_used_sizes[0], kXfbParams + 16);
    info.loads.Set(IR::Attribute::VertexId);
    info.loads.Set(IR::Attribute::InstanceId);
    return offset;
}

// Transform feedback desde un geometry shader (Options::xfb_gs_capture). Las salidas de un GS valen lo último que se
// escribió antes de cada EmitVertex, que puede estar en otro bloque: cada SetAttribute de un genérico guarda además su
// valor en memoria local (una palabra por componente, tras la que ya use el programa) y cada EmitVertex del stream 0
// hace, justo antes: i = atomicAdd(contador, 1); si i >= capacidad, deshace su +1 (atomicAdd(contador, -1): el contador
// acaba en min(emitidos, capacidad), como el hardware, que deja de escribir con el buffer lleno) y escribe en el último
// hueco (sin flujo de control nuevo; solo pasa si el juego desborda su buffer); las componentes van de la memoria local
// a base + i · stride. El orden de los vértices capturados no es el de las primitivas (lo decide el atómico): da igual
// para puntos sueltos, que es lo que capturan los GS de partículas. Devuelve el stride en bytes (0: nada que capturar).
uint32_t add_xfb_capture_gs(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    if (program.stage != Shader::Stage::Geometry) return 0;
    std::array<bool, 128> written{};
    std::vector<std::pair<IR::Block*, IR::Block::iterator>> sets, emits;
    for (IR::Block* block : program.blocks)
        for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() == IR::Opcode::EmitVertex) {
                const IR::Value stream = it->Arg(0);
                if (stream.IsImmediate() && stream.U32() == 0) emits.push_back({block, it});
                continue;
            }
            if (it->GetOpcode() != IR::Opcode::SetAttribute) continue;
            const IR::Attribute a = it->Arg(0).Attribute();
            if (!IR::IsGeneric(a)) continue;
            written[static_cast<size_t>(a) - static_cast<size_t>(IR::Attribute::Generic0X)] = true;
            sets.push_back({block, it});
        }
    std::array<uint32_t, 128> offset_of{};
    uint32_t offset = 0;
    for (size_t k = 0; k < written.size(); k++)
        if (written[k]) offset_of[k] = offset, offset += 4;
    if (!offset || emits.empty()) return 0;
    Shader::Info& info = program.info;
    const uint32_t data_binding = static_cast<uint32_t>(info.storage_buffers_descriptors.size());
    const uint32_t count_binding = data_binding + 1;
    const uint32_t first_word = info.uses_local_memory ? (program.local_memory_size + 3) / 4 : 0;
    for (auto& [block, it] : sets) {
        const size_t k = static_cast<size_t>(it->Arg(0).Attribute()) - static_cast<size_t>(IR::Attribute::Generic0X);
        const IR::Value value = it->Arg(1);
        IR::IREmitter ir{*block, std::next(it)};
        ir.WriteLocal(ir.Imm32(first_word + offset_of[k] / 4), ir.BitCast<IR::U32, IR::F32>(IR::F32{value}));
    }
    for (auto& [block, it] : emits) {
        IR::IREmitter ir{*block, it};  // delante del EmitVertex
        const IR::U32 c0 = ir.Imm32(0);
        const IR::U32 capacity = ir.GetCbuf(c0, ir.Imm32(kXfbParams));
        const IR::U32 stride = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 4));
        const IR::U32 base = ir.GetCbuf(c0, ir.Imm32(kXfbParams + 8));
        const IR::U32 index{IR::Value{&*block->PrependNewInst(it, IR::Opcode::StorageAtomicIAdd32,
                                                                {IR::Value{count_binding}, ir.Imm32(0), ir.Imm32(1)})}};
        const IR::U1 full = ir.IGreaterThanEqual(index, capacity, false);
        const IR::U32 undo{ir.Select(full, ir.Imm32(0xffffffffu), ir.Imm32(0u))};
        block->PrependNewInst(it, IR::Opcode::StorageAtomicIAdd32, {IR::Value{count_binding}, ir.Imm32(0), undo});
        const IR::U32 slot = ir.UMin(index, ir.ISub(capacity, ir.Imm32(1)));
        const IR::U32 start{ir.IAdd(base, ir.IMul(slot, stride))};
        for (size_t k = 0; k < written.size(); k++) {
            if (!written[k]) continue;
            const IR::U32 bits = ir.LoadLocal(ir.Imm32(first_word + offset_of[k] / 4));
            const IR::U32 at{ir.IAdd(start, ir.Imm32(offset_of[k]))};
            block->PrependNewInst(it, IR::Opcode::WriteStorage32, {IR::Value{data_binding}, at, bits});
        }
    }
    info.storage_buffers_descriptors.push_back({kXfbBank, 0, 1, true});
    info.storage_buffers_descriptors.push_back({kXfbCountBank, 0, 1, true});
    info.used_storage_buffer_types |= IR::Type::U32;
    info.used_constant_buffer_types |= IR::Type::U32;
    if (!(info.constant_buffer_mask & 1u)) {
        info.constant_buffer_mask |= 1u;
        info.constant_buffer_descriptors.insert(info.constant_buffer_descriptors.begin(), Shader::ConstantBufferDescriptor{0, 1});
    }
    info.constant_buffer_used_sizes[0] = std::max<u32>(info.constant_buffer_used_sizes[0], kXfbParams + 16);
    program.local_memory_size = std::max<u32>(program.local_memory_size, (first_word + offset / 4) * 4);
    info.uses_local_memory = true;
    return offset;
}

// Options::logic_or_targets: cada SetFragColor(k, c, v) de esos render targets pasa a escribir previo[c] | bits(v), con
// previo = texelFetch(usampler2D nueva, ivec2(gl_FragCoord.xy), 0): la operación lógica OR de NVN hecha en el shader.
// La textura (entera, banco kLogicOrBank, offset k) va tras las del juego: list_resources la da como un Resource más y
// el ejecutor le enlaza la copia del render target k de antes del draw. Una lectura por render target y bloque, antes
// de su primera escritura (domina a las de detrás en el bloque). Devuelve las escrituras cambiadas.
// Xenoblade (sondas IBL, 6b9edd01557d4e72 y 31714f9a79239621): una sonda escribe su índice y su peso en el byte
// popcount(máscara R32UI) del RG32UI y 0 si ya estaba o no caben más; sin el OR cada draw pisaba lo de los anteriores.
int emulate_logic_or(Shader::IR::Program& program, uint32_t targets) {
    using Shader::IR::Opcode;
    if (program.stage != Shader::Stage::Fragment || !targets) return 0;
    int descriptor[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // índice de la textura añadida para cada render target
    int changed = 0;
    for (Shader::IR::Block* block : program.blocks) {
        Shader::IR::Value previous[8];  // la lectura de cada render target en este bloque
        for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
            Shader::IR::Inst& inst = *it;
            if (inst.GetOpcode() != Opcode::SetFragColor) continue;
            const Shader::IR::Value index = inst.Arg(0), component = inst.Arg(1);
            if (!index.IsImmediate() || !component.IsImmediate()) continue;
            const u32 k = index.U32(), c = component.U32();
            if (k >= 8 || !(targets >> k & 1) || c > 3) continue;
            if (descriptor[k] < 0) {
                Shader::TextureDescriptor d{};
                d.type = Shader::TextureType::Color2D;
                d.is_integer = true;  // usampler2D: los bits tal cual
                d.cbuf_index = kLogicOrBank;
                d.cbuf_offset = k;
                d.count = 1;
                descriptor[k] = static_cast<int>(program.info.texture_descriptors.size());
                program.info.texture_descriptors.push_back(d);
            }
            Shader::IR::IREmitter ir{*block, it};  // delante de esta escritura
            if (previous[k].IsEmpty()) {
                const Shader::IR::U32 x{ir.ConvertFToU(32, ir.GetAttribute(Shader::IR::Attribute::PositionX))};
                const Shader::IR::U32 y{ir.ConvertFToU(32, ir.GetAttribute(Shader::IR::Attribute::PositionY))};
                const Shader::IR::Value coords{ir.CompositeConstruct(x, y)};
                Shader::IR::TextureInstInfo info{};
                info.type.Assign(Shader::TextureType::Color2D);
                info.descriptor_index.Assign(static_cast<u32>(descriptor[k]));
                // Ya con su descriptor (como las deja texture_pass): sin handle, sin offset, nivel 0, sin muestra.
                auto fetch = block->PrependNewInst(it, Opcode::ImageFetch,
                                                   {Shader::IR::Value{}, coords, Shader::IR::Value{}, ir.Imm32(0U), Shader::IR::Value{}},
                                                   info.raw);
                previous[k] = Shader::IR::Value{&*fetch};
            }
            const Shader::IR::U32 old{ir.BitCast<Shader::IR::U32>(Shader::IR::F32{ir.CompositeExtract(previous[k], c)})};
            const Shader::IR::U32 mine{ir.BitCast<Shader::IR::U32>(Shader::IR::F32{inst.Arg(2)})};
            inst.SetArg(2, ir.BitCast<Shader::IR::F32>(ir.BitwiseOr(old, mine)));
            changed++;
        }
    }
    if (changed) {
        program.info.loads.Set(Shader::IR::Attribute::PositionX, true);
        program.info.loads.Set(Shader::IR::Attribute::PositionY, true);
    }
    return changed;
}

// Atómicos de imagen de un solo descriptor (SUATOM/SURED, imageAtomicOr…): texture_pass les deja el índice vacío
// (IR::Value{}, tipo Void) y el emisor de Eden (ImageAtomicU32) hace index.U32(), con un DEBUG_ASSERT(type == U32) que en
// Eden (release) no se comprueba y que el shim (shim/common/assert.h) convierte en excepción: el shader no se traducía
// ("aserción (frontend/ir/value.h:378): type == Type::U32"). En el romfs de Xenoblade son 35 pixel shaders únicos de
// monolib/shader/shd_lgt.wishp (72 copias), todos con ImageAtomicOr32. Se les pone el 0 explícito: lo mismo que lee Eden
// de un Value{} (imm_u32 comparte la unión con un puntero nulo). Devuelve las instrucciones cambiadas.
int fix_image_atomic_index(Shader::IR::Program& program) {
    int changed = 0;
    for (Shader::IR::Block* block : program.blocks)
        for (Shader::IR::Inst& inst : block->Instructions())
            if (Shader::IR::NameOf(inst.GetOpcode()).starts_with("ImageAtomic") && inst.NumArgs() > 0 &&
                inst.Arg(0).Type() == Shader::IR::Type::Void) {
                inst.SetArg(0, Shader::IR::Value{0u});
                changed++;
            }
    return changed;
}

}  // namespace

uint32_t binding_base(Stage s) { return s == Stage::Compute ? 0 : static_cast<uint32_t>(s) * kBindingsPerStage; }

const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Vertex: return "vertex";
        case Stage::TessControl: return "tess-control";
        case Stage::TessEval: return "tess-eval";
        case Stage::Geometry: return "geometry";
        case Stage::Fragment: return "pixel";
        case Stage::Compute: return "compute";
    }
    return "?";
}

bool translate(const uint8_t* code, size_t max_size, const Options& options, Translated& out) {
    out = {};
    uint32_t magic = 0;
    if (max_size >= 4) std::memcpy(&magic, code, 4);
    const bool compute = magic == kNvnComputeCodeMagic || (options.compute_hint && magic != kNvnCodeMagic);
    if ((magic != kNvnCodeMagic && !compute) || max_size < kNvnHeaderSize + sizeof(Shader::ProgramHeader)) {
        out.error = "no es código NVN";
        return false;
    }
    try {
        NvnEnvironment env(code, max_size, options, compute);
        out.stage = to_stage(env.ShaderStage());
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
        Shader::ObjectPool<Shader::IR::Inst> insts;
        Shader::ObjectPool<Shader::IR::Block> blocks;
        Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress() + (compute ? 0 : sizeof(Shader::ProgramHeader)));
        Shader::HostTranslateInfo host{};
        host.support_conditional_barrier = options.keep_conditional_barriers;
        host.keep_uniform_barriers = options.uniform_barriers;
        // Bindless de los uniform buffers del juego: arrays hasta el final del cbuf (los decals de c9 son 256 registros de
        // 128 bytes en 32 KB; con el límite de 1024 descriptores repartido entre los arrays se cortaban). El Adreno admite
        // 524288 por etapa.
        host.bindless_cbuf_arrays = options.bindless_arrays;
        if (options.bindless_arrays)
            host.max_per_stage_descriptor_sampled_images = host.max_descriptor_set_sampled_images = host.max_per_stage_resources = 4096;
        ShimLog::conditional_barriers = ShimLog::uniform_barriers = 0;
        // Vertex shader doble: VertexA (env) y VertexB (env_b) traducidos por separado y unidos como hace Eden con los
        // dos programas de Maxwell (el epílogo de A y el prólogo de B se quitan y B sigue a A).
        const bool dual_vertex = !compute && env.ShaderStage() == Shader::Stage::VertexA;
        std::optional<NvnEnvironment> env_b;
        if (dual_vertex) {
            env_b.emplace(code, max_size, options, false, true);
            if (env_b->ShaderStage() != Shader::Stage::VertexB) {
                out.error = "vertex shader doble: el segundo programa no es VertexB";
                return false;
            }
        }
        auto translate_all = [&](Shader::ObjectPool<Shader::Maxwell::Flow::Block>& fb, Shader::ObjectPool<Shader::IR::Inst>& in,
                                 Shader::ObjectPool<Shader::IR::Block>& bl, Shader::Maxwell::Flow::CFG& cfg_a) {
            Shader::IR::Program first = Shader::Maxwell::TranslateProgram(in, bl, env, cfg_a, host);
            if (!dual_vertex) return first;
            Shader::Maxwell::Flow::CFG cfg_b(*env_b, fb, env_b->StartAddress() + sizeof(Shader::ProgramHeader));
            Shader::IR::Program second = Shader::Maxwell::TranslateProgram(in, bl, *env_b, cfg_b, host);
            return Shader::Maxwell::MergeDualVertexPrograms(first, second, *env_b);
        };
        Shader::IR::Program program = translate_all(flow_blocks, insts, blocks, cfg);
        // Texturas enteras: se sabe con el IR; se traduce otra vez con ellas declaradas U32 (solo si hay alguna).
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks2;
        Shader::ObjectPool<Shader::IR::Inst> insts2;
        Shader::ObjectPool<Shader::IR::Block> blocks2;
        // Más las que pide el draw (Options::integer_handles: su vista enlazada es entera).
        auto integers = integer_textures(program, options.integer_images);
        integers.insert(options.integer_handles.begin(), options.integer_handles.end());
        if (!integers.empty()) {
            out.integer_textures = static_cast<uint32_t>(integers.size());
            if (env_b) env_b->SetIntegerTextures(integers);
            env.SetIntegerTextures(std::move(integers));
            Shader::Maxwell::Flow::CFG cfg2(env, flow_blocks2,
                                            env.StartAddress() + (compute ? 0 : sizeof(Shader::ProgramHeader)));
            ShimLog::conditional_barriers = ShimLog::uniform_barriers = 0;
            program = translate_all(flow_blocks2, insts2, blocks2, cfg2);
        }
        out.conditional_barriers = ShimLog::conditional_barriers;
        out.uniform_barriers = ShimLog::uniform_barriers;
        // Adjuntos del pase leídos en su píxel (Options::input_attachment_handles): cada uno tiene que ser una textura 2D
        // sola (ni array de descriptores ni handle partido) de este shader; si alguna no está, no hay variante.
        if (!options.input_attachment_handles.empty()) {
            if (out.stage != Stage::Fragment) {
                out.error = "input_attachment_handles fuera de un pixel shader";
                return false;
            }
            for (const Options::InputAttachment& a : options.input_attachment_handles) {
                bool found = false;
                for (auto& d : program.info.texture_descriptors) {
                    if (d.cbuf_index != a.cbuf_index || d.cbuf_offset != a.cbuf_offset) continue;
                    if (d.count != 1 || d.has_secondary || d.is_multisample || d.is_integer ||
                        (d.type != Shader::TextureType::Color2D && d.type != Shader::TextureType::Color2DRect)) {
                        out.error = "input_attachment_handles: la textura no es una 2D sencilla";
                        return false;
                    }
                    d.input_attachment = static_cast<s32>(a.attachment);
                    found = true;
                }
                if (!found) {
                    out.error = "input_attachment_handles: el shader no tiene esa textura";
                    return false;
                }
            }
        }
        // Solo con las imágenes enteras: con sin_imagenes_enteras la imagen sigue float y el OpImageTexelPointer de Eden
        // (image_u32) no valdría con ella (SPIR-V no válido en vez de "no se pudo traducir"): como antes, sin traducir.
        if (options.integer_images) fix_image_atomic_index(program);
        if (options.shared_to_local) out.shared_to_local = shared_to_local(program, env.WorkgroupSize());
        const uint32_t xfb_stride = out.stage == Stage::Vertex ? add_xfb_capture(program, options.xfb_capture_always)
                                    : out.stage == Stage::Geometry && options.xfb_gs_capture ? add_xfb_capture_gs(program)
                                                                                              : 0;
        if (options.xfb_gs_capture && !xfb_stride) {
            out.error = "xfb_gs_capture sin genéricos ni EmitVertex que capturar";
            return false;
        }
        // Antes de list_resources (la textura que añade es un recurso más) y del volcado del IR (que se vea).
        if (options.logic_or_targets && !emulate_logic_or(program, options.logic_or_targets)) {
            out.error = "logic_or_targets sin escrituras de esos render targets";
            return false;
        }
        if (options.ir) {
            *options.ir = Shader::IR::DumpProgram(program);
            // Qué textura (índice de descriptor) usa cada instrucción de imagen: el volcado de Eden no lo dice.
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (Shader::IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                        char line[96];
                        snprintf(line, sizeof line, "[%016llx] descriptor %u\n",
                                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&inst)),
                                 static_cast<unsigned>(inst.Flags<Shader::IR::TextureInstInfo>().descriptor_index));
                        *options.ir += line;
                    }
        }

        // DEPURACIÓN: NIER_SIN_DESCARTE: los pixel shaders sin sus discard (con NIER_SIN_BLEND, el alfa que calculan
        // queda escrito tal cual en el render target).
        if (const char* keep = getenv("NIER_SIN_DESCARTE"); keep && *keep && out.stage == Stage::Fragment)
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (inst.GetOpcode() == Shader::IR::Opcode::DemoteToHelperInvocation)
                        inst.ReplaceOpcode(Shader::IR::Opcode::Void);

        out.has_brx = env.HasBrx() || (env_b && env_b->HasBrx());
        // Geometry shader de paso (SPH common0 bit 24, el geometry_passthrough de Eden): copia las salidas del vertex
        // shader. El ejecutor captura entonces el transform feedback del vertex shader (lo mismo que saldría de este).
        if (out.stage == Stage::Geometry) out.geometry_passthrough = env.SPH().common0.geometry_passthrough != 0;
        // Y el pixel shader de esos draws: la capa que lee (gl_Layer) es la del framebuffer de la tanda; la del juego es
        // esa más la primera de la tanda (la tabla de color de Unreal calcula el azul con ella).
        if (out.stage == Stage::Fragment && options.layer_chunk_base > 0)
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::GetAttribute ||
                        inst.Arg(0).Attribute() != Shader::IR::Attribute::Layer)
                        continue;
                    // Una lectura nueva delante y esta pasa a ser la suma (ReplaceUsesWith la deja como Identity).
                    Shader::IR::IREmitter ir{*block, Shader::IR::Block::InstructionList::s_iterator_to(inst)};
                    const Shader::IR::F32 raw{ir.GetAttribute(Shader::IR::Attribute::Layer)};
                    const Shader::IR::U32 absolute{ir.IAdd(ir.BitCast<Shader::IR::U32>(raw),
                                                           ir.Imm32(static_cast<uint32_t>(options.layer_chunk_base)))};
                    inst.ReplaceUsesWith(ir.BitCast<Shader::IR::F32>(absolute));
                }
        // Pintado por tandas de capas (Options::layer_chunk_base): capa relativa a la tanda, o la de descarte.
        // Con layer_chunk_cull, lo de fuera además se recorta: delante de cada EmitVertex se escribe la distancia de
        // recorte 0 con la condición de la capa (1 dentro: no toca la primitiva; -1 fuera, en todos sus vértices: no
        // queda nada de ella). En cada vértice y no junto a la capa: tras un EmitVertex las salidas quedan sin definir, y
        // el de la niebla de Xenoblade 3 (6a4e4cbf66a66461, un geometry shader de paso) escribe la capa una sola vez,
        // antes del primero. La condición de un EmitVertex es la de la última escritura de la capa de su bloque; si no
        // tiene, la de la única del programa cuando su bloque lo domina (no se llega a él sin pasar por ella). Todos o
        // ninguno: con uno solo sin condición, la variante queda como sin recorte. Tampoco si el shader ya escribe
        // distancias de recorte (la 0 sería la del juego, que no se declara: el perfil va sin ellas).
        struct CullEmit {
            Shader::IR::Block* block;
            Shader::IR::Inst* emit;
            const Shader::IR::Inst* layer;
        };
        std::vector<CullEmit> cull_emits;
        std::vector<std::pair<const Shader::IR::Inst*, Shader::IR::F32>> cull_distances;  // por escritura de la capa
        const auto writes_layer = [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::SetAttribute &&
                   inst.Arg(0).Attribute() == Shader::IR::Attribute::Layer;
        };
        bool chunk_cull = out.stage == Stage::Geometry && options.layer_chunk_base >= 0 && options.layer_chunk_cull &&
                          !program.info.stores.ClipDistances() && !program.post_order_blocks.empty();
        if (chunk_cull) {
            const Shader::IR::Inst* only = nullptr;  // la única escritura de la capa (si hay más, nulo) y su bloque
            const Shader::IR::Block* only_block = nullptr;
            int writes = 0;
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (writes_layer(inst)) writes++, only = &inst, only_block = block;
            if (writes != 1) only = nullptr, only_block = nullptr;
            // Los bloques a los que se llega desde la entrada sin pasar por el de esa escritura: los que no domina.
            std::set<const Shader::IR::Block*> undominated;
            if (only_block) {
                std::vector<const Shader::IR::Block*> pending;
                if (const Shader::IR::Block* entry = program.post_order_blocks.back(); entry != only_block)
                    pending.push_back(entry), undominated.insert(entry);
                while (!pending.empty()) {
                    const Shader::IR::Block* at = pending.back();
                    pending.pop_back();
                    for (const Shader::IR::Block* next : at->ImmSuccessors())
                        if (next != only_block && undominated.insert(next).second) pending.push_back(next);
                }
            }
            for (Shader::IR::Block* block : program.post_order_blocks) {
                const Shader::IR::Inst* last = nullptr;  // la última escritura de la capa de este bloque hasta aquí
                for (Shader::IR::Inst& inst : block->Instructions()) {
                    if (writes_layer(inst)) last = &inst;
                    if (inst.GetOpcode() != Shader::IR::Opcode::EmitVertex) continue;
                    const Shader::IR::Inst* from = last;
                    if (!from && only && block != only_block && !undominated.count(block)) from = only;
                    if (!from) chunk_cull = false;
                    cull_emits.push_back({block, &inst, from});
                }
            }
            if (cull_emits.empty()) chunk_cull = false;
        }
        if (out.stage == Stage::Geometry && options.layer_chunk_base >= 0)
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions()) {
                    if (!writes_layer(inst)) continue;
                    Shader::IR::IREmitter ir{*block, Shader::IR::Block::InstructionList::s_iterator_to(inst)};
                    const Shader::IR::U32 layer{ir.BitCast<Shader::IR::U32>(Shader::IR::F32{inst.Arg(1)})};
                    const Shader::IR::U32 relative{ir.ISub(layer, ir.Imm32(static_cast<uint32_t>(options.layer_chunk_base)))};
                    const Shader::IR::U32 size{ir.Imm32(options.layer_chunk_size)};
                    const Shader::IR::U1 inside{ir.ILessThan(relative, size, false)};
                    const Shader::IR::U32 chosen{ir.Select(inside, relative, size)};
                    inst.SetArg(1, ir.BitCast<Shader::IR::F32>(chosen));
                    if (chunk_cull) {
                        const Shader::IR::F32 distance{ir.Select(inside, ir.Imm32(1.0f), ir.Imm32(-1.0f))};
                        cull_distances.emplace_back(&inst, distance);
                    }
                }
        if (chunk_cull) {
            for (const CullEmit& e : cull_emits) {
                const auto distance = std::find_if(cull_distances.begin(), cull_distances.end(),
                                                   [&](const auto& d) { return d.first == e.layer; });
                Shader::IR::IREmitter ir{*e.block, Shader::IR::Block::InstructionList::s_iterator_to(*e.emit)};
                ir.SetAttribute(Shader::IR::Attribute::ClipDistance0, distance->second, ir.Imm32(0u));
            }
            // La salida la declara el emisor por lo que dice la información del programa (que sale de la cabecera del
            // shader, donde no está) y por el perfil (más abajo: una sola distancia, y solo en esta variante).
            program.info.stores.Set(Shader::IR::Attribute::ClipDistance0, true);
            program.info.used_clip_distances = 1;
            out.layer_cull = true;
        }

        if (out.stage == Stage::Geometry) {
            // Lecturas que no existen en un geometry shader (la cabecera las marca en algunos de NieR) y
            // que Eden declararía como entradas inválidas: se leerán como 0.
            using A = Shader::IR::Attribute;
            for (A a : {A::InstanceId, A::VertexId, A::FrontFace, A::PointSpriteS, A::PointSpriteT,
                        A::TessellationEvaluationPointU, A::TessellationEvaluationPointV, A::Layer, A::ViewportIndex})
                program.info.loads.Set(a, false), program.info.passthrough.Set(a, false);
        }
        const Shader::Info& info = program.info;
        out.constant_buffer_mask = info.constant_buffer_mask;
        for (size_t k = 0; k < 18 && k < info.constant_buffer_used_sizes.size(); k++)
            out.constant_buffer_used[k] = info.constant_buffer_used_sizes[k];
        for (size_t k = 0; k < 32; k++) {
            if (info.loads.Generic(k)) out.input_generics |= uint64_t{1} << k;
            if (info.stores.Generic(k)) out.output_generics |= uint64_t{1} << k;
        }
        for (size_t k = 0; k < info.stores_frag_color.size(); k++)
            if (info.stores_frag_color[k]) out.color_outputs |= 1u << k;
        const uint32_t base = binding_base(out.stage);
        list_resources(info, base, out);
        for (Resource& r : out.resources)  // el SSBO del transform feedback emulado lleva su stride (ver kXfbBank)
            if (r.kind == Resource::StorageBuffer && r.cbuf_index == kXfbBank) r.texture_type = xfb_stride;

        Shader::Profile profile{};
        profile.supported_spirv = options.spirv_version;
        profile.unified_descriptor_binding = true;
        profile.support_multi_viewport = true;  // el Adreno lo tiene (y el dispositivo activa todas sus features)
        if (out.layer_cull) profile.max_user_clip_distances = 1;  // la del recorte de tandas (Options::layer_chunk_cull)
        // Subgrupos: votos y ballot solo si el dispositivo los tiene, y en las etapas que los admite; si el
        // subgrupo es mayor que el warp de 32 de Maxwell, Eden emula los votos con ballot.
        constexpr uint32_t kVote = 0x2, kBallot = 0x8, kShuffle = 0x10, kShuffleRelative = 0x20, kQuad = 0x80;
        const uint32_t ops = options.subgroup_operations;
        const bool usable = (ops & kVote) && (ops & kBallot) && (ops & kShuffle);
        profile.support_vote = usable;
        profile.support_shuffle_relative = (ops & kShuffleRelative) != 0;
        profile.support_quad_shuffles = (ops & kQuad) != 0;
        profile.warp_size_potentially_larger_than_guest = options.subgroup_size > 32;
        profile.max_subgroup_size = options.subgroup_size;
        profile.has_broken_spirv_subgroup_shuffle = options.broken_subgroup_shuffle;
        profile.has_broken_spirv_unreachable = options.broken_unreachable;
        profile.support_sampled_image_array_nonuniform_indexing = options.nonuniform_indexing;
        profile.supported_subgroup_stages = 0;
        if (usable) {
            static const uint32_t kVkStage[] = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20};  // VertexB … Compute
            for (uint32_t k = 0; k < 6; k++)
                if (options.subgroup_stages & kVkStage[k]) profile.supported_subgroup_stages |= 1u << k;
        }
        Shader::RuntimeInfo runtime{};
        // NVN enlaza las etapas en el momento del draw (un programa por etapa): se asume que la etapa
        // anterior escribe todos los atributos genéricos, o Eden no declararía las entradas.
        // Solo genéricos y posición: con todo marcado, un geometry shader arrastraba como entradas valores
        // de sistema que no existen en su etapa (gl_InstanceIndex, gl_TessCoord…) y el driver lo rechazaba.
        for (size_t k = 0; k < 32; k++)
            if (options.previous_stores >> k & 1)
                for (size_t c = 0; c < 4; c++)
                    runtime.previous_stage_stores.Set(Shader::IR::Attribute::Generic0X + k * 4 + c);
        for (size_t c = 0; c < 4; c++) runtime.previous_stage_stores.Set(Shader::IR::Attribute::PositionX + c);
        runtime.input_topology = static_cast<Shader::InputTopology>(options.input_topology);
        // Salidas de color enteras (Options::unsigned_color_targets / signed_color_targets): uvec4/ivec4 y OpBitcast.
        for (uint32_t k = 0; k < 8; k++) {
            if (options.unsigned_color_targets >> k & 1)
                runtime.color_output_types[k] = Shader::AttributeType::UnsignedInt;
            else if (options.signed_color_targets >> k & 1)
                runtime.color_output_types[k] = Shader::AttributeType::SignedInt;
        }
        // Entradas de vértice enteras (Options::uint_vertex_inputs…): uvec4/ivec4; las "scaled" se convierten en el shader
        // (support_scaled_attributes = false: el Adreno no tiene formatos de vértice *SCALED).
        for (uint32_t k = 0; k < 32; k++) {
            if (options.uint_vertex_inputs >> k & 1) runtime.generic_input_types[k] = Shader::AttributeType::UnsignedInt;
            else if (options.sint_vertex_inputs >> k & 1) runtime.generic_input_types[k] = Shader::AttributeType::SignedInt;
            else if (options.uscaled_vertex_inputs >> k & 1)
                runtime.generic_input_types[k] = Shader::AttributeType::UnsignedScaled;
            else if (options.sscaled_vertex_inputs >> k & 1)
                runtime.generic_input_types[k] = Shader::AttributeType::SignedScaled;
        }
        runtime.convert_depth_mode = options.convert_depth_mode;
        Shader::Backend::Bindings bindings{};
        bindings.unified = base;
        out.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
        // Eden escribe gl_ClipDistance sin declarar la capacidad ClipDistance (spirv-val no lo mira, pero SPIR-V la pide
        // para usar la variable, y es lo que declara glslang: el camino que el compilador del Adreno tiene probado). Se
        // añade en la variante del recorte, la única que la escribe: OpCapability (17) ClipDistance (32), tras la cabecera.
        if (out.layer_cull && out.spirv.size() > 5) out.spirv.insert(out.spirv.begin() + 5, {uint32_t{2} << 16 | 17u, 32u});
        move_arrays_to_bindless_set(out);
        // El geometry shader de paso que pone la capa que este vertex shader calcula (ver Translated::layer_gs).
        if (out.stage == Stage::Vertex && program.info.requires_layer_emulation) {
            Shader::ObjectPool<Shader::IR::Inst> gs_insts;
            Shader::ObjectPool<Shader::IR::Block> gs_blocks;
            Shader::IR::Program gs = Shader::Maxwell::GenerateGeometryPassthrough(gs_insts, gs_blocks, host, program,
                                                                                 Shader::OutputTopology::TriangleStrip);
            Shader::RuntimeInfo gs_runtime{};
            gs_runtime.previous_stage_stores = program.info.stores;
            gs_runtime.input_topology = Shader::InputTopology::Triangles;
            Shader::Backend::Bindings gs_bindings{};
            gs_bindings.unified = binding_base(Stage::Geometry);
            out.layer_gs = Shader::Backend::SPIRV::EmitSPIRV(profile, gs_runtime, gs, gs_bindings);
        }
        if (options.optimize_stages >> static_cast<uint32_t>(out.stage) & 1) {
            const auto start = std::chrono::steady_clock::now();
            out.spirv_opt_from = static_cast<uint32_t>(out.spirv.size());
            out.spirv_opt = optimize_spirv(out.spirv, out.spirv_opt_error) ? 1 : -1;
            out.spirv_opt_us = static_cast<uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        }
        return true;
    } catch (const std::exception& e) {
        out.error = e.what();
        return false;
    }
}

}  // namespace nvnshader
