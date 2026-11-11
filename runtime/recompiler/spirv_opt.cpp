// SPDX-License-Identifier: GPL-3.0-or-later
// El SPIR-V de Eden por el optimizador de SPIRV-Tools (spirv_tools.cmake: el del NDK, v2022.4), en la Flip, al traducir.
//
// Por qué: el driver de Qualcomm compila mal el flujo con banderas que deja Eden al estructurar los gotos (un BRX deshecho
// en comparaciones y variables bool de función). En el skinning de la cara del cazador (793657b4f1e65631) tomaba el caso
// que pone los pesos a cero (1/0 → NaN y la cabeza no se dibuja); el mismo módulo pasado por spirv-opt -O en el Mac
// (banderas a SSA y plegadas, ramas muertas fuera, un único return) sale bien, y el intérprete de SPIR-V da lo mismo bit a
// bit que el original. Eso se hizo primero con un fichero por shader (<datos>/spirv_reemplazo); esto lo hace con todos.
//
// La lista de pases es la de -O tal cual (Optimizer::RegisterPerformancePasses de esta versión) con dos cambios para que
// el ejecutor no note nada: el ADCE con preserve_interface (sin él quita de la interfaz las entradas que no se leen) y
// preserve_bindings (sin él quita los descriptores sin usar). Además, después se comprueba que la interfaz y los recursos
// son los mismos (interface_summary) y que el resultado valida; si algo falla queda el SPIR-V de Eden, sin tocar.

#include <algorithm>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvn_shader.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

namespace nvnshader {

namespace {

constexpr uint32_t kMagic = 0x07230203;
constexpr uint32_t kOpEntryPoint = 15, kOpExecutionMode = 16, kOpFunction = 54, kOpVariable = 59, kOpDecorate = 71,
                   kOpExecutionModeId = 331;
constexpr uint32_t kUniformConstant = 0, kInput = 1, kUniform = 2, kOutput = 3, kPushConstant = 9, kImage = 11,
                   kStorageBuffer = 12;

// La versión de Vulkan que corresponde a la del módulo (Eden la pone de Options::spirv_version: 1.3 → Vulkan 1.1).
spv_target_env target_env(uint32_t version) {
    if (version >= 0x10600) return SPV_ENV_VULKAN_1_3;
    if (version >= 0x10500) return SPV_ENV_VULKAN_1_2;
    if (version >= 0x10400) return SPV_ENV_VULKAN_1_1_SPIRV_1_4;
    if (version >= 0x10300) return SPV_ENV_VULKAN_1_1;
    return SPV_ENV_VULKAN_1_0;
}

// Las instrucciones de nivel de módulo (hasta la primera OpFunction): fn(op, índice de la primera palabra, palabras).
// false si el módulo está mal formado.
template <typename F>
bool for_each_global(const std::vector<uint32_t>& w, F&& fn) {
    if (w.size() < 5 || w[0] != kMagic) return false;
    for (size_t k = 5; k < w.size();) {
        const uint32_t count = w[k] >> 16, op = w[k] & 0xFFFF;
        if (!count || k + count > w.size()) return false;
        if (op == kOpFunction) return true;
        fn(op, k, count);
        k += count;
    }
    return true;
}

// Palabras de la cadena literal (con su 0 final) que empieza en w[k], sin pasar de end.
size_t string_words(const std::vector<uint32_t>& w, size_t k, size_t end) {
    size_t n = 0;
    while (k + n < end) {
        const uint32_t x = w[k + n++];
        if (!(x & 0xFF) || !(x & 0xFF00) || !(x & 0xFF0000) || !(x & 0xFF000000)) break;
    }
    return n;
}

// Clase de almacenamiento de cada variable global.
std::unordered_map<uint32_t, uint32_t> global_storage(const std::vector<uint32_t>& w) {
    std::unordered_map<uint32_t, uint32_t> storage;
    for_each_global(w, [&](uint32_t op, size_t k, uint32_t count) {
        if (op == kOpVariable && count >= 4) storage[w[k + 2]] = w[k + 3];
    });
    return storage;
}

// SPIR-V < 1.4: la interfaz de un punto de entrada solo puede llevar variables Input y Output, y Eden mete también la
// memoria compartida (Workgroup; 17 de los 597 módulos de la caché de MH Rise). El validador lo rechaza y sin validar no
// se optimiza. Se quitan (en 1.3 no significan nada: es lo que el validador pide).
void strip_non_io_interface(std::vector<uint32_t>& w) {
    const auto storage = global_storage(w);
    std::vector<uint32_t> out(w.begin(), w.begin() + 5);
    out.reserve(w.size());
    size_t k = 5;
    for (; k < w.size();) {
        const uint32_t count = w[k] >> 16, op = w[k] & 0xFFFF;
        if (!count || k + count > w.size() || op == kOpFunction) break;
        if (op == kOpEntryPoint && count >= 4) {
            const size_t start = out.size();
            const size_t name = string_words(w, k + 3, k + count);
            out.insert(out.end(), w.begin() + k, w.begin() + k + 3 + name);
            for (size_t i = k + 3 + name; i < k + count; i++) {
                const auto it = storage.find(w[i]);
                if (it == storage.end() || it->second == kInput || it->second == kOutput) out.push_back(w[i]);
            }
            out[start] = static_cast<uint32_t>(out.size() - start) << 16 | kOpEntryPoint;
        } else {
            out.insert(out.end(), w.begin() + k, w.begin() + k + count);
        }
        k += count;
    }
    out.insert(out.end(), w.begin() + k, w.end());
    w = std::move(out);
}

// Lo que el optimizador no puede cambiar, sin ids (los renumera): cada punto de entrada (modelo y nombre), sus modos de
// ejecución, cada variable de su interfaz y cada variable de recursos (descriptores, push constants), estas dos como su
// clase de almacenamiento y sus decoraciones (Location, BuiltIn, Flat…, DescriptorSet, Binding, NonWritable…). Ordenado.
std::vector<std::vector<uint32_t>> interface_summary(const std::vector<uint32_t>& w) {
    std::unordered_map<uint32_t, std::vector<std::vector<uint32_t>>> decorations;
    std::unordered_map<uint32_t, uint32_t> storage;
    std::vector<std::vector<uint32_t>> out;
    std::vector<uint32_t> interface;
    for_each_global(w, [&](uint32_t op, size_t k, uint32_t count) {
        if (op == kOpEntryPoint && count >= 4) {
            const size_t name = string_words(w, k + 3, k + count);
            std::vector<uint32_t> entry{0xE0000000u, w[k + 1]};
            entry.insert(entry.end(), w.begin() + k + 3, w.begin() + k + 3 + name);
            out.push_back(std::move(entry));
            interface.insert(interface.end(), w.begin() + k + 3 + name, w.begin() + k + count);
        } else if ((op == kOpExecutionMode || op == kOpExecutionModeId) && count >= 3) {
            std::vector<uint32_t> mode{0xE1000000u | op};  // sin el id del punto de entrada (Eden pone uno)
            mode.insert(mode.end(), w.begin() + k + 2, w.begin() + k + count);
            out.push_back(std::move(mode));
        } else if (op == kOpDecorate && count >= 3) {
            decorations[w[k + 1]].emplace_back(w.begin() + k + 2, w.begin() + k + count);
        } else if (op == kOpVariable && count >= 4) {
            storage[w[k + 2]] = w[k + 3];
        }
    });
    const auto variable = [&](uint32_t tag, uint32_t id, uint32_t sc) {
        std::vector<uint32_t> v{tag, sc};
        auto d = decorations[id];
        std::sort(d.begin(), d.end());
        for (const auto& x : d) {
            v.push_back(0xDEC00000u | static_cast<uint32_t>(x.size()));
            v.insert(v.end(), x.begin(), x.end());
        }
        out.push_back(std::move(v));
    };
    for (uint32_t id : interface) variable(0xE2000000u, id, storage.count(id) ? storage[id] : ~0u);
    for (const auto& [id, sc] : storage)
        if (sc == kUniformConstant || sc == kUniform || sc == kPushConstant || sc == kImage || sc == kStorageBuffer)
            variable(0xE3000000u, id, sc);
    std::sort(out.begin(), out.end());
    return out;
}

// spirv-opt -O de v2022.4 (Optimizer::RegisterPerformancePasses, el mismo orden), con el ADCE que conserva la interfaz.
void register_passes(spvtools::Optimizer& o) {
    using namespace spvtools;
    const auto adce = [] { return CreateAggressiveDCEPass(true); };
    o.RegisterPass(CreateWrapOpKillPass())
        .RegisterPass(CreateDeadBranchElimPass())
        .RegisterPass(CreateMergeReturnPass())
        .RegisterPass(CreateInlineExhaustivePass())
        .RegisterPass(CreateEliminateDeadFunctionsPass())
        .RegisterPass(adce())
        .RegisterPass(CreatePrivateToLocalPass())
        .RegisterPass(CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(CreateLocalSingleStoreElimPass())
        .RegisterPass(adce())
        .RegisterPass(CreateScalarReplacementPass())
        .RegisterPass(CreateLocalAccessChainConvertPass())
        .RegisterPass(CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(CreateLocalSingleStoreElimPass())
        .RegisterPass(adce())
        .RegisterPass(CreateLocalMultiStoreElimPass())
        .RegisterPass(adce())
        .RegisterPass(CreateCCPPass())
        .RegisterPass(adce())
        .RegisterPass(CreateLoopUnrollPass(true))
        .RegisterPass(CreateDeadBranchElimPass())
        .RegisterPass(CreateRedundancyEliminationPass())
        .RegisterPass(CreateCombineAccessChainsPass())
        .RegisterPass(CreateSimplificationPass())
        .RegisterPass(CreateScalarReplacementPass())
        .RegisterPass(CreateLocalAccessChainConvertPass())
        .RegisterPass(CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(CreateLocalSingleStoreElimPass())
        .RegisterPass(adce())
        .RegisterPass(CreateSSARewritePass())
        .RegisterPass(adce())
        .RegisterPass(CreateVectorDCEPass())
        .RegisterPass(CreateDeadInsertElimPass())
        .RegisterPass(CreateDeadBranchElimPass())
        .RegisterPass(CreateSimplificationPass())
        .RegisterPass(CreateIfConversionPass())
        .RegisterPass(CreateCopyPropagateArraysPass())
        .RegisterPass(CreateReduceLoadSizePass())
        .RegisterPass(adce())
        .RegisterPass(CreateBlockMergePass())
        .RegisterPass(CreateRedundancyEliminationPass())
        .RegisterPass(CreateDeadBranchElimPass())
        .RegisterPass(CreateBlockMergePass())
        .RegisterPass(CreateSimplificationPass());
}

}  // namespace

bool optimize_spirv(std::vector<uint32_t>& spirv, std::string& error) {
    if (spirv.size() < 5 || spirv[0] != kMagic) {
        error = "no es SPIR-V";
        return false;
    }
    const spv_target_env env = target_env(spirv[1]);
    std::vector<uint32_t> input = spirv;
    if (spirv[1] < 0x10400) strip_non_io_interface(input);
    std::string message;  // el primer error del validador o de un pase
    const auto consumer = [&message](spv_message_level_t level, const char*, const spv_position_t& position,
                                     const char* text) {
        if (level > SPV_MSG_ERROR || !message.empty()) return;
        message = text ? text : "?";
        if (position.index) message += " (palabra " + std::to_string(position.index) + ")";
    };
    try {
        spvtools::Optimizer optimizer(env);
        optimizer.SetMessageConsumer(consumer);
        register_passes(optimizer);
        spvtools::OptimizerOptions options;
        options.set_run_validator(true);  // como spirv-opt: lo que no valida no se toca
        options.set_preserve_bindings(true);
        options.set_preserve_spec_constants(true);
        std::vector<uint32_t> result;
        if (!optimizer.Run(input.data(), input.size(), &result, options)) {
            error = "optimizador: " + (message.empty() ? std::string("falló sin mensaje") : message);
            return false;
        }
        // El resultado también tiene que validar (cada validación es ~1/8 del tiempo de todo esto).
        spvtools::SpirvTools tools(env);
        tools.SetMessageConsumer(consumer);
        if (!tools.Validate(result)) {
            error = "el optimizado no valida: " + message;
            return false;
        }
        if (interface_summary(result) != interface_summary(input)) {
            error = "el optimizado cambia la interfaz o los recursos";
            return false;
        }
        spirv = std::move(result);
        return true;
    } catch (const std::exception& e) {  // sin memoria, sobre todo
        error = std::string("optimizador: ") + e.what();
        return false;
    }
}

}  // namespace nvnshader
