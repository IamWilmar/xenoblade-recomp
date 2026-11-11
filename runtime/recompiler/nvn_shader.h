// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Traducción de una etapa NVN (código en memoria: cabecera NVN de 0x30 bytes, SPH de Maxwell e
// instrucciones) a SPIR-V con el recompilador de Eden. Lo usan libnnsdk.so y tools/shadertool.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nvnshader {

enum class Stage { Vertex, TessControl, TessEval, Geometry, Fragment, Compute };

// Un recurso del shader y su binding en el set 0 (bindings unificados, en el orden de Eden).
struct Resource {
    enum Kind { UniformBuffer, StorageBuffer, TextureBuffer, ImageBuffer, Texture, Image } kind;
    uint32_t binding;
    uint32_t cbuf_index;   // UBO: banco Maxwell (c0..c17). Resto: banco donde está el handle
    uint32_t cbuf_offset;  // offset del handle (texturas/imágenes/SSBO)
    uint32_t texture_type;  // Shader::TextureType para texturas e imágenes
    bool is_depth = false;
    bool is_written = false;
    // Textura o imagen declarada entera (usampler/uimage: Eden con is_integer). Lo decide translate: las de
    // integer_textures (lecturas con texelFetch solo como enteros, imágenes con formato entero o atómicos) y las de
    // Options::integer_handles (la vista enlazada es entera). Va en la caché de SPIR-V (la palabra de reserva).
    bool is_integer = false;
    // Texturas con handle partido (textura | sampler leídos de dos sitios y combinados con OR).
    bool has_secondary = false;
    uint32_t secondary_cbuf_index = 0, secondary_cbuf_offset = 0;
    uint32_t shift_left = 0, secondary_shift_left = 0;
    // Texturas: cuántos descriptores tiene (Eden hace un array si el handle sale de un cbuf con índice dinámico: en el
    // pixel shader de luz directa de MH Rise, los decals de c9[0x5c/0x60/0x64 + 128·i] y las sondas de reflexión de
    // c6[0xc83c + 64·i]). El elemento i lee su handle en cbuf_offset + (i << size_shift).
    uint32_t count = 1;
    uint32_t size_shift = 3;
    // Lo que hace el shader con la palabra leída: handle = (palabra & handle_and_mask) | handle_or (el sampler fijo que
    // el compilador ORea: las sondas son (v & 0x7fff) | 0x10b00000).
    uint32_t handle_and_mask = ~0u, handle_or = 0;
    // Set de descriptores: 0 (el del draw) o 1 (kBindlessSet: los arrays con índice dinámico, con bindings propios,
    // ver translate). El set 1 lo comparten los shaders que leen los mismos handles.
    uint32_t set = 0;
};

constexpr uint32_t kBindlessSet = 1;
constexpr uint32_t kBindlessPerStage = 16;  // bindings del set 1 por etapa: etapa · 16 + j

// Transform feedback emulado (el Adreno no tiene VK_EXT_transform_feedback). Un vertex shader de captura (escribe
// atributos genéricos y no la posición: Xenoblade deforma así con el viento la hierba alta del campo, draws de puntos con
// el rasterizador descartado que el juego captura en el buffer de skinning y que luego lee como stream 1 de vértices)
// guarda además sus salidas en un SSBO: el recurso lleva este banco ficticio en cbuf_index y el stride en bytes en
// texture_type. Las componentes escritas van seguidas, en el orden de los atributos (genérico 0 x,y,z,w, genérico 1…).
// Por draw lee de c0, en kXfbParams: vértices por instancia, stride efectivo (0 = no capturar: todo cae en los primeros
// bytes) y desplazamiento en bytes dentro del buffer de captura. El índice capturado es instancia · vértices + vértice.
constexpr uint32_t kXfbBank = 0x7e;
constexpr uint32_t kXfbParams = 0x6e0;

struct Translated {
    Stage stage = Stage::Vertex;
    std::vector<uint32_t> spirv;
    std::vector<Resource> resources;
    uint32_t constant_buffer_mask = 0;
    uint32_t constant_buffer_used[18] = {};  // bytes leídos de cada banco (0 si no se usa)
    uint64_t input_generics = 0;   // atributos genéricos que lee (vertex: entradas)
    uint64_t output_generics = 0;  // atributos genéricos que escribe (para la etapa siguiente)
    uint32_t color_outputs = 0;    // fragment: salidas de color escritas (bit por RT)
    bool has_brx = false;          // tiene un switch (BRX): Eden leyó su tabla de c1
    bool has_bindless_set = false; // tiene arrays de texturas en el set 1 (kBindlessSet)
    int conditional_barriers = 0;  // barreras de grupo dentro de un if/bucle que se quitaron (ver keep_conditional_barriers)
    uint32_t integer_textures = 0; // texturas declaradas U32 porque el shader las lee como enteros (no va en la caché)
    int shared_to_local = 0;       // accesos a memoria compartida privada de cada invocación pasados a memoria local
                                   // (ver Options::shared_to_local; no va en la caché de SPIR-V: 0 si sale de ella)
    // Optimizador de SPIRV-Tools (Options::optimize_stages): 1 si el SPIR-V salió de él, -1 si falló (spirv_opt_error; queda
    // el de Eden), 0 si no se intentó. Tampoco va en la caché: 0 si sale de ella.
    int spirv_opt = 0;
    uint32_t spirv_opt_us = 0;     // lo que tardó (validar, optimizar, validar y comparar la interfaz)
    uint32_t spirv_opt_from = 0;   // palabras del de Eden
    std::string spirv_opt_error;
    std::string error;             // vacío si salió bien
};

struct Options {
    uint32_t spirv_version = 0x00010300;  // Vulkan 1.1
    uint32_t texture_bound_buffer = 2;  // TEX con índice inmediato: handle en c2[índice·4] (tabla de NVN)
    bool convert_depth_mode = false;
    std::string* ir = nullptr;  // si no es nulo, recibe el IR de Eden en texto (depuración)
    // Compute (de la sección de control del shader): inicio del código, workgroup, memoria compartida.
    uint32_t compute_code_offset = 0x100;
    // La sección de control dice compute (+0x714 == 5). En MH Rise (SDK 11.4) el código de compute empieza sin la
    // cabecera 0x12345679 de NieR: directamente en las instrucciones, en el offset del control (0).
    bool compute_hint = false;
    // Eden quita las barreras de grupo (BAR) dentro de control de flujo condicional; con esto se dejan (correcto si el
    // flujo es uniforme en el grupo, como un bucle con el mismo número de vueltas para todos).
    bool keep_conditional_barriers = false;
    uint32_t workgroup[3] = {1, 1, 1};
    // Bindless de los uniform buffers del juego: arrays con índice dinámico contados hasta el final del cbuf (64 KB), con
    // la máscara y el sampler fijo del handle, y movidos al set 1. Sin esto (<datos>/sin_bindless_arrays al traducir), como
    // antes. nonuniform_indexing: el dispositivo tiene shaderSampledImageArrayNonUniformIndexing (cada píxel lee su
    // decal o su sonda: el índice no es uniforme en el draw).
    bool bindless_arrays = true;
    bool nonuniform_indexing = false;
    uint32_t shared_memory = 0, local_memory = 0;
    // Geometry: genéricos que escribe la etapa anterior (solo esos se declaran como entrada: el Adreno
    // admite 128 componentes y declarar los 32 + la posición se pasa) y topología de entrada
    // (InputTopology de Eden: puntos 0, líneas 1, líneas con adyacencia 2, triángulos 3, con adyacencia 4).
    uint64_t previous_stores = ~0ull;
    int input_topology = 0;
    // Subgrupos del dispositivo (VkPhysicalDeviceSubgroupProperties): tamaño, etapas (bits de
    // VkShaderStageFlagBits) y operaciones (VkSubgroupFeatureFlagBits). Sin nada, no se usan.
    uint32_t subgroup_size = 32, subgroup_stages = 0, subgroup_operations = 0;  // NVN NEAR_IS_MINUS_W: z de [-w,w] a [0,w] (Vulkan)
    // OpGroupNonUniformShuffle con índice dinámico roto en el driver propietario de Qualcomm: Eden lo emula con un
    // OpGroupNonUniformBroadcast por carril y selects (has_broken_spirv_subgroup_shuffle, que Eden pone para
    // VK_DRIVER_ID_QUALCOMM_PROPRIETARY). En MH Rise solo lo usan dos compute de reparto con atómicos por warp
    // (f9dd15e2e79505d8 y 6c483f3226c7c07b: ballot, atomicAdd del primer carril y su base repartida con shuffle).
    bool broken_subgroup_shuffle = false;
    // OpUnreachable tras un salto indirecto (BRX: ningún destino coincide) mal optimizado en el driver de Qualcomm: el
    // skinning de la cara del cazador (793657b4) tomaba el caso que pone los pesos a cero (1/0 → NaN, la cabeza no se
    // ve). Se emite OpReturn: nunca se ejecuta y no le da al compilador nada que suponer.
    bool broken_unreachable = false;
    // Memoria compartida que solo es privada de cada invocación (el compilador de NVN derrama registros ahí, en
    // tid·28 + k en el skinning de la cara) a la memoria local (Private): con esos stores a Workgroup dentro de los casos
    // de un switch, el driver de Qualcomm dejaba los pesos a cero (NaN en todo el grupo). Solo si se demuestra que las
    // ventanas de las invocaciones no se solapan (nvn_shader.cpp, shared_to_local). <datos>/sin_compartida_privada: como antes.
    bool shared_to_local = true;
    // Etapas (bit 1 << Stage) cuyo SPIR-V se pasa por el optimizador de SPIRV-Tools (spirv_opt.cpp: el -O de spirv-opt
    // sin tocar la interfaz ni los bindings) antes de devolverlo. El driver de Qualcomm compila mal el flujo con banderas
    // que deja Eden al estructurar los gotos (un BRX deshecho: el skinning de la cara del cazador, 793657b4, tomaba el
    // caso de los pesos a cero) y con -O se arregla. Si no valida o el optimizador falla, queda el de Eden
    // (Translated::spirv_opt). 0: nada (shadertool sin SPIRV_OPT).
    uint32_t optimize_stages = 0;
    // c1: las constantes que el compilador dejó tras el código (sección de control 0x700..0x704). Son parte
    // del programa, así que se conocen al traducir: Eden lee de ahí las tablas de salto de BRX/JMX
    // (un switch del shader). Sin ellas todas las entradas valían 0 y los casos no se traducían.
    const uint8_t* constants = nullptr;
    uint32_t constants_size = 0;
    // Operación lógica OR emulada (el dispositivo no tiene la feature logicOp: el Adreno 650). Bit k: el render target de
    // color k (entero) se escribe con OR. El pixel shader lee lo que ya tiene con un texelFetch entero en su píxel de una
    // textura más (Resource con cbuf_index kLogicOrBank y cbuf_offset k: el ejecutor le enlaza la copia "sombra" del
    // adjunto, la de antes del draw) y escribe previo | nuevo en cada componente. Xenoblade la pide en el RG32UI de las
    // máscaras de sondas IBL (ml::DrShdLgtGlobalIBL::renderLinkList). 0: nada (la traducción de siempre).
    uint32_t logic_or_targets = 0;
    // Render targets de color enteros del draw (bit k: el k): su salida del pixel shader se declara uvec4
    // (unsigned_color_targets) o ivec4 (signed_color_targets) y se escribe con los bits del registro tal cual (OpBitcast),
    // lo que hace Eden con RuntimeInfo::color_output_types en un dispositivo de verdad. Sin esto todas son vec4 y, en un
    // adjunto UINT/SINT, Vulkan no define lo que queda escrito (el tipo de la salida tiene que coincidir con el formato
    // numérico del adjunto). Xenoblade escribe así el RG32UI de las listas de sondas IBL y el R8UI del stencil a media
    // resolución (e300cd99). Lo pone program.cpp (integer_variant) con lo que tiene enlazado el draw. 0: vec4 (como antes).
    uint32_t unsigned_color_targets = 0, signed_color_targets = 0;
    // Texturas e imágenes que se declaran enteras (usampler/uimage) por la dirección de su handle (banco, offset), además
    // de las que deduce integer_textures del propio código: las que el draw tiene enlazadas con una vista entera (stencil
    // de un depth-stencil, R8UI…). Leer una vista entera con un sampler float es indefinido en Vulkan y en el Adreno da 0
    // (commit 9aa6cb0). Vacío: como siempre.
    std::vector<std::pair<uint32_t, uint32_t>> integer_handles;
    // Imágenes (SULD/SUST/SUATOM) con formato entero en la instrucción o con atómicos: declaradas uimage. Sin esto
    // (<datos>/sin_imagenes_enteras al traducir) un R32UI se declaraba OpTypeImage %float … R32ui (no válido en Vulkan:
    // VUID-StandaloneSpirv-Image-04965) y los atómicos ni se traducían.
    bool integer_images = true;
};

// Banco ficticio de las texturas que añade logic_or_targets (los de Maxwell van de 0 a 17).
constexpr uint32_t kLogicOrBank = 0x100;

constexpr uint32_t kNvnHeaderSize = 0x30;
constexpr uint32_t kNvnCodeMagic = 0x12345678;         // gráficos: cabecera + SPH en 0x30, código en 0x80
constexpr uint32_t kNvnComputeCodeMagic = 0x12345679;  // compute: sin SPH, código donde diga el control

// code apunta a la cabecera NVN; max_size acota las lecturas (hasta el final del memory pool).
bool translate(const uint8_t* code, size_t max_size, const Options& options, Translated& out);

const char* stage_name(Stage s);

// Pasa un módulo por el optimizador (spirv_opt.cpp). Si no se puede (no valida, falla un pase, el resultado no valida o
// cambia la interfaz o los recursos) devuelve false con el motivo en error y deja spirv como estaba.
bool optimize_spirv(std::vector<uint32_t>& spirv, std::string& error);

// Cada etapa numera sus recursos desde su propia base (todas comparten el set 0 del pipeline).
constexpr uint32_t kBindingsPerStage = 64;
uint32_t binding_base(Stage s);

}  // namespace nvnshader
