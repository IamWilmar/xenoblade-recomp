// SPDX-License-Identifier: GPL-3.0-or-later
// nvnProgram*: el motor usa un programa por etapa. Cada etapa trae su código (dirección de GPU =
// de CPU, dentro de un memory pool) y su sección de control. Al fijar los shaders se traducen a
// SPIR-V con el recompilador de Eden (runtime/recompiler) y se crea el VkShaderModule.
// Los que fallan se vuelcan a <datos>/shaders/ para estudiarlos con tools/shadertool.
// Las traducciones se guardan en nvn::spirv_cache (memoria y <datos>/spirv.bin): el mismo código con las mismas opciones
// no se vuelve a traducir, ni en esta partida ni en las siguientes con el mismo traductor.

#include <pthread.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/spirv_cache.h"
#include "nvn_shader.h"
#include "nvn_traductor_huella.h"  // NVN_TRADUCTOR_HUELLA (la genera runtime/recompiler/CMakeLists.txt)

namespace {

struct ShaderData {
    uint64_t code;        // dirección del código
    const void* control;  // sección de control
};

nvn::Objects<nvn::Program> g_programs;
std::atomic<int> g_translated{0}, g_failed{0};
// DEPURACIÓN: huellas distintas traducidas. Si son muchas menos que las traducciones, el juego crea varios programas con
// el mismo código y cada uno se traduce (2,5-4 ms en NvnSubmit) y lleva sus propias pipelines (ver translate_stage).
std::mutex g_prints_mutex;
std::unordered_set<uint64_t> g_prints;
std::atomic<int64_t> g_translate_us{0};
std::atomic<int> g_cache_hits{0};
std::atomic<int64_t> g_cache_us{0};
std::atomic<int> g_opt_done{0}, g_opt_failed{0};  // traducciones pasadas por spirv-opt y las que no pudo
std::atomic<int64_t> g_opt_us{0};
// El hilo de la precarga (prefetch_loop): sus fallos no se cuentan ni se vuelcan (el código pudo cambiar o liberarse entre
// SetShaders y la traducción; el ejecutor lo vuelve a intentar al enlazarlo).
thread_local bool t_prefetch = false;

// Cambia si cambia lo que entra en la clave de la caché (translation_key) o cómo se usa lo traducido.
// 2: Options::optimize_stages (spirv-opt).
constexpr uint32_t kKeyVersion = 5;  // 3: gl_FragDepth sin convertir de DepthMode; 4: texturas enteras (Xenoblade); 5: transform feedback emulado

// <datos>/spirv.bin: se empieza a leer con el primer programa del juego, bastante antes de su primer draw.
// <datos>/sin_cache_spirv al arrancar: sin fichero (y, en vivo, sin buscar ni guardar: se traduce todo como antes).
void open_spirv_cache() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (NVN_FLAG("sin_cache_spirv")) return;
        const char* data = getenv("NIER_DATA");
        nvn::spirv_cache::global().open(std::string(data ? data : "/sdcard/NieR") + "/spirv.bin", NVN_TRADUCTOR_HUELLA);
    });
}

// Copia hasta size bytes sin caerse si la memoria se acaba antes (devuelve lo copiado).
size_t safe_copy(void* dst, const void* src, size_t size) {
    iovec local{dst, size}, remote{const_cast<void*>(src), size};
    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return n > 0 ? static_cast<size_t>(n) : 0;
}

void dump_code(uint64_t code, const std::string& name) {
    const char* data = getenv("NIER_DATA");
    std::string dir = std::string(data ? data : "/sdcard/NieR") + "/shaders";
    mkdir(dir.c_str(), 0775);
    std::vector<uint8_t> buf(64 * 1024);
    size_t n = safe_copy(buf.data(), reinterpret_cast<const void*>(code), buf.size());
    if (FILE* f = fopen((dir + "/" + name + ".code").c_str(), "wb")) {
        fwrite(buf.data(), 1, n, f);
        fclose(f);
    }
}

// Hasta dónde se puede leer código: el final del memory pool que lo contiene.
size_t code_bound(uint64_t address) {
    size_t bound = 0;
    nvn::g_memory_pools.find_if([&](const void*, const nvn::MemoryPool& p) {
        auto start = reinterpret_cast<uint64_t>(p.storage);
        if (address < start || address >= start + p.size) return false;
        bound = start + p.size - address;
        return true;
    });
    return bound;
}

// Dónde acaba el código: donde empiezan sus constantes, code + control[0x700]. También sin constantes (c1 vacía: en la
// sección de control 0x700 == 0x704, ensure_translated no le pone constants): en MH Rise son el 39 % de los shaders
// volcados (266 de 680, casi todos los vertex y muchos compute), que antes no tenían huella (0) ni clave de la caché de
// SPIR-V y se volvían a traducir siempre, en cada programa repetido y en cada partida (segunda partida con la caché, 4-11:
// 824 de 1700 etapas no salieron de ella). Lo que el traductor lee acaba ahí: con todo lo de detrás cambiado al azar, los
// 680 shaders volcados salen iguales (comprobado en el Mac con el traductor de este commit). 0 si no se sabe (sin sección
// de control).
size_t code_length(const nvn::ProgramStage& stage) {
    if (stage.constants > stage.code) return static_cast<size_t>(stage.constants - stage.code);
    if (!stage.control) return 0;
    uint32_t begin = 0;
    memcpy(&begin, static_cast<const uint8_t*>(stage.control) + 0x700, 4);
    return begin <= 0x100000 ? begin : 0;
}

// Huella del código de un shader (FNV-1a hasta donde empiezan sus constantes): identifica el programa entre partidas.
// bound: hasta dónde se puede leer (code_bound).
uint64_t code_print(const nvn::ProgramStage& stage, size_t bound) {
    const size_t n = code_length(stage);
    if (!n || n > bound) return 0;
    const auto* p = reinterpret_cast<const uint8_t*>(stage.code);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < n; k++) h = (h ^ p[k]) * 0x100000001b3ULL;
    return h;
}

// ¿Está la huella en <datos>/<file>? Una huella hexadecimal por línea, o "*": todas. Cada lista se lee una vez.
bool listed_in(const char* file, uint64_t print) {
    static std::mutex m;
    static std::unordered_map<std::string, std::vector<uint64_t>> lists;
    std::lock_guard lock(m);
    auto [it, fresh] = lists.try_emplace(file);
    if (fresh) {
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/" + file).c_str(), "r")) {
            char line[64];
            while (fgets(line, sizeof line, f)) it->second.push_back(line[0] == '*' ? ~0ULL : strtoull(line, nullptr, 16));
            fclose(f);
        }
    }
    for (uint64_t x : it->second)
        if (x == print || x == ~0ULL) return true;
    return false;
}

// Compute con switch que se traducen con sus tablas de c1 a pesar de todo: los de <datos>/con_c1.
bool c1_allowed(uint64_t print) { return listed_in("con_c1", print); }

// Etapas cuyo SPIR-V pasa por el optimizador de SPIRV-Tools (nvnshader::Options::optimize_stages; recompiler/spirv_opt.cpp:
// el -O de spirv-opt sin tocar la interfaz ni los bindings). El driver de Qualcomm compila mal el flujo con banderas que
// deja Eden al deshacer un BRX (el skinning de la cara del cazador, 793657b4, tomaba el caso de los pesos a cero) y con
// -O sale bien. Los compute; con <datos>/spirv_opt_todas, todas las etapas. <datos>/sin_spirv_opt: ninguna, como antes.
// Se miran una vez, con el primer shader (al arrancar): van en la clave de la caché, así que cambiarlos solo retraduce.
uint32_t spirv_opt_stages() {
    static const uint32_t stages = [] {
        uint32_t s = 1u << static_cast<uint32_t>(nvnshader::Stage::Compute);
        if (NVN_FLAG("sin_spirv_opt")) s = 0;
        else if (NVN_FLAG("spirv_opt_todas")) s = 0x3F;
        NVN_LOG("spirv-opt: %s\n", !s ? "apagado (sin_spirv_opt)" : s == 0x3F ? "todas las etapas (spirv_opt_todas)" : "compute");
        return s;
    }();
    return stages;
}

// Lo que hizo spirv-opt con una traducción recién hecha (las de la caché no lo dicen): cuenta, y una línea por huella.
void note_spirv_opt(uint64_t print, const nvnshader::Translated& t) {
    if (!t.spirv_opt) return;
    (t.spirv_opt > 0 ? g_opt_done : g_opt_failed)++;
    g_opt_us += t.spirv_opt_us;
    static std::mutex m;
    static std::unordered_set<uint64_t> seen;
    {
        std::lock_guard lock(m);
        if (!seen.insert(print).second) return;
    }
    if (t.spirv_opt > 0)
        NVN_LOG("shader %016llx (%s): spirv-opt, %u → %zu palabras en %.1f ms\n", static_cast<unsigned long long>(print),
                nvnshader::stage_name(t.stage), t.spirv_opt_from, t.spirv.size(), t.spirv_opt_us / 1000.0);
    else
        NVN_LOG("shader %016llx (%s): spirv-opt no pudo (%s, %.1f ms); queda el SPIR-V de Eden\n",
                static_cast<unsigned long long>(print), nvnshader::stage_name(t.stage), t.spirv_opt_error.c_str(),
                t.spirv_opt_us / 1000.0);
}

// Compute cuyas barreras "en flujo condicional" están en bucles uniformes (el mismo número de vueltas en todo el grupo,
// sin returns antes: Eden las marca solo por estar dentro del bucle): dejarlas es lo correcto, y sin ellas la memoria
// compartida queda en carrera entre los waves de 64 del Adreno. Se dejan siempre, sin <datos>/con_barrera (que sigue
// valiendo para otras huellas). La traducción sin ellas se guarda en ProgramStage::stripped_module: cada una tiene su
// interruptor en vivo para volver a ella, y <datos>/sin_barrera_uniforme las vuelve todas.
// - 4be66a3c6a29b5be: la exposición automática (grupo de 1024, dispatch 1 1 1 tras el histograma 7 4 1 del 60x33).
//   Suma prefija Hillis-Steele de los 1024 bins en memoria compartida (paso 1, 2, 4… 512: leer el vecino, barrera,
//   escribir, barrera; la salida del bucle solo depende del paso), primer bin B con acumulado > total·c3[6], medida =
//   c3[2]·c3[3]^((B − 0,5)/1024) media con las 8 anteriores (s2: exposición + 8 medidas) y exposición = 1/media. Sin las
//   dos barreras del bucle los 16 waves suman a destiempo: en Kamura de día B salía ~704 (medida 0,827, exposición 1,21)
//   cuando el 91 % del 60x33 está bajo c3[2] = 0,15 (B = 0, exposición 6,67). El tonemap usa s0·ubo1[0] (EV de la escena
//   en el ubo1 del vertex shader: 0,574 en Kamura), así que salía 0,69 en vez de 3,83: unas 5,5 veces más oscuro. El
//   título no lo nota (rango [1, 1]: exposición 1) y la selección de personaje tampoco (no mide: usa la última
//   exposición, con EV 4).
// - acb571ff14d6945f: reducciones en árbol de 8 pasos (grupo de 256, min/max de floats en memoria compartida, una
//   barrera por vuelta, dos bucles con contador de 7 a 0). Mismo defecto; qué calcula aún sin confirmar.
struct UniformBarrierShader {
    uint64_t print;
    const char* name;  // para el log
};
constexpr UniformBarrierShader kUniformBarrierShaders[] = {
    {0x4be66a3c6a29b5beull, "exposición automática (sin_barrera_exposicion)"},
    {0xacb571ff14d6945full, "reducción de 256 (sin_barrera_reduccion)"},
};

int uniform_barrier_entry(uint64_t print) {
    for (size_t k = 0; k < std::size(kUniformBarrierShaders); k++)
        if (kUniformBarrierShaders[k].print == print) return static_cast<int>(k);
    return -1;
}

// Clave de la caché: todo lo que decide la traducción. El código hasta sus constantes (lo que hay tras ellas no es del
// programa), las constantes que se le dan (c1), las opciones (las del dispositivo y los interruptores incluidos) y las
// listas y variables de depuración que la cambian. Sin sección de control no se sabe dónde acaba el código: sin clave.
bool translation_key(const nvn::ProgramStage& stage, size_t bound, const nvnshader::Options& o, bool keep_barriers,
                     nvn::spirv_cache::Key& key) {
    const size_t n = code_length(stage);
    if (!n || n > bound) return false;
    nvn::spirv_cache::Hasher h;
    h.u32(kKeyVersion);
    h.u64(n);
    h.bytes(reinterpret_cast<const void*>(stage.code), n);
    h.u32(o.spirv_version);
    h.u32(o.texture_bound_buffer);
    h.u32(o.convert_depth_mode);
    h.u32(o.compute_code_offset);
    h.u32(o.compute_hint);
    for (uint32_t w : o.workgroup) h.u32(w);
    h.u32(o.shared_memory);
    h.u32(o.local_memory);
    h.u32(o.bindless_arrays);
    h.u32(o.nonuniform_indexing);
    h.u64(o.previous_stores);
    h.u32(static_cast<uint32_t>(o.input_topology));
    h.u32(o.subgroup_size);
    h.u32(o.subgroup_stages);
    h.u32(o.subgroup_operations);
    h.u32(o.broken_subgroup_shuffle);
    h.u32(o.broken_unreachable);
    h.u32(o.shared_to_local);
    h.u32(o.optimize_stages);
    h.u32(o.constants ? o.constants_size : ~0u);
    if (o.constants) h.bytes(o.constants, o.constants_size);
    h.u32(keep_barriers);
    const char* no_discard = getenv("NIER_SIN_DESCARTE");
    h.u32(no_discard && *no_discard);
    // La variante de logic_or_variant (OR emulado): solo si la hay, así las claves de siempre no cambian (la caché sigue).
    if (o.logic_or_targets) h.u32(0x4c4f5200u | o.logic_or_targets);
    // La variante entera (integer_variant) y <datos>/sin_imagenes_enteras: solo si las hay (las claves de siempre igual).
    if (o.unsigned_color_targets || o.signed_color_targets || !o.integer_handles.empty()) {
        h.u32(0x494e5400u);
        h.u32(o.unsigned_color_targets);
        h.u32(o.signed_color_targets);
        h.u32(static_cast<uint32_t>(o.integer_handles.size()));
        for (const auto& [bank, offset] : o.integer_handles) h.u32(bank), h.u32(offset);
    }
    if (!o.integer_images) h.u32(0x494d4700u);
    key = h.key();
    return true;
}

// bound: hasta dónde se puede leer el código; 0, el final de su memory pool (la precarga traduce una copia suya).
bool translate_stage(nvn::ProgramStage& out, uint64_t previous_stores = ~0ull, int input_topology = 0, size_t bound = 0) {
    if (!bound) bound = code_bound(out.code);
    if (!bound) {
        if (!t_prefetch) NVN_LOG("shader en 0x%llx fuera de cualquier memory pool\n", static_cast<unsigned long long>(out.code));
        return false;
    }
    auto start = std::chrono::steady_clock::now();
    nvnshader::Options options;
    options.convert_depth_mode = nvn::g_depth_mode == 0;
    if (out.control) {  // compute: inicio del código (+0x708), workgroup (+0x748) y memoria compartida (+0x754)
        const auto* control = static_cast<const uint8_t*>(out.control);
        memcpy(&options.compute_code_offset, control + 0x708, 4);
        memcpy(options.workgroup, control + 0x748, 12);
        memcpy(&options.shared_memory, control + 0x754, 4);
        for (uint32_t& w : options.workgroup) w = std::max(w, 1u);
        uint32_t stage = 0;
        memcpy(&stage, control + 0x714, 4);
        options.compute_hint = stage == 5;
    }
    // c1 (constantes tras el código): Eden lee de ahí las tablas de salto de los BRX (switch del shader). En los shaders
    // gráficos sí: sin ellas los casos no se traducen y el valor del switch queda con basura (la piel mojada de 2B y 9S,
    // con el azul a 13000-64512 y un halo de bloom enorme). En compute no: el único con switch que hemos visto (huella
    // 3e672e96e2eb85c5, un evaluador de curvas de 12 casos) con sus casos colgaba la GPU (en la demo del título, al
    // cargar las Ruinas, en combate) y sus resultados salían magenta (menú, salpicaduras, objetos resaltados). Sin
    // tablas cae siempre en el mismo camino, como hasta ahora. <datos>/con_c1 añade huellas (o "*": todos).
    uint32_t magic = 0;
    memcpy(&magic, reinterpret_cast<const void*>(out.code), 4);
    // Los compute de MH Rise no llevan la cabecera y sí usan sus tablas de c1: el skinning de los personajes es un compute
    // con switch y sin tablas los cuerpos desaparecen (solo quedaba el pelo). La excepción de NieR no aplica a ellos.
    const bool compute = magic == nvnshader::kNvnComputeCodeMagic;
    const uint64_t print = code_print(out, bound);
    out.print = print;
    {
        std::lock_guard lock(g_prints_mutex);
        g_prints.insert(print);
    }
    if (out.constants && (!compute || c1_allowed(print))) {
        options.constants = reinterpret_cast<const uint8_t*>(out.constants);
        options.constants_size = out.constants_size;
    }
    options.previous_stores = previous_stores;
    options.input_topology = input_topology;
    options.logic_or_targets = out.logic_or_targets;  // logic_or_variant (0 en las etapas del juego)
    options.unsigned_color_targets = out.unsigned_color_targets;  // integer_variant (0 y vacío en las del juego)
    options.signed_color_targets = out.signed_color_targets;
    options.integer_handles = out.integer_handles;
    // Imágenes con formato entero en la instrucción o con atómicos, declaradas uimage (nvnshader::Options::integer_images):
    // el R32UI de las sondas IBL se declaraba float y los 35 pixel shaders con imageAtomicOr de shd_lgt.wishp no validaban.
    // <datos>/sin_imagenes_enteras al traducir (al arrancar): como antes.
    options.integer_images = !NVN_FLAG("sin_imagenes_enteras");
    if (nvn::VkContext* c = nvn::vk()) {
        options.subgroup_size = c->subgroup.subgroupSize;
        options.subgroup_stages = c->subgroup.supportedStages;
        options.subgroup_operations = c->subgroup.supportedOperations;
        // Como Eden con el driver propietario de Qualcomm (vendorID 0x5143): shuffle con índice dinámico emulado.
        // <datos>/sin_shuffle_qualcomm al traducir (al arrancar): el OpGroupNonUniformShuffle de siempre.
        options.broken_subgroup_shuffle = c->props.vendorID == 0x5143 && !NVN_FLAG("sin_shuffle_qualcomm");
        // <datos>/sin_return_qualcomm al traducir (al arrancar): el OpUnreachable de siempre.
        options.broken_unreachable = c->props.vendorID == 0x5143 && !NVN_FLAG("sin_return_qualcomm");
        // <datos>/sin_no_uniforme al traducir: sin la decoración NonUniform (si el driver se lleva mal con ella).
        options.nonuniform_indexing = c->nonuniform_sampled && !NVN_FLAG("sin_no_uniforme");
    }
    // Bindless de los uniform buffers del juego (arrays hasta el final del cbuf, máscara y sampler fijo del handle; ver
    // nvnshader::Options). <datos>/sin_bindless_arrays al traducir (al arrancar): como antes; los arrays que ya hacía Eden
    // siguen en el set 1. <datos>/sin_bindless apaga la resolución en vivo.
    options.bindless_arrays = !NVN_FLAG("sin_bindless_arrays");
    // Memoria compartida privada de cada invocación (spill del compilador de NVN) a memoria local: el skinning de la cara
    // (793657b4f1e65631) salía NaN en el Adreno con esos stores a Workgroup en los casos de su switch.
    // <datos>/sin_compartida_privada al traducir (al arrancar): se queda en Workgroup, como antes.
    options.shared_to_local = !NVN_FLAG("sin_compartida_privada");
    // spirv-opt (spirv_opt_stages). <datos>/sin_spirv_opt_huellas: sin él para esas huellas (una por línea), para aislar
    // un shader que salga mal.
    options.optimize_stages = spirv_opt_stages();
    if (options.optimize_stages && listed_in("sin_spirv_opt_huellas", print)) options.optimize_stages = 0;
    // Barreras de grupo dentro de un if o un bucle: Eden las quita (una barrera que no todos alcanzan cuelga la GPU),
    // pero sin ellas la memoria compartida queda en carrera. Se dejan en las de kUniformBarrierShaders (bucles uniformes)
    // y en las de <datos>/con_barrera (huellas o "*"). De las primeras también se guarda la traducción sin ellas
    // (stripped), para sus interruptores en vivo (use_stripped_module); en la caché, con su propia clave (keep = false).
    const int barrier_entry = uniform_barrier_entry(print);
    const bool keep = barrier_entry >= 0 || listed_in("con_barrera", print);
    out.uniform_barrier_entry = static_cast<int8_t>(barrier_entry);
    open_spirv_cache();
    nvn::spirv_cache::Key key, stripped_key;
    const bool cacheable = !NVN_FLAG("sin_cache_spirv") && translation_key(out, bound, options, keep, key);
    // La clave de la traducción sin barreras: la que se guarda de las de la lista y la que queda si con ellas no se puede.
    const bool stripped_cacheable = cacheable && keep && translation_key(out, bound, options, false, stripped_key);
    const bool cached = cacheable && nvn::spirv_cache::global().find(key, out.shader);
    if (cached) out.cache_key = key;  // el precalentado de pipelines la apunta
    nvnshader::Translated stripped;
    bool have_stripped = false, barriers_kept = cached;
    bool ok = cached;
    if (!cached) {
        ok = nvnshader::translate(reinterpret_cast<const uint8_t*>(out.code), bound, options, out.shader);
        // Con qué clave queda out.shader en la caché: la de keep, salvo que con las barreras no se haya podido traducir
        // (entonces es la de sin ellas, y la de keep se vuelve a intentar en la partida siguiente).
        const nvn::spirv_cache::Key* final_key = cacheable ? &key : nullptr;
        if (ok && out.shader.shared_to_local)
            NVN_LOG("shader %016llx (%s): memoria compartida privada de cada invocación a memoria local (%d accesos)\n",
                    static_cast<unsigned long long>(print), nvnshader::stage_name(out.shader.stage), out.shader.shared_to_local);
        if (ok && out.shader.conditional_barriers) {
            NVN_LOG("shader %016llx (%s): %d barreras en flujo condicional, %s%s%s\n", static_cast<unsigned long long>(print),
                    nvnshader::stage_name(out.shader.stage), out.shader.conditional_barriers, keep ? "se dejan" : "quitadas",
                    barrier_entry >= 0 ? ", bucle uniforme conocido: " : "",
                    barrier_entry >= 0 ? kUniformBarrierShaders[barrier_entry].name : "");
            char name[40];
            snprintf(name, sizeof name, "barrera_%016llx", static_cast<unsigned long long>(print));
            dump_code(out.code, name);
            if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), std::string(name) + "_ctrl");
            if (keep) {
                options.keep_conditional_barriers = true;
                nvnshader::Translated kept;
                // Las barreras no cambian los recursos (el mismo layout vale para las dos). Si con ellas no se puede
                // traducir, se queda la de siempre (antes el fallo dejaba el shader sin traducir).
                if (nvnshader::translate(reinterpret_cast<const uint8_t*>(out.code), bound, options, kept) &&
                    kept.resources.size() == out.shader.resources.size()) {
                    if (barrier_entry >= 0) {
                        stripped = std::move(out.shader);
                        have_stripped = true;
                        if (stripped_cacheable) nvn::spirv_cache::global().insert(stripped_key, stripped);
                    }
                    out.shader = std::move(kept);
                    barriers_kept = true;
                } else {
                    NVN_LOG("shader %016llx: con las barreras no se pudo traducir (%s); se quedan quitadas\n",
                            static_cast<unsigned long long>(print), kept.error.c_str());
                    final_key = stripped_cacheable ? &stripped_key : nullptr;
                }
            }
        } else if (ok) {
            barriers_kept = true;  // no tenía: con o sin keep es la misma
        }
        if (ok && final_key) {
            nvn::spirv_cache::global().insert(*final_key, out.shader);
            out.cache_key = *final_key;
        }
        if (ok) note_spirv_opt(print, out.shader);
    }
    // Reemplazo: <datos>/spirv_reemplazo/<huella>.spv, el mismo SPIR-V (mismos recursos y bindings) pasado por spirv-opt
    // fuera. El driver de Qualcomm compila mal el flujo con banderas que deja Eden al deshacer un BRX en el skinning de la
    // cara del cazador (793657b4): toma el caso que pone los pesos a cero (1/0 → NaN y la cabeza no se dibuja); con el
    // switch reconstruido por spirv-opt -O sale bien (el intérprete da lo mismo bit a bit que el original). Es el mismo
    // shader para cualquier cara. Ahora lo hace spirv_opt_stages en la Flip con todos los compute (sale el mismo módulo);
    // si hay fichero, gana el fichero. <datos>/sin_reemplazo_spirv al arrancar: el traducido de siempre.
    const bool variant = out.logic_or_targets || out.unsigned_color_targets || out.signed_color_targets ||
                         !out.integer_handles.empty();
    if (ok && !NVN_FLAG("sin_reemplazo_spirv") && !variant) {  // el reemplazo es de la etapa del juego
        const char* data = getenv("NIER_DATA");
        char file[96];
        snprintf(file, sizeof file, "/spirv_reemplazo/%016llx.spv", static_cast<unsigned long long>(print));
        if (FILE* f = fopen((std::string(data ? data : ".") + file).c_str(), "rb")) {
            std::vector<uint32_t> words;
            uint32_t w;
            while (fread(&w, 4, 1, f) == 1) words.push_back(w);
            fclose(f);
            if (words.size() > 5 && words[0] == 0x07230203u) {
                out.shader.spirv = std::move(words);
                NVN_LOG("shader %016llx (%s): SPIR-V reemplazado por <datos>%s (%zu palabras)\n", static_cast<unsigned long long>(print),
                        nvnshader::stage_name(out.shader.stage), file, out.shader.spirv.size());
            }
        }
    }
    if (cached && barrier_entry >= 0) {  // de la caché: la sin barreras, de la suya o traducida ahora (una vez por partida)
        have_stripped = stripped_cacheable && nvn::spirv_cache::global().find(stripped_key, stripped);
        if (!have_stripped) {
            nvnshader::Options plain = options;
            plain.keep_conditional_barriers = false;
            have_stripped = nvnshader::translate(reinterpret_cast<const uint8_t*>(out.code), bound, plain, stripped) &&
                            stripped.resources.size() == out.shader.resources.size();
            if (have_stripped && stripped_cacheable) nvn::spirv_cache::global().insert(stripped_key, stripped);
        }
    }
    if (ok && barrier_entry >= 0)
        NVN_LOG("shader %016llx: %s, %s%s\n", static_cast<unsigned long long>(print), kUniformBarrierShaders[barrier_entry].name,
                barriers_kept ? "con las barreras de sus bucles" : "SIN las barreras de sus bucles (no se pudo con ellas)",
                barriers_kept && !have_stripped ? " (sin la traducción de antes para su interruptor)" : "");
    const int64_t us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (cached) {
        g_cache_hits++;
        g_cache_us += us;
    } else {
        g_translate_us += us;
        nvn::DropStats& d = nvn::g_drop_stats;
        (t_prefetch ? d.prefetched : d.translated_live).fetch_add(1, std::memory_order_relaxed);
        (t_prefetch ? d.prefetch_ns : d.translated_live_ns).fetch_add(static_cast<uint64_t>(us) * 1000, std::memory_order_relaxed);
    }
    if (!ok && t_prefetch) return false;
    if (!ok) {
        int n = g_failed++;
        NVN_LOG("shader %d no se pudo traducir: %s\n", n, out.shader.error.c_str());
        if (n < 32) {
            dump_code(out.code, "fallo" + std::to_string(n));
            if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), "fallo" + std::to_string(n) + "_ctrl");
        }
        return false;
    }
    if (listed_in("volcar_huellas", print)) {  // DEPURACIÓN: el código de esas huellas, para shadertool
        char name[40];
        snprintf(name, sizeof name, "huella_%016llx", static_cast<unsigned long long>(print));
        dump_code(out.code, name);
        if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), std::string(name) + "_ctrl");
        NVN_LOG("shader %016llx (%s) volcado\n", static_cast<unsigned long long>(print), nvnshader::stage_name(out.shader.stage));
    }
    if (out.shader.has_brx) {  // para ampliar la lista de c1_allowed sin adivinar
        NVN_LOG("shader %016llx (%s) con switch: tablas de c1 %s\n", static_cast<unsigned long long>(print),
                nvnshader::stage_name(out.shader.stage), options.constants ? "sí" : "no");
        if (NVN_FLAG("volcar_brx")) {  // DEPURACIÓN: su código y su sección de control, para estudiarlo con shadertool
            char name[40];
            snprintf(name, sizeof name, "brx_%016llx", static_cast<unsigned long long>(print));
            dump_code(out.code, name);
            if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), std::string(name) + "_ctrl");
        }
    }
    nvn::VkContext* c = nvn::vk();
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = out.shader.spirv.size() * sizeof(uint32_t);
    info.pCode = out.shader.spirv.data();
    if (VkResult r = vkCreateShaderModule(c->device, &info, nullptr, &out.module); r != VK_SUCCESS) {
        NVN_LOG("vkCreateShaderModule: %s\n", nvn::vk_result_name(r));
        return false;
    }
    out.spirv_id = nvn::spirv_identity(out.shader);
    if (have_stripped) {  // la sin las barreras de sus bucles, para el interruptor en vivo
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = stripped.spirv.size() * sizeof(uint32_t);
        si.pCode = stripped.spirv.data();
        if (vkCreateShaderModule(c->device, &si, nullptr, &out.stripped_module) != VK_SUCCESS) out.stripped_module = VK_NULL_HANDLE;
    }
    int n = ++g_translated;
    if (n % 100 == 0) {
        size_t distinct = 0;
        {
            std::lock_guard lock(g_prints_mutex);
            distinct = g_prints.size();
        }
        NVN_LOG("%d shaders traducidos (%lld ms en total, %zu huellas distintas), %d fallidos; %d de la caché de SPIR-V "
                "(%lld ms); spirv-opt: %d, %d que no pudo (%lld ms)\n",
                n, static_cast<long long>(g_translate_us / 1000), distinct, g_failed.load(), g_cache_hits.load(),
                static_cast<long long>(g_cache_us / 1000), g_opt_done.load(), g_opt_failed.load(),
                static_cast<long long>(g_opt_us / 1000));
    }
    return true;
}

}  // namespace

nvn::Program* nvn::find_program(const void* program) { return g_programs.get(program); }

namespace {

// Las constantes del propio shader (su c1), tras el código: code + control[0x700] hasta code + control[0x704].
void set_stage_constants(nvn::ProgramStage& stage) {
    if (!stage.control) return;
    const auto* control = static_cast<const uint8_t*>(stage.control);
    uint32_t begin, end;
    memcpy(&begin, control + 0x700, 4);
    memcpy(&end, control + 0x704, 4);
    if (end > begin && end - begin <= 0x10000) {
        stage.constants = stage.code + begin;
        stage.constants_size = end - begin;
    }
}

// --- Precarga ---
// Cada etapa se traducía la primera vez que se enlazaba, en NvnSubmit con el frame parado: 2,5-4 ms por shader y, los
// compute (pasan por spirv-opt), hasta segundos: los dos de ~197000 palabras de un combate del 5-11 (01ad9d0bbb9cbfc2 y
// 22bdcda404f16a3f) costaron 2,5 y 2,7 s, el juego congelado. El juego fija los shaders al cargar lo que va a usar
// (monstruo, efectos), bastante antes de dibujar con ellos: un hilo de prioridad baja los traduce entonces, a la caché de
// SPIR-V (de ahí los saca ensure_translated en una fracción de ms), y de los compute crea también la pipeline (la toma
// warm_take). Si el ejecutor llega antes, traduce él como siempre. Solo los compute: son los caros (spirv-opt) y los
// gráficos (2,5-4 ms) no valen lo que cuestan; el juego fija ~6000 shaders de los que usa ~650, y traducirlos todos llenó
// la caché de SPIR-V (128 MB, en RAM) y, con dos hilos con spirv-opt de compute de 200000 palabras a la vez (cientos de MB
// cada uno), el lowmemorykiller cerró el juego en plena partida (5-11). Un hilo en los núcleos pequeños, lo último que se
// fijó primero, y nada si quedan menos de 1,5 GB libres en el sistema. Se empieza 50 ms después de SetShaders como pronto
// (el juego podría acabar de escribir el código justo después; si aun así cambia, la huella es otra y no estorba).
// APAGADA por defecto (<datos>/precarga_shaders, en vivo, la enciende): aun solo con los compute, en 25 minutos llenó
// spirv.bin hasta su tope (128 MB) y dobló pipelines.bin (35 → 72 MB, las pipelines que calienta); al guardarlos la SD
// se saturaba (69 % de espera de E/S) y el juego, que lee sus datos de la misma SD, iba a 1 fps en el tablón de misiones
// (5-11). Las pipelines gráficas en segundo plano (executor.cpp) son las que quitan los tirones del combate.
struct PrefetchJob {
    uint64_t code;
    const void* control;
    std::chrono::steady_clock::time_point at;
    int tries = 0;
};

// Lo que se traduce en la precarga: una copia del código (hasta el final de sus constantes y 4 KB más, sin pasar del
// memory pool) y de la sección de control, para que el juego no la cambie a medias. El 5-11 el primer arranque con la
// precarga cayó en un abort de Sirit a los 5 s: una cabecera de tess-control (MH Rise no tiene) que el juego aún estaba
// escribiendo. Ahora se copia dos veces con 20 ms entre medias y solo se traduce si sale igual, y de los gráficos solo los
// tipos de MH Rise (vertex, geometry y pixel); los compute no llevan cabecera (su tipo va en el control, +0x714 == 5).
struct Snapshot {
    std::vector<uint8_t> code, control;
};
constexpr size_t kControlBytes = 0x760;  // lo que se lee de ella (hasta +0x754)

bool take_snapshot(uint64_t code, const void* control, Snapshot& out) {
    if (!code || !control) return false;
    const size_t bound = code_bound(code);
    uint32_t begin = 0, end = 0;
    memcpy(&begin, static_cast<const uint8_t*>(control) + 0x700, 4);
    memcpy(&end, static_cast<const uint8_t*>(control) + 0x704, 4);
    if (!begin || begin > 0x100000 || end > begin + 0x10000) return false;
    const size_t n = std::min<size_t>(std::max(begin, end) + 0x1000, bound);
    if (n < begin) return false;
    const auto* c = reinterpret_cast<const uint8_t*>(code);
    out.code.assign(c, c + n);
    out.control.assign(static_cast<const uint8_t*>(control), static_cast<const uint8_t*>(control) + kControlBytes);
    return true;
}

bool compute_snapshot(const Snapshot& s) {
    uint32_t magic = 0, stage = 0;
    memcpy(&magic, s.code.data(), 4);
    memcpy(&stage, s.control.data() + 0x714, 4);
    return magic == nvnshader::kNvnComputeCodeMagic || (magic != nvnshader::kNvnCodeMagic && stage == 5);
}

long available_mb() {
    long kb = -1;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
        fclose(f);
    }
    return kb < 0 ? 1 << 20 : kb / 1024;
}

bool known_stage(const Snapshot& s) {
    uint32_t magic = 0, stage = 0;
    memcpy(&magic, s.code.data(), 4);
    memcpy(&stage, s.control.data() + 0x714, 4);
    if (magic == nvnshader::kNvnComputeCodeMagic) return true;
    if (magic != nvnshader::kNvnCodeMagic) return stage == 5;  // compute de MH Rise, sin cabecera
    if (s.code.size() < nvnshader::kNvnHeaderSize + 4) return false;
    uint32_t common0 = 0;
    memcpy(&common0, s.code.data() + nvnshader::kNvnHeaderSize, 4);
    const uint32_t type = common0 >> 10 & 0xF;  // SPH: 1 vertex, 2/3 teselación, 4 geometry, 5 pixel
    return type == 1 || type == 4 || type == 5;
}
std::mutex g_prefetch_mutex;
std::condition_variable g_prefetch_cv;
std::deque<PrefetchJob> g_prefetch_queue;
std::unordered_set<uint64_t> g_prefetch_seen;  // código y sección de control ya encolados (2200 etapas, ~500 códigos)

void prefetch_loop() {
    pthread_setname_np(pthread_self(), "PrecargaShaders");
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);  // por debajo de los hilos del juego
    nvn::pin_background_thread();  // núcleos pequeños (el que lo crea puede ser un hilo crítico)
    t_prefetch = true;
    for (;;) {
        PrefetchJob job;
        {
            std::unique_lock lock(g_prefetch_mutex);
            g_prefetch_cv.wait(lock, [] { return !g_prefetch_queue.empty(); });
            job = g_prefetch_queue.back();
            g_prefetch_queue.pop_back();
        }
        std::this_thread::sleep_until(job.at + std::chrono::milliseconds(50));
        if (!NVN_FLAG("precarga_shaders")) continue;
        Snapshot first, second;
        if (!take_snapshot(job.code, job.control, first) || !known_stage(first) || !compute_snapshot(first)) continue;
        if (available_mb() < 1536) {  // el sistema va justo: más tarde
            std::this_thread::sleep_for(std::chrono::seconds(2));
            job.at = std::chrono::steady_clock::now();
            std::lock_guard lock(g_prefetch_mutex);
            g_prefetch_queue.push_front(job);
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (!take_snapshot(job.code, job.control, second) || first.code != second.code || first.control != second.control) {
            if (++job.tries < 4) {  // aún cambia: más tarde (al fondo de la cola, que se atiende por el final)
                job.at = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                std::lock_guard lock(g_prefetch_mutex);
                g_prefetch_queue.push_front(job);
            }
            continue;
        }
        nvn::ProgramStage stage;
        stage.code = reinterpret_cast<uint64_t>(second.code.data());
        stage.control = second.control.data();
        set_stage_constants(stage);
        if (translate_stage(stage, ~0ull, 0, second.code.size()) && stage.shader.stage == nvnshader::Stage::Compute)
            nvn::prefetch_compute_pipeline(stage);
        const VkDevice device = nvn::vk()->device;
        if (stage.module) vkDestroyShaderModule(device, stage.module, nullptr);
        if (stage.stripped_module) vkDestroyShaderModule(device, stage.stripped_module, nullptr);
    }
}

void prefetch(uint64_t code, const void* control) {
    if (!code || !NVN_FLAG("precarga_shaders")) return;
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread(prefetch_loop).detach();  // vive lo que el proceso
    });
    const uint64_t id = code ^ (reinterpret_cast<uint64_t>(control) * 0x9E3779B97F4A7C15ull);
    {
        std::lock_guard lock(g_prefetch_mutex);
        if (g_prefetch_seen.size() > 100000) g_prefetch_seen.clear();
        if (!g_prefetch_seen.insert(id).second) return;
        g_prefetch_queue.push_back({code, control, std::chrono::steady_clock::now()});
    }
    g_prefetch_cv.notify_one();
}

}  // namespace

bool nvn::ensure_translated(ProgramStage& stage) {
    if (!stage.tried) {
        stage.tried = true;
        set_stage_constants(stage);
        translate_stage(stage);
    }
    return stage.module != VK_NULL_HANDLE;
}

bool nvn::use_stripped_module(const ProgramStage& stage) {
    if (!stage.stripped_module) return false;
    if (NVN_FLAG("sin_barrera_uniforme")) return true;
    switch (stage.uniform_barrier_entry) {  // el orden de kUniformBarrierShaders
        case 0: return NVN_FLAG("sin_barrera_exposicion");
        case 1: return NVN_FLAG("sin_barrera_reduccion");
        default: return false;
    }
}

nvn::ProgramStage::Variant nvn::geometry_variant(ProgramStage& stage, uint64_t previous_stores, int input_topology) {
    const uint64_t key = previous_stores << 3 ^ static_cast<uint64_t>(input_topology);
    if (auto it = stage.variants.find(key); it != stage.variants.end()) return it->second;
    ProgramStage variant;
    variant.code = stage.code;
    variant.control = stage.control;
    variant.constants = stage.constants;
    variant.constants_size = stage.constants_size;
    variant.logic_or_targets = stage.logic_or_targets;  // la de los genéricos de una variante con OR emulado, también
    variant.unsigned_color_targets = stage.unsigned_color_targets;  // y de una entera
    variant.signed_color_targets = stage.signed_color_targets;
    variant.integer_handles = stage.integer_handles;
    ProgramStage::Variant v;
    if (translate_stage(variant, previous_stores, input_topology)) {
        v.module = variant.module;
        v.spirv_id = variant.spirv_id;
        v.cache_key = variant.cache_key;
    }
    return stage.variants[key] = v;
}

const nvn::ProgramStage* nvn::logic_or_variant(const ProgramStage& stage, uint32_t targets) {
    // Por la huella del código (o su dirección si no la tiene): la misma etapa la comparten varios programas del juego.
    static std::mutex mutex;
    static std::map<std::pair<uint64_t, uint32_t>, std::unique_ptr<ProgramStage>> variants;
    std::lock_guard lock(mutex);
    std::unique_ptr<ProgramStage>& slot = variants[{stage.print ? stage.print : stage.code, targets}];
    if (!slot) {
        slot = std::make_unique<ProgramStage>();
        ProgramStage& v = *slot;
        v.code = stage.code;
        v.control = stage.control;
        v.constants = stage.constants;
        v.constants_size = stage.constants_size;
        v.logic_or_targets = targets;
        v.tried = true;
        const bool ok = translate_stage(v);
        NVN_LOG("pixel %016llx: variante con la operación lógica OR emulada en los render targets 0x%x: %s (%zu recursos, "
                "%zu los del juego)\n", static_cast<unsigned long long>(stage.print), targets,
                ok ? "traducida" : v.shader.error.c_str(), v.shader.resources.size(), stage.shader.resources.size());
    }
    return slot->module ? slot.get() : nullptr;
}

const nvn::ProgramStage* nvn::integer_variant(const ProgramStage& stage, uint32_t uint_targets, uint32_t sint_targets,
                                              uint64_t textures) {
    // Por la huella del código (o su dirección), el OR de la etapa de partida y las claves.
    using Key = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint64_t>;
    static std::mutex mutex;
    static std::map<Key, std::unique_ptr<ProgramStage>> variants;
    std::lock_guard lock(mutex);
    std::unique_ptr<ProgramStage>& slot =
        variants[{stage.print ? stage.print : stage.code, stage.logic_or_targets, uint_targets, sint_targets, textures}];
    if (!slot) {
        slot = std::make_unique<ProgramStage>();
        ProgramStage& v = *slot;
        v.code = stage.code;
        v.control = stage.control;
        v.constants = stage.constants;
        v.constants_size = stage.constants_size;
        v.logic_or_targets = stage.logic_or_targets;
        v.unsigned_color_targets = uint_targets;
        v.signed_color_targets = sint_targets;
        for (size_t i = 0; i < stage.shader.resources.size() && i < 64; i++)
            if (textures >> i & 1)
                v.integer_handles.push_back({stage.shader.resources[i].cbuf_index, stage.shader.resources[i].cbuf_offset});
        v.tried = true;
        const bool ok = translate_stage(v);
        // El ejecutor enlaza con los recursos de la etapa de partida: tienen que ser los mismos (solo cambia is_integer).
        bool same = ok && v.shader.resources.size() == stage.shader.resources.size();
        for (size_t i = 0; same && i < v.shader.resources.size(); i++) {
            const nvnshader::Resource &a = v.shader.resources[i], &b = stage.shader.resources[i];
            same = a.kind == b.kind && a.binding == b.binding && a.set == b.set && a.cbuf_index == b.cbuf_index &&
                   a.cbuf_offset == b.cbuf_offset && a.count == b.count;
        }
        NVN_LOG("pixel %016llx: variante entera (salidas uint 0x%x int 0x%x, texturas 0x%llx%s): %s\n",
                static_cast<unsigned long long>(stage.print), uint_targets, sint_targets, static_cast<unsigned long long>(textures),
                stage.logic_or_targets ? ", sobre la del OR" : "",
                !ok ? v.shader.error.c_str() : same ? "traducida" : "otros recursos: no se usa");
        if (ok && !same && v.module) {
            vkDestroyShaderModule(nvn::vk()->device, v.module, nullptr);
            v.module = VK_NULL_HANDLE;
        }
    }
    return slot->module ? slot.get() : nullptr;
}

// 8 bytes por paso y mezcla final (como hash_bytes del ejecutor): ~10 µs para el SPIR-V de un pixel shader de MH Rise.
uint64_t nvn::spirv_identity(const nvnshader::Translated& t) {
    uint64_t h = 0x84222325cbf29ce4ULL;
    auto mix = [&h](uint64_t w) {
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    };
    mix(static_cast<uint64_t>(t.stage) | uint64_t{t.has_bindless_set} << 8 | uint64_t{t.spirv.size()} << 16);
    const uint32_t* w = t.spirv.data();
    size_t k = 0;
    for (; k + 2 <= t.spirv.size(); k += 2) mix(uint64_t{w[k]} | uint64_t{w[k + 1]} << 32);
    if (k < t.spirv.size()) mix(w[k]);
    mix(t.input_generics);
    mix(t.output_generics);
    mix(t.resources.size());
    for (const nvnshader::Resource& r : t.resources)
        mix(uint64_t{static_cast<uint32_t>(r.kind)} | uint64_t{r.set} << 8 | uint64_t{r.binding} << 16 | uint64_t{r.count} << 40);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h ? h : 1;  // 0 es "sin traducir"
}

bool nvn::shader_cache_ready(int timeout_ms) {
    if (NVN_FLAG("sin_cache_spirv")) return false;
    open_spirv_cache();
    return nvn::spirv_cache::global().wait_loaded(std::chrono::milliseconds(timeout_ms));
}

bool nvn::cached_translation(const spirv_cache::Key& key, nvnshader::Translated& out) {
    return nvn::spirv_cache::global().find(key, out);
}

void nvn::dump_stage_code(const ProgramStage& stage, const std::string& name) {
    dump_code(stage.code, name);
    if (stage.control) dump_code(reinterpret_cast<uint64_t>(stage.control), name + "_ctrl");
}

extern "C" {

nvn::Boolean nvnProgramInitialize(void* program, void*) {
    open_spirv_cache();
    g_programs.create(program);
    return 1;
}
// El juego puede finalizar un programa con comandos que lo usan aún sin reproducir (Xenoblade DE, Valle de la Espada:
// BindProgram de un programa ya destruido leído por scan_pass -> fallo de segmentación en 'tried', offset 0x2c), y un
// pipeline nuevo de un draw pendiente necesitaría sus módulos. Como las texturas (nvnTextureFinalize): el objeto se saca
// de la tabla y se conserva, módulos incluidos (pocos programas se finalizan: al cambiar de escena).
void nvnProgramFinalize(void* program) {
    if (std::unique_ptr<nvn::Program> p = g_programs.take(program)) {
        static std::mutex m;
        static std::vector<std::unique_ptr<nvn::Program>> finalized;
        std::lock_guard lock(m);
        finalized.push_back(std::move(p));
        static int logged = 0;
        if (nvn::first_times(logged, 4))
            NVN_LOG("programa %p finalizado por el juego: se conserva por si queda grabado (%zu conservados)\n", program,
                    finalized.size());
    }
}

nvn::Boolean nvnProgramSetShaders(void* program, int count, const ShaderData* stages) {
    nvn::Program* p = g_programs.get(program);
    if (!p) return 0;
    for (nvn::ProgramStage& s : p->stages) {
        if (s.module) vkDestroyShaderModule(nvn::vk()->device, s.module, nullptr);
        if (s.stripped_module) vkDestroyShaderModule(nvn::vk()->device, s.stripped_module, nullptr);
    }
    p->stages.clear();
    p->stages.resize(count);
    // Solo se apunta la dirección: el juego puede terminar de escribir el código después (NVN lo lee
    // al dibujar), así que la etapa es del executor la primera vez que se enlaza; la precarga la deja traducida antes.
    for (int k = 0; k < count; k++) p->stages[k].code = stages[k].code, p->stages[k].control = stages[k].control;
    for (int k = 0; k < count; k++) prefetch(stages[k].code, stages[k].control);  // traducidas antes del primer draw
    // Única en toda la ejecución: un Program nuevo puede reutilizar la memoria de otro ya finalizado, y las
    // cachés del ejecutor (pipelines, layouts, sets) van por (ProgramStage*, generación).
    static std::atomic<uint64_t> serial{0};
    p->generation = ++serial;
    return 1;
}
// Etiquetas para herramientas de depuración de GPU: sin VK_EXT_debug_utils en el Adreno no hay quien las lea.
void nvnProgramSetDebugLabel(void*, const char*) {}

// NVNshaderDataExt: lo de NVNshaderData y un puntero a datos de depuración (24 bytes; como lo lee el port de MHGU), que
// aquí no se usan: las mismas etapas que SetShaders.
struct ShaderDataExt {
    uint64_t code;
    const void* control;
    const void* debug;
};
static_assert(sizeof(ShaderDataExt) == 24);
nvn::Boolean nvnProgramSetShadersExt(void* program, int count, const ShaderDataExt* stages) {
    std::vector<ShaderData> plain(count > 0 && stages ? count : 0);
    for (size_t k = 0; k < plain.size(); k++) plain[k] = {stages[k].code, stages[k].control};
    return nvnProgramSetShaders(program, static_cast<int>(plain.size()), plain.data());
}

// Subrutinas de GLSL: el traductor no las tiene (los shaders que las usen llaman siempre a la primera). Se acepta el
// enlace para que el programa siga valiendo y se avisa.
nvn::Boolean nvnProgramSetSubroutineLinkage(void*, int count, const void* const*) {
    static std::atomic<int> logged{0};
    if (count > 0 && nvn::first_times(logged, 4)) NVN_LOG("enlace de subrutinas (%d mapas): sin soporte\n", count);
    return 1;
}

// NVNsampleShadingMode (DEFAULT_FRAGMENT 0, FORCE_SAMPLE_SHADING_ON 1, FORCE_SAMPLE_SHADING_OFF 2): se guarda en el
// programa. Solo cambia algo con multisample, que el port no usa en sus pases (sampleShadingEnable sería otro campo de
// PipelineKey).
void nvnProgramSetSampleShading(void* program, int mode) {
    if (nvn::Program* p = g_programs.get(program)) p->sample_shading = mode;
}

}  // extern "C"

NVN_PROC(nvnProgramInitialize);
NVN_PROC(nvnProgramFinalize);
NVN_PROC(nvnProgramSetDebugLabel);
NVN_PROC(nvnProgramSetShaders);
NVN_PROC(nvnProgramSetShadersExt);
NVN_PROC(nvnProgramSetSubroutineLinkage);
NVN_PROC(nvnProgramSetSampleShading);
