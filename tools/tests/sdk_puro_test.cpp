// Prueba en el Mac (Apple Silicon) de las cuentas de nn::hid y nn::irsensor que el SDK hace dentro del proceso, contra
// el propio sdk del juego: carga work/scope/sdk.elf (AArch64, el mismo código que corre en la Switch) en memoria
// ejecutable, lo reloca contra sí mismo y llama a sus funciones con las mismas entradas que a las de libnnsdk. Compara
// byte a byte, también lo que ninguna de las dos debe tocar (los búferes van rellenos de 0xA5).
//
//   c++ -std=c++20 -O1 -I tools/tests/shim -I runtime/src tools/tests/sdk_puro_test.cpp runtime/src/nnsdk/hid.cpp \
//       -o /tmp/sdk_puro && /tmp/sdk_puro work/scope/sdk.elf
//
// Solo se llaman funciones del sdk que no hablan con ningún servicio ni escriben en sus propios datos (su imagen queda
// de solo lectura y ejecución): los ficheros .bnvib (sdk+0x220110 analizar, +0x220240 muestra, +0x2202d0 generar,
// +0x2204c0 tamaño), el argumento del applet de mandos (sdk+0x244d38 por defecto, +0x244bf0 texto), los handles de
// vibración y de sensor (sdk+0x22a6d0, +0x22a4b0), el cuaternión del sensor (sdk+0x223ae0), las configuraciones por
// defecto de irsensor (sdk+0x2b6850, +0x2b6a40, +0x2b6b10, +0x2b6b30), el handle de la cámara (sdk+0x2b04e0), los
// estados por defecto (sdk+0x2b4874, +0x2b5ec8) y la estadística de una región de momentos (sdk+0x2b6910). De todas,
// solo la última usa x18 (el contador de columnas de su bucle, sdk+0x2b6998 y +0x2b69d0), que macOS reserva y pone a
// cero cuando interrumpe al hilo: una pasada de cada muchos miles suma columnas de más, y por eso ahí una diferencia
// solo cuenta si se repite. Las direcciones son las del sdk de Disco Elysium 1.0.9 (build id 5bb30f7980902b19); con
// otro sdk la prueba no arranca.
//
// Lo que en la consola contesta el servicio hid (lo que el juego le deja dicho, la vibración, los eventos) no se
// puede comparar: de eso se comprueba aquí que libnnsdk sea coherente consigo misma.

#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../runtime/src/nnsdk/hid_vibration.cpp"
#include "../../runtime/src/nnsdk/irsensor.cpp"

#include "nnsdk/os.h"

// hid.cpp va en su propia unidad (repite tipos de hid_vibration.cpp): aquí, lo que se usa de él, con los mismos tipos.
namespace nn::util {
template <int N, typename Tag>
struct BitFlagSet {
    uint32_t storage[(N + 31) / 32];
};
namespace neon {
struct Vector4fType {
    float v[4];
};
}  // namespace neon
}  // namespace nn::util
namespace nn::hid {
struct NpadStyleTag;
using NpadStyleSet = util::BitFlagSet<32, NpadStyleTag>;
struct NpadFullKeyState;
struct NpadHandheldState;
struct NpadJoyDualState;
struct NpadJoyLeftState;
struct NpadJoyRightState;
struct ControllerSupportArg {
    uint8_t storage[0x430];
    void SetDefault();
};
struct ControllerSupportResultInfo {
    int8_t player_count;
    char pad[3];
    uint32_t selected_id;
};
struct ControllerFirmwareUpdateArg {
    bool enable_force_update;
    uint8_t reserved[3];
};
struct SixAxisSensorHandle {
    uint32_t storage;
};
struct SixAxisSensorState {
    int64_t delta_time, sampling_number;
    float acceleration[3], angular_velocity[3], angle[3], direction[9];
    uint32_t attributes, reserved;
    void GetQuaternion(util::neon::Vector4fType* out) const;
};
enum NpadHandheldActivationMode : int;
enum NpadCommunicationMode : int;
enum NpadJoyDeviceType : int;
enum GyroscopeZeroDriftMode : int;
void SetSupportedNpadStyleSet(NpadStyleSet);
NpadStyleSet GetSupportedNpadStyleSet();
NpadStyleSet GetNpadStyleSet(const uint32_t&);
void SetNpadHandheldActivationMode(NpadHandheldActivationMode);
NpadHandheldActivationMode GetNpadHandheldActivationMode();
void SetNpadCommunicationMode(NpadCommunicationMode);
NpadCommunicationMode GetNpadCommunicationMode();
void BindNpadStyleSetUpdateEvent(const uint32_t&, os::SystemEventType*, os::EventClearMode);
int GetNpadStates(NpadFullKeyState*, int, const uint32_t&);
int GetNpadStates(NpadHandheldState*, int, const uint32_t&);
int GetNpadStates(NpadJoyDualState*, int, const uint32_t&);
int GetNpadStates(NpadJoyLeftState*, int, const uint32_t&);
int GetNpadStates(NpadJoyRightState*, int, const uint32_t&);
bool SetNpadJoyAssignmentModeSingle(uint32_t*, const uint32_t&, NpadJoyDeviceType);
Result ShowControllerSupport(const ControllerSupportArg&);
Result ShowControllerSupport(ControllerSupportResultInfo*, const ControllerSupportArg&);
void SetExplainText(ControllerSupportArg*, const char*, const uint32_t&);
Result ShowControllerStrapGuide();
Result ShowControllerFirmwareUpdate(const ControllerFirmwareUpdateArg&);
int GetSixAxisSensorHandles(SixAxisSensorHandle*, int, const uint32_t&, NpadStyleSet);
void SetGyroscopeZeroDriftMode(const SixAxisSensorHandle&, const GyroscopeZeroDriftMode&);
GyroscopeZeroDriftMode GetGyroscopeZeroDriftMode(const SixAxisSensorHandle&);
void EnableSixAxisSensorFusion(const SixAxisSensorHandle&, bool);
bool IsSixAxisSensorFusionEnabled(const SixAxisSensorHandle&);
void SetSixAxisSensorFusionParameters(const SixAxisSensorHandle&, float, float);
void GetSixAxisSensorFusionParameters(float*, float*, const SixAxisSensorHandle&);
void ResetSixAxisSensorFusionParameters(const SixAxisSensorHandle&);
bool IsSixAxisSensorAtRest(const SixAxisSensorHandle&);
bool IsFirmwareUpdateAvailableForSixAxisSensor(const SixAxisSensorHandle&);
void GetSixAxisSensorState(SixAxisSensorState*, const SixAxisSensorHandle&);
int GetSixAxisSensorStates(SixAxisSensorState*, int, const SixAxisSensorHandle&);
int GetVibrationDeviceHandles(VibrationDeviceHandle*, int, const uint32_t&, NpadStyleSet);
void SendVibrationValue(const VibrationDeviceHandle&, const VibrationValue&);
void GetActualVibrationValue(VibrationValue*, const VibrationDeviceHandle&);
bool IsVibrationPermitted();
}  // namespace nn::hid

// Lo que los .cpp piden al resto de libnnsdk: el mando conectado (así el portátil existe y su cámara sale "sin cámara")
// y los eventos de nn::os, de los que aquí solo se apunta cómo se crean.
namespace {
NnsdkInput g_input;
struct CreatedEvent {
    nn::os::SystemEventType* event;
    bool auto_clear, signaled;
};
std::vector<CreatedEvent> g_created_events;
}  // namespace
namespace nnsdk {
const NnsdkHost& host() {
    static NnsdkHost h = [] {
        NnsdkHost x{};
        x.input = &g_input;
        x.game_delete = free;
        return x;
    }();
    return h;
}
void create_system_event(nn::os::SystemEventType* event, bool auto_clear, bool signaled) {
    g_created_events.push_back({event, auto_clear, signaled});
}
}  // namespace nnsdk

namespace {

// Cuenta por nombre de comprobación; de cada una enseña solo los primeros fallos.
struct Tally {
    const char* what;
    int checks, failures;
};
std::vector<Tally> g_tallies;
void expect(bool ok, const char* what, long long a = 0, long long b = 0) {
    Tally* t = nullptr;
    for (Tally& x : g_tallies)
        if (strcmp(x.what, what) == 0) t = &x;
    if (!t) t = &g_tallies.emplace_back(Tally{what, 0, 0});
    t->checks++;
    if (ok) return;
    if (t->failures++ < 4) printf("  FALLA %s (%lld, %lld)\n", what, a, b);
}

// --- El sdk en memoria ---

struct Elf64Phdr {
    uint32_t type, flags;
    uint64_t offset, vaddr, paddr, filesz, memsz, align;
};
struct Elf64Sym {
    uint32_t name;
    uint8_t info, other;
    uint16_t shndx;
    uint64_t value, size;
};
struct Elf64Rela {
    uint64_t offset, info;
    int64_t addend;
};

[[noreturn]] void unresolved() {
    printf("el sdk llamó a un import sin resolver\n");
    abort();
}

struct Sdk {
    uint8_t* base = nullptr;
    template <typename F>
    F at(uint64_t address) const {
        return reinterpret_cast<F>(base + address);
    }
};

Sdk load_sdk(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("no puedo abrir %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    std::vector<uint8_t> file(static_cast<size_t>(ftell(f)));
    fseek(f, 0, SEEK_SET);
    if (fread(file.data(), 1, file.size(), f) != file.size()) exit(2);
    fclose(f);
    uint64_t phoff;
    uint16_t phentsize, phnum;
    memcpy(&phoff, &file[0x20], 8);
    memcpy(&phentsize, &file[0x36], 2);
    memcpy(&phnum, &file[0x38], 2);
    std::vector<Elf64Phdr> phdrs(phnum);
    uint64_t size = 0, dynamic = 0;
    for (int k = 0; k < phnum; k++) {
        memcpy(&phdrs[k], &file[phoff + size_t(k) * phentsize], sizeof(Elf64Phdr));
        if (phdrs[k].type == 1 && phdrs[k].vaddr + phdrs[k].memsz > size) size = phdrs[k].vaddr + phdrs[k].memsz;
        if (phdrs[k].type == 2) dynamic = phdrs[k].vaddr;
    }
    size = (size + 0x3fff) & ~uint64_t{0x3fff};
    // MAP_JIT: lo único que macOS deja ejecutar sin firmar. Mientras se ejecuta, toda la imagen es de solo lectura.
    auto* base = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0));
    if (base == MAP_FAILED) {
        perror("mmap MAP_JIT");
        exit(2);
    }
    pthread_jit_write_protect_np(0);
    for (const Elf64Phdr& p : phdrs)
        if (p.type == 1) memcpy(base + p.vaddr, &file[p.offset], p.filesz);
    uint64_t rela = 0, relasz = 0, jmprel = 0, pltrelsz = 0, symtab = 0, strtab = 0;
    for (const uint8_t* d = base + dynamic;; d += 16) {
        int64_t tag;
        uint64_t value;
        memcpy(&tag, d, 8);
        memcpy(&value, d + 8, 8);
        if (tag == 0) break;
        if (tag == 7) rela = value;
        if (tag == 8) relasz = value;
        if (tag == 23) jmprel = value;
        if (tag == 2) pltrelsz = value;
        if (tag == 6) symtab = value;
        if (tag == 5) strtab = value;
    }
    int relative = 0, own = 0, foreign = 0;
    auto apply = [&](uint64_t table, uint64_t bytes) {
        for (uint64_t o = 0; o + sizeof(Elf64Rela) <= bytes; o += sizeof(Elf64Rela)) {
            Elf64Rela r;
            memcpy(&r, base + table + o, sizeof r);
            const uint32_t type = static_cast<uint32_t>(r.info);
            uint64_t value;
            if (type == 1027) {  // R_AARCH64_RELATIVE
                value = reinterpret_cast<uint64_t>(base) + static_cast<uint64_t>(r.addend);
                relative++;
            } else if (type == 1025 || type == 1026 || type == 257) {  // GLOB_DAT, JUMP_SLOT, ABS64
                Elf64Sym s;
                memcpy(&s, base + symtab + (r.info >> 32) * sizeof(Elf64Sym), sizeof s);
                if (s.shndx) {
                    value = reinterpret_cast<uint64_t>(base) + s.value + static_cast<uint64_t>(r.addend);
                    own++;
                } else {
                    value = reinterpret_cast<uint64_t>(&unresolved);
                    foreign++;
                }
            } else {
                continue;
            }
            memcpy(base + r.offset, &value, 8);
        }
    };
    apply(rela, relasz);
    apply(jmprel, pltrelsz);
    (void)strtab;
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(base, size);
    // Las primeras instrucciones de tres de las funciones que se van a llamar: si no son estas, es otro sdk.
    const struct {
        uint64_t address;
        uint32_t first, second;
    } fingerprints[] = {{0x2204c0, 0xb9400008, 0x531e7508}, {0x2b6850, 0x52927c08, 0x72a00088}, {0x244bf0, 0xb9400048, 0x7100411f}};
    for (const auto& f : fingerprints) {
        uint32_t code[2] = {};
        if (size >= f.address + 8) memcpy(code, base + f.address, sizeof code);
        if (code[0] != f.first || code[1] != f.second) {
            printf("%s no es el sdk de Disco Elysium 1.0.9 (sdk+0x%llx no coincide)\n", path, static_cast<unsigned long long>(f.address));
            exit(2);
        }
    }
    printf("sdk cargado: %llu MB, %d relocaciones relativas, %d a símbolos suyos, %d imports sin resolver\n",
           static_cast<unsigned long long>(size >> 20), relative, own, foreign);
    return {base};
}

// --- Ficheros de vibración ---

using SdkParse = uint64_t (*)(nn::hid::VibrationFileInfo*, void*, const void*, size_t);
using SdkRetrieve = uint64_t (*)(nn::hid::VibrationValue*, int, void*);
using SdkGenerate = uint64_t (*)(size_t*, uint8_t*, size_t, const nn::hid::VibrationValueArrayInfo&, const nn::hid::VibrationValue*);
using SdkFileSize = size_t (*)(const nn::hid::VibrationValueArrayInfo&);

void test_vibration_files(const Sdk& sdk, std::mt19937& rng) {
    const auto sdk_parse = sdk.at<SdkParse>(0x220110);
    const auto sdk_retrieve = sdk.at<SdkRetrieve>(0x220240);
    const auto sdk_generate = sdk.at<SdkGenerate>(0x2202d0);
    const auto sdk_size = sdk.at<SdkFileSize>(0x2204c0);
    std::uniform_real_distribution<float> amplitude(-0.2f, 1.3f), hz(0.5f, 3000.0f);
    int frequency_bytes_off = 0;
    int results_seen[8] = {};
    for (int round = 0; round < 4000; round++) {
        nn::hid::VibrationValueArrayInfo info{};
        info.sample_length = 1 + static_cast<int>(rng() % 48);
        info.is_loop = rng() % 2;
        if (info.is_loop) {
            info.loop_end_position = rng() % (static_cast<uint32_t>(info.sample_length) + 1);
            info.loop_start_position = rng() % (info.loop_end_position + 1);
            info.loop_interval = rng() % 3 ? static_cast<int>(rng() % 40) : 0;
        }
        std::vector<nn::hid::VibrationValue> values(static_cast<size_t>(info.sample_length));
        for (auto& v : values) {
            // De vez en cuando una frecuencia de las que salen exactas de un byte, un cero y un negativo.
            auto frequency = [&] {
                switch (rng() % 8) {
                    case 0: return 10.0f * std::exp2(static_cast<float>(rng() % 256) / 32.0f);
                    case 1: return 0.0f;
                    case 2: return -50.0f;
                    default: return hz(rng);
                }
            };
            v = {amplitude(rng), frequency(), amplitude(rng), frequency()};
        }
        const size_t expected = sdk_size(info);
        expect(expected == nn::hid::CalculateVibrationFileSize(info), "CalculateVibrationFileSize", expected);

        std::vector<uint8_t> a(expected + 32, 0xA5), b(expected + 32, 0xA5);
        size_t size_a = 777, size_b = 777;
        sdk_generate(&size_a, a.data(), expected, info, values.data());
        nn::hid::GenerateVibrationFile(&size_b, b.data(), expected, info, values.data());
        expect(size_a == size_b && size_a == expected, "GenerateVibrationFile: tamaño", size_a, size_b);
        // Solo puede diferir, y en 1, el byte de una frecuencia cuyo log2 cae justo entre dos: el log2f del sdk (el suyo,
        // de musl) y el de la libm de aquí no tienen por qué redondear igual.
        int other = 0;
        for (size_t k = 0; k < a.size(); k++)
            if (a[k] != b[k]) (std::abs(a[k] - b[k]) == 1 ? frequency_bytes_off : other)++;
        expect(other == 0, "GenerateVibrationFile: contenido", other);

        // El fichero bueno y unos cuantos estropeados, por los dos analizadores. Se parte del que generó el sdk.
        for (int variant = 0; variant < 8; variant++) {
            std::vector<uint8_t> file(a.begin(), a.begin() + static_cast<long>(expected));
            size_t size = expected;
            if (variant == 1) file[4] = 2;                                   // formato desconocido
            if (variant == 2) file[6] = 100;                                 // otra frecuencia de muestreo
            if (variant == 3) size = expected > 0x18 ? expected - 4 : expected;  // datos cortados
            if (variant == 4 && info.is_loop) file[0xc] = 0xff;               // el bucle acaba fuera
            if (variant == 5 && info.is_loop && info.loop_interval) file[0x13] = 0x80;  // pausa negativa
            if (variant == 6) file[0] = 3;                                   // metainformación imposible
            if (variant == 7 && !info.is_loop) size = 8;                      // ni la cabecera cabe
            nn::hid::VibrationFileInfo info_a, info_b;
            memset(&info_a, 0xA5, sizeof info_a);
            memset(&info_b, 0xA5, sizeof info_b);
            uint64_t context_a[10], context_b[10];
            memset(context_a, 0xA5, sizeof context_a);
            memset(context_b, 0xA5, sizeof context_b);
            const uint32_t result_a = static_cast<uint32_t>(sdk_parse(&info_a, context_a, file.data(), size));
            const nn::Result result_b = nn::hid::ParseVibrationFile(
                &info_b, reinterpret_cast<nn::hid::VibrationFileParserContext*>(context_b), file.data(), size);
            expect(result_a == result_b.value, "ParseVibrationFile: resultado", result_a, result_b.value);
            results_seen[result_a ? (result_a >> 9) - 140 : 0]++;  // 0 éxito, 1..5 los errores 141..145
            expect(memcmp(&info_a, &info_b, sizeof info_a) == 0, "ParseVibrationFile: info", variant, result_a);
            expect(memcmp(context_a, context_b, sizeof context_a) == 0, "ParseVibrationFile: contexto", variant, result_a);
            if (variant != 0 || result_a != 0) continue;
            expect(info_b.sample_length == info.sample_length, "ParseVibrationFile: muestras", info_b.sample_length);
            for (int k = 0; k < info.sample_length; k++) {
                nn::hid::VibrationValue va, vb;
                memset(&va, 0xA5, sizeof va);
                memset(&vb, 0xA5, sizeof vb);
                sdk_retrieve(&va, k, context_a);
                nn::hid::RetrieveVibrationValue(&vb, k, reinterpret_cast<nn::hid::VibrationFileParserContext*>(context_b));
                expect(memcmp(&va, &vb, sizeof va) == 0, "RetrieveVibrationValue", k);
            }
        }
    }
    // Todas las frecuencias y amplitudes que caben en un byte.
    for (int byte = 0; byte < 256; byte++) {
        const uint8_t sample[4] = {static_cast<uint8_t>(byte), static_cast<uint8_t>(byte), static_cast<uint8_t>(255 - byte),
                                   static_cast<uint8_t>(255 - byte)};
        struct {
            const uint8_t* samples;
            uint32_t reserved;
            int32_t count;
        } context{sample, 0, 1};
        nn::hid::VibrationValue va, vb;
        sdk_retrieve(&va, 0, &context);
        nn::hid::RetrieveVibrationValue(&vb, 0, reinterpret_cast<nn::hid::VibrationFileParserContext*>(&context));
        expect(memcmp(&va, &vb, sizeof va) == 0, "RetrieveVibrationValue: byte", byte);
    }
    printf("ficheros de vibración: %d bytes de frecuencia a 1 de distancia por el redondeo de log2f\n", frequency_bytes_off);
    printf("ParseVibrationFile: %d con éxito y %d, %d, %d, %d, %d con los errores 141 a 145\n", results_seen[0],
           results_seen[1], results_seen[2], results_seen[3], results_seen[4], results_seen[5]);
}

// --- irsensor ---

template <typename Config>
void test_default_config(const Sdk& sdk, uint64_t address, const char* name) {
    uint8_t a[sizeof(Config) + 16], b[sizeof(Config) + 16];
    memset(a, 0xA5, sizeof a);
    memset(b, 0xA5, sizeof b);
    sdk.at<void (*)(void*)>(address)(a);
    void (*mine)(Config*) = nullptr;
    if constexpr (std::is_same_v<Config, nn::irsensor::MomentProcessorConfig>) mine = nn::irsensor::GetMomentProcessorDefaultConfig;
    if constexpr (std::is_same_v<Config, nn::irsensor::ClusteringProcessorConfig>) mine = nn::irsensor::GetClusteringProcessorDefaultConfig;
    if constexpr (std::is_same_v<Config, nn::irsensor::ImageTransferProcessorConfig>) mine = nn::irsensor::GetImageTransferProcessorDefaultConfig;
    if constexpr (std::is_same_v<Config, nn::irsensor::ImageTransferProcessorExConfig>) mine = nn::irsensor::GetImageTransferProcessorDefaultConfig;
    mine(reinterpret_cast<Config*>(b));
    expect(memcmp(a, b, sizeof a) == 0, name);
}

void test_irsensor(const Sdk& sdk, std::mt19937& rng) {
    using namespace nn::irsensor;
    test_default_config<MomentProcessorConfig>(sdk, 0x2b6850, "GetMomentProcessorDefaultConfig");
    test_default_config<ClusteringProcessorConfig>(sdk, 0x2b6a40, "GetClusteringProcessorDefaultConfig");
    test_default_config<ImageTransferProcessorConfig>(sdk, 0x2b6b10, "GetImageTransferProcessorDefaultConfig");
    test_default_config<ImageTransferProcessorExConfig>(sdk, 0x2b6b30, "GetImageTransferProcessorDefaultConfig (Ex)");

    for (uint32_t id : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0x10u, 0x20u}) {
        uint32_t a = 0xA5A5A5A5;
        sdk.at<void (*)(uint32_t*, const uint32_t&)>(0x2b04e0)(&a, id);
        const IrCameraHandle h = GetIrCameraHandle(id);
        uint32_t b;
        memcpy(&b, &h, 4);
        expect(a == b, "GetIrCameraHandle", a, b);
    }

    // Los estados por defecto: la sesión del sdk solo aporta si la imagen va en negativo (+0xd0 + jugador) y la ventana
    // (+0xda + 8·jugador), que es lo que apunta RunMomentProcessor.
    const auto sdk_default_moment = sdk.at<void (*)(const uint8_t*, void*, const IrCameraHandle*)>(0x2b4874);
    const Rect windows[] = {{0, 0, 320, 240}, {16, 8, 160, 120}, {0, 0, 200, 100}, {3, 5, 47, 35}, {0, 0, 8, 6}, {0, 0, 0, 0}};
    for (uint32_t id : {0u, 3u, 7u, 0x20u}) {
        const IrCameraHandle handle = GetIrCameraHandle(id);
        for (const Rect& window : windows) {
            for (bool negative : {false, true}) {
                uint8_t session[0x400] = {};
                session[0xd0 + handle.player_number] = negative;
                memcpy(session + 0xda + 8 * handle.player_number, &window, sizeof window);
                std::vector<uint8_t> a(5 * sizeof(MomentProcessorState), 0xA5), b = a;
                sdk_default_moment(session, a.data(), &handle);
                MomentProcessorConfig config;
                GetMomentProcessorDefaultConfig(&config);
                config.camera.is_negative_image_used = negative;
                config.window_of_interest = window;
                RunMomentProcessor(handle, config);
                int count = 99;
                const nn::Result r = GetMomentProcessorStates(reinterpret_cast<MomentProcessorState*>(b.data()), &count, 5, handle);
                expect(a == b, "estado de momentos por defecto", window.width, negative);
                // Lanzado y sin muestras: el error del estado de la cámara (el portátil no la lleva; los demás, sin mando).
                expect(count == 0 && r.value == (id == 0x20 ? 0xdecdu : 0xdccdu), "GetMomentProcessorStates sin cámara", r.value);
                StopImageProcessor(handle);
                expect(GetMomentProcessorStates(reinterpret_cast<MomentProcessorState*>(b.data()), &count, 5, handle).value == 0x140cd,
                       "GetMomentProcessorStates sin lanzar");
            }
        }
    }
    {
        const IrCameraHandle handle = GetIrCameraHandle(0x20);
        uint8_t a[0x198], b[0x198];
        memset(a, 0xA5, sizeof a);
        memset(b, 0xA5, sizeof b);
        sdk.at<void (*)(const void*, void*, const IrCameraHandle*)>(0x2b5ec8)(nullptr, a, &handle);
        int count = 99;
        GetClusteringProcessorStates(reinterpret_cast<ClusteringProcessorState*>(b), &count, 5, handle);
        expect(memcmp(a, b, sizeof a) == 0 && count == 0, "estado de agrupaciones por defecto");
    }

    // La estadística de una región, con rectángulos dentro de la rejilla y cuentas a cero. Las entradas son siempre
    // las mismas (semilla fija) y sin lo de abajo fallaban una o dos de las 20000 en cuatro de cada diez pasadas, cada
    // vez otras: el x18 del sdk, que macOS borra al interrumpir. Se repite la llamada al sdk; una diferencia de verdad
    // sale las cinco veces.
    const auto sdk_statistic = sdk.at<MomentStatistic (*)(const MomentProcessorState*, const Rect&, int, int, int, int)>(0x2b6910);
    std::uniform_real_distribution<float> intensity(0.0f, 255.0f), unit(0.0f, 1.0f);
    int repeated = 0;
    for (int round = 0; round < 20000; round++) {
        const Rect window = windows[rng() % 5];
        MomentProcessorState state{};
        const bool empty = rng() % 16 == 0;
        for (int k = 0; k < kMomentRows * kMomentColumns; k++) {
            state.blocks[k].average_intensity = empty || rng() % 6 == 0 ? 0.0f : intensity(rng);
            state.blocks[k].centroid_x = unit(rng) * static_cast<float>(window.width);
            state.blocks[k].centroid_y = unit(rng) * static_cast<float>(window.height);
        }
        const int row = static_cast<int>(rng() % kMomentRows), column = static_cast<int>(rng() % kMomentColumns);
        const int rows = static_cast<int>(rng() % static_cast<unsigned>(kMomentRows - row + 1));
        const int columns = static_cast<int>(rng() % static_cast<unsigned>(kMomentColumns - column + 1));
        MomentStatistic a = sdk_statistic(&state, window, row, column, rows, columns);
        const MomentStatistic b = CalculateMomentRegionStatistic(&state, window, row, column, rows, columns);
        for (int retry = 0; retry < 4 && memcmp(&a, &b, sizeof a) != 0; retry++) {
            repeated++;
            a = sdk_statistic(&state, window, row, column, rows, columns);
        }
        expect(memcmp(&a, &b, sizeof a) == 0, "CalculateMomentRegionStatistic", rows, columns);
    }
    printf("CalculateMomentRegionStatistic: %d llamadas al sdk repetidas (x18 borrado por macOS)\n", repeated);
}

// --- nn::hid ---

void test_hid(const Sdk& sdk, std::mt19937& rng) {
    using namespace nn::hid;
    const uint32_t ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 0x10, 0x20};

    // El argumento del applet de mandos: valores por defecto y textos de todos los largos, también para los ids que
    // el sdk se salta.
    {
        uint8_t a[0x430 + 32], b[0x430 + 32];
        memset(a, 0xA5, sizeof a);
        memset(b, 0xA5, sizeof b);
        sdk.at<void (*)(void*)>(0x244d38)(a);
        reinterpret_cast<ControllerSupportArg*>(b)->SetDefault();
        expect(memcmp(a, b, sizeof a) == 0, "ControllerSupportArg::SetDefault");
        const auto sdk_text = sdk.at<void (*)(void*, const char*, const uint32_t&)>(0x244bf0);
        for (uint32_t id : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 0x10u, 0x20u, 0x21u}) {
            for (size_t length : {0u, 1u, 2u, 5u, 126u, 127u, 128u, 129u, 130u, 300u}) {
                std::string text(length, 'x');
                for (char& c : text) c = static_cast<char>('!' + rng() % 90);
                sdk_text(a, text.c_str(), id);
                SetExplainText(reinterpret_cast<ControllerSupportArg*>(b), text.c_str(), id);
                expect(memcmp(a, b, sizeof a) == 0, "SetExplainText", id, static_cast<long long>(length));
            }
        }
    }

    // Los handles de vibración (un estilo cada vez) y de sensor (con varios estilos manda el primero de la lista del
    // sdk), para todos los huecos.
    const auto sdk_vibration = sdk.at<uint64_t (*)(uint32_t*, int*, int, uint32_t, uint32_t)>(0x22a6d0);
    const auto sdk_six_axis = sdk.at<uint64_t (*)(uint32_t*, int*, int, uint32_t, uint32_t)>(0x22a4b0);
    for (uint32_t id : ids) {
        for (int count = 1; count <= 3; count++) {
            for (uint32_t style = 1; style < 64; style++) {
                uint32_t a[4], b[4];
                memset(a, 0xA5, sizeof a);
                memset(b, 0xA5, sizeof b);
                int na = -1;
                if (__builtin_popcount(style) == 1) {
                    const uint64_t r = sdk_vibration(a, &na, count, id, style);
                    const int nb = GetVibrationDeviceHandles(reinterpret_cast<VibrationDeviceHandle*>(b), count, id, {{style}});
                    expect(r == 0 && na == nb && memcmp(a, b, sizeof a) == 0, "GetVibrationDeviceHandles", id, style);
                    memset(a, 0xA5, sizeof a);
                    memset(b, 0xA5, sizeof b);
                }
                const uint64_t r = sdk_six_axis(a, &na, count, id, style);
                const int nb = GetSixAxisSensorHandles(reinterpret_cast<SixAxisSensorHandle*>(b), count, id, {{style}});
                expect(r == 0 && na == nb && memcmp(a, b, sizeof a) == 0, "GetSixAxisSensorHandles", id, style);
            }
        }
    }

    // El cuaternión de la matriz de dirección del sensor: rotaciones de verdad y matrices cualesquiera.
    {
        const auto sdk_quaternion = sdk.at<void (*)(const SixAxisSensorState*, nn::util::neon::Vector4fType*)>(0x223ae0);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        int exact = 0, total = 0;
        float worst = 0.0f;
        for (int round = 0; round < 20000; round++) {
            SixAxisSensorState state{};
            if (round % 4 == 0) {
                for (float& d : state.direction) d = unit(rng);
            } else {
                float q[4];
                float norm = 0.0f;
                for (float& c : q) c = unit(rng), norm += c * c;
                norm = std::sqrt(norm);
                if (norm < 1e-3f) continue;
                const float x = q[0] / norm, y = q[1] / norm, z = q[2] / norm, w = q[3] / norm;
                const float m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w),
                                    2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                                    2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y)};
                memcpy(state.direction, m, sizeof m);
            }
            nn::util::neon::Vector4fType a{}, b{};
            sdk_quaternion(&state, &a);
            state.GetQuaternion(&b);
            total++;
            exact += memcmp(&a, &b, sizeof a) == 0;
            for (int k = 0; k < 4; k++) worst = std::fmax(worst, std::fabs(a.v[k] - b.v[k]));
        }
        // Con matrices cualesquiera las dos cuentas se separan en el último bit (otro orden de operaciones); con la
        // única que da libnnsdk, la del reposo, tienen que dar lo mismo.
        printf("GetQuaternion: %d de %d idénticos bit a bit, mayor diferencia %g\n", exact, total, worst);
        expect(worst < 1e-5f, "SixAxisSensorState::GetQuaternion");
        SixAxisSensorState rest{};
        rest.direction[0] = rest.direction[4] = rest.direction[8] = 1.0f;
        nn::util::neon::Vector4fType a{}, b{};
        sdk_quaternion(&rest, &a);
        rest.GetQuaternion(&b);
        expect(memcmp(&a, &b, sizeof a) == 0 && b.v[3] == 1.0f, "GetQuaternion en reposo");
    }

    // Lo que guarda el servicio hid: lo que entra es lo que sale.
    SetSupportedNpadStyleSet({{7}});
    expect(GetSupportedNpadStyleSet().storage[0] == 7, "GetSupportedNpadStyleSet");
    expect(GetNpadHandheldActivationMode() == 0 && GetNpadCommunicationMode() == 3, "modos de Npad por defecto");
    SetNpadHandheldActivationMode(static_cast<NpadHandheldActivationMode>(2));
    SetNpadCommunicationMode(static_cast<NpadCommunicationMode>(1));
    expect(GetNpadHandheldActivationMode() == 2 && GetNpadCommunicationMode() == 1, "modos de Npad guardados");

    // El evento de cambio de estilo nace señalado, con el modo de borrado que pida el juego.
    {
        alignas(8) uint8_t storage[2][0x30] = {};
        auto* manual = reinterpret_cast<nn::os::SystemEventType*>(storage[0]);
        auto* automatic = reinterpret_cast<nn::os::SystemEventType*>(storage[1]);
        BindNpadStyleSetUpdateEvent(0x20, manual, nn::os::EventClearMode_ManualClear);
        BindNpadStyleSetUpdateEvent(0, automatic, nn::os::EventClearMode_AutoClear);
        expect(g_created_events.size() == 2 && g_created_events[0].event == manual && !g_created_events[0].auto_clear &&
                   g_created_events[0].signaled && g_created_events[1].event == automatic && g_created_events[1].auto_clear &&
                   g_created_events[1].signaled,
               "BindNpadStyleSetUpdateEvent");
    }

    // El historial de cada estilo: una muestra nueva cada vez; solo el del portátil viene conectado y con botones.
    {
        struct State {
            int64_t sampling;
            uint64_t buttons;
            int32_t sticks[4];
            uint32_t attributes, reserved;
        } states[16];
        const uint32_t handheld = 0x20, no1 = 0;
        g_input.buttons = 0x5;
        g_input.sticks[0] = 1234;
        expect(GetNpadStyleSet(handheld).storage[0] == 2 && GetNpadStyleSet(no1).storage[0] == 0, "GetNpadStyleSet");
        int64_t last = 0;
        auto check = [&](int n, bool connected, const char* what) {
            const State& s = states[0];
            expect(n == 1 && s.sampling > last && s.attributes == (connected ? 0x3Fu : 0u) && s.buttons == (connected ? 0x5u : 0u) &&
                       s.sticks[0] == (connected ? 1234 : 0),
                   what, n, static_cast<long long>(s.attributes));
            last = s.sampling;
        };
        memset(states, 0xA5, sizeof states);
        check(GetNpadStates(reinterpret_cast<NpadHandheldState*>(states), 16, handheld), true, "GetNpadStates portátil");
        check(GetNpadStates(reinterpret_cast<NpadFullKeyState*>(states), 16, handheld), false, "GetNpadStates Pro");
        check(GetNpadStates(reinterpret_cast<NpadJoyDualState*>(states), 16, handheld), false, "GetNpadStates dual");
        check(GetNpadStates(reinterpret_cast<NpadJoyLeftState*>(states), 16, handheld), false, "GetNpadStates izquierdo");
        check(GetNpadStates(reinterpret_cast<NpadJoyRightState*>(states), 16, no1), false, "GetNpadStates derecho");
        expect(states[1].sampling == static_cast<int64_t>(0xA5A5A5A5A5A5A5A5), "GetNpadStates: solo escribe una");
        expect(GetNpadStates(reinterpret_cast<NpadFullKeyState*>(states), 0, handheld) == 0, "GetNpadStates sin sitio");
        g_input.buttons = 0;
        g_input.sticks[0] = 0;
        uint32_t other = 0xA5A5A5A5;
        expect(!SetNpadJoyAssignmentModeSingle(&other, no1, {}) && other == no1, "SetNpadJoyAssignmentModeSingle con destino");
    }

    // Los applets vuelven al momento: un jugador con el portátil, y cancelado si el juego exige más de uno.
    {
        ControllerSupportArg arg;
        arg.SetDefault();
        ControllerSupportResultInfo info{};
        expect(ShowControllerSupport(arg).value == 0 && ShowControllerSupport(&info, arg).value == 0 && info.player_count == 1 &&
                   info.selected_id == 0x20,
               "ShowControllerSupport");
        arg.storage[0] = 2;
        expect(ShowControllerSupport(arg).value == 0x183aca, "ShowControllerSupport con dos jugadores");
        expect(ShowControllerStrapGuide().value == 0 && ShowControllerFirmwareUpdate({}).value == 0, "applets de correa y firmware");
    }

    // Los sensores: ajustes por handle y el estado de reposo.
    {
        SixAxisSensorHandle handles[2];
        const uint32_t handheld = 0x20;
        expect(GetSixAxisSensorHandles(handles, 2, handheld, {{2}}) == 1, "handle del sensor del portátil");
        const SixAxisSensorHandle h = handles[0], other{0x10005};
        float power = 0, range = 0;
        GetSixAxisSensorFusionParameters(&power, &range, h);
        expect(IsSixAxisSensorFusionEnabled(h) && power == 0.03f && range == 0.4f && GetGyroscopeZeroDriftMode(h) == 1 &&
                   IsSixAxisSensorAtRest(h) && !IsFirmwareUpdateAvailableForSixAxisSensor(h),
               "sensor: valores de fábrica");
        EnableSixAxisSensorFusion(h, false);
        SetSixAxisSensorFusionParameters(h, 0.5f, 0.25f);
        SetGyroscopeZeroDriftMode(h, static_cast<GyroscopeZeroDriftMode>(2));
        GetSixAxisSensorFusionParameters(&power, &range, h);
        expect(!IsSixAxisSensorFusionEnabled(h) && power == 0.5f && range == 0.25f && GetGyroscopeZeroDriftMode(h) == 2 &&
                   IsSixAxisSensorFusionEnabled(other) && GetGyroscopeZeroDriftMode(other) == 1,
               "sensor: ajustes guardados por handle");
        ResetSixAxisSensorFusionParameters(h);
        GetSixAxisSensorFusionParameters(&power, &range, h);
        expect(power == 0.03f && range == 0.4f && !IsSixAxisSensorFusionEnabled(h), "sensor: vuelta a los parámetros de fábrica");
        SixAxisSensorState states[16];
        memset(states, 0xA5, sizeof states);
        nn::util::neon::Vector4fType q{};
        const int n = GetSixAxisSensorStates(states, 16, h);
        states[0].GetQuaternion(&q);
        expect(n == 1 && states[0].attributes == 1 && states[0].acceleration[2] == -1.0f && q.v[0] == 0 && q.v[1] == 0 &&
                   q.v[2] == 0 && q.v[3] == 1 && states[1].attributes == 0xA5A5A5A5,
               "GetSixAxisSensorStates en reposo");
    }

    // La vibración: los dos actuadores del portátil sobre el único motor (NnsdkInput::rumble).
    {
        VibrationDeviceHandle handheld[2], no1[2];
        const uint32_t id_handheld = 0x20, id_no1 = 0;
        expect(GetVibrationDeviceHandles(handheld, 2, id_handheld, {{2}}) == 2 && GetVibrationDeviceHandles(no1, 2, id_no1, {{4}}) == 2,
               "handles de vibración");
        const VibrationValue silence{0.0f, 160.0f, 0.0f, 320.0f};
        VibrationValue actual;
        GetActualVibrationValue(&actual, handheld[0]);
        expect(IsVibrationPermitted() && memcmp(&actual, &silence, sizeof actual) == 0 && g_input.rumble == 0, "vibración: en silencio");
        // El toque del juego: 0,08 repartido entre las dos bandas.
        const VibrationValue tick{0.05f, 160.0f, 0.03f, 320.0f};
        const uint32_t serial = g_input.rumble_serial;
        SendVibrationValue(handheld[0], tick);
        GetActualVibrationValue(&actual, handheld[0]);
        expect(g_input.rumble == 72 && g_input.rumble_serial == serial + 1 && memcmp(&actual, &tick, sizeof actual) == 0,
               "vibración: el toque del juego", g_input.rumble);
        SendVibrationValue(handheld[1], {0.5f, 160.0f, 0.75f, 320.0f});
        expect(g_input.rumble == 255, "vibración: manda el actuador que más pide", g_input.rumble);
        SendVibrationValue(handheld[1], silence);
        expect(g_input.rumble == 72, "vibración: vuelve al otro actuador", g_input.rumble);
        SendVibrationValue(no1[0], {1.0f, 160.0f, 1.0f, 320.0f});
        GetActualVibrationValue(&actual, no1[0]);
        expect(g_input.rumble == 72 && memcmp(&actual, &silence, sizeof actual) == 0, "vibración: un hueco sin mando no vibra");
        SendVibrationValue(handheld[0], {std::nanf(""), 160.0f, -3.0f, 320.0f});
        expect(g_input.rumble == 0, "vibración: valores sin sentido", g_input.rumble);
        SendVibrationValue(handheld[0], {1e-6f, 160.0f, 0.0f, 320.0f});
        expect(g_input.rumble == 1, "vibración: lo mínimo sigue siendo vibrar", g_input.rumble);
        SendVibrationValue(handheld[0], silence);
        expect(g_input.rumble == 0 && g_input.rumble_serial == serial + 6, "vibración: parar", g_input.rumble);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("uso: %s work/scope/sdk.elf\n", argv[0]);
        return 2;
    }
    setenv("NIER_DATA", "/nonexistent", 1);  // los interruptores en vivo (sin_vibracion) no encuentran nada
    const Sdk sdk = load_sdk(argv[1]);
    std::mt19937 rng(20261109);
    test_vibration_files(sdk, rng);
    test_irsensor(sdk, rng);
    test_hid(sdk, rng);
    int failures = 0;
    for (const Tally& t : g_tallies) {
        printf("%-44s %7d comprobaciones, %d fallos\n", t.what, t.checks, t.failures);
        failures += t.failures;
    }
    return failures ? 1 : 0;
}
