// movie:: (la librería de vídeo de Nintendo, subsdk0) sobre AMediaCodec y AMediaExtractor. Es un envoltorio fino
// de Stagefright: el Extractor es un NuMediaExtractor (el mismo que hay detrás de AMediaExtractor), el Decoder un
// MediaCodec, con las mismas claves de formato ("mime", "width", "csd-0"…), los mismos índices de buffer y el
// mismo flag de fin de stream (4). El juego solo importa los constructores y unas pocas funciones; el resto lo
// llama por vtable, y el orden de los métodos virtuales es el del vtable que exporta subsdk0 (tools/vtable.py),
// así el compilador genera uno idéntico.
//
// MH Rise reproduce MP4 (H.264 + AAC) que RE Engine guarda dentro de sus pak: el Extractor lee el contenedor a
// través de un movie::ClientStreamReader del juego, cada pista va a su Decoder con el formato tal cual sale del
// extractor, y el juego copia los frames NV12 a sus texturas (main+0x71315a8).

#include <fcntl.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaDataSource.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/os.h"
#include "nnsdk/runtime.h"

namespace movie {

enum DecoderType : int {};
enum DecoderOutputFormat : int {};
enum DecoderMode : int {};
enum ContainerType : int {};

// movie::Status: lo que da NnMediaCommon::AndroidErrorToNnError (subsdk0+0x111fbc) a partir del status_t de
// Stagefright. El juego mira kEndOfStream (bit 13) tras Extractor::Read y GetTrackIndexForAvailableData para
// mandar el fin de stream a los decoders; del resto solo distingue cero de no cero.
using Status = int;
constexpr Status kOk = 0;
constexpr Status kError = 0x1000;        // UNKNOWN_ERROR y el resto; también "sin inicializar"
constexpr Status kUnsupported = 0x1006;  // ERROR_UNSUPPORTED
constexpr Status kIoError = 0x1009;      // ERROR_IO
constexpr Status kEndOfStream = 0x2000;  // ERROR_END_OF_STREAM
constexpr Status kTryAgain = 0x5003;     // -EWOULDBLOCK: MediaCodec sin buffer libre (INFO_TRY_AGAIN_LATER)
constexpr Status kMalformed = 0x5004;    // ERROR_MALFORMED

static Status from_media(media_status_t s) {
    switch (s) {
    case AMEDIA_OK: return kOk;
    case AMEDIA_ERROR_END_OF_STREAM: return kEndOfStream;
    case AMEDIA_ERROR_MALFORMED: return kMalformed;
    case AMEDIA_ERROR_UNSUPPORTED: return kUnsupported;
    case AMEDIA_ERROR_IO: return kIoError;
    case AMEDIA_ERROR_WOULD_BLOCK: return kTryAgain;
    default: return kError;
    }
}

// Layout de subsdk0 (64 bytes): offset/tamaño en +0, "es dueño" en +0x10, capacidad en +0x18,
// datos en +0x28. El juego reserva el objeto; Base()/Capacity()/SetRange se leen de aquí.
// Los datos sueltos (Set/GetInt32Data, Uint32, Int64) van en +0x8, +0xc y +0x20, como en subsdk0 (Xenoblade los usa).
class Buffer {
public:
    Buffer();
    explicit Buffer(size_t capacity);
    Buffer(void* data, size_t capacity);
    Buffer(const Buffer& other);
    ~Buffer();
    Buffer& operator=(const Buffer& other);
    void SetDataAndCapacity(void* data, size_t capacity);
    void SetRange(int offset, int size);
    void* Base() const;
    void* Data() const;
    size_t Capacity() const;
    int Offset() const;
    int Size() const;
    void SetInt32Data(int value);
    void SetUint32Data(uint32_t value);
    void SetInt64Data(long value);
    uint32_t GetUint32Data() const;
    long GetInt64Data() const;
    void SetNvnDev(void* device);
    void SetNvnSync(void* sync);

    void copy_from(const Buffer& other);

    int offset_ = 0, size_ = 0;
    int32_t int32_data_ = 0;
    uint32_t uint32_data_ = 0;
    bool owned_ = false;
    size_t capacity_ = 0;
    int64_t int64_data_ = 0;
    void* base_ = nullptr;
    uint64_t reserved_[2] = {};
};
static_assert(sizeof(Buffer) == 0x40 && offsetof(Buffer, int32_data_) == 0x8 && offsetof(Buffer, uint32_data_) == 0xc &&
              offsetof(Buffer, capacity_) == 0x18 && offsetof(Buffer, int64_data_) == 0x20 &&
              offsetof(Buffer, base_) == 0x28);

Buffer::Buffer() {}
// Con capacidad: el buffer es suyo (owned en +0x10) y su rango empieza lleno, como en subsdk0.
Buffer::Buffer(size_t capacity) : size_(static_cast<int>(capacity)), owned_(true), capacity_(capacity) {
    base_ = malloc(capacity ? capacity : 1);
}
Buffer::Buffer(void* data, size_t capacity) : size_(static_cast<int>(capacity)), capacity_(capacity), base_(data) {}
Buffer::Buffer(const Buffer& other) { copy_from(other); }
Buffer::~Buffer() {
    if (owned_) free(base_);
    base_ = nullptr;
}
Buffer& Buffer::operator=(const Buffer& other) {
    if (this != &other) copy_from(other);
    return *this;
}
// Si el otro es dueño de sus datos, se copian sus bytes [offset, offset + size) a memoria propia (reusando la propia si
// cabe); si no, se comparte su memoria sin ser dueño.
void Buffer::copy_from(const Buffer& other) {
    if (other.owned_) {
        const size_t n = other.size_ > 0 ? static_cast<size_t>(other.size_) : 0;
        if (!owned_ || capacity_ < n) {
            if (owned_) free(base_);
            base_ = malloc(n ? n : 1);
            capacity_ = n;
            owned_ = true;
        }
        if (n && other.base_) memcpy(base_, static_cast<const uint8_t*>(other.base_) + other.offset_, n);
        offset_ = 0;
        size_ = static_cast<int>(n);
    } else {
        if (owned_) free(base_);
        owned_ = false;
        base_ = other.base_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        size_ = other.size_;
    }
    int32_data_ = other.int32_data_;
    uint32_data_ = other.uint32_data_;
    int64_data_ = other.int64_data_;
}
void Buffer::SetDataAndCapacity(void* data, size_t capacity) {
    base_ = data;
    capacity_ = capacity;
}
void Buffer::SetRange(int offset, int size) {
    offset_ = offset;
    size_ = size;
}
void* Buffer::Base() const { return base_; }
// Donde empiezan los datos válidos: la base más el offset (Unity lee así los frames y las muestras).
void* Buffer::Data() const { return base_ ? static_cast<uint8_t*>(base_) + offset_ : nullptr; }
size_t Buffer::Capacity() const { return capacity_; }
int Buffer::Offset() const { return offset_; }
int Buffer::Size() const { return size_; }
void Buffer::SetInt32Data(int value) { int32_data_ = value; }
void Buffer::SetUint32Data(uint32_t value) { uint32_data_ = value; }
void Buffer::SetInt64Data(long value) { int64_data_ = value; }
uint32_t Buffer::GetUint32Data() const { return uint32_data_; }
long Buffer::GetInt64Data() const { return int64_data_; }
// El dispositivo NVN y el sync del frame cuando el decoder escribe en una textura (el reproductor propio de Unreal
// los pone antes de pedir el frame): en subsdk0 son dos punteros en +0x30 y +0x38 que solo se guardan. Aquí los
// frames se entregan siempre copiados a la memoria del juego, así que nadie los lee.
void Buffer::SetNvnDev(void* device) { reserved_[0] = reinterpret_cast<uint64_t>(device); }
void Buffer::SetNvnSync(void* sync) { reserved_[1] = reinterpret_cast<uint64_t>(sync); }

// En subsdk0 es un puntero a su almacén interno (un AMessage de Stagefright); el juego le reserva solo esos 8
// bytes (main+0x712de0c: la variable siguiente está en +0xc). Aquí apunta a un mapa de claves y, si viene del
// extractor, al AMediaFormat de la pista entero: el juego se lo pasa sin tocar a Decoder::Configure, y el
// decoder necesita todo lo que trae (csd-0/csd-1 con el SPS/PPS de H.264 o la config de AAC, max-input-size…).
class MediaData {
public:
    MediaData();
    MediaData(const MediaData& other);
    ~MediaData();
    MediaData& operator=(const MediaData& other);
    void Clear();
    bool FindInt32(const char* key, int* out) const;
    bool FindInt64(const char* key, long* out) const;
    bool FindString(const char* key, const char** out) const;
    bool FindRect(const char* key, int* left, int* top, int* right, int* bottom) const;
    void SetInt32(const char* key, int value);

    struct Value {
        enum Kind { kInt32, kInt64, kString, kRect } kind = kInt32;
        int64_t i = 0;
        std::string s;
        int rect[4] = {};
    };
    struct Store {
        std::map<std::string, Value> values;
        AMediaFormat* format = nullptr;  // nuestro; las claves de values mandan sobre las suyas
    };
    Store* store_;
};
static_assert(sizeof(MediaData) == 8);

MediaData::MediaData() : store_(new Store) {}
MediaData::MediaData(const MediaData& other) : store_(new Store) { *this = other; }
MediaData::~MediaData() {
    Clear();
    delete store_;
}
void MediaData::Clear() {
    store_->values.clear();
    if (store_->format) AMediaFormat_delete(store_->format);
    store_->format = nullptr;
}
// Copia del almacén (en subsdk0, la del AMessage): las claves y, si lo hay, el formato del extractor.
MediaData& MediaData::operator=(const MediaData& other) {
    if (this == &other) return *this;
    Clear();
    store_->values = other.store_->values;
    if (other.store_->format) {
        store_->format = AMediaFormat_new();
        AMediaFormat_copy(store_->format, other.store_->format);
    }
    return *this;
}
// Como AMessage::findInt32/findInt64: cada una solo encuentra su propio tipo.
bool MediaData::FindInt32(const char* key, int* out) const {
    auto it = store_->values.find(key);
    if (it != store_->values.end()) {
        if (it->second.kind != Value::kInt32) return false;
        *out = static_cast<int>(it->second.i);
        return true;
    }
    return store_->format && AMediaFormat_getInt32(store_->format, key, out);
}
bool MediaData::FindInt64(const char* key, long* out) const {
    auto it = store_->values.find(key);
    if (it != store_->values.end()) {
        if (it->second.kind != Value::kInt64) return false;
        *out = it->second.i;
        return true;
    }
    int64_t v;
    if (!store_->format || !AMediaFormat_getInt64(store_->format, key, &v)) return false;
    *out = v;
    return true;
}
// La cadena tiene que vivir lo que el MediaData (el juego guarda el puntero): la del AMediaFormat se copia al
// mapa, que no se toca hasta el próximo Clear.
bool MediaData::FindString(const char* key, const char** out) const {
    auto it = store_->values.find(key);
    if (it == store_->values.end()) {
        const char* s;
        if (!store_->format || !AMediaFormat_getString(store_->format, key, &s)) return false;
        Value& value = store_->values[key];
        value.kind = Value::kString;
        value.s = s;
        it = store_->values.find(key);
    }
    if (it->second.kind != Value::kString) return false;
    *out = it->second.s.c_str();
    return true;
}
bool MediaData::FindRect(const char* key, int* left, int* top, int* right, int* bottom) const {
    auto it = store_->values.find(key);
    if (it == store_->values.end() || it->second.kind != Value::kRect) return false;
    *left = it->second.rect[0], *top = it->second.rect[1], *right = it->second.rect[2], *bottom = it->second.rect[3];
    return true;
}

namespace {

void set_int(MediaData* d, const char* key, int32_t v) {
    MediaData::Value& value = d->store_->values[key];
    value = {};
    value.i = v;
}
void set_string(MediaData* d, const char* key, const char* v) {
    MediaData::Value& value = d->store_->values[key];
    value = {};
    value.kind = MediaData::Value::kString;
    value.s = v;
}

// Lo que el juego lee del formato de salida (main+0xa689cc): enteros, el rectángulo "crop" y las
// claves nv12-* que añade la librería de Nintendo (posición y stride de los planos Y y UV dentro del
// buffer; sin ellas el juego copia el plano UV desde un puntero basura).
void fill_from(MediaData* d, AMediaFormat* f) {
    d->Clear();
    for (const char* key : {"width", "height", "stride", "slice-height", "color-format", "sample-rate",
                            "channel-count"}) {
        int32_t v;
        if (AMediaFormat_getInt32(f, key, &v)) set_int(d, key, v);
    }
    int32_t width = 0, height = 0, stride = 0, slice = 0;
    AMediaFormat_getInt32(f, "width", &width);
    AMediaFormat_getInt32(f, "height", &height);
    if (!AMediaFormat_getInt32(f, "stride", &stride) || stride <= 0) stride = width;
    if (!AMediaFormat_getInt32(f, "slice-height", &slice) || slice <= 0) slice = height;
    set_int(d, "nv12-y-offset", 0);
    set_int(d, "nv12-y-stride", stride);
    set_int(d, "nv12-uv-offset", stride * slice);
    set_int(d, "nv12-uv-stride", stride);
    int32_t l, t, r, b;
    if (AMediaFormat_getRect(f, "crop", &l, &t, &r, &b)) {
        MediaData::Value& value = d->store_->values["crop"];
        value = {};
        value.kind = MediaData::Value::kRect;
        value.rect[0] = l, value.rect[1] = t, value.rect[2] = r, value.rect[3] = b;
    }
}

// El MediaData como AMediaFormat para MediaCodec: el formato de la pista entero y encima lo que haya en el mapa.
AMediaFormat* to_format(const MediaData* d) {
    AMediaFormat* f = AMediaFormat_new();
    if (d->store_->format) AMediaFormat_copy(f, d->store_->format);
    for (auto& [key, value] : d->store_->values) {
        if (value.kind == MediaData::Value::kString)
            AMediaFormat_setString(f, key.c_str(), value.s.c_str());
        else if (value.kind == MediaData::Value::kInt32)
            AMediaFormat_setInt32(f, key.c_str(), static_cast<int32_t>(value.i));
        else if (value.kind == MediaData::Value::kInt64)
            AMediaFormat_setInt64(f, key.c_str(), value.i);
    }
    return f;
}

}  // namespace

bool CreateVp9DecoderConfigurationData(int width, int height, int, MediaData* out) {
    if (!out) return false;
    set_string(out, "mime", "video/x-vnd.on2.vp9");
    set_int(out, "width", width);
    set_int(out, "height", height);
    set_int(out, "max-input-size", width * height);
    return true;
}

// CRI Mana sube así el "max-input-size" del formato VP9 cuando sus fotogramas no caben (main+0x5e18f80).
void MediaData::SetInt32(const char* key, int value) {
    if (key) set_int(this, key, value);
}

namespace {

// Lector de bits de un NAL de H.264 ya sin sus bytes de prevención de emulación (00 00 03).
struct BitReader {
    const std::vector<uint8_t>& data;
    size_t pos = 0;
    bool overrun = false;
    uint32_t bit() {
        if (pos >= data.size() * 8) return overrun = true, 0;
        const uint32_t b = (data[pos >> 3] >> (7 - (pos & 7))) & 1;
        pos++;
        return b;
    }
    uint32_t bits(int count) {
        uint32_t v = 0;
        while (count-- > 0) v = v << 1 | bit();
        return v;
    }
    uint32_t ue() {  // Exp-Golomb sin signo
        int zeros = 0;
        while (!overrun && bit() == 0 && zeros < 32) zeros++;
        return zeros >= 32 ? 0 : (1u << zeros) - 1 + bits(zeros);
    }
    int32_t se() {
        const uint32_t v = ue();
        return v & 1 ? static_cast<int32_t>((v + 1) / 2) : -static_cast<int32_t>(v / 2);
    }
};

// Ancho y alto visibles (con el recorte) que declara un SPS de H.264 (ITU-T H.264 7.3.2.1.1). El NAL llega sin
// código de inicio, con su byte de cabecera.
bool sps_dimensions(const uint8_t* sps, size_t size, int* width, int* height) {
    if (!sps || size < 5 || (sps[0] & 0x1F) != 7) return false;
    std::vector<uint8_t> rbsp;
    rbsp.reserve(size);
    for (size_t k = 1; k < size; k++) {
        if (k + 2 < size && sps[k] == 0 && sps[k + 1] == 0 && sps[k + 2] == 3) {
            rbsp.push_back(0), rbsp.push_back(0);
            k += 2;
        } else {
            rbsp.push_back(sps[k]);
        }
    }
    BitReader r{rbsp};
    const uint32_t profile = r.bits(8);
    r.bits(16);  // restricciones y nivel
    r.ue();      // seq_parameter_set_id
    uint32_t chroma = 1;
    bool separate_planes = false;
    switch (profile) {
    case 100: case 110: case 122: case 244: case 44: case 83: case 86: case 118: case 128: case 138: case 139:
    case 134: case 135:
        chroma = r.ue();
        if (chroma == 3) separate_planes = r.bit();
        r.ue(), r.ue();  // bit_depth_luma_minus8, bit_depth_chroma_minus8
        r.bit();         // qpprime_y_zero_transform_bypass_flag
        if (r.bit()) {   // seq_scaling_matrix_present_flag
            const int lists = chroma != 3 ? 8 : 12;
            for (int i = 0; i < lists; i++) {
                if (!r.bit()) continue;
                int last = 8, next = 8;
                for (int j = 0, n = i < 6 ? 16 : 64; j < n; j++) {
                    if (next != 0) next = (last + r.se() + 256) % 256;
                    last = next == 0 ? last : next;
                }
            }
        }
        break;
    default: break;
    }
    r.ue();  // log2_max_frame_num_minus4
    const uint32_t poc_type = r.ue();
    if (poc_type == 0) {
        r.ue();
    } else if (poc_type == 1) {
        r.bit(), r.se(), r.se();
        for (uint32_t k = 0, n = r.ue(); k < n && k < 256; k++) r.se();
    }
    r.ue();   // max_num_ref_frames
    r.bit();  // gaps_in_frame_num_value_allowed_flag
    const uint32_t width_mbs = r.ue() + 1, height_units = r.ue() + 1;
    const uint32_t frame_only = r.bit();
    if (!frame_only) r.bit();  // mb_adaptive_frame_field_flag
    r.bit();                   // direct_8x8_inference_flag
    uint32_t left = 0, right = 0, top = 0, bottom = 0;
    if (r.bit()) left = r.ue(), right = r.ue(), top = r.ue(), bottom = r.ue();
    if (r.overrun) return false;
    const uint32_t array_type = separate_planes ? 0 : chroma;
    const uint32_t unit_x = array_type == 1 || array_type == 2 ? 2 : 1;
    const uint32_t unit_y = (array_type == 1 ? 2 : 1) * (2 - frame_only);
    const int64_t w = int64_t{width_mbs} * 16 - int64_t{unit_x} * (left + right);
    const int64_t h = int64_t{2 - frame_only} * height_units * 16 - int64_t{unit_y} * (top + bottom);
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192) return false;
    *width = static_cast<int>(w), *height = static_cast<int>(h);
    return true;
}

void set_csd(AMediaFormat* f, const char* key, const uint8_t* nal, size_t size) {
    std::vector<uint8_t> csd{0, 0, 0, 1};
    csd.insert(csd.end(), nal, nal + size);
    AMediaFormat_setBuffer(f, key, csd.data(), csd.size());
}

}  // namespace

// La configuración de un decoder de H.264 a partir del SPS y el PPS sueltos (sin código de inicio), como la saca
// CRI Mana de la cabecera de sus vídeos (main+0x5e18764). La librería de la consola arma con ellos un avcC para su
// Stagefright (subsdk0+0x129210: "mime", "width", "height", "avcc", "max-input-size" = ancho·alto); MediaCodec quiere
// lo mismo como csd-0 y csd-1, cada uno con su código de inicio. Devuelve false solo si falta algún argumento.
bool CreateH264DecoderConfigurationData(const uint8_t* sps, size_t sps_size, const uint8_t* pps, size_t pps_size,
                                        MediaData* out) {
    if (!sps || !pps || !sps_size || !out) return false;
    int width = 0, height = 0;
    if (!sps_dimensions(sps, sps_size, &width, &height)) {
        printf("[movie] SPS de H.264 ilegible (%zu bytes, cabecera %#x): sin dimensiones\n", sps_size, sps[0]);
        width = 1280, height = 720;
    }
    set_string(out, "mime", "video/avc");
    set_int(out, "width", width);
    set_int(out, "height", height);
    set_int(out, "max-input-size", width * height);
    if (!out->store_->format) out->store_->format = AMediaFormat_new();
    set_csd(out->store_->format, "csd-0", sps, sps_size);
    if (pps_size) set_csd(out->store_->format, "csd-1", pps, pps_size);
    printf("[movie] H.264 %dx%d (SPS %zu bytes, PPS %zu)\n", width, height, sps_size, pps_size);
    return true;
}

// El siguiente NAL de un flujo Annex B: su principio (tras el código de inicio 00 00 01) y su tamaño, sin los ceros
// que preceden al código siguiente; *data y *size pasan a lo que queda desde ese código (nulo y 0 si era el último).
// Sin ningún código de inicio devuelve false y deja los dos últimos bytes por mirar, como subsdk0+0x128400.
bool GetH264NalUnitStartPositionAndNalSize(uint8_t** data, size_t* size, uint8_t** nal, size_t* nal_size) {
    if (!data || !size || !nal || !nal_size) return false;
    uint8_t* p = *data;
    const size_t n = *size;
    *nal = nullptr;
    *nal_size = 0;
    if (n < 3 || !p) return false;
    size_t i = 0;
    while (i + 2 < n && !(p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1)) i++;
    if (i + 2 >= n) {
        *data = p + n - 2;
        *size = 2;
        return false;
    }
    const size_t start = i + 3;
    size_t next = start;  // primer cero del código de inicio siguiente
    while (next + 2 < n && !(p[next] == 0 && p[next + 1] == 0 && p[next + 2] == 1)) next++;
    const bool more = next + 2 < n;
    size_t end = more ? next : n;
    if (more)
        while (end > start + 1 && p[end - 1] == 0) end--;
    *nal = p + start;
    *nal_size = end - start;
    if (more && next + 4 < n) {
        *data = p + next;
        *size = n - next;
    } else {
        *data = nullptr;
        *size = 0;
    }
    return true;
}

void SetAllocator(void* (*)(size_t, size_t, void*), void (*)(void*, void*), void* (*)(void*, size_t, void*), void*) {}
void SetCoreMask(uint64_t) {}
int GetThreadPriority() { return 16; }
void SetThreadPriority(int) {}

// Eventos que el juego espera con WaitAny (main+0xa68a54). El orden sale de qué rama atiende cada
// holder en su hilo de vídeo (main+0xa68944): el 1º lleva a AcquireInputBufferIndex, el 2º a
// AcquireOutputBufferIndex, el 3º a GetOutputFormat; el 4º es el de error.
struct DecoderEvents {
    nn::os::EventType* input_available;
    nn::os::EventType* output_available;
    nn::os::EventType* format_changed;
    nn::os::EventType* error;
};

class Decoder {
public:
    Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int);
    virtual ~Decoder();
    virtual Status Configure(MediaData* format);
    virtual Status RegisterClientEvents(DecoderEvents* events);
    virtual Status GetOutputFormat(MediaData* out);
    virtual Status Start();
    virtual Status Stop();
    virtual Status Reset();
    virtual Status Flush();
    virtual Status AcquireInputBufferIndex(int* index, int* remaining);
    virtual Status GetInputBuffer(int index, Buffer* out);
    virtual Status SendInputBufferForDecode(int index, int offset, int size, long time_us, unsigned flags);
    virtual Status AcquireOutputBufferIndex(int* index, long* time_us, unsigned* flags, int* remaining);
    virtual Status GetOutputBuffer(int index, Buffer* out);
    virtual Status GetOutputBufferSize(int* a, int* b);
    virtual Status ReleaseOutputBufferIndex(int index);
    virtual Status GetLastError();
    virtual Status GetOutputBuffer(int index, Buffer* out, MediaData* format);
    virtual Status GetOutputBufferFormat(int index, MediaData* format);
    // MH Rise reserva el objeto (16 bytes) con su operator new (main+0x712df2c) y lo borra por el destructor
    // virtual: la memoria vuelve a su allocator.
    static void operator delete(void* p) { nnsdk::host().game_delete(p); }

    struct Impl;  // público: lo usan los callbacks de MediaCodec

private:
    Impl* impl_;
};
static_assert(sizeof(Decoder) == 0x10);

// Cómo vienen los planos de un frame de vídeo, del último cambio de formato de salida.
struct FrameLayout {
    int32_t width = 0, height = 0, stride = 0, slice = 0, color = 0;
    int32_t crop[4] = {0, 0, -1, -1};  // izquierda, arriba, derecha, abajo (inclusivos)
};

struct Decoder::Impl {
    std::mutex m;
    AMediaCodec* codec = nullptr;
    // Estado de MediaCodec visto desde la API de Nintendo: la de NVIDIA acepta Start dos veces (tras Flush) y Start
    // tras Stop; MediaCodec no (Stop vuelve a "sin configurar", y un segundo start es INVALID_OPERATION, -38).
    enum class State { kNone, kConfigured, kRunning, kStopped } state = State::kNone;
    AMediaFormat* configured = nullptr;  // el formato con que se configuró: Start tras Stop lo vuelve a configurar
    std::string mime;
    AMediaFormat* last_output = nullptr;  // el último formato de salida dado al juego (GetOutputFormat sin codec)
    DecoderEvents events{};
    std::deque<int> inputs;
    std::deque<std::pair<int, AMediaCodecBufferInfo>> outputs;
    std::map<int, AMediaCodecBufferInfo> given;  // índices de salida entregados y no liberados
    media_status_t error = AMEDIA_OK;
    bool video = false;
    FrameLayout layout;
    bool logged_first_frame = false;
    int decoded = 0, sent = 0, released = 0;  // DEPURACIÓN

    void signal(nn::os::EventType* e) {
        if (e) nn::os::SignalEvent(e);
    }
    void clear(nn::os::EventType* e) {
        if (e) nn::os::ClearEvent(e);
    }
    // Los eventos de entrada y salida dicen si su cola tiene algo (se ponen y se quitan con m tomado). MH Rise espera
    // con WaitAny los cuatro y atiende el primero que esté señalado: WaitAny no consume el evento (como en el SDK), así
    // que si la entrada se quedaba puesta con la cola vacía, el juego pedía entradas para siempre y nunca recogía los
    // frames de vídeo; el decoder se llenaba y dejaba de aceptar entradas.
    void clear_queues() {
        std::lock_guard lock(m);
        inputs.clear();
        outputs.clear();
        given.clear();
        clear(events.input_available);
        clear(events.output_available);
    }
};

namespace {

using Impl = Decoder::Impl;

constexpr int32_t kColorYuv420Planar = 19, kColorYuv420SemiPlanar = 21;
// COLOR_FormatYUV420Flexible: lo que dicen los decoders por software (c2.android.*); en buffers de memoria es I420.
constexpr int32_t kColorYuv420Flexible = 0x7F420888;
bool is_planar(int32_t color) { return color == kColorYuv420Planar || color == kColorYuv420Flexible; }

FrameLayout layout_of(AMediaFormat* f) {
    FrameLayout l;
    AMediaFormat_getInt32(f, "width", &l.width);
    AMediaFormat_getInt32(f, "height", &l.height);
    if (!AMediaFormat_getInt32(f, "stride", &l.stride) || l.stride <= 0) l.stride = l.width;
    if (!AMediaFormat_getInt32(f, "slice-height", &l.slice) || l.slice <= 0) l.slice = l.height;
    AMediaFormat_getInt32(f, "color-format", &l.color);
    if (!AMediaFormat_getRect(f, "crop", &l.crop[0], &l.crop[1], &l.crop[2], &l.crop[3]))
        l.crop[0] = 0, l.crop[1] = 0, l.crop[2] = l.width - 1, l.crop[3] = l.height - 1;
    return l;
}

// MH Rise copia el frame a dos texturas del tamaño del vídeo (main+0x71314d4): Y de ancho·alto bytes y justo
// detrás UV entrelazado de ancho·alto/2, sin stride ni relleno (NVDEC lo entrega así). MediaCodec da los planos
// con el stride y el alto alineados del decoder (Qualcomm: 128 y 32) o, en un decoder por software, en I420: se
// recolocan en NV12 compacto del recorte visible. Devuelve los bytes escritos.
size_t pack_nv12(const FrameLayout& l, const uint8_t* src, size_t src_size, uint8_t* dst, size_t capacity) {
    int w = (l.crop[2] - l.crop[0] + 1) & ~1, h = (l.crop[3] - l.crop[1] + 1) & ~1;
    if (w <= 0 || h <= 0 || l.stride < w) return 0;
    h = std::min<int64_t>(h, capacity * 2 / 3 / w) & ~1;  // si el juego reservó menos, las filas que quepan
    size_t chroma = size_t(l.stride) * l.slice;           // inicio del plano UV (o U en I420)
    bool planar = is_planar(l.color);
    size_t c_stride = planar ? l.stride / 2 : l.stride, c_rows = (l.slice + 1) / 2;
    if (chroma + c_stride * c_rows * (planar ? 2 : 1) > src_size) return 0;
    const uint8_t* y = src + size_t(l.crop[1]) * l.stride + l.crop[0];
    for (int r = 0; r < h; r++) memcpy(dst + size_t(r) * w, y + size_t(r) * l.stride, w);
    uint8_t* out = dst + size_t(w) * h;
    if (planar) {
        const uint8_t* u = src + chroma + size_t(l.crop[1] / 2) * c_stride + l.crop[0] / 2;
        const uint8_t* v = u + c_stride * c_rows;
        for (int r = 0; r < h / 2; r++, out += w)
            for (int c = 0; c < w / 2; c++) {
                out[2 * c] = u[r * c_stride + c];
                out[2 * c + 1] = v[r * c_stride + c];
            }
    } else {
        const uint8_t* uv = src + chroma + size_t(l.crop[1] / 2) * c_stride + (l.crop[0] & ~1);
        for (int r = 0; r < h / 2; r++, out += w) memcpy(out, uv + size_t(r) * c_stride, w);
    }
    return out - dst;
}

void on_input(AMediaCodec*, void* user, int32_t index) {
    auto* d = static_cast<Impl*>(user);
    {
        std::lock_guard lock(d->m);
        d->inputs.push_back(index);
        static int n = 0;
        if (++n <= 40) printf("[movie] on_input %d (índice %d, en cola %zu)\n", n, index, d->inputs.size());
        d->signal(d->events.input_available);
    }
}

void on_output(AMediaCodec*, void* user, int32_t index, AMediaCodecBufferInfo* info) {
    auto* d = static_cast<Impl*>(user);
    {
        std::lock_guard lock(d->m);
        d->outputs.emplace_back(index, *info);
        static int n = 0;
        if (++n <= 40 || info->flags)
            printf("[movie] on_output %d (índice %d, t=%lld, flags %u, %d bytes)\n", n, index,
                   static_cast<long long>(info->presentationTimeUs), info->flags, info->size);
        if (++d->decoded % 60 == 0)
            printf("[movie] %d frames decodificados (t=%lld us), %d entradas enviadas, %d liberados\n", d->decoded,
                   static_cast<long long>(info->presentationTimeUs), d->sent, d->released);
        if (!d->logged_first_frame) {
            d->logged_first_frame = true;
            printf("[movie] primer frame decodificado (%d bytes, t=%lld us)\n", info->size,
                   static_cast<long long>(info->presentationTimeUs));
        }
        d->signal(d->events.output_available);
    }
}

void on_format(AMediaCodec*, void* user, AMediaFormat* format) {
    auto* d = static_cast<Impl*>(user);
    if (d->video && format) {
        FrameLayout l = layout_of(format);
        printf("[movie] salida %dx%d, stride %d, corte %d, color %d, recorte (%d,%d)-(%d,%d)\n", l.width, l.height,
               l.stride, l.slice, l.color, l.crop[0], l.crop[1], l.crop[2], l.crop[3]);
        std::lock_guard lock(d->m);
        d->layout = l;
    }
    d->signal(d->events.format_changed);
}

void on_error(AMediaCodec*, void* user, media_status_t error, int32_t, const char* detail) {
    auto* d = static_cast<Impl*>(user);
    printf("[movie] error de MediaCodec %d: %s\n", error, detail ? detail : "");
    d->error = error;
    d->signal(d->events.error);
}

}  // namespace

static void trace(const char* what, int n) {  // DEPURACIÓN
    if (n <= 6 || n % 100 == 0) printf("[movie] %s #%d\n", what, n);
}

Decoder::Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int) : impl_(new Impl) {}

Decoder::~Decoder() {
    if (impl_->codec) {
        AMediaCodec_stop(impl_->codec);
        AMediaCodec_delete(impl_->codec);
    }
    if (impl_->configured) AMediaFormat_delete(impl_->configured);
    if (impl_->last_output) AMediaFormat_delete(impl_->last_output);
    delete impl_;
}

Status Decoder::RegisterClientEvents(DecoderEvents* events) {
    impl_->events = *events;
    return kOk;
}

// Vídeo VP9/VP8 por software (libvpx de Android, c2.android.*). El de Qualcomm (OMX.qcom.video.decoder.vp9), al dar
// los frames en buffers de memoria, convierte cada uno de su formato en mosaico a NV12 lineal con el blitter C2D de la
// GPU: 40 ms por frame con la GPU del juego al 99 % (Xenoblade 3, la película de la intro), 25 fps para un vídeo de 30.
// Todos los frames llegaban tarde, el juego los tiraba sin pedirlos (2300 sacados, 57 copiados) y la pantalla se quedaba
// en el primero (negro) mientras el audio y los subtítulos seguían. 720p VP9 por CPU son pocos ms en los A77.
// <datos>/video_hw: el decoder de hardware de antes.
AMediaCodec* create_decoder(const std::string& type) {
    if (!NVN_FLAG("video_hw")) {
        const char* sw = type == "video/x-vnd.on2.vp9"   ? "c2.android.vp9.decoder"
                         : type == "video/x-vnd.on2.vp8" ? "c2.android.vp8.decoder"
                                                         : nullptr;
        if (sw)
            if (AMediaCodec* c = AMediaCodec_createCodecByName(sw)) {
                printf("[movie] %s por software (%s)\n", type.c_str(), sw);
                return c;
            }
    }
    return AMediaCodec_createDecoderByType(type.c_str());
}

Status Decoder::Configure(MediaData* format) {
    const char* mime = nullptr;
    if (!format->FindString("mime", &mime)) return kError;
    std::string type = mime;
    // Reconfigurar (el juego lo repite si Start falla, y al reutilizar el decoder tras Stop) no puede dejar el
    // codec anterior vivo: es una sesión de vídeo por hardware con sus buffers, y sus callbacks seguían
    // metiendo índices viejos en las colas.
    if (impl_->codec) {
        AMediaCodec_stop(impl_->codec);
        AMediaCodec_delete(impl_->codec);
        impl_->codec = nullptr;
        impl_->clear_queues();
    }
    static int failures = 0;
    AMediaCodec* codec = create_decoder(type);
    if (!codec) {
        if (failures++ < 8) printf("[movie] no hay decoder para %s\n", type.c_str());
        return kError;
    }
    impl_->video = type.compare(0, 6, "video/") == 0;
    AMediaFormat* f = to_format(format);
    // Vídeo en NV12 (lo que espera MH Rise): un decoder por software daría I420. Si el decoder no acepta la
    // petición, se configura sin ella y pack_nv12 convierte lo que venga.
    int32_t color;
    bool asked_color = impl_->video && !AMediaFormat_getInt32(f, "color-format", &color);
    if (asked_color) AMediaFormat_setInt32(f, "color-format", kColorYuv420SemiPlanar);
    media_status_t r = AMediaCodec_configure(codec, f, nullptr, nullptr, 0);
    if (r != AMEDIA_OK && asked_color) {
        AMediaCodec_delete(codec);
        codec = create_decoder(type);
        AMediaFormat_delete(f);
        f = to_format(format);
        r = codec ? AMediaCodec_configure(codec, f, nullptr, nullptr, 0) : AMEDIA_ERROR_UNKNOWN;
    }
    if (r != AMEDIA_OK) {
        AMediaFormat_delete(f);
        if (failures++ < 8) printf("[movie] AMediaCodec_configure: %d\n", r);
        if (codec) AMediaCodec_delete(codec);
        return from_media(r);
    }
    AMediaCodecOnAsyncNotifyCallback callbacks{on_input, on_output, on_format, on_error};
    AMediaCodec_setAsyncNotifyCallback(codec, callbacks, impl_);
    impl_->codec = codec;
    impl_->state = Impl::State::kConfigured;
    if (impl_->configured) AMediaFormat_delete(impl_->configured);
    impl_->configured = f;  // el mismo que se pasó a configure (con la petición de NV12): Flush y Start tras Stop
    impl_->mime = type;
    char* name = nullptr;
    AMediaCodec_getName(codec, &name);
    printf("[movie] decoder %s listo (%s)\n", type.c_str(), name ? name : "?");
    if (name) AMediaCodec_releaseName(codec, name);
    return kOk;
}

Status Decoder::GetOutputFormat(MediaData* out) {
    static int calls = 0;
    trace("GetOutputFormat", ++calls);
    // Sin codec (entre Flush y Start): el último formato de salida que se dio.
    if (!impl_->codec) {
        if (!impl_->last_output) return kError;
        fill_from(out, impl_->last_output);
        return kOk;
    }
    AMediaFormat* f = AMediaCodec_getOutputFormat(impl_->codec);
    fill_from(out, f);
    if (impl_->last_output) AMediaFormat_delete(impl_->last_output);
    impl_->last_output = f;
    return kOk;
}

// Xenoblade 3 encadena las películas de la intro con el mismo decoder: Flush (que aquí ya arranca) y Start. Con el
// decoder por software el segundo start daba -38 y la película siguiente no empezaba (se quedaba la última imagen).
Status Decoder::Start() {
    printf("[movie] Start (estado %d)\n", static_cast<int>(impl_->state));
    if (impl_->state == Impl::State::kRunning && impl_->codec) return kOk;
    // Tras Flush (o Stop) el codec no existe o está parado: uno nuevo con el formato de Configure (stop + configure del
    // mismo no vuelve a avisar con el decoder por software).
    if (!impl_->codec || impl_->state == Impl::State::kStopped) {
        if (!impl_->configured) return kError;
        if (impl_->codec) {
            AMediaCodec_delete(impl_->codec);
            impl_->codec = nullptr;
        }
        impl_->clear_queues();
        AMediaCodec* codec = create_decoder(impl_->mime);
        media_status_t c = codec ? AMediaCodec_configure(codec, impl_->configured, nullptr, nullptr, 0) : AMEDIA_ERROR_UNKNOWN;
        if (c != AMEDIA_OK) {
            printf("[movie] Start: no se pudo reconfigurar (%d)\n", c);
            if (codec) AMediaCodec_delete(codec);
            return from_media(c);
        }
        AMediaCodecOnAsyncNotifyCallback callbacks{on_input, on_output, on_format, on_error};
        AMediaCodec_setAsyncNotifyCallback(codec, callbacks, impl_);
        impl_->codec = codec;
    }
    media_status_t r = AMediaCodec_start(impl_->codec);
    if (r == AMEDIA_OK) impl_->state = Impl::State::kRunning;
    else printf("[movie] AMediaCodec_start: %d\n", r);
    return from_media(r);
}

Status Decoder::Stop() {
    printf("[movie] Stop (estado %d)\n", static_cast<int>(impl_->state));
    if (impl_->codec) AMediaCodec_stop(impl_->codec);
    impl_->clear_queues();
    if (impl_->state != Impl::State::kNone) impl_->state = Impl::State::kStopped;
    return kOk;
}

Status Decoder::Reset() { return Stop(); }

// Flush de NVIDIA: vacía y deja el decoder parado hasta el siguiente Start. Xenoblade 3 encadena así las películas de la
// intro (Flush, abre la siguiente, Start); entre medias su hilo de entrada, aún en el final de la película anterior,
// sigue mandando marcas de fin. Con flush + start de MediaCodec el decoder las aceptaba y devolvía el fin de stream
// en el codec ya arrancado: la película siguiente "terminaba" sin empezar y el juego se quedaba esperando (se congelaba
// la última imagen). Aquí el codec se suelta y Start crea uno nuevo: sin buffers de entrada entre medias, como en la
// consola. <datos>/flush_mediacodec: el flush + start de MediaCodec de antes.
Status Decoder::Flush() {
    printf("[movie] Flush (estado %d)\n", static_cast<int>(impl_->state));
    if (!impl_->codec) return kError;
    if (NVN_FLAG("flush_mediacodec") || !impl_->configured) {
        AMediaCodec_flush(impl_->codec);
        impl_->clear_queues();
        media_status_t r = AMediaCodec_start(impl_->codec);
        if (r != AMEDIA_OK) printf("[movie] start tras flush: %d\n", r);
        return from_media(r);
    }
    AMediaCodec_stop(impl_->codec);
    AMediaCodec_delete(impl_->codec);
    impl_->codec = nullptr;
    impl_->clear_queues();
    impl_->state = Impl::State::kStopped;
    return kOk;
}

Status Decoder::AcquireInputBufferIndex(int* index, int* remaining) {
    std::lock_guard lock(impl_->m);
    static int calls = 0, empty = 0;
    trace(impl_->inputs.empty() ? "AcquireInput vacío" : "AcquireInput", impl_->inputs.empty() ? ++empty : ++calls);
    if (impl_->inputs.empty()) {
        impl_->clear(impl_->events.input_available);
        return kTryAgain;
    }
    *index = impl_->inputs.front();
    impl_->inputs.pop_front();
    *remaining = static_cast<int>(impl_->inputs.size());
    if (impl_->inputs.empty()) impl_->clear(impl_->events.input_available);
    return kOk;
}

Status Decoder::GetInputBuffer(int index, Buffer* out) {
    size_t capacity = 0;
    uint8_t* data = impl_->codec ? AMediaCodec_getInputBuffer(impl_->codec, index, &capacity) : nullptr;
    if (!data) return kError;
    out->SetDataAndCapacity(data, capacity);
    out->SetRange(0, 0);
    return kOk;
}

Status Decoder::SendInputBufferForDecode(int index, int offset, int size, long time_us, unsigned flags) {
    if (!impl_->codec) return kError;
    if (impl_->sent++ < 40 || (flags & 4))
        printf("[movie] entrada %d: %d bytes t=%ld flags 0x%x\n", impl_->sent, size, time_us, flags);
    const media_status_t r =
        AMediaCodec_queueInputBuffer(impl_->codec, index, offset, size, time_us < 0 ? 0 : time_us, flags);
    if (r != AMEDIA_OK) printf("[movie] queueInputBuffer %d (flags 0x%x): %d\n", index, flags, r);
    return from_media(r);
}

Status Decoder::AcquireOutputBufferIndex(int* index, long* time_us, unsigned* flags, int* remaining) {
    std::lock_guard lock(impl_->m);
    static int calls = 0, empty = 0;
    trace(impl_->outputs.empty() ? "AcquireOutput vacío" : "AcquireOutput", impl_->outputs.empty() ? ++empty : ++calls);
    if (impl_->outputs.empty()) {
        impl_->clear(impl_->events.output_available);
        return kTryAgain;
    }
    auto [i, info] = impl_->outputs.front();
    impl_->outputs.pop_front();
    if (impl_->outputs.empty()) impl_->clear(impl_->events.output_available);
    impl_->given[i] = info;
    if (info.flags) printf("[movie] AcquireOutput entrega índice %d con flags %u (%d bytes)\n", i, info.flags, info.size);
    *index = i;
    *time_us = info.presentationTimeUs;
    *flags = info.flags;
    *remaining = static_cast<int>(impl_->outputs.size());
    return kOk;
}

// Como en la API de Nintendo, el frame se copia en la memoria del juego, que lo pasa ya preparado con
// SetDataAndCapacity. NieR pregunta antes el tamaño con GetOutputBufferSize (main+0xa676a0) y lee los planos
// con los offsets nv12-* del formato de salida: se le copia tal cual. MH Rise reserva justo el NV12 compacto
// del vídeo (menos que el frame con relleno) y lo lee sin preguntar: se recoloca.
Status Decoder::GetOutputBuffer(int index, Buffer* out) {
    size_t capacity = 0;
    uint8_t* data = impl_->codec ? AMediaCodec_getOutputBuffer(impl_->codec, index, &capacity) : nullptr;
    if (!data) return kError;
    AMediaCodecBufferInfo info;
    FrameLayout layout;
    {
        std::lock_guard lock(impl_->m);
        info = impl_->given[index];
        layout = impl_->layout;
    }
    if (!out->Base() || !out->Capacity()) {  // sin memoria del juego: se le da la del decoder
        out->SetDataAndCapacity(data + info.offset, info.size);
        out->SetRange(0, info.size);
        return kOk;
    }
    if (impl_->video && info.size > 0 && layout.width > 0 &&
        (out->Capacity() < static_cast<size_t>(info.size) || is_planar(layout.color))) {
        size_t n = pack_nv12(layout, data + info.offset, info.size, static_cast<uint8_t*>(out->Base()),
                             out->Capacity());
        static int logged = 0;
        if (nvn::first_times(logged, 2) || NVN_FLAG("diag_video")) {
            // DEPURACIÓN: brillo medio (Y) del frame del decoder y del que queda en la memoria del juego.
            const uint8_t* src = data + info.offset;
            const auto* dst = static_cast<const uint8_t*>(out->Base());
            uint64_t ys = 0, yd = 0, k = 0;
            for (size_t i = 0; i < size_t(layout.stride) * layout.height && i < size_t(info.size); i += 4093, k++)
                ys += src[i], yd += dst[std::min(i, n - 1)];
            printf("[movie] frame recolocado en NV12 compacto: %zu de %zu bytes (el juego da %zu en %p) t=%lld, Y medio "
                   "decoder %llu juego %llu\n", n, static_cast<size_t>(info.size), out->Capacity(), out->Base(),
                   static_cast<long long>(info.presentationTimeUs), static_cast<unsigned long long>(k ? ys / k : 0),
                   static_cast<unsigned long long>(k ? yd / k : 0));
        }
        out->SetRange(0, static_cast<int>(n));
        return n ? kOk : kError;
    }
    size_t n = std::min(out->Capacity(), static_cast<size_t>(info.size));
    memcpy(out->Base(), data + info.offset, n);
    out->SetRange(0, static_cast<int>(n));
    return kOk;
}

// Tamaño del buffer que el juego debe reservar para un frame: NV12 con el stride y el alto de corte
// del decoder, redondeado a 4 KB como lo entrega el de Qualcomm (1600x904 -> 2170880). En audio no hay
// dimensiones: manda el tamaño de los buffers de PCM que ya ha dado el decoder.
Status Decoder::GetOutputBufferSize(int* size, int* count) {
    int32_t width = 0, height = 0, stride = 0, slice = 0;
    if (impl_->codec) {
        AMediaFormat* f = AMediaCodec_getOutputFormat(impl_->codec);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_WIDTH, &width);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_HEIGHT, &height);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_STRIDE, &stride);
        AMediaFormat_getInt32(f, "slice-height", &slice);
        AMediaFormat_delete(f);
    }
    size_t bytes = size_t(std::max(stride, width)) * std::max(slice, height) * 3 / 2;
    {
        std::lock_guard lock(impl_->m);
        for (const auto& [i, info] : impl_->given) bytes = std::max(bytes, static_cast<size_t>(info.size));
        for (const auto& [i, info] : impl_->outputs) bytes = std::max(bytes, static_cast<size_t>(info.size));
    }
    *size = static_cast<int>((bytes + 4095) & ~size_t{4095});
    *count = 1;
    static int logged = 0;
    if (nvn::first_times(logged, 8))
        printf("[movie] tamaño de frame %d bytes (%dx%d, stride %d, corte %d)\n", *size, width, height, stride, slice);
    return kOk;
}

Status Decoder::ReleaseOutputBufferIndex(int index) {
    static int calls = 0;
    trace("ReleaseOutput", ++calls);
    {
        std::lock_guard lock(impl_->m);
        impl_->given.erase(index);
        impl_->released++;
    }
    return impl_->codec ? from_media(AMediaCodec_releaseOutputBuffer(impl_->codec, index, false)) : kError;
}

Status Decoder::GetLastError() { return impl_->error == AMEDIA_OK ? kOk : from_media(impl_->error); }

Status Decoder::GetOutputBuffer(int index, Buffer* out, MediaData* format) {
    Status s = GetOutputBuffer(index, out);
    return s == kOk ? GetOutputFormat(format) : s;
}

Status Decoder::GetOutputBufferFormat(int, MediaData* format) { return GetOutputFormat(format); }

// El origen de datos que el juego implementa para el Extractor. Los slots salen del vtable del lector de MH
// Rise (main+0x1263c078) y de cómo los llama android::FileSource de subsdk0 (+0x2c7c0c): Open guarda el handle
// si devuelve true, ReadAt devuelve los bytes leídos (0 si el handle no vale), Close al destruir la fuente.
// RE Engine sirve los bytes del vídeo desde su pak (en memoria o por su sistema de archivos) y el nombre que
// pasa ("dummy.mp4") no significa nada.
class ClientStreamReader {
public:
    virtual ~ClientStreamReader();
    virtual bool Open(const char* path, void** handle) = 0;
    virtual long ReadAt(void* handle, long offset, void* buffer, size_t size) = 0;
    virtual bool Close(void* handle) = 0;
    virtual void GetSize(void* handle, long* size) = 0;
};

class Extractor {
public:
    explicit Extractor(ContainerType type);
    virtual ~Extractor();
    virtual Status SetDataSource(const char* path);
    virtual Status SetDataSource(const void* data, int size);
    virtual Status GetTrackCount(int* count);
    virtual Status GetTrackConfiguration(int index, MediaData* out);
    virtual Status SelectTrack(int index);
    virtual Status UnselectTrack(int index);
    virtual Status SeekTo(long time_us);
    virtual Status GetTrackIndexForAvailableData(size_t* index);
    virtual Status Read(Buffer* out);
    virtual Status Advance();
    virtual Status GetSampleTime(long* time_us);
    virtual Status GetSampleMetaData(MediaData* out);
    virtual Status GetBufferedRange(long* start_us, long* end_us);
    virtual Status SetDataSource(ClientStreamReader* reader, const char* path);
    // El juego reserva el objeto (16 bytes, como el de subsdk0: vtable + implementación) con su operator new
    // (main+0x7130c14) y lo borra por el destructor virtual.
    static void operator delete(void* p) { nnsdk::host().game_delete(p); }

    struct Impl;

private:
    Impl* impl_;
};
static_assert(sizeof(Extractor) == 0x10);

namespace {

// Lo que lee el AMediaExtractor: el lector del juego o un bloque de memoria. AMediaExtractor parsea en el
// proceso media.extractor y pide los bytes por binder; el driver entrega esas llamadas al hilo que está
// esperando la respuesta (el hilo de vídeo del juego, que es quien llama al extractor), así que el lector del
// juego corre en su hilo como en la Switch. El NDK no lo promete: si llegan por otro hilo, el mutex las ordena.
// Vive hasta que lo sueltan los dos dueños (el Extractor y el primer close del framework): una lectura tardía
// nunca toca memoria liberada, y sin lector (el vídeo ya se cerró) cualquier lectura es fin de stream.
struct Source {
    std::mutex m;
    ClientStreamReader* reader = nullptr;
    void* handle = nullptr;
    const uint8_t* memory = nullptr;
    int64_t size = -1;
    std::atomic<int> refs{2};
    std::atomic<bool> closed{false};

    void release() {
        if (--refs == 0) delete this;
    }
};

ssize_t source_read(void* user, off64_t offset, void* buffer, size_t size) {
    auto* s = static_cast<Source*>(user);
    if (size == 0) return 0;
    std::lock_guard lock(s->m);
    if (offset < 0 || (s->size >= 0 && offset >= s->size)) return -1;
    if (s->size >= 0) size = std::min<int64_t>(size, s->size - offset);
    long n = 0;
    if (s->memory) {
        memcpy(buffer, s->memory + offset, size);
        n = static_cast<long>(size);
    } else if (s->reader) {
        n = s->reader->ReadAt(s->handle, offset, buffer, size);
    }
    return n > 0 ? n : -1;
}

ssize_t source_size(void* user) {
    auto* s = static_cast<Source*>(user);
    std::lock_guard lock(s->m);
    return s->size;
}

void source_close(void* user) {
    auto* s = static_cast<Source*>(user);
    if (!s->closed.exchange(true)) s->release();
}

}  // namespace

struct Extractor::Impl {
    AMediaExtractor* ex = nullptr;
    AMediaDataSource* data_source = nullptr;
    Source* source = nullptr;
    int fd = -1;  // el archivo de la RomFS, si se abrió por descriptor

    // Una fuente nueva sustituye a la anterior (SetDataSource se puede repetir): el extractor se rehace.
    void reset() {
        if (ex) AMediaExtractor_delete(ex);
        ex = nullptr;
        if (fd >= 0) close(fd);
        fd = -1;
        if (data_source) AMediaDataSource_delete(data_source);
        data_source = nullptr;
        if (source) {
            std::unique_lock lock(source->m);
            if (source->reader && source->handle) source->reader->Close(source->handle);
            source->reader = nullptr;
            source->memory = nullptr;
            source->size = -1;
            lock.unlock();
            source->release();
        }
        source = nullptr;
    }

    Status open(Source* s) {
        reset();
        source = s;
        data_source = AMediaDataSource_new();
        AMediaDataSource_setUserdata(data_source, s);
        AMediaDataSource_setReadAt(data_source, source_read);
        AMediaDataSource_setGetSize(data_source, source_size);
        AMediaDataSource_setClose(data_source, source_close);
        ex = AMediaExtractor_new();
        return report(AMediaExtractor_setDataSourceCustom(ex, data_source));
    }

    Status report(media_status_t r) {
        if (r != AMEDIA_OK) {
            printf("[movie] el extractor no abre el vídeo: %d\n", r);
            return from_media(r);
        }
        size_t tracks = AMediaExtractor_getTrackCount(ex);
        printf("[movie] extractor: %zu pistas\n", tracks);
        for (size_t i = 0; i < tracks; i++) {
            AMediaFormat* f = AMediaExtractor_getTrackFormat(ex, i);
            printf("[movie]   pista %zu: %s\n", i, AMediaFormat_toString(f));
            AMediaFormat_delete(f);
        }
        return tracks ? kOk : kUnsupported;
    }
};

// El tipo de contenedor (1-4) no hace falta: AMediaExtractor lo reconoce por el contenido.
Extractor::Extractor(ContainerType) : impl_(new Impl) {}

Extractor::~Extractor() {
    impl_->reset();
    delete impl_;
}

Status Extractor::SetDataSource(const char* path) {
    impl_->reset();
    impl_->ex = AMediaExtractor_new();
    printf("[movie] extractor sobre %s\n", path);
    return impl_->report(AMediaExtractor_setDataSource(impl_->ex, path));
}

Status Extractor::SetDataSource(const void* data, int size) {
    if (!data || size <= 0) return kError;
    auto* s = new Source;
    s->memory = static_cast<const uint8_t*>(data);
    s->size = size;
    printf("[movie] extractor sobre %d bytes en memoria\n", size);
    return impl_->open(s);
}

Status Extractor::SetDataSource(ClientStreamReader* reader, const char* path) {
    if (!reader) return kError;
    impl_->reset();  // el handle anterior se cierra antes de abrir el nuevo
    // El extractor que Android elige para los webm de Xenoblade (VP9 + Vorbis) es el de Qualcomm (MMParserExtractor), que
    // no lee de una fuente por callbacks ("FileSourceWrapper::New Init returned -1004", 0 pistas): necesita un archivo.
    // Si el vídeo es un archivo de la RomFS en la SD, va su descriptor (los MP4 de MH Rise venían de dentro de sus pak y
    // no lo son); si no, el lector del juego como antes.
    // Unity (Disco Elysium) guarda los VideoClip dentro de los .resource y su lector entrega solo ese trozo: se busca
    // dónde empieza dentro del archivo (sus primeros bytes) y se le da al extractor el descriptor con ese desplazamiento.
    void* handle = nullptr;
    if (!reader->Open(path, &handle) || !handle) {
        printf("[movie] el lector del juego no abre %s\n", path ? path : "(sin nombre)");
        return kIoError;
    }
    long size = -1;
    reader->GetSize(handle, &size);
    if (const std::string host = path ? nnsdk::fs_host_path(path) : std::string{}; !host.empty()) {
        const int fd = open(host.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat st {};
        if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
            int64_t offset = 0, length = st.st_size;
            if (size > 0 && size < st.st_size) {
                offset = -1;
                uint8_t head[256];
                const long want = std::min<long>(size, sizeof head);
                if (reader->ReadAt(handle, 0, head, want) == want) {
                    void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
                    if (map != MAP_FAILED) {
                        const auto* bytes = static_cast<const uint8_t*>(map);
                        if (const void* hit = memmem(bytes, st.st_size, head, want))
                            offset = static_cast<const uint8_t*>(hit) - bytes;
                        munmap(map, st.st_size);
                    }
                }
                length = size;
            }
            if (offset >= 0 && offset + length <= st.st_size) {
                impl_->fd = fd;
                impl_->ex = AMediaExtractor_new();
                printf("[movie] extractor sobre %s (descriptor, %lld bytes desde %lld)\n", host.c_str(),
                       static_cast<long long>(length), static_cast<long long>(offset));
                if (impl_->report(AMediaExtractor_setDataSourceFd(impl_->ex, fd, offset, length)) == kOk) {
                    reader->Close(handle);
                    return kOk;
                }
                impl_->reset();
            } else {
                close(fd);
            }
        } else if (fd >= 0) {
            close(fd);
        }
    }
    auto* s = new Source;
    s->reader = reader;
    s->handle = handle;
    s->size = size;
    printf("[movie] extractor sobre el lector del juego (%s, %ld bytes)\n", path ? path : "", size);
    return impl_->open(s);
}

Status Extractor::GetTrackCount(int* count) {
    if (!impl_->ex) return kError;
    *count = static_cast<int>(AMediaExtractor_getTrackCount(impl_->ex));
    return kOk;
}

Status Extractor::GetTrackConfiguration(int index, MediaData* out) {
    if (!impl_->ex) return kError;
    AMediaFormat* f = AMediaExtractor_getTrackFormat(impl_->ex, index);
    if (!f) return kError;
    out->Clear();
    out->store_->format = f;
    return kOk;
}

Status Extractor::SelectTrack(int index) {
    return impl_->ex ? from_media(AMediaExtractor_selectTrack(impl_->ex, index)) : kError;
}

Status Extractor::UnselectTrack(int index) {
    return impl_->ex ? from_media(AMediaExtractor_unselectTrack(impl_->ex, index)) : kError;
}

// Como NnExtractor::SeekTo (subsdk0+0x1028e8): hacia delante al siguiente fotograma clave, hacia atrás al
// anterior.
Status Extractor::SeekTo(long time_us) {
    if (!impl_->ex) return kError;
    SeekMode mode = AMediaExtractor_getSampleTime(impl_->ex) < time_us ? AMEDIAEXTRACTOR_SEEK_NEXT_SYNC
                                                                        : AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC;
    return from_media(AMediaExtractor_seekTo(impl_->ex, time_us, mode));
}

// La pista de la próxima muestra; al acabarse todas, fin de stream (el juego lo manda entonces a los
// decoders). AMediaExtractor da -1 tanto al final como si falla la lectura: las dos cierran el vídeo.
Status Extractor::GetTrackIndexForAvailableData(size_t* index) {
    if (!impl_->ex) return kError;
    int track = AMediaExtractor_getSampleTrackIndex(impl_->ex);
    if (track < 0) return kEndOfStream;
    *index = track;
    return kOk;
}

// La muestra actual al principio del buffer (NuMediaExtractor::readSampleData deja el rango en [0, tamaño]).
Status Extractor::Read(Buffer* out) {
    if (!impl_->ex) return kError;
    if (AMediaExtractor_getSampleTrackIndex(impl_->ex) < 0) return kEndOfStream;
    ssize_t need = AMediaExtractor_getSampleSize(impl_->ex);
    if (!out->Base() || need > static_cast<ssize_t>(out->Capacity())) {
        printf("[movie] la muestra no cabe: %zd bytes en %zu\n", need, out->Capacity());
        return kError;
    }
    ssize_t n = AMediaExtractor_readSampleData(impl_->ex, static_cast<uint8_t*>(out->Base()), out->Capacity());
    if (n < 0) return kEndOfStream;
    out->SetRange(0, static_cast<int>(n));
    return kOk;
}

Status Extractor::Advance() {
    if (!impl_->ex) return kError;
    return AMediaExtractor_advance(impl_->ex) ? kOk : kEndOfStream;
}

Status Extractor::GetSampleTime(long* time_us) {
    if (!impl_->ex) return kError;
    int64_t t = AMediaExtractor_getSampleTime(impl_->ex);
    if (t < 0) return kEndOfStream;
    *time_us = t;
    return kOk;
}

Status Extractor::GetSampleMetaData(MediaData* out) {
    if (!impl_->ex) return kError;
    AMediaFormat* f = AMediaFormat_new();
    media_status_t r = AMediaExtractor_getSampleFormat(impl_->ex, f);
    if (r != AMEDIA_OK) {
        AMediaFormat_delete(f);
        return from_media(r);
    }
    out->Clear();
    out->store_->format = f;
    return kOk;
}

// Todo está ya "en caché": los datos son locales. El final es la duración más larga de las pistas.
Status Extractor::GetBufferedRange(long* start_us, long* end_us) {
    if (!impl_->ex) return kError;
    int64_t end = 0;
    for (size_t i = 0, n = AMediaExtractor_getTrackCount(impl_->ex); i < n; i++) {
        AMediaFormat* f = AMediaExtractor_getTrackFormat(impl_->ex, i);
        int64_t duration;
        if (AMediaFormat_getInt64(f, AMEDIAFORMAT_KEY_DURATION, &duration)) end = std::max(end, duration);
        AMediaFormat_delete(f);
    }
    *start_us = 0;
    *end_us = end;
    return kOk;
}

}  // namespace movie
