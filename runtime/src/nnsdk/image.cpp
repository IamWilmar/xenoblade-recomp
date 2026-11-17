// nn::image: decodificadores PNG y JPEG del SDK. No hay ninguno aquí y el juego no los necesita para jugar:
// - PngDecoder: carga la imagen de copyright que la consola estampa en las capturas (main+0x6bb98ac, antes de
//   oe::SetCopyrightImage). Analyze devuelve 1, el error que da el SDK cuando libpng no arranca; el juego trata 1 y 2
//   como fallo, libera y sigue sin imagen de copyright.
// - JpegDecoder: con AImageDecoder de Android (API 30). Xenoblade 3 trae JPEG dentro de sus layouts (las miniaturas de
//   los consejos de carga, menu/image/*/strm_tips_thumb*.wilay) y sin decodificarlos leía un puntero nulo y se caía al
//   terminar la película de la intro (main+0x12745d0). Antes (Xenoblade DE: imágenes de perfil de un script) Analyze
//   devolvía JpegStatus_UnsupportedFormat y el juego daba la textura nula.
// Los objetos son del juego (en su pila o en su heap); se respeta el layout del SDK en los campos que se tocan. La vtable
// (offset 0) es como la del SDK: [0] destructor, [1] destructor que libera con el operator delete del juego. Xenoblade 3
// crea el JpegDecoder con su allocator (main+0x1274620) y lo destruye por la vtable (main+0x12745d0): con la vtable a
// nulo se caía al terminar la intro (segfault en 0x8).

#include <android/bitmap.h>
#include <android/imagedecoder.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "nnsdk/host.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::image {

enum PixelFormat {};

enum JpegStatus {
    JpegStatus_Ok = 0,
    JpegStatus_WrongFormat = -32,
    JpegStatus_UnsupportedFormat = -33,
    JpegStatus_OutOfMemory = -64,
};

struct Dimension {
    int32_t width;
    int32_t height;
};

namespace {

constexpr int kPngFailed = 1;

// Vtables como las del SDK (D1 no hace nada; D0 libera con el delete del juego, que es quien lo reservó).
void destroy(void*) {}
void destroy_and_delete(void* self) {
    if (auto del = nnsdk::host().game_delete) del(self);
}
void* const kVtable[] = {reinterpret_cast<void*>(&destroy), reinterpret_cast<void*>(&destroy_and_delete)};

}  // namespace

class PngDecoder {
public:
    PngDecoder();
    ~PngDecoder();
    void Initialize(void* (*allocate)(size_t, void*), void* allocate_arg, void (*free)(void*, void*), void* free_arg);
    void SetImageData(const void* data, size_t size);
    void SetOutputPixelFormat(PixelFormat format);
    int Analyze();
    Dimension GetDimension() const;
    size_t GetRowStrideMin() const;
    int Decode(void* out, size_t out_size, size_t row_stride);

private:
    [[maybe_unused]] void* vtable_;
    int state_;  // 0 creado, 1 con allocator, 2 con datos, 3 analizado, 4 decodificado
    const void* data_;
    size_t size_;
    PixelFormat format_;
    [[maybe_unused]] int bit_depth_;
    Dimension dimension_;
    size_t row_stride_;
};
static_assert(sizeof(PngDecoder) == 0x38);

PngDecoder::PngDecoder()
    : vtable_(const_cast<void**>(kVtable)), state_(0), data_(nullptr), size_(0), format_(), bit_depth_(0), dimension_{},
      row_stride_(0) {}

PngDecoder::~PngDecoder() {}

void PngDecoder::Initialize(void* (*)(size_t, void*), void*, void (*)(void*, void*), void*) { state_ = 1; }

void PngDecoder::SetImageData(const void* data, size_t size) {
    data_ = data;
    size_ = size;
    state_ = 2;
}

void PngDecoder::SetOutputPixelFormat(PixelFormat format) {
    format_ = format;
    state_ = state_ > 1 ? 2 : 1;
}

int PngDecoder::Analyze() {
    printf("[image] PNG de %zu bytes: sin decodificador\n", size_);
    return kPngFailed;
}

Dimension PngDecoder::GetDimension() const { return dimension_; }
size_t PngDecoder::GetRowStrideMin() const { return row_stride_; }

int PngDecoder::Decode(void*, size_t, size_t) {
    state_ = 4;
    return kPngFailed;
}

class JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();
    void SetImageData(const void* data, size_t size);
    void SetPixelFormat(PixelFormat format);
    void SetResolutionDenominator(int denominator);
    JpegStatus Analyze();
    Dimension GetAnalyzedDimension() const;
    size_t GetAnalyzedWorkBufferSize() const;
    JpegStatus Decode(void* out, size_t out_size, int alignment, void* work, size_t work_size);

private:
    [[maybe_unused]] void* vtable_;
    int state_;  // 0 sin datos, 1 con datos, 2 analizado
    const void* data_;
    size_t size_;
    int denominator_;
    PixelFormat format_;
    Dimension dimension_;
    size_t work_size_;
};
static_assert(sizeof(JpegDecoder) == 0x38);

// Como el SDK: denominador 1 y formato 14 (el que pide el juego) por defecto.
JpegDecoder::JpegDecoder()
    : vtable_(const_cast<void**>(kVtable)), state_(0), data_(nullptr), size_(0), denominator_(1), format_(static_cast<PixelFormat>(14)),
      dimension_{}, work_size_(0) {}

JpegDecoder::~JpegDecoder() {}

void JpegDecoder::SetImageData(const void* data, size_t size) {
    data_ = data;
    size_ = size;
    state_ = 1;
}

void JpegDecoder::SetPixelFormat(PixelFormat format) {
    format_ = format;
    state_ = data_ != nullptr;
}

void JpegDecoder::SetResolutionDenominator(int denominator) {
    denominator_ = denominator;
    state_ = data_ != nullptr;
}

// Layout del SDK 12 (Xenoblade 3; JpegDecoder::Analyze del sdk): dimensión en +0x28, tamaño del buffer de trabajo en
// +0x30, "hay EXIF" en +0x38 (y su puntero y tamaño en +0x40/+0x48). El juego lee la dimensión del objeto sin llamar a
// GetAnalyzedDimension; los campos de EXIF quedan a cero (sin EXIF). El objeto es del juego, del tamaño del SDK.
JpegStatus JpegDecoder::Analyze() {
    if (!data_ || !size_) return JpegStatus_WrongFormat;
    AImageDecoder* dec = nullptr;
    if (AImageDecoder_createFromBuffer(data_, size_, &dec) != ANDROID_IMAGE_DECODER_SUCCESS || !dec) {
        printf("[image] JPEG de %zu bytes: no se pudo leer\n", size_);
        return JpegStatus_WrongFormat;
    }
    const AImageDecoderHeaderInfo* info = AImageDecoder_getHeaderInfo(dec);
    const int d = denominator_ == 2 || denominator_ == 4 || denominator_ == 8 ? denominator_ : 1;
    dimension_.width = (AImageDecoderHeaderInfo_getWidth(info) + d - 1) / d;
    dimension_.height = (AImageDecoderHeaderInfo_getHeight(info) + d - 1) / d;
    AImageDecoder_delete(dec);
    work_size_ = 4096;  // no hace falta (AImageDecoder tiene su memoria): lo mínimo para que el juego no pida nada raro
    uint8_t* self = reinterpret_cast<uint8_t*>(this);
    memset(self + 0x38, 0, 0x12);  // sin EXIF (+0x38 bandera, +0x40 puntero, +0x48 tamaño)
    state_ = 2;
    static int logged = 0;
    if (logged++ < 8) printf("[image] JPEG de %zu bytes: %dx%d (denominador %d)\n", size_, dimension_.width,
                             dimension_.height, d);
    return JpegStatus_Ok;
}

Dimension JpegDecoder::GetAnalyzedDimension() const { return dimension_; }
size_t JpegDecoder::GetAnalyzedWorkBufferSize() const { return work_size_; }

// Salida según SetPixelFormat, como el JpegDecoder::Decode del sdk (que solo admite estos dos): 0xe, RGBA8 (lee RGB y
// pone el alfa a 0xff, 4 bytes por píxel); 0xa, RGB8 tal cual sale de libjpeg (3 bytes). Filas alineadas a `alignment`
// píxeles. Xenoblade 3 pide 0xa para las miniaturas de los consejos; antes el formato salía del tamaño de out (RGBA si
// cabía) y, con un buffer holgado, el juego leía como RGB filas de RGBA: la imagen salía torcida y desvaída.
JpegStatus JpegDecoder::Decode(void* out, size_t out_size, int alignment, void*, size_t) {
    if (state_ < 2 || !out) return JpegStatus_WrongFormat;
    const int w = dimension_.width, h = dimension_.height;
    const int align = alignment > 0 ? alignment : 1;
    const size_t row_px = (static_cast<size_t>(w) + align - 1) / align * align;
    const int bpp = static_cast<int>(format_) == 0xa ? 3 : 4;
    if (w <= 0 || h <= 0 || out_size < row_px * h * bpp) return JpegStatus_OutOfMemory;
    static int logged = 0;
    if (logged++ < 4) printf("[image] JPEG %dx%d a formato 0x%x (%d bytes por píxel), salida de %zu bytes\n", w, h,
                             static_cast<int>(format_), bpp, out_size);
    AImageDecoder* dec = nullptr;
    if (AImageDecoder_createFromBuffer(data_, size_, &dec) != ANDROID_IMAGE_DECODER_SUCCESS || !dec)
        return JpegStatus_WrongFormat;
    AImageDecoder_setAndroidBitmapFormat(dec, ANDROID_BITMAP_FORMAT_RGBA_8888);
    AImageDecoder_setUnpremultipliedRequired(dec, true);
    if (denominator_ > 1) AImageDecoder_setTargetSize(dec, w, h);
    int r;
    if (bpp == 4) {
        r = AImageDecoder_decodeImage(dec, out, row_px * 4, row_px * 4 * h);
    } else {
        std::vector<uint8_t> rgba(static_cast<size_t>(w) * 4 * h);
        r = AImageDecoder_decodeImage(dec, rgba.data(), static_cast<size_t>(w) * 4, rgba.size());
        auto* o = static_cast<uint8_t*>(out);
        for (int y = 0; y < h && r == ANDROID_IMAGE_DECODER_SUCCESS; y++)
            for (int x = 0; x < w; x++) memcpy(o + (y * row_px + x) * 3, &rgba[(static_cast<size_t>(y) * w + x) * 4], 3);
    }
    AImageDecoder_delete(dec);
    if (r != ANDROID_IMAGE_DECODER_SUCCESS) {
        printf("[image] JPEG %dx%d: decodeImage %d\n", w, h, r);
        return JpegStatus_WrongFormat;
    }
    return JpegStatus_Ok;
}

}  // namespace nn::image
