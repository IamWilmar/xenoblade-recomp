// nn::image: decodificadores PNG y JPEG del SDK. No hay ninguno aquí y el juego no los necesita para jugar:
// - PngDecoder: carga la imagen de copyright que la consola estampa en las capturas (main+0x6bb98ac, antes de
//   oe::SetCopyrightImage). Analyze devuelve 1, el error que da el SDK cuando libpng no arranca; el juego trata 1 y 2
//   como fallo, libera y sigue sin imagen de copyright.
// - JpegDecoder: crea una textura desde los bytes de un JPEG que le pasa un script (main+0x6f3ac40: imágenes de
//   perfil o descargadas). Analyze devuelve JpegStatus_UnsupportedFormat; el juego entonces da la textura nula.
// Los objetos son del juego (en su pila); se respeta el layout del SDK en los campos que se tocan y no se escribe
// más allá. La vtable (offset 0) no se usa: el juego llama a los métodos directamente.

#include <cstdint>
#include <cstdio>

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
    : vtable_(nullptr), state_(0), data_(nullptr), size_(0), format_(), bit_depth_(0), dimension_{}, row_stride_(0) {}

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
    : vtable_(nullptr), state_(0), data_(nullptr), size_(0), denominator_(1), format_(static_cast<PixelFormat>(14)),
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

JpegStatus JpegDecoder::Analyze() {
    printf("[image] JPEG de %zu bytes: sin decodificador\n", size_);
    return JpegStatus_UnsupportedFormat;
}

Dimension JpegDecoder::GetAnalyzedDimension() const { return dimension_; }
size_t JpegDecoder::GetAnalyzedWorkBufferSize() const { return work_size_; }

JpegStatus JpegDecoder::Decode(void*, size_t, int, void*, size_t) { return JpegStatus_UnsupportedFormat; }

}  // namespace nn::image
