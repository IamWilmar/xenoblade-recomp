// nn::album: el álbum de capturas de la consola. El modo foto del juego (main+0x6bb96ac) entrega la captura en
// RGBA 1280x720 y mira el Result: 0 es "guardada", y unos pocos errores del módulo 214 (desc 100..199, 500..599) los
// enseña al jugador. Aquí no hay álbum: la captura se guarda como PNG en <datos>/capturas, así el modo foto sigue
// funcionando. Compartir (SaveAndShareScreenshot) necesita red: se guarda igual y se da por hecho, como si el jugador
// cerrara el diálogo de publicar.

#include <sys/stat.h>
#include <time.h>
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nnsdk/types.h"

namespace nn::account {
struct Uid;
}

namespace nn::album {

enum ImageSize { ImageSize_1280x720 = 0 };
enum ImageOrientation {};
enum AlbumReportOption {};

namespace {

constexpr int kWidth = 1280, kHeight = 720;
// Error genérico del álbum (módulo 214, desc 100..199): el juego lo enseña como "no se pudo guardar".
constexpr Result kResultAlbumError{214 | (100 << 9)};

std::string capture_dir() {
    const char* data = getenv("NIER_DATA");
    std::string dir = std::string(data ? data : "/sdcard/NieR") + "/capturas";
    mkdir(dir.c_str(), 0775);
    return dir;
}

void put_be32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 3; i >= 0; i--) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put_chunk(std::vector<uint8_t>& out, const char* type, const uint8_t* data, size_t size) {
    put_be32(out, static_cast<uint32_t>(size));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    if (size) out.insert(out.end(), data, data + size);
    put_be32(out, static_cast<uint32_t>(crc32(0, out.data() + start, static_cast<uInt>(size + 4))));
}

// PNG RGB de 8 bits sin entrelazar: el alfa del framebuffer no es transparencia, se tira.
bool write_png(const char* path, const uint8_t* rgba, int width, int height) {
    const size_t row = 1 + size_t(width) * 3;
    std::vector<uint8_t> raw(row * height);
    for (int y = 0; y < height; y++) {
        uint8_t* dst = &raw[row * y];
        const uint8_t* src = rgba + size_t(width) * 4 * y;
        *dst++ = 0;  // filtro None
        for (int x = 0; x < width; x++, src += 4) {
            *dst++ = src[0];
            *dst++ = src[1];
            *dst++ = src[2];
        }
    }
    uLongf packed_size = compressBound(raw.size());
    std::vector<uint8_t> packed(packed_size);
    if (compress2(packed.data(), &packed_size, raw.data(), raw.size(), Z_BEST_SPEED) != Z_OK) return false;

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> header;
    put_be32(header, width);
    put_be32(header, height);
    header.insert(header.end(), {8, 2, 0, 0, 0});  // 8 bits, RGB, deflate, filtro 0, sin entrelazar
    put_chunk(png, "IHDR", header.data(), header.size());
    put_chunk(png, "IDAT", packed.data(), packed_size);
    put_chunk(png, "IEND", nullptr, 0);

    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const bool ok = fwrite(png.data(), 1, png.size(), f) == png.size();
    return fclose(f) == 0 && ok;
}

Result save(const void* image, size_t size, ImageSize image_size) {
    if (image_size != ImageSize_1280x720 || !image || size < size_t(kWidth) * kHeight * 4) return kResultAlbumError;
    const time_t now = time(nullptr);
    tm local{};
    localtime_r(&now, &local);
    char name[64];
    strftime(name, sizeof name, "%Y%m%d_%H%M%S", &local);
    const std::string dir = capture_dir();
    std::string path;
    for (int n = 0;; n++) {  // varias en el mismo segundo
        path = dir + "/Xenoblade_" + name + (n ? "_" + std::to_string(n) : "") + ".png";
        struct stat st;
        if (stat(path.c_str(), &st) != 0) break;
    }
    if (!write_png(path.c_str(), static_cast<const uint8_t*>(image), kWidth, kHeight)) {
        printf("[album] no se pudo escribir %s\n", path.c_str());
        return kResultAlbumError;
    }
    printf("[album] captura: %s\n", path.c_str());
    return ResultSuccess;
}

}  // namespace

void Initialize() {}
void Finalize() {}

int GetImageWidth(ImageSize size) { return size == ImageSize_1280x720 ? kWidth : 0; }
int GetImageHeight(ImageSize size) { return size == ImageSize_1280x720 ? kHeight : 0; }

// La orientación solo es un dato para el visor del álbum: la imagen llega ya como se ve.
Result SaveScreenshot(const void* image, size_t size, ImageSize image_size, ImageOrientation, AlbumReportOption) {
    return save(image, size, image_size);
}

Result SaveAndShareScreenshot(const void* image, size_t size, ImageSize image_size, ImageOrientation,
                              AlbumReportOption, const account::Uid&, const char*) {
    return save(image, size, image_size);
}

// El applet del álbum no existe: vuelve enseguida, como si el jugador lo hubiera cerrado (el juego ignora el Result).
Result ShowAlbumFiles() {
    printf("[album] ShowAlbumFiles: no hay applet del álbum\n");
    return ResultSuccess;
}

}  // namespace nn::album
