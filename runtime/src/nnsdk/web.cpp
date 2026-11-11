// nn::web: el navegador del sistema (applet). El juego lo abre desde un script (main+0x6c9de18): una página web o un
// documento HTML de la propia aplicación (el manual), y le devuelve al script el Result. No hay applet: se responde con
// el único error que el SDK devuelve en vez de abortar (módulo 210, desc 1: el applet terminó sin datos de vuelta), y
// el valor de vuelta se queda como lo dejó su constructor.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::web {

namespace {

constexpr Result kResultWebNotAvailable{210 | (1 << 9)};

}  // namespace

// Los argumentos son 8 KB de TLV que el SDK empuja al applet: {u16 número de entradas, u16, u32 tipo} y entradas
// {u16 tipo, u16 tamaño, u32 0, datos} pegadas sin alinear.
class ShowWebPageArg {
public:
    explicit ShowWebPageArg(const char* url);

private:
    uint8_t data_[0x2000];
};

class WebPageReturnValue {
public:
    WebPageReturnValue();

private:
    uint8_t data_[0x2000];
};

class ShowOfflineHtmlPageArg {
public:
    explicit ShowOfflineHtmlPageArg(const char* path);

private:
    uint8_t data_[0x2000];
};

class OfflineHtmlPageReturnValue {
public:
    OfflineHtmlPageReturnValue();

private:
    [[maybe_unused]] uint64_t exit_reason_;
    char last_url_[0x1008];
};
static_assert(sizeof(OfflineHtmlPageReturnValue) == 0x1010);

// Igual que el SDK: cabecera, la entrada 0xd (1 byte = 1) y la URL como entrada 1 de 0xc00 bytes.
ShowWebPageArg::ShowWebPageArg(const char* url) {
    memset(data_ + 2, 0, sizeof data_ - 2);
    const uint64_t header = 0x0001000d00000005ull;
    memcpy(data_ + 4, &header, sizeof header);
    data_[0x10] = 1;
    const uint16_t count = 1;
    memcpy(data_, &count, sizeof count);
    const size_t size = strlen(url) + 1;
    if (size > 0xc00) return;
    const uint64_t entry = 0x0c000001ull;
    memcpy(data_ + 0x11, &entry, sizeof entry);
    memcpy(data_ + 0x19, url, size);
    const uint16_t two = 2;
    memcpy(data_, &two, sizeof two);
}

WebPageReturnValue::WebPageReturnValue() { memset(data_, 0, sizeof data_); }

// Nadie lee este argumento (no hay applet al que empujarlo): basta con dejarlo limpio.
ShowOfflineHtmlPageArg::ShowOfflineHtmlPageArg(const char*) { memset(data_, 0, sizeof data_); }

// 0x14: lo que pone el SDK antes de que el applet conteste.
OfflineHtmlPageReturnValue::OfflineHtmlPageReturnValue() : exit_reason_(0x14) { memset(last_url_, 0, sizeof last_url_); }

Result ShowWebPage(WebPageReturnValue*, const ShowWebPageArg&) {
    printf("[web] ShowWebPage: no hay navegador\n");
    return kResultWebNotAvailable;
}

Result ShowOfflineHtmlPage(OfflineHtmlPageReturnValue*, const ShowOfflineHtmlPageArg&) {
    printf("[web] ShowOfflineHtmlPage: no hay navegador\n");
    return kResultWebNotAvailable;
}

}  // namespace nn::web
