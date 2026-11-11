// nn::ngc: el filtro de palabras malsonantes. Sin diccionarios del sistema todo texto sale limpio: ninguna palabra
// marcada y nada enmascarado.
//
// El juego reserva el objeto (0x30 bytes, con su operator new) y solo importa el constructor, Initialize y
// Finalize; el resto lo llama por la vtable: +0x30 CheckProfanityWords (char16_t), +0x40 MaskProfanityWordsInText
// (char16_t) y +0x08 el destructor que borra. La clase tiene las funciones virtuales del SDK en su orden, así la
// vtable que pone nuestro constructor tiene las mismas posiciones, y el delete va al allocator del juego.

#include <cstddef>
#include <cstdint>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::ngc {

// Oculta salvo lo que importa main: las virtuales solo se alcanzan por la vtable.
class [[gnu::visibility("hidden")]] ProfanityFilter {
public:
    [[gnu::visibility("default")]] ProfanityFilter();
    virtual ~ProfanityFilter();
    virtual uint32_t GetContentVersion();
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const char** words, size_t count);
    virtual Result MaskProfanityWordsInText(int* count, char* text, uint32_t patterns);
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const uint16_t** words, size_t count);
    virtual Result CheckProfanityWords(uint32_t* results, uint32_t patterns, const char16_t** words, size_t count);
    virtual Result MaskProfanityWordsInText(int* count, uint16_t* text, uint32_t patterns);
    virtual Result MaskProfanityWordsInText(int* count, char16_t* text, uint32_t patterns);

    [[gnu::visibility("default")]] Result Initialize(void* work, unsigned long size);
    [[gnu::visibility("default")]] void Finalize();

    static void operator delete(void* p) { nnsdk::host().game_delete(p); }

private:
    bool initialized_ = false;
};
static_assert(sizeof(ProfanityFilter) <= 0x30);

namespace {

Result clean(uint32_t* results, size_t count) {
    for (size_t i = 0; results && i < count; i++) results[i] = 0;  // ningún patrón la marca
    return ResultSuccess;
}

Result unmasked(int* count) {
    if (count) *count = 0;  // el juego pasa nullptr
    return ResultSuccess;
}

}  // namespace

ProfanityFilter::ProfanityFilter() {}
ProfanityFilter::~ProfanityFilter() { Finalize(); }

Result ProfanityFilter::Initialize(void*, unsigned long) {
    initialized_ = true;
    return ResultSuccess;
}

void ProfanityFilter::Finalize() { initialized_ = false; }

uint32_t ProfanityFilter::GetContentVersion() { return 0; }

Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t, const char**, size_t count) {
    return clean(results, count);
}
Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t, const uint16_t**, size_t count) {
    return clean(results, count);
}
Result ProfanityFilter::CheckProfanityWords(uint32_t* results, uint32_t, const char16_t**, size_t count) {
    return clean(results, count);
}
Result ProfanityFilter::MaskProfanityWordsInText(int* count, char*, uint32_t) { return unmasked(count); }
Result ProfanityFilter::MaskProfanityWordsInText(int* count, uint16_t*, uint32_t) { return unmasked(count); }
Result ProfanityFilter::MaskProfanityWordsInText(int* count, char16_t*, uint32_t) { return unmasked(count); }

}  // namespace nn::ngc
