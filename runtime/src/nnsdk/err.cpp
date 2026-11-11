// nn::err: los applets de error del sistema. Aquí no hay pantalla del sistema que los enseñe: el error va al
// log y se vuelve como si el usuario hubiera cerrado el applet. Sin red, los que llegan son casi siempre de
// servicios en línea que no existen (el juego sigue en modo sin conexión).
//
// ErrorResultVariant es del juego (el sdk lo deja casi todo en línea): +0 el estado y en +4 el Result o el
// ErrorCode de 8 bytes, sin alinear (sdk+0x15cef0).

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::err {

struct ErrorCode {
    uint32_t category;
    uint32_t number;
    bool IsValid() const;
};

enum ErrorResultVariantState : int {
    ErrorResultVariantState_None = 0,
    ErrorResultVariantState_HasResult = 1,
    ErrorResultVariantState_HasErrorCode = 2,
};

class ErrorResultVariant {
public:
    ErrorResultVariant();
    ErrorResultVariant(const Result& result);
    ErrorResultVariant& operator=(const ErrorCode& code);
    ErrorResultVariant& operator=(const Result& result);
    ErrorResultVariantState GetState() const;
    operator ErrorCode() const;
    operator Result() const;

private:
    ErrorResultVariantState state_;
    uint32_t value_[2];
};
static_assert(sizeof(ErrorResultVariant) == 12);

// El contexto de error de la plataforma de red (NPLN): no hay nada que añadirle.
class NplnContext {
public:
    static void Make(const void* data, size_t size);
};

template <typename Context>
Result UpdateResultWithContext(Result result, const Context& context);

namespace {

// Lo que la consola enseña: 2xxx-yyyy, con el módulo (+2000) y la descripción del Result.
void log_result(const char* what, Result r) {
    printf("[err] %s %04u-%04u (0x%x)\n", what, 2000 + (r.value & 0x1FF), (r.value >> 9) & 0x1FFF, r.value);
}

}  // namespace

bool ErrorCode::IsValid() const { return category != 0; }

ErrorResultVariant::ErrorResultVariant() : state_(ErrorResultVariantState_None) {}
ErrorResultVariant::ErrorResultVariant(const Result& result) { *this = result; }

// El Result se guarda sin los bits altos (reservados), como hace el sdk.
ErrorResultVariant& ErrorResultVariant::operator=(const Result& result) {
    state_ = ErrorResultVariantState_HasResult;
    value_[0] = result.value & 0x3FFFFF;
    return *this;
}
ErrorResultVariant& ErrorResultVariant::operator=(const ErrorCode& code) {
    state_ = ErrorResultVariantState_HasErrorCode;
    memcpy(value_, &code, sizeof code);
    return *this;
}
ErrorResultVariantState ErrorResultVariant::GetState() const { return state_; }
ErrorResultVariant::operator ErrorCode() const {
    ErrorCode code;
    memcpy(&code, value_, sizeof code);
    return code;
}
ErrorResultVariant::operator Result() const { return Result{value_[0]}; }

void NplnContext::Make(const void*, size_t) {}

template <>
Result UpdateResultWithContext<NplnContext>(Result result, const NplnContext&) {
    return result;
}

void ShowError(const ErrorResultVariant& error) {
    switch (error.GetState()) {
    case ErrorResultVariantState_HasResult:
        log_result("applet de error:", error);
        break;
    case ErrorResultVariantState_HasErrorCode: {
        ErrorCode code = error;
        printf("[err] applet de error: %04u-%04u\n", code.category, code.number);
        break;
    }
    default:
        printf("[err] applet de error sin código\n");
    }
}

}  // namespace nn::err
