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
    static ErrorCode GetInvalidErrorCode();
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
    ErrorResultVariant(const ErrorCode& code);
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

// Los códigos con los que el applet de error del SDK enseña en su lugar el aviso de "hace falta una suscripción de
// Nintendo Switch Online" (nn::account::ShowLicenseRequirementsForNetworkServiceForUnspecifiedUsers): 2306-0821, 0823
// y 0824, 2307-2118 y 2618-0590, 0592 y 0593 (sdk+0x15bdb0 con ErrorCode, sdk+0x15bc80 con Result).
bool is_license_requirement(const ErrorCode& code) {
    const uint32_t n = code.number;
    if (code.category == 2306) return n == 821 || n == 823 || n == 824;
    if (code.category == 2307) return n == 2118;
    if (code.category == 2618) return n == 590 || n == 592 || n == 593;
    return false;
}

// Ninguno de los dos applets existe aquí: el código va al log y se vuelve como si el usuario lo hubiera cerrado.
void show(const ErrorCode& code) {
    printf("[err] applet de error: %04u-%04u%s\n", code.category, code.number,
           is_license_requirement(code) ? " (en la consola, el aviso de suscripción al servicio en línea)" : "");
}

}  // namespace

bool ErrorCode::IsValid() const { return category != 0; }
// {0, 0}: el sdk devuelve x0 = 0 (sdk+0x15bc4c); IsValid lo da por no válido (categoría 0, sdk+0x15bc3c).
ErrorCode ErrorCode::GetInvalidErrorCode() { return {0, 0}; }

ErrorResultVariant::ErrorResultVariant() : state_(ErrorResultVariantState_None) {}
ErrorResultVariant::ErrorResultVariant(const Result& result) { *this = result; }
ErrorResultVariant::ErrorResultVariant(const ErrorCode& code) { *this = code; }

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

// Las dos formas directas (las que importa Disco Elysium, solo desde el plugin de Unity del SDK: main+0x155b420 y
// main+0x155b430, sin llamadores en el código gestionado). Con un Result el sdk enseña 2xxx-yyyy: 2000 + módulo y la
// descripción (sdk+0x15bc80); un ErrorCode llega entero en x0, la categoría en la mitad baja (sdk+0x15bdb0).
void ShowError(Result result) { show(ErrorCode{2000 + (result.value & 0x1FF), (result.value >> 9) & 0x1FFF}); }
void ShowError(ErrorCode code) { show(code); }

// "La versión de la aplicación (o de su contenido descargable) no vale para jugar con otros": el sdk abre el applet de
// error con un Result fijo del sistema, 2162-0100 (0xc8a2, sdk+0x15bfe8) y 2162-0101 (0xcaa2, sdk+0x15c048).
void ShowUnacceptableApplicationVersionError() { show(ErrorCode{2162, 100}); }
void ShowUnacceptableAddOnContentVersionError() { show(ErrorCode{2162, 101}); }

}  // namespace nn::err
