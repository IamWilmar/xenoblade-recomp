// nn::prepo: los informes de juego (telemetría que la consola manda a Nintendo). Se aceptan y se tiran:
// cada Add y cada Save responden éxito sin escribir nada en el búfer del juego.
//
// Lo único que el juego lee de vuelta es PlayReport::GetCount, y solo para no guardar un informe vacío
// (main+0x7282190): se cuentan las entradas añadidas. Los objetos los reserva el juego con el tamaño del
// SDK (PlayReport 0x38, Array y Struct 0x20) y se usan sus mismos campos: búfer y tamaño donde los pone
// SetBuffer, y la cuenta en el campo que el SDK usa para la posición de escritura (PlayReport, +0x30)
// o para la cuenta (Array y Struct, +0x18).

#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::account {
struct Uid;
}

namespace nn::prepo {

struct Any64BitId {
    uint64_t id;
};

class Struct;

class Array {
public:
    Array();
    void SetBuffer(void* buffer, size_t size);
    void Clear();
    Result Add(int64_t value);
    Result Add(float value);
    Result Add(double value);
    Result Add(bool value);
    Result Add(const char* value);
    Result Add(const Any64BitId& value);
    Result Add(const Struct& value);

private:
    Result add() {
        count_++;
        return ResultSuccess;
    }

    void* buffer_;
    size_t size_;
    [[maybe_unused]] size_t position_;
    size_t count_;  // +0x18
};
static_assert(sizeof(Array) == 0x20);

class Struct {
public:
    Struct();
    void SetBuffer(void* buffer, size_t size);
    void Clear();
    Result Add(const char* key, int64_t value);
    Result Add(const char* key, float value);
    Result Add(const char* key, double value);
    Result Add(const char* key, bool value);
    Result Add(const char* key, const char* value);
    Result Add(const char* key, const Any64BitId& value);

private:
    Result add() {
        count_++;
        return ResultSuccess;
    }

    void* buffer_;
    size_t size_;
    [[maybe_unused]] size_t position_;
    size_t count_;  // +0x18
};
static_assert(sizeof(Struct) == 0x20);

class PlayReport {
public:
    PlayReport();
    explicit PlayReport(const char* event_id);
    Result SetEventId(const char* id);
    void SetBuffer(void* buffer, size_t size);
    void Clear();
    int GetCount() const;
    Result Add(const char* key, int64_t value);
    Result Add(const char* key, float value);
    Result Add(const char* key, double value);
    Result Add(const char* key, bool value);
    Result Add(const char* key, const char* value);
    Result Add(const char* key, const Any64BitId& value);
    Result Add(const char* key, const Array& value);
    Result Add(const char* key, const Struct& value);
    Result Add(const char* key, const void* data, size_t size);
    Result Save();
    Result Save(const account::Uid& user);

private:
    friend Result AddSessionId(PlayReport* report, const char* id);
    Result add() {
        count_++;
        return ResultSuccess;
    }

    char event_id_[0x20];
    void* buffer_;  // +0x20
    size_t size_;   // +0x28
    size_t count_;  // +0x30
};
static_assert(sizeof(PlayReport) == 0x38);

Array::Array() { memset(this, 0, sizeof *this); }
void Array::SetBuffer(void* buffer, size_t size) {
    buffer_ = buffer;
    size_ = size;
    position_ = count_ = 0;
}
void Array::Clear() { position_ = count_ = 0; }
Result Array::Add(int64_t) { return add(); }
Result Array::Add(float) { return add(); }
Result Array::Add(double) { return add(); }
Result Array::Add(bool) { return add(); }
Result Array::Add(const char*) { return add(); }
Result Array::Add(const Any64BitId&) { return add(); }
Result Array::Add(const Struct&) { return add(); }

Struct::Struct() { memset(this, 0, sizeof *this); }
void Struct::SetBuffer(void* buffer, size_t size) {
    buffer_ = buffer;
    size_ = size;
    position_ = count_ = 0;
}
void Struct::Clear() { position_ = count_ = 0; }
Result Struct::Add(const char*, int64_t) { return add(); }
Result Struct::Add(const char*, float) { return add(); }
Result Struct::Add(const char*, double) { return add(); }
Result Struct::Add(const char*, bool) { return add(); }
Result Struct::Add(const char*, const char*) { return add(); }
Result Struct::Add(const char*, const Any64BitId&) { return add(); }

PlayReport::PlayReport() { memset(this, 0, sizeof *this); }
PlayReport::PlayReport(const char* event_id) : PlayReport() { SetEventId(event_id); }

Result PlayReport::SetEventId(const char* id) {
    strncpy(event_id_, id ? id : "", sizeof event_id_ - 1);
    event_id_[sizeof event_id_ - 1] = 0;
    return ResultSuccess;
}

void PlayReport::SetBuffer(void* buffer, size_t size) {
    buffer_ = buffer;
    size_ = size;
    count_ = 0;
}
void PlayReport::Clear() { count_ = 0; }
int PlayReport::GetCount() const { return static_cast<int>(count_); }

Result PlayReport::Add(const char*, int64_t) { return add(); }
Result PlayReport::Add(const char*, float) { return add(); }
Result PlayReport::Add(const char*, double) { return add(); }
Result PlayReport::Add(const char*, bool) { return add(); }
Result PlayReport::Add(const char*, const char*) { return add(); }
Result PlayReport::Add(const char*, const Any64BitId&) { return add(); }
Result PlayReport::Add(const char*, const Array&) { return add(); }
Result PlayReport::Add(const char*, const Struct&) { return add(); }
Result PlayReport::Add(const char*, const void*, size_t) { return add(); }

// Guardar es entregarlo al servicio, que lo mandaría cuando haya red: aquí se descarta.
Result PlayReport::Save() { return ResultSuccess; }
Result PlayReport::Save(const account::Uid&) { return ResultSuccess; }

Result AddSessionId(PlayReport* report, const char*) { return report->add(); }
Result RequestImmediateTransmission() { return ResultSuccess; }

}  // namespace nn::prepo
