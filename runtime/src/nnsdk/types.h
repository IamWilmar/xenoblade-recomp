#pragma once
// Tipos del SDK de Nintendo con los nombres exactos que usa el mangling de los imports.
// Los tipos de objetos (MutexType, ThreadType…) son opacos: el juego reserva su memoria y
// libnnsdk guarda el estado real en tablas propias indexadas por su dirección.

#include <cstdint>

namespace nn {

// Trivialmente copiables: se pasan y devuelven en registros, igual que en el SDK.
struct Result {
    uint32_t value;
};
struct TimeSpan {
    int64_t ns;
};

constexpr Result ResultSuccess{0};

namespace os {

struct MutexType;
struct ConditionVariableType;
struct EventType;
struct SystemEventType;
struct SemaphoreType;
struct ReaderWriterLockType;
struct ThreadType;
struct MultiWaitType;
struct MultiWaitHolderType;

struct Tick {
    int64_t value;
};
struct TlsSlot {
    uint32_t inner;
};

enum EventClearMode { EventClearMode_ManualClear = 0, EventClearMode_AutoClear = 1 };
enum ConditionVariableStatus { ConditionVariableStatus_Timeout = 0, ConditionVariableStatus_NoTimeout = 1 };

using ThreadFunction = void (*)(void*);
using TlsDestructor = void (*)(uintptr_t);

}  // namespace os
}  // namespace nn
