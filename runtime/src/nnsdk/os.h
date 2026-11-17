#pragma once
// Parte de nn::os que otros módulos de libnnsdk reutilizan (cxx.cpp monta std::mutex encima).

#include <cstddef>

#include "nnsdk/types.h"

namespace nn::os {

void SignalEvent(EventType* e);
void ClearEvent(EventType* e);
void LockMutex(MutexType* m);
void UnlockMutex(MutexType* m);
bool TryLockMutex(MutexType* m);
void FinalizeMutex(MutexType* m);
void SignalConditionVariable(ConditionVariableType* c);
void BroadcastConditionVariable(ConditionVariableType* c);
void WaitConditionVariable(ConditionVariableType* c, MutexType* m);
ConditionVariableStatus TimedWaitConditionVariable(ConditionVariableType* c, MutexType* m, TimeSpan t);
void FinalizeConditionVariable(ConditionVariableType* c);

}  // namespace nn::os

namespace nnsdk {

// Señala un SystemEvent que crea el "sistema" (nosotros) y que el juego solo espera. El juego lo puede esperar
// con Wait/TryWait/TimedWaitSystemEvent o en un MultiWait (InitializeMultiWaitHolder con el SystemEventType).
void signal_system_event(nn::os::SystemEventType* event);
// Lo mismo, pero solo si el evento sigue existiendo (false si el juego ya lo destruyó o nunca se creó): para quien lo
// señala desde un hilo propio sin saber cuándo lo destruye el juego. No lo crea, y no se puede destruir a medias.
bool signal_system_event_if_alive(nn::os::SystemEventType* event);
bool clear_system_event_if_alive(nn::os::SystemEventType* event);
// Crea (o reinicia) uno de esos SystemEvent con su modo; si nadie la llama, el primer uso lo crea ManualClear y
// sin señalar.
void create_system_event(nn::os::SystemEventType* event, bool auto_clear, bool signaled);

// Para QueryMemoryInfo (os_memory.cpp): hilos de nn::os vivos y la suma de sus pilas.
void thread_totals(size_t* stack_bytes, int* count);
// Bytes comprometidos ahora mismo con nn::os::AllocateMemoryPages (memoria de direcciones virtuales).
size_t va_committed();
// Copia [src, src + size) a dst si está entero en una región de nn::os::AllocateAddressRegion mapeada legible: ahí leer no
// falla nunca (las páginas soltadas con FreeMemoryPages quedan a cero), así que es un memcpy, sin llamada al kernel. false,
// sin copiar nada, si no está en una (o la región se está liberando): quien llama lee entonces como si pudiera fallar.
bool copy_readable(void* dst, const void* src, size_t size);

}  // namespace nnsdk

namespace nnsdk {
// DEPURACIÓN: reloj y registro de esperas largas (os.cpp).
int64_t wait_clock_ns();
void log_slow_wait(const char* what, const void* object, int64_t start_ns, const void* caller);
void log_signal(const char* what, const void* object, const void* caller);
}  // namespace nnsdk
