#pragma once
// Parte de nn::os que otros módulos de libnnsdk reutilizan (cxx.cpp monta std::mutex encima).

#include <cstddef>

#include "nnsdk/types.h"

namespace nn::os {

void SignalEvent(EventType* e);
void ClearEvent(EventType* e);
void LockMutex(MutexType* m);
void UnlockMutex(MutexType* m);
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
// Crea (o reinicia) uno de esos SystemEvent con su modo; si nadie la llama, el primer uso lo crea ManualClear y
// sin señalar.
void create_system_event(nn::os::SystemEventType* event, bool auto_clear, bool signaled);

// Para QueryMemoryInfo (os_memory.cpp): hilos de nn::os vivos y la suma de sus pilas.
void thread_totals(size_t* stack_bytes, int* count);

}  // namespace nnsdk

namespace nnsdk {
// DEPURACIÓN: reloj y registro de esperas largas (os.cpp).
int64_t wait_clock_ns();
void log_slow_wait(const char* what, const void* object, int64_t start_ns, const void* caller);
void log_signal(const char* what, const void* object, const void* caller);
}  // namespace nnsdk
