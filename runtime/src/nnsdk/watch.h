#pragma once
// Vigilancia de escrituras de la CPU en la memoria del juego (el memfd del heap y sus alias en los pools virtuales de
// NVN), para los espejos de buffers del executor (nvn/executor.cpp): en vez de revisar megas por hash en cada envío,
// sus páginas se dejan de solo lectura y la primera escritura de cada una salta (SIGSEGV), sube su versión y la vuelve
// a abrir. Quien copia vuelve a cerrar las páginas abiertas antes de leerlas, así ninguna escritura se pierde.
//
// Lo que escribe el kernel (pread, recv…) en memoria vigilada no salta: falla con EFAULT. Quien haga eso en memoria
// del juego llama antes a before_kernel_write. <datos>/sin_vigilancia lo apaga (los espejos vuelven al hash).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nnsdk::watch {

// os_memory: la reserva del heap (offset 0 del memfd en map, size bytes) y lo que le pasa.
void init_heap(uintptr_t map, size_t size);
void heap_changed(uintptr_t address, size_t size);  // páginas vaciadas (punch hole): otro contenido sin escribir
void heap_forget(uintptr_t address, size_t size);   // fuera del heap (PROT_NONE): ya no se vigilan
// nvn/memory.cpp: un pool virtual y lo que MapVirtual hace aparecer en él (heap_address 0: ceros).
void add_virtual_range(uintptr_t base, size_t size);
void virtual_mapped(uintptr_t address, size_t size, uintptr_t heap_address);

// Antes de que el kernel escriba en [p, p + size): las páginas vigiladas se abren (y cuentan como escritas).
void before_kernel_write(void* p, size_t size);

// ¿Se puede vigilar todo [address, address + size)? (heap o pools virtuales conocidos, y la vigilancia encendida)
bool watchable(uintptr_t address, size_t size);
// Las páginas de 4 KB de [address, address + size) que cambiaron desde la última llamada con estas claves (keys: una
// por página, del que llama; vacío la primera vez): quedan cerradas y se devuelven en tramos (offset, bytes) desde
// address, para copiarlas justo después.
void collect(uintptr_t address, size_t size, std::vector<uint64_t>& keys, std::vector<std::pair<size_t, size_t>>* runs);

// ¿Tiene memoria detrás todo [address, address + size)? Falso si alguna página es un hueco de un pool virtual (lo que
// MapVirtual dejó sin mapear: se lee como ceros). Lo que no es de un pool virtual, siempre.
bool fully_mapped(uintptr_t address, size_t size);

// Perfil: escrituras vigiladas que saltaron, tramos cerrados (llamadas a mprotect) y el tiempo en ellas.
uint64_t faults();
uint64_t protects();
uint64_t protect_ns();

}  // namespace nnsdk::watch
