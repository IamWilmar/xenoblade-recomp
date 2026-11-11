#pragma once
// thread_local del juego. Su código lee las variables con offsets fijos desde TPIDR_EL0 (el linker relajó los
// descriptores TLS a local-exec: movz/movk x0, #off; mrs xN, tpidr_el0; ldr [xN, x0]), con la disposición de musl
// en Horizon: el bloque del módulo empieza en TP+16. En Android TPIDR_EL0 es el TP de bionic y de TP+16 a TP+63
// están sus slots (el canario de la pila en TP+40, el hilo de ART en TP+56): el juego los pisaría.
//
// Así que cada `mrs xN, tpidr_el0` del text del juego se reescribe al cargar como un salto a un stub propio que
// deja en xN el TP del juego: un bloque por hilo con la imagen de .tdata, cuyo puntero va en el slot
// TLS_SLOT_APP de bionic (TP+16, libre desde que errno dejó de usarlo). El código de bionic y el nuestro siguen
// viendo su TPIDR_EL0 de siempre.

#include <cstddef>
#include <cstdint>
#include <string>

#include "loader/nso.h"

namespace tls {

// Reescribe los `mrs xN, tpidr_el0` del text de mod (antes de protect: el text aún es RW).
bool patch(Module& mod, std::string& err);

// La plantilla del bloque del módulo: image_size bytes de .tdata en image, total_size con .tbss (a cero) y su
// alineación. La da el crt0 del juego al llamar a __nnmusl_init_dso desde DT_INIT; los bloques creados antes
// (el hilo principal) la reciben entonces.
void set_template(const void* image, size_t image_size, size_t total_size, size_t align);

// Al empezar un hilo del juego: crea su bloque. Si un hilo llega al código del juego sin pasar por aquí (un
// pthread_create del propio juego), el stub lo crea la primera vez que el juego lee su TLS.
void thread_start();

}  // namespace tls
