// Puntos de control de la GPU (checkpoints): marcas que el driver de depuración de Nvidia deja en el flujo de comandos
// para saber, tras un cuelgue de la GPU, hasta dónde llegó. Disco Elysium pide los siete procs por nombre al cargar los
// punteros de NVN (main+0x151f474…0x151f51c) y no los llama nunca: cada puntero queda en una variable propia
// (main+0x50c79b8…0x50c79e8) cuyo slot del GOT (main+0x4d5cd40…0x4d5cd70) solo lo lee ese cargador.
//
// En el driver de venta de la consola (el del sdk de este juego) los siete son la misma función vacía que devuelve
// NVN_TRUE: `mov w0, #1; ret` en sdk+0x45d0b0 (InsertCheckpointStatic), +0x45d0c0 (InsertCheckpointDynamic), +0x45d0d0
// (SetAutomaticCheckpointsEnable), +0x45d0e0 (…SamplingInterval), +0x45d0f0 (…ByDebugGroup), +0x45d100
// (SetCheckpointCallStacksEnable) y +0x45d110 (SetCheckpointMaxCallStacksDepth); salen de la tabla de implementaciones
// del driver (sdk+0xc63e58, indexada por el nombre ordenado de sdk+0xabe898). Los checkpoints solo existen en el driver
// de desarrollo, así que aquí hacen lo mismo que en la consola: nada, y NVN_TRUE. No hay getters ni estado que guardar
// (el driver no escribe nada), y su equivalente en Vulkan (VK_NV_device_diagnostic_checkpoints) es una extensión del
// driver de Nvidia, no del Adreno.
//
// Como el driver no lee ningún argumento y el juego no tiene ni un sitio de llamada, de la firma solo se sabe el objeto
// (el NVNcommandBuffer o el NVNdevice, como en todo proc de su familia); el resto de registros se ignora.

#include "nnsdk/nvn/nvn.h"

extern "C" {

nvn::Boolean nvnCommandBufferInsertCheckpointStatic(void*) { return 1; }
nvn::Boolean nvnCommandBufferInsertCheckpointDynamic(void*) { return 1; }
nvn::Boolean nvnDeviceSetAutomaticCheckpointsEnable(void*) { return 1; }
nvn::Boolean nvnDeviceSetAutomaticCheckpointsSamplingInterval(void*) { return 1; }
nvn::Boolean nvnDeviceSetAutomaticCheckpointsByDebugGroup(void*) { return 1; }
nvn::Boolean nvnDeviceSetCheckpointCallStacksEnable(void*) { return 1; }
nvn::Boolean nvnDeviceSetCheckpointMaxCallStacksDepth(void*) { return 1; }

}  // extern "C"

NVN_PROC(nvnCommandBufferInsertCheckpointStatic);
NVN_PROC(nvnCommandBufferInsertCheckpointDynamic);
NVN_PROC(nvnDeviceSetAutomaticCheckpointsEnable);
NVN_PROC(nvnDeviceSetAutomaticCheckpointsSamplingInterval);
NVN_PROC(nvnDeviceSetAutomaticCheckpointsByDebugGroup);
NVN_PROC(nvnDeviceSetCheckpointCallStacksEnable);
NVN_PROC(nvnDeviceSetCheckpointMaxCallStacksDepth);
