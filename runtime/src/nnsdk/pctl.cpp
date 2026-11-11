// nn::pctl: control parental. No hay restricciones: la comunicación libre (contenido de otros jugadores,
// chat) está siempre permitida. Sin red el juego no tiene a quién mostrarle nada, pero no ve bloqueos.

namespace nn::pctl {

bool IsFreeCommunicationAvailable() { return true; }
// La consola pide el PIN si hay restricción (con el bool a true); sin ella, empieza sin preguntar.
bool TryBeginFreeCommunication(bool) { return true; }
void EndFreeCommunication() {}

}  // namespace nn::pctl
