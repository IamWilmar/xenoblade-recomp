// El teclado de verdad para nn::swkbd: un cuadro de texto de Android (AlertDialog con un EditText y el teclado del
// sistema) encima del juego. Lo abre swkbd.cpp cuando el juego pide un texto (el nombre del cazador, del Felyne…).
#pragma once

#include <string>

namespace nn::swkbd::android_ui {

struct Pedido {
    std::u16string titulo;   // la cabecera del juego (o su texto guía si no trae cabecera)
    std::u16string guia;     // el texto gris dentro del campo vacío
    std::u16string mensaje;  // debajo del título: por qué se rechazó el texto anterior
    std::u16string inicial;  // el texto con el que se abre
    std::u16string ok;       // el texto del botón de aceptar (vacío: el de Android)
    int max = 0;             // longitud máxima en caracteres UTF-16 (0: sin límite)
    bool clave = false;      // contraseña: puntos en vez de letras
    bool lineas = false;     // admite saltos de línea
};

enum Estado { Abierto = 0, Aceptado = 1, Cancelado = 2, Fallo = 3 };

// Si hay que usar el cuadro: hay actividad de Android y no está <datos>/nombre.txt (las pruebas automáticas dan los
// nombres por ahí) ni <datos>/sin_teclado.
bool disponible();
// Abre el cuadro sin esperar. false si no se puede (otro abierto, sin actividad): el que llama sigue como antes.
bool abrir(const Pedido& pedido);
// Cómo va el cuadro abierto: Abierto mientras el jugador escribe; al acabar, Aceptado (con el texto en *texto),
// Cancelado o Fallo (no se pudo mostrar), una sola vez, y el cuadro queda libre para el siguiente.
Estado estado(std::u16string* texto);
// Lo cierra sin respuesta (el juego retiró su teclado); estado() dará Cancelado.
void cerrar();
// abrir() y esperar a que acabe (para ShowKeyboard, que bloquea al juego como el applet).
Estado preguntar(const Pedido& pedido, std::u16string* texto);
// El idioma del sistema es español (para los pocos textos que pone el port).
bool en_espanol();

}  // namespace nn::swkbd::android_ui
