# Game folder on the device (<data>): NIER_DATA, or the one chosen in the app's start screen (files/carpeta.txt).
# Usage: source "$ROOT/tools/datos.sh" (sets PKG and DATA).
# Carpeta del juego en el dispositivo: NIER_DATA, o la que eligió el usuario en la pantalla de inicio de la app.
PKG=io.github.iamwilmar.xenobladeport
DATA="${NIER_DATA:-$(adb shell run-as "$PKG" cat files/carpeta.txt 2> /dev/null | tr -d '\r')}"
if [ -z "$DATA" ]; then
  echo "Game folder unknown: choose it once in the app, or run with NIER_DATA=<folder on the device>." >&2
  exit 1
fi
