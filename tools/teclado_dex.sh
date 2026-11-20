#!/bin/bash
# Compila runtime/src/nnsdk/teclado/Teclado.java a un dex y lo deja como array de C en teclado_dex.inc (va dentro de
# libnnsdk; teclado_android.cpp lo carga en memoria). Hay que volver a correrlo solo si cambia Teclado.java.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
BT="$SDK/build-tools/35.0.0"
JAR="$SDK/platforms/android-33/android.jar"
SRC="$ROOT/runtime/src/nnsdk/teclado"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
javac --release 8 -nowarn -cp "$JAR" -d "$TMP/clases" "$SRC/Teclado.java" 2> >(grep -v 'obsolete\|-Xlint' >&2 || true)
"$BT/d8" --release --min-api 26 --lib "$JAR" --output "$TMP" $(find "$TMP/clases" -name '*.class')
{
  echo "// Generado por tools/teclado_dex.sh a partir de Teclado.java: no editar a mano."
  echo "alignas(4) static const unsigned char kTecladoDex[] = {"
  xxd -i < "$TMP/classes.dex"
  echo "};"
} > "$SRC/teclado_dex.inc"
echo "teclado_dex.inc: $(wc -c < "$TMP/classes.dex") bytes de dex"
