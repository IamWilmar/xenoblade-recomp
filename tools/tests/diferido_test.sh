#!/bin/sh
# Compila tools/tests/diferido_test.cpp contra las fuentes de este árbol (en el Mac, con ASan y UBSan) y lo corre en sus
# cuatro modos: normal y con cada interruptor en vivo (sin_refs_llamadas, limbo_sin_valvulas, limbo_recientes).
# No toca build/ ni la consola: todo va a una carpeta temporal.
#
#   tools/tests/diferido_test.sh
#   BASE=8b8e464 tools/tests/diferido_test.sh   # además, ese commit prueba a prueba (ASan corta en el primer error)
#
# Las cabeceras de Vulkan salen del NDK (solo vulkan/ y vk_video/: el resto del sysroot es de Android).
set -e
tree="$(cd "$(dirname "$0")/../.." && pwd)"
ndk="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/28.2.13676358}/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/inc" "$tmp/datos"
ln -s "$ndk/vulkan" "$tmp/inc/vulkan"
ln -s "$ndk/vk_video" "$tmp/inc/vk_video"

build() {  # build <árbol> <salida> [-DBASE]
    clang++ -std=gnu++20 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -Wno-everything $3 \
        -I"$1/runtime/src" -I"$1/runtime/recompiler" -I"$tmp/inc" \
        "$tree/tools/tests/diferido_test.cpp" "$1/runtime/src/nnsdk/nvn/texture.cpp" "$1/runtime/src/nnsdk/nvn/cmdbuf.cpp" \
        "$1/runtime/src/nnsdk/nvn/layout.cpp" "$1/runtime/src/nnsdk/nvn/formats.cpp" -o "$2"
}

status=0
build "$tree" "$tmp/prueba"
for mode in normal sin_refs sin_valvulas recientes; do
    if "$tmp/prueba" "$tmp/datos" "$mode" > "$tmp/salida.txt" 2>&1; then code=0; else code=$?; fi
    errors="$(grep -c 'AddressSanitizer\|runtime error' "$tmp/salida.txt" || true)"
    echo "$mode: salida $code; $(tail -1 "$tmp/salida.txt"); errores de ASan/UBSan: $errors"
    if [ "$code" != 0 ] || [ "$errors" != 0 ]; then
        status=1
        grep 'FALLA\|ERROR\|runtime error' "$tmp/salida.txt" | head -20
    fi
done

if [ -n "$BASE" ]; then
    echo "== el código de $BASE, una prueba cada vez (sin transferir menciones, sin válvulas: lo que ese código esperaba)"
    mkdir -p "$tmp/base"
    git -C "$tree" archive --format=tar "$BASE" runtime/src/nnsdk runtime/recompiler/nvn_shader.h | tar -x -C "$tmp/base"
    build "$tmp/base" "$tmp/prueba_base" -DBASE
    for t in copia_pendiente copia_sin_enviar llamadas reinicio pool_finalizado valvula_edad valvula_tope recientes \
        cajas_sin_subir render_target pool_de_texturas; do
        mode=sin_valvulas
        [ "$t" = llamadas ] && mode=sin_refs
        if "$tmp/prueba_base" "$tmp/datos" "$mode" "$t" > "$tmp/salida.txt" 2>&1; then code=0; else code=$?; fi
        asan="$(grep -m1 -o 'AddressSanitizer: [a-z-]*' "$tmp/salida.txt" || true)"
        echo "$t: salida $code; $(grep -c '^FALLA' "$tmp/salida.txt" || true) comprobaciones fallidas; ${asan:-sin errores de ASan}"
    done
fi
exit $status
