#!/usr/bin/env bash
# Compila los shaders propios del runtime (runtime/src/nnsdk/nvn/shaders/*.comp) a SPIR-V con el glslc del NDK y los
# deja como arrays en shaders/spirv.h (generado: se sube al repo para no depender de glslc al compilar).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GLSLC="${ANDROID_HOME:-$HOME/Library/Android/sdk}/ndk/28.2.13676358/shader-tools/darwin-x86_64/glslc"
DIR="$ROOT/runtime/src/nnsdk/nvn/shaders"
FSR="$ROOT/third_party/eden/externals/FidelityFX-FSR/ffx-fsr"
OUT="$DIR/spirv.h"
{
  echo "#pragma once"
  echo "// Generado por tools/shaders.sh: no editar."
  echo "#include <cstdint>"
  for src in "$DIR"/*.comp; do
    name=$(basename "$src" .comp)
    "$GLSLC" -O --target-env=vulkan1.1 -I "$FSR" "$src" -o "$DIR/$name.spv"
    echo "inline constexpr uint32_t k_${name}_spv[] = {"
    python3 -c "import struct,sys; d=open('$DIR/$name.spv','rb').read(); w=struct.unpack('<%dI'%(len(d)//4),d); print(','.join('0x%08x'%x for x in w))"
    echo "};"
    rm "$DIR/$name.spv"
  done
} > "$OUT"
echo "$OUT"
