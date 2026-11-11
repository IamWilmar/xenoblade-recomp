#!/usr/bin/env bash
# Fetches third_party/ (not in git): the sources used by Eden's shader recompiler, all pinned to known versions, with
# this project's patches on top (tools/patches/eden-*.patch).
# Prepara third_party/ (no va en git): las fuentes que usa el recompilador de shaders de Eden, fijadas a versiones
# conocidas, con los parches propios encima.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone() {  # <folder> <repo> <tag>: shallow clone of a fixed version
  [ -d "$TP/$1" ] || git clone -q --depth 1 --branch "$3" "$2" "$TP/$1"
}

# Eden (GPL-3.0-or-later): only src/shader_recompiler and the headers in src/common, at the commit this port was built
# and tested with.
EDEN_COMMIT=060026dc2d1daa749cf0b39378f132f506cbfd93
if [ ! -d "$TP/eden" ]; then
  git init -q "$TP/eden"
  git -C "$TP/eden" remote add origin https://git.eden-emu.dev/eden-emu/eden.git
  git -C "$TP/eden" sparse-checkout set src/shader_recompiler src/common
  git -C "$TP/eden" fetch -q --depth 1 --filter=blob:none origin "$EDEN_COMMIT"
  git -C "$TP/eden" checkout -q FETCH_HEAD
  for patch in "$ROOT"/tools/patches/eden-*.patch; do git -C "$TP/eden" apply "$patch"; done
fi
# Same versions as Eden's cpmfile.json.
clone fmt https://github.com/fmtlib/fmt.git 12.1.0
clone sirit https://github.com/eden-emulator/sirit.git v1.0.5
clone SPIRV-Headers https://github.com/KhronosGroup/SPIRV-Headers.git vulkan-sdk-1.4.341.0
# libopus (BSD-3-Clause): nn::codec's HardwareOpusDecoder (runtime/src/nnsdk/codec.cpp).
clone opus https://github.com/xiph/opus.git v1.6.1
# sirit expects SPIRV-Headers as a submodule in externals/.
rmdir "$TP/sirit/externals/SPIRV-Headers" 2> /dev/null || true
[ -e "$TP/sirit/externals/SPIRV-Headers" ] || ln -s ../../SPIRV-Headers "$TP/sirit/externals/SPIRV-Headers"

# Official Boost 1.92.0: only each module's headers (everything used is header-only).
if [ ! -d "$TP/boost" ]; then
  tmp="$TP/boost-dl"
  mkdir -p "$tmp"
  curl -fsSL -o "$tmp/boost.tar.xz" \
    https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.xz
  if tar --version 2> /dev/null | grep -q GNU; then
    tar xf "$tmp/boost.tar.xz" -C "$tmp" --wildcards --wildcards-match-slash '*/libs/*/include/*'
  else
    tar xf "$tmp/boost.tar.xz" -C "$tmp" --include='*/libs/*/include/*'
  fi
  mv "$tmp/boost-1.92.0" "$TP/boost"
  rm -rf "$tmp"
fi
echo "third_party ready"
