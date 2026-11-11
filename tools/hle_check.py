#!/usr/bin/env python3
"""Compila un .cpp de runtime/src/nnsdk con las mismas opciones que libnnsdk y comprueba qué imports del juego da.

Uso: python3 tools/hle_check.py runtime/src/nnsdk/<archivo>.cpp [más .cpp]

Por cada archivo: errores de compilación, cuántos nombres de work/analysis/missing.tsv define (con el mangling exacto
que pide el juego) y los símbolos nn::/C que exporta sin estar en la lista: casi siempre un tipo de parámetro mal
declarado (el juego no los encontraría). Necesita build/android/compile_commands.json (tools/apk.sh o cmake lo
generan con -DCMAKE_EXPORT_COMPILE_COMMANDS=ON).
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NM = Path.home() / "Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-nm"


def template_command() -> list[str]:
    db = json.loads((ROOT / "build/android/compile_commands.json").read_text())
    for entry in db:
        if entry["file"].endswith("src/nnsdk/os.cpp"):
            return shlex.split(entry["command"])
    raise SystemExit("no encuentro os.cpp en compile_commands.json")


def compile_one(src: Path, out: Path) -> tuple[bool, str]:
    cmd = template_command()
    args, skip = [], False
    for a in cmd:
        if skip:
            skip = False
            continue
        if a in ("-o", "-c", "-MF", "-MT", "-MD"):
            skip = a in ("-o", "-c", "-MF", "-MT")
            continue
        if a.endswith("os.cpp"):
            continue
        args.append(a)
    args += ["-c", str(src), "-o", str(out)]
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def main():
    wanted = {}
    for line in (ROOT / "work/analysis/missing.tsv").read_text().splitlines()[1:]:
        group, mangled, pretty, kind = line.split("\t")
        wanted[mangled] = pretty
    imported = set((ROOT / "work/analysis/main_imports.txt").read_text().split())
    for path in sys.argv[1:]:
        src = Path(path).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "x.o"
            ok, errs = compile_one(src, obj)
            print(f"== {src.relative_to(ROOT)}: {'compila' if ok else 'NO COMPILA'}")
            if not ok:
                print("\n".join(l for l in errs.splitlines() if "error" in l)[:4000])
                continue
            warns = [l for l in errs.splitlines() if "warning" in l]
            if warns:
                print("\n".join(warns[:20]))
            syms = subprocess.run([str(NM), "-g", "--defined-only", "--format=just-symbols", str(obj)],
                                  capture_output=True, text=True).stdout.split()
        # Las de libc con nombre de bionic se exportan como nngame_<nombre> (loader/imports.cpp).
        syms = [s[len("nngame_"):] if s.startswith("nngame_") else s for s in syms]
        given = [s for s in syms if s in wanted]
        print(f"   da {len(given)} imports de la lista")
        odd = [s for s in syms if s not in imported and (s.startswith("_ZN2nn") or s.startswith("nn"))]
        if odd:
            demangled = subprocess.run(["c++filt"], input="\n".join(odd), capture_output=True, text=True).stdout
            print("   exporta y main NO lo importa con ese nombre (¿mangling?, o lo usaba NieR):")
            for s, d in zip(odd, demangled.splitlines()):
                print(f"     {d}   [{s}]")


if __name__ == "__main__":
    main()
