#!/usr/bin/env python3
"""Lee un vtable exportado de un NSO y nombra cada slot con el símbolo de la función.

Uso: python3 tools/vtable.py <nso> <clase C++>      p. ej. movie::Decoder

Busca el símbolo `vtable for <clase>`, lee sus entradas (relocaciones RELATIVE o
ABS64) y muestra cada slot desde el address point (+0x10), que es lo que el juego
indexa con `ldr x8, [x8, #slot]`.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nso  # noqa: E402

R_ABS64, R_RELATIVE = 257, 1027
DT_RELA, DT_RELASZ = 7, 8


def main():
    mod = nso.Nso(Path(sys.argv[1]).name, Path(sys.argv[1]).read_bytes())
    wanted = f"vtable for {sys.argv[2]}"
    names = nso.demangle(s.name for s in mod.symbols if s.name)
    vt = next((s for s in mod.symbols if s.shndx and names.get(s.name) == wanted), None)
    if not vt:
        raise SystemExit(f"no encuentro '{wanted}' en {sys.argv[1]}")

    img = mod.image
    tags, off = {}, mod.dynamic_off
    while True:
        tag, val = struct.unpack_from("<qQ", img, off)
        off += 16
        if tag == 0:
            break
        tags.setdefault(tag, val)
    relocs = {}
    for o in range(tags[DT_RELA], tags[DT_RELA] + tags[DT_RELASZ], 24):
        r_off, info, addend = struct.unpack_from("<QQq", img, o)
        if vt.value <= r_off < vt.value + vt.size:
            relocs[r_off] = (info & 0xFFFFFFFF, info >> 32, addend)

    by_addr = {s.value: names.get(s.name, s.name) for s in mod.symbols if s.shndx and s.type == 2}
    for slot_off in range(0x10, vt.size, 8):
        kind, sym, addend = relocs.get(vt.value + slot_off, (0, 0, 0))
        if kind == R_RELATIVE:
            target = by_addr.get(addend, f"{mod.name}+0x{addend:x}")
        elif kind == R_ABS64:
            target = names.get(mod.symbols[sym].name, mod.symbols[sym].name)
        else:
            target = "(0)"
        print(f"+0x{slot_off - 0x10:03x}  {target}")


if __name__ == "__main__":
    main()
