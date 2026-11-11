#!/usr/bin/env python3
"""Traduce direcciones de la PLT de un NSO al import al que saltan.

Uso: python3 tools/plt.py <nso> <dirección_hex>...
     python3 tools/plt.py <nso> --all          (toda la tabla, para anotar desensamblados)

Cada entrada de la PLT es `adrp x16, página; ldr x17, [x16, #off]; add x16, …; br x17`:
se lee el slot del GOT de esas dos instrucciones y se busca su JUMP_SLOT.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nso  # noqa: E402

DT_JMPREL, DT_PLTRELSZ = 23, 2


def plt_targets(mod: nso.Nso) -> dict:
    """{dirección de la entrada PLT: nombre del import}."""
    img = mod.image
    tags = {}
    off = mod.dynamic_off
    while True:
        tag, val = struct.unpack_from("<qQ", img, off)
        off += 16
        if tag == 0:
            break
        tags.setdefault(tag, val)
    slot_to_name = {}
    for o in range(tags[DT_JMPREL], tags[DT_JMPREL] + tags[DT_PLTRELSZ], 24):
        r_off, info, _ = struct.unpack_from("<QQq", img, o)
        slot_to_name[r_off] = mod.symbols[info >> 32].name

    start, size = mod.segments[0]
    result = {}
    for pc in range(start, start + size - 12, 4):
        adrp, ldr, add, br = struct.unpack_from("<IIII", img, pc)
        # adrp x16 / ldr x17, [x16, #imm] / add x16, x16, #imm / br x17
        if adrp & 0x9F00001F != 0x90000010 or ldr & 0xFFC003FF != 0xF9400211 or br != 0xD61F0220:
            continue
        immlo, immhi = adrp >> 29 & 3, adrp >> 5 & 0x7FFFF
        page = (pc & ~0xFFF) + (((immhi << 2 | immlo) << 12) if immhi < 0x40000 else ((immhi << 2 | immlo) - (1 << 21)) << 12)
        slot = page + (ldr >> 10 & 0xFFF) * 8
        if slot in slot_to_name:
            result[pc] = slot_to_name[slot]
    return result


if __name__ == "__main__":
    mod = nso.Nso(Path(sys.argv[1]).name, Path(sys.argv[1]).read_bytes())
    targets = plt_targets(mod)
    pretty = nso.demangle(targets.values())
    wanted = targets if sys.argv[2:] == ["--all"] else {int(a, 16): None for a in sys.argv[2:]}
    for addr in sorted(wanted):
        print(f"{addr:x}\t{pretty.get(targets.get(addr), '?') if addr in targets else '(no es PLT)'}")
