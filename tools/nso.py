#!/usr/bin/env python3
"""Lee los NSO del ExeFS (main, subsdk*, sdk, rtld) y mide qué tiene que dar el runtime.

Uso: python3 tools/nso.py <carpeta_exefs> <carpeta_salida>

Deja en la salida:
  <módulo>.elf        ELF64 AArch64 con segmentos y tabla dinámica, para Ghidra
  <módulo>.imports.txt imports demangleados, con su grupo y quién los exporta
  needed.txt          lo que el juego importa y ningún módulo del juego da (hoy: sdk)
  nvn_functions.txt   funciones NVN que el juego pide por nombre
  SCOPE.md            resumen con los números
"""
from __future__ import annotations

import re
import struct
import subprocess
import sys
from collections import Counter, namedtuple
from pathlib import Path

DT_NULL, DT_NEEDED, DT_PLTRELSZ, DT_HASH, DT_STRTAB, DT_SYMTAB = 0, 1, 2, 4, 5, 6
DT_RELA, DT_RELASZ, DT_STRSZ, DT_JMPREL, DT_GNU_HASH = 7, 8, 10, 23, 0x6FFFFEF5
SHT_PROGBITS, SHT_STRTAB, SHT_DYNAMIC, SHT_NOBITS, SHT_DYNSYM = 1, 3, 6, 8, 11
SHN_ABS = 0xFFF1
STB_GLOBAL, STB_WEAK = 1, 2

# Instrucciones que en Android no hacen lo mismo que en Horizon: (máscara, valor).
PATCH_SITES = {
    "svc": (0xFFE0001F, 0xD4000001),
    "mrs tpidrro_el0": (0xFFFFFFE0, 0xD53BD060),
    "mrs tpidr_el0": (0xFFFFFFE0, 0xD53BD040),
    "msr tpidr_el0": (0xFFFFFFE0, 0xD51BD040),
    "mrs cntpct_el0": (0xFFFFFFE0, 0xD53BE020),
    "mrs cntvct_el0": (0xFFFFFFE0, 0xD53BE040),
    "mrs cntfrq_el0": (0xFFFFFFE0, 0xD53BE000),
}
NVN_NAME = re.compile(rb"(?<![A-Za-z0-9_])(nvn[A-Z][A-Za-z0-9]+)\0")
RUNTIME_MODULES = ("sdk", "rtld")  # los sustituye el runtime; el resto es código del juego

Symbol = namedtuple("Symbol", "name bind type shndx value size")


def lz4_block(src: bytes, size: int) -> bytes:
    """Descomprime un bloque LZ4 crudo (formato de los segmentos NSO)."""
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        dist = src[i] | src[i + 1] << 8
        i += 2
        length = token & 15
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        length += 4
        start = len(dst) - dist
        if dist >= length:
            dst += dst[start:start + length]
        else:
            dst += (dst[start:] * (length // dist + 1))[:length]
    if len(dst) != size:
        raise ValueError(f"LZ4: salieron {len(dst)} bytes, se esperaban {size}")
    return bytes(dst)


def masked_regex(mask: int, value: int) -> re.Pattern:
    """Regex que encuentra (solapadas) las palabras de 32 bits con word & mask == value."""
    parts = []
    for k in range(4):
        m, v = mask >> 8 * k & 0xFF, value >> 8 * k & 0xFF
        allowed = b"".join(re.escape(bytes([b])) for b in range(256) if b & m == v)
        parts.append(b"[" + allowed + b"]")
    return re.compile(b"(?=(" + b"".join(parts) + b"))", re.DOTALL)


PATCH_REGEX = {name: masked_regex(*mv) for name, mv in PATCH_SITES.items()}


class Nso:
    def __init__(self, name: str, data: bytes):
        if data[:4] != b"NSO0":
            raise ValueError(f"{name}: no es un NSO")
        self.name = name
        flags = struct.unpack_from("<I", data, 0xC)[0]
        self.segments = []  # (offset en memoria, tamaño) de text, rodata, data
        image = bytearray()
        for idx, (header, csize_at) in enumerate(((0x10, 0x60), (0x20, 0x64), (0x30, 0x68))):
            file_off, mem_off, size = struct.unpack_from("<III", data, header)
            csize = struct.unpack_from("<I", data, csize_at)[0]
            raw = data[file_off:file_off + csize]
            segment = lz4_block(raw, size) if flags >> idx & 1 else raw[:size]
            if len(image) < mem_off + size:
                image += bytes(mem_off + size - len(image))
            image[mem_off:mem_off + size] = segment
            self.segments.append((mem_off, size))
        self.bss_size = struct.unpack_from("<I", data, 0x3C)[0]
        self.build_id = data[0x40:0x60].rstrip(b"\0").hex()
        self.image = image
        self._parse_dynamic()
        self.patch_sites = self._scan_text()
        self.nvn_names = {m.group(1).decode() for m in NVN_NAME.finditer(image)}

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.image, off)[0]

    def cstr(self, off: int) -> str:
        return self.image[off:self.image.index(0, off)].decode("utf-8", "replace")

    def _parse_dynamic(self):
        img = self.image
        mod0 = self.u32(4)
        if img[mod0:mod0 + 4] != b"MOD0":
            raise ValueError(f"{self.name}: no encuentro la cabecera MOD0")
        self.dynamic_off = mod0 + struct.unpack_from("<i", img, mod0 + 4)[0]
        tags = {}
        off = self.dynamic_off
        while True:
            tag, val = struct.unpack_from("<qQ", img, off)
            off += 16
            if tag == DT_NULL:
                break
            tags.setdefault(tag, []).append(val)
        self.dynamic_size = off - self.dynamic_off
        strtab, symtab = tags[DT_STRTAB][0], tags[DT_SYMTAB][0]
        self.strtab, self.strtab_size, self.symtab = strtab, tags[DT_STRSZ][0], symtab
        self.needed = [self.cstr(strtab + v) for v in tags.get(DT_NEEDED, [])]

        if DT_HASH in tags:
            count = self.u32(tags[DT_HASH][0] + 4)
        elif DT_GNU_HASH in tags:
            count = self._gnu_hash_count(tags[DT_GNU_HASH][0])
        else:
            count = (strtab - symtab) // 24
        self.symbols = []
        for k in range(count):
            name, info, _, shndx, value, size = struct.unpack_from("<IBBHQQ", img, symtab + 24 * k)
            self.symbols.append(Symbol(self.cstr(strtab + name), info >> 4, info & 15, shndx, value, size))
        self.imports = [s for s in self.symbols if s.name and s.shndx == 0]
        self.exports = {s.name for s in self.symbols
                        if s.name and s.shndx != 0 and s.bind in (STB_GLOBAL, STB_WEAK)}

        self.relocs = Counter()  # tipo de relocación -> cuántas
        for start_tag, size_tag in ((DT_RELA, DT_RELASZ), (DT_JMPREL, DT_PLTRELSZ)):
            if start_tag in tags:
                start = tags[start_tag][0]
                for off in range(start, start + tags[size_tag][0], 24):
                    self.relocs[struct.unpack_from("<Q", img, off + 8)[0] & 0xFFFFFFFF] += 1

    def _gnu_hash_count(self, off: int) -> int:
        nbuckets, symoffset, bloom_size, _ = struct.unpack_from("<IIII", self.image, off)
        buckets_off = off + 16 + bloom_size * 8
        last = max(struct.unpack_from(f"<{nbuckets}I", self.image, buckets_off))
        if last < symoffset:
            return symoffset
        chain_off = buckets_off + nbuckets * 4
        while not self.u32(chain_off + (last - symoffset) * 4) & 1:
            last += 1
        return last + 1

    def _scan_text(self) -> dict:
        start, size = self.segments[0]
        text = bytes(self.image[start:start + size])
        sites = {}
        for name, regex in PATCH_REGEX.items():
            sites[name] = [start + m.start() for m in regex.finditer(text) if m.start() % 4 == 0]
        return sites

    def svc_numbers(self) -> Counter:
        return Counter(self.u32(addr) >> 5 & 0xFFFF for addr in self.patch_sites["svc"])

    def write_elf(self, path: Path):
        base = 0x1000
        phdrs = []
        for (off, size), flags in zip(self.segments, (5, 4, 6)):  # R-X, R--, RW-
            memsz = size + (self.bss_size if flags == 6 else 0)
            phdrs.append(struct.pack("<IIQQQQQQ", 1, flags, base + off, off, off, size, memsz, 0x1000))
        d, n = self.dynamic_off, self.dynamic_size
        phdrs.append(struct.pack("<IIQQQQQQ", 2, 6, base + d, d, d, n, n, 8))
        # Secciones para text/rodata/data/bss y la parte dinámica: así readelf, objdump y Ghidra
        # ven el código y los símbolos.
        names = b"\0.text\0.rodata\0.data\0.bss\0.dynamic\0.dynsym\0.dynstr\0.shstrtab\0"
        first_global = next((k for k, s in enumerate(self.symbols) if s.bind), len(self.symbols))
        shstrtab_off = base + len(self.image)
        shdrs_off = (shstrtab_off + len(names) + 7) & ~7
        (t_off, t_size), (r_off, r_size), (d_off, d_size) = self.segments
        bss_off = d_off + d_size
        sections = [
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (names.index(b".text"), SHT_PROGBITS, 6, t_off, base + t_off, t_size, 0, 0, 16, 0),
            (names.index(b".rodata"), SHT_PROGBITS, 2, r_off, base + r_off, r_size, 0, 0, 16, 0),
            (names.index(b".data"), SHT_PROGBITS, 3, d_off, base + d_off, d_size, 0, 0, 16, 0),
            (names.index(b".bss"), SHT_NOBITS, 3, bss_off, base + bss_off, self.bss_size, 0, 0, 16, 0),
            (names.index(b".dynamic"), SHT_DYNAMIC, 3, d, base + d, n, 7, 0, 8, 16),
            (names.index(b".dynsym"), SHT_DYNSYM, 2, self.symtab, base + self.symtab,
             24 * len(self.symbols), 7, first_global, 8, 24),
            (names.index(b".dynstr"), SHT_STRTAB, 2, self.strtab, base + self.strtab, self.strtab_size, 0, 0, 1, 0),
            (names.index(b".shstrtab"), SHT_STRTAB, 0, 0, shstrtab_off, len(names), 0, 0, 1, 0),
        ]
        ident = b"\x7fELF\x02\x01\x01" + bytes(9)
        header = struct.pack("<16sHHIQQQIHHHHHH", ident, 3, 183, 1, 0, 64, shdrs_off, 0, 64, 56, len(phdrs),
                             64, len(sections), len(sections) - 1)
        out = bytearray(header + b"".join(phdrs))
        out += bytes(base - len(out))
        image = bytearray(self.image)
        # Los st_shndx originales apuntan a secciones que el NSO no conserva: se reapuntan a las nuestras.
        for k, s in enumerate(self.symbols):
            if s.shndx and s.shndx != SHN_ABS:
                shndx = 1 if s.value < r_off else 2 if s.value < d_off else 3 if s.value < bss_off else 4
                struct.pack_into("<H", image, self.symtab + 24 * k + 6, shndx)
        out += image + names
        out += bytes(shdrs_off - len(out))
        out += b"".join(struct.pack("<IIQQQQIIQQ", *s) for s in sections)
        path.write_bytes(out)


def demangle(names) -> dict:
    names = sorted(set(names))
    if not names:
        return {}
    # -n: el c++filt de macOS quita un "_" inicial por defecto (Mach-O) y rompe _Z...
    result = subprocess.run(["c++filt", "-n"], input="\n".join(names),
                            capture_output=True, text=True, check=True)
    return dict(zip(names, result.stdout.splitlines()))


def group_of(raw: str, pretty: str) -> str:
    if raw.startswith("_Z"):
        m = re.match(r"(nn::\w+)", pretty)
        if m:
            return m.group(1)
        if pretty.startswith("std::"):
            return "std (libc++)"
        m = re.match(r"(\w+)::", pretty)
        return f"{m.group(1)}::" if m else "C++ global"
    m = re.match(r"([a-z]+)[A-Z_]", raw)
    return f"{m.group(1)}*" if m else "C (libc y otros)"


def module_order(name: str):
    if name == "main":
        return (0, name)
    if name.startswith("subsdk"):
        return (1, name)
    return (2 if name == "sdk" else 3, name)


def mb(n: int) -> str:
    return f"{n / 1048576:.1f} MB"


def analyze(exefs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mods = {}
    for path in sorted(exefs.iterdir(), key=lambda p: module_order(p.name)):
        if not path.is_file():
            continue
        with path.open("rb") as f:
            if f.read(4) != b"NSO0":
                continue
        mods[path.name] = Nso(path.name, path.read_bytes())
    if "main" not in mods:
        raise SystemExit(f"No hay NSO 'main' en {exefs}")

    providers = {}  # símbolo -> primer módulo que lo exporta, en orden de carga
    for mod in mods.values():
        for name in mod.exports:
            providers.setdefault(name, mod.name)
    pretty = demangle(s.name for mod in mods.values() for s in mod.imports)
    game = [m for m in mods.values() if m.name not in RUNTIME_MODULES]
    game_names = {m.name for m in game}

    for mod in mods.values():
        mod.write_elf(out / f"{mod.name}.elf")
        rows = sorted((group_of(s.name, pretty[s.name]), pretty[s.name], providers.get(s.name, "—"))
                      for s in mod.imports)
        (out / f"{mod.name}.imports.txt").write_text("".join(f"{g}\t{p}\t{src}\n" for g, p, src in rows))

    needed = {}  # lo que el runtime tiene que dar: imports del juego que no resuelve el juego
    for mod in game:
        for s in mod.imports:
            if providers.get(s.name) not in game_names:
                needed[s.name] = providers.get(s.name, "—")
    groups = Counter(group_of(raw, pretty[raw]) for raw in needed)
    unresolved = sorted(pretty[raw] for raw, src in needed.items() if src == "—")
    rows = sorted((group_of(raw, pretty[raw]), pretty[raw], src) for raw, src in needed.items())
    (out / "needed.txt").write_text("".join(f"{g}\t{p}\t{src}\n" for g, p, src in rows))
    nvn = sorted(set().union(*(m.nvn_names for m in game)))
    (out / "nvn_functions.txt").write_text("".join(f"{n}\n" for n in nvn))

    lines = [
        "# Alcance medido",
        "",
        "Generado por `tools/nso.py` a partir de tu ExeFS.",
        "",
        "| módulo | build id | .text | .rodata | .data | .bss | importa | exporta |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for m in mods.values():
        (_, t), (_, r), (_, d) = m.segments
        lines.append(f"| {m.name} | `{m.build_id[:16]}` | {mb(t)} | {mb(r)} | {mb(d)} | {mb(m.bss_size)} "
                     f"| {len(m.imports)} | {len(m.exports)} |")
    lines += [
        "",
        f"## El runtime tiene que dar {len(needed)} funciones",
        "",
        "Imports de " + ", ".join(sorted(game_names)) + " que no resuelve ningún módulo del juego "
        "(hoy los da `sdk`). Lista completa en `needed.txt`.",
        "",
        "| grupo | funciones |",
        "|---|--:|",
    ]
    lines += [f"| {g} | {c} |" for g, c in groups.most_common()]
    lines += [
        "",
        f"## NVN: {len(nvn)} funciones pedidas por nombre",
        "",
        "Lista en `nvn_functions.txt`. Es la superficie de la capa NVN → Vulkan.",
        "",
        "## Instrucciones a parchear al cargar (solo código del juego)",
        "",
        "| módulo | " + " | ".join(PATCH_SITES) + " |",
        "|---|" + "--:|" * len(PATCH_SITES),
    ]
    for m in game:
        lines.append(f"| {m.name} | " + " | ".join(str(len(m.patch_sites[k])) for k in PATCH_SITES) + " |")
    svcs = sum((m.svc_numbers() for m in game), Counter())
    if svcs:
        lines += ["", "SVC usados por el juego: " + ", ".join(f"0x{n:X}×{c}" for n, c in sorted(svcs.items()))]
    lines += ["", f"## Imports que no exporta ningún módulo: {len(unresolved)}", ""]
    lines += [f"- `{p}`" for p in unresolved[:50]]
    if len(unresolved) > 50:
        lines.append(f"- … y {len(unresolved) - 50} más (en `needed.txt`, proveedor «—»)")
    (out / "SCOPE.md").write_text("\n".join(lines) + "\n")
    return mods


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    analyze(Path(sys.argv[1]), Path(sys.argv[2]))
    print((Path(sys.argv[2]) / "SCOPE.md").read_text())
