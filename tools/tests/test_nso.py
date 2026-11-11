#!/usr/bin/env python3
"""Prueba tools/nso.py con módulos sintéticos: dos .so del NDK convertidos a NSO.

main importa nn::fs::OpenFile (lo exporta sdk) y malloc (no lo exporta nadie), y
lleva un svc, un mrs tpidrro_el0, un mrs cntpct_el0 y dos nombres de NVN. Los
segmentos se comprimen con el LZ4 de referencia (lz4.c de https://github.com/lz4/lz4: carpeta lib/ en LZ4_DIR, por defecto third_party/lz4/lib).
"""
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
LZ4_DIR = Path(os.environ.get("LZ4_DIR", TOOLS.parent / "third_party" / "lz4" / "lib"))
NDK_BIN = Path.home() / "Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin"
sys.path.insert(0, str(TOOLS))
import nso  # noqa: E402

MAIN_SRC = r"""
namespace nn { namespace fs { int OpenFile(void** handle, const char* path, int mode); } }
extern "C" void* malloc(unsigned long);
extern "C" __attribute__((visibility("default")))
const char* const kNvnNames[] = {"nvnDeviceInitialize", "nvnQueueSubmitCommands"};
extern "C" __attribute__((visibility("default"))) long nnMain() {
    void* handle;
    unsigned long tls, tick;
    asm volatile("svc #0x26");
    asm volatile("mrs %0, tpidrro_el0" : "=r"(tls));
    asm volatile("mrs %0, cntpct_el0" : "=r"(tick));
    return nn::fs::OpenFile(&handle, kNvnNames[1], 1) + (long)malloc(4) + tls + tick;
}
"""
SDK_SRC = r"""
namespace nn { namespace fs {
__attribute__((visibility("default"))) int OpenFile(void**, const char*, int) { return 0; }
} }
"""
LZ4_HELPER = r"""
#include <stdio.h>
#include "lz4.h"
static char in[1 << 24], out[1 << 25];
int main(void) {
    int n = (int)fread(in, 1, sizeof in, stdin);
    fwrite(out, 1, LZ4_compress_default(in, out, n, (int)sizeof out), stdout);
    return 0;
}
"""


def build_so(tmp: Path, name: str, src: str) -> Path:
    cpp, so = tmp / f"{name}.cpp", tmp / f"{name}.so"
    cpp.write_text(src)
    subprocess.run([NDK_BIN / "clang++", "--target=aarch64-linux-android30", "-O1", "-fPIC", "-shared",
                    "-nostdlib", "-Wl,--no-rosegment,-z,norelro", "-o", so, cpp], check=True)
    return so


def elf_to_nso(so: Path, compress) -> bytes:
    """Arma un NSO con los dos PT_LOAD del .so (RX como text, RW como data)."""
    elf = so.read_bytes()
    phoff, = struct.unpack_from("<Q", elf, 0x20)
    phnum, = struct.unpack_from("<H", elf, 0x38)
    loads, dynamic = [], None
    for k in range(phnum):
        p_type, _, p_off, p_vaddr, _, p_filesz, p_memsz, _ = struct.unpack_from("<IIQQQQQQ", elf, phoff + 56 * k)
        if p_type == 1:
            loads.append((p_vaddr, elf[p_off:p_off + p_filesz], p_memsz))
        elif p_type == 2:
            dynamic = p_vaddr
    assert len(loads) == 2, f"se esperaban 2 PT_LOAD, hay {len(loads)}"
    (text_at, text, _), (data_at, data, data_mem) = loads
    assert text_at == 0
    # MOD0 dentro de la cabecera ELF, que ya no se usa: puntero en +4, cabecera en +8.
    text = bytearray(text)
    struct.pack_into("<I", text, 4, 8)
    struct.pack_into("<4siiiiii", text, 8, b"MOD0", dynamic - 8, 0, 0, 0, 0, 0)
    segments = [(0, bytes(text)), (data_at, b""), (data_at, data)]
    blobs = [compress(raw) for _, raw in segments]
    header = bytearray(0x100)
    header[0:4] = b"NSO0"
    struct.pack_into("<I", header, 0xC, 7)  # los tres segmentos comprimidos
    file_off = len(header)
    for idx, ((mem_off, raw), blob) in enumerate(zip(segments, blobs)):
        struct.pack_into("<III", header, 0x10 + 0x10 * idx, file_off, mem_off, len(raw))
        struct.pack_into("<I", header, 0x60 + 4 * idx, len(blob))
        file_off += len(blob)
    struct.pack_into("<I", header, 0x3C, data_mem - len(data))
    header[0x40:0x48] = bytes.fromhex("123456789abcdef0")
    return bytes(header) + b"".join(blobs)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "lz4blk.c").write_text(LZ4_HELPER)
        lz4blk = tmp / "lz4blk"
        subprocess.run(["cc", "-O2", "-I", LZ4_DIR, "-o", lz4blk, tmp / "lz4blk.c", LZ4_DIR / "lz4.c"],
                       check=True)

        def compress(raw: bytes) -> bytes:
            return subprocess.run([lz4blk], input=raw, capture_output=True, check=True).stdout

        # LZ4: literales largos, copias solapadas (distancia < longitud) y repeticiones lejanas.
        rnd = random.Random(1)
        pieces = [bytes(rnd.getrandbits(8) for _ in range(5000)), b"ab" * 3000, bytes(70000),
                  bytes(range(256)) * 50, b"x"]
        sample = b"".join(rnd.choice(pieces) for _ in range(40))
        assert nso.lz4_block(compress(sample), len(sample)) == sample
        assert nso.lz4_block(compress(b""), 0) == b""

        exefs = tmp / "exefs"
        exefs.mkdir()
        for name, src in (("main", MAIN_SRC), ("sdk", SDK_SRC)):
            (exefs / name).write_bytes(elf_to_nso(build_so(tmp, name, src), compress))
        (exefs / "main.npdm").write_bytes(b"META")

        out = tmp / "analysis"
        mods = nso.analyze(exefs, out)
        main_mod = mods["main"]
        open_file = "_ZN2nn2fs8OpenFileEPPvPKci"
        assert {s.name for s in main_mod.imports} == {open_file, "malloc"}, main_mod.imports
        assert "nnMain" in main_mod.exports
        assert open_file in mods["sdk"].exports
        assert main_mod.build_id == "123456789abcdef0"
        sites = {k: len(v) for k, v in main_mod.patch_sites.items() if v}
        assert sites == {"svc": 1, "mrs tpidrro_el0": 1, "mrs cntpct_el0": 1}, sites
        assert main_mod.svc_numbers() == {0x26: 1}
        assert main_mod.nvn_names == {"nvnDeviceInitialize", "nvnQueueSubmitCommands"}
        assert main_mod.relocs, "sin relocaciones"

        needed = (out / "needed.txt").read_text()
        assert "nn::fs\tnn::fs::OpenFile(void**, char const*, int)\tsdk" in needed, needed
        assert "C (libc y otros)\tmalloc\t—" in needed, needed

        readelf = subprocess.run([NDK_BIN / "llvm-readelf", "-S", "-d", "--dyn-syms", out / "main.elf"],
                                 capture_output=True, text=True, check=True)
        assert "nnMain" in readelf.stdout and open_file in readelf.stdout, readelf.stdout
        assert "warning" not in readelf.stderr, readelf.stderr
        print((out / "SCOPE.md").read_text())
    print("OK")


if __name__ == "__main__":
    main()
