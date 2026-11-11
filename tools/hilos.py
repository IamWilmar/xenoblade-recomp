#!/usr/bin/env python3
"""Resume la última instantánea de hilos de un log del runtime (<datos>/hilos o el watchdog): por hilo, pc, lr y
los primeros frames de la cadena de x29, en una línea.

Uso: python3 tools/hilos.py <log.txt> [filtro]
"""
import re
import sys

text = open(sys.argv[1], errors="replace").read()
blocks = re.split(r"\n(?=hilo ')", text)
last = {}
for b in blocks:
    m = re.match(r"hilo '([^']*)' \(tid (\d+)\)", b)
    if not m:
        continue
    pc = re.search(r"^pc (\S+)", b, re.M)
    lr = re.search(r"^lr (\S+)", b, re.M)
    frames = re.findall(r"^  #\d+\s+(\S+)", b, re.M)[:4]
    last[m.group(2)] = (m.group(1), pc.group(1) if pc else "?", lr.group(1) if lr else "?", frames)
flt = sys.argv[2] if len(sys.argv) > 2 else ""
for tid, (name, pc, lr, frames) in sorted(last.items(), key=lambda kv: kv[1][0]):
    if flt in name:
        print(f"{name:18} {tid:>6}  pc {pc:28} lr {lr:28} | {' < '.join(frames)}")
