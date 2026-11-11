"""Cobertura del port de Xenoblade: qué pide el juego (imports del SDK y procs de NVN), cuántas veces lo llama y en qué estado está."""
import json, os, re, subprocess, sys, collections
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R + '/tools')
import nso, plt
NM = os.path.expanduser('~/Library/Android/sdk/ndk/28.2.13676358/') + 'toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-nm'
FILT = os.path.expanduser('~/Library/Android/sdk/ndk/28.2.13676358/') + 'toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-cxxfilt'

# tamaños de lo que exporta libnnsdk (4 bytes = solo ret; 8 = devuelve un valor fijo)
sizes = {}
for l in subprocess.run([NM, '-S', '--defined-only', R + '/build/android/libnnsdk.so'], capture_output=True, text=True).stdout.splitlines():
    p = l.split()
    if len(p) == 4: sizes[p[3]] = int(p[1], 16)

# cuerpo de cada función corta de libnnsdk: solo 'ret' = vacía; 'mov w0,#k; ret' = valor fijo
import struct
so = open(R + '/build/android/libnnsdk.so', 'rb').read()
phoff, phnum = struct.unpack_from('<Q', so, 0x20)[0], struct.unpack_from('<H', so, 0x38)[0]
loads = []
for i in range(phnum):
    t, fl, off, va, pa, fsz, msz, al = struct.unpack_from('<IIQQQQQQ', so, phoff + 56 * i)
    if t == 1: loads.append((va, off, fsz))
faddr = {}
for l in subprocess.run([NM, '-S', '--defined-only', R + '/build/android/libnnsdk.so'], capture_output=True, text=True).stdout.splitlines():
    q = l.split()
    if len(q) == 4: faddr[q[3]] = int(q[0], 16)
def body(name):
    if name not in faddr or sizes.get(name, 99) > 12: return 'hecho'
    va = faddr[name]
    for v, o, fs in loads:
        if v <= va < v + fs:
            words = struct.unpack_from('<%dI' % (sizes[name] // 4), so, o + va - v)
            if words == (0xd65f03c0,): return 'vacio'
            if len(words) == 2 and words[1] == 0xd65f03c0 and (words[0] & 0x7f80001f) in (0x52800000, 0x12800000) : return 'fijo'
            return 'hecho'
    return 'hecho'

# sitios de llamada: bl/b a cada entrada PLT de main
mod = nso.Nso('main', open(R + '/work/exefs/main', 'rb').read())
targets = plt.plt_targets(mod)
calls = collections.Counter()
pat = re.compile(r'\tb(?:l)?\t0x([0-9a-f]+)')
for line in open('/tmp/main.dis'):
    m = pat.search(line)
    if m: calls[int(m.group(1), 16)] += 1
sites = collections.Counter()
for addr, name in targets.items(): sites[name] += calls.get(addr, 0)

status_tsv = {}
for l in open(R + '/work/analysis/missing.tsv').read().splitlines()[1:]:
    p = l.split('\t'); status_tsv[p[2]] = p[0]
cats = {}
for l in open(R + '/work/analysis/main.imports.txt').read().splitlines():
    q = l.split('\t')
    if len(q) >= 2: cats[q[1]] = q[0]
dyn = set()
for l in subprocess.run([NM, '-D', '--defined-only', R + '/build/android/libnnsdk.so'], capture_output=True, text=True).stdout.splitlines():
    q = l.split()
    if q: dyn.add(q[-1])
host = set(re.findall(r'S\((\w+)\)', open(R + '/runtime/src/loader/host_symbols.c').read()))
imps = mod.imports
names = [x.name for x in imps]
dem = subprocess.run([FILT, '-n'], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()

def sdk_status(x):
    n = x.name
    for cand in ('nngame_' + n, n):
        if cand in dyn: return body(cand) if cand in faddr else 'hecho'
    if n in host: return 'android'
    if x.bind == 2: return 'datos'
    if x.type == 1: return 'datos'
    return 'falta'
sdk = []
for x, d in zip(imps, dem):
    sdk.append({'g': cats.get(d, 'otros'), 'n': d, 'm': x.name, 'c': sites.get(x.name, 0), 's': sdk_status(x)})

# NVN: procs que pide el juego, sitios de llamada (nvn_uso.tsv) y si están registrados
uso = {}
for l in open(R + '/work/analysis/nvn_uso.tsv'):
    p = l.rstrip('\n').split('\t')
    if len(p) >= 2 and p[0].isdigit(): uso[p[1]] = int(p[0])
registered = set()
import glob
for f in glob.glob(R + '/runtime/src/nnsdk/nvn/*.cpp'):
    registered.update(re.findall(r'NVN_PROC\((nvn\w+)\)', open(f).read()))
APROX = {  # implementados a medias: lo que el Adreno no tiene o aún no se emula
    'nvnCommandBufferBeginTransformFeedback': 'transform feedback: el Adreno no tiene la extensión',
    'nvnCommandBufferEndTransformFeedback': 'transform feedback',
    'nvnCommandBufferPauseTransformFeedback': 'transform feedback',
    'nvnCommandBufferResumeTransformFeedback': 'transform feedback',
    'nvnCommandBufferBindTransformFeedbackBuffer': 'transform feedback',
    'nvnCommandBufferBindTransformFeedbackBuffers': 'transform feedback',
    'nvnCommandBufferDrawTransformFeedback': 'transform feedback',
}
def grupo_nvn(n):
    m = re.match(r'nvn(CommandBuffer|Texture(?:Builder|View)?|Sampler(?:Builder)?|Queue(?:Builder)?|Device(?:Builder)?|Program|MemoryPool(?:Builder)?|Buffer(?:Builder)?|Window(?:Builder)?|Sync|Event(?:Builder)?|BlendState|ChannelMaskState|ColorState|DepthStencilState|MultisampleState|PolygonState|VertexAttribState|VertexStreamState|TexturePool|SamplerPool|Counter|Shader)', n)
    if not m: return 'otros'
    g = m.group(1)
    if g.endswith('State'): return 'estados'
    for base in ('Texture', 'Sampler', 'Queue', 'Device', 'MemoryPool', 'Buffer', 'Window', 'Event'):
        if g.startswith(base): return base
    return g
nvn = []
for n in sorted(set(l.strip() for l in open(R + '/work/analysis/nvn_functions.txt') if l.strip())):
    c = uso.get(n, 0)
    if n in APROX: s = 'aprox'
    elif n in registered: s = body(n)
    else: s = 'falta'
    nvn.append({'g': grupo_nvn(n), 'n': n, 'c': c, 's': s, 'nota': APROX.get(n, '')})

out = {'sdk': sdk, 'nvn': nvn}
json.dump(out, open(R + '/work/cobertura/cobertura.json', 'w'), ensure_ascii=False)
for k in ('sdk', 'nvn'):
    cnt = collections.Counter(x['s'] for x in out[k]); calls_by = collections.Counter()
    for x in out[k]: calls_by[x['s']] += x['c']
    print(k, len(out[k]), dict(cnt), 'llamadas', dict(calls_by))
print('vacíos llamados (nvn):', [(x['n'], x['c']) for x in nvn if x['s'] == 'vacio' and x['c']][:40])
print('vacíos llamados (sdk):', [(x['n'][:60], x['c']) for x in sdk if x['s'] == 'vacio' and x['c']][:40])
print('faltan llamados:', [(x['n'][:60], x['c']) for x in sdk + nvn if x['s'] == 'falta' and x['c']])
