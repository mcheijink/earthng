"""Prebuilt far and horizon tiles (docs/PREBUILT-TILES.md step 1): bake a region's far ring tiles
(without decor) and its 5 km horizon blocks into a small add-on ZIP.

The bake runs this repository's own game Lua (native.lua's buildTile, generator.lua, sweep.lua) under
LuaJIT (pip install lupa) over the pack's own source cells, so a baked layer is the layer the game
would generate. bake.lua packs each one; the game reads them back when the switch is on
(extensions.e2r_world.bakedTiles(true)) and falls back to the generator for anything missing or stale.

Output, beside the region in the country's namespace (the pack itself is not rewritten):

    earthng/countries/<country>/baked/<region>/baked.json      digests and switches of the bake
    earthng/countries/<country>/baked/<region>/<ci>_<cj>.bin    one per 1 km source cell

baked.json holds digests of the region manifest, of the Lua that builds these layers (bake.CODE) and of
every climate block the bake read. The game uses a bake only when all of them match what it runs, so
the planet ZIP must be built from the same commit (earthng/build_mod.py) and the pack must be the one
baked. Rebuild the bake after either changes.

    python earthng/bake_tiles.py --pack <country.zip> --out <baked.zip> [--climate out/climate/<v>]
        [--region <id>] [--layers far,horizon] [--cells 12,40 12,41] [--jobs 4]
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / 'mod'
CLIMATE = '/earthng/planet/climate/'
LAYERS = ('far', 'horizon')


class VFS:
    """The game's virtual file system from ZIPs and folders mounted at '/', the first mount that has
    a file wins. A folder can be mounted under a prefix (the climate blocks)."""

    def __init__(self):
        self.mounts = []
        self.touched = set()

    def mount(self, source, prefix='/', only=None):
        source = Path(source)
        if source.is_dir():
            self.mounts.append((prefix, 'dir', source, only))
        elif zipfile.is_zipfile(source):
            z = zipfile.ZipFile(source)
            names = {'/' + n for n in z.namelist() if not n.endswith('/')}
            self.mounts.append((prefix, 'zip', (z, names), only))
        else:
            raise SystemExit(f'not a folder or ZIP: {source}')

    def _find(self, path):
        for prefix, kind, source, only in self.mounts:
            if not path.startswith(prefix) or (only and not path.startswith(only)):
                continue
            rel = path[len(prefix):]
            if kind == 'dir':
                f = source / rel
                if f.is_file():
                    return kind, f
            elif '/' + rel in source[1]:
                return kind, (source[0], rel)
        return None

    def _note(self, path):
        if path.startswith(CLIMATE):
            self.touched.add(path)

    def exists(self, path):
        if not isinstance(path, str):
            return False
        self._note(path)
        return self._find(path) is not None

    def read(self, path):
        if not isinstance(path, str):
            return None
        self._note(path)
        hit = self._find(path)
        if not hit:
            return None
        kind, where = hit
        return where.read_bytes() if kind == 'dir' else where[0].read(where[1])

    def find(self, root, name):
        out = set()
        for prefix, kind, source, only in self.mounts:
            names = ([prefix + p.relative_to(source).as_posix() for p in source.rglob(name)] if kind == 'dir'
                     else [prefix + n[1:] for n in source[1]])
            out.update(p for p in names if p.startswith(root) and p.rsplit('/', 1)[-1] == name
                       and (not only or p.startswith(only)))
        return sorted(out)


def lua_value(L, value):
    if isinstance(value, dict):
        return L.table_from({k: lua_value(L, v) for k, v in value.items()})
    if isinstance(value, list):
        return L.table_from([lua_value(L, v) for v in value])
    return value


PRELUDE = r'''
package.path = REPO_MOD .. '/lua/?.lua;' .. package.path
local realOpen = io.open
FS = {fileExists = function(_, p) return VFS_EXISTS(p) end,
      findFiles = function(_, root, name) return VFS_FIND(root, name) end}
io.open = function(path, mode)
  if type(path) == 'string' and path:sub(1, 1) == '/' and VFS_EXISTS(path) then
    local data = VFS_READ(path)
    return {read = function() return data end, close = function() end}
  end
  return realOpen(path, mode)
end
jsonReadFile = function(path) return VFS_JSON(path) end
LOGS = {}
log = function(level, tag, message) LOGS[#LOGS + 1] = level .. ' ' .. tostring(message) end
hptimer = function() local t = os.clock(); return {stop = function() return (os.clock() - t) * 1000 end} end
GENERATOR = require('ge/extensions/e2r/generator')
NATIVE = require('ge/extensions/e2r/native')
BAKE = require('ge/extensions/e2r/bake')
CATALOG = require('ge/extensions/e2r/countries')
CELLFORMAT = require('ge/extensions/e2r/cellformat')

function SETTINGS(swept, mode, heights)
  GENERATOR.SWEPT.on = swept
  if mode then GENERATOR.SWEPT.mode = mode end
  GENERATOR.HEIGHTS.auto = heights
end

-- the countries and regions the mounted packs hold: {{country, region}, ...}
function REGIONS()
  local out = {}
  for _, c in ipairs(CATALOG.refresh()) do
    if c.runtime == 'native' then
      for _, d in ipairs(c.regions) do out[#out + 1] = {c.id, d.id} end
    end
  end
  return out
end

-- open a region; opts.useBake: read its bake as the game does (tests), opts.pause: the build's pause
function OPEN(countryId, regionId, opts)
  for _, c in ipairs(CATALOG.refresh()) do
    if c.id == countryId then
      for _, d in ipairs(c.regions) do
        if d.id == regionId then
          local data, settings, dir = NATIVE.bakeOpen(c, d, opts)
          REGION = {data = data, settings = settings, dir = dir, manifest = d.manifest}
          return REGION
        end
      end
    end
  end
  error('no region ' .. tostring(regionId) .. ' of ' .. tostring(countryId) .. ' in the mounted packs')
end

-- the present source cells {{ci, cj}, ...}
function CELLS()
  local d, out = REGION.data, {}
  if d.schemaVersion == 2 then
    local b = d.blockCells
    for key, block in pairs(d.blocks) do
      local bi, bj = key:match('^(-?%d+)_(-?%d+)$')
      bi, bj = tonumber(bi), tonumber(bj)
      if bi and not block.memory then
        for slot = 0, b * b - 1 do
          if CELLFORMAT.present(block.mask, slot) then out[#out + 1] = {bi * b + slot % b, bj * b + math.floor(slot / b)} end
        end
      end
    end
  else
    for key in pairs(d.cells) do
      local i, j = key:match('^(-?%d+)_(-?%d+)$')
      out[#out + 1] = {tonumber(i), tonumber(j)}
    end
  end
  table.sort(out, function(a, b) return a[1] == b[1] and a[2] < b[2] or a[1] < b[1] end)
  return out
end

-- bake one source cell into path: the number of layers, the bytes and the failures
function BAKECELL(ci, cj, far, horizon, path)
  local payloads, count, failed = {}, 0, {}
  local layers = NATIVE.bakeLayers(ci, cj)
  local keys = {}
  for key in pairs(layers) do keys[#keys + 1] = key end
  table.sort(keys)
  for _, key in ipairs(keys) do
    local wanted = layers[key]
    if (wanted.horizon and horizon) or (not wanted.horizon and far) then
      local ok, layer = pcall(NATIVE.bakeLayer, key, wanted)
      if ok and type(layer) == 'table' then
        payloads[wanted.horizon and key or 'far:' .. key] = BAKE.encode({size = wanted.size, step = wanted.step, manifest = layer})
        count = count + 1
      else
        failed[#failed + 1] = key .. ': ' .. tostring(layer)
      end
    end
  end
  if count == 0 then return 0, 0, failed end
  local data = BAKE.file(payloads)
  local f = assert(realOpen(path, 'wb'))
  f:write(data)
  f:close()
  return count, #data, failed
end

-- the digest of a mounted file ('missing' when it is not there)
function DIGEST(path)
  return BAKE.digest(VFS_EXISTS(path) and VFS_READ(path) or nil)
end
'''


def runtime(vfs):
    """A LuaJIT state with the game's Lua over vfs."""
    try:
        import lupa.luajit21 as lupa
    except ImportError:
        raise SystemExit('LuaJIT is needed: pip install lupa (lupa.luajit21)')
    L = lupa.LuaRuntime(unpack_returned_tuples=True)
    g = L.globals()
    g.REPO_MOD = MOD.as_posix()
    g.VFS_EXISTS = vfs.exists
    g.VFS_READ = vfs.read
    g.VFS_FIND = lambda root, name: L.table_from(vfs.find(root, name))

    def read_json(path):
        data = vfs.read(path)
        if data is None:
            raise FileNotFoundError(path)
        return lua_value(L, json.loads(data))
    g.VFS_JSON = read_json
    L.execute(PRELUDE)
    return L


def mounts(packs, climate=None, extra=()):
    """The VFS of a bake: this repository's Lua first (the code that runs is the code digested),
    then the climate blocks, extra mounts and the packs."""
    vfs = VFS()
    vfs.mount(MOD, '/', only='/lua/')
    if climate:
        vfs.mount(climate, CLIMATE)
    for m in extra:
        vfs.mount(m)
    for p in packs:
        vfs.mount(p)
    return vfs


def default_climate():
    try:
        sys.path.insert(0, str(ROOT / 'earthng'))
        import build_climate
        out = build_climate.OUT / build_climate.VERSION
        return out if out.is_dir() else None
    except Exception:
        return None


_WORKER = {}


def _worker_init(packs, climate, extra, country, region, settings):
    vfs = mounts(packs, climate, extra)
    L = runtime(vfs)
    g = L.globals()
    g.SETTINGS(*settings)
    g.OPEN(country, region)
    _WORKER.update(vfs=vfs, L=L)


def _bake_cells(task):
    cells, far, horizon, out_dir = task
    g = _WORKER['L'].globals()
    done = []
    for ci, cj in cells:
        path = Path(out_dir) / f'{ci}_{cj}.bin'
        count, size, failed = g.BAKECELL(ci, cj, far, horizon, str(path))
        done.append((ci, cj, count, size, list(failed.values()) if failed else []))
    return done, sorted(_WORKER['vfs'].touched)


def bake_region(packs, country, region, out_dir, climate=None, extra=(), layers=LAYERS, cells=None, jobs=1,
                swept=True, swept_mode=None, heights=True, report=print):
    """Bake one region into out_dir (the files of its baked/<region>/ folder). Returns baked.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = (swept, swept_mode, heights)
    _worker_init(packs, climate, extra, country, region, settings)
    g = _WORKER['L'].globals()
    info = g.REGION
    present = [(int(c[1]), int(c[2])) for c in g.CELLS().values()]
    if cells:
        wanted = set(cells)
        present = [c for c in present if c in wanted]
    far, horizon = 'far' in layers, 'horizon' in layers
    # tasks of up to 8 cells of one source block, so a worker reads few block files
    by_block = {}
    for ci, cj in present:
        by_block.setdefault((ci // 8, cj // 8), []).append((ci, cj))
    tasks = [(c[k:k + 8], far, horizon, str(out_dir)) for _, c in sorted(by_block.items()) for k in range(0, len(c), 8)]
    started, results, touched = time.time(), [], set(_WORKER['vfs'].touched)
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(jobs, _worker_init, (packs, climate, extra, country, region, settings)) as pool:
            for done, seen in pool.imap_unordered(_bake_cells, tasks):
                results.extend(done)
                touched.update(seen)
                report(f'  {len(results)} / {len(present)} cells')
    else:
        for task in tasks:
            done, seen = _bake_cells(task)
            results.extend(done)
            touched.update(seen)
            report(f'  {len(results)} / {len(present)} cells')
    failures = [f for r in results for f in r[4]]
    for f in failures[:20]:
        report('  failed: ' + f)
    vfs = _WORKER['vfs']
    touched.update(vfs.touched)
    s = info.settings
    baked = {
        'format': g.BAKE.FORMAT, 'version': g.BAKE.VERSION,
        'region': g.DIGEST(info.manifest),
        'cellSize': info.data.cellSize, 'tileSize': info.data.tileSize, 'horizonStep': 100,
        'layers': [x for x in LAYERS if x in layers],
        'settings': {'swept': bool(s.swept), 'heights': bool(s.heights), **({'sweptMode': s.sweptMode} if s.sweptMode else {})},
        'code': {path: g.DIGEST(path) for path in g.BAKE.CODE.values()},
        'climate': {path: g.DIGEST(path) for path in sorted(touched) if path.startswith(CLIMATE)},
        'cells': sum(1 for r in results if r[2]), 'bytes': sum(r[3] for r in results),
        'failed': len(failures), 'seconds': round(time.time() - started, 1),
    }
    (out_dir / 'baked.json').write_text(json.dumps(baked, indent=1), encoding='utf8')
    return baked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pack', action='append', required=True, help='country pack ZIP or folder (repeat for several)')
    ap.add_argument('--out', type=Path, required=True, help='the add-on ZIP to write')
    ap.add_argument('--country', help='country id (default: every native country in the packs)')
    ap.add_argument('--region', action='append', help='region id (default: every region)')
    ap.add_argument('--climate', type=Path, help='climate blocks folder (default: build_climate output, if built)')
    ap.add_argument('--mount', action='append', default=[], help='another ZIP or folder at / (a planet ZIP)')
    ap.add_argument('--layers', default='far,horizon', help='far, horizon or both')
    ap.add_argument('--cells', nargs='*', help='only these source cells, ci,cj')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no-swept', action='store_true', help='bake as with sweptRoads(false)')
    ap.add_argument('--swept-mode', choices=('fast', 'fitted'))
    ap.add_argument('--no-height-pipeline', action='store_true', help='bake as with heightPipeline(false)')
    args = ap.parse_args(argv)
    layers = tuple(x for x in args.layers.split(',') if x)
    if not layers or any(x not in LAYERS for x in layers):
        raise SystemExit('--layers takes far, horizon or far,horizon')
    climate = args.climate or default_climate()
    if not climate and not args.mount:
        print('warning: no climate blocks (--climate or --mount <planet.zip>): the bake will not match a game whose '
              'planet has them, and the game will not use it')
    cells = [tuple(int(v) for v in c.split(',')) for c in args.cells] if args.cells else None
    L = runtime(mounts(args.pack, climate, args.mount))
    regions = [(c, r) for c, r in (tuple(x.values()) for x in L.globals().REGIONS().values())
               if (not args.country or c == args.country) and (not args.region or r in args.region)]
    if not regions:
        raise SystemExit('no matching native region in the packs')
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for country, region in regions:
            print(f'{country}/{region}:')
            folder = Path(tmp) / country / region
            baked = bake_region(args.pack, country, region, folder, climate, args.mount, layers, cells, args.jobs,
                                not args.no_swept, args.swept_mode, not args.no_height_pipeline)
            for f in sorted(folder.iterdir()):
                z.write(f, f'earthng/countries/{country}/baked/{region}/{f.name}')
            print(f'  {baked["cells"]} cells, {baked["bytes"] / 1e6:.1f} MB before deflate, '
                  f'{baked["failed"]} layers failed, {baked["seconds"]} s')
    print(f'wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
