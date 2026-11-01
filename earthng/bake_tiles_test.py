"""Tests of the prebuilt far and horizon tiles (bake.lua, native.lua BAKED, bake_tiles.py).

No local data: a synthetic pack of 3 x 3 source cells is written to a temporary folder in the
pack's own format (country_format), baked with the game's Lua under LuaJIT, and read back through
the game's path. Needs lupa (pip install lupa). Run: python earthng/bake_tiles_test.py
"""
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import lupa.luajit21  # noqa: F401
    HAVE_LUPA = True
except ImportError:
    HAVE_LUPA = False

import country_format as cf  # noqa: E402

COUNTRY, REGION = 'zz', 'synth'
NS = f'earthng/countries/{COUNTRY}/'
# metres to degrees at the equator (a synthetic place, no real one)
DEG = 1 / 111320.


def z_at(x, y):
    return 30 + 8 * math.sin(x / 230) + 6 * math.cos(y / 270 + 1) + 2 * math.sin((x + y) / 120)


def synthetic_cell(ci, cj):
    x0, y0 = ci * 1000., cj * 1000.
    n = 101
    heights = [round(z_at(x0 + 10 * ix, y0 + 10 * iy), 1) for iy in range(n) for ix in range(n)]
    cell = {'terrain': {'nx': n, 'ny': n, 'heights': heights}, 'roads': [], 'junctions': [], 'buildings': [],
            'areas': [], 'lines': [], 'points': []}

    def pt(x, y):
        return [x, y, round(z_at(x, y), 2)]
    if cj == 1:   # an east-west road through the middle row
        cell['roads'].append({'id': f'r{ci}', 'width': 7., 'tags': {'highway': 'secondary', 'lanes': '2'},
                              'nodes': [f'e{ci}', f'e{ci + 1}'],
                              'points': [pt(x0 + 50 * k, 1500.) for k in range(21)]})
    if ci == 1:   # a north-south track through the middle column
        cell['roads'].append({'id': f't{cj}', 'width': 3., 'tags': {'highway': 'track'},
                              'nodes': [f'n{cj}', f'n{cj + 1}'],
                              'points': [pt(1500., y0 + 100 * k) for k in range(11)]})
    ring = [pt(x0 + 100, y0 + 100), pt(x0 + 400, y0 + 100), pt(x0 + 400, y0 + 400), pt(x0 + 100, y0 + 400), pt(x0 + 100, y0 + 100)]
    cell['areas'].append({'id': f'a{ci}_{cj}', 'kind': 'farmland', 'tags': {'landuse': 'farmland'}, 'rings': [ring]})
    return cell


def town_cell(ci, cj):
    """generator_budget's dense synthetic town (600 houses, a motorway, 93 streets) moved to cell ci, cj."""
    import generator_budget as gb
    cell, _ = gb.synthetic_cell(10, random.Random(1))
    dx, dy, tag = ci * 1000., cj * 1000., f'_{ci}_{cj}'

    def move(p):
        return [p[0] + dx, p[1] + dy] + list(p[2:])
    out = {'terrain': {k: cell['terrain'][k] for k in ('nx', 'ny', 'heights')}}
    for kind in ('roads', 'junctions', 'buildings', 'areas', 'lines', 'points'):
        out[kind] = []
        for f in cell.get(kind, []):
            f = dict(f, id=str(f['id']) + tag)
            if 'nodes' in f:
                f['nodes'] = [str(n) + tag for n in f['nodes']]
            if 'points' in f:
                f['points'] = [move(p) for p in f['points']]
            if 'rings' in f:
                f['rings'] = [[move(p) for p in ring] for ring in f['rings']]
            if 'pos' in f:
                f['pos'] = move(f['pos'])
            out[kind].append(f)
    return out


def write_pack(root, make_cell=synthetic_cell):
    """The synthetic country pack as a folder (what the ZIP holds), 3 x 3 cells of 1 km."""
    root = Path(root)
    payloads = {cf.slot(ci, cj): cf.encode_cell(make_cell(ci, cj), (ci * 1000., cj * 1000.))
                for ci in range(3) for cj in range(3)}
    block, mask = cf.encode_block(payloads)
    (root / NS / 'blocks').mkdir(parents=True)
    (root / NS / 'regions').mkdir(parents=True)
    (root / NS / 'blocks/0_0.bin').write_bytes(block)
    box = [0., 0., 3000 * DEG, 3000 * DEG]
    region = {'schemaVersion': 2, 'id': REGION, 'runtime': 'lua', 'sourceFormat': 'earthng-features-v2',
              'tileSize': 500, 'cellSize': 1000, 'bounds': box, 'geo': [0, 0, 0, 0, DEG, 0, 0, DEG],
              'blockCells': 8, 'blocks': {'0_0': {'mask': mask, 'path': f'/{NS}blocks/0_0.bin'}},
              'coverage': [box], 'landing': [1250, 1250], 'cellCount': 9}
    (root / NS / f'regions/{REGION}.json').write_text(json.dumps(region), encoding='utf8')
    country = {'schemaVersion': 1, 'kind': 'earthng-country', 'id': COUNTRY, 'title': 'Synthetic', 'version': '1',
               'runtime': 'native', 'sourceFormat': 'earthng-features-v2', 'coverage': [box],
               'regions': [{'id': REGION, 'manifest': f'regions/{REGION}.json', 'coverage': [box]}]}
    (root / NS / 'manifest.json').write_text(json.dumps(country), encoding='utf8')
    return root


# Lua helpers of the tests: build a layer generated and baked, compare them
CHECK = r'''
function LAYER(key, wanted, useBake, yielding)
  local yields = 0
  OPEN(TEST_COUNTRY, TEST_REGION, {useBake = useBake, pause = yielding and function() yields = yields + 1; coroutine.yield() end or nil})
  local co = coroutine.create(function() return NATIVE.bakeLayer(key, wanted) end)
  local ok, layer
  repeat ok, layer = coroutine.resume(co) until not ok or coroutine.status(co) == 'dead'
  if not ok then error(layer) end
  local st = NATIVE.status().stream
  return layer, st.baked, yields
end
-- the largest differences of two manifests, or an error when their structure differs
function COMPARE(a, b)
  local worst = {v = 0, n = 0, u = 0, items = #a.items}
  assert(#a.items == #b.items, 'item count')
  assert(a.triangles == b.triangles, 'triangles')
  for k, it in ipairs(a.items) do
    local bt = b.items[k]
    assert(it.name == bt.name and it.kind == bt.kind and it.triangles == bt.triangles and it.layer == bt.layer, 'item ' .. k)
    for c = 1, 3 do assert(it.pos[c] == bt.pos[c], 'item position') end
    assert(#it.meshes == #bt.meshes, 'mesh count')
    for m, mesh in ipairs(it.meshes) do
      local bm = bt.meshes[m]
      assert(mesh.material == bm.material, 'material')
      assert(#mesh.verts == #bm.verts and #mesh.normals == #bm.normals and #mesh.uvs == #bm.uvs and #mesh.faces == #bm.faces, 'mesh sizes')
      for q, p in ipairs(mesh.verts) do local o = bm.verts[q]; worst.v = math.max(worst.v, math.abs(o.x - p.x), math.abs(o.y - p.y), math.abs(o.z - p.z)) end
      for q, p in ipairs(mesh.normals) do local o = bm.normals[q]; worst.n = math.max(worst.n, math.abs(o.x - p.x), math.abs(o.y - p.y), math.abs(o.z - p.z)) end
      for q, p in ipairs(mesh.uvs) do local o = bm.uvs[q]; worst.u = math.max(worst.u, math.abs(o.u - p.u), math.abs(o.v - p.v)) end
      for q, f in ipairs(mesh.faces) do local o = bm.faces[q]; assert(f.v == o.v and f.n == o.n and f.u == o.u, 'face') end
    end
  end
  return worst
end
'''

FAR = ('1_2', {'i': 1, 'j': 2, 'far': True, 'decor': False, 'size': 500})
H1000 = ('h1000:1_1', {'i': 1, 'j': 1, 'horizon': True, 'far': True, 'decor': False, 'size': 1000, 'step': 100})
H500 = ('h500:3_2', {'i': 3, 'j': 2, 'horizon': True, 'far': True, 'decor': False, 'size': 500, 'step': 100})


@unittest.skipUnless(HAVE_LUPA, 'needs lupa (pip install lupa)')
class Baked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import bake_tiles
        cls.bt = bake_tiles
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pack = write_pack(Path(cls.tmp.name) / 'pack')
        cls.addon = Path(cls.tmp.name) / 'addon'
        cls.baked_dir = cls.addon / NS / 'baked' / REGION
        cls.baked = bake_tiles.bake_region([cls.pack], COUNTRY, REGION, cls.baked_dir, report=lambda m: None)
        cls.pristine = {f.name: f.read_bytes() for f in cls.baked_dir.iterdir()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # every test starts from the bake as written
        for f in self.baked_dir.iterdir():
            f.unlink()
        for name, data in self.pristine.items():
            (self.baked_dir / name).write_bytes(data)

    def runtime(self, on=True, missing=()):
        vfs = self.bt.mounts([self.pack, self.addon], missing=missing)
        L = self.bt.runtime(vfs)
        g = L.globals()
        g.TEST_COUNTRY, g.TEST_REGION = COUNTRY, REGION
        L.execute(CHECK)
        g.NATIVE.BAKED.on = on
        return L, g

    def lua_wanted(self, L, wanted):
        return self.bt.lua_value(L, wanted)

    def test_bake_writes_every_cell_and_its_digests(self):
        b = self.baked
        self.assertEqual(b['cells'], 9)
        self.assertEqual(b['failed'], 0)
        self.assertEqual(b['format'], 'earthng-baked-v1')
        self.assertEqual(sorted(b['layers']), ['far', 'horizon'])
        self.assertTrue(all(':' in d for d in b['code'].values()) and 'missing' not in b['code'].values())
        self.assertEqual(len(b['code']), 8)
        # the stock shapes and decal maps the generator asked for, taken as there
        self.assertTrue(b['game'] and all(p.startswith(('/assets/', '/levels/', '/art/')) for p in b['game']))
        L, g = self.runtime()
        # 4 far tiles, 4 horizon quadrants and one 1 km horizon block in a 1 km cell
        data = (self.baked_dir / '1_1.bin').read_bytes()
        keys = sorted(g.BAKE.index(data).keys())
        self.assertEqual(keys, sorted(['far:2_2', 'far:2_3', 'far:3_2', 'far:3_3', 'h500:2_2', 'h500:2_3', 'h500:3_2',
                                       'h500:3_3', 'h1000:1_1']))

    def test_baked_layers_match_the_generator(self):
        L, g = self.runtime()
        for key, wanted in (FAR, H1000, H500):
            w = self.lua_wanted(L, wanted)
            generated, _, _ = g.LAYER(key, w, False, False)
            baked, info, yields = g.LAYER(key, w, True, True)
            self.assertEqual(info.used, 1, key)
            self.assertGreater(yields, 0, key)          # the read yields, as it must in the game's coroutine
            worst = g.COMPARE(generated, baked)
            self.assertGreater(worst['items'], 0)
            self.assertLessEqual(worst['v'], .0005 + 1e-9, key)    # mm
            self.assertLessEqual(worst['n'], .00005 + 1e-9, key)   # 1e-4
            self.assertLessEqual(worst['u'], 1 / 8192 + 1e-9, key)  # 1/4096 of a repeat

    def test_switch_off_decor_and_vista_use_the_generator(self):
        L, g = self.runtime(on=False)
        _, info, _ = g.LAYER(FAR[0], self.lua_wanted(L, FAR[1]), True, False)
        self.assertIsNone(info)
        g.NATIVE.BAKED.on = True
        decor = dict(FAR[1], decor=True)
        _, info, _ = g.LAYER(FAR[0], self.lua_wanted(L, decor), True, False)
        self.assertIsNone(info)
        vista = dict(H1000[1], vista=True)
        _, info, _ = g.LAYER('v1000:1_1', self.lua_wanted(L, vista), True, False)
        self.assertIsNone(info)

    def off_reason(self, L, g):
        layer, info, _ = g.LAYER(FAR[0], self.lua_wanted(L, FAR[1]), True, False)
        self.assertGreater(len(layer['items']), 0)        # the generator built it instead
        return info.off

    def test_stale_code_is_not_used(self):
        b = json.loads((self.baked_dir / 'baked.json').read_text())
        path = '/lua/ge/extensions/e2r/generator.lua'
        b['code'][path] = '00000000:1'
        (self.baked_dir / 'baked.json').write_text(json.dumps(b))
        L, g = self.runtime()
        self.assertIn('another ' + path, self.off_reason(L, g))

    def test_a_game_without_a_file_the_bake_took_as_there_is_not_used(self):
        path = self.baked['game'][0]
        L, g = self.runtime(missing=[path])
        self.assertIn('the game lacks ' + path, self.off_reason(L, g))

    def test_another_export_of_the_region_is_not_used(self):
        b = json.loads((self.baked_dir / 'baked.json').read_text())
        b['region'] = '00000000:1'
        (self.baked_dir / 'baked.json').write_text(json.dumps(b))
        L, g = self.runtime()
        self.assertIn('another export', self.off_reason(L, g))

    def test_other_switches_are_not_used(self):
        L, g = self.runtime()
        g.GENERATOR.SWEPT.on = not g.GENERATOR.SWEPT.on
        self.assertIn('swept', self.off_reason(L, g))
        g.GENERATOR.SWEPT.on = not g.GENERATOR.SWEPT.on
        # a switch changed after the landing's check: that layer is generated
        swept = L.eval(r'''function(key, wanted)
          OPEN(TEST_COUNTRY, TEST_REGION, {useBake = true})
          NATIVE.bakeLayer(key, wanted)
          GENERATOR.SWEPT.on = not GENERATOR.SWEPT.on
          NATIVE.bakeLayer(key, wanted)
          GENERATOR.SWEPT.on = not GENERATOR.SWEPT.on
          local info = NATIVE.status().stream.baked
          return info.used, info.switched
        end''')
        used, switched = swept(FAR[0], self.lua_wanted(L, FAR[1]))
        self.assertEqual(used, 1)
        self.assertIn('swept', switched)
        b = json.loads((self.baked_dir / 'baked.json').read_text())
        b['settings']['heights'] = not b['settings']['heights']
        (self.baked_dir / 'baked.json').write_text(json.dumps(b))
        self.assertIn('height', self.off_reason(L, g))

    def test_missing_or_broken_cells_fall_back_per_layer(self):
        (self.baked_dir / '1_2.bin').unlink()
        L, g = self.runtime()
        # 0xff bytes in the far tile 1_1's own payload (cell 0_0): an endless varint
        broken = bytearray((self.baked_dir / '0_0.bin').read_bytes())
        first = int(g.BAKE.index(bytes(broken))['far:1_1'][1]) - 1
        broken[first + 10:first + 50] = b'\xff' * 40
        (self.baked_dir / '0_0.bin').write_bytes(bytes(broken))
        layer, info, _ = g.LAYER('2_4', self.lua_wanted(L, dict(FAR[1], i=2, j=4)), True, False)   # cell 1_2: no file
        self.assertGreater(len(layer['items']), 0)
        self.assertEqual(info.missing, 1)
        layer, info, _ = g.LAYER('1_1', self.lua_wanted(L, dict(FAR[1], i=1, j=1)), True, False)   # cell 0_0: broken
        self.assertGreater(len(layer['items']), 0)
        self.assertEqual(info.failed, 1)
        self.assertTrue(any('not read' in v for v in g.LOGS.values()))
        layer, info, _ = g.LAYER(H1000[0], self.lua_wanted(L, H1000[1]), True, False)   # cell 1_1: intact
        self.assertEqual(info.used, 1)

    def test_no_bake_leaves_the_pack_as_it_was(self):
        shutil.rmtree(self.baked_dir)
        try:
            L, g = self.runtime()
            self.assertIn('no bake', self.off_reason(L, g))
        finally:
            self.baked_dir.mkdir(parents=True)

    def test_streaming_reads_far_and_horizon_from_the_bake(self):
        """The game's own loop: land(), then update() frames at 30 m/s until nothing is queued."""
        L, g = self.runtime()
        stream = L.eval(r'''function(lat, lon)
          NATIVE.refresh()
          NATIVE.setTune({nearM = 400, farM = 800})
          local job = NATIVE.land({seq = 1, lat = lat, lon = lon})
          assert(job.state == 'working', tostring(job.message))
          local st
          for frame = 1, 5000 do
            -- fast enough that the far ring wants no decor (native.lua priorities, farFast)
            NATIVE.pose({x = 1250, y = 1250, vx = 30, vy = 0})
            NATIVE.update(1 / 60, 8)
            st = NATIVE.status().stream
            if frame > 10 and st.queued == 0 and st.building == 0 then break end
          end
          local far, near = 0, 0
          for key in pairs(st.far) do far = far + 1 end
          for key in pairs(st.ready) do near = near + 1 end
          local horizon = 0
          for key in pairs(st.horizon) do horizon = horizon + 1 end
          return {baked = st.baked, far = far, near = near, horizon = horizon, queued = st.queued,
            failed = next(st.failed) and tostring(next(st.failed)) or nil}
        end''')
        out = stream(1250 * DEG, 1250 * DEG)
        if os.environ.get("BAKE_TEST_VERBOSE"):
            print(dict(near=out.near, far=out.far, horizon=out.horizon, used=out.baked.used))
        self.assertEqual(out.queued, 0)
        self.assertIsNone(out.failed)
        self.assertGreater(out.near, 0)
        self.assertGreater(out.far, 0)
        self.assertGreater(out.horizon, 0)
        self.assertIsNone(out.baked.off)
        self.assertEqual(out.baked.failed, 0)
        # every far tile and horizon block of the pack came from the bake
        self.assertEqual(out.baked.used, out.far + out.horizon)

    def test_codec_edges(self):
        L, g = self.runtime()
        roundtrip = L.eval(r'''function()
          local v = {a = 1.25, b = -7, c = 'x', d = {1, 2, 'x'}, e = true, f = false, big = 2^40 + .5, tiny = 1e-300,
            m = {verts = {{x = 1.0004, y = -2, z = 3}}, normals = {{x = 0, y = 0, z = 1}}, uvs = {{u = .5, v = -.25}},
              faces = {{v = 0, n = 0, u = 0}, {v = 0, n = 0, u = 0}}, material = 'e2rf_ground'}}
          local data = BAKE.encode(v)
          local back = BAKE.decode(data, 1, #data)
          assert(back.a == 1.25 and back.b == -7 and back.c == 'x' and back.d[3] == 'x' and back.e == true and back.f == false)
          assert(back.big == 2^40 + .5 and back.tiny == 1e-300)
          assert(back.m.material == 'e2rf_ground' and back.m.verts[1].x == 1 and back.m.uvs[1].v == -.25)
          assert(back.m.faces[1] == back.m.faces[2])          -- a corner seen again shares its face
          assert(not pcall(BAKE.decode, data, 1, #data - 1))  -- truncated
          assert(not pcall(BAKE.encode, {x = 0 / 0}))         -- NaN is not stored
          local file = BAKE.file({k1 = 'abc', k2 = data})
          local index = BAKE.index(file)
          assert(file:sub(index.k1[1], index.k1[2]) == 'abc')
          assert(not pcall(BAKE.index, file .. 'x'))
          assert(BAKE.digest('a\r\nb') == BAKE.digest('a\nb') and BAKE.digest('a\nb') ~= BAKE.digest('a\nc'))
          return true
        end''')
        self.assertTrue(roundtrip())


@unittest.skipUnless(HAVE_LUPA, 'needs lupa (pip install lupa)')
class FarInputs(unittest.TestCase):
    """The premise of baking the far ring and the horizon, on a dense synthetic town (generator_budget):
    without decor they read no quality preset and no landing point. They do read the game install
    (which stock shapes and decal maps are there), which is why a bake lists those files. Each case
    runs in a fresh LuaJIT state: the generator keeps what it found on disk for the session."""

    def digest(self, layer, extra, files):
        import generator_budget as gb
        L, _ = gb.lua_runtime()
        g = L.globals()
        g.REPO_ROOT = str(Path(__file__).resolve().parent.parent)
        g.FILES = files
        L.execute("package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path")
        L.execute("FS = {fileExists = function() return FILES end}")
        L.execute("GEN = require('ge/extensions/e2r/generator'); BAKE = require('ge/extensions/e2r/bake')")
        cell, _ = gb.synthetic_cell(10, random.Random(1))
        build = L.eval(r'''function(cells, opts)
          local _, _, far = GEN.build(cells, opts.i, opts.j, opts)
          return BAKE.digest(BAKE.encode(far))
        end''')
        base = {'tileSize': 500, 'regionId': 's', 'materials': 'm', 'packs': 'p', 'farStep': 100, 'horizonStep': 100,
                'latitude': 0, 'seaLevel': 0, 'country': 'zz', 'heightPipeline': True, 'farDecor': False}
        return build(gb.to_lua(L, [cell]), gb.to_lua(L, {**base, **layer, **extra}))

    def test_far_and_horizon_ignore_tune_and_landing_but_not_the_game_files(self):
        tuned = {'maxTrees': 10, 'maxFarTrees': 5, 'detail': 10, 'forestSpacing': 30, 'groundSpacing': 40,
                 'propSpacing': 99, 'parkedCars': 0, 'maxGrass': 1, 'maxRocks': 1, 'center': [5000, -3000]}
        for layer in ({'far': True, 'i': 0, 'j': 0}, {'far': True, 'horizon': True, 'i': 0, 'j': 0, 'tileSize': 1000}):
            full = self.digest(layer, {}, True)
            self.assertEqual(self.digest(layer, tuned, True), full, layer)
            self.assertNotEqual(self.digest(layer, {}, False), full, layer)


def bench(repeat=3):
    """Time per layer through native.lua's buildTile: generated (source cells decoded, generator.build)
    against read from the bake, on the light pack and on the dense town. Run: python
    earthng/bake_tiles_test.py --bench. The machine and the JIT differ from the game: read ratios."""
    import time
    import bake_tiles as bt
    # the region is opened once a mode, as a landing does: the bake's digests are checked by the first
    # baked layer (timed apart) and each cell file is read once for its nine layers
    timed = r'''function(key, wanted)
      collectgarbage(); collectgarbage('stop')
      local before = collectgarbage('count')
      local t = os.clock()
      local layer = NATIVE.bakeLayer(key, wanted)
      local ms = (os.clock() - t) * 1000
      local kb = collectgarbage('count') - before
      collectgarbage('restart')
      return ms, kb, layer.triangles
    end'''
    for name, make in (('light', synthetic_cell), ('town', town_cell)):
        with tempfile.TemporaryDirectory() as tmp:
            pack = write_pack(Path(tmp) / 'pack', make)
            addon = Path(tmp) / 'addon'
            t = time.time()
            info = bt.bake_region([pack], COUNTRY, REGION, addon / NS / 'baked' / REGION, report=lambda m: None)
            import zlib
            deflated = sum(len(zlib.compress(f.read_bytes(), 6)) for f in (addon / NS / 'baked' / REGION).glob('*.bin'))
            print(f'{name}: bake {time.time() - t:.1f} s for 9 cells; a cell {info["bytes"] / 9 / 1024:.0f} kB, '
                  f'{deflated / 9 / 1024:.0f} kB deflated')
            L = bt.runtime(bt.mounts([pack, addon]))
            g = L.globals()
            g.TEST_COUNTRY, g.TEST_REGION = COUNTRY, REGION
            L.execute(CHECK)
            g.NATIVE.BAKED.on = True
            run = L.eval(timed)
            best = {}
            for use in (False, True):
                g.OPEN(COUNTRY, REGION, bt.lua_value(L, {'useBake': use}))
                if use:
                    t = time.time()
                    run('h500:0_0', bt.lua_value(L, dict(H500[1], i=0, j=0)))
                    print(f'  the bake\'s digest check, once a landing: {(time.time() - t) * 1000:.0f} ms')
                for key, wanted in (FAR, H1000, H500):
                    w = bt.lua_value(L, wanted)
                    for _ in range(repeat):
                        ms, kb, tris = run(key, w)
                        if (key, use) not in best or ms < best[key, use][0]:
                            best[key, use] = (ms, kb, tris)
            for key, wanted in (FAR, H1000, H500):
                (g_ms, g_kb, tris), (b_ms, b_kb, _) = best[key, False], best[key, True]
                print(f'  {key:10} {int(tris):6} tris: generated {g_ms:6.1f} ms {g_kb / 1024:5.1f} MB garbage, '
                      f'baked {b_ms:5.1f} ms {b_kb / 1024:4.1f} MB ({g_ms / b_ms:.0f}x)')


if __name__ == '__main__':
    if '--bench' in sys.argv:
        bench()
    else:
        unittest.main()
