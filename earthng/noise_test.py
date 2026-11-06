"""Tests of noise barriers: the exporter's line kind (fast_style) and the generator's placement and walls
(noise.lua, run under LuaJIT with lupa; skipped without it). Run: python earthng/noise_test.py

The synthetic cell has a dual carriageway for keep-right traffic: eastbound at y = 100, westbound at
y = 114 (its median on each carriageway's left), both 11 m paved. Barriers stand outside it, on its hard
shoulder, in its median, across it, on a bridge's deck, and in an older pack's shape (kind 'fence').
"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))
import fast_style  # noqa: E402
import looks  # noqa: E402


def setUpModule():
    fast_style.NOISE = True        # noise barriers are their own kind only with EARTHNG_NOISE


def tearDownModule():
    fast_style.NOISE = False

EAST_Y, WEST_Y, PAVED = 100., 114., 11.
CLEAR = 1.5
SOUTH_EDGE = EAST_Y - PAVED / 2 - CLEAR        # 93: the eastbound's outer corridor edge
NORTH_EDGE = WEST_Y + PAVED / 2 + CLEAR        # 121: the westbound's


def terrain_z(x, y):
    return 30 + .006 * x + .004 * y + .8 * math.sin(x / 110) * math.cos(y / 140)


class Exporter(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(fast_style.line_kind({'barrier': 'noise_barrier'}), 'noise_barrier')
        self.assertEqual(fast_style.line_kind({'barrier': 'wall', 'wall': 'noise_barrier'}), 'noise_barrier')
        self.assertEqual(fast_style.line_kind({'barrier': 'wall'}), 'fence')
        self.assertEqual(fast_style.line_kind({'barrier': 'fence'}), 'fence')
        self.assertTrue(looks.keep_scenery('noise_barrier', {'barriers': True}))
        self.assertFalse(looks.keep_scenery('noise_barrier', {'barriers': False}))


def lua_runtime():
    for name in ('luajit21', 'luajit20'):
        try:
            return __import__('lupa.' + name, fromlist=[name]).LuaRuntime(unpack_returned_tuples=True)
        except ImportError:
            pass
    return None


LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
ROADS3 = require('ge/extensions/e2r/roads3')
function BUILD(cells, i, j, options) return {GENERATOR.build(cells, i, j, options)} end
function SWEPT(bytes, version)
  return ROADS3.merge({ROADS3.cell(bytes, 1, #bytes, 0, 0, version, function() end)}, nil)
end
-- {x, y, z} of every vertex of the triangles drawn in material, from the road and scene manifests
function VERTS(result, material)
  local out = {}
  for layer = 1, 3 do
    for _, item in ipairs(result[layer] and result[layer].items or {}) do
      for m, mesh in ipairs(item.meshes or {}) do
        if ((item.mats or {})[m] or mesh.material) == material then
          local o = item.pos or {0, 0, 0}
          for _, f in ipairs(mesh.faces) do
            local v = mesh.verts[f.v + 1]
            out[#out + 1] = v.x + o[1]; out[#out + 1] = v.y + o[2]; out[#out + 1] = v.z + (o[3] or 0)
          end
        end
      end
    end
  end
  return out
end
'''


@unittest.skipIf(lua_runtime() is None, 'needs lupa with LuaJIT')
class Generator(unittest.TestCase):
    swept = False

    @classmethod
    def setUpClass(cls):
        L = cls.L = lua_runtime()
        L.globals().REPO_ROOT = ROOT
        L.execute(LUA)
        cls.cell, cls.carriageways = cls.make_cell()
        cls.roads3 = None
        if cls.swept:
            import generator_budget
            data, version = generator_budget.swept_bytes(cls.carriageways)
            cls.roads3 = L.globals().SWEPT(data, version)

    @classmethod
    def make_cell(cls):
        grid = 10
        n = 1000 // grid + 1
        heights = [round(terrain_z(ix * grid, iy * grid), 1) for iy in range(n) for ix in range(n)]

        def pt(x, y, dz=0.):
            return [x, y, round(terrain_z(x, y) + dz, 2)]

        mtags = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes'}
        east = {'id': '2001', 'width': PAVED, 'tags': dict(mtags), 'nodes': ['e0', 'e1'],
                'points': [pt(x, EAST_Y) for x in range(0, 1001, 20)]}
        west = {'id': '2002', 'width': PAVED, 'tags': dict(mtags), 'nodes': ['w0', 'w1'],
                'points': [pt(x, WEST_Y) for x in range(1000, -1, -20)]}
        bridge = {'id': '2003', 'width': 8., 'tags': {'highway': 'secondary', 'bridge': 'yes', 'layer': '1'},
                  'nodes': ['b0', 'b1'], 'points': [pt(x, 300., 6.) for x in range(180, 421, 20)]}
        roads = [east, west, bridge]

        def barrier(fid, xy, kind='noise_barrier', **tags):
            tags = tags or {'barrier': 'noise_barrier'}
            return {'id': fid, 'kind': kind, 'tags': tags, 'points': [pt(x, y) for x, y in xy]}
        lines = [
            barrier('outside', [(100, 91), (300, 91)]),
            barrier('shoulder', [(400, 96), (600, 96)]),
            barrier('median', [(700, 106), (900, 106)]),
            barrier('across', [(950, 60), (950, 140)]),
            barrier('tall', [(100, 125), (300, 125)], barrier='noise_barrier', height='5.5', material='wood'),
            barrier('oldpack', [(400, 118), (480, 118)], kind='fence', barrier='wall', wall='noise_barrier'),
            barrier('deck', [(220, 304.5), (380, 304.5)]),
        ]
        cell = {'terrain': {'rect': [0., 0., 1000., 1000.], 'nx': n, 'ny': n, 'heights': heights}, 'roads': roads,
                'junctions': [], 'buildings': [], 'areas': [], 'lines': lines, 'points': []}
        return cell, [east, west]

    def to_lua(self, value):
        if isinstance(value, dict):
            return self.L.table_from({k: self.to_lua(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return self.L.table_from([self.to_lua(v) for v in value])
        return value

    def build(self, i, j, far=False):
        options = self.to_lua({'tileSize': 500, 'regionId': 'test', 'far': far, 'horizon': False, 'farDecor': far,
                               'farStep': 100, 'latitude': 52.5, 'seaLevel': 0, 'country': 'de', 'heightPipeline': True})
        if self.roads3 is not None:
            options.roads3 = self.roads3
        return self.L.globals().BUILD(self.to_lua([self.cell]), i, j, options)

    def verts(self, result, material, x0, x1, y0, y1):
        flat = list(self.L.globals().VERTS(result, 'e2rf_' + material).values())
        pts = [flat[k:k + 3] for k in range(0, len(flat), 3)]
        return [p for p in pts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]

    def centre_y(self, pts, x0, x1):
        # the wall's middle between its two faces, over the run's inner part (its ends ease)
        inner = [p for p in pts if x0 <= p[0] <= x1]
        self.assertTrue(inner)
        ys = [p[1] for p in inner]
        return (min(ys) + max(ys)) / 2, max(ys) - min(ys)

    def test_outside_stays(self):
        near = self.build(0, 0)
        y, spread = self.centre_y(self.verts(near, 'wall_precast', 100, 300, 80, 99), 120, 280)
        self.assertAlmostEqual(y, 91, delta=.05)
        self.assertLess(spread, .3)

    def paved(self, result, x0, x1):
        # the drawn carriageways' outer edges (a swept road's width is the exporter's, not the way's; its
        # meshes go whole to the tile owning their chunk, so both tiles are read)
        ys = [p[1] for r in (self.build(0, 0), self.build(1, 0)) for m in ('road', 'road_major')
              for p in self.verts(r, m, x0, x1, 80, 135)]
        south = min(y for y in ys if y < (EAST_Y + WEST_Y) / 2)
        north = max(y for y in ys if y > (EAST_Y + WEST_Y) / 2)
        return south, north

    def test_shoulder_moves_out(self):
        near = self.build(0, 0)
        y, spread = self.centre_y(self.verts(near, 'wall_precast', 400, 500, 85, 99), 420, 500)
        self.assertAlmostEqual(y, self.paved(near, 420, 480)[0] - CLEAR, delta=.1)
        self.assertLess(spread, .3)

    def test_median_leaves_the_median(self):
        near = self.build(1, 0)
        pts = self.verts(near, 'wall_precast', 700, 900, 80, 130)
        y, _ = self.centre_y(pts, 720, 880)
        self.assertAlmostEqual(y, self.paved(near, 720, 880)[0] - CLEAR, delta=.1)
        self.assertFalse([p for p in pts if EAST_Y + PAVED / 2 < p[1] < WEST_Y - PAVED / 2])

    def test_across_leaves_a_gap(self):
        near = self.build(1, 0)
        pts = self.verts(near, 'wall_precast', 945, 955, 0, 200)
        self.assertTrue([p for p in pts if p[1] < SOUTH_EDGE])
        self.assertTrue([p for p in pts if p[1] > NORTH_EDGE])
        self.assertFalse([p for p in pts if EAST_Y - PAVED / 2 < p[1] < WEST_Y + PAVED / 2])

    def test_tagged_height_and_material(self):
        near = self.build(0, 0)
        pts = self.verts(near, 'hw_boards', 100, 300, 120, 130)
        # top over foot at one place: the tagged height and the .4 m the foot sinks into the ground
        spans = {}
        for x, y, z in pts:
            key = (round(x, 2), round(y, 2))
            lo, hi = spans.get(key, (z, z))
            spans[key] = (min(lo, z), max(hi, z))
        heights = [hi - lo for lo, hi in spans.values() if hi - lo > 1]
        self.assertTrue(heights)
        for h in heights:
            self.assertAlmostEqual(h, 5.5 + .4, delta=.01)

    def test_old_pack_wall_is_a_noise_barrier(self):
        near = self.build(0, 0)
        y, _ = self.centre_y(self.verts(near, 'wall_precast', 400, 480, 110, 130), 420, 460)
        self.assertAlmostEqual(y, self.paved(near, 420, 460)[1] + CLEAR, delta=.1)
        # and not also drawn as the fence's concrete wall
        self.assertFalse(self.verts(near, 'fence_wall', 400, 480, 110, 130))

    def test_on_the_deck(self):
        near = self.build(0, 0)
        pts = [p for p in self.verts(near, 'wall_precast', 240, 360, 300, 310)]
        self.assertTrue(pts)
        deck = terrain_z(300, 300) + 6 + .10
        low = min(p[2] for p in pts if 280 <= p[0] <= 320)
        self.assertAlmostEqual(low, deck - .4, delta=.3)

    def test_tiles_meet(self):
        a, b = self.build(0, 0), self.build(1, 0)
        def edge(result):
            return sorted({(round(p[1], 3), round(p[2], 2)) for p in self.verts(result, 'wall_precast', 499.999, 500.001, 85, 99)})
        self.assertTrue(edge(a))
        self.assertEqual(edge(a), edge(b))

    def test_triangles_per_km(self):
        near, far = self.build(0, 0), self.build(0, 0, far=True)
        wall = len(self.verts(near, 'wall_precast', 100, 300, 80, 99)) / 3
        posts = len(self.verts(near, 'clad_metal', 100, 300, 80, 99)) / 3
        wall_far = len(self.verts(far, 'wall_precast', 100, 300, 80, 99)) / 3
        print('\nnoise barrier triangles per km: near %.0f (walls %.0f, posts %.0f), far %.0f'
              % ((wall + posts) / .2, wall / .2, posts / .2, wall_far / .2))
        self.assertLess((wall + posts) / .2, 3000)
        self.assertEqual(len(self.verts(far, 'clad_metal', 100, 300, 80, 99)), 0)


class Swept(Generator):
    """The same, with the two carriageways as swept roads (sweep.lua)."""
    swept = True


if __name__ == '__main__':
    unittest.main()
