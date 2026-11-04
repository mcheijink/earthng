"""Shores and quays of the Lua tile generator (generator.lua ROADS.seaQuads, ROADS.seaWater, ROADS.quayAt) on a
synthetic coast, under LuaJIT through lupa: the water follows the shore exactly, the shaped ground crosses
the plane once, and a road along deep water gets a quay wall. Run: python earthng/shore_test.py"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import lupa.luajit21 as lupa
except ImportError:  # pragma: no cover
    lupa = None

PLANE = .3            # the sea level 0 plus SEA_ABOVE
ROAD_Y, ROAD_Z, ROAD_W = 300., 2.5, 7.


def dem(x, y):
    """Deep flat sea in the south, a beach rising through the sea level with a ripple in the west, a
    steep bank under a harbour road in the east, land in the north."""
    if y < 150:
        return -2.
    if x < 250:
        ripple = .12 * math.sin(x / 7.) * math.cos(y / 9.)
        return -1. + (min(y, 270.) - 150.) / 120. * 2.6 + ripple + max(0., y - 270.) * .02
    if y < 285:
        return -2.
    return ROAD_Z - .1


PRELUDE = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
local function flat(m, out)
  for _, it in ipairs(m.items) do
    if it.kind == 'proc' then
      for k, mesh in ipairs(it.meshes) do
        local mat = mesh.material or it.mats[k]
        for f = 1, #mesh.faces, 3 do
          local t = {mat}
          for c = 0, 2 do
            local v = mesh.verts[mesh.faces[f + c].v + 1]
            t[#t + 1] = v.x + it.pos[1]; t[#t + 1] = v.y + it.pos[2]; t[#t + 1] = v.z + it.pos[3]
          end
          out[#out + 1] = t
        end
      end
    end
  end
  return out
end
function BUILD(cells)
  local opts = {tileSize = 500, regionId = 'test', far = false, horizon = false, farStep = 100, horizonStep = 100,
    latitude = 53.5, seaLevel = 0, country = 'germany', heightPipeline = true, detail = 50,
    maxTrees = 0, maxGrass = 0, maxRocks = 0, maxFarTrees = 0, parkedCars = 1, maxCars = 0}
  local road, scene = GENERATOR.build(cells, 0, 0, opts)
  return flat(road, {}), flat(scene, {})
end
'''


def cell():
    n = 101
    heights = [round(dem(ix * 10., iy * 10.), 2) for iy in range(n) for ix in range(n)]
    road = {'id': '9', 'width': ROAD_W, 'tags': {'highway': 'tertiary'}, 'nodes': ['a', 'b'],
            'points': [[250., ROAD_Y, ROAD_Z], [600., ROAD_Y, ROAD_Z], [1000., ROAD_Y, ROAD_Z]]}
    return {'terrain': {'rect': [0., 0., 1000., 1000.], 'nx': n, 'ny': n, 'heights': heights},
            'roads': [road], 'junctions': [], 'buildings': [], 'areas': [], 'lines': [], 'points': []}


GROUND = ('e2rf_ground', 'e2rf_forest', 'e2rf_meadow', 'e2rf_park', 'e2rf_beach', 'e2rf_bare_ground')


def inside(tri, x, y):
    (ax, ay), (bx, by), (cx, cy) = [(p[0], p[1]) for p in tri]
    d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(d) < 1e-12:
        return None
    l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / d
    l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / d
    l3 = 1 - l1 - l2
    if min(l1, l2, l3) < -1e-9:
        return None
    return l1 * tri[0][2] + l2 * tri[1][2] + l3 * tri[2][2]


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Shore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = lupa.LuaRuntime()
        L.globals().REPO_ROOT = ROOT
        L.execute(PRELUDE)

        def lua(v):
            if isinstance(v, dict):
                return L.table_from({k: lua(x) for k, x in v.items()})
            if isinstance(v, (list, tuple)):
                return L.table_from([lua(x) for x in v])
            return v
        road, scene = L.globals().BUILD(lua([cell()]))

        def tris(t):
            out = []
            for tri in t.values():
                v = list(tri.values())
                out.append((v[0], [tuple(v[1:4]), tuple(v[4:7]), tuple(v[7:10])]))
            return out
        cls.road, cls.scene = tris(road), tris(scene)
        cls.ground = [p for m, p in cls.road if m.startswith(GROUND)]
        cls.water = [p for m, p in cls.scene if m == 'e2rf_water']
        # ground triangles by 10 m bin for lookups
        cls.bins = {}
        for p in cls.ground:
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            for bx in range(int(min(xs) // 10), int(max(xs) // 10) + 1):
                for by in range(int(min(ys) // 10), int(max(ys) // 10) + 1):
                    cls.bins.setdefault((bx, by), []).append(p)

    def ground_at(self, x, y):
        for p in self.bins.get((int(x // 10), int(y // 10)), ()):
            z = inside(p, x, y)
            if z is not None:
                return z
        return None

    def wet(self, x, y):
        return any(inside(p, x, y) is not None for p in self.water)

    def test_water_only_over_ground_under_the_plane(self):
        self.assertTrue(self.water)
        for p in self.water:
            for q in p:
                self.assertAlmostEqual(q[2], PLANE, places=4)
                g = self.ground_at(q[0], q[1])
                if g is not None:
                    self.assertLessEqual(g, PLANE + .01, q)

    def test_water_edge_follows_the_shore(self):
        # on the beach, every point with ground clearly under the plane is water and none clearly over it
        misses = 0
        for x in range(12, 240, 4):
            for y in range(150, 272, 2):
                g = self.ground_at(x + .5, y + .5)
                if g is None or abs(g - PLANE) < .03:
                    continue
                if (g < PLANE) != self.wet(x + .5, y + .5):
                    misses += 1
        self.assertEqual(misses, 0)

    def test_ground_crosses_the_plane_once(self):
        for x in (15.5, 55.5, 105.5, 155.5, 205.5, 235.5):
            signs = []
            for y in range(140, 290):
                g = self.ground_at(x, y + .5)
                if g is not None:
                    signs.append(g > PLANE)
            crossings = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
            self.assertEqual(crossings, 1, x)

    def test_no_ground_skims_the_plane_on_the_beach(self):
        # inside the tile, away from its border and the road, no lattice node lies within the shaped band
        for p in self.ground:
            for q in p:
                if 5 < q[0] < 240 and 5 < q[1] < 290:
                    self.assertFalse(PLANE - .5 + .01 < q[2] < PLANE + .15 - .01, q)

    def test_quay_wall_along_the_harbour_road(self):
        walls = [p for m, p in self.scene if m == 'e2rf_plinth' and all(255 <= q[0] <= 500 and 270 <= q[1] < ROAD_Y for q in p)]
        self.assertTrue(walls, 'a quay wall')
        zs = [q[2] for p in walls for q in p]
        self.assertAlmostEqual(min(zs), PLANE - 1, places=3)
        self.assertAlmostEqual(max(zs), ROAD_Z + .1 - .02, places=3)
        for p in walls:   # upright
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = p
            nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            self.assertLess(abs(nz), 1e-6)
        tops = [p for m, p in self.scene if m == 'e2rf_sidewalk' and all(255 <= q[0] <= 500 and 270 <= q[1] < ROAD_Y for q in p)]
        self.assertTrue(tops, 'a quay top')
        self.assertAlmostEqual(max(q[1] for p in tops for q in p), ROAD_Y - ROAD_W / 2, places=3)
        # the water reaches the wall
        foot = min(q[1] for p in walls for q in p)
        self.assertTrue(self.wet(400., foot - .5))

    def test_beach_has_no_quay(self):
        self.assertFalse([p for m, p in self.scene if m == 'e2rf_plinth' and all(q[0] < 240 for q in p)])


if __name__ == '__main__':
    unittest.main()
