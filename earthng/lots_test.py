"""Car park bay lines and bus stop markings (mod/lua/ge/extensions/e2r/lots.lua through generator.lua),
the generator under LuaJIT through lupa. Run: python earthng/lots_test.py"""
import math
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import lupa.luajit21 as lupa
except ImportError:  # pragma: no cover
    lupa = None

Z, LIFT = 40., .1
ROAD_Y, HW = 250., 3.5
LOT = (100., 320., 160., 360.)          # a car park 60 x 40 m north of the street
SHED = (100., 340., 112., 360.)         # a building in its north-west corner
STOP = (300., ROAD_Y + 4.)             # a bus stop on the north side of the street

PRELUDE = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
function BUILD(cells, statics)
  local opts = {tileSize = 500, regionId = 'test', far = false, horizon = false, farStep = 100, horizonStep = 100,
    latitude = 52.5, seaLevel = -100, country = 'germany', heightPipeline = true, lots = true, detail = 50,
    look = {statics = statics}, maxTrees = 0, maxGrass = 0, maxRocks = 0, maxFarTrees = 0, parkedCars = 1, maxCars = 300}
  local _, scene = GENERATOR.build(cells, 0, 0, opts)
  local out = {}
  for _, it in ipairs(scene.items) do
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
  local spots = {}
  for _, s in ipairs(scene.parking or {}) do spots[#spots + 1] = {s[1], s[2]} end
  return out, spots
end
'''


def ring(x0, y0, x1, y1):
    return [[x0, y0, Z], [x1, y0, Z], [x1, y1, Z], [x0, y1, Z], [x0, y0, Z]]


def cell():
    n = 101
    road = {'id': '31', 'width': 2 * HW, 'tags': {'highway': 'residential'}, 'nodes': ['a', 'b'],
            'points': [[0., ROAD_Y, Z], [250., ROAD_Y, Z], [500., ROAD_Y, Z]]}
    lot = {'id': 'a1', 'kind': 'paved', 'tags': {'amenity': 'parking'}, 'rings': [ring(*LOT)]}
    shed = {'id': 'b1', 'kind': 'yes', 'height': 3., 'minHeight': 0., 'tags': {'building': 'yes'}, 'rings': [ring(*SHED)]}
    stop = {'id': 's1', 'kind': 'stop', 'tags': {'highway': 'bus_stop'}, 'pos': [STOP[0], STOP[1], Z]}
    return {'terrain': {'rect': [0., 0., 1000., 1000.], 'nx': n, 'ny': n, 'heights': [Z] * (n * n)},
            'roads': [road], 'junctions': [], 'buildings': [shed], 'areas': [lot], 'lines': [], 'points': [stop]}


def inside(box, x, y, m=0.):
    return box[0] - m <= x <= box[2] + m and box[1] - m <= y <= box[3] + m


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Lots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = lupa.LuaRuntime(unpack_returned_tuples=True)
        L.globals().REPO_ROOT = ROOT
        L.execute(PRELUDE)

        def lua(v):
            if isinstance(v, dict):
                return L.table_from({k: lua(x) for k, x in v.items()})
            if isinstance(v, (list, tuple)):
                return L.table_from([lua(x) for x in v])
            return v

        def build(statics):
            tris, spots = L.globals().BUILD(lua([cell()]), statics)
            out = []
            for t in tris.values():
                v = list(t.values())
                out.append((v[0], [tuple(v[1 + 3 * k:4 + 3 * k]) for k in range(3)]))
            return out, [tuple(s.values()) for s in spots.values()]
        cls.tris, cls.spots = build('street')
        cls.sparse, _ = build('sparse')

    def marks(self, tris, box, m=0.):
        return [p for mat, p in tris if mat == 'e2rf_mark' and all(inside(box, q[0], q[1], m) for q in p)]

    def test_bay_lines(self):
        lines = self.marks(self.tris, LOT, .2)
        self.assertTrue(lines)
        for p in lines:
            for q in p:
                self.assertAlmostEqual(q[2], Z - .25 + .033, delta=.02)    # the drawn ground (groundDrop) + drape + paint
                self.assertFalse(inside(SHED, q[0], q[1], -.2), q)
        # the lines stand every 2.7 m across the rows, and every parked car's spot is between two of them
        xs = sorted({round(sum(q[0] for q in p) / 3, 0) for p in lines})
        self.assertGreater(len(xs), 8)
        spots = [s for s in self.spots if inside(LOT, s[0], s[1])]
        self.assertTrue(spots)
        for x, y in spots:
            near = [p for p in lines if abs(sum(q[1] for q in p) / 3 - y) < 2.6]
            left = [p for p in near if abs(sum(q[0] for q in p) / 3 - (x - 1.35)) < .2]
            right = [p for p in near if abs(sum(q[0] for q in p) / 3 - (x + 1.35)) < .2]
            self.assertTrue(left and right, (x, y))

    def test_bus_stop_zigzag(self):
        box = (STOP[0] - 12, ROAD_Y, STOP[0] + 12, ROAD_Y + HW)
        legs = self.marks(self.tris, box, .2)
        self.assertEqual(len(legs), 2 * 12)                # 12 legs of 1.5 m, two triangles each
        for p in legs:
            for q in p:
                self.assertAlmostEqual(q[2], Z + LIFT + .008, places=4)
                self.assertGreater(q[1], ROAD_Y + HW - 1.25 - .15)   # within the lane's outer 1.25 m
                self.assertLess(q[1], ROAD_Y + HW - .25 + .15)
        # nothing on the far side of the street
        self.assertFalse(self.marks(self.tris, (STOP[0] - 12, ROAD_Y - HW, STOP[0] + 12, ROAD_Y - 1)))

    def test_without_lamps_the_stop_is_still_painted(self):
        box = (STOP[0] - 12, ROAD_Y, STOP[0] + 12, ROAD_Y + HW)
        self.assertEqual(len(self.marks(self.sparse, box, .2)), 2 * 12)


if __name__ == '__main__':
    unittest.main()
