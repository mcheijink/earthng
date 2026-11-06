"""Buildings of the Lua tile generator: passages over roads, half-hipped and skillion roofs, the north
look (mod/lua/ge/extensions/e2r/generator.lua, under LuaJIT through lupa).
Run: python earthng/building_test.py"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))

try:
    import lupa.luajit21 as lupa
except ImportError:  # pragma: no cover
    lupa = None

Z = 40.0
LANE = 11.0          # the primary road's carriageway width
ROAD_Y = 250.0

PRELUDE = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
-- every triangle of the scene (or far) layer: {material, x1, y1, z1, x2, y2, z2, x3, y3, z3}
function TRIANGLES(cells, far, latitude, look)
  local opts = {tileSize = 500, regionId = 'test', far = far, horizon = false, farDecor = far, farStep = 100,
    horizonStep = 100, latitude = latitude, seaLevel = 0, country = 'germany', heightPipeline = true, detail = 50,
    look = look, maxTrees = 0, maxGrass = 0, maxRocks = 0, maxFarTrees = 0, parkedCars = 1, maxCars = 0}
  local _, scene, farm = GENERATOR.build(cells, 0, 0, opts)
  local m = far and farm or scene
  local out = {}
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
'''


def box(x0, y0, x1, y1):
    return [[x0, y0, Z], [x1, y0, Z], [x1, y1, Z], [x0, y1, Z], [x0, y0, Z]]


def building(bid, ring, tags, height):
    return {'id': bid, 'kind': tags.get('building', 'yes'), 'height': height, 'minHeight': 0., 'tags': tags, 'rings': [ring]}


def cell(buildings):
    n = 51
    terrain = {'rect': [0., 0., 1000., 1000.], 'nx': n, 'ny': n, 'heights': [Z] * (n * n)}
    road = {'id': '7', 'width': LANE, 'tags': {'highway': 'primary', 'lanes': '2'}, 'nodes': ['a', 'b'],
            'points': [[0., ROAD_Y, Z], [500., ROAD_Y, Z], [1000., ROAD_Y, Z]]}
    return {'terrain': terrain, 'roads': [road], 'junctions': [], 'buildings': buildings, 'areas': [], 'lines': [], 'points': []}


WALLS = ('e2rf_wall_', 'e2rf_hw_', 'e2rf_plinth', 'e2rf_gable')
ROOFS = ('e2rf_roof',)
LOOK = {'block': 'western', 'asphalt': 'smooth', 'markings': 'white', 'verge': 'grass', 'buildings': 'mixed',
        'trees': 'regional', 'statics': 'street'}


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Buildings(unittest.TestCase):
    def setUp(self):
        self.lua = lupa.LuaRuntime()
        self.lua.globals().REPO_ROOT = ROOT
        self.lua.execute(PRELUDE)

    def t(self, value):
        if isinstance(value, dict):
            return self.lua.table_from({k: self.t(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return self.lua.table_from([self.t(v) for v in value])
        return value

    def triangles(self, buildings, far=False, latitude=52.5, look=LOOK):
        got = self.lua.globals().TRIANGLES(self.t([cell(buildings)]), far, latitude, self.t(look))
        out = []
        for tri in got.values():
            v = list(tri.values())
            out.append((v[0], [tuple(v[1:4]), tuple(v[4:7]), tuple(v[7:10])]))
        return out

    def near(self, tris, x0, y0, x1, y1, kinds):
        return [(m, p) for m, p in tris if m.startswith(kinds)
                and all(x0 - .01 <= q[0] <= x1 + .01 and y0 - .01 <= q[1] <= y1 + .01 for q in p)]

    def test_passage_keeps_the_lanes_free(self):
        # a 4-floor block standing across the road: its walls start 4.5 m over the lanes, and stand on the
        # ground beside them
        ring = box(200., ROAD_Y - 10, 230., ROAD_Y + 10)
        tris = self.triangles([building('p', ring, {'building': 'yes', 'building:levels': '4'}, 12.)])
        walls = self.near(tris, 200., ROAD_Y - 10, 230., ROAD_Y + 10, WALLS)
        self.assertTrue(walls)
        half = LANE / 2 + .3
        for _, p in walls:
            cy = sum(q[1] for q in p) / 3
            if abs(cy - ROAD_Y) < half - .05:
                self.assertGreaterEqual(min(q[2] for q in p), Z + .1 + 4.5 - .05, p)
        feet = [p for _, p in walls if min(q[2] for q in p) < Z + .5]
        self.assertTrue(feet, 'the walls beside the road stand on the ground')
        for p in feet:
            for q in p:
                if q[2] < Z + .5:
                    self.assertGreaterEqual(abs(q[1] - ROAD_Y), half - .01, p)

    def test_maxheight_and_a_grazing_road(self):
        # a passage tagged with its own clearance; a house whose wall overlaps the carriageway's edge
        # by a metre (the centreline does not run through it) keeps its walls down to the ground
        road_tags = {'highway': 'primary', 'lanes': '2', 'maxheight': '3.2', 'tunnel': 'building_passage'}
        c = cell([building('p', box(300., ROAD_Y - 10, 320., ROAD_Y + 10), {'building': 'yes', 'building:levels': '3'}, 9.),
                  building('g', box(100., ROAD_Y + LANE / 2 - 1, 112., ROAD_Y + LANE / 2 + 8), {'building': 'house'}, 6.)])
        c['roads'][0]['tags'] = road_tags
        got = self.lua.globals().TRIANGLES(self.t([c]), False, 52.5, self.t(LOOK))
        tris = [(list(t.values())[0], [tuple(list(t.values())[1 + 3 * k:4 + 3 * k]) for k in range(3)]) for t in got.values()]
        passage = [p for m, p in self.near(tris, 300., ROAD_Y - 10, 320., ROAD_Y + 10, WALLS)
                   if abs(sum(q[1] for q in p) / 3 - ROAD_Y) < LANE / 2]
        self.assertTrue(passage)
        self.assertAlmostEqual(min(min(q[2] for q in p) for p in passage), Z + .1 + 3.4, delta=.06)
        grazing = self.near(tris, 100., ROAD_Y + LANE / 2 - 1, 112., ROAD_Y + LANE / 2 + 8, WALLS)
        self.assertLess(min(min(q[2] for q in p) for _, p in grazing), Z)

    def test_half_hipped_and_skillion_roofs(self):
        hh = building('h', box(100., 400., 112., 409.), {'building': 'house', 'roof:shape': 'half-hipped', 'roof:colour': 'black'}, 6.)
        sk = building('s', box(150., 400., 160., 406.), {'building': 'house', 'roof:shape': 'skillion', 'roof:direction': 'N'}, 3.)
        tris = self.triangles([hh, sk])
        roof = self.near(tris, 99., 399., 113., 410., ROOFS)
        self.assertEqual({m for m, _ in roof}, {'e2rf_roof_slate_dark'})
        self.assertEqual(len(roof), 10)                 # two hexagons (4 each) and two hips
        ridge = max(q[2] for _, p in roof for q in p)
        self.assertGreater(ridge, Z + 6 + 1)
        lean = self.near(tris, 149., 399., 161., 407., ROOFS)
        self.assertEqual(len(lean), 2)
        zs = sorted({round(q[2], 2) for _, p in lean for q in p})
        # facing north: low on the north side (y = 406), high on the south (one plane over the
        # walls, no eaves: the lean-to is the generic skillion plane, BLD.skillion)
        north = [q for _, p in lean for q in p if q[1] > 405]
        south = [q for _, p in lean for q in p if q[1] < 401]
        self.assertLess(max(q[2] for q in north), min(q[2] for q in south))
        self.assertEqual(len(zs), 2)

    def test_north_look_and_elsewhere(self):
        # 40 untagged houses: in the north over half are brick under pitched roofs; south of 51.3 the old mix
        houses = [building(f'b{k}', box(20. + 15 * k, 300., 30. + 15 * k, 308.), {'building': 'yes'}, 5.) for k in range(30)]
        houses += [building(f'a{k}', box(20. + 30 * k, 340., 44. + 30 * k, 352.), {'building': 'apartments', 'building:levels': '4'}, 12.)
                   for k in range(10)]

        def counts(latitude):
            tris = self.triangles(houses, latitude=latitude)
            walls = [m for m, _ in tris if m.startswith(('e2rf_hw_', 'e2rf_wall_'))]
            brick = sum(1 for m in walls if 'brick' in m or 'brk' in m)
            blocks = self.near(tris, 19., 339., 315., 353., ROOFS)
            pitched = sum(1 for m, p in blocks if m != 'e2rf_roof')
            return brick / max(1, len(walls)), pitched
        north, south = counts(52.5), counts(48.0)
        self.assertGreater(north[0], .5)
        self.assertGreater(north[0], south[0] + .15)
        self.assertGreater(north[1], 0)                  # the 4-floor blocks are gabled in the north
        self.assertEqual(south[1], 0)                    # and flat elsewhere, as before

    def test_far_tiles_keep_their_budget(self):
        ring = box(200., ROAD_Y - 10, 230., ROAD_Y + 10)
        tris = self.triangles([building('p', ring, {'building': 'yes', 'building:levels': '4'}, 12.)], far=True)
        walls = self.near(tris, 200., ROAD_Y - 10, 230., ROAD_Y + 10, WALLS)
        # the two walls along the road keep plinth and wall (2 quads); the two across it are a plinth and
        # wall each side of the lanes and one wall over them (5 quads): 28 triangles, 16 without the road
        self.assertEqual(len(walls), 2 * (2 * 2 + 2 * 5))


if __name__ == '__main__':
    unittest.main()
