"""Guard rails at bridge decks, in LuaJIT through lupa (skipped without it): a rail runs on to the deck and joins its parapet.

Two paths draw rails. The generator's (generator.lua: ROADS.offDecks, ROADS.joinDecks, ROADS.substructure) draws the mapped
barrier ways of a tile and a plain approach rail; upstream keeps both off the deck (8 m off a high one, a few metres short
of any end), which the pack manifest's `extras.railsOffDecks` switches off. The swept roads' path (sweep.lua: M.railPaths)
draws the pack's furniture runs, which the exporter ends a sample or a station short of a bridge (a rule median rail one
station of 10 m), and runs them on to the deck. Measured here as the gap between a rail's end and the deck's end plane
(about 0: the rail's end is inside the parapet), on both sides of the bridge, at both its ends, median rails too, and as the
step between the rail's face and the parapet's face (none: no end sticks out toward the traffic, nothing snags).

Run: python -m unittest discover -s earthng -p bridge_rail_test.py
"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import numpy as np
    import fast_roads as fr
    from lupa import luajit21
except ImportError:  # pragma: no cover
    luajit21 = None

LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
SWEEP = require('ge/extensions/e2r/sweep')
COLLISION = require('ge/extensions/e2r/collision')
ROADS3 = require('ge/extensions/e2r/roads3')
function BUILD(cells, i, j, options) return {GENERATOR.build(cells, i, j, options)} end
function SWEPT(bytes, version)
  return ROADS3.merge({ROADS3.cell(bytes, 1, #bytes, 0, 0, version, function() end)}, nil)
end
-- the world triangles drawn in a material: a flat list of x, y, z, three vertices to a triangle
function TRIS(result, material)
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
-- collision.lua's verdict on a mesh of this material in a layer: true when it has none
function QUIET(material, layer) return COLLISION.quiet({{material = material}}, layer) end
'''

RAIL = 'italy_guardrails'     # the stock guard rail material (sweep.lua M.RAIL; no engine here: the stock look)
WALL = 'e2rf_fence_wall'       # a concrete deck's parapet
Z = 30.
TILE = 500


def runtime():
    if luajit21 is None:
        return None
    L = luajit21.LuaRuntime(unpack_returned_tuples=True)
    L.globals().REPO_ROOT = ROOT.replace(chr(92), '/')
    L.execute(LUA)
    return L


def to_lua(L, v):
    if isinstance(v, dict):
        return L.table_from({k: to_lua(L, x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return L.table_from([to_lua(L, x) for x in v])
    return v


def flat_terrain():
    return {'rect': [0., 0., 1000., 1000.], 'nx': 101, 'ny': 101, 'heights': [Z] * 101 * 101}


def cell(roads=(), lines=()):
    return {'terrain': flat_terrain(), 'roads': list(roads), 'junctions': [], 'buildings': [], 'areas': [], 'lines': list(lines), 'points': []}


class Scene:
    """Builds the two tiles of a 1000 m cell and collects the triangles of a material as an (n, 3) array."""

    def __init__(self, L, data, roads3=None):
        self.L, self.data, self.roads3 = L, data, roads3

    def build(self, i, **options):
        o = {'tileSize': TILE, 'regionId': 'test', 'far': False, 'horizon': False, 'farDecor': False, 'farStep': 100,
             'latitude': 52.5, 'seaLevel': 0, 'country': 'de', 'heightPipeline': True}
        o.update(options)
        lua = to_lua(self.L, o)
        if self.roads3 is not None:
            lua.roads3 = self.roads3
        return self.L.globals().BUILD(to_lua(self.L, [self.data]), i, 0, lua)

    def points(self, material, **options):
        out = []
        for i in (0, 1):
            flat = list(self.L.globals().TRIS(self.build(i, **options), material).values())
            out += [flat[k:k + 3] for k in range(0, len(flat), 3)]
        return np.array(out) if out else np.zeros((0, 3))


def materials(result):
    """the materials of the meshes of a tile's three manifests"""
    out = set()
    for layer in (1, 2, 3):
        if result[layer] is not None:
            for item in result[layer]['items'].values():
                for mesh in (item['meshes'] or {}).values():
                    out.add(mesh['material'])
    return out


def window(pts, x0, x1, y0, y1):
    return pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]


# ---------------------------------------------------------------------------------------------------------------------
# The generator path: a road y = 100 with a bridge from x = 200 to 400, mapped guard rails 7 m off its axis
# ---------------------------------------------------------------------------------------------------------------------
HW = 5.5                       # the road's half width (an 11 m way)


def generator_cell(highway, deck_dz, mapped):
    def pt(x, y, dz=0.):
        return [x, y, Z + dz]
    tags = {'highway': highway, 'lanes': '2'}
    roads = [
        {'id': '3001', 'width': 2 * HW, 'tags': dict(tags), 'nodes': ['a0', 'a1'], 'points': [pt(x, 100) for x in range(0, 201, 20)]},
        {'id': '3002', 'width': 2 * HW, 'tags': dict(tags, bridge='yes', layer='1'), 'nodes': ['a1', 'a2'],
         'points': [pt(x, 100, deck_dz) for x in range(200, 401, 20)]},
        {'id': '3003', 'width': 2 * HW, 'tags': dict(tags), 'nodes': ['a2', 'a3'], 'points': [pt(x, 100) for x in range(400, 801, 20)]}]
    lines = []
    if mapped:
        for n, xy in enumerate([[(20, 93), (200, 93)], [(20, 107), (200, 107)], [(400, 93), (780, 93)], [(400, 107), (780, 107)]]):
            lines.append({'id': 'r%d' % n, 'kind': 'guard_rail', 'tags': {'barrier': 'guard_rail'}, 'points': [pt(x, y) for x, y in xy]})
    return cell(roads, lines)


@unittest.skipIf(luajit21 is None, 'needs lupa with LuaJIT')
class GeneratorRails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = runtime()

    def scene(self, highway, deck_dz, mapped):
        return Scene(self.L, generator_cell(highway, deck_dz, mapped))

    def ends(self, rails, lo=0.0, hi=1000.0):
        """gap between the deck's end planes (x 200 and 400) and the nearest rail end on each side of each end, as a dict
        {(end, side): gap}; a rail that stops short has a positive gap, one inside the parapet a negative one"""
        gaps = {}
        for plane, direction in ((200., 1), (400., -1)):
            for side, sel in (('south', lambda y: y < 100), ('north', lambda y: y > 100)):
                near = rails[(abs(rails[:, 0] - plane) < 25) & sel(rails[:, 1])]
                near = near[near[:, 0] * direction < plane * direction + 1e-9 + 5] if len(near) else near
                if len(near):
                    gaps[(plane, side)] = (plane - near[:, 0].max()) if direction > 0 else (near[:, 0].min() - plane)
        return gaps

    def test_upstream_keeps_a_mapped_rail_off_the_deck(self):
        # the default: the rail stops 4 m short of a low deck's end, 12 m short of a high one (8 m off its edge)
        low = self.ends(self.scene('secondary', 0., True).points(RAIL))
        high = self.ends(self.scene('secondary', 4., True).points(RAIL))
        self.assertEqual(len(low), 4)
        self.assertTrue(all(g > 3 for g in low.values()), low)
        self.assertTrue(all(g > 8 for g in high.values()), high)

    def test_switched_off_a_mapped_rail_runs_on_to_the_deck(self):
        for dz in (0., 4.):
            gaps = self.ends(self.scene('secondary', dz, True).points(RAIL, railsOffDecks=False))
            self.assertEqual(len(gaps), 4, gaps)                                # both ends, both sides
            for key, g in gaps.items():
                self.assertTrue(-.5 <= g <= .1, 'deck rise %g, %s: gap %.2f' % (dz, key, g))

    def faces(self, rails, walls, plane, side):
        """distance from the axis of the rail's face toward the traffic at the deck's end plane and of the parapet's inner face"""
        ylo, yhi = (80, 99) if side < 0 else (101, 120)
        # the parapet stands on the bridge side of the plane (upstream's starts a piece or two later)
        x0, x1 = (plane, plane + 4) if plane < 300 else (plane - 4, plane)
        r = window(rails, plane - .5, plane + .5, ylo, yhi)
        w = window(walls, x0, x1, ylo, yhi)
        return np.abs(r[:, 1] - 100).min(), np.abs(w[:, 1] - 100).min()

    def test_the_plain_approach_rail_ends_in_the_parapet_too(self):
        # a trunk road's own rail on the 40 m of approach, no mapped barrier. It stood 0.5 m outside the parapet's outer face,
        # the parapet's end 0.75 m proud of its beam; switched off it draws in and the parapet starts at the deck's end
        sc = self.scene('trunk', 0., False)
        for plane in (200., 400.):
            for side in (-1, 1):
                r0, w0 = self.faces(sc.points(RAIL), sc.points(WALL), plane, side)
                self.assertGreater(r0 - w0, .5, 'default: %.2f' % (r0 - w0))
                r1, w1 = self.faces(sc.points(RAIL, railsOffDecks=False), sc.points(WALL, railsOffDecks=False), plane, side)
                self.assertTrue(-.02 <= r1 - w1 <= .06, 'face step %.3f at %g side %d' % (r1 - w1, plane, side))

    def test_the_parapet_starts_at_the_deck_end_beside_a_plain_approach(self):
        walls = self.scene('trunk', 0., False)
        up, fork = walls.points(WALL), walls.points(WALL, railsOffDecks=False)
        top = lambda p: p[p[:, 2] > 30.9]                                          # the parapet's top (the deck surface is 30.1)
        self.assertGreater(top(window(up, 190, 215, 80, 99))[:, 0].min(), 201.9)      # upstream: the first piece is left out
        self.assertAlmostEqual(top(window(fork, 190, 215, 80, 99))[:, 0].min(), 200., delta=.01)
        self.assertAlmostEqual(top(window(fork, 385, 410, 80, 99))[:, 0].max(), 400., delta=.01)

    def test_the_rail_reaches_the_end_plane_on_both_sides_and_ends(self):
        gaps = self.ends(self.scene('trunk', 0., False).points(RAIL, railsOffDecks=False))
        self.assertEqual(len(gaps), 4, gaps)
        for key, g in gaps.items():
            self.assertTrue(-.5 <= g <= .1, '%s: gap %.2f' % (key, g))

    def test_the_mapped_rail_draws_in_to_the_parapet_without_a_step(self):
        sc = self.scene('secondary', 0., True)
        rails, walls = sc.points(RAIL, railsOffDecks=False), sc.points(WALL, railsOffDecks=False)
        for plane in (200., 400.):
            for side in (-1, 1):
                r1, w1 = self.faces(rails, walls, plane, side)
                self.assertTrue(-.02 <= r1 - w1 <= .06, 'step %.3f' % (r1 - w1))
                # 25 m before the end it is on its mapped line, 7 m off the axis: the draw-in is gradual
                ylo, yhi = (80, 99) if side < 0 else (101, 120)
                far = window(rails, plane - 45, plane - 25, ylo, yhi) if plane < 300 else window(rails, plane + 25, plane + 45, ylo, yhi)
                self.assertAlmostEqual(abs(far[:, 1] - 100).mean(), 7., delta=.3)

    def test_the_manifest_switch_and_the_options_agree(self):
        g = self.L.globals().GENERATOR
        self.assertTrue(g.EXTRAS.railsOffDecks)                                  # upstream's by default
        sc = self.scene('secondary', 0., True)
        try:
            g.EXTRAS.railsOffDecks = False                                       # as a pack's extras set it (native.lua)
            self.assertTrue(all(-.5 <= v <= .1 for v in self.ends(sc.points(RAIL)).values()))
            # a call's own option wins over the pack's
            self.assertTrue(all(v > 3 for v in self.ends(sc.points(RAIL, railsOffDecks=True)).values()))
        finally:
            g.EXTRAS.railsOffDecks = True

    def test_the_rail_and_the_parapet_keep_their_collision(self):
        # collision.lua leaves a mesh without collision when every material of it is in QUIET
        for material in (RAIL, 'e2rf_guardrail', 'e2rf_clad_metal', WALL, 'e2rf_hw_stone', 'e2rf_hw_oldbrick'):
            for layer in ('road', 'scene'):
                self.assertFalse(self.L.globals().QUIET(material, layer), '%s is quiet in %s' % (material, layer))
        sc = self.scene('trunk', 0., True)
        for options in ({}, {'railsOffDecks': False}):
            self.assertTrue(len(sc.points(RAIL, **options)) and len(sc.points(WALL, **options)))
        # the joined piece is drawn in the same material as the rest of the rail: no kind of its own
        mats = set()
        for i in (0, 1):
            result = sc.build(i, railsOffDecks=False)
            mats |= materials(result)
        self.assertIn(RAIL, mats)
        self.assertFalse([m for m in mats if m != WALL and ('rail' in m or 'beam' in m or 'guard' in m) and m != RAIL], mats)


# ---------------------------------------------------------------------------------------------------------------------
# The swept path: a dual carriageway (pack v3) with a bridge from x = 200 to 400 on each carriageway
# ---------------------------------------------------------------------------------------------------------------------
EAST_Y, WEST_Y = 100., 111.


def segment(i, xy, tags, nodes, width=7.5):
    length = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    return {'id': i, 'way': i, 'tags': tags, 'xy': [list(p) for p in xy], 'nodes': nodes, 'width': width, 'prof': [[0., length], [Z, Z]]}


def swept_pack(barriers):
    motorway = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes'}
    bridge = dict(motorway, bridge='yes', layer='1')

    def line(y, x0, x1):
        step = 20 if x1 > x0 else -20
        return [(x, y) for x in range(x0, x1 + step, step)]
    segs = [segment(101, line(EAST_Y, 0, 200), motorway, ['a0', 'a1']), segment(102, line(EAST_Y, 200, 400), bridge, ['a1', 'a2']),
            segment(103, line(EAST_Y, 400, 800), motorway, ['a2', 'a3']), segment(104, line(WEST_Y, 800, 400), motorway, ['b3', 'b2']),
            segment(105, line(WEST_Y, 400, 200), bridge, ['b2', 'b1']), segment(106, line(WEST_Y, 200, 0), motorway, ['b1', 'b0'])]
    graph = fr.Graph(segs)
    strokes = fr.build_strokes(segs, graph)
    fr.structures(strokes, segs, [])
    fr.furniture(strokes, segs, barriers)
    fr.median_rails(strokes)
    rows, nodes = fr.records(strokes, graph, segs, 1000.)
    return strokes, fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.)), fr.VERSION


def barrier(wid, xy):
    return (wid, ('guard_rail',), np.array(xy, float))


@unittest.skipIf(luajit21 is None, 'needs lupa with LuaJIT')
class SweptRails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = runtime()
        # mapped rails on both outer sides, each ending 4 m short of the bridge (and starting 4 m past it); the carriageways'
        # edges are 3.5 m apart, so the exporter adds a median rail by rule (it ends one 10 m station short)
        cls.strokes, data, version = swept_pack([
            barrier(11, [(20, 94.9), (196, 94.9)]), barrier(12, [(404, 94.9), (780, 94.9)]),
            barrier(13, [(20, 116.1), (196, 116.1)]), barrier(14, [(404, 116.1), (780, 116.1)])])
        cls.scene = Scene(cls.L, cell(), cls.L.globals().SWEPT(data, version))

    def setUp(self):
        self.sweep = self.L.globals().SWEEP

    def zones(self, rails):
        return {'south': window(rails, -10, 810, 80, 98), 'median': window(rails, -10, 810, 103, 108),
                'north': window(rails, -10, 810, 112, 125)}

    def gaps(self, rails):
        out = {}
        for name, pts in self.zones(rails).items():
            for plane, direction in ((200., 1), (400., -1)):
                near = pts[(pts[:, 0] > plane - 20) & (pts[:, 0] < plane + 20)]
                before = near[near[:, 0] < plane + 1] if direction > 0 else near[near[:, 0] > plane - 1]
                if len(before):
                    out[(name, plane)] = (plane - before[:, 0].max()) if direction > 0 else (before[:, 0].min() - plane)
        return out

    def test_the_exporter_stops_rails_short_of_the_bridge(self):
        # the cause: a mapped run ends on a 2 m sample of its barrier, the rule median rail at a 10 m station
        runs = {(st.id, f['way'], f['s0'], f['s1']) for st in self.strokes for f in st.furniture}
        self.assertIn(('101_a0_a3', '11', 20.0, 196.0), runs)
        self.assertIn(('101_a0_a3', 'rule:median', 0.0, 190.0), runs)

    def test_without_the_join_the_gap_is_what_the_exporter_left(self):
        old = self.sweep.TUNE.railJoin
        try:
            self.sweep.TUNE.railJoin = 0
            gaps = self.gaps(self.scene.points(RAIL))
        finally:
            self.sweep.TUNE.railJoin = old
        self.assertAlmostEqual(gaps[('south', 200.)], 4., delta=.3)
        self.assertGreater(gaps[('median', 200.)], 9.)

    def test_a_rail_runs_on_into_the_parapet_both_sides_both_ends_median_too(self):
        gaps = self.gaps(self.scene.points(RAIL))
        for zone in ('south', 'north', 'median'):
            for plane in (200., 400.):
                self.assertIn((zone, plane), gaps)
                self.assertTrue(-.5 <= gaps[(zone, plane)] <= .1, '%s at %g: gap %.2f' % (zone, plane, gaps[(zone, plane)]))

    def test_the_single_rail_draws_in_to_the_parapet_without_a_step(self):
        rails, walls = self.scene.points(RAIL), self.scene.points(WALL)
        for plane in (200., 400.):
            for name, (ylo, yhi), toward in (('south', (80, 99), 1), ('north', (112, 130), -1)):
                r = window(rails, plane - .4, plane + .4, ylo, yhi)
                w = window(walls, plane - .05, plane + .05, ylo, yhi)
                self.assertTrue(len(r) and len(w), (name, plane))
                # the face toward the traffic: the largest y of the south side, the smallest of the north side
                face = lambda p: p[:, 1].max() if toward > 0 else p[:, 1].min()
                self.assertLessEqual(abs(face(r) - face(w)), .06, '%s at %g: step %.3f' % (name, plane, face(r) - face(w)))
                # and it does not stand out toward the traffic past the parapet's face
                self.assertLessEqual((face(r) - face(w)) * toward, .01)
                # 22 m back it is on its mapped line
                far = window(rails, plane - 40, plane - 22, ylo, yhi) if plane < 300 else window(rails, plane + 22, plane + 40, ylo, yhi)
                self.assertTrue(len(far))

    def test_the_median_rail_ends_level_with_the_parapets(self):
        # a two-faced rail on its posts between two decks that stand apart: it ends at the end plane, on its own line
        rails = self.zones(self.scene.points(RAIL))['median']
        for plane in (200., 400.):
            near = window(rails, plane - 15, plane + 15, 103, 108)
            self.assertAlmostEqual((near[:, 1].min() + near[:, 1].max()) / 2, 105.5, delta=.1)
            self.assertLess(near[:, 1].max() - near[:, 1].min(), .6)

    def test_the_rail_and_the_parapet_keep_their_collision(self):
        for layer in ('road', 'scene'):
            self.assertFalse(self.L.globals().QUIET(RAIL, layer))
            self.assertFalse(self.L.globals().QUIET(WALL, layer))
        # the rail's pieces (joined or not) and the deck are drawn in these materials, in one mesh item
        mats = set()
        for i in (0, 1):
            mats |= materials(self.scene.build(i))
        self.assertTrue({RAIL, WALL} <= mats, mats)
        self.assertFalse([m for m in mats if self.L.globals().QUIET(m, 'road') and m in (RAIL, WALL)])


if __name__ == '__main__':
    unittest.main()
