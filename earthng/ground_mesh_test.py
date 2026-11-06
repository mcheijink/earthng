"""The ground mesh of the Lua tile generator, switch off and on (ground_mesh_check.py is the harness).

Synthetic: a 10 m DEM with a cross fall and bumps, swept motorway and trunk strokes (sweep.lua) made from synthetic
pack v3 chunks: on a bank, in a cut, on a bend, across a tile border, over a bridge. Every scene is built as a block
(the tile and its eight neighbours: a swept chunk is drawn whole by the tile that holds its middle), once with
generator.CONFORMAL.on off and once on.

Off, today's ground lies under the roads (the baseline). On, the gate:
  - no hole anywhere, a tile built alone too (it draws the road inside its own rect): every boundary edge of the ground, the carriageway and the verge bands has a surface beside it,
    tile borders included (zero tolerance: 1 mm in plan, the weld quantum)
  - no ground under a carriageway (the bridge deck's ground stays)
  - every corner on a carriageway's free edge is welded to a ground corner (the verge band's)
  - the road edges stand where the stroke puts them, at the profile plus the reduced lift
  - (real tiles) no coplanar carriageway on carriageway at forks and merges
  - no end wall of a band (the face that closes it at a bridge or tunnel) or other upright concrete stands inside a carriageway
    (zero tolerance: walls_in_road), over a bridge on a bank and where a road passes under it; each wall comes down to the surface
    that lies below it (a wall hanging over the ground leaves a slit the car sees through)

Real tiles of a staged pack (E2R_TEST_PACK=<dir>, E2R_TEST_TILES="i,j;i,j"): the same gate. Run: python earthng/ground_mesh_test.py [-v]"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ground_mesh_check as G      # noqa: E402

GRID, X0, N = 10., -600., 221       # DEM: 10 m, from -600 m, 221 x 221 nodes (the block's tiles are 0..500 and round)
LIFT = .02                          # CONFORMAL.TUNE.lift


def dem(x, y):
    return 20 + .03 * x + .01 * y + 2.5 * math.sin(x / 37.) * math.cos(y / 53.) + .6 * math.sin(x / 9.1 + y / 6.7)


def terrain():
    heights = [dem(X0 + i * GRID, X0 + j * GRID) for j in range(N) for i in range(N)]
    return {'rect': [X0, X0, X0 + (N - 1) * GRID, X0 + (N - 1) * GRID], 'nx': N, 'ny': N, 'heights': heights}


def line(a, b, step=2.):
    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // step))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1)]


def arc(c, r, a0, a1, step=2.):
    n = max(2, int(abs(a1 - a0) * r // step))
    return [(c[0] + r * math.cos(a0 + (a1 - a0) * k / n), c[1] + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


def chunk(sid, path, half, offset, highway, structures=(), **tags):
    """one pack v3 chunk along path (stations about 2 m apart), its profile on the DEM plus offset (a bank > 0, a cut < 0)"""
    st = {k: [] for k in ('s', 'x', 'y', 'z', 'h', 'k', 'eL', 'eR', 'fl', 'fr')}
    s = 0.
    for i, (x, y) in enumerate(path):
        if i:
            s += math.hypot(x - path[i - 1][0], y - path[i - 1][1])
        q = path[min(i + 1, len(path) - 1)] if i < len(path) - 1 else None
        p = path[i - 1] if i else None
        a, b = (p or (x, y)), (q or (x, y))
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        for key, v in (('s', s), ('x', x), ('y', y), ('z', dem(x, y) + offset), ('h', h), ('k', 0.), ('eL', half), ('eR', half),
                       ('fl', -.025), ('fr', -.025)):
            st[key].append(v)
    n = len(path)
    t = dict(highway=highway, **tags)
    c = {'stroke': sid, 'index': 1, 'chunks': 1, 'oneway': False, 'closed': False, 'fitted': True, 'highway': highway,
         'length': s, 'd0': 0., 'd1': s, 'n': n,
         'parts': [{'way': sid, 'segment': 1, 'reversed': False, 's0': 0., 's1': s, 'tags': t}],
         'lanes': [], 'events': [], 'furniture': [], 'structures': list(structures), 'attach': [], 'nodes': []}
    c.update(st)
    return c


def bridge(s0, s1):
    return {'kind': 'bridge', 's0': s0, 's1': s1, 'layer': 1, 'structure': 'bridge', 'clearance': 5., 'water': False, 'floor': 0.}


# the scenes, each a list of chunks across the block
SCENES = {
    # a trunk on a 1 m bank, straight across three tiles
    'bank': [chunk('T1', line((-400, 250), (1400, 290)), 5.5, 1., 'trunk')],
    # a motorway in a 1.5 m cut, on a bend of 600 m radius
    'cut_bend': [chunk('M1', arc((250, 900), 600, math.radians(-130), math.radians(-50)), 7., -1.5, 'motorway')],
    # a motorway on a 3 m embankment along the diagonal of the central tile
    'embankment': [chunk('M2', line((-300, -200), (900, 700)), 7., 3., 'motorway')],
}


@unittest.skipIf(G.lupa is None, 'lupa.luajit21 not installed')
class Ground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = G.Runtime()
        cls.cells = [{'terrain': terrain(), 'roads': [], 'junctions': [], 'buildings': [], 'areas': [], 'lines': [], 'points': []}]
        cls.made = {}
        cls.m = {}

    @classmethod
    def block(cls, scene, on):
        """(Tile block, measures) of a scene: a name of SCENES, or (name, chunks)"""
        name, chunks = (scene, SCENES[scene]) if isinstance(scene, str) else scene
        key = (name, on)
        if key not in cls.made:
            extra = {'roads3': cls.rt.P.merge(cls.rt.table(chunks))}
            tile = cls.rt.block(0, 0, conformal=on, cells=cls.cells, extra=extra)
            cls.made[key] = tile
            cls.m[key] = G.measure(tile)
        return cls.made[key], cls.m[key]

    def test_the_switch_is_off_by_default(self):
        self.assertTrue(self.rt.has_switch)
        self.assertFalse(self.rt.L.eval("require('ge/extensions/e2r/generator').CONFORMAL.on"))

    def test_off_ground_lies_under_the_road(self):
        # the baseline: the lattice runs under every carriageway (and its hidden share of the area is the cost)
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, m = self.block(scene, False)
                self.assertGreater(m['carriageway_m2'], 2000)
                self.assertGreater(m['under_m2'], .95 * m['carriageway_m2'])

    def test_on_no_ground_under_the_carriageway(self):
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, m = self.block(scene, True)
                self.assertGreater(m['carriageway_m2'], 2000)
                self.assertLessEqual(m['under_m2'], 1., m)

    def test_on_no_hole_and_no_open_seam(self):
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, m = self.block(scene, True)
                self.assertGreater(m['cracks']['edges'], 100)
                self.assertEqual(m['scope']['hole'], 0, m['scope']['hole_at'][:5])
                self.assertEqual(m['scope']['seam_hole'], 0)

    def test_on_a_tile_alone_has_no_hole(self):
        # the tile built before the neighbour that holds the chunk's middle: it draws the road inside its own rect
        for scene in SCENES:
            with self.subTest(scene=scene):
                tile, _ = self.block(scene, True)
                alone = tile.alone
                self.assertGreater(len(alone.of('sw_road')), 100, 'the road is drawn by the tile itself')
                sc = G.scope(alone, G.cracks(alone)['found'])
                self.assertEqual(sc['hole'], 0, sc['hole_at'][:5])

    def test_on_road_edges_are_welded_to_the_ground(self):
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, m = self.block(scene, True)
                st = m['stitch']
                self.assertGreater(st['edge_corners'], 200)
                self.assertEqual(st['open'], 0, st['open_at'][:5])
                self.assertEqual(st['to_ground'], st['edge_corners'])

    def test_on_band_meets_ground_with_no_step(self):
        # nothing but the stroke and the ground: no surface beside another at another height in the corridor
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, m = self.block(scene, True)
                self.assertEqual(m['scope']['corridor_step'], 0, m['scope']['corridor_at'][:5])

    def test_on_far_less_double_layers(self):
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, off = self.block(scene, False)
                _, on = self.block(scene, True)
                self.assertLess(on['layers']['depth'][2], .35 * off['layers']['depth'][2])

    def test_on_triangles_stay_in_proportion(self):
        # the pieces along the toe cost triangles; the hidden lattice and the double layers go. Bounded, not a saving
        for scene in SCENES:
            with self.subTest(scene=scene):
                _, off = self.block(scene, False)
                _, on = self.block(scene, True)
                self.assertLess(on['tris']['gnd'], 1.6 * off['tris']['gnd'])

    def test_on_road_at_the_profile_plus_the_reduced_lift(self):
        tile, _ = self.block('bank', True)
        road = tile.of('sw_road').reshape(-1, 3)
        inside = road[(road[:, 0] > 100) & (road[:, 0] < 400)]
        # the vertices of the centre line: the profile (DEM + 1 m bank) plus the lift
        centre = inside[np.abs(inside[:, 1] - (250 + (inside[:, 0] + 400) / 1800 * 40)) < .05]
        self.assertGreater(len(centre), 20)
        want = np.array([dem(x, y) + 1. + LIFT for x, y, _ in centre])
        self.assertLess(float(np.abs(centre[:, 2] - want).max()), .002)

    def test_bridge_keeps_the_ground_under_its_deck(self):
        chunks = [chunk('B1', line((-400, 300), (1400, 300)), 5.5, 6., 'trunk', structures=[bridge(800., 1000.)])]
        tile, m = self.block(('bridge', chunks), True)
        deck = tile.of('deck')
        self.assertGreater(len(deck), 0, 'a deck')
        # ground under the deck: the deck's cells have ground
        cells = G.bridge_cells(tile)
        gnd = tile.of('gnd')
        _, gc, _ = G.raster(gnd, tile.rect[0], tile.rect[1], 500, 500, 1.)
        under = len(cells & set(gc.tolist()))
        self.assertGreater(under, .9 * len(cells))
        self.assertEqual(m['scope']['hole'], 0)

    # a trunk on a 6 m bank with a bridge over a motorway that crosses it at 70 degrees: the band of the bank ends at the
    # bridge, the road under it has its own bands there, so the two stack
    OVERPASS = ('overpass', [chunk('B1', line((-400, 300), (1400, 300)), 5.5, 6., 'trunk', structures=[bridge(780., 1020.)]),
                             chunk('M3', line((630, -300), (1070, 900)), 7., 0., 'motorway')])

    def test_on_end_walls_stand_on_what_is_below_them(self):
        chunks = [chunk('B1', line((-400, 300), (1400, 300)), 5.5, 6., 'trunk', structures=[bridge(800., 1000.)])]
        tile, m = self.block(('bridge', chunks), True)
        walls = G.upright_walls(tile)
        self.assertGreater(len(walls), 0, 'the bands end at the bridge: walls close them')
        gnd = tile.of('gnd')
        hanging = 0
        for w in walls:
            for k in range(3):
                x, y, z = w[3 * k:3 * k + 3]
                l1, l2, l3 = G.bary(gnd, np.full(len(gnd), x), np.full(len(gnd), y))
                inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
                if not inside.any():
                    continue
                zs = (l1 * gnd[:, 2] + l2 * gnd[:, 5] + l3 * gnd[:, 8])[inside]
                if np.abs(zs - z).min() > .02 and z < zs.max():
                    hanging += 1     # a corner below the drawn ground is a wall over a gap; one above it, on the band, is fine
        self.assertEqual(hanging, 0)

    def test_on_no_wall_stands_in_a_carriageway(self):
        for scene in ('bridge', self.OVERPASS):
            if scene == 'bridge':
                scene = ('bridge', [chunk('B1', line((-400, 300), (1400, 300)), 5.5, 6., 'trunk', structures=[bridge(800., 1000.)])])
            with self.subTest(scene=scene[0]):
                tile, m = self.block(scene, True)
                res = G.walls_in_road(tile)
                self.assertEqual(res['hits'], 0, res['found'][:5])
                self.assertEqual(m['scope']['hole'], 0)

    def test_the_wall_check_sees_a_wall_in_the_road(self):
        # the check itself: a made-up wall across the carriageway is found, one beside it is not
        tile, _ = self.block(('bridge', [chunk('B1', line((-400, 300), (1400, 300)), 5.5, 6., 'trunk', structures=[bridge(800., 1000.)])]), True)
        road = tile.of('sw_road')
        c = road[len(road) // 2].reshape(3, 3).mean(0)
        zz = float(c[2])

        class T:      # the tile plus two made-up upright triangles
            pass
        t = T()
        t.D = np.concatenate([tile.D, np.array([[c[0], c[1] - 3, zz - 1, c[0], c[1] + 3, zz - 1, c[0], c[1] + 3, zz + 1],
                                                [c[0] + 60, c[1] - 3, zz + 40, c[0] + 60, c[1] + 3, zz + 40, c[0] + 60, c[1] + 3, zz + 42]])])
        t.cls = np.concatenate([tile.cls, [G.CI['deck'], G.CI['deck']]])
        t.grp = np.concatenate([tile.grp, [len(tile.groups)] * 2])
        t.groups = tile.groups + [['road', 'wall', 'fence_wall', '0']]
        t.rect, t.size = tile.rect, tile.size
        t.of = lambda *n: t.D[np.isin(t.cls, [G.CI[k] for k in n])]
        res = G.walls_in_road(t)
        self.assertGreater(res['hits'], 0)
        self.assertTrue(all(abs(h[0] - c[0]) < 1. for h in res['found']))

    def test_wall_texture_runs_along_the_wall_and_with_its_height(self):
        # conformal.wallUV: a face without plan area was mapped from its plan position, and the texture ran down it in streaks
        L = self.rt.L
        f = L.eval("function(a, b, z0, z1, rep) local w = {{a[1], a[2], z1}, {b[1], b[2], z1}, {b[1], b[2], z0}, {a[1], a[2], z0}}; "
                   "require('ge/extensions/e2r/conformal').wallUV(w, rep); local out = {} for _, c in ipairs(w) do out[#out + 1] = {c[4], c[5]} end return out end")
        uv = [tuple(c.values()) for c in f(L.table_from([10., 20.]), L.table_from([13., 24.]), 5., 9., 2.5).values()]
        self.assertAlmostEqual(uv[1][0] - uv[0][0], 5. / 2.5, 6)            # 5 m along the wall
        self.assertAlmostEqual(uv[0][0], uv[3][0], 6)                       # u does not change down a vertical edge
        self.assertAlmostEqual(uv[3][1] - uv[0][1], 4. / 2.5, 6)            # v follows the height, down is more
        # world-locked: a wall of the same line built in two pieces meets with the same coordinates
        uv2 = [tuple(c.values()) for c in f(L.table_from([13., 24.]), L.table_from([16., 28.]), 5., 9., 2.5).values()]
        self.assertAlmostEqual(uv2[0][0], uv[1][0], 6)

    def test_the_budget_check_flags_an_explosion(self):
        off = {'tris': 20000, 'pieces': 30, 'solid': 16000, 'wall_tris': 0, 'wall_area': 0., 'wall_height': 0., 'degenerate': 0,
               'thin': 150, 'long': 0, 'dup': 100, 'gap_ms': 20.}
        self.assertEqual(G.check_budget(dict(off, tris=21000, gap_ms=30.), off), [])
        bad = G.check_budget(dict(off, tris=40000, pieces=80, gap_ms=224.), off)
        self.assertEqual(sorted(b[0] for b in bad), ['gap_ms', 'pieces', 'tris'])

    def test_on_costs_the_game_no_more_than_a_bounded_share_over_off(self):
        # per tile, alone: triangles, pieces, solid (collision) triangles, walls, and the longest stretch of the build that
        # cannot be split (a hitch of its frame) against the switch off
        for scene in list(SCENES) + [self.OVERPASS]:
            name, chunks = (scene, SCENES[scene]) if isinstance(scene, str) else scene
            extra = {'roads3': self.rt.P.merge(self.rt.table(chunks))}
            with self.subTest(scene=name):
                off = self.rt.budget(0, 0, False, cells=self.cells, extra=extra)
                on = self.rt.budget(0, 0, True, cells=self.cells, extra=extra)
                self.assertEqual(G.check_budget(on, off), [], (off, on))

    def test_the_end_span_is_cut_to_the_drawn_band(self):
        # conformal.endSpans: only the part of the end line that is an edge of the drawn piece
        L = self.rt.L
        f = L.eval("function(piece, a, b) local r = require('ge/extensions/e2r/conformal').endSpans(piece, a, b); "
                   "local out = {} for _, s in ipairs(r) do out[#out + 1] = {s[1][1], s[1][2], s[2][1], s[2][2]} end return out end")
        piece = L.table_from([L.table_from([0., 0., 5.]), L.table_from([4., 0., 5.]), L.table_from([4., 3., 5.]), L.table_from([0., 3., 5.])])
        a, b = L.table_from([-2., 0.]), L.table_from([10., 0.])
        got = [list(v.values()) for v in f(piece, a, b).values()]
        self.assertEqual(len(got), 1)
        self.assertEqual(sorted([got[0][0], got[0][2]]), [0., 4.])
        # the end line's own length cut the span: a is the line's start, b its end
        a, b = L.table_from([1., 0.]), L.table_from([3., 0.])
        got = [list(v.values()) for v in f(piece, a, b).values()]
        self.assertEqual(sorted([got[0][0], got[0][2]]), [1., 3.])


@unittest.skipIf(G.lupa is None, 'lupa.luajit21 not installed')
class BandUV(unittest.TestCase):
    """A band face steeper than 45 degrees is box mapped (the generator collector's hooks.box, set while the section bands are drawn):
    the flat top-down mapping of the ground streaks a steep face, the texture runs down it along a metre or two of plan"""

    @classmethod
    def setUpClass(cls):
        cls.rt = G.Runtime()
        cls.face = cls.rt.L.eval('''function(box, z0, run)
          local C = GEN.collector({0, 0, 100, 100}, 't', 'road', function() end, false)
          if box then C.hooks.box = GEN.SECTION.box end
          C.polygon('ground', {{10, 10, z0}, {20, 10, z0}, {20, 10 + run, 0}, {10, 10 + run, 0}}, true)
          C.finish()
          local m = C.items[1].meshes[1]
          local o = C.items[1].pos
          local out = {}
          for _, f in ipairs(m.faces) do
            local v, uv = m.verts[f.v + 1], m.uvs[f.u + 1]
            out[#out + 1] = {v.x + o[1], v.y + o[2], v.z + o[3], uv.u, uv.v}
          end
          return out
        end''')

    def rows(self, box, z0, run):
        return [[float(c) for c in r.values()] for r in self.face(box, z0, run).values()]

    def test_a_steep_face_is_mapped_along_its_height(self):
        # 6 m of height over 3 m of plan: 63 degrees. The ground's texture repeats every 24 m
        for x, y, z, u, v in self.rows(True, 6., 3.):
            self.assertAlmostEqual(u * 24., x, 3)
            self.assertAlmostEqual(v * 24., z, 3)

    def test_without_the_hook_a_steep_face_keeps_the_plan_mapping(self):
        for x, y, z, u, v in self.rows(False, 6., 3.):
            self.assertAlmostEqual(v * 24., y, 3)

    def test_a_gentle_face_keeps_the_plan_mapping(self):
        # 3 m over 10 m: 17 degrees
        for x, y, z, u, v in self.rows(True, 3., 10.):
            self.assertAlmostEqual(u * 24., x, 3)
            self.assertAlmostEqual(v * 24., y, 3)


@unittest.skipUnless(os.environ.get('E2R_TEST_PACK') and G.lupa, 'set E2R_TEST_PACK (and E2R_TEST_TILES) to run real tiles')
class StagedPack(unittest.TestCase):
    def test_gate_on_real_tiles(self):
        rt = G.Runtime(os.environ['E2R_TEST_PACK'])
        tiles = [tuple(int(v) for v in q.split(',')) for q in os.environ.get('E2R_TEST_TILES', '').split(';') if q.strip()]
        self.assertTrue(tiles, 'E2R_TEST_TILES="i,j;i,j"')
        for i, j in tiles:
            tile = rt.block(i, j, conformal=True)
            m = G.measure(tile)
            with self.subTest(tile=(i, j)):
                self.assertEqual(m['scope']['hole'], 0, m['scope']['hole_at'][:5])
                self.assertLessEqual(m['under_m2'], .005 * max(m['carriageway_m2'], 1.))
                # forks and merges: no carriageway lies on another at the same height
                self.assertLessEqual(m['road_on_road']['coplanar'], 5., m['road_on_road'])
                # no end wall or other upright concrete in a carriageway (zero tolerance)
                res = G.walls_in_road(tile)
                self.assertEqual(res['hits'], 0, res['found'][:5])

    def test_budget_on_real_tiles(self):
        # a game retest: 3 of 4 ON runs had 55-120 ms frames just after a high bridge, 0 of 3 OFF. The cause was a build stretch
        # of 100-220 ms without a pause point (the faces between stacked bands); ON may not exceed OFF by the BUDGET factors
        rt = G.Runtime(os.environ['E2R_TEST_PACK'])
        tiles = [tuple(int(v) for v in q.split(',')) for q in os.environ.get('E2R_TEST_TILES', '').split(';') if q.strip()]
        for i, j in tiles:
            with self.subTest(tile=(i, j)):
                off, on = rt.budget(i, j, False), rt.budget(i, j, True)
                self.assertEqual(G.check_budget(on, off), [], (off, on))


if __name__ == '__main__':
    unittest.main(argv=[a for a in sys.argv if a != '-v'], verbosity=2 if '-v' in sys.argv else 1)
