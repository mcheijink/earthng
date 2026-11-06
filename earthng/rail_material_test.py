"""Tests of the guard rails' material and UVs (sweep.lua M.RAIL, M.railLook, M.railMesh; generator.lua ROADS.rail,
ROADS.guardRail), in LuaJIT through lupa (pip install lupa): the stock game material is the look of the beam and the posts,
one panel of the stock sheet every 4 m of u (a joint on every post) and its W strip in v; without it the rail falls back (logged once) to our own
cladding with the old UVs; the posts stand behind the beam, seen from the traffic, one-faced and median.
Run: python earthng/rail_material_test.py"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import generator_budget as gb
    from lupa import luajit21  # noqa: F401
except ImportError:  # pragma: no cover
    gb = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STOCK = 'italy_guardrails'
OWN = 'e2rf_clad_metal'
SHEET_PATH = '/assets/materials/trim/metal/m_italy_guardrails/'

# a fresh sweep.lua with engine globals stubbed as the test says: calls (log, jsonWriteFile, loadJsonMaterialsFile) are kept in CALLS
SWEEP = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
CALLS = {log = {}, write = {}, load = {}}
return function(setup)
  loadstring(setup)()
  return require('ge/extensions/e2r/sweep')
end'''

# a straight beam along +x from s 0 to `length`, a station every 2 m, at y 0; the triangles' corners with uv
BEAM = r'''
return function(SW, kind, side, double, length, look)
  local pts = {}
  for s = 0, length, 2 do pts[#pts + 1] = {s, 0, 0, s, 0} end
  local mesh = {verts = {}, normals = {}, uvs = {}, faces = {}, material = 'm', vmap = {}, vkz = {}, nmap = {}, umap = {}, tris = 0}
  SW.railMesh({}, {}, {s = {0, length}, n = 2}, {kind = kind, side = side, double = double, points = pts, a = 0, b = length},
    mesh, {0, 0, 0}, 0, look)
  local out = {}
  for k = 1, #mesh.faces, 3 do
    local t = {}
    for q = 0, 2 do
      local f = mesh.faces[k + q]
      local p, uv = mesh.verts[f.v + 1], mesh.uvs[f.u + 1]
      t[#t + 1] = {p.x, p.y, p.z, uv.u, uv.v}
    end
    out[#out + 1] = t
  end
  return out
end'''

TRIANGLES = r'''
return function(cells, options)
  options.pause = function() end
  local road, scene = GENERATOR.build(cells, 0, 0, options)
  local out, names = {}, {}
  for _, m in ipairs({road, scene}) do
    for _, it in ipairs(m.items) do
      if it.kind == 'proc' then
        for _, mesh in ipairs(it.meshes) do
          names[mesh.material] = true
          if mesh.material == 'italy_guardrails' or mesh.material == 'e2rf_clad_metal' or mesh.material == 'e2rf_guardrail' then
            for k = 1, #mesh.faces, 3 do
              local t = {}
              for q = 0, 2 do
                local f = mesh.faces[k + q]
                local p, uv = mesh.verts[f.v + 1], mesh.uvs[f.u + 1]
                t[#t + 1] = {p.x + it.pos[1], p.y + it.pos[2], p.z + it.pos[3], uv.u, uv.v, mesh.material}
              end
              out[#out + 1] = t
            end
          end
        end
      end
    end
  end
  return out, names
end'''


def tris_of(res, n=6):
    return [[tuple(res[i][q][c] for c in range(1, n + 1)) for q in (1, 2, 3)] for i in range(1, len(res) + 1)]


@unittest.skipIf(gb is None, 'lupa.luajit21 not installed')
class Look(unittest.TestCase):
    """M.railLook: stock material by name, our own definition from the game's texture paths, the fallback; one log"""

    def look(self, setup, material=None):
        L, _ = gb.lua_runtime()
        L.globals().REPO_ROOT = ROOT
        SW = L.execute(SWEEP)(setup)
        R = SW.RAIL
        if material:
            R = L.table_from({'material': material, 'fallback': 'clad_metal'})
        look = SW.railLook(R)
        again = SW.railLook(R)
        self.assertTrue(L.eval('function(a, b) return a == b end')(look, again))          # one decision a session
        return L, SW, look

    def test_stock_material_when_the_game_has_it(self):
        L, SW, look = self.look("scenetree = {findObject = function(n) if n == '%s' then return {} end end}" % STOCK)
        self.assertEqual((look.tier, look.stock, look.name, look.kind), (1, True, STOCK, '=' + STOCK))
        self.assertEqual(SW.RAIL.material, STOCK)       # the default of the map is the stock material
        self.assertEqual(SW.RAIL.fallback, 'clad_metal')

    def test_offline_takes_the_stock_material(self):
        L, SW, look = self.look('')
        self.assertEqual((look.tier, look.name), (1, STOCK))

    def test_missing_falls_back_to_our_cladding_and_logs_once(self):
        L, SW, look = self.look("scenetree = {findObject = function() return nil end}; "
                                "log = function(level, tag, text) CALLS.log[#CALLS.log + 1] = level .. ' ' .. text end")
        self.assertEqual((look.tier, look.stock, look.name, look.kind), (3, False, OWN, 'clad_metal'))
        SW.railLook(SW.RAIL)
        SW.railLook(SW.RAIL)
        logs = L.globals().CALLS.log
        self.assertEqual(len(logs), 1)
        self.assertIn(STOCK, logs[1])
        self.assertTrue(logs[1].startswith('W'))

    def test_own_definition_from_the_game_texture_paths(self):
        # the stock definition is missing but its sheet is there: e2rf_guardrail pointing at the game's own files
        L, SW, look = self.look(
            "scenetree = {findObject = function() return nil end}; "
            "FS = {fileExists = function(self, p) return p:sub(1, 8) == '/assets/' and p:sub(-4) == '.dds' end}; "
            "jsonWriteFile = function(path, t) CALLS.write[#CALLS.write + 1] = {path = path, t = t} end; "
            "loadJsonMaterialsFile = function(path) CALLS.load[#CALLS.load + 1] = path end")
        self.assertEqual((look.tier, look.stock, look.name, look.kind), (2, True, 'e2rf_guardrail', 'guardrail'))
        calls = L.globals().CALLS
        self.assertEqual(len(calls.write), 1)
        self.assertEqual(len(calls.load), 1)
        stage = calls.write[1].t['e2rf_guardrail'].Stages[1]
        for field in ('baseColorMap', 'normalMap', 'metallicMap', 'roughnessMap'):
            self.assertTrue(stage[field].startswith(SHEET_PATH), stage[field])
        # a map the game lacks: no definition, the fallback
        L, SW, look = self.look(
            "scenetree = {findObject = function() return nil end}; "
            "FS = {fileExists = function(self, p) return p:find('_nm%.normal') == nil end}; "
            "jsonWriteFile = function() end; loadJsonMaterialsFile = function() end")
        self.assertEqual((look.tier, look.name), (3, OWN))

    def test_switched_off_never_asks(self):
        L, SW, look = self.look("scenetree = {findObject = function() error('asked') end}", material='clad_metal')
        self.assertEqual((look.tier, look.stock, look.name), (3, False, OWN))


@unittest.skipIf(gb is None, 'lupa.luajit21 not installed')
class SweptBeam(unittest.TestCase):
    """sweep.lua M.railMesh: u along the rail, v across the profile, the posts behind the beam"""

    @classmethod
    def setUpClass(cls):
        cls.L, _ = gb.lua_runtime()
        cls.L.globals().REPO_ROOT = ROOT
        cls.SW = cls.L.execute(SWEEP)('')
        cls.beam = cls.L.execute(BEAM)
        cls.look = cls.SW.railLook(cls.SW.RAIL)

    def tris(self, kind='beam', side=1, double=False, length=40., look=True):
        return tris_of(self.beam(self.SW, kind, side, double, length, self.look if look else None), 5)

    @staticmethod
    def split(tris):
        """(post triangles, beam triangles): a post reaches under the base (its sink), the beam stays over it"""
        posts = [t for t in tris if min(p[2] for p in t) < 0]
        return posts, [t for t in tris if min(p[2] for p in t) >= 0]

    def test_u_runs_along_the_rail_one_panel_every_post_spacing(self):
        posts, beam = self.split(self.tris())
        us = [p[3] for t in beam for p in t]
        self.assertAlmostEqual(min(us), 0., places=6)
        self.assertAlmostEqual(max(us), 40. / self.look.panel, places=6)
        self.assertEqual(self.look.panel, 4)
        for t in beam:                              # u grows by 1 over every 4 m of the rail
            for a in t:
                for b in t:
                    if abs(a[2] - b[2]) < 1e-9 and abs(a[0] - b[0]) > 1e-6:
                        self.assertAlmostEqual((b[3] - a[3]) / (b[0] - a[0]), 1 / 4., places=6)

    def test_v_covers_the_w_strip_of_the_sheet(self):
        posts, beam = self.split(self.tris())
        vs = [p[4] for t in beam for p in t]
        self.assertAlmostEqual(max(vs), .441, places=6)      # the beam's foot
        self.assertAlmostEqual(min(vs), .342, places=6)      # its top: 350 to 451 px of 1024, the stock mesh's strip
        # monotonic with height: lower on the beam, larger v (v runs down the image)
        for t in beam:
            for a in t:
                for b in t:
                    if a[2] > b[2] + 1e-6:
                        self.assertLess(a[4], b[4] + 1e-9)

    def test_posts_sample_the_plain_patch(self):
        posts, _ = self.split(self.tris())
        self.assertEqual(len({round(p[0], 3) for t in posts for p in t if p[2] < 0}), 11 * 2)    # 11 posts, 2 x of their foot
        for t in posts:
            for p in t:
                self.assertTrue(.39 - 1e-9 <= p[3] <= .63 + 1e-9 and .285 - 1e-9 <= p[4] <= .332 + 1e-9, p)
        # the foot at the larger v (down the image), the top at the smaller
        self.assertTrue(all(p[4] > .33 for t in posts for p in t if p[2] < 0))
        self.assertTrue(all(p[4] < .29 for t in posts for p in t if p[2] > .5))

    def test_the_old_look_keeps_its_uvs(self):
        posts, beam = self.split(self.tris(look=False))
        self.assertAlmostEqual(max(p[3] for t in beam for p in t), 40. / 3, places=6)
        self.assertAlmostEqual(max(p[4] for t in beam for p in t), .75 / 3, places=6)
        self.assertAlmostEqual(max(p[3] for t in posts for p in t), .14, places=6)
        self.assertAlmostEqual(max(p[4] for t in posts for p in t), .75, places=6)

    def test_posts_stand_behind_the_beam_seen_from_the_traffic(self):
        # side +1: the rail on the left of the stroke, the road on its right (y < 0); the beam's face is towards the road
        for side in (1, -1):
            posts, beam = self.split(self.tris(side=side))
            traffic = -side                              # the sign of y towards the carriageway
            toward = max(traffic * p[1] for t in beam for p in t)       # the beam's road-side extent
            self.assertGreater(toward, 0)
            for t in posts:
                for p in t:
                    self.assertLessEqual(traffic * p[1], -self.SW.RAIL.thick + 1e-6, 'a post on the traffic side, side %d' % side)

    def test_a_median_rail_has_a_beam_each_side_and_the_posts_in_the_middle(self):
        posts, beam = self.split(self.tris(double=True))
        ys = [p[1] for t in beam for p in t]
        self.assertGreater(max(ys), .05)
        self.assertLess(min(ys), -.05)
        for t in posts:
            self.assertLessEqual(max(abs(p[1]) for p in t), self.SW.RAIL.post / 2 + 1e-6)
        # the beams clear the posts' faces
        self.assertGreaterEqual(min(abs(p[1]) for t in beam for p in t), self.SW.RAIL.post / 2 - 1e-6)

    def test_a_concrete_barrier_keeps_its_own_uvs(self):
        tris = self.tris(kind='jersey')
        self.assertAlmostEqual(max(p[3] for t in tris for p in t), 40. / 3, places=6)       # s / 3, the old rule
        self.assertAlmostEqual(max(p[4] for t in tris for p in t), .81 / 3, places=6)


@unittest.skipIf(gb is None, 'lupa.luajit21 not installed')
class MappedRail(unittest.TestCase):
    """generator.lua ROADS.guardRail on generator_budget's synthetic cell: a guard rail 7.5 m north of the motorway (12 m wide)"""

    def build(self, setup=''):
        L, _ = gb.lua_runtime()
        L.globals().REPO_ROOT = ROOT
        L.execute(setup or 'local _ = 0')
        L.globals().load(gb.PRELUDE, '=harness')()
        cell, _ = gb.synthetic_cell(10, random.Random(1))
        cells = gb.to_lua(L, [cell])

        class A:
            far = False
            no_height_pipeline = False
        res, names = L.execute(TRIANGLES)(cells, gb.tile_options(L, A, None))
        return tris_of(res), {k for k in names.keys()}

    @staticmethod
    def centre(x):
        import math
        return 80 + 30 * math.sin(x / 300)

    def parts(self, tris):
        # a post's side: under 0.2 m across and taller than the beam (the beam's end caps are narrow too, but 0.32 m tall)
        posts = [t for t in tris if max(p[0] for p in t) - min(p[0] for p in t) < .2 and max(p[1] for p in t) - min(p[1] for p in t) < .2
                 and max(p[2] for p in t) - min(p[2] for p in t) > .5]
        beam = [t for t in tris if t not in posts]
        return posts, beam

    def test_stock_material_uvs_and_posts_behind(self):
        tris, names = self.build()
        self.assertIn(STOCK, names)
        self.assertNotIn('e2rf_guardrail', names)
        tris = [t for t in tris if t[0][5] == STOCK]
        posts, beam = self.parts(tris)
        self.assertGreater(len(posts), 100)
        self.assertGreater(len(beam), 100)
        # u: one panel every post spacing (4 m) along the rail (the edges between stations run along it)
        ratios = []
        for t in beam:
            for a in t:
                for b in t:
                    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5
                    if d > .5 and abs(a[3] - b[3]) > 1e-9:
                        ratios.append(d / abs(a[3] - b[3]))
        self.assertGreater(len(ratios), 100)
        self.assertTrue(all(3.85 < r < 4.15 for r in ratios), (min(ratios), max(ratios)))
        # v: the beam in the W strip (.342-.441), the posts in their patch
        vb = [p[4] for t in beam for p in t]
        self.assertTrue(min(vb) >= .342 - 2e-3 and max(vb) <= .441 + 2e-3, (min(vb), max(vb)))
        for t in posts:
            for p in t:
                self.assertTrue(.39 - 1e-3 <= p[3] <= .63 + 1e-3 and .285 - 1e-3 <= p[4] <= .332 + 1e-3, p)
        # the posts stand on the far side of the beam from the carriageway (north: larger offset from its centre line)
        off = lambda t: sum(p[1] - self.centre(p[0]) for p in t) / 3
        beam_off = sum(off(t) for t in beam) / len(beam)
        post_off = sum(off(t) for t in posts) / len(posts)
        self.assertAlmostEqual(beam_off, 7.5, delta=.1)
        self.assertGreater(post_off - beam_off, .08, 'the posts are not behind the beam')

    def test_fallback_keeps_the_old_look(self):
        tris, names = self.build("scenetree = {findObject = function() return nil end}")
        self.assertNotIn(STOCK, names)
        tris = [t for t in tris if t[0][5] == OWN]
        posts, beam = self.parts(tris)
        self.assertGreater(len(beam), 100)
        # u = along / 4 and v a quarter of the beam's height at most: the old mapping
        ratios = []
        for t in beam:
            for a in t:
                for b in t:
                    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5
                    if d > .5 and abs(a[3] - b[3]) > 1e-9:
                        ratios.append(d / abs(a[3] - b[3]))
        self.assertTrue(all(3.8 < r < 4.2 for r in ratios), (min(ratios), max(ratios)))
        self.assertLessEqual(max(p[4] for t in beam for p in t), .25 + 1e-6)
        self.assertGreater(len(posts), 100)


if __name__ == '__main__':
    unittest.main()
