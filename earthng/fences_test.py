"""Tests of fences, walls and hedges: fast_style's line kinds, and generator.lua under LuaJIT (pip install lupa; skipped
without it) on generator_budget's synthetic town cell with OSM hedges and walls added.
Run: python earthng/fences_test.py
"""
import os
import random
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_style  # noqa: E402
import generator_budget as GB  # noqa: E402


class Kinds(unittest.TestCase):
    def test_line_kinds(self):
        self.assertEqual(fast_style.line_kind({'barrier': 'hedge'}), 'hedge')
        for b in ('fence', 'wall', 'retaining_wall', 'city_wall'):
            self.assertEqual(fast_style.line_kind({'barrier': b}), 'fence', b)


def runtime():
    try:
        return GB.lua_runtime()[0]
    except SystemExit:
        return None


def pt(x, y, dz=0.):
    return [x, y, round(GB.terrain_z(x, y) + dz, 2)]


@unittest.skipIf(runtime() is None, 'needs LuaJIT: pip install lupa')
class Drawn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell, _ = GB.synthetic_cell(10, random.Random(1))
        lines = cell['lines']
        # a garden hedge inside the block between the streets x = 200..300, y = 200..300 (1.2 m tall), one across the street
        # at x = 300 (none of it on the road), a retaining wall and a city wall
        lines.append({'id': 'h1', 'kind': 'hedge', 'tags': {'barrier': 'hedge', 'height': '1.2'}, 'points': [pt(225, 250), pt(275, 250)]})
        lines.append({'id': 'h2', 'kind': 'hedge', 'tags': {'barrier': 'hedge'}, 'points': [pt(280, 340), pt(320, 340)]})
        lines.append({'id': 'w1', 'kind': 'fence', 'tags': {'barrier': 'retaining_wall'}, 'points': [pt(120, 120), pt(160, 120)]})
        lines.append({'id': 'w2', 'kind': 'fence', 'tags': {'barrier': 'city_wall', 'height': '6'}, 'points': [pt(120, 140), pt(160, 140)]})
        L = runtime()
        L.globals().REPO_ROOT = GB.ROOT
        L.globals().load(GB.PRELUDE, '=harness')()
        cells = GB.to_lua(L, [cell])
        args = types.SimpleNamespace(far=False, no_height_pipeline=False)
        build = L.eval('''function(cells, options)
          options.pause, options.mark = function() end, function() end
          local road, scene = GENERATOR.build(cells, 0, 0, options)
          local out = {}
          for _, it in ipairs(scene.items) do
            out[#out + 1] = {it.name or '', it.kind or '', it.shape or '', it.material or '', it.pos and it.pos[1] or 0,
              it.pos and it.pos[2] or 0, it.scale or 1, it.collision or ''}
          end
          -- the world (x, y) of every vertex of the walls' material
          local walls = {}
          for _, it in ipairs(scene.items) do
            if it.kind == 'proc' then
              for k, mat in ipairs(it.mats or {}) do
                if mat:find('fence_wall') then
                  for _, v in ipairs(it.meshes[k].verts) do walls[#walls + 1] = {v.x + it.pos[1], v.y + it.pos[2], v.z} end
                end
              end
            end
          end
          return out, walls
        end''')
        items, walls = build(cells, GB.tile_options(L, args, None))
        cls.items = [list(v.values()) for v in items.values()]
        cls.walls = [list(v.values()) for v in walls.values()]

    def test_hedges_along_their_lines(self):
        hedges = [it for it in self.items if '_hedge_' in it[0]]
        garden = [it for it in hedges if abs(it[5] - 250) < .01]
        # one piece every hedgeLong (2.38) x scale (1.2 / 3) m: 0.95 m apart, from 225 to 275
        self.assertGreaterEqual(len(garden), 48)
        xs = sorted(it[4] for it in garden)
        self.assertLess(max(b - a for a, b in zip(xs, xs[1:])), 1.0)
        self.assertTrue(all(abs(it[6] - .4) < 1e-6 for it in garden))
        self.assertTrue(all(it[7] == 'None' for it in hedges))
        # the one across the street: pieces on both sides, none within the residential road (6 m wide) at x = 300
        cross = [it for it in hedges if abs(it[5] - 340) < .01]
        self.assertTrue(any(it[4] < 296 for it in cross) and any(it[4] > 304 for it in cross))
        self.assertEqual([it for it in cross if abs(it[4] - 300) < 3.3], [])
        self.assertLessEqual(len(hedges), 600)

    def test_retaining_and_city_walls_are_drawn(self):
        # fence() draws them as concrete boxes (material fence_wall) along their lines; untagged, a retaining wall is 1 m tall,
        # a city wall's 6 m is held to the 3 m cap
        def wall(y):
            vs = [v for v in self.walls if abs(v[1] - y) < .3 and 115 <= v[0] <= 165]
            return vs, (max(v[2] - GB.terrain_z(v[0], v[1]) for v in vs) if vs else 0.)
        vs, top = wall(120)
        self.assertGreater(len(vs), 10)
        self.assertAlmostEqual(top, 1.0, delta=.35)
        vs, top = wall(140)
        self.assertGreater(len(vs), 10)
        self.assertAlmostEqual(top, 3.0, delta=.35)

if __name__ == '__main__':
    unittest.main()
