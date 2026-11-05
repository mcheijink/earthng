"""Tests of the bridge decks (generator.lua ROADS.parapetCap, deckEndCap, deckMaterial), in LuaJIT through
lupa on generator_budget.py's synthetic cell, whose service road crosses a bridge in tile 0,0.
Run: python earthng/bridge_test.py"""
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

TRIANGLES = r'''
return function(cells, options, material)
  options.pause = function() end
  local road, scene = GENERATOR.build(cells, 0, 0, options)
  local out = {}
  for _, m in ipairs({road, scene}) do
    for _, it in ipairs(m.items) do
      if it.kind == 'proc' then
        for _, mesh in ipairs(it.meshes) do
          if mesh.material == material then
            for k = 1, #mesh.faces, 3 do
              local t = {}
              for q = 0, 2 do
                local p = mesh.verts[mesh.faces[k + q].v + 1]
                t[#t + 1] = {p.x + it.pos[1], p.y + it.pos[2], p.z + it.pos[3]}
              end
              out[#out + 1] = t
            end
          end
        end
      end
    end
  end
  return out
end'''


def ray(o, d, tri):
    """distance along d from o to triangle tri, or None (Moller-Trumbore)"""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    e1, e2 = (bx - ax, by - ay, bz - az), (cx - ax, cy - ay, cz - az)
    p = (d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0])
    det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2]
    if abs(det) < 1e-12:
        return None
    s = (o[0] - ax, o[1] - ay, o[2] - az)
    u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det
    q = (s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0])
    v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det
    if u < 0 or v < 0 or u + v > 1:
        return None
    t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det
    return t if t > 1e-6 else None


@unittest.skipIf(gb is None, 'lupa.luajit21 not installed')
class Decks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L, _ = gb.lua_runtime()
        cls.L.globals().REPO_ROOT = ROOT
        cls.L.globals().load(gb.PRELUDE, '=harness')()

    def triangles(self, extra_tags=None):
        cell, _ = gb.synthetic_cell(10, random.Random(1))
        for r in cell['roads']:
            if r.get('tags', {}).get('bridge') not in (None, 'no'):
                r['tags'].update(extra_tags or {})
        cells = gb.to_lua(self.L, [cell])

        class A:
            far = False
            no_height_pipeline = False
        res = self.L.execute(TRIANGLES)(cells, gb.tile_options(self.L, A, None), 'e2rf_fence_wall')
        return [[tuple(res[i][q][c] for c in (1, 2, 3)) for q in (1, 2, 3)] for i in range(1, len(res) + 1)]

    def test_deck_ends_are_closed(self):
        # the bridge runs along y at x 300 from y 20 to 140, its deck sloping; rays from 2 m off each end, in
        # the parapets' band and through the slab, must stop at the end face
        tris = self.triangles()
        for yend, d in ((20.0, 1), (140.0, -1)):
            top = max(p[2] for t in tris for p in t if abs(p[1] - yend) < .3 and 297 < p[0] < 303)
            surface = top - .9
            rays = [(x, surface + .45) for x in (297.65, 302.35)] + [(x, z) for x in (298.5, 300, 301.5) for z in (surface - .3, surface - .7)]
            for x, z in rays:
                hits = [t for t in (ray((x, yend - 2 * d, z), (0, d, 0), tr) for tr in tris) if t is not None]
                self.assertTrue(hits and abs(min(hits) - 2) < .05, 'open at y %g, x %g, z %.2f' % (yend, x, z))

    def test_material_by_tags(self):
        m = self.L.globals().GENERATOR.DECKMATERIAL
        self.assertEqual(m(self.L.table_from({})), 'fence_wall')
        self.assertEqual(m(self.L.table_from({'bridge:structure': 'arch'})), 'hw_stone')
        self.assertEqual(m(self.L.table_from({'bridge:structure': 'truss'})), 'clad_metal')
        # material wins over the structure: a concrete arch stays concrete
        self.assertEqual(m(self.L.table_from({'bridge:structure': 'arch', 'material': 'concrete'})), 'fence_wall')
        self.assertEqual(m(self.L.table_from({'material': 'brick'})), 'hw_oldbrick')
        self.assertEqual(m(self.L.table_from({'bridge:structure': 'beam'})), 'fence_wall')

    def test_a_tagged_bridge_leaves_no_concrete(self):
        self.assertEqual(self.triangles({'material': 'stone'}), [])


if __name__ == '__main__':
    unittest.main()
