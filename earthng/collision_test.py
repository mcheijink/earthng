"""Tests of collision.lua (which meshes a be:reloadCollision keeps), in LuaJIT through lupa.
Run: python earthng/collision_test.py"""
import os
import unittest

try:
    from lupa import luajit21 as lupa
except ImportError:  # pragma: no cover
    lupa = None

COLLISION = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mod', 'lua', 'ge', 'extensions', 'e2r',
                         'collision.lua')


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Rebuild(unittest.TestCase):
    def setUp(self):
        self.lua = lupa.LuaRuntime()
        with open(COLLISION, encoding='utf-8') as f:
            self.lua.execute('collision = (function() ' + f.read() + ' end)()')

    def run_rebuild(self, setup):
        # objects: id -> {hidden, seen}; seen = hidden state while the engine rebuilds
        return self.lua.execute('''
        local objects = {}
        local function obj(id)
          local o = {id = id, hidden = false}
          function o:isHidden() return self.hidden end
          function o:setHidden(h) self.hidden = h end
          objects[id] = o
          return o
        end
        local c = collision.new()
        local near = {i = 0, j = 0}
        local far = {i = 0, j = 0, far = true}
        local away = {i = 4, j = 0, far = true}
        for id, owner in pairs({[1] = near, [2] = far, [3] = away}) do obj(id); c:track(owner, id, false) end
        obj(4); c:track(near, 4, true)
        ''' + setup + '''
        local seen = {}
        c:rebuild({{250, 250, 300, 500}}, 500, function(id) return objects[id] end,
          function() for id, o in pairs(objects) do seen[id] = o.hidden end end)
        local after = {}
        for id, o in pairs(objects) do after[id] = o.hidden end
        return seen, after, c.lastSolid
        ''')

    def test_far_tile_in_reach_collides_without_near_ground(self):
        seen, after, solid = self.run_rebuild('')
        self.assertFalse(seen[1])   # near solid
        self.assertFalse(seen[2])   # far ground in reach stays
        self.assertTrue(seen[3])    # far tile 2 km away is left out
        self.assertTrue(seen[4])    # quiet mesh (markings) is left out
        self.assertEqual(solid, 2)
        self.assertFalse(any(after.values()))  # everything is shown again

    def test_covered_far_tile_is_left_out(self):
        seen, after, solid = self.run_rebuild('far.covered = true')
        self.assertFalse(seen[1])
        self.assertTrue(seen[2])
        self.assertEqual(solid, 1)
        self.assertFalse(any(after.values()))

    def test_failed_rebuild_shows_everything(self):
        with self.assertRaises(Exception):
            self.lua.execute('''
            local o = {hidden = false}
            function o:isHidden() return self.hidden end
            function o:setHidden(h) self.hidden = h end
            shown = o
            local c = collision.new()
            c:track({i = 9, j = 9, far = true}, 1, false)
            c:rebuild({{0, 0, 10, 20}}, 500, function() return o end, function() error('engine') end)
            ''')
        self.assertFalse(self.lua.eval('shown.hidden'))


if __name__ == '__main__':
    unittest.main()
