"""Power lines and wind turbines: the exporter's span runs (power.py) and the runtime meshes
(mod/lua/ge/extensions/e2r/power.lua, under LuaJIT through lupa when installed).
Run: python earthng/power_test.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import power  # noqa: E402

try:
    import lupa.luajit21 as lupa
except ImportError:  # pragma: no cover
    lupa = None

POWER_LUA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mod', 'lua', 'ge', 'extensions', 'e2r', 'power.lua')


class Exporter(unittest.TestCase):
    def test_tags(self):
        self.assertTrue(power.is_line({'power': 'line', 'voltage': '380000'}))
        self.assertFalse(power.is_line({'power': 'minor_line'}))
        self.assertFalse(power.is_line({'power': 'line', 'line': 'busbar'}))
        self.assertFalse(power.is_line({'power': 'line', 'location': 'underground'}))
        self.assertTrue(power.is_turbine({'power': 'generator', 'generator:source': 'wind'}))
        self.assertFalse(power.is_turbine({'power': 'generator', 'generator:source': 'solar'}))
        self.assertEqual(power.tower_attr({'power': 'tower', 'design': 'donau', 'height': '55'}), 't,donau,55')
        self.assertEqual(power.tower_attr({'power': 'pole', 'line_arrangement': 'flat'}), 'p,flat,')
        self.assertEqual(power.tower_attr(None), '')

    def test_runs_keep_whole_spans_and_one_more(self):
        # towers every 300 m along x; a padded cell from 950 to 1150 reaches the spans 900-1200
        coords = [(300.0 * k, 0.0) for k in range(10)]
        self.assertEqual(power.runs(coords, (950, -50, 1150, 50)), [(2, 5)])
        # at the way's start the run cannot grow backwards
        self.assertEqual(power.runs(coords, (-50, -50, 100, 50)), [(0, 2)])
        self.assertEqual(power.runs(coords, (5000, 5000, 5100, 5100)), [])

    def test_runs_merge_when_the_gap_is_short(self):
        # a way that leaves the cell for one span and comes back: one run
        coords = [(0, 0), (300, 0), (300, 300), (0, 300), (0, 600), (300, 600)]
        self.assertEqual(power.runs(coords, (-50, -50, 50, 650)), [(0, 5)])

    def test_line_runs_carry_ids_tower_facts_and_few_tags(self):
        coords = [(300.0 * k, 0.0) for k in range(6)]
        towers = {power.key(600.0, 0.0): {'power': 'tower', 'design': 'barrel', 'height': '42'}}
        tags = {'power': 'line', 'voltage': '110000', 'cables': '6', 'name': 'x', 'operator': 'y'}
        out = power.line_runs(77, 0, coords, tags, (650, -50, 700, 50), towers)
        self.assertEqual(len(out), 1)
        run = out[0]
        self.assertEqual(run['id'], '77:0:p1')
        self.assertEqual(run['xy'], [(300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0)])
        self.assertEqual(run['tags'][power.TOWERS], '|t,barrel,42||')
        self.assertNotIn('name', run['tags'])
        self.assertNotIn('operator', run['tags'])
        self.assertEqual(run['tags']['voltage'], '110000')


try:
    import numpy as np
    import shapely
    import build_country
except ImportError:  # pragma: no cover
    build_country = None


class _Flat:
    def sample(self, xy):
        return np.full(len(xy), 10.)


class _Store:
    """compile_cell's view of a source store: no roads, flat ground, scenery from a list"""
    terrain = _Flat()

    def __init__(self, records):
        self.records = records

    def query(self, rect, kinds=None):
        return []

    def scenery(self, rect):
        box = shapely.box(*rect)
        return [r for r in self.records if r[3].intersects(box)]


@unittest.skipIf(build_country is None, 'the exporter needs numpy and shapely')
class Cell(unittest.TestCase):
    def test_compile_cell_keeps_whole_spans_and_tower_facts(self):
        xs = [-1050. + 300 * k for k in range(12)]          # towers at -1050, -750, ... 2250
        records = [(7, 'power_line', {'tags': {'power': 'line', 'voltage': '380000', 'cables': '6', 'name': 'n'}},
                    shapely.LineString([(x, 500.) for x in xs]))]
        records += [(-(10**12 + 100 + k), 'power_tower', {'tags': {'power': 'tower', 'design': 'donau', 'height': str(50 + k)}},
                     shapely.Point(x, 500.)) for k, x in enumerate(xs)]
        records.append((-(10**12 + 99), 'wind_turbine', {'tags': {'power': 'generator', 'generator:source': 'wind',
                                                                  'height': '120', 'operator': 'o'}}, shapely.Point(400., 300.)))
        cell = build_country.compile_cell(_Store(records), [0., 0., 1000., 1000.])
        lines = [f for f in cell['lines'] if f['kind'] == 'power_line']
        self.assertEqual(len(lines), 1)
        line = lines[0]
        # the spans reaching the padded cell (-100..1100) are -150..150 (3) to 1050..1350 (7); one more each side
        self.assertEqual(line['id'], '7:0:p2')
        self.assertEqual([p[0] for p in line['points']], xs[2:10])
        self.assertTrue(all(p[2] == 10. for p in line['points']))
        self.assertEqual(line['tags'][power.TOWERS], '|'.join(f't,donau,{50 + k}' for k in range(2, 10)))
        self.assertNotIn('name', line['tags'])
        self.assertEqual([f['kind'] for f in cell['points']], ['wind_turbine'])
        self.assertEqual(cell['points'][0]['tags'], {'power': 'generator', 'generator:source': 'wind', 'height': '120'})
        # the binary cell format round-trips the run
        import country_format
        self.assertEqual(country_format.decode_cell(country_format.encode_cell(cell, (0., 0.)), (0., 0.), [0., 0., 1000., 1000.])['lines'][0]['tags'],
                         line['tags'])

    def test_power_layer_switch(self):
        records = [(7, 'power_line', {'tags': {'power': 'line'}}, shapely.LineString([(0., 500.), (900., 500.)]))]
        cell = build_country.compile_cell(_Store(records), [0., 0., 1000., 1000.], include={'power': False})
        self.assertEqual(cell['lines'], [])


HARNESS = r'''
local src = ...
local P = assert(loadstring(src, 'power.lua'))()
local function collect(rect, lines, points, far)
  local out = {}
  local ctx = {rect = rect, lines = lines, points = points, far = far,
    ground = function(x, y) return 10 end,
    triangle = function(kind, a, b, c)
      out[#out + 1] = {kind, a, b, c}
    end}
  local counts = P.build(ctx)
  counts.list = out
  return counts
end
return {P = P, collect = collect}
'''


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Runtime(unittest.TestCase):
    def setUp(self):
        self.lua = lupa.LuaRuntime()
        with open(POWER_LUA, encoding='utf-8') as f:
            self.h = self.lua.execute(HARNESS, f.read())

    def t(self, value):
        if isinstance(value, dict):
            t = self.lua.table()
            for k, v in value.items():
                t[k] = self.t(v)
            return t
        if isinstance(value, (list, tuple)):
            return self.lua.table(*[self.t(v) for v in value])
        return value

    def run_feature(self, fid, first, xs, facts, tags=None):
        tags = dict(tags or {'power': 'line', 'voltage': '380000', 'cables': '6'})
        tags[power.TOWERS] = '|'.join(facts)
        return {'id': f'{fid}:0:p{first}', 'kind': 'power_line', 'tags': tags,
                'points': [[x, 0.0, 10.0] for x in xs]}

    def build(self, rect, lines=(), points=(), far=False):
        return self.h.collect(self.t(list(rect)), self.t(list(lines)), self.t(list(points)), far)

    def test_each_tower_and_span_once_over_tiles_and_cells(self):
        xs = [100.0 + 300 * k for k in range(8)]           # towers at 100, 400, ... 2200
        facts = ['t,donau,'] * 8
        # two cells hold overlapping runs of the same way (different first vertices)
        a = self.run_feature(5, 0, xs[0:5], facts[0:5])
        b = self.run_feature(5, 2, xs[2:8], facts[2:8])
        towers = spans = 0
        for i in range(5):
            rect = (500.0 * i, -250.0, 500.0 * (i + 1), 250.0)
            got = self.build(rect, [a, b])
            towers += got['towers']
            spans += got['spans']
        self.assertEqual(towers, 8)
        self.assertEqual(spans, 7)

    def test_wire_ends_meet_the_tower_drawn_in_another_tile(self):
        xs = [100.0 + 300 * k for k in range(6)]
        facts = ['t,donau,'] * 6
        # the tile with tower 3 sees vertices 1..6, the tile with the span 3-4 only 2..5
        whole = self.t(self.run_feature(9, 0, xs, facts))
        part = self.t(self.run_feature(9, 1, xs[1:5], facts[1:5]))
        info = self.lua.table_from({'kv': 380, 'cables': 6})
        P = self.h.P
        mine = P.towerAt(whole.points, 4, self.lua.table(*[self.lua.table_from({'kind': 't', 'design': 'donau'})] * 6), info)
        theirs = P.towerAt(part.points, 3, self.lua.table(*[self.lua.table_from({'kind': 't', 'design': 'donau'})] * 4), info)
        for k in range(1, 7):
            self.assertAlmostEqual(mine.slots[k][1] * mine.ax, theirs.slots[k][1] * theirs.ax)
            self.assertAlmostEqual(mine.slots[k][2], theirs.slots[k][2])
        self.assertAlmostEqual(mine.ax, theirs.ax)
        self.assertAlmostEqual(mine.ay, theirs.ay)

    def test_wires_hang_between_slots_with_sag(self):
        xs = [0.0, 300.0]
        line = self.run_feature(3, 0, xs, ['t,donau,', 't,donau,'])
        got = self.build((-10.0, -50.0, 310.0, 50.0), [line])
        wires = [tri for tri in got['list'].values() if tri[1] == 'power_wire']
        self.assertGreater(len(wires), 0)
        zs = [corner[3] for tri in wires for corner in (tri[2], tri[3], tri[4])]
        # 380 kV Donau: lower slots at .70 * 58 m - 5 m insulator, sag 3 % of 300 m
        self.assertLess(min(zs), 10 + .70 * 58 - 5.0 - 8)
        self.assertGreater(min(zs), 10 + 6)

    def test_triangle_counts(self):
        counts = {}
        for design in ('donau', 'one-level', 'barrel'):
            for far in (False, True):
                line = self.run_feature(1, 0, [100.0], [f't,{design},'])
                line['points'].append([700.0, 0.0, 10.0])
                got = self.build((0.0, -50.0, 200.0, 50.0), [line], far=far)
                counts[(design, far)] = got['tris']
                self.assertEqual(got['towers'], 1)
        for (design, far), n in counts.items():
            self.assertLess(n, 260 if not far else 170, (design, far, n))
        span = self.run_feature(2, 0, [0.0, 350.0], ['t,donau,', 't,donau,'])
        near = self.build((100.0, -50.0, 250.0, 50.0), [span])
        far = self.build((100.0, -50.0, 250.0, 50.0), [span], far=True)
        self.assertEqual((near['towers'], near['spans']), (0, 1))
        self.assertLessEqual(near['tris'], 7 * 10 * 4)
        self.assertLessEqual(far['tris'], 7 * 4 * 2)
        turbine = {'id': 'w', 'kind': 'wind_turbine', 'pos': [50.0, 0.0, 10.0],
                   'tags': {'power': 'generator', 'generator:source': 'wind', 'height': '120', 'rotor:diameter': '115'}}
        got = self.build((0.0, -50.0, 200.0, 50.0), points=[turbine])
        self.assertEqual(got['turbines'], 1)
        self.assertLessEqual(got['tris'], 40)
        top = max(c[3] for tri in got['list'].values() for c in (tri[2], tri[3], tri[4]))
        # hub at 120 m over the ground, a blade tip at most the radius above it
        self.assertGreater(top, 10 + 120)
        self.assertLessEqual(top, 10 + 120 + 115 / 2 + .5)

    def test_pole_line_under_45_kv(self):
        line = self.run_feature(4, 0, [10.0, 120.0], ['', ''], {'power': 'line', 'voltage': '20000'})
        got = self.build((0.0, -50.0, 200.0, 50.0), [line])
        self.assertEqual(got['towers'], 2)
        self.assertLess(got['tris'], 2 * 16 + 3 * 10 * 4 + 1)


if __name__ == '__main__':
    unittest.main()
