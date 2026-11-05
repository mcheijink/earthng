"""Traffic signals and speed cameras through the runtime generator (generator.lua + signals.lua) in LuaJIT.

Needs lupa (lupa.luajit21); skipped without it. Run: python earthng/signals_lua_test.py
The roads are synthetic, on flat ground round a tile's centre."""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import lupa.luajit21 as lupa
except ImportError:      # pragma: no cover
    lupa = None

C = (250., 250.)


def road(rid, pts, nodes, width, **tags):
    return {'id': rid, 'points': [[x, y, 0.] for x, y in pts], 'nodes': nodes, 'width': width, 'tags': tags}


def point(pid, kind, xy, **tags):
    return {'id': pid, 'kind': kind, 'tags': tags, 'pos': [xy[0], xy[1], 0.]}


class Runtime:
    def __init__(self, missing=()):
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(f"package.path = '{ROOT}/mod/lua/?.lua;' .. package.path")
        if missing:
            gone = ','.join(f"['{p}'] = true" for p in missing)
            self.lua.execute("FS = {fileExists = function(self, p) return not (%s)[p] end, gone = {%s}}"
                             % ('({' + gone + '})', gone))
        self.gen = self.lua.eval(f"dofile('{ROOT}/mod/lua/ge/extensions/e2r/generator.lua')")
        self.signals = self.lua.eval("require('ge/extensions/e2r/signals')")

    def table(self, v):
        if isinstance(v, dict):
            t = self.lua.table()
            for k, x in v.items():
                t[k] = self.table(x)
            return t
        if isinstance(v, (list, tuple)):
            t = self.lua.table()
            for i, x in enumerate(v):
                t[i + 1] = self.table(x)
            return t
        return v

    def build(self, roads, points):
        cell = self.table({'roads': roads, 'points': points})
        result = self.gen.build(self.lua.table(cell), 0, 0, self.table({'tileSize': 500}))
        scene = result[1]
        out = []
        for item in scene['items'].values():
            name = str(item['name'])
            if item['kind'] == 'static' and ('_sig' in name or '_cam' in name):
                yaw = 2 * math.atan2(item['rot'][3], item['rot'][4])
                fields = dict(item['fields'].items()) if item['fields'] else None
                out.append({'name': name.split('_', 3)[-1], 'shape': str(item['shape']), 'pos': tuple(item['pos'].values()),
                            'yaw': yaw, 'fields': fields, 'prop': bool(item['prop']), 'collision': item['collision']})
        return out


def local_y(yaw):
    """the direction the shape's local +Y turns to (generator.lua: X to (cos, -sin), Y to (sin, cos))"""
    return math.sin(yaw), math.cos(yaw)


def local_x(yaw):
    return math.cos(yaw), -math.sin(yaw)


def crossroads():
    """a wide primary east-west and a residential north-south street crossing at C (node 1)"""
    return [road('p0', [C, (C[0] + 120, C[1])], [1, 11], 14, highway='primary', lanes='4'),
            road('p1', [C, (C[0] - 120, C[1])], [1, 12], 14, highway='primary', lanes='4'),
            road('r0', [C, (C[0], C[1] + 120)], [1, 13], 7, highway='residential'),
            road('r1', [C, (C[0], C[1] - 120)], [1, 14], 7, highway='residential')]


def off_carriageways(test, items, roads):
    for it in items:
        x, y = it['pos'][0], it['pos'][1]
        for r in roads:
            (ax, ay, _), (bx, by, _) = r['points'][0], r['points'][-1]
            dx, dy = bx - ax, by - ay
            t = max(0., min(1., ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(x - ax - dx * t, y - ay - dy * t)
            test.assertGreater(d, r['width'] / 2, f"{it['name']} stands on {r['id']}")


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Signals(unittest.TestCase):
    def test_junction_node_gets_one_signal_per_approach(self):
        rt = Runtime()
        roads = crossroads()
        items = rt.build(roads, [point('s1', 'signal', C, highway='traffic_signals')])
        booms = [i for i in items if i['shape'].endswith('s_trafficlight_boom_sn.dae')]
        heads = [i for i in items if 'trafficlight_city1' in i['shape']]
        poles = [i for i in items if i['shape'].endswith('pole_traffic1.dae')]
        self.assertEqual((len(booms), len(heads), len(poles)), (2, 2, 2))
        off_carriageways(self, items, roads)
        for b in booms:
            # the boom stands right of the travel into the junction, its +Y along the travel
            tx, ty = local_y(b['yaw'])
            self.assertAlmostEqual(abs(tx), 1, delta=.02)
            rx, ry = ty, -tx
            self.assertGreater((b['pos'][0] - C[0]) * rx + (b['pos'][1] - C[1]) * ry, 7)
            self.assertLess((b['pos'][0] - C[0]) * tx + (b['pos'][1] - C[1]) * ty, 0)   # before the node
            self.assertEqual(b['fields']['instanceColor2'], '1 1 1 1')                   # the primary: green
            self.assertFalse(b['prop'])
        for h in heads:
            # the lenses (+X) face the traffic coming up the residential street; red
            lx, ly = local_x(h['yaw'])
            self.assertAlmostEqual(abs(ly), 1, delta=.02)
            self.assertGreater((h['pos'][0] - C[0]) * lx + (h['pos'][1] - C[1]) * ly, 0)   # before the node, facing back up
            self.assertEqual(h['fields']['instanceColor'], '1 1 1 1')
            self.assertAlmostEqual(h['pos'][2] - min(p['pos'][2] for p in poles), 3., delta=.3)
        for p in poles:
            self.assertTrue(p['prop'])
            self.assertEqual(p['collision'], 'Collision Mesh')

    def test_a_stop_line_signal_after_the_junction_one_adds_nothing(self):
        rt = Runtime()
        roads = crossroads()
        items = rt.build(roads, [point('s1', 'signal', C, highway='traffic_signals'),
                                 point('s2', 'signal', (C[0] - 12, C[1]), highway='traffic_signals', **{'e2r:heading': '90'})])
        self.assertEqual(len([i for i in items if 'boom' in i['shape']]), 2)

    def test_mid_block_signal_with_a_heading(self):
        rt = Runtime()
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 7, highway='residential')]
        items = rt.build(roads, [point('s', 'signal', (250, 250), highway='traffic_signals', **{'e2r:heading': '90.0'})])
        poles = [i for i in items if 'pole_traffic1' in i['shape']]
        self.assertEqual(len(poles), 1)
        self.assertLess(poles[0]['pos'][1], 250 - 3.5)          # right of eastbound traffic: the south side
        self.assertEqual(len([i for i in items if 'boom' in i['shape']]), 0)
        off_carriageways(self, items, roads)

    def test_mid_block_signal_without_a_direction_serves_both_ways(self):
        rt = Runtime()
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 7, highway='residential')]
        items = rt.build(roads, [point('s', 'signal', (250, 250), highway='traffic_signals')])
        poles = sorted(i['pos'][1] for i in items if 'pole_traffic1' in i['shape'])
        self.assertEqual(len(poles), 2)
        self.assertLess(poles[0], 250 - 3.5)
        self.assertGreater(poles[1], 250 + 3.5)

    def test_one_way_street(self):
        rt = Runtime()
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 7, highway='residential', oneway='-1')]
        items = rt.build(roads, [point('s', 'signal', (250, 250), highway='traffic_signals')])
        poles = [i for i in items if 'pole_traffic1' in i['shape']]
        self.assertEqual(len(poles), 1)
        self.assertGreater(poles[0]['pos'][1], 250 + 3.5)       # westbound: the north side

    def test_speed_camera_on_the_verge_facing_the_traffic(self):
        rt = Runtime()
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 8, highway='secondary')]
        items = rt.build(roads, [point('c', 'speed_camera', (250, 250), highway='speed_camera', direction='W')])
        parts = sorted((i for i in items if '_cam' in '_' + i['name']), key=lambda i: i['pos'][2])
        self.assertEqual([p['shape'].rsplit('/', 1)[-1] for p in parts],
                         ['s_camera_pole.dae', 's_camera_elbow.dae', 's_camera_speed1.dae'])
        base = parts[0]['pos']
        self.assertGreater(base[1], 250 + 4 + 1)                # right of westbound traffic: the north side
        self.assertAlmostEqual(parts[2]['pos'][2] - base[2], 4.988, places=3)
        # the lens (-Y) looks east, upstream of westbound traffic, and a little toward the road
        lx, ly = local_y(parts[2]['yaw'])
        self.assertGreater(-lx, .9)
        self.assertLess(-ly, 0)
        self.assertEqual(len({round(p['yaw'], 6) for p in parts}), 1)
        off_carriageways(self, items, roads)

    def test_camera_beside_the_road_without_direction(self):
        rt = Runtime()
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 8, highway='secondary')]
        items = rt.build(roads, [point('c', 'speed_camera', (250, 243), highway='speed_camera')])
        base = min((i for i in items if 's_camera_pole' in i['shape']), key=lambda i: i['pos'][2])
        self.assertLess(base['pos'][1], 250 - 4)                # it stays on the side it was mapped on

    def test_fallbacks_when_level_shapes_are_missing(self):
        wco = '/levels/west_coast_usa/art/shapes/objects/'
        roads = [road('a', [(100, 250), (400, 250)], [1, 2], 7, highway='residential')]
        pts = [point('s', 'signal', (250, 250), highway='traffic_signals', **{'e2r:heading': '90'}),
               point('c', 'speed_camera', (300, 250), highway='speed_camera', direction='E')]
        # no West Coast USA: the common boom for every signal, no camera
        items = Runtime(missing=[wco + 'pole_traffic1.dae', wco + 's_camera_pole.dae']).build(roads, pts)
        self.assertEqual([i['shape'].rsplit('/', 1)[-1] for i in items], ['s_trafficlight_boom_sn.dae'])
        # no boom either: nothing at all
        items = Runtime(missing=[wco + 'pole_traffic1.dae', wco + 's_camera_pole.dae',
                                 '/art/shapes/objects/s_trafficlight_boom_sn.dae']).build(roads, pts)
        self.assertEqual(items, [])

    def test_swept_block_approach_and_a_node_on_a_swept_road(self):
        rt = Runtime()
        out = rt.lua.execute("""
          local signals = require('ge/extensions/e2r/signals')
          -- a stub stroke along +x: centre y = 0, 3.5 m either side, a signal approach for +x travel at s = 50
          local S = {highway = 'secondary', oneway = false, approaches = {
            {control = 'signals', stop = 50, sigma = 1, lanes = 1, J = {id = 'j1'}},
            {control = 'give_way', stop = 150, sigma = -1, lanes = 1, J = {id = 'j2'}}}}
          function S:at(s) return {x = s, y = 0, z = 0, h = 0, eL = 3.5, eR = 3.5} end
          local put = {}
          local ctx = {rect = {0, -250, 500, 250}, strips = {}, stars = {}, ground = function() return 0 end, sidewalk = 2,
            oneway = function() return false, false end, rankOf = function() return 3 end, pause = function() end,
            SW = {strokes = {s1 = S}},
            sweep = {surfaceWhere = function(plan, x, y) if math.abs(y) < 3.5 then return 0, S, x end end},
            offroad = {nearAsphalt = function() return false end, nearestOn = function() return nil end,
              underDeck = function() return false end, walkAt = function() return 0 end,
              sweptClear = function(x, y, z) if math.abs(y) < 3.5 + 1.2 then return nil end return x, y, z end},
            points = {{id = 'n1', kind = 'signal', tags = {}, pos = {60, 0, 0}},
                      {id = 'n2', kind = 'signal', tags = {['e2r:heading'] = '270'}, pos = {300, 1, 0}},
                      {id = 'c1', kind = 'speed_camera', tags = {}, pos = {400, -2, 0}}},
            put = function(name, shape, x, y, z, yaw) put[#put + 1] = {name, shape:match('[^/]+$'), x, y, z, yaw} end}
          local n = signals.place(ctx)
          local rows = {}
          for _, p in ipairs(put) do rows[#rows + 1] = string.format('%s %s %.2f %.2f %.2f', p[1], p[2], p[3], p[4], p[5]) end
          return n .. '|' .. table.concat(rows, ';')
        """)
        count, rows = out.split('|')
        rows = [r.split() for r in rows.split(';')]
        self.assertEqual(count, '3')
        # the block's approach: at its stop line, 3.5 + 1.2 m right of +x travel; the node near it adds nothing
        self.assertEqual(rows[0][0], 'sigp_j1_s1_f')
        self.assertEqual((float(rows[0][2]), float(rows[0][3])), (50., -4.7))
        # the westbound node mid-stroke: north side
        n2 = [r for r in rows if r[0].startswith('sigp_n2')]
        self.assertEqual(len(n2), 1)
        self.assertGreater(float(n2[0][3]), 3.5)
        self.assertFalse([r for r in rows if '_n1' in r[0]])
        # the camera beside the eastbound lanes (south of the centre): south verge
        cams = [r for r in rows if r[0].startswith('cam')]
        self.assertEqual(len(cams), 3)
        self.assertLess(float(cams[0][3]), -3.5 - 1.5 + 1e-9)

    def test_kit_reports_missing_paths(self):
        rt = Runtime(missing=['/levels/west_coast_usa/art/shapes/objects/s_camera_elbow.dae'])
        kit = rt.signals.kit()
        self.assertFalse(kit['camera'])
        self.assertEqual(list(kit['missing'].values()), ['/levels/west_coast_usa/art/shapes/objects/s_camera_elbow.dae'])


if __name__ == '__main__':
    unittest.main()
