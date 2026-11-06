"""Hedges, tree rows, mapped trees and road clear zones through the runtime generator in LuaJIT.

Needs lupa (lupa.luajit21); skipped without it. Run: python earthng/vegetation_lua_test.py
Everything is synthetic, on flat ground in one 500 m tile."""
import math
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')   # forward slashes: a Windows path in a Lua string needs no escapes
try:
    import lupa.luajit21 as lupa
except ImportError:      # pragma: no cover
    lupa = None

CLEAR = {'motorway': 6, 'trunk': 6, 'primary': 4.5, 'secondary': 4.5, 'tertiary': 4.5, 'unclassified': 2.5}


def road(rid, pts, width, **tags):
    return {'id': rid, 'points': [[x, y, 0.] for x, y in pts], 'nodes': [rid + 'a', rid + 'b'], 'width': width, 'tags': tags}


def line(lid, kind, pts, **tags):
    return {'id': lid, 'kind': kind, 'points': [[x, y, 0.] for x, y in pts], 'tags': tags}


def house(hid, x, y, size=10):
    ring = [[x, y, 0.], [x + size, y, 0.], [x + size, y + size, 0.], [x, y + size, 0.], [x, y, 0.]]
    return {'id': hid, 'kind': 'house', 'height': 6., 'minHeight': 0., 'tags': {'building': 'house'}, 'rings': [ring]}


class Runtime:
    def __init__(self):
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(f"package.path = '{ROOT}/mod/lua/?.lua;' .. package.path")
        self.gen = self.lua.eval(f"dofile('{ROOT}/mod/lua/ge/extensions/e2r/generator.lua')")

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

    def build(self, roads=(), lines=(), points=(), areas=(), buildings=(), **options):
        cell = self.table({'roads': list(roads), 'lines': list(lines), 'points': list(points), 'areas': list(areas),
                           'buildings': list(buildings)})
        opts = {'tileSize': 500, 'maxCars': 0, 'yard': False}
        opts.update(options)
        result = self.gen.build(self.lua.table(cell), 0, 0, self.table(opts))
        out = []
        for item in result[1]['items'].values():
            if item['kind'] == 'static':
                name = re.sub(r'^e2rl_0_0_', '', str(item['name']))
                out.append({'name': name, 'shape': str(item['shape']), 'x': item['pos'][1], 'y': item['pos'][2],
                            'yaw': 2 * math.atan2(item['rot'][3], item['rot'][4]), 'scale': item['scale']})
        return out


def distance_to(r, x, y):
    best = math.inf
    pts = r['points']
    for (ax, ay, _), (bx, by, _) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0., min(1., ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(x - ax - dx * t, y - ay - dy * t))
    return best


def vegetation(items):
    return [i for i in items if '/trees/' in i['shape'] or 'trees_library' in i['shape'] or 'scots_pine' in i['shape']]


@unittest.skipIf(lupa is None, 'lupa.luajit21 not installed')
class Vegetation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def test_rural_hedge_density_and_forest_path(self):
        # an east-west and a diagonal field hedge, far from any road
        hedges = [line('h1', 'hedge', [(50, 100), (450, 100)], barrier='hedge'),
                  line('h2', 'hedge', [(50, 150), (330, 430)], barrier='hedge')]
        items = self.rt.build(lines=hedges)
        straight = [i for i in items if i['name'].startswith('hedge_') and abs(i['y'] - 100) < 1]
        diagonal = [i for i in items if i['name'].startswith('hedge_') and abs(i['y'] - 100) >= 1]
        per_km = len(straight) / .4, len(diagonal) / (280 * math.sqrt(2) / 1000)
        print(f'\nrural hedge items per km: straight {per_km[0]:.0f}, diagonal {per_km[1]:.0f}')
        self.assertTrue(300 < per_km[0] < 520 and 220 < per_km[1] < 400, per_km)
        planted = straight + diagonal
        self.assertEqual(len(vegetation(planted)), len(planted))      # every one a tree or bush shape: Forest items
        trees = [i for i in straight if 'bush' not in i['shape']]
        self.assertTrue(0 < len(trees) < len(straight) / 5)            # a small tree now and then

    def test_village_hedge_is_trimmed_pieces_along_the_line(self):
        houses = [house(f'b{k}', 40 + 30 * k, 112) for k in range(14)] + [house(f'c{k}', 40 + 30 * k, 78) for k in range(14)]
        items = self.rt.build(lines=[line('h', 'hedge', [(50, 100), (450, 100)], barrier='hedge')], buildings=houses)
        # between the houses (the ends past the last houses are field hedge)
        pieces = [i for i in items if i['name'].startswith('hedge_') and 110 < i['x'] < 380]
        self.assertTrue(pieces)
        self.assertTrue(all('cypress_hedge' in i['shape'] for i in pieces))
        # the piece's long axis (local X: (cos, -sin) of the yaw) runs along the line
        self.assertTrue(all(abs(math.cos(i['yaw'])) > .999 for i in pieces))
        print(f'village hedge items per km: {len(pieces) / .27:.0f}')

    def test_hedge_and_tree_row_keep_out_of_the_clear_zone(self):
        roads = [road('p', [(0, 250), (500, 250)], 7, highway='primary')]
        lines = [line('h', 'hedge', [(30, 253), (470, 253)], barrier='hedge'),          # on the verge, 3 m off the centre
                 line('r', 'tree_row', [(30, 245), (470, 245)], natural='tree_row')]     # the other side, 5 m off
        items = self.rt.build(roads=roads, lines=lines)
        veg = [i for i in items if i['name'].startswith(('hedge_', 'tree_row_'))]
        self.assertTrue(len([i for i in veg if i['name'].startswith('hedge_')]) > 150)
        self.assertTrue(len([i for i in veg if i['name'].startswith('tree_row_')]) > 40)
        for i in veg:
            self.assertGreaterEqual(distance_to(roads[0], i['x'], i['y']), 3.5 + 4.5, i['name'])
        # they moved out to the zone's edge on their own side, not across the road
        self.assertTrue(all(i['y'] > 250 for i in veg if i['name'].startswith('hedge_')))
        self.assertTrue(all(i['y'] < 250 for i in veg if i['name'].startswith('tree_row_')))
        self.assertTrue(all(abs(abs(i['y'] - 250) - 8.05) < .01 for i in veg))

    def test_town_street_keeps_only_the_kerb_strip(self):
        roads = [road('s', [(0, 250), (500, 250)], 7, highway='residential')]
        # 0.5 m off the kerb line: out to 0.75 m
        items = self.rt.build(roads=roads, lines=[line('r', 'tree_row', [(30, 246), (470, 246)], natural='tree_row')])
        row = [i for i in items if i['name'].startswith('tree_row_')]
        self.assertTrue(row)
        self.assertTrue(all(abs(i['y'] - (250 - 3.5 - .75 - .05)) < .01 for i in row))

    def test_mapped_trees_species_and_clear_zone(self):
        roads = [road('t', [(0, 250), (500, 250)], 6, highway='tertiary')]
        points = [{'id': 'n1', 'kind': 'tree', 'tags': {'natural': 'tree', 'leaf_type': 'needleleaved'}, 'pos': [100, 254, 0]},
                  {'id': 'n2', 'kind': 'tree', 'tags': {'natural': 'tree', 'genus': 'Populus'}, 'pos': [200, 100, 0]},
                  {'id': 'n3', 'kind': 'tree', 'tags': {'natural': 'tree', 'leaf_type': 'broadleaved'}, 'pos': [300, 100, 0]},
                  {'id': 'n4', 'kind': 'tree', 'tags': {'natural': 'tree'}, 'pos': [400, 100, 0]},
                  {'id': 'n5', 'kind': 'tree', 'tags': {'natural': 'tree'}, 'pos': [450, 250.2, 0]}]  # on the asphalt: left out
        items = {i['name']: i for i in self.rt.build(roads=roads, points=points)}
        self.assertIn('scots_pine', items['point_n1']['shape'])
        self.assertGreaterEqual(distance_to(roads[0], items['point_n1']['x'], items['point_n1']['y']), 3 + 4.5)
        self.assertIn('poplar', items['point_n2']['shape'])
        self.assertIn('/trees/', items['point_n3']['shape'])
        self.assertTrue(items['point_n4']['shape'].endswith('tree_oak_sml_a.dae'))       # untagged: the oak as before
        self.assertNotIn('point_n5', items)

    def test_tree_row_leaf_type(self):
        rows = [line('a', 'tree_row', [(50, 100), (450, 100)], natural='tree_row', leaf_type='needleleaved'),
                line('b', 'tree_row', [(50, 300), (450, 300)], natural='tree_row', leaf_type='mixed')]
        items = self.rt.build(lines=rows)
        a = [i for i in items if i['name'].startswith('tree_row_') and abs(i['y'] - 100) < 1]
        b = [i for i in items if i['name'].startswith('tree_row_') and abs(i['y'] - 300) < 1]
        self.assertTrue(a and all('scots_pine' in i['shape'] for i in a))
        pines = sum('scots_pine' in i['shape'] for i in b)
        self.assertTrue(.25 < pines / len(b) < .75, (pines, len(b)))
        print(f'tree row items per km: {len(a) / .4:.0f}')

    def test_forest_lattice_keeps_the_clear_zone(self):
        roads = [road('m', [(0, 250), (500, 250)], 12, highway='motorway')]
        wood = [[150, 200, 0], [350, 200, 0], [350, 300, 0], [150, 300, 0], [150, 200, 0]]
        items = self.rt.build(roads=roads, areas=[{'id': 'w', 'kind': 'forest', 'tags': {'landuse': 'forest'}, 'rings': [wood]}])
        trees = [i for i in items if i['name'].startswith('tree_forest_')]
        self.assertTrue(trees)
        self.assertGreaterEqual(min(distance_to(roads[0], i['x'], i['y']) for i in trees), 6 + 6)

    def test_hedge_cap(self):
        items = self.rt.build(lines=[line(f'h{k}', 'hedge', [(10, 10 + 40 * k), (490, 10 + 40 * k)]) for k in range(12)],
                              maxHedges=300)
        self.assertEqual(len([i for i in items if i['name'].startswith('hedge_')]), 300)


if __name__ == '__main__':
    unittest.main()
