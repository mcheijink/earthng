"""Tests of the AI traffic graph on swept roads (nav_check.py). Run: python earthng/nav_check_test.py

A synthetic diamond interchange for keep-right traffic: a motorway's eastbound carriageway at y = 490 and
westbound at y = 510 (swept), an exit and an entry slip road on each side (swept), and a primary road
crossing over the motorway on a bridge at x = 500, whose two at-grade nodes the slip roads end at (a
generator road). A car must be able to leave each carriageway for the primary road and join each one from
it, with no dead end and no one-way segment against its way.
"""
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))
import nav_check as nc  # noqa: E402


def terrain_z(x, y):
    return 20 + .004 * x + .003 * y


def lupa_ok():
    try:
        nc.lua()
        return True
    except SystemExit:
        return False


def arc(p, q, bulge, n=6):
    """Points from p to q bowed by bulge m to the left of p -> q."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    return [(x0 + dx * t + nx * bulge * math.sin(math.pi * t), y0 + dy * t + ny * bulge * math.sin(math.pi * t))
            for t in [k / n for k in range(n + 1)]]


def interchange():
    roads, ids = [], [3000]

    def way(nodes, xy, tags, width, dz=None):
        ids[0] += 1
        pts = [[round(x, 2), round(y, 2), round(terrain_z(x, y) + (dz(x, y) if dz else 0), 2)] for x, y in xy]
        roads.append({'id': str(ids[0]), 'width': width, 'tags': tags, 'nodes': list(nodes), 'points': pts})
        return roads[-1]
    mw = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'maxspeed': '120'}
    link = {'highway': 'motorway_link', 'lanes': '1', 'oneway': 'yes'}
    line = lambda x0, x1, y: [(x, y) for x in (range(int(x0), int(x1) + 1, 20) if x1 > x0 else range(int(x0), int(x1) - 1, -20))]
    swept = [
        way(('E0', 'Ex'), line(0, 420, 490), dict(mw), 7.5), way(('Ex', 'En'), line(420, 580, 490), dict(mw), 7.5),
        way(('En', 'E1'), line(580, 1000, 490), dict(mw), 7.5),
        way(('W1', 'Wx'), line(1000, 580, 510), dict(mw), 7.5), way(('Wx', 'Wn'), line(580, 420, 510), dict(mw), 7.5),
        way(('Wn', 'W0'), line(420, 0, 510), dict(mw), 7.5),
        way(('Ex', 'PS'), arc((420, 490), (500, 420), 12), dict(link), 4.),
        way(('PS', 'En'), arc((500, 420), (580, 490), 12), dict(link), 4.),
        way(('Wx', 'PN'), arc((580, 510), (500, 580), 12), dict(link), 4.),
        way(('PN', 'Wn'), arc((500, 580), (420, 510), 12), dict(link), 4.),
    ]
    ptag = {'highway': 'primary', 'lanes': '2', 'maxspeed': '70'}

    def rise(x, y):
        # up to 7 m over the motorway between y = 465 and 535, eased from the slip roads' ends
        if y <= 420 or y >= 580:
            return 0.
        if 465 <= y <= 535:
            return 7.
        return 7. * ((y - 420) / 45 if y < 465 else (580 - y) / 45)
    generator = [
        way(('P0', 'PS'), [(500, y) for y in range(0, 421, 20)], dict(ptag), 8.),
        way(('PS', 'Pb0'), [(500, y) for y in range(420, 466, 15)], dict(ptag), 8., rise),
        way(('Pb0', 'Pb1'), [(500, y) for y in range(465, 536, 10)], dict(ptag, bridge='yes', layer='1'), 8., rise),
        way(('Pb1', 'PN'), [(500, y) for y in range(535, 581, 15)], dict(ptag), 8., rise),
        way(('PN', 'P1'), [(500, y) for y in range(580, 1001, 20)], dict(ptag), 8.),
    ]
    grid = 10
    n = 1000 // grid + 1
    heights = [round(terrain_z(ix * grid, iy * grid), 1) for iy in range(n) for ix in range(n)]
    cell = {'terrain': {'rect': [0., 0., 1000., 1000.], 'nx': n, 'ny': n, 'heights': heights},
            'roads': swept + generator, 'junctions': [], 'buildings': [], 'areas': [], 'lines': [], 'points': []}
    return cell, swept


@unittest.skipIf(not lupa_ok(), 'needs lupa with LuaJIT')
class Interchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = cls.L = nc.lua()
        cls.cell, cls.swept = interchange()
        cls.roads3 = nc.swept_roads3(L, cls.swept)
        cls.segments = nc.build_nav(L, cls.cell, [(0, 0), (1, 0), (0, 1), (1, 1)], cls.roads3)
        cls.graph = nc.Graph(cls.segments)

    def inside(self, x, y):
        return 40 < x < 960 and 40 < y < 960

    def nearest(self, x, y):
        return min(self.graph.pos, key=lambda n: (self.graph.pos[n][0] - x) ** 2 + (self.graph.pos[n][1] - y) ** 2)

    def test_no_dead_end(self):
        sinks, sources = self.graph.dead_ends(self.inside)
        self.assertEqual(sinks, [])
        self.assertEqual(sources, [])

    def test_one_piece(self):
        lines = [l for l in nc.report(self.graph, self.inside) if 'cut off' in l]
        self.assertEqual(lines, [])

    def test_one_way_with_the_way(self):
        # every one-way segment on a carriageway runs the way's own direction (east at y 490, west at 510)
        for s in self.graph.segments.values():
            lanes = s[3]
            pts = [s[k:k + 3] for k in range(7, len(s), 3)]
            if '-' in lanes or len(pts) < 2:
                continue
            ys = [p[1] for p in pts]
            if all(abs(y - 490) < 3 for y in ys):
                self.assertGreater(pts[-1][0], pts[0][0], s[0])
            if all(abs(y - 510) < 3 for y in ys):
                self.assertLess(pts[-1][0], pts[0][0], s[0])

    def test_lanes_follow_the_tags(self):
        for s in self.graph.segments.values():
            pts = [s[k:k + 3] for k in range(7, len(s), 3)]
            if all(abs(p[1] - 490) < 3 for p in pts) and all(abs(p[0] - 500) > 100 for p in pts):
                self.assertEqual(s[3], '++', s[0])

    def reach(self, start):
        seen, stack = {start}, [start]
        while stack:
            n = stack.pop()
            for m in self.graph.out.get(n, ()):
                if m not in seen:
                    seen.add(m); stack.append(m)
        return seen

    def test_every_movement(self):
        east_in, east_out = self.nearest(100, 490), self.nearest(900, 490)
        west_in, west_out = self.nearest(900, 510), self.nearest(100, 510)
        south, north = self.nearest(500, 100), self.nearest(500, 900)
        for start, goal, what in ((east_in, south, 'eastbound off to the primary road'),
                                  (west_in, north, 'westbound off to the primary road'),
                                  (south, east_out, 'the primary road on to eastbound'),
                                  (north, west_out, 'the primary road on to westbound'),
                                  (east_in, east_out, 'eastbound straight on'),
                                  (west_in, west_out, 'westbound straight on'),
                                  (south, north, 'the primary road over the motorway')):
            self.assertIn(goal, self.reach(start), what)


@unittest.skipIf(not lupa_ok(), 'needs lupa with LuaJIT')
class BudgetCell(unittest.TestCase):
    """The generator_budget cell: its motorway's slip road ends where a generator road begins."""

    def test_slip_road_joins_the_town(self):
        import generator_budget
        L = nc.lua()
        cell, swept = generator_budget.synthetic_cell(10, random.Random(1))
        graph = nc.Graph(nc.build_nav(L, cell, [(0, 0), (1, 0)], nc.swept_roads3(L, swept)))
        sinks, sources = graph.dead_ends(lambda x, y: 30 < x < 970 and 30 < y < 470)
        self.assertEqual(sinks, [])
        self.assertEqual(sources, [])
        pieces = graph.pieces()
        motorway = min(graph.pos, key=lambda n: (graph.pos[n][0] - 300) ** 2 + (graph.pos[n][1] - (80 + 30 * math.sin(1))) ** 2)
        town = min(graph.pos, key=lambda n: (graph.pos[n][0] - 400) ** 2 + (graph.pos[n][1] - 300) ** 2)
        self.assertTrue(any(motorway in p and town in p for p in pieces))


if __name__ == '__main__':
    unittest.main()
