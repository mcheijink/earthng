"""Tests of fast_curves (plan-view curves). Run: python earthng/fast_curves_test.py"""
import math
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_curves as fc  # noqa: E402

LINK = fc.CLASSES['motorway_link']
MOTORWAY = fc.CLASSES['motorway']


def radii(pts):
    """circumradius of every consecutive point triple"""
    out = []
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        out.append(ab * bc * ca / (2 * cross) if cross > 1e-9 else math.inf)
    return out


def headings(pts):
    return [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts, pts[1:])]


def turn(a, b):
    return abs((b - a + math.pi) % (2 * math.pi) - math.pi)


def distance_to_polyline(p, line):
    best = math.inf
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dy * dy
        t = 0 if l2 == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
        best = min(best, math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]))
    return best


def arc_nodes(radius, spacing, angle, start=0):
    n = int(angle * radius / spacing)
    return [(radius * math.sin(k * spacing / radius), radius * (1 - math.cos(k * spacing / radius))) for k in range(n + 1)]


def way(points, first=1, tags=None):
    return (tags or {'highway': 'motorway_link'}, [(first + k, x, y) for k, (x, y) in enumerate(points)])


class Curves(unittest.TestCase):
    def test_arc_keeps_its_radius(self):
        # a 90 m loop ramp drawn with a node every 15 m, straight lead-in and lead-out
        bend = arc_nodes(90., 15., math.pi / 2)
        lead = [(-60., 0.), (-30., 0.)]
        out = [(bend[-1][0] + 90 * 0, bend[-1][1] + d) for d in (30., 60.)]
        pts = lead + bend + out
        ways = {1: way(pts)}
        refs = Counter(n for _, nodes in ways.values() for n, _, _ in nodes)
        line, index = fc.fit(ways, refs)[1]
        self.assertEqual(line[0], pts[0])
        self.assertEqual(line[-1], pts[-1])
        self.assertEqual(sorted(index), [0, len(pts) - 1])
        r = sorted(radii(line))
        # the middle of the bend at the bend's radius (within the chords' sagitta), no tighter anywhere
        self.assertGreater(r[0], 80.)
        mid = [q for q in radii(line) if q < 200]
        self.assertTrue(85. < sorted(mid)[len(mid) // 2] < 95.)
        # no kink: consecutive chords turn by at most MAX_TURN (plus rounding)
        h = headings(line)
        self.assertLess(max(turn(a, b) for a, b in zip(h, h[1:])), fc.MAX_TURN + 1e-3)
        # never further from the OSM line than the soft bound plus the transition shift
        self.assertLess(max(distance_to_polyline(p, pts) for p in line), LINK[0] + 0.5)

    def test_shared_node_kept_with_one_tangent(self):
        # a bend split into two ways at a node: both pass the node in the same direction
        bend = arc_nodes(300., 40., 0.8)
        a, b = bend[:6], bend[5:]
        ways = {1: way(a, 1), 2: way(b, 6)}
        ways[2] = (ways[2][0], [(6 + k, x, y) for k, (x, y) in enumerate(b)])
        refs = Counter(n for _, nodes in ways.values() for n, _, _ in nodes)
        res = fc.fit(ways, refs)
        la, ia = res[1]
        lb, ib = res[2]
        self.assertEqual(la[ia[5]], a[5])
        self.assertEqual(lb[ib[0]], b[0])
        ha = headings(la)[-1]
        hb = headings(lb)[0]
        self.assertLess(turn(ha, hb), fc.MAX_TURN + 1e-3)      # chords: no more than along the curve
        # the bend's radius on both sides of the joint
        for line in (la, lb):
            r = [q for q in radii(line) if q < 2000]
            self.assertTrue(r and min(r) > 200.)

    def test_junction_node_inside_a_way(self):
        # a node shared with a side road inside the way stays exactly where it is
        bend = arc_nodes(200., 30., 1.0)
        ways = {1: way(bend, 1), 2: ({'highway': 'residential'}, [(4, *bend[3]), (99, bend[3][0] + 5, bend[3][1] - 40)])}
        refs = Counter(n for _, nodes in ways.values() for n, _, _ in nodes)
        line, index = fc.fit(ways, refs)[1]
        self.assertIn(3, index)
        self.assertEqual(line[index[3]], bend[3])
        self.assertNotIn(2, fc.fit(ways, refs))     # residential: not curved

    def test_isolated_corner_is_rounded_inside(self):
        # one node turning 60 degrees between 200 m straights: an arc that cuts the corner by at most the
        # hard bound, no S-bend
        pts = [(0., 0.), (200., 0.), (200. + 200 * math.cos(math.pi / 3), 200 * math.sin(math.pi / 3))]
        ways = {1: way(pts, tags={'highway': 'secondary'})}
        refs = Counter(n for _, nodes in ways.values() for n, _, _ in nodes)
        line, _ = fc.fit(ways, refs)[1]
        soft, hard, window, rmin = fc.CLASSES['secondary']
        self.assertLess(distance_to_polyline((200., 0.), line), hard + fc.SHIFT * soft + 0.01)
        h = headings(line)
        steps = [((b - a + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(h, h[1:])]
        self.assertTrue(all(s >= -2e-4 for s in steps))      # turns one way only
        self.assertTrue(all(abs(p[1]) < 1e-6 or p[0] > 150 for p in line))

    def test_straight_way_unchanged(self):
        pts = [(0., 0.), (100., 0.), (250., 0.)]
        ways = {1: way(pts, tags={'highway': 'motorway'})}
        refs = Counter(n for _, nodes in ways.values() for n, _, _ in nodes)
        line, index = fc.fit(ways, refs)[1]
        self.assertEqual(line, [(0., 0.), (250., 0.)])

    def test_switch_and_signature(self):
        self.assertIn('curves-', fc.SIGNATURE)


if __name__ == '__main__':
    unittest.main()
