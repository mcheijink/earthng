"""Tests of sweep.lua's fork markings on synthetic geometry (pack v3 cells from fast_roads, sweep.lua under LuaJIT): the host's
edge line along a fork branch is solid, at the edge line's width where the host keeps its lanes and a lane line's where a
lane leaves; two links that end (or start) side by side on a third road get a paved, hatched gore. Needs lupa (pip install lupa); skipped without it.
Run: python earthng/marking_test.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_roads as fr  # noqa: E402
import lane_split_test as T  # noqa: E402

# lane_split_test's harness, with each line piece's width
LUA = T.LUA.replace('SW.debug = function(S, a, b, u0, u1, t0, t1, id)', 'SW.debug = function(S, a, b, u0, u1, t0, t1, id, width)') \
    .replace('ax + (bx - ax) * t1, ay + (by - ay) * t1}', 'ax + (bx - ax) * t1, ay + (by - ay) * t1, width}')


def run(segs):
    g, strokes, _ = T.build(segs)
    rows, nodes = fr.records(strokes, g, segs, 1000.)
    data = fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.))
    L = T.lua_runtime()
    L.globals().REPO_ROOT = T.ROOT
    lines, _, info, plan, SW = L.execute(LUA)(data, fr.VERSION)
    lines = [list(p.values()) for p in lines.values()]
    info = {p[1]: list(p.values()) for p in info.values()}
    return strokes, lines, info, plan, SW


def pieces(lines, sid, lid):
    """(x0, x1, width) of a stroke's line pieces, sorted (the geometry runs along +x)"""
    return sorted((min(p[2], p[4]), max(p[2], p[4]), p[6]) for p in lines if p[0] == sid and p[1] == lid)


@unittest.skipIf(T.lua_runtime() is None, 'needs LuaJIT: pip install lupa')
class ForkLine(unittest.TestCase):
    def test_solid_where_the_host_keeps_its_lanes(self):
        # a two-lane link that stays two lanes; a one-lane link leaves it without an added lane
        segs = [T.seg((1, 2), [(-300, 0), (0, 0)], 2), T.seg((2, 3), T.arc(3000, 1), 2), T.seg((2, 4), T.arc(400, -1), 1)]
        strokes, lines, info, _, _ = run(segs)
        H = [st for st in strokes if len(st.parts) == 2][0]
        fork = pieces(lines, H.id, 'forkR')
        self.assertTrue(fork)
        # one unbroken line at the right edge line's width (Autobahn class: broad), from the edge line it continues
        edge = [p for p in pieces(lines, H.id, 'edgeR') if p[1] <= fork[0][0] + 1e-3]
        self.assertAlmostEqual(edge[-1][1], fork[0][0], delta=.01)
        for (a0, a1, w), (b0, b1, _) in zip(fork, fork[1:]):
            self.assertLess(b0 - a1, .01)
        self.assertTrue(all(abs(w - .30) < 1e-6 for _, _, w in fork))
        self.assertGreater(fork[-1][1] - fork[0][0], 50.)
        # no line of the branch on the host's lanes (its far edge line was kept to its own edge where the cut left it no
        # surface: 0.5 m inside the host's right lane for 15 m from the node)
        B = [st for st in strokes if st is not H and st.lanes[0][2] == 1][0]
        on = [p for p in lines if p[0] == B.id and max(p[3], p[5]) > -3.5 + .2]
        self.assertEqual(on, [])

    def test_narrow_and_unbroken_where_a_lane_leaves(self):
        # a lane split (fast_roads.lane_splits): the line between the two lanes is a lane line's width, and unbroken to the
        # nose and on as the gore's outline (game test: the dashed one was missing beside the car and left a loose dash)
        strokes, lines, _, _, _ = run(T.split())
        H = [st for st in strokes if len(st.parts) == 2][0]
        fork = pieces(lines, H.id, 'forkR')
        self.assertTrue(fork)
        # (the last piece joins the edge line it becomes at the gore's end, at that line's width)
        self.assertTrue(all(abs(w - .15) < 1e-6 for _, _, w in fork[:-1]))
        for (a0, a1, _), (b0, b1, _) in zip(fork, fork[1:]):
            self.assertLess(b0 - a1, .01)


@unittest.skipIf(T.lua_runtime() is None, 'needs LuaJIT: pip install lupa')
class LinksOnAThirdRoad(unittest.TestCase):
    def check(self, segs, d):
        """d: +1 the links start at the node (along +x), -1 they end there"""
        strokes, lines, info, plan, SW = run(segs)
        forked = [v for v in info.values() if v[2] > 0]
        self.assertEqual(len(forked), 1, info)
        sid, _, blend0, gore = forked[0]
        B = [st for st in strokes if st.id == sid][0]
        nose = blend0 if d > 0 else B.length() - blend0         # m from the node
        self.assertGreater(nose, 20.)
        self.assertGreater(gore, 0)                              # chevrons
        # the wedge between the two links is paved from the node to the nose, at one height
        for t in range(2, int(nose) - 4):
            hits = [h.z for h in SW.surfacesAt(plan, d * t, 0., .01).values()]
            self.assertTrue(hits, f'no surface between the links {t} m from the node')
            self.assertLess(max(hits) - min(hits), .02, f'{t} m: surfaces {min(hits):.3f}..{max(hits):.3f}')
        # the host's line along the branch: a lane line (both links' lanes go on as the third road's two lanes)
        host = [st for st in strokes if st.id != sid and st.highway == 'motorway_link'][0]
        fork = sorted(pieces(lines, host.id, 'forkR') + pieces(lines, host.id, 'forkL'), key=lambda p: d * p[0])
        self.assertTrue(fork and all(abs(w - .15) < 1e-6 for _, _, w in fork[:-1]))     # the last one joins the edge line

    def test_merge(self):
        back = lambda pts: [(-x, y) for x, y in pts][::-1]
        self.check([T.seg((3, 2), back(T.arc(600, 1)), 1), T.seg((4, 2), back(T.arc(600, -1)), 1),
                    T.seg((2, 1), [(0, 0), (150, 0), (300, 0)], 2, 'trunk')], -1.)

    def test_diverge(self):
        self.check([T.seg((1, 2), [(-300, 0), (-150, 0), (0, 0)], 2, 'trunk'), T.seg((2, 3), T.arc(600, 1), 1),
                    T.seg((2, 4), T.arc(600, -1), 1)], 1.)


if __name__ == '__main__':
    unittest.main()
