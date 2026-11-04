"""Tests of fast_roads.lane_splits: a two-lane link parting into two one-lane links at a node (and the mirror merge) keeps both
lanes side by side, one straight lane line between them, until each branch's line reaches its own lane's centre.

The exporter checks run anywhere fast_roads imports. The end-to-end checks run sweep.lua under LuaJIT (pip install lupa) on the
exporter's own encoded cell and read every lane line it draws (M.debug), the lane blocks of its rows and its surface; they are
skipped without lupa. Run: python earthng/lane_split_test.py
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_roads as fr  # noqa: E402
import fast_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ID = [0]


def seg(nodes, xy, lanes, highway='motorway_link', **tags):
    """a one-way segment as fast_sources._prepare makes it, flat at z = 10"""
    _ID[0] += 1
    t = {'highway': highway, 'oneway': 'yes', 'lanes': str(lanes)}
    t.update(tags)
    xy = [list(map(float, p)) for p in xy]
    L = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    return {'id': _ID[0], 'way': 1000 + _ID[0], 'tags': t, 'xy': xy, 'nodes': list(nodes), 'width': fast_sources.road_width(t),
            'prof': [[0., L], [10., 10.]]}


def arc(R, side, L=220., n=12):
    """a branch's OSM line: leaves the origin along +x and bends to `side` at radius R"""
    return [(R * math.sin(s / R), side * R * (1 - math.cos(s / R))) for s in np.linspace(0., L, n)]


def split(RL=600., RR=600., merge=False):
    """a 300 m two-lane link and two one-lane links at node 2 (merge: the mirror, traffic towards +x)"""
    if not merge:
        return [seg((1, 2), [(-300, 0), (-150, 0), (0, 0)], 2), seg((2, 3), arc(RL, 1), 1), seg((2, 4), arc(RR, -1), 1)]
    back = lambda pts: [(-x, y) for x, y in pts][::-1]
    return [seg((2, 1), [(0, 0), (150, 0), (300, 0)], 2), seg((3, 2), back(arc(RL, 1)), 1), seg((4, 2), back(arc(RR, -1)), 1)]


def build(segs):
    g = fr.Graph(segs)
    strokes = fr.build_strokes(segs, g)
    fr.lane_events(strokes, g, segs)
    stats = fr.lane_splits(strokes, g, segs)
    return g, strokes, stats


def host_and_branch(strokes):
    H = [st for st in strokes if len(st.parts) == 2][0]
    B = [st for st in strokes if st is not H and len(st.parts) == 1 and st.lanes[0][2] == 1][0]
    return H, B


def frame(H, node=2):
    """the joint road's frame at the node: (x, y, heading). The joint road runs along +x in these tests (the stroke's own
    node tangent halves the turn to its branch's first OSM chord: 0.5 degrees off here)"""
    sN = [sv for n, sv, _ in H.nodes if n == node][0]
    i = int(np.argmin(np.abs(H.s - sN)))
    return sN, (float(H.x[i]), float(H.y[i]), 0.)


def rel(f, x, y):
    """(along, left offset) of world points in frame f (the reference line is straight in these tests: k = 0 at the node)"""
    x0, y0, h0 = f
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    return dx * math.cos(h0) + dy * math.sin(h0), -dx * math.sin(h0) + dy * math.cos(h0)


def block_edges(st, s):
    """the world offset of a stroke's lane block edges at s as sweep.lua's fitBlock places them: (centre offset, right, left)"""
    run = next(r for r in st.lanes if r[0] - 1e-6 <= s <= r[1] + 1e-6)
    eL, eR = np.interp(s, st.s, st.eL), np.interp(s, st.s, st.eR)
    left, right = fr._fit_block(run[2], run[5], eL, eR)
    return right, left


class Exporter(unittest.TestCase):
    def test_diverge_keeps_the_lanes_of_the_joint_road(self):
        segs = split()
        g, strokes, stats = build(segs)
        self.assertEqual(stats.get('diverge'), 1)
        H, B = host_and_branch(strokes)
        self.assertEqual(B.split, [{'node': '2', 'host': H.id}])
        _, nodes = fr.records(strokes, g, segs, 1000.)
        rec = [n[3] for n in nodes if n[0] == 2][0]
        self.assertEqual(rec['roles'], {B.id: ['split', H.id]})
        self.assertEqual(rec['plate'], [])
        sN, f = frame(H)
        w = 3.75
        # the approach keeps its section up to the node (build_strokes had narrowed its far side over TAPER before it)
        for s in np.arange(sN - 60., sN + 1e-6, 2.):
            self.assertAlmostEqual(np.interp(s, H.s, H.eL), 4.0, places=6)
            self.assertAlmostEqual(np.interp(s, H.s, H.eR), 4.75, places=6)
        # from the node on: B's lane right of the divider, H's left of it, each one approach lane wide, for 30 m
        for t in np.arange(SPLIT_FIRST, 30., 1.):
            for X, s, lo, hi in ((H, sN + t, 0., w), (B, t, -w, 0.)):
                _, o = rel(f, np.interp(s, X.s, X.x), np.interp(s, X.s, X.y))
                right, left = block_edges(X, s)
                self.assertAlmostEqual(o + right, lo, delta=.03, msg=f'{X is H} t={t}')
                self.assertAlmostEqual(o + left, hi, delta=.03, msg=f'{X is H} t={t}')
            # the paved edges continue the approach's: 4.0 left, 4.75 right
            _, oH = rel(f, np.interp(sN + t, H.s, H.x), np.interp(sN + t, H.s, H.y))
            _, oB = rel(f, np.interp(t, B.s, B.x), np.interp(t, B.s, B.y))
            self.assertAlmostEqual(oH + np.interp(sN + t, H.s, H.eL), 4.0, delta=.03)
            self.assertAlmostEqual(oB - np.interp(t, B.s, B.eR), -4.75, delta=.03)
        # far from the node both are their own sections again
        self.assertAlmostEqual(B.eL[-1], segs[2]['width'] - 4.0 - 1.0 + 2.0, places=6)
        self.assertAlmostEqual(H.eR[-1], 3.0, places=6)

    def test_merge_is_the_mirror(self):
        segs = split(merge=True)
        _, strokes, stats = build(segs)
        self.assertEqual(stats.get('merge'), 1)
        H, B = host_and_branch(strokes)
        sN, f = frame(H)
        for t in np.arange(SPLIT_FIRST, 30., 1.):
            for X, s, lo, hi in ((H, sN - t, 0., 3.75), (B, B.length() - t, -3.75, 0.)):
                _, o = rel(f, np.interp(s, X.s, X.x), np.interp(s, X.s, X.y))
                right, left = block_edges(X, s)
                self.assertAlmostEqual(o + right, lo, delta=.03)
                self.assertAlmostEqual(o + left, hi, delta=.03)
        # the run boundary sits at the step's upstream end (sweep.lua's row there takes the earlier run: the branch's)
        self.assertTrue(any(abs(r[1] - (sN - fr.SPLIT_STEP)) < 1e-6 and r[2] == 1 for r in H.lanes))

    def test_several_lane_runs_of_one_count(self):
        # each branch and the approach in two ways whose lane runs differ only in their turn lanes: the chains of runs walk
        # over both (the pack build crashed there: a float direction as a list index), and each run gets the lane width
        A = arc(600., 1, L=400., n=20)
        Bv = arc(600., -1, L=400., n=20)
        segs = [seg((1, 5), [(-300, 0), (-150, 0)], 2, **{'turn:lanes': 'through|through'}),
                seg((5, 2), [(-150, 0), (0, 0)], 2),
                seg((2, 6), A[:9], 1), seg((6, 3), A[8:], 1, **{'turn:lanes': 'through'}),
                seg((2, 7), Bv[:9], 1), seg((7, 4), Bv[8:], 1, **{'turn:lanes': 'through'})]
        g, strokes, stats = build(segs)
        self.assertEqual(stats.get('diverge'), 1)
        B = [st for st in strokes if st.split][0]
        H = [st for st in strokes if st.id == B.split[0]['host']][0]
        for X in (H, B):
            ones = [r for r in X.lanes if r[2] == 1]
            self.assertGreaterEqual(len(ones), 2)
            self.assertTrue(all(abs(r[5] - 3.75) < 1e-9 for r in ones))
        # and the mirror, in reverse: the chains walk backwards
        back = lambda pts: [(-x, y) for x, y in pts][::-1]
        segs = [seg((5, 1), [(150, 0), (300, 0)], 2, **{'turn:lanes': 'through|through'}),
                seg((2, 5), [(0, 0), (150, 0)], 2),
                seg((6, 2), back(A[:9]), 1), seg((3, 6), back(A[8:]), 1, **{'turn:lanes': 'through'}),
                seg((7, 2), back(Bv[:9]), 1), seg((4, 7), back(Bv[8:]), 1, **{'turn:lanes': 'through'})]
        g, strokes, stats = build(segs)
        self.assertEqual(stats.get('merge'), 1)

    def test_attached_exit_and_kept_lanes_are_not_splits(self):
        # a lane-attached exit (the host keeps its lanes and gets an added lane): unchanged
        h1 = seg((1, 2), [(0, 0), (1000, 0)], 2, 'motorway')
        h2 = seg((2, 3), [(1000, 0), (2000, 0)], 2, 'motorway')
        link = seg((2, 5), [(1000, 0), (1060, -4), (1120, -12), (1200, -30), (1300, -70)], 1)
        segs = [h1, h2, link]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        fr.lane_events(strokes, g, segs)
        before = [(st.s.copy(), st.eL.copy(), st.eR.copy(), [list(r) for r in st.lanes]) for st in strokes]
        self.assertEqual(dict(fr.lane_splits(strokes, g, segs)), {})
        for st, (s, eL, eR, lanes) in zip(strokes, before):
            self.assertTrue(np.array_equal(st.s, s) and np.array_equal(st.eL, eL) and np.array_equal(st.eR, eR))
            self.assertEqual(st.lanes, lanes)
            self.assertEqual(st.split, [])
        # a two-lane link that keeps both lanes at a one-lane branch: no split either
        segs = [seg((1, 2), [(-300, 0), (0, 0)], 2), seg((2, 3), arc(3000, 1), 2), seg((2, 4), arc(500, -1), 1)]
        _, strokes, stats = build(segs)
        self.assertEqual(dict(stats), {})


def network(rng, count=12):
    """a synthetic network of `count` lane splits and merges 1.2 km apart: 2-3 approach lanes parting into branches whose
    lanes add up (or, now and then, not), branches of R 150-3000 m to either side and 30-400 m long, ways split by their
    turn lanes, width or speed, a side road joining some branches, curved approaches, node order against the travel"""
    segs, nid = [], [100]

    def node():
        nid[0] += 1
        return nid[0]

    def bend(x0, y0, h, R, side, L, step):
        pts = []
        for s in np.arange(0., L + 1e-6, step):
            a = s / R
            dx, dy = R * math.sin(a), side * R * (1 - math.cos(a))
            pts.append((x0 + dx * math.cos(h) - dy * math.sin(h), y0 + dx * math.sin(h) + dy * math.cos(h)))
        return pts

    def ways(pts, n0, n1, lanes, hw, merge):
        """the points as 1-3 ways (their own tags), in travel order; merge: traffic runs towards n1 (reversed below)"""
        cuts = sorted(rng.sample(range(1, len(pts) - 1), min(len(pts) - 2, rng.randint(0, 2)))) if len(pts) > 3 else []
        bounds = [0] + cuts + [len(pts) - 1]
        ids = [n0] + [node() for _ in cuts] + [n1]
        for k in range(len(bounds) - 1):
            extra = rng.choice([{}, {'turn:lanes': '|'.join(['through'] * lanes)}, {'maxspeed': '80'}, {'width': str(3.5 * lanes + 1.5)}])
            xy = pts[bounds[k]:bounds[k + 1] + 1]
            a, b = ids[k], ids[k + 1]
            if rng.random() < .3:                          # node order against the travel
                segs.append(seg((b, a), xy[::-1], lanes, hw, oneway='-1', **extra))
            else:
                segs.append(seg((a, b), xy, lanes, hw, **extra))

    for k in range(count):
        x0 = 1200. * k
        hw = rng.choice(['motorway_link', 'trunk_link', 'primary'])
        nA = rng.choice([2, 2, 3])
        nH = rng.randint(1, nA - 1)
        nB = nA - nH if rng.random() < .85 else nA           # sometimes no split (lanes do not add up)
        merge = rng.random() < .4
        kA = rng.choice([0., 0., 1 / 400., -1 / 300.])
        approach = [(x0 - 300 + s, 0.) for s in np.arange(0., 300.1, rng.choice([20., 50., 150.]))] if kA == 0 else \
            [(x0 + math.sin(kA * (s - 300)) / kA, (1 - math.cos(kA * (s - 300))) / kA) for s in np.arange(0., 300.1, 25.)]
        h = math.atan2(approach[-1][1] - approach[-2][1], approach[-1][0] - approach[-2][0])
        N = node()
        sideH = rng.choice([1, -1])
        H_pts = bend(x0, 0., h, rng.choice([150., 600., 3000., 1e6]), sideH, rng.choice([30., 90., 250., 400.]), rng.choice([10., 20., 30.]))
        B_pts = bend(x0, 0., h, rng.choice([150., 300., 600., 1500.]), -sideH, rng.choice([30., 90., 250., 400.]), rng.choice([10., 20., 30.]))
        if not merge:
            ways(approach, node(), N, nA, hw, False)
            ways(H_pts, N, node(), nH, hw, False)
            endB = node()
            ways(B_pts, N, endB, nB, hw, False)
        else:
            flip = lambda pts: [(2 * x0 - x, y) for x, y in pts][::-1]      # mirrored about x0, in travel order
            ways(flip(approach), N, node(), nA, hw, True)
            ways(flip(H_pts), node(), N, nH, hw, True)
            endB = node()
            ways(flip(B_pts), endB, N, nB, hw, True)
        if rng.random() < .3:                                # a side road joining the branch at its far end
            far = B_pts[-1] if not merge else (2 * x0 - B_pts[-1][0], B_pts[-1][1])
            segs.append(seg((endB, node()), [far, (far[0], far[1] + 120.)], 2, 'secondary', oneway='no'))
    return segs


SPLIT_FIRST = 2 * fr.SPLIT_STEP


SPLIT_FIRST = 2 * fr.SPLIT_STEP


# ---------------------------------------------------------------------------------------------------------------------------
# End to end: sweep.lua on the exporter's cell
# ---------------------------------------------------------------------------------------------------------------------------
def lua_runtime():
    for name in ('luajit21', 'luajit20'):
        try:
            return __import__('lupa.' + name, fromlist=[name]).LuaRuntime(unpack_returned_tuples=True)
        except ImportError:
            pass
    return None


LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local SW = require('ge/extensions/e2r/sweep')
local R3 = require('ge/extensions/e2r/roads3')
local function mesh() return {verts = {}, normals = {}, uvs = {}, faces = {}, material = 'm', vmap = {}, vkz = {}, nmap = {},
  umap = {}, tris = 0} end
return function(bytes, version)
  local cell = R3.cell(bytes, 1, #bytes, 0, 0, version, function() end)
  local plan = SW.plan(R3.merge({cell}, nil), {rect = {-1000, -1000, 1000, 1000}, classes = SW.CLASSES.fitted})
  local lines, blocks, info = {}, {}, {}
  SW.debug = function(S, a, b, u0, u1, t0, t1, id)
    local ax, ay, bx, by = a.x + a.nx * u0, a.y + a.ny * u0, b.x + b.nx * u1, b.y + b.ny * u1
    lines[#lines + 1] = {S.id, id, ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1}
  end
  for sid, S in pairs(plan.strokes) do
    local gore = mesh()
    for _, c in ipairs(S.list) do
      if c.rows and #c.rows > 0 then
        SW.markMesh(plan, S, c, mesh(), {0, 0, 0})
        if S.cutA then SW.goreMesh(plan, S, c, gore, {0, 0, 0}) end
        for _, list in ipairs(c.rows) do
          for _, r in ipairs(list) do
            local k = SW.laneBlock(S, r.s)
            blocks[#blocks + 1] = {sid, r.s, r.x + r.nx * k.right, r.y + r.ny * k.right, r.x + r.nx * k.left, r.y + r.ny * k.left}
          end
        end
      end
    end
    info[#info + 1] = {sid, S.split and 1 or 0, S.blend0 or -1, gore.tris}
  end
  SW.debug = nil
  return lines, blocks, info, plan, SW
end
'''


@unittest.skipIf(lua_runtime() is None, 'needs LuaJIT: pip install lupa')
class EndToEnd(unittest.TestCase):
    def run_lua(self, segs):
        g, strokes, _ = build(segs)
        rows, nodes = fr.records(strokes, g, segs, 1000.)
        data = fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.))
        L = lua_runtime()
        L.globals().REPO_ROOT = ROOT
        lines, blocks, info, plan, SW = L.execute(LUA)(data, fr.VERSION)
        H, B = host_and_branch(strokes)
        _, f = frame(H)
        out = {'H': H, 'B': B, 'frame': f, 'plan': plan, 'SW': SW, 'lines': [], 'blocks': {H.id: [], B.id: []}, 'info': {}}
        for p in lines.values():
            sid, lid, xa, ya, xb, yb = list(p.values())
            ta, oa = rel(f, xa, ya)
            tb, ob = rel(f, xb, yb)
            if ta > tb:
                ta, oa, tb, ob = tb, ob, ta, oa
            out['lines'].append(('H' if sid == H.id else 'B', lid, float(ta), float(oa), float(tb), float(ob)))
        for p in blocks.values():
            sid, s, xr, yr, xl, yl = list(p.values())
            t, oR = rel(f, xr, yr)
            _, oL = rel(f, xl, yl)
            out['blocks'][sid].append((float(t), float(oR), float(oL)))
        for p in info.values():
            sid, sp, blend0, gore = list(p.values())
            out['info'][sid] = (sp, blend0, gore)
        return out

    def check_together(self, run, d, hold=30.):
        """from 40 m before the node to the lanes' parting (d: +1 diverge, -1 merge, along the zone): the right edge line at
        -3.75, the left at +3.75, one lane line at 0, and no line anywhere else. hold: where the bend may begin (the line 1 m
        inside its lane's centre: 32 m at R 600, 26 m at R 400)"""
        w, near = 3.75, .06
        allowed = (-w, 0., w)
        divider = []
        for who, lid, ta, oa, tb, ob in run['lines']:
            for t, o in ((ta, oa), (tb, ob), ((ta + tb) / 2, (oa + ob) / 2)):
                if -40. <= t * d <= hold:
                    self.assertTrue(min(abs(o - v) for v in allowed) < near, f'{who}:{lid} at {t:.2f} m lies at {o:.2f}')
            if abs(oa) < near and abs(ob) < near and -40. <= ta * d and tb * d <= hold + 10:
                divider.append((min(ta * d, tb * d), max(ta * d, tb * d)))
        # the lane line is dashed (Leitlinie 6/12): no gap in it longer than a dash's gap, from the approach to the parting
        divider.sort()
        reach = -40.
        for a, b in divider:
            self.assertLess(a - reach, 12. + .5, f'lane line missing from {reach:.1f} to {a:.1f} m')
            reach = max(reach, b)
        self.assertGreater(reach, hold - 12.)
        # both edge lines are there all the way
        for target in (-w, w):
            spans = sorted((min(ta * d, tb * d), max(ta * d, tb * d)) for _, _, ta, oa, tb, ob in run['lines']
                           if abs(oa - target) < near and abs(ob - target) < near)
            reach = -40.
            for a, b in spans:
                if b < -40.:
                    continue
                if a > hold:
                    break
                self.assertLess(a - reach, .2, f'edge line at {target} missing from {reach:.1f} to {a:.1f} m')
                reach = max(reach, b)
            self.assertGreater(reach, hold)

    def check_lanes(self, run, d, hold=30., until=100.):
        """each branch's lane: one approach lane wide in its joint place up to the bend, then out without coming back (until:
        where a branch that never left comes back to its line, fast_roads.lane_splits)"""
        H, B = run['H'], run['B']
        for X, (lo, hi), out in ((H, (0., 3.75), 1.), (B, (-3.75, 0.), -1.)):
            rows = sorted((t * d, oR, oL) for t, oR, oL in run['blocks'][X.id] if 0. < t * d <= until)
            kept = []
            for t, oR, oL in rows:
                if t <= hold:
                    self.assertAlmostEqual(oR, lo, delta=.06, msg=f't={t}')
                    self.assertAlmostEqual(oL, hi, delta=.06, msg=f't={t}')
                if not kept or t - kept[-1][0] >= fr.SPLIT_SPACING - .01:      # the stations' chords: a corner at each
                    kept.append((t, (oR + oL) / 2))
            for (t0, c0), (t1, c1), (t2, c2) in zip(kept, kept[1:], kept[2:]):
                # outward only, and no kink: the lane's centre bends at no less than R 80 m (the arcs are R 400-600; the bend
                # from the nose onto them, fast_roads.SPLIT_BEND, R 90-150; the old 40 m window moved it 1.9 m in 10 m)
                self.assertGreaterEqual((c1 - c0) * out, -.02, f't={t1}')
                bend = 2 * ((c2 - c1) / (t2 - t1) - (c1 - c0) / (t1 - t0)) / (t2 - t0)
                self.assertLess(abs(bend), 1 / 80., f't={t1}')

    def check_surface(self, run, d):
        """the driven surface covers both lanes and the approach's shoulders without a hole from the node on"""
        plan, SW, f = run['plan'], run['SW'], run['frame']
        x0, y0, h0 = f
        for t in np.arange(.02, 30., .5):
            for o in np.arange(-4.6, 3.9, .25):
                ta = t * d
                x = x0 + ta * math.cos(h0) - o * math.sin(h0)
                y = y0 + ta * math.sin(h0) + o * math.cos(h0)
                hits = SW.surfacesAt(plan, x, y, .01)
                zs = [h.z for h in hits.values()]
                self.assertTrue(zs, f'no surface at {ta:.2f} m, {o:.2f} m')
                self.assertLess(max(zs) - min(zs), .02, f'surfaces {min(zs):.3f}..{max(zs):.3f} at {ta:.2f}, {o:.2f}')

    def check_marks(self, run, d, nose):
        """the marks as drawn (game test of a real split: the divider missing beside the car, a loose line piece before the
        gore, the gore's outline in fragments and the branch's lines starting before the nose). nose: m from the node where
        the leaving branch's OSM line reaches its lane's centre (where the lanes part)"""
        B = run['B']
        blend0 = run['info'][B.id][1]
        gore_end = blend0 if d > 0 else B.length() - blend0          # sweep.lua's nose: the gore's end, m from the node
        self.assertGreater(gore_end, nose + 5.)

        def cover(spans, a, b, what):
            """spans [(t0, t1)] cover a..b with no gap over 0.3 m"""
            reach = a
            for s0, s1 in sorted(spans):
                if s1 < reach:
                    continue
                self.assertLess(s0 - reach, .3, f'{what}: missing from {reach:.1f} to {s0:.1f} m')
                reach = max(reach, s1)
            self.assertGreaterEqual(reach, b - .3, f'{what}: ends at {reach:.1f} m, not {b:.1f}')

        def along(p):
            return min(p[2] * d, p[4] * d), max(p[2] * d, p[4] * d)
        # 1. one unbroken divider from the node to the nose, on the joint road's lane line
        div = [along(p) for p in run['lines'] if p[0] == 'H' and abs(p[3]) < .1 and abs(p[5]) < .1 and along(p)[1] > 0.]
        cover(div, .3, nose - .5, 'divider')
        # 2. the host's line along the branch runs on unbroken to the gore's end: the gore's outline, nothing loose
        right = np.mean([p[3] for p in run['lines'] if p[0] == 'B' and abs(p[2]) < 100.]) < 0
        side = 'R' if right else 'L'
        host = [along(p) for p in run['lines'] if p[0] == 'H' and p[1] in ('fork' + side, 'edge' + side) and along(p)[1] > 0.]
        cover(host, .3, gore_end - .5, "host's line along the gore")
        # 3. the branch's inner line (the gore's other side) starts at the nose, not before, and runs on to the gore's end
        inner = 'edgeL' if right else 'edgeR'
        own = [along(p) for p in run['lines'] if p[0] == 'B' and p[1] == inner and along(p)[0] < gore_end + 5.]
        self.assertTrue(own, 'no inner line on the branch')
        self.assertGreaterEqual(min(a for a, _ in own), nose - 1., "the branch's inner line starts before the nose")
        cover(own, min(a for a, _ in own), gore_end - .5, "branch's inner line")

    def test_diverge(self):
        run = self.run_lua(split())
        self.assertEqual(run['info'][run['B'].id][0], 1)
        self.check_together(run, 1.)
        self.check_lanes(run, 1.)
        self.check_surface(run, 1.)
        self.check_marks(run, 1., math.sqrt(2 * 600. * 1.875))
        # the gore keeps its chevrons at the nose
        self.assertGreater(run['info'][run['B'].id][2], 0)

    def test_merge(self):
        run = self.run_lua(split(merge=True))
        self.assertEqual(run['info'][run['B'].id][0], 1)
        self.check_together(run, -1.)
        self.check_lanes(run, -1.)
        self.check_surface(run, -1.)
        self.check_marks(run, -1., math.sqrt(2 * 600. * 1.875))
        self.assertGreater(run['info'][run['B'].id][2], 0)

    def test_straight_through_branch(self):
        # the through branch mapped straight on: its lane stays in place and the other leaves
        run = self.run_lua(split(RL=1e7, RR=400.))
        self.check_together(run, 1., hold=25.)
        self.check_lanes(run, 1., hold=25., until=fr.SPLIT_REACH - fr.SPLIT_EASE - fr.TAPER['link'])
        self.check_marks(run, 1., math.sqrt(2 * 400. * 1.875))


class Network(unittest.TestCase):
    def test_random_networks(self):
        """the whole road stage (as fast_roads.build, less the junction surfaces) over random networks, then sweep.lua's plan
        and items on each cell when LuaJIT is there: nothing raises, every split keeps finite, positive extents"""
        import random
        L = lua_runtime()
        splits = 0
        for seed in range(40):
            rng = random.Random(seed)
            segs = network(rng)
            fr.lane_pinch(segs)
            g = fr.Graph(segs)
            strokes = fr.build_strokes(segs, g)
            fr.lane_events(strokes, g, segs)
            stats = fr.lane_splits(strokes, g, segs)
            splits += stats.get('diverge', 0) + stats.get('merge', 0)
            fr.clamp_pairs(strokes)
            fr.structures(strokes, segs, [])
            fr.furniture(strokes, segs, [])
            fr.median_rails(strokes)
            for st in strokes:
                for name in ('x', 'y', 'h', 'eL', 'eR'):
                    self.assertTrue(np.all(np.isfinite(getattr(st, name))), f'seed {seed} {st.id} {name}')
                self.assertTrue(np.all(np.diff(st.s) > 0), f'seed {seed} {st.id}: stations out of order')
                self.assertGreaterEqual(min(st.eL.min(), st.eR.min()), -1e-9, f'seed {seed} {st.id}')
            rows, nodes = fr.records(strokes, g, segs, 1000.)
            if L is None:
                continue
            by_cell = {}
            for r in rows:
                by_cell.setdefault((r[2], r[3]), []).append(r[4])
            L.globals().REPO_ROOT = ROOT
            run = L.execute(NETWORK_LUA)
            for (ci, cj), chunks in by_cell.items():
                origin = (ci * 1000., cj * 1000.)
                ns = [n[3] for n in nodes if (n[1], n[2]) == (ci, cj)]
                data = fr.encode_cell(chunks, ns, origin)
                res = run(data, fr.VERSION, origin[0], origin[1])
                ok, err = (res, None) if isinstance(res, bool) else (res[0], res[1])
                self.assertTrue(ok, f'seed {seed} cell {ci},{cj}: {err}')
        self.assertGreater(splits, 120)          # 148 of 480 nodes: the rest too short, lanes not adding up, sides unclear, the wrong way


NETWORK_LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local SW = require('ge/extensions/e2r/sweep')
local R3 = require('ge/extensions/e2r/roads3')
return function(bytes, version, ox, oy)
  return pcall(function()
    local cell = R3.cell(bytes, 1, #bytes, ox, oy, version, function() end)
    local plan = SW.plan(R3.merge({cell}, nil), {rect = {ox, oy, ox + 1000, oy + 1000}, classes = SW.CLASSES.fitted})
    SW.items(plan, {tileKey = 'n', markKind = 'mark'})
    SW.nav(plan, plan.rect, 50)
  end)
end
'''


if __name__ == '__main__':
    unittest.main()
