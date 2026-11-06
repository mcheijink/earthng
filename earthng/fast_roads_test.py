"""Tests of fast_roads (pack v3 road geometry) and the floors of fast_profile. Run: python earthng/fast_roads_test.py"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_profile  # noqa: E402
import fast_roads as fr  # noqa: E402
import fast_sources  # noqa: E402

_ID = [0]


def seg(nodes, xy, highway='motorway', **tags):
    """a segment as fast_sources._prepare makes it, with a flat profile at z"""
    _ID[0] += 1
    t = dict(highway=highway, **tags)
    xy = [list(map(float, p)) for p in xy]
    L = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    s = {'id': _ID[0], 'way': 1000 + _ID[0], 'tags': t, 'xy': xy, 'nodes': list(nodes), 'width': fast_sources.road_width(t)}
    s['prof'] = [[0., L], [10., 10.]]
    return s


def arc(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Strokes(unittest.TestCase):
    def test_joints_make_one_stroke_in_travel_direction(self):
        # a motorway drawn against its direction in three ways (oneway=-1 on the middle one, reversed)
        a = seg((1, 2), [(0, 0), (200, 0)])
        b = seg((3, 2), [(400, 0), (300, 1), (200, 0)], oneway='-1')
        c = seg((3, 4), [(400, 0), (700, 0)])
        g = fr.Graph([a, b, c])
        strokes = fr.build_strokes([a, b, c], g)
        self.assertEqual(len(strokes), 1)
        st = strokes[0]
        self.assertAlmostEqual(st.x[0], 0.)
        self.assertAlmostEqual(st.x[-1], 700.)
        self.assertEqual(g.kind[2], 'joint')
        # the way joints are free points of the fit: no node is fixed inside, and the line is smooth there
        d = np.diff(np.column_stack([st.x, st.y]), axis=0)
        turn = np.abs(wrap(np.diff(np.arctan2(d[:, 1], d[:, 0]))))
        self.assertLess(np.degrees(turn).max(), 2.6)

    def test_through_road_passes_a_t_junction_and_keeps_its_node(self):
        main1 = seg((1, 2), [(0, 0), (100, 5), (200, 0)], 'tertiary')
        main2 = seg((2, 3), [(200, 0), (300, -5), (400, 0)], 'tertiary')
        side = seg((2, 4), [(200, 0), (200, 150)], 'residential')
        g = fr.Graph([main1, main2, side])
        strokes = {st.highway: st for st in fr.build_strokes([main1, main2, side], g)}
        st = strokes['tertiary']
        self.assertEqual(len(st.parts), 2)
        n = next(sv for nd, sv, _ in st.nodes if nd == 2)
        i = int(np.argmin(np.abs(st.s - n)))
        self.assertAlmostEqual(st.x[i], 200., places=6)
        self.assertAlmostEqual(st.y[i], 0., places=6)
        self.assertEqual(g.kind[2], 'junction')

    def test_slip_road_leaves_along_the_host_tangent(self):
        # a motorway bend with a link leaving at a node inside it
        host1 = seg((1, 2), arc(0, 1000, 1000, -math.pi / 2, -math.pi / 2 + .3, 8))
        p = host1['xy'][-1]
        host2 = seg((2, 3), arc(0, 1000, 1000, -math.pi / 2 + .3, -math.pi / 2 + .6, 8))
        link = seg((2, 5), [p, (p[0] + 60, p[1] + 2), (p[0] + 120, p[1] - 6), (p[0] + 180, p[1] - 30)], 'motorway_link', oneway='yes')
        g = fr.Graph([host1, host2, link])
        self.assertEqual(g.kind[2], 'diverge')
        strokes = {st.highway: st for st in fr.build_strokes([host1, host2, link], g)}
        H, B = strokes['motorway'], strokes['motorway_link']
        sN = next(sv for nd, sv, _ in H.nodes if nd == 2)
        self.assertLess(abs(wrap(np.interp(sN, H.s, H.h) - B.h[0])), 1e-9)
        self.assertAlmostEqual(B.x[0], p[0], places=6)

    def test_heights_follow_the_profile_through_the_fit(self):
        a = seg((1, 2), [(0, 0), (100, 10), (200, 0)], 'secondary')
        a['prof'] = [[0., 100., a['prof'][0][1]], [10., 12., 14.]]
        st = fr.build_strokes([a], fr.Graph([a]))[0]
        self.assertAlmostEqual(st.z[0], 10.)
        self.assertAlmostEqual(st.z[-1], 14.)
        self.assertTrue(np.all(np.diff(st.z) >= -1e-9))


class Sections(unittest.TestCase):
    def test_tight_one_lane_link_is_wide_and_flat(self):
        s = np.arange(0., 400., 5.)
        n = len(s)
        # a right-hand loop (kappa < 0), R = 60 m
        k = np.full(n, -1 / 60.)
        eL, eR = np.full(n, 3.0), np.full(n, 3.0)
        W = fr.loop_widen(s, k, eL, eR)
        self.assertAlmostEqual(float((W[0] + W[1])[n // 2]), fr.LOOP_WIDTH, places=3)
        self.assertAlmostEqual(float((W[0] + W[1])[0]), 6.0, places=6)       # the ends keep the width: the host's lanes lie there
        self.assertGreater(float(W[1][n // 2]), 3.0)       # the inner side of a right turn is the right
        self.assertEqual(float(W[0][n // 2]), 3.0)
        Wl = fr.loop_widen(s, -k, eL, eR)                  # a left-hand loop: the extra goes left
        self.assertGreater(float(Wl[0][n // 2]), 3.0)
        self.assertEqual(float(Wl[1][n // 2]), 3.0)
        # R = 200 m: untouched
        W2 = fr.loop_widen(s, np.full(n, -1 / 200.), eL, eR)
        self.assertEqual(float((W2[0] + W2[1])[n // 2]), 6.0)
        fl, fr_ = fr.crossfall(s, k, 'motorway_link', True, W[0], W[1])
        self.assertLessEqual(float(np.abs(fl).max()), fr.LOOP_Q_MAX + 1e-9)
        fl, fr_ = fr.crossfall(s, np.full(n, -1 / 150.), 'motorway_link', True, eL, eR)
        self.assertGreater(float(np.abs(fl).max()), fr.LOOP_Q_MAX)       # a wider bend keeps the ramp's own bank

    def test_crossfall_straight_and_curve(self):
        s = np.arange(0, 2001, 5.)
        k = np.zeros(len(s))
        fl, fr_ = fr.crossfall(s, k, 'motorway', True, np.full(len(s), 6.), np.full(len(s), 7.))
        self.assertTrue(np.allclose(fl, .025) and np.allclose(fr_, -.025))      # one-way: falls to the right
        fl, fr_ = fr.crossfall(s, k, 'secondary', False, np.full(len(s), 3.), np.full(len(s), 3.))
        self.assertTrue(np.allclose(fl, -.025) and np.allclose(fr_, -.025))     # two-way: crown
        # a left-hand curve at the motorway minimum radius in the middle: 6 % toward the inside (left lower)
        k = np.where((s > 800) & (s < 1200), 1 / 720., 0.)
        fl, fr_ = fr.crossfall(s, k, 'motorway', True, np.full(len(s), 6.), np.full(len(s), 7.))
        self.assertAlmostEqual(fl[np.searchsorted(s, 1000)], -.06)
        # run-off within the RAA limit: the edge 7 m out rises at most 0.9 % against the axis
        self.assertLessEqual(np.max(np.abs(np.diff(fl)) / np.diff(s)) * 7., .009 + 1e-9)
        # a built-up two-way road keeps its crown in a curve
        fl, fr_ = fr.crossfall(s, np.full(len(s), 1 / 100.), 'secondary', False, np.full(len(s), 3.), np.full(len(s), 3.), True)
        self.assertTrue(np.allclose(fl, -.025) and np.allclose(fr_, -.025))

    def test_q_of_radius(self):
        rmin, rout, qmax, verified = fr.BANK['motorway']
        self.assertEqual((rmin, rout, qmax, verified), (720., 4000., .06, True))   # RAA 2008 Tabelle 26

    def test_extents_follow_shoulders_in_travel_direction(self):
        a = seg((1, 2), [(0, 0), (500, 0)], 'motorway', lanes='3')
        st = fr.build_strokes([a], fr.Graph([a]))[0]
        sl, sr = fast_sources.SHOULDER['motorway']                  # (0.25, 3.3): the ALKIS / aerial hard shoulder
        lw = (a['width'] - sl - sr) / 3
        self.assertAlmostEqual(st.eL[0], 1.5 * lw + sl)
        self.assertAlmostEqual(st.eR[0], 1.5 * lw + sr)


class PerSideSections(unittest.TestCase):
    def test_official_edges_are_the_extents_and_swap_with_the_part(self):
        a = seg((1, 2), [(0, 0), (500, 0)], 'motorway_link', oneway='yes', lanes='1')
        a['edges'], a['lane'] = [2.3, 5.5], 3.9
        self.assertEqual(fr.section_of(a, False, True), (2.3, 5.5, 3.9))
        self.assertEqual(fr.section_of(a, True, True), (5.5, 2.3, 3.9))
        st = fr.build_strokes([a], fr.Graph([a]))[0]
        self.assertAlmostEqual(float(st.eL[3]), 2.3)
        self.assertAlmostEqual(float(st.eR[3]), 5.5)
        self.assertEqual(st.lanes[0][5], 3.9)                          # the lane width is not the shoulder's

    def test_loop_without_an_official_pavement_widens_inside_with_one_it_keeps_its_edges(self):
        pts = arc(0, 0, 60, 0., -math.pi, 40)                          # clockwise: a right-hand loop, R 60
        a = seg((1, 2), pts, 'motorway_link', oneway='yes', lanes='1')
        st = fr.build_strokes([a], fr.Graph([a]), fit=False)[0]
        self.assertGreater(float(st.eR.max()), float(st.eL.max()) + 1.)     # inside = right
        b = seg((1, 2), pts, 'motorway_link', oneway='yes', lanes='1')
        b['edges'], b['lane'] = [2.1, 4.0], 3.8
        st = fr.build_strokes([b], fr.Graph([b]), fit=False)[0]
        self.assertAlmostEqual(float(st.eL.max()), 2.1)
        self.assertAlmostEqual(float(st.eR.max()), 4.0)

    def test_crossfall_changes_take_100_m(self):
        s = np.arange(0, 3001, 5.)
        k = np.where((s > 1200) & (s < 1800), 1 / 720., 0.)
        fl, fr_ = fr.crossfall(s, k, 'motorway', True, np.full(len(s), 6.), np.full(len(s), 7.))
        self.assertAlmostEqual(fl[np.searchsorted(s, 1500)], -.06)
        self.assertLessEqual(float(np.max(np.abs(np.diff(fl)) / np.diff(s))), .085 / fr.CROSSFALL_EASE + 1e-9)
        # an S-bend: the flip from the one bank to the other, both ramps at once, still takes the 100 m
        k = np.where((s > 1000) & (s < 1300), 1 / 150., 0.) + np.where((s > 1300) & (s < 1600), -1 / 150., 0.)
        fl, _ = fr.crossfall(s, k, 'motorway_link', True, np.full(len(s), 3.), np.full(len(s), 6.))
        self.assertLessEqual(float(np.max(np.abs(np.diff(fl)) / np.diff(s))) * fr.CROSSFALL_EASE, (fl.max() - fl.min()) + 1e-9)


class Jogs(unittest.TestCase):
    def strokes(self):
        a = seg((1, 2), [(0, 0), (100, 0), (200, 0)], 'motorway_link', oneway='yes', lanes='1')
        b = seg((2, 3), [(200, 0), (208, -3.5), (240, -3.5), (400, -3.5)], 'motorway_link', oneway='yes', lanes='1')
        c = seg((2, 4), [(200, 0), (260, 50), (300, 120)], 'motorway_link', oneway='yes', lanes='1')
        segs = [a, b, c]
        return fr.build_strokes(segs, fr.Graph(segs))

    def worst_turn(self, strokes):
        st = max(strokes, key=lambda s: float(s.x.max() - s.x.min()))      # the through stroke a + b
        m = (st.x >= 190) & (st.x <= 250)
        return float(np.degrees(np.abs(st.h[m] - np.median(st.h[m])).max()))

    def test_an_s_shaped_step_beside_a_fork_is_smoothed(self):
        zone = fr.JOG_ZONE
        fr.JOG_ZONE = 150.
        n0 = fr.JOG_COUNT[0]
        try:
            smooth = self.worst_turn(self.strokes())
            self.assertEqual(fr.JOG_COUNT[0], n0 + 1)
            fr.JOG_ZONE = 0.
            kinked = self.worst_turn(self.strokes())
        finally:
            fr.JOG_ZONE = zone
        self.assertLess(smooth, 8.)
        self.assertGreater(kinked, 12.)

    def test_a_hook_at_the_start_of_a_branch_is_smoothed(self):
        def branch(zone):
            h1 = seg((1, 2), [(0, 0), (100, 0), (200, 0)], 'motorway_link', oneway='yes', lanes='1')
            h2 = seg((2, 3), [(200, 0), (300, 0), (400, 0)], 'motorway_link', oneway='yes', lanes='1')
            b = seg((2, 4), [(200, 0), (208, -3.5), (240, -3.5), (300, -20), (380, -80)], 'motorway_link', oneway='yes', lanes='1')
            old = fr.JOG_ZONE
            fr.JOG_ZONE = zone
            try:
                strokes = fr.build_strokes([h1, h2, b], fr.Graph([h1, h2, b]))
            finally:
                fr.JOG_ZONE = old
            st = min(strokes, key=lambda s: float(s.x.min() if False else abs(s.x[0] - 200.) + abs(s.y[0])))
            m = (st.x < 235) & (st.x >= 199)
            return float(np.degrees(np.abs(st.h[m]).max())), st
        n0 = fr.JOG_COUNT[0]
        smooth, st = branch(150.)
        self.assertEqual(fr.JOG_COUNT[0], n0 + 1)
        kinked, _ = branch(0.)
        self.assertLess(smooth, 12.)
        self.assertGreater(kinked, 20.)

    def test_a_real_bend_is_not_a_jog(self):
        a = seg((1, 2), [(0, 0), (100, 0), (200, 0)], 'motorway_link', oneway='yes', lanes='1')
        b = seg((2, 3), [(200, 0), (220, 12), (260, 40), (300, 100)], 'motorway_link', oneway='yes', lanes='1')
        c = seg((2, 4), [(200, 0), (260, -50), (300, -120)], 'motorway_link', oneway='yes', lanes='1')
        n0, zone = fr.JOG_COUNT[0], fr.JOG_ZONE
        fr.JOG_ZONE = 150.
        try:
            fr.build_strokes([a, b, c], fr.Graph([a, b, c]))
        finally:
            fr.JOG_ZONE = zone
        self.assertEqual(fr.JOG_COUNT[0], n0)


class Pinch(unittest.TestCase):
    def pieces(self, lanes_after=('1', '1'), length=15.):
        u = seg((1, 2), [(0, 0), (100, 0)], 'motorway_link', oneway='yes', lanes='2')
        p = seg((2, 3), [(100, 0), (100 + length, 0)], 'motorway_link', oneway='yes', lanes='1')
        x = seg((3, 4), [(100 + length, 0), (250, 20)], 'motorway_link', oneway='yes', lanes=lanes_after[0])
        y = seg((3, 5), [(100 + length, 0), (250, -20)], 'motorway_link', oneway='yes', lanes=lanes_after[1])
        return [u, p, x, y]

    def test_short_one_lane_piece_before_a_two_branch_fork_takes_the_lanes_before_it(self):
        segs = self.pieces()
        self.assertEqual(fr.lane_pinch(segs), 1)
        self.assertEqual(segs[1]['tags']['lanes'], '2')
        self.assertEqual(segs[0]['tags']['lanes'], '2')

    def test_not_when_the_branches_carry_fewer_lanes_or_the_piece_is_long(self):
        segs = self.pieces(lanes_after=('1', None))
        del segs[3]['tags']['lanes']
        self.assertEqual(fr.lane_pinch(segs), 0)
        self.assertEqual(fr.lane_pinch(self.pieces(length=60.)), 0)
        one = self.pieces()
        one[0]['tags']['lanes'] = '1'
        self.assertEqual(fr.lane_pinch(one), 0)


class MedianRails(unittest.TestCase):
    def build(self, offset, **tags):
        a = seg((1, 2), [(0, 0), (600, 0)], 'motorway', lanes='2')
        b = seg((3, 4), [(600, offset), (0, offset)], 'motorway', lanes='2')
        st = fr.build_strokes([a, b], fr.Graph([a, b]))
        return st

    def test_narrow_median_gets_one_rail_wide_and_mapped_ones_not(self):
        gap = 4.5
        st = self.build(0.)
        eA, eB = float(st[0].eL[3]), float(st[1].eL[3])
        st = self.build(eA + eB + gap)
        n = fr.median_rails(st)
        self.assertEqual(n, 1)
        rails = [(i, f) for i, s in enumerate(st) for f in s.furniture]
        self.assertEqual(len(rails), 1)
        f = rails[0][1]
        self.assertEqual(f['way'], 'rule:median')
        self.assertAlmostEqual(f['offset'], gap / 2, delta=.05)
        self.assertGreater(f['s1'] - f['s0'], 500.)
        # a median 7 m wide: none; a barrier mapped on the other carriageway: none
        st = self.build(eA + eB + 7.)
        self.assertEqual(fr.median_rails(st), 0)
        st = self.build(eA + eB + gap)
        for s in st:
            s.furniture.append({'kind': 'guard_rail', 'side': 1, 's0': 0., 's1': 600., 'offset': .5, 'way': '9'})
        self.assertEqual(fr.median_rails(st), 0)


class Exits(unittest.TestCase):
    def build(self):
        h1 = seg((1, 2), [(0, 0), (1000, 0)], 'motorway', lanes='2')
        h2 = seg((2, 3), [(1000, 0), (2000, 0)], 'motorway', lanes='2')
        link = seg((2, 5), [(1000, 0), (1060, -4), (1120, -12), (1200, -30), (1300, -70)], 'motorway_link', oneway='yes')
        segs = [h1, h2, link]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        fr.lane_events(strokes, g, segs)
        return {st.highway: st for st in strokes}

    def test_exit_has_a_parallel_lane_and_taper(self):
        st = self.build()
        H, B = st['motorway'], st['motorway_link']
        self.assertEqual(len(H.events), 1)
        e = H.events[0]
        lA, lZ = fr.EXIT['motorway']
        self.assertEqual(e['kind'], 'exit')
        self.assertEqual(e['side'], -1)
        self.assertAlmostEqual(e['parallel'] - e['taper'], lZ, places=6)
        self.assertAlmostEqual(e['separate'] - e['parallel'], lA, places=6)
        self.assertLess(e['separate'], e['nose'])
        # the host is a lane wider on the right over the parallel lane, as before outside the event
        mid = (e['parallel'] + e['separate']) / 2
        base = H.eR[0]
        self.assertAlmostEqual(np.interp(mid, H.s, H.eR) - base, e['width'], places=6)
        self.assertAlmostEqual(np.interp(e['taper'] - 50, H.s, H.eR), base, places=6)
        # the slip road starts on the added lane's centre, parallel to the host, and runs on without a gap or kink
        a = B.attach[0]
        self.assertAlmostEqual(B.y[0], -a['offset'], places=6)
        self.assertLess(abs(wrap(B.h[0])), 1e-9)
        d = np.diff(np.column_stack([B.x, B.y]), axis=0)
        self.assertLess(np.hypot(d[:, 0], d[:, 1]).max(), fr.MAX_STATION + 1e-6)
        turn = np.abs(wrap(np.diff(np.arctan2(d[:, 1], d[:, 0]))))
        self.assertLess(np.degrees(turn).max(), 3.)
        self.assertGreater(B.draw[0], 0.)

    def test_osm_mapped_deceleration_lane_is_the_added_lane(self):
        # OSM maps the deceleration lane as a third lane of the host up to the split node (250 m here)
        h0 = seg((1, 7), [(0, 0), (750, 0)], 'motorway', lanes='2')
        h1 = seg((7, 2), [(750, 0), (1000, 0)], 'motorway', lanes='3', **{'turn:lanes': 'none|none|slight_right'})
        h2 = seg((2, 3), [(1000, 0), (2000, 0)], 'motorway', lanes='2')
        link = seg((2, 5), [(1000, 0), (1060, -4), (1120, -12), (1200, -30), (1300, -70)], 'motorway_link', oneway='yes')
        segs = [h0, h1, h2, link]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        fr.lane_events(strokes, g, segs)
        H = [st for st in strokes if st.highway == 'motorway'][0]
        self.assertEqual(len(H.events), 1)
        e = H.events[0]
        # the parallel lane follows the mapped lane (AUX_SHIFT), the taper is AUX_TAPER
        self.assertAlmostEqual(e['parallel'], 750. + fr.AUX_SHIFT['exit'], places=6)
        self.assertAlmostEqual(e['parallel'] - e['taper'], fr.AUX_TAPER, places=6)
        # one added lane, not the mapped lane and another on top of it; the lane runs carry the through lanes
        base = H.eR[0]
        mid = (e['parallel'] + e['separate']) / 2
        self.assertAlmostEqual(np.interp(mid, H.s, H.eR) - base, e['width'], places=6)
        # the median side stays at the through section (no half lane towards the median)
        self.assertAlmostEqual(np.interp(mid, H.s, H.eL), H.eL[0], places=6)
        self.assertTrue(all(r[2] == 2 for r in H.lanes))
        self.assertEqual([r[6] for r in H.lanes if r[0] >= 749. and r[1] <= 1001.], ['none|none'])
        # S-shaped at the stations inside the taper (smoothstep; the game adds its own rows between them)
        inside = (H.s > e['taper']) & (H.s < e['parallel'])
        self.assertTrue(inside.any())
        u = (H.s[inside] - e['taper']) / fr.AUX_TAPER
        np.testing.assert_allclose(H.eR[inside] - base, e['width'] * u * u * (3 - 2 * u), atol=1e-9)

    def test_connector_keeps_both_cuts(self):
        # a link that leaves one motorway and joins another: drawn from the exit's nose to the entry's nose
        h1a = seg((1, 2), [(0, 0), (1000, 0)], 'motorway', lanes='2')
        h1b = seg((2, 3), [(1000, 0), (2500, 0)], 'motorway', lanes='2')
        h2a = seg((4, 5), [(0, -300), (1900, -300)], 'motorway', lanes='2')
        h2b = seg((5, 6), [(1900, -300), (3500, -300)], 'motorway', lanes='2')
        link = seg((2, 5), [(1000, 0), (1060, -4), (1150, -20), (1300, -80), (1450, -170), (1600, -240), (1750, -285),
                            (1900, -300)], 'motorway_link', oneway='yes')
        segs = [h1a, h1b, h2a, h2b, link]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        fr.lane_events(strokes, g, segs)
        B = [st for st in strokes if st.highway == 'motorway_link'][0]
        kinds = sorted(a['kind'] for a in B.attach)
        self.assertEqual(kinds, ['entry', 'exit'])
        self.assertGreater(B.draw[0], 0.)
        self.assertLess(B.draw[1], B.length() - 1.)
        for a in B.attach:
            self.assertTrue(0. <= a['lane'][0] < a['lane'][1] <= B.length() + 1e-6)

    def test_lane_gain_tapers_are_s_shaped_and_none_at_a_merge(self):
        a = seg((1, 2), [(0, 0), (500, 0)], 'motorway', lanes='2')
        b = seg((2, 3), [(500, 0), (1500, 0)], 'motorway', lanes='3')
        st = fr.build_strokes([a, b], fr.Graph([a, b]))[0]
        e0, e1 = st.eR[0], st.eR[-1]
        T = fr.TAPER['motorway']
        q = np.interp(500. + T / 4, st.s, st.eR)
        self.assertLess(q - e0, .2 * (e1 - e0))
        self.assertAlmostEqual(np.interp(500. + T / 2, st.s, st.eR) - e0, .5 * (e1 - e0), places=6)
        # a ramp joins at the node where the lane comes: the width changes at the node
        ramp = seg((4, 2), [(400, -40), (450, -10), (500, 0)], 'motorway_link', oneway='yes')
        segs = [a, b, ramp]
        H = [s for s in fr.build_strokes(segs, fr.Graph(segs)) if s.highway == 'motorway'][0]
        self.assertAlmostEqual(np.interp(510., H.s, H.eR), e1, places=6)

    def test_lane_attached_slip_road_has_its_role_and_no_plate(self):
        h1 = seg((1, 2), [(0, 0), (1000, 0)], 'motorway', lanes='2')
        h2 = seg((2, 3), [(1000, 0), (2000, 0)], 'motorway', lanes='2')
        link = seg((2, 5), [(1000, 0), (1060, -4), (1120, -12), (1200, -30), (1300, -70)], 'motorway_link', oneway='yes')
        segs = [h1, h2, link]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        fr.lane_events(strokes, g, segs)
        _, nodes = fr.records(strokes, g, segs, 1000.)
        rec = [n[3] for n in nodes if n[0] == 2][0]
        B = [st for st in strokes if st.highway == 'motorway_link'][0]
        self.assertEqual(rec['roles'], {B.id: ['lane', B.attach[0]['host']]})
        self.assertEqual(rec['plate'], [])


class Floors(unittest.TestCase):
    def test_bridge_over_a_navigable_river_gets_its_clearance(self):
        approach1 = seg((1, 2), [(-600, 0), (-150, 0)], 'motorway')
        bridge = seg((2, 3), [(-150, 0), (150, 0)], 'motorway', bridge='yes', layer='1')
        approach2 = seg((3, 4), [(150, 0), (600, 0)], 'motorway')
        segs = [approach1, bridge, approach2]
        water = [(7, ({'waterway': 'river', 'CEMT': 'Va'},), np.array([[0., -300.], [0., 300.]]))]
        extras = {'water': water, 'marks': [], 'barriers': []}
        dem = lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), 50.)      # inland: 30 m over the sea
        floors, report = fr.bridge_floors(segs, extras, dem)
        self.assertEqual(len(floors), 1)
        self.assertEqual(report[0]['rule'], 'cemt')
        self.assertAlmostEqual(report[0]['floor'], 50. + 5.25 + fr.structure_depth(300., bridge['tags']))
        # a river at the sea: the clearance counts from mean high water, not from the DEM's surface
        low = lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), fast_sources.LIFT - 1.6)
        _, tidal = fr.bridge_floors(segs, extras, low)
        self.assertAlmostEqual(tidal[0]['water'], fast_sources.LIFT + fr.MHW)
        # a seamark wins over the class
        extras['marks'] = [(9, {'seamark:bridge:clearance_height': '12.5'}, 3., 2.)]
        floors, report = fr.bridge_floors(segs, extras, dem)
        self.assertEqual((report[0]['rule'], report[0]['clearance']), ('seamark', 12.5))
        self.assertTrue(report[0]['approach_short'] is False)
        # the profile holds it: the stations over the channel at least at the floor, the approaches graded
        profiles, _ = fast_profile.solve(segs, dem, floors=floors)
        s, h = profiles[1]
        z = floors[0][3]
        over = (s >= floors[0][1]) & (s <= floors[0][2])
        self.assertGreaterEqual(h[over].min(), z - fast_profile.FLOOR_TOL)
        for ps, ph in profiles:
            self.assertLess(np.max(np.abs(np.diff(ph) / np.diff(ps))), fast_profile.GRADE_MAX['motorway'] + .006)

    def test_the_floors_are_off_unless_switched_on(self):
        self.assertEqual(fr.FLOORS, os.environ.get('EARTHNG_BRIDGE_FLOORS', '').strip().lower() not in ('', '0', 'off', 'no', 'false'))
        self.assertEqual(':floors' in fr.SIGNATURE, fr.FLOORS)

    def test_no_class_no_floor(self):
        bridge = seg((2, 3), [(-50, 0), (50, 0)], 'secondary', bridge='yes')
        water = [(7, ({'waterway': 'river'},), np.array([[0., -30.], [0., 30.]]))]
        floors, _ = fr.bridge_floors([bridge], {'water': water, 'marks': [], 'barriers': []}, lambda xy: np.zeros(len(xy)))
        self.assertEqual(floors, [])


class Codec(unittest.TestCase):
    def test_round_trip(self):
        h1 = seg((1, 2), [(5, 5), (900, 40)], 'motorway', lanes='2', name='A 1')
        g = fr.Graph([h1])
        strokes = fr.build_strokes([h1], g)
        strokes[0].furniture.append({'kind': 'guard_rail', 'side': -1, 's0': 10., 's1': 300., 'offset': 1.2, 'way': '77'})
        rows, nodes = fr.records(strokes, g, [h1], 1000.)
        chunks = [r[4] for r in rows]
        data = fr.encode_cell(chunks, [n[3] for n in nodes], (0., 0.))
        back, bnodes = fr.decode_cell(data, (0., 0.))
        self.assertEqual(len(back), len(chunks))
        for a, b in zip(chunks, back):
            self.assertEqual(a['stroke'], b['stroke'])
            for key in fr.STATION_KEYS:
                self.assertTrue(np.allclose(a['stations'][key], b['stations'][key], atol=1e-2))
            self.assertEqual(a['parts'][0]['tags'], b['parts'][0]['tags'])
        # chunks share their end stations exactly
        for a, b in zip(back, back[1:]):
            self.assertEqual([a['stations'][k][-1] for k in fr.STATION_KEYS], [b['stations'][k][0] for k in fr.STATION_KEYS])
        block, mask = fr.encode_block({0: data})
        self.assertEqual(block[:4], fr.MAGIC)
        self.assertEqual(fr.decode_block(block), {0: (len(block) - len(data), len(block))})

    def test_fan_plate_at_a_crossing(self):
        # a secondary road crossing a tertiary: one plate, every mouth cut by 0.8 x the widest road
        a = seg((1, 5), [(-300, 0), (0, 0)], 'secondary')
        b = seg((5, 2), [(0, 0), (300, 0)], 'secondary')
        c = seg((3, 5), [(0, -300), (0, 0)], 'tertiary')
        d = seg((5, 4), [(0, 0), (0, 300)], 'tertiary')
        segs = [a, b, c, d]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        rows, nodes = fr.records(strokes, g, segs, 1000.)
        rec = [n[3] for n in nodes if n[0] == 5][0]
        self.assertEqual(rec['kind'], 'cross')
        self.assertEqual(len(rec['plate']), 4)
        widest = max(st.eL[0] + st.eR[0] for st in strokes)
        for sid, d, cut, left, centre, right in rec['plate']:
            self.assertAlmostEqual(cut, round(fr.PLATE_CUT * widest, 2), places=2)
            self.assertAlmostEqual(math.hypot(centre[0], centre[1]), cut, delta=.05)
        data = fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.))
        _, back = fr.decode_cell(data, (0., 0.))
        got = [n for n in back if n['node'] == '5'][0]['plate']
        for p, q in zip(rec['plate'], got):
            self.assertEqual(p[:2], q[:2])
            for u, v in zip(p[3:], q[3:]):
                self.assertTrue(np.allclose(u, v, atol=.006))

    def test_side_road_mouth_clears_the_through_road(self):
        # a residential road joining a primary -> secondary change: its mouth is cut past the through road's edge
        a = seg((1, 5), [(-300, 0), (0, 0)], 'primary')
        b = seg((5, 2), [(0, 0), (300, 0)], 'secondary')
        c = seg((5, 4), [(0, 0), (0, 300)], 'residential')
        segs = [a, b, c]
        g = fr.Graph(segs)
        strokes = fr.build_strokes(segs, g)
        _, nodes = fr.records(strokes, g, segs, 1000.)
        rec = [n[3] for n in nodes if n[0] == 5][0]
        hw = {st.id: st.highway for st in strokes}
        side = [p for p in rec['plate'] if hw[p[0]] == 'residential'][0]
        self.assertGreater(side[2], max(st.eL[0] for st in strokes if st.highway != 'residential'))

    def test_code_identity_is_stable(self):
        self.assertEqual(fast_sources.code_identity(), fast_sources.code_identity())
        self.assertIn('fast_roads', fast_sources.INDEX_CODE)


if __name__ == '__main__':
    unittest.main()
