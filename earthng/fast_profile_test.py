"""Tests of the DEM rules in fast_profile (EARTHNG_DEM_RULES). Run: python earthng/fast_profile_test.py"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_profile as fp  # noqa: E402
import fast_sources  # noqa: E402


def setUpModule():
    # the German class rules are off unless asked for (fast_profile.set_road_rules)
    fast_sources.set_road_rules(True)


def tearDownModule():
    fast_sources.set_road_rules(False)

_ID = [0]


def seg(nodes, xy, highway='motorway', **tags):
    _ID[0] += 1
    t = dict(highway=highway, **tags)
    xy = [list(map(float, p)) for p in xy]
    return {'id': _ID[0], 'way': 1000 + _ID[0], 'tags': t, 'xy': xy, 'nodes': list(nodes),
            'width': fast_sources.road_width(t)}


def line(*pts, n0=1, highway='motorway'):
    """a chain of segments through the points, nodes n0, n0+1, ..."""
    return [seg((n0 + k, n0 + k + 1), [a, b], highway) for k, (a, b) in enumerate(zip(pts, pts[1:]))]


def at(profiles, segments, k, s):
    ps, ph = profiles[k]
    return float(np.interp(s, ps, ph))


def heights_along_x(profiles, segments, xs):
    """heights at x on segments that run along the x axis (y = 0)"""
    out = []
    for x in xs:
        for k, sg in enumerate(segments):
            (x0, y0), (x1, y1) = sg['xy'][0], sg['xy'][-1]
            if abs(y0) < 1e-6 and abs(y1) < 1e-6 and min(x0, x1) <= x <= max(x0, x1):
                out.append(at(profiles, segments, k, abs(x - x0)))
                break
    return np.array(out)


class Rules(unittest.TestCase):
    def test_off_by_default_and_cover_ignored_when_off(self):
        if 'EARTHNG_DEM_RULES' not in os.environ:
            self.assertFalse(fp.RULES)
        segs = line((0, 0), (1000, 0), (2000, 0))
        dem = lambda xy: 10. + 3. * np.exp(-((np.asarray(xy).reshape(-1, 2)[:, 0] - 1000) / 80.) ** 2)
        a, _ = fp.solve(segs, dem, rules=False)
        cover = fp.Cover([__import__('shapely').box(800, -300, 1200, 300)], [])
        b, _ = fp.solve(segs, dem, rules=False, cover=cover)
        for (s1, h1), (s2, h2) in zip(a, b):
            self.assertTrue(np.array_equal(h1, h2))

    def test_forest_hump_is_taken_out(self):
        import shapely
        # flat ground at 10 m; forest both sides over x 1500-2500 lifts the DEM by its canopy (8 m)
        segs = line((0, 0), (1500, 0), (2500, 0), (4000, 0))
        woods = [shapely.box(1500, 5, 2500, 400), shapely.box(1500, -400, 2500, -5)]
        cover = fp.Cover(woods, [])
        share = lambda xy: cover.forest_share(xy)
        dem = lambda xy: 10. + 8. * share(xy) * (fp.FOREST_BIAS / 8.)   # a DEM the rule fits exactly
        xs = np.arange(1700, 2301, 50.)
        raw, _ = fp.solve(segs, dem, rules=False)
        ruled, _ = fp.solve(segs, dem, rules=True, cover=cover)
        hr, hc = heights_along_x(raw, segs, xs), heights_along_x(ruled, segs, xs)
        self.assertGreater(hr.max() - 10., 4.)
        self.assertLess(np.abs(hc - 10.).max(), .5)

    def test_hump_under_a_deck_is_bridged_and_clearance_kept(self):
        # a motorway along x, a street bridge over it at x = 2000; the DEM carries the deck as a 3 m hump
        mw = line((0, 0), (1000, 0), (3000, 0), (4000, 0))
        ramp1 = seg((20, 21), [(2000, -400), (2000, -30)], 'secondary')
        deck = seg((21, 22), [(2000, -30), (2000, 30)], 'secondary', bridge='yes', layer='1')
        ramp2 = seg((22, 23), [(2000, 30), (2000, 400)], 'secondary')
        segs = mw + [ramp1, deck, ramp2]
        def dem(xy):
            xy = np.asarray(xy).reshape(-1, 2)
            return 10. + 3. * np.exp(-((xy[:, 0] - 2000) ** 2 + xy[:, 1] ** 2) / (2 * 60. ** 2))
        xs = np.arange(1800, 2201, 25.)
        raw, _ = fp.solve(segs, dem, rules=False)
        ruled, _ = fp.solve(segs, dem, rules=True)
        hr, hc = heights_along_x(raw, segs, xs), heights_along_x(ruled, segs, xs)
        self.assertGreater(hr.max() - 10., 1.2)         # the motorway keeps much of the DEM's hump (HOLD_UNDER)
        self.assertLess(hc.max() - 10., 1.)             # the rules bridge it from the clean ground
        # the deck still clears the motorway under it
        top = at(ruled, segs, 5, 30.)
        self.assertGreater(top - hc[np.argmin(np.abs(xs - 2000))], fp.CLEARANCE - .2)
        # and the bridged stretch keeps the class's grade
        for (ps, ph), sg in zip(ruled, segs):
            g = np.abs(np.diff(ph) / np.diff(ps))
            self.assertLess(g.max(), fp.GRADE_MAX.get(sg['tags']['highway'], fp.GRADE_DEFAULT) + .01)

    def test_single_outlier_is_down_weighted(self):
        segs = line((0, 0), (2000, 0))
        spike_x = 1000.

        def dem(xy):
            xy = np.asarray(xy).reshape(-1, 2)
            return np.where(np.abs(xy[:, 0] - spike_x) < 1., 30., 10.)
        weight = np.ones(101)
        pos = np.column_stack([np.linspace(0, 2000, 101), np.zeros(101)])
        h0 = dem(pos)
        mass = np.full(101, 20.)
        ea, eb = np.arange(100), np.arange(1, 101)
        ta, tb, tc = np.arange(99), np.arange(1, 100), np.arange(2, 101)
        chains = fp.Chains(np.arange(101), np.array([0]), 101, 101, [])
        old = fp.IRLS
        try:
            fp.IRLS = 0
            plain, _ = fp.ground_line(h0, pos, mass, weight, (ea, eb, fp.ALPHA / 20. * np.ones(100)),
                                      (ta, tb, tc, 50. ** 4 / 20. ** 3 * np.ones(99)), chains)
            fp.IRLS = 4
            robust, low = fp.ground_line(h0, pos, mass, weight, (ea, eb, fp.ALPHA / 20. * np.ones(100)),
                                         (ta, tb, tc, 50. ** 4 / 20. ** 3 * np.ones(99)), chains)
        finally:
            fp.IRLS = old
        self.assertLess(robust[50] - 10., (plain[50] - 10.) * .6)
        self.assertLess(np.abs(robust[np.r_[0:45, 56:101]] - 10.).max(), .2)
        self.assertGreater(low, 0.)


class Helpers(unittest.TestCase):
    def test_near_distance_matches_brute_force(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1000, (500, 2)); t = rng.uniform(0, 1000, (60, 2))
        d = fp.near_distance(p, t, 120.)
        brute = np.sqrt(((p[:, None, :] - t[None, :, :]) ** 2).sum(-1)).min(1)
        brute[brute > 120.] = np.inf
        self.assertTrue(np.allclose(d, brute))

    def test_forest_share(self):
        import shapely
        cover = fp.Cover([shapely.box(0, -5000, 5000, 5000)], [])
        s = cover.forest_share(np.array([[-1000., 0.], [0., 0.], [1000., 0.]]))
        self.assertLess(s[0], .01)
        self.assertAlmostEqual(s[1], .5, delta=.08)
        self.assertGreater(s[2], .99)

    def test_share_grid_is_global_bilinear_and_round_trips(self):
        import shapely
        cover = fp.Cover([shapely.box(0, -5000, 5000, 5000)], [])
        # two different query windows read the same share at the same point (no seams between cells)
        a = cover.share_grid((-300., -300., 300., 300.)).at([[17., 3.]])
        b = cover.share_grid((-2000., -100., 900., 2500.)).at([[17., 3.]])
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=6)
        g = cover.share_grid((-1000., -1000., 1000., 1000.))
        x = np.linspace(-400., 400., 801)
        s = g.at(np.column_stack([x, np.zeros_like(x)]))
        self.assertLess(np.abs(np.diff(s)).max(), .005)              # bilinear: no 20 m stair steps
        self.assertTrue(g.covers([[0., 0.], [900., -900.]]))
        self.assertFalse(g.covers([[5000., 0.]]))
        cover.grid = g
        self.assertTrue(np.allclose(cover.forest_share([[0., 0.]]), g.at([[0., 0.]])))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'forest.npz')
            g.save(path)
            back = fp.ShareGrid.load(path)
            self.assertTrue(np.array_equal(back.at(np.column_stack([x, x])), g.at(np.column_stack([x, x]))))

    def test_structure_weight(self):
        w = fp.structure_weight(np.array([0., fp.STRUCT_NEAR, fp.STRUCT_FAR, np.inf]))
        self.assertAlmostEqual(w[0], fp.STRUCT_W)
        self.assertAlmostEqual(w[1], fp.STRUCT_W)
        self.assertAlmostEqual(w[2], 1.)
        self.assertAlmostEqual(w[3], 1.)

    def test_read_cover(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6">
 <node id="1" lat="50.000" lon="15.000"/><node id="2" lat="50.000" lon="15.010"/>
 <node id="3" lat="50.010" lon="15.010"/><node id="4" lat="50.010" lon="15.000"/>
 <node id="5" lat="50.020" lon="15.000"/><node id="6" lat="50.020" lon="15.010"/>
 <node id="7" lat="50.030" lon="15.000"/><node id="8" lat="50.030" lon="15.010"/>
 <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="landuse" v="forest"/></way>
 <way id="11"><nd ref="5"/><nd ref="6"/><tag k="railway" v="rail"/><tag k="bridge" v="yes"/></way>
 <way id="12"><nd ref="7"/><nd ref="8"/><tag k="highway" v="primary"/><tag k="bridge" v="yes"/></way>
</osm>
"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'tile.osm')
            with open(path, 'w', encoding='utf8') as f:
                f.write(xml)
            part = fp.read_cover(path)
        self.assertEqual(len(part['forest']), 1)
        self.assertEqual(list(part['structures']), [11])      # roads are in the graph already
        cover = fp.cover_of([part, part], fast_sources.Frame.at(15.005, 50.015))
        self.assertEqual((len(cover.forest), len(cover.structures)), (1, 1))
        self.assertAlmostEqual(cover.forest[0].area, 715. * 1112., delta=0.05 * 715. * 1112.)


if __name__ == '__main__':
    unittest.main()
