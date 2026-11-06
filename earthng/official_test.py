"""Tests of official (ALKIS / ATKIS width prior, DGM1 medians). Run: python earthng/official_test.py"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_roads  # noqa: E402
import fast_sources  # noqa: E402
import official  # noqa: E402

CX, CY = 650000., 5400000.            # the frame origin: ETRS89 / UTM 32 and WGS 84 / UTM 32 agree to the mm, so the data tile is 650_5400
FRAME = fast_sources.Frame('EPSG:32632', (CX, CY))


def rect(x0, y0, x1, y1):
    """a closed ring in the data's CRS (frame x, y + origin)"""
    return [[CX + x0, CY + y0], [CX + x1, CY + y0], [CX + x1, CY + y1], [CX + x0, CY + y1], [CX + x0, CY + y0]]


def tile(polys, directory, dlm=()):
    """polys: [(ring, bez)] as fetchpack11 stores them; written as the data tile 650_5400"""
    alkis = [{'id': f'p{k}', 'n': 'Straßenverkehr', 'b': bez, 'r': [['e', ring]]} for k, (ring, bez) in enumerate(polys)]
    with open(os.path.join(directory, '650_5400.json'), 'w', encoding='utf8') as f:
        json.dump({'alkis': alkis, 'dlm': list(dlm)}, f)


def draft(way, pts, nodes, ordinal=0, **tags):
    t = {'highway': 'primary', **tags}
    return {'way': way, 'ord': ordinal, 'tags': t, 'pts': [(float(x), float(y)) for x, y in pts], 'nodes': nodes,
            'width': fast_sources.road_width(t)}


def run(drafts, directory, height=None):
    data = official.Data.load(FRAME, (-50., -50., 1000., 1000.), directory)
    return official.refine(drafts, FRAME, fast_roads.travel, fast_sources.shoulder, fast_sources.oneway, height=height, data=data)


def dual(lanes=1, **tags):
    """a dual carriageway along x: east-bound at y = -5, west-bound at y = +5, both ends in a bend (no junction)"""
    return [draft(1, [(100, -5), (500, -5)], (11, 12), oneway='yes', lanes=str(lanes), **tags),
            draft(2, [(500, 5), (100, 5)], (22, 21), oneway='yes', lanes=str(lanes), **tags)]


class Bump(unittest.TestCase):
    u = np.arange(-14., 14.01, .5)

    def profile(self, centre, width, height):
        return np.where(np.abs(self.u - centre) <= width / 2, height, 0.)

    def test_raised_strip_found(self):
        r = official.bump(self.u, self.profile(0., 3., .12), -9., 9., -5., 5.)
        self.assertIsNotNone(r)
        self.assertAlmostEqual((r[0] + r[1]) / 2, 0., delta=.3)
        self.assertAlmostEqual(r[1] - r[0], 3., delta=.6)

    def test_flat_low_and_thin_strips_are_not_medians(self):
        self.assertIsNone(official.bump(self.u, np.zeros_like(self.u), -9., 9., -5., 5.))
        self.assertIsNone(official.bump(self.u, self.profile(0., 3., .05), -9., 9., -5., 5.))      # kerb-less verge
        self.assertIsNone(official.bump(self.u, self.profile(0., .5, .15), -9., 9., -5., 5.))      # a kerb stone line
        self.assertIsNone(official.bump(self.u, self.profile(0., 3., .6), -9., 9., -5., 5.))       # an embankment

    def test_outside_the_pair_is_not_the_median(self):
        self.assertIsNone(official.bump(self.u, self.profile(7., 3., .12), -9., 9., -5., 5.))      # a verge beyond a carriageway


class Pieces(unittest.TestCase):
    def test_equal_samples_are_one_piece(self):
        s = np.arange(10.) * 20 + 10
        self.assertEqual(official.plan_pieces(s, np.full(10, 7.), np.full(10, 2), 200.), [(0., 200., 7., 2)])

    def test_a_step_makes_two_pieces_and_a_short_one_is_absorbed(self):
        s = np.arange(20.) * 20 + 10
        W = np.r_[np.full(10, 7.), np.full(10, 10.)]
        p = official.plan_pieces(s, W, np.full(20, 2), 400.)
        self.assertEqual([(a, b) for a, b, _, _ in p], [(0., 200.), (200., 400.)])
        W2 = np.r_[np.full(8, 7.), [10.], np.full(11, 7.)]
        self.assertEqual(len(official.plan_pieces(s, W2, np.full(20, 2), 400.)), 1)

    def test_cut_polyline_interpolates_the_ends(self):
        c = official.cut_polyline([(0, 0), (100, 0), (100, 100)], 50., 150.)
        self.assertEqual(c.tolist(), [[50., 0.], [100., 0.], [100., 50.]])


class WidthPrior(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_dual_carriageway_tagged_one_lane_gets_two(self):
        # J09 south in the study: the envelope is 18 m, each way is tagged lanes=1 and drawn 4-5 m
        tile([(rect(0, -9, 600, 9), None)], self.dir.name)
        out, islands, stats = run(dual(), self.dir.name)
        self.assertEqual(islands, [])
        for d in out:
            self.assertEqual(d['tags']['lanes'], '2')
            self.assertAlmostEqual(d['width'], 9., delta=.1)                 # 4 m outer + 5 m to the midway line
        self.assertEqual(stats['lanes_raised'], 2)
        self.assertEqual(len(out), 2)

    def test_lanes_are_never_lowered_and_widths_not_below_85_percent(self):
        tile([(rect(0, -9, 600, 9), None)], self.dir.name)
        d = dual(lanes=3)
        out, _, _ = run(d, self.dir.name)
        for o in out:
            self.assertEqual(o['tags']['lanes'], '3')
        # a narrow envelope under a wide default
        tile([(rect(0, -7, 600, -3), None)], self.dir.name)
        one = [draft(1, [(100, -5), (500, -5)], (11, 12), lanes='2')]
        out, _, _ = run(one, self.dir.name)
        self.assertGreaterEqual(out[0]['width'], .85 * one[0]['width'] - 1e-6)

    def test_bez_polygons_are_not_carriageway(self):
        tile([(rect(0, -9, 600, 9), 'Begleitfläche Straßenverkehr')], self.dir.name)
        out, _, stats = run(dual(), self.dir.name)
        self.assertEqual(stats['valid'], 0)
        self.assertEqual([o['width'] for o in out], [d['width'] for d in dual()])

    def test_plaza_is_not_a_road(self):
        tile([(rect(0, -40, 600, 40), None)], self.dir.name)
        one = [draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2')]
        out, _, stats = run(one, self.dir.name)
        self.assertEqual(out[0]['width'], one[0]['width'])
        self.assertGreater(stats['plaza'], 0)

    def test_two_way_road_keeps_its_lanes(self):
        # a two-way street's envelope holds parking, cycle lanes and turn pockets: wider, never more lanes (1 Nov)
        tile([(rect(0, -6.5, 600, 6.5), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2')], self.dir.name)
        self.assertEqual(out[0].get('tags', {}).get('lanes', '2'), '2')
        self.assertLessEqual(out[0]['width'], 2 * official.LANE_CAP + official.EDGE_STRIP + 1.)   # capped to 2 lanes
        tile([(rect(0, -4.5, 600, 4.5), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2')], self.dir.name)
        self.assertEqual(out[0]['tags']['lanes'], '2')                       # 9 m (parking): wider, lanes stay
        self.assertLessEqual(out[0]['width'], 2 * official.LANE_CAP + 1.)

    def test_osm_width_tag_within_tolerance_stands(self):
        tile([(rect(0, -3.5, 600, 3.5), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', width='6.5')], self.dir.name)
        self.assertEqual(out[0]['width'], 6.5)

    def test_junction_mouths_are_left_out_and_the_cross_section_changes_by_pieces(self):
        # envelope 7 m for the first half, 12 m for the second; both ends at junctions (three ways meet there)
        tile([(rect(0, -3.5, 300, 3.5), None), (rect(300, -6, 600, 6), None)], self.dir.name)
        way = draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2')
        side = [draft(7, [(100, 0), (100, 50)], (11, 13)), draft(8, [(100, 0), (100, -50)], (11, 14)),
                draft(9, [(500, 0), (500, 50)], (12, 15)), draft(10, [(500, 0), (500, -50)], (12, 16))]
        out, _, stats = run([way] + side, self.dir.name)
        pieces = [o for o in out if o['way'] == 1]
        self.assertEqual(len(pieces), 2)
        self.assertLess(pieces[0]['width'], pieces[1]['width'])
        self.assertEqual(pieces[0]['nodes'][1], pieces[1]['nodes'][0])       # the new node joins them
        self.assertLess(pieces[0]['nodes'][1], 0)
        self.assertEqual((pieces[0]['nodes'][0], pieces[1]['nodes'][1]), (11, 12))
        self.assertAlmostEqual(float(np.linalg.norm(pieces[0]['pts'][-1] - pieces[1]['pts'][0])), 0., places=6)

    def test_a_short_arm_between_junctions_keeps_its_middle_cross_sections(self):
        # a 44 m roundabout arm: both ends are junction nodes, the 20 m mouth rule alone would leave nothing
        tile([(rect(0, -2.8, 600, 2.8), None)], self.dir.name)
        arm = draft(1, [(100, 0), (144, 0)], (11, 12), lanes='1', oneway='yes', highway='secondary')
        side = [draft(7, [(100, 0), (100, 50)], (11, 13)), draft(8, [(100, 0), (100, -50)], (11, 14)),
                draft(9, [(144, 0), (144, 50)], (12, 15)), draft(10, [(144, 0), (144, -50)], (12, 16))]
        out, _, stats = run([arm] + side, self.dir.name)
        self.assertGreater(stats['valid'], 0)
        self.assertGreater([o for o in out if o['way'] == 1][0]['width'], arm['width'] + .3)
        self.assertEqual([o for o in out if o['way'] == 1][0]['tags']['lanes'], '1')     # 5.6 m: one wide lane + edge strip

    def test_bridge_roundabout_and_residential_are_left_alone(self):
        tile([(rect(0, -9, 600, 9), None)], self.dir.name)
        ds = [draft(1, [(100, -5), (500, -5)], (1, 2), bridge='yes'), draft(2, [(100, 0), (500, 0)], (3, 4), junction='roundabout'),
              draft(3, [(100, 3), (500, 3)], (5, 6))]
        ds[2]['tags']['highway'] = 'residential'
        out, _, stats = run(ds, self.dir.name)
        self.assertEqual(stats['ways'], 0)
        self.assertEqual(out, ds)

    def test_basis_dlm_width_where_alkis_is_absent(self):
        tile([], self.dir.name, dlm=[{'t': 'Strassenachse', 'w': '9.0', 'l': '2', 'pts': [[CX + 0, CY + 0.5], [CX + 600, CY + .5]]}])
        out, _, stats = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2')], self.dir.name)
        self.assertEqual(stats['dlm_ways'], 1)
        self.assertGreater(out[0]['width'], 7.5)
        self.assertEqual(out[0]['tags']['lanes'], '2')                       # lanes come from OSM only

    def test_deterministic(self):
        tile([(rect(0, -9, 600, 9), None)], self.dir.name)
        a, ia, sa = run(dual(), self.dir.name)
        b, ib, sb = run(dual(), self.dir.name)
        self.assertEqual([(d['width'], d['tags'], d['nodes']) for d in a], [(d['width'], d['tags'], d['nodes']) for d in b])
        self.assertEqual(sa, sb)


class PerSide(unittest.TestCase):
    """bundle8: the pavement's position against the OSM line, the hard shoulder, the lane width (1 Nov 2026)"""
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def edges(self, y0, y1, **tags):
        tile([(rect(0, y0, 600, y1), None)], self.dir.name)
        t = dict(highway='motorway_link', lanes='1', oneway='yes')
        t.update(tags)
        d = draft(1, [(100, 0), (500, 0)], (11, 12), **t)
        d['tags'] = {k: v for k, v in t.items()}
        d['width'] = fast_sources.road_width(d['tags'])
        out, _, stats = run([d], self.dir.name)
        return out, stats

    def test_two_way_road_keeps_the_envelopes_position(self):
        tile([(rect(0, -3.5, 600, 4.5), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', highway='secondary')], self.dir.name)
        l, r = out[0]['edges']
        self.assertAlmostEqual(l, 4.5, delta=.15)                     # left of +x is +y
        self.assertAlmostEqual(r, 3.5, delta=.15)
        self.assertAlmostEqual(l + r, out[0]['width'], delta=.02)

    def test_link_hard_shoulder_is_on_the_right_of_travel(self):
        out, stats = self.edges(-5.5, 2.3)
        p = out[0]
        self.assertAlmostEqual(p['edges'][0], 2.3, delta=.15)
        self.assertAlmostEqual(p['edges'][1], 5.5, delta=.15)
        self.assertLessEqual(p['lane'], 4.0 + 1e-9)                   # the lane stays a lane
        self.assertEqual(stats['hard_pieces'], 1)
        # the same road mapped against its direction (oneway=-1): the right of travel is the node order's left
        out, _ = self.edges(-2.3, 5.5, oneway='-1')
        self.assertAlmostEqual(out[0]['edges'][0], 5.5, delta=.15)
        self.assertAlmostEqual(out[0]['edges'][1], 2.3, delta=.15)

    def test_link_hard_shoulder_is_capped(self):
        out, _ = self.edges(-9., 2.3)
        half = 2.0                                                    # one 4 m lane
        self.assertLessEqual(out[0]['edges'][1], half + official.HARD_SHOULDER + official.EDGE_STRIP + .01)

    def test_a_pavement_that_shifts_makes_a_new_piece(self):
        # the right edge moves out by 2 m half way: same width, so the old rule saw one piece
        tile([(rect(0, -3, 300, 3), None), (rect(300, -5, 600, 1), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', highway='trunk')], self.dir.name)
        self.assertEqual(len(out), 2)
        self.assertGreater(out[1]['edges'][1], out[0]['edges'][1] + 1.5)
        # other classes are cut by width and lanes only (their joints are seams of the generator's roads), the edges ride along
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', highway='secondary')], self.dir.name)
        self.assertEqual(len(out), 1)

    def test_an_offcentre_pavement_is_not_followed_beyond_2_m_on_ordinary_roads(self):
        tile([(rect(0, -2, 600, 9), None)], self.dir.name)
        out, _, _ = run([draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', highway='secondary')], self.dir.name)
        l, r = out[0]['edges']
        self.assertAlmostEqual(l - r, 2 * official.MAX_SHIFT, delta=.05)
        self.assertAlmostEqual(l + r, out[0]['width'], delta=.02)

    def test_a_neighbouring_residential_street_leaves_room_to_its_own_width(self):
        # a tertiary beside a residential street 10 m to its right (6 m wide), inside one envelope: it may use the 7 m up to that
        # street's own edge, not only the midway 5 m (a plaza-wide envelope; the 2 m shift and the width cap still apply)
        tile([(rect(0, -12, 600, 4), None)], self.dir.name)
        ds = [draft(1, [(100, 0), (500, 0)], (11, 12), lanes='2', highway='tertiary'), draft(2, [(100, -10), (500, -10)], (13, 14), highway='residential')]
        out, _, _ = run(ds, self.dir.name)
        way1 = [o for o in out if o['way'] == 1][0]
        self.assertAlmostEqual(way1['edges'][0] + way1['edges'][1], way1['width'], delta=.02)
        self.assertGreater(way1['width'], 7.5)                        # the midway split alone gave 4 + 5 = 9, capped; the room is not the limit
        self.assertGreater(way1['edges'][1], 3.5)

    def test_plan_pieces_with_edges(self):
        s = np.arange(20.) * 20 + 10
        E = np.column_stack([np.r_[np.full(10, 3.), np.full(10, 3.)], np.r_[np.full(10, 3.), np.full(10, 5.)]])
        p = official.plan_pieces(s, E.sum(1) * 0 + 6., np.full(20, 2), 400., E=E)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[1][4:], (3., 5.))


class Medians(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        tile([(rect(0, -9, 600, 9), None)], self.dir.name)

    @staticmethod
    def ground(strip=None, height=.12):
        """a DTM: a slight fall across and a raised strip y in strip (frame coordinates, 2 sloping kerb pixels)"""
        def h(xy):
            xy = np.asarray(xy, float)
            y = np.floor(xy[:, 1]) + .5           # 1 m pixels
            z = 50. - .005 * y
            if strip is not None:
                z = z + np.where((y >= strip[0]) & (y <= strip[1]), height, 0.)
            return z
        return h

    def test_median_between_a_pair_is_detected_narrows_the_carriageways_and_is_emitted(self):
        out, islands, stats = run(dual(), self.dir.name, height=self.ground((-1.5, 1.5)))
        self.assertGreater(stats['median_samples'], 10)
        self.assertEqual(len(islands), 1)
        ring = np.asarray(islands[0])
        self.assertAlmostEqual(ring[:, 1].min(), -1.5, delta=1.)
        self.assertAlmostEqual(ring[:, 1].max(), 1.5, delta=1.)
        self.assertGreater(ring[:, 0].max() - ring[:, 0].min(), 300.)
        for d in out:
            self.assertAlmostEqual(d['width'], 4. + 3.5, delta=.8)           # 4 m outer side + to the median's edge
            self.assertEqual(d['tags']['lanes'], '2')

    def test_no_median_without_relief_or_without_dtm(self):
        _, islands, _ = run(dual(), self.dir.name, height=self.ground(None))
        self.assertEqual(islands, [])
        out, islands, stats = run(dual(), self.dir.name, height=None)
        self.assertEqual((islands, stats['median_samples']), ([], 0))
        self.assertAlmostEqual(out[0]['width'], 9., delta=.1)                # the midway line

    def test_a_one_way_road_alone_has_no_median(self):
        _, islands, stats = run([dual()[0]], self.dir.name, height=self.ground((-1.5, 1.5)))
        self.assertEqual((islands, stats['pair_samples']), ([], 0))


class Switch(unittest.TestCase):
    def test_default_is_off_and_credits_name_the_source(self):
        self.assertFalse(official.ENABLED or os.environ.get('EARTHNG_OFFICIAL'))
        c = official.credits()[0]
        self.assertIn('CC BY 4.0', c['license'])
        self.assertIn('LGLN', c['credit'])

    def test_identity_changes_with_the_data(self):
        with tempfile.TemporaryDirectory() as d:
            old = official.DATA
            official.DATA = Path(d)
            try:
                a = official.identity()
                Path(d, '650_5400.json').write_text('{}')
                self.assertNotEqual(a, official.identity())
            finally:
                official.DATA = old


if __name__ == '__main__':
    unittest.main()
