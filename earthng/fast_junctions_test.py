"""Tests of fast_junctions (phase 3 junction surfaces). Run: python earthng/fast_junctions_test.py"""
import math
import os
import sys
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_junctions as fj  # noqa: E402


def view(sid, pts, highway='secondary', oneway=False, eL=3.5, eR=3.5, z=10., grade=0., fl=-.025, fr=-.025, tags=None,
         nodes=(), lanes=None):
    """a straight or polyline stroke sampled every metre, with its stations"""
    P = np.asarray(pts, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.r_[0., np.cumsum(seg)]
    s = np.arange(0., cum[-1] + 1e-9, 1.)
    if s[-1] < cum[-1] - 1e-6:
        s = np.r_[s, cum[-1]]
    x, y = np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])
    h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    n = len(s)
    arrays = dict(s=s, x=x, y=y, z=z + grade * s, h=h, eL=np.full(n, eL), eR=np.full(n, eR), fl=np.full(n, fl), fr=np.full(n, fr))
    t = dict(highway=highway, **(tags or {}))
    lanes = lanes or [[0., float(s[-1]), 1, 0 if oneway else 1, 0, 3.5, '', '']]
    return fj.View(sid, highway, oneway, False, arrays, list(nodes), [(0., float(s[-1]), False, t)], lanes)


def node(nid, x, y, z, mouths, kind='junction'):
    return {'node': nid, 'kind': kind, 'xyz': [x, y, z], 'mouths': mouths, 'roles': {}}


def crossroads(minor='residential', major='secondary', tags_minor=None, tags_major=None, oneway_major=False,
               lanes_major=None):
    """a major road west-east through (0, 0) and a minor one south-north, both through"""
    A = view('A', [(-100, 0), (100, 0)], major, oneway=oneway_major, fl=.025 if oneway_major else -.025, tags=tags_major,
             nodes=[('n0', 100.)], lanes=lanes_major)
    B = view('B', [(0, -80), (0, 80)], minor, eL=2.75, eR=2.75, tags=tags_minor, nodes=[('n0', 80.)])
    rec = node('n0', 0., 0., 10., [['A', 100., -1], ['A', 100., 1], ['B', 80., -1], ['B', 80., 1]], 'cross')
    return {'A': A, 'B': B}, [rec]


def tee():
    A = view('A', [(-100, 0), (100, 0)], 'primary', nodes=[('n0', 100.)])
    B = view('B', [(0, 0), (0, 80)], 'residential', eL=2.75, eR=2.75, nodes=[('n0', 0.)])
    rec = node('n0', 0., 0., 10., [['A', 100., -1], ['A', 100., 1], ['B', 0., 1]], 'junction')
    return {'A': A, 'B': B}, [rec]


class Clusters(unittest.TestCase):
    def test_close_nodes_merge_far_ones_stay(self):
        # a dual carriageway crossed by a minor road: two nodes 14 m apart are one junction; a third 60 m on is not
        N = view('N', [(-100, 7), (100, 7)], 'primary', oneway=True, nodes=[('a', 100.), ('c', 160.)])
        S = view('S', [(100, -7), (-100, -7)], 'primary', oneway=True, nodes=[('b', 100.)])
        M = view('M', [(0, -80), (0, 80)], 'tertiary', nodes=[('b', 73.), ('a', 87.)])
        C = view('C', [(60, 7), (60, 80)], 'residential', nodes=[('c', 0.)])
        recs = [node('a', 0, 7, 10, [['N', 100., -1], ['N', 100., 1], ['M', 87., -1], ['M', 87., 1]], 'cross'),
                node('b', 0, -7, 10, [['S', 100., -1], ['S', 100., 1], ['M', 73., -1], ['M', 73., 1]], 'cross'),
                node('c', 60, 7, 10, [['N', 160., -1], ['N', 160., 1], ['C', 0., 1]], 'junction')]
        got = fj.clusters(recs, {'N': N, 'S': S, 'M': M, 'C': C})
        self.assertEqual(got, [['a', 'b'], ['c']])

    def test_roundabout_ring_is_one_cluster(self):
        ring = [(15 * math.cos(a), 15 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 73)]
        R = view('R', ring, 'secondary', oneway=True, fl=.025, fr=-.025, tags={'junction': 'roundabout'},
                 nodes=[('r0', 0.), ('r1', 23.6), ('r2', 47.1), ('r3', 70.7), ('r0', 94.2)])
        recs = [node('r%d' % k, *ring[k * 18], 10., [['R', 23.56 * k, -1], ['R', 23.56 * k, 1]], 'junction') for k in range(4)]
        got = fj.clusters(recs, {'R': R})
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 4)


class Surface(unittest.TestCase):
    def check_mesh(self, r):
        V, T = np.asarray(r['verts']), np.asarray(r['tris'])
        cnt = Counter()
        for a, b, c in T:
            for e in ((a, b), (b, c), (c, a)):
                cnt[(min(e), max(e))] += 1
        self.assertFalse([e for e, n in cnt.items() if n > 2], 'non-manifold edges')
        # every triangle counter-clockwise (the game draws them facing up)
        area = (V[T[:, 1], 0] - V[T[:, 0], 0]) * (V[T[:, 2], 1] - V[T[:, 0], 1]) - (V[T[:, 1], 1] - V[T[:, 0], 1]) * (V[T[:, 2], 0] - V[T[:, 0], 0])
        self.assertTrue((area > 0).all())
        # the cut rows: the mesh's vertices on them have the road section's z exactly
        for m in r['mouths']:
            for P, Q in ((m['R'], m['C']), (m['C'], m['L'])):
                P3, Q3 = np.asarray(P), np.asarray(Q)
                d = Q3[:2] - P3[:2]
                t = np.clip(((V[:, :2] - P3[:2]) @ d) / (d @ d), 0, 1)
                on = np.hypot(*(V[:, :2] - (P3[:2] + t[:, None] * d)).T) < .01
                self.assertTrue(on.sum() >= 2)
                self.assertLess(np.abs(V[on, 2] - (P3[2] + (Q3[2] - P3[2]) * t[on])).max(), .0015)

    def test_crossroads_fillets_cuts_and_heights(self):
        V, recs = crossroads(minor='unclassified')
        r = fj.junction(['n0'], recs, V, {})
        self.assertIsNotNone(r)
        self.check_mesh(r)
        self.assertEqual(len(r['mouths']), 4)
        # four kerb fillets, rural corners of a minor road (KERB, not verified): 12 m
        self.assertEqual(len(r['radii']), 4)
        self.assertTrue(all(abs(x - fj.KERB['rural']['minor']) < 1e-6 for x in r['radii']))
        # the minor road is cut past its fillet: at least the radius beyond the major's paved edge
        for m in r['mouths']:
            if m['stroke'] == 'B':
                self.assertGreater(m['cut'], 3.5 + 12. - 1.)
        # the major's cross-fall is kept inside the junction: the crown at the centre, its edge 2.5 % lower
        Vt = np.asarray(r['verts'])
        z = fj.height_field(np.array([[0., 0.], [0., 3.4]]), [dict(m, v=V[m['stroke']], long=60.) for m in r['mouths']],
                            [], ['A'], V)
        self.assertAlmostEqual(z[0], 10., places=3)
        self.assertAlmostEqual(z[1], 10. - .025 * 3.4, places=3)
        # a smooth surface: no neighbouring triangles folded
        self.assertLess(np.abs(Vt[:, 2] - 10.).max(), .5)

    def test_urban_corners_are_smaller(self):
        V, recs = crossroads(tags_minor={'maxspeed': '30'}, tags_major={'maxspeed': '50'})
        r = fj.junction(['n0'], recs, V, {})
        self.assertTrue(all(abs(x - fj.KERB['urban']['local']) < 1e-6 for x in r['radii']))

    def test_tee_kerb_tangent_and_no_overlap_beyond_cuts(self):
        import shapely
        V, recs = tee()
        r = fj.junction(['n0'], recs, V, {})
        self.check_mesh(r)
        poly = shapely.union_all([shapely.Polygon(np.asarray(r['verts'])[t, :2]) for t in r['tris']]).buffer(0)
        for m in r['mouths']:
            v = V[m['stroke']]
            beyond = fj._strip(v, m['s'] + m['dir'] * (m['cut'] + .02), m['s'] + m['dir'] * (m['cut'] + 20.))
            self.assertLess(poly.intersection(beyond).area, .01)
        # the kerb leaves each corner of the minor road along its edge (tangent within 3 degrees)
        Vt = np.asarray(r['verts'])
        nxt = {a: b for a, b in r['kerbs']}
        prev = {b: a for a, b in r['kerbs']}
        m = [m for m in r['mouths'] if m['stroke'] == 'B'][0]
        for P in (m['L'], m['R']):
            i = int(np.argmin(np.hypot(Vt[:, 0] - P[0], Vt[:, 1] - P[1])))
            for step in (nxt, prev):
                j, walked = i, 0.
                while j in step and walked < .5:
                    k = step[j]; walked += float(np.hypot(*(Vt[k, :2] - Vt[j, :2]))); j = k
                if j != i:
                    d = Vt[j, :2] - Vt[i, :2]
                    # back towards the node: -y
                    ang = math.degrees(math.acos(max(-1., min(1., -d[1] / np.hypot(*d)))))
                    self.assertLess(ang, 3.)

    def test_islands_from_a_divided_mouth(self):
        # a minor road that splits into two one-way branches before the main road: the triangle between stays
        A = view('A', [(-100, 0), (100, 0)], 'primary', nodes=[('w', 94.), ('e', 106.)])
        In = view('In', [(-6, 0), (-1, 25), (0, 35), (0, 80)], 'tertiary', eL=2., eR=2., nodes=[('w', 0.), ('t', 35.)])
        recs = [node('w', -6, 0, 10, [['A', 94., -1], ['A', 94., 1], ['In', 0., 1]], 'junction'),
                node('e', 6, 0, 10, [['A', 106., -1], ['A', 106., 1], ['Out', 35.67, -1]], 'junction')]
        Out = view('Out', [(0, 80), (0, 35), (1, 25), (6, 0)], 'tertiary', eL=2., eR=2., nodes=[('t', 45.), ('e', 80.67)])
        V = {'A': A, 'In': In, 'Out': Out}
        recs[1]['mouths'][2] = ['Out', float(Out.L), -1]
        got = fj.clusters(recs, V)
        self.assertEqual(got, [['e', 'w']])
        r = fj.junction(got[0], recs, V, {})
        self.assertIsNotNone(r)
        self.check_mesh(r)


class Approaches(unittest.TestCase):
    def test_priority_and_signals(self):
        V, recs = tee()
        r = fj.junction(['n0'], recs, V, {})
        ctl = {(a['stroke'], a['dir']): a['control'] for a in r['approaches']}
        self.assertEqual(ctl[('B', 1)], 'give_way')          # a residential road onto a primary
        self.assertEqual(ctl[('A', -1)], 'none')
        sig = {'controls': [(1, {'highway': 'traffic_signals'}, 0., 0.)]}
        recs[0]['node'] = '1'
        r = fj.junction(['1'], recs, V, fj.inputs(sig))
        self.assertTrue(r['signals'])
        self.assertTrue(all(a['control'] == 'signals' for a in r['approaches']))

    def test_give_way_sign_and_crossing_before_the_stop_line(self):
        V, recs = crossroads(minor='tertiary', major='tertiary')
        # a give_way node on B 20 m south, a zebra 2 m outside the cut on B's south mouth
        cut = None
        r0 = fj.junction(['n0'], recs, V, {})
        for m in r0['mouths']:
            if m['stroke'] == 'B' and m['dir'] == -1:
                cut = m['cut']
        ex = fj.inputs({'controls': [(7, {'highway': 'give_way'}, 0., -20.)],
                        'crossings': [(9, ({'footway': 'crossing', 'crossing': 'zebra'},), np.array([[-6., -cut - 4.], [6., -cut - 4.]]))]})
        r = fj.junction(['n0'], recs, V, ex)
        a = [a for a in r['approaches'] if a['stroke'] == 'B' and a['dir'] == -1][0]
        self.assertEqual(a['control'], 'give_way')
        self.assertEqual([c['kind'] for c in a['crossings']], ['zebra'])
        # the line stands upstream of the zebra (travel +s on B: the stop s is smaller than the zebra's)
        z = a['crossings'][0]
        self.assertLess(a['stop'], z['s'] - z['width'] / 2)

    def test_turn_lanes_of_the_approach(self):
        lanes = [[0., 200., 2, 2, 0, 3.25, 'left|through', 'through|right']]
        V, recs = crossroads(lanes_major=lanes, tags_major={'lanes': '4', 'lanes:forward': '2', 'lanes:backward': '2'})
        r = fj.junction(['n0'], recs, V, {})
        a = {(a['stroke'], a['dir']): a for a in r['approaches']}
        self.assertEqual(a[('A', -1)]['turns'], 'left|through')
        self.assertEqual(a[('A', 1)]['turns'], 'through|right')
        self.assertEqual(a[('A', -1)]['lanes'], 2)

    def test_crossing_kinds(self):
        self.assertEqual(fj.crossing_kind({'crossing': 'zebra'})[0], 'zebra')
        self.assertEqual(fj.crossing_kind({'crossing': 'traffic_signals'})[0], 'signals')
        self.assertEqual(fj.crossing_kind({'highway': 'cycleway', 'cycleway': 'crossing', 'surface:colour': 'red'}), ('cycle', True))
        self.assertIsNone(fj.crossing_kind({'crossing': 'unmarked'})[0])


class Binary(unittest.TestCase):
    def test_round_trip(self):
        V, recs = crossroads()
        r = fj.junction(['n0'], recs, V, {})
        origin = (-1000., -1000.)
        data = fj.encode_cell([r], origin)
        got = fj.decode_cell(data, origin)[0]
        self.assertEqual(got['id'], r['id'])
        self.assertEqual(len(got['verts']), len(r['verts']))
        self.assertEqual(got['tris'], [list(t) for t in r['tris']])
        self.assertEqual(got['kerbs'], [list(k) for k in r['kerbs']])
        self.assertLess(np.abs(np.asarray(got['verts']) - np.asarray(r['verts'])).max(), .00501)
        self.assertEqual([a['control'] for a in got['approaches']], [a['control'] for a in r['approaches']])
        block, mask = fj.encode_block({5: data})
        self.assertEqual(bytes(block[:4]), fj.MAGIC)
        self.assertEqual(list(fj.decode_block(block)), [5])

    def test_build_rows_by_cell(self):
        V, recs = crossroads()
        rows = fj.build(recs, V, {}, 1000., workers=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1:3], (0, 0))


class RoundedMouths(unittest.TestCase):
    def test_mouth_s_rounded_to_a_centimetre_keeps_the_mouth(self):
        V, recs = crossroads()
        # the record's s is rounded: 3 mm off the stroke's own node s (80 / 100)
        for m in recs[0]['mouths']:
            m[1] = m[1] + .003 * (1 if m[2] > 0 else -1) * -1
        r = fj.junction(['n0'], recs, V, {})
        self.assertIsNotNone(r)
        self.assertEqual(len(r['mouths']), 4)


class Degenerate(unittest.TestCase):
    """229 junctions of the Niedersachsen pack had NaN heights: a collapsed strip (zero extent) became an empty polygon
    and the distance to an empty polygon is NaN"""
    def test_collapsed_strip_is_not_empty(self):
        import shapely
        v = view('Z', [(0, 0), (50, 0)], eL=0., eR=0.)
        for lo, hi in ((0., 50.), (20., 20.)):
            strip = fj._strip(v, lo, hi)
            self.assertFalse(strip.is_empty)
            self.assertTrue(np.isfinite(shapely.distance(shapely.points([10., 60.], [5., 5.]), strip)).all())

    def test_zero_extent_minor_gives_finite_heights(self):
        V, recs = crossroads()
        V['B'].eL[:] = 0.
        V['B'].eR[:] = 0.
        r = fj.junction(['n0'], recs, V, {})
        self.assertIsNotNone(r)
        self.assertTrue(np.isfinite(np.asarray(r['verts'])).all())

    def test_height_field_ignores_nan_distance(self):
        self.assertTrue(np.isfinite(fj._finite_d([float('nan'), 2.])).all())

    def test_encode_skips_a_non_finite_junction(self):
        import warnings
        V, recs = crossroads()
        r = fj.junction(['n0'], recs, V, {})
        bad = dict(r, id='jbad', verts=[[0., 0., float('nan')]] + [list(p) for p in r['verts'][1:]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            data = fj.encode_cell([bad, r], (-1000., -1000.))
        self.assertTrue(any('jbad' in str(x.message) for x in w))
        got = fj.decode_cell(data, (-1000., -1000.))
        self.assertEqual([g['id'] for g in got], [r['id']])


if __name__ == '__main__':
    unittest.main()
