"""Tests of the national DTM source (dtm.py) and its use in fast_profile.solve and the export grid.
Run: python earthng/dtm_test.py"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtm  # noqa: E402
import fast_profile as fp  # noqa: E402
import fast_sources  # noqa: E402
from fast_profile_test import at, heights_along_x, line, seg  # noqa: E402

PATTERN = r'^t_(?P<e>\d+)_(?P<n>\d+)_(?P<year>\d{4})\.tif$'


def write_tiles(folder, keys, height, year=2021, size=100, res=1.):
    """GeoTIFF tiles of size x size m (EPSG:25832), height(e, n) at pixel centres"""
    import rasterio
    from rasterio.transform import from_origin
    folder.mkdir(parents=True, exist_ok=True)
    n_px = int(size / res)
    for i, j in keys:
        e0, n1 = i * size, (j + 1) * size
        ce = e0 + (np.arange(n_px) + .5) * res
        cn = n1 - (np.arange(n_px) + .5) * res
        E, N = np.meshgrid(ce, cn)
        a = np.asarray(height(E, N), np.float32)
        # the pattern reads km; these test tiles are `size` m, so the names carry the index itself
        with rasterio.open(folder / f't_{i}_{j}_{year}.tif', 'w', driver='GTiff', width=n_px, height=n_px, count=1,
                           dtype='float32', crs='EPSG:25832', transform=from_origin(e0, n1, res, res), nodata=-9999.) as ds:
            ds.write(a, 1)


class Frame:
    """a frame in EPSG:25832 itself (no projection), origin at 0"""
    crs, cx, cy = 'EPSG:25832', 0., 0.


def tileset(folder, size=100.):
    # the pattern's e/n are in km in real providers: scale the test names to `size`
    ts = dtm.TileSet(folder, 'EPSG:25832', size, '', 'test')
    return ts


class Tiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'dtm'
        # 3 x 2 tiles of 100 m, a plane z = 5 + 0.01 e
        write_tiles(self.root, [(i, j) for i in range(3) for j in range(2)], lambda e, n: 5 + .01 * e)
        self.ts = dtm.TileSet(self.root, 'EPSG:25832', 100.)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_and_point(self):
        self.assertEqual(len(self.ts), 6)
        z = self.ts.at([10.5, 250.5, 400.], [10., 150., 50.])
        self.assertAlmostEqual(float(z[0]), 5.105, places=4)
        self.assertAlmostEqual(float(z[1]), 7.505, places=4)
        self.assertTrue(np.isnan(z[2]))                 # no tile there

    def test_coverage_share_and_fade(self):
        s, f = self.ts.coverage([150., 150., 150., 350.], [100., 195., 250., 100.], blend=100., fade=100.)
        self.assertAlmostEqual(s[0], 1.)                # 100 m from any missing tile (the box edge is at 0 / 200)
        self.assertLess(s[1], .01)                      # 5 m inside the north edge
        self.assertEqual((s[2], s[3]), (0., 0.))
        self.assertAlmostEqual(f[0], 1.)
        self.assertAlmostEqual(f[2], dtm._smoothstep(.5), places=6)   # 50 m north of the coverage
        self.assertAlmostEqual(f[3], dtm._smoothstep(.5), places=6)   # 50 m east of it

    def test_newest_year_wins(self):
        write_tiles(self.root, [(0, 0)], lambda e, n: 0. * e + 99., year=2015)
        ts = dtm.TileSet(self.root, 'EPSG:25832', 100., r'^t_(?P<e>\d+)_(?P<n>\d+)_(?P<year>\d{4})\.tif$')
        # the pattern reads e/n in km: these names are tile indices, so only the year rule is tested here
        self.assertEqual(sorted({y for _, y in ts.files.values()}), [2021])

    def test_road_footprint_median_and_lattice_mean(self):
        d = dtm.Dtm([(dtm.Provider('test', 'p', 'a', 'EPSG:25832', 'l', 'c', 'u', tile=100.), self.ts)], Frame())
        xy = np.array([[150., 100.], [150., 30.]])
        h, share = d.road(xy, np.array([[1., 0.], [1., 0.]]), np.array([5., 5.]))
        self.assertAlmostEqual(float(h[0] - dtm.LIFT), 5 + .01 * 150.5, places=3)
        self.assertAlmostEqual(float(share[0]), .5, places=6)   # 100 m inside the edge of BLEND 200 m
        xs = np.arange(100., 201., 10.); ys = np.arange(50., 151., 10.)
        g, gs = d.lattice(xs, ys)
        gx, _ = np.meshgrid(xs, ys)
        inner = gs.reshape(len(ys), len(xs)) > 0
        self.assertLess(np.abs((g.reshape(len(ys), len(xs)) - dtm.LIFT - (5 + .01 * gx)))[inner].max(), .02)

    def test_meta_round_trip_and_credits(self):
        p = dtm.Provider('de-ni', 'DGM1', 'Niedersachsen', 'EPSG:25832', 'CC BY 4.0', '(c) GeoBasis-DE/LGLN {year}', 'u',
                         tile=100.)
        d = dtm.Dtm([(p, self.ts)], Frame(), 2.5)
        meta = d.meta()
        self.assertEqual(meta['providers'][0]['tiles'], 6)
        self.assertEqual(meta['offset'], 2.5)
        c = d.credits()[0]
        self.assertIn('GeoBasis-DE/LGLN 2021', c['source'])
        self.assertEqual(c['license'], 'CC BY 4.0')

    def test_blend_is_continuous_and_carries_the_offset(self):
        p = dtm.Provider('test', 'p', 'a', 'EPSG:25832', 'l', 'c', 'u', tile=100.)
        d = dtm.Dtm([(p, self.ts)], Frame(), 3.)
        x = np.arange(-150., 451., 1.)
        xy = np.column_stack([x, np.full(len(x), 100.)])
        fine = d.point(xy)
        coarse = np.where(np.isfinite(fine), fine, 5 + .01 * x + dtm.LIFT) - 3.   # the coarse DEM 3 m under the DTM
        out = d.blend(coarse, xy)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(np.abs(np.diff(out)).max(), .05)   # no step at the edge or where the offset fades


class FakeReference:
    """a DTM given by a function and a coverage box (share 1 inside, 0 outside, a ramp of `edge` m)"""
    def __init__(self, height, box, edge=0.):
        self.height, self.box, self.edge = height, box, edge

    def road(self, xy, tangent, half):
        xy = np.asarray(xy, float).reshape(-1, 2)
        x0, y0, x1, y1 = self.box
        d = np.minimum.reduce([xy[:, 0] - x0, x1 - xy[:, 0], xy[:, 1] - y0, y1 - xy[:, 1]])
        share = np.clip(d / self.edge, 0, 1) if self.edge else (d >= 0).astype(float)
        return np.where(share > 0, self.height(xy), np.nan), share


class Solve(unittest.TestCase):
    def test_measured_stations_follow_the_dtm_and_the_rest_the_dem(self):
        segs = line((0, 0), (2000, 0), (4000, 0), (6000, 0))
        # coarse DEM: flat 10 m with a 6 m forest-like hump at x 3000; the DTM (x 2000-4000): a real 2 m dip
        dem = lambda xy: 10. + 6. * np.exp(-((np.asarray(xy).reshape(-1, 2)[:, 0] - 3000) / 150.) ** 2)
        truth = lambda xy: 10. - 2. * np.exp(-((np.asarray(xy).reshape(-1, 2)[:, 0] - 3000) / 200.) ** 2)
        ref = FakeReference(truth, (2000, -1000, 4000, 1000), edge=200.)
        prof, _ = fp.solve(segs, dem, reference=ref)
        xs = np.arange(2600, 3401, 50.)
        h = heights_along_x(prof, segs, xs)
        self.assertLess(np.abs(h - truth(np.column_stack([xs, 0 * xs]))).max(), .15)
        far = heights_along_x(prof, segs, np.array([500., 5500.]))
        self.assertLess(np.abs(far - 10.).max(), .2)
        # the coverage edge: no step, the class grade holds
        for (ps, ph), sg in zip(prof, segs):
            self.assertLess(np.abs(np.diff(ph) / np.diff(ps)).max(), fp.GRADE_MAX['motorway'] + .002)

    def test_reference_turns_the_rules_on_and_without_it_nothing_changes(self):
        segs = line((0, 0), (1000, 0), (2000, 0))
        dem = lambda xy: 10. + 3. * np.exp(-((np.asarray(xy).reshape(-1, 2)[:, 0] - 1000) / 80.) ** 2)
        a, _ = fp.solve(segs, dem, rules=False)
        b, _ = fp.solve(segs, dem, rules=False, reference=None)
        for (s1, h1), (s2, h2) in zip(a, b):
            self.assertTrue(np.array_equal(h1, h2))

    def test_deck_ignores_the_dtm_valley_and_keeps_its_rules(self):
        # a motorway bridge (x 900-1100) over a valley: the DTM has the river bed (-8 m) under the deck
        mw = [seg((1, 2), [(0, 0), (900, 0)]), seg((2, 3), [(900, 0), (1100, 0)], bridge='yes', layer='1'),
              seg((3, 4), [(1100, 0), (2000, 0)])]
        dem = lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), 10.)
        def truth(xy):
            x = np.asarray(xy).reshape(-1, 2)[:, 0]
            return np.where(np.abs(x - 1000) < 80, 2., 10.)
        ref = FakeReference(truth, (-100, -100, 2100, 100))
        prof, _ = fp.solve(mw, dem, reference=ref)
        deck = prof[1][1]
        self.assertGreater(deck.min(), 9.5)              # spans its abutments, not the valley
        self.assertLess(np.abs(heights_along_x(prof, mw, np.array([300., 1700.])) - 10.).max(), .1)

    def test_road_under_a_deck_bridges_the_dtm_bulge_and_holds(self):
        # a secondary road along x under a street bridge at x = 2000; the DTM bulges 3 m under the deck
        road = line((0, 0), (1000, 0), (3000, 0), (4000, 0), highway='secondary')
        ramp1 = seg((20, 21), [(2000, -400), (2000, -30)], 'tertiary')
        deck = seg((21, 22), [(2000, -30), (2000, 30)], 'tertiary', bridge='yes', layer='1')
        ramp2 = seg((22, 23), [(2000, 30), (2000, 400)], 'tertiary')
        segs = road + [ramp1, deck, ramp2]
        dem = lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), 10.)
        def truth(xy):
            # the crossing street's embankments (kept in a DTM) up to its deck, 7 m over the road
            xy = np.asarray(xy).reshape(-1, 2)
            bulge = 3. * np.exp(-((xy[:, 0] - 2000) ** 2 + xy[:, 1] ** 2) / (2 * 8. ** 2))
            bank = np.where(np.abs(xy[:, 0] - 2000) < 12, 7. * np.clip(1 - (np.abs(xy[:, 1]) - 30) / 120, 0, 1), 0.)
            return 10. + np.where(np.abs(xy[:, 1]) < 30, bulge, bank)
        ref = FakeReference(truth, (-100, -500, 4100, 500))
        prof, _ = fp.solve(segs, dem, reference=ref)
        xs = np.arange(1900, 2101, 10.)
        h = heights_along_x(prof, segs, xs)
        self.assertLess(np.abs(h - 10.).max(), .3)       # the bulge is gone, the road is not dug in
        top = at(prof, segs, 4, 30.)                     # the deck's middle
        self.assertGreater(top - h[np.argmin(np.abs(xs - 2000))], fp.CLEARANCE - .2)   # the deck clears it


class GroundGrid(unittest.TestCase):
    def test_cell_grid_is_the_dtm_inside_and_meets_the_coarse_dem_without_a_step(self):
        import build_country
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dtm'
            # 1 km tiles over x 0-2000 (a plane); the coarse DEM 2 m under it with a 3 m forest bump at x 2500
            write_tiles(root, [(0, 0), (1, 0), (0, 1), (1, 1)], lambda e, n: 5 + .002 * e, size=1000., res=1.)
            ts = dtm.TileSet(root, 'EPSG:25832', 1000.)
            p = dtm.Provider('test', 'p', 'a', 'EPSG:25832', 'l', 'c', 'u')
            d = dtm.Dtm([(p, ts)], Frame(), 2.)

            class Coarse:
                def sample(self, xy):
                    xy = np.asarray(xy, float).reshape(-1, 2)
                    return 3 + .002 * xy[:, 0] + 3. * np.exp(-((xy[:, 0] - 2500) / 150.) ** 2) + dtm.LIFT
            ground = dtm.Ground(Coarse(), d)
            xs = np.arange(1000., 3001., 10.); ys = np.arange(500., 1501., 10.)
            h = ground.cell_grid(xs, ys).reshape(len(ys), len(xs))
            inside = xs <= 1800.
            plane = 5 + .002 * xs + dtm.LIFT
            self.assertLess(np.abs(h[:, inside] - plane[inside]).max(), .01)
            self.assertLess(np.abs(np.diff(h, axis=1)).max(), .2)           # 10 m apart: no step at x = 2000
            store = type('S', (), {'terrain': ground})()
            self.assertEqual(build_country.grid_step(store), build_country.GRID_FINE)


class OneGround(unittest.TestCase):
    """Roads and ground from one corrected DEM (ALTITUDE-REFERENCE.md "Road against ground")."""

    def test_coarse_part_takes_the_forest_bias_by_the_same_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dtm'
            write_tiles(root, [(0, 0), (1, 0), (0, 1), (1, 1)], lambda e, n: 5 + .002 * e, size=1000., res=1.)
            ts = dtm.TileSet(root, 'EPSG:25832', 1000.)
            p = dtm.Provider('test', 'p', 'a', 'EPSG:25832', 'l', 'c', 'u')
            d = dtm.Dtm([(p, ts)], Frame(), 2.)

            class Coarse:
                def sample(self, xy):
                    xy = np.asarray(xy, float).reshape(-1, 2)
                    return 3 + .002 * xy[:, 0] + dtm.LIFT
            lower = lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), 4.)       # a 4 m canopy bias everywhere
            plain, corrected = dtm.Ground(Coarse(), d), dtm.Ground(Coarse(), d, lower)
            xs = np.arange(1000., 3501., 10.); ys = np.arange(500., 1501., 10.)
            a = plain.cell_grid(xs, ys).reshape(len(ys), len(xs))
            b = corrected.cell_grid(xs, ys).reshape(len(ys), len(xs))
            share, _ = d.coverage(np.column_stack([xs, np.full(len(xs), 1000.)]))
            # inside the coverage nothing changes, outside the full bias, in the blend zone (1 - share) of it
            self.assertTrue(np.allclose(a[50] - b[50], 4. * (1. - share), atol=1e-6))
            self.assertTrue(np.allclose(plain.sample([[3400., 1000.]]) - corrected.sample([[3400., 1000.]]), 4.))
            self.assertTrue(np.allclose(plain.sample([[500., 1000.]]), corrected.sample([[500., 1000.]])))

    def test_export_reads_the_index_forest_share_and_marks_the_pipeline(self):
        import build_country
        import shapely
        with tempfile.TemporaryDirectory() as tmp:
            cover = fp.Cover([shapely.box(0, -5000, 5000, 5000)], [])
            cover.grid = cover.share_grid((-2000., -2000., 2000., 2000.))
            cover.grid.save(Path(tmp) / 'forest.npz')

            class Flat:
                def sample(self, xy):
                    return np.full(len(np.asarray(xy).reshape(-1, 2)), 30.)
            store = type('S', (), {})()
            store.directory, store.terrain = Path(tmp), Flat()
            store.meta = {'ground': {'version': 1, 'forest': 'forest.npz', 'bias': fp.FOREST_BIAS}}
            build_country._despike(store)
            self.assertIsInstance(store.terrain, build_country.Lowered)
            pts = np.array([[-1500., 0.], [0., 0.], [1500., 0.]])
            want = 30. - fp.FOREST_BIAS * cover.forest_share(pts)
            self.assertTrue(np.allclose(store.terrain.sample(pts), want, atol=1e-3))
            xs = np.arange(-50., 51., 25.)
            g = store.terrain.grid(xs, xs)
            gx, gy = np.meshgrid(xs, xs)
            self.assertTrue(np.allclose(g, 30. - fp.FOREST_BIAS * cover.forest_share(np.column_stack([gx.ravel(), gy.ravel()])), atol=1e-3))
            self.assertEqual(build_country._height_pipeline(store), {'format': build_country.HEIGHTS_FORMAT, 'dtm': False, 'rules': True})
            old = type('S', (), {'meta': {}, 'terrain': Flat()})()
            build_country._despike(old)
            self.assertIs(type(old.terrain), build_country.Despiked)          # an older index: as before
            self.assertIsNone(build_country._height_pipeline(old))


class Pull(unittest.TestCase):
    def test_pull_matches_the_generator_rule(self):
        xs = np.arange(0., 201., 10.); ys = np.arange(0., 201., 10.)
        base = np.full(len(xs) * len(ys), 10.)
        roads = [{'points': [[0., 100., 14.], [200., 100., 14.]], 'width': 8., 'tags': {'highway': 'primary'}},
                 {'points': [[100., 0., 30.], [100., 200., 30.]], 'width': 8., 'tags': {'bridge': 'yes'}}]
        out = dtm.pull_to_roads(base, xs, ys, roads, np.ones(len(base)), 12., 70.).reshape(len(ys), len(xs))
        self.assertAlmostEqual(out[10, 3], 14.)          # on the road
        self.assertAlmostEqual(out[11, 3], 14.)          # 10 m from the centre, 6 m from the edge: full pull
        s = (70. - (30. - 4.)) / 58.
        self.assertAlmostEqual(out[13, 3], 10. + 4. * s * s * (3 - 2 * s))
        self.assertAlmostEqual(out[0, 3], 10.)           # 100 m away
        half = dtm.pull_to_roads(base, xs, ys, roads, np.full(len(base), .5), 12., 70.).reshape(len(ys), len(xs))
        self.assertAlmostEqual(half[10, 3], 12.)         # weight 0.5: half way
        self.assertAlmostEqual(out[0, 10], 10.)          # bridges do not pull


class Fetch(unittest.TestCase):
    def test_plan_is_offline_and_counts_missing_tiles(self):
        import dtm_fetch
        p = dtm.BY_ID['de-ni']
        tiles = dtm_fetch.tiles_in(p, (10.0, 51.6, 10.03, 51.62), 0.)
        self.assertTrue(4 <= len(tiles) <= 12)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(dtm_fetch.missing(p, tiles, data=d)), len(tiles))
        with self.assertRaises(SystemExit):
            dtm_fetch.main(['de-ni', '--box', '10.0', '51.6', '10.03', '51.62', '--download'])   # needs --yes


class Switch(unittest.TestCase):
    def test_off_by_default_and_not_in_the_identity(self):
        if 'EARTHNG_DTM' not in os.environ:
            self.assertFalse(dtm.ENABLED)
        self.assertIn('de-ni', dtm.BY_ID)
        self.assertTrue(dtm.BY_ID['de-ni'].verified)
        self.assertFalse(any(p.verified for p in dtm.PROVIDERS if p.id != 'de-ni'))


if __name__ == '__main__':
    unittest.main()
