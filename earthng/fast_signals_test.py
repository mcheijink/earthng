"""Tests of fast_signals and its hooks in fast_sources' scenery reader. Run: python earthng/fast_signals_test.py

The OSM file is synthetic, round a made-up point in open sea (0.5 E, 0.5 N)."""
import collections
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_signals as fs  # noqa: E402
import fast_style  # noqa: E402
import looks  # noqa: E402

LON, LAT = .5, .5
D = .0005   # about 55 m


def osm(nodes, ways, relations=()):
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
    for nid, lon, lat, tags in nodes:
        out.append(f'<node id="{nid}" version="1" lat="{lat:.7f}" lon="{lon:.7f}">')
        out += [f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()]
        out.append('</node>')
    for wid, refs, tags in ways:
        out.append(f'<way id="{wid}" version="1">')
        out += [f'<nd ref="{r}"/>' for r in refs]
        out += [f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()]
        out.append('</way>')
    for rid, members, tags in relations:
        out.append(f'<relation id="{rid}" version="1">')
        out += [f'<member type="node" ref="{r}" role="{role}"/>' for r, role in members]
        out += [f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()]
        out.append('</relation>')
    out.append('</osm>')
    return '\n'.join(out)


def scenery(text):
    """The point records fast_sources' reader makes of an OSM file: {node id: (kind, tags)}."""
    import fast_sources
    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, 'tile.osm'), 'w', encoding='utf8') as f:
        f.write(text)
    frame = fast_sources.Frame.at(LON, LAT)
    store = type('Store', (), {})()
    store.frame, store.meta, store.scenes = frame, {'source': folder}, collections.OrderedDict()
    store.source_index = {'cell': 1., 'buffer': .1, 'tiles': {'0_0': {'coverage': [-1., -1., 2., 2.], 'file': 'tile.osm'}}}
    x, y = frame.local(LON, LAT)
    records = fast_sources.Store._scenery(store, [float(x) - 400, float(y) - 400, float(x) + 400, float(y) + 400])
    return {-fid - fs.POINT: (kind, data['tags']) for fid, kind, data, _geom in records if fid <= -fs.POINT}


class Kinds(unittest.TestCase):
    def test_point_kinds(self):
        self.assertEqual(fast_style.point_kind({'highway': 'traffic_signals'}), 'signal')
        self.assertEqual(fast_style.point_kind({'highway': 'speed_camera'}), 'speed_camera')
        self.assertIsNone(fs.point_kind({'highway': 'crossing'}))

    def test_kept_with_the_road(self):
        self.assertTrue(looks.keep_scenery('speed_camera', {'cover': False, 'stops': False}))
        self.assertTrue(looks.keep_scenery('signal', {'cover': False}))

    def test_heading_of_tags(self):
        self.assertEqual(fs.heading({'direction': 'NE'}), 45.)
        self.assertEqual(fs.heading({'direction': '-90'}), 270.)
        self.assertEqual(fs.heading({fs.HEADING: '12.5', 'direction': 'S'}), 12.5)
        self.assertIsNone(fs.heading({'direction': 'forward'}))
        self.assertIsNone(fs.heading({'direction': 'nan'}))

    def test_relative(self):
        self.assertEqual(fs.relative({'traffic_signals:direction': 'backward'}, fs.SIGNAL), 'backward')
        self.assertEqual(fs.relative({'direction': 'forward'}, fs.CAMERA), 'forward')
        self.assertIsNone(fs.relative({'traffic_signals:direction': 'forward'}, fs.CAMERA))


class Reader(unittest.TestCase):
    def test_way_direction_becomes_a_heading(self):
        # an east-running street through two signals (forward, backward) and one without a direction
        nodes = [(1, LON - D, LAT, {}), (2, LON, LAT, {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}),
                 (3, LON + D, LAT, {'highway': 'traffic_signals', 'traffic_signals:direction': 'backward'}),
                 (4, LON + 2 * D, LAT, {'highway': 'traffic_signals'}), (5, LON + 3 * D, LAT, {})]
        points = scenery(osm(nodes, [(10, [1, 2, 3, 4, 5], {'highway': 'residential'})]))
        self.assertEqual(points[2][0], 'signal')
        self.assertAlmostEqual(float(points[2][1][fs.HEADING]), 90., delta=.5)
        self.assertAlmostEqual(float(points[3][1][fs.HEADING]), 270., delta=.5)
        self.assertNotIn(fs.HEADING, points[4][1])

    def test_a_footway_does_not_set_a_road_signal(self):
        nodes = [(1, LON, LAT - D, {}), (2, LON, LAT, {'highway': 'traffic_signals', 'direction': 'forward'}), (3, LON, LAT + D, {})]
        points = scenery(osm(nodes, [(10, [1, 2, 3], {'highway': 'footway'})]))
        self.assertNotIn(fs.HEADING, points[2][1])

    def test_enforcement_relation(self):
        # a north-running road: the device of the first check is an untagged node beside it, the second a camera node
        nodes = [(1, LON, LAT - D, {}), (2, LON, LAT, {}), (3, LON, LAT + D, {}), (4, LON, LAT + 2 * D, {}),
                 (7, LON + D / 10, LAT - D / 2, {}),
                 (8, LON - D / 10, LAT + 1.5 * D, {'highway': 'speed_camera'})]
        relations = [(20, [(7, 'device'), (1, 'from'), (2, 'to')], {'type': 'enforcement', 'enforcement': 'maxspeed', 'maxspeed': '50'}),
                     (21, [(8, 'device'), (4, 'from'), (3, 'to')], {'type': 'enforcement', 'enforcement': 'maxspeed'})]
        points = scenery(osm(nodes, [(10, [1, 2, 3, 4], {'highway': 'primary'})], relations))
        self.assertEqual(points[7][0], 'speed_camera')
        self.assertEqual(points[7][1]['maxspeed'], '50')
        self.assertAlmostEqual(float(points[7][1][fs.HEADING]), 0., delta=.5)
        self.assertAlmostEqual(float(points[8][1][fs.HEADING]), 180., delta=.5)

    def test_a_camera_direction_stays_as_mapped(self):
        nodes = [(1, LON, LAT, {'highway': 'speed_camera', 'direction': 'W'})]
        points = scenery(osm(nodes, []))
        self.assertNotIn(fs.HEADING, points[1][1])
        self.assertEqual(fs.heading(points[1][1]), 270.)


if __name__ == '__main__':
    unittest.main()
