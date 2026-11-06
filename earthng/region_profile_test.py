"""The exporter's region profile (region_profile.py): EARTHNG_PROFILE=fork puts the Lua generator's switches
into every region manifest as `extras`; without a profile the manifest is the upstream exporter's.

Run with: python -m unittest discover -s earthng -p region_profile_test.py
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import build_country as country
import fast_sources as sources
import region_profile
import test_country as fixture

SWITCHES = ('signals', 'lots', 'roadProfiles', 'lampRows', 'openSea')


def without_profile():
    env = {k: v for k, v in os.environ.items() if k != 'EARTHNG_PROFILE'}
    return patch.dict(os.environ, env, clear=True)


class ProfileTests(unittest.TestCase):
    def test_no_profile_sets_nothing(self):
        with without_profile():
            self.assertIsNone(region_profile.extras())
            before = dict(os.environ)
            region_profile.apply_env()
            self.assertEqual(dict(os.environ), before)

    def test_unknown_profile_sets_nothing(self):
        with without_profile(), patch.dict(os.environ, {'EARTHNG_PROFILE': 'other'}):
            self.assertIsNone(region_profile.extras())

    def test_fork_profile_names_the_five_switches_and_defaults_the_exporter_switches(self):
        with without_profile(), patch.dict(os.environ, {'EARTHNG_PROFILE': ' Fork ', 'EARTHNG_NOISE': '0'}):
            got = region_profile.extras()
            self.assertEqual(sorted(got), sorted(SWITCHES))
            self.assertTrue(all(v is True for v in got.values()))
            got['signals'] = False                                   # a copy: the profile itself is unchanged
            self.assertTrue(region_profile.extras()['signals'])
            region_profile.apply_env()
            self.assertEqual(os.environ['EARTHNG_NOISE'], '0')        # a variable set by hand wins
            self.assertEqual([os.environ[k] for k in ('EARTHNG_ROAD_RULES', 'EARTHNG_POWER', 'EARTHNG_CAMERAS')], ['1'] * 3)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        venues = patch.object(country.venues, 'catalogue', return_value=[])
        venues.start()
        self.addCleanup(venues.stop)
        self.root = Path(self.temp.name)
        self.region = self.root / 'source'
        self.region.mkdir()
        self.store = fixture.Store()
        meta = {'id': 'test_region', 'generator': sources.VERSION, 'bbox': [30, 50, 31, 51],
                'frame': self.store.frame.spec(), 'dem': '/private/dem', 'source': '/private/osm'}
        (self.region / 'region.json').write_text(json.dumps(meta), encoding='utf8')
        (self.region / 'sources.sqlite').write_bytes(b'author-only database')

    def manifest(self, name):
        output = self.root / name
        with patch.object(sources, 'Store', return_value=self.store):
            path = country.build_country('ua', 'Country', '0.1.0', [self.region], output,
                                         (-1000, 0, 2000, 1000), emit=lambda _: None)
        with zipfile.ZipFile(path) as archive:
            index = json.loads(archive.read('earthng/countries/ua/manifest.json'))
            return json.loads(archive.read(index['regions'][0]['manifest'].lstrip('/')))

    def test_fork_profile_writes_extras_into_the_region_manifest(self):
        with without_profile(), patch.dict(os.environ, {'EARTHNG_PROFILE': 'fork'}):
            region = self.manifest('fork')
        self.assertEqual(region['extras'], dict.fromkeys(SWITCHES, True))

    def test_a_pack_without_a_profile_has_no_extras(self):
        with without_profile():
            region = self.manifest('plain')
        self.assertNotIn('extras', region)


if __name__ == '__main__':
    unittest.main()
