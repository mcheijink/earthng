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
OFF_SWITCHES = ('shops', 'landmarks', 'railsOffDecks')       # upstream's behaviour the fork switches off (art its planet does not ship, rails kept off decks): default on, off in the profile


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
            self.assertEqual(sorted(got), sorted(SWITCHES + OFF_SWITCHES))
            self.assertTrue(all(got[k] is True for k in SWITCHES))
            self.assertTrue(all(got[k] is False for k in OFF_SWITCHES))
            got['signals'] = False                                   # a copy: the profile itself is unchanged
            self.assertTrue(region_profile.extras()['signals'])
            region_profile.apply_env()
            self.assertEqual(os.environ['EARTHNG_NOISE'], '0')        # a variable set by hand wins
            self.assertEqual([os.environ[k] for k in ('EARTHNG_ROAD_RULES', 'EARTHNG_POWER', 'EARTHNG_CAMERAS')], ['1'] * 3)
            self.assertEqual([os.environ[k] for k in ('EARTHNG_BRIDGE_FLOORS', 'EARTHNG_PCG')], ['1'] * 2)
            self.assertEqual([os.environ[k] for k in ('EARTHNG_SPANS', 'EARTHNG_TIDAL')], ['0'] * 2)

    def test_switch_reads_the_environment_with_a_default(self):
        with without_profile(), patch.dict(os.environ, {'EARTHNG_X': '0', 'EARTHNG_Y': 'yes', 'EARTHNG_Z': ''}):
            self.assertFalse(region_profile.switch('EARTHNG_X', True))
            self.assertTrue(region_profile.switch('EARTHNG_Y', False))
            self.assertTrue(region_profile.switch('EARTHNG_Z', True))
            self.assertFalse(region_profile.switch('EARTHNG_UNSET', False))

    def test_exporter_modules_default_to_upstream_and_the_fork_profile_flips_four(self):
        import subprocess
        import sys
        code = ('import fast_profile as p, fast_roads as r, build_country as c, fast_style as s;'
                'print(int(p.PCG), int(p.SPANS), int(r.FLOORS), int(c.TIDAL), int(s.CAMERAS))')
        here = os.path.dirname(os.path.abspath(__file__))
        base = {k: v for k, v in os.environ.items() if not k.startswith('EARTHNG_')}
        def run(extra):
            out = subprocess.run([sys.executable, '-c', code], cwd=here, env=dict(base, **extra), capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout.split()[-5:]
        self.assertEqual(run({}), ['0', '1', '0', '1', '0'])
        self.assertEqual(run({'EARTHNG_PROFILE': 'fork'}), ['1', '0', '1', '0', '1'])
        self.assertEqual(run({'EARTHNG_PROFILE': 'fork', 'EARTHNG_SPANS': '1'}), ['1', '1', '1', '0', '1'])


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
        self.assertEqual(region['extras'], dict(dict.fromkeys(SWITCHES, True), **dict.fromkeys(OFF_SWITCHES, False)))

    def test_a_pack_without_a_profile_has_no_extras(self):
        with without_profile():
            region = self.manifest('plain')
        self.assertNotIn('extras', region)


if __name__ == '__main__':
    unittest.main()
