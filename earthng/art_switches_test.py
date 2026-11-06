"""The generator's `shops` and `landmarks` switches (generator.lua ROADS.extras; the fork profile turns them off).

Upstream's shop signs, storefronts, glyph letters and landmark walls draw in materials that only upstream's newer
planet ships (brand_sign, storefront, brand_photo, glyphs, hw_ashlar, roof_gold). On by default (upstream's
behaviour); off, a tile asks for none of them.

Run with: python -m unittest discover -s earthng -p art_switches_test.py
"""
import unittest
from pathlib import Path

from lupa.luajit21 import LuaRuntime

ROOT = Path(__file__).resolve().parents[1]
STREET = {'id': 'street', 'nodes': ['a', 'b'], 'width': 8, 'tags': {'highway': 'residential', 'name': 'Hauptstrasse'},
          'points': [[0, 90, 0], [400, 90, 0]]}
SHOP_MATERIALS = {'e2rf_brand_sign', 'e2rf_storefront', 'e2rf_glyphs'}   # the town below draws these; venues and linked shops also brand_photo
LANDMARK_MATERIALS = {'e2rf_hw_ashlar', 'e2rf_roof_gold'}


def square(x, y, side):
    return [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]


def building(ident, ring, height, **tags):
    return {'id': ident, 'height': height, 'tags': {'building': 'yes', **tags}, 'rings': [ring]}


class ArtSwitchTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.g = self.lua.execute((ROOT / 'mod/lua/ge/extensions/e2r/generator.lua').read_text(encoding='utf8'))

    def table(self, value):
        if isinstance(value, dict):
            return self.lua.table_from({k: self.table(v) for k, v in value.items()})
        if isinstance(value, list):
            return self.lua.table_from([self.table(v) for v in value])
        return value

    def town(self):
        """Everything of upstream's art on one tile: a branded shop, a named cafe in a block, a fuel site with an
        open canopy, a numbered house (address plate), a church, a ruin, a gold onion dome."""
        fuel = {'id': 'site', 'kind': 'fuel_site', 'tags': {'e2r_brand': 'okko'}, 'rings': [square(200, 20, 60)]}
        points = [
            {'id': 'p1', 'kind': 'grocery', 'pos': [130, 60, 0], 'tags': {'e2r_brand': 'velmart'}},
            {'id': 'p2', 'kind': 'service', 'pos': [330, 60, 0], 'tags': {'amenity': 'cafe', 'name': 'Cafe Eck'}},
            {'id': 'p3', 'kind': 'fuel', 'pos': [250, 70, 0], 'tags': {'e2r_brand': 'okko'}}]
        buildings = [
            building('shop', square(100, 40, 40), 7, building='retail', e2r_brand='velmart'),
            building('cafe', square(300, 40, 50), 7, building='retail'),
            building('canopy', [[210, 30], [240, 30], [240, 44], [210, 44]], 5, building='roof'),
            building('house', square(20, 40, 12), 7, **{'addr:housenumber': '12', 'addr:street': 'Hauptstrasse'}),
            building('church', square(100, 200, 14), 40, building='church'),
            building('ruin', square(200, 200, 14), 9, building='ruins'),
            building('dome', square(300, 200, 14), 12, **{'roof:shape': 'onion', 'roof:colour': 'gold'})]
        return [{'roads': [STREET], 'areas': [fuel], 'buildings': buildings, 'points': points}]

    def build(self, options=None, cells=None):
        opts = {'terrainStep': 100, 'maxTrees': 0, 'experimentalDetail': True}
        opts.update(options or {})
        scene = self.g.build(self.table(cells or self.town()), 0, 0, self.table(opts))[1]
        drawn = {mesh.material for item in scene['items'].values() for mesh in (item.meshes or self.lua.table()).values()}
        listed = {scene['mats'][k] for k in range(1, len(scene['mats']) + 1)}
        self.assertEqual(drawn, listed)                       # the materials the tile writes are the ones it draws
        return listed

    def test_the_defaults_are_upstreams_art(self):
        self.assertTrue(self.g.EXTRAS.shops and self.g.EXTRAS.landmarks)
        drawn = self.build()
        self.assertEqual(SHOP_MATERIALS - drawn, set())
        self.assertEqual(LANDMARK_MATERIALS - drawn, set())
        self.assertEqual(self.build({'shops': True, 'landmarks': True}), drawn)     # on by name is the default

    def test_shops_off_asks_for_no_shop_material(self):
        drawn = self.build({'shops': False})
        self.assertEqual(drawn & (SHOP_MATERIALS | {'e2rf_brand_photo'}), set())
        self.assertEqual(LANDMARK_MATERIALS - drawn, set())                  # the other switch is untouched

    def test_landmarks_off_asks_for_no_landmark_material_and_walls_fall_back(self):
        drawn = self.build({'landmarks': False})
        self.assertEqual(drawn & LANDMARK_MATERIALS, set())
        self.assertEqual(SHOP_MATERIALS - drawn, set())
        self.assertTrue([m for m in drawn if m.startswith('e2rf_wall_') or m.startswith('e2rf_pls_') or m.startswith('e2rf_brk_')])

    def test_both_off_in_the_generators_extras_as_a_pack_sets_them(self):
        self.g.EXTRAS.shops, self.g.EXTRAS.landmarks = False, False
        self.g.applyExtras()
        drawn = self.build()
        self.assertEqual(drawn & (SHOP_MATERIALS | LANDMARK_MATERIALS), set())
        self.assertFalse([m for m in drawn if 'brand' in m])
        self.assertEqual(self.build({'shops': True, 'landmarks': True}) & (SHOP_MATERIALS | LANDMARK_MATERIALS),
                         SHOP_MATERIALS | LANDMARK_MATERIALS)             # a call's own option still wins

    def test_a_canopy_is_a_plain_building_when_shops_are_off(self):
        roof = [{'roads': [STREET], 'buildings': [building('roof', [[100, 100], [120, 100], [120, 108], [100, 108]], 4, building='roof')]}]
        self.assertIn('e2rf_brand_sign', self.build(cells=roof))
        off = self.build({'shops': False}, cells=roof)
        self.assertNotIn('e2rf_brand_sign', off)
        self.assertTrue([m for m in off if m.startswith('e2rf_wall_')])


if __name__ == '__main__':
    unittest.main()
