"""National 1 m terrain models (DTM) as the regular height source where they exist (EARTHNG_DTM=1).

The design decision (ALTITUDE-REFERENCE.md, "National DTM"): where a free 1 m bare-earth model covers
the ground, road profiles and the ground grid come from it; elsewhere the coarse DEM (Terrarium) with the
DEM rules (forest correction, structure masking) stays the fallback. It is one pipeline with different
inputs and weights, not a second solver:

* fast_profile.solve takes a `reference` (a Dtm). Each station's data term is a blend: the DTM, sampled
  over the carriageway's footprint, with a high weight where it covers (`share` 1), the coarse DEM's
  rule ground line elsewhere (`share` 0), smoothly in between over BLEND m inside the coverage edge. The
  road-design rules (class grade and vertical-curve limits, bridge and deck clearance, navigable-water
  floors, carriageway coupling, junction flattening) act on every station as before.
* The DTM has no bridges (LGLN removes them): decks and tunnels never take it as a target, a road under
  a deck takes the DTM line bridged from its clear stations on either side, and the last metres before
  a road's own bridge are shaped by the deck's rule chain (clearance, floors, grade) and the DTM ramp.
* The coarse DEM is shifted by one datum offset per region (median DTM - coarse on open land inside the
  coverage: +1.7 m in Niedersachsen, where Terrarium sits under DHHN2016; at road centres the difference is
  about 3 m, because the coarse DEM also misses the embankments), faded out over FADE m outside the
  coverage, so coarse and fine heights meet without a step.
* The export's ground grid (build_country) is the same blend on a 10 m lattice: the DTM's block mean where
  it covers, the despiked coarse DEM (pulled to the roads as the generator does on a 25 m grid) elsewhere.

Providers are tile folders under data/ (GeoTIFF, any projected CRS, square tiles). Only LGLN DGM1 for
Niedersachsen is implemented and verified; PROVIDERS lists other free national models with their
licences, all marked unverified, none downloaded. A pack built with a DTM carries the provider's credit
(credits()). No position is printed or logged here.
"""
import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ENABLED = os.environ.get('EARTHNG_DTM', '0').strip().lower() not in ('', '0', 'off', 'no', 'false')
VERSION = 'dtm-1'
SIGNATURE = f'{VERSION}:{"on" if ENABLED else "off"}'
DATA = Path(__file__).resolve().parent.parent / 'data'
BLEND = 200.        # m inside the coverage edge over which the DTM's share rises from 0 to 1
FADE = 1000.        # m outside the coverage over which the datum offset fades to 0 (<= the tile size)
FOOT_ALONG = (-8., -4., 0., 4., 8.)   # m along the road: a station's footprint (stations are 20 m apart)
FOOT_ACROSS = 1.5   # m either side of the centre line, at most half the width less 1 m
CACHE_TILES = 24    # whole tiles kept per process (4 MB each for 1 km at 1 m)
CALIBRATE_STEP = 100.
OPEN_LAND = .05     # forest share (fast_profile.Cover, sigma 100 m) under which a point calibrates the offset
LIFT = 20.0         # fast_sources.LIFT (not imported: fast_sources imports this module)


@dataclass(frozen=True)
class Provider:
    """A national or state terrain model. `folder` is under data/; `pattern` reads a tile's south-west
    corner (km) and year from its file name (else the GeoTIFF headers are read)."""
    id: str
    product: str
    area: str
    crs: str
    licence: str
    credit: str
    url: str
    datum: str = ''
    resolution: float = 1.
    tile: float = 1000.
    folder: str = ''
    pattern: str = ''
    surface: str = ''           # the matching surface model (DSM) folder: deck checks only
    surface_pattern: str = ''
    index: str = ''             # machine-readable tile index (downloader)
    verified: bool = False
    notes: str = ''


# Only de-ni is implemented and checked (S1.1 of ALTITUDE-REFERENCE.md). The rest are what is known about
# other free 1 m models, from memory and earlier reading: URLs, licences, CRS and tiling are UNVERIFIED and
# nothing was downloaded. Check the provider's current terms before adding a folder for it.
PROVIDERS = (
    Provider('de-ni', 'DGM1 (Digitales Gelaendemodell 1 m)', 'Niedersachsen', 'EPSG:25832', 'CC BY 4.0',
             '(c) GeoBasis-DE/LGLN {year}, CC BY 4.0, modified (sampled and smoothed road profiles and ground grid)',
             'https://ni-lgln-opengeodata.hub.arcgis.com/', 'DHHN2016 (NHN)', 1., 1000., 'dgm1/ni',
             r'^dgm1_32_(?P<e>\d{3})_(?P<n>\d{4})_1_ni_(?P<year>\d{4})\.tif$', 'dom1/ni',
             r'^dom1_32_(?P<e>\d{3})_(?P<n>\d{4})_1_ni_(?P<year>\d{4})\.tif$',
             'https://services-eu1.arcgis.com/4v3xxN52w88W065F/arcgis/rest/services/lgln_opengeodata_dgm1/FeatureServer/0',
             True, 'AGNB par. 7; COG re-derivation of Q1/2026; bridges removed; STAC https://dgm.stac.lgln.niedersachsen.de'),
    Provider('de-hb', 'DGM1', 'Bremen', 'EPSG:25832', 'CC BY 4.0 (unverified)', '(c) GeoBasis-DE/GeoInformation Bremen {year}',
             'https://www.geo.bremen.de/', 'DHHN2016', notes='not covered by the LGLN tiles; the Niedersachsen pack holds Bremen'),
    Provider('de-nw', 'DGM1', 'Nordrhein-Westfalen', 'EPSG:25832', 'dl-de/zero-2.0 (unverified)',
             'Geobasis NRW {year}', 'https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/', 'DHHN2016'),
    Provider('de-sh', 'DGM1', 'Schleswig-Holstein', 'EPSG:25832', 'CC BY 4.0 (unverified)', '(c) GeoBasis-DE/LVermGeo SH {year}',
             'https://geodaten.schleswig-holstein.de/', 'DHHN2016'),
    Provider('de-hh', 'DGM1', 'Hamburg', 'EPSG:25832', 'dl-de/by-2.0 (unverified)', 'Freie und Hansestadt Hamburg, LGV {year}',
             'https://transparenz.hamburg.de/', 'DHHN2016'),
    Provider('de-st', 'DGM1', 'Sachsen-Anhalt', 'EPSG:25832', 'dl-de/by-2.0 (unverified)', '(c) GeoBasis-DE/LVermGeo ST {year}',
             'https://www.lvermgeo.sachsen-anhalt.de/', 'DHHN2016'),
    Provider('de-th', 'DGM1', 'Thueringen', 'EPSG:25832', 'dl-de/by-2.0 (unverified)', '(c) GDI-Th {year}',
             'https://geoportal.thueringen.de/', 'DHHN2016'),
    Provider('de-sn', 'DGM1', 'Sachsen', 'EPSG:25833', 'dl-de/by-2.0 (unverified)', '(c) GeoSN {year}',
             'https://www.geodaten.sachsen.de/', 'DHHN2016'),
    Provider('de-bb', 'DGM1', 'Brandenburg', 'EPSG:25833', 'dl-de/by-2.0 (unverified)', '(c) GeoBasis-DE/LGB {year}',
             'https://data.geobasis-bb.de/geobasis/daten/dgm/', 'DHHN2016'),
    Provider('de-be', 'DGM1', 'Berlin', 'EPSG:25833', 'dl-de/zero-2.0 (unverified)', 'Geoportal Berlin {year}',
             'https://gdi.berlin.de/', 'DHHN2016'),
    Provider('de-mv', 'DGM1', 'Mecklenburg-Vorpommern', 'EPSG:25833', 'dl-de/by-2.0 (unverified)', '(c) GeoBasis-DE/M-V {year}',
             'https://www.geoportal-mv.de/', 'DHHN2016'),
    Provider('de-by', 'DGM1', 'Bayern', 'EPSG:25832', 'CC BY 4.0 (unverified)', 'Bayerische Vermessungsverwaltung {year}',
             'https://geodaten.bayern.de/opengeodata/', 'DHHN2016'),
    Provider('de-rp', 'DGM1', 'Rheinland-Pfalz', 'EPSG:25832', 'dl-de/by-2.0 (unverified)', '(c) GeoBasis-DE/LVermGeoRP {year}',
             'https://geobasis-rlp.de/', 'DHHN2016'),
    Provider('nl', 'AHN4/AHN5 DTM 0.5 m', 'Nederland', 'EPSG:28992', 'CC0 1.0 (unverified)', 'AHN (Rijkswaterstaat / Het Waterschapshuis)',
             'https://www.ahn.nl/ (PDOK download service)', 'NAP', .5, 5000.,
             notes='tiles are 5 x 6.25 km map sheets; water and buildings are nodata: use a gap-filled product or fill first'),
    Provider('be-vl', 'DHM Vlaanderen II DTM 1 m', 'Vlaanderen', 'EPSG:31370', 'Modellicentie Gratis Hergebruik (unverified)',
             'Digitaal Vlaanderen', 'https://www.vlaanderen.be/datavindplaats', 'TAW'),
    Provider('lu', 'LiDAR DTM 0.5 m', 'Luxembourg', 'EPSG:2169', 'CC0 (unverified)', 'Administration du cadastre et de la topographie',
             'https://data.public.lu/', 'NG95', .5),
    Provider('dk', 'DHM/Terraen 0.4 m', 'Danmark', 'EPSG:25832', 'free with an account (unverified)', 'Klimadatastyrelsen, DHM',
             'https://dataforsyningen.dk/', 'DVR90', .4),
    Provider('fr', 'RGE ALTI 1 m / LiDAR HD MNT', 'France', 'EPSG:2154', 'Licence Ouverte Etalab 2.0 (unverified)', 'IGN',
             'https://geoservices.ign.fr/', 'NGF-IGN69'),
    Provider('ch', 'swissALTI3D 0.5 m', 'Schweiz', 'EPSG:2056', 'OGD, free (unverified)', '(c) swisstopo',
             'https://www.swisstopo.admin.ch/', 'LN02', .5),
    Provider('pl', 'NMT 1 m', 'Polska', 'EPSG:2180', 'free reuse (unverified)', 'GUGiK', 'https://www.geoportal.gov.pl/', 'PL-EVRF2007-NH'),
    Provider('cz', 'DMR 5G', 'Cesko', 'EPSG:5514', 'CC BY 4.0 (unverified)', 'CUZK', 'https://geoportal.cuzk.cz/', 'Bpv', 2.),
    Provider('uk-en', 'LIDAR Composite DTM 1 m', 'England', 'EPSG:27700', 'OGL v3 (unverified)', 'Environment Agency',
             'https://environment.data.gov.uk/', 'ODN'),
    Provider('no', 'DTM 1 m', 'Norge', 'EPSG:25833', 'CC BY 4.0 (unverified)', 'Kartverket', 'https://hoydedata.no/', 'NN2000'),
    Provider('fi', 'Elevation model 2 m', 'Suomi', 'EPSG:3067', 'CC BY 4.0 (unverified)', 'National Land Survey of Finland',
             'https://www.maanmittauslaitos.fi/', 'N2000', 2.),
    Provider('se', 'Markhoejdmodell 1 m', 'Sverige', 'EPSG:3006', 'CC0 (unverified)', 'Lantmaeteriet', 'https://www.lantmateriet.se/', 'RH2000'),
    Provider('es', 'MDT02', 'Espana', 'EPSG:25830', 'CC BY 4.0 (unverified)', 'CNIG / IGN', 'https://centrodedescargas.cnig.es/', 'EGM08-REDNAP', 2.),
    Provider('us', '3DEP 1 m DEM', 'United States', 'various UTM', 'public domain (unverified)', 'USGS 3DEP',
             'https://www.usgs.gov/3d-elevation-program', 'NAVD88'),
)
BY_ID = {p.id: p for p in PROVIDERS}


def _smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3 - 2 * u)


class TileSet:
    """One provider's square tiles in one folder: tile index (from names or headers), an LRU of whole
    tiles (a 1 km float32 tile is 4 MB), nearest-pixel and block-mean reads, coverage distances."""

    def __init__(self, root, crs, tile=1000., pattern='', provider=''):
        self.root, self.crs, self.size, self.provider = Path(root), crs, float(tile), provider
        self.files = {}                         # (i, j) tile index -> (path, year)
        rx = re.compile(pattern) if pattern else None
        for path in sorted(self.root.glob('*.tif')) if self.root.is_dir() else ():
            if rx is not None:
                m = rx.match(path.name)
                if not m:
                    continue
                e, n = int(m['e']) * 1000., int(m['n']) * 1000.
                year = int(m['year']) if 'year' in m.groupdict() and m['year'] else 0
            else:
                import rasterio
                with rasterio.open(path) as ds:
                    b = ds.bounds
                e, n = b.left, b.bottom
                found = re.search(r'(?<!\d)(19\d\d|20\d\d)(?!\d)', path.stem)
                year = int(found[1]) if found else 0
            key = (int(math.floor(e / self.size + .5)), int(math.floor(n / self.size + .5)))
            if key not in self.files or year > self.files[key][1]:      # the newest flight wins
                self.files[key] = (path, year)
        self.codes = np.array(sorted(self._code(i, j) for i, j in self.files), dtype=np.int64)
        self.cache = OrderedDict()

    @staticmethod
    def _code(i, j):
        return (np.int64(i) + 1_000_000) * 2_000_000 + (np.int64(j) + 1_000_000)

    def __len__(self):
        return len(self.files)

    def has(self, i, j):
        """vectorised: tile (i, j) present"""
        code = self._code(np.asarray(i, np.int64), np.asarray(j, np.int64))
        if not len(self.codes):
            return np.zeros(np.shape(code), bool)
        k = np.clip(np.searchsorted(self.codes, code), 0, len(self.codes) - 1)
        return self.codes[k] == code

    def keys_in(self, box, margin=0.):
        """tiles touching box (e0, n0, e1, n1) in the provider's CRS"""
        e0, n0, e1, n1 = box
        return sorted(k for k in self.files
                      if k[0] * self.size < e1 + margin and (k[0] + 1) * self.size > e0 - margin
                      and k[1] * self.size < n1 + margin and (k[1] + 1) * self.size > n0 - margin)

    def identity(self, keys=None):
        h = hashlib.sha256(f'{self.provider}:{self.crs}:{self.size}'.encode())
        for key in sorted(self.files if keys is None else keys):
            if key in self.files:
                p = self.files[key][0]
                st = p.stat()
                h.update(f'{p.name}:{st.st_size}:{st.st_mtime_ns}\0'.encode())
        return h.hexdigest()

    def years(self, keys=None):
        return sorted({self.files[k][1] for k in (self.files if keys is None else keys) if k in self.files and self.files[k][1]})

    def tile(self, key):
        """(array rows north to south, west edge, north edge, pixel size) or None"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        entry = self.files.get(key)
        out = None
        if entry is not None:
            import rasterio
            with rasterio.open(entry[0]) as ds:
                a = ds.read(1).astype(np.float32)
                nodata, t = ds.nodata, ds.transform
            if nodata is not None:
                a[a == nodata] = np.nan
            a[~np.isfinite(a) | (a < -1000.)] = np.nan
            out = (a, float(t.c), float(t.f), float(t.a))
        self.cache[key] = out
        if len(self.cache) > CACHE_TILES:
            self.cache.popitem(last=False)
        return out

    def at(self, e, n):
        """nearest pixel (NaN outside the tiles or on nodata)"""
        e = np.asarray(e, float).ravel(); n = np.asarray(n, float).ravel()
        out = np.full(len(e), np.nan, np.float32)
        if not len(e) or not self.files:
            return out
        i = np.floor(e / self.size).astype(np.int64); j = np.floor(n / self.size).astype(np.int64)
        present = self.has(i, j)
        code = self._code(i, j)
        for c in np.unique(code[present]).tolist():
            m = code == c
            key = (int(c // 2_000_000 - 1_000_000), int(c % 2_000_000 - 1_000_000))
            t = self.tile(key)
            if t is None:
                continue
            a, west, north, res = t
            col = np.clip(np.floor((e[m] - west) / res).astype(np.int64), 0, a.shape[1] - 1)
            row = np.clip(np.floor((north - n[m]) / res).astype(np.int64), 0, a.shape[0] - 1)
            out[m] = a[row, col]
        return out

    def mosaic(self, e0, n0, e1, n1):
        """the pixels over a box (NaN where no tile): (array north to south, west, north, res)"""
        keys = self.keys_in((e0, n0, e1, n1))
        if not keys:
            return None
        res = self.tile(keys[0])[3] if self.tile(keys[0]) else 1.
        west = math.floor(e0 / res) * res; north = math.ceil(n1 / res) * res
        nx = int(math.ceil((e1 - west) / res)) + 1; ny = int(math.ceil((north - n0) / res)) + 1
        grid = np.full((ny, nx), np.nan, np.float32)
        for key in keys:
            t = self.tile(key)
            if t is None:
                continue
            a, tw, tn, tr = t
            c0 = int(round((tw - west) / res)); r0 = int(round((north - tn) / res))
            rs, re_ = max(0, r0), min(ny, r0 + a.shape[0]); cs, ce = max(0, c0), min(nx, c0 + a.shape[1])
            if rs < re_ and cs < ce:
                grid[rs:re_, cs:ce] = a[rs - r0:re_ - r0, cs - c0:ce - c0]
        return grid, west, north, res

    def coverage(self, e, n, blend=BLEND, fade=FADE):
        """share (0 outside, smoothstep to 1 at `blend` m inside the edge of the tiles) and fade (1 inside,
        smoothstep to 0 at `fade` m outside), from the distance to the nearest missing or present tile"""
        e = np.asarray(e, float).ravel(); n = np.asarray(n, float).ravel()
        S = self.size
        i = np.floor(e / S).astype(np.int64); j = np.floor(n / S).astype(np.int64)
        inside = self.has(i, j)
        d_in = np.full(len(e), np.inf); d_out = np.full(len(e), np.inf)
        reach = int(math.ceil(max(blend, fade) / S))
        for di in range(-reach - 1, reach + 2):
            for dj in range(-reach - 1, reach + 2):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                dx = np.maximum(np.maximum(ii * S - e, e - (ii + 1) * S), 0.)
                dy = np.maximum(np.maximum(jj * S - n, n - (jj + 1) * S), 0.)
                d = np.hypot(dx, dy)
                there = self.has(ii, jj)
                d_in = np.where(inside & ~there, np.minimum(d_in, d), d_in)
                d_out = np.where(~inside & there, np.minimum(d_out, d), d_out)
        share = np.where(inside, _smoothstep(d_in / blend), 0.)
        fading = np.where(inside, 1., _smoothstep(1. - d_out / fade))
        return share, fading


class Dtm:
    """The national terrain models of one region (index window) in its frame.

    Heights are returned in the exporter's datum: metres above the model's datum + LIFT (the coarse
    sampler's convention). `offset` (datum, m) is added to coarse heights inside the coverage and faded
    out over FADE m outside it."""

    def __init__(self, sets, frame, offset=0., surfaces=None):
        self.sets = list(sets)                   # [(Provider, TileSet)], in priority order
        self.surfaces = dict(surfaces or {})     # provider id -> TileSet of its DSM
        self.frame, self.offset = frame, float(offset)
        self._proj = {}

    # -- construction ---------------------------------------------------------------------------------
    @classmethod
    def region(cls, frame, box, data=None):
        """The providers with local tiles under the lon/lat box (None when none covers it); data: DATA."""
        data = DATA if data is None else data
        sets, surfaces = [], {}
        for p in PROVIDERS:
            if not p.folder or not (Path(data) / p.folder).is_dir():
                continue
            ts = TileSet(Path(data) / p.folder, p.crs, p.tile, p.pattern, p.id)
            if not len(ts) or not ts.keys_in(_project_box(box, p.crs), FADE):
                continue
            sets.append((p, ts))
            if p.surface and (Path(data) / p.surface).is_dir():
                surfaces[p.id] = TileSet(Path(data) / p.surface, p.crs, p.tile, p.surface_pattern, p.id + ':surface')
        return cls(sets, frame, 0., surfaces) if sets else None

    @classmethod
    def from_meta(cls, meta, frame):
        """The index's DTM (region.json 'dtm', written by meta())."""
        sets, surfaces = [], {}
        for entry in meta['providers']:
            p = BY_ID[entry['id']]
            sets.append((p, TileSet(entry['root'], p.crs, p.tile, p.pattern, p.id)))
            if entry.get('surface'):
                surfaces[p.id] = TileSet(entry['surface'], p.crs, p.tile, p.surface_pattern, p.id + ':surface')
        return cls(sets, frame, meta.get('offset', 0.), surfaces)

    def meta(self, box=None, **extra):
        out = {'version': VERSION, 'offset': round(self.offset, 3), 'blend': BLEND, 'fade': FADE, 'providers': []}
        for p, ts in self.sets:
            keys = ts.keys_in(_project_box(box, p.crs), FADE) if box is not None else None
            sf = self.surfaces.get(p.id)
            out['providers'].append({'id': p.id, 'root': str(ts.root.resolve()), 'tiles': len(keys if keys is not None else ts.files),
                                     'years': ts.years(keys), 'surface': str(sf.root.resolve()) if sf is not None else None})
        out.update(extra)
        return out

    def credits(self):
        """{source, data, license, url} per provider, for the pack's manifest and CREDITS.txt"""
        out = []
        for p, ts in self.sets:
            years = ts.years()
            year = '-'.join(str(y) for y in (years[0], years[-1])) if len(years) > 1 and years[0] != years[-1] else (
                str(years[0]) if years else '')
            credit = p.credit.format(year=year).strip()
            out.append({'source': f'{p.product}, {p.area}: {credit}', 'data': 'road and ground heights',
                        'license': p.licence, 'url': p.url, 'credit': credit})
        return out

    @staticmethod
    def identity(box, data=None):
        """name, size and mtime of every local tile under the lon/lat box (the index cache key)"""
        data = DATA if data is None else data
        h = hashlib.sha256(SIGNATURE.encode())
        for p in PROVIDERS:
            for folder, pattern in ((p.folder, p.pattern), (p.surface, p.surface_pattern)):
                if folder and (Path(data) / folder).is_dir():
                    ts = TileSet(Path(data) / folder, p.crs, p.tile, pattern, p.id)
                    keys = ts.keys_in(_project_box(box, p.crs), FADE)
                    if keys:
                        h.update(f'{p.id}:{folder}:'.encode() + ts.identity(keys).encode())
        return h.hexdigest()

    # -- sampling -------------------------------------------------------------------------------------
    def _to(self, crs, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if crs not in self._proj:
            import pyproj
            tr = None if pyproj.CRS(crs) == pyproj.CRS(self.frame.crs) else \
                pyproj.Transformer.from_crs(self.frame.crs, crs, always_xy=True)
            if tr is not None:
                # WGS 84 / UTM and ETRS89 / UTM of one zone: pyproj's transformation is the identity (to the
                # millimetre) here; skipping it saves a third of a cell's DTM time
                tx = self.frame.cx + np.array([-5e4, 0., 5e4]); ty = self.frame.cy + np.array([-5e4, 0., 5e4])
                e, n = tr.transform(tx, ty)
                if np.abs(np.asarray(e) - tx).max() < 1e-3 and np.abs(np.asarray(n) - ty).max() < 1e-3:
                    tr = None
            self._proj[crs] = tr
        t = self._proj[crs]
        x, y = xy[:, 0] + self.frame.cx, xy[:, 1] + self.frame.cy
        if t is None:
            return x, y
        e, n = t.transform(x, y)
        return np.asarray(e, float), np.asarray(n, float)

    def near(self, xy):
        """False when every point is more than FADE m from every tile (the DTM changes nothing there)"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not len(xy):
            return False
        lo, hi = xy.min(0), xy.max(0)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        for p, ts in self.sets:
            e, n = self._to(p.crs, corners)
            if ts.keys_in((e.min(), n.min(), e.max(), n.max()), FADE + 1.):
                return True
        return False

    def coverage(self, xy):
        """(share, fade) of each frame point, the largest over the providers"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        share, fade = np.zeros(len(xy)), np.zeros(len(xy))
        if not self.near(xy):
            return share, fade
        for p, ts in self.sets:
            e, n = self._to(p.crs, xy)
            s, f = ts.coverage(e, n)
            share, fade = np.maximum(share, s), np.maximum(fade, f)
        return share, fade

    def point(self, xy):
        """nearest-pixel DTM height + LIFT (NaN where none)"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        for p, ts in self.sets:
            todo = ~np.isfinite(out)
            if not todo.any():
                break
            e, n = self._to(p.crs, xy[todo])
            out[todo] = ts.at(e, n)
        return out + LIFT

    def road(self, xy, tangent, half):
        """Per station: the median of the DTM over its footprint (FOOT_ALONG m along the road, up to
        FOOT_ACROSS m either side within the carriageway) + LIFT, and its share (0 where none)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        share, _ = self.coverage(xy)
        out = np.full(len(xy), np.nan)
        m = share > 0
        if not m.any():
            return out, share
        t = np.asarray(tangent, float).reshape(-1, 2)[m]
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        across = np.minimum(FOOT_ACROSS, np.maximum(np.asarray(half, float).ravel()[m] - 1., 0.))
        # every footprint point in one read (each tile is loaded once per call)
        q = np.concatenate([xy[m] + a * t + (b * across)[:, None] * nrm for a in FOOT_ALONG for b in (-1., 0., 1.)])
        samples = self.point(q).reshape(len(FOOT_ALONG) * 3, int(m.sum()))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[m] = np.nanmedian(samples, axis=0)
        share = np.where(np.isfinite(out), share, 0.)
        return out, share

    def lattice(self, xs, ys):
        """The DTM's mean over each node's step x step block of the lattice ys x xs (row-major) + LIFT,
        and the node's share (0 where none)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        step = float(xs[1] - xs[0]) if len(xs) > 1 else 1.
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        share, _ = self.coverage(pts)
        out = np.full(len(pts), np.nan)
        for p, ts in self.sets:
            todo = (share > 0) & ~np.isfinite(out)
            if not todo.any():
                break
            e, n = self._to(p.crs, pts[todo])
            half = step / 2
            got = ts.mosaic(e.min() - half - 2, n.min() - half - 2, e.max() + half + 2, n.max() + half + 2)
            if got is None:
                continue
            grid, west, north, res = got
            ok = np.isfinite(grid)
            S = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1)); C = np.zeros_like(S)
            S[1:, 1:] = np.where(ok, grid, 0.).astype(np.float64).cumsum(0).cumsum(1)    # float32 sums lose cm
            C[1:, 1:] = ok.cumsum(0).cumsum(1)
            c0 = np.clip(np.round((e - half - west) / res).astype(np.int64), 0, grid.shape[1])
            c1 = np.clip(np.round((e + half - west) / res).astype(np.int64), 0, grid.shape[1])
            r0 = np.clip(np.round((north - (n + half)) / res).astype(np.int64), 0, grid.shape[0])
            r1 = np.clip(np.round((north - (n - half)) / res).astype(np.int64), 0, grid.shape[0])
            box = lambda A: A[r1, c1] - A[r0, c1] - A[r1, c0] + A[r0, c0]
            cnt = box(C)
            with np.errstate(invalid='ignore', divide='ignore'):
                val = np.where(cnt > 0, box(S) / np.maximum(cnt, 1), np.nan)
            out[todo] = val
        share = np.where(np.isfinite(out), share, 0.)
        return out + LIFT, share

    def surface(self, xy, tangent, half, q=20):
        """DSM (DOM1) low percentile over 5 m x (width - 2 m) and its roughness: (p_q + LIFT, smooth),
        NaN without a surface model. Only a check on decks: never a target."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.full(len(xy), np.nan); sd = np.full(len(xy), np.nan)
        if not self.surfaces:
            return out, np.zeros(len(xy), bool)
        t = np.asarray(tangent, float).reshape(-1, 2)
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        across = np.maximum(np.asarray(half, float).ravel() - 1., 0.)
        vals = []
        for p, _ in self.sets:
            sf = self.surfaces.get(p.id)
            if sf is None:
                continue
            pts = np.concatenate([xy + a * t + (b * across)[:, None] * nrm for a in (-2., -1., 0., 1., 2.)
                                  for b in np.linspace(-1., 1., 5)])
            e, n = self._to(p.crs, pts)
            vals = list(sf.at(e, n).reshape(25, len(xy)))
            break
        if not vals:
            return out, np.zeros(len(xy), bool)
        import warnings
        v = np.array(vals, float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.nanpercentile(v, q, axis=0)
            sd = np.nanstd(v, axis=0)
        return out + LIFT, np.isfinite(out) & (sd < .5)

    # -- blending -------------------------------------------------------------------------------------
    def corrected(self, coarse, xy):
        """coarse heights shifted by the datum offset where they meet the DTM (faded outside it)"""
        if not self.offset:
            return np.asarray(coarse, float)
        _, fade = self.coverage(xy)
        return np.asarray(coarse, float) + self.offset * fade

    def blend(self, coarse, xy, fine=None, share=None, fade=None):
        """share x DTM + (1 - share) x (coarse + offset x fade); fine/share default to point samples"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if share is None or fade is None:
            s, f = self.coverage(xy)
            share = s if share is None else share
            fade = f if fade is None else fade
        if fine is None:
            fine = self.point(xy)
            share = np.where(np.isfinite(fine), share, 0.)
        coarse = np.asarray(coarse, float) + self.offset * fade
        out = share * np.where(np.isfinite(fine), fine, 0.) + (1 - share) * np.where(np.isfinite(coarse), coarse, 0.)
        bad = ((share < 1) & ~np.isfinite(coarse)) | ((share > 0) & ~np.isfinite(fine))
        return np.where(bad, np.nan, out)

    def calibrate(self, coarse, box=None, keep=None):
        """offset = median(DTM - coarse) on a CALIBRATE_STEP grid inside the coverage (share 1), within
        the frame box (x0, y0, x1, y1) when given, at the points keep(xy) accepts (open land: a coarse DEM
        reads the canopy in forest); coarse(xy) may return NaN. Returns (offset, n)."""
        pts = []
        g = (np.arange(CALIBRATE_STEP / 2, 1000., CALIBRATE_STEP))
        for p, ts in self.sets:
            import pyproj
            back = None if pyproj.CRS(p.crs) == pyproj.CRS(self.frame.crs) else \
                pyproj.Transformer.from_crs(p.crs, self.frame.crs, always_xy=True)
            for i, j in sorted(ts.files):
                ge, gn = np.meshgrid(i * ts.size + g * ts.size / 1000., j * ts.size + g * ts.size / 1000.)
                e, n = ge.ravel(), gn.ravel()
                if back is not None:
                    e, n = back.transform(e, n)
                pts.append(np.column_stack([np.asarray(e) - self.frame.cx, np.asarray(n) - self.frame.cy]))
        if not pts:
            return self.offset, 0
        xy = np.concatenate(pts)
        if box is not None:
            x0, y0, x1, y1 = box
            xy = xy[(xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)]
        share, _ = self.coverage(xy)
        xy = xy[share >= 1]
        if keep is not None and len(xy):
            xy = xy[np.asarray(keep(xy), bool)]
        if not len(xy):
            return self.offset, 0
        d = self.point(xy) - np.asarray(coarse(xy), float)
        d = d[np.isfinite(d)]
        if len(d) >= 20:
            self.offset = float(np.median(d))
        return self.offset, int(len(d))

    def deck_check(self, segments, profiles, step=5.):
        """Bridge stations against the DSM's deck (p20, smooth test): (n, median, p95 |d|), heights in m.
        A report only: OSM decks and DOM1 decks need not be the same structure."""
        xy, tang, half, h = [], [], [], []
        for seg, (ps, ph) in zip(segments, profiles):
            if seg['tags'].get('bridge') in (None, 'no'):
                continue
            pts = np.asarray(seg['xy'], float)
            cum = np.r_[0., np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            if cum[-1] < 4 * step:
                continue
            s = np.arange(step, cum[-1] - step, step)          # interior: the abutments may be off the deck
            x, y = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
            gx, gy = np.gradient(x), np.gradient(y); L = np.maximum(np.hypot(gx, gy), 1e-9)
            xy.append(np.column_stack([x, y])); tang.append(np.column_stack([gx / L, gy / L]))
            half.append(np.full(len(s), seg.get('width', 7.) / 2)); h.append(np.interp(s, ps, ph))
        if not xy or not self.surfaces:
            return 0, float('nan'), float('nan')
        top, smooth = self.surface(np.concatenate(xy), np.concatenate(tang), np.concatenate(half))
        d = (np.concatenate(h) - top)[smooth]
        if not len(d):
            return 0, float('nan'), float('nan')
        return int(len(d)), float(np.median(d)), float(np.percentile(np.abs(d - np.median(d)), 95))


def _project_box(box, crs):
    """lon/lat box -> (e0, n0, e1, n1) in crs"""
    import pyproj
    t = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    w, s, e, n = box
    lon = np.r_[np.linspace(w, e, 9), np.linspace(w, e, 9), np.full(9, w), np.full(9, e)]
    lat = np.r_[np.full(9, s), np.full(9, n), np.linspace(s, n, 9), np.linspace(s, n, 9)]
    x, y = t.transform(lon, lat)
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


class Ground:
    """The export's ground sampler with a DTM (build_country): the despiked coarse DEM blended with the
    DTM. sample(xy) for scenery points; cell_grid(xs, ys, roads) for a cell's elevation grid.

    lower(xy), when given, is taken off the coarse DEM before the blend: the DEM rules' forest bias
    (fast_profile.FOREST_BIAS x the index's forest share), so the ground away from the coverage is the same
    corrected coarse DEM the road profiles were solved on, blended into the DTM by the same share."""
    PULL_FULL, PULL_ZERO = 12., 70.         # the generator's TUNE pull for a 25 m grid (generator.lua)

    def __init__(self, coarse, dtm, lower=None):
        self.coarse, self.dtm = coarse, dtm
        self.lower = lower
        self.terrain = getattr(coarse, 'terrain', coarse)

    def _coarse(self, heights, xy):
        heights = np.asarray(heights, float)
        return heights if self.lower is None else heights - np.asarray(self.lower(xy), float)

    def sample(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not len(xy):
            return np.array([])
        coarse = self._coarse(self.coarse.sample(xy), xy)
        return self.dtm.blend(coarse, xy) if self.dtm.near(xy) else coarse

    def cell_grid(self, xs, ys, roads=()):
        """The lattice heights (row-major): the DTM's block means where it covers, the coarse DEM elsewhere
        (shifted by the datum offset near the coverage), the coarse part pulled to the roads' profiles as
        the generator pulls a 25 m grid (a 10 m grid switches it to its LiDAR pull of 3-15 m)."""
        grid = getattr(self.coarse, 'grid', None)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        coarse = self._coarse(grid(xs, ys) if grid else self.coarse.sample(pts), pts)
        if self.dtm.near(pts):
            fine, share = self.dtm.lattice(xs, ys)
            _, fade = self.dtm.coverage(pts)
            base = self.dtm.blend(coarse, pts, fine, share, fade)
        else:                                   # more than FADE m from the DTM: the coarse DEM as it is
            share, base = np.zeros(len(pts)), coarse
        if (share < 1).any() and roads:
            base = pull_to_roads(base, xs, ys, roads, 1. - share, self.PULL_FULL, self.PULL_ZERO)
        return base


def pull_to_roads(heights, xs, ys, roads, weight, pull_full, pull_zero):
    """The generator's `pulled` (generator.lua): each lattice node moves toward the profile of the ground
    roads within pull_zero m of their edge (fully within pull_full), nearer roads weighing more, by
    weight (0..1) of the full pull. roads: compile_cell's road features (points x, y, z; width; tags)."""
    heights = np.asarray(heights, float).copy()
    nx = len(xs)
    step = float(xs[1] - xs[0]) if nx > 1 else 1.
    A, B, W = [], [], []
    for road in roads:
        tags = road.get('tags', {})
        if tags.get('bridge') not in (None, 'no') or tags.get('tunnel') not in (None, 'no', 'building_passage'):
            continue
        p = np.asarray(road['points'], float)
        if len(p) < 2:
            continue
        A.append(p[:-1]); B.append(p[1:]); W.append(np.full(len(p) - 1, max(.5, min(80., float(road['width'])))))
    if not A:
        return heights
    A, B, W = np.concatenate(A), np.concatenate(B), np.concatenate(W)
    keep = np.hypot(*(B - A)[:, :2].T) > 1e-6
    A, B, W = A[keep], B[keep], W[keep]
    reach = W / 2 + pull_zero
    lo_x = np.clip(np.ceil((np.minimum(A[:, 0], B[:, 0]) - reach - xs[0]) / step), 0, nx).astype(np.int64)
    hi_x = np.clip(np.floor((np.maximum(A[:, 0], B[:, 0]) + reach - xs[0]) / step) + 1, 0, nx).astype(np.int64)
    lo_y = np.clip(np.ceil((np.minimum(A[:, 1], B[:, 1]) - reach - ys[0]) / step), 0, len(ys)).astype(np.int64)
    hi_y = np.clip(np.floor((np.maximum(A[:, 1], B[:, 1]) + reach - ys[0]) / step) + 1, 0, len(ys)).astype(np.int64)
    cx, cy = np.maximum(hi_x - lo_x, 0), np.maximum(hi_y - lo_y, 0)
    count = cx * cy
    sa = np.zeros(len(heights)); sz = np.zeros(len(heights)); pull = np.zeros(len(heights))
    span = max(1., pull_zero - pull_full)
    # all (segment, node) pairs in chunks of about 2 M
    order = np.flatnonzero(count)
    start = 0
    while start < len(order):
        c = np.cumsum(count[order[start:]])
        end = start + max(1, int(np.searchsorted(c, 2_000_000)))
        seg = order[start:end]
        n = count[seg]
        k = np.repeat(seg, n)
        local = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ix = lo_x[k] + local % cx[k]; iy = lo_y[k] + local // cx[k]
        px, py = xs[ix], ys[iy]
        a, b = A[k], B[k]
        dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        t = np.clip(((px - a[:, 0]) * dx + (py - a[:, 1]) * dy) / (dx * dx + dy * dy), 0., 1.)
        d = np.maximum(0., np.hypot(px - (a[:, 0] + t * dx), py - (a[:, 1] + t * dy)) - W[k] / 2)
        near = d < pull_zero
        s = np.clip((pull_zero - d) / span, 0., 1.)
        w = s * s * (3 - 2 * s)
        wt = np.where(near, w / np.maximum(d, 1.), 0.)
        z = a[:, 2] + t * (b[:, 2] - a[:, 2])
        node = iy * nx + ix
        sa += np.bincount(node, wt, len(heights)); sz += np.bincount(node, wt * z, len(heights))
        np.maximum.at(pull, node[near], w[near])
        start = end
    has = sa > 0
    target = np.where(has, sz / np.where(has, sa, 1.), heights)
    return heights + (target - heights) * pull * np.asarray(weight, float)
