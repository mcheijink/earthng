"""Fetch national DTM tiles (dtm.PROVIDERS) into data/<folder>, with a sha256 manifest. Only LGLN
(de-ni, DGM1 and its DOM1) is implemented: the OpenGeoData.NI tile index FeatureServer gives each 1 km
tile's file URL (plain HTTPS, no account, CC BY 4.0, ALTITUDE-REFERENCE.md S1.1).

    python earthng/dtm_fetch.py de-ni --box W S E N [--surface] [--buffer M]           (plan, offline)
    python earthng/dtm_fetch.py de-ni --box W S E N --download --yes --max-tiles N     (network)

The default only lists the 1 km tiles under the lon/lat box that are not on disk yet (counts and an
estimated size; no network, no tile names printed). Downloading needs --download and --yes, stops at
--max-tiles and at --max-mb, resumes partial files, keeps the newest flight of a tile, and records
name, size and sha256 in the folder's manifest.json. Nothing is downloaded without those flags.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm  # noqa: E402

AGENT = 'EarthNG/0.1 (+https://github.com/Alezx311/earth2road)'
MB_PER_TILE = {'de-ni': 3.7, 'de-ni:surface': 3.1}
INDEX = {'de-ni': 'https://services-eu1.arcgis.com/4v3xxN52w88W065F/arcgis/rest/services/lgln_opengeodata_%s/FeatureServer/0/query'}


def tiles_in(provider, box, buffer=0.):
    """(i, j) 1 km tile indices under the lon/lat box, buffered by `buffer` m"""
    e0, n0, e1, n1 = dtm._project_box(box, provider.crs)
    s = provider.tile
    return [(i, j) for i in range(math.floor((e0 - buffer) / s), math.floor((e1 + buffer) / s) + 1)
            for j in range(math.floor((n0 - buffer) / s), math.floor((n1 + buffer) / s) + 1)]


def missing(provider, tiles, surface=False, data=None):
    data = dtm.DATA if data is None else data
    folder = Path(data) / (provider.surface if surface else provider.folder)
    have = dtm.TileSet(folder, provider.crs, provider.tile, provider.surface_pattern if surface else provider.pattern,
                       provider.id) if folder.is_dir() else None
    return [t for t in tiles if have is None or t not in have.files]


def _lgln_urls(product, tiles, session):
    """{(i, j): newest file URL} from the LGLN tile index (tile_id 32EEENNNN)"""
    urls = {}
    ids = ['32%03d%04d' % t for t in tiles]
    for k in range(0, len(ids), 40):
        q = {'where': 'tile_id IN (%s)' % ','.join(ids[k:k + 40]), 'outFields': f'tile_id,{product},Aktualitaet',
             'returnGeometry': 'false', 'f': 'json'}
        reply = session.get(INDEX['de-ni'] % product, params=q, timeout=60).json()
        for f in reply.get('features', ()):
            a = f['attributes']
            tid = str(a['tile_id'])      # the service now returns the id as a number
            key = (int(tid[2:5]), int(tid[5:9]))
            urls.setdefault(key, []).append(a[product])
        time.sleep(.5)
    return {k: sorted(v)[-1] for k, v in urls.items()}     # the newest flight's name sorts last


def download(provider, tiles, surface, max_tiles, max_mb, emit=print, data=None):
    import requests
    data = dtm.DATA if data is None else data
    if provider.id != 'de-ni':
        raise SystemExit(f'{provider.id}: no downloader (only de-ni is implemented)')
    product = 'dom1' if surface else 'dgm1'
    folder = Path(data) / (provider.surface if surface else provider.folder)
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = AGENT
    tiles = tiles[:max_tiles]
    urls = _lgln_urls(product, tiles, session)
    emit(f'{product}: the index has {len(urls)} of {len(tiles)} tiles')
    sizes = {}
    for url in urls.values():
        h = session.head(url, timeout=60)
        if h.status_code == 200:
            sizes[url] = int(h.headers.get('Content-Length', 0))
        time.sleep(.2)
    total = sum(sizes.values()) / 1e6
    emit(f'{product}: {len(sizes)} files, {total:.1f} MB')
    if total > max_mb:
        raise SystemExit(f'stop: {total:.0f} MB is over --max-mb {max_mb:.0f}')
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'source': 'LGLN OpenGeoData.NI, CC BY 4.0, (c) GeoBasis-DE/LGLN', 'files': []}
    known = {f['name'] for f in manifest['files']}
    done = 0
    for url, want in sizes.items():
        dst = folder / url.rsplit('/', 1)[1]
        if dst.exists() and dst.stat().st_size == want:
            continue
        part = dst.with_name(dst.name + '.part')
        for attempt in range(4):
            try:
                have = part.stat().st_size if part.exists() else 0
                headers = {'Range': f'bytes={have}-'} if have else {}
                with session.get(url, headers=headers, stream=True, timeout=120) as g:
                    if have and g.status_code != 206:
                        have = 0
                    g.raise_for_status()
                    with open(part, 'ab' if have else 'wb') as fo:
                        for chunk in g.iter_content(1 << 20):
                            fo.write(chunk)
                if part.stat().st_size == want:
                    part.replace(dst)
                    done += 1
                    break
            except Exception as exc:                     # network: retried, then counted as failed
                emit(f'retry {attempt + 1}: {type(exc).__name__}')
                time.sleep(3 * (attempt + 1))
        if dst.exists() and dst.name not in known:
            manifest['files'].append({'name': dst.name, 'size': dst.stat().st_size,
                                      'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
        time.sleep(.3)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    emit(f'{product}: {done} new files')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('provider', choices=sorted(dtm.BY_ID))
    ap.add_argument('--box', type=float, nargs=4, required=True, metavar=('W', 'S', 'E', 'N'))
    ap.add_argument('--buffer', type=float, default=0.)
    ap.add_argument('--surface', action='store_true', help='the surface model (DOM1) instead of the DTM')
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--yes', action='store_true', help='confirm the download')
    ap.add_argument('--max-tiles', type=int, default=0)
    ap.add_argument('--max-mb', type=float, default=1500.)
    a = ap.parse_args(argv)
    p = dtm.BY_ID[a.provider]
    if not p.folder:
        raise SystemExit(f'{p.id}: documented only ({p.licence}, {p.url}); no folder or downloader yet')
    todo = missing(p, tiles_in(p, a.box, a.buffer), a.surface)
    mb = MB_PER_TILE.get(p.id + (':surface' if a.surface else ''), 3.5)
    print(f'{p.id}{" surface" if a.surface else ""}: {len(todo)} tiles not on disk under the box, about {len(todo) * mb:.0f} MB')
    if not a.download:
        return
    if not a.yes or a.max_tiles <= 0:
        raise SystemExit('downloading needs --yes and --max-tiles N (the download is large; confirm it first)')
    download(p, todo, a.surface, a.max_tiles, a.max_mb)


if __name__ == '__main__':
    main()
