"""Official road data as exporter rules (EARTHNG_OFFICIAL=1, default off): the real pavement width and the medians.

Data: ALKIS "Tatsaechliche Nutzung" (vereinfacht) and ATKIS Basis-DLM, LGLN Niedersachsen, open data, CC BY 4.0
("ALKIS/ATKIS LGLN 2026, CC BY 4.0"), downloaded once as compact JSON per 2 x 2 km tile (EPSG:25832) into
EARTHNG_OFFICIAL_DIR (default data/official/de-ni; <e km>_<n km>.json, keys 'alkis' [{n, b, r}] and 'dlm' [{t, w, l, pts}]).
Nothing here is fetched at export time and no position is printed or logged.

1. Width prior (study p11): the ALKIS carriageway polygons (Strassenverkehr without a 'bez') are the real pavement
   envelope. Every ~STEP m along an OSM way of the swept classes a cross-section is cast perpendicular to it; the interval
   of the union of the polygons that holds the centre line is the envelope. Other ways running parallel inside the
   envelope (the opposite carriageway of a dual road, a slip road) divide it at the midway line, or at the edge of the
   median where one is detected (step 2). The way's width is what is left of the envelope on both sides of its centre line.
   Samples within MOUTH m of a junction node are left out, a median filter (FILTER samples) smooths the rest, and the way
   is cut into pieces of equal width / lanes (the exporter's part tapers blend the steps). Lanes are raised where the
   width clearly holds more than OSM maps, never lowered. Where ALKIS has nothing under the way the Basis-DLM
   Fahrbahnbreite is the fallback; otherwise the way is unchanged.
2. Medians (study p11, method C): between the two ways of a one-way pair, a raised strip in the DGM1 (relief against a
   12 m running median, MEDIAN_MIN..MEDIAN_MAX high, 1.5..6 m wide as measured, which overstates the kerb-to-kerb width by
   about 1 m) is a kerbed median. The carriageways end at its edges; it is emitted as a synthetic scenery island
   (kind 'island', tags area:highway=traffic_island), which the game's ROADS.island raises on the road beside it.

Values marked "study" were measured in the p11 study on 14 cross-sections at 6 sites; the rest are rules of thumb.
"""
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np

ENABLED = os.environ.get('EARTHNG_OFFICIAL', '0').strip().lower() not in ('', '0', 'off', 'no', 'false')
VERSION = 'official-1'
SIGNATURE = f'{VERSION}:{"on" if ENABLED else "off"}'
DATA = Path(os.environ.get('EARTHNG_OFFICIAL_DIR') or (Path(__file__).resolve().parent.parent / 'data' / 'official' / 'de-ni'))
YEAR = os.environ.get('EARTHNG_OFFICIAL_YEAR', '2026')
TILE = 2000.                        # m: the download's tile size
CRS = 'EPSG:25832'

SWEPT = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified'}
STEP = 20.                          # m between cross-sections
RAY = 45.                           # m: the cross-section's half length
MOUTH = 20.                         # m from a node with three or more ways: no samples (turn lanes, fillets)
MOUTH_SHARE = .22                   # ... and at most this share of a short way
FILTER = 5                          # samples of the median filter along a way
GAP = .5                            # m: polygon pieces closer than this are one envelope
CENTRE_TOL = 1.5                    # m: the centre line may lie this far outside the envelope (mapping offset)
PARALLEL = math.cos(math.radians(25.))
PLAZA = 2.5                         # an envelope share above this many times the OSM-implied width is a plaza, not a road
MAX_WIDTH = 30.
KEEP = .85                          # never narrower than this share of the width the tags imply
LANE_MIN, LANE_CAP = 2.9, 4.0       # m per lane: floor (explicit lanes tags), and cap (parking, gutters)
EDGE_STRIP = 1.0                    # m: gutter / marked edge strip beyond the lanes' cap
LANE_STEP, LANE_BIAS = 3.1, .4      # lanes = floor((width - shoulders + BIAS) / STEP): 6.0 m -> 2, 7.8 -> 2, 9.5 -> 3
TAG_WIDTH_TOL = .3                  # an OSM width tag within 30 % of the official width stands
PIECE_TOL = .8                      # m: width change that starts a new piece
EDGE_TOL = .6                       # m: change of one paved edge (relative to the OSM line) that starts a new piece
MIN_SIDE = 1.8                      # m: the narrower side of an ordinary road's pavement
MAX_SHIFT = 2.0                     # m: the centre of the pavement lies at most this far from the OSM line (not on a motorway / trunk)
SHORT_LINK = 60.                    # m: shorter links get no hard shoulder
HARD_SHOULDER = 3.3                 # m of unmarked paved shoulder beyond the edge line of a one-lane link (aerials: 3.0-4.3 m)
PIECE_MIN = 80.                     # m: shortest piece
MAX_PIECES = 15

# DGM1 medians (study)
PROFILE_BACK = 7.                   # m of profile beyond the envelope (the running median needs +-6 m)
PROFILE_STEP = .5
ALONG = np.arange(-8., 8.1, 2.)     # m: profile lines along the road, averaged
RUN_WINDOW = 25                     # samples = 12 m
MEDIAN_SEED, MEDIAN_KEEP = .05, .04   # m relief: a strip starts at SEED and extends while >= KEEP
MEDIAN_MIN, MEDIAN_MAX = .07, .35   # m: highest relief in the strip
MEDIAN_WIDTH = (1.5, 6.0)           # m measured (= true width + 1)
MEDIAN_EDGE = 1.5                   # m: the strip lies this far inside the envelope
MEDIAN_RUN = 40.                    # m: shortest median emitted
MEDIAN_ROUND = .75                  # m: corners rounded
MAX_PARTNER = 40.                   # m: the opposite carriageway's centre line is this close

ISLAND_ID = -(2 * 10 ** 12)         # synthetic scenery ids: ISLAND_ID - k (OSM ids are positive, point nodes use -(10**12 + id))
ISLAND_TAGS = {'area:highway': 'traffic_island', 'source': 'dgm1-median'}
REPORT = []                         # per changed draft: [way, ord, width before, lanes before, [[s0, s1, width, lanes], ...]] of the last refine


def credits():
    """{source, data, license, url, credit} for the pack's manifest and CREDITS.txt"""
    credit = f'ALKIS and ATKIS Basis-DLM, LGLN {YEAR}, CC BY 4.0 (processed: road widths and medians)'
    return [{'source': f'ALKIS (Tatsaechliche Nutzung) and ATKIS Basis-DLM, Niedersachsen: LGLN ({YEAR}) Creative Commons '
                       'Namensnennung - 4.0 International (CC BY 4.0)', 'data': 'road widths and medians',
             'license': 'CC BY 4.0', 'url': 'https://opendata.lgln.niedersachsen.de/', 'credit': credit}]


def identity():
    """the switch, the version and the name, size and mtime of the data files (the index cache key)"""
    h = hashlib.sha256(SIGNATURE.encode())
    if DATA.is_dir():
        for p in sorted(DATA.glob('*.json')):
            st = p.stat()
            h.update(f'{p.name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return h.hexdigest()


# =========================================================================================================
# data
# =========================================================================================================

def converters(frame):
    """(forward, inverse) between the index frame and ETRS89 / UTM 32 (the data's CRS): the shifted identity for UTM 32 in
    either datum (the transformation is under a millimetre there), else pyproj"""
    import pyproj
    cx, cy = frame.cx, frame.cy
    shift = (lambda x, y: (np.asarray(x, float) + cx, np.asarray(y, float) + cy),
             lambda e, n: (np.asarray(e, float) - cx, np.asarray(n, float) - cy))
    if pyproj.CRS(frame.crs) == pyproj.CRS(CRS):
        return shift
    tr = pyproj.Transformer.from_crs(frame.crs, CRS, always_xy=True)
    tx = cx + np.array([-5e4, 0., 5e4]); ty = cy + np.array([-5e4, 0., 5e4])
    e, n = tr.transform(tx, ty)
    if np.abs(np.asarray(e) - tx).max() < 1e-3 and np.abs(np.asarray(n) - ty).max() < 1e-3:
        return shift
    back = pyproj.Transformer.from_crs(CRS, frame.crs, always_xy=True)

    def inverse(e, n):
        x, y = back.transform(np.asarray(e, float), np.asarray(n, float))
        return np.asarray(x) - cx, np.asarray(y) - cy
    return (lambda x, y: tr.transform(np.asarray(x, float) + cx, np.asarray(y, float) + cy)), inverse


def _pair(inv, p):
    e, n = inv(p[:, 0], p[:, 1])
    return np.column_stack([e, n])


def _pieces(g, cell=150., big=300):
    """polygons with many vertices cut on a grid: a cross-section meets small pieces (their merge is the envelope again)"""
    import shapely
    out = []
    for p in (g.geoms if hasattr(g, 'geoms') else [g]):
        if p.geom_type != 'Polygon' or p.is_empty or p.area < .5:
            continue
        if shapely.get_num_coordinates(p) <= big:
            out.append(p)
            continue
        x0, y0, x1, y1 = p.bounds
        xs = np.arange(math.floor(x0 / cell) * cell, x1 + cell, cell)
        ys = np.arange(math.floor(y0 / cell) * cell, y1 + cell, cell)
        boxes = [shapely.box(a, b, a + cell, b + cell) for a in xs[:-1] for b in ys[:-1]]
        for q in shapely.intersection(p, boxes):
            out.extend(r for r in (q.geoms if hasattr(q, 'geoms') else [q]) if r.geom_type == 'Polygon' and r.area > .05)
    return out


class Data:
    """ALKIS carriageway polygons and DLM axes of the tiles under a box, in the index frame (E - cx, N - cy)."""

    def __init__(self, polys, axes=(), tiles=0):
        import shapely
        self.polys = np.empty(len(polys), dtype=object)
        self.polys[:] = list(polys)
        self.tree = shapely.STRtree(self.polys) if len(self.polys) else None
        self.axis_geoms = np.empty(len(axes), dtype=object)
        self.axis_geoms[:] = [a[0] for a in axes]
        self.axis_width = np.asarray([a[1] for a in axes], float)
        self.axis_tree = shapely.STRtree(self.axis_geoms) if len(axes) else None
        self.tiles = tiles

    @classmethod
    def load(cls, frame, box, directory=None, margin=150.):
        """box = (x0, y0, x1, y1) in the frame. Missing tiles are simply absent (no carriageway there)."""
        import shapely
        directory = Path(directory) if directory else DATA
        fwd, inv = converters(frame)
        x0, y0, x1, y1 = box
        e, n = fwd(np.array([x0 - margin, x1 + margin, x0 - margin, x1 + margin]),
                   np.array([y0 - margin, y0 - margin, y1 + margin, y1 + margin]))
        i0, i1 = int(math.floor(float(e.min()) / TILE)), int(math.floor(float(e.max()) / TILE))
        j0, j1 = int(math.floor(float(n.min()) / TILE)), int(math.floor(float(n.max()) / TILE))
        polys, axes, tiles = [], [], 0
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                f = directory / f'{int(i * TILE // 1000)}_{int(j * TILE // 1000)}.json'
                if not f.is_file():
                    continue
                try:
                    d = json.loads(f.read_text(encoding='utf8'))
                except (OSError, ValueError):
                    continue
                tiles += 1
                for a in d.get('alkis', ()):
                    if not str(a.get('n', '')).startswith('Stra') or a.get('b'):
                        continue                     # carriageway only: no Begleitflaeche, Weg or Platz
                    rings = [(r[0], np.asarray(r[1], float)) for r in a.get('r', ()) if len(r[1]) >= 4]
                    ext = [r for t, r in rings if t == 'e']
                    if not ext:
                        continue
                    g = shapely.Polygon(_pair(inv, ext[0]), [_pair(inv, r) for t, r in rings if t != 'e'])
                    if not g.is_valid:
                        g = shapely.make_valid(g)
                    polys.extend(_pieces(g))
                for a in d.get('dlm', ()):
                    try:
                        w = float(a['w']) if a.get('w') else None
                    except ValueError:
                        w = None
                    if w and len(a.get('pts', ())) >= 2:
                        axes.append((shapely.LineString(_pair(inv, np.asarray(a['pts'], float))), w))
        return cls(polys, axes, tiles)


# =========================================================================================================
# cross-sections
# =========================================================================================================

def way_samples(pts, step=STEP):
    """x, y, unit tangent (tx, ty) and arc length of the cross-sections along a polyline (about every `step` m), and its length"""
    P = np.asarray(pts, float)
    cum = np.r_[0., np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    L = float(cum[-1])
    n = max(1, int(round(L / step)))
    s = (np.arange(n) + .5) * L / n
    a, b = np.clip(s - 1.5, 0., L), np.clip(s + 1.5, 0., L)
    x, y = np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])
    tx = np.interp(b, cum, P[:, 0]) - np.interp(a, cum, P[:, 0])
    ty = np.interp(b, cum, P[:, 1]) - np.interp(a, cum, P[:, 1])
    h = np.maximum(np.hypot(tx, ty), 1e-9)
    return x, y, tx / h, ty / h, s, L


def envelopes(data, X, Y, TX, TY, chunk=50000):
    """The ALKIS envelope under each cross-section: (a, b, hits) with a <= 0 <= b the interval's offsets (m, left of the
    direction positive) or NaN, and the number of polygons the section touches at all (0 = no ALKIS here)."""
    import shapely
    N = len(X)
    A = np.full(N, np.nan); B = np.full(N, np.nan); hits = np.zeros(N, int)
    if data.tree is None:
        return A, B, hits
    for c0 in range(0, N, chunk):
        sl = slice(c0, min(N, c0 + chunk))
        x, y, tx, ty = X[sl], Y[sl], TX[sl], TY[sl]
        nx, ny = -ty, tx
        rays = shapely.linestrings(np.stack([np.column_stack([x - RAY * nx, y - RAY * ny]),
                                              np.column_stack([x + RAY * nx, y + RAY * ny])], axis=1))
        ri, pi = data.tree.query(rays, predicate='intersects')
        hits[sl] = np.bincount(ri, minlength=len(x))
        if not len(ri):
            continue
        parts, owner = shapely.get_parts(shapely.intersection(rays[ri], data.polys[pi]), return_index=True)
        line = shapely.get_type_id(parts) == 1
        parts, owner = parts[line], owner[line]
        if not len(parts):
            continue
        p0 = shapely.get_coordinates(shapely.get_point(parts, 0)); p1 = shapely.get_coordinates(shapely.get_point(parts, -1))
        o = ri[owner]
        u0 = (p0[:, 0] - x[o]) * nx[o] + (p0[:, 1] - y[o]) * ny[o]
        u1 = (p1[:, 0] - x[o]) * nx[o] + (p1[:, 1] - y[o]) * ny[o]
        lo, hi = np.minimum(u0, u1), np.maximum(u0, u1)
        order = np.lexsort((lo, o))
        o, lo, hi = o[order].tolist(), lo[order].tolist(), hi[order].tolist()
        # the pieces of one section (touching polygons) merge; the interval holding the centre line is the envelope
        q = 0
        while q < len(o):
            r = q
            while r < len(o) and o[r] == o[q]:
                r += 1
            best, cl, ch = None, lo[q], hi[q]
            for t in range(q + 1, r + 1):
                if t < r and lo[t] <= ch + GAP:
                    ch = max(ch, hi[t])
                    continue
                d = 0. if cl <= 0. <= ch else min(abs(cl), abs(ch))
                if d <= CENTRE_TOL and (best is None or d < best[0]):
                    best = (d, cl, ch)
                if t < r:
                    cl, ch = lo[t], hi[t]
            if best is not None:
                A[c0 + o[q]], B[c0 + o[q]] = min(best[1], -.1), max(best[2], .1)
            q = r
    return A, B, hits


# =========================================================================================================
# medians
# =========================================================================================================

def bump(u, rel, lo, hi, centre_lo, centre_hi):
    """The raised strip in a relief profile (relief against the 12 m running median): (u_a, u_b, height) of the strip whose
    centre lies in (centre_lo, centre_hi) and that lies fully inside (lo + MEDIAN_EDGE, hi - MEDIAN_EDGE), else None.
    Method C of the study: starts at >= MEDIAN_SEED, extends while >= MEDIAN_KEEP, measured width = extent + 1 m."""
    best = None
    i, n = 0, len(u)
    while i < n:
        if rel[i] >= MEDIAN_SEED and lo + MEDIAN_EDGE < u[i] < hi - MEDIAN_EDGE:
            j = i
            while j + 1 < n and rel[j + 1] >= MEDIAN_KEEP:
                j += 1
            w = u[j] - u[i] + 1.
            top = float(np.nanmax(rel[i:j + 1]))
            mid = (u[i] + u[j]) / 2
            if (MEDIAN_WIDTH[0] <= w <= MEDIAN_WIDTH[1] and MEDIAN_MIN <= top <= MEDIAN_MAX and u[j] < hi - MEDIAN_EDGE
                    and centre_lo < mid < centre_hi):
                c = abs(mid - (centre_lo + centre_hi) / 2)
                if best is None or c < best[0]:
                    best = (c, float(u[i]), float(u[j]), top)
            i = j + 1
        else:
            i += 1
    return None if best is None else best[1:]


def running_relief(profile):
    """profile (S, U) heights -> relief against the running median of RUN_WINDOW samples (NaN-aware, shrinking at the ends)"""
    from numpy.lib.stride_tricks import sliding_window_view
    S, U = profile.shape
    pad = RUN_WINDOW // 2
    padded = np.pad(profile, ((0, 0), (pad, pad)), constant_values=np.nan)
    out = np.empty_like(profile)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for c0 in range(0, S, 1000):
            win = sliding_window_view(padded[c0:c0 + 1000], RUN_WINDOW, axis=1)
            out[c0:c0 + 1000] = profile[c0:c0 + 1000] - np.nanmedian(win, axis=2)
    return out


def medians(height, X, Y, TX, TY, side, dist, A, B):
    """For cross-sections with an opposite carriageway on `side` (+1 left, -1 right) at `dist` m: the median strip between
    the two centre lines, as (near, far, relief): offsets from the way's centre line towards the partner (m, > 0), or NaN.
    A, B: the envelope (offsets, left positive). height(xy (N, 2)) -> heights, NaN where unknown (the DGM1).
    The cross-profile is the mean of the profiles at ALONG m along the road; the strip is found by `bump`."""
    N = len(X)
    near = np.full(N, np.nan); far = np.full(N, np.nan); top = np.full(N, np.nan)
    order = np.lexsort((np.floor(Y / 1000.), np.floor(X / 1000.)))      # one area at a time: the DTM reads whole tiles
    for c0 in range(0, N, 300):
        ix = order[c0:c0 + 300]
        x, y, tx, ty, sd = X[ix], Y[ix], TX[ix], TY[ix], side[ix]
        lo = np.where(sd > 0, A[ix], -B[ix])                           # the envelope in the direction towards the partner
        hi = np.where(sd > 0, B[ix], -A[ix])
        u = np.arange(float(lo.min()) - PROFILE_BACK, float(hi.max()) + PROFILE_BACK + 1e-9, PROFILE_STEP)
        nx, ny = -ty * sd, tx * sd
        px = x[:, None, None] + nx[:, None, None] * u[None, :, None] + tx[:, None, None] * ALONG[None, None, :]
        py = y[:, None, None] + ny[:, None, None] * u[None, :, None] + ty[:, None, None] * ALONG[None, None, :]
        h = np.asarray(height(np.column_stack([px.ravel(), py.ravel()])), float).reshape(len(ix), len(u), len(ALONG))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prof = np.nanmean(h, axis=2)
        rel = running_relief(prof)
        for q, k in enumerate(ix.tolist()):
            r = bump(u, rel[q], float(lo[q]), float(hi[q]), 0., float(dist[k]))
            if r is not None:
                near[k], far[k], top[k] = r
    return near, far, top


# =========================================================================================================
# the refinement
# =========================================================================================================

def _median_filter(v, k=FILTER):
    """NaN-aware running median over k samples (the window shrinks at the ends); NaN where fewer than 2 samples are valid"""
    n = len(v)
    out = np.full(n, np.nan)
    h = k // 2
    for i in range(n):
        w = v[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 2 or (len(w) == 1 and n <= 2):
            out[i] = float(np.median(w))
    return out


def _fill(v):
    """NaN gaps filled from the nearest valid sample"""
    ok = np.flatnonzero(np.isfinite(v))
    if not len(ok) or len(ok) == len(v):
        return v
    idx = np.arange(len(v))
    return v[ok[np.abs(idx[:, None] - ok[None, :]).argmin(axis=1)]]


def plan_pieces(s, W, N, L, tol=PIECE_TOL, minimum=PIECE_MIN, E=None, etol=EDGE_TOL):
    """Cut a way into pieces of equal lanes and width (within tol): [(s0, s1, width, lanes)]. s, W, N per sample (no NaN).
    E (n, 2): the paved extent on each side of the OSM line per sample; a piece then also holds each edge within etol and
    carries its median extents as two more items (s0, s1, width, lanes, left, right)."""
    n = len(s)
    runs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and N[j + 1] == N[i] and abs(W[j + 1] - float(np.median(W[i:j + 1]))) <= tol and (
                E is None or all(abs(E[j + 1, c] - float(np.median(E[i:j + 1, c]))) <= etol for c in (0, 1))):
            j += 1
        runs.append((i, j))
        i = j + 1
    cuts = [0.] + [float((s[r[1]] + s[r[1] + 1]) / 2) for r in runs[:-1]] + [L]
    pieces = [[cuts[q], cuts[q + 1], float(np.median(W[r[0]:r[1] + 1])), int(N[r[0]])]
              + ([float(np.median(E[r[0]:r[1] + 1, 0])), float(np.median(E[r[0]:r[1] + 1, 1]))] if E is not None else [])
              for q, r in enumerate(runs)]
    limit = max(minimum, L / MAX_PIECES)

    def apart(p, q):
        return abs(p[2] - q[2]) + (abs(p[4] - q[4]) + abs(p[5] - q[5]) if E is not None else 0.)
    while len(pieces) > 1:                           # a short piece goes to the neighbour of the same lanes and nearest width
        lens = [p[1] - p[0] for p in pieces]
        k = int(np.argmin(lens))
        if lens[k] >= limit:
            break
        near = [q for q in (k - 1, k + 1) if 0 <= q < len(pieces)]
        q = min(near, key=lambda c: (pieces[c][3] != pieces[k][3], apart(pieces[c], pieces[k])))
        a, b = min(q, k), max(q, k)
        keep = pieces[q]
        pieces[a:b + 1] = [[pieces[a][0], pieces[b][1]] + keep[2:]]
    out = [pieces[0]]
    for p in pieces[1:]:
        if p[3] == out[-1][3] and abs(p[2] - out[-1][2]) <= tol / 2 and (
                E is None or (abs(p[4] - out[-1][4]) <= etol / 2 and abs(p[5] - out[-1][5]) <= etol / 2)):
            out[-1][1] = p[1]
        else:
            out.append(p)
    return [tuple(p) for p in out]


def cut_polyline(pts, s0, s1):
    """the part of a polyline between arc lengths s0 and s1 (interpolated ends)"""
    P = np.asarray(pts, float)
    cum = np.r_[0., np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    inner = (cum > s0 + 1e-6) & (cum < s1 - 1e-6)
    a = [float(np.interp(s0, cum, P[:, 0])), float(np.interp(s0, cum, P[:, 1]))]
    b = [float(np.interp(s1, cum, P[:, 0])), float(np.interp(s1, cum, P[:, 1]))]
    return np.vstack([a, P[inner], b])


def lane_tags(tags, n):
    """tags with lanes = n (one-way: per carriageway); the forward / backward split and turn lanes that no longer fit go"""
    out = dict(tags)
    out['lanes'] = str(int(n))
    for key in ('lanes:forward', 'lanes:backward', 'lanes:both_ways'):
        out.pop(key, None)
    for key in ('turn:lanes', 'turn:lanes:forward', 'turn:lanes:backward'):
        v = out.get(key)
        if v is not None and (key != 'turn:lanes' or len(str(v).split('|')) != n):
            out.pop(key, None)
    return out


def _num(v):
    try:
        return float(str(v).split(';')[0].replace(' m', '').replace(',', '.'))
    except (TypeError, ValueError):
        return None


def _lanes_tagged(tags):
    v = _num(tags.get('lanes'))
    return int(v) if v is not None and v >= 1 else None


def refine(drafts, frame, travel, shoulder, oneway, height=None, data=None, emit=None):
    """Apply the width prior and the medians to the drafts of fast_sources._prepare (one per segment between shared nodes):
    {'way', 'ord', 'tags', 'pts', 'nodes': (n0, n1), 'width'} with pts in the frame. Returns (drafts, islands, stats):
    the drafts cut into pieces of the official width / lanes (tags copied), islands [ring (N, 2) in the frame], counts.
    travel(tags) / shoulder(tags) / oneway(tags) are fast_roads.travel, fast_sources.shoulder and .oneway;
    height(xy) -> DGM1 heights (NaN unknown) switches the medians on."""
    import shapely
    stats = dict.fromkeys(('ways', 'sampled', 'valid', 'alkis_absent', 'dlm_ways', 'widened', 'narrowed', 'lanes_raised',
                           'tag_kept', 'pair_samples', 'median_samples', 'medians', 'median_m', 'pieces_added', 'plaza', 'sided',
                           'hard_pieces'), 0)
    del REPORT[:]
    if not drafts:
        return drafts, [], stats
    if data is None:
        xy = np.concatenate([np.asarray(d['pts'], float) for d in drafts])
        data = Data.load(frame, (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
    stats['tiles'] = data.tiles
    if data.tree is None and data.axis_tree is None:
        return drafts, [], stats
    degree = {}                                     # drafts meeting at a node: three or more are a junction
    for d in drafts:
        for nd in d['nodes']:
            degree[nd] = degree.get(nd, 0) + 1
    cand = []
    for k, d in enumerate(drafts):
        t = d['tags']
        if (t.get('highway', '').replace('_link', '') not in SWEPT or t.get('junction') in ('roundabout', 'circular')
                or t.get('bridge') not in (None, 'no') or t.get('tunnel') not in (None, 'no') or str(t.get('layer', '0')) not in ('0', '')):
            continue
        cand.append(k)
    stats['ways'] = len(cand)
    if not cand:
        return drafts, [], stats
    smp = [way_samples(drafts[k]['pts']) for k in cand]
    X, Y, TX, TY, S = (np.concatenate([p[i] for p in smp]) for i in range(5))
    LL = np.concatenate([np.full(len(p[0]), p[5]) for p in smp])
    OWN = np.concatenate([np.full(len(p[0]), k) for p, k in zip(smp, cand)])
    N = len(X)
    stats['sampled'] = int(N)
    NX, NY = -TY, TX
    A, B, hits = envelopes(data, X, Y, TX, TY)
    slot = np.searchsorted(np.asarray(cand), OWN)    # the junction mouths: no samples within MOUTH m of such a node
    j0 = np.array([degree.get(drafts[k]['nodes'][0], 0) > 2 for k in cand])
    j1 = np.array([degree.get(drafts[k]['nodes'][1], 0) > 2 for k in cand])
    reach = np.minimum(MOUTH, MOUTH_SHARE * LL)      # a short arm keeps its middle cross-sections
    mouth = (j0[slot] & (S < reach)) | (j1[slot] & (LL - S < reach))
    valid = np.isfinite(A) & ~mouth
    # neighbours: ways running parallel inside the envelope, and the opposite carriageway of a one-way pair
    G = np.empty(len(drafts), dtype=object)
    G[:] = [shapely.LineString(d['pts']) for d in drafts]
    ltree = shapely.STRtree(G)
    way_of = np.array([d['way'] for d in drafts])
    tr = np.array([travel(d['tags']) for d in drafts])
    swept = np.array([d['tags'].get('highway', '').replace('_link', '') in SWEPT for d in drafts])
    ring = np.array([d['tags'].get('junction') in ('roundabout', 'circular') for d in drafts])
    dwidth = np.array([float(d['width']) for d in drafts])
    uL = np.full(N, np.inf)                         # the room left of the centre line (> 0) and right of it (< 0) the neighbours leave
    uR = np.full(N, -np.inf)
    partner = np.full(N, np.nan)                    # offset of the opposite carriageway
    vi = np.flatnonzero(valid)
    for c0 in range(0, len(vi), 80000):
        v = vi[c0:c0 + 80000]
        rays = shapely.linestrings(np.stack([np.column_stack([X[v] - RAY * NX[v], Y[v] - RAY * NY[v]]),
                                              np.column_stack([X[v] + RAY * NX[v], Y[v] + RAY * NY[v]])], axis=1))
        ri, wi = ltree.query(rays, predicate='intersects')
        keep = way_of[wi] != way_of[OWN[v][ri]]
        ri, wi = ri[keep], wi[keep]
        if not len(ri):
            continue
        parts, owner = shapely.get_parts(shapely.intersection(rays[ri], G[wi]), return_index=True)
        ok = shapely.get_type_id(parts) == 0
        parts, owner = parts[ok], owner[ok]
        if not len(parts):
            continue
        wj, si = wi[owner], v[ri[owner]]
        pc = shapely.get_coordinates(parts)
        u = (pc[:, 0] - X[si]) * NX[si] + (pc[:, 1] - Y[si]) * NY[si]
        loc = shapely.line_locate_point(G[wj], parts)
        ln = shapely.length(G[wj])
        p0 = shapely.get_coordinates(shapely.line_interpolate_point(G[wj], np.clip(loc - 1.5, 0, ln)))
        p1 = shapely.get_coordinates(shapely.line_interpolate_point(G[wj], np.clip(loc + 1.5, 0, ln)))
        dx, dy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        cosang = (dx * TX[si] + dy * TY[si]) / np.maximum(np.hypot(dx, dy), 1e-9)
        inside = (u >= A[si] - .3) & (u <= B[si] + .3) & (np.abs(u) > 1.0) & (np.abs(cosang) >= PARALLEL) & ~ring[wj]
        # the room a neighbour leaves: half the way between the two centre lines where it is a swept road with an envelope of
        # its own to claim its half; a street the rule does not touch (residential, service) keeps its own default half width
        # (bundle8: with the midway line for it too, a tertiary beside a residential street left 2 m of the lane unpaved,
        # seen in testing, once the two sides were placed separately)
        room = np.where(swept[wj], np.abs(u) / 2, np.maximum(np.abs(u) / 2, np.abs(u) - dwidth[wj] / 2))
        for q in np.flatnonzero(inside).tolist():
            k = si[q]
            if u[q] > 0:
                uL[k] = min(uL[k], room[q])
            else:
                uR[k] = max(uR[k], -room[q])
        own = tr[OWN[si]]
        opp = inside & (own != 0) & (tr[wj] != 0) & (own * tr[wj] * cosang < 0) & swept[wj] & (np.abs(u) <= MAX_PARTNER)
        for q in np.flatnonzero(opp).tolist():
            k = si[q]
            if np.isnan(partner[k]) or abs(u[q]) < abs(partner[k]):
                partner[k] = u[q]
    stats['pair_samples'] = int(np.isfinite(partner).sum())
    # medians from the DGM1, between the centre lines of a pair
    near = np.full(N, np.nan); far = np.full(N, np.nan)
    pi_ = np.flatnonzero(np.isfinite(partner))
    if height is not None and len(pi_):
        nr, fr, _ = medians(height, X[pi_], Y[pi_], TX[pi_], TY[pi_], np.sign(partner[pi_]), np.abs(partner[pi_]), A[pi_], B[pi_])
        near[pi_], far[pi_] = nr, fr
        stats['median_samples'] = int(np.isfinite(nr).sum())
    pside = np.sign(partner)
    # the width of a cross-section: what is left of the envelope on both sides of the centre line
    W = np.full(N, np.nan)
    left = np.where(np.isfinite(uL), np.minimum(B, uL), B)
    right = np.where(np.isfinite(uR), np.minimum(-A, -uR), -A)
    left = np.where(np.isfinite(near) & (pside > 0), near, left)
    right = np.where(np.isfinite(near) & (pside < 0), near, right)
    LEFT = np.full(N, np.nan); RIGHT = np.full(N, np.nan)       # the paved extent on each side of the OSM line (node order)
    LEFT[vi], RIGHT[vi] = np.maximum(left[vi], 0.), np.maximum(right[vi], 0.)
    W[vi] = LEFT[vi] + RIGHT[vi]
    # Basis-DLM Fahrbahnbreite where the section meets no ALKIS polygon at all
    absent = (hits == 0) & ~mouth
    stats['alkis_absent'] = int(absent.sum())
    dlm = np.full(N, np.nan)
    if data.axis_tree is not None and absent.any():
        ai = np.flatnonzero(absent)
        pt = shapely.points(X[ai], Y[ai])
        qi, axi = data.axis_tree.query(pt, predicate='dwithin', distance=6.)
        for q, a in zip(qi.tolist(), axi.tolist()):
            k = ai[q]
            g = data.axis_geoms[a]
            loc, ln = shapely.line_locate_point(g, pt[q]), shapely.length(g)
            c0_ = shapely.get_coordinates(shapely.line_interpolate_point(g, max(0., loc - 2.)))[0]
            c1_ = shapely.get_coordinates(shapely.line_interpolate_point(g, min(ln, loc + 2.)))[0]
            vx, vy = c1_ - c0_
            if abs(vx * TX[k] + vy * TY[k]) / max(math.hypot(vx, vy), 1e-9) >= PARALLEL and (np.isnan(dlm[k]) or data.axis_width[a] > dlm[k]):
                dlm[k] = data.axis_width[a]
    idx = np.argsort(OWN, kind='stable')
    so = OWN[idx]
    b0 = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
    order = {int(so[s0]): idx[s0:e0] for s0, e0 in zip(b0.tolist(), np.r_[b0[1:], len(idx)].tolist())}
    out = []
    for k, d in enumerate(drafts):
        ix = order.get(k)
        res = _decide(d, ix, W, dlm, S, LL, shoulder, oneway, stats, LEFT, RIGHT, travel(d['tags'])) if ix is not None else None
        if res is None:
            out.append(d)
            continue
        out.extend(res)
        stats['pieces_added'] += len(res) - 1
    islands = _median_polygons(cand, order, X, Y, TX, TY, near, far, pside, mouth, stats)
    if emit:
        emit('official: ' + ', '.join(f'{k} {v}' for k, v in stats.items()))
    return out, islands, stats


def _shoulders_travel(tags, shoulder, trav):
    """(left, right) class shoulders in the direction of travel (tagged ones are given in node order)"""
    sl, sr = shoulder(tags)
    if tags.get('shoulder:left:width') is not None and tags.get('shoulder:right:width') is not None and trav < 0:
        sl, sr = sr, sl
    return float(sl), float(sr)


def _decide(d, ix, W, dlm, S, LL, shoulder, oneway, stats, LEFT=None, RIGHT=None, trav=0):
    """The pieces of one draft with their official width and lanes, or None when it stays as it is. With LEFT / RIGHT (the
    paved extent on each side of the OSM line, node order) a piece also gets its edges ('edges': (left, right) of the node
    order) and, on a one-way motorway / trunk (a link: the right side may carry the hard shoulder), its lane width."""
    tags = d['tags']
    w0 = float(d['width'])
    sl, sr = shoulder(tags)
    shoulders = float(sl + sr)
    one = bool(oneway(tags))
    Wk = W[ix].copy()
    Lk = LEFT[ix].copy() if LEFT is not None else None
    Rk = RIGHT[ix].copy() if RIGHT is not None else None
    use_dlm = False
    if not np.isfinite(Wk).any() and np.isfinite(dlm[ix]).any():
        Wk, use_dlm, Lk, Rk = dlm[ix] + shoulders, True, None, None
    if not np.isfinite(Wk).any():
        return None
    bad = Wk > min(MAX_WIDTH, PLAZA * max(w0, 3.))   # an envelope far wider than the tags imply is a plaza, not this road
    if bad.any():
        stats['plaza'] += int(bad.sum())
        Wk[bad] = np.nan
        if Lk is not None:
            Lk[bad] = Rk[bad] = np.nan
    if not np.isfinite(Wk).any():
        return None
    stats['valid'] += int(np.isfinite(Wk).sum())
    Ws = _fill(_median_filter(Wk))
    if not np.isfinite(Ws).all():
        Ws = _fill(Wk)
    Ls = Rs = None
    if Lk is not None:
        Ls, Rs = _fill(_median_filter(Lk)), _fill(_median_filter(Rk))
        if not (np.isfinite(Ls).all() and np.isfinite(Rs).all()):
            Ls = Rs = None
    n_tag = _lanes_tagged(tags)
    n_osm = n_tag if n_tag else (2 if (tags.get('highway') == 'motorway' or not one) else 1)
    explicit = tags.get('lanes:forward') is not None or tags.get('lanes:backward') is not None
    highway = tags.get('highway', '')
    base = highway.replace('_link', '')
    link = highway.endswith('_link')
    changeable = base not in ('motorway', 'trunk') and not explicit and not use_dlm     # the exits' aux lanes read these
    hard = Ls is not None and one and base in ('motorway', 'trunk')    # per-side caps; the right side may carry a hard shoulder
    tag_w = _num(tags.get('width'))
    m = len(ix)
    Wf = np.empty(m); Nf = np.full(m, n_osm)
    Ef = np.full((m, 2), np.nan) if Ls is not None else None
    Qf = np.full(m, np.nan)                          # lane width (hard classes)
    slt, srt = _shoulders_travel(tags, shoulder, trav)
    lw_nom = min(LANE_CAP, max(LANE_MIN, (w0 - shoulders) / n_osm))
    kept = 0
    for q in range(m):
        w = float(Ws[q])
        n = n_osm
        keep_tag = tag_w is not None and abs(w - tag_w) <= TAG_WIDTH_TOL * tag_w
        if keep_tag:
            w, kept = tag_w, kept + 1                # an OSM width within reach of the official one stands
        elif hard:
            # a one-way motorway / trunk: each side keeps what the envelope holds up to its cap; the right side of a link
            # (and the motorway's) carries the unmarked hard shoulder, 3.0-4.3 m beyond the edge line (aerials)
            lt, rt = (float(Rs[q]), float(Ls[q])) if trav < 0 else (float(Ls[q]), float(Rs[q]))
            half = n * lw_nom / 2
            lt = min(lt, half + slt + EDGE_STRIP)
            # (a link shorter than SHORT_LINK lies inside its junction: no hard shoulder there; one stood 6 cm off in testing)
            rt = min(rt, half + (HARD_SHOULDER if link and float(LL[ix[0]]) >= SHORT_LINK else srt) + EDGE_STRIP)
            lt, rt = max(lt, 1.), max(rt, 1.)
            short = max(KEEP * w0, n * LANE_MIN + shoulders) - (lt + rt)
            if short > 0:
                lt, rt = lt + short / 2, rt + short / 2
            w = lt + rt
            Qf[q] = min(lw_nom, max(LANE_MIN, (w - shoulders) / n))
            Ef[q] = (rt, lt) if trav < 0 else (lt, rt)
            Wf[q], Nf[q] = w, n
            continue
        else:
            n_new = int(math.floor((w - shoulders + LANE_BIAS) / LANE_STEP))
            # lanes only on one-way carriageways (J09: a pair tagged 1 lane each, really 2): a two-way street's
            # ALKIS envelope usually holds parking bays, cycle lanes and turn pockets, so 12 m there is 2 lanes plus
            # parking far more often than 4 lanes (review, 1 Nov); its extra width stays at the edges (capped below)
            if changeable and one and n_new >= n_osm + 1:
                n = min(n_new, 6)
            w = min(w, n * LANE_CAP + shoulders + EDGE_STRIP)
            w = max(w, KEEP * w0, n * LANE_MIN + shoulders)
        Wf[q], Nf[q] = w, n
        if Ef is not None:
            # the extents keep their position against the OSM line: scaled to the width where the cap cut it, widened evenly
            # where a floor raised it (the centre of the pavement stays where the envelope has it)
            l, r = float(Ls[q]), float(Rs[q])
            tot = l + r
            if tot > 1e-6 and w <= tot:
                l, r = l * w / tot, r * w / tot
            else:
                l, r = l + (w - tot) / 2, r + (w - tot) / 2
            # the OSM line is rarely more than 2 m off the middle of a street's pavement: a larger offset is parking, a cycle strip
            # or a junction's flare on one side (in testing: 10.4 m left / 3.1 m right of a secondary road left a hole at the
            # plate), so the pavement keeps its width and moves at most MAX_SHIFT towards it
            c = max(-MAX_SHIFT, min(MAX_SHIFT, (l - r) / 2))
            l, r = (l + r) / 2 + c, (l + r) / 2 - c
            # a side is never narrower than a lane's half and a margin (a pavement edge 1.2 m from the line is the line's offset)
            if min(l, r) < MIN_SIDE <= (l + r) / 2:
                l, r = (MIN_SIDE, l + r - MIN_SIDE) if l < r else (l + r - MIN_SIDE, MIN_SIDE)
            Ef[q] = (l, r)
    stats['tag_kept'] += int(kept == len(ix))
    # the pieces also follow the pavement's position where the road is drawn by the exporter's own strokes as a motorway or trunk
    # (an edge notch is worth a joint there); other classes are cut by width and lanes as before (every joint of a street the
    # generator draws is a seam: one got a hole at a plate with the finer cut), their pieces still carry their edges
    pieces = plan_pieces(S[ix], Wf, Nf, float(LL[ix[0]]), E=Ef, etol=EDGE_TOL if base in ('motorway', 'trunk') else 99.)
    # the extents a section without an official edge would have (node order), to tell a piece that changes nothing
    lanes0 = w0 - shoulders
    d_l, d_r = (lanes0 / 2 + srt, lanes0 / 2 + slt) if trav < 0 else (lanes0 / 2 + slt, lanes0 / 2 + srt)
    if all(abs(p[2] - w0) <= .3 and p[3] == n_osm and (Ef is None or (abs(p[4] - d_l) <= .3 and abs(p[5] - d_r) <= .3))
           for p in pieces):
        return None
    stats['widened'] += int(any(p[2] > w0 + .3 for p in pieces))
    stats['narrowed'] += int(all(p[2] < w0 - .3 for p in pieces))
    stats['lanes_raised'] += int(any(p[3] != n_osm for p in pieces))
    stats['dlm_ways'] += int(use_dlm)
    way = int(d['way'])
    sv = S[ix]
    lane_of = []
    for p in pieces:
        sel = np.isfinite(Qf) & (sv >= p[0]) & (sv <= p[1])
        lane_of.append(float(np.median(Qf[sel])) if sel.any() else None)
    if Ef is not None:
        stats['sided'] += len(pieces)
        # a link piece whose right side (of the travel) holds more than a lane's half and 2 m: the hard shoulder
        stats['hard_pieces'] += sum(1 for p, ln in zip(pieces, lane_of)
                                    if ln is not None and link and (p[4] if trav < 0 else p[5]) > lw_nom * n_osm / 2 + 2.)
    REPORT.append([way, int(d.get('ord', 0)), round(w0, 2), n_osm,
                   [[round(p[0], 1), round(p[1], 1), round(p[2], 2), p[3]] + ([round(p[4], 2), round(p[5], 2)] if Ef is not None else [])
                    for p in pieces]])
    chain = [d['nodes'][0]] + [-((way * 1024 + int(d.get('ord', 0))) * 16 + q + 1) for q in range(len(pieces) - 1)] + [d['nodes'][1]]
    out = []
    for q, p in enumerate(pieces):
        n = p[3]
        extra = {}
        if Ef is not None:
            extra['edges'] = (round(float(p[4]), 2), round(float(p[5]), 2))
            if lane_of[q] is not None:
                extra['lane'] = round(lane_of[q], 2)
        out.append(dict(d, tags=dict(tags) if n == n_osm else lane_tags(tags, n), pts=cut_polyline(d['pts'], p[0], p[1]),
                        nodes=(chain[q], chain[q + 1]), width=float(round(p[2], 2)), official=True, **extra))
    return out


def _median_polygons(cand, order, X, Y, TX, TY, near, far, pside, mouth, stats):
    """Runs of cross-sections with a detected strip -> polygons in the frame (edges smoothed, ends capped half a step on)."""
    import shapely
    polys = []
    for k in cand:
        ix = order.get(k)
        if ix is None:
            continue
        nr, fr = near[ix], far[ix]
        if not np.isfinite(nr).any():
            continue
        sd = np.where(np.isfinite(pside[ix]), pside[ix], 0.)
        lo = np.where(sd > 0, nr, -fr)               # strip edges as offsets from the centre line (left positive)
        hi = np.where(sd > 0, fr, -nr)
        det = np.isfinite(nr) & ~mouth[ix]
        det &= np.convolve(np.pad(det.astype(float), 1), np.ones(3), 'valid') >= 2     # a lone detection is noise
        q, n = 0, len(ix)
        while q < n:
            if not det[q]:
                q += 1
                continue
            r = q
            while r + 1 < n and det[r + 1]:
                r += 1
            if (r - q + 1) * STEP >= MEDIAN_RUN:
                sl = ix[q:r + 1]
                l_, h_ = _fill(_median_filter(lo[q:r + 1], 3)), _fill(_median_filter(hi[q:r + 1], 3))
                nx, ny = -TY[sl], TX[sl]
                a = np.array([-TX[sl[0]], -TY[sl[0]]]) * STEP / 2
                b = np.array([TX[sl[-1]], TY[sl[-1]]]) * STEP / 2
                left = np.column_stack([X[sl] + nx * l_, Y[sl] + ny * l_])
                right = np.column_stack([X[sl] + nx * h_, Y[sl] + ny * h_])
                left[0] += a; right[0] += a; left[-1] += b; right[-1] += b
                g = shapely.Polygon(np.vstack([left, right[::-1]]))
                if not g.is_valid:
                    g = shapely.make_valid(g)
                g = g.buffer(-MEDIAN_ROUND).buffer(MEDIAN_ROUND, quad_segs=4)
                if not g.is_empty:
                    polys.append(g)
            q = r + 1
    if not polys:
        return []
    merged = shapely.union_all(polys).buffer(.4).buffer(-.4)      # the pair's two ways find the same strip
    out = []
    for g in (merged.geoms if hasattr(merged, 'geoms') else [merged]):
        if g.geom_type != 'Polygon' or g.area < 20.:
            continue
        g = shapely.simplify(g, .1)
        out.append(np.asarray(g.exterior.coords)[:-1].round(3))
        stats['median_m'] += int(round(g.length / 2))
    stats['medians'] = len(out)
    return out
