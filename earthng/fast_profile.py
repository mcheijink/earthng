"""Road heights for the whole graph at once (fast generator, index time).

Per-segment profiles from a 24 m DEM gave 112 % ramps and 3.5 m pits on short pieces
(docs/PERFORMANCE.md). Here every road is sampled into stations (shared at graph
nodes) and the heights minimise

    sum w*m*(h - target)^2 + ALPHA * sum (dh)^2/l + BETA * sum (d2h)^2/l^3

(data, grade, change of grade), solved by Jacobi-preconditioned conjugate gradients. A node
has one height for every road that meets there, so tiles and junction mouths agree by
construction. Bridges pull up and real tunnels pull down only softly; their approaches become
ramps. A bridge spans the height of its abutments: it does not follow the river bed under
the span. Where it crosses a road in plan, the deck and the road under it share the
CLEARANCE: the deck rises and the lower road dips into a cutting. A 24 m DEM has no cuttings,
and the soft pull alone left Juri's overpasses 1.3 m over the ring road. Letting only the deck
rise made an 11 m hill of every short overpass, and a dual carriageway's two bridges (their
ribbons overlap, their roads are joined) lifted each other 800 m on the Tallinna ringtee.
tunnel=building_passage is an arch through a house, at ground level.
"""
import math
import os

import numpy as np

STEP = 20.0            # station spacing, m
ALPHA = 40.0 ** 2      # grade smoothing length 40 m
BETA = 50.0 ** 4       # grade-change smoothing length 50 m
BRIDGE_CLEAR, TUNNEL_DEPTH, SOFT = 6.5, 7.0, 0.5
# Grade weight on the first station gap out of a junction (3+ roads): roads leave it level, so
# roads that overlap near it in plan are at one height. With free grades a link and the main
# road diverged by 0.78 m over the main road's sidewalk, and junction polygons tilted 24 %.
JUNCTION_FLAT = 10.0
# Ground roads whose surfaces touch in plan (two carriageways of a divided road, a lane beside
# a street) are tied to one height: on Khreshchatyk the carriageways overlapped by 1 m with
# profiles up to 1.02 m apart.
COUPLE = 1000.0
# A deck over a road: 4.5 m headroom + the deepest girder (ROADS.deckDepth of a
# main road, 1.5 m), per layer. A street slab is 1 m and keeps a little more air.
CLEARANCE, CLEAR_WEIGHT, CLEAR_PASSES = 6.0, 1e4, 12
# A motorway or trunk under a bridge keeps its ground line: its stations within HOLD_REACH m of the
# crossing (along it) hold their target HOLD_WEIGHT times harder, so the deck and its ramps take the
# whole clearance. Shared, the clearance dug the motorway into cuttings of 2-6 m and 150-300 m under nearly
# every overpass of a Niedersachsen pack, which the DEM does not have: the generator drew trenches with
# sags of 300-750 m radius at 180-207 km/h (the motorway's minimum is 5,700 m).
HOLD_UNDER, HOLD_REACH, HOLD_WEIGHT = ('motorway', 'trunk'), 200., 1e3
# Reserve the possible sidewalks of the rendered deck; clearance is measured at its
# soffit over the whole lower carriageway, including skew crossings. Numerical slack
# stays above the 4.5 m headroom rather than silently accepting a shortfall from it.
DECK_OVERHANG, CLEAR_SLACK, CLEAR_TOL = 2., .1, .05
CLEAR_EXTRA = 6.
# Steepest grade the solve may make by class, where the ground itself is gentler. The clearance
# springs alone raised a deck over 50-80 m: Kyiv's Borshchahivska bridge 14 %, the Povitrianykh
# Syl overpass 11 %. An edge over its limit gets a stiffer grade weight and the solve runs again,
# so the rise spreads over a longer approach (real urban overpasses: 4-6 %).
# Motorways: RAA 2008 Tabelle 26 (2. Korrektur 2012), Hoechstlaengsneigung EKA 1 B 4.5 % (EKA 1 A 4.0 %).
GRADE_MAX = {'motorway': .045, 'trunk': .05, 'primary': .06, 'secondary': .06, 'tertiary': .07,
             'motorway_link': .07, 'trunk_link': .07, 'primary_link': .07, 'secondary_link': .07}
GRADE_DEFAULT, GRADE_PASSES, STIFF_MAX, CURVE_STIFF_MAX = .08, 8, 30., 1e4
# A limited triple starts at CURVE_START x BETA, scaled by its class's crest radius over 10,000 m: from
# BETA alone, 8 passes left 3 % of the motorway and trunk triples under their radius (0.8 % so)
CURVE_START = 30.
# Smallest crest and sag radius (m) by class, from the German design guides (the generator's
# ROADS.profile.classes has the same table and its sources): RAA 2008 EKA 1 B for motorways, RAL 2012
# EKL 1-4 for trunk to tertiary, RAA ramp group II for links. A station triple bending tighter gets a
# stiffer change-of-grade weight in the same passes as the grades (BETA alone is a 50 m smoothing
# length: a motorway deck over a street rose and fell within 100 m, crests of 400-700 m radius at
# 120-180 km/h). Other classes keep BETA alone.
# CG's preconditioner solves each segment's run of interior stations exactly, in blocks of at most CHAIN
# stations (a banded LDL^T): with the stiff change-of-grade weights of CURVE_MIN, a diagonal (Jacobi)
# preconditioner took CG to its 400-iteration cap in every pass (a Niedersachsen profile 13 s -> 537 s).
CHAIN = 48
# Lower bounds on station heights (fast_roads.bridge_floors: a deck over navigable water clears the channel,
# water level + clearance + structure depth). An active set like the clearance springs: a station under
# its floor gets a spring of FLOOR_WEIGHT to it, aiming higher by what it still lacks (up to FLOOR_EXTRA),
# and lets go when the solve holds it FLOOR_RELEASE m over the floor without it. The approaches then rise at
# their class's grade (GRADE_MAX stiffening); the 24 m DEM there is only a soft target.
FLOOR_WEIGHT, FLOOR_TOL, FLOOR_EXTRA, FLOOR_RELEASE = 1e4, .1, 2., .05
CURVE_MIN = {'motorway': (10000., 5700.), 'trunk': (8000., 4000.), 'primary': (6000., 3500.),
             'secondary': (5000., 3000.), 'tertiary': (3000., 2000.), 'motorway_link': (2500., 1000.),
             'trunk_link': (2500., 1000.), 'primary_link': (2500., 1000.), 'secondary_link': (1500., 800.),
             'tertiary_link': (1500., 800.)}

# The class rules above, as the German design guides give them, are the fork's: HOLD_UNDER (the motorway
# and trunk keep their line under a deck), the motorway grade of 4.5 % instead of 5 %, 8 grade passes instead
# of 4 and the CURVE_MIN radii. They are off unless EARTHNG_ROAD_RULES=1 (or set_road_rules(True)); off, the
# solve is upstream's: its own grades (GRADE_MAX at 5 % on motorways, 4 passes), no radii, no held line.
# The deck floors over water (FLOOR_*) only act when the exporter hands floors in.
_FORK_RULES = dict(HOLD_UNDER=HOLD_UNDER, GRADE_MOTORWAY=GRADE_MAX['motorway'], GRADE_PASSES=GRADE_PASSES,
                   CURVE_MIN=dict(CURVE_MIN))
ROAD_RULES = False


def set_road_rules(on):
    """Switch the fork's class rules on or off (module state read at the next solve); returns the flag."""
    global HOLD_UNDER, GRADE_PASSES, CURVE_MIN, ROAD_RULES, SIGNATURE
    ROAD_RULES = bool(on)
    HOLD_UNDER = _FORK_RULES['HOLD_UNDER'] if on else ()
    GRADE_MAX['motorway'] = _FORK_RULES['GRADE_MOTORWAY'] if on else .05
    GRADE_PASSES = _FORK_RULES['GRADE_PASSES'] if on else 4
    CURVE_MIN = dict(_FORK_RULES['CURVE_MIN']) if on else {}
    SIGNATURE = f'profile-rules:{"on" if RULES else "off"}:roads:{"on" if on else "off"}'
    return ROAD_RULES


# ---------------------------------------------------------------------------------------------------
# DEM rules (EARTHNG_DEM_RULES=1; off by default until validated, ALTITUDE-REFERENCE.md "DEM
# contamination"). The coarse DEM (Terrarium z12, an effective footprint of sigma about 60 m here)
# measured against LGLN DGM1 along road centrelines: forest beside the road lifts it by about
# 10 m per unit forest share within 150 m (motorway humps of 6-12 m seen in testing were forest),
# roads under a deck carry a hump of +1.4 m (0-20 m) falling to +0.4 m at 100-200 m, and the approaches of
# a road's own bridge sit 1 m low (the narrow ramp embankments are below the DEM's resolution). Rules:
# 1. a ground line is solved first from the DEM minus the forest bias, with samples near structures
#    down-weighted (STRUCT_W at STRUCT_NEAR m, full weight from STRUCT_FAR m), decks and tunnels
#    masked, Huber reweighting (HUBER m, IRLS passes) and the class's own smoothing length
#    (GROUND_LENGTH: a masked stretch is bridged by that curve, under GRADE_MAX and CURVE_MIN below);
# 2. the profile solve then takes the ground line as its target instead of the raw DEM; near structures
#    its data weight is lowered to RAMP_W so the decks' clearance and grade rules shape the ramps.
RULES = os.environ.get('EARTHNG_DEM_RULES', '0').strip() not in ('', '0', 'off', 'no', 'false')
SIGNATURE = f'profile-rules:{"on" if RULES else "off"}'
set_road_rules(os.environ.get('EARTHNG_ROAD_RULES', '0').strip().lower() not in ('', '0', 'off', 'no', 'false'))
STRUCT_NEAR, STRUCT_FAR, STRUCT_W = 80., 250., .05
FOREST_BIAS, FOREST_SIGMA, COVER_CELL = 10., 100., 20.
HUBER, IRLS = 1.5, 3
RAMP_W = .2
GROUND_LENGTH = {'motorway': 130., 'trunk': 130., 'primary': 110., 'secondary': 90., 'tertiary': 70.,
                 'motorway_link': 60., 'trunk_link': 60.}
GROUND_LENGTH_DEFAULT = 50.
# ---------------------------------------------------------------------------------------------------
# A national 1 m DTM (dtm.py, EARTHNG_DTM=1; ALTITUDE-REFERENCE.md "National DTM"): solve(reference=)
# blends each ground station's data term between the DTM (its share: 1 inside the coverage, 0 outside)
# and the coarse DEM's rule ground line. The same solve, other inputs and weights:
# * FINE_W: the DTM's data weight (the coarse DEM's is 1), a smoothing length of about 8 m on the grade
#   and 22 m on the change of grade (ALPHA, BETA unchanged): the measured profile is already smooth (1 cm
#   plane residuals on asphalt), so the class rules (GRADE_MAX, CURVE_MIN) only act where it breaks them;
# * decks and tunnels never take it (the DTM has no bridges, and shows the hill over a tunnel); a ground
#   station within its road's half width + UNDER_MARGIN m of another structure (a deck over it) takes the
#   DTM line bridged from the clear stations either side (the DTM bulges by metres under wide decks: 3.6 m
#   in testing), and holds it like HOLD_UNDER, so the deck takes the clearance;
# * the stations within OWN_FAR m of their own bridge's deck take the clear ramp's line extended and only
#   OWN_W of the weight, so the deck's rule chain (clearance, floors, grade) meets the DTM embankment.
FINE_W = 25.
UNDER_MARGIN = 6.
OWN_FAR, OWN_W = 15., .05
# Land cover read from the OSM source tiles for the rules (read_cover): forest and wood areas, and the
# structures that are not roads (railway, waterway, path and man_made bridges)
FOREST_TAGS = {('landuse', 'forest'), ('natural', 'wood')}
ROAD_CLASSES = ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential',
                'living_street', 'service', 'road', 'raceway', 'motorway_link', 'trunk_link', 'primary_link',
                'secondary_link', 'tertiary_link')


def _blur(grid, cells):
    """Gaussian of `cells` sigma (three box passes per axis), zero outside the grid."""
    w = max(1, int(round(math.sqrt(4 * cells * cells + 1)))) | 1
    h = w // 2
    out = grid.astype(np.float64)
    for axis in (0, 1):
        for _ in range(3):
            a = np.moveaxis(out, axis, -1)
            c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)
            n = a.shape[-1]
            hi = np.clip(np.arange(n) + h + 1, 0, n); lo = np.clip(np.arange(n) - h, 0, n)
            out = np.moveaxis((c[..., hi] - c[..., lo]) / w, -1, axis)
    return out


class ShareGrid:
    """A blurred cover share on the global COVER_CELL lattice (cell centres at (i + .5) x COVER_CELL), stored to
    1/65535 and read bilinearly. Its value is a function of the point alone, so the road profiles (the DEM
    rules' forest bias) and every cell of the export's ground grid read the same share, with no steps
    between neighbouring cells (ALTITUDE-REFERENCE.md "Road against ground")."""

    def __init__(self, i0, j0, share):
        self.i0, self.j0 = int(i0), int(j0)
        self.share = np.round(np.clip(np.asarray(share, float), 0., 1.) * 65535.) / 65535.

    def covers(self, pos):
        pos = np.asarray(pos, float).reshape(-1, 2)
        if not len(pos):
            return True
        ny, nx = self.share.shape
        lo, hi = pos.min(0) / COVER_CELL - .5, pos.max(0) / COVER_CELL - .5
        return bool(lo[0] >= self.i0 and lo[1] >= self.j0 and hi[0] <= self.i0 + nx - 1 and hi[1] <= self.j0 + ny - 1)

    def at(self, pos):
        pos = np.asarray(pos, float).reshape(-1, 2)
        ny, nx = self.share.shape
        fx, fy = pos[:, 0] / COVER_CELL - .5 - self.i0, pos[:, 1] / COVER_CELL - .5 - self.j0
        ix = np.clip(np.floor(fx).astype(np.int64), 0, max(0, nx - 2))
        iy = np.clip(np.floor(fy).astype(np.int64), 0, max(0, ny - 2))
        tx, ty = np.clip(fx - ix, 0., 1.), np.clip(fy - iy, 0., 1.)
        s = self.share
        ix1, iy1 = np.minimum(ix + 1, nx - 1), np.minimum(iy + 1, ny - 1)
        v = (s[iy, ix] * (1 - tx) + s[iy, ix1] * tx) * (1 - ty) + (s[iy1, ix] * (1 - tx) + s[iy1, ix1] * tx) * ty
        return np.clip(v, 0., 1.)

    def save(self, path):
        np.savez_compressed(path, i0=self.i0, j0=self.j0, cell=COVER_CELL,
                            share=np.round(self.share * 65535.).astype(np.uint16))

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            if float(d['cell']) != COVER_CELL:
                raise ValueError(f'{path}: cover cell {float(d["cell"])} m, expected {COVER_CELL} m')
            return cls(int(d['i0']), int(d['j0']), d['share'].astype(float) / 65535.)


class Cover:
    """Forest polygons and non-road structure lines in the region frame, for the DEM rules."""

    def __init__(self, forest=(), structures=()):
        self.forest = list(forest)
        self.structures = [np.asarray(line, float).reshape(-1, 2) for line in structures]
        self.grid = None            # a ShareGrid over the whole region (fast_sources), read by forest_share

    def share_grid(self, bounds):
        """The forest share (Gaussian, FOREST_SIGMA) on the global COVER_CELL lattice over bounds (x0, y0, x1, y1)."""
        import shapely
        x0, y0, x1, y1 = bounds
        pad = 3 * FOREST_SIGMA + COVER_CELL
        i0, j0 = int(math.floor((x0 - pad) / COVER_CELL)), int(math.floor((y0 - pad) / COVER_CELL))
        nx = int(math.ceil((x1 + pad) / COVER_CELL)) - i0 + 1
        ny = int(math.ceil((y1 + pad) / COVER_CELL)) - j0 + 1
        grid = np.zeros((ny, nx), np.float32)
        for poly in self.forest:
            b = poly.bounds
            a0, a1 = max(0, int(math.floor(b[0] / COVER_CELL)) - i0), min(nx, int(math.ceil(b[2] / COVER_CELL)) - i0 + 1)
            c0, c1 = max(0, int(math.floor(b[1] / COVER_CELL)) - j0), min(ny, int(math.ceil(b[3] / COVER_CELL)) - j0 + 1)
            if a0 >= a1 or c0 >= c1:
                continue
            gx, gy = np.meshgrid((i0 + np.arange(a0, a1) + .5) * COVER_CELL, (j0 + np.arange(c0, c1) + .5) * COVER_CELL)
            inside = shapely.contains_xy(poly, gx, gy)
            grid[c0:c1, a0:a1] = np.maximum(grid[c0:c1, a0:a1], inside)
        return ShareGrid(i0, j0, _blur(grid, FOREST_SIGMA / COVER_CELL))

    def forest_share(self, pos):
        """Share of forest round each point, Gaussian-weighted (FOREST_SIGMA), from a COVER_CELL raster on the
        global lattice (self.grid when it covers the points), read bilinearly."""
        pos = np.asarray(pos, float).reshape(-1, 2)
        if not self.forest or not len(pos):
            return np.zeros(len(pos))
        grid = self.grid if self.grid is not None and self.grid.covers(pos) else \
            self.share_grid((*pos.min(0), *pos.max(0)))
        return grid.at(pos)


def read_cover(path, sha256=None):
    """{'forest': {id: wkb (lon/lat)}, 'structures': {way: [(lon, lat)]}} of one OSM source tile."""
    import hashlib
    import osmium
    from pathlib import Path
    if sha256 is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError(f'OSM checksum mismatch: {path}')
    forest, structures = {}, {}
    factory = osmium.geom.WKBFactory()

    class Reader(osmium.SimpleHandler):
        def way(self, w):
            t = w.tags
            if t.get('highway') in ROAD_CLASSES or w.id in structures:
                return                                  # roads are in the graph with their own tags
            if t.get('bridge') in (None, 'no') and t.get('man_made') != 'bridge':
                return
            try:
                pts = [(n.lon, n.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            if len(pts) >= 2:
                structures[w.id] = pts

        def area(self, a):
            t = a.tags
            if ('landuse', t.get('landuse')) not in FOREST_TAGS and ('natural', t.get('natural')) not in FOREST_TAGS:
                return
            key = (a.from_way(), a.orig_id())
            if key in forest:
                return
            try:
                forest[key] = factory.create_multipolygon(a)
            except Exception:                           # a broken multipolygon is skipped, as osmium does
                return

    Reader().apply_file(str(path), locations=True)
    return {'forest': forest, 'structures': structures}


def cover_of(parts, frame):
    """Cover in the region frame from read_cover's parts (the first copy of a feature wins)."""
    import shapely
    from shapely import wkb
    forest, structures = {}, {}
    for part in parts:
        for k, v in part['forest'].items():
            forest.setdefault(k, v)
        for k, v in part['structures'].items():
            structures.setdefault(k, v)
    polys = []
    for hexwkb in forest.values():
        g = wkb.loads(hexwkb, hex=True)
        c = shapely.get_coordinates(g)
        x, y = frame.local(c[:, 0], c[:, 1])
        g = shapely.make_valid(shapely.set_coordinates(g, np.column_stack([x, y])))
        if not g.is_empty:
            shapely.prepare(g)
            polys.append(g)
    lines = []
    for pts in structures.values():
        lon, lat = zip(*pts)
        x, y = frame.local(lon, lat)
        lines.append(np.column_stack([x, y]))
    return Cover(polys, lines)


def near_distance(points, targets, reach, index=False):
    """Distance from each point to the nearest target point (inf beyond reach). Grid hashing, no SciPy.
    With index, also that target's row (-1 beyond reach)."""
    out = np.full(len(points), np.inf)
    which = np.full(len(points), -1, np.int64)
    if not len(points) or not len(targets):
        return (out, which) if index else out
    cell = float(reach)
    M = 1 << 21
    def keys(xy, dx=0, dy=0):
        return (np.floor(xy[:, 0] / cell).astype(np.int64) + dx + M) * (2 * M) + np.floor(xy[:, 1] / cell).astype(np.int64) + dy + M
    tkey = keys(targets); order = np.argsort(tkey, kind='stable'); skey = tkey[order]
    for start in range(0, len(points), 200000):
        p = points[start:start + 200000]
        best = np.full(len(p), np.inf)
        arg = np.full(len(p), -1, np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nk = keys(p, dx, dy)
                lo = np.searchsorted(skey, nk, 'left'); hi = np.searchsorted(skey, nk, 'right')
                cnt = hi - lo
                if not cnt.sum():
                    continue
                ia = np.repeat(np.arange(len(p)), cnt)
                it = order[lo[ia] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
                d = np.hypot(*(p[ia] - targets[it]).T)
                np.minimum.at(best, ia, d)
                if index:
                    hit = d <= best[ia]
                    arg[ia[hit]] = it[hit]
        out[start:start + len(p)] = best
        which[start:start + len(p)] = arg
    far = out > reach
    out[far] = np.inf
    which[far] = -1
    return (out, which) if index else out


def densify(lines, step=10.):
    pts = []
    for xy in lines:
        xy = np.asarray(xy, float).reshape(-1, 2)
        if len(xy) < 2:
            pts.append(xy)
            continue
        cum = np.r_[0., np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        t = np.linspace(0., cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1))
        pts.append(np.column_stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])]))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def structure_weight(d):
    """Data weight by distance to the nearest structure: STRUCT_W within STRUCT_NEAR m, 1 from STRUCT_FAR m."""
    return STRUCT_W + (1 - STRUCT_W) * np.clip((d - STRUCT_NEAR) / (STRUCT_FAR - STRUCT_NEAR), 0., 1.)


def ground_line(dem, pos, mass, weight, edges, triples, chains, iterations=400, tol=1e-6):
    """The DEM's ground under the roads: min sum m w hub (h - dem)^2 + sum g (dh)^2 + sum c (d2h)^2, with
    Huber weights hub from IRLS passes (HUBER m). edges (a, b, g) and triples (a, b, c, k) carry their own
    stiffness (the class's smoothing length). Returns the heights and the share of stations the
    reweighting took below half weight."""
    nvar = len(dem)
    ea, eb, g = edges
    ta, tb, tc, k = triples
    rows = [ea, eb, ea, eb]; cols = [ea, eb, eb, ea]; vals = [g, g, -g, -g]
    for i, (a, ca) in enumerate(((ta, 1.), (tb, -2.), (tc, 1.))):
        for b, cb in ((ta, 1.), (tb, -2.), (tc, 1.)):
            rows.append(a); cols.append(b); vals.append(k * ca * cb)
    ptr, col, data, _ = csr(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), nvar)
    chains.bands(ptr, col, data)
    rowof = np.repeat(np.arange(nvar), np.diff(ptr))
    smooth_diag = np.bincount(rowof, np.where(col == rowof, data, 0.), nvar)

    def smooth(v):
        return np.add.reduceat(data * v[col], ptr[:-1])
    hub = np.ones(nvar)
    y = np.zeros(nvar)
    rhs0 = -smooth(dem)                                 # solve for the offset y = h - dem
    low = 0.
    for _ in range(IRLS + 1):
        d = mass * weight * hub
        diag = smooth_diag + d
        chains.factor(diag)
        r = rhs0 - (smooth(y) + d * y)
        norm0 = math.sqrt(rhs0 @ rhs0) or 1.
        z = chains.apply(r); p = z.copy(); rz = r @ z
        for _it in range(iterations):
            if math.sqrt(r @ r) < tol * norm0 or rz <= 0:
                break
            ap = smooth(p) + d * p
            alpha = rz / (p @ ap)
            y += alpha * p; r -= alpha * ap
            z = chains.apply(r); rz2 = r @ z
            p = z + (rz2 / rz) * p; rz = rz2
        res = np.abs(y)
        hub = np.where(res > HUBER, HUBER / np.maximum(res, 1e-9), 1.)
        low = float(((hub < .5) & (weight > .5)).mean())
    return dem + y, low


def crossings(pos, seg_of, same, upper, lower, widths, levels):
    """Vertices of the overlap between upper deck and lower road station ribbons.

    Heights vary affinely along each straight station edge, so their least separation
    over the common polygon occurs at a vertex. Centre-line intersections alone missed
    a wide road's edges and bridges that overlap a road without their axes crossing.
    Returns edge starts (a, b) and longitudinal parameters (u, t) at each vertex.
    """
    e0 = np.flatnonzero(same)
    top, bottom = e0[upper[e0]], e0[lower[e0]]
    if not len(top) or not len(bottom):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0)
    cell = 32.0
    mid = (pos[bottom] + pos[bottom + 1]) / 2
    key = np.floor(mid / cell).astype(np.int64)
    bins = {}
    for k, (x, y) in enumerate(key.tolist()):
        bins.setdefault((x, y), []).append(k)
    # A midpoint hash keeps country-sized indexes compact. Every lower edge is at
    # most STEP long; expand a deck's query by the largest lower half width and length.
    reach = float(np.max(widths[bottom])) / 2 + STEP / 2

    def ribbon(p, r, half):
        normal = np.array([-r[1], r[0]]) * (half / max(float(np.linalg.norm(r)), 1e-9))
        return [p - normal, p + r - normal, p + r + normal, p + normal]

    def clip(poly, q, d, half):
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            return []
        axis = d / length
        normal = np.array([-axis[1], axis[0]])
        # dot(p-q, n) >= bound: longitudinal ends and both lateral edges.
        for n, bound in ((axis, 0.), (-axis, -length), (normal, -half), (-normal, -half)):
            if not poly:
                break
            result = []
            a = poly[-1]; da = float((a - q) @ n) - bound
            for b in poly:
                db = float((b - q) @ n) - bound
                if (da >= 0) != (db >= 0):
                    result.append(a + (b - a) * (da / (da - db)))
                if db >= 0:
                    result.append(b)
                a, da = b, db
            poly = result
        return poly

    out = ([], [], [], [])
    for a in top.tolist():
        p, r = pos[a], pos[a + 1] - pos[a]
        poly = ribbon(p, r, widths[a] / 2 + DECK_OVERHANG)
        low = np.min(poly, axis=0); high = np.max(poly, axis=0)
        lo = np.floor((low - reach) / cell).astype(int)
        hi = np.floor((high + reach) / cell).astype(int)
        near = [k for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)
                for k in bins.get((x, y), ())]
        if not near:
            continue
        b = bottom[near]
        b = b[(seg_of[a] != seg_of[b]) & (levels[a] > levels[b])]
        q, d = pos[b], pos[b + 1] - pos[b]
        half = widths[b] / 2
        hit = np.all(np.maximum(q, q + d) + half[:, None] >= low, axis=1)
        hit &= np.all(np.minimum(q, q + d) - half[:, None] <= high, axis=1)
        for k in np.flatnonzero(hit).tolist():
            overlap = clip(poly, q[k], d[k], half[k])
            for point in overlap:
                u = np.clip(float((point - p) @ r) / max(float(r @ r), 1e-9), 0, 1)
                t = np.clip(float((point - q[k]) @ d[k]) / max(float(d[k] @ d[k]), 1e-9), 0, 1)
                out[0].append(a); out[1].append(int(b[k])); out[2].append(u); out[3].append(t)
    return (np.array(out[0], int), np.array(out[1], int), np.array(out[2]), np.array(out[3]))


def close_pairs(pos, seg_of, width, ground, same):
    """Station -> nearby edge of another ground segment whose strip touches it: lateral
    distance to the edge < half widths + 0.5 m. Returns station rows a, edge start rows e
    (edge = e, e+1) and the parameter t of the closest point. Grid hashing, no SciPy."""
    cell = 24.0
    e0 = np.flatnonzero(same)                     # edge e0 -> e0+1, both of one segment
    mid = (pos[e0] + pos[e0 + 1]) / 2
    M = 1 << 21
    def keys(xy, dx=0, dy=0):
        return (np.floor(xy[:, 0] / cell).astype(np.int64) + dx + M) * (2 * M) + np.floor(xy[:, 1] / cell).astype(np.int64) + dy + M
    ekey = keys(mid); order = np.argsort(ekey, kind='stable'); skey = ekey[order]
    out = ([], [], [])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nk = keys(pos, dx, dy)
            lo = np.searchsorted(skey, nk, 'left'); hi = np.searchsorted(skey, nk, 'right')
            cnt = hi - lo
            if not cnt.sum():
                continue
            ia = np.repeat(np.arange(len(pos)), cnt)
            ie = e0[order[lo[ia] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]]
            keep = (seg_of[ia] != seg_of[ie]) & ground[ia] & ground[ie]
            ia, ie = ia[keep], ie[keep]
            a, b = pos[ie], pos[ie + 1]
            ab = b - a; L2 = np.maximum((ab ** 2).sum(1), 1e-9)
            t = np.clip(((pos[ia] - a) * ab).sum(1) / L2, 0, 1)
            d = np.linalg.norm(pos[ia] - (a + ab * t[:, None]), axis=1)
            keep = d < (width[ia] + width[ie]) / 2 + .5
            out[0].append(ia[keep]); out[1].append(ie[keep]); out[2].append(t[keep])
    if not out[0]:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return tuple(np.concatenate(v) for v in out)


def layer(tags):
    try:
        value = abs(float(str(tags.get('layer', 1)).split(';')[0]))
        return value if math.isfinite(value) else 1.
    except (TypeError, ValueError):
        return 1.


def kind(tags):
    if tags.get('bridge') not in (None, 'no'):
        return 'bridge'
    if tags.get('tunnel') not in (None, 'no', 'building_passage'):
        return 'tunnel'
    return 'road'


def held_stations(segments, kinds, s, seg_of, run, xb, xt, count, classes=''):
    """Stations of a HOLD_UNDER road (classes '': HOLD_UNDER now; None: any; not a bridge or tunnel) within HOLD_REACH m
    of a crossing under a deck, measured along that road: a bool mask over the stations."""
    if classes == '':
        classes = HOLD_UNDER
    held = np.zeros(count, bool)
    bounds = np.r_[run, count]
    lower = seg_of[xb]
    for w in np.unique(lower):
        if kinds[w] != 0 or (classes is not None and segments[w]['tags'].get('highway') not in classes):
            continue
        mine = lower == w
        at = s[xb[mine]] + xt[mine] * (s[np.minimum(xb[mine] + 1, count - 1)] - s[xb[mine]])
        sl = slice(int(bounds[w]), int(bounds[w + 1]))
        held[sl] = np.min(np.abs(s[sl][:, None] - at[None, :]), axis=1) <= HOLD_REACH
    return held


def joints(segments, s, ids, pos, run):
    """Station triples across the nodes where roads run on: (previous station's variable, the two
    segments) as rows of ja, the node's and the next station's variables, and the triple's mean gap;
    and the links (segment, its end 0/1, segment, its end) that run on through each node.
    At a node of two segments they pair whatever their angle; at a junction, segments of one highway
    pair when they leave it nearly straight on (cos < -0.7), most opposite first."""
    bounds = np.r_[run, len(s)]
    ends = {}
    for k in range(len(segments)):
        a0, a1 = int(bounds[k]), int(bounds[k + 1]) - 1
        if a1 <= a0:
            continue
        for e, (at, nb) in enumerate(((a0, a0 + 1), (a1, a1 - 1))):
            ends.setdefault(int(ids[at]), []).append((k, nb, at, e))
    rows, mid, last, gaps, links = [], [], [], [], []
    for node, here in ends.items():
        if len(here) < 2:
            continue
        heads = []
        for k, nb, at, e in here:
            d = pos[nb] - pos[at]
            heads.append(d / max(float(np.hypot(d[0], d[1])), 1e-9))
        options = []
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                c = float(heads[i] @ heads[j])
                hw_i = segments[here[i][0]]['tags'].get('highway')
                if len(here) == 2 or (c < -.7 and hw_i == segments[here[j][0]]['tags'].get('highway')):
                    options.append((c, i, j))
        used = set()
        for c, i, j in sorted(options):
            if i in used or j in used:
                continue
            (u, nu, au, eu), (w, nw, aw, ew) = here[i], here[j]
            if ids[nu] == ids[nw] or u == w:
                continue                       # two segments between the same two nodes, or a loop
            used |= {i, j}
            rows.append((int(ids[nu]), u, w)); mid.append(node); last.append(int(ids[nw]))
            gaps.append(max((abs(s[nu] - s[au]) + abs(s[nw] - s[aw])) / 2, .5))
            links.append((u, eu, w, ew))
    if not rows:
        return np.zeros((0, 3), int), np.zeros(0, int), np.zeros(0, int), np.zeros(0), links
    return np.array(rows, int), np.array(mid, int), np.array(last, int), np.array(gaps), links


class Chains:
    """CG's preconditioner: roads as chains of variables (a segment's interior stations, the node to
    the segment it runs on into, that one's stations...), cut into blocks of at most CHAIN, each
    block's part of the matrix (5 bands: grade and change of grade along the road) factored as
    L D L^T. Everything else (other nodes, couplings, springs off the bands) stays diagonal."""

    def __init__(self, ids, run, total, nvar, links):
        bounds = np.r_[run, total]
        nseg = len(run)
        partner = {}
        for u, eu, w, ew in links:
            partner[u, eu] = (w, ew); partner[w, ew] = (u, eu)

        def inner(k, forward):
            v = ids[int(bounds[k]) + 1:int(bounds[k + 1]) - 1]
            return v if forward else v[::-1]

        def node(k, e):
            return int(ids[int(bounds[k]) if e == 0 else int(bounds[k + 1]) - 1])
        seen = np.zeros(nseg, bool)
        placed = np.zeros(nvar, bool)
        order = []
        for k0 in range(nseg):
            if seen[k0]:
                continue
            # back to the chain's first segment: leave k0 by its end 0 while a partner runs on
            k, e = k0, 0
            steps = 0
            while (k, e) in partner and steps < nseg:
                w, ew = partner[k, e]
                if seen[w] or w == k0:
                    break
                k, e = w, 1 - ew
                steps += 1
            # k is the first segment; it is entered at its end e and left at 1 - e
            chain = []
            while True:
                seen[k] = True
                chain.append(inner(k, e == 0))
                out = 1 - e
                if (k, out) not in partner:
                    break
                w, ew = partner[k, out]
                n = node(k, out)
                if seen[w] or placed[n]:
                    break
                chain.append(np.array([n])); placed[n] = True
                k, e = w, ew
            chain = np.concatenate(chain) if chain else np.zeros(0, int)
            if len(chain):
                order.append(chain)
        perm = np.concatenate(order) if order else np.zeros(0, int)
        lengths = np.array([len(o) for o in order], int)
        rest = np.setdiff1d(np.arange(nvar), perm)
        self.perm = np.r_[perm, rest].astype(int)
        # blocks of at most CHAIN positions, longest first for the level sweeps
        starts = np.r_[0, np.cumsum(lengths)[:-1]] if len(lengths) else np.zeros(0, int)
        cuts = (lengths + CHAIN - 1) // CHAIN
        owner = np.repeat(np.arange(len(lengths)), cuts)
        j = np.arange(cuts.sum()) - np.repeat(np.cumsum(cuts) - cuts, cuts)
        bstart = starts[owner] + j * CHAIN
        blen = np.minimum(CHAIN, lengths[owner] - j * CHAIN)
        sort = np.argsort(-blen, kind='stable')
        bstart, blen = bstart[sort], blen[sort]
        self.block = np.full(nvar + 2, -1)
        for b, (a, n) in enumerate(zip(bstart.tolist(), blen.tolist())):
            self.block[a:a + n] = b
        top = int(blen.max()) if len(blen) else 0
        self.levels = [bstart[:int((blen > k).sum())] + k for k in range(top)]
        self.nvar, self.factored = nvar, None

    def bands(self, ptr, col, data):
        """The matrix's entries between chain neighbours one and two positions apart in a block."""
        n, perm = self.nvar, self.perm
        key = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr)) * n + col
        self.a1, self.a2, self.factored = np.zeros(n + 2), np.zeros(n + 2), None
        q = np.arange(n)
        for off, a in ((1, self.a1), (2, self.a2)):
            ok = (q + off < n) & (self.block[q] >= 0)
            ok[ok] &= self.block[q[ok] + off] == self.block[q[ok]]
            qq = q[ok]
            want = perm[qq].astype(np.int64) * n + perm[qq + off]
            at = np.minimum(np.searchsorted(key, want), len(key) - 1)
            hit = key[at] == want
            a[qq[hit]] = data[at[hit]]

    def factor(self, diag):
        n = self.nvar
        d = np.r_[diag[self.perm], 1., 1.].astype(float)
        l1, l2 = np.zeros(n + 2), np.zeros(n + 2)
        a1, a2 = self.a1, self.a2
        for k, cur in enumerate(self.levels):
            if k >= 1:
                p1 = cur - 1
                dd = d[cur] - l1[p1] ** 2 * d[p1]
                if k >= 2:
                    p2 = cur - 2
                    dd -= l2[p2] ** 2 * d[p2]
                d[cur] = dd
                l1[cur] = (a1[cur] - l2[p1] * d[p1] * l1[p1]) / dd
            else:
                l1[cur] = a1[cur] / d[cur]
            l2[cur] = a2[cur] / d[cur]
        self.d, self.l1, self.l2, self.factored = d, l1, l2, diag.copy()

    def apply(self, r):
        d, l1, l2 = self.d, self.l1, self.l2
        y = np.r_[r[self.perm], 0., 0.]
        for k, cur in enumerate(self.levels):
            if k >= 1:
                y[cur] -= l1[cur - 1] * y[cur - 1]
                if k >= 2:
                    y[cur] -= l2[cur - 2] * y[cur - 2]
        z = y / d
        for cur in reversed(self.levels):
            z[cur] -= l1[cur] * z[cur + 1] + l2[cur] * z[cur + 2]
        out = np.empty(self.nvar)
        out[self.perm] = z[:self.nvar]
        return out


def stations(segments, node_index):
    """All segments' stations at once (162k segments: a Python loop cost 12 s).

    Returns per-station arc length s, variable id, position, owning segment and the start of
    each segment's run; ends map to their graph node, interior stations get fresh ids."""
    n_seg = len(segments)
    lens = np.array([len(seg['xy']) for seg in segments])
    xy = np.concatenate([np.asarray(seg['xy'], dtype=float) for seg in segments])
    first = np.r_[0, np.cumsum(lens)[:-1]]
    step = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    step[first] = 1.                       # a 1 m gap keeps the global arc length increasing
    cum = np.cumsum(step)
    base = cum[first]                      # each segment's start on the global arc length
    length = cum[first + lens - 1] - base
    n = np.maximum(1, np.ceil(length / STEP)).astype(np.int64)
    count = n + 1
    run = np.r_[0, np.cumsum(count)[:-1]]
    seg_of = np.repeat(np.arange(n_seg), count)
    k = np.arange(count.sum()) - run[seg_of]
    s = length[seg_of] * k / n[seg_of]
    g = base[seg_of] + s
    pos = np.column_stack([np.interp(g, cum, xy[:, 0]), np.interp(g, cum, xy[:, 1])])
    a = np.array([node_index[seg['nodes'][0]] for seg in segments])
    b = np.array([node_index[seg['nodes'][1]] for seg in segments])
    inner_run = len(node_index) + np.r_[0, np.cumsum(n - 1)[:-1]]
    ids = inner_run[seg_of] + k - 1
    ids[k == 0] = a[seg_of[k == 0]]
    ends = k == n[seg_of]
    ids[ends] = b[seg_of[ends]]
    nvar = len(node_index) + int((n - 1).sum())
    return s, ids, pos, seg_of, run, nvar


def csr(rows, cols, vals, n, pattern=None):
    """COO -> CSR with duplicates summed (no SciPy here). With the `pattern` of an earlier call on the
    same rows and columns, only the values are summed again (the grade passes change weights only)."""
    if pattern is None:
        key = rows * n + cols
        order = np.argsort(key, kind='stable')
        uniq, start = np.unique(key[order], return_index=True)
        r, c = uniq // n, uniq % n
        ptr = np.r_[0, np.cumsum(np.bincount(r, minlength=n))]
        pattern = order, start, ptr, c
    order, start, ptr, c = pattern
    return ptr, c, np.add.reduceat(vals[order], start), pattern


def _rule_ground(segments, raw, known, pos, ids, seg_of, k_st, nnodes, mass, ea, eb, el, ta, tb, tc, tl, tri_pair,
                 run, nvar, links, cover):
    """The ground line of the DEM rules (RULES): returns it, each variable's distance to the nearest
    structure (inf beyond STRUCT_FAR) and a note for the solve's summary."""
    # structures: every station of a bridge segment, and the cover's other structures (railway, path)
    deck = np.unique(ids[k_st > 0])
    points = [pos[deck]]
    if cover is not None and cover.structures:
        points.append(densify(cover.structures))
    near = near_distance(pos, np.concatenate(points), STRUCT_FAR)
    weight = np.where(known, structure_weight(near), 1e-3)
    # decks and tunnels: the DEM there is the valley, the deck or the hill above
    structure = np.zeros(nvar, bool); structure[ids[(k_st != 0) & (ids >= nnodes)]] = True
    weight[structure] = 1e-4
    share = cover.forest_share(pos) if cover is not None else np.zeros(nvar)
    dem = raw - FOREST_BIAS * share
    # the class's smoothing length on the change of grade (BETA's 50 m elsewhere); grades as the solve's
    def length(hw):
        return GROUND_LENGTH.get(hw, GROUND_LENGTH_DEFAULT)
    g = ALPHA / el * np.ones(len(ea))
    tri_len = np.array([min(length(u), length(w)) for u, w in tri_pair])
    c = tri_len ** 4 / tl ** 3
    chains = Chains(ids, run, len(seg_of), nvar, links)
    ground, low = ground_line(dem, pos, np.maximum(mass, .5), weight, (ea, eb, g), (ta, tb, tc, c), chains)
    note = (f'; DEM rules: {int((near < STRUCT_FAR).sum())} stations near structures, '
            f'{int((share > .05).sum())} forest-corrected (mean {FOREST_BIAS * share[share > .05].mean() if (share > .05).any() else 0:.1f} m), '
            f'{100 * low:.1f} % reweighted (Huber {HUBER} m)')
    return ground, near, note


def _fine_ground(segments, reference, ground, raw, known, pos_st, ids, seg_of, s, k_st, run, nvar, cover):
    """The DTM's part of the data term (reference: a dtm.Dtm, or anything with road(xy, tangent, half) ->
    (height, share)). Returns the blended target base and raw ground, each variable's DTM share (0 on
    decks and tunnels), its weight factor (OWN_W near its own deck), the known mask and a note."""
    n_st = len(ids)
    same = seg_of[1:] == seg_of[:-1]
    rows = np.arange(n_st)
    prev = np.where(np.r_[False, same], rows - 1, rows)
    nxt = np.where(np.r_[same, False], rows + 1, rows)
    tangent = pos_st[nxt] - pos_st[prev]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)[:, None]
    widths = np.clip(np.array([seg.get('width', 4.) for seg in segments], float), .5, 80.)
    fine, share = reference.road(pos_st, tangent, widths[seg_of] / 2)
    fine, share = np.asarray(fine, float), np.asarray(share, float)
    share = np.where(np.isfinite(fine), share, 0.)
    ground_row = k_st == 0
    # structures: the bridge segments' lines (5 m) and the cover's other bridges (railway, path)
    bridges = [k for k, seg in enumerate(segments) if kind(seg['tags']) == 'bridge']
    pts, owner = [], []
    for k in bridges:
        p = densify([segments[k]['xy']], 5.)
        pts.append(p); owner.append(np.full(len(p), k))
    if cover is not None and cover.structures:
        p = densify(cover.structures, 5.)
        pts.append(p); owner.append(np.full(len(p), -1))
    under = np.zeros(n_st, bool); near_own = np.zeros(n_st, bool); d_own = np.full(n_st, np.inf)
    if pts:
        targets, owner = np.concatenate(pts), np.concatenate(owner)
        d, which = near_distance(pos_st, targets, float(widths.max()) / 2 + UNDER_MARGIN + OWN_FAR, index=True)
        hit = np.flatnonzero(np.isfinite(d) & ground_row & (share > 0))
        if len(hit):
            o = owner[which[hit]]
            mine = seg_of[hit]
            na = np.array([segments[k]['nodes'][0] for k in range(len(segments))])
            nb = np.array([segments[k]['nodes'][1] for k in range(len(segments))])
            oo = np.maximum(o, 0)
            own = (o >= 0) & ((na[mine] == na[oo]) | (na[mine] == nb[oo]) | (nb[mine] == na[oo]) | (nb[mine] == nb[oo]))
            half_o = np.where(o >= 0, widths[oo] / 2, 4.)
            under[hit] = ~own & (d[hit] < widths[mine] / 2 + half_o + UNDER_MARGIN)
            near_own[hit] = own & (d[hit] < OWN_FAR)
            d_own[hit] = np.where(own, d[hit], np.inf)
    # a road under a deck: its DTM line bridged from the clear stations either side; near its own deck:
    # the clear ramp's line (the last up to three stations within 60 m) extended to the abutment
    fixed = fine.copy()
    bounds = np.r_[run, n_st]
    for k in np.unique(seg_of[under | near_own]).tolist():
        sl = slice(int(bounds[k]), int(bounds[k + 1]))
        ss, ff = s[sl], fine[sl]
        u, o = under[sl], near_own[sl]
        clear = ~u & ~o & (share[sl] > 0)
        if u.any():
            if clear.any():
                fixed[sl][u] = np.interp(ss[u], ss[clear], ff[clear])
            else:
                share[sl][u] = 0.                 # no clear station: the neighbours and the rules decide
        for r in np.flatnonzero(o):
            near = np.flatnonzero(clear & (np.abs(ss - ss[r]) <= 60.))
            if not len(near):
                continue
            near = near[np.argsort(np.abs(ss[near] - ss[r]))[:3]]
            if len(near) >= 2 and np.ptp(ss[near]) > 5.:
                a, b = np.polyfit(ss[near], ff[near], 1)
                fixed[sl][r] = a * ss[r] + b
            else:
                fixed[sl][r] = ff[near[0]]
    factor = np.where(near_own, OWN_W + (1 - OWN_W) * np.clip(d_own / OWN_FAR, 0., 1.), 1.)
    # stations -> variables: the ground rows decide (a node between two decks has none: no DTM there)
    share_v = np.zeros(nvar); fine_v = np.zeros(nvar); own_v = np.ones(nvar)
    geo_v = np.zeros(nvar); raw_fine = np.zeros(nvar)
    ok = np.isfinite(fine)
    geo_v[ids[ok]] = share[ok]; raw_fine[ids[ok]] = fine[ok]
    g = ground_row & (share > 0)
    share_v[ids[g]] = share[g]; fine_v[ids[g]] = fixed[g]
    np.minimum.at(own_v, ids[g], factor[g])
    dem = share_v * fine_v + (1 - share_v) * ground
    raw_b = geo_v * raw_fine + (1 - geo_v) * raw
    known = known | (geo_v > 0)
    note = (f'; DTM: {int((share_v >= 1).sum())} stations measured, {int(((share_v > 0) & (share_v < 1)).sum())} '
            f'blended at the coverage edge, {int((under & g).sum())} under decks bridged, '
            f'{int((near_own & g).sum())} at their own deck')
    return dem, raw_b, share_v, own_v, known, note


def spans(segments, node_index, kinds, dem):
    """{node variable: height} of the nodes inside a bridge mapped in pieces.

    Only an end on the ground is an abutment: a node where nothing but bridges meet is a split
    of the way, and the DEM under it is the water (the Golden Gate's deck, split at its north
    tower, dived to the sea). Between the abutments the heights run linearly along the deck."""
    ends = [(k, node_index[seg['nodes'][0]], node_index[seg['nodes'][1]]) for k, seg in enumerate(segments)]
    ground = {n for k, a, b in ends if kinds[k] <= 0 for n in (a, b)}
    links = {}
    for k, a, b in ends:
        if kinds[k] > 0 and a != b:
            xy = np.asarray(segments[k]['xy'], dtype=float)
            weight = 1. / max(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), .5)
            links.setdefault(a, []).append((b, weight))
            links.setdefault(b, []).append((a, weight))
    inside = {n for n, near in links.items() if n not in ground and len(near) >= 2}
    heights, seen = {}, set()
    for start in inside:
        if start in seen:
            continue
        part, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            part.append(n)
            for m, _ in links[n]:
                if m in inside and m not in seen:
                    seen.add(m)
                    stack.append(m)
        where = {n: i for i, n in enumerate(part)}
        system, values, anchored = np.zeros((len(part), len(part))), np.zeros(len(part)), False
        for n, i in where.items():
            for m, weight in links[n]:
                system[i, i] += weight
                if m in where:
                    system[i, where[m]] -= weight
                else:
                    values[i] += weight * dem[m]
                    anchored = True
        if anchored:                       # a ring of bridges with no way down keeps its DEM
            heights.update(zip(part, np.linalg.solve(system, values).tolist()))
    return heights


def solve(segments, sample, iterations=400, tol=1e-3, emit=None, floors=None, rules=None, cover=None, reference=None):
    """Heights for every segment's stations. sample(xy) -> DEM height (vectorised, NaN outside).
    floors: [(segment index, s0, s1, z)]: the segment's stations within s0..s1 stay at least at z.
    rules: the DEM rules (None: RULES, the EARTHNG_DEM_RULES switch, or a reference); cover: a Cover
    (forest, other structures) for them. reference: a national DTM (dtm.Dtm; FINE_W above), whose
    datum the coarse sample must already share (Dtm.corrected).

    Returns a list of (s, h) per segment and {node id: h}."""
    nodes = sorted({n for seg in segments for n in seg['nodes']})
    node_index = {n: k for k, n in enumerate(nodes)}
    s, ids, pos_st, seg_of, run, nvar = stations(segments, node_index)
    pos = np.zeros((nvar, 2)); pos[ids] = pos_st
    same = seg_of[1:] == seg_of[:-1]                       # consecutive stations of one segment
    ds = np.diff(s)
    ea, eb, el = ids[:-1][same], ids[1:][same], np.maximum(ds[same], .5)
    mass = np.bincount(ea, el / 2, nvar) + np.bincount(eb, el / 2, nvar)
    tri = same[:-1] & same[1:]
    ta, tb, tc = ids[:-2][tri], ids[1:-1][tri], ids[2:][tri]
    tl = np.maximum((ds[:-1][tri] + ds[1:][tri]) / 2, .5)
    tri_hw = [segments[k]['tags'].get('highway') for k in seg_of[1:-1][tri].tolist()]
    tri_pair = list(zip(tri_hw, tri_hw))
    # Triples across a node: two segments that run on through it (the only two there, or two of one
    # highway nearly straight on) share a change-of-grade term. Within segments alone the grade could
    # turn freely at every node: a Niedersachsen pack had the motorway go from +2.4 % to -2.0 % at a way
    # joint (a crest of about 500 m radius at 123 km/h).
    if ROAD_RULES:
        ja, jb, jc, jl, links = joints(segments, s, ids, pos_st, run)
    else:                       # upstream's solve: the grade may turn freely at a way joint
        ja, jb, jc, jl, links = np.zeros((0, 3), int), np.zeros(0, int), np.zeros(0, int), np.zeros(0), []
    if len(ja):
        ta, tb, tc, tl = np.r_[ta, ja[:, 0]], np.r_[tb, jb], np.r_[tc, jc], np.r_[tl, jl]
        tri_pair += [(segments[u]['tags'].get('highway'), segments[w]['tags'].get('highway')) for u, w in ja[:, 1:].tolist()]
    dem = sample(pos)
    # whole OSM ways reach past the downloaded DEM (their tiles fail no_dem anyway): there the
    # height only follows the neighbours
    known = np.isfinite(dem)
    dem[~known] = np.median(dem[known]) if known.any() else 0.
    kinds = np.array([{'road': 0, 'bridge': 1, 'tunnel': -1}[kind(seg['tags'])] for seg in segments])
    layers = np.array([max(1., layer(seg['tags'])) for seg in segments])
    k_st = kinds[seg_of]
    rules = (RULES or reference is not None) if rules is None else bool(rules)
    raw = dem
    rule_note, near_struct = '', None
    if rules:
        dem, near_struct, rule_note = _rule_ground(segments, raw, known, pos, ids, seg_of, k_st, len(node_index),
                                                   mass, ea, eb, el, ta, tb, tc, tl, tri_pair, run, nvar, links, cover)
    fine_share = None
    if reference is not None:
        # the DTM where it covers, the coarse ground line elsewhere (the decks' clearance over the DTM too)
        dem, raw, fine_share, fine_own, known, fine_note = _fine_ground(
            segments, reference, dem, raw, known, pos_st, ids, seg_of, s, k_st, run, nvar, cover)
        rule_note += fine_note
    target = dem.copy(); weight = np.where(known, 1., 1e-3)
    inner = (k_st < 0) & (ids >= len(node_index))
    target[ids[inner]] = raw[ids[inner]] - TUNNEL_DEPTH * layers[seg_of[inner]]
    weight[ids[inner]] = np.where(known[ids[inner]], SOFT, 1e-3)
    # A span keeps the chord between its abutments. The river bed is not a target, or the
    # deck sags into the valley. Flat overpasses still want BRIDGE_CLEAR above the ground.
    # (With the rules the abutments are the ground line's, the clearance is over the raw DEM.)
    bounds = np.r_[run, len(s)]
    deck = spans(segments, node_index, kinds, dem) if (kinds > 0).any() else {}
    for k in np.flatnonzero(kinds > 0):
        sl = slice(int(bounds[k]), int(bounds[k + 1]))
        sid = ids[sl]
        for n in (int(sid[0]), int(sid[-1])):
            if n in deck:
                target[n] = max(deck[n], raw[n] + BRIDGE_CLEAR * layers[k])
                weight[n] = SOFT if known[n] else 1e-3
        interior = sid >= len(node_index)
        if not interior.any():
            continue
        ss = s[sl]
        span = float(ss[-1] - ss[0])
        t = (ss - ss[0]) / span if span > 1e-6 else np.zeros(len(ss))
        chord = deck.get(int(sid[0]), dem[sid[0]]) * (1 - t) + deck.get(int(sid[-1]), dem[sid[-1]]) * t
        want = np.maximum(chord, raw[sid] + BRIDGE_CLEAR * layers[k])
        target[sid[interior]] = want[interior]
        weight[sid[interior]] = np.where(known[sid[interior]], SOFT, 1e-3)
    if near_struct is not None:
        # ramps: near a structure the ground line is only a guide, the deck and its grades decide
        ground_var = np.zeros(nvar, bool); ground_var[ids[k_st == 0]] = True
        weight = np.where(ground_var, weight * np.maximum(structure_weight(near_struct), RAMP_W), weight)
    if fine_share is not None:
        # measured stations: the DTM's weight (decks and tunnels have share 0 and keep theirs)
        weight = (1 - fine_share) * weight + fine_share * FINE_W * fine_own
    mw = weight * np.maximum(mass, .5)
    degree = np.bincount(np.array([node_index[n] for seg in segments for n in seg['nodes']]), minlength=nvar)
    flat = np.where((degree[ea] >= 3) | (degree[eb] >= 3), JUNCTION_FLAT, 1.)
    ce, ct = ALPHA * flat / el, BETA / tl ** 3
    # each station triple's smallest crest and sag radius (inf: no limit), by its middle station's road
    # a joint of two classes takes the looser limits; with a class without one, none
    def pair_class(u, w):
        a, b = CURVE_MIN.get(u), CURVE_MIN.get(w)
        return (min(a[0], b[0]), min(a[1], b[1])) if a and b else None
    tri_class = [CURVE_MIN.get(u) if u == w else pair_class(u, w) for u, w in tri_pair]
    crest_min = np.array([c[0] if c else np.inf for c in tri_class])
    sag_min = np.array([c[1] if c else np.inf for c in tri_class])
    # only these triples are checked and stiffened: with no limit, 1 / inf = 0 counted every bend of every
    # minor road as too tight (261,268 in a Niedersachsen build, a divide by zero in the ratio) and drove
    # their weight to CURVE_STIFF_MAX, which straightened them and took CG to its 400-iteration cap (×40 time)
    limited = np.isfinite(crest_min)
    fast_tri = np.array([u in HOLD_UNDER and w in HOLD_UNDER for u, w in tri_pair], bool)
    cstiff = np.where(limited, np.maximum(1., CURVE_START * np.where(limited, crest_min, 0.) / 10000), 1.)
    edge_max = np.array([GRADE_MAX.get(segments[k]['tags'].get('highway'), GRADE_DEFAULT) for k in seg_of[:-1][same]])
    edge_max = np.maximum(edge_max, np.abs(dem[eb] - dem[ea]) / el + .01)   # a hill street keeps its hill
    stiff = np.ones(len(ea))
    widths = np.clip(np.array([seg.get('width', 4.) for seg in segments]), .5, 80.)
    pa, pe, pt = close_pairs(pos_st, seg_of, widths[seg_of], k_st == 0, same)
    va, vb1, vb2 = ids[pa], ids[pe], ids[pe + 1]
    keep = (va != vb1) & (va != vb2)
    va, vb1, vb2, pt = va[keep], vb1[keep], vb2[keep], pt[keep]
    # a bridge over a road: the deck (not the road under it) reaches clear above that road.
    # A lower road may itself be a bridge (stacked motorway interchanges).
    # Compare vertical levels, not just bridge versus non-bridge: short spans
    # have no interior station and their soft DEM offset alone can be zero.
    levels = kinds * layers
    xa, xb, xu, xt = crossings(pos_st, seg_of, same, k_st > 0, np.ones(len(k_st), bool),
                               widths[seg_of], levels[seg_of])
    xv = (ids[xa], ids[xa + 1], ids[xb], ids[xb + 1])
    apart = levels[seg_of[xa]] > levels[seg_of[xb]]
    for i in range(2):
        for j in range(2, 4):
            apart &= xv[i] != xv[j]                # not a ramp leaving the bridge's own node
    xa, xb, xu, xt = xa[apart], xb[apart], xu[apart], xt[apart]
    # A dual carriageway's two bridges run side by side: each one's deck ribbon overlaps the other
    # direction's approach, whose road is joined to the first by the carriageway's own nodes. Lifting
    # that road to clear the deck lifts the deck again (a feedback of 800 m on the Tallinna ringtee).
    # Only a pair whose centre lines really cross in plan is an overpass.
    if len(xa):
        def centre_lines_cross(u, w):
            a, b = np.asarray(segments[u]['xy'], float), np.asarray(segments[w]['xy'], float)
            p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
            q, t = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]
            cross = r[..., 0] * t[..., 1] - r[..., 1] * t[..., 0]
            d = q - p
            with np.errstate(divide='ignore', invalid='ignore'):
                su = (d[..., 0] * t[..., 1] - d[..., 1] * t[..., 0]) / cross
                sv = (d[..., 0] * r[..., 1] - d[..., 1] * r[..., 0]) / cross
            return bool(np.any((np.abs(cross) > 1e-9) & (su >= 0) & (su <= 1) & (sv >= 0) & (sv <= 1)))
        verdict = {}
        for u, w in set(zip(seg_of[xa].tolist(), seg_of[xb].tolist())):
            verdict[u, w] = centre_lines_cross(u, w)
        genuine = np.array([verdict[u, w] for u, w in zip(seg_of[xa].tolist(), seg_of[xb].tolist())], bool)
        xa, xb, xu, xt = xa[genuine], xb[genuine], xu[genuine], xt[genuine]
    # The deck and the road under it share the clearance, each taking its part: a deck alone
    # lifted the full 6 m over a 24 m DEM that has no cutting (an 11 m hill at every short
    # overpass of the Tartu mnt), and a lower road tied to its own deck then raised it again.
    # Except under a motorway or trunk (HOLD_UNDER): that road keeps its line, the deck rises.
    # One clearance over the road or deck directly under it, whatever the layer numbers: OSM's
    # layer orders crossings, it does not count decks (CLEARANCE x the layer difference put a
    # layer=2 bridge ramp 12 m over a street, which then dipped 5.8 m under the sea level; a deck
    # over another deck clears that one, which clears what is under it).
    xv = (ids[xa], ids[xa + 1], ids[xb], ids[xb + 1])
    xc = (1 - xu, xu, -(1 - xt), -xt)
    separation = np.full(len(xa), CLEARANCE + CLEAR_SLACK)
    held = held_stations(segments, kinds, s, seg_of, run, xb, xt, len(s))
    if held.any():
        mw[ids[held]] *= HOLD_WEIGHT
    if fine_share is not None and len(xb):
        # a measured road of any class keeps its line under a deck, as HOLD_UNDER: the deck takes the clearance
        lower = held_stations(segments, kinds, s, seg_of, run, xb, xt, len(s), classes=None) & ~held
        v = np.setdiff1d(ids[lower], ids[held])
        mw[v] *= 1 + fine_share[v] * (HOLD_WEIGHT - 1)
    def clearance(h):
        return separation + 0 * h[ids[xb]]
    clear = clearance(target)
    # the road under a bridge passes close to the bridge's approaches in plan: they are not
    # one surface, so no coupling ties them (it dragged the trunk down to the underpass)
    if len(xa):
        touching = {}
        for k, seg in enumerate(segments):
            for n in seg['nodes']:
                touching.setdefault(n, set()).add(k)
        def approach(u, hops=2):
            near = {u}
            for _ in range(hops):
                near |= {m for k in near for n in segments[k]['nodes'] for m in touching[n]}
            return near
        apart_of = {}
        for u, w in set(zip(seg_of[xa].tolist(), seg_of[xb].tolist())):
            over, below = approach(u), approach(w) - {u}
            for k in over - below:
                apart_of.setdefault(k, set()).update(below - over)
            for k in below - over:
                apart_of.setdefault(k, set()).update(over - below)
        sa, sb = seg_of[pa[keep]], seg_of[pe[keep]]
        free = np.array([b not in apart_of.get(a, ()) for a, b in zip(sa.tolist(), sb.tolist())], bool)
        va, vb1, vb2, pt = va[free], vb1[free], vb2[free], pt[free]
    # residual h_a - (1-t) h_b1 - t h_b2, weight COUPLE
    cv, cw = (va, vb1, vb2), (np.ones(len(va)), -(1 - pt), -pt)
    trip, coef = (ta, tb, tc), (1., -2., 1.)
    # lower bounds: one floor per variable (the highest), springs outside the CSR matrix like the clearance
    fv, fz = np.zeros(0, int), np.zeros(0)
    if floors:
        parts_v, parts_z = [], []
        for k, s0, s1, z in floors:
            sl = slice(int(bounds[k]), int(bounds[k + 1]))
            m = (s[sl] >= s0 - 1e-6) & (s[sl] <= s1 + 1e-6)
            if not m.any():
                # a span between two stations: the nearer one carries it
                m = np.zeros(sl.stop - sl.start, bool)
                m[int(np.argmin(np.abs(s[sl] - (s0 + s1) / 2)))] = True
            parts_v.append(ids[sl][m]); parts_z.append(np.full(int(m.sum()), float(z)))
        fv, fz = np.concatenate(parts_v), np.concatenate(parts_z)
        o = np.lexsort((-fz, fv))
        fv, fz = fv[o], fz[o]
        first = np.r_[True, fv[1:] != fv[:-1]]
        fv, fz = fv[first], fz[first]
        if not np.isfinite(fz).all():
            raise FloatingPointError('profile: a floor is not finite')
    factive = np.zeros(len(fv))
    faim = fz.copy()

    pattern = None

    def assemble():
        nonlocal pattern
        g = ce * stiff
        rows = [np.arange(nvar), ea, eb, ea, eb]
        cols = [np.arange(nvar), ea, eb, eb, ea]
        vals = [mw, g, g, -g, -g]
        for i in range(3):
            for j in range(3):
                rows.append(cv[i]); cols.append(cv[j]); vals.append(COUPLE * cw[i] * cw[j])
        for i in range(3):
            for j in range(3):
                rows.append(trip[i]); cols.append(trip[j]); vals.append(ct * cstiff * coef[i] * coef[j])
        ptr, col, data, pattern = csr(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), nvar, pattern)
        diag0 = np.bincount(np.repeat(np.arange(nvar), np.diff(ptr)), np.where(
            col == np.repeat(np.arange(nvar), np.diff(ptr)), data, 0.), nvar)
        return ptr, col, data, diag0
    ptr, col, data, diag0 = assemble()
    chains = Chains(ids, run, len(s), nvar, links)
    chains.bands(ptr, col, data)

    # the clearance springs act only where a deck would be too low (a lower bound): on a
    # crossing in an active set, outside the CSR matrix
    active = np.zeros(len(xa))

    def gap(h):
        return sum(c * h[v] for c, v in zip(xc, xv))

    # the springs touch few variables: accumulate on those (a bincount over all stations per term and
    # CG iteration was a quarter of the solve)
    touched = np.unique(np.concatenate(xv)) if len(xa) else np.zeros(0, int)
    local = [np.searchsorted(touched, v) for v in xv]

    def matvec(h, springs=True, matrix=True):
        out = np.add.reduceat(data * h[col], ptr[:-1]) if matrix else np.zeros(nvar)
        if springs and active.any():
            g = CLEAR_WEIGHT * active * gap(h)
            acc = np.zeros(len(touched))
            for c, v in zip(xc, local):
                acc += np.bincount(v, g * c, len(touched))
            out[touched] += acc
        if springs and factive.any():
            out[fv] += FLOOR_WEIGHT * factive * h[fv]
        return out

    def cg(b, x0=None):
        diag = diag0 + sum(np.bincount(v, CLEAR_WEIGHT * active * c * c, nvar) for c, v in zip(xc, xv))
        if factive.any():
            diag = diag.copy()
            diag[fv] += FLOOR_WEIGHT * factive
        # warm start from the last pass: a clearance or grade pass changes the answer only locally
        x = np.zeros(nvar) if x0 is None else x0.copy()
        r = b.copy() if x0 is None else b - matvec(x)
        if chains.factored is None or not np.array_equal(chains.factored, diag):
            chains.factor(diag)
        z = chains.apply(r); p = z.copy(); rz = r @ z
        norm0 = math.sqrt(b @ b) or 1.
        it = 0
        for it in range(iterations):
            if math.sqrt(r @ r) < tol * norm0 or rz <= 0:
                break                              # e.g. a flat DEM is already the answer
            ap = matvec(p)
            alpha = rz / (p @ ap)
            x += alpha * p; r -= alpha * ap
            if math.sqrt(r @ r) < tol * norm0:
                break
            z = chains.apply(r); rz2 = r @ z
            p = z + (rz2 / rz) * p; rz = rz2
        return x, it + 1, math.sqrt(r @ r) / norm0

    # solve for the offset from the DEM: the start residual is small and well scaled. A spring
    # gives part of the clearance only: its target is raised by the shortfall and solved again.
    base = mw * target
    aim = clear.copy()
    ceiling = aim + CLEAR_EXTRA
    steep = 0
    x, total, adem = None, 0, None
    for grade_pass in range(GRADE_PASSES + 1):
        for _ in range(CLEAR_PASSES + 1):
            # the matrix part of A dem is the same in every clearance pass
            if adem is None:
                adem = matvec(dem, springs=False)
            b = base - adem - matvec(dem, matrix=False)
            for c, v in zip(xc, xv):
                b += np.bincount(v, CLEAR_WEIGHT * active * aim * c, nvar)
            if factive.any():
                b[fv] += FLOOR_WEIGHT * factive * faim     # A dem's floor part is in matvec(dem, matrix=False)
            x, its, residual = cg(b, x)
            total += its
            # A lower bridge can itself have risen in this solve. Its DEM target is
            # no longer the road below us: measure the solved surfaces every pass.
            clear = clearance(dem + x)
            short = clear - gap(dem + x)
            fshort = fz - (dem + x)[fv]
            fchange = False
            if len(fv):
                # a floor joins when the solve is under it, aims higher by what it lacks, lets go when held over it
                low = fshort > FLOOR_TOL
                faim = np.minimum(np.where(factive > 0, faim + np.maximum(fshort, 0), fz), fz + FLOOR_EXTRA)
                release = (factive > 0) & (fshort < -FLOOR_RELEASE) & (faim <= fz + 1e-9)
                fchange = bool((low & (factive == 0)).any() or release.any())
                factive = np.where(low, 1., np.where(release, 0., factive))
            if (not len(xa) or short.max() < CLEAR_TOL) and (not len(fv) or (fshort.max() < FLOOR_TOL and not fchange)):
                break
            if not len(xa):
                continue
            # a low crossing joins the set; an active one aims higher by what it still lacks
            # The aim stops CLEAR_EXTRA m over the first one: a crossing whose lower road is tied to
            # the deck (an interchange's ramps) would otherwise chase it up a hill.
            aim = np.minimum(np.where(active > 0, aim + np.maximum(short, 0), clear), ceiling)
            active = np.where(short > CLEAR_TOL, 1., active)
        h = dem + x
        grade = np.abs(h[eb] - h[ea]) / el
        over = grade > edge_max + .002
        steep = int(over.sum())
        # change of grade per metre at each triple: + a sag, - a crest; its limit 1 / radius
        bend = (h[ta] - 2 * h[tb] + h[tc]) / tl ** 2
        limit = np.where(bend > 0, 1 / sag_min, 1 / crest_min)
        sharp = limited & (np.abs(bend) > limit * 1.05)
        tight = int(sharp.sum())
        if (not steep and not tight) or grade_pass == GRADE_PASSES:
            break
        # the grade weight goes with the square of the excess; neighbours take the rest next pass
        # capped, so the clearance springs (CLEAR_WEIGHT) still win where both cannot hold
        stiff[over] = np.minimum(stiff[over] * np.clip(3 * (grade[over] / edge_max[over]) ** 2, 2, 30), STIFF_MAX)
        # the same for a bend over its class's radius, with room to grow further: a crest of 500 m
        # in a 10,000 m class needs the weight a few hundred times over BETA's
        ratio = np.abs(bend[sharp]) / limit[sharp]
        cstiff[sharp] = np.minimum(cstiff[sharp] * np.clip(3 * ratio ** 2, 2, 100), CURVE_STIFF_MAX)
        ptr, col, data, diag0 = assemble()
        chains.bands(ptr, col, data)
        adem = None
    h = dem + x
    if not np.isfinite(h).all():
        raise FloatingPointError(f'profile: {int((~np.isfinite(h)).sum())} heights not finite')
    if emit:
        low = int((short > .5).sum()) if len(xa) else 0
        worst = f', worst {short.max():.2f} m' if low else ''
        share = f' ({100 * (sharp & fast_tri).sum() / max(1, fast_tri.sum()):.2f} % of motorway and trunk triples)'
        floor_note = ''
        if len(fv):
            fs = fz - h[fv]
            floor_note = f', {len(fv)} floored stations ({int((fs > .5).sum())} more than 0.5 m under, worst {max(0., fs.max()):.2f} m)'
        emit(f'profile: {nvar} stations, {len(va)} close station-edge pairs, {len(xa)} bridge crossings '
             f'({low} short of clearance{worst}){floor_note}, {steep} edges over their grade and {tight} of {int(limited.sum())} '
             f'bends of the classes with a radius under it{share} after {grade_pass} grade passes, '
             f'{total} CG iterations in all ({its} in the last), residual {residual:.1e}{rule_note}')
    hs = h[ids]
    bounds = np.r_[run, len(s)]
    profiles = [(s[bounds[k]:bounds[k + 1]], hs[bounds[k]:bounds[k + 1]]) for k in range(len(segments))]
    return profiles, {n: float(h[k]) for n, k in node_index.items()}


class Mosaic:
    """Terrain.sample for millions of points: the PNG tiles round them stitched, BLOCK x BLOCK
    tiles at a time. One grid over all the points' bounding box was 4.7 GiB for Estonia at z13
    (the sea and the empty bogs included), and the solve samples the DEM once."""
    BLOCK = 8

    def __init__(self, terrain, xy=None):
        self.zoom = getattr(terrain, 'zoom', 12)
        self.terrain = terrain

    def _grid(self, tx0, ty0, tx1, ty1, needed):
        """The pixels of tiles tx0..tx1, ty0..ty1, of those in `needed` only (NaN where a tile is
        missing or not asked for). The edge tiles of a downloaded DEM may be missing on disk, and
        z13 upsamples from z12 neighbours: reading just what the points reach keeps to the tiles
        a grid over them read."""
        # a z13 grid has four times the pixels: single precision (millimetres at 1000 m)
        grid = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan,
                       dtype=np.float32 if self.zoom > 12 else float)
        for a, b in sorted(needed):
            if tx0 <= a <= tx1 and ty0 <= b <= ty1:
                if self.terrain.has(a, b):
                    ix = np.arange(a * 256, a * 256 + 256); iy = np.arange(b * 256, b * 256 + 256)
                    gx, gy = np.meshgrid(ix, iy)
                    grid[(b - ty0) * 256:(b - ty0 + 1) * 256,
                         (a - tx0) * 256:(a - tx0 + 1) * 256] = self.terrain.pixels(gx.ravel(), gy.ravel()).reshape(256, 256)
        return grid

    @staticmethod
    def _px(lon, lat, zoom=12):
        scale = 256 * 2 ** zoom
        return (np.asarray(lon) + 180) / 360 * scale, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * scale

    def sample(self, xy, smooth=True, missing=False):
        """Same filter and result as Terrain.sample (smooth 5-tap bilinear)."""
        from fast_sources import LIFT
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lon, lat = self.terrain.frame.geographic(xy[:, 0], xy[:, 1])
        x, y = self._px(lon, lat, self.zoom)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        u, v = x - ix, y - iy
        offsets = [(0, 0, 1.)] if not smooth else [(0, 0, .5), (-1, 0, .125), (1, 0, .125), (0, -1, .125), (0, 1, .125)]
        z = np.zeros(len(x))
        # the points of one block of tiles at a time, in the order of their block key
        bx, by = ix // 256 // self.BLOCK, iy // 256 // self.BLOCK
        key = bx * 65536 + by
        order = np.argsort(key, kind='stable')
        starts = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
        for first, last in zip(starts, np.r_[starts[1:], len(order)]):
            at = order[first:last]
            # a tile of margin each side: the 5-tap filter and the bilinear corner reach one pixel out
            x0, y0 = int(bx[at[0]]) * self.BLOCK - 1, int(by[at[0]]) * self.BLOCK - 1
            # the pixels the filter reads: from one before (ix, iy) to two after, in 256 px tiles
            reach = np.unique(np.concatenate([((ix[at] + dx) // 256) * 65536 + (iy[at] + dy) // 256
                                              for dx in (-1, 2) for dy in (-1, 2)]))
            g = self._grid(x0, y0, x0 + self.BLOCK + 1, y0 + self.BLOCK + 1,
                           [divmod(int(k), 65536) for k in reach])
            gx, gy = ix[at] - x0 * 256, iy[at] - y0 * 256
            for dx, dy, w in offsets:
                a, b = gx + dx, gy + dy
                z[at] += w * ((g[b, a] * (1 - u[at]) + g[b, a + 1] * u[at]) * (1 - v[at])
                              + (g[b + 1, a] * (1 - u[at]) + g[b + 1, a + 1] * u[at]) * v[at])
        if not missing and not np.isfinite(z).all():
            raise FileNotFoundError('no_dem: profile stations outside the downloaded DEM')
        return z + LIFT
