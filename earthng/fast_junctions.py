"""Pack v3 junctions (phase 3, patches/ROAD-OBJECTS.md section 9): one drivable surface per at-grade junction,
built here once for the whole graph so every tile draws the same shape.

- **Clusters.** Junction and crossing nodes of the stroke graph, merged with their neighbours when a stroke
  joins them within MERGE m (dual carriageways, slip-road triangles), and every node of a roundabout ring.
- **Surface.** The paved strips of the cluster's strokes (their exporter extents), each mouth cut where
  its kerb fillets end; between neighbouring mouths a fillet of the kerb radius (KERB, by class and
  context: a morphological closing of the two strips, restricted to the wedge between them). Areas the
  strokes enclose (a slip road's triangle, a divided mouth, the roundabout's centre) stay islands, as do
  mapped ones (area:highway=traffic_island, traffic_calming=island areas).
- **Heights.** The major road's own surface (its cross-fall extended across the junction); each minor
  road blends from it to its own section over its stub, with a C1 smoothstep, so the surface meets every
  mouth's cut section exactly (no steps) and keeps the major road's cross-fall.
- **Mesh.** Split along every centre line (the crowns stay edges), gridded every GRID m, constrained
  Delaunay per piece; z per vertex from the height field, the cut rows forced to the roads' exact
  sections.
- **Approaches.** Per inbound mouth: control (signals, stop, give_way, implied give_way from the classes or
  priority_road, roundabout entry, none), the stop or give-way line position, turn lanes and the zebra,
  signal and cycle crossings on it (red where the tags say so). The game draws the markings from these with
  its own lane block (sweep.lua), so lines, arrows and lanes agree.

Values from the German guides are marked: none of the RASt 06 / RAL 2012 / RMS values below was checked
against the guide text in this session (memory or secondary sources). Never prints positions.
"""
import math
import time
from collections import defaultdict

import numpy as np

FORMAT = 'earthng-junctions-v1'
VERSION = 1
MAGIC = b'EGJ1'
SCALE = 100

MERGE = 30.0           # m of stroke between two junction nodes that are one junction (dual carriageway, slip pair)
SPAN = 70.0            # m: a merged cluster's nodes stay within this of each other (towns: junctions 30 m apart stay two)
RING_SPAN = 160.0      # m: a roundabout ring's nodes, whatever the spacing
REACH = 60.0           # m of each mouth's strip used for the fillets
CUT_SHARE = .45        # a mouth is cut at most this share of the way to its next node
CUT_MAX = 45.0         # m, and at most this far
CUT_MARGIN = .5        # m past the fillet's tangent point
CLEAR = .3             # m: a mouth's cut clears the other roads' paved edges by this
GRID = 3.0             # m: Steiner grid of the mesh (the height field's blend needs it)
DENSE = 1.0            # m: boundary points at most this far apart (the arcs are finer: QUAD per quarter)
QUAD = 32              # segments per quarter circle of a fillet (2.8 degrees each; the check allows 3)
ISLAND_MIN = 4.0       # m2: an enclosed area smaller than this is paved over
ISLAND_ROUND = 1.0     # m: an enclosed island's corners are rounded by this
SLIVER = .25          # m: slivers and spikes of the set operations narrower than twice this go
ISLAND_HEIGHT = .15    # m: raised island (kerb) height. RASt 06 Bordhoehe 0.10-0.15 m: memory, NOT verified
SURFACE_CLASSES = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
                   'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
RING_TAGS = ('roundabout', 'circular')

# Kerb radii (Eckausrundung) by corner class and context, m. NOT verified (from memory of RASt 06 6.3.8 and
# RAL 2012 6.4: built-up main-road corners 8-12 m for a lorry's swept path, side streets 5-8 m, rural
# junctions 12-15 m and more, usually as three-centred curves; one circle stands in here).
KERB = {'urban': {'main': 10., 'minor': 8., 'local': 6., 'service': 4.},
        'rural': {'main': 15., 'minor': 12., 'local': 10., 'service': 6.}}
KERB_SOURCE = 'memory (RASt 06 / RAL 2012), not verified'
MAIN = {'trunk', 'primary', 'secondary'}
MINOR = {'tertiary', 'unclassified'}
LOCAL = {'residential', 'road', 'living_street'}

# Markings positions (the game holds the line widths, sweep.lua M.MARKS). Distances, m; NOT verified.
STOP_BEFORE = .5       # stop / give-way line this far before the junction surface's edge
CROSSING_GAP = 1.0     # m between a stop line and the crossing in front of it
ZEBRA_WIDTH = {'urban': 4.0, 'rural': 3.0}   # Z 293 length along the road: 3-4 m (memory)
FURT_WIDTH = 4.0       # signalled pedestrian crossing (Fussgaengerfurt), memory
CYCLE_WIDTH = 2.0      # cycle crossing (Radfahrerfurt), memory
CONTROL_REACH = 50.0   # m upstream of the cut a signal / stop / give-way node still controls the approach
CROSSING_REACH = 40.0  # m upstream of the cut a crossing is the approach's

RANK = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3, 'unclassified': 2,
        'residential': 2, 'road': 2, 'living_street': 1.5, 'service': 1}


def rank(highway):
    base = highway.replace('_link', '')
    return RANK.get(base, 1) - (.5 if highway.endswith('_link') else 0.)


def corner_class(highway):
    base = highway.replace('_link', '')
    if base in MAIN or base == 'motorway':
        return 'main'
    if base in MINOR:
        return 'minor'
    if base in LOCAL:
        return 'local'
    return 'service'


def built_up(tags):
    v = str(tags.get('maxspeed', '')).lower()
    if 'urban' in v or 'zone' in v or v in ('walk', 'living_street'):
        return True
    try:
        if float(v.split(';')[0]) <= 50:
            return True
    except ValueError:
        pass
    if str(tags.get('zone:traffic', '')).lower().endswith('urban'):
        return True
    if tags.get('highway') in ('residential', 'living_street'):
        return True
    return tags.get('sidewalk') in ('both', 'left', 'right', 'separate') or tags.get('sidewalk:both') in ('yes', 'separate')


def kerb_radius(ha, ta, hb, tb):
    """one corner's kerb radius from its two roads (the lesser class decides, a built-up context the table)"""
    order = ['main', 'minor', 'local', 'service']
    ca, cb = corner_class(ha), corner_class(hb)
    c = order[max(order.index(ca), order.index(cb))]
    ctx = 'urban' if built_up(ta) or built_up(tb) else 'rural'
    return KERB[ctx][c]


# =========================================================================================================
# A stroke as the junction builder sees it (the index's Stroke, or a stroke decoded from a pack)
# =========================================================================================================

class View:
    """arrays s, x, y, z, h, eL, eR, fl, fr; nodes [(node, s)]; parts [(s0, s1, reversed, tags)]; lanes runs
    [s0, s1, f, b, both, width, tf, tb]"""

    def __init__(self, sid, highway, oneway, closed, arrays, nodes, parts, lanes):
        self.id, self.highway, self.oneway, self.closed = sid, highway, oneway, closed
        for k, v in arrays.items():
            setattr(self, k, np.asarray(v, float))
        self.nodes = sorted(nodes, key=lambda p: p[1])
        self.parts = parts
        self.lanes = lanes
        self.L = float(self.s[-1])

    def tags(self, sv):
        for s0, s1, _, t in self.parts:
            if s0 - 1e-6 <= sv <= s1 + 1e-6:
                return t
        return self.parts[0][3] if self.parts else {}

    def reversed_at(self, sv):
        for s0, s1, rev, _ in self.parts:
            if s0 - 1e-6 <= sv <= s1 + 1e-6:
                return rev
        return False

    def run(self, sv):
        for r in self.lanes:
            if r[0] - 1e-6 <= sv <= r[1] + 1e-6:
                return r
        return None

    def at(self, sv, key):
        return float(np.interp(sv, self.s, getattr(self, key)))

    def frame(self, sv):
        x, y, h = self.at(sv, 'x'), self.at(sv, 'y'), self.at(sv, 'h')
        return x, y, math.cos(h), math.sin(h)

    def section(self, sv):
        """(left, centre, right) points at s (z on the section: centre z + cross-fall x extent)"""
        x, y, z, h = self.at(sv, 'x'), self.at(sv, 'y'), self.at(sv, 'z'), self.at(sv, 'h')
        eL, eR, fl, fr = self.at(sv, 'eL'), self.at(sv, 'eR'), self.at(sv, 'fl'), self.at(sv, 'fr')
        nx, ny = -math.sin(h), math.cos(h)
        return ((x + nx * eL, y + ny * eL, z + fl * eL), (x, y, z), (x - nx * eR, y - ny * eR, z + fr * eR))

    def ring(self):
        t = self.parts[0][3] if self.parts else {}
        return t.get('junction') in RING_TAGS


def view_of_stroke(st, segments):
    """a View of fast_roads.Stroke (the index build)"""
    parts = [(float(st.part_s[p]), float(st.part_s[p + 1]), bool(rev), segments[k]['tags']) for p, (k, rev) in enumerate(st.parts)]
    arrays = {k: getattr(st, k) for k in ('s', 'x', 'y', 'z', 'h', 'eL', 'eR', 'fl', 'fr')}
    return View(st.id, st.highway, st.oneway, st.closed, arrays, [(str(n), float(sv)) for n, sv, _ in st.nodes], parts,
                [list(r) for r in st.lanes])


def view_of_chunks(chunks):
    """a View from a stroke's decoded pack chunks (fast_roads.decode_cell), all of them, in index order"""
    chunks = sorted(chunks, key=lambda c: c['index'])
    cols = {k: [] for k in ('s', 'x', 'y', 'z', 'h', 'eL', 'eR', 'fl', 'fr')}
    nodes, parts, lanes, seen = [], [], [], set()
    for c in chunks:
        first = not cols['s']
        for k in cols:
            cols[k] += c['stations'][k] if first else c['stations'][k][1:]
        for n, sv in c['nodes']:
            if (n, sv) not in seen:
                seen.add((n, sv)); nodes.append((n, sv))
        for p in c['parts']:
            key = (p['segment'], p['s0'])
            if key not in seen:
                seen.add(key); parts.append((p['s0'], p['s1'], p['reversed'], p['tags']))
        for r in c['lanes']:
            key = ('lane', r[0], r[1])
            if key not in seen:
                seen.add(key); lanes.append(list(r))
    c0 = chunks[0]
    parts.sort(key=lambda p: p[0])
    lanes.sort(key=lambda r: r[0])
    return View(c0['stroke'], c0['highway'], c0['oneway'], c0['closed'], cols, nodes, parts, lanes)


# =========================================================================================================
# Clusters
# =========================================================================================================

def clusters(records, views):
    """[[node ids]] of the junction clusters (records: node rows' data, views: {stroke id: View})"""
    rec = {r['node']: r for r in records}
    xy = {n: (r['xyz'][0], r['xyz'][1]) for n, r in rec.items()}

    def fitted(n):
        return any(views[m[0]].highway in SURFACE_CLASSES for m in rec[n]['mouths'] if m[0] in views)

    def motorway(n):
        return any(views[m[0]].highway == 'motorway' for m in rec[n]['mouths'] if m[0] in views)
    seeds = {n for n, r in rec.items() if r['kind'] in ('junction', 'cross') and fitted(n) and not motorway(n)}
    joinable = seeds | {n for n, r in rec.items() if r['kind'] in ('fork', 'diverge', 'merge') and fitted(n) and not motorway(n)}
    parent = {n: n for n in joinable}
    members = {n: [n] for n in joinable}
    ring_group = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b, span):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        pts = members[ra] + members[rb]
        P = np.asarray([xy[n] for n in pts])
        if len(P) > 1 and float(np.max(np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1)))) > span:
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] = pts
        del members[rb]
    # roundabout rings: every junction node of the ring strokes, joined through shared nodes
    for v in views.values():
        if v.ring():
            ring_nodes = [n for n, _ in v.nodes if n in rec and rec[n]['kind'] not in ('end', 'joint')]
            for n in ring_nodes:
                if n not in parent:
                    parent[n] = n; members[n] = [n]; joinable.add(n); seeds.add(n)
                ring_group[n] = True
            for a, b in zip(ring_nodes, ring_nodes[1:]):
                union(a, b, RING_SPAN)
    # short stroke pieces between junction nodes, shortest first
    edges = []
    for v in views.values():
        ns = [(n, sv) for n, sv in v.nodes if n in joinable]
        for (a, sa), (b, sb) in zip(ns, ns[1:]):
            if a != b and abs(sb - sa) <= MERGE and (a in seeds or b in seeds):
                edges.append((abs(sb - sa), a, b))
    for _, a, b in sorted(edges):
        union(a, b, RING_SPAN if (a in ring_group and b in ring_group) else SPAN)
    out = []
    for root, ms in members.items():
        if any(n in seeds for n in ms):
            out.append(sorted(ms))
    out.sort()
    return out


# =========================================================================================================
# Geometry of one cluster
# =========================================================================================================

def _strip(v, sa, sb, step=DENSE):
    """the paved strip of v between s sa and sb (any order) as a shapely polygon"""
    import shapely
    a, b = min(sa, sb), max(sa, sb)
    n = max(2, int(math.ceil((b - a) / step)) + 1)
    sv = np.linspace(a, b, n)
    x, y, h = np.interp(sv, v.s, v.x), np.interp(sv, v.s, v.y), np.interp(sv, v.s, v.h)
    eL, eR = np.interp(sv, v.s, v.eL), np.interp(sv, v.s, v.eR)
    nx, ny = -np.sin(h), np.cos(h)
    left = np.column_stack([x + nx * eL, y + ny * eL])
    right = np.column_stack([x - nx * eR, y - ny * eR])
    ring = np.vstack([right, left[::-1]])
    poly = shapely.Polygon(ring)
    if not poly.is_valid:
        poly = shapely.make_valid(poly).buffer(0)
    if poly.is_empty or poly.area < 1e-4:
        # a collapsed strip (zero extent or a zero-length range): buffer(0) of its line is empty, and the distance to
        # an empty polygon is NaN (the 'non-finite junction number' of 229 junctions); a thin strip along the centre
        # line stands in
        width = max(.05, .5 * float(np.max(eL + eR)))
        centre = np.column_stack([x, y])
        if len(np.unique(centre.round(6), axis=0)) < 2:
            poly = shapely.Point(centre[0]).buffer(width)
        else:
            poly = shapely.LineString(centre).buffer(width, cap_style='flat')
    return poly


def _centre(v, sa, sb, step=DENSE):
    import shapely
    a, b = min(sa, sb), max(sa, sb)
    n = max(2, int(math.ceil((b - a) / step)) + 1)
    sv = np.linspace(a, b, n)
    return shapely.LineString(np.column_stack([np.interp(sv, v.s, v.x), np.interp(sv, v.s, v.y)]))


def _project(v, P, lo, hi):
    """(s, u) of points P (n x 2) on v's centre line within s range [lo, hi] (u + left)"""
    lo, hi = max(0., lo), min(v.L, hi)
    i0 = max(0, int(np.searchsorted(v.s, lo)) - 1)
    i1 = min(len(v.s), int(np.searchsorted(v.s, hi)) + 1)
    sv = np.unique(np.r_[lo, v.s[i0:i1][(v.s[i0:i1] > lo) & (v.s[i0:i1] < hi)], hi])
    if len(sv) < 2:
        sv = np.array([lo, max(hi, lo + .01)])
    Q = np.column_stack([np.interp(sv, v.s, v.x), np.interp(sv, v.s, v.y)])
    a, b = Q[:-1], Q[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - a[None]) * d[None]).sum(2) / L2[None], 0, 1)
    c = a[None] + t[..., None] * d[None]
    dist = ((P[:, None, :] - c) ** 2).sum(2)
    j = dist.argmin(1)
    k = np.arange(len(P))
    s = sv[j] + t[k, j] * (sv[j + 1] - sv[j])
    h = np.interp(s, v.s, v.h)
    x, y = np.interp(s, v.s, v.x), np.interp(s, v.s, v.y)
    u = (P[:, 0] - x) * -np.sin(h) + (P[:, 1] - y) * np.cos(h)
    return s, u


def _plane(v, P, lo, hi):
    """z of v's own section (its cross-fall extended past the edges) at points P, projected within [lo, hi]"""
    s, u = _project(v, P, lo, hi)
    z = np.interp(s, v.s, v.z)
    f = np.where(u >= 0, np.interp(s, v.s, v.fl), np.interp(s, v.s, v.fr))
    return z + f * np.abs(u), s, u


def _smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3 - 2 * t)


def _wedge(cx, cy, a0, a1, r=400.):
    import shapely
    span = (a1 - a0) % (2 * math.pi)
    if span < 1e-6:
        span = 2 * math.pi
    n = max(2, int(math.ceil(span / math.radians(20))) + 1)
    pts = [(cx, cy)] + [(cx + r * math.cos(a0 + span * k / (n - 1)), cy + r * math.sin(a0 + span * k / (n - 1))) for k in range(n)]
    return shapely.Polygon(pts)


def _polys(geom):
    """the polygonal parts of a geometry, as a list"""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    return [g for part in getattr(geom, 'geoms', []) for g in _polys(part)]


def _largest(geom):
    import shapely
    if geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        return geom
    polys = [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon']
    if not polys:
        return shapely.Polygon()
    return max(polys, key=lambda g: g.area)


NODE_TOL = .05          # m: a node this close to a mouth's s is the mouth's own node


def junction(cluster, records, views, extras=None):
    """the junction record of one cluster, or None (nothing drawable)"""
    import shapely
    from shapely import ops
    rec = {r['node']: r for r in records}
    members = set(cluster)
    lane_roles = set()
    for n in cluster:
        for sid, role in rec[n].get('roles', {}).items():
            if role[0] == 'lane':
                lane_roles.add((n, sid))
    # mouths and internal pieces
    mouths, internal = [], []
    for n in cluster:
        for sid, sv, d in rec[n]['mouths']:
            v = views.get(sid)
            if v is None or (n, sid) in lane_roles:
                continue
            # the record's mouth s is rounded to the centimetre, the stroke's node s is not: the node itself must not
            # count as the next node (it did at 3 mm: the through roads of a crossing lost their mouths)
            nxt = [(m, s2) for m, s2 in v.nodes if (s2 - sv) * d > NODE_TOL]
            if v.closed and not nxt:
                nxt = [(m, s2 + (v.L if d > 0 else -v.L)) for m, s2 in v.nodes if abs(s2 - sv) > NODE_TOL]
            if nxt:
                m2, s2 = min(nxt, key=lambda p: abs(p[1] - sv))
            else:
                m2, s2 = None, (v.L if d > 0 else 0.)
            if m2 in members:
                a, b = sorted((sv, min(max(s2, 0.), v.L)))
                if b - a > 1e-3:
                    internal.append((sid, round(a, 3), round(b, 3)))
            else:
                reach = abs(s2 - sv)
                if reach > .5:
                    mouths.append({'stroke': sid, 'node': n, 's': float(sv), 'dir': int(d), 'reach': reach, 'v': v})
    internal = sorted(set(internal))
    if len(mouths) < 2 and not internal:
        return None
    if len({m['stroke'] for m in mouths} | {p[0] for p in internal}) < 2:
        return None
    pts = np.asarray([rec[n]['xyz'] for n in cluster], float)
    cx, cy, cz = (float(v) for v in pts.mean(0))
    ring = any(views[sid].ring() for sid, _, _ in internal)
    ring_kind = None
    if ring:
        for sid, _, _ in internal:
            if views[sid].ring():
                ring_kind = views[sid].parts[0][3].get('junction')
    # majors: the highest-ranked strokes that pass through (a ring is always the major)
    through = defaultdict(int)
    for m in mouths:
        through[m['stroke']] += 1
    for sid, _, _ in internal:
        through[sid] += 2
    def key(sid):
        v = views[sid]
        width = v.at(v.L / 2, 'eL') + v.at(v.L / 2, 'eR')
        return (10 if v.ring() else 0) + rank(v.highway), width
    sids = sorted(through, key=lambda s_: (key(s_), s_), reverse=True)
    top = key(sids[0])[0]
    majors = [s_ for s_ in sids if key(s_)[0] == top and through[s_] >= 2] or [sids[0]]
    # long strips per mouth and the internal strips
    for m in mouths:
        v = m['v']
        m['long'] = min(.95 * m['reach'], REACH)
        m['maxcut'] = max(.5, min(CUT_SHARE * m['reach'], CUT_MAX))
        m['strip_long'] = _strip(v, m['s'], m['s'] + m['dir'] * m['long'])
        x, y, tx, ty = v.frame(m['s'] + m['dir'] * min(6., m['long']))
        m['ux'], m['uy'] = tx * m['dir'], ty * m['dir']
        px, py = v.at(m['s'] + m['dir'] * min(8., m['long']), 'x'), v.at(m['s'] + m['dir'] * min(8., m['long']), 'y')
        m['angle'] = math.atan2(py - cy, px - cx)
        m['tags'] = v.tags(m['s'] + m['dir'] * .5)
    istrips = [_strip(views[sid], a, b) for sid, a, b in internal]
    U = shapely.unary_union([m['strip_long'] for m in mouths] + istrips)
    holes = []
    for g in (U.geoms if hasattr(U, 'geoms') else [U]):
        if g.geom_type == 'Polygon':
            holes += [shapely.Polygon(r) for r in g.interiors]
    holes = [h for h in holes if h.area >= ISLAND_MIN and h.distance(shapely.Point(cx, cy)) < SPAN]
    hole_union = shapely.unary_union(holes) if holes else shapely.Polygon()
    # fillets between neighbouring mouths
    mouths.sort(key=lambda m: (m['angle'], m['stroke'], m['dir']))
    corners, radii = [], []
    n_m = len(mouths)
    for k in range(n_m if n_m > 1 else 0):
        a, b = mouths[k], mouths[(k + 1) % n_m]
        if n_m == 2 and k == 1 and a is b:
            continue
        span = (b['angle'] - a['angle']) % (2 * math.pi)
        theta = math.acos(max(-1., min(1., a['ux'] * b['ux'] + a['uy'] * b['uy'])))
        if span > math.pi - math.radians(8) and theta > math.pi - math.radians(8):
            continue                        # straight on: no corner
        if theta < math.radians(5):
            continue                        # parallel mouths (a divided road): the island between stays
        R = kerb_radius(a['v'].highway, a['tags'], b['v'].highway, b['tags'])
        half = max(theta, 1e-3) / 2
        cap = min(a['maxcut'], b['maxcut']) - CUT_MARGIN
        R = min(R, max(0., cap) * math.tan(half))
        if R < .75:
            continue
        region = shapely.unary_union([a['strip_long'], b['strip_long']] + istrips)
        closed = region.buffer(R, quad_segs=QUAD).buffer(-R, quad_segs=QUAD)
        corner = closed.difference(U).intersection(_wedge(cx, cy, a['angle'], b['angle']))
        if not hole_union.is_empty:
            corner = corner.difference(hole_union)
        corner = shapely.MultiPolygon([g for g in _polys(corner) if g.area > .01])
        if corner.area > .05:
            corners.append((k, corner))
            radii.append(R)
    # cuts: past the fillets, clear of the other roads
    for k, m in enumerate(mouths):
        v = m['v']
        others = [mm['strip_long'] for mm in mouths if mm is not m and mm['stroke'] != m['stroke']]
        others += [p for p, (sid, _, _) in zip(istrips, internal) if sid != m['stroke']]
        other = shapely.unary_union(others).buffer(CLEAR) if others else shapely.Polygon()
        need = 0.
        d = .25
        while d <= m['maxcut'] and not other.is_empty:
            L, _, R_ = v.section(m['s'] + m['dir'] * d)
            if not (other.contains(shapely.Point(L[0], L[1])) or other.contains(shapely.Point(R_[0], R_[1]))):
                need = d
                break
            d += .25
        else:
            need = m['maxcut'] if not other.is_empty else 0.
        ext = 0.
        for kk, corner in corners:
            if kk == k or (kk + 1) % n_m == k:
                P = np.asarray([c for g in corner.geoms for c in g.exterior.coords])
                if len(P):
                    s_, _ = _project(v, P, min(m['s'], m['s'] + m['dir'] * m['long']), max(m['s'], m['s'] + m['dir'] * m['long']))
                    ext = max(ext, float(np.max((s_ - m['s']) * m['dir'])) + CUT_MARGIN)
        m['cut'] = round(min(m['maxcut'], max(need, ext, 1.0)), 2)
        m['clear'] = min(m['cut'], max(need, 1.0))   # where the mouth leaves the other roads' paved area (their edge + CLEAR)
        m['sc'] = m['s'] + m['dir'] * m['cut']
        m['L'], m['C'], m['R'] = v.section(m['sc'])
    # the surface: strips to the cuts, internal strips, fillets; clipped to the hull of the cut rows
    clip = [_strip(m['v'], m['s'], m['sc']) for m in mouths]
    J = shapely.unary_union(clip + istrips + [c for _, c in corners])
    J = J.buffer(.002, quad_segs=2).buffer(-.002, quad_segs=2)
    hull_pts = [(p[0], p[1]) for m in mouths for p in (m['L'], m['R'])] + [tuple(p[:2]) for p in pts]
    for sid, a, b in internal:
        hull_pts += [(p[0], p[1]) for sv in (a, b) for p in views[sid].section(sv)[::2]]
    hull = shapely.MultiPoint(hull_pts).convex_hull
    J = J.intersection(hull.buffer(.001, join_style='mitre'))
    for m in mouths:
        beyond = _strip(m['v'], m['sc'], m['s'] + m['dir'] * max(m['long'], m['cut'] + 1.))
        J = J.difference(beyond)
    # spikes and slivers of the set operations go (an opening), the cut rows' exact corners come back
    rows_near = [_strip(m['v'], m['s'] + m['dir'] * max(0., m['cut'] - 1.), m['sc']) for m in mouths]
    J = shapely.unary_union([J.buffer(-SLIVER, quad_segs=QUAD).buffer(SLIVER, quad_segs=QUAD)] + [J.intersection(r) for r in rows_near])
    islands = []
    for h in holes:
        isl = h.buffer(-ISLAND_ROUND).buffer(ISLAND_ROUND, quad_segs=8)
        if isl.area >= ISLAND_MIN:
            islands.append(('kerb' if isl.area < 400 else 'grass', _largest(isl)))
    for wid, tags, xy in (extras or {}).get('islands', ()):
        if len(xy) >= 3:
            poly = shapely.Polygon(xy)
            if poly.is_valid and poly.area >= 1. and poly.intersects(J):
                islands.append(('kerb', poly.intersection(J.buffer(.5))))
    for _, poly in islands:
        J = J.difference(poly)
    # every part that holds a road's strip (a cluster's junctions apart stay one record)
    keep = shapely.unary_union(clip + istrips)
    parts = [g for g in _polys(J) if g.area > .5 and g.intersects(keep)]
    J = shapely.MultiPolygon(parts) if len(parts) > 1 else (parts[0] if parts else shapely.Polygon())
    islands = [(k_, _largest(p_)) for k_, p_ in islands if not p_.is_empty]
    if J.is_empty or J.area < 4.:
        return None
    J = J.simplify(.005)
    # the mesh: split along the centre lines, gridded, constrained Delaunay per piece
    lines = []
    for m in mouths:
        lines.append(_centre(m['v'], m['s'] - m['dir'] * 2., m['sc'] + m['dir'] * 2.))
    for sid, a, b in internal:
        lines.append(_centre(views[sid], max(0., a - 2.), min(views[sid].L, b + 2.)))
    pieces = _polys(J)
    for ln in lines:
        nxt = []
        for p in pieces:
            try:
                got = ops.split(p, ln)
                nxt += [g for g in got.geoms if g.geom_type == 'Polygon' and g.area > 1e-4]
            except Exception:
                nxt.append(p)
        pieces = nxt
    major = views[majors[0]]
    hm = major.at(_nearest_s(major, cx, cy), 'h')
    hx, hy = math.cos(hm), math.sin(hm)
    tris_xy = []
    x0, y0, x1, y1 = J.bounds
    rad = math.hypot(x1 - x0, y1 - y0)
    nG = int(math.ceil(rad / GRID)) + 1
    cells = []
    for i in range(-nG, nG):
        for j in range(-nG, nG):
            a0, b0 = i * GRID, j * GRID
            q = [(a0, b0), (a0 + GRID, b0), (a0 + GRID, b0 + GRID), (a0, b0 + GRID)]
            cells.append(shapely.Polygon([(cx + u * hx - w * hy, cy + u * hy + w * hx) for u, w in q]))
    grid = shapely.STRtree(cells)
    for p in pieces:
        for idx in grid.query(p, predicate='intersects'):
            g = p.intersection(cells[idx])
            for poly in (g.geoms if hasattr(g, 'geoms') else [g]):
                if poly.geom_type != 'Polygon' or poly.area < 1e-5:
                    continue
                t = shapely.constrained_delaunay_triangles(poly)
                for tri in t.geoms:
                    c = list(tri.exterior.coords)[:3]
                    if shapely.area(tri) > 1e-6:
                        tris_xy.append(c)
    if not tris_xy:
        return None
    # weld at 1 mm
    index, verts, tris = {}, [], []
    for tri in tris_xy:
        ids = []
        for x, y in tri:
            k_ = (int(round(x * 1000)), int(round(y * 1000)))
            if k_ not in index:
                index[k_] = len(verts)
                verts.append((x, y))
            ids.append(index[k_])
        if len(set(ids)) == 3:
            a, b, c = (verts[i] for i in ids)
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                ids = [ids[0], ids[2], ids[1]]
            tris.append(ids)
    V = np.asarray(verts, float)
    Z = height_field(V, mouths, internal, majors, views)
    if not np.all(np.isfinite(Z)):
        raise ValueError('non-finite junction height')       # counted as failed by _batch, the pack goes on
    # the cut rows exactly: vertices on a row segment take its linear z
    on_row = np.zeros(len(V), bool)
    for m in mouths:
        for P, Q in ((m['R'], m['C']), (m['C'], m['L'])):
            P2, Q2 = np.asarray(P[:2]), np.asarray(Q[:2])
            d = Q2 - P2
            l2 = float(d @ d)
            if l2 < 1e-8:
                continue
            t = np.clip(((V - P2) @ d) / l2, 0, 1)
            dist = np.hypot(*(V - (P2 + t[:, None] * d)).T)
            hit = dist < .01
            Z[hit] = P[2] + (Q[2] - P[2]) * t[hit]
            on_row |= hit
    # kerb edges: boundary edges off the cut rows, oriented with the surface on their left
    count = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            count[(min(e), max(e))] += 1
    kerbs = []
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            if count[(min(e), max(e))] == 1 and not (on_row[e[0]] and on_row[e[1]]):
                kerbs.append(e)
    # islands: z from the field under them
    isl_out = []
    for kind, poly in islands:
        if poly.is_empty or poly.geom_type != 'Polygon':
            continue
        P = np.asarray(_densify(poly, 2.).exterior.coords)[:-1]
        if len(P) < 3:
            continue
        zi = height_field(P, mouths, internal, majors, views)
        isl_out.append({'kind': kind, 'height': ISLAND_HEIGHT, 'ring': [[round(float(x), 3), round(float(y), 3), round(float(z), 3)]
                                                                      for (x, y), z in zip(P, zi)]})
    record = {
        'id': 'j' + cluster[0], 'kind': ring_kind or 'junction', 'nodes': list(cluster),
        'xyz': [round(cx, 3), round(cy, 3), round(cz, 3)], 'majors': majors,
        'mouths': [{'stroke': m['stroke'], 'node': m['node'], 'dir': m['dir'], 's': round(m['s'], 2), 'cut': m['cut'],
                    'long': round(max(m['long'], m['cut'] + 1.), 2),
                    'role': 'major' if m['stroke'] in majors else 'minor',
                    'L': [round(v_, 3) for v_ in m['L']], 'C': [round(v_, 3) for v_ in m['C']], 'R': [round(v_, 3) for v_ in m['R']]}
                   for m in mouths],
        'internal': [[sid, a, b] for sid, a, b in internal],
        'verts': [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for (x, y), z in zip(V, Z)],
        'tris': tris, 'kerbs': kerbs, 'islands': isl_out, 'radii': [round(r, 2) for r in radii],
        'area': round(J.area, 1)}
    record['approaches'], record['signals'] = approaches(record, mouths, majors, views, extras, ring_kind)
    return record


def _nearest_s(v, x, y):
    i = int(np.argmin((v.x - x) ** 2 + (v.y - y) ** 2))
    return float(v.s[i])


def _densify(poly, step):
    import shapely
    return shapely.segmentize(poly, step)


def _finite_d(d):
    """distances with NaN (an empty geometry) read as far away, so a weight is never NaN"""
    d = np.asarray(d, float)
    return np.where(np.isfinite(d), d, 1e4)


def height_field(P, mouths, internal, majors, views):
    """z at points P: the majors' own sections (several: the nearest by inverse distance), each minor mouth
    blending from it to its own section over its stub (C1)"""
    import shapely
    P = np.asarray(P, float)
    n = len(P)
    pts = shapely.points(P[:, 0], P[:, 1])

    def ranges(sid):
        rs = [(a, b) for s_, a, b in internal if s_ == sid]
        rs += [(min(m['s'], m['s'] + m['dir'] * m['long']), max(m['s'], m['s'] + m['dir'] * m['long'])) for m in mouths if m['stroke'] == sid]
        return rs or [(0., views[sid].L)]
    zM = np.zeros(n)
    wM = np.zeros(n)
    major_area = []
    for sid in majors:
        v = views[sid]
        best_z, best_d = None, None
        for lo, hi in ranges(sid):
            strip = _strip(v, lo, hi)
            major_area.append(strip)
            z, _, _ = _plane(v, P, lo, hi)
            d = _finite_d(shapely.distance(pts, strip))
            if best_d is None:
                best_z, best_d = z, d
            else:
                take = d < best_d
                best_z, best_d = np.where(take, z, best_z), np.minimum(d, best_d)
        w = 1. / (best_d + .05) ** 4
        zM += w * best_z
        wM += w
    zM /= np.maximum(wM, 1e-30)
    area = shapely.unary_union(major_area)
    dM = _finite_d(shapely.distance(pts, area))
    num, den = np.zeros(n), np.zeros(n)
    for m in mouths:
        if m['stroke'] in majors:
            continue
        v = m['v']
        lo, hi = sorted((m['s'], m['s'] + m['dir'] * m['long']))
        z, s_, _ = _plane(v, P, lo, hi)
        strip = _strip(v, lo, hi)
        d = _finite_d(shapely.distance(pts, strip))
        # where the mouth's centre line leaves the major area, and its cut
        a0 = 0.
        for t in np.arange(0., m['cut'] + 1e-6, .25):
            x_, y_, _, _ = v.frame(m['s'] + m['dir'] * t)
            if area.distance(shapely.Point(x_, y_)) > 1e-3:
                a0 = t
                break
        else:
            a0 = m['cut']
        span = max(1., m['cut'] - a0)
        a = (s_ - m['s']) * m['dir']
        w_own = _smooth((a - a0) / span) * _smooth(dM / 1.0)
        psi = 1. / (d + .05) ** 4
        num += psi * w_own * (z - zM)
        den += psi
    return zM + np.where(den > 0, num / np.maximum(den, 1e-30), 0.)


# =========================================================================================================
# Approaches: control, stop line, crossings, turn lanes
# =========================================================================================================

def _direction_ok(tags, forward_travel):
    """does a node's direction tag (traffic_signals:direction, direction) apply to traffic travelling
    forward (along the way's node order) or backward"""
    d = str(tags.get('traffic_signals:direction', tags.get('direction', ''))).lower()
    if d in ('forward',):
        return forward_travel
    if d in ('backward',):
        return not forward_travel
    return True


def approaches(record, mouths, majors, views, extras, ring_kind):
    """the inbound approaches of a junction record (and the crossings of the one-way roads leaving it, as approaches
    without control), and whether it is signalled"""
    controls = (extras or {}).get('controls', ())
    crossings = (extras or {}).get('crossings', ())
    nodes = set(record['nodes'])
    cluster_signals = any(str(nid) in nodes and tags.get('highway') == 'traffic_signals' for nid, tags, _, _ in controls)
    major_rank = max(rank(views[s_].highway) for s_ in majors)
    priority = any(str(views[s_].parts[0][3].get('priority_road', '')) in ('designated', 'yes_unposted') for s_ in majors)
    out, leaving = [], set()
    near = []
    cx, cy = record['xyz'][0], record['xyz'][1]
    for nid, tags, x, y in controls:
        if math.hypot(x - cx, y - cy) < CONTROL_REACH + CUT_MAX + 40:
            near.append((nid, tags, x, y))
    near_x = []
    for wid, tags, xy in crossings:
        P = np.asarray(xy, float)
        if len(P) >= 2 and np.min(np.hypot(P[:, 0] - cx, P[:, 1] - cy)) < CROSSING_REACH + CUT_MAX + 40:
            near_x.append((wid, tags, P))
    # crossing nodes on the road itself (highway=crossing, crossing=*): a short line across it
    for nid, tags, x, y in near:
        if tags.get('highway') == 'crossing' or 'crossing' in tags:
            near_x.append((nid, tags, None, x, y))
    for m in mouths:
        v = m['v']
        # inbound: traffic travels towards the node along this mouth
        inbound_forward = m['dir'] < 0           # travelling in +s arrives at the node from s < node s
        # a one-way road leaving the junction: no line and no control, only the crossings on it (a signalled junction's
        # Furt across the outbound carriageway of a dual carriageway was not drawn)
        outbound = v.oneway and not inbound_forward
        control, own_signal = None, False
        for nid, tags, x, y in ([] if outbound else near):
            s_, u = _project(v, np.asarray([[x, y]]), min(m['s'], m['s'] + m['dir'] * m['long']), max(m['s'], m['s'] + m['dir'] * m['long']))
            dist_along = (s_[0] - m['s']) * m['dir']
            if abs(u[0]) > max(v.at(s_[0], 'eL'), v.at(s_[0], 'eR')) + 1.5 or dist_along > m['cut'] + CONTROL_REACH or dist_along < -1.:
                continue
            fwd = inbound_forward != v.reversed_at(s_[0])     # the approach's travel along the way's node order
            hw = tags.get('highway')
            if not _direction_ok(tags, fwd):
                continue
            if hw == 'traffic_signals':
                own_signal = True
            elif hw == 'stop' and control is None:
                control = 'stop'
            elif hw == 'give_way' and control is None:
                control = 'give_way'
        if outbound:
            control = 'none'
        elif own_signal or cluster_signals:
            control = 'signals'
        m['_own_signal'] = own_signal
        if control is None:
            if ring_kind == 'roundabout' and m['stroke'] not in majors:
                control = 'give_way'
            elif m['stroke'] not in majors and (priority or (major_rank >= RANK['secondary'] and rank(v.highway) < major_rank)):
                control = 'give_way'
            else:
                control = 'none'
        # crossings on the approach
        xs = []
        lo_, hi_ = min(m['s'], m['s'] + m['dir'] * m['long']), max(m['s'], m['s'] + m['dir'] * m['long'])
        c = _centre(v, lo_, hi_)
        for item in near_x:
            import shapely
            if item[2] is None:
                _, tags, _, x, y = item
                s_, u_ = _project(v, np.asarray([[x, y]]), lo_, hi_)
                if abs(u_[0]) > 2.5:
                    continue
            else:
                _, tags, P = item
                hit = c.intersection(shapely.LineString(P))
                if hit.is_empty or hit.geom_type not in ('Point', 'MultiPoint'):
                    continue
                hp = hit if hit.geom_type == 'Point' else hit.geoms[0]
                s_, _ = _project(v, np.asarray([[hp.x, hp.y]]), lo_, hi_)
            along = (s_[0] - m['s']) * m['dir']
            if along > m['cut'] + CROSSING_REACH or along < 0:
                continue
            kind, red = crossing_kind(tags)
            if kind is None:
                continue
            width = _width(tags, kind, built_up(m['tags']))
            # a crossing mapped inside the junction surface sits just outside it
            along = max(along, m['cut'] + width / 2 + .3)
            xs.append({'s': round(m['s'] + m['dir'] * along, 2), 'width': round(width, 2), 'kind': kind, 'red': bool(red)})
        xs.sort(key=lambda c: (c['s'] - m['s']) * m['dir'])
        # a crossing mapped twice (its way and its node) is one
        kept = []
        for c in xs:
            if kept and abs(c['s'] - kept[-1]['s']) < 2.:
                if kept[-1]['kind'] in ('marked', 'signals') and c['kind'] not in ('marked', 'signals'):
                    kept[-1] = c
                continue
            kept.append(c)
        xs = kept
        if outbound:
            if not xs:
                continue
            out.append({'stroke': m['stroke'], 'dir': m['dir'], 'stop': round(m['s'] + m['dir'] * m['cut'], 2), 'control': 'none',
                        'turns': '', 'lanes': 1, 'crossings': xs})
            leaving.add(len(out) - 1)
            continue
        # the stop / give-way line: before the junction, or before the crossing next to it. A give-way or stop line stands
        # STOP_BEFORE before the edge of the road it joins (the Wartelinie in line with that road's carriageway edge: memory
        # of the RMS, NOT verified), not before the end of the kerb fillets (cut): it stood a kerb radius back, 11 m from a
        # roundabout's ring (synthetic R 18 m ring, secondary arms). A signalled junction's stop line keeps its place.
        stop_along = (m['clear'] if control in ('give_way', 'stop') else m['cut']) + STOP_BEFORE
        for c in xs:
            near_edge = (c['s'] - m['s']) * m['dir'] - c['width'] / 2
            if near_edge - stop_along < 6.:
                stop_along = (c['s'] - m['s']) * m['dir'] + c['width'] / 2 + CROSSING_GAP
        run = v.run(m['s'] + m['dir'] * stop_along)
        if run is None:
            turns, lanes = '', 1
        else:
            turns = run[6] if inbound_forward else run[7]
            lanes = (run[2] if inbound_forward else run[3]) + run[4]
        out.append({'stroke': m['stroke'], 'dir': m['dir'], 'stop': round(m['s'] + m['dir'] * stop_along, 2),
                    'control': control, 'turns': turns or '', 'lanes': int(max(1, lanes)), 'crossings': xs})
    signals = any(a['control'] == 'signals' for a in out)
    if signals:
        for k, a in enumerate(out):
            if k not in leaving:
                a['control'] = 'signals'
    return out, signals


def crossing_kind(tags):
    """(kind, red) of a crossing way or node: zebra, signals (a signalled pedestrian crossing), marked, cycle,
    both (pedestrian and cycle side by side), or None (unmarked)"""
    c = str(tags.get('crossing', '')).lower()
    ref = str(tags.get('crossing_ref', '')).lower()
    marks = str(tags.get('crossing:markings', '')).lower()
    cycle = tags.get('highway') == 'cycleway' or tags.get('cycleway') == 'crossing' or tags.get('bicycle') == 'designated'
    red = any(str(tags.get(k, '')).lower() == 'red' for k in ('surface:colour', 'colour', 'crossing:colour',
                                                              'cycleway:surface:colour', 'surface:color'))
    if c in ('unmarked', 'no', 'informal') or marks == 'no':
        return None, False
    if c == 'zebra' or ref == 'zebra' or marks == 'zebra':
        return 'zebra', red
    if cycle and tags.get('foot') in ('yes', 'designated') and tags.get('segregated') == 'yes':
        return 'both', red
    if cycle:
        return 'cycle', red
    if c in ('traffic_signals',) or tags.get('crossing:signals') == 'yes':
        return 'signals', red
    # a marked pedestrian crossing without signals is a zebra (StVO Z 293 Fussgaengerueberweg is the only marking for one;
    # the Furt's two dashed lines are for signalled crossings: memory, NOT verified against the text) unless its markings
    # are mapped as something else
    if c in ('marked', 'uncontrolled') and marks in ('', 'yes'):
        return 'zebra', red
    if c in ('marked', 'uncontrolled') or marks not in ('', 'no'):
        return 'marked', red
    return None, False


def _width(tags, kind, urban):
    try:
        w = float(str(tags.get('width', '')).split(';')[0].replace(',', '.'))
        if 1. <= w <= 8.:
            return w
    except ValueError:
        pass
    if kind == 'zebra':
        return ZEBRA_WIDTH['urban' if urban else 'rural']
    if kind == 'cycle':
        return CYCLE_WIDTH
    if kind == 'both':
        return FURT_WIDTH + CYCLE_WIDTH
    return FURT_WIDTH


# =========================================================================================================
# All junctions of a graph
# =========================================================================================================

def inputs(extras):
    """fast_roads.localise's junction inputs as this module reads them: controls [(id, tags, x, y)], crossings and
    islands [(id, tags, xy)] (localise keeps a way's record less its points as a tuple)"""
    if not extras:
        return {}
    def lines(items):
        return [(wid, rec[0] if isinstance(rec, tuple) else rec, xy) for wid, rec, xy in items]
    return {'controls': list(extras.get('controls', ())), 'crossings': lines(extras.get('crossings', ())),
            'islands': lines(extras.get('islands', ()))}


class Near:
    """the junction inputs binned by 100 m, for the ones round a junction"""
    BIN = 100.

    def __init__(self, extras):
        self.bins = {}

        def bin_(b):
            return self.bins.setdefault(b, {'controls': [], 'crossings': [], 'islands': []})
        for item in extras.get('controls', ()):
            bin_((int(item[2] // self.BIN), int(item[3] // self.BIN)))['controls'].append(item)
        for key in ('crossings', 'islands'):
            for item in extras.get(key, ()):
                xy = np.asarray(item[2], float)
                if len(xy) == 0:
                    continue
                seen = set()
                for x, y in xy:
                    b = (int(x // self.BIN), int(y // self.BIN))
                    if b not in seen:
                        seen.add(b)
                        bin_(b)[key].append(item)

    def around(self, x, y, r):
        out = {'controls': [], 'crossings': [], 'islands': []}
        seen = set()
        for i in range(int((x - r) // self.BIN), int((x + r) // self.BIN) + 1):
            for j in range(int((y - r) // self.BIN), int((y + r) // self.BIN) + 1):
                b = self.bins.get((i, j))
                if not b:
                    continue
                for key, items in b.items():
                    for item in items:
                        k = (key, item[0])
                        if k not in seen:
                            seen.add(k)
                            out[key].append(item)
        return out


WORKERS = 8           # processes for a large graph (the index stage's other pools use as many)
BATCH = 250           # clusters per task


def _batch(task):
    """the junction records of one batch of clusters (a worker process's task)"""
    groups, rec, views, near, cell_size = task
    rows, failed = [], 0
    for cl in groups:
        P = np.asarray([rec[n]['xyz'][:2] for n in cl], float)
        try:
            r = junction(cl, [rec[n] for n in cl], views, near.around(*P.mean(0), SPAN + CUT_MAX + CONTROL_REACH + 60.))
        except Exception:                      # one odd junction must not stop the pack
            failed += 1
            r = None
        if r is None:
            continue
        x, y = r['xyz'][0], r['xyz'][1]
        rows.append((r['id'], int(math.floor(x / cell_size)), int(math.floor(y / cell_size)), r))
    return rows, failed


def build(records, views, extras=None, cell_size=1000., emit=None, workers=None):
    """[(id, ci, cj, record)] of every junction cluster (records: node rows' data dicts; extras: fast_roads.localise).
    Large graphs run in batches of nearby clusters on worker processes, each with only its strokes and inputs."""
    t0 = time.perf_counter()
    extras = inputs(extras)
    near = Near(extras)
    groups = clusters(records, views)
    rec = {r['node']: r for r in records}
    workers = WORKERS if workers is None else workers
    if workers <= 1 or len(groups) < 2 * BATCH:
        rows, failed = _batch((groups, rec, views, near, cell_size))
    else:
        # nearby clusters together (their strokes and inputs overlap), in a fixed order
        def where(cl):
            x, y = np.mean([rec[n]['xyz'][:2] for n in cl], axis=0)
            return (int(x // 2000), int(y // 2000), cl[0])
        groups = sorted(groups, key=where)
        tasks = []
        for k in range(0, len(groups), BATCH):
            part = groups[k:k + BATCH]
            sids = {m[0] for cl in part for n in cl for m in rec[n]['mouths']}
            sub_rec = {n: rec[n] for cl in part for n in cl}
            P = np.asarray([rec[n]['xyz'][:2] for cl in part for n in cl], float)
            x0, y0 = P.min(0) - 400.
            x1, y1 = P.max(0) + 400.
            sub_near = Near({})
            for b, items in near.bins.items():
                if x0 // Near.BIN <= b[0] <= x1 // Near.BIN and y0 // Near.BIN <= b[1] <= y1 // Near.BIN:
                    sub_near.bins[b] = items
            tasks.append((part, sub_rec, {sid: views[sid] for sid in sids if sid in views}, sub_near, cell_size))
        from concurrent.futures import ProcessPoolExecutor
        rows, failed = [], 0
        with ProcessPoolExecutor(workers) as pool:
            for r_, f_ in pool.map(_batch, tasks):
                rows += r_
                failed += f_
        rows.sort(key=lambda r: r[0])
    if emit:
        emit(f'junctions: {len(rows)} surfaces from {len(groups)} clusters ({failed} failed) in {time.perf_counter() - t0:.1f}s')
    return rows


# =========================================================================================================
# Binary: one cell's junctions (EGJ1 blocks; the EGB2 layout with its own magic)
# =========================================================================================================

def _q(v):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError('non-finite junction number')
    return int(round(v * SCALE))


def _finite_record(j):
    ok = _all_finite(j)
    if not ok:
        import warnings
        warnings.warn('junction %s skipped: non-finite number' % j.get('id'))
    return ok


def _all_finite(o):
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(_all_finite(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return all(_all_finite(v) for v in o)
    return True


def encode_cell(junctions, origin):
    """payload := strings u(n) n*junction (spec: ROAD-OBJECTS.md 9.4); a junction with a non-finite number is
    skipped with a warning (never stops the pack)"""
    from country_format import _Writer
    junctions = [j for j in junctions if _finite_record(j)]
    w = _Writer()
    table = {}

    def add(v):
        table.setdefault(str(v), len(table))
    for j in junctions:
        add(j['id']); add(j['kind'])
        for n in j['nodes']:
            add(n)
        for m in j['mouths']:
            add(m['stroke']); add(m['node']); add(m['role'])
        for p in j['internal']:
            add(p[0])
        for i in j['islands']:
            add(i['kind'])
        for a in j['approaches']:
            add(a['stroke']); add(a['control']); add(a['turns'])
            for c in a['crossings']:
                add(c['kind'])
    w.u(len(table))
    for text in table:
        data = text.encode('utf8')
        w.u(len(data))
        w.out += data

    def ref(v):
        w.u(table[str(v)])
    ox, oy = _q(origin[0]), _q(origin[1])

    def point(p):
        w.s(_q(p[0]) - ox); w.s(_q(p[1]) - oy); w.s(_q(p[2]))
    w.u(len(junctions))
    for j in junctions:
        ref(j['id']); ref(j['kind']); w.u(1 if j.get('signals') else 0)
        point(j['xyz'])
        w.u(len(j['nodes']))
        for n in j['nodes']:
            ref(n)
        w.u(len(j['mouths']))
        for m in j['mouths']:
            ref(m['stroke']); ref(m['node']); w.s(int(m['dir'])); w.s(_q(m['s'])); w.s(_q(m['cut'])); ref(m['role'])
            for key in ('L', 'C', 'R'):
                point(m[key])
        w.u(len(j['internal']))
        for sid, a, b in j['internal']:
            ref(sid); w.s(_q(a)); w.s(_q(b))
        w.u(len(j['verts']))
        prev = [ox, oy, 0]
        for p in j['verts']:
            q = [_q(p[0]), _q(p[1]), _q(p[2])]
            for f in range(3):
                w.s(q[f] - prev[f])
            prev = q
        w.u(len(j['tris']))
        pa = 0
        for a, b, c in j['tris']:
            w.s(a - pa); w.s(b - a); w.s(c - a)
            pa = a
        w.u(len(j['kerbs']))
        pa = 0
        for a, b in j['kerbs']:
            w.s(a - pa); w.s(b - a)
            pa = a
        w.u(len(j['islands']))
        for i in j['islands']:
            ref(i['kind']); w.u(_q(i['height'])); w.u(len(i['ring']))
            for p in i['ring']:
                point(p)
        w.u(len(j['approaches']))
        for a in j['approaches']:
            ref(a['stroke']); w.s(int(a['dir'])); w.s(_q(a['stop'])); ref(a['control']); ref(a['turns']); w.u(int(a['lanes']))
            w.u(len(a['crossings']))
            for c in a['crossings']:
                w.s(_q(c['s'])); w.u(_q(c['width'])); ref(c['kind']); w.u(1 if c['red'] else 0)
    return bytes(w.out)


def decode_cell(data, origin, at=0, end=None):
    from country_format import _Reader
    r = _Reader(data, at, end)
    strings = []
    for _ in range(r.u()):
        n = r.u()
        strings.append(bytes(data[r.at:r.at + n]).decode('utf8'))
        r.at += n

    def ref():
        return strings[r.u()]
    ox, oy = _q(origin[0]), _q(origin[1])

    def point():
        return [(r.s() + ox) / SCALE, (r.s() + oy) / SCALE, r.s() / SCALE]
    out = []
    for _ in range(r.u()):
        j = {'id': ref(), 'kind': ref(), 'signals': bool(r.u())}
        j['xyz'] = point()
        j['nodes'] = [ref() for _ in range(r.u())]
        j['mouths'] = []
        for _ in range(r.u()):
            m = {'stroke': ref(), 'node': ref(), 'dir': r.s(), 's': r.s() / SCALE, 'cut': r.s() / SCALE, 'role': ref()}
            m['L'], m['C'], m['R'] = point(), point(), point()
            j['mouths'].append(m)
        j['internal'] = [[ref(), r.s() / SCALE, r.s() / SCALE] for _ in range(r.u())]
        j['verts'] = []
        prev = [ox, oy, 0]
        for _ in range(r.u()):
            q = [prev[f] + r.s() for f in range(3)]
            prev = q
            j['verts'].append([q[0] / SCALE, q[1] / SCALE, q[2] / SCALE])
        j['tris'] = []
        pa = 0
        for _ in range(r.u()):
            a = pa + r.s(); b = a + r.s(); c = a + r.s()
            pa = a
            j['tris'].append([a, b, c])
        j['kerbs'] = []
        pa = 0
        for _ in range(r.u()):
            a = pa + r.s(); b = a + r.s()
            pa = a
            j['kerbs'].append([a, b])
        j['islands'] = []
        for _ in range(r.u()):
            i = {'kind': ref(), 'height': r.u() / SCALE}
            i['ring'] = [point() for _ in range(r.u())]
            j['islands'].append(i)
        j['approaches'] = []
        for _ in range(r.u()):
            a = {'stroke': ref(), 'dir': r.s(), 'stop': r.s() / SCALE, 'control': ref(), 'turns': ref(), 'lanes': r.u()}
            a['crossings'] = []
            for _ in range(r.u()):
                a['crossings'].append({'s': r.s() / SCALE, 'width': r.u() / SCALE, 'kind': ref(), 'red': bool(r.u())})
            j['approaches'].append(a)
        out.append(j)
    if r.at != r.end:
        raise ValueError('trailing bytes after junction cell')
    return out


def encode_block(payloads):
    import country_format as binary
    data, mask = binary.encode_block(payloads)
    return MAGIC + data[len(binary.MAGIC):], mask


def decode_block(data):
    import country_format as binary
    if bytes(data[:4]) != MAGIC:
        raise ValueError('not an EarthNG junction block')
    return binary.decode_block(binary.MAGIC + bytes(data[4:]))
