"""Plan-view curves for OSM road polylines: the exporter's centre lines get arcs and transition curves.

OSM draws a road as straight chords between nodes; a motorway node every 80 m and a slip road node every
15 m bend the road by 1-2 and 5-15 degrees at each node, and the game drove and drew those kinks. Here
each way's polyline is turned into the horizontal alignment road design would give it:

1. Every free node (one no other way shares) is a point of intersection (PI) of two tangents and gets a
   circular arc, as in road design (RAA 2008 / RAL 2012) and BeamNG's own engineered spline mode. The
   radius is the largest that fits: the arc may move the line at most `soft` m off the node (`hard` m
   if the class's minimum radius needs it), and two arcs share the chord between them (tangent lengths
   relaxed until they fit). On a densely mapped bend this gives arcs that meet at the chords' middles:
   the bend's own radius, shifted inward by the chords' sagitta (a few decimetres).
2. A node shared with another way stays exactly where it is. Where the road runs on through it (the
   same way, or the way continuing it end to end), the line passes it on the bisector of its two
   chords, between two virtual PIs on that tangent line; both ways see the same three OSM nodes, so
   they leave the node in the same direction. A road ending at a junction meets it along its last chord.
3. Transition curves: the heading of the arc-and-line chain is averaged over a class window (`window`)
   between fixed nodes. Averaging the heading is averaging the curvature diagram with a box: every
   curvature step becomes a linear ramp of that length (a clothoid), the arcs keep their radius, and
   the line moves inward by the textbook shift Ls^2 / (24 R). The window shrinks to zero at the fixed
   nodes, so their position and direction are kept (a small smooth correction closes the end).

The result is sampled densely enough that a chord leaves the curve by at most `TOLERANCE` m. Only
the classes in CLASSES are curved; the rest keep their polylines. The switch is EARTHNG_CURVES
(0 = off); SIGNATURE goes into the source index's identity, so changing it rebuilds the index.

Review (30 Oct 2026, pack v3): `curve()` is sound and is what fast_roads uses, per stroke (OSM ways
joined), with fast_roads choosing the fixed nodes and their tangents. `fit()` below (per OSM way, used
only with EARTHNG_CURVES=1 for the v2 segments) has three faults and stays off:
- every shared node is fixed, way joints included: the transition window shrinks to zero there, so the
  curvature steps at every joint (fast_roads fixes only junction nodes; joints are free points);
- a slip road meets its node along its own last chord, not the host's tangent (no G1 at a diverge);
  fast_roads passes the host's tangent to every branch;
- mapping noise (a vertex 0.3-2 m from the next, at joints and junctions) is fitted as a 1-2 m radius
  corner; fast_roads drops such free vertices first (MIN_LEG, SIMPLIFY).
Longer transition windows than CLASSES' (tried: motorway 200 m) put kinks back: the end-error correction
of `_emit` grows with the window.
"""
import math
import os

import numpy as np

ENABLED = os.environ.get('EARTHNG_CURVES', '0') != '0'   # off until the road-objects design is agreed (draft from a stopped agent)

# class: (soft, hard, window, minimum radius). soft/hard: the furthest the line may move off a free
# OSM node (m); hard only to reach the class's minimum horizontal radius (RAA 2008: EKA 1 B 720 m,
# verified in Tabelle 26 of its 2012 correction; the RAL radii below are not verified;
# RAL 2012 recommended radii: EKL 1 >= 500, EKL 2 400-900, EKL 3 300-600, EKL 4 200-400 m; ramps:
# none found online, 50 m assumed for loops). window: the transition length (m).
CLASSES = {
    'motorway': (2.0, 4.0, 60., 720.), 'trunk': (2.0, 4.0, 50., 500.),
    'primary': (1.5, 3.0, 40., 400.), 'secondary': (1.5, 3.0, 30., 300.),
    'tertiary': (1.25, 2.5, 25., 200.),
    'motorway_link': (1.5, 3.0, 20., 50.), 'trunk_link': (1.5, 3.0, 20., 50.),
    'primary_link': (1.5, 3.0, 20., 50.), 'secondary_link': (1.0, 2.0, 15., 30.),
    'tertiary_link': (1.0, 2.0, 15., 30.),
}
TOLERANCE = 0.02         # m: largest gap between an emitted chord and the curve
MAX_STEP = 25.0          # m: longest emitted chord on a curve (straights keep their ends only)
MAX_TURN = math.radians(2.0)   # heading change per emitted chord
MIN_STEP = 0.5           # m: shortest emitted chord (the generator merges knots under 0.25 m)
DEDUPE = 0.05            # m: a free node this close to the previous one is dropped
THROUGH = math.radians(0.05)   # a deflection under this needs no arc
HAIRPIN = math.radians(150.)   # over this: left as it is (U-turn artefacts)
CONTINUE = math.radians(30.)   # two way ends at a node continue each other under this deflection
SHIFT = 0.25             # transition shift: at most this share of the soft bound
SIGNATURE = 'curves-1:%s' % ('on' if ENABLED else 'off')


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n, n) if n > 1e-9 else (0., 0., 0.)


def _deflection(a, b, c):
    """signed heading change at b (rad, + left) and the two leg lengths"""
    ux, uy, l1 = _unit(b[0] - a[0], b[1] - a[1])
    vx, vy, l2 = _unit(c[0] - b[0], c[1] - b[1])
    if l1 == 0. or l2 == 0.:
        return 0., l1, l2
    return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy), l1, l2


def _radius_for(delta, e):
    """radius whose arc lies e m inside the PI of a corner of deflection delta"""
    k = 1. / math.cos(abs(delta) / 2) - 1.
    return e / k if k > 1e-15 else 1e12


class _Alignment:
    """An arc-and-line chain: pieces (s0, length, x0, y0, h0, kappa); fixed points at known s."""

    def __init__(self, x, y, h):
        self.s, self.x, self.y, self.h = 0., x, y, h
        self.pieces = []
        self.fixed = []          # (s, x, y)

    def line_to(self, x, y):
        dx, dy = x - self.x, y - self.y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            self.x, self.y = x, y
            return
        h = math.atan2(dy, dx)
        # keep the heading continuous (unwrapped) for the smoothing
        h += 2 * math.pi * round((self.h - h) / (2 * math.pi))
        self.pieces.append((self.s, n, self.x, self.y, h, 0.))
        self.s += n
        self.x, self.y, self.h = x, y, h

    def arc(self, length, kappa):
        if length < 1e-6:
            return
        self.pieces.append((self.s, length, self.x, self.y, self.h, kappa))
        h1 = self.h + kappa * length
        self.x += (math.sin(h1) - math.sin(self.h)) / kappa
        self.y -= (math.cos(h1) - math.cos(self.h)) / kappa
        self.s += length
        self.h = h1

    def fix(self, x, y):
        self.x, self.y = x, y   # within rounding of the pieces' end
        self.fixed.append((self.s, x, y))


def _evaluate(pieces, s):
    """heading and curvature of the chain at the arc lengths s (array)"""
    s0 = np.array([p[0] for p in pieces])
    h0 = np.array([p[4] for p in pieces])
    k = np.array([p[5] for p in pieces])
    i = np.clip(np.searchsorted(s0, s, side='right') - 1, 0, len(pieces) - 1)
    return h0[i] + k[i] * (s - s0[i]), k[i]


def _emit(s, h, kappa, x0, y0, x1, y1, out):
    """append the points of one span (s grid, smoothed heading) after its start, ending exactly at
    (x1, y1); positions integrated from the heading, the end error closed with a smoothstep"""
    ds = np.diff(s)
    cx, cy = np.cos(h), np.sin(h)
    x = x0 + np.r_[0., np.cumsum((cx[1:] + cx[:-1]) * ds / 2)]
    y = y0 + np.r_[0., np.cumsum((cy[1:] + cy[:-1]) * ds / 2)]
    # spread over the curved part (by turned angle): the straights before and after stay straight
    C = np.r_[0., np.cumsum((np.abs(kappa[1:]) + np.abs(kappa[:-1])) * ds / 2)]
    u = C / C[-1] if C[-1] > 1e-12 else (s - s[0]) / (s[-1] - s[0])
    w = u * u * (3 - 2 * u)
    x -= (x[-1] - x1) * w
    y -= (y[-1] - y1) * w
    # output density: chord gap TOLERANCE, at most MAX_TURN, at most MAX_STEP
    ak = np.abs(kappa)
    with np.errstate(divide='ignore'):
        step = np.maximum(MIN_STEP, .9 * np.minimum(np.minimum(np.sqrt(8 * TOLERANCE / ak), MAX_TURN / ak), MAX_STEP))
    density = 1. / step
    D = np.r_[0., np.cumsum((density[1:] + density[:-1]) * ds / 2)]
    m = max(1, int(math.ceil(D[-1] - 1e-9)))
    if m > 1:
        at = np.interp(np.arange(1, m) * D[-1] / m, D, s)
        out.extend(zip(np.interp(at, s, x).tolist(), np.interp(at, s, y).tolist()))
    out.append((x1, y1))


def _span(pieces, a, b, window, shift, out):
    """points of the chain between fixed arc lengths a < b (end points (x, y) given)"""
    (sa, xa, ya), (sb, xb, yb) = a, b
    L = sb - sa
    inside = [p for p in pieces if p[0] < sb - 1e-9 and p[0] + p[1] > sa + 1e-9]
    if L < 1e-6:
        return
    if not any(p[5] for p in inside):
        out.append((xb, yb))                 # a straight line
        return
    # the transition moves an arc inward by Ls^2 / (24 R): at most shift m at the span's tightest arc
    kmax = max(abs(p[5]) for p in inside)
    w = min(window, L / 2, math.sqrt(24 * shift / kmax))
    ds = max(0.05, min(1.0, w / 8 if w > 0 else 1.0, L / 16))
    n = max(2, int(math.ceil(L / ds)))
    s = sa + L * np.arange(n + 1) / n
    h, k = _evaluate(inside, s)
    if w > 1.:
        # moving average of the heading over a window shrinking to 0 at the span's ends
        I = np.r_[0., np.cumsum((h[1:] + h[:-1]) * np.diff(s) / 2)]
        hw = np.minimum(w / 2, np.minimum(s - sa, sb - s))
        lo, hi = np.interp(s - hw, s, I), np.interp(s + hw, s, I)
        with np.errstate(invalid='ignore', divide='ignore'):
            hs = np.where(hw > 1e-9, (hi - lo) / (2 * hw), h)
        # curvature for the output density: the raw arcs' or the averaged one, whichever is larger
        with np.errstate(invalid='ignore', divide='ignore'):
            ks = np.where(hw > 1e-9, (np.interp(s + hw, s, h) - np.interp(s - hw, s, h)) / (2 * hw), k)
        k = np.maximum(np.abs(k), np.abs(ks))
        h = hs
    _emit(s, h, k, xa, ya, xb, yb, out)


def _polygon(pts, fixed, tangent, soft):
    """control polygon: [(x, y, kind)] with kind 'fixed' or 'corner'; a fixed node the road runs on through
    (in tangent) gets a virtual PI on that tangent on each side that lies inside this way"""
    poly = []
    n = len(pts)
    for k in range(n):
        p = pts[k]
        if not fixed[k]:
            poly.append((p[0], p[1], 'corner'))
            continue
        t = tangent.get(k)
        if t is None:
            poly.append((p[0], p[1], 'fixed'))
            continue
        tx, ty, half = t
        if k > 0:
            l1 = math.hypot(p[0] - pts[k - 1][0], p[1] - pts[k - 1][1])
            d = min(.4 * l1, soft / max(math.sin(half), 1e-9))
            if d > 1e-3:
                poly.append((p[0] - d * tx, p[1] - d * ty, 'corner'))
        poly.append((p[0], p[1], 'fixed'))
        if k < n - 1:
            l2 = math.hypot(pts[k + 1][0] - p[0], pts[k + 1][1] - p[1])
            d = min(.4 * l2, soft / max(math.sin(half), 1e-9))
            if d > 1e-3:
                poly.append((p[0] + d * tx, p[1] + d * ty, 'corner'))
    return poly


def curve(pts, fixed, tangent, params):
    """One way: pts [(x, y)], fixed[k] (node kept exactly), tangent {k: (tx, ty, half deflection)} for the
    fixed nodes the road runs on through. Returns (points, index of each fixed node in points)."""
    soft, hard, window, rmin = params
    poly = _polygon(pts, fixed, tangent, soft)
    m = len(poly)
    corner = [q[2] == 'corner' for q in poly]
    delta = [0.] * m
    legs = [math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1]) for i in range(m - 1)]
    R = [0.] * m
    for i in range(1, m - 1):
        if not corner[i]:
            continue
        d, l1, l2 = _deflection(poly[i - 1], poly[i], poly[i + 1])
        if abs(d) < THROUGH or abs(d) > HAIRPIN:
            corner[i] = False
            continue
        delta[i] = d
        R[i] = max(_radius_for(d, soft), min(rmin, _radius_for(d, hard)))
    corner[0] = corner[-1] = False
    tan2 = [math.tan(abs(d) / 2) for d in delta]
    # tangent lengths: each corner wants R tan(delta / 2); a chord shared by two corners gives each at
    # least half of it, and a corner may take what its neighbour leaves (a chord with one corner is its)
    want = [R[i] * tan2[i] if corner[i] else 0. for i in range(m)]
    leg = [q * .999 for q in legs]
    half = [0.] * m
    for i in range(1, m - 1):
        if corner[i]:
            half[i] = min(want[i], leg[i - 1] / (2 if corner[i - 1] else 1), leg[i] / (2 if corner[i + 1] else 1))
    T = [0.] * m
    for i in range(1, m - 1):
        if corner[i]:
            T[i] = min(want[i], leg[i - 1] - (half[i - 1] if corner[i - 1] else 0.),
                       leg[i] - (half[i + 1] if corner[i + 1] else 0.))
    for j in range(m - 1):              # two corners that both grew into one chord share it
        if T[j] + T[j + 1] > leg[j] > 0:
            f = leg[j] / (T[j] + T[j + 1])
            T[j] *= f
            T[j + 1] *= f
    for i in range(1, m - 1):
        if corner[i]:
            R[i] = T[i] / tan2[i]
            if T[i] < 1e-4:
                corner[i] = False
    al = _Alignment(poly[0][0], poly[0][1], math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0]))
    al.fix(poly[0][0], poly[0][1])
    for i in range(1, m):
        x, y, kind = poly[i]
        if i < m - 1 and corner[i]:
            ux, uy, _ = _unit(x - poly[i - 1][0], y - poly[i - 1][1])
            T = R[i] * tan2[i]
            al.line_to(x - T * ux, y - T * uy)
            # the arc starts on the incoming tangent's heading
            h_in = math.atan2(uy, ux)
            al.h = h_in + 2 * math.pi * round((al.h - h_in) / (2 * math.pi))
            al.arc(R[i] * abs(delta[i]), math.copysign(1. / R[i], delta[i]))
        else:
            al.line_to(x, y)
            if kind == 'fixed':
                al.fix(x, y)
    out = [(poly[0][0], poly[0][1])]
    index = [0]
    for a, b in zip(al.fixed, al.fixed[1:]):
        _span(al.pieces, a, b, window, SHIFT * soft, out)
        index.append(len(out) - 1)
    return out, index


def fit(ways, refs, classes=None):
    """ways: {way id: (tags, [(node id, x, y), ...])}; refs: node id -> number of way references.
    Returns {way id: (points [(x, y)], {original vertex index: index in points})} for the ways of
    `classes` (default CLASSES); the mapped vertices are the shared nodes and the two ends, exactly."""
    classes = CLASSES if classes is None else classes
    chosen = {w: v for w, v in ways.items() if v[0].get('highway') in classes and len(v[1]) >= 2}
    # free nodes closer than DEDUPE to the previous kept node go
    clean = {}
    for w, (tags, nodes) in chosen.items():
        keep, last = [], len(nodes) - 1
        for k, (nid, x, y) in enumerate(nodes):
            shared = k == 0 or k == last or refs[nid] > 1
            if keep and not shared and math.hypot(x - keep[-1][2], y - keep[-1][3]) < DEDUPE:
                continue
            keep.append((k, nid, x, y))
        clean[w] = keep
    # way ends continuing each other at a node: mutual best, nearly straight on
    ends = {}
    for w, keep in clean.items():
        for e, (k, nid, x, y) in ((0, keep[0]), (1, keep[-1])):
            if refs[nid] < 2 or len(keep) < 2:
                continue
            nb = keep[1] if e == 0 else keep[-2]
            ax, ay, n = _unit(nb[2] - x, nb[3] - y)
            if n > 0:
                ends.setdefault(nid, []).append((w, e, ax, ay, nb[2], nb[3]))
    partner = {}
    for nid, lst in ends.items():
        if len(lst) < 2:
            continue
        best = {}
        for i, a in enumerate(lst):
            for j, b in enumerate(lst):
                if i == j:
                    continue
                dev = math.pi - math.acos(max(-1., min(1., a[2] * b[2] + a[3] * b[3])))
                if dev < CONTINUE and (i not in best or dev < best[i][0]):
                    best[i] = (dev, j)
        for i, (dev, j) in best.items():
            if best.get(j, (0, -1))[1] == i:
                partner[(lst[i][0], lst[i][1])] = (lst[j][4], lst[j][5])
    out = {}
    for w, keep in clean.items():
        params = classes[chosen[w][0]['highway']]
        pts = [(q[2], q[3]) for q in keep]
        n = len(pts)
        fixed = [q[0] == 0 or q[0] == len(chosen[w][1]) - 1 or refs[q[1]] > 1 for q in keep]
        tangent = {}
        for k in range(n):
            if not fixed[k]:
                continue
            a = pts[k - 1] if k > 0 else partner.get((w, 0))
            b = pts[k + 1] if k < n - 1 else partner.get((w, 1))
            if a is None or b is None:
                continue
            d, l1, l2 = _deflection(a, pts[k], b)
            if abs(d) < THROUGH or abs(d) > HAIRPIN:
                continue
            ux, uy, _ = _unit(pts[k][0] - a[0], pts[k][1] - a[1])
            vx, vy, _ = _unit(b[0] - pts[k][0], b[1] - pts[k][1])
            tx, ty, _ = _unit(ux + vx, uy + vy)
            tangent[k] = (tx, ty, abs(d) / 2)
        if n < 2 or (n == 2 and not tangent):
            out[w] = (pts, {q[0]: i for i, q in enumerate(keep)})
            continue
        line, index = curve(pts, fixed, tangent, params)
        out[w] = (line, {keep[k][0]: index[i] for i, k in enumerate([k for k in range(n) if fixed[k]])})
    return out
