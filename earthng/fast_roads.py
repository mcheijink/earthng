"""Pack v3 road geometry: the exporter side of the new road builder (patches/ROAD-OBJECTS.md, section 7).

The index knows the whole road graph once, so everything that must be the same in every tile is
decided here and written into the pack; the game only sweeps it (phase 2):

- strokes: OSM way pieces joined through way joints and through the straight-on pair of a junction
  (same highway, same one-way-ness), oriented in the direction of travel when one-way;
- a fitted centre line per stroke (fast_curves: arcs at the free PIs, transitions by heading
  averaging) with one tangent per shared node: the bisector of the node's through pair, which a slip
  road leaving there takes as well. Junction nodes stay exactly where they are; way joints inside a
  stroke are free points of the fit (fixing them stopped the transitions at every joint);
- stations (s, x, y, z, heading, curvature, paved extents left and right, cross-fall of each half);
  heights are the solved profile (fast_profile), carried over by arc length between fixed nodes;
- lane runs from the OSM lane tags, lane gains and drops as linear tapers on the wider side;
- exits and entries to the design rules (RAA 2008): a parallel deceleration / acceleration lane and
  its taper on the host, the slip road re-anchored on that lane's centre and blended into its own
  line, instead of OSM's split point on the host's centre line;
- furniture: OSM-mapped barriers (guard rails, jersey and cable barriers, walls) on or along a road,
  as runs per side; nothing is invented;
- structures: bridges, tunnels, embankments and cuttings as runs, with the waterway clearance a deck
  over navigable water was given (bridge_floors, a lower bound in fast_profile).

Off unless EARTHNG_PACK >= 3; bridge_floors (the profile fix) is always on. Never prints positions.
"""
import hashlib
import math
import os
import time
from collections import defaultdict

import numpy as np

PACK = int(os.environ.get('EARTHNG_PACK', '2') or 2)
ENABLED = PACK >= 3
FORMAT = 'earthng-roads-v3'
VERSION = 3                   # the section's version in the manifest: 2 adds the junction records' fan plates; 3: the
                              # added lanes and lane tapers are S-shaped (smoothstep), OSM-mapped auxiliary lanes are
                              # the exits' and entries' added lanes (the game shapes its lane marks to match)
# phase 3 junction surfaces (fast_junctions) in pack v3: EARTHNG_JUNCTIONS=0 leaves them out
JUNCTIONS = ENABLED and os.environ.get('EARTHNG_JUNCTIONS', '1') != '0'
SIGNATURE = f'roads-v3.6:{"on" if ENABLED else "off"}{":junctions" if JUNCTIONS else ""}'

CHUNK = 200.0                 # m of stroke per chunk (the game's unit of creation and streaming)
MAX_STATION = 10.0            # m between stations on a straight (z is linear between them: <= 1.3 cm at R 1,000 m)
THROUGH = math.radians(35.)   # two mouths of a junction within this of straight on run through it
BRANCH = math.radians(30.)    # a mouth within this of a through mouth is a branch of it (diverge, merge, fork)
REACH = 2.0                   # m along a road for the direction it leaves a node in
CURVE_WINDOW = 10.0           # m either side: curvature from the heading change over 20 m
MIN_LEG = 2.0                 # m: a free OSM vertex this close to a kept one is mapping noise to the fit
SIMPLIFY = 0.10               # m: a free OSM vertex this close to its neighbours' chord adds nothing to the fit
RANK = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3, 'unclassified': 2,
        'residential': 2, 'road': 2, 'living_street': 1.5, 'service': 1, 'raceway': 1}

# --- cross-section -------------------------------------------------------------------------------------
# Cross-fall q on a straight: 2.5 % (design decision; RAA 2008 Tabelle 26 "Mindestquerneigung 2,5 %").
# One-way carriageways fall to their right (outer) edge, two-way roads get a crown of 2.5 % each way.
# In a curve tighter than R_out the road banks toward the inside: q(R) rises from 2.5 % at R_out to q_max at
# R_min, linear in 1/R (the shape of the RAA/RAL q-R diagrams), and stays at q_max below R_min.
Q0 = .025
# highway: (R_min, R_out, q_max, verified). Verified (RAA 2008, 2. Korrektur Mai 2012, Tabelle 26, and the
# A 70 Feststellungsentwurf 2020, Tabelle 7, which quotes the RAA ramp limits): motorway EKA 1 B R_min 720 m,
# outward cross-fall allowed from R 4,000 m, q_max 6.0 %; EKA 2 R_min 470 m; ramp q_max 6.0 %, ramp
# R_min 80 m at V_Rampe 50 km/h. The two-way classes (RAL 2012 / RASt 06) are NOT verified here: radii from
# the fast_curves header, q_max 7 % and R_out from memory; residential, service and living streets keep
# their crown (RASt: no superelevation in built-up areas).
BANK = {'motorway': (720., 4000., .06, True), 'trunk_oneway': (470., 4000., .06, True),
        'link': (80., 1000., .06, True),
        'trunk': (500., 2500., .07, False), 'primary': (400., 2000., .07, False),
        'secondary': (300., 1500., .07, False), 'tertiary': (200., 1000., .07, False),
        'unclassified': (150., 800., .07, False), 'road': (150., 800., .07, False)}
# Run-off (Verwindung): the edge may rise against the axis by at most max ds = 0.9 % where the edge is
# a >= 4 m from the axis of rotation, 0.225 a % below that (RAA 2008 Tabelle 26, EKA 1-2; verified).
# The axis is the centre line, a the wider side.
RUNOFF_MAX, RUNOFF_SMALL = .009, .00225
# Tight one-lane links (cloverleaf loops, bundle6 M6): paved about 8.5 m wide where R < LOOP_R (DOP20 aerial 8.5-10 m at the
# motorway loops, measured), the extra on the inner side of the curve (ALKIS: the pavement lies inside of the OSM line);
# cross-fall at most 4 % there (DGM1 3.5-4.1 %, measured; the 6 % of BANK['link'] is the RAA ramp limit). The rule itself (R < 100 m,
# 8.5 m, 4 %) is the task brief's, NOT from a guideline.
LOOP_R, LOOP_WIDTH, LOOP_Q_MAX, LOOP_EASE = 100., 8.5, .04, 100.
# Cross-fall changes (bank in and out of bends, an S-bend's flip) are eased over at least this many metres (review: LOOP_EASE's
# idea for every road; the RAA run-off rate RUNOFF_MAX still applies on top)
CROSSFALL_EASE = 100.

# Lane gains and drops (lane counts changing at a way joint): a taper on the wider side of the joint, S-shaped
# (two opposed parabolas, the smoothstep: no kink at either end; the shape is RAL / RASt practice from memory, NOT
# verified). The lengths are not from a guideline: RAA Bild 66 (verified) has a 200 m Verziehung for climbing
# lanes only. At a merge, diverge or fork node the lanes come or go with the branch, whose surface stands in beside
# the host there: no taper on the branch's side, the width changes at the node (DOP20 aerial: the
# second lane starts where the joining ramp's lane does, measured); the other side keeps its taper.
TAPER = {'motorway': 120., 'trunk': 90., 'link': 40., 'default': 40.}     # default 20 -> 40 (bundle8: edge notches at piece joints)

# Exits and entries (RAA 2008, Ausfahrttyp A 1 / Einfahrttyp E 1). The structure is verified (Bild 56 a,
# 2. Korrektur 2012: a parallel lane of length l_A and a taper l_Z for type A 1); the lengths are NOT
# verified here (from memory: l_A = l_E = 250 m, l_Z = 60 m on EKA 1; shorter on EKA 2/3).
# The ramp leaves at an angle of at least 6 gon, 12 gon desired (RAA 6.4, verified): BLEND is the length
# over which the re-anchored slip road turns from the parallel lane into its own line.
EXIT = {'motorway': (250., 60.), 'trunk': (150., 45.)}
ENTRY = {'motorway': (250., 60.), 'trunk': (150., 45.)}
ADDED_LANE = 3.5              # m: RAA lanes are 3.25-3.75 m (Bild 1, verified); 3.50 m for the added lane; DOP20
                              # aerials: the deceleration lane 3.4-3.6 m (measured)
# OSM-mapped auxiliary lanes (bundle5, 31 Oct): German motorway mapping tags the deceleration lane as one more
# lane of the host way up to the exit's split node (turn:lanes none|none|slight_right) and the acceleration lane
# from the entry's split node on (none|none|merge_to_left): 47 of 48 exits and entries checked.
# The RAA lane was added on top of it (a lane too many, a 120 m bulge). Where the host's lane count before an exit
# (after an entry) is the through count + 1 or 2 over AUX_MIN-AUX_MAX m, that run IS the added lane: it leaves the
# host's lane runs and extents and the event's parallel lane follows it. Measured on the DOP20 aerials (LGLN 2026,
# 20 cm; straightened views along the host): exits reach full width 4-20 m before the
# OSM run starts (AUX_SHIFT exit -10 m); entries keep it 5-30 m past the OSM run's end
# (+15 m); the taper is 16-25 m at all five (AUX_TAPER 20 m; the RAA l_Z of 60 m from memory is not what is built
# there). Exits and entries without a mapped lane keep EXIT / ENTRY.
AUX_SHIFT = {'exit': -10., 'entry': 15.}
AUX_TAPER = 20.
AUX_MIN, AUX_MAX = 30., 700.
BLEND = 60.0                  # m of slip road that turns from the parallel lane into its own line
SEARCH = 400.0                # m along a slip road searched for where it reaches the added lane's centre

# --- furniture -----------------------------------------------------------------------------------------
# Only what OSM maps (design decision): barrier ways along a road, barrier tags on the road itself.
BARRIER_KIND = {'guard_rail': 'guard_rail', 'crash_barrier': 'guard_rail', 'jersey_barrier': 'jersey_barrier',
                'cable_barrier': 'cable_barrier', 'wall': 'wall', 'retaining_wall': 'retaining_wall',
                'noise_barrier': 'noise_wall'}
# junction inputs (fast_junctions, phase 3): the node and way tags kept for signals, signs and crossings
CONTROL_HIGHWAYS = {'traffic_signals', 'stop', 'give_way', 'crossing', 'mini_roundabout'}
CONTROL_KEYS = ('highway', 'crossing', 'crossing_ref', 'crossing:markings', 'crossing:signals', 'traffic_signals',
                'traffic_signals:direction', 'direction', 'stop', 'bicycle', 'foot', 'segregated', 'cycleway', 'footway',
                'path', 'width', 'surface:colour', 'colour', 'crossing:colour', 'cycleway:surface:colour', 'surface:color',
                'traffic_calming', 'button_operated')
BARRIER_REACH = (-1.0, 4.0)   # m: a barrier between 1 m inside and 4 m outside a paved edge belongs to it
BARRIER_GAP = 6.0             # m: closer pieces of one barrier along a road are one run

# --- bridges over water ----------------------------------------------------------------------------------
WATERWAYS = {'river', 'canal', 'fairway', 'tidal_channel'}
WATER_KEYS = ('waterway', 'CEMT', 'ship', 'boat', 'motorboat', 'maxheight', 'width', 'name', 'tidal')
MARK_KEYS = ('seamark:type', 'seamark:bridge:clearance_height', 'seamark:bridge:clearance_height_closed',
             'seamark:bridge:category', 'maxheight', 'name')
# Minimum clearance under a fixed bridge by CEMT class (m). NOT verified here (AGN Annex III from memory):
# classes IV-Vb 5.25 m (two container layers), VIa-VIc 7.00 m (three), VII 9.10 m (four); I-III about 4 m.
# Class 0 (recreational) has no AGN minimum: such a bridge keeps the default (rule 3).
CEMT_CLEARANCE = {'I': 4.0, 'II': 4.0, 'III': 4.0, 'IV': 5.25, 'Va': 5.25, 'Vb': 5.25,
                  'VIa': 7.0, 'VIb': 7.0, 'VIc': 7.0, 'VII': 9.1}
SMALL_CRAFT = 4.0             # ship=yes or boat + motorboat without a class
MARK_REACH = 60.0             # m from the crossing within which a seamark or maxheight node is the crossing's
WATER_REACH = 100.0           # m of waterway either side of the crossing searched for its lowest DEM
WATER_WIDTH = {'river': 40., 'canal': 30., 'fairway': 40., 'tidal_channel': 30.}
# Clearances in tidal waters refer to mean high water (MHW), inland ones to the highest navigable level; the
# DEM reads about the mean surface (the Hunte: -1.65 m NHN). A river, fairway or tidal channel whose DEM
# water level is within TIDAL_BAND m of the sea (or tagged tidal=yes) takes MHW m over the sea as its datum
# (+1.5 m: the design note's value for the Hunte, not verified here). Canals are regulated: their DEM level stays.
TIDAL_BAND, MHW = 2.0, 1.5


def _number(value, default=None):
    try:
        return float(str(value).split(';')[0].strip().replace(' m', '').replace(',', '.'))
    except (TypeError, ValueError):
        return default


def _oneway(tags):
    import fast_sources
    return fast_sources.oneway(tags)


def travel(tags):
    """+1 one-way along the node order, -1 one-way against it, 0 two-way"""
    if not _oneway(tags):
        return 0
    return -1 if str(tags.get('oneway', '')).strip() == '-1' else 1


def rank(tags):
    hw = tags.get('highway', '')
    base = hw.replace('_link', '')
    return RANK.get(base, 1) - (.5 if hw.endswith('_link') else 0.)


def family(highway):
    if highway.endswith('_link'):
        return 'link'
    return highway


# =========================================================================================================
# OSM extras of a source tile: waterways, clearance marks, barriers
# =========================================================================================================

def barrier_kind(tags):
    b = tags.get('barrier')
    if b == 'wall' and tags.get('wall') == 'noise_barrier':
        return 'noise_wall'
    if b in BARRIER_KIND:
        return BARRIER_KIND[b]
    if tags.get('crash_barrier') not in (None, 'no'):
        return 'guard_rail'
    return None


def read_extras(path, sha256):
    """{'water': {way: (tags, [(lon, lat)])}, 'marks': {node: (tags, lon, lat)}, 'barriers': {way: (kind, tags,
    [(lon, lat)])}} of one source tile. A maxheight on a node of a waterway counts as a mark. With the v3
    section on (ENABLED) also the junctions' inputs (fast_junctions): 'controls' {node: (tags, lon, lat)}
    (signals, stop and give-way signs, crossings, mini roundabouts), 'crossings' {way: (tags, [(lon, lat)])}
    (footway / cycleway crossing ways) and 'islands' {way: (tags, [(lon, lat)])} (mapped traffic islands)."""
    import osmium
    from pathlib import Path
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError(f'OSM checksum mismatch: {path}')
    water, marks, barriers, heights = {}, {}, {}, {}
    controls, crossings, islands = {}, {}, {}
    junction_inputs = ENABLED

    def points(w):
        out = []
        for n in w.nodes:
            try:
                out.append((n.lon, n.lat))
            except osmium.InvalidLocationError:
                return None
        return out

    class Reader(osmium.SimpleHandler):
        def node(self, n):
            t = n.tags
            if junction_inputs and n.location.valid() and (t.get('highway') in CONTROL_HIGHWAYS or 'crossing' in t
                                                            or t.get('traffic_calming') == 'island'):
                controls[n.id] = ({k: t[k] for k in CONTROL_KEYS if k in t}, n.location.lon, n.location.lat)
            if 'seamark:bridge:clearance_height' in t or 'seamark:bridge:clearance_height_closed' in t:
                marks[n.id] = ({k: t[k] for k in MARK_KEYS if k in t}, n.location.lon, n.location.lat)
            elif 'maxheight' in t and n.location.valid():
                heights[n.id] = (t['maxheight'], n.location.lon, n.location.lat)

        def way(self, w):
            t = w.tags
            if t.get('waterway') in WATERWAYS and w.id not in water:
                pts = points(w)
                if pts and len(pts) >= 2:
                    water[w.id] = ({k: t[k] for k in WATER_KEYS if k in t}, pts)
                for n in w.nodes:
                    if n.ref in heights and n.ref not in marks:
                        v, lon, lat = heights[n.ref]
                        marks[n.ref] = ({'maxheight': v}, lon, lat)
            if junction_inputs and w.id not in crossings and (t.get('footway') == 'crossing' or t.get('cycleway') == 'crossing'
                                                               or t.get('path') == 'crossing' or (t.get('highway') in ('footway', 'cycleway', 'path')
                                                                                                  and 'crossing' in t)):
                pts = points(w)
                if pts and len(pts) >= 2:
                    crossings[w.id] = ({k: t[k] for k in CONTROL_KEYS if k in t}, pts)
            if junction_inputs and w.id not in islands and (t.get('area:highway') == 'traffic_island'
                                                             or (t.get('traffic_calming') == 'island' and len(w.nodes) > 3)):
                pts = points(w)
                if pts and len(pts) >= 4 and pts[0] == pts[-1]:
                    islands[w.id] = ({k: t[k] for k in ('area:highway', 'traffic_calming', 'barrier') if k in t}, pts)
            kind = barrier_kind(t)
            if kind and w.id not in barriers and t.get('highway') is None:
                pts = points(w)
                if pts and len(pts) >= 2:
                    barriers[w.id] = (kind, {k: t[k] for k in ('barrier', 'wall', 'height', 'material') if k in t}, pts)

    Reader().apply_file(str(path), locations=True)
    out = {'water': water, 'marks': marks, 'barriers': barriers}
    if junction_inputs:
        out.update(controls=controls, crossings=crossings, islands=islands)
    return out


def merge_extras(parts):
    out = {'water': {}, 'marks': {}, 'barriers': {}, 'controls': {}, 'crossings': {}, 'islands': {}}
    for part in parts:
        for key in out:
            for k, v in part.get(key, {}).items():
                out[key].setdefault(k, v)
    return out


def localise(extras, frame):
    """The extras in the index frame: water lines and barrier lines as (id, tags, xy array), marks as
    (id, tags, x, y)."""
    def lines(items):
        out = []
        for wid, rec in sorted(items.items()):
            pts = rec[-1]
            x, y = frame.local([p[0] for p in pts], [p[1] for p in pts])
            out.append((wid, rec[:-1], np.column_stack([x, y])))
        return out
    def points_of(items):
        out = []
        for nid, (tags, lon, lat) in sorted(items.items()):
            x, y = frame.local([lon], [lat])
            out.append((nid, tags, float(x[0]), float(y[0])))
        return out
    return {'water': lines(extras['water']), 'marks': points_of(extras['marks']), 'barriers': lines(extras['barriers']),
            'controls': points_of(extras.get('controls', {})), 'crossings': lines(extras.get('crossings', {})),
            'islands': lines(extras.get('islands', {}))}


# =========================================================================================================
# Bridges over navigable water: a lower bound on the deck (fast_profile.solve(floors=...))
# =========================================================================================================

def _cemt(value):
    v = str(value or '').strip()
    for key in sorted(CEMT_CLEARANCE, key=len, reverse=True):
        if v.upper() == key.upper():
            return CEMT_CLEARANCE[key]
    return None


def structure_depth(length, tags):
    """Deck structure under the road surface (m): about 2 m for spans to 40 m, up to 4.5 m for long
    girders; a structure carried from above (arch, truss, cable-stayed, suspension) about 1.5 m."""
    kind = str(tags.get('bridge:structure', '')).lower()
    if kind in ('arch', 'truss', 'cable-stayed', 'suspension', 'simple-suspension', 'tied-arch'):
        return 1.5
    return float(np.clip(2.0 + (length - 40.) / 160. * 2.5, 2.0, 4.5))


def bridge_floors(segments, extras, sample, emit=None):
    """Lower bounds for decks over navigable water, by the rule chain of ROAD-OBJECTS.md 4.7:
    1. a seamark clearance (closed position of a movable bridge) or a maxheight on a node of the
       waterway within MARK_REACH m of the crossing;
    2. else the waterway's class (CEMT, or ship=yes, or boat=yes with motorboat=yes);
    3. else nothing (the default BRIDGE_CLEAR over the DEM stays).
    The deck top = water level (lowest DEM along the waterway within WATER_REACH m) + clearance +
    structure depth. Returns (floors [(segment index, s0, s1, z)], report [dict per crossing])."""
    import shapely
    from fast_profile import kind as segment_kind
    water = extras['water']
    if not water or not segments:
        return [], []
    wlines = shapely.linestrings(np.concatenate([w[2] for w in water]),
                                 indices=np.repeat(np.arange(len(water)), [len(w[2]) for w in water]))
    tree = shapely.STRtree(wlines)
    marks = extras['marks']
    mtree = shapely.STRtree(shapely.points([(m[2], m[3]) for m in marks])) if marks else None
    floors, report = [], []
    for k, seg in enumerate(segments):
        if segment_kind(seg['tags']) != 'bridge':
            continue
        line = shapely.linestrings(np.asarray(seg['xy'], float))
        for j in tree.query(line, predicate='intersects').tolist():
            wtags, = water[j][1]
            hit = shapely.intersection(line, wlines[j])
            for p in shapely.get_parts(hit).tolist():
                if p.geom_type != 'Point':
                    p = p.centroid
                rule, clearance, mark = None, None, None
                if mtree is not None:
                    near = mtree.query(p, predicate='dwithin', distance=MARK_REACH).tolist()
                    if near:
                        best = min(near, key=lambda i: math.hypot(marks[i][2] - p.x, marks[i][3] - p.y))
                        t = marks[best][1]
                        v = _number(t.get('seamark:bridge:clearance_height_closed'))
                        rule = 'seamark_closed'
                        if v is None:
                            v, rule = _number(t.get('seamark:bridge:clearance_height')), 'seamark'
                        if v is None:
                            v, rule = _number(t.get('maxheight')), 'maxheight'
                        if v is not None and 0 < v < 80:
                            clearance, mark = v, marks[best][0]
                if clearance is None:
                    v = _cemt(wtags.get('CEMT'))
                    if v is not None:
                        clearance, rule = v, 'cemt'
                    elif wtags.get('ship') == 'yes' or (wtags.get('boat') == 'yes' and wtags.get('motorboat') == 'yes'):
                        clearance, rule = SMALL_CRAFT, 'small_craft'
                if clearance is None:
                    continue
                # water level: the lowest DEM along the waterway near the crossing
                wl = wlines[j]
                at = shapely.line_locate_point(wl, p)
                ts = np.arange(max(0., at - WATER_REACH), min(shapely.length(wl), at + WATER_REACH) + 1e-6, 5.)
                pts = shapely.line_interpolate_point(wl, ts)
                level = sample(np.column_stack([shapely.get_x(pts), shapely.get_y(pts)]))
                level = level[np.isfinite(level)]
                if not len(level):
                    continue
                water_z = float(level.min())
                from fast_sources import LIFT
                if wtags.get('tidal') == 'yes' or (wtags.get('waterway') in ('river', 'fairway', 'tidal_channel')
                                                   and water_z < LIFT + TIDAL_BAND):
                    water_z = max(water_z, LIFT + MHW)
                L = float(shapely.length(line))
                depth = structure_depth(L, seg['tags'])
                z = water_z + clearance + depth
                # the stations over the channel: half its width across the bridge, plus 10 m
                width = _number(wtags.get('width'), WATER_WIDTH.get(wtags.get('waterway'), 30.))
                sp = float(shapely.line_locate_point(line, p))
                xy = np.asarray(seg['xy'], float)
                cum = np.r_[0., np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
                i = int(np.clip(np.searchsorted(cum, sp), 1, len(xy) - 1))
                d = xy[i] - xy[i - 1]
                q = shapely.line_interpolate_point(wl, [max(0., at - 5.), at + 5.])
                wd = np.array([shapely.get_x(q[1]) - shapely.get_x(q[0]), shapely.get_y(q[1]) - shapely.get_y(q[0])])
                sin = abs(d[0] * wd[1] - d[1] * wd[0]) / max(1e-9, np.hypot(*d) * np.hypot(*wd))
                half = width / 2 / max(sin, .3) + 10.
                floors.append((k, sp - half, sp + half, z))
                report.append({'segment': k, 'way': seg.get('way'), 'highway': seg['tags'].get('highway'),
                               'rule': rule, 'clearance': clearance, 'water': water_z, 'depth': depth, 'floor': z,
                               'span': [sp - half, sp + half], 'bridge_length': L, 'mark': mark,
                               'waterway': wtags.get('waterway'), 'cemt': wtags.get('CEMT')})
    _approach_check(segments, report, sample)
    if emit:
        rules = defaultdict(int)
        for r in report:
            rules[r['rule']] += 1
        short = sum(1 for r in report if r['approach_short'])
        emit(f'bridge floors: {len(floors)} deck spans over navigable water ({dict(rules)}); '
             f'{short} with an OSM approach shorter than the rise needs at the class grade')
    return floors, report


def _approach_check(segments, report, sample, reach=3000.):
    """The rise to a floor needs rise / GRADE_MAX m of approach on each side. Available: the road from the
    bridge's ends along ways of its class to the next junction (3+ ways), at most reach m; embankment=yes
    ways on it are counted apart. Writes approach_need, approach_have [two sides], embankment and
    approach_short (a side shorter than needed) into each report row."""
    from fast_profile import GRADE_MAX, GRADE_DEFAULT
    ends = defaultdict(list)
    for k, seg in enumerate(segments):
        for e in (0, 1):
            ends[seg['nodes'][e]].append((k, e))
    lengths = {}

    def length(k):
        if k not in lengths:
            xy = np.asarray(segments[k]['xy'], float)
            lengths[k] = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
        return lengths[k]
    for r in report:
        k = r['segment']
        seg = segments[k]
        hw = seg['tags'].get('highway')
        grade = GRADE_MAX.get(hw, GRADE_DEFAULT)
        ground = sample(np.asarray([seg['xy'][0], seg['xy'][-1]], float))
        have, emb = [], []
        for e in (0, 1):
            node, total, bank = seg['nodes'][e], 0., 0.
            seen = {k}
            while total < reach:
                nxt = [m for m in ends[node] if m[0] not in seen]
                if len(ends[node]) > 2 or len(nxt) != 1:
                    break
                m, me = nxt[0]
                if segments[m]['tags'].get('highway') != hw:
                    break
                seen.add(m)
                total += length(m)
                if segments[m]['tags'].get('embankment') not in (None, 'no') or segments[m]['tags'].get('bridge') not in (None, 'no'):
                    bank += length(m)
                node = segments[m]['nodes'][1 - me]
            have.append(round(total, 1)); emb.append(round(bank, 1))
        g = ground[np.isfinite(ground)]
        rise = r['floor'] - (float(g.min()) if len(g) else r['water'])
        need = max(0., rise) / grade
        r.update(approach_need=round(need, 1), approach_have=have, embankment=emb,
                 approach_short=bool(min(have) + length(k) / 2 < need))


# =========================================================================================================
# Graph: mouths, through pairs, branches, strokes
# =========================================================================================================

def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-12 else (0., 0.)


def _direction(xy, end, reach=REACH):
    """unit direction from a segment end into the segment"""
    n = len(xy)
    a = xy[0] if end == 0 else xy[-1]
    order = range(1, n) if end == 0 else range(n - 2, -1, -1)
    b = None
    for i in order:
        b = xy[i]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= reach:
            break
    return _unit(b[0] - a[0], b[1] - a[1])


class Graph:
    """Mouths at every node, the through pairs (joined into strokes or not), branches and node kinds."""

    def __init__(self, segments):
        self.segments = segments
        self.mouths = defaultdict(list)                 # node -> [(k, end, dx, dy)]
        self.travel = [travel(s['tags']) for s in segments]
        self.rank = [rank(s['tags']) for s in segments]
        for k, seg in enumerate(segments):
            xy = seg['xy']
            for e in (0, 1):
                dx, dy = _direction(xy, e)
                self.mouths[seg['nodes'][e]].append((k, e, dx, dy))
        self.join = {}                                   # (k, e) -> (k2, e2): one stroke through the node
        self.partner = {}                                # (k, e) -> (k2, e2): a through pair of two strokes
        self.pairs = {}                                  # node -> [(i, j, joined)] (mouth indices)
        self.branch = {}                                 # (k, e) -> (host mouth (k, e), node)
        self.kind = {}
        cos_through, cos_branch = math.cos(THROUGH), math.cos(BRANCH)
        for node, ms in self.mouths.items():
            n = len(ms)
            if n == 1:
                self.kind[node] = 'end'
                continue
            flows = [self.flow(m) for m in ms]
            options = []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = ms[i], ms[j]
                    if a[0] == b[0]:
                        continue                         # a loop way's two ends
                    c = a[2] * b[2] + a[3] * b[3]
                    if n > 2 and c > -cos_through:
                        continue
                    if flows[i] * flows[j] > 0:
                        continue                         # both flow in or both out: not one road
                    ta, tb = segments[a[0]]['tags'], segments[b[0]]['tags']
                    ring_a = ta.get('junction') in ('roundabout', 'circular')
                    if n > 2 and ring_a != (tb.get('junction') in ('roundabout', 'circular')):
                        continue
                    joined = ta.get('highway') == tb.get('highway') and (flows[i] == 0) == (flows[j] == 0)
                    options.append((not joined, c, i, j, joined))
            used, pairs = set(), []
            for _, c, i, j, joined in sorted(options):
                if i in used or j in used:
                    continue
                used |= {i, j}
                pairs.append((i, j, joined))
                a, b = (ms[i][0], ms[i][1]), (ms[j][0], ms[j][1])
                (self.join if joined else self.partner)[a] = b
                (self.join if joined else self.partner)[b] = a
            self.pairs[node] = pairs
            rest = [i for i in range(n) if i not in used]
            branches = 0
            for i in rest:
                m = ms[i]
                best = None
                for p in pairs:
                    for h in p[:2]:
                        hm = ms[h]
                        c = m[2] * hm[2] + m[3] * hm[3]
                        if c <= cos_branch or self.rank[m[0]] > self.rank[hm[0]] + 1e-9:
                            continue
                        fm, fh = flows[i], flows[h]
                        if fm != 0 and fh != 0 and fm != fh:
                            continue
                        if fm != 0 and fh == 0 or fm == 0 and fh != 0:
                            continue
                        if best is None or c > best[0]:
                            best = (c, h)
                if best is not None:
                    self.branch[(m[0], m[1])] = ((ms[best[1]][0], ms[best[1]][1]), node)
                    branches += 1
            if n == 2 and pairs:
                self.kind[node] = 'joint'
            elif pairs and branches == len(rest) and rest:
                f = [flows[i] for i in rest]
                self.kind[node] = 'diverge' if all(v > 0 for v in f) else 'merge' if all(v < 0 for v in f) else 'fork'
            elif pairs and not rest:
                self.kind[node] = 'cross' if n > 2 else 'joint'
            else:
                self.kind[node] = 'junction'

    def flow(self, m):
        """+1 the road's traffic leaves the node, -1 it arrives, 0 two-way"""
        t = self.travel[m[0]]
        return t * (1 if m[1] == 0 else -1)

    def strokes(self):
        """[(parts [(k, reversed)], closed)] oriented in the direction of travel (one-way) or from the lower
        end node id (two-way); a ring starts at its lowest node."""
        nseg = len(self.segments)
        seen = [False] * nseg
        out = []
        for k0 in range(nseg):
            if seen[k0]:
                continue
            # back to the first segment: leave by end 0 while a joined pair runs on
            k, e, steps = k0, 0, 0
            ring = False
            while (k, e) in self.join and steps <= nseg:
                w, ew = self.join[(k, e)]
                if seen[w]:
                    break
                if w == k0:
                    ring = True
                    break
                k, e = w, 1 - ew
                steps += 1
            parts, closed = [], False
            cur = (k0, 0) if ring else (k, e)
            entry = cur[1]
            first = cur[0]
            while True:
                k, e = cur
                seen[k] = True
                parts.append((k, e == 1))
                out_end = 1 - e
                nxt = self.join.get((k, out_end))
                if nxt is None:
                    break
                if nxt[0] == first and nxt[1] == entry:
                    closed = True
                    break
                if seen[nxt[0]]:
                    break
                cur = nxt
            nodes = [self.segments[k]['nodes'][1 if r else 0] for k, r in parts] + \
                    [self.segments[parts[-1][0]]['nodes'][0 if parts[-1][1] else 1]]
            t = self.travel[parts[0][0]] * (-1 if parts[0][1] else 1)
            flip = t < 0 if t != 0 else (not closed and nodes[0] > nodes[-1])
            if flip:
                parts = [(k, not r) for k, r in reversed(parts)]
                nodes = nodes[::-1]
            if closed:
                # a ring starts at its lowest node: the same stroke whichever segment was met first
                at = int(np.argmin(nodes[:-1]))
                parts = parts[at:] + parts[:at]
            out.append((parts, closed))
        return out


# =========================================================================================================
# Stations of a stroke
# =========================================================================================================

def _project(P, Q):
    """arc length along polyline Q of the closest point to each of P (brute force; small arrays)"""
    a, b = Q[:-1], Q[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - a[None]) * d[None]).sum(2) / L2[None], 0, 1)
    c = a[None] + t[..., None] * d[None]
    dist = ((P[:, None, :] - c) ** 2).sum(2)
    j = dist.argmin(1)
    cum = np.r_[0., np.cumsum(np.sqrt(L2))]
    return cum[j] + t[np.arange(len(P)), j] * np.sqrt(L2[j]), np.sqrt(dist[np.arange(len(P)), j])


def _simplify(V, fixed):
    """indices of the vertices a fit uses: every fixed one; free ones MIN_LEG m from the kept neighbours and
    more than SIMPLIFY m off the chord between them (Douglas-Peucker between consecutive fixed vertices)"""
    n = len(V)
    keep = [0]
    for i in range(1, n):
        if not fixed[i] and math.hypot(V[i, 0] - V[keep[-1], 0], V[i, 1] - V[keep[-1], 1]) < MIN_LEG:
            continue
        if fixed[i] and len(keep) > 1 and not fixed[keep[-1]] and                 math.hypot(V[i, 0] - V[keep[-1], 0], V[i, 1] - V[keep[-1], 1]) < MIN_LEG:
            keep[-1] = i
            continue
        keep.append(i)
    if keep[-1] != n - 1:
        keep.append(n - 1)
    out = []
    anchors = [i for i in keep if fixed[i]]
    pos = {v: j for j, v in enumerate(keep)}
    for a, b in zip(anchors, anchors[1:]):
        span = keep[pos[a]:pos[b] + 1]
        flag = [False] * len(span)
        flag[0] = flag[-1] = True
        stack = [(0, len(span) - 1)]
        while stack:
            lo, hi = stack.pop()
            if hi - lo < 2:
                continue
            p, q = V[span[lo]], V[span[hi]]
            d = q - p
            L = math.hypot(d[0], d[1])
            pts = V[span[lo + 1:hi]]
            if L < 1e-9:
                dist = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
            else:
                dist = np.abs((pts[:, 0] - p[0]) * d[1] - (pts[:, 1] - p[1]) * d[0]) / L
            j = int(np.argmax(dist))
            if dist[j] > SIMPLIFY:
                m = lo + 1 + j
                flag[m] = True
                stack += [(lo, m), (m, hi)]
        out += [v for v, f in zip(span, flag) if f][:-1]
    out.append(anchors[-1])
    return out


# --- S-shaped jogs in the OSM line (bundle8 review) --------------------------------------------------------------------
# A way's first vertices after a fork node can lie 2-3 m beside the line of the way before the node: the line turns away by 25-30
# degrees, runs 5-10 m and turns back (an S-shaped sideways step; the DOP20 aerial and ALKIS show a straight road, the map has
# the node a little off). Rule: four consecutive kept vertices a, b, c, d of a motorway / trunk / link line with opposite turns at
# b and c of at least JOG_TURN each that nearly cancel (|sum| <= JOG_NET), b-c no longer than JOG_LEN and the step
# |bc| * sin(turn) between JOG_SHIFT's limits: the vertex that is free (c, else b; both when both are) is dropped and the line
# runs on from the other; a fixed node keeps its place and takes the chord a-d as its tangent. Interchange zone only: a link, or
# a motorway / trunk within JOG_ZONE m (along the line) of a diverge, merge or fork node (EARTHNG_ZONE; 0 = rule off).
# OFF by default (EARTHNG_ZONE=150 turns it on): it finds 15 jogs in the Niedersachsen pack, 2 along each of the two routes, and at those
# two a smoothed link line made 5-7 cm steps at the junction plates in testing, so it is not worth its risk
# until the zone rule is settled (the tested "jog" is not one of the 15: see ROAD-OBJECTS.md 14).
JOG_ZONE = float(os.environ.get('EARTHNG_ZONE', '0') or 0.)
JOG_TURN, JOG_NET, JOG_LEN, JOG_SHIFT = math.radians(10.), math.radians(12.), 40., (.8, 4.5)
JOG_MIN_STROKE = 100.   # m: shorter strokes only get the hook rule at their ends
JOG_LOG = []            # (x, y, highway, step m) of every jog found; local debugging aid, never written by the exporter
JOG_COUNT = [0]


def _hook(V, b, c, d, h1):
    """is c a hook of the line b-c-d: it turns away from the heading h1 (the way before b) and back within the JOG limits?"""
    bc, cd = V[c] - V[b], V[d] - V[c]
    lbc, lcd = math.hypot(*bc), math.hypot(*cd)
    if min(lbc, lcd) < 2. or lbc > JOG_LEN:
        return False
    h2, h3 = math.atan2(bc[1], bc[0]), math.atan2(cd[1], cd[0])
    t1 = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
    t2 = (h3 - h2 + math.pi) % (2 * math.pi) - math.pi
    return (t1 * t2 < 0 and abs(t1) >= JOG_TURN and abs(t2) >= JOG_TURN and abs(t1 + t2) <= JOG_NET
            and JOG_SHIFT[0] <= lbc * abs(math.sin(t1)) <= JOG_SHIFT[1])


def _unjog(V, fixed, keep, vnode, Sosm, graph, highway, tangents=None):
    """(dropped vertex indices, {fixed vertex: new tangent}) for the S-shaped jogs among the kept vertices. Also the hook at the
    start or end of a branch stroke: its first free vertex turns 25-30 degrees away from the host's direction at the fork node and
    back within 10 m (seen in testing: the first vertex of a loop lies 4 m beside the host line, the lane markings follow it)."""
    if JOG_ZONE <= 0. or not (highway.endswith('_link') or highway in ('motorway', 'trunk')) or len(keep) < 3:
        return set(), {}
    forkish = ('diverge', 'merge', 'fork')
    ends = set()
    if tangents:
        for pos, step in ((0, 1), (len(keep) - 1, -1)):
            b = keep[pos]
            if fixed[b] and b in tangents and graph.kind.get(vnode[b]) in forkish and 0 <= pos + 2 * step < len(keep):
                c, d = keep[pos + step], keep[pos + 2 * step]
                t = tangents[b]
                h1 = math.atan2(t[1], t[0]) if step > 0 else math.atan2(-t[1], -t[0])      # the host's heading, seen along the walk b -> c -> d
                if not fixed[c] and _hook(V, b, c, d, h1):
                    ends.add(c)
                    JOG_COUNT[0] += 1
                    JOG_LOG.append((float(V[b][0]), float(V[b][1]), highway, 0.))
    if ends:
        keep = [i for i in keep if i not in ends]
    # an interior jog needs room: a short link between two junction plates (53 m, in testing) got 5-7 cm steps at its plates
    if len(keep) < 4 or Sosm[-1] < JOG_MIN_STROKE:
        return set(ends), {}
    link = highway.endswith('_link')
    forks = [Sosm[i] for i in range(len(V)) if vnode[i] is not None and graph.kind.get(vnode[i]) in ('diverge', 'merge', 'fork')]
    seq = list(keep)
    drop, retan = set(), {}
    j = 1
    while j + 2 < len(seq):
        a, b, c, d = seq[j - 1], seq[j], seq[j + 1], seq[j + 2]
        ab, bc, cd = V[b] - V[a], V[c] - V[b], V[d] - V[c]
        lab, lbc, lcd = math.hypot(*ab), math.hypot(*bc), math.hypot(*cd)
        if min(lab, lbc, lcd) < 2. or lbc > JOG_LEN or (fixed[b] and fixed[c]):
            j += 1
            continue
        h1, h2, h3 = math.atan2(ab[1], ab[0]), math.atan2(bc[1], bc[0]), math.atan2(cd[1], cd[0])
        t1 = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
        t2 = (h3 - h2 + math.pi) % (2 * math.pi) - math.pi
        step = lbc * abs(math.sin(t1))
        ok = (t1 * t2 < 0 and abs(t1) >= JOG_TURN and abs(t2) >= JOG_TURN and abs(t1 + t2) <= JOG_NET and JOG_SHIFT[0] <= step <= JOG_SHIFT[1])
        if ok and not link:
            ok = any(abs(f - Sosm[b]) <= JOG_ZONE for f in forks)
        if not ok:
            j += 1
            continue
        JOG_COUNT[0] += 1
        JOG_LOG.append((float(V[b][0]), float(V[b][1]), highway, round(step, 2)))
        if not fixed[c] and not fixed[b]:
            for v in (b, c): drop.add(v)
            del seq[j:j + 2]
        elif not fixed[c]:
            drop.add(c)
            chord = V[d] - V[a]; l = math.hypot(*chord)
            retan[b] = (chord[0] / l, chord[1] / l)
            del seq[j + 1]
        else:
            drop.add(b)
            chord = V[d] - V[a]; l = math.hypot(*chord)
            retan[c] = (chord[0] / l, chord[1] / l)
            del seq[j]
    return drop | ends, retan


def lane_pinch(segments):
    """A short one-way piece with fewer lanes than the road before it, between that road and a fork whose branches carry at least
    that many lanes together, has the lanes of the road before it (review: a two-lane slip road that narrows
    to one lane for 15 m before it splits into two slip roads). Returns the number of pieces changed; the pieces' tags are copied."""
    ends = defaultdict(list)                 # node -> [(segment index, enters here?)]
    for k, s in enumerate(segments):
        t = travel(s['tags'])
        if t == 0 or s['tags'].get('highway', '').replace('_link', '') not in ('motorway', 'trunk'):
            continue
        n0, n1 = s['nodes']
        head, tail = (n0, n1) if t > 0 else (n1, n0)         # traffic enters at head, leaves at tail
        s['_head'], s['_tail'] = head, tail
        ends[head].append((k, True)); ends[tail].append((k, False))
    changed = 0

    def lanes(s):
        return _number(s['tags'].get('lanes'))
    for k, s in enumerate(segments):
        if '_head' not in s or lanes(s) is None:
            continue
        L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(s['xy'], s['xy'][1:]))
        if L >= PINCH_LEN:
            continue
        before = [segments[q] for q, enters in ends[s['_head']] if not enters and q != k]
        after = [segments[q] for q, enters in ends[s['_tail']] if enters and q != k]
        if len(before) != 1 or len(after) < 2 or lanes(before[0]) is None or any(lanes(x) is None for x in after):
            continue
        up = lanes(before[0])
        if lanes(s) < up and sum(lanes(x) for x in after) >= up:
            tags = dict(s['tags'])
            tags['lanes'] = str(int(up))
            tags.pop('turn:lanes', None)
            s['tags'] = tags
            s['width'] = max(float(s['width']), float(before[0]['width']))
            if before[0].get('edges') is not None and s.get('edges') is not None and travel(before[0]['tags']) == travel(tags):
                s['edges'] = [max(a, b) for a, b in zip(s['edges'], before[0]['edges'])]
                s['width'] = max(s['width'], sum(s['edges']))
            s.pop('lane', None)
            PINCH_LOG.append((float(s['xy'][0][0]), float(s['xy'][0][1]), int(lanes(before[0])), round(L, 1)))
            changed += 1
    for s in segments:
        s.pop('_head', None); s.pop('_tail', None)
    return changed


PINCH_LEN = 40.
PINCH_LOG = []


def _resample(F, step):
    """F with points inserted so no chord is longer than step; returns (points, index of each F point)"""
    out, index = [F[0]], [0]
    for a, b in zip(F[:-1], F[1:]):
        n = int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step - 1e-9))
        for i in range(1, n):
            out.append((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n))
        out.append(b)
        index.append(len(out) - 1)
    return np.asarray(out, float), np.asarray(index, int)


def _headings(P):
    d = np.diff(P, axis=0)
    ch = np.arctan2(d[:, 1], d[:, 0])
    h = np.empty(len(P))
    h[0], h[-1] = ch[0], ch[-1]
    if len(P) > 2:
        u = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
        m = u[:-1] + u[1:]
        h[1:-1] = np.arctan2(m[:, 1], m[:, 0])
    return h


def _unwrap_to(h, ref):
    return h + 2 * math.pi * np.round((ref - h) / (2 * math.pi))


def _curvature(s, h, w=CURVE_WINDOW):
    lo, hi = np.maximum(s - w, s[0]), np.minimum(s + w, s[-1])
    span = np.maximum(hi - lo, 1e-6)
    k = (np.interp(hi, s, h) - np.interp(lo, s, h)) / span
    return np.where(hi - lo > 1., k, 0.)


def _dilate(s, v, rate):
    """max over j of v[j] - rate |s_i - s_j| (v >= 0): a requirement spread at a limited slope"""
    out = v.copy()
    if not out.any():
        return out
    for i in range(1, len(s)):
        out[i] = max(out[i], out[i - 1] - rate * (s[i] - s[i - 1]))
    for i in range(len(s) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - rate * (s[i + 1] - s[i]))
    return out


def urban(tags):
    """a built-up road by its speed limit (RASt: no superelevation there)"""
    v = str(tags.get('maxspeed', '')).lower()
    if 'urban' in v or 'zone' in v or v in ('walk', 'living_street'):
        return True
    n = _number(v)
    return n is not None and n <= 50


def bank_params(highway, oneway, built_up=False):
    if built_up and not oneway:
        return None
    fam = family(highway)
    if fam == 'trunk' and oneway:
        fam = 'trunk_oneway'
    return BANK.get(fam)


def crossfall(s, kappa, highway, oneway, eL, eR, built_up=False):
    """(fl, fr): slope of the left and the right half of the section, rise per metre going outward from
    the centre line (crown: both -2.5 %; one-way on a straight: fl +2.5 %, fr -2.5 %)."""
    n = len(s)
    params = bank_params(highway, oneway, built_up)
    a = float(np.median(np.maximum(eL, eR))) if n else 4.
    rate = RUNOFF_MAX / a if a >= 4. else RUNOFF_SMALL
    if params is None:
        tau = np.full(n, Q0 if oneway else 0.)
    else:
        rmin, rout, qmax, _ = params
        R = 1. / np.maximum(np.abs(kappa), 1e-9)
        q = Q0 + (qmax - Q0) * np.clip((1. / R - 1. / rout) / (1. / rmin - 1. / rout), 0., 1.)
        if oneway and str(highway).endswith('_link'):
            q = np.where(R < LOOP_R, np.minimum(q, LOOP_Q_MAX), q)
        curve = R < rout
        # tau: + left edge higher (a right-hand curve banks so)
        tau_t = np.where(curve, -np.sign(kappa) * q, 0.)
        base = Q0 if oneway else 0.
        if oneway:
            tau_t = np.where(curve, tau_t, Q0)
        delta = tau_t - base
        pos, neg = np.maximum(delta, 0.), np.maximum(-delta, 0.)
        # a change of cross-fall takes at least CROSSFALL_EASE m: the bank spreads at no more than the swing over that length
        # (an S-bend's two ramps run in opposite senses at the same time: half the rate each; the loop end of a tested loop
        # flipped +5.3 % -> -3.9 % in 50 m)
        swing = float(pos.max() + neg.max()) if pos.any() and neg.any() else float(max(pos.max(), neg.max()))
        rate = min(rate, swing / (CROSSFALL_EASE * (2. if pos.any() and neg.any() else 1.))) if swing > 0 else rate
        tau = base + _dilate(s, pos, rate) - _dilate(s, neg, rate)
    if oneway:
        return tau, -tau
    u = np.abs(tau)
    outer = np.where(u < Q0, 2 * u - Q0, u)
    inner = -np.maximum(Q0, u)
    left_outer = tau > 0
    return np.where(left_outer, outer, inner), np.where(left_outer, inner, outer)


def loop_widen(s, kappa, eL, eR):
    """(eL, eR) with the paved width raised to LOOP_WIDTH on the inner side of a bend tighter than LOOP_R, easing in over the
    bend's own length (smoothstep of the curvature between 1/(1.5 R) and 1/R). Inner: the ALKIS pavement of a measured
    loop lies 5.0-6.8 m right of the OSM line of a right-hand loop and 2.1-3.6 m left (the extra goes inside)."""
    k = np.abs(kappa)
    w = np.clip((k - 1. / (1.5 * LOOP_R)) / (1. / LOOP_R - 1. / (1.5 * LOOP_R)), 0., 1.)
    w = w * w * (3 - 2 * w)
    # not at the ends: there the link leaves or joins its host's carriageway, whose lanes the extra width would overlap
    e = np.clip(np.minimum(s - s[0], s[-1] - s) / LOOP_EASE, 0., 1.)
    w = w * e * e * (3 - 2 * e)
    if not w.any():
        return eL, eR
    extra = np.maximum(LOOP_WIDTH - (eL + eR), 0.) * w
    # kappa > 0: a left turn, the inner side is the left
    return np.where(kappa >= 0, eL + extra, eL), np.where(kappa < 0, eR + extra, eR)


def lanes_of(tags, rev):
    """(n forward, n backward, n both ways) in the stroke's direction, and the turn lanes forward/backward"""
    import fast_sources
    t = travel(tags)
    motorway = tags.get('highway') == 'motorway'
    n = int(max(1, round(fast_sources.number(tags.get('lanes'), 2 if motorway or t == 0 else 1))))
    if t != 0:
        fwd, bwd, both = n, 0, 0
        tf, tb = tags.get('turn:lanes', ''), ''
        if (t < 0) != rev:
            fwd, bwd, tf, tb = bwd, fwd, tb, tf
        return fwd, bwd, both, tf, tb
    f = fast_sources.number(tags.get('lanes:forward'), None)
    b = fast_sources.number(tags.get('lanes:backward'), None)
    if f is None or b is None:
        if n == 1:
            f, b = 0, 0
        else:
            f = b = n // 2
    f, b = int(f), int(b)
    both = max(0, n - f - b)
    tf, tb = tags.get('turn:lanes:forward', ''), tags.get('turn:lanes:backward', '')
    if rev:
        f, b, tf, tb = b, f, tb, tf
    return f, b, both, tf, tb


def section_of(seg, rev, oneway):
    """paved extents (left, right) of the lane centre line in the stroke's direction, and the lane width"""
    import fast_sources
    tags = seg['tags']
    width = float(seg['width'])
    sl, sr = fast_sources.shoulder(tags)
    if seg.get('edges') is not None:
        # the official pavement (official.py): its edges against the OSM line, node order; the lane width is its own
        # where the extra width is a hard shoulder (a lane block of 4 m is not 7 m wide because the shoulder is)
        l, r = seg['edges']
        f, b, both, _, _ = lanes_of(tags, rev)
        n = max(1, f + b + both)
        lane = float(seg['lane']) if seg.get('lane') else max(.5, (l + r - sl - sr) / n)
        return (r, l, lane) if rev else (l, r, lane)
    tagged = fast_sources.number(tags.get('shoulder:left:width'), None) is not None
    # tagged shoulders are left / right of the node order, class shoulders of the direction of travel
    if tagged and rev:
        sl, sr = sr, sl
    lanes = width - sl - sr
    f, b, both, _, _ = lanes_of(tags, rev)
    n = max(1, f + b + both)
    return lanes / 2 + sl, lanes / 2 + sr, lanes / n


class Stroke:
    __slots__ = ('id', 'parts', 'closed', 'highway', 'oneway', 'tags', 's', 'x', 'y', 'z', 'h', 'k', 'eL', 'eR',
                 'fl', 'fr', 'part_of', 'part_s', 'nodes', 'fitted', 'lanes', 'events', 'furniture', 'structures',
                 'draw', 'attach', 'osm_dev', 'vertex_s', 'split')

    def length(self):
        return float(self.s[-1])


def _split_joint(graph, segments, node, ms, i, j):
    """the joint road's mouth (k, e, dx, dy) when through pair (i, j) at node is a lane split's (lane_splits): a one-way
    diverge or merge of three mouths whose joint road carries the lanes of both others, else None"""
    if len(ms) != 3 or graph.kind.get(node) not in ('diverge', 'merge'):
        return None
    r = 3 - i - j
    if any(graph.travel[m[0]] == 0 for m in ms):
        return None
    flows = [graph.flow(m) for m in ms]
    jt = i if flows[i] != flows[r] else j            # the pair's mouth that flows the other way than the branch
    other = j if jt == i else i

    def lanes(m):
        f, b_, both, _, _ = lanes_of(segments[m[0]]['tags'], False)
        return f + b_ + both
    if lanes(ms[jt]) != lanes(ms[other]) + lanes(ms[r]):
        return None
    return ms[jt]


def build_strokes(segments, graph, emit=None, fit=True):
    """Every stroke with its stations; heights from the segments' solved profiles."""
    import fast_curves
    t0 = time.perf_counter()
    # node tangent lines: one per through pair (sign-free), by the pair's mouth directions
    pair_line = {}
    for node, pairs in graph.pairs.items():
        ms = graph.mouths[node]
        for i, j, _ in pairs:
            a, b = ms[i], ms[j]
            tx, ty = _unit(b[2] - a[2], b[3] - a[3])
            joint = _split_joint(graph, segments, node, ms, i, j)
            if joint is not None:
                # a lane split: its branches leave along the joint road, so the tangent is the joint road's own direction (the
                # bisector took half the turn of the through branch's first OSM chord: 0.6 degrees with vertices 20 m apart,
                # and the fit bent the approach 0.26 m sideways within 36 m of the node)
                tx, ty = _unit(-joint[2], -joint[3])
            pair_line[(a[0], a[1])] = (tx, ty)
            pair_line[(b[0], b[1])] = (tx, ty)

    def mouth_line(k, e):
        if (k, e) in pair_line:
            return pair_line[(k, e)]
        host = graph.branch.get((k, e))
        if host is not None:
            return pair_line.get(host[0])
        return None

    strokes = []
    ids = set()
    fitted_n = 0
    for parts, closed in graph.strokes():
        st = Stroke()
        k0, r0 = parts[0]
        seg0 = segments[k0]
        st.parts, st.closed = parts, closed
        st.highway = seg0['tags'].get('highway', '')
        st.tags = seg0['tags']
        st.oneway = graph.travel[k0] != 0
        start_node = seg0['nodes'][1 if r0 else 0]
        kl, rl = parts[-1]
        end_node = segments[kl]['nodes'][0 if rl else 1]
        # OSM ids only: the same stroke in every index of the same graph
        sid = f'{seg0.get("way", k0)}_{start_node}_{end_node}'
        while sid in ids:
            sid += '+'
        ids.add(sid)
        st.id = sid
        # the stroke's vertices, nodes at part boundaries, cumulative OSM arc length and profile
        V, vnode, vpart = [], [], []
        Sprof, Hprof = [], []
        off = 0.
        bounds = [0.]
        for p, (k, rev) in enumerate(parts):
            seg = segments[k]
            xy = np.asarray(seg['xy'], float)
            ps, ph = (np.asarray(seg['prof'][0], float), np.asarray(seg['prof'][1], float))
            L = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
            if rev:
                xy = xy[::-1]
                ps, ph = L - ps[::-1], ph[::-1]
            n0, n1 = (seg['nodes'][1], seg['nodes'][0]) if rev else (seg['nodes'][0], seg['nodes'][1])
            for i in range(len(xy)):
                if p > 0 and i == 0:
                    continue
                V.append(xy[i])
                vnode.append(n0 if i == 0 else (n1 if i == len(xy) - 1 else None))
                vpart.append(p)
            keep = slice(1, None) if p > 0 else slice(None)
            Sprof.append(ps[keep] + off)
            Hprof.append(ph[keep])
            off += L
            bounds.append(off)
        V = np.asarray(V, float)
        Sosm = np.r_[0., np.cumsum(np.hypot(*np.diff(V, axis=0).T))]
        Sprof, Hprof = np.concatenate(Sprof), np.concatenate(Hprof)
        Sprof = np.maximum.accumulate(Sprof)
        # fixed vertices: stroke ends and nodes where anything else meets the stroke
        nv = len(V)
        fixed = np.zeros(nv, bool)
        fixed[0] = fixed[-1] = True
        for i in range(1, nv - 1):
            if vnode[i] is not None and len(graph.mouths[vnode[i]]) > 2:
                fixed[i] = True
        # tangent lines at fixed vertices, signed along the stroke
        tangents = {}
        for i in np.flatnonzero(fixed).tolist():
            node = vnode[i]
            p = vpart[i]
            if i == 0:
                k, rev = parts[0]
                line = mouth_line(k, 1 if rev else 0)
            elif i == nv - 1:
                k, rev = parts[-1]
                line = mouth_line(k, 0 if rev else 1)
            else:
                k, rev = parts[p]
                line = mouth_line(k, 0 if rev else 1)
            if line is None:
                continue
            a, b = V[max(0, i - 1)], V[min(nv - 1, i + 1)]
            if (b[0] - a[0]) * line[0] + (b[1] - a[1]) * line[1] < 0:
                line = (-line[0], -line[1])
            tangents[i] = line
        params = fast_curves.CLASSES.get(st.highway) if fit else None
        st.fitted = bool(params is not None and nv >= 2 and (nv > 2 or tangents))
        if st.fitted:
            # mapping noise goes before the fit: free vertices under MIN_LEG from a kept one, then those within
            # SIMPLIFY of their neighbours' chord (a 1 m zigzag at a joint was rounded at a radius of 1-2 m)
            keep = _simplify(V, fixed)
            drop, retan = _unjog(V, fixed, keep, vnode, Sosm, graph, st.highway, tangents)
            if drop:
                keep = [i for i in keep if i not in drop]
                tangents.update(retan)
            kv = V[keep]
            kfixed = [bool(fixed[i]) for i in keep]
            ktan = {}
            for n_, i in enumerate(keep):
                if i in tangents:
                    tx, ty = tangents[i]
                    if 0 < n_ < len(keep) - 1:
                        u = _unit(*(kv[n_] - kv[n_ - 1])); v = _unit(*(kv[n_ + 1] - kv[n_]))
                        half = math.acos(max(-1., min(1., u[0] * v[0] + u[1] * v[1]))) / 2
                    else:
                        c = kv[1] - kv[0] if n_ == 0 else kv[-1] - kv[-2]
                        u = _unit(*c)
                        half = math.acos(max(-1., min(1., u[0] * tx + u[1] * ty))) / 2
                    ktan[n_] = (tx, ty, max(half, 1e-3))
            pts, index = fast_curves.curve([tuple(q) for q in kv], kfixed, ktan, params)
            F = np.asarray(pts, float)
            fidx = np.asarray(index, int)                          # F index of each kept fixed vertex
            kept_fixed = [keep[n_] for n_ in range(len(keep)) if kfixed[n_]]
            fitted_n += 1
        else:
            F = V.copy()
            fidx = np.flatnonzero(fixed)
            kept_fixed = fidx.tolist()
        P, Pidx = _resample(F, MAX_STATION)
        s = np.r_[0., np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        h = _headings(P)
        # exact node tangents at the fixed vertices
        for vi, fi in zip(kept_fixed, fidx.tolist()):
            if vi in tangents:
                h[Pidx[fi]] = math.atan2(tangents[vi][1], tangents[vi][0])
        h = np.unwrap(h)
        # where every OSM vertex lies on the fitted line: fixed ones exactly, the rest projected between
        vs = np.empty(nv)
        anchor_v = list(kept_fixed)
        anchor_s = [s[Pidx[fi]] for fi in fidx.tolist()]
        for v, sv in zip(anchor_v, anchor_s):
            vs[v] = sv
        dev = []
        if st.fitted:
            for (va, sa), (vb, sb) in zip(zip(anchor_v, anchor_s), zip(anchor_v[1:], anchor_s[1:])):
                if vb - va <= 1:
                    continue
                ia, ib = int(np.searchsorted(s, sa - 1e-6)), int(np.searchsorted(s, sb + 1e-6))
                Q = P[ia:max(ib, ia + 2)]
                proj, dist = _project(V[va + 1:vb], Q)
                vs[va + 1:vb] = np.clip(s[ia] + proj, sa, sb)
                dev.append(dist)
            vs = np.maximum.accumulate(vs)
        else:
            vs = s[Pidx]
        st.osm_dev = np.concatenate(dev) if dev else np.zeros(0)
        st.vertex_s = vs
        st.s, st.x, st.y, st.h = s, P[:, 0], P[:, 1], h
        st.k = _curvature(s, h)
        sosm = np.interp(s, vs, Sosm)
        st.z = np.interp(sosm, Sprof, Hprof)
        # parts on the stations, nodes at their s
        bs = np.interp(np.asarray(bounds), Sosm, vs)
        st.part_s = bs
        st.part_of = np.clip(np.searchsorted(bs, s, side='right') - 1, 0, len(parts) - 1)
        st.nodes = [(vnode[i], float(vs[i]), i) for i in range(nv) if vnode[i] is not None]
        # extents and lanes per part, tapers on the wider side of a change
        eL = np.empty(len(s)); eR = np.empty(len(s))
        sec = [section_of(segments[k], rev, st.oneway) for k, rev in parts]
        for p, (a, b, _) in enumerate(sec):
            m = st.part_of == p
            eL[m], eR[m] = a, b
        taper = TAPER.get(family(st.highway), TAPER['default'])
        for p in range(1, len(parts)):
            sj = bs[p]
            nj = next((n for n, sv, _ in st.nodes if abs(sv - sj) < 1e-3), None)
            # a merge, diverge or fork: the side its branch lies on (+1 left, -1 right; 0 none or both)
            bside = 0
            if graph.kind.get(nj) in ('merge', 'diverge', 'fork'):
                hj = float(np.interp(sj, s, h))
                own = (parts[p - 1][0], parts[p][0])
                sides = {1 if math.cos(hj) * dy - math.sin(hj) * dx > 0 else -1
                         for k2, _, dx, dy in graph.mouths[nj] if k2 not in own}
                bside = sides.pop() if len(sides) == 1 else 0
            for arr, side in ((eL, 0), (eR, 1)):
                a, b = sec[p - 1][side], sec[p][side]
                if abs(a - b) < 1e-3 or bside == (1 if side == 0 else -1):
                    continue
                if b > a:
                    T = min(taper, (bs[p + 1] - sj) / 2) if p + 1 < len(bs) else taper
                    m = (s > sj) & (s < sj + T)
                    arr[m] = a + (b - a) * _smoothstep((s[m] - sj) / max(T, 1e-6))
                else:
                    T = min(taper, (sj - bs[p - 1]) / 2)
                    m = (s > sj - T) & (s < sj)
                    arr[m] = a + (b - a) * _smoothstep((s[m] - (sj - T)) / max(T, 1e-6))
        if st.oneway and st.highway.endswith('_link') and max(lanes_of(segments[k]['tags'], rev)[0] + lanes_of(segments[k]['tags'], rev)[1]
                                                               for k, rev in parts) == 1:
            wL, wR = loop_widen(s, st.k, eL, eR)
            # a part with an official pavement (ALKIS edges) already has its own side and width: the rule is for the rest
            own = np.asarray([segments[k].get('edges') is not None for k, _ in parts])[st.part_of]
            eL, eR = np.where(own, eL, wL), np.where(own, eR, wR)
        st.eL, st.eR = eL, eR
        st.fl, st.fr = crossfall(s, st.k, st.highway, st.oneway, eL, eR, urban(st.tags))
        if st.oneway and seg0['tags'].get('junction') in ('roundabout', 'circular'):
            # a roundabout's ring falls outward at Q0 all round, no banking (phase 3; RASt 06 / Merkblatt
            # Kreisverkehre from memory, NOT verified): the side towards the ring's centre is the higher
            inward = 1. if float(np.median(st.k)) >= 0 else -1.
            st.fl, st.fr = np.full(len(s), inward * Q0), np.full(len(s), -inward * Q0)
        runs = []
        for p, (k, rev) in enumerate(parts):
            f, b, both, tf, tb = lanes_of(segments[k]['tags'], rev)
            lw = sec[p][2]
            row = [float(bs[p]), float(bs[p + 1]), f, b, both, round(lw, 2), tf, tb]
            if runs and runs[-1][2:] == row[2:] and abs(runs[-1][1] - row[0]) < 1e-6:
                runs[-1][1] = row[1]
            else:
                runs.append(row)
        st.lanes = runs
        st.events, st.furniture, st.structures = [], [], []
        st.draw = [0., float(s[-1])]
        st.attach = []
        st.split = []
        strokes.append(st)
    if emit:
        emit(f'roads v3: {len(strokes)} strokes ({fitted_n} fitted) from {len(segments)} segments in '
             f'{time.perf_counter() - t0:.1f}s')
    return strokes


# =========================================================================================================
# Exits and entries
# =========================================================================================================

def _frame_at(st, sv):
    """position, unit tangent, left normal, z, (fl, fr) and heading of a stroke at arc lengths sv"""
    s = st.s
    x, y = np.interp(sv, s, st.x), np.interp(sv, s, st.y)
    h = np.interp(sv, s, st.h)
    tx, ty = np.cos(h), np.sin(h)
    return x, y, tx, ty, -ty, tx, np.interp(sv, s, st.z), np.interp(sv, s, st.fl), np.interp(sv, s, st.fr), h


def _offset_of(st, px, py, s_hint, reach=80.):
    """(s, lateral offset + left) of points on stroke st, projected on its stations within reach of s_hint"""
    lo, hi = np.searchsorted(st.s, s_hint - reach), np.searchsorted(st.s, s_hint + reach)
    lo, hi = max(0, lo - 1), min(len(st.s), hi + 1)
    Q = np.column_stack([st.x[lo:hi], st.y[lo:hi]])
    if len(Q) < 2:
        return None, None
    sp, _ = _project(np.column_stack([px, py]), Q)
    sp = sp + st.s[lo]
    x, y, tx, ty, nx, ny, *_ = _frame_at(st, sp)
    return sp, (np.asarray(px) - x) * nx + (np.asarray(py) - y) * ny


def _smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3 - 2 * u)


def _aux_lanes(H, sN, exit_):
    """The OSM-mapped auxiliary lane of an exit (the runs before the split node with 1-2 lanes more than after
    it) or an entry (after it): (s0, s1, lanes, [runs]) or None"""
    runs = sorted(H.lanes, key=lambda r: r[0])

    def at(sv):
        return next((j for j, r in enumerate(runs) if r[0] - 1e-6 <= sv <= r[1] + 1e-6), None)

    def count(r):
        return r[2] + r[3] + r[4]
    jb, ja = at(sN - 1.), at(sN + 1.)
    if jb is None or ja is None or jb == ja:
        return None
    j_aux, j_thru = (jb, ja) if exit_ else (ja, jb)
    n_aux = count(runs[j_aux]) - count(runs[j_thru])
    if n_aux < 1 or n_aux > 2:
        return None
    n, js, j = count(runs[j_aux]), [j_aux], j_aux
    step = -1 if exit_ else 1
    while 0 <= j + step < len(runs) and count(runs[j + step]) == n:
        a, b = (runs[j + step][1], runs[j][0]) if exit_ else (runs[j][1], runs[j + step][0])
        if abs(a - b) > .05:
            break
        j += step
        js.append(j)
    lo, hi = min(runs[q][0] for q in js), max(runs[q][1] for q in js)
    if not (AUX_MIN <= hi - lo <= AUX_MAX):
        return None
    return lo, hi, n_aux, [runs[q] for q in js]


def _drop_aux(H, aux, side, segments, sN, d, kept, done):
    """The auxiliary lane leaves the host's lane runs and its paved extents: both sides go back to the through
    section of the part beyond the split node (the OSM line runs on along the through lanes; the lane centre
    convention of section_of had widened the median side by half a lane too). kept: {(id(H), side): (lo, hi)
    ranges of added lanes already drawn there}"""
    lo, hi, n_aux, runs = aux
    i = int(np.clip(np.searchsorted(H.s, sN + d * 5.), 0, len(H.s) - 1))
    k, rev = H.parts[int(H.part_of[i])]
    eL0, eR0, _ = section_of(segments[k], rev, H.oneway)
    T = TAPER.get(family(H.highway), TAPER['default'])
    for sd, ext, thru in ((side, H.eL if side > 0 else H.eR, eL0 if side > 0 else eR0),
                          (-side, H.eR if side > 0 else H.eL, eR0 if side > 0 else eL0)):
        m = (H.s >= lo - T - 1.) & (H.s <= hi + T + 1.)
        for a, b in kept[(id(H), sd)]:
            m &= ~((H.s >= a) & (H.s <= b))
        ext[m] = np.minimum(ext[m], thru)
    for r in runs:
        if id(r) in done:
            continue
        done.add(id(r))
        r[2] = max(1, r[2] - n_aux)
        tokens = r[6].split('|') if r[6] else []
        if len(tokens) > n_aux:
            r[6] = '|'.join(tokens[:-n_aux] if side < 0 else tokens[n_aux:])


def lane_events(strokes, graph, segments, emit=None):
    """RAA exits and entries where a one-way slip road leaves or joins a motorway or trunk: an added lane
    on the host (taper l_Z, parallel lane l_A, then out to the slip road until the two surfaces separate)
    and the slip road re-anchored on the added lane's centre, blended into its own line over BLEND m."""
    by_seg = {}
    for st in strokes:
        for p, (k, rev) in enumerate(st.parts):
            by_seg[k] = (st, p)
    stats = defaultdict(int)
    done = set()
    kept, runs_done = defaultdict(list), set()
    for (k, e), (host, node) in sorted(graph.branch.items()):
        hk, _ = host
        if k not in by_seg or hk not in by_seg:
            continue
        H, _ = by_seg[hk]
        B, bp = by_seg[k]
        hw = H.highway
        if hw not in EXIT or not H.oneway or not B.oneway or B is H:
            continue
        if B.highway in EXIT:
            stats['connector'] += 1                  # a motorway-class connector: kept as fitted (G1 at the node)
            continue
        kind = 'exit' if graph.flow((k, e)) > 0 else 'entry'
        exit_ = kind == 'exit'
        if exit_ and not (bp == 0 and B.nodes[0][0] == node) or not exit_ and not (bp == len(B.parts) - 1 and B.nodes[-1][0] == node):
            stats['not_at_end'] += 1
            continue
        if (id(B), kind) in done:
            continue
        sN = next((sv for n, sv, _ in H.nodes if n == node), None)
        if sN is None:
            continue
        lA, lZ = (EXIT if exit_ else ENTRY)[hw]
        d = 1. if exit_ else -1.
        LB = B.length()
        # the side: where B is 40 m from the node
        tb = min(40., LB * .5) if exit_ else max(LB - 40., LB * .5)
        _, lat = _offset_of(H, [np.interp(tb, B.s, B.x)], [np.interp(tb, B.s, B.y)], sN + d * 40.)
        if lat is None:
            continue
        side = 1. if lat[0] > 0 else -1.
        row = next((r for r in H.lanes if r[0] - 1e-6 <= sN + d <= r[1] + 1e-6), H.lanes[0])
        lw, nthru = row[5], max(1, row[2])
        lanes_edge = nthru * lw / 2.
        brow = B.lanes[0] if exit_ else B.lanes[-1]
        w = min(2, max(1, int(brow[2]))) * ADDED_LANE
        c = lanes_edge + w / 2.
        aux = _aux_lanes(H, sN, exit_)
        # where B's centre line reaches the added lane's centre, searched from the node along B
        along = B.s if exit_ else LB - B.s
        order = np.arange(len(B.s)) if exit_ else np.arange(len(B.s) - 1, -1, -1)
        cand = order[along[order] <= SEARCH]
        if len(cand) < 3:
            stats['short'] += 1
            continue
        hs, off = _offset_of(H, B.x[cand], B.y[cand], sN + d * SEARCH / 2, reach=SEARCH)
        if hs is None:
            continue
        out = np.flatnonzero(off * side >= c)
        if not len(out):
            stats['never_out'] += 1
            continue
        t_c, s_c = float(B.s[cand[out[0]]]), float(hs[out[0]])
        t_b = t_c + d * BLEND / 2
        if not (5. < t_b < LB - 5.):
            stats['short'] += 1
            continue
        # a fork or junction on B inside the part that is replaced or eased keeps B as it is
        far = B.nodes[-1][0] if exit_ else B.nodes[0][0]
        if any((len(graph.mouths[nd]) > 2 or nd == far) and nd != node and ((sv <= t_b + 31.) if exit_ else (sv >= t_b - 31.))
               for nd, sv, _ in B.nodes):
            stats['fork_near'] += 1
            continue
        s_sep = s_c - d * BLEND / 2
        if aux is not None:
            # the OSM-mapped auxiliary lane is the parallel lane (AUX_SHIFT, AUX_TAPER: measured, see there)
            s_par = aux[0] + AUX_SHIFT['exit'] if exit_ else aux[1] + AUX_SHIFT['entry']
            if (s_sep - s_par) * d < 10.:
                s_par = s_sep - d * 10.
            lZ = AUX_TAPER
        else:
            s_par = s_sep - d * lA
        s_tap = s_par - d * lZ
        if not (0. <= s_tap <= H.length()):
            stats['clipped'] += 1
            s_tap = float(np.clip(s_tap, 0., H.length()))
            s_par = float(np.clip(s_par, 0., H.length()))
        if not (0. <= s_sep <= H.length()) or abs(s_sep - s_tap) < 10.:
            stats['short'] += 1
            continue
        if aux is not None:
            _drop_aux(H, aux, side, segments, sN, d, kept, runs_done)
            stats['osm_lane'] += 1
        # the added lane's centre along H (increasing s in both cases: exit s_tap -> s_sep, entry s_sep -> s_tap)
        a0, a1 = min(s_tap, s_sep), max(s_tap, s_sep)
        hsamp = np.linspace(a0, a1, max(2, int(math.ceil((a1 - a0) / 2.)) + 1))
        x, y, _, _, nx, ny, _, _, _, hh = _frame_at(H, hsamp)
        lane = np.column_stack([x + nx * c * side, y + ny * c * side])
        # Hermite blend between the lane at s_sep and B at t_b
        x0, y0, _, _, nx0, ny0, _, _, _, h0 = _frame_at(H, s_sep)
        p0 = np.array([x0 + nx0 * c * side, y0 + ny0 * c * side])
        p1 = np.array([np.interp(t_b, B.s, B.x), np.interp(t_b, B.s, B.y)])
        h1 = float(np.interp(t_b, B.s, B.h))
        L = float(np.hypot(*(p1 - p0)))
        if exit_:
            pa, pb, ha, hb = p0, p1, float(h0), h1
        else:
            pa, pb, ha, hb = p1, p0, h1, float(h0)
        u = np.linspace(0, 1, max(3, int(L) + 1))
        m0 = np.array([math.cos(ha), math.sin(ha)]) * L
        m1 = np.array([math.cos(hb), math.sin(hb)]) * L
        bl = ((2 * u ** 3 - 3 * u ** 2 + 1)[:, None] * pa + (u ** 3 - 2 * u ** 2 + u)[:, None] * m0
              + (-2 * u ** 3 + 3 * u ** 2)[:, None] * pb + (u ** 3 - u ** 2)[:, None] * m1)
        if exit_:
            own = B.s > t_b + 1e-6
            P = np.vstack([lane[:-1], bl, np.column_stack([B.x[own], B.y[own]])])
            n_lane, n_own = len(lane) - 1, int(own.sum())
        else:
            own = B.s < t_b - 1e-6
            P = np.vstack([np.column_stack([B.x[own], B.y[own]]), bl, lane[1:]])
            n_lane, n_own = len(lane) - 1, int(own.sum())
        keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > .02]
        if exit_:
            keep[n_lane + len(bl) - 1] = True
        P = P[keep]
        own_h = B.h[own]
        snew = np.r_[0., np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        hnew = np.unwrap(_headings(P))
        # B's own stations keep their headings (the exact node tangents among them)
        oi = np.arange(len(P) - n_own, len(P)) if exit_ else np.arange(0, n_own)
        if len(oi) == len(own_h):
            hnew[oi] = _unwrap_to(own_h, hnew[oi])
        if exit_:
            lane_idx = np.arange(0, n_lane)
            j_end = len(P) - n_own - 1                       # the blend's end (B at t_b)
            hnew[lane_idx] = _unwrap_to(hh[:-1], hnew[lane_idx])
            hnew[n_lane] = _unwrap_to(np.array([ha]), hnew[n_lane])[0]
            hnew[j_end] = _unwrap_to(np.array([hb]), hnew[j_end])[0]
            S_b = snew[j_end]
            old = np.where(np.arange(len(P)) >= j_end, snew - S_b + t_b, t_b)
            g0 = snew[n_lane]
            g1 = S_b + 30.
            wgt = _smoothstep((snew - g0) / max(g1 - g0, 1e-6))
        else:
            j_end = n_own                                    # the blend's start (B at t_b)
            lane_idx = np.arange(len(P) - n_lane, len(P))
            hnew[lane_idx] = _unwrap_to(hh[1:], hnew[lane_idx])
            hnew[j_end] = _unwrap_to(np.array([ha]), hnew[j_end])[0]
            hnew[len(P) - n_lane - 1] = _unwrap_to(np.array([hb]), hnew[len(P) - n_lane - 1])[0]
            S_b = snew[j_end]
            old = np.where(np.arange(len(P)) <= j_end, snew - S_b + t_b, t_b)
            g1 = snew[len(P) - n_lane - 1]
            g0 = S_b - 30.
            wgt = 1. - _smoothstep((snew - g0) / max(g1 - g0, 1e-6))
        zb = np.interp(old, B.s, B.z)
        eLb, eRb = np.interp(old, B.s, B.eL), np.interp(old, B.s, B.eR)
        flb, frb = np.interp(old, B.s, B.fl), np.interp(old, B.s, B.fr)
        hs_new, off_new = _offset_of(H, P[:, 0], P[:, 1], (s_sep + s_tap) / 2, reach=lA + lZ + BLEND + 150.)
        _, _, _, _, _, _, zh, flh, frh, _ = _frame_at(H, hs_new)
        zplane = zh + np.where(off_new > 0, flh, frh) * np.abs(off_new)
        half = w * _smoothstep(np.abs(hs_new - s_tap) / lZ) / 2
        slope_l = np.where(side > 0, flh, -frh)              # H's plane: the slope rising to the left
        # the separated surfaces: B's inner edge beyond H's own paved edge on that side
        base_ext = (H.eL if side > 0 else H.eR).copy()
        new_eL = (1 - wgt) * half + wgt * eLb
        new_eR = (1 - wgt) * half + wgt * eRb
        inner = np.abs(off_new) - (new_eR if side > 0 else new_eL)
        outer = np.abs(off_new) + (new_eL if side > 0 else new_eR)
        # (beyond the separation only: a two-lane slip road's lane centre lies outside the through section's
        # paved edge, so at the taper's start, where it is 0 m wide, it was 'clear' (bundle5))
        clear = (inner >= np.interp(hs_new, H.s, base_ext)) & (wgt > 1e-3) & ((hs_new - s_sep) * d >= 0.)
        idx = np.flatnonzero(clear)
        if not len(idx):
            stats['no_nose'] += 1
            nose_i = j_end
        else:
            nose_i = int(idx[0]) if exit_ else int(idx[-1])
        s_nose = float(hs_new[nose_i])
        # everything of B on the new arc length: unchanged beyond the blend, proportional over the replaced part
        old_len, new_len = LB, float(snew[-1])

        def remap(sv):
            if exit_:
                return float(sv - t_b + S_b) if sv >= t_b else float(S_b * sv / max(t_b, 1e-6))
            return float(sv - t_b + S_b) if sv <= t_b else float(S_b + (sv - t_b) * (new_len - S_b) / max(old_len - t_b, 1e-6))
        B.part_s = np.array([remap(v) for v in B.part_s])
        for att in B.attach:                  # an earlier event at B's other end (a connector), on the new s
            att['lane'] = [round(remap(v), 2) for v in att['lane']]
        B.vertex_s = np.array([remap(v) for v in B.vertex_s])
        B.nodes = [(nd, remap(sv), i) for nd, sv, i in B.nodes]
        B.lanes = [[remap(r[0]), remap(r[1])] + r[2:] for r in B.lanes]
        B.x, B.y, B.s, B.h = P[:, 0], P[:, 1], snew, hnew
        B.z = (1 - wgt) * zplane + wgt * zb
        B.eL, B.eR = new_eL, new_eR
        B.fl = (1 - wgt) * slope_l + wgt * flb
        B.fr = (1 - wgt) * (-slope_l) + wgt * frb
        B.k = _curvature(snew, hnew)
        B.part_of = np.clip(np.searchsorted(B.part_s, snew, side='right') - 1, 0, len(B.parts) - 1)
        # the other end keeps its own cut: a connector that leaves one motorway and joins another gets both (the
        # second event reset the first's start to 0, so the slip road was drawn over the exit lane; bundle5)
        d0_old, d1_old = B.draw
        if exit_:
            B.draw = [float(snew[nose_i]), remap(d1_old) if d1_old < old_len - 1e-6 else new_len]
        else:
            B.draw = [remap(d0_old) if d0_old > 1e-6 else 0., float(snew[nose_i])]
        # H's paved edge on that side: + the added lane (taper, parallel) and out to B's outer edge until the nose
        ext = H.eL if side > 0 else H.eR
        lo, hi = min(s_tap, s_nose), max(s_tap, s_nose)
        m = (H.s >= lo) & (H.s <= hi)
        if m.any():
            sv = H.s[m]
            add = base_ext[m] + w * _smoothstep(np.abs(sv - s_tap) / lZ)
            beyond = (sv - s_sep) * d > 0
            if beyond.any():
                o = np.argsort(hs_new)
                add[beyond] = np.maximum(base_ext[m][beyond] + w, np.interp(sv[beyond], hs_new[o], outer[o]))
            ext[m] = np.maximum(ext[m], add)
            kept[(id(H), side)].append((lo, hi))
        i_lane = (0, n_lane - 1) if exit_ else (len(P) - n_lane, len(P) - 1)
        H.events.append({'kind': kind, 'side': int(side), 'taper': round(float(s_tap), 2), 'parallel': round(float(s_par), 2),
                         'separate': round(float(s_sep), 2), 'nose': round(s_nose, 2), 'node_s': round(float(sN), 2),
                         'width': w, 'branch': B.id, 'branch_draw': [round(v, 2) for v in B.draw], 'rule': 'raa'})
        B.attach.append({'kind': kind, 'host': H.id, 'node': str(node), 'host_s': round(float(sN), 2), 'side': int(side),
                         'lane': [round(float(snew[i_lane[0]]), 2), round(float(snew[i_lane[1]]), 2)],
                         'host_lane': [round(float(min(s_tap, s_sep)), 2), round(float(max(s_tap, s_sep)), 2)], 'offset': round(c, 3)})
        done.add((id(B), kind))
        stats[kind] += 1
    if emit:
        emit(f'roads v3: lane events {dict(stats)}')
    return stats


# =========================================================================================================
# Lane splits
# =========================================================================================================
# A one-way road whose lanes part at a node into two branches (or two branches whose lanes join into one road): on
# the real road the lanes run on side by side, one lane line between them, to the gore nose, and only there part. The
# OSM node lies before it (a review of interchanges: about 40 m), so each branch's line already leaves the
# divider at the node; a section centred on it drifts across the other lane and its lines step (the lane block of the
# through stroke moved over sweep.lua's 40 m window, and the approach narrowed over TAPER before the node). Here each
# branch keeps its lanes where they are in the joint carriageway until its own line reaches its lane's centre there,
# then leaves on a bend tangent to both; only the paved extents and the lane width change, so the centre lines, the
# nav lines and the heights stay the exporter's. sweep.lua places a lane block off its centre line by the clamp that
# keeps it MARKS.inside within the paved edges (fitBlock): the inner extent is set so the clamp puts the block there.
SPLIT_REACH = 150.     # m along a branch: as far as sweep.lua's fork rule (TUNE.forkReach) looks for the nose
SPLIT_BEND = 80.       # m (own choice): from the nose, where its line reaches its lane's centre, a branch's lane bends onto
                       # its own line over this (smoothstep of the line's offset: at its sharpest R ~145 m on R 600 arcs and
                       # ~110 m on R 400, a third of the line's own; 40 m gave R 85 on R 600, 100 m no more than 80), shorter
                       # where the branch's next junction is nearer (at least SPLIT_BEND_MIN). Game test of
                       # a real split: a bend that began 1 m inside the centre (14 m before the nose at R 460) parted the
                       # lanes early and drew the branch's line before the nose
SPLIT_BEND_MIN = 15.
SPLIT_EASE = 20.       # m (own choice): the paved extents then go back to the branch's own section
SPLIT_STEP = .05       # m: the through stroke's section steps from the joint road to its own over this at the node
SPLIT_SPACING = 5.     # m between stations on a branch's held stretch (10 m drew the bend in three chords)
LANE_INSIDE = .25      # sweep.lua MARKS.inside


def _fit_block(n, width, avail_l, avail_r, inside=LANE_INSIDE):
    """(left, right) of a lane block in a paved section, as sweep.lua's fitBlock places it"""
    half = n * width / 2
    left, right = half, -half
    if left > avail_l - inside:
        d = left - (avail_l - inside)
        left, right = left - d, right - d
    if right < -(avail_r - inside):
        d = -(avail_r - inside) - right
        left, right = left + d, right + d
    if left > avail_l - inside:
        left = avail_l - inside
    if left - right < n * 2.5:
        left, right = half, -half
    return left, right


def _add_stations(st, extra):
    """stations added at arc lengths `extra` on the chords between st's own (the line is unchanged)"""
    extra = np.asarray([v for v in extra if 0. <= v <= st.s[-1] and np.min(np.abs(st.s - v)) > .02], float)
    if not len(extra):
        return
    s = np.sort(np.r_[st.s, extra])
    for name in ('x', 'y', 'z', 'h', 'k', 'eL', 'eR', 'fl', 'fr'):
        setattr(st, name, np.interp(s, st.s, getattr(st, name)))
    st.s = s
    st.part_of = np.clip(np.searchsorted(st.part_s, s, side='right') - 1, 0, len(st.parts) - 1)


def _way_end(xy, rev, ends_at_node, reach=150.):
    """(heading, curvature + left) along the travel of an OSM way at its end at a node: the curvature from the heading
    change between its first and last chord within `reach` m of the node over their distance (0 for a straight way), the
    heading the end chord's turned by half its own turn. rev: the stroke runs against the way's node order; ends_at_node:
    the travel arrives at the node along it (else it leaves the node)"""
    P = np.asarray(xy, float)
    if rev:
        P = P[::-1]                                   # travel order
    c = np.r_[0., np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    near = (c[-1] - c <= reach) if ends_at_node else (c <= reach)
    keep = P[near] if near.sum() >= 3 else (P[-3:] if ends_at_node else P[:3])
    end = keep[-1] - keep[-2] if ends_at_node else keep[1] - keep[0]
    h_end = math.atan2(end[1], end[0])
    if len(keep) < 3:
        return h_end, 0.
    a, b = keep[1] - keep[0], keep[-1] - keep[-2]
    dh = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    run = np.r_[0., np.cumsum(np.hypot(*np.diff(keep, axis=0).T))]
    span = (run[-1] - np.hypot(*b) / 2) - np.hypot(*a) / 2
    k = float(dh / span) if span > 1. else 0.
    half = k * float(np.hypot(*end)) / 2
    return h_end + (half if ends_at_node else -half), k


def _count(run):
    return run[2] + run[3] + run[4]


def _run_chain(st, s0, d):
    """the lane runs of st from s0 in direction d (> 0: increasing s) up to the first change of the lane count"""
    d = 1 if d > 0 else -1                            # a list step (lane_splits passes +-1.0: the pack build crashed on it)
    runs = sorted(st.lanes, key=lambda r: r[0])
    j = next((q for q, r in enumerate(runs) if r[0] - 1e-6 <= s0 + d * .5 <= r[1] + 1e-6), None)
    if j is None:
        return []
    out = [runs[j]]
    while 0 <= j + d < len(runs) and _count(runs[j + d]) == _count(runs[j]) and \
            abs((runs[j][1] if d > 0 else runs[j][0]) - (runs[j + d][0] if d > 0 else runs[j + d][1])) < .05:
        j += d
        out.append(runs[j])
    return out


def lane_splits(strokes, graph, segments, emit=None):
    """Lane splits (above): the extents, lane widths and stations of both branches, and a 'split' role for the one that
    starts or ends at the node (sweep.lua: the through stroke's lane block steps there instead of a window)."""
    by_seg = {}
    for st in strokes:
        for p, (k, rev) in enumerate(st.parts):
            by_seg[k] = (st, p)
    stats = defaultdict(int)
    for node, ms in graph.mouths.items():
        kind = graph.kind.get(node)
        if len(ms) != 3 or kind not in ('diverge', 'merge'):
            continue
        rest = [(k, e) for k, e, _, _ in ms if (k, e) in graph.branch]
        if len(rest) != 1 or rest[0][0] not in by_seg or graph.branch[rest[0]][0][0] not in by_seg:
            continue
        B, bp = by_seg[rest[0][0]]
        H, _ = by_seg[graph.branch[rest[0]][0][0]]
        out = kind == 'diverge'
        if B is H or not (H.oneway and B.oneway) or H.closed or B.closed:
            continue
        if out and not (bp == 0 and B.nodes[0][0] == node) or not out and not (bp == len(B.parts) - 1 and B.nodes[-1][0] == node):
            continue
        here = [sv for n, sv, _ in H.nodes if n == node]
        if len(here) != 1 or any(a['node'] == str(node) for a in B.attach) or any(e['branch'] == B.id for e in H.events):
            continue
        sN = here[0]
        d = 1. if out else -1.                       # along H, away from the node into the branch's side
        if any(min(e['taper'], e['nose']) - SPLIT_REACH <= sN <= max(e['taper'], e['nose']) + SPLIT_REACH for e in H.events):
            stats['event_near'] += 1
            continue
        joint = _run_chain(H, sN, -d)
        own_h, own_b = _run_chain(H, sN, d), _run_chain(B, 0. if out else B.length(), d)
        if not (joint and own_h and own_b):
            continue
        nA, nH, nB = _count(joint[0]), _count(own_h[0]), _count(own_b[0])
        if nA != nH + nB:
            continue
        w = joint[0][5]
        # the joint road's section at the node and its lane block (as sweep.lua fits it)
        iN = int(np.argmin(np.abs(H.s - sN)))
        bN = int(np.argmin(np.abs(np.asarray(H.part_s) - sN)))
        if abs(H.s[iN] - sN) > 1e-3 or abs(H.part_s[bN] - sN) > .01 or not 0 < bN < len(H.parts):
            continue
        pj = bN - 1 if out else bN                   # the joint road's part (part_of at the node station may be either)
        kj, rj = H.parts[pj]
        eLa, eRa, _ = section_of(segments[kj], rj, H.oneway)
        left_a, right_a = _fit_block(nA, w, eLa, eRa)
        # the reference: the joint road's centre line carried on past the node at the heading and curvature of its own OSM way
        # there (the stroke's node tangent halves the turn to the through branch's first chord, and the fit bends towards it
        # over ~60 m: with OSM vertices 20 m apart the held divider rose 0.5 m in 40 m on a straight approach)
        x0, y0 = float(H.x[iN]), float(H.y[iN])
        h0, k0 = _way_end(segments[kj]['xy'], rj, out)
        h0 = float(_unwrap_to(np.array([h0]), np.array([H.h[iN]]))[0])

        def ref(px, py):
            """(offset + left, heading) of points against the reference"""
            dx, dy = px - x0, py - y0
            a, b = dx * math.cos(h0) + dy * math.sin(h0), -dx * math.sin(h0) + dy * math.cos(h0)
            if abs(k0) < 1e-5:
                return b, np.full(np.shape(a), h0)
            r = 1. / k0
            vx, vy = a / r, (b - r) / r
            tau = np.arctan2(vx, -vy) / k0
            return np.sign(k0) * (abs(r) - np.hypot(a, b - r)), h0 + k0 * tau
        # which side B leaves on: the two lines 40 m (or half the shorter branch) from the node
        reach = []
        for X, sx in ((H, sN), (B, 0. if out else B.length())):
            stops = [sv for n, sv, _ in X.nodes if (sv - sx) * d > 1e-3 and len(graph.mouths[n]) > 2]
            stops.append(X.length() if d > 0 else 0.)
            if X is B:
                stops.append(B.draw[1] if d > 0 else B.draw[0])
            chain = own_h if X is H else own_b
            stops.append(chain[-1][1] if d > 0 else chain[-1][0])
            reach.append(min(SPLIT_REACH, min(abs(v - sx) for v in stops)))
        t_side = min(40., .5 * min(reach))
        oH, _ = ref(np.interp(sN + d * t_side, H.s, H.x), np.interp(sN + d * t_side, H.s, H.y))
        sb = t_side if out else B.length() - t_side
        oB, _ = ref(np.interp(sb, B.s, B.x), np.interp(sb, B.s, B.y))
        if abs(oB - oH) < .5:
            stats['side_unclear'] += 1
            continue
        sideB = 1. if oB > oH else -1.
        WB, WH = nB * w, nH * w
        if sideB < 0:
            div = right_a + WB
            lanes = {id(B): (div - WB / 2, -1., eRa + right_a), id(H): (div + WH / 2, 1., eLa - left_a)}
        else:
            div = left_a - WB
            lanes = {id(B): (div + WB / 2, 1., eLa - left_a), id(H): (div - WH / 2, -1., eRa + right_a)}
        T_ret = TAPER.get(family(H.highway), TAPER['default'])
        if min(reach) < SPLIT_EASE + T_ret + 10.:
            stats['short'] += 1
            continue
        # each branch planned first, then changed: a branch whose line runs the wrong way leaves both as they are
        plans, wrong = [], False
        for X, sx, L, W in ((H, sN, reach[0], WH), (B, 0. if out else B.length(), reach[1], WB)):
            _add_stations(X, [sx + d * SPLIT_STEP] + [sx + d * t for t in np.arange(SPLIT_SPACING, L, SPLIT_SPACING)])
            along = (X.s - sx) * d
            # H's station at the node keeps the joint road's section; B's is its first row (cut to nothing by sweep.lua)
            idx = np.flatnonzero((along > (1e-6 if X is H else -1e-6)) & (along <= L + 1e-6))
            idx = idx[np.argsort(along[idx])]
            t = along[idx]
            c, sg, sO = lanes[id(X)]
            o, hr = ref(X.x[idx], X.y[idx])
            cs = np.maximum(np.cos(X.h[idx] - hr), .5)
            dd = sg * (o - c)
            uq = sg * np.maximum(0., -dd / cs - W / 2 * (1. / cs - 1.))   # the block's centre off the line, held in place
            gone = np.flatnonzero((dd >= 0.) & (t <= L - SPLIT_EASE - SPLIT_BEND_MIN))
            t_hold = float(t[gone[0]]) if len(gone) else L - SPLIT_EASE
            # held, the line must come towards its lane's centre: one that goes more than 1 m further from it than at the node
            # (a branch mapped straight on beside a curved approach, the lines crossing) is no lane split (a random network:
            # a lane held 18 m off its line, its extent negative)
            if np.any(dd[t <= t_hold] < -(abs(c) + 1.)):
                wrong = True
                break
            plans.append((X, idx, t, hr, dd, uq, gone, L, W, sg, sO))
        if wrong:
            stats['wrong_way'] += 1
            continue
        back_n = 0
        for X, idx, t, hr, dd, uq, gone, L, W, sg, sO in plans:
            if len(gone):
                # the nose: the line is at its lane's centre (the block on it); from there the stations move onto the bend, a
                # smoothstep from the joint lane to the line over SPLIT_BEND, and the block stays on them
                j = gone[0]
                t_nose = float(t[j])
                uq[j:] = 0.
                L_b = min(SPLIT_BEND, L - SPLIT_EASE - t_nose)
                bend = (t >= t_nose) & (t <= t_nose + L_b)
                shift = -sg * dd * (1. - _smoothstep((t - t_nose) / L_b))      # lane centre - line, across the reference
                k = idx[bend]
                X.x[k] = X.x[k] - np.sin(hr[bend]) * shift[bend]
                X.y[k] = X.y[k] + np.cos(hr[bend]) * shift[bend]
                seg_ = np.arange(max(0, k.min() - 1), min(len(X.s) - 1, k.max() + 1) + 1)
                X.h[seg_] = _unwrap_to(np.arctan2(np.gradient(X.y[seg_], X.s[seg_]), np.gradient(X.x[seg_], X.s[seg_])), X.h[seg_])
                t_dep = t_nose + L_b
            else:
                # the line never reaches its lane's centre (a straight branch mapped on the joint road's centre line): the lane
                # comes back to its line over TAPER before the extents go back
                t_dep = L - SPLIT_EASE
                uq = uq * (1. - _smoothstep((t - (t_dep - T_ret)) / T_ret))
                back_n += 1
            # the paved extents that make sweep.lua's fitBlock put the block there: the inner one MARKS.inside beyond the
            # block (under the other branch, which is cut along it; at least 0 where the line starts a hair past the divider),
            # the outer one with the joint road's shoulder; then the own
            inner = np.maximum(0., W / 2 - sg * uq + LANE_INSIDE)
            outer = sg * uq + W / 2 + sO
            eL_new, eR_new = (outer, inner) if sg > 0 else (inner, outer)
            back = _smoothstep((t - t_dep) / SPLIT_EASE)
            X.eL[idx] = eL_new + (X.eL[idx] - eL_new) * back
            X.eR[idx] = eR_new + (X.eR[idx] - eR_new) * back
            # the joint road's lane width up to the branch's next change of lane count (a one-lane link is 4.0 m between its
            # shoulders: its own lane block was 25 cm wider than the lane it continues)
            for r in (own_h if X is H else own_b):
                r[5] = w
        # the joint road keeps its section to the node (build_strokes tapered its far side over TAPER before it, over at most
        # the half of the part next to the node)
        reach_j = min(T_ret, abs(H.part_s[pj + 1] - H.part_s[pj]) / 2)
        jm = ((H.s - sN) * d <= 1e-6) & (np.abs(H.s - sN) <= reach_j + 1e-6)
        H.eL[jm], H.eR[jm] = eLa, eRa
        if not out:
            # the row at a run boundary takes the earlier run (sweep.lua lanesAt): before a merge's node that is the branch's,
            # so the boundary moves to the step's upstream end
            for r in H.lanes:
                if abs(r[1] - sN) < 1e-3:
                    r[1] = sN - SPLIT_STEP
                elif abs(r[0] - sN) < 1e-3:
                    r[0] = sN - SPLIT_STEP
        B.split.append({'node': str(node), 'host': H.id})
        stats[kind] += 1
        if back_n:
            stats['brought_back'] += back_n
    if emit:
        emit(f'roads v3: lane splits {dict(stats)}')
    return stats


# =========================================================================================================
# Paired carriageways, furniture, structures
# =========================================================================================================

def _near_pairs(ax, ay, bx, by, cell):
    """candidate pairs (i, j) of points a and b in the same or a neighbouring grid cell"""
    M = 1 << 20

    def key(x, y, dx=0, dy=0):
        return (np.floor(x / cell).astype(np.int64) + dx + M) * (2 * M) + np.floor(y / cell).astype(np.int64) + dy + M
    kb = key(bx, by)
    order = np.argsort(kb, kind='stable')
    sk = kb[order]
    IA, IB = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ka = key(ax, ay, dx, dy)
            lo, hi = np.searchsorted(sk, ka, 'left'), np.searchsorted(sk, ka, 'right')
            cnt = hi - lo
            tot = int(cnt.sum())
            if not tot:
                continue
            IA.append(np.repeat(np.arange(len(ax)), cnt))
            IB.append(order[np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))])
    if not IA:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(IA), np.concatenate(IB)


def _all_stations(strokes, sel=None):
    sel = range(len(strokes)) if sel is None else sel
    cols = {k: [] for k in ('x', 'y', 'h', 's', 'eL', 'eR', 'id', 'i')}
    for si in sel:
        st = strokes[si]
        n = len(st.s)
        for k in ('x', 'y', 'h', 's', 'eL', 'eR'):
            cols[k].append(getattr(st, k))
        cols['id'].append(np.full(n, si))
        cols['i'].append(np.arange(n))
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}


def clamp_pairs(strokes, emit=None):
    """Two opposite one-way carriageways of one class whose paved extents overlap in the median: each keeps
    its share of the gap (ROAD-OBJECTS.md 4.1: 3.2 % of the test site's motorway overlapped, up to 0.2 m)."""
    sel = [i for i, st in enumerate(strokes) if st.oneway and st.highway in ('motorway', 'trunk')]
    if len(sel) < 2:
        return 0
    A = _all_stations(strokes, sel)
    ia, ib = _near_pairs(A['x'], A['y'], A['x'], A['y'], 30.)
    keep = A['id'][ia] != A['id'][ib]
    ia, ib = ia[keep], ib[keep]
    tx, ty = np.cos(A['h'][ia]), np.sin(A['h'][ia])
    opp = tx * np.cos(A['h'][ib]) + ty * np.sin(A['h'][ib]) < -.9
    dx, dy = A['x'][ib] - A['x'][ia], A['y'][ib] - A['y'][ia]
    along, lat = dx * tx + dy * ty, -dx * ty + dy * tx
    ok = opp & (np.abs(along) <= 6.) & (lat > 0) & (lat < 45.)
    ia, ib, lat = ia[ok], ib[ok], lat[ok]
    if not len(ia):
        return 0
    # the nearest opposite station for each station
    o = np.lexsort((lat, ia))
    ia, ib, lat = ia[o], ib[o], lat[o]
    first = np.r_[True, ia[1:] != ia[:-1]]
    ia, ib, lat = ia[first], ib[first], lat[first]
    need = A['eL'][ia] + A['eL'][ib]
    over = need > lat
    clamped = 0
    for a, b_, l, nd in zip(ia[over].tolist(), ib[over].tolist(), lat[over].tolist(), need[over].tolist()):
        st = strokes[int(A['id'][a])]
        i = int(A['i'][a])
        st.eL[i] = max(0., l * A['eL'][a] / nd)
        clamped += 1
    if emit:
        emit(f'roads v3: {clamped} median stations clamped against the opposite carriageway')
    return clamped


# Rule-based median rail (an exception to "rails only where OSM maps them"): where two opposite one-way
# carriageways of motorway / trunk class (links too) run parallel with less than MEDIAN_GAP m between their paved edges and no
# barrier is mapped on either side, one guard rail runs in the middle of the gap. German medians of that width carry one
# (RPS: Schutzeinrichtung im Mittelstreifen, memory, not verified). Not on bridges and in tunnels (parapet / wall), and not
# where the edges are closer than MEDIAN_MIN (overlapping lanes, gores). The runs carry way = 'rule:median', not an OSM way id.
MEDIAN_GAP = 6.0
MEDIAN_MIN = 1.0
MEDIAN_RUN = 20.0
MEDIAN_CLASSES = ('motorway', 'trunk', 'motorway_link', 'trunk_link')


def median_rails(strokes, emit=None):
    from bisect import bisect_right
    sel = [i for i, st in enumerate(strokes) if st.oneway and st.highway in MEDIAN_CLASSES]
    if len(sel) < 2:
        return 0
    A = _all_stations(strokes, sel)
    ia, ib = _near_pairs(A['x'], A['y'], A['x'], A['y'], 16.)
    keep = A['id'][ia] != A['id'][ib]
    ia, ib = ia[keep], ib[keep]
    tx, ty = np.cos(A['h'][ia]), np.sin(A['h'][ia])
    opp = tx * np.cos(A['h'][ib]) + ty * np.sin(A['h'][ib]) < -.9
    dx, dy = A['x'][ib] - A['x'][ia], A['y'][ib] - A['y'][ia]
    along, lat = dx * tx + dy * ty, -dx * ty + dy * tx
    # the facing extent of the other carriageway: its left where it lies on the left (opposite headings), else its right
    gap = np.abs(lat) - np.where(lat > 0, A['eL'][ia], A['eR'][ia]) - np.where(lat > 0, A['eL'][ib], A['eR'][ib])
    ok = opp & (np.abs(along) <= 4.) & (gap >= MEDIAN_MIN) & (gap < MEDIAN_GAP)
    ia, ib, gap, lat = ia[ok], ib[ok], gap[ok], lat[ok]
    if not len(ia):
        return 0
    side = np.where(lat > 0, 1, -1)
    o = np.lexsort((gap, side, ia))
    ia, ib, gap, side = ia[o], ib[o], gap[o], side[o]
    first = np.r_[True, (ia[1:] != ia[:-1]) | (side[1:] != side[:-1])]
    ia, ib, gap, side = ia[first], ib[first], gap[first], side[first]
    # one carriageway of the pair carries the rail: the one with the lower stroke index
    mine = A['id'][ia] < A['id'][ib]
    ia, ib, gap, side = ia[mine], ib[mine], gap[mine], side[mine]
    mapped = defaultdict(list)                       # (stroke, side) -> sorted [(s0, s1)] of mapped barrier runs
    for si, st in enumerate(strokes):
        for f in st.furniture:
            mapped[(si, f['side'])].append((f['s0'] - 3., f['s1'] + 3.))
    for v in mapped.values():
        v.sort()

    def covered(si, sd, sv):
        v = mapped.get((si, sd))
        if not v:
            return False
        k = bisect_right(v, (sv, 1e18)) - 1
        return any(v[q][0] <= sv <= v[q][1] for q in range(max(0, k - 3), k + 1))

    def structured(st, sv):
        return any(r['kind'] in ('bridge', 'tunnel') and r['s0'] - 1. <= sv <= r['s1'] + 1. for r in st.structures)
    cand = defaultdict(list)                         # (stroke, side) -> [(s, gap)]
    for a, b, g, sd in zip(ia.tolist(), ib.tolist(), gap.tolist(), side.tolist()):
        si, sj = int(A['id'][a]), int(A['id'][b])
        sa, sb = float(A['s'][a]), float(A['s'][b])
        if covered(si, sd, sa) or covered(sj, sd, sb) or structured(strokes[si], sa) or structured(strokes[sj], sb):
            continue
        cand[(si, sd)].append((sa, g))
    runs = 0
    for (si, sd), v in cand.items():
        v.sort()
        cur = None
        for sv, g in v + [(None, None)]:
            if cur and sv is not None and sv - cur[1] <= 12.:
                cur[1] = sv
                cur[2].append(g)
                continue
            if cur and cur[1] - cur[0] >= MEDIAN_RUN:
                strokes[si].furniture.append({'kind': 'guard_rail', 'side': int(sd), 's0': round(cur[0], 2), 's1': round(cur[1], 2),
                                              'offset': round(float(np.median(cur[2])) / 2, 2), 'way': 'rule:median'})
                runs += 1
            cur = [sv, sv, [g]] if sv is not None else None
    for st in strokes:
        st.furniture.sort(key=lambda f: (f['s0'], f['side']))
    if emit:
        emit(f'roads v3: {runs} median rail runs by rule (carriageways {MEDIAN_MIN:g}-{MEDIAN_GAP:g} m apart, no mapped barrier)')
    return runs


def furniture(strokes, segments, barriers, emit=None):
    """OSM-mapped barriers as runs along strokes: a barrier way between 1 m inside and 4 m outside a paved
    edge and nearly parallel to it is a run on that side (a median rail on both carriageways); barrier tags
    on a road way itself (barrier:left / :right, guard_rail:left / :right, barrier, crash_barrier) a run
    along its part."""
    runs = 0
    if barriers and strokes:
        A = _all_stations(strokes)
        bx, by, bdx, bdy, bw = [], [], [], [], []
        for n_, (wid, rec, xy) in enumerate(barriers):
            cum = np.r_[0., np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            if cum[-1] < 2.:
                continue
            t = np.arange(0., cum[-1] + 1e-6, 2.)
            px, py = np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])
            gx, gy = np.gradient(px), np.gradient(py)
            g = np.maximum(np.hypot(gx, gy), 1e-9)
            bx.append(px); by.append(py); bdx.append(gx / g); bdy.append(gy / g); bw.append(np.full(len(t), n_))
        if bx:
            bx, by, bdx, bdy, bw = (np.concatenate(v) for v in (bx, by, bdx, bdy, bw))
            ib, ia = _near_pairs(bx, by, A['x'], A['y'], 20.)
            tx, ty = np.cos(A['h'][ia]), np.sin(A['h'][ia])
            par = np.abs(bdx[ib] * tx + bdy[ib] * ty) > .8
            dx, dy = bx[ib] - A['x'][ia], by[ib] - A['y'][ia]
            along, lat = dx * tx + dy * ty, -dx * ty + dy * tx
            edge = np.where(lat > 0, A['eL'][ia], A['eR'][ia])
            beyond = np.abs(lat) - edge
            ok = par & (np.abs(along) <= 6.) & (beyond >= BARRIER_REACH[0]) & (beyond <= BARRIER_REACH[1])
            ib, ia, along, lat, beyond = ib[ok], ia[ok], along[ok], lat[ok], beyond[ok]
            sid = A['id'][ia].astype(int)
            side = np.where(lat > 0, 1, -1)
            # one hit per barrier point and stroke side: the nearest station along
            o = np.lexsort((np.abs(along), side, sid, ib))
            ib, sid, side, sv, beyond = ib[o], sid[o], side[o], (A['s'][ia] + along)[o], beyond[o]
            first = np.r_[True, (ib[1:] != ib[:-1]) | (sid[1:] != sid[:-1]) | (side[1:] != side[:-1])]
            ib, sid, side, sv, beyond = ib[first], sid[first], side[first], sv[first], beyond[first]
            way = bw[ib]
            o = np.lexsort((sv, side, sid, way))
            way, sid, side, sv, beyond = way[o], sid[o], side[o], sv[o], beyond[o]
            cur = None
            for w_, si, sd, s_, b_ in zip(way.tolist(), sid.tolist(), side.tolist(), sv.tolist(), beyond.tolist()):
                if cur and cur[0] == (w_, si, sd) and s_ - cur[2] <= BARRIER_GAP:
                    cur[2] = s_
                    cur[3].append(b_)
                    continue
                if cur:
                    runs += _add_run(strokes, barriers, cur)
                cur = [(w_, si, sd), s_, s_, [b_]]
            if cur:
                runs += _add_run(strokes, barriers, cur)
    for st in strokes:
        for p, (k, rev) in enumerate(st.parts):
            tags = segments[k]['tags']
            s0, s1 = float(st.part_s[p]), float(st.part_s[p + 1])
            for key, side in (('barrier:left', 1), ('barrier:right', -1), ('guard_rail:left', 1), ('guard_rail:right', -1)):
                v = tags.get(key)
                if v in (None, 'no'):
                    continue
                st.furniture.append({'kind': BARRIER_KIND.get(v, 'guard_rail'), 'side': -side if rev else side,
                                     's0': round(s0, 2), 's1': round(s1, 2), 'offset': .5, 'way': str(segments[k].get('way')), 'on_way': True})
                runs += 1
            kind = barrier_kind({'barrier': tags.get('barrier'), 'crash_barrier': tags.get('crash_barrier')})
            if kind:
                for sd in ((-1,) if st.oneway else (1, -1)):
                    st.furniture.append({'kind': kind, 'side': sd, 's0': round(s0, 2), 's1': round(s1, 2), 'offset': .5,
                                         'way': str(segments[k].get('way')), 'on_way': True})
                    runs += 1
    for st in strokes:
        st.furniture.sort(key=lambda f: (f['s0'], f['side']))
    if emit:
        emit(f'roads v3: {runs} furniture runs from {len(barriers)} OSM barrier ways')
    return runs


def _add_run(strokes, barriers, cur):
    (w_, si, sd), s0, s1, b = cur
    if s1 - s0 < 2.:
        return 0
    wid, rec, _ = barriers[w_]
    st = strokes[si]
    st.furniture.append({'kind': rec[0], 'side': int(sd), 's0': round(max(0., s0), 2), 's1': round(min(st.length(), s1), 2),
                         'offset': round(float(np.median(b)), 2), 'way': str(wid)})
    return 1


def structures(strokes, segments, floors_report):
    """bridges, tunnels, embankments and cuttings as runs; a deck over navigable water carries its rule"""
    from fast_profile import kind as segment_kind, layer
    by_seg = defaultdict(list)
    for r in floors_report:
        by_seg[r['segment']].append(r)
    for st in strokes:
        for p, (k, rev) in enumerate(st.parts):
            tags = segments[k]['tags']
            s0, s1 = float(st.part_s[p]), float(st.part_s[p + 1])
            sk = segment_kind(tags)
            rows = []
            if sk != 'road':
                row = {'kind': sk, 's0': s0, 's1': s1, 'layer': int(layer(tags)) if sk == 'bridge' else -int(layer(tags)),
                       'structure': tags.get('bridge:structure') or tags.get('tunnel', '')}
                for r in by_seg.get(k, ()):
                    row.update(rule=r['rule'], clearance=r['clearance'], water=round(r['water'], 2), floor=round(r['floor'], 2))
                rows.append(row)
            for key in ('embankment', 'cutting'):
                if tags.get(key) not in (None, 'no'):
                    rows.append({'kind': key, 's0': s0, 's1': s1, 'layer': 0, 'structure': ''})
            for row in rows:
                last = st.structures[-1] if st.structures else None
                if last and last['kind'] == row['kind'] and abs(last['s1'] - row['s0']) < 1e-6 and \
                        {k_: v for k_, v in last.items() if k_ not in ('s0', 's1')} == {k_: v for k_, v in row.items() if k_ not in ('s0', 's1')}:
                    last['s1'] = row['s1']
                else:
                    st.structures.append(row)


# =========================================================================================================
# Chunks and node records (what the index stores and the pack carries)
# =========================================================================================================

def chunks_of(st):
    """[(i0, i1)] station ranges of ~CHUNK m each (shared end stations), from the stroke alone"""
    L = st.length()
    n = max(1, int(math.ceil(L / CHUNK - 1e-9)))
    cuts = [0]
    for c in range(1, n):
        target = L * c / n
        i = int(np.searchsorted(st.s, target))
        i = min(max(i, cuts[-1] + 1), len(st.s) - 1)
        if st.s[i] - target > target - st.s[i - 1] and i - 1 > cuts[-1]:
            i -= 1
        if i >= len(st.s) - 1:
            break
        cuts.append(i)
    cuts.append(len(st.s) - 1)
    return list(zip(cuts[:-1], cuts[1:]))


# --- fan junction plates (phase 2 decision 2: the exporter emits them, so every tile gets one answer) ----
# The generator's rule (fast.junction_star): every mouth is cut back PLATE_CUT x the widest paved road at the
# node, at most PLATE_SHARE of the road to its next node; the plate's ring is each mouth's section there
# (left edge, centre, right edge at their banked heights), fanned from the node. Nodes of kind end and joint
# get none, and neither do mouths of a lane-attached slip road (their taper is the host's widened surface).
PLATE_CUT = .8
PLATE_SHARE = .25
PLATE_REACH = .45             # a mouth may be cut further, to clear the other roads, up to this share of its road
PLATE_CLEAR = .5              # m past the other road's edge
PLATE_STRAIGHT = math.cos(math.radians(30))   # mouths within 30 degrees of straight on or of each other: no clearing
PLATE_KINDS = {'junction', 'cross', 'fork', 'diverge', 'merge'}


def _section_at(st, sv):
    """(left, centre, right) points of a stroke's paved section at s (left of the stroke's direction)"""
    def at(a):
        return float(np.interp(sv, st.s, a))
    x, y, z, h = at(st.x), at(st.y), at(st.z), at(st.h)
    eL, eR, fl, fr = at(st.eL), at(st.eR), at(st.fl), at(st.fr)
    nx, ny = -math.sin(h), math.cos(h)
    return ((x + nx * eL, y + ny * eL, z + fl * eL), (x, y, z), (x - nx * eR, y - ny * eR, z + fr * eR))


def fan_plate(node, kind, mouths, roles):
    """[(stroke, dir, cut, left, centre, right)] of a node's fan plate, or [] (mouths: [(stroke, s, dir)])"""
    if kind not in PLATE_KINDS:
        return []
    ms = [(st, sv, d) for st, sv, d in mouths if st.id not in roles]
    if len(ms) < 2 or len({st.id for st, _, _ in ms}) < 2:
        return []
    width = max(float(np.interp(sv, st.s, st.eL) + np.interp(sv, st.s, st.eR)) for st, sv, _ in ms)
    radius = PLATE_CUT * width
    # each mouth's direction away from the node and its half width there
    dirs = []
    for st, sv, d in ms:
        h = float(np.interp(sv, st.s, st.h))
        dirs.append((math.cos(h) * d, math.sin(h) * d,
                     max(float(np.interp(sv, st.s, st.eL)), float(np.interp(sv, st.s, st.eR)))))
    out = []
    for k, (st, sv, d) in enumerate(ms):
        others = [v for _, v, _ in st.nodes if (v - sv) * d > 1e-6]
        reach = (min(others, key=lambda v: abs(v - sv)) - sv) * d if others else (st.length() - sv if d > 0 else sv)
        cut = max(0., min(radius, PLATE_SHARE * reach))
        # a mouth must clear the other roads' surfaces, or its corners lie on them and the fan folds
        # (a side road cut 1.7 m from a 4 m wide main road's centre line): half the other's width over
        # the sine of the angle between them, up to PLATE_REACH of the way to its next node
        need = 0.
        for q, (dx, dy, half) in enumerate(dirs):
            if q != k:
                c = dirs[k][0] * dx + dirs[k][1] * dy
                if -PLATE_STRAIGHT < c < PLATE_STRAIGHT:
                    need = max(need, half / math.sqrt(max(1e-9, 1 - c * c)) + PLATE_CLEAR)
        cut = max(cut, min(need, PLATE_REACH * reach))
        left, centre, right = _section_at(st, sv + d * cut)
        out.append((st.id, d, round(cut, 2), [round(v, 3) for v in left], [round(v, 3) for v in centre],
                    [round(v, 3) for v in right]))
    return out


def records(strokes, graph, segments, cell_size):
    """(chunk rows [(stroke id, index, ci, cj, data)], node rows [(node, ci, cj, data)])"""
    rows = []
    by_node = defaultdict(list)
    for st in strokes:
        ranges = chunks_of(st)
        meta = {'stroke': st.id, 'highway': st.highway, 'oneway': st.oneway, 'closed': st.closed, 'fitted': bool(st.fitted),
                'length': round(st.length(), 2), 'chunks': len(ranges), 'draw': [round(v, 2) for v in st.draw]}
        parts = [{'way': str(segments[k].get('way')), 'segment': int(segments[k]['id']), 'reversed': bool(rev),
                  's0': round(float(st.part_s[p]), 2), 's1': round(float(st.part_s[p + 1]), 2), 'tags': segments[k]['tags']}
                 for p, (k, rev) in enumerate(st.parts)]
        for ci, (i0, i1) in enumerate(ranges):
            s0, s1 = float(st.s[i0]), float(st.s[i1])
            sl = slice(i0, i1 + 1)
            mid = (i0 + i1) // 2
            cx, cy = int(math.floor(st.x[mid] / cell_size)), int(math.floor(st.y[mid] / cell_size))

            def within(a, b):
                return a <= s1 + 1e-6 and b >= s0 - 1e-6
            data = dict(meta, index=ci, s0=round(s0, 2), s1=round(s1, 2),
                        stations={'s': np.round(st.s[sl], 3).tolist(), 'x': np.round(st.x[sl], 3).tolist(),
                                  'y': np.round(st.y[sl], 3).tolist(), 'z': np.round(st.z[sl], 3).tolist(),
                                  'h': np.round(st.h[sl], 6).tolist(), 'k': np.round(st.k[sl], 7).tolist(),
                                  'eL': np.round(st.eL[sl], 3).tolist(), 'eR': np.round(st.eR[sl], 3).tolist(),
                                  'fl': np.round(st.fl[sl], 5).tolist(), 'fr': np.round(st.fr[sl], 5).tolist()},
                        parts=[p for p in parts if within(p['s0'], p['s1'])],
                        lanes=[r for r in st.lanes if within(r[0], r[1])],
                        events=[e for e in st.events if within(min(e['taper'], e['nose']), max(e['taper'], e['nose']))],
                        furniture=[f for f in st.furniture if within(f['s0'], f['s1'])],
                        structures=[r for r in st.structures if within(r['s0'], r['s1'])],
                        attach=st.attach,
                        nodes=[[str(n), round(sv, 2)] for n, sv, _ in st.nodes if s0 - 1e-6 <= sv <= s1 + 1e-6])
            rows.append((st.id, ci, cx, cy, data))
        # mouths of the stroke at nodes (for the node records)
        for n, sv, _ in st.nodes:
            by_node[n].append((st, sv))
    node_rows = []
    for node, lst in by_node.items():
        kind = graph.kind.get(node, 'end')
        if len(graph.mouths[node]) < 2 or kind == 'joint' and len(lst) == 1:
            continue
        mouths = []
        for st, sv in lst:
            if sv > 1e-6:
                mouths.append([st.id, round(sv, 2), -1])
            if sv < st.length() - 1e-6:
                mouths.append([st.id, round(sv, 2), 1])
        # roles from the graph: branch hosts and lane attachments
        roles = {}
        for st, sv in lst:
            for a in st.attach:
                if a['node'] == str(node):        # attach stores the node id as text, the graph as given
                    roles[st.id] = ['lane', a['host']]
            for a in st.split:
                if a['node'] == str(node):        # lane_splits: the branch takes its lanes from the host at the node
                    roles[st.id] = ['split', a['host']]
        st0, sv0 = lst[0]
        x = float(np.interp(sv0, st0.s, st0.x)); y = float(np.interp(sv0, st0.s, st0.y)); z = float(np.interp(sv0, st0.s, st0.z))
        # a lane-attached slip road no longer passes the node: its position is the host's
        for st, sv in lst:
            if st.id not in roles:
                x = float(np.interp(sv, st.s, st.x)); y = float(np.interp(sv, st.s, st.y)); z = float(np.interp(sv, st.s, st.z))
                break
        plate = fan_plate(node, kind, [(st, sv, d) for st, sv in lst for d in (-1, 1)
                                       if (d < 0 and sv > 1e-6) or (d > 0 and sv < st.length() - 1e-6)], roles)
        node_rows.append((node, int(math.floor(x / cell_size)), int(math.floor(y / cell_size)),
                          {'node': str(node), 'kind': kind, 'xyz': [round(x, 3), round(y, 3), round(z, 3)],
                           'mouths': mouths, 'roles': roles, 'plate': [list(p) for p in plate]}))
    return rows, node_rows


def build(segments, extras, floors_report, cell_size=1000., emit=None):
    """The whole v3 road section of an index: chunk rows, node rows and a summary."""
    t0 = time.perf_counter()
    pinched = lane_pinch(segments)
    graph = Graph(segments)
    strokes = build_strokes(segments, graph, emit)
    if emit:
        emit(f'roads v3: {JOG_COUNT[0]} S-shaped jogs smoothed (zone {JOG_ZONE:g} m), {pinched} one-lane pinches given the lanes before them')
    if os.environ.get('EARTHNG_RULE_LOG'):          # local debugging: where the two rules acted (frame metres)
        import json
        with open(os.environ['EARTHNG_RULE_LOG'], 'w') as f:
            json.dump({'jogs': JOG_LOG, 'pinches': PINCH_LOG}, f)
    ev = lane_events(strokes, graph, segments, emit)
    lane_splits(strokes, graph, segments, emit)
    clamp_pairs(strokes, emit)
    structures(strokes, segments, floors_report)
    furniture(strokes, segments, extras['barriers'] if extras else [], emit)
    median_rails(strokes, emit)
    rows, node_rows = records(strokes, graph, segments, cell_size)
    if emit:
        emit(f'roads v3: {len(rows)} chunks, {len(node_rows)} junction records in {time.perf_counter() - t0:.1f}s')
    # phase 3: one surface per at-grade junction (fast_junctions), stored beside the chunks
    import fast_junctions
    junction_rows = []
    if JUNCTIONS:
        views = {st.id: fast_junctions.view_of_stroke(st, segments) for st in strokes}
        junction_rows = fast_junctions.build([d for _, _, _, d in node_rows], views, extras, cell_size, emit=emit)
    return rows, node_rows, {'strokes': len(strokes), 'events': dict(ev), 'junctions': len(junction_rows),
                             'seconds': time.perf_counter() - t0}, junction_rows


# =========================================================================================================
# Index storage and the pack's binary road section (earthng-roads-v3; spec: ROAD-OBJECTS.md section 7)
# =========================================================================================================

def write_rows(path, rows, node_rows, junction_rows=()):
    """The v3 road rows into an index database (the temporary file write_index made); junction_rows: phase 3
    (fast_junctions.build)."""
    import json
    import sqlite3
    db = sqlite3.connect(path)
    db.executescript('CREATE TABLE roads3(stroke TEXT, idx INTEGER, ci INTEGER, cj INTEGER, data TEXT, PRIMARY KEY(stroke, idx));'
                     'CREATE TABLE roads3_nodes(node TEXT PRIMARY KEY, ci INTEGER, cj INTEGER, data TEXT);')
    db.executemany('INSERT INTO roads3 VALUES(?,?,?,?,?)',
                   ((sid, ci, cx, cy, json.dumps(d, separators=(',', ':'))) for sid, ci, cx, cy, d in rows))
    db.executemany('INSERT INTO roads3_nodes VALUES(?,?,?,?)',
                   ((str(n), cx, cy, json.dumps(d, separators=(',', ':'))) for n, cx, cy, d in node_rows))
    db.execute('CREATE TABLE roads3_junctions(id TEXT PRIMARY KEY, ci INTEGER, cj INTEGER, data TEXT)')
    db.executemany('INSERT INTO roads3_junctions VALUES(?,?,?,?)',
                   ((jid, cx, cy, json.dumps(d, separators=(',', ':'))) for jid, cx, cy, d in junction_rows))
    db.execute('CREATE INDEX roads3_junctions_cell ON roads3_junctions(ci, cj)')
    db.execute('CREATE INDEX roads3_cell ON roads3(ci, cj)')
    db.execute('CREATE INDEX roads3_nodes_cell ON roads3_nodes(ci, cj)')
    db.commit()
    db.close()


def read_cell(db, i, j):
    """(chunks, nodes) of index cell (i, j), in key order"""
    import json
    chunks = [json.loads(d) for (d,) in db.execute('SELECT data FROM roads3 WHERE ci=? AND cj=? ORDER BY stroke, idx', (i, j))]
    nodes = [json.loads(d) for (d,) in db.execute('SELECT data FROM roads3_nodes WHERE ci=? AND cj=? ORDER BY node', (i, j))]
    return chunks, nodes


def read_junctions(db, i, j):
    """the junction records (fast_junctions) of index cell (i, j), in id order; none in an older index"""
    import json
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='roads3_junctions'").fetchone() is None:
        return []
    return [json.loads(d) for (d,) in db.execute('SELECT data FROM roads3_junctions WHERE ci=? AND cj=? ORDER BY id', (i, j))]


def has_rows(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='roads3'").fetchone() is not None


MAGIC = b'EGR3'
SCALE = 100            # cm for lengths and positions
H_SCALE = 1e5          # heading: 1e-5 rad
K_SCALE = 1e6          # curvature: 1e-6 / m
F_SCALE = 1e4          # cross-fall: 0.01 %
STATION_KEYS = ('s', 'x', 'y', 'z', 'h', 'k', 'eL', 'eR', 'fl', 'fr')
STATION_SCALES = (SCALE, SCALE, SCALE, SCALE, H_SCALE, K_SCALE, SCALE, SCALE, F_SCALE, F_SCALE)
EVENT_KEYS = ('taper', 'parallel', 'separate', 'nose', 'node_s')


def _q(value, scale):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite road number')
    return int(round(value * scale))


def encode_cell(chunks, nodes, origin):
    """Bytes of one cell's v3 road payload; origin is the cell's south-west corner (region metres)."""
    from country_format import _Writer
    w = _Writer()
    table = {}

    def add(v):
        table.setdefault(str(v), len(table))
    for c in chunks:
        add(c['stroke']); add(c['highway'])
        for p in c['parts']:
            add(p['way'])
            for k, v in p['tags'].items():
                add(k); add(v)
        for r in c['lanes']:
            add(r[6]); add(r[7])
        for e in c['events']:
            add(e['kind']); add(e['branch'])
        for f in c['furniture']:
            add(f['kind']); add(f['way'])
        for r in c['structures']:
            add(r['kind']); add(r.get('structure', '')); add(r.get('rule', ''))
        for a in c['attach']:
            add(a['kind']); add(a['host']); add(a['node'])
        for n, _ in c['nodes']:
            add(n)
    for n in nodes:
        add(n['node']); add(n['kind'])
        for m in n['mouths']:
            add(m[0])
        for sid, role in n['roles'].items():
            add(sid); add(role[0]); add(role[1])
        for p in n.get('plate', ()):
            add(p[0])
    w.u(len(table))
    for text in table:
        data = text.encode('utf8')
        w.u(len(data))
        w.out += data

    def ref(v):
        w.u(table[str(v)])
    ox, oy = _q(origin[0], SCALE), _q(origin[1], SCALE)
    w.u(len(chunks))
    for c in chunks:
        ref(c['stroke']); w.u(c['index']); w.u(c['chunks'])
        w.u((1 if c['oneway'] else 0) | (2 if c['closed'] else 0) | (4 if c['fitted'] else 0))
        ref(c['highway']); w.u(_q(c['length'], SCALE)); w.s(_q(c['draw'][0], SCALE)); w.s(_q(c['draw'][1], SCALE))
        st = c['stations']
        n = len(st['s'])
        w.u(n)
        prev = [0, ox, oy, 0, 0, 0, 0, 0, 0, 0]
        cols = [[_q(v, sc) for v in st[key]] for key, sc in zip(STATION_KEYS, STATION_SCALES)]
        for i in range(n):
            for f in range(10):
                v = cols[f][i]
                w.s(v - prev[f])
                prev[f] = v
        w.u(len(c['parts']))
        for p in c['parts']:
            ref(p['way']); w.u(p['segment']); w.u(1 if p['reversed'] else 0)
            w.s(_q(p['s0'], SCALE)); w.s(_q(p['s1'], SCALE))
            w.u(len(p['tags']))
            for k in sorted(p['tags']):
                ref(k); ref(p['tags'][k])
        w.u(len(c['lanes']))
        for r in c['lanes']:
            w.s(_q(r[0], SCALE)); w.s(_q(r[1], SCALE)); w.u(r[2]); w.u(r[3]); w.u(r[4]); w.u(_q(r[5], SCALE)); ref(r[6]); ref(r[7])
        w.u(len(c['events']))
        for e in c['events']:
            ref(e['kind']); w.s(e['side'])
            for key in EVENT_KEYS:
                w.s(_q(e[key], SCALE))
            w.u(_q(e['width'], SCALE)); ref(e['branch']); w.s(_q(e['branch_draw'][0], SCALE)); w.s(_q(e['branch_draw'][1], SCALE))
        w.u(len(c['furniture']))
        for f in c['furniture']:
            ref(f['kind']); w.s(f['side']); w.s(_q(f['s0'], SCALE)); w.s(_q(f['s1'], SCALE)); w.s(_q(f['offset'], SCALE))
            ref(f['way']); w.u(1 if f.get('on_way') else 0)
        w.u(len(c['structures']))
        for r in c['structures']:
            ref(r['kind']); w.s(_q(r['s0'], SCALE)); w.s(_q(r['s1'], SCALE)); w.s(int(r['layer'])); ref(r.get('structure', ''))
            ref(r.get('rule', '')); w.s(_q(r.get('clearance', 0.), SCALE)); w.s(_q(r.get('water', 0.), SCALE))
            w.s(_q(r.get('floor', 0.), SCALE))
        w.u(len(c['attach']))
        for a in c['attach']:
            ref(a['kind']); ref(a['host']); ref(a['node']); w.s(_q(a['host_s'], SCALE)); w.s(a['side'])
            for v in list(a['lane']) + list(a['host_lane']):
                w.s(_q(v, SCALE))
            w.s(_q(a['offset'], SCALE))
        w.u(len(c['nodes']))
        for nd, sv in c['nodes']:
            ref(nd); w.s(_q(sv, SCALE))
    w.u(len(nodes))
    for n in nodes:
        ref(n['node']); ref(n['kind'])
        x, y, z = n['xyz']
        w.s(_q(x, SCALE) - ox); w.s(_q(y, SCALE) - oy); w.s(_q(z, SCALE))
        w.u(len(n['mouths']))
        for sid, sv, d in n['mouths']:
            ref(sid); w.s(_q(sv, SCALE)); w.s(int(d))
        w.u(len(n['roles']))
        for sid in sorted(n['roles']):
            role = n['roles'][sid]
            ref(sid); ref(role[0]); ref(role[1])
        # version 2: the fan plate, per mouth (stroke, dir, cut, left, centre, right); x, y from the cell origin
        plate = n.get('plate', ())
        w.u(len(plate))
        for sid, d, cut, *points in plate:
            ref(sid); w.s(int(d)); w.s(_q(cut, SCALE))
            for px, py, pz in points:
                w.s(_q(px, SCALE) - ox); w.s(_q(py, SCALE) - oy); w.s(_q(pz, SCALE))
    return bytes(w.out)


def decode_cell(data, origin, at=0, end=None, version=VERSION):
    """The tables encode_cell wrote (reference decoder for tests and for the game's reader); version: the
    manifest's roads.version (1 has no fan plates)."""
    from country_format import _Reader
    r = _Reader(data, at, end)
    strings = []
    for _ in range(r.u()):
        n = r.u()
        strings.append(bytes(data[r.at:r.at + n]).decode('utf8'))
        r.at += n

    def ref():
        return strings[r.u()]
    ox, oy = _q(origin[0], SCALE), _q(origin[1], SCALE)
    chunks = []
    for _ in range(r.u()):
        c = {'stroke': ref(), 'index': r.u(), 'chunks': r.u()}
        flags = r.u()
        c.update(oneway=bool(flags & 1), closed=bool(flags & 2), fitted=bool(flags & 4), highway=ref(),
                 length=r.u() / SCALE)
        c['draw'] = [r.s() / SCALE, r.s() / SCALE]
        st = {k: [] for k in STATION_KEYS}
        prev = [0, ox, oy, 0, 0, 0, 0, 0, 0, 0]
        for _ in range(r.u()):
            for f, (key, sc) in enumerate(zip(STATION_KEYS, STATION_SCALES)):
                prev[f] += r.s()
                st[key].append(prev[f] / sc)
        c['stations'] = st
        c['parts'] = []
        for _ in range(r.u()):
            p = {'way': ref(), 'segment': r.u(), 'reversed': bool(r.u())}
            p['s0'] = r.s() / SCALE
            p['s1'] = r.s() / SCALE
            p['tags'] = {}
            for _ in range(r.u()):
                k = ref()
                p['tags'][k] = ref()
            c['parts'].append(p)
        c['lanes'] = []
        for _ in range(r.u()):
            row = [r.s() / SCALE, r.s() / SCALE, r.u(), r.u(), r.u(), r.u() / SCALE]
            row += [ref(), ref()]
            c['lanes'].append(row)
        c['events'] = []
        for _ in range(r.u()):
            e = {'kind': ref(), 'side': r.s()}
            for key in EVENT_KEYS:
                e[key] = r.s() / SCALE
            e['width'] = r.u() / SCALE
            e['branch'] = ref()
            e['branch_draw'] = [r.s() / SCALE, r.s() / SCALE]
            c['events'].append(e)
        c['furniture'] = []
        for _ in range(r.u()):
            f = {'kind': ref(), 'side': r.s()}
            f['s0'] = r.s() / SCALE
            f['s1'] = r.s() / SCALE
            f['offset'] = r.s() / SCALE
            f['way'] = ref()
            f['on_way'] = bool(r.u())
            c['furniture'].append(f)
        c['structures'] = []
        for _ in range(r.u()):
            row = {'kind': ref()}
            row['s0'] = r.s() / SCALE
            row['s1'] = r.s() / SCALE
            row['layer'] = r.s()
            row['structure'] = ref()
            row['rule'] = ref()
            row['clearance'] = r.s() / SCALE
            row['water'] = r.s() / SCALE
            row['floor'] = r.s() / SCALE
            c['structures'].append(row)
        c['attach'] = []
        for _ in range(r.u()):
            a = {'kind': ref(), 'host': ref(), 'node': ref()}
            a['host_s'] = r.s() / SCALE
            a['side'] = r.s()
            v = [r.s() / SCALE for _ in range(4)]
            a['lane'], a['host_lane'] = v[:2], v[2:]
            a['offset'] = r.s() / SCALE
            c['attach'].append(a)
        c['nodes'] = []
        for _ in range(r.u()):
            nd = ref()
            c['nodes'].append([nd, r.s() / SCALE])
        chunks.append(c)
    nodes = []
    for _ in range(r.u()):
        n = {'node': ref(), 'kind': ref()}
        x = (r.s() + ox) / SCALE
        y = (r.s() + oy) / SCALE
        n['xyz'] = [x, y, r.s() / SCALE]
        n['mouths'] = []
        for _ in range(r.u()):
            sid = ref()
            n['mouths'].append([sid, r.s() / SCALE, r.s()])
        n['roles'] = {}
        for _ in range(r.u()):
            sid = ref()
            n['roles'][sid] = [ref(), ref()]
        n['plate'] = []
        for _ in range(r.u() if version >= 2 else 0):
            row = [ref(), r.s(), r.s() / SCALE]
            for _ in range(3):
                x = (r.s() + ox) / SCALE
                y = (r.s() + oy) / SCALE
                row.append([x, y, r.s() / SCALE])
            n['plate'].append(row)
        nodes.append(n)
    if r.at != r.end:
        raise ValueError('trailing bytes after road cell')
    return chunks, nodes


def encode_block(payloads):
    """payloads {slot: bytes} -> (block file, presence mask): country_format.encode_block's layout, own magic"""
    import country_format as binary
    data, mask = binary.encode_block(payloads)
    return MAGIC + data[len(binary.MAGIC):], mask


def decode_block(data):
    import country_format as binary
    if bytes(data[:4]) != MAGIC:
        raise ValueError('not an EarthNG road block')
    return binary.decode_block(binary.MAGIC + bytes(data[4:]))
