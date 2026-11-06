"""Tests of the junction markings on synthetic geometry: fast_roads strokes and fast_junctions records, encoded as in the pack,
drawn by sweep.lua under LuaJIT (pip install lupa; the Lua checks are skipped without it).

- a give-way line stands at the edge of the road it joins (a tee, a roundabout's ring), not a kerb radius back;
- a crossing on a one-way road leaving a junction is drawn (the outbound carriageway of a dual carriageway);
- no lane or centre line runs over a crossing, nor over a junction surface;
- a roundabout's ring carries no lane lines, its entries a give-way line each.
Run: python earthng/junction_marks_test.py
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_junctions as fj  # noqa: E402
import fast_roads as fr  # noqa: E402
import fast_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ID = [100]


def seg(nodes, xy, highway, **tags):
    """a segment as fast_sources._prepare makes it, flat at z = 10"""
    _ID[0] += 1
    t = dict(highway=highway, **tags)
    xy = [list(map(float, p)) for p in xy]
    L = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    return {'id': _ID[0], 'way': 1000 + _ID[0], 'tags': t, 'xy': xy, 'nodes': list(nodes), 'width': fast_sources.road_width(t),
            'prof': [[0., L], [10., 10.]]}


def extras(**kw):
    out = {'controls': [], 'crossings': [], 'islands': [], 'barriers': [], 'water': [], 'marks': []}
    out.update(kw)
    return out


def build(segs, ex):
    """strokes, junction records, and the road and junction cells' bytes"""
    g = fr.Graph(segs)
    strokes = fr.build_strokes(segs, g)
    fr.lane_events(strokes, g, segs)
    rows, nodes = fr.records(strokes, g, segs, 1000.)
    views = {st.id: fj.view_of_stroke(st, segs) for st in strokes}
    jrows = fj.build([d for _, _, _, d in nodes], views, ex, 1000., workers=1)
    data = fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.))
    jdata = fj.encode_cell([r[3] for r in jrows], (0., 0.))
    return strokes, [r[3] for r in jrows], data, jdata


def tee(minor='tertiary', **kw):
    return [seg((1, 5), [(-300, 0), (0, 0)], 'primary', lanes='2'), seg((5, 2), [(0, 0), (300, 0)], 'primary', lanes='2'),
            seg((5, 4), [(0, 0), (0, 300)], minor, lanes='2', **kw)]


def dual():
    """a dual carriageway (y = +-7) crossed by a secondary road at x = 0"""
    return [seg((1, 5), [(-300, 7), (0, 7)], 'primary', lanes='2', oneway='yes'),
            seg((5, 2), [(0, 7), (300, 7)], 'primary', lanes='2', oneway='yes'),
            seg((12, 6), [(300, -7), (0, -7)], 'primary', lanes='2', oneway='yes'),
            seg((6, 11), [(0, -7), (-300, -7)], 'primary', lanes='2', oneway='yes'),
            seg((3, 6), [(0, -300), (0, -7)], 'secondary'), seg((6, 5), [(0, -7), (0, 7)], 'secondary'),
            seg((5, 4), [(0, 7), (0, 300)], 'secondary')]


RING_R = 18.


def roundabout(ring_lanes='1'):
    """a one-way ring of radius RING_R (counter-clockwise) with four two-way arms"""
    ang, ids, segs = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi], [10, 11, 12, 13, 10], []
    for k in range(4):
        pts = [(RING_R * math.cos(a), RING_R * math.sin(a)) for a in np.linspace(ang[k], ang[k + 1], 10)]
        segs.append(seg((ids[k], ids[k + 1]), pts, 'secondary', junction='roundabout', oneway='yes', lanes=ring_lanes))
    for k, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        segs.append(seg((20 + k, ids[k]), [(dx * 300, dy * 300), (dx * RING_R, dy * RING_R)], 'secondary'))
    return segs


def at(st, s):
    return float(np.interp(s, st.s, st.x)), float(np.interp(s, st.s, st.y))


class Records(unittest.TestCase):
    def test_give_way_line_at_the_major_road(self):
        strokes, J, _, _ = build(tee(), extras())
        a = [a for a in J[0]['approaches'] if a['control'] == 'give_way'][0]
        minor = [st for st in strokes if st.id == a['stroke']][0]
        major = [st for st in strokes if st.highway == 'primary'][0]
        x, y = at(minor, a['stop'])
        # 0.5 m (STOP_BEFORE) past CLEAR beyond the primary's paved edge; it stood at the kerb fillets' end (16.5 m)
        self.assertAlmostEqual(y, major.eL[0] + fj.CLEAR + fj.STOP_BEFORE, delta=.4)

    def test_roundabout_entries_yield_at_the_ring(self):
        strokes, J, _, _ = build(roundabout(), extras())
        self.assertEqual(J[0]['kind'], 'roundabout')
        ring = [st for st in strokes if st.tags.get('junction') == 'roundabout'][0]
        aps = J[0]['approaches']
        self.assertEqual(sorted(a['control'] for a in aps), ['give_way'] * 4)
        for a in aps:
            st = [s for s in strokes if s.id == a['stroke']][0]
            r = math.hypot(*at(st, a['stop']))
            # (the clearance is searched in 0.25 m steps; the line stood at 31.5 m, 11 m out on the arm)
            self.assertAlmostEqual(r, RING_R + ring.eL[0] + fj.CLEAR + fj.STOP_BEFORE, delta=.4)

    def test_crossing_on_the_outbound_carriageway(self):
        # a signalled pedestrian crossing across both carriageways, 30 m west of the junction
        ex = extras(controls=[(5, {'highway': 'traffic_signals'}, 0., 7.), (6, {'highway': 'traffic_signals'}, 0., -7.)],
                    crossings=[(91, ({'highway': 'footway', 'footway': 'crossing', 'crossing': 'traffic_signals'},),
                                np.array([[-30., -20.], [-30., 20.]]))])
        strokes, J, _, _ = build(dual(), ex)
        found = {}
        for a in J[0]['approaches']:
            st = [s for s in strokes if s.id == a['stroke']][0]
            for c in a['crossings']:
                x, y = at(st, c['s'])
                found[round(y)] = (c['kind'], round(x), a['control'])
        # the inbound (north) carriageway's, and the outbound one's, which leaves the junction westward: no line on it
        self.assertEqual(found, {7: ('signals', -30, 'signals'), -7: ('signals', -30, 'none')})

    def test_crossing_kinds(self):
        self.assertEqual(fj.crossing_kind({'crossing': 'uncontrolled'})[0], 'zebra')
        self.assertEqual(fj.crossing_kind({'crossing': 'marked', 'crossing:markings': 'yes'})[0], 'zebra')
        self.assertEqual(fj.crossing_kind({'crossing': 'marked', 'crossing:markings': 'lines'})[0], 'marked')
        self.assertEqual(fj.crossing_kind({'crossing': 'marked', 'crossing:signals': 'yes'})[0], 'signals')
        self.assertEqual(fj.crossing_kind({'crossing': 'uncontrolled', 'bicycle': 'designated'})[0], 'cycle')


# ---------------------------------------------------------------------------------------------------------------------------
# sweep.lua
# ---------------------------------------------------------------------------------------------------------------------------
def lua_runtime():
    for name in ('luajit21', 'luajit20'):
        try:
            return __import__('lupa.' + name, fromlist=[name]).LuaRuntime(unpack_returned_tuples=True)
        except ImportError:
            pass
    return None


LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local SW = require('ge/extensions/e2r/sweep')
local R3 = require('ge/extensions/e2r/roads3')
local function mesh() return {verts = {}, normals = {}, uvs = {}, faces = {}, material = 'm', vmap = {}, vkz = {}, nmap = {},
  umap = {}, tris = 0} end
return function(bytes, version, jbytes)
  local cell = R3.cell(bytes, 1, #bytes, 0, 0, version, function() end)
  local js = R3.junctionCell(jbytes, 1, #jbytes, 0, 0, function() end)
  local plan = SW.plan(R3.merge({cell}, {js}), {rect = {-1000, -1000, 1000, 1000}, classes = SW.CLASSES.fitted})
  local lines, fills = {}, {}
  SW.debug = function(S, a, b, u0, u1, t0, t1, id, width)
    local ax, ay, bx, by = a.x + a.nx * u0, a.y + a.ny * u0, b.x + b.nx * u1, b.y + b.ny * u1
    lines[#lines + 1] = {S.id, id, ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1}
  end
  local fill = SW.fillSU
  SW.fillSU = function(p, S, m, o, s0, s1, u0, u1)
    fills[#fills + 1] = {S.id, s0, s1, u0, u1}
    return fill(p, S, m, o, s0, s1, u0, u1)
  end
  for _, S in pairs(plan.strokes) do
    for _, c in ipairs(S.list) do
      if c.rows and #c.rows > 0 then SW.markMesh(plan, S, c, mesh(), {0, 0, 0}) end
    end
  end
  for _, pl in ipairs(plan.plates) do if pl.junction then SW.junctionMarks(plan, pl.junction, mesh(), {0, 0, 0}) end end
  SW.debug, SW.fillSU = nil, fill
  return lines, fills
end
'''


def draw(data, jdata):
    """sweep.lua's line pieces (stroke, id, x0, y0, x1, y1) and filled marks (stroke, s0, s1, u0, u1)"""
    L = lua_runtime()
    L.globals().REPO_ROOT = ROOT
    lines, fills = L.execute(LUA)(data, fr.VERSION, jdata)
    return [list(p.values()) for p in lines.values()], [list(f.values()) for f in fills.values()]


def plate(J, inset=.3):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    V = J['verts']
    tris = [Polygon([V[a][:2], V[b][:2], V[c][:2]]) for a, b, c in J['tris']]
    return unary_union([t for t in tris if t.area > 1e-9]).buffer(-inset)


def over(lines, area, step=.5):
    """the line pieces with a point in area"""
    from shapely.geometry import Point
    out = []
    for p in lines:
        _, _, xa, ya, xb, yb = p
        n = int(math.hypot(xb - xa, yb - ya) / step) + 1
        if any(area.contains(Point(xa + (xb - xa) * k / n, ya + (yb - ya) * k / n)) for k in range(n + 1)):
            out.append(p)
    return out


@unittest.skipIf(lua_runtime() is None, 'needs LuaJIT: pip install lupa')
class Drawn(unittest.TestCase):
    def test_no_line_over_a_junction_surface(self):
        layouts = {
            'tee': (tee(), extras()),
            'signals and turn lanes': ([seg((1, 5), [(-300, 0), (0, 0)], 'primary', lanes='3', **{'lanes:forward': '2', 'lanes:backward': '1',
                                                                                         'turn:lanes:forward': 'left|through'}),
                                        seg((5, 2), [(0, 0), (300, 0)], 'primary', lanes='2'),
                                        seg((3, 5), [(0, -300), (0, 0)], 'secondary', lanes='2'),
                                        seg((5, 4), [(0, 0), (0, 300)], 'secondary', lanes='2')],
                                       extras(controls=[(5, {'highway': 'traffic_signals'}, 0., 0.)])),
            'dual carriageway': (dual(), extras(controls=[(5, {'highway': 'traffic_signals'}, 0., 7.)])),
            'roundabout': (roundabout('2'), extras()),
        }
        for name, (segs, ex) in layouts.items():
            strokes, J, data, jdata = build(segs, ex)
            self.assertEqual(len(J), 1, name)
            lines, _ = draw(data, jdata)
            self.assertGreater(len(lines), 50, name)
            self.assertEqual(over(lines, plate(J[0])), [], name)
            if name == 'roundabout':
                ring = [st.id for st in strokes if st.tags.get('junction') == 'roundabout']
                self.assertEqual([p for p in lines if p[0] in ring], [])

    def test_no_line_over_a_zebra_up_the_approach(self):
        from shapely.geometry import box
        # a zebra 25 m up a two-lane side road: the give-way line stays at the main road, the centre line stops at the zebra
        ex = extras(crossings=[(9, ({'footway': 'crossing', 'crossing': 'zebra'},), np.array([[-8., 25.], [8., 25.]]))])
        strokes, J, data, jdata = build(tee(), ex)
        a = [a for a in J[0]['approaches'] if a['control'] == 'give_way'][0]
        z = a['crossings'][0]
        minor = [st for st in strokes if st.id == a['stroke']][0]
        self.assertEqual(z['kind'], 'zebra')
        lines, fills = draw(data, jdata)
        _, y0 = at(minor, z['s'] - z['width'] / 2)
        _, y1 = at(minor, z['s'] + z['width'] / 2)
        lo, hi = min(y0, y1) + .05, max(y0, y1) - .05
        self.assertEqual(over([p for p in lines if p[0] == minor.id], box(-3., lo, 3., hi)), [])
        # the give-way line (0.5 m dashes) is drawn at its record's place, and the zebra's stripes
        stops = [f for f in fills if f[0] == minor.id and abs(max(f[1], f[2]) - a['stop']) < 1e-6 and abs(abs(f[2] - f[1]) - .5) < 1e-6]
        self.assertGreaterEqual(len(stops), 3)
        stripes = [f for f in fills if f[0] == minor.id and abs((f[1] + f[2]) / 2 - z['s']) < 1e-6]
        self.assertGreaterEqual(len(stripes), 5)

    def test_roundabout_give_way_lines_drawn_at_the_ring(self):
        strokes, J, data, jdata = build(roundabout(), extras())
        _, fills = draw(data, jdata)
        for a in J[0]['approaches']:
            got = [f for f in fills if f[0] == a['stroke'] and abs(max(f[1], f[2]) - a['stop']) < .51 and abs(min(f[1], f[2]) - a['stop']) < .51]
            self.assertGreaterEqual(len(got), 3, a['stroke'])

    def test_outbound_crossing_drawn_without_lines_over_it(self):
        from shapely.geometry import box
        ex = extras(controls=[(5, {'highway': 'traffic_signals'}, 0., 7.)],
                    crossings=[(91, ({'highway': 'footway', 'footway': 'crossing', 'crossing': 'traffic_signals'},),
                                np.array([[-30., -20.], [-30., 20.]]))])
        strokes, J, data, jdata = build(dual(), ex)
        lines, fills = draw(data, jdata)
        south = [st for st in strokes if st.y[0] < 0 and st.oneway][0]
        furt = [f for f in fills if f[0] == south.id]
        self.assertGreaterEqual(len(furt), 4)                    # its two dashed lines
        self.assertEqual(over([p for p in lines if p[0] == south.id and not p[1].startswith('edge')], box(-31.9, -10., -28.1, -4.)), [])


if __name__ == '__main__':
    unittest.main()
