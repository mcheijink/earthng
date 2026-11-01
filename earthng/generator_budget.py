"""Offline timing of the Lua tile generator: the stretches of work between its pause() calls.

native.lua gives generator.build() a pause() that hands the frame back when the budget is spent, so a
long stretch without a pause is a long frame. This runs mod/lua/ge/extensions/e2r/generator.lua under
LuaJIT outside the game (pip install lupa) on a synthetic 1 km cell (a town grid with houses, blocks,
a park, a pond, a motorway with a slip road and a bridge, fences, lamps, trees, a rural road and
fields) and reports the longest stretches, where they ran, and the time per stage (options.mark).

The machine, the data and the JIT state differ from the game, so the milliseconds are only a ranking:
a stretch that is long here is long there. The collector is stopped during a timed build (--with-gc keeps
it running): its steps land in any stretch and hid the ranking behind noise; in the game they are the
`less-garbage` task's concern. Run: python earthng/generator_budget.py [--grid 10|25] [--tiles 0,0 1,1]
[--far] [--swept] [--repeat 2] [--top 20] [--threshold 4] [--with-gc]
"""
import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))

CELL = 1000.


def lua_runtime():
    for name in ('luajit21', 'luajit20'):
        try:
            module = __import__('lupa.' + name, fromlist=[name])
            return module.LuaRuntime(unpack_returned_tuples=True), name
        except ImportError:
            pass
    raise SystemExit('LuaJIT is needed: pip install lupa (lupa.luajit21 or lupa.luajit20)')


def to_lua(L, value):
    if isinstance(value, dict):
        return L.table_from({k: to_lua(L, v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return L.table_from([to_lua(L, v) for v in value])
    return value


# ---------------------------------------------------------------------------------------------------
# A synthetic cell in the shape cellformat.cell() decodes (fields as the generator reads them)
# ---------------------------------------------------------------------------------------------------
def terrain_z(x, y):
    return 40 + 6 * math.sin(x / 170) + 5 * math.cos(y / 210 + 1) + 2 * math.sin((x + y) / 90)


def synthetic_cell(grid, rng):
    n = int(round(CELL / grid)) + 1
    heights = [round(terrain_z(ix * grid, iy * grid), 1) for iy in range(n) for ix in range(n)]
    terrain = {'rect': [0., 0., CELL, CELL], 'nx': n, 'ny': n, 'heights': heights}
    roads, buildings, areas, lines, points = [], [], [], [], []
    ids = {'n': 0}

    def next_id(prefix='w'):
        ids['n'] += 1
        return '%s%d' % (prefix, ids['n'])

    def pt(x, y, dz=0.):
        return [round(x, 2), round(y, 2), round(terrain_z(x, y) + dz, 2)]

    def road(nodes, xy, tags, width, dz=0., id=None):
        r = {'id': id or next_id(), 'width': width, 'tags': tags, 'nodes': list(nodes), 'points': [pt(x, y, dz) for x, y in xy]}
        roads.append(r)
        return r

    # the town: streets every 100 m from 200 to 800; the middle ones are a secondary and a primary road
    xs = [200 + 100 * k for k in range(7)]
    node = lambda i, j: 'n%d_%d' % (i, j)
    for i, x in enumerate(xs):
        for j in range(6):
            ya, yb = xs[j], xs[j + 1]
            if x == 500:
                tags, width = {'highway': 'secondary', 'lanes': '2', 'maxspeed': '50', 'lit': 'yes', 'sidewalk': 'both'}, 9.
            else:
                tags, width = {'highway': 'residential', 'maxspeed': '30', 'lit': 'yes'}, 6.
            road((node(i, j), node(i, j + 1)), [(x, ya), (x, yb)], tags, width)
    for j, y in enumerate(xs):
        for i in range(6):
            xa, xb = xs[i], xs[i + 1]
            if y == 500:
                tags, width = {'highway': 'primary', 'lanes': '2', 'maxspeed': '50', 'lit': 'yes', 'sidewalk': 'both'}, 11.
            else:
                tags, width = {'highway': 'residential', 'maxspeed': '30', 'lit': 'yes'}, 6.
            road((node(i, j), node(i + 1, j)), [(xa, y), (xb, y)], tags, width)
    # the secondary runs on south to the motorway's slip road and north to the edge
    road(('s_south', node(3, 0)), [(500, 150), (500, 200)], {'highway': 'secondary', 'lanes': '2', 'maxspeed': '50'}, 9.)
    road((node(3, 6), 's_north'), [(500, 800), (500, 1000)], {'highway': 'secondary', 'lanes': '2', 'maxspeed': '70'}, 9.)
    # the primary runs on to both edges
    road(('p_west', node(0, 3)), [(0, 500), (200, 500)], {'highway': 'primary', 'lanes': '2', 'maxspeed': '70'}, 11.)
    road((node(6, 3), 'p_east'), [(800, 500), (1000, 500)], {'highway': 'primary', 'lanes': '2', 'maxspeed': '70'}, 11.)
    # a motorway along the south edge in two ways (a node for the slip road at x = 500), a slip road to
    # the secondary, a service road over the motorway on a bridge
    def motorway_xy(x0, x1):
        return [(x, 80 + 30 * math.sin(x / 300)) for x in range(int(x0), int(x1) + 1, 25)]
    mtags = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'maxspeed': 'none', 'ref': 'A'}
    motorway = [road(('m_w', 'm_mid'), motorway_xy(0, 500), dict(mtags), 12., id='1001'),
                road(('m_mid', 'm_e'), motorway_xy(500, 1000), dict(mtags), 12., id='1002')]
    link = road(('m_mid', 's_south'), [(500, 80 + 30 * math.sin(500 / 300)), (520, 110), (515, 135), (500, 150)],
                {'highway': 'motorway_link', 'lanes': '1', 'oneway': 'yes'}, 5.5, id='1003')
    road(('b_s', 'b_n'), [(300, 20), (300, 60), (300, 100), (300, 140)],
         {'highway': 'service', 'bridge': 'yes', 'layer': '1'}, 5., dz=6.)
    # a winding rural road along the north edge
    rural = [(x, 900 + 40 * math.sin(x / 120)) for x in range(0, 1001, 40)]
    road(('r_w', 'r_e'), rural, {'highway': 'tertiary', 'maxspeed': '100'}, 7.)

    # houses along the streets (both sides, every 20 m, clear of the junctions); apartment blocks, a
    # park, a pond and a car park take over some blocks
    special = {(4, 4): 'park', (1, 1): 'water', (4, 1): 'parking', (2, 4): 'apartments', (1, 3): 'apartments'}

    def block_of(x, y):
        return (int((x - 200) // 100), int((y - 200) // 100))

    def rect_ring(x0, y0, x1, y1):
        return [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1), pt(x0, y0)]

    def house(cx, cy, along, w, d, tags, height):
        ux, uy = along
        nx, ny = -uy, ux
        ring = [pt(cx + ux * a + nx * b, cy + uy * a + ny * b) for a, b in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]
        ring.append(ring[0])
        buildings.append({'id': next_id('b'), 'kind': tags['building'], 'height': height, 'minHeight': 0., 'tags': tags, 'rings': [ring]})
        return cx, cy

    for r in roads:
        if r['tags']['highway'] not in ('residential', 'secondary', 'primary') or len(r['points']) != 2:
            continue
        (xa, ya, _), (xb, yb, _) = r['points']
        if not (200 <= xa <= 800 and 200 <= ya <= 800 and 200 <= xb <= 800 and 200 <= yb <= 800):
            continue
        length = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / length, (yb - ya) / length
        for side in (-1, 1):
            off = r['width'] / 2 + 9
            for k in range(1, 5):
                s = 20 * k
                cx, cy = xa + ux * s - uy * side * off, ya + uy * s + ux * side * off
                special_kind = special.get(block_of(cx, cy))
                if special_kind in ('park', 'water', 'parking'):
                    continue
                if special_kind == 'apartments':
                    if k == 2:
                        house(cx - uy * side * 6, cy + ux * side * 6, (ux, uy), 34, 14,
                              {'building': 'apartments', 'building:levels': '6', 'roof:shape': 'flat'}, 19.)
                    continue
                levels = 1 + rng.randrange(2)
                house(cx, cy, (ux, uy), 10 + rng.random() * 3, 8 + rng.random() * 2,
                      {'building': 'house', 'building:levels': str(levels), 'roof:shape': 'gabled'}, 4 + 3 * levels)
                if rng.random() < .25:
                    px, py = cx - uy * side * 4, cy + ux * side * 4
                    lines.append({'id': next_id('f'), 'kind': 'fence', 'tags': {'barrier': 'fence'},
                                  'points': rect_ring(px - 10, py - 10, px + 10, py + 10)})

    areas.append({'id': next_id('a'), 'kind': 'residential', 'tags': {'landuse': 'residential'}, 'rings': [rect_ring(150, 150, 850, 850)]})
    for (bi, bj), kind in special.items():
        x0, y0 = 200 + 100 * bi + 8, 200 + 100 * bj + 8
        if kind == 'park':
            areas.append({'id': next_id('a'), 'kind': 'park', 'tags': {'leisure': 'park'}, 'rings': [rect_ring(x0, y0, x0 + 84, y0 + 84)]})
            for gx in range(10):
                for gy in range(10):
                    points.append({'id': next_id('t'), 'kind': 'tree', 'tags': {'natural': 'tree'}, 'pos': pt(x0 + 4 + gx * 8.4, y0 + 4 + gy * 8.4)})
            lines.append({'id': next_id('l'), 'kind': 'path', 'tags': {'highway': 'footway'},
                          'points': [pt(x0, y0), pt(x0 + 42, y0 + 30), pt(x0 + 84, y0 + 84)]})
        elif kind == 'water':
            ring = [pt(x0 + 42 + 40 * math.cos(a), y0 + 42 + 36 * math.sin(a)) for a in [k * 2 * math.pi / 16 for k in range(16)]]
            ring.append(ring[0])
            areas.append({'id': next_id('a'), 'kind': 'water', 'tags': {'natural': 'water'}, 'rings': [ring]})
        elif kind == 'parking':
            areas.append({'id': next_id('a'), 'kind': 'parking', 'tags': {'amenity': 'parking'}, 'rings': [rect_ring(x0, y0, x0 + 84, y0 + 84)]})
    # woods and fields outside the town, a meadow by the pond
    for ring in ([pt(0, 150), pt(140, 150), pt(170, 400), pt(120, 700), pt(150, 850), pt(0, 850)],
                 [pt(850, 600), pt(1000, 600), pt(1000, 860), pt(850, 860)]):
        ring.append(ring[0])
        areas.append({'id': next_id('a'), 'kind': 'forest', 'tags': {'natural': 'wood'}, 'rings': [ring]})
    areas.append({'id': next_id('a'), 'kind': 'farmland', 'tags': {'landuse': 'farmland'}, 'rings': [rect_ring(850, 150, 1000, 600)]})
    areas.append({'id': next_id('a'), 'kind': 'meadow', 'tags': {'landuse': 'meadow'}, 'rings': [rect_ring(150, 860, 850, 1000)]})
    lines.append({'id': next_id('l'), 'kind': 'path', 'tags': {'highway': 'track', 'tracktype': 'grade2'},
                  'points': [pt(860, 160 + 40 * k) for k in range(11)]})
    # a guard rail beside the motorway
    lines.append({'id': next_id('l'), 'kind': 'guard_rail', 'tags': {'barrier': 'guard_rail'},
                  'points': [pt(x, y + 7.5) for x, y in motorway_xy(100, 900)]})
    # lamps along the town streets, bus stops, zebra crossings
    for r in roads:
        if r['tags'].get('lit') == 'yes' and len(r['points']) == 2:
            (xa, ya, _), (xb, yb, _) = r['points']
            length = math.hypot(xb - xa, yb - ya)
            ux, uy = (xb - xa) / length, (yb - ya) / length
            for s in (25, 50, 75):
                points.append({'id': next_id('p'), 'kind': 'lamp', 'tags': {'highway': 'street_lamp'},
                               'pos': pt(xa + ux * s - uy * (r['width'] / 2 + 1), ya + uy * s + ux * (r['width'] / 2 + 1))})
    for x, y in ((450, 500), (550, 500), (500, 450), (500, 550)):
        points.append({'id': next_id('p'), 'kind': 'stop', 'tags': {'highway': 'bus_stop'}, 'pos': pt(x, y)})
    for i, j in ((2, 2), (3, 2), (4, 3), (3, 4), (2, 3), (4, 4)):
        points.append({'id': next_id('p'), 'kind': 'crossing', 'tags': {'highway': 'crossing', 'crossing': 'zebra'}, 'pos': pt(xs[i], xs[j])})
    cell = {'terrain': terrain, 'roads': roads, 'junctions': [], 'buildings': buildings, 'areas': areas, 'lines': lines, 'points': points}
    return cell, motorway + [link]


def swept_bytes(swept_roads):
    """Pack v3 road bytes of the motorway and its slip road (fast_roads: the exporter's own encoder)."""
    import fast_roads as fr
    segments = []
    for r in swept_roads:
        xy = [[p[0], p[1]] for p in r['points']]
        length = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
        z = sum(p[2] for p in r['points']) / len(r['points'])
        # the paved width as fast_sources.road_width gives it for these tags (that module needs the
        # exporter's GIS stack): the tagged lanes at 3.5 m, a 3.75 m lane on the motorway
        lanes = float(r['tags'].get('lanes', 1))
        width = lanes * (3.75 if r['tags']['highway'] == 'motorway' else 3.5)
        segments.append({'id': int(r['id']), 'way': int(r['id']), 'tags': dict(r['tags']), 'xy': xy, 'nodes': list(r['nodes']),
                         'width': width, 'prof': [[0., length], [z, z]]})
    graph = fr.Graph(segments)
    strokes = fr.build_strokes(segments, graph)
    rows, nodes = fr.records(strokes, graph, segments, CELL)
    return fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.)), fr.VERSION


# ---------------------------------------------------------------------------------------------------
# The Lua side: the profiling pause and the build
# ---------------------------------------------------------------------------------------------------
PRELUDE = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local clock = os.clock
P = {top = {}, sites = {}, stages = {}, keep = 40}
local function where(level)
  local parts = {}
  for l = level, level + 2 do
    local info = debug.getinfo(l, 'Sl')
    if not info then break end
    parts[#parts + 1] = (info.short_src:match('([^/\\]+)%.lua$') or info.short_src) .. ':' .. tostring(info.currentline)
  end
  return table.concat(parts, ' < ')
end
-- --profile: a sample of the running line every 1000 VM instructions, kept per stretch (the hook turns the
-- JIT off, so the times are several times longer; the shares within a stretch hold)
local function hook()
  local info = debug.getinfo(2, 'Sl')
  if info then
    local k = (info.short_src:match('([^/\\]+)%.lua$') or info.short_src) .. ':' .. tostring(info.currentline)
    P.hist[k] = (P.hist[k] or 0) + 1
  end
end
function P.start()
  P.count, P.total, P.worst, P.last, P.lastSite, P.stage, P.stageAt = 0, 0, 0, clock(), 'start', 'start', clock()
  P.top, P.sites, P.stages, P.ring, P.worst8, P.hist = {}, {}, {}, {}, 0, {}
  if P.profile then debug.sethook(hook, '', 1000) end
end
local function record(dt, from, to)
  P.count, P.total = P.count + 1, P.total + dt
  if dt > P.worst then P.worst = dt end
  -- the work between every 8th pause: what native.lua's pause() let through while it read the clock
  -- only at every 8th call
  local ring = P.ring
  ring[P.count % 8 + 1] = dt
  local sum = 0
  for k = 1, 8 do sum = sum + (ring[k] or 0) end
  if sum > P.worst8 then P.worst8 = sum end
  local site = P.sites[to]
  if not site then site = {n = 0, sum = 0, max = 0, from = from}; P.sites[to] = site end
  site.n, site.sum = site.n + 1, site.sum + dt
  if dt > site.max then site.max, site.from = dt, from end
  local top = P.top
  if #top < P.keep or dt > top[#top].dt then
    top[#top + 1] = {dt = dt, from = from, to = to, hist = P.hist}
    table.sort(top, function(a, b) return a.dt > b.dt end)
    if #top > P.keep then top[#top] = nil end
  end
end
function P.pause()
  local now = clock()
  local here = where(3)
  record((now - P.last) * 1000, P.lastSite, here)
  P.hist = {}
  P.last, P.lastSite = clock(), here
end
function P.mark(name)
  local now = clock()
  P.stages[#P.stages + 1] = {P.stage, (now - P.stageAt) * 1000}
  P.stage, P.stageAt = name, now
end
function P.finish()
  if P.profile then debug.sethook() end
  local now = clock()
  record((now - P.last) * 1000, P.lastSite, 'end')
  P.stages[#P.stages + 1] = {P.stage, (now - P.stageAt) * 1000}
end
GENERATOR = require('ge/extensions/e2r/generator')
ROADS3 = require('ge/extensions/e2r/roads3')
function BUILD(cells, i, j, options, withGc)
  options.pause, options.mark = P.pause, P.mark
  if not withGc then collectgarbage(); collectgarbage('stop') end
  P.start()
  local road, scene, far, nav = GENERATOR.build(cells, i, j, options)
  P.finish()
  if not withGc then collectgarbage('restart'); collectgarbage() end
  local function count(m) local n = 0; for _, it in ipairs(m and m.items or {}) do n = n + (it.triangles or 0) end; return n end
  return count(road), count(scene), count(far), nav and #nav.segments or 0
end
function SWEPT(bytes, version)
  local cell = ROADS3.cell(bytes, 1, #bytes, 0, 0, version, function() end)
  return ROADS3.merge({cell}, nil)
end
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--grid', type=float, default=10, help='elevation grid step in m (10 = a national DTM, 25 = Terrarium)')
    ap.add_argument('--tiles', nargs='*', default=['0,0', '1,1'], help='500 m tiles i,j of the 1 km cell at the origin')
    ap.add_argument('--far', action='store_true', help='build the far layer instead of the near one')
    ap.add_argument('--swept', action='store_true', help='hand the tile pack v3 road cells of the motorway (sweep.lua)')
    ap.add_argument('--no-height-pipeline', action='store_true', help='the old height passes and ground pull')
    ap.add_argument('--repeat', type=int, default=2, help='builds per tile; the last one is reported (the JIT is warm)')
    ap.add_argument('--top', type=int, default=20, help='stretches listed')
    ap.add_argument('--threshold', type=float, default=4., help='ms: stretches over this are counted as long')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--with-gc', action='store_true', help='leave the collector running during the timed build')
    ap.add_argument('--profile', action='store_true', help='sample the running line and show where each long stretch spends its time')
    args = ap.parse_args()

    L, flavour = lua_runtime()
    L.globals().REPO_ROOT = ROOT
    L.globals().load(PRELUDE, '=harness')()
    L.globals().P.profile = args.profile
    cell, swept_roads = synthetic_cell(args.grid, random.Random(args.seed))
    print('LuaJIT via lupa.%s; cell: %d roads, %d buildings, %d areas, %d lines, %d points, grid %g m' % (
        flavour, len(cell['roads']), len(cell['buildings']), len(cell['areas']), len(cell['lines']), len(cell['points']), args.grid))
    cells = to_lua(L, [cell])
    roads3 = None
    if args.swept:
        data, version = swept_bytes(swept_roads)
        roads3 = L.globals().SWEPT(data, version)
        print('swept roads: %d chunks, %d nodes from %d bytes' % (len(roads3.chunks), len(roads3.nodes), len(data)))
    exit_code = 0
    for tile in args.tiles:
        i, j = (int(v) for v in tile.split(','))
        for _ in range(args.repeat):
            options = L.table_from({
                'tileSize': 500, 'regionId': 'synthetic', 'materials': '/earthng/planet/materials.json',
                'packs': '/e2r_stream/fast-art/v10/packs.json', 'far': args.far, 'horizon': False, 'farDecor': args.far,
                'farStep': 100, 'horizonStep': 100, 'latitude': 52.5, 'seaLevel': 0, 'country': 'de',
                'heightPipeline': not args.no_height_pipeline, 'detail': 50})
            if roads3 is not None:
                options.roads3 = roads3
            result = L.globals().BUILD(cells, i, j, options, args.with_gc)
        P = L.globals().P
        stretches = [(e.dt, e['from'], e.to, dict(e.hist.items())) for e in P.top.values()]
        sites = sorted(((s.max, s.sum, s.n, to, s['from']) for to, s in P.sites.items()), reverse=True)
        long_ones = sum(1 for s in sites if s[0] > args.threshold)
        print('\n=== tile %d,%d (%s): road %d tris, scene %d tris, far %d tris, nav %d segments' % (
            i, j, 'far' if args.far else 'near', result[0], result[1], result[2], result[3]))
        print('build %.0f ms in %d pauses; worst stretch %.1f ms, worst 8 in a row %.1f ms; %d call sites with a stretch over %.0f ms' % (
            P.total, P.count, P.worst, P.worst8, long_ones, args.threshold))
        print('stages (ms):', ', '.join('%s %.0f' % (name, ms) for name, ms in (tuple(st.values()) for st in P.stages.values())))
        print('longest stretches (ms, ended at the pause at .. after the one at ..):')
        for dt, frm, to, hist in stretches[:args.top]:
            print('  %6.1f  %s  <-  %s' % (dt, to, frm))
            if args.profile and hist:
                total = sum(hist.values())
                lines = sorted(hist.items(), key=lambda kv: -kv[1])[:5]
                print('          ' + ', '.join('%s %d%%' % (k, round(100 * v / total)) for k, v in lines))
        print('call sites by their worst stretch (max ms, total ms, count):')
        for mx, total, n, to, frm in sites[:args.top]:
            print('  %6.1f %7.0f %6d  %s' % (mx, total, n, to))
        if P.worst > args.threshold * 2:
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
