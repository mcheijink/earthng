"""Top-down images of what the Lua tile generator hands the game, on synthetic scenes: no game, no pack data.

Runs mod/lua/ge/extensions/e2r/generator.lua (and through it sweep.lua) under LuaJIT (pip install lupa) on a made-up cell,
with the swept roads encoded by the exporter's own fast_roads.build (lanes, clamps, mapped barriers, the median rule), and
draws every near tile it builds, seen from above: ground and areas, road surfaces, sidewalks and kerbs, markings, bridge
decks, buildings, guard rails (the beam and, darker, the posts), fences, lamps (with their arm), trees and other props.
Each scene writes an overview and close-ups where a detail matters, so a session without the game can look at its work:
a rail's posts on the side away from the traffic, a lamp clear of the kerb, a lane line where it should be.

Scenes (all made up, no real place):
  rails   OSM guard rails beside a two-way road, mapped in both directions on both sides; a divided motorway with mapped
          outer rails, a mapped median barrier and a stretch left to the median rule (sweep.lua)
  town    generator_budget.py's synthetic town: a street grid, a motorway with a slip road, a bridge, houses, lamps
  split   a two-lane slip road parting into two one-lane ones (the shape of lane_split_test.py's scenes)

Run: python earthng/render_plan.py [--scene rails town split] [--out out/render] [--px 1.0] [--list]
Images go to out/render/<scene>_<view>.png (out/ is not committed). Other scripts can call scene_images(scene, out) with a
Scene of their own (segments in fast_sources' shape, as lane_split_test.py builds them).
"""
import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))

CELL = 1000.
TILE = 500.

# ---------------------------------------------------------------------------------------------------
# Scenes: a generator cell (cellformat.cell()'s fields) and the segments the pack's v3 roads are built from
# ---------------------------------------------------------------------------------------------------


class Scene:
    """cell: the generator cell; swept: segments for fast_roads.build (fast_sources' shape); barriers: OSM barrier ways
    for fast_roads.furniture ((way id, (kind,), [[x, y], ...])); views: (name, x0, y0, x1, y1, px per m)"""

    def __init__(self, name, cell, swept=(), barriers=(), views=(), note=''):
        self.name, self.cell, self.swept, self.barriers, self.views, self.note = name, cell, list(swept), list(barriers), list(views), note


def flat_terrain(z=40., grid=10.):
    n = int(round(CELL / grid)) + 1
    return {'rect': [0., 0., CELL, CELL], 'nx': n, 'ny': n, 'heights': [z] * (n * n)}


def empty_cell(z=40.):
    return {'terrain': flat_terrain(z), 'roads': [], 'junctions': [], 'buildings': [], 'areas': [], 'lines': [], 'points': []}


def segment(sid, nodes, xy, tags, z=40.):
    """a road as fast_sources._prepare makes it (flat at z), for fast_roads.build"""
    import fast_sources
    xy = [[float(x), float(y)] for x, y in xy]
    length = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    return {'id': sid, 'way': sid, 'tags': dict(tags), 'xy': xy, 'nodes': list(nodes), 'width': fast_sources.road_width(tags),
            'prof': [[0., length], [z, z]]}


def cell_road(sid, nodes, xy, tags, width, z=40.):
    return {'id': str(sid), 'width': width, 'tags': dict(tags), 'nodes': list(nodes), 'points': [[x, y, z] for x, y in xy]}


def rails_scene():
    import fast_sources
    cell = empty_cell()
    # a two-way road (drawn by the generator: not a swept class) along y = 100, a guard rail mapped on each side in each
    # direction: the way's direction must not decide which side the posts stand on
    tags = {'highway': 'secondary', 'lanes': '2', 'maxspeed': '100'}
    width = fast_sources.road_width(tags)
    cell['roads'].append(cell_road(1, ('a', 'b'), [(0, 100), (1000, 100)], tags, width))
    off = width / 2 + 1.
    for name, y, x0, x1 in (('north_east', 100 + off, 60, 200), ('north_west', 100 + off, 400, 260),
                            ('south_east', 100 - off, 60, 200), ('south_west', 100 - off, 400, 260)):
        cell['lines'].append({'id': 'rail_' + name, 'kind': 'guard_rail', 'tags': {'barrier': 'guard_rail'},
                              'points': [[x0, y, 40.], [x0 + (x1 - x0) / 2, y, 40.], [x1, y, 40.]]})
    # a divided motorway (swept): the eastbound carriageway at y = 300, the westbound one 12 m north. Mapped outer rails
    # from x 100 to 350, a mapped median barrier from 450 to 700 (the pack gives it to both carriageways), and the
    # median left to the rule elsewhere (fast_roads.median_rails: edges under 6 m apart, nothing mapped)
    mt = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'maxspeed': 'none'}
    east = [(x, 300.) for x in range(0, 1001, 50)]
    west = [(x, 312.) for x in range(1000, -1, -50)]
    swept = [segment(2001, ('e0', 'e1'), east, mt), segment(2002, ('w0', 'w1'), west, mt)]
    for s in swept:
        cell['roads'].append(cell_road(s['id'], s['nodes'], s['xy'], s['tags'], s['width']))
    half = swept[0]['width'] / 2
    barriers = [(3001, [[100, 300 - half - 1.], [350, 300 - half - 1.]]), (3002, [[350, 312 + half + 1.], [100, 312 + half + 1.]]),
                (3003, [[450, 306.], [700, 306.]])]
    for wid, xy in barriers:
        cell['lines'].append({'id': 'barrier_%d' % wid, 'kind': 'guard_rail', 'tags': {'barrier': 'guard_rail'},
                              'points': [[x, y, 40.] for x, y in xy]})
    # close-ups at 80 px/m, 10 m along and 6 m across each rail (a post stands 0.12-0.24 m off the beam's line), and wider
    # views for the context
    def close(name, x, y):
        return (name, x - 5, y - 3, x + 5, y + 3, 80)
    so = 300 - half - 1.
    views = [close('road_north_way_east', 125, 100 + off), close('road_south_way_east', 125, 100 - off),
             close('road_north_way_west', 325, 100 + off), close('road_south_way_west', 325, 100 - off),
             close('motorway_outer_south', 225, so), close('motorway_outer_north', 225, 612 - so),
             close('motorway_mapped_median', 585, 306), close('motorway_rule_median', 845, 306),
             ('road', 110, 88, 350, 112, 5), ('motorway', 150, 285, 900, 327, 2), ('overview', 0, 50, 1000, 350, 1.2)]
    return Scene('rails', cell, swept, [(w, ('guard_rail',), xy) for w, xy in barriers], views,
                 'guard rails: posts belong on the side away from the traffic; a median rail faces both carriageways')


def town_scene():
    import generator_budget as gb
    cell, swept_roads = gb.synthetic_cell(10, random.Random(1))
    swept = []
    for r in swept_roads:
        xy = [[p[0], p[1]] for p in r['points']]
        z = sum(p[2] for p in r['points']) / len(r['points'])
        s = segment(int(r['id']), r['nodes'], xy, r['tags'], z)
        swept.append(s)
    views = [('overview', 0, 0, 1000, 1000, 1.0), ('junction', 440, 440, 560, 560, 8), ('street', 200, 270, 320, 330, 8),
             ('slip_road', 440, 40, 600, 180, 5)]
    return Scene('town', cell, swept, [], views, 'lamps clear of the kerb and the asphalt, rails, markings, props')


def split_scene():
    def arc(R, side, L=220., n=12):
        return [(R * math.sin(s / R), side * R * (1 - math.cos(s / R))) for s in [L * k / (n - 1) for k in range(n)]]
    shift = lambda pts: [(x + 400., y + 500.) for x, y in pts]
    t2 = {'highway': 'motorway_link', 'oneway': 'yes', 'lanes': '2'}
    t1 = {'highway': 'motorway_link', 'oneway': 'yes', 'lanes': '1'}
    swept = [segment(4001, (1, 2), shift([(-300, 0), (-150, 0), (0, 0)]), t2),
             segment(4002, (2, 3), shift(arc(600., 1)), t1), segment(4003, (2, 4), shift(arc(600., -1)), t1)]
    cell = empty_cell()
    for s in swept:
        cell['roads'].append(cell_road(s['id'], s['nodes'], s['xy'], s['tags'], s['width']))
    views = [('overview', 50, 440, 650, 560, 2.5), ('parting', 360, 485, 480, 515, 12)]
    return Scene('split', cell, swept, [], views, 'a two-lane slip road parting into two one-lane ones: lanes and lines')


SCENES = {'rails': rails_scene, 'town': town_scene, 'split': split_scene}

# ---------------------------------------------------------------------------------------------------
# The Lua side: build the tiles, hand back every mesh as flat triangles and every static item
# ---------------------------------------------------------------------------------------------------
LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
local R3 = require('ge/extensions/e2r/roads3')
function SWEPT(bytes, version)
  local cell = R3.cell(bytes, 1, #bytes, 0, 0, version, function() end)
  return R3.merge({cell}, nil)
end
-- {material, layer, 'x y z x y z ...' (three corners a triangle)} per mesh; {shape, kind, x, y, z, yaw} per static
function TILE(cells, i, j, options)
  options.pause, options.mark = function() end, function() end
  local road, scene = GENERATOR.build(cells, i, j, options)
  local meshes, statics = {}, {}
  for li, m in ipairs({road, scene}) do
    for _, it in ipairs(m and m.items or {}) do
      if it.kind == 'proc' then
        local ox, oy, oz = it.pos[1], it.pos[2], it.pos[3] or 0
        for k, mesh in ipairs(it.meshes) do
          local v, f, flat = mesh.verts, mesh.faces, {}
          for q = 1, #f do
            local p = v[f[q].v + 1]
            flat[#flat + 1] = string.format('%.3f %.3f %.3f', p.x + ox, p.y + oy, p.z + oz)
          end
          meshes[#meshes + 1] = {tostring(it.mats and it.mats[k] or mesh.material), li == 1 and 'road' or 'scene', table.concat(flat, ' ')}
        end
      elseif it.kind == 'static' and it.pos then
        local r = it.rot or {0, 0, 0, 1}
        statics[#statics + 1] = {tostring(it.shape), tostring(it.streetKind or ''), it.pos[1], it.pos[2], it.pos[3] or 0,
          2 * math.atan2(r[3] or 0, r[4] or 1), it.name or ''}
      end
    end
  end
  return meshes, statics
end
'''


def lua_runtime():
    for name in ('luajit21', 'luajit20'):
        try:
            return __import__('lupa.' + name, fromlist=[name]).LuaRuntime(unpack_returned_tuples=True)
        except ImportError:
            pass
    raise SystemExit('LuaJIT is needed: pip install lupa (lupa.luajit21 or lupa.luajit20)')


def to_lua(L, value):
    if isinstance(value, dict):
        return L.table_from({k: to_lua(L, v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return L.table_from([to_lua(L, v) for v in value])
    return value


def pack_roads(scene):
    """the scene's swept segments as one pack v3 road cell, through the exporter's whole road pass"""
    import fast_roads as fr
    extras = {'barriers': [(w, rec, __import__('numpy').array(xy, float)) for w, rec, xy in scene.barriers]}
    rows, nodes, _, _ = fr.build(scene.swept, extras, [], CELL)
    return fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.)), fr.VERSION


def build_scene(scene, L=None):
    """every near tile the scene touches: (triangles by material and layer, statics)"""
    L = L or lua_runtime()
    L.globals().REPO_ROOT = ROOT
    if L.globals().TILE is None:
        L.execute(LUA)
    cells = to_lua(L, [scene.cell])
    roads3 = None
    if scene.swept:
        data, version = pack_roads(scene)
        roads3 = L.globals().SWEPT(data, version)
    xs = [p[0] for r in scene.cell['roads'] for p in r['points']] or [0., CELL]
    ys = [p[1] for r in scene.cell['roads'] for p in r['points']] or [0., CELL]
    tiles = [(i, j) for i in range(int(max(0, min(xs)) // TILE), int(min(CELL - 1, max(xs)) // TILE) + 1)
             for j in range(int(max(0, min(ys)) // TILE), int(min(CELL - 1, max(ys)) // TILE) + 1)]
    meshes, statics = [], []
    for i, j in tiles:
        options = L.table_from({'tileSize': TILE, 'regionId': 'synthetic', 'materials': '/earthng/planet/materials.json',
                                'packs': '/e2r_stream/fast-art/v10/packs.json', 'far': False, 'horizon': False, 'farDecor': False,
                                'farStep': 100, 'horizonStep': 100, 'latitude': 52.5, 'seaLevel': 0, 'country': 'de',
                                'heightPipeline': True, 'detail': 50})
        if roads3 is not None:
            options.roads3 = roads3
        m, s = L.globals().TILE(cells, i, j, options)
        for e in m.values():
            mat, layer, flat = e[1], e[2], e[3]
            nums = [float(v) for v in flat.split()] if flat else []
            meshes.append((mat.replace('e2rf_', ''), layer, [nums[k:k + 9] for k in range(0, len(nums) - 8, 9)]))
        for e in s.values():
            statics.append((e[1], e[2], e[3], e[4], e[5], e[6], e[7]))
    return meshes, statics, tiles

# ---------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------
# (draw order, fill) by material; the first key the material starts with or contains wins
GROUND = {'ground': (176, 190, 150), 'meadow': (186, 204, 150), 'farmland': (205, 196, 150), 'forest': (120, 150, 105),
          'park': (160, 196, 130), 'edge_grass': (168, 188, 135), 'edge_dirt': (176, 160, 130), 'water': (120, 160, 205)}
MATS = [('road', 2, (72, 72, 78)), ('asphalt', 2, (72, 72, 78)), ('sidewalk', 3, (168, 168, 162)),
        ('footway', 3, (180, 172, 150)), ('curb', 3, (205, 205, 200)), ('mark', 4, (250, 250, 250)),
        ('line', 4, (250, 250, 250)), ('parapet', 5, (150, 120, 95)), ('fence_wall', 5, (150, 120, 95)),
        ('deck', 5, (150, 120, 95)), ('fence', 6, (110, 110, 110)), ('clad_metal', 7, (235, 120, 30))]
BUILDING = (205, 190, 165)
POST = (150, 20, 20)
GROUNDCOVER = ('grass', 'clover', 'daisies', 'fern', 'generibush')


def style(mat, layer):
    m = mat.lower()
    for key, fill in GROUND.items():
        if m.startswith(key):
            return 0 if key != 'water' else 1, fill
    for key, order, fill in MATS:
        if key in m:
            return order, fill
    return (5, BUILDING) if layer == 'scene' else (2, (90, 90, 96))


def draw(meshes, statics, view, path, title, groundcover=False):
    from PIL import Image, ImageDraw
    name, x0, y0, x1, y1, px = view
    W, H = max(1, int((x1 - x0) * px)), max(1, int((y1 - y0) * px))
    img = Image.new('RGB', (W, H + 34), (236, 236, 230))
    d = ImageDraw.Draw(img)
    to = lambda x, y: ((x - x0) * px, 34 + (y1 - y) * px)
    tris = []
    for mat, layer, ts in meshes:
        order, fill = style(mat, layer)
        for t in ts:
            xs, ys = t[0::3], t[1::3]
            if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
                continue
            zs = t[2::3]
            # a rail's post reaches below the ground (sink) the full rail height: darker than the beam
            fill_t = POST if mat == 'clad_metal' and max(zs) - min(zs) > .9 else fill
            tris.append((order, sum(zs) / 3, fill_t, t))
    tris.sort(key=lambda e: (e[0], e[1]))
    for order, _, fill, t in tris:
        pts = [to(t[k], t[k + 1]) for k in (0, 3, 6)]
        area = abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]))
        if area < 1.:
            # an upright face seen from above: its edge
            a, b = max(((p, q) for p in pts for q in pts), key=lambda e: (e[0][0] - e[1][0]) ** 2 + (e[0][1] - e[1][1]) ** 2)
            d.line([a, b], fill=fill, width=2 if order >= 7 else 1)
        else:
            d.polygon(pts, fill=fill)
    counts = {}
    for shape, kind, x, y, z, yaw, iname in statics:
        if not (x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3):
            continue
        low = shape.lower()
        if not groundcover and any(g in low for g in GROUNDCOVER):
            continue
        cx, cy = to(x, y)
        if kind == 'lamp':
            r = max(3., .35 * px)
            # lamp1's arm is its local +Y: (sin yaw, cos yaw) in the world (YARD)
            ax, ay = to(x + 1.6 * math.sin(yaw), y + 1.6 * math.cos(yaw))
            d.line([(cx, cy), (ax, ay)], fill=(60, 60, 20), width=max(1, int(.12 * px)))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 215, 0), outline=(60, 60, 20))
            counts['lamps'] = counts.get('lamps', 0) + 1
        elif kind == 'tree' or 'tree' in low or 'pine' in low:
            r = max(2., 1.6 * px)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(40, 110, 40))
            counts['trees'] = counts.get('trees', 0) + 1
        else:
            r = max(2., .4 * px)
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(60, 90, 200))
            counts['props'] = counts.get('props', 0) + 1
    # the title, a 10 m scale bar and the legend
    d.rectangle([0, 0, W, 33], fill=(250, 250, 248))
    d.text((4, 2), '%s  x %g..%g  y %g..%g  (north up)' % (title, x0, x1, y0, y1), fill=(0, 0, 0))
    legend = [((72, 72, 78), 'road'), ((168, 168, 162), 'sidewalk'), ((250, 250, 250), 'marks'), ((235, 120, 30), 'rail beam'),
              (POST, 'rail post'), ((150, 120, 95), 'deck/wall'), ((255, 215, 0), 'lamp'), ((60, 90, 200), 'prop')]
    x = 4
    for fill, label in legend:
        d.rectangle([x, 20, x + 9, 29], fill=fill, outline=(0, 0, 0))
        d.text((x + 12, 19), label, fill=(0, 0, 0))
        x += 14 + 7 * len(label)
    bar = 10 * px if 10 * px <= W / 3 else 100 * px
    d.line([(W - 8 - bar, 26), (W - 8, 26)], fill=(0, 0, 0), width=2)
    d.text((W - 8 - bar, 12), '10 m' if 10 * px <= W / 3 else '100 m', fill=(0, 0, 0))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    img.save(path)
    return counts


def scene_images(scene, out, L=None, scale=1.):
    """build the scene and write its views: [(path, counts)]"""
    meshes, statics, tiles = build_scene(scene, L)
    written = []
    for view in scene.views:
        v = (view[0], view[1], view[2], view[3], view[4], view[5] * scale)
        path = os.path.join(out, '%s_%s.png' % (scene.name, view[0]))
        written.append((path, draw(meshes, statics, v, path, '%s / %s' % (scene.name, view[0]))))
    return written, tiles, sum(len(t) for _, _, t in meshes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scene', nargs='*', default=list(SCENES), help='scenes to draw (default: all)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'render'))
    ap.add_argument('--px', type=float, default=1., help='scale every view by this')
    ap.add_argument('--list', action='store_true', help='list the scenes and their views')
    args = ap.parse_args()
    if args.list:
        for name, make in SCENES.items():
            s = make()
            print('%-6s %s' % (name, s.note))
            for v in s.views:
                print('         %-32s x %g..%g y %g..%g at %g px/m' % v)
        return 0
    L = lua_runtime()
    for name in args.scene:
        if name not in SCENES:
            raise SystemExit('unknown scene %r: %s' % (name, ', '.join(SCENES)))
        scene = SCENES[name]()
        written, tiles, tris = scene_images(scene, args.out, L, args.px)
        print('%s: tiles %s, %d triangles' % (name, ' '.join('%d,%d' % t for t in tiles), tris))
        for path, counts in written:
            print('  %s %s' % (os.path.relpath(path, ROOT), ' '.join('%s %d' % kv for kv in sorted(counts.items()))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
