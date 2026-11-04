"""Offline top-down renders of swept lane splits: the exporter (fast_roads) on synthetic geometry, the real sweep.lua under
LuaJIT (pip install lupa matplotlib), and every triangle it makes for the road surface and the markings (lane, edge and fork
lines, the gore's outline and chevrons), as the game would spawn them (sweep.items).

Synthetic data only: two-lane links (3.5 m lanes, trunk_link) whose OSM node lies about 40 m before the nose, where each
branch's line reaches its lane's centre (arcs of R 460 m); the mirror merge; three lanes parting into two and one. Thin
overlays: the expected divider (red, the joint road's lane line carried on to the nose) and the expected nose (red cross).
Two fork cases more: a host that keeps its lanes as a link leaves it (red: its edge line), and two links ending on a third
road.

Run: python earthng/split_render.py [--label after] [--out docs/renders] [--cases diverge merge ...]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_roads as fr  # noqa: E402
import fast_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_BRANCH = 460.          # m: 1.75 m (half a 3.5 m lane) off the joint road's line 40 m after the node
NOSE = math.sqrt(2 * R_BRANCH * 1.75)
_ID = [0]


def seg(nodes, xy, lanes, highway='trunk_link', **tags):
    _ID[0] += 1
    t = dict(highway=highway, oneway='yes', lanes=str(lanes), **tags)
    xy = [list(map(float, p)) for p in xy]
    L = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    return {'id': _ID[0], 'way': 1000 + _ID[0], 'tags': t, 'xy': xy, 'nodes': list(nodes), 'width': fast_sources.road_width(t),
            'prof': [[0., L], [10., 10.]]}


def arc(R, side, L=220., step=20.):
    """an OSM line from the origin along +x bending to `side` (vertices every `step` m, as mapped)"""
    return [(R * math.sin(s / R), side * R * (1 - math.cos(s / R))) for s in np.arange(0., L + 1e-6, step)]


def back(pts):
    return [(-x, y) for x, y in pts][::-1]


def cases():
    """{name: (segments, expected divider [(x, y)], nose (x, y), view box)}"""
    out = {}
    straight = [(-300, 0), (-200, 0), (-100, 0), (0, 0)]
    out['diverge'] = ([seg((1, 2), straight, 2), seg((2, 3), arc(R_BRANCH, 1), 1), seg((2, 4), arc(R_BRANCH, -1), 1)],
                      [(-60, 0), (NOSE, 0)], (NOSE, 0))
    # the through branch mapped straight on, the other leaving to the left
    out['diverge-left'] = ([seg((1, 2), straight, 2), seg((2, 3), arc(R_BRANCH / 2, 1), 1), seg((2, 4), arc(1e6, -1), 1)],
                           [(-60, 0), (NOSE / math.sqrt(2), 0)], (NOSE / math.sqrt(2), 0))
    out['merge'] = ([seg((2, 1), back(straight), 2), seg((3, 2), back(arc(R_BRANCH, 1)), 1), seg((4, 2), back(arc(R_BRANCH, -1)), 1)],
                    [(60, 0), (-NOSE, 0)], (-NOSE, 0))
    # three lanes parting into two and one: the divider between the second and third lane (from the left)
    w, r3 = 3.5, R_BRANCH * 1.7
    nose3 = math.sqrt(2 * r3 * w)               # its lane's centre 3.5 m right of the three-lane road's line
    out['3-way'] = ([seg((1, 2), straight, 3), seg((2, 3), arc(4 * R_BRANCH, 1), 2), seg((2, 4), arc(r3, -1), 1)],
                    [(-60, -w / 2), (nose3, -w / 2)], (nose3, -w / 2))
    # a fork where the host keeps both lanes and a one-lane link leaves it (no lane leaves: the host's edge line runs on)
    out['host-keeps-lanes'] = ([seg((1, 2), straight, 2), seg((2, 3), arc(1e6, 1), 2), seg((2, 4), arc(R_BRANCH / 2, -1), 1)],
                               [(-60, -3.5), (60, -3.5)], (math.sqrt(2 * R_BRANCH / 2 * 5.25), -3.5))
    # two one-lane links ending side by side on a two-lane trunk road (a fan plate at the node, no stroke through it)
    out['links-on-a-third'] = ([seg((3, 2), back(arc(R_BRANCH, 1)), 1), seg((4, 2), back(arc(R_BRANCH, -1)), 1),
                                seg((2, 1), back(straight), 2, 'trunk')], [(60, 0), (-NOSE, 0)], (-NOSE, 0))
    return out


LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local SW = require('ge/extensions/e2r/sweep')
local R3 = require('ge/extensions/e2r/roads3')
return function(bytes, version)
  local cell = R3.cell(bytes, 1, #bytes, 0, 0, version, function() end)
  local plan = SW.plan(R3.merge({cell}, nil), {rect = {-1000, -1000, 1000, 1000}, classes = SW.CLASSES.fitted})
  local items = SW.items(plan, {tileKey = 'r', markKind = 'mark'})
  -- flat triangle lists: {material, x1, y1, x2, y2, x3, y3} (world)
  local out = {}
  for _, it in ipairs(items) do
    for _, m in ipairs(it.meshes) do
      local f = m.faces
      for k = 1, #f, 3 do
        local a, b, c = m.verts[f[k].v + 1], m.verts[f[k + 1].v + 1], m.verts[f[k + 2].v + 1]
        out[#out + 1] = {m.material, a.x + it.pos[1], a.y + it.pos[2], b.x + it.pos[1], b.y + it.pos[2], c.x + it.pos[1], c.y + it.pos[2]}
      end
    end
  end
  return out
end
'''


def triangles(segs):
    """sweep.lua's triangles of these segments: {material: [((x, y) * 3)]}"""
    import lupa.luajit21 as lj
    g = fr.Graph(segs)
    strokes = fr.build_strokes(segs, g)
    fr.lane_events(strokes, g, segs)
    if hasattr(fr, 'lane_splits'):
        fr.lane_splits(strokes, g, segs)
    rows, nodes = fr.records(strokes, g, segs, 1000.)
    data = fr.encode_cell([r[4] for r in rows], [n[3] for n in nodes], (0., 0.))
    L = lj.LuaRuntime(unpack_returned_tuples=True)
    L.globals().REPO_ROOT = ROOT
    out = {}
    for t in L.execute(LUA)(data, fr.VERSION).values():
        v = list(t.values())
        out.setdefault(v[0], []).append(((v[1], v[2]), (v[3], v[4]), (v[5], v[6])))
    return out, strokes


def render(name, segs, divider, nose, label, path, box=(-70, 110, -14, 14)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    tris, strokes = triangles(segs)
    fig, ax = plt.subplots(figsize=(18, 4.2), dpi=110)
    ax.set_facecolor('#4d6b3c')
    for mat, ts in tris.items():
        if mat.endswith('mark'):
            continue
        ax.add_collection(PolyCollection(ts, facecolors='#55585c', edgecolors='#55585c', linewidths=.2))
    ax.add_collection(PolyCollection(tris.get('e2rf_mark', []), facecolors='white', edgecolors='white', linewidths=.15))
    for st in strokes:
        ax.plot(st.x, st.y, color='#ffcc33', lw=.5, ls=':')
    ax.plot(*zip(*divider), color='red', lw=.35, alpha=.8)
    ax.plot([nose[0]], [nose[1]], marker='+', color='red', ms=14, mew=1., alpha=.8)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect('equal')
    ax.set_title(f'{name} ({label}): grey surface, white marks, yellow dotted OSM lines, red expected divider and nose', fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return tris


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--label', default='after')
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'renders'))
    ap.add_argument('--cases', nargs='*')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (segs, divider, nose) in cases().items():
        if args.cases and name not in args.cases:
            continue
        box = (-70, 110, -14, 14) if divider[0][0] < 0 else (-110, 70, -14, 14)
        path = os.path.join(args.out, f'{name}-{args.label}.png')
        tris = render(name, segs, divider, nose, args.label, path, box)
        print(path, {k: len(v) for k, v in tris.items()})


if __name__ == '__main__':
    main()
