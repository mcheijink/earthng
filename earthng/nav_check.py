"""Offline check of the AI traffic graph (the tiles' nav segments, as world.lua's navBuild joins them).

A tile's build (generator.lua, sweep.lua for swept roads) returns nav segments
{id, node a, node b, lanes ('+' with the drawing direction, '-' against it), speed (m/s), radius,
drivability, x, y, z, ...}; world.lua joins them by their end nodes into the graph BeamNG's AI drives.
This builds tiles under LuaJIT (lupa), joins their segments the same way and reports what traps or
strands a car: dead ends (a node a one-way flow reaches and cannot leave, or leaves and cannot reach),
pieces cut off from the rest, lanes that differ from the way's tags, one-way segments drawn against
their way, and segments that repeat another between the same two nodes.

Used by earthng/nav_check_test.py; run it on the generator_budget cell with
python earthng/nav_check.py [--swept] [--tiles 0,0 1,0 0,1 1,1]
"""
import argparse
import collections
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'earthng'))


def segments_of(result_nav):
    """Python lists of a Lua nav table's segments."""
    out = []
    for s in (result_nav.segments.values() if result_nav is not None and result_nav.segments is not None else []):
        out.append(list(s.values()))
    return out


class Graph:
    """The graph world.lua's navBuild makes of a set of segments (inner knots of a segment are its own)."""

    def __init__(self, segments):
        self.segments = {}
        for s in segments:
            self.segments.setdefault(s[0], s)      # one copy per id, as world.lua's S.nav
        self.pos = {}
        self.out = collections.defaultdict(set)    # node -> nodes a car may drive to
        self.inn = collections.defaultdict(set)
        self.edges = collections.Counter()         # unordered end pairs, for repeats
        for sid, s in self.segments.items():
            a, b, lanes = s[1], s[2], s[3]
            pts = [s[k:k + 3] for k in range(7, len(s), 3)]
            if len(pts) < 2:
                continue
            ids = ['n:' + str(a)] + ['s:%s:%d' % (sid, k) for k in range(1, len(pts) - 1)] + ['n:' + str(b)]
            for i, p in zip(ids, pts):
                self.pos.setdefault(i, p)
            forward, backward = '+' in lanes, '-' in lanes
            for p, q in zip(ids, ids[1:]):
                if p == q:
                    continue
                if forward:
                    self.out[p].add(q); self.inn[q].add(p)
                if backward:
                    self.out[q].add(p); self.inn[p].add(q)
            self.edges[tuple(sorted((str(a), str(b))))] += 1

    def dead_ends(self, inside):
        """(sinks, sources): nodes inside(x, y) that a car reaches and cannot leave, or leaves and cannot reach."""
        sinks, sources = [], []
        for node, p in self.pos.items():
            if not inside(p[0], p[1]):
                continue
            out = self.out.get(node, set()) - {node}
            inn = self.inn.get(node, set()) - {node}
            if inn and not out:
                sinks.append((node, p))
            if out and not inn:
                sources.append((node, p))
        return sinks, sources

    def pieces(self):
        """The weakly connected pieces, largest first, as lists of nodes."""
        seen, out = set(), []
        neighbours = collections.defaultdict(set)
        for a, bs in self.out.items():
            for b in bs:
                neighbours[a].add(b); neighbours[b].add(a)
        for start in self.pos:
            if start in seen:
                continue
            stack, piece = [start], []
            seen.add(start)
            while stack:
                n = stack.pop()
                piece.append(n)
                for m in neighbours[n]:
                    if m not in seen:
                        seen.add(m); stack.append(m)
            out.append(piece)
        return sorted(out, key=len, reverse=True)

    def repeats(self):
        return [pair for pair, n in self.edges.items() if n > 1]


def heading(s):
    pts = [s[k:k + 3] for k in range(7, len(s), 3)]
    (x0, y0, _), (x1, y1, _) = pts[0], pts[-1]
    return math.atan2(y1 - y0, x1 - x0)


def report(graph, inside, ways=None):
    """Findings as text lines; ways: {way id: (oneway direction 1/-1/0, lanes tag, points)} to check against."""
    lines = []
    sinks, sources = graph.dead_ends(inside)
    for kind, found in (('dead end (no way out)', sinks), ('dead end (no way in)', sources)):
        for node, p in found:
            lines.append('%s at %.0f, %.0f (%s)' % (kind, p[0], p[1], node))
    pieces = graph.pieces()
    for piece in pieces[1:]:
        inner = [n for n in piece if inside(*graph.pos[n][:2])]
        if inner:
            x, y = graph.pos[inner[0]][:2]
            lines.append('a piece of %d nodes cut off from the rest, near %.0f, %.0f' % (len(piece), x, y))
    for pair in graph.repeats():
        lines.append('segments repeated between %s and %s' % pair)
    return lines


# ------------------------------------------------------------------------------------------------
LUA = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
GENERATOR = require('ge/extensions/e2r/generator')
ROADS3 = require('ge/extensions/e2r/roads3')
function NAV(cells, i, j, options)
  local road, scene, far, nav = GENERATOR.build(cells, i, j, options)
  return nav
end
function SWEPT(bytes, version)
  return ROADS3.merge({ROADS3.cell(bytes, 1, #bytes, 0, 0, version, function() end)}, nil)
end
'''


def lua():
    for name in ('luajit21', 'luajit20'):
        try:
            L = __import__('lupa.' + name, fromlist=[name]).LuaRuntime(unpack_returned_tuples=True)
            L.globals().REPO_ROOT = ROOT
            L.execute(LUA)
            return L
        except ImportError:
            pass
    raise SystemExit('needs lupa with LuaJIT')


def to_lua(L, value):
    if isinstance(value, dict):
        return L.table_from({k: to_lua(L, v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return L.table_from([to_lua(L, v) for v in value])
    return value


def build_nav(L, cell, tiles, roads3=None, tile_size=500):
    """Every segment the tiles give (each tile its own, as world.lua reads them)."""
    segments = []
    for i, j in tiles:
        options = to_lua(L, {'tileSize': tile_size, 'regionId': 'nav', 'far': False, 'horizon': False,
                             'farStep': 100, 'latitude': 52.5, 'seaLevel': 0, 'country': 'de', 'heightPipeline': True})
        if roads3 is not None:
            options.roads3 = roads3
        segments += segments_of(L.globals().NAV(to_lua(L, [cell]), i, j, options))
    return segments


def swept_roads3(L, roads):
    import generator_budget
    data, version = generator_budget.swept_bytes(roads)
    return L.globals().SWEPT(data, version)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--swept', action='store_true')
    ap.add_argument('--tiles', nargs='*', default=['0,0', '1,0', '0,1', '1,1'])
    args = ap.parse_args()
    import generator_budget
    L = lua()
    cell, swept = generator_budget.synthetic_cell(10, random.Random(1))
    roads3 = swept_roads3(L, swept) if args.swept else None
    tiles = [tuple(int(v) for v in t.split(',')) for t in args.tiles]
    segments = build_nav(L, cell, tiles, roads3)
    graph = Graph(segments)
    x0, y0 = min(i for i, _ in tiles) * 500, min(j for _, j in tiles) * 500
    x1, y1 = (max(i for i, _ in tiles) + 1) * 500, (max(j for _, j in tiles) + 1) * 500
    inside = lambda x, y: x0 + 30 < x < x1 - 30 and y0 + 30 < y < y1 - 30
    print('%d segments, %d nodes, %d pieces' % (len(graph.segments), len(graph.pos), len(graph.pieces())))
    for line in report(graph, inside):
        print('  ' + line)


if __name__ == '__main__':
    main()
