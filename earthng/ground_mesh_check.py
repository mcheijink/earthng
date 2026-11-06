"""Ground-mesh checks through the real tile generator (mod/lua generator.lua) in LuaJIT, with lupa.

The ground of a near tile is the lattice in the road layer, and the swept roads (sweep.lua) lie on it. With
generator.CONFORMAL.on the ground is cut by the swept fast roads (carriageway and verge bands), the toe of a band is
split at every lattice line, and the road edges stand on the bands' own vertices. This module builds tiles with the
switch off and on and measures them; ground_mesh_test.py runs the gates on synthetic tiles, and the command line runs
real tiles of a staged pack:

    python earthng/ground_mesh_check.py --pack <pack dir> --tiles "i,j;i,j" [--switch off|on|both] [--alone] [-v]

A swept chunk is drawn whole by the tile that holds its middle, so a tile built alone has gaps where a neighbour's road
runs into it. The checks therefore run on a block: the tile with its eight neighbours loaded, measured inside the
centre tile (--alone measures the tile by itself).

Per tile (metres):
  tris      triangles per class (ground and verge bands, generator road, swept road, plate, deck, drape, ...)
  layers    share of the tile's 2 m cells with 1, 2, 3 or more stacked triangles (no drape, deck, sea or markings), and
            the pairs that make the second layer (the p18 tallies)
  under     area of swept carriageway (not under a bridge deck's soffit) that has ground in the same 1 m cell; and the
            area of carriageway over carriageway: coplanar (under 1 cm, z-fight: forks and merges), within 1 m, overpass
  stitch    corners of the carriageway's free edges: welded to a ground corner within 1 mm in x, y and z, to another
            road (a corner of two meshes, or standing on its surface at its height), or open
  cracks    boundary edges of the ground, carriageway and plates, each tried at three points with a probe 1.5 mm past
            it: a 'hole' when no triangle holds the probe (the car falls through), a 'step' when one does at another
            height. Holes anywhere, and on the tile border (the seam), are the gate; steps in the corridor of the swept
            road away from what stays on top of the ground are listed with the classes that meet
  walls     upright concrete (the end walls that close a band at a bridge or tunnel, wing walls) inside a swept carriageway: a hit is
            a wall the car can drive into (walls_in_road; zero tolerance). A bridge's own parapets are not tried
  leaks     rays from the carriageway sideways and a little downward that hit nothing of the tile and end under the surface
            (see_through): the view through the world. cracks() looks in plan view and has no say where two layers of surface
            stack (the embankment of one road over the cut of another) or where a band ends at a bridge: there is a surface
            under every point, so no hole, and the step between the layers is counted in the corridor only away from decks (the
            6 m round anything that stays on top of the ground, a deck included, is exempt), so a 6 m cliff at an abutment
            passed. A face that closes it stands in no plan view. Compare with the switch off: leaks happen there too
  build     seconds of generator.build for the centre tile (not the pack parse)"""
import argparse
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import lupa.luajit21 as lupa
except ImportError:      # pragma: no cover
    lupa = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = .001               # m: the collector's weld quantum, the crack tolerance

PRELUDE = r'''
package.path = REPO_ROOT .. '/mod/lua/?.lua;' .. package.path
local ffi = require('ffi')
local floor = math.floor
GEN = require('ge/extensions/e2r/generator')
local CF = require('ge/extensions/e2r/cellformat')
local R = require('ge/extensions/e2r/roads3')
P = {}
local root, blocks, rblocks, rversion, jblocks, CS
local cache = {}
function P.setup(r, b, rb, rv, jb, cs) root, blocks, rblocks, rversion, jblocks, CS = r, b, rb, rv, jb, cs; cache = {} end
local function bytes(path)
  local hit = cache[path]
  if hit then return hit end
  local f = assert(io.open(root .. path, 'rb')); hit = f:read('*a'); f:close()
  cache[path] = hit
  return hit
end
-- the pack's cells round a tile (the generator reads 40 m past it), and its swept road cells (250 m past)
function P.cells(i, j, size)
  local x0, y0 = i * size, j * size
  local rect = {x0, y0, x0 + size, y0 + size}
  local out, seen = {}, {}
  for ci = floor((x0 - 40) / CS), math.ceil((x0 + size + 40) / CS) - 1 do
    for cj = floor((y0 - 40) / CS), math.ceil((y0 + size + 40) / CS) - 1 do
      local key = ci .. '_' .. cj
      local p = blocks[floor(ci / 8) .. '_' .. floor(cj / 8)]
      if p and not seen[key] then
        local data = bytes(p)
        local range = CF.block(data)[ci % 8 + (cj % 8) * 8]
        if range then
          seen[key] = true
          out[#out + 1] = CF.cell(data, range[1], range[2], ci * CS, cj * CS, {ci * CS, cj * CS, (ci + 1) * CS, (cj + 1) * CS}, nil, rect)
        end
      end
    end
  end
  return out, rect
end
function P.roads3(rect)
  local rc, jc = {}, jblocks and {} or nil
  for ci = floor((rect[1] - 250) / CS), floor((rect[3] + 250) / CS) do
    for cj = floor((rect[2] - 250) / CS), floor((rect[4] + 250) / CS) do
      local slot = ci % 8 + (cj % 8) * 8
      local p = rblocks[floor(ci / 8) .. '_' .. floor(cj / 8)]
      if p then
        local data = bytes(p)
        local range = R.block(data)[slot]
        if range then rc[#rc + 1] = R.cell(data, range[1], range[2], ci * CS, cj * CS, rversion) end
      end
      local q = jblocks and jblocks[floor(ci / 8) .. '_' .. floor(cj / 8)]
      if q and R.junctionBlock then
        local data = bytes(q)
        local range = R.junctionBlock(data)[slot]
        if range then jc[#jc + 1] = R.junctionCell(data, range[1], range[2], ci * CS, cj * CS) end
      end
    end
  end
  return R.merge(rc, jc)
end
-- synthetic v3 chunks (a table of chunk tables) as the road set a tile reads
function P.merge(chunks) return R.merge({{chunks = chunks, nodes = {}}}) end
local function options(size, rect)
  return {tileSize = size, regionId = 'test', pause = function() end, seaLevel = 20, materials = '/m.json',
    packs = '/p.json', latitude = 52.9, heightPipeline = true, look = {asphalt = 'smooth', block = 'western',
      buildings = 'mixed', markings = 'white', statics = 'street', trees = 'regional', verge = 'grass'}}
end
-- one near tile: cells (a table of cell tables, or nil for the pack's), the road layer and the scene
function P.build(i, j, mode, conformal, cells, extra)
  local size = 500
  local rect = {i * size, j * size, (i + 1) * size, (j + 1) * size}
  local opts = options(size, rect)
  if not cells then
    cells = P.cells(i, j, size)
    opts.roads3 = P.roads3(rect)
  end
  for k, v in pairs(extra or {}) do opts[k] = v end
  GEN.SWEPT.on = opts.roads3 ~= nil
  GEN.SWEPT.mode = mode or 'fast'
  if GEN.CONFORMAL then GEN.CONFORMAL.on = conformal and true or false end
  local t0 = os.clock()
  local road, scene = GEN.build(cells, i, j, opts)
  local seconds = os.clock() - t0
  GEN.SWEPT.on = false
  if GEN.CONFORMAL then GEN.CONFORMAL.on = false end
  return road, scene, seconds
end
-- a pause that times the stretches of work between its calls (CPU seconds): the longest is a hitch of the frame that runs it,
-- as the game's pause hands the frame back only when its budget is spent, and a stretch without a pause call does not
function P.stopwatch()
  local last, worst = os.clock(), 0
  return function()
    local now = os.clock()
    if now - last > worst then worst = now - last end
    last = os.clock()
  end, function() return worst end
end
-- pieces (meshes), triangles, and the triangles of the objects that join the static collision (collision.lua quiet: markings,
-- covers, roofs are hidden from the rebuild; the rest of a ProceduralMesh is solid)
local COLL = require('ge/extensions/e2r/collision')
function P.tally(mans)
  local pieces, tris, solid = 0, 0, 0
  for _, mm in ipairs(mans) do
    for _, it in ipairs(mm[2].items or {}) do
      local n = 0
      for _, m in ipairs(it.meshes or {}) do n = n + floor(#m.faces / 3) end
      pieces, tris = pieces + #(it.meshes or {}), tris + n
      if it.meshes and #it.meshes > 0 and it.collision ~= 'None' and not COLL.quiet(it.meshes, mm[1]) then solid = solid + n end
    end
  end
  return pieces, tris, solid
end
-- the triangles of the manifests as nine floats each (x, y, z world), the group of each, and the groups
function P.pack(mans)
  local n = 0
  for _, mm in ipairs(mans) do
    for _, it in ipairs(mm[2].items or {}) do
      for _, m in ipairs(it.meshes or {}) do n = n + floor(#m.faces / 3) end
    end
  end
  local buf = ffi.new('double[?]', 9 * n + 1)
  local meta = ffi.new('int32_t[?]', n + 1)
  local groups, t = {}, 0
  for _, mm in ipairs(mans) do
    for _, it in ipairs(mm[2].items or {}) do
      local o = it.pos
      for _, m in ipairs(it.meshes or {}) do
        local mat = (m.material or ''):gsub('^e2rf_', '')
        groups[#groups + 1] = table.concat({mm[1], it.name or '', mat, it.decor and 1 or 0, it.layer or '', it.stroke or ''}, string.char(9))
        local g = #groups - 1
        local f, v = m.faces, m.verts
        for k = 1, #f - 2, 3 do
          local a, b, c = v[f[k].v + 1], v[f[k + 1].v + 1], v[f[k + 2].v + 1]
          local q = t * 9
          buf[q], buf[q + 1], buf[q + 2] = a.x + o[1], a.y + o[2], a.z + o[3]
          buf[q + 3], buf[q + 4], buf[q + 5] = b.x + o[1], b.y + o[2], b.z + o[3]
          buf[q + 6], buf[q + 7], buf[q + 8] = c.x + o[1], c.y + o[2], c.z + o[3]
          meta[t] = g
          t = t + 1
        end
      end
    end
  end
  return ffi.string(buf, 72 * t), ffi.string(meta, 4 * t), table.concat(groups, string.char(10)), t
end
'''

CLASSES = ['gnd', 'gwat', 'sea', 'drape', 'road', 'sw_road', 'sw_plate', 'kerb', 'walk', 'island', 'deck']
CI = {c: i for i, c in enumerate(CLASSES)}
COVER = {'ground', 'meadow', 'wetland', 'forest', 'scrub', 'park', 'farmland', 'bare_ground', 'trodden', 'orchard', 'beach',
         'paved', 'heath', 'sand', 'grass', 'dry_grass', 'scree', 'bare_rock', 'glacier', 'vineyard', 'residential', 'dirt'}
ROADM = {'road', 'road_major', 'road_worn', 'road_gravel', 'road_dirt', 'road_broken', 'road_track'}
SW_NAME = re.compile(r'_sw_(\d*)([a-z]+)$')
# classes that are one ground-attached surface (level 0), water bodies and the generator's kerbs and sidewalks included: the ones the crack check closes
LEVEL0 = ('gnd', 'gwat', 'road', 'sw_road', 'sw_plate', 'kerb', 'walk', 'island')


def classify(layer, name, mat, decor):
    """class index of a mesh, or -1 (markings, rails, walls, buildings: not surfaces)"""
    if decor == '1' or 'marks' in name or mat in ('mark', 'edge_grass', 'edge_dirt', 'mark_yellow'):
        return -1
    m = SW_NAME.search(name)
    if m:
        k = m.group(2)
        if k == 's':
            return CI['sw_road'] if mat.startswith('road') else CI['sw_plate']
        if k == 'plates':
            return CI['sw_plate']
        if k == 'k':
            return CI['kerb'] if 'curb' in mat else CI['walk']
        if k == 'islands':
            return CI['kerb'] if 'curb' in mat else (CI['island'] if mat == 'sidewalk' else CI['gnd'])
        if k == 'f':
            return CI['deck'] if mat == 'fence_wall' else -1
        return -1
    if mat == 'fence_wall':
        return CI['deck']
    if layer == 'scene':
        if mat == 'water':
            return CI['sea']
        if mat in COVER or mat in ('footway', 'sidewalk'):
            return CI['drape']
        return -1
    if mat == 'water':
        return CI['gwat']
    if mat in COVER:
        return CI['gnd']
    if mat in ROADM:
        return CI['road']
    if mat == 'curb':
        return CI['kerb']
    if mat in ('sidewalk', 'footway', 'cycleway'):
        return CI['walk']
    return -1


class Tile:
    """a built near tile: D (n, 9) triangle corners, cls (n,) class index, grp (n,) group index"""

    def __init__(self, i, j, data, meta, groups, seconds, size=500., label=''):
        self.i, self.j, self.seconds, self.size, self.label = i, j, seconds, size, label
        self.rect = (i * size, j * size, (i + 1) * size, (j + 1) * size)
        D = np.frombuffer(data, dtype=np.float64).reshape(-1, 9).copy()
        meta = np.frombuffer(meta, dtype=np.int32)
        G = [g.split('\t') for g in groups.split('\n')] if groups else []
        gcls = np.array([classify(*g[:4]) for g in G], dtype=np.int8)
        self.groups = G
        self.cls = gcls[meta] if len(meta) else np.zeros(0, np.int8)
        self.grp = meta
        self.D = D
        keep = self.cls >= 0      # surfaces only: markings, rails, buildings are class -1
        self.D, self.cls, self.grp = self.D[keep], self.cls[keep], self.grp[keep]

    def of(self, *names):
        return self.D[np.isin(self.cls, [CI[n] for n in names])]

    def counts(self):
        return {c: int((self.cls == CI[c]).sum()) for c in CLASSES}


class Runtime:
    """one LuaJIT state with the generator; build tiles of a staged pack or from cells made in Python"""

    def __init__(self, pack=None):
        # latin-1 both ways: the packed triangle bytes come back as str and go through .encode('latin-1')
        self.L = lupa.LuaRuntime(unpack_returned_tuples=True, encoding='latin-1')
        self.L.globals().REPO_ROOT = ROOT.replace('\\', '/')
        self.L.execute(PRELUDE)
        self.P = self.L.globals().P
        self.kept = {}
        self.has_switch = bool(self.L.eval("require('ge/extensions/e2r/generator').CONFORMAL ~= nil"))
        if pack:
            self.setup_pack(pack)

    def setup_pack(self, pack):
        import glob
        import json
        manifest = json.load(open(glob.glob(os.path.join(pack, 'earthng/countries/*/regions/*/manifest.json'))[0]))
        blocks = {k: b['path'] for k, b in manifest['blocks'].items()}
        rb = {k: b['path'] for k, b in manifest['roads']['blocks'].items()}
        jb = {k: b['path'] for k, b in manifest['roads'].get('junctions', {}).get('blocks', {}).items()} or None
        self.P.setup(pack.replace('\\', '/').rstrip('/') + '/', self.table(blocks), self.table(rb), manifest['roads']['version'],
                     self.table(jb) if jb else None, manifest['cellSize'])

    def table(self, v):
        if isinstance(v, dict):
            t = self.L.table()
            for k, x in v.items():
                t[k] = self.table(x)
            return t
        if isinstance(v, (list, tuple)):
            t = self.L.table()
            for n, x in enumerate(v):
                t[n + 1] = self.table(x)
            return t
        return v

    def block(self, i, j, conformal=False, mode='fast', cells=None, extra=None, ring=1):
        """tile (i, j) as it looks with its neighbours loaded: the triangles of the (2 ring + 1) squared tiles round it
        in one Tile (a swept chunk is drawn whole by the tile that holds its middle, so a tile's own mesh alone has
        gaps where the neighbour's road runs into it). The rect stays the centre tile's; seconds is the centre's."""
        tiles = [self.build(i + di, j + dj, conformal, mode, cells, extra)
                 for dj in range(-ring, ring + 1) for di in range(-ring, ring + 1)]
        centre = tiles[len(tiles) // 2]
        out = Tile.__new__(Tile)
        out.i, out.j, out.size, out.label, out.seconds, out.rect = i, j, centre.size, centre.label, centre.seconds, centre.rect
        out.groups, out.D = [], []
        cls, grp, base = [], [], 0
        for t in tiles:
            out.D.append(t.D)
            cls.append(t.cls)
            grp.append(t.grp + base)
            out.groups += t.groups
            base += len(t.groups)
        out.D, out.cls, out.grp = np.concatenate(out.D), np.concatenate(cls), np.concatenate(grp)
        # only what can touch the centre tile (60 m round it): the checks are Python loops over triangles
        x0, y0, x1, y1 = out.rect
        xs, ys = out.D[:, [0, 3, 6]], out.D[:, [1, 4, 7]]
        near = (xs.max(1) > x0 - 60) & (xs.min(1) < x1 + 60) & (ys.max(1) > y0 - 60) & (ys.min(1) < y1 + 60)
        out.D, out.cls, out.grp = out.D[near], out.cls[near], out.grp[near]
        out.alone = centre
        return out

    def build(self, i, j, conformal=False, mode='fast', cells=None, extra=None, label=''):
        # tiles of a staged pack are kept: a block reads the same neighbours for the next tile along the road
        key = (i, j, bool(conformal), mode) if cells is None and extra is None else None
        if key in self.kept:
            return self.kept[key]
        tile = self._build(i, j, conformal, mode, cells, extra, label)
        if key:
            self.kept[key] = tile
        return tile

    def budget(self, i, j, conformal=False, mode='fast', cells=None, extra=None, repeats=2):
        """what tile (i, j), built alone, costs the game (see `budget`): the tile, its pieces and solid triangles, and the longest
        stretch of the build without a pause call"""
        # (the longest stretch is the least of `repeats` builds: a garbage-collector step or another process lands in one)
        gap = math.inf
        for _ in range(repeats):
            pause, worst = self.P.stopwatch()
            road, scene, seconds = self.P.build(i, j, mode, bool(conformal), self.table(cells) if cells is not None else None,
                                                self.table(dict(extra or {}, pause=pause)))
            gap = min(gap, worst())
        mans = self.L.table_from([self.L.table_from(['road', road]), self.L.table_from(['scene', scene])])
        data, meta, groups, n = self.P.pack(mans)
        tile = Tile(i, j, data.encode('latin-1'), meta.encode('latin-1'), groups, seconds)
        pieces, tris, solid = self.P.tally(mans)
        return budget(tile, pieces, tris, solid, gap)

    def _build(self, i, j, conformal, mode, cells, extra, label):
        road, scene, seconds = self.P.build(i, j, mode, bool(conformal), self.table(cells) if cells is not None else None,
                                            self.table(extra) if extra else None)
        mans = self.L.table_from([self.L.table_from(['road', road]), self.L.table_from(['scene', scene])])
        data, meta, groups, n = self.P.pack(mans)
        return Tile(i, j, data.encode('latin-1'), meta.encode('latin-1'), groups, seconds, label=label)


# ------------------------------------------------------------------------------------------------------------------
# sampling helpers
# ------------------------------------------------------------------------------------------------------------------

def bary(D, px, py):
    """barycentric weights of points (px, py) in triangles D (n, 9), both (n,)"""
    x1, y1, x2, y2, x3, y3 = D[:, 0], D[:, 1], D[:, 3], D[:, 4], D[:, 6], D[:, 7]
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
    l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
    return l1, l2, 1 - l1 - l2


def raster(D, x0, y0, w, h, step):
    """cells (iy * w + ix) whose centre lies in each triangle: (tri index, cell index, z at the centre)"""
    if len(D) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    tis, cells, zs = [], [], []
    ox, oy = .5171, .4829    # off the lattice diagonals
    xs, ys = D[:, [0, 3, 6]], D[:, [1, 4, 7]]
    ix0 = np.maximum(np.ceil((xs.min(1) - x0) / step - ox), 0).astype(np.int64)
    ix1 = np.minimum(np.floor((xs.max(1) - x0) / step - ox), w - 1).astype(np.int64)
    iy0 = np.maximum(np.ceil((ys.min(1) - y0) / step - oy), 0).astype(np.int64)
    iy1 = np.minimum(np.floor((ys.max(1) - y0) / step - oy), h - 1).astype(np.int64)
    ww, hh = ix1 - ix0 + 1, iy1 - iy0 + 1
    live = np.nonzero((ww > 0) & (hh > 0))[0]
    for t in live:
        gx, gy = np.meshgrid(np.arange(ww[t]), np.arange(hh[t]))
        gx, gy = (gx.ravel() + ix0[t]), (gy.ravel() + iy0[t])
        px, py = x0 + (gx + ox) * step, y0 + (gy + oy) * step
        d = D[t:t + 1].repeat(len(px), 0)
        l1, l2, l3 = bary(d, px, py)
        m = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if m.any():
            tis.append(np.full(int(m.sum()), t, np.int64))
            cells.append(gy[m] * w + gx[m])
            zs.append((l1 * D[t, 2] + l2 * D[t, 5] + l3 * D[t, 8])[m])
    if not tis:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(tis), np.concatenate(cells), np.concatenate(zs)


# ------------------------------------------------------------------------------------------------------------------
# the measures
# ------------------------------------------------------------------------------------------------------------------

def layers(tile, step=2.):
    """stack depth of the tile's cells (no drape, deck, sea) and the class pairs of the second layer.
    -> {'depth': {1: share, 2:, 3: (3 or more)}, 'pairs': {'a+b': share of cells}, 'road_cells': share}"""
    skip = [CI['drape'], CI['deck'], CI['sea']]
    keep = ~np.isin(tile.cls, skip)
    D, C = tile.D[keep], tile.cls[keep]
    x0, y0 = tile.rect[0], tile.rect[1]
    w = int(round(tile.size / step))
    ti, cell, _ = raster(D, x0, y0, w, w, step)
    n = w * w
    depth = np.bincount(cell, minlength=n)
    out = {'depth': {k: float((depth == k).sum()) / n for k in (1, 2)}, 'pairs': {}, 'cells': n}
    out['depth'][3] = float((depth >= 3).sum()) / n
    out['depth'][0] = float((depth == 0).sum()) / n
    # class sets per cell
    present = np.zeros((n, len(CLASSES)), bool)
    present[cell, C[ti]] = True
    roadish = present[:, [CI['road'], CI['sw_road'], CI['sw_plate'], CI['kerb'], CI['walk']]].any(1)
    out['road_cells'] = float(roadish.sum()) / n
    two = depth >= 2
    for a in range(len(CLASSES)):
        for b in range(a, len(CLASSES)):
            m = two & present[:, a] & present[:, b]
            if a == b:
                # two triangles of the same class in a cell
                cnt = np.bincount(cell[C[ti] == a], minlength=n)
                m = cnt >= 2
            if m.any():
                out['pairs'][CLASSES[a] + '+' + CLASSES[b]] = float(m.sum()) / n
    return out


def bridge_cells(tile, step=1.):
    """set of cell indices (step m) under a bridge deck's soffit: the ground there is meant to stay"""
    deck = tile.of('deck')
    if len(deck) == 0:
        return set()
    w = int(round(tile.size / step))
    _, cell, _ = raster(deck, tile.rect[0], tile.rect[1], w, w, step)
    return set(cell.tolist())


def under(tile, step=1.):
    """m2 of swept carriageway (not on a bridge) over a cell that has ground, and m2 of carriageway in all"""
    road = tile.of('sw_road')
    if len(road) == 0:
        return 0., 0.
    w = int(round(tile.size / step))
    _, rc, _ = raster(road, tile.rect[0], tile.rect[1], w, w, step)
    _, gc, _ = raster(tile.of('gnd'), tile.rect[0], tile.rect[1], w, w, step)
    rc = np.unique(rc)
    bridge = bridge_cells(tile, step)
    if bridge:
        rc = np.array([c for c in rc if c not in bridge], dtype=np.int64)
    hit = np.isin(rc, np.unique(gc))
    return float(hit.sum()) * step * step, float(len(rc)) * step * step


def road_on_road(tile, step=1.):
    """m2 of 1 m cells with two or more swept carriageway triangles over each other, by the height between the lowest and
    the highest: coplanar (under 1 cm: they z-fight), near (up to 1 m) and over (an overpass, by design)"""
    D = tile.of('sw_road')
    w = int(round(tile.size / step))
    ti, cell, z = raster(D, tile.rect[0], tile.rect[1], w, w, step)
    out = {'coplanar': 0., 'near': 0., 'over': 0.}
    if len(cell) == 0:
        return out
    order = np.argsort(cell, kind='stable')
    c2, z2 = cell[order], z[order]
    first = np.r_[0, np.nonzero(np.diff(c2))[0] + 1]
    cnt = np.diff(np.r_[first, len(c2)])
    for f, n in zip(first, cnt):
        if n >= 2:
            gap = z2[f:f + n].max() - z2[f:f + n].min()
            out['coplanar' if gap < .01 else 'near' if gap < 1. else 'over'] += step * step
    return out


def weld(D, q=TOL):
    """integer corner ids for triangles D (n, 9) and the unique corners (m, 3)"""
    P = np.round(D.reshape(-1, 3) / q).astype(np.int64)
    uniq, inv = np.unique(P, axis=0, return_inverse=True)
    return inv.reshape(-1, 3), uniq * q


def stitch(tile):
    """the swept carriageway's boundary corners (the ends of its free edges, in the tile, not under a bridge): welded to
    a ground corner within 1 mm in x, y and z ('to_ground'), or standing on another road's surface at its height
    ('to_road': a plate's corner, a slip road's edge on its host), or open.
    -> {'edge_corners': n, 'to_ground': n, 'to_road': n, 'open': n, 'open_at': [(x, y, z)...]}"""
    bridge = bridge_cells(tile)
    out = {'edge_corners': 0, 'to_ground': 0, 'to_road': 0, 'open': 0, 'open_at': []}
    road_idx = np.nonzero(tile.cls == CI['sw_road'])[0]
    if len(road_idx) == 0:
        return out
    road = tile.D[road_idx]
    ids, pts = weld(road)
    e = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    e.sort(1)
    key = e[:, 0] * len(pts) + e[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    corners = np.unique(e[cnt[inv] == 1].ravel())
    gset = {tuple(p) for p in np.round(weld(tile.of('gnd'))[1] / TOL).astype(np.int64)}
    other_idx = np.nonzero(np.isin(tile.cls, [CI['sw_road'], CI['sw_plate']]))[0]
    other = tile.D[other_idx]
    other_grp = tile.grp[other_idx]
    grid = Grid(other)
    # the mesh groups each corner of the carriageway belongs to
    owners = defaultdict(set)
    rg = tile.grp[road_idx]
    for t in range(len(road)):
        for c in ids[t]:
            owners[int(c)].add(int(rg[t]))
    xmin, ymin, xmax, ymax = tile.rect
    for c in corners:
        x, y, z = pts[c]
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        if (int(x - xmin) + int(y - ymin) * int(tile.size)) in bridge:
            continue
        out['edge_corners'] += 1
        key3 = (int(round(x / TOL)), int(round(y / TOL)), int(round(z / TOL)))
        if near_corner(gset, key3):
            out['to_ground'] += 1
            continue
        welded = len(owners[int(c)]) > 1      # a corner of two meshes is welded
        cand = list(grid.near(x, y)) if not welded else []
        if cand:
            d = other[cand]
            l1, l2, l3 = bary(d, np.full(len(cand), x), np.full(len(cand), y))
            inside = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
            for u in np.nonzero(inside)[0]:
                t = d[u]
                own = any(abs(t[3 * k] - x) < 2 * TOL and abs(t[3 * k + 1] - y) < 2 * TOL and abs(t[3 * k + 2] - z) < 2 * TOL for k in range(3))
                if own and other_grp[cand[u]] in owners[int(c)]:
                    continue       # one of the corner's own mesh's triangles
                zz = l1[u] * t[2] + l2[u] * t[5] + l3[u] * t[8]
                if abs(zz - z) <= TOL:
                    welded = True
                    break
        if welded:
            out['to_road'] += 1
        else:
            out['open'] += 1
            if len(out['open_at']) < 400:
                out['open_at'].append((float(x), float(y), float(z)))
    return out


def near_corner(cset, key):
    """a corner within one weld cell (1 mm) of key in x, y and z"""
    kx, ky, kz = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (kx + dx, ky + dy, kz + dz) in cset:
                    return True
    return False


class Grid:
    """triangles by 8 m bin, for point queries"""

    def __init__(self, D, cell=8.):
        self.D, self.cell = D, cell
        self.bins = defaultdict(list)
        xs, ys = D[:, [0, 3, 6]], D[:, [1, 4, 7]]
        x0, x1 = np.floor(xs.min(1) / cell).astype(int), np.floor(xs.max(1) / cell).astype(int)
        y0, y1 = np.floor(ys.min(1) / cell).astype(int), np.floor(ys.max(1) / cell).astype(int)
        for t in range(len(D)):
            for bx in range(x0[t], x1[t] + 1):
                for by in range(y0[t], y1[t] + 1):
                    self.bins[(bx, by)].append(t)

    def near(self, x, y):
        return self.bins.get((int(math.floor(x / self.cell)), int(math.floor(y / self.cell))), ())


def cracks(tile, tol=TOL, classes=LEVEL0):
    """boundary edges of the level-0 surface (ground, carriageway, plates) with nothing closing them. An edge is tried
    at its middle and near both ends with a probe 1.5 tol past it, away from its triangle; the probe is covered when a
    triangle other than the edge's own holds it (1e-4 slack) and that triangle's plane at the edge point is within tol
    in z. Not covered: a 'hole' when no triangle holds the probe (plan view: the car falls through), a 'step' when
    one does at another height (a cliff or a road standing on the ground). Edges on the tile border are the seam check's;
    the carriageway over a bridge deck is open by design.
    -> {'edges', 'hole', 'step', 'worst_step', 'found': [(x, y, z, kind, gap, class of the edge's triangle)...]}"""
    sel = np.nonzero(np.isin(tile.cls, [CI[c] for c in classes]))[0]
    D = tile.D[sel]
    out = {'edges': 0, 'hole': 0, 'step': 0, 'worst_step': 0., 'found': []}
    if len(D) == 0:
        return out
    # upright faces (a wall closing a cut, a bridge's abutment): no plan area, so nothing is 'beside' an edge in plan, but an
    # edge they share corner for corner is closed
    allD = tile.D[np.isin(tile.cls, [CI['deck'], CI['gnd']])]
    area = np.abs((allD[:, 3] - allD[:, 0]) * (allD[:, 7] - allD[:, 1]) - (allD[:, 6] - allD[:, 0]) * (allD[:, 4] - allD[:, 1]))
    W = allD[(area < 1e-8) & (np.abs(allD[:, [2, 5, 8]].max(1) - allD[:, [2, 5, 8]].min(1)) > .01)]
    ids_all, pts = weld(np.concatenate([D, W]) if len(W) else D)
    ids, wids = ids_all[:len(D)], ids_all[len(D):]
    walled = set()
    for t in wids:
        for a_, b_ in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            walled.add((min(a_, b_), max(a_, b_)))
    e = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    owner = np.tile(np.arange(len(D)), 3)
    srt = np.sort(e, axis=1)
    key = srt[:, 0] * len(pts) + srt[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    boundary = np.nonzero(cnt[inv] == 1)[0]
    boundary = np.array([b for b in boundary if (int(srt[b, 0]), int(srt[b, 1])) not in walled], dtype=np.int64)
    grid = Grid(D)
    bridge = bridge_cells(tile)
    xmin, ymin, xmax, ymax = tile.rect
    border = 5 * tol
    seam = hasattr(tile, 'alone')
    ntile = int(tile.size)
    for b in boundary:
        a, c = pts[e[b, 0]], pts[e[b, 1]]
        length = math.hypot(c[0] - a[0], c[1] - a[1])
        if length < 1e-6:
            continue
        if not seam and ((abs(a[0] - c[0]) < border and (abs(a[0] - xmin) < border or abs(a[0] - xmax) < border)) or
                         (abs(a[1] - c[1]) < border and (abs(a[1] - ymin) < border or abs(a[1] - ymax) < border))):
            continue
        t = owner[b]
        tri = D[t]
        cx, cy = (tri[0] + tri[3] + tri[6]) / 3, (tri[1] + tri[4] + tri[7]) / 3
        nx, ny = (c[1] - a[1]) / length, -(c[0] - a[0]) / length
        if (cx - a[0]) * nx + (cy - a[1]) * ny > 0:
            nx, ny = -nx, -ny
        out['edges'] += 1
        bad = None
        for f in (.5, .02, .98):
            px, py, pz = a[0] + (c[0] - a[0]) * f, a[1] + (c[1] - a[1]) * f, a[2] + (c[2] - a[2]) * f
            lo = -border if seam else border
            if not (xmin + lo < px < xmax - lo and ymin + lo < py < ymax - lo):
                continue      # past the tile (a chunk is built whole by its owner), or on its border when alone
            if (int(px - xmin) + int(py - ymin) * ntile) in bridge:
                continue
            qx, qy = px + nx * tol * 1.5, py + ny * tol * 1.5
            cand = [u for u in grid.near(qx, qy) if u != t]
            kind, gap, other = 'hole', None, None
            if cand:
                d = D[cand]
                l1, l2, l3 = bary(d, np.full(len(cand), qx), np.full(len(cand), qy))
                inside = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
                if inside.any():
                    ll1, ll2, ll3 = bary(d[inside], np.full(int(inside.sum()), px), np.full(int(inside.sum()), py))
                    zz = ll1 * d[inside, 2] + ll2 * d[inside, 5] + ll3 * d[inside, 8]
                    near = int(np.argmin(np.abs(zz - pz)))
                    gap = float(abs(zz[near] - pz))
                    other = CLASSES[tile.cls[sel[np.array(cand)[inside][near]]]]
                    kind = 'closed' if gap <= tol else 'step'
            if kind != 'closed':
                bad = (px, py, pz, kind, gap, other)
                break
        if bad:
            out[bad[3]] += 1
            if bad[3] == 'step':
                out['worst_step'] = max(out['worst_step'], bad[4])
            out['found'].append(bad + (CLASSES[tile.cls[sel[t]]],))   # ... , class beside it, class of the edge's triangle
    return out


BRIDGE_F = re.compile(r'_sw_\d*f$')


def upright_walls(tile, fences=False):
    """the upright concrete faces of the tile (material fence_wall, no plan area): the end walls of conformal bands and the
    wing walls of abutments; a bridge's own parapets and piers (the sweep's 'f' meshes) only when `fences`.
    -> (n, 9) triangles"""
    sel = [gi for gi, g in enumerate(tile.groups) if g[2] == 'fence_wall' and (fences or not BRIDGE_F.search(g[1]))]
    D = tile.D[(tile.cls == CI['deck']) & np.isin(tile.grp, sel)]
    if len(D) == 0:
        return D
    area = np.abs((D[:, 3] - D[:, 0]) * (D[:, 7] - D[:, 1]) - (D[:, 6] - D[:, 0]) * (D[:, 4] - D[:, 1]))
    zr = D[:, [2, 5, 8]].max(1) - D[:, [2, 5, 8]].min(1)
    return D[(area < 1e-8) & (zr > .01)]


def walls_in_road(tile, step=.25, slack=.02):
    """upright walls (end walls and wing walls) that stand inside a swept carriageway or plate: a sample of a wall's plan line
    (every `step` m) strictly inside a carriageway triangle (barycentric weights over 1e-4: a wall on the road's edge is
    not inside) where the wall's height range at that point holds the road's height by more than `slack` at both ends (a wall that
    only touches the road's underside, its top flush with a carriageway that goes on over it, is under the road, not in it).
    Walls over or under a road (a bridge's wall above the road below) do not count. Zero tolerance: any hit is a wall standing in the road.
    -> {'walls': n checked, 'hits': n, 'found': [(x, y, z wall low, z wall high, z road)...]}"""
    W = upright_walls(tile)
    R = tile.D[np.isin(tile.cls, [CI['sw_road'], CI['sw_plate']])]
    out = {'walls': len(W), 'hits': 0, 'found': []}
    if len(W) == 0 or len(R) == 0:
        return out
    grid = Grid(R)
    for w in W:
        P = w.reshape(3, 3)
        # the plan line: the two corners furthest apart
        pairs = [(0, 1), (1, 2), (0, 2)]
        a, b = max(pairs, key=lambda ij: math.hypot(*(P[ij[0], :2] - P[ij[1], :2])))
        L = math.hypot(*(P[b, :2] - P[a, :2]))
        if L < 1e-6:
            continue
        ux, uy = (P[b, 0] - P[a, 0]) / L, (P[b, 1] - P[a, 1]) / L
        # the triangle in (t along the line, z): corners and the vertical line at t
        T = np.array([[(p[0] - P[a, 0]) * ux + (p[1] - P[a, 1]) * uy, p[2]] for p in P])
        for t in np.linspace(0, L, max(2, int(L / step) + 1)):
            zs = []
            for i in range(3):
                (t0, z0), (t1, z1) = T[i], T[(i + 1) % 3]
                if abs(t1 - t0) < 1e-12:
                    if abs(t - t0) < 1e-9:
                        zs += [z0, z1]
                elif min(t0, t1) - 1e-9 <= t <= max(t0, t1) + 1e-9:
                    zs.append(z0 + (z1 - z0) * (t - t0) / (t1 - t0))
            if not zs:
                continue
            lo, hi = min(zs), max(zs)
            x, y = P[a, 0] + ux * t, P[a, 1] + uy * t
            cand = list(grid.near(x, y))
            if not cand:
                continue
            d = R[cand]
            l1, l2, l3 = bary(d, np.full(len(cand), x), np.full(len(cand), y))
            for u in np.nonzero((l1 > 1e-4) & (l2 > 1e-4) & (l3 > 1e-4))[0]:
                zr = l1[u] * d[u, 2] + l2[u] * d[u, 5] + l3[u] * d[u, 8]
                if lo + slack < zr < hi - slack:
                    out['hits'] += 1
                    out['found'].append((float(x), float(y), float(lo), float(hi), float(zr)))
                    break
    return out


def _ray_hit(orig, dirs, T):
    """Moeller-Trumbore: does the ray orig + t dir (0 < t < 1, dir not normalised) hit any triangle of T (n, 9)? dirs (m, 3) -> (m,) bool"""
    A, E1, E2 = T[:, 0:3], T[:, 3:6] - T[:, 0:3], T[:, 6:9] - T[:, 0:3]
    hit = np.zeros(len(dirs), bool)
    for i, d in enumerate(dirs):
        P = np.cross(d, E2)
        det = (E1 * P).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        Tv = orig - A
        u = (Tv * P).sum(1) * inv
        Q = np.cross(Tv, E1)
        v = (Q * d).sum(1) * inv
        t = (E2 * Q).sum(1) * inv
        hit[i] = bool((ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6) & (t < 1)).any())
    return hit


def see_through(tile, spacing=12., eye=1.2, reach=45., pitches=(-1., -4., -9.), yaws=16, area=None, limit=None):
    """Rays that leave the surface unhit and end under it. From points along the swept carriageway (a point every `spacing` m, not
    under a bridge deck), `eye` m over the road, rays go out sideways in `yaws` directions at each pitch (degrees, below the
    horizontal) for `reach` m. A ray that hits no triangle of the tile (ground, road, deck, wall) and ends below the top of the
    surface at its end point passed through a gap in it: a ray that ends above the surface is in open air. This is the view
    through the world that cracks() and open_edges() cannot give: a pocket between two layers of surface (an embankment over the
    cut of a road under it) or an open end of a band has no plan hole and may have no edge that is open in 3D by more than
    centimetres, yet shows the sky or the far trees.
    -> {'rays': n, 'leaks': n, 'found': [(x, y, z, yaw, pitch, end x, end y, end z, surface z at the end)...]}"""
    xmin, ymin, xmax, ymax = area or tile.rect
    surf = tile.D[np.isin(tile.cls, [CI[c] for c in ('gnd', 'gwat', 'road', 'sw_road', 'sw_plate', 'kerb', 'walk', 'island')])]
    allD = tile.D[~np.isin(tile.cls, [CI['drape'], CI['sea']])]
    grid = Grid(allD, cell=4.)
    top = Grid(surf, cell=4.)
    road = tile.of('sw_road')
    bridge = bridge_cells(tile)
    seen, pts = set(), []
    for t in road:
        cx, cy = (t[0] + t[3] + t[6]) / 3, (t[1] + t[4] + t[7]) / 3
        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
            continue
        key = (int(cx // spacing), int(cy // spacing))
        if key in seen or (int(cx - tile.rect[0]) + int(cy - tile.rect[1]) * int(tile.size)) in bridge:
            continue
        seen.add(key)
        pts.append(((t[2] + t[5] + t[8]) / 3, cx, cy))
    pts.sort(key=lambda p: (round(p[1] / spacing), p[2]))
    if limit:
        pts = pts[::max(1, len(pts) // limit)]
    out = {'rays': 0, 'leaks': 0, 'found': []}
    for z, cx, cy in pts:
        orig = np.array([cx, cy, z + eye])
        for k in range(yaws):
            yaw = 2 * math.pi * k / yaws
            for pitch in pitches:
                pr = math.radians(pitch)
                d = np.array([math.cos(yaw) * math.cos(pr), math.sin(yaw) * math.cos(pr), math.sin(pr)]) * reach
                # the triangles along the ray: the bins its samples fall in
                cand = set()
                for f in np.linspace(0, 1, int(reach / 3) + 1):
                    q = orig + d * f
                    cand.update(grid.near(q[0], q[1]))
                    for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                        cand.update(grid.near(q[0] + dx, q[1] + dy))
                out['rays'] += 1
                if cand and _ray_hit(orig, d[None], allD[sorted(cand)])[0]:
                    continue
                e = orig + d
                if not (xmin - 60 < e[0] < xmax + 60 and ymin - 60 < e[1] < ymax + 60):
                    continue
                if not (tile.rect[0] <= e[0] < tile.rect[2] and tile.rect[1] <= e[1] < tile.rect[3]) or                         (int(e[0] - tile.rect[0]) + int(e[1] - tile.rect[1]) * int(tile.size)) in bridge:
                    continue     # past the tile, or under a deck (the road on it is no surface the ray can end under)
                cand = list(top.near(e[0], e[1]))
                if not cand:
                    continue
                D = surf[cand]
                l1, l2, l3 = bary(D, np.full(len(cand), e[0]), np.full(len(cand), e[1]))
                ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
                if not ins.any():
                    continue
                zs = (l1 * D[:, 2] + l2 * D[:, 5] + l3 * D[:, 8])[ins]
                if e[2] < zs.max() - .1:
                    out['leaks'] += 1
                    out['found'].append((float(cx), float(cy), float(z), math.degrees(yaw), pitch, float(e[0]), float(e[1]), float(e[2]), float(zs.max())))
    return out


def zones(tile, near=40., exempt=6., step=2.):
    """boolean cell grids (step m): the corridor of the swept carriageway (within `near` m of it) and the
    exempt cells (within `exempt` m of anything that stays on top of the ground in phase 1: generator roads, plates,
    kerbs, sidewalks, islands, bridge decks)"""
    w = int(round(tile.size / step))

    def mask(classes):
        D = tile.D[np.isin(tile.cls, [CI[c] for c in classes])]
        m = np.zeros(w * w, bool)
        _, cell, _ = raster(D, tile.rect[0], tile.rect[1], w, w, step)
        m[cell] = True
        return m.reshape(w, w)

    def grow(m, r):
        n = int(math.ceil(r / step))
        out = m.copy()
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                if dx * dx + dy * dy <= n * n:
                    out |= np.roll(np.roll(m, dy, 0), dx, 1)
        return out
    return (grow(mask(['sw_road']), near),
            grow(mask(['road', 'sw_plate', 'kerb', 'walk', 'island', 'deck']), exempt), w, step)


def scope(tile, found, zs=None):
    """split cracks()' findings: steps in the corridor of the swept road away from the structures that stay on top of
    the ground ('corridor': the gate) and the rest ('elsewhere': generator roads on the lattice, bridge ends, plates)"""
    corridor, exempt, w, step = zs or zones(tile)
    out = {'corridor_step': 0, 'elsewhere_step': 0, 'hole': 0, 'corridor_at': [], 'hole_at': [], 'pairs': defaultdict(int),
           'small': 0, 'large': 0, 'seam_hole': 0, 'seam_step': 0}
    xmin, ymin, xmax, ymax = tile.rect
    for x, y, z, kind, gap, other, cls in found:
        on_border = min(abs(x - xmin), abs(x - xmax), abs(y - ymin), abs(y - ymax)) < 5 * TOL
        if on_border:
            out['seam_hole' if kind == 'hole' else 'seam_step'] += 1
        ix, iy = int((x - tile.rect[0]) / step), int((y - tile.rect[1]) / step)
        inside = 0 <= ix < w and 0 <= iy < w
        if kind == 'hole':
            out['hole'] += 1
            out['hole_at'].append((x, y, z, cls))
            continue
        if inside and corridor[iy, ix] and not exempt[iy, ix]:
            out['corridor_step'] += 1
            out['corridor_at'].append((x, y, z, gap, cls, other))
            out['pairs'][cls + '>' + other] += 1
            out['small' if gap <= .02 else 'large'] += 1
        else:
            out['elsewhere_step'] += 1
    return out


def measure(tile, leaks=False):
    """all the per-tile numbers (leaks: the rays too, 15 s a tile)"""
    u, tot = under(tile)
    lay = layers(tile)
    st = stitch(tile)
    cr = cracks(tile)
    sc = scope(tile, cr['found'])
    return {'tile': (tile.i, tile.j), 'build_s': tile.seconds, 'tris': getattr(tile, 'alone', tile).counts(), 'layers': lay,
            'under_m2': u, 'carriageway_m2': tot, 'road_on_road': road_on_road(tile), 'stitch': st, 'cracks': cr, 'scope': sc,
            'walls': walls_in_road(tile), 'leaks': see_through(tile, spacing=8.) if leaks else None}


def budget(tile, pieces, tris, solid, gap):
    """the load a tile puts on the game, whatever it looks like (a game retest: 3 of 4 ON runs had 55-120 ms frames, 0 of 3 OFF):
      tris        triangles of everything the tile draws; pieces: its meshes (one ProceduralMesh each, the game's object count)
      solid       triangles of the objects that join the static collision (the rebuild costs 0.27 ms per 1000, the physics sweeps them)
      wall_tris, wall_area, wall_height   upright concrete (end walls, faces between stacked bands, wing walls; not a bridge's parapets):
                  triangles, plan-free area, the tallest triangle
      degenerate  triangles under a square millimetre; thin: area under 1 % of the longest edge squared; long: an edge over 50 m;
                  dup: the same triangle twice (z-fight, double work for the physics)
      gap_ms      the longest stretch of the build between two pause calls, in CPU ms: the game cannot split it, so it is a frame
    -> dict"""
    D = tile.D
    e1, e2, e3 = D[:, 3:6] - D[:, 0:3], D[:, 6:9] - D[:, 0:3], D[:, 6:9] - D[:, 3:6]
    area = .5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    longest = np.maximum.reduce([np.linalg.norm(e, axis=1) for e in (e1, e2, e3)])
    corners = np.round(D * 1000).astype(np.int64).reshape(-1, 3, 3)
    seen, dup = set(), 0
    for t, a in zip(corners, area):
        key = tuple(sorted(map(tuple, t)))
        dup += (key in seen) and a >= 1e-6       # (a triangle with no area drawn twice is no z-fight)
        seen.add(key)
    W = upright_walls(tile)
    wa = .5 * np.linalg.norm(np.cross(W[:, 3:6] - W[:, 0:3], W[:, 6:9] - W[:, 0:3]), axis=1) if len(W) else np.zeros(0)
    wh = (W[:, [2, 5, 8]].max(1) - W[:, [2, 5, 8]].min(1)) if len(W) else np.zeros(0)
    return {'tile': (tile.i, tile.j), 'tris': int(tris), 'surface_tris': len(D), 'pieces': int(pieces), 'solid': int(solid),
            'wall_tris': len(W), 'wall_area': float(wa.sum()), 'wall_height': float(wh.max()) if len(wh) else 0.,
            'degenerate': int((area < 1e-6).sum()), 'thin': int(((area >= 1e-6) & (area < .01 * longest ** 2)).sum()),
            'long': int((longest > 50.).sum()), 'dup': int(dup), 'gap_ms': 1000. * gap}


# what ON may cost over OFF, per metric: the limit as a function of the OFF value and of ON's triangle count. The factors are
# generous on purpose (ON cuts the ground into pieces and builds walls: some growth is the design); they are meant to catch an
# explosion, and a stretch of work that cannot be split
BUDGET = {
    'tris': lambda off, n: 1.25 * off + 300,
    'pieces': lambda off, n: 1.5 * off + 4,
    'solid': lambda off, n: 1.25 * off + 300,
    'wall_tris': lambda off, n: 2. * off + 1500,
    'wall_area': lambda off, n: 2. * off + 2500.,
    'wall_height': lambda off, n: off + 20.,
    'degenerate': lambda off, n: off + .01 * n,
    'thin': lambda off, n: 10. * off + 600,      # (the toe of a band is cut at every lattice line: slivers by design, harmless)
    'long': lambda off, n: 1.5 * off + 20,
    'dup': lambda off, n: 1.5 * off + 60,
    'gap_ms': lambda off, n: max(40., 1.5 * off),
}


def check_budget(on, off):
    """the metrics where ON is over its limit against OFF (budget() of the same tile): [(metric, on, limit, off)]"""
    bad = []
    for key, limit in BUDGET.items():
        lim = limit(off[key], on['tris'])
        if on[key] > lim:
            bad.append((key, on[key], lim, off[key]))
    return bad


def fmt(m):
    t = m['tris']
    lay = m['layers']
    st = m['stitch']
    cr = m['cracks']
    sc = m['scope']
    pairs = ' '.join(f"{k} {100 * v:.2f}" for k, v in sorted(lay['pairs'].items(), key=lambda kv: -kv[1])[:6])
    rows = [f"tile {m['tile']}  build {m['build_s']:.2f} s",
            '  tris   ' + ' '.join(f"{k} {v}" for k, v in t.items() if v) + f"  total {sum(t.values())}",
            f"  layers depth 1 {100 * lay['depth'][1]:.2f} %  2 {100 * lay['depth'][2]:.2f} %  3+ {100 * lay['depth'][3]:.2f} %  "
            f"(uncovered {100 * lay['depth'][0]:.2f} %)  road-family cells {100 * lay['road_cells']:.2f} %  second layers: {pairs}",
            f"  under  carriageway {m['carriageway_m2']:.0f} m2, with ground in the cell {m['under_m2']:.0f} m2; road over road: "
            f"coplanar {m['road_on_road']['coplanar']:.0f} m2, within 1 m {m['road_on_road']['near']:.0f} m2, overpass {m['road_on_road']['over']:.0f} m2",
            f"  stitch edge corners {st['edge_corners']}: to ground {st['to_ground']}, to road {st['to_road']}, open {st['open']}",
            f"  cracks boundary edges {cr['edges']}: holes {sc['hole']}, steps in the road corridor {sc['corridor_step']} "
            f"({sc['small']} up to 2 cm, {sc['large']} more), steps elsewhere {sc['elsewhere_step']} (worst step {cr['worst_step']:.3f} m); "
            f"on the tile border: {sc['seam_hole']} holes, {sc['seam_step']} steps"]
    rows.append(f"  walls  upright walls {m['walls']['walls']}, inside a carriageway {m['walls']['hits']}"
                + (f"; rays {m['leaks']['rays']}, through the surface {m['leaks']['leaks']}" if m.get('leaks') else ''))
    if sc['pairs']:
        rows.append('  steps by class (edge > beside it): ' + ', '.join(f'{k} {v}' for k, v in sorted(sc['pairs'].items(), key=lambda kv: -kv[1])[:6]))
    return '\n'.join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pack', required=True, help='staged pack directory (the one that holds earthng/countries)')
    ap.add_argument('--tiles', required=True, help='"i,j;i,j"')
    ap.add_argument('--switch', default='both', choices=('off', 'on', 'both'))
    ap.add_argument('--mode', default='fast')
    ap.add_argument('--alone', action='store_true', help='the tile alone, not with its eight neighbours loaded (a chunk is drawn whole by one tile)')
    ap.add_argument('--leaks', action='store_true', help='cast the rays that look for gaps in the surface too (15 s a tile)')
    ap.add_argument('--budget', action='store_true', help='only the load on the game (tris, pieces, collision, walls, longest stretch), ON against OFF, tiles alone')
    ap.add_argument('-v', action='store_true', help='list the first open cracks (tile-local metres)')
    a = ap.parse_args(argv)
    if lupa is None:
        sys.exit('lupa.luajit21 is not installed')
    rt = Runtime(a.pack)
    tiles = [tuple(int(v) for v in q.split(',')) for q in a.tiles.split(';') if q.strip()]
    if a.budget:
        worst = 0
        for i, j in tiles:
            off, on = rt.budget(i, j, False, mode=a.mode), rt.budget(i, j, True, mode=a.mode)
            bad = check_budget(on, off)
            worst += len(bad)
            print(f'tile {(i, j)}  ' + '  '.join(f"{k} {off[k]:.0f}>{on[k]:.0f}" for k in ('tris', 'pieces', 'solid', 'wall_tris', 'wall_area', 'wall_height',
                                                                                  'degenerate', 'thin', 'dup', 'gap_ms')) + ('  OVER: ' + ', '.join(f'{k} {v:.0f} > {l:.0f}' for k, v, l, _ in bad) if bad else '  ok'))
        sys.exit(1 if worst else 0)
    for on in {'off': [False], 'on': [True], 'both': [False, True]}[a.switch]:
        print(f"=== conformal ground {'ON' if on else 'OFF'}" + ('' if rt.has_switch else '  (this generator has no switch)'))
        for i, j in tiles:
            tile = rt.build(i, j, conformal=on, mode=a.mode) if a.alone else rt.block(i, j, conformal=on, mode=a.mode)
            m = measure(tile, a.leaks)
            print(fmt(m))
            if a.v:
                for p in m['scope']['hole_at'][:10]:
                    print('    hole at x %.2f y %.2f z %.3f (%s)' % (p[0] - tile.rect[0], p[1] - tile.rect[1], p[2], p[3]))
                for p in m['scope']['corridor_at'][:10]:
                    print('    step at x %.2f y %.2f z %.3f gap %.3f (%s beside %s)' % (p[0] - tile.rect[0], p[1] - tile.rect[1], p[2], p[3], p[4], p[5]))
                for p in m['stitch']['open_at'][:10]:
                    print('    open edge corner at x %.2f y %.2f z %.3f' % (p[0] - tile.rect[0], p[1] - tile.rect[1], p[2]))


if __name__ == '__main__':
    main()
