-- Pack v3 road blocks (earthng-roads-v3, EGR3): the exporter's strokes, chunks and junction records
-- (patches/ROAD-OBJECTS.md section 7; reference encoder and decoder earthng/fast_roads.py). The block
-- layout is cellformat's EGB2 with its own magic; numbers are LEB128 varints, signed ones zigzagged,
-- lengths and positions in cm (x, y from the cell's south-west corner), heading 1e-5 rad, curvature
-- 1e-6 / m, cross-fall 1e-4. Every station field is delta-coded within its chunk, so a chunk's last
-- station and the next chunk's first decode to the same doubles in every tile.
-- Version 1: as staged; version 2: the junction records carry the exporter's fan plates; version 3 (bundle5): the
-- same layout, the exporter's added lanes and lane tapers S-shaped and OSM-mapped auxiliary lanes used as the exits'
-- and entries' added lanes (sweep.lua draws the S-shape with every version).
local M = {}
local byte, sub, floor = string.byte, string.sub, math.floor
M.BLOCK = 8
M.FORMAT = 'earthng-roads-v3'
M.VERSIONS = {[1] = true, [2] = true, [3] = true}
local STATION = {'s', 'x', 'y', 'z', 'h', 'k', 'eL', 'eR', 'fl', 'fr'}
local SCALES = {100, 100, 100, 100, 1e5, 1e6, 100, 100, 1e4, 1e4}
M.STATION = STATION

-- Byte ranges {slot = {first, last}} of the cells in a road block file (1-based, inclusive).
function M.block(data)
  if type(data) ~= 'string' or sub(data, 1, 4) ~= 'EGR3' then error('not an EarthNG road block') end
  local pos, size = 5, #data
  local function u()
    local value, mul = 0, 1
    while true do
      if pos > size or mul > 2 ^ 49 then error('truncated road block') end
      local b = byte(data, pos)
      pos = pos + 1
      if b < 128 then return value + b * mul end
      value, mul = value + (b - 128) * mul, mul * 128
    end
  end
  local count, entries = u(), {}
  if count > M.BLOCK * M.BLOCK then error('too many cells in a road block') end
  for k = 1, count do entries[k] = {u(), u()} end
  local out, at = {}, pos
  for _, e in ipairs(entries) do
    if e[1] >= M.BLOCK * M.BLOCK or out[e[1]] then error('invalid road block slot') end
    out[e[1]] = {at, at + e[2] - 1}
    at = at + e[2]
  end
  if at ~= size + 1 then error('road block size mismatch') end
  return out
end

-- One cell from data[first..last]; ox, oy: its south-west corner (region m); version: the manifest's
-- roads.version. Returns {chunks = {...}, nodes = {...}} (field names: see the end of this file).
function M.cell(data, first, last, ox, oy, version, pause)
  pause = pause or function() end
  version = version or 1
  local pos = first
  local function u()
    local value, mul = 0, 1
    while true do
      if pos > last or mul > 2 ^ 49 then error('truncated road cell') end
      local b = byte(data, pos)
      pos = pos + 1
      if b < 128 then return value + b * mul end
      value, mul = value + (b - 128) * mul, mul * 128
    end
  end
  local function s()
    local v = u()
    if v % 2 == 0 then return v / 2 end
    return -(v + 1) / 2
  end
  local strings = {}
  for k = 1, u() do
    local n = u()
    if pos + n - 1 > last then error('truncated road string') end
    strings[k] = sub(data, pos, pos + n - 1)
    pos = pos + n
  end
  local function ref()
    local v = strings[u() + 1]
    if v == nil then error('invalid road string reference') end
    return v
  end
  -- the cell origin in cm, as the encoder rounds it (fast_roads._q)
  local ocx, ocy = floor(ox * 100 + .5), floor(oy * 100 + .5)
  local chunks, nodes = {}, {}
  for c = 1, u() do
    local ch = {stroke = ref(), index = u(), chunks = u()}
    local flags = u()
    ch.oneway, ch.closed, ch.fitted = flags % 2 == 1, floor(flags / 2) % 2 == 1, floor(flags / 4) % 2 == 1
    ch.highway = ref()
    ch.length = u() / 100
    ch.d0, ch.d1 = s() / 100, s() / 100
    local n = u()
    if n < 2 then error('road chunk with fewer than two stations') end
    local cols = {}
    for f = 1, 10 do cols[f] = {} end
    local prev = {0, ocx, ocy, 0, 0, 0, 0, 0, 0, 0}
    for i = 1, n do
      for f = 1, 10 do
        local v = prev[f] + s()
        prev[f] = v
        cols[f][i] = v / SCALES[f]
      end
    end
    for f = 1, 10 do ch[STATION[f]] = cols[f] end
    ch.n = n
    local parts = {}
    for p = 1, u() do
      local part = {way = ref(), segment = u(), reversed = u() == 1}
      part.s0, part.s1 = s() / 100, s() / 100
      local tags = {}
      for _ = 1, u() do local key = ref(); tags[key] = ref() end
      part.tags = tags
      parts[p] = part
    end
    ch.parts = parts
    local lanes = {}
    for l = 1, u() do
      local r = {s0 = s() / 100, s1 = s() / 100, f = u(), b = u(), both = u()}
      r.width = u() / 100
      r.tf, r.tb = ref(), ref()
      lanes[l] = r
    end
    ch.lanes = lanes
    local events = {}
    for e = 1, u() do
      local ev = {kind = ref(), side = s()}
      ev.taper, ev.parallel, ev.separate, ev.nose, ev.nodeS = s() / 100, s() / 100, s() / 100, s() / 100, s() / 100
      ev.width = u() / 100
      ev.branch = ref()
      ev.bd0, ev.bd1 = s() / 100, s() / 100
      events[e] = ev
    end
    ch.events = events
    local furniture = {}
    for f = 1, u() do
      local fu = {kind = ref(), side = s()}
      fu.s0, fu.s1, fu.offset = s() / 100, s() / 100, s() / 100
      fu.way = ref()
      fu.onWay = u() == 1
      furniture[f] = fu
    end
    ch.furniture = furniture
    local structures = {}
    for r = 1, u() do
      local st = {kind = ref()}
      st.s0, st.s1 = s() / 100, s() / 100
      st.layer = s()
      st.structure, st.rule = ref(), ref()
      st.clearance, st.water, st.floor = s() / 100, s() / 100, s() / 100
      structures[r] = st
    end
    ch.structures = structures
    local attach = {}
    for a = 1, u() do
      local at = {kind = ref(), host = ref(), node = ref()}
      at.hostS, at.side = s() / 100, s()
      at.lane0, at.lane1, at.hostLane0, at.hostLane1 = s() / 100, s() / 100, s() / 100, s() / 100
      at.offset = s() / 100
      attach[a] = at
    end
    ch.attach = attach
    local cn = {}
    for k = 1, u() do
      local node = ref()
      cn[k] = {node = node, s = s() / 100}
    end
    ch.nodes = cn
    chunks[c] = ch
    if c % 16 == 0 then pause() end
  end
  for k = 1, u() do
    local nd = {node = ref(), kind = ref()}
    nd.x, nd.y, nd.z = (s() + ocx) / 100, (s() + ocy) / 100, s() / 100
    local mouths = {}
    for m = 1, u() do
      local sid = ref()
      mouths[m] = {stroke = sid, s = s() / 100, dir = s()}
    end
    nd.mouths = mouths
    local roles = {}
    for _ = 1, u() do
      local sid = ref()
      roles[sid] = {role = ref(), host = ref()}
    end
    nd.roles = roles
    local plate = {}
    if version >= 2 then
      for p = 1, u() do
        local row = {stroke = ref(), dir = s(), cut = s() / 100}
        local pts = {}
        for q = 1, 3 do pts[q] = {(s() + ocx) / 100, (s() + ocy) / 100, s() / 100} end
        row.L, row.C, row.R = pts[1], pts[2], pts[3]
        plate[p] = row
      end
    end
    nd.plate = plate
    nodes[k] = nd
    if k % 64 == 0 then pause() end
  end
  if pos ~= last + 1 then error('trailing bytes after road cell') end
  return {chunks = chunks, nodes = nodes}
end

-- Junction blocks (earthng-junctions-v1, EGJ1; phase 3, ROAD-OBJECTS.md section 9; reference coder
-- earthng/fast_junctions.py): the same block layout, its own magic and files (manifest roads.junctions).
M.JUNCTIONS = 'earthng-junctions-v1'
function M.junctionBlock(data)
  if type(data) ~= 'string' or sub(data, 1, 4) ~= 'EGJ1' then error('not an EarthNG junction block') end
  return M.block('EGR3' .. sub(data, 5))
end

-- The junctions of one cell (data[first..last]): {id, kind, signals, x, y, z, nodes, mouths {stroke, node,
-- dir, s, cut, role, L, C, R}, internal {stroke, s0, s1}, verts {{x, y, z}}, tris {{a, b, c}} (1-based),
-- kerbs {{a, b}} (1-based, the surface on their left), islands {kind, height, ring}, approaches {stroke,
-- dir, stop, control, turns, lanes, crossings {s, width, kind, red}}}.
function M.junctionCell(data, first, last, ox, oy, pause)
  pause = pause or function() end
  local pos = first
  local function u()
    local value, mul = 0, 1
    while true do
      if pos > last or mul > 2 ^ 49 then error('truncated junction cell') end
      local b = byte(data, pos)
      pos = pos + 1
      if b < 128 then return value + b * mul end
      value, mul = value + (b - 128) * mul, mul * 128
    end
  end
  local function s()
    local v = u()
    if v % 2 == 0 then return v / 2 end
    return -(v + 1) / 2
  end
  local strings = {}
  for k = 1, u() do
    local n = u()
    if pos + n - 1 > last then error('truncated junction string') end
    strings[k] = sub(data, pos, pos + n - 1)
    pos = pos + n
  end
  local function ref()
    local v = strings[u() + 1]
    if v == nil then error('invalid junction string reference') end
    return v
  end
  local ocx, ocy = floor(ox * 100 + .5), floor(oy * 100 + .5)
  local function point() return {(s() + ocx) / 100, (s() + ocy) / 100, s() / 100} end
  local out = {}
  for k = 1, u() do
    local j = {id = ref(), kind = ref(), signals = u() == 1}
    local c = point()
    j.x, j.y, j.z = c[1], c[2], c[3]
    j.nodes = {}
    for n = 1, u() do j.nodes[n] = ref() end
    j.mouths = {}
    for n = 1, u() do
      local m = {stroke = ref(), node = ref(), dir = s()}
      m.s, m.cut = s() / 100, s() / 100
      m.role = ref()
      m.L, m.C, m.R = point(), point(), point()
      j.mouths[n] = m
    end
    j.internal = {}
    for n = 1, u() do
      local sid = ref()
      j.internal[n] = {stroke = sid, s0 = s() / 100, s1 = s() / 100}
    end
    j.verts = {}
    local px, py, pz = ocx, ocy, 0
    for n = 1, u() do
      px, py, pz = px + s(), py + s(), pz + s()
      j.verts[n] = {px / 100, py / 100, pz / 100}
    end
    j.tris = {}
    local pa = 0
    for n = 1, u() do
      local a = pa + s()
      local b, c3 = a + s(), a + s()
      pa = a
      j.tris[n] = {a + 1, b + 1, c3 + 1}
    end
    j.kerbs = {}
    pa = 0
    for n = 1, u() do
      local a = pa + s()
      local b = a + s()
      pa = a
      j.kerbs[n] = {a + 1, b + 1}
    end
    j.islands = {}
    for n = 1, u() do
      local isl = {kind = ref(), height = u() / 100, ring = {}}
      for q = 1, u() do isl.ring[q] = point() end
      j.islands[n] = isl
    end
    j.approaches = {}
    for n = 1, u() do
      local a = {stroke = ref(), dir = s()}
      a.stop = s() / 100
      a.control, a.turns, a.lanes = ref(), ref(), u()
      a.crossings = {}
      for q = 1, u() do
        local cs = {s = s() / 100, width = u() / 100}
        cs.kind, cs.red = ref(), u() == 1
        a.crossings[q] = cs
      end
      j.approaches[n] = a
    end
    out[k] = j
    if k % 32 == 0 then pause() end
  end
  if pos ~= last + 1 then error('trailing bytes after junction cell') end
  return out
end

-- The manifest's junction section (inside roads), checked: nil when absent or of another format.
function M.junctionSection(roads)
  local j = type(roads) == 'table' and roads.junctions
  if type(j) ~= 'table' or j.format ~= M.JUNCTIONS or j.version ~= 1 or type(j.blocks) ~= 'table' then return nil end
  return j
end

-- Is slot k (0..63) present in a 16-hex-digit block mask? Digit d holds slots 4d..4d+3.
function M.present(mask, k)
  local digit = tonumber(sub(mask, floor(k / 4) + 1, floor(k / 4) + 1), 16)
  return digit ~= nil and floor(digit / 2 ^ (k % 4)) % 2 == 1
end

-- The manifest's roads section, checked: nil when absent or of another format or version.
function M.section(manifest)
  local r = type(manifest) == 'table' and manifest.roads
  if type(r) ~= 'table' or r.format ~= M.FORMAT or not M.VERSIONS[r.version] or type(r.blocks) ~= 'table'
    or r.blockCells ~= M.BLOCK or type(r.cellSize) ~= 'number' or r.cellSize <= 0 then return nil end
  return r
end

-- Chunks and junction records of several decoded cells as one set: a chunk is stored once (in the cell
-- of its middle station), a record once (the cell of its node), so nothing repeats; keyed for lookups.
function M.merge(cells, junctionCells)
  local out = {chunks = {}, nodes = {}, byStroke = {}, byNode = {}}
  -- junctions (phase 3): stored once, in the cell of their centre
  if junctionCells then
    out.junctions, out.byJunction = {}, {}
    for _, list in ipairs(junctionCells) do
      for _, j in ipairs(list) do
        if not out.byJunction[j.id] then
          out.byJunction[j.id] = j
          out.junctions[#out.junctions + 1] = j
        end
      end
    end
    table.sort(out.junctions, function(a, b) return a.id < b.id end)
  end
  for _, cell in ipairs(cells) do
    for _, c in ipairs(cell.chunks) do
      local list = out.byStroke[c.stroke]
      if not list then list = {}; out.byStroke[c.stroke] = list end
      if not list[c.index] then
        list[c.index] = c
        out.chunks[#out.chunks + 1] = c
      end
    end
    for _, n in ipairs(cell.nodes) do
      if not out.byNode[n.node] then
        out.byNode[n.node] = n
        out.nodes[#out.nodes + 1] = n
      end
    end
  end
  table.sort(out.chunks, function(a, b) return a.stroke == b.stroke and a.index < b.index or a.stroke < b.stroke end)
  return out
end

-- Fields. chunk: stroke, index, chunks, oneway, closed, fitted, highway, length, d0, d1 (draw range),
-- n stations s, x, y, z, h, k, eL, eR, fl, fr (arrays), parts {way, segment, reversed, s0, s1, tags},
-- lanes {s0, s1, f, b, both, width, tf, tb}, events {kind, side, taper, parallel, separate, nose, nodeS,
-- width, branch, bd0, bd1}, furniture {kind, side, s0, s1, offset, way, onWay}, structures {kind, s0, s1,
-- layer, structure, rule, clearance, water, floor}, attach {kind, host, node, hostS, side, lane0, lane1,
-- hostLane0, hostLane1, offset}, nodes {node, s}. node: node, kind, x, y, z, mouths {stroke, s, dir},
-- roles {[stroke] = {role, host}}, plate {stroke, dir, cut, L, C, R} (version 2).
return M
