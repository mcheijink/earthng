-- Conformal ground (patches/GROUND-MESH-DESIGN.md, phase 1): the ground lattice is cut by the footprint of the swept
-- fast roads (carriageway rows and verge/slope bands), so no ground lies under them and the road edges stand on the
-- verge bands, whose toe edge is split at every lattice line so that it meets the ground vertex for vertex.
-- Pure Lua, no engine call (runs offline in LuaJIT); generator.lua feeds it the footprints and takes the pieces.
-- Off by default: M.on is the runtime switch (world.lua conformalGround).
--
-- The cut is convex polygon minus convex polygons (the lattice triangle against each footprint piece, an edge at a
-- time), so every ground piece stays convex and is fanned by the collector: no constrained Delaunay at run time.
-- A footprint that is not convex (a quad twisted on a bend) is cut as the two triangles it is drawn as.
local M = {}
local floor, abs, min, max, sqrt = math.floor, math.abs, math.min, math.max, math.sqrt
local NONE = {}

M.on = false
-- the last tile's counts (M.new resets them): footprints added, lattice triangles cut, ground pieces and triangles they make
M.stats = {footprints = 0, cuts = 0, pieces = 0, triangles = 0}
M.TUNE = {
  drop = .05,        -- m: the ground under the DEM (generator TUNE.groundDrop is .25); roads no longer lie on it
  lift = .02,        -- m: the road over its profile (generator ROAD_LIFT is .10)
  -- classes of the swept strokes that cut the ground (sweep.lua M.CLASSES.fast; links go with their road)
  classes = {motorway = true, trunk = true},
  sliver = 1e-7,     -- m2 (twice): a ring smaller than this has no area
  thin = 1e-5,       -- m: a ground piece thinner than this (area over its longest edge) is a rounding sliver of the cut
  snap = 1e-6,       -- m: the corners of a ground piece are kept on this grid, so the corners two cuts share are one
  -- join ground pieces whose union is convex: fewer triangles (-12 %), but 1 run in 2 left a hole of 2-3 mm in an
  -- interchange tile, in a different place each time (the order the strokes are visited in changes)
  merge = false,
  walls = true,      -- a wall closes the open end of a band at a bridge or a tunnel (the ground goes on there at its own height)
  wallPiece = 1,     -- m: an end wall is built in pieces this long, each down to what is drawn below it
  wallMerge = .02,   -- m: two corners of an end wall nearer than this are one (a crossing of the ground's lattice lines beside a piece corner)
  stack = true,      -- faces between stacked footprints too (stackWalls): an embankment over the cut of the road under it. Needs walls
  wallMin = .5,      -- m: a face between stacked footprints lower than this is not built (two layers of one road a hand apart; no one sees through it)
  groundMax = 60,    -- m: the ground is a floor for a stack wall only when it lies less than this under the edge (not where the lattice has no height)
  wallStep = .05,    -- m: a footprint this near in height is the same surface (the edge is joined to it)
  wallPast = .02,    -- m: that is read this far past the end line of the band
  dup = 1e-9,        -- m: corners nearer than this are one
  near = 1e-9,       -- m: a corner this near an edge line is on it
}

-- is this swept stroke one that cuts the ground
function M.stroke(S)
  return S ~= nil and S.swept and M.TUNE.classes[S.base] == true
end

-- the switch is on and the tile is one where the ground can follow the roads: the new height pipeline on a fine
-- ground grid (the cross-section bands exist there), not the vista rings
function M.wanted(options, terrain, fine)
  local on = options.conformal
  if on == nil then on = M.on end
  if not on or not options.roads3 or not options.heightPipeline or options.vista or options.horizon then return false end
  local grid = math.huge
  for _, g in ipairs(terrain) do
    local r = g.rect
    if r and (g.nx or 0) > 1 then grid = min(grid, (r[3] - r[1]) / (g.nx - 1)) end
  end
  return grid <= fine
end

-- twice the signed area (counter-clockwise > 0) of a ring of {x, y}
local function area2(p)
  local s = 0
  for k = 1, #p do local a, b = p[k], p[k % #p + 1]; s = s + a[1] * b[2] - b[1] * a[2] end
  return s
end
M.area2 = area2

-- a copy of poly as {x, y, z} corners without repeats, counter-clockwise; nil when it has no area
function M.ring(poly)
  local out = {}
  local dup = M.TUNE.dup
  for _, p in ipairs(poly) do
    local q = out[#out]
    if not q or abs(p[1] - q[1]) > dup or abs(p[2] - q[2]) > dup then out[#out + 1] = {p[1], p[2], p[3] or 0} end
  end
  local f, l = out[1], out[#out]
  if #out > 1 and abs(f[1] - l[1]) <= dup and abs(f[2] - l[2]) <= dup then out[#out] = nil end
  if #out < 3 then return nil end
  local a = area2(out)
  if abs(a) < M.TUNE.sliver then return nil end
  if a < 0 then
    local n = #out
    for k = 1, floor(n / 2) do out[k], out[n + 1 - k] = out[n + 1 - k], out[k] end
  end
  return out
end

-- a counter-clockwise ring turns left at every corner (collinear corners allowed) and turns once round: a star that
-- winds twice passes the first test alone
local function convex(r)
  local n = #r
  local turn = 0
  for k = 1, n do
    local p, q, s = r[k], r[k % n + 1], r[(k + 1) % n + 1]
    local ax, ay, bx, by = q[1] - p[1], q[2] - p[2], s[1] - q[1], s[2] - q[2]
    local cross = ax * by - ay * bx
    if cross < -1e-9 then return false end
    turn = turn + math.atan2(cross, ax * bx + ay * by)
  end
  return abs(turn - 2 * math.pi) < 1e-6
end

-- the convex counter-clockwise rings that cover what the collector draws of poly: poly itself when it is convex,
-- else its fan from the first corner (the collector's triangulation)
function M.parts(poly)
  local r = M.ring(poly)
  if not r then return NONE end
  if convex(r) then return {r} end
  local out = {}
  for k = 2, #poly - 1 do
    local t = M.ring({poly[1], poly[k], poly[k + 1]})
    if t then out[#out + 1] = t end
  end
  return out
end

-- the polygons to draw for poly: itself when it is convex, else the triangles of its fan from the first corner (the
-- collector's triangulation; the same corner tables, so a mark such as the toe's stays on them). A cut of a twisted quad
-- as a whole (generator emit) leaves pieces whose fan is not its fan, a sliver of ground or road lost along the cut.
function M.pieces(poly)
  local r = M.ring(poly)
  if not r then return {poly} end
  if convex(r) then return {poly} end
  local out = {}
  for k = 2, #poly - 1 do
    if M.ring({poly[1], poly[k], poly[k + 1]}) then out[#out + 1] = {poly[1], poly[k], poly[k + 1]} end
  end
  return out
end

-- The stretches of the ring `piece` ({x, y, z} corners) that lie on the segment a -> b (the end line of a band): edges of the
-- piece within 10 micrometres of that line, cut to the segment, each as {{x, y, z}, {x, y, z}} with the piece's own heights.
-- A band that is drawn only in part (a road passes through its end) has its end only where it is drawn.
function M.endSpans(piece, a, b)
  local out = {}
  local dx, dy = b[1] - a[1], b[2] - a[2]
  local len = sqrt(dx * dx + dy * dy)
  if len < 1e-9 then return out end
  local tol = 1e-5
  local n = #piece
  for k = 1, n do
    local u, v = piece[k], piece[k % n + 1]
    local du = abs(dx * (u[2] - a[2]) - dy * (u[1] - a[1])) / len
    local dv = abs(dx * (v[2] - a[2]) - dy * (v[1] - a[1])) / len
    if du <= tol and dv <= tol then
      local tu = ((u[1] - a[1]) * dx + (u[2] - a[2]) * dy) / (len * len)
      local tv = ((v[1] - a[1]) * dx + (v[2] - a[2]) * dy) / (len * len)
      local lo, hi = max(0, min(tu, tv)), min(1, max(tu, tv))
      if (hi - lo) * len > 1e-3 then
        local function at(t)
          local f = (t - tu) / (tv - tu)
          return {u[1] + (v[1] - u[1]) * f, u[2] + (v[2] - u[2]) * f, (u[3] or 0) + ((v[3] or 0) - (u[3] or 0)) * f}
        end
        out[#out + 1] = tu < tv and {at(lo), at(hi)} or {at(hi), at(lo)}
      end
    end
  end
  return out
end

-- Texture coordinates for an upright face (the four corners {x, y, z}, set in place as [4] u, [5] v): the collector maps a face
-- without them from its plan position, and a wall has no plan area, so the texture ran down it in streaks. Here u runs along the
-- wall and v with the height, `rep` m a repeat, both from the world's origin so the pieces of a wall and the walls that meet
-- carry one texture on (as the bridges' own faces do, ROADS.box)
function M.wallUV(wall, rep)
  local a = wall[1]
  local dx, dy = 0, 0
  for k = 2, #wall do
    dx, dy = wall[k][1] - a[1], wall[k][2] - a[2]
    if dx * dx + dy * dy > 1e-8 then break end
  end
  local l = sqrt(dx * dx + dy * dy)
  if l < 1e-4 then dx, dy, l = 1, 0, 1 end
  dx, dy = dx / l, dy / l
  for _, c in ipairs(wall) do
    c[4], c[5] = (c[1] * dx + c[2] * dy) / rep, -(c[3] or 0) / rep
  end
  return wall
end

-- The end walls of the bands (generator.lua): walls = {{p, q, ox, oy}...}, each along p -> q ({x, y, z} on the band's end
-- edge, z the band's), looking out along (ox, oy), away from the road. Each is cut into pieces of TUNE.wallPiece m; a piece
-- goes from the band edge down (or up) to what is drawn below it: the footprint surface just beyond the end line (another
-- road's band or carriageway that lies there), else the lattice ground. `ground(x, y)` is the lattice; `emit(wall)` takes
-- the four corners {{x, y, z}...} with the face looking out.
function M.buildWalls(walls, index, ground, emit, pause)
  local piece, past = M.TUNE.wallPiece, M.TUNE.wallPast
  for _, w in ipairs(walls) do
    if pause then pause() end
    local p, q = w[1], w[2]
    local ol = sqrt(w[3] * w[3] + w[4] * w[4])
    if ol > 1e-9 then
      local ox, oy = w[3] / ol, w[4] / ol
      local len = sqrt((q[1] - p[1]) ^ 2 + (q[2] - p[2]) ^ 2)
      local n = max(1, math.ceil(len / piece))
      -- the pieces' corners: evenly spaced, and where the line crosses an edge of the ground's triangles (the ground bends
      -- there, so a piece across it would stand off the ground in the middle)
      local ts = {}
      for k = 0, n do ts[#ts + 1] = k / n end
      if index.step and len > 1e-9 then
        for _, c in ipairs(M.crossings(p, q, index.step)) do
          ts[#ts + 1] = ((c[1] - p[1]) * (q[1] - p[1]) + (c[2] - p[2]) * (q[2] - p[2])) / (len * len)
        end
        table.sort(ts)
        local kept = {0}
        for k = 2, #ts - 1 do
          if (ts[k] - kept[#kept]) * len > M.TUNE.wallMerge and (1 - ts[k]) * len > M.TUNE.wallMerge then kept[#kept + 1] = ts[k] end
        end
        kept[#kept + 1] = 1
        ts = kept
      end
      local pts = {}
      for _, t in ipairs(ts) do
        local x, y, z = p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, (p[3] or 0) + ((q[3] or 0) - (p[3] or 0)) * t
        -- below the band's edge there is what lies just past the end line: the highest footprint under the band's height
        -- (a footprint above it that lies there as well has cut the ground away: the ground is no floor under that one), else
        -- the ground's plane
        local below = index:surface(x + ox * past, y + oy * past, z - 1e-3)
        if not below then below = ground(x, y) end
        pts[#pts + 1] = {x, y, z, below}
      end
      n = #pts - 1
      for k = 1, n do
        local a, b = pts[k], pts[k + 1]
        if abs(a[3] - a[4]) > 1e-3 or abs(b[3] - b[4]) > 1e-3 then
          local wall = {{a[1], a[2], a[3]}, {b[1], b[2], b[3]}, {b[1], b[2], b[4]}, {a[1], a[2], a[4]}}
          -- looks out: the normal of the corner fan points along (ox, oy)
          local ux, uy, uz = wall[2][1] - wall[1][1], wall[2][2] - wall[1][2], wall[2][3] - wall[1][3]
          local vx, vy, vz = wall[4][1] - wall[1][1], wall[4][2] - wall[1][2], wall[4][3] - wall[1][3]
          if (uy * vz - uz * vy) * ox + (uz * vx - ux * vz) * oy < 0 then wall = {wall[4], wall[3], wall[2], wall[1]} end
          emit(wall)
        end
      end
    end
  end
end

-- The vertical faces that close the free edges of the footprints where the surface goes on at another height: a band that
-- stands over another road's band or carriageway (an embankment over the cut of a road that passes under it) has no ground
-- under it any more to show a face, so its edge would hang open. For every edge of every footprint, at points TUNE.wallPiece m
-- apart: when a footprint other than its own lies at the edge's height just past it (within TUNE.wallStep), the edge is joined
-- to it and there is nothing to close; else, when other footprints lie lower there (past the edge, or under it), a face runs
-- from the edge down to the highest of them. A footprint's edge over the lattice ground is the end walls' (buildWalls).
-- A point with nothing lower under or beside it (the edge runs on over the lattice ground there) goes down to the ground's plane
-- (`ground(x, y)`, the lattice), not down to what a neighbouring point rests on: a face that took its neighbour's floor stood
-- off the ground where the ground bends.
-- emit(wall) takes the four corners {{x, y, z}...} with the face looking out of the footprint.
function M.stackWalls(index, emit, decks, pause, ground)
  local T = M.TUNE
  index:fine(pause)
  local made = {}
  if decks then decks:fine(pause) end
  local piece, past, tol = T.wallPiece, T.wallPast, T.wallStep
  for _, entry in ipairs(index.all) do
    local d = entry.drawn
    local n = #d
    -- the ring's turn: an edge looks out on its right in a counter-clockwise ring, on its left otherwise
    local sign = area2(d) > 0 and 1 or -1
    for k = 1, n do
      local a, b = d[k], d[k % n + 1]
      local dx, dy = b[1] - a[1], b[2] - a[2]
      local len = sqrt(dx * dx + dy * dy)
      -- (an edge is a stretch of work; the pause points of the collector come only with the faces, and a busy tile has few)
      if pause then pause() end
      if len > 0.02 then
        local ox, oy = sign * dy / len, -sign * dx / len
        local m = max(1, math.ceil(len / piece))
        local pts = {}
        local any, closed = false, 0
        for i = 0, m do
          local t = i / m
          local x, y, z = a[1] + dx * t, a[2] + dy * t, (a[3] or 0) + ((b[3] or 0) - (a[3] or 0)) * t
          local zs, rs = index:levels(x + ox * past, y + oy * past, entry)
          local joined, low, lowRoad = false, nil, false
          -- under a bridge deck or in a tunnel: no face there (it would stand in the structure)
          local covered = decks and (decks:covers(x, y) or decks:covers(x + ox * past, y + oy * past))
          if covered then zs = {} end
          for n, zo in ipairs(zs) do
            if abs(zo - z) <= tol then joined = true elseif zo < z - tol and (not low or zo > low) then low, lowRoad = zo, rs[n] end
          end
          if not joined and not covered then
            -- under the edge itself: another footprint that this one hangs over
            local zs2, rs2 = index:levels(x, y, entry)
            for n, zo in ipairs(zs2) do
              if abs(zo - z) <= tol then joined = true elseif zo < z - tol and (not low or zo > low) then low, lowRoad = zo, rs2[n] end
            end
          end
          if joined then closed = closed + 1 end
          -- nothing is drawn in front of the edge but the lattice ground (the footprint it rests on lies under the edge, not before
          -- it): the face goes on down to that ground, or it would hang over the gap
          if ground and low and not joined and not covered and #zs == 0 then
            local g = ground(x + ox * past, y + oy * past)
            if g and g < low - tol and z - g < T.groundMax then low, lowRoad = g, false end
          end
          -- a face that rests on a carriageway stands on its edge, not in it: not when the road lies under the edge's own side too
          if lowRoad and not joined then
            local zi, ri = index:levels(x - ox * past, y - oy * past, entry)
            for n, zo in ipairs(zi) do
              if ri[n] and abs(zo - low) < .3 then covered = true end
            end
          end
          pts[#pts + 1] = {x, y, z, (not joined) and low or nil, covered = covered, joined = joined}
          if pts[#pts][4] then any = true end
        end
        if any then
          -- a point with nothing below it takes the nearest one that has (not one under a deck: no face there)
          for i = 1, #pts do
            if not pts[i][4] and not pts[i].covered then
              local g = ground and not pts[i].joined and ground(pts[i][1], pts[i][2])
              if g and g < pts[i][3] - tol and pts[i][3] - g < T.groundMax then
                pts[i][4] = g
              else
                for r = 1, #pts do
                  local lo, hi = pts[i - r], pts[i + r]
                  if lo and lo[4] then pts[i][4] = lo[4]; break end
                  if hi and hi[4] then pts[i][4] = hi[4]; break end
                end
              end
            end
          end
          for i = 1, m do
            local p, q = pts[i], pts[i + 1]
            if p[4] and q[4] and not p.covered and not q.covered and (abs(p[3] - p[4]) > T.wallMin or abs(q[3] - q[4]) > T.wallMin) then
              local wall = {{p[1], p[2], p[3]}, {q[1], q[2], q[3]}, {q[1], q[2], q[4]}, {p[1], p[2], p[4]}}
              local ux, uy, uz = wall[2][1] - wall[1][1], wall[2][2] - wall[1][2], wall[2][3] - wall[1][3]
              local vx, vy, vz = wall[4][1] - wall[1][1], wall[4][2] - wall[1][2], wall[4][3] - wall[1][3]
              if (uy * vz - uz * vy) * ox + (uz * vx - ux * vz) * oy < 0 then wall = {wall[4], wall[3], wall[2], wall[1]} end
              -- where two footprints meet over a third, each stands at the other's edge: one face, not two in one plane
              -- (the material is double sided, and two coplanar faces z-fight)
              local key = {}
              for c = 1, 4 do key[c] = string.format('%d,%d,%d', floor(wall[c][1] * 1000 + .5), floor(wall[c][2] * 1000 + .5), floor(wall[c][3] * 1000 + .5)) end
              table.sort(key)
              key = table.concat(key, ';')
              if not made[key] then
                made[key] = true
                emit(wall)
              end
            end
          end
        end
      end
    end
  end
end

-- a ring of {x, y} without repeated corners (a clip leaves the corner on the cutting line twice), on the grid g if
-- given; nil when it has no area or is thinner than TUNE.thin
local function tidy(poly, g)
  local out, dup = {}, M.TUNE.dup
  for _, p in ipairs(poly) do
    local x, y = p[1], p[2]
    if g then x, y = floor(x / g + .5) * g, floor(y / g + .5) * g end
    local q = out[#out]
    if not q or abs(x - q[1]) > dup or abs(y - q[2]) > dup then out[#out + 1] = {x, y} end
  end
  local f, l = out[1], out[#out]
  if #out > 1 and abs(f[1] - l[1]) <= dup and abs(f[2] - l[2]) <= dup then out[#out] = nil end
  local n = #out
  if n < 3 then return nil end
  local a, longest = abs(area2(out)), 0
  for k = 1, n do
    local p, q = out[k], out[k % n + 1]
    longest = max(longest, (q[1] - p[1]) ^ 2 + (q[2] - p[2]) ^ 2)
  end
  if a < M.TUNE.sliver or a / sqrt(longest) < M.TUNE.thin then return nil end
  return out
end

-- a ring without the corners a clip leaves twice on the cutting line (the same corner tables, no copy); the ring has no
-- area when fewer than three corners are left
local function dedupe(poly)
  local n, dup = #poly, M.TUNE.dup
  local out = {}
  for k = 1, n do
    local p, q = poly[k], out[#out]
    if not q or abs(p[1] - q[1]) > dup or abs(p[2] - q[2]) > dup then out[#out + 1] = p end
  end
  local f, l = out[1], out[#out]
  if #out > 1 and abs(f[1] - l[1]) <= dup and abs(f[2] - l[2]) <= dup then out[#out] = nil end
  return #out >= 3 and out or NONE
end

-- cutting a ring along the line a -> b: the part on its left (inside a counter-clockwise cutter) and on its right
local function split(poly, a, b)
  local n = #poly
  local ax, ay, dx, dy = a[1], a[2], b[1] - a[1], b[2] - a[2]
  local e = M.TUNE.near * sqrt(dx * dx + dy * dy)
  -- wholly on one side (the usual case: a footprint that only touches the piece's bounds): the ring itself, no copy
  local over, under = false, false
  for k = 1, n do
    local p = poly[k]
    local sp = dx * (p[2] - ay) - dy * (p[1] - ax)
    if sp > e then over = true elseif sp < -e then under = true end
  end
  if not under then return poly, NONE end
  if not over then return NONE, poly end
  local inside, outside = {}, {}
  for k = 1, n do
    local p, q = poly[k], poly[k % n + 1]
    local sp = dx * (p[2] - ay) - dy * (p[1] - ax)
    local sq = dx * (q[2] - ay) - dy * (q[1] - ax)
    if sp >= -e then inside[#inside + 1] = p end
    if sp <= e then outside[#outside + 1] = p end
    if (sp > e and sq < -e) or (sp < -e and sq > e) then
      local t = sp / (sp - sq)
      local x = {p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t}
      inside[#inside + 1], outside[#outside + 1] = x, x
    end
  end
  return dedupe(inside), dedupe(outside)
end

-- some edge of the counter-clockwise ring b has every corner of a on or past its outer side: the two do not overlap
-- (no allocation; most pieces of a cell are clear of most footprints, whose bounds only touch the cell)
local function apart(a, b)
  local nb = #b
  for k = 1, nb do
    local p, q = b[k], b[k % nb + 1]
    local dx, dy = q[1] - p[1], q[2] - p[2]
    local e = 1e-9 * (abs(dx) + abs(dy))
    local clear = true
    for i = 1, #a do
      local v = a[i]
      if dx * (v[2] - p[2]) - dy * (v[1] - p[1]) > e then clear = false; break end
    end
    if clear then return true end
  end
  return false
end

-- the pieces of the convex ring `piece` outside the convex counter-clockwise ring `cutter`, appended to out
local function subtract(piece, cutter, out)
  local rest = piece
  local n = #cutter
  for k = 1, n do
    local inner, outer = split(rest, cutter[k], cutter[k % n + 1])
    if #outer >= 3 then
      if #inner < 3 then out[#out + 1] = rest; return end   -- what is left lies wholly outside this edge
      out[#out + 1] = outer
    end
    rest = inner
    if #rest < 3 then return end
  end
end
M.subtract = subtract

-- Convex pieces that share a whole edge and whose union is still convex become one piece (the Hertel-Mehlhorn
-- step): the cut leaves a wedge per footprint along a curved edge, and where the edge bends away they join again.
-- Corners are one when within TUNE.dup of each other (the cuts of two neighbours meet at the same point).
local function merge(pieces)
  local dup = M.TUNE.snap * 2
  local function same(p, q) return abs(p[1] - q[1]) <= dup and abs(p[2] - q[2]) <= dup end
  local again = true
  while again and #pieces > 1 do
    again = false
    for i = 1, #pieces - 1 do
      local A = pieces[i]
      local na = #A
      for k = 1, na do
        -- the edge A[k] -> A[k + 1] against every later piece's edge in the opposite direction
        local u, v = A[k], A[k % na + 1]
        for j = i + 1, #pieces do
          local B = pieces[j]
          local nb = #B
          for m = 1, nb do
            if same(B[m], v) and same(B[m % nb + 1], u) then
              -- A from v round to u, then B from after u round to before v
              local out = {}
              for s = 1, na do out[#out + 1] = A[(k + s - 1) % na + 1] end   -- v ... u
              for s = 1, nb - 2 do out[#out + 1] = B[(m + s) % nb + 1] end   -- B after u ... before v
              if convex(out) and abs(area2(out) - area2(A) - area2(B)) < 1e-9 then
                pieces[i] = out
                table.remove(pieces, j)
                again = true
              end
              break
            end
          end
          if again then break end
        end
        if again then break end
      end
      if again then break end
    end
  end
  return pieces
end

-- a piece without the corners that lie on the straight line between their neighbours: the piece is the same shape and
-- fans into fewer triangles (a ground piece lies in one lattice plane, so it is the same surface in z as well)
local function straighten(piece)
  local out = {}
  for k = 1, #piece do out[k] = piece[k] end
  local k = 1
  -- one corner at a time against its neighbours as they now are: two corners each within the tolerance of the
  -- line past the other would, dropped together, move the edge by more than it
  while k <= #out and #out > 3 do
    local n = #out
    local p, q, r = out[(k + n - 2) % n + 1], out[k], out[k % n + 1]
    local dx, dy = r[1] - p[1], r[2] - p[2]
    local len = sqrt(dx * dx + dy * dy)
    -- the corner goes when it is within 2 micrometres of the line p -> r and between them
    if len >= 1e-12 and abs(dx * (q[2] - p[2]) - dy * (q[1] - p[1])) / len <= 2 * M.TUNE.snap
        and (q[1] - p[1]) * dx + (q[2] - p[2]) * dy >= 0 and (q[1] - r[1]) * dx + (q[2] - r[2]) * dy <= 0 then
      table.remove(out, k)
    else
      k = k + 1
    end
  end
  return out
end

-- ---------------------------------------------------------------------------------------------------
-- An index of footprints by lattice cell
-- ---------------------------------------------------------------------------------------------------

local Index = {}
Index.__index = Index

local function cellKey(gx, gy) return (gx + 32768) * 65536 + gy + 32768 end

-- mark the 1 m cells a convex polygon (corners {x, y}) overlaps, row by row: surface() asks the mask first, and the
-- scene asks it hundreds of thousands of times, nearly always far from any road
local function mark(mask, poly, bins, entry)
  local n = #poly
  local y0, y1 = math.huge, -math.huge
  for k = 1, n do y0, y1 = min(y0, poly[k][2]), max(y1, poly[k][2]) end
  for row = floor(y0), floor(y1) do
    -- the polygon's x extent inside the strip row .. row + 1: its corners in it, its edges at the strip's sides
    local lo, hi = math.huge, -math.huge
    for k = 1, n do
      local p, q = poly[k], poly[k % n + 1]
      if p[2] >= row and p[2] <= row + 1 then lo, hi = min(lo, p[1]), max(hi, p[1]) end
      for _, y in ipairs({row, row + 1}) do
        if (p[2] - y) * (q[2] - y) < 0 then
          local x = p[1] + (q[1] - p[1]) * (y - p[2]) / (q[2] - p[2])
          lo, hi = min(lo, x), max(hi, x)
        end
      end
    end
    for col = floor(lo), floor(hi) do
      local k = col * 262144 + row
      if mask then mask[k] = true end
      if bins then
        -- (the parts of one entry follow each other: a cell two of them touch has the entry last)
        local list = bins[k]
        if not list then bins[k] = {entry} elseif list[#list] ~= entry then list[#list + 1] = entry end
      end
    end
  end
end

-- a footprint index for a lattice of `step` m over `rect` ({x0, y0, x1, y1}): only footprints touching the rect
function M.new(step, rect, quiet)
  local s = M.stats
  if not quiet then s.footprints, s.cuts, s.pieces, s.triangles = 0, 0, 0, 0 end
  return setmetatable({step = step, rect = rect, bins = {}, mask = {}, all = {}, quiet = quiet}, Index)
end

-- add a footprint (corners {x, y, z}, the corners the collector draws with): it cuts the ground and answers
-- surface(x, y) with its drawn height
function Index:add(poly, road)
  local rect, step = self.rect, self.step
  local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
  for _, p in ipairs(poly) do
    x0, x1 = min(x0, p[1]), max(x1, p[1])
    y0, y1 = min(y0, p[2]), max(y1, p[2])
  end
  if x1 < rect[1] or x0 > rect[3] or y1 < rect[2] or y0 > rect[4] then return end
  -- the z of the drawn surface comes from the collector's fan of the polygon as given
  local parts = M.parts(poly)
  if #parts == 0 then return end
  for _, part in ipairs(parts) do mark(self.mask, part) end
  local entry = {drawn = poly, parts = parts, box = {x0, y0, x1, y1}, road = road}
  self.all[#self.all + 1] = entry
  if not self.quiet then M.stats.footprints = M.stats.footprints + 1 end
  local gx0, gx1, gy0, gy1 = floor(x0 / step), floor(x1 / step), floor(y0 / step), floor(y1 / step)
  local wide = (gx1 - gx0 + 1) * (gy1 - gy0 + 1) > 2
  for gy = gy0, gy1 do
    for gx = gx0, gx1 do
      -- only the cells the footprint really touches (a skewed band's bounds hold many that it misses)
      local touches = not wide
      if wide then
        touches = false
        local cell = {{gx * step, gy * step}, {(gx + 1) * step, gy * step}, {(gx + 1) * step, (gy + 1) * step}, {gx * step, (gy + 1) * step}}
        for _, part in ipairs(parts) do
          if not (apart(cell, part) or apart(part, cell)) then touches = true; break end
        end
      end
      if touches then
        local k = cellKey(gx, gy)
        local bin = self.bins[k]
        if not bin then bin = {}; self.bins[k] = bin end
        bin[#bin + 1] = entry
      end
    end
  end
end

-- the footprints of the lattice cell (gx, gy), or nil
function Index:cell(gx, gy)
  return self.bins[cellKey(gx, gy)]
end

-- the convex pieces of ring (corners {x, y}, counter-clockwise, inside lattice cell gx, gy) that no footprint covers
function Index:cut(ring, gx, gy)
  local bin = self.bins[cellKey(gx, gy)]
  if not bin then return {ring} end
  local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
  for _, p in ipairs(ring) do
    x0, x1 = min(x0, p[1]), max(x1, p[1])
    y0, y1 = min(y0, p[2]), max(y1, p[2])
  end
  local pieces = {ring}
  for _, entry in ipairs(bin) do
    local b = entry.box
    if b[1] <= x1 and b[3] >= x0 and b[2] <= y1 and b[4] >= y0 then
      local out = {}
      for _, piece in ipairs(pieces) do
        -- a piece's bounds are worked out once (it is cut again by every footprint of the cell)
        local pb = piece.box
        if not pb then
          pb = {math.huge, math.huge, -math.huge, -math.huge}
          for _, p in ipairs(piece) do
            if p[1] < pb[1] then pb[1] = p[1] end
            if p[1] > pb[3] then pb[3] = p[1] end
            if p[2] < pb[2] then pb[2] = p[2] end
            if p[2] > pb[4] then pb[4] = p[2] end
          end
          piece.box = pb
        end
        if pb[1] > b[3] or pb[3] < b[1] or pb[2] > b[4] or pb[4] < b[2] then
          out[#out + 1] = piece
        else
          -- what is left of the piece after each part of the footprint in turn
          local rest = {piece}
          for _, cutter in ipairs(entry.parts) do
            local after = {}
            for _, r in ipairs(rest) do
              if apart(r, cutter) or apart(cutter, r) then after[#after + 1] = r else subtract(r, cutter, after) end
            end
            rest = after
            if #rest == 0 then break end
          end
          for _, r in ipairs(rest) do out[#out + 1] = r end
        end
      end
      pieces = out
      if #pieces == 0 then return pieces end
    end
  end
  -- the corners on the snap grid (the corners two cuts share are one), the rounding slivers gone, the pieces that make
  -- a convex one joined
  local kept = {}
  for _, piece in ipairs(pieces) do
    local t = tidy(piece, M.TUNE.snap)
    if t then kept[#kept + 1] = t end
  end
  if M.TUNE.merge then kept = merge(kept) end
  for k, piece in ipairs(kept) do
    local s = straighten(piece)
    if #s >= 3 then kept[k] = s end
  end
  local s = M.stats
  s.cuts, s.pieces = s.cuts + 1, s.pieces + #kept
  for _, piece in ipairs(kept) do s.triangles = s.triangles + #piece - 2 end
  return kept
end

-- the drawn height of one footprint at (x, y), or nil when it does not cover the point
local function entryZ(entry, x, y)
  local b = entry.box
  if x < b[1] or x > b[3] or y < b[2] or y > b[4] then return nil end
  local d = entry.drawn
  local a = d[1]
  for k = 2, #d - 1 do
    local p, q = d[k], d[k + 1]
    local den = (p[2] - q[2]) * (a[1] - q[1]) + (q[1] - p[1]) * (a[2] - q[2])
    if abs(den) > 1e-12 then
      local l1 = ((p[2] - q[2]) * (x - q[1]) + (q[1] - p[1]) * (y - q[2])) / den
      local l2 = ((q[2] - a[2]) * (x - q[1]) + (a[1] - q[1]) * (y - q[2])) / den
      local l3 = 1 - l1 - l2
      if l1 >= -1e-9 and l2 >= -1e-9 and l3 >= -1e-9 then return l1 * a[3] + l2 * p[3] + l3 * q[3] end
    end
  end
  return nil
end

-- the drawn height of the footprints at (x, y) (the highest where they overlap), or nil. With `below`, only the ones drawn under
-- that height; `skip` is a footprint to leave out
function Index:surface(x, y, below, skip)
  if not self.mask[floor(x) * 262144 + floor(y)] then return nil end
  local step = self.step
  local bin = self.bins[cellKey(floor(x / step), floor(y / step))]
  if not bin then return nil end
  local best
  for _, entry in ipairs(bin) do
    if entry ~= skip then
      local z = entryZ(entry, x, y)
      if z and (not below or z < below) and (not best or z > best) then best = z end
    end
  end
  return best
end

-- the footprints by 1 m cell (the cells of the mask): a point asks the one to three footprints that touch its cell, not the
-- ~16 of its lattice bin, most of which lie a few metres off. Built when asked for (stackWalls, which asks for hundreds of
-- thousands of points in a busy tile); the answers are the same, as the mask is read first anyway
function Index:fine(pause)
  if self.fineBins then return end
  local bins = {}
  for k, entry in ipairs(self.all) do
    for _, part in ipairs(entry.parts) do mark(nil, part, bins, entry) end
    if pause and k % 32 == 0 then pause() end
  end
  self.fineBins = bins
end

-- the drawn heights of all the footprints at (x, y) but `skip`, as a list; and a list of whether each is a carriageway
function Index:levels(x, y, skip)
  local cell = floor(x) * 262144 + floor(y)
  -- (nothing there: the one shared empty list, to be read, never written)
  if not self.mask[cell] then return NONE, NONE end
  local out, roads = {}, {}
  local list
  if self.fineBins then
    list = self.fineBins[cell]
  else
    local step = self.step
    list = self.bins[cellKey(floor(x / step), floor(y / step))]
  end
  for _, entry in ipairs(list or NONE) do
    if entry ~= skip then
      local z = entryZ(entry, x, y)
      if z then out[#out + 1] = z; roads[#out] = entry.road or false end
    end
  end
  return out, roads
end

-- does any footprint cover (x, y)
function Index:covers(x, y)
  local cell = floor(x) * 262144 + floor(y)
  if not self.mask[cell] then return false end
  local list
  if self.fineBins then
    list = self.fineBins[cell]
  else
    local step = self.step
    list = self.bins[cellKey(floor(x / step), floor(y / step))]
  end
  for _, entry in ipairs(list or NONE) do
    if entryZ(entry, x, y) then return true end
  end
  return false
end

-- z at (x, y) on the plane of the triangle p, q, r ({x, y, z})
function M.planeZ(p, q, r, x, y)
  local den = (q[2] - r[2]) * (p[1] - r[1]) + (r[1] - q[1]) * (p[2] - r[2])
  local l1 = ((q[2] - r[2]) * (x - r[1]) + (r[1] - q[1]) * (y - r[2])) / den
  local l2 = ((r[2] - p[2]) * (x - r[1]) + (p[1] - r[1]) * (y - r[2])) / den
  return l1 * p[3] + l2 * q[3] + (1 - l1 - l2) * r[3]
end

-- ---------------------------------------------------------------------------------------------------
-- The toe edge of a band
-- ---------------------------------------------------------------------------------------------------

-- the points where the segment p -> q crosses a line of the lattice triangles (x = k step, y = k step, x - y = k step),
-- in order along it, as {x, y} (any other field of p and q interpolated); the ground's edges are those lines, so the
-- edge of a band cut along the segment has a corner wherever a ground piece does
function M.crossings(p, q, step)
  local ts = {}
  local function lines(a0, da)
    if abs(da) < 1e-12 then return end
    local lo, hi = min(a0, a0 + da) / step, max(a0, a0 + da) / step
    for k = floor(lo) + 1, math.ceil(hi) - 1 do ts[#ts + 1] = (k * step - a0) / da end
  end
  local dx, dy = q[1] - p[1], q[2] - p[2]
  lines(p[1], dx)
  lines(p[2], dy)
  lines(p[1] - p[2], dx - dy)
  table.sort(ts)
  local out, last = {}, 0
  for _, t in ipairs(ts) do
    if t > last + 1e-9 and t < 1 - 1e-9 then
      local n = max(#p, #q)
      local r = {}
      for k = 1, n do r[k] = (p[k] or 0) + ((q[k] or 0) - (p[k] or 0)) * t end
      out[#out + 1] = r
      last = t
    end
  end
  return out
end

-- a band's corners {inner a, outer a, outer b, inner b} with the lattice crossings of its toe edge (outer a -> outer b)
-- between the two outer corners; the corners and the marks on them (toe: [9]) stay the same tables
function M.withToe(quad, step)
  local mid = M.crossings(quad[2], quad[3], step)
  if #mid == 0 then return quad end
  local out = {quad[1], quad[2]}
  for _, p in ipairs(mid) do
    p[9] = 1
    out[#out + 1] = p
  end
  out[#out + 1], out[#out + 2] = quad[3], quad[4]
  out.toe = quad.toe
  return out
end

return M
