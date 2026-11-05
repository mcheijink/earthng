-- Noise barriers for the tile generator (generator.lua build: ROADS.noise). The exporter gives
-- barrier=noise_barrier and wall=noise_barrier ways as lines of kind 'noise_barrier' (an older pack: kind
-- 'fence' with wall=noise_barrier, taken here too). A barrier is a wall of its tagged height (TUNE.height
-- when untagged) on the ground, or on a bridge's deck where it runs along one, in the tile's static
-- collision, with posts as decoration.
--
-- OSM draws a barrier where the imagery showed it, and the pack's carriageways are as wide as their lanes:
-- a barrier often falls on the hard shoulder, or between the two carriageways of a motorway. Each sample
-- (every TUNE.step m) that lies inside a road's corridor (its paved width and TUNE.clear m beside it) moves
-- out square to the road to the corridor's nearest outer edge. A one-way carriageway's corridor reaches
-- over the median to its partner's outer edge, so a barrier is never left in the median. A sample that
-- would move more than TUNE.maxMove m, or where the barrier crosses the road, is dropped (a gap).
local M = {}
local floor, sqrt, abs, min, max = math.floor, math.sqrt, math.abs, math.min, math.max
local NONE = {}

M.TUNE = {
  height = 4, minHeight = 1.5, maxHeight = 8,   -- m: untagged, and the range a tag is held to
  thick = .2,          -- m: the panels
  clear = 1.5,         -- m: past the paved edge (the shoulder's guard rail stands in it)
  step = 4,            -- m: samples along the barrier
  partner = 30,        -- m: a one-way carriageway's partner lies on its left within this (keep-right traffic)
  maxMove = 25,        -- m: a sample further inside is dropped, not moved (out of a median over a 3-lane
                       -- carriageway, ~15 m paved, is ~20 m)
  smooth = 2,          -- samples each side: the shift is averaged over them
  simplify = .15,      -- m: points nearer the line between their neighbours go (up to maxSeg m apart)
  maxSeg = 12,
  sink = .4,           -- m: the wall's foot under the ground
  post = 5, postHalf = .08, postProud = .06,   -- m: posts' spacing, half width and reach past the panels
  deckReach = 2.5,     -- m: a barrier this near a bridge's edge, along it, stands on its deck
  rightHand = true,
}
local T = M.TUNE

-- materials: existing building kinds (solid in the collision rebuild), by the barrier's material tag
local MATERIAL = {concrete = 'wall_precast', metal = 'clad_metal', aluminium = 'clad_metal', steel = 'clad_metal',
  wood = 'hw_boards', glass = 'clad_metal', plastic = 'clad_metal', acrylic_glass = 'clad_metal'}
local SKIP = {earth = true, soil = true, gravel = true}   -- an earth bank is terrain, not a wall

-- the lines this module draws (generator.lua leaves them out of its fences)
function M.owns(line)
  local tags = line.tags or NONE
  return line.kind == 'noise_barrier' or (line.kind == 'fence' and tags.wall == 'noise_barrier')
end

local function heightOf(tags)
  local text = tostring(tags.height or ''):gsub(',', '.')
  local h = tonumber(text:match('^%s*([%d%.]+)'))
  if not h then return T.height end
  return max(T.minHeight, min(T.maxHeight, h))
end
M.heightOf = heightOf

local function onewayOf(tags)
  local o = tags.oneway
  if o == 'yes' or o == '1' or o == 'true' then return 1 end
  if o == '-1' or o == 'reverse' then return -1 end
  if o == 'no' then return 0 end
  local h = tags.highway
  if h == 'motorway' or h == 'motorway_link' or tags.junction == 'roundabout' then return 1 end
  return 0
end

local function binKey(x, y) return (floor(x / 50) + 32768) * 65536 + floor(y / 50) + 32768 end

-- The shift (sx, sy) that takes (x, y) out of the corridor of the road nearest it, or 0, 0 if it lies in
-- none; nil if the barrier crosses that road there (tx, ty: its own direction) or the shift is too long.
local function shiftAt(bins, x, y, tx, ty)
  local best, state, mx, my = math.huge, 0, 0, 0     -- state: 0 in no corridor, 1 move by (mx, my), 2 a gap
  local bin = bins[binKey(x, y)] or NONE
  for _, s in ipairs(bin) do
    if s.surface and s.len2 > 1e-9 then
      local a = s.a
      local t = ((x - a[1]) * s.dx + (y - a[2]) * s.dy) / s.len2
      if t >= 0 and t <= 1 then
        local l = sqrt(s.len2)
        local ux, uy = s.dx / l, s.dy / l
        local lx, ly = -uy, ux                         -- left of the way's drawing direction
        local u = (x - a[1]) * lx + (y - a[2]) * ly    -- across, left positive
        local hw = s.width / 2
        local lo, hi = -(hw + T.clear), hw + T.clear
        local dir = onewayOf(s.tags or NONE)
        if dir ~= 0 then
          -- travel direction (dir * u); keep-right traffic: the median and the partner lie on its left
          local side = (T.rightHand and 1 or -1) * dir
          -- the partner: another one-way way running the other way, alongside, on that side
          local reach = nil
          for _, p in ipairs(bin) do
            if p ~= s and p.len2 > 1e-9 and onewayOf(p.tags or NONE) ~= 0 then
              local pl = sqrt(p.len2)
              local cos = (p.dx * ux + p.dy * uy) / pl * onewayOf(p.tags or NONE) * dir
              if cos < -.9 then
                -- the partner's centre across from this way at (x, y)
                local pt = ((x - p.a[1]) * p.dx + (y - p.a[2]) * p.dy) / p.len2
                if pt >= -.05 and pt <= 1.05 then
                  local px, py = p.a[1] + p.dx * pt, p.a[2] + p.dy * pt
                  local pu = (px - a[1]) * lx + (py - a[2]) * ly
                  if pu * side > 0 and abs(pu) < T.partner then
                    local edge = abs(pu) + p.width / 2 + T.clear
                    if not reach or edge < reach then reach = edge end
                  end
                end
              end
            end
          end
          if reach then
            if side > 0 then hi = max(hi, reach) else lo = min(lo, -reach) end
          end
        end
        if u > lo and u < hi then
          local d = abs(u)
          if d < best then
            best = d
            -- crossing the road (more across it than along): a gap, not a move
            local move = ((u - lo < hi - u) and lo or hi) - u
            if abs(tx * ux + ty * uy) < .5 or abs(move) > T.maxMove then
              state = 2
            else
              state, mx, my = 1, lx * move, ly * move
            end
          end
        end
      end
    end
  end
  if state == 2 then return nil end
  return mx, my
end
M.shiftAt = shiftAt

-- a bridge's deck under (x, y) when the barrier runs along it at its edge: the deck's drawn height, or nil
local function deckAt(bins, x, y, tx, ty, lift)
  for _, s in ipairs(bins[binKey(x, y)] or NONE) do
    local tags = s.tags or NONE
    if not s.surface and s.len2 > 1e-9 and tags.bridge and tags.bridge ~= 'no' then
      local a = s.a
      local t = ((x - a[1]) * s.dx + (y - a[2]) * s.dy) / s.len2
      if t >= 0 and t <= 1 then
        local l = sqrt(s.len2)
        local qx, qy = a[1] + s.dx * t, a[2] + s.dy * t
        local d = sqrt((x - qx) ^ 2 + (y - qy) ^ 2)
        if d < s.width / 2 + T.deckReach and abs(tx * s.dx + ty * s.dy) / l > .8 then
          return (a[3] or 0) + ((s.b[3] or 0) - (a[3] or 0)) * t + lift
        end
      end
    end
  end
  return nil
end

-- The drawn verges and side slopes by 25 m bin (generator roads' strip.verges, swept roads' SW.skirts): a
-- barrier beside an embankment stands on the slope as drawn, not on the ground lattice under it
local function vergeIndex(strips, skirts)
  local bins = {}
  local function add(poly)
    if not poly or #poly < 3 then return end
    local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
    for _, p in ipairs(poly) do x0, y0, x1, y1 = min(x0, p[1]), min(y0, p[2]), max(x1, p[1]), max(y1, p[2]) end
    for gx = floor(x0 / 25), floor(x1 / 25) do
      for gy = floor(y0 / 25), floor(y1 / 25) do
        local key = (gx + 32768) * 65536 + gy + 32768
        local bin = bins[key] or {}
        bins[key] = bin
        bin[#bin + 1] = poly
      end
    end
  end
  for _, strip in ipairs(strips or NONE) do
    for _, item in ipairs(strip.verges or NONE) do if not item.pressOnly then add(item[1]) end end
  end
  for _, item in ipairs(skirts or NONE) do add(item[1]) end
  return bins
end

-- the highest drawn verge over (x, y), or nil
local function vergeZ(bins, x, y)
  local best = nil
  for _, poly in ipairs(bins[(floor(x / 25) + 32768) * 65536 + floor(y / 25) + 32768] or NONE) do
    local a = poly[1]
    for k = 2, #poly - 1 do
      local b, c = poly[k], poly[k + 1]
      local den = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
      if abs(den) > 1e-9 then
        local u = ((b[2] - c[2]) * (x - c[1]) + (c[1] - b[1]) * (y - c[2])) / den
        local v = ((c[2] - a[2]) * (x - c[1]) + (a[1] - c[1]) * (y - c[2])) / den
        if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6 then
          local z = u * (a[3] or 0) + v * (b[3] or 0) + (1 - u - v) * (c[3] or 0)
          if not best or z > best then best = z end
        end
      end
    end
  end
  return best
end

-- The barrier's runs: lists of {x, y} placed beside the roads, simplified.
local function place(line, bins)
  local pts = line.points
  -- samples every step m (and the line's ends)
  local X, Y = {}, {}
  for k = 1, #pts - 1 do
    local a, b = pts[k], pts[k + 1]
    local len = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    local n = max(1, math.ceil(len / T.step))
    for i = (k == 1 and 0 or 1), n do
      X[#X + 1], Y[#Y + 1] = a[1] + (b[1] - a[1]) * i / n, a[2] + (b[2] - a[2]) * i / n
    end
  end
  local n = #X
  if n < 2 then return {} end
  local SX, SY, OK = {}, {}, {}
  for k = 1, n do
    local a, b = max(1, k - 1), min(n, k + 1)
    local tx, ty = X[b] - X[a], Y[b] - Y[a]
    local l = sqrt(tx * tx + ty * ty)
    if l > 1e-9 then tx, ty = tx / l, ty / l end
    local sx, sy = shiftAt(bins, X[k], Y[k], tx, ty)
    OK[k], SX[k], SY[k] = sx ~= nil, sx or 0, sy or 0
  end
  -- the shifts averaged along the barrier (a moved stretch eases in and out), over kept samples only
  local runs, run = {}, nil
  for k = 1, n do
    if OK[k] then
      local sx, sy, w = 0, 0, 0
      for j = max(1, k - T.smooth), min(n, k + T.smooth) do
        if OK[j] then sx, sy, w = sx + SX[j], sy + SY[j], w + 1 end
      end
      local x, y = X[k] + sx / w, Y[k] + sy / w
      run = run or {}
      run[#run + 1] = {x, y}
    elseif run then
      if #run >= 2 then runs[#runs + 1] = run end
      run = nil
    end
  end
  if run and #run >= 2 then runs[#runs + 1] = run end
  -- simplified: a point off the line between its kept neighbours by under `simplify` m goes
  for r, pts2 in ipairs(runs) do
    local out, last = {pts2[1]}, 1
    for k = 2, #pts2 - 1 do
      local a, c = pts2[last], pts2[k + 1]
      local dx, dy = c[1] - a[1], c[2] - a[2]
      local l = sqrt(dx * dx + dy * dy)
      local far = false
      for j = last + 1, k do
        local p = pts2[j]
        if l < 1e-9 or abs((p[1] - a[1]) * dy - (p[2] - a[2]) * dx) / l > T.simplify then far = true; break end
      end
      if far or l > T.maxSeg then
        out[#out + 1] = pts2[k]
        last = k
      end
    end
    out[#out + 1] = pts2[#pts2]
    runs[r] = out
  end
  return runs
end
M.place = place

-- Drawing: four scratch corners turned to face (ox, oy, oz), read by the collector and kept by none.
local C = {{}, {}, {}, {}}
local FWD, BACK = {C[1], C[2], C[3], C[4]}, {C[4], C[3], C[2], C[1]}
local function at(i, x, y, z, u, v)
  local p = C[i]
  p[1], p[2], p[3], p[4], p[5] = x, y, z, u, v
end
local function emit(col, kind, ox, oy, oz)
  local nx, ny, nz = 0, 0, 0
  for k = 1, 4 do
    local a, b = C[k], C[k % 4 + 1]
    nx, ny, nz = nx + (a[2] - b[2]) * (a[3] + b[3]), ny + (a[3] - b[3]) * (a[1] + b[1]), nz + (a[1] - b[1]) * (a[2] + b[2])
  end
  col.polygon(kind, nx * ox + ny * oy + nz * oz >= 0 and FWD or BACK, false)
end

local function inside(rect, x, y) return x >= rect[1] and x < rect[3] and y >= rect[2] and y < rect[4] end

-- draw(ctx): ctx: lines, rect, segmentBins, lift, ground(x, y), wall (the scene collector: static collision),
-- decor (no collision), far (no posts), strips and skirts (the drawn verges: generator roads', swept
-- roads'), pause. Returns the triangles drawn.
function M.draw(ctx)
  local rect, bins, ground, lift = ctx.rect, ctx.segmentBins, ctx.ground, ctx.lift or 0
  local tris, done, verges = 0, 0, nil
  for _, line in ipairs(ctx.lines) do
    local tags = line.tags or NONE
    if M.owns(line) and #(line.points or NONE) >= 2 and not SKIP[tags.material or ''] then
      local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
      for _, p in ipairs(line.points) do x0, y0, x1, y1 = min(x0, p[1]), min(y0, p[2]), max(x1, p[1]), max(y1, p[2]) end
      local pad = T.maxMove + 5
      if x1 >= rect[1] - pad and x0 <= rect[3] + pad and y1 >= rect[2] - pad and y0 <= rect[4] + pad then
        local h, kind, half = heightOf(tags), MATERIAL[tags.material or ''] or 'wall_precast', T.thick / 2
        verges = verges or vergeIndex(ctx.strips, ctx.skirts)
        for _, run in ipairs(place(line, bins)) do
          -- per point: the foot (ground or deck), the mitred normal
          local n = #run
          local base, NX, NY = {}, {}, {}
          for k = 1, n do
            local a, b = run[max(1, k - 1)], run[min(n, k + 1)]
            local tx, ty = b[1] - a[1], b[2] - a[2]
            local l = max(1e-9, sqrt(tx * tx + ty * ty))
            tx, ty = tx / l, ty / l
            local mitre = 1
            if k > 1 and k < n then
              local sx, sy = run[k][1] - run[k - 1][1], run[k][2] - run[k - 1][2]
              local sl = sqrt(sx * sx + sy * sy)
              if sl > 1e-9 then mitre = 1 / max(.5, (sx * tx + sy * ty) / sl) end
            end
            NX[k], NY[k] = -ty * mitre, tx * mitre
            local p = run[k]
            base[k] = deckAt(bins, p[1], p[2], tx, ty, lift) or max(ground(p[1], p[2]), vergeZ(verges, p[1], p[2]) or -math.huge)
          end
          local along = 0
          for k = 1, n - 1 do
            local a, b = run[k], run[k + 1]
            local len = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
            local za, zb = base[k], base[k + 1]
            local u0, u1 = along / 3, (along + len) / 3
            for side = -1, 1, 2 do
              local ax, ay = a[1] + NX[k] * half * side, a[2] + NY[k] * half * side
              local bx, by = b[1] + NX[k + 1] * half * side, b[2] + NY[k + 1] * half * side
              at(1, ax, ay, za - T.sink, u0, 0); at(2, bx, by, zb - T.sink, u1, 0)
              at(3, bx, by, zb + h, u1, (h + T.sink) / 3); at(4, ax, ay, za + h, u0, (h + T.sink) / 3)
              emit(ctx.wall, kind, (NX[k] + NX[k + 1]) * side, (NY[k] + NY[k + 1]) * side, 0)
            end
            -- the top
            at(1, a[1] - NX[k] * half, a[2] - NY[k] * half, za + h, u0, 0); at(2, b[1] - NX[k + 1] * half, b[2] - NY[k + 1] * half, zb + h, u1, 0)
            at(3, b[1] + NX[k + 1] * half, b[2] + NY[k + 1] * half, zb + h, u1, .07); at(4, a[1] + NX[k] * half, a[2] + NY[k] * half, za + h, u0, .07)
            emit(ctx.wall, kind, 0, 0, 1)
            tris = tris + 6
            -- posts every `post` m from the run's start (owned by the tile holding the post's foot)
            if not ctx.far then
              local s = math.ceil(along / T.post) * T.post
              while s < along + len do
                local t = (s - along) / len
                local x, y, z = a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, za + (zb - za) * t
                if inside(rect, x, y) then
                  local ux, uy = (b[1] - a[1]) / len, (b[2] - a[2]) / len
                  local nx, ny = -uy, ux
                  local hb, ha = half + T.postProud, T.postHalf
                  -- its two faces over the panels (its sides stand out postProud m: not drawn)
                  for f = 3, 4 do
                    local sa, sb = (f == 1 and 1) or (f == 2 and -1) or 0, (f == 3 and 1) or (f == 4 and -1) or 0
                    local cx, cy = x + ux * ha * sa + nx * hb * sb, y + uy * ha * sa + ny * hb * sb
                    local px, py, w = sa ~= 0 and nx or ux, sa ~= 0 and ny or uy, sa ~= 0 and hb or ha
                    at(1, cx - px * w, cy - py * w, z - T.sink, 0, 0); at(2, cx + px * w, cy + py * w, z - T.sink, 1, 0)
                    at(3, cx + px * w, cy + py * w, z + h + .1, 1, 1); at(4, cx - px * w, cy - py * w, z + h + .1, 0, 1)
                    emit(ctx.decor, 'clad_metal', ux * sa + nx * sb, uy * sa + ny * sb, 0)
                  end
                  tris = tris + 4
                end
                s = s + T.post
              end
            end
            along = along + len
          end
          -- the run's two ends
          for e = 1, n, n - 1 do
            local p = run[e]
            local o = e == 1 and run[2] or run[n - 1]
            local ox, oy = p[1] - o[1], p[2] - o[2]
            at(1, p[1] - NX[e] * half, p[2] - NY[e] * half, base[e] - T.sink, 0, 0)
            at(2, p[1] + NX[e] * half, p[2] + NY[e] * half, base[e] - T.sink, .07, 0)
            at(3, p[1] + NX[e] * half, p[2] + NY[e] * half, base[e] + h, .07, 1)
            at(4, p[1] - NX[e] * half, p[2] - NY[e] * half, base[e] + h, 0, 1)
            emit(ctx.wall, kind, ox, oy, 0)
            tris = tris + 2
          end
        end
        done = done + 1
        if done % 4 == 0 and ctx.pause then ctx.pause() end
      end
    end
  end
  return tris
end

return M
