-- Power lines and wind turbines (earthng/power.py): runs of OSM power=line with their towers'
-- facts per vertex (tag 'e2r:towers'), and power=generator wind turbines, as procedural meshes of
-- few triangles in the near scene and the far layer (generator.lua build). They are seen from a
-- few hundred metres out to the far ring, so a lattice tower is legs, cross bracing and arms as
-- boxes and single quads, not its real members. No stock pole: the game has no lattice pylon.
--
-- A tile draws the towers that stand in it and the spans whose midpoint lies in it, so a span is
-- drawn once however many cells hold its run. Both towers of a span are worked out from the run
-- alone (the vertex, its neighbours, the tower facts, the line's tags): the tile that draws a
-- tower and the tile that draws its span hang the wire on the same point. Heights are the
-- exporter's DEM samples of the vertices for that reason, not the tile's own ground.
local M = {}
local EPS = 1e-6
local sqrt, floor, cos, sin, pi = math.sqrt, math.floor, math.cos, math.sin, math.pi

-- German mast types (OSM design / line_arrangement) as the arms they carry: z as a share of the
-- tower's height, reach as a share of the main reach, slots as shares of the reach (each side).
-- Donau: two levels, two conductors a side below and one above. Einebenen: all on one arm.
-- Tonne (barrel): three levels, one a side, the middle arm the widest.
local DESIGNS = {
  donau = {body = .70, top = .93, arms = {{z = .70, reach = 1, slots = {1, .5}}, {z = .85, reach = .58, slots = {1}}}},
  one = {body = .88, top = .93, arms = {{z = .88, reach = 1.35, slots = {1, .66, .33}}}},
  barrel = {body = .64, top = .94, arms = {{z = .64, reach = .62, slots = {1}}, {z = .77, reach = .82, slots = {1}},
    {z = .90, reach = .62, slots = {1}}}},
}
-- OSM values to the three shapes; anything else (asymmetric, y-frame, ...) is drawn as Donau, the
-- commonest German tower of 110 to 380 kV
local DESIGN_OF = {donau = 'donau', donau_inverse = 'donau', ['two-level'] = 'donau', triangle = 'donau', mixed = 'donau',
  ['one-level'] = 'one', flat = 'one', delta = 'one', portal = 'one', ['h-frame'] = 'one',
  barrel = 'barrel', ['three-level'] = 'barrel', ['six-level'] = 'barrel', vertical = 'barrel'}
-- prevailing wind over northern Germany is from the west-south-west: every rotor faces it (they
-- turn together in the wind, so a field of them pointing anywhere read as wrong)
M.WIND_FROM = 250
M.DESIGNS, M.DESIGN_OF = DESIGNS, DESIGN_OF

-- the first number in a tag ('120', '120 m', '6;6' gives 6)
local function number(value)
  if type(value) == 'number' then return value end
  if type(value) ~= 'string' then return nil end
  return tonumber(value:match('^%s*(%d+%.?%d*)'))
end

-- the highest voltage of a line in kV (OSM gives volts; a small number is read as kV already)
local function kvOf(tags)
  local best
  for part in tostring(tags.voltage or ''):gmatch('[^;]+') do
    local v = number(part)
    if v then
      if v >= 1000 then v = v / 1000 end
      if not best or v > best then best = v end
    end
  end
  return math.max(10, math.min(1000, best or 110))
end
M.kvOf = kvOf

-- conductors a span carries: the sum over cables=6;6, else two circuits (one under 45 kV)
local function cablesOf(tags, kv)
  local total
  for part in tostring(tags.cables or ''):gmatch('[^;]+') do
    local v = number(part)
    if v then total = (total or 0) + v end
  end
  return math.max(1, math.min(24, floor(total or (kv < 45 and 3 or 6))))
end

local function defaultHeight(kv)
  if kv >= 300 then return 58 elseif kv >= 200 then return 50 elseif kv >= 100 then return 38
  elseif kv >= 45 then return 28 end
  return 12
end

-- 'kind,design,height' per vertex (power.py tower_attr)
local function towerFacts(tags, n)
  local out, k = {}, 0
  for field in (tostring(tags['e2r:towers'] or '') .. '|'):gmatch('([^|]*)|') do
    k = k + 1
    local kind, design, height = field:match('^([^,]*),([^,]*),([^,]*)$')
    out[k] = {kind = kind or '', design = design or '', height = number(height)}
  end
  for j = k + 1, n do out[j] = {kind = '', design = ''} end
  return out
end

-- The tower at vertex k of a run: where it stands, which way its arms point (across the bisector
-- of its spans, the left of the way's direction first), its shape and the points the wires hang on.
local function towerAt(pts, k, facts, line)
  local p = pts[k]
  local prev, nxt = pts[k - 1], pts[k + 1]
  local tx, ty = 0, 0
  for _, pair in ipairs({{prev, p}, {p, nxt}}) do
    local a, b = pair[1], pair[2]
    if a and b then
      local dx, dy = b[1] - a[1], b[2] - a[2]
      local l = sqrt(dx * dx + dy * dy)
      if l > EPS then tx, ty = tx + dx / l, ty + dy / l end
    end
  end
  local l = sqrt(tx * tx + ty * ty)
  if l < EPS then tx, ty, l = 1, 0, 1 end
  tx, ty = tx / l, ty / l
  local fact = facts[k] or {}
  local kv = line.kv
  local pole = fact.kind == 'p' or (fact.kind == '' and kv < 45)
  local h = math.max(8, math.min(150, fact.height or (pole and math.min(defaultHeight(kv), 25) or defaultHeight(kv))))
  local design = DESIGN_OF[fact.design] or DESIGN_OF[line.design] or (fact.kind == 'o' and 'one')
    or (line.cables <= 4 and 'one' or 'donau')
  local t = {x = p[1], y = p[2], z = p[3] or 0, tx = tx, ty = ty, ax = -ty, ay = tx, h = h, kv = kv, pole = pole,
    design = design, slots = {}}
  local reach = 3 + kv * .026
  t.ins = .8 + kv * .011
  if pole then
    -- a crossarm a metre under the top with a conductor at each end, the third on the top
    local r = 1 + kv * .03
    t.arms = {{z = h - 1, reach = r}}
    t.slots = {{r, h - 1 + .3}, {-r, h - 1 + .3}, {0, h + .3}}
    t.ins = 0
    return t
  end
  local shape = DESIGNS[design]
  t.shape = shape
  t.base = math.max(1.5, .08 * h + .5)
  t.waist = math.max(.6, .02 * h + .6)
  t.arms = {}
  for _, arm in ipairs(shape.arms) do
    t.arms[#t.arms + 1] = {z = arm.z * h, reach = arm.reach * reach}
  end
  -- slots bottom up, the left side's outer slot first: a single circuit hangs on one side
  for _, side in ipairs({1, -1}) do
    for _, arm in ipairs(shape.arms) do
      for _, s in ipairs(arm.slots) do
        t.slots[#t.slots + 1] = {side * s * arm.reach * reach, arm.z * h - t.ins}
      end
    end
  end
  t.peak = kv >= 60 and h or nil
  return t
end
M.towerAt = towerAt

local function world(t, s, z)
  return {t.x + t.ax * s, t.y + t.ay * s, t.z + z}
end

-- Emitters. ctx.triangle is the scene collector's (generator.lua collector); ctx.tris counts.
local function tri(ctx, kind, a, b, c)
  ctx.triangle(kind, a, b, c)
  ctx.tris = ctx.tris + 1
end

-- a quad whose face points along (ox, oy, oz)
local function face(ctx, kind, a, b, c, d, ox, oy, oz)
  local ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
  local nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
  if nx * ox + ny * oy + nz * oz < 0 then a, b, c, d = d, c, b, a end
  tri(ctx, kind, a, b, c)
  tri(ctx, kind, a, c, d)
end

local function cross(ax, ay, az, bx, by, bz)
  return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
end

local function unit(x, y, z)
  local l = sqrt(x * x + y * y + z * z)
  if l < EPS then return 0, 0, 0, 0 end
  return x / l, y / l, z / l, l
end

-- a square tube from p to q, w each side of its axis (no caps)
local function bar(ctx, kind, p, q, w)
  local dx, dy, dz, l = unit(q[1] - p[1], q[2] - p[2], q[3] - p[3])
  if l < EPS then return end
  local ux, uy, uz
  if math.abs(dz) < .9 then ux, uy, uz = unit(-dy, dx, 0) else ux, uy, uz = 1, 0, 0 end
  local vx, vy, vz = cross(dx, dy, dz, ux, uy, uz)
  local corners = {{ux + vx, uy + vy, uz + vz}, {-ux + vx, -uy + vy, -uz + vz}, {-ux - vx, -uy - vy, -uz - vz}, {ux - vx, uy - vy, uz - vz}}
  for k = 1, 4 do
    local c, e = corners[k], corners[k % 4 + 1]
    face(ctx, kind, {p[1] + c[1] * w, p[2] + c[2] * w, p[3] + c[3] * w}, {p[1] + e[1] * w, p[2] + e[2] * w, p[3] + e[3] * w},
      {q[1] + e[1] * w, q[2] + e[2] * w, q[3] + e[3] * w}, {q[1] + c[1] * w, q[2] + c[2] * w, q[3] + c[3] * w},
      c[1] + e[1], c[2] + e[2], c[3] + e[3])
  end
end

-- a flat strip from p to q, w each side, spread along (sx, sy, sz) (two-sided material)
local function strip(ctx, kind, p, q, w, sx, sy, sz)
  sx, sy, sz = sx * w, sy * w, sz * w
  tri(ctx, kind, {p[1] + sx, p[2] + sy, p[3] + sz}, {p[1] - sx, p[2] - sy, p[3] - sz}, {q[1] - sx, q[2] - sy, q[3] - sz})
  tri(ctx, kind, {p[1] + sx, p[2] + sy, p[3] + sz}, {q[1] - sx, q[2] - sy, q[3] - sz}, {q[1] + sx, q[2] + sy, q[3] + sz})
end

-- a strip in the plane with normal n, across its own direction
local function inPlane(ctx, kind, p, q, w, nx, ny, nz)
  local dx, dy, dz, l = unit(q[1] - p[1], q[2] - p[2], q[3] - p[3])
  if l < EPS then return end
  local sx, sy, sz = unit(cross(nx, ny, nz, dx, dy, dz))
  strip(ctx, kind, p, q, w, sx, sy, sz)
end

-- The tower as drawn: four legs to the lowest arm and on to the top, X bracing on the four faces,
-- each arm a box with a strut under it, the insulators (near only) and the earth-wire peak.
local function drawTower(ctx, t, bottom)
  local steel = 'power_steel'
  if t.pole then
    bar(ctx, steel, {t.x, t.y, bottom}, {t.x, t.y, t.z + t.h}, .22)
    local arm = t.arms[1]
    bar(ctx, steel, world(t, -arm.reach - .2, arm.z), world(t, arm.reach + .2, arm.z), .1)
    return
  end
  local h, base, waist, leg = t.h, t.base, t.waist, .12 + .004 * t.h
  local zBody, zTop = t.shape.body * h, t.shape.top * h
  local ax, ay, tx, ty = t.ax, t.ay, t.tx, t.ty
  local function corner(sa, st, half, z)
    return {t.x + (ax * sa + tx * st) * half, t.y + (ay * sa + ty * st) * half, t.z + z}
  end
  local function half(z)
    if z <= zBody then return base + (waist - base) * math.max(0, z) / zBody end
    return waist - .25 * waist * (z - zBody) / math.max(EPS, zTop - zBody)
  end
  local below = bottom - t.z
  for _, c in ipairs({{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}) do
    bar(ctx, steel, corner(c[1], c[2], base + (base - waist) * (-below) / zBody, below), corner(c[1], c[2], waist, zBody), leg)
    bar(ctx, steel, corner(c[1], c[2], waist, zBody), corner(c[1], c[2], half(zTop), zTop), leg * .8)
  end
  -- X panels about as tall as the body is wide; the far ring has half as many
  local panels = math.max(2, math.min(6, floor(zBody / (base + waist) + .5)))
  if ctx.far then panels = math.max(2, floor(panels / 2)) end
  local sections = {{0, zBody, panels}}
  if not ctx.far then sections[2] = {zBody, zTop, 1} end
  for _, sec in ipairs(sections) do
    local z0, z1, n = sec[1], sec[2], sec[3]
    for k = 0, n - 1 do
      local za, zb = z0 + (z1 - z0) * k / n, z0 + (z1 - z0) * (k + 1) / n
      local ha, hb = half(za), half(zb)
      -- faces across the arms (normal +-t) and along them (normal +-a)
      for _, f in ipairs({{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) do
        local nx, ny = ax * f[1] + tx * f[2], ay * f[1] + ty * f[2]
        local function c(s, hh, z)
          if f[1] ~= 0 then return corner(f[1], s, hh, z) end
          return corner(s, f[2], hh, z)
        end
        inPlane(ctx, steel, c(-1, ha, za), c(1, hb, zb), leg * .55, nx, ny, 0)
        inPlane(ctx, steel, c(1, ha, za), c(-1, hb, zb), leg * .55, nx, ny, 0)
      end
    end
  end
  for _, arm in ipairs(t.arms) do
    local w = half(arm.z)
    for _, side in ipairs({1, -1}) do
      bar(ctx, steel, world(t, side * w, arm.z), world(t, side * arm.reach, arm.z), leg * .8)
      -- the strut from the body under the arm to its end: the triangle that reads as a truss
      local drop = math.min(.3 * arm.reach, arm.z - 1)
      local a, b = world(t, side * half(arm.z - drop), arm.z - drop), world(t, side * arm.reach, arm.z)
      inPlane(ctx, steel, a, b, leg * .5, tx, ty, 0)
    end
  end
  if not ctx.far and t.ins > 0 then
    for k = 1, math.min(#t.slots, t.used or #t.slots) do
      local s = t.slots[k]
      local top, low = world(t, s[1], s[2] + t.ins), world(t, s[1], s[2])
      strip(ctx, steel, top, low, .14, ax, ay, 0)
      strip(ctx, steel, top, low, .14, tx, ty, 0)
    end
  end
  if t.peak then
    bar(ctx, steel, {t.x, t.y, t.z + zTop}, {t.x, t.y, t.z + t.peak}, leg * .6)
  end
end
M.drawTower = drawTower

-- One conductor from a to b, sagging by sag at mid-span: segments of two crossed strips near
-- (one, upright, far), so a wire shows from the side and from below.
local function drawWire(ctx, a, b, sag, w, n)
  local dx, dy = b[1] - a[1], b[2] - a[2]
  local hx, hy = unit(-dy, dx, 0)
  local prev
  for k = 0, n do
    local s = k / n
    local p = {a[1] + dx * s, a[2] + dy * s, a[3] + (b[3] - a[3]) * s - 4 * sag * s * (1 - s)}
    if prev then
      strip(ctx, 'power_wire', prev, p, w, 0, 0, 1)
      if not ctx.far then strip(ctx, 'power_wire', prev, p, w, hx, hy, 0) end
    end
    prev = p
  end
end

local function inRect(rect, x, y)
  return x >= rect[1] and x < rect[3] and y >= rect[2] and y < rect[4]
end

local function line(ctx, run)
  local pts, tags = run.points or {}, run.tags or {}
  if #pts < 2 then return end
  local base, first = tostring(run.id or ''):match('^(.*):p(%d+)$')
  base, first = base or tostring(run.id or ''), tonumber(first) or 0
  local kv = kvOf(tags)
  local info = {kv = kv, cables = cablesOf(tags, kv), design = tags.design or tags.line_arrangement}
  local facts = towerFacts(tags, #pts)
  local towers = {}
  local function tower(k)
    towers[k] = towers[k] or towerAt(pts, k, facts, info)
    return towers[k]
  end
  local rect = ctx.rect
  for k = 1, #pts do
    local p = pts[k]
    local id, at = base .. ':' .. (first + k - 1), floor(p[1] * 2) .. ':' .. floor(p[2] * 2)
    if inRect(rect, p[1], p[2]) and not ctx.seen[id] and not ctx.seen[at] then
      ctx.seen[id], ctx.seen[at] = true, true
      local t = tower(k)
      t.used = info.cables
      drawTower(ctx, t, math.min(t.z, ctx.ground(p[1], p[2])) - 1.5)
      ctx.towers = ctx.towers + 1
      ctx.pause()
    end
  end
  local w, clear = .06 + kv * .00035, 6 + kv * .02
  for k = 1, #pts - 1 do
    local a, b = pts[k], pts[k + 1]
    local mx, my = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
    local id = base .. ':' .. (first + k - 1) .. '>'
    local length = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    if inRect(rect, mx, my) and not ctx.seen[id] and length > 1 then
      ctx.seen[id] = true
      local ta, tb = tower(k), tower(k + 1)
      local wires = math.min(#ta.slots, #tb.slots, info.cables)
      local ends = {}
      for n = 1, wires do
        ends[n] = {world(ta, ta.slots[n][1], ta.slots[n][2]), world(tb, tb.slots[n][1], tb.slots[n][2]), w}
      end
      if ta.peak and tb.peak then ends[#ends + 1] = {world(ta, 0, ta.peak), world(tb, 0, tb.peak), w * .6} end
      -- 3 % of the span, less where the lowest wire would come nearer the ground than `clear`
      local sag, gm = .03 * length, ctx.ground(mx, my)
      for _, e in ipairs(ends) do sag = math.min(sag, (e[1][3] + e[2][3]) / 2 - gm - clear) end
      sag = math.max(.2, sag)
      local n = ctx.far and 4 or math.max(4, math.min(12, math.ceil(length / 35)))
      for _, e in ipairs(ends) do drawWire(ctx, e[1], e[2], sag, e[3], n) end
      ctx.spans = ctx.spans + 1
      ctx.pause()
    end
  end
end

-- A wind turbine at (x, y) on ground z: an eight-sided tapered tower, the nacelle, a nose cone and
-- three blades, the rotor into the prevailing wind. OSM height is the hub height.
local function turbine(ctx, point)
  local p, tags = point.pos, point.tags or {}
  local x, y = p[1], p[2]
  local z = ctx.ground(x, y)
  local hub = math.max(20, math.min(200, number(tags.height) or 100))
  local d = math.max(10, math.min(2 * (hub - 8), number(tags['rotor:diameter']) or hub * .9))
  local r = d / 2
  local seed = (floor(x * 7) * 31 + floor(y * 13)) % 1000
  local yaw = (M.WIND_FROM + (seed % 17) - 8) * pi / 180
  local fx, fy = sin(yaw), cos(yaw)           -- towards the wind (y north, x east)
  local sx, sy = -fy, fx
  local kind = 'turbine'
  local wn = .012 * d + 1
  local top = z + hub - wn * .9
  local rb = math.max(1.5, .022 * hub + 1.2)
  local rt = rb * .45
  for k = 0, 7 do
    local a0, a1 = k * pi / 4, (k + 1) * pi / 4
    local c0, s0, c1, s1 = cos(a0), sin(a0), cos(a1), sin(a1)
    face(ctx, kind, {x + c0 * rb, y + s0 * rb, z - 1}, {x + c1 * rb, y + s1 * rb, z - 1},
      {x + c1 * rt, y + s1 * rt, top}, {x + c0 * rt, y + s0 * rt, top}, c0 + c1, s0 + s1, 0)
  end
  -- the nacelle: a box from 1.5 m upwind of the tower axis back along the wind
  local length = .1 * d + 4
  local hz = z + hub
  local function at(along, across, up) return {x + fx * along + sx * across, y + fy * along + sy * across, hz + up} end
  local f0, f1, hh = 1.5, 1.5 - length, wn * .9
  local box = {{f0, wn, hh}, {f0, -wn, hh}, {f0, -wn, -hh}, {f0, wn, -hh}, {f1, wn, hh}, {f1, -wn, hh}, {f1, -wn, -hh}, {f1, wn, -hh}}
  local c = {}
  for k, v in ipairs(box) do c[k] = at(v[1], v[2], v[3]) end
  face(ctx, kind, c[1], c[2], c[3], c[4], fx, fy, 0)
  face(ctx, kind, c[5], c[6], c[7], c[8], -fx, -fy, 0)
  face(ctx, kind, c[1], c[2], c[6], c[5], 0, 0, 1)
  face(ctx, kind, c[4], c[3], c[7], c[8], 0, 0, -1)
  face(ctx, kind, c[1], c[4], c[8], c[5], sx, sy, 0)
  face(ctx, kind, c[2], c[3], c[7], c[6], -sx, -sy, 0)
  -- the nose cone: four triangles from the front face to a point
  local nose = at(f0 + .04 * d + 1, 0, 0)
  local cone = {at(f0, wn * .8, hh * .8), at(f0, -wn * .8, hh * .8), at(f0, -wn * .8, -hh * .8), at(f0, wn * .8, -hh * .8)}
  for k = 1, 4 do tri(ctx, kind, cone[k], cone[k % 4 + 1], nose) end
  -- blades in the rotor plane, a little upwind of the nacelle, at a phase of their own
  local phase = seed / 1000 * 2 * pi / 3
  local centre = at(f0 + .6, 0, 0)
  for k = 0, 2 do
    local a = phase + k * 2 * pi / 3
    local bx, by, bz = sx * cos(a), sy * cos(a), sin(a)
    local wx, wy, wz = -sx * sin(a), -sy * sin(a), cos(a)
    local root, tip, cr, ct = 1, r, .075 * r, .025 * r
    local function q(s, wd) return {centre[1] + bx * s + wx * wd, centre[2] + by * s + wy * wd, centre[3] + bz * s + wz * wd} end
    tri(ctx, kind, q(root, cr / 2), q(root, -cr / 2), q(tip, -ct / 2))
    tri(ctx, kind, q(root, cr / 2), q(tip, -ct / 2), q(tip, ct / 2))
  end
  ctx.turbines = ctx.turbines + 1
  ctx.pause()
end

-- ctx: rect, lines and points (the tile's lists), ground(x, y), triangle(kind, a, b, c) of the
-- collector to draw into, pause, far (the far layer: fewer bracing panels and wire segments, no
-- insulators). Returns counts {towers, spans, turbines, tris}.
function M.build(ctx)
  ctx.tris, ctx.towers, ctx.spans, ctx.turbines, ctx.seen = 0, 0, 0, 0, {}
  ctx.pause = ctx.pause or function() end
  for _, run in ipairs(ctx.lines or {}) do
    if run.kind == 'power_line' then line(ctx, run) end
  end
  for _, point in ipairs(ctx.points or {}) do
    local p = point.pos
    if point.kind == 'wind_turbine' and p and inRect(ctx.rect, p[1], p[2]) then turbine(ctx, point) end
  end
  return {towers = ctx.towers, spans = ctx.spans, turbines = ctx.turbines, tris = ctx.tris}
end

return M
