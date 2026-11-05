-- Traffic signals and speed cameras of a near tile: OSM highway=traffic_signals and highway=speed_camera
-- nodes, the cameras the exporter made of enforcement=maxspeed devices (fast_signals.py), and the signal
-- approaches of the swept roads' junction blocks. Static: no phases, no AI signals.
--
-- Stock shapes, referenced by path and never copied. A level's shapes may be missing (a level not
-- installed, or not mounted): every shape is checked with FS:fileExists once and an assembly falls back
-- to the next (M.kit). Outside the game (no FS) all count as present.
--
-- Placement only: generator.lua hands in the tile's points, strips, junction stars, swept plan and its
-- clearance tests (OFFROAD), and M.place calls ctx.put(name, shape, x, y, z, yaw, fields, solid) for
-- every shape. yaw as generator.lua's static(): local X turns to (cos, -sin), local Y to (sin, cos).
local M = {}

local sqrt, atan2, max, min = math.sqrt, math.atan2, math.max, math.min
local NONE = {}

local OBJ, WCO = '/art/shapes/objects/', '/levels/west_coast_usa/art/shapes/objects/'
M.SHAPES = {
  -- the game's own signal boom (pole, mast arm and heads in one shape, common art): the editor's traffic
  -- manager ("Use Default", editor/trafficManager.lua) stands it (road radius + 1) m right of the
  -- approach's centre at the stop point, its +Y along the travel
  boom = OBJ .. 's_trafficlight_boom_sn.dae',
  -- a 3.47 m pole and a head 0.9 m tall, lenses on +X, origin at its centre (measured from the DAEs);
  -- West Coast USA stands trafficlight_city2 heads on pole_traffic1 2.97-3.01 m over the pole's foot and
  -- 0.16-0.35 m in front of its axis
  pole = WCO .. 'pole_traffic1.dae',
  heads = {OBJ .. 'trafficlight_city1.dae', WCO .. 'trafficlight_city1.dae'},
  -- West Coast USA's speed camera: the pole, the elbow 4.188 m and the camera 4.988 m over its foot, all
  -- turned alike (its speedCamera groups); the lens looks along -Y, at the stretch the trigger checks
  camera = {WCO .. 's_camera_pole.dae', WCO .. 's_camera_elbow.dae', WCO .. 's_camera_speed1.dae'},
}
M.TUNE = {
  headUp = 3.0, headAhead = .25,  -- the head over the pole's foot, and ahead of its axis toward the traffic
  boomOut = 1,                    -- m past the carriageway's half width (the traffic manager's default)
  poleOut = .7, poleVerge = 1.2,  -- m past the kerb into a sidewalk, or onto the verge (the lamps' rule)
  boomWidth = 9,                  -- a carriageway this wide (m), or two lanes into the junction, gets the boom
  cameraOut = 1.5,                -- m past the kerb or the sidewalk: on the verge
  cameraUp = {0, 4.188, 4.988},
  cameraAim = 25,                 -- m upstream: the lens looks at the lanes there
  reach = 12,                     -- m from a centre line at which a node still belongs to that road
  junction = 2,                   -- m: a signal node this near a junction's node is the junction's
  near = 40,                      -- m: a signal with no direction this near a junction controls the way into it
  dedupe = 15,                    -- m: one signal per approach and direction (OSM maps a junction's and its stop lines')
  sweptStop = 40,                 -- m: a swept road's signal node this near a block's signal approach is that approach's
  top = 8,                        -- m: no pole under a deck lower than this over its foot
}
-- The lamps a static head shows (core_trafficSignals sets instanceColor, instanceColor1, instanceColor2
-- top to bottom, on '1 1 1 1', off '0 0 0 1'). A Forest item has no instance colour and lit every lamp,
-- so the heads and booms are TSStatics. The road of a junction's highest class shows green, the others red.
M.LAMPS = {
  red = {instanceColor = '1 1 1 1', instanceColor1 = '0 0 0 1', instanceColor2 = '0 0 0 1'},
  green = {instanceColor = '0 0 0 1', instanceColor1 = '0 0 0 1', instanceColor2 = '1 1 1 1'},
}

local seen = {}
local function have(path)
  local found = seen[path]
  if found == nil then
    found = true
    if FS then
      local ok, exists = pcall(function() return FS:fileExists(path) end)
      found = ok and exists and true or false
    end
    seen[path] = found
  end
  return found
end

-- The assemblies this game has: {boom, pole, head, camera (true when all its parts are there), missing = {paths}}
function M.kit()
  if M.cached then return M.cached end
  local kit = {missing = {}}
  local function check(path)
    if have(path) then return path end
    kit.missing[#kit.missing + 1] = path
  end
  kit.boom = check(M.SHAPES.boom)
  for _, h in ipairs(M.SHAPES.heads) do if have(h) then kit.head = h; break end end
  if not kit.head then kit.missing[#kit.missing + 1] = M.SHAPES.heads[1] end
  kit.pole = check(M.SHAPES.pole)
  kit.camera = true
  for _, p in ipairs(M.SHAPES.camera) do if not check(p) then kit.camera = false end end
  M.cached = kit
  return kit
end

-- the heading in a point's tags (fast_signals.HEADING, or a camera's degrees or cardinal direction) as a
-- unit vector in the region's frame (y north), else nil
local CARDINAL = {N = 0, NNE = 22.5, NE = 45, ENE = 67.5, E = 90, ESE = 112.5, SE = 135, SSE = 157.5,
  S = 180, SSW = 202.5, SW = 225, WSW = 247.5, W = 270, WNW = 292.5, NW = 315, NNW = 337.5}
function M.heading(tags)
  for _, key in ipairs({'e2r:heading', 'direction'}) do
    local value = tags and tags[key]
    if value then
      local degrees = CARDINAL[tostring(value):upper()] or tonumber(value)
      if degrees and degrees == degrees and degrees ~= math.huge and degrees ~= -math.huge then
        local r = math.rad(degrees)
        return math.sin(r), math.cos(r)
      end
    end
  end
  return nil
end

local function unit(x, y)
  local l = sqrt(x * x + y * y)
  if l < 1e-9 then return nil end
  return x / l, y / l
end

-- the travel directions at a road: {sign...} (+1 along tx, ty) from a heading, the road's one-way, or both
local function directions(hx, hy, tx, ty, oneway, reversed)
  if hx then return {(hx * tx + hy * ty) >= 0 and 1 or -1} end
  if oneway then return {reversed and -1 or 1} end
  return {1, -1}
end

-- Off the drawn asphalt (half a metre at a time along nx, ny, 8 m at most), off every other road's centre
-- line at grade and on decks, from under a deck and off the swept surfaces (OFFROAD's lamp tests):
-- x, y, z, or nil
local function clear(ctx, x, y, nx, ny, except)
  local O = ctx.offroad
  for _ = 1, 16 do
    if not O.nearAsphalt(x, y, .3) then break end
    x, y = x + nx * .5, y + ny * .5
  end
  if O.nearAsphalt(x, y, .3) then return nil end
  local z = ctx.ground(x, y)
  if O.nearestOn(x, y, z, false, except) or O.nearestOn(x, y, z, true, except) then return nil end
  if O.underDeck(x, y, z, M.TUNE.top) then return nil end
  return O.sweptClear(x, y, z, nx, ny)
end

local function inside(ctx, x, y)
  local r = ctx.rect
  return x >= r[1] and x < r[3] and y >= r[2] and y < r[4]
end

local function safe(name) return (tostring(name):gsub('[^%w_]', '_')) end

-- One signal: q the stop point on the centre line, tx, ty the travel there, out the metres to the
-- carriageway's right edge, walk the sidewalk there, wide (a boom), lamps ('red'/'green').
-- placed: the signals so far ({x, y, tx, ty}); nothing twice for one approach.
local function signal(ctx, name, qx, qy, tx, ty, out, walk, wide, lamps, except)
  local T, kit = M.TUNE, ctx.kit
  for _, p in ipairs(ctx.placed) do
    if (p[1] - qx) ^ 2 + (p[2] - qy) ^ 2 < T.dedupe ^ 2 and p[3] * tx + p[4] * ty > .7 then return end
  end
  local rx, ry = ty, -tx
  local boom = kit.boom and (wide or not (kit.pole and kit.head))
  if not boom and not (kit.pole and kit.head) then return end
  local off = boom and out + T.boomOut or out + (walk > 0 and T.poleOut or T.poleVerge)
  local x, y, z = clear(ctx, qx + rx * off, qy + ry * off, rx, ry, except)
  if not x then return end
  ctx.placed[#ctx.placed + 1] = {qx, qy, tx, ty}
  if not inside(ctx, x, y) then return end
  local fields = M.LAMPS[lamps] or M.LAMPS.red
  if boom then
    ctx.put('sigb_' .. name, kit.boom, x, y, z, atan2(tx, ty), fields, false)
  else
    ctx.put('sigp_' .. name, kit.pole, x, y, z, atan2(ty, -tx), nil, true)
    ctx.put('sigh_' .. name, kit.head, x - tx * T.headAhead, y - ty * T.headAhead, z + T.headUp, atan2(ty, -tx), fields, false)
  end
  ctx.count = ctx.count + 1
end

-- One camera on the verge right of the travel tx, ty at q, the lens to the lanes cameraAim m upstream.
local function camera(ctx, name, qx, qy, tx, ty, out, walk, lane, except)
  local T = M.TUNE
  local rx, ry = ty, -tx
  local off = out + walk + T.cameraOut
  local x, y, z = clear(ctx, qx + rx * off, qy + ry * off, rx, ry, except)
  if not x or not inside(ctx, x, y) then return end
  local ax, ay = qx - tx * T.cameraAim + rx * lane, qy - ty * T.cameraAim + ry * lane
  local yaw = atan2(-(ax - x), -(ay - y))
  for k, shape in ipairs(M.SHAPES.camera) do
    ctx.put('cam' .. k .. '_' .. name, shape, x, y, z + T.cameraUp[k], yaw, nil, true)
  end
  ctx.count = ctx.count + 1
end

-- The strips that end at a junction's node: {strip, k (its end), tx, ty (into the junction), dir (+1: travel
-- along the strip's points)}, and the junction's highest road class
local function mouths(ctx, star)
  local cx, cy = star.node[1], star.node[2]
  if star.mouths then return star.mouths, star.rank end
  local out, rank = {}, -math.huge
  for _, strip in ipairs(ctx.strips) do
    local pts, full = strip.points, strip.full or strip.points
    if strip.surface and #pts >= 2 then
      for _, e in ipairs({{1, 2, full[1], -1}, {#pts, #pts - 1, full[#full], 1}}) do
        local node = e[3]
        if node and (node[1] - cx) ^ 2 + (node[2] - cy) ^ 2 < 1 then
          local p, q = pts[e[1]], pts[e[2]]
          local tx, ty = unit(p[1] - q[1], p[2] - q[2])
          if tx then
            out[#out + 1] = {strip = strip, k = e[1], seg = min(e[1], e[2]), tx = tx, ty = ty, dir = e[4]}
            rank = max(rank, ctx.rankOf(strip.tags))
          end
        end
      end
    end
  end
  star.mouths, star.rank = out, rank
  return out, rank
end

-- the junction star whose node is within reach of (x, y), nearest first
local function starNear(ctx, x, y, reach)
  local best, hit = reach * reach, nil
  for _, star in ipairs(ctx.stars) do
    local d = (star.node[1] - x) ^ 2 + (star.node[2] - y) ^ 2
    if d < best then best, hit = d, star end
  end
  return hit, sqrt(best)
end

local function wideRoad(tags, hw, sign)
  local lanes = tonumber(tags and (sign > 0 and tags['lanes:forward'] or tags['lanes:backward']) or nil)
  if not lanes and tags and tags.lanes then
    local all = tonumber(tags.lanes)
    local one = tags.oneway == 'yes' or tags.oneway == '1' or tags.oneway == '-1'
    lanes = all and (one and all or all / 2)
  end
  return 2 * hw >= M.TUNE.boomWidth or (lanes or 1) >= 2
end

-- A junction node's signal: one per approach (a strip into the node that its one-way allows)
local function junctionSignals(ctx, point, star)
  local list, rank = mouths(ctx, star)
  for n, m in ipairs(list) do
    local strip = m.strip
    local one, rev = ctx.oneway(strip.tags)
    local ok = not one or (m.dir > 0) == not rev
    if ok then
      local p = strip.points[m.k]
      local side = m.dir > 0 and -1 or 1
      local walk = strip.walk[m.seg] and strip.walk[m.seg][side] and ctx.sidewalk or 0
      signal(ctx, safe(point.id) .. '_' .. n, p[1], p[2], m.tx, m.ty, strip.hw, walk,
        wideRoad(strip.tags, strip.hw, m.dir), ctx.rankOf(strip.tags) >= rank and 'green' or 'red', strip.road)
    end
  end
end

-- the green or red of a signal on a road of class `rank` at (x, y): green where no junction is near or
-- its road is the junction's highest
local function lampsAt(ctx, x, y, rank)
  local star = starNear(ctx, x, y, M.TUNE.near)
  if not star then return 'green' end
  local _, top = mouths(ctx, star)
  return rank >= top and 'green' or 'red'
end

-- A signal or camera node on a generator road at grade
local function onRoad(ctx, point, p, hit)
  local T = M.TUNE
  local tags = hit.road.tags or NONE
  local tx, ty = hit.ny, -hit.nx
  local hx, hy = M.heading(point.tags)
  local one, rev = ctx.oneway(tags)
  local rank = ctx.rankOf(tags)
  if point.kind == 'signal' then
    local list = directions(hx, hy, tx, ty, one, rev)
    -- no direction, near a junction: the way into it
    if not hx and not one then
      local star, d = starNear(ctx, hit.x, hit.y, T.near)
      if star and d > T.junction then
        list = {((star.node[1] - hit.x) * tx + (star.node[2] - hit.y) * ty) >= 0 and 1 or -1}
      end
    end
    for n, sign in ipairs(list) do
      local walk = ctx.offroad.walkAt(hit.road, hit.x, hit.y, -sign)
      signal(ctx, safe(point.id) .. '_' .. n, hit.x, hit.y, tx * sign, ty * sign, hit.hw, walk,
        wideRoad(tags, hit.hw, sign), lampsAt(ctx, hit.x, hit.y, rank), hit.road)
    end
  else
    local sign
    if hx then sign = (hx * tx + hy * ty) >= 0 and 1 or -1
    elseif one then sign = rev and -1 or 1
    else
      -- beside the road: the traffic whose right side it stands on; on the centre line, the way's direction
      local u = (p[1] - hit.x) * hit.nx + (p[2] - hit.y) * hit.ny
      sign = u > hit.hw * .5 and -1 or 1
    end
    local walk = ctx.offroad.walkAt(hit.road, hit.x, hit.y, -sign)
    camera(ctx, safe(point.id), hit.x, hit.y, tx * sign, ty * sign, hit.hw, walk, one and 0 or hit.hw / 2, hit.road)
  end
end

-- A signal or camera node on a swept road: S at station s
local function onSwept(ctx, point, p, S, s)
  local st = S:at(s)
  if not st then return end
  local tx, ty = math.cos(st.h), math.sin(st.h)
  local hx, hy = M.heading(point.tags)
  local function out(sign) return sign > 0 and st.eR or st.eL end
  if point.kind == 'signal' then
    -- a block's signal approach near it stands for it
    for _, ap in ipairs(S.approaches or NONE) do
      if ap.control == 'signals' and math.abs(ap.stop - s) < M.TUNE.sweptStop then return end
    end
    for n, sign in ipairs(directions(hx, hy, tx, ty, S.oneway, false)) do
      signal(ctx, safe(point.id) .. '_' .. n, st.x, st.y, tx * sign, ty * sign, out(sign), 0,
        wideRoad(nil, (st.eL + st.eR) / 2, sign), 'green', nil)
    end
  else
    local sign
    if hx then sign = (hx * tx + hy * ty) >= 0 and 1 or -1
    elseif S.oneway then sign = 1
    else
      local u = (p[1] - st.x) * -ty + (p[2] - st.y) * tx
      sign = u > 0 and -1 or 1
    end
    camera(ctx, safe(point.id), st.x, st.y, tx * sign, ty * sign, out(sign), 0, S.oneway and 0 or out(sign) / 2, nil)
  end
end

-- The swept junction blocks' signal approaches: a signal at each stop line, right of the travel
local function sweptApproaches(ctx)
  local T = M.TUNE
  local strokes = {}
  for sid, S in pairs(ctx.SW.strokes or NONE) do
    if S.approaches then strokes[#strokes + 1] = {sid, S} end
  end
  table.sort(strokes, function(a, b) return tostring(a[1]) < tostring(b[1]) end)
  local tops = {}
  for _, e in ipairs(strokes) do
    for _, ap in ipairs(e[2].approaches) do
      if ap.control == 'signals' then
        tops[ap.J] = max(tops[ap.J] or -math.huge, ctx.rankOf({highway = e[2].highway}))
      end
    end
  end
  for _, e in ipairs(strokes) do
    local sid, S = e[1], e[2]
    local rank = ctx.rankOf({highway = S.highway})
    for _, ap in ipairs(S.approaches) do
      local st = ap.control == 'signals' and S:at(ap.stop)
      local r = ctx.rect
      if st and st.x > r[1] - T.near and st.x < r[3] + T.near and st.y > r[2] - T.near and st.y < r[4] + T.near then
        local sigma = ap.sigma >= 0 and 1 or -1
        local tx, ty = math.cos(st.h) * sigma, math.sin(st.h) * sigma
        local out = sigma > 0 and st.eR or st.eL
        signal(ctx, safe(ap.J.id) .. '_' .. safe(sid) .. '_' .. (sigma > 0 and 'f' or 'b'), st.x, st.y, tx, ty, out, 0,
          (ap.lanes or 1) >= 2 or 2 * out >= T.boomWidth, rank >= (tops[ap.J] or rank) and 'green' or 'red', nil)
      end
    end
  end
end

-- ctx: rect, points, strips, stars, SW (the swept plan or nil), sweep (sweep.lua), offroad (generator's
-- OFFROAD), ground(x, y), sidewalk (m), oneway(tags), rankOf(tags), put(...), pause.
-- Returns the number of signals and cameras placed.
function M.place(ctx)
  ctx.kit, ctx.placed, ctx.count = M.kit(), {}, 0
  if ctx.SW and ctx.sweep then sweptApproaches(ctx) end
  local cameras = ctx.kit.camera
  -- the junction nodes' signals first: a stop line's own node then adds nothing for that approach
  local order = {}
  for _, point in ipairs(ctx.points) do
    if point.pos and (point.kind == 'signal' or (point.kind == 'speed_camera' and cameras)) then
      local star = point.kind == 'signal' and starNear(ctx, point.pos[1], point.pos[2], M.TUNE.junction)
      order[#order + 1] = {point, star}
    end
  end
  table.sort(order, function(a, b)
    if (a[2] ~= nil and a[2] ~= false) ~= (b[2] ~= nil and b[2] ~= false) then return a[2] and true or false end
    return tostring(a[1].id) < tostring(b[1].id)
  end)
  for _, e in ipairs(order) do
    local point, star = e[1], e[2]
    local p = point.pos
    if star then
      junctionSignals(ctx, point, star)
    else
      local z = ctx.ground(p[1], p[2])
      local hit = ctx.offroad.nearestOn(p[1], p[2], z, false, nil, M.TUNE.reach)
      if hit then
        onRoad(ctx, point, p, hit)
      elseif ctx.SW and ctx.sweep then
        local _, S, s = ctx.sweep.surfaceWhere(ctx.SW, p[1], p[2], 2)
        if S then onSwept(ctx, point, p, S, s) end
      end
    end
    ctx.pause()
  end
  return ctx.count
end

return M
