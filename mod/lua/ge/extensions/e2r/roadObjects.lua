-- EarthNG road objects (prototype): carriageways as the engine's own MeshRoad objects, junction
-- surfaces as small ProceduralMesh plates, guard rails as swept meshes or stock TSStatic pieces,
-- lane lines and the AI graph as DecalRoads. Behind a switch: the generator's road meshes stay the
-- default (patches/ROAD-OBJECTS.md).
--
-- Two halves:
-- * M.plan(roads, lines, tune): pure Lua, no engine call, runs offline in LuaJIT. Pack road records
--   ({id, width, tags, nodes = {a, b}, points = {{x, y, z}, ...}}, as cellformat.cell decodes them)
--   become MeshRoad node lists (chunks), junction plates, rail lines, nav and marking lines.
-- * M.create(plan, opts) makes the engine objects (timed per kind); M.remove(handle) deletes them.
--
-- Methods (re-implemented, nothing copied): cubic Hermite through the pack's stations with shared
-- tangents at nodes (G1 where ways meet), cross-fall and banking by class and radius with linear
-- run-off, and a diverge/merge model where the slip road leaves the through carriageway's plane.
local M = {}

local sqrt, abs, floor, max, min, atan2 = math.sqrt, math.abs, math.floor, math.max, math.min, math.atan2
local cos, sin, huge = math.cos, math.sin, math.huge
local EPS = 1e-9

M.TUNE = {
  chunk = 200,                  -- m of road per MeshRoad (one object, one collision piece)
  step = 2,                     -- m between MeshRoad nodes at least (tight curves)
  straightStep = 24,            -- m between nodes on a straight
  turn = math.rad(1.0),         -- heading change between MeshRoad nodes at most
  sample = 2,                   -- m: dense sampling of the fitted line
  depth = .8,                   -- MeshRoad thickness under its top
  merge = .25,                  -- knots closer than this are one (generator ROADS.knots.merge)
  through = math.rad(35),       -- two mouths within this of straight on continue each other
  branch = math.rad(30),        -- a mouth within this of a through direction diverges or merges
  wedgeReach = 450,             -- m along a slip road searched for where it leaves its host
  goreClear = 4,                -- m between host edge and slip road edge where the gore nose stands
  starCut = .8,                 -- a star junction takes this x its widest road off every mouth
  ease = {motorway = 60, trunk = 50, link = 30, default = 25},   -- run-off / height ease length (m)
  -- cross-fall q (positive: left edge higher). One-way roads fall to the right on straights; in a
  -- curve tighter than `reverse` they bank inward at clamp(c / R, qmin, qmax). Two-way roads stay
  -- level on straights (no crown in a MeshRoad) and bank inward in curves.
  bank = {
    motorway = {q0 = .025, c = 36, qmin = .025, qmax = .06, reverse = 4000},
    trunk = {q0 = .025, c = 36, qmin = .025, qmax = .06, reverse = 4000},
    link = {q0 = .025, c = 2.5, qmin = .025, qmax = .06, reverse = 1500},
    default = {q0 = 0, c = 20, qmin = .025, qmax = .07, reverse = 800},
    radiusWindow = 30,          -- m over which the heading change gives the radius
  },
  rail = {offset = .5, height = .75, beam = .32, thick = .12, post = .14, postStep = 4, chunk = 200,
    piece = 3, classes = {motorway = true, trunk = true, motorway_link = true, trunk_link = true}},
  marks = {classes = {motorway = true, trunk = true, motorway_link = true, trunk_link = true, primary = true},
    width = .15, lift = .03, step = 10},
  -- paved shoulders (m) left and right of the direction of travel, as the pack widened the road by
  -- them (fast_sources.shoulder, generator ROADS.shoulder): the OSM line is the lanes' centre, so a
  -- one-way road's surface centre lies (right - left) / 2 to its right
  shoulders = {motorway = {.75, 2.5}, trunk = {1, 1}, primary = {.5, .5}, secondary = {.25, .25},
    motorway_link = {.25, 1}, trunk_link = {.25, 1}},
  shouldersOneway = {trunk = {.5, 1.5}, primary = {.25, .75}},
  nav = {step = 10},
}

M.MATERIALS = {
  top = 'm_asphalt_new_01', side = 'm_asphalt_new_01', bottom = 'm_asphalt_new_01',
  plate = 'm_asphalt_new_01', rail = 'm_metal_brushed', post = 'm_metal_brushed',
  line = 'm_line_white', dash = 'm_line_white_discontinue', nav = 'road_invisible',
  railShape = 'art/shapes/objects/italy_guardrails_basic.dae',
}

local LINK = {motorway_link = true, trunk_link = true, primary_link = true, secondary_link = true, tertiary_link = true}
local RANK = {motorway = 7, trunk = 6, primary = 5, secondary = 4, tertiary = 3, unclassified = 2, residential = 2}
local NO_ONEWAY = {no = true, ['false'] = true, ['0'] = true, reversible = true, alternating = true}

local function classOf(tags)
  local hw = tags.highway or ''
  return hw, LINK[hw] and 'link' or hw
end

local function rankOf(tags)
  local hw = tags.highway or ''
  local base = hw:gsub('_link$', '')
  return (RANK[base] or 1) - (LINK[hw] and .5 or 0)
end

local function onewayOf(tags)
  local o = tags.oneway
  if o and not NO_ONEWAY[o] then return o == '-1' and -1 or 1 end
  local hw = tags.highway or ''
  if hw == 'motorway' or LINK[hw] and hw ~= 'tertiary_link' and hw ~= 'secondary_link' then return 1 end
  if tags.junction == 'roundabout' then return 1 end
  return 0
end

local function lanesOf(tags, oneway)
  local n = tonumber(tags.lanes or '')
  if n and n >= 1 then return floor(n) end
  return oneway ~= 0 and 1 or 2
end

-- ---------------------------------------------------------------------------------------------
-- Plan
-- ---------------------------------------------------------------------------------------------

-- the pack's stations without knots closer than `merge` (the higher height stays; the ends stay)
local function cleanPoints(points, merge)
  local out = {}
  for k, p in ipairs(points) do
    local q = out[#out]
    local x, y, z = p[1], p[2], p[3] or 0
    if q and (x - q[1]) ^ 2 + (y - q[2]) ^ 2 < merge * merge then
      if k == #points then q[1], q[2] = x, y end
      q[3] = max(q[3], z)
    else
      out[#out + 1] = {x, y, z}
    end
  end
  return out
end

local function unit(dx, dy)
  local l = sqrt(dx * dx + dy * dy)
  if l < EPS then return 0, 0, 0 end
  return dx / l, dy / l, l
end

-- direction from a road's end into the road, over the first `reach` m of its polyline
local function into(P, e, reach)
  local n = #P
  local a = e == 1 and P[1] or P[n]
  local dist, k, step = 0, e == 1 and 2 or n - 1, e == 1 and 1 or -1
  local b = P[k]
  while b do
    dist = dist + sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    if dist >= reach then break end
    local nk = k + step
    if not P[nk] then break end
    k = nk
    b = P[k]
  end
  local x, y = unit(b[1] - a[1], b[2] - a[2])
  return x, y
end

-- Cubic Hermite through P with tangents T (unit, plan view) scaled by the chord; heights on a
-- Hermite of the grades too (C1 in z); half widths (P[k][4]) linear. Samples every `step` m at
-- most and finer where it turns. knots[k]: the sample index of P[k].
local function hermite(P, T, G, step)
  local out, knots = {}, {}
  local s = 0
  for k = 1, #P - 1 do
    local a, b = P[k], P[k + 1]
    local L = sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    local ta, tb = T[k], T[k + 1]
    local turn = abs(atan2(ta[1] * tb[2] - ta[2] * tb[1], ta[1] * tb[1] + ta[2] * tb[2]))
    local n = max(1, math.ceil(L / step), math.ceil(turn / math.rad(.5)))
    local z0, z1, g0, g1 = a[3], b[3], G[k] * L, G[k + 1] * L
    for i = (k == 1 and 0 or 1), n do
      local t = i / n
      local t2, t3 = t * t, t * t * t
      local h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
      local x = h00 * a[1] + h10 * L * ta[1] + h01 * b[1] + h11 * L * tb[1]
      local y = h00 * a[2] + h10 * L * ta[2] + h01 * b[2] + h11 * L * tb[2]
      local z = h00 * z0 + h10 * g0 + h01 * z1 + h11 * g1
      if #out > 0 then
        local q = out[#out]
        s = s + sqrt((x - q.x) ^ 2 + (y - q.y) ^ 2)
      end
      out[#out + 1] = {x = x, y = y, z = z, s = s, hw = a[4] + (b[4] - a[4]) * t}
      if i == 0 then knots[k] = #out end
      if i == n then knots[k + 1] = #out end
    end
  end
  local n = #out
  for k = 1, n do
    local o = out[k]
    if k == 1 then o.tx, o.ty = T[1][1], T[1][2]
    elseif k == n then o.tx, o.ty = T[#T][1], T[#T][2]
    else
      local a, b = out[k - 1], out[k + 1]
      o.tx, o.ty = unit(b.x - a.x, b.y - a.y)
    end
  end
  -- the tangent exactly at every knot (a shared node's direction is the same in every road there)
  for k, i in pairs(knots) do out[i].tx, out[i].ty = T[k][1], T[k][2] end
  return out, knots
end

local function headingDiff(a, b)
  return atan2(a.tx * b.ty - a.ty * b.tx, a.tx * b.tx + a.ty * b.ty)
end

-- index of the sample at or before arc length s (binary search)
local function at(D, s)
  local lo, hi = 1, #D
  if s <= D[1].s then return 1, 0 end
  if s >= D[hi].s then return hi - 1, 1 end
  while hi - lo > 1 do
    local mid = floor((lo + hi) / 2)
    if D[mid].s <= s then lo = mid else hi = mid end
  end
  local a, b = D[lo], D[lo + 1]
  return lo, (s - a.s) / max(EPS, b.s - a.s)
end

local function sampleAt(D, s)
  local k, t = at(D, s)
  local a, b = D[k], D[k + 1] or D[k]
  local tx, ty = unit(a.tx + (b.tx - a.tx) * t, a.ty + (b.ty - a.ty) * t)
  return {x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t, z = a.z + (b.z - a.z) * t, s = s,
    tx = tx, ty = ty, q = (a.q or 0) + ((b.q or 0) - (a.q or 0)) * t, hw = a.hw + (b.hw - a.hw) * t}
end

-- box filter of a sample field over length L along s (a step becomes a linear ramp of length L)
local function smooth(D, field, L)
  local n = #D
  local prefix = {0}
  for k = 2, n do prefix[k] = prefix[k - 1] + (D[k][field] + D[k - 1][field]) / 2 * (D[k].s - D[k - 1].s) end
  local function integral(s)
    local k, t = at(D, s)
    local ds = ((D[k + 1] or D[k]).s - D[k].s)
    local qa, qb = D[k][field], (D[k + 1] or D[k])[field]
    return prefix[k] + (qa + (qa + (qb - qa) * t)) / 2 * ds * t
  end
  local out = {}
  for k = 1, n do
    local a, b = max(0, D[k].s - L / 2), min(D[n].s, D[k].s + L / 2)
    out[k] = b - a > EPS and (integral(b) - integral(a)) / (b - a) or D[k][field]
  end
  for k = 1, n do D[k][field] = out[k] end
end

-- signed curvature over a window, then the cross-fall of the class with a linear run-off
local function banking(road, tune)
  local D, B = road.D, road.bank
  local w = tune.bank.radiusWindow
  local n = #D
  for k = 1, n do
    local s = D[k].s
    local a = sampleAt(D, max(0, s - w / 2))
    local b = sampleAt(D, min(D[n].s, s + w / 2))
    local ds = max(1, b.s - a.s)
    local kappa = headingDiff(a, b) / ds
    D[k].kappa = kappa
  end
  for k = 1, n do
    local kappa = D[k].kappa
    local R = abs(kappa) > 1e-7 and 1 / abs(kappa) or huge
    local q
    local inward = R < B.reverse and min(B.qmax, max(B.qmin, B.c / R)) or 0
    if road.oneway ~= 0 then
      -- falls to the right of the direction of travel on a straight (+q0 there: left edge higher)
      local turnRight = kappa * road.oneway < 0
      if turnRight then q = max(B.q0, inward)
      elseif inward > 0 then q = -inward
      else q = B.q0 end
      q = q * road.oneway
    else
      q = kappa < 0 and inward or -inward
    end
    D[k].q = q
  end
  smooth(D, 'q', road.ease)
end

-- blend a road's q (and z) towards a fixed value at one end, from `from` m in, over its ease length
local function easeEnd(road, e, q, z, from)
  local D = road.D
  local n = #D
  local total = D[n].s
  -- at most half the drawn length: the far end's own correction stays exact
  local L = min(road.ease, max(1, (total - road.cut[1] - road.cut[2]) / 2))
  local origin = e == 1 and (from or 0) or (total - (from or 0))
  local here = sampleAt(D, origin)
  local dq = q and (q - here.q) or 0
  local dz = z and (z - here.z) or 0
  for k = 1, n do
    local d = e == 1 and (D[k].s - origin) or (origin - D[k].s)
    local f = 1
    if d > 0 then
      f = d < L and (1 - d / L) or 0
      f = f * f * (3 - 2 * f)
    end
    if f > 0 then
      if q then D[k].q = D[k].q + dq * f end
      if z then D[k].z = D[k].z + dz * f end
    end
  end
end

-- the road's surface height at plan point (x, y) beside sample o: centre height + q * lateral
local function planeZ(o, x, y)
  local lat = (x - o.x) * -o.ty + (y - o.y) * o.tx
  return o.z + (o.q or 0) * lat, lat
end

-- project (x, y) onto samples C[lo..hi] ({o = sample} entries): interpolated sample, lateral, index
local function projectChain(C, x, y, hint, reach)
  local best, bi, bt = huge, 1, 0
  local lo, hi = 1, #C - 1
  if hint then lo, hi = max(1, hint - reach), min(#C - 1, hint + reach) end
  for i = lo, hi do
    local a, b = C[i].o, C[i + 1].o
    local dx, dy = b.x - a.x, b.y - a.y
    local l2 = dx * dx + dy * dy
    local t = l2 > EPS and max(0, min(1, ((x - a.x) * dx + (y - a.y) * dy) / l2)) or 0
    local d = (x - a.x - dx * t) ^ 2 + (y - a.y - dy * t) ^ 2
    if d < best then best, bi, bt = d, i, t end
  end
  local a, b = C[bi].o, C[bi + 1].o
  local o = {x = a.x + (b.x - a.x) * bt, y = a.y + (b.y - a.y) * bt, z = a.z + (b.z - a.z) * bt,
    q = a.q + (b.q - a.q) * bt, hw = a.hw + (b.hw - a.hw) * bt, s = a.s + (b.s - a.s) * bt}
  o.tx, o.ty = unit(a.tx + (b.tx - a.tx) * bt, a.ty + (b.ty - a.ty) * bt)
  local _, lat = planeZ(o, x, y)
  return o, lat, bi, C[bi].c + (C[bi + 1].c - C[bi].c) * bt
end

-- Plan: segments (pack road records between shared nodes), node kinds, strokes (segments joined
-- through way joints and through diverges/merges), fitted lines, cross-fall, junction surfaces,
-- MeshRoad chunks, rails, lane lines and nav lines.
function M.plan(records, lines, tune)
  tune = tune or M.TUNE
  local t0 = os.clock()
  local stats = {segments = 0, strokes = 0, km = 0, joints = 0, wedges = 0, stars = 0, wedgeFail = 0, ends = 0}
  -- 1. segments and their mouths at the nodes
  local segs, mouths = {}, {}
  for _, r in ipairs(records) do
    local tags = r.tags or {}
    local P = cleanPoints(r.points or {}, tune.merge)
    if #P >= 2 and r.nodes and r.nodes[1] and r.nodes[2] then
      local hw, cls = classOf(tags)
      local oneway = onewayOf(tags)
      local seg = {id = r.id, tags = tags, hw = (r.width or 7) / 2, highway = hw, cls = cls, P = P, oneway = oneway,
        lanes = lanesOf(tags, oneway), rank = rankOf(tags),
        bridge = tags.bridge ~= nil and tags.bridge ~= 'no', tunnel = tags.tunnel ~= nil and tags.tunnel ~= 'no',
        nodes = {tostring(r.nodes[1]), tostring(r.nodes[2])}, link = {}}
      segs[#segs + 1] = seg
      stats.segments = stats.segments + 1
      for e = 1, 2 do
        local id = seg.nodes[e]
        mouths[id] = mouths[id] or {}
        local dx, dy = into(P, e, 3)
        table.insert(mouths[id], {seg = seg, e = e, dx = dx, dy = dy})
      end
    end
  end
  -- 2. node kinds: end, joint (two mouths straight on), wedge (a through pair and branches that
  -- leave it at a shallow angle) or star (anything else with more than one mouth)
  local cosThrough, cosBranch = cos(tune.through), cos(tune.branch)
  local nodes = {}
  for id, list in pairs(mouths) do
    local node = {id = id, mouths = list, kind = 'end', branches = {}}
    nodes[id] = node
    if #list > 1 then
      local bi, bj, best = nil, nil, -cosThrough
      for i = 1, #list do
        for j = i + 1, #list do
          local a, b = list[i], list[j]
          local d = a.dx * b.dx + a.dy * b.dy - (a.seg.rank + b.seg.rank) * 1e-3
          if d < best then bi, bj, best = i, j, d end
        end
      end
      if bi then
        local a, b = list[bi], list[bj]
        a.through, b.through = true, true
        node.through = {a, b}
        node.line = {unit(b.dx - a.dx, b.dy - a.dy)}
      end
      local star = not bi
      for _, m in ipairs(list) do
        if not m.through then
          local host
          for _, h in ipairs(node.through or {}) do
            if m.dx * h.dx + m.dy * h.dy > cosBranch and m.seg.rank <= h.seg.rank then host = h end
          end
          if host then
            m.host = host
            node.branches[#node.branches + 1] = m
          else
            star = true
          end
        end
      end
      node.kind = star and 'star' or (#node.branches > 0 and 'wedge' or 'joint')
    end
  end
  -- 3. strokes: segments joined through joints and wedge nodes along the through pair, while
  -- class and one-way-ness agree (one MeshRoad line, one fitted curve, one cross-fall run)
  local function joinable(node)
    if not node or not node.through or (node.kind ~= 'joint' and node.kind ~= 'wedge') then return nil end
    local a, b = node.through[1], node.through[2]
    if a.seg == b.seg or a.seg.highway ~= b.seg.highway or (a.seg.oneway == 0) ~= (b.seg.oneway == 0) then return nil end
    return a, b
  end
  local strokes = {}
  local inStroke = {}
  for _, seg in ipairs(segs) do
    if not inStroke[seg] then
      -- walk back to the stroke's first segment
      local first, fe = seg, 1          -- fe: the end of `first` that starts the stroke
      local guard = 0
      while guard < 10000 do
        guard = guard + 1
        local a, b = joinable(nodes[first.nodes[fe]])
        if not a then break end
        local other = (a.seg == first and a.e == fe) and b or ((b.seg == first and b.e == fe) and a or nil)
        if not other or other.seg == seg or inStroke[other.seg] then break end
        first, fe = other.seg, 3 - other.e
        if first == seg then break end
      end
      -- walk forward collecting the segments
      local stroke = {parts = {}, passes = {}}
      local cur, ce = first, fe
      guard = 0
      while cur and not inStroke[cur] and guard < 10000 do
        guard = guard + 1
        inStroke[cur] = stroke
        stroke.parts[#stroke.parts + 1] = {seg = cur, reversed = ce == 2}
        local outEnd = 3 - ce
        local node = nodes[cur.nodes[outEnd]]
        local a, b = joinable(node)
        local nextSeg, nextE
        if a then
          local other = (a.seg == cur and a.e == outEnd) and b or ((b.seg == cur and b.e == outEnd) and a or nil)
          if other and not inStroke[other.seg] then nextSeg, nextE = other.seg, other.e end
        end
        if nextSeg then stroke.passes[#stroke.passes + 1] = node.id end
        cur, ce = nextSeg, nextE
      end
      strokes[#strokes + 1] = stroke
    end
  end
  -- stroke geometry: points with half widths, the node ids at its ends and at its passes
  local roads = {}
  for _, st in ipairs(strokes) do
    local P, knotOf = {}, {}
    local main = st.parts[1].seg
    for _, part in ipairs(st.parts) do
      local seg = part.seg
      if seg.rank > main.rank then main = seg end
      local src = seg.P
      local n = #src
      local startNode = part.reversed and seg.nodes[2] or seg.nodes[1]
      if #P == 0 then knotOf[startNode] = 1 end
      for i = 1, n do
        local p = src[part.reversed and n - i + 1 or i]
        if not (#P > 0 and i == 1) then P[#P + 1] = {p[1], p[2], p[3], seg.hw} end
        if i == 1 and #P > 1 then P[#P][3] = max(P[#P][3], p[3]) end
      end
      knotOf[part.reversed and seg.nodes[1] or seg.nodes[2]] = #P
    end
    local f, l = st.parts[1], st.parts[#st.parts]
    local road = {id = main.id, tags = main.tags, highway = main.highway, cls = main.cls, rank = main.rank,
      lanes = main.lanes, P = P, knotOf = knotOf, passes = st.passes, parts = st.parts,
      bridge = main.bridge, tunnel = main.tunnel, cut = {0, 0}, endTan = {},
      nodes = {f.reversed and f.seg.nodes[2] or f.seg.nodes[1], l.reversed and l.seg.nodes[1] or l.seg.nodes[2]}}
    -- one-way in the stroke's direction: +1 along it, -1 against
    road.oneway = f.seg.oneway * (f.reversed and -1 or 1)
    road.bank = tune.bank[road.cls] or tune.bank[road.highway] or tune.bank.default
    road.ease = tune.ease[road.cls] or tune.ease[road.highway] or tune.ease.default
    roads[#roads + 1] = road
    for _, part in ipairs(st.parts) do part.seg.road = road end
  end
  stats.strokes = #roads
  -- mouths in stroke terms: which stroke end is a segment end, and where a stroke passes a node
  local function strokeEnd(m)
    local road = m.seg.road
    if road.nodes[1] == m.seg.nodes[m.e] and road.parts[1].seg == m.seg then return road, 1 end
    if road.nodes[2] == m.seg.nodes[m.e] and road.parts[#road.parts].seg == m.seg then return road, 2 end
    return road, nil
  end
  -- 4. tangents: interior knots the bisector of their chords; a joint between two strokes (class
  -- change) the shared line; a branch the tangent of its host where it leaves; else its own chord
  local function chordTangent(P, k)
    local n = #P
    local a, b = P[max(1, k - 1)], P[min(n, k + 1)]
    local ux, uy = unit(P[k][1] - a[1], P[k][2] - a[2])
    local vx, vy = unit(b[1] - P[k][1], b[2] - P[k][2])
    local tx, ty = unit(ux + vx, uy + vy)
    if tx == 0 and ty == 0 then tx, ty = unit(b[1] - a[1], b[2] - a[2]) end
    return {tx, ty}
  end
  for _, road in ipairs(roads) do
    road.T = {}
    for k = 1, #road.P do road.T[k] = chordTangent(road.P, k) end
  end
  local function setEndTangent(road, e, m, tx, ty)
    local sgn = (tx * m.dx + ty * m.dy) >= 0 and 1 or -1
    tx, ty = tx * sgn, ty * sgn
    if e == 2 then tx, ty = -tx, -ty end
    road.T[e == 1 and 1 or #road.P] = {tx, ty}
  end
  for _, node in pairs(nodes) do
    if node.through and node.kind ~= 'star' then
      for _, m in ipairs(node.through) do
        local road, e = strokeEnd(m)
        if e then setEndTangent(road, e, m, node.line[1], node.line[2]); stats.joints = stats.joints + .5 end
      end
    end
  end
  for _, node in pairs(nodes) do
    for _, m in ipairs(node.branches) do
      local road, e = strokeEnd(m)
      local host = m.host.seg.road
      local k = host.knotOf[node.id]
      if e and k then
        local t = host.T[k]
        setEndTangent(road, e, m, t[1], t[2])
      end
    end
    if node.kind == 'end' then stats.ends = stats.ends + 1 end
  end
  stats.joints = floor(stats.joints)
  -- 5. fitted line, grades, widths (lane gain and drop tapers) and cross-fall of every stroke
  for _, road in ipairs(roads) do
    local P = road.P
    local n = #P
    local G, dist = {}, {0}
    for k = 2, n do dist[k] = dist[k - 1] + sqrt((P[k][1] - P[k - 1][1]) ^ 2 + (P[k][2] - P[k - 1][2]) ^ 2) end
    for k = 1, n do
      local ka, kb = max(1, k - 1), min(n, k + 1)
      G[k] = (P[kb][3] - P[ka][3]) / max(.1, dist[kb] - dist[ka])
    end
    road.D, road.knots = hermite(P, road.T, G, tune.sample)
    -- shoulders: an asymmetric one-way section moves its surface centre off the lane centre line
    local sh = (road.oneway ~= 0 and tune.shouldersOneway[road.highway]) or tune.shoulders[road.highway]
    if road.tags.width or not sh then sh = {0, 0} end
    road.sL, road.sR = sh[1], sh[2]
    if road.oneway ~= 0 and sh[1] ~= sh[2] then
      local shift = (sh[2] - sh[1]) / 2 * road.oneway
      for _, o in ipairs(road.D) do o.x, o.y = o.x + o.ty * shift, o.y - o.tx * shift end
      road.shift = shift
    end
    road.length = road.D[#road.D].s
    smooth(road.D, 'hw', tune.taper or 60)
    banking(road, tune)
    stats.km = stats.km + road.length / 1000
  end
  -- joints between two strokes (a class change straight on): both ends take the mean cross-fall
  for _, node in pairs(nodes) do
    if node.through and node.kind ~= 'star' then
      local ends = {}
      for _, m in ipairs(node.through) do
        local road, e = strokeEnd(m)
        if e then ends[#ends + 1] = {road = road, e = e} end
      end
      if #ends == 2 then
        local qs, sg = {}, {}
        for i, en in ipairs(ends) do
          local D = en.road.D
          local o = en.e == 1 and D[1] or D[#D]
          sg[i] = (o.tx * node.line[1] + o.ty * node.line[2]) >= 0 and 1 or -1
          qs[i] = o.q * sg[i]
        end
        local q = (qs[1] + qs[2]) / 2
        for i, en in ipairs(ends) do easeEnd(en.road, en.e, q * sg[i]) end
      end
    end
  end
  -- 6. Diverges and merges. The slip road starts where its inner edge leaves the host's edge (its
  -- first section in the host's plane); before that a wedge in the host's plane fills between the
  -- host's edge and the slip road's outer edge, from where that edge meets the host's.
  local plates, wedges = {}, {}
  local starNodes = {}
  for _, node in pairs(nodes) do
    for _, m in ipairs(node.branches) do
      local B, eb = strokeEnd(m)
      local H = m.host.seg.road
      local kH = H.knotOf[node.id] and H.knots[H.knotOf[node.id]]
      local w = {node = node.id, B = B, e = eb, H = H}
      wedges[#wedges + 1] = w
      local ok = eb ~= nil and kH ~= nil and B ~= H
      local rows, sStart, sOut, sGore, side, C, dir
      if ok then
        -- the host's samples from the node in the slip road's direction, flipped to that direction
        local o = H.D[kH]
        dir = (o.tx * m.dx + o.ty * m.dy) >= 0 and 1 or -1
        C = {}
        local k, c = kH, 0
        while H.D[k] and c <= tune.wedgeReach + 50 do
          local h = H.D[k]
          if #C > 0 then c = c + abs(h.s - H.D[k - dir].s) end
          local hh = h
          if dir < 0 then hh = {x = h.x, y = h.y, z = h.z, s = h.s, hw = h.hw, tx = -h.tx, ty = -h.ty, q = -h.q} end
          C[#C + 1] = {o = hh, c = c, k = k}
          k = k + dir
        end
        ok = #C >= 2
      end
      if ok then
        rows = {}
        local D, n = B.D, #B.D
        local hint
        for i = 1, n do
          local k = eb == 1 and i or n - i + 1
          local o = D[k]
          local s = eb == 1 and o.s or B.length - o.s
          if s > tune.wedgeReach then break end
          local ho, lat, hi, hc = projectChain(C, o.x, o.y, hint, 60)
          hint = hi
          rows[#rows + 1] = {s = s, o = o, ho = ho, latS = lat, hi = hi, hc = hc}
          -- far enough out to know the side: the shoulder shifts leave the slip road's first
          -- metres a little on the wrong side of the host's surface centre (an exit to the right
          -- started 0.5 m left of it, and the rail gap went to the median)
          if abs(lat) > ho.hw + o.hw then break end
        end
        local last = rows[#rows]
        side = last and (last.latS >= 0 and 1 or -1)
        for k, row in ipairs(rows) do
          local o, ho = row.o, row.ho
          local a = row.latS * side
          local bl = {-o.ty, o.tx}
          local hl = {-ho.ty * side, ho.tx * side}
          row.outer = (bl[1] * hl[1] + bl[2] * hl[2]) >= 0 and 1 or -1
          row.lat = a
          if not sStart and a + o.hw >= ho.hw then sStart = k end
          if not sOut and a - o.hw >= ho.hw then sOut = k end
        end
        -- the gore nose: where the gap between the two surfaces reaches goreClear (searched on)
        if sOut then
          for i = (eb == 1 and 1 or n), (eb == 1 and n or 1), (eb == 1 and 1 or -1) do
            local o = D[i]
            local s = eb == 1 and o.s or B.length - o.s
            if s > rows[sOut].s then
              if s > tune.wedgeReach then break end
              local ho, lat, hi, hc = projectChain(C, o.x, o.y, hint, 60)
              hint = hi
              rows[#rows + 1] = {s = s, o = o, ho = ho, latS = lat, lat = lat * side, hi = hi, hc = hc, outer = rows[sOut].outer}
              if lat * side - o.hw - ho.hw >= tune.goreClear then sGore = #rows; break end
            end
          end
        end
        ok = sStart and sOut and side and rows[sOut].s <= .6 * B.length
      end
      if not ok then
        stats.wedgeFail = stats.wedgeFail + 1
        starNodes[node.id] = true
        w.failed = true
      else
        stats.wedges = stats.wedges + 1
        w.sOut, w.sGore = rows[sOut].s, rows[sGore or #rows].s
        local r = rows[sOut]
        local zPlane = planeZ(r.ho, r.o.x, r.o.y)
        local same = ((-r.o.ty) * (-r.ho.ty) + r.o.tx * r.ho.tx) >= 0 and 1 or -1
        B.cut[eb] = max(B.cut[eb], w.sOut)
        easeEnd(B, eb, r.ho.q * same, zPlane, w.sOut)
        local strip = {}
        for k = sStart, sOut do
          local row = rows[k]
          -- the slip road's outer edge after the ease (its first section is in the host's plane)
          local o = sampleAt(B.D, eb == 1 and row.s or B.length - row.s)
          local ox = o.x + -o.ty * o.hw * row.outer
          local oy = o.y + o.tx * o.hw * row.outer
          local ho, lat = projectChain(C, ox, oy, row.hi, 60)
          local sgn = lat >= 0 and 1 or -1
          local ix = ho.x + -ho.ty * ho.hw * sgn
          local iy = ho.y + ho.tx * ho.hw * sgn
          local iz = planeZ(ho, ix, iy)
          local oz = planeZ(ho, ox, oy)
          if k == sOut then oz = o.z + o.q * o.hw * row.outer end
          if abs(lat) <= ho.hw then ox, oy, oz = ix, iy, iz end
          strip[#strip + 1] = {{ix, iy, iz}, {ox, oy, oz}}
        end
        plates[#plates + 1] = {kind = 'wedge', node = node.id, strip = strip}
        -- rail gaps: the host's side from 5 m before the wedge to the gore nose; the slip road's
        -- inner side up to the nose
        w.hostGap = {side = side * dir, k0 = rows[sStart].hc - 5, k1 = rows[sGore or #rows].hc, C = C}
        w.innerSide = -rows[sOut].outer
      end
    end
  end
  -- 7. star junctions: every stroke end there cut back by starCut x the widest road; the plate a
  -- fan over the mouths' corners at their surface heights. A stroke passing a star does not exist
  -- (stars are never joined through).
  for _, node in pairs(nodes) do
    if node.kind == 'star' or starNodes[node.id] then
      local radius = 0
      for _, m in ipairs(node.mouths) do radius = max(radius, m.seg.hw * 2 * tune.starCut) end
      local corners, centre = {}, nil
      for _, m in ipairs(node.mouths) do
        local road, e = strokeEnd(m)
        if e then
          local cut = min(radius, road.length * .25)
          road.cut[e] = max(road.cut[e], cut)
          local o = sampleAt(road.D, e == 1 and road.cut[1] or road.length - road.cut[2])
          local lx, ly = -o.ty * o.hw, o.tx * o.hw
          corners[#corners + 1] = {o.x + lx, o.y + ly, o.z + o.q * o.hw}
          corners[#corners + 1] = {o.x - lx, o.y - ly, o.z - o.q * o.hw}
          local en = e == 1 and road.D[1] or road.D[#road.D]
          centre = centre or {en.x, en.y}
        end
      end
      if centre and #corners >= 4 then
        local angle = {}
        for _, c in ipairs(corners) do angle[c] = atan2(c[2] - centre[2], c[1] - centre[1]) end
        table.sort(corners, function(a, b) return angle[a] < angle[b] end)
        local zc = 0
        for _, c in ipairs(corners) do zc = zc + c[3] end
        plates[#plates + 1] = {kind = 'star', node = node.id, centre = {centre[1], centre[2], zc / #corners}, ring = corners}
        stats.stars = stats.stars + 1
      end
    end
  end
  -- 8. MeshRoad chunks over each stroke's drawn range; a node where the line turns `turn` or
  -- every straightStep m
  local chunks, nodesTotal = {}, 0
  for _, road in ipairs(roads) do
    local D = road.D
    local s0, s1 = road.cut[1], road.length - road.cut[2]
    if s1 - s0 > .5 then
      local count = math.max(1, math.ceil((s1 - s0) / tune.chunk))
      local len = (s1 - s0) / count
      for c = 1, count do
        local a, b = s0 + (c - 1) * len, s0 + c * len
        local list = {sampleAt(D, a)}
        local last = list[1]
        local k = at(D, a) + 1
        while D[k] and D[k].s < b - .5 do
          local o = D[k]
          local d = o.s - last.s
          if (d >= tune.step and abs(headingDiff(last, o)) >= tune.turn) or d >= tune.straightStep
            or abs(o.hw - last.hw) > .25 then
            list[#list + 1] = o
            last = o
          end
          k = k + 1
        end
        list[#list + 1] = sampleAt(D, b)
        local out = {}
        for i, o in ipairs(list) do
          local lx, ly = -o.ty, o.tx
          local nx, ny, nz = -o.q * lx, -o.q * ly, 1
          local l = sqrt(nx * nx + ny * ny + nz * nz)
          out[i] = {o.x, o.y, o.z, o.hw * 2, tune.depth, nx / l, ny / l, nz / l}
        end
        nodesTotal = nodesTotal + #out
        chunks[#chunks + 1] = {name = 'e2rRO_' .. road.id .. '_' .. c, road = road.id, highway = road.highway,
          bridge = road.bridge, nodes = out, length = b - a}
      end
    end
  end
  -- 9. guard rails along motorway, trunk and slip road edges, minus the junction gaps
  local gaps = {}
  local function addGap(road, side, s0, s1)
    gaps[road] = gaps[road] or {}
    table.insert(gaps[road], {side, s0, s1})
  end
  for _, w in ipairs(wedges) do
    if w.hostGap then
      for _, c in ipairs(w.hostGap.C) do
        if c.c >= w.hostGap.k0 and c.c <= w.hostGap.k1 then addGap(w.H, w.hostGap.side, c.o.s - 2.5, c.o.s + 2.5) end
      end
      if w.e == 1 then addGap(w.B, w.innerSide, 0, w.sGore) else addGap(w.B, w.innerSide, w.B.length - w.sGore, w.B.length) end
    end
  end
  local rails = {}
  for _, road in ipairs(roads) do
    if tune.rail.classes[road.highway] and not road.tunnel then
      for _, side in ipairs({1, -1}) do
        local s0, s1 = road.cut[1], road.length - road.cut[2]
        local runs, cur = {}, nil
        for _, o in ipairs(road.D) do
          local ok = o.s >= s0 + (road.cut[1] > 0 and 6 or 0) and o.s <= s1 - (road.cut[2] > 0 and 6 or 0)
          for _, g in ipairs(gaps[road] or {}) do
            if g[1] == side and o.s >= g[2] and o.s <= g[3] then ok = false end
          end
          if ok then
            local off = o.hw + tune.rail.offset
            local lx, ly = -o.ty * side, o.tx * side
            if not cur then cur = {}; runs[#runs + 1] = cur end
            cur[#cur + 1] = {o.x + lx * off, o.y + ly * off, o.z + o.q * off * side}
          else
            cur = nil
          end
        end
        for _, run in ipairs(runs) do
          if #run >= 3 then rails[#rails + 1] = {road = road.id, side = side, points = run} end
        end
      end
    end
  end
  for _, l in ipairs(lines or {}) do
    if (l.kind == 'guard_rail' or (l.tags and l.tags.barrier == 'guard_rail')) and l.points and #l.points >= 2 then
      local pts = {}
      for _, p in ipairs(l.points) do pts[#pts + 1] = {p[1], p[2], p[3] or 0} end
      rails[#rails + 1] = {osm = l.id, points = pts}
    end
  end
  -- 10. lane lines over the drawn range, and the AI graph's lines (whole strokes)
  local marks, nav = {}, {}
  for _, road in ipairs(roads) do
    local D = road.D
    local s0, s1 = road.cut[1], road.length - road.cut[2]
    if tune.marks.classes[road.highway] and s1 - s0 > 5 then
      -- edge lines at the shoulders (left and right of the direction of travel)
      local sl, sr = road.sL or .3, road.sR or .3
      if road.oneway < 0 then sl, sr = sr, sl end
      if road.oneway == 0 then sl, sr = max(.3, sl), max(.3, sr) end
      local defs = {{'left', 'line'}, {'right', 'line'}}
      for l = 1, road.lanes - 1 do defs[#defs + 1] = {l, 'dash'} end
      for _, def in ipairs(defs) do
        local pts = {}
        local s = s0
        while true do
          local o = sampleAt(D, min(s, s1))
          local laneW = (o.hw * 2 - sl - sr) / max(1, road.lanes)
          local off = def[1] == 'left' and (o.hw - sl) or def[1] == 'right' and -(o.hw - sr) or (o.hw - sl - def[1] * laneW)
          local lx, ly = -o.ty, o.tx
          pts[#pts + 1] = {o.x + lx * off, o.y + ly * off, o.z + o.q * off + tune.marks.lift}
          if s >= s1 then break end
          s = min(s1, s + tune.marks.step)
        end
        marks[#marks + 1] = {road = road.id, kind = def[2], points = pts, width = tune.marks.width}
      end
    end
    local pts, s = {}, 0
    while true do
      local o = sampleAt(D, min(s, road.length))
      pts[#pts + 1] = {o.x, o.y, o.z}
      if s >= road.length then break end
      s = min(road.length, s + tune.nav.step)
    end
    nav[#nav + 1] = {road = road.id, points = pts, width = sampleAt(D, road.length / 2).hw * 2, oneway = road.oneway,
      lanes = road.lanes, highway = road.highway}
  end
  local railKm = 0
  for _, r in ipairs(rails) do
    for k = 2, #r.points do
      local a, b = r.points[k - 1], r.points[k]
      railKm = railKm + sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2) / 1000
    end
  end
  stats.chunks, stats.mrNodes, stats.plates, stats.rails, stats.marks, stats.nav = #chunks, nodesTotal, #plates, #rails, #marks, #nav
  stats.railKm = railKm
  stats.planMs = (os.clock() - t0) * 1000
  return {roads = roads, chunks = chunks, plates = plates, rails = rails, marks = marks, nav = nav, stats = stats,
    wedges = wedges, nodes = nodes}
end

-- ---------------------------------------------------------------------------------------------
-- Meshes for plates and rails (ProceduralMesh tables: zero-based, clockwise faces)
-- ---------------------------------------------------------------------------------------------

local function newMesh(material)
  return {verts = {}, normals = {}, uvs = {}, faces = {}, material = material}
end

local function tri(mesh, o, a, b, c, up)
  local ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
  local nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
  local l = sqrt(nx * nx + ny * ny + nz * nz)
  if l < 1e-10 then return end
  if up and nz < 0 then b, c, nx, ny, nz = c, b, -nx, -ny, -nz end
  local V, N, U, F = mesh.verts, mesh.normals, mesh.uvs, mesh.faces
  N[#N + 1] = {x = nx / l, y = ny / l, z = nz / l}
  local n = #N - 1
  for _, p in ipairs({a, c, b}) do
    V[#V + 1] = {x = p[1] - o[1], y = p[2] - o[2], z = p[3] - o[3]}
    U[#U + 1] = {u = p[4] or p[1] / 4, v = p[5] or p[2] / 4}
    F[#F + 1] = {v = #V - 1, n = n, u = #U - 1}
  end
end

local function quad(mesh, o, a, b, c, d, up)
  tri(mesh, o, a, b, c, up)
  tri(mesh, o, a, c, d, up)
end

-- one plate's surface (a strip or a fan) and its skirt 0.5 m down, origin at its first point
function M.plateMesh(plate, material, lift)
  lift = lift or 0
  local mesh = newMesh(material)
  local o
  if plate.kind == 'wedge' then
    local S = plate.strip
    o = {S[1][1][1], S[1][1][2], S[1][1][3]}
    for k = 1, #S - 1 do
      local a, b = S[k], S[k + 1]
      local function L(p) return {p[1], p[2], p[3] + lift} end
      quad(mesh, o, L(a[1]), L(b[1]), L(b[2]), L(a[2]), true)
      -- skirt under the outer edge
      local p, q = a[2], b[2]
      quad(mesh, o, {p[1], p[2], p[3] + lift}, {q[1], q[2], q[3] + lift}, {q[1], q[2], q[3] - .5}, {p[1], p[2], p[3] - .5})
    end
  else
    local R, c = plate.ring, plate.centre
    o = {c[1], c[2], c[3]}
    for k = 1, #R do
      local a, b = R[k], R[k % #R + 1]
      tri(mesh, o, {c[1], c[2], c[3] + lift}, {a[1], a[2], a[3] + lift}, {b[1], b[2], b[3] + lift}, true)
    end
  end
  return mesh, o
end

-- a guard rail as a swept box beam and posts, in pieces of `chunk` m
function M.railMeshes(rail, tune, beamMat, postMat)
  local R = tune.rail
  local out = {}
  local P = rail.points
  local k = 1
  while k < #P do
    local beam, post = newMesh(beamMat), newMesh(postMat)
    local o = {P[k][1], P[k][2], P[k][3]}
    local run, lastPost = 0, -huge
    local start = k
    while k < #P and run < R.chunk do
      local a, b = P[k], P[k + 1]
      local tx, ty, L = unit(b[1] - a[1], b[2] - a[2])
      if L > 1e-3 then
        local nx, ny = -ty * R.thick / 2, tx * R.thick / 2
        local z0a, z1a = a[3] + R.height - R.beam, a[3] + R.height
        local z0b, z1b = b[3] + R.height - R.beam, b[3] + R.height
        -- two faces and the top of the beam
        quad(beam, o, {a[1] + nx, a[2] + ny, z0a}, {b[1] + nx, b[2] + ny, z0b}, {b[1] + nx, b[2] + ny, z1b}, {a[1] + nx, a[2] + ny, z1a})
        quad(beam, o, {b[1] - nx, b[2] - ny, z0b}, {a[1] - nx, a[2] - ny, z0a}, {a[1] - nx, a[2] - ny, z1a}, {b[1] - nx, b[2] - ny, z1b})
        quad(beam, o, {a[1] + nx, a[2] + ny, z1a}, {b[1] + nx, b[2] + ny, z1b}, {b[1] - nx, b[2] - ny, z1b}, {a[1] - nx, a[2] - ny, z1a}, true)
        -- posts every postStep m
        local s = run
        while lastPost + R.postStep <= s + L do
          local d = max(0, lastPost + R.postStep - s)
          lastPost = s + d
          local f = d / L
          local px, py, pz = a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f
          -- a post: its two faces across the rail (behind the beam, seen from the road and the verge)
          local h = R.post / 2
          local ax, ay, bx, by = px - tx * h, py - ty * h, px + tx * h, py + ty * h
          local zt = pz + R.height - R.beam / 2
          local ox2, oy2 = -ty * R.thick, tx * R.thick
          quad(post, o, {ax - ox2, ay - oy2, pz - .3}, {bx - ox2, by - oy2, pz - .3}, {bx - ox2, by - oy2, zt}, {ax - ox2, ay - oy2, zt})
          quad(post, o, {bx - ox2, by - oy2, pz - .3}, {ax - ox2, ay - oy2, pz - .3}, {ax - ox2, ay - oy2, zt}, {bx - ox2, by - oy2, zt})
        end
        run = run + L
      end
      k = k + 1
    end
    out[#out + 1] = {origin = o, meshes = {beam, post}, from = start, to = k}
  end
  return out
end

-- ---------------------------------------------------------------------------------------------
-- Engine objects
-- ---------------------------------------------------------------------------------------------

local function timer()
  if hptimer then return hptimer() end
  local t = os.clock()
  return {stop = function() return (os.clock() - t) * 1000 end}
end

local function v3(x, y, z) return vec3(x, y, z) end

-- opts: group (name), nodeMode ('field' or 'insert'), rails ('mesh', 'shape' or false),
-- marks (bool), nav (bool), plates (bool), materials (table), only (set of kinds)
function M.create(plan, opts)
  opts = opts or {}
  local mats = setmetatable(opts.materials or {}, {__index = M.MATERIALS})
  local tune = opts.tune or M.TUNE
  local h = {objects = {}, times = {}, counts = {}, group = nil}
  local function track(kind, obj, ms)
    h.objects[#h.objects + 1] = obj
    h.times[kind] = (h.times[kind] or 0) + ms
    h.counts[kind] = (h.counts[kind] or 0) + 1
  end
  local gname = opts.group or 'e2rRoadObjects'
  local old = scenetree.findObject(gname)
  if old then old:deleteAllObjects(); old:delete() end
  local g = createObject('SimGroup')
  g:registerObject(gname)
  scenetree.MissionGroup:addObject(g)
  h.group = g
  local total = timer()
  -- MeshRoads
  for _, c in ipairs(plan.chunks) do
    local t = timer()
    local obj = createObject('MeshRoad')
    obj:setField('topMaterial', 0, mats.top)
    obj:setField('bottomMaterial', 0, mats.bottom)
    obj:setField('sideMaterial', 0, mats.side)
    obj:setField('textureLength', 0, tostring(opts.textureLength or 5))
    obj:setField('breakAngle', 0, tostring(opts.breakAngle or 3))
    obj:setField('widthSubdivisions', 0, tostring(opts.widthSubdivisions or 0))
    if opts.nodeMode == 'insert' then
      obj:registerObject('')
      for i, n in ipairs(c.nodes) do
        obj:insertNode(v3(n[1], n[2], n[3]), n[4], n[5], v3(n[6], n[7], n[8]), i - 1)
      end
    else
      for _, n in ipairs(c.nodes) do
        obj:setField('Node', 0, string.format('%.3f %.3f %.3f %.3f %.3f %.5f %.5f %.5f', n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]))
      end
      obj:registerObject('')
    end
    obj.canSave = false
    g:addObject(obj)
    track('meshRoad', obj, t:stop())
  end
  -- junction plates
  if opts.plates ~= false then
    for k, p in ipairs(plan.plates) do
      local t = timer()
      local mesh, o = M.plateMesh(p, mats.plate, opts.plateLift or 0)
      if #mesh.faces > 0 then
        local obj = createObject('ProceduralMesh')
        obj:registerObject('e2rRO_plate_' .. k)
        obj.canSave = false
        g:addObject(obj.obj)
        obj:createMesh({{mesh}})
        obj:setPosition(v3(o[1], o[2], o[3]))
        track('plate', obj, t:stop())
      end
    end
  end
  -- guard rails
  if opts.rails == 'mesh' then
    for k, r in ipairs(plan.rails) do
      for j, piece in ipairs(M.railMeshes(r, tune, mats.rail, mats.post)) do
        local t = timer()
        local obj = createObject('ProceduralMesh')
        obj:registerObject('e2rRO_rail_' .. k .. '_' .. j)
        obj.canSave = false
        g:addObject(obj.obj)
        obj:createMesh({piece.meshes})
        obj:setPosition(v3(piece.origin[1], piece.origin[2], piece.origin[3]))
        track('railMesh', obj, t:stop())
      end
    end
  elseif opts.rails == 'forest' then
    -- stock rail pieces as Forest items (instanced, ~1-2 us each, collidable through the shape's
    -- collision mesh): one ForestItemData kind, one Forest for the prototype
    local t = timer()
    local kind = scenetree.findObject('e2rRO_railKind')
    if not kind then
      kind = createObject('ForestItemData')
      kind:setField('shapeFile', 0, mats.railShape)
      kind:setField('collidable', 0, '1')
      kind:setField('windScale', 0, '0')
      kind:registerObject('e2rRO_railKind')
    end
    local forest = createObject('Forest')
    forest.canSave = false
    forest:registerObject(gname .. '_forest')
    scenetree.MissionGroup:addObject(forest)
    h.forest = forest
    local data = forest:getData()
    local m, euler, at = MatrixF(true), vec3(0, 0, 0), vec3(0, 0, 0)
    local len = tune.rail.piece
    local items = 0
    for _, r in ipairs(plan.rails) do
      local P = r.points
      local dist = {0}
      for k = 2, #P do dist[k] = dist[k - 1] + sqrt((P[k][1] - P[k - 1][1]) ^ 2 + (P[k][2] - P[k - 1][2]) ^ 2) end
      local j = 1
      local function pointAt(s)
        while j < #P - 1 and dist[j + 1] < s do j = j + 1 end
        local f = max(0, min(1, (s - dist[j]) / max(EPS, dist[j + 1] - dist[j])))
        return P[j][1] + (P[j + 1][1] - P[j][1]) * f, P[j][2] + (P[j + 1][2] - P[j][2]) * f, P[j][3] + (P[j + 1][3] - P[j][3]) * f
      end
      local s = 0
      while s + len <= dist[#P] + .01 do
        local ax, ay, az = pointAt(s)
        local bx, by, bz = pointAt(s + len)
        -- yaw only (heading of the piece's chord); the rail's grade is a few per cent
        euler.x, euler.y, euler.z = 0, 0, atan2(by - ay, bx - ax) + (opts.shapeYaw or 0)
        m:setFromEuler(euler)
        at.x, at.y, at.z = (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2
        m:setPosition(at)
        if data:createNewItem(kind, m, 1) then items = items + 1 end
        s = s + len
      end
    end
    h.counts.railForest = items
    h.times.railForest = t:stop()
  elseif opts.rails == 'shape' then
    local len = tune.rail.piece
    for _, r in ipairs(plan.rails) do
      local P = r.points
      local dist = {0}
      for k = 2, #P do dist[k] = dist[k - 1] + sqrt((P[k][1] - P[k - 1][1]) ^ 2 + (P[k][2] - P[k - 1][2]) ^ 2) end
      local function pointAt(s)
        for k = 1, #P - 1 do
          if s <= dist[k + 1] or k == #P - 1 then
            local f = max(0, min(1, (s - dist[k]) / max(EPS, dist[k + 1] - dist[k])))
            return P[k][1] + (P[k + 1][1] - P[k][1]) * f, P[k][2] + (P[k + 1][2] - P[k][2]) * f, P[k][3] + (P[k + 1][3] - P[k][3]) * f
          end
        end
      end
      local s = 0
      while s + len <= dist[#P] + .01 do
        local t = timer()
        local ax, ay, az = pointAt(s)
        local bx, by, bz = pointAt(s + len)
        local yaw = atan2(by - ay, bx - ax) + (opts.shapeYaw or 0)
        local pitch = atan2(bz - az, sqrt((bx - ax) ^ 2 + (by - ay) ^ 2))
        local sz, cz = sin(yaw / 2), cos(yaw / 2)
        local sy, cy = sin(-pitch / 2), cos(-pitch / 2)
        local obj = createObject('TSStatic')
        obj:setField('shapeName', 0, mats.railShape)
        obj:setField('collisionType', 0, 'Collision Mesh')
        obj:setField('decalType', 0, 'None')
        obj.useInstanceRenderData = true
        local mx, my, mz = (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2
        if opts.shapeAnchor == 'start' then mx, my, mz = ax, ay, az end
        obj:setPosRot(mx, my, mz, -sz * sy, cz * sy, sz * cy, cz * cy)
        obj:registerObject('')
        obj.canSave = false
        g:addObject(obj)
        track('railShape', obj, t:stop())
        s = s + len
      end
    end
  end
  -- lane lines
  if opts.marks then
    for _, m in ipairs(plan.marks) do
      local t = timer()
      local obj = createObject('DecalRoad')
      obj:setField('material', 0, m.kind == 'dash' and mats.dash or mats.line)
      obj:setField('overObjects', 0, 'true')
      obj:setField('improvedSpline', 0, 'true')
      obj:setField('drivability', 0, '-1')
      obj:setField('textureLength', 0, m.kind == 'dash' and '18' or '5')
      obj:setField('renderPriority', 0, '9')
      obj:registerObject('')
      obj.canSave = false
      g:addObject(obj)
      -- nodes without a rebuild each, then one (edges built inside this call, not next frame)
      for _, p in ipairs(m.points) do obj:insertNodeNoRegen(v3(p[1], p[2], p[3]), m.width, -1) end
      obj:regenerate()
      track('markDecal', obj, t:stop())
    end
  end
  -- AI graph roads (invisible): map.lua builds its graph from DecalRoads with drivability > 0
  if opts.nav then
    for _, n in ipairs(plan.nav) do
      local t = timer()
      local obj = createObject('DecalRoad')
      obj:setField('material', 0, mats.nav)
      obj:setField('overObjects', 0, 'true')
      obj:setField('improvedSpline', 0, 'true')
      obj:setField('drivability', 0, '1')
      obj:setField('autoLanes', 0, 'false')
      obj:setField('oneWay', 0, n.oneway ~= 0 and 'true' or 'false')
      obj:setField('lanesLeft', 0, tostring(n.oneway ~= 0 and 0 or max(1, floor(n.lanes / 2))))
      obj:setField('lanesRight', 0, tostring(n.oneway ~= 0 and n.lanes or max(1, n.lanes - floor(n.lanes / 2))))
      if n.oneway < 0 then obj:setField('flipDirection', 0, 'true') end
      obj:registerObject('')
      obj.canSave = false
      g:addObject(obj)
      for _, p in ipairs(n.points) do obj:insertNodeNoRegen(v3(p[1], p[2], p[3] + .2), n.width, -1) end
      obj:regenerate()
      track('navDecal', obj, t:stop())
    end
  end
  h.totalMs = total:stop()
  return h
end

function M.remove(h)
  local t = timer()
  if h and h.forest then h.forest:delete(); h.forest = nil end
  if h and h.group and h.group.deleteAllObjects then
    h.group:deleteAllObjects()
    h.group:delete()
  end
  return t:stop()
end

return M
