-- Swept roads from pack v3 (design notes, section 7, and the phase 2 decisions): the
-- carriageway as our own ProceduralMesh swept along the exporter's stations (paved extents left and
-- right, cross-fall of each half), lane markings baked into the road's own items, exits and entries
-- with the host's added lane, the slip road cut along the host's edge line and the gore hatched, fan
-- plates from the exporter's junction records, and guard rails only where OSM maps them (a swept beam).
-- Heights are the exporter's; the only runtime height rule is a slip road lying in its host's plane up
-- to the gore nose and easing into its own profile after it (TUNE.noseEase).
-- Phase 3 (ROAD-OBJECTS.md section 9): where the pack carries junction surfaces (roads.junctions,
-- fast_junctions.py) they replace the fan plates and the fork rule at their nodes: the exporter's mesh
-- with kerb fillets and islands, the roads cut at its mouths; stop and give-way lines, turn arrows,
-- crossings and guidance lines from its approach records, drawn with the road's own lane block; lane
-- lines end at the stop line; the navgraph turns through the junction on one path per movement.
-- Without junction records everything stays as in phase 2.
--
-- Pure Lua, no engine call (runs offline in LuaJIT): M.plan(data, opts) -> plan; generator.lua calls
-- the plan's steps with its own ground functions (press, verge, asphalt) and takes the items.
-- Units: region metres; s along a stroke; u across it, + left of the stroke's direction.
-- Ownership: a chunk is built whole by the tile holding its middle station (the cell rule of the pack),
-- a plate by the tile holding its node, so every object exists once and chunk ends meet exactly (the
-- exporter writes a chunk's last station and the next chunk's first bit for bit).
local M = {}
local sqrt, abs, floor, max, min, atan2, cos, sin = math.sqrt, math.abs, math.floor, math.max, math.min, math.atan2, math.cos, math.sin
local EPS = 1e-9
local NONE = {}

M.TUNE = {
  lift = .10,          -- the generator's ROAD_LIFT: v3 z is the profile without it
  planeSep = .004,     -- m: a branch row in its host's plane lies this far under it (never coplanar: no z-fight)
  noseEase = 30,       -- m after the gore nose (before it, for entries) over which a slip road leaves its host's plane
  press = 2,           -- m between press samples along a carriageway
  margin = .12,        -- the ground stays this far under the surface (generator TUNE.margin)
  railStop = 6,        -- m: rails stop short of a fan junction's cut
  railMin = 2,         -- m: shorter rail pieces go
  railClear = .45,     -- m: a rail point within this of a drawn surface is cut there (phase 3: .25 left the beam's
                       -- face and posts within 0.3 m of another road's surface, in testing)
  railMinOffset = .30, -- m outside the paved edge at least (a mapped barrier inside it is drawn at the edge)
  railBehind = {2, 4}, -- m behind a rail where another carriageway makes it a median rail (rule:median: edges under
                       -- 6 m apart, the rail in the middle; a mapped one up to 4 m out); 2 m apart, so a lane of 3 m
                       -- whose near edge lies within 4 m is never missed
  -- Where a rail meets a bridge deck it joins the deck's parapet (M.railPaths): a rail that stops within railJoin m of the
  -- bridge's end runs on to it, its last railTaper m (at least 8 x the sideways step) draw in to the parapet, and it ends
  -- railOverrun m inside it, its face railJoinOffset m (a concrete barrier: railJoinJersey) from the parapet's inner face, so
  -- there is no gap and no rail end toward the traffic. railJoin = 0: off, a rail ends where it is mapped.
  railJoin = 14, railTaper = 6, railOverrun = .3, railJoinOffset = .08, railJoinJersey = .15,
  navStep = 80,        -- m between nav knots at most (generator navSegment)
  spawnStep = 1,
  -- Transitions along a road (own choice, NOT from a guideline): where two strokes meet straight on
  -- (a class change: motorway -> trunk, link -> primary), each blends its extents, cross-fall and height
  -- to the pair's mean over `joint` m; a lane gain or drop inside a stroke moves its lane block over
  -- `lanes` m (the exporter's own taper of the extents, fast_roads.TAPER, is the same length).
  joint = {motorway = 60, trunk = 45, primary = 30, secondary = 25, tertiary = 20, link = 20, default = 15},
  lanes = {motorway = 120, trunk = 90, link = 40, default = 20},
  forkAngle = math.rad(35),   -- a branch this close to a through road's direction merges along its edge
  forkReach = 150,            -- m along a branch searched for where it parts from its host
  -- bundle8 (gores): the paved wedge between a branch and its host ends (the nose) where the branch's inner edge lies this far
  -- clear of the host's paved edge, not where the edges first part. Aerials (the chevrons end 3.5 m wide) and
  -- a measured example: 3 m; the 0.5 m of an earlier example put the nose 35 m early in another.
  goreGap = 3.0,
  -- blend a road into a plate mouth whose section differs from its own (a class change through a
  -- junction): off, it gave more lane-line steps over 1 cm offline (559 vs 386 in 36 tiles) than the
  -- plate's own fan does; the plate then carries the change over its cut length
  plateBlend = false,
  -- M2 (bundle6): a junction surface's heights blend to its mouths' exact sections within this many metres, so the
  -- plate's edge z is the joining stroke's edge z (the exporter's height field was 10-25 cm off the stroke's own
  -- corner in testing: the replay's jolt); 0 = off
  plateJoin = 8,
  plateFlatJoin = true,  -- M2: a plate's mouth onto a generator road (not swept) has that road's flat section
  -- M4 (bundle6): kerbs and implied sidewalks on swept streets (fitted classes primary..unclassified)
  kerbHeight = .15,      -- m (own choice: the usual 15 cm Hochbord)
  walkWidth = 2.5,       -- m of an implied sidewalk (sidewalk=both|left|right), memory-not-verified
  streetKerbs = true,
  -- M2 (bundle6): a free end of a swept stroke on another swept stroke's surface (a loop's end on its host: the loops
  -- of a cloverleaf in testing stood 5-31 cm above or below the driven lane, an 18 cm step) blends to that
  -- stroke's plane over this many metres; 0 = off
  endJoin = 20,
  -- bundle7 (P2): the lower-ranked of two strokes ending side by side takes the other plane over their overlap
  -- (at most overlapReach m from its drawn end, its paved edge more than overlapIn m inside the other extent), joins
  -- it over overlapLead m at a plate mouth and leaves it over overlapEase m; 0 ease = off
  overlapEase = 20,
  overlapReach = 60,
  overlapIn = .3,
  overlapLead = 4,
  overlapMouth = true,   -- the fan plate's mouth of the pulled stroke goes into the other's plane as well
  -- Work per tile (bundle5, 31 Oct): the cells read round a tile hold up to 2 x 2 km of strokes, of which
  -- the tile owns about a sixteenth. Rows and the footprint index are built only for chunks reaching
  -- `rowMargin` m round the tile, and further chunks when a surface query reaches them (M.ensureAt: an owned
  -- chunk's rails, decks and marks, the generator's crossing ways); the ground steps (press, verges,
  -- footprints, spawn) only take chunks reaching `groundMargin` m (the lattice reaches one terrain step,
  -- <= 25 m, past the tile; skirts reach ROADS.section.reach, 30 m, past the paved edge). The owned items
  -- are the same as with every chunk (route-tile digests, p5/perf). `pauseRows`: a pause point every this
  -- many rows in the long loops (press, verges, footprints, marks), so no stretch runs past the frame budget:
  -- the verges' rows cost the most (the generator cuts a skirt from each), 48 of them ran 26 ms in one piece,
  -- 8 up to 8 ms (earthng/generator_budget.py --swept).
  rowMargin = 70,
  groundMargin = 70,
  pauseRows = 4,
}
-- Classes swept per switch value (world.lua sweptRoads): 'fast' = motorways and trunks with their
-- links (the phase-1 scope, the test interchange); 'fitted' = every class the exporter fits.
M.CLASSES = {
  fast = {motorway = true, motorway_link = true, trunk = true, trunk_link = true},
  fitted = {motorway = true, motorway_link = true, trunk = true, trunk_link = true, primary = true, primary_link = true,
    secondary = true, secondary_link = true, tertiary = true, tertiary_link = true, unclassified = true},
}

-- Longitudinal markings (RMS: Richtlinien fuer die Markierung von Strassen). Every value carries its
-- source. 'web' = secondary sources found on 30 Oct 2026 (FAQ pages quoting the RMS), not the RMS
-- text; 'memory' = not checked against any document. Verify against RMS Teil A/C before the default.
M.MARKS = {
  narrow = {autobahn = .15, other = .12, source = 'web'},      -- Schmalstrich
  broad = {autobahn = .30, other = .25, source = 'web'},       -- Breitstrich
  guide = {rural = {6, 12}, urban = {3, 6}, source = 'web (6/12 Autobahn); memory (3/6 innerorts)'},  -- Leitlinie
  block = {autobahn = {6, 6}, other = {3, 3}, source = 'memory'},  -- Blockmarkierung at exit/entry lanes
  -- the Autobahn's right edge line broad, its left (median) edge line narrow; other roads narrow
  edge = {source = 'memory (right broad / left narrow on Autobahn)'},
  hatch = {every = 5, stripe = .5, margin = .3, angle = math.rad(45), chevron = true, forkMinGap = 1.6,   -- bundle9: a fork's gore is hatched only where it is 1.6 m wide (aerials: the last 12-20 m before the nose)
    forkLast = 20,   -- and only within 20 m of its nose (S.blend0; a merge's nose is at its start): a loop leaving at a shallow angle (in testing) had 25 m of chevrons 45-70 m early
    source = 'memory; spacing as generator GORE; chevrons (phase 3) as in a reference image'},
  lift = .008,         -- over the surface (generator DECAL.markLift)
  inside = .25,        -- m: the lane block's edge lines stay this far inside the paved edge at least
  jump = 1.0,          -- m: a line moving more than this between two rows is broken there
  spread = 1.45,       -- the line texture's quad is this x the line's width (generator DECAL.markSpread)
  repeatV = 6.4,       -- m of line texture along (generator DECAL.markRepeat)
  -- Phase 3, junctions (StVO Zeichen numbers; dimensions NOT verified against the RMS text: 'memory').
  stop = {width = .50, source = 'memory (Z 294 Haltlinie, Breitstrich 0.50 m)'},
  giveway = {width = .50, dash = {.50, .25}, source = 'memory (Z 341 Wartelinie: 0.50 m wide, 0.50 m dashes, 0.25 m gaps)'},
  teeth = {on = true, base = .50, height = .60, gap = .25, behind = .30, source = 'memory (Z 342 Haifischzaehne)'},
  zebra = {stripe = .50, gap = .50, inside = .25, source = 'memory (Z 293: 0.50 m stripes and gaps, 3-4 m long)'},
  furt = {width = .12, dash = {.50, .20}, source = 'memory (Fussgaengerfurt: Schmalstrich, short dashes)'},
  cycle = {width = .25, dash = {.50, .20}, source = 'memory (Radfahrerfurt: Breitstrich, short dashes)'},
  arrow = {length = {urban = 5.0, rural = 7.5}, shaft = .16, head = .60, headLength = 1.6, before = 2.0, repeatAt = 30,
    source = 'memory (Z 297 Richtungspfeile, 5 m innerorts / 7.5 m ausserorts)'},
  solid = {urban = 20, rural = 30, source = 'own choice (Fahrstreifenbegrenzung before the stop line)'},
  guidance = {width = .12, dash = {1.5, 1.5}, source = 'memory (Leitlinie im Knotenpunkt)'},
  mouthEdge = {dash = {1.0, 1.0}, source = 'memory (the major road edge line across a side road mouth)'},
  sub = .5,            -- m: marks on the surface are subdivided this fine (the surface's crowns lie under them)
  -- bundle5: a lane that comes or goes (Fahrstreifenaddition / -subtraktion) gets its lane line only where it is
  -- this share of a full lane wide: the line starts at the end of the widening and ends where the narrowing
  -- begins (memory of the RMS practice, NOT verified; the route aerials show no line inside a taper)
  laneFrom = .9,
  -- the host's edge where a branch is drawn along it (forks, merges and diverges without a lane attachment):
  -- a lane line (Leitlinie) between the two, as on a DOP20 aerial (measured), was no line
  forkLine = true,
  clip = .05,          -- m: a fork branch's line farther than this beyond its cut edge is dropped or kept to the edge (bundle9c)
  -- exits' and entries' added lane: its taper S-shaped (smoothstep, as the exporter from pack roads version 3)
  -- and drawn with this many rows (the stations are up to 10 m apart, the measured taper is 16-25 m)
  taperRows = 8,
}

-- material: the look of the beam and the posts, a stock game material by name. The game loads the common art
-- (art/shapes/objects/main.materials.json among it) for every level, a level's own materials only with that level.
-- M.railLook picks what the game can draw: 1 the stock material by name; 2 our own e2rf_guardrail, defined at runtime
-- from the stock sheet's texture paths, if only the stock definition is missing; 3 `fallback`, our own kind
-- (e2rf_<kind>). material = nil, or = fallback: always 3. jerseyMaterial: our own kind for the concrete barriers
-- (the stock material is a W-beam sheet).
-- stock: the layout of that sheet (t_italy_guardrails_b: 1024 x 1024 px, 341 px a metre, v down the image as the engine
-- reads it): a beam panel is `panel` m of u, its bolts at the panel's start: the post spacing, so a joint sits on every
-- post (the stock mesh stretches it over 3 m; 4 m stretches the sheet by a third, which its bolt rows take); its W strip is image
-- rows 350-451, v0 at the beam's foot and v0 + dv at its top; the posts sample the plain patch post = {u0, v top, u1, v bottom}.
M.RAIL = {height = .75, beam = .32, thick = .10, depth = .06, post = .14, posts = 4, sink = .3,
  material = 'italy_guardrails', fallback = 'clad_metal', jerseyMaterial = 'clad_metal',
  stock = {panel = 4, v0 = .441, dv = -.099, post = {.39, .285, .63, .332},
    define = {name = 'e2rf_guardrail', dir = '/assets/materials/trim/metal/m_italy_guardrails/', metallicFactor = .75,
      maps = {baseColorMap = 't_italy_guardrails_b.color.png', normalMap = 't_italy_guardrails_nm.normal.png',
        metallicMap = 't_italy_guardrails_m.data.png', roughnessMap = 't_italy_guardrails_r.data.png'}}},
  kinds = {guard_rail = 'beam', cable_barrier = 'beam', jersey_barrier = 'jersey'},
  jersey = {height = .81, base = .61, top = .15}, postSource = 'memory (Pfostenabstand 4 m einfache Schutzplanke)'}
M.DECK = {parapet = .9, thick = .3, material = 'fence_wall', piece = 2, clear = .3}

-- What the game can draw for a rail (see M.RAIL). Asks the engine through guarded globals (scenetree, FS, log,
-- jsonWriteFile, loadJsonMaterialsFile); without them (offline) the stock material is taken. Once per setting, logged once.
-- Returns {tier, stock, name (the engine's material), kind (the generator's: '=' .. name for a stock material, else the
-- e2rf_ kind) and, with stock, panel, v0, dv, post}.
local railLooks = {}
local function inGame(name)
  if not scenetree or not scenetree.findObject then return true end
  return scenetree.findObject(name) ~= nil
end
local function defineRail(d)
  if not (FS and scenetree and jsonWriteFile and loadJsonMaterialsFile) then return false end
  local stage = {metallicFactor = d.metallicFactor}
  for field, file in pairs(d.maps) do
    local path = d.dir .. file
    if not (FS:fileExists(path) or FS:fileExists(path:sub(1, -5) .. '.dds')) then return false end
    stage[field] = path
  end
  if not scenetree.findObject(d.name) then
    jsonWriteFile('e2r_world/rail_material.json', {[d.name] = {name = d.name, mapTo = d.name, class = 'Material',
      Stages = {stage, {}, {}, {}}, annotation = 'GUARD_RAIL', groundType = 'METAL', dynamicCubemap = true,
      translucentBlendOp = 'None', version = 1.5}}, false)
    loadJsonMaterialsFile('/e2r_world/rail_material.json')
  end
  return true
end
function M.railLook(R)
  R = R or M.RAIL
  local key = tostring(R.material) .. '>' .. tostring(R.fallback)
  local look = railLooks[key]
  if look then return look end
  local S = R.stock or M.RAIL.stock
  look = {tier = 3, stock = false, kind = R.fallback, name = 'e2rf_' .. R.fallback}
  if R.material and R.material ~= R.fallback and S then
    local function stockLook(tier, name, kind)
      return {tier = tier, stock = true, kind = kind, name = name, panel = S.panel, v0 = S.v0, dv = S.dv, post = S.post}
    end
    local def = S.define
    if inGame(R.material) then
      look = stockLook(1, R.material, '=' .. R.material)
      if log then log('I', 'e2r', 'guard rails: stock material ' .. R.material) end
    elseif def and defineRail(def) then
      look = stockLook(2, def.name, (def.name:gsub('^e2rf_', '')))
      if log then log('W', 'e2r', 'guard rails: stock material ' .. R.material .. ' is not loaded, ' .. def.name .. ' made from its texture paths') end
    elseif log then
      log('W', 'e2r', 'guard rails: stock material ' .. R.material .. ' is not in the game, using ' .. R.fallback)
    end
  end
  railLooks[key] = look
  return look
end
local SPEED = {motorway = 110, trunk = 90, primary = 60, secondary = 60, tertiary = 50, unclassified = 50,
  residential = 40, living_street = 20, service = 20}
local DRIVE = {motorway = 1, trunk = 1, primary = 1, secondary = 1, tertiary = .9, unclassified = .7,
  residential = .7, living_street = .4, service = .4}
local EDGED = {motorway = true, trunk = true, primary = true, secondary = true}
local CENTRED = {motorway = true, trunk = true, primary = true, secondary = true, tertiary = true}

local function baseClass(highway) return (highway or ''):gsub('_link$', '') end
local function smooth(t) t = max(0, min(1, t)); return t * t * (3 - 2 * t) end
local function number(v, d)
  local n = tonumber((tostring(v or ''):match('^%s*([%d%.]+)')))
  return n or d
end

-- ---------------------------------------------------------------------------------------------------
-- Strokes: the loaded chunks of each, its events, furniture, structures, lanes and cuts
-- ---------------------------------------------------------------------------------------------------

-- the station index i of chunk c with c.s[i] <= s < c.s[i + 1] (clamped)
local function seek(c, s)
  local lo, hi = 1, c.n - 1
  if s <= c.s[1] then return 1 end
  if s >= c.s[c.n] then return c.n - 1 end
  while lo < hi do
    local mid = floor((lo + hi + 1) / 2)
    if c.s[mid] <= s then lo = mid else hi = mid - 1 end
  end
  return lo
end

-- stations of chunk c interpolated at s: {s, x, y, z, h, eL, eR, fl, fr} (one table, the fields unrolled:
-- this runs for every row, mark point and projection)
local function lerpAt(col, i, t)
  local a = col[i]
  if t == 0 then return a end
  local b = col[i + 1]
  if t == 1 then return b end
  return a + (b - a) * t
end
local function sample(c, s)
  local i = seek(c, s)
  local s0, s1 = c.s[i], c.s[i + 1]
  local t = s1 > s0 and (s - s0) / (s1 - s0) or 0
  if s == s0 then t = 0 elseif s == s1 then t = 1 end
  return {s = s, x = lerpAt(c.x, i, t), y = lerpAt(c.y, i, t), z = lerpAt(c.z, i, t), h = lerpAt(c.h, i, t),
    eL = lerpAt(c.eL, i, t), eR = lerpAt(c.eR, i, t), fl = lerpAt(c.fl, i, t), fr = lerpAt(c.fr, i, t)}
end

local function dedupe(list, key)
  local seen, out = {}, {}
  for _, v in ipairs(list) do
    local k = key(v)
    if not seen[k] then seen[k] = true; out[#out + 1] = v end
  end
  return out
end

local Stroke = {}
Stroke.__index = Stroke

-- the loaded chunk holding s (nil if none)
function Stroke:chunkAt(s)
  for _, c in ipairs(self.list) do
    if s >= c.s[1] - 1e-6 and s <= c.s[c.n] + 1e-6 then return c end
  end
end

function Stroke:at(s)
  local c = self:chunkAt(s)
  return c and sample(c, s)
end

-- lane run at s (or a default from the tags and extents)
function Stroke:lanesAt(s)
  for _, r in ipairs(self.lanes) do
    if s >= r.s0 - 1e-6 and s <= r.s1 + 1e-6 then return r end
  end
  return self.defaultLanes
end

-- the host's projection of (x, y) near host s `hint`: s, u (+ left), frame
function Stroke:project(x, y, hint)
  local best, bs, bu
  for _, c in ipairs(self.list) do
    if c.s[c.n] >= hint - 80 and c.s[1] <= hint + 80 then
      for i = 1, c.n - 1 do
        if c.s[i + 1] >= hint - 80 and c.s[i] <= hint + 80 then
          local ax, ay, bx, by = c.x[i], c.y[i], c.x[i + 1], c.y[i + 1]
          local dx, dy = bx - ax, by - ay
          local l2 = dx * dx + dy * dy
          local t = l2 > EPS and max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) or 0
          local px, py = ax + dx * t, ay + dy * t
          local d2 = (x - px) ^ 2 + (y - py) ^ 2
          if not best or d2 < best then
            best, bs = d2, c.s[i] + (c.s[i + 1] - c.s[i]) * t
            local l = sqrt(max(l2, EPS))
            bu = ((x - ax) * -dy + (y - ay) * dx) / l
          end
        end
      end
    end
  end
  if not best then return nil end
  local st = self:at(bs)
  -- the lateral offset in the interpolated frame (the chord's normal differs by the station turn)
  local nx, ny = -sin(st.h), cos(st.h)
  bu = (x - st.x) * nx + (y - st.y) * ny
  return bs, bu, st
end

-- the host's surface plane (its own section, extended) at (x, y)
function Stroke:planeZ(x, y, hint)
  local s, u, st = self:project(x, y, hint)
  if not s then return nil end
  return st.z + (u >= 0 and st.fl or st.fr) * abs(u), s, u, st
end

-- along the ray (x, y) + t (dx, dy), the t where the host's lateral offset is `target` (secant steps)
function Stroke:rayTo(x, y, dx, dy, target, hint)
  local function f(t)
    local s, u = self:project(x + dx * t, y + dy * t, hint)
    return s and (u - target) or nil
  end
  local t0, t1 = 0, 1
  local f0, f1 = f(t0), f(t1)
  if not f0 or not f1 then return nil end
  for _ = 1, 4 do
    if abs(f1 - f0) < 1e-9 then break end
    local t2 = t1 - f1 * (t1 - t0) / (f1 - f0)
    t0, f0 = t1, f1
    t1, f1 = t2, f(t2)
    if not f1 then return nil end
    if abs(f1) < 1e-4 then break end
  end
  return t1
end

local function inRect(rect, x, y) return x >= rect[1] and x < rect[3] and y >= rect[2] and y < rect[4] end

local function noPause() end
-- a pause point every TUNE.pauseRows calls (the long loops of the ground steps, marks and rails)
local function pacer(plan)
  local pause = plan.opts and plan.opts.pause or noPause
  local every, n = M.TUNE.pauseRows or 48, 0
  return function() n = n + 1; if n >= every then n = 0; pause() end end
end
M.pacer = pacer

-- the footprint index's bins (m), and the grid of the chunks whose rows wait for a query (m)
local BIN = 16
local LAZY = 64
local function binKey(bx, by) return (bx + 1048576) * 2097152 + by + 1048576 end

local function byClass(t, S) return t[S.link and 'link' or S.base] or t.default end

-- the lane block of run `run` at s fitted into the paved section (extents as given)
local function fitBlock(run, availL, availR)
  local n = max(1, run.f + run.b + run.both)
  local half = n * run.width / 2
  local m = M.MARKS.inside
  local left, right = half, -half
  if left > availL - m then local d = left - (availL - m); left, right = left - d, right - d end
  if right < -(availR - m) then local d = -(availR - m) - right; left, right = left + d, right + d end
  if left > availL - m then left = availL - m end
  if left - right < n * 2.5 then left, right = half, -half end
  return {n = n, f = run.f, b = run.b, both = run.both, w = (left - right) / n, left = left, right = right}
end

-- ---------------------------------------------------------------------------------------------------
-- Plan
-- ---------------------------------------------------------------------------------------------------

-- bundle6 (M2): free ends of swept strokes that lie on another swept stroke's surface. S.endJoins = {{s = end s, into = +1
-- (-1: the body lies at smaller s), T, host, hs (the host's s there)}}; M.row blends the rows over T to the host's plane
local function rankOf(S)
  local r = ({motorway = 7, trunk = 6, primary = 5, secondary = 4, tertiary = 3})[S.base] or 2
  return S.link and r - .5 or r
end
local function strokeList(plan)
  local list = {}
  for _, S in pairs(plan.strokes) do
    if S.swept and #S.list > 0 then
      list[#list + 1] = S
      for _, c in ipairs(S.list) do
        if not c.sbox then
          local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
          for i = 1, c.n do
            local x, y = c.x[i], c.y[i]
            if x < x0 then x0 = x end
            if x > x1 then x1 = x end
            if y < y0 then y0 = y end
            if y > y1 then y1 = y end
          end
          c.sbox = {x0 - 25, y0 - 25, x1 + 25, y1 + 25}
        end
      end
    end
  end
  return list
end
local function endJoins(plan)
  local T = M.TUNE.endJoin
  if not (T and T > 0) then return end
  local list = strokeList(plan)
  for _, S in ipairs(list) do
    do
      for e = 1, 2 do
        local sEnd = e == 1 and S.D0 or S.D1
        local into = e == 1 and 1 or -1
        -- a slip road's host end has its own plane rule; the other end is free
        local free = not (S.cutA and ((S.blendDir == 1 and e == 1) or (S.blendDir == -1 and e == 2)))
        for _, r in ipairs(S.cuts) do if abs(r[1] - sEnd) < 1 or abs(r[2] - sEnd) < 1 then free = false end end
        for _, sn in ipairs(S.snaps) do if abs(sn.s - sEnd) < 1 then free = false end end
        local st = free and S:at(sEnd)
        if st then
          local best
          for _, H in ipairs(list) do
            if H ~= S and (rankOf(H) > rankOf(S) or (rankOf(H) == rankOf(S) and tostring(H.id) < tostring(S.id))) then
              local nearS, nearD
              for _, c in ipairs(H.list) do
                local b = c.sbox
                if st.x >= b[1] and st.x <= b[3] and st.y >= b[2] and st.y <= b[4] then
                  for i = 1, c.n - 1 do
                    local ax, ay, bx, by = c.x[i], c.y[i], c.x[i + 1], c.y[i + 1]
                    local dx, dy = bx - ax, by - ay
                    local l2 = dx * dx + dy * dy
                    local t = l2 > EPS and max(0, min(1, ((st.x - ax) * dx + (st.y - ay) * dy) / l2)) or 0
                    local d2 = (st.x - ax - dx * t) ^ 2 + (st.y - ay - dy * t) ^ 2
                    if not nearD or d2 < nearD then nearD, nearS = d2, c.s[i] + (c.s[i + 1] - c.s[i]) * t end
                  end
                end
              end
              if nearS and nearD < 900 and (not best or nearD < best.d2) then
                local hs, hu, hst = H:project(st.x, st.y, nearS)
                if hs and hs >= H.D0 and hs <= H.D1 then
                  local ext = hu >= 0 and hst.eL or hst.eR
                  local hz = hst.z + (hu >= 0 and hst.fl or hst.fr) * abs(hu)
                  if abs(hu) <= ext + 1 and abs(hz - st.z) < .6 and abs(hz - st.z) > .01 then best = {d2 = nearD, H = H, hs = hs} end
                end
              end
            end
          end
            if best then
            S.endJoins = S.endJoins or {}
            S.endJoins[#S.endJoins + 1] = {s = sEnd, into = into, T = T, host = best.H, hs = best.hs}
          end
        end
      end
    end
  end
end

-- bundle7 (P2): where two swept strokes end side by side on one carriageway (the loops of a cloverleaf merging into a
-- fan plate, or leaving a link, without a junction record) their rows overlap for 15-30 m at heights that differ by up to
-- 27 cm (in testing): a surface over the driven lane. The lower-ranked
-- stroke (rankOf, then the id) takes the other plane (the same blend as endJoins) over the stretch from its drawn end in
-- which its paved edge lies in the other paved extent at the same level, and leaves it over `overlapEase` m; at a plate
-- mouth it keeps its exact section and joins the plane over `overlapLead` m. Ends at a junction record are left to the
-- record own heights. S.overlaps = {{s = drawn end s, into, hold, T, lead, host, hs}}
local function overlapJoins(plan)
  local T = M.TUNE.overlapEase
  if not (T and T > 0) then return end
  local list = strokeList(plan)
  local step, reach = 2, M.TUNE.overlapReach
  for _, S in ipairs(list) do
    -- the plates' cut ranges, adjacent ones joined (a record's mouths on both sides of its node)
    local ranges = {}
    local sorted = {}
    for _, r in ipairs(S.cuts) do sorted[#sorted + 1] = r end
    table.sort(sorted, function(p, q) return p[1] < q[1] end)
    for _, r in ipairs(sorted) do
      local g = ranges[#ranges]
      if g and r[1] <= g.b + 1 then
        g.b = max(g.b, r[2]); g.J = g.J or r.J
      else
        ranges[#ranges + 1] = {a = r[1], b = r[2], J = r.J}
      end
    end
    -- the drawn ends: the stroke's own, and either side of every plate inside it (a loop runs through a fan plate)
    local ends = {}
    for e = 1, 2 do
      local sEnd = e == 1 and S.D0 or S.D1
      local into = e == 1 and 1 or -1
      local hosted = S.cutA and ((S.blendDir == 1 and e == 1) or (S.blendDir == -1 and e == 2))
      if not hosted then
        local sEff, pinned, recorded = sEnd, false, false
        for _, g in ipairs(ranges) do
          local near1, near2 = abs(g.a - sEnd) < 1, abs(g.b - sEnd) < 1
          if near1 or near2 then
            pinned, recorded = true, recorded or g.J
            local far = near1 and g.b or g.a
            if (far - sEff) * into > 0 then sEff = far end
          end
        end
        ends[#ends + 1] = {sEff, into, pinned, recorded}
      end
    end
    for _, g in ipairs(ranges) do
      if g.a > S.D0 + 1 and g.b < S.D1 - 1 then
        ends[#ends + 1] = {g.a, -1, true, g.J}
        ends[#ends + 1] = {g.b, 1, true, g.J}
      end
    end
    for _, en in ipairs(ends) do
      local sEff, into, pinned, recorded = en[1], en[2], en[3], en[4]
      local st = not recorded and S:at(sEff)
      if st then
        local best
        for _, H in ipairs(list) do
          if H ~= S and (rankOf(H) > rankOf(S) or (rankOf(H) == rankOf(S) and tostring(H.id) < tostring(S.id))) then
            local nearS, nearD
            for _, c in ipairs(H.list) do
              local b = c.sbox
              if st.x >= b[1] and st.x <= b[3] and st.y >= b[2] and st.y <= b[4] then
                for i = 1, c.n - 1 do
                  local ax, ay, bx, by = c.x[i], c.y[i], c.x[i + 1], c.y[i + 1]
                  local dx, dy = bx - ax, by - ay
                  local l2 = dx * dx + dy * dy
                  local t = l2 > EPS and max(0, min(1, ((st.x - ax) * dx + (st.y - ay) * dy) / l2)) or 0
                  local d2 = (st.x - ax - dx * t) ^ 2 + (st.y - ay - dy * t) ^ 2
                  if not nearD or d2 < nearD then nearD, nearS = d2, c.s[i] + (c.s[i + 1] - c.s[i]) * t end
                end
              end
            end
            if nearS and nearD < 400 then
              local hint, hold, hs0 = nearS, nil, nil
              local k = 0
              while k * step <= reach do
                local sv = sEff + into * k * step
                if sv < S.D0 - 1e-6 or sv > S.D1 + 1e-6 then break end
                local q = S:at(sv)
                if not q then break end
                local hs, hu, hst = H:project(q.x, q.y, hint)
                if not hs or hs < H.D0 or hs > H.D1 then break end
                hint = hs
                if not hs0 then hs0 = hs end
                local same = cos(q.h - hst.h) >= 0
                local extH = hu >= 0 and hst.eL or hst.eR
                local extS = ((hu >= 0) == same) and q.eR or q.eL
                local hz = hst.z + (hu >= 0 and hst.fl or hst.fr) * abs(hu)
                if abs(hu) - extS < extH - M.TUNE.overlapIn and abs(hz - q.z) < .6 then hold = k * step else break end
                k = k + 1
              end
              if hold and hold >= step and (not best or hold > best.hold) then best = {H = H, hs = hs0, hold = hold} end
            end
          end
        end
        if best then
          S.overlaps = S.overlaps or {}
          local ov = {s = sEff, into = into, hold = best.hold, T = T, lead = pinned and M.TUNE.overlapLead or 0, host = best.H, hs = best.hs}
          S.overlaps[#S.overlaps + 1] = ov
          -- a fan plate's mouth of this stroke lies in the other's plane too (the snap's three points are the plate's own:
          -- the fan's ring and the stroke's end row both read them)
          if pinned and M.TUNE.overlapMouth then
            for _, sn in ipairs(S.snaps) do
              if abs(sn.s - sEff) < 1e-3 then
                for _, pt in ipairs({sn.R, sn.C, sn.L}) do
                  local hz = best.H:planeZ(pt[1], pt[2], best.hs)
                  if hz and abs(hz - pt[3]) < .6 then pt[3] = hz; ov.lead = 0 end
                end
              end
            end
          end
        end
      end
    end
  end
end

-- the exact sections' pull on a junction surface point: blend (x, y, z) to the nearest points of the rows
-- R-C-L (the mouths' exact sections, the strokes' ends at the junction's internal pieces; inverse distance,
-- falling to 0 at TUNE.plateJoin m); exact on a row
local function joinDelta(rows, x, y, z)
  local B = M.TUNE.plateJoin
  local sw, sd, wmax = 0, 0, 0
  for _, m in ipairs(rows) do
    local best, bz
    for q = 1, 2 do
      local P, Q = q == 1 and m.R or m.C, q == 1 and m.C or m.L
      local dx, dy = Q[1] - P[1], Q[2] - P[2]
      local l2 = dx * dx + dy * dy
      if l2 > 1e-8 then
        local t = max(0, min(1, ((x - P[1]) * dx + (y - P[2]) * dy) / l2))
        local d = sqrt((x - P[1] - t * dx) ^ 2 + (y - P[2] - t * dy) ^ 2)
        if not best or d < best then best, bz = d, P[3] + (Q[3] - P[3]) * t end
      end
    end
    if best and best < B then
      local w = smooth(1 - best / B)
      local lam = w / (best + .02) ^ 2
      sw, sd, wmax = sw + lam, sd + lam * (bz - z), max(wmax, w)
    end
  end
  if sw > 0 then return wmax * sd / sw end
  return 0
end

-- every junction surface's verts and island rings joined to the exact rows of its mouths and of the swept strokes'
-- ends at its internal pieces (before the footprints index the plates: the tris share the vert tables)
local function joinPlates(plan)
  if not (M.TUNE.plateJoin > 0) then return end
  for _, pl in ipairs(plan.plates) do
    local J = pl.junction
    if J then
    local rows = {}
    for _, m in ipairs(J.mouths) do rows[#rows + 1] = m end
    for _, p in ipairs(J.internal) do
      local S = plan.strokes[p.stroke]
      if S and S.swept then
        for _, sv in ipairs({p.s0, p.s1}) do
          for _, c in ipairs(S.list) do
            if sv >= c.s[1] - 1e-6 and sv <= c.s[c.n] + 1e-6 then
              local row = M.row(plan, S, c, sv)
              rows[#rows + 1] = {R = row.R, C = row.M, L = row.L}
              break
            end
          end
        end
      end
    end
    -- from the record's own heights (a record read once serves several plans)
    local z0 = J.z0
    if not z0 then
      z0 = {verts = {}, rings = {}}
      for k, v in ipairs(J.verts) do z0.verts[k] = v[3] end
      for k, isl in ipairs(J.islands) do
        z0.rings[k] = {}
        for q, pt in ipairs(isl.ring) do z0.rings[k][q] = pt[3] end
      end
      J.z0 = z0
    end
    for k, v in ipairs(J.verts) do v[3] = z0.verts[k] + joinDelta(rows, v[1], v[2], z0.verts[k]) end
    for k, isl in ipairs(J.islands) do
      for q, pt in ipairs(isl.ring) do pt[3] = z0.rings[k][q] + joinDelta(rows, pt[1], pt[2], z0.rings[k][q]) end
    end
    end
  end
end

-- data: roads3.merge of the cells round the tile. opts: rect (the tile), classes (a set of highways),
-- tileKey, lift, material(tags) -> material kind, markKind ('mark', 'mark_yellow' or nil),
-- deckDepth(tags), pause, far (no items: far surfaces only).
-- the lane count of S at s (its run there, or the default from its tags and extents)
local function lanesOf(S, s)
  local r = S:lanesAt(s)
  return r.f + r.b + r.both
end

-- B (mouth m at node nd) drawn along H's edge line from the node to where they part (TUNE.goreGap): the fork rule of
-- M.plan. far: the lane count beyond the node (H's other side, or the road the two end on), nil if none
local function forkAlong(T, nd, H, B, m, far)
  local b0 = B:at(m.s)
  local best, bm
  for _, hm in ipairs(nd.mouths) do
    if hm.stroke == H.id then
      local h0 = H:at(hm.s)
      if b0 and h0 then
        local c = cos(b0.h) * cos(h0.h) * m.dir * hm.dir + sin(b0.h) * sin(h0.h) * m.dir * hm.dir
        if not best or c > best then best, bm = c, hm end
      end
    end
  end
  if best and best > cos(T.forkAngle) then
    -- where they part: B's inner edge clear of H's edge (marching away from the node)
    local part, sideH
    for d = 2, T.forkReach, 2 do
      local sb = m.s + m.dir * d
      local st = B:at(sb)
      if not st then break end
      local hs, hu, hst = H:project(st.x, st.y, bm.s + bm.dir * d)
      if not hs then break end
      local sgn = hu >= 0 and 1 or -1
      local inner = sgn > 0 and st.eR or st.eL      -- B's side towards H (H lies to B's -sgn... see innerSide)
      local turn = abs(cos(st.h - hst.h))
      -- B lies on H's side sgn; H lies to B's side -sgn (B's left is +, same sense as H's when aligned)
      local dirSame = cos(st.h - hst.h) >= 0
      local innerSide = dirSame and -sgn or sgn
      inner = innerSide > 0 and st.eL or st.eR
      local edgeH = sgn > 0 and hst.eL or hst.eR
      if abs(hu) - inner * max(.5, turn) - edgeH >= T.goreGap then part, sideH = d, sgn; B.forkInner = innerSide; break end
    end
    if part then
      local sB, dB, sH, dH = m.s, m.dir, bm.s, bm.dir
      B.fork = {node = nd.node, H = H}
      B.host, B.innerSide, B.hostSide = H, B.forkInner, sideH
      B.cutA, B.cutB = min(sB, sB + dB * part), max(sB, sB + dB * part)
      B.blend0, B.blendDir = sB + dB * part, dB
      B.hint = function(sv) return sH + dH * dB * (sv - sB) end
      B.hostEdge = function(hs) local hst = H:at(hs); return hst and (sideH > 0 and hst.eL or hst.eR) end
      -- the host's edge line along the branch: a lane line (narrow) where a lane leaves or joins the host there (its lanes on
      -- the node's far side are its own here plus the branch's: a lane split), else the host's through edge line; solid both
      H.edgeGaps[#H.edgeGaps + 1] = {a = min(sH, sH + dH * part), b = max(sH, sH + dH * part), side = sideH,
        lane = far ~= nil and far > lanesOf(H, sH + dH) and far == lanesOf(H, sH + dH) + lanesOf(B, sB + dB)}
      -- a lane split (the pack's 'split' role, fast_roads.lane_splits): the branch takes its lanes from the host at the
      -- node, where the exporter steps the host's section over 5 cm; the host's lane block steps there too instead of
      -- moving over the window (its lines crossed the branch's lane for 20 m either side of the node)
      local role = nd.roles and nd.roles[B.id]
      if role and role.role == 'split' and role.host == H.id then
        B.split = true
        H.edgeGaps[#H.edgeGaps].lane = true
        for _, w in ipairs(H.windows) do
          if abs(w.r0.s1 - sH) < .1 then w.a, w.b = w.r0.s1, w.r0.s1 end
        end
      end
    end
  end
end

function M.plan(data, opts)
  local T = M.TUNE
  local pause = opts.pause or function() end
  local rect = opts.rect
  local classes = opts.classes or M.CLASSES.fast
  local plan = {strokes = {}, segments = {}, covered = {}, plated = {}, stats = {chunks = 0, owned = 0, plates = 0},
    rect = rect, opts = opts, lift = opts.lift or T.lift, conformal = opts.conformal}
  local tick = pacer(plan)
  -- strokes
  for sid, byIndex in pairs(data.byStroke) do
    local list = {}
    for _, c in pairs(byIndex) do list[#list + 1] = c end
    table.sort(list, function(a, b) return a.index < b.index end)
    local first = list[1]
    local S = setmetatable({id = sid, list = list, chunks = byIndex, highway = first.highway, oneway = first.oneway,
      closed = first.closed, d0 = first.d0, d1 = first.d1, length = first.length}, Stroke)
    S.base = baseClass(S.highway)
    S.link = S.highway ~= S.base
    S.swept = classes[S.highway] == true
    local ev, fu, stc, ln, at, parts = {}, {}, {}, {}, {}, {}
    for _, c in ipairs(list) do
      for _, e in ipairs(c.events) do ev[#ev + 1] = e end
      for _, f in ipairs(c.furniture) do fu[#fu + 1] = f end
      for _, r in ipairs(c.structures) do stc[#stc + 1] = r end
      for _, r in ipairs(c.lanes) do ln[#ln + 1] = r end
      for _, a in ipairs(c.attach) do at[#at + 1] = a end
      for _, p in ipairs(c.parts) do parts[#parts + 1] = p end
    end
    S.events = dedupe(ev, function(e) return e.kind .. e.branch .. e.taper end)
    S.furniture = dedupe(fu, function(f) return f.kind .. f.side .. f.s0 .. ':' .. f.s1 .. f.way end)
    S.structures = dedupe(stc, function(r) return r.kind .. r.s0 .. ':' .. r.s1 end)
    S.lanes = dedupe(ln, function(r) return r.s0 .. ':' .. r.s1 end)
    S.attach = at[1]
    S.parts = dedupe(parts, function(p) return p.segment .. ':' .. p.s0 end)
    S.tags = (S.parts[1] or NONE).tags or NONE
    S.urban = false
    local speed = tostring(S.tags.maxspeed or ''):lower()
    if speed:find('urban', 1, true) or (number(speed, 999) <= 50) then S.urban = true end
    S.cuts, S.snaps = {}, {}
    plan.strokes[sid] = S
    tick()
  end
  -- default lanes: from the tags of the first part and the extents
  for _, S in pairs(plan.strokes) do
    local c = S.list[1]
    local width = c.eL[1] + c.eR[1]
    local n = math.max(1, math.floor(number(S.tags.lanes, S.oneway and math.max(1, math.floor(width / 3.5 + .2)) or 2)))
    local f, b = n, 0
    if not S.oneway then b = math.floor(n / 2); f = n - b end
    S.defaultLanes = {s0 = -math.huge, s1 = math.huge, f = f, b = b, both = 0, width = math.min(3.75, width / n)}
  end
  -- covered v2 segments, and the segment of each (stroke, s) for the cuts of v2 mouths at plates
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, p in ipairs(S.parts) do plan.covered[tostring(p.segment)] = true end
    end
  end
  local function segmentAt(S, s, dir)
    local probe = s + dir * .5
    for _, p in ipairs(S.parts) do
      if probe >= p.s0 - 1e-6 and probe <= p.s1 + 1e-6 then return tostring(p.segment) end
    end
  end
  -- transitions: lane gains and drops inside a stroke (windows over byClass(TUNE.lanes)), and joints
  -- where two swept strokes meet straight on (each end blends to the pair's mean section)
  for _, S in pairs(plan.strokes) do
    S.joints, S.windows, S.edgeGaps = {}, {}, {}
    local runs = {}
    for _, r in ipairs(S.lanes) do runs[#runs + 1] = r end
    table.sort(runs, function(a, b) return a.s0 < b.s0 end)
    local len = byClass(T.lanes, S)
    for k = 1, #runs - 1 do
      local a, b = runs[k], runs[k + 1]
      if abs(a.s1 - b.s0) < .05 and (a.f + a.b + a.both ~= b.f + b.b + b.both or abs(a.width - b.width) > .05) then
        -- the window: where the exporter's paved width changes round the run boundary (its taper sits
        -- on the wider side), else byClass(TUNE.lanes) centred on it
        local s0, s1 = a.s1 - len / 2, a.s1 + len / 2
        local function width(sv) local st = S:at(sv); return st and st.eL + st.eR end
        local reach = 1.2 * len
        local w0, w1 = width(a.s1 - reach), width(a.s1 + reach)
        if w0 and w1 and abs(w1 - w0) > .1 then
          local first, last
          for d = -reach, reach, 2 do
            local w = width(a.s1 + d)
            if w and not first and abs(w - w0) > .02 then first = a.s1 + d - 2 end
            if w and abs(w - w1) > .02 then last = a.s1 + d + 2 end
          end
          if first and last and last > first + 4 then s0, s1 = min(first, a.s1 - 1), max(last, a.s1 + 1) end
        end
        S.windows[#S.windows + 1] = {a = s0, b = s1, r0 = a, r1 = b}
      end
    end
  end
  for _, nd in ipairs(data.nodes) do
    if nd.kind == 'joint' and #nd.mouths == 2 and nd.mouths[1].stroke ~= nd.mouths[2].stroke then
      local m1, m2 = nd.mouths[1], nd.mouths[2]
      local A, B = plan.strokes[m1.stroke], plan.strokes[m2.stroke]
      if A and B and A.swept and B.swept then
        local a, b = A:at(m1.s), B:at(m2.s)
        if a and b then
          -- B's section in A's frame: the same direction through the node unless both leave or both arrive
          local swap = m1.dir == m2.dir
          local bL, bR = swap and b.eR or b.eL, swap and b.eL or b.eR
          local bfl, bfr = swap and b.fr or b.fl, swap and b.fl or b.fr
          local ka, kb = M.laneBlock(A, m1.s), M.laneBlock(B, m2.s)
          local kbl, kbr = swap and -kb.right or kb.left, swap and -kb.left or kb.right
          local len = max(byClass(T.joint, A), byClass(T.joint, B))
          local mean = {eL = (a.eL + bL) / 2, eR = (a.eR + bR) / 2, fl = (a.fl + bfl) / 2, fr = (a.fr + bfr) / 2, z = (a.z + b.z) / 2,
            left = (ka.left + kbl) / 2, right = (ka.right + kbr) / 2}
          A.joints[#A.joints + 1] = {s = m1.s, dir = m1.dir, T = len, eL = mean.eL, eR = mean.eR, fl = mean.fl, fr = mean.fr,
            z = mean.z, left = mean.left, right = mean.right}
          B.joints[#B.joints + 1] = {s = m2.s, dir = m2.dir, T = len, eL = swap and mean.eR or mean.eL, eR = swap and mean.eL or mean.eR,
            fl = swap and mean.fr or mean.fl, fr = swap and mean.fl or mean.fr, z = mean.z,
            left = swap and -mean.right or mean.left, right = swap and -mean.left or mean.right}
        end
      end
    end
  end
  -- phase 3: the exporter's junction surfaces (data.junctions, roads3 junction blocks). At their nodes
  -- they replace the fan plates and the fork rule: every mouth is cut at the record's cut with its exact
  -- section, internal pieces (between two nodes of one junction) are not drawn, the v2 roads there are
  -- cut where the surface meets them, and the approaches' records give the markings and nav knots.
  plan.junctionAt, plan.jplates = {}, {}
  for _, J in ipairs(data.junctions or NONE) do
    local any = false
    for _, m in ipairs(J.mouths) do local S = plan.strokes[m.stroke]; if S and S.swept then any = true end end
    for _, p in ipairs(J.internal) do local S = plan.strokes[p.stroke]; if S and S.swept then any = true end end
    if any and #J.tris > 0 then
      local ring = J.kind == 'roundabout' or J.kind == 'circular'
      for _, n in ipairs(J.nodes) do plan.junctionAt[n] = J end
      local radius, cutBy, sc = 0, {}, {}
      for k, m in ipairs(J.mouths) do
        radius = max(radius, m.cut)
        local S = plan.strokes[m.stroke]
        sc[k] = m.s + m.dir * m.cut
        if S and S.swept then
          S.cuts[#S.cuts + 1] = {min(m.s, sc[k]), max(m.s, sc[k]), J = true}
          S.snaps[#S.snaps + 1] = {s = sc[k], L = m.L, C = m.C, R = m.R}
          if not ring then
            S.navKnots = S.navKnots or {}
            S.navKnots[#S.navKnots + 1] = {s = sc[k], id = 'jm:' .. J.id .. ':' .. m.stroke .. ':' .. m.dir}
            S.navGaps = S.navGaps or {}
            S.navGaps[#S.navGaps + 1] = {min(m.s, sc[k]), max(m.s, sc[k])}
          end
        elseif S then
          local seg = segmentAt(S, m.s, m.dir)
          if seg then cutBy[m.node] = cutBy[m.node] or {}; cutBy[m.node][seg] = m.cut end
        end
        -- a road of the generator (not swept) is drawn flat at its centre line's height: the plate's edge there is
        -- too (the exporter's section edges carry the cross-fall: 10-17 cm under the road they join, in testing)
        if M.TUNE.plateFlatJoin and not (S and S.swept) then
          m.R[3], m.L[3] = m.C[3], m.C[3]
        end
      end
      for _, p in ipairs(J.internal) do
        local S = plan.strokes[p.stroke]
        if S and S.swept then
          S.cuts[#S.cuts + 1] = {p.s0, p.s1, J = true}
          if not ring then S.navGaps = S.navGaps or {}; S.navGaps[#S.navGaps + 1] = {p.s0, p.s1} end
        elseif S then
          -- a v2 piece between two nodes of the junction goes whole (the generator cuts at most half of it per end)
          local seg = segmentAt(S, p.s0, 1)
          if seg then for _, n in ipairs(J.nodes) do cutBy[n] = cutBy[n] or {}; cutBy[n][seg] = 1e9 end end
        end
      end
      for _, n in ipairs(J.nodes) do plan.plated[n] = {radius = radius, cut = cutBy[n] or {}} end
      -- approaches: the stop line zone (no lane or centre lines from the stop line to the surface) and
      -- the solid zone before it (lane lines solid where they are turn lanes or the junction is signalled)
      for _, a in ipairs(J.approaches) do
        local S = plan.strokes[a.stroke]
        if S and S.swept then
          local cutS
          for k, m in ipairs(J.mouths) do if m.stroke == a.stroke and m.dir == a.dir then cutS = sc[k] end end
          if cutS then
            local sigma = -a.dir
            local ap = {J = J, dir = a.dir, sigma = sigma, stop = a.stop, cut = cutS, control = a.control, turns = a.turns,
              lanes = a.lanes, crossings = a.crossings}
            S.approaches = S.approaches or {}
            S.approaches[#S.approaches + 1] = ap
            S.zones = S.zones or {}
            if a.control ~= 'none' then
              S.zones[#S.zones + 1] = {a = min(a.stop, cutS), b = max(a.stop, cutS), kind = 'clear', ap = ap}
            end
            if a.control == 'signals' or (a.turns ~= '' and a.lanes >= 2) then
              local L = S.urban and M.MARKS.solid.urban or M.MARKS.solid.rural
              local far = a.stop - sigma * L
              S.zones[#S.zones + 1] = {a = min(a.stop, far), b = max(a.stop, far), kind = 'solid', ap = ap}
            end
          end
        end
      end
      local tris = {}
      for _, t in ipairs(J.tris) do tris[#tris + 1] = {J.verts[t[1]], J.verts[t[2]], J.verts[t[3]]} end
      local rows, seen = {}, {}
      for _, m in ipairs(J.mouths) do
        local S = plan.strokes[m.stroke]
        if S and not seen[S] then seen[S] = true; rows[#rows + 1] = {S = S} end
      end
      plan.jplates[#plan.jplates + 1] = {node = {node = J.id, x = J.x, y = J.y, z = J.z, mouths = NONE, plate = NONE},
        rows = rows, tris = tris, junction = J, owned = inRect(rect, J.x, J.y)}
    end
  end
  -- forks, diverges and merges of swept roads that are not lane-attached: a branch leaving a through
  -- road within TUNE.forkAngle is drawn cut along that road's edge line and in its plane until the two
  -- part, instead of a fan plate (two surfaces overlapping at their own cross-falls were steps)
  -- forks, diverges and merges of swept roads that are not lane-attached: a branch leaving a through
  -- road within TUNE.forkAngle is drawn cut along that road's edge line and in its plane until the two
  -- part, instead of a fan plate (two surfaces overlapping at their own cross-falls were steps)
  for _, nd in ipairs(data.nodes) do
    tick()
    if (nd.kind == 'diverge' or nd.kind == 'merge' or nd.kind == 'fork' or nd.kind == 'junction') and not plan.junctionAt[nd.node] then
      local count = {}
      for _, m in ipairs(nd.mouths) do count[m.stroke] = (count[m.stroke] or 0) + 1 end
      local H
      for sid, n in pairs(count) do
        local cand = plan.strokes[sid]
        if n >= 2 and cand and cand.swept and (not H or (cand.base == 'motorway' and H.base ~= 'motorway')) then H = cand end
      end
      for _, m in ipairs(H and nd.mouths or NONE) do
        local B = plan.strokes[m.stroke]
        if B and B ~= H and B.swept and count[m.stroke] == 1 and not B.attach and not B.fork then
          local far
          for _, hm in ipairs(nd.mouths) do
            if hm.stroke == H.id and hm.dir ~= m.dir then far = lanesOf(H, hm.s + hm.dir) end
          end
          forkAlong(T, nd, H, B, m, far)
        end
      end
      -- two one-way strokes that end (or start) at the node side by side, neither along a through road (links ending on a
      -- third road: its fan plate left the wedge between them unpaved and without a gore): the lower-ranked one is drawn
      -- along the other as a fork branch, the plate keeps the rest
      local free = {}
      for _, m in ipairs(nd.mouths) do
        local S = plan.strokes[m.stroke]
        if S and S.swept and S.oneway and count[m.stroke] == 1 and not S.attach and not S.fork and S ~= H then free[#free + 1] = m end
      end
      for i = 1, #free - 1 do
        for j = i + 1, #free do
          local a, b = free[i], free[j]
          local A, B = plan.strokes[a.stroke], plan.strokes[b.stroke]
          if a.dir == b.dir and not A.fork and not B.fork then
            if rankOf(B) > rankOf(A) or (rankOf(B) == rankOf(A) and tostring(B.id) < tostring(A.id)) then a, b, A, B = b, a, B, A end
            -- the road they end on: the one other mouth on the node's far side
            local far, others = nil, 0
            for _, o in ipairs(nd.mouths) do
              if o.stroke ~= A.id and o.stroke ~= B.id then
                others = others + 1
                if o.dir ~= a.dir and plan.strokes[o.stroke] then far = lanesOf(plan.strokes[o.stroke], o.s + o.dir) end
              end
            end
            forkAlong(T, nd, A, B, b, others == 1 and far or nil)
          end
        end
      end
    end
  end
  -- fan plates: drawn where a swept stroke has a mouth; their cuts on every swept stroke
  plan.plates = plan.jplates
  for _, nd in ipairs(data.nodes) do
    tick()
    if #nd.plate > 0 and not plan.junctionAt[nd.node] then
      local any = false
      for _, row in ipairs(nd.plate) do
        local S = plan.strokes[row.stroke]
        if S and S.swept then any = true end
      end
      local kept = {}
      for _, row in ipairs(nd.plate) do
        local S = plan.strokes[row.stroke]
        if not (S and ((S.attach and S.attach.node == nd.node) or (S.fork and S.fork.node == nd.node))) then kept[#kept + 1] = row end
      end
      local distinct, nDistinct = {}, 0
      for _, row in ipairs(kept) do
        if not distinct[row.stroke] then distinct[row.stroke] = true; nDistinct = nDistinct + 1 end
      end
      if nDistinct < 2 then any = false end
      if any then
        local rows, radius = {}, 0
        for _, row in ipairs(kept) do
          radius = max(radius, row.cut)
          local s
          for _, m in ipairs(nd.mouths) do
            if m.stroke == row.stroke and m.dir == row.dir then s = m.s end
          end
          local S = plan.strokes[row.stroke]
          rows[#rows + 1] = {row = row, s = s, S = S}
          if S and s then
            local sc = s + row.dir * row.cut
            if S.swept then
              S.cuts[#S.cuts + 1] = {min(s, sc), max(s, sc)}
              S.snaps[#S.snaps + 1] = {s = sc, L = row.L, C = row.C, R = row.R}
              -- the plate's section differs from the road's own there (a through pair of two classes
              -- meets with one section): the road blends into it over byClass(TUNE.joint)
              local own = S:at(sc)
              if own and M.TUNE.plateBlend then
                local eL = sqrt((row.L[1] - row.C[1]) ^ 2 + (row.L[2] - row.C[2]) ^ 2)
                local eR = sqrt((row.R[1] - row.C[1]) ^ 2 + (row.R[2] - row.C[2]) ^ 2)
                local fl = eL > .1 and (row.L[3] - row.C[3]) / eL or own.fl
                local fr = eR > .1 and (row.R[3] - row.C[3]) / eR or own.fr
                if abs(eL - own.eL) + abs(eR - own.eR) > .05 or abs(fl - own.fl) + abs(fr - own.fr) > .002 then
                  local blk = M.laneBlock(S, sc)
                  S.joints[#S.joints + 1] = {s = sc, dir = row.dir, T = byClass(T.joint, S), eL = eL, eR = eR, fl = fl, fr = fr,
                    z = row.C[3], left = blk.left + (eL - own.eL), right = blk.right - (eR - own.eR)}
                end
              end
            end
          end
        end
        local cut = {}
        for _, r in ipairs(rows) do
          if r.S and r.s and not r.S.swept then
            local seg = segmentAt(r.S, r.s, r.row.dir)
            if seg then cut[seg] = r.row.cut end
          end
        end
        plan.plated[nd.node] = {radius = radius, cut = cut}
        plan.plates[#plan.plates + 1] = {node = nd, rows = rows, owned = inRect(rect, nd.x, nd.y)}
      end
    end
  end
  -- exits and entries: the host's widened side clamped to its own edge where the slip road is drawn
  for _, S in pairs(plan.strokes) do
    S.D0, S.D1 = S.d0, S.d1
    S.clamps = {}
    if S.swept then
      for _, e in ipairs(S.events) do
        local B = plan.strokes[e.branch]
        local a, b = min(e.separate, e.nose), max(e.separate, e.nose)
        if B and B.swept and B.attach and B.attach.host == S.id and b > a then
          local at = S:at(e.nose)
          if at then
            S.clamps[#S.clamps + 1] = {a = a, b = b, side = e.side, base = e.side > 0 and at.eL or at.eR, event = e}
          end
        end
      end
    end
  end
  for _, S in pairs(plan.strokes) do
    if S.swept then
      local A = S.attach
      local H = A and plan.strokes[A.host]
      if H and H.swept and A.lane1 > A.lane0 then
        local hostEvent
        for _, e in ipairs(H.events) do if e.branch == S.id then hostEvent = e end end
        if hostEvent then
          S.host, S.hostEvent = H, hostEvent
          S.innerSide = -hostEvent.side
          local clampBase
          for _, cl in ipairs(H.clamps) do if cl.event == hostEvent then clampBase = cl.base end end
          S.hostBase = clampBase
          if A.kind == 'exit' and A.lane1 < S.d0 then
            S.D0, S.cutA, S.cutB = A.lane1, A.lane1, S.d0
            S.blend0, S.blendDir = S.d0, 1
          elseif A.kind == 'entry' and A.lane0 > S.d1 then
            S.D1, S.cutA, S.cutB = A.lane0, S.d1, A.lane0
            S.blend0, S.blendDir = S.d1, -1
          end
          S.hint = function(sv) return A.hostLane0 + (sv - A.lane0) * (A.hostLane1 - A.hostLane0) / max(EPS, A.lane1 - A.lane0) end
          S.hostEdge = function() return clampBase end
          S.widenOuter = true
          if not clampBase then S.D0, S.D1, S.cutA, S.cutB, S.host = S.d0, S.d1, nil, nil, nil end
        end
      end
    end
  end
  endJoins(plan)
  overlapJoins(plan)
  -- footprint index (every drawn row quad of the swept chunks near the tile) for rails, lines and tests;
  -- chunks further than TUNE.rowMargin get their rows only when a surface query reaches their box
  -- (M.ensureAt: the generator's sweptCrossing asks along every way of the cells, and its answers must not
  -- depend on how far the chunk is), those within TUNE.groundMargin feed the ground steps (c.ground)
  plan.bins = {}
  -- the box of every chunk's possible surface (a v2 way outside it meets no swept surface)
  plan.box = {math.huge, math.huge, -math.huge, -math.huge}
  plan.lazy = {}
  local rm, gm = T.rowMargin or math.huge, T.groundMargin or math.huge
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, c in ipairs(S.list) do
        plan.stats.chunks = plan.stats.chunks + 1
        local x0, y0, x1, y1, e = math.huge, math.huge, -math.huge, -math.huge, 0
        local cx, cy, ceL, ceR = c.x, c.y, c.eL, c.eR
        for i = 1, c.n do
          local x, y = cx[i], cy[i]
          if x < x0 then x0 = x end
          if x > x1 then x1 = x end
          if y < y0 then y0 = y end
          if y > y1 then y1 = y end
          if ceL[i] > e then e = ceL[i] end
          if ceR[i] > e then e = ceR[i] end
        end
        -- the widest row stays within twice the widest station extent (a slip road widened to its host's
        -- paved edge, a joint's mean section)
        e = 2 * e + 5
        c.box = {x0 - e, y0 - e, x1 + e, y1 + e}
        local box = c.box
        local pb = plan.box
        pb[1], pb[2], pb[3], pb[4] = min(pb[1], box[1]), min(pb[2], box[2]), max(pb[3], box[3]), max(pb[4], box[4])
        c.near = box[3] >= rect[1] - rm and box[1] <= rect[3] + rm and box[4] >= rect[2] - rm and box[2] <= rect[4] + rm
        c.ground = box[3] >= rect[1] - gm and box[1] <= rect[3] + gm and box[4] >= rect[2] - gm and box[2] <= rect[4] + gm
        local mid = floor((c.n - 1) / 2) + 1
        c.owned = inRect(rect, c.x[mid], c.y[mid])
        if c.owned then plan.stats.owned = plan.stats.owned + 1 end
        if c.near or c.owned then
          c.rows = M.rows(plan, S, c)
          M.index(plan, S, c)
          pause()
        else
          c.rows = NONE
          c.lazyS = S
          for gx = floor(box[1] / LAZY), floor(box[3] / LAZY) do
            for gy = floor(box[2] / LAZY), floor(box[4] / LAZY) do
              local k = binKey(gx, gy)
              local list = plan.lazy[k]
              if not list then list = {}; plan.lazy[k] = list end
              list[#list + 1] = c
            end
          end
        end
      end
    end
  end
  joinPlates(plan)
  M.plateRings(plan)
  return plan
end

-- ---------------------------------------------------------------------------------------------------
-- Rows: the drawn sections of a chunk
-- ---------------------------------------------------------------------------------------------------

local function insideAny(list, s)
  for _, r in ipairs(list) do if s > r[1] + 1e-6 and s < r[2] - 1e-6 then return true end end
  return false
end

-- the drawn intervals of chunk c of S: [D0, D1] minus the plate cuts
local function intervals(S, c)
  local a, b = max(c.s[1], S.D0), min(c.s[c.n], S.D1)
  if b - a < 1e-6 then return {} end
  local out, cur = {}, {a, b}
  local cuts = {}
  for _, r in ipairs(S.cuts) do if r[2] > a and r[1] < b then cuts[#cuts + 1] = r end end
  table.sort(cuts, function(p, q) return p[1] < q[1] end)
  for _, r in ipairs(cuts) do
    if r[1] > cur[1] + 1e-6 then out[#out + 1] = {cur[1], min(cur[2], r[1])} end
    cur[1] = max(cur[1], r[2])
    if cur[1] >= cur[2] - 1e-6 then break end
  end
  if cur[2] - cur[1] > 1e-6 then out[#out + 1] = cur end
  return out
end

-- break points inside [a, b]: stations, events, lane runs, structures, the slip's cut and ease
local function breaks(S, c, a, b)
  local set, list = {}, {}
  local function add(s)
    if s >= a - 1e-9 and s <= b + 1e-9 then
      s = max(a, min(b, s))
      local k = floor(s * 1000 + .5)
      if not set[k] then set[k] = true; list[#list + 1] = s end
    end
  end
  add(a); add(b)
  for i = 1, c.n do add(c.s[i]) end
  for _, e in ipairs(S.events) do
    add(e.taper); add(e.parallel); add(e.separate); add(e.nose)
    local nT = math.ceil((M.MARKS.taperRows or 0) * max(1, e.width / 3.5))
    for k = 1, nT - 1 do add(e.taper + (e.parallel - e.taper) * k / nT) end
  end
  for _, r in ipairs(S.lanes) do add(r.s0); add(r.s1) end
  for _, r in ipairs(S.structures) do add(r.s0); add(r.s1) end
  if S.cutA then
    add(S.cutA); add(S.cutB)
    for k = 1, 6 do add(S.blend0 + S.blendDir * M.TUNE.noseEase * k / 6) end
  end
  for _, p in ipairs(S.parts) do add(p.s0); add(p.s1) end
  for _, e in ipairs(S.endJoins or NONE) do for k = 0, 6 do add(e.s + e.into * e.T * k / 6) end end
  for _, e in ipairs(S.overlaps or NONE) do
    add(e.s + e.into * e.lead)
    for k = 0, 6 do add(e.s + e.into * (e.hold + e.T * k / 6)) end
  end
  for _, w in ipairs(S.windows) do add(w.a); add(w.b) end
  for _, z in ipairs(S.zones or NONE) do add(z.a); add(z.b) end
  for _, j in ipairs(S.joints) do
    for k = 1, 4 do add(j.s + j.dir * j.T * k / 4) end
  end
  table.sort(list)
  -- a station a millimetre from a break is the break
  local out = {}
  for _, s in ipairs(list) do
    if #out == 0 or s - out[#out] > 1e-3 then out[#out + 1] = s else
      for i = 1, c.n do if abs(c.s[i] - s) < 1e-9 then out[#out] = s end end
    end
  end
  return out
end

-- the slip road's weight of its own profile at s (0: in the host's plane)
local function ownWeight(S, s)
  if not S.cutA then return 1 end
  return smooth(S.blendDir * (s - S.blend0) / M.TUNE.noseEase)
end

-- the joints blending at s: {{weight, joint}} (applied in turn: each weight falls to 0 at its end, so
-- two overlapping blends on a short road stay continuous)
local function jointsAt(S, s)
  local out
  for _, j in ipairs(S.joints) do
    local d = (s - j.s) * j.dir
    if d >= -1e-6 and d < j.T then
      out = out or {}
      out[#out + 1] = {smooth(1 - d / j.T), j}
    end
  end
  return out or NONE
end
M.jointsAt = jointsAt
function M.jointAt(S, s)
  local list = jointsAt(S, s)
  return list[1] and list[1][1] or 0
end

-- the host's exporter extent on `side` at s (the widened paved edge, before the clamp)
local function hostOuterExtent(H, s, side)
  local st = H:at(s)
  if not st then return 0 end
  return side > 0 and st.eL or st.eR
end

-- one row at s: the section's three vertices R, M, L ({x, y, z}), its frame and extents
function M.row(plan, S, c, s)
  local st = sample(c, s)
  local nx, ny = -sin(st.h), cos(st.h)
  local eL, eR, fl, fr = st.eL, st.eR, st.fl, st.fr
  -- a joint with the next stroke: extents, cross-fall and height blend to the pair's mean at the node
  local joints = jointsAt(S, s)
  if joints[1] then
    local z = st.z
    for _, e in ipairs(joints) do
      local jw, jn = e[1], e[2]
      eL, eR = eL + (jn.eL - eL) * jw, eR + (jn.eR - eR) * jw
      fl, fr = fl + (jn.fl - fl) * jw, fr + (jn.fr - fr) * jw
      z = z + (jn.z - z) * jw
    end
    st = {x = st.x, y = st.y, z = z, h = st.h}
  end
  -- host: an exit's or entry's added lane grows S-shaped over its taper (smoothstep) from the paved edge just
  -- outside it, whatever the stations between hold (10 m apart, linear between them)
  for _, e in ipairs(S.events) do
    local L = e.parallel - e.taper
    local t = abs(L) > .5 and (s - e.taper) / L or -1
    if t > 0 and t < 1 then
      local out = S:at(e.taper - (L > 0 and 2 or -2))
      if out then
        local edge = (e.side > 0 and out.eL or out.eR) + e.width * smooth(t)
        if e.side > 0 then eL = edge else eR = edge end
      end
    end
  end
  -- host: the widened side clamped to its own edge where the slip road is drawn
  for _, cl in ipairs(S.clamps) do
    if s > cl.a - 1e-6 and s < cl.b + 1e-6 then
      if cl.side > 0 then eL = min(eL, cl.base) else eR = min(eR, cl.base) end
    end
  end
  local row = {s = s, x = st.x, y = st.y, z = st.z, h = st.h, nx = nx, ny = ny, fl = fl, fr = fr, w = 1}
  -- slip road: its inner side cut along the host's edge line, outer side to the host's paved edge, the
  -- surface in the host's plane up to the nose
  if S.cutA and s >= S.D0 - 1e-6 and s <= S.D1 + 1e-6 then
    local H, side = S.host, S.innerSide
    local hint = S.hint(s)
    row.hint = hint
    row.w = ownWeight(S, s)
    if s > S.cutA - 1e-6 and s < S.cutB + 1e-6 then
      local hostSide = S.hostSide or -side
      local base = S.hostEdge(hint)
      local t = base and H:rayTo(st.x, st.y, nx * side, ny * side, hostSide * base, hint)
      if t then
        -- the host's edge line on the branch's far side: nothing left to draw (a zero-width section)
        if side > 0 then eL = max(t, -eR) else eR = max(t, -eL) end
      end
      local hs, hu, hst
      if S.widenOuter then hs, hu, hst = H:project(st.x, st.y, hint) end
      if hs then
        local hostOuter = hostOuterExtent(H, hs, -side)
        local reach = hostOuter - abs(hu)
        local turn = cos(st.h - hst.h)
        if side > 0 then eR = max(eR, reach / max(.5, turn)) else eL = max(eL, reach / max(.5, turn)) end
      end
    end
  end
  if S.endJoins then
    for _, e in ipairs(S.endJoins) do
      local dd = (s - e.s) * e.into
      if dd < e.T then
        local w = smooth(max(0, dd) / e.T)
        if w < row.w then row.w, row.endHost, row.hint = w, e.host, e.hs end
      end
    end
  end
  if S.overlaps then
    for _, e in ipairs(S.overlaps) do
      local dd = (s - e.s) * e.into
      if dd < e.hold + e.T then
        local w = smooth(max(0, dd - e.hold) / e.T)
        if dd < e.lead then w = max(w, 1 - smooth(dd / e.lead)) end
        if w < row.w then row.w, row.endHost, row.hint = w, e.host, e.hs end
      end
    end
  end
  row.eL, row.eR = eL, eR
  if row.w < 1 then row.zfun = function(u) return M.point(plan, S, row, u)[3] end end
  local uM = max(-eR, min(eL, 0))
  row.uM = uM
  row.R = M.point(plan, S, row, -eR)
  row.M = M.point(plan, S, row, uM)
  row.L = M.point(plan, S, row, eL)
  -- a fan plate's mouth: its exact points
  for _, sn in ipairs(S.snaps) do
    if abs(sn.s - s) < 1e-3 then
      row.R, row.M, row.L = {sn.R[1], sn.R[2], sn.R[3]}, {sn.C[1], sn.C[2], sn.C[3]}, {sn.L[1], sn.L[2], sn.L[3]}
      row.snapped = true
    end
  end
  return row
end

-- a point of the row's surface u across (no lift)
function M.point(plan, S, row, u)
  local x, y = row.x + row.nx * u, row.y + row.ny * u
  local own = row.z + (u >= 0 and row.fl or row.fr) * abs(u)
  local H = row.endHost or S.host
  if row.w >= 1 or not H then return {x, y, own} end
  local hz = H:planeZ(x, y, row.hint)
  if not hz then return {x, y, own} end
  -- where the row lies in the host's plane (w = 0: a fork's or merge's lower-ranked branch, a stroke ending on another) it
  -- sits TUNE.planeSep under it, fading out with the blend: two coplanar surfaces over one stretch z-fought (in testing,
  -- 15.3, 37.5, 50.6; the cut leaves the rows near a nose over the host's lanes), now the host's is always the upper
  -- (conformal ground: none, the host's surface is not drawn twice there, see M.forkPieces)
  local sep = plan.conformal and plan.conformal[S.base] and 0 or M.TUNE.planeSep
  return {x, y, hz + (own - hz) * row.w - sep * (1 - row.w)}
end

-- the surface z of a row at u, from its vertices (piecewise linear R-M-L, extended past the edges)
function M.rowZ(row, u)
  if row.zfun and not row.snapped then return row.zfun(u) end
  local uR, uM, uL = -row.eR, row.uM, row.eL
  if u <= uM then
    if uM - uR < 1e-6 then return row.M[3] + (u - uM) * row.fr * -1 end
    return row.R[3] + (row.M[3] - row.R[3]) * (u - uR) / (uM - uR)
  end
  if uL - uM < 1e-6 then return row.M[3] + (u - uM) * row.fl end
  return row.M[3] + (row.L[3] - row.M[3]) * (u - uM) / (uL - uM)
end

function M.rows(plan, S, c)
  local out = {}
  local tick = pacer(plan)
  for _, iv in ipairs(intervals(S, c)) do
    local list = {}
    for _, s in ipairs(breaks(S, c, iv[1], iv[2])) do list[#list + 1] = M.row(plan, S, c, s); tick() end
    if #list >= 2 then out[#out + 1] = list end
  end
  return out
end

-- ---------------------------------------------------------------------------------------------------
-- Footprint index: is (x, y) on a drawn surface (and at what height)?
-- ---------------------------------------------------------------------------------------------------

local function addQuad(plan, poly, S, s, c)
  local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
  for _, p in ipairs(poly) do x0, y0, x1, y1 = min(x0, p[1]), min(y0, p[2]), max(x1, p[1]), max(y1, p[2]) end
  local entry = {poly = poly, S = S, s = s, box = {x0, y0, x1, y1}, c = c}
  for bx = floor(x0 / BIN), floor(x1 / BIN) do
    for by = floor(y0 / BIN), floor(y1 / BIN) do
      local k = binKey(bx, by)
      local bin = plan.bins[k]
      if not bin then bin = {}; plan.bins[k] = bin end
      bin[#bin + 1] = entry
    end
  end
end

function M.index(plan, S, c)
  for _, list in ipairs(c.rows) do
    for k = 1, #list - 1 do
      local a, b = list[k], list[k + 1]
      if a.eR + a.uM > 1e-4 or b.eR + b.uM > 1e-4 then addQuad(plan, {a.R, b.R, b.M, a.M}, S, a.s, c) end
      if a.eL - a.uM > 1e-4 or b.eL - b.uM > 1e-4 then addQuad(plan, {a.M, b.M, b.L, a.L}, S, a.s, c) end
    end
  end
end

-- the z of a triangle (a, b, c) at (x, y) if inside (with `grow` m of slack across its edges), else nil
local function triZ(a, b, c, x, y, grow)
  local d = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
  if abs(d) < 1e-12 then return nil end
  local l1 = ((b[2] - c[2]) * (x - c[1]) + (c[1] - b[1]) * (y - c[2])) / d
  local l2 = ((c[2] - a[2]) * (x - c[1]) + (a[1] - c[1]) * (y - c[2])) / d
  local l3 = 1 - l1 - l2
  local tol = 0
  if grow and grow > 0 then
    -- slack in barycentric units: grow over the triangle's smallest height
    local area = abs(d) / 2
    local e = max(sqrt((a[1] - b[1]) ^ 2 + (a[2] - b[2]) ^ 2), sqrt((b[1] - c[1]) ^ 2 + (b[2] - c[2]) ^ 2), sqrt((c[1] - a[1]) ^ 2 + (c[2] - a[2]) ^ 2))
    tol = grow * e / max(EPS, 2 * area)
  end
  if l1 >= -tol and l2 >= -tol and l3 >= -tol then return l1 * a[3] + l2 * b[3] + l3 * c[3] end
  return nil
end

local ensureAt   -- forward declaration: assigned below (surfacesAt used it as an undefined global)
-- every drawn surface z at (x, y) (+ grow m of slack), with the strokes; plates included
function M.surfacesAt(plan, x, y, grow)
  ensureAt(plan, x, y, grow)
  local out = {}
  for _, e in ipairs(plan.bins[binKey(floor(x / BIN), floor(y / BIN))] or NONE) do
    local b = e.box
    local g = grow or 0
    if x >= b[1] - g and x <= b[3] + g and y >= b[2] - g and y <= b[4] + g then
      local p = e.poly
      local z = triZ(p[1], p[2], p[3], x, y, grow)
      if not z and #p == 4 then z = triZ(p[1], p[3], p[4], x, y, grow) end
      if z then out[#out + 1] = {z = z, S = e.S, s = e.s} end
    end
  end
  return out
end

-- the rows and index of the chunks without them whose box holds (x, y) +- g (a query reached them)
ensureAt = function(plan, x, y, g)
  local lazy = plan.lazy
  if not lazy or not next(lazy) then return end
  g = g or 0
  for gx = floor((x - g) / LAZY), floor((x + g) / LAZY) do
    for gy = floor((y - g) / LAZY), floor((y + g) / LAZY) do
      local k = binKey(gx, gy)
      local list = lazy[k]
      if list then
        local built, live = false, 0
        for q = 1, #list do
          local c = list[q]
          if c.lazyS then
            local b = c.box
            if x >= b[1] - g and x <= b[3] + g and y >= b[2] - g and y <= b[4] + g then
              local S = c.lazyS
              c.lazyS = nil
              c.rows = M.rows(plan, S, c)
              M.index(plan, S, c)
              plan.stats.lazy = (plan.stats.lazy or 0) + 1
              built = true
            else
              live = live + 1
            end
          end
        end
        if live == 0 then
          lazy[k] = nil
        elseif built then
          local keep = {}
          for q = 1, #list do if list[q].lazyS then keep[#keep + 1] = list[q] end end
          lazy[k] = keep
        end
      end
    end
  end
end
M.ensureAt = ensureAt

-- the first drawn surface at (x, y) (+ grow m of slack) for which test(z, S, s) holds, no table per call
-- (the generator's sweptCrossing asks every 2 m of every way): z, S, s or nil
function M.surfaceWhere(plan, x, y, grow, test)
  ensureAt(plan, x, y, grow)
  local bin = plan.bins[binKey(floor(x / BIN), floor(y / BIN))]
  if not bin then return nil end
  local g = grow or 0
  for k = 1, #bin do
    local e = bin[k]
    local b = e.box
    if x >= b[1] - g and x <= b[3] + g and y >= b[2] - g and y <= b[4] + g then
      local p = e.poly
      local z = triZ(p[1], p[2], p[3], x, y, grow)
      if not z and #p == 4 then z = triZ(p[1], p[3], p[4], x, y, grow) end
      if z and (not test or test(z, e.S, e.s)) then return z, e.S, e.s end
    end
  end
  return nil
end

-- is (x, y, z) on a drawn surface within grow m (and within dz m of its height)? skip(hit) sees one
-- scratch table {z, S, s} (it must not keep it)
local HIT = {}
function M.onSurface(plan, x, y, z, grow, dz, skip)
  ensureAt(plan, x, y, grow)
  local bin = plan.bins[binKey(floor(x / BIN), floor(y / BIN))]
  if not bin then return false end
  local g, lim = grow or 0, dz or 2
  for k = 1, #bin do
    local e = bin[k]
    local b = e.box
    if x >= b[1] - g and x <= b[3] + g and y >= b[2] - g and y <= b[4] + g then
      local p = e.poly
      local hz = triZ(p[1], p[2], p[3], x, y, grow)
      if not hz and #p == 4 then hz = triZ(p[1], p[3], p[4], x, y, grow) end
      if hz and (not z or abs(hz - z) < lim) then
        if not skip then return true end
        HIT.z, HIT.S, HIT.s = hz, e.S, e.s
        if not skip(HIT) then return true end
      end
    end
  end
  return false
end

-- The nearest point of a drawn road surface to (x, y) within `reach` m (bundle7, P1: a separate footway beside a raised
-- road takes this edge's height): ex, ey, ez (the drawn height, lift included), distance; nothing when none is that near.
-- Bridge and tunnel stretches are not edges (a path under a deck does not climb onto it); distance 0 = on a surface.
local function nearOnSeg(px, py, ax, ay, bx, by)
  local dx, dy = bx - ax, by - ay
  local l2 = dx * dx + dy * dy
  local t = l2 > EPS and max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) or 0
  local qx, qy = ax + dx * t, ay + dy * t
  return (px - qx) ^ 2 + (py - qy) ^ 2, qx, qy, t
end
function M.edgeNear(plan, x, y, reach)
  ensureAt(plan, x, y, reach)
  local bestD, bx, by, bz = reach * reach, nil, nil, nil
  for gx = floor((x - reach) / BIN), floor((x + reach) / BIN) do
    for gy = floor((y - reach) / BIN), floor((y + reach) / BIN) do
      local bin = plan.bins[binKey(gx, gy)]
      for k = 1, bin and #bin or 0 do
        local e = bin[k]
        local b = e.box
        if x >= b[1] - reach and x <= b[3] + reach and y >= b[2] - reach and y <= b[4] + reach
            and not (e.S and (M.structureAt(e.S, e.s, 'bridge') or M.structureAt(e.S, e.s, 'tunnel'))) then
          local p = e.poly
          local inside = triZ(p[1], p[2], p[3], x, y, 0) or (#p == 4 and triZ(p[1], p[3], p[4], x, y, 0))
          if inside then return x, y, inside + plan.lift, 0 end
          for i = 1, #p do
            local a, c = p[i], p[i % #p + 1]
            local d2, qx, qy, t = nearOnSeg(x, y, a[1], a[2], c[1], c[2])
            if d2 < bestD then bestD, bx, by, bz = d2, qx, qy, a[3] + (c[3] - a[3]) * t end
          end
        end
      end
    end
  end
  if not bx then return nil end
  return bx, by, bz + plan.lift, sqrt(bestD)
end

-- the plates' fans into the index (after M.plan: their rings need no stroke rows)
function M.plateRings(plan)
  for _, pl in ipairs(plan.plates) do
    if pl.junction then
      for _, t in ipairs(pl.tris) do addQuad(plan, t, nil, nil) end
      goto next_plate
    end
    local nd = pl.node
    local mouths = {}
    for _, r in ipairs(pl.rows) do
      local row = r.row
      local pts = row.dir > 0 and {row.R, row.C, row.L} or {row.L, row.C, row.R}
      local a = atan2(row.C[2] - nd.y, row.C[1] - nd.x)
      mouths[#mouths + 1] = {a = a, pts = pts, S = r.S}
    end
    table.sort(mouths, function(p, q) return p.a < q.a end)
    -- each mouth's edge direction outwards along its road (from the stroke's heading at the cut)
    for _, m in ipairs(mouths) do
      local S, row = m.S, nil
      for _, r in ipairs(pl.rows) do if r.S == S and r.row.C == m.pts[2] then row = r end end
      local st = row and row.S and row.s and row.S:at(row.s + row.row.dir * row.row.cut)
      if st then
        local d = row.row.dir
        m.dx, m.dy = cos(st.h) * d, sin(st.h) * d
      end
    end
    local ring = {}
    local centre = {nd.x, nd.y, nd.z}
    for k, m in ipairs(mouths) do
      for _, p in ipairs(m.pts) do ring[#ring + 1] = p end
      local nm = mouths[k % #mouths + 1]
      if #mouths > 1 and m.dx and nm.dx then
        -- a = this mouth's last corner, b = the next mouth's first: a - t (dx, dy) = b - u (ndx, ndy)
        local a, b = m.pts[3], nm.pts[1]
        local det = -m.dx * nm.dy + m.dy * nm.dx
        if abs(det) > .2 then
          local rx, ry = b[1] - a[1], b[2] - a[2]
          local t = (rx * nm.dy - ry * nm.dx) / det
          local u = (-m.dx * ry + m.dy * rx) / det
          local x, y = a[1] - m.dx * t, a[2] - m.dy * t
          local cross = (b[1] - a[1]) * (y - a[2]) - (b[2] - a[2]) * (x - a[1])
          local nodeSide = (b[1] - a[1]) * (centre[2] - a[2]) - (b[2] - a[2]) * (centre[1] - a[1])
          local reach = sqrt((a[1] - centre[1]) ^ 2 + (a[2] - centre[2]) ^ 2)
          if t > 0 and u > 0 and t < reach and u < reach and cross * nodeSide < 0 then
            ring[#ring + 1] = {x, y, (a[3] + b[3]) / 2}
          end
        end
      end
    end
    local tris = {}
    for k = 1, #ring do
      local a, b = ring[k], ring[k % #ring + 1]
      local cross = (a[1] - centre[1]) * (b[2] - centre[2]) - (a[2] - centre[2]) * (b[1] - centre[1])
      if cross > 1e-6 then tris[#tris + 1] = {centre, a, b} end
    end
    pl.ring, pl.centre, pl.tris = ring, centre, tris
    for _, t in ipairs(tris) do addQuad(plan, t, nil, nil) end
    ::next_plate::
  end
end

-- ---------------------------------------------------------------------------------------------------
-- Markings: the lines of a stroke at s ({id, u, width, dash = {on, off} or nil}), and the gore
-- ---------------------------------------------------------------------------------------------------
-- the lane block at s fitted into the paved section: {n, f, b, both, w, left, right} (u of its edges).
-- The exporter's extents can be narrower than the tagged lanes (the median clamp of two carriageways
-- whose OSM lines are too close): the block then shifts, and if it must, narrows, to stay
-- M.MARKS.inside m inside the paved edges. The added lanes of exits and entries are not part of it.
function M.laneBlock(S, s)
  local run = S:lanesAt(s)
  local n = max(1, run.f + run.b + run.both)
  local half = n * run.width / 2
  local st = S:at(s)
  if not st then return {n = n, f = run.f, b = run.b, both = run.both, w = run.width, left = half, right = -half} end
  local blk = M.rawBlock(S, s, run, st)
  -- a lane gain or drop: the block moves from the old run's to the new run's over the window; the lane
  -- that comes or goes is the one on the side whose edge moves outward (inward), its width grows from 0
  for _, w in ipairs(S.windows) do
    if s > w.a and s < w.b then
      local A, B = M.rawBlock(S, s, w.r0, st), M.rawBlock(S, s, w.r1, st)
      local t = smooth((s - w.a) / (w.b - w.a))
      local left, right = A.left + (B.left - A.left) * t, A.right + (B.right - A.right) * t
      local nA, nB = A.n, B.n
      local widths = {}
      local nn = max(nA, nB)
      if nA ~= nB then
        local gainLeft = (B.left - A.left) - (A.right - B.right)
        local small = nA < nB and A or B
        local grow = nA < nB and t or 1 - t
        -- the lanes that stay keep their width; the coming or going one takes the rest
        local wStay = small.w + ((nA < nB and B.w or A.w) - small.w) * grow
        local rest = max(0, (left - right) - wStay * (nn - 1))
        local sideLeft = (nA < nB) == (gainLeft >= 0)
        for k = 1, nn do widths[k] = wStay end
        if sideLeft then widths[nn] = rest else widths[1] = rest end
      else
        for k = 1, nn do widths[k] = (left - right) / nn end
      end
      blk = {n = nn, f = (nA >= nB and A or B).f, b = (nA >= nB and A or B).b, both = (nA >= nB and A or B).both,
        w = (left - right) / nn, left = left, right = right, widths = widths}
    end
  end
  for _, e in ipairs(jointsAt(S, s)) do
    local jw, j = e[1], e[2]
    local left, right = blk.left + (j.left - blk.left) * jw, blk.right + (j.right - blk.right) * jw
    blk = {n = blk.n, f = blk.f, b = blk.b, both = blk.both, w = (left - right) / blk.n, left = left, right = right, widths = blk.widths}
    if blk.widths then
      local sum = 0
      for _, v in ipairs(blk.widths) do sum = sum + v end
      for k, v in ipairs(blk.widths) do blk.widths[k] = v * (left - right) / max(EPS, sum) end
    end
  end
  return blk
end

-- the block of one lane run at s, fitted into the paved section less the exits' and entries' lanes
function M.rawBlock(S, s, run, st)
  local availL, availR = st.eL, st.eR
  for _, e in ipairs(jointsAt(S, s)) do
    local jw, j = e[1], e[2]
    availL, availR = availL + (j.eL - availL) * jw, availR + (j.eR - availR) * jw
  end
  for _, e in ipairs(S.events) do
    local lo, hi = min(e.taper, e.nose), max(e.taper, e.nose)
    if s >= lo - 1e-6 and s <= hi + 1e-6 then
      local extra
      if s >= min(e.taper, e.separate) - 1e-6 and s <= max(e.taper, e.separate) + 1e-6 then
        local grow = e.kind == 'exit' and (s - e.taper) / max(EPS, e.parallel - e.taper) or (e.taper - s) / max(EPS, e.taper - e.parallel)
        extra = e.width * smooth(grow)
      else
        local at = S:at(e.nose)
        extra = at and max(0, (e.side > 0 and st.eL - at.eL or st.eR - at.eR)) or 0
      end
      if e.side > 0 then availL = availL - extra else availR = availR - extra end
    end
  end
  return fitBlock(run, availL, availR)
end

function M.linesAt(S, s)
  local out = {}
  local autobahn = S.base == 'motorway'
  local narrow = autobahn and M.MARKS.narrow.autobahn or M.MARKS.narrow.other
  local broad = autobahn and M.MARKS.broad.autobahn or M.MARKS.broad.other
  local guide = S.urban and M.MARKS.guide.urban or M.MARKS.guide.rural
  local block = autobahn and M.MARKS.block.autobahn or M.MARKS.block.other
  local paint = tostring(S.tags.lane_markings or '')
  if paint == 'no' then return out end
  local painted = CENTRED[S.base] or S.tags.lanes ~= nil or paint == 'yes'
  local run = M.laneBlock(S, s)
  local n = run.n
  local w = run.w
  local half = (run.left - run.right) / 2
  local edged = EDGED[S.base] or (S.base == 'tertiary' and 2 * half >= 6) or paint == 'yes'
  if not painted then return out end
  local zone = M.zoneAt(S, s)
  if S.oneway then
    local right, left = run.right, run.left
    -- exits and entries on the host: the added lane, its outer edge line and the block line
    local events = {}
    for _, e in ipairs(S.events) do
      local a, b = min(e.taper, e.nose), max(e.taper, e.nose)
      if s >= a - 1e-6 and s <= b + 1e-6 then events[#events + 1] = e end
    end
    local blockSide = {}
    for _, e in ipairs(events) do
      local lo, hi = min(e.taper, e.separate), max(e.taper, e.separate)
      if s >= lo - 1e-6 and s <= hi + 1e-6 then
        local grow = e.kind == 'exit' and (s - e.taper) / max(EPS, e.parallel - e.taper) or (e.taper - s) / max(EPS, e.taper - e.parallel)
        local lane = e.width * smooth(grow)
        local edge = (e.side > 0 and left or right) + e.side * lane
        out[#out + 1] = {id = 'added' .. e.side, u = edge, width = broad}
        blockSide[e.side] = true
      end
    end
    -- a branch drawn along this edge: one solid line, narrow where a lane leaves or joins there (the fork gap's lane: the
    -- divider to the nose, then the gore's outline), else the through edge line at its own width (an aerial review:
    -- solid along a branch that leaves the through lanes; game test of a split: the dashed divider was missing beside the car)
    local gap = {}
    for _, g in ipairs(S.edgeGaps) do if s > g.a + 1e-6 and s < g.b - 1e-6 then gap[g.side] = g end end
    if not gap[-1] then
      out[#out + 1] = {id = 'edgeR', u = right, width = autobahn and broad or narrow, dash = blockSide[-1] and block or nil}
    elseif M.MARKS.forkLine then
      local lane = gap[-1].lane
      out[#out + 1] = {id = 'forkR', u = right, width = (lane or not autobahn) and narrow or broad}
    end
    if not gap[1] then out[#out + 1] = {id = 'edgeL', u = left, width = narrow, dash = blockSide[1] and block or nil}
    elseif M.MARKS.forkLine then out[#out + 1] = {id = 'forkL', u = left, width = narrow} end
    local at = right
    for k = 1, n - 1 do
      local wk = run.widths and run.widths[k] or w
      at = at + wk
      -- a lane coming or going: its line only once it is half a metre wide
      local wn = run.widths and run.widths[k + 1] or w
      local full = M.MARKS.laneFrom * w
      if run.widths then
        full = 0
        for q = 1, n do full = max(full, run.widths[q]) end
        full = M.MARKS.laneFrom * full
      end
      if wk >= max(.5, full) and wn >= max(.5, full) then out[#out + 1] = {id = 'lane' .. k, u = at, width = narrow, dash = guide} end
    end
    if not edged and not S.link then
      -- no edge lines on this class: only the lane lines
      local keep = {}
      for _, l in ipairs(out) do if l.id:find('^lane') then keep[#keep + 1] = l end end
      out = keep
    end
  else
    if run.both == 0 and run.f + run.b >= 2 then
      local centre = run.left - run.b * w
      local solid = run.f >= 2 and run.b >= 2
      out[#out + 1] = {id = 'centre', u = centre, width = narrow, dash = not solid and guide or nil}
      for k = 1, run.f - 1 do out[#out + 1] = {id = 'f' .. k, u = centre - k * w, width = narrow, dash = guide} end
      for k = 1, run.b - 1 do out[#out + 1] = {id = 'b' .. k, u = centre + k * w, width = narrow, dash = guide} end
    end
    if edged then
      out[#out + 1] = {id = 'edgeR', u = run.right, width = narrow}
      out[#out + 1] = {id = 'edgeL', u = run.left, width = narrow}
    end
  end
  if zone then out = M.zoneLines(S, zone, out) end
  return out
end

-- ---------------------------------------------------------------------------------------------------
-- Meshes
-- ---------------------------------------------------------------------------------------------------
local function newMesh(material)
  return {verts = {}, normals = {}, uvs = {}, faces = {}, material = material, vmap = {}, vkz = {}, nmap = {}, umap = {}, tris = 0}
end

-- vertices welded at 1 mm: one index per (x, y) key, its z key beside it; another z at the same (x, y)
-- (walls, kerbs) in a second map (no table per vertex: bundle5)
local function vIndex(m, o, p)
  local x, y, z = p[1] - o[1], p[2] - o[2], p[3] - o[3]
  local kx, ky, kz = floor(x * 1000 + .5), floor(y * 1000 + .5), floor(z * 1000 + .5)
  local k = (kx + 4194304) * 8388608 + (ky + 4194304)
  local i = m.vmap[k]
  if i and m.vkz[i] == kz then return i end
  if i then
    local more = m.vmore
    if not more then more = {}; m.vmore = more end
    local k2 = k .. ':' .. kz
    local j = more[k2]
    if j then return j end
    j = #m.verts
    m.verts[j + 1] = {x = x, y = y, z = z}
    m.vkz[j] = kz
    more[k2] = j
    return j
  end
  i = #m.verts
  m.verts[i + 1] = {x = x, y = y, z = z}
  m.vkz[i] = kz
  m.vmap[k] = i
  return i
end

local function nIndex(m, nx, ny, nz)
  local k = floor((nx + 1) * 10000 + .5) * 1073741824 + floor((ny + 1) * 10000 + .5) * 32768 + floor((nz + 1) * 10000 + .5)
  local i = m.nmap[k]
  if not i then i = #m.normals; m.normals[i + 1] = {x = nx, y = ny, z = nz}; m.nmap[k] = i end
  return i
end

local function uIndex(m, u, v)
  -- a numeric key while |u|, |v| < 32768 (1/1024 steps), else the text key (a rail's s / 3 on a long stroke)
  local fu, fv = floor(u * 1024 + .5), floor(v * 1024 + .5)
  local k
  if fu > -33554432 and fu < 33554432 and fv > -33554432 and fv < 33554432 then
    k = (fu + 33554432) * 67108864 + (fv + 33554432)
  else
    k = fu .. ':' .. fv
  end
  local i = m.umap[k]
  if not i then i = #m.uvs; m.uvs[i + 1] = {u = u, v = v}; m.umap[k] = i end
  return i
end

-- a triangle seen from the side its (a, b, c) normal points to; uv(p) -> u, v
local CORNERS = {}   -- tri()'s three corners in drawing order, one table for every call (uv() never calls tri)
local function tri(m, o, a, b, c, uv, facing)
  local ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
  local nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
  local l = sqrt(nx * nx + ny * ny + nz * nz)
  if l < 1e-9 then return end
  nx, ny, nz = nx / l, ny / l, nz / l
  if facing and nx * facing[1] + ny * facing[2] + nz * facing[3] < 0 then
    b, c, nx, ny, nz = c, b, -nx, -ny, -nz
  end
  local n = nIndex(m, nx, ny, nz)
  -- BeamNG ProceduralMesh: clockwise seen from the visible side (generator collector: a, c, b)
  -- (one loop over the three corners: unrolled, LuaJIT ran this 4-5 times slower offline; a new
  -- table of them per triangle was garbage, generator_budget.py --garbage --swept)
  CORNERS[1], CORNERS[2], CORNERS[3] = a, c, b
  for _, p in ipairs(CORNERS) do
    local u, v = uv(p)
    m.faces[#m.faces + 1] = {v = vIndex(m, o, p), n = n, u = uIndex(m, u, v)}
  end
  m.tris = m.tris + 1
end

local UP = {0, 0, 1}
local function lifted(p, dz) return {p[1], p[2], p[3] + dz, p[4], p[5]} end

local function finish(m)
  m.vmap, m.vkz, m.vmore, m.nmap, m.umap = nil, nil, nil, nil, nil
  local n = m.tris
  m.tris = nil
  return m, n
end

-- ---------------------------------------------------------------------------------------------------
-- Build: the items of the chunks and plates this tile owns
-- ---------------------------------------------------------------------------------------------------
local function planUV(scale) return function(p) return p[1] / scale, p[2] / scale end end
local function attrUV(p) return p[4] or 0, p[5] or 0 end

-- ---------------------------------------------------------------------------------------------------
-- Phase 3: approaches to the junction surfaces (stop and give-way lines, arrows, crossings)
-- ---------------------------------------------------------------------------------------------------
-- the approach zone at s ('clear' wins over 'solid'); its ends belong to the road outside it
function M.zoneAt(S, s)
  local best
  for _, z in ipairs(S.zones or NONE) do
    if s > z.a + 1e-6 and s < z.b - 1e-6 and (not best or z.kind == 'clear') then best = z end
  end
  return best
end

-- the lines of a row in an approach zone: from the stop line on only the edge lines (and the added
-- lanes'), before it the inbound lane lines solid (and the centre line where the junction is signalled)
function M.zoneLines(S, zone, lines)
  local out = {}
  for _, l in ipairs(lines) do
    local id = l.id
    if zone.kind == 'clear' then
      if id:find('^edge') or id:find('^added') then out[#out + 1] = l end
    else
      local inbound = id:find('^lane') or (zone.ap.sigma > 0 and id:find('^f%d')) or (zone.ap.sigma < 0 and id:find('^b%d'))
        or (id == 'centre' and zone.ap.control == 'signals')
      if inbound then
        out[#out + 1] = {id = id, u = l.u, width = l.width}
      else
        out[#out + 1] = l
      end
    end
  end
  return out
end

-- the surface z at (x, y): the drawn surface nearest zref (within 1.5 m), else zref
function M.surfaceZ(plan, x, y, zref)
  ensureAt(plan, x, y, .03)
  local bin = plan.bins[binKey(floor(x / BIN), floor(y / BIN))]
  if not bin then return zref end
  local best, bd
  for k = 1, #bin do
    local e = bin[k]
    local b = e.box
    if x >= b[1] - .03 and x <= b[3] + .03 and y >= b[2] - .03 and y <= b[4] + .03 then
      local p = e.poly
      local z = triZ(p[1], p[2], p[3], x, y, .03)
      if not z and #p == 4 then z = triZ(p[1], p[3], p[4], x, y, .03) end
      if z then
        local d = abs(z - zref)
        if d < 1.5 and (not bd or d < bd) then best, bd = z, d end
      end
    end
  end
  return best or zref
end

-- a point s along, u across a stroke, on the drawn surface, lifted for a mark; nil where S is not loaded
local function markPoint(plan, S, s, u, U, V)
  local st = S:at(s)
  if not st then return nil end
  local x, y = st.x - sin(st.h) * u, st.y + cos(st.h) * u
  local zref = st.z + (u >= 0 and st.fl or st.fr) * abs(u)
  return {x, y, M.surfaceZ(plan, x, y, zref) + plan.lift + M.MARKS.lift, U or .5, V or s / M.MARKS.repeatV}
end
M.markPoint = markPoint

-- a filled rectangle s0..s1 along, u0..u1 across S, subdivided so it lies on the surface's crowns
function M.fillSU(plan, S, mesh, o, s0, s1, u0, u1)
  local sub = M.MARKS.sub
  local nS, nU = max(1, math.ceil(abs(s1 - s0) / sub)), max(1, math.ceil(abs(u1 - u0) / sub))
  local grid = {}
  for i = 0, nS do
    grid[i] = {}
    for j = 0, nU do
      grid[i][j] = markPoint(plan, S, s0 + (s1 - s0) * i / nS, u0 + (u1 - u0) * j / nU)
      if not grid[i][j] then return end
    end
  end
  for i = 0, nS - 1 do
    for j = 0, nU - 1 do
      local a, b, c, d = grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]
      tri(mesh, o, a, b, c, attrUV, UP); tri(mesh, o, a, c, d, attrUV, UP)
    end
  end
end

-- arrow shapes (Z 297): triangles in (a along travel from the tail, b to the travel's left), length L
local function arrowTris(token, L)
  local A = M.MARKS.arrow
  local sw, hw, hl = A.shaft * L / 5, A.head * L / 5, A.headLength * L / 5
  local out = {}
  local function quad(p, q, w)
    local dx, dy = q[1] - p[1], q[2] - p[2]
    local l = max(EPS, sqrt(dx * dx + dy * dy))
    local nx, ny = -dy / l * w / 2, dx / l * w / 2
    local a, b, c, d = {p[1] - nx, p[2] - ny}, {q[1] - nx, q[2] - ny}, {q[1] + nx, q[2] + ny}, {p[1] + nx, p[2] + ny}
    out[#out + 1] = {a, b, c}; out[#out + 1] = {a, c, d}
  end
  local function head(p, dx, dy, len, w)
    local nx, ny = -dy * w / 2, dx * w / 2
    out[#out + 1] = {{p[1] - nx, p[2] - ny}, {p[1] + dx * len, p[2] + dy * len}, {p[1] + nx, p[2] + ny}}
  end
  local straight = token:find('through') ~= nil
  local left, right = token:find('left') ~= nil, token:find('right') ~= nil
  local r2 = math.sqrt(.5)
  local h2, d = .8 * hl, .9 * L / 5
  local a1 = L - (d + h2) * r2
  if straight then
    quad({0, 0}, {L - hl, 0}, sw)
    head({L - hl, 0}, 1, 0, hl, hw)
  end
  for _, side in ipairs({left and 1 or 0, right and -1 or 0}) do
    if side ~= 0 then
      local base = straight and a1 - 1.4 * L / 5 or a1
      if not straight then quad({0, 0}, {base, 0}, sw) end
      local e = {base + d * r2, side * d * r2}
      quad({base, 0}, e, sw)
      head(e, r2, side * r2, h2, hw * .9)
    end
  end
  return out
end
M.arrowTris = arrowTris

-- the inbound lanes at s: u of their two edges, lane k's centre (k from the travel's left), their count
local function inbound(S, s, sigma)
  local blk = M.laneBlock(S, s)
  local w = blk.w
  if S.oneway then
    if sigma > 0 then return blk.right, blk.left, function(k) return blk.left - (k - .5) * w end, blk.n end
    return blk.right, blk.left, function(k) return blk.right + (k - .5) * w end, blk.n
  end
  local centre = blk.left - blk.b * w
  if blk.both > 0 or blk.f + blk.b < 2 then centre = (blk.left + blk.right) / 2 end
  if sigma > 0 then
    return blk.right, centre, function(k) return centre - (k - .5) * w end, max(1, blk.f)
  end
  return centre, blk.left, function(k) return centre + (k - .5) * w end, max(1, blk.b)
end
M.inboundLanes = inbound

local function inChunk(c, s, last)
  return s >= c.s[1] - 1e-6 and (s < c.s[c.n] - 1e-6 or (last and s <= c.s[c.n] + 1e-6))
end

function M.approachMarks(plan, S, c, mesh, o)
  local MK = M.MARKS
  local last = c.index == c.chunks - 1
  for _, ap in ipairs(S.approaches) do
    local sg = ap.sigma
    -- the stop or give-way line, its downstream edge on the stop position
    if inChunk(c, ap.stop, last) and (ap.control == 'signals' or ap.control == 'stop' or ap.control == 'give_way') then
      local uA, uB = inbound(S, ap.stop, sg)
      if ap.control == 'give_way' then
        local w, on, off = MK.giveway.width, MK.giveway.dash[1], MK.giveway.dash[2]
        local u = uA + off / 2
        while u + on <= uB - off / 2 + 1e-6 do
          M.fillSU(plan, S, mesh, o, ap.stop - sg * w, ap.stop, u, u + on)
          u = u + on + off
        end
        if MK.teeth.on then
          local T = MK.teeth
          local b0 = ap.stop - sg * (w + T.behind)
          local u2 = uA + T.gap
          while u2 + T.base <= uB - T.gap + 1e-6 do
            local p, q, r = markPoint(plan, S, b0, u2), markPoint(plan, S, b0, u2 + T.base), markPoint(plan, S, b0 - sg * T.height, u2 + T.base / 2)
            if p and q and r then tri(mesh, o, p, q, r, attrUV, UP) end
            u2 = u2 + T.base + T.gap
          end
        end
      else
        M.fillSU(plan, S, mesh, o, ap.stop - sg * MK.stop.width, ap.stop, uA, uB)
      end
    end
    -- crossings on the approach: across the whole paved width
    for _, cr in ipairs(ap.crossings) do
      if inChunk(c, cr.s, last) then
        local st = S:at(cr.s)
        if st then
          local uR, uL = -st.eR + MK.zebra.inside, st.eL - MK.zebra.inside
          local s0, s1 = cr.s - cr.width / 2, cr.s + cr.width / 2
          local function dashedAcross(sv, w, dash)
            local u = uR
            while u < uL - 1e-6 do
              M.fillSU(plan, S, mesh, o, sv - w / 2, sv + w / 2, u, min(uL, u + dash[1]))
              u = u + dash[1] + dash[2]
            end
          end
          if cr.kind == 'zebra' then
            local u = uR
            while u + MK.zebra.stripe <= uL + 1e-6 do
              M.fillSU(plan, S, mesh, o, s0, s1, u, u + MK.zebra.stripe)
              u = u + MK.zebra.stripe + MK.zebra.gap
            end
          elseif cr.kind == 'cycle' then
            dashedAcross(s0, MK.cycle.width, MK.cycle.dash); dashedAcross(s1, MK.cycle.width, MK.cycle.dash)
          elseif cr.kind == 'both' then
            local split = sg > 0 and s0 + (s1 - s0) * 4 / 6 or s1 - (s1 - s0) * 4 / 6
            dashedAcross(s0, MK.furt.width, MK.furt.dash); dashedAcross(split, MK.cycle.width, MK.cycle.dash)
            dashedAcross(s1, MK.cycle.width, MK.cycle.dash)
          else
            dashedAcross(s0, MK.furt.width, MK.furt.dash); dashedAcross(s1, MK.furt.width, MK.furt.dash)
          end
          if cr.red then
            plan.redMarks = plan.redMarks or {}
            plan.redMarks[#plan.redMarks + 1] = {S = S, s0 = s0, s1 = s1, uR = uR, uL = uL}
          end
        end
      end
    end
    -- arrows from turn:lanes, their tips MARKS.arrow.before m before the stop line (again repeatAt m before)
    if ap.turns ~= '' and (ap.control ~= 'none' or ap.lanes >= 2) then
      local tokens = {}
      for t in (ap.turns .. '|'):gmatch('([^|]*)|') do tokens[#tokens + 1] = t end
      local L = S.urban and MK.arrow.length.urban or MK.arrow.length.rural
      for again = 0, 1 do
        local tip = ap.stop - sg * (MK.arrow.before + again * MK.arrow.repeatAt)
        local run0, run1 = S:lanesAt(ap.stop - sg * 1), S:lanesAt(tip - sg * L)
        if inChunk(c, tip, last) and run0 == run1 and not (again == 1 and #tokens < 2) then
          local _, _, centreOf, n = inbound(S, tip, sg)
          for k = 1, min(n, #tokens) do
            local tok = tokens[k]
            if tok ~= '' and tok ~= 'none' and not tok:find('merge') and not tok:find('reverse') then
              local uc = centreOf(k)
              for _, t in ipairs(arrowTris(tok, L)) do
                local pts = {}
                for q = 1, 3 do
                  local a, b = t[q][1], t[q][2]
                  pts[q] = markPoint(plan, S, tip - sg * (L - a), uc + (sg > 0 and b or -b))
                end
                if pts[1] and pts[2] and pts[3] then tri(mesh, o, pts[1], pts[2], pts[3], attrUV, UP) end
              end
            end
          end
        end
      end
    end
  end
end

-- the tags of the OSM part under s (the stroke's first part's elsewhere)
function M.partTags(S, s)
  for _, p in ipairs(S.parts) do
    if s >= p.s0 - 1e-6 and s <= p.s1 + 1e-6 then return p.tags end
  end
  return S.tags
end

function M.reversedAt(S, s)
  for _, p in ipairs(S.parts) do
    if s >= p.s0 - 1e-6 and s <= p.s1 + 1e-6 then return p.reversed and true or false end
  end
  return false
end

-- Conformal ground, forks and merges: a slip road's rows in its host's plane lay over the host's own rows (the host's
-- surface drawn twice, apart by TUNE.planeSep to keep them from z-fighting). With plan.conformal the branch's quad
-- keeps only the pieces outside the host's drawn quads, and a piece corner on the host's edge takes the host's height
-- there: the gore between the two is one surface bounded by shared edges, no road lies on a road.
-- (The same for a stroke that ends on another or runs over it, TUNE.endJoin and overlapEase: its rows in the other's
-- plane are cut by the other's quads.)
-- H: the stroke whose plane the quad's rows are in (hostOf); poly: the quad's four corners ({x, y, z}, lifted, in the
-- order the mesh fans them from the first); -> {pieces: lists of corners {x, y, z}} (the quad itself when no host
-- row covers it)
local Conf
do
  local ok, mod = pcall(require, 'ge/extensions/e2r/conformal')
  Conf = ok and type(mod) == 'table' and mod or nil
end
-- the stroke the rows a, b of S lie in the plane of (they blend to it: w < 1), when S cuts the ground
local function hostOf(plan, S, a, b)
  if not (plan.conformal and plan.conformal[S.base]) then return nil end
  if a.w < 1 then return a.endHost or S.host end
  if b.w < 1 then return b.endHost or S.host end
end
function M.forkPieces(plan, S, H, poly)
  if not (Conf and H and plan.conformal and plan.conformal[S.base]) then return {poly} end
  local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
  for _, p in ipairs(poly) do x0, y0, x1, y1 = min(x0, p[1]), min(y0, p[2]), max(x1, p[1]), max(y1, p[2]) end
  ensureAt(plan, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0 + y1 - y0) / 2 + 1)
  local hosts, seen = {}, {}
  for bx = floor(x0 / BIN), floor(x1 / BIN) do
    for by = floor(y0 / BIN), floor(y1 / BIN) do
      for _, e in ipairs(plan.bins[binKey(bx, by)] or NONE) do
        local b = e.box
        -- (only a host that is drawn here: its chunk is this tile's own or inside its ground margin)
        if e.S == H and e.c and (e.c.owned or e.c.ground) and not seen[e] and b[1] <= x1 and b[3] >= x0 and b[2] <= y1 and b[4] >= y0 then
          seen[e] = true
          hosts[#hosts + 1] = e
        end
      end
    end
  end
  if #hosts == 0 then return {poly} end
  local ring = Conf.ring(poly)
  if not ring then return NONE end
  local pieces = {ring}
  for _, e in ipairs(hosts) do
    local out = {}
    for _, cutter in ipairs(Conf.parts(e.poly)) do
      for _, piece in ipairs(pieces) do
        local after = {}
        Conf.subtract(piece, cutter, after)
        for _, r in ipairs(after) do out[#out + 1] = r end
      end
      pieces, out = out, {}
      if #pieces == 0 then return NONE end
    end
  end
  -- the corners: the quad's own keep their height, a corner cut on a host quad takes that quad's, else the branch's
  local function branchZ(x, y)
    local z = triZ(poly[1], poly[2], poly[3], x, y, 1e-6)
    if not z and poly[4] then z = triZ(poly[1], poly[3], poly[4], x, y, 1e-6) end
    return z
  end
  local lift = plan.lift
  local function hostZ(x, y)
    for _, e in ipairs(hosts) do
      local p = e.poly
      local z = triZ(p[1], p[2], p[3], x, y, 1e-6)
      if not z and #p == 4 then z = triZ(p[1], p[3], p[4], x, y, 1e-6) end
      if z then return z + lift end
    end
  end
  for _, piece in ipairs(pieces) do
    for k, p in ipairs(piece) do
      if not p[3] then
        local z = hostZ(p[1], p[2]) or branchZ(p[1], p[2])
        piece[k] = {p[1], p[2], z or poly[1][3]}
      end
    end
  end
  return pieces
end

local function drawPieces(mesh, o, pieces, uv)
  for _, piece in ipairs(pieces) do
    for k = 2, #piece - 1 do tri(mesh, o, piece[1], piece[k], piece[k + 1], uv, UP) end
  end
end

-- a convex polygon ({x, y, z, ...}) clipped to rect, z interpolated: the same intersections whichever tile clips it
local function clipRect(poly, rect)
  for side = 1, 4 do
    local axis, bound, greater = (side == 1 or side == 3) and 1 or 2, rect[side], side < 3
    local out, n = {}, #poly
    for k = 1, n do
      local p, q = poly[k], poly[k % n + 1]
      local ip = greater and p[axis] >= bound or not greater and p[axis] <= bound
      local iq = greater and q[axis] >= bound or not greater and q[axis] <= bound
      if ip then out[#out + 1] = p end
      if ip ~= iq then
        local t = (bound - p[axis]) / (q[axis] - p[axis])
        local r = {p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, p[3] + (q[3] - p[3]) * t}
        r[axis] = bound
        out[#out + 1] = r
      end
    end
    poly = out
    if #poly < 3 then return NONE end
  end
  return poly
end

-- clip: the tile's rect when the stroke cuts the ground (conformal): every tile draws the carriageway inside its own rect,
-- so its ground never depends on the neighbour that holds the chunk's middle
function M.surfaceMesh(plan, S, c, meshOf, o, lift, uv, clip)
  local tick = pacer(plan)
  for _, list in ipairs(c.rows) do
    for k = 1, #list - 1 do
      local a, b = list[k], list[k + 1]
      tick()
      local mesh = meshOf(M.partTags(S, (a.s + b.s) / 2))
      local Ra, Ma, La, Rb, Mb, Lb = lifted(a.R, lift), lifted(a.M, lift), lifted(a.L, lift), lifted(b.R, lift), lifted(b.M, lift), lifted(b.L, lift)
      local H = hostOf(plan, S, a, b)
      if clip then
        local x0, x1 = min(a.x, b.x) - max(a.eL, a.eR, b.eL, b.eR), max(a.x, b.x) + max(a.eL, a.eR, b.eL, b.eR)
        local y0, y1 = min(a.y, b.y) - max(a.eL, a.eR, b.eL, b.eR), max(a.y, b.y) + max(a.eL, a.eR, b.eL, b.eR)
        if x1 >= clip[1] and x0 <= clip[3] and y1 >= clip[2] and y0 <= clip[4] then
          for _, quad in ipairs({{Ra, Rb, Mb, Ma}, {Ma, Mb, Lb, La}}) do
            for _, piece in ipairs(H and M.forkPieces(plan, S, H, quad) or {quad}) do
              local inside = true
              for _, p in ipairs(piece) do
                if p[1] < clip[1] or p[1] > clip[3] or p[2] < clip[2] or p[2] > clip[4] then inside = false; break end
              end
              drawPieces(mesh, o, {inside and piece or clipRect(piece, clip)}, uv)
            end
          end
        end
      elseif H then
        -- rows in another stroke's plane: the quads without what that stroke's surface already is
        if a.eR + a.uM > 1e-4 or b.eR + b.uM > 1e-4 then drawPieces(mesh, o, M.forkPieces(plan, S, H, {Ra, Rb, Mb, Ma}), uv) end
        if a.eL - a.uM > 1e-4 or b.eL - b.uM > 1e-4 then drawPieces(mesh, o, M.forkPieces(plan, S, H, {Ma, Mb, Lb, La}), uv) end
      else
        tri(mesh, o, Ra, Rb, Mb, uv, UP); tri(mesh, o, Ra, Mb, Ma, uv, UP)
        tri(mesh, o, Ma, Mb, Lb, uv, UP); tri(mesh, o, Ma, Lb, La, uv, UP)
      end
    end
  end
end

-- a mark quad along rows a -> b between fractions t0, t1, at lateral u0 (row a) .. u1 (row b)
local function markPiece(plan, S, mesh, o, a, b, u0, u1, width, t0, t1, id)
  if M.debug then M.debug(S, a, b, u0, u1, t0, t1, id, width) end
  local hw = width * M.MARKS.spread / 2
  local function at(row, u) local x, y = row.x + row.nx * u, row.y + row.ny * u; return x, y, M.rowZ(row, u) end
  local lift = plan.lift + M.MARKS.lift
  -- a vertex's z on the drawn triangles of rows a, b (as M.surfaceMesh splits them), else the rows' z
  local tris = {{a.R, b.R, b.M}, {a.R, b.M, a.M}, {a.M, b.M, b.L}, {a.M, b.L, a.L}}
  local function onMesh(x, y, z)
    for _, t in ipairs(tris) do
      local zz = triZ(t[1], t[2], t[3], x, y, .02)
      if zz then return zz end
    end
    return z
  end
  local function lerp(p, q, t) return p + (q - p) * t end
  local function vertex(x1, y1, z1, x2, y2, z2, t, u, v)
    local x, y = lerp(x1, x2, t), lerp(y1, y2, t)
    return {x, y, onMesh(x, y, lerp(z1, z2, t)) + lift, u, v}
  end
  local ax1, ay1, az1 = at(a, u0 - hw); local ax2, ay2, az2 = at(a, u0 + hw)
  local bx1, by1, bz1 = at(b, u1 - hw); local bx2, by2, bz2 = at(b, u1 + hw)
  local sA, sB = a.s + (b.s - a.s) * t0, a.s + (b.s - a.s) * t1
  local v0 = sA / M.MARKS.repeatV
  local v1 = v0 + (sB - sA) / M.MARKS.repeatV
  local p1 = vertex(ax1, ay1, az1, bx1, by1, bz1, t0, 0, v0)
  local p2 = vertex(ax2, ay2, az2, bx2, by2, bz2, t0, 1, v0)
  local p3 = vertex(ax2, ay2, az2, bx2, by2, bz2, t1, 1, v1)
  local p4 = vertex(ax1, ay1, az1, bx1, by1, bz1, t1, 0, v1)
  -- u - hw is the right side: p1 (right, a) -> p4 (right, b) -> p3 -> p2 is counter-clockwise
  tri(mesh, o, p1, p4, p3, attrUV, UP); tri(mesh, o, p1, p3, p2, attrUV, UP)
end

function M.markMesh(plan, S, c, mesh, o)
  local tick = pacer(plan)
  for _, list in ipairs(c.rows) do
    local defs = {}
    for k = 1, #list do defs[k] = M.linesAt(S, list[k].s) end
    -- the line of `set` nearest u of the same width and pattern, within MARKS.jump
    local function near(set, l)
      local best, bd
      -- an exit's or entry's added lane edge moves across its taper: it follows its own id (it met the
      -- through edge line of the same width at the taper's start, bundle5)
      if l.id:find('^added') then
        for _, o in ipairs(set) do
          if o.id == l.id and abs(o.u - l.u) <= 3 * M.MARKS.jump then return o end
        end
      end
      -- a host's line along a branch goes on as its edge line where the gore ends (the gore's outline had a gap there, the
      -- two of different widths)
      local side = l.id:match('^fork(%u)$') or l.id:match('^edge(%u)$')
      for _, o in ipairs(set) do
        local same = abs(o.width - l.width) < 1e-6
        if not same and side and o.id ~= l.id and (o.id == 'fork' .. side or o.id == 'edge' .. side) then same = true end
        if same then
          local d = abs(o.u - l.u)
          if d <= M.MARKS.jump and (not bd or d < bd) then best, bd = o, d end
        end
      end
      return best
    end
    for k = 1, #list - 1 do
      local a, b = list[k], list[k + 1]
      tick()
      local mids = M.linesAt(S, (a.s + b.s) / 2)
      for _, la in ipairs(defs[k]) do
        local id = la.id
        local lb, lm = near(defs[k + 1], la), near(mids, la)
        -- a line that jumps between two rows (a lane gain or drop: the OSM line moves with the lanes'
        -- centre) ends there instead of crossing the lanes
        if lb and lm and abs(lb.u - la.u) > M.MARKS.jump and not id:find('^added') then lb = nil end
        -- a fork's branch is cut along its host's edge line over S.cutA..S.cutB (the rows' eL / eR are the cut surface), but its
        -- lane block is fitted to the uncut section: the block's lines then lie on the host's lanes, and run diagonally across
        -- them where the branch's centre line swings (in testing: the loop's edge lines across the driven lane). A line on the
        -- cut side beyond the cut edge is not drawn, except the branch's far-side edge line, which keeps to the cut edge (the
        -- host's paved edge) until the branch's own line is inside the surface (bundle9c)
        local ua, ub = la.u, lb and lb.u
        if lb and S.fork and a.s >= S.cutA - 1e-6 and b.s <= S.cutB + 1e-6 then
          local tol, cutSide = M.MARKS.clip, S.innerSide
          local far = id == (cutSide > 0 and 'edgeR' or 'edgeL')
          local function fit(row, u)
            local over = cutSide > 0 and u - (row.eL + tol) or (-row.eR - tol) - u
            if over <= 0 then return u end
            -- (not where the cut leaves the branch no surface: its own edge there lies on the host's lane, a render of a
            -- fork whose host keeps its lanes drew it 0.5 m inside the host's lane for 15 m from the node)
            if far and row.eL + row.eR > .1 then return cutSide > 0 and row.eL or -row.eR end
          end
          ua, ub = fit(a, ua), fit(b, ub)
          if not (ua and ub) then lb = nil end
          -- a lane split's first rows, 5 cm apart: the far line kept to the cut edge in the zero-width row at the node would
          -- run across the lane to its own line in the next
          if lb and S.split and b.s - a.s < .5 and abs(ub - ua) > .25 then lb = nil end
        end
        if lb and lm then
          -- the line inside the drawn surface only (a cut slip road's inner line lies on its host)
          local len = b.s - a.s
          if lm.dash then
            local on, period = lm.dash[1], lm.dash[1] + lm.dash[2]
            local start = floor(a.s / period) * period
            while start < b.s do
              local d0, d1 = max(a.s, start), min(b.s, start + on)
              if d1 - d0 > 1e-3 then markPiece(plan, S, mesh, o, a, b, ua, ub, lm.width, (d0 - a.s) / len, (d1 - a.s) / len, id) end
              start = start + period
            end
          else
            markPiece(plan, S, mesh, o, a, b, ua, ub, lm.width, 0, 1, id)
          end
        end
      end
    end
  end
  -- the gore of a slip road: chevrons between the host's through edge line and its own inner line
  if S.cutA and (S.hostEvent or S.fork) and c.rows[1] then M.goreMesh(plan, S, c, mesh, o) end
  -- phase 3: the approaches' stop and give-way lines, arrows and crossings in this chunk
  if S.approaches then M.approachMarks(plan, S, c, mesh, o) end
end

function M.goreMesh(plan, S, c, mesh, o)
  local H, side = S.host, S.innerSide
  local hevent = S.hostEvent
  local hb, hostEdge
  if hevent then
    hb = M.laneBlock(H, hevent.separate)
    hostEdge = side > 0 and hb.right or hb.left
  else
    -- a fork, diverge or merge without a lane attachment: the host's edge line on the side the branch lies (bundle8)
    hb = M.laneBlock(H, S.hint((S.cutA + S.cutB) / 2))
    hostEdge = (S.hostSide or -side) > 0 and hb.left or hb.right
  end
  local ob = M.laneBlock(S, (S.cutA + S.cutB) / 2)
  local own = side > 0 and ob.left or ob.right
  local H2 = M.MARKS.hatch
  local lift = plan.lift + M.MARKS.lift
  -- the gore's two lines in slip rows: u of the slip's inner line, u of the host's through edge line
  local pts = {}
  for _, list in ipairs(c.rows) do
    for _, r in ipairs(list) do
      if r.s >= S.cutA - 1e-6 and r.s <= S.cutB + 1e-6 then
        local he, ui = hostEdge, own
        if S.split then
          -- a lane split's blocks sit off their centre lines up to the nose (fast_roads.lane_splits): both lines per row
          local b = M.laneBlock(H, r.hint)
          he = (S.hostSide or -side) > 0 and b.left or b.right
          local ob2 = M.laneBlock(S, r.s)
          ui = side > 0 and ob2.left or ob2.right
        end
        local t = H:rayTo(r.x, r.y, r.nx * side, r.ny * side, he, r.hint)
        if t then pts[#pts + 1] = {row = r, uIn = ui, uHost = t * side} end
      end
    end
  end
  if #pts < 2 then return end
  local function at(p, u)
    local x, y = p.row.x + p.row.nx * u, p.row.y + p.row.ny * u
    return {x, y, M.rowZ(p.row, u) + lift}
  end
  -- stripes every `every` m along the slip, from its inner line forward to the host's edge line
  local s0 = pts[1].row.s
  local first = math.ceil((s0 - S.cutA) / H2.every - .5) * H2.every + S.cutA + H2.every / 2
  local lastS = pts[#pts].row.s
  local function interp(s)
    for k = 1, #pts - 1 do
      local p, q = pts[k], pts[k + 1]
      if s >= p.row.s - 1e-6 and s <= q.row.s + 1e-6 then
        local t = (s - p.row.s) / max(EPS, q.row.s - p.row.s)
        return p, q, t
      end
    end
  end
  local function lineAt(s, which)
    local p, q, t = interp(s)
    if not p then return nil end
    local a, b = at(p, p[which]), at(q, q[which])
    return {a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t}
  end
  local s = first
  while s < lastS do
    local A = lineAt(s, 'uIn')
    if A then
      local p, q, t = interp(s)
      local gap = abs((p.uHost - p.uIn) + ((q.uHost - q.uIn) - (p.uHost - p.uIn)) * t)
      local sB = s + gap / math.tan(H2.angle)
      local B = lineAt(min(sB, lastS), 'uHost')
      if B and gap > 2 * H2.margin + .3 and H2.chevron and (not S.fork or (gap >= H2.forkMinGap and abs(s - (S.blend0 or lastS)) <= H2.forkLast)) then
        M.chevron(plan, mesh, o, s, gap, lineAt, interp, lastS, H2)
      elseif B and gap > 2 * H2.margin + .3 and not H2.chevron then
        if sB > lastS then
          -- the gore ends at the nose: shorten the stripe on its line
          local f = (lastS - s) / max(EPS, sB - s)
          B = {A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f, A[3] + (B[3] - A[3]) * f}
        end
        local dx, dy = B[1] - A[1], B[2] - A[2]
        local l = sqrt(dx * dx + dy * dy)
        if l > 2 * H2.margin + .2 then
          dx, dy = dx / l, dy / l
          local nx, ny = -dy * H2.stripe / 2, dx * H2.stripe / 2
          local a0 = {A[1] + dx * H2.margin, A[2] + dy * H2.margin, A[3] + (B[3] - A[3]) * H2.margin / l}
          local b0 = {B[1] - dx * H2.margin, B[2] - dy * H2.margin, B[3] - (B[3] - A[3]) * H2.margin / l}
          local v1 = (l - 2 * H2.margin) / M.MARKS.repeatV
          local q1 = {a0[1] - nx, a0[2] - ny, a0[3], 0, 0}
          local q2 = {b0[1] - nx, b0[2] - ny, b0[3], 0, v1}
          local q3 = {b0[1] + nx, b0[2] + ny, b0[3], 1, v1}
          local q4 = {a0[1] + nx, a0[2] + ny, a0[3], 1, 0}
          tri(mesh, o, q1, q2, q3, attrUV, UP); tri(mesh, o, q1, q3, q4, attrUV, UP)
        end
      end
    end
    s = s + H2.every
  end
end

-- a gore chevron (Sperrflaeche, Z 298): its apex on the gore's middle at s, its arms forward to both
-- lines
function M.chevron(plan, mesh, o, s, gap, lineAt, interp, lastS, H2)
  local sB = s + (gap / 2) / math.tan(H2.angle)
  if sB > lastS then return end
  local A1, A2, B1, B2 = lineAt(s, 'uIn'), lineAt(s, 'uHost'), lineAt(sB, 'uIn'), lineAt(sB, 'uHost')
  if not (A1 and A2 and B1 and B2) then return end
  local apex = {(A1[1] + A2[1]) / 2, (A1[2] + A2[2]) / 2, (A1[3] + A2[3]) / 2}
  for _, B in ipairs({B1, B2}) do
    local dx, dy = B[1] - apex[1], B[2] - apex[2]
    local l = sqrt(dx * dx + dy * dy)
    if l > 2 * H2.margin + .2 then
      dx, dy = dx / l, dy / l
      local nx, ny = -dy * H2.stripe / 2, dx * H2.stripe / 2
      local a0 = {apex[1] + dx * H2.margin * .2, apex[2] + dy * H2.margin * .2, apex[3]}
      local b0 = {B[1] - dx * H2.margin, B[2] - dy * H2.margin, B[3] - (B[3] - apex[3]) * H2.margin / l}
      local v1 = (l - 1.2 * H2.margin) / M.MARKS.repeatV
      local q1 = {a0[1] - nx, a0[2] - ny, a0[3], 0, 0}
      local q2 = {b0[1] - nx, b0[2] - ny, b0[3], 0, v1}
      local q3 = {b0[1] + nx, b0[2] + ny, b0[3], 1, v1}
      local q4 = {a0[1] + nx, a0[2] + ny, a0[3], 1, 0}
      tri(mesh, o, q1, q2, q3, attrUV, UP); tri(mesh, o, q1, q3, q4, attrUV, UP)
    end
  end
end

-- structures of S at s of a kind
local function structureAt(S, s, kind)
  for _, r in ipairs(S.structures) do
    if r.kind == kind and s >= r.s0 - 1e-6 and s <= r.s1 + 1e-6 then return r end
  end
end
M.structureAt = structureAt

-- Does the deck of S have a parapet on `side` at s, as deckMesh decides it: on an outer edge (no lane boundary) with no other
-- surface at about its height within DECK.clear outside it. s is clamped into chunk c, depth is the deck's.
function M.parapetStands(plan, S, c, s, side, depth)
  local D = M.DECK
  s = max(c.s[1], min(c.s[c.n], s))
  local row = M.row(plan, S, c, s)
  local e = side > 0 and row.eL or row.eR
  if e <= .05 or M.sharedSide(S, s, side) then return false end
  local own = function(hit) return hit.S == S and abs(hit.s - s) < 40 end
  local z = M.rowZ(row, side * e)
  for _, du in ipairs({.05, D.thick, D.thick + D.clear}) do
    local u = side * (e + du)
    if M.onSurface(plan, row.x + row.nx * u, row.y + row.ny * u, z - depth / 2, .02, depth / 2 + 2.0, own) then return false end
  end
  return true
end

-- The bridges a mapped rail run (f.s0..f.s1) ends at: the one that begins just past its end (hi) and the one that ends just
-- before its start (lo), each within TUNE.railJoin m. A rail starting or ending on a bridge is left as mapped.
local function railJoins(S, f, T)
  if not (T.railJoin and T.railJoin > 0) then return nil, nil end
  local lo, hi
  for _, r in ipairs(S.structures or NONE) do
    if r.kind == 'bridge' then
      if r.s0 >= f.s1 - 1e-6 and r.s0 - f.s1 <= T.railJoin and (not hi or r.s0 < hi.s0) then hi = r end
      if r.s1 <= f.s0 + 1e-6 and f.s0 - r.s1 <= T.railJoin and (not lo or r.s1 > lo.s1) then lo = r end
    end
  end
  return lo, hi
end

-- rails: the mapped runs in chunk c's range, sampled at its rows, cut where they reach a drawn surface. A run that ends
-- within TUNE.railJoin m of a bridge deck runs on to it and joins its parapet (railJoins, TUNE.railTaper); depth: the deck's.
function M.railPaths(plan, S, c, depth)
  local T = M.TUNE
  local tick = pacer(plan)
  local out = {}
  local c0, c1 = c.s[1], c.s[c.n]
  for _, f in ipairs(S.furniture) do
    local kind = M.RAIL.kinds[f.kind]
    local lo, hi = nil, nil
    if kind then lo, hi = railJoins(S, f, T) end
    local jo = kind == 'jersey' and T.railJoinJersey or T.railJoinOffset
    -- the parapet must stand at that end of the deck, else the rail is left as mapped
    if lo and not M.parapetStands(plan, S, c, lo.s1 - 1, f.side, depth or 1.5) then lo = nil end
    if hi and not M.parapetStands(plan, S, c, hi.s0 + 1, f.side, depth or 1.5) then hi = nil end
    local f0 = lo and lo.s1 - T.railOverrun or f.s0
    local f1 = hi and hi.s0 + T.railOverrun or f.s1
    local a, b = max(f0, c0, S.D0), min(f1, c1, S.D1)
    if kind and b - a > 1e-3 then
      -- sample s: the stations and rows inside, the ends; skip the fan junctions (+ railStop)
      local stops = {}
      for _, r in ipairs(S.cuts) do stops[#stops + 1] = {r[1] - T.railStop, r[2] + T.railStop} end
      local ss, set = {}, {}
      local function add(s)
        if s >= a - 1e-9 and s <= b + 1e-9 then
          local k = floor(s * 1000 + .5)
          if not set[k] then set[k] = true; ss[#ss + 1] = s end
        end
      end
      add(a); add(b)
      for i = 1, c.n do add(c.s[i]) end
      -- and the rows (bundle5: an added lane's S-shaped taper lies between two stations; the beam cut its corner)
      for _, list in ipairs(c.rows) do for _, r in ipairs(list) do add(r.s) end end
      for _, r in ipairs(stops) do add(r[1]); add(r[2]) end
      -- a joined end draws in to the parapet: the samples there are 1.5 m apart (and at the bridge's end itself)
      local reach = T.railTaper + 8 * max(0, max(f.offset, T.railMinOffset) - jo)
      if lo then add(lo.s1); for d = 1.5, reach, 1.5 do add(lo.s1 + d) end end
      if hi then add(hi.s0); for d = 1.5, reach, 1.5 do add(hi.s0 - d) end end
      table.sort(ss)
      -- the sideways offset at s from the paved edge: as mapped, drawn in to the parapet within reach of a joined end
      local function offsetAt(s, full)
        local d = math.huge
        if hi then d = min(d, hi.s0 - s) end
        if lo then d = min(d, s - lo.s1) end
        local len = max(T.railTaper, 8 * (full - jo))
        if d >= len or full <= jo then return full end
        return jo + (full - jo) * smooth(d / len)
      end
      local run = {}
      local function close()
        local length = 0
        for k = 2, #run do length = length + sqrt((run[k][1] - run[k - 1][1]) ^ 2 + (run[k][2] - run[k - 1][2]) ^ 2) end
        if #run >= 2 and length >= T.railMin then
          local join = {}
          if lo and run.a <= lo.s1 + 1e-6 then join.a = lo.s1 end
          if hi and run.b >= hi.s0 - 1e-6 then join.b = hi.s0 end
          out[#out + 1] = {kind = kind, side = f.side, points = run, a = run.a, b = run.b, double = run.double, join = join}
        end
        run = {}
      end
      for _, s in ipairs(ss) do
        tick()
        if insideAny(stops, s) then close() else
          local row = M.row(plan, S, c, s)
          local e = f.side > 0 and row.eL or row.eR
          local full = max(f.offset, T.railMinOffset)
          local u = f.side * (e + full)
          local x, y = row.x + row.nx * u, row.y + row.ny * u
          local zEdge = M.rowZ(row, f.side * e)
          local own = function(hit) return hit.S == S and abs(hit.s - s) < 40 end
          -- a carriageway behind: a median rail, two-faced (or drawn by the other carriageway: 2). It runs on to the bridge as
          -- it is (the decks stand apart, and a two-faced rail ends level with their parapets); a single rail draws in
          local median = kind == 'beam' and M.railBehind(plan, S, x, y, zEdge, row.nx * f.side, row.ny * f.side) or 0
          if median == 0 and (lo or hi) then
            u = f.side * (e + offsetAt(s, full))
            x, y = row.x + row.nx * u, row.y + row.ny * u
          end
          if median == 2 or M.onSurface(plan, x, y, zEdge, T.railClear + M.RAIL.thick, 2.5, own) then close() else
            if #run > 0 and run.double ~= (median == 1) then close() end
            if #run == 0 then run.a, run.double = s, median == 1 end
            run[#run + 1] = {x, y, zEdge, s, row.h}
            run.b = s
          end
        end
      end
      close()
    end
  end
  return out
end

-- Which way a rail faces. Its side comes from the stroke it runs along (fast_roads.furniture: the side of the nearest
-- station, never the barrier way's direction), so the beam faces that stroke's traffic and the posts stand behind it.
-- A rail with another carriageway behind it (an OSM median barrier, which the pack gives to both carriageways, or a
-- rule:median rail, which it gives to one) is a median rail: a beam on each face and the posts in the middle. One-faced,
-- the other carriageway saw the posts in front of a flat back, or the posts of the second copy through its beam.
-- Returns 0 (no carriageway behind), 1 (a median rail this stroke draws) or 2 (the other carriageway carries the same
-- rail on its facing side and its id sorts first: it draws it). (ux, uy): unit across, away from this stroke's road.
local BEHIND = {}
local function behindTest(z, S2) return S2 ~= nil and S2 ~= BEHIND.S and abs(z - BEHIND.z) < 2.5 end
function M.railBehind(plan, S, x, y, z, ux, uy)
  BEHIND.S, BEHIND.z = S, z
  local S2, s2
  for _, d in ipairs(M.TUNE.railBehind) do
    local hz, hS, hs = M.surfaceWhere(plan, x + ux * d, y + uy * d, .05, behindTest)
    if hz then S2, s2 = hS, hs; break end
  end
  BEHIND.S = nil
  if not S2 then return 0 end
  local st = S2:at(s2)
  if not st or not s2 then return 1 end
  -- the side of S2 the rail stands on
  local towards = (x - st.x) * -sin(st.h) + (y - st.y) * cos(st.h) > 0 and 1 or -1
  for _, f in ipairs(S2.furniture) do
    if M.RAIL.kinds[f.kind] == 'beam' and f.side == towards and s2 >= f.s0 - 2 and s2 <= f.s1 + 2 then
      return tostring(S2.id) < tostring(S.id) and 2 or 1
    end
  end
  return 1
end

-- a W-beam (two folds on the road side, a flat back, a top) and posts, or a concrete barrier profile; a median rail
-- (path.double) mirrors the beam about the posts and centres them on the rail's line
local ONE, BOTH = {1}, {1, -1}
function M.railMesh(plan, S, c, path, mesh, o, lift, look)
  local R = M.RAIL
  -- the stock sheet's UVs (M.RAIL.stock) for a beam and its posts; a concrete barrier keeps ours
  local K = look and look.stock and path.kind ~= 'jersey' and look
  local pts = path.points
  local side = path.side
  local c0, c1 = c.s[1], c.s[c.n]
  local function frame(k)
    local p, q = pts[max(1, k - 1)], pts[min(#pts, k + 1)]
    local dx, dy = q[1] - p[1], q[2] - p[2]
    local l = max(EPS, sqrt(dx * dx + dy * dy))
    -- the road side: towards the stroke's centre line
    return dx / l, dy / l
  end
  local uv = function(p) return (p[4] or 0), (p[5] or 0) end
  local prof
  if path.kind == 'jersey' then
    local J = R.jersey
    prof = {{0, 0}, {J.base / 2 - .05, .08}, {J.top / 2 + .03, .33}, {J.top / 2, J.height}, {-J.top / 2, J.height},
      {-J.top / 2 - .03, .33}, {-J.base / 2 + .05, .08}, {-J.base / 2, 0}}
    -- across: + towards the road
    for _, q in ipairs(prof) do q[1] = q[1] end
  else
    local low = R.height - R.beam
    prof = {{0, low}, {R.depth, low + R.beam * .3}, {R.depth, low + R.beam * .7}, {0, R.height}, {-R.thick, R.height}, {-R.thick, low}}
  end
  -- a median rail: the posts on the line, each beam's back on them
  local double = path.double and path.kind == 'beam'
  local shift = double and R.thick + R.post / 2 or 0
  local function vert(k, i, m)
    local p = pts[k]
    local tx, ty = frame(k)
    -- (rx, ry): unit across towards the road (m = -1: the mirrored beam, towards the carriageway behind)
    local rx, ry = ty * side, -tx * side
    local q = prof[i]
    local u = m * (q[1] + shift)
    if K then
      return {p[1] + rx * u, p[2] + ry * u, p[3] + lift + q[2], (p[4] or 0) / K.panel,
        K.v0 + (q[2] - (R.height - R.beam)) / R.beam * K.dv}
    end
    return {p[1] + rx * u, p[2] + ry * u, p[3] + lift + q[2], (p[4] or 0) / 3, q[2] / 3}
  end
  local closed = path.kind == 'jersey'
  for _, m in ipairs(double and BOTH or ONE) do
    for k = 1, #pts - 1 do
      for i = 1, #prof - (closed and 0 or 1) do
        local j = i % #prof + 1
        if not (path.kind ~= 'jersey' and i == #prof) then
          local a, b, cc, d = vert(k, i, m), vert(k + 1, i, m), vert(k + 1, j, m), vert(k, j, m)
          -- facing: away from the beam's inside (its centroid)
          local tx, ty = frame(k)
          local rx, ry = ty * side * m, -tx * side * m
          local cu, cz = 0, 0
          for _, q in ipairs(prof) do cu, cz = cu + q[1], cz + q[2] end
          cu, cz = cu / #prof, cz / #prof
          local fu = (prof[i][1] + prof[j][1]) / 2 - cu
          local fz = (prof[i][2] + prof[j][2]) / 2 - cz
          local facing = {rx * fu, ry * fu, fz}
          tri(mesh, o, a, b, cc, uv, facing); tri(mesh, o, a, cc, d, uv, facing)
        end
      end
    end
    -- end caps where the run ends inside the chunk (not at a chunk join)
    for _, e in ipairs({{1, path.a}, {#pts, path.b}}) do
      local k, s = e[1], e[2]
      if s and s > c0 + 1e-6 and s < c1 - 1e-6 then
        local tx, ty = frame(k)
        local f = k == 1 and -1 or 1
        local ring = {}
        for i = 1, #prof do ring[i] = vert(k, i, m) end
        for i = 2, #ring - 1 do tri(mesh, o, ring[1], ring[i], ring[i + 1], uv, {tx * f, ty * f, 0}) end
      end
    end
  end
  -- posts every R.posts m of the stroke's s (behind the beam; a median rail's on its line), the same in every chunk
  if path.kind == 'beam' then
    local sA, sB = pts[1][4], pts[#pts][4]
    local p0 = math.ceil(sA / R.posts) * R.posts
    for ps = p0, sB, R.posts do
      local k = 1
      while k < #pts - 1 and pts[k + 1][4] < ps do k = k + 1 end
      local p, q = pts[k], pts[k + 1]
      local t = (ps - p[4]) / max(EPS, q[4] - p[4])
      local x, y, z = p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, p[3] + (q[3] - p[3]) * t
      local tx, ty = q[1] - p[1], q[2] - p[2]
      local l = max(EPS, sqrt(tx * tx + ty * ty))
      tx, ty = tx / l, ty / l
      local rx, ry = ty * side, -tx * side
      local back = double and 0 or R.thick + R.post / 2
      local cx, cy = x - rx * back, y - ry * back
      local h = R.post / 2
      local z0, z1 = z + lift - R.sink, z + lift + R.height - .02
      -- (u0, v0) the post's foot corner, (u1, v1) its top corner: its own metres, or the stock sheet's plain patch
      local u0, v0, u1, v1 = 0, 0, R.post, R.height
      if K then u0, v1, u1, v0 = K.post[1], K.post[2], K.post[3], K.post[4] end
      local corners = {{cx + tx * h + rx * h, cy + ty * h + ry * h}, {cx - tx * h + rx * h, cy - ty * h + ry * h},
        {cx - tx * h - rx * h, cy - ty * h - ry * h}, {cx + tx * h - rx * h, cy + ty * h - ry * h}}
      for i = 1, 4 do
        local a, b = corners[i], corners[i % 4 + 1]
        local mx, my = (a[1] + b[1]) / 2 - cx, (a[2] + b[2]) / 2 - cy
        local A, B = {a[1], a[2], z0, u0, v0}, {b[1], b[2], z0, u1, v0}
        local C, D = {b[1], b[2], z1, u1, v1}, {a[1], a[2], z1, u0, v1}
        tri(mesh, o, A, B, C, uv, {mx, my, 0}); tri(mesh, o, A, C, D, uv, {mx, my, 0})
      end
    end
  end
end

-- is side `side` (+1 left) of S at s a lane boundary rather than an outer edge: an exit or entry's added
-- lane or gore (taper to nose), a branch drawn along this edge (fork gap), the clamped host side, a slip
-- road's inner side cut along its host, or a junction's mouth
local function sharedSide(S, s, side)
  for _, e in ipairs(S.events) do
    if e.side == side and s >= min(e.taper, e.nose) - 2 and s <= max(e.taper, e.nose) + 2 then return true end
  end
  for _, g in ipairs(S.edgeGaps) do
    if g.side == side and s >= g.a - 2 and s <= g.b + 2 then return true end
  end
  for _, cl in ipairs(S.clamps or NONE) do
    if cl.side == side and s >= cl.a - 2 and s <= cl.b + 2 then return true end
  end
  if S.cutA and side == S.innerSide then
    local lo = min(S.cutA, S.blend0 + S.blendDir * M.TUNE.noseEase)
    local hi = max(S.cutB, S.blend0 + S.blendDir * M.TUNE.noseEase)
    if s >= lo - 2 and s <= hi + 2 then return true end
  end
  for _, r in ipairs(S.cuts) do
    if s >= r[1] - 2 and s <= r[2] + 2 then return true end
  end
  return false
end
M.sharedSide = sharedSide

-- Rule rails at a drop. The generator's ROADS.embankmentRails walks only its own strips, and
-- the swept roads leave them, so a motorway or a slip road on a high embankment had a rail only where one was mapped.
-- Where a swept carriageway's paved edge stands `height` m or more over the raw DEM `probe` m out from it for at least
-- `run` m, and the pack has no barrier on that side, a guard rail runs `offset` m outside the edge: railMesh with the
-- edge's side, so the beam faces the road and the posts stand behind it (as the mapped ones). None on bridges (parapets)
-- or in tunnels, on a lane boundary (gores, added lanes, fan cuts: sharedSide), where another carriageway at about this
-- height lies out to the probe (a divided road's inner side), or on a stroke whose rails come from the road register
-- (e2r:rails=register: its own barrier ways). The run is judged on one lattice of s (`step` m) over the chunk and `run`
-- m either side of it, so two chunks agree where they meet. Values: ROADS.embankment's (3 m over 8 m, 30 m runs; memory,
-- not checked against the RPS).
M.DROP = {height = 3, probe = 8, run = 30, step = 4, offset = .5}
local function mappedSide(S, s, side)
  for _, f in ipairs(S.furniture) do
    if f.side == side then
      -- a mapped run that ends short of a bridge runs on to it (railPaths): the stretch up to it is mapped too
      local lo, hi = railJoins(S, f, M.TUNE)
      local a, b = lo and lo.s1 - M.TUNE.railOverrun or f.s0, hi and hi.s0 + M.TUNE.railOverrun or f.s1
      if s >= a - 3 and s <= b + 3 then return true end
    end
  end
  return false
end
-- A rule rail that ends at a bridge runs on to it like a mapped one (TUNE.railJoin, railPaths): samples just outside the
-- bridge's start and end, every 1.5 m over the draw-in, and one `railOverrun` m inside the bridge, at the parapet. depth: the deck's.
function M.dropRails(plan, S, c, dem, depth)
  local D, T = M.DROP, M.TUNE
  local out = {}
  if not dem or (S.tags or NONE)['e2r:rails'] == 'register' then return out end
  local tick = pacer(plan)
  local c0, c1 = c.s[1], c.s[c.n]
  local lo, hi = max(c0 - D.run, S.D0 or c0 - D.run), min(c1 + D.run, S.D1 or c1 + D.run)
  if hi - lo < D.run then return out end
  local ss = {}
  for k = math.ceil(lo / D.step), floor(hi / D.step) do ss[#ss + 1] = k * D.step end
  ss[#ss + 1] = max(c0, lo); ss[#ss + 1] = min(c1, hi)
  -- the bridges whose ends lie in the range: the samples that join a rail to them (and the stretch of the draw-in)
  local joins, inside = {}, {}
  if T.railJoin and T.railJoin > 0 then
    local reach = T.railTaper + 8 * max(0, D.offset - T.railJoinOffset)
    for _, r in ipairs(S.structures or NONE) do
      if r.kind == 'bridge' then
        for _, e in ipairs({{r.s0, -1}, {r.s1, 1}}) do
          local s, away = e[1], e[2]
          if s >= lo and s <= hi then
            joins[#joins + 1] = {s = s, away = away, reach = reach}
            ss[#ss + 1] = s + away * 1e-3
            for d = 1.5, reach, 1.5 do ss[#ss + 1] = s + away * d end
            local deep = s - away * T.railOverrun
            ss[#ss + 1] = deep
            inside[floor(deep * 1000 + .5)] = joins[#joins]
          end
        end
      end
    end
  end
  table.sort(ss)
  -- the sideways offset at s: D.offset, drawn in to the parapet within reach of a bridge's end where one stands on `side`
  local function offsetAt(s, side, standing)
    local off = D.offset
    for _, j in ipairs(joins) do
      local d = (s - j.s) * j.away
      if standing[j] and d < j.reach then
        local w = smooth(max(0, d) / j.reach)
        off = min(off, T.railJoinOffset + (D.offset - T.railJoinOffset) * w)
      end
    end
    return off
  end
  for _, side in ipairs({1, -1}) do
    -- the bridge ends with a parapet on this side
    local standing = {}
    for _, j in ipairs(joins) do
      local cc = j.s >= c0 - 1e-6 and j.s <= c1 + 1e-6 and c or S:chunkAt(j.s)
      if cc and M.parapetStands(plan, S, cc, j.s - j.away, side, depth or 1.5) then standing[j] = true end
    end
    -- per sample: its rail point when the edge stands over a drop, else false
    local at = {}
    for i, s in ipairs(ss) do
      tick()
      local cc = s >= c0 - 1e-6 and s <= c1 + 1e-6 and c or S:chunkAt(s)
      at[i] = false
      -- a sample inside a bridge is the overrun of a joined rail, only where the parapet stands
      local deep = inside[floor(s * 1000 + .5)]
      deep = deep and standing[deep]
      if cc and (deep or not structureAt(S, s, 'bridge')) and not structureAt(S, s, 'tunnel') and not sharedSide(S, s, side)
          and (deep or not mappedSide(S, s, side)) then
        local row = M.row(plan, S, cc, s)
        local e = side > 0 and row.eL or row.eR
        local zEdge = M.rowZ(row, side * e)
        local px, py = row.x + row.nx * side * (e + D.probe), row.y + row.ny * side * (e + D.probe)
        local below = dem(px, py)
        if below and zEdge - below >= D.height then
          BEHIND.S, BEHIND.z = S, zEdge
          local mx, my = row.x + row.nx * side * (e + D.probe / 2), row.y + row.ny * side * (e + D.probe / 2)
          local other = M.surfaceWhere(plan, px, py, .05, behindTest) or M.surfaceWhere(plan, mx, my, .05, behindTest)
          BEHIND.S = nil
          if not other then
            local u = side * (e + offsetAt(s, side, standing))
            local x, y = row.x + row.nx * u, row.y + row.ny * u
            local own = function(hit) return hit.S == S and abs(hit.s - s) < 40 end
            if not M.onSurface(plan, x, y, zEdge, T.railClear + M.RAIL.thick, 2.5, own) then at[i] = {x, y, zEdge, s, row.h} end
          end
        end
      end
    end
    -- the runs of `run` m and more; their samples inside the chunk
    local i = 1
    while i <= #ss do
      if not at[i] then i = i + 1 else
        local j = i
        while j < #ss and at[j + 1] do j = j + 1 end
        if ss[j] - ss[i] >= D.run then
          local run = {}
          for k = i, j do
            if ss[k] >= c0 - 1e-6 and ss[k] <= c1 + 1e-6 and (#run == 0 or ss[k] > run[#run][4] + 1e-6) then run[#run + 1] = at[k] end
          end
          if #run >= 2 then out[#out + 1] = {kind = 'beam', side = side, points = run, a = run[1][4], b = run[#run][4], rule = 'drop'} end
        end
        i = j + 1
      end
    end
  end
  return out
end

-- a bridge deck's edge faces, soffit and parapets along the rows inside bridge structures, in pieces of
-- at most DECK.piece m. A side is an outer edge only where it is no lane boundary (sharedSide) and no
-- other driving surface lies within DECK.clear m outside it at about its height; only there does the
-- deck get a parapet, its outer face and the soffit's overhang. Elsewhere the deck ends at its paved
-- edge with nothing hanging (an adjacent carriageway or slip road at another height met a wall:
-- in-game crashes in testing). Phase 2 tested one point per row interval and always hung the
-- outer face.
-- The tags that choose a chunk's deck material: the way's, with bridge:structure from the chunk's
-- bridge run (fast_roads.structures writes it there) where the way's tags lack it.
function M.deckTags(S, c)
  local tags = S.tags or NONE
  if tags['bridge:structure'] then return tags end
  for _, r in ipairs(S.structures or NONE) do
    if r.kind == 'bridge' and r.structure and r.structure ~= '' and r.s1 >= c.s[1] and r.s0 <= c.s[c.n] then
      return {material = tags.material, ['bridge:structure'] = r.structure}
    end
  end
  return tags
end

-- The ends are closed: a parapet where it starts or stops (beside another surface, at the deck's ends)
-- and the slab between its outer faces where a bridge stretch of the rows begins or ends. They stood
-- open, and only the double-sided concrete hid the hollow; a deck of ROADS.deckLook's single-sided
-- materials would show it.
function M.deckMesh(plan, S, c, mesh, o, lift, depth)
  local D = M.DECK
  local uv = function(p) return (p[4] or 0), (p[5] or 0) end
  local tick = pacer(plan)
  -- a point du outside the row edge on `side`, dz over the surface, at t between rows a and b
  local function edgeAt(a, b, side, t, du, dz, U, V)
    local function P(row, u)
      return {row.x + row.nx * u, row.y + row.ny * u, M.rowZ(row, u > 0 and min(u, row.eL) or max(u, -row.eR)) + lift + dz}
    end
    local p = P(a, (side > 0 and a.eL or -a.eR) + side * du)
    local q = P(b, (side > 0 and b.eL or -b.eR) + side * du)
    return {p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, p[3] + (q[3] - p[3]) * t, U, V}
  end
  local function along(a, b, dir)
    local tx, ty = b.x - a.x, b.y - a.y
    local l = math.sqrt(tx * tx + ty * ty)
    return l > 1e-9 and {tx / l * dir, ty / l * dir, 0} or nil
  end
  -- a parapet's cross-section at t, facing along the rows (dir -1 back, +1 ahead)
  local function parapetCap(a, b, side, t, dir)
    local facing = along(a, b, dir)
    if not facing then return end
    local w, h = D.thick / 2.5, D.parapet / 2.5
    local p0, p1 = edgeAt(a, b, side, t, 0, 0, 0, h), edgeAt(a, b, side, t, D.thick, 0, w, h)
    local p2, p3 = edgeAt(a, b, side, t, D.thick, D.parapet, w, 0), edgeAt(a, b, side, t, 0, D.parapet, 0, 0)
    tri(mesh, o, p0, p1, p2, uv, facing)
    tri(mesh, o, p0, p2, p3, uv, facing)
  end
  -- the slab's cross-section at t between its outer faces (reach: per side, how far they stand out)
  local function slabCap(a, b, t, reach, dir)
    local facing = along(a, b, dir)
    if not facing then return end
    local w = (a.eL + a.eR + reach[1] + reach[-1]) / 2.5
    local p0, p1 = edgeAt(a, b, -1, t, reach[-1], -depth, 0, depth / 2.5), edgeAt(a, b, 1, t, reach[1], -depth, w, depth / 2.5)
    local p2, p3 = edgeAt(a, b, 1, t, reach[1], 0, w, 0), edgeAt(a, b, -1, t, reach[-1], 0, 0, 0)
    tri(mesh, o, p0, p1, p2, uv, facing)
    tri(mesh, o, p0, p2, p3, uv, facing)
  end
  for _, list in ipairs(c.rows) do
    -- was: per side, whether the previous bridge piece's parapet stood (nil off a bridge);
    -- last: that piece {a, b, t1, reach}, for the caps where the bridge stretch ends
    local was, last = {}, nil
    local function close()
      if not last then return end
      for _, side in ipairs({1, -1}) do
        if was[side] then parapetCap(last[1], last[2], side, last[3], 1) end
      end
      slabCap(last[1], last[2], last[3], last[4], 1)
      was, last = {}, nil
    end
    for k = 1, #list - 1 do
      local a, b = list[k], list[k + 1]
      tick()
      if not structureAt(S, (a.s + b.s) / 2, 'bridge') then close() end
      if structureAt(S, (a.s + b.s) / 2, 'bridge') then
        local n = max(1, math.ceil((b.s - a.s) / D.piece))
        local function P(row, u, dz)
          return {row.x + row.nx * u, row.y + row.ny * u, M.rowZ(row, u > 0 and min(u, row.eL) or max(u, -row.eR)) + lift + dz}
        end
        local function own(hit) return hit.S == S and abs(hit.s - (a.s + b.s) / 2) < 40 end
        for i = 0, n - 1 do
          local t0, t1 = i / n, (i + 1) / n
          local sm = a.s + (b.s - a.s) * (t0 + t1) / 2
          local u0, u1 = (a.s + (b.s - a.s) * t0) / 2.5, (a.s + (b.s - a.s) * t1) / 2.5
          local reach = {}
          for _, side in ipairs({1, -1}) do
            local ea, eb = side > 0 and a.eL or -a.eR, side > 0 and b.eL or -b.eR
            local function at(t, du, dz, U, V)
              local p, q = P(a, ea + side * du, dz), P(b, eb + side * du, dz)
              return {p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, p[3] + (q[3] - p[3]) * t, U, V}
            end
            local nx, ny = (a.nx + b.nx) / 2 * side, (a.ny + b.ny) / 2 * side
            local free = abs(ea) > .05 and abs(eb) > .05 and not sharedSide(S, sm, side)
            if free then
              for _, t in ipairs({t0, (t0 + t1) / 2, t1}) do
                for _, du in ipairs({.05, D.thick, D.thick + D.clear}) do
                  local p = at(t, du, -lift, 0, 0)
                  -- any other surface at about this height, or below within the deck's depth
                  if M.onSurface(plan, p[1], p[2], p[3] - depth / 2, .02, depth / 2 + 2.0, own) then free = false end
                end
              end
            end
            reach[side] = free and D.thick or 0
            if free and not was[side] then parapetCap(a, b, side, t0, -1) end
            if not free and was[side] then parapetCap(a, b, side, t0, 1) end
            was[side] = free
            if free then
              local top, low = D.parapet, -depth
              tri(mesh, o, at(t0, D.thick, top, u0, 0), at(t1, D.thick, top, u1, 0), at(t1, D.thick, low, u1, 1), uv, {nx, ny, 0})
              tri(mesh, o, at(t0, D.thick, top, u0, 0), at(t1, D.thick, low, u1, 1), at(t0, D.thick, low, u0, 1), uv, {nx, ny, 0})
              tri(mesh, o, at(t0, 0, top, u0, 0), at(t1, 0, top, u1, 0), at(t1, 0, 0, u1, .4), uv, {-nx, -ny, 0})
              tri(mesh, o, at(t0, 0, top, u0, 0), at(t1, 0, 0, u1, .4), at(t0, 0, 0, u0, .4), uv, {-nx, -ny, 0})
              tri(mesh, o, at(t0, 0, top, u0, 0), at(t1, 0, top, u1, 0), at(t1, D.thick, top, u1, .1), uv, UP)
              tri(mesh, o, at(t0, 0, top, u0, 0), at(t1, D.thick, top, u1, .1), at(t0, D.thick, top, u0, .1), uv, UP)
            end
          end
          -- soffit, out to the outer faces where they stand, else to the paved edge
          local function Q(t, side)
            local row0, row1 = a, b
            local e0 = side > 0 and row0.eL + reach[1] or -(row0.eR + reach[-1])
            local e1 = side > 0 and row1.eL + reach[1] or -(row1.eR + reach[-1])
            local p, q = P(row0, e0, -depth), P(row1, e1, -depth)
            local sv = a.s + (b.s - a.s) * t
            return {p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t, p[3] + (q[3] - p[3]) * t, sv / 2.5, side}
          end
          tri(mesh, o, Q(t0, -1), Q(t1, -1), Q(t1, 1), uv, {0, 0, -1})
          tri(mesh, o, Q(t0, -1), Q(t1, 1), Q(t0, 1), uv, {0, 0, -1})
          if not last then slabCap(a, b, t0, reach, -1) end
          last = {a, b, t1, reach}
        end
      end
    end
    close()
  end
end

-- the plates' surfaces (fans from the node over the mouths' sections)
function M.plateMesh(plan, pl, mesh, o, lift, uv)
  for _, t in ipairs(pl.tris) do
    tri(mesh, o, lifted(t[1], lift), lifted(t[2], lift), lifted(t[3], lift), uv, UP)
  end
end

-- the material of a plate: its most important swept road's
local function plateTags(pl)
  local best, rank = nil, -1
  for _, r in ipairs(pl.rows) do
    if r.S then
      local rk = ({motorway = 7, trunk = 6, primary = 5, secondary = 4, tertiary = 3})[r.S.base] or 2
      if r.S.link then rk = rk - .5 end
      if rk > rank then best, rank = r.S.tags, rk end
    end
  end
  return best or NONE
end

-- Kerbs and sidewalks of a swept street (primary..unclassified, not links, not a bridge or tunnel): a kerb face TUNE.kerbHeight
-- high along each paved edge where the street is built up (a sidewalk tagged, or maxspeed 50 or less), and a sidewalk of
-- TUNE.walkWidth where the tags say so (sidewalk=both|left|right; separate: the kerb only). Sides are the stroke's (u > 0
-- left); a part drawn against its way's direction swaps the OSM sides. The rows between two junction cuts only: at a mouth
-- the junction surface's own kerbs stand (M.junctionKerbs).
local STREETS = {primary = true, secondary = true, tertiary = true, unclassified = true}
local function walkSides(tags, rev)
  local both = tags.sidewalk
  local l, r = tags['sidewalk:left'], tags['sidewalk:right']
  local function on(v) return v ~= nil and v ~= 'no' and v ~= 'none' and v ~= 'separate' end
  local left, right = on(l), on(r)
  if both == 'both' then left, right = true, true
  elseif both == 'left' then left = true
  elseif both == 'right' then right = true
  elseif tags['sidewalk:both'] and on(tags['sidewalk:both']) then left, right = true, true end
  if rev then left, right = right, left end
  return left, right
end
function M.streetEdges(plan, S, c, kerb, walk, o, lift)
  if not (M.TUNE.streetKerbs and STREETS[S.base] and not S.link) then return end
  local h, W = M.TUNE.kerbHeight, M.TUNE.walkWidth
  local tick = pacer(plan)
  for _, list in ipairs(c.rows) do
    for k = 1, #list - 1 do
      local a, b = list[k], list[k + 1]
      tick()
      local sm = (a.s + b.s) / 2
      if not structureAt(S, sm, 'bridge') and not structureAt(S, sm, 'tunnel') then
        local tags = M.partTags(S, sm)
        local left, right = walkSides(tags, M.reversedAt(S, sm))
        local speed = number(tostring(tags.maxspeed or ''):lower(), 999)
        local built = left or right or speed <= 50
        if built then
          for _, side in ipairs({1, -1}) do
            local pa, pb = side > 0 and a.L or a.R, side > 0 and b.L or b.R
            if (side > 0 and a.eL or a.eR) > .3 and (side > 0 and b.eL or b.eR) > .3 then
              local A, B = lifted(pa, lift), lifted(pb, lift)
              local A2, B2 = lifted(pa, lift + h), lifted(pb, lift + h)
              local out = {a.nx * side, a.ny * side, 0}
              tri(kerb, o, A, B, B2, attrUV, out); tri(kerb, o, A, B2, A2, attrUV, out)
              if (side > 0 and left) or (side < 0 and right) then
                local C = {A2[1] + a.nx * side * W, A2[2] + a.ny * side * W, A2[3] - .02 * W}
                local D = {B2[1] + b.nx * side * W, B2[2] + b.ny * side * W, B2[3] - .02 * W}
                tri(walk, o, A2, B2, D, planUV(3), UP); tri(walk, o, A2, D, C, planUV(3), UP)
              else
                -- no sidewalk: the kerb's top as a narrow flush strip so the edge reads as a kerb
                local C = {A2[1] + a.nx * side * .25, A2[2] + a.ny * side * .25, A2[3]}
                local D = {B2[1] + b.nx * side * .25, B2[2] + b.ny * side * .25, B2[3]}
                tri(walk, o, A2, B2, D, planUV(3), UP); tri(walk, o, A2, D, C, planUV(3), UP)
              end
            end
          end
        end
      end
    end
  end
end

-- the kerb faces round a junction surface (its kerb edges, the surface on their left)
function M.junctionKerbs(plan, J, kerb, o, lift)
  local h = M.TUNE.kerbHeight
  for _, e in ipairs(J.kerbs) do
    local p, q = J.verts[e[1]], J.verts[e[2]]
    local A, B = lifted(p, lift), lifted(q, lift)
    local A2, B2 = lifted(p, lift + h), lifted(q, lift + h)
    local dx, dy = q[1] - p[1], q[2] - p[2]
    local out = {dy, -dx, 0}
    tri(kerb, o, A, B, B2, attrUV, out); tri(kerb, o, A, B2, A2, attrUV, out)
  end
end

-- items of this tile: one surface, one markings and one furniture item per owned chunk, one for the plates
function M.items(plan, opts)
  local T, out = M.TUNE, {}
  local rect = plan.rect
  local o = {(rect[1] + rect[3]) / 2, (rect[2] + rect[4]) / 2, 0}
  local lift = plan.lift
  local key = opts.tileKey or 'tile'
  local pause = opts.pause or function() end
  local scale = opts.roadScale or 30
  local uv = planUV(scale)
  local look = opts.railLook or M.railLook(M.RAIL)
  local materials = {}
  local stats = {tris = 0, markTris = 0, railTris = 0, items = 0}
  local function item(meshes, name, decor, stroke)
    local mats, tris = {}, 0
    local list = {}
    for _, m in ipairs(meshes) do
      if m.tris > 0 then
        local mesh, n = finish(m)
        list[#list + 1] = mesh
        mats[#mats + 1] = mesh.material
        materials[mesh.material] = true
        tris = tris + n
      end
    end
    if #list == 0 then return 0 end
    out[#out + 1] = {kind = 'proc', name = 'e2rl_' .. key .. '_sw_' .. name, pos = o, meshes = list, triangles = tris,
      mats = mats, layer = 'road', decor = decor or nil, stroke = stroke}
    stats.items = stats.items + 1
    return tris
  end
  local n = 0
  for _, S in pairs(plan.strokes) do
    if S.swept then
      local cuts = plan.conformal and plan.conformal[S.base] and rect
      for _, c in ipairs(S.list) do
        -- a stroke that cuts the ground: a chunk near the tile that another tile holds is drawn here too, inside the rect
        -- (its rows can lie past the box of its stations: surfaceMesh leaves out the quads off the rect)
        if cuts and not c.owned and c.ground and #c.rows > 0 then
          n = n + 1
          local byMat, list = {}, {}
          M.surfaceMesh(plan, S, c, function(tags)
            local mat = 'e2rf_' .. (opts.material and opts.material(tags) or 'road_major')
            if not byMat[mat] then byMat[mat] = newMesh(mat); list[#list + 1] = byMat[mat] end
            return byMat[mat]
          end, o, lift, uv, cuts)
          table.sort(list, function(p, q) return p.material < q.material end)
          stats.tris = stats.tris + item(list, n .. 's', nil, S.id)
          pause()
        end
        if c.owned and #c.rows > 0 then
          n = n + 1
          local byMat, list = {}, {}
          M.surfaceMesh(plan, S, c, function(tags)
            local mat = 'e2rf_' .. (opts.material and opts.material(tags) or 'road_major')
            if not byMat[mat] then byMat[mat] = newMesh(mat); list[#list + 1] = byMat[mat] end
            return byMat[mat]
          end, o, lift, uv, cuts)
          table.sort(list, function(p, q) return p.material < q.material end)
          stats.tris = stats.tris + item(list, n .. 's', nil, S.id)
          pause()
          if M.TUNE.streetKerbs and STREETS[S.base] and not S.link then
            local kerb, walk = newMesh('e2rf_' .. (opts.kerbKind or 'curb')), newMesh('e2rf_' .. (opts.islandKind or 'sidewalk'))
            M.streetEdges(plan, S, c, kerb, walk, o, lift)
            stats.tris = stats.tris + item({kerb, walk}, n .. 'k')
            pause()
          end
          if opts.markKind then
            local marks = newMesh('e2rf_' .. opts.markKind)
            M.markMesh(plan, S, c, marks, o)
            stats.markTris = stats.markTris + item({marks}, n .. 'm', true)
            pause()
          end
          local rail = newMesh(look.name)
          -- a stock look is a W-beam sheet: a concrete barrier gets a mesh of its own kind (else the same mesh)
          local jersey = look.stock and newMesh('e2rf_' .. M.RAIL.jerseyMaterial) or rail
          local paths = M.railPaths(plan, S, c, opts.deckDepth and opts.deckDepth(S.tags) or 1.5)
          -- and the rule rails at a drop (opts.dem: the generator's raw DEM)
          for _, path in ipairs(M.dropRails(plan, S, c, opts.dem, opts.deckDepth and opts.deckDepth(S.tags) or 1.5)) do paths[#paths + 1] = path end
          for _, path in ipairs(paths) do M.railMesh(plan, S, c, path, path.kind == 'jersey' and jersey or rail, o, lift, look) end
          c.railPaths = paths
          pause()
          -- the deck's material by its tags (generator ROADS.deckMaterial: material=*, bridge:structure=*)
          local deck = newMesh('e2rf_' .. (opts.deckMaterial and opts.deckMaterial(M.deckTags(S, c)) or M.DECK.material))
          M.deckMesh(plan, S, c, deck, o, lift, opts.deckDepth and opts.deckDepth(S.tags) or 1.5)
          stats.railTris = stats.railTris + item(jersey ~= rail and {rail, deck, jersey} or {rail, deck}, n .. 'f', nil, S.id)
          pause()
        end
      end
    end
  end
  local byMat = {}
  for _, pl in ipairs(plan.plates) do
    if pl.owned and pl.tris then
      local mat = 'e2rf_' .. (opts.material and opts.material(plateTags(pl)) or 'road_major')
      byMat[mat] = byMat[mat] or newMesh(mat)
      M.plateMesh(plan, pl, byMat[mat], o, lift, uv)
      plan.stats.plates = plan.stats.plates + 1
    end
  end
  local meshes = {}
  for _, m in pairs(byMat) do meshes[#meshes + 1] = m end
  table.sort(meshes, function(a, b) return a.material < b.material end)
  stats.tris = stats.tris + item(meshes, 'plates')
  -- phase 3: the junctions' islands, and their own markings (lines through, guidance lines)
  local isl = {kerb = newMesh('e2rf_' .. (opts.kerbKind or 'curb')), top = newMesh('e2rf_' .. (opts.islandKind or 'sidewalk')),
    grass = newMesh('e2rf_' .. (opts.grassKind or 'ground'))}
  local jm = opts.markKind and newMesh('e2rf_' .. opts.markKind)
  for _, pl in ipairs(plan.plates) do
    if pl.owned and pl.junction then
      M.islandMesh(plan, pl.junction, isl, o, lift)
      if M.TUNE.streetKerbs and pl.junction.kind ~= 'roundabout' then
        local street = false
        for _, m in ipairs(pl.junction.mouths) do
          local S = plan.strokes[m.stroke]
          if S and S.swept and STREETS[S.base] and not S.link then street = true end
        end
        if street then M.junctionKerbs(plan, pl.junction, isl.kerb, o, lift) end
      end
      if jm then M.junctionMarks(plan, pl.junction, jm, o) end
    end
  end
  stats.tris = stats.tris + item({isl.kerb, isl.top, isl.grass}, 'islands')
  if jm then stats.markTris = stats.markTris + item({jm}, 'jmarks', true) end
  return out, materials, stats
end

-- ---------------------------------------------------------------------------------------------------
-- Phase 3: the junctions' islands, their own markings, their navgraph
-- ---------------------------------------------------------------------------------------------------
-- triangles of a simple polygon ({{x, y, ...}}, either orientation) by ear clipping: index triples
local function earClip(ring)
  local n = #ring
  local idx = {}
  local area = 0
  for i = 1, n do
    local p, q = ring[i], ring[i % n + 1]
    area = area + p[1] * q[2] - q[1] * p[2]
  end
  for i = 1, n do idx[i] = area >= 0 and i or n + 1 - i end
  local out = {}
  local function cross(a, b, c) return (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]) end
  local guard = 0
  while #idx > 3 and guard < 4 * n * n do
    guard = guard + 1
    local found = false
    for k = 1, #idx do
      local ia, ib, ic = idx[(k - 2) % #idx + 1], idx[k], idx[k % #idx + 1]
      local a, b, c = ring[ia], ring[ib], ring[ic]
      if cross(a, b, c) > 1e-9 then
        local inside = false
        for _, j in ipairs(idx) do
          if j ~= ia and j ~= ib and j ~= ic then
            local p = ring[j]
            if cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0 then inside = true; break end
          end
        end
        if not inside then
          out[#out + 1] = {ia, ib, ic}
          table.remove(idx, k)
          found = true
          break
        end
      end
    end
    if not found then break end
  end
  if #idx == 3 then out[#out + 1] = {idx[1], idx[2], idx[3]} end
  return out
end
M.earClip = earClip

-- raised islands (kerb sides, a paved or grass top) on the junction surface
function M.islandMesh(plan, J, meshes, o, lift)
  for _, isl in ipairs(J.islands) do
    local ring = isl.ring
    local n = #ring
    if n >= 3 then
      local cx, cy = 0, 0
      for _, p in ipairs(ring) do cx, cy = cx + p[1], cy + p[2] end
      cx, cy = cx / n, cy / n
      local top = isl.kind == 'grass' and meshes.grass or meshes.top
      local h = isl.height
      local zt = -math.huge
      for _, p in ipairs(ring) do zt = max(zt, p[3]) end
      local along = 0
      for k = 1, n do
        local p, q = ring[k], ring[k % n + 1]
        local l = sqrt((q[1] - p[1]) ^ 2 + (q[2] - p[2]) ^ 2)
        local mx, my = (p[1] + q[1]) / 2 - cx, (p[2] + q[2]) / 2 - cy
        local A, B = {p[1], p[2], p[3] + lift, along, 0}, {q[1], q[2], q[3] + lift, along + l, 0}
        local C, D = {q[1], q[2], q[3] + lift + h, along + l, h}, {p[1], p[2], p[3] + lift + h, along, h}
        tri(meshes.kerb, o, A, B, C, attrUV, {mx, my, 0}); tri(meshes.kerb, o, A, C, D, attrUV, {mx, my, 0})
        along = along + l
      end
      for _, t in ipairs(earClip(ring)) do
        local a, b, c = ring[t[1]], ring[t[2]], ring[t[3]]
        tri(top, o, {a[1], a[2], a[3] + lift + h}, {b[1], b[2], b[3] + lift + h}, {c[1], c[2], c[3] + lift + h}, planUV(3), UP)
      end
    end
  end
end

-- a line along points ({x, y, zref}) of width w on the surface, dashed by s (dash {on, off}) or solid;
-- s0: the arc length of the first point on the dash phase (a stroke's s keeps the road's dashes)
local function polyLine(plan, mesh, o, pts, width, dash, s0)
  local hw = width * M.MARKS.spread / 2
  local lift = plan.lift + M.MARKS.lift
  local cum = {0}
  for k = 2, #pts do cum[k] = cum[k - 1] + sqrt((pts[k][1] - pts[k - 1][1]) ^ 2 + (pts[k][2] - pts[k - 1][2]) ^ 2) end
  local function at(d)
    for k = 1, #pts - 1 do
      if d <= cum[k + 1] + 1e-9 or k == #pts - 1 then
        local t = (d - cum[k]) / max(EPS, cum[k + 1] - cum[k])
        local p, q = pts[k], pts[k + 1]
        local dx, dy = q[1] - p[1], q[2] - p[2]
        local l = max(EPS, sqrt(dx * dx + dy * dy))
        return p[1] + dx * t, p[2] + dy * t, p[3] + (q[3] - p[3]) * t, dx / l, dy / l
      end
    end
  end
  local function piece(d0, d1)
    local n = max(1, math.ceil((d1 - d0) / M.MARKS.sub))
    local prev
    for i = 0, n do
      local d = d0 + (d1 - d0) * i / n
      local x, y, zr, tx, ty = at(d)
      local nx, ny = -ty * hw, tx * hw
      local v = (s0 + d) / M.MARKS.repeatV
      local L = {x + nx, y + ny, M.surfaceZ(plan, x + nx, y + ny, zr) + lift, 1, v}
      local R = {x - nx, y - ny, M.surfaceZ(plan, x - nx, y - ny, zr) + lift, 0, v}
      if prev then
        tri(mesh, o, prev[2], R, L, attrUV, UP); tri(mesh, o, prev[2], L, prev[1], attrUV, UP)
      end
      prev = {L, R}
    end
  end
  local total = cum[#cum]
  if total < .05 then return end
  if not dash then piece(0, total); return end
  local period = dash[1] + dash[2]
  local start = floor(s0 / period) * period - s0
  while start < total do
    local a, b = max(0, start), min(total, start + dash[1])
    if b - a > .02 then piece(a, b) end
    start = start + period
  end
end
M.polyLine = polyLine

local function wrap(a) return (a + math.pi) % (2 * math.pi) - math.pi end

-- the mouths of a junction with their strokes (swept only), cut s, travel headings in and out
local function mouthsOf(plan, J)
  local out = {}
  for _, m in ipairs(J.mouths) do
    local S = plan.strokes[m.stroke]
    local sc = m.s + m.dir * m.cut
    local st = S and S:at(sc)
    if S and S.swept and st then
      out[#out + 1] = {m = m, S = S, sc = sc, st = st,
        hIn = st.h + (m.dir < 0 and 0 or math.pi), hOut = st.h + (m.dir > 0 and 0 or math.pi),
        canIn = not (S.oneway and m.dir > 0), canOut = not (S.oneway and m.dir < 0)}
    end
  end
  return out
end

local function bezier(P0, P1, P2, P3, n)
  local out = {}
  for i = 0, n do
    local t = i / n
    local a, b, c, d = (1 - t) ^ 3, 3 * (1 - t) ^ 2 * t, 3 * (1 - t) * t * t, t ^ 3
    out[#out + 1] = {a * P0[1] + b * P1[1] + c * P2[1] + d * P3[1], a * P0[2] + b * P1[2] + c * P2[2] + d * P3[2],
      a * P0[3] + b * P1[3] + c * P2[3] + d * P3[3]}
  end
  return out
end

-- a point s, u of S (zref on its own section)
local function pointSU(S, s, u)
  local st = S:at(s)
  if not st then return nil end
  return {st.x - sin(st.h) * u, st.y + cos(st.h) * u, st.z + (u >= 0 and st.fl or st.fr) * abs(u)}
end

-- the junction's own markings: the major road's lines through a junction that is not signalled (the
-- edge line dashed across a side road's mouth), and the left-turn guidance lines through a signalled one
function M.junctionMarks(plan, J, mesh, o)
  local MK = M.MARKS
  local ms = mouthsOf(plan, J)
  if J.kind == 'roundabout' or J.kind == 'circular' then return end
  if not J.signals then
    local byStroke = {}
    for _, mm in ipairs(ms) do
      if mm.m.role == 'major' then
        byStroke[mm.S] = byStroke[mm.S] or {}
        table.insert(byStroke[mm.S], mm)
      end
    end
    for S, list in pairs(byStroke) do
      if #list == 2 then
        local a, b = list[1].sc, list[2].sc
        if a > b then a, b = b, a end
        local la, lb = M.linesAt(S, a), M.linesAt(S, b)
        for _, l in ipairs(la) do
          local m2
          for _, q in ipairs(lb) do if q.id == l.id then m2 = q end end
          if m2 and abs(m2.u - l.u) < 1.5 then
            local n = max(1, math.ceil((b - a) / 1))
            if l.id:find('^edge') then
              -- solid where nothing opens beside the edge, the mouth pattern where a side road opens
              local side = l.u >= 0 and 1 or -1
              local runs, cur = {}, nil
              for i = 0, n do
                local s = a + (b - a) * i / n
                local u = l.u + (m2.u - l.u) * i / n
                local p, q = pointSU(S, s, u), pointSU(S, s, u + side * 1.2)
                if p and q then
                  local open = #M.surfacesAt(plan, q[1], q[2], 0) > 0
                  if not cur or cur.open ~= open then cur = {open = open, pts = cur and {cur.pts[#cur.pts]} or {}, s0 = s}; runs[#runs + 1] = cur end
                  cur.pts[#cur.pts + 1] = p
                end
              end
              for _, r in ipairs(runs) do
                if #r.pts >= 2 then polyLine(plan, mesh, o, r.pts, l.width, r.open and MK.mouthEdge.dash or nil, r.s0) end
              end
            else
              local pts = {}
              for i = 0, n do
                local s = a + (b - a) * i / n
                pts[#pts + 1] = pointSU(S, s, l.u + (m2.u - l.u) * i / n)
              end
              if #pts == n + 1 then polyLine(plan, mesh, o, pts, l.width, l.dash, a) end
            end
          end
        end
      end
    end
    return
  end
  -- signalled: the left-turners' inner edge from the stop line's road to the road turned into
  for _, A in ipairs(ms) do
    if A.canIn then
      for _, B in ipairs(ms) do
        if B ~= A and B.canOut and not (B.S == A.S and B.m.dir == A.m.dir) then
          local d = wrap(B.hOut - A.hIn)
          if d > math.rad(45) and d < math.rad(135) then
            local sgA, sgB = -A.m.dir, B.m.dir
            local _, uB_A = M.inboundLanes(A.S, A.sc, sgA)
            local uA0 = sgA > 0 and uB_A or select(1, M.inboundLanes(A.S, A.sc, sgA))
            local blkB = M.laneBlock(B.S, B.sc)
            local uB0
            if B.S.oneway then uB0 = sgB > 0 and blkB.left or blkB.right
            else uB0 = (blkB.both > 0 or blkB.f + blkB.b < 2) and (blkB.left + blkB.right) / 2 or blkB.left - blkB.b * blkB.w end
            local P0, P3 = pointSU(A.S, A.sc, uA0), pointSU(B.S, B.sc, uB0)
            if P0 and P3 then
              local chord = sqrt((P3[1] - P0[1]) ^ 2 + (P3[2] - P0[2]) ^ 2)
              if chord >= 15 then
                local k = chord * .4
                local P1 = {P0[1] + cos(A.hIn) * k, P0[2] + sin(A.hIn) * k, P0[3]}
                local P2 = {P3[1] - cos(B.hOut) * k, P3[2] - sin(B.hOut) * k, P3[3]}
                polyLine(plan, mesh, o, bezier(P0, P1, P2, P3, max(8, math.ceil(chord / 1))), MK.guidance.width, MK.guidance.dash, 0)
              end
            end
          end
        end
      end
    end
  end
end

-- one path per movement through each junction near the rect (generator navSegment format): from the
-- inbound mouth's knot through the turning lane's centre to the outbound mouth's knot; v2 roads meet
-- the junction at its nodes, linked to every swept mouth
function M.junctionNav(plan, rect, margin, out)
  local lift = plan.lift
  for _, pl in ipairs(plan.plates) do
    local J = pl.junction
    if J and J.kind ~= 'roundabout' and J.kind ~= 'circular' and J.x >= rect[1] - margin and J.x <= rect[3] + margin
        and J.y >= rect[2] - margin and J.y <= rect[4] + margin then
      local ms = mouthsOf(plan, J)
      local function knot(mm) return 'jm:' .. J.id .. ':' .. mm.m.stroke .. ':' .. mm.m.dir end
      local function speedOf(S)
        local t = tostring(S.tags.maxspeed or ''):lower()
        local v = number(t, SPEED[S.base] or 50)
        if t == 'none' then v = 130 end
        return v / 3.6
      end
      local allowed = {}
      for _, ap in ipairs(J.approaches) do
        local set = {}
        local any = false
        for t in (ap.turns .. '|'):gmatch('([^|]*)|') do
          if t:find('left') then set.left, any = true, true end
          if t:find('right') then set.right, any = true, true end
          if t:find('through') or t == 'none' or t == '' then set.through, any = true, true end
        end
        if ap.turns ~= '' and any then allowed[ap.stroke .. ':' .. ap.dir] = set end
      end
      for _, A in ipairs(ms) do
        if A.canIn then
          for _, B in ipairs(ms) do
            if B ~= A and B.canOut and not (B.S == A.S and B.m.dir == A.m.dir) then
              local d = wrap(B.hOut - A.hIn)
              local kind = abs(d) < math.rad(35) and 'through' or d > 0 and 'left' or 'right'
              local set = allowed[A.m.stroke .. ':' .. A.m.dir]
              if abs(d) < math.rad(170) and (not set or set[kind]) then
                local sgA, sgB = -A.m.dir, B.m.dir
                local _, _, centreA, nA = M.inboundLanes(A.S, A.sc, sgA)
                local _, _, centreB, nB = M.inboundLanes(B.S, B.sc, sgB)    -- the lanes of B's travel away from it
                local kA = kind == 'left' and 1 or kind == 'right' and nA or math.ceil(nA / 2)
                local kB = kind == 'left' and 1 or kind == 'right' and nB or math.ceil(nB / 2)
                local P0, P3 = pointSU(A.S, A.sc, centreA(kA)), pointSU(B.S, B.sc, centreB(kB))
                local K0, K3 = pointSU(A.S, A.sc, 0), pointSU(B.S, B.sc, 0)
                if P0 and P3 and K0 and K3 then
                  local chord = sqrt((P3[1] - P0[1]) ^ 2 + (P3[2] - P0[2]) ^ 2)
                  local k = chord * .4
                  local P1 = {P0[1] + cos(A.hIn) * k, P0[2] + sin(A.hIn) * k, P0[3]}
                  local P2 = {P3[1] - cos(B.hOut) * k, P3[2] - sin(B.hOut) * k, P3[3]}
                  local v = min(speedOf(A.S), speedOf(B.S))
                  if kind ~= 'through' then
                    local R = chord / max(.2, 2 * sin(abs(d) / 2))
                    v = min(v, sqrt(3 * R))
                  end
                  local blk = M.laneBlock(A.S, A.sc)
                  local seg = {'jx:' .. J.id .. ':' .. A.m.stroke .. ':' .. A.m.dir .. '>' .. B.m.stroke .. ':' .. B.m.dir,
                    knot(A), knot(B), '+', v, max(1.5, blk.w / 2), DRIVE[A.S.base] or .7}
                  local pts = {K0}
                  for _, p in ipairs(bezier(P0, P1, P2, P3, max(3, math.ceil(chord / 3)))) do pts[#pts + 1] = p end
                  pts[#pts + 1] = K3
                  local last
                  for _, p in ipairs(pts) do
                    if not last or (p[1] - last[1]) ^ 2 + (p[2] - last[2]) ^ 2 > .25 then
                      seg[#seg + 1], seg[#seg + 2], seg[#seg + 3] = p[1], p[2], M.surfaceZ(plan, p[1], p[2], p[3]) + lift
                      last = p
                    end
                  end
                  if #seg >= 13 then out[#out + 1] = seg end
                end
              end
            end
          end
        end
      end
      -- v2 roads (not swept) of the junction meet it at their node: linked to every swept mouth
      for _, m in ipairs(J.mouths) do
        local S = plan.strokes[m.stroke]
        if not (S and S.swept) then
          local node
          for _, n in ipairs(J.nodes) do if n == m.node then node = n end end
          if node then
            for _, B in ipairs(ms) do
              local K = pointSU(B.S, B.sc, 0)
              if K then
                for _, dirn in ipairs({1, -1}) do
                  if (dirn > 0 and B.canOut) or (dirn < 0 and B.canIn) then
                    -- the node end at the junction's centre (a one-node junction's node)
                    local a, b, pa, pb = node, knot(B), {J.x, J.y, J.z}, K
                    if dirn < 0 then a, b, pa, pb = b, a, pb, pa end
                    out[#out + 1] = {'jv:' .. J.id .. ':' .. a .. '>' .. b, a, b, '+', 30 / 3.6, 2, .7,
                      pa[1], pa[2], pa[3] + lift, pb[1], pb[2], pb[3] + lift}
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end

-- ---------------------------------------------------------------------------------------------------
-- What the generator needs from the plan for its ground, verges, navgraph and spawn
-- ---------------------------------------------------------------------------------------------------

-- every drawn quad half ({poly (lifted), S, s, bridge, tunnel}) of the loaded chunks, and the plates' fans
function M.footprints(plan, fn)
  local lift = plan.lift
  local tick = pacer(plan)
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, c in ipairs(S.list) do
        for _, list in ipairs(c.ground and c.rows or NONE) do
          for k = 1, #list - 1 do
            local a, b = list[k], list[k + 1]
            tick()
            local sm = (a.s + b.s) / 2
            local bridge, tunnel = structureAt(S, sm, 'bridge') ~= nil, structureAt(S, sm, 'tunnel') ~= nil
            if a.eR + a.uM > 1e-4 or b.eR + b.uM > 1e-4 then
              local quad = {lifted(a.R, lift), lifted(b.R, lift), lifted(b.M, lift), lifted(a.M, lift)}
              for _, piece in ipairs(M.forkPieces(plan, S, hostOf(plan, S, a, b), quad)) do fn(piece, S, a.s, bridge, tunnel, c) end
            end
            if a.eL - a.uM > 1e-4 or b.eL - b.uM > 1e-4 then
              local quad = {lifted(a.M, lift), lifted(b.M, lift), lifted(b.L, lift), lifted(a.L, lift)}
              for _, piece in ipairs(M.forkPieces(plan, S, hostOf(plan, S, a, b), quad)) do fn(piece, S, a.s, bridge, tunnel, c) end
            end
          end
        end
      end
    end
  end
  for _, pl in ipairs(plan.plates) do
    for _, t in ipairs(pl.tris or NONE) do fn({lifted(t[1], lift), lifted(t[2], lift), lifted(t[3], lift)}, nil, nil, false, false) end
  end
end

-- press(x, y, z): the generator's lattice press; samples every TUNE.press m and across each row. skip(S): strokes
-- that cut the ground instead (conformal.lua) are not pressed
local ACROSS = {0, .25, .5, .75, 1}
function M.press(plan, press, margin, skip)
  margin = margin or M.TUNE.margin
  local lift = plan.lift
  local tick = pacer(plan)
  local ax, ay, az, bx, by, bz = {}, {}, {}, {}, {}, {}
  for _, S in pairs(plan.strokes) do
    if S.swept and not (skip and skip(S)) then
      for _, c in ipairs(S.list) do
        for _, list in ipairs(c.ground and c.rows or NONE) do
          for k = 1, #list - 1 do
            local a, b = list[k], list[k + 1]
            tick()
            if not structureAt(S, (a.s + b.s) / 2, 'tunnel') then
              local n = max(1, math.ceil((b.s - a.s) / M.TUNE.press))
              -- across the row from R to L through M, on both rows' vertices (once per row pair)
              for q = 1, 5 do
                local f = ACROSS[q]
                local u = -a.eR + (a.eL + a.eR) * f
                ax[q], ay[q], az[q] = a.x + a.nx * u, a.y + a.ny * u, M.rowZ(a, u)
                u = -b.eR + (b.eL + b.eR) * f
                bx[q], by[q], bz[q] = b.x + b.nx * u, b.y + b.ny * u, M.rowZ(b, u)
              end
              for i = 0, n do
                local t = i / n
                for q = 1, 5 do
                  press(ax[q] + (bx[q] - ax[q]) * t, ay[q] + (by[q] - ay[q]) * t, az[q] + (bz[q] - az[q]) * t + lift - margin)
                end
              end
            end
          end
        end
      end
    end
  end
  for _, pl in ipairs(plan.plates) do
    for _, isl in ipairs(pl.junction and pl.junction.islands or NONE) do
      for _, p in ipairs(isl.ring) do press(p[1], p[2], p[3] + lift - margin) end
    end
    for _, t in ipairs(pl.tris or NONE) do
      local c = {(t[1][1] + t[2][1] + t[3][1]) / 3, (t[1][2] + t[2][2] + t[3][2]) / 3, (t[1][3] + t[2][3] + t[3][3]) / 3}
      for _, p in ipairs({t[1], t[2], t[3], c}) do press(p[1], p[2], p[3] + lift - margin) end
      for i = 1, 3 do
        local p, q = t[i], t[i % 3 + 1]
        press((p[1] + q[1]) / 2, (p[2] + q[2]) / 2, (p[3] + q[3]) / 2 + lift - margin)
      end
    end
  end
end

-- pull segments for the generator's ground pull and its road index: {a, b, width, surface, tags}
function M.pullSegments(plan)
  local out = {}
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, c in ipairs(S.list) do
        for i = 1, c.n - 1 do
          local function centre(k)
            local u = (c.eL[k] - c.eR[k]) / 2
            local nx, ny = -sin(c.h[k]), cos(c.h[k])
            return {c.x[k] + nx * u, c.y[k] + ny * u, c.z[k] + (u >= 0 and c.fl[k] or c.fr[k]) * abs(u)}
          end
          local sm = (c.s[i] + c.s[i + 1]) / 2
          local inside = sm >= S.D0 and sm <= S.D1
          local bridge, tunnel = structureAt(S, sm, 'bridge'), structureAt(S, sm, 'tunnel')
          out[#out + 1] = {a = centre(i), b = centre(i + 1), width = max(.5, (c.eL[i] + c.eR[i] + c.eL[i + 1] + c.eR[i + 1]) / 2),
            surface = inside and not bridge and not tunnel, tags = S.tags, S = S}
        end
      end
    end
  end
  return out
end

-- verges: {inner a, outer a, outer b, inner b} per row interval and side (outer z from ground later)
function M.verges(plan, shoulder, sink, fn)
  local lift = plan.lift
  local tick = pacer(plan)
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, c in ipairs(S.list) do
        for _, list in ipairs(c.ground and c.rows or NONE) do
          for k = 1, #list - 1 do
            local a, b = list[k], list[k + 1]
            tick()
            local sm = (a.s + b.s) / 2
            if not structureAt(S, sm, 'bridge') and not structureAt(S, sm, 'tunnel') then
              -- conformal ground: a band next to a bridge or a tunnel has an open end there (endA, endB), closed by a wall
              local endA, endB
              if plan.conformal and plan.conformal[S.base] then
                if k > 1 then
                  local s0 = (list[k - 1].s + a.s) / 2
                  endA = structureAt(S, s0, 'bridge') ~= nil or structureAt(S, s0, 'tunnel') ~= nil
                end
                if k < #list - 1 then
                  local s1 = (b.s + list[k + 2].s) / 2
                  endB = structureAt(S, s1, 'bridge') ~= nil or structureAt(S, s1, 'tunnel') ~= nil
                end
              end
              for _, side in ipairs({1, -1}) do
                local ea, eb = side > 0 and a.eL or -a.eR, side > 0 and b.eL or -b.eR
                local pa, pb = side > 0 and a.L or a.R, side > 0 and b.L or b.R
                local oa, ob = ea + side * shoulder, eb + side * shoulder
                local quad = {{pa[1], pa[2], pa[3] + lift - sink},
                  {a.x + a.nx * oa, a.y + a.ny * oa, pa[3]}, {b.x + b.nx * ob, b.y + b.ny * ob, pb[3]},
                  {pb[1], pb[2], pb[3] + lift - sink}}
                quad[2][9], quad[3][9] = 1, 1
                quad.endA, quad.endB = endA, endB
                fn(quad, S, sm)
              end
            end
          end
        end
      end
    end
  end
  -- plates: verges off the ring's edges between mouths; a junction's off its kerb edges
  for _, pl in ipairs(plan.plates) do
    local J = pl.junction
    for _, e in ipairs(J and J.kerbs or NONE) do
      tick()
      local p, q = J.verts[e[1]], J.verts[e[2]]
      local dx, dy = q[1] - p[1], q[2] - p[2]
      local l = max(EPS, sqrt(dx * dx + dy * dy))
      local ox, oy = dy / l * shoulder, -dx / l * shoulder
      local quad = {{p[1], p[2], p[3] + lift - sink}, {p[1] + ox, p[2] + oy, p[3]}, {q[1] + ox, q[2] + oy, q[3]}, {q[1], q[2], q[3] + lift - sink}}
      quad[2][9], quad[3][9] = 1, 1
      fn(quad, nil, nil)
    end
    for _, t in ipairs(J and NONE or pl.tris or NONE) do
      local p, q, cn = t[2], t[3], t[1]
      local mx, my = (p[1] + q[1]) / 2 - cn[1], (p[2] + q[2]) / 2 - cn[2]
      local l = max(EPS, sqrt(mx * mx + my * my))
      local ox, oy = mx / l * shoulder, my / l * shoulder
      local quad = {{p[1], p[2], p[3] + lift - sink}, {p[1] + ox, p[2] + oy, p[3]}, {q[1] + ox, q[2] + oy, q[3]}, {q[1], q[2], q[3] + lift - sink}}
      quad[2][9], quad[3][9] = 1, 1
      fn(quad, nil, nil)
    end
  end
end

-- navgraph segments (generator navSegment format) of the chunks reaching the rect: split at the chunk's
-- nodes so other roads meeting there link to it; a lane-attached slip road starts (ends) at its node
function M.nav(plan, rect, margin)
  local out = {}
  local lift = plan.lift
  for _, S in pairs(plan.strokes) do
    if S.swept then
      local A = S.attach
      local from, to = -math.huge, math.huge
      local endKnot
      if A and S.hostEvent then
        -- the slip road's s where its host's added lane reaches full width: the AI changes lanes there
        local e = S.hostEvent
        local sv = A.lane0 + (e.parallel - A.hostLane0) * (A.lane1 - A.lane0) / max(EPS, A.hostLane1 - A.hostLane0)
        local hst = S.host and S.host:at(e.parallel)   -- a version-1 pack's records name no host
        if hst then
          endKnot = {id = 'lane:' .. S.id, x = hst.x, y = hst.y, z = hst.z}
          if A.kind == 'exit' then from = sv else to = sv end
        end
      elseif A then
        local ns
        for _, c in ipairs(S.list) do for _, nd in ipairs(c.nodes) do if nd.node == A.node then ns = nd.s end end end
        if ns then if A.kind == 'exit' then from = ns else to = ns end end
      end
      -- a host: a knot at each of its swept slip roads' merge points
      local laneKnots = {}
      for _, e in ipairs(S.events) do
        local B = plan.strokes[e.branch]
        if B and B.swept and B.hostEvent == e then laneKnots[#laneKnots + 1] = {s = e.parallel, id = 'lane:' .. e.branch} end
      end
      local speedText = tostring(S.tags.maxspeed or ''):lower()
      local speed = number(speedText, SPEED[S.base] or 50)
      if speedText == 'none' then speed = 130 end
      for _, c in ipairs(S.list) do
        local x0, y0, x1, y1 = math.huge, math.huge, -math.huge, -math.huge
        for i = 1, c.n do x0, y0, x1, y1 = min(x0, c.x[i]), min(y0, c.y[i]), max(x1, c.x[i]), max(y1, c.y[i]) end
        if x1 >= rect[1] - margin and x0 <= rect[3] + margin and y1 >= rect[2] - margin and y0 <= rect[4] + margin then
          -- knots: the chunk's ends and nodes inside [from, to]
          local cuts = {{s = max(c.s[1], from), id = nil}, {s = min(c.s[c.n], to), id = nil}}
          if cuts[2].s - cuts[1].s > .5 then
            local function nodeId(s)
              for _, nd in ipairs(c.nodes) do if abs(nd.s - s) < .01 then return nd.node end end
            end
            cuts[1].id = nodeId(cuts[1].s) or (S.id .. '@' .. c.index)
            cuts[2].id = nodeId(cuts[2].s) or (S.id .. '@' .. (c.index + 1))
            if A and A.kind == 'exit' and abs(cuts[1].s - from) < 1e-6 then cuts[1].id, cuts[1].pos = endKnot and endKnot.id or A.node, endKnot end
            if A and A.kind == 'entry' and abs(cuts[2].s - to) < 1e-6 then cuts[2].id, cuts[2].pos = endKnot and endKnot.id or A.node, endKnot end
            for _, nd in ipairs(c.nodes) do
              if nd.s > cuts[1].s + .5 and nd.s < cuts[2].s - .5 then cuts[#cuts + 1] = {s = nd.s, id = nd.node} end
            end
            for _, kn in ipairs(laneKnots) do
              if kn.s > cuts[1].s + .5 and kn.s < cuts[2].s - .5 then cuts[#cuts + 1] = {s = kn.s, id = kn.id} end
            end
            -- phase 3: knots at the junction mouths; nothing inside a junction (its movements stand in)
            for _, kn in ipairs(S.navKnots or NONE) do
              if kn.s > cuts[1].s + .5 and kn.s < cuts[2].s - .5 then cuts[#cuts + 1] = {s = kn.s, id = kn.id}
              elseif abs(kn.s - cuts[1].s) <= .5 then cuts[1].id = kn.id
              elseif abs(kn.s - cuts[2].s) <= .5 then cuts[2].id = kn.id end
            end
            table.sort(cuts, function(p, q) return p.s < q.s end)
            for k = 1, #cuts - 1 do
              local a, b = cuts[k], cuts[k + 1]
              if insideAny(S.navGaps or NONE, (a.s + b.s) / 2) then goto next_cut end
              local run = S:lanesAt((a.s + b.s) / 2)
              local lanes = S.oneway and string.rep('+', min(4, max(1, run.f + run.both)))
                or string.rep('-', min(4, max(1, run.b + run.both))) .. string.rep('+', min(4, max(1, run.f + run.both)))
              local mid = sample(c, (a.s + b.s) / 2)
              local seg = {S.id .. ':' .. c.index .. ':' .. k, a.id, b.id, lanes, speed / 3.6, (mid.eL + mid.eR) / 2, DRIVE[S.base] or .7}
              local function push(s, pos)
                local st = pos or sample(c, s)
                local n = #seg
                if n >= 10 and (seg[n - 2] - st.x) ^ 2 + (seg[n - 1] - st.y) ^ 2 < .25 then return end
                seg[n + 1], seg[n + 2], seg[n + 3] = st.x, st.y, st.z + lift
              end
              push(a.s, a.pos)
              for i = 1, c.n do if c.s[i] > a.s + .5 and c.s[i] < b.s - .5 then push(c.s[i]) end end
              push(b.s, b.pos)
              if #seg >= 13 then out[#out + 1] = seg end
              ::next_cut::
            end
          end
        end
      end
    end
  end
  M.junctionNav(plan, rect, margin, out)
  return out
end

-- (x, y) within extent + margin of any swept centre line (the corridor a v2 barrier line must leave)
function M.inCorridor(plan, x, y, margin)
  return M.surfaceWhere(plan, x, y, margin) ~= nil
end

-- a v2 line's runs off the swept roads: guard rails leave the furniture corridor (the pack's rails
-- stand in), other barriers only the drawn surface
function M.cutLine(plan, points, kind)
  local margin = kind == 'guard_rail' and 4.5 or .5
  local runs, run = {}, {}
  local function close() if #run >= 2 then runs[#runs + 1] = run end; run = {} end
  for k = 1, #points do
    local p = points[k]
    if k > 1 then
      local q = points[k - 1]
      local l = sqrt((p[1] - q[1]) ^ 2 + (p[2] - q[2]) ^ 2)
      local n = max(1, math.ceil(l / 2))
      for i = 1, n - 1 do
        local t = i / n
        local x, y, z = q[1] + (p[1] - q[1]) * t, q[2] + (p[2] - q[2]) * t, (q[3] or 0) + ((p[3] or 0) - (q[3] or 0)) * t
        if M.inCorridor(plan, x, y, margin) then close() else run[#run + 1] = {x, y, z} end
      end
    end
    if M.inCorridor(plan, p[1], p[2], margin) then close() else run[#run + 1] = {p[1], p[2], p[3]} end
  end
  close()
  return runs
end

-- the nearest drivable point to target inside rect: {pos, yaw}, distance
function M.spawn(plan, rect, target)
  local best, bd = nil, math.huge
  for _, S in pairs(plan.strokes) do
    if S.swept then
      for _, c in ipairs(S.list) do
        for _, list in ipairs(c.ground and c.rows or NONE) do
          for _, r in ipairs(list) do
            if inRect(rect, r.x, r.y) and not structureAt(S, r.s, 'tunnel') then
              local d = (r.x - target[1]) ^ 2 + (r.y - target[2]) ^ 2
              if d < bd then
                bd = d
                best = {pos = {r.x, r.y, r.z + plan.lift}, yaw = atan2(cos(r.h), sin(r.h))}
              end
            end
          end
        end
      end
    end
  end
  return best, bd
end

return M
