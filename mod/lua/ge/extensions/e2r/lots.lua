-- Paint for car parks and bus stops (generator.lua build): the white lines between the bays of a car
-- park, on the same bay lattice the parked cars use, and the zig-zag band that marks a bus stop on the
-- carriageway beside its shelter. Decals in the markings' material: two triangles a line or a leg,
-- no collision. ctx.emit(poly) takes a quad with the markings' UVs (u across the painted width,
-- v along it, generator.lua paintQuad).
local M = {}
local sqrt = math.sqrt

-- Car park bays: 2.7 m wide, 5.5 m rows (the parked cars' lattice), lines 12 cm wide over the
-- paved drape (2.5 cm) plus the markings' lift; an aisle every third row.
M.BAY = {width = 2.7, depth = 5.5, line = .12, lift = .025 + .008, aisle = 3, bend = .02}
-- A bus stop: a zig-zag along the kerb on the shelter's side, `length` m centred on the stop, its legs
-- `leg` m along between `inner` and `outer` m in from the carriageway's edge.
M.STOP = {length = 18, leg = 1.5, inner = .25, outer = 1.25, line = .12, lift = .008}
-- the markings' sheet (generator.lua DECAL: markSpread, markRepeat)
M.SPREAD, M.REPEAT = 1.45, 6.4

local function quad(emit, ax, ay, az, bx, by, bz, width)
  local dx, dy = bx - ax, by - ay
  local l = sqrt(dx * dx + dy * dy)
  if l < 1e-6 then return 0 end
  local nx, ny = -dy / l * width * M.SPREAD / 2, dx / l * width * M.SPREAD / 2
  emit({{ax + nx, ay + ny, az, 0, 0}, {ax - nx, ay - ny, az, 1, 0},
    {bx - nx, by - ny, bz, 1, l / M.REPEAT}, {bx + nx, by + ny, bz, 0, l / M.REPEAT}})
  return 2
end

-- The separators of a car park's bays. ctx: frame (generator footprintFrame of its outer ring), rings,
-- rect (a line is drawn by the tile holding its middle), inside(rings, x, y), blocked(x, y) (a
-- building), ground(x, y), emit. A bay counts where its four corners lie in the car park and neither
-- they nor its centre stand in a building. Returns the triangles.
function M.bays(ctx)
  local f, cfg = ctx.frame, M.BAY
  local seen, tris, row = {}, 0, 0
  local function point(u, v) return u * f.ax + v * f.bx, u * f.ay + v * f.by end
  for v = f.v0 + cfg.depth / 2, f.v1 - cfg.depth / 2 + .05, cfg.depth do
    if row % cfg.aisle ~= 1 then
      local k = 0
      for u = f.u0 + cfg.width / 2, f.u1 - cfg.width / 2 + .05, cfg.width do
        local fits = true
        for _, c in ipairs({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}) do
          local x, y = point(u + c[1] * cfg.width / 2, v + c[2] * cfg.depth / 2)
          if not ctx.inside(ctx.rings, x, y) or ctx.blocked(x, y) then fits = false; break end
        end
        local cx, cy = point(u, v)
        if fits and not ctx.blocked(cx, cy) then
          -- the bay's two side lines; a line between two bays is drawn once
          for _, side in ipairs({k, k + 1}) do
            if not seen[side] then
              seen[side] = true
              local su = f.u0 + side * cfg.width
              local ax, ay = point(su, v - cfg.depth / 2 + .1)
              local bx, by = point(su, v + cfg.depth / 2 - .1)
              local mx, my = (ax + bx) / 2, (ay + by) / 2
              local r = ctx.rect
              if mx >= r[1] and mx < r[3] and my >= r[2] and my < r[4] then
                -- one quad, or two halves where the ground bends under it by more than `bend` m
                local za, zm, zb = ctx.ground(ax, ay) + cfg.lift, ctx.ground(mx, my) + cfg.lift, ctx.ground(bx, by) + cfg.lift
                if math.abs(zm - (za + zb) / 2) <= cfg.bend then
                  tris = tris + quad(ctx.emit, ax, ay, za, bx, by, zb, cfg.line)
                else
                  tris = tris + quad(ctx.emit, ax, ay, za, mx, my, zm, cfg.line) + quad(ctx.emit, mx, my, zm, bx, by, zb, cfg.line)
                end
              end
            end
          end
        end
        k = k + 1
      end
    end
    seen = {}
    row = row + 1
  end
  return tris
end

-- A bus stop's zig-zag. ctx: points (the strip's centreline with z), k (its segment at the stop), x, y
-- (the stop's foot on it), side (1 left of the segment's direction, -1 right), hw (half width), lift
-- (the road's), rect, emit. Kept on the segment's straight run; nothing on a segment shorter than the
-- band. Returns the triangles.
function M.stop(ctx)
  local cfg, pts = M.STOP, ctx.points
  local a, b = pts[ctx.k], pts[ctx.k + 1]
  local dx, dy = b[1] - a[1], b[2] - a[2]
  local len = sqrt(dx * dx + dy * dy)
  if len < cfg.length + 2 or ctx.hw < cfg.outer + 1.5 then return 0 end
  local r = ctx.rect
  if ctx.x < r[1] or ctx.x >= r[3] or ctx.y < r[2] or ctx.y >= r[4] then return 0 end
  local ux, uy = dx / len, dy / len
  local nx, ny = -uy * ctx.side, ux * ctx.side
  local at = ((ctx.x - a[1]) * ux + (ctx.y - a[2]) * uy)
  at = math.max(cfg.length / 2 + 1, math.min(len - cfg.length / 2 - 1, at))
  local function point(s, inward)
    local t = s / len
    local off = ctx.hw - inward
    return a[1] + ux * s + nx * off, a[2] + uy * s + ny * off,
      (a[3] or 0) + ((b[3] or 0) - (a[3] or 0)) * t + ctx.lift + cfg.lift
  end
  local tris = 0
  local legs = math.floor(cfg.length / cfg.leg + .5)
  local s0 = at - legs * cfg.leg / 2
  for n = 0, legs - 1 do
    local ax, ay, az = point(s0 + n * cfg.leg, n % 2 == 0 and cfg.inner or cfg.outer)
    local bx, by, bz = point(s0 + (n + 1) * cfg.leg, n % 2 == 0 and cfg.outer or cfg.inner)
    tris = tris + quad(ctx.emit, ax, ay, az, bx, by, bz, cfg.line)
  end
  return tris
end

return M
