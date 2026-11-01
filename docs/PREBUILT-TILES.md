# Prebuilt tiles: design

Task 7 of `docs/CLOUD-TASKS.md`. This document is a design only. It changes no code. Whether to build it is a design decision.

The question: can the exporter bake geometry into the pack that `generator.lua` now builds in Lua while the player drives, and by how much would that cut the stutter?

Short answer: partly. The ground lattice, the road layer, the far ring and the horizon depend only on pack data and a few switches, so they can be baked. Baking them removes roughly half of the generator's work in town and most of it in open country (measured below on synthetic tiles). The near scene (trees, street furniture, buildings) is the larger half in town. It depends on the quality preset, so it stays at run time. Baking does not touch the two other measured spikes, the collision rebuild and `createMesh`. The recommendation is at the end.

## 1. What runs while driving today

The path of one 500 m tile, with file references:

1. `native.priorities()` (`native.lua:592`) ranks the wanted set. The set covers near tiles within `nearM`, the far ring out to `farM`, horizon blocks of 1 km and 500 m out to 5 km, and vista blocks only when the camera is parked on high ground.
2. `native.update()` (`native.lua:1152`) resumes one build coroutine per step within `BUDGET_MS` (4 ms; 2 ms for far work, `FAR_BUDGET_MS`; 6 ms when a road is urgent).
3. `buildTile()` (`native.lua:995`) decodes the source cells round the tile (`cellformat.cell`, with a `keep` filter), reads the pack v3 road chunks (`loadRoads3`), and calls `generator.build()`.
4. `generator.build()` (`generator.lua:6596`) runs these stages, named by its own `mark()` calls: `segments`, `lattice`, `strips`, `plates`, `sidewalks`, `verges`, `marks`, `groundsea`, then the ground lattice is emitted and the **road layer is handed out** (`options.roadReady`, `generator.lua:8621`). After that come `areas`, `buildings`, `lines`, `decor` and `trees` (street furniture, yards, trees, grass and rocks), which make up the **scene layer**. Far and horizon tiles run the same ground shaping and emit a coarse ground (`generator.lua:8625`).
5. `world.lua` takes each layer (`native.manifest`) into `S.queue`. `stepQueue()` (`world.lua:1557`) spawns one item at a time. A `proc` item becomes a `ProceduralMesh` through `createMesh`. Everything else becomes a Forest item or a `TSStatic`.
6. `updateCollision()` (`world.lua:1676`) calls `be:reloadCollision()` through `collision.lua` when a loaded tile near the car lacks collision.

Measured baseline (from `CLOUD-TASKS.md`): `native` 5.9 ms mean and 81 ms worst; `collision` 0.07 ms mean and 142 ms worst; `queue` 1.4 ms mean and 52 ms worst; about 12 tiles built in 15 s at 120 km/h.

What a bake can change:

| Part | Changed by a bake? | Why |
| --- | --- | --- |
| `native` (generator) | Yes: the layers that are baked are no longer generated | This is the work a bake removes. |
| `queue` (`createMesh`) | No, and it gains the decode | `createMesh` takes Lua tables (`world.lua:647`). Baked bytes must still become the same tables. |
| `collision` | No | Same triangles, same `be:reloadCollision()`. |
| `tiles` | Slightly | Fewer manifests are held in Lua. |

The 81 ms worst `native` frame is a pause-granularity problem: long stretches without `pause()`, and a clock that is checked only every 8th call (`native.lua:574`). Task 1 (`generator-budget`) addresses it directly. A bake removes some of those stretches but does not fix the mechanism.

## 2. Measurements made for this document

No pack, DEM or game is available in the cloud session. Every number below comes from running this branch's `generator.lua` and `sweep.lua` unchanged under LuaJIT 2.1 (the `lupa` wheel, as `earthng/bench_native.py` on `master` does) on **synthetic cells**. The machine has 4 vCPUs (Xeon, 2.1 GHz), which is slower than a desktop CPU. Read the milliseconds as **ratios** between stages, not as in-game times. The scripts are not committed, because this task is document-only. They stub `FS`, `log` and `hptimer` the way `bench_native.py` does.

The synthetic cells are:

- `rural`: one `secondary` road across the tile, one farm track, and a farmland polygon.
- `town`: a street grid every 100 m, residential, `sidewalk=both`, lit, with four 14 × 12 m houses per block.
- `city`: the same grid every 70 m.
- The DEM is a smooth hill field on a 10 m grid (the grid this branch's packs carry, `build_country.py:43` `GRID_FINE`). `rural25` and `town25` use 25 m.
- Biome is off: `biome.lua` is not loaded, so there is no climate grid. There is no landcover and no pack v3 road chunks, so swept roads are not exercised. There are no forests.

Missing on purpose: forests and landcover, which feed the `trees` stage; swept motorways (`sweep.lua`); real DEM noise; and OSM's real road mix. Each of these makes real tiles heavier than the synthetic ones. Each is noted where it matters below.

### 2.1 Build time and output per 500 m tile

Times are a warm run on this machine. The variation between runs was about ±20 %.

| Case | Near tile, whole (ms) | Road layer handed out at (ms) | Far tile, no decor (ms) | Horizon tile, 100 m step (ms) | Road-layer triangles | Scene triangles + statics |
| --- | --- | --- | --- | --- | --- | --- |
| rural25 | 61 | 35 | 17 | 14 | 2 744 | 0 + 1 328 |
| rural10 | 84 | 55 | 28 | 16 | 6 944 | 0 + 1 328 |
| town25 | 539 | 116 | 121 | 50 | 6 020 | 10 290 + 5 273 |
| town10 | 441 | 96 | 99 | 39 | 10 220 | 10 290 + 5 273 |
| city10 | 604 | 245 | 140 | 42 | 14 763 | 20 104 + 4 403 |

Stage times for one run (ms; `areas` includes the lattice emission and the road hand-out, because no `mark()` sits between them):

| Case | segments | lattice | strips | plates | sidewalks | verges | marks | areas | buildings | trees |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rural10 | 3 | 8 | 6 | 0 | 0 | 13 | 3 | 40 | 0 | 22 |
| town10 | 3 | 2 | 13 | 8 | 21 | 39 | 1 | 102 | 23 | 273 |
| city10 | 25 | 3 | 36 | 7 | 23 | 51 | 1 | 15 | 190 | 619 |

In town the road layer is 20–40 % of a near tile, and the scene (`trees` plus `buildings`) is the rest. This matches the comment at the hand-out (`generator.lua:8603`): "the scene is ~3/4 of a near tile (... 230 of 820 ms is the road)". A far tile without decor costs about as much as a near road layer, because it runs the same ground shaping (`generator.lua:6600`: "Both shape the ground like near").

### 2.2 Baked bytes, decode time and garbage

A prototype codec encoded each layer's `proc` items in the scheme the pack's junction blocks already use (`roads3.lua:213`, EGJ1): LEB128 varints and zigzag deltas. Positions are in mm, normals in 1e-4, and UVs in 1/1024. Those are the weld quanta of the collector (`generator.lua:940-987`), so nothing that the collector distinguishes is lost. Corners are `{v, n, u}` index triples, and faces are corner indices. The decoder rebuilds exactly the tables `createMesh` takes. Sizes are after zlib level 6 (the pack writes deflated entries, `build_country.py` `_write_deflated`).

| Case | Road layer raw / deflated (kB) | of which ground lattice, generic (kB) | rest of road layer (kB) | Far (kB) | Horizon 500 m (kB) | Decode road layer to tables (ms) | Road-layer build garbage (MB) | Decode garbage (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rural10 | 105 / 24.0 | 18.3 | 5.7 | 4.9 | 2.5 | 7.7 | 13.5 | 3.6 |
| town10 | 173 / 45.4 | 37.0 | 7.4 | 13.8 | 2.4 | 14.7 | 23.2 | 5.9 |
| city10 | 277 / 84.7 | 71.7 | 12.4 | 26.6 | 3.1 | 34.9 | 35.5 | 9.3 |

Inflating one tile's road blob took 0.2–0.8 ms (Python's zlib).

The ground dominates the generic encoding. It does not have to be stored that way. Read in the code, the near ground is a regular lattice whose nodes are only moved in height by `press()` (`generator.lua:8127-8240`). Each cell is two triangles split along its `(x0, y0)-(x1, y1)` diagonal, with one material per triangle (`generator.lua:8551-8566`). The corner normal comes from heights at ±one step (`groundNormal`, `generator.lua:7003`), and the UV is position over `COVER_SCALE[kind]` (`generator.lua:995`, `1031-1036`). So the ground is fully described by the pressed node heights plus one material code per triangle. At a 10 m step that is 51 × 51 = 2 601 heights and 5 000 codes per tile. The smooth synthetic heights compress to under 1 kB, which is unrealistically low. For real ground the estimate rests on varint arithmetic: neighbouring heights of a 10 m DTM differ by 0.1–8 m, which is 2 bytes in zigzag mm, and noise barely deflates. That gives **5–8 kB per tile** for the lattice, about as many samples as the elevation grid the pack already carries for the same area (`GRID_FINE` = 10 m).

## 3. What to bake and what stays at run time

### 3.1 What each layer reads

Everything `generator.build()` reads before the road hand-out was checked (`generator.lua:6596-8625`, all `options.*`). The road layer reads:

- Pack data: the cells, the v3 road chunks, `look`, `seaLevel`, `geo`, `nature`, and `country`.
- The planet's climate grid through `biome.lua`. It sets the ground material outside mapped cover.
- Switches: `generator.SWEPT.on/mode` (on and `'fast'` by default in `generator.lua:1142`; the comment at `world.lua:146` still says "Off by default") and `generator.HEIGHTS.auto` (`world.lua:164`).
- The game install, through `decalsPresent()` (`generator.lua:472`), which gates the edge strips only.
- `options.center`, the landing point. It only picks the spawn.
- **No quality setting.** `maxTrees`, `propSpacing`, `detail`, `parkedCars` and the spacings are first read after the hand-out.

The scene reads the quality preset throughout. `detail` changes buildings in three bands (`BLD.bays`, `BLD.mass`, `generator.lua:3257-3269`). `forestSpacing` and `groundSpacing` set the lattice on which trees are placed (`generator.lua:9473-9479`, `9758`), so a lower preset is not a subset of a higher one. `maxTrees`, `maxGrass`, `maxRocks` and `propSpacing` cap and space the rest.

The scene also reads the road layer's intermediate state: `ground` (the pressed lattice, 60 uses after the hand-out), `strips` (15), `segmentBins` (9), `SW`, `plates`, `asphaltBins` and `pathShelf`. That is how trees and props keep off roads and sidewalk furniture follows the kerb.

### 3.2 The split

| Layer | Bake | Basis |
| --- | --- | --- |
| Horizon, 1 km blocks and 500 m quadrants, 100 m step | **Yes** | DEM and roads only, no quality input. 1.5–3 kB per 500 m. It costs 14–50 ms to generate, which is the best ratio of time saved to bytes. |
| Far ring ground and far road surfaces | **Yes** | It runs the full ground shaping. 17–140 ms to generate, 5–27 kB. |
| Far decor (buildings and trees when the car is slower than `farFast`) | No | It depends on `maxFarTrees`. Today it is a rebuild of the far tile with decor (`native.lua:1216`) and stays that. |
| Near ground lattice | **Yes**, as heights plus material codes | See 2.2. 5–8 kB. |
| Near road, sidewalk, junction plate, kerb, verge, bridge, marking and edge meshes | **Yes**, as generic meshes | 6–12 kB synthetic, more in real towns. Edge strips are dropped at load when `decalsPresent()` is false. They are decoration and already in `collision.QUIET`. |
| `navSegments` | **Yes** | Small; built with the road layer (`generator.lua:6768`). |
| Spawn for the landing | No | It depends on the landing point. The landing square keeps using the generator (section 7). |
| Near scene: buildings, fences, area drapes, water, furniture, yards, trees, grass, parked cars | No, in phase 1 and 2 | Quality-dependent. See phase 3. |
| Vista (beyond 5 km, only when parked high) | No | Elevation only and never scheduled while driving (`native.lua:24-28`). |
| Grass and rocks of the detail stream (`detail.lua`) | No | Placed round the car by preset. |

### 3.3 The coupling problem

If the near road layer is baked but the scene is still generated, the scene still needs `strips`, `segmentBins`, `plates` and `asphaltBins`. There are two ways to provide them:

1. **Rerun the road analysis without emitting meshes.** The stages are `segments`, `strips`, `plates` and `sidewalks` (town10: 45 ms of about 100 ms of road work), with `ground` coming from the baked lattice. This is simple, but it saves only the emission, `verges`, the press and `marks`: about half of the near road time.
2. **Bake a compact street context** with the strips' centre lines, half-widths, walk flags, plates and asphalt bins. This saves all of the near road time. However, it serialises the generator's internal state, and every change to it becomes a pack format change.

Option 1 is the honest first step. Option 2 should only follow if measurement shows the analysis stages matter after task 1.

## 4. Pack format

An additive manifest section, like `roads` and `roads.junctions` (`build_country.py:813-819`). A reader that does not know it ignores it, and a pack without it plays as today.

```
"baked": {
  "format": "earthng-baked-v1", "version": 1,
  "generator": "<hash of generator.lua, sweep.lua, biome.lua, cellformat.lua, roads3.lua>",
  "climate": "<hash of the planet climate grid>",
  "settings": {"swept": true, "sweptMode": "fast", "heights": true, "terrainStep": 10},
  "cellSize": 1000, "blockCells": 8,
  "blocks": {"<bi>_<bj>": {"dir": "/<ns>/baked/<bi>_<bj>/", "mask": "<16 hex>"}}
}
```

- **One file per 1 km source cell**, `<slot>.bin`, holding that cell's four 500 m tiles. Each tile has a near road layer (lattice and meshes, nav) and a far layer, plus the cell's horizon block and its four horizon quadrants. A small header gives the byte ranges, as `cellformat.block` does. The source blocks group 8 × 8 cells per file. For baked data that would be 5–9 MB per file on average and 12–22 MB in a city (section 5), read in one non-yielding `io.open():read('*a')` (`native.lua:109`), which is a stall in itself. Per-cell files are 75–350 kB.
- The encoding is section 2.2's codec plus the lattice codec, with one string table per file.
- Items keep the collector's item boundaries (`PIECE_TRIS` = 2 500, `MIXED_TRIS` = 4 000, `generator.lua:900`) and names, so `world.lua` spawns the same objects.
- Each item stores its triangle and vertex counts, so `native.entrySize` can budget without decoding.

The exporter runs the **game's own Lua** for the bake, not a port. That means `native.lua`'s `buildTile` under LuaJIT (lupa), over the region's own finished blocks, the way `bench_native.py` (on `master`) already drives it. It needs an explicit `native.bakeTile(key, wanted)` export instead of the upvalue walk in the bench. Then the cells, the `keep` margins, the road chunks and the options are by construction the ones the game would use. The bake runs after a region's cells are written, because a tile needs its neighbours' cells. It joins `_PART_CODE` (`build_country.py:854`) with the Lua hash, so a resumed build re-bakes only when the generator changed.

## 5. Pack size

Per 500 m tile, from section 2.2: lattice 5–8 kB (arithmetic, see above), rest of the road layer 5.7–12.4 kB, far 4.9–26.6 kB, and horizon about 2–3 kB per 500 m. A 1 km cell holds 4 tiles.

| Area type | Per km², synthetic basis | Per km², with ×2 on roads and far for real OSM density (assumption) |
| --- | --- | --- |
| Open country | ≈ 75 kB | ≈ 120 kB |
| Town | ≈ 120 kB | ≈ 205 kB |
| City | ≈ 190 kB | ≈ 350 kB |

For a state-sized pack of ≈ 47 700 km² (the size of this branch's region), with an **assumed** mix of 85 % open, 12 % town and 3 % city, this is **≈ 4.0–6.5 GB added**. Phase 1 alone (horizon and far) is ≈ 1.6–2.9 GB.

For scale, the only pack size in the repo is the README's 2.96 GB for a country of about 600 000 km², about 5 kB/km² on a 25 m grid. A 10 m elevation grid alone is 101 × 101 samples per km², about 20 kB at 2 bytes each. So a full bake makes a pack several times larger than its source data. That is the main cost of this design. Levers, in order of effect:

- The far layer's border runs at the fine step so its edge matches the near tile (`generator.lua:8625`). Encoding it like the lattice instead of generically would cut most of the far bytes.
- Phase 1 could bake the horizon only (about 8–12 kB/km², about 0.4–0.6 GB).

## 6. Time

### 6.1 In the game

On this machine, decoding a baked road layer into `createMesh` tables cost **4–35 ms per tile, against 35–245 ms to generate it**: about 7–14× less. It also left about 25 % of the garbage (3.6–9.3 MB against 13.5–35.5 MB per tile, excluding the cell decode that is no longer needed). Far and horizon decode are smaller still (2–27 kB of input).

The decode is a flat loop that pauses easily. It belongs inside `spawnItem` for one item (at most 2 500 triangles, a few ms here), as `binMeshes` already does for the old prebuilt pieces (`world.lua:289-331`). The tables then exist only for the one item being spawned, instead of a whole layer waiting in the queue.

Generator work per 500 m driven at Balanced (`nearM` 1000, `farM` 2400, `quality.lua`): the in-game count is about 12 tiles per 15 s. The code allows up to about 5 near and 14 far builds per 500 m: a front row of the near disc and the corridor, the far annulus front, and near tiles left behind becoming far (`native.lua:617`, `627`). The measured 12 says fewer far builds happen. With 5 near, 7 far and about 5 horizon blocks (1 km horizon blocks assumed at 2× the measured 500 m tile), on this machine:

| Case | Generator ms per 500 m | Removed by baking horizon + far | + near road layer (option 1: about half) | + near road layer (option 2: all) | Left: near scene |
| --- | --- | --- | --- | --- | --- |
| rural10 | ≈ 780 | 46 % | 63 % | 81 % | 19 % |
| town10 | ≈ 3 300 | 33 % | 40 % | 48 % | 52 % |
| city10 | ≈ 4 400 | 32 % | 46 % | 59 % | 41 % |

Real open country has forests, and the `trees` stage then dominates the scene, so the rural share is an upper bound. With the generator budget-bound while driving, its mean per-frame time should fall roughly in proportion. That would take the measured 5.9 ms mean to about 3–4 ms in town. **This is an estimate to verify in the game, not a measurement.** The worst `native` frame depends on task 1 more than on this design, and the `collision` (142 ms) and `queue` (52 ms) worst frames are unchanged.

### 6.2 In the exporter

The bake costs what the generator costs for the baked layers (road layer, far and horizon). On this machine that is ≈ 100, 235 and 430 ms per tile for rural, town and city. With the assumed 85/12/3 mix, the state's ≈ 190 000 tiles take ≈ 6.7 core-hours. With ×1.5–2 for real density, swept roads and encoding, that is 10–13 core-hours, about 40–50 minutes on 16 cores. The README reports 1.5 h on a 16-core PC for a whole country export. The bake needs LuaJIT in the exporter, which means the `lupa` wheel in `requirements-fast.txt`. `master`'s benches already depend on it.

## 7. Changes to `native.lua` and `world.lua`

### `native.lua`

1. **`validateRegion`** (`native.lua:207`): read `baked`, check `format`, `version`, the `generator` hash against a constant the planet ZIP carries, the `climate` hash, and `settings` against `generator.SWEPT` and `generator.HEIGHTS` at landing. Any mismatch leaves `r.baked = nil`, and the region plays as today.
2. **A cell cache**, like `loadRoads3`'s (`native.lua:916`): it reads the cell file, keeps the string (a string costs the collector nothing, `native.lua:377`), and holds ≤ 24 cells, which is at most ≈ 8.5 MB of strings at city density.
3. **`update()`**: when `active.region.baked` covers the wanted key and the layer, the work coroutine is `bakedTile(key, entry)` instead of `buildTile`.
   - **Horizon and far** become a manifest whose items reference byte ranges. There is no geometry decode and no `generator.build`.
   - **Near, option 1**: the road manifest comes from the bake and goes out through the same path as `roadReady` (`native.lua:1035`). The scene coroutine then calls `generator.build` in a new scene-only mode that takes `ground` from the baked lattice and skips the road emission.
   - Priorities, `AHEAD`, parking, eviction and `release`/`unload` stay as they are. They see the same layers and the same `st.ready`, `st.far` and `st.horizon` bookkeeping.
4. **Landing**: the required square (`active.startupPinned`) keeps `buildTile`. It needs `options.center` for the spawn, it runs once, and it is not the stutter.
5. **Fallback per tile**: any error in `bakedTile` (missing slot, bad bytes) is caught as `loadRoads3` catches its own. The tile is then marked and built by `buildTile`, with a log line.
6. **`setTune`** (`native.lua:1651`): a quality change no longer needs to drop baked road and far layers. Only scenes depend on it.
7. **`build()` in `generator.lua`** is at LuaJIT's limit of 200 locals (`CLOUD-TASKS.md` rule 6). The scene-only mode must live in a helper table (as `ROADS` and `OFFROAD` do), not in new locals.

### `world.lua`

1. **`spawnItem`** (`world.lua:629`): a `proc` item with `it.baked = {cell, first, last}` decodes there, with `pause()` every few hundred vertices, like `binMeshes`. Nothing else in the queue changes, and `pairsLeft`, the collision tracking and `collision.quiet` see the same meshes.
2. **`updateTiles`, `stepQueue` and `updateCollision`**: no change.
3. A console switch `extensions.e2r_world.bakedTiles(on)`, like `sweptRoads` (`world.lua:152`), for A/B in the same session. The profiler (`S.prof`) gains a `decode` part so the queue's share stays visible.

## 8. Migration path

Each step ships behind the switch and keeps the generator as the fallback for every layer. A pack without `baked`, or with a stale hash, behaves exactly as today.

0. **Measure on a real pack first.** Port `bench_native.py` to this branch, add the codec, and record per-tile triangles, bytes, decode ms and garbage for representative tiles: open country with forest, a village, a town centre, and a motorway interchange with swept roads. Replace this document's synthetic numbers. If real bytes per km² come out above about twice section 5, stop or narrow the scope.
1. **Horizon only, then far ground and far roads.** This needs no generator change, gives the best time-to-bytes ratio, and has a simple runtime path (no scene). Check in the game: horizon and far edges against near tiles (no step, no gap), and the far decor rebuild when slow.
2. **Near road layer, option 1.** Lattice codec, generic meshes, nav and the scene-only mode. Check in the game: kerbs, plates and marks against the generated version on the same route. Props and trees must stay off the roads, and the profiler should show the expected drop.
3. **Option 2 (street context) or a scene bake**, only if step 2's measurements justify it. A scene bake would mean one placement is accepted for all presets: placements baked at the top density with a rank, and lower presets taking a prefix. That is a visible change at low quality, not a refactor.

## 9. Risks

- **Version coupling.** Any change to `generator.lua`, `sweep.lua`, `biome.lua`, or the climate grid that alters baked output makes every baked pack fall back to the generator until it is rebuilt. This branch changes road code often (bundles 5–9 in `sweep.lua`). Baking before the road generator settles means re-exports, or packs that silently gain nothing. The hash makes this safe, but not free.
- **Pack size.** +4–6.5 GB for a state, several times the source data (section 5). That affects download and disk, and the README's build pre-check of about 24 GB per GB of source would need a new factor.
- **Determinism between LuaJIT builds.** The exporter's LuaJIT (lupa, often on another OS) and the game's may differ in the last bit of `sin`, `atan2` and the like (C library calls). With mm quantization almost all differences vanish. A value on a rounding edge can still move by 1 mm, which is a hairline where a baked tile meets a generated neighbour in a fallback. Test: build the same tile in both ways and compare after quantization.
- **The scene still needs road state** (3.3). Option 1 halves the near road saving. Option 2 freezes internal state into the format.
- **Unchanged spikes.** `be:reloadCollision` (142 ms) and `createMesh` (`queue`, 52 ms) are untouched. Players will still see those hitches. Tasks 1–3 and `collision-spread` address them.
- **Memory.** The cell cache adds up to about 8.5 MB of strings, while the generator's per-tile garbage of 13–35 MB disappears for baked layers. The net effect should be positive, but it is unmeasured in the game.
- **Untested inputs.** Swept roads, landcover, forests and real DEM noise were not in the synthetic bench. Step 0 exists for this.

## 10. Recommendation

**Do not start with a full bake.** The measured worst frames (81 ms `native`, 142 ms `collision`, 52 ms `queue`) come from pause granularity, the collision rebuild and `createMesh`. Tasks 1–3 and `collision-spread` act on those directly and cost no pack size. A bake reduces the mean generator load, not those peaks.

Do it in this order:

1. Land tasks 1–3 first.
2. Then do step 0, the real-pack measurement: two or three days with the bench that already exists on `master`.
3. If step 0 confirms section 5's bytes, bake **the horizon and the far ring** (step 1). It removes about a third of the generator's work while driving through town and close to half in open country (synthetic). It needs no generator refactor and adds a modest amount to the pack.
4. Bake the **near road layer** (step 2) only after the road generator has stopped changing every week, and only if the profiler after tasks 1–3 still shows `native` above its budget at speed.
5. Do not bake the scene unless one placement is accepted for all quality presets.

## 11. Phase 1 as built

Step 1 of section 8 (horizon and far ring) is implemented behind `extensions.e2r_world.bakedTiles(true)`, which is off by default. Step 0, the real-pack measurement, has not been done. The numbers below come from synthetic packs.

**Baked:** the far ring's tiles without decor (`far:<i>_<j>`), and the 5 km horizon's 1 km blocks and 500 m quadrants at the 100 m step (`h1000:`, `h500:`).

**Not baked:**
- the near tiles (road layer and scene);
- the far ring's decor, which is built whenever the car is slower than `farFast` (25 m/s);
- the vista.

All of these are generated as before. A far tile is read from the bake only while the car is above 25 m/s. Below that the ring wants decor, and the generator builds it as today.

**Where it lives:** the design in section 4 puts a `baked` section in the region manifest. The build does not. It writes a sidecar in the country's namespace, `baked/<region>/baked.json` plus one `<ci>_<cj>.bin` per source cell. This way a bake ships as its own small ZIP, the pack is not rewritten, and no path is mounted twice. A layer is stored in the cell of its south-west corner.

**When it is used:**
- The bake is checked once per landing. `baked.json` must match all of these:
  - the digest of the region manifest, so a re-exported pack drops an old bake;
  - the digests of the eight Lua files that shape these layers (`bake.CODE`);
  - the digests of every climate block the bake read;
  - the swept-roads switch and the height-pipeline switch.
- The two switches are checked again for every layer.
- A digest is FNV-1a over the bytes without CR, so LF and CRLF checkouts agree.
- Any mismatch, a missing cell file or a bad byte leaves that layer to the generator. `bakedTiles()` reports `used`, `missing`, `failed` and `off`.

**Codec (`bake.lua`):** a tagged tree of the manifest. Meshes are delta-coded varints:
- positions in mm and normals in 1e-4, the collector's own weld quanta;
- UVs in 1/4096 of a repeat;
- faces exact.

The decoded tables are the ones the generator would have handed to `world.lua`, so `world.lua`'s queue, `createMesh` and collision are unchanged. The design's lattice codec and the decode inside `spawnItem` are left for later.

**Measured** with `python earthng/bake_tiles_test.py --bench`: through `buildTile`, on a 4-vCPU cloud machine, so read the ratios, not the milliseconds.

| Pack | Layer | Generated (cells decoded and built) | Read from the bake |
| --- | --- | --- | --- |
| light (one road, fields) | far 500 m | 30 ms, 2.8 MB garbage | 1.1 ms, 0.5 MB |
| light | horizon 1 km | 45 ms | 0.9 ms |
| light | horizon 500 m | 14 ms | 0.4 ms |
| dense town (`generator_budget.py`'s cell) | far 500 m, 16 500 triangles | 194 ms, 26 MB | 16 ms, 9 MB |
| dense town | horizon 1 km | 160 ms | 1.0 ms |
| dense town | horizon 500 m | 48 ms | 0.4 ms |

The once-per-landing check costs 11–15 ms there, spread over pauses.

**Size**, deflated, per 1 km cell. The first two columns deflate each layer alone. The last is the cell file as the ZIP stores it, deflated as one entry, which is why it is smaller than their sum:

| Pack | Far tiles | Horizon (one 1 km block and four quadrants) | Whole cell file |
| --- | --- | --- | --- |
| light | 27 kB | 23 kB | 34 kB |
| dense town | 131 kB | 26 kB | 152 kB |

For the assumed 85/12/3 mix of section 5, this is about 50 kB/km², or about 2.5 GB for a 47 700 km² state. Real roads and DEM noise could double it. `--layers horizon` keeps only the horizon, at most about 25 kB/km².
