# Cloud tasks for the `niedersachsen` branch

Instructions for a Claude Code session working on GitHub, without the game. Pick one task, finish it, open one PR.

## What this is

EarthNG builds a drivable BeamNG.drive world from real map data. A Python exporter (`earthng/`) writes a pack; Lua in the game (`mod/lua/ge/extensions/e2r/`) turns the pack into tiles of 500 m while the player drives. This branch adds official German survey data, swept road meshes (`sweep.lua`) and many road fixes. The main problem now is stutter while driving.

## Rules for every session

1. **Base:** check out `niedersachsen`. The default branch `master` is the upstream project and has none of this work.
2. **Branch:** one branch per task, `fix/<task-name>`, from `niedersachsen`.
3. **PR:** into `niedersachsen` of `mcheijink/earthng`. Not into `Alezx311/earthng`: GitHub offers upstream as the default target, so change it.
4. **You cannot run BeamNG here.** The pack, the downloaded data and the game are not in this repo. Say plainly in the PR what was tested and what needs a test in the game. Never write that something works in the game.
5. **Tests:** run the `earthng/*_test.py` files that apply. If a test needs local data, say so and skip it. Compile every changed Lua file with LuaJIT or `luac -p` if available (`earthng/lua_compile_test.py` compiles them all). For a change you could see in the game, run `python earthng/render_plan.py` and look at its top-down images of the real generator's output on synthetic scenes (`--list` shows them). The workflow `niedersachsen checks` runs the tests and the renders on every PR into `niedersachsen` and uploads the images as the `renders` artifact.
6. **Lua:** keep CRLF line endings. `build()` in `generator.lua` is at LuaJIT's limit of 200 locals: add no local there, put helpers in a table as the code already does (`OFFROAD`, `ROADS`).
7. **Style:** match the surrounding code and its comment density. Comments give the measured reason for a value.
8. **Privacy:** this repo is public. Never add coordinates, addresses, personal routes, trip data or screenshots of them.
9. **Scope:** one task per PR, smallest change that does it. No reformatting.

Every PR says: what changed, why, what was tested, and exactly what to check in the game.

## Measured baseline

Measured in the game at 120 km/h on a desktop CPU / GPU, 2560×1440. Median 49 fps, 1 % low 32 fps, about 23 fps in a dense harbour town. The GPU is flat at 4–5 ms and the UI is near zero. The lag is bursts of CPU work on the main thread every 1–3 s.

The mod's own profiler (`S.prof` in `world.lua`):

| Part | Mean per frame | Worst frame |
| --- | --- | --- |
| `native` (tile generator) | 5.9 ms | 81 ms |
| `collision` | 0.07 ms | 142 ms |
| `queue` (spawning items) | 1.4 ms | 52 ms |
| `tiles` (load and unload) | 0.25 ms | 77 ms |

About 12 tiles were built in 15 s. Lua garbage is 63 kB per frame, with 6 GC cycles per second on a 357 MB heap.

## Tasks

Ordered by value for effort. Tasks 1–3 touch different code and can run in parallel, except that 1 and 3 both touch generator hot paths: if both run, the second one rebases.

### 1. `generator-budget`: keep the generator inside its frame budget

`world.lua` gives the generator 4 ms per frame (`BUDGET_MS`), but single frames reach 81 ms. In `native.lua`, `active.pause` compares the clock only on every 8th call, and `generator.lua` has long stretches without a `pause()`.

- Check the clock on every `pause()` if the timer cost allows it.
- Find the longest stretches between pauses in `generator.lua` and `sweep.lua` and add pauses, or split the work.
- Goal: no single generator step above about 8 ms. A coroutine yield must not leave shared state half built.
- Test: a small timing harness is welcome if it can run outside the game; otherwise reason from the code and list the stretches changed.

### 2. `collision-spread`: avoid the 142 ms collision rebuild

`updateCollision` and `collisionPoints` in `world.lua` rebuild collision in one block when tiles change.

- Find out what the rebuild does and what can be spread over frames, delayed, or limited to the tiles that changed. Check the existing notes in the code first; it has been tuned before.
- The engine call may be a single blocking call that cannot be split. If so, say so and reduce how often it runs and how much it covers, without leaving a car over a tile with no collision.

### 3. `less-garbage`: cut Lua garbage in the hot paths

- Find per-frame and per-tile allocations in `generator.lua`, `sweep.lua` and `native.lua`: tables, closures and strings in tight loops. Reuse tables where it is safe.
- Goal: well under the 63 kB per frame. Look at `collectgarbage` step settings only after that.
- Do not change geometry. Output must stay identical.

### 4. `drape-debug`: find why the aerial drape shows no photo

Branch `drape-wip` holds a first drape: near ground uses a per-tile material `e2rf_dop_<i>_<j>` with a DDS `baseColorMap` from `/art/e2r_dop/`, UVs from position. The textures are not in the repo.

In the game the ground became flat pale beige with no photo detail. An offline dump shows the material assigned and UVs inside 0–1, so simple UV arithmetic is probably not the fault.

- Read how the ground lattice reaches the GPU: does that path keep per-vertex UVs, or does it replace them (terrain shading, tiling scale, a material layer setup)?
- Compare the material definition with those the mod already creates: version, stage layout, `baseColorMap` key, colour-space flags.
- Report the most likely causes, ranked, and fix what the code proves. Base this task on `drape-wip`, and open its PR against `drape-wip`.

### 5. `horizon-stale`: stop old horizon blocks overlapping newer ones

`native.lua` replaces a 1000 m horizon block with 500 m quads as the car nears, but the old block stays drawn until cache eviction (`trimHorizon`, about 16 stale entries; `M.horizonShow` is unused). `world.lua` only drops blocks that left the cache. Dropping at once would leave holes.

- Drop a stale block only when everything that replaces it is spawned.

### 6. `town-overlaps`: remove coplanar road layers in town

In towns `generator.lua` emits strips, junction plates and sidewalks that overlap within 2 cm, which flickers. The `subtract()` cut uses the asphalt bins only, so sidewalks and plates are not cut against each other.

- Cut them against each other at junctions. Keep kerbs and sidewalks intact.

### 7. `prebuilt-tiles`: design first, no code yet

The main cure for the stutter is to stop generating geometry while driving. Write `docs/PREBUILT-TILES.md`:

- what the exporter could bake into the pack (ground lattice, roads, marks), and what must stay at run time;
- pack size and load time estimates, with their basis;
- how `native.lua` and `world.lua` would change, and a migration path keeping the current generator as a fallback;
- risks. End with a recommendation. Open a PR with the document only.
