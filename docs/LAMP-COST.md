# Lamp and prop cost

Queue task `lamp-cost`: measure and cut the draw cost of lamps and props (instancing, distance limits, lights only near the
car). Written by a cloud session without the game: everything below that is a count or a code fact was checked offline;
nothing here is a frame-time measurement. The last section is the measurement to make in the game before cutting anything.

## What the code already does

| Asked for | State | Where |
| --- | --- | --- |
| Instancing | Lamps, street props, trees and groundcover are items of the level's Forest, one `ForestItemData` per shape, not a `TSStatic` each. A `TSStatic` is only the fallback when Forest items fail to create (after 3 failures). Solid kinds (`#solid`: lamps, props) commit at most `LIMIT.propMax` = 600 items, other kinds `LIMIT.forestMax` = 2500, one kind per `LIMIT.forestStep` = 0.12 s. | `world.lua` `forestCommit`, `plantDataFor`, `LIMIT` |
| Distance limits | Lamps and street props are built for near tiles only (`wantLamps and not distant` in `generator.lua`); far and horizon tiles have none. Their draw distance is then the shapes' own LODs under the Forest renderer. | `generator.lua` `build()` |
| Lights only near the car | At night a pool of `env.lights.count` (default 24) `PointLight`s, no shadows, is moved every 0.4 s onto the lamps nearest the camera within 400 m. Lamps never carry a light of their own. | `world.lua` `showLights`, `lightsStep` |

## Offline counts

Static items per near tile of the synthetic town (`earthng/render_plan.py`'s `town` scene: `generator_budget.py`'s cell,
built by the real `generator.lua` with the swept roads, base `adb5a79`), counted from what `generator.build()` returns:

| Tile | Lamps | Street props | Trees | Groundcover |
| --- | ---: | ---: | ---: | ---: |
| 0,0 | 83 | 1 | 2510 | 3261 |
| 0,1 | 100 | 0 | 2183 | 3659 |
| 1,0 | 81 | 2 | 1366 | 3381 |
| 1,1 | 99 | 118 (100 street trees) | 1753 | 3646 |

Lamps are 1 to 2 % of a near tile's Forest items, street props fewer still; trees and groundcover are about 98 %. Per item
the Forest draws instanced, so the lamps' share of the draw cost should be of the same order, and cutting lamps cannot
move the frame much. The CPU side of the lights is also small: `lightsStep` walks the lamps kept within 2 km every 0.4 s
(a few thousand in a town) and sorts those within 400 m.

What the counts cannot tell: the GPU cost of the 24 point lights and their flares at night (`flareType` BNG_LightFlare_1;
a flare is a visibility query per light per frame), and whether `lamp1.dae` has LODs that end before the Forest stops
drawing it.

## Measuring in the game

One drive, the same route, speed and time of day each run (night for the lights), at the quality preset in use; the
baseline method of `docs/CLOUD-TASKS.md` (median fps, 1 % low, GPU ms):

| Run | Switch | Shows |
| --- | --- | --- |
| A | none | the baseline |
| B | `test.lights = 0` (an empty pool) | the point lights and flares |
| C | `test.propSpacing = 0` | the street props (lamps stay) |
| D | look `statics = sparse` | the lamps and all street furniture |

Decide by the GPU time, not the fps:

- B saves more than 1 ms: shorten the lights' reach from 400 m to about 150 m (a light of 20 m radius lights nothing
  farther away; only its flare shows), then halve the pool. Check that a night street still reads as lit from the car.
- D saves more than 0.5 ms: check `lamp1.dae`'s LODs in the editor; without a small far LOD, give the lamps their own
  shorter draw distance, or space inferred lamps 35 m instead of 30 (`VEG.lampRule` in the street-furniture PR).
- Otherwise lamps and props are not where the frame goes; the counts above point at trees and groundcover (the vegetation
  tasks).
