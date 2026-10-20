# HD Graphics & Animation Plan (phase Z31)

Written 2026-10-20 from a full-map screenshot of the current build. Goal: turn the world from a noisy tile map into a readable, good-looking, animated game. This is phase **Z31** in the master plan (World Bible §69). Work it **one step at a time**, stop for approval after each step, and update the docs as you go (see `AGENTS.md`).

> **Status 2026-10-20 — being redone as an isometric, sprite-based, gritty survival look (Project Zomboid + GTA 1–2), rendered with PixiJS.** After G0–G2 the owner's verdict was *"it looks better, but not what I want."* The target look is now decided (§0.1) and the new step order is I0–I8 (§0.1b). G3–G8 in §4 are superseded. G0–G2 code is in the working tree (not committed).

---

## 0. Redo: what we must settle and what must still change

### 0.1 Target look — DECIDED by the owner, 2026-10-20
| Question | Answer |
| :--- | :--- |
| Reference games | **Project Zomboid** (isometric town survival, detailed houses and interiors) and **GTA 1–2 / Hotline Miami** (bold, readable streets and cars) |
| Camera | **Isometric** (2:1 diamond tiles, 45° view) |
| Renderer | **PixiJS 2D WebGL** (approved; add `pixi.js` as a dependency) |
| Art source | **Sprite sheets** (AI-generated with the Gemini guide, or drawn), high resolution |
| Mood | **Gritty survival**: muted, realistic, tense |
| Zoom | **Both**: default close (a street corner, like Zomboid), player can zoom out to a block |
| Tone rules | **Mild blood allowed** (blood splats and wounds, no extreme gore). This replaces "no blood" in World Bible §70 item 9 and the cartoon-tone rule in `AGENTS.md`. Pets never die stays unless the owner says otherwise. |

Reference images still to collect: 3–5 screenshots of Project Zomboid and GTA 2 / Hotline Miami in `docs/screens/reference/`.

### 0.1b What this means for the plan
- The **straight top-down Canvas look is retired**. G0–G2 code stays only where it carries over: the road graph and town layouts (`src/sim/roads.ts`, `worldgen.ts`), biome sizes, and the deterministic world data. The Canvas ground painter (`groundField.ts`), road ribbons and procedural building drawing are replaced by isometric sprites.
- The **sim does not change**: world, tiles, buildings, roads stay in world (x, y) metres. Only the renderer projects them to isometric screen space: `screenX = (x − y) · TILE_W/2`, `screenY = (x + y) · TILE_H/2 − z · TILE_Z`, with `TILE_W = 2 · TILE_H` (128 × 64 px at 1×) and `TILE_Z` = 78 px per metre at 1× (matches the Blender camera, `STYLE_BIBLE.md` §0.8). Depth sort by `x + y` (then height).
- **New step order (replaces G3–G8 below):**
  1. **I0 Reference and style lock** — collect references; write the isometric style sheet in `STYLE_BIBLE.md` (palette: muted greens, greys, browns, rust; light direction; outline or not; tile size, e.g. 128×64 px per 1 m tile at close zoom; scale sheet: player 1.8 m, door 2.1 m, car 4.5 × 1.8 m, house 8–12 m, street 7 m).
  2. **I1 PixiJS isometric renderer** — new `src/render/pixi/` behind `?renderer=iso`; ground tiles as a texture atlas, depth-sorted sprite layer, camera with close/block zoom levels; old Canvas renderer stays until the new one matches.
  3. **I2 Vertical slice** — one town block, a street corner, a lake edge, at final quality with placeholder-free sprites: ground tiles with transitions, road tiles (asphalt, kerbs, markings), 2–3 houses with walls, windows, doors and roofs, cut-away walls when the player is inside (Zomboid style), a few trees and props, a car, the player and a zombie. Owner approves before rollout.
  4. **I3 Sprite pipeline** — Gemini prompts updated for isometric, gritty, consistent light (`GEMINI_ASSET_GENERATION_GUIDE.md`); sprite sheets in `public/assets/<kind>/` with JSON atlases referenced from `src/data/*` by id.
  5. **I4 Roll out** — all ground types, roads, building types (modular walls, roofs, floors), vegetation, props, vehicles, characters and zombies (8 directions, walk/run/attack/hit/die).
  6. **I5 Lighting and atmosphere** — day/night with light sprites (street lights, windows, headlights, torches), shadows, fog and rain, darker gritty grading.
  7. **I6 Animation and life** — character and zombie animation sets, blood splats (mild), smoke, fire, wind, birds, flickering lights.
  8. **I7 Zoom levels and map view** — close and block zoom; a clean map view.
  9. **I8 Performance presets** — 60 fps desktop with 150 entities, 45 fps phone with 60.

### 0.1c Status of I0 and how to make the art
- **I0 done (2026-10-20), except the images**: the isometric style sheet is written in `STYLE_BIBLE.md` §0 (projection, 128 × 64 px tiles at 1×, 72 px per metre of height, 2× authoring, light from the north-west with 100 / 85 / 62 % face values, cast shadows, muted palette with hex swatches, painted-not-pixel style, outline rules, scale sheet, sprite specs, readability rules). The reference image checklist is `screens/reference/README.md`; the owner collects those images.
- **Recommended art pipeline** (owner asked whether Gemini can do it):
  1. **Gemini (image generation) for concept and style frames**: a street corner, a house, a night scene, in the style sheet's palette and angle. Use these to agree the look before making production art. Good at mood; weak at keeping the exact same angle, scale and light across many images.
  2. **Gemini for single static pieces** that only need one view: ground textures, decals (cracks, stains, leaves, blood splats), props, roof and wall textures. Give it a reference image plus the prompt rules from `GEMINI_ASSET_GENERATION_GUIDE.md` (update that guide for isometric, gritty, 2:1, light from the north-west), then clean up: remove the background, fix the angle, cut to the tile grid.
  3. **3D models rendered to sprites for anything that turns or animates**: characters, zombies, vehicles, and modular building pieces. Make or take CC0 low-poly models (e.g. Kenney, Quaternius; animations from free retargeting tools), set up one Blender scene with the fixed isometric camera and sun from the style sheet (`STYLE_BIBLE.md` §0.8), and render 8 or 16 directions × all animation frames to sprite sheets. This is how isometric games keep every frame consistent; AI image models cannot reliably produce 8 matching directions with walk cycles.
  4. **Pack** the renders and cleaned images into atlases (PixiJS JSON) in `public/assets/<kind>/`.
- **Blender is the 3D tool (owner, 2026-10-20).** Render setup in `STYLE_BIBLE.md` §0.8. Status of the Blender scripts:
  - `world4bros-core/tools/blender/generate_building.py` (made by Antigravity, 2026-10-20) adds a "Building Gen" panel (sidebar `N`) that builds a greybox mid-/high-rise: 2–30 floors, width/depth sliders, a setback step, floor ledges, window frames, a door canopy, a roof parapet, and roof clutter (elevator house, antenna, AC units, water tower). Seeded, so the same seed gives the same building.
  - **Where it lives**: `world4bros-core/tools/blender/` (moved there from the repo root on 2026-10-20). Keep all Blender scripts in this folder; nothing goes outside `world4bros-core`. In Blender: Scripting tab → Open this file → Run Script, then press `N` in the 3D view and open the "Building Gen" tab.
  - **Fits**: downtown blocks in metro towns (apartments, offices) as blockouts for the vertical slice.
  - **Gaps against the style sheet**:
    1. Floor height defaults to 3.0 m; the scale sheet says 2.7 m.
    2. Window "recesses" are boxes centred on the wall, so they stick out 6 cm instead of cutting in; use a boolean or inset faces.
    3. The door is 2.0 × 2.3 m (fine for apartment double doors; houses need 0.9 × 2.1 m).
    4. It makes one combined mesh. The game needs **modular per-tile pieces** (walls in both orientations, corners, doors, windows, floors, roof pieces) for depth sorting and the wall cut-away; split the output or build from a kit.
    5. Small-town buildings are missing: 1–2 storey houses with gable/hip roofs, porches, garages, sheds, shops (the building types in `src/data/buildings.ts`).
    6. No materials yet (greybox on purpose); the palette and grime pass come later.
  - **Production pipeline**: `BUILDING_PIPELINE_PLAN.md` (phase Z33) holds the full checklist, research on existing generators and sprite renderers, and an improvement list.
  - **Scripts still needed** in `world4bros-core/tools/blender/`:
    1. `render_iso.py`: sets up the camera, sun, ortho scale, transparent PNG output and colour management from §0.8; renders the selected object in 8 or 16 directions and all animation frames; writes `public/assets/<kind>/` with frame names `kind_id_variant_dir_frame`.
    2. `house_kit.py`: small-town house generator from modular pieces (footprint 8–12 × 7–10 m, 1–2 storeys, gable/hip roof, porch, windows, doors), matching `BuildingDef` sizes and roof types.
    3. `pack_atlas` step (TexturePacker or a small script) to make PixiJS atlases.
- **Tools the owner uses**: Visual Studio (with Copilot), Antigravity IDE (Gemini), Blender and GitHub. Keep all docs and assets in this repo and push to GitHub so every tool and agent sees the same plan.

### 0.2 What still looks wrong after G0–G2 (from the 2026-10-20 screenshots)
- **Too map-like**: from above, towns read as a grid plan with coloured rectangles; buildings have no visible height, walls or windows.
- **Flat lighting**: no building or tree shadows on the ground; nothing gives depth.
- **Uniform streets**: every town street is the same grey tile pattern; whole intersections are painted as crosswalks; no road markings variety, kerbs, parking, wear or decals.
- **Same-y towns**: every town is a square grid; no main square, park, church, school, gas station, farmland or edge-of-town transition.
- **Vegetation**: trees are small dark star shapes with no canopy volume, no shadow and no sway.
- **Water**: lakes have a gradient and foam but no movement, reflections or reeds; town streets run straight into lakes; no bridges.
- **Props**: street lights, cars and props are tiny and hard to read; scale between player, cars and houses still feels off.
- **No life**: nothing moves in the world except characters.

### 0.3 Must-do list for the redo (whatever look is chosen)
1. Fix the target (0.1) and lock it in `STYLE_BIBLE.md` with reference images before writing code.
2. Build one **vertical slice** first: one town block plus a road and a lake edge, at the chosen camera and zoom, at final quality. Get it approved before rolling out.
3. **Buildings with height**: visible front walls (3/4 tilt) or real 3D, windows, doors, roof shapes with lit and shadow sides, and a cast shadow.
4. **Shadows and light** for everything that stands up: buildings, trees, cars, props, characters.
5. **Town layouts that are not grids**: a main street with shops, a square, residential streets that curve, parks, lots with driveways and fences, edges that fade into farmland.
6. **Streets with detail**: kerbs, lane markings only where they make sense, crosswalks only at crossings, parking bays, cracks, puddles, leaves.
7. **Trees and plants with volume**: layered canopies, trunks, shadows, wind sway; hedges and flower beds in towns.
8. **Living water**: moving ripples, foam that animates, reeds, docks and bridges where roads cross.
9. **Readable scale**: one agreed scale sheet (player, car, door, house, street, tree) in `STYLE_BIBLE.md`, and every asset checked against it.
10. **Animation and life**: wind, smoke, birds, flickering lights, idle motion (G6) once the look is fixed.
11. **Performance budget** kept (60 fps desktop with 150 entities) — decide the renderer (0.1 item 3) before scaling up art.
12. Keep `docs/screens/` before/after shots for every step, and update this file, World Bible §69, `PROGRESS.md` and `ROADMAP.md` each time.

## 1. What is wrong today

Ranked by how much each hurts the look and the play. Causes are from the code where known.

| # | Problem | What you see | Cause |
| :- | :--- | :--- | :--- |
| 1 | **Blue squares on almost every roof** | A bright blue box with an X and yellow ropes on most houses; reads as a missing-texture marker | `paintRoofTarp` in `src/render/buildings.ts` (~line 1262) is drawn on every `tin` and `broken` roof at the same spot (`w*0.55, h*0.45`). 4 building types have the `tarp` feature. |
| 2 | **Staircase roads** | Highways, the ring road and diagonal trails are jagged stair steps with thick black edges | Roads are decided per tile (`getStreetDetails` in `src/sim/worldgen.ts`, sine "harmonic" bends) and painted per tile in `src/render/tiles.ts`. A curve drawn tile by tile is always a staircase. |
| 3 | **Noisy, patchy ground** | Grass, dirt, rock and teal blotches in random patches; towns sit on a mix of 4 ground types; no clear biome | Ground colour is one fill per tile plus random blotches (`paintChunk` steps 1–3). Biomes are picked per tile by noise, and nothing flattens or clears the ground under towns. |
| 4 | **Town layout does not read as a town** | A square grid of grey cobblestone streets, most blocks empty, a few houses floating near block edges, a highway ring around nothing | Town streets are a fixed grid from `getStreetDetails`; houses only fill some lots (`townBuildings`, 14% vacant + rejections); no driveways, sidewalks to doors, parks or shops in the centre. |
| 5 | **Wrong scale** | Streets are as wide as houses; houses are tiny next to the road grid; props are specks | Houses are 4–14 tiles; town streets are several tiles wide; highways 4.6 m. Nothing was tuned together. |
| 6 | **Blurry water edges** | Lakes have soft stair-stepped shores and a flat cyan band | Same per-tile ground painting as #3; no shoreline shape, foam or depth gradient. |
| 7 | **Flat, dark vegetation** | Trees are dark spiky blobs with no volume, no shadow, no sway | Tree sprites in `src/render/objects.ts` are small, dark, and drawn without a ground shadow or highlight side. |
| 8 | **No light or depth** | Everything has the same brightness; no shadows from houses or trees; no time-of-day mood at map scale | Lighting (`src/render/lighting.ts`, `atmosphere.ts`) is a full-screen tint; objects do not cast directional shadows onto the ground. |
| 9 | **Little animation** | The world is still: no wind, water movement, smoke, birds, or reactive grass | Sprites are cached as still images; only characters animate. |
| 10 | **Hard to play** | You cannot find the player, roads or points of interest at a glance | Low contrast between player, ground and props; the busy ground hides everything. |

---

## 2. The big decision: renderer

The sim and the renderer are already separate (`src/sim` never imports `src/render`), so the renderer can be replaced without touching gameplay.

| Option | What it means | Good | Bad |
| :--- | :--- | :--- | :--- |
| **A. Keep Canvas 2D, fix it** | Keep `src/render/*`, fix the causes above | No new dependency; steps 1–4 below work either way | Shaders, smooth lighting, wind sway and thousands of animated sprites are slow or impossible |
| **B. WebGL with PixiJS (recommended)** | Draw with PixiJS v8: sprite batching, filters and custom shaders. Procedural art still works: draw to a canvas once, upload as a texture | 10–50× more sprites, real lighting, water and wind shaders, particles, smooth zoom | New dependency (ask first, per `AGENTS.md`); renderer rewrite of 3–5 days |
| C. Full 3D (three.js) | Real 3D models with a top-down camera | Best depth and lighting | Biggest rewrite; needs 3D models; drops the current ink style |

**Recommendation**: do steps G0–G3 in Canvas 2D now (they fix the worst problems and carry over unchanged), then move to PixiJS for G4–G7, where shaders and batching matter. Get the owner's OK before adding PixiJS.

---

## 3. Art direction for "HD"

Keep the World Bible style (Part 10: pop-up paper and rubber-hose cartoon, ink outlines, no gore) but raise the quality bar:

- **Camera**: top-down with a slight 3/4 tilt for buildings and trees (you see the roof and a strip of the front wall). Ground stays flat top-down.
- **Resolution**: author sprites at 2× the largest zoom (at least 64 px per metre), and pick mip levels by zoom. Respect `devicePixelRatio` (already in `sprites.ts` `ppuFor`).
- **Value hierarchy**: ground is the calmest and lowest-contrast layer; roads one step brighter; buildings and props stronger; characters and pickups strongest, with a light rim so they pop.
- **One biome per area**: biomes change over large distances (hundreds of metres) with a soft 6–12 m blend, never as speckles.
- **Light from the top-left**: every object has a lit side, a shadow side and a soft ground shadow toward the bottom-right.
- **Palette**: biome palettes from `STYLE_BIBLE.md` §4; one accent colour per object at most.

---

## 4. Step-by-step plan

Each step: the change, the files, and how to tell it is done. Take before/after screenshots from the same seed and camera position each time (see §6).

### G0. Quick fixes (Canvas 2D, about 1 day)
1. **Roof tarps**: draw them on at most ~1 in 6 `broken`/`tin` roofs (pick by `hash2(seed, b.x, b.y)`), at a random position and angle, in a dull weathered blue, with thinner ropes. Never on `cottage`, `townhouse` or `clinic`. File: `src/render/buildings.ts`.
2. **Clear ground under towns**: inside a town radius, force one ground type (grass or packed dirt by biome) and skip rock/teal patches. Files: `src/sim/worldgen.ts` (terrain stage), `src/render/tiles.ts`.
3. **Thinner road outlines**: drop the thick black edge; use a 1-tile kerb or soft shoulder instead.
4. **Player readability**: a soft light ring under the player and a 1 px light outline on characters.

Done when: no blue squares in a town screenshot except the odd broken house; towns sit on one ground type; the player is the first thing you see.

### G1. Smooth ground and water (Canvas 2D, 2–3 days)
1. **Blend weights, not tile picks**: per tile corner, keep a weight for each nearby ground type (from the same climate noise). Paint each ground type as a texture masked by its weight, so boundaries are smooth curves. Replace the per-tile fill in `paintChunk`. `terrainRenderer.ts` is an unused dual-grid attempt; delete it or reuse the corner-mask idea.
2. **Bigger biome scale**: lower the climate noise frequency so a biome covers hundreds of metres; keep small detail only as decals (flowers, pebbles, cracks).
3. **Ground textures**: one tileable texture per ground type (grass, dirt, sand, rock, snow, mud, ash), drawn once at high resolution and reused. Add small decals (grass tufts, stones, flowers) by deterministic scatter.
4. **Shorelines**: trace the water edge as a smooth contour (marching squares on the water weight), then draw: wet sand band, a white foam line, shallow-to-deep colour gradient.

Done when: zoomed out, you can draw a clean line around each biome and lake; no square steps anywhere on the ground.

### G2. Roads as curves, towns as places (2–4 days)
1. **Road graph first**: generate roads as a graph of points (towns and POIs as nodes, connected by a minimum spanning tree plus a few loops), then smooth each edge as a spline. Store the polylines; per-tile queries (`getRoadType`) look up the distance to the nearest polyline. File: `src/sim/worldgen.ts`.
2. **Draw roads as ribbons**: render each polyline as a stroked path with the right width, an asphalt texture, edge lines, centre dashes along the curve, and rounded junctions. No per-tile road painting. File: `src/render/tiles.ts` (or a new `src/render/roads.ts`).
3. **Town layout by size** (World Bible §24, §113):
   - *Hamlet* (3–8 houses on one road), *village* (a main street and 1–2 side streets), *town* (a main street with shops, a square, residential side streets), *city* (a grid core with blocks of shops and apartments, suburbs around it).
   - Build streets first, then cut lots along both sides of every street, so every lot faces a street. Fill lots fully: shops and apartments in the centre, houses with yards outside it, sheds and farms at the edge.
   - Add squares, parks, parking lots, driveways and front paths. No empty grey blocks.
4. **Scale pass**: 1 tile = 1 m. Residential street 6–7 m, main street 9–10 m, highway 12 m with shoulders. Houses 8–12 m × 7–10 m. Characters about 0.6 m wide. Update `BuildingDef` sizes in `src/data/buildings.ts` and the street widths together.

Done when: a zoomed-out screenshot of a town looks like a town, with houses lining every street and no empty grid; roads curve smoothly.

### G3. HD buildings (2–3 days)
1. **Roof shapes**: gable roofs with a lit and a shadow slope; hip roofs as four faces; flat roofs with parapet, gravel and HVAC; tin roofs with ribs. Ink outline on every roof. Ridge and eave lines.
2. **3/4 front wall**: show a strip of the front wall below the roof edge with windows, door, trim and shutters, so a house reads as a house even from above.
3. **Variety**: palette per building (already started), plus random dormers, chimneys, gutters, solar panels, satellite dishes and porch types, chosen by hash. Make roofs and features data (`roofs` and `features` registries) so a new style needs no code.
4. **Shadows**: every building casts a soft shadow to the bottom-right, its length set by building height.
5. **Interiors**: keep the roof-fade on entry; make floors, rugs and furniture readable at close zoom.

Done when: no two neighbouring houses look the same; at map zoom you can tell houses, shops, apartments and garages apart.

### G4. Move rendering to PixiJS (3–5 days, needs approval)
1. Add `pixi.js` (ask first). New `src/render/pixi/` with a `PixiRenderer` that reads the same sim state as `renderer.ts`.
2. Ground chunks become textures (keep `paintChunk` output as the source at first). Objects, buildings and characters become sprites in a z-sorted container (sort by y).
3. Sprite atlases: pack cached sprites into atlases per chunk or per kind to keep draw calls low.
4. Keep the old Canvas renderer behind a `?renderer=canvas` flag until the new one matches it.

Done when: the game looks the same as before on Pixi, runs at 60 fps with 150 entities on a mid-range laptop, and `?renderer=canvas` still works.

### G5. Lighting and atmosphere (2–3 days, Pixi)
1. **Light layer**: render lights (campfires, street lights, headlights, windows at night) into a light texture and multiply over the world. Smooth day/night colour grading per biome (lookup tables from `STYLE_BIBLE.md`).
2. **Directional shadows**: sun angle moves through the day; shadows of trees and buildings lengthen at dusk.
3. **Weather**: rain streaks and puddle ripples, fog layers, snow, heat shimmer in the desert, with the same `WEATHERS` rows already in `src/render/atmosphere.ts`.
4. **Ambient occlusion**: soft dark contact under every object.

Done when: night looks dark and warm-lit, not just tinted; dusk shadows are long.

### G6. Animation (3–5 days)
1. **World motion** (shaders on Pixi): tree and grass wind sway, water ripples and moving foam, chimney smoke, flickering fire light, flags and laundry, birds and butterflies, leaves falling.
2. **Characters**: a 2D bone rig per character (hips, spine, head, arms, legs) with walk, run, idle breathing, attack wind-up, hit flinch, fall and get-up clips. Blend between clips. Stride matching so feet do not slide (World Bible §64, §70).
3. **Secondary motion**: springs on hair, backpacks, hats, ears and tails; squash on landing; lean into turns.
4. **Combat feel** (World Bible §79): smear frames, hit-stop on the attacker and victim only, screen punch scaled by the reduced-motion setting, comic sound words.
5. **Effects** (ties into Z26): particles for acid drips, flames, sparks, frost, spores, smoke, explosions.
6. **Reactive world**: grass bends when walked through, bushes shake, doors swing, cars bounce on suspension, objects wobble when hit.

Done when: standing still for 10 seconds, something in view is always moving; every attack has a wind-up, impact and follow-through.

### G7. Zoom levels and map view (1–2 days)
1. **Level of detail**: close zoom shows everything; mid zoom drops small decals and props; far zoom shows a clean map style (flat biome colours, road lines, building blocks, icons).
2. **Minimap and world map** use the same clean map style.
3. **Points of interest** get an icon and a name at far zoom.

Done when: the full-map screenshot looks like a designed map, not a noisy texture.

### G8. Performance and quality settings
- Budget: 60 fps desktop with 150 entities, 45 fps phone with 60 (World Bible §70).
- Quality presets Low / Medium / High: shadow quality, decal density, sway shaders, particle count, render scale.
- Cache everything that does not move; never redraw complex ink shapes each frame.

---

## 5. Assets: procedural, generated or drawn

- **Keep procedural**: ground textures, roads, water, terrain decals, simple props. They scale to any resolution.
- **Use high-res sprite sheets** for hero art: characters, zombies, vehicles, trees, building roof and wall pieces. Make them with the Gemini prompts in `GEMINI_ASSET_GENERATION_GUIDE.md`, or draw them, at 2× the largest zoom, on transparent backgrounds, with consistent top-left light.
- **Rules**: original or CC0 only, no real brands, cartoon tone, no gore (World Bible §70).
- Put sprite sheets in a new `public/assets/<kind>/` folder in `world4bros-core` with a JSON atlas, and reference them from data (`src/data/*`) by id so art can change without code changes.

---

## 6. How to work on it (for any agent)

1. One step (G0, G1, ...) at a time. At the end of each: `npm run verify`, play it with `npm run dev`, check the console, and take screenshots.
2. **Fixed screenshot set**: seed `4242`, these views, before and after: spawn town at close zoom, spawn town at map zoom, a lake shore, a highway curve, a forest edge, night at the campfire. Save them as `docs/screens/<step>-<view>.jpg` (or `.png`).
3. Stop and show the owner before starting the next step.
4. Keep the sim untouched: all of this is in `src/render` and `src/data`, except road graph and town layout (G2), which are in `src/sim/worldgen.ts` and need determinism tests (in `tests/map.test.ts` or a new `tests/worldgen.test.ts`).
5. Update the docs each step: World Bible §69 (Z31 row and "Where to resume"), `PROGRESS.md`, `ROADMAP.md`, `STYLE_BIBLE.md` for any new art rule, and this file (tick off what is done).

## 7. Order and rough size

| Step | What | Renderer | Size |
| :--- | :--- | :--- | :--- |
| G0 | Quick fixes: tarps, clean ground under towns, road edges, player readability | Canvas | ~1 day |
| G1 | Smooth ground, biomes, shorelines | Canvas | 2–3 days |
| G2 | Road graph, smooth roads, town layouts, scale | Canvas + sim | 2–4 days |
| G3 | HD buildings | Canvas | 2–3 days |
| G4 | Move to PixiJS (needs approval) | Pixi | 3–5 days |
| G5 | Lighting and weather | Pixi | 2–3 days |
| G6 | Animation | Pixi | 3–5 days |
| G7 | Zoom levels and map view | Pixi | 1–2 days |
| G8 | Performance and quality presets | Pixi | ongoing |

## 8. Status
- [x] G0 Quick fixes (2026-10-20), except road edges:
  - Roof tarps: `tinRoofTarp` in `src/render/buildings.ts` puts a faded tarp on about 1 tin roof in 6, placed and turned by the building's position; broken roofs keep a faded tarp over the hole. Test: `tests/roof-tarps.test.ts`.
  - Town ground: `townGround` in `src/render/tiles.ts` draws grass (or sand/snow in desert and tundra towns) within 10 tiles of any building, never over water. Drawing only; the sim terrain is unchanged.
  - Player: a soft glow and a slow-breathing ring under the player (`playerMarker` in `src/render/renderer.ts`), at least ~14 px on screen at any zoom.
  - Road edges: skipped at the time because roads were switched off; done in G2 (roads are ribbons now).
  - Screenshots: `screens/G0-before-close.jpg`, `G0-after-close.jpg`, `G0-before-map.jpg`, `G0-after-map.jpg` (seed 4242).
- [x] G1 Smooth ground and water (2026-10-20):
  - New `src/render/groundField.ts`: each pixel blends the four nearest tiles after a small noise warp and takes the ground type with the most weight, so boundaries are smooth curves. Painted at 16 px per tile and scaled up. Replaces the old per-tile fill, seam feathering, blotches, tile-edge relief lines, shoreline strokes and 2×2 biome tint in `tiles.ts`.
  - Water: depth gradient from tile height (light at the shore, dark far out), a foam line inside the edge, and darker wet ground on the shore.
  - Relief: soft top-left lighting from the slope of the height field instead of lines along tile edges.
  - Bigger biomes (`src/sim/worldgen.ts` `WorldGen.sample`): `BIOME_FREQ` 0.0038 → 0.0024, height weight in the biome pick 0.8 → 0.35, in-biome patches from 3-octave fbm at `PATCH_FREQ` 0.045 instead of raw noise at 0.09. This changes the world layout for every seed. Test: `tests/biome-scale.test.ts` (average biome stretch ≥ 60 tiles).
  - Cost: the ground pass is about 9 ms per chunk (warm); a whole chunk paints in about 20–25 ms and chunks are painted a few per frame by the renderer's budget. The town-ground lookup now runs once per chunk.
  - Not done: textured ground (G1 item 3) beyond tone noise and grain; detail strokes (blades, pebbles) are still per tile. Remaining straight facets on some patch edges at map zoom.
  - Screenshots: `screens/G1-before-close.jpg`, `G1-after-close.jpg`, `G1-before-map.jpg`, `G1-after-map.jpg` (seed 4242).
- [x] G2 Roads and towns (2026-10-20):
  - Roads are back on. Copilot's earlier rewrite had switched them off (`ROADS_ENABLED = false`) and added a test that roads stay off; both are replaced.
  - Road graph: new `src/sim/roads.ts`. Towns are nodes; highways follow a minimum spanning tree plus the two shortest extra links, leaving each town from the main-street end that faces the other town. Paths are smoothed (Chaikin corner cutting plus moving-average passes) and bend around deep water. One to three dirt tracks (55–115 m) leave each town from side-street ends. A spatial hash answers point queries, so speed, tree scatter, parked cars and lot placement use the same geometry. The old endless 96 m grid of highways, dirt roads and diagonal trails is gone.
  - Drawing: new `src/render/roadRibbons.ts` strokes highways (gravel shoulder, asphalt, wheel wear, white edge lines, yellow centre dashes) and dirt tracks (verge, ruts, grass centre strip) along the curves. Town streets keep their tile surfaces (they are straight).
  - Towns by size: the small / big / metro profiles from Copilot's rewrite are kept and scaled 1.3×; regional towns are spread 1.35× wider so the bigger towns never overlap.
  - Scale (1 tile = 1 m): main streets 7 m with 0.9 m sidewalks (`MAIN_HALF` 3.5), side streets 4 m (`SIDE_HALF` 2.0), highways 6 m (`HIGHWAY_HALF` 3.0), dirt tracks 3.6 m. Lots are about 11 m wide (`LOT_SIZE`). Houses are now 7–10 m (cottage 7–9 × 6–8, log cabin 8–10 × 7–9, townhouse 7–9 × 8–10), shops 8–11 m, sheds 5–6 m. This is short of the 12 m highway in §4 G2; widen later if cars need it.
  - Tests: `tests/map.test.ts` (every town has a main street and a highway; roads are smooth curves) replaces the roads-off test; all 143 tests pass.
  - Not done: town street grids still ignore water (a boulevard can run into a lake); bridges are not drawn; supermarkets and apartment blocks rarely fit an 11 m lot; whole intersections are painted as crosswalk; no parks or squares beyond the plaza.
  - Screenshots: `screens/G2-before-close.jpg`, `G2-after-close.jpg`, `G2-before-map.jpg`, `G2-after-map.jpg` (seed 4242).
- [ ] G3 HD buildings
- [ ] G4 PixiJS renderer
- [ ] G5 Lighting and atmosphere
- [ ] G6 Animation
- [ ] G7 Zoom levels and map view
- [ ] G8 Performance and quality presets
