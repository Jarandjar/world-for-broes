# Generation Ideas: Playable Houses, Rooms and Vehicles

Written 2026-10-20 after looking at **CityFoundry** (`http://localhost:4040/`). Goal: everything the generators must make so the world is **playable** like Project Zomboid — houses you can enter, rooms with loot, doors and windows that matter, and vehicles built from parts. Related: [BUILDING_PIPELINE_PLAN.md](BUILDING_PIPELINE_PLAN.md) (Z33), [OPEN_SOURCE_TOOLKIT.md](OPEN_SOURCE_TOOLKIT.md), [HD_GRAPHICS_PLAN.md](HD_GRAPHICS_PLAN.md) §0, [STYLE_BIBLE.md](STYLE_BIBLE.md) §0, World Bible §14–§16, §26, §38–§49.

---

## 1. What CityFoundry does today, and what it is missing

**Today** (Blender 5.2 command line behind a web UI): seed, block layout (2×2, 3×2, 4×2 buildings), skyline profile (3–6, 4–14, 8–18 floors), toggles for setbacks, roof clutter, sidewalk trees, corner lamps and tactical cover; downloads `.glb` / `.blend`; exports a "tactical JSON" with building footprints, floors, heights, one entrance point, cover points, sniper perches and spawn points.

**Missing for a playable game:**
1. **No interiors**: buildings are solid blocks. Nothing to enter, loot or hide in.
2. **Only city towers**: no small-town houses, shops, garages, farms (most of our world is small towns, World Bible §24).
3. **No rooms, doors or windows in the data**: the JSON has one `entrance` per building; the game needs every door, window, wall and room.
4. **Not on our grid**: footprints like 10.5 × 12 m and heights like 11.4 m don't snap to whole 1 m tiles; storey height is ~2.8 m (style sheet: 2.7 m).
5. **No link to the game's world**: blocks are made on their own; they should come from the game's town lots (`townBuildings()` in `src/sim/worldgen.ts`).
6. **No sprite output**: the game is isometric 2D; we need layered isometric renders (floor, walls, roof, shadow), not only `.glb`.
7. **Tactical data is city-shooter style** (sniper perches 39 m up): our game needs ground-level survival data (entry points, barricade spots, loot rooms, hiding spots, noise).
8. **Uniform look**: identical window bands on every face; no wear, damage states or variety per district.

---

## 2. What "playable house" means (must-haves)
- Enter through doors; **every room reachable** from an outside door.
- **Interior walls with collision**, matching the art exactly.
- **Doors**: open/close, locked (key, lockpick, crowbar, kick), barricade, noise when kicked.
- **Windows**: look through, break (noise, glass), climb through (slow, can cut you), board up, curtains (hide light), smash-and-grab.
- **Rooms with purpose** and matching loot (kitchen → food, bathroom → medicine, garage → tools).
- **Containers** you search: fridge, cupboards, wardrobe, desk, toolbox, safe.
- **Hiding spots**: wardrobe, under bed, bathtub, closet (already in World Bible §26).
- **Cut-away**: walls and roof hide when you are inside (Zomboid style).
- **Light**: dark interiors at night, light switches if power is on, flashlight.
- **Stairs** to upper floors, basements and attics.
- **Zombies inside** some houses; sounds through walls.
- **Safehouse**: claim a house, barricade it, sleep, store items (ties into base building, World Bible §28).

---

## 3. House generator: the pipeline
One deterministic pipeline, seeded by the building id, run headless in Blender and mirrored in the sim (TypeScript) so data and art always match.

1. **Lot** (from the game): size, street side, district, town size, biome.
2. **Archetype pick**: by district and town size (see §5), weighted.
3. **Footprint**: rectangle, L, T, U, with optional garage wing, porch, back deck; snapped to the 1 m grid.
4. **Floors**: 1–2 storeys (+ basement, attic); stairs placed first so floors connect.
5. **Floor plan** (rooms inside the footprint), see §4.
6. **Openings**: doors between rooms (by adjacency rules), outside doors (front + back/side), windows on outer walls by room type (no window in a closet, big windows in living rooms).
7. **Furniture**: per room template, against walls, keeping door swings and paths clear.
8. **Containers and loot slots**: from furniture (fridge, cupboard, wardrobe) with loot table ids.
9. **Exterior**: roof type, siding/brick, porch, chimney, gutters, garage door, yard (fence, shed, garden, pool, trampoline, doghouse).
10. **Story layer**: who lived here (§8) — decides decor and special loot.
11. **Damage state**: pristine, worn, looted, burnt, flooded, collapsed, overgrown, barricaded (World Bible §26).
12. **Validation** (§7) — reject and reseed if a rule fails.
13. **Outputs**: `building.json` (data contract, §6) + Blender model + layered isometric sprites + a thumbnail.

---

## 4. Floor-plan generation methods (pick one, mix later)
- **Template library first (fastest to playable)**: 20–40 hand-made plans per archetype in JSON (or drawn in Tiled), mirrored and rotated; generator fills furniture, loot and damage. Most reliable.
- **Squarified treemap**: split the footprint into rooms sized by area targets (living 25 %, kitchen 12 %, bedrooms…) — classic procedural floor-plan method; gives clean rectangles.
- **BSP (binary space partition)**: recursively split the footprint with walls; good for offices and apartments.
- **Room-graph first**: build the adjacency graph (entrance → hall → living → kitchen; hall → bedrooms → bathroom), then lay it out; guarantees sensible flow.
- **Grid growth**: seed rooms on the grid and grow them until they fill the footprint (works for odd shapes like L and U).
- **Wave Function Collapse** on a room-tile set: high variety for big buildings (schools, hospitals, malls).
- **Corridor rule**: houses ≥ 3 bedrooms get a hallway; apartments get a central corridor with units off it.

---

## 5. Building catalogue (max list)

### Homes
Cottage, bungalow, ranch house, two-storey family house, townhouse/row house, duplex, trailer/mobile home (trailer parks!), log cabin, farmhouse, mansion with pool, small apartment block (walk-up), apartment tower lobby + stairwell, survivor shack, boarded crack house, cabin by the lake, fishing hut.

### Shops and services
Corner grocery, supermarket, pharmacy, gas station with shop and pumps, diner, fast food, pizza place, bar, liquor store, hardware store, gun shop, sporting goods, clothing store, electronics store, bookstore, pawn shop, laundromat, barber, car dealership, auto repair garage, car wash, motel, hotel, bank (vault), post office, mall (later).

### Public and emergency
Police station (armoury, cells), fire station (truck bay, gear), hospital / clinic (pharmacy, wards), school (classrooms, gym, cafeteria), church + graveyard, town hall, library, community centre, radio station, military checkpoint, army base buildings (already in the game), prison.

### Industrial and rural
Warehouse, factory, lumber yard, scrapyard office, power substation, water tower, barn, silo, stable, greenhouse, farm shed, grain elevator, train station, bus depot, construction site (half-built shell — great playable ruin).

### Small structures
Shed, garage (detached), carport, outhouse, doghouse, gazebo, bus stop, phone booth, kiosk, hunting stand, treehouse, playground, pool house.

Each type gets: room templates, loot tables per room, exterior style, sizes from `STYLE_BIBLE.md` §0.5, and which districts/town sizes it appears in.

---

## 6. Data contract (`building.json`) — one file per building
The sim reads this; the art is made from the same file.
- `id`, `archetype`, `seed`, `lot` (x, y, w, h, street side), `floors`, `roofType`, `damageState`, `story` (owner profile).
- `walls[]`: segments on the 1 m grid (x0, y0, x1, y1, floor, exterior?, material).
- `doors[]`: position, floor, from-room → to-room, type (front, interior, garage, sliding, cellar), locked, keyId, barricadable.
- `windows[]`: position, floor, size, room, breakable, climbable, curtains.
- `rooms[]`: id, type, polygon/rect, floor, light (power, switch position), loot table id.
- `furniture[]`: id, kind, position, rotation, blocks movement?, container (slots, loot table), hiding spot?, light source?
- `stairs[]`, `ladders[]`, `hatches[]` (attic, cellar).
- `spawns`: zombies inside (count, rooms), survivor/NPC slots, pet slots.
- `poi`: name for signs and map (shop names generated, fictional).
- `sprites`: atlas ids for floor, walls (per orientation), roof, shadow, cut-away variants.
- Validation report and generator version.

---

## 7. Validation rules (so every generated house is playable)
- Every room reachable from an outside door (flood fill on the walk grid).
- At least two ways out of most houses (front + back door or window).
- No furniture blocks a door swing, window or stair.
- Minimum room sizes (bathroom 2×2 m, bedroom 3×3 m, kitchen 2.5×3 m).
- Windows only on exterior walls; no windows between rooms.
- Stairs connect existing floors; upper floor area ≤ ground floor.
- Every container reachable; loot tables exist in `src/data`.
- Walls on the 1 m grid; footprint inside the lot with the setback.
- Render check: sprite size matches the style sheet's cube test.

---

## 8. Ideas that make houses feel alive (gameplay + story)
- **Owner profiles**: nurse, cop, hunter, mechanic, farmer, gamer, prepper, old couple, student, family with kids — each changes decor and loot (prepper basement with canned food and a gun safe; hunter's rifle cabinet; nurse's med kit).
- **Notes and photos**: generated diary notes, fridge notes, family photos — small story hints and map clues ("keys under the mat", "went to the school shelter").
- **Spare keys**: under the mat, in a flower pot, on a key hook — tie doors and cars together.
- **Already looted** houses: open drawers, mess, fewer items; tells the story of other survivors.
- **Barricaded by survivors**: planks on windows, a survivor note, maybe a hostile or friendly NPC.
- **Infected household**: zombie family inside, locked bedroom with a scratching sound.
- **Alarm systems**: some houses trigger an alarm when you break in (noise draws zombies).
- **Power and water**: lights, fridge cold for a while, taps work until the grid fails (time-based).
- **Pets**: dog left in a backyard to rescue (fits the pet system), cat hiding upstairs.
- **Secret stashes**: loose floorboard, hidden safe behind a picture, cellar hatch under a rug.
- **Garage content**: car (maybe with keys), workbench, tools, fuel cans, bikes.
- **Upgrades as a base**: reinforce doors, metal shutters, rain collector, garden beds, generator in the garage.
- **Weather effects**: leaky roof in damaged houses, flooded basements, snow-covered yards.
- **Day-one remnants**: TV on with static, dinner on the table, packed suitcases by the door.

---

## 9. Vehicle generator (max list)

### 9.1 Vehicle families
Small car, hatchback, sedan, station wagon, coupe, sports car, SUV, pickup (single/double cab, long bed), van (panel, camper), minivan, police cruiser, ambulance, fire truck, school bus, city bus, box truck, semi tractor + trailer, tow truck, garbage truck, ice cream truck, food truck, tractor, quad bike, motorbike, scooter, bicycle, golf cart, forklift, army jeep, armoured truck, plus the goofy roster already in the game (rusty hatchback, clown car, bathtub buggy, golf cart, ice cream truck, hot dog van, roadster; World Bible §14, §38–§43).

### 9.2 Parametric body (one generator, many cars)
- Parameters: length, width, height, wheelbase, track, wheel size, ground clearance, cabin position and length, hood length, trunk/bed length, roof shape (sedan, hatch, wagon, van box, open bed), door count (2/4/sliding), bumper style.
- Blender builds the mesh from these (like Building Tools for cars): body shell, cabin glass, wheels, lights, mirrors, bumpers, roof rack.
- Kits for special types (police light bar, ambulance box, fire ladder, bus windows).

### 9.3 Parts that match the repair system
Every car is a set of **parts with slots** that the sim already uses (World Bible §15, Z7: engine, battery, spark plugs, radiator, tyres) plus doors, hood, trunk lid, windows, headlights, seats, fuel tank, muffler, brakes, suspension, radio, trunk/bed storage.
- Each part has a condition (good / worn / broken / missing) and **its own sprite layer**, so a missing door or a flat tyre is visible.
- Stats come from parts: engine → acceleration and top speed, tyres → grip, mass from body size, brakes → stopping, armour plates → protection.
- **Auto-derive `VehicleDef`**: the generator writes the game data (mass, speed, fuel, trunk size, seats, part slots) so new cars need no hand tuning.

### 9.4 Variety
- Paint from a muted palette + decal sets (racing stripes, company logos — fictional, police markings), rust and dirt amount, dents, broken windows, flat tyres.
- **Condition by place**: highway wrecks (burnt, crashed), parking-lot cars (locked, OK), garage cars (best condition, keys nearby), police cars (shotgun in the rack), ambulances (medical supplies).
- **Contents**: glovebox (map, snacks, gun sometimes), trunk loot by owner type, a zombie trapped in the driver seat.
- **Keys**: in the ignition, in the house next door, on a zombie nearby, or none (hotwire).
- **Plates**: generated fictional plates by town.

### 9.5 Rendering
- 16 directions (smooth turning), wheels as separate layers (steer and spin), doors open variants, lights as light sprites, brake lights, damage layers per part, shadow pass.
- Optional 32 directions for the player's car only.

### 9.6 Vehicle gameplay ideas
Hotwire mini-game, siphon fuel, push a dead car, jump-start with another car, tow with a rope or tow truck, trailers (World Bible §44), roof as a safe spot, horn draws zombies, car alarm trap, run-over damage, crash damage to parts, tyre blowouts, vehicle mods (ram plates, spikes, window grilles, roof rack, extra fuel tank, spotlight), radio stations, sleeping in the car, car as mobile storage, garage repair bench.

---

## 10. World-level generation ideas
- Towns by size pick a **building mix**: hamlet (houses, farm, gas station), village (+ diner, church, store), town (+ police, school, clinic, main street shops), city (+ apartments, mall, hospital, downtown towers from CityFoundry).
- **Traffic jam generator** on highways out of town (wrecks, abandoned cars, military roadblock) — great loot and danger.
- **Parking lots** generated next to shops and schools.
- **Story places**: crashed helicopter, overturned truck with supplies, evacuation camp, quarantine fence, burnt-out block.
- **Infection map**: some districts overrun, some quiet.
- **Loot balance by distance**: richer far from spawn, more zombies too.

---

## 11. CityFoundry improvement list
1. Add a **"Small Town" profile**: 1–2 storey houses, shops, garages on lots with yards (not only towers).
2. **Generate interiors**: walls, doors, windows, rooms per the pipeline in §3; ground floors of towers at least (lobby, shop, stairwell).
3. Snap everything to the **1 m grid**, storey 2.7 m, door 0.9 × 2.1 m (`STYLE_BIBLE.md` §0.5).
4. **Read jobs from the game** (town lots) instead of its own block grid; seed = building id.
5. Export the **`building.json` data contract** (§6) instead of / next to the tactical JSON.
6. Replace sniper perches with **survival data**: entry points, barricade spots, loot rooms, hiding spots, noise sources, zombie spawn rooms.
7. Add the **isometric sprite export** with the camera from `STYLE_BIBLE.md` §0.8 and layered output for the cut-away.
8. Add **damage states**, wear and district styles (suburban, downtown, industrial, ruined).
9. Add a **vehicle tab** using the parametric car generator (§9).
10. Add a **validation panel** (§7) that shows red/green per rule before export.
11. Keep the tool and its server inside `world4bros-core/tools/` and document how to start it (port 4040) in `OPEN_SOURCE_TOOLKIT.md`.

---

## 12. Snap-kit building ("build it like a building game") and suggest-and-fix
Owner idea (2026-10-20): build houses the way a player builds in a building game — snap walls, floors, doors and roof tiles together on a grid — and do it with code; the owner suggests a model ("merge these", "add a garage") and code fixes it.

**Yes, this works, and it is the best way to do it.** One snap kit does three jobs: the generator builds with it, the owner (or an AI agent) edits with it, and players use the same pieces in build mode (World Bible §26, §28, key `B`).

### 12.1 The snap kit
- **Grid**: 1 m tiles, walls sit on tile **edges** (north or west edge of a tile), floors and furniture on tiles, 2.7 m per storey (`STYLE_BIBLE.md` §0.5).
- **Pieces** (each one Blender model + isometric sprites + a data row): floor (wood, tile, carpet, concrete), wall (plain, window, door, double door, garage door, broken, boarded), wall corner and T-junction (auto-picked), stairs, railing, roof pieces (gable slope, ridge, hip corner, flat with parapet, edge), porch post, fence, column, chimney.
- **Snap rules**: walls connect at grid corners; corners and T-junctions are picked automatically from neighbours (autotile, like Zomboid/The Sims); doors and windows replace a wall piece in place; roofs are computed from the footprint, not placed by hand.
- **Data**: a building is just a list of pieces on the grid, `{ piece, x, y, floor, edge, rotation, material }`. That list *is* `building.json` (§6) — walls, doors and windows come straight from it.

### 12.2 Code builds with the kit
- The generator (§3) places pieces instead of modelling meshes: footprint → outer walls → room walls → doors/windows → floors → roof → furniture.
- Blender only has to model each **piece once**; any building is assembled from them, so art stays consistent and renders are cheap (render pieces, not whole houses; the game assembles them per tile, which also makes the cut-away and depth sorting correct).
- Players use the same pieces in build mode, so player bases and generated houses look and behave the same.

### 12.3 Suggest a model, code fixes it
How the owner (or an agent) changes or merges buildings:
1. **Suggest**: pick a building in a debug/editor view and say what you want — *"merge these two houses"*, *"add a garage on the left"*, *"make the kitchen bigger"*, *"turn this into a pharmacy"* — as a text command, by dragging pieces in an editor, or by giving a rough model (a Blender blockout or a sketch on the grid).
2. **Convert**: the suggestion becomes edit operations on the piece list (add room, remove wall, extend footprint, change archetype). A rough Blender model is snapped to the grid ("voxelized" into floor tiles and wall edges).
3. **Auto-fix (the "repair" pass)**, applied until the building passes §7:
   - close gaps in outer walls; pick correct corner/T pieces;
   - remove walls left inside a merged room, or add a doorway between merged parts;
   - make every room reachable (add a door on the shared wall with the shortest path);
   - add an outside door if none, a second exit if needed;
   - move furniture that blocks doors or windows;
   - fix window placement (outer walls only, sill height);
   - rebuild the roof from the new footprint;
   - re-pick loot tables if room types changed.
4. **Show the result**: before/after in the editor with a list of what was fixed; the owner accepts or undoes.
5. **Save** the fixed `building.json` as a **template** so the generator can reuse it (§4 template library) — hand-made quality, generated variety.

### 12.4 AI-assisted version
An agent (Antigravity/Gemini, Copilot or Claude) can take a plain-language request or a reference image, write the edit operations or a new piece list, and run the same auto-fix and validation. The validator — not the AI — decides if the building is playable, so bad suggestions get repaired or rejected, never shipped broken.

### 12.5 Prompt only, or reference images?
- **A prompt is enough for buildings made from the kit.** The style lives in the pieces and the style rules (`STYLE_BIBLE.md` §0), not in the AI. A prompt only decides the layout and choices — *"two-storey farmhouse, porch, attached garage, 3 bedrooms, worn, rural style"* — so every result comes out in the game's look automatically.
- **Prompt vocabulary** (what the editor/agent understands): archetype (§5), size or footprint, floors, rooms and counts, features (porch, garage, basement, attic, pool, fence), condition (§3 step 11), district style preset (`suburban`, `downtown`, `industrial`, `rural`, `ruined` = sets of materials and pieces), owner profile (§8).
- **Reference images help in three cases**:
  1. **A new style or new pieces** (e.g. a Victorian house, a gas station canopy): the kit does not have those pieces yet, so they must be modelled or generated first, and references keep them on-style.
  2. **Copying a specific layout**: a photo, a floor-plan sketch or a drawing on graph paper of a real building is easier than describing it.
  3. **Checking the look**: comparing renders with `screens/reference/` before approving new art.
- **Rule**: prompts never invent new art on the fly; they only use pieces that exist. If a prompt asks for something the kit lacks, the editor says which piece is missing and adds it to the art to-do list.

### 12.6 What to build for this
1. Piece registry in `src/data/buildingPieces.ts` (ids, footprint, edge/tile, blocks movement, sprite ids).
2. Piece-list ↔ `building.json` converter and walls/doors for the sim (replacing today's 4-wall boxes in `wallRects`).
3. Auto-fix + validator module in `src/sim/` (pure TypeScript, unit tested), so it runs in the game, in tests, in Node scripts and before Blender.
4. A simple **building editor page** (`?panel=building-editor` or a dev page) to place pieces, merge buildings, run auto-fix and export templates.
5. Blender: model each piece once at kit scale; `render_iso.py` renders pieces in both wall orientations.
6. Hook build mode (`B`) to the same piece registry.

## 13. Destruction test and orbit viewer (built 2026-10-20)
Owner asked: can we orbit the models, smash walls and make roofs fall, with colour? **Yes — tested and working.**

- **Script**: `tools/blender/destruction_demo.py` builds a coloured house (palette from `STYLE_BIBLE.md` §0.3): a brick front wall with a door and two windows, siding side and back walls, a gable roof, a sidewalk and a truck. Blender rigid-body physics: the truck drives through the brick wall, bricks, glass and the door fly, and the roof strips above the hole drop into the house onto the truck. Runs headless:
  `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b --factory-startup --python-exit-code 1 -P tools/blender/destruction_demo.py -- --out tools/viewer/models --frames docs/screens --wall fracture`
- **Outputs**: `tools/viewer/models/house_destruction.glb` (animated), `house_destruction.blend`, and isometric renders `docs/screens/destruction-before.png`, `-impact.png`, `-after.png` with the style-sheet camera.
- **Orbit viewer**: `tools/viewer/index.html` (three.js): drag to orbit, right-drag to pan, wheel to zoom, Play / Restart / scrub slider, "Iso view" button. With the dev server running (`npm run dev`), open `/tools/viewer/index.html`. Add new models to the `MODELS` list at the top of the script.

**What we learned (rules for future destruction assets):**
1. Never start rigid bodies overlapping (roof slabs meeting at the ridge, a roof touching the lintel): the solver launches them across the map. Leave small gaps.
2. A kinematic (animated) pusher must not wedge under walls or roofs; keep vehicles below the wall top.
3. Thin ground lets heavy pieces tunnel through; use a thick ground box.
4. Two loose roof slabs slide off a 35° pitch on their own; one rigid roof just bridges the hole. **Roof strips (truss-like 1 m sections) resting on the front and back walls** give a believable partial collapse.
5. Bricks should start awake but be allowed to sleep: bricks hanging over a hole then fall; the rest of the wall stays still.
6. Blender's "bake to keyframes" needs a window; headless scripts record each frame's transform and keyframe it themselves.
7. The glTF export writes one animation per object; the viewer plays all of them on one timeline.

**Cell Fracture walls (added 2026-10-20, now the default):**
- `--wall fracture` cuts the front wall into **Voronoi chunks** — the same method as Blender's Cell Fracture add-on, written into the script (`voronoi_cells`, `fractured_wall`) so it runs headless with nothing to install (Cell Fracture is a separate extension since Blender 4.2 and was not installed). The wall is split into solid panels around the door and windows; seed points are denser near the impact, so the breach shatters into small debris while the corners stay as big slabs. `--wall bricks` gives the first brick version.
- **Glue** (`glue()`): breakable "Fixed" rigid-body constraints between touching chunks and to the ground/side walls (the same idea as "Connect" after Cell Fracture). Loose Voronoi chunks don't stack — without glue a third of the wall fell before the truck arrived.
- **Glue tuning that worked**: ties to the ground and side walls strong (60), chunk-to-chunk ties weaker (30). Result: nothing moves before the hit; ~86 of 114 chunks break loose; the roof strips over the breach drop 0.5–1.1 m onto the truck; the outer strips stay up.
  - Strong chunk ties (60) → the band over the breach hangs like a beam and the roof does not sag.
  - Weak side ties (12) or none → the wall slumps under the roof before the truck arrives.
- Runtime: about 20 s headless for 114 chunks and ~330 constraints, 140 frames.

**Slicing (owner question 2026-10-20):**
- *Cutting 3D models into pieces* is done in Blender: **Bisect** (cut along a plane), **Knife**, **Boolean** modifier (cut one shape out of another, e.g. a hole), and the **Cell Fracture** add-on (shatter into random chunks; comes with Blender as an extension). Use these to make breakable walls instead of hand-placed bricks.
- *Cutting one big image into sprites* is Photoshop's **Slice** tool. We don't need it: `render_iso.py` renders each piece/frame separately and the atlas packer (`OPEN_SOURCE_TOOLKIT.md` §3) packs them; a script can also cut a render into per-tile pieces.

**How this reaches the game (2D isometric):**
- Pre-render destruction as **sprite sequences** per building piece and event (wall breach, roof sag, window smash), using the snap kit (§12) so each wall piece has intact / cracked / breached / rubble frames.
- The sim decides *when* (vehicle impact, explosion, zombie horde on a barricade) and swaps pieces to damaged states, spawns rubble props, opens new paths (a breached wall becomes a doorway), and lowers roof pieces over breached spans so the cut-away still works.
- Keep real-time physics out of the game for now; use Blender physics to author the frames.

## 14. Order to make it playable (MVP first)
0. **Snap kit first** (§12): piece registry, piece list ↔ `building.json`, auto-fix and validator in the sim.
1. **Sim first, art later**: interior walls, doors and windows in the sim from a `building.json` for 3 house templates — playable in the current renderer.
2. Rooms with loot containers and search; doors (open, lock, kick, barricade); windows (break, climb, board).
3. Floor-plan generator (templates → treemap) + validation, for cottage, family house, townhouse.
4. Blender: build the same houses from `building.json`; render layered isometric sprites (vertical slice I2).
5. Commercial buildings (gas station, grocery, pharmacy, police) with room templates.
6. Parametric vehicle generator → `VehicleDef` + part layers; 16-direction renders.
7. Story layer, damage states, alarms, keys, notes.
8. CityFoundry upgrades (§11) for downtown blocks.
