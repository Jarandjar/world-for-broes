# Building Generator & Pipeline Plan (Checklist)

Status: proposed  
Scope: procedural low-poly building generation, batch list-driven creation, validation, headless exports, and game-world integration.

## Goals
- Turn `tools/blender/generate_building.py` into a reproducible production pipeline.
- Support list-driven batch generation from JSON/CSV with deterministic seeds.
- Export game-ready `.glb/.fbx/.obj` assets with consistent pivots, naming, and metadata.
- Add quality gates so broken geometry and invalid assets fail early.

## Phase 0 — Foundation
- [ ] Freeze asset conventions (units, axis, scale, pivot, naming).
- [ ] Define archetypes (`house`, `apartment`, `office`, `warehouse`, `shop`).
- [ ] Define style profiles (`suburban`, `downtown`, `industrial`, `ruined`).
- [ ] Add parameter bounds and defaults for all generator controls.

## Phase 1 — Generator Core
- [ ] Add lot-aware footprint generation and snap-to-grid.
- [ ] Add shape templates (rectangle, L, U, courtyard, tower).
- [ ] Add floor-core logic (stairs/elevator shafts where applicable).
- [ ] Add weighted random choices with deterministic per-building seeds.
- [ ] Add regenerate-selected flow in Blender UI.

## Phase 2 — Facade/Roof Systems
- [ ] Add facade modules (window bays, balconies, fire escapes).
- [ ] Add storefront/entry variants for ground floors.
- [ ] Add roof systems (flat/gable/hip/mansard).
- [ ] Add rooftop clutter placement with spacing/collision rules.
- [ ] Add decay variants (cracks, boarded windows, patch geometry placeholders).

## Phase 3 — List-Driven Batch Generation (MVP)
- [ ] Define JSON schema for batch jobs.
- [ ] Add CSV importer mapped to schema fields.
- [ ] Validate input rows and emit per-row errors.
- [ ] Generate one building per row with per-row seed and output ID.
- [ ] Write generation report (`success`, `failed`, `reason`, `outputPath`).

## Phase 4 — Export Pipeline
- [ ] Add one-click export to `.glb`, `.fbx`, `.obj`.
- [ ] Normalize origin/pivot (bottom-center default).
- [ ] Apply transforms and clean object hierarchy before export.
- [ ] Add optional collision mesh export (`_col`).
- [ ] Emit export manifest JSON (bbox, tris, lods, source params).

## Phase 5 — Validation & QA Gates
- [ ] Add geometry checks (non-manifold, flipped normals, n-gons policy).
- [ ] Add scale/bounding checks against style metrics.
- [ ] Add UV checks (overlap policy + texel density policy).
- [ ] Add naming lint checks.
- [ ] Add fail-fast mode for CI/headless runs.

## Phase 6 — Headless Automation
- [ ] Add headless Blender command wrapper (`blender -b -P ...`).
- [ ] Add batch generation command with path/seed/profile overrides.
- [ ] Add incremental mode (skip unchanged rows/seed outputs).
- [ ] Add timing logs and crash-safe checkpoints.
- [ ] Add nightly CI job for regeneration and artifact upload.

## Phase 7 — Game Integration
- [ ] Define import contract for engine-side placement metadata.
- [ ] Add district/zoning tags and lot-fit metadata fields.
- [ ] Add roof-cut/interior visibility helper markers if needed.
- [ ] Add destruction-state variant slots (`clean`, `damaged`, `ruined`).
- [ ] Validate runtime spawning from generated manifest.

## Phase 8 — Team Workflow
- [ ] Add preset library (`/tools/blender/presets`).
- [ ] Add review checklist for generated assets.
- [ ] Add approved seed catalog for stable production sets.
- [ ] Version the generator behavior and migration notes.
- [ ] Document RFC process for new archetypes.

## Job File Format (Draft)
Use this as the initial JSON batch input shape.

```json
{
  "version": 1,
  "defaults": {
	"style": "downtown",
	"floors": 8,
	"width": 18,
	"depth": 14,
	"seed": 1001,
	"setback": true,
	"roofClutter": true,
	"exportFormats": ["glb"]
  },
  "jobs": [
	{
	  "id": "districtA_blk02_lot07",
	  "style": "industrial",
	  "floors": 5,
	  "width": 24,
	  "depth": 16,
	  "seed": 22017,
	  "setback": false,
	  "roofClutter": true,
	  "exportFormats": ["glb", "fbx"]
	}
  ]
}
```

## Free Asset/Reference Links
- Poly Haven (CC0 textures): https://polyhaven.com/textures
- ambientCG (CC0 materials): https://ambientcg.com
- TextureCan: https://www.texturecan.com
- Kenney assets: https://kenney.nl/assets
- Quaternius assets: https://quaternius.com
- Poly Pizza: https://poly.pizza
- Sketchfab downloadable filter: https://sketchfab.com/feed?features=downloadable
- Terrain Party heightmaps: https://terrain.party
- OpenTopography: https://opentopography.org
- USGS EarthExplorer: https://earthexplorer.usgs.gov
- ESA WorldCover: https://worldcover2021.esa.int
- OpenStreetMap export: https://www.openstreetmap.org/export

## Research: Tools and Generators to Learn From (2026-10-20)
See also [OPEN_SOURCE_TOOLKIT.md](OPEN_SOURCE_TOOLKIT.md) for headless Blender commands, licence rules (GPL tools stay in `tools/`, never in `src/`) and generators for characters, trees, props and atlas packing. **Building Tools** ([ranjian0/building_tools](https://github.com/ranjian0/building_tools/wiki), MIT) is the best open code to base the house generator on.

Licences were not all confirmed; check each one before using code or assets.

### Blender building generators
| Tool | What it does | Licence / cost | How it helps us |
| :--- | :--- | :--- | :--- |
| [Buildify](https://www.cgchannel.com/?p=137897) (Pavel Oliva) | Geometry Nodes library: modular walls and flat roofs on any footprint; works with Blosm for whole cities | Free; licence not confirmed | Study how it turns a footprint into floors and wall modules; good for downtown blocks |
| [Auto-Building](https://github.com/Renart84/Auto-Building/blob/main/docs/source/index.rst) ([docs](https://auto-building.readthedocs.io/en/latest/_sources/index.rst.txt)) | Geometry Nodes add-on that scatters windows, doors and balconies on a low-poly mass; v1.2 adds roofs, interiors and UVs | GitHub; Blender 4.5+ | Closest open example of "mass model → facade parts"; worth reading its node setup |
| [BuildingNodes](https://github.com/durman/buildingnodes) | You model facade panels, define a style in a node editor, apply to a base mesh | GitHub; no roof tools | Good idea: facade "styles" as data applied to any mass |
| [Blosm](https://github.com/vvoovv/blosm) ([docs](https://github.com/vvoovv/blosm/wiki/Documentation)) | Imports OpenStreetMap buildings (heights, floors, many roof shapes) and roads with widths | Source GPL; base version pay-what-you-want | Import a real small town as **reference** for lot sizes, footprints, roof types and street widths. OSM data is ODbL, so use it for reference, not shipped map data, unless we follow ODbL. |
| Paid: [Procedural Building Generator](https://superhivemarket.com/products/procedural-building-generator-for-blender), [Geo-Buildings](https://superhivemarket.com/products/geobuildings), [BRiCKiTECT](https://superhivemarket.com/products/brickitect), [SCI-FIX](https://superhivemarket.com/products/scifix/docs) | Geometry Nodes building kits | Paid | Only if we want to buy time; not needed |

### Rendering 3D to isometric sprites
| Tool | Notes |
| :--- | :--- |
| [Omnidirectional Isometric Render](https://itch.io/t/4776963/free-blender-addon-omnidirectional-isometric-render) | Free add-on: renders an object from several isometric directions, to single images or a sprite sheet (installs Pillow). Licence not stated. |
| [SpriteMaker](https://superhivemarket.com/products/spritemaker) | 4 / 8 directions; GPL; about $5. |
| [Pre Render Creator](https://viniguerrero.itch.io/pre-render-creator), [8 Directions Render Plugin](https://auteddy.itch.io/8-directions-render-plugin-for-blender), [Sprite Render Kit](https://superhivemarket.com/products/sprite-render-kit), [Character Renderer](https://dener.itch.io/character-renderer-for-2d-games) | Paid alternatives with presets and sheet packing. |
| Our own `render_iso.py` | Recommended: ~150 lines of Python with the exact camera, sun, ortho scale and naming from `STYLE_BIBLE.md` §0.8, so every asset matches the game. Use the add-ons above to compare results. |

### How Project Zomboid does it
- Artists build most tiles in 3D (Maya + Photoshop), render them with a toon shader and fixed lighting, and get all facing directions "for free" ([PZ blog](https://projectzomboid.com/blog/news/2023/05/mod-spotlight-daddydirkiedirks-tiles/)).
- Tiles go into tilesheets plus a `.tiles` definition file that sets behaviour (floor, wall, container…) ([PZwiki: Adding new tiles](https://pzwiki.net/wiki/New_Tiles)). Maps are built in **TileZed**, their fork of the Tiled editor.
- Characters were made from 3D models turned into 2D frames, later generated on the graphics card ([Indie Stone post](https://theindiestone.com/binky/wp-json/wp/v2/posts/440)); walls and floors moved from drawn sprites to textured polygons for speed ([Indie DB](https://www.indiedb.com/news/iso-revolution)).
- **Lesson for us**: 3D source → fixed camera and light → sprites, plus a definition file per tile/piece with its game behaviour. That is exactly the split between Blender output and our `src/data/*` registries.

### Town and road generators
| Project | Idea to borrow |
| :--- | :--- |
| [Watabou Medieval Fantasy City Generator](https://watabou.itch.io/) ([itch post](https://itch.io/e/629037/watabou-published-medieval-fantasy-city-generator)) | Open source (Haxe/OpenFL), maps free to use: wards/districts, organic streets, lots. Its Village Generator shows spread-out buildings and farmland edges. |
| [gen-city](https://github.com/neki-dev/gen-city) ([npm](https://npmjs.com/package/gen-city)) | TypeScript, seeded street grids with tunable turn/intersection chance and building spacing — directly portable ideas for `src/sim/worldgen.ts`. |
| [agent-q1/Cities](https://github.com/agent-q1/Cities) | Apache 2.0: sectors → long roads → lots and small streets around settlement centres. |
| [twymer/town-gen](https://github.com/twymer/town-gen), [TownSim paper](https://www.pcgworkshop.com/archive/song2019agentbased.pdf) | Agent-based road growth: naturalistic, non-grid towns. |
| [pjreddie procedural city](https://pjreddie.com/projects/procedural-city-generation/) | Voronoi districts → blocks → lots → buildings. |
| [Infinite WFC city](https://gigazine.net/news/20181116-infinite-procedurally-generated-city) | Wave Function Collapse for tiled building/city pieces — good for roof and block variety. |

### Building rules (shape grammar)
- **CGA shape grammar** ([Müller et al., "Procedural Modeling of Buildings"](https://course.ccs.neu.edu/cs5150f13/readings/muller_buildings.pdf); used in [CityEngine](https://doc.arcgis.com/en/cityengine/latest/get-started/get-started-rule-based-modeling.htm)): mass model → split into floors → split facades into bays → windows, doors and ornaments. Structure our generator this way so new styles are rule data, not new code.

### Isometric rendering in PixiJS
- [iceoh](https://github-redirect.dependabot.com/sammccord/iceoh): zero-dependency TypeScript isometric/2D tilemap helper built with PixiJS in mind; returns a depth value for sorting.
- [pixi-tile-utilities](https://unpkg.com/pixi-tile-utilities@4.5.3/README.md): isometric picking, collision and **Tiled** map import.
- Depth sorting: one sort list for tiles and entities by `x + y` plus a height term ([common fix](https://bugnet.io/blog/how-to-fix-isometric-tilemap-sprite-depth-sort-overlap)).
- **Tiled editor** (isometric maps) is a good way to hand-build special places (army base, gas stations) on top of the generator, like PZ's TileZed.

### Free assets to start from
- **Kenney** (CC0): [Isometric Buildings 1](https://opengameart.org/content/isometric-buildings-1) (2D sprites), [Modular Buildings](https://kenney-assets.itch.io/modular-buildings) and [Building Kit](https://kenney-assets.itch.io/building-kit) (3D, OBJ/FBX/glTF), City Kits ([industrial](https://gamedev.net/news/5434-city-kit-industrial/)). Clean and cartoony; useful as greybox/blockout pieces, then re-texture gritty.
- **Quaternius**: packs listed as CC0 (e.g. [Medieval Village MegaKit](https://quaternius.com/packs/medievalvillagemegakit.html), [LowPoly Buildings](https://opengameart.org/content/lowpoly-buildings-pack)); a newer Quaternius licence (no resale of the raw assets) is reported for new packs ([guide](https://app.cinevva.com/de/guides/free-rigged-3d-character-models)) — check each pack.
- **Mixamo** animations: royalty-free for games per Adobe's FAQ as quoted [here](https://discussions.unity.com/t/asset-rejected-due-to-licenses-required-for-third-party-assets/871153); service is in maintenance mode. Check Adobe's current terms for rendered sprite use before shipping.

## Improvement List for This Plan (2026-10-20)
1. **One source of truth for sizes**: Phase 0 should not re-decide units and scale. Use `STYLE_BIBLE.md` §0.5 (scale sheet) and §0.8 (Blender camera): 1 unit = 1 m, storey 2.7 m, door 0.9 × 2.1 m, origin at ground centre. The current generator defaults (3.0 m floors, 2.0 × 2.3 m door) must change.
2. **The game drives the job list**: instead of hand-written JSON/CSV, add a small Node script (`npm run art:jobs`) that exports every town lot from `townBuildings()` in `src/sim/worldgen.ts` (building def, width, depth, door side, yard) to `tools/blender/jobs/*.json`. Blender builds exactly what the game places; seed = building id. Hand-written jobs stay possible for special buildings.
3. **Output is sprites, not only meshes**: this is a 2D isometric game. Add a phase **"Render to sprites"** after export: `render_iso.py` renders each building with the §0.8 camera and writes PixiJS atlases to `public/assets/buildings/`. Keep `.glb` as the editable source; `.fbx`/`.obj` are optional.
4. **Render in layers for the Zomboid cut-away**: per building and per door orientation (N/E/S/W), render (a) floor + interior + back walls, (b) front walls, (c) roof, (d) shadow pass. The game hides (b) and (c) when the player is inside or behind. Alternatively render modular per-tile pieces; decide in the vertical slice (I2).
5. **Small-town first**: start with the archetypes the game already uses (`src/data/buildings.ts`: cottage, log cabin, townhouse, corner shop, supermarket, auto garage, clinic, warehouse, tool shed, survivor shack, apartment block) and their roof types (gable, hip, flat, tin, broken). Skyscrapers with setbacks come later for metro downtowns.
6. **Shape-grammar structure** (CGA style): footprint → mass → floors → facade bays → elements, with rules in JSON per style (`suburban`, `downtown`, `industrial`, `ruined`). New styles = new rule files.
7. **Kit pieces over raw boxes**: assemble buildings from modular 1 m-grid pieces (wall, window wall, door wall, corner, roof segments) like Kenney/Quaternius kits. Easier texturing, cleaner silhouettes, and the same pieces work for per-tile rendering.
8. **Materials and grime**: CC0 textures from Poly Haven / ambientCG, palette-locked to `STYLE_BIBLE.md` §0.3, colour management "Standard", plus a grime/edge-wear pass and decal slots (stains, graffiti, boarded windows).
9. **More damage states**: use the World Bible §26 settlement conditions — pristine, worn, burnt, flooded, collapsed, overgrown, barricaded — instead of only clean/damaged/ruined.
10. **Metadata contract with the sim**: export per building the footprint, door side and position (`doorAt`), window positions, room rectangles and container slots, so `wallRects`, doors, windows and loot in the game match the art exactly.
11. **Interior floor plans**: generate rooms from templates (kitchen, bedroom, bath, living, storage per World Bible §26) inside the footprint, with doors between rooms; render furniture as separate props.
12. **Automatic QA on renders**: render a 1 m reference cube and a 1.8 m figure in every batch and check their pixel sizes (256 × 128 diamond, ~157 px per metre at 2×); image-diff renders against approved ones to catch regressions.
13. **Headless runs from npm**: `npm run art:buildings` → `blender -b -P tools/blender/batch.py -- --jobs …`. Run locally; CI only if a Blender build is available (it is large).
14. **Asset storage**: big PNG atlases may need Git LFS; decide before committing hundreds of renders.
15. **Tests without Blender**: validate the job schema and the metadata contract with a normal Node test (`tests/building-jobs.test.ts`), since Python tests did not run and `roadmap_agent.py` does not exist in this repo.
16. **Fit the master plan**: this pipeline is step **I3 (sprite pipeline)** of the isometric redo in `HD_GRAPHICS_PLAN.md` §0.1b; build only what the I2 vertical slice needs first (2–3 houses, one shop), get approval, then batch the rest.

## Extra Ideas
- **Lit-window patterns** at night chosen per building seed; some flicker (power cuts).
- **Procedural shop signs** with generated (fictional) names and faded paint.
- **Seasonal variants**: snow on roofs and ledges, autumn leaves in gutters.
- **Barricade overlay layer** drawn from sim state (boarded windows, planks, sandbags) instead of baked into the building.
- **Graffiti and posters** as decal sprites placed by district (survivor messages tie into the story).
- **Roof clutter for small towns**: chimneys, satellite dishes, solar panels, TV aerials, skylights — not only city water towers.
- **WFC for roofs and blocks**: let Wave Function Collapse pick matching roof pieces and block fill for variety without breaking rules.

## Playable Modular Interior & Door Traversal Ideas
- **Door-first runtime entities**: treat each door as a gameplay entity with open/closed/locked/broken state, interaction point, and nav-cost update hooks.
- **Unified piece graph**: represent walls, doors, floors, stairs and room portals in one graph so generator, editor and NPC pathing read identical data.
- **Interior reachability validator**: hard gate that checks every room is reachable from the main entrance and that larger buildings have a secondary exit.
- **Auto-repair pass for merges**: when two buildings are merged, remove duplicate interior walls, insert connector doors and rebuild room connectivity.
- **Door clearance rules**: reserve swing/clearance tiles so furniture, props and clutter never block traversal.
- **Exterior-only window enforcement**: interior partitions cannot host windows; only perimeter walls can receive window pieces.
- **Stair/vertical connector model**: encode stairs as explicit floor-to-floor links with landing clearance checks for NPC routing.
- **Room semantics layer**: tag rooms (`kitchen`, `bedroom`, `storage`, `shop_floor`) and drive loot, spawn logic and NPC schedules from tags.
- **NPC door interaction behavior**: approach handle point, open/wait/pass/close cycle, with fallback to alternate routes if blocked.
- **Dynamic nav updates**: when door state changes, patch local nav graph (not full rebuild) for responsive AI indoors.
- **Roof/cutaway sync**: wall fade and roof-lift logic should follow occupancy + camera direction so indoor gameplay remains readable.
- **Template promotion workflow**: after edit/repair/approval, save as reusable template with randomizable facade-only variation.
- **Before/after diff tooling**: show piece-level changes and validator fixes so designers can accept/undo with confidence.
- **Playable test map set**: maintain small benchmark maps specifically for indoor traversal, choke points and multi-floor door routing.
- **Regression tests**: add automated checks for room reachability, door passability, no blocked exits and deterministic piece assembly.

## Prioritized Sprint Checklist (M1 → M3)

### M1 — Modular Building Data + Runtime Assembly (Foundation)
Goal: switch from implicit outer walls to explicit snap pieces, while keeping compatibility.

- [ ] Finalize piece schema (`wall`, `door`, `floor`, `stair`, `portal`) with optional metadata (`exterior`, `roomId`, `flags`).
  - Files: `src/sim/worldgen.ts`, `src/data/schema.ts`
- [ ] Generate modular pieces for every town building definition and static placed building.
  - Files: `src/sim/worldgen.ts`, `src/sim/world.ts`
- [ ] Ensure save/load persists piece lists and legacy saves still load.
  - Files: `src/sim/staticmap.ts`, `src/sim/game.ts`
- [ ] Keep wall collision derived from piece list first, legacy fallback second.
  - Files: `src/sim/worldgen.ts`, `src/sim/world.ts`
- [ ] Add/expand baseline tests for deterministic piece assembly and door openings.
  - Files: `tests/buildings.test.ts`, `tests/staticmap.test.ts`

### M2 — Door-Aware Navigation + Validator/Repair (Playability Gate)
Goal: guarantee characters and NPCs can enter and traverse interiors safely.

- [ ] Build room/door connectivity graph from pieces (including vertical connectors for stairs).
  - Files: `src/sim/worldgen.ts` (or new `src/sim/buildingGraph.ts`)
- [ ] Implement validator checks: reachable rooms, at least one valid entrance, no blocked door clearances.
  - Files: new `src/sim/buildingValidator.ts`, tests in `tests/buildings.test.ts`
- [ ] Add auto-repair pass: fill gaps, remove duplicate merge walls, insert connector doors.
  - Files: new `src/sim/buildingRepair.ts`, integrated in generation flow
- [ ] Enforce exterior-only windows and interior door placement rules.
  - Files: `src/sim/worldgen.ts`, `src/data/buildings.ts`
- [ ] Add regression suite for merge/repair and traversal edge-cases.
  - Files: `tests/buildings.test.ts`, new `tests/building-validator.test.ts`

### M3 — Editor MVP + NPC Indoor Behavior (Usability)
Goal: enable fast iteration and realistic indoor NPC behavior using the same modular kit.

- [ ] Add simple grid editor (select/place/rotate/delete snap pieces) with before/after diff.
  - Files: `src/ui/ui.ts` (new panel), optional new `src/ui/buildingEditor.ts`
- [ ] Add "Validate & Repair" action and human-readable fix report.
  - Files: `src/ui/ui.ts`, `src/sim/buildingValidator.ts`, `src/sim/buildingRepair.ts`
- [ ] Implement door interaction behavior for NPCs (approach/open/pass/fallback).
  - Files: `src/sim/colony.ts`, `src/sim/game.ts`, `src/sim/components.ts`
- [ ] Add dynamic nav patching when door state changes (avoid full rebuild).
  - Files: `src/sim/game.ts`, `src/sim/world.ts`, nav helpers module
- [ ] Add indoor behavior tests (NPC enters building, crosses rooms, exits).
  - Files: new `tests/npc-indoor-pathing.test.ts`, `tests/colony.test.ts`

### Delivery Rule
- M1 must be complete and stable before M2.
- M2 validator must gate templates before any M3 editor publish action.
- M3 can ship incrementally, but only with validator/repair always available.

## Suggested Execution Order (Practical)
1. Phase 0 + 3 MVP (list-driven generation)
2. Phase 4 export pipeline
3. Phase 5 QA checks
4. Phase 6 headless automation
5. Phase 1/2 quality expansion
6. Phase 7 integration + Phase 8 workflow hardening
