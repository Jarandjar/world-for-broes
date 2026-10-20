# Asset Pipeline: How We Make Every Game Object

The plan for turning free/open models and our own Blender generators into World 4 Bro's art. The full list of what to make is [ASSET_CATALOG.md](ASSET_CATALOG.md) (generated: `npm run art:catalog`, 456 assets on 2026-10-20). Sources and licences: [OPEN_SOURCE_TOOLKIT.md](OPEN_SOURCE_TOOLKIT.md). Buildings: [GENERATION_IDEAS.md](GENERATION_IDEAS.md) §12. Written 2026-10-20.

---

## 1. Our style (the rules every asset must pass)
From `STYLE_BIBLE.md` §0 — the owner's decision of 2026-10-20:
- **Look**: gritty survival like **Project Zomboid**, with the bold readable streets and cars of **GTA 1–2 / Hotline Miami**. Muted greys, greens, browns, rust; nothing looks new.
- **Camera**: isometric 2:1 (128 × 64 px tile at 1×, 78 px per metre of height), drawn at 2×. Blender ortho camera at 30° elevation, 45° turn (§0.8).
- **Light**: sun from the north-west; roof 100 %, left wall 85 %, right wall 62 %; soft shadow to the lower-right.
- **Scale**: 1 tile = 1 m; storey 2.7 m; door 0.9 × 2.1 m; player 1.8 m; car 4.5 × 1.8 m (§0.5).
- **Rendering**: painted, not pixel art; thin outline only on characters, vehicles and interactables.
- **Tone**: mild blood allowed (splats, wounds), no extreme gore.
- **Runtime**: PixiJS in the browser; Blender (and optionally Unreal) only make the art.

## 2. The three production lanes
Every catalog entry goes through one of these. The catalog's "source" column says which.

| Lane | Use for | How |
| :--- | :--- | :--- |
| **A. Download + tweak** | props, furniture, items, vehicles (base), rocks | Take a CC0 model (Kenney, Quaternius, Poly Pizza, Poly Haven), then run the **"make it ours" pass** (§3). Fast and safe. |
| **B. Generate** | buildings (snap kit), trees, ground tiles, roads, damage states | Our Blender scripts: snap-kit pieces, `destruction_demo.py`-style fracture for broken states, Sapling/tree-gen for trees, CC0 textures baked to iso tiles. Everything seeded and repeatable. |
| **C. Rig + animate + render** | survivors, zombies, animals, clothing layers | Base body from MPFB 2 (CC0 assets) or the UE5 mannequin; animations from Quaternius' CC0 Universal Animation Library (incl. zombie locomotion) or Mixamo (private use only, no raw redistribution); render 8 directions × all animations; clothing as separate layers. |

AI (Gemini) is for **concept frames and 2D decals/icons only** — never for things that need 8 or 16 matching directions.

### 2b. Which tool: Unreal-first for Arena, Blender for scratch models and web sprites (owner decision 2026-10-20)

**For the Unreal Arena (Phase Z34): Unreal first.**
Download, tweak inside Unreal, and test directly against the arena bots (the project is built on the 5.7 First Person template, Arena Shooter variant; `UE_PIPELINE.md` §1):

| Job | In Unreal |
| :--- | :--- |
| **Recolour, grime, rust, wear** | Material Editor: one master grime material (`UE_ARENA_PLAN.md` U10) |
| **Small mesh edits** (cut, bend, merge, simple props) | Modeling Mode, built into Unreal Editor |
| **Characters** | The template's UE5 mannequins with gritty material instances and CC0 accessories; CC0 animations (Quaternius Universal Animation Library, incl. zombies) via the IK Retargeter (U12). MetaHuman after the demo. |
| **Posing and animation fixes** | Control Rig and Sequencer |
| **Breaking walls and props** | Chaos Fracture (Fracture Mode) |
| **Placing whole maps** | `npm run ue:arena` (`tools/unreal/pipeline/build_arena.py`, tested), later an art-dressing script over the blockout (U11) |

**Blender is reserved for:**
1. **Making new models from scratch**: Unique guns (potato cannon, nailgun SMG, slingshot), vehicles, and custom props. Modeling Mode can do basic edits, but Blender is vastly faster for precision polygon modeling and UV unwrap.
2. **Rigging new creatures**: Non-human rigs that do not fit the standard human mannequin skeleton (zombie dogs, quadrupeds, mutant beasts).
3. **The web game's isometric sprite sheets**: The PixiJS 2.5D browser game engine relies on `tools/blender/render_iso.py` for automated multi-angle sprite baking (`STYLE_BIBLE.md` §0.8). If the entire project ever moves permanently to Unreal, sprite rendering goes away and Blender serves strictly as a 3D modeling tool.


## 3. The "make it ours" pass (turn any download into a World 4 Bro's asset)
One Blender script per category (`tools/blender/make_ours_<category>.py`), run headless, so every asset gets the same treatment:
1. **Import** (glTF/FBX/OBJ) into the template scene with the style camera and sun (`STYLE_BIBLE.md` §0.8).
2. **Scale** to the scale sheet; set the origin at the ground contact point; snap the footprint to the 1 m grid.
3. **Simplify** shapes that are too small to read at 1× (merge tiny parts, keep a strong silhouette).
4. **Re-material** from our shared material library (`tools/blender/materials.blend`, to make): palette colours from §0.3, colour management "Standard", plus a **grime/wear pass** (edge wear, dirt at the bottom, rust on metal).
5. **Variants**: colour variants from the palette; **damage states** (dented, broken, burnt) by script — Voronoi fracture for breakables, bent/missing parts for vehicles, blood decals (mild) where it fits.
6. **Rename** to the game id from the catalog (`kind_id_variant_dir_frame`).
7. **Render** with `render_iso.py` (directions, frames, layers from the catalog's "make" column) and **pack** atlases into `public/assets/<category>/`.
8. **Log** the source, author and licence in `CREDITS.md`; set the entry's status to `done` (the catalog script will read a `public/assets/manifest.json` once it exists).

## 3b. Unreal Master Material Suite & Asset Hardening
For all 3D assets in Unreal (Phase Z34):
- **Master Material (`M_Master_Grime`)**: Parameterized material featuring BaseColor, Normal, packed ORM (Occlusion, Roughness, Metallic), world-aligned bottom contact dirt gradient (0.5m), edge-wear mask, rust overlay, and mild blood decal reception.
- **District Instances (`MI_`)**: Parameter presets for Industrial (heavy rust, oil, grease), Suburban (weathered wood, peeling siding), and Ruined (charred concrete, broken glass).
- **Damage Variant Standard**:
  - `_Clean`: Intact baseline asset.
  - `_Damaged`: Chipped, dented, or worn variant.
  - `_Ruined`: Chaos fractured geometry collection or rubble pile.
- **Mesh Import Presets**: Scale ($1\text{m} = 100\text{uu}$), Collision (BlockAll on solid walls, crates, sandbags; Overlap on foliage canopies), Nanite enabled on all static geometry $\ge 1,000$ triangles.

## 3c. Web-Game Coexistence Plan
1. **Isolated Blender Sprite Pipeline**: `tools/blender/render_iso.py` and `tools/blender/pipeline_runner.py` operate completely independently of Unreal, producing 2.5D PixiJS sprites into `public/assets/`.
2. **Shared Source Assets**: Net-new 3D models created in Blender (weapons, unique props, vehicles) are stored as canonical `.glb`/`.fbx` source files in `tools/blender/` that feed both the Blender isometric sprite baker and Unreal through `npm run ue:import -- <manifest.json>` (size-checked; glTF in metres, never scaled by 100; `UE_PIPELINE.md` §4.4). The old import folder `tools/unreal/World4Bros/Content/Import/` is retired with that project.
3. **Asset Conversion Checklist**:
   - **Unreal-Ready**: Static Mesh (`SM_`), proper collision hull, metric scale, `M_Master_Grime` instance applied.
   - **Sprite-Ready**: 8/16 direction turntable rendered with NW lighting (`render_iso.py`), transparent PNG, anchor point recorded in `manifest.json`.
4. **Future Prototyping**: If Unreal Engine becomes the sole platform after Decision Gate 1, explore using Unreal's Movie Render Queue (MRQ) with an orthographic camera if 2D sprite exports are ever needed.

## 4. Download list (start here)
Check each licence page before downloading; record it in `CREDITS.md`.
| What | Pack | Covers in the catalog |
| :--- | :--- | :--- |
| Furniture | Kenney Furniture Kit (CC0) | Furniture (34) |
| Building pieces (blockout) | Kenney Building Kit / Modular Buildings (CC0) | Snap kit reference |
| Town props, roads | Kenney City Kit (Roads, Suburban, Industrial) (CC0) | World props, Roads & streets |
| Vehicles | Kenney Car Kit (CC0) | Vehicles (7) as base shapes |
| Survival items, weapons | Kenney Survival Kit, Weapon Pack (CC0); Quaternius packs (check licence per pack) | Items, attachments |
| Animals | Quaternius animated animal pack (check licence) | Animals (7) |
| Bodies | MPFB 2 (assets CC0) / UE5 mannequins (Unreal templates) | Characters, zombies |
| Animations | Quaternius Universal Animation Library 1 + 2 (CC0, incl. zombies); Mixamo (Adobe terms, no raw redistribution) | Characters, zombies |
| Textures | Poly Haven, ambientCG (CC0): asphalt, concrete, brick, siding, wood floor, tile, grass, dirt, sand, snow, rust, metal | Ground tiles, roads, snap kit, re-materials |
| Trees | Sapling Tree Gen / tree-gen (generators, output ours) | Vegetation |

Unreal-only sources (City Sample kits) are **not** for shipped sprites in this non-Unreal game unless the licence allows it; use them as reference. Megascans are paid since 2025 (also inside Unreal), except a few free samples.

## 5. Order (what to make first)
1. **Vertical slice set (step I2, ~45 assets)**: grass, dirt, asphalt, sidewalk tiles; kerb, lane line, crosswalk; snap-kit siding + brick walls (plain, window, door), wood floor, gable roof pieces; 10 furniture (bed, wardrobe, fridge, stove, sink, toilet, sofa, table, chair, cupboard); 1 car (rusty hatchback); survivor body + 2 clothing sets; walker zombie; 3 trees, 2 bushes, 2 rocks; 5 props (dumpster, street light, fence, mailbox, trash can); blood splat, crack, leaves decals.
2. Rest of the snap kit and furniture → every building type in the catalog.
3. All vehicles with damage layers.
4. All zombies, animals, clothing and hair layers.
5. All item icons and held items.
6. FX, lights, remaining decals.

## 6. Tools Built (in `tools/`)
| Tool | Status | Purpose |
| :--- | :--- | :--- |
| `asset-catalog.ts` (`npm run art:catalog`) | **done** | Generates catalog from `src/data` and syncs status against `manifest.json` |
| `blender/render_iso.py` | **done** | Orthographic camera & NW sun renderer adhering strictly to `STYLE_BIBLE.md` §0.8 |
| `blender/snap_kit.py` | **done** | Procedural 1m modular walls, floors, roofs, furniture, and vehicles |
| `blender/pipeline_runner.py` (`npm run art:slice`) | **done** | Headless batch runner reading `asset_catalog.json` with `--slice` and `--unreal-export` |
| `run-blender.mjs` | **done** | Cross-platform Node CLI wrapper for Blender batch tools |
| `blender/destruction_demo.py` | **done** | Voronoi fracture + physics, reusable for damage states |
| `unreal/pipeline/build_arena.py` + `verify_arena.py` (`npm run ue:arena`) | **done, tested 2026-10-20** | Headless level builder from `arenas.json` with a verify step; replaces `unreal/build_arena.py` (crashed headless, 100× scale, mirrored arenas) |
| `unreal/pipeline/import_assets.py` (`npm run ue:import`) | **done, tested** | Headless glTF/FBX import with box collision and a scale check |
| `unreal/pipeline/shoot.ps1` (`npm run ue:shots`) | **done, tested** | PNG of each Unreal level from its preview camera |
| `pack-atlases.ts` (`npm run art:pack`) | planned | Texture packer for PixiJS sprite sheets |

## 7. Keeping it in sync
- Adding content to `src/data` (a new item, zombie, vehicle…) adds it to the catalog on the next `npm run art:catalog`.
- Planned non-registry art (kit pieces, furniture, decals) is listed in `tools/asset-catalog.ts` → `planned()`; move entries into `src/data` registries when the game starts using them.
- Update this file, the catalog, `UE_ARENA_PLAN.md`, and World Bible §69 when a lane or tool changes.
