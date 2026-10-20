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
| **C. Rig + animate + render** | survivors, zombies, animals, clothing layers | Base body from MPFB 2 (CC0 assets) or the Unreal Lyra mannequin; animations from Mixamo / Lyra; render 8 directions × all animations; clothing as separate layers. |

AI (Gemini) is for **concept frames and 2D decals/icons only** — never for things that need 8 or 16 matching directions.

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
| Bodies | MPFB 2 (assets CC0) / Unreal Lyra mannequins | Characters, zombies |
| Animations | Mixamo (Adobe terms), Lyra (Epic, check use outside Unreal) | Characters, zombies |
| Textures | Poly Haven, ambientCG (CC0): asphalt, concrete, brick, siding, wood floor, tile, grass, dirt, sand, snow, rust, metal | Ground tiles, roads, snap kit, re-materials |
| Trees | Sapling Tree Gen / tree-gen (generators, output ours) | Vegetation |

Unreal-only sources (City Sample kits, Megascans free tier) are **not** for shipped sprites in this non-Unreal game unless the licence allows it; use them as reference.

## 5. Order (what to make first)
1. **Vertical slice set (step I2, ~45 assets)**: grass, dirt, asphalt, sidewalk tiles; kerb, lane line, crosswalk; snap-kit siding + brick walls (plain, window, door), wood floor, gable roof pieces; 10 furniture (bed, wardrobe, fridge, stove, sink, toilet, sofa, table, chair, cupboard); 1 car (rusty hatchback); survivor body + 2 clothing sets; walker zombie; 3 trees, 2 bushes, 2 rocks; 5 props (dumpster, street light, fence, mailbox, trash can); blood splat, crack, leaves decals.
2. Rest of the snap kit and furniture → every building type in the catalog.
3. All vehicles with damage layers.
4. All zombies, animals, clothing and hair layers.
5. All item icons and held items.
6. FX, lights, remaining decals.

## 6. Tools to build (in `tools/`)
| Tool | Status |
| :--- | :--- |
| `asset-catalog.ts` (`npm run art:catalog`) | **done** — generates the catalog from `src/data` |
| `blender/destruction_demo.py` | done — Voronoi fracture + physics, reusable for damage states |
| `blender/render_iso.py` | todo — style camera, directions, frames, layers, atlas naming |
| `blender/make_ours_*.py` + `materials.blend` | todo — the §3 pass per category |
| `blender/snap_kit.py` | todo — models every snap-kit piece |
| atlas packer (`art:pack`) | todo — free-tex-packer-core (MIT) |
| asset manifest + status read-back | todo — catalog shows `done` automatically |

## 7. Keeping it in sync
- Adding content to `src/data` (a new item, zombie, vehicle…) adds it to the catalog on the next `npm run art:catalog`.
- Planned non-registry art (kit pieces, furniture, decals) is listed in `tools/asset-catalog.ts` → `planned()`; move entries into `src/data` registries when the game starts using them.
- Update this file, the catalog and World Bible §69 when a lane or tool changes.
