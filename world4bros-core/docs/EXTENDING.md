# Extending the Game: Developer Recipes

## 1. Add a Wearable Outfit Item
1. `src/data/items.ts`: `gear('parka', 'Parka', 'torso', cs(primary, secondary, accent))`.
2. `src/render/parts.ts`: `PARTS['torso:parka'] = (p) => { ... }` using `p.c.primary` and `ink(...)`.
3. Preview in `dev/sheet.html`. It appears automatically in the Character Customizer (`C`).
4. Generate pixel art references or color schemes using the [Gemini Asset Generation Guide](GEMINI_ASSET_GENERATION_GUIDE.md#1-better-wear--character-outfits).

## 2. Add a Firearm or Gun Attachment
1. `src/data/schema.ts`: Define any new attachment IDs in `AttachmentId`.
2. `src/data/content.ts`: Add entry to `guns.set('gun_id', { ... })` specifying damage, falloff, spread, fire rate, and projectile speed.
3. For attachments, add entry to `ATTACHMENTS` array in `content.ts` specifying slot, stat multipliers, and visual flags.
4. Test weapon balance headless via `npm test` or in-game via `SwarmLab [J]`.

## 3. Add a World Object or Arena Cover
1. `src/data/schema.ts`: Add kind to `ObjectKind`.
2. `src/data/content.ts`: Add `objects.set('kind', { solid: true, radius: 0.45, hp: 200, tool: 'shovel', drawer: 'obj:kind' })`.
3. `src/render/objects.ts`: Implement Canvas drawing function for the object.
4. For Arena bullet-stopping covers, register rectangle coordinates in `ARENAS['duel_pit'].crates` or `sandbags` in `src/sim/arena.ts`.

## 4. Add or Customize Bot AI Tiers
1. `src/sim/components.ts`: Inspect `RaiderC` (`level`, `sight`, `aimSpread`, `shotInterval`, `speed`, `hp`).
2. `src/sim/game.ts`: Customize tier specs in `spawnArenaBot(level, weapon, spawnIndex)` and behavior branches in `updateRaiders(dt)`.
3. Verified in tests: `tests/arena.test.ts`.

## 5. Add a Diagnostic Dev Log or Bug Event
1. Import `devLog`: `import { devLog } from '../utils/dev-log';`
2. Log anywhere in simulation or UI:
   - `devLog.combat('Bullet hit target')`
   - `devLog.warn('sim', 'Entity out of range', { id })`
   - `devLog.bug('system', 'Crash caught', undefined, err.stack)`
3. Open in-game via `DevLog [F2]` or `~` to inspect live logs and export diagnostic reports.

## 6. Add a Biome
1. `src/data/biomes.ts`: add a `BiomeDef` entry with `climate`, `terrain`, `tint` and `objects` scatter. Biomes are picked by nearest climate, so no engine change is needed.
2. Draw any new props as objects (recipe 3).
3. Planned for Z27: hazards, effect lists, ambient feel and spawn tables move into the same definition. Spec and the 6 proposed biomes: World Bible §119–§120 and `design/biomes.html`.

## 7. Add a Status Effect (Z26, data layer in code; sim planned)
1. `src/data/schema.ts`: add the id to `EffectId` and `EFFECT_IDS`.
2. `src/data/effects.ts`: add an `EffectDef` row to `EFFECTS` (colour, icon id, sound word, tooltip, from/does/ends, numbers) and any `REACTIONS` rows keyed by effect × surface tag (`flammable`, `wet`, `conductive`, `explosive`, `oil`, `frozen`, `metal`).
3. `npm run verify`: `validateContent` checks colours, icons, tags and references.
4. Not yet in code (see [Z26_PLAN.md](Z26_PLAN.md)): the `EffectSystem` that applies rows in `src/sim/effects.ts`, and the icon drawer in `src/render/effectIcons.ts`.
Spec: World Bible §115–§118 and `design/index.html`.

## 8. Add a Building Style
1. `src/data/buildings.ts`: add a `BuildingDef` with size range, `roof`, `walls`, `door`, `layout`, `features`, `colors`, and optional `palettes` (extra colour sets; one is picked per building from a hash of seed and position).
2. Town lots pick styles by `weight` and `zone`; houses are placed street-facing with a yard automatically (`townBuildings` in `src/sim/worldgen.ts`).
3. Drawing is in `src/render/buildings.ts`. Check it with `npm run dev` and `tests/buildings.test.ts`.

## 9. Add a Town or Change Roads
1. `src/sim/worldgen.ts`: add a `TownCenter` to `REGIONAL_TOWNS` with a `size` (`small`, `big`, `metro`). Keep towns at least the sum of their `TOWN_PROFILES` radii apart.
2. Highways and dirt tracks are generated from the towns automatically (`src/sim/roads.ts`); widths are `HIGHWAY_HALF` / `DIRT_HALF` there and `MAIN_HALF` / `SIDE_HALF` in `worldgen.ts`.
3. Road looks are in `src/render/roadRibbons.ts` (`ROAD_LOOK`). Check with `tests/map.test.ts`.

## 9b. Add Art for New Content
1. Add the content to `src/data` as usual.
2. Run `npm run art:catalog`: the new entry appears in `docs/ASSET_CATALOG.md` with what to make and the suggested source.
3. Make it following `docs/ASSET_PIPELINE.md` (download + "make it ours", generate, or rig + render) and log the licence in `docs/CREDITS.md`.

## 10. Add a Generation Stage (Roads, Military Bases, Dungeons)
1. Implement `GenStage` (`src/sim/worldgen.ts`) and include in `DEFAULT_STAGES`. Derive all positions deterministically from `ctx.seed` and chunk coordinates `(chunk.cx, chunk.cy)`.
2. Filter or clear natural scatter within special boundaries (see `shootingRangeStage`).
3. Add a determinism unit test in `tests/worldgen.test.ts`.

## 11. Add a Weather or Screen Effect
Everything lives in `src/render/atmosphere.ts`; `renderer.ts` needs no change.
1. **New weather kind** = one row in `WEATHERS`: `wx(id, name, weight, params, climate?, fogColor?)`. Params are 0..1 (`wet, fog, clouds, wind, rain, snow, leaves, gloom, dust`); existing effects react to them automatically. `climate` limits where it rolls (biome `temp`/`moist`).
2. **New visual** = `registerAtmosEffect({ id, layer, minQuality, active?, draw })`. Layers: `ground` (after terrain, world units), `over` (after roofs, world units), `screen` (after lighting, pixels). `draw(f, s)` gets the frame `f` (camera, `f.w` weather, `f.sun`, `f.darkness`, `f.lod`, `f.quality`) and its own scratch object `s` for pools and caches. Use the helpers `pool`, `worldPattern`, `cloudTexture`, `glowSprite`, `memo`; never allocate canvases per frame and keep counts low when `f.lod === 0`.
3. Put numbers in `ATMOS_TUNING`, not inside `draw`. Try it with `?weather=<id>&gfx=high`, or `__atmos.forcedWeather = 'id'` in the console.
4. Run `npm run verify`: `tests/atmosphere.test.ts` checks unique ids, layers, tiers and 0..1 params.

Worked example, a "pollen" weather with drifting yellow specks:
```ts
// in the WEATHERS array literal:
// wx('pollen', 'Pollen', 1, { wind: 0.3, leaves: 0.2 }, { minTemp: 0.4, maxTemp: 0.8, minMoist: 0.4 }),
registerAtmosEffect({
  id: 'pollen', layer: 'over', minQuality: 'low',
  active: (f) => f.darkness < 0.4,
  draw: (f, s) => {
    const P = pool(s, 'p', 40, 31); f.g.save(); f.g.setTransform(1, 0, 0, 1, 0, 0); f.g.fillStyle = 'rgba(250,224,120,0.7)';
    for (let i = 0; i < 40; i++) { P.x[i] = (P.x[i]! + f.dt * 0.02) % 1; f.g.fillRect(P.x[i]! * f.W, P.y[i]! * f.H, 3, 3); }
    f.g.restore();
  },
});
```
Helpers (`wx`, `pool`, `memo`, `worldPattern`, `toScreen`) are exported, so an effect can also live in its own file that imports them and calls `registerAtmosEffect` (import that file once from `atmosphere.ts`).

## 12. Add a Vehicle Decal or Paint Job
Paint jobs are optional data; drawing is a registry keyed by name, so no switch needs editing.
1. Paint only: in `src/data/vehicles.ts`, give the vehicle id an entry in `VEHICLE_LOOKS` (`trim`, `glass`, `accent`, `accent2`, `decal`, `number`, `grime`, `rust`, `dirt`, `rustTone`; all optional, missing fields use `DEFAULT_VEHICLE_LOOK`).
2. New decal: add its name to the `VehicleDecal` type in `src/data/vehicles.ts`, then add a painter with the same name to `DECALS` in `src/render/vehicleRenderer.ts`. TypeScript reports a missing painter. A painter draws in vehicle units (+x is forward, `c.halfL` / `c.halfW` are half the length / width) and is already clipped to the body.
3. Add the name to the `DECALS` list in `tests/vehicle-looks.test.ts`, then run `npm run verify` and look at it with `npm run dev` and `?fresh&fleet`.

Worked example, a checkered stripe:
```ts
// src/data/vehicles.ts
export type VehicleDecal = 'none' | 'stripes' | /* … */ | 'checker';
clown_car: { /* … */ decal: 'checker', accent: '#f8fafc', accent2: '#15171b' },
// src/render/vehicleRenderer.ts, inside DECALS
checker: (g, c) => {
  for (let i = 0; i < 12; i++) for (let j = 0; j < 2; j++) {
    g.fillStyle = (i + j) % 2 ? c.look.accent : c.look.accent2;
    g.fillRect(-c.halfL + i * 0.1, -0.1 + j * 0.1, 0.1, 0.1);
  }
},
```
New body shapes work the same way: add a `BodyDrawer` to `BODIES` keyed by `VehicleKind` and build it from the shared helpers `paintBody`, `glass`, `chrome` and `seams`, plus a `LAMPS` row for lamp positions. Bodies are drawn once into the sprite cache. Anything that moves every frame goes in `ANIMS` (above the body) or `UNDER` (below it).