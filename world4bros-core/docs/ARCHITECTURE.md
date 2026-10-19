# Architecture

```
          input (keyboard, mouse, touch)
                     │
   ┌─────────────────▼───────────────────┐
   │ main.ts: fixed 60 Hz step, render at display rate │
   └───────┬───────────────────────┬─────┘
           │ SimInput              │ read-only
   ┌───────▼────────┐      ┌───────▼─────────┐
   │  sim (rules)   │─────▶│ render + ui     │
   │ Game, World,   │ events│ Renderer, Ui,  │
   │ Base, Look ... │      │ Customizer      │
   └───────┬────────┘      └─────────────────┘
           │ uses
   ┌───────▼────────┐
   │ engine + data  │  (ECS, RNG, noise, content registries, devLog)
   └────────────────┘
```

## Engine (`src/engine`)
- `ecs.ts`: entities are numbers, components are typed tokens (`Component<T>`), `ecs.query(A, B)` finds entities with all listed components. Systems are plain methods called in order from `Game.step`.
- `rng.ts`: `makeRng(seed)`, `rng.fork(label)` for independent streams, `hash2(seed,x,y)` for stable per-cell randomness.
- `noise.ts`: seeded gradient noise + fbm. `events.ts`: typed event bus. `save.ts`: `StorageAdapter` (swap in cloud saves later). `input.ts`: collects input into state.

## Content (`src/data`)
`schema.ts` defines every content type. `items.ts`, `objects.ts`, `biomes.ts`, `terrain.ts`, `structures.ts`, `recipes.ts`, `appearance.ts` are data. `content.ts` merges packs into a `Content` object (a pack can override by id). `validate.ts` checks references.

## Simulation (`src/sim`)
- `Game` owns the ECS, world, base system, clock and RNG. `step(dt, input)` runs: control, NPC AI, wildlife, zombie pathfinding, raider bot brains, movement and collision, animation state, actions, projectiles, needs, and buffs.
- `GameWorld`: chunk cache over `WorldGen`. Edits are stored in `mods` by object id, so chunks can be dropped and regenerated without losing changes. Collision: `world.blocked(x, y, r)` checks terrain, solid objects, structures, and active arena obstacles.
- `WorldGen`: runs an ordered list of `GenStage`s per chunk (`terrain`, `road`, `scatter`, `town`, `military`, `shooting_range`). Everything derives from `(seed, chunk coords)`.
- `BaseSystem`: structure footprints on a tile map, claim radius rules, `checkPlace`, `place`, `remove`, `solidAt`.

## Tactical Combat & Ballistics
- **Ballistics Engine (`updateProjectiles`)**: Discrete sub-step continuous collision detection. Calculates projectile speed, distance-based damage falloff, penetration counts, and impact particle effects.
- **Aim Stance**: When `aimHeld` is true (right-click or aim input), the character enters an aim stance, locking torso facing toward cursor and raising weapon.
- **Aim Arena & Cover (`src/sim/arena.ts`)**:
  - Implements **Duel Pit** (40x24m) and **FFA Yard** (64x40m) stationed inside the Army Base.
  - Distinguishes solid covers (crates, sandbags, low walls) that stop both movement and live bullets (`arenaBlocksBullet`), from soft covers (trees, barrels) that block movement only.
  - Symmetrical half-turn geometry verified by `checkDuelPitSymmetry`.
- **Bot Combat Brains (`updateRaiders`)**:
  - Autonomous agents (`Comp.Raider`) across 4 tiers: **Scarecrow**, **Regular**, **Sharp**, and **Menace**.
  - Tactical strafing, kiting, distance management (8–12m), line-of-sight checks through arena obstacles, and live projectile firing with weapon-specific aim spread.

## Diagnostics & Dev Log (`src/utils/dev-log.ts`)
- Global `DevLog` singleton captures runtime errors, uncaught window exceptions, simulation events, combat traces, and performance stats.
- Exposes `window.devLog` in browser console.
- In-game UI overlay (`F2` / `~`) with live filtering tabs, pause stream, and one-click Markdown/JSON bug report generation.

## Headless Telemetry (`src/sim/swarm.ts`)
- Pure, headless, deterministic simulation (`runSwarmHeadless`, `compareWeaponsPaired`).
- Simulates 150+ zombies in sub-second CPU time to produce weapon balance matrices and noise pressure telemetry without rendering overhead.

## Rendering (`src/render`)
- `ink.ts`: the style engine. `inked(g, path, fill, opts)` = muted wash fill + painted volume gradient + hatching on the shadow side + grain and wear + doubled hand-drawn outline.
- `SpriteCache`: draws procedural art once into canvases, keyed by item and colors and resolution (`ppu`).
- `parts.ts`: drawers keyed `slot:item` for every part of the character. `characterRenderer.ts` composes them in layers and animates: stride, arm swing, breathing, and swing arc.
- `lighting.ts`: night darkness with light holes and warm glow. `renderer.ts`: frame orchestration, y-sorting, flames, particles, target ring, build ghost, vignette and grain.
- `atmosphere.ts`: weather and screen polish as a registry of effects (`ATMOS_EFFECTS`, `registerAtmosEffect`). Frame order: terrain → `atmos.draw('ground')` (wet ground, puddles) → sun drop shadows → buildings, y-sorted entities, roofs → `atmos.draw('over')` (cloud shadows, mist, gloom grade, rain, snow, leaves) → overlays → `lighting.render` (plus `buildingLights` window pools) → `atmos.draw('screen')` (fireflies, flashes, vignette, paper grain). Render-only: weather is rolled from day + biome climate and never written to the sim. Quality dial `off/low/high` also picks sprite tiers (`sprites.ts` `ppuTier`) and the darkness buffer scale; `camera.lod` hides small details when zoomed out.

## UI (`src/ui`)
Plain DOM over the canvas. `Ui.frame` handles hotkeys, build-mode mouse input, HUD, prompts and panels:
`Pack [I]`, `Craft [R]`, `Build [B]`, `Gunsmith [U]`, `Range [Y]`, `Arena [H]`, `DevLog [F2]`, `SwarmLab [J]`, `Garage [V]`, `Overseer [O]`, `Faction [P]`, `Map [TAB]`, `Character [C]`.

## Testing
`npm test` runs 120+ unit and integration tests using `node:test` through `tsx`. Covers ECS, collision, ballistics, bot combat, arena geometry, dev logging, swarm simulation, saves, and content integrity.
