# Z26 plan: Status Effects, Reactions & Stagger

Status: **planned, data layer done (step 1)**. Spec: `WORLD_BIBLE.md` Part 18 §115–§118. Visual target: `design/index.html`. Z27 (biomes) builds on this.

Follow `AGENTS.md`: sim/data never import render/ui, deterministic RNG only (`game.rng.fork(...)`), content is data, add a node:test for every sim rule, run `npm run verify` after each step.

## Step 1 — Data layer (DONE 2026-10-19, `npm run verify` green, 122 tests)
Already in the code. Nothing in sim reads it yet, so the game behaves exactly as before.
- `src/data/schema.ts`: `EffectId`, `EFFECT_IDS`, `SurfaceTag`, `SURFACE_TAGS`, `HazardKind`, `EffectDef`, `ReactionDef`, `StaggerStep`, `StaggerStepDef`, `EffectLimits`. `Content` gained `effects`, `reactions`, `staggerLadder`, `effectLimits`. Optional `tags` on `TerrainDef` and `ObjectDef`; optional `poise`, `tags`, `immune`, `attackEffect` on `ZombieDef`.
- `src/data/effects.ts`: `EFFECTS` (12 rows, §115 colours, icon id, sound word, tooltip, numbers), `REACTIONS` (§116 as effect × tag rows), `STAGGER_LADDER` (20/50/80/100%), `EFFECT_LIMITS` (fires 24, puddles 32, clouds 12, chain depth 4, chain targets 24, player poise 60, regen delay 1.5 s at 50%/s).
- `src/data/content.ts`: packs can add or override effects, reactions (keyed effect+tag), ladder and limits.
- `src/data/validate.ts`: all 12 ids present, `#rrggbb` colours, icon/sound/tooltip required, known tags and effect refs, no duplicate reaction, `spawnHazard` effect must have a `hazard`, ladder shares rising, caps ≥ 1, zombie immunities/tags/attack effects valid.
- Tags: terrain `grass` flammable, `shallow`/`deep` wet. Objects: trees, bushes, logs, crates, fences flammable; fuel pump and slime drum explosive; cars, lamps, guardrails metal/conductive (map at the bottom of `objects.ts`).
- Zombies: poise per kind (runner 22 … goliath 220). Brute and goliath hit with `stagger`, spitter with `acid` (and is acid-immune), sapper is `explosive` and attacks with `explosion`, burrower is blind-immune, goliath is `metal`.

Column → tag mapping used by the reaction table: grass/wood = `flammable`, water = `wet`, oil = `oil`, metal = `metal` (+ `conductive` row for electricity chains), ice = `frozen`, gas/barrel = `explosive`.

## Step 2 — Status component (`src/sim/components.ts`)
```ts
interface ActiveEffect { id: EffectId; left: number; power: number; stacks: number; frozen?: boolean }
interface StatusC {
  effects: ActiveEffect[];
  baseTags: SurfaceTag[];      // from ZombieDef.tags, vehicle = ['metal','conductive']
  immune: EffectId[];
  poise: number; maxPoise: number; sinceHit: number;
  stagger: { step: StaggerStep; left: number } | null;
  stall: number;               // vehicle engine stall seconds
}
```
Add `Comp.Status`. Attach in `spawnPlayer` (poise = `effectLimits.playerPoise`), `spawnZombie` (poise = `def.poise ?? def.maxHp`), raiders, animals (poise = maxHp), vehicles (`baseTags` metal + conductive). Structures are out of scope for Z26 (note for later).

## Step 3 — Effect system (`src/sim/effects.ts`, new)
`export class EffectSystem` in the same style as `SiegeSystem` (`import type { Game }`), owned by `Game` as `readonly effects`.

State: `hazards: Hazard[]` with `{ id, effect, kind, x, y, r, left, power }`, plus a per-step chain budget.

API:
- `tagsOf(e)`: `baseTags` ∪ `grantsTags` of active effects ∪ `atMax.tags` when frozen ∪ terrain tags under the entity.
- `apply(e, id, opts?: { power?, x?, y?, depth?, source? }): boolean`
  1. Immune → false.
  2. `def.clears` strips listed effects (water clears burn/acid/goo, burn clears cold).
  3. For each tag in `tagsOf(e)`, look up the reaction (`content.reactions`, index once into a `Map<'effect:tag', ReactionDef>`). Apply `remove`, `apply` (depth + 1), `powerMult`, `stunMult`, `knockMult`, `stallVehicle`, `spawnHazard`, `explode` (→ `explode()` at depth + 1), `chainRadius` (→ `apply` same effect to entities within radius that share the tag, depth + 1). `cancels` stops step 4. Emit `reaction` event `{ effect, tag, label, x, y }`.
  4. Instant parts: `burst` damage × power; `impact` → `hitPoise`; `stun` sets/refreshes the timed entry. Timed effects: add or refresh (`left = duration`), stacking effects add a stack; at `maxStacks` set `frozen` and `left = atMax.duration`.
  5. Emit `effectApplied` `{ entity, id, x, y }` (render shows the sound word).
  - Chain guard: stop when `depth > maxChainDepth` or the chain's touched count > `maxChainTargets`; track visited entities per chain so A↔B can't ping-pong.
- `explode(x, y, power, depth, source?)`: entities within `explosion.radius`, damage `burst × power × (1 − d/r)`, knockback, `apply(e,'explosion',…)` (blast → knockdown). Ignites oil puddles and fire hazards in range, sets off `explosive` world objects via `world.damageObject` (respecting depth). `emitNoise(x,y,20)`. Emit `effectBurst`.
- `hitPoise(e, impact, blast)`: `poise -= impact`, `sinceHit = 0`; share = 1 − poise/maxPoise; pick the highest ladder step with `share ≥ step.share` (blast → knockdown), skip steps whose `ignoredFromPoise ≤ maxPoise`, never downgrade a running step; apply pushback away from the hit; after knockdown refill poise.
- `spawnHazard(effectId, x, y, power)`: uses `EffectDef.hazard`; cap per kind (`maxFires`/`maxPuddles`/`maxClouds`) by dropping the oldest; fire on a `wet` tile does not start; fire on an oil puddle doubles and consumes it.
- `update(dt)`:
  - Timers: tick `left`, remove expired; DoT `dps × power × dt` and `sanityDps` (player only); `endsInLight` uses `game.isLit`.
  - Stagger timer, poise regen after `poiseRegenDelay` at `poiseRegenRate × maxPoise` per second, vehicle stall timer.
  - Hazards: tick and expire; entities inside get the hazard's effect refreshed (once per 0.5 s); fires try to spread to `flammable` world objects within 1.5 m with `rng.fork('fire:'+id+':'+tick)` chance, and damage them; standing on `wet` terrain applies `wet`.
- `mods(e)`: `{ speedMult, canMove, canAttack, aimKept, sight, hidden, slide, damageTaken }` from active effects + stagger step. Pure function, easy to test.
- `damage(e, amount)`: route to player needs, `damageZombie(e, dmg, { knock: false })`, `damageAnimal`, raider hp, vehicle hp; multiply by `mods(e).damageTaken`.

## Step 4 — Wire into `src/sim/game.ts`
- `readonly effects = new EffectSystem(this)`; call `this.effects.update(dt)` in `step()` after `updateZombies`.
- New events in `GameEvents`: `effectApplied`, `effectBurst`, `reaction`.
- `controlPlayer`: multiply speed by `mods.speedMult`, zero when `!canMove`; `slide` → `damp` velocity toward input instead of setting it.
- `updateActions` / `shootGun`: no shooting when `!aimKept`; blind widens spread.
- `updateZombies`: speed × `speedMult`, skip attack when `!canAttack`, detection × `sight`, player detection × player's `hidden`. On attack, apply `def.attackEffect` to the target (spitter → acid + puddle, brute/goliath → stagger power, sapper → `explode` at itself then remove the sapper).
- `damageZombie(e, dmg, opts?)`: add optional `{ knock?: boolean }` (default true, keeps existing tests); every hit calls `hitPoise(e, dmg)`; `damageTaken` multiplier.
- `controlVehicle`: no throttle while `stall > 0`.
- Hook existing items (cartoon sources already in the game): `fireBreathTimer` → `apply(z,'burn')` instead of raw hp loss; `cream_pie` → `blind`; `water_balloon` → `wet` + puddle; `stink_sock_bomb` → smoke cloud; `air_horn` → `stagger`; `pool_noodle_of_justice` → `stagger` power 3; `road_flare` → small fire hazard.
- Save: optional `effects?: unknown` in `SaveData` (hazards + per-entity effects); load sanitises unknown ids. No `v` bump.

## Step 5 — Render (`src/render`)
- `effectIcons.ts` (new): one cached canvas per effect (`SpriteCache`-style map), keyed by `EffectDef.icon`: drip, flame, starburst, impact lines, bolt, snowflake, three spore puffs, goo blob, water droplet, oil slick with sheen, eye with pie splat, cloud. Colour from `EffectDef.color` with ink outline (`ink.ts`).
- `renderer.ts`:
  - Icon row above each entity with effects (max 4, small) and a HUD row for the player (bottom-left, with the tooltip text in `ui.ts` as `title`).
  - Body visuals: flames for burn (`this.flame`), green drips, frost ring / ice block when frozen, pink goo blob, blue drip for wet, purple spore ring, short sparks for electricity, starburst + crater decal for explosions, dizzy stars for stumble, flat body for knockdown.
  - Hazards: puddle ellipses, fires, soft smoke/spore clouds (draw under entities).
  - Sound words: push the `soundWord` into the existing `floaters` on `effectApplied`/`reaction`.
  - Reduced flashing: a setting (default ON for safety: no fast flashes) read by render only; electricity and explosion use a soft glow instead of a white flash when on.

## Step 6 — Tests (`tests/effects.test.ts`, node:test)
One per rule:
1. `validateContent` is clean and has 12 effects / reactions keyed by known tags.
2. Burn ticks damage, ends after its duration.
3. Wet clears burn; burn on a wet target is cancelled ("Put out").
4. Burn on an oiled target does double damage and removes the slick.
5. Electricity on a wet target chains to nearby wet targets but not dry ones.
6. Chains stop at `maxChainDepth`: a ring of sappers explodes at most depth+1 times.
7. Cold stacks slow, freeze at max stacks; burn on frozen melts to wet; explosion on frozen does ×3.
8. Immunity: spitter ignores acid.
9. Poise ladder: hits worth 20/50/80/100% of poise give flinch/stagger/stumble/knockdown; goliath ignores flinch; blast → knockdown; poise refills after the delay.
10. Hazard caps: spawning 40 fires keeps ≤ `maxFires`; puddles and clouds likewise.
11. Fire spreads to a flammable object nearby, never on a wet tile.
12. Determinism: two games with the same seed and inputs end with identical effect state.
13. Same rules for player and zombies: stagger stops the player moving; spitter bite puts acid on the player.
14. Save/load round-trips active effects and hazards.

## Step 7 — Docs (AGENTS.md rule)
- `WORLD_BIBLE.md`: §69 and §112 Z26 row → COMPLETE, §110 note, Part 18 "in code" notes (file names).
- `PROGRESS.md`: Z26 row → COMPLETE.
- `docs/ROADMAP.md`: tick Z26.
- `docs/EXTENDING.md` recipe 7: real steps (add a row to `EFFECTS`, an icon drawer, reaction rows; validate; test).
- `docs/ARCHITECTURE.md`: `EffectSystem` and `Comp.Status`.
- `STYLE_BIBLE.md`, `CLAUDE_HANDOVER_OVERVIEW.md`, `README.md`: one line each.

Done = `npm run verify` green, effects visibly apply to player and zombies in-game (`npm run dev`, screenshot with `?still=1`).

## Decisions (settled 2026-10-19)
- **Reduced flashing: ON by default.** §118 says "nothing flashes fast by default". Players can turn it off in settings.
- **Sappers explode on attack: yes.** Matches their data (`damage: 45`, suicide blast). They are not in the zombie-mode spawn table yet, so this cannot surprise current play.
- **Structures: later.** Z26 covers bros, zombies, raiders, animals and vehicles. Walls and turrets get `Comp.Status` with Z27, where biome hazards first hit bases. §118's "same rows for structures" stays the target.
- **Phase names:** the gunsmith build in `WEAPONS.md` was renumbered Z26a–e → Z29a–e so Z26 means only this phase.
