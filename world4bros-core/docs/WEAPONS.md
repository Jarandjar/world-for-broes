# Weapons, attachments and info gizmos: design brief for Antigravity

Save as `docs/WEAPONS.md`. **Claude designed this; Antigravity implements it.** Visual designs are on the design canvas
"World 4 Bro's: Gunsmith and Info HUD" (https://claude.ai/artifact/GUQP7Y4MEWy4vqCTwUkQLY): desktop Gunsmith, phone Gunsmith, the info-HUD states,
and the part-card and homemade-workshop sheet. Build to those. A working reference prototype exists (`weapons-reference-prototype.zip`); it is optional.

## 1. What the player asked for

1. A full **menu** to fit attachments to weapons, covering **every weapon**: guns, goofy tube and toy guns, and melee.
2. Lots of **strange, goofy, homemade** parts (junk plus duct tape).
3. **Visual info only if you have an info attachment**: the aim HUD shows nothing extra unless a fitted part gives that readout.

## 2. What the repo already has (read before building)

- `GunDef` on `ItemDef.gun` (`src/data/schema.ts`): `ammoItem, ammoPerShot, aimSystem (laser|bloom|arc|burst), projectileSpeed, maxRange, baseDamage, minDamage, falloffStartDist, spreadAngle (rad), fireRate (shots/s), noiseRadius, penetration, arced, pelletCount, ...`.
- `Game.shootGun()` reads the held item's `gun`; ammo comes straight from the pack. There are no magazines, reloads or jams. `ProjectileC` carries damage, falloff and pierce. Hits go through `damageZombie`.
- The shooting range (`teleportToShootingRange`) and `tests/ballistics-range.test.ts` already exist. Keep them passing.
- `AimRenderState` (`src/render/renderer.ts`) draws a reticle per `aimSystem`, with the ammo count in `RenderState.aim`.

Design rule: **layer on top of this**. Do not replace `GunDef`; derive an *effective* `GunDef` from the base gun plus fitted parts.

## 3. Data model (the contract)

```ts
type SlotId = 'muzzle'|'barrel'|'optic'|'info'|'underbarrel'|'side'|'feed'|'stock'|'grip'|'charm'|'head'|'shaft'|'pommel';
type Mount  = 'thread'|'tube'|'rail'|'mag'|'tape'|'any';           // 'tape' = the homemade mount
type InfoMode = 'ammo'|'range'|'falloff'|'noise'|'spread'|'trajectory'|'enemy_hp'|'threat';
interface Stats { damage; pellets; rate; spread; range; noise; pen; speed; light; weight; ammo }   // units = GunDef's own
interface AttachmentDef { id; name; slot; mount; rarity: 'J'|'C'|'U'|'R'|'X'; desc;
  add?: Partial<Stats>; mul?: Partial<Stats>; quirks?: Quirk[]; info?: InfoMode[];
  requires?: string[]; forbid?: string[]; conflicts?: string[]; tags?; homemade?: boolean; recipe?: Record<ItemId, number> }
interface Profile { cls; tags: string[]; slots: Partial<Record<SlotId, Mount[]>> }   // per weapon: which slots exist and which mounts each accepts
```

- **Loadout** = `Record<SlotId, attachmentId>`, stored **per weapon item id** (one loadout per weapon type; per-instance is a later option). Persist it in `SaveData.loadouts`.
- **Every attachment is an inventory item** (id equals the attachment id, generated from the catalog). Fitting takes one from the pack; removing or replacing returns it. Creative mode and the shooting range make parts free.
- **Compatibility** (`canFit`): the slot exists, the slot accepts the part's mount, the weapon has all `requires` tags, none of `forbid`, no `conflicts`. A failure must give a *human sentence* (the UI prints it). A fitted part in the same slot is *replaced*.
- **Derive**: apply all `add` first, then all `mul`, then clamp (damage >= 0, pellets and pen and ammo rounded >= 1, rate >= 0.1, spread >= 0.002, range >= 1, speed >= 3). Scale `minDamage` with damage and `falloffStartDist` with range. Melee uses `damage, rate (tool.speed), range (tool.reach), weight`.
- **Weapon profiles** (slot layouts) are data, one per gun id. Firearms get all firearm slots; toys and tube guns get fewer and mostly `tape` mounts; **any melee weapon** gets the melee layout (`head, shaft, grip, pommel, side, info, charm`).
- **Intrinsic quirks**: `GunDef.quirks` (a foam blaster always has a small stun chance; the chicken cannon always squeaks).

## 4. Quirks (all of these must work; data only, no per-item code)

`noise_pulse{extra}` extra noise radius per shot (a lure on purpose) · `wobble{rad}` extra random spread · `backfire{chance}` soot and a little self-damage ·
`confetti` muzzle burst · `stun{chance,secs}` target frozen (`ZombieC.stunT`, zombie AI skips while > 0) · `knockback{power}` shove the target ·
`ammo_leak{chance}` eats an extra round · `ammo_save{chance}` free shot · `sanity{amount}` +sanity per shot · `recoil_boing{strength}` shoves the player backwards.
Melee weapons use the same quirks on hit. Quirks stack; sum `wobble`, roll the others independently with a deterministic RNG fork.

## 5. Info gating: the headline feature

The aim HUD draws **extra readouts only for modes in the union of fitted parts' `info`**. A weapon with no info part keeps only its own aim style (laser, bloom, arc, burst) and a **vague ammo word** instead of the number (pack count > 30 "plenty of", 11-30 "some", 4-10 "a few", 1-3 "the last", 0 "no").

| Mode | Draws | Given by |
|---|---|---|
| `ammo` | exact number by the reticle | Ammo Counter Clip, Smart Scope 9000 |
| `trajectory` | line from you toward the cursor | Periscope Sight, Trajectory Ruler, Laser Level, Smart Scope 9000 |
| `spread` | wedge of real effective spread (incl. wobble) | Trajectory Ruler, Smart Scope 9000 |
| `range` | arcs every 5 m plus the maximum range | Half a Binocular, Rangefinder Gizmo, Smart Scope 9000 |
| `falloff` | aim strip fades from full damage (at `falloffStartDist`) to minimum; damage at the cursor | Rangefinder Gizmo, Smart Scope 9000 |
| `noise` | dashed circle of the *effective* noise radius (includes suppressors and `noise_pulse`) | Decibel Doodad |
| `enemy_hp` | bar and `hp/max` over zombies within 1.2x range | Zombie Health Monocle, Threat Radar Lamp, Smart Scope 9000 |
| `threat` | edge arrows plus distance for zombies within 32 m | Rear-View Mirror, Threat Radar Lamp |

Rules: readouts use the **real effective stats** (never fake numbers). They draw only while aiming. Style per the Lantern Hollow board: ink-dark labels, cream text, ember for noise, pale gold for damage, moon-rim blue for range. Exact layout is on the "Info attachments" artboard.
Plumbing suggestion: `Game.aimInfo()` returns `{ modes, range, falloffStart, noise, spread, damage, minDamage, ammo, enemies[], threats[] }` already filtered by modes; `RenderState.aim` carries it; the renderer only draws.

## 6. The catalog

78 parts. Canonical data: `src/weapons/attachments.ts` in the prototype (names, stats, quirks, recipes). By slot: muzzle 11, barrel 6, sight 10, info 6, underbarrel 8, side 8, feed 6, stock 5, grip 3, charm 5, melee head 6, shaft 2, pommel 2. About 35 are `homemade` with a junk `recipe`; the generated recipes appear in the crafting panel, hand-crafted with no station.
New junk items needed: `duct_tape, mismatched_sock, empty_bottle, wire, hose, glue`. **They need loot sources** (not done): suggest garages, houses, junk piles and the scrapyard.
New guns (all use the existing `gunTool`): `plinker_pistol, bark_revolver, foam_dart_blaster, cork_crossbow, chicken_cannon, rubber_band_gatling`, with ammo `pistol_rounds, corks, squeaker_chickens, rubber_bands` (foam darts already exist). Each needs an inked drawer (the render-contract test enforces it); the prototype has placeholder-quality drawers.

## 7. Screens (see the canvas)

- **Gunsmith panel**, key `U`, `PanelName 'gunsmith'`, built like the existing panels. Left: weapons grouped Firearms, Homemade and goofy, Melee (only weapons you have). Centre: the bench with a socket per slot (filled, empty, selected) and the "what changed" stat bars, the "readouts unlocked" grid and the quirks list. Right: parts for the selected slot as cards.
- **Part card states**: fits, replaces, fitted, not in pack, does not fit (with the reason). Colour is never the only signal (arrows, words, tags).
- **Stat colours**: green-ish good, salmon bad, with arrows and a word. "Better" means the stat's own direction (lower spread, noise and weight are better).
- **Phone** (landscape 844x390): three columns (slots, parts, summary). 48 px targets, 12 px gaps, safe areas, as in the control layouts. Left-handed mode mirrors it.
- **Homemade workshop**: recipe cards in the crafting panel for `homemade` parts, showing owned/needed junk and what you get.
- **Quirk badges** each get an icon and a plain sentence (sheet on the canvas).

## 8. Out of scope for this pass (decided)

Magazines, reload, jams, heat and gun wear; battery-powered smart gizmos; per-instance loadouts; part loot tables; balance tuning. They are good later steps; do not build them now. The `feed` slot therefore only changes rate, weight, `ammo_save` and `ammo_leak`.

## 9. Build plan (stop for approval after each)

Numbered Z29 (renamed from Z26a–e on 2026-10-19: Z26 is Status Effects and Z27 the Biome Atlas in the World Bible §69 master plan, and Z28 is the Aim Arena in `ARENA.md`).

1. **Z29a Data and derive**: catalog, profiles, `canFit`, derive, generated items and recipes, junk items, the six new guns and drawers, `GunDef.quirks`. Pure `src/weapons`, no rendering.
2. **Z29b Game wiring**: loadouts and save, `shootGun` uses the effective gun, quirks, melee bonus and quirks, zombie stun, range armory, `aimInfo()`.
3. **Z29c Gunsmith UI**: the panel from the canvas, desktop then phone.
4. **Z29d Info HUD**: gated overlays in the renderer per section 5, plus the vague-ammo baseline.
5. **Z29e Loot and balance**: junk and part loot tables, a balance pass on the numbers.

## 10. Tests to add

Data: unique ids; every part fits at least one weapon; every recipe input exists; every gun has a profile; every equippable has a drawer.
Rules: wrong mount, missing tag and forbidden tag are rejected with a reason; replacing returns the old part; add-then-mul order; clamps.
Behaviour: a suppressor lowers the emitted noise radius; a scope narrows pellet spread; a choke ring needs a pellet weapon; foam darts stun and a stunned zombie does not move; knockback moves a zombie away; `ammo_save` and `ammo_leak` change ammo use over many shots (fixed seed); `recoil_boing` pushes the player; melee studs add exact damage; save and load keeps loadouts.
Info: with no info part `aimInfo().modes` is empty and no extra data is computed; each giver turns on exactly its modes; enemy and threat lists are empty without their modes.
Existing suites (`ballistics-range`, `weapons-food`, `render-contract`) must stay green.

## 11. Open questions for the owner

- Per-instance loadouts (two shotguns, two loadouts) or per-type for now? (This brief: per-type.)
- Should some info gizmos need batteries later? (Deferred.)
- Where should the junk come from: houses and garages, or only the scrapyard?
- Art direction for icons: one slot glyph plus a rarity frame per part is the cheapest consistent option.

## 12. Paste-ready prompt for Antigravity

```
Read docs/WEAPONS.md (the brief) and open the design canvas it links to. Build phase Z29a only: the pure src/weapons module (catalog, profiles,
canFit, derive), generated attachment items and homemade recipes, the six new guns with ammo and drawers, and GunDef.quirks. You may adopt the
reference prototype in weapons-reference-prototype.zip or write your own. Keep src/weapons free of Game and rendering. Add the tests listed in
section 10 for data and rules. Run the full suite and keep ballistics-range, weapons-food and render-contract green. Stop for my approval
after the plan and again when done.
```
