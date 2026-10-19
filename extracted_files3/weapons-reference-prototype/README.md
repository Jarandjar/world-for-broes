# Weapons attachment system: reference prototype (optional)

Written by Claude against `world4bros-core` so the design could be checked against real code. **Antigravity owns the logic**:
adopt it, rewrite it, or ignore it. The design spec is `WEAPONS_BRIEF.md`.

- `weapons-reference.patch`: apply from the repo root with `patch -p2 < weapons-reference.patch` (verified to apply cleanly to the uploaded snapshot).
- `src-weapons/`: the same new module on its own (`src/weapons/*`): schema, 78-part catalog, per-gun slot layouts, derive/compat maths.

Status: `tsc --noEmit` clean and all 99 existing tests pass with the patch applied. **No new tests were written** and the Gunsmith panel and HUD
overlays are NOT built (they are designed, see the design canvas).
What the patch touches: `data/items.ts` (6 new guns, 4 ammo, 6 junk items, generated attachment items), `data/recipes.ts` (generated homemade recipes),
`data/schema.ts` (`GunDef.quirks`), `render/parts.ts` (6 drawers), `sim/components.ts` (`stunT`, `Projectile.quirks`), `sim/game.ts`
(loadouts, effective gun in `shootGun`, quirk effects, melee bonus, zombie stun, save/load, range armory, `aimInfo()` for the HUD).
