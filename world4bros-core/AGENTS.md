# Agent rules for this repo

Read this file first. To know what to build next, read "Where to resume" at the top of the master plan: `../docs/WORLD_BIBLE.md` §69. Then docs/ARCHITECTURE.md (how it fits together) and docs/EXTENDING.md (recipes for common changes).

## The one rule that keeps this codebase healthy
**`src/engine`, `src/data` and `src/sim` must never import from `src/render` or `src/ui`, and must never touch `document`, `window` or canvas.**
They are the simulation. They run in Node (tests) and later on a server. Rendering and UI only *read* sim state.

## Hard rules
1. Determinism: use `makeRng` / `hash2` / `makeNoise`. Never `Math.random` or `Date.now` in sim or data code.
2. Content is data. New items, structures, biomes, objects, recipes, hair styles go in `src/data/*` (or a `ContentPack`), validated by `validateContent`. Do not hard-code content in engine or sim code.
3. Colors come from item/terrain data or `src/render/ink.ts`. Do not scatter hex codes through logic.
4. Everything the player can wear is a `Slot` + a drawer in `src/render/parts.ts` keyed `slot:itemId`. Colors are always overridable per item.
5. Saves must stay backwards compatible: add fields as optional, bump `v` only for breaking changes, and keep `sanitizeLook` strict about untrusted data.
6. Cartoon tone: no gore, pets never die, zombies pop into spore puffs. Original or CC0 assets only (the art is procedural today).
7. Performance: sprites are cached (`SpriteCache`); never draw complex ink shapes per frame. Budget: 60 fps with 150 entities on a mid-range laptop.

## Workflow
- Make small changes. After each: `npm run verify` (type check + tests). Add a test for every new sim rule.
- Look at it: `npm run dev`, play the change, check the console, and take a screenshot for visual work (`?still=1` renders one frame and sets `window.__done`).
- Update docs/ROADMAP.md when a roadmap item is done. Ask before adding dependencies or changing the build.
- Always keep the docs in step with every design or code change: the master phase plan (World Bible §69 and §112), `../docs/WORLD_BIBLE.md`, `../docs/PROGRESS.md`, `../docs/STYLE.md`, `../docs/CLAUDE_HANDOVER_OVERVIEW.md`, and this repo's README and `docs/*.md`. Design reference pages live in `../docs/design/`.

## Where things live
`src/engine` primitives · `src/data` content + schema · `src/sim` rules (world, base, game, look, inventory) · `src/render` all drawing · `src/ui` DOM panels · `src/dev` dev pages and demos · `tests` node:test files.
