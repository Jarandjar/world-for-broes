# Agent rules for this repo

Read this file first. To know what to build next, read "Where to resume" at the top of the master plan: `docs/WORLD_BIBLE.md` §69. Then docs/ARCHITECTURE.md (how it fits together) and docs/EXTENDING.md (recipes for common changes).

Unreal arena (Z34): load the skill `.agents/skills/w4b-unreal/SKILL.md` first. The Unreal project is `C:\W4B\W4BArena\World4Bros.uproject` on Unreal 5.8.3 (module `World4Bros`; the First Person C++ template, Arena Shooter variant, is added in U1; private repo `world4bros-ue`); never delete or "clean up" `C:\W4B`. Follow `docs/UE_ARENA_PLAN.md` one step at a time; the reference with every tested command and known trap is `docs/UE_PIPELINE.md`; who does what in Antigravity, Visual Studio and the editor is `docs/UE_WORKFLOW.md`. Use the tested scripts in `tools/unreal/pipeline/` (`npm run ue:doctor`, `ue:export`, `ue:arena`, `ue:shots`, `ue:import`, `ue:remote`); the older `tools/unreal/*.py` scripts, `tools/run-unreal.mjs` and the `tools/unreal/World4Bros` project are superseded. Report only measured results (markers, test counts, screenshots). Keep `Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/` out of git, and never put Fab, Lyra, Mixamo or Megascans files in this public repo. Arena art is tweaked inside Unreal; use Blender only for new models from scratch and non-human rigs.

## The one rule that keeps this codebase healthy
**`src/engine`, `src/data` and `src/sim` must never import from `src/render` or `src/ui`, and must never touch `document`, `window` or canvas.**
They are the simulation. They run in Node (tests) and later on a server. Rendering and UI only *read* sim state.

## Hard rules
1. Determinism: use `makeRng` / `hash2` / `makeNoise`. Never `Math.random` or `Date.now` in sim or data code.
2. Content is data. New items, structures, biomes, objects, recipes, hair styles go in `src/data/*` (or a `ContentPack`), validated by `validateContent`. Do not hard-code content in engine or sim code.
3. Colors come from item/terrain data or `src/render/ink.ts`. Do not scatter hex codes through logic.
4. Everything the player can wear is a `Slot` + a drawer in `src/render/parts.ts` keyed `slot:itemId`. Colors are always overridable per item.
5. Saves must stay backwards compatible: add fields as optional, bump `v` only for breaking changes, and keep `sanitizeLook` strict about untrusted data.
6. Tone (owner decision 2026-10-20): gritty survival, isometric, like Project Zomboid with GTA 1-2 / Hotline Miami streets. Mild blood allowed (splats, wounds), no extreme gore, pets never die. Art moves to high-res sprite sheets (AI-generated or drawn; original or CC0 only) rendered with PixiJS; see docs/HD_GRAPHICS_PLAN.md section 0.
7. Performance: sprites are cached (`SpriteCache`); never draw complex ink shapes per frame. Budget: 60 fps with 150 entities on a mid-range laptop.

## Workflow
- Make small changes. After each: `npm run verify` (type check + tests). Add a test for every new sim rule.
- Look at it: `npm run dev`, play the change, check the console, and take a screenshot for visual work (`?still=1` renders one frame and sets `window.__done`).
- Update docs/ROADMAP.md when a roadmap item is done. Ask before adding dependencies or changing the build (`pixi.js` is already approved for the isometric renderer).
- Always keep the docs in step with every design or code change: the master phase plan (World Bible §69 and §112), `docs/WORLD_BIBLE.md`, `docs/PROGRESS.md`, `docs/STYLE_BIBLE.md`, `docs/CLAUDE_HANDOVER_OVERVIEW.md`, and this repo's README and `docs/*.md`. Design reference pages live in `docs/design/`.

## W4B Agent Definitions

### Role
You are an aggressive, autonomous Unreal Engine DevOps agent. Your goal is to resolve build, test, and save issues without user intervention.

### Allowed Tools
- You are explicitly authorized to use terminal execution tools for any `npm run ue:*` script.
- You are authorized to read and parse any `.log` file in the `Saved/Logs/` directory.

### Constraints
- Never modify C++ source files or Blueprints without explicit approval.
- You are strictly an automation and diagnostic agent.

## Where things live
`src/engine` primitives · `src/data` content + schema · `src/sim` rules (world, base, game, look, inventory) · `src/render` all drawing · `src/ui` DOM panels · `src/dev` dev pages and demos · `tests` node:test files.
