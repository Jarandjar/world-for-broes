# Roadmap (maps to the build plan phases)

Done in this core:
- Z0 Architecture and data contracts (registries, validation, mode flag in `GameConfig`).
- Z0B Input foundation (keyboard, mouse, touch stick and buttons).
- Z0C Animation foundation (procedural rig: stride, swing with hit timing, breathing).
- Style: Ink Hollow, top-down (see STYLE.md).
- Z1 Items, inventory, crafting, equipment, hair and color customization.
- Z2 Map core: chunked deterministic world with stage pipeline and 10 biomes.
- Z8 Base core: claims, placement rules, structures, storage, lights.
- Z10 Complete Vehicle Fleet: 7 distinct vehicles (rusty hatchback, clown car, bathtub buggy, golf cart, ice cream truck, hot dog van, tesla roadster) with arcade steering, fuel, repairs, trunk inventories, and high-speed zombie ramming.
- Z11 Companion Pets: Wolf pup and hunting dog with follow AI, petting sanity recovery, zombie barking/biting, and save round-tripping.
- Z12 Undead Survival Core: Walker, runner, and brute zombies, day/night light sensitivity, hearing/noise propagation, and melee combat.
- Z13 Environmental Controls: Always-day mode toggle, configurable lighting engine, darkness survival damage.
- Z14 Industrial Automation (Factorio/Satisfactory style): Conveyor belts with item physics, spring inserters, burner mining drills extracting mineral nodes, and automated smelting furnaces.
- Z15 RimWorld Colony Simulation: Autonomous colonists with generative skills, passions, traits, 1-4 work priority matrix, psychological mood thoughts, slapstick mental breaks, and AI Storyteller directors (Cassandra, Phoebe, Randy).
- Z16 Evolved Zombie Threat & Industrial Sieges: Chunk pollution tracking, natural dissipation, horde siege waves, specialized siege mutants, automated turrets, and belt-fed ammo reloading.
- Z17 PvPvE Faction Warfare & Base Raiding: Tri-zone risk topology, Clan Territory Totem with automated Tesla dome shields, 2-hour active raid windows, 48h non-aggression peace treaties, C4 demolition charges, and black market bounty contracts.
- Z18 Master Weapons, Goofy Gadgets & Side-Effect Food: Slapstick weaponry, wearable gear traits, tactical throwables, and culinary buff foods.
- Z19 Cows, Livestock & Ambient Wildlife: 7 distinct animated animal archetypes, interactive farm mechanics, ambient wildlife AI, zombie predation, and hunting/cooking expansion.
- Z20 Ballistics Proving Grounds & Firing Line: Precision target lanes (5m, 10m, 15m, 20m, 25m, 30m, 40m, 50m, 60m, 75m, 100m), continuous line-segment bullet collisions, distance-based damage falloff, wobbling steel target feedback, and CQB killhouse lanes.
- Z21 Gunsmith Workbench & Homemade Attachment System: 20+ weapon attachments (suppressors, scopes, extended magazines, lasers, bayonets), dynamic stat recalculation, visual info flags, and procedural sounds.
- Z22 Army Base Combat Arena & Fight Bench (`Arena [H]`): Duel Pit (40x24m) and FFA Yard (64x40m) relocated to the Army Base complex with bullet-stopping cover mechanics, 4 autonomous bot difficulty tiers (Scarecrow, Regular, Sharp, Menace) with line-of-sight checks, configurable quarantined zombie hordes, and 3-way multi-faction skirmishes.
- Z23 SwarmLab Headless Telemetry & Weapon Matrix (`SwarmLab [J]`): Sub-second headless simulation of 150+ zombies, deterministic paired-seed weapon balance matrices, and noise pressure monitoring.
- Z24 Runtime Dev Log & Bug Engine (`DevLog [F2]`): Global error trapping (uncaught exceptions and promise rejections), color-coded telemetry stream, filtering tabs (Bugs, Combat, Sim, Perf), and one-click Markdown/JSON bug report generation.
- Z25 Gemini Asset Generation Bible & Prompts: Production guide, prompt templates, negative prompt rules, and schema bindings for generating wear, items, objects, terrain autotiles, and effects with Gemini.

Phase numbers: the master plan in the World Bible (§69) is the source of truth. The Z20–Z25 items above are finished side projects that reused those numbers; they are not the planned Z20–Z23 (multiplayer, audio, underground, endgame) in §69 and §112.

Next, in order (see "Where to resume" in World Bible §69):
0. Z30 Town Houses, Yards & HD Graphics: houses face the street with porches, paths, mailboxes, fenced yards and colour palettes (done, `tests/buildings.test.ts`). Next: shaped gable and hip roofs with ink outlines, wall details, yard props, a looser small-town layout, saving `yard`/`front` in `src/sim/staticmap.ts`, and roof/feature registries. Lighting and atmosphere pass done (sun shadows, night window pools, weather, quality dial; `src/render/atmosphere.ts`, `tests/atmosphere.test.ts`); next for it: a quality toggle in the settings UI and real sim weather (rain muffles noise, fog cuts vision, per the World Bible weather rules) feeding the same `WEATHERS` rows. HD passes for roads, props, vehicles and characters were mid-work on 2026-10-19: run `npm run verify` and finish any half-added content first.
1. Multiplayer networking: Headless server simulation loop, client snapshot reconciliation, and deterministic client input streaming.
2. Audio & Soundscapes: Deferred to Beta phase — custom Foley, voice lines, and soundscape design will be produced by the owner when the game reaches Beta.
3. Expanded biome flora and underground bunker ruins with high-tier tech loot.
4. Z26 Status Effects, Reactions & Stagger: 12 effects as data rows, a data reaction table with a chain depth limit, poise and the stagger ladder. Spec: World Bible Part 18 §115–§118; design page `../docs/design/index.html`. Data layer done (`src/data/effects.ts`); step-by-step build plan in [Z26_PLAN.md](Z26_PLAN.md).
5. Z27 Biome Atlas Expansion: hazard, effect, prop and spawn data for the 10 biomes in `src/data/biomes.ts`, then 6 proposed biomes (Toxic Marsh, Cinder Wastes, Storm Plains, Salt Flats, Crystal Caverns, Ruined Downtown). Needs Z26. Spec: World Bible §1, §119–§120; design page `../docs/design/biomes.html`.
6. Z28 Aim Arena & Test Bench: arenas and bot levels are in (listed above as the Army Base Combat Arena); next are rounds, score and the match HUD, then the headless Test Bench, online 1v1 and free-for-all. Plan: [ARENA.md](ARENA.md) steps Z28a–e.
7. Z29 Weapon Builder, Gunsmith & Info HUD: pure `src/weapons` module, loadouts, the Gunsmith panel, the info HUD, then loot and balance. Builds on the Gunsmith Workbench listed above. Plan: [WEAPONS.md](WEAPONS.md) steps Z29a–e.

- Z30 vehicle HD pass done (2026-10-19): cached vehicle sprites, data paint jobs and decals, hp-driven wear, reacting lamps. Registries `BODIES` / `DECALS` in `src/render/vehicleRenderer.ts`; recipe in [EXTENDING.md](EXTENDING.md) §10.