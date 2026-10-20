# World 4 Bro's — Comprehensive Project Overview & Sandbox Architecture Briefing for Claude AI

> **Purpose**: This document gives Claude AI a complete, authoritative status report of the project as of October 2026.
> **Strategic Goal**: Focus on **expanding and refining the engine and sandbox foundation first** (grounded in [`WORLD_BIBLE.md`](WORLD_BIBLE.md)), before locking down rigid linear game loops.

---

## 1. Executive Summary & Vision

**World 4 Bro's** (also known as *Lantern Hollow*) is a **2.5D top-down slapstick survival sandbox** — described as *"Don't Starve meets Paper Mario and classic slapstick comic books"*. 

- **Aesthetic**: Pop-up papercraft diorama & rubber-hose hybrid. Scored cardboard edges, hand-inked cross-hatching, comic onomatopoeia (`"BONK!"`, `"POOF!"`, `"SQUEAK!"`), and atmospheric candlelight/darkness contrast.
- **Tone**: Physical comedy meets eerie survival tension. Darkness is genuinely dangerous, but tools, traps, gadgets, and creatures are delightfully silly and reactive.
- **Core Design Rule (from World Bible §0)**:
  > *"Every object, item, and entity must be pure data (TypeScript/JSON definitions) so new content, behaviors, and biomes never require engine rewrites."*

---

## 2. Where the Project Is Right Now (Live & Verified)

The project currently has a production deployment live at **[https://world4bros-core.vercel.app](https://world4bros-core.vercel.app)** with **38 passing automated tests** across simulation, ECS, worldgen, vehicle physics, pet AI, and cartography.

### Working Engine Systems in Code:
1. **Pure Data Simulation & ECS (`src/sim/`, `src/engine/ecs.ts`)**:
   - Zero DOM or Canvas dependencies in the simulation core (`Game`, `GameWorld`, `Clock`).
   - Clean Component-Entity-System (`Comp.Transform`, `Comp.Character`, `Comp.Inventory`, `Comp.Vehicle`, `Comp.Pet`, `Comp.Zombie`, `Comp.Structure`).
   - Fixed 60Hz tick simulation with variable-rate rendering.
2. **Infinite Procedural Macro-Continent (`src/sim/worldgen.ts`, `src/sim/world.ts`)**:
   - Scaled noise-driven biome blending across endless streaming chunks (16×16m chunks with seamless eviction and state persistence).
   - Seeded cross-country highway grid (`isRoad()`) linking continental regions.
   - 5 procedural major POIs:
     - **Abandoned Towns (`🏘️`)**: Street crossings, wrecked cars, vintage fuel pumps, supply crates, tents.
     - **Gas Stations (`⛽`)**: Paved forecourts, fuel pumps, scrap salvage.
     - **Radio Towers (`📡`)**: Hilltop antennas that ping and reveal a massive 450m radius across the map.
     - **Military Outposts (`🛡️`)**: Sandbag redoubts, weapon crates, barricades.
     - **Survivor Camps & Settlements (`⛺`)**: Wilderness campsites and log shelters.
   - **Town Spawn**: New games and respawns place the player, truck, and pet directly in the town square.
3. **Cartography & Dynamic Fog of War (`src/ui/ui.ts`)**:
   - **Corner Compass Minimap**: Brass-rimmed radar in top-right HUD tracking terrain, roads, POI symbols, truck, pet, and player heading.
   - **Interactive World Map (`[TAB]`)**: Full-screen parchment canvas with pan/zoom, coordinate tracking, continuous sector discovery (42m on foot, 70m driving, 450m via radio tower), and mystery cross-hatch fog.
   - Dedicated `🏘️ Spawn at Town` fast-travel controls.
4. **Vehicles & Arcade Driving Physics (`src/sim/game.ts`)**:
   - Pickup truck with steering, acceleration, braking, and reverse.
   - Fuel consumption and siphoning/refueling from fuel canisters.
   - Durability degradation and scrap metal repairs.
   - Trunk storage inventory (accessible via `[F]` trunk interaction).
   - Kinetic ramming mechanics: high-speed collisions crush zombies into green smoke puffs.
   - Car horn (`[H]`) with sound FX and zombie sound attraction.
5. **Companion Pet System (Barkley the Hunting Dog)**:
   - Follows the survivor with distance-based leash physics.
   - Restores sanity when petted (`[E]`).
   - Threat detection: barks/growls at approaching zombies and bites them in combat.
6. **Survival & Night Terror Mechanics**:
   - Hunger, Health, and Sanity meters.
   - 20-minute day/dusk/night planetary cycle.
   - Lethal darkness: leaving light sources drains sanity and inflicts darkness damage when terror sets in.
7. **Base Building & Construction (`src/sim/base.ts`)**:
   - Claim pennants establishing protected build zones.
   - Grid-socketed construction (wooden walls, wooden floors, storage chests, campfires).
8. **Audio Engine (`src/audio/sound.ts`)**:
   - 100% Web Audio procedural sound synthesizer (footsteps, axe swings, wood chops, pickaxe clinks, vehicle engines, dog barks, zombie groans, horn honks, radio pings). Zero external audio asset dependencies.

---

## 3. The Current Strategic Mandate

> ⚠️ **Focus Directive**: We want to **deepen the Engine and Sandbox mechanics first** before scripting linear game loops (such as story quests, rigid objective chains, or endgame win screens).

We want the world to feel like an extraordinarily rich, tactile, unpredictable physics playground where interacting with *anything* yields delightful reactions, slapstick consequences, and emergent survival opportunities.

**Side track since 2026-10-20 — the Unreal arena (phase Z34).** The Aim Arena is being rebuilt as a finished demo in Unreal Engine 5.8.3 (project `C:\W4B\W4BArena\World4Bros.uproject`), on Unreal's First Person C++ template (Arena Shooter variant) plus the Twin Stick camera code, in `C:\W4B\W4BArena` (private repo). The web game stays the source of truth for arenas, guns, bot levels and rules (`npm run ue:export`). Start with `docs/UE_ARENA_PLAN.md` (steps U0–U14 with Antigravity prompts) and `docs/UE_PIPELINE.md` (tested commands and known traps); Antigravity uses the skill `.agents/skills/w4b-unreal/`. Where to resume: World Bible §69.

---

## 4. Key Sandbox Pillars to Design with Claude (from the World Bible)

### Pillar A: The 5-State Interactive Object Architecture (§0 & §9)
Every object in the world should conform to five standard interactive states:
1. **`idle`**: Resting with subtle ambient breathing / wind sway.
2. **`hover`**: White inked outline highlight, prompt tooltip, gentle squash ready for interaction.
3. **`hit`**: Directional recoil away from impact, wood chips/sparks/debris particles, comic onomatopoeia popup (`"WHACK!"`, `"BONK!"`, `"CLANG!"`), hit-stop freeze (30–45ms).
4. **`destroy`**: Cartoon collapse, burst of loot arcing out with gravity bounce, cloud of dust or green pop.
5. **`respawn`**: Pop-up book unfold scale tween with wobble.

### Pillar B: Slapstick Goofy Objects & Gadgets (§3 & §4)
The World Bible specifies a huge variety of sandbox toys to design:
- **`rubber_chicken`**: Low damage, huge knockback, rubbery stretch, pitch-varied squeaks.
- **`banana_peel_patch`**: Step on it and spin-slide 3 meters with circling stars; placeable as enemy stun trap.
- **`whoopee_boulder` & `whoopee_cushion_trap`**: Emits comical `pbbbt` sound that stuns/embarrasses enemies and scatters birds.
- **`springy_stump` & `boing_boots`**: Launches players high into the air with bouncy spring recoil and dust rings.
- **`suspicious_bush`**: Creeps closer only when the camera looks away; drops loot when caught.
- **`vending_machine_from_nowhere`**: Insert gold coins for wacky gadgets, or kick it to risk catching fire.
- **`flush_teleporter`**: Outhouse with flush sound that pops you out of another outhouse across the map.
- **`disco_stump`**: Forces nearby players and critters to dance, regenerating sanity.

### Pillar C: Environmental Reactions & Biome Sandbox Hazards (§1)
Each biome should act as a distinct physical sandbox system:
- **Squelchy Bog**: Mud terrain reduces walking speed by 40%; bouncy swamp mushrooms; explosive gas bubbles that can be detonated near enemies.
- **Sunburn Flats**: Heat-haze distortion shader; thirst/temperature meter; vanishing oasis mirages; tumbleweed buddies rolling past with googly eyes.
- **Brrington Tundra**: Zero-friction ice skating physics; snowdrifts; snowmen watchers whose heads swivel when you turn away.
- **The Hollow**: Extreme sanity drain; chromatic aberration; whispers; shadows with glowing eyes that retreat from lantern light.
- **Scrapyard Gulch**: Spring traps, conveyor belts, magnetic cranes, and scrap piles.

**Status effects and the biome atlas (design approved 2026-10-19)**: twelve effects (acid, burn, explosion, stagger, electricity, cold, spores, goo, wet, oil slick, blind, smoke), a data-driven reaction table, the stagger ladder, and a 16-biome atlas with 6 proposed biomes. Spec in World Bible Part 18 (§115–§120); planned as phases Z26 and Z27 (§69). Visual reference: `design/index.html` and `design/biomes.html`.

### Pillar D: Expanded Transport & Physics Sandbox (§6, §38, §44)
Expand vehicles beyond just driving:
- **Part-Based Car Health**: 5 distinct components (`battery`, `spark_plugs`, `radiator_coolant`, `engine_block`, `tires`) requiring diagnostic checks and salvage.
- **Trailers & Towing**: Pendulum hitch physics with wobble, load distribution, and rollover risks.
- **Bikes & Watercraft**: Bicycles (speed boosted by stamina/food) and rafts/boats with water wake ripple physics.

### Pillar E: Visual Shell & Pop-Up Cardboard Diorama (§0 & §68)
- 3D diorama papercraft presentation: 2.5D isometric projection with tilt-shift depth-of-field.
- Pop-up hinged roof flaps: Walking into a building doorway smoothly folds the cardboard roof upright ($-77^\circ$) to reveal the interior without cutscenes.
- Hiding spots: Wardrobes, dumpsters, and beds with cartoon peeking eyes.

### Pillar F: Factory Automation & Belt Logistics (Factorio & Satisfactory Integration, §82–§87)
- **Discrete Geode Nodes & Purity Tiers**: Impure (0.5x), Normal (1.0x), Pure (2.0x), Jackpot (4.0x) nodes for Iron, Copper, Coal, Limestone, Oil, Bauxite, and Glow Crystals.
- **Continuous Belt Logistics**: Wooden chute belts, rubber conveyor belts, high-speed mag-belts, and vertical conveyor lifts.
- **Processing Loops**: Smash crushers, blast furnaces, assemblers, and oil cracker refineries with modular power shard overclocking sockets.
- **Power Grid & Fluids**: Burner engines, steam boilers, solar arrays, Tesla accumulator vaults, and pressurized pipe manifolds.
- **Macro Logistics**: Automated freight tractors, narrow-gauge trains with cartoon cowcatchers, and pneumatic hypertubes for high-speed slapstick player transit.
- **Industrial Smog & Acoustic Heatmaps**: Factory smoke and machine noise create dynamic environmental aggro ripples.

### Pillar G: Colony Overseer, Pawn Psychology & Storyteller AI (RimWorld Integration, §88–§93)
- **Quirky Survivors with Traits & Passions**: Grease Monkeys, Pyromaniacs, Gourmands, Slackers, and Brawlers with 1-flame and 2-flame work passions.
- **1-to-4 Priority Work Matrix (`Key: O`)**: Automated delegation across Firefight, Doctor, Build, Farm, Mine, Haul, Craft, Guard, and Research.
- **Mood Dynamics & Slapstick Mental Breaks**: Binge eating canned beans, locking in outhouses, tool tantrums, conveyor belt carousels, and feather pillow berserk brawls.
- **Colony Habitability & Slapstick Medicine**: Impressive bedrooms, banquet halls (no "ate without table" misery!), medical triage, and goofy prosthetics (peg legs, turbo spring boots, socket wrench arms).
- **Dynamic Storyteller AI**: *Caspar Chill* (peaceful pacing), *Randy Ruckus* (pure chaotic comedy), and *Hilda Hardcore* (ruthless tactical scaling).
- **Faction Diplomacy & Captive Rehabilitation**: Trade caravans, guest rooms, bandit holding cells, and peaceful survivor conversion.

### Pillar H: Evolved Zombie Threat & Industrial Siege Dynamics (§94–§96)
- **Pollution-Evolved Mutations**: Sapper Zombies carrying explosive kegs toward power grids, Acid Spitters melting conveyor belts, Tunnel Burrowers popping into barracks, Screamer Queens triggering chain swarms, and Goliath Demolishers wielding telephone poles.
- **Targeted Industrial Sabotage**: AI prioritizes generators, wire poles, and ammo supply belts.
- **Automated Base Defense Grids**: Belt-fed bolter turrets, pipe-fed liquid flame turrets, rubber mallet catapults, electric zap-fences, and automated repair drone bays.

### Pillar I: PvPvE Territorial Warfare & Base Raiding (§97–§100)
- **Tri-Zone World Topology**:
  - *Green Zones*: Civilized Havens (zero PvP damage, peaceful trading hubs, starter camps).
  - *Yellow Zones*: Contested Biomes (opt-in faction flagging, formal clan war declarations).
  - *Red Zones*: Lawless Deep Wilderness (full open PvPvE, full loot drops, 80% of pure/jackpot nodes).
- **Clan Claim Totems & Active Raid Windows**: Scheduled 2-hour daily vulnerability windows with impenetrable Tesla shields during offline hours to prevent offline griefing.
- **Tactical Raiding & Ambushes**: C4 plunger charges with disarm timers, vehicle rams, supply convoy ambushes with spike strips, and black market radio bounties.

---

## 5. Recommended Workflow Between Claude & Antigravity

1. **Claude's Role (Architect & Systems Designer)**:
   - Design schemas, data contracts, and interaction state machines.
   - Design the visual shell (HTML/CSS layouts, tactile paper UI components, responsive mobile controls).
   - Formulate the physics formulas, juice curves, and sensory feedback loops (spring constants $f$ & $\zeta$, loot arc gravity $g$, onomatopoeia triggers).
   - Design the machine recipes, conveyor slot routing, and pawn work prioritization algorithms.
2. **Antigravity's Role (Engine & Implementation Lead)**:
   - Translate Claude's designs into clean, typed TypeScript in `src/`.
   - Maintain 100% test coverage and verify all edge cases.
   - Bundle production builds and deploy live to Vercel for instant browser testing.

---

## 6. Questions & Decisions for Claude to Begin Designing

1. **Automation Grid & Belt Ticking**: How should conveyor belt slots and item entities be structured in the ECS for 60Hz deterministic throughput without bogging down mobile CPU budgets?
2. **Pawn Job Reservation & Pathing**: What is the most lightweight, tick-budgeted reservation system for survivor pawns hauling goods to/from conveyor hoppers and chests?
3. **Territorial PvPvE Boundary Validation**: How should clan territory totems and raid vulnerability states be networked between client and authoritative server?
4. **Juicy Industrial Soundscapes**: What procedural Web Audio synthesis profiles should be generated for conveyor clicks, steam boiler chuffs, pneumatic hypertubes, and automated bolter fire?
