# World 4 Bro's: Core

The pure TypeScript, zero-dependency post-apocalyptic survival, tactical combat, and base-building game engine rendered in an ink-and-hatching comic style. Built to be extended by developers and coding agents.

---

## 🚀 Quick Start

```bash
npm install
npm run dev        # Starts Vite dev server (e.g., http://localhost:5174)
npm run verify     # Typecheck (tsc --noEmit) + full unit/integration test suite
npm run test       # Run 120+ Node.js native tests
npm run bundle     # Builds single self-contained playable dist/single.html
```

### Useful URL Query Flags
- `?seed=4242` — Set specific world seed
- `?mode=zombie` — Enable zombie survival mode (default: classic)
- `?creative=1` — Infinite materials & instant building
- `?panel=arena` — Launch directly into the Army Base Combat Arena & Fight Bench
- `?panel=gunsmith` — Open Gunsmith Modding Bench
- `?panel=range` — Open Ballistics Proving Grounds

---

## 🎮 Key Controls & Hotkeys

| Key | Action | Description |
| :--- | :--- | :--- |
| **WASD** | Movement | Move player character |
| **Shift** | Sprint | High-speed dash |
| **Right Click / Aim** | Aim Stance | Locks aim angle & raises equipped firearm |
| **Left Click / Space** | Shoot / Action | Fire equipped weapon, harvest, chop, mine |
| **1 – 8** | Hotbar | Equip weapon, tool, or consumable item |
| **H** | **Arena [H]** | Open Army Base Combat Arena & Fight Picker Menu |
| **Y** | **Range [Y]** | Teleport to Ballistics Proving Grounds (5m–100m lanes) |
| **U** | **Gunsmith [U]** | Open Gunsmith attachment workbench |
| **F2 / ~** | **DevLog [F2]** | Open Dev Log & Bug Tracking Console |
| **F8** | **Toasts [F8]** | Toggle Toast Notifications (ALL, QUIET, OFF) |
| **J** | **SwarmLab [J]** | Open Headless Swarm Testing & Weapon Balance Lab |
| **V** | **Garage [V]** | Vehicle fleet spawn & driving interface |
| **I / R / B** | Inventory / Craft / Build | Core survival backpacks, crafting, base claims |
| **C** | Character | Layered outfit & appearance customizer |
| **F3 / F5** | Debug / Save | Toggle FPS/pos HUD (`F3`) and quicksave (`F5`) |

---

## 🏟️ Army Base Combat Arena & Fight Bench (`Arena [H]`)

Located at the **Army Base & Proving Grounds Complex** (`rx = spawn.x - 24, ry = spawn.y - 12`):
- **Duel Pit (40x24m)**: Symmetrical 1v1 tactical layout with bullet-stopping crates, sandbags, and low walls.
- **FFA Yard (64x40m)**: 8 player spawns, perimeter walls, and crossfire corridors.
- **4 Autonomous Bot Tiers (`Comp.Raider`)**:
  - 🌾 **Scarecrow** (50 HP, wide spread `±12°`, 1.2s delay): Target practice dummy.
  - 🎖️ **Regular** (65 HP, balanced strafe `±7°`, 0.8s delay): Standard fair duel.
  - ⚡ **Sharp** (80 HP, agile kiting `±3.5°`, 0.55s delay): Fast target acquisition.
  - 💀 **Menace** (100 HP, laser aim `±1.5°`, 0.35s delay): Deadly flanking ace.
- **Zombie Horde Swarms**: Release 1x, 3x, 6x, or 12x Walkers, Runners, Brutes, or Screamers.
- **🔥 3-Way Battle Royale**: Pit bots against zombies and player in an all-out tactical crossfire.

---

## 🎯 Ballistics Proving Grounds & Firing Line (`Range [Y]`)

- **Multi-Distance Precision Lanes**: Steel gongs & target dummies stationed at **5m, 10m, 15m, 20m, 25m, 30m, 40m, 50m, 60m, 75m, and 100m**.
- **Distance Damage Falloff**: Continuous line-segment projectile collisions with damage attenuation curves based on weapon range.
- **Wobbling Steel Targets**: Live hit feedback with impact damage and distance toast readout.
- **Quarantined Containment Pens**: Observation decks housing live mutant specimens.

---

## 🛠️ Gunsmith Workbench & Weapon Mods (`Gunsmith [U]`)

- **5 Weapon Archetypes**: Hunting Rifle, Pump Shotgun, Nailgun SMG, Potato Cannon, and Slingshot.
- **20+ Attachments**: Homemade suppressors, tactical flashlights, extended magazines, red dot sights, bayonets.
- **Stat Recalculation**: Live modifications to damage, spread, fire rate, noise radius, bullet speed, and recoil.
- **Visual Flags & Audio**: Visible tape, bayonets, lasers, and goofy sounds (rubber duck squeaks, car horns).

---

## 🐛 Dev Log & Bug Engine (`DevLog [F2]`)

- **Live Diagnostic Stream**: Color-coded tracking for bugs, errors, combat events, and world simulation.
- **Global Error Trapping**: Automatically captures unhandled exceptions and promise rejections with stack traces.
- **One-Click Export**: Copy a complete Markdown bug report to clipboard or download a structured JSON diagnostic snapshot.
- **Gemini Asset Generator Helper**: Interactive prompt recipes for generating custom wear, weapons, props, and tilesets.

---

## 📚 Technical Documentation

- [Gemini Asset Generation Guide](docs/GEMINI_ASSET_GENERATION_GUIDE.md) — Style bible, prompts, and schemas for AI generation.
- [Dev Log & Bug Tracking Manual](docs/DEV_LOG_AND_BUGS.md) — Runtime logging, error trapping, and diagnostic API.
- [Aim Arena & Test Bench Architecture](docs/ARENA.md) — Layout rules, symmetry tests, bot specs, and line-of-sight checks.
- [Weapons & Ballistics Reference](docs/WEAPONS.md) — Projectile physics, spread models, and attachment matrices.
- [Swarm Telemetry & Headless Balance](docs/SWARM_TELEMETRY.md) — Deterministic headless simulations and paired-seed tests.
- [Architecture & Engine Design](docs/ARCHITECTURE.md) — ECS, worldgen stages, renderer, and simulation loop.
- [Roadmap](docs/ROADMAP.md) — Done and next phases, including Z26 status effects and Z27 biome atlas.
- Status effects and biome atlas design pages — `../docs/design/index.html` and `../docs/design/biomes.html` (spec in World Bible Part 18).
