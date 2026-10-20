# Visual Studio Process for the Unreal Arena (and what the "engine from scratch" guide misses)

Written 2026-10-20 as an answer to a pasted guide, "Building a game and custom engine from scratch with Visual Studio and an agentic IDE" (GLFW + OpenGL + a hand-made ECS, 6 layers, 4 prompts). This file says what to keep from it, what it misses, and gives a step-by-step Visual Studio process for **our** project. Related: [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md) (steps U0-U14), [UE_WORKFLOW.md](UE_WORKFLOW.md) (who does what), [UE_PIPELINE.md](UE_PIPELINE.md) (commands and traps).

## 1. Short answer: do not write an engine; the six layers already exist

| Guide layer | In our Unreal arena | In our web game | Our work |
| :--- | :--- | :--- | :--- |
| 1 Platform (window, input) | Unreal: window, Enhanced Input, gamepads | browser + `src/engine/input.ts` | input actions in C++ (U3) |
| 2 Renderer | Unreal: Lumen, Nanite, shadows, post-process | Canvas now, PixiJS planned (Z31) | lighting and materials (U10) |
| 3 Core (math, assets) | Unreal math, Interchange import (glTF/FBX) | TypeScript + `src/data` registries | `npm run ue:import`, data JSON (U4) |
| 4 ECS rig | Actors + Components (and Mass for crowds) | **already an ECS**: `src/engine/ecs.ts` | C++ components: health, weapon, bot brain (U4-U6) |
| 5 Loop and physics | Unreal tick, Chaos physics, fixed step with `-benchmark` | deterministic sim loop | match rules, Test Bench (U7, U9) |
| 6 Game code to beta | our C++ in `Source/W4BArena/W4B/` | `src/sim`, `src/ui` | U3-U14 |

A hand-made engine (layers 1-5) is months of work before the first bot fight and would still lack an editor, importers, animation, navigation, audio and packaging. The guide's own "Beta" (playable start to finish without crashes) is what our U14 delivers on Unreal. Keep the guide's good ideas: layer-by-layer order, the **Prompt → Verify → Refactor** cycle, tweakable numbers at the top (ours are in `src/data` → `game_data.json`), and pasting crash logs back to the agent.

## 2. The max list: what else you need

Status: ✅ have, 🟡 started, ❌ missing.

### 2.1 Tools and setup
- ❌ **Visual Studio C++ for Unreal** (Game development with C++, MSVC 14.44, Windows 11 SDK, IDE support for Unreal): the blocker for U1. Install by hand (UE_ARENA_PLAN.md U0).
- ✅ Unreal 5.7.3, Blender 5.2, git + Git LFS on PATH, `C:\W4B`, 68 GB free.
- ❌ Private GitHub repo `world4bros-ue` for the Unreal project (U1).
- 🟡 GitHub `main` now holds a different project (swarm/fitness); the game is on `game-main`. Decide which project owns `main`.
- ❌ Copilot instructions for the Unreal repo (`.github/copilot-instructions.md`, template in §4) so Copilot in Visual Studio follows the same rules as Antigravity.
- ❌ Unreal **editor debug symbols** (Epic Launcher > Unreal 5.7 > Options > "Editor symbols for debugging", tens of GB): only if you need to step into engine code; skip while disk is tight.
- ❌ A gamepad for testing twin-stick controls.
- ✅ Antigravity skill `w4b-unreal`; ✅ tested pipeline `npm run ue:*`.

### 2.2 Engineering practices the guide does not mention
1. **Source control discipline**: one commit per approved step, Git LFS for assets, never commit `Binaries/Intermediate/Saved/DerivedDataCache`.
2. **Automated tests** (Unreal automation tests, run headless) for every rule: damage, rounds, spawns.
3. **A data pipeline**: numbers live in data, not code (`npm run ue:export`).
4. **Logging with categories** (`DEFINE_LOG_CATEGORY(LogW4B)`), and markers agents can grep.
5. **Crash handling**: read `Saved/Crashes`, reproduce with F5 in Visual Studio.
6. **Profiling and rendering**: `stat unit`, `stat gpu`, Unreal Insights; DirectX 12, Lumen hardware ray tracing, TSR/DLSS, Reflex, scalability presets, frame targets: UE_PIPELINE.md section 10b.
7. **Performance budgets**: bots, projectiles and decals pooled; tick only what must tick.
8. **Code review**: Copilot or a second agent reviews every diff before you approve.
9. **Asset rules**: names, scale check, collision, licences in `CREDITS.md`.
10. **Save and settings**: graphics, volume and key bindings saved (`UGameUserSettings`).
11. **UI flow**: title, pick a match, HUD, results, pause, settings.
12. **Input remapping and gamepad navigation** in menus.
13. **Accessibility**: readable fonts, colour-blind-safe team colours, aim assist option, subtitles for stingers.
14. **Audio**: sounds, mixing, volume sliders.
15. **AI debugging**: Gameplay Debugger (`'` key), Visual Logger.
16. **Determinism notes**: seeds recorded; Unreal is not fully deterministic.
17. **Packaging and smoke tests**: a build that runs from the desktop, tested on a second PC if possible.
18. **Versioning and release notes**: `W4BArena-demo-<date>.zip` with a README.
19. **Backups**: private repo pushed after every step; the downloads folder is not the only copy of bought assets.
20. **Documentation**: the master plan and the docs updated after every step (already a rule).

### 2.3 Game features a finished arena demo needs (beyond UE_ARENA_PLAN.md §1)
Kill feed, hit markers, damage numbers off by default, reload and ammo display, weapon switch, spawn protection effect, round start countdown, round-end slow motion (optional), scoreboard (Tab), results with accuracy and time to kill, "play again" with the same settings, pause with restart, settings (graphics, audio, controls), camera zoom limits, cursor that shows the aim point at the combat plane, bots that announce their level, a short tutorial overlay for controls.

### 2.4 Only if you ever build your own engine (not recommended now)
Everything in 2.2 plus: window and input layer, renderer (shaders, materials, shadows, batching), asset importers, scene format and serialization, an editor or level tool, animation system (skeletons, blending, retargeting), physics and collision queries, navigation and pathfinding, audio engine, UI framework, threading and job system, memory allocators, hot reload, profiler, crash reporter, packaging, platform builds. The web game already has a small engine of its own (ECS in `src/engine`, deterministic sim, Canvas renderer, PixiJS planned); that is where custom-engine ideas belong.

## 3. Step by step: Visual Studio in our project

### V0 — Install (once)
1. Visual Studio Installer > **Modify** Visual Studio 2026: workloads **Game development with C++** and **Desktop development with C++**; individual components **MSVC v143 … x64/x86 build tools (v14.44)**, **Windows 11 SDK (10.0.22621.0)**, **IDE support for Unreal Engine**. Wait until it says done.
2. In `world4bros-core`: `npm run ue:doctor`. Every line PASS except `W4B_UE_PROJECT` (that comes in U1).

### V1 — Open the project (after U1 created it)
1. Right-click `C:\W4B\W4BArena\W4BArena.uproject` > **Generate Visual Studio project files**.
2. Open `W4BArena.sln`. Toolbar: **Development Editor** + **Win64**. Right-click **W4BArena** > **Set as Startup Project**.
3. **Build > Build Solution** once (Ctrl+Shift+B). The first build takes a few minutes; later ones seconds.
4. **F5**: the Unreal Editor starts with the debugger attached. Press Play, Esc to stop.
5. Add `C:\W4B\W4BArena\.github\copilot-instructions.md` (§4) and commit it.

### V2 — Daily use (each U-step)
1. Antigravity writes the code and builds (UE_WORKFLOW.md §3). Visual Studio stays open but **you do not build in it while Antigravity builds**.
2. When Visual Studio says files changed outside, click **Yes to all**.
3. **Git Changes** window: read the diff of the step; ask Copilot Chat "Review these changes for bugs, null pointers and Unreal mistakes (UPROPERTY, garbage collection, Tick cost)".
4. Play in the editor; if something is wrong, debug (V3) before asking Antigravity to fix it, so you can tell it the exact line.

### V3 — Debugging recipes
| Problem | In Visual Studio |
| :--- | :--- |
| A function does the wrong thing | Breakpoint (F9) in it, F5, Play, trigger it; hover variables; F10 step over, F11 step in |
| The editor is already running | Debug > Attach to Process > `UnrealEditor.exe` |
| Crash | F5 and reproduce: Visual Studio stops on the crash line; copy the **Call Stack** window to Antigravity |
| Crash in a packaged build | Open the `.dmp` from `Saved/Crashes` with File > Open, then **Debug with Native Only** |
| Value changes and you do not know where | Data breakpoint (Debug > New Breakpoint > Data Breakpoint) on the address |
| Bot does something odd | In the game press `'` (Gameplay Debugger) to see its perception and path; breakpoint in `UW4BBotBrainComponent` |
| Slow frame | Unreal Insights (`-trace=default`), or Visual Studio **Debug > Performance Profiler > CPU Usage** on `UnrealEditor.exe` |
| Small C++ change with the editor open | Save in Visual Studio, then **Ctrl+Alt+F11** in the editor (Live Coding). Header layout changes: close the editor, Build, F5 |

### V4 — Tests in Visual Studio
With "IDE support for Unreal Engine" installed, **Test > Test Explorer** lists the Unreal automation tests (`W4B.*`) after a build; run them there or headless with the command in UE_PIPELINE.md §4.5.

### V5 — Copilot in Visual Studio: prompts by layer
Use Copilot Chat (Agent mode if your version has it) for questions, reviews and small fixes; big changes go through Antigravity so only one agent writes at a time. The guide's four prompts, rewritten for our project:

1. **Layer 1, input (U3)**: "In `Source/W4BArena/W4B/Core/W4BPlayerController.cpp`, explain how the Enhanced Input actions are created in code and list anything that would break gamepad support. Do not change files."
2. **Layer 2, look (U10)**: "List every material instance and light the arena uses and check them against `docs/STYLE_BIBLE.md` section 0.3 (palette) and the sun rule in section 0.2."
3. **Layer 4, components (U4-U6)**: "Review `UW4BHealthComponent`, `AW4BWeapon` and `UW4BBotBrainComponent` for: missing `UPROPERTY()` on UObject pointers, ticking when not needed, work done every frame that could be a timer, and data that should come from `game_data.json`."
4. **Layer 6, tweaking (U4-U9)**: "Which numbers in our C++ are hard-coded but should be in `src/data` (exported by `npm run ue:export`)? Give a table: file, line, value, proposed JSON field."
5. **Beta, crash**: "Here is the call stack and log from `Saved/Crashes/...`. Find the cause in our code (not engine code) and propose the smallest fix."

### V6 — Release (U14)
1. Configuration **Shipping** + **Win64**, Build once to catch Shipping-only errors (for example code inside `#if !UE_BUILD_SHIPPING`).
2. Packaging runs through Antigravity's RunUAT command; then run the `.exe` outside Visual Studio.

## 4. Template: `C:\W4B\W4BArena\.github\copilot-instructions.md`
```markdown
# World 4 Bro's Arena (Unreal 5.7.3, C++)
- Our code lives in Source/W4BArena/W4B/. Template code (Variant_Shooter, Variant_Horror, FirstPerson) is reference: derive or copy, do not rewrite.
- Numbers come from Content/W4B/Data/game_data.json (exported from the web game by `npm run ue:export`); never hard-code gun, bot or rule values.
- Coordinates: 1 m = 100 units; bullets and bot sight use the combat plane at 90 cm.
- Collision profiles: W4BCover (stops bullets and sight), W4BBulletPass (barrels, trees).
- Every UObject pointer member needs UPROPERTY(). Prefer timers over Tick. Log with LogW4B.
- Never edit .uasset/.umap files. Never commit Binaries/, Intermediate/, Saved/, DerivedDataCache/.
- Steps and rules: docs/UE_ARENA_PLAN.md and docs/UE_PIPELINE.md in the world4bros-core repo.
```
