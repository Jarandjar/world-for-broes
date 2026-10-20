# Unreal Arena: Pipeline, Tools and Rules (reference for phase Z34)

The step-by-step build is [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md). Day-to-day execution flow between user, Antigravity and Visual Studio is in [UE_ARENA_WORKFLOW.md](UE_ARENA_WORKFLOW.md). This file is the reference behind the pipeline: what we build on, where everything lives, how data, levels, code, art, tests and builds flow, and every rule learned the hard way. Written 2026-10-20 after reading up on Unreal 5.7/5.8, reading the engine and template source on this PC, and testing every pipeline command on this PC (section 6).

Antigravity: load the skill `.agents/skills/w4b-unreal/SKILL.md` (short version of this file) before any Unreal work.

---

## 0. Summary

| Question | Answer |
| :--- | :--- |
| Engine | **Unreal Engine 5.7.3**, already installed. Stay on 5.7 for the demo (section 1.3 explains 5.8). |
| Base project | **Unreal's own First Person C++ template with the Arena Shooter variant**: bots that see, move, take cover and shoot (StateTree + perception), guns with ammo and pickups, health and team score, all in about 100 KB of plain C++. We add the top-down camera and aim from the **Top Down template's Twin Stick variant**. **Not Lyra** (section 1.2). **Approved by the owner 2026-10-20.** |
| Where | `C:\W4B\W4BArena\` (outside OneDrive, short path) in its own **private** GitHub repo with Git LFS. This public repo keeps docs, tools and exported data. |
| Data | The web game is the source of truth: `npm run ue:export` writes `tools/unreal/arenas.json` and `game_data.json` (guns, bot levels, match rules). Levels are built from the JSON; game code reads the JSON at runtime. |
| Levels | `npm run ue:arena` builds every arena headless in ~8 s and verifies it; `npm run ue:shots` saves a PNG of each. **Tested.** |
| Code | C++ in the template module; agents build with `Build.bat` (editor closed) or Live Coding through `npm run ue:remote` (editor open). |
| Art | Unreal first (materials, Modeling Mode); download CC0 (Kenney, Quaternius, Poly Haven); Blender only for new models; imports go through `npm run ue:import` with size checks. **Tested.** |
| Tests | Unreal automation tests from the command line; bot-vs-bot matches headless with `-benchmark -fps=30` (120 s of game in 8.6 s wall clock, **tested**). |
| Agents | Antigravity drives Unreal through scripts and Python remote execution (**tested**), not by editing binary assets. Workflows stopped working on 2026-10-19; we use an **Agent Skill**. |
| Blockers today | This PC has **no C++ compiler** (Visual Studio 2026 has only the core editor) and git is not on PATH. Fix in step U0. |

## 1. What we build on, and why

### 1.1 Recommended: the 5.7 templates (C++)
Unreal 5.6 added genre "variants" to the templates ([Epic, 5.6 templates](https://www.unrealengine.com/news/updated-game-templates-for-unreal-engine-5-6available-now)). Read on this PC in `C:\Program Files\Epic Games\UE_5.7\Templates\`:

| Template / variant | What it gives us (files) |
| :--- | :--- |
| First Person → **Arena Shooter** (`Variant_Shooter`) | `ShooterNPC` (HP, team byte, ragdoll death, aim cone `AimVarianceHalfAngle`), `ShooterAIController` (AI Perception sight + hearing, StateTree), `ShooterStateTreeUtility` (tasks: Sense Enemies, Shoot at Target, Face Actor, line-of-sight condition), EQS queries (roam, sniping spots), `ShooterWeapon` (projectile, magazine, refire rate, full/semi auto, recoil, shot noise for AI), `ShooterPickup` (weapon pickups with respawn), `DT_WeaponData` (pistol, rifle, grenade launcher), third-person rifle/pistol animation blueprints, team score in `ShooterGameMode`, UI widgets. |
| Top Down → **Twin Stick** (`Variant_TwinStick`) | `TwinStickCharacter`: spring-arm camera at -50° pitch, mouse aim (`GetHitResultUnderCursorByChannel`), right-stick aim with auto-fire, dash, knock-back; `TwinStickNPC` + `ST_TwinStickNPC`: chasers with RVO avoidance (our zombies later); spawner; projectile; AoE attack. |

The C++ template "includes code and content for all variants" (TemplateDefs.ini), the code is small and readable, and it is meant to be changed. That suits agents: almost everything is C++ and text, not binary assets.

### 1.2 Why not Lyra (my earlier advice, now withdrawn)
| Problem | Evidence |
| :--- | :--- |
| Your repo is **public**; Lyra/Fab content may not be published uncooked | `github.com/Jarandjar/world-for-broes` is public (GitHub API, 2026-10-20). Fab's licence allows sharing with collaborators in a private repo, not standalone redistribution ([Fab licences](https://dev.epicgames.com/documentation/en-us/fab/licenses-and-pricing-in-fab)). Template code is "Examples", which the EULA lets you share ([EULA note](https://forums.unrealengine.com/t/launcher-eula-update/15813)). |
| Disk and Git LFS | ~54-69 GB free on the only disk; Lyra + its editor build + shader cache is tens of GB. GitHub Free LFS is **10 GiB storage + 10 GiB/month bandwidth** ([GitHub docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage)). |
| Agent-hostile | Lyra is configured through Experiences, Game Feature data assets, ability sets and Blueprints: binary assets agents cannot diff or edit safely. |
| Top-down aim | Lyra's ranged-weapon ability traces from the camera; a top-down game needs engine-sample code changes. |
| No C++ compiler yet | Both need Visual Studio C++, but the template builds in minutes, Lyra in much longer. |

Lyra stays useful as a **reference** (animation, networking) in a separate folder, never in the repo. The earlier "Lyra map" table in the old plan was filled in before Lyra was installed; ignore it.

### 1.3 Unreal 5.8
5.8 (June 2026) adds an **experimental first-party MCP plugin** (AI agents connect to the editor), Lumen Lite, State Tree start states and orthographic viewport fixes ([5.8 release notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-8-release-notes)). It needs ~27 GB more disk and Epic recommends the 14.44 or 14.50 toolset for it ([forum, Epic staff](https://forums.unrealengine.com/t/ue-5-8-fails-to-compile-with-visual-studio-2026-hash-map-missing-source-fix/2732357)). Decision: **finish the demo on 5.7**; Python remote execution already gives agents live editor control. Revisit 5.8 at the decision gate (UE_ARENA_PLAN.md U14).

## 2. This PC (checked 2026-10-20) and what it needs

| Item | Found | Needed |
| :--- | :--- | :--- |
| CPU / RAM / GPU | Ryzen 5 8400F (6c/12t), 64 GB, RTX 5070 | fine (Lumen, Nanite, VSM OK) |
| Disk | one NVMe, C: with 54-69 GB free | ≥ 45 GB free before U1; keep ≥ 25 GB free always |
| Unreal | 5.7.3 at `C:\Program Files\Epic Games\UE_5.7` (26.6 GB) | done |
| Visual Studio | **VS 2026 Professional 18.11 with only the core editor; no C++, no MSVC, no Windows SDK** | see below |
| Git | Git 2.x + **Git LFS 3.7.1** at `C:\Program Files\Git`, **not on PATH** | add `C:\Program Files\Git\cmd` to PATH |
| Python | 3.15 on PATH; Unreal has its own 3.11 | fine |
| Long paths | enabled | still keep paths short |
| Blender | 5.2 | fine |

**Visual Studio components.** Unreal 5.7 officially supports VS 2022 17.8+; VS 2026 is "Experimental" ([Epic VS setup](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-visual-studio-development-environment-for-cplusplus-projects-in-unreal-engine)). The engine's own `Engine/Config/Windows/Windows_SDK.json` says: preferred MSVC **14.44.35207+**, banned 14.39-14.43 and 14.44.0-14.44.35210, minimum 14.38, Windows SDK 10.0.22621, minimum VS 2026 version 18.0. The newest VS 2026 toolset (14.5x) can break Unreal headers, so install the 14.44 toolset next to it; Unreal picks it automatically because it is the preferred version. Owner, in any **PowerShell** window (it asks for admin rights itself; close Visual Studio first), or tick the same items in Visual Studio Installer > Modify:

```powershell
Start-Process "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" -Verb RunAs -Wait -ArgumentList 'modify --installPath "C:\Program Files\Microsoft Visual Studio\18\Professional" --add Microsoft.VisualStudio.Workload.NativeGame --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Component.Unreal.Ide --add Component.Unreal.Debugger --includeRecommended --passive --norestart'
```

The 14.44 toolset is installable in VS 2026 (GitHub's VS 2026 runner images ship its `VC.14.44.17.14` components; Microsoft added 14.30-14.43 to the VS 2026 installer in February 2026, [MSVC blog](https://devblogs.microsoft.com/cppblog/msvc-build-tools-versions-14-30-14-43-now-available-in-visual-studio-2026/)). If the installer still rejects an ID, tick "MSVC v143 ... build tools (v14.44)" under Individual components by hand, or install the free **Build Tools for Visual Studio 2022** with the same items. To force a toolset, `%APPDATA%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml`:
```xml
<?xml version="1.0" encoding="utf-8" ?>
<Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">
  <WindowsPlatform><CompilerVersion>14.44.35207</CompilerVersion></WindowsPlatform>
</Configuration>
```
`npm run ue:doctor` checks all of this and prints the fix for each problem.

**Disk budget for the demo**: Visual Studio C++ ~10-15 GB, project + C++ build ~3-6 GB, shader cache (shared, `%LOCALAPPDATA%\UnrealEngine\Common\DerivedDataCache`) ~5-10 GB, downloads ~1-5 GB, packaged builds ~1-2 GB each. About **25-45 GB**.

## 3. Where things live

```
C:\W4B\W4BArena\                 Unreal project (private repo "world4bros-ue", Git LFS)  <- W4B_UE_PROJECT points here
  W4BArena.uproject
  Config\DefaultEngine.ini        includes tools/unreal/pipeline/DefaultEngine.W4B.ini (merged by hand once)
  Source\W4BArena\                template code (Variant_Shooter, Variant_Horror) + ours in W4B\Core, W4B\Combat, W4B\AI, W4B\UI, W4B\Tests
  Content\W4B\Maps\               L_W4B_DuelPit, L_W4B_FFAYard, ... (built by build_arena.py)
  Content\W4B\Data\               arenas.json, game_data.json (copied by npm run ue:export; staged as non-asset files)
  Content\W4B\{Input,Materials,Props,Characters,Weapons,FX,Audio,UI}
  Content\FirstPerson, Content\Variant_Shooter ...   template content, left as it is
  Binaries\ Intermediate\ Saved\ DerivedDataCache\   generated, never committed
C:\W4B\Builds\                    packaged demo builds
C:\Users\Jalle\OneDrive\Desktop\BUCKS\game\world4bros-core\   this repo (public)
  tools\export-unreal.ts          npm run ue:export
  tools\unreal\arenas.json, game_data.json   generated, committed (small)
  tools\unreal\pipeline\          the tested Unreal scripts (section 4)
  .agents\skills\w4b-unreal\      the Antigravity skill
  docs\UE_PIPELINE.md, docs\UE_ARENA_PLAN.md
```

Why not inside `world4bros-core`: OneDrive syncs and locks the thousands of files Unreal writes (`Saved`, `Intermediate`, shader cache), the path is already ~70 characters long, and the public repo must not hold Epic/Fab content. **`C:\W4B` is part of the project; never clean it up as "old files".**

Environment variables (set once with `setx`, then open a new terminal):
| Variable | Value |
| :--- | :--- |
| `W4B_UE_PROJECT` | `C:\W4B\W4BArena\W4BArena.uproject` |
| `W4B_UE_ROOT` | only if Unreal is not in `C:\Program Files\Epic Games\UE_5.7` |
| `W4B_UE_PROJECT_NAME` | `W4BArena` (default; used by `ue:remote` to pick the right editor) |

Private repo `.gitignore`: `Binaries/ DerivedDataCache/ Intermediate/ Saved/ .vs/ .vscode/ *.sln *.suo *.VC.db *.VC.opendb compile_commands.json Plugins/*/Binaries/ Plugins/*/Intermediate/`. `.gitattributes`: `*.uasset *.umap *.fbx *.glb *.blend *.png *.tga *.wav *.ogg` → `filter=lfs diff=lfs merge=lfs -text`. Keep LFS under 10 GiB: commit art once it is approved, not every experiment.

## 4. The pipelines

```
web game (src/sim, src/data) --npm run ue:export--> arenas.json + game_data.json --+--> build_arena.py --> levels --> verify_arena.py --> shots (PNG)
                                                                                   +--> Content/W4B/Data --> C++ data subsystem (runtime)
Blender / CC0 / Fab --> glTF/FBX (metres) --> npm run ue:import (size checks) --> materials (master + instances) --> art dressing over the blockout
C++ (agent) --> Build.bat or Live Coding --> automation tests --> headless bot matches (-benchmark) --> results JSON --> playtest (owner) --> package
```

### 4.1 Data
- Source of truth: `src/sim/arena.ts` (layouts), `src/data/items.ts` (guns), `src/data/arenaBots.ts` (bot levels, also used by the web sim), `src/data/arenaRules.ts` (match rules, ARENA.md §4).
- `npm run ue:export` writes `tools/unreal/arenas.json` (with `solids`: the exact collision rectangles and circles the web sim uses) and `tools/unreal/game_data.json`. With `W4B_UE_PROJECT` set it also copies both into `Content/W4B/Data/`.
- `arenas.json` is generated: **add arenas in `src/sim/arena.ts`**. An arena typed only into the JSON is kept with a warning (today: `square_arena`).
- Coordinates: game metres, x east, y south, origin top-left. **Unreal X = (x − width/2)·100, Y = (y − height/2)·100. No Y flip**: Unreal is left-handed, so seen from above with +X to the right, +Y points down the screen, exactly like the web game. (The old rule "y → −Y" mirrored the arenas.)
- Runtime: a C++ `UW4BDataSubsystem` reads `Content/W4B/Data/*.json` with `FJsonObjectConverter` into USTRUCTs at startup (console `w4b.ReloadData` for live tuning). Packaging: `DefaultGame.ini` → `[/Script/UnrealEd.ProjectPackagingSettings]` `+DirectoriesToAlwaysStageAsNonUFS=(Path="W4B/Data")`.

### 4.2 Levels (tested)
- `npm run ue:arena` = `build_arena.py` + `verify_arena.py`, headless, ~8 s each, for every arena (or `npm run ue:arena -- duel_pit`).
- Built per arena, all tagged `W4BGenerated`: floor, roads (visual), perimeter walls, crates/low walls/sandbags (profile `W4BCover`: block movement, bullets, sight), barrels/trees (profile `W4BBulletPass`: block movement only), PlayerStarts facing the centre, pie markers (`TargetPoint` tagged `W4B.Pickup.Pie`; game code spawns the pickups), NavMeshBoundsVolume, sun, sky light, atmosphere, fog, preview camera (`W4B.PreviewCamera`).
- Rebuilds delete only `W4BGenerated` actors: hand-placed art survives (tested).
- Heights: crate 1.4 m, low wall 1.1 m, sandbags 1.0 m, barrel 1.0 m, tree 4.0 m; bullets and bot sight run in a flat **combat plane at 0.9 m**, so everything that stops bullets is taller than it (the web sim is 2D).
- `npm run ue:shots` renders each level off-screen from its preview camera into `<project>/Saved/W4B/shots/<id>.png` (~16 s each after the first run; first run compiles shaders). Agents look at these PNGs.
- Blockout first, art later: the art pass places visual-only meshes on top; the blockout keeps the exact collision and is hidden in game.

### 4.3 Code
- All gameplay in C++ (`Source/W4BArena/W4B/...`), derived from or copied out of the template classes. Blueprints only as thin subclasses for asset references (meshes, sounds), made once by the owner or by Python.
- Build with the editor closed: `"C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" W4BArenaEditor Win64 Development -Project="C:\W4B\W4BArena\W4BArena.uproject" -WaitMutex`.
- With the editor open: Live Coding, triggered by an agent with `npm run ue:remote -- --exec "unreal.SystemLibrary.execute_console_command(None, 'LiveCoding.Compile')"` (console command exists in 5.7: `LiveCodingModule.cpp`). Header/UPROPERTY layout changes still need a full build with the editor closed.
- IDE files: Visual Studio 2026 solution and a VS Code workspace (Antigravity is a VS Code fork): `"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.exe" -projectfiles -project="C:\W4B\W4BArena\W4BArena.uproject" -game -rocket -progress -vscode` (drop `-vscode` for the Visual Studio solution). For C++ code intelligence in Antigravity use the clangd extension (Microsoft's C++ extension is licensed for Microsoft's VS Code only) with `UnrealBuildTool.exe -mode=GenerateClangDatabase -project="C:\W4B\W4BArena\W4BArena.uproject" W4BArenaEditor Win64 Development -OutputDir="C:\W4B\W4BArena"` (writes `compile_commands.json`).

### 4.4 Art (tested import)
- Order of preference: tweak in Unreal → download CC0 → generate/model in Blender (UE_ARENA_PLAN.md §1b of the old plan stays true).
- Blender exports: glTF 2.0, **metres**, apply transforms, origin at the ground contact point. Interchange converts to centimetres; **never scale actors by 100** (the dumpster imported at 2.04 × 1.05 × 1.30 m, the Blender bot at 1.7 m tall).
- `npm run ue:import -- manifest.json`: imports headless, optional single box collision, writes `Saved/W4B/import_report.json`, and **fails if a mesh is off by more than 25 % from `expect_m` or is absurdly big/small**.
- Characters must be **skeletal meshes on the UE5 mannequin skeleton** (or retargeted with the IK Retargeter); the current Blender `bot_jax`/`zombie` are static meshes and cannot animate.
- Materials: one master material with grime/wear/rust parameters made once in the editor (owner, or a free CC0 master), then agents create and assign Material Instances with Python.

### 4.5 Tests and the Test Bench
- C++ automation tests (`IMPLEMENT_SIMPLE_AUTOMATION_TEST`, category `W4B.`) run headless:
  `UnrealEditor-Cmd.exe <uproject> -ExecCmds="Automation RunTests W4B;Quit" -unattended -nullrhi -nosplash -nosound -testexit="Automation Test Queue Empty" -ReportExportPath=C:/W4B/reports -log`
- Bot matches headless at full speed: `UnrealEditor-Cmd.exe <uproject> /Game/W4B/Maps/L_W4B_DuelPit -game -nullrhi -nosound -unattended -benchmark -fps=30 -benchmarkseconds=<game seconds> <our -W4B... options>`. `-benchmark` switches to a fixed time step and fixed seed; `-fps=30` sets the step; the run ends after `benchmarkseconds` of game time (LaunchEngineLoop.cpp, UnrealEngine.cpp). Tested: 120 s of game time took 8.6 s wall clock on an empty level.
- Unreal is not fully deterministic (threads, physics). Record the seed and settings with every result and report how far repeated runs differ, instead of promising identical results (ARENA.md §6 asked for determinism; this is the honest Unreal version).

### 4.6 Packaging
`"C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun -project="C:\W4B\W4BArena\W4BArena.uproject" -platform=Win64 -clientconfig=Development -build -cook -stage -pak -archive -archivedirectory="C:\W4B\Builds" -map=/Game/W4B/Maps/L_W4B_DuelPit+/Game/W4B/Maps/L_W4B_FFAYard -unattended -utf8output`. Smoke test: run the .exe with `-benchmark -benchmarkseconds=60` and check the log.

### 4.7 Source control
- Private UE repo: commit after each approved step. Code, config and JSON are text; assets go through LFS.
- Public repo: docs, tools, `tools/unreal/*.json`, screenshots in `docs/screens/`. Never Epic sample content beyond template-derived code, never Fab/Mixamo/Megascans files.

## 5. Settings to merge into the project
`tools/unreal/pipeline/DefaultEngine.W4B.ini` (tested): the `Projectile` collision channel, the `W4BCover` and `W4BBulletPass` profiles, `RuntimeGeneration=Dynamic` for the navmesh (headless-built levels never need "Build Paths"), and `Interchange.FeatureFlags.Import.SyncToBrowser=0`. Add for live control:
```ini
[/Script/PythonScriptPlugin.PythonScriptPluginSettings]
bRemoteExecution=True
```
Edit `.ini` files as text; scripts must never rewrite them (the old builder replaced the whole `DefaultEngine.ini`).

## 6. Verified facts and traps (tested 2026-10-20 in a throwaway 5.7.3 project)

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Import from a script crashes the commandlet: `Assertion failed: CurrentApplication.IsValid()` | `Interchange.FeatureFlags.Import.SyncToBrowser` defaults to true and overrides the import task's "don't sync"; syncing the Content Browser needs Slate (`AssetTools.cpp:3872`, `InterchangeEditorModule.cpp:25`) | set the CVar to 0 (ini or first line of the script). `import_assets.py` does it. |
| `EXCEPTION_ACCESS_VIOLATION reading address 0x40` when spawning | `spawn_actor_from_object(<mesh>)` and the old `EditorLevelLibrary` spawns need the editor viewport, which a commandlet does not have | `EditorActorSubsystem.spawn_actor_from_class(StaticMeshActor)` then `set_static_mesh` (tested) |
| `StaticMeshEditorSubsystem` is `None` headless | not created in commandlets | fall back to `unreal.EditorStaticMeshLibrary` (tested) |
| `GameplayStatics.begin_deferred_actor_spawn_from_class` missing | not exposed to Python | use `EditorActorSubsystem` |
| Python cannot pick custom collision channels | `unreal.CollisionChannel` lists only the built-in ones | named collision profiles in config (section 5) |
| Lights point up, actors face the wrong way | `unreal.Rotator(a, b, c)` is **(roll, pitch, yaw)** | always `unreal.Rotator(roll=..., pitch=..., yaw=...)` |
| Bots 170 m tall, props 100× too big | glTF is already converted to centimetres | never scale imports by 100 |
| Arenas mirrored | game y mapped to Unreal −Y | Y = +y (section 4.1) |
| Cover crates invisible and without collision | `StaticMeshActor` spawned without a mesh | set a mesh on every blockout actor |
| `-ExecutePythonScript` closes the editor right after the script returns | `EditorPythonExecuter.cpp` quits unless keep-alive is set | `unreal.EditorPythonScripting.set_keep_python_script_alive(True)` while waiting (shoot_level.py) |
| Editor exit code is a crash after a good screenshot; a start right after another Unreal process fails | shutdown crash / file contention | judge success by the `W4B_SHOT OK` marker + file, retry once (`shoot.ps1`) |
| Camera-facing walls come out dark | sun on the far side of the scene (shadows to screen lower-right) | sun behind-left of the camera (`SUN rel_yaw_deg = 10`): left wall bright, right wall dimmer, shadows fall away from the camera; see `docs/screens/ue-arena-*.jpg`. Style approval in U10. |
| An agent's "headless simulation" printed results | `tools/unreal/run_headless_match.py` printed made-up numbers with `time.sleep` | removed from the pipeline; `npm run ue:sim:headless` now says so. Real bench = U9. |
| Every editor/commandlet start rebuilt the arena | `Content/Python/init_unreal.py` runs on every start | never put build steps in `init_unreal.py` |
| Remote control could hit the wrong editor | several editors may run | `ue_remote.py --project <name>` only talks to that project (tested; refuses others) |

Live control (tested): with `bRemoteExecution=True`, `npm run ue:remote -- --exec "..."` ran Python in the open editor, `--file build_arena.py --args "--arena duel_pit"` rebuilt the level live, and `quit_editor()` closed it.

## 7. How Antigravity works on Unreal
- **Skill, not workflows**: Antigravity workflows stopped working on 2026-10-19 ([Antigravity docs](https://antigravity.google/docs/ide/workflows/)); skills live in `.agents/skills/<name>/SKILL.md` with `name`/`description` frontmatter and load when relevant, or with `/w4b-unreal` ([skills](https://antigravity.google/docs/skills)). Rules: `AGENTS.md`/`GEMINI.md` (always on).
- **Workspace**: open `world4bros-core` and add `C:\W4B\W4BArena` as a second folder, so docs, tools and the project are visible together.
- **The loop for every step**: read the step → change C++/Python/JSON/ini (text only) → build → run tests → `npm run ue:arena` / `ue:shots` or the game screenshot command → open the PNGs and check them → write what was verified (markers, test counts, screenshots) → update docs → stop for the owner.
- **Never**: edit `.uasset`/`.umap` files directly; print results that were not measured; scale imports by 100; rewrite `.ini` files from scripts; add build steps to `init_unreal.py`; commit `Saved`, `Intermediate`, `Binaries`, `DerivedDataCache`; put Fab, Mixamo, Lyra or Megascans files in the public repo.
- **Owner-only steps** (GUI or accounts): Visual Studio Installer, Epic Launcher, Fab downloads, creating the GitHub repo, playtests and look approvals. Agents write a checklist for these and wait.
- **MCP**: not needed on 5.7 (Python remote execution covers it). Third-party Unreal MCP servers exist but are unverified; 5.8's first-party plugin is experimental.

## 8. Content standards (kept from the earlier plan)
- Prefixes: `SM_` static mesh, `SK_` skeletal mesh, `M_` master material, `MI_` material instance, `T_` texture (`_BaseColor`, `_Normal`, `_ORM`), `GC_` geometry collection, `WBP_` widget, `B_` blueprint, `DA_` data asset, `L_` level, `IA_`/`IMC_` input, `NS_` Niagara, `S_`/`MS_` sound.
- Damage variants `_Clean`, `_Damaged`, `_Ruined` (Chaos fracture or rubble).
- Master material parameters: base colour, normal, packed ORM, world-aligned dirt on the bottom 0.5 m, edge wear, rust, wetness, mild blood decal receiver.
- Static meshes ≥ 1,000 triangles use Nanite; small props may keep LODs. Collision: `W4BCover` for anything that must stop bullets.

## 9. Asset sources and licences (corrected 2026-10-20)
| Source | Use | Licence notes |
| :--- | :--- | :--- |
| Kenney, Quaternius (incl. **Universal Animation Library 1 and 2**: 8-way locomotion, deaths, **zombie locomotion**, Unreal-ready GLB), Poly Haven, ambientCG | props, kits, animations, textures | **CC0**, safe everywhere ([UAL2](https://quaternius.itch.io/universal-animation-library-2)) |
| Unreal templates, Game Animation Sample (5.7, free) | code, mannequins, locomotion | Epic content: fine in the private repo and shipped builds |
| Fab (free and paid) | props, characters | private repo only; never raw files in the public repo |
| **Megascans** | surfaces, props | **no longer free since 2025** except some samples (e.g. Derelict Corridor, free indefinitely) ([80.lv](https://80.lv/articles/megascans-no-longer-free-after-2024/), [CG Channel](https://www.cgchannel.com/2025/05/get-260-free-3d-scanned-assets-for-creating-a-derelict-interior-in-ue5/)). Earlier docs said otherwise. |
| City Sample | reference only | Unreal-only and very large; skip for the demo |
| MetaHuman | later, not the demo | usable in any engine since 5.6 ([licence](https://www.metahuman.com/license)); heavy and wasted at top-down size |
| Mixamo | animations | free incl. commercial use, no redistribution of raw files: private repo only ([Adobe FAQ copy](https://community.adobe.com/t5/mixamo-discussions/mixamo-faq-licensing-royalties-ownership-eula-and-tos/m-p/13234775)) |
Log every asset in `docs/CREDITS.md`.

## 10. Risks
| Risk | Mitigation |
| :--- | :--- |
| VS 2026 newest toolset breaks Unreal headers | install 14.44 (preferred by 5.7); pin with BuildConfiguration.xml |
| Disk runs out | budget in section 2; `npm run ue:doctor` warns under 45 GB, fails under 25 GB |
| Agents claim results they did not measure | every step's "Done when" needs markers, test counts or PNGs; the owner checks them |
| Binary assets edited blind | code and text first; GUI steps are owner checklists |
| Unreal is not deterministic | record seeds and spread of repeated runs |
| Two agents in the same files | Unreal work happens in `C:\W4B` and `tools/unreal/pipeline`; the old `tools/unreal/*.py` and `tools/run-unreal.mjs` are superseded |

## 10b. Rendering and performance: DirectX 12, DLSS, Reflex and the best of Unreal 5.7 (RTX 5070)

Checked on this PC 2026-10-20: the First Person template's `DefaultEngine.ini` already sets **DirectX 12 with Shader Model 6** (`DefaultGraphicsRHI_DX12`, `PCD3D_SM6`; DX11 SM5 stays as a fallback), **Lumen** for indirect light and reflections, no baked lighting, mesh distance fields; the engine sets `r.PSOPrecache.GlobalShaders=1`; and Unreal 5.7 ships its own **NVIDIA Reflex** plugin (v1.6, `Engine/Plugins/Runtime/Nvidia/Reflex`, off by default).

### Settings for the arena
| Feature | Setting | Why |
| :--- | :--- | :--- |
| Graphics API | DirectX 12, SM6 (template default; keep) | needed for Lumen hardware ray tracing, Nanite, Virtual Shadow Maps, DLSS frame generation |
| Lighting | Lumen GI + reflections, **hardware ray tracing on** (`r.RayTracing=True` and Project Settings > Rendering > Lumen > Use Hardware Ray Tracing when available) | in 5.7 the hardware path is the fast one ([Unreal forum](https://forums.unrealengine.com/t/lumen-performance-less-relies-on-hwrt-in-5-7/2731297)); the RTX 5070 has it |
| Shadows | Virtual Shadow Maps (default) | sharp shadows under the top-down camera |
| Meshes | Nanite on static meshes ≥ 1,000 triangles | section 8 |
| Many lights | **MegaLights off** for the demo | Beta in 5.7, production-ready only in 5.8 ([MegaLights docs](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine)) |
| Anti-aliasing / upscaling | **TSR** (built in) by default; **DLSS** when the NVIDIA plugin is in (below) | TSR works on every GPU |
| Latency | **Reflex On** (engine plugin `Reflex`, enable in the `.uproject`) | lower input lag matters in a shooter |
| Exposure | fixed exposure (no eye adaptation) | readable, steady image from above |

### DLSS (NVIDIA plugin)
- Current: **DLSS 4.5 plugin** for Unreal (Super Resolution with the transformer model, DLAA, Frame Generation and **Multi Frame Generation** on RTX 50 cards, Ray Reconstruction, Reflex, NIS), built on Streamline ([NVIDIA DLSS](https://developer.nvidia.com/rtx/dlss/get-started)). **Not yet confirmed for 5.7**: the owner downloads it from NVIDIA's developer site only if its page lists Unreal 5.7; otherwise the archive has older versions.
- Install into `C:\W4B\W4BArena\Plugins\` (private repo only; NVIDIA's licence decides redistribution) and enable it. Use either its Reflex or the engine's Reflex plugin, not both.
- Settings menu: Upscaler (TSR / DLSS / DLAA), DLSS mode (Quality, Balanced, Performance), Frame Generation (Off, 2x, 3x, 4x), Reflex (Off, On, On + Boost). Frame generation adds input lag, so it only turns on together with Reflex, and **defaults to Off** in this shooter.

### Smooth first launch (no shader stutter)
- PSO precaching is on in 5.x; check the packaged build's first match for hitches (`stat unit`, Insights). If it hitches, record and ship a PSO cache (Epic "PSO caching" docs) and show a short loading screen until shaders are ready.

### Game-side performance rules
- Pool projectiles, impact effects and decals; decals fade after 30 s.
- Bot brains tick at 10 Hz, not every frame; use timers instead of `Tick` where possible.
- Camera sees one arena: short shadow and view distances, no far LODs.
- Simple collision on props; projectiles use one sphere sweep.
- Niagara effects with a per-effect budget; no translucent overdraw stacks.

### Scalability and settings
Low / Medium / High / Epic through `UGameUserSettings`, plus a first-run auto-detect (`RunHardwareBenchmark`). Low turns off Lumen (screen-space GI instead) and lowers shadows. Also: resolution, window mode, VSync, frame-rate cap (`t.MaxFPS`), upscaler, Reflex, FOV-free zoom limits.

### Targets and how to measure
| Target (1440p, this PC, 8-bot FFA Yard) | Measure with |
| :--- | :--- |
| ≥ 60 fps native at Epic, frame time ≤ 16.6 ms, GPU ≤ 12 ms | `stat unit`, `stat gpu` |
| ≥ 120 fps with DLSS Quality (+ Reflex) | same, plus `stat fps` |
| 1 % lows within 30 % of the average | Unreal Insights trace (`-trace=default,gpu`) or `csvprofile start/stop` |
| No hitch > 50 ms in the first minute of a packaged build | Insights |
Steps: U10 sets lighting, shadows, Lumen and TSR; U14 adds DLSS/Reflex, the settings menu, scalability, PSO checks and the benchmark run.

## 11. Sources
Epic: [VS setup](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-visual-studio-development-environment-for-cplusplus-projects-in-unreal-engine), [5.6 templates](https://www.unrealengine.com/news/updated-game-templates-for-unreal-engine-5-6available-now), [5.8 release notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-8-release-notes), [Lyra](https://dev.epicgames.com/documentation/unreal-engine/lyra-sample-game-in-unreal-engine), [Python scripting](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python), [orthographic camera](https://dev.epicgames.com/documentation/en-us/unreal-engine/orthographic-camera-in-unreal-engine), [Game Animation Sample 5.7](https://www.unrealengine.com/tech-blog/explore-the-updates-to-the-game-animation-sample-project-in-ue-5-7), [Fab licences](https://dev.epicgames.com/documentation/en-us/fab/licenses-and-pricing-in-fab), [UE-201004 spawn crash](https://issues.unrealengine.com/issue/UE-201004). Antigravity: [rules](https://antigravity.google/docs/rules), [skills](https://antigravity.google/docs/skills), [workflows (deprecated)](https://antigravity.google/docs/ide/workflows/), [workflows to skills](https://antigravity.google/docs/migration/workflows-to-skills/). GitHub: [LFS quotas](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage). Engine source on this PC: `LaunchEngineLoop.cpp` (`-benchmark`, `-fps=`, `-deterministic`, `-testexit=`), `UnrealEngine.cpp:2703`, `AssetTools.cpp:3872`, `InterchangeEditorModule.cpp:25`, `EditorPythonExecuter.cpp:73`, `PythonScriptRemoteExecution.cpp:264`, `LiveCodingModule.cpp:431`, `Engine/Config/Windows/Windows_SDK.json`.
