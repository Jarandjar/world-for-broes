---
name: w4b-unreal
description: Use for any World 4 Bro's Unreal Engine work (the Unreal arena, phase Z34): building or checking arena levels, Unreal C++ or Python, importing art, screenshots, tests, packaging, or a step from docs/UE_ARENA_PLAN.md.
---

# World 4 Bro's: working on the Unreal arena

Read first: `docs/UE_ARENA_PLAN.md` (the step you were given, and only that step) and the `docs/UE_PIPELINE.md` sections it names. `docs/UE_WORKFLOW.md` says what the owner and Visual Studio do around your work. Section 6 of UE_PIPELINE.md lists every known trap.

## Facts
- Unreal 5.7.3 at `C:\Program Files\Epic Games\UE_5.7`. Project `C:\W4B\W4BArena\W4BArena.uproject` (env `W4B_UE_PROJECT`), made from the First Person C++ template, Arena Shooter variant. Our C++ goes in `Source/W4BArena/W4B/`. Template code is reference: derive or copy, do not rewrite it.
- The web game is the source of truth: `npm run ue:export` writes `tools/unreal/arenas.json` and `game_data.json` (guns, bot levels, match rules). Add arenas in `src/sim/arena.ts`, never by hand in the JSON.
- Coordinates: Unreal X = (x - width/2) * 100, Y = (y - height/2) * 100. No Y flip. Combat plane (bullets, bot sight) at 90 cm.
- Collision profiles: `W4BCover` stops movement, bullets and sight; `W4BBulletPass` (barrels, trees) blocks movement only.

## Commands (all tested; look for the markers)
| Command | Marker |
| :--- | :--- |
| `npm run ue:doctor` | `W4B_DOCTOR all good` |
| `npm run ue:export` | `exported N arenas` |
| `npm run ue:arena` | `W4B_BUILD_OK`, `W4B_VERIFY <id> OK` |
| `npm run ue:shots` | `W4B_SHOT try=1 ok=True`; then open the PNGs in `<project>/Saved/W4B/shots/` |
| `npm run ue:import -- manifest.json` | `W4B_IMPORT_OK` (fails on wrong scale) |
| `npm run ue:remote -- --exec "..."` | `W4B_REMOTE ...` (runs in the open editor) |
| Build (editor closed) | `"C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" W4BArenaEditor Win64 Development -Project="C:\W4B\W4BArena\W4BArena.uproject" -WaitMutex` |
| Live Coding (editor open) | `npm run ue:remote -- --exec "unreal.SystemLibrary.execute_console_command(None, 'LiveCoding.Compile')"` |
| Tests | `UnrealEditor-Cmd.exe <uproject> -ExecCmds="Automation RunTests W4B;Quit" -unattended -nullrhi -nosplash -nosound -testexit="Automation Test Queue Empty" -ReportExportPath=C:/W4B/reports` |
| Fast headless game | `UnrealEditor-Cmd.exe <uproject> <map> -game -nullrhi -nosound -unattended -benchmark -fps=30 -benchmarkseconds=<s>` |

Run scripts headless with `tools/unreal/pipeline/ue.ps1` (exit code = Unreal's). Help: `python tools/unreal/pipeline/build_arena.py --help`.

## Every step
1. Plan from the step's prompt. 2. Change only text: C++, Python, JSON, `.ini`, docs. 3. Build. 4. Run the tests and the step's checks. 5. Make screenshots and look at them. 6. Report only what you measured: markers, test counts, PNG paths, numbers. 7. Update `docs/UE_ARENA_PLAN.md` §5, World Bible §69, `PROGRESS.md`, `ROADMAP.md`. 8. Stop and wait for the owner.

## Never
- Print results you did not measure (an earlier script printed a made-up "simulation").
- Edit `.uasset` or `.umap` files directly; spawn with `spawn_actor_from_object(mesh)` headless; use `unreal.Rotator(a, b, c)` without keywords.
- Scale imported meshes by 100; flip game y; rewrite `.ini` files from scripts; put build steps in `Content/Python/init_unreal.py`.
- Commit `Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/`; put Fab, Lyra, Mixamo or Megascans files in the public repo.
- Delete or "clean up" `C:\W4B`; move or delete files the owner did not approve.
- Do owner-only steps (Visual Studio Installer, Epic Launcher, Fab, GitHub repos, look approvals): write a checklist and wait.
