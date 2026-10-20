# W4B Unreal pipeline (tested on UE 5.7.3, 2026-10-20)

Scripts that build and check the Unreal arena from the web game's data. Full reference: `docs/UE_PIPELINE.md`; steps: `docs/UE_ARENA_PLAN.md`. Set `W4B_UE_PROJECT` to the `.uproject` first.

| Command | What it does | Script |
| :--- | :--- | :--- |
| `npm run ue:doctor` | checks Visual Studio C++, MSVC 14.44, Windows SDK, git/LFS, disk, project location and settings | `doctor.ps1` |
| `npm run ue:export` | `src/sim/arena.ts`, `src/data/items.ts`, `arenaBots.ts`, `arenaRules.ts` → `tools/unreal/arenas.json` + `game_data.json` (+ copies into `Content/W4B/Data` of the project) | `tools/export-unreal.ts` |
| `npm run ue:arena [-- id]` | builds the arena levels headless, then verifies counts, tags and collision profiles | `build_arena.py`, `verify_arena.py`, `ue.ps1` |
| `npm run ue:shots [-- id]` | PNG of each level from its preview camera → `<project>/Saved/W4B/shots/` | `shoot_level.py`, `shoot.ps1` |
| `npm run ue:import -- manifest.json` | imports glTF/FBX headless, optional box collision, fails on wrong scale | `import_assets.py` |
| `npm run ue:remote -- --exec "..."` | runs Python inside the open editor (needs `bRemoteExecution=True`); `--file x.py --args "..."` runs a file | `ue_remote.py` |
| `npm run ue:open` | opens the editor | `w4b-ue.mjs` |
| `python tools/unreal/pipeline/build_arena.py --plan` | prints the placement plan without Unreal | `build_arena.py` |

Settings the project needs: `DefaultEngine.W4B.ini` (merge into `Config/DefaultEngine.ini` once, as text).

Markers to look for in output: `W4B_BUILD_OK`, `W4B_VERIFY <id> OK`, `W4B_SHOT OK`, `W4B_IMPORT_OK`, `W4B_REMOTE`, `W4B_RUN exit=0`, `W4B_DOCTOR all good`.

Rules (each one cost a crash or a wrong result once; details in `docs/UE_PIPELINE.md` section 6):
- Spawn with `EditorActorSubsystem.spawn_actor_from_class`, never `spawn_actor_from_object(mesh)` headless.
- `unreal.Rotator(roll=, pitch=, yaw=)` with keywords.
- Imports are already in centimetres; never scale by 100. Game y maps to Unreal +Y (no flip).
- Only actors tagged `W4BGenerated` are rebuilt; hand-placed art survives.
- Scripts never rewrite `.ini` files and never live in `Content/Python/init_unreal.py`.

The older scripts in `tools/unreal/` (`build_arena.py`, `create_map.py`, `setup_arena_level.py`, `run_headless_match.py`, `ue_controller.py`, ...) and `tools/run-unreal.mjs` are superseded by this folder; `run_headless_match.py` printed made-up results.
