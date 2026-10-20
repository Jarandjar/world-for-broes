# Unreal Arena Workflow: Antigravity + Visual Studio, step by step

How you, Antigravity (AG IDE) and Visual Studio 2026 work together to build the Unreal arena demo. **What** to build is in [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md) (steps U0-U14 with paste-ready prompts); **how the tools work** is in [UE_PIPELINE.md](UE_PIPELINE.md); the detailed Visual Studio process (install, debugging recipes, tests, Copilot prompts, Copilot instructions template) and the "what else do I need" list are in [VS_UNREAL_PROCESS.md](VS_UNREAL_PROCESS.md). Written 2026-10-20.

## 1. Who does what
| Who / tool | Job | Never |
| :--- | :--- | :--- |
| **You** | Install things, click in launchers and stores, play-test, approve looks, decide, commit | Approve a step without seeing its screenshots or playing it |
| **Antigravity** | Writes and changes the code (C++, Python, JSON, `.ini`), runs builds, tests and `npm run ue:*` commands, looks at screenshots, updates the docs | Edit `.uasset`/`.umap` files, claim results it did not measure, push to GitHub, delete files you did not approve |
| **Visual Studio 2026 + Copilot** | Debugging (breakpoints, call stacks, watching values), reading and reviewing code, Copilot Chat questions, the Error List after a failed build | Large edits at the same time as Antigravity (one editor writes code at a time) |
| **Unreal Editor** | Play in Editor, look at levels, the few GUI tasks (materials in U10), Live Coding | Running while a full build with `Build.bat` or Visual Studio runs |
| **Epic Games Launcher** | Starting Unreal, creating the project (U1), Fab downloads | |
| **GitHub Desktop or git** | Commits: the private repo `C:\W4B\W4BArena` and this repo | Committing `Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/` |

## 2. One-time setup

### 2.1 Antigravity (after U1)
1. Open the folder `world4bros-core`, then **File > Add Folder to Workspace** → `C:\W4B\W4BArena`. Save the workspace (for example `C:\W4B\w4b.code-workspace`) and always open that file.
2. The skill `.agents/skills/w4b-unreal/SKILL.md` loads by itself on Unreal work; you can also start a prompt with `/w4b-unreal`.
3. Agent settings > terminal: **Auto** mode. Allow: `npm`, `node`, `python`, `powershell`, `git status`, `git diff`, `git add`, `git commit`, `Build.bat`, `UnrealEditor-Cmd.exe`, `UnrealBuildTool.exe`. Deny: `git push`, `Remove-Item -Recurse`, `rm -rf`, `del /s`, anything touching `C:\W4B` outside the project.
4. Install the **clangd** extension (C++ code help). After U1, Antigravity writes `compile_commands.json` (UE_PIPELINE.md §4.3).
5. Restart Antigravity after `setx W4B_UE_PROJECT ...` so it sees the variable.

### 2.2 Visual Studio 2026 (after U1)
1. In File Explorer, right-click `C:\W4B\W4BArena\W4BArena.uproject` > **Generate Visual Studio project files**, or let Antigravity run the command in UE_PIPELINE.md §4.3 without `-vscode`.
2. Open `C:\W4B\W4BArena\W4BArena.sln`. Toolbar: **Development Editor**, **Win64**; right-click **W4BArena** > **Set as Startup Project**.
3. **F5** builds and starts the Unreal Editor with the debugger attached; **Ctrl+F5** without it. Breakpoints in our C++ work when you then press Play in the editor.
4. Already running editor: **Debug > Attach to Process > UnrealEditor.exe**.
5. **Copilot Chat**: ask it about the code (for example "why does AW4BWeapon not hit through W4BBulletPass?") and paste its answer to Antigravity, or let it review Antigravity's changes (**Git Changes** window).
6. When Visual Studio says files changed outside it, click **Yes to all** (Antigravity changed them).

### 2.3 Unreal Editor
- **Live Coding** is on by default (Ctrl+Alt+F11 compiles C++ while the editor runs). Antigravity can trigger it with `npm run ue:remote` (UE_PIPELINE.md §4.3).
- Remote execution (`bRemoteExecution=True`, set in U1) lets Antigravity run Python in the open editor.
- Play: the green **Play** button (Play in Editor). Stop with Esc.

## 3. The loop for every step
1. **You**: open `docs/UE_ARENA_PLAN.md`, find the next unticked step in §5, read its "Owner" part and do it first (if it has one).
2. **You**: copy the step's **Antigravity prompt** into a new Antigravity chat.
3. **Antigravity**: plans, changes text files, builds, runs tests and screenshots, and ends with a report: what changed, the markers (`W4B_VERIFY ... OK`, test counts), screenshot paths, open problems. It stops.
4. **You**: look at the screenshots it lists (`C:\W4B\W4BArena\Saved\W4B\shots\`), then open the editor and **Play** for 5-10 minutes in the Duel Pit.
5. **If something is wrong**:
   - Build error: open the **Error List** in Visual Studio (or the build log Antigravity names) and paste the first error to Antigravity.
   - Crash: Antigravity reads `C:\W4B\W4BArena\Saved\Crashes\...` and `Saved\Logs\W4BArena.log`; you can reproduce with **F5** in Visual Studio to stop on the crash line.
   - Wrong behaviour: set a breakpoint in Visual Studio (F5, Play, trigger it), or describe what you saw to Antigravity with a screenshot.
   - Then: "Fix: <what you saw>. Same step, same rules." to Antigravity, back to point 3.
6. **You approve**: tell Antigravity "Approved, mark U<n> done and update the docs".
7. **You commit**: GitHub Desktop → repo `world4bros-ue` (`C:\W4B\W4BArena`): message `U<n>: <title>`, Commit, Push. Then this repo (`world-for-broes`) for docs, tools and JSON.

Rules for building: one builder at a time. Editor **closed** → Antigravity's `Build.bat` or Visual Studio's build. Editor **open** → Live Coding only. Changing headers (`.h` with `UPROPERTY`/`UFUNCTION` or class layout) → close the editor and do a full build.

## 4. Step by step
| Step | You | Antigravity | Visual Studio | Done when |
| :--- | :--- | :--- | :--- | :--- |
| **U0** Prepare PC | Run the Visual Studio command (UE_ARENA_PLAN.md U0), git on PATH, `C:\W4B` exists, ≥ 45 GB free | `npm run ue:doctor`, report | Installer only | doctor fails only on `W4B_UE_PROJECT` |
| **U1** Project | Launcher: First Person, C++, Arena Shooter, `C:\W4B`, `W4BArena`; play `Lvl_Shooter`; `setx W4B_UE_PROJECT ...`; create private repo `world4bros-ue`; set up Antigravity (§2.1) | Merge settings, `.gitignore`/`.gitattributes`, LFS, build, project files, `compile_commands.json`, doctor, remote check | Generate project files, open `.sln`, F5 once (§2.2) | build OK, doctor all PASS, remote prints `5.7.3` |
| **U2** Arenas | Look at the PNGs; approve retiring the old `tools/unreal/World4Bros` | `ue:export`, `ue:arena`, `ue:shots`, startup map, list old files | – | `W4B_VERIFY` OK for every arena |
| **U3** Player | Play: walk, aim, zoom; judge camera angle | C++ character, camera, input in code, game mode, game screenshots | Breakpoint in the aim code if aiming feels off | you approve the controls |
| **U4** Guns | Play: shoot crates (stop) and trees (pass) | Data subsystem, weapons, projectiles, tests | Debug a weapon that does not fire; Copilot review of the falloff formula | tests green, guns feel different |
| **U5** Health, pies | Play: kill a dummy, take a pie | Health component, pies, respawn, tests | – | tests green |
| **U6** Bots | Play against each level; check bots never shoot through crates | C++ bot brain, perception, four levels from data, screenshots | Gameplay debugger in the editor (key `'` while playing) and breakpoints in the brain | levels feel different |
| **U7** Rules | Play a full duel and FFA | Game mode/state, rules, results JSON, tests | – | tests green, results files written |
| **U8** UI | Click through title → match → results with mouse and gamepad | UMG widgets in C++, screenshots | – | you approve the screens |
| **U9** Test Bench | Read the numbers | Bench subsystem, `npm run ue:bench`, experiments, results in `docs/bench/` | Profile a slow bench run if needed | Sharp ≥ 70 % vs Scarecrow over 50 duels |
| **U10** Look | Build `M_W4B_Master` and the outline material from Antigravity's recipe (editor GUI, ~1 h); approve the look | Recipe first, then instances, lighting, effects, screenshots | – | you approve the look |
| **U11** Art kit | Download CC0 packs into `C:\W4B\Downloads` | Import manifest, `dress_arena.py`, Blender lane for missing models, credits | – | Duel Pit dressed, verify OK |
| **U12** Characters | Approve survivors (and decide zombies) | Materials, accessories, optional zombie retarget | – | you approve |
| **U13** Sound | Listen | CC0 sounds, attenuation, volume settings | – | you approve |
| **U14** Release | Final playthrough on the packaged .exe; decide the next phase | Profiling, scalability, package, soak test, content check, zip, release notes | Profile with Unreal Insights if fps is low | every point of UE_ARENA_PLAN.md §1 ticked |

## 5. Daily routine (about 1-2 hours a session)
1. Start: open the Antigravity workspace; in Visual Studio open `W4BArena.sln` only if you plan to debug.
2. One step (or one fix round) per session; do not start the next step before you approved and committed the last one.
3. End: commit both repos; if a step is half done, ask Antigravity to write where it stopped into UE_ARENA_PLAN.md §5.

## 6. If you get stuck
| Problem | What to do |
| :--- | :--- |
| `npm run ue:doctor` shows FAIL | Do the "fix" line it prints |
| Visual Studio command does nothing | Use the PowerShell version in UE_ARENA_PLAN.md U0 (the first version did not run in PowerShell) |
| "The following modules are missing or built with a different engine version" when opening the project | Click **Yes** to rebuild, or close the editor and let Antigravity run `Build.bat` |
| Build fails with MSVC errors in engine headers | Check that MSVC 14.44 is installed (doctor); pin it with `BuildConfiguration.xml` (UE_PIPELINE.md §2) |
| Antigravity says something works but you cannot see it | Ask for the exact command it ran and its output, plus the screenshot path |
| Two tools changed the same file | Keep Antigravity's version unless you changed it on purpose; let Copilot or Antigravity explain the diff |
