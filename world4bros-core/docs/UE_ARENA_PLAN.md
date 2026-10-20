# Unreal Arena: Step-by-Step to a Finished Demo (phase Z34)

Owner decision (2026-10-20): build the Aim Arena in Unreal Engine and play against bots there. This is the build plan for Antigravity (AG IDE), one step at a time with paste-ready prompts. **Rewritten 2026-10-20 after a pipeline review** (read-up, engine and template source, and hands-on tests on this PC). The reference behind every step is [UE_PIPELINE.md](UE_PIPELINE.md); who does what in Antigravity, Visual Studio and the editor at each step is [UE_WORKFLOW.md](UE_WORKFLOW.md); the Visual Studio details and the full "what else we need" list are in [VS_UNREAL_PROCESS.md](VS_UNREAL_PROCESS.md). Design and rules: [ARENA.md](ARENA.md). Style: [STYLE_BIBLE.md](STYLE_BIBLE.md) §0.

## 0. What changed in this version
- **Base is Unreal's First Person C++ template, Arena Shooter variant, plus the Twin Stick camera code. Not Lyra.** **Approved by the owner on 2026-10-20.** Reasons in UE_PIPELINE.md §1.2 (public repo and Epic/Fab licences, disk, Git LFS quota, agent-friendly C++ instead of binary data assets, top-down aim).
- **The project moves to `C:\W4B\W4BArena`** (outside OneDrive) in a **private** GitHub repo. The old `tools/unreal/World4Bros` project (blank, Blueprint-only, placeholder static meshes, crashing scripts) is retired in U2.
- **The pipeline is built and tested** (`tools/unreal/pipeline/`, `npm run ue:*`): data export, headless level build + verify, screenshots, asset import with size checks, live editor control, fast headless game runs.
- Corrected facts: no Y flip between game and Unreal coordinates; imports are already in centimetres; Megascans are no longer free; Antigravity workflows are replaced by skills.
- Old step numbers U0-U20 are replaced; mapping in §6.

## 1. The finished demo (definition of done)
A zipped Windows build that runs from the desktop on this PC:
1. **Flow**: title → pick a match (Duel or Free-for-all, arena, bot level per bot, round target or kill/time limit, loadout) → play → results (winner, score, kills, accuracy, time to kill) → play again or back.
2. **Arenas**: Duel Pit and FFA Yard built from `arenas.json`, same layout as the web game (checked by `verify_arena.py`).
3. **Controls and camera**: top-down at the game angle (yaw -45°, pitch -55°), mouse-wheel zoom from close to the whole arena, WASD relative to the screen, aim at the cursor, gamepad twin-stick.
4. **Guns**: at least hunting rifle, pump shotgun and nailgun SMG with the web game's numbers (`game_data.json`); potato cannon and slingshot if time allows.
5. **Bots**: Scarecrow, Regular, Sharp, Menace with measurably different reaction, aim error and fire rate; they see only what is in line of sight and hear shots; **a Sharp bot beats a Scarecrow in at least 70 % of 50 headless duels**.
6. **Rules**: ARENA.md §4: duel rounds (first to 3/5/7/10, 2 s pause, 30 s draw, time limit); FFA (kill or time limit, 3 s respawn at the safest spawn ≥ 12 m from enemies, 2 s spawn protection); pies heal 25 and come back after 20 s.
7. **HUD**: health, ammo, round score or scoreboard, timer, kill feed; no enemy health bars.
8. **Look**: gritty and readable from above: master grime material on all surfaces, CC0 art kit on the Duel Pit at least, lit camera-facing walls, outlined characters, muzzle flashes, impacts, mild blood decals.
9. **Performance**: DirectX 12, Lumen with hardware ray tracing, TSR and DLSS (if the NVIDIA plugin supports 5.7), Reflex; 60 fps native and 120+ with DLSS at 1440p in an 8-bot FFA on this PC (UE_PIPELINE.md section 10b).
10. **Quality**: automation tests green; a 30-minute headless bot soak run without a crash; all third-party assets in `docs/CREDITS.md`.

Out of scope for the demo (later phases): online play, Chaos destruction, PCG clutter, mood presets, MetaHumans, the open world.

## 2. Milestones and rough effort
| Milestone | Steps | Result | Effort (agent hours, rough) |
| :--- | :--- | :--- | :--- |
| M0 Ready | U0-U2 | PC ready, project created, arenas built and checked | owner 2-3 h, agent 2-3 h |
| M1 Greybox shooter | U3-U5 | walk, aim, shoot, take damage, heal on pies in the Duel Pit | 15-25 h |
| M2 Bot fights | U6-U7 | four bot levels, duel and FFA rules | 15-25 h |
| M3 Game flow | U8-U9 | menus, HUD, results, headless Test Bench | 15-20 h |
| M4 Look and feel | U10-U12 | style pass, art kit, survivors (and zombies, optional) | 20-35 h |
| M5 Demo | U13-U14 | sound, performance, packaged build, release | 8-14 h |

Stop for the owner after every step. After every step update: this file's status list (§5), World Bible §69 (Z34 row and "Where to resume"), `PROGRESS.md`, `ROADMAP.md`.

## 3. The daily loop
1. **Owner** opens Antigravity on `world4bros-core` with `C:\W4B\W4BArena` added as a second folder, and pastes the next step's prompt.
2. **Antigravity** loads the `w4b-unreal` skill, reads the step and the UE_PIPELINE.md sections it names, changes C++/Python/JSON/ini only, builds, runs tests, makes screenshots and looks at them, then reports **only what it measured** (markers such as `W4B_VERIFY ... OK`, test counts, PNG paths) and updates the docs.
3. **Owner** plays it in the editor for 5-10 minutes (Play in the Duel Pit), approves or writes notes, and commits: the private repo (`C:\W4B\W4BArena`) and this repo (docs, tools, JSON).
4. **Visual Studio 2026 + Copilot** is for debugging: open `W4BArena.sln`, set breakpoints, attach to `UnrealEditor.exe`.

## 4. The steps

### U0 — Prepare this PC (owner, about 1-2 hours mostly downloads)
- **Goal**: everything `npm run ue:doctor` checks is green except the project, which comes in U1.
- **Owner**:
  1. Visual Studio C++ for Unreal: close Visual Studio, then in any **PowerShell** window (it asks for admin rights; answer Yes; 10-30 min, ~10-15 GB). Or tick the same items in Visual Studio Installer > Modify. (The first version of this command started with a quoted path, which PowerShell rejects with "Unexpected token 'modify'"; nothing was installed.)
     ```powershell
     Start-Process "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" -Verb RunAs -Wait -ArgumentList 'modify --installPath "C:\Program Files\Microsoft Visual Studio\18\Professional" --add Microsoft.VisualStudio.Workload.NativeGame --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Component.Unreal.Ide --add Component.Unreal.Debugger --includeRecommended --passive --norestart'
     ```
  2. Add `C:\Program Files\Git\cmd` to PATH (Settings > System > About > Advanced system settings > Environment Variables), then `git lfs install` in a new terminal.
  3. Free disk to at least 45 GB on C:.
  4. Create the folder `C:\W4B`. It is part of the project: never clean it up.
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U0 of docs/UE_ARENA_PLAN.md only: run `npm run ue:doctor` and report every
  line. Do not install anything yourself. If something besides W4B_UE_PROJECT fails, give me the exact fix from
  docs/UE_PIPELINE.md section 2 and stop.
  ```
- **Done when**: the doctor shows PASS for Unreal, the "Game development with C++" workload, MSVC 14.44, Windows SDK, git, git lfs and disk; only `W4B_UE_PROJECT` fails.

### U1 — Create the project (owner 10 min, then Antigravity)
- **Owner**:
  1. Epic Games Launcher > Unreal Engine 5.7 > Launch > **Games > First Person**, **C++**, **Variant: Arena Shooter**, Starter Content off, location `C:\W4B`, name **`W4BArena`** > Create. Wait for the first compile; play `Variant_Shooter/Lvl_Shooter` once (bots should fight); close the editor.
  2. `setx W4B_UE_PROJECT C:\W4B\W4BArena\W4BArena.uproject` and open a new terminal.
  3. Create an empty **private** GitHub repo `world4bros-ue` (GitHub Desktop or the website).
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U1 of docs/UE_ARENA_PLAN.md only. The project is C:\W4B\W4BArena (from the
  First Person C++ template, Arena Shooter variant).
  1. Merge tools/unreal/pipeline/DefaultEngine.W4B.ini into C:\W4B\W4BArena\Config\DefaultEngine.ini as text
     (keep one copy of the Projectile channel line) and add the bRemoteExecution block from docs/UE_PIPELINE.md section 5.
  2. Write .gitignore and .gitattributes exactly as docs/UE_PIPELINE.md section 3 says; git init; git lfs install;
     first commit; add the remote for the private repo world4bros-ue (ask me for the URL); do not push without asking.
  3. Build: Build.bat W4BArenaEditor Win64 Development (command in docs/UE_PIPELINE.md section 4.3). Generate the
     VS Code workspace and compile_commands.json (same section). Report the build result lines.
  4. Run `npm run ue:doctor` (expect all PASS) and, with the editor open,
     `npm run ue:remote -- --exec "print(unreal.SystemLibrary.get_engine_version())"`.
  Do not change any template code. Update the docs (status list, World Bible section 69, PROGRESS.md).
  ```
- **Done when**: build succeeded from the command line, doctor all PASS, remote call printed `5.7.3`, first commit made.

### U2 — Arenas and pipeline in the project (Antigravity, about 1 hour)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U2 of docs/UE_ARENA_PLAN.md only.
  1. `npm run ue:export` (copies arenas.json and game_data.json into Content/W4B/Data), then `npm run ue:arena`
     and `npm run ue:shots`. Open every PNG in C:\W4B\W4BArena\Saved\W4B\shots and describe what you see.
  2. Set EditorStartupMap and GameDefaultMap to /Game/W4B/Maps/L_W4B_DuelPit in Config/DefaultEngine.ini (text edit).
  3. Retire the old project: list what is in world4bros-core/tools/unreal/World4Bros and the superseded scripts
     (tools/unreal/*.py outside pipeline/, tools/run-unreal.mjs) and ask me before moving them to
     tools/unreal/_retired/. Never run its Content/Python/init_unreal.py again.
  4. square_arena exists only in arenas.json: ask me whether to add it to src/sim/arena.ts.
  Report the W4B_BUILD_OK / W4B_VERIFY lines. Update the docs.
  ```
- **Done when**: `W4B_VERIFY ... OK` for every arena, the owner has seen the PNGs (like `docs/screens/ue-arena-duel-pit-blockout.jpg`), the editor opens on the Duel Pit.

### U3 — Top-down player (Antigravity C++, owner playtest)
- **Goal**: the player walks, aims and looks right at the game angle. No shooting yet.
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U3 of docs/UE_ARENA_PLAN.md only. Work in C:\W4B\W4BArena\Source\W4BArena\W4B\.
  Reference code (read, do not modify): the project's Variant_Shooter folder, and the Twin Stick variant at
  C:\Program Files\Epic Games\UE_5.7\Templates\TP_TopDown\Source\TP_TopDown\Variant_TwinStick\TwinStickCharacter.cpp.
  1. AW4BCharacter (ACharacter, shared base for player and bots): third-person mannequin from the template with the
     rifle animation blueprint (ABP_TP_Rifle) so it can strafe; faces its aim direction; collision capsule 34 x 88 cm.
  2. AW4BPlayerCharacter: spring-arm camera, yaw -45, pitch -55, arm 2200 cm, wheel zoom 1200-4500 cm, camera lag;
     WASD relative to the camera yaw; mouse aim = cursor ray intersected with the plane Z = 90 cm (the combat
     plane, not GetHitResultUnderCursor, so tall cover does not steal the aim); gamepad right stick aims.
  3. AW4BPlayerController: shows the cursor; input actions and the mapping context are created in C++ at startup
     (NewObject<UInputAction>, UInputMappingContext::MapKey, swizzle/negate modifiers for WASD) so no input assets
     are needed.
  4. AW4BArenaGameMode spawns the player at a W4B.Spawn PlayerStart. Set it as the default game mode in config.
  5. Add a development screenshot hook: UW4BAutoShotSubsystem (GameInstanceSubsystem) reads -W4BShotAt=2,4 and
     -W4BQuitAt=5 from the command line, runs "HighResShot 1600x900" at those game times and quits. Add
     `npm run ue:gameshot` to tools/unreal/pipeline/w4b-ue.mjs that runs the game from the editor binaries:
     UnrealEditor.exe <uproject> /Game/W4B/Maps/L_W4B_DuelPit -game -RenderOffScreen -windowed -ResX=1600 -ResY=900
     -W4BShotAt=2,4 -W4BQuitAt=5, then copies the PNGs to Saved/W4B/shots/game_*.png.
  Build with Build.bat (editor closed). Look at the game screenshots yourself. Update the docs.
  ```
- **Done when**: the build passes; game screenshots show the mannequin at the Duel Pit spawn from the game angle; the owner walks around in Play mode and approves the camera and controls.

### U4 — Guns from the web data (Antigravity C++)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U4 of docs/UE_ARENA_PLAN.md only.
  1. UW4BDataSubsystem (GameInstanceSubsystem): load Content/W4B/Data/game_data.json with FJsonObjectConverter into
     USTRUCTs FW4BGunDef, FW4BBotLevel, FW4BMatchRules (fields exactly as in the JSON; metres -> centimetres in
     code). Console command w4b.ReloadData. Add +DirectoriesToAlwaysStageAsNonUFS=(Path="W4B/Data") to DefaultGame.ini.
  2. AW4BWeapon from ShooterWeapon (keep its structure): fire rate, spreadAngle cone (radians), pelletCount,
     projectileSpeed, maxRange (projectile life), damage falloff baseDamage -> minDamage after falloffStartDist using
     the same formula as the web game (search falloffStartDist in src/sim), arced shots use gravity, noiseRadius ->
     MakeNoise for AI hearing. Shots leave the muzzle level in the combat plane (90 cm), aimed at the cursor point.
  3. Projectiles use the Projectile collision channel, so trees and barrels (W4BBulletPass) let them through and
     cover (W4BCover) stops them.
  4. The player gets nailgun_smg, hunting_rifle and pump_shotgun (keys 1-3); placeholder meshes from the template.
  5. Automation tests W4B.Combat.*: falloff values equal the web formula at 0, falloffStartDist, halfway and
     maxRange for every gun; shots per second match fireRate; a projectile passes a W4BBulletPass cylinder and stops
     at a W4BCover box. Run them headless (command in docs/UE_PIPELINE.md section 4.5) and report the counts.
  Update the docs.
  ```
- **Done when**: tests green; in Play mode bullets stop on crates and pass trees; the three guns feel different.

### U5 — Health, death, pies (Antigravity C++)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U5 of docs/UE_ARENA_PLAN.md only.
  1. UW4BHealthComponent on AW4BCharacter: max HP (player 100; bots from botLevels.hp), damage via
     ApplyPointDamage, death event, ragdoll on death like ShooterNPC::Die, a small blood decal at the hit point
     (placeholder material, mild).
  2. AW4BPiePickup: the game mode spawns one at every TargetPoint tagged W4B.Pickup.Pie; overlap heals
     matchRules.pickups.pieHeal and hides it for pieRespawn seconds.
  3. Respawn after death at a PlayerStart (rules come in U7).
  4. Tests W4B.Health.*: damage, death, heal clamp, pie respawn timing (use a test world).
  Update the docs.
  ```
- **Done when**: tests green; in Play mode a dummy bot (static target) dies, a pie heals and comes back.

### U6 — Bots with four levels (Antigravity C++)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U6 of docs/UE_ARENA_PLAN.md only.
  Write the bot brain in C++ (do not edit StateTree or Behavior Tree assets; use the template's
  ShooterStateTreeUtility.cpp and ShooterAIController as reference code).
  1. AW4BBotController (AIController) + UW4BBotBrainComponent ticking at 10 Hz with states Idle, Engage,
     Reposition, TakeCover, SeekPie. AI Perception: sight radius botLevels.sight*100, eyes at the combat plane
     (90 cm), hearing from weapon noise. A bot never shoots without line of sight (Visibility trace at 90 cm).
  2. Level numbers from game_data.json botLevels: reaction (delay before the first shot after acquiring a target),
     aimSpread (random yaw error within +-aimSpread radians), shotInterval, speed, hp. Behaviour per ARENA.md
     section 5: Scarecrow walks straight, no cover, no pies; Regular strafes and takes cover when hurt; Sharp leads
     shots (projectile speed) and peeks from cover, seeks pies under 40 % HP; Menace also flanks.
  3. Bots use AW4BCharacter and the same guns and HP rules as the player. Navigation through the dynamic navmesh.
  4. Command-line options for quick tests: -W4BBots=sharp,scarecrow -W4BGun=nailgun_smg. Make game screenshots
     of a bot fight with npm run ue:gameshot and look at them.
  Update the docs.
  ```
- **Done when**: in Play mode each level behaves visibly differently; bots never fire through crates; screenshots show bots in cover.

### U7 — Match rules (Antigravity C++)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U7 of docs/UE_ARENA_PLAN.md only.
  1. AW4BArenaGameMode + AW4BArenaGameState with duel and free-for-all rules from game_data.json matchRules
     (ARENA.md section 4): duel rounds with pause, reset to spawns, 30 s draw, round target and time limit; FFA kill
     or time limit, 3 s respawn at the spawn farthest from enemies and never within 12 m, 2 s spawn protection that
     ends when the player fires.
  2. Match setup from the command line (-W4BMode=duel|ffa -W4BArena=duel_pit -W4BBots=... -W4BTarget=5
     -W4BSeed=1) and from a struct the menu will fill in U8.
  3. FW4BMatchResult (winner, scores, per fighter shots, hits, kills, deaths, time to kill, who fired first per round)
     written as JSON to Saved/W4B/results/.
  4. Tests W4B.Rules.*: a draw round resets, the kill limit and the time limit end the match, spawn protection
     blocks damage and ends on firing, the safe-spawn rule.
  Update the docs.
  ```
- **Done when**: tests green; a full duel and a full FFA play to a result file.

### U8 — Menus, HUD, results (Antigravity C++, owner review)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U8 of docs/UE_ARENA_PLAN.md only. Build the UI as UMG widgets constructed in C++
  (UUserWidget subclasses that build their WidgetTree in code), so no widget assets need editing.
  Screens from the Aim Arena design canvas linked in docs/ARENA.md: Title, Pick a match (mode, arena, bot levels,
  bot count, round target or limits, loadout), HUD (health, ammo, round score or scoreboard, timer, kill feed; no enemy
  health bars), Results (winner, score, kills, accuracy, time to kill; Play again, Back), Pause. Mouse and gamepad
  focus navigation. Palette from STYLE_BIBLE.md section 0.3. Make game screenshots of every screen and look at them.
  Update the docs.
  ```
- **Done when**: the owner can go title → match → results → play again without the editor; screenshots approved.

### U9 — Test Bench, headless (Antigravity C++)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U9 of docs/UE_ARENA_PLAN.md only.
  1. UW4BBenchSubsystem, active with -W4BBench: plays -W4BMatches=N matches back to back with the U7 setup options,
     swaps sides every match, seeds from -W4BSeed, writes all FW4BMatchResult rows plus settings to
     -W4BOut=<file>.json and a CSV next to it, then exits.
  2. Add `npm run ue:bench` to tools/unreal/pipeline/w4b-ue.mjs: UnrealEditor-Cmd.exe <uproject>
     /Game/W4B/Maps/L_W4B_DuelPit -game -nullrhi -nosound -unattended -benchmark -fps=30 -W4BBench ... and print a
     summary (win rates, average round length, time to kill, wall-clock speed as a multiple of real time).
  3. Experiments: bot ladder (each level against the next, 50 duels) and gun matchups (every core gun against every
     other, 25 duels each). Run the same seed three times and report how far the results differ.
  4. Save results under docs/bench/ with the date.
  Report the numbers you measured only. Update the docs.
  ```
- **Done when**: Sharp beats Scarecrow in ≥ 70 % of 50 duels; the ladder goes up with each level; results committed.

### U10 — Look pass (owner + Antigravity)
- **Owner** (editor, about 1 hour, a written recipe from Antigravity first): the master material `M_W4B_Master` (grime, edge wear, rust, wetness, blood receiver) and a post-process outline material using Custom Depth on characters.
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U10 of docs/UE_ARENA_PLAN.md only.
  1. Before the owner builds anything, write a click-by-click recipe for M_W4B_Master and the outline material in
     docs/UE_PIPELINE.md (new section 12) and stop until the owner says they exist. Apply the rendering settings in UE_PIPELINE.md section 10b (DirectX 12, Lumen hardware ray tracing, Virtual Shadow Maps, TSR, fixed exposure).
  2. With Python: material instances for floor, walls, crates, sandbags, barrels, trees with STYLE_BIBLE.md palette
     values; assign them in build_arena.py through an optional --materials JSON (keep the blockout working without it).
  3. Lighting: keep the sun relative to the camera (SUN in build_arena.py), fixed exposure, a little fog, colour
     grading toward the palette; characters write Custom Depth for the outline.
  4. Effects: muzzle flash, bullet impacts per surface, mild blood decals that fade after 30 s (pooled).
  5. Shots of both arenas and of a fight (ue:shots, ue:gameshot). Look at them and compare with STYLE_BIBLE.md
     section 0. Update the docs.
  ```
- **Done when**: the owner approves the look from the game camera.

### U11 — Art kit (owner downloads, Antigravity places)
- **Owner**: download CC0 packs (Kenney City Kit Suburban/Industrial/Roads, Kenney Survival Kit, a Quaternius modular pack; check each page says CC0) into `C:\W4B\Downloads\`.
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U11 of docs/UE_ARENA_PLAN.md only.
  1. Write an import manifest with expect_m sizes and run `npm run ue:import -- <manifest>` (fails on wrong scale).
  2. Write tools/unreal/pipeline/dress_arena.py: reads an art-kit JSON (kind -> meshes, random variants from the
     seed, scale rules) and places visual-only meshes (NoCollision, tagged W4BGenerated and W4B.Art) over the
     blockout; blockout actors become hidden in game but keep their collision. Rebuild + verify + shots.
  3. New models that cannot be downloaded (potato cannon, nailgun SMG): Blender lane (glTF, metres) via
     tools/blender, then the same import.
  4. Every asset into docs/CREDITS.md with source and licence.
  Update the docs.
  ```
- **Done when**: the Duel Pit (at least) is dressed, verify still passes, the owner approves the shots.

### U12 — Characters (and optional zombies)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U12 of docs/UE_ARENA_PLAN.md only.
  1. Survivors: keep the template mannequin skeleton; gritty material instances (palette, grime), team colours,
     simple CC0 accessories (hats, backpacks) attached to sockets. Readable from the game camera with the outline.
  2. Optional zombie mode (ask me first): Quaternius Universal Animation Library 2 zombie animations (CC0, GLB)
     retargeted to the mannequin with the IK Retargeter; chasers based on the Twin Stick NPC code.
  3. Shots, credits, docs.
  ```
- **Done when**: owner approves the survivors (and zombies if chosen).

### U13 — Sound (Antigravity)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U13 of docs/UE_ARENA_PLAN.md only. CC0 sounds only (for example Kenney audio
  packs): a shot per gun, impacts per surface, footsteps, pie, UI clicks, round start/end stingers; attenuation for
  3D sounds; master and effects volume in the pause menu. Credits and docs.
  ```

### U14 — Performance, package, release (Antigravity + owner)
- **Antigravity prompt**:
  ```
  Use the w4b-unreal skill. Do step U14 of docs/UE_ARENA_PLAN.md only.
  1. Measure an 8-bot FFA at 1440p (stat unit, stat gpu, an Unreal Insights trace with -trace=default,gpu). Enable the engine Reflex plugin; if the owner installed the NVIDIA DLSS plugin for 5.7, add DLSS/DLAA/Frame Generation to the settings menu (Frame Generation off by default, only with Reflex). Targets and rules: docs/UE_PIPELINE.md section 10b.
     Fix the worst costs until 60 fps holds. Scalability Low/Medium/High in the pause menu.
  2. Package with RunUAT BuildCookRun (command in docs/UE_PIPELINE.md section 4.6) into C:\W4B\Builds, both
     Development and Shipping. Smoke test the .exe; run a 30-minute headless bench soak.
  3. Content check script: naming prefixes, missing collision on W4BCover actors, assets missing from CREDITS.md.
  4. Zip C:\W4B\Builds\W4BArena-demo-<date>.zip with a README (controls, credits). Write release notes in
     docs/UE_ARENA_PLAN.md section 5.
  Update all docs.
  ```
- **Done when**: every point in §1 is ticked by the owner. Then the **decision gate**: move the whole game to Unreal or keep the web game? Online 1v1? Upgrade to 5.8 (first-party MCP plugin)?

## 5. Status
- [x] Pipeline built and tested on 2026-10-20 (`tools/unreal/pipeline/`, UE_PIPELINE.md §6); `npm run ue:export` writes arenas.json + game_data.json; bot levels moved to `src/data/arenaBots.ts`, match rules in `src/data/arenaRules.ts`
- [ ] U0 Prepare this PC (2026-10-20 doctor: git + Git LFS on PATH OK, `C:\W4B` created; **Visual Studio C++ not installed yet**: the first command did not run in PowerShell; the fixed one left no installer log on the second try either (UAC prompt not accepted, or run outside PowerShell), so use Visual Studio Installer > Modify by hand)
- [ ] U1 Create the project
- [ ] U2 Arenas and pipeline in the project
- [ ] U3 Top-down player
- [ ] U4 Guns
- [ ] U5 Health, death, pies
- [ ] U6 Bots with four levels
- [ ] U7 Match rules
- [ ] U8 Menus, HUD, results
- [ ] U9 Test Bench
- [ ] U10 Look pass
- [ ] U11 Art kit
- [ ] U12 Characters
- [ ] U13 Sound
- [ ] U14 Performance, package, release

## 6. Old step numbers (U0-U20 of the earlier version)
| Old | Now |
| :--- | :--- |
| U0 Lyra setup, U1 Lyra baseline, U2 Game Feature plugin | dropped (template base); setup is U0-U1 |
| U3 headless arena builder | done as the pipeline; used in U2 |
| U4 top-down camera | U3 |
| U5 bot tiers | U6 |
| U6 match loop | U7 |
| U7 CommonUI HUD | U8 (UMG in C++) |
| U8 guns | U4 |
| U9 master grime material | U10 |
| U10 modular kit | U11 |
| U11 survivors, U12 zombies | U12 |
| U13 Chaos fracture, U15 PCG clutter, U16 mood presets | after the demo |
| U14 decals | U10 |
| U17 performance, U19 content check, U20 package | U14 |
| U18 headless bench | U9 |

## 7. Open questions for the owner
1. ~~Approve the switch from Lyra to the template base (§0)?~~ **Decided 2026-10-20: approved.**
2. `square_arena`: Antigravity typed it into `arenas.json` only. Add it to `src/sim/arena.ts` (then the web game has it too) or drop it?
3. Zombies in the demo (U12) or bots only?
4. Camera yaw -45° (turned like Project Zomboid) or -90° (north up like the web arena)? `build_arena.py --camera-yaw` makes both; decide from the U2 screenshots.
