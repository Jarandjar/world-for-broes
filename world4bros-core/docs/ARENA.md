# Aim Arena and Test Bench: design and build plan for Antigravity

Save as `docs/ARENA.md`. **Claude designed this; Antigravity builds it.** Every screen is drawn on the design canvas "World 4 Bro's: Aim Arena" (https://claude.ai/artifact/8zwMs4m3fmyA2pCy5vgULq): Pick a match, Online lobby, The two arenas, In a match (desktop and phone), Results, Bots and fair play, and Test Bench. Build to those boards. This brief has no code on purpose. Earlier prototypes (`swarm-sim-reference.html`, `terrain-lab.html`) are reference material only; do not copy their code.

> **Update 2026-10-20 (owner decision):** the arena is now built in **Unreal Engine 5.8** (phase Z34; started on 5.7, project World4Bros) on Unreal's First Person C++ template, Arena Shooter variant (not Lyra; reasons in [UE_PIPELINE.md](UE_PIPELINE.md) §1). Step-by-step to a finished demo: [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md). This file stays the design source (maps, rules, bot levels, Test Bench): maps export from `src/sim/arena.ts`, bot levels from `src/data/arenaBots.ts` and the §4 numbers from `src/data/arenaRules.ts` through `npm run ue:export`. Unreal is not fully deterministic, so the Unreal Test Bench records seeds and the spread of repeated runs (UE_PIPELINE.md §4.5). The tone rule changed too: **mild blood is allowed**, so the "no blood, green puff" line in §4 is replaced by small blood splats and decals.

## 1. What the owner asked for

1. A place to **aim and shoot**: a 1v1 against a bot, and a free-for-all with several players (bots stand in for real players when testing).
2. **Watch it in real time first**, then run the same matches **as fast as possible** to gather data on guns, attachments and bot logic.
3. Build order: offline bot duel, then online 1v1, then free-for-all. The first step needs no network.

## 2. What the repo already has (Verified in the uploaded `world4bros-core`)

The shooting range (`teleportToShootingRange`, target dummies) and `shootGun` with projectiles, falloff and penetration; zombies with noise and senses; a noise system; `tests/ballistics-range.test.ts`. **Not present:** match rules, bots that shoot, rounds, respawns, a headless runner, a command queue or replay. Raiders do not exist in the repo. Build on the range and gun code; do not replace it. The attachment system in `docs/WEAPONS.md` should land first or alongside, so guns can be tested with attachments.

## 3. Arena maps (data, Proposed)

Units are metres (tiles). Size, spawns, cover and pickups are all data, so a new arena needs no engine change. Crates, low walls and sandbags stop bullets and block movement. Barrels and trees block movement only.

**Duel Pit, 40 x 24 m.** Spawns: 1 (4,12), 2 (36,12). Cream pies: (20,3), (20,21). Crates (x, y, width, height): (17,10,2,2) (21,12,2,2) (19,5,2,1) (19,18,2,1). Sandbag rows, 0.5 m thick (x, y, length, direction): (11,6,5,h) (24,17,5,h) (11,17,5,h) (24,6,5,h). Low walls: (26,9,1,3) (13,12,1,3). Barrels: (9,12) (31,12) (16,3) (24,21). Trees: (3,3) (37,21) (3,21) (37,3). Turn it half a turn and it matches itself, so both sides get identical cover.

**Free-for-all Yard, 64 x 40 m.** Spawns 1 to 8: (6,6) (32,4) (58,6) (60,20) (58,34) (32,36) (6,34) (4,20). Pies: (32,10) (32,30) (14,20) (50,20). Crates: (28,15,2,2) (34,15,2,2) (28,23,2,2) (34,23,2,2) (10,12,2,2) (52,26,2,2) (52,12,2,2) (10,26,2,2) (31,19,2,2). Sandbags: (14,8,6,h) (44,30,6,h) (44,8,6,h) (14,30,6,h) (22,18,4,v) (40,18,4,v) (22,28,4,h) (38,11,4,h). Low walls: (26,10,1,3) (37,27,1,3) (12,18,3,1) (49,21,3,1). Barrels: (20,14) (44,25) (20,25) (44,14) (32,12) (32,28) (8,20) (56,20). Trees: (2,2) (62,2) (2,38) (62,38) (16,4) (48,36) (48,4) (16,36). Roads (open ground, visual only): horizontal 18 to 22 m, vertical 30 to 34 m. **The Yard is not mirrored.** The Test Bench must check each spawn's win rate, and layout changes follow from that data.

Layout rules: no spawn sees another spawn within 12 m; everywhere is within 6 m of something that stops bullets. Both maps use the same scale as the Proving Grounds so distance markers mean the same everywhere.

## 4. Match rules (all Proposed starting values)

- **Bot duel and online 1v1:** rounds. Both fighters reset to their spawns after each kill, with a short pause. First to 3, 5, 7 or 10. A round with no kill after 30 s is a draw and resets. A match time limit ends it with the leader winning.
- **Free-for-all:** 4 to 8 players, kill limit (10, 15, 25) or time limit (4, 6, 8 min). Respawn after 3 s at the spawn farthest from enemies (never within 12 m of one, never in view if avoidable). Spawn protection 2 s, ended early when you fire.
- **Cream pies:** heal a little (suggest 25), respawn after 20 s. Bots seek them when hurt.
- **Hits** pop into a green puff and a "bonk". No blood. Cannot be turned off in the arena.
- **Loadouts** come from the Gunsmith, are free here, and never touch the save.
- **Readout rule stays:** no enemy health bars unless a fitted part gives them. The Test Bench is a developer tool and may show them.

## 5. Bots (Proposed)

Four levels, from the "Bots and fair play" board: **Scarecrow** (reacts in 0.9 s, misses by up to 12 degrees, shoots every 1.2 s, walks straight, no cover, no pies), **Regular** (0.6 s, 7 degrees, strafes randomly, takes cover when hurt), **Sharp** (0.35 s, 3.5 degrees, leads shots, peeks from cover, sometimes throws a pie), **Menace** (0.2 s, 1.5 degrees, leads shots, flanks, uses pies to blind). Rules for every bot: same guns, fire rate, damage and health as a player; sees only through its own view and hears by noise radius (no wallhacks); never aims perfectly; level fixed for the whole match (no rubber-banding). A bot plays a player's slot in tests, so a free-for-all can be tested with no network.

## 6. Test Bench (the new part; see the board)

- **Two modes, one match code.** Watch draws the match; Data run draws nothing and steps as fast as the machine allows. Both must call the same simulation, so a result never depends on the mode.
- **Speed:** 1x, 2x, 4x, 16x, Max (as fast as a frame budget allows while still drawing). Show the achieved speed as a multiple of real time. Pause and single step.
- **Deterministic.** Fixed time step, seeded random numbers, no wall-clock in the simulation. The same seed and settings give the same result in Watch and in Data run. This is the main acceptance test.
- **Overlays (Watch):** health bars, bot modes and targets, noise rings, heat map (where enemies died, where shots came from, where players were hurt).
- **Experiments (Data run):** *Gun matchups* (every gun against every other, win rate matrix); *Bot level ladder*; *Ping fairness* (win rate of a high-ping player against ping, lag help on and off); *Free-for-all balance* (placement, kills, K/D over many matches). Rules: same seeds for every pairing, sides swapped every match, a count of matches per pairing (10, 25, 50), progress with time left, Stop, and results as a table.
- **Output:** copy as CSV or JSON, and save a file. Each result records the settings and the seed so it can be replayed.
- **Telemetry:** reuse `docs/SWARM_TELEMETRY.md` section 4 (event log, per-weapon stats, distance bands, time to kill, accuracy counted once per bullet), plus per round: who fired first, who hit first, round length, and whether the first shooter won.
- **Simulated lag (before real networking):** each fighter has a ping. A fighter sees others as they were one ping ago, so it aims at stale positions. With lag help on, the server checks a shot against where the target was up to 150 ms ago. Measure fairness as win rate against ping. This is only a model; real networking is a separate approved step.

## 7. Build plan (stop for approval after each)

1. **Z28a Arenas as data:** both maps as data definitions, drawn in the renderer, solids and bullet-stoppers honoured, spawns and pies. *Exit:* walk and shoot in both; bullets stop on crates; the Duel Pit passes a half-turn symmetry test.
2. **Z28b Bot duel (offline):** the four bot levels, rounds, score, pause between rounds, the match HUD and results. *Exit:* a Sharp bot beats a Scarecrow most of the time; a bot cannot see through a crate.
3. **Z28c Test Bench:** the deterministic headless runner, Watch and Data run, speeds, telemetry, the four experiments, export. *Exit:* same seed, same result in both modes; a stated measured speed in Data run; the matchup matrix runs end to end.
4. **Z28d Online 1v1:** first the simulated-lag model above, then a separate plan for real networking (server in charge, rooms and codes) for approval.
5. **Z28e Free-for-all:** the Yard, eight spawns, spawn rules and protection, pies, respawn, scoreboard, lobby, results.

## 8. Tests to add

Determinism (same seed, same summary; Watch equals Data run). Accuracy never above 100%. A bullet cannot pass a crate. A bot cannot target through a crate. Spawn protection blocks damage and ends when you fire. A pie heals once and respawns on time. Kill limit and time limit end the match. A draw round resets cleanly. At zero ping on both sides the win rate is about even over many matches. No spawn is within 12 m of an enemy at respawn when a farther spawn exists. Headless duel speed is measured and reported (no target is promised until it is measured).

## 9. Out of scope

Real networking, voice chat, ranked ladders, spectating other people's matches, new weapons, ammo and reload (the game has none yet).

## 10. Open questions for the owner

- Where does the Arena sit in the game: a main-menu mode, or a place in the world like the Proving Grounds?
- Should the Test Bench ship in normal builds or only in developer builds? (Suggest developer only.)
- Do you want the Yard changed after the first spawn-fairness numbers?
- Are the proposed numbers (bot levels, 25 pie heal, 150 ms lag help) a good starting point?

## 11. Paste-ready prompt for Antigravity

```
Read docs/ARENA.md and open the design canvas it links to. Build phase Z28a only: the Duel Pit and Free-for-all Yard as data definitions (size, spawns,
cover, pickups, roads), drawn in the existing renderer, with bullet-stopping and movement-blocking solids honoured, using the Proving Grounds and
shootGun code as the base. Do not build bots, rounds or the Test Bench yet. Keep simulation code free of rendering. Add the Z28a tests from section 8
(bullets stop on crates; half-turn symmetry for the Duel Pit) and run the full suite. Stop for my approval after the plan and again when done.
```
