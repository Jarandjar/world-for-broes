# Swarm testing and telemetry: brief for Antigravity

Save as `docs/SWARM_TELEMETRY.md`. **Claude designed this; Antigravity implements it in the game.** `swarm-sim-reference.html` is a working lab that shows every number below; its pure `Sim` section (between the banner comments) can be read or lifted. The lab is a testbed, not the game's logic.

## 1. What it is for

Run a **swarm of zombies and raiders** against a bro squad, log everything, and use the numbers to (a) compare guns and attachments, (b) tune zombie and raider behaviour, (c) catch balance problems before players do. Two modes: **PvE** (bros vs zombies) and **PvPvE** (bros vs raiders, with zombies attacking everyone and drawn in by gunfire).

## 2. Rules the sim must follow

- **Fixed step 1/30 s, seeded RNG, no `Math.random`.** The same seed and settings must give the same result. Test it.
- **Pure core.** Simulation has no rendering or DOM, so a batch can run headless and fast (the lab does a 45 s run with ~120 zombies in well under a second).
- **Noise is the universal pressure.** Every shot emits a noise radius (weapon noise plus any `noise_pulse`). Zombies inside it wake or turn toward it; raiders inside it investigate. Log how many it woke.
- **Zombie numbers come from `src/data/zombies.ts`** (hp, damage, speed, runSpeed, detectRadius, attackCooldown). The lab adds radius and attack reach. **Raider numbers are proposals** (the repo has none): raider hp 60, speed 3.0; heavy hp 100, speed 2.4; marksman hp 50, speed 2.8.
- **Same weapon rules for everyone.** Bullets use the existing `GunDef` fields and damage falloff. No friendly fire between bros, between raiders, or between zombies.

## 3. Bot brain (shared by bros and raiders)

Modes: `kite` (back away from zombies inside a set distance), `engage` (advance to the gun's ideal range, hold or strafe, back off if too close), `cover` (when hurt, go behind the nearest bullet-stopping object, away from the threat), `investigate` (raiders walk to heard gunfire), `advance` (raiders push toward the bros), `regroup` (bros stay near their leader).
Perception: sight radius with a **real line-of-sight check** (bullet-stopping objects block it); hearing by noise radius. A bot never knows where something is without seeing or hearing it.
Targeting: score = nearness, a bonus for zombies inside 4.5 m, a bonus for enemy raiders/bros over zombies, a bonus for low-health targets, a little stickiness to the current target.
Tunable per team (the lab's sliders): `reaction` (s before first shot at a new target), `aimErr` (std dev, radians), `kite` (m), `lead` (0..1, shot leading), `cover` (0..1), `aggr` (raid push speed).
Aim: aim at the target (leading by `lead`), add Gaussian `aimErr`, then the gun's own spread (x1.7 when moving, x0.85 when still, as in `shootGun`).

Zombie logic: `sleep` (wakes on noise within 80% of radius, or a target within half its detect radius), `wander`, `alert` (go to a noise or scream), `chase`, attack on contact. Screamers alert all zombies within 24 m. Sappers explode on contact. Spitters attack from ~5.5 m. **No pathfinding** in the lab: zombies steer, slide along obstacles and drift sideways to avoid sticking. The real game should keep its own movement.

## 4. Data to gather

**Event log**, one row per event, CSV: `t, type (shot|hit|kill), actorTeam, actorKind, weapon, victimTeam, victimKind, distM, dmg, hpAfter`. Cap it (the lab caps at 150k rows). Optional in normal play, on in test runs.
**Per weapon (key = team + weapon + attachments):** shots, pellets fired, pellets that hit (a piercing bullet counts once), damage, kills, overkill, noise emitted, zombies woken, kill distances, time-to-kill samples (first damage to death), and per distance band (0-3, 3-6, 6-10, 10-15, 15-25, 25+ m): pellets aimed, pellets hit, damage.
**Per team:** alive counts sampled every 0.5 s, damage taken, deaths, kills by victim kind, and 2 m heat grids for where enemies died, where shots came from, and where bros were hurt.
**Run summary** (for comparing): outcome (`cleared | wiped | time`), kills, kills/min, accuracy, damage dealt, shots per kill, average kill distance, average time-to-kill, overkill %, noise per kill, zombies woken per kill, bro damage taken, bros lost.
Accuracy by distance uses the distance **aimed at when the shot was fired**, so it can never exceed 100%.

## 5. Comparing guns

Same scenario, same seeds for every gun, N runs each, report mean and spread. The lab runs this headless and shows the table. Pair seeds across guns so differences come from the gun, not the luck. Use enough runs (10 or more) before trusting a difference; two runs swing a lot.

## 6. First results from the lab (PvE, 80 zombies, 45 s, seed 5; one run each, so only a smoke test)

Rifle: 51 kills, no damage taken, one-shots most zombies (0.09 s to kill), about 12 m kill distance. Shotgun: 60 kills but took damage and had about 22% pellet accuracy. SMG: 52 kills at about 8 shots per kill. Foam blaster: 31 kills at about 14 shots per kill. Slingshot: 34 kills. Nail bat: squad wiped. Treat these as a sanity check that the pipeline works, not as balance advice.

## 7. Not modelled (decided)

No ammo limits, reloads, jams or heat (the game has none yet). No pathfinding around cover. Raiders do not use vehicles, throwables or healing. Raider stats are guesses. In the **waves** pattern zombies arrive already alerted, so "zombies woken per kill" reads near zero; use the **sleepers** pattern to measure noise.

## 8. Tests to add

Determinism (same seed, same summary; different seed differs). Accuracy never above 100%. A zombie cannot be hit through a bullet-stopping object. Stunned zombies do not move. Noise wakes sleepers only inside the radius. Headless run of 300 zombies stays fast (the lab: about 0.6 ms per step). Comparison runner returns paired-seed means.

## 9. Open questions

- Should raiders become a real faction in the game (data file, spawn rules, loot), and where do the proposed stats live?
- Do you want ammo and reload added before gun comparisons, since a big magazine changes everything?
- Which scenario is the standard benchmark (suggest: PvE, 120 zombies, waves, 60 s, 10 seeds)?

## 10. Codespaces + Spark implementation notes

### Codespaces

- Dev container file: `.devcontainer/devcontainer.json`
- Post-create bootstrap: `.devcontainer/postCreate.sh`
- Forward port `5173`, then run:

```bash
npm run dev -- --host
```

Set forwarded port visibility to **Public** for remote playtests.

### Spark telemetry analysis

A local Spark batch analyzer is included:

- Script: `tools/spark/analyze_swarm.py`
- Python deps: `tools/spark/requirements.txt`
- NPM wrapper:

```bash
npm run telemetry:spark
```

This reads event CSV files (`tools/spark/input/*.csv`) and writes aggregate parquet outputs:

- `tools/spark/out/per_weapon`
- `tools/spark/out/per_band`
- `tools/spark/out/run_summary`

### CI verification

- GitHub Actions workflow: `.github/workflows/verify.yml`
- Runs `npm run verify` on push/PR to `main`.

## 11. Paste-ready prompt

```
Read docs/SWARM_TELEMETRY.md and open swarm-sim-reference.html (the Sim section is the reference). Build a headless test harness in the game repo:
a seeded, fixed-step scenario runner using the real Game/zombie/gun code, the event log and per-weapon stats in section 4, and the paired-seed gun
comparison in section 5. Do not copy the lab's movement code; use the game's own. Add the tests in section 8. Stop for my approval after the plan and
again when done.
```
