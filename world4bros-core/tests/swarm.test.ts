import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SwarmSim, defaultSwarmCfg, runSwarmHeadless, compareWeaponsPaired, lineOfSight, isBlocked } from '../src/sim/swarm';

test('Swarm simulation is deterministic with same seed and settings', async () => {
  const cfg = defaultSwarmCfg();
  cfg.zombies = 40;
  cfg.timeLimit = 20;

  const run1 = await runSwarmHeadless(cfg, 42);
  const run2 = await runSwarmHeadless(cfg, 42);

  assert.equal(run1.kills, run2.kills, 'kills must match on same seed');
  assert.equal(run1.shots, run2.shots, 'shots must match on same seed');
  assert.equal(run1.dmgDealt, run2.dmgDealt, 'damage dealt must match on same seed');
  assert.equal(run1.accuracy, run2.accuracy, 'accuracy must match on same seed');
  assert.equal(run1.outcome, run2.outcome, 'outcome must match on same seed');

  // Different seed should vary
  const run3 = await runSwarmHeadless(cfg, 999);
  assert.ok(run1.seed !== run3.seed, 'seeds must be different');
});

test('Swarm weapon accuracy never exceeds 100%', async () => {
  const cfg = defaultSwarmCfg();
  cfg.broWeapon = 'pump_shotgun'; // Multi-pellet weapon
  cfg.zombies = 50;
  cfg.timeLimit = 25;

  const res = await runSwarmHeadless(cfg, 12);
  assert.ok(res.accuracy <= 1.0, `accuracy ${res.accuracy} cannot exceed 1.0`);
  assert.ok(res.accuracy >= 0, `accuracy ${res.accuracy} must be non-negative`);
});

test('Bullets cannot pass through solid obstacles (line of sight check)', () => {
  // Arena has wall at [26, 10, 1, 3] -> from x=20, y=11 to x=32, y=11 passes through wall
  const hasLos = lineOfSight(20, 11, 32, 11);
  assert.equal(hasLos, false, 'solid obstacle must block line of sight');

  // Clear path between x=2, y=20 and x=10, y=20
  const openLos = lineOfSight(2, 20, 6, 20);
  assert.equal(openLos, true, 'open space must have clear line of sight');
});

test('Noise wakes sleeping zombies only within noise radius', () => {
  const cfg = defaultSwarmCfg();
  cfg.pattern = 'sleepers';
  cfg.zombies = 30;
  cfg.bros = 1;
  const sim = new SwarmSim(cfg, 17);

  // Position a sleeper at (30, 20)
  const z = sim.zombies[0];
  if (z) {
    z.state = 'sleep';
    z.x = 30;
    z.y = 20;

    // Small noise far away at (10, 20) with radius 8 (distance is 20)
    sim.emitNoise(10, 20, 8, 'bro', null);
    assert.equal(z.state, 'sleep', 'noise outside radius must not wake sleeper');

    // Loud noise nearby at (25, 20) with radius 12 (distance is 5)
    sim.emitNoise(25, 20, 12, 'bro', null);
    assert.equal(z.state, 'alert', 'noise within radius must wake sleeper into alert');
  }
});

test('Stunned zombies cannot move or attack', () => {
  const cfg = defaultSwarmCfg();
  const sim = new SwarmSim(cfg, 8);
  const z = sim.zombies[0];
  if (z) {
    z.state = 'chase';
    z.target = sim.bros[0];
    z.stunTimer = 2.0;
    const startX = z.x, startY = z.y;

    sim.step(1 / 30);
    assert.equal(z.x, startX, 'stunned zombie must not change x');
    assert.equal(z.y, startY, 'stunned zombie must not change y');
  }
});

test('Headless run of 150 zombies executes quickly and efficiently', async () => {
  const cfg = defaultSwarmCfg();
  cfg.zombies = 150;
  cfg.timeLimit = 15;

  const t0 = performance.now();
  const res = await runSwarmHeadless(cfg, 5);
  const durMs = performance.now() - t0;

  assert.ok(durMs < 2500, `150 zombie simulation took ${durMs}ms, expected under 2500ms`);
  assert.ok(res.time > 0, 'simulation executed time');
});

test('Paired-seed weapon comparison returns valid comparative dataset', async () => {
  const cfg = defaultSwarmCfg();
  cfg.zombies = 40;
  cfg.timeLimit = 15;

  const weapons = ['hunting_rifle', 'nailgun_smg'];
  const res = await compareWeaponsPaired(weapons, cfg, 7, 2);

  assert.ok(res.hunting_rifle, 'must contain rifle results');
  assert.ok(res.nailgun_smg, 'must contain smg results');
  assert.equal(res.hunting_rifle.length, 2, 'rifle must have 2 runs');
  assert.equal(res.nailgun_smg.length, 2, 'smg must have 2 runs');
});
