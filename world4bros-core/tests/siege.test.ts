import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('running industrial machines generate pollution in chunk', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('stone_furnace')!;
  const tx = 10, ty = 10;
  const e = g.base.place(def, tx, ty, 'E');
  const m = g.ecs.get(e, Comp.Machine)!;
  m.fuel = 20;
  m.progress = 1.0;

  const initialPollution = g.siege.getPollutionAt(tx, ty);
  assert.equal(initialPollution, 0);

  // Step simulation forward
  for (let i = 0; i < 10; i++) g.step(0.1, NO_INPUT);

  const updatedPollution = g.siege.getPollutionAt(tx, ty);
  assert.ok(updatedPollution > 0, 'furnace operation should generate measurable chunk pollution');
});

test('natural pollution dissipates over time', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.siege.addPollution(0, 0, 5.0);
  assert.equal(g.siege.getPollutionAt(0, 0), 5.0);

  // Dissipation tick occurs every 2.0s
  for (let i = 0; i < 30; i++) g.step(0.1, NO_INPUT);

  const remaining = g.siege.getPollutionAt(0, 0);
  assert.ok(remaining < 5.0, 'pollution should naturally decay over time');
});

test('belt_bolter turret engages nearest zombie, consumes ammo, and deals damage', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('belt_bolter')!;
  const te = g.base.place(def, 10, 10, 'E');

  const ze = g.spawnZombie(12, 10, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const initialHp = z.hp;

  // Step to attach turret and fire
  g.step(0.1, NO_INPUT);

  const turret = g.ecs.get(te, Comp.Turret)!;
  assert.ok(turret, 'belt_bolter should have Turret component');
  assert.equal(turret.ammo, 29, 'should consume 1 ammo per shot');
  assert.equal(z.hp, initialHp - turret.damage, 'zombie should receive bolt damage');
});

test('belt_bolter auto-reloads from adjacent conveyor belt carrying turret_ammo', () => {
  const g = new Game(DEFAULT_CONFIG);
  const bolterDef = g.content.structures.get('belt_bolter')!;
  const convDef = g.content.structures.get('conveyor_belt')!;

  const be = g.base.place(bolterDef, 10, 10, 'E');
  const ce = g.base.place(convDef, 10, 11, 'E');

  // Step once to register turret component
  g.step(0.1, NO_INPUT);
  const turret = g.ecs.get(be, Comp.Turret)!;
  turret.ammo = 15; // depleted from 30

  // Add turret ammo onto conveyor
  const conv = g.ecs.get(ce, Comp.Conveyor)!;
  conv.items.push({ item: 'turret_ammo', progress: 0.5 });

  // Step to trigger auto-feed logic
  g.step(0.1, NO_INPUT);

  assert.equal(turret.ammo, 25, 'turret should reload +10 ammo from adjacent belt');
  assert.equal(conv.items.length, 0, 'ammo item should be consumed from belt');
});

test('zap_fence shocks and knocks back encroaching zombie', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('zap_fence')!;
  g.base.place(def, 10, 10, 'E');

  const ze = g.spawnZombie(11.1, 10.5, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const zb = g.ecs.get(ze, Comp.Body)!;
  const initialHp = z.hp;

  g.step(0.1, NO_INPUT);

  assert.ok(z.hp < initialHp, 'zombie should be shocked by zap fence');
  assert.ok(zb.vx > 0, 'zombie should receive electrical knockback impulse');
});

test('sapper mutant explodes on contact with industrial machine', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('stone_furnace')!;
  g.base.place(def, 10, 10, 'E');

  const ze = g.spawnZombie(10.8, 10, 'sapper');
  assert.equal(g.ecs.get(ze, Comp.Zombie)!.def.kind, 'sapper');

  // Step forward: sapper touches machine and explodes
  g.step(0.1, NO_INPUT);

  assert.ok(!g.ecs.has(ze, Comp.Zombie), 'sapper should be destroyed upon detonation');
  assert.ok(g.siege.getPollutionAt(10, 10) >= 8, 'sapper powder-keg blast should generate heavy soot/pollution');
});

test('siege data persists across save and load', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.siege.addPollution(32, 32, 28.5);
  g.siege.totalSieges = 3;

  const save = g.toSave();
  assert.ok(save.siege, 'save data should contain siege state');

  const g2 = Game.fromSave(save);
  assert.equal(g2.siege.getPollutionAt(32, 32), 28.5);
  assert.equal(g2.siege.totalSieges, 3);
});

test('screamer queen rally toasts are throttled across multiple screamers', () => {
  const g = new Game(DEFAULT_CONFIG);
  let toastCount = 0;
  g.events.on('toast', (e) => {
    if (e.text.includes('Screamer Queen')) toastCount++;
  });

  // Spawn 3 screamers close by
  const z1 = g.spawnZombie(5, 5, 'screamer');
  const z2 = g.spawnZombie(6, 5, 'screamer');
  const z3 = g.spawnZombie(7, 5, 'screamer');

  // Trigger attack timers to 0 simultaneously
  g.ecs.get(z1, Comp.Zombie)!.attackTimer = 0;
  g.ecs.get(z2, Comp.Zombie)!.attackTimer = 0;
  g.ecs.get(z3, Comp.Zombie)!.attackTimer = 0;

  g.step(0.1, NO_INPUT);

  // Despite 3 screamers screaming simultaneously, only 1 toast is emitted
  assert.equal(toastCount, 1, 'concurrent screamers should emit only 1 throttled rally toast');
});
