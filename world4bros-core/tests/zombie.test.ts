import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('zombies spawn in zombie mode with correct archetypes', () => {
  const g = new Game({ ...DEFAULT_CONFIG, mode: 'zombie' });
  const zombies = g.ecs.query(Comp.Zombie);
  assert.ok(zombies.length >= 5, 'should spawn initial zombies in zombie mode');

  const kinds = new Set(zombies.map((e) => g.ecs.get(e, Comp.Zombie)!.def.kind));
  assert.ok(kinds.has('walker') || kinds.has('runner') || kinds.has('brute'));
});

test('noise alerts nearby zombies toward source', () => {
  const g = new Game(DEFAULT_CONFIG);
  const ze = g.spawnZombie(10, 10, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  assert.equal(z.state, 'idle');

  // Emit noise at (9, 9) within hearing radius
  g.emitNoise(9, 9, 8);
  assert.equal(z.state, 'alert');
  assert.deepEqual(z.alertPos, { x: 9, y: 9 });
});

test('player can target and attack zombies with weapons', () => {
  const g = new Game(DEFAULT_CONFIG);
  const ze = g.spawnZombie(g.playerT.x + 1.2, g.playerT.y, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const initialHp = z.hp;

  g.playerT.angle = 0; // face zombie
  const target = g.findTarget();
  assert.ok(target);
  assert.equal(target?.kind, 'zombie');

  // Swing at zombie
  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  // Progress swing to hit point (0.55 of swingDur)
  for (let i = 0; i < 20; i++) g.step(1 / 60, NO_INPUT);

  assert.ok(z.hp < initialHp, 'zombie should take damage');
  assert.equal(z.state, 'chase', 'damaged zombie should chase attacker');
});

test('vehicle can ram and crush zombies at speed', () => {
  const g = new Game(DEFAULT_CONFIG);
  const veh = g.ecs.query(Comp.Vehicle)[0]!;
  const vt = g.ecs.get(veh, Comp.Transform)!;
  const ze = g.spawnZombie(vt.x + 0.8, vt.y, 'runner');
  const z = g.ecs.get(ze, Comp.Zombie)!;

  // Set car speed to ramming speed
  const v = g.ecs.get(veh, Comp.Vehicle)!;
  v.driver = g.player;
  v.speed = 4.5;

  g.step(1 / 60, { ...NO_INPUT, move: { x: 1, y: 0 } });
  assert.ok(z.hp < z.def.maxHp || !g.ecs.has(ze, Comp.Zombie), 'ramming at speed should inflict damage');
});

test('zombies persist across save and load', () => {
  const g = new Game({ ...DEFAULT_CONFIG, mode: 'zombie' });
  const ze = g.spawnZombie(15, 20, 'brute');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  z.hp = 42;

  const save = g.toSave();
  assert.ok(save.zombies && save.zombies.length > 0);

  const g2 = Game.fromSave(save);
  const loadedZombies = g2.ecs.query(Comp.Zombie);
  assert.ok(loadedZombies.length > 0);
  const foundBrute = loadedZombies.find((e) => g2.ecs.get(e, Comp.Zombie)!.hp === 42);
  assert.ok(foundBrute, 'saved brute HP should round-trip');
});
