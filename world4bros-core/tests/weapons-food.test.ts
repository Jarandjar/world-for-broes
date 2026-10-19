import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('pool_noodle_of_justice delivers massive cartoon knockback impulse', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false);
  g.playerInv.inv.add('pool_noodle_of_justice', 1);
  g.wear('pool_noodle_of_justice');

  const startDist = 1.5;
  const ze = g.spawnZombie(g.playerT.x + startDist, g.playerT.y, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const zt = g.ecs.get(ze, Comp.Transform)!;
  const initialHp = z.hp;
  const initialX = zt.x;

  g.playerT.angle = 0; // face zombie
  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  for (let i = 0; i < 20; i++) g.step(1 / 60, NO_INPUT);

  assert.equal(z.hp, initialHp - 2, 'pool noodle deals precisely 2 damage');
  assert.ok(zt.x - initialX > 0.8, 'pool noodle launches zombie backwards');
});

test('chainsaw_of_regret deals massive damage and generates loud noise', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false);
  g.playerInv.inv.add('chainsaw_of_regret', 1);
  g.wear('chainsaw_of_regret');

  const ze = g.spawnZombie(g.playerT.x + 1.2, g.playerT.y, 'brute');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const initialHp = z.hp;

  g.playerT.angle = 0;
  let noiseEmitted = false;
  g.events.on('noise', (e) => {
    if (e.radius >= 20) noiseEmitted = true;
  });

  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  for (let i = 0; i < 20; i++) g.step(1 / 60, NO_INPUT);

  assert.equal(z.hp, initialHp - 35, 'chainsaw should deliver 35 heavy damage');
  assert.ok(noiseEmitted, 'chainsaw should emit large noise radius');
});

test('slingshot consumes pebbles and damages targets from range', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false);
  g.playerInv.inv.add('slingshot', 1);
  g.playerInv.inv.add('pebbles', 5);
  g.wear('slingshot');

  const ze = g.spawnZombie(g.playerT.x + 5.0, g.playerT.y, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const initialHp = z.hp;

  g.playerT.angle = 0;
  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  for (let i = 0; i < 20; i++) g.step(1 / 60, NO_INPUT);

  assert.equal(g.playerInv.inv.count('pebbles'), 4, 'should consume 1 pebble');
  assert.equal(z.hp, initialHp - 8, 'slingshot pebble should inflict 8 damage at range');
});

test('side-effect culinary foods apply buffs and comedic mechanics', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.playerInv.inv.add('hiccup_berries', 1);
  g.playerInv.inv.add('chili_of_regret', 1);
  g.playerInv.inv.add('rainbow_soup', 1);
  g.playerInv.inv.add('crispy_sock_chips', 1);

  // 1. Hiccup berries
  const hiccupIdx = g.playerInv.inv.slots.findIndex((s) => s?.item === 'hiccup_berries');
  g.eat(hiccupIdx);
  assert.equal(g.buffs.hiccupTimer, 30.0, 'eating hiccup berries sets 30s hiccup buff');

  // 2. Chili of regret
  const chiliIdx = g.playerInv.inv.slots.findIndex((s) => s?.item === 'chili_of_regret');
  g.eat(chiliIdx);
  assert.equal(g.buffs.fireBreathTimer, 12.0, 'eating chili sets 12s fire breath');

  // 3. Rainbow soup
  const rainbowIdx = g.playerInv.inv.slots.findIndex((s) => s?.item === 'rainbow_soup');
  g.eat(rainbowIdx);
  assert.equal(g.buffs.rainbowGlowTimer, 60.0);
  const lights = g.lights();
  assert.ok(lights.some((l) => l.radius >= 8.0), 'rainbow soup generates personal radiant aura');

  // 4. Crispy sock chips
  const sockIdx = g.playerInv.inv.slots.findIndex((s) => s?.item === 'crispy_sock_chips');
  g.eat(sockIdx);
  assert.equal(g.buffs.speedBoostTimer, 25.0);
});

test('throwables (cream pie, water balloon, road flare) affect undead targets', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.playerInv.inv.add('cream_pie', 1);
  g.playerInv.inv.add('water_balloon', 1);
  g.playerInv.inv.add('road_flare', 1);

  const ze = g.spawnZombie(10, 10, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  z.state = 'chase';
  z.target = g.player;

  // Throw cream pie
  const pieOk = g.throwItem('cream_pie', 10, 10);
  assert.ok(pieOk);
  assert.equal(z.state, 'wander', 'cream pie should blind zombie into wandering');

  // Throw water balloon
  const balloonOk = g.throwItem('water_balloon', 10, 10);
  assert.ok(balloonOk);

  // Throw road flare
  const flareOk = g.throwItem('road_flare', 12, 12);
  assert.ok(flareOk);
  assert.equal(g.activeFlares.length, 1);
  assert.ok(g.lights().some((l) => l.x === 12 && l.y === 12), 'road flare should emit 8m light source');
});

test('wearable gear traits (pillow vest damage reduction, boing boots speed)', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false);

  // Test boing boots speed boost
  const b = g.ecs.get(g.player, Comp.Body)!;
  g.step(0.1, { ...NO_INPUT, move: { x: 1, y: 0 } });
  const normalSpeed = Math.hypot(b.vx, b.vy);

  g.playerInv.inv.add('boing_boots', 1);
  g.wear('boing_boots');
  g.step(0.1, { ...NO_INPUT, move: { x: 1, y: 0 } });
  const boostedSpeed = Math.hypot(b.vx, b.vy);
  assert.ok(boostedSpeed > normalSpeed, 'boing boots should grant a passive movement speed boost');

  // Test pillow vest damage reduction
  g.playerInv.inv.add('pillow_vest', 1);
  g.wear('pillow_vest');
  const initialHp = g.playerNeeds.health;

  // Spawn zombie right on player to trigger attack
  const ze = g.spawnZombie(g.playerT.x + 0.3, g.playerT.y, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  z.state = 'chase';
  z.target = g.player;

  g.step(0.1, NO_INPUT);
  const dmgTaken = initialHp - g.playerNeeds.health;
  assert.ok(dmgTaken <= Math.round(z.def.damage * 0.75), 'pillow vest should absorb 25% of bite damage');
});

test('buffs and master items persist across save and load', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.buffs.fireBreathTimer = 8.5;
  g.buffs.rainbowGlowTimer = 44.0;
  g.playerInv.inv.add('pool_noodle_of_justice', 1);

  const save = g.toSave();
  assert.ok(save.buffs, 'save data should preserve buffs');

  const g2 = Game.fromSave(save);
  assert.equal(g2.buffs.fireBreathTimer, 8.5);
  assert.equal(g2.buffs.rainbowGlowTimer, 44.0);
  assert.equal(g2.playerInv.inv.count('pool_noodle_of_justice'), 1);
});
