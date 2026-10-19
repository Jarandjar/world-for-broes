import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('starter companion pet spawns by player spawn', () => {
  const g = new Game(DEFAULT_CONFIG);
  const pets = g.ecs.query(Comp.Pet);
  assert.equal(pets.length, 1, 'should spawn starter companion pet');

  const pet = g.ecs.get(pets[0]!, Comp.Pet)!;
  assert.equal(pet.name, 'Barkley');
  assert.equal(pet.state, 'follow');
  assert.equal(pet.hp, 80);
});

test('companion follows the player when moving away', () => {
  const g = new Game(DEFAULT_CONFIG);
  const petEntity = g.ecs.query(Comp.Pet)[0]!;
  const pt = g.ecs.get(petEntity, Comp.Transform)!;

  // Move player far to the east
  g.playerT.x += 10;
  const initialDist = Math.hypot(g.playerT.x - pt.x, g.playerT.y - pt.y);

  // Step simulation so pet runs after player
  for (let i = 0; i < 40; i++) {
    g.step(0.05, NO_INPUT);
  }

  const newDist = Math.hypot(g.playerT.x - pt.x, g.playerT.y - pt.y);
  assert.ok(newDist < initialDist, 'pet should run toward player to close distance');
});

test('player can target and pet Barkley to restore sanity', () => {
  const g = new Game(DEFAULT_CONFIG);
  const petEntity = g.ecs.query(Comp.Pet)[0]!;
  const pt = g.ecs.get(petEntity, Comp.Transform)!;

  // Place pet right in front of player
  pt.x = g.playerT.x + 0.8;
  pt.y = g.playerT.y;

  g.playerNeeds.sanity = 50;

  const target = g.findTarget();
  assert.ok(target);
  assert.equal(target?.kind, 'pet');
  assert.equal(target?.prompt, 'Pet Barkley');

  let barkReceived = false;
  g.events.on('petBark', () => { barkReceived = true; });

  // Press E to interact
  g.step(0.016, { ...NO_INPUT, interactPressed: true });

  assert.equal(barkReceived, true, 'petting should trigger petBark event');
  assert.ok(g.playerNeeds.sanity > 50, 'petting Barkley should increase player sanity');
});

test('pet detects, barks at, and bites nearby zombies', () => {
  const g = new Game(DEFAULT_CONFIG);
  const petEntity = g.ecs.query(Comp.Pet)[0]!;
  const pt = g.ecs.get(petEntity, Comp.Transform)!;

  // Spawn zombie within detection radius (e.g. 2.0 meters away)
  const ze = g.spawnZombie(pt.x + 2.0, pt.y, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;
  const initialHp = z.hp;

  let barkHeard = false;
  g.events.on('petBark', () => { barkHeard = true; });

  // Run initial steps: pet detects zombie and enters attack state
  g.step(0.05, NO_INPUT);
  const pet = g.ecs.get(petEntity, Comp.Pet)!;
  assert.equal(pet.state, 'attack', 'pet should switch to attack state against zombie');

  // Run sim so pet reaches zombie and bites
  for (let i = 0; i < 40; i++) {
    g.step(0.03, NO_INPUT);
  }

  assert.ok(barkHeard, 'pet should bark when detecting zombie');
  assert.ok(z.hp < initialHp || !g.ecs.has(ze, Comp.Zombie), 'pet bite should inflict damage on zombie');
});

test('pet persists across save and load cycles', () => {
  const g = new Game(DEFAULT_CONFIG);
  const petEntity = g.ecs.query(Comp.Pet)[0]!;
  const pt = g.ecs.get(petEntity, Comp.Transform)!;
  pt.x = 42.5;
  pt.y = 88.2;
  const pet = g.ecs.get(petEntity, Comp.Pet)!;
  pet.name = 'Timber';
  pet.kind = 'wolf_pup';
  pet.hp = 65;

  const save = g.toSave();
  assert.ok(save.pets && save.pets.length > 0, 'save should contain pets');

  const loaded = Game.fromSave(save);
  const loadedPets = loaded.ecs.query(Comp.Pet);
  assert.equal(loadedPets.length, 1);

  const lp = loaded.ecs.get(loadedPets[0]!, Comp.Pet)!;
  const lpt = loaded.ecs.get(loadedPets[0]!, Comp.Transform)!;
  assert.equal(lp.name, 'Timber');
  assert.equal(lp.kind, 'wolf_pup');
  assert.equal(lp.hp, 65);
  assert.equal(lpt.x, 42.5);
  assert.equal(lpt.y, 88.2);
});
