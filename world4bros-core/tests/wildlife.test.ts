import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';
import type { AnimalKind } from '../src/data/schema';

test('all 7 animal archetypes spawn with valid stats and components', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  const kinds: AnimalKind[] = ['cow', 'sheep', 'pig', 'chicken', 'deer', 'rabbit', 'goose'];

  for (let i = 0; i < kinds.length; i++) {
    const kind = kinds[i];
    const ae = g.spawnAnimal(g.playerT.x + i * 2, g.playerT.y, kind);
    const a = g.ecs.get(ae, Comp.Animal)!;
    const at = g.ecs.get(ae, Comp.Transform)!;
    const ab = g.ecs.get(ae, Comp.Body)!;

    assert.ok(a, `animal ${kind} should have Animal component`);
    assert.equal(a.kind, kind);
    assert.ok(a.hp > 0, `animal ${kind} has positive HP`);
    assert.ok(at, `animal ${kind} has Transform`);
    assert.ok(ab, `animal ${kind} has Body`);
    assert.ok(a.def.drops.length > 0, `animal ${kind} has drop table`);
  }
});

test('cows can be milked, sheep sheared, and chicken eggs collected', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);

  // 1. Cow milking
  const cowE = g.spawnAnimal(g.playerT.x + 1.2, g.playerT.y, 'cow', 'Daisy');
  const cow = g.ecs.get(cowE, Comp.Animal)!;
  cow.produceTimer = 0; // ready for milking

  g.playerT.angle = 0; // face cow
  const target = g.findTarget();
  assert.equal(target?.kind, 'animal');
  assert.equal((target as any).action, 'milk');

  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });
  assert.equal(g.playerInv.inv.count('fresh_milk'), 1, 'should receive 1 fresh_milk');
  assert.ok(cow.produceTimer > 50, 'cow produceTimer should reset after milking');

  // 2. Sheep shearing
  const sheepE = g.spawnAnimal(g.playerT.x, g.playerT.y + 1.2, 'sheep', 'Fluffy');
  const sheep = g.ecs.get(sheepE, Comp.Animal)!;
  sheep.produceTimer = 0;

  g.playerT.angle = Math.PI / 2; // face sheep
  const sheepTarget = g.findTarget();
  assert.equal(sheepTarget?.kind, 'animal');
  assert.equal((sheepTarget as any).action, 'shear');

  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });
  assert.ok(g.playerInv.inv.count('wool_bundle') >= 2, 'should receive wool bundles');

  // 3. Chicken eggs
  const henE = g.spawnAnimal(g.playerT.x - 1.2, g.playerT.y, 'chicken', 'Henrietta');
  const hen = g.ecs.get(henE, Comp.Animal)!;
  hen.produceTimer = 0;

  g.playerT.angle = Math.PI; // face chicken
  const henTarget = g.findTarget();
  assert.equal(henTarget?.kind, 'animal');
  assert.equal((henTarget as any).action, 'eggs');

  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });
  assert.ok(g.playerInv.inv.count('fresh_egg') >= 1, 'should receive fresh egg');
});

test('wildlife flees when threatened by encroaching zombies', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  const de = g.spawnAnimal(g.playerT.x + 5.0, g.playerT.y, 'deer', 'Bambi');
  const d = g.ecs.get(de, Comp.Animal)!;
  const dt = g.ecs.get(de, Comp.Transform)!;

  // Spawn zombie close to the deer
  g.spawnZombie(dt.x + 2.0, dt.y, 'walker');

  assert.equal(d.state, 'graze');
  g.step(1 / 60, NO_INPUT);

  assert.equal(d.state, 'flee', 'deer should detect encroaching zombie and flee');
  assert.ok(d.fleeTimer > 0, 'deer should have active flee timer');
});

test('zombies predate and attack nearby wildlife', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  // Put player far away (50m) so zombie targets the closer animal
  g.playerT.x = 50.0;
  g.playerT.y = 50.0;

  const ce = g.spawnAnimal(10.0, 10.0, 'cow', 'Daisy');
  const c = g.ecs.get(ce, Comp.Animal)!;
  const initialHp = c.hp;

  const ze = g.spawnZombie(10.8, 10.0, 'walker');
  const z = g.ecs.get(ze, Comp.Zombie)!;

  // Step simulation so zombie detects cow and attacks
  g.step(1 / 60, NO_INPUT);
  assert.equal(z.state, 'chase');
  assert.equal(z.target, ce, 'zombie should target nearest cow');

  // Let zombie execute attack
  z.attackTimer = 0;
  g.step(1 / 60, NO_INPUT);

  assert.ok(c.hp < initialHp, 'cow should take damage from zombie attack');
});

test('animals can be fed and tamed with favorite food', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  // Put wild carrot in active hotbar slot 0 and hold it
  g.playerInv.inv.slots[0] = { item: 'wild_carrot', qty: 5 };
  g.holdSlot(0);

  const pe = g.spawnAnimal(g.playerT.x + 1.2, g.playerT.y, 'pig', 'Barnaby');
  const p = g.ecs.get(pe, Comp.Animal)!;
  p.hp = 15; // hurt pig

  g.playerT.angle = 0;
  const target = g.findTarget();
  assert.equal(target?.kind, 'animal');
  assert.equal((target as any).action, 'feed');

  const prevSanity = g.playerNeeds.sanity;
  g.playerNeeds.sanity = 70;
  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });

  assert.equal(p.tamed, true, 'pig should become tamed after feeding');
  assert.ok(p.hp > 15, 'feeding should heal the animal');
  assert.ok(g.playerNeeds.sanity > 70, 'feeding animal restores player sanity');
});

test('hunting animals drops meats and materials, enabling cooking', () => {
  const g = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  g.playerInv.inv.add('chainsaw_of_regret', 1);
  g.wear('chainsaw_of_regret');

  const ce = g.spawnAnimal(g.playerT.x + 1.2, g.playerT.y, 'cow', 'Daisy');
  const c = g.ecs.get(ce, Comp.Animal)!;
  c.hp = 20; // low HP

  g.playerT.angle = 0;
  const actionTarget = g.findActionTarget();
  assert.equal(actionTarget?.kind, 'animal');

  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  for (let i = 0; i < 20; i++) g.step(1 / 60, NO_INPUT);

  assert.ok(g.playerInv.inv.count('raw_beef') >= 2, 'cow defeat drops raw beef');
  assert.ok(g.playerInv.inv.count('leather') >= 2, 'cow defeat drops leather');

  // Test recipe cooking: raw_beef + wood -> cooked_steak
  g.playerInv.inv.add('wood', 2);
  assert.ok(g.craft('cooked_steak'), 'should craft sizzling cooked steak');
  assert.equal(g.playerInv.inv.count('cooked_steak'), 1);
});

test('animals persist across save and load cycles', () => {
  const g1 = new Game(DEFAULT_CONFIG, undefined, false, false, false);
  g1.spawnAnimal(12.5, 8.5, 'cow', 'Daisy');
  g1.spawnAnimal(14.0, 9.0, 'sheep', 'Fluffy');
  g1.spawnAnimal(10.0, 7.0, 'deer', 'Bambi');

  const saved = g1.toSave();
  assert.ok(saved.animals, 'saved data must include animals');
  assert.equal(saved.animals?.length, 3);

  const g2 = Game.fromSave(saved);
  const loadedAnimals = g2.ecs.query(Comp.Animal);
  assert.equal(loadedAnimals.length, 3, 'all 3 animals should reload from save');

  const cows = loadedAnimals.filter((e) => g2.ecs.get(e, Comp.Animal)?.kind === 'cow');
  assert.equal(cows.length, 1);
  const loadedCow = g2.ecs.get(cows[0], Comp.Animal)!;
  assert.equal(loadedCow.name, 'Daisy');
});
