import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('mineral nodes generate in world chunks with valid drops', () => {
  const g = new Game({ seed: 4242, mode: 'classic', creative: true, darknessDamage: false }, undefined, false, false);
  const chunk = g.world.chunkAt(0, 0);
  const nodes = chunk.objects.filter((o) =>
    o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode'
  );
  // We scan a 3x3 chunk area around spawn to guarantee encountering geodes
  let foundNodes = 0;
  for (let cy = -2; cy <= 2; cy++) {
    for (let cx = -2; cx <= 2; cx++) {
      const ch = g.world.chunkAt(cx, cy);
      const m = ch.objects.filter((o) =>
        o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode'
      );
      foundNodes += m.length;
    }
  }
  assert.ok(foundNodes > 0, 'Found discrete mineral nodes in nearby chunks');
});

test('conveyor belt transports items and passes to next conveyor', () => {
  const g = new Game({ seed: 1001, mode: 'classic', creative: true, darknessDamage: false }, undefined, false, false);
  const p = g.playerT;
  const tx = Math.floor(p.x) + 3, ty = Math.floor(p.y);
  const b1 = g.base.place(g.content.structures.get('conveyor_belt')!, tx, ty, 'E');
  const b2 = g.base.place(g.content.structures.get('conveyor_belt')!, tx + 1, ty, 'E');
  assert.ok(b1 !== undefined && b2 !== undefined);

  const c1 = g.ecs.get(b1, Comp.Conveyor)!;
  const c2 = g.ecs.get(b2, Comp.Conveyor)!;
  assert.equal(c1.dir, 'E');
  assert.equal(c2.dir, 'E');

  // Put an iron_ore on the first belt
  c1.items.push({ item: 'iron_ore', progress: 0.8 });

  // Step simulation forward
  g.step(0.2); // 0.8 + 0.2 * 1.5 = 1.1 -> should advance to b2
  assert.equal(c1.items.length, 0, 'Item moved off first belt');
  assert.equal(c2.items.length, 1, 'Item arrived on second belt');
  assert.equal(c2.items[0].item, 'iron_ore');
});

test('spring inserter grabs item from belt and deposits into chest', () => {
  const g = new Game({ seed: 1002, mode: 'classic', creative: true, darknessDamage: false }, undefined, false, false);
  const p = g.playerT;
  const tx = Math.floor(p.x) + 3, ty = Math.floor(p.y);
  // Belt at (tx, ty) facing E
  // Inserter at (tx + 1, ty) with pickup W (tx, ty), drop E (tx + 2, ty)
  // Chest at (tx + 2, ty)
  const b1 = g.base.place(g.content.structures.get('conveyor_belt')!, tx, ty, 'E');
  const insE = g.base.place(g.content.structures.get('spring_inserter')!, tx + 1, ty, 'E');
  const chestE = g.base.place(g.content.structures.get('chest')!, tx + 2, ty, 'E');
  assert.ok(b1 !== undefined && insE !== undefined && chestE !== undefined);

  const conv = g.ecs.get(b1, Comp.Conveyor)!;
  conv.items.push({ item: 'copper_ore', progress: 0.5 });

  const ins = g.ecs.get(insE, Comp.Inserter)!;
  assert.equal(ins.pickupDir, 'W');
  assert.equal(ins.dropDir, 'E');

  // Step 1: Inserter picks up item
  g.step(0.05);
  assert.equal(conv.items.length, 0, 'Item picked from belt');
  assert.equal(ins.heldItem, 'copper_ore', 'Inserter holds copper_ore');

  // Step 2: Swing arm to completion (speed = 1.4, needs ~0.8s)
  for (let i = 0; i < 12; i++) g.step(0.1);

  assert.equal(ins.heldItem, null, 'Inserter deposited item');
  const chest = g.ecs.get(chestE, Comp.Structure)!.storage!;
  assert.equal(chest.count('copper_ore'), 1, 'Chest received copper_ore');
});

test('stone furnace smelts ore into finished metal products', () => {
  const g = new Game({ seed: 1003, mode: 'classic', creative: true, darknessDamage: false }, undefined, false, false);
  const p = g.playerT;
  const tx = Math.floor(p.x) + 3, ty = Math.floor(p.y);
  const furnaceE = g.base.place(g.content.structures.get('stone_furnace')!, tx, ty, 'E');
  assert.ok(furnaceE !== undefined);

  const mach = g.ecs.get(furnaceE, Comp.Machine)!;
  mach.targetOre = 'iron_ore';

  // Step for 3 seconds (> 2.8s required)
  for (let i = 0; i < 35; i++) g.step(0.1);

  assert.equal(mach.outputItem, 'iron_plate', 'Furnace smelted iron_ore into iron_plate');
});

test('automation structures persist through save and load', () => {
  const g1 = new Game({ seed: 2026, mode: 'classic', creative: true, darknessDamage: false }, undefined, false, false);
  const p = g1.playerT;
  const tx = Math.floor(p.x) + 3, ty = Math.floor(p.y);
  const b1 = g1.base.place(g1.content.structures.get('conveyor_belt')!, tx, ty, 'S');
  const insE = g1.base.place(g1.content.structures.get('spring_inserter')!, tx, ty + 1, 'S');
  const furnE = g1.base.place(g1.content.structures.get('stone_furnace')!, tx, ty + 2, 'E');

  const c1 = g1.ecs.get(b1, Comp.Conveyor)!;
  c1.items.push({ item: 'coal_chunk', progress: 0.45 });

  const ins = g1.ecs.get(insE, Comp.Inserter)!;
  ins.heldItem = 'iron_ore';
  ins.armProgress = 0.5;

  const mach = g1.ecs.get(furnE, Comp.Machine)!;
  mach.targetOre = 'copper_ore';
  mach.progress = 1.2;

  const save = g1.toSave();
  const g2 = Game.fromSave(save);

  const b2 = g2.base.structureAt(tx, ty)!;
  const c2 = g2.ecs.get(b2, Comp.Conveyor)!;
  assert.equal(c2.dir, 'S');
  assert.equal(c2.items.length, 1);
  assert.equal(c2.items[0].item, 'coal_chunk');
  assert.equal(c2.items[0].progress, 0.45);

  const insE2 = g2.base.structureAt(tx, ty + 1)!;
  const ins2 = g2.ecs.get(insE2, Comp.Inserter)!;
  assert.equal(ins2.dropDir, 'S');
  assert.equal(ins2.heldItem, 'iron_ore');
  assert.equal(ins2.armProgress, 0.5);

  const furnE2 = g2.base.structureAt(tx, ty + 2)!;
  const mach2 = g2.ecs.get(furnE2, Comp.Machine)!;
  assert.equal(mach2.targetOre, 'copper_ore');
  assert.equal(mach2.progress, 1.2);
});
