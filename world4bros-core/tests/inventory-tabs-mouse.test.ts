import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../src/sim/game';
import { insideBuilding, type Building } from '../src/sim/worldgen';

test('base storage inventory stores 30 slots and survives save/load cycles', () => {
  const g = new Game({ seed: 42, mode: 'classic', creative: false, darknessDamage: false }, undefined, false, false, false);
  assert.equal(g.baseStorage.slots.length, 30);

  // Add items to base storage
  g.baseStorage.add('wood', 50);
  g.baseStorage.add('canned_beans', 5);
  g.baseStorage.add('boing_boots', 1);

  assert.equal(g.baseStorage.count('wood'), 50);
  assert.equal(g.baseStorage.count('canned_beans'), 5);
  assert.equal(g.baseStorage.count('boing_boots'), 1);

  // Save and restore
  const save = g.toSave();
  assert.ok(save.baseStorage);

  const restored = Game.fromSave(save);
  assert.equal(restored.baseStorage.slots.length, 30);
  assert.equal(restored.baseStorage.count('wood'), 50);
  assert.equal(restored.baseStorage.count('canned_beans'), 5);
  assert.equal(restored.baseStorage.count('boing_boots'), 1);
});

test('base storage deposit all, quick stack, and withdraw all operations', () => {
  const g = new Game({ seed: 42, mode: 'classic', creative: false, darknessDamage: false }, undefined, false, false, false);
  const pack = g.playerInv.inv;
  const base = g.baseStorage;

  // Clear pack and base
  pack.slots.fill(null);
  base.slots.fill(null);

  // Setup pack items
  pack.add('stone', 30);
  pack.add('iron_ore', 15);
  pack.add('fresh_milk', 2);

  // 1. Deposit All into Base
  for (let i = 0; i < pack.slots.length; i++) {
    const st = pack.slots[i];
    if (!st) continue;
    const left = base.add(st.item, st.qty);
    pack.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
  }

  assert.equal(pack.count('stone'), 0);
  assert.equal(pack.count('iron_ore'), 0);
  assert.equal(pack.count('fresh_milk'), 0);
  assert.equal(base.count('stone'), 30);
  assert.equal(base.count('iron_ore'), 15);
  assert.equal(base.count('fresh_milk'), 2);

  // 2. Add some more stone to pack, and Quick Stack
  pack.add('stone', 20);
  pack.add('wild_carrot', 5); // Base does NOT have wild_carrot yet

  for (let i = 0; i < pack.slots.length; i++) {
    const st = pack.slots[i];
    if (!st) continue;
    if (base.count(st.item) > 0) {
      const left = base.add(st.item, st.qty);
      pack.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
    }
  }

  // Stone should have moved into base, but wild_carrot remains in pack
  assert.equal(pack.count('stone'), 0);
  assert.equal(base.count('stone'), 50);
  assert.equal(pack.count('wild_carrot'), 5);
  assert.equal(base.count('wild_carrot'), 0);

  // 3. Withdraw All from Base to Pack
  for (let i = 0; i < base.slots.length; i++) {
    const st = base.slots[i];
    if (!st) continue;
    const left = pack.add(st.item, st.qty);
    base.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
  }

  assert.equal(pack.count('stone'), 50);
  assert.equal(pack.count('iron_ore'), 15);
  assert.equal(pack.count('fresh_milk'), 2);
  assert.equal(pack.count('wild_carrot'), 5);
  assert.equal(base.count('stone'), 0);
  assert.equal(base.count('iron_ore'), 0);
});

test('interior shelter detection correctly identifies when coordinates are inside building', () => {
  const building: Building = {
    id: '0,0:bld_test',
    def: 'cottage',
    x: 10,
    y: 20,
    w: 8,
    h: 6,
    door: 'S',
    doorAt: 4,
  };

  // Inside the building interior
  assert.equal(insideBuilding(building, 14, 23), true);
  assert.equal(insideBuilding(building, 11, 21), true);

  // Outside the building
  assert.equal(insideBuilding(building, 9, 23), false);
  assert.equal(insideBuilding(building, 19, 23), false);
  assert.equal(insideBuilding(building, 14, 19), false);
  assert.equal(insideBuilding(building, 14, 27), false);
});
