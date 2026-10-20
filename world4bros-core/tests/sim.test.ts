import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildContent } from '../src/data/content';
import { validateContent } from '../src/data/validate';
import { makeRng } from '../src/engine/rng';
import { Ecs } from '../src/engine/ecs';
import { Component } from '../src/engine/ecs';
import { GameWorld } from '../src/sim/world';
import { CHUNK } from '../src/sim/worldgen';
import { Inventory } from '../src/sim/inventory';
import { defaultLook, equip, lookSignature, randomLook, sanitizeLook } from '../src/sim/look';
import { Game, NO_INPUT, DEFAULT_CONFIG } from '../src/sim/game';
import { Comp } from '../src/sim/components';

const content = buildContent();

test('content is consistent', () => { assert.deepEqual(validateContent(content), []); });

test('rng is deterministic and forks are stable', () => {
  const a = makeRng(7), b = makeRng(7);
  for (let i = 0; i < 5; i++) assert.equal(a.next(), b.next());
  assert.equal(makeRng(7).fork('x').next(), makeRng(7).fork('x').next());
  assert.notEqual(makeRng(7).fork('x').next(), makeRng(7).fork('y').next());
});

test('ecs query returns entities with all components', () => {
  const ecs = new Ecs(); const A = new Component<number>('a'); const B = new Component<number>('b');
  const e1 = ecs.create(), e2 = ecs.create();
  ecs.add(e1, A, 1); ecs.add(e1, B, 2); ecs.add(e2, A, 3);
  assert.deepEqual(ecs.query(A, B), [e1]); assert.deepEqual(ecs.query(A).sort(), [e1, e2]);
  ecs.destroy(e1); assert.deepEqual(ecs.query(A, B), []);
});

test('world generation is deterministic per seed', () => {
  const w1 = new GameWorld(content, 99), w2 = new GameWorld(content, 99), w3 = new GameWorld(content, 100);
  const c1 = w1.chunkAt(3, -2), c2 = w2.chunkAt(3, -2), c3 = w3.chunkAt(3, -2);
  assert.deepEqual(Array.from(c1.terrain), Array.from(c2.terrain));
  assert.deepEqual(c1.objects.map((o) => o.id + o.kind), c2.objects.map((o) => o.id + o.kind));
  assert.notEqual(Array.from(c1.terrain).join(), Array.from(c3.terrain).join());
});

test('spawn is on walkable meadow and clear of objects', () => {
  const w = new GameWorld(content, 4242);
  assert.ok(w.walkable(w.spawn.x, w.spawn.y));
  assert.equal(w.blocked(w.spawn.x, w.spawn.y, 0.3), false);
});

test('several biomes appear across the map', () => {
  const w = new GameWorld(content, 4242); const seen = new Set<string>();
  for (let y = -400; y <= 400; y += 20) for (let x = -400; x <= 400; x += 20) seen.add(w.biomeAt(x, y));
  assert.ok(seen.size >= 5, `only saw ${[...seen].join(', ')}`);
});

test('object edits persist across chunk eviction', () => {
  const w = new GameWorld(content, 4242);
  const o = w.objectsNear(w.spawn.x, w.spawn.y, 30).find((x) => x.kind === 'pine');
  assert.ok(o);
  for (let i = 0; i < 20; i++) w.damageObject(o!.id, 1);
  assert.equal(w.getObject(o!.id)?.kind, 'stump');
  w.evictFar(10000, 10000, 1);
  assert.equal(w.getObject(o!.id)?.kind, 'stump');
  const [cx] = o!.id.split(':'); assert.ok(cx);
  assert.ok(Math.abs(o!.x) < 1e6 && CHUNK === 16);
});

test('inventory stacks, removes and round-trips', () => {
  const inv = new Inventory(content, 4);
  assert.equal(inv.add('wood', 25), 0); assert.equal(inv.count('wood'), 25);
  assert.equal(inv.slots[0]?.qty, 20);
  assert.equal(inv.add('axe', 5), 3);   // axes do not stack, and only 2 slots are free after the wood
  assert.ok(inv.remove('wood', 22)); assert.equal(inv.count('wood'), 3);
  assert.equal(inv.remove('wood', 5), false);
  const copy = new Inventory(content, 4); copy.load(inv.toJSON()); assert.equal(copy.count('wood'), 3);
});

test('equipment: slot rules, colors and signatures', () => {
  const look = defaultLook();
  assert.equal(equip(look, content, 'head', 'axe'), false);               // wrong slot
  assert.equal(equip(look, content, 'head', 'beanie', { primary: '#112233' }), true);
  const sig1 = lookSignature(look);
  look.appearance.hairColor = '#a8456f';
  assert.notEqual(lookSignature(look), sig1);                              // hair color changes the signature
  const sig2 = lookSignature(look);
  look.appearance.wheelchair = true;
  assert.notEqual(lookSignature(look), sig2);                              // wheelchair state changes the signature
  equip(look, content, 'head', null); assert.equal(look.outfit.head, undefined);
});

test('sanitizeLook rejects bad data and keeps good data', () => {
  const l = sanitizeLook(content, { appearance: { hairColor: 'red', hairStyle: 'nope', build: 9, wheelchair: 'yes' as unknown as boolean }, outfit: { head: { item: 'axe' }, torso: { item: 'sweater', colors: { primary: '#abcdef', secondary: 'bad' } } } });
  assert.match(l.appearance.hairColor, /^#[0-9a-f]{6}$/i); assert.equal(l.appearance.hairStyle, 'messy'); assert.equal(l.appearance.build, 1.2); assert.equal(l.appearance.wheelchair, false);
  const valid = sanitizeLook(content, { appearance: { wheelchair: true } });
  assert.equal(valid.appearance.wheelchair, true);
  assert.equal(l.outfit.head, undefined); assert.equal(l.outfit.torso?.colors?.primary, '#abcdef'); assert.equal(l.outfit.torso?.colors?.secondary, undefined);
});

test('random looks are valid and varied', () => {
  const rng = makeRng(5); const sigs = new Set<string>();
  for (let i = 0; i < 20; i++) { const l = randomLook(content, rng); sigs.add(lookSignature(l)); assert.deepEqual(lookSignature(sanitizeLook(content, l)).length > 0, true); }
  assert.ok(sigs.size > 15);
});

test('player can chop a tree with an axe and gets wood', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: false });
  const tree = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 40).filter((o) => o.kind === 'pine' || o.kind === 'oak').sort((a, b) => Math.hypot(a.x - g.world.spawn.x, a.y - g.world.spawn.y) - Math.hypot(b.x - g.world.spawn.x, b.y - g.world.spawn.y))[0];
  assert.ok(tree, 'a tree exists near spawn');
  const t = g.playerT; t.x = tree!.x - 0.9; t.y = tree!.y; t.angle = 0;
  const before = g.playerInv.inv.count('wood');
  for (let swing = 0; swing < 12; swing++) {
    g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
    for (let i = 0; i < 40; i++) g.step(1 / 60, NO_INPUT);
  }
  assert.ok(g.playerInv.inv.count('wood') > before);
  assert.equal(g.world.getObject(tree!.id)?.kind, 'stump');
});

test('wrong tool gives a toast and does not harvest', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: false }); let toasts = 0; g.events.on('toast', () => toasts++);
  const rock = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 60).find((o) => o.kind === 'rock');
  assert.ok(rock);
  const t = g.playerT; t.x = rock!.x - 0.8; t.y = rock!.y; t.angle = 0;
  g.holdSlot(2);   // torch, not a pickaxe
  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  assert.equal(toasts, 1); assert.equal(g.world.getObject(rock!.id)?.hp, rock!.hp);
});

test('base building: claim rules, costs, collision and refunds', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: false });
  const p = g.playerT; const tx = Math.floor(p.x) + 3, ty = Math.floor(p.y);
  assert.equal(g.checkBuild(content.structures.get('wall_wood')!, tx, ty).ok, false);      // needs a claim
  assert.ok(g.build('claim_flag', tx, ty));
  const wallsBefore = g.playerInv.inv.count('wood');
  const e = g.build('wall_wood', tx + 2, ty); assert.ok(e);
  assert.equal(g.playerInv.inv.count('wood'), wallsBefore - 3);
  assert.equal(g.world.blocked(tx + 2.5, ty + 0.5, 0.2), true);                             // wall blocks movement
  assert.equal(g.build('wall_wood', tx + 2, ty), null);                                      // occupied
  g.demolish(e!); assert.equal(g.world.blocked(tx + 2.5, ty + 0.5, 0.2), false);
  assert.equal(g.build('claim_flag', tx + 3, ty), null);                                     // overlapping claim
});

test('save and load round-trips the whole game', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: true });
  const p = g.playerT; g.build('claim_flag', Math.floor(p.x) + 3, Math.floor(p.y));
  g.build('chest', Math.floor(p.x) + 3, Math.floor(p.y) + 2);
  g.playerChar.look.appearance.hairColor = '#a8456f'; equip(g.playerChar.look, content, 'head', 'beanie', { primary: '#445566' });
  g.world.mods.set('0,0:0', { removed: true });
  const json = JSON.stringify(g.toSave());
  const g2 = Game.fromSave(JSON.parse(json));
  assert.equal(g2.playerChar.look.appearance.hairColor, '#a8456f');
  assert.equal(g2.playerChar.look.outfit.head?.colors?.primary, '#445566');
  assert.equal(g2.base.entities().length, 2);
  assert.deepEqual(JSON.parse(JSON.stringify(g2.toSave())), JSON.parse(json));
  assert.equal(g2.ecs.query(Comp.Player).length, 1);
});

test('day cycle darkens and survival needs react to light', () => {
  const g = new Game({ ...DEFAULT_CONFIG, darknessDamage: true });
  g.clock.t = 0.8; assert.ok(g.clock.darkness > 0.8);
  g.holdSlot(2); assert.equal(g.isLit(g.playerT.x, g.playerT.y), true);   // torch lights the player
  g.holdSlot(7);
  assert.equal(g.isLit(g.playerT.x, g.playerT.y), false);
  const h0 = g.playerNeeds.health;
  for (let i = 0; i < 60 * 10; i++) g.step(1 / 60, NO_INPUT);
  assert.ok(g.playerNeeds.health < h0 || g.playerNeeds.sanity < 100);
});

test('alwaysDay locks sunlight at t=0.25 and persists across saves', () => {
  const g = new Game({ ...DEFAULT_CONFIG, alwaysDay: true });
  assert.equal(g.clock.alwaysDay, true);
  assert.equal(g.clock.t, 0.25);
  assert.equal(g.clock.darkness, 0);
  assert.equal(g.clock.phase, 'day');
  g.step(10, NO_INPUT);
  assert.equal(g.clock.t, 0.25);
  assert.equal(g.clock.darkness, 0);

  // Toggle off allows night progression
  g.clock.alwaysDay = false;
  g.clock.t = 0.8;
  assert.ok(g.clock.darkness > 0.8);
  assert.equal(g.clock.phase, 'night');

  // Serialization preserves alwaysDay flag
  g.clock.alwaysDay = true;
  const save = g.toSave();
  assert.equal(save.alwaysDay, true);
  const loaded = Game.fromSave(save);
  assert.equal(loaded.clock.alwaysDay, true);
  assert.equal(loaded.clock.darkness, 0);
  assert.equal(loaded.clock.phase, 'day');
});

