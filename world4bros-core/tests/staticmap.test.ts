import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';

test('static map bakes, edits, and survives a save round-trip', () => {
  const g = new Game(DEFAULT_CONFIG);
  const w = g.world;
  const sp = w.spawn;
  const before = w.terrainIdAt(sp.x + 40, sp.y + 40);
  w.setStatic(true);
  assert.equal(w.staticActive, true);
  assert.equal(w.terrainIdAt(sp.x + 40, sp.y + 40), before, 'baking must not change the terrain');
  const savedElevation = w.elevationAt(sp.x + 40, sp.y + 40);

  const target = before === 'snow' ? 'sand' : 'snow';
  assert.ok(w.paint(sp.x + 40, sp.y + 40, 2, target, null) > 0);
  assert.equal(w.terrainIdAt(sp.x + 40, sp.y + 40), target);
  assert.ok(w.dirty.size > 0, 'edits mark chunks dirty for the renderer');

  const o = w.placeObject('tent', sp.x + 40.5, sp.y + 40.5);
  assert.ok(o);
  assert.ok(w.objectsNear(sp.x + 40.5, sp.y + 40.5, 0.5).some((x) => x.id === o!.id));
  assert.ok(w.eraseObjects(sp.x + 40.5, sp.y + 40.5, 0.6) >= 1);
  assert.ok(!w.getObject(o!.id));

  assert.equal(w.terrainIdAt(100000, 100000), 'deep', 'outside the bounds is open water');
  assert.equal(w.paint(100000, 100000, 2, 'grass', null), 0, 'cannot paint outside the bounds');

  const save = JSON.parse(JSON.stringify(g.toSave()));
  const g2 = Game.fromSave(save);
  assert.equal(g2.world.staticActive, true);
  assert.equal(g2.world.terrainIdAt(sp.x + 40, sp.y + 40), target);
  assert.equal(g2.world.elevationAt(sp.x + 40, sp.y + 40), savedElevation, 'elevation must survive static-map save/load');

  const legacySave = JSON.parse(JSON.stringify(g.toSave()));
  for (const chunk of legacySave.staticMap.chunks) delete chunk.elevation;
  const legacyLoaded = Game.fromSave(legacySave);
  assert.ok(Math.abs(legacyLoaded.world.elevationAt(sp.x + 40, sp.y + 40) - savedElevation) <= 1 / 255,
    'older static-map saves should regenerate compatible elevation values');

  w.setStatic(false);
  assert.equal(w.terrainIdAt(sp.x + 40, sp.y + 40), before, 'generative world is untouched by static edits');
  w.setStatic(true);
  assert.equal(w.terrainIdAt(sp.x + 40, sp.y + 40), target, 'edits are kept when toggling back');
});
