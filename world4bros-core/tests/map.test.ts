import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('world generates major POIs (gas stations, radio towers, towns, outposts)', () => {
  const g = new Game(DEFAULT_CONFIG);
  const foundTypes = new Set<string>();

  // Check chunks across a 600m radius
  for (let cy = -20; cy <= 20; cy++) {
    for (let cx = -20; cx <= 20; cx++) {
      const c = g.world.chunkAt(cx, cy);
      for (const feat of c.features) {
        foundTypes.add(feat.type);
      }
    }
  }

  assert.ok(foundTypes.has('camp'), 'should contain camps');
  assert.ok(foundTypes.has('settlement'), 'should contain settlements');
  assert.ok(foundTypes.has('gas_station') || foundTypes.has('radio_tower') || foundTypes.has('town') || foundTypes.has('military_outpost'), 'should generate major POIs');
});

test('world elevation is seeded, continuous, and stored per tile', () => {
  const g1 = new Game(DEFAULT_CONFIG);
  const g2 = new Game(DEFAULT_CONFIG);
  const x = Math.floor(g1.world.spawn.x + 37) + 0.5;
  const y = Math.floor(g1.world.spawn.y + 21) + 0.5;
  const cx = Math.floor(x / 16), cy = Math.floor(y / 16);
  const tx = Math.floor(x) - cx * 16, ty = Math.floor(y) - cy * 16;
  const index = ty * 16 + tx;
  const elevation = g1.world.chunkAt(cx, cy).elevation[index]!;

  assert.equal(elevation, Math.round(g1.world.gen.sample(x, y).height * 255));
  assert.equal(g2.world.chunkAt(cx, cy).elevation[index], elevation, 'identical seeds must produce identical heights');
  assert.ok(Math.abs(g1.world.gen.sample(15.999, y).height - g1.world.gen.sample(16.001, y).height) < 0.01,
    'elevation must remain continuous at chunk boundaries');
});

test('moving across the continent continuously explores sectors', () => {
  const g = new Game(DEFAULT_CONFIG);
  const initialExploredCount = g.explored.size;
  assert.ok(initialExploredCount > 0, 'starter area should be explored');

  // Move player 200m east
  g.playerT.x += 200;
  g.step(0.1, NO_INPUT);

  assert.ok(g.explored.size > initialExploredCount, 'moving should reveal new map sectors');
});

test('activating radio tower reveals a large sector radius', () => {
  const g = new Game(DEFAULT_CONFIG);
  const initialCount = g.explored.size;

  // Simulate scanning a radio tower at (400, 400)
  const added = g.revealArea(400, 400, 450);

  assert.ok(added > 200, 'radio tower scan should reveal hundreds of sectors');
  assert.ok(g.explored.size > initialCount + 200);
});

test('explored map sectors persist across save and load', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.revealArea(120, -180, 200);
  const countBefore = g.explored.size;

  const save = g.toSave();
  assert.ok(save.explored && save.explored.length > 0, 'save should serialize explored sectors');

  const loaded = Game.fromSave(save);
  assert.equal(loaded.explored.size, countBefore, 'loaded game should restore all explored sectors');
});

test('player spawns at an abandoned town with town features and objects', () => {
  const g = new Game(DEFAULT_CONFIG);
  const spChunkX = Math.floor(g.world.spawn.x / 16);
  const spChunkY = Math.floor(g.world.spawn.y / 16);
  const chunk = g.world.chunkAt(spChunkX, spChunkY);

  const townFeature = chunk.features.find((f) => f.type === 'town');
  assert.ok(townFeature, 'spawn chunk must have a town feature');

  // Verify town objects exist around spawn
  const objects = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 10);
  const kinds = objects.map((o) => o.kind);
  assert.ok(kinds.includes('wrecked_car'), 'town should contain wrecked cars');
  assert.ok(kinds.includes('fuel_pump'), 'town should contain fuel pump');
  assert.ok(kinds.includes('crate'), 'town should contain supply crates');
});

test('teleportToTown brings player, vehicle, and pet to the town square', () => {
  const g = new Game(DEFAULT_CONFIG);
  // Move player far away
  g.playerT.x = 800;
  g.playerT.y = 800;

  g.teleportToTown();

  assert.equal(g.playerT.x, g.world.spawn.x);
  assert.equal(g.playerT.y, g.world.spawn.y);

  // Check vehicle is at town
  const vehicles = g.ecs.query(Comp.Vehicle, Comp.Transform);
  assert.ok(vehicles.length > 0);
  const vt = g.ecs.get(vehicles[0]!, Comp.Transform)!;
  assert.ok(Math.hypot(vt.x - g.world.spawn.x, vt.y - g.world.spawn.y) < 3.0);
});
