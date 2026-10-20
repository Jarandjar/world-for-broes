import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';
import { getStreetDetails, roadPathsNear, REGIONAL_TOWNS } from '../src/sim/worldgen';

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

test('teleportToTown brings player, vehicle, and pet to fixed town plaza slots', () => {
  const g = new Game(DEFAULT_CONFIG);
  // Move player far away
  g.playerT.x = 800;
  g.playerT.y = 800;

  g.teleportToTown();

  assert.equal(g.playerT.x, g.world.spawn.x);
  assert.equal(g.playerT.y, g.world.spawn.y);

  const vehicles = g.ecs.query(Comp.Vehicle, Comp.Transform);
  assert.ok(vehicles.length > 0);
  const vt = g.ecs.get(vehicles[0]!, Comp.Transform)!;
  assert.equal(vt.x, g.world.spawn.x + 1.9);
  assert.equal(vt.y, g.world.spawn.y + 0.3);

  const pets = g.ecs.query(Comp.Pet, Comp.Transform);
  assert.ok(pets.length > 0);
  const pt = g.ecs.get(pets[0]!, Comp.Transform)!;
  assert.equal(pt.x, g.world.spawn.x - 1.1);
  assert.equal(pt.y, g.world.spawn.y + 0.8);
});

test('teleportToTownDistrict moves player to district anchors', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  g.teleportToTownDistrict('commercial');
  assert.equal(g.playerT.x, sp.x + 16);
  assert.equal(g.playerT.y, sp.y);

  g.teleportToTownDistrict('residential');
  assert.equal(g.playerT.x, sp.x);
  assert.equal(g.playerT.y, sp.y + 16);

  g.teleportToTownDistrict('industrial');
  assert.equal(g.playerT.x, sp.x - 16);
  assert.equal(g.playerT.y, sp.y);
});

test('every town has a main street and is linked to the highway network (HD plan G2)', () => {
  const g = new Game(DEFAULT_CONFIG);
  const spawn = g.world.spawn;
  const towns = [{ x: spawn.x, y: spawn.y }, ...REGIONAL_TOWNS];
  const all = roadPathsNear(DEFAULT_CONFIG.seed, spawn, -400, -400, 400, 400, 0);
  const highways = all.filter((p) => p.kind === 'highway');
  assert.ok(highways.length >= towns.length - 1, `a spanning set of highways, got ${highways.length}`);
  for (const t of towns) {
    assert.equal(getStreetDetails(DEFAULT_CONFIG.seed, t.x + 5.5, t.y, spawn).kind, 'highway', 'town main street');
    const linked = highways.some((p) => [p.pts[0]!, p.pts[p.pts.length - 1]!].some(([x, y]) => Math.hypot(x - t.x, y - t.y) < 70));
    assert.ok(linked, `town at ${t.x},${t.y} has a highway`);
  }
  assert.ok(all.some((p) => p.kind === 'dirt_road'), 'dirt tracks lead out of towns');
});

test('highways and dirt tracks are smooth curves, not staircases', () => {
  const g = new Game(DEFAULT_CONFIG);
  for (const p of roadPathsNear(DEFAULT_CONFIG.seed, g.world.spawn, -400, -400, 400, 400, 0)) {
    for (let i = 1; i < p.pts.length - 1; i++) {
      const [ax, ay] = p.pts[i - 1]!, [bx, by] = p.pts[i]!, [cx, cy] = p.pts[i + 1]!;
      const a1 = Math.atan2(by - ay, bx - ax), a2 = Math.atan2(cy - by, cx - bx);
      let turn = Math.abs(a2 - a1); if (turn > Math.PI) turn = 2 * Math.PI - turn;
      assert.ok(turn < 0.6, `turn of ${turn.toFixed(2)} rad between road samples`);
    }
  }
});

test('regional town centers spawn deterministic named town features', () => {
  const g = new Game(DEFAULT_CONFIG);

  let checked = 0;
  for (const town of REGIONAL_TOWNS) {
    const terrain = g.world.gen.sample(town.x, town.y).terrain;
    if (terrain === 'deep' || terrain === 'shallow') continue;

    const cx = Math.floor(town.x / 16);
    const cy = Math.floor(town.y / 16);
    const chunk = g.world.chunkAt(cx, cy);
    const feature = chunk.features.find((f) => f.type === 'town' && f.name === town.name);
    assert.ok(feature, `expected town feature for ${town.name} at chunk ${cx},${cy}`);
    checked++;
  }

  assert.ok(checked >= 4, 'should verify multiple regional towns on valid land tiles');
});

test('regional town streets include parked-car variety near curbs', () => {
  const g = new Game(DEFAULT_CONFIG);
  let parked = 0;

  for (const town of REGIONAL_TOWNS) {
    const terrain = g.world.gen.sample(town.x, town.y).terrain;
    if (terrain === 'deep' || terrain === 'shallow') continue;
    const objs = g.world.objectsNear(town.x, town.y, 36);
    for (const o of objs) {
      if (o.kind === 'abandoned_car' || o.kind === 'wrecked_car') parked++;
    }
  }

  assert.ok(parked >= 6, `expected multiple parked cars around regional towns, got ${parked}`);
});
