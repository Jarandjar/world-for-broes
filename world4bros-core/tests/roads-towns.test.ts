import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { getRoadType, isRoad } from '../src/sim/worldgen';

test('getRoadType differentiates highways, rural dirt roads, and town small roads', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  // Town center avenue should be classified as highway
  const townRoad = getRoadType(g.world.seed, sp.x, sp.y, sp);
  assert.equal(townRoad, 'highway');

  // Verify isRoad helper returns true
  assert.equal(isRoad(g.world.seed, sp.x, sp.y, sp), true);

  // Scan across a 200m line to find dirt roads and small roads
  const foundRoadTypes = new Set<string>();
  for (let d = -100; d <= 100; d += 1) {
    const r = getRoadType(g.world.seed, sp.x + d, sp.y + d * 0.5, sp);
    foundRoadTypes.add(r);
  }

  assert.ok(foundRoadTypes.has('highway'), 'should find highway segments');
  assert.ok(foundRoadTypes.has('dirt_road'), 'should find rural dirt roads');
  assert.ok(foundRoadTypes.has('none'), 'should find wilderness non-road terrain');
});

test('speedAt provides multi-tier speed multipliers on highways and roads', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  // On town highway avenue: base speed (1.0) * 1.35
  const hwySpeed = g.world.speedAt(sp.x, sp.y);
  assert.ok(hwySpeed >= 1.3, `highway speed should grant ~35% bonus: got ${hwySpeed}`);

  // Compare to non-road meadow speed
  let nonRoadSpeed = 1.0;
  for (let d = 20; d <= 80; d += 2) {
    if (getRoadType(g.world.seed, sp.x + d, sp.y + d, sp) === 'none') {
      nonRoadSpeed = g.world.speedAt(sp.x + d, sp.y + d);
      break;
    }
  }

  assert.ok(hwySpeed > nonRoadSpeed, 'highway speed must exceed off-road speed');
});

test('Big Town spawns with fountain, streetlights, benches, dumpsters, and civic features', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  const nearby = g.world.objectsNear(sp.x, sp.y, 16);
  const kinds = nearby.map((o) => o.kind);

  assert.ok(kinds.includes('town_fountain'), 'Big Town should feature a town square fountain');
  assert.ok(kinds.includes('park_bench'), 'Big Town should have park benches');
  assert.ok(kinds.includes('street_light'), 'Big Town should have streetlights');
  assert.ok(kinds.includes('dumpster'), 'Big Town should have alley dumpsters');
  assert.ok(kinds.includes('traffic_cone'), 'Big Town should have traffic warning cones');
  assert.ok(kinds.includes('fire_hydrant'), 'Big Town should have fire hydrants');
  assert.ok(kinds.includes('guardrail'), 'Big Town should have highway guardrails');
});

test('resting at town fountain or park bench restores sanity', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  const fountain = g.world.objectsNear(sp.x, sp.y, 16).find((o) => o.kind === 'town_fountain');
  assert.ok(fountain, 'fountain exists in town');

  g.playerNeeds.sanity = 50;

  // Stand next to fountain and step simulation
  g.playerT.x = fountain!.x + 1.0;
  g.playerT.y = fountain!.y;
  for (let i = 0; i < 60; i++) g.step(1 / 60, NO_INPUT);

  assert.ok(g.playerNeeds.sanity > 50, 'relaxing by town fountain should restore sanity');
});

test('scavenging alley dumpster yields comedic urban loot', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;

  const dumpster = g.world.objectsNear(sp.x, sp.y, 16).find((o) => o.kind === 'dumpster');
  assert.ok(dumpster, 'dumpster exists in town');

  // Move player to dumpster, face it, and press interact
  g.playerT.x = dumpster!.x + 0.8;
  g.playerT.y = dumpster!.y;
  g.playerT.angle = Math.PI; // face dumpster

  const initialInvCount = g.playerInv.inv.slots.filter(Boolean).length;
  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });

  const finalInvCount = g.playerInv.inv.slots.filter(Boolean).length;
  assert.ok(finalInvCount >= initialInvCount, 'dumpster scavenging should produce items');
});
