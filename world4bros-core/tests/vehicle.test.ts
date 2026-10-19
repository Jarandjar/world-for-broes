import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('vehicle spawns near spawn with full specs and trunk', () => {
  const g = new Game(DEFAULT_CONFIG);
  const vehicles = g.ecs.query(Comp.Vehicle);
  assert.equal(vehicles.length, 1);
  const v = g.ecs.get(vehicles[0]!, Comp.Vehicle)!;
  assert.equal(v.def.id, 'rusty_hatchback');
  assert.equal(v.hp, 100);
  assert.equal(v.fuel, 80);
  assert.equal(v.trunk.slots.length, 12);
  assert.equal(v.driver, null);
});

test('player can enter, drive, and exit the vehicle', () => {
  const g = new Game(DEFAULT_CONFIG);
  const veh = g.ecs.query(Comp.Vehicle)[0]!;
  const vt = g.ecs.get(veh, Comp.Transform)!;
  const pt = g.playerT;

  // Move player near driver door of vehicle and face it
  pt.x = vt.x;
  pt.y = vt.y + 0.8;
  pt.angle = Math.atan2(vt.y - pt.y, vt.x - pt.x);

  const target = g.findTarget();
  assert.ok(target);
  assert.equal(target?.kind, 'vehicle');
  if (target?.kind === 'vehicle') {
    assert.equal(target.part, 'cabin');
    assert.match(target.prompt, /Drive/);
  }

  // Press E to enter
  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });
  assert.equal(g.drivingVehicle(), veh);

  // Drive forward: press move up (y: -1) / W
  const startX = vt.x;
  for (let i = 0; i < 30; i++) {
    g.step(1 / 60, { ...NO_INPUT, move: { x: 1, y: 0 } });
  }

  const v = g.ecs.get(veh, Comp.Vehicle)!;
  assert.ok(v.speed > 1.0, 'vehicle should accelerate forward');
  assert.ok(vt.x > startX, 'vehicle position should advance');
  assert.equal(pt.x, vt.x, 'player position should follow vehicle');

  // Target when driving should be Exit Vehicle
  const driveTarget = g.findTarget();
  assert.equal(driveTarget?.kind, 'vehicle');
  if (driveTarget?.kind === 'vehicle') {
    assert.equal(driveTarget.part, 'exit');
  }

  // Press E to exit
  g.step(1 / 60, { ...NO_INPUT, interactPressed: true });
  assert.equal(g.drivingVehicle(), null);
  assert.equal(v.driver, null);
});

test('vehicle can be refueled and repaired', () => {
  const g = new Game(DEFAULT_CONFIG);
  const veh = g.ecs.query(Comp.Vehicle)[0]!;
  const v = g.ecs.get(veh, Comp.Vehicle)!;

  v.fuel = 30;
  v.hp = 50;

  // Refuel requires fuel_canister in pack
  g.playerInv.inv.add('fuel_canister', 1);
  assert.ok(g.refuelVehicle(veh));
  assert.equal(v.fuel, 70);
  assert.equal(g.playerInv.inv.count('fuel_canister'), 0);

  // Repair requires scrap_metal in pack
  g.playerInv.inv.add('scrap_metal', 2);
  assert.ok(g.repairVehicle(veh));
  assert.equal(v.hp, 75);
});

test('vehicle trunk can store items and persist across save/load', () => {
  const g = new Game(DEFAULT_CONFIG);
  const veh = g.ecs.query(Comp.Vehicle)[0]!;
  const v = g.ecs.get(veh, Comp.Vehicle)!;
  const vt = g.ecs.get(veh, Comp.Transform)!;

  v.trunk.add('scrap_metal', 8);
  v.trunk.add('wood', 15);
  v.fuel = 65;
  v.hp = 88;
  vt.x = 22.5;
  vt.y = -14.2;
  vt.angle = 1.35;

  const json = JSON.stringify(g.toSave());
  const g2 = Game.fromSave(JSON.parse(json));

  const veh2 = g2.ecs.query(Comp.Vehicle)[0]!;
  const v2 = g2.ecs.get(veh2, Comp.Vehicle)!;
  const vt2 = g2.ecs.get(veh2, Comp.Transform)!;

  assert.equal(v2.trunk.count('scrap_metal'), 8);
  assert.equal(v2.trunk.count('wood'), 15);
  assert.equal(v2.fuel, 65);
  assert.equal(v2.hp, 88);
  assert.equal(vt2.x, 22.5);
  assert.equal(vt2.y, -14.2);
});

test('all 7 vehicle fleet types can be spawned and have valid distinct specs', () => {
  const g = new Game(DEFAULT_CONFIG);
  const fleetIds = ['rusty_hatchback', 'clown_car', 'ice_cream_truck', 'golf_cart', 'bathtub_buggy', 'hot_dog_van', 'tesla_roadster'];

  for (const id of fleetIds) {
    const e = g.spawnVehicle(g.playerT.x + 8, g.playerT.y, id);
    const v = g.ecs.get(e, Comp.Vehicle)!;
    assert.equal(v.def.id, id);
    assert.ok(v.def.maxSpeed > 0, `${id} should have positive maxSpeed`);
    assert.ok(v.def.maxHp > 0, `${id} should have positive maxHp`);
    assert.ok(v.def.trunkSlots > 0, `${id} should have trunk capacity`);
    assert.ok(v.def.color.startsWith('#'), `${id} should have hex color`);
  }
});

test('spawnFleet populates the fleet around town plaza', () => {
  const g = new Game(DEFAULT_CONFIG);
  const initialCount = g.ecs.query(Comp.Vehicle).length;
  assert.equal(initialCount, 1);

  const spawned = g.spawnFleet();
  assert.equal(spawned.length, 6);

  const total = g.ecs.query(Comp.Vehicle).length;
  assert.equal(total, 7);
});

