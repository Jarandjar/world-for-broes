import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VEHICLES, VEHICLE_LOOKS, DEFAULT_VEHICLE_LOOK, vehicleLook } from '../src/data/vehicles';

const HEX = /^#[0-9a-f]{6}$/i;
const DECALS = ['none', 'stripes', 'flames', 'number', 'stars', 'pinstripe', 'bolt', 'ducks'];

test('every vehicle has a paint job with valid colours and a known decal', () => {
  for (const v of VEHICLES) {
    assert.ok(VEHICLE_LOOKS[v.id], `${v.id} has no VEHICLE_LOOKS entry`);
    const look = vehicleLook(v.id);
    for (const k of ['trim', 'glass', 'accent', 'accent2', 'dirt', 'rustTone'] as const) assert.match(look[k], HEX, `${v.id}.${k}`);
    assert.ok(DECALS.includes(look.decal), `${v.id} decal ${look.decal}`);
    assert.ok(look.grime >= 0 && look.grime <= 1 && look.rust >= 0 && look.rust <= 1, `${v.id} grime/rust in 0..1`);
  }
});

test('looks are optional: unknown ids fall back to the default paint', () => {
  assert.deepEqual(vehicleLook('no_such_vehicle'), DEFAULT_VEHICLE_LOOK);
  for (const id of Object.keys(VEHICLE_LOOKS)) assert.ok(VEHICLES.some((v) => v.id === id), `look for unknown vehicle ${id}`);
});
