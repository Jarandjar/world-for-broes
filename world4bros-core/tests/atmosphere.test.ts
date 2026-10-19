import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ATMOS_EFFECTS, ATMOS_LAYERS, PARAM_KEYS, QUALITIES, WEATHERS, buildingLights, effectsFor, pickWeather, sunAt } from '../src/render/atmosphere';
import { PPU_TIERS, ppuTier } from '../src/render/sprites';
import { lodFor } from '../src/render/camera';
import type { Building } from '../src/sim/worldgen';

test('atmosphere effect registry contract', () => {
  const ids = ATMOS_EFFECTS.map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length, 'effect ids are unique');
  for (const e of ATMOS_EFFECTS) {
    assert.ok(ATMOS_LAYERS.includes(e.layer), `${e.id} has a known layer`);
    assert.ok((['low', 'high'] as const).includes(e.minQuality), `${e.id} has a quality tier`);
    assert.equal(typeof e.draw, 'function');
  }
  for (const l of ATMOS_LAYERS) assert.deepEqual(effectsFor(l, 'off'), [], 'off quality draws nothing');
  assert.ok(effectsFor('screen', 'high').length >= effectsFor('screen', 'low').length);
});

test('weather data is in range and the roll is deterministic and climate-aware', () => {
  const ids = WEATHERS.map((w) => w.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const w of WEATHERS) for (const k of PARAM_KEYS) assert.ok(w[k] >= 0 && w[k] <= 1, `${w.id}.${k} in 0..1`);
  const hot = { temp: 0.95, moist: 0.1 }, cold = { temp: 0.05, moist: 0.4 };
  for (let slot = 0; slot < 60; slot++) {
    assert.equal(pickWeather(slot, hot, 4242).id, pickWeather(slot, hot, 4242).id);
    assert.notEqual(pickWeather(slot, hot, 4242).id, 'snow', 'no snow in the desert');
    assert.notEqual(pickWeather(slot, cold, 4242).id, 'rain', 'cold climates snow instead of rain');
  }
});

test('sun shadows swing from west to east and stretch at low sun', () => {
  const morning = sunAt(0.02, 0), noon = sunAt(0.28, 0), evening = sunAt(0.55, 0.05);
  assert.ok(morning.dx < 0 && evening.dx > 0);
  assert.ok(morning.len > noon.len && evening.len > noon.len);
  assert.ok(sunAt(0.28, 0, 1).alpha < noon.alpha, 'clouds soften shadows');
});

test('night window lights are deterministic and capped', () => {
  const b: Building = { id: 'b1', def: 'cottage', x: 0, y: 0, w: 10, h: 8, door: 'S', doorAt: 5 };
  assert.equal(buildingLights([b], 0, 1).length, 0, 'no lights by day');
  const a = buildingLights([b], 0.9, 3), c = buildingLights([b], 0.9, 3);
  assert.deepEqual(a, c);
  assert.ok(a.length >= 1);
  assert.ok(buildingLights(Array.from({ length: 200 }, (_, i) => ({ ...b, id: `b${i}` })), 0.9, 1).length <= 48);
});

test('sprite resolution tiers and camera LOD', () => {
  assert.equal(ppuTier(300, 'high'), 288);
  assert.equal(ppuTier(300, 'low'), 144);
  assert.equal(ppuTier(40, 'high'), 56);
  for (const q of QUALITIES) assert.deepEqual([...PPU_TIERS[q]].sort((x, y) => x - y), [...PPU_TIERS[q]]);
  assert.equal(lodFor(8), 0); assert.equal(lodFor(54), 1); assert.equal(lodFor(150), 2);
});
