import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tinRoofTarp } from '../src/render/buildings';
import type { Building } from '../src/sim/worldgen';

const bld = (x: number, y: number): Building => ({ id: `t${x},${y}`, def: 'tool_shed', x, y, w: 5, h: 4, door: 'S', doorAt: 2.5 });

test('roof tarps are rare, deterministic and stay on the roof', () => {
  let with_ = 0, total = 0;
  for (let y = -60; y < 60; y += 3) for (let x = -60; x < 60; x += 3) {
    const b = bld(x, y), t = tinRoofTarp(b);
    total++;
    assert.deepEqual(t, tinRoofTarp(bld(x, y)), 'same building, same tarp');
    if (!t) continue;
    with_++;
    assert.ok(t.x >= 0 && t.y >= 0 && t.x + t.w <= b.w && t.y + t.h <= b.h, 'tarp fits on the roof');
  }
  const share = with_ / total;
  assert.ok(share > 0.08 && share < 0.26, `about 1 roof in 6 gets a tarp, got ${(share * 100).toFixed(1)}%`);
});
