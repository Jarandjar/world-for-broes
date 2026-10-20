import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';

test('biomes span large areas, not speckles (HD plan G1)', () => {
  const g = new Game({ ...DEFAULT_CONFIG, seed: 4242 });
  let runs = 0, total = 0;
  for (const row of [-600, 0, 600]) {
    let last = -1;
    for (let x = -1500; x <= 1500; x += 2) {
      const b = g.world.gen.sample(x, row).biome;
      if (b !== last) { runs++; last = b; }
      total++;
    }
  }
  const meanRunTiles = (total * 2) / runs;
  assert.ok(meanRunTiles >= 60, `average biome stretch along a line should be at least 60 tiles, got ${meanRunTiles.toFixed(1)}`);
});
