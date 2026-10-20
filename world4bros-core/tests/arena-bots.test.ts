import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ARENA_BOT_LEVELS, ARENA_BOT_LEVEL_IDS } from '../src/data/arenaBots';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('arena bot levels get harder in order (ARENA.md section 5)', () => {
  const rows = ARENA_BOT_LEVEL_IDS.map((id) => ARENA_BOT_LEVELS[id]);
  for (let i = 1; i < rows.length; i++) {
    const easier = rows[i - 1]!, harder = rows[i]!;
    assert.ok(harder.reaction < easier.reaction, 'reacts faster');
    assert.ok(harder.aimSpread < easier.aimSpread, 'misses less');
    assert.ok(harder.shotInterval < easier.shotInterval, 'shoots more often');
    assert.ok(harder.sight >= easier.sight, 'sees at least as far');
  }
});

test('spawned arena bots take their numbers from the data table', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  for (const level of ARENA_BOT_LEVEL_IDS) {
    const e = g.spawnArenaBot(level, 'nailgun_smg', 0);
    assert.ok(e !== null);
    const r = g.ecs.get(e!, Comp.Raider)!;
    const row = ARENA_BOT_LEVELS[level];
    assert.equal(r.reactionTimer, row.reaction);
    assert.equal(r.aimSpread, row.aimSpread);
    assert.equal(r.shotInterval, row.shotInterval);
    assert.equal(r.hp, row.hp);
  }
});
