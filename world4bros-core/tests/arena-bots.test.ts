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
    assert.equal(r.team, 'bot');
  }
});

test('spawned ally bots use same level stats, carry weapon, and are marked ally team', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const e = g.spawnArenaAllyBot('sharp', 'hunting_rifle', 0);
  assert.ok(e !== null);

  const r = g.ecs.get(e!, Comp.Raider)!;
  const ch = g.ecs.get(e!, Comp.Character)!;
  const row = ARENA_BOT_LEVELS.sharp;

  assert.equal(r.team, 'ally');
  assert.equal(r.weapon, 'hunting_rifle');
  assert.equal(r.reactionTimer, row.reaction);
  assert.equal(r.aimSpread, row.aimSpread);
  assert.equal(r.shotInterval, row.shotInterval);
  assert.equal(r.hp, row.hp);
  assert.equal(ch.look.outfit.mainHand?.item, 'hunting_rifle');
});

test('arena settings toggle APIs flip and report values', () => {
  const g = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const defaults = g.getArenaSettings();

  assert.equal(defaults.ammoConsumption, true);
  assert.equal(defaults.botsTargetPlayer, true);
  assert.equal(defaults.allyFriendlyFire, false);

  const ammo = g.toggleArenaSetting('ammoConsumption');
  const target = g.toggleArenaSetting('botsTargetPlayer');
  const ff = g.toggleArenaSetting('allyFriendlyFire');

  assert.equal(ammo, false);
  assert.equal(target, false);
  assert.equal(ff, true);
  assert.equal(g.getArenaSettings().ammoConsumption, false);
  assert.equal(g.getArenaSettings().botsTargetPlayer, false);
  assert.equal(g.getArenaSettings().allyFriendlyFire, true);

  g.setArenaSetting('ammoConsumption', true);
  assert.equal(g.getArenaSettings().ammoConsumption, true);
});
