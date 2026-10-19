import { test } from 'node:test';
import assert from 'node:assert/strict';
import { devLog } from '../src/utils/dev-log';

test('DevLog: records entries, filters by level and category, and tracks bug count', () => {
  devLog.clear();
  assert.equal(devLog.getLogs().length, 0);
  assert.equal(devLog.getBugCount(), 0);

  devLog.log('system', 'System started successfully');
  devLog.combat('Fired hunting_rifle dealing 40 damage');
  devLog.warn('world', 'Entity out of normal bounds', { entityId: 42 });
  devLog.error('render', 'Failed to compile test shader', { shader: 'water' });
  devLog.bug('bot', 'Bot target unreachable', undefined, 'Error: unreachable\n  at botBrain.ts:40');

  const all = devLog.getLogs();
  assert.equal(all.length, 5);

  // Bug count should be 2 (1 error + 1 bug)
  assert.equal(devLog.getBugCount(), 2);

  // Filter by level
  const combatLogs = devLog.getLogs({ level: 'combat' });
  assert.equal(combatLogs.length, 1);
  assert.ok(combatLogs[0].message.includes('hunting_rifle'));

  // Filter by category
  const botLogs = devLog.getLogs({ category: 'bot' });
  assert.equal(botLogs.length, 1);
  assert.ok(botLogs[0].message.includes('Bot target unreachable'));

  // Search filter
  const searchResults = devLog.getLogs({ search: 'shader' });
  assert.equal(searchResults.length, 1);
  assert.equal(searchResults[0].category, 'render');
});

test('DevLog: generates complete diagnostic bug report snapshot and markdown formatting', () => {
  devLog.clear();
  devLog.combat('Player hit target dummy at 15m dealing 35 DMG');
  devLog.bug('system', 'Null pointer reference in collision checker', { x: 12, y: 15 }, 'Error: null at sim.ts:12');

  const snapshot = devLog.generateBugReport({
    seed: 9999,
    mode: 'zombie',
    playerPos: { x: 14.5, y: -2.3 },
    playerHealth: 85,
    activeArena: 'Duel Pit',
    entityCounts: { zombies: 4, bots: 1, projectiles: 2 },
    fps: 60,
    simTime: 12.4,
  });

  assert.equal(snapshot.gameState.seed, 9999);
  assert.equal(snapshot.gameState.mode, 'zombie');
  assert.equal(snapshot.gameState.playerHealth, 85);
  assert.equal(snapshot.gameState.activeArena, 'Duel Pit');
  assert.equal(snapshot.errorsCount, 1);
  assert.equal(snapshot.recentLogs.length, 2);

  const md = devLog.formatMarkdownReport(snapshot);
  assert.ok(md.includes('# 🐛 World4Bros Diagnostic Bug Report'));
  assert.ok(md.includes('Seed:** `9999`'));
  assert.ok(md.includes('Duel Pit'));
  assert.ok(md.includes('Null pointer reference in collision checker'));
  assert.ok(md.includes('## 💥 Stack Traces'));
});
