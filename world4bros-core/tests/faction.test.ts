import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('tri-zone risk topology returns green, yellow, and red zones according to distance', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sys = g.faction;

  assert.equal(sys.getZoneAt(0, 0, 0, 0), 'green');
  assert.equal(sys.getZoneAt(40, 40, 0, 0), 'green'); // ~56.5m < 65m
  assert.equal(sys.getZoneAt(100, 0, 0, 0), 'yellow'); // 100m (65 - 200m)
  assert.equal(sys.getZoneAt(140, 140, 0, 0), 'yellow'); // ~198m <= 200m
  assert.equal(sys.getZoneAt(210, 0, 0, 0), 'red'); // 210m > 200m
  assert.equal(sys.getZoneAt(300, 300, 0, 0), 'red'); // ~424m > 200m
});

test('clan_territory_totem Tesla shield activates and deactivates based on raid window', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('clan_territory_totem')!;
  const te = g.base.place(def, 10, 10, 'E');

  // Hour 12:00 (midday, outside 18:00 - 20:00 raid window)
  g.clock.t = 12 / 24;
  g.step(0.1, NO_INPUT);

  const totem = g.ecs.get(te, Comp.Totem)!;
  assert.ok(totem, 'totem should have Totem component');
  assert.equal(totem.shieldActive, true, 'Tesla shield should be active during protected hours');

  // Hour 19:00 (within active raid window 18:00 - 20:00)
  g.clock.t = 19 / 24;
  g.step(0.1, NO_INPUT);

  assert.equal(totem.shieldActive, false, 'Tesla shield should drop during active defense raid window');
});

test('peace treaty prevents shield shutdown during active raid window', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('clan_territory_totem')!;
  const te = g.base.place(def, 10, 10, 'E');

  // Sign 48-hour non-aggression pact
  g.faction.signPeaceTreaty(2);
  assert.equal(g.faction.treatyDaysLeft, 2);

  // Hour 19:00 (within active raid window)
  g.clock.t = 19 / 24;
  g.step(0.1, NO_INPUT);

  const totem = g.ecs.get(te, Comp.Totem)!;
  assert.equal(totem.shieldActive, true, 'Tesla shield should remain active during peace treaty');
});

test('active raid override forces shield shutdown', () => {
  const g = new Game(DEFAULT_CONFIG);
  const def = g.content.structures.get('clan_territory_totem')!;
  const te = g.base.place(def, 10, 10, 'E');

  // Outside normal raid window (10:00 AM)
  g.clock.t = 10 / 24;
  g.faction.activeRaidOverride = true;
  g.step(0.1, NO_INPUT);

  const totem = g.ecs.get(te, Comp.Totem)!;
  assert.equal(totem.shieldActive, false, 'active raid override should lower shields immediately');
});

test('C4 plunger breaching charge arms, counts down, and detonates dealing structural damage', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.playerInv.inv.add('c4_plunger_charge', 1);

  const wallDef = g.content.structures.get('wall_wood')!;
  const we = g.base.place(wallDef, 10, 10, 'E');
  const sc = g.ecs.get(we, Comp.Structure)!;
  const initialHp = sc.hp;

  // Move player away so player is not hurt
  g.playerT.x = 25;
  g.playerT.y = 25;

  const armed = g.faction.armC4(10.5, 10.5, g);
  assert.equal(armed, true, 'should successfully arm C4 with inventory charge');
  assert.equal(g.playerInv.inv.count('c4_plunger_charge'), 0, 'charge should be consumed from inventory');
  assert.equal(g.faction.activeC4Charges.length, 1);

  // Countdown timer is 4.0s. Advance 4.2s.
  for (let i = 0; i < 42; i++) g.step(0.1, NO_INPUT);

  assert.equal(g.faction.activeC4Charges.length, 0, 'charge should detonate and clear');
  assert.ok(sc.hp < initialHp || !g.ecs.has(we, Comp.Structure), 'target structure should sustain massive blast damage');
});

test('radio black market bounty advances on completion and grants credits and reward item', () => {
  const g = new Game(DEFAULT_CONFIG);
  const initialCredits = g.faction.credits;
  const initialAmmo = g.playerInv.inv.count('turret_ammo');

  // Advance zombie slay bounty (target is 3)
  g.faction.advanceBounty('slay_zombies', 3, g);

  const bounty = g.faction.bounties.find((b) => b.id === 'slay_zombies')!;
  assert.equal(bounty.completed, true);
  assert.equal(g.faction.credits, initialCredits + 75);
  assert.equal(g.playerInv.inv.count('turret_ammo'), initialAmmo + 1);
});

test('faction state persists across save and load', () => {
  const g = new Game(DEFAULT_CONFIG);
  g.faction.clanName = 'Apex Outlaws';
  g.faction.credits = 620;
  g.faction.treatyDaysLeft = 3;

  const save = g.toSave();
  assert.ok(save.faction, 'save data should include faction');

  const g2 = Game.fromSave(save);
  assert.equal(g2.faction.clanName, 'Apex Outlaws');
  assert.equal(g2.faction.credits, 620);
  assert.equal(g2.faction.treatyDaysLeft, 3);
});
