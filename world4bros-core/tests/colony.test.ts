import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../src/sim/game';
import { Comp } from '../src/sim/components';
import { createRandomPawn, ColonySystem } from '../src/sim/colony';
import { makeRng } from '../src/engine/rng';

test('pawn generation generates valid skills, passions, traits and priorities', () => {
  const rng = makeRng(12345);
  const pawn = createRandomPawn(rng);

  assert.ok(pawn.traits.length >= 1, 'Pawn should have at least 1 trait');
  assert.equal(typeof pawn.mood, 'number');
  assert.ok(pawn.mood >= 0 && pawn.mood <= 100);

  // Check 8 skills
  const expectedSkills = ['construction', 'mining', 'crafting', 'cooking', 'medicine', 'plants', 'combat', 'hauling'] as const;
  for (const sk of expectedSkills) {
    assert.ok(pawn.skills[sk], `Skill ${sk} should exist`);
    assert.ok(pawn.skills[sk].level >= 0 && pawn.skills[sk].level <= 20);
    assert.ok(['none', 'single', 'burning'].includes(pawn.skills[sk].passion));
  }

  // Check 9 work priorities
  const expectedRoles = ['firefight', 'patient', 'doctor', 'build', 'farm', 'mine', 'haul', 'craft', 'guard'] as const;
  for (const role of expectedRoles) {
    assert.ok(pawn.workPriorities[role] >= 0 && pawn.workPriorities[role] <= 4);
  }
});

test('environmental comfort objects grant psychological thoughts and shift mood', () => {
  const game = new Game({ seed: 777, mode: 'classic', creative: true, darknessDamage: false });
  const p = game.playerT;

  // Place a grand banquet table, campfire, and horseshoe post near colonist
  const colonist = game.spawnColonist(p.x + 1, p.y + 1, 'Barnaby');
  const pawn = game.ecs.get(colonist, Comp.Pawn)!;

  // Place campfire
  const fireDef = game.content.structures.get('campfire')!;
  game.base.place(fireDef, Math.floor(p.x / 2) * 2, Math.floor(p.y / 2) * 2);

  // Place banquet table
  const tableDef = game.content.structures.get('grand_banquet_table')!;
  game.base.place(tableDef, Math.floor(p.x / 2) * 2 + 2, Math.floor(p.y / 2) * 2);

  // Place horseshoe post
  const horseshoeDef = game.content.structures.get('horseshoe_pin_post')!;
  game.base.place(horseshoeDef, Math.floor(p.x / 2) * 2 - 2, Math.floor(p.y / 2) * 2);

  // Run game step
  game.step(1.0);

  // Check that positive thoughts were added
  const hasTableThought = pawn.thoughts.some((th) => th.id === 'ate_table');
  const hasWarmthThought = pawn.thoughts.some((th) => th.id === 'warm_fire');
  const hasHorseshoeThought = pawn.thoughts.some((th) => th.id === 'horseshoes');

  assert.ok(hasTableThought, 'Barnaby should gain Ate at Dining Table thought');
  assert.ok(hasWarmthThought, 'Barnaby should gain Comfortable Warmth thought');
  assert.ok(hasHorseshoeThought, 'Barnaby should gain Horseshoe Toss recreation thought');
});

test('slapstick mental breaks trigger on severe low mood and resolve with catharsis', () => {
  const game = new Game({ seed: 888, mode: 'classic', creative: true, darknessDamage: false });
  const colonist = game.spawnColonist(game.playerT.x + 2, game.playerT.y + 2, 'Grumble');
  const pawn = game.ecs.get(colonist, Comp.Pawn)!;

  // Force mood to near zero and clear positive thoughts
  pawn.mood = 5;
  pawn.thoughts = [{ id: 'ate_without_table', label: 'Ate without table (-12)', delta: -12, duration: 60 }];

  // Advance simulation to trigger mental break
  game.step(1.0);

  assert.notEqual(pawn.mentalBreak.kind, 'none', 'Pawn should be experiencing a mental break');
  assert.ok(pawn.mentalBreak.timer > 0, 'Mental break timer should be running');

  // Advance simulation until mental break timer expires (game.step clamps dt to 0.1)
  const steps = Math.ceil((pawn.mentalBreak.timer + 1) / 0.1);
  for (let s = 0; s < steps; s++) {
    game.step(0.1);
  }

  assert.equal(pawn.mentalBreak.kind, 'none', 'Mental break should have cleared');
  assert.ok(pawn.thoughts.some((th) => th.id === 'catharsis'), 'Pawn should have Catharsis (+20) thought');
});

test('1-4 priority work matrix directs pawns to highest priority active jobs', () => {
  const game = new Game({ seed: 999, mode: 'classic', creative: true, darknessDamage: false });
  const colonist = game.spawnColonist(game.playerT.x + 1, game.playerT.y + 1, 'WorkerBro');
  const pawn = game.ecs.get(colonist, Comp.Pawn)!;

  // Disable all jobs except mine
  for (const k of Object.keys(pawn.workPriorities) as (keyof typeof pawn.workPriorities)[]) {
    pawn.workPriorities[k] = 0;
  }
  pawn.workPriorities.mine = 1;

  // Ensure high mood so no mental break interferes
  pawn.mood = 90;

  // Step game
  game.step(1.0);

  // Since mineral nodes generate in world chunks, worker should pick up a mine job or wander if none within 25m
  if (pawn.currentJob) {
    assert.equal(pawn.currentJob.role, 'mine', 'Assigned job should match priority 1 role');
  }

  // Now change priority: set craft to 1, mine to 4
  pawn.workPriorities.craft = 1;
  pawn.workPriorities.mine = 4;
  pawn.currentJob = null;

  // Place a stone furnace nearby
  const furnaceDef = game.content.structures.get('stone_furnace')!;
  game.base.place(furnaceDef, Math.floor(game.playerT.x / 2) * 2, Math.floor(game.playerT.y / 2) * 2);

  game.step(1.0);
  const updatedPawn = game.ecs.get(colonist, Comp.Pawn)!;
  assert.ok(updatedPawn.currentJob, 'Pawn should find a job');
  assert.equal(updatedPawn.currentJob!.role, 'craft', 'Pawn should pick priority 1 craft job over priority 4 mine job');
});

test('storyteller AI triggers incidents and switches director personas', () => {
  const game = new Game({ seed: 101, mode: 'zombie', creative: true, darknessDamage: false });
  const colony = game.colony;

  assert.equal(colony.storyteller, 'caspar_chill');
  colony.storyteller = 'randy_ruckus';
  assert.equal(colony.storyteller, 'randy_ruckus');

  // Trigger wanderer_joins incident
  const initialPawns = game.ecs.query(Comp.Pawn).length;
  colony.triggerIncident('wanderer_joins', game);
  const afterPawns = game.ecs.query(Comp.Pawn).length;
  assert.equal(afterPawns, initialPawns + 1, 'Wanderer incident should spawn a new colonist entity');

  // Trigger supply drop incident
  const initialMetal = game.playerInv.inv.count('scrap_metal');
  colony.triggerIncident('supply_drop', game);
  const afterMetal = game.playerInv.inv.count('scrap_metal');
  assert.ok(afterMetal > initialMetal, 'Supply drop should deliver metal supplies');

  // Trigger screamer swarm incident
  const initialZombies = game.ecs.query(Comp.Zombie).length;
  colony.triggerIncident('screamer_swarm', game);
  const afterZombies = game.ecs.query(Comp.Zombie).length;
  assert.equal(afterZombies, initialZombies + 2, 'Screamer swarm should spawn 2 runner zombies');
});

test('colonists and colony storyteller state persist across save and load', () => {
  const game = new Game({ seed: 333, mode: 'classic', creative: true, darknessDamage: false });
  game.colony.storyteller = 'hilda_hardcore';
  game.colony.incidentTimer = 42;

  const colonist = game.spawnColonist(game.playerT.x + 3, game.playerT.y + 4, 'TestColonist');
  const pawn = game.ecs.get(colonist, Comp.Pawn)!;
  pawn.mood = 67;
  pawn.workPriorities.guard = 1;
  pawn.workPriorities.mine = 2;
  pawn.traits = ['gourmand', 'night_owl'];

  // Save game
  const saved = game.toSave();
  assert.ok(saved.colonists && saved.colonists.length > 0, 'Saved data should contain colonists array');
  assert.ok(saved.colony, 'Saved data should contain colony state');

  // Load from save
  const loaded = Game.fromSave(saved);
  assert.equal(loaded.colony.storyteller, 'hilda_hardcore', 'Storyteller director should match');
  assert.equal(loaded.colony.incidentTimer, 42, 'Incident timer should match');

  const loadedPawns = loaded.ecs.query(Comp.Pawn);
  assert.ok(loadedPawns.length >= 1, 'Loaded game should contain colonists');

  const loadedColonist = loadedPawns.find((e) => loaded.ecs.get(e, Comp.Character)?.name === 'TestColonist');
  assert.ok(loadedColonist, 'TestColonist should exist in loaded game');

  const loadedP = loaded.ecs.get(loadedColonist!, Comp.Pawn)!;
  assert.equal(loadedP.mood, 67, 'Mood should persist');
  assert.equal(loadedP.workPriorities.guard, 1, 'Priority work matrix should persist');
  assert.deepEqual(loadedP.traits, ['gourmand', 'night_owl'], 'Traits should persist');
});
