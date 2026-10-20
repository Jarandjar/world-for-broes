import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DUEL_PIT,
  FFA_YARD,
  checkDuelPitSymmetry,
  arenaBlocksBullet,
  arenaBlocksMovement,
  getArenaRectSolids,
  getArenaCircleSolids,
} from '../src/sim/arena';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('Z28a: Duel Pit passes half-turn symmetry test', () => {
  const result = checkDuelPitSymmetry(DUEL_PIT);
  assert.equal(result.symmetric, true, `Duel pit symmetry failed: ${result.errors.join(', ')}`);
  assert.equal(result.errors.length, 0);

  // Explicit check: Spawns (4,12) and (36,12) rotate into each other
  assert.equal(DUEL_PIT.spawns.length, 2);
  const s0 = DUEL_PIT.spawns[0];
  const s1 = DUEL_PIT.spawns[1];
  assert.equal(40 - s0.x, s1.x);
  assert.equal(24 - s0.y, s1.y);

  // Pies (20,3) and (20,21) rotate into each other
  assert.equal(DUEL_PIT.pies.length, 2);
  const p0 = DUEL_PIT.pies[0];
  const p1 = DUEL_PIT.pies[1];
  assert.equal(40 - p0.x, p1.x);
  assert.equal(24 - p0.y, p1.y);
});

test('Z28a: Bullets stop on crates, sandbags, and low walls', () => {
  // Crate at (17, 10, w=2, h=2) in Duel Pit
  // Shot passing straight through center: (15, 11) to (20, 11)
  const blockedByCrate = arenaBlocksBullet(DUEL_PIT, 15, 11, 20, 11);
  assert.equal(blockedByCrate, true, 'Bullet through crate center must be blocked');

  // Shot missing crate: (15, 8) to (20, 8)
  const clearShot = arenaBlocksBullet(DUEL_PIT, 15, 8, 20, 8);
  assert.equal(clearShot, false, 'Bullet through clear space must not be blocked');

  // Sandbag at (11, 6, length=5, direction=h)
  const blockedBySandbag = arenaBlocksBullet(DUEL_PIT, 13, 4, 13, 8);
  assert.equal(blockedBySandbag, true, 'Bullet crossing sandbag row must be blocked');

  // Low wall at (26, 9, w=1, h=3)
  const blockedByLowWall = arenaBlocksBullet(DUEL_PIT, 24, 10, 28, 10);
  assert.equal(blockedByLowWall, true, 'Bullet crossing low wall must be blocked');
});

test('Z28a: Barrels and trees block movement but do NOT stop bullets', () => {
  // Barrels in Duel Pit: (9, 12), (31, 12), (16, 3), (24, 21)
  // Movement into barrel at (9, 12)
  const walkIntoBarrel = arenaBlocksMovement(DUEL_PIT, 9, 12, 0.3);
  assert.equal(walkIntoBarrel, true, 'Movement into barrel must be blocked');

  // Bullet shot passing through barrel at (9, 12)
  // Section 3: "Barrels and trees block movement only"
  const bulletThroughBarrel = arenaBlocksBullet(DUEL_PIT, 7, 12, 11, 12);
  assert.equal(bulletThroughBarrel, false, 'Bullet through barrel must NOT be blocked');

  // Tree in Duel Pit at (3, 3)
  const walkIntoTree = arenaBlocksMovement(DUEL_PIT, 3, 3, 0.4);
  assert.equal(walkIntoTree, true, 'Movement into tree must be blocked');

  const bulletThroughTree = arenaBlocksBullet(DUEL_PIT, 1, 3, 5, 3);
  assert.equal(bulletThroughTree, false, 'Bullet through tree must NOT be blocked');
});

test('Z28a: Free-for-all Yard data matches specifications', () => {
  assert.equal(FFA_YARD.width, 64);
  assert.equal(FFA_YARD.height, 40);
  assert.equal(FFA_YARD.spawns.length, 8);
  assert.equal(FFA_YARD.pies.length, 4);
  assert.equal(FFA_YARD.crates.length, 9);
  assert.equal(FFA_YARD.sandbags.length, 8);
  assert.equal(FFA_YARD.lowWalls.length, 4);
  assert.equal(FFA_YARD.barrels.length, 8);
  assert.equal(FFA_YARD.trees.length, 8);
  assert.ok(FFA_YARD.roads && FFA_YARD.roads.length === 2);

  // Layout rule: no spawn sees another spawn within 12 m
  for (let i = 0; i < FFA_YARD.spawns.length; i++) {
    for (let j = i + 1; j < FFA_YARD.spawns.length; j++) {
      const d = Math.hypot(FFA_YARD.spawns[i].x - FFA_YARD.spawns[j].x, FFA_YARD.spawns[i].y - FFA_YARD.spawns[j].y);
      assert.ok(d >= 12, `Spawns ${i} and ${j} are too close: ${d}m < 12m`);
    }
  }
});

test('Z28a: In-game teleport to arena sets active arena, collision blocker, and stops live bullets on crates', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToArena('duel_pit', 0);

  assert.ok(game.activeArena);
  assert.equal(game.activeArena!.def.id, 'duel_pit');

  const ox = game.activeArena!.originX;
  const oy = game.activeArena!.originY;

  // Player spawned at spawn 0: (ox + 4, oy + 12)
  assert.equal(game.playerT.x, ox + 4);
  assert.equal(game.playerT.y, oy + 12);

  // World blocker is registered and blocks movement outside arena
  assert.equal(game.world.blocked(ox - 1, oy + 12, 0.3), true, 'Left arena perimeter must block movement');

  // World blocker blocks movement into crate at (17, 10, w=2, h=2) -> center at (ox + 18, oy + 11)
  assert.equal(game.world.blocked(ox + 18, oy + 11, 0.3), true, 'Walking into crate center must be blocked');

  // Spawn a target dummy behind the crate at (ox + 20, oy + 11)
  const dummyEnt = game.ecs.create();
  game.ecs.add(dummyEnt, Comp.Transform, { x: ox + 20, y: oy + 11, angle: 0 });
  game.ecs.add(dummyEnt, Comp.TargetDummy, {
    totalHits: 0,
    lastDamageTaken: 0,
    lastHitDistance: 0,
    lastHitTime: 0,
    wobbleTimer: 0,
    distanceLabel: 'behind_crate',
  });

  // Position player directly in front of the crate and shoot rifle through the crate toward the dummy
  game.playerT.x = ox + 15;
  game.playerT.y = oy + 11;
  game.holdSlot(0); // hunting_rifle

  const dummyBefore = game.ecs.get(dummyEnt, Comp.TargetDummy)!;
  assert.equal(dummyBefore.totalHits, 0);

  // Shoot directly at the dummy behind the crate
  game.step(0.016, {
    move: { x: 0, y: 0 },
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld: { x: ox + 20, y: oy + 11 },
    shootPressed: true,
  });

  // Step simulation for 30 frames
  for (let i = 0; i < 30; i++) {
    game.step(0.016, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
  }

  // The bullet hit the crate and was destroyed before reaching the dummy
  const dummyAfter = game.ecs.get(dummyEnt, Comp.TargetDummy)!;
  assert.equal(dummyAfter.totalHits, 0, 'Bullet must NOT penetrate crate to hit target dummy');
});

test('Army Base Arena: Arena is relocated directly inside the Army Base complex', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToArena('duel_pit', 0);

  const rx = game.world.spawn.x - 24;
  const ry = game.world.spawn.y - 12;

  assert.equal(game.activeArena!.originX, rx + 6, 'Arena originX must match Army Base rx + 6');
  assert.equal(game.activeArena!.originY, ry + 20, 'Arena originY must match Army Base ry + 20');
});

test('Army Base Arena: Spawning bots, zombie swarms, bot combat AI and bullet damage', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToArena('duel_pit', 0);

  // 1. Spawning arena bot with regular tier
  const botEnt = game.spawnArenaBot('regular', 'nailgun_smg', 1)!;
  assert.ok(botEnt !== null);
  const botRaider = game.ecs.get(botEnt, Comp.Raider)!;
  assert.equal(botRaider.level, 'regular');
  assert.equal(botRaider.weapon, 'nailgun_smg');
  assert.equal(botRaider.hp, 65);
  assert.equal(botRaider.team, 'bot');

  // 2. Spawning zombie swarm into arena
  game.spawnArenaZombies(4, 'runner');
  const zombies = [...game.ecs.query(Comp.Zombie)];
  assert.ok(zombies.length >= 4, 'At least 4 zombies spawned in arena');

  // 3. Clear combatants removes bots and arena zombies
  game.clearArenaCombatants();
  assert.equal([...game.ecs.query(Comp.Raider)].length, 0, 'All raider bots cleared');

  // 4. Test live bot combat AI: Spawn a menace bot directly facing player with hunting_rifle
  const ox = game.activeArena!.originX;
  const oy = game.activeArena!.originY;
  const testBot = game.spawnArenaBot('menace', 'hunting_rifle', 1)!;
  const bT = game.ecs.get(testBot, Comp.Transform)!;
  const rComp = game.ecs.get(testBot, Comp.Raider)!;
  rComp.reactionTimer = 0;
  rComp.cooldown = 0;

  // Position player in line of sight of bot (no crates in direct path)
  game.playerT.x = bT.x - 6;
  game.playerT.y = bT.y;
  const playerInitialHp = game.playerNeeds.health;

  // Step simulation - bot AI should detect player, aim, and shoot live projectile
  let projectileFired = false;
  for (let i = 0; i < 40; i++) {
    game.step(0.02, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
    if ([...game.ecs.query(Comp.Projectile)].length > 0) {
      projectileFired = true;
    }
  }
  assert.ok(projectileFired, 'Bot AI must fire projectile bullets when player is in line of sight');

  // 5. Test shooting a bot: player shoots bot directly
  game.playerT.x = bT.x - 3;
  game.playerT.y = bT.y;
  game.holdSlot(0); // hunting_rifle (40 base dmg)
  const initialBotHp = rComp.hp;

  game.step(0.016, {
    move: { x: 0, y: 0 },
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld: { x: bT.x, y: bT.y },
    shootPressed: true,
  });

  // Advance projectile simulation until bullet hits
  for (let i = 0; i < 30; i++) {
    game.step(0.016, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
  }

  assert.ok(rComp.hp < initialBotHp, `Bot must take bullet damage: ${rComp.hp} < ${initialBotHp}`);
});

test('Army Base Arena: Bots shoot each other and zombies swarm bots', () => {
  const game = new Game(DEFAULT_CONFIG);
  game.teleportToArena('duel_pit', 0);
  game.clearArenaCombatants();

  // Move player far outside arena view so bots only engage each other
  game.playerT.x = 999;
  game.playerT.y = 999;

  // Spawn Bot A at spawn 0 and Bot B at spawn 1
  const botA = game.spawnArenaBot('sharp', 'hunting_rifle', 0)!;
  const botB = game.spawnArenaBot('sharp', 'hunting_rifle', 1)!;

  const tA = game.ecs.get(botA, Comp.Transform)!;
  const tB = game.ecs.get(botB, Comp.Transform)!;
  const rA = game.ecs.get(botA, Comp.Raider)!;
  const rB = game.ecs.get(botB, Comp.Raider)!;

  // Position bots within sight (10m apart) along a clear lane
  const ar = game.activeArena!;
  tA.x = ar.originX + 8;
  tA.y = ar.originY + 5;
  tB.x = ar.originX + 16;
  tB.y = ar.originY + 5;

  // Clear reaction delays
  rA.reactionTimer = 0;
  rA.cooldown = 0;
  rB.reactionTimer = 0;
  rB.cooldown = 0;

  // Step simulation: bots should target each other and fire projectiles
  let botFired = false;
  for (let i = 0; i < 40; i++) {
    game.step(0.02, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
    if ([...game.ecs.query(Comp.Projectile)].length > 0) {
      botFired = true;
    }
  }
  assert.ok(botFired, 'Bots must engage and fire projectiles at each other in the arena');

  // Spawn a fresh bot and a zombie near it
  const botC = game.spawnArenaBot('regular', 'nailgun_smg', 0)!;
  const tC = game.ecs.get(botC, Comp.Transform)!;
  const rC = game.ecs.get(botC, Comp.Raider)!;
  tC.x = ar.originX + 20;
  tC.y = ar.originY + 12;
  const initialHpC = rC.hp;

  const ze = game.spawnZombie(tC.x + 0.8, tC.y, 'runner');
  const z = game.ecs.get(ze, Comp.Zombie)!;
  z.state = 'chase';
  z.target = botC;
  z.attackTimer = 0;

  // Step simulation: zombie should attack Bot C and reduce Bot C HP
  for (let i = 0; i < 20; i++) {
    game.step(0.02, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
  }

  assert.ok(rC.hp < initialHpC, `Zombie horde must attack and damage bots: ${rC.hp} < ${initialHpC}`);
});

