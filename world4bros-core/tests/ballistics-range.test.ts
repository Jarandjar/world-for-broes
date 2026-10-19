import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';
import { Comp } from '../src/sim/components';

test('defines distinct aim systems and ballistics curves for all weapons', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const items = game.content.items;
  
  // Slingshot: bloom cone, pebbles
  const sling = items.get('slingshot')?.gun;
  assert.ok(sling);
  assert.equal(sling!.aimSystem, 'bloom');
  assert.equal(sling!.ammoItem, 'pebbles');
  assert.equal(sling!.baseDamage, 8);

  // Hunting Rifle: laser sight, high speed, penetration
  const rifle = items.get('hunting_rifle')?.gun;
  assert.ok(rifle);
  assert.equal(rifle!.aimSystem, 'laser');
  assert.equal(rifle!.ammoItem, 'rifle_ammo');
  assert.ok(rifle!.projectileSpeed > 40);
  assert.equal(rifle!.baseDamage, 55);
  assert.equal(rifle!.penetration, 2);

  // Pump Shotgun: wide bloom, multi-pellet
  const shotgun = items.get('pump_shotgun')?.gun;
  assert.ok(shotgun);
  assert.equal(shotgun!.aimSystem, 'bloom');
  assert.equal(shotgun!.pelletCount, 6);
  assert.equal(shotgun!.ammoItem, 'shotgun_shells');

  // Nailgun SMG: burst diamond, fast firing
  const nailgun = items.get('nailgun_smg')?.gun;
  assert.ok(nailgun);
  assert.equal(nailgun!.aimSystem, 'burst');
  assert.equal(nailgun!.fireRate, 5.0);
  assert.equal(nailgun!.ammoItem, 'scrap_nails');

  // Potato Cannon: parabolic arc
  const cannon = items.get('potato_cannon')?.gun;
  assert.ok(cannon);
  assert.equal(cannon!.aimSystem, 'arc');
  assert.equal(cannon!.baseDamage, 32);
});

test('teleports to the shooting range with armory and target dummies', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToShootingRange();
  
  // Player inventory has all firearms and ammunition
  assert.ok(game.playerInv.inv.count('hunting_rifle') >= 1);
  assert.ok(game.playerInv.inv.count('rifle_ammo') >= 30);
  assert.ok(game.playerInv.inv.count('pump_shotgun') >= 1);
  assert.ok(game.playerInv.inv.count('shotgun_shells') >= 20);
  assert.ok(game.playerInv.inv.count('nailgun_smg') >= 1);
  assert.ok(game.playerInv.inv.count('scrap_nails') >= 50);
  assert.ok(game.playerInv.inv.count('slingshot') >= 1);
  assert.ok(game.playerInv.inv.count('pebbles') >= 30);

  // Target dummies exist at distance intervals
  const dummies = Array.from(game.ecs.query(Comp.TargetDummy));
  assert.ok(dummies.length >= 5);

  const labels = dummies.map((e) => game.ecs.get(e, Comp.TargetDummy)!.distanceLabel);
  assert.ok(labels.includes('5m'));
  assert.ok(labels.includes('10m'));
  assert.ok(labels.includes('15m'));
  assert.ok(labels.includes('25m'));
  assert.ok(labels.includes('40m'));
});

test('locks player aim orientation and enters aim stance when aimHeld is true', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToShootingRange();
  // Equip rifle in slot 0
  game.holdSlot(0);
  const held = game.playerChar.look.outfit.mainHand?.item;
  assert.equal(held, 'hunting_rifle');

  const aimWorld = { x: game.playerT.x + 10, y: game.playerT.y }; // Directly East (angle 0)
  
  // Step with aim stance held
  game.step(0.1, {
    move: { x: 0, y: 1 }, // Moving South while aiming East
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld,
    shootPressed: false,
  });

  assert.equal(game.aimActive, true);
  assert.ok(Math.abs(game.playerT.angle) < 0.1);
});

test('spawns projectiles, traverses space, and hits target dummies with distance damage', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToShootingRange();
  game.holdSlot(0); // hunting_rifle
  
  // Find the 5m target dummy
  let dummy5m = -1;
  for (const e of game.ecs.query(Comp.TargetDummy)) {
    if (game.ecs.get(e, Comp.TargetDummy)!.distanceLabel === '5m') {
      dummy5m = e;
      break;
    }
  }
  assert.ok(dummy5m >= 0);
  const dPos = game.ecs.get(dummy5m, Comp.Transform)!;

  // Aim directly at the 5m dummy and shoot
  const initialAmmo = game.playerInv.inv.count('rifle_ammo');
  game.step(0.016, {
    move: { x: 0, y: 0 },
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld: { x: dPos.x, y: dPos.y },
    shootPressed: true,
  });

  // 1 rifle ammo consumed
  assert.equal(game.playerInv.inv.count('rifle_ammo'), initialAmmo - 1);

  // Active projectile exists
  const projs = Array.from(game.ecs.query(Comp.Projectile));
  assert.ok(projs.length > 0);

  // Step simulation until projectile travels 5m and hits
  for (let i = 0; i < 30; i++) {
    game.step(0.016, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: true,
      aimWorld: { x: dPos.x, y: dPos.y },
      shootPressed: false,
    });
  }

  const dummyState = game.ecs.get(dummy5m, Comp.TargetDummy)!;
  assert.ok(dummyState.totalHits > 0);
  assert.ok(dummyState.lastDamageTaken > 40); // 5m is full base damage (55)
});

test('applies damage falloff over longer distances', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToShootingRange();
  game.holdSlot(0); // hunting_rifle
  
  // Find the 40m target dummy
  let dummy40m = -1;
  for (const e of game.ecs.query(Comp.TargetDummy)) {
    if (game.ecs.get(e, Comp.TargetDummy)!.distanceLabel === '40m') {
      dummy40m = e;
      break;
    }
  }
  assert.ok(dummy40m >= 0);
  const dPos = game.ecs.get(dummy40m, Comp.Transform)!;

  // Fire directly at 40m target
  game.step(0.016, {
    move: { x: 0, y: 0 },
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld: { x: dPos.x, y: dPos.y },
    shootPressed: true,
  });

  // Step to let bullet reach 40m
  for (let i = 0; i < 60; i++) {
    game.step(0.016, {
      move: { x: 0, y: 0 },
      sprint: false,
      actionPressed: false,
      interactPressed: false,
      aimHeld: false,
      shootPressed: false,
    });
  }

  const dummyState = game.ecs.get(dummy40m, Comp.TargetDummy)!;
  assert.ok(dummyState.totalHits > 0);
  // At 40m (past falloffStartDist 20m), damage should be falloff reduced below base damage 55
  assert.ok(dummyState.lastDamageTaken < 55);
  assert.ok(dummyState.lastDamageTaken >= 28); // minDamage
});

test('shotgun fires multi-pellet spread consuming 1 shotgun shell', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game.teleportToShootingRange();
  
  // Find which hotbar slot has pump_shotgun
  let shotgunSlot = -1;
  for (let s = 0; s < 8; s++) {
    if (game.playerInv.inv.slots[s]?.item === 'pump_shotgun') {
      shotgunSlot = s;
      break;
    }
  }
  assert.ok(shotgunSlot >= 0);
  game.holdSlot(shotgunSlot);

  const initialShells = game.playerInv.inv.count('shotgun_shells');
  const dPos = { x: game.playerT.x + 10, y: game.playerT.y };

  game.step(0.016, {
    move: { x: 0, y: 0 },
    sprint: false,
    actionPressed: false,
    interactPressed: false,
    aimHeld: true,
    aimWorld: dPos,
    shootPressed: true,
  });

  assert.equal(game.playerInv.inv.count('shotgun_shells'), initialShells - 1);
  const projs = Array.from(game.ecs.query(Comp.Projectile));
  assert.equal(projs.length, 6); // 6 buckshot pellets
});
