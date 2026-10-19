import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';
import { STRUCTURE_DRAWERS } from '../src/render/structures';

test('registers 20+ homemade and visual attachments in content', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const atts = game.content.attachments;
  assert.ok(atts.size >= 20, `Expected at least 20 attachments, got ${atts.size}`);

  // Optics
  assert.ok(atts.has('optic_rangefinder'));
  assert.ok(atts.has('optic_camcorder'));
  assert.ok(atts.has('optic_googly'));
  assert.ok(atts.has('optic_toilet_roll'));

  // Muzzles
  assert.ok(atts.has('muzzle_potato_silencer'));
  assert.ok(atts.has('muzzle_soup_can'));
  assert.ok(atts.has('muzzle_car_horn'));
  assert.ok(atts.has('muzzle_kazoo'));

  // Underbarrels
  assert.ok(atts.has('under_cat_laser'));
  assert.ok(atts.has('under_plunger_bayonet'));
  assert.ok(atts.has('under_kitchen_fork'));
  assert.ok(atts.has('under_flashlight'));

  // Magazines
  assert.ok(atts.has('mag_twin_soda'));
  assert.ok(atts.has('mag_rubber_band'));

  // Goofy charms
  assert.ok(atts.has('goofy_rubber_duck'));
  assert.ok(atts.has('goofy_fuzzy_dice'));
  assert.ok(atts.has('goofy_whoopee'));
  assert.ok(atts.has('goofy_windup_clock'));
});

test('dynamically computes effective weapon stats and visual info flags', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const baseRifle = game.content.items.get('hunting_rifle')?.gun;
  assert.ok(baseRifle);

  // Reset to stock
  game.setWeaponAttachment('hunting_rifle', 'optic', null);
  const stock = game.getEffectiveGun('hunting_rifle');
  assert.equal(stock.baseDamage, baseRifle!.baseDamage);
  assert.equal(stock.maxRange, baseRifle!.maxRange);
  assert.equal(stock.visualInfo?.rangefinder, undefined);
  assert.equal(stock.visualInfo?.threatScanner, undefined);

  // Equip Tactical Rangefinder HUD (rangeMod: 14)
  game.setWeaponAttachment('hunting_rifle', 'optic', 'optic_rangefinder');
  const modded1 = game.getEffectiveGun('hunting_rifle');
  assert.equal(modded1.visualInfo?.rangefinder, true);
  assert.equal(modded1.visualInfo?.targetHealthBar, true);
  assert.equal(modded1.maxRange, baseRifle!.maxRange + 14);

  // Equip Russet Potato Silencer
  game.setWeaponAttachment('hunting_rifle', 'muzzle', 'muzzle_potato_silencer');
  const modded2 = game.getEffectiveGun('hunting_rifle');
  assert.ok(modded2.noiseRadius < baseRifle!.noiseRadius * 0.25, 'Potato silencer should dampen noise by ~82%');
  assert.ok(modded2.spreadAngle > modded1.spreadAngle, 'Potato silencer has slight barrel friction increasing spread');

  // Equip Lucky Fuzzy Dice and Plunger Bayonet
  game.setWeaponAttachment('hunting_rifle', 'goofy', 'goofy_fuzzy_dice');
  game.setWeaponAttachment('hunting_rifle', 'underbarrel', 'under_plunger_bayonet');
  const modded3 = game.getEffectiveGun('hunting_rifle');
  assert.equal(modded3.baseDamage, baseRifle!.baseDamage + 10 - 2, 'Fuzzy dice (+10) and Potato (-2)');
  assert.equal(modded3.meleeDmg, 38, 'Plunger bayonet should grant 38 melee damage');

  // Equip CRT Handycam Night Sight for threat scanner and night vision
  game.setWeaponAttachment('hunting_rifle', 'optic', 'optic_camcorder');
  const modded4 = game.getEffectiveGun('hunting_rifle');
  assert.equal(modded4.visualInfo?.nightVisionReticle, true);
  assert.equal(modded4.visualInfo?.threatScanner, true);
});

test('supports goofy sounds: rubber duck squeak and car horn blast', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  
  // Equip rubber duck squeaker on shotgun
  game.setWeaponAttachment('pump_shotgun', 'goofy', 'goofy_rubber_duck');
  const moddedShotgun = game.getEffectiveGun('pump_shotgun');
  assert.equal(moddedShotgun.squeakSound, true);

  // Equip car horn on shotgun
  game.setWeaponAttachment('pump_shotgun', 'muzzle', 'muzzle_car_horn');
  const loudShotgun = game.getEffectiveGun('pump_shotgun');
  assert.equal(loudShotgun.honkSound, true);
  assert.ok(loudShotgun.noiseRadius > 35, 'Car horn should create an ear-splitting noise ring');
});

test('persists custom weapon attachments across save and load', () => {
  const game1 = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  game1.setWeaponAttachment('nailgun_smg', 'optic', 'optic_googly');
  game1.setWeaponAttachment('nailgun_smg', 'muzzle', 'muzzle_kazoo');
  game1.setWeaponAttachment('nailgun_smg', 'magazine', 'mag_twin_soda');

  const saveJson = JSON.stringify(game1.toSave());
  const parsed = JSON.parse(saveJson);

  const game2 = Game.fromSave(parsed);

  const mods = game2.getWeaponMods('nailgun_smg');
  assert.equal(mods.optic, 'optic_googly');
  assert.equal(mods.muzzle, 'muzzle_kazoo');
  assert.equal(mods.magazine, 'mag_twin_soda');

  const eff = game2.getEffectiveGun('nailgun_smg');
  assert.equal(eff.visualInfo?.targetHealthBar, true);
  assert.equal(eff.visualInfo?.trajectoryArc, true);
});

test('gunsmith workbench table exists in world structures with visual ink drawer', () => {
  const game = new Game({ ...DEFAULT_CONFIG, creative: true, mode: 'zombie' });
  const bench = game.content.structures.get('gunsmith_bench');
  assert.ok(bench, 'gunsmith_bench structure def must exist');
  assert.equal(bench!.interact, 'gunsmith');

  // Verify ink render drawer exists in render contract
  assert.ok(STRUCTURE_DRAWERS['gunsmith_bench'], 'STRUCTURE_DRAWERS must include gunsmith_bench');
});
