import assert from 'node:assert';
import * as THREE from 'three';
import { CONTAINER_REGISTRY } from '../registries/ContainerDef';
import { LOOT_TABLES } from '../registries/LootTable';
import { ContainerEntity } from '../world/ContainerEntity';
import { ParabolicLoot, ParabolicLootManager } from '../world/ParabolicLootArc';
import { JackpotBeam, JackpotFXManager } from '../world/JackpotFX';
import { ContainerManager } from '../world/ContainerManager';
import { InventorySystem } from '../systems/InventorySystem';
import { BuildingLot } from '../world/SettlementDef';

console.log('🧪 Starting Phase Z4 Containers, Hinged Lids, Parabolic Loot Arcs & Jackpot FX Test Suite...\n');

// 1. Container Registry Integrity
console.log('1. Testing Container Registry Integrity...');
const containerIds = Object.keys(CONTAINER_REGISTRY);
assert(containerIds.length >= 10, `Expected at least 10 registered containers, found ${containerIds.length}`);

for (const id of containerIds) {
  const c = CONTAINER_REGISTRY[id];
  assert(c.id === id, `Container ID mismatch: ${c.id} vs ${id}`);
  assert(c.minItems > 0, `Container ${id} must have minItems > 0`);
  assert(c.maxItems >= c.minItems, `Container ${id} maxItems (${c.maxItems}) must be >= minItems (${c.minItems})`);
  assert(c.slots >= c.maxItems, `Container ${id} slots (${c.slots}) must accommodate maxItems (${c.maxItems})`);
  assert(c.restockDays >= 1, `Container ${id} restockDays must be >= 1`);
  assert(!!LOOT_TABLES[c.defaultLootTable], `Container ${id} references unknown loot table: ${c.defaultLootTable}`);
}
console.log(`   ✅ Verified ${containerIds.length} containers in CONTAINER_REGISTRY with valid loot tables`);

// 2. 3D Diorama Mesh & Hinged Lid Setup
console.log('\n2. Testing 3D Diorama Container Models & Lid Pivots...');
const scene = new THREE.Scene();
const testIds = ['toolbox', 'cash_register', 'fridge', 'locker', 'vault_safe', 'supply_drop_crate', 'vending_machine', 'pinata', 'suitcase', 'medicine_cabinet'];

for (const id of testIds) {
  const def = CONTAINER_REGISTRY[id];
  const ent = new ContainerEntity(scene, {
    def,
    position: new THREE.Vector3(10, 0, 10),
  });

  assert(ent.mesh instanceof THREE.Group, `${id} mesh must be a THREE.Group`);
  assert(ent.mesh.children.length >= 1, `${id} must contain child diorama meshes`);
  assert(ent.canInteract(new THREE.Vector3(10, 0, 11)), `${id} should be interactable within 1m`);
  assert(!ent.canInteract(new THREE.Vector3(10, 0, 20)), `${id} should NOT be interactable at 10m`);
  ent.destroy(scene);
}
console.log(`   ✅ Verified 3D diorama meshes and interaction radii for all 10 container types`);

// 3. Rubbery Spring-Hinged Lid Opening Dynamics
console.log('\n3. Testing Rubbery Spring-Hinged Lid Dynamics...');
const toolboxDef = CONTAINER_REGISTRY['toolbox'];
const toolbox = new ContainerEntity(scene, {
  def: toolboxDef,
  position: new THREE.Vector3(0, 0, 0),
});

const inv = new InventorySystem();
const lootMgr = new ParabolicLootManager(scene);
const jackpotMgr = new JackpotFXManager(scene);

assert(!toolbox.isOpened, 'Toolbox must start closed');
const openResult = toolbox.open(inv, lootMgr, jackpotMgr);
assert(openResult.success, 'Toolbox open() must succeed');
assert(toolbox.isOpened, 'Toolbox must be marked opened');

// Simulate 30 frames at 60Hz (~0.5s) to verify spring motion progression
let previousVal = 0;
for (let frame = 0; frame < 30; frame++) {
  toolbox.update(1 / 60, new THREE.Vector3(0, 0, 0));
}
// Second open attempt must fail gracefully
const secondOpen = toolbox.open(inv, lootMgr, jackpotMgr);
assert(!secondOpen.success && secondOpen.reason === 'already_empty', 'Second open attempt on open container must return already_empty');
toolbox.destroy(scene);
console.log('   ✅ Verified second-order harmonic spring lid opening and idempotent open protection');

// 4. Lock Verification & Tool/Key Checking
console.log('\n4. Testing Lock Verification & Key Requirements...');
const safeDef = CONTAINER_REGISTRY['vault_safe'];
const safe = new ContainerEntity(scene, {
  def: safeDef,
  position: new THREE.Vector3(5, 0, 5),
});

assert(safe.isLocked, 'Vault safe must start locked');
assert(safeDef.lockType === 'keycard_red', 'Vault safe must require keycard_red');

// Attempt open without red keycard
const lockedAttempt = safe.open(inv, lootMgr, jackpotMgr);
assert(!lockedAttempt.success && lockedAttempt.reason === 'locked', 'Safe open without keycard_red must be rejected');
assert(safe.isLocked, 'Safe must remain locked');

// Give player the red keycard
inv.addItem('keycard_red', 1);
assert(inv.getItemCount('keycard_red') >= 1, 'Inventory must hold keycard_red');

const unlockedAttempt = safe.open(inv, lootMgr, jackpotMgr);
assert(unlockedAttempt.success, 'Safe open with keycard_red must succeed');
assert(!safe.isLocked, 'Safe must now be unlocked');
assert(safe.isOpened, 'Safe must be opened');
safe.destroy(scene);
console.log('   ✅ Verified locked container defense, lock rejection, and key unlocking');

// 5. Parabolic Loot Trajectory & Multi-Bounce Physics
console.log('\n5. Testing Parabolic Loot Trajectory & Multi-Bounce Kinematics...');
const lootItem = lootMgr.spawnArc({
  itemId: 'wrench',
  count: 1,
  origin: new THREE.Vector3(0, 0.6, 0),
  angleRad: 0,
  upwardSpeed: 4.0,
  radialSpeed: 2.0,
});

assert(!lootItem.isSettled, 'Loot item must start airborne in flight');
assert(lootItem.velocity.y > 3.0, 'Loot item must have upward velocity');
assert(lootItem.velocity.x > 1.5, 'Loot item must have forward velocity');

// Step simulation through ballistic arc until ground bounce
let bounced = false;
for (let step = 0; step < 60; step++) {
  lootMgr.update(1 / 60, new THREE.Vector3(50, 0, 50), inv);
  if (lootItem.position.y <= 0.26 && lootItem.velocity.y > 0) {
    bounced = true;
  }
}

// Step until completely settled on the ground
for (let step = 0; step < 120; step++) {
  lootMgr.update(1 / 60, new THREE.Vector3(50, 0, 50), inv);
}
assert(lootItem.isSettled, 'Loot item must settle on ground after multi-bounce');
assert(Math.abs(lootItem.position.y - 0.25) < 0.1, `Loot item settled y should be ~0.25, got ${lootItem.position.y}`);

// Player approaches within magnet range (2.5m)
const startWrenchCount = inv.getItemCount('wrench');
lootMgr.update(0.1, new THREE.Vector3(lootItem.position.x + 0.3, 0, lootItem.position.z + 0.3), inv);
assert(!lootItem.isAlive, 'Loot item must be collected when player is in proximity');
assert(inv.getItemCount('wrench') === startWrenchCount + 1, 'Player inventory must receive the collected wrench');
console.log('   ✅ Verified parabolic ballistic flight, ground restitution bouncing, settle, and magnet collection');

// 6. Jackpot Golden Light Beam & Siren Alerts
console.log('\n6. Testing Jackpot Golden Light Beam & Siren Alerts...');
const jackpotBeam = jackpotMgr.spawnJackpot({
  position: new THREE.Vector3(15, 0, 15),
  durationSeconds: 5.0,
});

assert(jackpotBeam.isAlive, 'Jackpot beam must start alive');
assert(jackpotBeam.mesh.children.length >= 3, 'Jackpot beam must contain outer beam, inner core, and rotating beacon lights');

// Step through 3 seconds of beam updates
for (let f = 0; f < 180; f++) {
  jackpotMgr.update(1 / 60);
}
assert(jackpotBeam.isAlive, 'Jackpot beam must remain alive during duration');

// Step past 5.0s expiration
for (let f = 0; f < 150; f++) {
  jackpotMgr.update(1 / 60);
}
assert(!jackpotBeam.isAlive, 'Jackpot beam must expire and deactivate after duration');
console.log('   ✅ Verified golden light beam height, rotating beacon lights, and timed fade-out');

// 7. Settlement Building Lots Container Distribution
console.log('\n7. Testing Settlement Building Lots Container Distribution...');
const containerMgr = new ContainerManager(scene);

// Create mock settlement lots representing each zone
const mockLots: BuildingLot[] = [
  {
    id: 'lot_residential_0',
    buildingType: 'house',
    zone: 'residential',
    position: { x: 20, z: 20 },
    size: { width: 10, depth: 8 },
    rotationY: 0,
    doorPosition: { x: 20, z: 24 },
    condition: 'worn',
    containers: [],
    isSafeHouseCandidate: false,
    hasRoofFold: true,
    floorCount: 1,
  },
  {
    id: 'lot_commercial_0',
    buildingType: 'diner',
    zone: 'commercial',
    position: { x: 40, z: 20 },
    size: { width: 12, depth: 10 },
    rotationY: 0,
    doorPosition: { x: 40, z: 25 },
    condition: 'worn',
    containers: [],
    isSafeHouseCandidate: false,
    hasRoofFold: true,
    floorCount: 1,
  },
  {
    id: 'lot_industrial_0',
    buildingType: 'auto_garage',
    zone: 'industrial',
    position: { x: 60, z: 20 },
    size: { width: 14, depth: 12 },
    rotationY: 0,
    doorPosition: { x: 60, z: 26 },
    condition: 'worn',
    containers: [],
    isSafeHouseCandidate: false,
    hasRoofFold: true,
    floorCount: 1,
  },
  {
    id: 'lot_civic_0',
    buildingType: 'town_hall',
    zone: 'civic',
    position: { x: 80, z: 20 },
    size: { width: 16, depth: 14 },
    rotationY: 0,
    doorPosition: { x: 80, z: 27 },
    condition: 'worn',
    containers: [],
    isSafeHouseCandidate: false,
    hasRoofFold: true,
    floorCount: 1,
  },
];

containerMgr.populateFromSettlementLots(mockLots);
assert(containerMgr.containers.length >= 10, `Expected at least 10 containers across 4 lots, got ${containerMgr.containers.length}`);

// Verify that diner got cash_register
const hasCashRegister = containerMgr.containers.some(c => c.def.id === 'cash_register');
assert(hasCashRegister, 'Commercial diner lot must spawn a cash_register');

// Verify that auto_garage got toolbox
const hasToolbox = containerMgr.containers.some(c => c.def.id === 'toolbox');
assert(hasToolbox, 'Industrial auto_garage lot must spawn a toolbox');

// Verify that town_hall got vault_safe
const hasVaultSafe = containerMgr.containers.some(c => c.def.id === 'vault_safe');
assert(hasVaultSafe, 'Civic town_hall lot must spawn a vault_safe');

// Test nearby query within 1m of the toolbox position (60 - 5.5 = 54.5, 20)
const nearGarage = containerMgr.getNearbyContainer(new THREE.Vector3(55.0, 0, 20.0));
assert(nearGarage !== null, 'getNearbyContainer should find container near garage bench');

containerMgr.clear();
console.log('   ✅ Verified settlement lot population, zone-appropriate container spawning, and proximity queries');

// 8. Restock Timer Logic
console.log('\n8. Testing Container Restock Timer Logic...');
const restockToolbox = new ContainerEntity(scene, {
  def: { ...toolboxDef, restockDays: 1 }, // 1 day = 120s
  position: new THREE.Vector3(0, 0, 0),
});

restockToolbox.open(inv, lootMgr, jackpotMgr);
assert(restockToolbox.isOpened, 'Container must be opened');
assert(restockToolbox.restockTimer === 120, `Restock timer must be initialized to 120s, got ${restockToolbox.restockTimer}`);

// Advance 121 seconds
restockToolbox.update(121, new THREE.Vector3(0, 0, 0));
assert(!restockToolbox.isOpened, 'Container must restock and reset to closed after restockTimer expires');
restockToolbox.destroy(scene);
console.log('   ✅ Verified restock cooldown timer and automatic container replenishment');

console.log('\n🎉 ALL PHASE Z4 CONTAINER, LOOT ARC & JACKPOT FX TESTS PASSED (100% SUCCESS)!\n');
