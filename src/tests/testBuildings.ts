import * as THREE from 'three';
import { SettlementGenerator } from '../world/SettlementGenerator';
import { BuildingManager } from '../world/BuildingManager';
import { HidingSpot } from '../world/HidingSpot';
import { gameConfig } from '../config/GameConfig';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  }
}

console.log('🧪 Starting Phase Z3 Buildings, Pop-up Roof Fold & Hiding Spots Test Suite...\n');

const settlements = new SettlementGenerator(undefined, 4242);
const manager = new BuildingManager(settlements);
const buildings = Array.from(manager.buildings.values());

// 1. Building Kit Generation & Completeness
console.log('1. Testing Building Kit Generation & Lot Coverage...');
console.log(`   Generated ${buildings.length} 3D buildings from settlement lots`);
assert(buildings.length >= 30, `Expected at least 30 buildings, got ${buildings.length}`);

for (const bldg of buildings) {
  assert(bldg.group.children.length >= 2, `Building ${bldg.lot.id} missing visual components`);
  assert(bldg.wallHeight >= 3.0, `Building ${bldg.lot.id} wall height too low: ${bldg.wallHeight}`);
  if (bldg.lot.hasRoofFold) {
    assert(bldg.roofGroup.children.length > 0, `Building ${bldg.lot.id} has roof fold flag but no roof mesh`);
  }
}
console.log('   ✅ All 3D modular buildings constructed with walls, doorways, and roof groups');

// 2. Pop-up Roof Fold Tween & Entry Dynamics
console.log('2. Testing Cardboard Pop-up Roof Fold & Interior Cutaway (Section 26 & 66)...');
const townHall = manager.buildings.get('town_bro_ville_town_hall')!;
assert(!!townHall, 'Town hall building not found');

// Initial state: outside building (15m north where street is clear)
const outsidePos = new THREE.Vector3(townHall.lot.position.x, 0, townHall.lot.position.z - 15);
townHall.update(0.016, outsidePos);
assert(!townHall.isPlayerInside, 'Player at 15m distance should NOT be inside town hall');
assert(townHall.roofFoldProgress === 0, 'Roof should be closed (progress = 0) when outside');
assert(Math.abs(townHall.roofGroup.rotation.x) < 0.01, 'Roof rotation should be near 0 when closed');

// Simulate player walking through doorway into interior
const insidePos = new THREE.Vector3(townHall.lot.position.x, 0, townHall.lot.position.z);
townHall.update(0.016, insidePos);
assert(townHall.isPlayerInside, 'Player at center should be detected inside town hall');

// Simulate multiple frames of delta time to test smooth ~0.25s pop-up roof fold tween
for (let f = 0; f < 30; f++) {
  townHall.update(0.016, insidePos);
}
assert(townHall.roofFoldProgress > 0.85, `Roof fold progress should approach 1.0, got ${townHall.roofFoldProgress.toFixed(2)}`);
assert(townHall.roofGroup.rotation.x < -1.0, `Roof should be folded upright, got angle ${townHall.roofGroup.rotation.x.toFixed(2)} rad`);
console.log(`   Roof folded upright to ${(townHall.roofGroup.rotation.x * (180 / Math.PI)).toFixed(1)}° in ~0.25s`);

// Simulate player exiting building: roof must smoothly fold back down
for (let f = 0; f < 35; f++) {
  townHall.update(0.016, outsidePos);
}
assert(!townHall.isPlayerInside, 'Player should be detected outside after exiting');
assert(townHall.roofFoldProgress < 0.05, `Roof should return to closed state, got ${townHall.roofFoldProgress.toFixed(2)}`);
assert(Math.abs(townHall.roofGroup.rotation.x) < 0.08, 'Roof rotation should rest at closed angle');
console.log('   ✅ Pop-up roof fold tween smoothly opens and closes on entry/exit');

// 3. Hiding Spots Roster & Peek States
console.log('3. Testing Slapstick Hiding Spots (Wardrobe, Under-Bed, Dumpster)...');
let totalHidingSpots = 0;
let wardrobeCount = 0;
let bedCount = 0;
let dumpsterCount = 0;

for (const bldg of buildings) {
  totalHidingSpots += bldg.hidingSpots.length;
  for (const spot of bldg.hidingSpots) {
    if (spot.type === 'wardrobe') wardrobeCount++;
    else if (spot.type === 'under_bed') bedCount++;
    else if (spot.type === 'dumpster') dumpsterCount++;
  }
}
console.log(`   Found ${totalHidingSpots} total hiding spots (${wardrobeCount} wardrobes, ${bedCount} beds, ${dumpsterCount} dumpsters)`);
assert(wardrobeCount > 0, 'Must have wardrobe hiding spots');
assert(bedCount > 0, 'Must have under-bed hiding spots');
assert(dumpsterCount > 0, 'Must have dumpster hiding spots');

// Test interaction state machine on sample spot
const sampleSpot = new HidingSpot('dumpster', new THREE.Vector3(10, 0, 10));
assert(!sampleSpot.isOccupied, 'New hiding spot must be empty');
assert(!sampleSpot.isPeeking, 'New hiding spot must not be peeking');

// Player enters spot
const entered = sampleSpot.toggleHide();
assert(entered === true, 'toggleHide should return true on entry');
assert(sampleSpot.isOccupied === true, 'Spot must be occupied after entry');
assert(sampleSpot.isPeeking === true, 'Spot must be peeking with cartoon eyes');

// Player exits spot
const exited = sampleSpot.toggleHide();
assert(exited === false, 'toggleHide should return false on exit');
assert(sampleSpot.isOccupied === false, 'Spot must be unoccupied after exit');
assert(sampleSpot.isPeeking === false, 'Spot must stop peeking');
console.log('   ✅ Hiding spot state machine (empty -> occupied/peeking -> exit) validated');

// 4. BuildingManager Proximity & Spatial Tracking
console.log('4. Testing BuildingManager Spatial Tracking & Auto-Exit...');
// Outside test in open street intersection
const openIntersectionPos = new THREE.Vector3(0, 0, 0);
manager.update(0.016, openIntersectionPos);
assert(manager.currentBuilding === null, 'No building should be active when outside in open intersection');
assert(!manager.isPlayerHiding(), 'Player should not be marked as hiding');

// Inside test
manager.update(0.016, insidePos);
assert(manager.currentBuilding?.lot.id === townHall.lot.id, `Current building should be town hall, got ${manager.currentBuilding?.lot.id}`);

// Proximity hiding spot interaction test
const diner = manager.buildings.get('strip_desert_oasis_diner')!;
assert(!!diner, 'Diner building not found');
const dinerDumpster = diner.hidingSpots.find(s => s.type === 'dumpster')!;
assert(!!dinerDumpster, 'Diner dumpster not found');

// Walk directly to dumpster
const atDumpsterPos = dinerDumpster.position.clone();
const nearbyFound = manager.getNearbyHidingSpot(atDumpsterPos);
assert(nearbyFound?.type === 'dumpster', 'Manager should locate nearby dumpster within reach');

// Interact to hide
const hideRes = manager.interactWithHidingSpot(atDumpsterPos);
assert(hideRes.success && hideRes.isHiding, 'Should successfully hide in dumpster');
assert(manager.isPlayerHiding(), 'Manager should report player as hiding');

// Interact again to exit
const exitRes = manager.interactWithHidingSpot(atDumpsterPos);
assert(exitRes.success && !exitRes.isHiding, 'Should successfully exit dumpster');
assert(!manager.isPlayerHiding(), 'Manager should report player as no longer hiding');
console.log('   ✅ BuildingManager spatial proximity and interaction tracking verified');

// 5. Classic Mode Independence
console.log('5. Testing Classic Mode Independence...');
gameConfig.update({ mode: 'classic' });
manager.setVisible(gameConfig.isZombieMode());
assert(!manager.group.visible, 'Building network must be hidden in classic mode');

gameConfig.update({ mode: 'zombie' });
manager.setVisible(gameConfig.isZombieMode());
assert(manager.group.visible, 'Building network must be visible in zombie mode');
gameConfig.update({ mode: 'classic' }); // reset
console.log('   ✅ Classic mode independence verified');

console.log('\n🎉 ALL PHASE Z3 BUILDINGS, POP-UP ROOFS & HIDING SPOTS TESTS PASSED (100% SUCCESS)!');
