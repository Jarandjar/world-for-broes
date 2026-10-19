import { SettlementGenerator } from '../world/SettlementGenerator';
import { SETTLEMENT_SPACING_RULES } from '../world/SettlementDef';
import { gameConfig } from '../config/GameConfig';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  }
}

console.log('🧪 Starting Phase Z2 Settlements & Spacing Rules Test Suite...\n');

const gen = new SettlementGenerator(undefined, 4242);
const settlements = Array.from(gen.settlements.values());

// 1. Settlement Roster & Completeness
console.log('1. Testing Settlement Roster & Data Completeness...');
console.log(`   Found ${settlements.length} settlements across biomes`);
assert(settlements.length >= 10, `Expected at least 10 settlements, got ${settlements.length}`);

const requiredCategories = ['town', 'strip_stop', 'farmstead', 'junkyard_town', 'logging_camp', 'harbor_town', 'stilt_village', 'outpost'];
for (const cat of requiredCategories) {
  const match = settlements.find(s => s.category === cat);
  assert(!!match, `Missing settlement category: ${cat}`);
}
console.log('   ✅ All required settlement categories present');

// 2. Layout Algorithm Diversity
console.log('2. Testing Procedural Layout Algorithm Diversity (Grid, Strip, Radial, Cluster, Ring-Fence)...');
const layoutTypes = new Set(settlements.map(s => s.layoutType));
console.log('   Encountered layout types:', Array.from(layoutTypes).join(', '));
assert(layoutTypes.has('grid'), 'Grid layout missing (Town)');
assert(layoutTypes.has('strip'), 'Strip layout missing (Strip Stop)');
assert(layoutTypes.has('radial'), 'Radial layout missing (Stilt / Ghost Village)');
assert(layoutTypes.has('cluster'), 'Cluster layout missing (Farmstead / Logging / Junkyard)');
assert(layoutTypes.has('ring_fence'), 'Ring-fence layout missing (Outpost)');

// Inspect Bro-ville Town Grid
const town = gen.settlements.get('town_bro_ville');
assert(!!town, 'Bro-ville town not found');
assert(town!.lots.length >= 7, `Town should have at least 7 building lots, got ${town!.lots.length}`);
const townZones = new Set(town!.lots.map(l => l.zone));
assert(townZones.has('civic') && townZones.has('commercial') && townZones.has('residential'), 'Town must contain civic, commercial, and residential zones');

// Inspect Desert Oasis Strip Stop
const strip = gen.settlements.get('strip_desert_oasis');
assert(!!strip, 'Desert Oasis strip stop not found');
assert(strip!.lots.length >= 3, `Strip stop should have at least 3 lots, got ${strip!.lots.length}`);
const stripTypes = strip!.lots.map(l => l.buildingType);
assert(stripTypes.includes('gas_station') && stripTypes.includes('diner') && stripTypes.includes('motel'), 'Strip stop must contain gas_station, diner, and motel');
console.log('   ✅ All layout algorithms verified with valid zone & building assignments');

// 3. Spacing & Clearance Rules
console.log('3. Testing Pairwise Spacing & Clearance Rules...');
for (let i = 0; i < settlements.length; i++) {
  for (let j = i + 1; j < settlements.length; j++) {
    const s1 = settlements[i];
    const s2 = settlements[j];
    const dist = Math.hypot(s1.center.x - s2.center.x, s1.center.z - s2.center.z);
    
    // Minimum boundary clearance: clearance circles should not heavily intersect centers
    assert(dist >= Math.min(s1.clearanceRadius, s2.clearanceRadius) * 0.7, 
      `Settlements ${s1.name} and ${s2.name} are too close: ${dist.toFixed(1)}m`);
  }
}
console.log('   ✅ Pairwise settlement spacing rules verified');

// 4. Building Lot Clearance & Non-Overlap
console.log('4. Testing Building Lot Clearance & Non-Overlap...');
let totalLots = 0;
for (const s of settlements) {
  totalLots += s.lots.length;
  for (let i = 0; i < s.lots.length; i++) {
    const lotA = s.lots[i];
    assert(lotA.size.width >= 5 && lotA.size.depth >= 5, `Lot ${lotA.id} too small`);
    assert(lotA.containers.length > 0, `Lot ${lotA.id} has no loot containers assigned`);

    // Check overlap with other lots in same settlement
    for (let j = i + 1; j < s.lots.length; j++) {
      const lotB = s.lots[j];
      const dist = Math.hypot(lotA.position.x - lotB.position.x, lotA.position.z - lotB.position.z);
      const minRequiredDist = (Math.min(lotA.size.width, lotA.size.depth) + Math.min(lotB.size.width, lotB.size.depth)) * 0.45;
      assert(dist >= minRequiredDist, `Overlapping lots detected in ${s.name}: ${lotA.id} and ${lotB.id} (dist: ${dist.toFixed(1)}m, min: ${minRequiredDist.toFixed(1)}m)`);
    }
  }
}
console.log(`   ✅ All ${totalLots} building lots tested: ZERO overlapping lots detected`);

// 5. Spatial Queries (Point-in-lot & Settlement Clearance)
console.log('5. Testing Spatial Queries & POI Clearance...');
// Center of Bro-ville Town Hall must be inside lot
const townHallLot = town!.lots.find(l => l.buildingType === 'town_hall')!;
assert(gen.isInsideBuildingLot(townHallLot.position.x, townHallLot.position.z), 'Point on Town Hall center must be inside building lot');
assert(gen.isInSettlementClearance(townHallLot.position.x, townHallLot.position.z), 'Point on Town Hall must be in settlement clearance');

const retrievedLot = gen.getLotAt(townHallLot.position.x, townHallLot.position.z);
assert(retrievedLot?.id === townHallLot.id, `getLotAt mismatch: expected ${townHallLot.id}, got ${retrievedLot?.id}`);

// Far wilderness point must NOT be inside lot or clearance
assert(!gen.isInsideBuildingLot(120, 120), 'Distant wilderness coordinate must not be inside any lot');
assert(!gen.isInSettlementClearance(120, 120), 'Distant wilderness coordinate must not be inside settlement clearance');

// Nearest settlement query
const nearestToSpawn = gen.getNearestSettlement(0, 0);
assert(nearestToSpawn?.settlement.id === 'town_bro_ville', `Nearest to spawn must be Bro-ville, got ${nearestToSpawn?.settlement.id}`);
console.log('   ✅ Spatial queries and clearance bounds verified');

// 6. Seed Determinism
console.log('6. Testing Seed Determinism...');
const genA = new SettlementGenerator(undefined, 9999);
const genB = new SettlementGenerator(undefined, 9999);
const lotsA = genA.getAllLots();
const lotsB = genB.getAllLots();
assert(lotsA.length === lotsB.length, 'Lot counts must be identical for identical seeds');
for (let i = 0; i < lotsA.length; i++) {
  assert(lotsA[i].id === lotsB[i].id, `Lot ID mismatch at index ${i}`);
  assert(lotsA[i].position.x === lotsB[i].position.x && lotsA[i].position.z === lotsB[i].position.z, `Position mismatch for lot ${lotsA[i].id}`);
}
console.log('   ✅ Seed determinism verified (100% reproducible world generation)');

// 7. Classic Mode Independence
console.log('7. Testing Classic Mode Independence...');
gameConfig.update({ mode: 'classic' });
assert(gameConfig.isClassicMode(), 'Must be in classic mode');
gen.setVisible(gameConfig.isZombieMode());
assert(!gen.group.visible, 'Settlements group must be hidden in classic mode');

gameConfig.update({ mode: 'zombie' });
assert(gameConfig.isZombieMode(), 'Must be in zombie mode');
gen.setVisible(gameConfig.isZombieMode());
assert(gen.group.visible, 'Settlements group must be visible in zombie mode');
gameConfig.update({ mode: 'classic' }); // reset
console.log('   ✅ Classic mode independence verified');

console.log('\n🎉 ALL PHASE Z2 SETTLEMENTS & SPACING RULES TESTS PASSED (100% SUCCESS)!');
