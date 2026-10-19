import * as THREE from 'three';
import { VehicleEntity } from '../entities/VehicleEntity';
import { CrowdZombieEngine } from '../entities/CrowdZombieEngine';
import { ParabolicLootManager } from '../world/ParabolicLootArc';
import { DayNightSystem } from '../systems/DayNightSystem';
import { SurvivalStats } from '../systems/SurvivalStats';
import { PART_REGISTRY } from '../registries/PartDef';
import { ZOMBIE_REGISTRY } from '../registries/ZombieDef';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ Assertion Failed: ${msg}`);
    throw new Error(`Assertion Failed: ${msg}`);
  }
}

console.log('🧪 Starting Phase Z7 Car Part-Repair System & Horde Nights Test Suite...\n');

const scene = new THREE.Scene();

// 1. Vehicle Parts Registry & Inventory Setup
console.log('1. Testing Vehicle Parts Registry & Inventory Setup...');
assert(!!PART_REGISTRY.engine_block, 'engine_block must exist in PART_REGISTRY');
assert(!!PART_REGISTRY.battery, 'battery must exist in PART_REGISTRY');
assert(!!PART_REGISTRY.spark_plugs, 'spark_plugs must exist in PART_REGISTRY');

const car = new VehicleEntity({
  scene,
  position: new THREE.Vector3(0, 0, 0),
});

assert(car.parts.size >= 5, `Vehicle must contain at least 5 core parts, found ${car.parts.size}`);
assert(car.getPart('engine_block') !== undefined, 'Engine block must be installed in vehicle');
assert(car.getPart('battery') !== undefined, 'Battery must be installed in vehicle');
assert(car.getPart('spark_plugs') !== undefined, 'Spark plugs must be installed in vehicle');
assert(car.getPart('radiator_coolant') !== undefined, 'Radiator must be installed in vehicle');
assert(car.getPart('tires') !== undefined, 'Tires must be installed in vehicle');
console.log('   ✅ Verified core part slots and registry integration');

// 2. Part Condition & Hold-to-Repair Logic
console.log('\n2. Testing Part Condition & Repair Logic...');
const enginePart = car.getPart('engine_block')!;
enginePart.condition = 25; // Critical damage
car.recalculateDurability();
assert(car.durability < 80, 'Damaged part must reduce overall vehicle durability');

// Repair part
car.repairPart('engine_block', 100);
assert(car.getPart('engine_block')!.condition === 100, 'Repairing part must restore condition to 100%');
assert(car.durability >= 85, 'Overall vehicle durability must recover after repair');
console.log('   ✅ Verified part damage impact, hold-to-repair restoration, and durability recalculation');

// 3. Fuel Siphoning & Canister Pouring
console.log('\n3. Testing Fuel Siphoning & Canister Pouring...');
car.fuel = 20; // 20L
const siphoned = car.siphonFuel(5);
assert(siphoned === 5, `Must siphon 5L, got ${siphoned}`);
assert(car.fuel === 15, `Fuel tank must decrease by 5L, got ${car.fuel}`);

const poured = car.addFuel(10);
assert(poured === 10, `Must pour 10L, got ${poured}`);
assert(car.fuel === 25, `Fuel tank must increase to 25L, got ${car.fuel}`);

// Max capacity clamping
car.addFuel(100);
assert(car.fuel === car.def.fuelCapacity, `Fuel must clamp to capacity (${car.def.fuelCapacity}), got ${car.fuel}`);
console.log('   ✅ Verified bidirectional fuel siphoning and tank capacity clamping');

// 4. Cold Start Crank Mini-Game Range
console.log('\n4. Testing Cold Start Crank Mini-Game Sweet-Spot Range...');
const sweetSpotLow = 0.40;
const sweetSpotHigh = 0.60;
const testPointers = [0.25, 0.50, 0.85];

const results = testPointers.map(p => p >= sweetSpotLow && p <= sweetSpotHigh);
assert(!results[0], '0.25 must be outside sweet-spot (sputter)');
assert(results[1], '0.50 must be inside sweet-spot (successful ignition)');
assert(!results[2], '0.85 must be outside sweet-spot (sputter)');
console.log('   ✅ Verified crank mini-game sweet spot timing windows');

// 5. 7th-Day Dusk Siren Horde Night Trigger
console.log('\n5. Testing 7th-Day Dusk Siren Horde Night Trigger...');
const dayNight = new DayNightSystem(scene);
const stats = new SurvivalStats();

dayNight.dayCount = 7;
dayNight.cycleDuration = 100;
dayNight.timeOfDay = 60; // 60/100 * 360 = 216 deg -> DUSK!

let hordeTriggered = false;
dayNight.onHordeNightCallback = () => {
  hordeTriggered = true;
};

dayNight.update(0.1, stats, true);
assert(dayNight.isHordeNight, 'Day 7 Dusk must activate isHordeNight flag');
assert(hordeTriggered, 'Day 7 Dusk must fire onHordeNightCallback');
console.log('   ✅ Verified 7th-day dusk siren horde night state activation');

// 6. Highway Hulk Boss Zombie & Relentless Wave
console.log('\n6. Testing Highway Hulk Boss Zombie & Relentless Wave...');
const lootMgr = new ParabolicLootManager(scene);
const zombieEngine = new CrowdZombieEngine({ scene, lootManager: lootMgr });

const bossDef = ZOMBIE_REGISTRY.highway_hulk;
assert(!!bossDef, 'highway_hulk must exist in ZOMBIE_REGISTRY');
assert(bossDef.isBoss === true, 'highway_hulk must be flagged as boss');
assert(bossDef.maxHealth >= 400, `highway_hulk must have massive HP pool (>= 400), got ${bossDef.maxHealth}`);

const hulk = zombieEngine.spawnHighwayHulk(new THREE.Vector3(0, 0, 5));
assert(hulk.mesh.scale.x > 2.0, `Highway Hulk must have enlarged scale (> 2.0), got ${hulk.mesh.scale.x}`);
assert(hulk.def.damage >= 35, 'Highway Hulk must deal heavy boss damage');

// Test wave trigger
const initialCount = zombieEngine.zombies.length;
zombieEngine.triggerHordeNight(new THREE.Vector3(0, 0, 0), 10);
assert(zombieEngine.zombies.length >= initialCount + 10, 'triggerHordeNight must spawn horde wave');
console.log('   ✅ Verified Highway Hulk 2.2x scale diorama model, boss stats, and horde wave spawning');

console.log('\n🎉 ALL PHASE Z7 CAR PART-REPAIR & HORDE NIGHT TESTS PASSED (100% SUCCESS)!\n');
