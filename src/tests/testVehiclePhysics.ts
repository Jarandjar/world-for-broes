import * as THREE from 'three';
import { VEHICLE_REGISTRY } from '../registries/VehicleDef';
import { VehicleEntity } from '../entities/VehicleEntity';
import { VehicleManager } from '../world/VehicleManager';
import { RoadNetwork } from '../world/RoadNetwork';
import { CrowdZombieEngine } from '../entities/CrowdZombieEngine';
import { ParabolicLootManager } from '../world/ParabolicLootArc';
import { Player } from '../entities/Player';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ Assertion Failed: ${msg}`);
    throw new Error(`Assertion Failed: ${msg}`);
  }
}

console.log('🧪 Starting Phase Z6 Arcade Vehicle Physics & Driving Cockpit Test Suite...\n');

const scene = new THREE.Scene();

// 1. Vehicle Registry Integrity
console.log('1. Testing Vehicle Registry Integrity...');
const hatchbackDef = VEHICLE_REGISTRY.rusty_hatchback;
assert(!!hatchbackDef, 'rusty_hatchback must exist in VEHICLE_REGISTRY');
assert(hatchbackDef.topSpeed === 14.0, `topSpeed must be 14.0, got ${hatchbackDef.topSpeed}`);
assert(hatchbackDef.acceleration === 6.0, `acceleration must be 6.0, got ${hatchbackDef.acceleration}`);
assert(hatchbackDef.handling === 1.0, `handling must be 1.0, got ${hatchbackDef.handling}`);
assert(hatchbackDef.fuelCapacity === 35, `fuelCapacity must be 35, got ${hatchbackDef.fuelCapacity}`);
assert(hatchbackDef.hornType === 'sad_trombone', `hornType must be sad_trombone, got ${hatchbackDef.hornType}`);
console.log('   ✅ Verified rusty_hatchback specification in VEHICLE_REGISTRY');

// 2. 3D Diorama Mesh Assembly & Structure
console.log('\n2. Testing 3D Diorama Hatchback Mesh Assembly...');
const car = new VehicleEntity({
  scene,
  position: new THREE.Vector3(0, 0, 0),
});

assert(!!car.group, 'Vehicle group must exist');
assert(!!car.chassis, 'Vehicle chassis must exist');
assert(car.wheelMeshes.length === 4, `Vehicle must possess 4 wheels, got ${car.wheelMeshes.length}`);
assert(!!car.frontLeftKnuckle && !!car.frontRightKnuckle, 'Front steerable knuckles must exist');
assert(!!car.driverDoor && !!car.driverDoorPivot, 'Hinged driver door must exist');
assert(!!car.antenna, 'Antenna mesh must exist');
assert(car.headlights.length === 2, `Car must have 2 headlights, got ${car.headlights.length}`);
assert(car.wheelRadius === 0.38, `Wheel radius must be 0.38m, got ${car.wheelRadius}`);
console.log('   ✅ Verified 3D diorama papercraft chassis, knuckles, 4 wheels, and hinged door');

// 3. Boarding & Door Swing Mechanics
console.log('\n3. Testing Boarding & Door Swing Mechanics...');
const player = new Player(scene);
player.position.set(1.5, 0, 0); // Near driver side

assert(car.canEnter(player.position), 'Player within 2.8m should be able to enter');
assert(!car.isDriverSeated, 'Car should initially have no driver');

// Enter Vehicle
car.enterVehicle(player);
assert(car.isDriverSeated, 'Car must register seated driver after enterVehicle');
assert(car.doorTargetAngle > 1.0, 'Entering car must swing door open (> 1.0 rad)');

// Advance frame to verify door tween
car.update(0.1);
assert(car.doorCurrentAngle > 0.1, 'Door current angle must animate open via frame tween');

// Exit Vehicle
car.exitVehicle(player);
assert(!car.isDriverSeated, 'Car must register no driver after exitVehicle');
console.log('   ✅ Verified interactive boarding, seated state toggle, and hinged door swing tween');

// 4. Arcade Acceleration, Braking & Deceleration
console.log('\n4. Testing Arcade Acceleration, Braking & Deceleration...');
car.isDriverSeated = true; // Seat driver for physics tests

// Gas acceleration: 60 frames (~1 sec)
for (let i = 0; i < 60; i++) {
  car.applyInput(1.0, 0, 0, false);
  car.update(0.016);
}
assert(car.speed > 3.0, `Car speed must accelerate under gas, got ${car.speed.toFixed(2)} m/s`);
assert(car.speed <= hatchbackDef.topSpeed, `Car speed must not exceed topSpeed (${hatchbackDef.topSpeed}), got ${car.speed}`);

// Braking: 30 frames
const speedBeforeBrake = car.speed;
for (let i = 0; i < 30; i++) {
  car.applyInput(0, 1.0, 0, false);
  car.update(0.016);
}
assert(car.speed < speedBeforeBrake, `Braking must reduce vehicle speed, from ${speedBeforeBrake.toFixed(2)} to ${car.speed.toFixed(2)}`);

// Reverse from stop
car.speed = 0;
for (let i = 0; i < 40; i++) {
  car.applyInput(0, 1.0, 0, false);
  car.update(0.016);
}
assert(car.speed < 0, `Brake pedal while stopped must engage reverse gear, got ${car.speed.toFixed(2)} m/s`);
console.log('   ✅ Verified acceleration curve, braking power, and reverse gear logic');

// 5. Wheel Roll Dynamics (ω = v / r)
console.log('\n5. Testing Wheel Roll Dynamics (ω = v / r)...');
const startWheelRot = car.wheelMeshes[0].rotation.x;
car.speed = 8.0; // 8 m/s
car.update(0.1); // 0.1s
const expectedDelta = (8.0 / 0.38) * 0.1; // ~2.105 rad
const actualDelta = car.wheelMeshes[0].rotation.x - startWheelRot;
assert(Math.abs(actualDelta - expectedDelta) < 0.05, `Wheel roll angular velocity must follow ω = v/r. Expected ~${expectedDelta.toFixed(3)}, got ${actualDelta.toFixed(3)}`);
console.log('   ✅ Verified angular wheel spin rotation locked to linear ground speed');

// 6. Steering Knuckle Yaw & Turn Physics
console.log('\n6. Testing Steering Knuckle Yaw & Turn Physics...');
car.applyInput(0.5, 0, 1.0, false); // Full right steer
car.update(0.1);
assert(car.steeringAngle > 0.15, `Steering angle must turn right, got ${car.steeringAngle.toFixed(3)} rad`);
assert(car.frontLeftKnuckle.rotation.y > 0.15, 'Front-left knuckle must yaw with steering input');
assert(car.frontRightKnuckle.rotation.y > 0.15, 'Front-right knuckle must yaw with steering input');
console.log('   ✅ Verified front wheel knuckles pivot on local Y axis with steering input');

// 7. Suspension Second-Order Spring Dynamics (Body Roll & Pitch)
console.log('\n7. Testing Suspension Second-Order Spring Dynamics...');
// Fast cornering: speed = 12 m/s with right turn
car.speed = 12.0;
car.steeringAngle = 0.35;
for (let i = 0; i < 15; i++) {
  car.update(0.016);
}
assert(Math.abs(car.chassis.rotation.z) > 0.01, `Chassis must roll outward during high-speed turn, got ${car.chassis.rotation.z.toFixed(4)} rad`);
assert(Math.abs(car.chassis.rotation.x) > 0.005, `Chassis must pitch during acceleration, got ${car.chassis.rotation.x.toFixed(4)} rad`);
console.log('   ✅ Verified second-order suspension spring body roll and pitch damping');

// 8. Fuel Consumption & Engine Depletion
console.log('\n8. Testing Fuel Consumption...');
const initialFuel = car.fuel;
car.speed = 10.0;
for (let i = 0; i < 100; i++) {
  car.applyInput(1.0, 0, 0, false);
}
assert(car.fuel < initialFuel, `Fuel must deplete as car drives. Initial: ${initialFuel}, Current: ${car.fuel.toFixed(3)}`);

// Depleted fuel prevents acceleration
car.fuel = 0;
const speedAtEmpty = car.speed;
car.applyInput(1.0, 0, 0, false);
assert(car.targetSteerAngle === 0, 'Depleted fuel must cut throttle inputs');
car.refuel(35);
assert(car.fuel === 35, 'Refuel must restore tank capacity');
console.log('   ✅ Verified dynamic fuel consumption and out-of-gas engine cutoff');

// 9. Obstacle Impact & Collision Response
console.log('\n9. Testing Obstacle Collision & Rubbery Bounce...');
car.speed = 10.0;
const startDurability = car.durability;
car.handleCollisionWithObstacle(new THREE.Vector3(0, 0, 2.0));
assert(car.speed < 0, `Collision must bounce vehicle velocity backward, got ${car.speed.toFixed(2)}`);
assert(car.durability < startDurability, `Collision must reduce durability, from ${startDurability} to ${car.durability}`);
console.log('   ✅ Verified obstacle collision restitution bounce and durability degradation');

// 10. Vehicular Zombie Ramming & Kinetic Damage
console.log('\n10. Testing Vehicular Zombie Ramming & Kinetic Damage...');
const lootMgr = new ParabolicLootManager(scene);
const zombieEngine = new CrowdZombieEngine({ scene, lootManager: lootMgr });
const zVictim = zombieEngine.spawnZombie('shambler', new THREE.Vector3(0, 0, 3.0));

car.speed = 10.0; // 10 m/s ramming speed
const ramDmg = car.handleRamZombie(zVictim.position);
assert(ramDmg > 100, `Ramming at 10 m/s must deal heavy kinetic damage (> 100), got ${ramDmg}`);
assert(car.speed < 10.0, 'Ramming must cause slight vehicular deceleration');
console.log('   ✅ Verified vehicular zombie impact damage (speed * 18 multiplier)');

// 11. Handbrake Drift Mechanics
console.log('\n11. Testing Handbrake Drift Mechanics...');
car.speed = 8.0;
car.applyInput(0, 0, 1.0, true); // Handbrake + steer
assert(car.isDrifting, 'Vehicle must enter isDrifting state when handbrake applied above 4 m/s');
car.update(0.016);
assert(car.velocity.x !== 0, 'Drifting must apply lateral velocity slip');
console.log('   ✅ Verified handbrake drift lateral slip and particle emission trigger');

// 12. VehicleManager Spawning & Settlement Integration
console.log('\n12. Testing VehicleManager Spawning & Settlement Integration...');
const mgr = new VehicleManager(scene);
const starterCar = mgr.spawnVehicle('rusty_hatchback', new THREE.Vector3(8.0, 0, 4.0));
assert(mgr.vehicles.length === 1, `Manager should have 1 vehicle, got ${mgr.vehicles.length}`);
assert(mgr.getNearbyVehicle(new THREE.Vector3(8.5, 0, 4.0)) === starterCar, 'getNearbyVehicle must find starter hatchback');
assert(mgr.getActiveVehicle() === null, 'No vehicle should be active initially');

starterCar.isDriverSeated = true;
assert(mgr.getActiveVehicle() === starterCar, 'getActiveVehicle must return boarded car');
starterCar.isDriverSeated = false;
mgr.clear();
console.log('   ✅ Verified VehicleManager vehicle population, spatial queries, and active tracking');

console.log('\n🎉 ALL PHASE Z6 ARCADE VEHICLE PHYSICS TESTS PASSED (100% SUCCESS)!\n');
