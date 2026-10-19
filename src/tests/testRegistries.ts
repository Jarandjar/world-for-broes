import { gameConfig } from '../config/GameConfig';
import { ITEM_REGISTRY } from '../registries/ItemDef';
import { PART_REGISTRY } from '../registries/PartDef';
import { VEHICLE_REGISTRY } from '../registries/VehicleDef';
import { ZOMBIE_REGISTRY } from '../registries/ZombieDef';
import { CONTAINER_REGISTRY } from '../registries/ContainerDef';
import { LOOT_TABLES, rollLootFromTable } from '../registries/LootTable';
import { ROAD_REGISTRY } from '../registries/RoadDef';
import { POI_REGISTRY } from '../registries/POIDef';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  }
}

console.log('🧪 Starting Phase Z0 Registry & Config Verification...\n');

// 1. Config Test
console.log('1. Testing GameConfig...');
assert(gameConfig.isClassicMode(), 'Default mode must be classic');
let notified = false;
const unsub = gameConfig.subscribe((cfg) => {
  if (cfg.mode === 'zombie') notified = true;
});
gameConfig.update({ mode: 'zombie', lootDensity: 'high', zombieDensity: 'normal' });
assert(gameConfig.isZombieMode(), 'Mode update to zombie failed');
assert(notified, 'GameConfig change listener was not notified');
unsub();
gameConfig.update({ mode: 'classic' }); // reset
console.log('   ✅ GameConfig tests passed');

// 2. Item Registry Test
console.log('2. Testing Item Registry...');
const itemCount = Object.keys(ITEM_REGISTRY).length;
console.log(`   Found ${itemCount} items in ITEM_REGISTRY`);
assert(itemCount >= 70, `Expected at least 70 items, got ${itemCount}`);

// Verify core survival items
const baseItems = ['log', 'twigs', 'grass', 'flint', 'berries', 'cooked_berries', 'axe', 'torch', 'pan', 'chicken', 'marshmallow', 'horn'];
for (const id of baseItems) {
  assert(!!ITEM_REGISTRY[id], `Base survival item missing: ${id}`);
}

// Verify goofy weapons & gadgets
const goofyItems = ['rubber_chicken', 'mega_mallet', 'chainsaw_of_regret', 'boing_boots', 'air_horn', 'banana_peel', 'golden_toilet_seat'];
for (const id of goofyItems) {
  assert(!!ITEM_REGISTRY[id], `Goofy item missing: ${id}`);
  const item = ITEM_REGISTRY[id];
  if (item.noise !== undefined) {
    assert(item.noise >= 0 && item.noise <= 5, `Noise scale must be 0 to 5, got ${item.noise} on ${id}`);
  }
}
console.log('   ✅ Item Registry tests passed');

// 3. Car Parts Registry Test
console.log('3. Testing Car Parts Registry...');
const partsCount = Object.keys(PART_REGISTRY).length;
console.log(`   Found ${partsCount} parts in PART_REGISTRY`);
assert(partsCount >= 15, `Expected at least 15 parts, got ${partsCount}`);

const requiredParts = ['battery', 'spark_plugs', 'engine_block', 'ignition_key', 'tire_standard', 'radiator_coolant', 'fan_belt'];
for (const id of requiredParts) {
  assert(!!PART_REGISTRY[id], `Required car part missing: ${id}`);
  assert(PART_REGISTRY[id].isRequired, `Part ${id} must be marked isRequired`);
}

const jankParts = ['pancake_stack_tire', 'sock_fan_belt', 'bucket_headlight', 'rubber_chicken_horn'];
for (const id of jankParts) {
  assert(!!PART_REGISTRY[id], `Jank fix part missing: ${id}`);
  assert(!!PART_REGISTRY[id].isJankFix, `Part ${id} must be marked isJankFix`);
}
console.log('   ✅ Car Parts Registry tests passed');

// 4. Vehicle Registry Test (Expanded Fleet)
console.log('4. Testing Vehicle Registry...');
const vehicleCount = Object.keys(VEHICLE_REGISTRY).length;
console.log(`   Found ${vehicleCount} vehicles in VEHICLE_REGISTRY`);
assert(vehicleCount >= 50, `Expected at least 50 vehicles from Sections 14 and 38-44, got ${vehicleCount}`);

// Verify Whirlybird helicopter
const whirlybird = VEHICLE_REGISTRY['whirlybird'];
assert(!!whirlybird && !!whirlybird.isAircraft, 'Whirlybird must be defined as aircraft');
assert(whirlybird.altitudeBands?.includes('high') ?? false, 'Whirlybird must have high altitude band');

// Verify Boats & Water
const trawler = VEHICLE_REGISTRY['fishing_trawler'];
assert(!!trawler && !!trawler.floatsOnWater, 'Fishing trawler must float on water');
const hovercraft = VEHICLE_REGISTRY['swamp_hovercraft'];
assert(!!hovercraft && hovercraft.offroadGripMultiplier >= 0.9, 'Swamp hovercraft must have high grip');

// Verify Big Rigs & Trailers
const semi = VEHICLE_REGISTRY['semi_rig'];
assert(!!semi && !!semi.towsTrailer, 'Semi rig must be able to tow trailers');
const trailer = VEHICLE_REGISTRY['trailer_cargo'];
assert(!!trailer && !!trailer.isTrailer && trailer.trunkSlots === 30, 'Cargo trailer must have 30 slots');

console.log('   ✅ Vehicle Registry tests passed');

// 5. Zombie Registry Test
console.log('5. Testing Zombie Registry...');
const zombieCount = Object.keys(ZOMBIE_REGISTRY).length;
console.log(`   Found ${zombieCount} zombies in ZOMBIE_REGISTRY`);
assert(zombieCount >= 14, `Expected at least 14 zombies, got ${zombieCount}`);

const boss = ZOMBIE_REGISTRY['highway_hulk'];
assert(!!boss && !!boss.isBoss, 'Highway Hulk boss must be defined');
assert(boss.maxHealth >= 400, 'Highway Hulk boss HP must be >= 400');

for (const id in ZOMBIE_REGISTRY) {
  const z = ZOMBIE_REGISTRY[id];
  assert(z.walkSpeed > 0 && z.chaseSpeed > z.walkSpeed, `Speeds invalid for zombie: ${id}`);
  assert(z.hearingSensitivity >= 0.5, `Hearing sensitivity invalid for ${id}`);
  assert(z.drops.length > 0, `Zombie ${id} must drop items`);
  for (const drop of z.drops) {
    assert(!!ITEM_REGISTRY[drop.itemId], `Zombie drop item ${drop.itemId} not in ITEM_REGISTRY`);
  }
}
console.log('   ✅ Zombie Registry tests passed');

// 6. Containers and Loot Tables Test
console.log('6. Testing Containers & Loot Tables...');
assert(Object.keys(CONTAINER_REGISTRY).length >= 10, 'Containers missing');
for (const cid in CONTAINER_REGISTRY) {
  const c = CONTAINER_REGISTRY[cid];
  assert(!!LOOT_TABLES[c.defaultLootTable], `Container ${cid} references unknown loot table: ${c.defaultLootTable}`);
}

// Test loot rolling
for (const tid in LOOT_TABLES) {
  const drops = rollLootFromTable(tid, 2, 4, 1.0);
  assert(drops.length >= 1, `Loot table ${tid} failed to roll any drops`);
  for (const d of drops) {
    assert(!!ITEM_REGISTRY[d.itemId], `Rolled item ${d.itemId} not in ITEM_REGISTRY`);
    assert(d.count >= 1, `Rolled count must be >= 1 for ${d.itemId}`);
  }
}
console.log('   ✅ Containers & Loot Table tests passed');

// 7. Road and POI Registry Test
console.log('7. Testing Roads & POIs...');
const roadCount = Object.keys(ROAD_REGISTRY).length;
console.log(`   Found ${roadCount} road biomes in ROAD_REGISTRY`);
assert(roadCount === 10, `Expected 10 road biomes from Section 12, got ${roadCount}`);

const poiCount = Object.keys(POI_REGISTRY).length;
console.log(`   Found ${poiCount} POIs in POI_REGISTRY`);
assert(poiCount >= 15, `Expected at least 15 POIs, got ${poiCount}`);

assert(!!POI_REGISTRY['evac_dock']?.isEvacPoint, 'evac_dock must be marked isEvacPoint');
assert(!!POI_REGISTRY['radio_tower']?.isRadioQuestSite, 'radio_tower must be marked isRadioQuestSite');
console.log('   ✅ Roads & POIs tests passed');

// 8. RoadNetwork & A* Reachability Validation (Phase Z1)
console.log('8. Testing Phase Z1 RoadNetwork & A* Reachability...');
import { RoadNetwork } from '../world/RoadNetwork';

const network = new RoadNetwork();
assert(network.nodes.size === 11, `Expected 11 road nodes, got ${network.nodes.size}`);
assert(network.segments.length >= 13, `Expected at least 13 road segments, got ${network.segments.length}`);

// Verify Catmull-Rom splines
let totalProps = 0;
for (const seg of network.segments) {
  assert(seg.length > 5, `Segment ${seg.id} has invalid length: ${seg.length}`);
  assert(seg.width >= 4.0, `Segment ${seg.id} has invalid width: ${seg.width}`);
  assert(seg.waypoints.length >= 2, `Segment ${seg.id} has insufficient waypoints`);
  totalProps += seg.props.length;
}
console.log(`   Verified ${network.segments.length} splines with ${totalProps} roadside props generated`);
assert(totalProps >= 15, `Expected at least 15 roadside props, got ${totalProps}`);

// Test A* Pathfinding
const pathSpawnToGarage = network.findPath('meadow_spawn', 'scrapyard_garage');
assert(!!pathSpawnToGarage && pathSpawnToGarage.length >= 2, 'Failed to find path from spawn to auto garage');
console.log(`   A* Path Spawn -> Auto Garage: ${pathSpawnToGarage?.join(' -> ')}`);

const pathGarageToCove = network.findPath('scrapyard_garage', 'cove_dock');
assert(!!pathGarageToCove && pathGarageToCove.length >= 3, 'Failed to find path from auto garage to splash cove evac docks');
console.log(`   A* Path Auto Garage -> Evac Docks: ${pathGarageToCove?.join(' -> ')}`);

// Full graph reachability test from spawn
const reachability = network.validateReachability();
assert(reachability.valid, `Unreachable POIs found: ${reachability.unreachableNodes.join(', ')}`);
console.log(`   ✅ 100% of POIs reachable from spawn (${reachability.checkedPairs} routes verified)`);

// Spatial queries in Zombie Mode
gameConfig.update({ mode: 'zombie' });
assert(network.isOnRoad(0, 0), 'Spawn (0,0) must be on road in zombie mode');
assert(network.getSpeedBonusAt(0, 0) >= 1.08, 'Speed bonus on road must be >= 1.08');
assert(network.getGripAt(0, 0) >= 0.90, 'Grip on meadow country lane must be >= 0.90');

// Off-road check
assert(!network.isOnRoad(60, -60), '(60,-60) must be off road');
assert(network.getSpeedBonusAt(60, -60) === 1.0, 'Off-road speed bonus must be 1.0');
assert(network.getGripAt(60, -60) === 0.90, 'Off-road grip must be 0.90');

// Tundra Ice Road grip check
const iceGrip = network.getGripAt(40, 38);
assert(iceGrip <= 0.30, `Ice road grip must be slippery (<= 0.30), got ${iceGrip}`);

// Nearest POI query
const nearestToDesert = network.getNearestPOI(42, -18);
assert(nearestToDesert?.poi.id === 'gas_station', 'Expected gas station near (42, -18)');

// Classic Mode Preservation Guarantee
gameConfig.update({ mode: 'classic' });
assert(!network.isOnRoad(0, 0), 'In classic mode, isOnRoad must always return false');
assert(network.getSpeedBonusAt(0, 0) === 1.0, 'In classic mode, speed bonus must be 1.0');
assert(network.getGripAt(0, 0) === 0.90, 'In classic mode, grip must be default 0.90');
console.log('   ✅ Spatial queries and classic mode preservation tests passed');

// 9. ActionMap & Input Architecture Test (Phase Z0B)
console.log('9. Testing Phase Z0B ActionMap & Input Buffering...');
import { actionMap } from '../input/ActionMap';

// Reset map
actionMap.reset();

// Test physical key code mapping (KeyW, KeyA, KeyS, KeyD)
actionMap.handleKeyDown('KeyW', 1000);
assert(actionMap.isActionActive('MOVE_FORWARD'), 'KeyW must activate MOVE_FORWARD');
let move = actionMap.getMovementVector();
assert(move.x === 0 && move.y === -1, `Move vector must be (0, -1), got (${move.x}, ${move.y})`);

actionMap.handleKeyDown('KeyD', 1000);
assert(actionMap.isActionActive('MOVE_RIGHT'), 'KeyD must activate MOVE_RIGHT');
move = actionMap.getMovementVector();
// Normalized diagonal length should be ~1
assert(Math.abs(Math.hypot(move.x, move.y) - 1.0) < 0.001, 'Diagonal movement must be normalized');

actionMap.handleKeyUp('KeyW');
actionMap.handleKeyUp('KeyD');
assert(!actionMap.isActionActive('MOVE_FORWARD'), 'KeyW release must deactivate MOVE_FORWARD');
assert(!actionMap.isActionActive('MOVE_RIGHT'), 'KeyD release must deactivate MOVE_RIGHT');

// Test Sneak Toggle (KeyC)
assert(!actionMap.isActionActive('SNEAK'), 'Sneak must start false');
actionMap.handleKeyDown('KeyC', 1000);
assert(actionMap.isActionActive('SNEAK'), 'KeyC must toggle Sneak to true');
actionMap.handleKeyUp('KeyC');
assert(actionMap.isActionActive('SNEAK'), 'Sneak remains true after keyup because it is a toggle');
actionMap.handleKeyDown('KeyC', 1000);
assert(!actionMap.isActionActive('SNEAK'), 'Second KeyC must toggle Sneak to false');
actionMap.handleKeyUp('KeyC');

// Test 120ms Action Input Buffer
actionMap.reset();
actionMap.bufferAction('ACTION_PRIMARY', 1000); // Buffered at t=1000ms

// Consume within buffer window (t=1060ms -> delta 60ms <= 120ms)
const consumedOk = actionMap.consumeBufferedAction('ACTION_PRIMARY', 1060);
assert(consumedOk, 'Action buffered within 120ms window must be consumed');

// Second consume attempt must return false (already consumed)
const consumedTwice = actionMap.consumeBufferedAction('ACTION_PRIMARY', 1070);
assert(!consumedTwice, 'Action cannot be consumed more than once');

// Test Buffer Expiry (> 120ms)
actionMap.bufferAction('ACTION_PRIMARY', 2000);
const expiredConsume = actionMap.consumeBufferedAction('ACTION_PRIMARY', 2130); // 130ms later
assert(!expiredConsume, 'Action buffered >120ms ago must expire and not be consumed');

// Test Action Listeners (Events)
let honkDispatched = false;
actionMap.on('HONK', () => {
  honkDispatched = true;
});
actionMap.handleKeyDown('KeyH', 3000);
assert(honkDispatched, 'KeyH must dispatch HONK event listener');
actionMap.handleKeyUp('KeyH');

// Test Analog Virtual Stick Blending
actionMap.reset();
actionMap.analogMove.x = 0.6;
actionMap.analogMove.y = -0.8;
move = actionMap.getMovementVector();
assert(Math.abs(move.x - 0.6) < 0.001 && Math.abs(move.y - (-0.8)) < 0.001, 'Analog stick vector must match');

actionMap.reset();
console.log('   ✅ Universal ActionMap & 120ms buffer tests passed');

console.log('\n🎉 ALL PHASE Z0, Z1 & Z0B TESTS PASSED CLEANLY (100% SUCCESS)!');
