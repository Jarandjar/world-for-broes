import type { ColorSet, GunDef, ItemDef, Slot, ToolKind } from './schema';

const cs = (primary: string, secondary: string, accent: string): ColorSet => ({ primary, secondary, accent });
const mat = (id: string, name: string, stack = 20, weight = 0.2, tags: string[] = []): ItemDef => ({ id, name, kind: 'material', stack, weight, tags });
const food = (id: string, name: string, hunger: number, health = 0, sanity = 0): ItemDef => ({ id, name, kind: 'food', stack: 10, weight: 0.15, food: { hunger, health, sanity } });
const gear = (id: string, name: string, slot: Slot, colors: ColorSet, extra: Partial<ItemDef['equip']> = {}): ItemDef => ({ id, name, kind: 'gear', stack: 1, weight: 0.5, equip: { slot, colors, ...extra } });
const tool = (id: string, name: string, slot: 'mainHand' | 'offHand', kind: ToolKind, colors: ColorSet, power: number, speed: number, reach: number, damage: number, extra: Partial<ItemDef['equip']> = {}): ItemDef => ({
  id, name, kind: 'tool', stack: 1, weight: 1, equip: { slot, colors, ...extra }, tool: { kind, power, speed, reach, damage },
});
const gunTool = (
  id: string, name: string, colors: ColorSet, damage: number,
  gun: GunDef, extra: Partial<ItemDef['equip']> = {}
): ItemDef => ({
  id, name, kind: 'tool', stack: 1, weight: 1.5,
  equip: { slot: 'mainHand', colors, twoHanded: true, ...extra },
  tool: { kind: 'weapon', power: 0, speed: gun.fireRate, reach: gun.maxRange, damage },
  gun,
});

export const ITEMS: ItemDef[] = [
  // materials
  mat('wood', 'Wood', 20, 0.5), mat('twig', 'Twig'), mat('stone', 'Stone', 20, 0.6), mat('flint', 'Flint'), mat('pine_cone', 'Pine Cone'), mat('acorn', 'Acorn'),
  mat('fiber', 'Plant Fiber'), mat('scrap_metal', 'Scrap Metal', 20, 0.5), mat('cloth', 'Cloth'),
  mat('fuel_canister', 'Fuel Canister', 5, 1.2), mat('car_battery', 'Car Battery', 2, 2.5), mat('scrap_tire', 'Scrap Tire', 4, 1.8),
  // automation minerals & logistics
  mat('iron_ore', 'Iron Ore', 20, 0.8), mat('copper_ore', 'Copper Ore', 20, 0.7), mat('coal_chunk', 'Coal Chunk', 20, 0.5),
  mat('crude_oil_canister', 'Crude Oil', 5, 1.5), mat('energetic_crystal', 'Energetic Crystal', 15, 0.4),
  mat('iron_plate', 'Iron Plate', 20, 0.6), mat('copper_wire', 'Copper Wire', 30, 0.2), mat('heavy_gear', 'Heavy Gear', 20, 0.5),
  mat('conveyor_belt', 'Conveyor Belt', 20, 0.3), mat('spring_inserter', 'Spring Inserter', 10, 0.8),
  mat('burner_drill', 'Burner Drill', 2, 3.5), mat('stone_furnace', 'Smelting Furnace', 2, 4.0),
  // siege defenses & raiding gear (Phase Z16 & Z17)
  mat('turret_ammo', 'Turret Bolt Ammo', 40, 0.1),
  mat('flame_fuel', 'Pressurized Flame Fuel', 10, 0.5),
  mat('c4_plunger_charge', 'C4 Plunger Breaching Charge', 5, 1.5),
  mat('repair_drone_kit', 'Micro Repair Drone', 5, 0.6),
  mat('clan_totem_crest', 'Faction Clan Crest', 1, 0.2),
  food('bandage', 'Bandage', 0, 35, 8), food('rotten_meat', 'Rotten Meat', 4, -8, -15),
  // food & side-effect culinary (Phase Z18)
  food('berries', 'Berries', 8, 1, 0), food('mushroom', 'Mushroom', 6, 0, -2), food('cactus_pulp', 'Cactus Pulp', 5, 1, 0), food('jerky', 'Mushroom Jerky', 18, 0, 2), food('canned_beans', 'Canned Beans', 26, 2, 3),
  food('wild_carrot', 'Wild Carrot', 12, 2, 1),
  food('hiccup_berries', 'Hiccup Berries', 15, 3, 2),
  food('giggle_mushroom', 'Giggle Mushroom', 10, 0, 20),
  food('chili_of_regret', 'Chili of Regret', 45, 10, 5),
  food('pancake_stack', 'Fluffy Pancake Stack', 60, 15, 25),
  food('rainbow_soup', 'Iridescent Rainbow Soup', 40, 20, 30),
  food('crispy_sock_chips', 'Crispy Sock Chips', 15, -5, 5),
  food('bubble_tea_of_buoyancy', 'Bubble Tea of Buoyancy', 20, 5, 10),
  food('antidote_gel', 'Bio-Antidote Gel', 0, 40, 15),
  // livestock, wildlife & farm foods (Phase Z19)
  mat('raw_beef', 'Raw Prime Beef', 15, 0.4),
  mat('raw_mutton', 'Raw Mutton', 15, 0.35),
  mat('raw_pork', 'Raw Pork Chop', 15, 0.35),
  mat('raw_poultry', 'Raw Poultry', 15, 0.25),
  mat('raw_venison', 'Raw Venison Steak', 15, 0.4),
  mat('rabbit_meat', 'Tender Rabbit Cut', 20, 0.2),
  mat('leather', 'Tanned Leather Hide', 20, 0.3),
  mat('wool_bundle', 'Fluffy Wool Bundle', 20, 0.15),
  mat('feather', 'Soft Down Feather', 50, 0.02),
  mat('deer_antlers', 'Branching Antlers', 10, 0.5),
  mat('lucky_rabbit_foot', 'Lucky Rabbit Foot Charm', 5, 0.1),
  mat('animal_feed', 'Golden Grain Feed', 30, 0.15),
  food('fresh_milk', 'Fresh Creamy Milk', 16, 12, 18),
  food('fresh_egg', 'Fresh Farm Egg', 8, 2, 2),
  food('cooked_steak', 'Sizzling Beef Steak', 48, 16, 12),
  food('cooked_bacon', 'Crispy Country Bacon', 36, 10, 15),
  food('roast_chicken', 'Golden Roast Chicken', 42, 14, 10),
  food('cooked_venison', 'Campfire Roast Venison', 52, 20, 16),
  food('fluffy_omelet', 'Country Herb Omelet', 38, 12, 20),
  // throwables, ammo & traps (Phase Z18 & Ballistics)
  mat('pebbles', 'Smooth River Pebbles', 50, 0.05),
  mat('rifle_ammo', 'Precision Rifle Cartridges', 60, 0.03),
  mat('shotgun_shells', 'Heavy 12-Gauge Buckshot', 40, 0.04),
  mat('scrap_nails', 'Box of Scrap Nails', 100, 0.02),
  mat('foam_darts', 'Foam Suction Darts', 40, 0.02),
  mat('cream_pie', 'Custard Cream Pie', 10, 0.3),
  mat('water_balloon', 'Water Balloon', 10, 0.2),
  mat('road_flare', 'Handheld Road Flare', 10, 0.3),
  mat('stink_sock_bomb', 'Stink Sock Bomb', 10, 0.2),
  mat('whoopee_cushion_trap', 'Whoopee Cushion Trap', 5, 0.4),
  mat('googly_eyes', 'Self-Adhesive Googly Eyes', 20, 0.05),
  // tools and weapons (main hand)
  tool('axe', 'Axe', 'mainHand', 'axe', cs('#8a6a4a', '#6f7780', '#c9c3b2'), 1, 1, 1.3, 8),
  tool('pickaxe', 'Pickaxe', 'mainHand', 'pickaxe', cs('#8a6a4a', '#5f666d', '#c9c3b2'), 1, 1, 1.3, 7),
  tool('shovel', 'Shovel', 'mainHand', 'shovel', cs('#8a6a4a', '#6f7780', '#c9c3b2'), 1, 1, 1.3, 5),
  tool('torch', 'Torch', 'mainHand', 'torch', cs('#6f4f33', '#ff9a3c', '#ffd27a'), 0, 1, 1.1, 3, { lightRadius: 5 }),
  tool('nail_bat', 'Nail Bat', 'mainHand', 'weapon', cs('#8a6a4a', '#9aa0a6', '#c9c3b2'), 0, 1.1, 1.4, 14),
  tool('crowbar', 'Crowbar', 'mainHand', 'weapon', cs('#4a4f55', '#7a7f85', '#c9c3b2'), 0, 1, 1.3, 11),
  tool('rubber_chicken', 'Rubber Chicken', 'mainHand', 'weapon', cs('#c9b04a', '#e8dcb0', '#b5441d'), 0, 1.6, 1.2, 2),
  tool('fishing_rod', 'Fishing Rod', 'mainHand', 'none', cs('#6b4a32', '#c9c3b2', '#b5441d'), 0, 1, 1.8, 0),
  // master weapons & gadgets (Phase Z18)
  tool('pool_noodle_of_justice', 'Pool Noodle of Justice', 'mainHand', 'weapon', cs('#38bdf8', '#0284c7', '#f43f5e'), 0, 1.4, 2.2, 2),
  tool('chainsaw_of_regret', 'Chainsaw of Regret', 'mainHand', 'weapon', cs('#ea580c', '#334155', '#fbbf24'), 0, 1.5, 1.4, 35),
  tool('traffic_sign_axe', 'Traffic Sign Great-Axe', 'mainHand', 'weapon', cs('#dc2626', '#f8fafc', '#475569'), 0, 0.7, 1.6, 20),
  tool('frying_pan', 'Cast Iron Frying Pan', 'mainHand', 'weapon', cs('#1e293b', '#475569', '#94a3b8'), 0, 1.0, 1.3, 10),
  tool('stale_baguette', 'Stale Baguette', 'mainHand', 'weapon', cs('#d97706', '#b45309', '#fde68a'), 0, 1.2, 1.3, 6),
  // ballistic firearms & projectile guns
  gunTool('slingshot', 'Forked Slingshot', cs('#854d0e', '#ca8a04', '#fef08a'), 8, {
    ammoItem: 'pebbles', ammoName: 'River Pebbles', ammoPerShot: 1,
    aimSystem: 'bloom', projectileSpeed: 16, maxRange: 16,
    baseDamage: 8, minDamage: 4, falloffStartDist: 6, spreadAngle: 0.08,
    fireRate: 1.6, noiseRadius: 3, bulletColor: '#94a3b8', bulletRadius: 0.12,
  }),
  gunTool('potato_cannon', 'Pressurized Potato Cannon', cs('#475569', '#64748b', '#cbd5e1'), 32, {
    ammoItem: 'wild_carrot', ammoName: 'Wild Carrots / Beans', ammoPerShot: 1,
    aimSystem: 'arc', projectileSpeed: 13, maxRange: 22,
    baseDamage: 32, minDamage: 16, falloffStartDist: 14, spreadAngle: 0.05,
    fireRate: 0.85, noiseRadius: 16, arced: true, bulletColor: '#ea580c', bulletRadius: 0.18,
  }),
  gunTool('hunting_rifle', 'Precision Hunting Rifle', cs('#334155', '#475569', '#94a3b8'), 55, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 45, maxRange: 48,
    baseDamage: 55, minDamage: 28, falloffStartDist: 22, spreadAngle: 0.015,
    fireRate: 0.75, noiseRadius: 28, penetration: 2, bulletColor: '#facc15', bulletRadius: 0.1,
  }),
  gunTool('pump_shotgun', 'Heavy Pump Shotgun', cs('#1e293b', '#3b4252', '#d8dee9'), 16, {
    ammoItem: 'shotgun_shells', ammoName: '12-Gauge Buckshot', ammoPerShot: 1,
    aimSystem: 'bloom', projectileSpeed: 22, maxRange: 16,
    baseDamage: 16, minDamage: 5, falloffStartDist: 6, spreadAngle: 0.24,
    fireRate: 0.9, noiseRadius: 26, pelletCount: 6, bulletColor: '#fdba74', bulletRadius: 0.08,
  }),
  gunTool('nailgun_smg', 'Scrap Pneumatic SMG', cs('#ea580c', '#334155', '#fbbf24'), 14, {
    ammoItem: 'scrap_nails', ammoName: 'Scrap Nails', ammoPerShot: 1,
    aimSystem: 'burst', projectileSpeed: 24, maxRange: 20,
    baseDamage: 14, minDamage: 7, falloffStartDist: 10, spreadAngle: 0.11,
    fireRate: 5.0, noiseRadius: 14, bulletColor: '#94a3b8', bulletRadius: 0.09,
  }),
  // expanded weapon pack: tactical + DIY + goofy + prototype
  gunTool('plinker_pistol', 'Plinker Sidearm', cs('#64748b', '#334155', '#f8fafc'), 14, {
    ammoItem: 'scrap_nails', ammoName: 'Scrap Nails', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 30, maxRange: 18,
    baseDamage: 14, minDamage: 7, falloffStartDist: 8, spreadAngle: 0.07,
    fireRate: 3.0, noiseRadius: 24, bulletColor: '#e2e8f0', bulletRadius: 0.09,
  }),
  gunTool('bark_revolver', 'Bark Revolver', cs('#78350f', '#92400e', '#f59e0b'), 30, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 34, maxRange: 22,
    baseDamage: 30, minDamage: 15, falloffStartDist: 10, spreadAngle: 0.05,
    fireRate: 1.4, noiseRadius: 30, penetration: 1, bulletColor: '#fb923c', bulletRadius: 0.1,
  }),
  gunTool('foam_dart_blaster', 'Foam Dart Blaster', cs('#22d3ee', '#0f766e', '#fde047'), 3, {
    ammoItem: 'foam_darts', ammoName: 'Foam Darts', ammoPerShot: 1,
    aimSystem: 'burst', projectileSpeed: 18, maxRange: 12,
    baseDamage: 3, minDamage: 2, falloffStartDist: 5, spreadAngle: 0.09,
    fireRate: 4.0, noiseRadius: 3, bulletColor: '#67e8f9', bulletRadius: 0.12,
  }),
  gunTool('cork_crossbow', 'Cork Crossbow', cs('#166534', '#14532d', '#facc15'), 20, {
    ammoItem: 'scrap_nails', ammoName: 'Cork Bolts', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 28, maxRange: 26,
    baseDamage: 20, minDamage: 12, falloffStartDist: 12, spreadAngle: 0.02,
    fireRate: 0.7, noiseRadius: 3, penetration: 1, bulletColor: '#84cc16', bulletRadius: 0.1,
  }),
  gunTool('chicken_cannon', 'Chicken Cannon', cs('#fb923c', '#7c2d12', '#fef08a'), 12, {
    ammoItem: 'wild_carrot', ammoName: 'Bird Feed Cartridges', ammoPerShot: 1,
    aimSystem: 'arc', projectileSpeed: 12, maxRange: 20,
    baseDamage: 12, minDamage: 8, falloffStartDist: 10, spreadAngle: 0.06,
    fireRate: 0.7, noiseRadius: 12, arced: true, bulletColor: '#fdba74', bulletRadius: 0.18,
  }),
  gunTool('garage_lmg', 'Garage Plate LMG', cs('#0f172a', '#334155', '#22d3ee'), 19, {
    ammoItem: 'scrap_nails', ammoName: 'Scrap Nails', ammoPerShot: 1,
    aimSystem: 'bloom', projectileSpeed: 32, maxRange: 30,
    baseDamage: 19, minDamage: 9, falloffStartDist: 14, spreadAngle: 0.12,
    fireRate: 7.0, noiseRadius: 30, penetration: 1, bulletColor: '#cbd5e1', bulletRadius: 0.08,
  }),
  gunTool('pipe_dmr', 'Pipework DMR', cs('#1f2937', '#4b5563', '#fbbf24'), 42, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 50, maxRange: 52,
    baseDamage: 42, minDamage: 24, falloffStartDist: 24, spreadAngle: 0.01,
    fireRate: 1.6, noiseRadius: 29, penetration: 2, bulletColor: '#fde68a', bulletRadius: 0.09,
  }),
  gunTool('double_barrel', 'Double Barrel Thunder', cs('#111827', '#6b7280', '#f59e0b'), 22, {
    ammoItem: 'shotgun_shells', ammoName: '12-Gauge Buckshot', ammoPerShot: 2,
    aimSystem: 'bloom', projectileSpeed: 24, maxRange: 14,
    baseDamage: 22, minDamage: 7, falloffStartDist: 5, spreadAngle: 0.28,
    fireRate: 0.55, noiseRadius: 34, pelletCount: 8, bulletColor: '#fb923c', bulletRadius: 0.1,
  }),
  gunTool('urban_carbine', 'Urban Carbine', cs('#1d4ed8', '#1e3a8a', '#93c5fd'), 24, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 1,
    aimSystem: 'burst', projectileSpeed: 40, maxRange: 34,
    baseDamage: 24, minDamage: 12, falloffStartDist: 16, spreadAngle: 0.06,
    fireRate: 3.8, noiseRadius: 26, penetration: 1, bulletColor: '#93c5fd', bulletRadius: 0.09,
  }),
  gunTool('handmade_mortar', 'Handmade Bean Mortar', cs('#7f1d1d', '#991b1b', '#fca5a5'), 54, {
    ammoItem: 'canned_beans', ammoName: 'Bean Can Shells', ammoPerShot: 1,
    aimSystem: 'arc', projectileSpeed: 11, maxRange: 24,
    baseDamage: 54, minDamage: 20, falloffStartDist: 12, spreadAngle: 0.08,
    fireRate: 0.35, noiseRadius: 40, arced: true, bulletColor: '#f87171', bulletRadius: 0.22,
  }),
  gunTool('rail_spike_driver', 'Rail Spike Driver', cs('#3f3f46', '#71717a', '#a3a3a3'), 62, {
    ammoItem: 'scrap_nails', ammoName: 'Rail Spikes', ammoPerShot: 3,
    aimSystem: 'laser', projectileSpeed: 58, maxRange: 60,
    baseDamage: 62, minDamage: 36, falloffStartDist: 28, spreadAngle: 0.012,
    fireRate: 0.5, noiseRadius: 42, penetration: 3, bulletColor: '#d4d4d8', bulletRadius: 0.11,
  }),
  gunTool('bottle_rocket_launcher', 'Bottle Rocket Launcher', cs('#ef4444', '#dc2626', '#f59e0b'), 36, {
    ammoItem: 'road_flare', ammoName: 'Road Flares', ammoPerShot: 1,
    aimSystem: 'arc', projectileSpeed: 15, maxRange: 26,
    baseDamage: 36, minDamage: 14, falloffStartDist: 13, spreadAngle: 0.07,
    fireRate: 0.65, noiseRadius: 35, arced: true, bulletColor: '#fb7185', bulletRadius: 0.18,
  }),
  gunTool('garage_burst_pdw', 'Garage Burst PDW', cs('#0f766e', '#134e4a', '#5eead4'), 16, {
    ammoItem: 'scrap_nails', ammoName: 'Scrap Nails', ammoPerShot: 1,
    aimSystem: 'burst', projectileSpeed: 29, maxRange: 24,
    baseDamage: 16, minDamage: 8, falloffStartDist: 11, spreadAngle: 0.09,
    fireRate: 6.2, noiseRadius: 15, bulletColor: '#5eead4', bulletRadius: 0.08,
  }),
  gunTool('street_sweeper', 'Street Sweeper Auto Shotgun', cs('#52525b', '#27272a', '#f97316'), 12, {
    ammoItem: 'shotgun_shells', ammoName: '12-Gauge Buckshot', ammoPerShot: 1,
    aimSystem: 'bloom', projectileSpeed: 20, maxRange: 13,
    baseDamage: 12, minDamage: 4, falloffStartDist: 5, spreadAngle: 0.3,
    fireRate: 2.3, noiseRadius: 33, pelletCount: 6, bulletColor: '#fdba74', bulletRadius: 0.09,
  }),
  gunTool('silent_needle', 'Silent Needle Carbine', cs('#4338ca', '#312e81', '#c4b5fd'), 18, {
    ammoItem: 'foam_darts', ammoName: 'Needle Darts', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 35, maxRange: 28,
    baseDamage: 18, minDamage: 10, falloffStartDist: 12, spreadAngle: 0.025,
    fireRate: 1.8, noiseRadius: 4, penetration: 1, bulletColor: '#c4b5fd', bulletRadius: 0.08,
  }),
  gunTool('junkyard_sniper', 'Junkyard Sniper', cs('#111827', '#374151', '#e5e7eb'), 78, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 2,
    aimSystem: 'laser', projectileSpeed: 64, maxRange: 68,
    baseDamage: 78, minDamage: 42, falloffStartDist: 34, spreadAngle: 0.008,
    fireRate: 0.32, noiseRadius: 46, penetration: 3, bulletColor: '#f9fafb', bulletRadius: 0.1,
  }),
  gunTool('flare_burner', 'Flare Burner', cs('#b91c1c', '#7f1d1d', '#f97316'), 10, {
    ammoItem: 'road_flare', ammoName: 'Road Flares', ammoPerShot: 1,
    aimSystem: 'burst', projectileSpeed: 18, maxRange: 18,
    baseDamage: 10, minDamage: 6, falloffStartDist: 7, spreadAngle: 0.14,
    fireRate: 3.2, noiseRadius: 20, bulletColor: '#fb923c', bulletRadius: 0.14,
  }),
  gunTool('beanbag_launcher', 'Beanbag Crowd Launcher', cs('#059669', '#065f46', '#99f6e4'), 26, {
    ammoItem: 'canned_beans', ammoName: 'Beanbag Rounds', ammoPerShot: 1,
    aimSystem: 'arc', projectileSpeed: 14, maxRange: 19,
    baseDamage: 26, minDamage: 12, falloffStartDist: 9, spreadAngle: 0.1,
    fireRate: 1.1, noiseRadius: 16, arced: true, bulletColor: '#34d399', bulletRadius: 0.16,
  }),
  gunTool('drillgun', 'Rotary Drillgun', cs('#f59e0b', '#b45309', '#1f2937'), 28, {
    ammoItem: 'scrap_nails', ammoName: 'Hardened Drill Nails', ammoPerShot: 2,
    aimSystem: 'bloom', projectileSpeed: 27, maxRange: 21,
    baseDamage: 28, minDamage: 13, falloffStartDist: 10, spreadAngle: 0.12,
    fireRate: 4.2, noiseRadius: 27, penetration: 2, bulletColor: '#fbbf24', bulletRadius: 0.1,
  }),
  gunTool('coil_carbine', 'Coil Capacitor Carbine', cs('#0ea5e9', '#0369a1', '#e0f2fe'), 27, {
    ammoItem: 'turret_ammo', ammoName: 'Coil Slugs', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 42, maxRange: 38,
    baseDamage: 27, minDamage: 14, falloffStartDist: 18, spreadAngle: 0.045,
    fireRate: 2.6, noiseRadius: 22, bulletColor: '#7dd3fc', bulletRadius: 0.1,
  }),
  tool('air_horn', 'Canister Air Horn', 'mainHand', 'weapon', cs('#ef4444', '#f97316', '#fde047'), 0, 1.0, 8.5, 0),
  // off hand
  tool('lantern', 'Lantern', 'offHand', 'none', cs('#3c4146', '#ffd27a', '#ff9a3c'), 0, 1, 1, 0, { lightRadius: 6 }),
  tool('torch_off', 'Torch (off hand)', 'offHand', 'torch', cs('#6f4f33', '#ff9a3c', '#ffd27a'), 0, 1, 1, 0, { lightRadius: 5 }),
  tool('pot_lid', 'Pot Lid Shield', 'offHand', 'none', cs('#8d949b', '#5f666d', '#c9c3b2'), 0, 1, 1, 0),
  // head
  gear('beanie', 'Beanie', 'head', cs('#3c4f4a', '#c9c3b2', '#b5651d')), gear('hood', 'Hood', 'head', cs('#3a3f45', '#2b2f34', '#4a4f55')),
  gear('cap', 'Baseball Cap', 'head', cs('#2f4660', '#c9c3b2', '#b5651d')), gear('bucket_hat', 'Bucket Hat', 'head', cs('#6b6a4a', '#4d4c36', '#c9c3b2')),
  gear('cone_hat', 'Traffic Cone Hat', 'head', cs('#c9641d', '#d8d4c8', '#8a4a1d')), gear('straw_hat', 'Straw Hat', 'head', cs('#c9b06a', '#8a7a46', '#b5441d')),
  gear('pot_helmet', 'Pot Helmet', 'head', cs('#8d949b', '#5f666d', '#c9c3b2')), gear('bandana', 'Bandana', 'head', cs('#8a3a3a', '#d8d4c8', '#2b2b2e')),
  gear('propeller_beanie', 'Propeller Beanie', 'head', cs('#3b82f6', '#ef4444', '#eab308')),
  // face
  gear('goggles', 'Goggles', 'face', cs('#5a4a32', '#9ac6d6', '#2b2b2e')), gear('round_glasses', 'Round Glasses', 'face', cs('#2b2b2e', '#bcd4de', '#2b2b2e')),
  gear('fake_mustache', 'Fake Mustache', 'face', cs('#2b1d16', '#4a3022', '#15110f')),
  // neck
  gear('scarf', 'Scarf', 'neck', cs('#8a4a3a', '#c9c3b2', '#5a2f26')),
  // torso
  gear('hoodie', 'Hoodie Jacket', 'torso', cs('#3a3f45', '#4a4f55', '#2b2f34')), gear('tshirt', 'T-Shirt', 'torso', cs('#6b7a8a', '#4a5560', '#c9c3b2')),
  gear('trench_coat', 'Trench Coat', 'torso', cs('#6b5a3c', '#4a3f2b', '#2b2b2e')), gear('poncho', 'Rain Poncho', 'torso', cs('#a8a24a', '#7a752f', '#4a4818')),
  gear('hivis_vest', 'Hi-Vis Vest', 'torso', cs('#c98a1d', '#4a4f55', '#d8d4c8')), gear('sweater', 'Knit Sweater', 'torso', cs('#8a3a4a', '#6a2a38', '#c9c3b2')),
  gear('pillow_vest', 'Feather Pillow Vest', 'torso', cs('#f8fafc', '#e2e8f0', '#94a3b8')),
  // legs
  gear('jeans', 'Jeans', 'legs', cs('#2f4660', '#243649', '#8aa0b4')), gear('cargo_pants', 'Cargo Pants', 'legs', cs('#4d5a3a', '#3a452c', '#2b2b2e')),
  gear('shorts', 'Shorts', 'legs', cs('#6b5a3c', '#4a3f2b', '#c9c3b2')), gear('overalls', 'Overalls', 'legs', cs('#3c5a8a', '#2b4466', '#c9b06a')),
  // feet
  gear('boots', 'Work Boots', 'feet', cs('#3a2f28', '#241c17', '#c9c3b2')), gear('sneakers', 'Sneakers', 'feet', cs('#c9c3b2', '#2f4660', '#b5441d')),
  gear('wellies', 'Wellies', 'feet', cs('#2f5a3c', '#1f3d29', '#c9c3b2')),
  gear('boing_boots', 'Spring Boing Boots', 'feet', cs('#10b981', '#059669', '#f59e0b')),
  // hands
  gear('leather_gloves', 'Leather Gloves', 'hands', cs('#4a3a2c', '#2f241b', '#8a7a62')), gear('mittens', 'Mittens', 'hands', cs('#8a3a4a', '#c9c3b2', '#6a2a38')),
  // back
  gear('hiker_pack', 'Hiker Pack', 'back', cs('#4a4f3a', '#6b6450', '#2b2b2e')), gear('school_pack', 'School Backpack', 'back', cs('#8a3a3a', '#2b2b2e', '#c9c3b2')),
  gear('satchel', 'Satchel', 'back', cs('#6b4a32', '#4a3322', '#c9b06a')),
  // belt
  gear('tool_belt', 'Tool Belt', 'belt', cs('#5a4a32', '#8a7a62', '#9aa0a6')),
  gear('fanny_pack_of_holding', 'Fanny Pack of Holding', 'belt', cs('#ec4899', '#a855f7', '#38bdf8')),
  // W4B Showcase Outfits: Breacher, Welder & Duck Hazmat
  gear('breacher_helmet', 'Tactical Breacher Helmet', 'head', cs('#1e293b', '#334155', '#38bdf8')),
  gear('breacher_vest', 'Heavy Breacher Plate Carrier', 'torso', cs('#334155', '#1e293b', '#ea580c')),
  gear('welder_mask', 'Industrial Welder Mask', 'head', cs('#1e293b', '#475569', '#10b981')),
  gear('welder_apron', 'Heavy Leather Welder Apron', 'torso', cs('#78350f', '#b45309', '#f59e0b')),
  gear('hazmat_hood', 'Duck Hazmat Filter Hood', 'head', cs('#facc15', '#eab308', '#38bdf8')),
  gear('hazmat_suit', 'Sealed Duck Hazmat Suit', 'torso', cs('#facc15', '#1e293b', '#eab308')),
  gear('quack_pak', 'Pressurized Quack-Pak', 'back', cs('#facc15', '#ea580c', '#38bdf8')),
  // W4B Showcase Weapons: Recon Marksman DMR & Pneumatic Scrap Nailgun
  gunTool('recon_rifle', 'Recon Marksman DMR', cs('#1e293b', '#475569', '#38bdf8'), 68, {
    ammoItem: 'rifle_ammo', ammoName: 'Rifle Cartridges', ammoPerShot: 1,
    aimSystem: 'laser', projectileSpeed: 52, maxRange: 55,
    baseDamage: 68, minDamage: 40, falloffStartDist: 30, spreadAngle: 0.008,
    fireRate: 1.2, noiseRadius: 32, penetration: 3, bulletColor: '#38bdf8', bulletRadius: 0.11,
  }),
  gunTool('scrap_nailgun', 'Pneumatic Scrap Nail Driver', cs('#ea580c', '#1e293b', '#facc15'), 18, {
    ammoItem: 'scrap_nails', ammoName: 'Scrap Nails', ammoPerShot: 1,
    aimSystem: 'bloom', projectileSpeed: 28, maxRange: 22,
    baseDamage: 18, minDamage: 9, falloffStartDist: 12, spreadAngle: 0.08,
    fireRate: 6.5, noiseRadius: 12, bulletColor: '#facc15', bulletRadius: 0.09,
  }),
];
