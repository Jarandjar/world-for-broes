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
