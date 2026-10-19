import type { StructureDef } from './schema';
export const STRUCTURES: StructureDef[] = [
  { id: 'claim_flag', name: 'Claim Flag', size: [1, 1], solid: true, hp: 30, cost: { wood: 2, cloth: 1 }, requiresClaim: false, claimRadius: 12 },
  { id: 'campfire', name: 'Campfire', size: [1, 1], solid: false, hp: 20, cost: { wood: 3, stone: 2 }, requiresClaim: false, light: { radius: 6.5, color: '#ff9a3c', flicker: 0.12 }, interact: 'fire' },
  { id: 'wall_wood', name: 'Wooden Wall', size: [1, 1], solid: true, hp: 60, cost: { wood: 3 }, requiresClaim: true },
  { id: 'door_wood', name: 'Wooden Door', size: [1, 1], solid: true, hp: 40, cost: { wood: 4 }, requiresClaim: true, interact: 'door' },
  { id: 'floor_wood', name: 'Wooden Floor', size: [1, 1], solid: false, hp: 20, cost: { wood: 1 }, requiresClaim: true },
  { id: 'chest', name: 'Storage Chest', size: [1, 1], solid: true, hp: 30, cost: { wood: 4 }, requiresClaim: true, interact: 'storage', slots: 20 },
  { id: 'bed', name: 'Bedroll Bed', size: [1, 2], solid: false, hp: 20, cost: { cloth: 3, wood: 2 }, requiresClaim: true, interact: 'sleep' },
  { id: 'workbench', name: 'Workbench', size: [2, 1], solid: true, hp: 40, cost: { wood: 6, scrap_metal: 2 }, requiresClaim: true, interact: 'craft' },
  { id: 'torch_post', name: 'Torch Post', size: [1, 1], solid: true, hp: 15, cost: { wood: 2, twig: 2 }, requiresClaim: false, light: { radius: 4.5, color: '#ffb35a', flicker: 0.18 } },
  // automation structures
  { id: 'conveyor_belt', name: 'Conveyor Belt', size: [1, 1], solid: false, hp: 25, cost: { wood: 1, scrap_metal: 1 }, requiresClaim: false },
  { id: 'spring_inserter', name: 'Spring Inserter', size: [1, 1], solid: true, hp: 30, cost: { scrap_metal: 2, twig: 2 }, requiresClaim: false },
  { id: 'burner_drill', name: 'Burner Mining Drill', size: [2, 2], solid: true, hp: 80, cost: { scrap_metal: 4, stone: 4, wood: 2 }, requiresClaim: false },
  { id: 'stone_furnace', name: 'Smelting Furnace', size: [2, 2], solid: true, hp: 70, cost: { stone: 6, wood: 2 }, requiresClaim: false },
  // colony structures (RimWorld)
  { id: 'grand_banquet_table', name: 'Grand Dining Table', size: [2, 1], solid: true, hp: 45, cost: { wood: 6 }, requiresClaim: true, interact: 'table' },
  { id: 'medical_triage_cot', name: 'Medical Triage Cot', size: [1, 2], solid: false, hp: 35, cost: { wood: 4, cloth: 2, scrap_metal: 2 }, requiresClaim: true, interact: 'sleep' },
  { id: 'horseshoe_pin_post', name: 'Horseshoe Pin Game', size: [1, 1], solid: true, hp: 20, cost: { wood: 2, scrap_metal: 1 }, requiresClaim: false },
  { id: 'colony_overseer_desk', name: 'Colony Overseer Desk', size: [2, 1], solid: true, hp: 40, cost: { wood: 6, scrap_metal: 2 }, requiresClaim: true, interact: 'overseer' },
  // base defense structures (Phase Z16)
  { id: 'belt_bolter', name: 'Belt-Fed Bolter Turret', size: [1, 1], solid: true, hp: 65, cost: { wood: 4, scrap_metal: 4 }, requiresClaim: true, interact: 'turret' },
  { id: 'flame_turret', name: 'Pressurized Flame Turret', size: [1, 1], solid: true, hp: 80, cost: { scrap_metal: 6, stone: 4 }, requiresClaim: true, interact: 'turret' },
  { id: 'zap_fence', name: 'Electrified Zap-Fence', size: [1, 1], solid: true, hp: 50, cost: { scrap_metal: 3, wood: 1 }, requiresClaim: true },
  { id: 'mallet_catapult', name: 'Spring Mallet Catapult', size: [2, 2], solid: true, hp: 90, cost: { wood: 8, stone: 4, scrap_metal: 3 }, requiresClaim: true },
  { id: 'repair_drone_bay', name: 'Autonomous Repair Drone Bay', size: [2, 2], solid: true, hp: 95, cost: { scrap_metal: 8, energetic_crystal: 2, wood: 4 }, requiresClaim: true },
  // faction & territory structures (Phase Z17)
  { id: 'clan_territory_totem', name: 'Clan Territory Totem', size: [2, 2], solid: true, hp: 220, cost: { stone: 10, scrap_metal: 6, energetic_crystal: 4 }, requiresClaim: false, claimRadius: 30, interact: 'totem' },
  // weapons modification (Gunsmith)
  { id: 'gunsmith_bench', name: 'Gunsmith Mod Workbench', size: [2, 1], solid: true, hp: 50, cost: { wood: 5, scrap_metal: 4 }, requiresClaim: false, interact: 'gunsmith' },
];
