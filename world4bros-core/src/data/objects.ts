import type { ObjectDef, ObjectKind, SurfaceTag } from './schema';
export const OBJECTS: ObjectDef[] = [
  { kind: 'pine', name: 'Pine Tree', radius: 0.45, solid: true, hp: 5, tool: 'axe', drops: [{ item: 'wood', min: 2, max: 3 }, { item: 'pine_cone', min: 0, max: 1 }], becomes: 'stump', prompt: 'Chop Pine Tree' },
  { kind: 'snow_pine', name: 'Snowy Pine', radius: 0.45, solid: true, hp: 5, tool: 'axe', drops: [{ item: 'wood', min: 2, max: 3 }], becomes: 'stump', prompt: 'Chop Snowy Pine' },
  { kind: 'oak', name: 'Oak Tree', radius: 0.5, solid: true, hp: 6, tool: 'axe', drops: [{ item: 'wood', min: 3, max: 4 }, { item: 'acorn', min: 0, max: 2 }], becomes: 'stump', prompt: 'Chop Oak Tree' },
  { kind: 'dead_tree', name: 'Dead Tree', radius: 0.35, solid: true, hp: 2, tool: 'axe', drops: [{ item: 'twig', min: 2, max: 4 }], prompt: 'Chop Dead Tree' },
  { kind: 'bush', name: 'Bush', radius: 0.3, solid: false, hp: 1, tool: 'hands', drops: [{ item: 'twig', min: 1, max: 2 }, { item: 'fiber', min: 0, max: 1 }], prompt: 'Pick Bush' },
  { kind: 'berry_bush', name: 'Berry Bush', radius: 0.3, solid: false, hp: 1, tool: 'hands', drops: [{ item: 'berries', min: 2, max: 3 }], becomes: 'bush', prompt: 'Pick Berries' },
  { kind: 'rock', name: 'Rock', radius: 0.3, solid: true, hp: 4, tool: 'pickaxe', drops: [{ item: 'stone', min: 1, max: 2 }, { item: 'flint', min: 0, max: 1 }], prompt: 'Mine Rock' },
  { kind: 'boulder', name: 'Boulder', radius: 0.7, solid: true, hp: 8, tool: 'pickaxe', drops: [{ item: 'stone', min: 3, max: 5 }, { item: 'flint', min: 1, max: 2 }], becomes: 'rock', prompt: 'Mine Boulder' },
  { kind: 'log', name: 'Fallen Log', radius: 0.35, solid: true, hp: 2, tool: 'axe', drops: [{ item: 'wood', min: 2, max: 2 }], prompt: 'Chop Log' },
  { kind: 'stump', name: 'Stump', radius: 0.3, solid: true, hp: 99, tool: 'none', drops: [], prompt: 'Stump' },
  { kind: 'mushroom', name: 'Mushrooms', radius: 0.2, solid: false, hp: 1, tool: 'hands', drops: [{ item: 'mushroom', min: 1, max: 2 }], prompt: 'Pick Mushrooms' },
  { kind: 'reeds', name: 'Reeds', radius: 0.25, solid: false, hp: 1, tool: 'hands', drops: [{ item: 'fiber', min: 1, max: 2 }], prompt: 'Cut Reeds' },
  { kind: 'cactus', name: 'Cactus', radius: 0.3, solid: true, hp: 2, tool: 'axe', drops: [{ item: 'cactus_pulp', min: 1, max: 2 }], prompt: 'Chop Cactus' },
  { kind: 'crate', name: 'Old Crate', radius: 0.35, solid: true, hp: 2, tool: 'axe', drops: [{ item: 'wood', min: 1, max: 2 }, { item: 'scrap_metal', min: 0, max: 2 }, { item: 'cloth', min: 0, max: 1 }], prompt: 'Smash Crate' },
  { kind: 'tent', name: 'Abandoned Tent', radius: 0.6, solid: true, hp: 3, tool: 'hands', drops: [{ item: 'cloth', min: 1, max: 2 }, { item: 'twig', min: 1, max: 2 }], prompt: 'Salvage Tent' },
  { kind: 'fuel_pump', name: 'Fuel Pump', radius: 0.5, solid: true, hp: 4, tool: 'hands', drops: [{ item: 'fuel_canister', min: 1, max: 2 }, { item: 'scrap_metal', min: 1, max: 2 }], prompt: 'Scavenge Fuel Pump' },
  { kind: 'radio_tower', name: 'Radio Tower', radius: 0.9, solid: true, hp: 20, tool: 'hands', drops: [{ item: 'scrap_metal', min: 2, max: 4 }], prompt: 'Scan Radio Frequency' },
  { kind: 'sandbags', name: 'Sandbag Barrier', radius: 0.5, solid: true, hp: 4, tool: 'hands', drops: [{ item: 'cloth', min: 1, max: 2 }, { item: 'stone', min: 1, max: 2 }], prompt: 'Dismantle Sandbags' },
  { kind: 'wrecked_car', name: 'Wrecked Car', radius: 0.9, solid: true, hp: 6, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 4 }, { item: 'car_battery', min: 0, max: 1 }, { item: 'scrap_tire', min: 0, max: 1 }], prompt: 'Scavenge Wrecked Car' },
  // mineral geode nodes (Factorio / Satisfactory extraction)
  { kind: 'iron_node', name: 'Iron Mineral Node', radius: 0.65, solid: true, hp: 12, tool: 'pickaxe', drops: [{ item: 'iron_ore', min: 2, max: 4 }, { item: 'stone', min: 1, max: 2 }], prompt: 'Mine Iron Node' },
  { kind: 'copper_node', name: 'Copper Mineral Node', radius: 0.65, solid: true, hp: 10, tool: 'pickaxe', drops: [{ item: 'copper_ore', min: 2, max: 4 }, { item: 'stone', min: 1, max: 2 }], prompt: 'Mine Copper Node' },
  { kind: 'coal_node', name: 'Coal Seam', radius: 0.6, solid: true, hp: 8, tool: 'pickaxe', drops: [{ item: 'coal_chunk', min: 3, max: 5 }], prompt: 'Mine Coal Seam' },
  { kind: 'oil_seam', name: 'Crude Oil Seam', radius: 0.55, solid: false, hp: 4, tool: 'hands', drops: [{ item: 'crude_oil_canister', min: 1, max: 2 }], prompt: 'Extract Crude Oil' },
  { kind: 'crystal_geode', name: 'Glow Crystal Geode', radius: 0.7, solid: true, hp: 14, tool: 'pickaxe', drops: [{ item: 'energetic_crystal', min: 2, max: 4 }, { item: 'flint', min: 1, max: 2 }], prompt: 'Mine Glow Crystal Geode' },
  // big town & road infrastructure
  { kind: 'town_fountain', name: 'Town Square Fountain', radius: 1.1, solid: true, hp: 40, tool: 'none', drops: [{ item: 'stone', min: 2, max: 4 }], prompt: 'Rest at Town Fountain (+Sanity)' },
  { kind: 'street_light', name: 'Street Lamppost', radius: 0.35, solid: true, hp: 12, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 3 }, { item: 'copper_wire', min: 1, max: 2 }], prompt: 'Examine Lamppost' },
  { kind: 'guardrail', name: 'Highway Guardrail', radius: 0.45, solid: true, hp: 15, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 4 }], prompt: 'Dismantle Guardrail' },
  { kind: 'picket_fence', name: 'Picket Fence', radius: 0.35, solid: true, hp: 6, tool: 'axe', drops: [{ item: 'wood', min: 1, max: 2 }], prompt: 'Chop Picket Fence' },
  { kind: 'park_bench', name: 'Park Bench', radius: 0.5, solid: true, hp: 8, tool: 'none', drops: [{ item: 'wood', min: 2, max: 3 }], prompt: 'Sit on Park Bench' },
  { kind: 'fire_hydrant', name: 'Fire Hydrant', radius: 0.3, solid: true, hp: 10, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 1, max: 2 }], prompt: 'Inspect Hydrant' },
  { kind: 'dumpster', name: 'Alley Dumpster', radius: 0.75, solid: true, hp: 16, tool: 'none', drops: [{ item: 'scrap_metal', min: 2, max: 4 }, { item: 'cloth', min: 1, max: 2 }], prompt: 'Scavenge Dumpster' },
  { kind: 'traffic_cone', name: 'Traffic Warning Cone', radius: 0.25, solid: false, hp: 2, tool: 'none', drops: [{ item: 'scrap_metal', min: 0, max: 1 }], prompt: 'Kick Traffic Cone' },
  { kind: 'target_dummy', name: 'Ballistic Target Dummy', radius: 0.42, solid: true, hp: 999, tool: 'none', drops: [], prompt: 'Inspect Target Dummy' },
  { kind: 'steel_target', name: 'Steel Pop-Up Plate', radius: 0.38, solid: true, hp: 999, tool: 'none', drops: [], prompt: 'Inspect Steel Target' },
  { kind: 'slime_drum', name: 'Bio-Slime Pressure Drum', radius: 0.42, solid: true, hp: 6, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 1, max: 2 }], prompt: 'Examine Slime Drum' },
  { kind: 'scrap_wall', name: 'Corrugated Scrap Barricade', radius: 0.6, solid: true, hp: 18, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 4 }, { item: 'wood', min: 1, max: 2 }], prompt: 'Dismantle Scrap Wall' },
  { kind: 'tactical_shield', name: 'Aegis Deploy Shield', radius: 0.55, solid: true, hp: 25, tool: 'hands', drops: [{ item: 'scrap_metal', min: 2, max: 3 }], prompt: 'Inspect Tactical Shield' },
  // town street life, placed by townPropStage (variant = facing 0..3 + 4 * style, see TOWN_PROPS in sim/worldgen.ts)
  { kind: 'stop_sign', name: 'Stop Sign', radius: 0.18, solid: true, hp: 6, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 1, max: 1 }], prompt: 'Pull Down Stop Sign' },
  { kind: 'street_sign', name: 'Street Name Sign', radius: 0.18, solid: true, hp: 6, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 1, max: 1 }], prompt: 'Read Street Sign' },
  { kind: 'bus_stop', name: 'Bus Stop Shelter', radius: 0.7, solid: true, hp: 14, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 3 }, { item: 'wood', min: 0, max: 1 }], prompt: 'Rest at Bus Stop' },
  { kind: 'trash_can', name: 'Trash Can', radius: 0.24, solid: true, hp: 4, tool: 'hands', drops: [{ item: 'cloth', min: 0, max: 1 }, { item: 'scrap_metal', min: 0, max: 1 }], prompt: 'Rummage Trash Can' },
  { kind: 'power_pole', name: 'Power Pole', radius: 0.22, solid: true, hp: 10, tool: 'axe', drops: [{ item: 'wood', min: 2, max: 3 }, { item: 'copper_wire', min: 0, max: 1 }], prompt: 'Chop Power Pole' },
  { kind: 'abandoned_car', name: 'Abandoned Car', radius: 0.85, solid: true, hp: 8, tool: 'pickaxe', drops: [{ item: 'scrap_metal', min: 2, max: 4 }, { item: 'scrap_tire', min: 0, max: 2 }, { item: 'car_battery', min: 0, max: 1 }], prompt: 'Scavenge Abandoned Car' },
  { kind: 'litter', name: 'Street Litter', radius: 0.2, solid: false, hp: 1, tool: 'hands', drops: [{ item: 'cloth', min: 0, max: 1 }], prompt: 'Sift Through Litter' },
];

/** Surface tags for effect reactions (§116): fire spreads to flammable props, sparks set off explosive ones. */
const TAGS: Partial<Record<ObjectKind, SurfaceTag[]>> = {
  pine: ['flammable'], snow_pine: ['flammable'], oak: ['flammable'], dead_tree: ['flammable'], bush: ['flammable'], berry_bush: ['flammable'],
  log: ['flammable'], stump: ['flammable'], reeds: ['flammable'], crate: ['flammable'], tent: ['flammable'], picket_fence: ['flammable'], park_bench: ['flammable'],
  fuel_pump: ['explosive', 'metal'], wrecked_car: ['metal', 'conductive'], street_light: ['metal', 'conductive'], guardrail: ['metal', 'conductive'],
  fire_hydrant: ['metal', 'wet'], dumpster: ['metal'], radio_tower: ['metal', 'conductive'], slime_drum: ['explosive', 'metal'], scrap_wall: ['metal'],
  oil_seam: ['oil', 'flammable'],
  stop_sign: ['metal', 'conductive'], street_sign: ['metal', 'conductive'], bus_stop: ['metal'], trash_can: ['metal'],
  power_pole: ['flammable', 'conductive'], abandoned_car: ['metal', 'conductive'], litter: ['flammable'],
};
for (const o of OBJECTS) o.tags ??= TAGS[o.kind];

/**
 * Town street life: where townPropStage puts props. One row per rule; add a row to add a prop (docs/EXTENDING.md).
 * - `verge`: along street lines, every `every` tiles from `phase`, `inset` tiles outside the street edge.
 *   `side`: 'both' sides, 'alt'ernating, or 'one' (the +x / +y side).
 * - `corner`: at every intersection, in one corner (chosen per rule and intersection), `inset` outside both street edges.
 * - `scatter`: like verge but jittered along the street and in depth (litter).
 * `styles`: how many looks the drawer has (variant = facing 0..3 + 4 * style). `chance`: share of spots that get one.
 * Props never land on a street, sidewalk, plaza, water, building footprint or doorway (tests/town-props.test.ts).
 */
export interface TownPropRule {
  kind: ObjectKind; place: 'verge' | 'corner' | 'scatter';
  streets: 'main' | 'cross' | 'all'; every: number; phase: number; inset: number;
  side: 'both' | 'alt' | 'one'; chance: number; styles: number;
}
export const TOWN_PROP_RULES: TownPropRule[] = [
  { kind: 'street_sign', place: 'corner', streets: 'all', every: 0, phase: 0, inset: 0.35, side: 'both', chance: 0.95, styles: 4 },
  { kind: 'stop_sign', place: 'corner', streets: 'all', every: 0, phase: 0, inset: 0.35, side: 'both', chance: 0.85, styles: 3 },
  { kind: 'fire_hydrant', place: 'corner', streets: 'all', every: 0, phase: 0, inset: 0.75, side: 'both', chance: 0.7, styles: 4 },
  { kind: 'street_light', place: 'verge', streets: 'all', every: 8, phase: 4, inset: 0.4, side: 'alt', chance: 0.9, styles: 2 },
  { kind: 'power_pole', place: 'verge', streets: 'cross', every: 6, phase: 1, inset: 0.55, side: 'one', chance: 1, styles: 2 },
  { kind: 'bus_stop', place: 'verge', streets: 'main', every: 21, phase: 9.5, inset: 0.8, side: 'alt', chance: 0.85, styles: 4 },
  { kind: 'park_bench', place: 'verge', streets: 'main', every: 13, phase: 2.5, inset: 0.55, side: 'both', chance: 0.35, styles: 3 },
  { kind: 'trash_can', place: 'verge', streets: 'all', every: 9, phase: 6.5, inset: 0.35, side: 'both', chance: 0.4, styles: 4 },
  { kind: 'abandoned_car', place: 'verge', streets: 'all', every: 10, phase: 6, inset: 0.95, side: 'both', chance: 0.2, styles: 5 },
  { kind: 'litter', place: 'scatter', streets: 'all', every: 2.5, phase: 0.5, inset: 0.3, side: 'both', chance: 0.4, styles: 4 },
];