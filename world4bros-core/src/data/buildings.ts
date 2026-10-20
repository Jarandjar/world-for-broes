import type { BuildingDef } from './schema';
import { hash2 } from '../engine/rng';

/** Town buildings. Add a building style by adding an entry; the town generator picks by weight. */
export const BUILDINGS: BuildingDef[] = [
  {
    id: 'cottage', name: 'Cozy Cottage', kind: 'house', weight: 4,
    minW: 7, maxW: 9, minH: 6, maxH: 8,
    roof: 'gable', walls: 'plank', door: 'single', layout: 'split',
    features: ['porch', 'chimney', 'flowerBoxes', 'mailbox', 'fence'],
    colors: { wall: '#b08a63', roof: '#a8432f', floor: '#7a5a3b', trim: '#e9dcc0' },
    palettes: [
      { wall: '#d9c48a', roof: '#2f6aa3', trim: '#f6f1e3' },
      { wall: '#9fb59a', roof: '#4f7d36', trim: '#f0ece1' },
      { wall: '#e3c9b0', roof: '#7c4a8c', trim: '#fbf6ea' },
      { wall: '#a9c3cf', roof: '#c0612b', trim: '#f4efe2' },
      { wall: '#e6dcc6', roof: '#2f7f7a', trim: '#8a5a3c' },
    ],
  },
  {
    id: 'log_cabin', name: 'Log Cabin', kind: 'house', weight: 3, zone: 'outskirts',
    minW: 8, maxW: 10, minH: 7, maxH: 9,
    roof: 'hip', walls: 'log', door: 'single', layout: 'split',
    features: ['porch', 'chimney', 'garden', 'fence'],
    colors: { wall: '#7d5836', roof: '#4d6b34', floor: '#5c4028', trim: '#cbb69d' },
    palettes: [
      { wall: '#8a6040', roof: '#9a3b2a', trim: '#d8c3a5' },
      { wall: '#6f4e33', roof: '#3f5f7d', trim: '#c9b48f' },
      { wall: '#946b45', roof: '#7a6a2e', trim: '#e0cdab' },
    ],
  },
  {
    id: 'townhouse', name: 'Brick Townhouse', kind: 'house', weight: 3, zone: 'core',
    minW: 7, maxW: 9, minH: 8, maxH: 10,
    roof: 'hip', walls: 'brick', door: 'single', layout: 'split',
    features: ['balcony', 'porch', 'flowerBoxes', 'mailbox', 'fence', 'garden'],
    colors: { wall: '#a85848', roof: '#3d4f6e', floor: '#8a6d4a', trim: '#f1ead8' },
    palettes: [
      { wall: '#944d42', roof: '#2f6b4f', trim: '#e0d8c3' },
      { wall: '#b9785a', roof: '#8a3a3a', trim: '#f4ecd8' },
      { wall: '#c9b79c', roof: '#5a4a7a', trim: '#fffaf0' },
    ],
  },
  {
    id: 'corner_shop', name: 'Corner Grocery Store', kind: 'shop', weight: 4, zone: 'core',
    minW: 8, maxW: 11, minH: 7, maxH: 9,
    roof: 'gable', walls: 'plaster', door: 'double', layout: 'backroom',
    features: ['awning', 'sign', 'glassFront'],
    colors: { wall: '#9aa8a0', roof: '#2f6f73', floor: '#b9b3a2', trim: '#f4f1e6' },
    palettes: [
      { wall: '#e0cf9e', roof: '#b5452f', trim: '#fffaf0' },
      { wall: '#c7a9a0', roof: '#35577d', trim: '#f4f1e6' },
    ],
  },
  {
    id: 'supermarket', name: 'Supermarket Megastore', kind: 'supermarket', weight: 2, zone: 'core',
    minW: 10, maxW: 14, minH: 8, maxH: 11,
    roof: 'flat', walls: 'concrete', door: 'double', layout: 'backroom',
    features: ['glassFront', 'sign', 'hvac', 'awning'],
    colors: { wall: '#8f949d', roof: '#3a3f47', floor: '#c0c5cc', trim: '#f59e0b' },
  },
  {
    id: 'auto_garage', name: 'Auto Repair Garage', kind: 'garage', weight: 2,
    minW: 8, maxW: 11, minH: 8, maxH: 10,
    roof: 'tin', walls: 'concrete', door: 'rollup', layout: 'none',
    features: ['sign', 'tarp'],
    colors: { wall: '#7a7d85', roof: '#4b4f58', floor: '#5d6068', trim: '#d9a441' },
    palettes: [
      { wall: '#8a7f70', roof: '#7a3b30', trim: '#e0b14a' },
    ],
  },
  {
    id: 'clinic', name: 'Emergency Clinic', kind: 'clinic', weight: 1, zone: 'core',
    minW: 8, maxW: 10, minH: 8, maxH: 10,
    roof: 'hip', walls: 'plaster', door: 'double', layout: 'apartment',
    features: ['sign', 'awning'],
    colors: { wall: '#e3e6e8', roof: '#b83b3b', floor: '#cfd8dc', trim: '#ffffff' },
  },
  {
    id: 'warehouse', name: 'Industrial Warehouse', kind: 'warehouse', weight: 1, zone: 'outskirts',
    minW: 9, maxW: 12, minH: 7, maxH: 10,
    roof: 'tin', walls: 'scrap', door: 'rollup', layout: 'none',
    features: ['hvac', 'tarp', 'sign'],
    colors: { wall: '#8a7a66', roof: '#5f6b5a', floor: '#6e6556', trim: '#c9b88f' },
  },
  {
    id: 'tool_shed', name: 'Tool & Storage Shed', kind: 'shed', weight: 4, zone: 'outskirts',
    minW: 5, maxW: 6, minH: 5, maxH: 6,
    roof: 'tin', walls: 'plank', door: 'single', layout: 'none',
    features: ['garden', 'tarp'],
    colors: { wall: '#6e5d4e', roof: '#50565e', floor: '#4a3f35', trim: '#a89882' },
    palettes: [
      { wall: '#7d6a4f', roof: '#6b4a3a', trim: '#b8a586' },
      { wall: '#5f6e5a', roof: '#4e6a7a', trim: '#a8a38a' },
    ],
  },
  {
    id: 'survivor_shack', name: 'Survivor Barricaded Shack', kind: 'house', weight: 3, zone: 'outskirts',
    minW: 6, maxW: 8, minH: 5, maxH: 7,
    roof: 'broken', walls: 'scrap', door: 'single', layout: 'none',
    features: ['boarded', 'tarp', 'chimney', 'vines'],
    colors: { wall: '#594d42', roof: '#5e4a3c', floor: '#453c35', trim: '#826f5d' },
    palettes: [
      { wall: '#5f5a48', roof: '#4a5a46', trim: '#8a7a5d' },
      { wall: '#61493d', roof: '#6a4038', trim: '#8f735d' },
    ],
  },
  {
    id: 'apartment_block', name: 'Residential Apartments', kind: 'apartment', weight: 2, zone: 'core',
    minW: 9, maxW: 11, minH: 9, maxH: 12,
    roof: 'flat', walls: 'brick', door: 'double', layout: 'apartment',
    features: ['balcony', 'hvac', 'awning'],
    colors: { wall: '#8b5a45', roof: '#2e333d', floor: '#827464', trim: '#e8dcce' },
    palettes: [
      { wall: '#a0806a', roof: '#3a3f35', trim: '#f0e6d6' },
    ],
  },
];

/** Seed mixed into the palette hash; building positions already differ per world, so this only decorrelates it from other hashes. */
export const PALETTE_SEED = 0x7a11e;

/**
 * The colour set one building uses: `colors` or one of its `palettes` (overriding wall/roof/trim), picked by
 * hash2 of its footprint corner. Deterministic, so a house keeps its colours across reloads and in tests.
 */
export function buildingColors(d: BuildingDef, x: number, y: number, seed = PALETTE_SEED): BuildingDef['colors'] {
  const alts = d.palettes ?? [];
  const i = (hash2(seed, x, y) >>> 0) % (alts.length + 1);
  return i === 0 ? d.colors : { ...d.colors, ...alts[i - 1]! };
}
