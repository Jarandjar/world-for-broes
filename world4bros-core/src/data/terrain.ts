import type { TerrainDef } from './schema';
/** Vibrant & stylized indie ground palette. colors = [base, shadow, highlight]. */
export const TERRAIN: TerrainDef[] = [
  { id: 'grass', walkable: true, speed: 1, colors: ['#3b823e', '#285e2b', '#5ec763'], strokes: 'blades', tags: ['flammable'] },
  { id: 'moss', walkable: true, speed: 0.95, colors: ['#2d6a4f', '#1b4332', '#40916c'], strokes: 'blades' },
  { id: 'dirt', walkable: true, speed: 1, colors: ['#78593d', '#523c28', '#9e7550'], strokes: 'pebbles' },
  { id: 'needles', walkable: true, speed: 1, colors: ['#624a35', '#453323', '#876749'], strokes: 'needles' },
  { id: 'mud', walkable: true, speed: 0.6, colors: ['#443528', '#2c2219', '#5f4b39'], strokes: 'waves' },
  { id: 'rock', walkable: true, speed: 0.95, colors: ['#5a6577', '#3c4554', '#7c8ba3'], strokes: 'cracks' },
  { id: 'sand', walkable: true, speed: 0.9, colors: ['#d99b45', '#ad7329', '#f5be6c'], strokes: 'pebbles' },
  { id: 'snow', walkable: true, speed: 0.85, colors: ['#e2effa', '#9ec0db', '#ffffff'], strokes: 'flakes' },
  { id: 'ash', walkable: true, speed: 0.95, colors: ['#34333b', '#1e1e23', '#545260'], strokes: 'cracks' },
  { id: 'shallow', walkable: true, speed: 0.55, colors: ['#06b6d4', '#0891b2', '#67e8f9'], strokes: 'waves', tags: ['wet'] },
  { id: 'deep', walkable: false, speed: 0, colors: ['#0f3b4c', '#08232e', '#164e63'], strokes: 'waves', tags: ['wet'] },
];
