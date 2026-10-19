import type { BiomeDef } from './schema';
/** Biomes are picked by nearest climate (temperature, moisture, elevation). Add a biome by adding an entry. */
export const BIOMES: BiomeDef[] = [
  { id: 'bro_meadow', name: 'Bro Meadow', climate: { temp: 0.5, moist: 0.5, elev: 0.35 }, terrain: { base: 'grass', patch: 'dirt', patchAmount: 0.18 }, tint: '#4ade80',
    objects: [{ kind: 'oak', density: 0.012, minSpacing: 3 }, { kind: 'pine', density: 0.01, minSpacing: 3 }, { kind: 'bush', density: 0.02, minSpacing: 1.2 }, { kind: 'berry_bush', density: 0.008, minSpacing: 2 }, { kind: 'rock', density: 0.006, minSpacing: 2 }, { kind: 'mushroom', density: 0.004, minSpacing: 2 }, { kind: 'log', density: 0.002, minSpacing: 3 }] },
  { id: 'pinewood_thicket', name: 'Pinewood Thicket', climate: { temp: 0.4, moist: 0.62, elev: 0.5 }, terrain: { base: 'needles', patch: 'moss', patchAmount: 0.25 }, tint: '#0d9488',
    objects: [{ kind: 'pine', density: 0.07, minSpacing: 1.5 }, { kind: 'dead_tree', density: 0.008, minSpacing: 3 }, { kind: 'bush', density: 0.012, minSpacing: 1.2 }, { kind: 'mushroom', density: 0.008, minSpacing: 2 }, { kind: 'log', density: 0.005, minSpacing: 3 }, { kind: 'stump', density: 0.004, minSpacing: 3 }, { kind: 'rock', density: 0.004, minSpacing: 2 }] },
  { id: 'squelchy_bog', name: 'Squelchy Bog', climate: { temp: 0.5, moist: 0.9, elev: 0.22 }, terrain: { base: 'mud', patch: 'shallow', patchAmount: 0.3 }, tint: '#65a30d',
    objects: [{ kind: 'reeds', density: 0.05, minSpacing: 1 }, { kind: 'dead_tree', density: 0.02, minSpacing: 2.5 }, { kind: 'mushroom', density: 0.02, minSpacing: 1.5 }, { kind: 'bush', density: 0.01, minSpacing: 1.5 }] },
  { id: 'rockbottom_ridge', name: 'Rockbottom Ridge', climate: { temp: 0.35, moist: 0.3, elev: 0.85 }, terrain: { base: 'rock', patch: 'dirt', patchAmount: 0.2 }, tint: '#64748b',
    objects: [{ kind: 'rock', density: 0.03, minSpacing: 1.5 }, { kind: 'boulder', density: 0.012, minSpacing: 3 }, { kind: 'dead_tree', density: 0.006, minSpacing: 3 }, { kind: 'pine', density: 0.008, minSpacing: 3 }] },
  { id: 'sunburn_flats', name: 'Sunburn Flats', climate: { temp: 0.92, moist: 0.1, elev: 0.3 }, terrain: { base: 'sand', patch: 'dirt', patchAmount: 0.2 }, tint: '#f59e0b',
    objects: [{ kind: 'cactus', density: 0.012, minSpacing: 2.5 }, { kind: 'rock', density: 0.01, minSpacing: 2 }, { kind: 'dead_tree', density: 0.005, minSpacing: 4 }, { kind: 'bush', density: 0.006, minSpacing: 2 }] },
  { id: 'brrington_tundra', name: 'Brrington Tundra', climate: { temp: 0.05, moist: 0.4, elev: 0.6 }, terrain: { base: 'snow', patch: 'rock', patchAmount: 0.12 }, tint: '#38bdf8',
    objects: [{ kind: 'snow_pine', density: 0.02, minSpacing: 2.5 }, { kind: 'rock', density: 0.01, minSpacing: 2 }, { kind: 'dead_tree', density: 0.006, minSpacing: 3 }, { kind: 'boulder', density: 0.004, minSpacing: 4 }] },
  { id: 'scrapyard_gulch', name: 'Scrapyard Gulch', climate: { temp: 0.62, moist: 0.25, elev: 0.42 }, terrain: { base: 'dirt', patch: 'ash', patchAmount: 0.3 }, tint: '#ea580c',
    objects: [{ kind: 'crate', density: 0.02, minSpacing: 1.8 }, { kind: 'rock', density: 0.01, minSpacing: 2 }, { kind: 'dead_tree', density: 0.006, minSpacing: 3 }, { kind: 'tent', density: 0.002, minSpacing: 6 }] },
  { id: 'the_hollow', name: 'The Hollow', climate: { temp: 0.3, moist: 0.78, elev: 0.45 }, terrain: { base: 'moss', patch: 'ash', patchAmount: 0.3 }, tint: '#8b5cf6',
    objects: [{ kind: 'dead_tree', density: 0.05, minSpacing: 1.6 }, { kind: 'mushroom', density: 0.02, minSpacing: 1.5 }, { kind: 'log', density: 0.006, minSpacing: 3 }, { kind: 'stump', density: 0.006, minSpacing: 3 }] },
  { id: 'splash_cove', name: 'Splash Cove', climate: { temp: 0.65, moist: 0.72, elev: 0.1 }, terrain: { base: 'sand', patch: 'shallow', patchAmount: 0.25 }, tint: '#06b6d4',
    objects: [{ kind: 'reeds', density: 0.02, minSpacing: 1.5 }, { kind: 'rock', density: 0.01, minSpacing: 2 }, { kind: 'bush', density: 0.006, minSpacing: 2 }, { kind: 'log', density: 0.004, minSpacing: 3 }] },
  { id: 'mount_grumbleguts', name: 'Mount Grumbleguts', climate: { temp: 0.97, moist: 0.3, elev: 0.96 }, terrain: { base: 'ash', patch: 'rock', patchAmount: 0.3 }, tint: '#ef4444',
    objects: [{ kind: 'boulder', density: 0.02, minSpacing: 2.5 }, { kind: 'rock', density: 0.03, minSpacing: 1.5 }, { kind: 'dead_tree', density: 0.006, minSpacing: 3 }] },
];
