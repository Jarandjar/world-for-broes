/**
 * Gemini Asset & Content Generation Recipes for World4Bros
 * Provides structured prompts, schema templates, and negative prompt rules
 * for generating wear/outfits, items/weapons, objects, terrain tilesets, and visual/sound effects.
 */

export interface GeminiPromptRecipe {
  id: string;
  category: 'wear' | 'items' | 'objects' | 'terrain' | 'effects';
  title: string;
  shortDesc: string;
  prompt: string;
  negativePrompt: string;
  schemaExample: string;
}

export const GEMINI_ASSET_RECIPES: GeminiPromptRecipe[] = [
  {
    id: 'wear_tactical_outfit',
    category: 'wear',
    title: 'Tactical & Comedic Wear (Hats, Hoodies, Armor, Boots)',
    shortDesc: 'Pixel art outfit items matching World4Bros Character Look & palette system',
    prompt: `16-bit top-down pixel art character outfit spritesheet on transparent background.
Asset: Post-apocalyptic survivalist gear item for top-down chibi character (head, torso, legs, feet).
Items to include: 
1) Reinforced Kevlar Tactical Vest with ammo pouches and radio clip
2) Improvised welder mask with glowing visor slit
3) Heavy duty combat cargo pants with knee pads
4) Spring-loaded bouncy combat boots
Style: Clean crisp pixel art, 1px dark charcoal outlines (#1e293b), vibrant comic highlights, no antialiasing, no blur, distinct 4-frame walk cycle silhouette. Color palette: olive drab (#4d7c0f), dark slate (#0f172a), caution yellow (#facc15), rust orange (#ea580c).`,
    negativePrompt: 'blurry, 3d render, realistic photo, vector art, smoothed pixels, dithering overload, gradient fill, messy outlines',
    schemaExample: `// src/sim/look.ts outfit slot entry:
head: { item: 'welder_mask', colors: { primary: '#334155', secondary: '#facc15' } },
torso: { item: 'tactical_vest', colors: { primary: '#3f6212', accent: '#ea580c' } },
legs: { item: 'cargo_pants', colors: { primary: '#1e293b' } },
feet: { item: 'spring_boots', colors: { primary: '#475569' } }`,
  },
  {
    id: 'items_firearms_attachments',
    category: 'items',
    title: 'Custom Firearms & Homemade Attachments',
    shortDesc: 'Guns, attachments, and ammo items matching GunDef and AttachmentDef schemas',
    prompt: `Top-down 2D pixel art inventory icon (48x48 pixels) on transparent background.
Subject: Post-apocalyptic customized firearm and tactical modular attachments.
Weapon: Scrappy pneumatic nailgun submachine gun with taped flashlight, extended drum magazine made from a tin can, and an improvised oil-filter suppressor.
Style: Chunky pixel art, strong silhouettes, vibrant retro palette (dark iron, rusted copper, brass highlights, electric cyan tape). Flat lighting with directional top-left rim highlight, distinct readable shape at 40x40 viewport.`,
    negativePrompt: 'photorealistic, modern generic 3d weapon, low contrast, washed out colors, gradient background, border frame',
    schemaExample: `// src/data/content.ts GunDef entry:
guns.set('nailgun_smg', {
  id: 'nailgun_smg',
  name: 'Nailgun SMG',
  ammoType: 'scrap_nails',
  magSize: 32,
  fireRate: 8.5, // 8.5 rounds/sec
  baseDamage: 16,
  minDamage: 8,
  falloffStartDist: 8,
  maxRange: 24,
  spreadRad: 0.12,
  projectileSpeed: 38,
  burstCount: 1,
  burstInterval: 0,
  noiseRadius: 18,
  bulletColor: '#38bdf8',
  bulletRadius: 0.1,
  penetration: 1,
});`,
  },
  {
    id: 'objects_military_props',
    category: 'objects',
    title: 'Military Base & CQB Objects (Barricades, Crates, Targets)',
    shortDesc: 'Solid obstacles, bullet-stopping covers, and target dummies matching ObjectDef',
    prompt: `Top-down 2D pixel art isometric game objects and military base barricades.
Sprites to include:
1) Olive green military weapon supply crate with stenciled skull logo (32x32px)
2) Triple-stacked sandbag defensive barrier (48x24px)
3) Wobbling steel silhouette target dummy on heavy spring base with bullseye ring (24x36px)
4) Chainlink quarantine security fence segment with barbed razor wire top (32x48px)
Perspective: Top-down 2.5D with 15-degree projection angle. Style: 1px dark contour outlines, rich weathered textures, chipped paint, visible bullet scratch decals.`,
    negativePrompt: 'flat 2D side-view, 3D render, photorealism, blurry edges, inconsistent shadows, perspective distortion',
    schemaExample: `// src/data/content.ts ObjectDef entry:
objects.set('sandbags', {
  id: 'sandbags',
  name: 'Sandbags',
  solid: true,
  radius: 0.45,
  hp: 200,
  tool: 'shovel',
  drawer: 'obj:sandbags',
});
objects.set('steel_target', {
  id: 'steel_target',
  name: 'Reactive Steel Target',
  solid: false, // Allows bullets to hit & register impact without hard stop
  radius: 0.35,
  hp: 9999,
  drawer: 'obj:steel_target',
});`,
  },
  {
    id: 'terrain_dual_grid_tileset',
    category: 'terrain',
    title: 'Dual-Grid Terrain & Edge Transitions (Asphalt, Swamp, Scrapyard)',
    shortDesc: 'Tile edge transitions, blend corner masks, and dual-grid terrain lab specs',
    prompt: `Pixel art seamless autotile tileset for 2D top-down game engine (16x16 pixel tiles).
Biomes: Transition between cracked dark asphalt roadway and overgrown mud/swamp ground with puddles.
Tile layout: Complete 16-tile dual-grid transition set including inner corners, outer corners, horizontal edges, vertical edges, and center fills.
Details: Subtle tire tread marks on asphalt, oily sheen on puddle edges, green moss creeping into cracked concrete seams. Harmonious muted color palette: asphalt slate (#1e293b), cracked stone (#475569), swamp mud (#3f3322), stagnant water (#283618).`,
    negativePrompt: 'mismatched seams, grid alignment errors, blurry textures, photorealistic satellite image, 3D perspective',
    schemaExample: `// Dual-grid bitmask mapping in src/render/renderer.ts:
// Bitmask: [NW: 1, NE: 2, SW: 4, SE: 8] (0 to 15)
const DUAL_GRID_OFFSETS: Record<number, [number, number]> = {
  0: [0, 0],   // All Primary
  1: [16, 0],  // NW Corner Secondary
  2: [32, 0],  // NE Corner Secondary
  3: [48, 0],  // North Edge Secondary
  5: [0, 16],  // West Edge Secondary
  15: [48, 48] // All Secondary Fill
};`,
  },
  {
    id: 'effects_canvas_particles',
    category: 'effects',
    title: 'Canvas2D Visual Effects (Muzzle Flashes, Impacts, Splatters)',
    shortDesc: 'Particle emitters, muzzle flashes, blood splatters, and debris puffs (Audio deferred to Beta)',
    prompt: `JavaScript Canvas2D procedural particle effect recipes.
Effects needed:
1) Ballistic Muzzle Flash: Rapid expanding starburst with emissive core (#ffffff), outer halo (#f97316), and 6 flying spark embers with drag.
2) Bullet Impact & Ricochet: Directional debris puff (sparks for metal/stone, wood splinters for crates, blood droplets with gravity for flesh).
3) Ambient Smoke / Dust: Low-opacity circular dust clouds expanding and fading out over 0.6s.
Note: Audio synthesis is suspended until Beta per project direction.`,
    negativePrompt: 'heavy external libraries, threejs dependencies, webgl-only shader without fallback, audio synthesis',
    schemaExample: `// Particle emitter structure in src/render/particles.ts:
export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  drag?: number;
}`,
  },
];
