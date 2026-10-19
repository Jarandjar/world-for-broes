# ✨ World4Bros Gemini Asset & Content Generation Bible

A complete production guide for generating pixel-perfect **wear/outfits**, **items & weapons**, **objects/props**, **map terrain**, and **visual/sound effects** using Gemini models and image generators for World4Bros.

---

## 🎨 Engine Visual Constraints & Style Bible

To ensure all AI-generated assets look cohesive with World4Bros, follow these core rules:
1. **Perspective**: Top-down with a 15° slight isometric inclination (chibi characters with 2.5D depth).
2. **Outlines**: 1px dark charcoal outlines (`#1e293b` or `#0f172a`), strictly avoiding pure black `#000000` to prevent harsh visual clipping.
3. **Palette Harmony**:
   - **Tactical / Military**: Slate gray (`#334155`), olive drab (`#3f6212`), gunmetal (`#1e293b`), hazard amber (`#f59e0b`).
   - **Post-Apocalyptic Survival**: Rust orange (`#ea580c`), weathered wood (`#78350f`), steel silver (`#94a3b8`).
   - **Comedic Slapstick**: Toxic green (`#22c55e`), hot pink (`#ec4899`), neon cyan (`#06b6d4`), duck yellow (`#facc15`).
4. **Resolution Standards**:
   - **Wear / Outfits**: 32x32 per frame (or vector SVG / Canvas paths with distinct head/torso/legs/feet slots).
   - **Inventory Icons**: 48x48 pixels centered on transparent canvas.
   - **World Objects**: 32x32 (single tile), 48x48 (medium), or 64x64 (large structures).
   - **Terrain Autotiles**: 16x16 pixels per sub-tile (dual-grid 4-corner bitmasking).

---

## 1. 👕 Better Wear & Character Outfits

### Schema Integration (`src/sim/look.ts`)
Each character look consists of modular slots with primary and accent color tinting:
```typescript
export interface Look {
  skin: string;
  hair: string;
  hairColor: string;
  facialHair?: string;
  outfit: {
    head?: { item: string; colors: { primary: string; secondary?: string } };
    torso?: { item: string; colors: { primary: string; accent?: string } };
    legs?: { item: string; colors: { primary: string } };
    feet?: { item: string; colors: { primary: string } };
    back?: { item: string; colors: { primary: string } };
    mainHand?: { item: string };
  };
}
```

### Gemini Generation Prompt Template (Wear)
```markdown
**Prompt:**
16-bit retro pixel art character outfit spritesheet on transparent background, top-down 2D perspective.
Character: Chibi apocalypse survivor wearing [ITEM_NAME] (e.g., Heavy Juggernaut Kevlar Armor with reinforced ceramic plates, high-collared neck guard, and shoulder pads).
Details:
- High contrast, crisp 1-pixel dark slate outline (#0f172a).
- Modular design: separate isolated headpiece, torso, legs, and boots.
- Color palette: [PRIMARY_COLOR, e.g. #334155], [ACCENT_COLOR, e.g. #ea580c], and rim lighting (#94a3b8).
- Clean pixel clusters, zero dithering noise, zero blur, no background shadow.

**Negative Prompt:**
blurry, 3D render, photorealistic, smoothed gradients, anti-aliased edges, vector art, side view, realistic human proportions.
```

---

## 2. 🔫 Weapons, Firearms & Attachments

### Schema Integration (`src/data/schema.ts` & `src/data/content.ts`)
Weapons define ballistics curves, fire rates, spread, and visual attachment slots:
```typescript
guns.set('custom_nailgun', {
  id: 'custom_nailgun',
  name: 'Custom Nailgun SMG',
  ammoType: 'scrap_nails',
  magSize: 30,
  fireRate: 8.0,
  baseDamage: 18,
  minDamage: 9,
  falloffStartDist: 10,
  maxRange: 26,
  spreadRad: 0.10,
  projectileSpeed: 38,
  burstCount: 1,
  burstInterval: 0,
  noiseRadius: 20,
  bulletColor: '#38bdf8',
  bulletRadius: 0.1,
  penetration: 1,
});
```

### Gemini Generation Prompt Template (Guns & Attachments)
```markdown
**Prompt:**
2D pixel art inventory item icon, 48x48 pixels, transparent background.
Subject: Post-apocalyptic customized [WEAPON_TYPE] (e.g. Scrappy Pneumatic Submachine Gun with welded bicycle chain, tin-can muzzle compensator, and taped tactical red laser).
Style: Chunky 16-bit pixel art, distinct readable silhouette at small screen sizes, sharp 1px dark contour, metallic highlights on top edges.
Palette: Worn gunmetal grey (#1e293b), rust iron (#78350f), caution electrical yellow (#eab308).

**Negative Prompt:**
photorealistic 3D weapon, gradient background, border box, low contrast, washed out colors.
```

---

## 3. 📦 World Objects, Barricades & Interactive Props

### Schema Integration (`src/sim/worldgen.ts` & `src/data/schema.ts`)
```typescript
objects.set('tactical_sandbag_wall', {
  id: 'tactical_sandbag_wall',
  name: 'Heavy Sandbag Barricade',
  solid: true,       // Blocks player & bot movement
  radius: 0.45,
  hp: 250,
  tool: 'shovel',
  drawer: 'obj:sandbags',
});
```

### Gemini Generation Prompt Template (Props & Covers)
```markdown
**Prompt:**
Top-down 2.5D pixel art military base prop on transparent background.
Subject: [OBJECT_TYPE] (e.g. Quad-stacked fortified sandbags with embedded rebar and olive drab camouflage tarp).
Footprint: 48 pixels wide, 28 pixels high, 15-degree isometric projection angle.
Style: Chunky pixel art, 1px dark charcoal outline, chipped surface texture, battle-worn scratches.
Palette: Khaki burlap (#d4b996), olive tarp (#3f6212), dark iron (#1e293b).

**Negative Prompt:**
isometric 45-degree angle, flat side-view, blurry textures, modern sterile 3D render.
```

---

## 4. 🗺️ Map, Biomes & Dual-Grid Terrain

### Dual-Grid Autotiling Rules
World4Bros uses a **dual-grid bitmask transition system** (16 combinations per tile transition):
- **Bitmask Table**: `NW: 1`, `NE: 2`, `SW: 4`, `SE: 8` (Values `0` to `15`).
- The transition between two biomes (e.g. Asphalt Road to Muddy Swamp) uses seamless 16x16 corner and edge masks.

### Gemini Generation Prompt Template (Terrain)
```markdown
**Prompt:**
Seamless 16x16 pixel art autotile tileset for 2D top-down game.
Biomes: Transition between [PRIMARY_TERRAIN, e.g. Cracked asphalt roadway] and [SECONDARY_TERRAIN, e.g. Overgrown muddy swamp with murky puddles].
Format: 16-tile dual-grid autotiling sheet arranged in 4x4 grid containing:
- Inner corners, outer corners, vertical borders, horizontal borders, and solid centers.
Palette: Slate asphalt (#1e293b), cracked road paint (#facc15), dark mud (#3b2f2f), moss green (#4d7c0f).
Style: Pixel-perfect tiling, no visible repeating seam patterns, crisp pixel art.

**Negative Prompt:**
blurry textures, mismatched seams, perspective distortion, photorealistic satellite terrain.
```

---

## 5. 💥 Visual Effects & Web Audio Sound Synthesis

### Canvas2D Particle Shaders
Implement particles with decaying alpha and velocity damping:
```typescript
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
}
```

### 🔇 Audio Synthesis Notice (On Hold Until Beta)
> [!IMPORTANT]
> **Audio Synthesis Suspended**: Do NOT synthesize Web Audio procedural nodes or generate synthetic audio assets. Audio design, voice lines, and Foley effects will be designed and implemented by the owner when the game reaches Beta status.

