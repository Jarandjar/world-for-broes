/**
 * Road & ground detail painters (hand-drawn tile detail, baked into cached chunk canvases by tiles.ts).
 *
 * Two registries, both keyed by id so new looks are data + one function:
 *  - SURFACES: man-made road surfaces (asphalt, sidewalk, cobbles...). `surfaceOf` maps worldgen StreetDetails to an id.
 *  - GROUND_DETAILS: natural terrain detail keyed by TerrainDef.strokes (blades, pebbles, waves...).
 * Painters draw in tile units (1 = one tile) and must only use the per-tile rng they get, so every tile
 * always looks the same. Light comes from the top-left: lit edges up/left, shadows down/right. Never pure black.
 *
 * The legacy dual-grid autotile helpers at the bottom of this file are kept for the old terrain lab.
 */
import type { Rng } from '../engine/rng';
import { type G, INK, TAU, alpha, lumpy, mix, rrect, shade, stroke } from './ink';
import type { StreetDetails } from '../sim/worldgen';
import type { TerrainDef } from '../data/schema';

// ---------------------------------------------------------------- data: ids, palettes, chances
export type SurfaceId = 'avenue' | 'crosswalk' | 'sidewalk' | 'highway' | 'dirt_road' | 'plaza' | 'cobble';
export type Side = 'N' | 'E' | 'S' | 'W';
/** Side name with the neighbour offset it looks at. */
export const SIDES: readonly (readonly [Side, number, number])[] = [['N', 0, -1], ['E', 1, 0], ['S', 0, 1], ['W', -1, 0]];
/** 0 = zoomed far out (few marks), 1 = normal, 2 = close-up (all hairlines, gum, treads). */
export type Lod = 0 | 1 | 2;
export const lodFor = (px: number): Lod => (px >= 40 ? 2 : px >= 24 ? 1 : 0);

export interface SurfaceLook {
  base: string; dark: string; light: string;
  /** Lane / crossing paint. */
  paint?: string;
  /** Edge (fog) line paint. */
  line?: string;
  /** Per-tile chances for the scattered details; missing = never. */
  chance?: { crack?: number; pothole?: number; patch?: number; manhole?: number; drain?: number; skid?: number; oil?: number; weed?: number; gum?: number; puddle?: number };
}

/** Asphalt-like surfaces that get kerbs from the sidewalk next to them. */
export const KERBED: ReadonlySet<SurfaceId> = new Set<SurfaceId>(['avenue', 'crosswalk', 'highway']);
export const KERB = { top: '#a7b0bc', lit: '#d3d9e0', face: '#5b6370', joint: '#47505c' };
export const METAL = { rim: '#3d424b', plate: '#2c3037', grid: '#1d2026', lit: '#7d8592', rust: '#8a5a36' };
export const PUDDLE = { water: '#34495c', sheen: '#cfe6f2' };
export const WEED = { dark: '#3f7a37', light: '#79bd57' };
export const GUM = ['#d4d1c9', '#8e9097', '#6b6f78'] as const;
export const SOIL = { band: '#65482e', crumb: '#8c765c' };
/** Wildflower chance per tile, by terrain id. */
export const WILDFLOWERS: Partial<Record<string, number>> = { grass: 0.16, moss: 0.04 };
/** Flower colours with relative weights: dandelion, poppy, daisy, forget-me-not, clover. */
export const FLOWER_COLOURS: readonly (readonly [string, number])[] = [['#facc15', 28], ['#f43f5e', 24], ['#ffffff', 22], ['#38bdf8', 14], ['#f472b6', 12]];
/** Puddle chance per tile on natural ground, by terrain id. */
export const GROUND_PUDDLES: Partial<Record<string, number>> = { mud: 0.12, dirt: 0.015 };
/** Tyre-track chance per tile on natural ground, by terrain id. */
export const GROUND_TRACKS: Partial<Record<string, number>> = { dirt: 0.05, sand: 0.03, mud: 0.04 };

export const SURFACE_LOOKS: Record<SurfaceId, SurfaceLook> = {
  avenue: { base: '#1c1e22', dark: '#121316', light: '#30333a', paint: '#facc15', chance: { crack: 0.32, pothole: 0.035, patch: 0.07, manhole: 0.045, drain: 0.12, skid: 0.035, oil: 0.06 } },
  crosswalk: { base: '#1c1e22', dark: '#121316', light: '#30333a', paint: '#f8fafc', chance: { crack: 0.2, patch: 0.04 } },
  sidewalk: { base: '#8f9aa8', dark: '#6f7987', light: '#a9b2be', chance: { crack: 0.22, weed: 0.3, gum: 0.55 } },
  highway: { base: '#1b1c20', dark: '#111215', light: '#2f3139', paint: '#facc15', line: '#f0f8ff', chance: { crack: 0.4, pothole: 0.05, patch: 0.09, manhole: 0.01, skid: 0.04, oil: 0.04 } },
  dirt_road: { base: '#68503b', dark: '#453324', light: '#8c7258', chance: { puddle: 0.07, weed: 0.5 } },
  plaza: { base: '#5e6573', dark: '#3d434e', light: '#8a93a3', chance: { crack: 0.12, weed: 0.18, gum: 0.4 } },
  cobble: { base: '#505663', dark: '#2d313a', light: '#7d8596', chance: { weed: 0.35, pothole: 0.03 } },
};

/** Which surface a worldgen street cell uses (null = natural ground). */
export function surfaceOf(st: StreetDetails): SurfaceId | null {
  if (st.kind === 'none') return null;
  if (st.kind === 'dirt_road') return 'dirt_road';
  if (st.kind === 'small_road') return st.isPlaza ? 'plaza' : 'cobble';
  if (!st.isTown) return 'highway';
  return st.isSidewalk ? 'sidewalk' : st.isCrosswalk ? 'crosswalk' : 'avenue';
}

// ---------------------------------------------------------------- painter contexts
export interface TileCtx {
  g: G; rng: Rng; lod: Lod;
  /** Tile origin inside the chunk canvas, and integer world tile coords. */
  tx: number; ty: number; wx: number; wy: number;
  st: StreetDetails; id: SurfaceId; look: SurfaceLook;
  /** Smooth world noise in -0.5..0.5 for large-scale tonal drift. */
  tone: number;
  /** Surface of a neighbouring tile (null = natural ground). */
  near(dx: number, dy: number): SurfaceId | null;
  /** Natural terrain of a neighbouring tile. */
  ground(dx: number, dy: number): TerrainDef;
}
export interface SurfaceDef { look: SurfaceLook; paint(c: TileCtx): void }

export interface GroundCtx { g: G; rng: Rng; lod: Lod; tx: number; ty: number; d: TerrainDef }
export type GroundPainter = (c: GroundCtx) => void;

// ---------------------------------------------------------------- shared drawing helpers (tile units)
/** [x, y, w, h] of the strip of `depth` along one side of a tile. */
export function edgeRect(tx: number, ty: number, side: Side, depth: number, inset = 0): [number, number, number, number] {
  switch (side) {
    case 'N': return [tx, ty + inset, 1, depth];
    case 'S': return [tx, ty + 1 - inset - depth, 1, depth];
    case 'W': return [tx + inset, ty, depth, 1];
    default: return [tx + 1 - inset - depth, ty, depth, 1];
  }
}
/** A point `along` (0..1) the side and `depth` into the tile from it. */
function sidePoint(tx: number, ty: number, side: Side, along: number, depth: number): [number, number] {
  switch (side) {
    case 'N': return [tx + along, ty + depth];
    case 'S': return [tx + along, ty + 1 - depth];
    case 'W': return [tx + depth, ty + along];
    default: return [tx + 1 - depth, ty + along];
  }
}

/** Toned base fill of a surface tile (slightly over-sized to hide seams). Drawn for every surface before details. */
export function paintSurfaceBase(c: TileCtx, k = 0.55): void {
  const { g, look, tone } = c;
  g.fillStyle = mix(look.base, tone > 0 ? look.light : look.dark, Math.min(0.6, Math.abs(tone) * k));
  g.fillRect(c.tx - 0.04, c.ty - 0.04, 1.08, 1.08);
}

/** Aggregate specks: lit chips and dark pits. */
export function grain(c: TileCtx, n: number, size = 0.035): void {
  const { g, rng, look, tx, ty } = c;
  const count = c.lod === 0 ? Math.ceil(n / 4) : c.lod === 1 ? Math.ceil(n / 2) : n;
  for (let k = 0; k < count; k++) {
    const r = rng.next();
    g.fillStyle = r < 0.45 ? alpha(look.light, 0.75) : r < 0.85 ? alpha(look.dark, 0.85) : alpha(KERB.top, 0.35);
    const s = size * rng.range(0.5, 1.2);
    g.fillRect(tx + rng.next() * (1 - s), ty + rng.next() * (1 - s), s, s);
  }
}

/** Hairline crack that wanders and maybe forks, with a faint lit lip down-right. */
export function crack(g: G, rng: Rng, x: number, y: number, len: number, ang: number, col: string, w: number, lip?: string, fork = true): void {
  const pts: [number, number][] = [[x, y]];
  const n = 4;
  for (let i = 1; i <= n; i++) {
    ang += rng.range(-0.55, 0.55);
    x += Math.cos(ang) * len / n; y += Math.sin(ang) * len / n;
    pts.push([x, y]);
    if (fork && i === 2 && rng.chance(0.55)) crack(g, rng, x, y, len * 0.45, ang + (rng.chance(0.5) ? 1 : -1) * rng.range(0.6, 1.2), col, w * 0.75, lip, false);
  }
  if (lip) stroke(g, pts.map(([px, py]) => [px + w * 0.8, py + w * 0.8] as [number, number]), w * 0.7, lip, 0.45);
  stroke(g, pts, w, col, 0.8);
}

/** Small sky-lit puddle (lit streak top-left). */
export function puddle(g: G, rng: Rng, x: number, y: number, rx: number, ry: number): void {
  g.beginPath(); lumpy(g, x, y, rx, ry, rng, 8, 0.16);
  g.fillStyle = alpha(PUDDLE.water, 0.88); g.fill();
  g.strokeStyle = alpha(INK, 0.35); g.lineWidth = Math.max(0.008, rx * 0.06); g.stroke();
  stroke(g, [[x - rx * 0.55, y - ry * 0.05], [x - rx * 0.1, y - ry * 0.45]], Math.max(0.01, ry * 0.18), PUDDLE.sheen, 0.65);
  g.beginPath(); g.arc(x + rx * 0.35, y + ry * 0.2, Math.min(rx, ry) * 0.14, 0, TAU);
  g.fillStyle = alpha(PUDDLE.sheen, 0.45); g.fill();
}

/** Pothole: crumbly rim, dark hollow, lit far wall (bottom-right), sometimes holding water. */
export function pothole(c: TileCtx, x: number, y: number, r: number): void {
  const { g, rng, look } = c;
  g.beginPath(); lumpy(g, x, y, r * 1.3, r * 0.98, rng, 9, 0.2);
  g.fillStyle = alpha(look.light, 0.55); g.fill();
  g.beginPath(); lumpy(g, x, y, r, r * 0.76, rng, 9, 0.22);
  g.fillStyle = shade(look.dark, -0.4); g.fill();
  g.strokeStyle = alpha(INK, 0.75); g.lineWidth = 0.02; g.stroke();
  g.beginPath(); g.ellipse(x + r * 0.08, y + r * 0.06, r * 0.8, r * 0.58, 0, -0.1, Math.PI * 0.75);
  g.strokeStyle = alpha(look.light, 0.6); g.lineWidth = 0.018; g.stroke();
  if (rng.chance(0.45)) puddle(g, rng, x + r * 0.05, y + r * 0.08, r * 0.62, r * 0.42);
  for (let k = 0; k < (c.lod ? 5 : 2); k++) {
    const a = rng.next() * TAU, d = r * rng.range(1.3, 1.8);
    g.fillStyle = alpha(rng.chance(0.5) ? look.light : look.dark, 0.85);
    g.fillRect(x + Math.cos(a) * d, y + Math.sin(a) * d * 0.8, 0.025, 0.022);
  }
}

/** Glossy tar repair patch with a raised seam. */
export function tarPatch(c: TileCtx): void {
  const { g, rng, tx, ty, look } = c;
  const w = rng.range(0.3, 0.6), h = rng.range(0.22, 0.5);
  const x = tx + rng.range(0.05, 0.95 - w), y = ty + rng.range(0.05, 0.95 - h);
  const pts: [number, number][] = [[x, y + rng.range(0, 0.04)], [x + w, y + rng.range(0, 0.04)], [x + w - rng.range(0, 0.04), y + h], [x + rng.range(0, 0.04), y + h]];
  g.beginPath(); pts.forEach(([px, py], i) => (i ? g.lineTo(px, py) : g.moveTo(px, py))); g.closePath();
  g.fillStyle = shade(look.dark, -0.25); g.fill();
  g.strokeStyle = alpha(look.light, 0.5); g.lineWidth = 0.018; g.stroke();
  if (c.lod) stroke(g, [[x + w * 0.15, y + h * 0.25], [x + w * 0.45, y + h * 0.18]], 0.02, '#5d6573', 0.45);
}

/** Round manhole: drop shadow down-right, rim, grid plate, lit arc top-left, shadow arc bottom-right. */
export function manhole(c: TileCtx, x: number, y: number, r = 0.2): void {
  const { g, rng } = c;
  g.beginPath(); g.arc(x + 0.02, y + 0.025, r + 0.015, 0, TAU); g.fillStyle = alpha(INK, 0.35); g.fill();
  g.beginPath(); g.arc(x, y, r, 0, TAU); g.fillStyle = METAL.rim; g.fill();
  g.strokeStyle = alpha(INK, 0.85); g.lineWidth = 0.022; g.stroke();
  g.beginPath(); g.arc(x, y, r * 0.8, 0, TAU); g.fillStyle = METAL.plate; g.fill();
  if (c.lod) {
    g.save(); g.beginPath(); g.arc(x, y, r * 0.76, 0, TAU); g.clip();
    g.strokeStyle = alpha(METAL.grid, 0.9); g.lineWidth = 0.011; g.beginPath();
    for (let k = -4; k <= 4; k++) {
      const o = k * r * 0.22;
      g.moveTo(x + o - r, y - r); g.lineTo(x + o + r, y + r);
      g.moveTo(x + o + r, y - r); g.lineTo(x + o - r, y + r);
    }
    g.stroke(); g.restore();
  }
  g.beginPath(); g.arc(x, y, r * 0.9, Math.PI * 0.95, Math.PI * 1.55); g.strokeStyle = alpha(METAL.lit, 0.8); g.lineWidth = 0.016; g.stroke();
  g.beginPath(); g.arc(x, y, r * 0.9, -0.05, Math.PI * 0.55); g.strokeStyle = alpha(INK, 0.6); g.stroke();
  if (rng.chance(0.4)) { g.beginPath(); g.ellipse(x + r * 0.5, y + r * 0.85, r * 0.35, r * 0.12, 0.3, 0, TAU); g.fillStyle = alpha(METAL.rust, 0.35); g.fill(); }
}

/** Storm drain grate set against the kerb on one side of a road tile. */
export function stormDrain(c: TileCtx, side: Side): void {
  const { g, tx, ty } = c;
  const vertical = side === 'E' || side === 'W';
  const len = 0.42, dep = 0.15, a0 = 0.29;
  const x = vertical ? (side === 'W' ? tx + 0.03 : tx + 1 - 0.03 - dep) : tx + a0;
  const y = vertical ? ty + a0 : (side === 'N' ? ty + 0.03 : ty + 1 - 0.03 - dep);
  const w = vertical ? dep : len, h = vertical ? len : dep;
  g.fillStyle = alpha(INK, 0.4); g.fillRect(x + 0.015, y + 0.02, w, h);
  g.fillStyle = METAL.rim; g.fillRect(x, y, w, h);
  g.fillStyle = METAL.grid; g.fillRect(x + 0.02, y + 0.02, w - 0.04, h - 0.04);
  g.fillStyle = METAL.lit;
  const n = 6;
  for (let k = 1; k < n; k++) {
    if (vertical) g.fillRect(x + 0.025, y + (h * k) / n - 0.008, w - 0.05, 0.016);
    else g.fillRect(x + (w * k) / n - 0.008, y + 0.025, 0.016, h - 0.05);
  }
  g.strokeStyle = alpha(INK, 0.8); g.lineWidth = 0.016; g.strokeRect(x, y, w, h);
  stroke(g, [[x, y + h], [x, y], [x + w, y]], 0.012, METAL.lit, 0.6);
}

/** Worn road paint: a rectangle with chips knocked out and scuffs. */
export function paintStripe(c: TileCtx, x: number, y: number, w: number, h: number, col: string, wear = 0.4): void {
  const { g, rng, look } = c;
  g.fillStyle = alpha(col, 0.9); g.fillRect(x, y, w, h);
  if (!c.lod) return;
  const chips = Math.round((1 + wear * 5) * (c.lod === 2 ? 1 : 0.5));
  const s = Math.min(w, h, 0.07);
  for (let k = 0; k < chips; k++) {
    const cw = s * rng.range(0.3, 0.8), ch = s * rng.range(0.25, 0.7);
    g.fillStyle = alpha(mix(col, look.base, rng.range(0.55, 1)), rng.range(0.45, 0.85));
    g.fillRect(x + rng.next() * (w - cw), y + rng.next() * (h - ch), cw, ch);
  }
  g.fillStyle = alpha(shade(col, -0.25), 0.35);
  g.fillRect(x, y + h - Math.min(0.012, h * 0.2), w, Math.min(0.012, h * 0.2));
}

/** Pair of curving rubber skid marks. */
export function skid(c: TileCtx): void {
  const { g, rng, tx, ty, st } = c;
  const alongX = st.axis !== 'y';
  const o = rng.range(0.25, 0.6), bend = rng.range(-0.15, 0.15);
  for (let k = 0; k < 2; k++) {
    const off = o + k * 0.2;
    g.beginPath();
    if (alongX) { g.moveTo(tx + 0.02, ty + off); g.quadraticCurveTo(tx + 0.5, ty + off + bend, tx + 0.98, ty + off + bend * 0.4); }
    else { g.moveTo(tx + off, ty + 0.02); g.quadraticCurveTo(tx + off + bend, ty + 0.5, tx + off + bend * 0.4, ty + 0.98); }
    g.strokeStyle = alpha(INK, 0.22); g.lineWidth = 0.055; g.lineCap = 'round'; g.stroke();
  }
}

/** Oil stain: soft dark blot with a faint rainbow-ish sheen ring. */
export function oilStain(c: TileCtx): void {
  const { g, rng, tx, ty } = c;
  const x = tx + rng.range(0.25, 0.75), y = ty + rng.range(0.25, 0.75), r = rng.range(0.1, 0.2);
  g.beginPath(); lumpy(g, x, y, r, r * 0.7, rng, 8, 0.25, rng.next() * 3);
  g.fillStyle = alpha('#0e1013', 0.4); g.fill();
  if (c.lod === 2) { g.strokeStyle = alpha('#7b6fa6', 0.18); g.lineWidth = 0.014; g.stroke(); }
}

/** Grass tuft: blades fanning out of one root, highlight blades on the lit (left) side, tiny shadow down-right. */
export function tuft(g: G, rng: Rng, x: number, y: number, h: number, dark: string, light: string, n = 4): void {
  g.beginPath(); g.ellipse(x + h * 0.15, y + h * 0.06, h * 0.32, h * 0.12, 0, 0, TAU);
  g.fillStyle = alpha(INK, 0.16); g.fill();
  g.lineCap = 'round';
  for (let k = 0; k < n; k++) {
    const t = n === 1 ? 0 : k / (n - 1) - 0.5;
    const lean = t * h * 0.9 + rng.range(-0.02, 0.02), bh = h * rng.range(0.7, 1.05);
    g.beginPath(); g.moveTo(x + t * h * 0.2, y);
    g.quadraticCurveTo(x + lean * 0.3, y - bh * 0.6, x + lean, y - bh);
    g.strokeStyle = t < 0 ? light : dark;
    g.lineWidth = Math.max(0.01, h * 0.11);
    g.stroke();
  }
}

/** Five-petal wildflower (just a dot when zoomed far out). */
export function flower(g: G, rng: Rng, x: number, y: number, col: string, lod: Lod): void {
  const r = rng.range(0.022, 0.032);
  if (!lod) { g.fillStyle = col; g.beginPath(); g.arc(x, y, r * 2, 0, TAU); g.fill(); return; }
  stroke(g, [[x, y + r], [x + 0.01, y + r * 4]], 0.01, WEED.dark, 0.9);
  const rot = rng.next() * TAU;
  g.fillStyle = col;
  for (let k = 0; k < 5; k++) {
    const a = rot + (k / 5) * TAU;
    g.beginPath(); g.arc(x + Math.cos(a) * r * 1.2, y + Math.sin(a) * r * 1.2, r, 0, TAU); g.fill();
  }
  g.strokeStyle = alpha(INK, 0.35); g.lineWidth = 0.006; g.stroke();
  g.fillStyle = col === '#facc15' ? '#b45309' : '#facc15';
  g.beginPath(); g.arc(x, y, r * 0.65, 0, TAU); g.fill();
}

export function pickFlower(rng: Rng): string {
  const total = FLOWER_COLOURS.reduce((s, [, w]) => s + w, 0);
  let r = rng.next() * total;
  for (const [col, w] of FLOWER_COLOURS) { r -= w; if (r <= 0) return col; }
  return FLOWER_COLOURS[0]?.[0] ?? '#ffffff';
}

/** Gum dots (old grey and newer pale) on paving. */
export function gum(c: TileCtx, n: number): void {
  const { g, rng, tx, ty } = c;
  for (let k = 0; k < n; k++) {
    g.fillStyle = alpha(rng.pick(GUM), 0.7);
    g.beginPath(); g.ellipse(tx + rng.range(0.08, 0.92), ty + rng.range(0.08, 0.92), rng.range(0.012, 0.022), rng.range(0.01, 0.018), 0, 0, TAU); g.fill();
  }
}

/** Pebble with a cast shadow down-right and a highlight up-left. */
export function pebble(g: G, x: number, y: number, r: number, body: string, rot: number): void {
  g.beginPath(); g.ellipse(x + r * 0.35, y + r * 0.4, r, r * 0.8, rot, 0, TAU); g.fillStyle = alpha(INK, 0.25); g.fill();
  g.beginPath(); g.ellipse(x, y, r, r * 0.8, rot, 0, TAU); g.fillStyle = body; g.fill();
  g.strokeStyle = alpha(INK, 0.55); g.lineWidth = Math.max(0.006, r * 0.22); g.stroke();
  g.beginPath(); g.arc(x - r * 0.35, y - r * 0.3, r * 0.3, 0, TAU); g.fillStyle = alpha('#efe8d6', 0.55); g.fill();
}

/** Short pair of tyre-tread lines across a tile (on dirt, sand, mud). */
export function tyreTrack(g: G, rng: Rng, tx: number, ty: number, col: string, lod: Lod): void {
  const vertical = rng.chance(0.5), o = rng.range(0.2, 0.55), bend = rng.range(-0.12, 0.12);
  for (let k = 0; k < 2; k++) {
    const off = o + k * 0.24;
    const pts: [number, number][] = [];
    for (let s = 0; s <= 6; s++) {
      const t = s / 6, wob = Math.sin(t * Math.PI) * bend;
      pts.push(vertical ? [tx + off + wob, ty + t] : [tx + t, ty + off + wob]);
    }
    stroke(g, pts, 0.06, col, 0.32);
    if (lod === 2) for (const [px, py] of pts) stroke(g, vertical ? [[px - 0.025, py], [px + 0.025, py + 0.02]] : [[px, py - 0.025], [px + 0.02, py + 0.025]], 0.01, col, 0.45);
  }
}

/** Soft blend into neighbouring natural ground: seam line plus that ground's own grass or crumbs creeping in. */
export function fringe(c: TileCtx, side: Side, dx: number, dy: number): void {
  const { g, rng, tx, ty } = c;
  const o = c.ground(dx, dy);
  if (o.id === 'deep' || o.id === 'shallow') return;
  const [x, y, w, h] = edgeRect(tx, ty, side, 0.07);
  g.fillStyle = alpha(o.colors[1], 0.45); g.fillRect(x, y, w, h);
  const n = c.lod === 0 ? 2 : c.lod === 1 ? 4 : 7;
  for (let k = 0; k < n; k++) {
    const [px, py] = sidePoint(tx, ty, side, rng.range(0.04, 0.96), rng.range(0.02, 0.12));
    if (o.strokes === 'blades' || o.strokes === 'needles') tuft(g, rng, px, py, rng.range(0.07, 0.13), shade(o.colors[1], -0.15), o.colors[2], rng.int(2, 4));
    else { g.fillStyle = alpha(rng.chance(0.5) ? o.colors[0] : o.colors[2], 0.85); g.fillRect(px, py, 0.035, 0.03); }
  }
}

/** Kerb stone strip on a sidewalk side that faces a road: lit top, joints, outer edge lit (N/W, facing the sun) or shadowed (S/E). */
export function kerb(c: TileCtx, side: Side): void {
  const { g, rng, tx, ty } = c;
  const d = 0.15;
  const [x, y, w, h] = edgeRect(tx, ty, side, d);
  g.fillStyle = KERB.top; g.fillRect(x, y, w, h);
  const [ix, iy, iw, ih] = edgeRect(tx, ty, side, 0.018, d - 0.018);
  g.fillStyle = alpha(KERB.joint, 0.8); g.fillRect(ix, iy, iw, ih);
  const [ox, oy, ow, oh] = edgeRect(tx, ty, side, 0.035);
  g.fillStyle = side === 'N' || side === 'W' ? KERB.lit : KERB.face; g.fillRect(ox, oy, ow, oh);
  const vertical = side === 'E' || side === 'W';
  g.fillStyle = alpha(KERB.joint, 0.9);
  for (const a of [0, 0.5]) {
    const j = a + rng.range(-0.02, 0.02);
    if (vertical) g.fillRect(x, ty + j, w, 0.016); else g.fillRect(tx + j, y, 0.016, h);
  }
  if (c.lod && rng.chance(0.35)) {
    const a = rng.range(0.1, 0.85);
    const [cx, cy] = sidePoint(tx, ty, side, a, 0.01);
    g.fillStyle = alpha(KERB.face, 0.9);
    g.beginPath(); g.ellipse(cx, cy, vertical ? 0.03 : 0.06, vertical ? 0.06 : 0.03, 0, 0, TAU); g.fill();
  }
}

/** Contact shadow a kerb / raised edge casts onto this tile from its N or W side (light is top-left). */
export function kerbShadow(c: TileCtx, side: Side): void {
  if (side !== 'N' && side !== 'W') return;
  const { g, tx, ty } = c;
  const [x, y, w, h] = edgeRect(tx, ty, side, 0.09);
  g.fillStyle = alpha(INK, 0.38); g.fillRect(x, y, w, h);
  const [x2, y2, w2, h2] = edgeRect(tx, ty, side, 0.06, 0.09);
  g.fillStyle = alpha(INK, 0.14); g.fillRect(x2, y2, w2, h2);
}

/** Weeds sprouting from a joint or crack. */
export function weed(c: TileCtx, x: number, y: number): void {
  tuft(c.g, c.rng, x, y, c.rng.range(0.06, 0.11), WEED.dark, WEED.light, c.rng.int(3, 5));
}

// ---------------------------------------------------------------- surface painters
const ch = (c: TileCtx, k: keyof NonNullable<SurfaceLook['chance']>): boolean => c.rng.chance(c.look.chance?.[k] ?? 0);
const crackOn = (c: TileCtx, w = 0.012): void => {
  const { rng, tx, ty, look } = c;
  if (!c.lod) return;
  crack(c.g, rng, tx + rng.range(0.15, 0.85), ty + rng.range(0.15, 0.85), rng.range(0.25, 0.5), rng.next() * TAU, shade(look.dark, -0.5), w, look.light);
};

/** Shared asphalt wear: grain, cracks, patches, potholes, oil, skids. */
function asphaltWear(c: TileCtx): void {
  grain(c, 26);
  if (ch(c, 'patch')) tarPatch(c);
  if (ch(c, 'crack')) crackOn(c);
  if (ch(c, 'oil')) oilStain(c);
  if (ch(c, 'pothole')) pothole(c, c.tx + c.rng.range(0.3, 0.7), c.ty + c.rng.range(0.3, 0.7), c.rng.range(0.08, 0.13));
  if (ch(c, 'skid')) skid(c);
}

/** Avenue/crosswalk edges: kerb shadow from a sidewalk to the N/W, gutter grime along any sidewalk, maybe a drain. */
function gutters(c: TileCtx, drains: boolean): void {
  for (const [side, dx, dy] of SIDES) {
    if (c.near(dx, dy) !== 'sidewalk') continue;
    const [x, y, w, h] = edgeRect(c.tx, c.ty, side, 0.16);
    c.g.fillStyle = alpha(c.look.dark, 0.6); c.g.fillRect(x, y, w, h);
    if (c.lod === 2) for (let k = 0; k < 3; k++) {
      const [px, py] = sidePoint(c.tx, c.ty, side, c.rng.next(), c.rng.range(0.03, 0.14));
      c.g.fillStyle = alpha(c.rng.chance(0.5) ? '#8a6a3c' : '#6b7a4a', 0.7); c.g.fillRect(px, py, 0.03, 0.022);
    }
    kerbShadow(c, side);
    if (drains && ch(c, 'drain')) stormDrain(c, side);
  }
}

/** Dashed double centre line (checker parity keeps dashes alternating along the road). */
function centreLine(c: TileCtx, col: string, axis: 'x' | 'y'): void {
  if ((c.wx + c.wy) % 2 !== 0) return;
  if (axis === 'x') { paintStripe(c, c.tx, c.ty + 0.44, 1, 0.05, col, 0.35); paintStripe(c, c.tx, c.ty + 0.52, 1, 0.05, col, 0.35); }
  else { paintStripe(c, c.tx + 0.44, c.ty, 0.05, 1, col, 0.35); paintStripe(c, c.tx + 0.52, c.ty, 0.05, 1, col, 0.35); }
}

const avenue: SurfaceDef = {
  look: SURFACE_LOOKS.avenue,
  paint(c) {
    asphaltWear(c);
    const paint = c.look.paint ?? '#facc15';
    if (c.st.axis === 'x' || c.st.axis === 'both') centreLine(c, paint, 'x');
    if (c.st.axis === 'y' || c.st.axis === 'both') centreLine(c, paint, 'y');
    const nextToWalk = SIDES.some(([, dx, dy]) => c.near(dx, dy) === 'sidewalk');
    if (!nextToWalk && ch(c, 'manhole')) manhole(c, c.tx + 0.5, c.ty + 0.5, c.rng.range(0.18, 0.23));
    gutters(c, true);
  },
};

const crosswalk: SurfaceDef = {
  look: SURFACE_LOOKS.crosswalk,
  paint(c) {
    grain(c, 22);
    if (ch(c, 'patch')) tarPatch(c);
    const paint = c.look.paint ?? '#f8fafc';
    const across = c.st.axis === 'x';
    for (let b = 0; b < 3; b++) {
      if (across) paintStripe(c, c.tx + 0.12 + b * 0.32, c.ty + 0.15, 0.16, 0.7, paint, 0.55);
      else paintStripe(c, c.tx + 0.15, c.ty + 0.12 + b * 0.32, 0.7, 0.16, paint, 0.55);
    }
    // Tyre lanes wear the bars grey where cars roll over them.
    if (c.lod) {
      const o = c.rng.range(0.25, 0.45);
      c.g.fillStyle = alpha(c.look.light, 0.28);
      for (const k of [o, o + 0.32]) {
        if (across) c.g.fillRect(c.tx, c.ty + k, 1, 0.09); else c.g.fillRect(c.tx + k, c.ty, 0.09, 1);
      }
    }
    if (ch(c, 'crack')) crackOn(c);
    gutters(c, false);
  },
};

const sidewalk: SurfaceDef = {
  look: SURFACE_LOOKS.sidewalk,
  paint(c) {
    const { g, rng, tx, ty, look } = c;
    const alongX = c.st.axis !== 'y';
    // Two slabs per tile, each with its own tone.
    for (let s = 0; s < 2; s++) {
      g.fillStyle = mix(look.base, rng.chance(0.5) ? look.light : look.dark, rng.range(0, 0.35) + Math.abs(c.tone) * 0.3);
      if (alongX) g.fillRect(tx + s * 0.5, ty, 0.5, 1); else g.fillRect(tx, ty + s * 0.5, 1, 0.5);
    }
    grain(c, 14, 0.025);
    // Expansion joints: dark groove with a lit far lip (down/right).
    const joint = (x: number, y: number, w: number, h: number): void => {
      g.fillStyle = shade(look.dark, -0.35); g.fillRect(x, y, w, h);
      g.fillStyle = alpha(look.light, 0.8);
      if (w < h) g.fillRect(x + w, y, 0.014, h); else g.fillRect(x, y + h, w, 0.014);
    };
    if (alongX) { joint(tx + 0.485, ty, 0.03, 1); joint(tx - 0.015, ty, 0.03, 1); }
    else { joint(tx, ty + 0.485, 1, 0.03); joint(tx, ty - 0.015, 1, 0.03); }
    // Chipped slab corner.
    if (c.lod && rng.chance(0.3)) {
      const cx = tx + (alongX ? (rng.chance(0.5) ? 0.5 : 0.02) : rng.range(0.1, 0.9)), cy = ty + (alongX ? rng.range(0.1, 0.9) : (rng.chance(0.5) ? 0.5 : 0.02));
      g.fillStyle = shade(look.dark, -0.15);
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + 0.07, cy + 0.01); g.lineTo(cx + 0.02, cy + 0.06); g.closePath(); g.fill();
    }
    if (ch(c, 'crack')) crackOn(c, 0.01);
    if (ch(c, 'weed')) {
      const a = rng.range(0.1, 0.9);
      if (alongX) weed(c, tx + 0.5, ty + a); else weed(c, tx + a, ty + 0.5);
    }
    if (c.lod === 2 && ch(c, 'gum')) gum(c, rng.int(1, 3));
    if (c.lod && rng.chance(0.12)) {
      g.beginPath(); lumpy(g, tx + rng.range(0.3, 0.7), ty + rng.range(0.3, 0.7), rng.range(0.08, 0.16), rng.range(0.06, 0.12), rng, 8, 0.25);
      g.fillStyle = alpha(look.dark, 0.25); g.fill();
    }
    for (const [side, dx, dy] of SIDES) {
      const n = c.near(dx, dy);
      if (n && KERBED.has(n)) kerb(c, side);
      else if (!n) fringe(c, side, dx, dy);
    }
  },
};

const highway: SurfaceDef = {
  look: SURFACE_LOOKS.highway,
  paint(c) {
    asphaltWear(c);
    if (ch(c, 'manhole')) manhole(c, c.tx + 0.5, c.ty + 0.5, 0.17);
    const ax = c.st.axis;
    if (ax === 'both') return;
    // Edge tiles get a fog line and a crumbling soil edge; interior tiles get the centre line.
    const perp: readonly (readonly [Side, number, number])[] = ax === 'x' ? [['N', 0, -1], ['S', 0, 1]] : [['W', -1, 0], ['E', 1, 0]];
    let edge = false;
    for (const [side, dx, dy] of perp) {
      if (c.near(dx, dy) !== null) continue;
      edge = true;
      const [x, y, w, h] = edgeRect(c.tx, c.ty, side, 0.035, 0.12);
      paintStripe(c, x, y, w, h, c.look.line ?? '#f0f8ff', 0.5);
      crumbleEdge(c, side, dx, dy);
    }
    if (!edge) centreLine(c, c.look.paint ?? '#facc15', ax);
  },
};

/** Ragged asphalt edge: soil bites, crumbs, then grass creeping in. */
function crumbleEdge(c: TileCtx, side: Side, dx: number, dy: number): void {
  const { g, rng, tx, ty } = c;
  g.fillStyle = SOIL.band;
  for (let k = 0; k < 6; k++) {
    const [px, py] = sidePoint(tx, ty, side, (k + rng.next()) / 6, rng.range(0, 0.05));
    g.beginPath(); g.ellipse(px, py, rng.range(0.06, 0.14), rng.range(0.04, 0.09), side === 'N' || side === 'S' ? 0 : Math.PI / 2, 0, TAU); g.fill();
  }
  for (let k = 0; k < (c.lod ? 6 : 2); k++) {
    const [px, py] = sidePoint(tx, ty, side, rng.next(), rng.range(0.04, 0.16));
    g.fillStyle = rng.chance(0.5) ? c.look.light : SOIL.crumb;
    g.fillRect(px, py, rng.range(0.02, 0.045), rng.range(0.02, 0.04));
  }
  fringe(c, side, dx, dy);
}

/** One wheel rut along the road: dark groove, deeper core, lit far wall, tread ticks close up. */
function rut(c: TileCtx, along: 'x' | 'y', at: number): void {
  const { g, rng, tx, ty, look } = c;
  const w = 0.14;
  const pts = (o: number): [number, number][] => {
    const p: [number, number][] = [];
    for (let s = 0; s <= 4; s++) {
      const t = s / 4, wob = Math.sin((c.wx + c.wy + t) * 2.1) * 0.012;
      p.push(along === 'x' ? [tx + t, ty + at + o + wob] : [tx + at + o + wob, ty + t]);
    }
    return p;
  };
  stroke(g, pts(w / 2), w, look.dark, 0.85);
  stroke(g, pts(w / 2 - 0.02), w * 0.35, shade(look.dark, -0.25), 0.55);
  stroke(g, pts(w - 0.01), 0.016, look.light, 0.55);
  if (c.lod === 2) for (let s = 0; s < 6; s++) {
    const t = (s + rng.next() * 0.4) / 6;
    const [x, y] = along === 'x' ? [tx + t, ty + at + 0.03] : [tx + at + 0.03, ty + t];
    stroke(g, along === 'x' ? [[x, y], [x + 0.03, y + w * 0.6]] : [[x, y], [x + w * 0.6, y + 0.03]], 0.012, shade(look.dark, -0.3), 0.5);
  }
}

const dirtRoad: SurfaceDef = {
  look: SURFACE_LOOKS.dirt_road,
  paint(c) {
    const { g, rng, tx, ty, look } = c;
    for (let k = 0; k < (c.lod ? 3 : 1); k++) {
      g.beginPath(); lumpy(g, tx + rng.next(), ty + rng.next(), rng.range(0.15, 0.35), rng.range(0.12, 0.28), rng, 8, 0.2);
      g.fillStyle = alpha(rng.chance(0.5) ? look.light : look.dark, 0.22); g.fill();
    }
    const ax = c.st.axis;
    if (ax !== 'y') { rut(c, 'x', 0.2); rut(c, 'x', 0.66); }
    if (ax !== 'x') { rut(c, 'y', 0.2); rut(c, 'y', 0.66); }
    if (ch(c, 'puddle')) {
      const at = rng.chance(0.5) ? 0.27 : 0.73, t = rng.range(0.3, 0.7);
      const [x, y] = ax === 'y' ? [tx + at, ty + t] : [tx + t, ty + at];
      puddle(g, rng, x, y, ax === 'y' ? 0.08 : 0.17, ax === 'y' ? 0.17 : 0.08);
    }
    if (ax !== 'both' && ch(c, 'weed')) {
      const grass = c.ground(0, 0);
      for (let k = 0; k < rng.int(1, 3); k++) {
        const t = rng.range(0.1, 0.9), o = 0.5 + rng.range(-0.05, 0.05);
        const [x, y] = ax === 'x' ? [tx + t, ty + o] : [tx + o, ty + t];
        tuft(g, rng, x, y, rng.range(0.06, 0.11), shade(grass.colors[1], -0.1), grass.colors[2], 3);
      }
    }
    for (let k = 0; k < (c.lod === 2 ? 5 : 2); k++) pebble(g, tx + rng.range(0.05, 0.95), ty + rng.range(0.05, 0.95), rng.range(0.018, 0.035), mix(look.light, '#b9a58a', rng.next()), rng.next() * 3);
    for (const [side, dx, dy] of SIDES) if (c.near(dx, dy) === null) fringe(c, side, dx, dy);
  },
};

/** Stone paving shared by plaza flagstones and cobbles: per-stone tone, lit top-left bevel, shadow bottom-right. */
function stone(c: TileCtx, x: number, y: number, w: number, h: number): void {
  const { g, rng, look } = c;
  if (c.lod && c.id === 'cobble' && rng.chance(0.025)) { // missing cobble: dirt shows through
    g.fillStyle = SOIL.band; g.fillRect(x, y, w, h);
    pebble(g, x + w * 0.5, y + h * 0.5, Math.min(w, h) * 0.18, SOIL.crumb, 0);
    return;
  }
  g.fillStyle = mix(look.base, rng.chance(0.5) ? look.light : look.dark, rng.range(0, 0.4));
  g.beginPath(); rrect(g, x, y, w, h, Math.min(w, h) * 0.22); g.fill();
  if (!c.lod) return;
  const b = Math.min(w, h) * 0.12;
  g.fillStyle = alpha(look.light, 0.45); g.fillRect(x + b, y + b * 0.4, w - b * 2, b); g.fillRect(x + b * 0.4, y + b, b, h - b * 2);
  g.fillStyle = alpha(INK, 0.28); g.fillRect(x + b, y + h - b * 1.4, w - b * 2, b); g.fillRect(x + w - b * 1.4, y + b, b, h - b * 2);
}

const plaza: SurfaceDef = {
  look: SURFACE_LOOKS.plaza,
  paint(c) {
    const { g, rng, tx, ty, look } = c;
    g.fillStyle = shade(look.dark, -0.2); g.fillRect(tx, ty, 1, 1);
    for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) stone(c, tx + i * 0.5 + 0.02, ty + j * 0.5 + 0.02, 0.46, 0.46);
    if (ch(c, 'crack')) crackOn(c, 0.009);
    if (ch(c, 'weed')) weed(c, tx + (rng.chance(0.5) ? 0.5 : 0.02), ty + rng.range(0.15, 0.85));
    if (c.lod === 2 && ch(c, 'gum')) gum(c, rng.int(1, 2));
  },
};

const cobble: SurfaceDef = {
  look: SURFACE_LOOKS.cobble,
  paint(c) {
    const { g, rng, tx, ty, look } = c;
    g.fillStyle = look.dark; g.fillRect(tx, ty, 1, 1);
    const rows = c.lod ? 3 : 2, h = 1 / rows;
    for (let r = 0; r < rows; r++) {
      const shift = ((c.wy * rows + r) % 2) * 0.25;
      for (let x = -shift; x < 1; x += 0.5) {
        const x0 = Math.max(0, x), x1 = Math.min(1, x + 0.5);
        if (x1 - x0 < 0.08) continue;
        stone(c, tx + x0 + 0.025, ty + r * h + 0.025, x1 - x0 - 0.05, h - 0.05);
      }
    }
    if (ch(c, 'pothole')) pothole(c, tx + rng.range(0.3, 0.7), ty + rng.range(0.3, 0.7), rng.range(0.07, 0.1));
    if (ch(c, 'weed')) weed(c, tx + rng.range(0.1, 0.9), ty + h * rng.int(1, rows - 1));
    for (const [side, dx, dy] of SIDES) if (c.near(dx, dy) === null) fringe(c, side, dx, dy);
  },
};

/** Registry of road surfaces. Add a look to SURFACE_LOOKS, a painter here, and map it in surfaceOf. */
export const SURFACES: Record<SurfaceId, SurfaceDef> = { avenue, crosswalk, sidewalk, highway, dirt_road: dirtRoad, plaza, cobble };

// ---------------------------------------------------------------- natural ground painters (keyed by TerrainDef.strokes)
export const GROUND_DETAILS: Record<TerrainDef['strokes'], GroundPainter> = {
  blades(c) {
    const { g, rng, tx, ty, d, lod } = c;
    const [, dark, light] = d.colors;
    if (lod && rng.chance(0.35)) { // colour variation clump
      g.beginPath(); g.ellipse(tx + rng.next(), ty + rng.next(), rng.range(0.2, 0.4), rng.range(0.15, 0.3), rng.next() * 3, 0, TAU);
      g.fillStyle = alpha(rng.chance(0.6) ? shade(dark, -0.1) : light, 0.16); g.fill();
    }
    for (let k = 0; k < (lod === 2 ? 8 : lod ? 6 : 4); k++) { // loose blades
      const x = tx + rng.next(), y = ty + 0.1 + rng.next() * 0.9, h = 0.08 + rng.next() * 0.12, lean = (rng.next() - 0.5) * 0.14;
      g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(x + lean * 0.4, y - h * 0.6, x + lean, y - h);
      g.strokeStyle = alpha(k % 3 ? shade(dark, -0.2) : light, 0.8); g.lineWidth = 0.012 + rng.next() * 0.006; g.lineCap = 'round'; g.stroke();
    }
    for (let k = 0; k < (lod === 2 ? 3 : lod ? 2 : 1); k++) tuft(g, rng, tx + rng.range(0.1, 0.9), ty + rng.range(0.25, 0.95), rng.range(0.1, 0.17), shade(dark, -0.25), light, rng.int(3, 5));
    if (lod === 2 && rng.chance(0.18)) { // clover leaves
      const x = tx + rng.range(0.15, 0.85), y = ty + rng.range(0.15, 0.85);
      g.fillStyle = alpha(shade(light, 0.1), 0.75);
      for (let k = 0; k < 3; k++) { const a = (k / 3) * TAU - 1.2; g.beginPath(); g.arc(x + Math.cos(a) * 0.025, y + Math.sin(a) * 0.025, 0.022, 0, TAU); g.fill(); }
    }
    if (rng.chance(0.2)) for (let k = -1; k <= 1; k++) { // inky grass mark
      const x = tx + 0.2 + rng.next() * 0.6, y = ty + 0.3 + rng.next() * 0.6;
      stroke(g, [[x, y], [x + k * 0.06, y - 0.17]], 0.014, shade(dark, -0.45), 0.85);
    }
    if (rng.chance(WILDFLOWERS[d.id] ?? 0)) {
      const col = pickFlower(rng);
      for (let k = 0; k < rng.int(1, 3); k++) flower(g, rng, tx + rng.range(0.15, 0.85), ty + rng.range(0.15, 0.85), col, lod);
    }
  },
  needles(c) {
    const { g, rng, tx, ty, d } = c;
    const [, dark, light] = d.colors;
    for (let k = 0; k < (c.lod ? 16 : 8); k++) {
      const x = tx + rng.next(), y = ty + rng.next(), a = rng.next() * TAU, l = 0.07 + rng.next() * 0.07;
      stroke(g, [[x, y], [x + Math.cos(a) * l, y + Math.sin(a) * l]], 0.011, k % 2 ? shade(dark, -0.3) : light, 0.75);
    }
    if (c.lod && rng.chance(0.08)) pebble(g, tx + rng.range(0.2, 0.8), ty + rng.range(0.2, 0.8), 0.05, '#7a5a3a', rng.next() * 3); // pine cone stub
  },
  pebbles(c) {
    const { g, rng, tx, ty, d, lod } = c;
    const [, dark, light] = d.colors;
    for (let k = 0; k < (lod === 2 ? 6 : lod ? 4 : 2); k++) pebble(g, tx + rng.next(), ty + rng.next(), 0.02 + rng.next() * 0.035, mix(light, '#efe8d6', rng.next() * 0.25), rng.next() * 3);
    for (let k = 0; k < 4; k++) { const x = tx + rng.next(), y = ty + rng.next(); stroke(g, [[x, y], [x + 0.06, y + (rng.next() - 0.5) * 0.04]], 0.008, shade(dark, -0.3), 0.6); }
    if (rng.chance(GROUND_TRACKS[d.id] ?? 0)) tyreTrack(g, rng, tx, ty, shade(dark, -0.2), lod);
    if (rng.chance(GROUND_PUDDLES[d.id] ?? 0)) puddle(g, rng, tx + rng.range(0.3, 0.7), ty + rng.range(0.3, 0.7), rng.range(0.12, 0.22), rng.range(0.07, 0.13));
  },
  cracks(c) {
    const { g, rng, tx, ty, d } = c;
    const [, dark, light] = d.colors;
    if (rng.chance(0.5)) crack(g, rng, tx + rng.next(), ty + rng.next(), 0.4, rng.next() * TAU, INK, 0.012, light);
    for (let k = 0; k < 4; k++) { const x = tx + rng.next(), y = ty + rng.next(); stroke(g, [[x, y], [x + 0.09, y + 0.05]], 0.009, light, 0.55); }
    if (c.lod && rng.chance(0.15)) pebble(g, tx + rng.range(0.2, 0.8), ty + rng.range(0.2, 0.8), 0.04, mix(dark, light, 0.6), rng.next() * 3);
  },
  waves(c) {
    const { g, rng, tx, ty, d } = c;
    const [, dark, light] = d.colors;
    for (let k = 0; k < 4; k++) {
      const x = tx + rng.next() * 0.5, y = ty + 0.15 + rng.next() * 0.7, pts: [number, number][] = [];
      for (let s = 0; s < 4; s++) pts.push([x + s * 0.12, y + Math.sin(s * 1.6 + rng.next()) * 0.03]);
      stroke(g, pts, 0.012, k % 2 ? light : shade(dark, -0.2), 0.65);
    }
    if (rng.chance(GROUND_TRACKS[d.id] ?? 0)) tyreTrack(g, rng, tx, ty, shade(dark, -0.2), c.lod);
    if (rng.chance(GROUND_PUDDLES[d.id] ?? 0)) puddle(g, rng, tx + rng.range(0.3, 0.7), ty + rng.range(0.3, 0.7), rng.range(0.14, 0.26), rng.range(0.08, 0.15));
  },
  flakes(c) {
    const { g, rng, tx, ty, d } = c;
    const [, dark] = d.colors;
    for (let k = 0; k < 8; k++) { g.fillStyle = alpha('#ffffff', 0.65); g.beginPath(); g.arc(tx + rng.next(), ty + rng.next(), 0.012 + rng.next() * 0.014, 0, TAU); g.fill(); }
    for (let k = 0; k < 3; k++) { const x = tx + rng.next(), y = ty + rng.next(); stroke(g, [[x, y], [x + 0.1, y + 0.02]], 0.01, shade(dark, -0.1), 0.5); }
  },
};

// ---------------------------------------------------------------- legacy dual-grid autotile lab

export const enum TerrainId {
  MudSwamp = 0,    // Base layer (#3b2f2f dark mud + murky puddles)
  MossGrass = 1,   // Secondary biome (#4d7c0f overgrown moss)
  AsphaltRoad = 2, // Road network (#1e293b slate asphalt + #facc15 yellow stripes)
}

export const TILE_SIZE = 32; // Base world grid tile size

/**
 * Computes the 4-corner dual-grid bitmask (0..15) at dual-grid vertex (vx, vy)
 */
export function getDualGridMask(
  grid: TerrainId[][],
  vx: number,
  vy: number,
  targetTerrain: TerrainId
): number {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const match = (tx: number, ty: number): boolean => {
    if (ty < 0 || ty >= rows || tx < 0 || tx >= cols) return false;
    return grid[ty][tx] === targetTerrain;
  };

  let mask = 0;
  if (match(vx, vy))         mask |= 1; // NW
  if (match(vx + 1, vy))     mask |= 2; // NE
  if (match(vx, vy + 1))     mask |= 4; // SW
  if (match(vx + 1, vy + 1)) mask |= 8; // SE
  return mask;
}

/**
 * Renders a single dual-grid transition tile at screen position (px, py)
 * using the 4-corner bitmask (NW=1, NE=2, SW=4, SE=8).
 */
export function drawDualGridSubtile(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  mask: number,
  fillPrimary: string,
  fillEdge: string,
  seed = 0
): void {
  if (mask === 0) return;

  const half = TILE_SIZE / 2; // 16px quadrants
  const corners = [
    { bit: 1, dx: 0, dy: 0 },       // NW
    { bit: 2, dx: half, dy: 0 },    // NE
    { bit: 4, dx: 0, dy: half },    // SW
    { bit: 8, dx: half, dy: half }, // SE
  ];

  // 1. Outer border / curb lip (1px dark charcoal or gravel edge)
  ctx.fillStyle = fillEdge;
  for (const c of corners) {
    if (mask & c.bit) {
      ctx.fillRect(px + c.dx - 1, py + c.dy - 1, half + 2, half + 2);
    }
  }

  // 2. Inner terrain fill
  ctx.fillStyle = fillPrimary;
  for (const c of corners) {
    if (mask & c.bit) {
      ctx.fillRect(px + c.dx, py + c.dy, half, half);
    }
  }

  // 3. Deterministic surface texture details
  if (mask === 15 && (seed % 5 === 0)) {
    ctx.fillStyle = fillEdge;
    ctx.fillRect(px + (seed % 20) + 4, py + ((seed >> 3) % 20) + 4, 3, 2);
  }
}

/**
 * Full Map Pass: Draws Mud Base -> Moss Dual-Grid -> Asphalt Road Dual-Grid + Yellow Centerlines
 */
export function renderTerrainMap(
  ctx: CanvasRenderingContext2D,
  grid: TerrainId[][]
): void {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (!rows || !cols) return;

  // Pass 1: Base Muddy Swamp Layer (#3b2f2f)
  ctx.fillStyle = '#3b2f2f';
  ctx.fillRect(0, 0, cols * TILE_SIZE, rows * TILE_SIZE);

  // Murky swamp puddles (#292020)
  ctx.fillStyle = '#292020';
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (((x * 37 + y * 13) & 7) === 0) {
        ctx.fillRect(x * TILE_SIZE + 6, y * TILE_SIZE + 8, 14, 8);
      }
    }
  }

  // Pass 2: Dual-Grid Mossy Wasteland Transition (#4d7c0f fill, #3f6212 edge)
  for (let vy = 0; vy < rows - 1; vy++) {
    for (let vx = 0; vx < cols - 1; vx++) {
      const px = vx * TILE_SIZE + TILE_SIZE / 2;
      const py = vy * TILE_SIZE + TILE_SIZE / 2;
      const seed = vx * 97 + vy * 31;

      const mossMask = getDualGridMask(grid, vx, vy, TerrainId.MossGrass);
      drawDualGridSubtile(ctx, px, py, mossMask, '#4d7c0f', '#3f6212', seed);
    }
  }

  // Pass 3: Dual-Grid Cracked Asphalt Road (#1e293b fill, #64748b gravel shoulder)
  for (let vy = 0; vy < rows - 1; vy++) {
    for (let vx = 0; vx < cols - 1; vx++) {
      const px = vx * TILE_SIZE + TILE_SIZE / 2;
      const py = vy * TILE_SIZE + TILE_SIZE / 2;
      const seed = vx * 53 + vy * 71;

      const roadMask = getDualGridMask(grid, vx, vy, TerrainId.AsphaltRoad);
      drawDualGridSubtile(ctx, px, py, roadMask, '#1e293b', '#64748b', seed);

      // Draw dashed #facc15 yellow road paint on full road interior tiles (mask === 15)
      if (roadMask === 15) {
        ctx.fillStyle = '#facc15';
        const isHorizontalRoad =
          grid[vy]?.[vx - 1] === TerrainId.AsphaltRoad ||
          grid[vy]?.[vx + 2] === TerrainId.AsphaltRoad;
        const isVerticalRoad =
          grid[vy - 1]?.[vx] === TerrainId.AsphaltRoad ||
          grid[vy + 2]?.[vx] === TerrainId.AsphaltRoad;

        if (isHorizontalRoad && !isVerticalRoad && vx % 2 === 0) {
          ctx.fillRect(px + 6, py + 14, 16, 3);
        } else if (isVerticalRoad && !isHorizontalRoad && vy % 2 === 0) {
          ctx.fillRect(px + 14, py + 6, 3, 16);
        }

        // Road surface cracks (#0f172a)
        if (seed % 3 === 0) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(px + 8, py + 6, 6, 1);
          ctx.fillRect(px + 13, py + 7, 1, 5);
        }
      }
    }
  }
}
