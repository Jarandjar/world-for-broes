import type { Rng } from '../../engine/rng';
import type { ObjectKind } from '../../data/schema';
import type { ObjDrawer } from '../objects';
import { POWER_POLE_SPACING } from '../../sim/worldgen';
import { type G, INK, TAU, inked, ell, cap, rrect, poly, lumpy, stroke, shade, alpha } from '../ink';

export interface ObjectRenderContext {
  ctx: CanvasRenderingContext2D;
  x: number;
  y: number;
  scale?: number;
  hpRatio?: number; // 0.0 (destroyed) to 1.0 (pristine), defaults to 1.0
  timeMs?: number;  // Optional timestamp for subtle idle animations (e.g. bubbling slime)
}

export type ObjectDrawerFn = (rc: ObjectRenderContext) => void;

const OUTLINE_DARK = '#1e293b';
const OUTLINE_DEEP = '#0f172a';

/**
 * Shared helper: Configures crisp 1px dark charcoal outlines (#1e293b)
 */
function applyStandardStroke(ctx: CanvasRenderingContext2D, color = OUTLINE_DARK): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
}

/**
 * Shared helper: Draws a 2.5D ground contact shadow (ambient occlusion without pure black)
 */
function drawGroundShadow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  rx: number,
  ry: number
): void {
  ctx.save();
  ctx.fillStyle = 'rgba(15, 23, 42, 0.28)'; // #0f172a at 28% alpha
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Shared helper: Draws a rounded polygon/pill sandbag in 2.5D (top face + front belly)
 */
function drawSandbagUnit(
  ctx: CanvasRenderingContext2D,
  bx: number,
  by: number,
  w: number,
  h: number
): void {
  const r = h * 0.45;
  // Base darkened burlap belly (front 2.5D face)
  ctx.fillStyle = '#b89b72';
  ctx.beginPath();
  ctx.roundRect(bx - w / 2, by - h / 2 + 2, w, h, r);
  ctx.fill();
  ctx.stroke();

  // Top sunlit burlap surface (15-degree top plane)
  ctx.fillStyle = '#d4b996';
  ctx.beginPath();
  ctx.roundRect(bx - w / 2, by - h / 2, w, h * 0.72, r * 0.8);
  ctx.fill();
  ctx.stroke();

  // Center burlap tie-off / seam crease
  ctx.strokeStyle = '#78350f';
  ctx.beginPath();
  ctx.moveTo(bx - w * 0.35, by - 1);
  ctx.lineTo(bx - w * 0.22, by - 1);
  ctx.stroke();
  ctx.strokeStyle = OUTLINE_DARK;
}

// ============================================================================
// 1. obj:slime_drum — Pressurized Bio-Slime Drum (32x36 footprint)
// Palette: Hot Pink (#ec4899), Duck Yellow (#facc15), Toxic Green (#22c55e),
//          Neon Cyan (#06b6d4), Steel Silver (#94a3b8), Outline (#1e293b)
// ============================================================================
export const drawSlimeDrum: ObjectDrawerFn = ({ ctx, x, y, scale = 1, hpRatio = 1.0, timeMs = 0 }) => {
  ctx.save();
  ctx.translate(x, y);
  if (scale !== 1) ctx.scale(scale, scale);
  applyStandardStroke(ctx);

  const rx = 13;          // Drum horizontal radius
  const ry = 6;           // 15° foreshortened top ellipse vertical radius
  const topY = -12;       // Top cap center Y
  const botY = 10;        // Bottom base center Y

  // 1. Ground shadow + leaking slime puddle if damaged
  drawGroundShadow(ctx, 0, botY + 3, rx + 3, ry + 2);
  if (hpRatio < 0.75) {
    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.ellipse(6, botY + 3, 9, 4, 0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // 2. Main Cylinder Body (Hot Pink #ec4899)
  ctx.fillStyle = '#ec4899';
  ctx.beginPath();
  ctx.moveTo(-rx, topY);
  ctx.lineTo(-rx, botY);
  ctx.ellipse(0, botY, rx, ry, 0, Math.PI, 0, true);
  ctx.lineTo(rx, topY);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 3. Center Hazard Band (Duck Yellow #facc15)
  const bandTop = -4;
  const bandBot = 3;
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.moveTo(-rx, bandTop);
  ctx.lineTo(-rx, bandBot);
  ctx.ellipse(0, bandBot, rx, ry * 0.85, 0, Math.PI, 0, true);
  ctx.lineTo(rx, bandTop);
  ctx.ellipse(0, bandTop, rx, ry * 0.85, 0, 0, Math.PI, false);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 4. Right-side cylindrical shading & left-side specular highlight (#94a3b8)
  ctx.fillStyle = 'rgba(30, 41, 59, 0.22)';
  ctx.fillRect(rx - 5, topY, 4, botY - topY);
  ctx.fillStyle = '#94a3b8';
  ctx.fillRect(-rx + 2, topY + 2, 2, botY - topY - 3);

  // 5. Corrugated Steel Chime Ribs (#94a3b8)
  ctx.strokeStyle = '#94a3b8';
  [-4, 3].forEach((ribY) => {
    ctx.beginPath();
    ctx.ellipse(0, ribY, rx + 0.5, ry * 0.85, 0, 0.15, Math.PI - 0.15);
    ctx.stroke();
  });
  ctx.strokeStyle = OUTLINE_DARK;

  // 6. 15° Top Lid Rim (Steel Silver #94a3b8) + Toxic Slime Pool (#22c55e)
  ctx.fillStyle = '#94a3b8';
  ctx.beginPath();
  ctx.ellipse(0, topY, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#22c55e';
  ctx.beginPath();
  ctx.ellipse(0, topY, rx - 2.5, ry - 1.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // 7. Neon Cyan (#06b6d4) Pressurized Valve Cap & Animated Slime Bubbles
  ctx.fillStyle = '#06b6d4';
  ctx.beginPath();
  ctx.ellipse(-4, topY - 1, 3, 1.8, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const pulse = Math.sin(timeMs * 0.008) * 0.8;
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.arc(4, topY + 0.5, 2.2 + pulse * 0.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // 8. Front Slime Overflow Drip
  ctx.fillStyle = '#22c55e';
  ctx.beginPath();
  ctx.moveTo(2, topY + ry - 1);
  ctx.quadraticCurveTo(4, topY + ry + 6, 3, topY + ry + 9);
  ctx.arc(4.5, topY + ry + 9, 1.8, Math.PI, 0, true);
  ctx.quadraticCurveTo(6, topY + ry + 5, 7, topY + ry - 1);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.restore();
};

// ============================================================================
// 2. obj:scrap_wall — Corrugated Junk Barricade (48x30 footprint)
// Palette: Weathered Wood (#78350f), Rust Orange (#ea580c),
//          Steel Silver (#94a3b8), Outline (#1e293b)
// ============================================================================
export const drawScrapWall: ObjectDrawerFn = ({ ctx, x, y, scale = 1, hpRatio = 1.0 }) => {
  ctx.save();
  ctx.translate(x, y);
  if (scale !== 1) ctx.scale(scale, scale);
  applyStandardStroke(ctx);

  drawGroundShadow(ctx, 0, 10, 24, 6);

  // 1. Rear Protruding Iron Rebar Stakes (#1e293b & #ea580c)
  [-17, 0, 17].forEach((rx) => {
    ctx.fillStyle = OUTLINE_DARK;
    ctx.fillRect(rx - 1.5, -15, 3, 24);
    ctx.fillStyle = '#ea580c';
    ctx.fillRect(rx - 0.5, -15, 1, 6);
  });

  // 2. Top Weathered Wood Support Beam (15° top plane + front edge)
  ctx.fillStyle = '#78350f';
  ctx.beginPath();
  ctx.roundRect(-22, -11, 44, 6, 1);
  ctx.fill();
  ctx.stroke();

  // 3. Corrugated Sheet Metal Body (#94a3b8 base with vertical fluted channels)
  ctx.fillStyle = '#94a3b8';
  ctx.beginPath();
  ctx.roundRect(-21, -7, 42, 17, 1.5);
  ctx.fill();
  ctx.stroke();

  // Vertical corrugated ribs (alternating slate shadow & silver highlight)
  for (let cx = -17; cx <= 17; cx += 4) {
    ctx.fillStyle = '#64748b';
    ctx.fillRect(cx, -6, 2, 15);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillRect(cx - 1, -6, 1, 15);
  }

  // 4. Rust Orange (#ea580c) Oxidation Patches & Salvaged Yield Sign Plate
  ctx.fillStyle = '#ea580c';
  ctx.beginPath();
  ctx.moveTo(-18, 2);
  ctx.lineTo(-8, -3);
  ctx.lineTo(-5, 9);
  ctx.lineTo(-19, 9);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Welded Diamond Scrap Patch (Right Side)
  ctx.fillStyle = '#ea580c';
  ctx.beginPath();
  ctx.moveTo(10, -5);
  ctx.lineTo(18, 1);
  ctx.lineTo(10, 8);
  ctx.lineTo(3, 1);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 5. Diagonal Weathered Wood Cross-Brace (#78350f)
  ctx.fillStyle = '#78350f';
  ctx.beginPath();
  ctx.moveTo(-22, 4);
  ctx.lineTo(22, -3);
  ctx.lineTo(22, 1);
  ctx.lineTo(-22, 8);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Rivet heads (#1e293b)
  ctx.fillStyle = OUTLINE_DARK;
  [-16, -4, 8, 17].forEach((nx) => {
    ctx.beginPath();
    ctx.arc(nx, 2 - nx * 0.14, 1.2, 0, Math.PI * 2);
    ctx.fill();
  });

  // 6. Damage state: jagged bullet holes & bent top corner
  if (hpRatio < 0.65) {
    ctx.fillStyle = OUTLINE_DEEP;
    ctx.beginPath();
    ctx.arc(-2, -1, 2.2, 0, Math.PI * 2);
    ctx.arc(12, 3, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
};

// ============================================================================
// 3. obj:tactical_shield — Aegis Tactical Deploy Shield (44x30 footprint)
// Palette: Slate Gray (#334155), Olive Drab (#3f6212), Gunmetal (#1e293b),
//          Hazard Amber (#f59e0b), Rim Light (#94a3b8)
// ============================================================================
export const drawTacticalShield: ObjectDrawerFn = ({ ctx, x, y, scale = 1, hpRatio = 1.0 }) => {
  ctx.save();
  ctx.translate(x, y);
  if (scale !== 1) ctx.scale(scale, scale);
  applyStandardStroke(ctx);

  drawGroundShadow(ctx, 0, 11, 22, 6);

  // 1. Rear Stabilizer Outriggers (Olive Drab #3f6212)
  ctx.fillStyle = '#3f6212';
  ctx.beginPath();
  ctx.roundRect(-19, -10, 6, 8, 1);
  ctx.roundRect(13, -10, 6, 8, 1);
  ctx.fill();
  ctx.stroke();

  // 2. Lower Olive Drab Blast Skirt (#3f6212)
  ctx.fillStyle = '#3f6212';
  ctx.beginPath();
  ctx.moveTo(-20, 4);
  ctx.lineTo(20, 4);
  ctx.lineTo(18, 11);
  ctx.lineTo(-18, 11);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 3. Main Ballistic Front Armor Plate (Slate Gray #334155)
  ctx.fillStyle = '#334155';
  ctx.beginPath();
  ctx.moveTo(-21, -5);
  ctx.lineTo(-14, -8);
  ctx.lineTo(14, -8);
  ctx.lineTo(21, -5);
  ctx.lineTo(20, 6);
  ctx.lineTo(13, 9);
  ctx.lineTo(-13, 9);
  ctx.lineTo(-20, 6);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 4. 15° Top Beveled Armor Lip (#94a3b8 specular metallic edge)
  ctx.fillStyle = '#94a3b8';
  ctx.beginPath();
  ctx.moveTo(-21, -5);
  ctx.lineTo(-14, -8);
  ctx.lineTo(14, -8);
  ctx.lineTo(21, -5);
  ctx.lineTo(14, -5);
  ctx.lineTo(-14, -5);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 5. Ballistic Glass Viewport Slit (Gunmetal frame #1e293b + Cyan/Steel tint)
  ctx.fillStyle = OUTLINE_DEEP;
  ctx.beginPath();
  ctx.roundRect(-11, -3, 22, 5, 1);
  ctx.fill();

  ctx.fillStyle = hpRatio > 0.4 ? '#38bdf8' : '#94a3b8';
  ctx.fillRect(-9, -1.5, 18, 2);

  // Cracked ballistic glass when damaged
  if (hpRatio < 0.7) {
    ctx.strokeStyle = OUTLINE_DEEP;
    ctx.beginPath();
    ctx.moveTo(-3, -2);
    ctx.lineTo(0, 1);
    ctx.lineTo(3, -1);
    ctx.stroke();
    ctx.strokeStyle = OUTLINE_DARK;
  }

  // 6. Hazard Amber (#f59e0b) Identification Chevrons on Side Wings
  ctx.fillStyle = '#f59e0b';
  // Left warning stripe
  ctx.beginPath();
  ctx.moveTo(-18, 0);
  ctx.lineTo(-14, -1);
  ctx.lineTo(-14, 4);
  ctx.lineTo(-18, 5);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Right warning stripe
  ctx.beginPath();
  ctx.moveTo(18, 0);
  ctx.lineTo(14, -1);
  ctx.lineTo(14, 4);
  ctx.lineTo(18, 5);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.restore();
};

// ============================================================================
// 4. obj:sandbags — Heavy Sandbag Barricade (48x28 footprint)
// Palette: Khaki Burlap (#d4b996), Olive Tarp (#3f6212),
//          Dark Iron Rebar (#1e293b), Weathered Cordage (#78350f)
// ============================================================================
export const drawSandbags: ObjectDrawerFn = ({ ctx, x, y, scale = 1, hpRatio = 1.0 }) => {
  ctx.save();
  ctx.translate(x, y);
  if (scale !== 1) ctx.scale(scale, scale);
  applyStandardStroke(ctx);

  // Footprint: 48px wide (-24 to +24), 28px high (-14 to +14)
  drawGroundShadow(ctx, 0, 9, 24, 6);

  // 1. Bottom Course (3 wide base bags)
  drawSandbagUnit(ctx, -14, 6, 16, 9);
  drawSandbagUnit(ctx, 14, 6, 16, 9);
  drawSandbagUnit(ctx, 0, 7, 17, 9);

  // 2. Middle Staggered Course (2 overlapping bags)
  drawSandbagUnit(ctx, -8, 0, 17, 9);
  drawSandbagUnit(ctx, 8, 0, 17, 9);

  // 3. Top Crest Course (2 fortified crown bags, right one omitted if heavily damaged)
  drawSandbagUnit(ctx, -7, -6, 15, 8);
  if (hpRatio > 0.35) {
    drawSandbagUnit(ctx, 7, -6, 15, 8);
  }

  // 4. Olive Drab (#3f6212) Camouflage Tarp Draped Over Right Crest
  ctx.fillStyle = '#3f6212';
  ctx.beginPath();
  ctx.moveTo(0, -10);
  ctx.lineTo(16, -9);
  ctx.lineTo(20, 2);
  ctx.lineTo(11, 5);
  ctx.lineTo(3, -1);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Tarp highlight fold
  ctx.strokeStyle = '#4d7c0f';
  ctx.beginPath();
  ctx.moveTo(3, -7);
  ctx.lineTo(13, 0);
  ctx.stroke();
  ctx.strokeStyle = OUTLINE_DARK;

  // 5. Embedded Rebar Anchor Pins (#1e293b with #94a3b8 top cap)
  [-13, 12].forEach((px) => {
    ctx.fillStyle = OUTLINE_DARK;
    ctx.beginPath();
    ctx.roundRect(px - 1.5, -12, 3, 5, 1);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillRect(px - 1, -12, 2, 1.5);
  });

  ctx.restore();
};

// ============================================================================
// Registry Map for World4Bros Object Renderer
// ============================================================================
export const OBJECT_DRAWERS: Record<string, ObjectDrawerFn> = {
  'obj:slime_drum': drawSlimeDrum,
  'obj:scrap_wall': drawScrapWall,
  'obj:tactical_shield': drawTacticalShield,
  'obj:sandbags': drawSandbags,
};

export function renderWorldObject(
  ctx: CanvasRenderingContext2D,
  drawerId: string,
  screenX: number,
  screenY: number,
  scale = 1,
  hp = 1,
  maxHp = 1,
  timeMs = 0
): void {
  const drawer = OBJECT_DRAWERS[drawerId];
  if (!drawer) return;

  drawer({
    ctx,
    x: screenX,
    y: screenY,
    scale,
    hpRatio: maxHp > 0 ? hp / maxHp : 1,
    timeMs,
  });
}

// ============================================================================
// Town street life (ink style, tile units, cached by SpriteCache via OBJECT_DRAWERS).
// Variant = facing (0..3: street to the N, E, S, W) + 4 * style; see TOWN_PROP_RULES in data/objects.ts.
// Tall props keep "up" as screen -y and only swing their arms/faces toward the street, so the
// top-left light and down-right shadows stay put. Looks (paints, names) are data tables below.
// ============================================================================
type P = [number, number];
/** Unit vector toward the street for each facing. */
export const FACING: P[] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const LIGHT: P = [-0.55, -0.8];
export const CAR_PAINTS = ['#a8473a', '#3f6a8a', '#c9a227', '#5f7d5a', '#d3cbb8'];
export const HYDRANT_PAINTS = ['#c8372d', '#d9a521', '#c8372d', '#3d6fb0'];
export const TRASH_PAINTS = ['#7c858e', '#4f6b4a', '#7c858e', '#5a6470'];
export const BENCH_WOODS = ['#9a6a3a', '#7d5a3c', '#a07b4c'];
export const POSTER_COLS = ['#d0643a', '#3f7fb0', '#e0b040', '#7a5aa0'];
export const STREET_NAMES: [string, string][] = [['MAIN ST', 'OAK AVE'], ['ELM ST', '1ST AVE'], ['PINE RD', '2ND ST'], ['MAPLE DR', 'MILL LN']];
export const LEAF_COLS = ['#b5652e', '#c9952f', '#8a5a2b', '#a2452c', '#7a7f3a'];

const face = (v: number): P => FACING[v & 3] as P;
const styleOf = (v: number, n: number): number => (v >> 2) % n;
/** Local street frame → sprite space: +x runs along the street, +y points at it. */
const rp = (v: number, x: number, y: number): P => { const [sx, sy] = face(v); return [x * sy + y * sx, -x * sx + y * sy]; };
const rps = (v: number, pts: P[]): P[] => pts.map(([x, y]) => rp(v, x, y));
const lift = (pts: P[], h: number): P[] => pts.map(([x, y]) => [x, y - h]);
const ik = (g: G, rng: Rng, path: () => void, fill: string, box: number, o: { hatch?: number; line?: number; light?: P; grain?: number; wear?: number } = {}): void => inked(g, path, fill, { rng, box, ...o });
/** Two-pass ink line: dark outline, then a coloured core. */
const wire = (g: G, pts: P[], w: number, col: string): void => { stroke(g, pts, w + 0.018, INK, 0.95); stroke(g, pts, w, col, 1); };
/** Rotate the canvas so local +x runs along the street; returns the light direction in that frame. */
function turned(g: G, v: number, draw: (light: P) => void): void {
  const [sx, sy] = face(v), ang = Math.atan2(-sx, sy), c = Math.cos(-ang), s = Math.sin(-ang);
  g.save(); g.rotate(ang); draw([LIGHT[0] * c - LIGHT[1] * s, LIGHT[0] * s + LIGHT[1] * c]); g.restore();
}
function label(g: G, text: string, x: number, y: number, size: number, col: string, sx = 1): void {
  g.save(); g.translate(x, y); g.scale(sx, 1); g.fillStyle = col; g.font = `bold ${size}px sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 0, size * 0.06); g.restore();
}
/** A tapered post from the ground (0,0) up to height h, with a lit stripe on the left. */
function post(g: G, rng: Rng, h: number, w0: number, w1: number, col: string): void {
  ik(g, rng, () => poly(g, [[-w0, 0], [w0, 0], [w1, -h], [-w1, -h]]), col, Math.max(0.3, h * 0.5), { hatch: 0.8, line: 0.016 });
  stroke(g, [[-w0 * 0.45, -0.04], [-w1 * 0.45, -h + 0.04]], Math.min(w0, w1) * 0.5, shade(col, 0.45), 0.55);
}
const footing = (g: G, rng: Rng, r: number, col = '#86827a'): void => ik(g, rng, () => ell(g, 0, 0, r, r * 0.7), col, r, { hatch: 0.5, line: 0.016 });

function streetLamp(g: G, rng: Rng, v: number): void {
  const [sx, sy] = face(v), H = 0.95, modern = styleOf(v, 2) === 1;
  footing(g, rng, 0.17);
  ik(g, rng, () => ell(g, 0, -0.02, 0.085, 0.06), '#3a3e45', 0.12, { hatch: 0.4, line: 0.012 });
  post(g, rng, H, 0.05, 0.03, modern ? '#7d858e' : '#3b4048');
  for (const h of [0.16, 0.52]) stroke(g, [[-0.045, -h], [0.045, -h]], 0.016, INK, 0.8);
  const ex = sx * 0.42 + (sx === 0 ? 0.16 : 0), ey = -H - 0.05 + sy * 0.22;
  wire(g, [[0, -H], [ex * 0.3, -H - 0.13 + sy * 0.05], [ex, ey]], 0.022, modern ? '#7d858e' : '#3b4048');
  const ang = Math.atan2(ey + H, ex);
  ik(g, rng, () => ell(g, ex, ey, 0.13, 0.065, ang), modern ? '#5f666f' : '#2f343b', 0.15, { hatch: 0.6, line: 0.016 });
  ik(g, rng, () => ell(g, ex + Math.cos(ang) * 0.02, ey + 0.03, 0.075, 0.035, ang), '#fff1b0', 0.09, { hatch: 0, line: 0.01 });
  const glow = g.createRadialGradient(ex, ey + 0.05, 0, ex, ey + 0.05, 0.42);
  glow.addColorStop(0, 'rgba(255,226,140,0.32)'); glow.addColorStop(1, 'rgba(255,226,140,0)');
  g.fillStyle = glow; g.beginPath(); g.arc(ex, ey + 0.05, 0.42, 0, TAU); g.fill();
  if (!modern) for (let i = 0; i < 3; i++) stroke(g, [[rng.range(-0.03, 0.03), -rng.range(0.1, 0.8)], [rng.range(-0.03, 0.03), -rng.range(0.1, 0.8)]], 0.006, '#8a5a33', 0.5);   // rust runs
}

function hydrant(g: G, rng: Rng, v: number): void {
  const [sx, sy] = face(v), red = HYDRANT_PAINTS[styleOf(v, 4)] as string, H = 0.32;
  ik(g, rng, () => ell(g, 0, 0.01, 0.17, 0.1), '#6b6f76', 0.2, { hatch: 0.5, line: 0.016 });
  ik(g, rng, () => { g.moveTo(-0.11, -H); g.lineTo(-0.11, 0); g.ellipse(0, 0, 0.11, 0.055, 0, Math.PI, 0, true); g.lineTo(0.11, -H); g.closePath(); }, red, 0.3, { hatch: 0.9, line: 0.02 });
  g.beginPath(); g.ellipse(0, -0.2, 0.115, 0.05, 0, 0.1, Math.PI - 0.1); g.strokeStyle = alpha(INK, 0.7); g.lineWidth = 0.012; g.stroke();
  for (const s of [-1, 1]) ik(g, rng, () => cap(g, s * 0.09, -0.17, s * 0.16, -0.17, 0.045), '#b8bec6', 0.08, { hatch: 0.3, line: 0.012 });
  stroke(g, [[-0.16, -0.15], [-0.13, -0.08], [-0.09, -0.11]], 0.006, '#c9cdd2', 0.8);                                       // chain
  ik(g, rng, () => ell(g, sx * 0.07, -0.13 + sy * 0.05, 0.065, 0.06), shade(red, -0.15), 0.08, { hatch: 0.4, line: 0.014 });   // pumper nozzle
  ik(g, rng, () => ell(g, sx * 0.07, -0.13 + sy * 0.05, 0.035, 0.032), '#c9a54a', 0.05, { hatch: 0, line: 0.01 });
  ik(g, rng, () => ell(g, 0, -H, 0.12, 0.07), shade(red, 0.18), 0.14, { hatch: 0.4, line: 0.016 });
  ik(g, rng, () => ell(g, 0, -H - 0.04, 0.04, 0.03), '#c9a54a', 0.05, { hatch: 0, line: 0.01 });
  stroke(g, [[-0.075, -0.27], [-0.075, -0.04]], 0.016, '#f1e6d0', 0.45);
  if (styleOf(v, 4) === 2) { g.fillStyle = alpha('#5d8fb0', 0.35); g.beginPath(); g.ellipse(sx * 0.2 + 0.08, 0.1 + sy * 0.08, 0.2, 0.08, 0.3, 0, TAU); g.fill(); }   // leaking puddle
}

function bench(g: G, rng: Rng, v: number): void {
  const wood = BENCH_WOODS[styleOf(v, 3)] as string, broken = styleOf(v, 3) === 1, iron = '#2b3029';
  for (const x of [-0.36, 0.36]) {
    ik(g, rng, () => poly(g, rps(v, [[x - 0.035, -0.2], [x + 0.035, -0.2], [x + 0.035, 0.15], [x - 0.035, 0.15]])), iron, 0.25, { hatch: 0.6, line: 0.014 });
    ik(g, rng, () => poly(g, lift(rps(v, [[x - 0.03, -0.22], [x + 0.03, -0.22], [x + 0.03, -0.16], [x - 0.03, -0.16]]), 0.14)), iron, 0.1, { hatch: 0.3, line: 0.012 });
  }
  const slat = (y: number, h: number, lifted: number, skew = 0): void => {
    const pts = lift(rps(v, [[-0.45, y], [0.45, y + skew], [0.45, y + h + skew], [-0.45, y + h]]), lifted);
    ik(g, rng, () => poly(g, pts), wood, 0.35, { hatch: 0.5, line: 0.014, wear: 0.8 });
    for (let k = 0; k < 2; k++) { const a = rng.range(-0.4, 0.1), yy = y + h * rng.range(0.3, 0.7); stroke(g, lift(rps(v, [[a, yy], [a + rng.range(0.15, 0.3), yy]]), lifted), 0.006, shade(wood, -0.5), 0.6); }
    for (const x of [-0.36, 0.36]) { const p = lift(rps(v, [[x, y + h / 2]]), lifted)[0] as P; g.fillStyle = INK; g.beginPath(); g.arc(p[0], p[1], 0.012, 0, TAU); g.fill(); }
  };
  slat(-0.24, 0.06, 0.16); slat(-0.16, 0.06, 0.16);   // backrest, away from the street
  slat(-0.07, 0.065, 0.02); if (!broken) slat(0.01, 0.065, 0.02); else slat(0.03, 0.06, 0.0, 0.06); slat(0.09, 0.065, 0.02);
}

function stopSign(g: G, rng: Rng, v: number): void {
  const [, sy] = face(v), st = styleOf(v, 3), H = 0.88, k = sy !== 0 ? 1 : 0.62, bent = st === 2 ? 0.16 : 0;
  footing(g, rng, 0.08);
  post(g, rng, H - 0.1, 0.025, 0.022, '#8d949b');
  for (let y = 0.15; y < H - 0.15; y += 0.07) { g.fillStyle = alpha(INK, 0.5); g.beginPath(); g.arc(0, -y, 0.006, 0, TAU); g.fill(); }   // punched holes
  g.save(); g.translate(0, -H); g.rotate(bent);
  const oct = (r: number): P[] => Array.from({ length: 8 }, (_, i) => { const a = Math.PI / 8 + (i * Math.PI) / 4; return [Math.cos(a) * r * k, Math.sin(a) * r] as P; });
  if (sy < 0) {   // street to the north: we see the grey back and its bracket
    ik(g, rng, () => poly(g, oct(0.2)), '#9aa1a8', 0.22, { hatch: 0.7, line: 0.018 });
    ik(g, rng, () => rrect(g, -0.025, -0.12, 0.05, 0.24, 0.01), '#6b7178', 0.12, { hatch: 0.3, line: 0.01 });
  } else {
    ik(g, rng, () => poly(g, oct(0.2)), '#f1ead8', 0.22, { hatch: 0.3, line: 0.018 });
    ik(g, rng, () => poly(g, oct(0.17)), '#c0352b', 0.2, { hatch: 0.6, line: 0.01 });
    label(g, 'STOP', 0, 0, 0.085, '#f4efe2', k);
    stroke(g, [[-0.1 * k, -0.11], [-0.03 * k, -0.15]], 0.012, '#f4efe2', 0.35);   // sheen
    if (st === 1) for (let i = 0; i < 3; i++) { const x = rng.range(-0.1, 0.1) * k, y = rng.range(-0.1, 0.1); g.fillStyle = '#d9d4c8'; g.beginPath(); g.arc(x, y, 0.02, 0, TAU); g.fill(); g.fillStyle = INK; g.beginPath(); g.arc(x, y, 0.011, 0, TAU); g.fill(); }
  }
  g.restore();
}

function streetSign(g: G, rng: Rng, v: number): void {
  const [a, b] = STREET_NAMES[styleOf(v, 4)] as [string, string], H = 1.0, green = '#2f6b48';
  footing(g, rng, 0.08);
  post(g, rng, H, 0.025, 0.022, '#8d949b');
  ik(g, rng, () => ell(g, 0, -H - 0.01, 0.03, 0.02), '#6b7178', 0.04, { hatch: 0, line: 0.01 });
  // the blade along the N-S street is foreshortened (seen end-on from above)
  ik(g, rng, () => poly(g, [[-0.07, -H - 0.2], [0.07, -H - 0.24], [0.07, -H - 0.16], [-0.07, -H - 0.12]]), shade(green, -0.1), 0.15, { hatch: 0.6, line: 0.014 });
  label(g, b.slice(0, 3), 0, -H - 0.18, 0.045, '#eef2ea', 0.8);
  ik(g, rng, () => rrect(g, -0.34, -H - 0.07, 0.68, 0.12, 0.02), green, 0.35, { hatch: 0.5, line: 0.016 });
  g.strokeStyle = alpha('#eef2ea', 0.85); g.lineWidth = 0.008; g.beginPath(); rrect(g, -0.32, -H - 0.055, 0.64, 0.09, 0.015); g.stroke();
  label(g, a, 0, -H - 0.01, 0.06, '#eef2ea');
}

function busStop(g: G, rng: Rng, v: number): void {
  const st = styleOf(v, 4), RH = 0.55, glass = '#8fb2c4', frame = '#4a535c';
  ik(g, rng, () => poly(g, rps(v, [[-0.8, -0.38], [0.8, -0.38], [0.8, 0.32], [-0.8, 0.32]])), '#a29d92', 0.8, { hatch: 0.4, line: 0.018, grain: 1, wear: 0.9 });
  for (const x of [-0.4, 0, 0.4]) stroke(g, rps(v, [[x, -0.38], [x, 0.32]]), 0.007, INK, 0.35);                        // slab joints
  // bench inside
  ik(g, rng, () => poly(g, lift(rps(v, [[-0.55, -0.26], [0.55, -0.26], [0.55, -0.12], [-0.55, -0.12]]), 0.12)), BENCH_WOODS[st % 3] as string, 0.5, { hatch: 0.5, line: 0.014 });
  // back glass panel: from the pad's back edge up to the roof
  const back = rps(v, [[-0.75, -0.32], [0.75, -0.32]]) as [P, P];
  const pane: P[] = [back[0], back[1], [back[1][0], back[1][1] - RH], [back[0][0], back[0][1] - RH]];
  inked(g, () => poly(g, pane), glass, { rng, box: 0.8, hatch: 0, line: 0.016, noWash: true });
  g.save(); g.globalAlpha = 0.5; for (let i = 0; i < 3; i++) { const t = 0.2 + i * 0.25; const p = [back[0][0] + (back[1][0] - back[0][0]) * t, back[0][1] + (back[1][1] - back[0][1]) * t]; stroke(g, [[p[0]! - 0.06, p[1]! - 0.08], [p[0]! + 0.06, p[1]! - RH + 0.1]], 0.02, '#eef6fa', 0.6); } g.restore();
  // ad poster on one end
  const ad = rps(v, [[0.38, -0.33], [0.7, -0.33]]) as [P, P];
  ik(g, rng, () => poly(g, [[ad[0][0], ad[0][1] - 0.06], [ad[1][0], ad[1][1] - 0.06], [ad[1][0], ad[1][1] - RH + 0.08], [ad[0][0], ad[0][1] - RH + 0.08]]), POSTER_COLS[st] as string, 0.3, { hatch: 0.3, line: 0.012 });
  // corner posts
  for (const [x, y] of [[-0.75, -0.32], [0.75, -0.32], [-0.75, 0.22], [0.75, 0.22]] as P[]) { const p = rp(v, x, y); wire(g, [p, [p[0], p[1] - RH]], 0.018, frame); }
  // roof
  const roof = lift(rps(v, [[-0.85, -0.42], [0.85, -0.42], [0.85, 0.28], [-0.85, 0.28]]), RH);
  ik(g, rng, () => poly(g, roof), '#5a646e', 0.9, { hatch: 0.8, line: 0.022 });
  stroke(g, lift(rps(v, [[-0.8, -0.36], [0.8, -0.36]]), RH), 0.012, '#b6c0c8', 0.6);
  if (st === 3) stroke(g, lift(rps(v, [[-0.2, -0.3], [0.1, 0.0], [0.3, 0.2]]), RH), 0.01, INK, 0.7);   // crack
  // bus sign pole on the street side
  const sp = rp(v, 0.68, 0.26);
  g.save(); g.translate(sp[0], sp[1]); post(g, rng, 0.75, 0.02, 0.018, '#8d949b');
  ik(g, rng, () => ell(g, 0, -0.78, 0.11, 0.11), '#2f6db0', 0.12, { hatch: 0.4, line: 0.014 });
  label(g, 'BUS', 0, -0.78, 0.06, '#f4efe2'); g.restore();
}

function trashCan(g: G, rng: Rng, v: number): void {
  const st = styleOf(v, 4), col = TRASH_PAINTS[st] as string, H = 0.36;
  if (st === 3) {   // knocked over, lid off, spilled
    ik(g, rng, () => cap(g, -0.22, 0, 0.14, 0.03, 0.15, 0.16), col, 0.3, { hatch: 1, line: 0.02 });
    for (const x of [-0.12, -0.02, 0.08]) stroke(g, [[x, -0.14], [x + 0.01, 0.17]], 0.01, shade(col, -0.4), 0.7);
    ik(g, rng, () => ell(g, 0.16, 0.03, 0.06, 0.15), '#26292d', 0.16, { hatch: 0, line: 0.016 });
    ik(g, rng, () => ell(g, -0.3, 0.26, 0.17, 0.08, -0.3), shade(col, 0.15), 0.18, { hatch: 0.4, line: 0.014 });
    for (let i = 0; i < 6; i++) ik(g, rng, () => lumpy(g, 0.28 + rng.range(0, 0.22), rng.range(-0.15, 0.2), 0.05, 0.04, rng, 6, 0.3), i % 3 ? '#e6e0cf' : '#a8b07a', 0.06, { hatch: 0.2, line: 0.01 });
    return;
  }
  ik(g, rng, () => { g.moveTo(-0.15, -H); g.lineTo(-0.13, 0); g.ellipse(0, 0, 0.13, 0.06, 0, Math.PI, 0, true); g.lineTo(0.15, -H); g.closePath(); }, col, 0.3, { hatch: 1, line: 0.02, wear: 0.9 });
  for (const x of [-0.09, -0.03, 0.03, 0.09]) stroke(g, [[x * 1.1, -H + 0.04], [x, -0.02]], 0.009, shade(col, -0.45), 0.7);   // ribs
  for (const y of [-0.26, -0.1]) { g.beginPath(); g.ellipse(0, y, 0.14, 0.045, 0, 0.15, Math.PI - 0.15); g.strokeStyle = alpha(INK, 0.6); g.lineWidth = 0.01; g.stroke(); }
  if (st === 1) { g.fillStyle = alpha(INK, 0.25); g.beginPath(); g.ellipse(0.05, -0.15, 0.05, 0.035, 0.4, 0, TAU); g.fill(); }   // dent
  ik(g, rng, () => ell(g, 0, -H, 0.17, 0.075), shade(col, 0.18), 0.18, { hatch: 0.4, line: 0.016 });
  ik(g, rng, () => lumpy(g, 0.08, -H - 0.03, 0.06, 0.03, rng, 6, 0.3), '#e6e0cf', 0.07, { hatch: 0.2, line: 0.01 });   // paper poking out
  ik(g, rng, () => cap(g, -0.05, -H - 0.02, 0.05, -H - 0.02, 0.018), '#3b4048', 0.06, { hatch: 0, line: 0.01 });          // handle
}

/** Power poles carry their half of each cable span to the neighbours; a missing neighbour leaves a snapped, dangling cable.
 *  Variant: bit 0 = line runs along y, bit 1 = pole to the W/N, bit 2 = pole to the E/S, bit 3 = transformer. */
function powerPole(g: G, rng: Rng, v: number): void {
  const alongY = (v & 1) === 1, prev = (v & 2) !== 0, next = (v & 4) !== 0, H = 1.35, D = POWER_POLE_SPACING, wood = '#6b5038';
  footing(g, rng, 0.14, '#6f6a5f');
  post(g, rng, H, 0.065, 0.045, wood);
  for (let i = 0; i < 4; i++) { const y = -rng.range(0.1, H - 0.1); stroke(g, [[rng.range(-0.03, 0.01), y], [rng.range(-0.01, 0.03), y - rng.range(0.1, 0.25)]], 0.006, shade(wood, -0.5), 0.6); }
  for (let y = 0.35, s = 1; y < H - 0.2; y += 0.17, s = -s) stroke(g, [[0, -y], [s * 0.09, -y + 0.01]], 0.016, '#5c6168', 0.9);   // climbing steps
  if (v & 8) { ik(g, rng, () => rrect(g, 0.05, -H + 0.25, 0.14, 0.24, 0.05), '#7d858e', 0.15, { hatch: 0.7, line: 0.014 }); stroke(g, [[0.12, -H + 0.25], [0.04, -H + 0.08]], 0.008, INK, 0.8); }
  const offs = [-0.24, 0, 0.24];
  if (alongY) ik(g, rng, () => rrect(g, -0.32, -H - 0.04, 0.64, 0.07, 0.015), shade(wood, -0.1), 0.35, { hatch: 0.5, line: 0.014 });
  else ik(g, rng, () => rrect(g, -0.04, -H - 0.3, 0.07, 0.6, 0.015), shade(wood, -0.1), 0.35, { hatch: 0.5, line: 0.014 });
  const sag = (u: number, S: number): number => (S * 4 * u * (D - u)) / (D * D);
  for (const o of offs) {
    for (const dirn of [-1, 1]) {
      const has = dirn < 0 ? prev : next;
      const pts: P[] = [];
      if (has) for (let i = 0; i <= 10; i++) { const u = (i / 10) * (D / 2); pts.push(alongY ? [o + sag(u, 0.2), -H + dirn * u] : [dirn * u, -H + o + sag(u, 0.38)]); }
      else if (o === 0) pts.push([0, -H], alongY ? [0.12, -H + dirn * 0.35] : [dirn * 0.35, -H + 0.2], alongY ? [0.3, -H * 0.45 + dirn * 0.45] : [dirn * 0.6, -H * 0.4], alongY ? [0.26, 0.05 + dirn * 0.4] : [dirn * 0.5, 0.06], alongY ? [0.4, 0.1 + dirn * 0.3] : [dirn * 0.66, 0.1]);
      if (pts.length) { stroke(g, pts, 0.026, INK, 0.9); stroke(g, pts, 0.01, '#3a3d42', 1); }
    }
    const ix = alongY ? o : 0, iy = alongY ? -H - 0.06 : -H + o - 0.03;
    ik(g, rng, () => ell(g, ix, iy, 0.035, 0.03), '#9cc7b6', 0.05, { hatch: 0, line: 0.01 });   // glass insulators
  }
}

function abandonedCar(g: G, rng: Rng, v: number): void {
  const st = styleOf(v, CAR_PAINTS.length), paint = CAR_PAINTS[st] as string, flat = rng.int(0, 3), flat2 = rng.chance(0.4) ? (flat + 2) % 4 : -1;
  turned(g, v, (light) => {
    const lk = (path: () => void, fill: string, box: number, o: { hatch?: number; line?: number; wear?: number } = {}): void => ik(g, rng, path, fill, box, { light, ...o });
    [[0.58, -0.45], [0.58, 0.45], [-0.6, -0.45], [-0.6, 0.45]].forEach(([x, y], i) => {
      const f = i === flat || i === flat2;
      lk(() => ell(g, x!, y! + (f ? Math.sign(y!) * 0.02 : 0), f ? 0.2 : 0.17, f ? 0.1 : 0.075), '#1d1f22', 0.2, { hatch: 0.3, line: 0.014 });
      if (f) { stroke(g, [[x! - 0.15, y! + Math.sign(y!) * 0.06], [x! + 0.15, y! + Math.sign(y!) * 0.07]], 0.01, '#4a4d52', 0.8); }
      else { g.fillStyle = '#8d949b'; g.beginPath(); g.ellipse(x!, y!, 0.06, 0.03, 0, 0, TAU); g.fill(); }
    });
    lk(() => rrect(g, -0.95, -0.46, 1.9, 0.92, 0.22), paint, 1, { hatch: 1, line: 0.026, wear: 1 });
    // chrome bumpers, lights
    for (const x of [-0.97, 0.93]) lk(() => rrect(g, x, -0.38, 0.05, 0.76, 0.02), '#a7adb3', 0.4, { hatch: 0.2, line: 0.012 });
    for (const y of [-0.3, 0.3]) { lk(() => ell(g, 0.88, y, 0.045, 0.06), '#f2e6b0', 0.07, { hatch: 0, line: 0.01 }); lk(() => rrect(g, -0.94, y - 0.06, 0.04, 0.12, 0.01), '#b33a30', 0.07, { hatch: 0, line: 0.01 }); }
    // hood and trunk seams
    stroke(g, [[0.45, -0.36], [0.45, 0.36]], 0.009, INK, 0.6); stroke(g, [[-0.72, -0.36], [-0.72, 0.36]], 0.009, INK, 0.6);
    stroke(g, [[0.5, 0], [0.86, 0]], 0.008, shade(paint, -0.5), 0.6);
    // cabin: windscreen, roof, rear window
    lk(() => poly(g, [[0.2, -0.33], [0.43, -0.38], [0.43, 0.38], [0.2, 0.33]]), '#6f8fa0', 0.35, { hatch: 0.4, line: 0.016 });
    for (let i = 0; i < 5; i++) { const a = rng.range(0, TAU); stroke(g, [[0.32, 0.05], [0.32 + Math.cos(a) * 0.12, 0.05 + Math.sin(a) * 0.2]], 0.006, '#eef6fa', 0.8); }   // cracked
    lk(() => poly(g, [[-0.62, -0.32], [-0.48, -0.3], [-0.48, 0.3], [-0.62, 0.32]]), '#5d7a8a', 0.3, { hatch: 0.4, line: 0.016 });
    lk(() => rrect(g, -0.48, -0.34, 0.68, 0.68, 0.08), shade(paint, 0.12), 0.45, { hatch: 0.6, line: 0.018 });
    for (const s of [-1, 1]) stroke(g, [[-0.44, s * 0.39], [0.16, s * 0.39]], 0.03, '#4f6470', 0.9);   // side windows
    for (const s of [-1, 1]) lk(() => ell(g, 0.2, s * 0.5, 0.04, 0.03), shade(paint, -0.2), 0.05, { hatch: 0, line: 0.01 });   // mirrors
    // open driver door on the kerb side (-y), showing the seat
    lk(() => rrect(g, -0.22, -0.45, 0.4, 0.16, 0.03), '#3a3430', 0.25, { hatch: 0.6, line: 0.012 });
    const hx = 0.18, hy = -0.46, b = 1.0, ex = hx - Math.cos(b) * 0.46, ey = hy - Math.sin(b) * 0.46;
    lk(() => cap(g, hx, hy, ex, ey, 0.035), paint, 0.3, { hatch: 0.6, line: 0.016 });
    stroke(g, [[hx - Math.cos(b) * 0.1, hy - Math.sin(b) * 0.1 + 0.015], [hx - Math.cos(b) * 0.3, hy - Math.sin(b) * 0.3 + 0.015]], 0.012, '#4f6470', 0.9);
    if (st === 2) lk(() => rrect(g, 0.47, -0.35, 0.38, 0.7, 0.04), shade(paint, 0.1), 0.4, { hatch: 0.8, line: 0.016 });   // hood popped
    for (let i = 0; i < 3; i++) lk(() => lumpy(g, rng.range(-0.8, 0.8), rng.range(-0.35, 0.35), rng.range(0.05, 0.11), rng.range(0.04, 0.08), rng, 7, 0.3), '#8a5a33', 0.12, { hatch: 0.3, line: 0.008 });   // rust
    g.fillStyle = alpha('#c2b9a3', 0.18); for (let i = 0; i < 40; i++) { g.beginPath(); g.arc(rng.range(-0.9, 0.9), rng.range(-0.42, 0.42), 0.008, 0, TAU); g.fill(); }   // dust
  });
}

function leaf(g: G, rng: Rng, x: number, y: number, a: number, l: number, col: string): void {
  const c = Math.cos(a), s = Math.sin(a), w = l * 0.38, tip: P = [x + c * l, y + s * l];
  ik(g, rng, () => { g.moveTo(x, y); g.quadraticCurveTo(x + c * l * 0.5 - s * w, y + s * l * 0.5 + c * w, tip[0], tip[1]); g.quadraticCurveTo(x + c * l * 0.5 + s * w, y + s * l * 0.5 - c * w, x, y); g.closePath(); }, col, l, { hatch: 0.2, line: 0.008, grain: 0.2 });
  stroke(g, [[x, y], tip], 0.005, shade(col, -0.5), 0.7);
}
function newspaper(g: G, rng: Rng, x: number, y: number, a: number): void {
  g.save(); g.translate(x, y); g.rotate(a);
  ik(g, rng, () => poly(g, [[-0.2, -0.14], [0.2, -0.15], [0.21, 0.13], [-0.19, 0.15]]), '#e6e0cf', 0.25, { hatch: 0.3, line: 0.012, wear: 0.8 });
  stroke(g, [[0.0, -0.15], [0.01, 0.14]], 0.006, INK, 0.4);   // fold
  stroke(g, [[-0.17, -0.1], [-0.03, -0.1]], 0.025, INK, 0.75);  // headline
  g.fillStyle = alpha('#6f747b', 0.7); g.fillRect(0.04, -0.12, 0.13, 0.09);
  for (let r = -0.05; r < 0.12; r += 0.03) { stroke(g, [[-0.17, r], [-0.03, r]], 0.006, INK, 0.35); stroke(g, [[0.04, r + 0.03], [0.17, r + 0.03]], 0.006, INK, 0.35); }
  g.restore();
}
function litter(g: G, rng: Rng, v: number): void {
  const st = styleOf(v, 4);
  if (st === 0 || st === 3) newspaper(g, rng, rng.range(-0.08, 0.08), rng.range(-0.05, 0.05), rng.range(-0.6, 0.6));
  if (st === 0) newspaper(g, rng, 0.1, 0.1, rng.range(0.5, 1.4));
  if (st === 1 || st === 3) for (let i = 0, n = st === 1 ? 10 : 4; i < n; i++) leaf(g, rng, rng.range(-0.3, 0.3), rng.range(-0.25, 0.25), rng.range(0, TAU), rng.range(0.07, 0.11), LEAF_COLS[rng.int(0, LEAF_COLS.length - 1)] as string);
  if (st === 2) {
    ik(g, rng, () => cap(g, -0.12, 0.02, 0.02, -0.03, 0.045, 0.05), '#b33a30', 0.1, { hatch: 0.5, line: 0.012 });   // crushed can
    ik(g, rng, () => ell(g, 0.02, -0.03, 0.025, 0.05, 0.3), '#c9cdd2', 0.05, { hatch: 0, line: 0.01 });
    ik(g, rng, () => cap(g, 0.08, 0.1, 0.22, 0.14, 0.04, 0.055), '#efe9dc', 0.12, { hatch: 0.3, line: 0.012 });     // paper cup
    stroke(g, [[0.14, 0.07], [0.13, 0.17]], 0.018, '#c0352b', 0.85);
    ik(g, rng, () => lumpy(g, -0.15, -0.14, 0.07, 0.05, rng, 7, 0.35), '#d9b44a', 0.08, { hatch: 0.2, line: 0.01 });  // wrapper
  }
}

/** Registry of town prop drawers, merged into OBJECT_DRAWERS. `upright` props keep their facing (no random spin). */
export const TOWN_PROP_DRAWERS: Partial<Record<ObjectKind, ObjDrawer>> = {
  street_light: { bounds: [-0.65, -1.45, 0.65, 0.35], shadow: [0.32, 0.16, 0.25, 0.1], upright: true, draw: streetLamp },
  fire_hydrant: { bounds: [-0.45, -0.5, 0.45, 0.35], shadow: [0.2, 0.12, 0.08, 0.06], upright: true, draw: hydrant },
  park_bench: { bounds: [-0.6, -0.75, 0.6, 0.6], shadow: [0.42, 0.22, 0.08, 0.1], upright: true, draw: bench },
  stop_sign: { bounds: [-0.4, -1.2, 0.4, 0.2], shadow: [0.22, 0.1, 0.18, 0.06], upright: true, draw: stopSign },
  street_sign: { bounds: [-0.45, -1.35, 0.45, 0.2], shadow: [0.25, 0.1, 0.2, 0.06], upright: true, draw: streetSign },
  bus_stop: { bounds: [-1.0, -1.6, 1.0, 1.0], shadow: [0.85, 0.45, 0.25, 0.15], upright: true, draw: busStop },
  trash_can: { bounds: [-0.55, -0.55, 0.55, 0.4], shadow: [0.2, 0.12, 0.08, 0.06], upright: true, draw: trashCan },
  power_pole: { bounds: [-3.1, -4.5, 3.1, 1.8], shadow: [0.3, 0.14, 0.3, 0.08], upright: true, draw: powerPole },
  abandoned_car: { bounds: [-1.3, -1.3, 1.3, 1.3], shadow: [0.85, 0.72, 0.12, 0.15], upright: true, draw: abandonedCar },
  litter: { bounds: [-0.45, -0.45, 0.45, 0.45], shadow: [0, 0, 0, 0], draw: litter },
};
