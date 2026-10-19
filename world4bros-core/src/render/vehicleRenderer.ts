import type { Rng } from '../engine/rng';
import { makeRng, hashStr } from '../engine/rng';
import type { VehicleC, Transform } from '../sim/components';
import type { VehicleKind } from '../data/schema';
import { vehicleLook, type VehicleDecal, type VehicleLook } from '../data/vehicles';
import { type G, INK, TAU, inked, ell, cap, rrect, poly, smooth, stroke, alpha, shade, lumpy, star } from './ink';
import { SpriteCache, blit } from './sprites';

const ik = (g: G, rng: Rng, path: () => void, fill: string, box: number, o: { hatch?: number; line?: number; wear?: number; grain?: number } = {}): void =>
  inked(g, path, fill, { rng, box, ...o });

/** Lamp lens and glow colours (render-only). */
const LAMP = { head: '#fdeb90', headOff: '#cfd6dc', tail: '#b53828', tailGlow: '#ff4d3a', amber: '#f5a524', reverse: '#f4f1e6', shine: '#ffffff' };

/** Everything a cached body drawer needs. `line` is the body outline: thicker when the vehicle has a driver (hero). */
export interface VehCtx {
  id: string; kind: VehicleKind; rng: Rng; halfL: number; halfW: number;
  color: string; look: Required<VehicleLook>; dmg: number; hero: boolean; lights: boolean; line: number;
}
export type BodyDrawer = (g: G, c: VehCtx) => void;
export type DecalPainter = (g: G, c: VehCtx) => void;
type Anim = (g: G, v: VehicleC, halfL: number, halfW: number, time: number) => void;

/** Lamp layout per kind as fractions of half-length / half-width: head lamp, its radius, and tail lamp. */
const LAMPS: Record<VehicleKind, { head: [number, number]; headR: number; tail: [number, number] }> = {
  hatchback: { head: [0.94, 0.65], headR: 0.065, tail: [-0.99, 0.72] },
  clown_car: { head: [0.88, 0.55], headR: 0.11, tail: [-0.92, 0.42] },
  ice_cream_truck: { head: [0.94, 0.65], headR: 0.08, tail: [-0.95, 0.74] },
  golf_cart: { head: [0.85, 0.45], headR: 0.06, tail: [-0.93, 0.5] },
  bathtub_buggy: { head: [0.85, 0.5], headR: 0.07, tail: [-0.9, 0.55] },
  hot_dog_van: { head: [0.92, 0.72], headR: 0.08, tail: [-0.93, 0.74] },
  tesla_roadster: { head: [0.92, 0.55], headR: 0.07, tail: [-0.95, 0.62] },
};

/** Damage bucket 0..3 from hp: dents, rust, scratches and cracked glass build up as it rises. */
export const damageLevel = (hp: number, maxHp: number): number => {
  const f = maxHp > 0 ? hp / maxHp : 1;
  return f > 0.75 ? 0 : f > 0.5 ? 1 : f > 0.25 ? 2 : 3;
};

// ---------------------------------------------------------------- shared painters (all draw in vehicle-local units, +x = forward)

/** Glossy painted body: inked fill, then decal, dirt, rust, dents and a clear-coat shine clipped inside it. */
export function paintBody(g: G, c: VehCtx, path: () => void, color: string, box: number, o: { hatch?: number; wear?: number; grain?: number } = {}): void {
  ik(g, c.rng, path, color, box, { line: c.line, ...o });
  g.save(); g.beginPath(); path(); g.clip();
  DECALS[c.look.decal](g, c);
  grime(g, c);
  rust(g, c);
  dents(g, c);
  shine(g, c);
  g.restore();
  g.save(); g.beginPath(); path(); g.lineJoin = 'round'; g.strokeStyle = INK; g.lineWidth = c.line; g.stroke(); g.restore();
}

/** Clear-coat: soft sheen on the lit (-y, top-left) flank plus one crisp specular streak. Dulls as dirt builds up. */
function shine(g: G, c: VehCtx): void {
  const { halfL: L, halfW: W } = c, k = Math.max(0.15, 1 - c.look.grime * 0.6 - c.dmg * 0.12);
  const gr = g.createLinearGradient(0, -W, 0, W * 0.2);
  gr.addColorStop(0, alpha(LAMP.shine, 0.28 * k)); gr.addColorStop(1, alpha(LAMP.shine, 0));
  g.fillStyle = gr; g.fillRect(-L * 1.1, -W * 1.1, L * 2.2, W * 1.3);
  g.fillStyle = alpha(LAMP.shine, 0.2 * k);
  g.beginPath(); ell(g, L * 0.05, -W * 0.6, L * 0.55, W * 0.12); g.fill();
  stroke(g, [[-L * 0.55, -W * 0.74], [L * 0.35, -W * 0.74]], 0.024, LAMP.shine, 0.55 * k);
  stroke(g, [[L * 0.45, -W * 0.74], [L * 0.58, -W * 0.72]], 0.018, LAMP.shine, 0.45 * k);
  // shaded far flank (+y) so the body reads as rounded
  const sh = g.createLinearGradient(0, W * 0.45, 0, W);
  sh.addColorStop(0, alpha(INK, 0)); sh.addColorStop(1, alpha(INK, 0.22));
  g.fillStyle = sh; g.fillRect(-L * 1.1, W * 0.4, L * 2.2, W * 0.7);
}

/** Road dirt along both rocker panels and splatter behind each wheel. */
function grime(g: G, c: VehCtx): void {
  const { halfL: L, halfW: W } = c, a = Math.min(1, c.look.grime + c.dmg * 0.15);
  if (a <= 0.01) return;
  const r = makeRng(hashStr(c.id + ':dirt'));
  for (const s of [-1, 1]) {
    const gr = g.createLinearGradient(0, s * W, 0, s * W * 0.5);
    gr.addColorStop(0, alpha(c.look.dirt, 0.6 * a)); gr.addColorStop(1, alpha(c.look.dirt, 0));
    g.fillStyle = gr; g.fillRect(-L * 1.1, s > 0 ? W * 0.5 : -W * 1.1, L * 2.2, W * 0.6);
    for (const wx of [-L * 0.62, L * 0.62]) {
      for (let i = 0, n = Math.round(18 * a); i < n; i++) {
        g.fillStyle = alpha(c.look.dirt, r.range(0.25, 0.6));
        g.beginPath(); g.arc(wx - r.range(0.05, 0.4), s * W * r.range(0.62, 0.98), r.range(0.006, 0.02), 0, TAU); g.fill();
      }
    }
  }
}

/** Seeded rust blooms near the edges; more appear as damage rises. Positions are stable per vehicle id. */
function rust(g: G, c: VehCtx): void {
  const { halfL: L, halfW: W } = c, amt = Math.min(1, c.look.rust + c.dmg * 0.22);
  const r = makeRng(hashStr(c.id + ':rust'));
  for (let i = 0; i < 8; i++) {
    const x = r.range(-L * 0.85, L * 0.85), y = (r.chance(0.5) ? -1 : 1) * W * r.range(0.62, 0.92), rad = r.range(0.04, 0.09), seed = r.next();
    if (i >= Math.round(amt * 8)) continue;
    const pr = makeRng(hashStr(c.id + i) ^ Math.floor(seed * 1e6));
    g.fillStyle = alpha(c.look.rustTone, 0.78);
    g.beginPath(); lumpy(g, x, y, rad, rad * 0.75, pr, 9, 0.3); g.fill();
    for (let d = 0; d < 9; d++) {
      g.fillStyle = alpha(d % 3 ? shade(c.look.rustTone, -0.45) : shade(c.look.rustTone, 0.35), 0.6);
      g.beginPath(); g.arc(x + pr.range(-rad, rad), y + pr.range(-rad, rad) * 0.7, pr.range(0.004, 0.011), 0, TAU); g.fill();
    }
    g.strokeStyle = alpha(INK, 0.35); g.lineWidth = 0.008;
    g.beginPath(); lumpy(g, x, y, rad, rad * 0.75, makeRng(hashStr(c.id + i) ^ Math.floor(seed * 1e6)), 9, 0.3); g.stroke();
  }
}

/** Dents (dark crescent + light rim) and bright scratches, two more per damage level. */
function dents(g: G, c: VehCtx): void {
  if (c.dmg <= 0) return;
  const { halfL: L, halfW: W } = c, r = makeRng(hashStr(c.id + ':dent'));
  for (let i = 0; i < 6; i++) {
    const x = r.range(-L * 0.8, L * 0.8), s = r.chance(0.5) ? -1 : 1, y = s * W * r.range(0.55, 0.85), rad = r.range(0.06, 0.11), a0 = r.range(0, TAU);
    const sx = r.range(-L * 0.7, L * 0.7), sy = r.range(-W * 0.7, W * 0.7), sa = r.range(0, TAU), sl = r.range(0.08, 0.2);
    if (i >= c.dmg * 2) continue;
    g.lineCap = 'round';
    g.strokeStyle = alpha(INK, 0.45); g.lineWidth = 0.018; g.beginPath(); g.arc(x, y, rad, a0, a0 + 2.2); g.stroke();
    g.strokeStyle = alpha(LAMP.shine, 0.4); g.lineWidth = 0.012; g.beginPath(); g.arc(x + 0.015, y - 0.015, rad * 0.8, a0 + 2.6, a0 + 4.4); g.stroke();
    g.strokeStyle = alpha(LAMP.shine, 0.55); g.lineWidth = 0.007;
    g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx + Math.cos(sa) * sl, sy + Math.sin(sa) * sl); g.lineTo(sx + Math.cos(sa) * sl + 0.03, sy + Math.sin(sa) * sl + 0.01); g.stroke();
  }
}

/** Tinted glass with a sky reflection band, crisp highlight streaks and, when badly hurt, a cracked star. */
export function glass(g: G, c: VehCtx, path: () => void, cx: number, cy: number, rx: number, ry: number, box = 0.45): void {
  ik(g, c.rng, path, c.look.glass, box, { hatch: 0.3, line: c.line * 0.72 });
  g.save(); g.beginPath(); path(); g.clip();
  const gr = g.createLinearGradient(cx - rx, cy - ry, cx + rx, cy + ry);
  gr.addColorStop(0, alpha('#a5c4d4', 0.45)); gr.addColorStop(0.35, alpha('#a5c4d4', 0.08)); gr.addColorStop(0.55, alpha('#a5c4d4', 0.3)); gr.addColorStop(1, alpha('#a5c4d4', 0));
  g.fillStyle = gr; g.fillRect(cx - rx, cy - ry, rx * 2, ry * 2);
  stroke(g, [[cx - rx * 0.5, cy - ry * 0.8], [cx + rx * 0.3, cy + ry * 0.3]], Math.min(rx, ry) * 0.12, LAMP.shine, 0.7);
  stroke(g, [[cx - rx * 0.1, cy - ry * 0.85], [cx + rx * 0.55, cy - ry * 0.1]], Math.min(rx, ry) * 0.06, LAMP.shine, 0.55);
  if (c.dmg >= 3) {
    const r = makeRng(hashStr(c.id + ':crack')), px = cx + rx * 0.2, py = cy + ry * 0.25;
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * TAU + r.range(-0.2, 0.2), l = r.range(0.4, 0.9) * Math.max(rx, ry);
      stroke(g, [[px, py], [px + Math.cos(a) * l * 0.5 + r.range(-0.02, 0.02), py + Math.sin(a) * l * 0.5], [px + Math.cos(a) * l, py + Math.sin(a) * l]], 0.008, LAMP.shine, 0.75);
    }
    g.strokeStyle = alpha(LAMP.shine, 0.6); g.lineWidth = 0.006; g.beginPath(); g.arc(px, py, Math.min(rx, ry) * 0.25, 0, TAU); g.stroke();
  }
  g.restore();
}

/** Chrome part: trim colour, a bright upper edge and a dark lower edge. */
export function chrome(g: G, c: VehCtx, path: () => void, cx: number, cy: number, rx: number, ry: number, box = 0.4): void {
  ik(g, c.rng, path, c.look.trim, box, { hatch: 0.25, line: 0.016, wear: 0.2, grain: 0.2 });
  g.save(); g.beginPath(); path(); g.clip();
  const gr = g.createLinearGradient(cx - rx, cy - ry, cx + rx, cy + ry);
  gr.addColorStop(0, alpha(LAMP.shine, 0.75)); gr.addColorStop(0.4, alpha(LAMP.shine, 0.1)); gr.addColorStop(0.6, alpha(INK, 0.1)); gr.addColorStop(1, alpha(INK, 0.35));
  g.fillStyle = gr; g.fillRect(cx - rx, cy - ry, rx * 2, ry * 2);
  g.restore();
}

/** Door / hood seams: hairline cuts with a light bevel, plus small chrome handles. */
export function seams(g: G, c: VehCtx, xs: number[], inset = 0.55, handles = true): void {
  const W = c.halfW;
  for (const x of xs) for (const s of [-1, 1]) {
    stroke(g, [[x, s * W * 0.98], [x - 0.02, s * W * inset]], 0.011, INK, 0.55);
    stroke(g, [[x + 0.014, s * W * 0.96], [x - 0.006, s * W * (inset + 0.04)]], 0.006, LAMP.shine, 0.35);
    if (handles) { g.fillStyle = c.look.trim; g.beginPath(); rrect(g, x - 0.13, s * W * 0.86 - 0.012, 0.07, 0.024, 0.01); g.fill(); stroke(g, [[x - 0.13, s * W * 0.86 + 0.012], [x - 0.06, s * W * 0.86 + 0.012]], 0.005, INK, 0.5); }
  }
}

/** Tail lenses and amber corner indicators (off state; the glow is drawn per frame). */
function lenses(g: G, c: VehCtx): void {
  const lp = LAMPS[c.kind], { halfL: L, halfW: W } = c;
  for (const s of [-1, 1]) {
    const tx = lp.tail[0] * L, ty = s * lp.tail[1] * W;
    ik(g, c.rng, () => rrect(g, tx - 0.02, ty - 0.06, 0.06, 0.12, 0.02), LAMP.tail, 0.08, { hatch: 0, line: 0.012 });
    stroke(g, [[tx, ty - 0.04], [tx + 0.02, ty + 0.03]], 0.01, LAMP.shine, 0.5);
    g.fillStyle = shade(LAMP.amber, -0.15); g.strokeStyle = INK; g.lineWidth = 0.008;
    g.beginPath(); ell(g, lp.head[0] * L - 0.04, s * Math.min(W * 0.92, lp.head[1] * W + lp.headR + 0.05), 0.03, 0.022); g.fill(); g.stroke();
  }
}

/** Glossy lens over a headlight: chrome ring and a highlight dot. */
function headLens(g: G, c: VehCtx): void {
  const lp = LAMPS[c.kind];
  for (const s of [-1, 1]) {
    const x = lp.head[0] * c.halfL, y = s * lp.head[1] * c.halfW, r = lp.headR;
    g.strokeStyle = c.look.trim; g.lineWidth = r * 0.28; g.beginPath(); g.arc(x, y, r * 1.02, 0, TAU); g.stroke();
    g.strokeStyle = INK; g.lineWidth = 0.008; g.beginPath(); g.arc(x, y, r * 1.18, 0, TAU); g.stroke();
    g.fillStyle = alpha(LAMP.shine, 0.85); g.beginPath(); g.arc(x - r * 0.35, y - r * 0.35, r * 0.28, 0, TAU); g.fill();
  }
}

// ---------------------------------------------------------------- decal registry (keyed by VehicleLook.decal)
export const DECALS: Record<VehicleDecal, DecalPainter> = {
  none: () => {},
  stripes: (g, c) => {
    const L = c.halfL;
    for (const y of [-0.11, 0.11]) {
      g.fillStyle = alpha(c.look.accent, 0.92); g.fillRect(-L * 1.1, y - 0.045, L * 2.2, 0.09);
      stroke(g, [[-L * 1.1, y - 0.045], [L * 1.1, y - 0.045]], 0.008, c.look.accent2, 0.7);
      stroke(g, [[-L * 1.1, y + 0.045], [L * 1.1, y + 0.045]], 0.008, c.look.accent2, 0.7);
    }
  },
  flames: (g, c) => {
    const { halfL: L, halfW: W } = c;
    for (const s of [-1, 1]) for (const [col, k] of [[c.look.accent2, 1], [c.look.accent, 0.62]] as [string, number][]) {
      const pts: [number, number][] = [[L * 0.95, s * W * 0.95], [L * 0.95, s * W * (1 - 0.5 * k)]];
      for (let i = 0; i < 5; i++) {
        const x = L * (0.75 - i * 0.28 * k);
        pts.push([x, s * W * (1 - 0.42 * k - (i % 2) * 0.08)], [x - L * 0.12 * k, s * W * (1 - 0.15 * k)]);
      }
      pts.push([L * (0.75 - 5 * 0.28 * k), s * W * 0.95]);
      g.fillStyle = col; g.beginPath(); poly(g, pts); g.fill();
      g.strokeStyle = alpha(INK, 0.6); g.lineWidth = 0.01; g.stroke();
    }
  },
  number: (g, c) => {
    const x = c.halfL * 0.62;
    g.fillStyle = c.look.accent; g.beginPath(); g.arc(x, 0, 0.15, 0, TAU); g.fill();
    g.strokeStyle = c.look.accent2; g.lineWidth = 0.03; g.stroke();
    g.save(); g.translate(x, 0); g.rotate(Math.PI / 2);
    g.fillStyle = INK; g.font = 'bold 0.17px "Fredoka", "Arial Black", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(String(c.look.number), 0, 0.01); g.restore();
  },
  stars: (g, c) => {
    const r = makeRng(hashStr(c.id + ':stars')), { halfL: L, halfW: W } = c;
    for (let i = 0; i < 6; i++) {
      g.fillStyle = i % 2 ? c.look.accent : c.look.accent2;
      g.beginPath(); star(g, r.range(-L * 0.8, L * 0.6), (i % 2 ? -1 : 1) * W * r.range(0.55, 0.85), 0.035, 0.075, 5, r.range(0, TAU)); g.fill();
      g.strokeStyle = alpha(INK, 0.55); g.lineWidth = 0.009; g.stroke();
    }
  },
  pinstripe: (g, c) => {
    const { halfL: L, halfW: W } = c;
    for (const s of [-1, 1]) for (const [col, off] of [[c.look.accent, 0.8], [c.look.accent2, 0.7]] as [string, number][]) {
      const pts: [number, number][] = [];
      for (let x = -L; x <= L; x += 0.08) pts.push([x, s * W * (off + Math.sin(x * 7) * 0.04)]);
      stroke(g, pts, 0.03, col, 0.9);
    }
  },
  bolt: (g, c) => {
    const { halfL: L, halfW: W } = c;
    for (const s of [-1, 1]) {
      const pts: [number, number][] = [[L * 0.8, s * W * 0.72], [L * 0.3, s * W * 0.62], [L * 0.15, s * W * 0.8], [-L * 0.35, s * W * 0.66], [-L * 0.48, s * W * 0.82], [-L * 0.9, s * W * 0.7]];
      stroke(g, pts, 0.05, c.look.accent, 0.9);
      stroke(g, pts, 0.016, c.look.accent2, 0.9);
    }
  },
  ducks: (g, c) => {
    const { halfL: L, halfW: W } = c;
    for (const [x, s] of [[-L * 0.45, -1], [L * 0.05, 1], [-L * 0.1, -1]] as [number, number][]) {
      const y = s * W * 0.74;
      g.fillStyle = c.look.accent; g.strokeStyle = alpha(INK, 0.7); g.lineWidth = 0.009;
      g.beginPath(); ell(g, x, y, 0.07, 0.045); g.fill(); g.stroke();
      g.beginPath(); ell(g, x + 0.06, y - 0.01, 0.033, 0.033); g.fill(); g.stroke();
      g.fillStyle = c.look.accent2; g.beginPath(); poly(g, [[x + 0.088, y - 0.018], [x + 0.12, y - 0.008], [x + 0.088, y + 0.002]]); g.fill();
    }
  },
};

// ---------------------------------------------------------------- body registry (keyed by VehicleDef.kind; drawn once into the sprite cache)
export const BODIES: Record<VehicleKind, BodyDrawer> = {
  hatchback: (g, c) => {
    const { rng, halfL, halfW } = c;
    const body = (): void => smooth(g, [
      [halfL * 0.92, -halfW * 0.85], [halfL, -halfW * 0.45], [halfL, halfW * 0.45], [halfL * 0.92, halfW * 0.85],
      [halfL * 0.8, halfW], [-halfL * 0.84, halfW], [-halfL * 0.98, halfW * 0.8], [-halfL, halfW * 0.4],
      [-halfL, -halfW * 0.4], [-halfL * 0.98, -halfW * 0.8], [-halfL * 0.84, -halfW], [halfL * 0.8, -halfW],
    ]);
    paintBody(g, c, body, c.color, 1.2, { hatch: 0.85, wear: 0.65, grain: 0.6 });
    // hood crease and door seams
    stroke(g, [[halfL * 0.42, -halfW * 0.5], [halfL * 0.86, -halfW * 0.38]], 0.01, INK, 0.45);
    stroke(g, [[halfL * 0.42, halfW * 0.5], [halfL * 0.86, halfW * 0.38]], 0.01, INK, 0.45);
    seams(g, c, [halfL * 0.3, -halfL * 0.3]);
    // bumpers
    chrome(g, c, () => rrect(g, halfL * 0.98, -halfW * 0.88, 0.08, halfW * 1.76, 0.03), halfL, 0, 0.06, halfW * 0.9, 0.5);
    chrome(g, c, () => rrect(g, -halfL - 0.05, -halfW * 0.88, 0.08, halfW * 1.76, 0.03), -halfL, 0, 0.06, halfW * 0.9, 0.5);
    // windshields
    glass(g, c, () => poly(g, [[0.36, -0.38], [0.36, 0.38], [0.12, 0.42], [0.12, -0.42]]), 0.24, 0, 0.14, 0.42);
    glass(g, c, () => poly(g, [[-0.65, -0.38], [-0.65, 0.38], [-0.86, 0.34], [-0.86, -0.34]]), -0.75, 0, 0.12, 0.38, 0.4);
    stroke(g, [[0.37, -0.36], [0.37, 0.36]], 0.012, c.look.trim, 0.9);
    // wing mirrors
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, 0.32, s * (halfW + 0.04), 0.05, 0.035), shade(c.color, -0.1), 0.08, { hatch: 0, line: 0.012 });
    // roof rack & wooden crate
    stroke(g, [[-0.54, -0.34], [0.08, -0.34]], 0.016, '#4a4f55');
    stroke(g, [[-0.54, 0.34], [0.08, 0.34]], 0.016, '#4a4f55');
    for (const x of [-0.5, 0.04]) stroke(g, [[x, -0.34], [x, 0.34]], 0.012, '#4a4f55', 0.8);
    ik(g, rng, () => rrect(g, -0.44, -0.24, 0.42, 0.48, 0.03), '#6a5645', 0.42, { hatch: 0.7, line: 0.016 });
    for (const y of [-0.08, 0.08]) stroke(g, [[-0.44, y], [-0.02, y]], 0.008, INK, 0.45);
    stroke(g, [[-0.44, -0.24], [-0.02, 0.24]], 0.014, '#c9b06a', 0.85);
    // headlights
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.94, s * halfW * 0.65, 0.065, 0.065), c.lights ? LAMP.head : LAMP.headOff, 0.1);
  },

  clown_car: (g, c) => {
    const { rng, halfL, halfW } = c;
    const body = (): void => smooth(g, [
      [halfL * 0.85, -halfW * 0.7], [halfL, 0], [halfL * 0.85, halfW * 0.7],
      [halfL * 0.4, halfW * 1.05], [-halfL * 0.7, halfW * 1.02], [-halfL * 0.95, halfW * 0.5],
      [-halfL * 0.95, -halfW * 0.5], [-halfL * 0.7, -halfW * 1.02], [halfL * 0.4, -halfW * 1.05],
    ]);
    paintBody(g, c, body, c.color, 1.1, { hatch: 0.5, wear: 0.2 });
    const dots: [number, number, string][] = [
      [0.35, -0.22, '#facc15'], [0.38, 0.24, '#38bdf8'], [-0.35, -0.32, '#facc15'],
      [-0.32, 0.32, '#a855f7'], [-0.55, 0.0, '#38bdf8'], [0.1, 0.38, '#facc15'],
    ];
    for (const [dx, dy, col] of dots) {
      ik(g, rng, () => ell(g, dx, dy, 0.08, 0.08), col, 0.15, { hatch: 0.2, line: 0.012 });
      g.fillStyle = alpha(LAMP.shine, 0.6); g.beginPath(); g.arc(dx - 0.025, dy - 0.025, 0.02, 0, TAU); g.fill();
    }
    // bug-eye headlights
    for (const s of [-1, 1]) {
      ik(g, rng, () => ell(g, halfL * 0.88, s * halfW * 0.55, 0.11, 0.11), c.lights ? '#fef08a' : '#f8fafc', 0.15, { line: 0.016 });
      ik(g, rng, () => ell(g, halfL * 0.91, s * halfW * 0.55, 0.04, 0.04), '#0f172a', 0.05);
    }
    // brass horn
    stroke(g, [[0.3, -halfW * 0.95], [0.45, -halfW * 1.05]], 0.02, '#ca8a04');
    ik(g, rng, () => ell(g, 0.50, -halfW * 1.10, 0.06, 0.09), '#eab308', 0.1, { hatch: 0.4 });
    g.fillStyle = alpha(LAMP.shine, 0.7); g.beginPath(); g.arc(0.48, -halfW * 1.14, 0.018, 0, TAU); g.fill();
    // bubble windshield
    glass(g, c, () => poly(g, [[0.28, -0.32], [0.28, 0.32], [0.08, 0.36], [0.08, -0.36]]), 0.18, 0, 0.12, 0.36, 0.35);
    chrome(g, c, () => rrect(g, -halfL * 0.98, -halfW * 0.5, 0.06, halfW, 0.03), -halfL * 0.95, 0, 0.05, halfW * 0.5, 0.3);
  },

  ice_cream_truck: (g, c) => {
    const { rng, halfL, halfW } = c;
    paintBody(g, c, () => rrect(g, -halfL * 0.96, -halfW * 0.92, halfL * 1.92, halfW * 1.84, 0.12), c.color, 1.4, { hatch: 0.5, wear: 0.3 });
    stroke(g, [[0.22, -halfW * 0.92], [0.22, halfW * 0.92]], 0.018, INK, 0.6);
    seams(g, c, [0.62], 0.6);
    // roof vents (rivet rows)
    for (let x = -halfL * 0.85; x <= 0.1; x += 0.16) for (const s of [-1, 1]) {
      g.fillStyle = alpha(INK, 0.35); g.beginPath(); g.arc(x, s * halfW * 0.82, 0.012, 0, TAU); g.fill();
    }
    glass(g, c, () => poly(g, [[halfL * 0.75, -halfW * 0.78], [halfL * 0.75, halfW * 0.78], [0.30, halfW * 0.82], [0.30, -halfW * 0.82]]), (halfL * 0.75 + 0.3) / 2, 0, (halfL * 0.75 - 0.3) / 2, halfW * 0.8, 0.5);
    // service window awning
    const awX = -halfL * 0.35, awY = halfW * 0.92;
    ik(g, rng, () => rrect(g, awX - 0.35, awY - 0.04, 0.70, 0.16, 0.02), '#f472b6', 0.35, { hatch: 0.6, line: 0.014 });
    for (let i = -3; i <= 3; i += 2) stroke(g, [[awX + i * 0.1, awY - 0.04], [awX + i * 0.1, awY + 0.12]], 0.018, '#ffffff');
    // soft-serve cone
    ik(g, rng, () => poly(g, [[-0.45, 0], [-0.05, -0.18], [-0.05, 0.18]]), '#d97706', 0.3, { hatch: 0.8, line: 0.016 });
    for (const k of [0.25, 0.55]) stroke(g, [[-0.45 + 0.4 * k, -0.18 * k], [-0.05, 0.18 * (1 - k)]], 0.006, INK, 0.45);
    ik(g, rng, () => ell(g, 0.02, 0, 0.22, 0.24), '#fdf2f8', 0.3, { hatch: 0.4, line: 0.016 });
    ik(g, rng, () => ell(g, -0.04, -0.06, 0.14, 0.14), '#fbcfe8', 0.2, { hatch: 0.3 });
    ik(g, rng, () => ell(g, 0.16, 0, 0.055, 0.055), '#dc2626', 0.08, { line: 0.012 });
    g.fillStyle = alpha(LAMP.shine, 0.8); g.beginPath(); g.arc(0.145, -0.018, 0.016, 0, TAU); g.fill();
    const sprinkles: [number, number, string][] = [
      [-0.6, -0.3, '#ec4899'], [-0.7, 0.25, '#06b6d4'], [-0.5, 0.35, '#eab308'], [-0.8, -0.15, '#10b981'], [-0.65, 0.0, '#8b5cf6'],
    ];
    for (const [sx, sy, scol] of sprinkles) stroke(g, [[sx - 0.03, sy - 0.02], [sx + 0.03, sy + 0.02]], 0.02, scol);
    chrome(g, c, () => rrect(g, halfL * 0.96, -halfW * 0.88, 0.1, halfW * 1.76, 0.03), halfL, 0, 0.06, halfW * 0.9, 0.4);
    chrome(g, c, () => rrect(g, -halfL * 1.0, -halfW * 0.85, 0.07, halfW * 1.7, 0.03), -halfL, 0, 0.05, halfW * 0.9, 0.4);
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.94, s * halfW * 0.65, 0.08, 0.08), c.lights ? '#fef08a' : '#cbd5e1', 0.1);
  },

  golf_cart: (g, c) => {
    const { rng, halfL, halfW } = c;
    paintBody(g, c, () => smooth(g, [
      [halfL * 0.88, -halfW * 0.75], [halfL * 0.95, 0], [halfL * 0.88, halfW * 0.75],
      [halfL * 0.6, halfW * 0.9], [-halfL * 0.85, halfW * 0.9], [-halfL * 0.95, halfW * 0.6],
      [-halfL * 0.95, -halfW * 0.6], [-halfL * 0.85, -halfW * 0.9], [halfL * 0.6, -halfW * 0.9],
    ]), c.color, 0.9, { hatch: 0.6, wear: 0.3 });
    ik(g, rng, () => rrect(g, -0.25, -halfW * 0.75, 0.35, halfW * 1.5, 0.05), '#d4b483', 0.35, { hatch: 0.5, line: 0.016 });
    stroke(g, [[-0.25, 0], [0.1, 0]], 0.016, INK, 0.4);
    for (let x = -0.2; x < 0.1; x += 0.08) stroke(g, [[x, -halfW * 0.7], [x, halfW * 0.7]], 0.006, INK, 0.25);
    const sx1 = -halfL * 0.55, sx2 = halfL * 0.45, sy = halfW * 0.78;
    for (const [x, y] of [[sx1, -sy], [sx1, sy], [sx2, -sy], [sx2, sy]] as [number, number][]) ik(g, rng, () => ell(g, x, y, 0.04, 0.04), '#0f172a', 0.05);
    // striped canopy with fabric sag and stitching
    ik(g, rng, () => rrect(g, -halfL * 0.62, -halfW * 0.82, halfL * 1.15, halfW * 1.64, 0.08), '#f5f5f4', 0.6, { hatch: 0.3, line: c.line * 0.8 });
    for (let y = -halfW * 0.7; y <= halfW * 0.7; y += 0.22) stroke(g, [[-halfL * 0.62, y], [halfL * 0.53, y]], 0.04, '#15803d');
    g.setLineDash([0.03, 0.025]); stroke(g, [[-halfL * 0.58, -halfW * 0.76], [halfL * 0.49, -halfW * 0.76], [halfL * 0.49, halfW * 0.76], [-halfL * 0.58, halfW * 0.76], [-halfL * 0.58, -halfW * 0.76]], 0.006, INK, 0.45); g.setLineDash([]);
    stroke(g, [[-halfL * 0.5, -halfW * 0.6], [halfL * 0.3, -halfW * 0.6]], 0.03, LAMP.shine, 0.3);
    DECALS[c.look.decal](g, c);
    glass(g, c, () => poly(g, [[halfL * 0.45, -halfW * 0.72], [halfL * 0.45, halfW * 0.72], [halfL * 0.42, halfW * 0.72], [halfL * 0.42, -halfW * 0.72]]), halfL * 0.435, 0, 0.02, halfW * 0.72, 0.3);
    const bagX = -halfL * 0.72;
    ik(g, rng, () => ell(g, bagX, 0, 0.16, 0.32), '#78350f', 0.25, { hatch: 0.6, line: 0.016 });
    stroke(g, [[bagX, -0.08], [bagX - 0.22, -0.24]], 0.014, '#94a3b8');
    ik(g, rng, () => ell(g, bagX - 0.24, -0.26, 0.05, 0.03), '#e2e8f0', 0.04);
    stroke(g, [[bagX, 0.05], [bagX - 0.25, 0.16]], 0.014, '#94a3b8');
    ik(g, rng, () => ell(g, bagX - 0.27, 0.18, 0.04, 0.03), '#e2e8f0', 0.04);
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.85, s * halfW * 0.45, 0.06, 0.06), c.lights ? '#fef08a' : '#cbd5e1', 0.08);
  },

  bathtub_buggy: (g, c) => {
    const { rng, halfL, halfW } = c;
    paintBody(g, c, () => smooth(g, [
      [halfL * 0.78, -halfW * 0.72], [halfL * 0.90, -halfW * 0.35], [halfL * 0.90, halfW * 0.35], [halfL * 0.78, halfW * 0.72],
      [halfL * 0.4, halfW * 0.88], [-halfL * 0.7, halfW * 0.88], [-halfL * 0.92, halfW * 0.65],
      [-halfL * 0.92, -halfW * 0.65], [-halfL * 0.7, -halfW * 0.88], [halfL * 0.4, -halfW * 0.88],
    ]), c.color, 1.1, { hatch: 0.35, wear: 0.2 });
    // rolled rim and inner basin with water
    ik(g, rng, () => ell(g, -0.05, 0, halfL * 0.65, halfW * 0.58), '#cbd5e1', 0.7, { hatch: 0.2, line: 0.016 });
    g.fillStyle = alpha('#7dd3fc', 0.35); g.beginPath(); ell(g, -0.1, 0.02, halfL * 0.52, halfW * 0.44); g.fill();
    stroke(g, [[-halfL * 0.4, -halfW * 0.25], [-halfL * 0.1, -halfW * 0.32]], 0.012, LAMP.shine, 0.6);
    ik(g, rng, () => ell(g, -halfL * 0.4, 0, 0.06, 0.06), '#64748b', 0.05);
    for (const [fx, fy] of [[halfL * 0.65, -halfW * 0.85], [halfL * 0.65, halfW * 0.85], [-halfL * 0.72, -halfW * 0.85], [-halfL * 0.72, halfW * 0.85]] as [number, number][]) {
      ik(g, rng, () => ell(g, fx, fy, 0.08, 0.08), '#eab308', 0.1, { hatch: 0.6, line: 0.014 });
      for (const k of [-1, 0, 1]) stroke(g, [[fx + k * 0.03, fy], [fx + k * 0.04, fy + Math.sign(fy) * 0.07]], 0.008, INK, 0.6);
    }
    // chrome V8 with blower and rocker covers
    const engX = halfL * 0.55;
    chrome(g, c, () => rrect(g, engX - 0.2, -0.24, 0.40, 0.48, 0.04), engX, 0, 0.2, 0.24, 0.35);
    for (const s of [-1, 1]) for (let k = 0; k < 4; k++) { g.fillStyle = INK; g.beginPath(); g.arc(engX - 0.14 + k * 0.09, s * 0.19, 0.014, 0, TAU); g.fill(); }
    ik(g, rng, () => poly(g, [[engX + 0.15, -0.16], [engX + 0.15, 0.16], [engX - 0.1, 0.12], [engX - 0.1, -0.12]]), '#dc2626', 0.2, { line: 0.014 });
    for (let k = -2; k <= 2; k++) stroke(g, [[engX - 0.08, k * 0.045], [engX + 0.13, k * 0.055]], 0.006, INK, 0.5);
    stroke(g, [[engX - 0.05, -0.24], [engX - 0.15, -halfW * 1.05]], 0.024, c.look.trim);
    stroke(g, [[engX - 0.05, 0.24], [engX - 0.15, halfW * 1.05]], 0.024, c.look.trim);
    const shX = -halfL * 0.75;
    stroke(g, [[shX, 0], [shX - 0.1, 0]], 0.03, '#94a3b8');
    ik(g, rng, () => ell(g, shX - 0.16, 0, 0.14, 0.14), '#ca8a04', 0.15, { hatch: 0.5, line: 0.016 });
    for (let i = 0; i < 8; i++) { const a = (i / 8) * TAU; g.fillStyle = INK; g.beginPath(); g.arc(shX - 0.16 + Math.cos(a) * 0.08, Math.sin(a) * 0.08, 0.01, 0, TAU); g.fill(); }
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.85, s * halfW * 0.5, 0.07, 0.07), c.lights ? '#fef08a' : '#eab308', 0.09);
  },

  hot_dog_van: (g, c) => {
    const { rng, halfL, halfW } = c;
    paintBody(g, c, () => rrect(g, -halfL * 0.94, -halfW * 0.90, halfL * 1.88, halfW * 1.80, 0.14), c.color, 1.3, { hatch: 0.6, wear: 0.4 });
    seams(g, c, [0.6, -halfL * 0.2], 0.6);
    glass(g, c, () => poly(g, [[halfL * 0.72, -halfW * 0.74], [halfL * 0.72, halfW * 0.74], [0.32, halfW * 0.78], [0.32, -halfW * 0.78]]), (halfL * 0.72 + 0.32) / 2, 0, (halfL * 0.72 - 0.32) / 2, halfW * 0.76, 0.45);
    // giant hot dog with sesame seeds
    for (const s of [-1, 1]) {
      ik(g, rng, () => ell(g, -0.15, s * 0.22, halfL * 0.65, 0.16), '#b45309', 0.45, { hatch: 0.7, line: 0.016 });
      const r = makeRng(hashStr(c.id + ':seed' + s));
      for (let i = 0; i < 9; i++) { g.fillStyle = '#f5e6c4'; g.beginPath(); ell(g, r.range(-halfL * 0.7, halfL * 0.4), s * (0.22 + r.range(-0.08, 0.08)), 0.014, 0.008, r.range(0, 3)); g.fill(); }
    }
    ik(g, rng, () => cap(g, -halfL * 0.78, 0, halfL * 0.48, 0, 0.15, 0.15), '#991b1b', 0.5, { hatch: 0.45, line: 0.022 });
    stroke(g, [[-halfL * 0.6, -0.08], [halfL * 0.3, -0.08]], 0.025, LAMP.shine, 0.35);
    const mustard: [number, number][] = [];
    for (let x = -halfL * 0.70; x <= halfL * 0.40; x += 0.12) mustard.push([x, ((Math.round((x + halfL) / 0.12) % 2 === 0) ? -1 : 1) * 0.07]);
    stroke(g, mustard, 0.05, INK, 0.9);
    stroke(g, mustard, 0.034, '#facc15');
    // toothy chrome grille
    chrome(g, c, () => rrect(g, halfL * 0.94, -halfW * 0.65, 0.08, halfW * 1.3, 0.03), halfL * 0.98, 0, 0.05, halfW * 0.65, 0.4);
    for (let gy = -halfW * 0.45; gy <= halfW * 0.45; gy += 0.15) stroke(g, [[halfL * 0.94, gy], [halfL * 1.02, gy]], 0.012, INK);
    chrome(g, c, () => rrect(g, -halfL * 0.98, -halfW * 0.85, 0.07, halfW * 1.7, 0.03), -halfL, 0, 0.05, halfW * 0.85, 0.4);
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.92, s * halfW * 0.72, 0.08, 0.08), c.lights ? '#fef08a' : '#e2e8f0', 0.1);
  },

  tesla_roadster: (g, c) => {
    const { rng, halfL, halfW } = c;
    paintBody(g, c, () => smooth(g, [
      [halfL * 0.98, -halfW * 0.65], [halfL * 1.02, 0], [halfL * 0.98, halfW * 0.65],
      [halfL * 0.75, halfW * 0.92], [halfL * 0.25, halfW * 0.85], [-halfL * 0.15, halfW * 0.82],
      [-halfL * 0.70, halfW * 0.98], [-halfL * 0.98, halfW * 0.75], [-halfL * 0.95, 0],
      [-halfL * 0.98, -halfW * 0.75], [-halfL * 0.70, -halfW * 0.98], [-halfL * 0.15, -halfW * 0.82],
      [halfL * 0.25, -halfW * 0.85], [halfL * 0.75, -halfW * 0.92],
    ]), c.color, 1.25, { hatch: 0.45, wear: 0.1, grain: 0.3 });
    ik(g, rng, () => rrect(g, -halfL * 1.02, -halfW * 0.78, 0.08, halfW * 1.56, 0.02), '#0f172a', 0.4, { hatch: 0.8, line: 0.018 });
    stroke(g, [[-halfL * 0.98, -halfW * 0.72], [-halfL * 0.98, halfW * 0.72]], 0.028, '#0f172a');
    for (let y = -halfW * 0.6; y <= halfW * 0.6; y += 0.06) stroke(g, [[-halfL * 1.01, y], [-halfL * 0.95, y + 0.03]], 0.004, '#475569', 0.8);
    glass(g, c, () => smooth(g, [
      [halfL * 0.38, -halfW * 0.62], [halfL * 0.42, 0], [halfL * 0.38, halfW * 0.62],
      [-halfL * 0.45, halfW * 0.65], [-halfL * 0.55, 0], [-halfL * 0.45, -halfW * 0.62],
    ]), -halfL * 0.05, 0, halfL * 0.48, halfW * 0.62, 0.65);
    stroke(g, [[-halfL * 0.25, -halfW * 0.45], [halfL * 0.28, 0.15]], 0.02, '#38bdf8', 0.6);
    stroke(g, [[0.25, -halfW * 0.4], [halfL * 0.85, -halfW * 0.28]], 0.016, INK, 0.6);
    stroke(g, [[0.25, halfW * 0.4], [halfL * 0.85, halfW * 0.28]], 0.016, INK, 0.6);
    stroke(g, [[halfL * 0.94, -halfW * 0.6], [halfL * 1.01, 0], [halfL * 0.94, halfW * 0.6]], 0.03, c.lights ? '#fef08a' : '#38bdf8');
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, halfL * 0.92, s * halfW * 0.55, 0.07, 0.04), c.lights ? '#ffffff' : '#7dd3fc', 0.08);
    stroke(g, [[-halfL * 0.96, -halfW * 0.68], [-halfL * 0.94, 0], [-halfL * 0.96, halfW * 0.68]], 0.026, '#f43f5e');
    const starX = -0.06, starY = halfW * 0.32;
    ik(g, rng, () => ell(g, starX - 0.14, starY, 0.13, 0.18), '#f1f5f9', 0.18, { line: 0.014 });
    ik(g, rng, () => ell(g, starX, starY, 0.12, 0.12), '#f8fafc', 0.14, { line: 0.016 });
    ik(g, rng, () => ell(g, starX + 0.04, starY, 0.06, 0.075), '#f59e0b', 0.08, { line: 0.012 });
    g.fillStyle = alpha(LAMP.shine, 0.8); g.beginPath(); g.arc(starX + 0.03, starY - 0.03, 0.015, 0, TAU); g.fill();
    stroke(g, [[0.18, -halfW * 0.28], [0.18, -halfW * 0.14]], 0.024, '#0f172a');
    stroke(g, [[0.22, -halfW * 0.21], [0.18, -halfW * 0.21]], 0.024, '#0f172a');
  },
};

/** Per-frame animated bits that cannot live in the cached sprite. Drawn above the body. */
const ANIMS: Partial<Record<VehicleKind, Anim>> = {
  clown_car: (g, v, _L, _W, time) => {
    const sw = Math.sin(time * 8) * 0.06 * (1 + Math.abs(v.speed) * 0.5);
    stroke(g, [[-0.1, 0], [-0.15, 0.02 + sw], [-0.12, -0.02 + sw * 1.5], [-0.2, sw * 2]], 0.016, '#64748b');
    g.fillStyle = '#ef4444'; g.strokeStyle = INK; g.lineWidth = 0.014;
    g.beginPath(); g.arc(-0.22, sw * 2, 0.075, 0, TAU); g.fill(); g.stroke();
    g.fillStyle = alpha(LAMP.shine, 0.7); g.beginPath(); g.arc(-0.245, sw * 2 - 0.025, 0.02, 0, TAU); g.fill();
  },
  bathtub_buggy: (g, v, L, _W, time) => {
    if (Math.abs(v.speed) <= 0.15) return;
    const shX = -L * 0.75;
    for (let i = 0; i < 3; i++) {
      const b = (time * 6 + i * 1.5) % 2.0;
      g.fillStyle = alpha('#7dd3fc', 0.4 / (1 + b));
      g.beginPath(); g.arc(shX - 0.2 - b * 0.22, Math.sin(time * 7 + i) * 0.08, 0.025 + b * 0.018, 0, TAU); g.fill();
    }
  },
};

/** Drawn under the body every frame (glows that should pulse). */
const UNDER: Partial<Record<VehicleKind, Anim>> = {
  tesla_roadster: (g, _v, L, W, time) => {
    g.fillStyle = alpha('#38bdf8', 0.28 + Math.sin(time * 3) * 0.08);
    g.beginPath(); rrect(g, -L * 0.95, -W * 1.08, L * 1.9, W * 2.16, 0.25); g.fill();
  },
};

interface LampState { s: number; t: number; brakeUntil: number }

export class VehicleRenderer {
  private cache = new SpriteCache();
  private lamps = new WeakMap<VehicleC, LampState>();

  draw(g: G, v: VehicleC, t: Transform, ppu: number, time: number): void {
    const kind = v.def.kind;
    const [len, wid] = v.def.size;
    const halfL = len / 2, halfW = wid / 2;
    const dmg = damageLevel(v.hp, v.def.maxHp), hero = v.driver !== null, look = vehicleLook(v.def.id);
    const brake = this.braking(v, time);

    // 1. ground shadow, offset down-right in world space (light from top-left)
    g.save();
    g.translate(t.x + 0.1, t.y + 0.12); g.rotate(t.angle);
    g.fillStyle = 'rgba(8,9,12,0.28)';
    g.beginPath(); rrect(g, -halfL, -halfW, len, wid + 0.04, 0.32); g.fill();
    g.restore();

    g.save();
    g.translate(t.x, t.y);
    g.rotate(t.angle);
    UNDER[kind]?.(g, v, halfL, halfW, time);

    // 2. wheels (cached sprites, front pair steers)
    const rearX = -halfL * 0.62, frontX = halfL * 0.62, trackY = halfW * 0.88;
    const [rw, rr, fw, fr, ry, fy] =
      kind === 'bathtub_buggy' ? [0.12, 0.22, 0.065, 0.16, 1.05, 0.9] :
      kind === 'ice_cream_truck' ? [0.10, 0.21, 0.085, 0.20, 0.98, 0.92] : [0.08, 0.19, 0.08, 0.19, 1, 1];
    const rearW = this.wheel(look, rw, rr, ppu), frontW = this.wheel(look, fw, fr, ppu);
    for (const s of [-1, 1]) {
      blit(g, rearW, rearX, s * trackY * ry);
      blit(g, frontW, frontX, s * trackY * fy, v.steer);
    }

    // 3. exhaust, tyre dust
    this.exhaust(g, v, halfL, halfW, time);
    if (Math.abs(v.speed) > 3 && kind !== 'tesla_roadster') {
      for (const s of [-1, 1]) for (let i = 0; i < 2; i++) {
        const p = (time * 5 + i * 0.9 + (s > 0 ? 0.4 : 0)) % 1.6;
        g.fillStyle = alpha(look.dirt, 0.22 / (1 + p * 2));
        g.beginPath(); g.arc(rearX - 0.2 - p * 0.45, s * (trackY + 0.05 + p * 0.12), 0.05 + p * 0.07, 0, TAU); g.fill();
      }
    }

    // 4. cached body
    const pad = 0.4;
    const key = `veh:${v.def.id}:${kind}:d${dmg}:${v.headlights ? 1 : 0}:${hero ? 1 : 0}`;
    const body = this.cache.get(key, ppu, -halfL - pad, -halfW - pad, halfL + pad, halfW + pad, (sg) => {
      const c: VehCtx = { id: v.def.id, kind, rng: makeRng(hashStr(v.def.id)), color: v.def.color, halfL, halfW, look, dmg, hero, lights: v.headlights, line: hero ? 0.036 : 0.024 };
      (BODIES[kind] ?? BODIES.hatchback)(sg, c);
      headLens(sg, c);
      lenses(sg, c);
    });
    blit(g, body);
    ANIMS[kind]?.(g, v, halfL, halfW, time);

    // 5. driver
    if (v.driver !== null) {
      const dx = kind === 'bathtub_buggy' ? -0.05 : kind === 'golf_cart' ? 0.0 : 0.04;
      const dy = kind === 'bathtub_buggy' ? 0 : -0.20;
      blit(g, this.driver(ppu), dx, dy);
    }

    // 6. lamp glows: tail / brake / reverse / turn signals / hazards
    this.lampGlow(g, v, halfL, halfW, time, brake, dmg);

    // 7. projected headlight beams
    if (v.headlights) {
      const beamY = halfW * 0.65;
      for (const s of [-1, 1]) {
        const beam = g.createRadialGradient(halfL, s * beamY, 0.05, halfL + 2.5, s * beamY, 4.0);
        beam.addColorStop(0, 'rgba(253, 235, 144, 0.42)');
        beam.addColorStop(0.4, 'rgba(253, 235, 144, 0.18)');
        beam.addColorStop(1, 'rgba(253, 235, 144, 0)');
        g.fillStyle = beam;
        g.beginPath(); poly(g, [[halfL, s * beamY], [halfL + 3.6, s * (beamY + 1.2)], [halfL + 3.8, -s * 0.1]]); g.fill();
      }
    }
    g.restore();
  }

  /** Brake lights come on while the car is losing speed fast, and linger briefly so they read. */
  private braking(v: VehicleC, time: number): boolean {
    const st = this.lamps.get(v), sp = Math.abs(v.speed);
    if (!st) { this.lamps.set(v, { s: sp, t: time, brakeUntil: -1 }); return false; }
    const dt = time - st.t;
    if (dt > 0) {
      if (dt < 0.5 && (st.s - sp) / dt > 2.2 && sp > 0.2) st.brakeUntil = time + 0.25;
      st.s = sp; st.t = time;
    }
    return time < st.brakeUntil;
  }

  private lampGlow(g: G, v: VehicleC, halfL: number, halfW: number, time: number, brake: boolean, dmg: number): void {
    const lp = LAMPS[v.def.kind];
    const wrecked = v.hp <= 0 || (dmg >= 3 && v.driver === null);
    const blink = Math.floor(time * 3) % 2 === 0;
    const turn = v.driver !== null && Math.abs(v.speed) > 0.4 && Math.abs(v.steer) > 0.22 ? Math.sign(v.steer) : 0;
    const tail = brake ? 0.9 : v.headlights ? 0.4 : 0;
    g.save();
    g.globalCompositeOperation = 'lighter';
    const glow = (x: number, y: number, r: number, col: string, a: number): void => {
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, alpha(col, a)); gr.addColorStop(0.35, alpha(col, a * 0.45)); gr.addColorStop(1, alpha(col, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill();
    };
    for (const s of [-1, 1]) {
      const tx = lp.tail[0] * halfL, ty = s * lp.tail[1] * halfW;
      if (tail > 0) glow(tx - 0.02, ty, brake ? 0.32 : 0.2, LAMP.tailGlow, tail);
      if (v.speed < -0.2) glow(tx, ty - s * 0.1, 0.18, LAMP.reverse, 0.7);
      if (v.headlights) glow(lp.head[0] * halfL + 0.02, s * lp.head[1] * halfW, lp.headR * 2.6, LAMP.head, 0.7);
      if (blink && (wrecked || turn === s)) {
        glow(lp.head[0] * halfL - 0.04, s * Math.min(halfW * 0.92, lp.head[1] * halfW + lp.headR + 0.05), 0.2, LAMP.amber, 0.9);
        glow(tx, ty + s * 0.04, 0.2, LAMP.amber, 0.9);
      }
    }
    g.restore();
  }

  private exhaust(g: G, v: VehicleC, halfL: number, halfW: number, time: number): void {
    const kind = v.def.kind, moving = Math.abs(v.speed) > 0.15;
    if (kind === 'golf_cart') return;
    if (kind === 'tesla_roadster') {
      if (!moving) return;
      for (let i = 0; i < 4; i++) {
        const p = (time * 12 + i * 1.7) % 2.0;
        const px = -halfL - 0.05 - p * 0.35, py = (i % 2 === 0 ? -1 : 1) * (halfW * 0.68) + Math.sin(i * 12.9) * 0.04;
        stroke(g, [[px + 0.08, py], [px + 0.04, py + (i % 2 === 0 ? 0.04 : -0.04)], [px, py]], 0.016, i % 2 === 0 ? '#38bdf8' : '#c084fc');
        g.fillStyle = alpha('#67e8f9', 0.5 / (1 + p));
        g.beginPath(); g.arc(px, py, 0.025, 0, TAU); g.fill();
      }
      return;
    }
    const ex = -halfL - 0.06, ey = -halfW * 0.65;
    stroke(g, [[ex + 0.12, ey], [ex, ey]], 0.05, INK);
    stroke(g, [[ex + 0.12, ey], [ex, ey]], 0.034, '#5f666d');
    stroke(g, [[ex + 0.1, ey - 0.008], [ex + 0.01, ey - 0.008]], 0.008, LAMP.shine, 0.5);
    // idle puffs are small and slow; driving puffs stream out behind
    const n = moving ? 3 : v.driver !== null ? 2 : 0;
    for (let i = 0; i < n; i++) {
      const p = ((time * (moving ? 7 : 2.2) + i * 1.3) % 2.5);
      const px = ex - p * (moving ? 0.28 : 0.08), py = ey + Math.sin(time * 5 + i) * 0.05 - (moving ? 0 : p * 0.06);
      const pr = (moving ? 0.035 : 0.025) + p * (moving ? 0.038 : 0.02);
      const smoky = v.hp / v.def.maxHp < 0.25;
      g.fillStyle = alpha(kind === 'clown_car' ? (i % 2 === 0 ? '#ffea00' : '#ff007f') : smoky ? '#4b4f57' : '#d8d0b8', (smoky ? 0.45 : 0.3) / (1 + p * 1.2));
      g.beginPath(); g.arc(px, py, pr, 0, TAU); g.fill();
      g.strokeStyle = alpha(INK, 0.2 / (1 + p * 1.2)); g.lineWidth = 0.008; g.stroke();
    }
  }

  /** Tyre with tread blocks, sidewall and a chrome hub with lug nuts; top-down so the tyre reads as a rounded slab. */
  private wheel(look: Required<VehicleLook>, tireW: number, tireR: number, ppu: number) {
    const pad = 0.04;
    return this.cache.get(`veh-wheel:${tireW}:${tireR}:${look.trim}`, ppu, -tireR - pad, -tireW - pad, tireR + pad, tireW + pad, (g) => {
      const rng = makeRng(hashStr('wheel' + tireR));
      ik(g, rng, () => cap(g, -tireR + tireW, 0, tireR - tireW, 0, tireW, tireW), '#1e2126', 0.24, { hatch: 0.6, line: 0.016, grain: 0.3, wear: 0.1 });
      g.save(); g.beginPath(); cap(g, -tireR + tireW, 0, tireR - tireW, 0, tireW, tireW); g.clip();
      for (let x = -tireR; x <= tireR; x += 0.045) {
        stroke(g, [[x, -tireW], [x + 0.015, -tireW * 0.45]], 0.014, '#3a3f45', 0.9);
        stroke(g, [[x + 0.02, tireW * 0.45], [x + 0.005, tireW]], 0.014, '#3a3f45', 0.9);
      }
      g.restore();
      g.fillStyle = look.trim; g.strokeStyle = INK; g.lineWidth = 0.01;
      g.beginPath(); ell(g, 0, 0, tireR * 0.42, tireW * 0.55); g.fill(); g.stroke();
      g.fillStyle = alpha(LAMP.shine, 0.7); g.beginPath(); ell(g, -tireR * 0.12, -tireW * 0.18, tireR * 0.2, tireW * 0.18); g.fill();
      g.fillStyle = INK;
      for (const k of [-0.22, 0.22]) { g.beginPath(); g.arc(tireR * k, 0, 0.009, 0, TAU); g.fill(); }
      stroke(g, [[-tireR * 0.7, -tireW * 0.75], [tireR * 0.5, -tireW * 0.75]], 0.008, LAMP.shine, 0.3);
    });
  }

  private driver(ppu: number) {
    return this.cache.get('veh-driver', ppu, -0.4, -0.3, 0.25, 0.3, (g) => {
      const rng = makeRng(hashStr('veh-driver'));
      ik(g, rng, () => ell(g, -0.16, 0, 0.15, 0.2), '#3a3f45', 0.2, { hatch: 0.5, line: 0.016 });
      stroke(g, [[-0.06, -0.06], [0.16, -0.02]], 0.022, '#3a3f45');
      stroke(g, [[-0.06, 0.06], [0.16, 0.02]], 0.022, '#3a3f45');
      ik(g, rng, () => ell(g, 0, 0, 0.13, 0.13), '#4a3022', 0.15, { hatch: 0.3, line: 0.014 });
      g.fillStyle = alpha(LAMP.shine, 0.35); g.beginPath(); g.arc(-0.04, -0.04, 0.04, 0, TAU); g.fill();
    });
  }
}
