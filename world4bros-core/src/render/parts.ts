import type { Rng } from '../engine/rng';
import type { ColorSet } from '../data/schema';
import { type G, type InkOpts, INK, TAU, inked, ell, cap, rrect, poly, smooth, lumpy, star, stroke, shade, mix, alpha } from './ink';

/** Context handed to every part drawer. Parts are drawn in local units, facing +x, pivot at (0,0). */
export interface PartCtx { g: G; rng: Rng; c: ColorSet; skin: string; hair: string; hand: string; cuff: string; build: number; side: number }
export type Drawer = (p: PartCtx) => void;
export const PARTS: Record<string, Drawer> = {};

const ik = (p: PartCtx, path: () => void, fill: string, o: InkOpts = {}): void => inked(p.g, path, fill, { rng: p.rng, box: 0.3, ...o });
const strands = (p: PartCtx, n: number, r: number, len: number, col: string, a = 0.5, w = 0.012, cx = 0, cy = 0): void => {
  for (let i = 0; i < n; i++) {
    const ang = p.rng.range(0, TAU), rr = Math.sqrt(p.rng.next()) * r, x = cx + Math.cos(ang) * rr, y = cy + Math.sin(ang) * rr, d = ang + p.rng.range(-0.5, 0.5);
    stroke(p.g, [[x, y], [x + Math.cos(d) * len, y + Math.sin(d) * len]], w, col, a);
  }
};
const flip = (p: PartCtx): void => { if (p.side < 0) p.g.scale(1, -1); };

// ---------------------------------------------------------------- cloth helpers (folds, seams, soft cast shadows)
/** Clips to `path` and paints cloth folds: armpit drag lines, a shadow under the collar and a lit shoulder ridge. */
const clothFolds = (p: PartCtx, path: () => void, col: string, w: number, n = 2): void => {
  const { g } = p;
  g.save(); g.beginPath(); path(); g.clip();
  const dark = shade(col, -0.55), lite = shade(col, 0.45);
  for (const s of [-1, 1]) {
    for (let i = 0; i < n; i++) {          // drag folds from each armpit toward the belly
      const y0 = s * w * (0.92 - i * 0.12), x0 = 0.06 - i * 0.07 + p.rng.range(-0.02, 0.02);
      const pts: [number, number][] = [[x0, y0], [x0 - 0.06, y0 - s * 0.06], [x0 - 0.1 - i * 0.03, s * w * 0.35]];
      stroke(g, pts, 0.012, dark, 0.45);
      stroke(g, pts.map(([x, y]) => [x + 0.012, y - s * 0.008] as [number, number]), 0.007, lite, 0.35);
    }
  }
  g.fillStyle = alpha(dark, 0.22); g.beginPath(); g.ellipse(0.06, 0, 0.09, w * 0.34, 0, 0, TAU); g.fill();      // shadow cast by the head
  stroke(g, [[-0.12, -w * 0.8], [-0.17, 0], [-0.12, w * 0.8]], 0.016, lite, 0.25);                                 // shoulder-blade ridge
  g.restore();
};
/** Elbow/knee creases: a few short curved ink ticks across a limb at `x`. */
const creases = (p: PartCtx, x: number, r: number, col: string): void => {
  for (let i = 0; i < 2; i++) { const xx = x + i * 0.02; p.g.beginPath(); p.g.arc(xx - r * 1.2, 0, r * 1.25, -0.55, 0.55); p.g.strokeStyle = alpha(shade(col, -0.6), 0.5 - i * 0.15); p.g.lineWidth = 0.009; p.g.stroke(); }
};

// ================================================================ body
PARTS['body:head'] = (p) => {
  const { g } = p;
  // soft contact shadow the head throws onto the shoulders (baked into the cached sprite)
  const sh = g.createRadialGradient(-0.02, 0.02, 0.1, -0.02, 0.02, 0.24);
  sh.addColorStop(0, 'rgba(14,12,20,0.32)'); sh.addColorStop(1, 'rgba(14,12,20,0)');
  g.fillStyle = sh; g.beginPath(); g.arc(-0.02, 0.02, 0.24, 0, TAU); g.fill();
  for (const s of [-1, 1]) {
    ik(p, () => ell(g, -0.02, s * 0.168, 0.034, 0.04), p.skin, { hatch: 0, line: 0.016, box: 0.1 });
    g.beginPath(); g.arc(-0.02, s * 0.172, 0.018, s > 0 ? -0.6 : 0.6 + Math.PI * 0.4, s > 0 ? 2.2 : 3.7); g.strokeStyle = alpha(shade(p.skin, -0.55), 0.6); g.lineWidth = 0.008; g.stroke();  // inner ear
  }
  ik(p, () => ell(g, 0, 0, 0.178, 0.166), p.skin, { hatch: 0.5, line: 0.024, box: 0.22 });
  g.fillStyle = alpha('#e8907a', 0.22); for (const s of [-1, 1]) { g.beginPath(); g.ellipse(0.1, s * 0.1, 0.035, 0.025, 0, 0, TAU); g.fill(); }   // cheeks
  ik(p, () => ell(g, 0.172, 0, 0.036, 0.03), shade(p.skin, -0.06), { hatch: 0, line: 0.015, box: 0.08 });
  g.beginPath(); g.arc(0.165, -0.01, 0.012, 3.4, 5.2); g.strokeStyle = alpha('#fff4dc', 0.6); g.lineWidth = 0.007; g.stroke();             // nose shine
};
PARTS['body:arm'] = (p) => {
  flip(p);
  const w = 0.062 * p.build;
  const sleeve = (): void => cap(p.g, 0, 0, 0.24, 0, w * 1.2, w);
  ik(p, sleeve, p.c.primary, { hatch: 0.7 });
  p.g.save(); p.g.beginPath(); sleeve(); p.g.clip();
  creases(p, 0.12, w, p.c.primary);
  stroke(p.g, [[0.02, -w * 0.75], [0.2, -w * 0.62]], 0.012, shade(p.c.primary, 0.5), 0.35);                     // lit top of the sleeve
  stroke(p.g, [[0.02, w * 0.9], [0.22, w * 0.75]], 0.018, shade(p.c.primary, -0.6), 0.3);                        // underside shadow
  p.g.restore();
  ik(p, () => cap(p.g, 0.222, 0, 0.255, 0, w * 1.02, w * 0.96), p.cuff, { hatch: 0, line: 0.018, box: 0.12 });
  ik(p, () => ell(p.g, 0.312, 0, 0.064, 0.058), p.hand, { hatch: 0.4, line: 0.02, box: 0.12 });
  for (let i = 0; i < 3; i++) stroke(p.g, [[0.338, -0.03 + i * 0.022], [0.364, -0.026 + i * 0.022]], 0.008, INK, 0.45);   // knuckles
};
PARTS['legs:none'] = (p) => legShape(p, p.skin, p.skin);
PARTS['feet:none'] = (p) => { ik(p, () => ell(p.g, 0.02, 0, 0.095, 0.058), p.skin, { hatch: 0.3, line: 0.02, box: 0.12 }); };
function legShape(p: PartCtx, col: string, seam: string, len = 0.21): void {
  const w = 0.072 * p.build;
  const path = (): void => cap(p.g, -0.04, 0, len, 0, w, w * 0.86);
  ik(p, path, col, { hatch: 0.7 });
  stroke(p.g, [[-0.02, -w * 0.7], [len - 0.03, -w * 0.55]], 0.011, seam, 0.5);
  p.g.save(); p.g.beginPath(); path(); p.g.clip(); creases(p, len * 0.55, w, col); p.g.restore();
}

// ================================================================ torso
PARTS['torso:none'] = (p) => ik(p, () => ell(p.g, 0, 0, 0.2, 0.31 * p.build), p.skin, { hatch: 0.6 });
PARTS['torso:hoodie'] = (p) => {
  const { g, c } = p, w = 0.31 * p.build;
  ik(p, () => lumpy(g, -0.17, 0, 0.11, 0.16, p.rng, 9, 0.1), c.secondary, { hatch: 0.8, box: 0.2 });          // hood bundled at the back
  ik(p, () => lumpy(g, 0, 0, 0.215, w, p.rng, 14, 0.04), c.primary, { hatch: 0.9, box: 0.4 });
  clothFolds(p, () => ell(g, 0, 0, 0.21, w * 0.97), c.primary, w, 3);
  stroke(g, [[0.19, 0], [-0.08, 0]], 0.013, INK, 0.7); stroke(g, [[0.19, -0.012], [-0.04, -0.012]], 0.006, shade(c.accent, 0.3), 0.7); // zip
  stroke(g, [[0.03, -w * 0.9], [0.05, -w * 0.45]], 0.011, INK, 0.5); stroke(g, [[0.03, w * 0.9], [0.05, w * 0.45]], 0.011, INK, 0.5);
  ik(p, () => ell(g, 0.02, 0.0, 0.03, 0.08), c.secondary, { hatch: 0, line: 0.014, box: 0.1 });               // collar
};
PARTS['torso:tshirt'] = (p) => {
  const { g, c } = p, w = 0.3 * p.build;
  ik(p, () => lumpy(g, 0, 0, 0.2, w, p.rng, 14, 0.03), c.primary, { hatch: 0.7, box: 0.4 });
  clothFolds(p, () => ell(g, 0, 0, 0.195, w * 0.97), c.primary, w);
  g.save(); g.beginPath(); ell(g, 0, 0, 0.2, w); g.clip(); stroke(g, [[0.07, -w], [0.07, w]], 0.05, c.accent, 0.35); stroke(g, [[-0.06, -w], [-0.06, w]], 0.02, c.secondary, 0.5); g.restore();
};
PARTS['torso:trench_coat'] = (p) => {
  const { g, c } = p, w = 0.335 * p.build;
  ik(p, () => smooth(g, [[0.12, -w * 0.8], [-0.12, -w], [-0.36, -w * 1.15], [-0.4, 0], [-0.36, w * 1.15], [-0.12, w], [0.12, w * 0.8], [0.22, 0]]), c.secondary, { hatch: 0.9, box: 0.45 });  // flared coat tails
  ik(p, () => lumpy(g, 0, 0, 0.24, w, p.rng, 14, 0.03), c.primary, { hatch: 0.9, box: 0.45 });
  clothFolds(p, () => ell(g, 0, 0, 0.235, w * 0.97), c.primary, w, 3);
  stroke(g, [[-0.2, -w * 0.97], [-0.2, w * 0.97]], 0.03, c.accent, 0.8);                                         // belt
  for (const s of [-1, 1]) { ik(p, () => rrect(g, 0.0, s * w * 0.78 - 0.035, 0.12, 0.07, 0.02), c.secondary, { hatch: 0, line: 0.015, box: 0.1 }); }
  for (let i = 0; i < 3; i++) { g.fillStyle = INK; g.beginPath(); g.arc(0.18 - i * 0.12, 0.0, 0.012, 0, TAU); g.fill(); }
};
PARTS['torso:poncho'] = (p) => {
  const { g, c } = p;
  ik(p, () => smooth(g, [[0.3, 0], [0.12, -0.34], [-0.18, -0.4], [-0.38, -0.2], [-0.4, 0], [-0.38, 0.2], [-0.18, 0.4], [0.12, 0.34]]), c.primary, { hatch: 1, box: 0.5 });
  for (const a of [-1.2, -0.55, 0.55, 1.2]) stroke(g, [[0.04, 0], [Math.cos(a) * 0.34 - 0.06, Math.sin(a) * 0.36]], 0.012, INK, 0.4);
  ik(p, () => ell(g, 0.03, 0, 0.075, 0.1), c.secondary, { hatch: 0.3, box: 0.15 });
  stroke(g, [[-0.37, -0.18], [-0.39, 0], [-0.37, 0.18]], 0.022, c.accent, 0.7);
};
PARTS['torso:hivis_vest'] = (p) => {
  const { g, c } = p, w = 0.31 * p.build;
  ik(p, () => ell(g, 0, 0, 0.205, w), c.secondary, { hatch: 0.8, box: 0.4 });
  clothFolds(p, () => ell(g, 0, 0, 0.2, w * 0.97), c.secondary, w);
  ik(p, () => rrect(g, 0.05, -w * 0.9, 0.1, w * 1.8, 0.02), c.primary, { hatch: 0.3, line: 0.016, box: 0.3 });
  ik(p, () => rrect(g, -0.17, -w * 0.9, 0.1, w * 1.8, 0.02), c.primary, { hatch: 0.3, line: 0.016, box: 0.3 });
  stroke(g, [[0.1, -w * 0.9], [0.1, w * 0.9]], 0.02, c.accent, 0.8); stroke(g, [[-0.12, -w * 0.9], [-0.12, w * 0.9]], 0.02, c.accent, 0.8);
};
PARTS['torso:sweater'] = (p) => {
  const { g, c } = p, w = 0.31 * p.build;
  ik(p, () => lumpy(g, 0, 0, 0.215, w, p.rng, 14, 0.04), c.primary, { hatch: 0.9, box: 0.4 });
  g.save(); g.beginPath(); ell(g, 0, 0, 0.215, w); g.clip();
  for (let y = -w; y < w; y += 0.045) { const pts: [number, number][] = []; for (let x = -0.22; x < 0.23; x += 0.03) pts.push([x, y + (Math.round(x / 0.03) % 2 ? 0.012 : -0.012)]); stroke(g, pts, 0.008, c.secondary, 0.55); }
  g.restore();
  clothFolds(p, () => ell(g, 0, 0, 0.21, w * 0.97), c.primary, w);
  ik(p, () => ell(g, 0.04, 0, 0.055, 0.1), c.secondary, { hatch: 0.2, line: 0.016, box: 0.12 });
};

// ================================================================ legs and feet
PARTS['legs:jeans'] = (p) => legShape(p, p.c.primary, p.c.accent);
PARTS['legs:cargo_pants'] = (p) => { legShape(p, p.c.primary, p.c.accent); ik(p, () => rrect(p.g, 0.06, -0.06, 0.09, 0.07, 0.015), p.c.secondary, { hatch: 0.2, line: 0.015, box: 0.12 }); };
PARTS['legs:overalls'] = (p) => { legShape(p, p.c.primary, p.c.accent); ik(p, () => rrect(p.g, 0.02, -0.045, 0.07, 0.06, 0.012), p.c.accent, { hatch: 0.1, line: 0.014, box: 0.1 }); };
PARTS['legs:shorts'] = (p) => {
  legShape(p, p.skin, p.skin, 0.21);
  const w = 0.074 * p.build;
  ik(p, () => cap(p.g, -0.04, 0, 0.1, 0, w * 1.12, w * 1.08), p.c.primary, { hatch: 0.7 });
  stroke(p.g, [[0.1, -w * 1.05], [0.1, w * 1.05]], 0.012, INK, 0.5);
};
PARTS['feet:boots'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0.03, 0, 0.115, 0.072), c.primary, { hatch: 0.8, box: 0.16 });
  ik(p, () => ell(g, -0.03, 0, 0.05, 0.062), c.secondary, { hatch: 0.4, line: 0.016, box: 0.1 });                  // heel and cuff
  for (let i = 0; i < 3; i++) stroke(g, [[0.03 + i * 0.03, -0.035], [0.03 + i * 0.03, 0.035]], 0.009, c.accent, 0.8);  // laces
};
PARTS['feet:sneakers'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0.03, 0, 0.11, 0.062), c.primary, { hatch: 0.5, box: 0.16 });
  stroke(g, [[-0.04, -0.03], [0.1, -0.045]], 0.016, c.secondary, 0.9); stroke(g, [[-0.04, 0.03], [0.1, 0.045]], 0.016, c.secondary, 0.9);
  ik(p, () => ell(g, 0.1, 0, 0.03, 0.045), c.accent, { hatch: 0, line: 0.012, box: 0.08 });
};
PARTS['feet:wellies'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.06, 0, 0.1, 0, 0.074, 0.066), c.primary, { hatch: 0.6, box: 0.2 });
  stroke(g, [[0.0, -0.045], [0.07, -0.03]], 0.014, shade(c.primary, 0.55), 0.7);
  ik(p, () => ell(g, -0.07, 0, 0.035, 0.075), c.secondary, { hatch: 0.2, line: 0.014, box: 0.1 });
};

// ================================================================ hair (top-down)
const hairCap = (p: PartCtx, rx = 0.19, ry = 0.182, cx = -0.015): void => ik(p, () => lumpy(p.g, cx, 0, rx, ry, p.rng, 10, 0.07), p.hair, { hatch: 0.9, box: 0.26 });
const hairHi = (p: PartCtx, n = 12): void => { strands(p, n, 0.15, 0.06, shade(p.hair, 0.38), 0.5, 0.011); strands(p, n, 0.15, 0.06, shade(p.hair, -0.55), 0.55, 0.011); };
PARTS['hair:bald'] = (p) => { stroke(p.g, [[-0.07, -0.07], [-0.02, -0.1]], 0.012, shade(p.skin, 0.4), 0.5); };
PARTS['hair:buzz'] = (p) => {
  ik(p, () => lumpy(p.g, -0.012, 0, 0.172, 0.165, p.rng, 12, 0.03), shade(p.hair, -0.1), { hatch: 0.3, box: 0.24, line: 0.018 });
  for (let i = 0; i < 70; i++) { const a = p.rng.range(0, TAU), r = Math.sqrt(p.rng.next()) * 0.15; p.g.fillStyle = alpha(shade(p.hair, p.rng.chance(0.5) ? 0.3 : -0.4), 0.55); p.g.beginPath(); p.g.arc(Math.cos(a) * r, Math.sin(a) * r, 0.007, 0, TAU); p.g.fill(); }
};
PARTS['hair:messy'] = (p) => {
  hairCap(p);
  for (let i = 0; i < 9; i++) { const a = (i / 9) * TAU + p.rng.range(-0.2, 0.2), r = 0.15; ik(p, () => cap(p.g, Math.cos(a) * r * 0.7, Math.sin(a) * r * 0.7, Math.cos(a) * (r + 0.07), Math.sin(a) * (r + 0.07), 0.03, 0.008), p.hair, { hatch: 0.4, line: 0.014, box: 0.1 }); }
  hairHi(p, 16);
};
PARTS['hair:side_part'] = (p) => { hairCap(p, 0.185, 0.18); stroke(p.g, [[0.12, -0.07], [0.02, -0.02], [-0.12, 0.03]], 0.014, INK, 0.7); hairHi(p, 10); };
PARTS['hair:long'] = (p) => {
  ik(p, () => smooth(p.g, [[0.1, -0.17], [-0.1, -0.25], [-0.34, -0.21], [-0.43, -0.04], [-0.42, 0.06], [-0.34, 0.21], [-0.1, 0.25], [0.1, 0.17]]), p.hair, { hatch: 1, box: 0.4 });
  for (let i = -3; i <= 3; i++) stroke(p.g, [[-0.12, i * 0.05], [-0.4, i * 0.05 + p.rng.range(-0.02, 0.02)]], 0.01, shade(p.hair, -0.5), 0.5);
  hairCap(p); hairHi(p, 10);
};
PARTS['hair:ponytail'] = (p) => {
  ik(p, () => cap(p.g, -0.14, 0, -0.42, 0.03, 0.05, 0.02), p.hair, { hatch: 0.9, box: 0.3 });
  stroke(p.g, [[-0.2, 0], [-0.38, 0.03]], 0.01, shade(p.hair, -0.5), 0.6);
  hairCap(p); ik(p, () => ell(p.g, -0.16, 0, 0.018, 0.05), '#8a3a3a', { hatch: 0, line: 0.012, box: 0.08 }); hairHi(p, 10);
};
PARTS['hair:bun'] = (p) => { hairCap(p); ik(p, () => ell(p.g, -0.03, 0, 0.09, 0.09), p.hair, { hatch: 0.9, box: 0.14 }); for (let r = 0.02; r < 0.08; r += 0.02) { p.g.beginPath(); p.g.arc(-0.03, 0, r, 0.4, 4.4); p.g.strokeStyle = alpha(shade(p.hair, -0.5), 0.6); p.g.lineWidth = 0.009; p.g.stroke(); } hairHi(p, 6); };
PARTS['hair:mohawk'] = (p) => {
  ik(p, () => lumpy(p.g, -0.012, 0, 0.17, 0.16, p.rng, 10, 0.03), shade(p.hair, -0.25), { hatch: 0.2, box: 0.24, line: 0.016 });
  const pts: [number, number][] = [[0.2, -0.05]]; for (let x = 0.16; x >= -0.24; x -= 0.05) { pts.push([x, -0.05 - 0.025 * (Math.round(x / 0.05) % 2)]); }
  for (let x = -0.24; x <= 0.2; x += 0.05) pts.push([x, 0.05 + 0.025 * (Math.round(x / 0.05) % 2)]);
  ik(p, () => poly(p.g, pts), p.hair, { hatch: 0.8, box: 0.26 }); hairHi(p, 5);
};
PARTS['hair:afro'] = (p) => {
  ik(p, () => lumpy(p.g, -0.02, 0, 0.26, 0.265, p.rng, 16, 0.05), p.hair, { hatch: 1, box: 0.34 });
  for (let i = 0; i < 26; i++) { const a = p.rng.range(0, TAU), r = Math.sqrt(p.rng.next()) * 0.22; p.g.beginPath(); p.g.arc(Math.cos(a) * r - 0.02, Math.sin(a) * r, 0.026, p.rng.range(0, 6), p.rng.range(0, 6) + 3.5); p.g.strokeStyle = alpha(shade(p.hair, p.rng.chance(0.4) ? 0.35 : -0.55), 0.55); p.g.lineWidth = 0.01; p.g.stroke(); }
};
PARTS['hair:braids'] = (p) => {
  for (const s of [-1, 1]) for (let i = 0; i < 5; i++) ik(p, () => ell(p.g, -0.13 - i * 0.075, s * (0.09 + i * 0.012), 0.04, 0.03, 0.3 * s), p.hair, { hatch: 0.7, line: 0.014, box: 0.08 });
  hairCap(p, 0.185, 0.178); hairHi(p, 8);
};

// ================================================================ headwear
PARTS['head:beanie'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, -0.005, 0, 0.2, 0.19), c.primary, { hatch: 0.9, box: 0.28 });
  for (let a = 0; a < TAU; a += 0.45) stroke(g, [[Math.cos(a) * 0.05, Math.sin(a) * 0.05], [Math.cos(a) * 0.17, Math.sin(a) * 0.165]], 0.01, INK, 0.28);
  g.beginPath(); ell(g, -0.005, 0, 0.19, 0.18); g.strokeStyle = c.secondary; g.lineWidth = 0.04; g.globalAlpha = 0.85; g.stroke(); g.globalAlpha = 1;      // cuff
  ik(p, () => ell(g, -0.01, 0, 0.045, 0.045), c.accent, { hatch: 0.3, line: 0.016, box: 0.08 });                                                           // pompom
};
PARTS['head:hood'] = (p) => {
  const { g, c } = p;
  ik(p, () => smooth(g, [[0.17, -0.12], [0.02, -0.22], [-0.2, -0.22], [-0.37, -0.03], [-0.38, 0.03], [-0.2, 0.22], [0.02, 0.22], [0.17, 0.12], [0.2, 0]]), c.primary, { hatch: 1, box: 0.4 });
  g.save(); g.beginPath(); smooth(g, [[0.17, -0.12], [0.02, -0.22], [-0.2, -0.22], [-0.37, -0.03], [-0.38, 0.03], [-0.2, 0.22], [0.02, 0.22], [0.17, 0.12], [0.2, 0]]); g.clip();
  stroke(g, [[0.2, -0.1], [-0.05, -0.16], [-0.3, -0.04]], 0.016, INK, 0.5); stroke(g, [[0.2, 0.1], [-0.05, 0.16], [-0.3, 0.04]], 0.016, INK, 0.5);   // fold lines
  g.restore();
  ik(p, () => ell(g, 0.1, 0, 0.1, 0.12), shade(c.secondary, -0.45), { hatch: 0, line: 0.016, box: 0.2, noWash: true });                                     // face opening in shadow
};
PARTS['head:cap'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, -0.01, 0, 0.19, 0.185), c.primary, { hatch: 0.9, box: 0.26 });
  ik(p, () => smooth(g, [[0.12, -0.15], [0.3, -0.12], [0.34, 0], [0.3, 0.12], [0.12, 0.15]]), c.secondary, { hatch: 0.8, box: 0.25 });                    // brim
  g.beginPath(); g.arc(-0.01, 0, 0.02, 0, TAU); g.fillStyle = INK; g.fill();
  stroke(g, [[-0.01, -0.18], [-0.01, 0.18]], 0.01, INK, 0.4);
};
PARTS['head:bucket_hat'] = (p) => {
  const { g, c } = p;
  ik(p, () => lumpy(g, 0, 0, 0.3, 0.29, p.rng, 14, 0.03), c.secondary, { hatch: 1, box: 0.4 });
  ik(p, () => ell(g, 0, 0, 0.195, 0.19), c.primary, { hatch: 0.8, box: 0.28 });
  g.beginPath(); ell(g, 0, 0, 0.195, 0.19); g.strokeStyle = c.accent; g.lineWidth = 0.024; g.globalAlpha = 0.9; g.stroke(); g.globalAlpha = 1;
};
PARTS['head:cone_hat'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0, 0, 0.27, 0.27), c.secondary, { hatch: 0.8, box: 0.4 });
  for (const [r, col] of [[0.215, c.primary], [0.16, c.secondary], [0.11, c.primary]] as [number, string][]) ik(p, () => ell(g, 0, 0, r, r), col, { hatch: 0.6, line: 0.016, box: 0.3 });
  ik(p, () => ell(g, 0, 0, 0.045, 0.045), shade(c.primary, -0.2), { hatch: 0, line: 0.014, box: 0.08 });
};
PARTS['head:straw_hat'] = (p) => {
  const { g, c } = p;
  ik(p, () => lumpy(g, 0, 0, 0.32, 0.31, p.rng, 16, 0.025), c.primary, { hatch: 0.9, box: 0.42 });
  for (let r = 0.08; r < 0.3; r += 0.035) { g.beginPath(); g.arc(0, 0, r, 0, TAU); g.strokeStyle = alpha(c.secondary, 0.55); g.lineWidth = 0.008; g.stroke(); }
  ik(p, () => ell(g, 0, 0, 0.17, 0.17), shade(c.primary, -0.08), { hatch: 0.5, box: 0.25 });
  g.beginPath(); ell(g, 0, 0, 0.17, 0.17); g.strokeStyle = c.accent; g.lineWidth = 0.025; g.stroke();
};
PARTS['head:pot_helmet'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0, 0, 0.215, 0.205), c.primary, { hatch: 0.9, box: 0.3 });
  g.beginPath(); g.arc(0, 0, 0.17, 0, TAU); g.strokeStyle = alpha(shade(c.primary, -0.4), 0.7); g.lineWidth = 0.014; g.stroke();
  ik(p, () => rrect(g, -0.07, -0.022, 0.14, 0.044, 0.02), c.accent, { hatch: 0, line: 0.014, box: 0.1 });
  g.beginPath(); g.arc(-0.1, -0.09, 0.05, 3.5, 5.2); g.strokeStyle = alpha('#efe8d6', 0.55); g.lineWidth = 0.014; g.stroke();
};
PARTS['head:bandana'] = (p) => {
  const { g, c } = p;
  g.beginPath(); ell(g, -0.01, 0, 0.18, 0.172); g.strokeStyle = INK; g.lineWidth = 0.075; g.stroke();
  g.beginPath(); ell(g, -0.01, 0, 0.18, 0.172); g.strokeStyle = wash2(c.primary); g.lineWidth = 0.055; g.stroke();
  for (let a = 0; a < TAU; a += 0.5) { g.fillStyle = alpha(c.secondary, 0.8); g.beginPath(); g.arc(-0.01 + Math.cos(a) * 0.18, Math.sin(a) * 0.172, 0.007, 0, TAU); g.fill(); }
  ik(p, () => poly(g, [[-0.15, -0.03], [-0.34, -0.1], [-0.3, 0.0], [-0.34, 0.1], [-0.15, 0.03]]), c.primary, { hatch: 0.8, line: 0.016, box: 0.2 });
};
const wash2 = (c: string): string => mix(c, '#777777', 0.2);

// ================================================================ face
PARTS['face:goggles'] = (p) => {
  const { g, c } = p;
  g.beginPath(); ell(g, 0.0, 0, 0.17, 0.162); g.strokeStyle = c.primary; g.lineWidth = 0.03; g.globalAlpha = 0.9; g.stroke(); g.globalAlpha = 1;
  for (const s of [-1, 1]) { ik(p, () => ell(g, 0.15, s * 0.07, 0.05, 0.058), c.secondary, { hatch: 0.3, line: 0.02, box: 0.1 }); g.beginPath(); g.arc(0.14, s * 0.065, 0.03, 3.6, 5.1); g.strokeStyle = alpha('#fff', 0.6); g.lineWidth = 0.01; g.stroke(); }
};
PARTS['face:round_glasses'] = (p) => {
  const { g, c } = p;
  for (const s of [-1, 1]) { g.beginPath(); g.arc(0.17, s * 0.065, 0.048, 0, TAU); g.fillStyle = alpha(c.secondary, 0.35); g.fill(); g.strokeStyle = c.primary; g.lineWidth = 0.014; g.stroke(); stroke(g, [[0.12, s * 0.07], [-0.1, s * 0.168]], 0.012, c.primary, 0.9); }
  stroke(g, [[0.17, -0.02], [0.17, 0.02]], 0.012, c.primary);
};
PARTS['face:fake_mustache'] = (p) => { ik(p, () => smooth(p.g, [[0.2, -0.075], [0.245, -0.03], [0.255, 0], [0.245, 0.03], [0.2, 0.075], [0.215, 0]]), p.c.primary, { hatch: 0.5, line: 0.014, box: 0.1 }); };

// ================================================================ neck, back, belt
PARTS['neck:scarf'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.2, 0.08, -0.4, 0.2, 0.05, 0.04), c.primary, { hatch: 0.8, line: 0.016, box: 0.2 });
  g.beginPath(); ell(g, -0.01, 0, 0.17, 0.215); g.strokeStyle = INK; g.lineWidth = 0.1; g.stroke();
  g.beginPath(); ell(g, -0.01, 0, 0.17, 0.215); g.strokeStyle = mix(c.primary, '#777', 0.2); g.lineWidth = 0.08; g.stroke();
  for (let a = 0; a < TAU; a += 0.35) { g.beginPath(); g.moveTo(-0.01 + Math.cos(a) * 0.14, Math.sin(a) * 0.18); g.lineTo(-0.01 + Math.cos(a + 0.12) * 0.2, Math.sin(a + 0.12) * 0.25); g.strokeStyle = alpha(c.secondary, 0.7); g.lineWidth = 0.012; g.stroke(); }
};
PARTS['back:hiker_pack'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.52, -0.25, -0.52, 0.25, 0.07, 0.07), c.secondary, { hatch: 1, box: 0.3 });                         // top bedroll
  for (const y of [-0.15, 0.15]) stroke(g, [[-0.6, y], [-0.44, y]], 0.022, INK, 0.8);
  ik(p, () => rrect(g, -0.46, -0.2, 0.08, 0.4, 0.04), shade(c.secondary, -0.15), { hatch: 0.3, line: 0.014, box: 0.2 });
  ik(p, () => rrect(g, -0.4, -0.23, 0.34, 0.46, 0.1), c.primary, { hatch: 1, box: 0.4 });                                   // main body
  ik(p, () => rrect(g, -0.31, -0.19, 0.2, 0.38, 0.07), shade(c.primary, 0.12), { hatch: 0.6, line: 0.016, box: 0.25 });     // top flap
  for (const y of [-0.1, 0.1]) { stroke(g, [[-0.3, y], [-0.12, y]], 0.018, c.accent, 0.9); g.fillStyle = '#c9c3b2'; g.beginPath(); g.arc(-0.14, y, 0.014, 0, TAU); g.fill(); }
  ik(p, () => rrect(g, -0.38, -0.3, 0.2, 0.09, 0.035), c.secondary, { hatch: 0.6, line: 0.016, box: 0.15 });                // side pockets
  ik(p, () => ell(g, -0.28, 0.285, 0.055, 0.05), '#6f7780', { hatch: 0.8, line: 0.016, box: 0.1 });                         // canteen
  ik(p, () => ell(g, -0.28, 0.285, 0.022, 0.02), '#3c4146', { hatch: 0, line: 0.01, box: 0.05 });
  for (const s of [-1, 1]) stroke(g, [[-0.05, s * 0.2], [0.02, s * 0.27]], 0.024, c.accent, 0.8);                           // shoulder straps
};
PARTS['back:school_pack'] = (p) => {
  const { g, c } = p;
  ik(p, () => rrect(g, -0.36, -0.19, 0.28, 0.38, 0.09), c.primary, { hatch: 1, box: 0.35 });
  ik(p, () => rrect(g, -0.3, -0.14, 0.14, 0.28, 0.05), c.accent, { hatch: 0.5, line: 0.016, box: 0.2 });
  stroke(g, [[-0.34, 0], [-0.1, 0]], 0.012, INK, 0.6);
  for (const s of [-1, 1]) stroke(g, [[-0.07, s * 0.17], [0.02, s * 0.26]], 0.022, c.secondary, 0.9);
};
PARTS['back:satchel'] = (p) => {
  const { g, c } = p;
  stroke(g, [[0.04, -0.28], [-0.05, 0], [-0.18, 0.3]], 0.04, INK, 0.9); stroke(g, [[0.04, -0.28], [-0.05, 0], [-0.18, 0.3]], 0.026, c.accent, 1);
  ik(p, () => rrect(g, -0.32, 0.18, 0.26, 0.2, 0.05), c.primary, { hatch: 1, box: 0.25 });
  ik(p, () => rrect(g, -0.3, 0.2, 0.14, 0.16, 0.04), c.secondary, { hatch: 0.5, line: 0.016, box: 0.15 });
  g.fillStyle = c.accent; g.beginPath(); g.arc(-0.2, 0.28, 0.015, 0, TAU); g.fill();
};
PARTS['belt:tool_belt'] = (p) => {
  const { g, c } = p;
  g.beginPath(); ell(g, -0.01, 0, 0.19, 0.3); g.strokeStyle = alpha(INK, 0.55); g.lineWidth = 0.05; g.stroke();
  g.beginPath(); ell(g, -0.01, 0, 0.19, 0.3); g.strokeStyle = alpha(c.primary, 0.9); g.lineWidth = 0.034; g.stroke();
  for (const s of [-1, 1]) ik(p, () => rrect(g, 0.0, s * 0.33 - 0.035, 0.1, 0.07, 0.02), c.secondary, { hatch: 0.5, line: 0.015, box: 0.12 });
  ik(p, () => cap(g, -0.1, 0.34, 0.06, 0.37, 0.014, 0.014), c.accent, { hatch: 0, line: 0.012, box: 0.08 });
};

// ================================================================ held items (pivot = the hand, item points along +x)
const handle = (p: PartCtx, x0: number, x1: number, r0 = 0.026, r1 = 0.022): void => ik(p, () => cap(p.g, x0, 0, x1, 0, r0, r1), p.c.primary, { hatch: 0.6, line: 0.018, box: 0.35 });
PARTS['mainHand:axe'] = (p) => {
  const { g, c } = p;
  handle(p, -0.14, 0.62);
  stroke(g, [[0.0, -0.01], [0.55, -0.012]], 0.007, shade(c.primary, 0.4), 0.5);
  ik(p, () => poly(g, [[0.5, -0.035], [0.63, -0.035], [0.68, 0.0], [0.7, 0.2], [0.6, 0.235], [0.5, 0.1]]), c.secondary, { hatch: 0.9, line: 0.022, box: 0.3 });
  stroke(g, [[0.68, 0.02], [0.7, 0.2], [0.6, 0.235]], 0.014, c.accent, 0.85);
  stroke(g, [[0.5, -0.03], [0.5, 0.03]], 0.014, '#6b4a32', 0.9); stroke(g, [[0.53, -0.03], [0.53, 0.03]], 0.014, '#6b4a32', 0.9);
};
PARTS['mainHand:pickaxe'] = (p) => {
  const { g, c } = p;
  handle(p, -0.14, 0.62);
  ik(p, () => poly(g, [[0.52, -0.28], [0.58, -0.07], [0.6, -0.03], [0.6, 0.03], [0.58, 0.07], [0.52, 0.28], [0.5, 0.06], [0.5, -0.06]]), c.secondary, { hatch: 0.9, line: 0.022, box: 0.35 });
  stroke(g, [[0.53, -0.26], [0.57, -0.06]], 0.01, c.accent, 0.7);
};
PARTS['mainHand:shovel'] = (p) => {
  const { g, c } = p;
  handle(p, -0.12, 0.6); ik(p, () => ell(g, -0.12, 0, 0.03, 0.065), c.primary, { hatch: 0.2, line: 0.016, box: 0.1 });
  ik(p, () => smooth(g, [[0.58, -0.09], [0.78, -0.08], [0.86, 0], [0.78, 0.08], [0.58, 0.09]]), c.secondary, { hatch: 0.9, line: 0.022, box: 0.3 });
  stroke(g, [[0.62, 0], [0.82, 0]], 0.008, c.accent, 0.6);
};
PARTS['mainHand:torch'] = (p) => {
  const { g, c } = p;
  handle(p, -0.12, 0.5, 0.03, 0.026);
  ik(p, () => lumpy(g, 0.54, 0, 0.085, 0.07, p.rng, 8, 0.12), mix(c.primary, '#2b2b2b', 0.4), { hatch: 0.8, line: 0.018, box: 0.15 });
  ik(p, () => ell(g, 0.58, 0, 0.04, 0.04), c.secondary, { hatch: 0, line: 0.01, box: 0.08, noWash: true });
};
PARTS['offHand:torch_off'] = PARTS['mainHand:torch'] as Drawer;
PARTS['mainHand:nail_bat'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.1, 0, 0.62, 0, 0.024, 0.062), c.primary, { hatch: 0.8, line: 0.02, box: 0.4 });
  for (let i = 0; i < 6; i++) { const a = i * 1.05, x = 0.42 + (i % 3) * 0.08; stroke(g, [[x, Math.sin(a) * 0.05], [x + 0.012, Math.sin(a) * 0.082]], 0.016, c.secondary, 1); }
};
PARTS['mainHand:crowbar'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.12, 0, 0.58, 0, 0.026, 0.026), c.primary, { hatch: 0.8, line: 0.02, box: 0.4 });
  ik(p, () => cap(g, 0.58, 0, 0.66, 0.09, 0.026, 0.02), c.primary, { hatch: 0.5, line: 0.018, box: 0.15 });
  stroke(g, [[0.66, 0.09], [0.7, 0.07]], 0.014, c.accent, 0.9);
};
PARTS['mainHand:rubber_chicken'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.06, 0, 0.2, 0, 0.022, 0.03), c.secondary, { hatch: 0.3, line: 0.016, box: 0.15 });
  for (const s of [-1, 1]) ik(p, () => ell(g, 0.36, s * 0.115, 0.09, 0.045, s * 0.5), c.primary, { hatch: 0.6, line: 0.018, box: 0.15 });
  ik(p, () => ell(g, 0.35, 0, 0.14, 0.1), c.primary, { hatch: 0.7, line: 0.02, box: 0.2 });
  ik(p, () => ell(g, 0.52, 0, 0.06, 0.055), c.primary, { hatch: 0.5, line: 0.018, box: 0.12 });
  ik(p, () => poly(g, [[0.57, -0.03], [0.66, 0], [0.57, 0.03]]), c.accent, { hatch: 0, line: 0.012, box: 0.08 });
  g.fillStyle = INK; g.beginPath(); g.arc(0.53, -0.025, 0.01, 0, TAU); g.fill(); g.beginPath(); g.arc(0.53, 0.025, 0.01, 0, TAU); g.fill();
};
PARTS['mainHand:fishing_rod'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.1, 0, 1.0, 0, 0.022, 0.008), c.primary, { hatch: 0.5, line: 0.016, box: 0.5 });
  ik(p, () => ell(g, 0.1, 0.05, 0.045, 0.045), c.secondary, { hatch: 0.3, line: 0.016, box: 0.1 });
  stroke(g, [[1.0, 0], [1.1, 0.12], [1.12, 0.3]], 0.005, c.secondary, 0.8);
  ik(p, () => ell(g, 1.12, 0.32, 0.03, 0.03), c.accent, { hatch: 0, line: 0.012, box: 0.06 });
};
PARTS['offHand:lantern'] = (p) => {
  const { g, c } = p;
  g.beginPath(); g.arc(0.14, 0, 0.07, 3.3, 6.0); g.strokeStyle = INK; g.lineWidth = 0.014; g.stroke();
  ik(p, () => rrect(g, 0.06, -0.075, 0.17, 0.15, 0.035), c.primary, { hatch: 0.7, line: 0.02, box: 0.2 });
  ik(p, () => rrect(g, 0.09, -0.05, 0.11, 0.1, 0.025), c.secondary, { hatch: 0, line: 0.012, box: 0.1, noWash: true });
  ik(p, () => ell(g, 0.145, 0, 0.03, 0.03), c.accent, { hatch: 0, line: 0.008, box: 0.06, noWash: true });
};
PARTS['offHand:pot_lid'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0.15, 0, 0.19, 0.19), c.primary, { hatch: 0.9, box: 0.3 });
  g.beginPath(); g.arc(0.15, 0, 0.14, 0, TAU); g.strokeStyle = alpha(c.secondary, 0.8); g.lineWidth = 0.012; g.stroke();
  ik(p, () => ell(g, 0.15, 0, 0.04, 0.04), c.accent, { hatch: 0.2, line: 0.014, box: 0.08 });
};

// ================================================================ Phase Z18 master weapons & held gadgets
PARTS['mainHand:pool_noodle_of_justice'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.15, 0, 0.95, 0, 0.048, 0.048), c.primary, { hatch: 0.6, line: 0.018, box: 0.4 });
  ik(p, () => ell(g, 0.95, 0, 0.02, 0.02), c.accent, { hatch: 0, line: 0.012, box: 0.06 });
  for (let x = 0.1; x < 0.9; x += 0.12) {
    stroke(g, [[x, -0.045], [x, 0.045]], 0.01, c.secondary, 0.7);
  }
};

PARTS['mainHand:chainsaw_of_regret'] = (p) => {
  const { g, c } = p;
  ik(p, () => rrect(g, -0.08, -0.09, 0.32, 0.18, 0.04), c.primary, { hatch: 0.8, line: 0.02, box: 0.25 });
  stroke(g, [[-0.06, -0.08], [-0.06, 0.08], [0.12, 0.08]], 0.018, '#1e293b', 0.9);
  ik(p, () => cap(g, 0.24, 0, 0.82, 0, 0.065, 0.045), c.secondary, { hatch: 0.7, line: 0.018, box: 0.3 });
  for (let x = 0.26; x <= 0.8; x += 0.055) {
    stroke(g, [[x, -0.065], [x + 0.02, -0.085]], 0.012, c.accent, 0.9);
    stroke(g, [[x, 0.065], [x + 0.02, 0.085]], 0.012, c.accent, 0.9);
  }
};

PARTS['mainHand:traffic_sign_axe'] = (p) => {
  const { g, c } = p;
  handle(p, -0.15, 0.75, 0.024, 0.02);
  const cx = 0.65, cy = 0.16, r = 0.18;
  ik(p, () => poly(g, [
    [cx - 0.12, cy - r], [cx + 0.12, cy - r],
    [cx + r, cy - 0.12], [cx + r, cy + 0.12],
    [cx + 0.12, cy + r], [cx - 0.12, cy + r],
    [cx - r, cy + 0.12], [cx - r, cy - 0.12]
  ]), c.primary, { hatch: 0.9, line: 0.022, box: 0.3 });
  g.beginPath(); g.arc(cx, cy, 0.12, 0, TAU); g.strokeStyle = c.secondary; g.lineWidth = 0.016; g.stroke();
  stroke(g, [[cx - 0.06, cy], [cx + 0.06, cy]], 0.02, c.secondary, 0.9);
};

PARTS['mainHand:frying_pan'] = (p) => {
  const { g, c } = p;
  handle(p, -0.14, 0.32, 0.024, 0.022);
  ik(p, () => ell(g, 0.48, 0, 0.18, 0.18), c.primary, { hatch: 0.9, box: 0.3 });
  ik(p, () => ell(g, 0.48, 0, 0.13, 0.13), c.secondary, { hatch: 0.3, line: 0.014, box: 0.15 });
  ik(p, () => ell(g, 0.46, -0.02, 0.045, 0.045), c.accent, { hatch: 0, line: 0.01, box: 0.08 });
};

PARTS['mainHand:stale_baguette'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.12, 0, 0.68, 0, 0.042, 0.038), c.primary, { hatch: 0.7, line: 0.018, box: 0.3 });
  for (let x = 0.02; x < 0.6; x += 0.1) {
    stroke(g, [[x - 0.02, -0.035], [x + 0.03, 0.035]], 0.014, c.secondary, 0.85);
  }
};

PARTS['mainHand:slingshot'] = (p) => {
  const { g, c } = p;
  handle(p, -0.12, 0.28, 0.024, 0.024);
  stroke(g, [[0.28, 0], [0.42, -0.12]], 0.022, c.primary, 0.9);
  stroke(g, [[0.28, 0], [0.42, 0.12]], 0.022, c.primary, 0.9);
  stroke(g, [[0.42, -0.12], [0.15, 0]], 0.012, c.secondary, 0.9);
  stroke(g, [[0.42, 0.12], [0.15, 0]], 0.012, c.secondary, 0.9);
  ik(p, () => ell(g, 0.15, 0, 0.032, 0.032), c.accent, { hatch: 0, line: 0.01, box: 0.06 });
};

PARTS['mainHand:potato_cannon'] = (p) => {
  const { g, c } = p;
  ik(p, () => cap(g, -0.15, 0, 0.78, 0, 0.045, 0.042), c.primary, { hatch: 0.7, line: 0.02, box: 0.35 });
  ik(p, () => rrect(g, -0.18, -0.08, 0.22, 0.16, 0.03), c.secondary, { hatch: 0.8, line: 0.018, box: 0.2 });
  ik(p, () => ell(g, 0.82, 0, 0.045, 0.038), c.accent, { hatch: 0.3, line: 0.012, box: 0.08 });
};

PARTS['mainHand:air_horn'] = (p) => {
  const { g, c } = p;
  ik(p, () => rrect(g, -0.06, -0.06, 0.22, 0.12, 0.03), c.primary, { hatch: 0.8, line: 0.018, box: 0.2 });
  ik(p, () => rrect(g, 0.02, -0.09, 0.06, 0.03, 0.01), c.secondary, { hatch: 0, line: 0.01, box: 0.05 });
  ik(p, () => poly(g, [[0.16, -0.02], [0.46, -0.14], [0.48, 0.14], [0.16, 0.02]]), c.accent, { hatch: 0.8, line: 0.02, box: 0.25 });
  ik(p, () => ell(g, 0.48, 0, 0.02, 0.14), shade(c.accent, -0.2), { hatch: 0, line: 0.014, box: 0.1 });
};

PARTS['mainHand:hunting_rifle'] = (p) => {
  const { g, c } = p;
  // Wooden stock & grip
  ik(p, () => rrect(g, -0.25, -0.04, 0.35, 0.08, 0.02), c.primary, { hatch: 0.8, line: 0.018, box: 0.2 });
  // Long steel barrel
  ik(p, () => cap(g, 0.05, 0, 0.95, 0, 0.028, 0.024), c.secondary, { hatch: 0.7, line: 0.016, box: 0.4 });
  // Telescopic optical sight
  ik(p, () => rrect(g, 0.02, -0.065, 0.28, 0.04, 0.01), c.accent, { hatch: 0.3, line: 0.012, box: 0.15 });
  stroke(g, [[0.06, -0.045], [0.06, -0.025]], 0.015, '#1e293b', 0.9);
  stroke(g, [[0.26, -0.045], [0.26, -0.025]], 0.015, '#1e293b', 0.9);
};

PARTS['mainHand:pump_shotgun'] = (p) => {
  const { g, c } = p;
  // Stock & receiver
  ik(p, () => rrect(g, -0.22, -0.05, 0.32, 0.1, 0.02), c.primary, { hatch: 0.8, line: 0.02, box: 0.25 });
  // Heavy twin barrel & mag tube
  ik(p, () => cap(g, 0.06, -0.015, 0.76, -0.015, 0.032, 0.03), c.secondary, { hatch: 0.7, line: 0.018, box: 0.35 });
  ik(p, () => cap(g, 0.06, 0.022, 0.65, 0.022, 0.025, 0.024), shade(c.secondary, -0.2), { hatch: 0.6, line: 0.016, box: 0.3 });
  // Ribbed pump slide
  ik(p, () => rrect(g, 0.28, 0.005, 0.18, 0.045, 0.01), c.accent, { hatch: 0.9, line: 0.016, box: 0.12 });
};

PARTS['mainHand:nailgun_smg'] = (p) => {
  const { g, c } = p;
  // Pneumatic main housing
  ik(p, () => rrect(g, -0.12, -0.06, 0.42, 0.12, 0.03), c.primary, { hatch: 0.9, line: 0.02, box: 0.25 });
  // Front muzzle & safety nose
  ik(p, () => rrect(g, 0.3, -0.03, 0.18, 0.06, 0.01), c.secondary, { hatch: 0.7, line: 0.016, box: 0.15 });
  // Curved nail strip magazine
  ik(p, () => poly(g, [[0.08, 0.06], [0.22, 0.06], [0.28, 0.26], [0.18, 0.28]]), c.accent, { hatch: 0.8, line: 0.018, box: 0.2 });
  // Air canister cylinder
  ik(p, () => cap(g, -0.16, 0.02, -0.02, 0.02, 0.035, 0.032), '#475569', { hatch: 0.5, line: 0.014, box: 0.1 });
};

// ================================================================ Phase Z18 gear & wearables
PARTS['head:propeller_beanie'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0, 0, 0.21, 0.20), c.primary, { hatch: 0.9, box: 0.3 });
  stroke(g, [[-0.2, 0], [0.2, 0]], 0.02, c.secondary, 0.8);
  ik(p, () => ell(g, 0, 0, 0.025, 0.025), '#1e293b', { hatch: 0, line: 0.012, box: 0.05 });
  ik(p, () => poly(g, [[-0.18, -0.03], [0, 0], [-0.18, 0.03]]), c.accent, { hatch: 0, line: 0.012, box: 0.1 });
  ik(p, () => poly(g, [[0.18, -0.03], [0, 0], [0.18, 0.03]]), c.accent, { hatch: 0, line: 0.012, box: 0.1 });
};

PARTS['torso:pillow_vest'] = (p) => {
  const { g, c } = p, w = 0.32 * p.build;
  ik(p, () => lumpy(g, 0, 0, 0.24, w, p.rng, 10, 0.06), c.primary, { hatch: 0.5, line: 0.022, box: 0.4 });
  stroke(g, [[-0.12, -w * 0.7], [0.12, w * 0.7]], 0.012, c.secondary, 0.7);
  stroke(g, [[-0.12, w * 0.7], [0.12, -w * 0.7]], 0.012, c.secondary, 0.7);
  stroke(g, [[-0.02, -w], [-0.02, w]], 0.024, c.accent, 0.9);
};

PARTS['feet:boing_boots'] = (p) => {
  const { g, c } = p;
  ik(p, () => ell(g, 0.03, 0, 0.12, 0.075), c.primary, { hatch: 0.8, line: 0.02, box: 0.2 });
  for (let i = 0; i < 4; i++) {
    const x = -0.06 + i * 0.045;
    stroke(g, [[x, -0.06], [x + 0.02, 0.06]], 0.016, c.accent, 0.9);
  }
};

PARTS['belt:fanny_pack_of_holding'] = (p) => {
  const { g, c } = p;
  g.beginPath(); ell(g, -0.01, 0, 0.19, 0.3); g.strokeStyle = alpha(c.secondary, 0.9); g.lineWidth = 0.035; g.stroke();
  ik(p, () => rrect(g, 0.16, -0.11, 0.1, 0.22, 0.04), c.primary, { hatch: 0.8, line: 0.018, box: 0.2 });
  stroke(g, [[0.22, -0.09], [0.22, 0.09]], 0.014, c.accent, 1.0);
};

// ================================================================ W4B Showcase: Breacher, Welder & Duck Hazmat
PARTS['head:breacher_helmet'] = (p) => {
  const { g, c } = p;
  // Kevlar dome
  ik(p, () => ell(g, -0.01, 0, 0.21, 0.20), c.primary, { hatch: 0.9, box: 0.3 });
  // Night-vision shroud mount on brow
  ik(p, () => rrect(g, 0.14, -0.04, 0.06, 0.08, 0.01), c.secondary, { hatch: 0, line: 0.014, box: 0.1 });
  // Curved ballistic visor slit
  ik(p, () => smooth(g, [[0.12, -0.12], [0.22, -0.08], [0.24, 0], [0.22, 0.08], [0.12, 0.12]]), c.accent, { hatch: 0.3, line: 0.016, box: 0.15 });
  stroke(g, [[-0.08, -0.18], [-0.08, 0.18]], 0.014, '#0f172a', 0.6);
};

PARTS['torso:breacher_vest'] = (p) => {
  const { g, c } = p, w = 0.33 * p.build;
  // High ballistic collar
  ik(p, () => rrect(g, -0.04, -0.14, 0.14, 0.28, 0.04), c.secondary, { hatch: 0.6, line: 0.018, box: 0.2 });
  // Heavy torso ceramic plate
  ik(p, () => rrect(g, -0.16, -w, 0.36, w * 2, 0.06), c.primary, { hatch: 0.9, box: 0.4 });
  // MOLLE webbing straps
  for (let x = -0.08; x <= 0.12; x += 0.065) {
    stroke(g, [[x, -w * 0.75], [x, w * 0.75]], 0.012, c.secondary, 0.8);
  }
  // Hazard chevron patch
  ik(p, () => poly(g, [[0.06, -0.05], [0.12, 0], [0.06, 0.05]]), c.accent, { hatch: 0, line: 0.01, box: 0.06 });
};

PARTS['head:welder_mask'] = (p) => {
  const { g, c } = p;
  // Full-face industrial hood
  ik(p, () => poly(g, [[-0.1, -0.18], [0.15, -0.17], [0.24, -0.08], [0.24, 0.08], [0.15, 0.17], [-0.1, 0.18]]), c.primary, { hatch: 0.85, box: 0.3 });
  // Dark emerald glass weld viewport
  ik(p, () => rrect(g, 0.08, -0.07, 0.12, 0.14, 0.02), c.accent, { hatch: 0.2, line: 0.016, box: 0.12 });
  // Brass pivot hinges
  stroke(g, [[-0.02, -0.18], [-0.02, 0.18]], 0.014, c.secondary, 0.7);
};

PARTS['torso:welder_apron'] = (p) => {
  const { g, c } = p, w = 0.31 * p.build;
  // Split leather shop apron
  ik(p, () => smooth(g, [[0.18, -w * 0.6], [0.14, -w * 0.9], [-0.22, -w], [-0.24, w], [0.14, w * 0.9], [0.18, w * 0.6]]), c.primary, { hatch: 0.9, box: 0.4 });
  // Front tool pouch
  ik(p, () => rrect(g, -0.05, -0.12, 0.15, 0.24, 0.02), c.secondary, { hatch: 0.5, line: 0.016, box: 0.18 });
  stroke(g, [[-0.12, -w * 0.8], [0.12, -w * 0.5]], 0.016, c.accent, 0.8);
  stroke(g, [[-0.12, w * 0.8], [0.12, w * 0.5]], 0.016, c.accent, 0.8);
};

PARTS['head:hazmat_hood'] = (p) => {
  const { g, c } = p;
  // Duck yellow pressurized bubble hood
  ik(p, () => ell(g, 0, 0, 0.23, 0.22), c.primary, { hatch: 0.7, box: 0.32 });
  // Panoramic clear/cyan bubble visor
  ik(p, () => ell(g, 0.08, 0, 0.12, 0.14), c.accent, { hatch: 0.2, line: 0.016, box: 0.18 });
  // Twin chemical gas cartridge filters on cheek sides
  for (const s of [-1, 1]) {
    ik(p, () => ell(g, 0.02, s * 0.19, 0.05, 0.045), c.secondary, { hatch: 0.4, line: 0.014, box: 0.08 });
  }
};

PARTS['torso:hazmat_suit'] = (p) => {
  const { g, c } = p, w = 0.33 * p.build;
  // Sealed rubberized hazmat torso
  ik(p, () => lumpy(g, 0, 0, 0.24, w, p.rng, 12, 0.04), c.primary, { hatch: 0.7, box: 0.4 });
  // Vulcanized black seam tape
  stroke(g, [[-0.18, 0], [0.18, 0]], 0.022, c.secondary, 0.9);
  stroke(g, [[0, -w * 0.8], [0, w * 0.8]], 0.016, c.secondary, 0.8);
  // Chest bio-hazard caution patch
  ik(p, () => ell(g, 0.06, 0.05, 0.035, 0.035), c.secondary, { hatch: 0, line: 0.01, box: 0.06 });
};

PARTS['back:quack_pak'] = (p) => {
  const { g, c } = p;
  // Twin pressurized duck-yellow fuel canisters
  for (const y of [-0.15, 0.15]) {
    ik(p, () => cap(g, -0.48, y, -0.22, y, 0.08, 0.08), c.primary, { hatch: 0.8, box: 0.25 });
    ik(p, () => ell(g, -0.5, y, 0.04, 0.06), c.secondary, { hatch: 0.4, line: 0.014, box: 0.08 }); // exhaust bells
    stroke(g, [[-0.42, y - 0.07], [-0.42, y + 0.07]], 0.016, '#1e293b', 0.9); // mounting band
  }
  // Center emergency rubber duck valve
  ik(p, () => ell(g, -0.32, 0, 0.06, 0.06), c.primary, { hatch: 0.4, line: 0.014, box: 0.08 });
  ik(p, () => poly(g, [[-0.34, -0.02], [-0.42, 0], [-0.34, 0.02]]), c.secondary, { hatch: 0, line: 0.01, box: 0.04 }); // orange duck beak
  for (const s of [-1, 1]) stroke(g, [[-0.1, s * 0.18], [0.02, s * 0.26]], 0.022, c.secondary, 0.9);
};

PARTS['mainHand:recon_rifle'] = (p) => {
  const { g, c } = p;
  // Composite skeletal stock
  ik(p, () => rrect(g, -0.28, -0.04, 0.36, 0.08, 0.02), c.primary, { hatch: 0.8, line: 0.018, box: 0.2 });
  // Long fluted precision barrel
  ik(p, () => cap(g, 0.08, 0, 1.05, 0, 0.026, 0.02), c.secondary, { hatch: 0.7, line: 0.016, box: 0.45 });
  // Elevated sniper scope with cyan lens glow
  ik(p, () => rrect(g, 0.08, -0.07, 0.32, 0.045, 0.01), c.secondary, { hatch: 0.5, line: 0.014, box: 0.15 });
  ik(p, () => ell(g, 0.4, -0.048, 0.018, 0.02), c.accent, { hatch: 0, line: 0.008, box: 0.04 });
  // Folded bipod legs
  stroke(g, [[0.78, 0.02], [0.92, 0.08]], 0.014, '#1e293b', 0.8);
};

PARTS['mainHand:scrap_nailgun'] = (p) => {
  const { g, c } = p;
  // Heavy workshop air-driver casing
  ik(p, () => rrect(g, -0.1, -0.07, 0.38, 0.14, 0.03), c.primary, { hatch: 0.85, line: 0.02, box: 0.25 });
  // Hardened steel front contact nose
  ik(p, () => rrect(g, 0.28, -0.035, 0.16, 0.07, 0.01), c.secondary, { hatch: 0.6, line: 0.016, box: 0.12 });
  // Straight strip nail magazine
  ik(p, () => rrect(g, 0.08, 0.07, 0.12, 0.24, 0.02), c.secondary, { hatch: 0.7, line: 0.016, box: 0.18 });
  for (let y = 0.1; y <= 0.28; y += 0.04) {
    stroke(g, [[0.1, y], [0.18, y]], 0.01, c.accent, 0.9);
  }
  // Coiled air hose connector
  stroke(g, [[-0.08, 0.06], [-0.14, 0.16], [-0.08, 0.24]], 0.018, c.accent, 0.9);
};


