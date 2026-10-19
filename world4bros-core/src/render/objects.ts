import type { Rng } from '../engine/rng';
import type { ObjectKind } from '../data/schema';
import { type G, INK, TAU, inked, ell, cap, rrect, poly, smooth, lumpy, star, stroke, shade, alpha, mix, type InkOpts } from './ink';
import { drawSandbags, drawSlimeDrum, drawScrapWall, drawTacticalShield, TOWN_PROP_DRAWERS } from './drawers/objectDrawers';

/** `upright`: drawn as-is (the variant carries its facing) instead of spun by a per-object random angle. */
export interface ObjDrawer { bounds: [number, number, number, number]; shadow: [number, number, number, number]; upright?: boolean; draw: (g: G, rng: Rng, v: number) => void }
const B = (r: number): [number, number, number, number] => [-r, -r, r, r];
const ik = (g: G, rng: Rng, path: () => void, fill: string, box: number, o: InkOpts = {}): void => inked(g, path, fill, { rng, box, ...o });
const sprinkle = (g: G, rng: Rng, n: number, r: number, col: string, a: number, len: number, w = 0.012, cx = 0, cy = 0): void => {
  for (let i = 0; i < n; i++) { const ang = rng.range(0, TAU), rr = Math.sqrt(rng.next()) * r, x = cx + Math.cos(ang) * rr, y = cy + Math.sin(ang) * rr, d = ang + rng.range(-0.6, 0.6); stroke(g, [[x, y], [x + Math.cos(d) * len, y + Math.sin(d) * len]], w, col, a); }
};

function pine(g: G, rng: Rng, snow: boolean): void {
  const rot = rng.range(0, TAU);
  const rings: [number, number, string][] = [[1.0, 11, '#34453d'], [0.76, 10, '#3d5046'], [0.53, 9, '#475c51'], [0.31, 8, '#52695c']];
  rings.forEach(([r, n, col], k) => {
    ik(g, rng, () => star(g, 0, 0, r * 0.66, r, n, rot + k * 0.37, rng, 0.22), col, r, { hatch: 1.1, light: [-0.5, -0.8] });
    // needle tufts: each star tip gets a darker centre rib and a lit edge toward the top-left
    for (let i = 0; i < n; i++) {
      const a = rot + k * 0.37 + (i / n) * TAU, tx = Math.cos(a) * r * 0.95, ty = Math.sin(a) * r * 0.95;
      stroke(g, [[tx * 0.55, ty * 0.55], [tx, ty]], 0.012, shade(col, -0.55), 0.55);
      if (Math.cos(a) * -0.55 + Math.sin(a) * -0.8 > 0.2) stroke(g, [[tx * 0.6 + 0.02, ty * 0.6], [tx * 0.92 + 0.015, ty * 0.92]], 0.008, shade(col, 0.45), 0.55);
    }
    sprinkle(g, rng, 26, r * 0.85, shade(col, -0.5), 0.55, 0.09, 0.011);
    sprinkle(g, rng, 14, r * 0.8, shade(col, 0.35), 0.45, 0.08, 0.009);
    if (snow) ik(g, rng, () => lumpy(g, -r * 0.22, -r * 0.24, r * 0.55, r * 0.42, rng, 9, 0.18, rng.range(0, 3)), '#c9d1d8', r * 0.5, { hatch: 0.5, line: 0.016, light: [-0.5, -0.8] });
  });
  ik(g, rng, () => ell(g, 0, 0, 0.06, 0.06), '#6a5645', 0.08, { hatch: 0.2, line: 0.014 });
}
function branch(g: G, rng: Rng, x: number, y: number, a: number, len: number, w: number, depth: number): void {
  const x2 = x + Math.cos(a) * len, y2 = y + Math.sin(a) * len, nx = -Math.sin(a), ny = Math.cos(a);
  ik(g, rng, () => poly(g, [[x + nx * w, y + ny * w], [x2 + nx * w * 0.55, y2 + ny * w * 0.55], [x2 - nx * w * 0.55, y2 - ny * w * 0.55], [x - nx * w, y - ny * w]]), '#4e463f', Math.max(0.12, len * 0.6), { hatch: 0.9, line: depth > 1 ? 0.02 : 0.014 });
  if (depth <= 0) return;
  const kids = depth > 1 ? 2 : rng.chance(0.5) ? 2 : 1;
  for (let k = 0; k < kids; k++) branch(g, rng, x2, y2, a + rng.range(-0.95, 0.95), len * rng.range(0.62, 0.8), w * 0.62, depth - 1);
}
function deadTree(g: G, rng: Rng): void {
  const n = 5 + rng.int(0, 2), a0 = rng.range(0, TAU);
  for (let i = 0; i < n; i++) branch(g, rng, 0, 0, a0 + (i / n) * TAU + rng.range(-0.3, 0.3), rng.range(0.34, 0.5), 0.055, 2);
  ik(g, rng, () => lumpy(g, 0, 0, 0.14, 0.14, rng, 9, 0.12), '#5a524a', 0.18, { hatch: 0.9, line: 0.024 });
  g.fillStyle = alpha(INK, 0.7); g.beginPath(); g.arc(rng.range(-0.04, 0.04), rng.range(-0.04, 0.04), 0.03, 0, TAU); g.fill();
}
function oak(g: G, rng: Rng): void {
  const blobs: [number, number, number, string][] = [];
  for (let i = 0; i < 8; i++) { const a = (i / 8) * TAU + rng.range(-0.2, 0.2), d = 0.55 + rng.range(-0.08, 0.1); blobs.push([Math.cos(a) * d, Math.sin(a) * d, rng.range(0.34, 0.46), i % 2 ? '#52633f' : '#4a5a3a']);
  }
  blobs.sort((p, q) => p[1] - q[1]);
  // limbs peek out between the outer clusters (trunk shading: dark underside, lit top edge)
  for (let i = 0; i < 5; i++) {
    const a = rng.range(0, TAU), l = rng.range(0.45, 0.7);
    stroke(g, [[0, 0], [Math.cos(a) * l * 0.5, Math.sin(a) * l * 0.5 + 0.02], [Math.cos(a) * l, Math.sin(a) * l]], 0.07, INK, 0.9);
    stroke(g, [[0, 0], [Math.cos(a) * l * 0.5, Math.sin(a) * l * 0.5 + 0.02], [Math.cos(a) * l, Math.sin(a) * l]], 0.045, '#5a4a3c', 1);
    stroke(g, [[-0.012, -0.012], [Math.cos(a) * l - 0.012, Math.sin(a) * l - 0.012]], 0.012, '#8a7560', 0.6);
  }
  blobs.push([0, 0, 0.55, '#5c6e47']);
  blobs.forEach(([x, y, r, col]) => {
    ik(g, rng, () => lumpy(g, x, y, r, r * 0.92, rng, 9, 0.14), col, r, { hatch: 1 });
    leafClusters(g, rng, x, y, r, col);
    sprinkle(g, rng, 14, r * 0.8, shade(col, -0.5), 0.5, 0.07, 0.01, x, y); sprinkle(g, rng, 9, r * 0.7, shade(col, 0.4), 0.5, 0.07, 0.009, x - r * 0.2, y - r * 0.25);
  });
}
/** Lit leaf clusters on the top-left of a canopy blob, a shadow pocket on the bottom-right. */
function leafClusters(g: G, rng: Rng, x: number, y: number, r: number, col: string): void {
  g.save(); g.beginPath(); g.ellipse(x + r * 0.28, y + r * 0.3, r * 0.6, r * 0.42, 0.6, 0, TAU); g.fillStyle = alpha(shade(col, -0.6), 0.28); g.fill(); g.restore();
  for (let k = 0; k < 4; k++) {
    const a = rng.range(-2.7, -0.5), d = r * rng.range(0.2, 0.55), cr = r * rng.range(0.2, 0.3), cx = x + Math.cos(a) * d, cy = y + Math.sin(a) * d;
    ik(g, rng, () => lumpy(g, cx, cy, cr, cr * 0.85, rng, 7, 0.22), shade(col, 0.16), cr, { hatch: 0.3, line: 0.009, grain: 0.3 });
    stroke(g, [[cx - cr * 0.5, cy - cr * 0.2], [cx - cr * 0.1, cy - cr * 0.6]], 0.008, shade(col, 0.55), 0.6);
  }
}
function bush(g: G, rng: Rng, berries: boolean): void {
  for (let i = 0; i < 5; i++) { const a = (i / 5) * TAU, d = i ? 0.2 : 0; ik(g, rng, () => lumpy(g, Math.cos(a) * d, Math.sin(a) * d, 0.22, 0.2, rng, 8, 0.15), i % 2 ? '#4a5c42' : '#566a4a', 0.22, { hatch: 1, line: 0.016 }); }
  for (let i = 0; i < 5; i++) { const a = (i / 5) * TAU, d = i ? 0.2 : 0; leafClusters(g, rng, Math.cos(a) * d, Math.sin(a) * d, 0.2, i % 2 ? '#4a5c42' : '#566a4a'); }
  for (let i = 0; i < 10; i++) { const a = rng.range(0, TAU), d = rng.range(0.22, 0.36), l = rng.range(0.04, 0.07), x = Math.cos(a) * d, y = Math.sin(a) * d; stroke(g, [[x, y], [x + Math.cos(a) * l, y + Math.sin(a) * l]], 0.016, '#4a5c42', 0.9); }   // stray leaves at the rim
  sprinkle(g, rng, 14, 0.3, '#2d3a2a', 0.6, 0.07, 0.01);
  if (berries) for (let i = 0; i < 7; i++) { const a = rng.range(0, TAU), d = rng.range(0.05, 0.3); ik(g, rng, () => ell(g, Math.cos(a) * d, Math.sin(a) * d, 0.035, 0.035), '#a1403a', 0.05, { hatch: 0.2, line: 0.01 }); }
}
function rock(g: G, rng: Rng, r: number, mossy: boolean): void {
  const pts: [number, number][] = []; const n = 8;
  for (let i = 0; i < n; i++) { const a = (i / n) * TAU + rng.range(-0.15, 0.15), k = r * rng.range(0.8, 1.08); pts.push([Math.cos(a) * k, Math.sin(a) * k * 0.9]); }
  ik(g, rng, () => poly(g, pts), '#6c7076', r, { hatch: 0.8, line: 0.026 });
  // facets: a fan from an apex nudged toward the light; faces turned to the top-left catch light, the rest fall into shade
  const ax = -r * 0.2 + rng.range(-0.05, 0.05) * r, ay = -r * 0.22;
  for (let i = 0; i < n; i++) {
    const p = pts[i] as [number, number], q = pts[(i + 1) % n] as [number, number];
    const mx = (p[0] + q[0]) / 2 - ax, my = (p[1] + q[1]) / 2 - ay, lit = (mx * -0.55 + my * -0.8) / Math.max(1e-3, Math.hypot(mx, my));
    g.fillStyle = lit > 0 ? alpha('#efe8d6', 0.2 * lit) : alpha(INK, 0.26 * -lit);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.closePath(); g.fill();
    stroke(g, [[ax, ay], [p[0] * 0.96, p[1] * 0.96]], 0.009, INK, 0.35);
  }
  for (let i = 0; i < 3; i++) { const p = pts[rng.int(0, n - 1)] as [number, number]; stroke(g, [[p[0] * 0.35, p[1] * 0.35], [p[0] * 0.6 + rng.range(-0.03, 0.03), p[1] * 0.6], [p[0] * 0.85, p[1] * 0.85]], 0.01, INK, 0.5); }
  sprinkle(g, rng, Math.round(r * 30), r * 0.8, '#3e4247', 0.5, 0.02, 0.01);
  for (let i = 0; i < 3; i++) { g.fillStyle = alpha('#c9c48a', 0.5); g.beginPath(); g.arc(rng.range(-r, r) * 0.6, rng.range(-r, r) * 0.6, r * rng.range(0.04, 0.07), 0, TAU); g.fill(); }   // lichen
  ik(g, rng, () => poly(g, [[-r * 0.5, -r * 0.45], [-r * 0.05, -r * 0.62], [r * 0.2, -r * 0.2], [-r * 0.3, -r * 0.1]]), '#8b9097', r * 0.5, { hatch: 0.2, line: 0.014 });
  if (mossy) ik(g, rng, () => lumpy(g, r * 0.15, -r * 0.3, r * 0.35, r * 0.2, rng, 8, 0.2), '#56684a', r * 0.4, { hatch: 0.5, line: 0.014 });
}
function log(g: G, rng: Rng): void {
  ik(g, rng, () => cap(g, -0.55, 0, 0.55, 0, 0.2, 0.19), '#6a5645', 0.55, { hatch: 1 });
  for (let i = 0; i < 9; i++) { const y = rng.range(-0.16, 0.16), x = rng.range(-0.45, 0.2); stroke(g, [[x, y], [x + rng.range(0.15, 0.35), y + rng.range(-0.02, 0.02)]], 0.011, INK, 0.5); }
  ik(g, rng, () => ell(g, 0.55, 0, 0.07, 0.19), '#a58d6c', 0.2, { hatch: 0.4, line: 0.018 });
  for (const r of [0.04, 0.09, 0.14]) { g.beginPath(); g.ellipse(0.55, 0, r * 0.4, r * 1.3, 0, 0, TAU); g.strokeStyle = alpha(INK, 0.5); g.lineWidth = 0.007; g.stroke(); }
  ik(g, rng, () => ell(g, -0.1, 0.05, 0.05, 0.04), '#3d3128', 0.08, { hatch: 0, line: 0.012 });
}
function stump(g: G, rng: Rng): void {
  for (let i = 0; i < 5; i++) { const a = (i / 5) * TAU + rng.range(-0.3, 0.3); ik(g, rng, () => cap(g, Math.cos(a) * 0.2, Math.sin(a) * 0.2, Math.cos(a) * 0.42, Math.sin(a) * 0.42, 0.07, 0.025), '#5a4838', 0.4, { hatch: 0.8, line: 0.016 }); }
  ik(g, rng, () => lumpy(g, 0, 0, 0.29, 0.28, rng, 10, 0.06), '#6a5645', 0.3, { hatch: 0.8 });
  ik(g, rng, () => ell(g, 0, 0, 0.23, 0.22), '#a58d6c', 0.25, { hatch: 0.3, line: 0.012 });
  for (let r = 0.04; r < 0.22; r += 0.04) { g.beginPath(); g.arc(0, 0, r + rng.range(-0.005, 0.005), 0, TAU); g.strokeStyle = alpha('#4a3b30', 0.75); g.lineWidth = 0.008; g.stroke(); }
  stroke(g, [[0, 0], [0.2, 0.08]], 0.009, INK, 0.5);
}
function mushroom(g: G, rng: Rng): void {
  for (let i = 0; i < 3; i++) { const x = (i - 1) * 0.12 + rng.range(-0.03, 0.03), y = rng.range(-0.06, 0.06), r = rng.range(0.07, 0.11); ik(g, rng, () => ell(g, x, y, r, r), i === 1 ? '#a8997f' : '#bba98c', r, { hatch: 0.7, line: 0.014 }); for (let k = 0; k < 3; k++) { g.fillStyle = alpha('#efe8d6', 0.8); g.beginPath(); g.arc(x + rng.range(-r * 0.5, r * 0.5), y + rng.range(-r * 0.5, r * 0.5), 0.012, 0, TAU); g.fill(); } }
}
function reeds(g: G, rng: Rng): void {
  for (let i = 0; i < 18; i++) { const a = rng.range(0, TAU), l = rng.range(0.15, 0.32); const mx = Math.cos(a) * l * 0.5 + rng.range(-0.04, 0.04), my = Math.sin(a) * l * 0.5; stroke(g, [[0, 0], [mx, my], [Math.cos(a) * l, Math.sin(a) * l]], 0.026, INK, 0.9); stroke(g, [[0, 0], [mx, my], [Math.cos(a) * l, Math.sin(a) * l]], 0.016, i % 3 ? '#7a8456' : '#98a06a', 1); }
  for (let i = 0; i < 3; i++) ik(g, rng, () => ell(g, rng.range(-0.12, 0.12), rng.range(-0.12, 0.12), 0.03, 0.05, rng.range(0, 3)), '#6a4a36', 0.06, { hatch: 0.2, line: 0.012 });
}
function cactus(g: G, rng: Rng): void {
  for (const s of [-1, 1]) ik(g, rng, () => cap(g, 0, 0, s * 0.3, rng.range(-0.1, 0.1), 0.07, 0.06), '#5f7a5a', 0.3, { hatch: 0.9, line: 0.018 });
  ik(g, rng, () => star(g, 0, 0, 0.2, 0.27, 9, rng.range(0, 3), rng, 0.05), '#688460', 0.28, { hatch: 1 });
  for (let a = 0; a < TAU; a += 0.7) stroke(g, [[Math.cos(a) * 0.05, Math.sin(a) * 0.05], [Math.cos(a) * 0.23, Math.sin(a) * 0.23]], 0.008, INK, 0.5);
  for (let i = 0; i < 14; i++) { const a = rng.range(0, TAU), d = rng.range(0.1, 0.27); g.fillStyle = '#d8d0b0'; g.beginPath(); g.arc(Math.cos(a) * d, Math.sin(a) * d, 0.008, 0, TAU); g.fill(); }
}
function crate(g: G, rng: Rng): void {
  ik(g, rng, () => rrect(g, -0.34, -0.34, 0.68, 0.68, 0.03), '#7a6248', 0.4, { hatch: 1, line: 0.026 });
  // four planks, each with its own tone, grain and a knot
  for (let i = 0; i < 4; i++) {
    const y0 = -0.34 + i * 0.17, tone = shade('#7a6248', rng.range(-0.12, 0.1));
    g.fillStyle = alpha(tone, 0.6); g.fillRect(-0.32, y0 + 0.012, 0.64, 0.146);
    for (let k = 0; k < 3; k++) { const y = y0 + rng.range(0.03, 0.14), x = rng.range(-0.3, 0); stroke(g, [[x, y], [x + 0.12, y + rng.range(-0.01, 0.01)], [x + rng.range(0.2, 0.3), y + rng.range(-0.015, 0.015)]], 0.006, '#3d3024', 0.5); }
    if (rng.chance(0.6)) { const kx = rng.range(-0.25, 0.25), ky = y0 + 0.085; g.beginPath(); g.ellipse(kx, ky, 0.03, 0.018, 0, 0, TAU); g.strokeStyle = alpha('#3d3024', 0.7); g.lineWidth = 0.007; g.stroke(); }
    if (i) stroke(g, [[-0.34, y0], [0.34, y0]], 0.012, INK, 0.65);
  }
  for (const x of [-0.34, 0.27]) ik(g, rng, () => rrect(g, x, -0.34, 0.07, 0.68, 0.015), '#5e4a36', 0.35, { hatch: 0.6, line: 0.014 });   // side battens
  for (const x of [-0.305, 0.305]) for (let y = -0.29; y < 0.34; y += 0.17) { g.fillStyle = '#9aa1a8'; g.beginPath(); g.arc(x, y, 0.012, 0, TAU); g.fill(); g.fillStyle = INK; g.beginPath(); g.arc(x + 0.004, y + 0.004, 0.006, 0, TAU); g.fill(); }   // nail heads
  stroke(g, [[-0.27, -0.27], [0.27, 0.27]], 0.06, INK, 0.9); stroke(g, [[-0.27, -0.27], [0.27, 0.27]], 0.04, '#5e4a36', 1); stroke(g, [[-0.27, -0.29], [0.25, 0.23]], 0.008, '#a58d6c', 0.6);   // brace
  if (rng.chance(0.5)) { g.save(); g.globalAlpha = 0.35; stroke(g, [[0.04, -0.2], [0.18, -0.2]], 0.025, '#2b2420'); stroke(g, [[0.11, -0.14], [0.11, -0.26], [0.07, -0.22]], 0.012, '#2b2420'); g.restore(); }   // stencil mark
  stroke(g, [[-0.33, 0.3], [-0.33, -0.33], [0.3, -0.33]], 0.01, '#efe8d6', 0.35);
}
function tent(g: G, rng: Rng): void {
  ik(g, rng, () => poly(g, [[-0.62, -0.5], [0.0, -0.46], [0.0, 0.46], [-0.62, 0.5]]), '#857a62', 0.6, { hatch: 1, line: 0.026, light: [-0.4, -0.9] });
  ik(g, rng, () => poly(g, [[0.0, -0.46], [0.62, -0.5], [0.62, 0.5], [0.0, 0.46]]), '#6e6550', 0.6, { hatch: 1.3, line: 0.026, light: [-0.4, -0.9] });
  stroke(g, [[0.0, -0.46], [0.0, 0.46]], 0.03, INK, 0.9);
  stroke(g, [[0.2, -0.45], [0.35, -0.05], [0.25, 0.3]], 0.012, INK, 0.5); stroke(g, [[-0.35, -0.2], [-0.2, 0.1]], 0.012, INK, 0.5);
  ik(g, rng, () => poly(g, [[-0.35, 0.12], [-0.12, 0.2], [-0.2, 0.36], [-0.42, 0.3]]), '#5c5443', 0.2, { hatch: 0.6, line: 0.014 });   // patch
  for (const [x, y] of [[-0.62, -0.5], [-0.62, 0.5], [0.62, -0.5], [0.62, 0.5]] as [number, number][]) stroke(g, [[x, y], [x * 1.35, y * 1.25]], 0.008, INK, 0.6);
}

function fuelPump(g: G, rng: Rng): void {
  ik(g, rng, () => rrect(g, -0.28, -0.4, 0.56, 0.8, 0.04), '#c4473d', 0.45, { hatch: 1.1, line: 0.024 });
  ik(g, rng, () => ell(g, 0, -0.15, 0.16, 0.14), '#f0ece1', 0.2, { hatch: 0.3, line: 0.016 });
  stroke(g, [[0, -0.15], [0.08, -0.22]], 0.014, INK, 0.9);
  stroke(g, [[0.28, -0.1], [0.36, 0.1], [0.3, 0.35], [0.22, 0.25]], 0.02, '#2b2f34', 1);
  for (let i = 0; i < 4; i++) stroke(g, [[rng.range(-0.2, 0.2), rng.range(0.05, 0.3)], [rng.range(-0.2, 0.2), rng.range(0.05, 0.3)]], 0.01, '#5a2f26', 0.8);
}

function radioTower(g: G, rng: Rng): void {
  for (const [x, y] of [[-0.45, -0.45], [0.45, -0.45], [-0.45, 0.45], [0.45, 0.45]] as [number, number][]) {
    ik(g, rng, () => ell(g, x, y, 0.09, 0.09), '#5a626a', 0.12, { hatch: 0.5, line: 0.02 });
  }
  stroke(g, [[-0.45, -0.45], [0.45, 0.45]], 0.022, INK, 0.85);
  stroke(g, [[-0.45, 0.45], [0.45, -0.45]], 0.022, INK, 0.85);
  stroke(g, [[-0.45, -0.45], [0.45, -0.45], [0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45]], 0.026, INK, 0.95);
  ik(g, rng, () => ell(g, 0, 0, 0.24, 0.24), '#8d949b', 0.3, { hatch: 0.9, line: 0.024 });
  ik(g, rng, () => ell(g, 0, 0, 0.08, 0.08), '#e8483b', 0.1, { hatch: 0, line: 0.012 });
}

function sandbags(g: G, rng: Rng): void {
  for (let i = -2; i <= 2; i++) {
    const a = i * 0.25;
    const x = Math.sin(a) * 0.38, y = (Math.cos(a) - 1) * 0.25;
    ik(g, rng, () => cap(g, x - 0.14, y, x + 0.14, y, 0.09, 0.08), '#9a8a6e', 0.25, { hatch: 1.2, line: 0.018 });
    stroke(g, [[x - 0.1, y], [x + 0.1, y]], 0.01, INK, 0.5);
  }
}

function wreckedCar(g: G, rng: Rng): void {
  ik(g, rng, () => rrect(g, -0.7, -0.42, 1.4, 0.84, 0.1), '#6a4a35', 0.75, { hatch: 1.2, line: 0.028 });
  ik(g, rng, () => rrect(g, -0.35, -0.32, 0.7, 0.64, 0.06), '#4a3526', 0.45, { hatch: 0.8, line: 0.02 });
  stroke(g, [[-0.2, -0.2], [-0.05, -0.1], [0.1, -0.25]], 0.01, '#c9d8e0', 0.65);
  for (const [x, y] of [[-0.48, -0.44], [0.48, -0.44], [-0.48, 0.44], [0.48, 0.44]] as [number, number][]) {
    ik(g, rng, () => ell(g, x, y, 0.12, 0.06), '#1e2124', 0.12, { hatch: 0.4, line: 0.014 });
  }
  for (let i = 0; i < 5; i++) {
    const hx = rng.range(-0.55, 0.55), hy = rng.range(-0.35, 0.35);
    stroke(g, [[hx, hy], [hx + 0.02, hy + 0.02]], 0.02, INK, 0.9);
  }
}

function mineralNode(g: G, rng: Rng, baseCol: string, veinCol: string, sparkleCol: string): void {
  const pts: [number, number][] = []; const n = 9;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU + rng.range(-0.15, 0.15), k = 0.55 * rng.range(0.8, 1.15);
    pts.push([Math.cos(a) * k, Math.sin(a) * k * 0.9]);
  }
  ik(g, rng, () => poly(g, pts), baseCol, 0.6, { hatch: 1.1, line: 0.026 });
  for (let i = 0; i < 4; i++) {
    const a = rng.range(0, TAU);
    const len = rng.range(0.2, 0.45);
    stroke(g, [[Math.cos(a) * 0.1, Math.sin(a) * 0.1], [Math.cos(a) * len, Math.sin(a) * len]], 0.032, veinCol, 0.95);
  }
  for (let i = 0; i < 3; i++) {
    const a = rng.range(0, TAU), d = rng.range(0.12, 0.35);
    ik(g, rng, () => poly(g, [[Math.cos(a) * d - 0.06, Math.sin(a) * d - 0.08], [Math.cos(a) * d + 0.06, Math.sin(a) * d - 0.04], [Math.cos(a) * d + 0.04, Math.sin(a) * d + 0.07], [Math.cos(a) * d - 0.05, Math.sin(a) * d + 0.06]]), sparkleCol, 0.15, { hatch: 0.2, line: 0.012 });
  }
}

function oilSeam(g: G, rng: Rng): void {
  ik(g, rng, () => lumpy(g, 0, 0, 0.52, 0.42, rng, 9, 0.16), '#1c1f24', 0.55, { hatch: 0.6, line: 0.02 });
  ik(g, rng, () => ell(g, -0.05, -0.05, 0.32, 0.24, 0.2), '#252a32', 0.35, { hatch: 0.3, line: 0.014 });
  for (const [r, col] of [[0.18, '#3b5c6e'], [0.12, '#506e57'], [0.06, '#724e68']] as [number, string][]) {
    g.beginPath(); g.ellipse(0, 0, r, r * 0.75, 0.4, 0, TAU);
    g.strokeStyle = alpha(col, 0.75); g.lineWidth = 0.016; g.stroke();
  }
  for (let i = 0; i < 3; i++) {
    const a = rng.range(0, TAU), d = rng.range(0.08, 0.25);
    ik(g, rng, () => ell(g, Math.cos(a) * d, Math.sin(a) * d, 0.04, 0.035), '#38414e', 0.05, { hatch: 0, line: 0.01 });
  }
}

function crystalGeode(g: G, rng: Rng): void {
  const pts: [number, number][] = []; const n = 8;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU + rng.range(-0.15, 0.15), k = 0.52 * rng.range(0.85, 1.15);
    pts.push([Math.cos(a) * k, Math.sin(a) * k * 0.88]);
  }
  ik(g, rng, () => poly(g, pts), '#48444e', 0.55, { hatch: 1.0, line: 0.024 });
  ik(g, rng, () => ell(g, 0, 0, 0.34, 0.28), '#2b2238', 0.35, { hatch: 0.4, line: 0.016 });
  const prismCols = ['#9c6ade', '#b88ef2', '#e4c4fc', '#7743b8'];
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * TAU + rng.range(-0.1, 0.1), d = rng.range(0.12, 0.28);
    const col = prismCols[i % prismCols.length];
    ik(g, rng, () => poly(g, [
      [Math.cos(a) * d, Math.sin(a) * d],
      [Math.cos(a) * (d + 0.14) - 0.03, Math.sin(a) * (d + 0.14)],
      [Math.cos(a) * (d + 0.18), Math.sin(a) * (d + 0.18)],
      [Math.cos(a) * (d + 0.14) + 0.03, Math.sin(a) * (d + 0.14)],
    ]), col, 0.2, { hatch: 0.1, line: 0.012 });
  }
}

export const OBJECT_DRAWERS: Record<ObjectKind, ObjDrawer> = {
  pine: { bounds: B(1.15), shadow: [0.7, 0.55, 0.3, 0.38], draw: (g, r) => pine(g, r, false) },
  snow_pine: { bounds: B(1.15), shadow: [0.7, 0.55, 0.3, 0.38], draw: (g, r) => pine(g, r, true) },
  oak: { bounds: B(1.25), shadow: [0.85, 0.65, 0.35, 0.45], draw: oak },
  dead_tree: { bounds: B(1.0), shadow: [0.5, 0.4, 0.2, 0.25], draw: deadTree },
  bush: { bounds: B(0.55), shadow: [0.3, 0.2, 0.1, 0.12], draw: (g, r) => bush(g, r, false) },
  berry_bush: { bounds: B(0.55), shadow: [0.3, 0.2, 0.1, 0.12], draw: (g, r) => bush(g, r, true) },
  rock: { bounds: B(0.55), shadow: [0.32, 0.22, 0.1, 0.14], draw: (g, r) => rock(g, r, 0.4, false) },
  boulder: { bounds: B(1.0), shadow: [0.7, 0.45, 0.2, 0.25], draw: (g, r) => rock(g, r, 0.82, true) },
  log: { bounds: [-0.8, -0.4, 0.8, 0.4], shadow: [0.6, 0.2, 0.1, 0.16], draw: log },
  stump: { bounds: B(0.55), shadow: [0.32, 0.22, 0.1, 0.12], draw: stump },
  mushroom: { bounds: B(0.35), shadow: [0.2, 0.1, 0.05, 0.06], draw: mushroom },
  reeds: { bounds: B(0.45), shadow: [0.2, 0.12, 0.05, 0.06], draw: reeds },
  cactus: { bounds: B(0.5), shadow: [0.3, 0.2, 0.1, 0.12], draw: cactus },
  crate: { bounds: B(0.55), shadow: [0.4, 0.3, 0.1, 0.12], draw: crate },
  tent: { bounds: B(0.95), shadow: [0.7, 0.5, 0.15, 0.2], draw: tent },
  fuel_pump: { bounds: B(0.65), shadow: [0.4, 0.25, 0.1, 0.14], draw: fuelPump },
  radio_tower: { bounds: B(1.1), shadow: [0.6, 0.6, 0.2, 0.2], draw: radioTower },
  sandbags: { bounds: [-0.55, -0.35, 0.55, 0.35], shadow: [0.5, 0.25, 0.05, 0.1], draw: (g) => drawSandbags({ ctx: g, x: 0, y: 0, scale: 1 / 48, hpRatio: 1.0 }) },
  slime_drum: { bounds: [-0.4, -0.45, 0.4, 0.45], shadow: [0.35, 0.25, 0.05, 0.1], draw: (g) => drawSlimeDrum({ ctx: g, x: 0, y: 0, scale: 1 / 48, hpRatio: 1.0, timeMs: 0 }) },
  scrap_wall: { bounds: [-0.55, -0.35, 0.55, 0.35], shadow: [0.55, 0.25, 0.05, 0.12], draw: (g) => drawScrapWall({ ctx: g, x: 0, y: 0, scale: 1 / 48, hpRatio: 1.0 }) },
  tactical_shield: { bounds: [-0.5, -0.35, 0.5, 0.35], shadow: [0.5, 0.25, 0.05, 0.12], draw: (g) => drawTacticalShield({ ctx: g, x: 0, y: 0, scale: 1 / 48, hpRatio: 1.0 }) },
  wrecked_car: { bounds: [-0.9, -0.6, 0.9, 0.6], shadow: [0.8, 0.45, 0.15, 0.2], draw: wreckedCar },
  iron_node: { bounds: B(0.7), shadow: [0.55, 0.4, 0.15, 0.2], draw: (g, r) => mineralNode(g, r, '#555b62', '#8fa1b3', '#d4e1ed') },
  copper_node: { bounds: B(0.7), shadow: [0.55, 0.4, 0.15, 0.2], draw: (g, r) => mineralNode(g, r, '#5e4e42', '#c97746', '#e8ab74') },
  coal_node: { bounds: B(0.7), shadow: [0.55, 0.4, 0.15, 0.2], draw: (g, r) => mineralNode(g, r, '#2c2e33', '#1e2024', '#555b63') },
  oil_seam: { bounds: B(0.65), shadow: [0.5, 0.35, 0.1, 0.15], draw: oilSeam },
  crystal_geode: { bounds: B(0.7), shadow: [0.55, 0.4, 0.15, 0.2], draw: crystalGeode },
  town_fountain: { bounds: B(1.2), shadow: [0.9, 0.7, 0.2, 0.25], draw: townFountain },
  guardrail: { bounds: [-0.55, -0.25, 0.55, 0.25], shadow: [0.45, 0.2, 0.05, 0.1], upright: true, draw: guardrailDraw },
  picket_fence: { bounds: [-0.45, -0.3, 0.45, 0.3], shadow: [0.4, 0.2, 0.05, 0.1], draw: picketFence },
  dumpster: { bounds: [-0.65, -0.45, 0.65, 0.45], shadow: [0.6, 0.4, 0.12, 0.18], draw: dumpsterDraw },
  traffic_cone: { bounds: B(0.3), shadow: [0.22, 0.16, 0.04, 0.06], draw: trafficCone },
  target_dummy: { bounds: B(0.48), shadow: [0.35, 0.25, 0.05, 0.1], draw: targetDummyDraw },
  steel_target: { bounds: B(0.42), shadow: [0.3, 0.22, 0.05, 0.08], draw: steelTargetDraw },
  street_light: TOWN_PROP_DRAWERS.street_light!,
  park_bench: TOWN_PROP_DRAWERS.park_bench!,
  fire_hydrant: TOWN_PROP_DRAWERS.fire_hydrant!,
  stop_sign: TOWN_PROP_DRAWERS.stop_sign!,
  street_sign: TOWN_PROP_DRAWERS.street_sign!,
  bus_stop: TOWN_PROP_DRAWERS.bus_stop!,
  trash_can: TOWN_PROP_DRAWERS.trash_can!,
  power_pole: TOWN_PROP_DRAWERS.power_pole!,
  abandoned_car: TOWN_PROP_DRAWERS.abandoned_car!,
  litter: TOWN_PROP_DRAWERS.litter!,
};

function townFountain(g: G, rng: Rng): void {
  ik(g, rng, () => star(g, 0, 0, 0.95, 1.05, 8, 0, rng, 0.05), '#60646c', 1.05, { hatch: 0.8, line: 0.024 });
  ik(g, rng, () => ell(g, 0, 0, 0.78, 0.78), '#3b6e8c', 0.8, { hatch: 0.2, line: 0.016 });
  for (let r = 0.25; r < 0.75; r += 0.22) {
    g.beginPath(); g.arc(0, 0, r, 0, TAU); g.strokeStyle = alpha('#93c5fd', 0.45); g.lineWidth = 0.015; g.stroke();
  }
  ik(g, rng, () => ell(g, 0, 0, 0.32, 0.32), '#7a7f88', 0.35, { hatch: 0.6, line: 0.018 });
  ik(g, rng, () => ell(g, 0, 0, 0.12, 0.12), '#cbd5e1', 0.15, { hatch: 0, line: 0.012 });
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * TAU;
    g.fillStyle = alpha('#ffffff', 0.75);
    g.beginPath(); g.arc(Math.cos(a) * 0.18, Math.sin(a) * 0.18, 0.03, 0, TAU); g.fill();
  }
}

function guardrailDraw(g: G, rng: Rng): void {
  for (const x of [-0.35, 0.35]) {
    ik(g, rng, () => rrect(g, x - 0.04, -0.1, 0.08, 0.2, 0.01), '#3f444c', 0.15, { hatch: 0.5, line: 0.014 });
  }
  ik(g, rng, () => rrect(g, -0.45, -0.12, 0.9, 0.24, 0.03), '#94a3b8', 0.45, { hatch: 0.7, line: 0.02 });
  stroke(g, [[-0.42, 0], [0.42, 0]], 0.02, '#cbd5e1', 0.8);
  g.fillStyle = '#ef4444'; g.fillRect(-0.43, -0.06, 0.06, 0.12);
  g.fillStyle = '#ffffff'; g.fillRect(0.37, -0.06, 0.06, 0.12);
}

function picketFence(g: G, rng: Rng): void {
  stroke(g, [[-0.4, -0.08], [0.4, -0.08]], 0.02, '#9ca3af', 0.6);
  stroke(g, [[-0.4, 0.08], [0.4, 0.08]], 0.02, '#9ca3af', 0.6);
  for (let x = -0.32; x <= 0.35; x += 0.14) {
    ik(g, rng, () => poly(g, [[x - 0.04, 0.2], [x + 0.04, 0.2], [x + 0.04, -0.16], [x, -0.22], [x - 0.04, -0.16]]), '#f1f5f9', 0.2, { hatch: 0.3, line: 0.014 });
  }
}

function dumpsterDraw(g: G, rng: Rng): void {
  ik(g, rng, () => rrect(g, -0.55, -0.38, 1.1, 0.76, 0.05), '#234a41', 0.6, { hatch: 0.9, line: 0.022 });
  for (const x of [-0.3, 0, 0.3]) {
    stroke(g, [[x, -0.34], [x, 0.34]], 0.02, '#18342e', 0.7);
  }
  for (const lx of [-0.27, 0.27]) {
    ik(g, rng, () => rrect(g, lx - 0.24, -0.35, 0.48, 0.28, 0.03), '#1f2937', 0.3, { hatch: 0.6, line: 0.018 });
  }
}

function trafficCone(g: G, rng: Rng): void {
  ik(g, rng, () => rrect(g, -0.2, -0.2, 0.4, 0.4, 0.04), '#ea580c', 0.25, { hatch: 0.8, line: 0.02 });
  ik(g, rng, () => ell(g, 0, 0, 0.14, 0.14), '#f97316', 0.15, { hatch: 0.3, line: 0.016 });
  ik(g, rng, () => ell(g, 0, 0, 0.08, 0.08), '#ffffff', 0.1, { hatch: 0, line: 0.014 });
}

function targetDummyDraw(g: G, rng: Rng): void {
  // Wooden stand post
  ik(g, rng, () => rrect(g, -0.06, -0.15, 0.12, 0.4, 0.02), '#78593d', 0.3, { hatch: 0.8, line: 0.02 });
  // Cross brace
  ik(g, rng, () => rrect(g, -0.28, 0.18, 0.56, 0.08, 0.02), '#5c4028', 0.35, { hatch: 0.6, line: 0.018 });
  // Straw/canvas torso
  ik(g, rng, () => ell(g, 0, -0.06, 0.24, 0.28), '#c2a677', 0.26, { hatch: 0.8, line: 0.02 });
  // Bullseye target outer ring
  ik(g, rng, () => ell(g, 0, -0.06, 0.17, 0.17), '#f8fafc', 0.18, { hatch: 0.2, line: 0.016 });
  // Inner red ring
  ik(g, rng, () => ell(g, 0, -0.06, 0.11, 0.11), '#dc2626', 0.12, { hatch: 0.4, line: 0.016 });
  // Center bullseye dot
  ik(g, rng, () => ell(g, 0, -0.06, 0.04, 0.04), '#ffffff', 0.05, { hatch: 0, line: 0.012 });
}

function steelTargetDraw(g: G, rng: Rng): void {
  // Heavy welded steel A-frame legs
  stroke(g, [[-0.22, 0.24], [0, -0.02]], 0.035, '#334155', 0.95);
  stroke(g, [[0.22, 0.24], [0, -0.02]], 0.035, '#334155', 0.95);
  stroke(g, [[-0.18, 0.14], [0.18, 0.14]], 0.025, '#475569', 0.9);
  // Steel gong disk
  ik(g, rng, () => ell(g, 0, -0.08, 0.22, 0.22), '#64748b', 0.24, { hatch: 0.9, line: 0.024 });
  ik(g, rng, () => ell(g, 0, -0.08, 0.14, 0.14), '#e2e8f0', 0.15, { hatch: 0.3, line: 0.018 });
  // Center hit marker
  ik(g, rng, () => ell(g, 0, -0.08, 0.05, 0.05), '#ea580c', 0.06, { hatch: 0, line: 0.014 });
}

// keep imports used in future drawers
void mix; void smooth;

