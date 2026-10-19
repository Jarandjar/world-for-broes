import type { Rng } from '../engine/rng';
export type G = CanvasRenderingContext2D;
export const INK = '#15171b';
export const TAU = Math.PI * 2;

/** Global look dial: how much user colors are pulled toward a muted ink-and-wash palette. */
export const STYLE = { desat: 0.34 };

export function parse(c: string): [number, number, number] {
  const h = c.startsWith('#') ? c.slice(1) : c;
  return [parseInt(h.slice(0, 2), 16) || 0, parseInt(h.slice(2, 4), 16) || 0, parseInt(h.slice(4, 6), 16) || 0];
}
export function toHex(r: number, g: number, b: number): string {
  const f = (v: number): string => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
  return `#${f(r)}${f(g)}${f(b)}`;
}
export function mix(a: string, b: string, t: number): string {
  const p = parse(a), q = parse(b);
  return toHex(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t);
}
/** amt < 0 darkens toward ink, amt > 0 lightens toward warm paper. */
export function shade(c: string, amt: number): string { return amt < 0 ? mix(c, '#0e1013', -amt) : mix(c, '#efe8d6', amt); }
export function wash(c: string, k = STYLE.desat): string {
  const [r, g, b] = parse(c);
  const l = 0.3 * r + 0.59 * g + 0.11 * b;
  return toHex(r + (l - r) * k - 2, g + (l - g) * k, b + (l - b) * k + 3);
}
export function alpha(c: string, a: number): string { const [r, g, b] = parse(c); return `rgba(${r},${g},${b},${a})`; }

// ---------------------------------------------------------------- path helpers (build paths, do not fill)
export function ell(g: G, cx: number, cy: number, rx: number, ry: number, rot = 0): void {
  g.moveTo(cx + rx * Math.cos(rot), cy + rx * Math.sin(rot));
  g.ellipse(cx, cy, rx, ry, rot, 0, TAU);
}
export function cap(g: G, x1: number, y1: number, x2: number, y2: number, r1: number, r2 = r1): void {
  const a = Math.atan2(y2 - y1, x2 - x1);
  g.moveTo(x1 + Math.cos(a + Math.PI / 2) * r1, y1 + Math.sin(a + Math.PI / 2) * r1);
  g.arc(x1, y1, r1, a + Math.PI / 2, a + Math.PI * 1.5);
  g.lineTo(x2 + Math.cos(a - Math.PI / 2) * r2, y2 + Math.sin(a - Math.PI / 2) * r2);
  g.arc(x2, y2, r2, a - Math.PI / 2, a + Math.PI / 2);
  g.closePath();
}
export function rrect(g: G, x: number, y: number, w: number, h: number, r: number): void {
  r = Math.min(r, w / 2, h / 2);
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
export function poly(g: G, pts: [number, number][], close = true): void {
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  if (close) g.closePath();
}
/** Smooth closed curve through points (hand-drawn feel). */
export function smooth(g: G, pts: [number, number][]): void {
  const n = pts.length;
  const mid = (i: number): [number, number] => { const a = pts[i % n] as [number, number], b = pts[(i + 1) % n] as [number, number]; return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; };
  const m0 = mid(n - 1);
  g.moveTo(m0[0], m0[1]);
  for (let i = 0; i < n; i++) { const p = pts[i] as [number, number], m = mid(i); g.quadraticCurveTo(p[0], p[1], m[0], m[1]); }
  g.closePath();
}
/** Lumpy blob: an ellipse with seeded bumps. */
export function lumpy(g: G, cx: number, cy: number, rx: number, ry: number, rng: Rng, n = 10, amp = 0.12, rot = 0): void {
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU, k = 1 + (rng.next() - 0.5) * 2 * amp;
    const x = Math.cos(a) * rx * k, y = Math.sin(a) * ry * k;
    pts.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
  }
  smooth(g, pts);
}
export function star(g: G, cx: number, cy: number, r0: number, r1: number, n: number, rot: number, rng?: Rng, jit = 0): void {
  const pts: [number, number][] = [];
  for (let i = 0; i < n * 2; i++) {
    const a = rot + (i / (n * 2)) * TAU, r = (i % 2 ? r0 : r1) * (1 + (rng ? (rng.next() - 0.5) * jit : 0));
    pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
  }
  poly(g, pts);
}

// ---------------------------------------------------------------- inked fill: wash + painted volume + hatching + doubled outline
export interface InkOpts { line?: number; hatch?: number; angle?: number; light?: [number, number]; rng?: Rng; noOutline?: boolean; box?: number; noWash?: boolean; grain?: number; wear?: number }

function hatchSide(g: G, light: [number, number], angle: number, density: number, box: number, th: number, a: number, rng?: Rng): void {
  const s = 0.03 / Math.max(0.25, density), ca = Math.cos(angle), sa = Math.sin(angle), len = box * 2;
  const dl = ca * -light[0] + sa * -light[1];       // how fast "shadowness" grows along the line direction
  g.strokeStyle = alpha(INK, a); g.lineWidth = 0.0092; g.lineCap = 'round';
  for (let d = -box * 1.5; d <= box * 1.5; d += s) {
    const px = -sa * d, py = ca * d, base = (px * -light[0] + py * -light[1]) / box;
    let t0 = -len, t1 = len;
    if (Math.abs(dl) < 1e-4) { if (base < th) continue; }
    else { const tc = (th * box - base * box) / dl; if (dl > 0) t0 = Math.max(t0, tc); else t1 = Math.min(t1, tc); }
    if (t0 >= t1) continue;
    const j = rng ? (rng.next() - 0.5) * 0.012 : 0;
    g.beginPath(); g.moveTo(px + ca * t0, py + sa * t0 + j); g.lineTo(px + ca * t1 + j, py + sa * t1); g.stroke();
  }
}


/** Paper-grain stipple and worn marks inside the current clip, so flat fills feel hand-inked. */
function texture(g: G, rng: Rng | undefined, base: string, box: number, grain: number, wear: number): void {
  if (!rng) return;
  const area = (box / 0.3) ** 2, dark = shade(base, -0.45), light = shade(base, 0.35);
  for (let i = 0, n = Math.round(46 * grain * area); i < n; i++) {
    g.fillStyle = alpha(rng.chance(0.62) ? dark : light, rng.range(0.1, 0.3));
    g.beginPath(); g.arc(rng.range(-box, box), rng.range(-box, box), rng.range(0.0035, 0.0085), 0, TAU); g.fill();
  }
  for (let i = 0, n = Math.round(3 * wear * area); i < n; i++) {   // soft smudges
    g.fillStyle = alpha(INK, rng.range(0.05, 0.12)); g.beginPath(); g.ellipse(rng.range(-box, box) * 0.8, rng.range(-box, box) * 0.8, rng.range(0.02, 0.06), rng.range(0.015, 0.04), rng.range(0, 3), 0, TAU); g.fill();
  }
  g.lineCap = 'round';
  for (let i = 0, n = Math.round(7 * wear * area); i < n; i++) {   // scratches
    const x = rng.range(-box, box), y = rng.range(-box, box), a = rng.range(0, TAU), l = rng.range(0.02, 0.07);
    g.strokeStyle = alpha(light, rng.range(0.25, 0.5)); g.lineWidth = 0.006; g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke();
  }
}

export function inked(g: G, path: () => void, fill: string, o: InkOpts = {}): void {
  const line = o.line ?? 0.022, light = o.light ?? [-0.55, -0.8], box = o.box ?? 0.7;
  const base = o.noWash ? fill : wash(fill);
  g.save();
  g.beginPath(); path(); g.fillStyle = base; g.fill();
  g.save(); g.clip();
  const gr = g.createLinearGradient(light[0] * box, light[1] * box, -light[0] * box, -light[1] * box);
  gr.addColorStop(0, alpha(shade(base, 0.22), 0.8)); gr.addColorStop(0.45, 'rgba(0,0,0,0)'); gr.addColorStop(1, alpha(shade(base, -0.5), 0.85));
  g.fillStyle = gr; g.fillRect(-box * 2, -box * 2, box * 4, box * 4);
  const hd = o.hatch ?? 0.7;
  if (hd > 0) {
    const ang = o.angle ?? 0.85;
    hatchSide(g, light, ang, hd, box, 0.12, 0.4, o.rng);
    hatchSide(g, light, ang + 1.25, hd, box, 0.52, 0.32, o.rng);
  }
  texture(g, o.rng, base, box, o.grain ?? 0.7, o.wear ?? 0.5);
  g.restore();
  if (!o.noOutline) {
    g.beginPath(); path();
    g.lineJoin = 'round'; g.lineCap = 'round'; g.strokeStyle = INK; g.lineWidth = line; g.stroke();
    g.save(); g.translate(line * 0.2, -line * 0.15); g.globalAlpha = 0.45; g.lineWidth = line * 0.38; g.stroke(); g.restore();
  }
  g.restore();
}
export function stroke(g: G, pts: [number, number][], w: number, color = INK, a = 1): void {
  g.save(); g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.strokeStyle = color; g.globalAlpha = a; g.lineWidth = w; g.lineCap = 'round'; g.lineJoin = 'round'; g.stroke(); g.restore();
}
