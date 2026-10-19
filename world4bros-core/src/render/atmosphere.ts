import { clamp, smoothstep } from '../engine/math';
import { hashStr, makeRng, rand2 } from '../engine/rng';
import type { Building } from '../sim/worldgen';

/**
 * Weather, atmosphere and screen polish. Render-only: reads sim state, never writes it.
 * Every visual is an entry in ATMOS_EFFECTS (id -> layer, quality tier, draw). Add one with
 * registerAtmosEffect; tunables live in the data tables below (WEATHERS, ATMOS_TUNING).
 * Nothing here touches `document` at import time, so node tests can load it.
 */

// ---------- quality dial ----------
export type Quality = 'off' | 'low' | 'high';
export const QUALITIES: readonly Quality[] = ['off', 'low', 'high'];
export const QUALITY_RANK: Record<Quality, number> = { off: 0, low: 1, high: 2 };
export const atLeast = (q: Quality, min: Quality): boolean => QUALITY_RANK[q] >= QUALITY_RANK[min];

// ---------- data: weather kinds ----------
export interface WeatherParams {
  /** 0..1 each. */
  wet: number; fog: number; clouds: number; wind: number; rain: number; snow: number; leaves: number; gloom: number; dust: number;
}
export interface WeatherDef extends WeatherParams {
  id: string; name: string;
  /** Relative chance when the climate allows it. */
  weight: number;
  /** Biome climate window (temp/moist 0..1) where this weather can roll. */
  climate?: { minTemp?: number; maxTemp?: number; minMoist?: number; maxMoist?: number };
  fogColor?: [number, number, number];
}
const W0: WeatherParams = { wet: 0, fog: 0, clouds: 0, wind: 0, rain: 0, snow: 0, leaves: 0, gloom: 0, dust: 0 };
export const PARAM_KEYS = Object.keys(W0) as (keyof WeatherParams)[];
export const wx = (id: string, name: string, weight: number, p: Partial<WeatherParams>, climate?: WeatherDef['climate'], fogColor?: WeatherDef['fogColor']): WeatherDef =>
  ({ id, name, weight, ...W0, ...p, climate, fogColor });

export const WEATHERS: readonly WeatherDef[] = [
  wx('clear', 'Clear', 5, { clouds: 0.15, wind: 0.15 }),
  wx('breezy', 'Breezy', 2, { clouds: 0.35, wind: 0.75, leaves: 0.7 }, { minTemp: 0.15, maxTemp: 0.85 }),
  wx('overcast', 'Overcast', 2, { clouds: 0.85, gloom: 0.3, wind: 0.3 }),
  wx('rain', 'Rain', 2, { wet: 1, clouds: 1, rain: 0.65, gloom: 0.38, wind: 0.3 }, { minTemp: 0.15, minMoist: 0.35 }),
  wx('storm', 'Storm', 1, { wet: 1, clouds: 1, rain: 1, gloom: 0.55, wind: 1 }, { minTemp: 0.2, minMoist: 0.55 }),
  wx('fog', 'Fog', 1.5, { fog: 0.85, wet: 0.35, clouds: 0.3, gloom: 0.2 }, { minMoist: 0.45 }),
  wx('snow', 'Snow', 4, { snow: 0.8, clouds: 0.9, gloom: 0.15, wind: 0.25 }, { maxTemp: 0.3 }),
  wx('dust', 'Dust haze', 2, { dust: 0.7, fog: 0.35, wind: 0.85 }, { minTemp: 0.75, maxMoist: 0.35 }, [214, 170, 110]),
];
export const weatherById = (id: string): WeatherDef | undefined => WEATHERS.find((w) => w.id === id);

/** Tunables. Effects read these; change numbers here, not inside the effects. */
export const ATMOS_TUNING = {
  weatherBlendSeconds: 8,
  cloudTileUnits: 56, cloudShadowAlpha: [0.07, 0.26] as [number, number],
  rainDrops: { low: 110, high: 260 }, snowFlakes: { low: 90, high: 200 }, leaves: { low: 10, high: 22 },
  fireflies: { low: 12, high: 28 }, motes: 26, splashes: 28,
  wetDarken: 0.2, snowCover: 0.16, gloomTint: [150, 160, 182] as [number, number, number],
  vignette: { base: 0.42, night: 0.2, gloom: 0.15 },
  grain: { tile: 128, alpha: 0.5, jitterHz: 10 },
  windowLight: { color: '#ffcf7a', radius: 2.6, porchColor: '#ffd890', porchRadius: 3.6, lit: 0.65, maxLights: 48 },
  buildingHeight: 2.2,
  leafColors: ['#c9772f', '#d9a441', '#8f5a2b', '#a8b04a'],
};

export interface Climate { temp: number; moist: number }
/** Deterministic weather for a half-day slot in a climate. Pure; same inputs, same weather. */
export function pickWeather(slot: number, climate: Climate, seed = 0): WeatherDef {
  const ok = WEATHERS.filter((w) => {
    const c = w.climate; if (!c) return true;
    return climate.temp >= (c.minTemp ?? 0) && climate.temp <= (c.maxTemp ?? 1) && climate.moist >= (c.minMoist ?? 0) && climate.moist <= (c.maxMoist ?? 1);
  });
  const total = ok.reduce((s, w) => s + w.weight, 0);
  let r = rand2(seed ^ 0x5eed, slot, Math.round(climate.temp * 100) * 101 + Math.round(climate.moist * 100)) * total;
  for (const w of ok) { r -= w.weight; if (r <= 0) return w; }
  return ok[ok.length - 1] ?? WEATHERS[0]!;
}

// ---------- sun ----------
export interface Sun { dx: number; dy: number; len: number; alpha: number; elevation: number }
/** Drop-shadow direction (unit vector), length per unit of height, and strength for a time of day. */
export function sunAt(dayT: number, darkness: number, clouds = 0): Sun {
  const tp = dayT > 0.92 ? dayT - 1 : dayT;
  const p = clamp((tp + 0.06) / 0.68, 0, 1);
  const elevation = Math.sin(p * Math.PI);
  let dx = -Math.cos(p * Math.PI), dy = 0.45;
  if (darkness > 0.5) { dx = 0.35; dy = 0.6; } // moon hangs high in the south
  const m = Math.hypot(dx, dy); dx /= m; dy /= m;
  const len = darkness > 0.5 ? 0.3 : clamp(0.32 / Math.max(elevation, 0.17), 0.32, 1.9);
  const alpha = Math.max(0, 0.28 * (1 - darkness) * (1 - 0.65 * clouds) + 0.08 * darkness);
  return { dx, dy, len, alpha, elevation };
}

// ---------- effect registry ----------
export type AtmosLayer = 'ground' | 'over' | 'screen';
export const ATMOS_LAYERS: readonly AtmosLayer[] = ['ground', 'over', 'screen'];
export interface AtmosFrame {
  g: CanvasRenderingContext2D; W: number; H: number; z: number; camX: number; camY: number;
  /** Half view size in world units (with a margin). */
  hw: number; hh: number;
  t: number; dt: number; dayT: number; darkness: number; day: number; lod: number;
  quality: Quality; w: WeatherParams; fogColor: [number, number, number]; sun: Sun;
  /** Camera movement this frame in screen px, for parallax of screen-space particles. */
  shiftX: number; shiftY: number;
}
export interface AtmosEffect {
  id: string;
  /** ground: after terrain, world transform. over: after roofs, world transform. screen: after lighting, identity transform. */
  layer: AtmosLayer;
  minQuality: Exclude<Quality, 'off'>;
  /** Skip the draw call cheaply when the effect would be invisible. */
  active?: (f: AtmosFrame) => boolean;
  /** `s` is this effect's own scratch state (pools, caches), kept between frames. */
  draw: (f: AtmosFrame, s: Record<string, unknown>) => void;
}
export const ATMOS_EFFECTS: AtmosEffect[] = [];
export function registerAtmosEffect(e: AtmosEffect): void {
  const i = ATMOS_EFFECTS.findIndex((x) => x.id === e.id);
  if (i >= 0) ATMOS_EFFECTS[i] = e; else ATMOS_EFFECTS.push(e);
}
export const effectsFor = (layer: AtmosLayer, q: Quality): AtmosEffect[] =>
  q === 'off' ? [] : ATMOS_EFFECTS.filter((e) => e.layer === layer && atLeast(q, e.minQuality));

// ---------- shared helpers ----------
type C2 = CanvasRenderingContext2D;
export function makeCanvas(w: number, h: number): [HTMLCanvasElement, C2] {
  const c = document.createElement('canvas'); c.width = Math.max(1, Math.ceil(w)); c.height = Math.max(1, Math.ceil(h));
  return [c, c.getContext('2d') as C2];
}
export const toScreen = (g: C2): void => { g.setTransform(1, 0, 0, 1, 0, 0); };
const rgba = (c: [number, number, number], a: number): string => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
/** Get-or-create a value in an effect's scratch state. */
export function memo<T>(s: Record<string, unknown>, key: string, make: () => T): T {
  let v = s[key] as T | undefined; if (v === undefined) { v = make(); s[key] = v; } return v;
}
const texCache = new Map<string, HTMLCanvasElement>();
/** Tileable soft blob texture (clouds, mist). Built once per colour. */
export function cloudTexture(color: string, size = 256): HTMLCanvasElement {
  const key = `${color}@${size}`; const hit = texCache.get(key); if (hit) return hit;
  const [c, g] = makeCanvas(size, size), rng = makeRng(hashStr(key));
  for (let i = 0; i < 16; i++) {
    const x = rng.range(0, size), y = rng.range(0, size), r = rng.range(size * 0.1, size * 0.26), a = rng.range(0.35, 0.7);
    for (const ox of [-size, 0, size]) for (const oy of [-size, 0, size]) {
      const gr = g.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, r);
      gr.addColorStop(0, color.replace('A', String(a))); gr.addColorStop(1, color.replace('A', '0'));
      g.fillStyle = gr; g.fillRect(x + ox - r, y + oy - r, r * 2, r * 2);
    }
  }
  texCache.set(key, c); return c;
}
/** Soft round glow sprite (fireflies, light pools). */
export function glowSprite(rgb: [number, number, number], size = 64): HTMLCanvasElement {
  const key = `glow:${rgb.join(',')}@${size}`; const hit = texCache.get(key); if (hit) return hit;
  const [c, g] = makeCanvas(size, size), h = size / 2, gr = g.createRadialGradient(h, h, 0, h, h, h);
  gr.addColorStop(0, rgba(rgb, 1)); gr.addColorStop(0.25, rgba(rgb, 0.55)); gr.addColorStop(1, rgba(rgb, 0));
  g.fillStyle = gr; g.fillRect(0, 0, size, size); texCache.set(key, c); return c;
}
/** Draw a texture as a world-anchored repeating pattern drifting by (ox,oy) world units. */
export function worldPattern(f: AtmosFrame, tex: HTMLCanvasElement, tileUnits: number, ox: number, oy: number, alpha: number, s: Record<string, unknown>, key: string): void {
  const { g } = f; const pat = memo(s, key, () => g.createPattern(tex, 'repeat'));
  if (!pat) return;
  const k = tileUnits / tex.width, mx = ((ox % tileUnits) + tileUnits) % tileUnits, my = ((oy % tileUnits) + tileUnits) % tileUnits;
  g.save(); g.globalAlpha = alpha; g.translate(mx, my); g.scale(k, k); g.fillStyle = pat;
  g.fillRect((f.camX - f.hw - mx) / k, (f.camY - f.hh - my) / k, (f.hw * 2) / k, (f.hh * 2) / k);
  g.restore();
}
/** A pool of screen-space particles in normalized [0,1) coordinates. */
export interface Pool { x: Float32Array; y: Float32Array; v: Float32Array; p: Float32Array }
export function pool(s: Record<string, unknown>, key: string, n: number, seed: number): Pool {
  const old = s[key] as Pool | undefined;
  if (old && old.x.length === n) return old;
  const rng = makeRng(seed), P: Pool = { x: new Float32Array(n), y: new Float32Array(n), v: new Float32Array(n), p: new Float32Array(n) };
  for (let i = 0; i < n; i++) { P.x[i] = rng.next(); P.y[i] = rng.next(); P.v[i] = rng.range(0.7, 1.3); P.p[i] = rng.next() * 6.28; }
  s[key] = P; return P;
}
const wrap01 = (v: number): number => v - Math.floor(v);
/** Scales particle counts down when zoomed far out (lod 0) and by quality. */
const count = (f: AtmosFrame, tiers: { low: number; high: number }, amount: number): number =>
  Math.round((f.quality === 'high' ? tiers.high : tiers.low) * clamp(amount, 0, 1) * (f.lod === 0 ? 0.6 : 1));

// ---------- the effects ----------
// ground layer
registerAtmosEffect({
  id: 'wet-ground', layer: 'ground', minQuality: 'low',
  active: (f) => f.w.wet > 0.02 || f.w.snow > 0.02,
  draw: (f) => {
    const { g, W, H, w } = f;
    g.save(); toScreen(g);
    if (w.wet > 0.02) { g.globalCompositeOperation = 'multiply'; g.fillStyle = `rgba(70,84,104,${ATMOS_TUNING.wetDarken * w.wet * 1.6})`; g.fillRect(0, 0, W, H); }
    if (w.snow > 0.02) { g.globalCompositeOperation = 'source-over'; g.fillStyle = `rgba(236,242,250,${ATMOS_TUNING.snowCover * w.snow})`; g.fillRect(0, 0, W, H); }
    g.restore();
  },
});
registerAtmosEffect({
  id: 'puddles', layer: 'ground', minQuality: 'high',
  active: (f) => f.w.wet > 0.3 && f.lod > 0,
  draw: (f) => {
    const { g, w, t } = f, cell = 3.2, a = (w.wet - 0.3) / 0.7;
    const x0 = Math.floor((f.camX - f.hw) / cell), x1 = Math.ceil((f.camX + f.hw) / cell), y0 = Math.floor((f.camY - f.hh) / cell), y1 = Math.ceil((f.camY + f.hh) / cell);
    g.save(); g.lineWidth = 0.035;
    g.fillStyle = `rgba(52,66,88,${0.3 * a})`; g.beginPath();
    const rings: [number, number, number][] = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (rand2(91, x, y) > 0.16) continue;
      const px = (x + rand2(92, x, y)) * cell, py = (y + rand2(93, x, y)) * cell, rx = 0.35 + rand2(94, x, y) * 0.7;
      g.moveTo(px + rx, py); g.ellipse(px, py, rx, rx * 0.55, 0, 0, Math.PI * 2);
      rings.push([px, py, rx]);
    }
    g.fill();
    g.strokeStyle = `rgba(205,222,240,${0.35 * a})`; g.beginPath();
    for (const [px, py, rx] of rings) { g.moveTo(px - rx * 0.5, py - rx * 0.3); g.ellipse(px, py, rx * 0.75, rx * 0.4, 0, Math.PI * 1.1, Math.PI * 1.6); }
    g.stroke();
    if (w.rain > 0.1) { // rain rings
      g.strokeStyle = `rgba(220,232,245,${0.4 * w.rain})`; g.beginPath();
      for (let i = 0; i < rings.length; i++) {
        const [px, py, rx] = rings[i]!, ph = wrap01(t * 1.3 + i * 0.37), rr = rx * 0.6 * ph;
        g.moveTo(px + rr, py); g.ellipse(px, py, rr, rr * 0.55, 0, 0, Math.PI * 2);
      }
      g.stroke();
    }
    g.restore();
  },
});

// over layer (world transform, after roofs)
registerAtmosEffect({
  id: 'cloud-shadows', layer: 'over', minQuality: 'low',
  active: (f) => f.darkness < 0.8,
  draw: (f, s) => {
    const [a0, a1] = ATMOS_TUNING.cloudShadowAlpha, a = (a0 + (a1 - a0) * f.w.clouds) * (1 - f.darkness);
    const speed = 0.6 + f.w.wind * 2.4;
    worldPattern(f, cloudTexture('rgba(18,24,40,A)'), ATMOS_TUNING.cloudTileUnits, f.t * speed, f.t * speed * 0.35, a, s, 'pat');
  },
});
registerAtmosEffect({
  id: 'dust-motes', layer: 'over', minQuality: 'high',
  active: (f) => f.darkness < 0.35 && f.w.rain < 0.1 && f.w.snow < 0.1 && f.lod > 0,
  draw: (f, s) => {
    const { g, t } = f, P = pool(s, 'p', ATMOS_TUNING.motes, 5), span = Math.max(f.hw, f.hh) * 2;
    g.save(); g.fillStyle = 'rgba(255,244,200,0.5)'; g.globalAlpha = 0.55 * (1 - f.darkness / 0.35) * (1 - f.w.clouds * 0.6); g.beginPath();
    for (let i = 0; i < P.x.length; i++) {
      const x = f.camX - f.hw + wrap01(P.x[i]! + t * 0.006 * P.v[i]! + Math.sin(t * 0.4 + P.p[i]!) * 0.004) * span;
      const y = f.camY - f.hh + wrap01(P.y[i]! - t * 0.004 * P.v[i]!) * span, r = 0.025 + 0.02 * Math.sin(t * 2 + P.p[i]!) ** 2;
      g.moveTo(x + r, y); g.arc(x, y, r, 0, Math.PI * 2);
    }
    g.fill(); g.restore();
  },
});
registerAtmosEffect({
  id: 'mist', layer: 'over', minQuality: 'low',
  active: (f) => f.w.fog > 0.03,
  draw: (f, s) => {
    const { g, W, H, w } = f, c = f.fogColor;
    g.save(); toScreen(g); g.fillStyle = rgba(c, 0.32 * w.fog); g.fillRect(0, 0, W, H); g.restore();
    if (f.quality === 'high') {
      const tex = cloudTexture(`rgba(${c[0]},${c[1]},${c[2]},A)`, 256), drift = f.t * (0.5 + w.wind * 2);
      worldPattern(f, tex, 38, drift, drift * 0.2, 0.55 * w.fog, s, 'a');
      worldPattern(f, tex, 23, -drift * 1.6, drift * 0.5, 0.35 * w.fog, s, 'b');
    }
  },
});
registerAtmosEffect({
  id: 'weather-grade', layer: 'over', minQuality: 'low',
  active: (f) => f.w.gloom > 0.02 || f.w.dust > 0.02,
  draw: (f) => {
    const { g, W, H, w } = f;
    g.save(); toScreen(g); g.globalCompositeOperation = 'multiply';
    if (w.gloom > 0.02) { g.fillStyle = rgba(ATMOS_TUNING.gloomTint, w.gloom * 0.55); g.fillRect(0, 0, W, H); }
    if (w.dust > 0.02) { g.fillStyle = `rgba(240,200,140,${w.dust * 0.35})`; g.fillRect(0, 0, W, H); }
    g.restore();
  },
});
registerAtmosEffect({
  id: 'rain', layer: 'over', minQuality: 'low',
  active: (f) => f.w.rain > 0.02,
  draw: (f, s) => {
    const { g, W, H, w, dt } = f, n = count(f, ATMOS_TUNING.rainDrops, w.rain), P = pool(s, `p${n}`, n, 11);
    const slant = 0.18 + w.wind * 0.45, len = (H / 1000) * (16 + w.rain * 10), fall = 1.6 + w.rain * 0.8;
    g.save(); toScreen(g); g.strokeStyle = `rgba(196,214,236,${0.32 + 0.25 * w.rain})`; g.lineWidth = Math.max(1, H / 900); g.lineCap = 'round'; g.beginPath();
    for (let i = 0; i < n; i++) {
      P.y[i] = wrap01(P.y[i]! + dt * fall * P.v[i]! - f.shiftY / H); P.x[i] = wrap01(P.x[i]! + dt * fall * slant * P.v[i]! * (H / W) - f.shiftX / W);
      const x = P.x[i]! * W, y = P.y[i]! * H, l = len * P.v[i]!;
      g.moveTo(x, y); g.lineTo(x - slant * l, y - l);
    }
    g.stroke();
    if (f.quality === 'high') { // splashes
      const S = pool(s, 'splash', ATMOS_TUNING.splashes, 12);
      g.strokeStyle = `rgba(214,228,244,${0.35 * w.rain})`; g.lineWidth = Math.max(1, H / 1200); g.beginPath();
      for (let i = 0; i < S.x.length; i++) {
        S.p[i] = S.p[i]! + dt * 3.2; if (S.p[i]! > 1) { S.p[i] = 0; S.x[i] = Math.random(); S.y[i] = Math.random(); }
        const r = (H / 1000) * 7 * S.p[i]!, x = S.x[i]! * W, y = S.y[i]! * H;
        g.moveTo(x + r, y); g.ellipse(x, y, r, r * 0.45, 0, 0, Math.PI * 2);
      }
      g.stroke();
    }
    g.restore();
  },
});
registerAtmosEffect({
  id: 'snow', layer: 'over', minQuality: 'low',
  active: (f) => f.w.snow > 0.02,
  draw: (f, s) => {
    const { g, W, H, w, dt, t } = f, n = count(f, ATMOS_TUNING.snowFlakes, w.snow), P = pool(s, `p${n}`, n, 13), u = H / 1000;
    g.save(); toScreen(g); g.fillStyle = 'rgba(246,249,255,0.9)'; g.beginPath();
    for (let i = 0; i < n; i++) {
      P.y[i] = wrap01(P.y[i]! + dt * 0.09 * P.v[i]! - f.shiftY / H);
      P.x[i] = wrap01(P.x[i]! + dt * (0.02 + w.wind * 0.06) * (H / W) - f.shiftX / W);
      const x = P.x[i]! * W + Math.sin(t * 1.3 + P.p[i]!) * 9 * u, y = P.y[i]! * H, r = (1.2 + P.v[i]! * 1.6) * u;
      g.moveTo(x + r, y); g.arc(x, y, r, 0, Math.PI * 2);
    }
    g.fill(); g.restore();
  },
});
registerAtmosEffect({
  id: 'falling-leaves', layer: 'over', minQuality: 'low',
  active: (f) => f.w.leaves > 0.05 && f.w.snow < 0.3,
  draw: (f, s) => {
    const { g, W, H, w, dt, t } = f, n = count(f, ATMOS_TUNING.leaves, w.leaves), P = pool(s, `p${n}`, n, 17), u = H / 1000, cols = ATMOS_TUNING.leafColors;
    g.save(); toScreen(g); g.strokeStyle = 'rgba(46,24,16,0.6)'; g.lineWidth = Math.max(1, u);
    for (let i = 0; i < n; i++) {
      P.x[i] = wrap01(P.x[i]! + dt * (0.04 + w.wind * 0.1) * P.v[i]! - f.shiftX / W);
      P.y[i] = wrap01(P.y[i]! + dt * 0.035 * P.v[i]! - f.shiftY / H);
      const x = P.x[i]! * W, y = P.y[i]! * H + Math.sin(t * 2 + P.p[i]!) * 12 * u, sz = (6 + P.v[i]! * 3) * u;
      g.save(); g.translate(x, y); g.rotate(t * 1.7 * P.v[i]! + P.p[i]!); g.scale(1, 0.55 + 0.45 * Math.sin(t * 3 + P.p[i]!));
      g.fillStyle = cols[i % cols.length]!; g.beginPath(); g.ellipse(0, 0, sz, sz * 0.5, 0, 0, Math.PI * 2); g.fill(); g.stroke();
      g.restore();
    }
    g.restore();
  },
});

// screen layer (after lighting)
registerAtmosEffect({
  id: 'fireflies', layer: 'screen', minQuality: 'low',
  active: (f) => f.darkness > 0.35 && f.w.rain < 0.2 && f.w.snow < 0.2,
  draw: (f, s) => {
    const { g, t, z } = f, n = count(f, ATMOS_TUNING.fireflies, 1), P = pool(s, `p${n}`, n, 23), span = Math.max(f.hw, f.hh) * 1.6;
    const spr = glowSprite([214, 255, 140], 48), a = smoothstep(0.35, 0.7, f.darkness), r = Math.max(6, z * 0.22);
    g.save(); toScreen(g); g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < n; i++) {
      const wx = f.camX - span / 2 + wrap01(P.x[i]! + Math.sin(t * 0.21 * P.v[i]! + P.p[i]!) * 0.03 + t * 0.002) * span;
      const wy = f.camY - span / 2 + wrap01(P.y[i]! + Math.cos(t * 0.17 * P.v[i]! + P.p[i]!) * 0.03) * span;
      const blink = Math.max(0, Math.sin(t * 1.4 * P.v[i]! + P.p[i]! * 3)) ** 2;
      if (blink < 0.05) continue;
      g.globalAlpha = a * blink;
      g.drawImage(spr, f.W / 2 + (wx - f.camX) * z - r, f.H / 2 + (wy - f.camY) * z - r, r * 2, r * 2);
    }
    g.restore();
  },
});
registerAtmosEffect({
  id: 'lightning', layer: 'screen', minQuality: 'high',
  active: (f) => f.w.rain > 0.8 && f.w.wind > 0.7,
  draw: (f, s) => {
    const next = memo(s, 'next', () => ({ at: f.t + 6, flash: 0 }));
    if (f.t > next.at) { next.flash = 1; next.at = f.t + 7 + Math.random() * 9; }
    if (next.flash <= 0) return;
    next.flash = Math.max(0, next.flash - f.dt * 4);
    const { g } = f; g.save(); toScreen(g); g.fillStyle = `rgba(225,232,255,${0.16 * next.flash})`; g.fillRect(0, 0, f.W, f.H); g.restore();
  },
});
registerAtmosEffect({
  id: 'vignette', layer: 'screen', minQuality: 'low',
  draw: (f, s) => {
    const { g, W, H } = f, V = ATMOS_TUNING.vignette, key = `${W}x${H}`;
    const cv = memo(s, 'cv', () => ({ key: '', c: null as HTMLCanvasElement | null }));
    if (cv.key !== key) {
      const [c, cg] = makeCanvas(W / 4, H / 4), cw = c.width, ch = c.height;
      const gr = cg.createRadialGradient(cw / 2, ch / 2, Math.min(cw, ch) * 0.32, cw / 2, ch / 2, Math.max(cw, ch) * 0.75);
      gr.addColorStop(0, 'rgba(22,15,36,0)'); gr.addColorStop(0.6, 'rgba(22,15,36,0.35)'); gr.addColorStop(1, 'rgba(22,15,36,1)');
      cg.fillStyle = gr; cg.fillRect(0, 0, cw, ch); cv.key = key; cv.c = c;
    }
    g.save(); toScreen(g); g.globalAlpha = clamp(V.base + V.night * f.darkness + V.gloom * f.w.gloom, 0, 0.9); g.imageSmoothingEnabled = true;
    g.drawImage(cv.c!, 0, 0, W, H); g.restore();
  },
});
registerAtmosEffect({
  id: 'paper-grain', layer: 'screen', minQuality: 'low',
  draw: (f, s) => {
    const { g, W, H } = f, G = ATMOS_TUNING.grain;
    const pat = memo(s, 'pat', () => {
      const [c, cg] = makeCanvas(G.tile, G.tile), rng = makeRng(7);
      cg.fillStyle = 'rgb(128,128,128)'; cg.fillRect(0, 0, G.tile, G.tile);
      for (let i = 0; i < 1400; i++) { const v = rng.chance(0.5) ? rng.int(70, 110) : rng.int(150, 190); cg.fillStyle = `rgb(${v},${v},${v})`; cg.fillRect(rng.range(0, G.tile), rng.range(0, G.tile), 1, 1); }
      cg.lineWidth = 0.6; // cardstock fibres
      for (let i = 0; i < 90; i++) { const v = rng.chance(0.5) ? 96 : 168, x = rng.range(0, G.tile), y = rng.range(0, G.tile), a = rng.range(0, 6.28), l = rng.range(3, 9); cg.strokeStyle = `rgb(${v},${v},${v})`; cg.beginPath(); cg.moveTo(x, y); cg.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); cg.stroke(); }
      return g.createPattern(c, 'repeat');
    });
    if (!pat) return;
    const step = f.quality === 'high' ? Math.floor(f.t * G.jitterHz) % 3 : 0; // hand-drawn 3-sample jitter
    g.save(); toScreen(g); g.globalCompositeOperation = 'soft-light'; g.globalAlpha = G.alpha; g.fillStyle = pat;
    g.translate(step * 41, step * 29); g.fillRect(-128, -128, W + 256, H + 256); g.restore();
  },
});

// ---------- building drop shadows and night window pools ----------
/** Footprint swept along the sun: a soft drop shadow for a box of height h. World transform. */
export function drawBoxShadow(g: C2, x: number, y: number, w: number, h: number, height: number, sun: Sun): void {
  if (sun.alpha < 0.01) return;
  const sx = sun.dx * sun.len * height, sy = sun.dy * sun.len * height;
  const pts: [number, number][] = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  g.save(); g.fillStyle = `rgba(21,18,30,${sun.alpha * 0.9})`; g.beginPath();
  for (let i = 0; i < 4; i++) { // each edge swept into a quad, wound consistently so the union fills once
    const [ax, ay] = pts[i]!, [bx, by] = pts[(i + 1) % 4]!;
    const cross = (bx - ax) * sy - (by - ay) * sx;
    if (cross >= 0) { g.moveTo(ax, ay); g.lineTo(bx, by); g.lineTo(bx + sx, by + sy); g.lineTo(ax + sx, ay + sy); }
    else { g.moveTo(ax, ay); g.lineTo(ax + sx, ay + sy); g.lineTo(bx + sx, by + sy); g.lineTo(bx, by); }
    g.closePath();
  }
  g.rect(x + sx, y + sy, w, h);
  g.fill('nonzero'); g.restore();
}
export interface PointLight { x: number; y: number; radius: number; color: string; flicker: number }
/** Warm porch and window pools for buildings at night. Deterministic per building and day. */
export function buildingLights(bs: readonly Building[], darkness: number, day: number, out: PointLight[] = []): PointLight[] {
  if (darkness < 0.25) return out;
  const T = ATMOS_TUNING.windowLight, k = smoothstep(0.25, 0.6, darkness);
  for (const b of bs) {
    if (out.length >= T.maxLights) break;
    const seed = hashStr(b.id), horiz = b.door === 'N' || b.door === 'S';
    const dx = horiz ? b.x + b.doorAt : b.door === 'W' ? b.x - 0.5 : b.x + b.w + 0.5;
    const dy = horiz ? (b.door === 'N' ? b.y - 0.5 : b.y + b.h + 0.5) : b.y + b.doorAt;
    out.push({ x: dx, y: dy, radius: T.porchRadius * k, color: T.porchColor, flicker: 0.03 });
    const len = horiz ? b.w : b.h;
    for (let p = 1.2, i = 0; p < len - 1.0 && out.length < T.maxLights; p += 2.2, i++) {
      if (Math.abs(p - b.doorAt) < 1.1 || rand2(seed, i, day) > T.lit) continue;
      const lx = horiz ? b.x + p + 0.45 : dx, ly = horiz ? dy : b.y + p + 0.45;
      out.push({ x: lx, y: ly, radius: T.radius * k, color: T.color, flicker: 0.015 });
    }
  }
  return out;
}

// ---------- the runtime ----------
export interface AtmosInput {
  W: number; H: number; z: number; camX: number; camY: number; dayT: number; darkness: number; day: number;
  lod: number; climate: Climate; seed: number;
}
/** Settings from ?gfx=off|low|high and ?weather=<id>, remembered in localStorage. */
export function loadGfx(): { quality: Quality; weather: string | null } {
  let quality: Quality = 'high', weather: string | null = null;
  try {
    const qs = new URLSearchParams(location.search), saved = localStorage.getItem('w4b.gfx');
    const q = qs.get('gfx') ?? saved;
    if (q && (QUALITIES as readonly string[]).includes(q)) quality = q as Quality;
    if (qs.has('gfx')) localStorage.setItem('w4b.gfx', quality);
    weather = qs.get('weather');
  } catch { /* storage blocked: defaults */ }
  return { quality, weather };
}

export class Atmosphere {
  quality: Quality = 'high';
  /** Force a weather id (dev / screenshots); null = follow the climate roll. */
  forcedWeather: string | null = null;
  readonly w: WeatherParams = { ...W0 };
  weather: WeatherDef = WEATHERS[0]!;
  sun: Sun = sunAt(0.25, 0);
  private fogColor: [number, number, number] = [206, 214, 224];
  private scratch = new Map<string, Record<string, unknown>>();
  private frame: AtmosFrame | null = null;
  private t = 0; private lastCam: [number, number] | null = null; private first = true;

  constructor() {
    const s = loadGfx(); this.quality = s.quality; this.forcedWeather = s.weather;
    try { (window as unknown as { __atmos: Atmosphere }).__atmos = this; } catch { /* node */ }
  }
  setQuality(q: Quality): void { this.quality = q; try { localStorage.setItem('w4b.gfx', q); } catch { /* ignore */ } }

  /** Advance weather blending and build this frame's shared parameters. */
  update(g: C2, dt: number, i: AtmosInput): void {
    this.t += dt;
    const target = (this.forcedWeather && weatherById(this.forcedWeather)) || pickWeather(i.day * 2 + (i.dayT >= 0.5 ? 1 : 0), i.climate, i.seed);
    this.weather = target;
    const k = this.first ? 1 : 1 - Math.exp(-dt * 3 / ATMOS_TUNING.weatherBlendSeconds); this.first = false;
    for (const key of PARAM_KEYS) this.w[key] += (target[key] - this.w[key]) * k;
    const mist = (1 - smoothstep(0.0, 0.12, i.dayT > 0.92 ? 0 : i.dayT)) * smoothstep(0.4, 0.8, i.climate.moist) * 0.35; // morning mist
    const fc = target.fogColor ?? [206, 214, 224];
    for (let c = 0; c < 3; c++) this.fogColor[c] = this.fogColor[c]! + (fc[c]! - this.fogColor[c]!) * k;
    this.sun = sunAt(i.dayT, i.darkness, this.w.clouds);
    const shiftX = this.lastCam ? (i.camX - this.lastCam[0]) * i.z : 0, shiftY = this.lastCam ? (i.camY - this.lastCam[1]) * i.z : 0;
    this.lastCam = [i.camX, i.camY];
    this.frame = {
      g, W: i.W, H: i.H, z: i.z, camX: i.camX, camY: i.camY, hw: i.W / i.z / 2 + 2, hh: i.H / i.z / 2 + 2,
      t: this.t, dt, dayT: i.dayT, darkness: i.darkness, day: i.day, lod: i.lod, quality: this.quality,
      w: { ...this.w, fog: Math.max(this.w.fog, mist) }, fogColor: [Math.round(this.fogColor[0]), Math.round(this.fogColor[1]), Math.round(this.fogColor[2])],
      sun: this.sun, shiftX: Math.abs(shiftX) > i.W ? 0 : shiftX, shiftY: Math.abs(shiftY) > i.H ? 0 : shiftY,
    };
  }
  /** Wind 0..1 for foliage sway. */
  get wind(): number { return this.w.wind; }
  draw(layer: AtmosLayer): void {
    const f = this.frame; if (!f) return;
    for (const e of effectsFor(layer, this.quality)) {
      if (e.active && !e.active(f)) continue;
      let s = this.scratch.get(e.id); if (!s) { s = {}; this.scratch.set(e.id, s); }
      e.draw(f, s);
    }
  }
}
