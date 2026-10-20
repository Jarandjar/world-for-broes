/**
 * Smooth ground painter (HD plan G1).
 *
 * Instead of one flat colour per tile, every pixel blends the four nearest tile centres
 * (bilinear weights) after a small noise warp, and takes the ground type with the most
 * weight. Boundaries become soft organic curves, water gets a depth gradient with a foam
 * line and wet sand at the shore, and slopes get gentle top-left lighting.
 * The field is painted at FIELD_RES pixels per tile and scaled up smoothly.
 */
import { makeNoise, fbm } from '../engine/noise';
import { hash2 } from '../engine/rng';
import { CHUNK, WATER_DEEP, WATER_SHALLOW } from '../sim/worldgen';
import type { TerrainDef } from '../data/schema';
import { type G, parse } from './ink';

/** Field pixels per tile. Scaled up to the chunk canvas with smoothing. */
export const FIELD_RES = 16;
/** How far boundaries wander from the tile grid (tiles). */
const WARP = 0.55;
/** Ground colours used for shading and the shore (light, dark, foam). */
const SHADE = { light: '#f4ead0', dark: '#17231b', foam: '#eaf7fb' };
const WATER_IDS = new Set(['shallow', 'deep']);

type RGB = [number, number, number];
interface Kind { def: TerrainDef; base: RGB; dark: RGB; hi: RGB; water: boolean }

export interface FieldInput {
  seed: number;
  cx: number;
  cy: number;
  /** Ground at tile (tx, ty), chunk-local, valid for -1..CHUNK. */
  ground(tx: number, ty: number): TerrainDef;
  /** Height 0..1 at tile (tx, ty), chunk-local, valid for -2..CHUNK+1. */
  height(tx: number, ty: number): number;
  /** Biome tint colour at tile (tx, ty), chunk-local, valid for -1..CHUNK. */
  tint(tx: number, ty: number): string;
  /** Water colours: [shallow highlight, shallow, deep, deep shadow]. */
  water: [string, string, string, string];
}

const noiseCache = new Map<number, { wx: (x: number, y: number) => number; wy: (x: number, y: number) => number; tone: (x: number, y: number) => number }>();
function noisesFor(seed: number): { wx: (x: number, y: number) => number; wy: (x: number, y: number) => number; tone: (x: number, y: number) => number } {
  let n = noiseCache.get(seed);
  if (!n) { n = { wx: makeNoise(seed ^ 0x61c88647), wy: makeNoise(seed ^ 0x2f6b8c1d), tone: makeNoise(seed ^ 0xabc123) }; noiseCache.set(seed, n); }
  return n;
}

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Paints the smooth ground for one chunk into `g` (scaled so 1 unit = 1 tile). */
export function paintGroundField(g: G, inp: FieldInput): void {
  const W = CHUNK + 2; // tiles -1..CHUNK
  const kinds: Kind[] = [];
  const kindOf = new Map<string, number>();
  const tileKind = new Int16Array(W * W);
  const tileTint: RGB[] = new Array(W * W);
  for (let ty = -1; ty <= CHUNK; ty++) for (let tx = -1; tx <= CHUNK; tx++) {
    const d = inp.ground(tx, ty);
    let k = kindOf.get(d.id);
    if (k === undefined) {
      k = kinds.length;
      kinds.push({ def: d, base: parse(d.colors[0]), dark: parse(d.colors[1]), hi: parse(d.colors[2]), water: WATER_IDS.has(d.id) });
      kindOf.set(d.id, k);
    }
    const i = (ty + 1) * W + tx + 1;
    tileKind[i] = k;
    tileTint[i] = parse(inp.tint(tx, ty));
  }
  // Heights and slopes at tile centres (slope from neighbours, for top-left lighting).
  const hgt = new Float32Array(W * W), slope = new Float32Array(W * W);
  for (let ty = -1; ty <= CHUNK; ty++) for (let tx = -1; tx <= CHUNK; tx++) {
    const i = (ty + 1) * W + tx + 1;
    hgt[i] = inp.height(tx, ty);
    slope[i] = (inp.height(tx + 1, ty) - inp.height(tx - 1, ty) + inp.height(tx, ty + 1) - inp.height(tx, ty - 1)) * 0.5;
  }

  const light = parse(SHADE.light), dark = parse(SHADE.dark), foam = parse(SHADE.foam);
  const [wHi, wShallow, wDeep, wAbyss] = inp.water.map(parse) as [RGB, RGB, RGB, RGB];
  const nz = noisesFor(inp.seed);
  const N = CHUNK * FIELD_RES;
  const img = new ImageData(N, N);
  const px = img.data;
  const weight = new Float32Array(kinds.length);
  const isWater = new Float32Array(kinds.map((k) => (k.water ? 1 : 0)));
  // Current pixel colour, mixed in place (no allocations in the pixel loop).
  let cr = 0, cg = 0, cb = 0;
  const set = (c: RGB): void => { cr = c[0]; cg = c[1]; cb = c[2]; };
  const toward = (c: RGB, t: number): void => { cr += (c[0] - cr) * t; cg += (c[1] - cg) * t; cb += (c[2] - cb) * t; };

  // Noise is smooth, so sample it on a coarse grid and interpolate (16x fewer noise calls).
  const STEP = 4, GN = N / STEP + 1;
  const gWarpX = new Float32Array(GN * GN), gWarpY = new Float32Array(GN * GN), gTone = new Float32Array(GN * GN);
  for (let gy = 0; gy < GN; gy++) for (let gx = 0; gx < GN; gx++) {
    const wx = inp.cx * CHUNK + (gx * STEP) / FIELD_RES, wy = inp.cy * CHUNK + (gy * STEP) / FIELD_RES;
    const i = gy * GN + gx;
    gWarpX[i] = (fbm(nz.wx, wx * 0.18, wy * 0.18, 2) - 0.5) * 2 * WARP;
    gWarpY[i] = (fbm(nz.wy, wx * 0.18, wy * 0.18, 2) - 0.5) * 2 * WARP;
    gTone[i] = fbm(nz.tone, wx * 0.14, wy * 0.14, 2) - 0.5;
  }
  const grid = (arr: Float32Array, gx: number, gy: number, fx: number, fy: number): number => {
    const i = gy * GN + gx;
    return lerp(lerp(arr[i]!, arr[i + 1]!, fx), lerp(arr[i + GN]!, arr[i + GN + 1]!, fx), fy);
  };

  for (let py = 0; py < N; py++) {
    const gy = Math.min(GN - 2, Math.floor(py / STEP)), fy = py / STEP - gy;
    for (let pxl = 0; pxl < N; pxl++) {
      const gx = Math.min(GN - 2, Math.floor(pxl / STEP)), fx = pxl / STEP - gx;
      const u = (pxl + 0.5) / FIELD_RES, v = (py + 0.5) / FIELD_RES;
      const wx = inp.cx * CHUNK + u, wy = inp.cy * CHUNK + v;
      // Warp the lookup so tile boundaries become organic curves.
      const su = u + grid(gWarpX, gx, gy, fx, fy) - 0.5;
      const sv = v + grid(gWarpY, gx, gy, fx, fy) - 0.5;
      let ix = Math.floor(su), iy = Math.floor(sv);
      let ax = su - ix, ay = sv - iy;
      if (ix < -1) { ix = -1; ax = 0; } else if (ix > CHUNK - 1) { ix = CHUNK - 1; ax = 1; }
      if (iy < -1) { iy = -1; ay = 0; } else if (iy > CHUNK - 1) { iy = CHUNK - 1; ay = 1; }
      const i00 = (iy + 1) * W + ix + 1, i10 = i00 + 1, i01 = i00 + W, i11 = i01 + 1;
      const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;

      const k00 = tileKind[i00]!, k10 = tileKind[i10]!, k01 = tileKind[i01]!, k11 = tileKind[i11]!;
      weight[k00] = 0; weight[k10] = 0; weight[k01] = 0; weight[k11] = 0;
      weight[k00]! += w00; weight[k10]! += w10; weight[k01]! += w01; weight[k11]! += w11;
      let best = k00;
      if (weight[k10]! > weight[best]!) best = k10;
      if (weight[k01]! > weight[best]!) best = k01;
      if (weight[k11]! > weight[best]!) best = k11;
      const waterW = isWater[k00]! * w00 + isWater[k10]! * w10 + isWater[k01]! * w01 + isWater[k11]! * w11;
      const kind = kinds[best]!;
      const h = hgt[i00]! * w00 + hgt[i10]! * w10 + hgt[i01]! * w01 + hgt[i11]! * w11;

      if (kind.water) {
        // Depth gradient: light at the shore, deep and dark further out.
        const depth = clamp01((WATER_SHALLOW - h) / (WATER_SHALLOW - WATER_DEEP + 0.05));
        if (depth < 0.35) { set(wHi); toward(wShallow, depth / 0.35); }
        else if (depth < 0.7) { set(wShallow); toward(wDeep, (depth - 0.35) / 0.35); }
        else { set(wDeep); toward(wAbyss, (depth - 0.7) / 0.3); }
        // Foam line just inside the water edge.
        const f = clamp01(1 - Math.abs(1 - waterW - 0.4) / 0.11);
        if (f > 0) toward(foam, f * 0.85);
      } else {
        const n = grid(gTone, gx, gy, fx, fy);
        set(kind.base); toward(n > 0 ? kind.hi : kind.dark, Math.min(0.5, Math.abs(n) * 1.6));
        // Wet, darker ground near water.
        if (waterW > 0.04) toward(kind.dark, Math.min(0.55, waterW * 1.3));
        // Gentle relief: slopes rising to the bottom-right face the top-left light.
        const s = slope[i00]! * w00 + slope[i10]! * w10 + slope[i01]! * w01 + slope[i11]! * w11;
        const sh = Math.max(-0.22, Math.min(0.22, s * 7));
        if (sh > 0) toward(light, sh); else if (sh < 0) toward(dark, -sh);
        // Biome mood tint from the nearest tile.
        const ti = (Math.min(CHUNK, Math.max(-1, Math.round(sv))) + 1) * W + Math.min(CHUNK, Math.max(-1, Math.round(su))) + 1;
        toward(tileTint[ti]!, 0.08);
      }
      // Fine paper grain.
      const grain = ((hash2(inp.seed, Math.floor(wx * FIELD_RES), Math.floor(wy * FIELD_RES)) & 15) - 7.5) * 0.9;
      const o = (py * N + pxl) * 4;
      px[o] = cr + grain; px[o + 1] = cg + grain; px[o + 2] = cb + grain; px[o + 3] = 255;
    }
  }

  const off = document.createElement('canvas');
  off.width = off.height = N;
  (off.getContext('2d') as CanvasRenderingContext2D).putImageData(img, 0, 0);
  g.save();
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(off, 0, 0, CHUNK, CHUNK);
  g.restore();
}
