import { CHUNK, type Building, type ChunkData, type Side, type WorldGen, type WorldObject } from './worldgen';
import type { ObjectKind, Terrain } from '../data/schema';

/** Half-size of the static map in chunks: it covers chunks [-R, R-1] on both axes (R=10 -> 320x320 tiles). */
export const STATIC_RADIUS = 10;

export interface StaticMapSave {
  v: 1;
  radius: number;
  /** Terrain ids in the order used by the saved indices (so saves survive content reordering). */
  order: string[];
  nextId: number;
  chunks: {
    cx: number; cy: number;
    /** One char per tile: char code - 48 = index. */
    terrain: string; biome: string; elevation?: string;
    objects: [ObjectKind, number, number, number, number][];
    features: [string, number, number, string?][];
    buildings?: [string, number, number, number, number, Side, number][];
  }[];
}

const enc = (a: Uint8Array): string => { let s = ''; for (const v of a) s += String.fromCharCode(48 + v); return s; };
const dec = (s: string, out: Uint8Array): void => { for (let i = 0; i < out.length; i++) out[i] = Math.max(0, (s.charCodeAt(i) || 48) - 48); };
const key = (cx: number, cy: number): string => `${cx},${cy}`;

/**
 * A finite world frozen from the generator so towns and wilderness can be hand-edited.
 * Pure data: no rendering. Everything outside the bounds reads as deep water.
 */
export class StaticMap {
  readonly chunks = new Map<string, ChunkData>();
  private voids = new Map<string, ChunkData>();
  nextId = 1;

  constructor(readonly radius: number = STATIC_RADIUS) {}

  inBounds(cx: number, cy: number): boolean { return cx >= -this.radius && cy >= -this.radius && cx < this.radius && cy < this.radius; }

  chunk(cx: number, cy: number, deepIdx: number): ChunkData {
    const c = this.chunks.get(key(cx, cy));
    if (c) return c;
    let v = this.voids.get(key(cx, cy));
    if (!v) {
      v = { cx, cy, terrain: new Uint8Array(CHUNK * CHUNK).fill(deepIdx), biome: new Uint8Array(CHUNK * CHUNK), elevation: new Uint8Array(CHUNK * CHUNK), objects: [], features: [], buildings: [] };
      this.voids.set(key(cx, cy), v);
    }
    return v;
  }

  /** Freeze the generator output (terrain, scatter, roads, towns, camps...) inside the bounds. */
  static bake(gen: WorldGen, radius: number = STATIC_RADIUS): StaticMap {
    const m = new StaticMap(radius);
    for (let cy = -radius; cy < radius; cy++) for (let cx = -radius; cx < radius; cx++) {
      const c = gen.chunk(cx, cy);
      c.objects = c.objects.filter((o) => !o.removed);
      m.chunks.set(key(cx, cy), c);
    }
    return m;
  }

  counts(): { objects: number; features: number } {
    let objects = 0, features = 0;
    for (const c of this.chunks.values()) { objects += c.objects.length; features += c.features.length; }
    return { objects, features };
  }

  toJSON(order: Terrain[]): StaticMapSave {
    const chunks: StaticMapSave['chunks'] = [];
    for (const c of this.chunks.values()) {
      chunks.push({
        cx: c.cx, cy: c.cy, terrain: enc(c.terrain), biome: enc(c.biome), elevation: enc(c.elevation),
        objects: c.objects.filter((o) => !o.removed).map((o) => [o.kind, Math.round(o.x * 100), Math.round(o.y * 100), o.variant, o.hp] as [ObjectKind, number, number, number, number]),
        features: c.features.map((f) => (f.name === undefined ? [f.type, f.x, f.y] : [f.type, f.x, f.y, f.name]) as [string, number, number, string?]),
        buildings: c.buildings.map((b) => [b.def, b.x, b.y, b.w, b.h, b.door, b.doorAt] as [string, number, number, number, number, Side, number]),
      });
    }
    return { v: 1, radius: this.radius, order: [...order], nextId: this.nextId, chunks };
  }

  static fromJSON(data: StaticMapSave, order: Terrain[], sampleHeight?: (x: number, y: number) => number): StaticMap {
    const m = new StaticMap(data.radius);
    m.nextId = data.nextId;
    const remap = data.order.map((id) => Math.max(0, order.indexOf(id as Terrain)));
    const same = remap.every((v, i) => v === i);
    for (const s of data.chunks) {
      const c: ChunkData = { cx: s.cx, cy: s.cy, terrain: new Uint8Array(CHUNK * CHUNK), biome: new Uint8Array(CHUNK * CHUNK), elevation: new Uint8Array(CHUNK * CHUNK), objects: [], features: [], buildings: [] };
      dec(s.terrain, c.terrain); dec(s.biome, c.biome);
      if (s.elevation) dec(s.elevation, c.elevation);
      else if (sampleHeight) {
        for (let ty = 0; ty < CHUNK; ty++) for (let tx = 0; tx < CHUNK; tx++) {
          const x = s.cx * CHUNK + tx + 0.5, y = s.cy * CHUNK + ty + 0.5;
          c.elevation[ty * CHUNK + tx] = Math.round(Math.max(0, Math.min(1, sampleHeight(x, y))) * 255);
        }
      }
      if (!same) for (let i = 0; i < c.terrain.length; i++) c.terrain[i] = remap[c.terrain[i] ?? 0] ?? 0;
      c.objects = s.objects.map(([kind, x, y, variant, hp], i): WorldObject => ({ id: `${s.cx},${s.cy}:${i}`, kind, x: x / 100, y: y / 100, variant, hp }));
      c.features = s.features.map(([type, x, y, name]) => (name === undefined ? { type, x, y } : { type, x, y, name }));
      c.buildings = (s.buildings ?? []).map(([def, x, y, w, h, door, doorAt], i): Building => ({ id: `${s.cx},${s.cy}:bld${i}`, def, x, y, w, h, door, doorAt }));
      m.chunks.set(key(s.cx, s.cy), c);
    }
    return m;
  }
}
