import { makeNoise, fbm } from '../engine/noise';
import { makeRng, rand2, hash2, type Rng } from '../engine/rng';
import { clamp } from '../engine/math';
import type { BiomeDef, Content, ObjectKind, Terrain } from '../data/schema';
import { buildRoadNet, roadHit, roadNet, roadsInBox, type RoadNode, type RoadPath } from './roads';
export type { RoadPath } from './roads';

export const CHUNK = 16;
export const POWER_POLE_SPACING = 6;

export interface WorldObject { id: string; kind: ObjectKind; x: number; y: number; variant: number; hp: number; removed?: boolean }
export interface Feature { type: string; x: number; y: number; name?: string }
export type Side = 'N' | 'E' | 'S' | 'W';
export type BuildingPieceKind = 'wall' | 'door' | 'floor';
/** One snap-kit piece expressed in world tile units. */
export interface BuildingPiece {
  kind: BuildingPieceKind;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  exterior?: boolean;
}
/** A walled building. (x,y) is the top-left tile, w/h are outer size in tiles. `doorAt` is the door centre measured along the door's wall. */
/** A placed building. `yard`/`front` (town lots only): free tiles behind the back wall and from the front wall to the street edge. */
export interface Building { id: string; def: string; x: number; y: number; w: number; h: number; door: Side; doorAt: number; yard?: number; front?: number; pieces?: BuildingPiece[] }
export interface ChunkData { cx: number; cy: number; terrain: Uint8Array; biome: Uint8Array; elevation: Uint8Array; objects: WorldObject[]; features: Feature[]; buildings: Building[] }
export interface Sample { height: number; temp: number; moist: number; biome: number; terrain: Terrain }

export interface GenContext {
  seed: number; content: Content; terrainOrder: Terrain[];
  sample(x: number, y: number): Sample;
  rng(cx: number, cy: number, label: string): Rng;
  spawn: { x: number; y: number };
}
/** A generation stage fills or modifies a chunk. Add stages (roads, settlements, ...) without touching existing ones. */
export interface GenStage { name: string; run(ctx: GenContext, chunk: ChunkData): void }

export const WATER_DEEP = 0.17, WATER_SHALLOW = 0.215;
/** Climate noise frequency (lower = bigger biomes), how much height sways the biome pick, and the size of in-biome ground patches. */
const BIOME_FREQ = 0.0024, BIOME_ELEV_WEIGHT = 0.35, PATCH_FREQ = 0.045;

/** Stage 1: terrain and biome per tile. */
export const terrainStage: GenStage = {
  name: 'terrain',
  run(ctx, chunk) {
    for (let ty = 0; ty < CHUNK; ty++) for (let tx = 0; tx < CHUNK; tx++) {
      const s = ctx.sample(chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5);
      const i = ty * CHUNK + tx;
      chunk.terrain[i] = ctx.terrainOrder.indexOf(s.terrain);
      chunk.biome[i] = s.biome;
      chunk.elevation[i] = Math.round(s.height * 255);
    }
  },
};

/** Stage 2: trees, rocks, bushes, scattered by biome density rules. */
export const scatterStage: GenStage = {
  name: 'scatter',
  run(ctx, chunk) {
    const rng = ctx.rng(chunk.cx, chunk.cy, 'scatter');
    let n = 0;
    for (let ty = 0; ty < CHUNK; ty++) for (let tx = 0; tx < CHUNK; tx++) {
      const biome = ctx.content.biomes[chunk.biome[ty * CHUNK + tx] ?? 0];
      if (!biome) continue;
      const terrain = ctx.terrainOrder[chunk.terrain[ty * CHUNK + tx] ?? 0];
      if (terrain === 'deep') continue;
      const roll = rng.next();
      let acc = 0;
      for (const rule of biome.objects) {
        acc += rule.density;
        if (roll >= acc) continue;
        if (terrain === 'shallow' && rule.kind !== 'reeds') break;
        const x = chunk.cx * CHUNK + tx + rng.range(0.1, 0.9), y = chunk.cy * CHUNK + ty + rng.range(0.1, 0.9);
        if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 3.2) break;
        if (isRoad(ctx.seed, x, y, ctx.spawn)) break;
        if (chunk.objects.some((o) => Math.hypot(o.x - x, o.y - y) < rule.minSpacing * 0.5)) break;
        const def = ctx.content.objects.get(rule.kind);
        chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:${n++}`, kind: rule.kind, x, y, variant: rng.int(0, 3), hp: def?.hp ?? 1 });
        break;
      }
    }
  },
};

export type RoadKind = 'highway' | 'dirt_road' | 'small_road' | 'none';
/**
 * Town street half-widths in tiles (1 tile = 1 m), HD plan G2 scale pass:
 * main streets are 7 m (two lanes plus 0.9 m sidewalks each side), side streets 4 m.
 */
const MAIN_HALF = 3.5, SIDE_HALF = 2.0, SIDEWALK = 0.9;

export interface StreetDetails {
  kind: RoadKind;
  isTown: boolean;
  isSidewalk: boolean;
  isCrosswalk: boolean;
  isPlaza: boolean;
  axis: 'x' | 'y' | 'both';
}

export type TownSize = 'small' | 'big' | 'metro';
export interface TownCenter { name: string; x: number; y: number; size: TownSize; spawn?: boolean }

/** Regional towns, spread 1.35x wider than the first layout so the larger G2 town footprints never overlap. */
export const REGIONAL_TOWNS: TownCenter[] = [
  { name: 'Rust Creek Junction', x: -97, y: 108, size: 'small' },
  { name: 'Port Seagull Harbor', x: 108, y: -130, size: 'big' },
  { name: 'Pine Ridge Outpost', x: 130, y: 86, size: 'small' },
  { name: 'Deadwood Oasis Hub', x: -119, y: -108, size: 'big' },
  { name: 'Copperfield Metropolis', x: 162, y: -43, size: 'metro' },
  { name: 'Ashfield Hamlet', x: -194, y: 65, size: 'small' },
  { name: 'Ironvale Borough', x: 205, y: 151, size: 'big' },
];

interface TownProfile {
  radius: number;
  mainSpan: number;
  crossSpan: number;
  lines: number[];
  plazaRadius: number;
  plazaOffset: { x: number; y: number };
}

type TownDistrict = 'plaza' | 'residential' | 'commercial' | 'industrial';
interface CarSpawnRule {
  every: number;
  chance: number;
  wreckBias: number;
  minSpacing: number;
}

/** Town footprints by size, about 1.3x the first pass so streets and lots fit real house sizes (HD plan G2). */
const TOWN_PROFILES: Record<TownSize, TownProfile> = {
  small: { radius: 39, mainSpan: 32, crossSpan: 29, lines: [0, 17], plazaRadius: 4.0, plazaOffset: { x: -2.6, y: -1.9 } },
  big: { radius: 52, mainSpan: 47, crossSpan: 44, lines: [0, 22, 40], plazaRadius: 4.6, plazaOffset: { x: -3.1, y: -2.2 } },
  metro: { radius: 60, mainSpan: 57, crossSpan: 52, lines: [0, 22, 43], plazaRadius: 5.0, plazaOffset: { x: -3.6, y: -2.4 } },
};

const CAR_SPAWN_RULES: Record<TownDistrict, CarSpawnRule> = {
  plaza: { every: 11, chance: 0.18, wreckBias: 0.35, minSpacing: 3.6 },
  residential: { every: 9, chance: 0.34, wreckBias: 0.25, minSpacing: 3.2 },
  commercial: { every: 8, chance: 0.42, wreckBias: 0.4, minSpacing: 3.0 },
  industrial: { every: 7, chance: 0.46, wreckBias: 0.7, minSpacing: 2.8 },
};

const townCenters = (spawn?: { x: number; y: number }): TownCenter[] => {
  const spX = spawn ? spawn.x : 0;
  const spY = spawn ? spawn.y : 0;
  return [{ name: 'Metropolis Oakhaven', x: spX, y: spY, size: 'metro', spawn: true }, ...REGIONAL_TOWNS];
};

const centerStreetLines = (seed: number, center: TownCenter): number[] => {
  const base = TOWN_PROFILES[center.size].lines;
  if (center.spawn) return [...base];
  return base.map((l, i) => {
    if (l === 0) return 0;
    const n = (hash2(seed + 1201 + i * 17, Math.round(center.x), Math.round(center.y)) % 3) - 1;
    return Math.max(6, l + n);
  }).sort((a, b) => a - b);
};

const centerCurves = (seed: number, center: TownCenter, x: number, y: number): { xBend: number; yBend: number } => {
  if (center.spawn) return { xBend: 0, yBend: 0 };
  const amp = center.size === 'small' ? 0.16 : center.size === 'big' ? 0.22 : 0.28;
  const f = 0.085;
  const xBend = Math.sin((y - center.y) * f + seed * 0.0013 + center.x * 0.03) * amp;
  const yBend = Math.cos((x - center.x) * f + seed * 0.0011 + center.y * 0.03) * amp;
  return { xBend, yBend };
};

const districtAt = (center: TownCenter, wx: number, wy: number): TownDistrict => {
  const profile = TOWN_PROFILES[center.size];
  const dx = wx - center.x, dy = wy - center.y;
  if (Math.hypot(dx - profile.plazaOffset.x, dy - profile.plazaOffset.y) < profile.plazaRadius + 1.2) return 'plaza';
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 'commercial' : 'industrial';
  return dy >= 0 ? 'residential' : 'plaza';
};

const centerAt = (seed: number, spawn: { x: number; y: number }, x: number, y: number): TownCenter | null => {
  for (const t of townCenters(spawn)) {
    const p = TOWN_PROFILES[t.size];
    if (Math.hypot(x - t.x, y - t.y) <= p.radius) return t;
  }
  return null;
};

const townBlockSpans = (p: TownProfile, linesOverride?: number[]): [number, number][] => {
  const out: [number, number][] = [];
  const lines = [...(linesOverride ?? p.lines)].sort((a, b) => a - b);
  for (let i = 0; i < lines.length - 1; i++) {
    const a = lines[i] ?? 0;
    const b = lines[i + 1] ?? 0;
    const start = a + streetHalf(a) + 1.1;
    const end = b - SIDE_HALF - 1.1;
    if (end - start > 2.2) out.push([start, end]);
  }
  const last = lines[lines.length - 1] ?? 0;
  const outerStart = last + streetHalf(last) + 1.1;
  const outerEnd = p.radius - 2.2;
  if (outerEnd - outerStart > 2.2) out.push([outerStart, outerEnd]);
  return out;
};

export function getStreetDetails(seed: number, x: number, y: number, spawn?: { x: number; y: number }): StreetDetails {
  // 1. Check Town Centers (spawn metro + regional centers with varied sizes)
  for (const t of townCenters(spawn)) {
    const profile = TOWN_PROFILES[t.size];
    const lines = centerStreetLines(seed, t);
    const { xBend, yBend } = centerCurves(seed, t, x, y);
    const dx = x - t.x, dy = y - t.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= profile.radius) {
      // Civic Plaza around fountain
      const plazaDist = Math.hypot(dx - profile.plazaOffset.x, dy - profile.plazaOffset.y);
      if (plazaDist < profile.plazaRadius && dx < -0.8) {
        return { kind: 'small_road', isTown: true, isSidewalk: false, isCrosswalk: false, isPlaza: true, axis: 'both' };
      }

      const crossLines = lines.filter((l) => l > 0);
      // Town Main Street (East-West Arterial Avenue), softly curved
      const onMainSt = Math.abs(dy - yBend) < MAIN_HALF && Math.abs(dx) <= profile.mainSpan;
      // Town Central Boulevard (North-South Arterial Avenue), softly curved
      const onCentralBlvd = Math.abs(dx - xBend) < MAIN_HALF && Math.abs(dy) <= profile.mainSpan;

      if (onMainSt || onCentralBlvd) {
        const kerb = MAIN_HALF - SIDEWALK;
        const isSidewalk = onMainSt && !onCentralBlvd ? Math.abs(dy - yBend) >= kerb : onCentralBlvd && !onMainSt ? Math.abs(dx - xBend) >= kerb : false;
        // Pedestrian Zebra Crosswalk at intersections and cross-street junctions
        const isCrosswalk = (Math.abs(dx - xBend) < MAIN_HALF && Math.abs(dy - yBend) < MAIN_HALF)
          || (onMainSt && crossLines.some((l) => Math.abs(Math.abs(dx - xBend) - l) < SIDE_HALF))
          || (onCentralBlvd && crossLines.some((l) => Math.abs(Math.abs(dy - yBend) - l) < SIDE_HALF));

        return {
          kind: 'highway',
          isTown: true,
          isSidewalk,
          isCrosswalk,
          isPlaza: false,
          axis: onMainSt && onCentralBlvd ? 'both' : onMainSt ? 'x' : 'y'
        };
      }

      // Secondary Town Cross-Streets & Cobblestone Alleys
      const onCrossX = crossLines.some((l) => Math.abs(Math.abs(dx - xBend) - l) < SIDE_HALF) && Math.abs(dy - yBend) <= profile.crossSpan;
      const onCrossY = crossLines.some((l) => Math.abs(Math.abs(dy - yBend) - l) < SIDE_HALF) && Math.abs(dx - xBend) <= profile.crossSpan;
      if (onCrossX || onCrossY) {
        return {
          kind: 'small_road',
          isTown: true,
          isSidewalk: false,
          isCrosswalk: false,
          isPlaza: false,
          axis: onCrossX && onCrossY ? 'both' : onCrossX ? 'y' : 'x'
        };
      }
    }
  }

  // 2. Highways between towns and dirt tracks into the countryside: smooth curves from the road graph.
  const hit = roadHit(worldRoadNet(seed, spawn), x, y);
  if (hit) return { kind: hit.kind, isTown: false, isSidewalk: false, isCrosswalk: false, isPlaza: false, axis: hit.axis };

  return { kind: 'none', isTown: false, isSidewalk: false, isCrosswalk: false, isPlaza: false, axis: 'both' };
}

/** Deep-water tests per seed, registered by WorldGen so roads can bend around lakes. */
const waterBySeed = new Map<number, (x: number, y: number) => boolean>();

/** Town exits (main-street ends) and spur starts (side-street ends) feed the road graph. */
function roadNodes(seed: number, spawn?: { x: number; y: number }): RoadNode[] {
  return townCenters(spawn).map((t) => {
    const p = TOWN_PROFILES[t.size];
    const exits: RoadNode['exits'] = [[t.x + p.mainSpan, t.y, 1, 0], [t.x - p.mainSpan, t.y, -1, 0], [t.x, t.y + p.mainSpan, 0, 1], [t.x, t.y - p.mainSpan, 0, -1]];
    const spurs: RoadNode['spurs'] = [];
    for (const l of centerStreetLines(seed, t).filter((v) => v > 0)) for (const s of [-1, 1]) {
      spurs.push([t.x + s * l, t.y + p.crossSpan, 0, 1], [t.x + s * l, t.y - p.crossSpan, 0, -1]);
      spurs.push([t.x + p.crossSpan, t.y + s * l, 1, 0], [t.x - p.crossSpan, t.y + s * l, -1, 0]);
    }
    return { x: t.x, y: t.y, radius: p.radius, exits, spurs };
  });
}

function worldRoadNet(seed: number, spawn?: { x: number; y: number }): ReturnType<typeof roadNet> {
  const sp = spawn ?? { x: 0, y: 0 };
  return roadNet(`${seed}:${sp.x},${sp.y}`, () => buildRoadNet(seed, roadNodes(seed, spawn), waterBySeed.get(seed) ?? (() => false)));
}

/** Highway and dirt-road polylines near a box, for drawing them as smooth ribbons. */
export function roadPathsNear(seed: number, spawn: { x: number; y: number }, x0: number, y0: number, x1: number, y1: number, margin: number): RoadPath[] {
  return roadsInBox(worldRoadNet(seed, spawn), x0, y0, x1, y1, margin);
}

export function getRoadType(seed: number, x: number, y: number, spawn?: { x: number; y: number }): RoadKind {
  return getStreetDetails(seed, x, y, spawn).kind;
}

export function isRoad(seed: number, x: number, y: number, spawn?: { x: number; y: number }): boolean {
  return getRoadType(seed, x, y, spawn) !== 'none';
}

/** Stage: road carving disabled to keep terrain fully natural (no asphalt/dirt road painting). */
export const roadStage: GenStage = {
  name: 'roads',
  run() {
    // Intentionally no-op: roads are removed visually and terrain remains biome-native.
  },
};

/** Stage 3: abandoned camps. Shows how a feature stage works: deterministic per region, adds objects and a feature record. */
export const campStage: GenStage = {
  name: 'camps',
  run(ctx, chunk) {
    const R = 6; // region size in chunks
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed, rx, ry * 31 + 7) > 0.6) return;
    const ccx = rx * R + (hash2(ctx.seed, rx, ry) % R), ccy = ry * R + (hash2(ctx.seed + 1, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 6) return;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 5);
    const r = ctx.rng(chunk.cx, chunk.cy, 'camp');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:c${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };
    add('tent', 0, -1.4); add('crate', 1.6, 0.4); add('crate', 2.2, 1.0); add('log', -1.8, 1.2); add('log', 1.2, 2.2); add('stump', -2.4, -0.2);
    chunk.features.push({ type: 'camp', x, y });
  },
};

/** Stage 4: Abandoned survivor settlements and scavenging sites with supplies. */
export const settlementStage: GenStage = {
  name: 'settlements',
  run(ctx, chunk) {
    const R = 8;
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed + 19, rx, ry * 31 + 13) > 0.55) return;
    const ccx = rx * R + (hash2(ctx.seed + 17, rx, ry) % R);
    const ccy = ry * R + (hash2(ctx.seed + 23, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 8) return;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 6);
    const r = ctx.rng(chunk.cx, chunk.cy, 'settlement');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:s${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };
    add('tent', -2.2, -1.8);
    add('crate', 1.8, -1.2);
    add('crate', 2.4, -0.6);
    add('crate', -1.4, 2.0);
    add('log', 0, 2.4);
    add('stump', 2.5, 2.0);
    chunk.features.push({ type: 'settlement', x, y, name: 'Survivor Settlement' });
  },
};

/** Stage 5: Highway gas stations with fuel pumps and scavengable vehicle supplies. */
export const gasStationStage: GenStage = {
  name: 'gas_stations',
  run(ctx, chunk) {
    const R = 14;
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed + 31, rx, ry * 29 + 17) > 0.6) return;
    const ccx = rx * R + (hash2(ctx.seed + 37, rx, ry) % R);
    const ccy = ry * R + (hash2(ctx.seed + 41, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 12) return;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 7);
    const r = ctx.rng(chunk.cx, chunk.cy, 'gas');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:g${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };
    add('fuel_pump', -1.5, -0.8);
    add('fuel_pump', 1.5, -0.8);
    add('wrecked_car', 0.2, 2.2);
    add('crate', -2.2, 1.4);
    add('crate', -2.8, 0.8);
    add('tent', -2.4, -2.2);
    chunk.features.push({ type: 'gas_station', x, y, name: 'Highway Gas Station' });
  },
};

/** Stage 6: Mountain and hilltop radio communication towers. */
export const radioTowerStage: GenStage = {
  name: 'radio_towers',
  run(ctx, chunk) {
    const R = 16;
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed + 47, rx, ry * 37 + 11) > 0.55) return;
    const ccx = rx * R + (hash2(ctx.seed + 43, rx, ry) % R);
    const ccy = ry * R + (hash2(ctx.seed + 53, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow' || s.height < 0.6) return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 16) return;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 6);
    const r = ctx.rng(chunk.cx, chunk.cy, 'tower');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:rt${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };
    add('radio_tower', 0, 0);
    add('crate', 1.6, 0.8);
    add('sandbags', -1.4, -1.2);
    chunk.features.push({ type: 'radio_tower', x, y, name: 'Signal Radio Tower' });
  },
};

/** Stage 7: Tiered town centers (small, big, metro) with deterministic placement and districts. */
export const townStage: GenStage = {
  name: 'towns',
  run(ctx, chunk) {
    const scx = Math.floor(ctx.spawn.x / CHUNK), scy = Math.floor(ctx.spawn.y / CHUNK);
    const isSpawnChunk = chunk.cx === scx && chunk.cy === scy;
    const isSpawnDistrict = Math.abs(chunk.cx - scx) <= 1 && Math.abs(chunk.cy - scy) <= 1;

    if (isSpawnDistrict && !isSpawnChunk) {
      const cx = chunk.cx * CHUNK + 8, cy = chunk.cy * CHUNK + 8;
      const dx = chunk.cx - scx, dy = chunk.cy - scy;
      const names: Record<string, string> = {
        '-1,0': 'Oakhaven Auto Garage',
        '1,0': 'Oakhaven Commercial High Street',
        '0,-1': 'Oakhaven Triage & Defense Post',
        '0,1': 'Oakhaven Residential Quarter',
      };
      const name = names[`${dx},${dy}`];
      if (name) chunk.features.push({ type: 'town_district', x: cx, y: cy, name });
      return;
    }

    const center = townCenters(ctx.spawn).find((t) => Math.floor(t.x / CHUNK) === chunk.cx && Math.floor(t.y / CHUNK) === chunk.cy);
    if (!center) return;

    const x = center.x, y = center.y;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (!center.spawn && Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 22) return;

    const clearR = center.size === 'small' ? 6.0 : center.size === 'big' ? 7.2 : 8.0;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > clearR);

    const r = ctx.rng(chunk.cx, chunk.cy, `town_${center.size}`);
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:tw${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };

    if (center.spawn) {
      // Keep spawn plaza clear around player start while still feeling like a metro hub.
      add('pine', 4.0, 3.0);
      add('rock', 5.0, 3.0);
      add('wrecked_car', 4.5, 4.0);
      add('crate', 4.0, 3.5);
      add('fuel_pump', 5.5, 5.0);
      add('tent', 6.0, 4.0);
      add('log', 5.0, 4.2);
      add('town_fountain', 5.0, 5.5);
      add('park_bench', 4.2, 5.5);
      add('street_light', 5.0, 7.0);
      add('dumpster', 6.0, 6.0);
      add('fire_hydrant', 3.5, 5.0);
      add('guardrail', 3.2, 6.0);
      add('traffic_cone', 4.2, 6.5);
    } else if (center.size === 'small') {
      add('town_fountain', 0, 0);
      add('street_light', -1.8, -1.8);
      add('park_bench', 1.4, 0.2);
      add('crate', -1.0, 1.0);
      add('traffic_cone', 0.8, 1.5);
      add('tent', -1.6, 2.1);
      add('wrecked_car', -2.2, -0.9);
    } else if (center.size === 'big') {
      add('town_fountain', 0, 0);
      add('street_light', -2.5, -2.5);
      add('street_light', 2.5, 2.5);
      add('park_bench', -1.8, 0);
      add('park_bench', 1.8, 0);
      add('wrecked_car', -3.2, -1.8);
      add('fuel_pump', 3.2, -1.8);
      add('dumpster', -2.8, 2.2);
      add('fire_hydrant', 2.6, -2.4);
      add('traffic_cone', 1.2, 1.8);
      add('tent', -2.2, 3.2);
      add('crate', -1.2, 1.4);
      add('crate', 1.4, 0.6);
      add('guardrail', 0, 3.5);
    } else {
      add('town_fountain', 0, 0);
      add('street_light', -3.0, -3.0);
      add('street_light', 3.0, -3.0);
      add('street_light', -3.0, 3.0);
      add('street_light', 3.0, 3.0);
      add('park_bench', -2.2, 0.2);
      add('park_bench', 2.2, -0.2);
      add('wrecked_car', -3.4, -2.1);
      add('wrecked_car', 3.3, 1.8);
      add('fuel_pump', 3.4, -1.8);
      add('dumpster', -3.2, 2.2);
      add('fire_hydrant', 2.8, -2.6);
      add('traffic_cone', 1.3, 2.0);
      add('tent', -2.4, 3.3);
      add('crate', -1.3, 1.5);
      add('crate', 1.5, 0.7);
      add('guardrail', 0, 3.6);
      add('pine', 4.0, 3.0);
      add('rock', 5.0, 3.0);
      add('log', 5.0, 4.2);
    }

    chunk.features.push({ type: 'town', x, y, name: center.name });
  },
};

/** Stage 8: District-aware parked cars along town curbs, deterministic per chunk. */
export const parkedCarsStage: GenStage = {
  name: 'parked_cars',
  run(ctx, chunk) {
    const x0 = chunk.cx * CHUNK, y0 = chunk.cy * CHUNK;
    const x1 = x0 + CHUNK, y1 = y0 + CHUNK;
    const touching = townBuildings(ctx).filter((b) => b.x - 0.9 < x1 && b.x + b.w + 0.9 > x0 && b.y - 0.9 < y1 && b.y + b.h + 0.9 > y0);
    const rng = ctx.rng(chunk.cx, chunk.cy, 'parked_cars');

    const blockedByBuilding = (x: number, y: number): boolean => touching.some((b) => x > b.x - 0.8 && x < b.x + b.w + 0.8 && y > b.y - 0.8 && y < b.y + b.h + 0.8);

    const nearObj = (x: number, y: number, r: number): boolean => chunk.objects.some((o) => Math.hypot(o.x - x, o.y - y) < r);

    for (let ty = 1; ty < CHUNK - 1; ty += 2) {
      for (let tx = 1; tx < CHUNK - 1; tx += 2) {
        const wx = chunk.cx * CHUNK + tx + 0.5;
        const wy = chunk.cy * CHUNK + ty + 0.5;

        const center = centerAt(ctx.seed, ctx.spawn, wx, wy);
        if (!center) continue;

        // Park just off the kerb, lined up with the street.
        let roadAxis: 'x' | 'y' | null = null;
        const here = getStreetDetails(ctx.seed, wx, wy, ctx.spawn);
        if (here.kind !== 'none') continue;
        const east = getStreetDetails(ctx.seed, wx + 1.2, wy, ctx.spawn);
        const west = getStreetDetails(ctx.seed, wx - 1.2, wy, ctx.spawn);
        const north = getStreetDetails(ctx.seed, wx, wy - 1.2, ctx.spawn);
        const south = getStreetDetails(ctx.seed, wx, wy + 1.2, ctx.spawn);
        const roadInfo = east.kind !== 'none' ? east : west.kind !== 'none' ? west : north.kind !== 'none' ? north : south.kind !== 'none' ? south : null;
        if (!roadInfo || !roadInfo.isTown || roadInfo.isPlaza || roadInfo.isCrosswalk) continue;
        if (east.kind !== 'none' || west.kind !== 'none') roadAxis = 'y';
        else if (north.kind !== 'none' || south.kind !== 'none') roadAxis = 'x';
        if (!roadAxis) continue;

        const district = districtAt(center, wx, wy);
        const rule = CAR_SPAWN_RULES[district];
        if (((Math.floor(wx * 2) + Math.floor(wy * 3) + hash2(ctx.seed + 7001, chunk.cx, chunk.cy)) % rule.every) !== 0) continue;
        if (rng.next() > rule.chance) continue;

        const sample = ctx.sample(wx, wy);
        if (sample.terrain === 'deep' || sample.terrain === 'shallow') continue;
        if (Math.hypot(wx - ctx.spawn.x, wy - ctx.spawn.y) < 5.5) continue;
        if (blockedByBuilding(wx, wy)) continue;
        if (nearObj(wx, wy, rule.minSpacing)) continue;

        const kind: ObjectKind = rng.next() < rule.wreckBias ? 'wrecked_car' : 'abandoned_car';
        const def = ctx.content.objects.get(kind);
        const facing = roadAxis === 'x' ? (rng.next() < 0.5 ? 0 : 2) : (rng.next() < 0.5 ? 1 : 3);
        chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:pc${chunk.objects.length}`, kind, x: wx, y: wy, variant: facing, hp: def?.hp ?? 1 });
      }
    }
  },
};

/** Stage 9: Fortified military checkpoints and outposts. */
export const militaryStage: GenStage = {
  name: 'military',
  run(ctx, chunk) {
    const R = 18;
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed + 79, rx, ry * 47 + 31) > 0.6) return;
    const ccx = rx * R + (hash2(ctx.seed + 83, rx, ry) % R);
    const ccy = ry * R + (hash2(ctx.seed + 89, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 20) return;
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 6.5);
    const r = ctx.rng(chunk.cx, chunk.cy, 'milit');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:mil${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };
    add('sandbags', -1.8, -1.2);
    add('sandbags', 1.8, -1.2);
    add('sandbags', 0, 2.2);
    add('tactical_shield', 0, -1.8);
    add('slime_drum', -2.8, -0.8);
    add('scrap_wall', 2.8, -0.6);
    add('crate', -1.2, 0.8);
    add('crate', 1.2, 0.8);
    add('wrecked_car', -2.5, 1.6);
    add('tent', 2.2, 1.8);
    chunk.features.push({ type: 'military_outpost', x, y, name: 'Military Outpost' });
  },
};

/** Stage 10: Discrete mineral nodes and geodes (Factorio / Satisfactory extraction). */
export const mineralGeodeStage: GenStage = {
  name: 'mineral_nodes',
  run(ctx, chunk) {
    const R = 4; // Region size 4x4 chunks
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    const hash = hash2(ctx.seed + 101, rx, ry);
    const ccx = rx * R + (hash % R);
    const ccy = ry * R + (hash2(ctx.seed + 107, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;

    const x = chunk.cx * CHUNK + 8;
    const y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 6) return;

    // Filter overlapping objects
    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 5);

    // Pick node type based on hash and biome
    const typeRoll = (hash >> 3) % 5;
    const nodeKind: ObjectKind = typeRoll === 0 ? 'iron_node'
      : typeRoll === 1 ? 'copper_node'
      : typeRoll === 2 ? 'coal_node'
      : typeRoll === 3 ? 'oil_seam'
      : 'crystal_geode';

    // Purity level (impure: 2 nodes, normal: 3 nodes, pure: 4 nodes)
    const purityRoll = (hash >> 6) % 10;
    const count = purityRoll > 7 ? 4 : purityRoll > 3 ? 3 : 2;
    const purityName = purityRoll > 7 ? 'Pure' : purityRoll > 3 ? 'Normal' : 'Impure';

    const def = ctx.content.objects.get(nodeKind);
    const r = ctx.rng(chunk.cx, chunk.cy, 'mineral');
    for (let i = 0; i < count; i++) {
      const ang = (i / count) * Math.PI * 2 + r.range(-0.3, 0.3);
      const dist = r.range(1.0, 2.2);
      const nx = x + Math.cos(ang) * dist;
      const ny = y + Math.sin(ang) * dist;
      chunk.objects.push({
        id: `${chunk.cx},${chunk.cy}:mn${chunk.objects.length}`,
        kind: nodeKind,
        x: nx,
        y: ny,
        variant: r.int(0, 3),
        hp: def?.hp ?? 10
      });
    }

    const featureName = `${purityName} ${def?.name ?? 'Mineral Deposit'}`;
    chunk.features.push({ type: 'mineral_deposit', x, y, name: featureName });
  },
};

// ---------------------------------------------------------------- buildings
export type Rect = [number, number, number, number];
export const WALL_T = 0.35, DOOR_W = 1.8;
const wallCache = new WeakMap<Building, Rect[]>();

const doorSegment = (b: Pick<Building, 'x' | 'y' | 'w' | 'h' | 'door' | 'doorAt'>): BuildingPiece => {
  const half = DOOR_W / 2;
  if (b.door === 'N') return { kind: 'door', x0: b.x + b.doorAt - half, y0: b.y, x1: b.x + b.doorAt + half, y1: b.y, exterior: true };
  if (b.door === 'S') return { kind: 'door', x0: b.x + b.doorAt - half, y0: b.y + b.h, x1: b.x + b.doorAt + half, y1: b.y + b.h, exterior: true };
  if (b.door === 'W') return { kind: 'door', x0: b.x, y0: b.y + b.doorAt - half, x1: b.x, y1: b.y + b.doorAt + half, exterior: true };
  return { kind: 'door', x0: b.x + b.w, y0: b.y + b.doorAt - half, x1: b.x + b.w, y1: b.y + b.doorAt + half, exterior: true };
};

const wallRectFromPiece = (p: BuildingPiece): Rect => {
  if (Math.abs(p.x1 - p.x0) >= Math.abs(p.y1 - p.y0)) {
    const x0 = Math.min(p.x0, p.x1), x1 = Math.max(p.x0, p.x1), y = p.y0;
    return [x0, y - WALL_T / 2, x1, y + WALL_T / 2];
  }
  const y0 = Math.min(p.y0, p.y1), y1 = Math.max(p.y0, p.y1), x = p.x0;
  return [x - WALL_T / 2, y0, x + WALL_T / 2, y1];
};

/** Build modular snap pieces for a building footprint and layout. */
export function buildSnapPieces(b: Pick<Building, 'x' | 'y' | 'w' | 'h' | 'door' | 'doorAt'>, layout: string = 'none'): BuildingPiece[] {
  const pieces: BuildingPiece[] = [];
  const half = DOOR_W / 2;
  const addSplit = (x0: number, y0: number, x1: number, y1: number, ext = false): void => {
    pieces.push({ kind: 'wall', x0, y0, x1, y1, exterior: ext });
  };
  const addSide = (side: Side): void => {
    if (side === 'N' || side === 'S') {
      const y = side === 'N' ? b.y : b.y + b.h;
      const x0 = b.x, x1 = b.x + b.w;
      if (b.door !== side) addSplit(x0, y, x1, y, true);
      else {
        addSplit(x0, y, b.x + b.doorAt - half, y, true);
        addSplit(b.x + b.doorAt + half, y, x1, y, true);
      }
    } else {
      const x = side === 'W' ? b.x : b.x + b.w;
      const y0 = b.y, y1 = b.y + b.h;
      if (b.door !== side) addSplit(x, y0, x, y1, true);
      else {
        addSplit(x, y0, x, b.y + b.doorAt - half, true);
        addSplit(x, b.y + b.doorAt + half, x, y1, true);
      }
    }
  };
  addSide('N'); addSide('S'); addSide('W'); addSide('E');
  pieces.push(doorSegment(b));

  const makeInteriorDivider = (ratio: number): void => {
    const clamped = clamp(ratio, 0.35, 0.72);
    if (b.door === 'N' || b.door === 'S') {
      const y = b.door === 'N' ? b.y + b.h * clamped : b.y + b.h * (1 - clamped);
      const g0 = clamp(b.x + b.doorAt - half, b.x + 1.0, b.x + b.w - 1.0), g1 = clamp(b.x + b.doorAt + half, b.x + 1.0, b.x + b.w - 1.0);
      addSplit(b.x + WALL_T, y, g0, y);
      addSplit(g1, y, b.x + b.w - WALL_T, y);
      pieces.push({ kind: 'door', x0: g0, y0: y, x1: g1, y1: y, exterior: false });
    } else {
      const x = b.door === 'W' ? b.x + b.w * clamped : b.x + b.w * (1 - clamped);
      const g0 = clamp(b.y + b.doorAt - half, b.y + 1.0, b.y + b.h - 1.0), g1 = clamp(b.y + b.doorAt + half, b.y + 1.0, b.y + b.h - 1.0);
      addSplit(x, b.y + WALL_T, x, g0);
      addSplit(x, g1, x, b.y + b.h - WALL_T);
      pieces.push({ kind: 'door', x0: x, y0: g0, x1: x, y1: g1, exterior: false });
    }
  };

  if (layout === 'split' || layout === 'apartment') makeInteriorDivider(0.5);
  else if (layout === 'backroom') makeInteriorDivider(0.62);

  return pieces;
}

/** Solid wall rectangles [x0,y0,x1,y1] of a building (the door is a gap). */
export function wallRects(b: Building): Rect[] {
  let r = wallCache.get(b);
  if (r) return r;
  const walls = b.pieces?.filter((p) => p.kind === 'wall');
  if (walls?.length) r = walls.map(wallRectFromPiece);
  else {
    const out: Rect[] = [];
    const side = (s: Side, len: number, mk: (a0: number, a1: number) => Rect): void => {
      if (b.door !== s) { out.push(mk(0, len)); return; }
      const g0 = Math.max(0, b.doorAt - DOOR_W / 2), g1 = Math.min(len, b.doorAt + DOOR_W / 2);
      if (g0 > 0) out.push(mk(0, g0));
      if (g1 < len) out.push(mk(g1, len));
    };
    side('N', b.w, (a0, a1) => [b.x + a0, b.y, b.x + a1, b.y + WALL_T]);
    side('S', b.w, (a0, a1) => [b.x + a0, b.y + b.h - WALL_T, b.x + a1, b.y + b.h]);
    side('W', b.h, (a0, a1) => [b.x, b.y + a0, b.x + WALL_T, b.y + a1]);
    side('E', b.h, (a0, a1) => [b.x + b.w - WALL_T, b.y + a0, b.x + b.w, b.y + a1]);
    r = out;
  }
  wallCache.set(b, r);
  return r;
}

/** True when (x,y) is under the roof, or standing in the doorway apron (so the roof lifts as you step in). */
export function insideBuilding(b: Building, x: number, y: number): boolean {
  if (x > b.x && x < b.x + b.w && y > b.y && y < b.y + b.h) return true;
  const half = DOOR_W / 2, apron = 0.9;
  if (b.door === 'N') return Math.abs(x - (b.x + b.doorAt)) < half && y <= b.y && y > b.y - apron;
  if (b.door === 'S') return Math.abs(x - (b.x + b.doorAt)) < half && y >= b.y + b.h && y < b.y + b.h + apron;
  if (b.door === 'W') return Math.abs(y - (b.y + b.doorAt)) < half && x <= b.x && x > b.x - apron;
  return Math.abs(y - (b.y + b.doorAt)) < half && x >= b.x + b.w && x < b.x + b.w + apron;
}

/** Does a circle touch any wall of the building? */
export function circleHitsWalls(b: Building, x: number, y: number, r: number): boolean {
  for (const [x0, y0, x1, y1] of wallRects(b)) {
    const nx = Math.max(x0, Math.min(x, x1)), ny = Math.max(y0, Math.min(y, y1));
    if ((x - nx) ** 2 + (y - ny) ** 2 < r * r) return true;
  }
  return false;
}

/** Lot blocks between town streets are derived per-town-size from the same profile used by getStreetDetails. */
const townCache = new WeakMap<GenContext, Building[]>();

/** Half-width of a town street line (Main Street / Central Blvd are wider). Matches getStreetDetails. */
const streetHalf = (line: number): number => (line === 0 ? MAIN_HALF : SIDE_HALF);
/** Target lot frontage along a block (tiles); sized for 8–12 m houses with side gaps. */
const LOT_SIZE = 11;
/** Houses sit this far inside the street edge of their lot, so fronts line up along the block. */
const FRONT_SETBACK = 1.0;
interface LotFront { axis: 'x' | 'y'; edge: number; back: number; dir: number; gap: number; score: number }

/** All buildings of all towns. Deterministic: lots sit in the blocks between streets, doors face the nearest street. */
export function townBuildings(ctx: GenContext): Building[] {
  const cached = townCache.get(ctx);
  if (cached) return cached;
  const out: Building[] = [];
  const defs = ctx.content.buildings;
  const total = defs.reduce((s, d) => s + d.weight, 0);
  const towns = townCenters(ctx.spawn);
  if (defs.length) towns.forEach((t, ti) => {
    let n = 0;
    const profile = TOWN_PROFILES[t.size];
    const lines = centerStreetLines(ctx.seed, t);
    const blockSpans = townBlockSpans(profile, lines);
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const [a0, a1] of blockSpans) for (const [b0, b1] of blockSpans) {
      const bx0 = sx > 0 ? a0 : -a1, bx1 = sx > 0 ? a1 : -a0, by0 = sy > 0 ? b0 : -b1, by1 = sy > 0 ? b1 : -b0;
      const nx = Math.max(1, Math.round((bx1 - bx0) / LOT_SIZE)), ny = Math.max(1, Math.round((by1 - by0) / LOT_SIZE));
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
        const lx0 = bx0 + ((bx1 - bx0) * i) / nx, lx1 = bx0 + ((bx1 - bx0) * (i + 1)) / nx;
        const ly0 = by0 + ((by1 - by0) * j) / ny, ly1 = by0 + ((by1 - by0) * (j + 1)) / ny;
        const wx0 = Math.round(t.x + lx0), wy0 = Math.round(t.y + ly0);
        if (rand2(ctx.seed + 211, wx0, wy0) < 0.14) continue; // vacant lot
        let pick = rand2(ctx.seed + 223, wx0, wy0) * total, def = defs[0]!;
        for (const d of defs) { pick -= d.weight; if (pick <= 0) { def = d; break; } }
        // The lot's front is the edge closest to a street (Main Street wins near-ties, other ties by hash).
        let front: LotFront | undefined;
        for (const axis of ['x', 'y'] as const) {
          const s = axis === 'x' ? sx : sy, l0 = axis === 'x' ? lx0 : ly0, l1 = axis === 'x' ? lx1 : ly1;
          const inner = s > 0 ? l0 : l1, outer = s > 0 ? l1 : l0;
          const lin = Math.max(...lines.filter((l) => l <= Math.abs(inner)));
          const gin = Math.abs(inner) - lin - streetHalf(lin);
          const cands: LotFront[] = [{ axis, edge: inner, back: outer, dir: s, gap: gin, score: gin - (lin === 0 ? 0.6 : 0) }];
          const lout = lines.find((l) => l > 0 && l >= Math.abs(outer));
          if (lout !== undefined) { const gap = lout - streetHalf(lout) - Math.abs(outer); cands.push({ axis, edge: outer, back: inner, dir: -s, gap, score: gap }); }
          for (const c of cands) {
            const tie = front && Math.abs(c.score - front.score) < 0.05;
            if (!front || c.score < front.score - 0.05 || (tie && (hash2(ctx.seed + 233, wx0, wy0) & 1))) front = c;
          }
        }
        const f = front!;
        const tx = f.axis === 'x' ? t.x : t.y, ta = f.axis === 'x' ? t.y : t.x;
        const a0 = ta + (f.axis === 'x' ? ly0 : lx0) + 0.5, a1 = ta + (f.axis === 'x' ? ly1 : lx1) - 0.5;
        // depth: front wall sits FRONT_SETBACK inside the lot edge (0.5 on shallow corner lots); the rest is back yard
        let wall0 = 0, depth = 0;
        for (const setback of [FRONT_SETBACK, 0.5]) {
          const F = tx + f.edge, B = tx + f.back - f.dir * 0.5;
          wall0 = f.dir > 0 ? Math.ceil(F + setback) : Math.floor(F - setback);
          depth = Math.min(def.maxH, Math.floor(Math.abs(B - wall0)));
          if (depth >= def.minH) break;
        }
        let along = Math.ceil(a0);
        const span = Math.min(def.maxW, Math.floor(a1 - along));
        if (span < def.minW || depth < def.minH) continue;
        along += Math.floor((a1 - along - span) / 2);
        const near = f.dir > 0 ? wall0 : wall0 - depth;
        const backWall = f.dir > 0 ? wall0 + depth : wall0 - depth;
        const yard = Math.abs(tx + f.back - backWall), frontGap = f.gap + Math.abs(wall0 - (tx + f.edge));
        let x: number, y: number, w: number, h: number, door: Side, doorAt: number;
        if (f.axis === 'x') { x = near; w = depth; y = along; h = span; door = f.dir > 0 ? 'W' : 'E'; doorAt = h / 2; }
        else { y = near; h = depth; x = along; w = span; door = f.dir > 0 ? 'N' : 'S'; doorAt = w / 2; }
        if (t.spawn && x + w > t.x + 2.5 && x < t.x + 8.5 && y + h > t.y + 2 && y < t.y + 8.5) continue; // keep the spawn plaza props
        // reject lots touching streets, plazas or water
        let ok = true;
        for (let py = y; py < y + h && ok; py++) for (let px = x; px < x + w && ok; px++) {
          if (getStreetDetails(ctx.seed, px + 0.5, py + 0.5, ctx.spawn).kind !== 'none') ok = false;
        }
        if (!ok) continue;
        for (const [px, py] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h], [x + w / 2, y + h / 2]] as [number, number][]) {
          const s = ctx.sample(px, py);
          if (s.terrain === 'deep' || s.terrain === 'shallow') ok = false;
        }
        if (!ok) continue;
        if (out.some((o) => x < o.x + o.w + 1 && x + w + 1 > o.x && y < o.y + o.h + 1 && y + h + 1 > o.y)) continue; // overlapping town grids
        out.push({ id: `b:${ti}:${n++}`, def: def.id, x, y, w, h, door, doorAt, yard, front: frontGap, pieces: buildSnapPieces({ x, y, w, h, door, doorAt }, def.layout) });
      }
    }
  });
  townCache.set(ctx, out);
  return out;
}

/** Stage 11: town buildings. Each chunk owns buildings whose centre lies in it and clears scatter inside every footprint touching it. */
export const buildingStage: GenStage = {
  name: 'buildings',
  run(ctx, chunk) {
    const x0 = chunk.cx * CHUNK, y0 = chunk.cy * CHUNK;
    const touching = townBuildings(ctx).filter((b) => b.x - 0.6 < x0 + CHUNK && b.x + b.w + 0.6 > x0 && b.y - 0.6 < y0 + CHUNK && b.y + b.h + 0.6 > y0);
    if (!touching.length) return;
    for (const b of touching) {
      if (Math.floor((b.x + b.w / 2) / CHUNK) === chunk.cx && Math.floor((b.y + b.h / 2) / CHUNK) === chunk.cy) chunk.buildings.push({ ...b });
    }
    chunk.objects = chunk.objects.filter((o) => !touching.some((b) => o.x > b.x - 0.6 && o.x < b.x + b.w + 0.6 && o.y > b.y - 0.6 && o.y < b.y + b.h + 0.6));
  },
};

/** Stage 12: Ballistic Shooting Range & Weapons Proving Grounds. */
export const shootingRangeStage: GenStage = {
  name: 'shooting_range',
  run(ctx, chunk) {
    const rx = ctx.spawn.x - 24, ry = ctx.spawn.y - 12;
    const x0 = chunk.cx * CHUNK, y0 = chunk.cy * CHUNK;
    const x1 = x0 + CHUNK, y1 = y0 + CHUNK;

    // Check if chunk overlaps shooting range & Army Base Arena bounds [rx - 8, rx + 115] x [ry - 18, ry + 68]
    if (x1 < rx - 8 || x0 > rx + 115 || y1 < ry - 18 || y0 > ry + 68) return;

    // Clear natural scatter inside the shooting corridor lanes and Army Base Arena without wiping town plaza
    chunk.objects = chunk.objects.filter((o) => {
      // Don't clear anything if it's near spawn
      if (Math.hypot(o.x - ctx.spawn.x, o.y - ctx.spawn.y) < 14) return true;
      // Never clear industrial mineral nodes
      if (o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode') return true;
      // Precision range corridor
      const inPrecision = o.x >= rx - 4 && o.x <= rx + 108 && o.y >= ry - 4 && o.y <= ry + 4;
      // CQB corridor
      const inCqb = o.x >= rx + 4 && o.x <= rx + 32 && o.y >= ry - 11 && o.y <= ry - 5;
      // Quarantine pen corridor
      const inPens = o.x >= rx + 12 && o.x <= rx + 40 && o.y >= ry + 5 && o.y <= ry + 12;
      // Army Base Combat Arena zone
      const inArena = o.x >= rx + 4 && o.x <= rx + 74 && o.y >= ry + 18 && o.y <= ry + 64;
      return !(inPrecision || inCqb || inPens || inArena);
    });

    // Ensure terrain under Army Base Combat Arena is dry, stable ground
    for (let ty = 0; ty < CHUNK; ty++) {
      for (let tx = 0; tx < CHUNK; tx++) {
        const wx = chunk.cx * CHUNK + tx;
        const wy = chunk.cy * CHUNK + ty;
        if (wx >= rx + 4 && wx <= rx + 74 && wy >= ry + 18 && wy <= ry + 64) {
          const idx = ty * CHUNK + tx;
          const currentTerrain = ctx.terrainOrder[chunk.terrain[idx] ?? 0];
          if (currentTerrain === 'deep' || currentTerrain === 'shallow' || currentTerrain === 'mud') {
            const dirtIdx = ctx.terrainOrder.indexOf('dirt');
            chunk.terrain[idx] = dirtIdx >= 0 ? dirtIdx : 0;
          }
        }
      }
    }

    const add = (kind: ObjectKind, x: number, y: number): void => {
      if (x >= x0 && x < x1 && y >= y0 && y < y1) {
        if (getStreetDetails(ctx.seed, x, y, ctx.spawn).kind !== 'none') return;
        const def = ctx.content.objects.get(kind);
        chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:sr${chunk.objects.length}`, kind, x, y, variant: 0, hp: def?.hp ?? 1 });
      }
    };

    // 1. Firing Line Command Deck & Covered Benches
    add('street_light', rx + 0.5, ry - 3.2);
    add('street_light', rx + 0.5, ry + 3.2);
    add('park_bench', rx + 0.5, ry - 1.2);
    add('park_bench', rx + 0.5, ry + 1.2);
    add('crate', rx + 1.0, ry - 2.0);
    add('crate', rx + 1.0, ry + 2.0);
    add('sandbags', rx + 1.8, ry - 3.4);
    add('sandbags', rx + 3.0, ry - 3.4);
    add('sandbags', rx + 1.8, ry + 3.4);
    add('sandbags', rx + 3.0, ry + 3.4);

    // 2. Open Precision Target Lanes (5m, 10m, 15m, 20m, 25m, 30m, 40m, 50m, 60m, 75m, 100m)
    // Lane 5m
    add('target_dummy', rx + 8.0, ry - 2.0);
    add('sandbags', rx + 9.4, ry - 2.0);
    // Lane 10m
    add('steel_target', rx + 13.0, ry + 2.0);
    add('sandbags', rx + 14.4, ry + 2.0);
    // Lane 15m
    add('target_dummy', rx + 18.0, ry - 1.5);
    add('sandbags', rx + 19.4, ry - 1.5);
    // Lane 20m
    add('steel_target', rx + 23.0, ry + 1.5);
    add('sandbags', rx + 24.4, ry + 1.5);
    // Lane 25m
    add('target_dummy', rx + 28.0, ry - 2.2);
    add('sandbags', rx + 29.4, ry - 2.2);
    // Lane 30m
    add('steel_target', rx + 33.0, ry + 2.2);
    add('sandbags', rx + 34.4, ry + 2.2);
    // Lane 40m
    add('target_dummy', rx + 43.0, ry - 1.0);
    add('rock', rx + 44.8, ry - 1.0);
    // Lane 50m
    add('steel_target', rx + 53.0, ry + 1.0);
    add('rock', rx + 54.8, ry + 1.0);
    // Lane 60m
    add('target_dummy', rx + 63.0, ry - 1.8);
    add('rock', rx + 64.8, ry - 1.8);
    // Lane 75m
    add('steel_target', rx + 78.0, ry + 1.8);
    add('sandbags', rx + 79.6, ry + 1.8);
    // Lane 100m (Extreme Sniper Apex)
    add('target_dummy', rx + 103.0, ry);
    add('sandbags', rx + 104.6, ry - 0.8);
    add('sandbags', rx + 104.6, ry + 0.8);
    add('street_light', rx + 105.0, ry);

    // 3. CQB Barricade / Killhouse Lane (Parallel South Corridor ry - 8)
    const cqbY = ry - 8;
    add('sandbags', rx + 6.0, cqbY - 1.2);
    add('sandbags', rx + 6.0, cqbY + 1.2);
    add('target_dummy', rx + 10.0, cqbY);
    add('sandbags', rx + 12.0, cqbY + 1.0);
    add('steel_target', rx + 15.0, cqbY - 1.0);
    add('crate', rx + 17.0, cqbY);
    add('target_dummy', rx + 21.0, cqbY + 1.2);
    add('sandbags', rx + 25.0, cqbY - 1.2);
    add('steel_target', rx + 28.0, cqbY);

    // 4. Quarantined Zombie Containment Pens (North Observation Deck ry + 8 to ry + 16)
    // Pen A: Walkers (rx + 16, ry + 8)
    const pAx = rx + 16, pAy = ry + 8;
    for (let dx = -2.5; dx <= 2.5; dx += 1.25) {
      add('guardrail', pAx + dx, pAy - 2.5);
      add('guardrail', pAx + dx, pAy + 2.5);
    }
    for (let dy = -1.5; dy <= 1.5; dy += 1.25) {
      add('guardrail', pAx - 2.5, pAy + dy);
      add('guardrail', pAx + 2.5, pAy + dy);
    }

    // Pen B: Runners & Brutes (rx + 26, ry + 8)
    const pBx = rx + 26, pBy = ry + 8;
    for (let dx = -2.5; dx <= 2.5; dx += 1.25) {
      add('guardrail', pBx + dx, pBy - 2.5);
      add('guardrail', pBx + dx, pBy + 2.5);
    }
    for (let dy = -1.5; dy <= 1.5; dy += 1.25) {
      add('guardrail', pBx - 2.5, pBy + dy);
      add('guardrail', pBx + 2.5, pBy + dy);
    }

    // Pen C: Screamer Isolation Cell (rx + 36, ry + 8)
    const pCx = rx + 36, pCy = ry + 8;
    for (let dx = -2.0; dx <= 2.0; dx += 1.25) {
      add('guardrail', pCx + dx, pCy - 2.0);
      add('guardrail', pCx + dx, pCy + 2.0);
    }
    for (let dy = -1.0; dy <= 1.0; dy += 1.25) {
      add('guardrail', pCx - 2.0, pCy + dy);
      add('guardrail', pCx + 2.0, pCy + dy);
    }

    if (rx >= x0 && rx < x1 && ry >= y0 && ry < y1) {
      chunk.features.push({ type: 'shooting_range', x: rx, y: ry, name: "Bro's Ballistics Proving Grounds" });
    }
  },
};

export const DEFAULT_STAGES: GenStage[] = [terrainStage, roadStage, scatterStage, campStage, settlementStage, gasStationStage, radioTowerStage, townStage, militaryStage, mineralGeodeStage, buildingStage, parkedCarsStage, shootingRangeStage];

export class WorldGen {
  readonly seed: number;
  readonly terrainOrder: Terrain[];
  readonly spawn: { x: number; y: number };
  private nElev: (x: number, y: number) => number;
  private nTemp: (x: number, y: number) => number;
  private nMoist: (x: number, y: number) => number;
  private nPatch: (x: number, y: number) => number;
  private ctx: GenContext;

  constructor(seed: number, private content: Content, private stages: GenStage[] = DEFAULT_STAGES) {
    this.seed = seed;
    this.terrainOrder = [...content.terrain.keys()];
    this.nElev = makeNoise(seed ^ 0x1111); this.nTemp = makeNoise(seed ^ 0x2222);
    this.nMoist = makeNoise(seed ^ 0x3333); this.nPatch = makeNoise(seed ^ 0x4444);
    this.spawn = { x: 0, y: 0 };
    waterBySeed.set(seed, (x, y) => this.sample(x, y).terrain === 'deep');
    this.ctx = { seed, content, terrainOrder: this.terrainOrder, sample: (x, y) => this.sample(x, y), rng: (cx, cy, label) => makeRng(hash2(seed, cx * 7919 + label.length, cy * 104729 + label.charCodeAt(0))), spawn: this.spawn };
    const sp = this.findSpawn();
    this.spawn.x = sp.x; this.spawn.y = sp.y;
  }

  sample(x: number, y: number): Sample {
    const e = fbm(this.nElev, x * 0.0075, y * 0.0075, 4);
    const height = clamp((e - 0.5) * 1.7 + 0.5, 0, 1);
    // Large climate zones (HD plan G1): biomes span hundreds of tiles; height only nudges the pick.
    const temp = clamp((fbm(this.nTemp, x * BIOME_FREQ + 40, y * BIOME_FREQ + 40, 3) - 0.5) * 1.9 + 0.5, 0, 1);
    const moist = clamp((fbm(this.nMoist, x * BIOME_FREQ * 1.1 - 80, y * BIOME_FREQ * 1.1 - 80, 3) - 0.5) * 1.9 + 0.5, 0, 1);
    let best = 0, bestD = Infinity;
    this.content.biomes.forEach((b, i) => {
      const d = (b.climate.temp - temp) ** 2 + (b.climate.moist - moist) ** 2 + BIOME_ELEV_WEIGHT * (b.climate.elev - height) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    const biome = this.content.biomes[best] as BiomeDef;
    let terrain: Terrain;
    if (height < WATER_DEEP) terrain = 'deep';
    else if (height < WATER_SHALLOW) terrain = 'shallow';
    else {
      // fbm keeps big patches but gives them curvy, organic edges.
      const p = clamp((fbm(this.nPatch, x * PATCH_FREQ, y * PATCH_FREQ, 3) - 0.5) * 1.8 + 0.5, 0, 1);
      terrain = p > 1 - biome.terrain.patchAmount * 1.6 ? biome.terrain.patch : biome.terrain.base;
      if (terrain === 'deep' || terrain === 'shallow') terrain = biome.terrain.base;
    }
    return { height, temp, moist, biome: best, terrain };
  }

  private findSpawn(): { x: number; y: number } {
    const meadow = this.content.biomes.findIndex((b) => b.id === 'bro_meadow');
    for (let r = 0; r < 600; r += 4) {
      for (let a = 0; a < 16; a++) {
        const x = Math.cos((a / 16) * Math.PI * 2) * r, y = Math.sin((a / 16) * Math.PI * 2) * r;
        const s = this.sample(x, y);
        if (s.biome === meadow && s.terrain !== 'deep' && s.terrain !== 'shallow') return { x: Math.round(x) + 0.5, y: Math.round(y) + 0.5 };
      }
    }
    return { x: 0.5, y: 0.5 };
  }

  chunk(cx: number, cy: number): ChunkData {
    const c: ChunkData = { cx, cy, terrain: new Uint8Array(CHUNK * CHUNK), biome: new Uint8Array(CHUNK * CHUNK), elevation: new Uint8Array(CHUNK * CHUNK), objects: [], features: [], buildings: [] };
    for (const st of this.stages) st.run(this.ctx, c);
    return c;
  }
}
