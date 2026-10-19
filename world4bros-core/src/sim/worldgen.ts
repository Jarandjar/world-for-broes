import { makeNoise, fbm } from '../engine/noise';
import { makeRng, rand2, hash2, type Rng } from '../engine/rng';
import { clamp } from '../engine/math';
import type { BiomeDef, Content, ObjectKind, Terrain } from '../data/schema';

export const CHUNK = 16;
export const POWER_POLE_SPACING = 6;

export interface WorldObject { id: string; kind: ObjectKind; x: number; y: number; variant: number; hp: number; removed?: boolean }
export interface Feature { type: string; x: number; y: number; name?: string }
export type Side = 'N' | 'E' | 'S' | 'W';
/** A walled building. (x,y) is the top-left tile, w/h are outer size in tiles. `doorAt` is the door centre measured along the door's wall. */
/** A placed building. `yard`/`front` (town lots only): free tiles behind the back wall and from the front wall to the street edge. */
export interface Building { id: string; def: string; x: number; y: number; w: number; h: number; door: Side; doorAt: number; yard?: number; front?: number }
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

const WATER_DEEP = 0.17, WATER_SHALLOW = 0.215;

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

export interface StreetDetails {
  kind: RoadKind;
  isTown: boolean;
  isSidewalk: boolean;
  isCrosswalk: boolean;
  isPlaza: boolean;
  axis: 'x' | 'y' | 'both';
}

export const REGIONAL_TOWNS = [
  { name: 'Rust Creek Junction', x: -72, y: 80 },
  { name: 'Port Seagull Harbor', x: 80, y: -96 },
  { name: 'Pine Ridge Outpost', x: 96, y: 64 },
  { name: 'Deadwood Oasis Hub', x: -88, y: -80 },
  { name: 'Copperfield Metropolis', x: 120, y: -32 },
];

export function getStreetDetails(seed: number, x: number, y: number, spawn?: { x: number; y: number }): StreetDetails {
  const spX = spawn ? spawn.x : 0;
  const spY = spawn ? spawn.y : 0;

  // 1. Check Town Centers (Metropolis Oakhaven at spawn & Regional Town hubs)
  const townCenters = [{ x: spX, y: spY }, ...REGIONAL_TOWNS];
  for (const t of townCenters) {
    const dx = x - t.x, dy = y - t.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= 46) {
      // Civic Plaza around fountain
      const plazaDist = Math.hypot(dx + 3.0, dy + 2.0);
      if (plazaDist < 4.2 && dx < -1.0) {
        return { kind: 'small_road', isTown: true, isSidewalk: false, isCrosswalk: false, isPlaza: true, axis: 'both' };
      }

      // Town Main Street (East-West Arterial Avenue)
      const onMainSt = Math.abs(dy) < 2.3 && Math.abs(dx) <= 44;
      // Town Central Boulevard (North-South Arterial Avenue)
      const onCentralBlvd = Math.abs(dx) < 2.3 && Math.abs(dy) <= 44;

      if (onMainSt || onCentralBlvd) {
        const isSidewalk = onMainSt && !onCentralBlvd ? Math.abs(dy) >= 1.55 : onCentralBlvd && !onMainSt ? Math.abs(dx) >= 1.55 : false;
        // Pedestrian Zebra Crosswalk at intersections
        const isCrosswalk = (Math.abs(dx) < 2.3 && Math.abs(dy) < 2.3) ||
          (onMainSt && (Math.abs(Math.abs(dx) - 16) < 1.2 || Math.abs(Math.abs(dx) - 32) < 1.2)) ||
          (onCentralBlvd && (Math.abs(Math.abs(dy) - 16) < 1.2 || Math.abs(Math.abs(dy) - 32) < 1.2));

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
      const onCrossX = (Math.abs(Math.abs(dx) - 16) < 1.2 || Math.abs(Math.abs(dx) - 32) < 1.2) && Math.abs(dy) <= 40;
      const onCrossY = (Math.abs(Math.abs(dy) - 16) < 1.2 || Math.abs(Math.abs(dy) - 32) < 1.2) && Math.abs(dx) <= 40;
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

  // 2. Continental Organic Highways (Primary Arterial Corridors)
  const sx = Math.sin(x * 0.025 + (seed % 100) * 0.1) * 3.8;
  const sy = Math.cos(y * 0.025 + (seed % 100) * 0.15) * 3.8;
  const modY = Math.abs(((y + sx) % 96 + 96) % 96 - 48);
  const modX = Math.abs(((x + sy) % 96 + 96) % 96 - 48);
  if (modY < 1.7 || modX < 1.7) {
    return {
      kind: 'highway',
      isTown: false,
      isSidewalk: false,
      isCrosswalk: false,
      isPlaza: false,
      axis: modY < 1.7 && modX < 1.7 ? 'both' : modY < 1.7 ? 'x' : 'y'
    };
  }

  // 3. Rural Dirt Roads (Secondary Connectors & Country Byways)
  const dCurveX = Math.sin(x * 0.04 + (seed % 50) * 0.12) * 3.2;
  const dCurveY = Math.cos(y * 0.04 + (seed % 50) * 0.12) * 3.2;
  const dModY = Math.abs(((y + 48 + dCurveX) % 96 + 96) % 96 - 48);
  const dModX = Math.abs(((x + 48 + dCurveY) % 96 + 96) % 96 - 48);
  if (dModY < 1.3 || dModX < 1.3) {
    return {
      kind: 'dirt_road',
      isTown: false,
      isSidewalk: false,
      isCrosswalk: false,
      isPlaza: false,
      axis: dModY < 1.3 && dModX < 1.3 ? 'both' : dModY < 1.3 ? 'x' : 'y'
    };
  }

  // 4. Wilderness Nature Trails & Cobblestone Byways
  const trailCurve = Math.sin(x * 0.05 + y * 0.03 + (seed % 70) * 0.1) * 2.6;
  const trailMod = Math.abs(((x - y + trailCurve) % 96 + 96) % 96 - 48);
  if (trailMod < 0.95) {
    return {
      kind: 'small_road',
      isTown: false,
      isSidewalk: false,
      isCrosswalk: false,
      isPlaza: false,
      axis: 'both'
    };
  }

  return { kind: 'none', isTown: false, isSidewalk: false, isCrosswalk: false, isPlaza: false, axis: 'both' };
}

export function getRoadType(seed: number, x: number, y: number, spawn?: { x: number; y: number }): RoadKind {
  return getStreetDetails(seed, x, y, spawn).kind;
}

export function isRoad(seed: number, x: number, y: number, spawn?: { x: number; y: number }): boolean {
  return getRoadType(seed, x, y, spawn) !== 'none';
}

/** Stage: Road network carving. Sets dirt terrain and connects regions. */
export const roadStage: GenStage = {
  name: 'roads',
  run(ctx, chunk) {
    const dirtIdx = ctx.terrainOrder.indexOf('dirt');
    if (dirtIdx < 0) return;
    for (let ty = 0; ty < CHUNK; ty++) {
      for (let tx = 0; tx < CHUNK; tx++) {
        const wx = chunk.cx * CHUNK + tx + 0.5, wy = chunk.cy * CHUNK + ty + 0.5;
        const curTerrain = ctx.terrainOrder[chunk.terrain[ty * CHUNK + tx] ?? 0];
        if (curTerrain === 'deep' || curTerrain === 'shallow') continue;
        if (isRoad(ctx.seed, wx, wy, ctx.spawn)) {
          chunk.terrain[ty * CHUNK + tx] = dirtIdx;
        }
      }
    }
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

/** Stage 7: Abandoned Big Towns & Metropolis Hubs with streets, squares, cars, and districts. */
export const townStage: GenStage = {
  name: 'towns',
  run(ctx, chunk) {
    const scx = Math.floor(ctx.spawn.x / CHUNK), scy = Math.floor(ctx.spawn.y / CHUNK);
    const isSpawnChunk = (chunk.cx === scx && chunk.cy === scy);
    const isSpawnDistrict = Math.abs(chunk.cx - scx) <= 1 && Math.abs(chunk.cy - scy) <= 1;

    // 1. Capitol Big Town at Spawn (Metropolis Oakhaven - 3x3 Chunks)
    if (isSpawnChunk) {
      const tx = ctx.spawn.x, ty = ctx.spawn.y;
      chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - tx, o.y - ty) > 7.2);
      const r = ctx.rng(chunk.cx, chunk.cy, 'town_spawn');
      const add = (kind: ObjectKind, dx: number, dy: number): void => {
        const def = ctx.content.objects.get(kind);
        chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:tw${chunk.objects.length}`, kind, x: tx + dx, y: ty + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
      };
      // Historic town items & natural nodes strictly within chunk bounds and within 10m
      add('pine', 4.0, 3.0);
      add('rock', 5.0, 3.0);
      add('wrecked_car', 4.5, 4.0);
      add('crate', 4.0, 3.5);
      add('fuel_pump', 5.5, 5.0);
      add('tent', 6.0, 4.0);
      add('log', 5.0, 4.2);
      // Municipal & road infrastructure in the civic plaza within 10m
      add('town_fountain', 5.0, 5.5);
      add('park_bench', 4.2, 5.5);
      add('street_light', 5.0, 7.0);
      add('dumpster', 6.0, 6.0);
      add('fire_hydrant', 3.5, 5.0);
      add('guardrail', 3.2, 6.0);
      add('traffic_cone', 4.2, 6.5);
      chunk.features.push({ type: 'town', x: tx, y: ty, name: 'Metropolis Oakhaven' });
      return;
    }

    if (isSpawnDistrict) {
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

    // 2. Distant Continental Big Towns & Themed Settlements
    const R = 20;
    const rx = Math.floor(chunk.cx / R), ry = Math.floor(chunk.cy / R);
    if (rand2(ctx.seed + 61, rx, ry * 43 + 23) > 0.52) return;
    const ccx = rx * R + (hash2(ctx.seed + 67, rx, ry) % R);
    const ccy = ry * R + (hash2(ctx.seed + 71, rx, ry) % R);
    if (chunk.cx !== ccx || chunk.cy !== ccy) return;
    const x = chunk.cx * CHUNK + 8, y = chunk.cy * CHUNK + 8;
    const s = ctx.sample(x, y);
    if (s.terrain === 'deep' || s.terrain === 'shallow') return;
    if (Math.hypot(x - ctx.spawn.x, y - ctx.spawn.y) < 22) return;

    chunk.objects = chunk.objects.filter((o) => Math.hypot(o.x - x, o.y - y) > 7.5);
    const r = ctx.rng(chunk.cx, chunk.cy, 'town');
    const add = (kind: ObjectKind, dx: number, dy: number): void => {
      const def = ctx.content.objects.get(kind);
      chunk.objects.push({ id: `${chunk.cx},${chunk.cy}:tw${chunk.objects.length}`, kind, x: x + dx, y: y + dy, variant: r.int(0, 3), hp: def?.hp ?? 1 });
    };

    const townNames = ['Rust Creek Junction', 'Port Seagull Harbor', 'Pine Ridge Outpost', 'Deadwood Oasis Hub', 'Copperfield Metropolis'];
    const townName = townNames[hash2(ctx.seed, rx, ry) % townNames.length] ?? 'Regional Big Town';

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
    chunk.features.push({ type: 'town', x, y, name: townName });
  },
};

/** Stage 8: Fortified military checkpoints and outposts. */
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

/** Solid wall rectangles [x0,y0,x1,y1] of a building (the door is a gap). */
export function wallRects(b: Building): Rect[] {
  let r = wallCache.get(b);
  if (r) return r;
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

/** Lot blocks between town streets (local coords, one quadrant; mirrored by sign). Matches getStreetDetails. */
const BLOCK_SPANS: [number, number][] = [[3.4, 14.0], [18.0, 30.0], [34.0, 43.5]];
const STREET_LINES = [0, 16, 32];
const townCache = new WeakMap<GenContext, Building[]>();

/** Half-width of a town street line (Main Street / Central Blvd are wider). Matches getStreetDetails. */
const streetHalf = (line: number): number => (line === 0 ? 2.3 : 1.2);
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
  const towns = [{ x: ctx.spawn.x, y: ctx.spawn.y, spawn: true }, ...REGIONAL_TOWNS.map((t) => ({ x: t.x, y: t.y, spawn: false }))];
  if (defs.length) towns.forEach((t, ti) => {
    let n = 0;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const [a0, a1] of BLOCK_SPANS) for (const [b0, b1] of BLOCK_SPANS) {
      const bx0 = sx > 0 ? a0 : -a1, bx1 = sx > 0 ? a1 : -a0, by0 = sy > 0 ? b0 : -b1, by1 = sy > 0 ? b1 : -b0;
      const nx = Math.max(1, Math.round((bx1 - bx0) / 7.5)), ny = Math.max(1, Math.round((by1 - by0) / 7.5));
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
          const lin = Math.max(...STREET_LINES.filter((l) => l <= Math.abs(inner)));
          const gin = Math.abs(inner) - lin - streetHalf(lin);
          const cands: LotFront[] = [{ axis, edge: inner, back: outer, dir: s, gap: gin, score: gin - (lin === 0 ? 0.6 : 0) }];
          const lout = STREET_LINES.find((l) => l > 0 && l >= Math.abs(outer));
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
        out.push({ id: `b:${ti}:${n++}`, def: def.id, x, y, w, h, door, doorAt, yard, front: frontGap });
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

export const DEFAULT_STAGES: GenStage[] = [terrainStage, roadStage, scatterStage, campStage, settlementStage, gasStationStage, radioTowerStage, townStage, militaryStage, mineralGeodeStage, buildingStage, shootingRangeStage];

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
    this.ctx = { seed, content, terrainOrder: this.terrainOrder, sample: (x, y) => this.sample(x, y), rng: (cx, cy, label) => makeRng(hash2(seed, cx * 7919 + label.length, cy * 104729 + label.charCodeAt(0))), spawn: this.spawn };
    const sp = this.findSpawn();
    this.spawn.x = sp.x; this.spawn.y = sp.y;
  }

  sample(x: number, y: number): Sample {
    const e = fbm(this.nElev, x * 0.0075, y * 0.0075, 4);
    const height = clamp((e - 0.5) * 1.7 + 0.5, 0, 1);
    const temp = clamp((fbm(this.nTemp, x * 0.0038 + 40, y * 0.0038 + 40, 3) - 0.5) * 1.9 + 0.5, 0, 1);
    const moist = clamp((fbm(this.nMoist, x * 0.0042 - 80, y * 0.0042 - 80, 3) - 0.5) * 1.9 + 0.5, 0, 1);
    let best = 0, bestD = Infinity;
    this.content.biomes.forEach((b, i) => {
      const d = (b.climate.temp - temp) ** 2 + (b.climate.moist - moist) ** 2 + 0.8 * (b.climate.elev - height) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    const biome = this.content.biomes[best] as BiomeDef;
    let terrain: Terrain;
    if (height < WATER_DEEP) terrain = 'deep';
    else if (height < WATER_SHALLOW) terrain = 'shallow';
    else {
      const p = this.nPatch(x * 0.09, y * 0.09);
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
