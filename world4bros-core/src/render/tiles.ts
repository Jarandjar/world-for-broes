import { makeRng, hash2 } from '../engine/rng';
import { makeNoise, fbm } from '../engine/noise';
import { type G, INK, TAU, alpha } from './ink';
import { CHUNK, REGIONAL_TOWNS, getStreetDetails, roadPathsNear, type Building, type ChunkData, type StreetDetails } from '../sim/worldgen';
import { drawRoadRibbons } from './roadRibbons';
import type { GameWorld } from '../sim/world';
import type { TerrainDef } from '../data/schema';
import { paintGroundField } from './groundField';
import { GROUND_DETAILS, SURFACES, lodFor, paintSurfaceBase, surfaceOf, type SurfaceId, type TileCtx } from './terrainRenderer';

/** How far around a building the ground is calmed to the town ground (tiles). */
const TOWN_GROUND_REACH = 10;
/** Ground types a town keeps as-is; everything else under a town becomes grass. Water is never replaced. */
const TOWN_KEEPS = new Set(['sand', 'snow']);
const KEEP_NATURAL = new Set(['shallow', 'deep']);
const townGroundCache = new WeakMap<GameWorld, Map<string, TerrainDef>>();

/** The calm ground used near town buildings: the ground at the nearest town centre if it is sand or snow, else grass. */
function townGround(world: GameWorld, near: readonly Building[], natural: TerrainDef, wx: number, wy: number): TerrainDef | null {
  if (KEEP_NATURAL.has(natural.id) || !near.length) return null;
  if (!near.some((b) => {
    const dx = Math.max(b.x - wx, 0, wx - (b.x + b.w)), dy = Math.max(b.y - wy, 0, wy - (b.y + b.h));
    return Math.hypot(dx, dy) <= TOWN_GROUND_REACH;
  })) return null;
  const centres = [world.spawn, ...REGIONAL_TOWNS];
  let best = centres[0]!, bestD = Infinity;
  for (const c of centres) { const d = Math.hypot(c.x - wx, c.y - wy); if (d < bestD) { bestD = d; best = c; } }
  let cache = townGroundCache.get(world);
  if (!cache) { cache = new Map(); townGroundCache.set(world, cache); }
  const key = `${Math.round(best.x)},${Math.round(best.y)}`;
  let ground = cache.get(key);
  if (!ground) {
    const atCentre = world.terrainDefAt(best.x, best.y);
    ground = TOWN_KEEPS.has(atCentre.id) ? atCentre : world.content.terrain.get('grass' as never) as TerrainDef;
    cache.set(key, ground);
  }
  return ground;
}

/** Paints one chunk of ground into a canvas (1 unit = 1 tile). Cached by the renderer. */
export function paintChunk(world: GameWorld, chunk: ChunkData, px: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = CHUNK * px;
  const g = canvas.getContext('2d') as G;
  g.scale(px, px);
  const rng = makeRng(hash2(world.seed, chunk.cx, chunk.cy) ^ 0x51ed);
  const order = world.gen.terrainOrder, content = world.content;
  const lod = lodFor(px);

  const rawDef = (tx: number, ty: number): TerrainDef => {
    if (tx >= 0 && ty >= 0 && tx < CHUNK && ty < CHUNK) {
      return content.terrain.get(order[chunk.terrain[ty * CHUNK + tx] ?? 0] as never) as TerrainDef;
    }
    return world.terrainDefAt(chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5);
  };
  // Towns sit on one calm ground type instead of the biome's patchwork (HD plan G0).
  // Buildings that can calm any tile of this chunk (plus its border), looked up once.
  const half = CHUNK / 2;
  const nearBuildings = world.buildingsNear(chunk.cx * CHUNK + half, chunk.cy * CHUNK + half, half + 3 + TOWN_GROUND_REACH);
  const defMemo = new Map<number, TerrainDef>();
  const def = (tx: number, ty: number): TerrainDef => {
    const k = (ty + 4) * (CHUNK + 8) + (tx + 4);
    let d = defMemo.get(k);
    if (!d) {
      const n = rawDef(tx, ty);
      d = townGround(world, nearBuildings, n, chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5) ?? n;
      defMemo.set(k, d);
    }
    return d;
  };
  const elevation = (tx: number, ty: number): number => {
    if (tx >= 0 && ty >= 0 && tx < CHUNK && ty < CHUNK) {
      return (chunk.elevation[ty * CHUNK + tx] ?? 0) / 255;
    }
    return world.gen.sample(chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5).height;
  };

  // Street layout for the chunk plus a one-tile border, so kerbs and edges can see their neighbours.
  const S = CHUNK + 2;
  const streets: StreetDetails[] = new Array(S * S);
  const surfaces: (SurfaceId | null)[] = new Array(S * S);
  for (let ty = -1; ty <= CHUNK; ty++) {
    for (let tx = -1; tx <= CHUNK; tx++) {
      const st = getStreetDetails(world.seed, chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5, world.spawn);
      const id = def(tx, ty).id;
      const i = (ty + 1) * S + tx + 1;
      streets[i] = st;
      // Town streets keep tile surfaces; highways and dirt tracks are drawn as ribbons below.
      surfaces[i] = id === 'deep' || id === 'shallow' || !st.isTown ? null : surfaceOf(st);
    }
  }
  const surf = (tx: number, ty: number): SurfaceId | null => surfaces[(ty + 1) * S + tx + 1] ?? null;

  // 1. Smooth ground: blended ground types with curved boundaries, water depth, foam, wet shore and relief (HD plan G1).
  const nz = makeNoise(world.seed ^ 0xabc123);
  const biomeAt = (tx: number, ty: number): number => (tx >= 0 && ty >= 0 && tx < CHUNK && ty < CHUNK)
    ? chunk.biome[ty * CHUNK + tx] ?? 0
    : world.gen.sample(chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5).biome;
  const shallow = content.terrain.get('shallow' as never) as TerrainDef, deep = content.terrain.get('deep' as never) as TerrainDef;
  paintGroundField(g, {
    seed: world.seed, cx: chunk.cx, cy: chunk.cy,
    ground: def,
    height: elevation,
    tint: (tx, ty) => content.biomes[biomeAt(tx, ty)]?.tint ?? '#808080',
    water: [shallow.colors[2], shallow.colors[0], deep.colors[0], deep.colors[1]],
  });

  // 4. Natural ground detail (blades, tufts, flowers, pebbles, tracks, puddles...), seeded per world tile.
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      if (surf(tx, ty)) continue;
      const d = def(tx, ty);
      const trng = makeRng(hash2(world.seed ^ 0x9e3779b9, chunk.cx * CHUNK + tx, chunk.cy * CHUNK + ty));
      GROUND_DETAILS[d.strokes]({ g, rng: trng, lod, tx, ty, d });
    }
  }

  // 5a. Highways and dirt tracks as smooth ribbons (HD plan G2). Whole paths are drawn; the canvas clips them,
  // and identical geometry in every chunk keeps the seams aligned.
  const ox = chunk.cx * CHUNK, oy = chunk.cy * CHUNK;
  const ribbons = roadPathsNear(world.seed, world.spawn, ox, oy, ox + CHUNK, oy + CHUNK, 5);
  if (ribbons.length) {
    g.save();
    g.translate(-ox, -oy);
    drawRoadRibbons(g, ribbons, content.terrain.get('dirt' as never) as TerrainDef, content.terrain.get('grass' as never) as TerrainDef);
    g.restore();
  }

  // 5. Road surfaces: all bases first (their oversize would cover neighbours' kerbs), then per-tile detail.
  const ctxs: TileCtx[] = [];
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      const id = surf(tx, ty);
      if (!id) continue;
      const wx = chunk.cx * CHUNK + tx, wy = chunk.cy * CHUNK + ty;
      const c: TileCtx = {
        g, lod, tx, ty, wx, wy, id,
        rng: makeRng(hash2(world.seed ^ 0x2545f491, wx, wy)),
        st: streets[(ty + 1) * S + tx + 1] as StreetDetails,
        look: SURFACES[id].look,
        tone: fbm(nz, wx * 0.21, wy * 0.21, 2) - 0.5,
        near: (dx, dy) => (Math.abs(dx) <= 1 && Math.abs(dy) <= 1 ? surf(tx + dx, ty + dy) : surfaceOf(getStreetDetails(world.seed, wx + dx + 0.5, wy + dy + 0.5, world.spawn))),
        ground: (dx, dy) => def(tx + dx, ty + dy),
      };
      paintSurfaceBase(c);
      ctxs.push(c);
    }
  }
  for (const c of ctxs) SURFACES[c.id].paint(c);

  // 6. Subtle paper speckle (shorelines and biome tint are part of the smooth ground pass)
  for (let i = 0; i < 220; i++) {
    g.fillStyle = alpha(rng.chance(0.6) ? INK : '#efe8d6', rng.range(0.04, 0.14));
    g.beginPath();
    g.arc(rng.range(0, CHUNK), rng.range(0, CHUNK), rng.range(0.006, 0.018), 0, TAU);
    g.fill();
  }

  return canvas;
}
