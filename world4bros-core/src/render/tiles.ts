import { makeRng, hash2 } from '../engine/rng';
import { makeNoise, fbm } from '../engine/noise';
import { type G, INK, TAU, alpha, mix, stroke } from './ink';
import { CHUNK, getStreetDetails, type ChunkData, type StreetDetails } from '../sim/worldgen';
import type { GameWorld } from '../sim/world';
import type { TerrainDef } from '../data/schema';
import { GROUND_DETAILS, SURFACES, lodFor, paintSurfaceBase, surfaceOf, type SurfaceId, type TileCtx } from './terrainRenderer';

/** Paints one chunk of ground into a canvas (1 unit = 1 tile). Cached by the renderer. */
export function paintChunk(world: GameWorld, chunk: ChunkData, px: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = CHUNK * px;
  const g = canvas.getContext('2d') as G;
  g.scale(px, px);
  const rng = makeRng(hash2(world.seed, chunk.cx, chunk.cy) ^ 0x51ed);
  const order = world.gen.terrainOrder, content = world.content;
  const lod = lodFor(px);

  const def = (tx: number, ty: number): TerrainDef => {
    if (tx >= 0 && ty >= 0 && tx < CHUNK && ty < CHUNK) {
      return content.terrain.get(order[chunk.terrain[ty * CHUNK + tx] ?? 0] as never) as TerrainDef;
    }
    return world.terrainDefAt(chunk.cx * CHUNK + tx + 0.5, chunk.cy * CHUNK + ty + 0.5);
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
      surfaces[i] = id === 'deep' || id === 'shallow' ? null : surfaceOf(st);
    }
  }
  const surf = (tx: number, ty: number): SurfaceId | null => surfaces[(ty + 1) * S + tx + 1] ?? null;

  // 1. Base fill: smooth world-space noise picks the tone for natural, seamless blending
  const nz = makeNoise(world.seed ^ 0xabc123);
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      const d = def(tx, ty);
      const wx = chunk.cx * CHUNK + tx, wy = chunk.cy * CHUNK + ty;
      const n = fbm(nz, wx * 0.14, wy * 0.14, 2) - 0.5;
      g.fillStyle = mix(d.colors[0], n > 0 ? d.colors[2] : d.colors[1], Math.min(0.55, Math.abs(n) * 1.7));
      g.fillRect(tx - 0.04, ty - 0.04, 1.08, 1.08);
    }
  }

  // 2. Ragged organic feathering where different terrain patches meet
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      const d = def(tx, ty);
      for (const [nx, ny, vertical] of [[tx + 1, ty, true], [tx, ty + 1, false]] as [number, number, boolean][]) {
        const o = def(nx, ny);
        if (o.id === d.id) continue;
        for (let k = 0; k < 8; k++) {
          const t = (k + rng.next()) / 8;
          const off = rng.range(-0.36, 0.36);
          const side = rng.chance(0.5) ? d : o;
          const bx = vertical ? tx + 1 + off : tx + t;
          const by = vertical ? ty + t : ty + 1 + off;
          g.fillStyle = mix(side.colors[0], side.colors[rng.chance(0.5) ? 1 : 2], rng.range(0, 0.35));
          g.beginPath();
          g.ellipse(bx, by, rng.range(0.16, 0.4), rng.range(0.14, 0.34), rng.range(0, 3), 0, TAU);
          g.fill();
        }
      }
    }
  }

  // 3. Soft organic ground blotches to eliminate any remaining tile grid artifacts
  for (let i = 0; i < 140; i++) {
    const tx = rng.int(0, CHUNK - 1), ty = rng.int(0, CHUNK - 1), d = def(tx, ty);
    g.fillStyle = alpha(rng.chance(0.5) ? d.colors[1] : d.colors[2], rng.range(0.09, 0.22));
    g.beginPath();
    g.ellipse(tx + rng.next(), ty + rng.next(), rng.range(0.9, 2.2), rng.range(0.8, 1.9), rng.range(0, 3), 0, TAU);
    g.fill();
  }

  // Elevation bands and steep transitions add readable relief without changing tile movement rules.
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      const h = elevation(tx, ty);
      const band = Math.floor(h * 8);
      const east = elevation(tx + 1, ty);
      const south = elevation(tx, ty + 1);
      if (Math.floor(east * 8) !== band || Math.abs(east - h) >= 0.08) {
        g.strokeStyle = alpha(east < h ? '#17231b' : '#f4ead0', Math.abs(east - h) >= 0.08 ? 0.38 : 0.20);
        g.lineWidth = Math.abs(east - h) >= 0.08 ? 0.09 : 0.045;
        g.beginPath();
        g.moveTo(tx + 0.98, ty + 0.1);
        g.lineTo(tx + 0.98, ty + 0.9);
        g.stroke();
      }
      if (Math.floor(south * 8) !== band || Math.abs(south - h) >= 0.08) {
        g.strokeStyle = alpha(south < h ? '#17231b' : '#f4ead0', Math.abs(south - h) >= 0.08 ? 0.38 : 0.20);
        g.lineWidth = Math.abs(south - h) >= 0.08 ? 0.09 : 0.045;
        g.beginPath();
        g.moveTo(tx + 0.1, ty + 0.98);
        g.lineTo(tx + 0.9, ty + 0.98);
        g.stroke();
      }
    }
  }

  // 4. Natural ground detail (blades, tufts, flowers, pebbles, tracks, puddles...), seeded per world tile.
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      if (surf(tx, ty)) continue;
      const d = def(tx, ty);
      const trng = makeRng(hash2(world.seed ^ 0x9e3779b9, chunk.cx * CHUNK + tx, chunk.cy * CHUNK + ty));
      GROUND_DETAILS[d.strokes]({ g, rng: trng, lod, tx, ty, d });
    }
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

  // 6. Shorelines & water boundary foam contours
  for (let ty = 0; ty < CHUNK; ty++) {
    for (let tx = 0; tx < CHUNK; tx++) {
      const d = def(tx, ty);
      for (const [nx, ny, x0, y0, x1, y1] of [[tx + 1, ty, tx + 1, ty, tx + 1, ty + 1], [tx, ty + 1, tx, ty + 1, tx + 1, ty + 1]] as [number, number, number, number, number, number][]) {
        const o = def(nx, ny);
        if (o.id === d.id) continue;
        const water = d.id === 'deep' || o.id === 'deep' || d.id === 'shallow' || o.id === 'shallow';
        const pts: [number, number][] = [];
        for (let k = 0; k <= 4; k++) {
          const t = k / 4;
          pts.push([
            x0 + (x1 - x0) * t + (y0 === y1 ? 0 : rng.range(-0.04, 0.04)),
            y0 + (y1 - y0) * t + (x0 === x1 ? 0 : rng.range(-0.04, 0.04))
          ]);
        }
        if (water) {
          // Rich ink shoreline contour & foamy wave lip
          stroke(g, pts, 0.048, INK, 0.45);
          stroke(g, pts.map(([x, y]) => [x + (x0 === x1 ? -0.06 : 0), y + (y0 === y1 ? -0.06 : 0)] as [number, number]), 0.016, '#67e8f9', 0.6);
        }
      }
    }
  }

  // 7. Atmospheric Biome Tint & subtle paper speckle
  for (let ty = 0; ty < CHUNK; ty += 2) {
    for (let tx = 0; tx < CHUNK; tx += 2) {
      const b = content.biomes[chunk.biome[ty * CHUNK + tx] ?? 0];
      if (b) {
        g.fillStyle = alpha(b.tint, 0.08);
        g.fillRect(tx, ty, 2, 2);
      }
    }
  }
  for (let i = 0; i < 220; i++) {
    g.fillStyle = alpha(rng.chance(0.6) ? INK : '#efe8d6', rng.range(0.04, 0.14));
    g.beginPath();
    g.arc(rng.range(0, CHUNK), rng.range(0, CHUNK), rng.range(0.006, 0.018), 0, TAU);
    g.fill();
  }

  return canvas;
}
