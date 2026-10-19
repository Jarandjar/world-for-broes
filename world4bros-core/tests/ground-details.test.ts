import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeRng } from '../src/engine/rng';
import { TERRAIN } from '../src/data/terrain';
import { getStreetDetails } from '../src/sim/worldgen';
import type { G } from '../src/render/ink';
import {
  GROUND_DETAILS, SURFACES, SURFACE_LOOKS, SIDES, lodFor, paintSurfaceBase, surfaceOf,
  type Lod, type SurfaceId, type TileCtx,
} from '../src/render/terrainRenderer';

/** Fake 2D context that records every call and property write, so painters can run in Node. */
function recorder(): { g: G; log: string[] } {
  const log: string[] = [];
  const g = new Proxy({}, {
    get: (_t, k) => (...args: unknown[]) => { log.push(`${String(k)}(${args.map((a) => (typeof a === 'number' ? a.toFixed(4) : String(a))).join(',')})`); },
    set: (_t, k, v) => { log.push(`${String(k)}=${String(v)}`); return true; },
  }) as unknown as G;
  return { g, log };
}

const IDS = Object.keys(SURFACES) as SurfaceId[];

function ctx(g: G, id: SurfaceId, seed: number, lod: Lod, near: SurfaceId | null): TileCtx {
  return {
    g, lod, tx: 3, ty: 5, wx: 40, wy: 41, id,
    rng: makeRng(seed),
    st: { kind: 'highway', isTown: true, isSidewalk: id === 'sidewalk', isCrosswalk: id === 'crosswalk', isPlaza: id === 'plaza', axis: 'x' },
    look: SURFACES[id].look, tone: 0.2,
    near: () => near,
    ground: () => TERRAIN[0]!,
  };
}

test('every surface id has a look and a painter, and looks never use pure black', () => {
  for (const id of IDS) {
    assert.equal(SURFACES[id].look, SURFACE_LOOKS[id], id);
    for (const col of [SURFACE_LOOKS[id].base, SURFACE_LOOKS[id].dark, SURFACE_LOOKS[id].light]) assert.notEqual(col.toLowerCase(), '#000000', id);
  }
  assert.equal(SIDES.length, 4);
});

test('every terrain stroke kind has a ground painter', () => {
  for (const t of TERRAIN) assert.equal(typeof GROUND_DETAILS[t.strokes], 'function', t.id);
});

test('surfaceOf maps every street cell around a town to a registered surface', () => {
  const seen = new Set<SurfaceId>();
  for (let y = -50; y <= 50; y++) for (let x = -50; x <= 50; x++) {
    const id = surfaceOf(getStreetDetails(7, x + 0.5, y + 0.5, { x: 0.4, y: 0.4 }));
    if (id) { assert.ok(SURFACES[id], id); seen.add(id); }
  }
  for (const id of ['avenue', 'crosswalk', 'sidewalk', 'plaza', 'cobble'] as SurfaceId[]) assert.ok(seen.has(id), `town has ${id}`);
});

test('surface painters are deterministic per tile seed and differ between seeds', () => {
  for (const id of IDS) for (const lod of [0, 1, 2] as Lod[]) for (const near of [null, 'avenue', 'sidewalk'] as (SurfaceId | null)[]) {
    const a = recorder(), b = recorder(), c = recorder();
    for (const [r, seed] of [[a, 11], [b, 11], [c, 12]] as const) {
      const t = ctx(r.g, id, seed, lod, near);
      paintSurfaceBase(t); SURFACES[id].paint(t);
    }
    assert.deepEqual(a.log, b.log, `${id} lod${lod} deterministic`);
    assert.ok(a.log.length > 3, `${id} draws something`);
    if (lod === 2) assert.notEqual(JSON.stringify(a.log), JSON.stringify(c.log), `${id} varies by seed`);
  }
});

test('ground painters are deterministic and lighter when zoomed out', () => {
  for (const d of TERRAIN) {
    const run = (seed: number, lod: Lod): string[] => { const r = recorder(); GROUND_DETAILS[d.strokes]({ g: r.g, rng: makeRng(seed), lod, tx: 0, ty: 0, d }); return r.log; };
    assert.deepEqual(run(5, 2), run(5, 2), d.id);
    let far = 0, close = 0;
    for (let s = 0; s < 40; s++) { far += run(s, 0).length; close += run(s, 2).length; }
    assert.ok(far <= close, `${d.id} lod0 not heavier than lod2`);
  }
});

test('lodFor follows the renderer chunk resolutions', () => {
  assert.equal(lodFor(18), 0);
  assert.equal(lodFor(40), 2);
  assert.equal(lodFor(112), 2);
});
