/**
 * Road graph (HD plan G2).
 *
 * Towns are the nodes. Highways join them along a minimum spanning tree plus a couple of
 * short extra links, leaving each town from the end of its main street, and meander along
 * smooth Catmull-Rom curves. Dirt roads branch off town side streets toward the countryside.
 * Every road is a polyline; point queries use a spatial hash, so the sim (speed, scatter,
 * lot placement) and the renderer (ribbons) read the same geometry.
 * Pure and deterministic: no DOM, no Math.random.
 */
import { hash2, rand2 } from '../engine/rng';

export type GraphRoadKind = 'highway' | 'dirt_road';
export interface RoadPath { kind: GraphRoadKind; half: number; pts: [number, number][]; /** Cumulative length at each point, for dash phase. */ len: number[] }
export interface RoadHit { kind: GraphRoadKind; dist: number; half: number; axis: 'x' | 'y' }

/** A town as the road graph sees it: centre, where its main streets end, and where its side streets end. */
export interface RoadNode { x: number; y: number; radius: number; exits: [number, number, number, number][]; spurs: [number, number, number, number][] }

/** Road half-widths in tiles (1 tile = 1 m): a 6 m two-lane highway and a 3.6 m dirt track. */
export const HIGHWAY_HALF = 3.0, DIRT_HALF = 1.8;
const SAMPLE_STEP = 1.5;
const CELL = 16;

interface Net { paths: RoadPath[]; grid: Map<number, number[]> }
const key = (cx: number, cy: number): number => cx * 73856093 ^ cy * 19349663;

/** Smooth curve through control points: Chaikin corner cutting (never overshoots), then even resampling. */
function spline(ctrl: [number, number][]): [number, number][] {
  let pts = ctrl;
  for (let it = 0; it < 5; it++) {
    const next: [number, number][] = [pts[0]!];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i]!, [x1, y1] = pts[i + 1]!;
      next.push([x0 * 0.75 + x1 * 0.25, y0 * 0.75 + y1 * 0.25], [x0 * 0.25 + x1 * 0.75, y0 * 0.25 + y1 * 0.75]);
    }
    next.push(pts[pts.length - 1]!);
    pts = next;
  }
  // Moving-average passes spread any remaining corner over 20+ tiles; the ends stay put so roads still meet their streets.
  let even = resample(pts);
  for (let pass = 0; pass < 6; pass++) {
    const sm = even.map((p) => [p[0], p[1]] as [number, number]);
    for (let i = 1; i < even.length - 1; i++) {
      // The window narrows toward the ends, so the curve eases into the fixed end points instead of kinking.
      const w = Math.min(3, i, even.length - 1 - i);
      let sx = 0, sy = 0;
      for (let k = -w; k <= w; k++) { sx += even[i + k]![0]; sy += even[i + k]![1]; }
      sm[i] = [sx / (2 * w + 1), sy / (2 * w + 1)];
    }
    even = resample(sm);
  }
  return even;
}

function resample(pts: [number, number][]): [number, number][] {
  const out: [number, number][] = [pts[0]!];
  let carry = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i]!, [x1, y1] = pts[i + 1]!;
    const seg = Math.hypot(x1 - x0, y1 - y0);
    if (seg < 1e-6) continue;
    let t = SAMPLE_STEP - carry;
    while (t <= seg) { out.push([x0 + ((x1 - x0) * t) / seg, y0 + ((y1 - y0) * t) / seg]); t += SAMPLE_STEP; }
    carry = seg - (t - SAMPLE_STEP);
  }
  const last = pts[pts.length - 1]!, tail = out[out.length - 1]!;
  if (Math.hypot(last[0] - tail[0], last[1] - tail[1]) > SAMPLE_STEP * 0.4) out.push(last);
  else out[out.length - 1] = last;
  return out;
}

/** A meandering road from a to b that leaves a along (adx, ady) and arrives at b along (bdx, bdy). */
function meander(seed: number, a: [number, number], adir: [number, number], b: [number, number], bdir: [number, number], wobble: number, isWater: (x: number, y: number) => boolean): [number, number][] {
  // A straight-ish lead out of each end; long enough that the turn toward the far end is a gentle curve.
  const lead = Math.min(26, Math.max(10, Math.hypot(b[0] - a[0], b[1] - a[1]) * 0.22));
  const a1: [number, number] = [a[0] + adir[0] * lead, a[1] + adir[1] * lead];
  const b1: [number, number] = [b[0] + bdir[0] * lead, b[1] + bdir[1] * lead];
  const dx = b1[0] - a1[0], dy = b1[1] - a1[1], L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L, ny = dx / L;
  const n = Math.max(1, Math.round(L / 34));
  const ctrl: [number, number][] = [a, a1];
  for (let i = 1; i < n; i++) {
    const t = i / n;
    const bx = a1[0] + dx * t, by = a1[1] + dy * t;
    const amp = Math.min(wobble, L * 0.14);
    let off = (rand2(seed + i * 131, Math.round(bx), Math.round(by)) - 0.5) * 2 * amp;
    // Bend away from lakes when a control point would sit in deep water.
    for (let tries = 0; tries < 4 && isWater(bx + nx * off, by + ny * off); tries++) off = tries % 2 ? -off * 1.4 : -off + Math.sign(off || 1) * amp;
    ctrl.push([bx + nx * off, by + ny * off]);
  }
  ctrl.push(b1, b);
  return spline(ctrl);
}

function withLengths(kind: GraphRoadKind, half: number, pts: [number, number][]): RoadPath {
  const len = [0];
  for (let i = 1; i < pts.length; i++) len.push(len[i - 1]! + Math.hypot(pts[i]![0] - pts[i - 1]![0], pts[i]![1] - pts[i - 1]![1]));
  return { kind, half, pts, len };
}

/** Builds the highway and dirt-road network for a set of towns. */
export function buildRoadNet(seed: number, nodes: RoadNode[], isWater: (x: number, y: number) => boolean): RoadPath[] {
  const paths: RoadPath[] = [];
  if (nodes.length < 2) return paths;
  // Minimum spanning tree (Prim) by distance, plus up to two short extra links for loops.
  const dist = (i: number, j: number): number => Math.hypot(nodes[i]!.x - nodes[j]!.x, nodes[i]!.y - nodes[j]!.y);
  const inTree = new Set([0]), edges: [number, number][] = [];
  while (inTree.size < nodes.length) {
    let best: [number, number] | null = null, bd = Infinity;
    for (const i of inTree) for (let j = 0; j < nodes.length; j++) if (!inTree.has(j) && dist(i, j) < bd) { bd = dist(i, j); best = [i, j]; }
    edges.push(best!); inTree.add(best![1]);
  }
  const extra: [number, number, number][] = [];
  for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
    if (!edges.some(([a, b]) => (a === i && b === j) || (a === j && b === i))) extra.push([i, j, dist(i, j)]);
  }
  extra.sort((p, q) => p[2] - q[2]);
  for (const [i, j] of extra.slice(0, 2)) edges.push([i, j]);

  // Each highway leaves from the main-street end of each town that faces the other town.
  const pickExit = (from: RoadNode, to: RoadNode): [number, number, number, number] => {
    // The exit whose direction points most nearly at the other town, so the road never doubles back.
    let best = from.exits[0]!, bd = -Infinity;
    for (const e of from.exits) {
      const dx = to.x - e[0], dy = to.y - e[1], l = Math.hypot(dx, dy) || 1;
      const facing = (e[2] * dx + e[3] * dy) / l;
      if (facing > bd) { bd = facing; best = e; }
    }
    return best;
  };
  for (const [i, j] of edges) {
    const A = nodes[i]!, B = nodes[j]!, ea = pickExit(A, B), eb = pickExit(B, A);
    const pts = meander(seed + i * 7 + j * 13, [ea[0], ea[1]], [ea[2], ea[3]], [eb[0], eb[1]], [eb[2], eb[3]], 16, isWater);
    paths.push(withLengths('highway', HIGHWAY_HALF, pts));
  }

  // One to three dirt tracks per town, from side-street ends out into the countryside.
  nodes.forEach((t, ti) => {
    const want = 1 + (hash2(seed + 401, ti, 7) % 3);
    const spurs = [...t.spurs].sort((p, q) => (hash2(seed + 409, Math.round(p[0]), Math.round(p[1])) % 97) - (hash2(seed + 409, Math.round(q[0]), Math.round(q[1])) % 97));
    for (const s of spurs.slice(0, want)) {
      const length = 55 + (hash2(seed + 419, Math.round(s[0]), Math.round(s[1])) % 60);
      const turn = (rand2(seed + 421, Math.round(s[0]), Math.round(s[1])) - 0.5) * 1.1;
      const c = Math.cos(turn), sn = Math.sin(turn);
      const dx = s[2] * c - s[3] * sn, dy = s[2] * sn + s[3] * c;
      const end: [number, number] = [s[0] + dx * length, s[1] + dy * length];
      if (isWater(end[0], end[1])) continue;
      const pts = meander(seed + 433 + ti, [s[0], s[1]], [s[2], s[3]], end, [-dx, -dy], 9, isWater);
      paths.push(withLengths('dirt_road', DIRT_HALF, pts));
    }
  });
  return paths;
}

function index(paths: RoadPath[]): Map<number, number[]> {
  const grid = new Map<number, number[]>();
  paths.forEach((p, pi) => {
    for (let i = 0; i < p.pts.length - 1; i++) {
      const [x0, y0] = p.pts[i]!, [x1, y1] = p.pts[i + 1]!;
      const m = p.half + 1;
      for (let cy = Math.floor((Math.min(y0, y1) - m) / CELL); cy <= Math.floor((Math.max(y0, y1) + m) / CELL); cy++) {
        for (let cx = Math.floor((Math.min(x0, x1) - m) / CELL); cx <= Math.floor((Math.max(x0, x1) + m) / CELL); cx++) {
          const k = key(cx, cy);
          let list = grid.get(k);
          if (!list) { list = []; grid.set(k, list); }
          list.push(pi * 100000 + i);
        }
      }
    }
  });
  return grid;
}

const nets = new Map<string, Net>();
/** The road network for a world, built once and cached. */
export function roadNet(cacheKey: string, build: () => RoadPath[]): Net {
  let n = nets.get(cacheKey);
  if (!n) {
    const paths = build();
    n = { paths, grid: index(paths) };
    nets.set(cacheKey, n);
    if (nets.size > 8) nets.delete(nets.keys().next().value!);
  }
  return n;
}

/** The road (if any) whose surface covers (x, y). Highways win over dirt roads. */
export function roadHit(net: Net, x: number, y: number): RoadHit | null {
  const list = net.grid.get(key(Math.floor(x / CELL), Math.floor(y / CELL)));
  if (!list) return null;
  let best: RoadHit | null = null;
  for (const code of list) {
    const p = net.paths[Math.floor(code / 100000)]!, i = code % 100000;
    const [x0, y0] = p.pts[i]!, [x1, y1] = p.pts[i + 1]!;
    const sx = x1 - x0, sy = y1 - y0, l2 = sx * sx + sy * sy || 1;
    const t = Math.max(0, Math.min(1, ((x - x0) * sx + (y - y0) * sy) / l2));
    const d = Math.hypot(x - (x0 + sx * t), y - (y0 + sy * t));
    if (d > p.half) continue;
    const better = !best || (p.kind === 'highway' && best.kind !== 'highway') || (p.kind === best.kind && d < best.dist);
    if (better) best = { kind: p.kind, dist: d, half: p.half, axis: Math.abs(sx) >= Math.abs(sy) ? 'x' : 'y' };
  }
  return best;
}

/** Road paths that pass within `margin` of the box [x0,x1]×[y0,y1], for drawing. */
export function roadsInBox(net: Net, x0: number, y0: number, x1: number, y1: number, margin: number): RoadPath[] {
  const seen = new Set<number>();
  for (let cy = Math.floor((y0 - margin) / CELL); cy <= Math.floor((y1 + margin) / CELL); cy++) {
    for (let cx = Math.floor((x0 - margin) / CELL); cx <= Math.floor((x1 + margin) / CELL); cx++) {
      for (const code of net.grid.get(key(cx, cy)) ?? []) seen.add(Math.floor(code / 100000));
    }
  }
  return [...seen].map((i) => net.paths[i]!);
}
