/**
 * Aim Arena Maps & Simulation Geometry (Z28a)
 * Pure data definitions for Duel Pit and Free-for-all Yard,
 * plus pure math segment & circle collision helpers.
 * Free of DOM and Canvas dependencies.
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface RectObstacle {
  x: number;
  y: number;
  w: number;
  h: number;
  stopsBullets: boolean;
  blocksMovement: boolean;
  kind: 'crate' | 'low_wall' | 'sandbags';
}

export interface CircleObstacle {
  x: number;
  y: number;
  radius: number;
  stopsBullets: boolean;
  blocksMovement: boolean;
  kind: 'barrel' | 'tree';
}

export interface SandbagDef {
  x: number;
  y: number;
  length: number;
  direction: 'h' | 'v';
}

export interface ArenaDef {
  id: 'duel_pit' | 'ffa_yard';
  name: string;
  width: number;
  height: number;
  spawns: Point2D[];
  pies: Point2D[];
  crates: { x: number; y: number; w: number; h: number }[];
  sandbags: SandbagDef[];
  lowWalls: { x: number; y: number; w: number; h: number }[];
  barrels: Point2D[];
  trees: Point2D[];
  roads?: { x: number; y: number; w: number; h: number }[];
}

/**
 * Duel Pit: 40 x 24m.
 * Mirrored half-turn symmetry: (40 - x, 24 - y).
 */
export const DUEL_PIT: ArenaDef = {
  id: 'duel_pit',
  name: 'Duel Pit',
  width: 40,
  height: 24,
  spawns: [
    { x: 4, y: 12 },
    { x: 36, y: 12 },
  ],
  pies: [
    { x: 20, y: 3 },
    { x: 20, y: 21 },
  ],
  crates: [
    { x: 17, y: 10, w: 2, h: 2 },
    { x: 21, y: 12, w: 2, h: 2 },
    { x: 19, y: 5, w: 2, h: 1 },
    { x: 19, y: 18, w: 2, h: 1 },
  ],
  sandbags: [
    { x: 11, y: 6, length: 5, direction: 'h' },
    { x: 24, y: 17, length: 5, direction: 'h' },
    { x: 11, y: 17, length: 5, direction: 'h' },
    { x: 24, y: 6, length: 5, direction: 'h' },
  ],
  lowWalls: [
    { x: 26, y: 9, w: 1, h: 3 },
    { x: 13, y: 12, w: 1, h: 3 },
  ],
  barrels: [
    { x: 9, y: 12 },
    { x: 31, y: 12 },
    { x: 16, y: 3 },
    { x: 24, y: 21 },
  ],
  trees: [
    { x: 3, y: 3 },
    { x: 37, y: 21 },
    { x: 3, y: 21 },
    { x: 37, y: 3 },
  ],
};

/**
 * Free-for-all Yard: 64 x 40m.
 * Asymmetric map for 4 to 8 combatants.
 */
export const FFA_YARD: ArenaDef = {
  id: 'ffa_yard',
  name: 'Free-for-all Yard',
  width: 64,
  height: 40,
  spawns: [
    { x: 6, y: 6 },
    { x: 32, y: 4 },
    { x: 58, y: 6 },
    { x: 60, y: 20 },
    { x: 58, y: 34 },
    { x: 32, y: 36 },
    { x: 6, y: 34 },
    { x: 4, y: 20 },
  ],
  pies: [
    { x: 32, y: 10 },
    { x: 32, y: 30 },
    { x: 14, y: 20 },
    { x: 50, y: 20 },
  ],
  crates: [
    { x: 28, y: 15, w: 2, h: 2 },
    { x: 34, y: 15, w: 2, h: 2 },
    { x: 28, y: 23, w: 2, h: 2 },
    { x: 34, y: 23, w: 2, h: 2 },
    { x: 10, y: 12, w: 2, h: 2 },
    { x: 52, y: 26, w: 2, h: 2 },
    { x: 52, y: 12, w: 2, h: 2 },
    { x: 10, y: 26, w: 2, h: 2 },
    { x: 31, y: 19, w: 2, h: 2 },
  ],
  sandbags: [
    { x: 14, y: 8, length: 6, direction: 'h' },
    { x: 44, y: 30, length: 6, direction: 'h' },
    { x: 44, y: 8, length: 6, direction: 'h' },
    { x: 14, y: 30, length: 6, direction: 'h' },
    { x: 22, y: 18, length: 4, direction: 'v' },
    { x: 40, y: 18, length: 4, direction: 'v' },
    { x: 22, y: 28, length: 4, direction: 'h' },
    { x: 38, y: 11, length: 4, direction: 'h' },
  ],
  lowWalls: [
    { x: 26, y: 10, w: 1, h: 3 },
    { x: 37, y: 27, w: 1, h: 3 },
    { x: 12, y: 18, w: 3, h: 1 },
    { x: 49, y: 21, w: 3, h: 1 },
  ],
  barrels: [
    { x: 20, y: 14 },
    { x: 44, y: 25 },
    { x: 20, y: 25 },
    { x: 44, y: 14 },
    { x: 32, y: 12 },
    { x: 32, y: 28 },
    { x: 8, y: 20 },
    { x: 56, y: 20 },
  ],
  trees: [
    { x: 2, y: 2 },
    { x: 62, y: 2 },
    { x: 2, y: 38 },
    { x: 62, y: 38 },
    { x: 16, y: 4 },
    { x: 48, y: 36 },
    { x: 48, y: 4 },
    { x: 16, y: 36 },
  ],
  roads: [
    // Roads (visual only): horizontal 18 to 22m (y from 18 to 22, full width), vertical 30 to 34m (x from 30 to 34, full height)
    { x: 0, y: 18, w: 64, h: 4 },
    { x: 30, y: 0, w: 4, h: 40 },
  ],
};

export const ARENAS: Record<string, ArenaDef> = {
  duel_pit: DUEL_PIT,
  ffa_yard: FFA_YARD,
};

/**
 * Returns all solid rectangular obstacles that stop bullets and block movement.
 */
export function getArenaRectSolids(arena: ArenaDef): RectObstacle[] {
  const list: RectObstacle[] = [];

  // Crates
  for (const c of arena.crates) {
    list.push({ x: c.x, y: c.y, w: c.w, h: c.h, stopsBullets: true, blocksMovement: true, kind: 'crate' });
  }

  // Low Walls
  for (const w of arena.lowWalls) {
    list.push({ x: w.x, y: w.y, w: w.w, h: w.h, stopsBullets: true, blocksMovement: true, kind: 'low_wall' });
  }

  // Sandbag rows (placed on 1-tile grid with 0.5m thick collision centered in the tile)
  for (const s of arena.sandbags) {
    if (s.direction === 'h') {
      list.push({
        x: s.x,
        y: s.y + 0.25, // centered 0.5m thick barrier in the 1m tile
        w: s.length,
        h: 0.5,
        stopsBullets: true,
        blocksMovement: true,
        kind: 'sandbags',
      });
    } else {
      list.push({
        x: s.x + 0.25,
        y: s.y,
        w: 0.5,
        h: s.length,
        stopsBullets: true,
        blocksMovement: true,
        kind: 'sandbags',
      });
    }
  }

  return list;
}

/**
 * Returns all circular obstacles (barrels, trees).
 * These block movement only, NOT stopping bullets according to brief Section 3.
 */
export function getArenaCircleSolids(arena: ArenaDef): CircleObstacle[] {
  const list: CircleObstacle[] = [];

  for (const b of arena.barrels) {
    list.push({ x: b.x, y: b.y, radius: 0.45, stopsBullets: false, blocksMovement: true, kind: 'barrel' });
  }

  for (const t of arena.trees) {
    list.push({ x: t.x, y: t.y, radius: 0.5, stopsBullets: false, blocksMovement: true, kind: 'tree' });
  }

  return list;
}

/** Line-segment vs Axis-Aligned Bounding Box intersection */
export function segmentIntersectsAABB(
  x0: number, y0: number,
  x1: number, y1: number,
  rx: number, ry: number, rw: number, rh: number
): boolean {
  // If line endpoint is inside rect
  if (x0 >= rx && x0 <= rx + rw && y0 >= ry && y0 <= ry + rh) return true;
  if (x1 >= rx && x1 <= rx + rw && y1 >= ry && y1 <= ry + rh) return true;

  const dx = x1 - x0;
  const dy = y1 - y0;

  let tMin = 0.0;
  let tMax = 1.0;

  // X slab
  if (Math.abs(dx) < 1e-9) {
    if (x0 < rx || x0 > rx + rw) return false;
  } else {
    const invD = 1.0 / dx;
    let t1 = (rx - x0) * invD;
    let t2 = (rx + rw - x0) * invD;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }

  // Y slab
  if (Math.abs(dy) < 1e-9) {
    if (y0 < ry || y0 > ry + rh) return false;
  } else {
    const invD = 1.0 / dy;
    let t1 = (ry - y0) * invD;
    let t2 = (ry + rh - y0) * invD;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }

  return true;
}

/** Circle vs AABB distance check for movement blocking */
export function circleHitsAABB(
  cx: number, cy: number, cr: number,
  rx: number, ry: number, rw: number, rh: number
): boolean {
  const closestX = Math.max(rx, Math.min(cx, rx + rw));
  const closestY = Math.max(ry, Math.min(cy, ry + rh));
  const dx = cx - closestX;
  const dy = cy - closestY;
  return (dx * dx + dy * dy) < (cr * cr);
}

/** Check if bullet path from (x0, y0) to (x1, y1) is blocked by arena cover */
export function arenaBlocksBullet(
  arena: ArenaDef,
  x0: number, y0: number,
  x1: number, y1: number
): boolean {
  const rectSolids = getArenaRectSolids(arena);
  for (const r of rectSolids) {
    if (r.stopsBullets && segmentIntersectsAABB(x0, y0, x1, y1, r.x, r.y, r.w, r.h)) {
      return true;
    }
  }
  return false;
}

/** Check if circle at (x, y) with radius r hits boundaries or obstacles */
export function arenaBlocksMovement(
  arena: ArenaDef,
  x: number, y: number,
  radius = 0.4
): boolean {
  // Perimeter boundaries
  if (x - radius < 0 || x + radius > arena.width || y - radius < 0 || y + radius > arena.height) {
    return true;
  }

  // Rectangular cover (crates, sandbags, low walls)
  const rectSolids = getArenaRectSolids(arena);
  for (const r of rectSolids) {
    if (r.blocksMovement && circleHitsAABB(x, y, radius, r.x, r.y, r.w, r.h)) {
      return true;
    }
  }

  // Circular cover (barrels, trees)
  const circleSolids = getArenaCircleSolids(arena);
  for (const c of circleSolids) {
    if (c.blocksMovement) {
      const d2 = (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y);
      const totalR = radius + c.radius;
      if (d2 < totalR * totalR) return true;
    }
  }

  return false;
}

/**
 * Checks half-turn symmetry for Duel Pit:
 * Rotating around center (20, 12): f(x, y) = (40 - x, 24 - y).
 */
export function checkDuelPitSymmetry(arena: ArenaDef = DUEL_PIT): { symmetric: boolean; errors: string[] } {
  const errors: string[] = [];
  const cx = arena.width / 2;  // 20
  const cy = arena.height / 2; // 12

  const matchPoint = (p: Point2D, list: Point2D[], tol = 0.01): boolean => {
    const rx = 2 * cx - p.x;
    const ry = 2 * cy - p.y;
    return list.some((other) => Math.hypot(other.x - rx, other.y - ry) <= tol);
  };

  // 1. Spawns
  for (const s of arena.spawns) {
    if (!matchPoint(s, arena.spawns)) errors.push(`Spawn (${s.x}, ${s.y}) has no symmetric match.`);
  }

  // 2. Pies
  for (const p of arena.pies) {
    if (!matchPoint(p, arena.pies)) errors.push(`Pie (${p.x}, ${p.y}) has no symmetric match.`);
  }

  // 3. Barrels
  for (const b of arena.barrels) {
    if (!matchPoint(b, arena.barrels)) errors.push(`Barrel (${b.x}, ${b.y}) has no symmetric match.`);
  }

  // 4. Trees
  for (const t of arena.trees) {
    if (!matchPoint(t, arena.trees)) errors.push(`Tree (${t.x}, ${t.y}) has no symmetric match.`);
  }

  // 5. Crates
  // Rotating rect with top-left (x, y) and size (w, h) by 180 deg around (cx, cy)
  // results in a rect with top-left (2*cx - x - w, 2*cy - y - h) and same size (w, h).
  for (const c of arena.crates) {
    const mx = 2 * cx - c.x - c.w;
    const my = 2 * cy - c.y - c.h;
    const found = arena.crates.some((o) => Math.abs(o.x - mx) < 0.01 && Math.abs(o.y - my) < 0.01 && o.w === c.w && o.h === c.h);
    if (!found) errors.push(`Crate at (${c.x}, ${c.y}, ${c.w}, ${c.h}) has no symmetric match.`);
  }

  // 6. Low walls
  for (const w of arena.lowWalls) {
    const mx = 2 * cx - w.x - w.w;
    const my = 2 * cy - w.y - w.h;
    const found = arena.lowWalls.some((o) => Math.abs(o.x - mx) < 0.01 && Math.abs(o.y - my) < 0.01 && o.w === w.w && o.h === w.h);
    if (!found) errors.push(`Low wall at (${w.x}, ${w.y}, ${w.w}, ${w.h}) has no symmetric match.`);
  }

  // 7. Sandbags (tile-placed with half-turn symmetry on the 1-tile grid)
  for (const s of arena.sandbags) {
    const tileW = s.direction === 'h' ? s.length : 1;
    const tileH = s.direction === 'h' ? 1 : s.length;
    const mx = 2 * cx - s.x - tileW;
    const my = 2 * cy - s.y - tileH;
    const found = arena.sandbags.some((o) => {
      const ow = o.direction === 'h' ? o.length : 1;
      const oh = o.direction === 'h' ? 1 : o.length;
      return Math.abs(o.x - mx) < 0.01 && Math.abs(o.y - my) < 0.01 && ow === tileW && oh === tileH && o.direction === s.direction;
    });
    if (!found) errors.push(`Sandbag at (${s.x}, ${s.y}, len=${s.length}, ${s.direction}) has no symmetric match.`);
  }

  // Also verify that generated solids for the entire arena are exactly symmetric
  const allSolids = getArenaRectSolids(arena);
  for (const r of allSolids) {
    const mx = 2 * cx - r.x - r.w;
    const my = 2 * cy - r.y - r.h;
    const found = allSolids.some((o) => Math.abs(o.x - mx) < 0.01 && Math.abs(o.y - my) < 0.01 && Math.abs(o.w - r.w) < 0.01 && Math.abs(o.h - r.h) < 0.01);
    if (!found) errors.push(`Collision solid at (${r.x}, ${r.y}, ${r.w}, ${r.h}) has no symmetric match.`);
  }

  return { symmetric: errors.length === 0, errors };
}
