/**
 * Highways and dirt tracks drawn as smooth ribbons along their polylines (HD plan G2),
 * instead of tile by tile, so curves never turn into staircases.
 */
import type { RoadPath } from '../sim/worldgen';
import type { TerrainDef } from '../data/schema';
import { type G, alpha } from './ink';

/** Road paint: gravel shoulder, asphalt, wear, edge lines, centre dashes. Dirt colours come from the dirt terrain. */
const ROAD_LOOK = { shoulder: '#7b7466', asphalt: '#2e3035', wear: '#3a3c42', edge: '#e8e6df', centre: '#f2c94c' };

function offsetLine(pts: [number, number][], d: number): [number, number][] {
  return pts.map((p, i) => {
    const a = pts[Math.max(0, i - 1)]!, b = pts[Math.min(pts.length - 1, i + 1)]!;
    const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1;
    return [p[0] - (dy / l) * d, p[1] + (dx / l) * d];
  });
}

function strokePath(g: G, pts: [number, number][], width: number, color: string): void {
  g.strokeStyle = color; g.lineWidth = width;
  g.beginPath();
  g.moveTo(pts[0]![0], pts[0]![1]);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i]![0], pts[i]![1]);
  g.stroke();
}

/** Draws road ribbons in world coordinates; `g` must already be translated so world units map to the target. */
export function drawRoadRibbons(g: G, paths: RoadPath[], dirt: TerrainDef, grass: TerrainDef): void {
  g.save();
  g.lineCap = 'round'; g.lineJoin = 'round';
  // Dirt first so highways sit on top where they meet.
  for (const p of paths.filter((q) => q.kind === 'dirt_road')) {
    strokePath(g, p.pts, p.half * 2 + 0.8, alpha(dirt.colors[1], 0.45));
    strokePath(g, p.pts, p.half * 2, dirt.colors[0]);
    for (const s of [-1, 1]) strokePath(g, offsetLine(p.pts, s * p.half * 0.45), 0.34, alpha(dirt.colors[1], 0.7));
    strokePath(g, p.pts, 0.3, alpha(grass.colors[0], 0.55));
  }
  for (const p of paths.filter((q) => q.kind === 'highway')) {
    strokePath(g, p.pts, p.half * 2 + 1.0, ROAD_LOOK.shoulder);
    strokePath(g, p.pts, p.half * 2, ROAD_LOOK.asphalt);
    for (const s of [-1, 1]) strokePath(g, offsetLine(p.pts, s * p.half * 0.5), p.half * 0.45, alpha(ROAD_LOOK.wear, 0.55));
    for (const s of [-1, 1]) strokePath(g, offsetLine(p.pts, s * (p.half - 0.35)), 0.12, alpha(ROAD_LOOK.edge, 0.85));
    g.save();
    g.lineCap = 'butt';
    g.setLineDash([2.4, 2.2]);
    g.lineDashOffset = 0;
    strokePath(g, p.pts, 0.14, ROAD_LOOK.centre);
    g.restore();
  }
  g.restore();
}
