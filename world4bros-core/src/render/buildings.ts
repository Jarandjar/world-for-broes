import type { Content, BuildingDef, BuildingFeature } from '../data/schema';
import { buildingColors } from '../data/buildings';
import { WALL_T, DOOR_W, wallRects, insideBuilding, type Building } from '../sim/worldgen';
import { SpriteCache, blit } from './sprites';
import { type G, INK, TAU, alpha, shade, wash, poly, rrect, ell, stroke } from './ink';

const MAX_PPU = 72;
/** Walls keep most of their colour; roofs barely wash so red/blue/green roofs read from far away. */
const WALL_DESAT = 0.22, ROOF_DESAT = 0.1;
/** Yard and front-path depth drawn at most (tiles), so sprites stay small. */
const MAX_YARD = 3.6, MAX_FRONT = 3.4;
/** Grounds colours (lawn, beds, vines, path, mailbox). */
const GROUND = {
  lawn: '#7fae5a', soil: '#6b4a2f', leaf: '#3f8a3a', leafHi: '#74bf4f', stem: '#2f5a2a',
  veg: ['#e0533d', '#f08a24', '#f2c94c', '#b0439a'], path: '#b9a98a', stone: '#a9a49a',
  landing: '#9c9a92', post: '#6b4a2f', flag: '#d94a3a', lamp: '#fbbf24',
};
type Colors = BuildingDef['colors'];
/** The def with this building's palette applied, so every painter reads one colour set. */
const tinted = (d: BuildingDef, b: Building): BuildingDef => ({ ...d, colors: buildingColors(d, b.x, b.y) });
const has = (d: BuildingDef, f: BuildingFeature): boolean => d.features?.includes(f) ?? false;

/** Draws town buildings: yards, porch entrance, floors, walls, furniture, and fading roofs. */
export class BuildingRenderer {
  private cache = new SpriteCache();
  private roofAlpha = new Map<string, number>();
  constructor(private content: Content) {}

  private def(b: Building): BuildingDef | undefined { return this.content.buildings.find((d) => d.id === b.def); }
  private key(b: Building, c: Colors, part: string): string {
    return `bld:${b.def}:${b.w}x${b.h}:${b.door}${b.doorAt}:${c.wall}${c.roof}${c.trim}:${(b.yard ?? 0).toFixed(1)}:${(b.front ?? 0).toFixed(1)}:${part}`;
  }

  /** Yard, path, porch entrance, floor, furniture and walls. */
  drawBase(g: G, b: Building, ppu: number): void {
    const d0 = this.def(b); if (!d0) return;
    const d = tinted(d0, b), p = Math.min(ppu, MAX_PPU);
    const [x0, y0, x1, y1] = baseBox(b);
    const spr = this.cache.get(this.key(b, d.colors, 'base'), p, x0, y0, x1, y1, (gg) => paintBase(gg, b, d));
    blit(g, spr, b.x, b.y);
  }

  /** Roof layer with architectural styles, chimneys, and smooth fading when player enters. */
  drawRoof(g: G, b: Building, ppu: number, dt: number, px: number, py: number): void {
    const d0 = this.def(b); if (!d0) return;
    const d = tinted(d0, b);
    const target = insideBuilding(b, px, py) ? 0 : 1;
    let a = this.roofAlpha.get(b.id) ?? 1;
    a += (target - a) * Math.min(1, dt * 7);
    if (Math.abs(target - a) < 0.01) a = target;
    this.roofAlpha.set(b.id, a);
    if (a <= 0.02) return;
    const p = Math.min(ppu, MAX_PPU);
    const spr = this.cache.get(this.key(b, d.colors, 'roof'), p, -1.2, -1.2, b.w + 1.2, b.h + 1.2, (gg) => paintRoof(gg, b, d));
    blit(g, spr, b.x, b.y, 0, 1, 1, a);
  }
}

/** Sprite box for the base layer: room for the back yard behind, the path to the street in front. */
function baseBox(b: Building): [number, number, number, number] {
  const back = Math.max(1.2, Math.min(b.yard ?? 0, MAX_YARD) + 0.4);
  const front = Math.max(2.6, Math.min(b.front ?? 0, MAX_FRONT) + 0.5);
  const side = 1.6, { w, h } = b;
  if (b.door === 'S') return [-side, -back, w + side, h + front];
  if (b.door === 'N') return [-side, -front, w + side, h + back];
  if (b.door === 'W') return [-front, -side, w + back, h + side];
  return [-back, -side, w + front, h + side];
}

/** Run `fn` in the building's street frame: frontage W along x, depth H, front wall at y = H facing +y, door at x = dx. */
function withFront(g: G, b: Building, fn: (W: number, H: number, dx: number) => void): void {
  const ns = b.door === 'N' || b.door === 'S';
  const W = ns ? b.w : b.h, H = ns ? b.h : b.w;
  const dx = b.door === 'S' || b.door === 'W' ? b.doorAt : W - b.doorAt;
  g.save();
  g.translate(b.w / 2, b.h / 2);
  g.rotate(b.door === 'S' ? 0 : b.door === 'N' ? Math.PI : b.door === 'E' ? -Math.PI / 2 : Math.PI / 2);
  g.translate(-W / 2, -H / 2);
  fn(W, H, dx);
  g.restore();
}

function paintBase(g: G, b: Building, d: BuildingDef): void {
  const { w, h } = b, c = d.colors;
  const half = DOOR_W / 2;

  // 1. Soft foundation ground shadow
  g.fillStyle = 'rgba(8,9,12,0.28)';
  g.fillRect(0.2, 0.35, w + 0.2, h + 0.2);

  // 2. Grounds: back yard, garden, path to the street, mailbox, porch or step
  paintGrounds(g, b, d);

  // 3. Interior Flooring
  g.fillStyle = wash(c.floor);
  g.fillRect(0, 0, w, h);

  // Floor plank/tile grooves
  g.strokeStyle = alpha(shade(c.floor, -0.45), 0.3);
  g.lineWidth = 0.025;
  g.beginPath();
  if (d.kind === 'house' || d.kind === 'shed' || d.walls === 'plank' || d.walls === 'log') {
    for (let y = 0.45; y < h; y += 0.45) { g.moveTo(0, y); g.lineTo(w, y); }
  } else {
    for (let y = 0.8; y < h; y += 0.8) { g.moveTo(0, y); g.lineTo(w, y); }
    for (let x = 0.8; x < w; x += 0.8) { g.moveTo(x, 0); g.lineTo(x, h); }
  }
  g.stroke();

  // 4. Interior Furniture & Rooms
  furnish(g, b, d);

  // 5. Interior Doormat in the threshold
  g.fillStyle = '#4a3b2c';
  if (b.door === 'N') g.fillRect(b.doorAt - half + 0.15, 0.05, DOOR_W - 0.3, WALL_T + 0.4);
  else if (b.door === 'S') g.fillRect(b.doorAt - half + 0.15, h - WALL_T - 0.45, DOOR_W - 0.3, WALL_T + 0.4);
  else if (b.door === 'W') g.fillRect(0.05, b.doorAt - half + 0.15, WALL_T + 0.4, DOOR_W - 0.3);
  else g.fillRect(w - WALL_T - 0.45, b.doorAt - half + 0.15, WALL_T + 0.4, DOOR_W - 0.3);

  // Woven mat texture
  g.fillStyle = '#6d573f';
  if (b.door === 'N' || b.door === 'S') {
    const my = b.door === 'N' ? 0.15 : h - 0.55;
    g.fillRect(b.doorAt - half + 0.3, my, DOOR_W - 0.6, 0.25);
  } else {
    const mx = b.door === 'W' ? 0.15 : w - 0.55;
    g.fillRect(mx, b.doorAt - half + 0.3, 0.25, DOOR_W - 0.6);
  }

  // 6. Visible Open Door Leaf (swung open into interior)
  paintDoorLeaf(g, b, d);

  // 7. Exterior Walls with Materials (brick, log, scrap, plank, concrete)
  paintWalls(g, b, d);
}

/** Everything outside the walls, drawn in the street frame so it works for any door side. */
function paintGrounds(g: G, b: Building, d: BuildingDef): void {
  const c = d.colors;
  const yard = Math.min(b.yard ?? 0, MAX_YARD), front = Math.min(b.front ?? 2.2, MAX_FRONT);
  const porch = has(d, 'porch'), porchD = porch ? 1.35 : 0.6;
  withFront(g, b, (W, H, dx) => {
    // Back yard: lawn patch and a Gothic picket fence from the side walls round the back of the lot
    if (has(d, 'fence') && yard >= 1) {
      const x0 = -0.45, x1 = W + 0.45, y0 = -yard + 0.3, yj = Math.min(1.2, H * 0.3);
      g.fillStyle = alpha(GROUND.lawn, 0.38);
      g.fillRect(x0, y0, x1 - x0, yj - y0);
      fence(g, [[0, yj], [x0, yj], [x0, y0], [x1, y0], [x1, yj], [W, yj]], c.trim);
    }

    // Garden beds: in the yard when there is room, else beside the porch
    if (has(d, 'garden')) {
      if (yard >= 1.6) bed(g, 0.4, -yard + 0.7, Math.min(2.8, W * 0.55), Math.min(1.3, yard - 1.1));
      else {
        const bw = dx - DOOR_W / 2 - (porch ? 1.5 : 0.6) - 0.2;
        if (bw >= 0.8) bed(g, 0.2, H + 0.15, bw, 0.75);
      }
    }

    // Path from the steps to the street, with natural irregular flagstone slabs
    const pathEnd = H + front;
    if (pathEnd - (H + porchD) > 0.3) {
      // Soft gravel bed
      g.fillStyle = alpha(GROUND.path, 0.8);
      g.fillRect(dx - 0.55, H + porchD, 1.1, pathEnd - H - porchD);
      g.strokeStyle = alpha(INK, 0.25); g.lineWidth = 0.02;
      g.strokeRect(dx - 0.55, H + porchD, 1.1, pathEnd - H - porchD);

      // Irregular flagstones
      for (let y = H + porchD + 0.32, i = 0; y < pathEnd - 0.12; y += 0.48, i++) {
        const sx = dx + (i % 2 ? 0.16 : -0.16);
        const col = i % 3 === 0 ? shade(GROUND.stone, 0.08) : i % 3 === 1 ? GROUND.stone : shade(GROUND.stone, -0.08);
        g.fillStyle = col;
        g.beginPath();
        g.roundRect(sx - 0.32, y - 0.14, 0.64, 0.28, 0.08);
        g.fill();
        g.strokeStyle = alpha(INK, 0.55); g.lineWidth = 0.025; g.stroke();
        // Tiny moss tuft between flagstones
        if (i % 2 === 0) {
          g.fillStyle = GROUND.leaf;
          g.beginPath(); g.arc(sx + 0.28, y + 0.14, 0.05, 0, TAU); g.fill();
        }
      }
    }

    // Curbside American Mailbox
    if (has(d, 'mailbox')) {
      const mx = dx + 1.0, my = Math.max(H + porchD + 0.4, pathEnd - 0.45);
      // Ground shadow
      g.fillStyle = 'rgba(8,9,12,0.28)';
      g.beginPath(); g.ellipse(mx + 0.08, my + 0.14, 0.32, 0.16, 0, 0, TAU); g.fill();
      // Wooden post & 45-deg brace
      g.fillStyle = GROUND.post;
      g.fillRect(mx - 0.06, my, 0.12, 0.26);
      g.beginPath();
      g.moveTo(mx, my + 0.05); g.lineTo(mx + 0.14, my - 0.08); g.lineTo(mx + 0.06, my - 0.08); g.closePath();
      g.fill();
      // Domed mailbox body
      g.fillStyle = shade(c.roof, 0.1);
      g.beginPath(); g.roundRect(mx - 0.28, my - 0.16, 0.56, 0.3, 0.12); g.fill();
      g.strokeStyle = INK; g.lineWidth = 0.035; g.stroke();
      // Mailbox door seam & handle
      g.strokeStyle = alpha(INK, 0.6); g.lineWidth = 0.02;
      g.strokeRect(mx - 0.25, my - 0.13, 0.1, 0.24);
      // Red alert flag
      g.fillStyle = GROUND.flag;
      g.fillRect(mx + 0.22, my - 0.3, 0.07, 0.24);
      g.fillRect(mx + 0.16, my - 0.3, 0.13, 0.09);
    }

    if (porch) {
      // High-detail wooden porch deck with tongue-and-groove planks, spindle railings, bench, and warm lantern
      const pw = Math.min(W - 0.6, DOOR_W + 2.5);
      const px = Math.max(0.3, Math.min(W - 0.3 - pw, dx - pw / 2));

      // Under-deck drop shadow
      g.fillStyle = 'rgba(8,9,12,0.32)';
      g.fillRect(px + 0.06, H + porchD - 0.04, pw, 0.14);

      // Deck foundation & tongue-and-groove planks
      g.fillStyle = wash(shade(c.floor, 0.12), WALL_DESAT);
      g.fillRect(px, H, pw, porchD);

      // Board seams
      g.strokeStyle = alpha(INK, 0.4); g.lineWidth = 0.02;
      g.beginPath();
      for (let x = px + 0.26; x < px + pw; x += 0.26) {
        g.moveTo(x, H); g.lineTo(x, H + porchD);
        // Countersunk nail heads at ends
        g.fillStyle = alpha(INK, 0.5);
        g.fillRect(x - 0.02, H + 0.04, 0.015, 0.015);
        g.fillRect(x - 0.02, H + porchD - 0.06, 0.015, 0.015);
      }
      g.stroke();

      // Deck perimeter border fascia
      g.strokeStyle = INK; g.lineWidth = 0.05; g.strokeRect(px, H, pw, porchD);

      // Porch Railing with turned spindles
      const railH = H + porchD - 0.07;
      const leftGapEnd = dx - 0.65, rightGapStart = dx + 0.65;

      // Bottom rail
      g.strokeStyle = shade(c.trim, -0.2); g.lineWidth = 0.04;
      g.beginPath();
      g.moveTo(px + 0.08, H); g.lineTo(px + 0.08, railH); g.lineTo(leftGapEnd, railH);
      g.moveTo(rightGapStart, railH); g.lineTo(px + pw - 0.08, railH); g.lineTo(px + pw - 0.08, H);
      g.stroke();

      // Top handrail
      g.strokeStyle = c.trim; g.lineWidth = 0.07;
      g.beginPath();
      g.moveTo(px + 0.08, H); g.lineTo(px + 0.08, railH); g.lineTo(leftGapEnd, railH);
      g.moveTo(rightGapStart, railH); g.lineTo(px + pw - 0.08, railH); g.lineTo(px + pw - 0.08, H);
      g.stroke();

      // Turned vertical spindles along front railing
      g.fillStyle = c.trim;
      g.strokeStyle = alpha(INK, 0.6); g.lineWidth = 0.015;
      for (let x = px + 0.3; x < leftGapEnd - 0.12; x += 0.22) {
        g.fillRect(x - 0.03, railH - 0.08, 0.06, 0.16);
        g.strokeRect(x - 0.03, railH - 0.08, 0.06, 0.16);
      }
      for (let x = rightGapStart + 0.22; x < px + pw - 0.25; x += 0.22) {
        g.fillRect(x - 0.03, railH - 0.08, 0.06, 0.16);
        g.strokeRect(x - 0.03, railH - 0.08, 0.06, 0.16);
      }

      // Sturdy square newel posts with beveled caps
      for (const x of [px, leftGapEnd - 0.05, rightGapStart - 0.12, px + pw - 0.24]) {
        g.fillStyle = c.trim;
        g.fillRect(x, H + porchD - 0.24, 0.24, 0.24);
        g.strokeStyle = INK; g.lineWidth = 0.035;
        g.strokeRect(x, H + porchD - 0.24, 0.24, 0.24);
        // Post cap pyramid
        g.fillStyle = shade(c.trim, 0.15);
        g.fillRect(x + 0.04, H + porchD - 0.2, 0.16, 0.16);
      }

      // Cozy Porch Bench / Rocking Chair on wider porches
      if (pw >= 3.4) {
        const bx = px + 0.35, by = H + 0.25;
        // Wooden bench seat
        g.fillStyle = '#5c4028';
        g.fillRect(bx, by, 0.9, 0.42);
        g.strokeStyle = INK; g.lineWidth = 0.03;
        g.strokeRect(bx, by, 0.9, 0.42);
        // Slatted backrest
        g.fillStyle = '#785536';
        g.fillRect(bx, by, 0.9, 0.14);
        g.strokeRect(bx, by, 0.9, 0.14);
        // Seat slats
        g.strokeStyle = alpha(INK, 0.5); g.lineWidth = 0.02;
        g.beginPath();
        for (let sx = bx + 0.18; sx < bx + 0.85; sx += 0.18) { g.moveTo(sx, by); g.lineTo(sx, by + 0.42); }
        g.stroke();
      }

      // Wooden front steps down to walkway
      g.fillStyle = wash(shade(c.floor, -0.15), WALL_DESAT);
      g.fillRect(dx - 0.58, H + porchD, 1.16, 0.26);
      g.strokeStyle = INK; g.lineWidth = 0.035;
      g.strokeRect(dx - 0.58, H + porchD, 1.16, 0.26);
      // Second step
      g.fillStyle = wash(shade(c.floor, -0.25), WALL_DESAT);
      g.fillRect(dx - 0.52, H + porchD + 0.13, 1.04, 0.14);
      g.strokeRect(dx - 0.52, H + porchD + 0.13, 1.04, 0.14);
    } else {
      // Heavy stone / concrete landing step
      g.fillStyle = GROUND.landing;
      g.fillRect(dx - DOOR_W / 2 - 0.24, H, DOOR_W + 0.48, porchD);
      g.strokeStyle = INK; g.lineWidth = 0.04;
      g.strokeRect(dx - DOOR_W / 2 - 0.24, H, DOOR_W + 0.48, porchD);
      // Beveled edge highlight
      g.fillStyle = alpha('#ffffff', 0.25);
      g.fillRect(dx - DOOR_W / 2 - 0.22, H + 0.02, DOOR_W + 0.44, 0.06);
    }

    // Porch carriage lantern with ambient illumination halo
    const lx = Math.min(W - 0.25, dx + DOOR_W / 2 + 0.35), ly = H + 0.22;
    // Ambient warm light halo
    const glow = g.createRadialGradient(lx, ly, 0.05, lx, ly, 0.9);
    glow.addColorStop(0, 'rgba(251, 191, 36, 0.45)');
    glow.addColorStop(0.5, 'rgba(251, 191, 36, 0.18)');
    glow.addColorStop(1, 'rgba(251, 191, 36, 0)');
    g.fillStyle = glow;
    g.beginPath(); g.arc(lx, ly, 0.9, 0, TAU); g.fill();

    // Brass/black iron lantern fixture
    g.fillStyle = '#1c1917';
    g.fillRect(lx - 0.08, ly - 0.14, 0.16, 0.28);
    g.strokeStyle = INK; g.lineWidth = 0.03;
    g.strokeRect(lx - 0.08, ly - 0.14, 0.16, 0.28);
    // Incandescent amber filament
    g.fillStyle = GROUND.lamp;
    g.fillRect(lx - 0.05, ly - 0.08, 0.1, 0.16);
    g.fillStyle = '#ffffff';
    g.fillRect(lx - 0.02, ly - 0.04, 0.04, 0.08);
  });
}

/** Picket fence with pointed Gothic pickets and sturdy rails. */
function fence(g: G, pts: [number, number][], col: string): void {
  g.lineCap = 'round';
  // Shadow under fence
  g.strokeStyle = 'rgba(8,9,12,0.2)'; g.lineWidth = 0.14;
  g.beginPath(); poly(g, pts, false); g.stroke();

  // Bottom & top rail
  g.strokeStyle = shade(col, -0.3); g.lineWidth = 0.08;
  g.beginPath(); poly(g, pts, false); g.stroke();
  g.strokeStyle = col; g.lineWidth = 0.05;
  g.beginPath(); poly(g, pts, false); g.stroke();

  // Gothic pointed pickets along the fence
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1]!, [bx, by] = pts[i]!;
    const len = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.round(len / 0.42));
    for (let k = 0; k <= n; k++) {
      const x = ax + ((bx - ax) * k) / n, y = ay + ((by - ay) * k) / n;
      // Pointed picket head
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(x - 0.06, y + 0.08);
      g.lineTo(x - 0.06, y - 0.04);
      g.lineTo(x, y - 0.1);
      g.lineTo(x + 0.06, y - 0.04);
      g.lineTo(x + 0.06, y + 0.08);
      g.closePath();
      g.fill();
      g.strokeStyle = INK; g.lineWidth = 0.022; g.stroke();
      // Fastener nail
      g.fillStyle = alpha(INK, 0.7);
      g.fillRect(x - 0.015, y, 0.03, 0.03);
    }
  }
  g.lineCap = 'butt';
}

/** Vegetable bed: timber retaining border with lush layered crop foliage. */
function bed(g: G, x: number, y: number, w: number, h: number): void {
  if (w < 0.6 || h < 0.4) return;
  // Wooden border frame
  g.fillStyle = '#4a321f';
  g.fillRect(x - 0.05, y - 0.05, w + 0.1, h + 0.1);
  g.strokeStyle = INK; g.lineWidth = 0.04;
  g.strokeRect(x - 0.05, y - 0.05, w + 0.1, h + 0.1);

  // Rich compost soil
  g.fillStyle = GROUND.soil; g.fillRect(x, y, w, h);

  // Furrow lines in soil
  g.strokeStyle = alpha('#2d1f14', 0.5); g.lineWidth = 0.02;
  g.beginPath();
  for (let ry = y + 0.22; ry < y + h - 0.08; ry += 0.38) {
    g.moveTo(x + 0.05, ry); g.lineTo(x + w - 0.05, ry);
  }
  g.stroke();

  let i = 0;
  for (let ry = y + 0.22; ry < y + h - 0.08; ry += 0.38) {
    for (let rx = x + 0.22; rx < x + w - 0.12; rx += 0.32, i++) {
      // Layered leafy crop
      g.fillStyle = i % 2 ? GROUND.leaf : GROUND.leafHi;
      g.beginPath(); g.arc(rx, ry, 0.13, 0, TAU); g.fill();
      g.fillStyle = shade(GROUND.leaf, -0.2);
      g.beginPath(); g.arc(rx + 0.03, ry + 0.03, 0.08, 0, TAU); g.fill();
      // Colored crop fruits (tomatoes, carrots, peppers)
      if (i % 3 === 0) {
        g.fillStyle = GROUND.veg[(i / 3) % GROUND.veg.length]!;
        g.beginPath(); g.arc(rx + 0.04, ry - 0.03, 0.06, 0, TAU); g.fill();
        g.strokeStyle = INK; g.lineWidth = 0.015; g.stroke();
      }
    }
  }
}

/** Climbing vine along a line: a wobbly stem with leaf clumps (seeded by index, so sprites stay stable). */
function vine(g: G, x0: number, y0: number, x1: number, y1: number, seed: number): void {
  const len = Math.hypot(x1 - x0, y1 - y0); if (len < 0.3) return;
  const n = Math.max(2, Math.floor(len / 0.26)), nx = -(y1 - y0) / len, ny = (x1 - x0) / len;
  const at = (i: number): [number, number] => {
    const t = i / n, wob = Math.sin(i * 1.7 + seed) * 0.12;
    return [x0 + (x1 - x0) * t + nx * wob, y0 + (y1 - y0) * t + ny * wob];
  };
  g.strokeStyle = GROUND.stem; g.lineWidth = 0.045;
  g.beginPath();
  for (let i = 0; i <= n; i++) { const [x, y] = at(i); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
  g.stroke();
  g.strokeStyle = alpha(INK, 0.5); g.lineWidth = 0.02;
  for (let i = 0; i <= n; i++) {
    const [x, y] = at(i), r = 0.1 + 0.03 * ((i * 7 + seed) % 3);
    g.fillStyle = (i + seed) % 3 ? GROUND.leaf : GROUND.leafHi;
    g.beginPath(); g.arc(x + nx * 0.06 * (i % 2 ? 1 : -1), y + ny * 0.06 * (i % 2 ? 1 : -1), r, 0, TAU); g.fill(); g.stroke();
  }
}

function paintDoorLeaf(g: G, b: Building, d: BuildingDef): void {
  const { w, h } = b, c = d.colors;
  const half = DOOR_W / 2;
  const doorThickness = 0.12;
  const leafLen = half * 0.95;

  g.save();
  g.fillStyle = shade(c.trim, -0.25);
  g.strokeStyle = INK;
  g.lineWidth = 0.03;

  if (b.door === 'S') {
    // Left door leaf swung open at 60 degrees into room
    g.translate(b.doorAt - half, h - WALL_T);
    g.rotate(-Math.PI * 0.35);
    g.fillRect(0, 0, leafLen, doorThickness);
    g.strokeRect(0, 0, leafLen, doorThickness);
    // Brass knob
    g.fillStyle = '#f59e0b';
    g.beginPath(); g.arc(leafLen - 0.12, doorThickness / 2, 0.045, 0, TAU); g.fill();
  } else if (b.door === 'N') {
    g.translate(b.doorAt - half, WALL_T);
    g.rotate(Math.PI * 0.35);
    g.fillRect(0, -doorThickness, leafLen, doorThickness);
    g.strokeRect(0, -doorThickness, leafLen, doorThickness);
    g.fillStyle = '#f59e0b';
    g.beginPath(); g.arc(leafLen - 0.12, -doorThickness / 2, 0.045, 0, TAU); g.fill();
  } else if (b.door === 'W') {
    g.translate(WALL_T, b.doorAt - half);
    g.rotate(Math.PI * 0.35);
    g.fillRect(0, 0, doorThickness, leafLen);
    g.strokeRect(0, 0, doorThickness, leafLen);
  } else {
    g.translate(w - WALL_T, b.doorAt - half);
    g.rotate(-Math.PI * 0.35);
    g.fillRect(-doorThickness, 0, doorThickness, leafLen);
    g.strokeRect(-doorThickness, 0, doorThickness, leafLen);
  }
  g.restore();
}

function paintWalls(g: G, b: Building, d: BuildingDef): void {
  const c = d.colors;
  const wallCol = wash(c.wall);

  for (const [x0, y0, x1, y1] of wallRects(b)) {
    const rx = x0 - b.x, ry = y0 - b.y, rw = x1 - x0, rh = y1 - y0;
    const horiz = rw >= rh, len = horiz ? rw : rh;

    // 1. Base wall fill
    g.fillStyle = wallCol;
    g.fillRect(rx, ry, rw, rh);

    // 2. High-fidelity wall material textures
    if (d.walls === 'brick') {
      // Running bond brickwork with mortar joints and weathered accent bricks
      const rowH = 0.22, brickW = 0.44;
      g.strokeStyle = alpha('#e2e8f0', 0.45);
      g.lineWidth = 0.02;

      if (horiz) {
        for (let y = ry + rowH, row = 0; y < ry + rh; y += rowH, row++) {
          g.beginPath(); g.moveTo(rx, y); g.lineTo(rx + rw, y); g.stroke();
          const offset = (row % 2) * (brickW / 2);
          for (let x = rx + offset; x < rx + rw; x += brickW) {
            // Occasional burnt/darker brick
            if ((Math.round(x * 7 + y * 13) % 7) === 0) {
              g.fillStyle = alpha(shade(c.wall, -0.35), 0.6);
              g.fillRect(x + 0.02, y - rowH + 0.02, brickW - 0.04, rowH - 0.04);
            }
            g.beginPath(); g.moveTo(x, y - rowH); g.lineTo(x, y); g.stroke();
          }
        }
      } else {
        for (let x = rx + rowH, col = 0; x < rx + rw; x += rowH, col++) {
          g.beginPath(); g.moveTo(x, ry); g.lineTo(x, ry + rh); g.stroke();
          const offset = (col % 2) * (brickW / 2);
          for (let y = ry + offset; y < ry + rh; y += brickW) {
            if ((Math.round(x * 11 + y * 5) % 7) === 0) {
              g.fillStyle = alpha(shade(c.wall, -0.35), 0.6);
              g.fillRect(x - rowH + 0.02, y + 0.02, rowH - 0.04, brickW - 0.04);
            }
            g.beginPath(); g.moveTo(x - rowH, y); g.lineTo(x, y); g.stroke();
          }
        }
      }

      // Limestone foundation water-table course along bottom of wall
      g.fillStyle = alpha('#cbd5e1', 0.5);
      if (horiz) g.fillRect(rx, ry + rh - 0.08, rw, 0.08);
      else g.fillRect(rx + rw - 0.08, ry, 0.08, rh);

    } else if (d.walls === 'log') {
      // Heavy timber logs with cylindrical shading and chinking mortar
      const logH = 0.38;
      if (horiz) {
        for (let y = ry; y < ry + rh - 0.05; y += logH) {
          // Log curve highlight & shadow
          g.fillStyle = alpha('#ffffff', 0.15);
          g.fillRect(rx, y + 0.04, rw, 0.08);
          g.fillStyle = alpha(INK, 0.3);
          g.fillRect(rx, y + logH - 0.08, rw, 0.08);
          // Mortar chinking seam
          g.strokeStyle = alpha('#e2d9c8', 0.65);
          g.lineWidth = 0.03;
          g.beginPath(); g.moveTo(rx, y + logH); g.lineTo(rx + rw, y + logH); g.stroke();
        }
      } else {
        for (let x = rx; x < rx + rw - 0.05; x += logH) {
          g.fillStyle = alpha('#ffffff', 0.15);
          g.fillRect(x + 0.04, ry, 0.08, rh);
          g.fillStyle = alpha(INK, 0.3);
          g.fillRect(x + logH - 0.08, ry, 0.08, rh);
          g.strokeStyle = alpha('#e2d9c8', 0.65);
          g.lineWidth = 0.03;
          g.beginPath(); g.moveTo(x + logH, ry); g.lineTo(x + logH, ry + rh); g.stroke();
        }
      }

    } else if (d.walls === 'plank' || d.kind === 'house' || d.kind === 'shed') {
      // Horizontal overlapping clapboard lap siding
      const plankH = 0.26;
      if (horiz) {
        for (let y = ry; y < ry + rh - 0.02; y += plankH) {
          // Top bevel highlight
          g.fillStyle = alpha('#ffffff', 0.16);
          g.fillRect(rx, y, rw, 0.04);
          // Bottom lap drop shadow
          g.fillStyle = alpha(INK, 0.28);
          g.fillRect(rx, y + plankH - 0.04, rw, 0.04);
          // Plank groove
          g.strokeStyle = alpha(shade(c.wall, -0.45), 0.5);
          g.lineWidth = 0.018;
          g.beginPath(); g.moveTo(rx, y + plankH); g.lineTo(rx + rw, y + plankH); g.stroke();
        }
      } else {
        for (let y = ry; y < ry + rh - 0.02; y += plankH) {
          g.fillStyle = alpha('#ffffff', 0.16);
          g.fillRect(rx, y, rw, 0.04);
          g.fillStyle = alpha(INK, 0.28);
          g.fillRect(rx, y + plankH - 0.04, rw, 0.04);
          g.strokeStyle = alpha(shade(c.wall, -0.45), 0.5);
          g.lineWidth = 0.018;
          g.beginPath(); g.moveTo(rx, y + plankH); g.lineTo(rx + rw, y + plankH); g.stroke();
        }
      }

      // Vertical corner trim boards
      g.fillStyle = c.trim;
      g.fillRect(rx, ry, Math.min(rw, 0.14), rh);
      g.fillRect(rx + rw - Math.min(rw, 0.14), ry, Math.min(rw, 0.14), rh);

    } else if (d.walls === 'scrap') {
      // Patchwork corrugated metal panels with rivets
      g.fillStyle = alpha(shade(c.wall, 0.28), 0.45);
      if (horiz) {
        g.fillRect(rx + 0.4, ry, Math.min(1.4, rw - 0.4), rh);
      } else {
        g.fillRect(rx, ry + 0.4, rw, Math.min(1.4, rh - 0.4));
      }
      // Rivet dots
      g.fillStyle = alpha(INK, 0.6);
      for (let p = 0.2; p < len; p += 0.45) {
        const px = horiz ? rx + p : rx + 0.06, py = horiz ? ry + 0.06 : ry + p;
        g.fillRect(px, py, 0.03, 0.03);
      }
    } else {
      // Concrete / Plaster: foundation plinth & expansion seams
      g.fillStyle = alpha(shade(c.wall, -0.22), 0.4);
      if (horiz) g.fillRect(rx, ry + rh - 0.1, rw, 0.1);
      else g.fillRect(rx + rw - 0.1, ry, 0.1, rh);
    }

    // Top highlight rim
    g.fillStyle = alpha(shade(c.wall, 0.45), 0.65);
    g.fillRect(rx, ry, rw, Math.min(rh, 0.08));

    // Outer inked wall perimeter
    g.strokeStyle = INK;
    g.lineWidth = 0.045;
    g.strokeRect(rx, ry, rw, rh);

    // 3. High-detail windows with reveals, curtains, glass reflections, and flowerboxes
    for (let p = 1.1; p < len - 1.1; p += 2.2) {
      const wx = horiz ? rx + p : rx, wy = horiz ? ry : ry + p;
      const ww = horiz ? 0.95 : rw, wh = horiz ? rh : 0.95;

      // Projecting window sill on bottom with cast shadow
      if (horiz) {
        g.fillStyle = 'rgba(8,9,12,0.3)';
        g.fillRect(wx - 0.06, wy + wh, ww + 0.12, 0.08);
        g.fillStyle = c.trim;
        g.fillRect(wx - 0.08, wy + wh - 0.04, ww + 0.16, 0.08);
        g.strokeStyle = INK; g.lineWidth = 0.025;
        g.strokeRect(wx - 0.08, wy + wh - 0.04, ww + 0.16, 0.08);
      }

      // Recessed window reveal
      g.fillStyle = 'rgba(0,0,0,0.35)';
      g.fillRect(wx, wy, ww, wh);

      // Cozy interior curtains drawn at sides
      g.fillStyle = '#fef3c7'; // Warm linen/cream
      if (horiz) {
        g.fillRect(wx + 0.04, wy + 0.04, 0.22, wh - 0.08);
        g.fillRect(wx + ww - 0.26, wy + 0.04, 0.22, wh - 0.08);
      } else {
        g.fillRect(wx + 0.04, wy + 0.04, ww - 0.08, 0.22);
        g.fillRect(wx + 0.04, wy + wh - 0.26, ww - 0.08, 0.22);
      }

      // Sky-blue window glass
      g.fillStyle = '#67e8f9';
      const gw = horiz ? ww - 0.14 : ww - 0.08, gh = horiz ? wh - 0.08 : wh - 0.14;
      const gx = horiz ? wx + 0.07 : wx + 0.04, gy = horiz ? wy + 0.04 : wy + 0.07;
      g.fillRect(gx, gy, gw, gh);

      // Specular diagonal reflection glare
      g.fillStyle = 'rgba(255,255,255,0.48)';
      g.beginPath();
      g.moveTo(gx + 0.05, gy);
      g.lineTo(gx + gw * 0.45, gy);
      g.lineTo(gx + gw * 0.15, gy + gh);
      g.lineTo(gx, gy + gh);
      g.closePath();
      g.fill();

      // Outer window moulding frame
      g.strokeStyle = c.trim;
      g.lineWidth = 0.06;
      g.strokeRect(wx, wy, ww, wh);

      // Center crossbars / muntins
      g.strokeStyle = shade(c.trim, -0.22);
      g.lineWidth = 0.025;
      g.beginPath();
      g.moveTo(wx + ww / 2, wy); g.lineTo(wx + ww / 2, wy + wh);
      g.moveTo(wx, wy + wh / 2); g.lineTo(wx + ww, wy + wh / 2);
      g.stroke();

      // Boarded-up survivor barricade planks across window
      if (has(d, 'boarded')) {
        g.save();
        g.translate(wx + ww / 2, wy + wh / 2);
        // Diagonal plank 1
        g.rotate(0.35);
        g.fillStyle = '#6b482b';
        g.fillRect(-ww * 0.55, -0.09, ww * 1.1, 0.18);
        g.strokeStyle = INK; g.lineWidth = 0.025;
        g.strokeRect(-ww * 0.55, -0.09, ww * 1.1, 0.18);
        // Iron nails
        g.fillStyle = '#1c1917';
        g.fillRect(-ww * 0.45, -0.04, 0.04, 0.04);
        g.fillRect(ww * 0.4, -0.04, 0.04, 0.04);
        // Diagonal plank 2
        g.rotate(-0.7);
        g.fillStyle = '#5a3d24';
        g.fillRect(-ww * 0.55, -0.09, ww * 1.1, 0.18);
        g.strokeRect(-ww * 0.55, -0.09, ww * 1.1, 0.18);
        g.fillStyle = '#1c1917';
        g.fillRect(-ww * 0.45, -0.04, 0.04, 0.04);
        g.fillRect(ww * 0.4, -0.04, 0.04, 0.04);
        g.restore();
      }

      // Flower box under window on cottages, townhouses, and cabins
      if ((has(d, 'flowerBoxes') || d.kind === 'house') && horiz && !has(d, 'boarded')) {
        // Cedar planter trough
        g.fillStyle = '#78350f';
        g.fillRect(wx - 0.06, wy + wh + 0.04, ww + 0.12, 0.22);
        g.strokeStyle = INK; g.lineWidth = 0.03;
        g.strokeRect(wx - 0.06, wy + wh + 0.04, ww + 0.12, 0.22);

        // Rich dark soil
        g.fillStyle = '#3e2723';
        g.fillRect(wx - 0.04, wy + wh + 0.06, ww + 0.08, 0.07);

        // Overflowing cascading foliage
        g.fillStyle = '#2e7d32';
        for (let fl = 0; fl < 5; fl++) {
          g.beginPath();
          g.arc(wx + 0.12 + fl * 0.18, wy + wh + 0.13, 0.1, 0, TAU);
          g.fill();
        }
        g.fillStyle = '#4caf50';
        for (let fl = 0; fl < 4; fl++) {
          g.beginPath();
          g.arc(wx + 0.2 + fl * 0.18, wy + wh + 0.18, 0.07, 0, TAU);
          g.fill();
        }

        // Vibrant blossoms spilling over the edges
        const petals = ['#f43f5e', '#facc15', '#a855f7', '#38bdf8', '#ec4899', '#fb923c'];
        for (let fl = 0; fl < 5; fl++) {
          const col = petals[(fl * 2 + Math.round(wx)) % petals.length]!;
          g.fillStyle = col;
          g.beginPath();
          g.arc(wx + 0.12 + fl * 0.18, wy + wh + 0.12, 0.065, 0, TAU);
          g.fill();
          g.fillStyle = '#ffffff';
          g.beginPath();
          g.arc(wx + 0.12 + fl * 0.18, wy + wh + 0.12, 0.025, 0, TAU);
          g.fill();
        }
      }
    }
  }

  // Store Sign above entrance for shops, supermarkets, garages, and clinics
  if (has(d, 'sign') || d.kind === 'shop' || d.kind === 'supermarket' || d.kind === 'clinic') {
    paintStoreSign(g, b, d);
  }
}

function paintStoreSign(g: G, b: Building, d: BuildingDef): void {
  const { h } = b;
  const signText = d.kind === 'supermarket' ? 'SUPERSTORE'
    : d.kind === 'shop' ? 'GROCERY'
    : d.kind === 'clinic' ? 'CLINIC'
    : d.kind === 'garage' ? 'AUTO REPAIR'
    : 'SUPPLIES';

  const signW = Math.max(2.8, signText.length * 0.34);
  const signH = 0.64;
  let sx = b.doorAt - signW / 2, sy = h + 0.06;
  if (b.door === 'N') sy = -signH - 0.06;

  // Sign drop shadow
  g.fillStyle = 'rgba(8,9,12,0.32)';
  g.fillRect(sx + 0.06, sy + 0.08, signW, signH);

  // Wrought iron mounting brackets
  g.strokeStyle = '#1e293b'; g.lineWidth = 0.045;
  g.beginPath();
  g.moveTo(sx + 0.2, sy); g.lineTo(sx + 0.2, sy - 0.12);
  g.moveTo(sx + signW - 0.2, sy); g.lineTo(sx + signW - 0.2, sy - 0.12);
  g.stroke();

  // Sign board body
  g.fillStyle = '#0f172a';
  g.fillRect(sx, sy, signW, signH);

  // Golden / cream beveled outer border
  g.strokeStyle = d.colors.trim;
  g.lineWidth = 0.05;
  g.strokeRect(sx, sy, signW, signH);

  // Inner decorative gold fillet
  g.strokeStyle = '#f59e0b';
  g.lineWidth = 0.018;
  g.strokeRect(sx + 0.06, sy + 0.06, signW - 0.12, signH - 0.12);

  // Sharp drop shadow on text
  g.font = 'bold 0.29px sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = 'rgba(0,0,0,0.85)';
  g.fillText(signText, sx + signW / 2 + 0.015, sy + signH / 2 + 0.02);

  // Main golden lettering
  g.fillStyle = '#facc15';
  g.fillText(signText, sx + signW / 2, sy + signH / 2);
}

function furnish(g: G, b: Building, d: BuildingDef): void {
  const c = d.colors, ns = b.door === 'N' || b.door === 'S';
  const W = ns ? b.w : b.h, H = ns ? b.h : b.w;
  g.save();
  g.translate(b.w / 2, b.h / 2);
  g.rotate(b.door === 'S' ? 0 : b.door === 'N' ? Math.PI : b.door === 'E' ? -Math.PI / 2 : Math.PI / 2);
  g.translate(-W / 2, -H / 2);

  const box = (x: number, y: number, bw: number, bh: number, col: string): void => {
    g.fillStyle = wash(col); g.fillRect(x, y, bw, bh);
    g.strokeStyle = alpha(INK, 0.8); g.lineWidth = 0.03; g.strokeRect(x, y, bw, bh);
  };
  const t = WALL_T + 0.25, wood = shade(c.floor, -0.3);

  // Interior dividing walls for split/apartment/backroom layouts
  if (d.layout === 'split') {
    box(W / 2 - 0.15, t, 0.3, H - 3.2, c.wall);
  } else if (d.layout === 'backroom') {
    box(t, H / 2 - 0.15, W - 2 * t - 1.8, 0.3, c.wall);
  }

  switch (d.kind) {
    case 'house':
      box(W / 2 - 1.4, H / 2 - 0.9, 2.8, 1.8, shade(c.roof, 0.1)); // Rug
      box(t, t, 1.5, 2.1, shade(c.trim, -0.1)); // Bed frame
      box(t + 0.1, t + 0.1, 1.3, 0.6, c.trim); // Pillow
      box(W - t - 1.6, t, 1.6, 1.0, wood); // Table
      box(W - t - 1.4, t + 1.2, 0.6, 0.6, wood); // Chair
      break;
    case 'shed':
      box(t, t, W - 2 * t, 0.7, wood); // Workbench
      box(t, t + 1.0, 0.8, 0.8, '#475569'); // Scrap crate
      break;
    case 'shop':
      box(t, H - 3.0, Math.max(2, W - 3.2), 0.85, wood); // Cash counter
      for (let x = t; x < W - t - 0.8; x += 1.6) box(x, t, 1.3, 0.6, shade(c.roof, 0.2)); // Shelves
      break;
    case 'supermarket':
      // Multiple double aisles of grocery shelves
      for (let y = t + 0.8; y < H - 2.8; y += 1.8) {
        box(t + 0.5, y, W - 2 * t - 1.0, 0.7, '#64748b');
      }
      box(t, H - 2.4, 2.2, 0.8, '#334155'); // Checkout lanes
      box(t + 3.0, H - 2.4, 2.2, 0.8, '#334155');
      break;
    case 'garage':
      box(W / 2 - 1.6, t + 0.6, 3.2, H - 3.2, '#3b3e45'); // Hydraulic lift bay
      g.fillStyle = c.trim;
      for (let y = t + 0.8; y < H - 2.8; y += 0.8) g.fillRect(W / 2 - 1.6, y, 3.2, 0.12);
      box(t, t, 1.8, 0.8, wood); // Tool chest
      box(W - t - 1.2, t, 1.2, 1.2, '#b45309'); // Oil drum cluster
      break;
    case 'clinic':
      box(t, t, 1.1, 2.0, c.trim); box(t + 1.5, t, 1.1, 2.0, c.trim); // Triage cots
      box(W - t - 1.8, H - 3, 1.8, 0.8, wood); // Doctor's desk
      // Red Cross emblem
      g.fillStyle = '#ef4444';
      g.fillRect(W / 2 - 0.14, H / 2 - 0.5, 0.28, 1.0);
      g.fillRect(W / 2 - 0.5, H / 2 - 0.14, 1.0, 0.28);
      break;
    case 'warehouse':
      // Industrial shipping pallet stacks
      for (let x = t; x < W - t - 1.2; x += 2.2) {
        for (let y = t; y < H - 3.2; y += 1.8) box(x, y, 1.5, 1.3, '#78350f');
      }
      break;
    case 'apartment':
      box(t, t, 1.4, 1.8, c.trim); // Bedroom 1
      box(W - t - 1.4, t, 1.4, 1.8, c.trim); // Bedroom 2
      box(W / 2 - 1.0, H / 2 - 0.7, 2.0, 1.4, wood); // Living table
      break;
  }
  g.restore();
}

function paintRoof(g: G, b: Building, d: BuildingDef): void {
  const { w, h } = b, c = d.colors, o = 0.52;

  // 1. Multi-pass soft shadow cast by roof onto ground
  g.fillStyle = 'rgba(8,12,18,0.14)';
  g.fillRect(-o + 0.35, -o + 0.48, w + 2 * o + 0.25, h + 2 * o + 0.32);
  g.fillStyle = 'rgba(8,12,18,0.24)';
  g.fillRect(-o + 0.16, -o + 0.26, w + 2 * o, h + 2 * o);

  // 2. Base Roof Color Wash
  const baseRoof = wash(c.roof, ROOF_DESAT);
  g.fillStyle = baseRoof;
  g.fillRect(-o, -o, w + 2 * o, h + 2 * o);

  const horizRidge = w >= h;

  // 3. Roof Style Shading, Textures & Architectural Geometry
  if (d.roof === 'tin') {
    // 3D Corrugated metal ribs
    const step = 0.3;
    for (let x = -o + 0.2; x < w + o; x += step) {
      // Highlight crest
      g.strokeStyle = alpha('#ffffff', 0.24);
      g.lineWidth = 0.04;
      g.beginPath(); g.moveTo(x, -o); g.lineTo(x, h + o); g.stroke();
      // Shadow trough
      g.strokeStyle = alpha(INK, 0.38);
      g.lineWidth = 0.04;
      g.beginPath(); g.moveTo(x + step / 2, -o); g.lineTo(x + step / 2, h + o); g.stroke();
    }

    // Horizontal purlin lines with fastener screws
    for (const py of [h * 0.25, h * 0.5, h * 0.75]) {
      g.strokeStyle = alpha(INK, 0.45); g.lineWidth = 0.02;
      g.beginPath(); g.moveTo(-o, py); g.lineTo(w + o, py); g.stroke();
      // Fastener dots
      g.fillStyle = alpha('#ffffff', 0.65);
      for (let x = -o + 0.2; x < w + o; x += step) {
        g.fillRect(x - 0.015, py - 0.015, 0.03, 0.03);
      }
    }

    // Weathered oxidized rust streaks running down
    const rustSpots = [w * 0.3, w * 0.72];
    for (const rx of rustSpots) {
      g.fillStyle = alpha('#b45309', 0.55);
      g.fillRect(rx - 0.1, -o + 0.2, 0.2, h * 0.6);
      g.fillStyle = alpha('#78350f', 0.4);
      g.fillRect(rx - 0.06, -o + 0.2, 0.12, h * 0.85);
    }

    // Emergency poly-tarp patch on weathered tin roofs
    if (has(d, 'tarp')) {
      paintRoofTarp(g, w * 0.55, h * 0.45, 1.8, 1.4);
    }

  } else if (d.roof === 'flat') {
    // Raised parapet wall with metal flashing and gravel membrane
    const pw = 0.32;
    // Parapet shadow
    g.fillStyle = 'rgba(8,9,12,0.35)';
    g.fillRect(-o + pw, -o + pw, w + 2 * o - 2 * pw, h + 2 * o - 2 * pw);

    // Pea-gravel tar bed
    g.fillStyle = wash(shade(c.roof, -0.32), ROOF_DESAT);
    g.fillRect(-o + pw + 0.04, -o + pw + 0.04, w + 2 * o - 2 * pw - 0.08, h + 2 * o - 2 * pw - 0.08);

    // Gravel stippling
    g.fillStyle = alpha('#ffffff', 0.14);
    for (let x = -o + pw + 0.3; x < w + o - pw - 0.2; x += 0.5) {
      for (let y = -o + pw + 0.3; y < h + o - pw - 0.2; y += 0.5) {
        g.fillRect(x + ((y * 7) % 0.2), y + ((x * 5) % 0.2), 0.03, 0.03);
      }
    }

    // Sky reflection rain puddle
    g.fillStyle = alpha('#38bdf8', 0.28);
    g.beginPath();
    g.ellipse(w * 0.32, h * 0.62, 0.85, 0.48, -0.2, 0, TAU);
    g.fill();
    g.strokeStyle = alpha('#0369a1', 0.4); g.lineWidth = 0.02; g.stroke();

    // Steel rooftop access bulkhead hatch
    const hx = w * 0.72, hy = h * 0.28;
    g.fillStyle = '#475569';
    g.fillRect(hx, hy, 1.1, 0.8);
    g.strokeStyle = INK; g.lineWidth = 0.035;
    g.strokeRect(hx, hy, 1.1, 0.8);
    // Angled bulkhead door & handle
    g.fillStyle = '#334155';
    g.fillRect(hx + 0.08, hy + 0.08, 0.94, 0.64);
    g.fillStyle = '#f59e0b';
    g.fillRect(hx + 0.86, hy + 0.36, 0.06, 0.08);

    // Industrial HVAC Rooftop Unit
    if (has(d, 'hvac') || d.kind === 'supermarket' || d.kind === 'warehouse' || d.kind === 'apartment') {
      paintHVAC(g, w / 2 - 1.1, h / 2 - 0.75, 2.2, 1.5);
    }

    // Parapet coping trim and mitered corner flashing
    g.strokeStyle = c.trim; g.lineWidth = pw;
    g.strokeRect(-o + pw / 2, -o + pw / 2, w + 2 * o - pw, h + 2 * o - pw);
    g.strokeStyle = INK; g.lineWidth = 0.03;
    g.strokeRect(-o + pw, -o + pw, w + 2 * o - 2 * pw, h + 2 * o - 2 * pw);

  } else if (d.roof === 'broken') {
    // Survivor collapsed roof with splintered rafter hole & heavy poly-tarp
    // Directional slope shadow
    g.fillStyle = 'rgba(0,0,0,0.2)';
    if (horizRidge) g.fillRect(-o, h / 2, w + 2 * o, h / 2 + o);
    else g.fillRect(w / 2, -o, w / 2 + o, h + 2 * o);

    // Large jagged breach
    g.fillStyle = '#111418'; // Dark interior abyss
    g.beginPath();
    g.ellipse(w / 2, h / 2, 1.5, 1.0, 0.25, 0, TAU);
    g.fill();
    g.strokeStyle = INK; g.lineWidth = 0.05; g.stroke();

    // Splintered timber rafters spanning across breach
    g.strokeStyle = '#78350f'; g.lineWidth = 0.07;
    g.beginPath();
    for (let rx = w / 2 - 1.3; rx <= w / 2 + 1.3; rx += 0.42) {
      g.moveTo(rx, h / 2 - 0.9); g.lineTo(rx + 0.1, h / 2 + 0.9);
    }
    g.stroke();

    // Blue emergency ripstop poly-tarp tied across half the hole
    paintRoofTarp(g, w / 2 - 0.4, h / 2 - 0.3, 1.9, 1.5);

  } else if (d.roof === 'hip') {
    // 4-way polygonal Hip Roof with distinct facet lighting
    const cx = w / 2, cy = h / 2;
    const rw = horizRidge ? Math.max(1.0, w - h + 0.6) : 0;
    const rh = !horizRidge ? Math.max(1.0, h - w + 0.6) : 0;

    const x0 = -o, y0 = -o, x1 = w + o, y1 = h + o;
    const rL = cx - rw / 2, rR = cx + rw / 2;
    const rT = cy - rh / 2, rB = cy + rh / 2;

    // Facet 1: North facet (Bright Sunlit)
    g.fillStyle = alpha('#ffffff', 0.14);
    g.beginPath();
    g.moveTo(x0, y0); g.lineTo(x1, y0); g.lineTo(rR, rT); g.lineTo(rL, rT);
    g.closePath(); g.fill();

    // Facet 2: South facet (Ambient Shadow)
    g.fillStyle = 'rgba(8,12,20,0.25)';
    g.beginPath();
    g.moveTo(x0, y1); g.lineTo(x1, y1); g.lineTo(rR, rB); g.lineTo(rL, rB);
    g.closePath(); g.fill();

    // Facet 3: West facet (Midtone)
    g.fillStyle = alpha('#ffffff', 0.05);
    g.beginPath();
    g.moveTo(x0, y0); g.lineTo(rL, rT); g.lineTo(rL, rB); g.lineTo(x0, y1);
    g.closePath(); g.fill();

    // Facet 4: East facet (Deep Shadow)
    g.fillStyle = 'rgba(8,12,20,0.35)';
    g.beginPath();
    g.moveTo(x1, y0); g.lineTo(x1, y1); g.lineTo(rR, rB); g.lineTo(rR, rT);
    g.closePath(); g.fill();

    // Hip rafter cap lines connecting corners to ridge ends
    g.strokeStyle = shade(c.roof, 0.25); g.lineWidth = 0.08;
    g.beginPath();
    g.moveTo(x0, y0); g.lineTo(rL, rT);
    g.moveTo(x1, y0); g.lineTo(rR, rT);
    g.moveTo(x0, y1); g.lineTo(rL, rB);
    g.moveTo(x1, y1); g.lineTo(rR, rB);
    if (horizRidge) { g.moveTo(rL, cy); g.lineTo(rR, cy); }
    else if (rh > 0) { g.moveTo(cx, rT); g.lineTo(cx, rB); }
    g.stroke();

    g.strokeStyle = INK; g.lineWidth = 0.03;
    g.stroke();

    // Decorative wrought-iron weather vane at peak
    paintWeatherVane(g, cx, cy);

  } else {
    // Gable Roof with realistic sun/shade planes, slate shingle courses, dormers, and finials
    // Sunlit pitch (North or West)
    g.fillStyle = alpha('#ffffff', 0.12);
    if (horizRidge) g.fillRect(-o, -o, w + 2 * o, h / 2 + o);
    else g.fillRect(-o, -o, w / 2 + o, h + 2 * o);

    // Shaded pitch (South or East)
    g.fillStyle = 'rgba(8,12,20,0.22)';
    if (horizRidge) g.fillRect(-o, h / 2, w + 2 * o, h / 2 + o);
    else g.fillRect(w / 2, -o, w / 2 + o, h + 2 * o);

    // Staggered slate shingle courses with notch cuts
    const shingleH = 0.42;
    g.lineWidth = 0.024;

    if (horizRidge) {
      for (let y = -o + shingleH, row = 0; y < h + o - 0.1; y += shingleH, row++) {
        // Shadow line under course
        g.strokeStyle = alpha(INK, 0.4);
        g.beginPath(); g.moveTo(-o, y); g.lineTo(w + o, y); g.stroke();
        // Top highlight on course
        g.strokeStyle = alpha('#ffffff', 0.16);
        g.beginPath(); g.moveTo(-o, y - 0.03); g.lineTo(w + o, y - 0.03); g.stroke();

        // Staggered vertical shingle gap notches
        const offset = (row % 2) * 0.35;
        g.strokeStyle = alpha(INK, 0.35);
        for (let x = -o + 0.3 + offset; x < w + o - 0.2; x += 0.7) {
          g.beginPath(); g.moveTo(x, y - shingleH); g.lineTo(x, y); g.stroke();
        }
      }
    } else {
      for (let x = -o + shingleH, col = 0; x < w + o - 0.1; x += shingleH, col++) {
        g.strokeStyle = alpha(INK, 0.4);
        g.beginPath(); g.moveTo(x, -o); g.lineTo(x, h + o); g.stroke();
        g.strokeStyle = alpha('#ffffff', 0.16);
        g.beginPath(); g.moveTo(x - 0.03, -o); g.lineTo(x - 0.03, h + o); g.stroke();

        const offset = (col % 2) * 0.35;
        g.strokeStyle = alpha(INK, 0.35);
        for (let y = -o + 0.3 + offset; y < h + o - 0.2; y += 0.7) {
          g.beginPath(); g.moveTo(x - shingleH, y); g.lineTo(x, y); g.stroke();
        }
      }
    }

    // Heavy rounded ridge cap
    g.fillStyle = shade(c.roof, 0.28);
    if (horizRidge) {
      g.fillRect(-o, h / 2 - 0.18, w + 2 * o, 0.36);
      g.strokeStyle = INK; g.lineWidth = 0.035;
      g.strokeRect(-o, h / 2 - 0.18, w + 2 * o, 0.36);
      // Segmented ridge cap tiles
      g.strokeStyle = alpha(INK, 0.5); g.lineWidth = 0.02;
      for (let x = -o + 0.45; x < w + o; x += 0.45) {
        g.beginPath(); g.moveTo(x, h / 2 - 0.18); g.lineTo(x, h / 2 + 0.18); g.stroke();
      }
    } else {
      g.fillRect(w / 2 - 0.18, -o, 0.36, h + 2 * o);
      g.strokeStyle = INK; g.lineWidth = 0.035;
      g.strokeRect(w / 2 - 0.18, -o, 0.36, h + 2 * o);
      g.strokeStyle = alpha(INK, 0.5); g.lineWidth = 0.02;
      for (let y = -o + 0.45; y < h + o; y += 0.45) {
        g.beginPath(); g.moveTo(w / 2 - 0.18, y); g.lineTo(w / 2 + 0.18, y); g.stroke();
      }
    }

    // Carved gable end bargeboards & peak finials
    g.fillStyle = c.trim;
    if (horizRidge) {
      // Finials on left and right ridge ends
      for (const fx of [-o, w + o]) {
        g.beginPath();
        g.arc(fx, h / 2, 0.12, 0, TAU);
        g.fill();
        g.strokeStyle = INK; g.lineWidth = 0.03; g.stroke();
      }
    } else {
      for (const fy of [-o, h + o]) {
        g.beginPath();
        g.arc(w / 2, fy, 0.12, 0, TAU);
        g.fill();
        g.strokeStyle = INK; g.lineWidth = 0.03; g.stroke();
      }
    }

    // Charming Attic Dormer Window on larger cottages and houses
    if (w >= 5.5 && h >= 4.5 && d.kind === 'house') {
      const dwx = w * 0.38, dwy = horizRidge ? h * 0.22 : h * 0.38;
      paintDormer(g, dwx, dwy, c);
    }
  }

  // 4. Multi-course brick chimney with corbelled cap, terracotta flue pot, and smoke curls
  if (has(d, 'chimney') || d.kind === 'house') {
    const chx = w - 1.4, chy = 0.32;
    paintChimney(g, chx, chy, c.trim);
  }

  // 5. Fascia trim with top highlight and perimeter outline
  g.strokeStyle = c.trim;
  g.lineWidth = 0.12;
  g.strokeRect(-o + 0.06, -o + 0.06, w + 2 * o - 0.12, h + 2 * o - 0.12);

  // Fascia top highlight
  g.strokeStyle = alpha('#ffffff', 0.28);
  g.lineWidth = 0.03;
  g.beginPath();
  g.moveTo(-o, -o); g.lineTo(w + o, -o);
  g.moveTo(-o, -o); g.lineTo(-o, h + o);
  g.stroke();

  // Inked perimeter stroke
  g.strokeStyle = INK;
  g.lineWidth = 0.045;
  g.strokeRect(-o, -o, w + 2 * o, h + 2 * o);
}

/** Pitch-roof attic dormer window set into the roof slope. */
function paintDormer(g: G, x: number, y: number, c: Colors): void {
  const dw = 1.2, dh = 1.0;
  // Dormer cast shadow
  g.fillStyle = 'rgba(8,12,20,0.35)';
  g.fillRect(x + 0.08, y + 0.1, dw, dh);

  // Dormer pitched roof
  g.fillStyle = wash(c.roof, ROOF_DESAT);
  g.fillRect(x, y, dw, dh);

  // Dormer sun/shade ridge
  g.fillStyle = alpha('#ffffff', 0.15);
  g.fillRect(x, y, dw / 2, dh);
  g.fillStyle = 'rgba(8,12,20,0.2)';
  g.fillRect(x + dw / 2, y, dw / 2, dh);

  // Dormer bargeboard trim
  g.strokeStyle = c.trim; g.lineWidth = 0.06;
  g.strokeRect(x, y, dw, dh);
  g.strokeStyle = INK; g.lineWidth = 0.025;
  g.strokeRect(x, y, dw, dh);

  // Miniature arched dormer window
  const wx = x + 0.22, wy = y + 0.22, ww = dw - 0.44, wh = dh - 0.44;
  g.fillStyle = '#67e8f9';
  g.fillRect(wx, wy, ww, wh);
  // Glass reflection glare
  g.fillStyle = 'rgba(255,255,255,0.45)';
  g.beginPath();
  g.moveTo(wx, wy); g.lineTo(wx + ww * 0.6, wy); g.lineTo(wx, wy + wh);
  g.closePath(); g.fill();
  // Window frame & crossbar
  g.strokeStyle = c.trim; g.lineWidth = 0.035;
  g.strokeRect(wx, wy, ww, wh);
  g.beginPath(); g.moveTo(wx + ww / 2, wy); g.lineTo(wx + ww / 2, wy + wh); g.stroke();
}

/** Brick chimney with corbelled coping, terracotta flue pot, soot rim, and smoke curls. */
function paintChimney(g: G, chx: number, chy: number, trimCol: string): void {
  // Brick chimney stack
  g.fillStyle = '#854d0e'; // Warm red-brown brick
  g.fillRect(chx, chy, 0.8, 0.8);

  // Brick course mortar lines
  g.strokeStyle = alpha('#e2e8f0', 0.5); g.lineWidth = 0.018;
  g.beginPath();
  g.moveTo(chx, chy + 0.26); g.lineTo(chx + 0.8, chy + 0.26);
  g.moveTo(chx, chy + 0.53); g.lineTo(chx + 0.8, chy + 0.53);
  g.moveTo(chx + 0.4, chy); g.lineTo(chx + 0.4, chy + 0.26);
  g.moveTo(chx + 0.25, chy + 0.26); g.lineTo(chx + 0.25, chy + 0.53);
  g.moveTo(chx + 0.55, chy + 0.26); g.lineTo(chx + 0.55, chy + 0.53);
  g.stroke();

  // Projecting corbelled stone coping crown
  g.fillStyle = trimCol;
  g.fillRect(chx - 0.06, chy - 0.06, 0.92, 0.92);
  g.strokeStyle = INK; g.lineWidth = 0.035;
  g.strokeRect(chx - 0.06, chy - 0.06, 0.92, 0.92);

  // Terracotta clay chimney flue pot
  g.fillStyle = '#c2410c';
  g.beginPath(); g.roundRect(chx + 0.16, chy + 0.16, 0.48, 0.48, 0.12); g.fill();
  g.strokeStyle = INK; g.lineWidth = 0.025; g.stroke();

  // Dark interior soot opening
  g.fillStyle = '#111418';
  g.beginPath(); g.arc(chx + 0.4, chy + 0.4, 0.18, 0, TAU); g.fill();

  // Wispy curls of drifting chimney smoke
  g.fillStyle = 'rgba(240, 243, 250, 0.38)';
  g.beginPath(); g.arc(chx + 0.48, chy - 0.22, 0.16, 0, TAU); g.fill();
  g.fillStyle = 'rgba(240, 243, 250, 0.25)';
  g.beginPath(); g.arc(chx + 0.68, chy - 0.48, 0.24, 0, TAU); g.fill();
  g.fillStyle = 'rgba(240, 243, 250, 0.14)';
  g.beginPath(); g.arc(chx + 0.94, chy - 0.78, 0.32, 0, TAU); g.fill();

  // Patches of green roof moss around chimney base
  g.fillStyle = '#4d7c0f';
  g.beginPath(); g.arc(chx - 0.12, chy + 0.6, 0.11, 0, TAU); g.fill();
  g.fillStyle = '#65a30d';
  g.beginPath(); g.arc(chx - 0.08, chy + 0.68, 0.08, 0, TAU); g.fill();
}

/** Rooftop emergency blue poly-tarp tied down by cords and concrete blocks. */
function paintRoofTarp(g: G, x: number, y: number, tw: number, th: number): void {
  // Blue tarp body
  g.fillStyle = '#2563eb';
  g.fillRect(x, y, tw, th);

  // Ripstop woven grid
  g.strokeStyle = alpha('#1d4ed8', 0.6); g.lineWidth = 0.015;
  g.beginPath();
  for (let gx = x + 0.18; gx < x + tw; gx += 0.18) { g.moveTo(gx, y); g.lineTo(gx, y + th); }
  for (let gy = y + 0.18; gy < y + th; gy += 0.18) { g.moveTo(x, gy); g.lineTo(x + tw, gy); }
  g.stroke();

  // Grommet hem border
  g.strokeStyle = '#1e40af'; g.lineWidth = 0.035; g.strokeRect(x, y, tw, th);

  // Yellow tie-down ropes
  g.strokeStyle = '#facc15'; g.lineWidth = 0.022;
  g.beginPath();
  g.moveTo(x, y); g.lineTo(x + tw, y + th);
  g.moveTo(x + tw, y); g.lineTo(x, y + th);
  g.stroke();

  // Concrete cinderblocks holding down corners
  g.fillStyle = '#94a3b8'; g.strokeStyle = INK; g.lineWidth = 0.02;
  for (const [bx, by] of [[x - 0.06, y - 0.06], [x + tw - 0.18, y - 0.06], [x - 0.06, y + th - 0.14], [x + tw - 0.18, y + th - 0.14]]) {
    g.fillRect(bx, by, 0.24, 0.16);
    g.strokeRect(bx, by, 0.24, 0.16);
  }
}

/** Decorative wrought-iron rooftop weather vane. */
function paintWeatherVane(g: G, cx: number, cy: number): void {
  // Wrought iron mast & cardinal cross
  g.strokeStyle = '#1e293b'; g.lineWidth = 0.04;
  g.beginPath();
  g.moveTo(cx - 0.35, cy); g.lineTo(cx + 0.35, cy);
  g.moveTo(cx, cy - 0.35); g.lineTo(cx, cy + 0.35);
  g.stroke();

  // Arrow / pointer quill
  g.fillStyle = '#f59e0b';
  g.beginPath();
  g.moveTo(cx + 0.42, cy); g.lineTo(cx + 0.25, cy - 0.08); g.lineTo(cx + 0.25, cy + 0.08);
  g.closePath(); g.fill();
  g.strokeStyle = INK; g.lineWidth = 0.02; g.stroke();

  // Center pivot sphere
  g.fillStyle = '#d97706';
  g.beginPath(); g.arc(cx, cy, 0.09, 0, TAU); g.fill();
  g.strokeStyle = INK; g.lineWidth = 0.025; g.stroke();
}

/** Industrial HVAC rooftop ventilation unit. */
function paintHVAC(g: G, x: number, y: number, w: number, h: number): void {
  // Unit drop shadow
  g.fillStyle = 'rgba(8,9,12,0.4)';
  g.fillRect(x + 0.1, y + 0.12, w, h);

  // Galvanized steel casing
  g.fillStyle = '#64748b';
  g.fillRect(x, y, w, h);

  // Bevel highlight
  g.fillStyle = alpha('#ffffff', 0.22);
  g.fillRect(x, y, w, 0.06);
  g.fillRect(x, y, 0.06, h);

  // Inked casing border
  g.strokeStyle = INK; g.lineWidth = 0.04;
  g.strokeRect(x, y, w, h);

  // Front intake louvers
  g.strokeStyle = '#334155'; g.lineWidth = 0.02;
  g.beginPath();
  for (let ly = y + 0.2; ly < y + h - 0.15; ly += 0.16) {
    g.moveTo(x + 0.15, ly); g.lineTo(x + w * 0.4, ly);
  }
  g.stroke();

  // Circular fan cowling & impeller blades
  const fx = x + w * 0.72, fy = y + h * 0.5, fr = Math.min(0.46, h * 0.38);
  g.fillStyle = '#1e293b';
  g.beginPath(); g.arc(fx, fy, fr, 0, TAU); g.fill();
  g.strokeStyle = '#94a3b8'; g.lineWidth = 0.035; g.stroke();

  // 4 angled impeller blades
  g.strokeStyle = '#cbd5e1'; g.lineWidth = 0.03;
  g.beginPath();
  for (let a = 0; a < Math.PI; a += Math.PI / 2) {
    g.moveTo(fx - Math.cos(a) * (fr - 0.06), fy - Math.sin(a) * (fr - 0.06));
    g.lineTo(fx + Math.cos(a) * (fr - 0.06), fy + Math.sin(a) * (fr - 0.06));
  }
  g.stroke();

  // Copper refrigerant conduits
  g.strokeStyle = '#b45309'; g.lineWidth = 0.04;
  g.beginPath();
  g.moveTo(x + w * 0.45, y + h); g.lineTo(x + w * 0.45, y + h + 0.24);
  g.moveTo(x + w * 0.55, y + h); g.lineTo(x + w * 0.55, y + h + 0.24);
  g.stroke();
}
