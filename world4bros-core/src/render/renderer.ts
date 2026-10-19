import { makeRng, hashStr } from '../engine/rng';
import { clamp } from '../engine/math';
import type { Game, Target } from '../sim/game';
import { Comp } from '../sim/components';
import { CHUNK } from '../sim/worldgen';
import { Camera } from './camera';
import { Lighting } from './lighting';
import { SpriteCache, blit, ppuFor, setSpriteQuality } from './sprites';
import { Atmosphere, buildingLights, drawBoxShadow, ATMOS_TUNING, type PointLight } from './atmosphere';
import { CharacterRenderer, type Pose } from './characterRenderer';
import { VehicleRenderer } from './vehicleRenderer';
import { ZombieRenderer } from './zombieRenderer';
import { PetRenderer } from './petRenderer';
import { AnimalRenderer } from './animalRenderer';
import { OBJECT_DRAWERS } from './objects';
import { drawSandbags, drawScrapWall, drawSlimeDrum } from './drawers/objectDrawers';
import { STRUCTURE_DRAWERS } from './structures';
import { itemIcon } from './icons';
import { paintChunk } from './tiles';
import { BuildingRenderer } from './buildings';
import { type G, INK, TAU, alpha, inked, poly, smooth } from './ink';
import type { ArenaDef } from '../sim/arena';

export interface BuildGhost { id: string; tx: number; ty: number; ok: boolean }
export interface AimRenderState {
  active: boolean;
  aimSystem: 'laser' | 'bloom' | 'arc' | 'burst';
  origin: { x: number; y: number };
  target: { x: number; y: number };
  spread: number;
  maxRange: number;
  gunName: string;
  ammoName?: string;
  ammoCount: number;
  visualInfo?: {
    rangefinder?: boolean;
    targetHealthBar?: boolean;
    threatScanner?: boolean;
    trajectoryArc?: boolean;
    nightVisionReticle?: boolean;
  };
  aimedTarget?: {
    name: string;
    hp: number;
    maxHp: number;
    dist: number;
  } | null;
  threats?: {
    x: number;
    y: number;
    name: string;
    hp: number;
    maxHp: number;
    dist: number;
  }[];
}
export interface RenderState {
  build: BuildGhost | null;
  target: Target;
  debug: boolean;
  showClaims: boolean;
  brush?: { x: number; y: number; r: number } | null;
  moveMarker?: { x: number; y: number } | null;
  aim?: AimRenderState | null;
}
interface Particle { x: number; y: number; vx: number; vy: number; life: number; max: number; color: string; size: number; rot: number }
interface Floater { x: number; y: number; text: string; life: number }
interface Item { y: number; draw: () => void }

const CHIP: Record<string, string> = {
  pine: '#587063', snow_pine: '#cfd6dc', oak: '#5c6e47', bush: '#566a4a', berry_bush: '#a1403a',
  rock: '#8b9097', boulder: '#8b9097', log: '#8a6a4a', crate: '#8a6a4a', dead_tree: '#6a625a',
  cactus: '#5f7a5a', mushroom: '#bba98c', reeds: '#7a8456', tent: '#857a62', fuel_pump: '#c4473d',
  radio_tower: '#8b9097', sandbags: '#9a8a6e', wrecked_car: '#6a4a35',
  iron_node: '#8fa1b3', copper_node: '#c97746', coal_node: '#343a40', oil_seam: '#252a32', crystal_geode: '#b88ef2',
};

export class Renderer {
  readonly cam = new Camera();
  readonly cache = new SpriteCache();
  readonly chars: CharacterRenderer;
  readonly vehicles = new VehicleRenderer();
  readonly zombies = new ZombieRenderer();
  readonly pets = new PetRenderer();
  readonly animals = new AnimalRenderer();
  private bld: BuildingRenderer;
  private lighting = new Lighting();
  /** Weather, atmosphere and screen polish (quality dial: atmos.setQuality('off'|'low'|'high')). */
  readonly atmos = new Atmosphere();
  private extraLights: PointLight[] = [];
  private chunks = new Map<string, HTMLCanvasElement>();
  private particles: Particle[] = [];
  private floaters: Floater[] = [];
  private grain = document.createElement('canvas');
  private g: G;
  private t = 0;
  W = 0; H = 0; dpr = 1;
  stats = { chunks: 0, drawn: 0 };

  constructor(readonly canvas: HTMLCanvasElement, private game: Game) {
    this.g = canvas.getContext('2d') as G;
    this.chars = new CharacterRenderer(this.cache, game.content);
    this.bld = new BuildingRenderer(game.content);
    const gg = this.grain.getContext('2d') as G; this.grain.width = this.grain.height = 192;
    const rng = makeRng(7);
    for (let i = 0; i < 2600; i++) { gg.fillStyle = rng.chance(0.5) ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)'; gg.fillRect(rng.range(0, 192), rng.range(0, 192), 1.3, 1.3); }
    game.events.on('hit', (e) => {
      const col = e.kind === 'flesh' ? '#9e2a2b' : CHIP[e.kind] ?? '#8a6a4a';
      for (let i = 0; i < (e.kind === 'flesh' ? 9 : 7); i++) {
        const a = makeRng(Math.floor(this.t * 1000) + i).range(0, TAU), sp = 1.2 + i * 0.25;
        this.particles.push({ x: e.x, y: e.y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 0.55, max: 0.55, color: col, size: e.kind === 'flesh' ? 0.065 : 0.05, rot: a });
      }
      this.cam.shake = Math.min(0.5, this.cam.shake + (e.kind === 'flesh' ? 0.35 : 0.25));
    });
    game.events.on('harvest', (e) => this.floaters.push({ x: e.x, y: e.y - 0.3, text: `+${e.qty} ${game.content.items.get(e.item)?.name ?? e.item}`, life: 1.4 }));
    const t = game.playerT; this.cam.snap(t.x, t.y);
  }

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.floor(this.canvas.clientWidth * this.dpr)), h = Math.max(2, Math.floor(this.canvas.clientHeight * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.W = w; this.H = h;
  }

  private tilePx = 40;
  private chunkCanvas(cx: number, cy: number, budget: { n: number }): HTMLCanvasElement | null {
    const k = `${cx},${cy}`;
    let c = this.chunks.get(k);
    if (!c && budget.n > 0) { budget.n--; c = paintChunk(this.game.world, this.game.world.chunkAt(cx, cy), this.tilePx); this.chunks.set(k, c); }
    return c ?? null;
  }

  render(dt: number, st: RenderState): void {
    this.t += dt; this.resize();
    const { g, W, H, game, cam } = this;
    const pt = game.playerT;
    cam.update(pt.x, pt.y, dt);
    setSpriteQuality(this.atmos.quality);
    const z = cam.zoom * this.dpr, ppu = ppuFor(cam.zoom), lod = cam.lod;
    const shx = (Math.random() - 0.5) * cam.shake * 0.12, shy = (Math.random() - 0.5) * cam.shake * 0.12;
    const cx = cam.x + shx, cy = cam.y + shy;
    const hw = W / z / 2 + 2, hh = H / z / 2 + 2;
    g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = '#1b1f22'; g.fillRect(0, 0, W, H);
    g.setTransform(z, 0, 0, z, W / 2 - cx * z, H / 2 - cy * z);

    // ground (resolution follows zoom so close-ups stay crisp and ultra-wide stays fast)
    const wantPx = cam.zoom < 24 ? 18 : cam.zoom <= 72 ? 40 : cam.zoom <= 120 ? 72 : 112;
    if (wantPx !== this.tilePx) { this.tilePx = wantPx; this.chunks.clear(); }
    const chunkRadius = Math.ceil(Math.max(hw, hh) / CHUNK);
    const budget = { n: cam.zoom < 28 ? Math.max(8, chunkRadius * 2) : wantPx > 72 ? 4 : 2 };
    const c0x = Math.floor((cx - hw) / CHUNK), c1x = Math.floor((cx + hw) / CHUNK), c0y = Math.floor((cy - hh) / CHUNK), c1y = Math.floor((cy + hh) / CHUNK);
    g.imageSmoothingEnabled = true;
    const dirty = this.game.world.dirty;
    if (dirty.size) { if (dirty.has('*')) this.chunks.clear(); else for (const k of dirty) this.chunks.delete(k); dirty.clear(); }
    for (let y = c0y; y <= c1y; y++) for (let x = c0x; x <= c1x; x++) { const c = this.chunkCanvas(x, y, budget); if (c) g.drawImage(c, x * CHUNK - 0.01, y * CHUNK - 0.01, CHUNK + 0.02, CHUNK + 0.02); }
    if (st.brush) { g.beginPath(); g.arc(st.brush.x, st.brush.y, Math.max(0.5, st.brush.r), 0, TAU); g.strokeStyle = 'rgba(255,255,255,0.9)'; g.lineWidth = 0.06; g.setLineDash([0.3, 0.2]); g.stroke(); g.setLineDash([]); g.fillStyle = 'rgba(255,255,255,0.12)'; g.fill(); }
    const maxCache = Math.max(160, (chunkRadius * 2 + 3) ** 2);
    if (this.chunks.size > maxCache) for (const k of this.chunks.keys()) { const [kx, ky] = k.split(',').map(Number) as [number, number]; if (Math.abs(kx - cx / CHUNK) > chunkRadius + 3 || Math.abs(ky - cy / CHUNK) > chunkRadius + 3) this.chunks.delete(k); }
    this.stats.chunks = this.chunks.size;

    // atmosphere: weather, wet ground, cloud shadows, ambient motes (see atmosphere.ts registry)
    const climate = game.content.biomes.find((b) => b.id === game.world.biomeAt(cx, cy))?.climate ?? { temp: 0.5, moist: 0.5 };
    this.atmos.update(g, dt, { W, H, z, camX: cx, camY: cy, dayT: game.clock.t, darkness: game.clock.darkness, day: game.clock.day, lod, climate, seed: game.config.seed });
    const sun = this.atmos.sun, fancy = this.atmos.quality !== 'off';
    this.atmos.draw('ground');

    // claim rings
    if (st.showClaims || st.build) for (const c of game.base.claims()) { g.beginPath(); g.arc(c.x, c.y, c.r, 0, TAU); g.setLineDash([0.3, 0.25]); g.strokeStyle = 'rgba(214,196,130,0.5)'; g.lineWidth = 0.05; g.stroke(); g.setLineDash([]); g.fillStyle = 'rgba(214,196,130,0.04)'; g.fill(); }

    const items: Item[] = [];
    const r = Math.max(hw, hh) + 1.5;
    // floors and conveyor belts under everything
    for (const e of game.base.entities()) {
      const s = game.ecs.get(e, Comp.Structure)!, t = game.ecs.get(e, Comp.Transform);
      if (s.def.id.startsWith('floor') || s.def.id === 'conveyor_belt') {
        this.drawStructure(s.def.id, s.def.size, s.tx, s.ty, false, ppu, t?.angle ?? 0);
      }
    }

    // Active Arena floor, roads and boundaries
    if (game.activeArena) {
      const ar = game.activeArena;
      const ox = ar.originX, oy = ar.originY;
      const def = ar.def;

      // Arena sand/gravel battle floor
      g.fillStyle = '#2b2d2f';
      g.fillRect(ox, oy, def.width, def.height);

      // Arena boundary perimeter warning border
      g.strokeStyle = '#f59e0b';
      g.lineWidth = 0.12;
      g.strokeRect(ox, oy, def.width, def.height);

      // Roads if defined
      if (def.roads) {
        g.fillStyle = '#1e2022';
        for (const road of def.roads) {
          g.fillRect(ox + road.x, oy + road.y, road.w, road.h);
        }
      }

      // Spawns (visual team markers)
      for (let i = 0; i < def.spawns.length; i++) {
        const sp = def.spawns[i];
        g.strokeStyle = i === 0 ? '#38bdf8' : i === 1 ? '#f43f5e' : '#a855f7';
        g.lineWidth = 0.05;
        g.beginPath();
        g.arc(ox + sp.x, oy + sp.y, 0.65, 0, TAU);
        g.stroke();
        g.fillStyle = 'rgba(255,255,255,0.1)';
        g.fill();
      }

      // Cream pies (pickups on ground)
      for (const p of def.pies) {
        items.push({
          y: oy + p.y + 0.1,
          draw: () => {
            const px = ox + p.x, py = oy + p.y;
            // Tin pie plate
            g.fillStyle = '#94a3b8';
            g.beginPath();
            g.arc(px, py, 0.28, 0, TAU);
            g.fill();
            g.strokeStyle = '#475569';
            g.lineWidth = 0.03;
            g.stroke();
            // Fluffy white/custard cream topping
            g.fillStyle = '#fef08a';
            g.beginPath();
            g.arc(px, py, 0.2, 0, TAU);
            g.fill();
            // Cherry on top
            g.fillStyle = '#dc2626';
            g.beginPath();
            g.arc(px, py - 0.02, 0.06, 0, TAU);
            g.fill();
          }
        });
      }

      // Crates (solid bullet stoppers)
      for (const c of def.crates) {
        items.push({
          y: oy + c.y + c.h,
          draw: () => {
            const cx0 = ox + c.x, cy0 = oy + c.y;
            g.fillStyle = '#854d0e';
            g.fillRect(cx0, cy0, c.w, c.h);
            g.strokeStyle = '#451a03';
            g.lineWidth = 0.06;
            g.strokeRect(cx0, cy0, c.w, c.h);
            // X-cross brace
            g.beginPath();
            g.moveTo(cx0, cy0); g.lineTo(cx0 + c.w, cy0 + c.h);
            g.moveTo(cx0 + c.w, cy0); g.lineTo(cx0, cy0 + c.h);
            g.stroke();
          }
        });
      }

      // Sandbags (bullet stoppers)
      for (const s of def.sandbags) {
        const sw = s.direction === 'h' ? s.length : 1;
        const sh = s.direction === 'h' ? 1 : s.length;
        items.push({
          y: oy + s.y + sh,
          draw: () => {
            if (s.direction === 'h') {
              for (let i = 0; i < s.length; i++) {
                drawSandbags({ ctx: g, x: ox + s.x + 0.5 + i, y: oy + s.y + 0.5, scale: 1 / 48, hpRatio: 1.0 });
              }
            } else {
              for (let i = 0; i < s.length; i++) {
                g.save();
                g.translate(ox + s.x + 0.5, oy + s.y + 0.5 + i);
                g.rotate(Math.PI / 2);
                drawSandbags({ ctx: g, x: 0, y: 0, scale: 1 / 48, hpRatio: 1.0 });
                g.restore();
              }
            }
          }
        });
      }

      // Low walls (bullet stoppers)
      for (const w of def.lowWalls) {
        items.push({
          y: oy + w.y + w.h,
          draw: () => {
            const count = Math.max(1, Math.round(w.w));
            const step = w.w / count;
            for (let i = 0; i < count; i++) {
              drawScrapWall({ ctx: g, x: ox + w.x + step * (i + 0.5), y: oy + w.y + w.h / 2, scale: 1 / 48, hpRatio: 1.0 });
            }
          }
        });
      }

      // Slime drums / barrels (movement blocking)
      for (const b of def.barrels) {
        items.push({
          y: oy + b.y + 0.4,
          draw: () => {
            const bx = ox + b.x, by = oy + b.y;
            drawSlimeDrum({ ctx: g, x: bx, y: by, scale: 1 / 48, hpRatio: 1.0, timeMs: this.t * 1000 });
          }
        });
      }

      // Trees (movement blocking only)
      for (const t of def.trees) {
        items.push({
          y: oy + t.y + 0.5,
          draw: () => {
            const tx = ox + t.x, ty = oy + t.y;
            // Tree trunk
            g.fillStyle = '#78350f';
            g.beginPath();
            g.arc(tx, ty, 0.3, 0, TAU);
            g.fill();
            // Canopy
            g.fillStyle = '#15803d';
            g.beginPath();
            g.arc(tx, ty - 0.2, 0.7, 0, TAU);
            g.fill();
            g.strokeStyle = '#14532d';
            g.lineWidth = 0.05;
            g.stroke();
          }
        });
      }
    }
    // shadows: one batched path; direction and length follow the sun (atmosphere.sunAt)
    const sunAng = Math.atan2(sun.dy, sun.dx);
    g.fillStyle = fancy ? `rgba(21,18,30,${sun.alpha.toFixed(3)})` : 'rgba(8,9,12,0.26)'; g.beginPath();
    for (const o of game.world.objectsNear(cx, cy, r)) {
      const d = OBJECT_DRAWERS[o.kind]; if (!d) continue;
      const [rx, ry, ox, oy] = d.shadow;
      if (!fancy) { g.moveTo(o.x + ox + rx, o.y + oy); g.ellipse(o.x + ox, o.y + oy, rx, ry, 0.5, 0, TAU); continue; }
      const reach = rx * sun.len * 0.7, sx = o.x + ox + sun.dx * reach, sy = o.y + oy + sun.dy * reach;
      g.moveTo(sx + rx, sy); g.ellipse(sx, sy, rx * (1 + sun.len * 0.35), ry * 0.92, sunAng, 0, TAU);
    }
    g.fill();
    // building floors, furniture and walls (people walk on top of these)
    const nearBuildings = game.world.buildingsNear(cx, cy, r);
    if (fancy) for (const b of nearBuildings) drawBoxShadow(g, b.x, b.y, b.w, b.h, ATMOS_TUNING.buildingHeight, sun);
    for (const b of nearBuildings) this.bld.drawBase(g, b, ppu);
    // world objects
    const swayKinds = new Set(['pine', 'snow_pine', 'oak', 'bush', 'berry_bush', 'reeds', 'cactus', 'mushroom']);
    for (const o of game.world.objectsNear(cx, cy, r)) {
      const d = OBJECT_DRAWERS[o.kind]; if (!d) continue;
      const rot = (hashStr(o.id) % 628) / 100, v = o.variant;
      const sway = lod > 0 && swayKinds.has(o.kind) ? Math.sin(this.t * (0.9 + this.atmos.wind * 1.4) + o.x * 0.7 + o.y * 0.5) * 0.012 * (1 + this.atmos.wind * 2.5) : 0;
      items.push({ y: o.y, draw: () => {
        const spr = this.cache.get(`o:${o.kind}:${v}`, ppu, d.bounds[0], d.bounds[1], d.bounds[2], d.bounds[3], (gg) => d.draw(gg, makeRng(hashStr(`${o.kind}:${v}`)), v));
        const baseRot = d.upright ? 0 : (o.kind === 'log' || o.kind === 'tent' ? rot * 0.25 : rot);
        blit(g, spr, o.x, o.y, baseRot + sway);
      } });
    }
    // structures
    for (const e of game.base.entities()) {
      const s = game.ecs.get(e, Comp.Structure)!, t = game.ecs.get(e, Comp.Transform);
      if (s.def.id.startsWith('floor') || s.def.id === 'conveyor_belt') continue;
      items.push({ y: s.ty + s.def.size[1], draw: () => {
        this.drawStructure(s.def.id, s.def.size, s.tx, s.ty, s.open, ppu, t?.angle ?? 0);
        if (s.def.id === 'campfire') this.flame(s.tx + 0.5, s.ty + 0.5, 1.1);
        if (s.def.id === 'torch_post') this.flame(s.tx + 0.5, s.ty + 0.5, 0.7);
        if (s.def.id === 'stone_furnace') {
          const mach = game.ecs.get(e, Comp.Machine);
          if (mach?.targetOre || mach?.outputItem) this.flame(s.tx + 1.0, s.ty + 1.15, 0.95);
        }
      } });
    }

    // items in transit on conveyor belts
    for (const e of game.ecs.query(Comp.Conveyor, Comp.Structure, Comp.Transform)) {
      const conv = game.ecs.get(e, Comp.Conveyor)!, st = game.ecs.get(e, Comp.Transform)!;
      const cosA = Math.cos(st.angle), sinA = Math.sin(st.angle);
      for (const item of conv.items) {
        const dist = item.progress - 0.5;
        const ix = st.x + cosA * dist, iy = st.y + sinA * dist;
        items.push({
          y: iy + 0.05,
          draw: () => {
            const icon = itemIcon(game.content, item.item, 36);
            g.drawImage(icon, ix - 0.16, iy - 0.16, 0.32, 0.32);
          }
        });
      }
    }

    // animated mechanical inserter arms
    for (const e of game.ecs.query(Comp.Inserter, Comp.Structure, Comp.Transform)) {
      const ins = game.ecs.get(e, Comp.Inserter)!, st = game.ecs.get(e, Comp.Transform)!;
      items.push({
        y: st.y + 0.1,
        draw: () => {
          const swingAng = st.angle + Math.PI - ins.armProgress * Math.PI;
          const armLen = 0.42;
          const hx = st.x + Math.cos(swingAng) * armLen, hy = st.y + Math.sin(swingAng) * armLen;
          g.save();
          g.strokeStyle = '#adb5bd';
          g.lineWidth = 0.045;
          g.beginPath();
          g.moveTo(st.x, st.y);
          g.lineTo(hx, hy);
          g.stroke();
          g.fillStyle = '#ef233c';
          g.beginPath();
          g.arc(hx, hy, 0.06, 0, Math.PI * 2);
          g.fill();
          if (ins.heldItem) {
            const icon = itemIcon(game.content, ins.heldItem, 32);
            g.drawImage(icon, hx - 0.14, hy - 0.14, 0.28, 0.28);
          }
          g.restore();
        }
      });
    }
    // characters
    const drivingVeh = game.drivingVehicle();
    for (const e of game.ecs.query(Comp.Character, Comp.Transform)) {
      if (e === game.player && drivingVeh) continue; // driver is rendered inside the vehicle
      const ch = game.ecs.get(e, Comp.Character)!, t = game.ecs.get(e, Comp.Transform)!;
      const pose: Pose = { phase: ch.anim.phase, walk: ch.anim.walkBlend, swing: ch.anim.state === 'swing' ? clamp(ch.anim.swingT / ch.anim.swingDur, 0, 1) : -1, time: this.t + e };
      items.push({ y: t.y + 0.2, draw: () => {
        g.save(); g.translate(t.x, t.y); g.rotate(t.angle);
        this.chars.draw(g, ch.look, pose, ppu);
        if (game.content.items.get(ch.look.outfit.mainHand?.item ?? '')?.equip?.lightRadius) this.flame(0.66, 0.22, 0.45, true);
        g.restore();
        if (e !== game.player && lod > 0) { g.font = '0.26px "Crimson Pro", Georgia, serif'; g.textAlign = 'center'; g.fillStyle = 'rgba(233,225,204,0.85)'; g.strokeStyle = 'rgba(10,12,16,0.8)'; g.lineWidth = 0.05; g.strokeText(ch.name, t.x, t.y - 0.62); g.fillText(ch.name, t.x, t.y - 0.62); }
      } });
    }
    // vehicles
    for (const e of game.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const v = game.ecs.get(e, Comp.Vehicle)!, t = game.ecs.get(e, Comp.Transform)!;
      items.push({
        y: t.y + 0.35,
        draw: () => {
          g.save();
          g.fillStyle = 'rgba(8,9,12,0.35)';
          g.beginPath();
          const vs = fancy ? 0.35 * sun.len : 0; // short sun-cast offset
          g.ellipse(t.x + sun.dx * vs, t.y + 0.04 + sun.dy * vs, v.def.size[0] * 0.54, v.def.size[1] * 0.48, t.angle, 0, TAU);
          g.fill();
          g.restore();
          this.vehicles.draw(g, v, t, ppu, this.t);
        }
      });
    }
    // zombies
    for (const e of game.ecs.query(Comp.Zombie, Comp.Transform)) {
      const z = game.ecs.get(e, Comp.Zombie)!, t = game.ecs.get(e, Comp.Transform)!;
      items.push({ y: t.y + 0.15, draw: () => this.zombies.draw(g, z, t.x, t.y, t.angle, 1 / 60) });
    }
    // pets
    for (const e of game.ecs.query(Comp.Pet, Comp.Transform)) {
      const pet = game.ecs.get(e, Comp.Pet)!, t = game.ecs.get(e, Comp.Transform)!;
      items.push({ y: t.y + 0.1, draw: () => this.pets.draw(g, pet, t.x, t.y, t.angle, 1 / 60) });
    }
    // animals & wildlife
    for (const e of game.ecs.query(Comp.Animal, Comp.Transform)) {
      const a = game.ecs.get(e, Comp.Animal)!, t = game.ecs.get(e, Comp.Transform)!;
      items.push({ y: t.y + a.def.radius * 0.4, draw: () => this.animals.draw(g, a, t.x, t.y, t.angle, 1 / 60) });
    }
    // target dummies & steel proving plates
    for (const de of game.ecs.query(Comp.TargetDummy, Comp.Transform)) {
      const dummy = game.ecs.get(de, Comp.TargetDummy)!, dtPos = game.ecs.get(de, Comp.Transform)!;
      items.push({
        y: dtPos.y + 0.25,
        draw: () => {
          g.save();
          const wobble = dummy.wobbleTimer > 0 ? Math.sin(dummy.wobbleTimer * 26) * (dummy.wobbleTimer * 0.24) : 0;
          g.translate(dtPos.x, dtPos.y);

          // Ground shadow
          g.fillStyle = 'rgba(10,12,15,0.45)';
          g.beginPath();
          g.ellipse(0, 0.12, 0.32, 0.12, 0, 0, TAU);
          g.fill();

          // Steel support A-frame tripod
          g.strokeStyle = '#334155';
          g.lineWidth = 0.045;
          g.beginPath();
          g.moveTo(-0.24, 0.15); g.lineTo(0, -0.32); g.lineTo(0.24, 0.15);
          g.stroke();

          // Reactive swinging plate (rotates around upper hinge with wobble)
          g.save();
          g.translate(0, -0.32);
          g.rotate(wobble);

          // Upper hinge bracket
          g.fillStyle = '#64748b';
          g.fillRect(-0.06, -0.04, 0.12, 0.08);

          // Silhouette / Steel Gong Target Plate
          const isSteel = dummy.distanceLabel.includes('m') && parseInt(dummy.distanceLabel, 10) % 20 === 0;
          if (isSteel) {
            // Circular Steel Gong
            g.fillStyle = dummy.wobbleTimer > 0 ? '#fde047' : '#94a3b8';
            g.beginPath();
            g.arc(0, 0.22, 0.22, 0, TAU);
            g.fill();
            g.strokeStyle = '#0f172a';
            g.lineWidth = 0.035;
            g.stroke();
            // Center red bullseye dot
            g.fillStyle = '#ef4444';
            g.beginPath();
            g.arc(0, 0.22, 0.08, 0, TAU);
            g.fill();
          } else {
            // Torso Silhouette Target Board
            g.fillStyle = dummy.wobbleTimer > 0 ? '#fef08a' : '#d97706';
            g.beginPath();
            // Head
            g.arc(0, 0.06, 0.1, 0, TAU);
            // Torso shoulders
            g.rect(-0.18, 0.14, 0.36, 0.28);
            g.fill();
            g.strokeStyle = '#1e1b18';
            g.lineWidth = 0.03;
            g.stroke();

            // Concentric Bullseye Rings
            g.strokeStyle = 'rgba(239, 68, 68, 0.85)';
            g.lineWidth = 0.025;
            g.beginPath(); g.arc(0, 0.26, 0.09, 0, TAU); g.stroke();
            g.fillStyle = '#ef4444';
            g.beginPath(); g.arc(0, 0.26, 0.04, 0, TAU); g.fill();
          }

          // Flash sparkle if recently hit
          if (dummy.wobbleTimer > 0.18) {
            g.fillStyle = '#ffffff';
            g.beginPath();
            g.arc(0, 0.22, 0.06 + dummy.wobbleTimer * 0.15, 0, TAU);
            g.fill();
          }

          g.restore(); // end swinging plate

          // Distance Tag Banner
          g.font = 'bold 0.22px "Outfit", sans-serif';
          g.textAlign = 'center';
          g.fillStyle = '#f8fafc';
          g.strokeStyle = '#0f172a';
          g.lineWidth = 0.045;
          g.strokeText(dummy.distanceLabel, 0, -0.48);
          g.fillText(dummy.distanceLabel, 0, -0.48);

          // Damage Number & Falloff readout
          if (dummy.lastDamageTaken > 0 && dummy.wobbleTimer > 0) {
            g.font = 'bold 0.26px "Outfit", sans-serif';
            g.fillStyle = '#ef4444';
            g.strokeStyle = '#18181b';
            g.lineWidth = 0.055;
            g.strokeText(`-${dummy.lastDamageTaken} DMG`, 0, -0.80 - (0.35 - dummy.wobbleTimer) * 0.4);
            g.fillText(`-${dummy.lastDamageTaken} DMG`, 0, -0.80 - (0.35 - dummy.wobbleTimer) * 0.4);
          }
          g.restore();
        }
      });
    }
    // flying ballistic projectiles
    for (const pe of game.ecs.query(Comp.Projectile, Comp.Transform)) {
      const proj = game.ecs.get(pe, Comp.Projectile)!, pt = game.ecs.get(pe, Comp.Transform)!;
      items.push({
        y: pt.y + 0.05,
        draw: () => {
          g.save();
          if (proj.trail.length > 1) {
            g.beginPath();
            g.moveTo(proj.trail[0].x, proj.trail[0].y);
            for (let i = 1; i < proj.trail.length; i++) g.lineTo(proj.trail[i].x, proj.trail[i].y);
            g.strokeStyle = alpha(proj.color, 0.45);
            g.lineWidth = proj.radius * 1.5;
            g.lineCap = 'round';
            g.stroke();
          }
          g.fillStyle = proj.color;
          g.beginPath();
          g.arc(pt.x, pt.y, proj.radius, 0, TAU);
          g.fill();
          g.strokeStyle = '#ffffff';
          g.lineWidth = 0.015;
          g.stroke();
          g.restore();
        }
      });
    }
    items.sort((a, b) => a.y - b.y);
    for (const it of items) it.draw();
    // roofs sit above everything and fade away while you are inside
    for (const b of nearBuildings) {
      this.bld.drawRoof(g, b, ppu, dt, pt.x, pt.y);
      const bDef = game.content.buildings.find((d) => d.id === b.def);
      if (bDef && (bDef.features?.includes('chimney') || bDef.kind === 'house')) {
        if (Math.random() < dt * 2.8) {
          const chx = b.x + b.w - 0.85;
          const chy = b.y + 0.65;
          this.particles.push({
            x: chx + (Math.random() - 0.5) * 0.14,
            y: chy,
            vx: (Math.random() - 0.5) * 0.18 + 0.1,
            vy: -0.55 - Math.random() * 0.35,
            life: 2.4,
            max: 2.4,
            color: 'rgba(210,215,220,0.4)',
            size: 0.13 + Math.random() * 0.08,
            rot: Math.random() * 6.28,
          });
        }
      }
    }
    this.atmos.draw('over'); // cloud shadows, mist, rain, snow, leaves above roofs
    this.stats.drawn = items.length;

    // click-to-move marker beacon
    if (st.moveMarker) {
      const mx = st.moveMarker.x, my = st.moveMarker.y;
      const pulse = (Math.sin(this.t * 9) + 1) * 0.5;
      const rOuter = 0.28 + pulse * 0.12;
      const rInner = 0.1 + (1 - pulse) * 0.04;
      g.save();
      // Outer animated dashed ring
      g.beginPath();
      g.arc(mx, my, rOuter, 0, TAU);
      g.setLineDash([0.14, 0.1]);
      g.lineDashOffset = this.t * 0.8;
      g.strokeStyle = 'rgba(74, 222, 128, 0.88)';
      g.lineWidth = 0.04;
      g.stroke();
      // Inner glowing core
      g.beginPath();
      g.arc(mx, my, rInner, 0, TAU);
      g.setLineDash([]);
      g.fillStyle = 'rgba(110, 245, 160, 0.5)';
      g.fill();
      g.strokeStyle = 'rgba(235, 255, 235, 0.95)';
      g.lineWidth = 0.025;
      g.stroke();
      // Crosshair ticks
      const tick = 0.09;
      g.beginPath();
      g.moveTo(mx - tick, my); g.lineTo(mx + tick, my);
      g.moveTo(mx, my - tick); g.lineTo(mx, my + tick);
      g.strokeStyle = 'rgba(255, 255, 255, 0.95)';
      g.lineWidth = 0.02;
      g.stroke();
      g.restore();
    }

    // target ring
    if (st.target?.kind === 'object') {
      const o = st.target.obj, def = game.content.objects.get(o.kind), rr = (def?.radius ?? 0.4) + 0.3;
      g.save(); g.beginPath(); g.ellipse(o.x, o.y, rr, rr * 0.9, 0, 0, TAU); g.setLineDash([0.18, 0.14]); g.lineDashOffset = -this.t * 0.6; g.strokeStyle = st.target.ok ? 'rgba(233,225,204,0.85)' : 'rgba(201,100,80,0.85)'; g.lineWidth = 0.045; g.stroke(); g.restore();
    } else if (st.target?.kind === 'vehicle') {
      const vt = game.ecs.get(st.target.entity, Comp.Transform);
      if (vt) {
        g.save(); g.beginPath(); g.ellipse(vt.x, vt.y, 1.45, 0.95, vt.angle, 0, TAU);
        g.setLineDash([0.22, 0.16]); g.lineDashOffset = -this.t * 0.6;
        g.strokeStyle = 'rgba(233,225,204,0.85)'; g.lineWidth = 0.045; g.stroke(); g.restore();
      }
    } else if (st.target?.kind === 'zombie') {
      const zt = game.ecs.get(st.target.entity, Comp.Transform);
      if (zt) {
        g.save(); g.beginPath(); g.ellipse(zt.x, zt.y, 0.48, 0.42, 0, 0, TAU);
        g.setLineDash([0.16, 0.12]); g.lineDashOffset = -this.t * 0.9;
        g.strokeStyle = 'rgba(230,80,60,0.92)'; g.lineWidth = 0.05; g.stroke(); g.restore();
      }
    } else if (st.target?.kind === 'pet') {
      const pt = game.ecs.get(st.target.entity, Comp.Transform);
      if (pt) {
        g.save(); g.beginPath(); g.ellipse(pt.x, pt.y, 0.44, 0.36, 0, 0, TAU);
        g.setLineDash([0.16, 0.12]); g.lineDashOffset = -this.t * 0.6;
        g.strokeStyle = 'rgba(249,196,90,0.95)'; g.lineWidth = 0.045; g.stroke(); g.restore();
      }
    } else if (st.target?.kind === 'animal') {
      const at = game.ecs.get(st.target.entity, Comp.Transform);
      const a = game.ecs.get(st.target.entity, Comp.Animal);
      if (at && a) {
        g.save(); g.beginPath(); g.ellipse(at.x, at.y, a.def.radius + 0.14, a.def.radius + 0.08, at.angle, 0, TAU);
        g.setLineDash([0.16, 0.12]); g.lineDashOffset = -this.t * 0.6;
        g.strokeStyle = 'rgba(74,222,128,0.95)'; g.lineWidth = 0.045; g.stroke(); g.restore();
      }
    }
    // build ghost
    if (st.build) {
      const def = game.content.structures.get(st.build.id);
      if (def) {
        g.save(); g.globalAlpha = 0.6; this.drawStructure(def.id, def.size, st.build.tx, st.build.ty, false, ppu); g.globalAlpha = 1;
        g.fillStyle = st.build.ok ? 'rgba(120,200,120,0.28)' : 'rgba(210,90,70,0.32)'; g.fillRect(st.build.tx, st.build.ty, def.size[0], def.size[1]);
        g.strokeStyle = st.build.ok ? 'rgba(160,230,150,0.9)' : 'rgba(240,120,100,0.9)'; g.lineWidth = 0.05; g.strokeRect(st.build.tx, st.build.ty, def.size[0], def.size[1]);
        if (def.claimRadius) { g.beginPath(); g.arc(st.build.tx + 0.5, st.build.ty + 0.5, def.claimRadius, 0, TAU); g.setLineDash([0.3, 0.25]); g.strokeStyle = 'rgba(214,196,130,0.8)'; g.lineWidth = 0.06; g.stroke(); g.setLineDash([]); }
        g.restore();
      }
    }
    // particles and floating text
    for (const p of this.particles) {
      p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.94; p.vy *= 0.94; p.rot += dt * 8;
      g.save(); g.translate(p.x, p.y); g.rotate(p.rot); g.globalAlpha = clamp(p.life / p.max, 0, 1); g.fillStyle = p.color; g.strokeStyle = INK; g.lineWidth = 0.012; g.fillRect(-p.size, -p.size * 0.5, p.size * 2, p.size); g.strokeRect(-p.size, -p.size * 0.5, p.size * 2, p.size); g.restore();
    }
    this.particles = this.particles.filter((p) => p.life > 0);
    for (const f of this.floaters) {
      f.life -= dt; f.y -= dt * 0.55;
      g.save(); g.globalAlpha = clamp(f.life, 0, 1); g.font = '0.34px "Special Elite", "Crimson Pro", Georgia, serif'; g.textAlign = 'center'; g.strokeStyle = 'rgba(10,12,16,0.9)'; g.lineWidth = 0.07; g.fillStyle = '#e9e1cc'; g.strokeText(f.text, f.x, f.y); g.fillText(f.text, f.x, f.y); g.restore();
    }
    this.floaters = this.floaters.filter((f) => f.life > 0);

    // Aim System reticles & ballistic trajectories
    if (st.aim?.active) {
      const aim = st.aim;
      const ox = aim.origin.x, oy = aim.origin.y;
      const tx = aim.target.x, ty = aim.target.y;
      const dx = tx - ox, dy = ty - oy;
      const d = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);

      g.save();
      if (aim.aimSystem === 'laser') {
        // Red laser sight line
        const endX = ox + Math.cos(angle) * Math.min(d, aim.maxRange);
        const endY = oy + Math.sin(angle) * Math.min(d, aim.maxRange);
        g.beginPath();
        g.moveTo(ox, oy);
        g.lineTo(endX, endY);
        g.strokeStyle = 'rgba(239, 68, 68, 0.65)';
        g.lineWidth = 0.025;
        g.stroke();

        // Laser dot & precision sniper reticle at target
        g.beginPath();
        g.arc(tx, ty, 0.06, 0, TAU);
        g.fillStyle = '#ef4444';
        g.fill();

        // Precision crosshair circle
        g.beginPath();
        g.arc(tx, ty, 0.28, 0, TAU);
        g.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        g.lineWidth = 0.025;
        g.stroke();

        // Crosshair ticks
        const tk = 0.12;
        g.beginPath();
        g.moveTo(tx - 0.28 - tk, ty); g.lineTo(tx - 0.28, ty);
        g.moveTo(tx + 0.28, ty); g.lineTo(tx + 0.28 + tk, ty);
        g.moveTo(tx, ty - 0.28 - tk); g.lineTo(tx, ty - 0.28);
        g.moveTo(tx, ty + 0.28); g.lineTo(tx, ty + 0.28 + tk);
        g.stroke();
      } else if (aim.aimSystem === 'bloom') {
        // Spread cone guidelines
        const leftA = angle - aim.spread * 0.5;
        const rightA = angle + aim.spread * 0.5;
        const dist = Math.min(d, aim.maxRange);
        g.beginPath();
        g.moveTo(ox, oy); g.lineTo(ox + Math.cos(leftA) * dist, oy + Math.sin(leftA) * dist);
        g.moveTo(ox, oy); g.lineTo(ox + Math.cos(rightA) * dist, oy + Math.sin(rightA) * dist);
        g.strokeStyle = 'rgba(250, 204, 21, 0.4)';
        g.lineWidth = 0.02;
        g.setLineDash([0.15, 0.12]);
        g.stroke();
        g.setLineDash([]);

        // Bloom crosshair at target
        const gap = 0.08 + aim.spread * 0.8;
        const len = 0.14;
        g.beginPath();
        g.moveTo(tx - gap - len, ty); g.lineTo(tx - gap, ty);
        g.moveTo(tx + gap, ty); g.lineTo(tx + gap + len, ty);
        g.moveTo(tx, ty - gap - len); g.lineTo(tx, ty - gap);
        g.moveTo(tx, ty + gap); g.lineTo(tx, ty + gap + len);
        g.strokeStyle = '#facc15';
        g.lineWidth = 0.025;
        g.stroke();
      } else if (aim.aimSystem === 'arc') {
        // Parabolic trajectory arc
        const steps = 18;
        g.beginPath();
        for (let i = 0; i <= steps; i++) {
          const tArc = i / steps;
          const px = ox + dx * tArc;
          const py = oy + dy * tArc - Math.sin(tArc * Math.PI) * 1.6;
          if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
        }
        g.strokeStyle = 'rgba(249, 115, 22, 0.75)';
        g.lineWidth = 0.03;
        g.setLineDash([0.16, 0.12]);
        g.stroke();
        g.setLineDash([]);

        // Target impact landing circle on ground
        const pulse = (Math.sin(this.t * 8) + 1) * 0.5;
        g.beginPath();
        g.arc(tx, ty, 0.35 + pulse * 0.08, 0, TAU);
        g.strokeStyle = '#ea580c';
        g.lineWidth = 0.035;
        g.stroke();
        g.beginPath();
        g.arc(tx, ty, 0.1, 0, TAU);
        g.fillStyle = '#ea580c';
        g.fill();
      } else if (aim.aimSystem === 'burst') {
        // Diamond burst reticle
        const size = 0.22 + aim.spread * 0.4;
        g.beginPath();
        g.moveTo(tx, ty - size);
        g.lineTo(tx + size, ty);
        g.lineTo(tx, ty + size);
        g.lineTo(tx - size, ty);
        g.closePath();
        g.strokeStyle = '#38bdf8';
        g.lineWidth = 0.028;
        g.stroke();

        // Center dot
        g.beginPath();
        g.arc(tx, ty, 0.04, 0, TAU);
        g.fillStyle = '#38bdf8';
        g.fill();
      }

      // Extra Visual Info Overlays
      const vInfo = aim.visualInfo;

      // 1. Ballistic Trajectory Arc Overlay (if active and not already drawn by 'arc' aimSystem)
      if (vInfo?.trajectoryArc && aim.aimSystem !== 'arc') {
        const steps = 18;
        g.beginPath();
        for (let i = 0; i <= steps; i++) {
          const tArc = i / steps;
          const px = ox + dx * tArc;
          const py = oy + dy * tArc - Math.sin(tArc * Math.PI) * 1.1;
          if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
        }
        g.strokeStyle = 'rgba(56, 189, 248, 0.75)';
        g.lineWidth = 0.025;
        g.setLineDash([0.14, 0.1]);
        g.stroke();
        g.setLineDash([]);
      }

      // 2. Threat Scanner HUD (Pulsating warning brackets on nearby threats)
      if (vInfo?.threatScanner && aim.threats && aim.threats.length > 0) {
        for (const th of aim.threats) {
          const pulse = (Math.sin(this.t * 9 + th.x * 2) + 1) * 0.5;
          const bs = 0.42;
          g.strokeStyle = '#ef4444';
          g.lineWidth = 0.024;
          // 4 corner brackets
          g.beginPath(); g.moveTo(th.x - bs, th.y - bs * 0.5); g.lineTo(th.x - bs, th.y - bs); g.lineTo(th.x - bs * 0.5, th.y - bs); g.stroke();
          g.beginPath(); g.moveTo(th.x + bs * 0.5, th.y - bs); g.lineTo(th.x + bs, th.y - bs); g.lineTo(th.x + bs, th.y - bs * 0.5); g.stroke();
          g.beginPath(); g.moveTo(th.x - bs, th.y + bs * 0.5); g.lineTo(th.x - bs, th.y + bs); g.lineTo(th.x - bs * 0.5, th.y + bs); g.stroke();
          g.beginPath(); g.moveTo(th.x + bs * 0.5, th.y + bs); g.lineTo(th.x + bs, th.y + bs); g.lineTo(th.x + bs, th.y + bs * 0.5); g.stroke();

          g.fillStyle = pulse > 0.4 ? '#ef4444' : '#f97316';
          g.font = 'bold 0.16px "Outfit", sans-serif';
          g.textAlign = 'center';
          g.fillText(`[!] ${th.dist}m`, th.x, th.y - bs - 0.06);
        }
      }

      // 3. Live Target HP Bar & Identification
      if (vInfo?.targetHealthBar && aim.aimedTarget) {
        const at = aim.aimedTarget;
        const bw = 1.0, bh = 0.12;
        const bx = tx - bw / 2, by = ty - 0.58;
        // Background plaque
        g.fillStyle = 'rgba(15, 23, 42, 0.9)';
        g.fillRect(bx - 0.06, by - 0.22, bw + 0.12, bh + 0.28);
        g.strokeStyle = '#38bdf8';
        g.lineWidth = 0.018;
        g.strokeRect(bx - 0.06, by - 0.22, bw + 0.12, bh + 0.28);

        // Target Name
        g.font = 'bold 0.13px "Outfit", sans-serif';
        g.textAlign = 'center';
        g.fillStyle = '#38bdf8';
        g.fillText(at.name.toUpperCase(), tx, by - 0.06);

        // HP Gauge Bar
        const pct = clamp(at.hp / Math.max(1, at.maxHp), 0, 1);
        g.fillStyle = 'rgba(239, 68, 68, 0.35)';
        g.fillRect(bx, by, bw, bh);
        g.fillStyle = pct > 0.5 ? '#22c55e' : pct > 0.25 ? '#eab308' : '#ef4444';
        g.fillRect(bx, by, bw * pct, bh);
        g.strokeStyle = '#ffffff';
        g.lineWidth = 0.012;
        g.strokeRect(bx, by, bw, bh);

        // HP Numbers
        g.font = 'bold 0.10px monospace';
        g.fillStyle = '#ffffff';
        g.fillText(`${Math.round(at.hp)} / ${at.maxHp} HP`, tx, by + 0.095);
      }

      // 4. Tactical Rangefinder vs Standard Reticle Tag
      if (vInfo?.rangefinder) {
        // High-tech rangefinder readout
        g.fillStyle = 'rgba(15, 23, 42, 0.88)';
        g.fillRect(tx - 0.8, ty + 0.3, 1.6, 0.44);
        g.strokeStyle = '#10b981';
        g.lineWidth = 0.02;
        g.strokeRect(tx - 0.8, ty + 0.3, 1.6, 0.44);

        g.font = 'bold 0.15px monospace';
        g.textAlign = 'center';
        g.fillStyle = '#10b981';
        g.fillText(`🎯 DIST: ${(Math.round(d * 10) / 10).toFixed(1)}m`, tx, ty + 0.48);
        g.font = 'bold 0.12px monospace';
        g.fillStyle = '#a7f3d0';
        g.fillText(`ETA: ${(d / 45).toFixed(2)}s | 💥 ${aim.ammoCount} RDS`, tx, ty + 0.65);
      } else {
        // Standard range & weapon info tag
        g.font = 'bold 0.2px "Outfit", sans-serif';
        g.textAlign = 'center';
        g.fillStyle = '#ffffff';
        g.strokeStyle = 'rgba(15, 23, 42, 0.85)';
        g.lineWidth = 0.035;
        const meterStr = `${Math.round(d * 10) / 10}m | 💥 ${aim.ammoCount}`;
        g.strokeText(meterStr, tx, ty + 0.45);
        g.fillText(meterStr, tx, ty + 0.45);
      }

      g.restore();
    }

    // light, vignette, grain
    const lights = this.extraLights; lights.length = 0; for (const l of game.lights()) lights.push(l);
    if (fancy) buildingLights(nearBuildings, game.clock.darkness, game.clock.day, lights); // warm window and porch pools
    this.lighting.render(g, W, H, cx, cy, z, lights, game.clock.darkness, game.clock.t, this.t, this.atmos.quality);
    g.setTransform(1, 0, 0, 1, 0, 0);
    if (fancy) this.atmos.draw('screen'); // fireflies, lightning, vignette, paper grain
    else {
    const vg = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
    vg.addColorStop(0, 'rgba(4,6,10,0)'); vg.addColorStop(1, 'rgba(4,6,10,0.62)'); g.fillStyle = vg; g.fillRect(0, 0, W, H);
    g.globalAlpha = 0.07; g.fillStyle = g.createPattern(this.grain, 'repeat') ?? '#000'; g.translate(Math.floor(this.t * 60) % 97, Math.floor(this.t * 37) % 89); g.fillRect(-100, -100, W + 200, H + 200); g.globalAlpha = 1; g.setTransform(1, 0, 0, 1, 0, 0);
    }

    if (st.debug) {
      g.strokeStyle = 'rgba(255,255,255,0.15)'; g.lineWidth = 1; g.setTransform(z, 0, 0, z, W / 2 - cx * z, H / 2 - cy * z); g.lineWidth = 0.03;
      for (let y = c0y; y <= c1y + 1; y++) { g.beginPath(); g.moveTo(c0x * CHUNK, y * CHUNK); g.lineTo((c1x + 1) * CHUNK, y * CHUNK); g.stroke(); }
      for (let x = c0x; x <= c1x + 1; x++) { g.beginPath(); g.moveTo(x * CHUNK, c0y * CHUNK); g.lineTo(x * CHUNK, (c1y + 1) * CHUNK); g.stroke(); }
      g.setTransform(1, 0, 0, 1, 0, 0);
    }
  }

  private drawStructure(id: string, size: [number, number], tx: number, ty: number, open: boolean, ppu: number, rot = 0): void {
    const fn = STRUCTURE_DRAWERS[id]; if (!fn) return;
    const spr = this.cache.get(`s:${id}:${open}`, ppu, -0.1, -0.1, size[0] + 0.1, size[1] + 0.1, (gg) => fn(gg, makeRng(hashStr(id)), size[0], size[1], open));
    if (rot !== 0) {
      this.g.save();
      this.g.translate(tx + size[0] / 2, ty + size[1] / 2);
      this.g.rotate(rot);
      this.g.translate(-size[0] / 2, -size[1] / 2);
      blit(this.g, spr, 0, 0);
      this.g.restore();
    } else {
      blit(this.g, spr, tx, ty);
    }
  }

  /** Animated ink-style flame at (x,y) in the current unit space. */
  private flame(x: number, y: number, s: number, local = false): void {
    const g = this.g, t = this.t * 9 + x * 3 + y;
    g.save(); g.translate(x, y); g.scale(s, s); if (local) g.scale(1, 1);
    const f = (k: number): number => Math.sin(t * k) * 0.5 + 0.5;
    const layer = (hgt: number, wid: number, col: string, line: number): void => inked(g, () => smooth(g, [[0, -hgt - f(1.7) * 0.08], [wid * 0.9, -hgt * 0.4], [wid, hgt * 0.15], [0, hgt * 0.3], [-wid, hgt * 0.15], [-wid * 0.9, -hgt * 0.4 + f(2.3) * 0.05]]), col, { line, hatch: 0, box: 0.2, noWash: true });
    layer(0.34, 0.17, '#c8501f', 0.02); layer(0.25, 0.12, '#f08a32', 0.012); layer(0.15, 0.07, '#fdeb90', 0.008);
    g.restore();
    void poly; void alpha;
  }
}
