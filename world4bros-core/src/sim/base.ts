import type { Ecs, Entity } from '../engine/ecs';
import type { Content, StructureDef } from '../data/schema';
import { Comp } from './components';
import type { GameWorld } from './world';
import { Inventory } from './inventory';

export interface PlaceCheck { ok: boolean; reason?: string }
interface Cell { solid?: Entity; floor?: Entity }
const k = (tx: number, ty: number): string => `${tx},${ty}`;

/** Base building core: claims, footprints, placement rules and structure collision. */
export class BaseSystem {
  private cells = new Map<string, Cell>();
  constructor(private ecs: Ecs, private world: GameWorld, private content: Content) {}

  entities(): Entity[] { return this.ecs.query(Comp.Structure, Comp.Transform); }
  claims(): { x: number; y: number; r: number; entity: Entity }[] {
    const out: { x: number; y: number; r: number; entity: Entity }[] = [];
    for (const e of this.entities()) {
      const s = this.ecs.get(e, Comp.Structure);
      if (s?.def.claimRadius) out.push({ x: s.tx + 0.5, y: s.ty + 0.5, r: s.def.claimRadius, entity: e });
    }
    return out;
  }
  inClaim(x: number, y: number): boolean { return this.claims().some((c) => Math.hypot(c.x - x, c.y - y) <= c.r); }

  private footprint(def: StructureDef, tx: number, ty: number): [number, number][] {
    const out: [number, number][] = [];
    for (let dy = 0; dy < def.size[1]; dy++) for (let dx = 0; dx < def.size[0]; dx++) out.push([tx + dx, ty + dy]);
    return out;
  }
  private isFloor(def: StructureDef): boolean { return (!def.solid && !def.light && !def.interact && def.id.startsWith('floor')) || def.id === 'conveyor_belt'; }

  checkPlace(def: StructureDef, tx: number, ty: number, avoid?: { x: number; y: number; r: number }): PlaceCheck {
    const tiles = this.footprint(def, tx, ty);
    const floor = this.isFloor(def);
    for (const [x, y] of tiles) {
      if (!this.world.walkable(x + 0.5, y + 0.5)) return { ok: false, reason: 'Too wet to build here' };
      const cell = this.cells.get(k(x, y));
      if (cell) { if (floor ? cell.floor : cell.solid) return { ok: false, reason: 'Something is already there' }; }
      for (const o of this.world.objectsNear(x + 0.5, y + 0.5, 1.2)) {
        const od = this.content.objects.get(o.kind);
        if (od?.solid && Math.abs(o.x - (x + 0.5)) < 0.5 + od.radius * 0.6 && Math.abs(o.y - (y + 0.5)) < 0.5 + od.radius * 0.6) return { ok: false, reason: `A ${od.name.toLowerCase()} is in the way` };
      }
      if (avoid && def.solid) {
        const cx = Math.max(x, Math.min(avoid.x, x + 1)), cy = Math.max(y, Math.min(avoid.y, y + 1));
        if (Math.hypot(avoid.x - cx, avoid.y - cy) < avoid.r) return { ok: false, reason: 'You are standing there' };
      }
    }
    if (def.claimRadius) {
      for (const c of this.claims()) if (Math.hypot(c.x - (tx + 0.5), c.y - (ty + 0.5)) < c.r * 0.9) return { ok: false, reason: 'Too close to another claim' };
    } else if (def.requiresClaim && !tiles.every(([x, y]) => this.inClaim(x + 0.5, y + 0.5))) return { ok: false, reason: 'Place a Claim Flag first' };
    return { ok: true };
  }

  place(def: StructureDef, tx: number, ty: number, dir: 'N' | 'E' | 'S' | 'W' = 'E'): Entity {
    const e = this.ecs.create();
    const ang = dir === 'E' ? 0 : dir === 'S' ? Math.PI / 2 : dir === 'W' ? Math.PI : -Math.PI / 2;
    this.ecs.add(e, Comp.Transform, { x: tx + def.size[0] / 2, y: ty + def.size[1] / 2, angle: ang });
    this.ecs.add(e, Comp.Structure, { def, tx, ty, hp: def.hp, open: false, storage: def.interact === 'storage' ? new Inventory(this.content, def.slots ?? 12) : undefined });
    if (def.light) this.ecs.add(e, Comp.Light, { ...def.light });
    if (def.id === 'conveyor_belt') {
      this.ecs.add(e, Comp.Conveyor, { dir, speed: 2.0, items: [] });
    } else if (def.id === 'spring_inserter') {
      const pickupDir = dir === 'E' ? 'W' : dir === 'W' ? 'E' : dir === 'N' ? 'S' : 'N';
      this.ecs.add(e, Comp.Inserter, { pickupDir, dropDir: dir, heldItem: null, armProgress: 0, speed: 1.8 });
    } else if (def.id === 'burner_drill') {
      this.ecs.add(e, Comp.Machine, { kind: 'burner_drill', fuel: 20, progress: 0 });
    } else if (def.id === 'stone_furnace') {
      this.ecs.add(e, Comp.Machine, { kind: 'stone_furnace', fuel: 20, progress: 0 });
    }
    const floor = this.isFloor(def);
    for (const [x, y] of this.footprint(def, tx, ty)) {
      const c = this.cells.get(k(x, y)) ?? {};
      if (floor) c.floor = e; else c.solid = e;
      this.cells.set(k(x, y), c);
    }
    return e;
  }

  remove(e: Entity): StructureDef | null {
    const s = this.ecs.get(e, Comp.Structure);
    if (!s) return null;
    for (const [x, y] of this.footprint(s.def, s.tx, s.ty)) {
      const c = this.cells.get(k(x, y));
      if (c) { if (c.solid === e) delete c.solid; if (c.floor === e) delete c.floor; if (!c.solid && !c.floor) this.cells.delete(k(x, y)); }
    }
    this.ecs.destroy(e);
    return s.def;
  }

  structureAt(tx: number, ty: number): Entity | undefined { const c = this.cells.get(k(tx, ty)); return c?.solid ?? c?.floor; }

  /** Circle vs solid structure tiles. Used by the world's collision. */
  solidAt(x: number, y: number, r: number): boolean {
    for (let ty = Math.floor(y - r); ty <= Math.floor(y + r); ty++) for (let tx = Math.floor(x - r); tx <= Math.floor(x + r); tx++) {
      const c = this.cells.get(k(tx, ty));
      if (!c?.solid) continue;
      const s = this.ecs.get(c.solid, Comp.Structure);
      if (!s || !s.def.solid || s.open) continue;
      const cx = Math.max(tx, Math.min(x, tx + 1)), cy = Math.max(ty, Math.min(y, ty + 1));
      if (Math.hypot(x - cx, y - cy) < r) return true;
    }
    return false;
  }
}
