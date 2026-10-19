import { WorldGen, CHUNK, getRoadType, circleHitsWalls, type Building, type Side, type ChunkData, type WorldObject, type GenStage, DEFAULT_STAGES } from './worldgen';
import { StaticMap, type StaticMapSave } from './staticmap';
import type { Content, ObjectKind, Terrain, TerrainDef } from '../data/schema';

export interface ObjectMod { hp?: number; kind?: ObjectKind; removed?: boolean }
export type Blocker = (x: number, y: number, r: number) => boolean;
const key = (cx: number, cy: number): string => `${cx},${cy}`;

/** The map: chunk streaming, object edits, terrain queries and collision. Pure data, no rendering. */
export class GameWorld {
  readonly gen: WorldGen;
  readonly mods = new Map<string, ObjectMod>();
  private chunks = new Map<string, ChunkData>();
  blocker: Blocker | null = null;
  /** Frozen, hand-editable copy of the world. Kept when toggled off so edits are never lost. */
  staticMap: StaticMap | null = null;
  staticActive = false;
  /** Chunk keys changed by edits; the renderer drains this to repaint. */
  readonly dirty = new Set<string>();

  constructor(readonly content: Content, readonly seed: number, stages: GenStage[] = DEFAULT_STAGES) {
    this.gen = new WorldGen(seed, content, stages);
  }
  get spawn(): { x: number; y: number } { return this.gen.spawn; }

  // ---------------------------------------------------------------- static map mode
  /** Switch between the infinite generator and the frozen map. The first switch-on bakes the generator output. */
  setStatic(on: boolean): void {
    if (on && !this.staticMap) this.staticMap = StaticMap.bake(this.gen);
    this.staticActive = on && !!this.staticMap;
    this.chunks.clear();
    this.markAllDirty();
  }
  /** Throw away edits and re-freeze the current generator output. */
  rebakeStatic(): void {
    this.staticMap = StaticMap.bake(this.gen);
    this.staticActive = true;
    this.chunks.clear();
    this.markAllDirty();
  }
  loadStatic(data: StaticMapSave, active: boolean): void {
    this.staticMap = StaticMap.fromJSON(data, this.gen.terrainOrder, (x, y) => this.gen.sample(x, y).height);
    this.staticActive = active;
    this.chunks.clear();
  }
  saveStatic(): StaticMapSave | undefined { return this.staticMap?.toJSON(this.gen.terrainOrder); }
  private markAllDirty(): void {
    if (this.staticMap) for (const k of this.staticMap.chunks.keys()) this.dirty.add(k);
    this.dirty.add('*');
  }
  private markTile(tx: number, ty: number): void {
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) this.dirty.add(key(Math.floor((tx + dx) / CHUNK), Math.floor((ty + dy) / CHUNK)));
  }
  private editable(x: number, y: number): ChunkData | null {
    if (!this.staticActive || !this.staticMap) return null;
    const cx = Math.floor(x / CHUNK), cy = Math.floor(y / CHUNK);
    return this.staticMap.inBounds(cx, cy) ? this.chunkAt(cx, cy) : null;
  }
  /** Paint a round brush of terrain (and optionally biome index) into the static map. Returns tiles changed. */
  paint(x: number, y: number, radius: number, terrain: Terrain | null, biome: number | null): number {
    if (!this.staticActive) return 0;
    const tIdx = terrain ? this.gen.terrainOrder.indexOf(terrain) : -1;
    let n = 0;
    const r = Math.max(0.5, radius);
    for (let ty = Math.floor(y - r); ty <= Math.floor(y + r); ty++) for (let tx = Math.floor(x - r); tx <= Math.floor(x + r); tx++) {
      if (Math.hypot(tx + 0.5 - x, ty + 0.5 - y) > r) continue;
      const c = this.editable(tx, ty);
      if (!c) continue;
      const i = (ty - c.cy * CHUNK) * CHUNK + (tx - c.cx * CHUNK);
      let hit = false;
      if (tIdx >= 0 && c.terrain[i] !== tIdx) { c.terrain[i] = tIdx; hit = true; }
      if (biome !== null && c.biome[i] !== biome) { c.biome[i] = biome; hit = true; }
      if (hit) { n++; this.markTile(tx, ty); }
    }
    return n;
  }
  /** Place one object into the static map. Returns the new object or null if out of bounds / not static. */
  placeObject(kind: ObjectKind, x: number, y: number): WorldObject | null {
    const c = this.editable(x, y);
    if (!c || !this.staticMap) return null;
    const def = this.content.objects.get(kind);
    const o: WorldObject = { id: `${c.cx},${c.cy}:e${this.staticMap.nextId++}`, kind, x, y, variant: Math.abs(Math.floor(x * 7 + y * 13)) % 4, hp: def?.hp ?? 1 };
    c.objects.push(o);
    this.markTile(Math.floor(x), Math.floor(y));
    return o;
  }
  /** Remove every object within radius of (x,y) from the static map. Returns how many were removed. */
  eraseObjects(x: number, y: number, radius: number): number {
    if (!this.staticActive) return 0;
    let n = 0;
    for (const o of this.objectsNear(x, y, radius)) {
      if (Math.hypot(o.x - x, o.y - y) > radius) continue;
      o.removed = true; n++;
    }
    if (n) this.markTile(Math.floor(x), Math.floor(y));
    return n;
  }
  /** Place a building into the static map (owned by the chunk holding its centre). */
  placeBuilding(def: string, x: number, y: number, w: number, h: number, door: Side = 'S'): Building | null {
    const cx0 = x + w / 2, cy0 = y + h / 2;
    const c = this.editable(cx0, cy0);
    if (!c || !this.staticMap) return null;
    const b: Building = { id: `${c.cx},${c.cy}:bld_e${this.staticMap.nextId++}`, def, x: Math.round(x), y: Math.round(y), w, h, door, doorAt: door === 'N' || door === 'S' ? w / 2 : h / 2 };
    c.buildings.push(b);
    for (let ty = b.y - 1; ty <= b.y + h + 1; ty += CHUNK) this.markTile(b.x, ty);
    return b;
  }
  /** Remove buildings containing (x,y) from the static map. Returns how many were removed. */
  eraseBuildings(x: number, y: number): number {
    if (!this.staticActive || !this.staticMap) return 0;
    let n = 0;
    for (const c of this.staticMap.chunks.values()) {
      const keep = c.buildings.filter((b) => !(x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h));
      n += c.buildings.length - keep.length;
      c.buildings = keep;
    }
    return n;
  }
  /** Buildings whose footprint overlaps the square around (x,y) with half-size r. */
  buildingsNear(x: number, y: number, r: number): Building[] {
    const out: Building[] = [];
    const reach = r + 14; // buildings are owned by the chunk of their centre and are at most ~12 tiles wide
    const c0x = Math.floor((x - reach) / CHUNK), c1x = Math.floor((x + reach) / CHUNK), c0y = Math.floor((y - reach) / CHUNK), c1y = Math.floor((y + reach) / CHUNK);
    for (let cy = c0y; cy <= c1y; cy++) for (let cx = c0x; cx <= c1x; cx++) {
      for (const b of this.chunkAt(cx, cy).buildings) if (b.x <= x + r && b.x + b.w >= x - r && b.y <= y + r && b.y + b.h >= y - r) out.push(b);
    }
    return out;
  }

  chunkAt(cx: number, cy: number): ChunkData {
    if (this.staticActive && this.staticMap) return this.staticMap.chunk(cx, cy, this.gen.terrainOrder.indexOf('deep'));
    const k = key(cx, cy);
    let c = this.chunks.get(k);
    if (!c) {
      c = this.gen.chunk(cx, cy);
      for (const o of c.objects) { const m = this.mods.get(o.id); if (m) this.applyMod(o, m); }
      this.chunks.set(k, c);
    }
    return c;
  }
  private applyMod(o: WorldObject, m: ObjectMod): void {
    if (m.kind) o.kind = m.kind;
    if (m.hp !== undefined) o.hp = m.hp;
    if (m.removed) o.removed = true;
  }
  /** Drop far chunks from memory. Their edits live in `mods` and are re-applied on regeneration. */
  evictFar(x: number, y: number, keepRadius = 8): void {
    const pcx = Math.floor(x / CHUNK), pcy = Math.floor(y / CHUNK);
    for (const [k, c] of this.chunks) if (Math.abs(c.cx - pcx) > keepRadius || Math.abs(c.cy - pcy) > keepRadius) this.chunks.delete(k);
  }
  loadedChunks(): ChunkData[] { return [...this.chunks.values()]; }

  terrainDefAt(x: number, y: number): TerrainDef {
    const cx = Math.floor(x / CHUNK), cy = Math.floor(y / CHUNK);
    const c = this.chunkAt(cx, cy);
    const tx = Math.floor(x) - cx * CHUNK, ty = Math.floor(y) - cy * CHUNK;
    const id = this.gen.terrainOrder[c.terrain[ty * CHUNK + tx] ?? 0];
    return this.content.terrain.get(id as never) as TerrainDef;
  }
  terrainIdAt(x: number, y: number): Terrain {
    return this.terrainDefAt(x, y).id;
  }
  elevationAt(x: number, y: number): number {
    const cx = Math.floor(x / CHUNK), cy = Math.floor(y / CHUNK);
    const c = this.chunkAt(cx, cy);
    const tx = Math.floor(x) - cx * CHUNK, ty = Math.floor(y) - cy * CHUNK;
    return (c.elevation[ty * CHUNK + tx] ?? 0) / 255;
  }
  biomeAt(x: number, y: number): string {
    const cx = Math.floor(x / CHUNK), cy = Math.floor(y / CHUNK);
    const c = this.chunkAt(cx, cy);
    const tx = Math.floor(x) - cx * CHUNK, ty = Math.floor(y) - cy * CHUNK;
    return this.content.biomes[c.biome[ty * CHUNK + tx] ?? 0]?.id ?? 'unknown';
  }
  walkable(x: number, y: number): boolean { return this.terrainDefAt(x, y).walkable; }
  speedAt(x: number, y: number): number {
    const base = this.terrainDefAt(x, y).speed;
    if (base <= 0) return 0;
    const rType = getRoadType(this.seed, x, y, this.spawn);
    if (rType === 'highway') return base * 1.35;
    if (rType === 'dirt_road') return base * 1.18;
    if (rType === 'small_road') return base * 1.10;
    return base;
  }

  objectsNear(x: number, y: number, r: number): WorldObject[] {
    const out: WorldObject[] = [];
    const c0x = Math.floor((x - r) / CHUNK), c1x = Math.floor((x + r) / CHUNK), c0y = Math.floor((y - r) / CHUNK), c1y = Math.floor((y + r) / CHUNK);
    for (let cy = c0y; cy <= c1y; cy++) for (let cx = c0x; cx <= c1x; cx++)
      for (const o of this.chunkAt(cx, cy).objects) if (!o.removed && Math.abs(o.x - x) <= r && Math.abs(o.y - y) <= r) out.push(o);
    return out;
  }
  getObject(id: string): WorldObject | undefined {
    const [cs] = id.split(':');
    const [cx, cy] = (cs ?? '').split(',').map(Number);
    if (cx === undefined || cy === undefined || Number.isNaN(cx) || Number.isNaN(cy)) return undefined;
    return this.chunkAt(cx, cy).objects.find((o) => o.id === id && !o.removed);
  }
  damageObject(id: string, dmg: number): { destroyed: boolean; obj?: WorldObject; becameKind?: ObjectKind } {
    const o = this.getObject(id);
    if (!o) return { destroyed: false };
    o.hp -= dmg;
    this.mods.set(id, { ...this.mods.get(id), hp: o.hp });
    if (o.hp > 0) return { destroyed: false, obj: o };
    const def = this.content.objects.get(o.kind);
    if (def?.becomes) {
      const becomeDef = this.content.objects.get(def.becomes);
      const kind = def.becomes;
      o.kind = kind; o.hp = becomeDef?.hp ?? 1;
      this.mods.set(id, { kind, hp: o.hp });
      return { destroyed: true, obj: o, becameKind: kind };
    }
    o.removed = true;
    this.mods.set(id, { removed: true });
    return { destroyed: true, obj: o };
  }

  /** True if a circle at (x,y) with radius r hits water, a solid object, or a structure. */
  blocked(x: number, y: number, r: number): boolean {
    if (!this.walkable(x - r, y) || !this.walkable(x + r, y) || !this.walkable(x, y - r) || !this.walkable(x, y + r)) return true;
    for (const o of this.objectsNear(x, y, r + 1)) {
      const d = this.content.objects.get(o.kind);
      if (d?.solid && Math.hypot(o.x - x, o.y - y) < d.radius + r) return true;
    }
    for (const b of this.buildingsNear(x, y, r + 0.5)) if (circleHitsWalls(b, x, y, r)) return true;
    return this.blocker ? this.blocker(x, y, r) : false;
  }
}
