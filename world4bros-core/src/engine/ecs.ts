/** Minimal entity-component store. Components are typed tokens; systems are plain functions. */
export type Entity = number;
export class Component<T> {
  declare readonly _t: T;
  constructor(public readonly name: string) {}
}
export class Ecs {
  private nextId = 1;
  private alive = new Set<Entity>();
  private stores = new Map<string, Map<Entity, unknown>>();
  create(): Entity { const e = this.nextId++; this.alive.add(e); return e; }
  exists(e: Entity): boolean { return this.alive.has(e); }
  destroy(e: Entity): void { this.alive.delete(e); this.stores.forEach((s) => s.delete(e)); }
  add<T>(e: Entity, c: Component<T>, value: T): T {
    let s = this.stores.get(c.name);
    if (!s) { s = new Map(); this.stores.set(c.name, s); }
    s.set(e, value);
    return value;
  }
  get<T>(e: Entity, c: Component<T>): T | undefined { return this.stores.get(c.name)?.get(e) as T | undefined; }
  has(e: Entity, c: Component<unknown>): boolean { return this.stores.get(c.name)?.has(e) ?? false; }
  remove(e: Entity, c: Component<unknown>): void { this.stores.get(c.name)?.delete(e); }
  /** Entities that have every listed component. */
  query(...cs: Component<unknown>[]): Entity[] {
    if (cs.length === 0) return [...this.alive];
    const stores = cs.map((c) => this.stores.get(c.name));
    if (stores.some((s) => !s)) return [];
    const smallest = (stores as Map<Entity, unknown>[]).reduce((a, b) => (a.size <= b.size ? a : b));
    const out: Entity[] = [];
    for (const e of smallest.keys()) if (this.alive.has(e) && (stores as Map<Entity, unknown>[]).every((s) => s.has(e))) out.push(e);
    return out;
  }
  count(): number { return this.alive.size; }
}
