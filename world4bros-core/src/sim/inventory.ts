import type { Content, Id } from '../data/schema';
export interface Stack { item: Id; qty: number }

export class Inventory {
  slots: (Stack | null)[];
  constructor(private content: Content, size: number) { this.slots = Array.from({ length: size }, () => null); }

  /** Adds items, filling stacks first. Returns how many did NOT fit. */
  add(item: Id, qty: number): number {
    const def = this.content.items.get(item);
    if (!def) return qty;
    let left = qty;
    for (const s of this.slots) if (left > 0 && s && s.item === item && s.qty < def.stack) { const t = Math.min(left, def.stack - s.qty); s.qty += t; left -= t; }
    for (let i = 0; i < this.slots.length && left > 0; i++) if (!this.slots[i]) { const t = Math.min(left, def.stack); this.slots[i] = { item, qty: t }; left -= t; }
    return left;
  }
  count(item: Id): number { return this.slots.reduce((n, s) => n + (s && s.item === item ? s.qty : 0), 0); }
  has(items: Record<Id, number>): boolean { return Object.entries(items).every(([k, v]) => this.count(k) >= v); }
  remove(item: Id, qty: number): boolean {
    if (this.count(item) < qty) return false;
    let left = qty;
    for (let i = 0; i < this.slots.length && left > 0; i++) {
      const s = this.slots[i];
      if (s && s.item === item) { const t = Math.min(left, s.qty); s.qty -= t; left -= t; if (s.qty === 0) this.slots[i] = null; }
    }
    return true;
  }
  take(items: Record<Id, number>): boolean {
    if (!this.has(items)) return false;
    for (const [k, v] of Object.entries(items)) this.remove(k, v);
    return true;
  }
  removeAt(i: number): Stack | null { const s = this.slots[i] ?? null; this.slots[i] = null; return s; }
  weight(): number { return this.slots.reduce((w, s) => w + (s ? (this.content.items.get(s.item)?.weight ?? 0) * s.qty : 0), 0); }
  toJSON(): (Stack | null)[] { return this.slots.map((s) => (s ? { ...s } : null)); }
  load(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.slots = Array.from({ length: this.slots.length }, (_, i) => {
      const s = data[i] as Stack | null | undefined;
      return s && typeof s.item === 'string' && this.content.items.has(s.item) && s.qty > 0 ? { item: s.item, qty: Math.floor(s.qty) } : null;
    });
  }
}
