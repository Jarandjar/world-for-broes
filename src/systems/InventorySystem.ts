import { ItemDef, ITEM_REGISTRY } from '../registries/ItemDef';
export type { ItemDef };
export { ITEM_REGISTRY };

export interface InventorySlot {
  itemId: string | null;
  count: number;
}

export class InventorySystem {
  public slots: InventorySlot[] = [];
  public selectedIndex: number = 0;
  private maxSlots: number = 8;
  private onChangeListeners: (() => void)[] = [];

  constructor() {
    for (let i = 0; i < this.maxSlots; i++) {
      this.slots.push({ itemId: null, count: 0 });
    }
  }

  public onChange(fn: () => void) {
    this.onChangeListeners.push(fn);
  }

  private notify() {
    this.onChangeListeners.forEach(fn => fn());
    this.renderHUD();
  }

  public addItem(itemId: string, count: number = 1): boolean {
    const def = ITEM_REGISTRY[itemId];
    if (!def) return false;

    // 1. Try stacking into existing slot
    for (const slot of this.slots) {
      if (slot.itemId === itemId && slot.count < def.maxStack) {
        const canAdd = Math.min(count, def.maxStack - slot.count);
        slot.count += canAdd;
        count -= canAdd;
        if (count <= 0) {
          this.notify();
          return true;
        }
      }
    }

    // 2. Put into empty slot
    for (const slot of this.slots) {
      if (slot.itemId === null) {
        slot.itemId = itemId;
        slot.count = count;
        this.notify();
        return true;
      }
    }

    this.notify();
    return false; // Inventory full
  }

  public removeItem(itemId: string, count: number = 1): boolean {
    if (this.getItemCount(itemId) < count) return false;

    let remaining = count;
    for (const slot of this.slots) {
      if (slot.itemId === itemId) {
        if (slot.count > remaining) {
          slot.count -= remaining;
          remaining = 0;
          break;
        } else {
          remaining -= slot.count;
          slot.itemId = null;
          slot.count = 0;
        }
      }
    }
    this.notify();
    return true;
  }

  public getItemCount(itemId: string): number {
    return this.slots.reduce((total, s) => (s.itemId === itemId ? total + s.count : total), 0);
  }

  public getTotalWeight(): number {
    return this.slots.reduce((total, s) => {
      if (!s.itemId) return total;
      const def = ITEM_REGISTRY[s.itemId];
      const w = def?.weight ?? 0.5;
      return total + w * s.count;
    }, 0);
  }

  public getSelectedItem(): InventorySlot | null {
    return this.slots[this.selectedIndex] || null;
  }

  public getSelectedItemId(): string | null {
    const slot = this.getSelectedItem();
    return slot ? slot.itemId : null;
  }

  public selectSlot(index: number) {
    if (index >= 0 && index < this.maxSlots) {
      this.selectedIndex = index;
      this.notify();
    }
  }

  public consumeSelected(): ItemDef | null {
    const slot = this.getSelectedItem();
    if (!slot || !slot.itemId) return null;
    const def = ITEM_REGISTRY[slot.itemId];
    if (def.category === 'food') {
      slot.count -= 1;
      if (slot.count <= 0) {
        slot.itemId = null;
      }
      this.notify();
      return def;
    }
    return null;
  }

  public renderHUD() {
    if (typeof document === 'undefined') return;
    const container = document.getElementById('inventory-bar');
    if (!container) return;
    container.innerHTML = '';

    this.slots.forEach((slot, idx) => {
      const slotEl = document.createElement('div');
      slotEl.className = `inv-slot ${idx === this.selectedIndex ? 'active' : ''}`;
      const itemDef = slot.itemId ? ITEM_REGISTRY[slot.itemId] : null;
      slotEl.innerHTML = `
        <span class="slot-key">${idx + 1}</span>
        ${itemDef ? `<span class="slot-icon" title="${itemDef.name}">${itemDef.icon}</span>` : ''}
        ${slot.count > 1 ? `<span class="slot-count">${slot.count}</span>` : ''}
      `;
      slotEl.onclick = () => this.selectSlot(idx);
      container.appendChild(slotEl);
    });
  }
}
