import type { InventorySystem, InventorySlot } from './InventorySystem';
import type { SurvivalStats } from './SurvivalStats';

const SAVE_KEY = 'world-4-bros-save-v1';

export interface SaveData {
  version: 1;
  savedAt: number;
  player: { x: number; z: number };
  inventory: { slots: InventorySlot[]; selectedIndex: number };
  stats: {
    health: number;
    hunger: number;
    sanity: number;
    treesFelled: number;
    itemsCrafted: number;
    daysSurvived: number;
  };
}

/**
 * Browser-local save system. Progress lives in localStorage on each device,
 * so PC and phone keep separate saves. No login required.
 */
export class SaveSystem {
  static load(): SaveData | null {
    try {
      const raw = localStorage.getItem(SAVE_KEY) || localStorage.getItem('wild-survivors-save-v1');
      if (!raw) return null;
      const data = JSON.parse(raw) as SaveData;
      return data?.version === 1 ? data : null;
    } catch {
      return null;
    }
  }

  static save(player: { x: number; z: number }, inventory: InventorySystem, stats: SurvivalStats) {
    if (stats.isDead) return; // never persist a dead character
    const data: SaveData = {
      version: 1,
      savedAt: Date.now(),
      player: { x: player.x, z: player.z },
      inventory: {
        slots: inventory.slots.map(s => ({ itemId: s.itemId, count: s.count })),
        selectedIndex: inventory.selectedIndex,
      },
      stats: {
        health: stats.health,
        hunger: stats.hunger,
        sanity: stats.sanity,
        treesFelled: stats.treesFelled,
        itemsCrafted: stats.itemsCrafted,
        daysSurvived: stats.daysSurvived,
      },
    };
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(data));
    } catch {
      // Storage full or disabled (private mode) — play on without saving
    }
  }

  static apply(data: SaveData, inventory: InventorySystem, stats: SurvivalStats) {
    data.inventory.slots.forEach((s, i) => {
      if (inventory.slots[i]) {
        inventory.slots[i].itemId = s.itemId;
        inventory.slots[i].count = s.count;
      }
    });
    inventory.selectedIndex = data.inventory.selectedIndex ?? 0;
    Object.assign(stats, data.stats);
    inventory.renderHUD();
    stats.renderHUD();
  }

  static clear() {
    try {
      localStorage.removeItem(SAVE_KEY);
    } catch {
      /* ignore */
    }
  }
}
