import { ITEM_REGISTRY, ItemDef } from './ItemDef';

export interface LootEntry {
  itemId: string;
  weight: number;
  countMin?: number;
  countMax?: number;
}

export interface LootTableDef {
  id: string;
  entries: LootEntry[];
}

export const LOOT_TABLES: Record<string, LootTableDef> = {
  house_kitchen: {
    id: 'house_kitchen',
    entries: [
      { itemId: 'berries', weight: 25, countMin: 2, countMax: 5 },
      { itemId: 'cooked_berries', weight: 15, countMin: 1, countMax: 3 },
      { itemId: 'wild_carrot', weight: 15, countMin: 1, countMax: 3 },
      { itemId: 'water_bottle', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'empty_bottle', weight: 10, countMin: 1, countMax: 3 },
      { itemId: 'frying_pan', weight: 5, countMin: 1, countMax: 1 },
      { itemId: 'broken_toaster', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'mismatched_sock', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'bandage', weight: 5, countMin: 1, countMax: 1 },
      { itemId: 'pancake_stack', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  house_bedroom: {
    id: 'house_bedroom',
    entries: [
      { itemId: 'pillow_vest', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'school_backpack', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'sleeping_bag', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'mismatched_sock', weight: 25, countMin: 1, countMax: 4 },
      { itemId: 'alarm_clock_decoy', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'bandage', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'painkillers', weight: 10, countMin: 1, countMax: 2 },
      { itemId: 'stick', weight: 5, countMin: 1, countMax: 1 },
      { itemId: 'rubber_duck', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'fanny_pack_of_holding', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  house_garage: {
    id: 'house_garage',
    entries: [
      { itemId: 'spark_plugs', weight: 20, countMin: 1, countMax: 2 },
      { itemId: 'fan_belt', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'wrench', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'ratchet', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'jerrycan', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'duct_tape', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'nail_bat', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'scrap_metal', weight: 20, countMin: 2, countMax: 6 },
      { itemId: 'battery', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  shop_general: {
    id: 'shop_general',
    entries: [
      { itemId: 'water_bottle', weight: 20, countMin: 1, countMax: 3 },
      { itemId: 'energy_drink', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'flashlight', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'batteries_aa', weight: 20, countMin: 2, countMax: 4 },
      { itemId: 'bandage', weight: 15, countMin: 1, countMax: 3 },
      { itemId: 'shiny_buttons', weight: 25, countMin: 5, countMax: 15 },
      { itemId: 'whoopee_cushion', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'rubber_chicken', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  pharmacy: {
    id: 'pharmacy',
    entries: [
      { itemId: 'bandage', weight: 30, countMin: 2, countMax: 4 },
      { itemId: 'painkillers', weight: 25, countMin: 1, countMax: 3 },
      { itemId: 'first_aid_kit', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'antidote_gel', weight: 15, countMin: 1, countMax: 2 }, // Infection cure
      { itemId: 'water_bottle', weight: 10, countMin: 1, countMax: 2 },
      { itemId: 'gas_mask', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  diner: {
    id: 'diner',
    entries: [
      { itemId: 'roast_sausage_stick', weight: 25, countMin: 1, countMax: 3 },
      { itemId: 'pancake_stack', weight: 20, countMin: 1, countMax: 2 },
      { itemId: 'water_bottle', weight: 20, countMin: 1, countMax: 3 },
      { itemId: 'thermos_of_cocoa', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'crispy_sock_chips', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'frying_pan', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  school: {
    id: 'school',
    entries: [
      { itemId: 'school_backpack', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'hockey_stick', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'foam_dart_blaster', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'foam_darts', weight: 20, countMin: 10, countMax: 25 },
      { itemId: 'bowling_ball', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'marbles', weight: 15, countMin: 1, countMax: 3 },
      { itemId: 'water_balloon', weight: 20, countMin: 3, countMax: 8 },
      { itemId: 'bandage', weight: 10, countMin: 1, countMax: 2 },
      { itemId: 'house_key', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  office: {
    id: 'office',
    entries: [
      { itemId: 'batteries_aa', weight: 25, countMin: 2, countMax: 6 },
      { itemId: 'lockpick', weight: 15, countMin: 1, countMax: 3 },
      { itemId: 'map_fragment', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'shiny_buttons', weight: 25, countMin: 4, countMax: 12 },
      { itemId: 'keycard_blue', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'multi_tool', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  police_locker: {
    id: 'police_locker',
    entries: [
      { itemId: 'nail_bat', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'crowbar', weight: 20, countMin: 1, countMax: 1 },
      { itemId: 'road_flare', weight: 20, countMin: 2, countMax: 4 },
      { itemId: 'bike_helmet', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'keycard_red', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'flare_gun', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'flare_cartridges', weight: 15, countMin: 3, countMax: 6 },
      { itemId: 'first_aid_kit', weight: 10, countMin: 1, countMax: 1 }
    ]
  },
  farm_barn: {
    id: 'farm_barn',
    entries: [
      { itemId: 'axe', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'pickaxe', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'shovel', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'jerrycan', weight: 20, countMin: 1, countMax: 1 },
      { itemId: 'hose', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'wild_carrot', weight: 20, countMin: 2, countMax: 6 },
      { itemId: 'potatoes', weight: 20, countMin: 3, countMax: 8 },
      { itemId: 'rope', weight: 15, countMin: 1, countMax: 3 }
    ]
  },
  workshop: {
    id: 'workshop',
    entries: [
      { itemId: 'spark_plugs', weight: 20, countMin: 1, countMax: 3 },
      { itemId: 'fan_belt', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'battery', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'wrench', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'ratchet', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'jack', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'welding_kit', weight: 8, countMin: 1, countMax: 1 },
      { itemId: 'tire_standard', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'engine_block', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  outpost_crate: {
    id: 'outpost_crate',
    entries: [
      { itemId: 'flare_gun', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'firework_launcher', weight: 5, countMin: 1, countMax: 1 },
      { itemId: 'bottle_rockets', weight: 15, countMin: 5, countMax: 12 },
      { itemId: 'first_aid_kit', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'antidote_gel', weight: 15, countMin: 1, countMax: 2 },
      { itemId: 'hockey_pads', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'hiker_backpack', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'keycard_green', weight: 8, countMin: 1, countMax: 1 },
      { itemId: 'evac_beacon', weight: 5, countMin: 1, countMax: 1 }
    ]
  },
  vault_jackpot: {
    id: 'vault_jackpot',
    entries: [
      { itemId: 'boat_key', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'rescue_radio_part_1', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'rescue_radio_part_2', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'rescue_radio_part_3', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'giant_novelty_check', weight: 15, countMin: 1, countMax: 1 },
      { itemId: 'golden_toilet_seat', weight: 10, countMin: 1, countMax: 1 },
      { itemId: 'first_aid_kit', weight: 15, countMin: 2, countMax: 3 },
      { itemId: 'shiny_buttons', weight: 30, countMin: 50, countMax: 100 }
    ]
  }
};

/**
 * Rolls loot items from a designated table according to density and container slot constraints.
 */
export function rollLootFromTable(
  tableId: string,
  minCount: number = 1,
  maxCount: number = 3,
  densityMultiplier: number = 1.0
): { itemId: string; count: number }[] {
  const table = LOOT_TABLES[tableId] || LOOT_TABLES['shop_general'];
  const totalWeight = table.entries.reduce((sum, e) => sum + e.weight, 0);

  const adjustedMin = Math.round(minCount * densityMultiplier);
  const adjustedMax = Math.max(adjustedMin, Math.round(maxCount * densityMultiplier));
  const numRolls = adjustedMin + Math.floor(Math.random() * (adjustedMax - adjustedMin + 1));

  const results: { itemId: string; count: number }[] = [];

  for (let i = 0; i < numRolls; i++) {
    let rand = Math.random() * totalWeight;
    for (const entry of table.entries) {
      rand -= entry.weight;
      if (rand <= 0) {
        const cMin = entry.countMin || 1;
        const cMax = entry.countMax || 1;
        const count = cMin + Math.floor(Math.random() * (cMax - cMin + 1));
        
        // Stack or push
        const existing = results.find(r => r.itemId === entry.itemId);
        if (existing) {
          existing.count += count;
        } else {
          results.push({ itemId: entry.itemId, count });
        }
        break;
      }
    }
  }

  return results;
}
