import { ItemRarity } from './ItemDef';

export interface ContainerDef {
  id: string;
  name: string;
  icon: string;
  defaultLootTable: string;
  minItems: number;
  maxItems: number;
  slots: number;
  isLocked?: boolean;
  lockType?: 'padlock' | 'keycard_red' | 'keycard_blue' | 'keycard_green' | 'crowbar';
  isJackpot?: boolean; // Triggers light beam and siren warning
  restockDays: number; // Restock cooldown in game days (e.g. 3 days)
  sparkleTier?: ItemRarity;
}

export const CONTAINER_REGISTRY: Record<string, ContainerDef> = {
  glove_box: {
    id: 'glove_box',
    name: 'Car Glove Box',
    icon: '📂',
    defaultLootTable: 'house_garage',
    minItems: 1,
    maxItems: 3,
    slots: 4,
    restockDays: 4,
    sparkleTier: 'C'
  },
  car_trunk: {
    id: 'car_trunk',
    name: 'Wreck Car Trunk',
    icon: '🚗',
    defaultLootTable: 'workshop',
    minItems: 2,
    maxItems: 5,
    slots: 8,
    isLocked: true,
    lockType: 'crowbar',
    restockDays: 5,
    sparkleTier: 'U'
  },
  suitcase: {
    id: 'suitcase',
    name: 'Traveler Luggage',
    icon: '🧳',
    defaultLootTable: 'house_bedroom',
    minItems: 2,
    maxItems: 4,
    slots: 6,
    restockDays: 3,
    sparkleTier: 'C'
  },
  locker: {
    id: 'locker',
    name: 'Metal Locker',
    icon: '🗄️',
    defaultLootTable: 'school',
    minItems: 1,
    maxItems: 3,
    slots: 6,
    isLocked: true,
    lockType: 'padlock',
    restockDays: 4,
    sparkleTier: 'U'
  },
  toolbox: {
    id: 'toolbox',
    name: 'Mechanic Red Toolbox',
    icon: '🧰',
    defaultLootTable: 'workshop',
    minItems: 2,
    maxItems: 4,
    slots: 6,
    restockDays: 3,
    sparkleTier: 'U'
  },
  fridge: {
    id: 'fridge',
    name: 'Kitchen Refrigerator',
    icon: '🧊',
    defaultLootTable: 'house_kitchen',
    minItems: 2,
    maxItems: 5,
    slots: 8,
    restockDays: 3,
    sparkleTier: 'C'
  },
  medicine_cabinet: {
    id: 'medicine_cabinet',
    name: 'Medicine Cabinet',
    icon: '💊',
    defaultLootTable: 'pharmacy',
    minItems: 2,
    maxItems: 4,
    slots: 6,
    restockDays: 3,
    sparkleTier: 'R'
  },
  cash_register: {
    id: 'cash_register',
    name: 'Antique Cash Register',
    icon: '📠',
    defaultLootTable: 'shop_general',
    minItems: 1,
    maxItems: 2,
    slots: 4,
    isLocked: true,
    lockType: 'crowbar',
    restockDays: 4,
    sparkleTier: 'C'
  },
  piggy_bank: {
    id: 'piggy_bank',
    name: 'Ceramic Piggy Bank',
    icon: '🐷',
    defaultLootTable: 'vault_jackpot',
    minItems: 1,
    maxItems: 2,
    slots: 2,
    restockDays: 5,
    sparkleTier: 'U'
  },
  supply_drop_crate: {
    id: 'supply_drop_crate',
    name: 'Parachute Supply Crate',
    icon: '🪂',
    defaultLootTable: 'outpost_crate',
    minItems: 4,
    maxItems: 7,
    slots: 10,
    isJackpot: true,
    restockDays: 7,
    sparkleTier: 'R'
  },
  vending_machine: {
    id: 'vending_machine',
    name: 'Vending Machine From Nowhere',
    icon: '🎰',
    defaultLootTable: 'shop_general',
    minItems: 1,
    maxItems: 2,
    slots: 4,
    restockDays: 3,
    sparkleTier: 'U'
  },
  pinata: {
    id: 'pinata',
    name: 'Mystery Crate Piñata',
    icon: '🪅',
    defaultLootTable: 'school',
    minItems: 3,
    maxItems: 6,
    slots: 8,
    restockDays: 4,
    sparkleTier: 'U'
  },
  vault_safe: {
    id: 'vault_safe',
    name: 'Titanium Vault Safe',
    icon: '🔒',
    defaultLootTable: 'vault_jackpot',
    minItems: 3,
    maxItems: 5,
    slots: 8,
    isLocked: true,
    lockType: 'keycard_red',
    isJackpot: true,
    restockDays: 10,
    sparkleTier: 'X'
  }
};
