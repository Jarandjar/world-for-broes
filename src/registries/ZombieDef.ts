import { ItemRarity } from './ItemDef';

export type ZombieAIState = 'wander' | 'investigate' | 'chase' | 'attack' | 'stunned' | 'flee_fire' | 'follow_horde';

export interface ZombieDef {
  id: string;
  name: string;
  rarity: ItemRarity;
  maxHealth: number;
  walkSpeed: number; // m/s
  chaseSpeed: number; // m/s
  damage: number;
  attackIntervalSeconds: number;
  hearingSensitivity: number; // 0.0 to 2.0 multiplier for noise ring alerts
  sightRadius: number; // in meters (day)
  sightRadiusNight: number; // in meters (night)
  isBoss?: boolean;
  biomeAffinity?: string;
  armorHits?: number; // e.g. traffic cone takes 1 hit before damage
  specialBehavior?: string;
  drops: { itemId: string; chance: number; count: [number, number] }[];
}

export const ZOMBIE_REGISTRY: Record<string, ZombieDef> = {
  shambler: {
    id: 'shambler',
    name: 'Shambling Bro-bie',
    rarity: 'C',
    maxHealth: 30,
    walkSpeed: 1.2,
    chaseSpeed: 2.2,
    damage: 8,
    attackIntervalSeconds: 1.4,
    hearingSensitivity: 1.0,
    sightRadius: 14,
    sightRadiusNight: 10,
    drops: [
      { itemId: 'gunk_goo', chance: 0.60, count: [1, 2] },
      { itemId: 'shiny_buttons', chance: 0.40, count: [1, 4] },
      { itemId: 'cloth_scrap', chance: 0.20, count: [1, 1] }
    ]
  },
  jogger: {
    id: 'jogger',
    name: 'Tracksuit Jogger',
    rarity: 'U',
    maxHealth: 22,
    walkSpeed: 2.4,
    chaseSpeed: 4.8,
    damage: 10,
    attackIntervalSeconds: 1.0,
    hearingSensitivity: 1.3,
    sightRadius: 18,
    sightRadiusNight: 14,
    specialBehavior: 'Prowls at dusk, sprints in small coordinated packs',
    drops: [
      { itemId: 'gunk_goo', chance: 0.50, count: [1, 2] },
      { itemId: 'energy_drink', chance: 0.25, count: [1, 1] },
      { itemId: 'shiny_buttons', chance: 0.50, count: [2, 6] }
    ]
  },
  traffic_cone_zombie: {
    id: 'traffic_cone_zombie',
    name: 'Cone-Headed Zombie',
    rarity: 'C',
    maxHealth: 35,
    walkSpeed: 1.3,
    chaseSpeed: 2.4,
    damage: 9,
    attackIntervalSeconds: 1.3,
    hearingSensitivity: 0.9,
    sightRadius: 12,
    sightRadiusNight: 9,
    armorHits: 1, // Cone deflects first blow completely
    specialBehavior: 'Orange cone absorbs first strike, flying off with a rubbery thwack',
    drops: [
      { itemId: 'gunk_goo', chance: 0.70, count: [1, 2] },
      { itemId: 'traffic_cone_hat', chance: 0.15, count: [1, 1] },
      { itemId: 'scrap_metal', chance: 0.35, count: [1, 2] }
    ]
  },
  clown_zombie: {
    id: 'clown_zombie',
    name: 'Squeaky Clown Zombie',
    rarity: 'U',
    maxHealth: 28,
    walkSpeed: 1.6,
    chaseSpeed: 3.2,
    damage: 7,
    attackIntervalSeconds: 1.2,
    hearingSensitivity: 1.5,
    sightRadius: 16,
    sightRadiusNight: 12,
    specialBehavior: 'Squeaks comically per stride, broadcasting noise that alerts other zombies',
    drops: [
      { itemId: 'gunk_goo', chance: 0.50, count: [1, 2] },
      { itemId: 'whoopee_cushion', chance: 0.30, count: [1, 1] },
      { itemId: 'rubber_chicken', chance: 0.10, count: [1, 1] }
    ]
  },
  bloater_balloon: {
    id: 'bloater_balloon',
    name: 'Bloater Balloon',
    rarity: 'U',
    maxHealth: 45,
    walkSpeed: 0.9,
    chaseSpeed: 1.8,
    damage: 16,
    attackIntervalSeconds: 2.0,
    hearingSensitivity: 0.8,
    sightRadius: 10,
    sightRadiusNight: 8,
    specialBehavior: 'Inflates and pops into a 4m toxic green sludge cloud that slows players 50%',
    drops: [
      { itemId: 'gunk_goo', chance: 0.90, count: [2, 4] },
      { itemId: 'stink_sock_bomb', chance: 0.20, count: [1, 1] }
    ]
  },
  brute_in_a_bathrobe: {
    id: 'brute_in_a_bathrobe',
    name: 'Bathrobe Heavyweight',
    rarity: 'R',
    maxHealth: 110,
    walkSpeed: 1.1,
    chaseSpeed: 2.8,
    damage: 26,
    attackIntervalSeconds: 2.2,
    hearingSensitivity: 1.2,
    sightRadius: 15,
    sightRadiusNight: 12,
    specialBehavior: 'Headbutts cars dealing chassis damage, stuns players for 1.2s',
    drops: [
      { itemId: 'gunk_goo', chance: 1.0, count: [3, 5] },
      { itemId: 'pillow_vest', chance: 0.35, count: [1, 1] },
      { itemId: 'spring', chance: 0.50, count: [2, 4] }
    ]
  },
  screamer_karaoke: {
    id: 'screamer_karaoke',
    name: 'Karaoke Screamer',
    rarity: 'R',
    maxHealth: 30,
    walkSpeed: 1.8,
    chaseSpeed: 3.6,
    damage: 5,
    attackIntervalSeconds: 1.5,
    hearingSensitivity: 2.0,
    sightRadius: 22,
    sightRadiusNight: 18,
    specialBehavior: 'Screeches into a mic when spotting players, instantly summoning nearby horde',
    drops: [
      { itemId: 'gunk_goo', chance: 0.60, count: [1, 2] },
      { itemId: 'batteries_aa', chance: 0.50, count: [2, 4] },
      { itemId: 'air_horn', chance: 0.20, count: [1, 1] }
    ]
  },
  crawler_roomba: {
    id: 'crawler_roomba',
    name: 'Grass Crawler',
    rarity: 'C',
    maxHealth: 18,
    walkSpeed: 1.0,
    chaseSpeed: 2.0,
    damage: 6,
    attackIntervalSeconds: 1.1,
    hearingSensitivity: 0.7,
    sightRadius: 8,
    sightRadiusNight: 6,
    specialBehavior: 'Low profile hiding in tall foliage; trips running players',
    drops: [
      { itemId: 'gunk_goo', chance: 0.40, count: [1, 1] },
      { itemId: 'cloth_scrap', chance: 0.30, count: [1, 2] }
    ]
  },
  zombie_goose: {
    id: 'zombie_goose',
    name: 'Undead Honk Goose',
    rarity: 'U',
    maxHealth: 20,
    walkSpeed: 2.2,
    chaseSpeed: 4.2,
    damage: 8,
    attackIntervalSeconds: 0.9,
    hearingSensitivity: 1.8,
    sightRadius: 20,
    sightRadiusNight: 15,
    specialBehavior: 'Honks aggressively, steals loose items, flees when honked at with H',
    drops: [
      { itemId: 'feather', chance: 0.80, count: [2, 5] },
      { itemId: 'cheese_hat', chance: 0.10, count: [1, 1] }
    ]
  },

  // --- BIOME VARIANTS ---
  frost_zombie: {
    id: 'frost_zombie',
    name: 'Brrington Frost Zombie',
    rarity: 'U',
    biomeAffinity: 'tundra',
    maxHealth: 40,
    walkSpeed: 1.1,
    chaseSpeed: 2.1,
    damage: 11,
    attackIntervalSeconds: 1.5,
    hearingSensitivity: 0.8,
    sightRadius: 12,
    sightRadiusNight: 9,
    specialBehavior: 'Chilling aura drains body warmth and slows victims',
    drops: [
      { itemId: 'ice_chunk', chance: 0.70, count: [1, 3] },
      { itemId: 'hand_warmers', chance: 0.25, count: [1, 2] }
    ]
  },
  husk_zombie: {
    id: 'husk_zombie',
    name: 'Sunburn Husk',
    rarity: 'U',
    biomeAffinity: 'desert',
    maxHealth: 32,
    walkSpeed: 1.4,
    chaseSpeed: 2.6,
    damage: 10,
    attackIntervalSeconds: 1.3,
    hearingSensitivity: 1.1,
    sightRadius: 16,
    sightRadiusNight: 12,
    specialBehavior: 'Attacks accelerate player thirst / heat meter',
    drops: [
      { itemId: 'water_pulp', chance: 0.40, count: [1, 2] },
      { itemId: 'sun_hat', chance: 0.15, count: [1, 1] }
    ]
  },
  mud_zombie: {
    id: 'mud_zombie',
    name: 'Squelchy Mud Zombie',
    rarity: 'U',
    biomeAffinity: 'bog',
    maxHealth: 38,
    walkSpeed: 0.9,
    chaseSpeed: 1.9,
    damage: 9,
    attackIntervalSeconds: 1.6,
    hearingSensitivity: 0.9,
    sightRadius: 10,
    sightRadiusNight: 8,
    specialBehavior: 'Burrows beneath mud puddles and grabs passing legs',
    drops: [
      { itemId: 'cattail_fiber', chance: 0.60, count: [1, 3] },
      { itemId: 'gunk_goo', chance: 0.80, count: [1, 3] }
    ]
  },
  ember_zombie: {
    id: 'ember_zombie',
    name: 'Volcanic Ash Zombie',
    rarity: 'R',
    biomeAffinity: 'volcano',
    maxHealth: 50,
    walkSpeed: 1.5,
    chaseSpeed: 3.0,
    damage: 15,
    attackIntervalSeconds: 1.2,
    hearingSensitivity: 1.0,
    sightRadius: 15,
    sightRadiusNight: 15,
    specialBehavior: 'Leaves an ember trail; ignites unfortified wooden structures',
    drops: [
      { itemId: 'ember_fruit', chance: 0.30, count: [1, 1] },
      { itemId: 'flint', chance: 0.70, count: [2, 4] }
    ]
  },
  ghost_sheet_zombie: {
    id: 'ghost_sheet_zombie',
    name: 'Hollow Sheet Specter',
    rarity: 'R',
    biomeAffinity: 'hollow',
    maxHealth: 25,
    walkSpeed: 1.8,
    chaseSpeed: 3.4,
    damage: 8,
    attackIntervalSeconds: 1.0,
    hearingSensitivity: 1.4,
    sightRadius: 16,
    sightRadiusNight: 16,
    specialBehavior: 'Floats over obstacles, emitting eerie whispers that rapidly drain sanity',
    drops: [
      { itemId: 'ghost_sheet', chance: 0.60, count: [1, 2] },
      { itemId: 'glow_shard', chance: 0.40, count: [1, 2] }
    ]
  },

  // --- BOSS ---
  highway_hulk: {
    id: 'highway_hulk',
    name: 'Highway Hulk, Wreckage Behemoth',
    rarity: 'X',
    maxHealth: 450,
    walkSpeed: 1.8,
    chaseSpeed: 3.8,
    damage: 40,
    attackIntervalSeconds: 2.5,
    hearingSensitivity: 1.5,
    sightRadius: 28,
    sightRadiusNight: 24,
    isBoss: true,
    specialBehavior: 'Stitched from engine blocks and fenders. Charges down highway lanes, headbutting roadblocks',
    drops: [
      { itemId: 'engine_block', chance: 1.0, count: [1, 1] },
      { itemId: 'shiny_buttons', chance: 1.0, count: [30, 60] },
      { itemId: 'welding_kit', chance: 1.0, count: [1, 1] },
      { itemId: 'golden_toilet_seat', chance: 0.5, count: [1, 1] }
    ]
  }
};
