export type ItemRarity = 'J' | 'C' | 'U' | 'R' | 'X'; // Junk, Common, Uncommon, Rare, Jackpot

export type ItemCategory =
  | 'resource'
  | 'tool'
  | 'food'
  | 'weapon_melee'
  | 'weapon_ranged'
  | 'throwable'
  | 'trap'
  | 'armor'
  | 'medical'
  | 'movement'
  | 'vehicle_part'
  | 'junk'
  | 'ammo'
  | 'currency'
  | 'quest'
  | 'cosmetic';

export interface ItemDef {
  id: string;
  name: string;
  icon: string;
  category: ItemCategory;
  rarity: ItemRarity;
  maxStack: number;
  weight: number; // 0.2 (light), 1.0 (medium), 3.0 (heavy), 8.0 (very heavy)
  description?: string;
  
  // Vitals
  hungerBonus?: number;
  healthBonus?: number;
  sanityBonus?: number;
  warmthBonus?: number;

  // Combat & Durability
  damage?: number;
  speed?: 'fast' | 'medium' | 'slow';
  noise?: number; // 0 (silent) to 5 (alarm)
  durability?: number;
  range?: number;
  ammoType?: string;
  
  // Armor & Protection
  protection?: number; // percentage damage mitigation
  biteResistance?: number; // percentage bite reduction
  coldProtection?: number;
  heatProtection?: number;

  // Effects & Special
  specialEffect?: string;
}

export const ITEM_REGISTRY: Record<string, ItemDef> = {
  // --- BASE & NATURAL RESOURCES ---
  log: {
    id: 'log',
    name: 'Log',
    icon: '🪵',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 1.0,
    description: 'Chipped from a pine or oak tree. Sturdy fuel or crafting lumber.'
  },
  twigs: {
    id: 'twigs',
    name: 'Twigs',
    icon: '🥢',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    description: 'Bendable sticks for basic tools and campfires.'
  },
  grass: {
    id: 'grass',
    name: 'Cut Grass',
    icon: '🌾',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    description: 'Flexible fibers for ropes, kindling, and light armor.'
  },
  flint: {
    id: 'flint',
    name: 'Flint',
    icon: '💎',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.5,
    description: 'Sharp sedimentary rock for blades, fire, and pickaxes.'
  },
  stone: {
    id: 'stone',
    name: 'Stone',
    icon: '🪨',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 1.0,
    description: 'Heavy granite chunks for campfire rings and masonry.'
  },
  gold_nugget: {
    id: 'gold_nugget',
    name: 'Gold Nugget',
    icon: '🪙',
    category: 'resource',
    rarity: 'U',
    maxStack: 30,
    weight: 0.5,
    description: 'Shiny malleable gold. Works in vending machines and high-tier gear.'
  },
  pine_cone: {
    id: 'pine_cone',
    name: 'Pine Cone',
    icon: '🌰',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    description: 'Plant in soil to regrow a pine tree.'
  },
  acorns: {
    id: 'acorns',
    name: 'Acorns',
    icon: '🥜',
    category: 'resource',
    rarity: 'U',
    maxStack: 20,
    weight: 0.2,
    description: 'Drops from oak trees. Plantable or roastable.'
  },
  ghost_sheet: {
    id: 'ghost_sheet',
    name: 'Ghost Sheet',
    icon: '👻',
    category: 'resource',
    rarity: 'U',
    maxStack: 10,
    weight: 0.5,
    description: 'Ethereal cloth dropped by hollow dead trees.'
  },
  feather: {
    id: 'feather',
    name: 'Goose Feather',
    icon: '🪶',
    category: 'resource',
    rarity: 'C',
    maxStack: 30,
    weight: 0.1,
    description: 'Fluffy white feather dropped by honk geese. Used for dart fletching and pillow armor.'
  },
  glow_shard: {
    id: 'glow_shard',
    name: 'Glow Shard',
    icon: '✨',
    category: 'resource',
    rarity: 'U',
    maxStack: 20,
    weight: 0.2,
    description: 'Luminescent crystal shard from Rockbottom Ridge.'
  },
  water_pulp: {
    id: 'water_pulp',
    name: 'Cactus Pulp',
    icon: '🌵',
    category: 'food',
    rarity: 'C',
    maxStack: 20,
    weight: 0.5,
    hungerBonus: 8,
    healthBonus: 2,
    warmthBonus: -5,
    description: 'Hydrating desert cactus pulp.'
  },
  cattail_fiber: {
    id: 'cattail_fiber',
    name: 'Cattail Fiber',
    icon: '🌾',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    description: 'Tough marsh reed fibers.'
  },
  ice_chunk: {
    id: 'ice_chunk',
    name: 'Ice Chunk',
    icon: '🧊',
    category: 'resource',
    rarity: 'U',
    maxStack: 20,
    weight: 0.5,
    description: 'Solid frozen block from Brrington Tundra.'
  },
  ember_fruit: {
    id: 'ember_fruit',
    name: 'Ember Fruit',
    icon: '🔥',
    category: 'food',
    rarity: 'R',
    maxStack: 10,
    weight: 0.5,
    hungerBonus: 20,
    healthBonus: 10,
    warmthBonus: 25,
    description: 'Warm glowing fruit picked near Mount Grumbleguts.'
  },

  // --- FOOD & WILD EDIBLES ---
  berries: {
    id: 'berries',
    name: 'Sweet Berries',
    icon: '🍓',
    category: 'food',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    hungerBonus: 15,
    healthBonus: 3,
    description: 'Fresh tart meadow berries.'
  },
  cooked_berries: {
    id: 'cooked_berries',
    name: 'Roasted Berries',
    icon: '🥧',
    category: 'food',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    hungerBonus: 25,
    healthBonus: 12,
    sanityBonus: 5,
    description: 'Caramelized warm berries.'
  },
  wild_carrot: {
    id: 'wild_carrot',
    name: 'Wild Carrot',
    icon: '🥕',
    category: 'food',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2,
    hungerBonus: 18,
    healthBonus: 5,
    description: 'Crunchy root pulled with a squeak.'
  },
  glow_mushroom: {
    id: 'glow_mushroom',
    name: 'Glow Mushroom',
    icon: '🍄',
    category: 'food',
    rarity: 'U',
    maxStack: 20,
    weight: 0.2,
    hungerBonus: 10,
    healthBonus: -2,
    sanityBonus: 15,
    description: 'Eerie glowing fungus that restores mental calm.'
  },
  honey: {
    id: 'honey',
    name: 'Wild Honey',
    icon: '🍯',
    category: 'food',
    rarity: 'U',
    maxStack: 10,
    weight: 0.5,
    hungerBonus: 25,
    healthBonus: 15,
    sanityBonus: 8,
    description: 'Golden sweetness harvested from buzz hives.'
  },
  raw_fish: {
    id: 'raw_fish',
    name: 'Raw Pond Fish',
    icon: '🐟',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5,
    hungerBonus: 12,
    healthBonus: -3,
    sanityBonus: -5,
    description: 'Freshly reeled fish. Better when roasted over fire.'
  },
  cooked_fish: {
    id: 'cooked_fish',
    name: 'Crispy Grilled Fish',
    icon: '🍳',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5,
    hungerBonus: 35,
    healthBonus: 20,
    sanityBonus: 10,
    description: 'Flaky golden fish with a charred skin.'
  },
  marshmallow: {
    id: 'marshmallow',
    name: 'Flaming Marshmallow',
    icon: '🍢',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    hungerBonus: 20,
    healthBonus: 5,
    sanityBonus: 15,
    durability: 60,
    description: 'Gooey roasted goodness on a stick.'
  },

  // --- GOOFY CAULDRON RECIPES ---
  hiccup_berries: {
    id: 'hiccup_berries',
    name: 'Hiccup Berries',
    icon: '🫐',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    hungerBonus: 25,
    specialEffect: 'Hiccups and hops slightly for 30s'
  },
  giggle_mushroom: {
    id: 'giggle_mushroom',
    name: 'Giggle Mushroom',
    icon: '🤪',
    category: 'food',
    rarity: 'U',
    maxStack: 10,
    weight: 0.2,
    hungerBonus: 15,
    sanityBonus: 30,
    specialEffect: 'Laugh bubbles and wobbly steering'
  },
  chili_of_regret: {
    id: 'chili_of_regret',
    name: 'Chili of Regret',
    icon: '🌶️',
    category: 'food',
    rarity: 'U',
    maxStack: 5,
    weight: 0.5,
    hungerBonus: 50,
    warmthBonus: 40,
    specialEffect: '10s of fire-breath puffs that scare zombies'
  },
  mystery_stew: {
    id: 'mystery_stew',
    name: 'Mystery Stew',
    icon: '🍲',
    category: 'food',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    hungerBonus: 40,
    specialEffect: 'Random comedic cosmetic effect'
  },
  pancake_stack: {
    id: 'pancake_stack',
    name: 'Pancake Stack',
    icon: '🥞',
    category: 'food',
    rarity: 'U',
    maxStack: 5,
    weight: 1.0,
    hungerBonus: 60,
    sanityBonus: 25,
    specialEffect: 'Wobbles comically while carried'
  },
  rainbow_soup: {
    id: 'rainbow_soup',
    name: 'Rainbow Soup',
    icon: '🥣',
    category: 'food',
    rarity: 'R',
    maxStack: 5,
    weight: 0.5,
    hungerBonus: 45,
    sanityBonus: 35,
    specialEffect: 'Sweat glows in rainbow colors illuminating darkness for 60s'
  },
  bubble_tea_of_buoyancy: {
    id: 'bubble_tea_of_buoyancy',
    name: 'Bubble Tea of Buoyancy',
    icon: '🧋',
    category: 'food',
    rarity: 'U',
    maxStack: 5,
    weight: 0.5,
    hungerBonus: 15,
    specialEffect: 'Float harmlessly on deep water for 30s'
  },
  crispy_sock_chips: {
    id: 'crispy_sock_chips',
    name: 'Crispy Sock Chips',
    icon: '🍟',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    hungerBonus: 10,
    specialEffect: 'Absurd crunch audio and temporary sprint boost'
  },
  roast_sausage_stick: {
    id: 'roast_sausage_stick',
    name: 'Roast Sausage Stick',
    icon: '🌭',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5,
    hungerBonus: 35,
    healthBonus: 10
  },

  // --- MELEE WEAPONS ---
  stick: {
    id: 'stick',
    name: 'Pointy Stick',
    icon: '🥢',
    category: 'weapon_melee',
    rarity: 'J',
    maxStack: 1,
    weight: 0.2,
    damage: 5,
    speed: 'fast',
    noise: 1,
    durability: 25,
    description: 'The humble starter branch. Snaps with a cartoon crack.'
  },
  stale_baguette: {
    id: 'stale_baguette',
    name: 'Stale Baguette',
    icon: '🥖',
    category: 'weapon_melee',
    rarity: 'J',
    maxStack: 1,
    weight: 0.5,
    damage: 6,
    speed: 'fast',
    noise: 0,
    durability: 20,
    description: 'Rock-hard bread. Breaks into crumbs that pigeons chase.'
  },
  nail_bat: {
    id: 'nail_bat',
    name: 'Nail Bat',
    icon: '🏏',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 14,
    speed: 'medium',
    noise: 2,
    durability: 60,
    description: 'Classic apocalyptic slugger. Nails glint with a satisfying thwack.'
  },
  crowbar: {
    id: 'crowbar',
    name: 'Crowbar',
    icon: '🔧',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 12,
    speed: 'medium',
    noise: 2,
    durability: 80,
    description: 'Opens locked crates, clears roadblocks, and smacks zombies.'
  },
  frying_pan: {
    id: 'frying_pan',
    name: 'Frying Pan',
    icon: '🍳',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 10,
    speed: 'medium',
    noise: 3,
    durability: 70,
    specialEffect: '15% stun chance with a loud cartoon BONG'
  },
  pan: {
    id: 'pan',
    name: 'Frying Pan',
    icon: '🍳',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 10,
    speed: 'medium',
    noise: 3,
    durability: 80
  },
  hockey_stick: {
    id: 'hockey_stick',
    name: 'Hockey Stick',
    icon: '🏒',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 11,
    speed: 'medium',
    noise: 2,
    durability: 60,
    range: 2.2
  },
  pool_noodle_of_justice: {
    id: 'pool_noodle_of_justice',
    name: 'Pool Noodle of Justice',
    icon: '🫧',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    damage: 2,
    speed: 'fast',
    noise: 0,
    durability: 999,
    range: 2.5,
    specialEffect: 'Huge knockback with a comical boing sound'
  },
  rubber_chicken: {
    id: 'rubber_chicken',
    name: 'Rubber Chicken',
    icon: '🐔',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    damage: 4,
    speed: 'fast',
    noise: 2,
    durability: 999,
    specialEffect: 'Stretchy swing, squeak with randomized pitch, scares critters'
  },
  chicken: {
    id: 'chicken',
    name: 'Rubber Chicken',
    icon: '🐔',
    category: 'weapon_melee',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    damage: 4,
    speed: 'fast',
    noise: 2,
    durability: 100
  },
  mega_mallet: {
    id: 'mega_mallet',
    name: 'Mega Mallet',
    icon: '🔨',
    category: 'weapon_melee',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    damage: 26,
    speed: 'slow',
    noise: 4,
    durability: 45,
    specialEffect: 'Ground crack, 60ms hit-stop, dizzy spinning stars'
  },
  traffic_sign_axe: {
    id: 'traffic_sign_axe',
    name: 'Traffic Sign Axe',
    icon: '🛑',
    category: 'weapon_melee',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    damage: 20,
    speed: 'slow',
    noise: 3,
    durability: 50
  },
  fire_axe: {
    id: 'fire_axe',
    name: 'Emergency Fire Axe',
    icon: '🪓',
    category: 'weapon_melee',
    rarity: 'R',
    maxStack: 1,
    weight: 2.0,
    damage: 28,
    speed: 'medium',
    noise: 3,
    durability: 80,
    description: 'Chops locked wooden doors, tall trees, and zombies alike.'
  },
  chainsaw_of_regret: {
    id: 'chainsaw_of_regret',
    name: 'Chainsaw of Regret',
    icon: '⚙️',
    category: 'weapon_melee',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    damage: 35,
    speed: 'fast',
    noise: 5,
    durability: 200,
    specialEffect: 'Burns fuel, shakes camera, pops zombies into green puffs'
  },

  // --- RANGED WEAPONS ---
  slingshot: {
    id: 'slingshot',
    name: 'Pocket Slingshot',
    icon: '🎯',
    category: 'weapon_ranged',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    damage: 8,
    noise: 1,
    durability: 75,
    ammoType: 'pebbles'
  },
  bow_of_branches: {
    id: 'bow_of_branches',
    name: 'Bow of Branches',
    icon: '🏹',
    category: 'weapon_ranged',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    damage: 12,
    noise: 1,
    durability: 50,
    ammoType: 'twigs'
  },
  foam_dart_blaster: {
    id: 'foam_dart_blaster',
    name: 'Foam Dart Blaster',
    icon: '🔫',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 0.5,
    damage: 6,
    speed: 'fast',
    noise: 1,
    durability: 120,
    ammoType: 'foam_darts'
  },
  crossbow_of_corks: {
    id: 'crossbow_of_corks',
    name: 'Crossbow of Corks',
    icon: '🎯',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    damage: 18,
    speed: 'slow',
    noise: 1,
    durability: 60,
    ammoType: 'corks'
  },
  squirt_cannon: {
    id: 'squirt_cannon',
    name: 'Squirt Cannon',
    icon: '💦',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    damage: 0,
    noise: 1,
    durability: 100,
    specialEffect: 'Slows and soaks zombies, puts out fires, refills at ponds'
  },
  plunger_launcher: {
    id: 'plunger_launcher',
    name: 'Plunger Launcher',
    icon: '🪠',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    damage: 4,
    noise: 2,
    durability: 30,
    specialEffect: 'Sticks zombie to wall or ground for 4 seconds'
  },
  potato_cannon: {
    id: 'potato_cannon',
    name: 'Potato Cannon',
    icon: '🥔',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    damage: 15,
    noise: 3,
    durability: 50,
    ammoType: 'potatoes'
  },
  flare_gun: {
    id: 'flare_gun',
    name: 'Emergency Flare Gun',
    icon: '🚨',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    damage: 5,
    noise: 4,
    durability: 20,
    ammoType: 'flare_cartridges',
    specialEffect: 'Lights up radius, scares night terrors or lures hordes'
  },
  nail_gun_of_justice: {
    id: 'nail_gun_of_justice',
    name: 'Nail Gun of Justice',
    icon: '🔨',
    category: 'weapon_ranged',
    rarity: 'R',
    maxStack: 1,
    weight: 3.0,
    damage: 10,
    speed: 'fast',
    noise: 3,
    durability: 100,
    ammoType: 'nails'
  },
  firework_launcher: {
    id: 'firework_launcher',
    name: 'Firework Launcher',
    icon: '🎆',
    category: 'weapon_ranged',
    rarity: 'R',
    maxStack: 1,
    weight: 3.0,
    damage: 40,
    noise: 5,
    durability: 25,
    ammoType: 'bottle_rockets',
    specialEffect: 'AoE explosion with colorful sparks and loud bangs'
  },
  confetti_cannon: {
    id: 'confetti_cannon',
    name: 'Confetti Cannon',
    icon: '🎉',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    damage: 2,
    noise: 3,
    durability: 15,
    specialEffect: 'Scares night creatures for 6s, restores player sanity'
  },
  bubble_wand: {
    id: 'bubble_wand',
    name: 'Bubble Wand',
    icon: '🫧',
    category: 'weapon_ranged',
    rarity: 'U',
    maxStack: 1,
    weight: 0.2,
    damage: 0,
    noise: 0,
    durability: 40,
    specialEffect: 'Traps small creatures or floats items in iridescent bubbles'
  },

  // --- THROWABLES ---
  water_balloon: {
    id: 'water_balloon',
    name: 'Water Balloon',
    icon: '🎈',
    category: 'throwable',
    rarity: 'J',
    maxStack: 10,
    weight: 0.2,
    specialEffect: 'Splashes water and slows targets'
  },
  cream_pie: {
    id: 'cream_pie',
    name: 'Cream Pie',
    icon: '🥧',
    category: 'throwable',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Blinds target for 4 seconds with pie splat decal'
  },
  road_flare: {
    id: 'road_flare',
    name: 'Road Flare',
    icon: '🧨',
    category: 'throwable',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Burns with red light for 45s, lures or scares zombies'
  },
  marbles: {
    id: 'marbles',
    name: 'Bag of Marbles',
    icon: '🔮',
    category: 'throwable',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Scatters marbles on floor, slipping chasing zombies'
  },
  banana_peel: {
    id: 'banana_peel',
    name: 'Banana Peel',
    icon: '🍌',
    category: 'throwable',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    specialEffect: 'Slide 3 meters with cartoon whoosh and spinning stars'
  },
  alarm_clock_decoy: {
    id: 'alarm_clock_decoy',
    name: 'Alarm Clock Decoy',
    icon: '⏰',
    category: 'throwable',
    rarity: 'C',
    maxStack: 3,
    weight: 0.5,
    specialEffect: 'Rings loudly after 5s delay, pulling all nearby zombies'
  },
  stink_sock_bomb: {
    id: 'stink_sock_bomb',
    name: 'Stink Sock Bomb',
    icon: '🧦',
    category: 'throwable',
    rarity: 'U',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Releases pungent green odor cloud making zombies stagger'
  },
  smoke_pot: {
    id: 'smoke_pot',
    name: 'Smoke Pot',
    icon: '💨',
    category: 'throwable',
    rarity: 'U',
    maxStack: 3,
    weight: 1.0,
    specialEffect: 'Dense smoke screen that cuts enemy line-of-sight'
  },
  boombox_decoy: {
    id: 'boombox_decoy',
    name: 'Boombox Decoy',
    icon: '📻',
    category: 'throwable',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    specialEffect: 'Plays funk tunes for 20s, pulling whole hordes away'
  },
  bowling_ball: {
    id: 'bowling_ball',
    name: 'Heavy Bowling Ball',
    icon: '🎳',
    category: 'throwable',
    rarity: 'U',
    maxStack: 2,
    weight: 3.0,
    damage: 30,
    specialEffect: 'Rolls down road knocking zombies over like pins'
  },
  glitter_bomb: {
    id: 'glitter_bomb',
    name: 'Glitter Bomb',
    icon: '✨',
    category: 'throwable',
    rarity: 'R',
    maxStack: 3,
    weight: 0.5,
    specialEffect: 'Blinds crowd and marks all zombies with sparkling trails'
  },

  // --- TRAPS & DEFENSES ---
  tripwire_cans: {
    id: 'tripwire_cans',
    name: 'Tripwire Tin Cans',
    icon: '🥫',
    category: 'trap',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    description: 'Noisy perimeter alarm when breached.'
  },
  spike_strip: {
    id: 'spike_strip',
    name: 'Spike Strip',
    icon: '⛓️',
    category: 'trap',
    rarity: 'C',
    maxStack: 5,
    weight: 1.0,
    damage: 15,
    description: 'Pierces zombie feet or car tires.'
  },
  barricade_planks: {
    id: 'barricade_planks',
    name: 'Barricade Planks',
    icon: '🪵',
    category: 'trap',
    rarity: 'C',
    maxStack: 10,
    weight: 1.0,
    description: 'Nail across doors and windows.'
  },
  sandbag: {
    id: 'sandbag',
    name: 'Sandbag',
    icon: '🧱',
    category: 'trap',
    rarity: 'C',
    maxStack: 5,
    weight: 3.0,
    description: 'Bulletproof and zombie-proof obstacle.'
  },
  shopping_cart_barrier: {
    id: 'shopping_cart_barrier',
    name: 'Shopping Cart Barrier',
    icon: '🛒',
    category: 'trap',
    rarity: 'C',
    maxStack: 2,
    weight: 3.0,
    description: 'Chained carts blocking alleys and doors.'
  },
  whoopee_cushion_trap: {
    id: 'whoopee_cushion_trap',
    name: 'Whoopee Cushion Trap',
    icon: '💨',
    category: 'trap',
    rarity: 'C',
    maxStack: 5,
    weight: 0.2,
    specialEffect: 'Zombies sit on it, become stunned and embarrassed'
  },
  barbed_wire_tangle: {
    id: 'barbed_wire_tangle',
    name: 'Barbed Wire Tangle',
    icon: '🪤',
    category: 'trap',
    rarity: 'U',
    maxStack: 3,
    weight: 2.0,
    damage: 10,
    specialEffect: 'Entangles and damages passing zombies'
  },
  sprinkler_slip_n_slide: {
    id: 'sprinkler_slip_n_slide',
    name: 'Sprinkler Slip-n-Slide',
    icon: '🚿',
    category: 'trap',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    specialEffect: 'Soaks ground, sliding zombies into walls'
  },
  clam_trap: {
    id: 'clam_trap',
    name: 'Giant Clam Trap',
    icon: '🦪',
    category: 'trap',
    rarity: 'U',
    maxStack: 2,
    weight: 3.0,
    damage: 25,
    specialEffect: 'Snaps shut on zombie legs'
  },
  scarecrow_decoy: {
    id: 'scarecrow_decoy',
    name: 'Old Stuffins Decoy',
    icon: '🌾',
    category: 'trap',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    specialEffect: 'Draws zombie attention away from base'
  },
  searchlight_tower: {
    id: 'searchlight_tower',
    name: 'Searchlight Tower',
    icon: '🔦',
    category: 'trap',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    specialEffect: 'Sweeping bright beam that banishes night lurkers'
  },
  foam_turret: {
    id: 'foam_turret',
    name: 'Automatic Foam Turret',
    icon: '🤖',
    category: 'trap',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    specialEffect: 'Auto-fires foam darts at approaching zombies; needs power'
  },
  flame_pit: {
    id: 'flame_pit',
    name: 'Fueled Flame Pit',
    icon: '🔥',
    category: 'trap',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    damage: 40,
    specialEffect: 'Burns zombies attempting to climb perimeter walls'
  },

  // --- ARMOR & WEARABLES ---
  bucket_helmet: {
    id: 'bucket_helmet',
    name: 'Bucket Helmet',
    icon: '🪣',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    protection: 15,
    specialEffect: 'Absorbs 1 heavy hit completely, then dents'
  },
  bike_helmet: {
    id: 'bike_helmet',
    name: 'Bike Helmet',
    icon: '🪖',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    protection: 10,
    specialEffect: 'Greatly reduces vehicle crash damage'
  },
  pot_helmet: {
    id: 'pot_helmet',
    name: 'Cooking Pot Helmet',
    icon: '🍲',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    protection: 10,
    specialEffect: 'Comical clang sound when hit'
  },
  pillow_vest: {
    id: 'pillow_vest',
    name: 'Pillow Vest',
    icon: '🦺',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    biteResistance: 20,
    specialEffect: 'Feathers burst out on impact'
  },
  cardboard_armor: {
    id: 'cardboard_armor',
    name: 'Cardboard Box Armor',
    icon: '📦',
    category: 'armor',
    rarity: 'J',
    maxStack: 1,
    weight: 0.5,
    protection: 8,
    specialEffect: 'Weakens when soaked by rain or water'
  },
  hockey_pads: {
    id: 'hockey_pads',
    name: 'Hockey Shoulder Pads',
    icon: '🥋',
    category: 'armor',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    protection: 20
  },
  rain_poncho: {
    id: 'rain_poncho',
    name: 'Yellow Rain Poncho',
    icon: '🧥',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Prevents getting soaked in rain'
  },
  winter_coat: {
    id: 'winter_coat',
    name: 'Puffy Winter Coat',
    icon: '🧥',
    category: 'armor',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    coldProtection: 50
  },
  sun_hat: {
    id: 'sun_hat',
    name: 'Straw Sun Hat',
    icon: '👒',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    heatProtection: 30
  },
  snow_boots: {
    id: 'snow_boots',
    name: 'Fuzzy Snow Boots',
    icon: '🥾',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Prevents slipping on ice roads'
  },
  swim_floaties: {
    id: 'swim_floaties',
    name: 'Duck Swim Floaties',
    icon: '🛟',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Prevents drowning in deep cove water'
  },
  hi_vis_vest: {
    id: 'hi_vis_vest',
    name: 'Neon Hi-Vis Vest',
    icon: '🦺',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Teammates show brighter on minimap'
  },
  fanny_pack_of_holding: {
    id: 'fanny_pack_of_holding',
    name: 'Fanny Pack of Holding',
    icon: '👜',
    category: 'armor',
    rarity: 'R',
    maxStack: 1,
    weight: 0.2,
    specialEffect: '+4 Quick hotbar inventory slots'
  },
  school_backpack: {
    id: 'school_backpack',
    name: 'School Backpack',
    icon: '🎒',
    category: 'armor',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    specialEffect: '+6 Inventory slots'
  },
  hiker_backpack: {
    id: 'hiker_backpack',
    name: 'Hiker Backpack',
    icon: '🎒',
    category: 'armor',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    specialEffect: '+12 Inventory slots'
  },

  // --- MEDICAL & SURVIVAL ---
  bandage: {
    id: 'bandage',
    name: 'Sterile Bandage',
    icon: '🩹',
    category: 'medical',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    healthBonus: 20
  },
  painkillers: {
    id: 'painkillers',
    name: 'Painkillers',
    icon: '💊',
    category: 'medical',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    healthBonus: 10,
    sanityBonus: 10
  },
  first_aid_kit: {
    id: 'first_aid_kit',
    name: 'First Aid Kit',
    icon: '🧰',
    category: 'medical',
    rarity: 'U',
    maxStack: 5,
    weight: 1.0,
    healthBonus: 60
  },
  antidote_gel: {
    id: 'antidote_gel',
    name: 'Antidote Gel',
    icon: '🧪',
    category: 'medical',
    rarity: 'R',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Cures zombie infection meter'
  },
  energy_drink: {
    id: 'energy_drink',
    name: 'Bro Energy Drink',
    icon: '⚡',
    category: 'medical',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5,
    specialEffect: 'Boosts stamina for 60s, followed by hilarious jitters'
  },
  thermos_of_cocoa: {
    id: 'thermos_of_cocoa',
    name: 'Thermos of Hot Cocoa',
    icon: '☕',
    category: 'food',
    rarity: 'U',
    maxStack: 5,
    weight: 0.5,
    hungerBonus: 15,
    sanityBonus: 20,
    warmthBonus: 35
  },
  water_bottle: {
    id: 'water_bottle',
    name: 'Fresh Water Bottle',
    icon: '🍶',
    category: 'food',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5,
    healthBonus: 5,
    sanityBonus: 5
  },
  water_purifier: {
    id: 'water_purifier',
    name: 'Water Purifier Tablets',
    icon: '🫧',
    category: 'medical',
    rarity: 'U',
    maxStack: 10,
    weight: 0.2
  },
  hand_warmers: {
    id: 'hand_warmers',
    name: 'Pocket Hand Warmers',
    icon: '🧤',
    category: 'medical',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    warmthBonus: 25
  },
  sunscreen: {
    id: 'sunscreen',
    name: 'SPF 100 Sunscreen',
    icon: '🧴',
    category: 'medical',
    rarity: 'C',
    maxStack: 5,
    weight: 0.2,
    specialEffect: 'Prevents desert heat meter drain'
  },
  gas_mask: {
    id: 'gas_mask',
    name: 'Gas Mask',
    icon: '🎭',
    category: 'armor',
    rarity: 'R',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Immunity to bloater balloon clouds and bog fumes'
  },
  sleeping_bag: {
    id: 'sleeping_bag',
    name: 'Cozy Sleeping Bag',
    icon: '🛏️',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Sleep in safe zones to skip night and restore sanity'
  },
  tent_kit: {
    id: 'tent_kit',
    name: 'Pop-Up Tent Kit',
    icon: '⛺',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 3.0,
    specialEffect: 'Deploys a portable shelter protecting from cold and rain'
  },

  // --- TOOLS & UTILITY ---
  axe: {
    id: 'axe',
    name: 'Flint Axe',
    icon: '🪓',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    durability: 50,
    description: 'Fells trees and chops wooden obstacles.'
  },
  pickaxe: {
    id: 'pickaxe',
    name: 'Stone Pickaxe',
    icon: '⛏️',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    durability: 50,
    description: 'Mines boulders, gold veins, and glowing crystals.'
  },
  shovel: {
    id: 'shovel',
    name: 'Sturdy Shovel',
    icon: '🥄',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    durability: 60,
    description: 'Digs up carrots, graves, and X marks the spot treasures.'
  },
  wrench: {
    id: 'wrench',
    name: 'Adjustable Wrench',
    icon: '🔧',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    durability: 100,
    description: 'Speeds up car part repair.'
  },
  ratchet: {
    id: 'ratchet',
    name: 'Mechanic Ratchet',
    icon: '🔩',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    durability: 120,
    description: 'Swiftly bolts engine parts in place.'
  },
  jack: {
    id: 'jack',
    name: 'Hydraulic Car Jack',
    icon: '🪜',
    category: 'tool',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    durability: 80,
    description: 'Lifts vehicles to fit or replace tires.'
  },
  welding_kit: {
    id: 'welding_kit',
    name: 'Portable Welding Kit',
    icon: '🔥',
    category: 'tool',
    rarity: 'U',
    maxStack: 1,
    weight: 3.0,
    durability: 50,
    description: 'Restores vehicle chassis and engine blocks past 80%.'
  },
  lockpick: {
    id: 'lockpick',
    name: 'Hairpin Lockpick',
    icon: '📎',
    category: 'tool',
    rarity: 'C',
    maxStack: 5,
    weight: 0.2,
    durability: 5,
    description: 'Quietly picks locked doors and containers.'
  },
  bolt_cutters: {
    id: 'bolt_cutters',
    name: 'Heavy Bolt Cutters',
    icon: '✂️',
    category: 'tool',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    durability: 40,
    description: 'Snips padlocks and chain-link fences.'
  },
  flashlight: {
    id: 'flashlight',
    name: 'Handy Flashlight',
    icon: '🔦',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    durability: 300,
    description: 'Projects a forward light cone in the dark.'
  },
  torch: {
    id: 'torch',
    name: 'Torch',
    icon: '🔥',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    durability: 100,
    description: 'Warm handheld light keeping night horrors away.'
  },
  lantern: {
    id: 'lantern',
    name: 'Brass Lantern',
    icon: '🏮',
    category: 'tool',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    durability: 200,
    description: 'Even 360-degree illumination that can be set down.'
  },
  horn: {
    id: 'horn',
    name: 'Mega Honker',
    icon: '📢',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    durability: 999,
    description: 'Blasts a comical honk that startles critters and pings bros.'
  },
  air_horn: {
    id: 'air_horn',
    name: 'Stadium Air Horn',
    icon: '📯',
    category: 'tool',
    rarity: 'U',
    maxStack: 1,
    weight: 0.5,
    durability: 50,
    noise: 5,
    specialEffect: 'Stuns nearby creatures, wakes sleepers, calls teammates'
  },
  rope: {
    id: 'rope',
    name: 'Coiled Rope',
    icon: '🪢',
    category: 'resource',
    rarity: 'C',
    maxStack: 10,
    weight: 0.5
  },
  duct_tape: {
    id: 'duct_tape',
    name: 'Roll of Duct Tape',
    icon: '🩹',
    category: 'tool',
    rarity: 'C',
    maxStack: 5,
    weight: 0.2,
    specialEffect: 'Quick patch for broken car parts, caps condition at 40%'
  },
  hose: {
    id: 'hose',
    name: 'Siphon Hose',
    icon: '🐍',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    description: 'Siphons fuel from abandoned car tanks.'
  },
  jerrycan: {
    id: 'jerrycan',
    name: 'Gasoline Jerrycan',
    icon: '⛽',
    category: 'tool',
    rarity: 'C',
    maxStack: 2,
    weight: 3.0,
    description: 'Holds 20 liters of vehicle fuel.'
  },
  fishing_rod: {
    id: 'fishing_rod',
    name: 'Bamboo Fishing Rod',
    icon: '🎣',
    category: 'tool',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    durability: 50
  },
  multi_tool: {
    id: 'multi_tool',
    name: 'Swiss Multi-Tool',
    icon: '🔪',
    category: 'tool',
    rarity: 'R',
    maxStack: 1,
    weight: 0.2,
    durability: 150,
    description: 'Works as screwdriver, knife, pliers, and bottle opener.'
  },
  grappling_hook_of_doubt: {
    id: 'grappling_hook_of_doubt',
    name: 'Grappling Hook of Doubt',
    icon: '🪝',
    category: 'movement',
    rarity: 'R',
    maxStack: 1,
    weight: 2.0,
    durability: 30,
    specialEffect: 'Zips player to rooftops (misses with a hammy plop 10% of the time)'
  },

  // --- MOVEMENT TOYS ---
  boing_boots: {
    id: 'boing_boots',
    name: 'Boing Boots',
    icon: '👢',
    category: 'movement',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Jump higher with bounce recoil and squeaks per step'
  },
  propeller_beanie: {
    id: 'propeller_beanie',
    name: 'Propeller Beanie',
    icon: '🧢',
    category: 'movement',
    rarity: 'U',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Glide through the air for 5 seconds'
  },
  umbrella_glider: {
    id: 'umbrella_glider',
    name: 'Umbrella Glider',
    icon: '☂️',
    category: 'movement',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Slow fall, blocks rain and falling frogs'
  },
  bean_burrito_booster: {
    id: 'bean_burrito_booster',
    name: 'Bean Burrito Booster',
    icon: '🌯',
    category: 'movement',
    rarity: 'R',
    maxStack: 5,
    weight: 0.5,
    specialEffect: 'Rocket dash leaving a green repulsive fart cloud'
  },
  skateboard: {
    id: 'skateboard',
    name: 'Kick Skateboard',
    icon: '🛹',
    category: 'movement',
    rarity: 'U',
    maxStack: 1,
    weight: 1.5,
    specialEffect: '+50% silent speed along paved roads'
  },
  pogo_stick: {
    id: 'pogo_stick',
    name: 'Pogo Stick',
    icon: '🦘',
    category: 'movement',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    specialEffect: 'Hop over low fences and road barriers'
  },

  // --- VEHICLE PARTS & ENGINE ITEMS ---
  battery: {
    id: 'battery',
    name: '12V Lead Battery',
    icon: '🔋',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 2,
    weight: 3.0,
    description: 'Powers car ignition and electrical circuits.'
  },
  spark_plugs: {
    id: 'spark_plugs',
    name: 'Copper Spark Plugs',
    icon: '⚡',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 8,
    weight: 0.2,
    description: 'Fires ignition cylinders.'
  },
  engine_block: {
    id: 'engine_block',
    name: 'Cast Iron Engine Block',
    icon: '🏎️',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    description: 'The heavy cast iron heart of any motor vehicle.'
  },
  ignition_key: {
    id: 'ignition_key',
    name: 'Car Ignition Key',
    icon: '🗝️',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 0.1,
    description: 'Starts a vehicle without needing hotwire tricks.'
  },
  tire_standard: {
    id: 'tire_standard',
    name: 'All-Terrain Rubber Tire',
    icon: '🛞',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 4,
    weight: 3.0,
    description: 'Standard road and off-road pneumatic rubber tire.'
  },
  radiator_coolant: {
    id: 'radiator_coolant',
    name: 'Radiator & Coolant Fluid',
    icon: '🧊',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 2,
    weight: 2.0,
    description: 'Prevents car engines from overheating and bursting into smoke.'
  },
  fan_belt: {
    id: 'fan_belt',
    name: 'Reinforced Rubber Fan Belt',
    icon: '🎗️',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 4,
    weight: 0.2,
    description: 'Connects the crankshaft to alternator and cooling fan.'
  },
  headlights: {
    id: 'headlights',
    name: 'Halogen Headlights',
    icon: '💡',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 2,
    weight: 0.5,
    description: 'Illuminates the road ahead in pitch dark and dense fog.'
  },
  bumper_steel: {
    id: 'bumper_steel',
    name: 'Heavy Steel Bumper',
    icon: '🛡️',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 5.0,
    description: 'Reinforced bull-bar that plows through zombie hordes.'
  },
  roof_rack: {
    id: 'roof_rack',
    name: 'Heavy Luggage Roof Rack',
    icon: '🧳',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 4.0,
    description: 'Expands vehicle storage by +12 slots.'
  },
  car_radio: {
    id: 'car_radio',
    name: 'AM/FM Post-Apoc Radio',
    icon: '📻',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    description: 'Tunes in to DJ Midnight and keeps driver sanity elevated.'
  },
  fuzzy_dice: {
    id: 'fuzzy_dice',
    name: 'Neon Fuzzy Dice',
    icon: '🎲',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 0.1,
    description: 'Hangs from rearview mirror. Increases luck on container loot rolls.'
  },
  seat_covers: {
    id: 'seat_covers',
    name: 'Velvet Leopard Seat Covers',
    icon: '🛋️',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    description: 'Soft and cozy. Continuously restores rider sanity.'
  },
  windshield: {
    id: 'windshield',
    name: 'Safety Glass Windshield',
    icon: '🪟',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 3.0,
    description: 'Protects passengers from weather and flying obstacles.'
  },
  pancake_stack_tire: {
    id: 'pancake_stack_tire',
    name: 'Pancake Stack Tire',
    icon: '🥞',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 2,
    weight: 1.5,
    description: 'Jank fix: bouncy boing ride over bumps at lower speed.'
  },
  sock_fan_belt: {
    id: 'sock_fan_belt',
    name: 'Stink Sock Fan Belt',
    icon: '🧦',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 4,
    weight: 0.1,
    description: 'Jank fix: works at 50% efficiency and puffs sock steam.'
  },
  bucket_headlight: {
    id: 'bucket_headlight',
    name: 'Bucket Headlight',
    icon: '🪣',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 2,
    weight: 1.0,
    description: 'Jank fix: swinging cone of light.'
  },
  rubber_chicken_horn: {
    id: 'rubber_chicken_horn',
    name: 'Rubber Chicken Horn',
    icon: '🐔',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    description: 'Jank fix: squeaks and scares small enemies.'
  },
  bean_burrito_fuel: {
    id: 'bean_burrito_fuel',
    name: 'Bean Burrito Jank Fuel',
    icon: '🌯',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 5,
    weight: 0.5,
    description: 'Jank fix: explosive speed boost that leaves toxic fart clouds.'
  },
  bucket_of_water_radiator: {
    id: 'bucket_of_water_radiator',
    name: 'Bucket of Pond Water Radiator',
    icon: '🪣',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 2.0,
    description: 'Jank fix: works until water boils dry.'
  },
  googly_eyes_dashboard: {
    id: 'googly_eyes_dashboard',
    name: 'Googly Eyes Dashboard',
    icon: '👀',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 0.1,
    description: 'Jank fix: tracks bumps and restores passenger sanity.'
  },
  cheese_wheel_tire: {
    id: 'cheese_wheel_tire',
    name: 'Aged Gouda Wheel Tire',
    icon: '🧀',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 2,
    weight: 2.5,
    description: 'Jank fix: rolls oddly and relentlessly attracts the goose.'
  },
  trailer_hitch: {
    id: 'trailer_hitch',
    name: 'Trailer Hitch',
    icon: '🪝',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 3.0,
    description: 'Couples trailers behind vehicles.'
  },
  winch_cable: {
    id: 'winch_cable',
    name: 'Steel Winch Cable',
    icon: '⛓️',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 2.5,
    description: 'Pulls wrecks and hauls vehicles.'
  },
  hydraulic_pump: {
    id: 'hydraulic_pump',
    name: 'Hydraulic Pump',
    icon: '⚙️',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 4.0,
    description: 'Powers ladders and dozer blades.'
  },
  siren_box: {
    id: 'siren_box',
    name: 'Emergency Siren Box',
    icon: '🚨',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 1.5,
    description: 'Loud siren that lures hordes.'
  },
  rotor_blades: {
    id: 'rotor_blades',
    name: 'Helicopter Rotor Blades',
    icon: '🚁',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 1,
    weight: 5.0,
    description: 'Main lift blades for the whirlybird.'
  },
  tail_rotor: {
    id: 'tail_rotor',
    name: 'Tail Rotor',
    icon: '🌀',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 1,
    weight: 2.0,
    description: 'Counter-torque tail rotor for the whirlybird.'
  },
  turbine_engine: {
    id: 'turbine_engine',
    name: 'Turboshaft Engine',
    icon: '🚀',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 1,
    weight: 8.0,
    description: 'High-RPM aviation turbine engine.'
  },
  landing_skids: {
    id: 'landing_skids',
    name: 'Helicopter Landing Skids',
    icon: '🪜',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 4.0,
    description: 'Shock-absorbing skids for the whirlybird.'
  },
  flight_gizmo: {
    id: 'flight_gizmo',
    name: 'Flight Gizmo',
    icon: '🧭',
    category: 'vehicle_part',
    rarity: 'R',
    maxStack: 1,
    weight: 1.0,
    description: 'Autopilot gizmo enabling hover assist.'
  },
  outboard_motor: {
    id: 'outboard_motor',
    name: 'Marine Outboard Motor',
    icon: '🚤',
    category: 'vehicle_part',
    rarity: 'U',
    maxStack: 1,
    weight: 6.0,
    description: 'Thrust engine for boats and speedboats.'
  },
  propeller: {
    id: 'propeller',
    name: 'Marine Propeller',
    icon: '🌀',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 2,
    weight: 2.0,
    description: 'Bronze underwater propulsion screw.'
  },
  hull_patch_kit: {
    id: 'hull_patch_kit',
    name: 'Hull Patch Kit',
    icon: '🩹',
    category: 'tool',
    rarity: 'C',
    maxStack: 5,
    weight: 0.5,
    description: 'Patches leaks in boat hulls.'
  },
  aviation_fuel: {
    id: 'aviation_fuel',
    name: 'High-Octane Aviation Fuel',
    icon: '⛽',
    category: 'resource',
    rarity: 'R',
    maxStack: 3,
    weight: 3.0,
    description: 'Rare kerosene fuel powering turbine engines.'
  },
  propane_tank: {
    id: 'propane_tank',
    name: 'Liquid Propane Tank',
    icon: '🛢️',
    category: 'resource',
    rarity: 'U',
    maxStack: 2,
    weight: 3.0,
    description: 'Powers hot-air balloons and camp stoves.'
  },
  track_gap_plates: {
    id: 'track_gap_plates',
    name: 'Railway Track Gap Plates',
    icon: '🛤️',
    category: 'resource',
    rarity: 'U',
    maxStack: 5,
    weight: 4.0,
    description: 'Patches missing rails on the train loop.'
  },
  ceiling_fan_rotor: {
    id: 'ceiling_fan_rotor',
    name: 'Ceiling Fan Jank Rotor',
    icon: '🪭',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 2.0,
    description: 'Jank fix: lifts whirlybird at 40% speed with frantic wobble.'
  },
  pool_noodle_oars: {
    id: 'pool_noodle_oars',
    name: 'Pool Noodle Jank Oars',
    icon: '🫧',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    description: 'Jank fix: bends and paddles boat slowly.'
  },
  umbrella_sail: {
    id: 'umbrella_sail',
    name: 'Beach Umbrella Sail',
    icon: '⛱️',
    category: 'vehicle_part',
    rarity: 'C',
    maxStack: 1,
    weight: 1.0,
    description: 'Jank fix: catches ocean breeze.'
  },

  // --- CRAFTING MATERIALS ---
  scrap_metal: {
    id: 'scrap_metal',
    name: 'Scrap Metal',
    icon: '🔩',
    category: 'resource',
    rarity: 'C',
    maxStack: 30,
    weight: 0.5
  },
  nails: {
    id: 'nails',
    name: 'Box of Nails',
    icon: '📍',
    category: 'resource',
    rarity: 'C',
    maxStack: 50,
    weight: 0.2
  },
  cloth_scrap: {
    id: 'cloth_scrap',
    name: 'Cloth Scrap',
    icon: '🧶',
    category: 'resource',
    rarity: 'C',
    maxStack: 30,
    weight: 0.2
  },
  glue: {
    id: 'glue',
    name: 'Super Glue Tube',
    icon: '🧴',
    category: 'resource',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2
  },
  wire: {
    id: 'wire',
    name: 'Copper Wire Spool',
    icon: '🪡',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2
  },
  spring: {
    id: 'spring',
    name: 'Mattress Spring',
    icon: '🌀',
    category: 'resource',
    rarity: 'C',
    maxStack: 15,
    weight: 0.5
  },
  gears: {
    id: 'gears',
    name: 'Brass Gears',
    icon: '⚙️',
    category: 'resource',
    rarity: 'U',
    maxStack: 20,
    weight: 0.5
  },
  empty_bottle: {
    id: 'empty_bottle',
    name: 'Empty Glass Bottle',
    icon: '🍾',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 0.2
  },
  battery_cell: {
    id: 'battery_cell',
    name: 'Battery Cell',
    icon: '🔋',
    category: 'resource',
    rarity: 'U',
    maxStack: 20,
    weight: 0.2
  },
  gunk_goo: {
    id: 'gunk_goo',
    name: 'Zombie Gunk Goo',
    icon: '🧪',
    category: 'resource',
    rarity: 'C',
    maxStack: 30,
    weight: 0.2,
    description: 'Slime left by defeated cartoon zombies. Used for jank fuel.'
  },
  plank: {
    id: 'plank',
    name: 'Milled Wooden Plank',
    icon: '🪵',
    category: 'resource',
    rarity: 'C',
    maxStack: 20,
    weight: 1.0
  },
  plank_bundle: {
    id: 'plank_bundle',
    name: 'Bundle of Boardwalk Planks',
    icon: '📦',
    category: 'resource',
    rarity: 'C',
    maxStack: 5,
    weight: 3.0,
    description: 'Used to repair missing planks on bog causeways.'
  },

  // --- AMMO & CURRENCY ---
  shiny_buttons: {
    id: 'shiny_buttons',
    name: 'Shiny Buttons',
    icon: '🔘',
    category: 'currency',
    rarity: 'C',
    maxStack: 999,
    weight: 0.0,
    description: 'Universal post-apocalyptic barter currency.'
  },
  pebbles: {
    id: 'pebbles',
    name: 'Smooth Pebbles',
    icon: '🪨',
    category: 'ammo',
    rarity: 'C',
    maxStack: 50,
    weight: 0.2
  },
  foam_darts: {
    id: 'foam_darts',
    name: 'Foam Darts',
    icon: '🎯',
    category: 'ammo',
    rarity: 'C',
    maxStack: 60,
    weight: 0.1
  },
  corks: {
    id: 'corks',
    name: 'Bottle Corks',
    icon: '🍾',
    category: 'ammo',
    rarity: 'C',
    maxStack: 40,
    weight: 0.1
  },
  potatoes: {
    id: 'potatoes',
    name: 'Potatoes',
    icon: '🥔',
    category: 'ammo',
    rarity: 'C',
    maxStack: 20,
    weight: 0.5,
    hungerBonus: 10
  },
  bottle_rockets: {
    id: 'bottle_rockets',
    name: 'Bottle Rockets',
    icon: '🚀',
    category: 'ammo',
    rarity: 'U',
    maxStack: 20,
    weight: 0.2
  },
  flare_cartridges: {
    id: 'flare_cartridges',
    name: 'Flare Cartridges',
    icon: '🚨',
    category: 'ammo',
    rarity: 'U',
    maxStack: 10,
    weight: 0.2
  },
  batteries_aa: {
    id: 'batteries_aa',
    name: 'AA Batteries',
    icon: '🔋',
    category: 'ammo',
    rarity: 'C',
    maxStack: 20,
    weight: 0.1
  },

  // --- JUNK & GOOFY RELICS ---
  mismatched_sock: {
    id: 'mismatched_sock',
    name: 'Mismatched Sock',
    icon: '🧦',
    category: 'junk',
    rarity: 'C',
    maxStack: 20,
    weight: 0.1,
    description: 'Fluttered from a sock tree. Craft into warmth items or jank fan belts.'
  },
  rubber_duck: {
    id: 'rubber_duck',
    name: 'Rubber Duck',
    icon: '🦆',
    category: 'junk',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2,
    description: 'Squeaks when pressed.'
  },
  whoopee_cushion: {
    id: 'whoopee_cushion',
    name: 'Whoopee Cushion',
    icon: '💨',
    category: 'junk',
    rarity: 'C',
    maxStack: 10,
    weight: 0.2
  },
  fake_mustache: {
    id: 'fake_mustache',
    name: 'Fake Mustache',
    icon: '🥸',
    category: 'cosmetic',
    rarity: 'U',
    maxStack: 1,
    weight: 0.1,
    specialEffect: 'Creatures ignore you for 10s (except the goose)'
  },
  lava_lamp: {
    id: 'lava_lamp',
    name: 'Hypnotic Lava Lamp',
    icon: '🪔',
    category: 'junk',
    rarity: 'U',
    maxStack: 1,
    weight: 1.0,
    specialEffect: 'Gentle light and soothing sanity aura'
  },
  garden_gnome: {
    id: 'garden_gnome',
    name: 'Garden Gnome',
    icon: '🧙‍♂️',
    category: 'junk',
    rarity: 'U',
    maxStack: 1,
    weight: 2.0,
    description: 'Turns its head when you look away. Find all 10 for a reward!'
  },
  broken_toaster: {
    id: 'broken_toaster',
    name: 'Broken Toaster',
    icon: '🍞',
    category: 'junk',
    rarity: 'C',
    maxStack: 2,
    weight: 1.5,
    description: 'Scraps for springs and heating coils.'
  },
  giant_novelty_check: {
    id: 'giant_novelty_check',
    name: 'Giant Novelty Check for $1,000,000',
    icon: '📜',
    category: 'junk',
    rarity: 'X',
    maxStack: 1,
    weight: 1.0,
    description: 'Worthless in the apocalypse, but amazing bragging rights.'
  },
  golden_toilet_seat: {
    id: 'golden_toilet_seat',
    name: 'Golden Toilet Seat',
    icon: '🚽',
    category: 'junk',
    rarity: 'X',
    maxStack: 1,
    weight: 3.0,
    description: 'The pinnacle of luxury loot. Restores party sanity on sight.'
  },

  // --- KEYS & QUEST ITEMS ---
  house_key: {
    id: 'house_key',
    name: 'House Key',
    icon: '🔑',
    category: 'quest',
    rarity: 'C',
    maxStack: 1,
    weight: 0.1
  },
  car_key: {
    id: 'car_key',
    name: 'Car Ignition Key',
    icon: '🗝️',
    category: 'quest',
    rarity: 'C',
    maxStack: 1,
    weight: 0.1
  },
  keycard_red: {
    id: 'keycard_red',
    name: 'Red Security Keycard',
    icon: '💳',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 0.1
  },
  keycard_blue: {
    id: 'keycard_blue',
    name: 'Blue Security Keycard',
    icon: '💳',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 0.1
  },
  keycard_green: {
    id: 'keycard_green',
    name: 'Green Security Keycard',
    icon: '💳',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 0.1
  },
  rescue_radio_part_1: {
    id: 'rescue_radio_part_1',
    name: 'Rescue Radio Frequency Oscillator',
    icon: '📻',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 2.0,
    description: 'Part 1/3 needed to restore the broadcast tower at Ridge.'
  },
  rescue_radio_part_2: {
    id: 'rescue_radio_part_2',
    name: 'Rescue Radio Vacuum Tube',
    icon: '💡',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 1.0,
    description: 'Part 2/3 needed to restore the broadcast tower at Ridge.'
  },
  rescue_radio_part_3: {
    id: 'rescue_radio_part_3',
    name: 'Rescue Radio Copper Antenna',
    icon: '📡',
    category: 'quest',
    rarity: 'R',
    maxStack: 1,
    weight: 3.0,
    description: 'Part 3/3 needed to restore the broadcast tower at Ridge.'
  },
  boat_key: {
    id: 'boat_key',
    name: 'Evac Dock Boat Key',
    icon: '🔑',
    category: 'quest',
    rarity: 'X',
    maxStack: 1,
    weight: 0.1,
    description: 'Unlocks the rescue engine at Splash Cove evac dock.'
  },
  map_fragment: {
    id: 'map_fragment',
    name: 'Treasure Map Fragment',
    icon: '🗺️',
    category: 'quest',
    rarity: 'U',
    maxStack: 5,
    weight: 0.1,
    description: 'Reveals a buried supply crate on the map.'
  },
  evac_beacon: {
    id: 'evac_beacon',
    name: 'Evacuation Distress Beacon',
    icon: '🚨',
    category: 'quest',
    rarity: 'X',
    maxStack: 1,
    weight: 3.0,
    description: 'Calls the rescue vessel once car engine is powered at the dock.'
  },

  // --- GOOFY COSMETICS ---
  traffic_cone_hat: {
    id: 'traffic_cone_hat',
    name: 'Traffic Cone Hat',
    icon: '🦺',
    category: 'cosmetic',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    specialEffect: 'Knocked flying with a comical thwack when hit'
  },
  cheese_hat: {
    id: 'cheese_hat',
    name: 'Wedge of Cheese Hat',
    icon: '🧀',
    category: 'cosmetic',
    rarity: 'C',
    maxStack: 1,
    weight: 0.5,
    specialEffect: 'Constantly attracts the goose'
  },
  flower_crown: {
    id: 'flower_crown',
    name: 'Meadow Flower Crown',
    icon: '👑',
    category: 'cosmetic',
    rarity: 'C',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Spills flower petals while dancing'
  },
  tiny_top_hat: {
    id: 'tiny_top_hat',
    name: 'Tiny Dapper Top Hat',
    icon: '🎩',
    category: 'cosmetic',
    rarity: 'U',
    maxStack: 1,
    weight: 0.2,
    specialEffect: 'Tips with dapper flair whenever you honk (H)'
  },
  shark_slippers: {
    id: 'shark_slippers',
    name: 'Shark Plush Slippers',
    icon: '🦈',
    category: 'cosmetic',
    rarity: 'U',
    maxStack: 1,
    weight: 0.5,
    specialEffect: 'Jaws flap with every footstep'
  },
  cape_of_dramatic_wind: {
    id: 'cape_of_dramatic_wind',
    name: 'Cape of Dramatic Wind',
    icon: '🦸',
    category: 'cosmetic',
    rarity: 'R',
    maxStack: 1,
    weight: 0.5,
    specialEffect: 'Billows majestically at high sprint speeds'
  }
};
