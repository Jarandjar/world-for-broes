export type PartSlotType =
  | 'battery'
  | 'spark_plugs'
  | 'engine_block'
  | 'ignition_key'
  | 'tire_fl'
  | 'tire_fr'
  | 'tire_rl'
  | 'tire_rr'
  | 'radiator_coolant'
  | 'fan_belt'
  | 'headlights'
  | 'horn'
  | 'windshield'
  | 'doors'
  | 'bumper'
  | 'roof_rack'
  | 'radio'
  | 'trunk_lock'
  | 'interior_mod'
  | 'exterior_mod'
  | 'rotor'
  | 'tail_rotor'
  | 'skids'
  | 'avionics'
  | 'propeller'
  | 'hull'
  | 'hitch'
  | 'winch'
  | 'hydraulics'
  | 'drive_chain'
  | 'balloon'
  | 'rail_gear';

export type PartQuality = 'junk' | 'standard' | 'shiny';

export interface CarPartDef {
  id: string;
  name: string;
  icon: string;
  slot: PartSlotType;
  isRequired: boolean;
  maxCondition: number; // default 100
  minOperableCondition: number; // default 20%
  baseRepairTimeSeconds: number; // e.g. 3.0s
  repairTools: string[]; // ['wrench', 'ratchet', etc.]
  isJankFix?: boolean;
  quirkDescription?: string;
  speedModifier?: number;
  noiseModifier?: number;
  sanityRateModifier?: number;
}

export const PART_REGISTRY: Record<string, CarPartDef> = {
  // --- REQUIRED ENGINE & DRIVETRAIN PARTS ---
  battery: {
    id: 'battery',
    name: '12V Lead Battery',
    icon: '🔋',
    slot: 'battery',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.5,
    repairTools: ['wrench', 'jump_leads', 'duct_tape']
  },
  spark_plugs: {
    id: 'spark_plugs',
    name: 'Copper Spark Plugs',
    icon: '⚡',
    slot: 'spark_plugs',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 2.5,
    repairTools: ['ratchet', 'wrench']
  },
  engine_block: {
    id: 'engine_block',
    name: 'Cast Iron Engine Block',
    icon: '🏎️',
    slot: 'engine_block',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 6.0,
    repairTools: ['welding_kit', 'wrench', 'ratchet']
  },
  ignition_key: {
    id: 'ignition_key',
    name: 'Vehicle Ignition Key',
    icon: '🗝️',
    slot: 'ignition_key',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 1.0,
    repairTools: ['lockpick']
  },
  tire_standard: {
    id: 'tire_standard',
    name: 'All-Terrain Rubber Tire',
    icon: '🛞',
    slot: 'tire_fl', // or fr, rl, rr
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 4.0,
    repairTools: ['jack', 'wrench', 'tire_patch']
  },
  radiator_coolant: {
    id: 'radiator_coolant',
    name: 'Radiator & Coolant Fluid',
    icon: '🧊',
    slot: 'radiator_coolant',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['wrench', 'duct_tape']
  },
  fan_belt: {
    id: 'fan_belt',
    name: 'Reinforced Rubber Fan Belt',
    icon: '🎗️',
    slot: 'fan_belt',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 2.5,
    repairTools: ['wrench']
  },

  // --- OPTIONAL ENHANCEMENT PARTS ---
  headlights: {
    id: 'headlights',
    name: 'Halogen Headlights',
    icon: '💡',
    slot: 'headlights',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 2.0,
    repairTools: ['duct_tape', 'wrench'],
    quirkDescription: 'Wide bright light cone piercing darkness and fog'
  },
  horn: {
    id: 'horn',
    name: 'Dual Tone Horn',
    icon: '📢',
    slot: 'horn',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 1.5,
    repairTools: ['wrench']
  },
  windshield: {
    id: 'windshield',
    name: 'Safety Glass Windshield',
    icon: '🪟',
    slot: 'windshield',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['duct_tape'],
    quirkDescription: 'Protects passengers from flying frogs and vomit clouds'
  },
  bumper_steel: {
    id: 'bumper_steel',
    name: 'Heavy Steel Bumper',
    icon: '🛡️',
    slot: 'bumper',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 4.0,
    repairTools: ['welding_kit', 'wrench'],
    quirkDescription: 'Smashes zombies without damaging engine block'
  },
  roof_rack: {
    id: 'roof_rack',
    name: 'Heavy Luggage Roof Rack',
    icon: '🧳',
    slot: 'roof_rack',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['wrench'],
    quirkDescription: 'Adds +12 extra cargo slots to vehicle'
  },
  car_radio: {
    id: 'car_radio',
    name: 'AM/FM Post-Apoc Radio',
    icon: '📻',
    slot: 'radio',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 2.0,
    repairTools: ['wire', 'duct_tape'],
    quirkDescription: 'Broadcasts DJ Midnight tunes, horde warnings, and restores sanity',
    sanityRateModifier: 1.5
  },
  fuzzy_dice: {
    id: 'fuzzy_dice',
    name: 'Neon Fuzzy Dice',
    icon: '🎲',
    slot: 'interior_mod',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 1.0,
    repairTools: [],
    quirkDescription: 'Increases luck on container loot rolls by 15%'
  },
  seat_covers: {
    id: 'seat_covers',
    name: 'Velvet Leopard Seat Covers',
    icon: '🛋️',
    slot: 'interior_mod',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 1.5,
    repairTools: [],
    quirkDescription: 'Cozy seats continuously restore sanity while driving',
    sanityRateModifier: 1.2
  },

  // --- GOOFY JANK FIXES ---
  bucket_headlight: {
    id: 'bucket_headlight',
    name: 'Bucket Headlight',
    icon: '🪣',
    slot: 'headlights',
    isRequired: false,
    maxCondition: 60,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 1.5,
    repairTools: ['duct_tape'],
    isJankFix: true,
    quirkDescription: 'Narrow cone of light swinging loosely as car turns'
  },
  rubber_chicken_horn: {
    id: 'rubber_chicken_horn',
    name: 'Rubber Chicken Horn',
    icon: '🐔',
    slot: 'horn',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 1.0,
    repairTools: ['duct_tape'],
    isJankFix: true,
    quirkDescription: 'Honks with a squeak that startles small critters and zombies'
  },
  pancake_stack_tire: {
    id: 'pancake_stack_tire',
    name: 'Pancake Stack Tire',
    icon: '🥞',
    slot: 'tire_fl',
    isRequired: true,
    maxCondition: 50,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 2.0,
    repairTools: ['syrup', 'duct_tape'],
    isJankFix: true,
    speedModifier: 0.75,
    quirkDescription: 'Comically bouncy ride that boings over bumps; reduced speed'
  },
  sock_fan_belt: {
    id: 'sock_fan_belt',
    name: 'Stink Sock Fan Belt',
    icon: '🧦',
    slot: 'fan_belt',
    isRequired: true,
    maxCondition: 50,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 1.5,
    repairTools: ['duct_tape'],
    isJankFix: true,
    quirkDescription: 'Operates at 50% efficiency and puffs smelly steam'
  },
  bean_burrito_fuel: {
    id: 'bean_burrito_fuel',
    name: 'Bean Burrito Jank Fuel',
    icon: '🌯',
    slot: 'radiator_coolant',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 1.0,
    repairTools: [],
    isJankFix: true,
    speedModifier: 1.3,
    quirkDescription: 'Violent speed boost that leaves toxic green fart clouds'
  },
  bucket_of_water_radiator: {
    id: 'bucket_of_water_radiator',
    name: 'Bucket of Pond Water Radiator',
    icon: '🪣',
    slot: 'radiator_coolant',
    isRequired: true,
    maxCondition: 40,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 1.5,
    repairTools: ['duct_tape'],
    isJankFix: true,
    quirkDescription: 'Cools engine temporarily until the water boils dry'
  },
  googly_eyes_dashboard: {
    id: 'googly_eyes_dashboard',
    name: 'Googly Eyes Dashboard',
    icon: '👀',
    slot: 'interior_mod',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 1,
    baseRepairTimeSeconds: 0.5,
    repairTools: [],
    isJankFix: true,
    sanityRateModifier: 1.4,
    quirkDescription: 'Wobbly pupils track bumps and keep passengers cheerful'
  },
  cheese_wheel_tire: {
    id: 'cheese_wheel_tire',
    name: 'Aged Gouda Wheel Tire',
    icon: '🧀',
    slot: 'tire_fl',
    isRequired: true,
    maxCondition: 60,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 2.0,
    repairTools: ['wrench'],
    isJankFix: true,
    speedModifier: 0.85,
    quirkDescription: 'Rolls irregularly and relentlessly attracts the honk goose'
  },

  // --- SECTION 46: EXPANDED FLEET PARTS ---
  trailer_hitch: {
    id: 'trailer_hitch',
    name: 'Class-IV Steel Trailer Hitch',
    icon: '🪝',
    slot: 'hitch',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['wrench', 'welding_kit'],
    quirkDescription: 'Couples heavy cargo, camper, and boat trailers to vehicles'
  },
  winch_cable: {
    id: 'winch_cable',
    name: 'Braided Steel Winch Cable',
    icon: '⛓️',
    slot: 'winch',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 2.5,
    repairTools: ['wrench'],
    quirkDescription: 'Tows wrecks and hoists cargo slings'
  },
  hydraulic_pump: {
    id: 'hydraulic_pump',
    name: 'High-Pressure Hydraulic Pump',
    icon: '⚙️',
    slot: 'hydraulics',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 4.0,
    repairTools: ['wrench', 'ratchet'],
    quirkDescription: 'Powers bulldozer blades and fire truck ladders'
  },
  siren_box: {
    id: 'siren_box',
    name: '120dB Emergency Siren Box',
    icon: '🚨',
    slot: 'exterior_mod',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 2.0,
    repairTools: ['wire', 'duct_tape'],
    quirkDescription: 'Tactical crowd-puller that lures hordes away from teammates'
  },

  // --- AIRCRAFT PARTS ---
  rotor_blades: {
    id: 'rotor_blades',
    name: 'Carbon Composite Main Rotor Blades',
    icon: '🚁',
    slot: 'rotor',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 5.0,
    repairTools: ['ratchet', 'wrench'],
    quirkDescription: 'Generates vertical lift; balancing mini-game required'
  },
  tail_rotor: {
    id: 'tail_rotor',
    name: 'Counter-Torque Tail Rotor',
    icon: '🌀',
    slot: 'tail_rotor',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.5,
    repairTools: ['wrench'],
    quirkDescription: 'Counteracts main rotor torque to prevent spinning'
  },
  turbine_engine: {
    id: 'turbine_engine',
    name: 'Turboshaft Helicopter Engine',
    icon: '🚀',
    slot: 'engine_block',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 25,
    baseRepairTimeSeconds: 8.0,
    repairTools: ['welding_kit', 'ratchet'],
    quirkDescription: 'High-RPM turbine requiring 4s spool-up before flight'
  },
  landing_skids: {
    id: 'landing_skids',
    name: 'Tubular Steel Landing Skids',
    icon: '🪜',
    slot: 'skids',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['welding_kit', 'wrench'],
    quirkDescription: 'Absorbs touchdown shock; zombies can grab low skids'
  },
  flight_gizmo: {
    id: 'flight_gizmo',
    name: 'Goofy Gyroscopic Flight Gizmo',
    icon: '🧭',
    slot: 'avionics',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['wire', 'duct_tape'],
    quirkDescription: 'Maintains auto-hover when pilot lifts thumbs from stick'
  },

  // --- BOAT & WATER PARTS ---
  outboard_motor: {
    id: 'outboard_motor',
    name: 'V6 Marine Outboard Motor',
    icon: '🚤',
    slot: 'engine_block',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 5.0,
    repairTools: ['wrench', 'ratchet'],
    quirkDescription: 'Provides marine thrust and steering'
  },
  propeller: {
    id: 'propeller',
    name: 'Bronze Marine Propeller',
    icon: '🌀',
    slot: 'propeller',
    isRequired: true,
    maxCondition: 100,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 2.5,
    repairTools: ['wrench'],
    quirkDescription: 'Pushes water; shears on shallow underwater rocks'
  },
  hull_patch_kit: {
    id: 'hull_patch_kit',
    name: 'Fiberglass Marine Hull Patch Kit',
    icon: '🩹',
    slot: 'hull',
    isRequired: false,
    maxCondition: 100,
    minOperableCondition: 20,
    baseRepairTimeSeconds: 3.0,
    repairTools: ['duct_tape'],
    quirkDescription: 'Quickly seals hull leaks during emergency patching mini-game'
  },

  // --- EXPANDED JANK FIXES ---
  ceiling_fan_rotor: {
    id: 'ceiling_fan_rotor',
    name: 'Ceiling Fan Jank Rotor',
    icon: '🪭',
    slot: 'rotor',
    isRequired: true,
    maxCondition: 45,
    minOperableCondition: 15,
    baseRepairTimeSeconds: 2.5,
    repairTools: ['duct_tape'],
    isJankFix: true,
    speedModifier: 0.60,
    quirkDescription: 'Works at 40% lift and wobbles hilariously in the breeze'
  },
  pool_noodle_oars: {
    id: 'pool_noodle_oars',
    name: 'Pool Noodle Jank Oars',
    icon: '🫧',
    slot: 'propeller',
    isRequired: true,
    maxCondition: 50,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 1.0,
    repairTools: ['duct_tape'],
    isJankFix: true,
    speedModifier: 0.50,
    quirkDescription: 'Flaccid foam paddles that splash noisily'
  },
  umbrella_sail: {
    id: 'umbrella_sail',
    name: 'Windy Beach Umbrella Sail',
    icon: '⛱️',
    slot: 'propeller',
    isRequired: false,
    maxCondition: 60,
    minOperableCondition: 10,
    baseRepairTimeSeconds: 1.5,
    repairTools: ['rope'],
    isJankFix: true,
    speedModifier: 0.70,
    quirkDescription: 'Catches tailwinds for gentle cruising across calm water'
  }
};
