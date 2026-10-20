import type { GunAttachmentDef } from './schema';

export const ATTACHMENTS: GunAttachmentDef[] = [
  // ==========================================
  // OPTICS & SIGHTS
  // ==========================================
  {
    id: 'optic_rangefinder',
    name: 'Tactical Rangefinder HUD',
    slot: 'optic',
    description: 'Digital HUD projection displaying target distance in meters, bullet travel time, and real-time target HP gauges.',
    rangeMod: 14,
    spreadMod: 0.75,
    visualInfo: {
      rangefinder: true,
      targetHealthBar: true,
    },
    recipe: { scrap_metal: 4, copper_wire: 3, glass: 2 },
  },
  {
    id: 'optic_camcorder',
    name: 'Taped Handycam CRT Night Sight',
    slot: 'optic',
    description: 'Vintage camcorder duct-taped to the receiver. Highlights approaching threats with tactical brackets and luminous night reticle.',
    rangeMod: 10,
    visualInfo: {
      threatScanner: true,
      nightVisionReticle: true,
      rangefinder: true,
    },
    recipe: { scrap_metal: 5, copper_wire: 4, battery: 1 },
  },
  {
    id: 'optic_googly',
    name: 'Googly Eye Smart Reticle',
    slot: 'optic',
    description: 'Bouncing googly eyes with quantum physics wiring that projects a continuous ballistic trajectory arc to bullet impact.',
    spreadMod: 0.85,
    visualInfo: {
      trajectoryArc: true,
      targetHealthBar: true,
    },
    recipe: { cloth: 2, glue: 1 },
  },
  {
    id: 'optic_magnifier',
    name: 'Taped Glass Magnifier',
    slot: 'optic',
    description: 'A pocket magnifying glass bound with black electrical tape. Tightens precision spread and extends effective range.',
    rangeMod: 16,
    spreadMod: 0.68,
    recipe: { glass: 2, duct_tape: 1 },
  },
  {
    id: 'optic_toilet_roll',
    name: 'Cardboard TP Tube Red-Dot',
    slot: 'optic',
    description: 'Cardboard tube with crossed fishing line inside. Simple, goofy, but gives quick target acquisition.',
    spreadMod: 0.9,
    recoilMod: 0.85,
    recipe: { cloth: 1 },
  },

  // ==========================================
  // MUZZLES & BARRELS
  // ==========================================
  {
    id: 'muzzle_potato_silencer',
    name: 'Russet Potato Silencer',
    slot: 'muzzle',
    description: 'Fresh organic spud jammed onto the muzzle. Dampens gunshot crack into a muffled "thwip" by -82% noise.',
    noiseMod: 0.18,
    damageMod: -2,
    spreadMod: 1.1,
    recipe: { wild_carrot: 2 },
  },
  {
    id: 'muzzle_soup_can',
    name: 'Campbell Soup Can Choke',
    slot: 'muzzle',
    description: 'Hollowed tin soup can acting as an aggressive shotgun choke, tightening pellet dispersion and boosting close punch.',
    spreadMod: 0.6,
    damageMod: 4,
    recipe: { canned_beans: 1, scrap_metal: 2 },
  },
  {
    id: 'muzzle_car_horn',
    name: 'Car Horn Blast Shroud',
    slot: 'muzzle',
    description: 'Repurposed automotive horn. Emits an obnoxious HONK on every shot that disorients targets and packs kinetic punch.',
    honkSound: true,
    noiseMod: 2.8,
    damageMod: 7,
    recipe: { scrap_metal: 4, copper_wire: 2 },
  },
  {
    id: 'muzzle_copper_pipe',
    name: 'Extended Copper Pipe Rifled Barrel',
    slot: 'muzzle',
    description: 'Polished plumbing pipe extending muzzle velocity, bullet flight stability, and sniper penetration.',
    rangeMod: 18,
    spreadMod: 0.72,
    damageMod: 9,
    recipe: { copper_wire: 4, scrap_metal: 3 },
  },
  {
    id: 'muzzle_kazoo',
    name: 'Party Kazoo Muzzle Brake',
    slot: 'muzzle',
    description: 'A party kazoo that vents propellant gases sideways, reducing muzzle climb with a festive buzz sound.',
    squeakSound: true,
    recoilMod: 0.65,
    recipe: { cloth: 1, scrap_metal: 1 },
  },

  // ==========================================
  // UNDERBARREL & FOREGRIPS
  // ==========================================
  {
    id: 'under_cat_laser',
    name: 'Kitty Cat Toy Laser Sight',
    slot: 'underbarrel',
    description: 'Bright red diode pointer. Gives laser hipfire stability (-45% spread) while amusing wildlife.',
    spreadMod: 0.55,
    recipe: { copper_wire: 2, glass: 1, battery: 1 },
  },
  {
    id: 'under_plunger_bayonet',
    name: 'Toilet Plunger Melee Bayonet',
    slot: 'underbarrel',
    description: 'Heavy duty rubber plunger mounted forward. Allows bludgeoning and suction-cupping encroaching zombies for heavy melee damage.',
    meleeDmg: 38,
    recipe: { wood: 3, cloth: 2 },
  },
  {
    id: 'under_kitchen_fork',
    name: 'Kitchen Meat Fork Bayonet',
    slot: 'underbarrel',
    description: 'Sharpened twin-tine carving fork. Delivers brutal pierce damage to anything trying to grab your barrel.',
    meleeDmg: 48,
    damageMod: 3,
    recipe: { scrap_metal: 3 },
  },
  {
    id: 'under_flashlight',
    name: 'Duct-Taped Tactical Flashlight',
    slot: 'underbarrel',
    description: 'Casts a focused forward beam in the direction of fire and provides night vision aiming reticle.',
    visualInfo: {
      nightVisionReticle: true,
    },
    recipe: { battery: 1, scrap_metal: 2 },
  },
  {
    id: 'under_salad_spoon',
    name: 'Salad Tongs Ergonomic Foregrip',
    slot: 'underbarrel',
    description: 'Wooden salad serving spoons clamped with radiator brackets. Dramatically cuts muzzle kickback.',
    recoilMod: 0.55,
    spreadMod: 0.8,
    recipe: { wood: 2, scrap_metal: 1 },
  },

  // ==========================================
  // MAGAZINES & FEED SYSTEMS
  // ==========================================
  {
    id: 'mag_twin_soda',
    name: 'Twin Soda Bottles Mag Extension',
    slot: 'magazine',
    description: 'Two 2-liter soda bottles taped end-to-end. Doubles ammunition capacity before needing to chamber rounds.',
    capacityMod: 2.0,
    recipe: { cloth: 2, scrap_metal: 2 },
  },
  {
    id: 'mag_rubber_band',
    name: 'Rubber Band Speedloader',
    slot: 'magazine',
    description: 'Interlocked rubber bands providing spring-assisted bolt cycling, accelerating weapon fire rate by +40%.',
    fireRateMod: 1.4,
    recipe: { cloth: 2 },
  },
  {
    id: 'mag_hopper_funnel',
    name: 'Gravitational Hopper Funnel',
    slot: 'magazine',
    description: 'Wide funnel hopper that swallows heaps of ammo, increasing capacity by +60% with slight bulkiness.',
    capacityMod: 1.6,
    spreadMod: 1.15,
    recipe: { scrap_metal: 3 },
  },

  // ==========================================
  // GOOFY & HOMEMADE CHARMS
  // ==========================================
  {
    id: 'goofy_rubber_duck',
    name: 'Rubber Duck Trigger Squeaker',
    slot: 'goofy',
    description: 'Bright yellow bath duck seated behind the trigger. Cushions trigger pull with a cheery squeak, steadying aim.',
    squeakSound: true,
    recoilMod: 0.6,
    recipe: { cloth: 2 },
  },
  {
    id: 'goofy_fuzzy_dice',
    name: 'Lucky Fuzzy Dice Charm',
    slot: 'goofy',
    description: 'Plush velvet dice dangling from the trigger guard. Boosts lucky ballistic critical damage by +10 DMG.',
    damageMod: 10,
    recipe: { cloth: 3 },
  },
  {
    id: 'goofy_whoopee',
    name: 'Whoopee Cushion Recoil Buffer',
    slot: 'goofy',
    description: 'Pneumatic prank cushion strapped to the stock. Cushions shoulder recoil with a comedic toot on high-power shots.',
    recoilMod: 0.45,
    recipe: { cloth: 2 },
  },
  {
    id: 'goofy_windup_clock',
    name: 'Wind-Up Clock Heartbeat Monitor',
    slot: 'goofy',
    description: 'Vintage brass alarm clock wired to the receiver. Synchronizes tactical scanning to reveal enemy threat brackets.',
    visualInfo: {
      threatScanner: true,
    },
    recipe: { copper_wire: 3, scrap_metal: 3 },
  },

  // ==========================================
  // EXPANDED OPTICS
  // ==========================================
  {
    id: 'optic_welder_mask_scope',
    name: 'Welder Mask Flip Scope',
    slot: 'optic',
    description: 'Flip-up dark welding lens reduces glare and steadies precision fire.',
    rangeMod: 12,
    spreadMod: 0.72,
    visualInfo: { targetHealthBar: true },
    recipe: { scrap_metal: 4, cloth: 2 },
  },
  {
    id: 'optic_binocular_half',
    name: 'Half-Binocular Scout Optic',
    slot: 'optic',
    description: 'One-tube binocular rig for cheap mid-range recon and cleaner sight picture.',
    rangeMod: 9,
    spreadMod: 0.82,
    visualInfo: { rangefinder: true },
    recipe: { glass: 2, scrap_metal: 2 },
  },
  {
    id: 'optic_periscope',
    name: 'Periscope Corner Sight',
    slot: 'optic',
    description: 'Offset mirror tube for peeking corners with threat highlighting.',
    spreadMod: 0.9,
    recoilMod: 0.9,
    visualInfo: { threatScanner: true },
    recipe: { scrap_metal: 3, glass: 1, cloth: 1 },
  },
  {
    id: 'optic_laser_grid',
    name: 'Laser Grid Composer',
    slot: 'optic',
    description: 'Projected micro-grid helps track recoil pattern and follow-up shots.',
    spreadMod: 0.78,
    recoilMod: 0.82,
    recipe: { copper_wire: 3, battery: 1, scrap_metal: 2 },
  },

  // ==========================================
  // EXPANDED MUZZLES
  // ==========================================
  {
    id: 'muzzle_bottle_suppressor',
    name: 'Bottle Suppressor',
    slot: 'muzzle',
    description: 'Plastic bottle baffling tames report at the cost of slight accuracy drift.',
    noiseMod: 0.35,
    spreadMod: 1.08,
    recipe: { cloth: 1, scrap_metal: 1 },
  },
  {
    id: 'muzzle_rebar_comp',
    name: 'Rebar Compensator',
    slot: 'muzzle',
    description: 'Cut rebar ports vent blast upward for tighter recoil control.',
    recoilMod: 0.72,
    noiseMod: 1.1,
    recipe: { scrap_metal: 3, copper_wire: 1 },
  },
  {
    id: 'muzzle_forked_brake',
    name: 'Forked Muzzle Brake',
    slot: 'muzzle',
    description: 'Kitchen fork prongs split gas flow and reduce climb while staying loud.',
    recoilMod: 0.7,
    noiseMod: 1.25,
    recipe: { scrap_metal: 2, cloth: 1 },
  },
  {
    id: 'muzzle_pressure_cone',
    name: 'Pressure Cone Booster',
    slot: 'muzzle',
    description: 'Forward pressure cone boosts velocity and close-range stopping power.',
    damageMod: 5,
    rangeMod: 6,
    recoilMod: 1.15,
    recipe: { scrap_metal: 4, copper_wire: 2 },
  },

  // ==========================================
  // EXPANDED UNDERBARRELS
  // ==========================================
  {
    id: 'under_angle_grip',
    name: 'Angled Scrap Grip',
    slot: 'underbarrel',
    description: 'Improvised angled foregrip for recoil smoothing and snap transitions.',
    recoilMod: 0.74,
    spreadMod: 0.88,
    recipe: { wood: 2, scrap_metal: 1 },
  },
  {
    id: 'under_wire_stock',
    name: 'Wire Tension Brace',
    slot: 'underbarrel',
    description: 'Cable tension brace stabilizes the frame under sustained fire.',
    recoilMod: 0.8,
    fireRateMod: 1.08,
    recipe: { copper_wire: 3, scrap_metal: 2 },
  },
  {
    id: 'under_stun_prong',
    name: 'Stun Prong Bayonet',
    slot: 'underbarrel',
    description: 'Charged improvised prong that improves weapon bash effectiveness.',
    meleeDmg: 42,
    damageMod: 2,
    recipe: { copper_wire: 2, battery: 1, scrap_metal: 2 },
  },
  {
    id: 'under_pipe_bipod',
    name: 'Pipe Bipod Kit',
    slot: 'underbarrel',
    description: 'Fold-out pipe bipod for steadier long-range shots.',
    spreadMod: 0.68,
    recoilMod: 0.86,
    recipe: { scrap_metal: 3, wood: 1 },
  },

  // ==========================================
  // EXPANDED MAGAZINES
  // ==========================================
  {
    id: 'mag_drum_diy',
    name: 'DIY Drum Mag',
    slot: 'magazine',
    description: 'Riveted drum housing dramatically increases ammo capacity.',
    capacityMod: 2.4,
    spreadMod: 1.1,
    recipe: { scrap_metal: 5, cloth: 2 },
  },
  {
    id: 'mag_quad_stack',
    name: 'Quad Stack Mag',
    slot: 'magazine',
    description: 'Four-lane feed stack with high capacity and modest reliability loss.',
    capacityMod: 1.9,
    spreadMod: 1.06,
    recipe: { scrap_metal: 4, copper_wire: 2 },
  },
  {
    id: 'mag_featherlight',
    name: 'Featherlight Race Mag',
    slot: 'magazine',
    description: 'Low-mass spring follower boosts cyclic speed for burst builds.',
    fireRateMod: 1.2,
    capacityMod: 1.1,
    recipe: { cloth: 1, copper_wire: 2 },
  },
  {
    id: 'mag_tape_jungle',
    name: 'Jungle Tape Dual Mag',
    slot: 'magazine',
    description: 'Two mags taped together for rapid flips and sustained fire.',
    capacityMod: 1.5,
    fireRateMod: 1.08,
    recipe: { cloth: 2, scrap_metal: 2 },
  },

  // ==========================================
  // EXPANDED GOOFY CHARMS
  // ==========================================
  {
    id: 'goofy_party_whistle',
    name: 'Party Whistle Trigger Charm',
    slot: 'goofy',
    description: 'Every shot toots with party chaos while dampening perceived kick.',
    squeakSound: true,
    recoilMod: 0.7,
    recipe: { cloth: 2 },
  },
  {
    id: 'goofy_lucky_sock',
    name: 'Lucky Sock Wrap',
    slot: 'goofy',
    description: 'A questionable lucky sock wrapped around the stock for confidence damage.',
    damageMod: 6,
    recipe: { cloth: 2 },
  },
  {
    id: 'goofy_honk_bulb',
    name: 'Honk Bulb Grip',
    slot: 'goofy',
    description: 'Bike honk bulb on the foregrip turns every burst into noisy panic.',
    honkSound: true,
    noiseMod: 1.7,
    damageMod: 2,
    recipe: { scrap_metal: 1, cloth: 1 },
  },
  {
    id: 'goofy_confetti_canister',
    name: 'Confetti Canister',
    slot: 'goofy',
    description: 'A ridiculous confetti module that improves morale and target callouts.',
    visualInfo: { targetHealthBar: true, threatScanner: true },
    recipe: { cloth: 2, copper_wire: 1 },
  },
];
