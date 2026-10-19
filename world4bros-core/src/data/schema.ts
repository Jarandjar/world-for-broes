/** Data contracts. Content is plain data validated against these types (see data/validate.ts). */
export type Id = string;

// ---- appearance and equipment ----
export type Slot = 'head' | 'face' | 'neck' | 'torso' | 'legs' | 'feet' | 'hands' | 'back' | 'belt' | 'mainHand' | 'offHand';
export const SLOTS: readonly Slot[] = ['head', 'face', 'neck', 'torso', 'legs', 'feet', 'hands', 'back', 'belt', 'mainHand', 'offHand'];
export const SLOT_LABELS: Record<Slot, string> = {
  head: 'Headwear', face: 'Face', neck: 'Neck', torso: 'Jacket and top', legs: 'Legs', feet: 'Footwear',
  hands: 'Gloves', back: 'Backpack', belt: 'Belt', mainHand: 'Main hand', offHand: 'Off hand',
};
export interface ColorSet { primary: string; secondary: string; accent: string }
export type ToolKind = 'axe' | 'pickaxe' | 'shovel' | 'weapon' | 'torch' | 'none';
export type AimSystemKind = 'laser' | 'bloom' | 'arc' | 'burst';
export type AttachmentSlot = 'optic' | 'muzzle' | 'underbarrel' | 'magazine' | 'goofy';

export interface VisualInfoMods {
  rangefinder?: boolean;       // Displays exact distance readout (e.g. '14.2m') and bullet travel time
  targetHealthBar?: boolean;   // Live enemy / target dummy HP gauge displayed in reticle HUD
  threatScanner?: boolean;     // Highlights zombies/animals with pulsating tactical brackets
  trajectoryArc?: boolean;     // Live 3D/parabolic trajectory path line preview
  nightVisionReticle?: boolean;// Luminous tactical green reticle with compass azimuth
}

export interface GunAttachmentDef {
  id: Id;
  name: string;
  slot: AttachmentSlot;
  description: string;
  icon?: string;
  damageMod?: number;
  rangeMod?: number;
  spreadMod?: number;          // multiplier on spreadAngle (e.g. 0.65 = 35% tighter)
  noiseMod?: number;           // multiplier on noiseRadius (e.g. 0.2 = 80% quieter)
  fireRateMod?: number;        // multiplier on fire rate
  recoilMod?: number;          // multiplier on recoil impulse
  capacityMod?: number;        // multiplier on ammo capacity / bursts
  meleeDmg?: number;           // melee damage bonus on bash
  squeakSound?: boolean;       // comedy squeak trigger
  honkSound?: boolean;         // comedy car horn trigger
  visualInfo?: VisualInfoMods;
  recipe?: Record<Id, number>;
}

export interface GunAttachments {
  optic?: Id | null;
  muzzle?: Id | null;
  underbarrel?: Id | null;
  magazine?: Id | null;
  goofy?: Id | null;
}

export interface GunDef {
  ammoItem: Id;
  ammoName?: string;
  ammoPerShot: number;
  aimSystem: AimSystemKind;
  projectileSpeed: number; // m/s
  maxRange: number;        // meters
  baseDamage: number;
  minDamage: number;
  falloffStartDist: number; // distance before damage drops
  spreadAngle: number;     // radians spread (bloom)
  fireRate: number;        // shots per second
  noiseRadius: number;     // sound alert radius
  penetration?: number;    // how many targets round pierces
  ricochets?: number;
  arced?: boolean;         // for parabolic trajectory
  pelletCount?: number;    // for shotguns
  bulletColor?: string;
  bulletRadius?: number;
  defaultAttachments?: GunAttachments;
}

export interface ItemDef {
  id: Id; name: string;
  kind: 'material' | 'food' | 'tool' | 'gear';
  stack: number; weight: number;
  /** Drawer id for the inventory icon (see render/icons.ts). Defaults to the item id. */
  icon?: string;
  equip?: { slot: Slot; colors: ColorSet; twoHanded?: boolean; lightRadius?: number };
  tool?: { kind: ToolKind; power: number; speed: number; reach: number; damage: number };
  gun?: GunDef;
  food?: { hunger: number; health?: number; sanity?: number };
  tags?: string[];
}

export interface HairStyleDef { id: Id; name: string }

// ---- world ----
export type Terrain = 'grass' | 'dirt' | 'needles' | 'mud' | 'rock' | 'sand' | 'snow' | 'ash' | 'shallow' | 'deep' | 'moss';
export interface TerrainDef {
  id: Terrain; walkable: boolean; speed: number; colors: [string, string, string]; strokes: 'blades' | 'needles' | 'pebbles' | 'cracks' | 'waves' | 'flakes';
  /** Surface tags for effect reactions (§116). Standing in a `wet` tile makes you wet. */
  tags?: SurfaceTag[];
}

// ---- status effects, reactions and stagger (Z26, World Bible §115–§118) ----
export type EffectId = 'acid' | 'burn' | 'explosion' | 'stagger' | 'electricity' | 'cold' | 'spores' | 'goo' | 'wet' | 'oil_slick' | 'blind' | 'smoke';
export const EFFECT_IDS: readonly EffectId[] = ['acid', 'burn', 'explosion', 'stagger', 'electricity', 'cold', 'spores', 'goo', 'wet', 'oil_slick', 'blind', 'smoke'];
export type SurfaceTag = 'flammable' | 'wet' | 'conductive' | 'explosive' | 'oil' | 'frozen' | 'metal';
export const SURFACE_TAGS: readonly SurfaceTag[] = ['flammable', 'wet', 'conductive', 'explosive', 'oil', 'frozen', 'metal'];
export type HazardKind = 'fire' | 'puddle' | 'cloud';

/** One status effect as data (§115). Numbers are starting values. */
export interface EffectDef {
  id: EffectId; name: string;
  /** Hex colour from §115. Never the only signal: every effect also has an icon and a sound word. */
  color: string;
  /** Icon drawer id (render/effectIcons.ts). */
  icon: string;
  /** Hand-lettered comic word shown when it lands. */
  soundWord: string;
  /** Plain-words tooltip. */
  tooltip: string;
  from: string; does: string; ends: string;
  /** Seconds it lasts. 0 = instant (explosion, stagger). */
  duration: number;
  /** Health lost per second while active (scaled by power). */
  dps?: number;
  /** Sanity lost per second while active (players). */
  sanityDps?: number;
  /** Instant damage when applied (scaled by power). */
  burst?: number;
  /** Movement multiplier while active (per stack for stacking effects). */
  slow?: number;
  /** Cannot move while active. */
  stun?: boolean;
  /** Ruins aim (no shooting, wide spread). */
  ruinsAim?: boolean;
  /** Sight multiplier for whoever has it (blind, smoke inside). */
  sight?: number;
  /** Multiplier on how far others can see you while you have it (smoke hides you). */
  hidden?: number;
  /** Damage taken multiplier while active (goo makes you easy to hit). */
  damageTaken?: number;
  /** Momentum carries: you slide. */
  slide?: boolean;
  /** Stacks build up to this; at max the `atMax` state kicks in (cold freezes). */
  maxStacks?: number;
  atMax?: { stun?: boolean; tags?: SurfaceTag[]; duration: number };
  /** Tags the holder gets while it is active (wet → wet; oil → oil + flammable). */
  grantsTags?: SurfaceTag[];
  /** Applying this effect removes these (water clears burn, acid, goo). */
  clears?: EffectId[];
  /** Ends early when the holder stands in light (spores). */
  endsInLight?: boolean;
  /** Poise impact when applied (stagger, explosion). Blast hits go straight to knockdown. */
  impact?: number;
  blast?: boolean;
  /** Area radius in metres for area effects (explosion). */
  radius?: number;
  /** Ground hazard this effect can leave (acid puddle, fire, smoke cloud). */
  hazard?: { kind: HazardKind; radius: number; duration: number };
  /** Flashes on screen: obeys the reduced-flashing setting. */
  flashes?: boolean;
}

/** What happens when an effect meets a surface tag (§116). Rows are data, so new tags need no engine change. */
export interface ReactionDef {
  effect: EffectId; tag: SurfaceTag;
  /** Short words shown in-game ("Spreads", "Put out"). */
  label: string;
  /** ▲ in §116: makes things bigger or chains. */
  amplifies?: boolean;
  /** The incoming effect is cancelled (fire on wet). */
  cancels?: boolean;
  /** Extra effects put on the target. */
  apply?: EffectId[];
  /** Effects stripped from the target. */
  remove?: EffectId[];
  /** Multiplier on the incoming effect's damage / power. */
  powerMult?: number;
  /** Multiplier on stun time (electricity on metal). */
  stunMult?: number;
  /** Multiplier on knockback (explosion on water: geyser). */
  knockMult?: number;
  /** Sets off an explosion at the target (barrels, sappers). */
  explode?: boolean;
  /** Passes the incoming effect on to targets within this radius that carry the same tag. */
  chainRadius?: number;
  /** Leaves a ground hazard of this effect. */
  spawnHazard?: EffectId;
  /** Stops a vehicle's engine for this many seconds. */
  stallVehicle?: number;
}

export type StaggerStep = 'flinch' | 'stagger' | 'stumble' | 'knockdown';
/** One rung of the stagger ladder (§117). */
export interface StaggerStepDef {
  step: StaggerStep;
  /** Share of max poise used up that triggers this step. */
  share: number;
  duration: number;
  speedMult: number;
  aimKept: boolean;
  canAttack: boolean;
  /** Metres pushed back from the hit. */
  pushback: number;
  /** Damage taken multiplier while in this step. */
  damageTaken: number;
  /** Creatures with at least this max poise ignore this step. */
  ignoredFromPoise?: number;
}
export interface EffectLimits {
  maxFires: number; maxPuddles: number; maxClouds: number;
  /** How deep chain reactions may go (barrel → barrel → barrel). */
  maxChainDepth: number;
  /** Total targets one chain may touch. */
  maxChainTargets: number;
  poiseRegenDelay: number;
  /** Share of max poise refilled per second after the delay. */
  poiseRegenRate: number;
  playerPoise: number;
}

export type NodePurity = 'impure' | 'normal' | 'pure' | 'jackpot';

export type ObjectKind =
  | 'pine' | 'snow_pine' | 'oak' | 'dead_tree' | 'bush' | 'berry_bush'
  | 'rock' | 'boulder' | 'log' | 'stump' | 'mushroom' | 'reeds' | 'cactus'
  | 'crate' | 'tent' | 'fuel_pump' | 'radio_tower' | 'sandbags' | 'wrecked_car'
  | 'iron_node' | 'copper_node' | 'coal_node' | 'oil_seam' | 'crystal_geode'
  | 'town_fountain' | 'street_light' | 'guardrail' | 'picket_fence'
  | 'park_bench' | 'fire_hydrant' | 'dumpster' | 'traffic_cone'
  | 'target_dummy' | 'steel_target'
  | 'slime_drum' | 'scrap_wall' | 'tactical_shield'
  // town street life (townPropStage)
  | 'stop_sign' | 'street_sign' | 'bus_stop' | 'trash_can' | 'power_pole' | 'abandoned_car' | 'litter';
export interface ObjectDef {
  kind: ObjectKind; name: string; radius: number; solid: boolean; hp: number;
  /** Tool needed to harvest, or 'hands'. */
  tool: ToolKind | 'hands';
  drops: { item: Id; min: number; max: number }[];
  becomes?: ObjectKind;
  prompt: string;
  /** Surface tags for effect reactions: fire spreads to `flammable` props. */
  tags?: SurfaceTag[];
}
export interface ObjectRule { kind: ObjectKind; density: number; minSpacing: number }
export interface BiomeDef {
  id: Id; name: string;
  climate: { temp: number; moist: number; elev: number };
  terrain: { base: Terrain; patch: Terrain; patchAmount: number };
  objects: ObjectRule[];
  tint: string; // fog and ambient tint
}

// ---- base building ----
export interface StructureDef {
  id: Id; name: string; size: [number, number]; solid: boolean; hp: number;
  cost: Record<Id, number>;
  requiresClaim: boolean; claimRadius?: number;
  light?: { radius: number; color: string; flicker: number };
  interact?: 'storage' | 'sleep' | 'door' | 'craft' | 'fire' | 'table' | 'overseer' | 'turret' | 'totem' | 'gunsmith';
  slots?: number;
}

export interface RecipeDef { id: Id; name: string; inputs: Record<Id, number>; output: { item: Id; qty: number } }

// ---- vehicles ----
export type VehicleKind = 'hatchback' | 'clown_car' | 'ice_cream_truck' | 'golf_cart' | 'bathtub_buggy' | 'hot_dog_van' | 'tesla_roadster';

export interface VehicleDef {
  id: Id;
  name: string;
  kind: VehicleKind;
  size: [number, number];
  maxSpeed: number;
  accel: number;
  turnSpeed: number;
  maxHp: number;
  maxFuel: number;
  trunkSlots: number;
  lightRadius: number;
  color: string;
  quirk?: string;
}

// ---- zombies ----
export type ZombieKind = 'walker' | 'runner' | 'brute' | 'sapper' | 'spitter' | 'burrower' | 'screamer' | 'goliath';

export interface ZombieDef {
  id: Id;
  name: string;
  kind: ZombieKind;
  maxHp: number;
  damage: number;
  speed: number;
  runSpeed: number;
  detectRadius: number;
  attackRadius: number;
  attackCooldown: number;
  drops: { item: Id; min: number; max: number; chance: number }[];
  /** Stagger resistance (§117). Defaults to maxHp. */
  poise?: number;
  /** Surface tags (a sapper is `explosive`). */
  tags?: SurfaceTag[];
  /** Effects that do nothing to this kind. */
  immune?: EffectId[];
  /** Effect put on whoever it hits. */
  attackEffect?: { effect: EffectId; power: number };
}

// ---- animals & wildlife (Phase Z19) ----
export type AnimalKind = 'cow' | 'pig' | 'sheep' | 'chicken' | 'deer' | 'rabbit' | 'goose';

export interface AnimalDef {
  id: Id;
  name: string;
  kind: AnimalKind;
  maxHp: number;
  radius: number;
  speed: number;
  runSpeed: number;
  fleeDistance: number;
  sound: string;
  favoriteFood: Id[];
  produce?: { item: Id; cooldown: number; actionPrompt: string };
  drops: { item: Id; min: number; max: number; chance: number }[];
}

// ---- automation & logistics (Factorio & Satisfactory) ----
export type MachineKind = 'miner' | 'conveyor' | 'lift' | 'splitter' | 'inserter' | 'smelter' | 'assembler' | 'refinery' | 'generator' | 'battery';
export interface MachineDef {
  id: Id;
  name: string;
  kind: MachineKind;
  powerDrawKw: number;
  pollutionPerMinute: number;
  noiseRadius: number;
  throughputPerSecond?: number;
}

// ---- colony & pawn psychology (RimWorld) ----
export type PawnWorkRole = 'firefight' | 'patient' | 'doctor' | 'build' | 'farm' | 'mine' | 'haul' | 'craft' | 'guard' | 'research';
export type PawnSkill = 'construction' | 'mining' | 'crafting' | 'cooking' | 'medicine' | 'plants' | 'combat' | 'hauling';
export type PawnPassion = 'none' | 'single' | 'burning';
export type PawnTrait = 'grease_monkey' | 'pyromaniac' | 'slacker' | 'brawler' | 'night_owl' | 'gourmand' | 'nervous_wreck' | 'clumsy' | 'fast_walker';
export type MentalBreakKind = 'none' | 'binge_eating' | 'hide_in_outhouse' | 'tool_tantrum' | 'conveyor_carousel' | 'sad_wander';
export type StorytellerKind = 'caspar_chill' | 'randy_ruckus' | 'hilda_hardcore';
export type IncidentKind = 'wanderer_joins' | 'supply_drop' | 'short_circuit' | 'stampede' | 'screamer_swarm';

export interface PawnTraitDef {
  id: Id;
  name: string;
  description: string;
  speedMod?: number;
  craftMod?: number;
  moodOffset?: number;
}

// ---- pvp & territorial warfare ----
export type ZoneRiskTier = 'green' | 'yellow' | 'red';
export interface TerritoryDef {
  id: Id;
  name: string;
  tier: ZoneRiskTier;
  allowPvP: boolean;
  canRaidBases: boolean;
  fullLootDrop: boolean;
}

export interface Content {
  items: Map<Id, ItemDef>;
  terrain: Map<Terrain, TerrainDef>;
  biomes: BiomeDef[];
  objects: Map<ObjectKind, ObjectDef>;
  structures: Map<Id, StructureDef>;
  recipes: RecipeDef[];
  hairStyles: HairStyleDef[];
  vehicles: Map<Id, VehicleDef>;
  zombies: Map<Id, ZombieDef>;
  animals: Map<Id, AnimalDef>;
  buildings: BuildingDef[];
  attachments: Map<Id, GunAttachmentDef>;
  effects: Map<EffectId, EffectDef>;
  reactions: ReactionDef[];
  staggerLadder: StaggerStepDef[];
  effectLimits: EffectLimits;
}

export type BuildingKind = 'house' | 'apartment' | 'shop' | 'store' | 'supermarket' | 'garage' | 'clinic' | 'warehouse' | 'shed';
export type RoofStyle = 'gable' | 'hip' | 'flat' | 'thatch' | 'tin' | 'broken';
export type WallStyle = 'plank' | 'log' | 'brick' | 'plaster' | 'scrap' | 'concrete';
export type BuildingFeature = 'porch' | 'chimney' | 'awning' | 'fence' | 'flowerBoxes' | 'boarded' | 'vines' | 'banner' | 'tarp' | 'sign' | 'garden' | 'hvac' | 'balcony' | 'glassFront' | 'mailbox';
export interface BuildingPalette { wall: string; roof: string; trim: string }

/** A kind of town building. Size ranges are in tiles (outer footprint). Everything about its look is data. */
export interface BuildingDef {
  id: Id; name: string; kind: BuildingKind;
  desc?: string;
  weight: number;
  /** 'core' favours the town centre, 'outskirts' the edges. */
  zone?: 'core' | 'outskirts';
  minW: number; maxW: number; minH: number; maxH: number;
  roof: RoofStyle; walls: WallStyle; door: 'single' | 'double' | 'rollup';
  colors: { wall: string; roof: string; floor: string; trim: string };
  /** Optional alternative colour sets; one is picked per building so neighbours differ. */
  palettes?: BuildingPalette[];
  features?: BuildingFeature[];
  /** Inner walls: 'split' two rooms, 'apartment' three rooms, 'backroom' storage behind the counter. */
  layout?: 'split' | 'apartment' | 'backroom' | 'none';
}

