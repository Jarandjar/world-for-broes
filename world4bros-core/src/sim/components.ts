import { Component } from '../engine/ecs';
import type { Vec2 } from '../engine/math';
import type { StructureDef, VehicleDef, ZombieDef } from '../data/schema';
import type { Entity } from '../engine/ecs';
import type { Look } from './look';
import type { Inventory } from './inventory';

export interface Transform { x: number; y: number; angle: number }
export interface Body { radius: number; speed: number; vx: number; vy: number }
export interface Anim { state: 'idle' | 'walk' | 'swing'; phase: number; walkBlend: number; swingT: number; swingDur: number; hitDone: boolean; hitFlash: number }
export interface CharacterC { name: string; look: Look; anim: Anim }
export interface Needs { hunger: number; health: number; sanity: number; maxHunger: number; maxHealth: number; maxSanity: number; darkTimer: number }
export interface InventoryC { inv: Inventory; held: number }
export interface LightC { radius: number; color: string; flicker: number }
export interface StructureC { def: StructureDef; tx: number; ty: number; hp: number; open: boolean; storage?: Inventory }
export interface VehicleC { def: VehicleDef; driver: Entity | null; fuel: number; hp: number; speed: number; steer: number; headlights: boolean; trunk: Inventory }
export interface Wander { target: Vec2 | null; wait: number }
export interface ZombieC {
  def: ZombieDef;
  hp: number;
  state: 'idle' | 'wander' | 'alert' | 'chase' | 'attack';
  target: Entity | null;
  alertPos: Vec2 | null;
  attackTimer: number;
  groanTimer: number;
  animPhase: number;
  hitFlash: number;
}
export interface PetC {
  name: string;
  kind: 'wolf_pup' | 'hunting_dog';
  hp: number;
  maxHp: number;
  owner: Entity;
  state: 'follow' | 'attack' | 'sit';
  target: Entity | null;
  barkTimer: number;
  biteTimer: number;
  animPhase: number;
  tailWag: number;
}

import type { AnimalDef, AnimalKind } from '../data/schema';
export interface AnimalC {
  def: AnimalDef;
  name: string;
  kind: AnimalKind;
  hp: number;
  maxHp: number;
  state: 'graze' | 'wander' | 'flee' | 'tamed';
  fleeTimer: number;
  grazeTimer: number;
  produceTimer: number;
  animPhase: number;
  eating: boolean;
  tamed: boolean;
  alertPos: Vec2 | null;
  target: Entity | null;
  tailWag: number;
}

export type CardinalDir = 'N' | 'E' | 'S' | 'W';

export interface ConveyorItem { item: string; progress: number }
export interface ConveyorC {
  dir: CardinalDir;
  speed: number;
  items: ConveyorItem[];
}

export interface InserterC {
  pickupDir: CardinalDir;
  dropDir: CardinalDir;
  heldItem: string | null;
  armProgress: number;
  speed: number;
}

import type { PawnSkill, PawnPassion, PawnTrait, MentalBreakKind, PawnWorkRole } from '../data/schema';
export type { PawnWorkRole };

export interface MachineC {
  kind: 'burner_drill' | 'stone_furnace';
  fuel: number;
  progress: number;
  targetOre?: string;
  outputItem?: string;
}

export interface PawnThought {
  id: string;
  label: string;
  delta: number;
  duration: number;
}

export interface PawnJob {
  role: PawnWorkRole;
  targetEntity?: Entity;
  targetPos?: { x: number; y: number };
  state: 'walking' | 'working';
  progress: number;
  duration: number;
}

export interface PawnC {
  skills: Record<PawnSkill, { level: number; passion: PawnPassion; xp: number }>;
  traits: PawnTrait[];
  mood: number;
  thoughts: PawnThought[];
  mentalBreak: { kind: MentalBreakKind; timer: number; targetPos?: { x: number; y: number } };
  workPriorities: Record<PawnWorkRole, number>;
  currentJob: PawnJob | null;
  assignedBed?: Entity;
}

export interface TurretC {
  kind: 'bolt' | 'flame' | 'zap' | 'mallet' | 'drone';
  range: number;
  damage: number;
  cooldown: number;
  timer: number;
  ammo: number;
  maxAmmo: number;
  target?: Entity;
}

export interface TotemC {
  clanName: string;
  radius: number;
  shieldActive: boolean;
  raidWindowStart: number;
  raidWindowEnd: number;
  treatyUntilDay?: number;
  bountyCredits: number;
}

export interface ProjectileC {
  source: Entity;
  startX: number;
  startY: number;
  vx: number;
  vy: number;
  speed: number;
  maxDist: number;
  distTraveled: number;
  baseDamage: number;
  minDamage: number;
  falloffStartDist: number;
  penetration: number;
  piercedCount: number;
  arced: boolean;
  color: string;
  radius: number;
  trail: { x: number; y: number }[];
}

export interface TargetDummyC {
  distanceLabel: string;
  lastDamageTaken: number;
  lastHitDistance: number;
  lastHitTime: number;
  totalHits: number;
  wobbleTimer: number;
}

export interface RaiderC {
  kind: 'raider' | 'heavy' | 'marksman';
  hp: number;
  maxHp: number;
  speed: number;
  sight: number;
  weapon: string;
  level: 'scarecrow' | 'regular' | 'sharp' | 'menace';
  team: 'bot' | 'raider';
  target: Entity | null;
  state: 'idle' | 'advance' | 'investigate' | 'engage' | 'cover' | 'flee';
  cooldown: number;
  reactionTimer: number;
  heardPos: Vec2 | null;
  heardTimer: number;
  coverPos: Vec2 | null;
  coverTimer: number;
  strafeDir: number;
  strafeTimer: number;
  firstDamageTime: number;
  aimSpread: number;
  shotInterval: number;
}

export const Comp = {
  Transform: new Component<Transform>('transform'),
  Body: new Component<Body>('body'),
  Character: new Component<CharacterC>('character'),
  Needs: new Component<Needs>('needs'),
  Inventory: new Component<InventoryC>('inventory'),
  Light: new Component<LightC>('light'),
  Structure: new Component<StructureC>('structure'),
  Vehicle: new Component<VehicleC>('vehicle'),
  Wander: new Component<Wander>('wander'),
  Zombie: new Component<ZombieC>('zombie'),
  Pet: new Component<PetC>('pet'),
  Player: new Component<true>('player'),
  Conveyor: new Component<ConveyorC>('conveyor'),
  Inserter: new Component<InserterC>('inserter'),
  Machine: new Component<MachineC>('machine'),
  Pawn: new Component<PawnC>('pawn'),
  Turret: new Component<TurretC>('turret'),
  Totem: new Component<TotemC>('totem'),
  Animal: new Component<AnimalC>('animal'),
  Projectile: new Component<ProjectileC>('projectile'),
  TargetDummy: new Component<TargetDummyC>('targetDummy'),
  Raider: new Component<RaiderC>('raider'),
};
export const newAnim = (): Anim => ({ state: 'idle', phase: 0, walkBlend: 0, swingT: 0, swingDur: 0.5, hitDone: true, hitFlash: 0 });
