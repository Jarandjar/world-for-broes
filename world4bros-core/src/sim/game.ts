import { Ecs, type Entity } from '../engine/ecs';
import { Events } from '../engine/events';
import { makeRng, type Rng } from '../engine/rng';
import { clamp, damp, lerpAngle } from '../engine/math';
import { buildContent, type ContentPack } from '../data/content';
import { ARENA_BOT_LEVELS, type ArenaBotLevel } from '../data/arenaBots';
import type { Content, Id, Slot, StructureDef, ToolKind, AnimalKind, AttachmentSlot, GunAttachments, VisualInfoMods, GunDef } from '../data/schema';
import { Comp, newAnim } from './components';
import { GameWorld, type ObjectMod } from './world';
import type { StaticMapSave } from './staticmap';
import type { WorldObject } from './worldgen';
import { BaseSystem } from './base';
import { Clock } from './clock';
import { Inventory } from './inventory';
import { defaultLook, randomLook, sanitizeLook, type Look } from './look';
import { ColonySystem, createRandomPawn, type ColonySaveData } from './colony';
import { SiegeSystem, type SiegeSaveData } from './siege';
import { FactionSystem, type FactionSaveData } from './faction';
import { type ArenaDef, ARENAS, arenaBlocksBullet, arenaBlocksMovement } from './arena';
import { devLog } from '../utils/dev-log';

export interface GameConfig { seed: number; mode: 'classic' | 'zombie'; creative: boolean; darknessDamage: boolean; alwaysDay?: boolean }
export const DEFAULT_CONFIG: GameConfig = { seed: 4242, mode: 'classic', creative: true, darknessDamage: true };
export const HOTBAR = 8;
export const INV_SIZE = 24;

export type GameEvents = {
  toast: { text: string };
  harvest: { item: Id; qty: number; x: number; y: number };
  hit: { x: number; y: number; kind: string };
  death: Record<string, never>;
  storage: { entity: Entity };
  vehicleTrunk: { entity: Entity };
  built: { id: Id; entity: Entity };
  sleep: Record<string, never>;
  overseer: Record<string, never>;
  faction: Record<string, never>;
  gunsmith: { entity?: Entity };
  zombieAttack: { zombie: Entity; damage: number };
  zombieKilled: { kind: string; x: number; y: number };
  noise: { x: number; y: number; radius: number };
  petBark: { pet: Entity; x: number; y: number };
  animalProduced: { animal: Entity; item: string; x: number; y: number };
  animalKilled: { kind: string; x: number; y: number };
  snapCamera: { x: number; y: number };
};
export interface SimInput {
  move: { x: number; y: number };
  sprint: boolean;
  actionPressed: boolean;
  interactPressed: boolean;
  aimHeld?: boolean;
  aimWorld?: { x: number; y: number };
  shootPressed?: boolean;
}
export const NO_INPUT: SimInput = { move: { x: 0, y: 0 }, sprint: false, actionPressed: false, interactPressed: false };
export type TownDistrict = 'plaza' | 'residential' | 'commercial' | 'industrial';

export type Target =
  | { kind: 'object'; id: string; obj: WorldObject; prompt: string; key: 'SPACE'; ok: boolean; need?: ToolKind }
  | { kind: 'structure'; entity: Entity; prompt: string; key: 'E' }
  | { kind: 'vehicle'; entity: Entity; prompt: string; key: 'E'; part: 'cabin' | 'trunk' | 'exit' }
  | { kind: 'zombie'; entity: Entity; prompt: string; key: 'SPACE' }
  | { kind: 'pet'; entity: Entity; prompt: string; key: 'E' }
  | { kind: 'animal'; entity: Entity; prompt: string; key: 'E' | 'SPACE'; action?: 'pet' | 'milk' | 'shear' | 'eggs' | 'feed' | 'hunt' }
  | null;

export interface ArenaSettings {
  ammoConsumption: boolean;
  noiseAttractsZombies: boolean;
  damageFalloff: boolean;
  arenaCoverBlocksBullets: boolean;
  botsFightEachOther: boolean;
  botsTargetPlayer: boolean;
  allyFriendlyFire: boolean;
}

export interface SaveData {
  v: 1; seed: number; mode: GameConfig['mode']; t: number; day: number; alwaysDay?: boolean;
  player: { x: number; y: number; angle: number; look: Look; inv: unknown; held: number; needs: { hunger: number; health: number; sanity: number } };
  structures: { id: Id; tx: number; ty: number; open: boolean; storage?: unknown; conveyor?: unknown; inserter?: unknown; machine?: unknown }[];
  vehicles?: { id: Id; x: number; y: number; angle: number; hp: number; fuel: number; trunk?: unknown }[];
  zombies?: { id: Id; x: number; y: number; angle: number; hp: number }[];
  pets?: { name: string; kind: 'wolf_pup' | 'hunting_dog'; x: number; y: number; angle: number; hp: number }[];
  animals?: { kind: AnimalKind; name: string; x: number; y: number; angle: number; hp: number; tamed: boolean; produceTimer: number }[];
  colonists?: { name: string; look: unknown; x: number; y: number; angle: number; pawn: unknown }[];
  colony?: unknown;
  siege?: unknown;
  faction?: unknown;
  buffs?: unknown;
  explored?: string[];
  mods: [string, ObjectMod][];
  staticMap?: StaticMapSave;
  staticActive?: boolean;
  baseStorage?: unknown;
  gunMods?: Record<string, GunAttachments>;
}

const NAMES = ['Wren', 'Moss', 'Ash', 'Birch', 'Sorrel', 'Pike', 'Juniper', 'Cinder', 'Marlow', 'Tansy'];

export class Game {
  readonly ecs = new Ecs();
  readonly events = new Events<GameEvents>();
  readonly content: Content;
  readonly world: GameWorld;
  readonly base: BaseSystem;
  readonly clock = new Clock();
  readonly rng: Rng;
  readonly explored = new Set<string>();
  fogOfWar = false;
  readonly colony = new ColonySystem();
  readonly siege = new SiegeSystem();
  readonly faction = new FactionSystem();
  readonly buffs = {
    hiccupTimer: 0,
    fireBreathTimer: 0,
    rainbowGlowTimer: 0,
    speedBoostTimer: 0,
    buoyancyTimer: 0,
  };
  readonly activeFlares: { x: number; y: number; timer: number }[] = [];
  readonly baseStorage: Inventory;
  readonly arenaSettings: ArenaSettings = {
    ammoConsumption: true,
    noiseAttractsZombies: true,
    damageFalloff: true,
    arenaCoverBlocksBullets: true,
    botsFightEachOther: true,
    botsTargetPlayer: true,
    allyFriendlyFire: false,
  };
  activeArena: { def: ArenaDef; originX: number; originY: number } | null = null;
  player: Entity;
  simTime = 0;
  private evictTimer = 0;
  private zombieSpawnTimer = 0;
  private wildlifeSpawnTimer = 0;

  constructor(readonly config: GameConfig = DEFAULT_CONFIG, packs?: ContentPack[], spawnStarterVehicle = true, spawnStarterPet = true, spawnStarterAnimals = spawnStarterVehicle && spawnStarterPet) {
    this.content = buildContent(packs);
    this.baseStorage = new Inventory(this.content, 30);
    this.rng = makeRng(config.seed);
    this.world = new GameWorld(this.content, config.seed);
    if (config.alwaysDay !== undefined) {
      this.clock.alwaysDay = config.alwaysDay;
      if (config.alwaysDay) this.clock.t = 0.25;
    }
    this.base = new BaseSystem(this.ecs, this.world, this.content);
    this.world.blocker = (x, y, r) => this.base.solidAt(x, y, r);
    this.player = this.spawnPlayer();
    this.spawnNpcs(3);
    this.spawnShootingRangeTargets();
    this.revealArea(this.world.spawn.x, this.world.spawn.y, 52);
    const starterSlots = this.townCompanionSlots('plaza');
    if (spawnStarterVehicle) {
      this.spawnVehicle(starterSlots.vehicle.x, starterSlots.vehicle.y);
    }
    if (spawnStarterPet) {
      this.spawnPet(starterSlots.pet.x, starterSlots.pet.y, 'Barkley', 'hunting_dog');
    }
    if (spawnStarterAnimals) {
      this.spawnAnimal(this.world.spawn.x + 3.2, this.world.spawn.y + 1.8, 'cow', 'Daisy');
      this.spawnAnimal(this.world.spawn.x + 4.5, this.world.spawn.y + 2.4, 'cow', 'Bessie');
      this.spawnAnimal(this.world.spawn.x + 2.8, this.world.spawn.y + 3.2, 'sheep', 'Fluffy');
      this.spawnAnimal(this.world.spawn.x + 3.6, this.world.spawn.y + 3.8, 'sheep', 'Cotton');
      this.spawnAnimal(this.world.spawn.x + 2.0, this.world.spawn.y + 4.5, 'pig', 'Barnaby');
      this.spawnAnimal(this.world.spawn.x + 1.4, this.world.spawn.y + 5.0, 'pig', 'Truffles');
      this.spawnAnimal(this.world.spawn.x + 3.0, this.world.spawn.y + 0.8, 'chicken', 'Clucky');
      this.spawnAnimal(this.world.spawn.x + 3.8, this.world.spawn.y + 0.4, 'chicken', 'Henrietta');
      this.spawnAnimal(this.world.spawn.x - 4.5, this.world.spawn.y + 4.0, 'deer', 'Bambi');
      this.spawnAnimal(this.world.spawn.x - 3.2, this.world.spawn.y + 2.5, 'rabbit', 'Thumper');
      this.spawnAnimal(this.world.spawn.x + 5.5, this.world.spawn.y - 2.0, 'goose', 'Honkers');
    }
    if (config.mode === 'zombie') {
      const zr = this.rng.fork('init_zombies');
      for (let i = 0; i < 5; i++) {
        const ang = zr.range(0, Math.PI * 2), dist = zr.range(12, 22);
        const zx = this.world.spawn.x + Math.cos(ang) * dist;
        const zy = this.world.spawn.y + Math.sin(ang) * dist;
        const kind = i === 0 ? 'brute' : i % 2 === 1 ? 'runner' : 'walker';
        this.spawnZombie(zx, zy, kind);
      }
    }
  }

  // ---------------------------------------------------------------- spawning
  private spawnPlayer(): Entity {
    const e = this.ecs.create();
    const sp = this.townAnchor('plaza');
    this.ecs.add(e, Comp.Transform, { x: sp.x, y: sp.y, angle: sp.angle });
    this.ecs.add(e, Comp.Body, { radius: 0.28, speed: 3.2, vx: 0, vy: 0 });
    this.ecs.add(e, Comp.Character, { name: 'You', look: defaultLook(), anim: newAnim() });
    this.ecs.add(e, Comp.Needs, { hunger: 85, health: 100, sanity: 100, maxHunger: 100, maxHealth: 100, maxSanity: 100, darkTimer: 0 });
    const inv = new Inventory(this.content, INV_SIZE);
    for (const [item, qty] of Object.entries({ axe: 1, pickaxe: 1, torch: 1, berries: 5, wood: 14, stone: 8, twig: 6, flint: 3, cloth: 4, scrap_metal: 4, fiber: 3 })) inv.add(item, qty);
    this.ecs.add(e, Comp.Inventory, { inv, held: 0 });
    this.ecs.add(e, Comp.Player, true);
    this.holdSlot(0, e);
    return e;
  }
  private spawnNpcs(n: number): void {
    const r = this.rng.fork('npcs');
    for (let i = 0; i < n; i++) {
      const e = this.ecs.create();
      const a = r.range(0, Math.PI * 2), d = r.range(3.5, 7);
      this.ecs.add(e, Comp.Transform, { x: this.world.spawn.x + Math.cos(a) * d, y: this.world.spawn.y + Math.sin(a) * d, angle: r.range(0, 6.28) });
      this.ecs.add(e, Comp.Body, { radius: 0.28, speed: 1.3, vx: 0, vy: 0 });
      this.ecs.add(e, Comp.Character, { name: NAMES[(i * 3 + 1) % NAMES.length] as string, look: randomLook(this.content, r), anim: newAnim() });
      this.ecs.add(e, Comp.Wander, { target: null, wait: r.range(0, 3) });
      this.ecs.add(e, Comp.Pawn, createRandomPawn(r));
    }
  }

  spawnColonist(x: number, y: number, name?: string): Entity {
    const r = this.rng.fork(`colonist_${Math.floor(this.simTime * 100)}`);
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle: r.range(0, Math.PI * 2) });
    this.ecs.add(e, Comp.Body, { radius: 0.28, speed: 1.4, vx: 0, vy: 0 });
    const chosenName = name ?? NAMES[Math.floor(r.range(0, NAMES.length))];
    this.ecs.add(e, Comp.Character, { name: chosenName, look: randomLook(this.content, r), anim: newAnim() });
    this.ecs.add(e, Comp.Wander, { target: null, wait: r.range(0, 2) });
    this.ecs.add(e, Comp.Pawn, createRandomPawn(r));
    return e;
  }

  // ---------------------------------------------------------------- accessors
  get playerT() { return this.ecs.get(this.player, Comp.Transform)!; }
  get playerChar() { return this.ecs.get(this.player, Comp.Character)!; }
  get playerNeeds() { return this.ecs.get(this.player, Comp.Needs)!; }
  get playerInv() { return this.ecs.get(this.player, Comp.Inventory)!; }
  get playerBody() { return this.ecs.get(this.player, Comp.Body)!; }
  gunCooldown = 0;
  aimActive = false;
  aimWorld: { x: number; y: number } | null = null;
  get shootingRangePos() { return { x: this.world.spawn.x - 23.5, y: this.world.spawn.y - 12 }; }

  private townAnchor(district: TownDistrict = 'plaza'): { x: number; y: number; angle: number } {
    const sp = this.world.spawn;
    if (district === 'commercial') return { x: sp.x + 16, y: sp.y, angle: Math.PI };
    if (district === 'industrial') return { x: sp.x - 16, y: sp.y, angle: 0 };
    if (district === 'residential') return { x: sp.x, y: sp.y + 16, angle: -Math.PI / 2 };
    return { x: sp.x, y: sp.y, angle: 0 };
  }

  private townCompanionSlots(district: TownDistrict = 'plaza'): { vehicle: { x: number; y: number; angle: number }; pet: { x: number; y: number; angle: number } } {
    const a = this.townAnchor(district);
    return {
      vehicle: { x: a.x + 1.9, y: a.y + 0.3, angle: a.angle },
      pet: { x: a.x - 1.1, y: a.y + 0.8, angle: a.angle },
    };
  }

  gunMods: Record<string, GunAttachments> = {
    hunting_rifle: { optic: 'optic_rangefinder', muzzle: null, underbarrel: null, magazine: null, goofy: null },
    pump_shotgun: { optic: null, muzzle: 'muzzle_soup_can', underbarrel: null, magazine: null, goofy: null },
    nailgun_smg: { optic: null, muzzle: null, underbarrel: 'under_cat_laser', magazine: null, goofy: 'goofy_rubber_duck' },
    potato_cannon: { optic: 'optic_googly', muzzle: null, underbarrel: null, magazine: null, goofy: null },
    slingshot: { optic: null, muzzle: null, underbarrel: null, magazine: null, goofy: null },
  };

  getWeaponMods(weaponId: string): GunAttachments {
    if (!this.gunMods[weaponId]) {
      this.gunMods[weaponId] = { optic: null, muzzle: null, underbarrel: null, magazine: null, goofy: null };
    }
    return this.gunMods[weaponId]!;
  }

  setWeaponAttachment(weaponId: string, slot: AttachmentSlot, attachmentId: string | null): void {
    const mods = this.getWeaponMods(weaponId);
    mods[slot] = attachmentId;
  }

  getEffectiveGun(weaponId: string): GunDef & {
    visualInfo: VisualInfoMods;
    meleeDmg: number;
    squeakSound: boolean;
    honkSound: boolean;
    recoilMod: number;
    activeAttachments: GunAttachments;
  } {
    const def = this.content.items.get(weaponId)?.gun;
    const base: GunDef = def ? { ...def } : {
      ammoItem: 'pebbles',
      ammoPerShot: 1,
      aimSystem: 'bloom',
      projectileSpeed: 20,
      maxRange: 20,
      baseDamage: 10,
      minDamage: 5,
      falloffStartDist: 8,
      spreadAngle: 0.1,
      fireRate: 1,
      noiseRadius: 15,
    };

    const mods = this.getWeaponMods(weaponId);
    let damageBonus = 0;
    let rangeBonus = 0;
    let spreadMul = 1.0;
    let noiseMul = 1.0;
    let fireRateMul = 1.0;
    let recoilMul = 1.0;
    let meleeDmg = 15;
    let squeak = false;
    let honk = false;
    const vInfo: VisualInfoMods = {};

    for (const slotKey of ['optic', 'muzzle', 'underbarrel', 'magazine', 'goofy'] as const) {
      const attId = mods[slotKey];
      if (!attId) continue;
      const att = this.content.attachments.get(attId);
      if (!att) continue;
      if (att.damageMod) damageBonus += att.damageMod;
      if (att.rangeMod) rangeBonus += att.rangeMod;
      if (att.spreadMod) spreadMul *= att.spreadMod;
      if (att.noiseMod) noiseMul *= att.noiseMod;
      if (att.fireRateMod) fireRateMul *= att.fireRateMod;
      if (att.recoilMod) recoilMul *= att.recoilMod;
      if (att.meleeDmg && att.meleeDmg > meleeDmg) meleeDmg = att.meleeDmg;
      if (att.squeakSound) squeak = true;
      if (att.honkSound) honk = true;
      if (att.visualInfo) {
        if (att.visualInfo.rangefinder) vInfo.rangefinder = true;
        if (att.visualInfo.targetHealthBar) vInfo.targetHealthBar = true;
        if (att.visualInfo.threatScanner) vInfo.threatScanner = true;
        if (att.visualInfo.trajectoryArc) vInfo.trajectoryArc = true;
        if (att.visualInfo.nightVisionReticle) vInfo.nightVisionReticle = true;
      }
    }

    return {
      ...base,
      baseDamage: Math.max(1, base.baseDamage + damageBonus),
      minDamage: Math.max(1, base.minDamage + damageBonus),
      maxRange: Math.max(5, base.maxRange + rangeBonus),
      spreadAngle: Math.max(0.005, base.spreadAngle * spreadMul),
      noiseRadius: Math.max(1, base.noiseRadius * noiseMul),
      fireRate: Math.max(0.2, base.fireRate * fireRateMul),
      recoilMod: recoilMul,
      visualInfo: vInfo,
      meleeDmg,
      squeakSound: squeak,
      honkSound: honk,
      activeAttachments: mods,
    };
  }

  heldTool(e: Entity = this.player): { kind: ToolKind; power: number; speed: number; reach: number; damage: number } | null {
    const inst = this.ecs.get(e, Comp.Character)?.look.outfit.mainHand;
    return (inst && this.content.items.get(inst.item)?.tool) || null;
  }

  // ---------------------------------------------------------------- vehicles
  spawnVehicle(x: number, y: number, defId: Id = 'rusty_hatchback'): Entity {
    const def = this.content.vehicles.get(defId);
    if (!def) throw new Error(`Unknown vehicle ${defId}`);
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle: 0 });
    this.ecs.add(e, Comp.Body, { radius: 0.75, speed: 0, vx: 0, vy: 0 });
    const trunk = new Inventory(this.content, def.trunkSlots);
    this.ecs.add(e, Comp.Vehicle, {
      def,
      driver: null,
      fuel: def.maxFuel * 0.8,
      hp: def.maxHp,
      speed: 0,
      steer: 0,
      headlights: false,
      trunk,
    });
    this.ecs.add(e, Comp.Light, { radius: def.lightRadius, color: '#fde58a', flicker: 0.04 });
    return e;
  }

  spawnFleet(baseX = this.world.spawn.x, baseY = this.world.spawn.y): Entity[] {
    const fleet: { id: Id; dx: number; dy: number; angle: number }[] = [
      { id: 'clown_car', dx: -3.2, dy: 0.2, angle: 0 },
      { id: 'bathtub_buggy', dx: -4.6, dy: -1.0, angle: 0 },
      { id: 'golf_cart', dx: -2.0, dy: 2.8, angle: 1.57 },
      { id: 'ice_cream_truck', dx: 3.4, dy: 3.2, angle: 1.57 },
      { id: 'hot_dog_van', dx: 3.4, dy: -3.2, angle: -1.57 },
      { id: 'tesla_roadster', dx: -4.8, dy: 1.2, angle: 0 },
    ];
    const out: Entity[] = [];
    for (const f of fleet) {
      const existing = this.ecs.query(Comp.Vehicle, Comp.Transform).some((e) => {
        const v = this.ecs.get(e, Comp.Vehicle)!, vt = this.ecs.get(e, Comp.Transform)!;
        return v.def.id === f.id && Math.hypot(vt.x - (baseX + f.dx), vt.y - (baseY + f.dy)) < 1.0;
      });
      if (!existing) {
        const ve = this.spawnVehicle(baseX + f.dx, baseY + f.dy, f.id);
        const vt = this.ecs.get(ve, Comp.Transform)!;
        vt.angle = f.angle;
        out.push(ve);
      }
    }
    return out;
  }

  drivingVehicle(e: Entity = this.player): Entity | null {
    for (const veh of this.ecs.query(Comp.Vehicle)) {
      if (this.ecs.get(veh, Comp.Vehicle)?.driver === e) return veh;
    }
    return null;
  }

  refuelVehicle(veh: Entity): boolean {
    const v = this.ecs.get(veh, Comp.Vehicle);
    if (!v) return false;
    if (v.fuel >= v.def.maxFuel) { this.events.emit('toast', { text: 'Fuel tank is already full' }); return false; }
    if (!this.playerInv.inv.remove('fuel_canister', 1)) { this.events.emit('toast', { text: 'Need a Fuel Canister' }); return false; }
    v.fuel = Math.min(v.def.maxFuel, v.fuel + 40);
    this.events.emit('toast', { text: `Refueled ${v.def.name} (${Math.round(v.fuel)}%)` });
    return true;
  }

  repairVehicle(veh: Entity): boolean {
    const v = this.ecs.get(veh, Comp.Vehicle);
    if (!v) return false;
    if (v.hp >= v.def.maxHp) { this.events.emit('toast', { text: 'Vehicle is in mint condition' }); return false; }
    if (!this.playerInv.inv.remove('scrap_metal', 2)) { this.events.emit('toast', { text: 'Need 2 Scrap Metal to repair' }); return false; }
    v.hp = Math.min(v.def.maxHp, v.hp + 25);
    this.events.emit('toast', { text: `Repaired ${v.def.name} (${Math.round(v.hp)} HP)` });
    return true;
  }

  private controlVehicle(veh: Entity, dt: number, input: SimInput): void {
    const v = this.ecs.get(veh, Comp.Vehicle)!, vt = this.ecs.get(veh, Comp.Transform)!, vb = this.ecs.get(veh, Comp.Body)!;
    const pt = this.playerT, pb = this.ecs.get(this.player, Comp.Body)!;
    const m = Math.hypot(input.move.x, input.move.y);

    let throttle = 0;
    let steerInput = 0;

    // Direct arcade car controls (W: forward, S: reverse/brake, A: left, D: right)
    if (input.move.y < -0.1) throttle += 1;
    if (input.move.y > 0.1) throttle -= 0.8;
    if (input.move.x < -0.1) steerInput -= 1;
    if (input.move.x > 0.1) steerInput += 1;

    // Vector fallback (e.g. mobile stick or directional test input):
    const forwardX = Math.cos(vt.angle), forwardY = Math.sin(vt.angle);
    const rightX = -Math.sin(vt.angle), rightY = Math.cos(vt.angle);
    const dotFwd = input.move.x * forwardX + input.move.y * forwardY;
    const dotRight = input.move.x * rightX + input.move.y * rightY;

    if (throttle === 0 && Math.abs(dotFwd) > 0.15) {
      throttle = dotFwd > 0 ? 1 : -0.8;
    }
    if (steerInput === 0 && Math.abs(dotRight) > 0.15) {
      steerInput = clamp(dotRight * 1.5, -1, 1);
    }

    if (m > 0.05 || Math.abs(throttle) > 0 || Math.abs(steerInput) > 0) {
      const targetSteer = clamp(steerInput * 0.65, -0.65, 0.65);
      v.steer = damp(v.steer, targetSteer, 14, dt);

      if (v.fuel > 0 && v.hp > 0) {
        const topSpeed = v.def.maxSpeed * (input.sprint ? 1.25 : 1);
        if (throttle > 0) {
          v.speed = clamp(v.speed + v.def.accel * dt, -topSpeed * 0.45, topSpeed);
          v.fuel = Math.max(0, v.fuel - dt * 0.12);
        } else if (throttle < 0) {
          v.speed = clamp(v.speed - v.def.accel * 0.85 * dt, -topSpeed * 0.45, topSpeed);
          v.fuel = Math.max(0, v.fuel - dt * 0.08);
        } else {
          v.speed = damp(v.speed, 0, 3.5, dt);
        }
      } else {
        v.speed = damp(v.speed, 0, 4.5, dt);
      }
    } else {
      v.steer = damp(v.steer, 0, 8, dt);
      v.speed = damp(v.speed, 0, 4.5, dt);
    }

    if (Math.abs(v.speed) > 0.05) {
      vt.angle += v.steer * (v.speed / v.def.maxSpeed) * v.def.turnSpeed * dt;
    }

    vb.vx = Math.cos(vt.angle) * v.speed;
    vb.vy = Math.sin(vt.angle) * v.speed;

    const k = this.world.speedAt(vt.x, vt.y) * dt;
    const nx = vt.x + vb.vx * k, ny = vt.y + vb.vy * k;

    if (!this.world.blocked(nx, vt.y, vb.radius)) {
      vt.x = nx;
    } else {
      this.handleVehicleObstacle(veh, nx, vt.y);
      v.speed *= 0.15; vb.vx = 0;
    }

    if (!this.world.blocked(vt.x, ny, vb.radius)) {
      vt.y = ny;
    } else {
      this.handleVehicleObstacle(veh, vt.x, ny);
      v.speed *= 0.15; vb.vx = 0;
    }

    // Vehicle ramming zombies and animals
    if (Math.abs(v.speed) > 1.8) {
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - vt.x, zt.y - vt.y) < vb.radius + 0.45) {
          const ramDmg = Math.round(Math.abs(v.speed) * 8);
          this.damageZombie(ze, ramDmg);
          v.speed *= 0.65;
          v.hp = Math.max(0, v.hp - 1);
        }
      }
      for (const ae of this.ecs.query(Comp.Animal, Comp.Transform, Comp.Body)) {
        const at = this.ecs.get(ae, Comp.Transform)!, ab = this.ecs.get(ae, Comp.Body)!;
        if (Math.hypot(at.x - vt.x, at.y - vt.y) < vb.radius + ab.radius + 0.2) {
          const ramDmg = Math.round(Math.abs(v.speed) * 7);
          this.damageAnimal(ae, ramDmg);
          ab.vx += Math.cos(vt.angle) * 7;
          ab.vy += Math.sin(vt.angle) * 7;
          v.speed *= 0.8;
        }
      }
    }
    if (Math.abs(v.speed) > 1.2) {
      this.emitNoise(vt.x, vt.y, 14);
    }

    v.headlights = this.clock.darkness > 0.32;
    pt.x = vt.x; pt.y = vt.y; pt.angle = vt.angle;
    pb.vx = vb.vx; pb.vy = vb.vy;
  }

  private handleVehicleObstacle(veh: Entity, x: number, y: number): void {
    const v = this.ecs.get(veh, Comp.Vehicle);
    if (!v || Math.abs(v.speed) < 2.0) return;
    const nearby = this.world.objectsNear(x, y, 1.8).find((o) => {
      const def = this.content.objects.get(o.kind);
      return def && def.solid;
    });
    if (nearby) {
      const { destroyed } = this.world.damageObject(nearby.id, 2);
      this.events.emit('hit', { x: nearby.x, y: nearby.y, kind: nearby.kind });
      v.hp = Math.max(0, v.hp - 2);
      if (destroyed) {
        this.events.emit('toast', { text: `Smashed through ${this.content.objects.get(nearby.kind)?.name ?? 'obstacle'}!` });
      }
    }
  }

  // ---------------------------------------------------------------- zombies
  spawnZombie(x: number, y: number, defId: Id = 'walker'): Entity {
    const def = this.content.zombies.get(defId) ?? this.content.zombies.get('walker')!;
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle: this.rng.range(0, 6.28) });
    this.ecs.add(e, Comp.Body, { radius: 0.32, speed: def.speed, vx: 0, vy: 0 });
    this.ecs.add(e, Comp.Zombie, {
      def,
      hp: def.maxHp,
      state: 'idle',
      target: null,
      alertPos: null,
      attackTimer: 0,
      groanTimer: this.rng.range(2, 8),
      animPhase: this.rng.range(0, 6.28),
      hitFlash: 0,
    });
    return e;
  }

  emitNoise(x: number, y: number, radius: number): void {
    this.events.emit('noise', { x, y, radius });
    for (const e of this.ecs.query(Comp.Zombie, Comp.Transform)) {
      const z = this.ecs.get(e, Comp.Zombie)!, zt = this.ecs.get(e, Comp.Transform)!;
      const d = Math.hypot(zt.x - x, zt.y - y);
      if (d < radius && z.state !== 'chase') {
        z.state = 'alert';
        z.alertPos = { x, y };
      }
    }
  }

  damageZombie(entity: Entity, dmg: number): boolean {
    const z = this.ecs.get(entity, Comp.Zombie), zt = this.ecs.get(entity, Comp.Transform);
    if (!z || !zt) return false;
    z.hp -= dmg;
    z.hitFlash = 0.35;
    z.state = 'chase';
    z.target = this.player;
    this.events.emit('hit', { x: zt.x, y: zt.y, kind: 'flesh' });
    this.emitNoise(zt.x, zt.y, 8);
    const pt = this.playerT;
    const dx = zt.x - pt.x, dy = zt.y - pt.y, d = Math.hypot(dx, dy);
    if (d > 0.05) {
      zt.x += (dx / d) * 0.28;
      zt.y += (dy / d) * 0.28;
    }
    if (z.hp <= 0) {
      this.events.emit('zombieKilled', { kind: z.def.kind, x: zt.x, y: zt.y });
      this.events.emit('toast', { text: `Defeated ${z.def.name}!` });
      const r = this.rng.fork(`zdrop:${entity}`);
      const inv = this.playerInv.inv;
      for (const drop of z.def.drops) {
        if (r.next() <= drop.chance) {
          const qty = r.int(drop.min, drop.max);
          const left = inv.add(drop.item, qty);
          if (left < qty) this.events.emit('harvest', { item: drop.item, qty: qty - left, x: zt.x, y: zt.y });
        }
      }
      this.ecs.destroy(entity);
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- pets
  spawnPet(x: number, y: number, name = 'Barkley', kind: 'wolf_pup' | 'hunting_dog' = 'hunting_dog'): Entity {
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle: 0 });
    this.ecs.add(e, Comp.Body, { radius: 0.22, speed: 3.6, vx: 0, vy: 0 });
    this.ecs.add(e, Comp.Pet, {
      name,
      kind,
      hp: 80,
      maxHp: 80,
      owner: this.player,
      state: 'follow',
      target: null,
      barkTimer: 2.0,
      biteTimer: 0,
      animPhase: 0,
      tailWag: 0,
    });
    return e;
  }

  private updatePets(dt: number): void {
    const pt = this.playerT;
    const drivingVeh = this.drivingVehicle();
    const leaderT = drivingVeh ? (this.ecs.get(drivingVeh, Comp.Transform) ?? pt) : pt;

    for (const e of this.ecs.query(Comp.Pet, Comp.Transform, Comp.Body)) {
      const pet = this.ecs.get(e, Comp.Pet)!, t = this.ecs.get(e, Comp.Transform)!, b = this.ecs.get(e, Comp.Body)!;
      pet.barkTimer -= dt;
      pet.biteTimer -= dt;

      // Scan for nearest zombie to protect owner
      let nearestZombie: Entity | null = null;
      let nearestDist = 6.5;
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        const zd = Math.hypot(zt.x - t.x, zt.y - t.y);
        if (zd < nearestDist) {
          nearestDist = zd;
          nearestZombie = ze;
        }
      }

      if (nearestZombie) {
        if (pet.state !== 'attack') {
          pet.barkTimer = 0;
        }
        pet.state = 'attack';
        pet.target = nearestZombie;
        const zt = this.ecs.get(nearestZombie, Comp.Transform)!;
        const zdx = zt.x - t.x, zdy = zt.y - t.y, zd = Math.hypot(zdx, zdy);
        t.angle = lerpAngle(t.angle, Math.atan2(zdy, zdx), 1 - Math.exp(-12 * dt));
        b.vx = Math.cos(t.angle) * 4.2;
        b.vy = Math.sin(t.angle) * 4.2;

        if (pet.barkTimer <= 0) {
          pet.barkTimer = this.rng.range(1.6, 2.8);
          this.events.emit('petBark', { pet: e, x: t.x, y: t.y });
          this.emitNoise(t.x, t.y, 10);
        }

        if (zd < 0.65 && pet.biteTimer <= 0) {
          pet.biteTimer = 0.9;
          this.damageZombie(nearestZombie, 6);
          pet.tailWag += 5;
        }
      } else {
        pet.state = 'follow';
        pet.target = null;
        const dx = leaderT.x - t.x, dy = leaderT.y - t.y, dist = Math.hypot(dx, dy);

        if (dist > 1.3) {
          t.angle = lerpAngle(t.angle, Math.atan2(dy, dx), 1 - Math.exp(-10 * dt));
          const chaseSpeed = dist > 6.0 ? 5.2 : 3.4;
          b.vx = Math.cos(t.angle) * chaseSpeed;
          b.vy = Math.sin(t.angle) * chaseSpeed;
          pet.tailWag += dt * 4;
        } else {
          b.vx = damp(b.vx, 0, 8, dt);
          b.vy = damp(b.vy, 0, 8, dt);
          pet.tailWag += dt * 8;
        }
      }

      // Obstacle sliding
      if (b.vx !== 0 || b.vy !== 0) {
        const k = this.world.speedAt(t.x, t.y) * dt;
        const nx = t.x + b.vx * k, ny = t.y + b.vy * k;
        if (!this.world.blocked(nx, t.y, b.radius)) t.x = nx; else b.vx = 0;
        if (!this.world.blocked(t.x, ny, b.radius)) t.y = ny; else b.vy = 0;
      }
    }
  }

  spawnAnimal(x: number, y: number, kind: AnimalKind = 'cow', name?: string): Entity {
    const def = this.content.animals.get(kind);
    if (!def) throw new Error(`Unknown animal ${kind}`);
    const r = this.rng.fork(`animal_${Math.floor(this.simTime * 100 + x * 10 + y)}`);
    const defaultNames: Record<AnimalKind, string[]> = {
      cow: ['Daisy', 'Bessie', 'Buttercup', 'Clarabelle', 'Moo-Moo'],
      sheep: ['Fluffy', 'Cotton', 'Woolly', 'Bo-Peep', 'Snowball'],
      pig: ['Barnaby', 'Hamlet', 'Truffles', 'Porky', 'Waddles'],
      chicken: ['Clucky', 'Henrietta', 'Penny', 'Nugget', 'Feathers'],
      deer: ['Bambi', 'Fawn', 'Buck', 'Swift', 'Willow'],
      rabbit: ['Thumper', 'Hoppy', 'Clover', 'Pip', 'Bun-Bun'],
      goose: ['Honkers', 'Gander', 'Waddles', 'Feathers', 'Goosey'],
    };
    const list = defaultNames[kind] ?? ['Critter'];
    const animalName = name ?? list[Math.floor(r.range(0, list.length))];
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle: r.range(0, Math.PI * 2) });
    this.ecs.add(e, Comp.Body, { radius: def.radius, speed: def.speed, vx: 0, vy: 0 });
    this.ecs.add(e, Comp.Animal, {
      def,
      name: animalName,
      kind,
      hp: def.maxHp,
      maxHp: def.maxHp,
      state: 'graze',
      fleeTimer: 0,
      grazeTimer: r.range(2, 6),
      produceTimer: def.produce?.cooldown ? r.range(def.produce.cooldown * 0.4, def.produce.cooldown) : 0,
      animPhase: r.range(0, 6.28),
      eating: false,
      tamed: false,
      alertPos: null,
      target: null,
      tailWag: r.range(0, 6.28),
    });
    return e;
  }

  spawnWildlifeHerd(centerX: number, centerY: number, kind: AnimalKind, count = 3): Entity[] {
    const list: Entity[] = [];
    const r = this.rng.fork(`herd_${kind}_${Math.floor(centerX)}_${Math.floor(centerY)}`);
    for (let i = 0; i < count; i++) {
      const ang = r.range(0, Math.PI * 2), dist = r.range(0.8, 3.2);
      const ax = centerX + Math.cos(ang) * dist;
      const ay = centerY + Math.sin(ang) * dist;
      list.push(this.spawnAnimal(ax, ay, kind));
    }
    return list;
  }

  damageAnimal(e: Entity, dmg: number): boolean {
    const a = this.ecs.get(e, Comp.Animal);
    const at = this.ecs.get(e, Comp.Transform);
    if (!a || !at) return false;
    a.hp -= dmg;
    a.state = 'flee';
    a.fleeTimer = 4.0;
    this.events.emit('hit', { x: at.x, y: at.y, kind: a.kind });
    if (a.hp <= 0) {
      const inv = this.playerInv.inv;
      const r = this.rng.fork(`drop_${a.kind}_${Math.floor(at.x * 10 + at.y * 10)}`);
      for (const d of a.def.drops) {
        if (r.next() <= d.chance) {
          const qty = r.int(d.min, d.max);
          if (qty > 0) {
            const left = inv.add(d.item, qty);
            if (left < qty) this.events.emit('harvest', { item: d.item, qty: qty - left, x: at.x, y: at.y });
          }
        }
      }
      this.events.emit('animalKilled', { kind: a.kind, x: at.x, y: at.y });
      this.events.emit('toast', { text: `${a.name} dropped raw meat and materials!` });
      this.ecs.destroy(e);
      return true;
    }
    return false;
  }

  private updateAnimals(dt: number): void {
    const pt = this.playerT;
    const drivingVeh = this.drivingVehicle();
    const vehBody = drivingVeh ? this.ecs.get(drivingVeh, Comp.Body) : null;
    const vehSpeed = vehBody ? Math.hypot(vehBody.vx, vehBody.vy) : 0;

    for (const e of this.ecs.query(Comp.Animal, Comp.Transform, Comp.Body)) {
      const a = this.ecs.get(e, Comp.Animal)!, at = this.ecs.get(e, Comp.Transform)!, ab = this.ecs.get(e, Comp.Body)!;

      // Production cooldown
      if (a.def.produce) {
        a.produceTimer -= dt;
      }

      // Check for threats: zombies, speeding cars, or close untamed players
      let threatX = 0, threatY = 0, threatFound = false;

      // Check zombies
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        const zd = Math.hypot(zt.x - at.x, zt.y - at.y);
        if (zd < a.def.fleeDistance) {
          threatX = zt.x; threatY = zt.y;
          threatFound = true;
          break;
        }
      }

      // Check speeding car or close untamed player
      if (!threatFound) {
        const pd = Math.hypot(pt.x - at.x, pt.y - at.y);
        if (vehSpeed > 3.0 && pd < a.def.fleeDistance * 1.2) {
          threatX = pt.x; threatY = pt.y; threatFound = true;
        } else if (!a.tamed && pd < a.def.fleeDistance * 0.5) {
          threatX = pt.x; threatY = pt.y; threatFound = true;
        }
      }

      if (threatFound) {
        a.state = 'flee';
        a.fleeTimer = 3.5;
        a.eating = false;
        const awayX = at.x - threatX, awayY = at.y - threatY;
        at.angle = lerpAngle(at.angle, Math.atan2(awayY, awayX), 1 - Math.exp(-10 * dt));
        ab.vx = Math.cos(at.angle) * a.def.runSpeed;
        ab.vy = Math.sin(at.angle) * a.def.runSpeed;
      } else if (a.state === 'flee') {
        a.fleeTimer -= dt;
        if (a.fleeTimer <= 0) {
          a.state = 'graze';
          a.grazeTimer = this.rng.range(3, 7);
          ab.vx = 0; ab.vy = 0;
        } else {
          ab.vx = Math.cos(at.angle) * a.def.runSpeed;
          ab.vy = Math.sin(at.angle) * a.def.runSpeed;
        }
      } else {
        // Graze / wander
        a.grazeTimer -= dt;
        if (a.grazeTimer <= 0) {
          a.grazeTimer = this.rng.range(2.5, 6);
          a.eating = !a.eating;
          if (!a.eating) {
            at.angle += this.rng.range(-1.2, 1.2);
            ab.vx = Math.cos(at.angle) * a.def.speed;
            ab.vy = Math.sin(at.angle) * a.def.speed;
          } else {
            ab.vx = 0; ab.vy = 0;
          }
        }
        if (a.eating) {
          ab.vx = damp(ab.vx, 0, 8, dt);
          ab.vy = damp(ab.vy, 0, 8, dt);
        }
      }

      // Obstacle sliding
      if (ab.vx !== 0 || ab.vy !== 0) {
        const k = this.world.speedAt(at.x, at.y) * dt;
        const nx = at.x + ab.vx * k, ny = at.y + ab.vy * k;
        if (!this.world.blocked(nx, at.y, ab.radius)) at.x = nx; else ab.vx = 0;
        if (!this.world.blocked(at.x, ny, ab.radius)) at.y = ny; else ab.vy = 0;
      }
    }
  }

  private updateWildlifeSpawns(dt: number): void {
    this.wildlifeSpawnTimer += dt;
    if (this.wildlifeSpawnTimer >= 18) {
      this.wildlifeSpawnTimer = 0;
      const count = this.ecs.query(Comp.Animal).length;
      if (count < 24) {
        const r = this.rng.fork(`wildlife_spawn_${Math.floor(this.simTime)}`);
        const ang = r.range(0, Math.PI * 2), dist = r.range(18, 30);
        const ax = this.playerT.x + Math.cos(ang) * dist;
        const ay = this.playerT.y + Math.sin(ang) * dist;
        const roll = r.next();
        const kind: AnimalKind = roll < 0.25 ? 'deer' : roll < 0.5 ? 'rabbit' : roll < 0.7 ? 'cow' : roll < 0.85 ? 'sheep' : roll < 0.95 ? 'chicken' : 'goose';
        this.spawnAnimal(ax, ay, kind);
      }
    }
  }

  private updateZombies(dt: number): void {
    const pt = this.playerT, n = this.playerNeeds;
    const isLit = this.isLit(pt.x, pt.y);
    const dark = this.clock.darkness > 0.45;
    const zombieSpeedScale = 0.72;

    for (const e of this.ecs.query(Comp.Zombie, Comp.Transform, Comp.Body)) {
      const z = this.ecs.get(e, Comp.Zombie)!, zt = this.ecs.get(e, Comp.Transform)!, zb = this.ecs.get(e, Comp.Body)!;
      z.groanTimer -= dt;
      z.animPhase += dt * (z.state === 'chase' ? 3.5 : 1.2);
      if (z.groanTimer <= 0) {
        z.groanTimer = this.rng.range(5, 14);
        if (Math.hypot(zt.x - pt.x, zt.y - pt.y) < 18) {
          this.emitNoise(zt.x, zt.y, 6);
        }
      }

      // Detection: player, nearby wildlife, or raider bots
      let nearestTarget = this.player;
      let nearestDist = Math.hypot(pt.x - zt.x, pt.y - zt.y);

      for (const ae of this.ecs.query(Comp.Animal, Comp.Transform)) {
        const at = this.ecs.get(ae, Comp.Transform)!;
        const ad = Math.hypot(at.x - zt.x, at.y - zt.y);
        if (ad < nearestDist) {
          nearestDist = ad;
          nearestTarget = ae;
        }
      }

      for (const re of this.ecs.query(Comp.Raider, Comp.Transform)) {
        const rt = this.ecs.get(re, Comp.Transform)!;
        const rd = Math.hypot(rt.x - zt.x, rt.y - zt.y);
        if (rd < nearestDist) {
          nearestDist = rd;
          nearestTarget = re;
        }
      }

      let detectR = z.def.detectRadius;
      if (dark) {
        detectR = isLit ? z.def.detectRadius * 1.35 : z.def.detectRadius * 0.45;
      }
      if (nearestDist < detectR) {
        z.state = 'chase';
        z.target = nearestTarget;
      }

      // Movement & States
      if (z.state === 'chase' && z.target) {
        const targetT = this.ecs.get(z.target, Comp.Transform);
        if (targetT) {
          const tdx = targetT.x - zt.x, tdy = targetT.y - zt.y, td = Math.hypot(tdx, tdy);
          zt.angle = lerpAngle(zt.angle, Math.atan2(tdy, tdx), 1 - Math.exp(-8 * dt));
          if (td > z.def.attackRadius * 0.85) {
            zb.vx = Math.cos(zt.angle) * z.def.runSpeed * zombieSpeedScale;
            zb.vy = Math.sin(zt.angle) * z.def.runSpeed * zombieSpeedScale;
          } else {
            zb.vx = 0; zb.vy = 0;
          }

          // Attack
          z.attackTimer -= dt;
          if (td <= z.def.attackRadius && z.attackTimer <= 0) {
            z.attackTimer = z.def.attackCooldown;
            if (this.ecs.has(z.target, Comp.Animal)) {
              const a = this.ecs.get(z.target, Comp.Animal)!;
              this.damageAnimal(z.target, z.def.damage);
              this.events.emit('toast', { text: `🧟 ${z.def.name} attacked ${a.name}!` });
            } else if (this.ecs.has(z.target, Comp.Raider)) {
              const r = this.ecs.get(z.target, Comp.Raider)!;
              const ch = this.ecs.get(z.target, Comp.Character);
              r.hp = Math.max(0, r.hp - z.def.damage);
              if (ch) ch.anim.hitFlash = 0.35;
              this.events.emit('hit', { kind: 'flesh', x: targetT.x, y: targetT.y });
              this.events.emit('toast', { text: `🧟 ${z.def.name} mauled ${ch?.name ?? 'Bot'}! (-${z.def.damage} HP)` });
              if (r.hp <= 0) {
                this.events.emit('toast', { text: `💀 ${ch?.name ?? 'Bot'} was killed by ${z.def.name}!` });
                this.ecs.destroy(z.target);
                z.target = null;
                z.state = 'idle';
              }
            } else {
              const drivingVeh = this.drivingVehicle();
              if (drivingVeh) {
                const v = this.ecs.get(drivingVeh, Comp.Vehicle)!;
                v.hp = Math.max(0, v.hp - Math.round(z.def.damage * 0.4));
                this.events.emit('toast', { text: `${z.def.name} attacked vehicle! (${Math.round(v.hp)} HP)` });
              } else {
                const hasPillowVest = this.playerChar.look.outfit.torso?.item === 'pillow_vest';
                const finalDmg = hasPillowVest ? Math.round(z.def.damage * 0.75) : z.def.damage;
                n.health = Math.max(0, n.health - finalDmg);
                this.playerChar.anim.hitFlash = 0.35;
                this.events.emit('zombieAttack', { zombie: e, damage: finalDmg });
                this.events.emit('toast', { text: `Bitten by ${z.def.name}! (-${finalDmg} HP${hasPillowVest ? ' · Pillow cushion!' : ''})` });
                if (n.health <= 0) {
                  this.events.emit('death', {});
                }
              }
            }
          }
        }
      } else if (z.state === 'alert' && z.alertPos) {
        const adx = z.alertPos.x - zt.x, ady = z.alertPos.y - zt.y, ad = Math.hypot(adx, ady);
        if (ad < 0.6) {
          z.state = 'wander';
          z.alertPos = null;
        } else {
          zt.angle = lerpAngle(zt.angle, Math.atan2(ady, adx), 1 - Math.exp(-6 * dt));
          zb.vx = (adx / ad) * z.def.speed * 1.1 * zombieSpeedScale;
          zb.vy = (ady / ad) * z.def.speed * 1.1 * zombieSpeedScale;
        }
      } else {
        zb.vx = damp(zb.vx, 0, 4, dt);
        zb.vy = damp(zb.vy, 0, 4, dt);
      }

      // Obstacle sliding
      if (zb.vx !== 0 || zb.vy !== 0) {
        const k = this.world.speedAt(zt.x, zt.y) * dt;
        const nx = zt.x + zb.vx * k, ny = zt.y + zb.vy * k;
        if (!this.world.blocked(nx, zt.y, zb.radius)) zt.x = nx; else zb.vx = 0;
        if (!this.world.blocked(zt.x, ny, zb.radius)) zt.y = ny; else zb.vy = 0;
      }
    }
  }

  private updateZombieSpawns(dt: number): void {
    if (this.config.mode !== 'zombie') return;
    this.zombieSpawnTimer += dt;
    const isNight = this.clock.darkness > 0.45;
    const interval = isNight ? 12 : 24;
    if (this.zombieSpawnTimer >= interval) {
      this.zombieSpawnTimer = 0;
      const count = this.ecs.query(Comp.Zombie).length;
      if (count < 16) {
        const r = this.rng.fork(`zombie_spawn_${Math.floor(this.simTime)}`);
        const ang = r.range(0, Math.PI * 2), dist = r.range(15, 24);
        const zx = this.playerT.x + Math.cos(ang) * dist;
        const zy = this.playerT.y + Math.sin(ang) * dist;
        const roll = r.next();
        const kind = roll < 0.2 ? 'brute' : roll < 0.55 ? 'runner' : 'walker';
        const ze = this.spawnZombie(zx, zy, kind);
        if (isNight) {
          const z = this.ecs.get(ze, Comp.Zombie)!;
          z.state = 'alert';
          z.alertPos = { x: this.playerT.x, y: this.playerT.y };
        }
      }
    }
  }

  // ---------------------------------------------------------------- main step
  step(dt: number, input: SimInput = NO_INPUT): void {
    dt = Math.min(dt, 0.1);
    this.simTime += dt;
    this.clock.advance(dt);
    this.controlPlayer(dt, input);
    this.colony.update(dt, this);
    this.siege.update(dt, this);
    this.faction.update(dt, this);
    this.updateNpcs(dt);
    this.updatePets(dt);
    this.updateAnimals(dt);
    this.updateWildlifeSpawns(dt);
    this.updateZombies(dt);
    this.updateZombieSpawns(dt);
    this.updateRaiders(dt);
    this.integrate(dt);
    this.updateAnim(dt);
    this.updateActions(dt, input);
    this.updateProjectiles(dt);
    this.updateAutomation(dt);
    this.updateNeeds(dt);
    this.updateBuffs(dt);
    this.evictTimer += dt;
    if (this.evictTimer > 5) { this.evictTimer = 0; this.world.evictFar(this.playerT.x, this.playerT.y); }
    const driving = this.drivingVehicle();
    this.revealArea(this.playerT.x, this.playerT.y, driving ? 70 : 42);
  }

  revealArea(centerX: number, centerY: number, radiusMeters = 250): number {
    const rChunks = Math.ceil(radiusMeters / 16);
    const ccx = Math.floor(centerX / 16), ccy = Math.floor(centerY / 16);
    let count = 0;
    for (let dy = -rChunks; dy <= rChunks; dy++) {
      for (let dx = -rChunks; dx <= rChunks; dx++) {
        if (Math.hypot(dx * 16, dy * 16) <= radiusMeters) {
          const k = `${ccx + dx},${ccy + dy}`;
          if (!this.explored.has(k)) {
            this.explored.add(k);
            count++;
          }
        }
      }
    }
    return count;
  }

  toggleFogOfWar(): boolean {
    this.fogOfWar = !this.fogOfWar;
    return this.fogOfWar;
  }

  setArenaSetting<K extends keyof ArenaSettings>(key: K, value: ArenaSettings[K]): ArenaSettings[K] {
    this.arenaSettings[key] = value;
    return this.arenaSettings[key];
  }

  toggleArenaSetting<K extends keyof ArenaSettings>(key: K): ArenaSettings[K] {
    this.arenaSettings[key] = !this.arenaSettings[key] as ArenaSettings[K];
    return this.arenaSettings[key];
  }

  getArenaSettings(): ArenaSettings {
    return { ...this.arenaSettings };
  }

  private controlPlayer(dt: number, input: SimInput): void {
    const drivingVeh = this.drivingVehicle();
    if (drivingVeh) {
      this.controlVehicle(drivingVeh, dt, input);
      return;
    }
    const b = this.ecs.get(this.player, Comp.Body)!, t = this.playerT, ch = this.playerChar;
    const m = Math.hypot(input.move.x, input.move.y);
    const hasBoingBoots = ch.look.outfit.feet?.item === 'boing_boots';
    const hasPropeller = ch.look.outfit.head?.item === 'propeller_beanie';
    const isAiming = !!input.aimHeld;
    const speedMult = (input.sprint && !isAiming ? 1.55 : 1)
      * (ch.anim.state === 'swing' ? 0.35 : 1)
      * (isAiming ? 0.52 : 1)
      * (this.buffs.speedBoostTimer > 0 ? 1.35 : 1)
      * (hasBoingBoots ? 1.15 : 1)
      * (hasPropeller ? 1.10 : 1);
    const sp = b.speed * speedMult;
    if (m > 0.05) {
      const nx = input.move.x / Math.max(1, m), ny = input.move.y / Math.max(1, m);
      b.vx = nx * sp; b.vy = ny * sp;
      if (!isAiming) {
        t.angle = lerpAngle(t.angle, Math.atan2(ny, nx), 1 - Math.exp(-14 * dt));
      }
    } else { b.vx = 0; b.vy = 0; }
    if (isAiming && input.aimWorld) {
      const targetAngle = Math.atan2(input.aimWorld.y - t.y, input.aimWorld.x - t.x);
      t.angle = lerpAngle(t.angle, targetAngle, 1 - Math.exp(-24 * dt));
    }
  }

  private updateNpcs(dt: number): void {
    for (const e of this.ecs.query(Comp.Wander, Comp.Transform, Comp.Body)) {
      const pawn = this.ecs.get(e, Comp.Pawn);
      if (pawn && (pawn.mentalBreak.kind !== 'none' || pawn.currentJob)) continue;
      const w = this.ecs.get(e, Comp.Wander)!, t = this.ecs.get(e, Comp.Transform)!, b = this.ecs.get(e, Comp.Body)!;
      if (!w.target) {
        w.wait -= dt; b.vx = 0; b.vy = 0;
        if (w.wait <= 0) { const a = this.rng.range(0, 6.28), d = this.rng.range(2, 6); w.target = { x: t.x + Math.cos(a) * d, y: t.y + Math.sin(a) * d }; }
        continue;
      }
      const dx = w.target.x - t.x, dy = w.target.y - t.y, d = Math.hypot(dx, dy);
      if (d < 0.3) { w.target = null; w.wait = this.rng.range(1.5, 5); b.vx = 0; b.vy = 0; continue; }
      b.vx = (dx / d) * b.speed; b.vy = (dy / d) * b.speed;
      t.angle = lerpAngle(t.angle, Math.atan2(dy, dx), 1 - Math.exp(-10 * dt));
    }
  }

  private updateRaiders(dt: number): void {
    for (const re of this.ecs.query(Comp.Raider, Comp.Transform, Comp.Body)) {
      const r = this.ecs.get(re, Comp.Raider)!, rt = this.ecs.get(re, Comp.Transform)!, rb = this.ecs.get(re, Comp.Body)!;
      if (r.cooldown > 0) r.cooldown -= dt;
      if (r.reactionTimer > 0) r.reactionTimer -= dt;

      // Find target:
      // - If in active arena or hostile: bots can target player, other bots (cross-fire / free-for-all), and zombies!
      // - Friendly/patrol bots outside arena prioritize zombies
      let targetPos: { x: number; y: number } | null = null;
      const pt = this.playerT;
      let nearestDist = Number.POSITIVE_INFINITY;

      // 1. Target other bots / raiders.
      // Enemy bots (`bot` / `raider`) can free-for-all each other if enabled.
      // Player-allied bots (`ally`) avoid targeting other allies.
      if (this.arenaSettings.botsFightEachOther) {
        for (const otherRe of this.ecs.query(Comp.Raider, Comp.Transform)) {
          if (otherRe === re) continue;
          const otherR = this.ecs.get(otherRe, Comp.Raider)!;
          if (r.team === 'ally' && otherR.team === 'ally') continue;
          const otherT = this.ecs.get(otherRe, Comp.Transform)!;
          const distOther = Math.hypot(otherT.x - rt.x, otherT.y - rt.y);
          if (distOther < r.sight && distOther < nearestDist) {
            nearestDist = distOther;
            targetPos = { x: otherT.x, y: otherT.y };
          }
        }
      }

      // 2. Target player only if bot is not player-allied and player targeting is enabled
      const distPlayer = Math.hypot(pt.x - rt.x, pt.y - rt.y);
      if (this.arenaSettings.botsTargetPlayer && r.team !== 'ally' && (r.team !== 'bot' || this.activeArena) && distPlayer < r.sight && distPlayer < nearestDist) {
        nearestDist = distPlayer;
        targetPos = { x: pt.x, y: pt.y };
      }

      // 3. Target zombies (high priority survival defense)
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        const dz = Math.hypot(zt.x - rt.x, zt.y - rt.y);
        if (dz < r.sight && dz < nearestDist) {
          nearestDist = dz;
          targetPos = { x: zt.x, y: zt.y };
        }
      }

      if (!targetPos || nearestDist > r.sight) {
        rb.vx = 0; rb.vy = 0;
        continue;
      }

      const dx = targetPos.x - rt.x, dy = targetPos.y - rt.y;
      const baseAngle = Math.atan2(dy, dx);
      rt.angle = lerpAngle(rt.angle, baseAngle, 1 - Math.exp(-12 * dt));

      // Movement behavior: Strafe / kite depending on level
      if (r.level === 'scarecrow') {
        // Scarecrow: walks straight slowly
        rb.vx = Math.cos(baseAngle) * (r.speed * 0.4);
        rb.vy = Math.sin(baseAngle) * (r.speed * 0.4);
      } else {
        // Regular / Sharp / Menace: strafe randomly, keep distance ~8-12m
        r.strafeTimer -= dt;
        if (r.strafeTimer <= 0) {
          r.strafeDir = this.rng.chance(0.5) ? 1 : -1;
          r.strafeTimer = this.rng.range(0.8, 2.2);
        }
        const perpX = -Math.sin(baseAngle) * r.strafeDir;
        const perpY = Math.cos(baseAngle) * r.strafeDir;

        const advance = nearestDist > 10 ? 0.6 : nearestDist < 5 ? -0.8 : 0;
        rb.vx = (perpX * 0.8 + Math.cos(baseAngle) * advance) * r.speed;
        rb.vy = (perpY * 0.8 + Math.sin(baseAngle) * advance) * r.speed;
      }

      // Check line of sight (crates/sandbags block bot targeting)
      let lineBlocked = false;
      if (this.activeArena) {
        const ar = this.activeArena;
        const lx0 = rt.x - ar.originX, ly0 = rt.y - ar.originY;
        const lx1 = targetPos.x - ar.originX, ly1 = targetPos.y - ar.originY;
        lineBlocked = arenaBlocksBullet(ar.def, lx0, ly0, lx1, ly1);
      }

      // Shooting logic
      if (!lineBlocked && r.cooldown <= 0 && r.reactionTimer <= 0 && nearestDist <= r.sight) {
        r.cooldown = r.shotInterval;

        // Apply bot aim spread error
        const spread = this.rng.range(-r.aimSpread, r.aimSpread);
        const shootAngle = baseAngle + spread;
        const vx = Math.cos(shootAngle) * 36;
        const vy = Math.sin(shootAngle) * 36;

        const pe = this.ecs.create();
        this.ecs.add(pe, Comp.Transform, {
          x: rt.x + Math.cos(baseAngle) * 0.45,
          y: rt.y + Math.sin(baseAngle) * 0.45,
          angle: shootAngle,
        });
        this.ecs.add(pe, Comp.Projectile, {
          source: re,
          startX: rt.x,
          startY: rt.y,
          vx, vy,
          speed: 36,
          maxDist: r.sight + 4,
          distTraveled: 0,
          baseDamage: r.weapon === 'hunting_rifle' ? 40 : r.weapon === 'pump_shotgun' ? 14 : 15,
          minDamage: 8,
          falloffStartDist: 15,
          penetration: 1,
          piercedCount: 0,
          arced: false,
          color: '#ef4444',
          radius: 0.12,
          trail: [],
        });

        this.emitNoise(rt.x, rt.y, 22);
      }
    }
  }

  private integrate(dt: number): void {
    for (const e of this.ecs.query(Comp.Transform, Comp.Body)) {
      const t = this.ecs.get(e, Comp.Transform)!, b = this.ecs.get(e, Comp.Body)!;
      if (b.vx === 0 && b.vy === 0) continue;
      const k = this.world.speedAt(t.x, t.y) * dt;
      const nx = t.x + b.vx * k, ny = t.y + b.vy * k;
      if (!this.world.blocked(nx, t.y, b.radius)) t.x = nx; else if (e !== this.player) { b.vx = 0; const w = this.ecs.get(e, Comp.Wander); if (w) w.target = null; }
      if (!this.world.blocked(t.x, ny, b.radius)) t.y = ny; else if (e !== this.player) { b.vy = 0; const w = this.ecs.get(e, Comp.Wander); if (w) w.target = null; }
    }
  }

  private updateAnim(dt: number): void {
    for (const e of this.ecs.query(Comp.Character, Comp.Body)) {
      const a = this.ecs.get(e, Comp.Character)!.anim, b = this.ecs.get(e, Comp.Body)!;
      const sp = Math.hypot(b.vx, b.vy);
      a.walkBlend = damp(a.walkBlend, sp > 0.1 ? 1 : 0, 14, dt);
      a.phase += sp * dt * 2.1;
      a.hitFlash = Math.max(0, a.hitFlash - dt * 4);
      if (a.state !== 'swing') a.state = sp > 0.1 ? 'walk' : 'idle';
    }
  }

  // ---------------------------------------------------------------- actions, targets and harvesting
  findTarget(): Target {
    const drivingVeh = this.drivingVehicle();
    if (drivingVeh) {
      return { kind: 'vehicle', entity: drivingVeh, prompt: 'Exit Vehicle', key: 'E', part: 'exit' };
    }
    const t = this.playerT, fx = Math.cos(t.angle), fy = Math.sin(t.angle);
    let best: Target = null, bestScore = Infinity;
    for (const o of this.world.objectsNear(t.x, t.y, 2.2)) {
      const def = this.content.objects.get(o.kind);
      if (!def || def.tool === 'none') continue;
      const dx = o.x - t.x, dy = o.y - t.y, d = Math.hypot(dx, dy);
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (d > def.radius + 1.15 || (dot < -0.1 && d > 0.9)) continue;
      const score = d - dot * 0.5;
      if (score < bestScore) {
        const held = this.heldTool();
        const ok = def.tool === 'hands' || held?.kind === def.tool;
        best = { kind: 'object', id: o.id, obj: o, prompt: def.prompt, key: 'SPACE', ok, need: def.tool === 'hands' ? undefined : def.tool };
        bestScore = score;
      }
    }
    for (const e of this.base.entities()) {
      const s = this.ecs.get(e, Comp.Structure)!;
      if (!s.def.interact) continue;
      const st = this.ecs.get(e, Comp.Transform)!;
      const d = Math.hypot(st.x - t.x, st.y - t.y);
      if (d > 1.7 + s.def.size[0] * 0.3) continue;
      const verb = { storage: 'Open', sleep: 'Sleep in', door: s.open ? 'Close' : 'Open', craft: 'Use', fire: 'Warm up at', table: 'Dine at', overseer: 'Configure', turret: 'Reload', totem: 'Manage Clan', gunsmith: 'Mod Weapons at' }[s.def.interact] ?? 'Use';
      if (d - 0.6 < bestScore) { best = { kind: 'structure', entity: e, prompt: `${verb} ${s.def.name}`, key: 'E' }; bestScore = d - 0.6; }
    }
    for (const e of this.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const v = this.ecs.get(e, Comp.Vehicle)!, vt = this.ecs.get(e, Comp.Transform)!;
      if (v.driver) continue;
      const dx = vt.x - t.x, dy = vt.y - t.y, d = Math.hypot(dx, dy);
      if (d > 2.6) continue;
      const dotLook = d > 0 ? (dx * fx + dy * fy) / d : 1;
      const forwardX = Math.cos(vt.angle), forwardY = Math.sin(vt.angle);
      const carToPlayerX = t.x - vt.x, carToPlayerY = t.y - vt.y;
      const dotFwd = carToPlayerX * forwardX + carToPlayerY * forwardY;
      const isRear = dotFwd < -0.4;
      const part = isRear ? 'trunk' : 'cabin';
      const prompt = isRear ? `Open Trunk (${v.trunk.slots.filter(Boolean).length} items)` : `Drive ${v.def.name}`;
      const score = d - dotLook * 0.9 - 0.5;
      if (score < bestScore) {
        best = { kind: 'vehicle', entity: e, prompt, key: 'E', part };
        bestScore = score;
      }
    }
    for (const e of this.ecs.query(Comp.Zombie, Comp.Transform)) {
      const z = this.ecs.get(e, Comp.Zombie)!, zt = this.ecs.get(e, Comp.Transform)!;
      const dx = zt.x - t.x, dy = zt.y - t.y, d = Math.hypot(dx, dy);
      if (d > 2.4) continue;
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (dot < -0.3 && d > 1.0) continue;
      const score = d - dot * 1.4 - 1.0;
      if (score < bestScore) {
        best = { kind: 'zombie', entity: e, prompt: `Attack ${z.def.name}`, key: 'SPACE' };
        bestScore = score;
      }
    }
    for (const pe of this.ecs.query(Comp.Pet, Comp.Transform)) {
      const p = this.ecs.get(pe, Comp.Pet)!, pt = this.ecs.get(pe, Comp.Transform)!;
      const dx = pt.x - t.x, dy = pt.y - t.y, d = Math.hypot(dx, dy);
      if (d > 1.9) continue;
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (dot < 0.25) continue;
      const score = d - dot * 0.6;
      if (score < bestScore) {
        best = { kind: 'pet', entity: pe, prompt: `Pet ${p.name}`, key: 'E' };
        bestScore = score;
      }
    }
    for (const ae of this.ecs.query(Comp.Animal, Comp.Transform)) {
      const a = this.ecs.get(ae, Comp.Animal)!, at = this.ecs.get(ae, Comp.Transform)!;
      const dx = at.x - t.x, dy = at.y - t.y, d = Math.hypot(dx, dy);
      if (d > 2.2) continue;
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (dot < 0.2) continue;
      const score = d - dot * 0.7;
      if (score < bestScore) {
        const heldItem = this.playerInv.inv.slots[this.playerInv.held]?.item ?? this.playerChar.look.outfit.mainHand?.item ?? '';
        let prompt = `Pet ${a.name}`;
        let action: 'pet' | 'milk' | 'shear' | 'eggs' | 'feed' = 'pet';

        const canFeed = a.def.favoriteFood.includes(heldItem) || heldItem === 'animal_feed';
        if (canFeed) {
          prompt = `Feed ${a.name} (${this.content.items.get(heldItem)?.name ?? heldItem})`;
          action = 'feed';
        } else if (a.def.produce && a.produceTimer <= 0) {
          prompt = `${a.def.produce.actionPrompt} (${a.name})`;
          action = a.kind === 'cow' ? 'milk' : a.kind === 'sheep' ? 'shear' : 'eggs';
        }
        best = { kind: 'animal', entity: ae, prompt, key: 'E', action };
        bestScore = score;
      }
    }
    return best;
  }

  findActionTarget(): Target {
    const t = this.playerT, fx = Math.cos(t.angle), fy = Math.sin(t.angle);
    let best: Target = null, bestScore = Infinity;
    for (const o of this.world.objectsNear(t.x, t.y, 2.2)) {
      const def = this.content.objects.get(o.kind);
      if (!def || def.tool === 'none') continue;
      const dx = o.x - t.x, dy = o.y - t.y, d = Math.hypot(dx, dy);
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (d > def.radius + 1.15 || (dot < -0.1 && d > 0.9)) continue;
      const score = d - dot * 0.5;
      if (score < bestScore) {
        const held = this.heldTool();
        const ok = def.tool === 'hands' || held?.kind === def.tool;
        best = { kind: 'object', id: o.id, obj: o, prompt: def.prompt, key: 'SPACE', ok, need: def.tool === 'hands' ? undefined : def.tool };
        bestScore = score;
      }
    }
    const held = this.heldTool();
    const maxZDist = Math.max(2.8, (held?.reach ?? 1.3) + 1.2);
    for (const e of this.ecs.query(Comp.Zombie, Comp.Transform)) {
      const z = this.ecs.get(e, Comp.Zombie)!, zt = this.ecs.get(e, Comp.Transform)!;
      const dx = zt.x - t.x, dy = zt.y - t.y, d = Math.hypot(dx, dy);
      if (d > maxZDist) continue;
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (dot < -0.3 && d > 1.0) continue;
      const score = d - dot * 1.4 - 1.0;
      if (score < bestScore) {
        best = { kind: 'zombie', entity: e, prompt: `Attack ${z.def.name}`, key: 'SPACE' };
        bestScore = score;
      }
    }
    for (const ae of this.ecs.query(Comp.Animal, Comp.Transform)) {
      const a = this.ecs.get(ae, Comp.Animal)!, at = this.ecs.get(ae, Comp.Transform)!;
      const dx = at.x - t.x, dy = at.y - t.y, d = Math.hypot(dx, dy);
      if (d > maxZDist) continue;
      const dot = d > 0 ? (dx * fx + dy * fy) / d : 1;
      if (dot < -0.3 && d > 1.0) continue;
      const score = d - dot * 1.2 - 0.4;
      if (score < bestScore && (!best || best.kind !== 'object' || !best.ok || d < 0.9)) {
        best = { kind: 'animal', entity: ae, prompt: `Hunt ${a.name} (${a.hp} HP)`, key: 'SPACE', action: 'hunt' };
        bestScore = score;
      }
    }
    return best;
  }

  private updateActions(dt: number, input: SimInput): void {
    const a = this.playerChar.anim;
    if (a.state === 'swing') {
      a.swingT += dt;
      if (!a.hitDone && a.swingT >= a.swingDur * 0.55) { a.hitDone = true; this.applyHit(); }
      if (a.swingT >= a.swingDur) a.state = 'idle';
    }

    const heldItem = this.playerChar.look.outfit.mainHand?.item;
    const def = heldItem ? this.content.items.get(heldItem) : null;
    const isGun = !!def?.gun;

    if (input.aimHeld && isGun) {
      this.aimActive = true;
      this.aimWorld = input.aimWorld ?? null;
      if (input.shootPressed || input.actionPressed) {
        this.shootGun(input.aimWorld);
      }
    } else {
      this.aimActive = false;
      this.aimWorld = null;
      if (input.actionPressed && a.state !== 'swing') {
        if (isGun) {
          this.shootGun(input.aimWorld);
        } else {
          this.beginAction();
        }
      }
    }

    if (input.interactPressed) this.useInteract();
  }

  shootGun(aimWorld?: { x: number; y: number }): boolean {
    const heldItem = this.playerChar.look.outfit.mainHand?.item;
    if (!heldItem) return false;
    const def = this.content.items.get(heldItem);
    const gun = def?.gun;
    if (!gun) return false;

    if (this.gunCooldown > 0) return false;

    const eff = this.getEffectiveGun(heldItem);

    // Check & consume ammo
    let hasAmmo = false;
    if (!this.arenaSettings.ammoConsumption) {
      hasAmmo = true;
    } else if (heldItem === 'potato_cannon') {
      if (this.playerInv.inv.remove('wild_carrot', 1)) {
        hasAmmo = true;
      } else if (this.playerInv.inv.remove('canned_beans', 1)) {
        hasAmmo = true;
      }
    } else {
      hasAmmo = this.playerInv.inv.remove(eff.ammoItem, eff.ammoPerShot);
    }

    if (!hasAmmo) {
      const name = eff.ammoName ?? (this.content.items.get(eff.ammoItem)?.name ?? eff.ammoItem);
      this.events.emit('toast', { text: `Need ${name} to shoot!` });
      this.gunCooldown = 0.35;
      return false;
    }

    this.gunCooldown = 1 / eff.fireRate;
    const t = this.playerT;
    const b = this.playerBody;
    const isMoving = Math.hypot(b.vx, b.vy) > 0.1;
    const pellets = eff.pelletCount ?? 1;

    // Special goofy sounds
    if (eff.honkSound) {
      this.events.emit('toast', { text: '📯 HONK! Car Horn Shroud Blast!' });
    }
    if (eff.squeakSound) {
      this.events.emit('toast', { text: '🐥 *SQUEAK!* Rubber Duck Damper!' });
    }

    // Noise alert
    if (this.arenaSettings.noiseAttractsZombies) {
      this.emitNoise(t.x, t.y, eff.noiseRadius);
    }

    // Aim direction
    const baseAngle = aimWorld ? Math.atan2(aimWorld.y - t.y, aimWorld.x - t.x) : t.angle;
    t.angle = baseAngle;

    const gr = this.rng.fork('gun_fire');
    for (let i = 0; i < pellets; i++) {
      const spread = (gr.range(-0.5, 0.5)) * eff.spreadAngle * (isMoving ? 1.7 : 0.85);
      const ang = baseAngle + spread;
      const vx = Math.cos(ang) * eff.projectileSpeed;
      const vy = Math.sin(ang) * eff.projectileSpeed;

      const pe = this.ecs.create();
      this.ecs.add(pe, Comp.Transform, {
        x: t.x + Math.cos(baseAngle) * 0.45,
        y: t.y + Math.sin(baseAngle) * 0.45,
        angle: ang,
      });
      this.ecs.add(pe, Comp.Projectile, {
        source: this.player,
        startX: t.x,
        startY: t.y,
        vx, vy,
        speed: eff.projectileSpeed,
        maxDist: eff.maxRange,
        distTraveled: 0,
        baseDamage: eff.baseDamage,
        minDamage: eff.minDamage,
        falloffStartDist: eff.falloffStartDist,
        penetration: eff.penetration ?? 1,
        piercedCount: 0,
        arced: !!eff.arced,
        color: eff.bulletColor ?? '#facc15',
        radius: eff.bulletRadius ?? 0.1,
        trail: [],
      });
    }

    this.events.emit('hit', { kind: 'wood', x: t.x + Math.cos(baseAngle) * 0.5, y: t.y + Math.sin(baseAngle) * 0.5 });
    return true;
  }

  private updateProjectiles(dt: number): void {
    if (this.gunCooldown > 0) this.gunCooldown -= dt;

    // Update target dummy wobble timers
    for (const de of this.ecs.query(Comp.TargetDummy)) {
      const dummy = this.ecs.get(de, Comp.TargetDummy)!;
      if (dummy.wobbleTimer > 0) dummy.wobbleTimer = Math.max(0, dummy.wobbleTimer - dt);
    }

    for (const pe of this.ecs.query(Comp.Projectile, Comp.Transform)) {
      const proj = this.ecs.get(pe, Comp.Projectile)!, pt = this.ecs.get(pe, Comp.Transform)!;
      const x0 = pt.x, y0 = pt.y;
      const dx = proj.vx * dt, dy = proj.vy * dt;
      const stepDist = Math.hypot(dx, dy);

      proj.distTraveled += stepDist;
      pt.x += dx;
      pt.y += dy;
      const x1 = pt.x, y1 = pt.y;

      proj.trail.push({ x: pt.x, y: pt.y });
      if (proj.trail.length > 5) proj.trail.shift();

      if (proj.distTraveled >= proj.maxDist) {
        this.ecs.destroy(pe);
        continue;
      }

      // Continuous collision segment distance helper
      const distToSeg = (cx: number, cy: number): number => {
        const l2 = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
        if (l2 === 0) return Math.hypot(cx - x0, cy - y0);
        const t = clamp(((cx - x0) * (x1 - x0) + (cy - y0) * (y1 - y0)) / l2, 0, 1);
        return Math.hypot(cx - (x0 + t * (x1 - x0)), cy - (y0 + t * (y1 - y0)));
      };

      // Check collision with solid map objects (e.g. sandbags, rocks)
      let hitSolid = false;
      for (const obj of this.world.objectsNear(pt.x, pt.y, proj.radius + 1.2)) {
        if (obj.kind === 'target_dummy' || obj.kind === 'steel_target') continue;
        const oDef = this.content.objects.get(obj.kind);
        if (!oDef?.solid) continue;
        const or = oDef.radius;
        if (distToSeg(obj.x, obj.y) <= or + proj.radius) {
          hitSolid = true;
          this.events.emit('hit', { kind: obj.kind === 'sandbags' ? 'dirt' : 'stone', x: pt.x, y: pt.y });
          break;
        }
      }

      // Check active Arena bullet-stopping obstacles (crates, sandbags, low walls)
      if (this.arenaSettings.arenaCoverBlocksBullets && !hitSolid && this.activeArena) {
        const ar = this.activeArena;
        const lx0 = x0 - ar.originX, ly0 = y0 - ar.originY;
        const lx1 = x1 - ar.originX, ly1 = y1 - ar.originY;
        if (arenaBlocksBullet(ar.def, lx0, ly0, lx1, ly1)) {
          hitSolid = true;
          this.events.emit('hit', { kind: 'wood', x: pt.x, y: pt.y });
        }
      }

      if (hitSolid) {
        this.ecs.destroy(pe);
        continue;
      }

      // Calculate travel damage falloff
      const calcDamage = (): number => {
        if (!this.arenaSettings.damageFalloff) return proj.baseDamage;
        if (proj.distTraveled <= proj.falloffStartDist) return proj.baseDamage;
        const ratio = (proj.distTraveled - proj.falloffStartDist) / Math.max(1, proj.maxDist - proj.falloffStartDist);
        return Math.max(proj.minDamage, Math.round(proj.baseDamage - (proj.baseDamage - proj.minDamage) * clamp(ratio, 0, 1)));
      };

      // Check Target Dummies
      let hitTarget = false;
      for (const de of this.ecs.query(Comp.TargetDummy, Comp.Transform)) {
        const dtPos = this.ecs.get(de, Comp.Transform)!;
        const dummy = this.ecs.get(de, Comp.TargetDummy)!;
        if (distToSeg(dtPos.x, dtPos.y) <= 0.48 + proj.radius) {
          const dmg = calcDamage();
          dummy.lastDamageTaken = dmg;
          dummy.lastHitDistance = Math.round(proj.distTraveled * 10) / 10;
          dummy.totalHits++;
          dummy.wobbleTimer = 0.35;
          this.events.emit('hit', { kind: 'wood', x: pt.x, y: pt.y });
          this.events.emit('toast', { text: `🎯 Target [${dummy.distanceLabel}] Hit: ${dmg} DMG at ${dummy.lastHitDistance}m!` });
          proj.piercedCount++;
          if (proj.piercedCount >= proj.penetration) {
            hitTarget = true;
            this.ecs.destroy(pe);
            break;
          }
        }
      }
      if (hitTarget) continue;

      // Check Zombies
      let hitZombie = false;
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (distToSeg(zt.x, zt.y) <= 0.48 + proj.radius) {
          const dmg = calcDamage();
          this.damageZombie(ze, dmg);
          this.events.emit('hit', { kind: 'flesh', x: pt.x, y: pt.y });
          this.events.emit('toast', { text: `🎯 Zombie hit! -${dmg} HP (${Math.round(proj.distTraveled * 10) / 10}m)` });
          proj.piercedCount++;
          if (proj.piercedCount >= proj.penetration) {
            hitZombie = true;
            this.ecs.destroy(pe);
            break;
          }
        }
      }
      if (hitZombie) continue;

      // Check Raider Bots
      let hitRaider = false;
      for (const re of this.ecs.query(Comp.Raider, Comp.Transform, Comp.Character)) {
        if (proj.source === re) continue;
        const rt = this.ecs.get(re, Comp.Transform)!, r = this.ecs.get(re, Comp.Raider)!, ch = this.ecs.get(re, Comp.Character)!;
        const sourceRaider = this.ecs.get(proj.source, Comp.Raider);
        if (!this.arenaSettings.allyFriendlyFire && sourceRaider?.team === 'ally' && r.team === 'ally') continue;
        if (distToSeg(rt.x, rt.y) <= 0.48 + proj.radius) {
          const dmg = calcDamage();
          r.hp -= dmg;
          r.firstDamageTime = this.simTime;
          ch.anim.hitFlash = 0.3;
          this.events.emit('hit', { kind: 'flesh', x: pt.x, y: pt.y });
          this.events.emit('toast', { text: `💥 Hit ${ch.name}! -${dmg} HP (${Math.max(0, r.hp)} HP left)` });
          if (r.hp <= 0) {
            this.events.emit('toast', { text: `🏆 ${ch.name} Eliminated!` });
            this.ecs.destroy(re);
          }
          proj.piercedCount++;
          if (proj.piercedCount >= proj.penetration) {
            hitRaider = true;
            this.ecs.destroy(pe);
            break;
          }
        }
      }
      if (hitRaider) continue;

      // Check Player (when shot by enemy bots)
      if (proj.source !== this.player) {
        const sourceRaider = this.ecs.get(proj.source, Comp.Raider);
        if (sourceRaider?.team === 'ally') continue;
        const ptPos = this.playerT;
        if (distToSeg(ptPos.x, ptPos.y) <= 0.45 + proj.radius) {
          const dmg = calcDamage();
          this.playerNeeds.health = Math.max(0, this.playerNeeds.health - dmg);
          this.playerChar.anim.hitFlash = 0.35;
          this.events.emit('hit', { kind: 'flesh', x: pt.x, y: pt.y });
          this.events.emit('toast', { text: `⚠️ Took -${dmg} DMG from Bot Gunfire! (${Math.round(this.playerNeeds.health)} HP left)` });
          this.ecs.destroy(pe);
          continue;
        }
      }

      // Check Animals
      let hitAnimal = false;
      for (const ae of this.ecs.query(Comp.Animal, Comp.Transform)) {
        const at = this.ecs.get(ae, Comp.Transform)!, a = this.ecs.get(ae, Comp.Animal)!;
        if (distToSeg(at.x, at.y) <= a.def.radius + proj.radius) {
          const dmg = calcDamage();
          this.damageAnimal(ae, dmg);
          this.events.emit('hit', { kind: 'flesh', x: pt.x, y: pt.y });
          proj.piercedCount++;
          if (proj.piercedCount >= proj.penetration) {
            hitAnimal = true;
            this.ecs.destroy(pe);
            break;
          }
        }
      }
      if (hitAnimal) continue;
    }
  }

  spawnShootingRangeTargets(): void {
    // Clear any existing target dummy entities first
    for (const de of this.ecs.query(Comp.TargetDummy)) {
      this.ecs.destroy(de);
    }

    const rx = this.world.spawn.x - 24, ry = this.world.spawn.y - 12;
    const lanes: [string, number, number][] = [
      // Primary Precision Lanes
      ['5m', rx + 8.0, ry - 2.0],
      ['10m', rx + 13.0, ry + 2.0],
      ['15m', rx + 18.0, ry - 1.5],
      ['20m', rx + 23.0, ry + 1.5],
      ['25m', rx + 28.0, ry - 2.2],
      ['30m', rx + 33.0, ry + 2.2],
      ['40m', rx + 43.0, ry - 1.0],
      ['50m', rx + 53.0, ry + 1.0],
      ['60m', rx + 63.0, ry - 1.8],
      ['75m', rx + 78.0, ry + 1.8],
      ['100m', rx + 103.0, ry],
      // CQB Killhouse Maze Lanes (ry - 8)
      ['CQB-1', rx + 10.0, ry - 8],
      ['CQB-2', rx + 15.0, ry - 9],
      ['CQB-3', rx + 21.0, ry - 6.8],
      ['CQB-4', rx + 28.0, ry - 8],
    ];

    for (const [dist, x, y] of lanes) {
      const e = this.ecs.create();
      this.ecs.add(e, Comp.Transform, { x, y, angle: 0 });
      this.ecs.add(e, Comp.TargetDummy, {
        distanceLabel: dist,
        lastDamageTaken: 0,
        lastHitDistance: 0,
        lastHitTime: 0,
        totalHits: 0,
        wobbleTimer: 0,
      });
    }
  }

  resetShootingRangeStats(): void {
    for (const de of this.ecs.query(Comp.TargetDummy)) {
      const dummy = this.ecs.get(de, Comp.TargetDummy)!;
      dummy.lastDamageTaken = 0;
      dummy.lastHitDistance = 0;
      dummy.totalHits = 0;
      dummy.wobbleTimer = 0;
    }
  }

  spawnProvingGroundsZombies(): void {
    const rx = this.world.spawn.x - 24, ry = this.world.spawn.y - 12;
    // Pen A: Walkers
    const pAx = rx + 16, pAy = ry + 8;
    this.spawnZombie(pAx - 0.8, pAy - 0.5, 'walker');
    this.spawnZombie(pAx + 0.8, pAy + 0.5, 'walker');

    // Pen B: Runners & Brutes
    const pBx = rx + 26, pBy = ry + 8;
    this.spawnZombie(pBx - 0.8, pBy, 'runner');
    this.spawnZombie(pBx + 0.8, pBy, 'brute');

    // Pen C: Screamer Isolation Cell
    const pCx = rx + 36, pCy = ry + 8;
    this.spawnZombie(pCx, pCy, 'screamer');
  }

  teleportToShootingRange(): void {
    const rx = this.world.spawn.x - 24, ry = this.world.spawn.y - 12;
    const t = this.playerT;
    t.x = rx + 3.0;
    t.y = ry;
    t.angle = 0;

    for (const pe of this.ecs.query(Comp.Pet, Comp.Transform)) {
      const pt = this.ecs.get(pe, Comp.Transform)!;
      pt.x = rx + 1.5;
      pt.y = ry + 1.0;
    }

    for (const ve of this.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const vt = this.ecs.get(ve, Comp.Transform)!;
      vt.x = rx - 1.5;
      vt.y = ry - 2.5;
      vt.angle = 0;
    }

    const inv = this.playerInv.inv;
    const guns = ['hunting_rifle', 'pump_shotgun', 'nailgun_smg', 'potato_cannon', 'slingshot'] as const;
    const displaced: { item: string; qty: number }[] = [];
    for (let i = 0; i < guns.length; i++) {
      const cur = inv.slots[i];
      if (cur && !guns.includes(cur.item as never)) {
        displaced.push(cur);
      }
      inv.slots[i] = { item: guns[i], qty: 1 };
    }
    for (const d of displaced) inv.add(d.item, d.qty);

    inv.add('rifle_ammo', 120);
    inv.add('shotgun_shells', 80);
    inv.add('scrap_nails', 200);
    inv.add('pebbles', 100);
    inv.add('wild_carrot', 30);
    inv.add('canned_beans', 20);

    this.holdSlot(0);

    // Ensure all target dummy lane entities exist
    this.spawnShootingRangeTargets();

    // Populate the three Quarantine Containment Pens
    const pAx = rx + 16, pAy = ry + 8;
    let hasPenZombies = false;
    for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
      const zt = this.ecs.get(ze, Comp.Transform)!;
      if (Math.hypot(zt.x - pAx, zt.y - pAy) < 5.0) { hasPenZombies = true; break; }
    }
    if (!hasPenZombies) {
      this.spawnProvingGroundsZombies();
    }

    this.revealArea(rx + 50, ry, 95);

    // Place Gunsmith Mod Workbench right at the firing line if not yet placed
    const benchDef = this.content.structures.get('gunsmith_bench');
    if (benchDef) {
      const bx = Math.floor(rx + 1.0), by = Math.floor(ry + 2.0);
      if (this.base.structureAt(bx, by) === undefined) {
        this.base.place(benchDef, bx, by, 'E');
      }
    }

    this.events.emit('toast', { text: "🎯 Arrived at Bro's Proving Grounds: 5m to 100m Targets & Quarantine Pens Active!" });
  }

  teleportToArena(arenaId: 'duel_pit' | 'ffa_yard' = 'duel_pit', spawnIndex = 0): void {
    const def = ARENAS[arenaId];
    if (!def) return;

    // Position arena directly inside the Army Base / Ballistic Proving Grounds
    // Army Base command line is at rx = spawn.x - 24, ry = spawn.y - 12
    const rx = this.world.spawn.x - 24;
    const ry = this.world.spawn.y - 12;
    const ox = rx + 6;
    const oy = arenaId === 'duel_pit' ? ry + 20 : ry + 20;
    this.activeArena = { def, originX: ox, originY: oy };

    // Register movement blocker on GameWorld
    this.world.blocker = (wx: number, wy: number, r: number) => {
      if (!this.activeArena) return false;
      const ar = this.activeArena;
      const lx = wx - ar.originX, ly = wy - ar.originY;
      // If within arena envelope, check arena bounds and obstacles
      if (lx >= -2 && lx <= ar.def.width + 2 && ly >= -2 && ly <= ar.def.height + 2) {
        return arenaBlocksMovement(ar.def, lx, ly, r);
      }
      return false;
    };

    // Exit driving vehicle if driving
    const drivingVeh = this.drivingVehicle();
    if (drivingVeh) {
      const v = this.ecs.get(drivingVeh, Comp.Vehicle);
      if (v) v.driver = null;
    }

    // Teleport player to chosen spawn
    const spawn = def.spawns[spawnIndex] ?? def.spawns[0];
    const t = this.playerT;
    t.x = ox + spawn.x;
    t.y = oy + spawn.y;
    t.angle = 0;

    // Move companion pet close to spawn
    for (const pe of this.ecs.query(Comp.Pet, Comp.Transform)) {
      const pt = this.ecs.get(pe, Comp.Transform)!;
      pt.x = ox + spawn.x - 1.2;
      pt.y = oy + spawn.y + 0.8;
    }

    // Equip with firearms and ammo for arena testing
    const inv = this.playerInv.inv;
    const guns = ['hunting_rifle', 'pump_shotgun', 'nailgun_smg', 'potato_cannon', 'slingshot'] as const;
    for (let i = 0; i < guns.length; i++) {
      inv.slots[i] = { item: guns[i], qty: 1 };
    }
    inv.add('rifle_ammo', 150);
    inv.add('shotgun_shells', 100);
    inv.add('scrap_nails', 250);
    inv.add('pebbles', 100);
    inv.add('wild_carrot', 40);
    inv.add('canned_beans', 30);
    this.holdSlot(0);

    // Reveal arena
    this.revealArea(ox + def.width / 2, oy + def.height / 2, Math.max(def.width, def.height) + 12);
    this.events.emit('snapCamera', { x: t.x, y: t.y });
    this.events.emit('toast', { text: `🏟️ Entered ${def.name} at Army Base! Solids & bullet-stopping active.` });
  }

  private spawnCombatBotAt(level: ArenaBotLevel, weapon: string, x: number, y: number, team: 'bot' | 'ally', roleLabel: string, angle = 0): Entity {
    const e = this.ecs.create();
    this.ecs.add(e, Comp.Transform, { x, y, angle });
    this.ecs.add(e, Comp.Body, { radius: 0.32, speed: 2.8, vx: 0, vy: 0 });

    const botLook = defaultLook();
    if (team === 'ally') {
      botLook.outfit.head = { item: 'beanie', colors: { primary: '#0ea5e9' } };
      botLook.outfit.torso = { item: 'jacket', colors: { primary: '#0369a1' } };
    } else {
      botLook.outfit.head = { item: 'hood', colors: { primary: '#b91c1c' } };
      botLook.outfit.torso = { item: 'hoodie', colors: { primary: '#991b1b' } };
    }
    botLook.outfit.mainHand = { item: weapon as never };
    this.ecs.add(e, Comp.Character, { name: `${roleLabel} (${level.toUpperCase()})`, look: botLook, anim: newAnim() });

    const specs = ARENA_BOT_LEVELS[level];
    this.ecs.add(e, Comp.Raider, {
      kind: weapon === 'pump_shotgun' ? 'heavy' : weapon === 'hunting_rifle' ? 'marksman' : 'raider',
      hp: specs.hp,
      maxHp: specs.hp,
      speed: specs.speed,
      sight: specs.sight,
      weapon,
      level,
      team,
      target: this.player,
      state: 'engage',
      cooldown: 0.5,
      reactionTimer: specs.reaction,
      heardPos: null,
      heardTimer: 0,
      coverPos: null,
      coverTimer: 0,
      strafeDir: 1,
      strafeTimer: 1.0,
      firstDamageTime: 0,
      aimSpread: specs.aimSpread,
      shotInterval: specs.shotInterval,
    });
    return e;
  }

  private spawnArenaBotCore(level: ArenaBotLevel, weapon: string, spawnIndex: number, team: 'bot' | 'ally', roleLabel: string): Entity | null {
    if (!this.activeArena) this.teleportToArena('duel_pit');
    const ar = this.activeArena!;
    const def = ar.def;
    const sp = def.spawns[spawnIndex % def.spawns.length] ?? def.spawns[0];
    const bx = ar.originX + sp.x;
    const by = ar.originY + sp.y;
    return this.spawnCombatBotAt(level, weapon, bx, by, team, roleLabel, team === 'ally' ? 0 : Math.PI);
  }

  /** Spawns an armed player-allied bot near player in the main world (non-arena). */
  spawnPlayerBot(level: ArenaBotLevel = 'regular', weapon = 'hunting_rifle', offsetX = 1.8, offsetY = 0): Entity {
    const pt = this.playerT;
    const e = this.spawnCombatBotAt(level, weapon, pt.x + offsetX, pt.y + offsetY, 'ally', 'Companion Bot', 0);
    this.events.emit('toast', { text: `🛡️ Companion Bot deployed with ${weapon}!` });
    return e;
  }

  /** Starts a district defense scenario in the main world (non-arena). */
  startTownDefense(district: TownDistrict = 'plaza', allyBots = 3, zombies = 12, botWeapon = 'hunting_rifle', zombieKind = 'runner'): void {
    this.teleportToTownDistrict(district);

    const pt = this.playerT;
    for (const re of this.ecs.query(Comp.Raider, Comp.Transform)) {
      const rt = this.ecs.get(re, Comp.Transform)!;
      if (Math.hypot(rt.x - pt.x, rt.y - pt.y) <= 40) this.ecs.destroy(re);
    }
    for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
      const zt = this.ecs.get(ze, Comp.Transform)!;
      if (Math.hypot(zt.x - pt.x, zt.y - pt.y) <= 40) this.ecs.destroy(ze);
    }

    for (let i = 0; i < allyBots; i++) {
      const a = (Math.PI * 2 * i) / Math.max(1, allyBots);
      this.spawnPlayerBot('regular', botWeapon, Math.cos(a) * 2.2, Math.sin(a) * 2.2);
    }

    for (let i = 0; i < zombies; i++) {
      const a = (Math.PI * 2 * i) / Math.max(1, zombies);
      const r = 12 + (i % 3) * 2.4;
      this.spawnZombie(pt.x + Math.cos(a) * r, pt.y + Math.sin(a) * r, zombieKind);
    }

    this.events.emit('toast', { text: `🧟 Town defense live: ${allyBots} companion bots vs ${zombies} zombies.` });
  }

  /** Spawns an armed enemy combat bot inside the arena */
  spawnArenaBot(level: ArenaBotLevel = 'regular', weapon = 'nailgun_smg', spawnIndex = 1): Entity | null {
    const e = this.spawnArenaBotCore(level, weapon, spawnIndex, 'bot', 'Bot');
    this.events.emit('toast', { text: `🤖 Spawned ${level.toUpperCase()} Bot armed with ${weapon}!` });
    return e;
  }

  /** Spawns an armed player-allied combat bot inside the arena */
  spawnArenaAllyBot(level: ArenaBotLevel = 'regular', weapon = 'hunting_rifle', spawnIndex = 0): Entity | null {
    const e = this.spawnArenaBotCore(level, weapon, spawnIndex, 'ally', 'Ally Bot');
    this.events.emit('toast', { text: `🛡️ Spawned allied ${level.toUpperCase()} Bot armed with ${weapon}!` });
    return e;
  }

  /** Spawns arena zombies inside the active arena */
  spawnArenaZombies(count = 5, kind = 'walker'): void {
    if (!this.activeArena) this.teleportToArena('duel_pit');
    const ar = this.activeArena!;
    const ox = ar.originX, oy = ar.originY;
    const def = ar.def;

    for (let i = 0; i < count; i++) {
      const zx = ox + def.width * (0.35 + (i / Math.max(1, count - 1)) * 0.3);
      const zy = oy + 4 + (i % 3) * 6;
      const ze = this.spawnZombie(zx, zy, kind);
      const z = this.ecs.get(ze, Comp.Zombie);
      if (z) {
        z.state = 'alert';
        z.alertPos = { x: this.playerT.x, y: this.playerT.y };
      }
    }
    this.events.emit('toast', { text: `🧟 Released ${count} ${kind.toUpperCase()} zombies into arena!` });
  }

  /** Clears all bots and zombies from the active arena */
  clearArenaCombatants(): void {
    let cleared = 0;
    for (const re of this.ecs.query(Comp.Raider)) {
      this.ecs.destroy(re);
      cleared++;
    }
    if (this.activeArena) {
      const ar = this.activeArena;
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (zt.x >= ar.originX - 2 && zt.x <= ar.originX + ar.def.width + 2 && zt.y >= ar.originY - 2 && zt.y <= ar.originY + ar.def.height + 2) {
          this.ecs.destroy(ze);
          cleared++;
        }
      }
    }
    this.events.emit('toast', { text: `🧹 Cleared ${cleared} arena combatants.` });
  }

  private beginAction(): void {
    const tool = this.heldTool(), target = this.findActionTarget();
    if (target?.kind === 'zombie' || target?.kind === 'animal') {
      const t = this.playerT, zt = this.ecs.get(target.entity, Comp.Transform)!;
      t.angle = Math.atan2(zt.y - t.y, zt.x - t.x);
    } else if (target?.kind === 'object') {
      if (!target.ok) { this.events.emit('toast', { text: `You need ${target.need === 'axe' ? 'an axe' : target.need === 'pickaxe' ? 'a pickaxe' : 'the right tool'}` }); return; }
      const t = this.playerT; t.angle = Math.atan2(target.obj.y - t.y, target.obj.x - t.x);
    } else if (!tool) return;
    const a = this.playerChar.anim;
    a.state = 'swing'; a.swingT = 0; a.swingDur = 0.5 / (tool?.speed ?? 1.2); a.hitDone = false;
  }

  private applyHit(): void {
    const target = this.findActionTarget();
    const tool = this.heldTool();
    const heldItem = this.playerChar.look.outfit.mainHand?.item;
    const itemDef = heldItem ? this.content.items.get(heldItem) : null;

    if (itemDef?.gun) {
      this.shootGun(this.aimWorld ?? undefined);
      return;
    }

    if (heldItem === 'air_horn') {
      this.events.emit('toast', { text: '📢 HONK! Air horn blasted soundwave!' });
      this.emitNoise(this.playerT.x, this.playerT.y, 28);
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - this.playerT.x, zt.y - this.playerT.y) <= 8.5) {
          const z = this.ecs.get(ze, Comp.Zombie)!;
          z.attackTimer = 3.5;
          z.state = 'alert';
        }
      }
      return;
    }

    if (heldItem === 'slingshot') {
      if (!this.playerInv.inv.remove('pebbles', 1)) {
        this.events.emit('toast', { text: 'Need River Pebbles for slingshot' });
        return;
      }
      this.emitNoise(this.playerT.x, this.playerT.y, 4);
      if (target?.kind === 'zombie') {
        this.damageZombie(target.entity, 8);
        this.events.emit('toast', { text: 'Pebble direct hit! (-8 HP)' });
      } else if (target?.kind === 'animal') {
        this.damageAnimal(target.entity, 8);
        this.events.emit('toast', { text: 'Pebble hit animal! (-8 HP)' });
      }
      return;
    }

    if (heldItem === 'potato_cannon') {
      const hasAmmo = this.playerInv.inv.remove('wild_carrot', 1) || this.playerInv.inv.remove('canned_beans', 1);
      if (!hasAmmo) {
        this.events.emit('toast', { text: 'Need Wild Carrots or Canned Beans as ammo' });
        return;
      }
      this.emitNoise(this.playerT.x, this.playerT.y, 16);
      this.events.emit('toast', { text: '🥔 THUMP! Spud cannon fired!' });
      if (target?.kind === 'zombie') {
        this.damageZombie(target.entity, 18);
      } else if (target?.kind === 'animal') {
        this.damageAnimal(target.entity, 18);
      }
      return;
    }

    if (target?.kind === 'animal') {
      let dmg = tool?.damage ?? 4;
      if (heldItem === 'pool_noodle_of_justice') dmg = 2;
      else if (heldItem === 'chainsaw_of_regret') dmg = 35;
      else if (heldItem === 'frying_pan') dmg = 10;
      else if (heldItem === 'traffic_sign_axe') dmg = 20;
      this.damageAnimal(target.entity, dmg);
      return;
    }

    if (target?.kind === 'zombie') {
      let dmg = tool?.damage ?? 4;
      if (heldItem === 'pool_noodle_of_justice') {
        dmg = 2;
        this.events.emit('toast', { text: 'BOING! 🌀 Pool Noodle of Justice slap!' });
        const zb = this.ecs.get(target.entity, Comp.Body);
        const zt = this.ecs.get(target.entity, Comp.Transform)!;
        const ang = Math.atan2(zt.y - this.playerT.y, zt.x - this.playerT.x);
        zt.x += Math.cos(ang) * 2.2;
        zt.y += Math.sin(ang) * 2.2;
        if (zb) {
          zb.vx += Math.cos(ang) * 12.0;
          zb.vy += Math.sin(ang) * 12.0;
        }
      } else if (heldItem === 'chainsaw_of_regret') {
        dmg = 35;
        this.emitNoise(this.playerT.x, this.playerT.y, 22);
        this.events.emit('toast', { text: 'VRRRR! 🪚 Chainsaw shreds target!' });
      } else if (heldItem === 'frying_pan') {
        dmg = 10;
        if (this.rng.next() < 0.35) {
          const z = this.ecs.get(target.entity, Comp.Zombie)!;
          z.attackTimer = 2.5;
          z.state = 'alert';
          this.events.emit('toast', { text: 'BONG! 🍳 Target dazed with cartoon stars!' });
        }
      } else if (heldItem === 'traffic_sign_axe') {
        dmg = 20;
        this.events.emit('toast', { text: 'CLANG! 🛑 Traffic sign cleaves down!' });
      }
      this.damageZombie(target.entity, dmg);
      return;
    }
    if (target?.kind !== 'object' || !target.ok) return;
    const def = this.content.objects.get(target.obj.kind);
    if (!def) return;
    const { destroyed, obj } = this.world.damageObject(target.id, 1);
    this.events.emit('hit', { x: target.obj.x, y: target.obj.y, kind: target.obj.kind });
    this.emitNoise(target.obj.x, target.obj.y, 7.5);
    if (!destroyed) return;
    const r = this.rng.fork(`${target.id}:${obj?.hp}`);
    const inv = this.playerInv.inv;
    for (const d of def.drops) {
      const qty = r.int(d.min, d.max);
      if (qty <= 0) continue;
      const left = inv.add(d.item, qty);
      if (left < qty) this.events.emit('harvest', { item: d.item, qty: qty - left, x: target.obj.x, y: target.obj.y });
      if (left > 0) this.events.emit('toast', { text: 'Your pack is full' });
    }
    if (target.obj.kind === 'radio_tower') {
      const added = this.revealArea(target.obj.x, target.obj.y, 450);
      this.events.emit('toast', { text: `📻 Radio Tower signal scanned! Mapped +${added} sectors.` });
      this.events.emit('built', { id: 'campfire' as Id, entity: this.player });
    }
  }

  private useInteract(): void {
    const target = this.findTarget();
    if (!target) return;
    if (target.kind === 'vehicle') {
      const v = this.ecs.get(target.entity, Comp.Vehicle)!, vt = this.ecs.get(target.entity, Comp.Transform)!;
      if (target.part === 'exit') {
        v.driver = null;
        v.speed = 0;
        this.playerT.x = vt.x - Math.sin(vt.angle) * 0.95;
        this.playerT.y = vt.y + Math.cos(vt.angle) * 0.95;
        this.events.emit('toast', { text: 'Exited vehicle' });
      } else if (target.part === 'cabin') {
        v.driver = this.player;
        this.events.emit('toast', { text: `Driving ${v.def.name} [E to exit]` });
      } else if (target.part === 'trunk') {
        this.events.emit('vehicleTrunk', { entity: target.entity });
      }
      return;
    }
    if (target.kind === 'pet') {
      const pet = this.ecs.get(target.entity, Comp.Pet)!;
      pet.tailWag += 12;
      this.playerNeeds.sanity = Math.min(this.playerNeeds.maxSanity, this.playerNeeds.sanity + 12);
      this.events.emit('petBark', { pet: target.entity, x: this.playerT.x, y: this.playerT.y });
      this.events.emit('toast', { text: `You pet ${pet.name}! (+12 Sanity *wag*)` });
      return;
    }
    if (target.kind === 'animal') {
      const a = this.ecs.get(target.entity, Comp.Animal)!;
      const at = this.ecs.get(target.entity, Comp.Transform)!;
      if (target.action === 'milk') {
        const left = this.playerInv.inv.add('fresh_milk', 1);
        if (left === 0) {
          a.produceTimer = a.def.produce?.cooldown ?? 75;
          this.events.emit('toast', { text: `🥛 Fresh Creamy Milk collected from ${a.name}!` });
          this.events.emit('animalProduced', { animal: target.entity, item: 'fresh_milk', x: at.x, y: at.y });
        } else {
          this.events.emit('toast', { text: 'Your pack is full for milk' });
        }
      } else if (target.action === 'shear') {
        const qty = this.rng.int(2, 3);
        const left = this.playerInv.inv.add('wool_bundle', qty);
        if (left < qty) {
          a.produceTimer = a.def.produce?.cooldown ?? 90;
          this.events.emit('toast', { text: `✂️ Sheared ${qty - left} Fluffy Wool from ${a.name}!` });
          this.events.emit('animalProduced', { animal: target.entity, item: 'wool_bundle', x: at.x, y: at.y });
        } else {
          this.events.emit('toast', { text: 'Your pack is full for wool' });
        }
      } else if (target.action === 'eggs') {
        const qty = this.rng.int(1, 2);
        const left = this.playerInv.inv.add('fresh_egg', qty);
        if (left < qty) {
          a.produceTimer = a.def.produce?.cooldown ?? 45;
          this.events.emit('toast', { text: `🥚 Collected ${qty - left} Fresh Farm Eggs from ${a.name}!` });
          this.events.emit('animalProduced', { animal: target.entity, item: 'fresh_egg', x: at.x, y: at.y });
        } else {
          this.events.emit('toast', { text: 'Your pack is full for eggs' });
        }
      } else if (target.action === 'feed') {
        const heldItem = this.playerInv.inv.slots[this.playerInv.held]?.item ?? this.playerChar.look.outfit.mainHand?.item ?? '';
        const foodItem = a.def.favoriteFood.includes(heldItem) ? heldItem : 'animal_feed';
        if (this.playerInv.inv.remove(foodItem, 1)) {
          a.tamed = true;
          a.hp = Math.min(a.maxHp, a.hp + 15);
          this.playerNeeds.sanity = Math.min(this.playerNeeds.maxSanity, this.playerNeeds.sanity + 12);
          this.events.emit('toast', { text: `♥ ${a.name} munched ${this.content.items.get(foodItem)?.name ?? foodItem}! (Tamed)` });
        }
      } else {
        // Pet
        this.playerNeeds.sanity = Math.min(this.playerNeeds.maxSanity, this.playerNeeds.sanity + 8);
        this.events.emit('toast', { text: `${a.def.sound} ${a.name} purred contentedly! (+8 Sanity)` });
      }
      return;
    }
    if (target.kind === 'object') {
      if (target.obj.kind === 'town_fountain') {
        this.playerNeeds.sanity = Math.min(this.playerNeeds.maxSanity, this.playerNeeds.sanity + 25);
        this.events.emit('toast', { text: '⛲ Rested at Town Fountain (+25 Sanity, pure crystal water)' });
        return;
      }
      if (target.obj.kind === 'park_bench') {
        this.playerNeeds.sanity = Math.min(this.playerNeeds.maxSanity, this.playerNeeds.sanity + 15);
        this.events.emit('toast', { text: '🪑 Sat on Park Bench (+15 Sanity, peaceful rest)' });
        return;
      }
      if (target.obj.kind === 'dumpster') {
        const pool = ['rubber_chicken', 'stale_baguette', 'canned_beans', 'fanny_pack_of_holding', 'propeller_beanie'];
        const item = pool[this.rng.int(0, pool.length - 1)] ?? 'canned_beans';
        this.playerInv.inv.add(item, 1);
        this.events.emit('toast', { text: `🗑️ Dumpster scavenged! Found ${item.replace(/_/g, ' ')}!` });
        return;
      }
    }
    if (target.kind === 'structure') {
      this.useStructure(target.entity);
    }
  }

  private useStructure(entity?: Entity): void {
    const target = this.findTarget();
    const targetEntity = entity ?? (target?.kind === 'structure' ? target.entity : null);
    if (!targetEntity) return;
    const s = this.ecs.get(targetEntity, Comp.Structure)!;
    switch (s.def.interact) {
      case 'storage': this.events.emit('storage', { entity: targetEntity }); break;
      case 'door': s.open = !s.open; break;
      case 'sleep':
        if (this.clock.darkness > 0.4) { this.clock.t = 0.1; this.playerNeeds.sanity = Math.min(100, this.playerNeeds.sanity + 30); this.events.emit('sleep', {}); this.events.emit('toast', { text: 'You sleep until dawn' }); }
        else this.events.emit('toast', { text: 'You are not tired yet' });
        break;
      case 'fire': this.playerNeeds.sanity = Math.min(100, this.playerNeeds.sanity + 6); this.events.emit('toast', { text: 'The fire is warm' }); break;
      case 'craft': this.events.emit('toast', { text: 'Craft from the Craft panel (C)' }); break;
      case 'table': this.playerNeeds.sanity = Math.min(100, this.playerNeeds.sanity + 10); this.events.emit('toast', { text: 'Ate at a table (+10 Sanity, no mental break!)' }); break;
      case 'overseer': this.events.emit('overseer', {}); break;
      case 'turret': {
        const turret = this.ecs.get(targetEntity, Comp.Turret);
        if (turret) {
          if (turret.ammo >= turret.maxAmmo) {
            this.events.emit('toast', { text: 'Turret ammo magazine is full' });
          } else if (turret.kind === 'flame') {
            if (this.playerInv.inv.remove('flame_fuel', 1)) {
              turret.ammo = Math.min(turret.maxAmmo, turret.ammo + 20);
              this.events.emit('toast', { text: `Reloaded Flame Fuel (${turret.ammo}/${turret.maxAmmo})` });
            } else {
              this.events.emit('toast', { text: 'Need Flame Fuel to reload turret' });
            }
          } else {
            if (this.playerInv.inv.remove('turret_ammo', 1)) {
              turret.ammo = Math.min(turret.maxAmmo, turret.ammo + 20);
              this.events.emit('toast', { text: `Reloaded Bolter Ammo (${turret.ammo}/${turret.maxAmmo})` });
            } else {
              this.events.emit('toast', { text: 'Need Turret Bolt Ammo to reload' });
            }
          }
        }
        break;
      }
      case 'totem': this.events.emit('faction', {}); break;
      case 'gunsmith': this.events.emit('gunsmith', { entity: targetEntity }); break;
      default: break;
    }
  }

  // ---------------------------------------------------------------- survival
  /** All light sources for rendering and survival: structures plus held lights. */
  lights(): { x: number; y: number; radius: number; color: string; flicker: number }[] {
    const out: { x: number; y: number; radius: number; color: string; flicker: number }[] = [];
    for (const e of this.ecs.query(Comp.Light, Comp.Transform)) {
      if (this.ecs.has(e, Comp.Vehicle)) continue; // Handled below with headlights toggle
      const l = this.ecs.get(e, Comp.Light)!, t = this.ecs.get(e, Comp.Transform)!;
      out.push({ x: t.x, y: t.y, ...l });
    }
    for (const e of this.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const v = this.ecs.get(e, Comp.Vehicle)!, t = this.ecs.get(e, Comp.Transform)!;
      if (v.headlights) {
        out.push({
          x: t.x + Math.cos(t.angle) * 1.15,
          y: t.y + Math.sin(t.angle) * 1.15,
          radius: v.def.lightRadius,
          color: '#fde58a',
          flicker: 0.04
        });
      }
    }
    for (const e of this.ecs.query(Comp.Character, Comp.Transform)) {
      const o = this.ecs.get(e, Comp.Character)!.look.outfit, t = this.ecs.get(e, Comp.Transform)!;
      const r = Math.max(this.content.items.get(o.mainHand?.item ?? '')?.equip?.lightRadius ?? 0, this.content.items.get(o.offHand?.item ?? '')?.equip?.lightRadius ?? 0);
      if (r > 0) out.push({ x: t.x + Math.cos(t.angle) * 0.3, y: t.y + Math.sin(t.angle) * 0.3, radius: r, color: '#ffb35a', flicker: 0.2 });
    }
    if (this.buffs.rainbowGlowTimer > 0) {
      const colors = ['#f43f5e', '#fbbf24', '#34d399', '#38bdf8', '#a855f7'];
      const ci = Math.floor((this.simTime * 3) % colors.length);
      out.push({ x: this.playerT.x, y: this.playerT.y, radius: 8.5, color: colors[ci], flicker: 0.1 });
    }
    for (const fl of this.activeFlares) {
      out.push({ x: fl.x, y: fl.y, radius: 8.0, color: '#ef4444', flicker: 0.08 });
    }
    const streetLights = this.world.objectsNear(this.playerT.x, this.playerT.y, 28).filter((o) => o.kind === 'street_light');
    for (const sl of streetLights) {
      out.push({ x: sl.x, y: sl.y - 0.35, radius: 5.8, color: '#fef08a', flicker: 0.03 });
    }
    return out;
  }
  isLit(x: number, y: number): boolean { return this.lights().some((l) => Math.hypot(l.x - x, l.y - y) < l.radius * 0.85); }

  private updateNeeds(dt: number): void {
    const n = this.playerNeeds, t = this.playerT;
    n.hunger = clamp(n.hunger - dt * 0.14, 0, n.maxHunger);
    const dark = this.clock.darkness > 0.55 && !this.isLit(t.x, t.y);
    if (dark) {
      n.darkTimer += dt; n.sanity = clamp(n.sanity - dt * 1.6, 0, n.maxSanity);
      if (n.darkTimer > 6 && this.config.darknessDamage) n.health -= dt * 2.5;
    } else { n.darkTimer = Math.max(0, n.darkTimer - dt * 2); if (this.clock.darkness > 0.3) n.sanity = clamp(n.sanity + dt * 0.4, 0, n.maxSanity); }
    const nearFountain = this.world.objectsNear(t.x, t.y, 3.5).some((o) => o.kind === 'town_fountain');
    const nearBench = this.world.objectsNear(t.x, t.y, 2.0).some((o) => o.kind === 'park_bench');
    if (nearFountain || nearBench) {
      n.sanity = clamp(n.sanity + dt * 2.5, 0, n.maxSanity);
    }
    if (n.hunger <= 0) n.health -= dt * 0.8;
    else if (n.hunger > 40 && !dark) n.health = clamp(n.health + dt * 0.25, 0, n.maxHealth);
    if (n.health <= 0) {
      const plaza = this.townAnchor('plaza');
      t.x = plaza.x;
      t.y = plaza.y;
      t.angle = plaza.angle;
      n.health = n.maxHealth * 0.6; n.hunger = 60; n.sanity = 70; n.darkTimer = 0;
      this.events.emit('death', {}); this.events.emit('toast', { text: 'You wake up in the town' });
    }
  }

  private updateBuffs(dt: number): void {
    if (this.buffs.hiccupTimer > 0) {
      this.buffs.hiccupTimer -= dt;
      if (Math.floor((this.buffs.hiccupTimer + dt) / 4) !== Math.floor(this.buffs.hiccupTimer / 4)) {
        this.events.emit('toast', { text: '🫧 *HICCUP!*' });
        const pb = this.ecs.get(this.player, Comp.Body);
        if (pb) {
          pb.vx += this.rng.range(-2.5, 2.5);
          pb.vy += this.rng.range(-2.5, 2.5);
        }
      }
    }
    if (this.buffs.fireBreathTimer > 0) {
      this.buffs.fireBreathTimer -= dt;
      const pt = this.playerT;
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - pt.x, zt.y - pt.y) <= 3.2) {
          const z = this.ecs.get(ze, Comp.Zombie)!;
          z.hp -= dt * 16;
          if (z.hp <= 0) {
            this.ecs.destroy(ze);
            this.events.emit('zombieKilled', { kind: z.def.kind, x: zt.x, y: zt.y });
          }
        }
      }
    }
    if (this.buffs.rainbowGlowTimer > 0) this.buffs.rainbowGlowTimer -= dt;
    if (this.buffs.speedBoostTimer > 0) this.buffs.speedBoostTimer -= dt;
    if (this.buffs.buoyancyTimer > 0) this.buffs.buoyancyTimer -= dt;

    for (let i = this.activeFlares.length - 1; i >= 0; i--) {
      this.activeFlares[i].timer -= dt;
      if (this.activeFlares[i].timer <= 0) this.activeFlares.splice(i, 1);
    }
  }

  teleportToTown(): void {
    this.teleportToTownDistrict('plaza');
  }

  teleportToTownDistrict(district: TownDistrict = 'plaza'): void {
    const a = this.townAnchor(district);
    const slots = this.townCompanionSlots(district);

    const t = this.playerT;
    t.x = a.x;
    t.y = a.y;
    t.angle = a.angle;

    for (const ve of this.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const vt = this.ecs.get(ve, Comp.Transform)!;
      vt.x = slots.vehicle.x;
      vt.y = slots.vehicle.y;
      vt.angle = slots.vehicle.angle;
    }
    for (const pe of this.ecs.query(Comp.Pet, Comp.Transform)) {
      const pt = this.ecs.get(pe, Comp.Transform)!;
      pt.x = slots.pet.x;
      pt.y = slots.pet.y;
      pt.angle = slots.pet.angle;
    }

    this.revealArea(a.x, a.y, district === 'plaza' ? 60 : 48);
    const label = district === 'plaza' ? 'Town Plaza' : district === 'residential' ? 'Residential Quarter' : district === 'commercial' ? 'Commercial High Street' : 'Industrial Auto Garage';
    this.events.emit('toast', { text: `Arrived at ${label}` });
  }

  private updateAutomation(dt: number): void {
    const dirOffsets: Record<'N' | 'E' | 'S' | 'W', { dx: number; dy: number }> = {
      N: { dx: 0, dy: -1 },
      E: { dx: 1, dy: 0 },
      S: { dx: 0, dy: 1 },
      W: { dx: -1, dy: 0 },
    };

    // 1. Conveyor Belts: advance items and pass forward
    for (const e of this.ecs.query(Comp.Conveyor, Comp.Structure)) {
      const conv = this.ecs.get(e, Comp.Conveyor)!, s = this.ecs.get(e, Comp.Structure)!;
      const off = dirOffsets[conv.dir];
      for (let i = conv.items.length - 1; i >= 0; i--) {
        const item = conv.items[i];
        if (!item) continue;
        item.progress += dt * conv.speed;
        if (item.progress >= 1.0) {
          const nextEntity = this.base.structureAt(s.tx + off.dx, s.ty + off.dy);
          let transferred = false;
          if (nextEntity !== undefined) {
            const nextConv = this.ecs.get(nextEntity, Comp.Conveyor);
            const nextStruct = this.ecs.get(nextEntity, Comp.Structure);
            const nextMach = this.ecs.get(nextEntity, Comp.Machine);
            if (nextConv && nextConv.items.length < 4) {
              nextConv.items.push({ item: item.item, progress: 0.0 });
              transferred = true;
            } else if (nextStruct?.storage) {
              const rem = nextStruct.storage.add(item.item, 1);
              if (rem === 0) transferred = true;
            } else if (nextMach && nextMach.kind === 'stone_furnace') {
              if (item.item === 'iron_ore' || item.item === 'copper_ore') {
                nextMach.targetOre = item.item;
                transferred = true;
              }
            }
          }
          if (transferred) {
            conv.items.splice(i, 1);
          } else {
            item.progress = Math.min(0.96, item.progress);
          }
        }
      }
    }

    // 2. Inserters: pickup from source, rotate, drop to destination
    for (const e of this.ecs.query(Comp.Inserter, Comp.Structure)) {
      const ins = this.ecs.get(e, Comp.Inserter)!, s = this.ecs.get(e, Comp.Structure)!;
      const pickOff = dirOffsets[ins.pickupDir], dropOff = dirOffsets[ins.dropDir];
      if (!ins.heldItem) {
        const sourceEntity = this.base.structureAt(s.tx + pickOff.dx, s.ty + pickOff.dy);
        if (sourceEntity !== undefined) {
          const sourceConv = this.ecs.get(sourceEntity, Comp.Conveyor);
          const sourceStruct = this.ecs.get(sourceEntity, Comp.Structure);
          const sourceMach = this.ecs.get(sourceEntity, Comp.Machine);
          if (sourceConv && sourceConv.items.length > 0) {
            const readyIdx = sourceConv.items.findIndex((it) => it.progress >= 0.25);
            if (readyIdx >= 0) {
              ins.heldItem = sourceConv.items[readyIdx].item;
              sourceConv.items.splice(readyIdx, 1);
              ins.armProgress = 0;
            }
          } else if (sourceStruct?.storage) {
            for (let si = 0; si < sourceStruct.storage.slots.length; si++) {
              const slot = sourceStruct.storage.slots[si];
              if (slot && slot.qty > 0) {
                ins.heldItem = slot.item;
                sourceStruct.storage.take({ [slot.item]: 1 });
                ins.armProgress = 0;
                break;
              }
            }
          } else if (sourceMach?.outputItem) {
            ins.heldItem = sourceMach.outputItem;
            sourceMach.outputItem = undefined;
            ins.armProgress = 0;
          }
        }
      } else {
        ins.armProgress += dt * ins.speed;
        if (ins.armProgress >= 1.0) {
          const destEntity = this.base.structureAt(s.tx + dropOff.dx, s.ty + dropOff.dy);
          let dropped = false;
          if (destEntity !== undefined) {
            const destConv = this.ecs.get(destEntity, Comp.Conveyor);
            const destStruct = this.ecs.get(destEntity, Comp.Structure);
            const destMach = this.ecs.get(destEntity, Comp.Machine);
            if (destConv && destConv.items.length < 4) {
              destConv.items.push({ item: ins.heldItem, progress: 0.1 });
              dropped = true;
            } else if (destStruct?.storage) {
              const rem = destStruct.storage.add(ins.heldItem, 1);
              if (rem === 0) dropped = true;
            } else if (destMach && destMach.kind === 'stone_furnace') {
              if (ins.heldItem === 'iron_ore' || ins.heldItem === 'copper_ore') {
                destMach.targetOre = ins.heldItem;
                dropped = true;
              }
            }
          }
          if (dropped) {
            ins.heldItem = null;
            ins.armProgress = 0;
          } else {
            ins.armProgress = 1.0;
          }
        }
      }
    }

    // 3. Machines: Burner Drills & Smelting Furnaces
    for (const e of this.ecs.query(Comp.Machine, Comp.Structure, Comp.Transform)) {
      const mach = this.ecs.get(e, Comp.Machine)!, s = this.ecs.get(e, Comp.Structure)!, t = this.ecs.get(e, Comp.Transform)!;
      if (mach.kind === 'burner_drill') {
        const nearbyObjects = this.world.objectsNear(t.x, t.y, 3.5);
        const mineralNode = nearbyObjects.find((o) =>
          o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode'
        );
        if (mineralNode) {
          mach.progress += dt;
          if (mach.progress >= 2.2) {
            mach.progress = 0;
            const oreItem = mineralNode.kind === 'iron_node' ? 'iron_ore'
              : mineralNode.kind === 'copper_node' ? 'copper_ore'
              : mineralNode.kind === 'coal_node' ? 'coal_chunk'
              : mineralNode.kind === 'oil_seam' ? 'crude_oil_canister' : 'energetic_crystal';
            const neighbors = [
              { tx: s.tx + s.def.size[0], ty: s.ty },
              { tx: s.tx - 1, ty: s.ty },
              { tx: s.tx, ty: s.ty + s.def.size[1] },
              { tx: s.tx, ty: s.ty - 1 },
            ];
            for (const n of neighbors) {
              const targetE = this.base.structureAt(n.tx, n.ty);
              if (targetE !== undefined) {
                const conv = this.ecs.get(targetE, Comp.Conveyor), chest = this.ecs.get(targetE, Comp.Structure)?.storage;
                if (conv && conv.items.length < 4) {
                  conv.items.push({ item: oreItem, progress: 0.1 });
                  break;
                } else if (chest) {
                  chest.add(oreItem, 1);
                  break;
                }
              }
            }
          }
        }
      } else if (mach.kind === 'stone_furnace') {
        if (mach.targetOre) {
          mach.progress += dt;
          if (mach.progress >= 2.8) {
            mach.progress = 0;
            mach.outputItem = mach.targetOre === 'iron_ore' ? 'iron_plate' : 'copper_wire';
            mach.targetOre = undefined;
          }
        }
      }
    }
  }

  // ---------------------------------------------------------------- inventory, equipment, crafting
  holdSlot(i: number, e: Entity = this.player): void {
    const ic = this.ecs.get(e, Comp.Inventory), ch = this.ecs.get(e, Comp.Character);
    if (!ic || !ch) return;
    ic.held = clamp(i, 0, HOTBAR - 1);
    const st = ic.inv.slots[ic.held];
    const def = st ? this.content.items.get(st.item) : undefined;
    if (def?.equip?.slot === 'mainHand') ch.look.outfit.mainHand = { item: def.id }; else delete ch.look.outfit.mainHand;
  }
  wear(itemOrIndex: number | string): boolean {
    const ic = this.playerInv;
    const invIndex = typeof itemOrIndex === 'number' ? itemOrIndex : ic.inv.slots.findIndex((s) => s?.item === itemOrIndex);
    if (invIndex < 0) return false;
    const st = ic.inv.slots[invIndex];
    const def = st ? this.content.items.get(st.item) : undefined;
    if (!st || !def?.equip) return false;
    const slot: Slot = def.equip.slot, look = this.playerChar.look;
    const prev = look.outfit[slot];
    ic.inv.remove(def.id, 1);
    if (prev && ic.inv.add(prev.item, 1) > 0) { ic.inv.add(def.id, 1); this.events.emit('toast', { text: 'Your pack is full' }); return false; }
    look.outfit[slot] = { item: def.id };
    if (def.equip.twoHanded) delete look.outfit.offHand;
    return true;
  }
  unwear(slot: Slot): boolean {
    const look = this.playerChar.look, cur = look.outfit[slot];
    if (!cur) return false;
    if (slot === 'mainHand') { delete look.outfit.mainHand; return true; }
    if (this.playerInv.inv.add(cur.item, 1) > 0) { this.events.emit('toast', { text: 'Your pack is full' }); return false; }
    delete look.outfit[slot];
    return true;
  }
  craft(recipeId: Id): boolean {
    const r = this.content.recipes.find((x) => x.id === recipeId), inv = this.playerInv.inv;
    if (!r || !inv.has(r.inputs)) return false;
    inv.take(r.inputs);
    if (inv.add(r.output.item, r.output.qty) > 0) { for (const [k, v] of Object.entries(r.inputs)) inv.add(k, v); this.events.emit('toast', { text: 'Your pack is full' }); return false; }
    this.events.emit('toast', { text: `Crafted ${this.content.items.get(r.output.item)?.name ?? r.id}` });
    return true;
  }
  eat(invIndex: number): boolean {
    const inv = this.playerInv.inv, st = inv.slots[invIndex], def = st ? this.content.items.get(st.item) : undefined;
    if (!st || !def?.food) return false;
    const n = this.playerNeeds;
    n.hunger = clamp(n.hunger + def.food.hunger, 0, n.maxHunger);
    n.health = clamp(n.health + (def.food.health ?? 0), 0, n.maxHealth);
    n.sanity = clamp(n.sanity + (def.food.sanity ?? 0), 0, n.maxSanity);

    if (def.id === 'hiccup_berries') {
      this.buffs.hiccupTimer = 30.0;
      this.events.emit('toast', { text: '🫧 Ate Hiccup Berries! Involuntary hops for 30s!' });
    } else if (def.id === 'giggle_mushroom') {
      this.events.emit('toast', { text: '🍄 Giggle Mushroom! Giggles restore sanity!' });
    } else if (def.id === 'chili_of_regret') {
      this.buffs.fireBreathTimer = 12.0;
      this.events.emit('toast', { text: '🔥 CHILI OF REGRET! Fiery breath scaring enemies!' });
    } else if (def.id === 'rainbow_soup') {
      this.buffs.rainbowGlowTimer = 60.0;
      this.events.emit('toast', { text: '🌈 Rainbow Soup! Glowing neon aura!' });
    } else if (def.id === 'crispy_sock_chips') {
      this.buffs.speedBoostTimer = 25.0;
      this.events.emit('toast', { text: '🧦 Crispy Sock Chips! Stamina speed surge!' });
    } else if (def.id === 'bubble_tea_of_buoyancy') {
      this.buffs.buoyancyTimer = 30.0;
      this.events.emit('toast', { text: '🫧 Bubble Tea! Floating buoyancy active!' });
    } else if (def.id === 'pancake_stack') {
      this.events.emit('toast', { text: '🥞 Fluffy Pancake Stack devoured! Maximum satisfaction!' });
    } else if (def.id === 'antidote_gel') {
      this.events.emit('toast', { text: '💉 Antidote Gel applied! System cleansed!' });
    }

    inv.remove(def.id, 1);
    return true;
  }

  throwItem(itemId: string, targetX: number, targetY: number): boolean {
    if (!this.playerInv.inv.remove(itemId, 1)) return false;
    this.events.emit('hit', { x: targetX, y: targetY, kind: 'rock' });
    if (itemId === 'cream_pie') {
      this.events.emit('toast', { text: '🥧 SPLAT! Custard cream pie blinded target!' });
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - targetX, zt.y - targetY) <= 3.0) {
          const z = this.ecs.get(ze, Comp.Zombie)!;
          z.state = 'wander';
          z.target = null;
        }
      }
    } else if (itemId === 'water_balloon') {
      this.events.emit('toast', { text: '💧 SPLASH! Water balloon drenched target!' });
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform, Comp.Body)) {
        const zt = this.ecs.get(ze, Comp.Transform)!, zb = this.ecs.get(ze, Comp.Body)!;
        if (Math.hypot(zt.x - targetX, zt.y - targetY) <= 3.2) {
          zb.vx *= 0.4;
          zb.vy *= 0.4;
        }
      }
    } else if (itemId === 'stink_sock_bomb') {
      this.events.emit('toast', { text: '🧦 PUFF! Stink bomb repels zombies!' });
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform, Comp.Body)) {
        const zt = this.ecs.get(ze, Comp.Transform)!, zb = this.ecs.get(ze, Comp.Body)!;
        if (Math.hypot(zt.x - targetX, zt.y - targetY) <= 4.5) {
          const ang = Math.atan2(zt.y - targetY, zt.x - targetX);
          zb.vx += Math.cos(ang) * 6.0;
          zb.vy += Math.sin(ang) * 6.0;
        }
      }
    } else if (itemId === 'whoopee_cushion_trap') {
      this.events.emit('toast', { text: '💨 PBBBT! Placed Whoopee Cushion Trap!' });
      for (const ze of this.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - targetX, zt.y - targetY) <= 1.8) {
          const z = this.ecs.get(ze, Comp.Zombie)!;
          z.attackTimer = 3.5;
          z.state = 'alert';
        }
      }
    } else if (itemId === 'road_flare') {
      this.events.emit('toast', { text: '🔥 Road Flare dropped! 60s radiant beacon!' });
      this.activeFlares.push({ x: targetX, y: targetY, timer: 60.0 });
    }
    return true;
  }

  // ---------------------------------------------------------------- building
  canAfford(def: StructureDef): boolean { return this.config.creative || this.playerInv.inv.has(def.cost); }
  checkBuild(def: StructureDef, tx: number, ty: number): { ok: boolean; reason?: string } {
    if (!this.canAfford(def)) return { ok: false, reason: 'Not enough materials' };
    const p = this.playerT;
    return this.base.checkPlace(def, tx, ty, { x: p.x, y: p.y, r: 0.35 });
  }
  build(defId: Id, tx: number, ty: number, dir: 'N' | 'E' | 'S' | 'W' = 'E'): Entity | null {
    const def = this.content.structures.get(defId);
    if (!def) return null;
    const c = this.checkBuild(def, tx, ty);
    if (!c.ok) { this.events.emit('toast', { text: c.reason ?? 'Cannot build here' }); return null; }
    if (!this.config.creative) this.playerInv.inv.take(def.cost);
    const e = this.base.place(def, tx, ty, dir);
    this.events.emit('built', { id: defId, entity: e });
    return e;
  }
  demolish(entity: Entity): boolean {
    const s = this.ecs.get(entity, Comp.Structure);
    if (!s) return false;
    const inv = this.playerInv.inv;
    for (const [k, v] of Object.entries(s.def.cost)) inv.add(k, Math.floor(v / 2));
    s.storage?.slots.forEach((st) => { if (st) inv.add(st.item, st.qty); });
    this.base.remove(entity);
    return true;
  }

  // ---------------------------------------------------------------- save and load
  toSave(): SaveData {
    const t = this.playerT, n = this.playerNeeds, ic = this.playerInv;
    return {
      v: 1, seed: this.config.seed, mode: this.config.mode, t: this.clock.t, day: this.clock.day, alwaysDay: this.clock.alwaysDay,
      player: { x: t.x, y: t.y, angle: t.angle, look: JSON.parse(JSON.stringify(this.playerChar.look)) as Look, inv: ic.inv.toJSON(), held: ic.held, needs: { hunger: n.hunger, health: n.health, sanity: n.sanity } },
      structures: this.base.entities().map((e) => {
        const s = this.ecs.get(e, Comp.Structure)!;
        const conv = this.ecs.get(e, Comp.Conveyor);
        const ins = this.ecs.get(e, Comp.Inserter);
        const mach = this.ecs.get(e, Comp.Machine);
        return {
          id: s.def.id, tx: s.tx, ty: s.ty, open: s.open, storage: s.storage?.toJSON(),
          conveyor: conv ? { dir: conv.dir, speed: conv.speed, items: conv.items } : undefined,
          inserter: ins ? { pickupDir: ins.pickupDir, dropDir: ins.dropDir, heldItem: ins.heldItem, armProgress: ins.armProgress, speed: ins.speed } : undefined,
          machine: mach ? { kind: mach.kind, fuel: mach.fuel, progress: mach.progress, targetOre: mach.targetOre, outputItem: mach.outputItem } : undefined,
        };
      }),
      vehicles: this.ecs.query(Comp.Vehicle).map((e) => {
        const v = this.ecs.get(e, Comp.Vehicle)!, vt = this.ecs.get(e, Comp.Transform)!;
        return { id: v.def.id, x: vt.x, y: vt.y, angle: vt.angle, hp: v.hp, fuel: v.fuel, trunk: v.trunk.toJSON() };
      }),
      zombies: this.ecs.query(Comp.Zombie).map((e) => {
        const z = this.ecs.get(e, Comp.Zombie)!, zt = this.ecs.get(e, Comp.Transform)!;
        return { id: z.def.id, x: zt.x, y: zt.y, angle: zt.angle, hp: z.hp };
      }),
      pets: this.ecs.query(Comp.Pet).map((e) => {
        const p = this.ecs.get(e, Comp.Pet)!, pt = this.ecs.get(e, Comp.Transform)!;
        return { name: p.name, kind: p.kind, x: pt.x, y: pt.y, angle: pt.angle, hp: p.hp };
      }),
      animals: this.ecs.query(Comp.Animal, Comp.Transform).map((e) => {
        const a = this.ecs.get(e, Comp.Animal)!, at = this.ecs.get(e, Comp.Transform)!;
        return { kind: a.kind, name: a.name, x: at.x, y: at.y, angle: at.angle, hp: a.hp, tamed: a.tamed, produceTimer: a.produceTimer };
      }),
      colonists: this.ecs.query(Comp.Pawn, Comp.Character, Comp.Transform).map((e) => {
        const p = this.ecs.get(e, Comp.Pawn)!;
        const ch = this.ecs.get(e, Comp.Character)!;
        const pt = this.ecs.get(e, Comp.Transform)!;
        return {
          name: ch.name,
          look: JSON.parse(JSON.stringify(ch.look)),
          x: pt.x,
          y: pt.y,
          angle: pt.angle,
          pawn: JSON.parse(JSON.stringify(p)),
        };
      }),
      colony: this.colony.toSave(),
      siege: this.siege.toSave(),
      faction: this.faction.toSave(),
      buffs: { ...this.buffs },
      explored: [...this.explored],
      mods: [...this.world.mods.entries()],
      staticMap: this.world.saveStatic(),
      staticActive: this.world.staticActive,
      baseStorage: this.baseStorage.toJSON(),
      gunMods: JSON.parse(JSON.stringify(this.gunMods)),
    };
  }
  static fromSave(data: SaveData, config?: Partial<GameConfig>, packs?: ContentPack[]): Game {
    const g = new Game({ ...DEFAULT_CONFIG, ...config, seed: data.seed, mode: data.mode, alwaysDay: data.alwaysDay }, packs, false, false, false);
    g.clock.t = data.t; g.clock.day = data.day;
    if (data.alwaysDay !== undefined) g.clock.alwaysDay = data.alwaysDay;
    if (data.baseStorage) g.baseStorage.load(data.baseStorage);
    if (data.gunMods) Object.assign(g.gunMods, data.gunMods);
    data.mods.forEach(([id, m]) => g.world.mods.set(id, m));
    if (data.staticMap) g.world.loadStatic(data.staticMap, !!data.staticActive);
    if (data.explored && Array.isArray(data.explored)) {
      for (const k of data.explored) g.explored.add(k);
    } else {
      g.revealArea(g.world.spawn.x, g.world.spawn.y, 52);
    }
    const t = g.playerT; t.x = data.player.x; t.y = data.player.y; t.angle = data.player.angle;
    const ch = g.playerChar; ch.look = sanitizeLook(g.content, data.player.look);
    g.playerInv.inv.load(data.player.inv); g.playerInv.held = clamp(data.player.held, 0, HOTBAR - 1);
    Object.assign(g.playerNeeds, data.player.needs);
    for (const s of data.structures) {
      const def = g.content.structures.get(s.id);
      if (!def) continue;
      const dir = (s as any).conveyor?.dir ?? (s as any).inserter?.dropDir ?? 'E';
      const e = g.base.place(def, s.tx, s.ty, dir);
      const sc = g.ecs.get(e, Comp.Structure)!; sc.open = s.open; sc.storage?.load(s.storage);
      if ((s as any).conveyor && g.ecs.has(e, Comp.Conveyor)) {
        Object.assign(g.ecs.get(e, Comp.Conveyor)!, (s as any).conveyor);
      }
      if ((s as any).inserter && g.ecs.has(e, Comp.Inserter)) {
        Object.assign(g.ecs.get(e, Comp.Inserter)!, (s as any).inserter);
      }
      if ((s as any).machine && g.ecs.has(e, Comp.Machine)) {
        Object.assign(g.ecs.get(e, Comp.Machine)!, (s as any).machine);
      }
    }
    if (data.vehicles && data.vehicles.length > 0) {
      for (const vData of data.vehicles) {
        const vEntity = g.spawnVehicle(vData.x, vData.y, vData.id);
        const vc = g.ecs.get(vEntity, Comp.Vehicle)!;
        vc.hp = vData.hp;
        vc.fuel = vData.fuel;
        if (vData.trunk) vc.trunk.load(vData.trunk);
        const vt = g.ecs.get(vEntity, Comp.Transform)!;
        vt.angle = vData.angle;
      }
    } else {
      const slots = g.townCompanionSlots('plaza');
      g.spawnVehicle(slots.vehicle.x, slots.vehicle.y);
    }
    if (data.zombies && data.zombies.length > 0) {
      for (const zData of data.zombies) {
        const ze = g.spawnZombie(zData.x, zData.y, zData.id);
        const zc = g.ecs.get(ze, Comp.Zombie)!;
        zc.hp = zData.hp;
        const zt = g.ecs.get(ze, Comp.Transform)!;
        zt.angle = zData.angle;
      }
    }
    if (data.pets && data.pets.length > 0) {
      for (const pData of data.pets) {
        const pe = g.spawnPet(pData.x, pData.y, pData.name, pData.kind);
        const pc = g.ecs.get(pe, Comp.Pet)!;
        pc.hp = pData.hp;
        const pt = g.ecs.get(pe, Comp.Transform)!;
        pt.angle = pData.angle;
      }
    } else if (data.pets === undefined) {
      const slots = g.townCompanionSlots('plaza');
      g.spawnPet(slots.pet.x, slots.pet.y, 'Barkley', 'hunting_dog');
    }
    if (data.animals && data.animals.length > 0) {
      for (const aData of data.animals) {
        const ae = g.spawnAnimal(aData.x, aData.y, aData.kind, aData.name);
        const ac = g.ecs.get(ae, Comp.Animal)!;
        ac.hp = aData.hp;
        ac.tamed = aData.tamed;
        ac.produceTimer = aData.produceTimer;
        const at = g.ecs.get(ae, Comp.Transform)!;
        at.angle = aData.angle;
      }
    }
    if (data.colony) {
      g.colony.load(data.colony as Partial<ColonySaveData>);
    }
    if (data.siege) {
      g.siege.load(data.siege as Partial<SiegeSaveData>);
    }
    if (data.faction) {
      g.faction.load(data.faction as Partial<FactionSaveData>);
    }
    if (data.buffs) {
      Object.assign(g.buffs, data.buffs);
    }
    if (data.colonists && data.colonists.length > 0) {
      for (const e of g.ecs.query(Comp.Pawn)) {
        g.ecs.destroy(e);
      }
      for (const cData of data.colonists) {
        const ce = g.spawnColonist(cData.x, cData.y, cData.name);
        const ch = g.ecs.get(ce, Comp.Character)!;
        if (cData.look) ch.look = sanitizeLook(g.content, cData.look as Look);
        const ct = g.ecs.get(ce, Comp.Transform)!;
        ct.angle = cData.angle;
        if (cData.pawn) {
          Object.assign(g.ecs.get(ce, Comp.Pawn)!, cData.pawn);
        }
      }
    }
    return g;
  }
}
