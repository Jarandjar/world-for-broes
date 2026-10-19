import type { Ecs, Entity } from '../engine/ecs';
import type { Rng } from '../engine/rng';
import { clamp, lerpAngle, damp } from '../engine/math';
import type { Game } from './game';
import { Comp, type PawnC, type PawnThought, type PawnJob, newAnim } from './components';
import type {
  PawnSkill,
  PawnPassion,
  PawnTrait,
  MentalBreakKind,
  StorytellerKind,
  IncidentKind,
  PawnWorkRole,
} from '../data/schema';
import { randomLook } from './look';

export const ALL_SKILLS: PawnSkill[] = [
  'construction',
  'mining',
  'crafting',
  'cooking',
  'medicine',
  'plants',
  'combat',
  'hauling',
];

export const ALL_ROLES: PawnWorkRole[] = [
  'firefight',
  'patient',
  'doctor',
  'build',
  'farm',
  'mine',
  'haul',
  'craft',
  'guard',
];

export const ALL_TRAITS: PawnTrait[] = [
  'grease_monkey',
  'pyromaniac',
  'slacker',
  'brawler',
  'night_owl',
  'gourmand',
  'nervous_wreck',
  'clumsy',
  'fast_walker',
];

export const PAWN_NAMES = [
  'Bucky', 'Sarge', 'Rusty', 'Faye', 'Otis', 'Milo', 'Clover', 'Hank',
  'Dottie', 'Gus', 'Pepper', 'Boomer', 'Hazel', 'Ziggy', 'Scout', 'Daisy',
];

export function createRandomPawn(rng: Rng): PawnC {
  const skills: PawnC['skills'] = {
    construction: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    mining: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    crafting: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    cooking: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    medicine: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    plants: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    combat: { level: rng.int(1, 9), passion: 'none', xp: 0 },
    hauling: { level: rng.int(2, 10), passion: 'none', xp: 0 },
  };

  // Assign 1-2 passions
  const shuffledSkills = [...ALL_SKILLS].sort(() => rng.next() - 0.5);
  skills[shuffledSkills[0]].passion = rng.chance(0.35) ? 'burning' : 'single';
  if (rng.chance(0.6)) {
    skills[shuffledSkills[1]].passion = 'single';
  }

  // Assign 1-2 traits
  const traitRolls = [...ALL_TRAITS].sort(() => rng.next() - 0.5);
  const traits: PawnTrait[] = [traitRolls[0]];
  if (rng.chance(0.4)) traits.push(traitRolls[1]);

  // Default work priorities (1 = highest, 4 = lowest, 0 = disabled)
  const workPriorities: Record<PawnWorkRole, number> = {
    firefight: 1,
    patient: 1,
    doctor: skills.medicine.level >= 5 ? 2 : 3,
    build: skills.construction.level >= 5 ? 2 : 3,
    farm: skills.plants.level >= 5 ? 2 : 3,
    mine: skills.mining.level >= 5 ? 2 : 3,
    haul: 2,
    craft: skills.crafting.level >= 5 ? 2 : 3,
    guard: skills.combat.level >= 5 ? 2 : 4,
    research: 3,
  };

  return {
    skills,
    traits,
    mood: 80,
    thoughts: [{ id: 'new_colony', label: 'New Hope (+10)', delta: 10, duration: 180 }],
    mentalBreak: { kind: 'none', timer: 0 },
    workPriorities,
    currentJob: null,
  };
}

export interface ColonySaveData {
  storyteller: StorytellerKind;
  incidentTimer: number;
}

export class ColonySystem {
  storyteller: StorytellerKind = 'caspar_chill';
  incidentTimer = 0;

  constructor(storyteller?: StorytellerKind) {
    if (storyteller) this.storyteller = storyteller;
  }

  /** Advances mood, mental breaks, work priorities, and the storyteller AI. */
  update(dt: number, game: Game): void {
    this.updateMoodAndBreaks(dt, game);
    this.updateWorkAI(dt, game);
    this.updateStoryteller(dt, game);
  }

  private updateMoodAndBreaks(dt: number, game: Game): void {
    const isNight = game.clock.darkness > 0.45;

    for (const e of game.ecs.query(Comp.Pawn, Comp.Character, Comp.Transform)) {
      const pawn = game.ecs.get(e, Comp.Pawn)!;
      const t = game.ecs.get(e, Comp.Transform)!;

      // 1. Tick thought timers and filter expired
      for (let i = pawn.thoughts.length - 1; i >= 0; i--) {
        pawn.thoughts[i].duration -= dt;
        if (pawn.thoughts[i].duration <= 0) pawn.thoughts.splice(i, 1);
      }

      // 2. Environmental passive thoughts
      const nearFire = game.base.entities().some((se) => {
        const s = game.ecs.get(se, Comp.Structure);
        const st = game.ecs.get(se, Comp.Transform);
        return s?.def.id === 'campfire' && st && Math.hypot(st.x - t.x, st.y - t.y) < 4.5;
      });
      const nearTable = game.base.entities().some((se) => {
        const s = game.ecs.get(se, Comp.Structure);
        const st = game.ecs.get(se, Comp.Transform);
        return s?.def.id === 'grand_banquet_table' && st && Math.hypot(st.x - t.x, st.y - t.y) < 6.0;
      });
      const nearRecreation = game.base.entities().some((se) => {
        const s = game.ecs.get(se, Comp.Structure);
        const st = game.ecs.get(se, Comp.Transform);
        return s?.def.id === 'horseshoe_pin_post' && st && Math.hypot(st.x - t.x, st.y - t.y) < 5.0;
      });

      // Warmth thought
      this.ensureThought(pawn, 'warm_fire', 'Comfortable Warmth (+6)', 6, nearFire ? 3 : 0);
      // Table thought vs ate without table
      if (nearTable) {
        this.ensureThought(pawn, 'ate_table', 'Ate at Dining Table (+6)', 6, 20);
        this.removeThought(pawn, 'no_table');
      }
      // Horseshoe recreation
      this.ensureThought(pawn, 'horseshoes', 'Horseshoe Toss (+8)', 8, nearRecreation ? 25 : 0);

      // Trait-specific thoughts
      if (pawn.traits.includes('night_owl')) {
        this.ensureThought(pawn, 'night_owl_perk', isNight ? 'Night Owl Thriving (+14)' : 'Groggy Day (-8)', isNight ? 14 : -8, 5);
      }
      if (pawn.traits.includes('slacker')) {
        this.ensureThought(pawn, 'slacker_zen', 'Taking it Easy (+8)', 8, 5);
      }

      // Darkness penalty
      if (isNight && !game.isLit(t.x, t.y)) {
        this.ensureThought(pawn, 'in_dark', 'Creepy Darkness (-8)', -8, 4);
      }

      // Calculate target mood
      let netDelta = 0;
      for (const th of pawn.thoughts) netDelta += th.delta;
      const targetMood = clamp(60 + netDelta, 5, 100);
      pawn.mood = clamp(pawn.mood + (targetMood - pawn.mood) * dt * 0.12, 0, 100);

      // 3. Mental Break Check
      const breakThreshold = pawn.traits.includes('nervous_wreck') ? 30 : 20;
      if (pawn.mentalBreak.kind === 'none') {
        const deficit = breakThreshold - pawn.mood;
        if (deficit > 0 && (pawn.mood < 10 || game.rng.chance(dt * 0.25))) {
          const breaks: MentalBreakKind[] = ['binge_eating', 'tool_tantrum', 'conveyor_carousel', 'sad_wander'];
          const chosen = breaks[Math.floor(game.rng.range(0, breaks.length))];
          pawn.mentalBreak = { kind: chosen, timer: 12 };
          const ch = game.ecs.get(e, Comp.Character)!;
          game.events.emit('toast', { text: `😤 ${ch.name} snapped: ${chosen.replace('_', ' ').toUpperCase()}!` });
        }
      } else {
        pawn.mentalBreak.timer -= dt;
        if (pawn.mentalBreak.timer <= 0) {
          pawn.mentalBreak.kind = 'none';
          pawn.thoughts.push({ id: 'catharsis', label: 'Catharsis (+20)', delta: 20, duration: 90 });
          const ch = game.ecs.get(e, Comp.Character)!;
          game.events.emit('toast', { text: `😌 ${ch.name} recovered from mental break.` });
        }
      }
    }
  }

  private ensureThought(pawn: PawnC, id: string, label: string, delta: number, dur: number): void {
    if (dur <= 0) {
      this.removeThought(pawn, id);
      return;
    }
    const existing = pawn.thoughts.find((th) => th.id === id);
    if (existing) {
      existing.duration = Math.max(existing.duration, dur);
    } else {
      pawn.thoughts.push({ id, label, delta, duration: dur });
    }
  }

  private removeThought(pawn: PawnC, id: string): void {
    const idx = pawn.thoughts.findIndex((th) => th.id === id);
    if (idx >= 0) pawn.thoughts.splice(idx, 1);
  }

  private updateWorkAI(dt: number, game: Game): void {
    for (const e of game.ecs.query(Comp.Pawn, Comp.Character, Comp.Transform, Comp.Body)) {
      const pawn = game.ecs.get(e, Comp.Pawn)!;
      const t = game.ecs.get(e, Comp.Transform)!;
      const b = game.ecs.get(e, Comp.Body)!;
      const ch = game.ecs.get(e, Comp.Character)!;

      // In mental break: execute slapstick breakdown
      if (pawn.mentalBreak.kind !== 'none') {
        this.executeMentalBreak(e, pawn, t, b, ch, dt, game);
        continue;
      }

      // If already working on a job
      if (pawn.currentJob) {
        this.executeJob(e, pawn, t, b, ch, dt, game);
        continue;
      }

      // Scan for top priority job (1 = emergency, 4 = idle)
      const sortedRoles = (Object.keys(pawn.workPriorities) as PawnWorkRole[])
        .filter((r) => pawn.workPriorities[r] > 0)
        .sort((r1, r2) => pawn.workPriorities[r1] - pawn.workPriorities[r2]);

      for (const role of sortedRoles) {
        const job = this.findJobForRole(e, role, t, game);
        if (job) {
          pawn.currentJob = job;
          break;
        }
      }
    }
  }

  private executeMentalBreak(
    e: Entity,
    pawn: PawnC,
    t: { x: number; y: number; angle: number },
    b: { speed: number; vx: number; vy: number },
    ch: { name: string; anim: { state: string; walkBlend: number } },
    dt: number,
    game: Game
  ): void {
    if (pawn.mentalBreak.kind === 'sad_wander') {
      // Wander aimlessly
      const ang = Math.sin(game.simTime * 0.8 + e) * Math.PI;
      b.vx = Math.cos(ang) * 1.0;
      b.vy = Math.sin(ang) * 1.0;
      t.angle = lerpAngle(t.angle, ang, 0.1);
      ch.anim.state = 'walk';
      ch.anim.walkBlend = 1;
    } else if (pawn.mentalBreak.kind === 'conveyor_carousel') {
      // Find nearest belt and ride on it
      let nearestBelt: Entity | undefined;
      let minDist = Infinity;
      for (const be of game.ecs.query(Comp.Conveyor, Comp.Transform)) {
        const bt = game.ecs.get(be, Comp.Transform)!;
        const d = Math.hypot(bt.x - t.x, bt.y - t.y);
        if (d < minDist) { minDist = d; nearestBelt = be; }
      }
      if (nearestBelt !== undefined) {
        const bt = game.ecs.get(nearestBelt, Comp.Transform)!;
        const conv = game.ecs.get(nearestBelt, Comp.Conveyor)!;
        if (minDist > 0.4) {
          const dx = bt.x - t.x, dy = bt.y - t.y;
          const a = Math.atan2(dy, dx);
          b.vx = Math.cos(a) * 1.5; b.vy = Math.sin(a) * 1.5;
          t.angle = a;
        } else {
          // Riding on belt!
          const dirOffsets: Record<'N'|'E'|'S'|'W', { dx: number; dy: number }> = {
            N: { dx: 0, dy: -1 }, E: { dx: 1, dy: 0 }, S: { dx: 0, dy: 1 }, W: { dx: -1, dy: 0 },
          };
          const off = dirOffsets[conv.dir];
          b.vx = off.dx * 1.2;
          b.vy = off.dy * 1.2;
          t.angle = Math.atan2(off.dy, off.dx);
        }
      }
    } else if (pawn.mentalBreak.kind === 'tool_tantrum') {
      // Smack nearest object / structure
      b.vx = 0; b.vy = 0;
      ch.anim.state = 'swing';
    } else {
      b.vx = 0; b.vy = 0;
      ch.anim.state = 'idle';
    }
  }

  private findJobForRole(e: Entity, role: PawnWorkRole, t: { x: number; y: number }, game: Game): PawnJob | null {
    if (role === 'guard') {
      // Guard: find nearest zombie within 20m
      for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = game.ecs.get(ze, Comp.Transform)!;
        if (Math.hypot(zt.x - t.x, zt.y - t.y) < 20) {
          return { role: 'guard', targetEntity: ze, state: 'walking', progress: 0, duration: 1.5 };
        }
      }
    } else if (role === 'mine') {
      // Mine: find nearest mineral node within 25m
      const nodes = game.world.objectsNear(t.x, t.y, 25).filter((o) =>
        o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode'
      );
      if (nodes.length > 0) {
        nodes.sort((a, b) => Math.hypot(a.x - t.x, a.y - t.y) - Math.hypot(b.x - t.x, b.y - t.y));
        return { role: 'mine', targetPos: { x: nodes[0].x, y: nodes[0].y }, state: 'walking', progress: 0, duration: 2.2 };
      }
    } else if (role === 'craft') {
      // Craft / Smelt: find empty furnace
      for (const fe of game.ecs.query(Comp.Machine, Comp.Transform)) {
        const mach = game.ecs.get(fe, Comp.Machine)!;
        const ft = game.ecs.get(fe, Comp.Transform)!;
        if (mach.kind === 'stone_furnace' && !mach.targetOre && Math.hypot(ft.x - t.x, ft.y - t.y) < 22) {
          return { role: 'craft', targetEntity: fe, targetPos: { x: ft.x, y: ft.y }, state: 'walking', progress: 0, duration: 2.0 };
        }
      }
    } else if (role === 'haul') {
      // Haul: find chest to organize
      for (const che of game.ecs.query(Comp.Structure, Comp.Transform)) {
        const s = game.ecs.get(che, Comp.Structure)!;
        const ct = game.ecs.get(che, Comp.Transform)!;
        if (s.def.interact === 'storage' && Math.hypot(ct.x - t.x, ct.y - t.y) < 20) {
          return { role: 'haul', targetEntity: che, targetPos: { x: ct.x, y: ct.y }, state: 'walking', progress: 0, duration: 1.2 };
        }
      }
    }
    return null;
  }

  private executeJob(
    _e: Entity,
    pawn: PawnC,
    t: { x: number; y: number; angle: number },
    b: { speed: number; vx: number; vy: number },
    ch: { name: string; anim: { state: string; walkBlend: number } },
    dt: number,
    game: Game
  ): void {
    const job = pawn.currentJob!;
    let targetX = job.targetPos?.x ?? t.x;
    let targetY = job.targetPos?.y ?? t.y;

    if (job.targetEntity !== undefined) {
      const tgt = game.ecs.get(job.targetEntity, Comp.Transform);
      if (tgt) { targetX = tgt.x; targetY = tgt.y; }
    }

    const dist = Math.hypot(targetX - t.x, targetY - t.y);

    if (job.state === 'walking') {
      if (dist > 1.4) {
        const a = Math.atan2(targetY - t.y, targetX - t.x);
        b.vx = Math.cos(a) * 1.8;
        b.vy = Math.sin(a) * 1.8;
        t.angle = lerpAngle(t.angle, a, 0.15);
        ch.anim.state = 'walk';
        ch.anim.walkBlend = 1;
      } else {
        job.state = 'working';
        b.vx = 0; b.vy = 0;
        ch.anim.state = 'swing';
      }
    } else if (job.state === 'working') {
      b.vx = 0; b.vy = 0;
      job.progress += dt;
      if (job.progress >= job.duration) {
        // Job complete!
        if (job.role === 'guard' && job.targetEntity) {
          const zc = game.ecs.get(job.targetEntity, Comp.Zombie);
          if (zc) zc.hp -= 8;
        } else if (job.role === 'mine' && job.targetPos) {
          pawn.skills.mining.xp += 15;
          if (pawn.skills.mining.passion === 'burning') pawn.mood = Math.min(100, pawn.mood + 2);
        } else if (job.role === 'craft' && job.targetEntity) {
          const mach = game.ecs.get(job.targetEntity, Comp.Machine);
          if (mach && mach.kind === 'stone_furnace') mach.targetOre = 'iron_ore';
        }
        pawn.currentJob = null;
        ch.anim.state = 'idle';
      }
    }
  }

  private updateStoryteller(dt: number, game: Game): void {
    this.incidentTimer += dt;
    const interval = this.storyteller === 'caspar_chill' ? 120
      : this.storyteller === 'randy_ruckus' ? 55
      : 40;

    if (this.incidentTimer >= interval) {
      this.incidentTimer = 0;
      const roll = game.rng.next();
      let kind: IncidentKind;
      if (this.storyteller === 'caspar_chill') {
        kind = roll < 0.6 ? 'wanderer_joins' : 'supply_drop';
      } else if (this.storyteller === 'hilda_hardcore') {
        kind = roll < 0.5 ? 'screamer_swarm' : roll < 0.8 ? 'short_circuit' : 'supply_drop';
      } else {
        // Randy Ruckus: fully randomized
        kind = roll < 0.35 ? 'wanderer_joins' : roll < 0.65 ? 'supply_drop' : roll < 0.85 ? 'stampede' : 'short_circuit';
      }
      this.triggerIncident(kind, game);
    }
  }

  triggerIncident(kind: IncidentKind, game: Game): string {
    const pt = game.playerT;
    if (kind === 'wanderer_joins') {
      const name = PAWN_NAMES[Math.floor(game.rng.range(0, PAWN_NAMES.length))];
      const colonist = game.spawnColonist(pt.x + game.rng.range(-4, 4), pt.y + game.rng.range(-4, 4), name);
      const pawn = game.ecs.get(colonist, Comp.Pawn)!;
      const traitName = pawn.traits[0].replace('_', ' ');
      const msg = `🌟 Wanderer Arrives: ${name} (${traitName}) joined the colony!`;
      game.events.emit('toast', { text: msg });
      return msg;
    } else if (kind === 'supply_drop') {
      // Parachute supply crate
      const sx = pt.x + game.rng.range(-6, 6);
      const sy = pt.y + game.rng.range(-6, 6);
      game.events.emit('toast', { text: '🪂 Supply Drop parachuted down near your camp!' });
      // Add items directly to player or nearby chest
      game.playerInv.inv.add('scrap_metal', 4);
      game.playerInv.inv.add('canned_beans', 2);
      return 'Supply Drop Arrived';
    } else if (kind === 'short_circuit') {
      game.events.emit('toast', { text: '⚡ Short Circuit! A power line popped with sparks.' });
      return 'Short Circuit';
    } else if (kind === 'screamer_swarm') {
      if (game.config.mode === 'zombie') {
        game.spawnZombie(pt.x + 14, pt.y + 14, 'runner');
        game.spawnZombie(pt.x - 14, pt.y - 14, 'runner');
        game.events.emit('toast', { text: '🚨 Screamer Swarm detected! Undead sprinting toward colony!' });
      }
      return 'Screamer Swarm';
    } else {
      game.events.emit('toast', { text: '🐾 A herd of wild critters dashed past the perimeter.' });
      return 'Stampede';
    }
  }

  toSave(): ColonySaveData {
    return {
      storyteller: this.storyteller,
      incidentTimer: this.incidentTimer,
    };
  }

  load(data: Partial<ColonySaveData>): void {
    if (data.storyteller) this.storyteller = data.storyteller;
    if (data.incidentTimer !== undefined) this.incidentTimer = data.incidentTimer;
  }
}
