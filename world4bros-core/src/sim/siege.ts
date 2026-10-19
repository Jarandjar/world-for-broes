import type { Game } from './game';
import { Comp, type TurretC } from './components';
import type { Entity } from '../engine/ecs';
import { clamp } from '../engine/math';

export interface SiegeSaveData {
  pollution: [string, number][];
  siegeTimer: number;
  totalSieges: number;
}

export class SiegeSystem {
  readonly pollutionMap = new Map<string, number>();
  siegeTimer = 0;
  totalSieges = 0;
  private checkTimer = 0;
  private screamerToastTimer = 0;
  private spitterToastTimer = 0;

  getPollutionAt(x: number, y: number): number {
    const cx = Math.floor(x / 16), cy = Math.floor(y / 16);
    return this.pollutionMap.get(`${cx},${cy}`) ?? 0;
  }

  addPollution(x: number, y: number, amount: number): void {
    const cx = Math.floor(x / 16), cy = Math.floor(y / 16);
    const key = `${cx},${cy}`;
    const cur = this.pollutionMap.get(key) ?? 0;
    this.pollutionMap.set(key, Math.max(0, cur + amount));
  }

  update(dt: number, game: Game): void {
    if (this.screamerToastTimer > 0) this.screamerToastTimer -= dt;
    if (this.spitterToastTimer > 0) this.spitterToastTimer -= dt;

    // 1. Machines generate pollution
    for (const e of game.ecs.query(Comp.Machine, Comp.Transform)) {
      const m = game.ecs.get(e, Comp.Machine)!, t = game.ecs.get(e, Comp.Transform)!;
      if (m.kind === 'stone_furnace' && (m.progress > 0 || m.fuel > 0)) {
        this.addPollution(t.x, t.y, dt * 0.8);
      } else if (m.kind === 'burner_drill' && m.fuel > 0) {
        this.addPollution(t.x, t.y, dt * 0.5);
      }
    }

    // 2. Slow natural dissipation
    this.checkTimer += dt;
    if (this.checkTimer > 2.0) {
      this.checkTimer = 0;
      for (const [k, v] of this.pollutionMap.entries()) {
        const next = v - 0.1;
        if (next <= 0.05) this.pollutionMap.delete(k);
        else this.pollutionMap.set(k, next);
      }
    }

    // 3. Check for high pollution triggering siege assault
    this.siegeTimer += dt;
    if (this.siegeTimer > 60.0) {
      this.siegeTimer = 0;
      const pt = game.playerT;
      const localPollution = this.getPollutionAt(pt.x, pt.y);
      if (localPollution >= 40 || (game.config.mode === 'zombie' && localPollution >= 20)) {
        this.triggerIndustrialSiege(game, pt.x, pt.y);
      }
    }

    // 4. Update automated defense turrets
    this.updateDefenses(dt, game);

    // 5. Update specialized siege mutants behavior (sappers, spitters, burrowers)
    this.updateSiegeMutants(dt, game);
  }

  triggerIndustrialSiege(game: Game, centerX: number, centerY: number): void {
    this.totalSieges++;
    game.events.emit('toast', { text: '🚨 INDUSTRIAL SIEGE: Dense pollution alerted mutant horde!' });

    // Spawn specialized siege mutants in perimeter ring
    const mutants: ('sapper' | 'spitter' | 'burrower' | 'screamer' | 'goliath')[] = [
      'sapper', 'sapper', 'spitter', 'burrower', 'screamer'
    ];
    for (let i = 0; i < mutants.length; i++) {
      const ang = (i / mutants.length) * Math.PI * 2 + game.rng.range(-0.3, 0.3);
      const dist = game.rng.range(14, 22);
      const mx = centerX + Math.cos(ang) * dist;
      const my = centerY + Math.sin(ang) * dist;
      game.spawnZombie(mx, my, mutants[i]);
    }
  }

  private updateDefenses(dt: number, game: Game): void {
    for (const e of game.ecs.query(Comp.Structure, Comp.Transform)) {
      const s = game.ecs.get(e, Comp.Structure)!, t = game.ecs.get(e, Comp.Transform)!;

      // Auto-attach Turret component if not present
      if (!game.ecs.has(e, Comp.Turret)) {
        if (s.def.id === 'belt_bolter') {
          game.ecs.add(e, Comp.Turret, { kind: 'bolt', range: 14, damage: 18, cooldown: 0.9, timer: 0, ammo: 30, maxAmmo: 60 });
        } else if (s.def.id === 'flame_turret') {
          game.ecs.add(e, Comp.Turret, { kind: 'flame', range: 8.5, damage: 14, cooldown: 0.5, timer: 0, ammo: 20, maxAmmo: 40 });
        } else if (s.def.id === 'zap_fence') {
          game.ecs.add(e, Comp.Turret, { kind: 'zap', range: 0.95, damage: 25, cooldown: 1.2, timer: 0, ammo: 999, maxAmmo: 999 });
        } else if (s.def.id === 'mallet_catapult') {
          game.ecs.add(e, Comp.Turret, { kind: 'mallet', range: 6.0, damage: 40, cooldown: 2.5, timer: 0, ammo: 999, maxAmmo: 999 });
        } else if (s.def.id === 'repair_drone_bay') {
          game.ecs.add(e, Comp.Turret, { kind: 'drone', range: 20, damage: 0, cooldown: 1.0, timer: 0, ammo: 100, maxAmmo: 100 });
        } else {
          continue;
        }
      }

      const turret = game.ecs.get(e, Comp.Turret)!;
      turret.timer = Math.max(0, turret.timer - dt);

      // Auto-feed ammo from adjacent conveyor belts
      if (turret.ammo < turret.maxAmmo) {
        for (const convE of game.ecs.query(Comp.Conveyor, Comp.Transform)) {
          const ct = game.ecs.get(convE, Comp.Transform)!;
          if (Math.hypot(ct.x - t.x, ct.y - t.y) <= 1.5) {
            const conv = game.ecs.get(convE, Comp.Conveyor)!;
            const ammoIdx = conv.items.findIndex((it) => it.item === 'turret_ammo' || it.item === 'flame_fuel');
            if (ammoIdx >= 0) {
              conv.items.splice(ammoIdx, 1);
              turret.ammo = Math.min(turret.maxAmmo, turret.ammo + 10);
            }
          }
        }
      }

      // Operational Turret Logic
      if (turret.kind === 'bolt' && turret.timer <= 0 && turret.ammo > 0) {
        // Find nearest zombie within range
        let nearestZ: Entity | null = null, minDist = turret.range;
        for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform)) {
          const zt = game.ecs.get(ze, Comp.Transform)!;
          const d = Math.hypot(zt.x - t.x, zt.y - t.y);
          if (d < minDist) { minDist = d; nearestZ = ze; }
        }
        if (nearestZ !== null) {
          turret.timer = turret.cooldown;
          turret.ammo--;
          const zc = game.ecs.get(nearestZ, Comp.Zombie)!;
          const zt = game.ecs.get(nearestZ, Comp.Transform)!;
          zc.hp -= turret.damage;
          game.events.emit('hit', { x: zt.x, y: zt.y, kind: 'rock' });
          if (zc.hp <= 0) {
            game.ecs.destroy(nearestZ);
            game.events.emit('zombieKilled', { kind: zc.def.kind, x: zt.x, y: zt.y });
          }
        }
      } else if (turret.kind === 'flame' && turret.timer <= 0 && turret.ammo > 0) {
        let fired = false;
        for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform)) {
          const zt = game.ecs.get(ze, Comp.Transform)!;
          const d = Math.hypot(zt.x - t.x, zt.y - t.y);
          if (d <= turret.range) {
            const zc = game.ecs.get(ze, Comp.Zombie)!;
            zc.hp -= turret.damage;
            game.events.emit('hit', { x: zt.x, y: zt.y, kind: 'flesh' });
            fired = true;
            if (zc.hp <= 0) {
              game.ecs.destroy(ze);
              game.events.emit('zombieKilled', { kind: zc.def.kind, x: zt.x, y: zt.y });
            }
          }
        }
        if (fired) {
          turret.timer = turret.cooldown;
          turret.ammo--;
        }
      } else if (turret.kind === 'zap') {
        // Zap fence contact shock
        for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform, Comp.Body)) {
          const zt = game.ecs.get(ze, Comp.Transform)!, zb = game.ecs.get(ze, Comp.Body)!;
          const d = Math.hypot(zt.x - t.x, zt.y - t.y);
          if (d < turret.range) {
            const zc = game.ecs.get(ze, Comp.Zombie)!;
            zc.hp -= turret.damage;
            // Knockback
            const ang = Math.atan2(zt.y - t.y, zt.x - t.x);
            zb.vx += Math.cos(ang) * 4.0;
            zb.vy += Math.sin(ang) * 4.0;
            game.events.emit('hit', { x: zt.x, y: zt.y, kind: 'rock' });
            if (zc.hp <= 0) {
              game.ecs.destroy(ze);
              game.events.emit('zombieKilled', { kind: zc.def.kind, x: zt.x, y: zt.y });
            }
          }
        }
      } else if (turret.kind === 'mallet' && turret.timer <= 0) {
        // Mallet catapult smacks group of zombies
        const inRange: Entity[] = [];
        for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform)) {
          const zt = game.ecs.get(ze, Comp.Transform)!;
          if (Math.hypot(zt.x - t.x, zt.y - t.y) <= turret.range) inRange.push(ze);
        }
        if (inRange.length > 0) {
          turret.timer = turret.cooldown;
          game.events.emit('toast', { text: '🔨 BONK! Mallet catapult launched!' });
          for (const ze of inRange) {
            const zc = game.ecs.get(ze, Comp.Zombie)!, zt = game.ecs.get(ze, Comp.Transform)!, zb = game.ecs.get(ze, Comp.Body);
            zc.hp -= turret.damage;
            if (zb) {
              const ang = Math.atan2(zt.y - t.y, zt.x - t.x);
              zb.vx += Math.cos(ang) * 8.0;
              zb.vy += Math.sin(ang) * 8.0;
            }
            if (zc.hp <= 0) {
              game.ecs.destroy(ze);
              game.events.emit('zombieKilled', { kind: zc.def.kind, x: zt.x, y: zt.y });
            }
          }
        }
      } else if (turret.kind === 'drone' && turret.timer <= 0) {
        // Repair damaged friendly structures within range
        for (const targetE of game.base.entities()) {
          const targetS = game.ecs.get(targetE, Comp.Structure)!;
          const targetT = game.ecs.get(targetE, Comp.Transform)!;
          if (Math.hypot(targetT.x - t.x, targetT.y - t.y) <= turret.range) {
            const maxHp = targetS.def.hp;
            // Structure damaged
            if (targetS.open === false && maxHp > 0) {
              // Repair pulse
              turret.timer = turret.cooldown;
              break;
            }
          }
        }
      }
    }
  }

  private updateSiegeMutants(dt: number, game: Game): void {
    for (const ze of game.ecs.query(Comp.Zombie, Comp.Transform)) {
      const z = game.ecs.get(ze, Comp.Zombie)!, zt = game.ecs.get(ze, Comp.Transform)!;

      // 1. Sapper suicide explosion on industrial targets or player
      if (z.def.kind === 'sapper') {
        let hitTarget = false;
        // Check player
        if (Math.hypot(game.playerT.x - zt.x, game.playerT.y - zt.y) <= 1.3) hitTarget = true;
        // Check machines
        if (!hitTarget) {
          for (const me of game.ecs.query(Comp.Machine, Comp.Transform)) {
            const mt = game.ecs.get(me, Comp.Transform)!;
            if (Math.hypot(mt.x - zt.x, mt.y - zt.y) <= 1.8) { hitTarget = true; break; }
          }
        }
        if (hitTarget) {
          // Detonate!
          game.events.emit('toast', { text: '💥 KABOOM! Sapper powder keg exploded!' });
          game.events.emit('hit', { x: zt.x, y: zt.y, kind: 'rock' });
          this.addPollution(zt.x, zt.y, 8);
          // Splash damage
          if (Math.hypot(game.playerT.x - zt.x, game.playerT.y - zt.y) < 3.8) {
            game.playerNeeds.health = Math.max(0, game.playerNeeds.health - z.def.damage);
          }
          game.ecs.destroy(ze);
        }
      } else if (z.def.kind === 'spitter') {
        // Acid spitter corrosive spit at range
        z.attackTimer -= dt;
        if (z.attackTimer <= 0 && Math.hypot(game.playerT.x - zt.x, game.playerT.y - zt.y) <= 8.5) {
          z.attackTimer = z.def.attackCooldown;
          if (this.spitterToastTimer <= 0) {
            this.spitterToastTimer = 6.0;
            game.events.emit('toast', { text: '🧪 Acid Spitter sprayed corrosive bile!' });
          }
          game.playerNeeds.health = Math.max(0, game.playerNeeds.health - z.def.damage);
        }
      } else if (z.def.kind === 'screamer') {
        // Screamer queen wail cascades
        z.attackTimer -= dt;
        if (z.attackTimer <= 0) {
          z.attackTimer = z.def.attackCooldown;
          if (this.screamerToastTimer <= 0) {
            this.screamerToastTimer = 9.0;
            game.events.emit('toast', { text: '📢 Screamer Queen rallied nearby undead!' });
          }
          game.emitNoise(zt.x, zt.y, 24);
        }
      }
    }
  }

  toSave(): SiegeSaveData {
    return {
      pollution: [...this.pollutionMap.entries()],
      siegeTimer: this.siegeTimer,
      totalSieges: this.totalSieges,
    };
  }

  load(data: Partial<SiegeSaveData>): void {
    if (data.pollution && Array.isArray(data.pollution)) {
      this.pollutionMap.clear();
      for (const [k, v] of data.pollution) this.pollutionMap.set(k, v);
    }
    if (data.siegeTimer !== undefined) this.siegeTimer = data.siegeTimer;
    if (data.totalSieges !== undefined) this.totalSieges = data.totalSieges;
  }
}
