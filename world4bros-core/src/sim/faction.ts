import type { Game } from './game';
import { Comp, type TotemC } from './components';
import type { ZoneRiskTier } from '../data/schema';
import type { Entity } from '../engine/ecs';

export interface BountyContract {
  id: string;
  title: string;
  desc: string;
  targetCount: number;
  currentCount: number;
  rewardCredits: number;
  rewardItem?: string;
  completed: boolean;
}

export interface FactionSaveData {
  clanName: string;
  credits: number;
  treatyDaysLeft: number;
  activeRaidOverride: boolean;
  bounties: BountyContract[];
}

export class FactionSystem {
  clanName = 'Wild Bros';
  credits = 100;
  treatyDaysLeft = 0;
  activeRaidOverride = false;
  bounties: BountyContract[] = [];
  activeC4Charges: { x: number; y: number; timer: number }[] = [];

  constructor() {
    this.refreshBounties();
  }

  getZoneAt(x: number, y: number, spawnX = 0, spawnY = 0): ZoneRiskTier {
    const dist = Math.hypot(x - spawnX, y - spawnY);
    if (dist < 65) return 'green';
    if (dist <= 200) return 'yellow';
    return 'red';
  }

  update(dt: number, game: Game): void {
    const sp = game.world.spawn;
    const currentHour = (game.clock.t * 24) % 24;

    // 1. Update Clan Totem Tesla Shields
    for (const e of game.ecs.query(Comp.Structure, Comp.Transform)) {
      const s = game.ecs.get(e, Comp.Structure)!, t = game.ecs.get(e, Comp.Transform)!;

      // Auto-attach Totem component to clan_territory_totem
      if (s.def.id === 'clan_territory_totem' && !game.ecs.has(e, Comp.Totem)) {
        game.ecs.add(e, Comp.Totem, {
          clanName: this.clanName,
          radius: 30,
          shieldActive: true,
          raidWindowStart: 18.0,
          raidWindowEnd: 20.0,
          bountyCredits: this.credits,
        });
      }

      if (game.ecs.has(e, Comp.Totem)) {
        const totem = game.ecs.get(e, Comp.Totem)!;
        const isRaidTime = currentHour >= totem.raidWindowStart && currentHour <= totem.raidWindowEnd;
        const hasTreaty = this.treatyDaysLeft > 0;

        // Active defense window lowers shield
        if (this.activeRaidOverride || (isRaidTime && !hasTreaty)) {
          if (totem.shieldActive) {
            totem.shieldActive = false;
            game.events.emit('toast', { text: `⚠️ RAID WINDOW ACTIVE: ${totem.clanName} Tesla Shields are DOWN!` });
          }
        } else {
          if (!totem.shieldActive) {
            totem.shieldActive = true;
            game.events.emit('toast', { text: `🛡️ TESLA SHIELDS ONLINE: ${totem.clanName} base is protected.` });
          }
        }
      }
    }

    // 2. Update Active C4 charges
    for (let i = this.activeC4Charges.length - 1; i >= 0; i--) {
      const c4 = this.activeC4Charges[i];
      c4.timer -= dt;
      if (c4.timer <= 0) {
        this.detonateC4(c4.x, c4.y, game);
        this.activeC4Charges.splice(i, 1);
      }
    }
  }

  armC4(x: number, y: number, game: Game): boolean {
    if (!game.playerInv.inv.remove('c4_plunger_charge', 1)) {
      game.events.emit('toast', { text: 'Need a C4 Plunger Breaching Charge' });
      return false;
    }
    this.activeC4Charges.push({ x, y, timer: 4.0 });
    game.events.emit('toast', { text: '🧨 C4 PLACED! Detonation in 4 seconds! TAKE COVER!' });
    return true;
  }

  private detonateC4(x: number, y: number, game: Game): void {
    game.events.emit('toast', { text: '💥 BOOM! C4 Breached Defenses with a shockwave!' });
    game.events.emit('hit', { x, y, kind: 'rock' });
    game.emitNoise(x, y, 35);

    // Damage structures within 3.8m
    for (const se of game.base.entities()) {
      const st = game.ecs.get(se, Comp.Transform);
      const sc = game.ecs.get(se, Comp.Structure);
      if (st && sc && Math.hypot(st.x - x, st.y - y) <= 3.8) {
        sc.hp = Math.max(0, sc.hp - 200);
        if (sc.hp <= 0) {
          game.base.remove(se);
        }
      }
    }

    // Damage player if caught in blast
    if (Math.hypot(game.playerT.x - x, game.playerT.y - y) <= 4.0) {
      game.playerNeeds.health = Math.max(0, game.playerNeeds.health - 60);
      game.events.emit('toast', { text: 'Caught in C4 blast! (-60 HP)' });
    }

    // Advance breach bounty
    this.advanceBounty('breach_c4', 1, game);
  }

  advanceBounty(kind: string, amount: number, game: Game): void {
    for (const b of this.bounties) {
      if (b.id === kind && !b.completed) {
        b.currentCount += amount;
        if (b.currentCount >= b.targetCount) {
          b.completed = true;
          this.credits += b.rewardCredits;
          if (b.rewardItem) game.playerInv.inv.add(b.rewardItem, 1);
          game.events.emit('toast', { text: `💰 BOUNTY COMPLETED: ${b.title} (+${b.rewardCredits} Credits)!` });
        }
      }
    }
  }

  refreshBounties(): void {
    this.bounties = [
      {
        id: 'slay_zombies',
        title: 'Clear Red Zone Hordes',
        desc: 'Eliminate 3 hostile mutants in the contested perimeter.',
        targetCount: 3,
        currentCount: 0,
        rewardCredits: 75,
        rewardItem: 'turret_ammo',
        completed: false,
      },
      {
        id: 'craft_plates',
        title: 'Factory Munitions Supply',
        desc: 'Smelt iron plates to sustain base defense operations.',
        targetCount: 5,
        currentCount: 0,
        rewardCredits: 60,
        rewardItem: 'repair_drone_kit',
        completed: false,
      },
      {
        id: 'breach_c4',
        title: 'Plunger Demolition Test',
        desc: 'Detonate a C4 Breaching Charge on an outpost perimeter.',
        targetCount: 1,
        currentCount: 0,
        rewardCredits: 120,
        rewardItem: 'energetic_crystal',
        completed: false,
      },
    ];
  }

  signPeaceTreaty(days = 2): void {
    this.treatyDaysLeft += days;
  }

  toSave(): FactionSaveData {
    return {
      clanName: this.clanName,
      credits: this.credits,
      treatyDaysLeft: this.treatyDaysLeft,
      activeRaidOverride: this.activeRaidOverride,
      bounties: this.bounties,
    };
  }

  load(data: Partial<FactionSaveData>): void {
    if (data.clanName) this.clanName = data.clanName;
    if (data.credits !== undefined) this.credits = data.credits;
    if (data.treatyDaysLeft !== undefined) this.treatyDaysLeft = data.treatyDaysLeft;
    if (data.activeRaidOverride !== undefined) this.activeRaidOverride = data.activeRaidOverride;
    if (data.bounties && Array.isArray(data.bounties)) this.bounties = data.bounties;
  }
}
