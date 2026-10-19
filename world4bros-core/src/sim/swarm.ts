import { makeRng, type Rng } from '../engine/rng';
import { clamp } from '../engine/math';

export interface SwarmBrainCfg {
  reaction: number;
  aimErr: number;
  kite: number;
  lead: number;
  cover: number;
  aggr: number;
}

export interface SwarmSimCfg {
  mode: 'pve' | 'pvpve';
  pattern: 'waves' | 'sleepers' | 'ring' | 'horde';
  zombies: number;
  raiders: number;
  bros: number;
  waves: number;
  waveEvery: number;
  timeLimit: number;
  broWeapon: string;
  broAtt?: Record<string, string>;
  raiderWeapon: string;
  broBrain: SwarmBrainCfg;
  raiderBrain: SwarmBrainCfg;
}

export interface SwarmEvent {
  t: number;
  type: 'shot' | 'hit' | 'kill';
  actorTeam: string;
  actorKind: string;
  weapon: string;
  victimTeam: string;
  victimKind: string;
  distM: number;
  dmg: number;
  hpAfter: number;
}

export interface WeaponDistanceBand {
  aim: number;
  hits: number;
  dmg: number;
  ah: number;
}

export interface WeaponStats {
  key: string;
  team: string;
  name: string;
  shots: number;
  pellets: number;
  hits: number;
  pHit: number;
  dmg: number;
  kills: number;
  over: number;
  band: WeaponDistanceBand[];
  ttk: number[];
  killDist: number;
  noise: number;
  alerts: number;
}

export interface SwarmSimSummary {
  outcome: 'cleared' | 'wiped' | 'time' | 'running';
  time: number;
  seed: number;
  kills: number;
  weaponKills: number;
  killsPerMin: number;
  shots: number;
  accuracy: number;
  dmgDealt: number;
  dmgPerShot: number;
  ammoPerKill: number | null;
  avgKillDist: number | null;
  avgTTK: number | null;
  overkillPct: number;
  noisePerKill: number | null;
  alertsPerKill: number | null;
  broDeaths: number;
  broDmgTaken: number;
  broAlive: number;
  zombieKills: number;
  raiderKills: number;
  raidersLost: number;
}

export const DIST_BANDS = [3, 6, 10, 15, 25, 1e9];
export const DIST_BAND_LABELS = ['0–3m', '3–6m', '6–10m', '10–15m', '15–25m', '25m+'];

export function getBandIndex(d: number): number {
  for (let i = 0; i < DIST_BANDS.length; i++) {
    if (d < DIST_BANDS[i]) return i;
  }
  return DIST_BANDS.length - 1;
}

export interface SimWeaponDef {
  key: string;
  name: string;
  dmg: number;
  min: number;
  fs: number;
  range: number;
  spread: number;
  rate: number;
  noise: number;
  pen: number;
  speed: number;
  pellets: number;
  melee?: boolean;
  reach?: number;
  attNames?: string[];
}

export const SIM_WEAPONS: Record<string, SimWeaponDef> = {
  hunting_rifle: { key: 'hunting_rifle', name: 'Hunting Rifle', dmg: 55, min: 28, fs: 22, range: 48, spread: 0.015, rate: 0.75, noise: 28, pen: 2, speed: 45, pellets: 1 },
  pump_shotgun: { key: 'pump_shotgun', name: 'Pump Shotgun', dmg: 16, min: 5, fs: 6, range: 16, spread: 0.24, rate: 0.9, noise: 26, pen: 1, speed: 22, pellets: 6 },
  nailgun_smg: { key: 'nailgun_smg', name: 'Nailgun SMG', dmg: 14, min: 7, fs: 10, range: 20, spread: 0.11, rate: 5, noise: 14, pen: 1, speed: 24, pellets: 1 },
  plinker_pistol: { key: 'plinker_pistol', name: 'Plinker Pistol', dmg: 14, min: 7, fs: 8, range: 18, spread: 0.07, rate: 3, noise: 24, pen: 1, speed: 30, pellets: 1 },
  bark_revolver: { key: 'bark_revolver', name: 'Bark Revolver', dmg: 30, min: 15, fs: 10, range: 22, spread: 0.05, rate: 1.4, noise: 30, pen: 1, speed: 34, pellets: 1 },
  slingshot: { key: 'slingshot', name: 'Slingshot', dmg: 8, min: 4, fs: 6, range: 16, spread: 0.08, rate: 1.6, noise: 3, pen: 1, speed: 16, pellets: 1 },
  potato_cannon: { key: 'potato_cannon', name: 'Potato Cannon', dmg: 32, min: 16, fs: 14, range: 22, spread: 0.05, rate: 0.85, noise: 16, pen: 1, speed: 13, pellets: 1 },
  foam_dart_blaster: { key: 'foam_dart_blaster', name: 'Foam Dart Blaster', dmg: 3, min: 2, fs: 5, range: 12, spread: 0.09, rate: 4, noise: 3, pen: 1, speed: 18, pellets: 1 },
  cork_crossbow: { key: 'cork_crossbow', name: 'Cork Crossbow', dmg: 20, min: 12, fs: 12, range: 26, spread: 0.02, rate: 0.7, noise: 3, pen: 1, speed: 28, pellets: 1 },
  chicken_cannon: { key: 'chicken_cannon', name: 'Chicken Cannon', dmg: 12, min: 8, fs: 10, range: 20, spread: 0.06, rate: 0.7, noise: 12, pen: 1, speed: 12, pellets: 1 },
  nail_bat: { key: 'nail_bat', name: 'Nail Bat', melee: true, dmg: 14, reach: 1.6, rate: 1.2, noise: 6, min: 14, fs: 1, range: 1.6, spread: 0, pen: 1, speed: 1, pellets: 1 },
};

export const SIM_ZOMBIES: Record<string, { hp: number; dmg: number; spd: number; run: number; det: number; cd: number; r: number; reach: number; w: number; boom?: boolean; ranged?: boolean; scream?: boolean }> = {
  walker: { hp: 28, dmg: 10, spd: 1.15, run: 2.1, det: 8.5, cd: 1.2, r: 0.35, reach: 0.85, w: 55 },
  runner: { hp: 18, dmg: 8, spd: 1.8, run: 3.7, det: 13, cd: 0.85, r: 0.32, reach: 0.9, w: 20 },
  brute: { hp: 65, dmg: 24, spd: 0.85, run: 1.85, det: 10, cd: 1.8, r: 0.5, reach: 1.0, w: 8 },
  spitter: { hp: 32, dmg: 14, spd: 1.1, run: 2.2, det: 14, cd: 2, r: 0.35, reach: 5.5, ranged: true, w: 6 },
  screamer: { hp: 30, dmg: 6, spd: 1.4, run: 2.8, det: 22, cd: 3, r: 0.35, reach: 0.9, scream: true, w: 4 },
  sapper: { hp: 24, dmg: 45, spd: 1.5, run: 3.2, det: 16, cd: 0.5, r: 0.33, reach: 1.2, boom: true, w: 5 },
  goliath: { hp: 140, dmg: 38, spd: 0.75, run: 1.6, det: 12, cd: 2.2, r: 0.8, reach: 1.6, w: 2 },
};

export const SIM_RAIDERS = {
  raider: { hp: 60, spd: 3.0, r: 0.4, sight: 18, wp: 'nailgun_smg' },
  heavy: { hp: 100, spd: 2.4, r: 0.46, sight: 16, wp: 'pump_shotgun' },
  marksman: { hp: 50, spd: 2.8, r: 0.4, sight: 26, wp: 'hunting_rifle' },
};

export interface SimShooter {
  id: number;
  team: 'bro' | 'raider';
  kind: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hp: number;
  maxHp: number;
  spd: number;
  sight: number;
  dead: boolean;
  stunTimer: number;
  firstDamageTime: number;
  w: SimWeaponDef;
  cooldown: number;
  brain: SwarmBrainCfg;
  reactionTimer: number;
  target: SimEntity | null;
  strafe: number;
  strafeTimer: number;
  heardTimer: number;
  heardPos: { x: number; y: number } | null;
  thinkTimer: number;
  angle: number;
  moving: boolean;
  coverPos: { x: number; y: number } | null;
  coverTimer: number;
}

export interface SimZombie {
  id: number;
  team: 'zombie';
  kind: string;
  def: (typeof SIM_ZOMBIES)[string];
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hp: number;
  maxHp: number;
  dead: boolean;
  stunTimer: number;
  firstDamageTime: number;
  state: 'sleep' | 'wander' | 'alert' | 'chase';
  target: SimShooter | null;
  alertPos: { x: number; y: number };
  alertTimer: number;
  wanderTimer: number;
  wanderAngle: number;
  attackTimer: number;
  acqTimer: number;
  speedMul: number;
  screamCooldown: number;
  drift: number;
  angle: number;
}

export type SimEntity = SimShooter | SimZombie;

export interface SimBullet {
  x: number;
  y: number;
  startX: number;
  startY: number;
  vx: number;
  vy: number;
  speed: number;
  w: SimWeaponDef;
  owner: SimShooter;
  team: string;
  distTraveled: number;
  penetration: number;
  hitSet: Set<number>;
  aimBand: number;
  counted: boolean;
}

export const ARENA_W = 64;
export const ARENA_H = 40;

export const ARENA_RECTS: [number, number, number, number][] = [
  [8, 8, 2, 2], [54, 30, 2, 2], [8, 30, 2, 2], [54, 8, 2, 2],
  [14, 8, 6, 0.5], [44, 30, 6, 0.5], [44, 8, 6, 0.5], [14, 30, 6, 0.5],
  [22, 18, 0.5, 4], [40, 18, 0.5, 4], [22, 28, 4, 0.5], [38, 11, 4, 0.5],
  [26, 10, 1, 3], [37, 27, 1, 3], [12, 18, 3, 1], [49, 21, 3, 1]
];

export const ARENA_CIRCLES: [number, number, number][] = [
  [20, 14, 0.42], [44, 25, 0.42], [20, 25, 0.42], [44, 14, 0.42],
  [32, 12, 0.42], [32, 28, 0.42], [8, 20, 0.42], [56, 20, 0.42]
];

function circleRectOverlap(x: number, y: number, r: number, b: [number, number, number, number]): boolean {
  const nx = clamp(x, b[0], b[0] + b[2]), ny = clamp(y, b[1], b[1] + b[3]);
  const dx = x - nx, dy = y - ny;
  return dx * dx + dy * dy < r * r;
}

function segRectIntersect(x1: number, y1: number, x2: number, y2: number, r: [number, number, number, number]): number {
  let t0 = 0, t1 = 1;
  const dx = x2 - x1, dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1 - r[0], r[0] + r[2] - x1, y1 - r[1], r[1] + r[3] - y1];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return -1;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) {
        if (t > t1) return -1;
        if (t > t0) t0 = t;
      } else {
        if (t < t0) return -1;
        if (t < t1) t1 = t;
      }
    }
  }
  return t0;
}

export function isBlocked(x: number, y: number, r: number): boolean {
  if (x < r + 1 || y < r + 1 || x > ARENA_W - 1 - r || y > ARENA_H - 1 - r) return true;
  for (const b of ARENA_RECTS) {
    if (circleRectOverlap(x, y, r, b)) return true;
  }
  for (const c of ARENA_CIRCLES) {
    const dx = x - c[0], dy = y - c[1], rr = r + c[2];
    if (dx * dx + dy * dy < rr * rr) return true;
  }
  return false;
}

export function lineOfSight(x1: number, y1: number, x2: number, y2: number): boolean {
  for (const b of ARENA_RECTS) {
    if (segRectIntersect(x1, y1, x2, y2, b) >= 0) return false;
  }
  return true;
}

function distToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1, dy = y2 - y1, l = dx * dx + dy * dy;
  let t = l ? ((px - x1) * dx + (py - y1) * dy) / l : 0;
  t = clamp(t, 0, 1);
  const cx = x1 + t * dx, cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

export class SwarmSim {
  readonly rng: Rng;
  time = 0;
  over: 'cleared' | 'wiped' | 'time' | null = null;
  bros: SimShooter[] = [];
  raiders: SimShooter[] = [];
  zombies: SimZombie[] = [];
  bullets: SimBullet[] = [];
  events: SwarmEvent[] = [];
  stats: {
    w: Record<string, WeaponStats>;
    kills: Record<string, Record<string, number>>;
    deaths: Record<string, number>;
    dmgTaken: Record<string, number>;
    alertsTotal: number;
    series: [number, number, number, number][];
  };

  private nextId = 1;
  private zSpawned = 0;
  private waveTimer = 1;
  private sampleTimer = 0;

  constructor(public readonly cfg: SwarmSimCfg, public readonly seed: number) {
    this.rng = makeRng(seed);
    this.stats = {
      w: {},
      kills: { bro: {}, raider: {}, zombie: {} },
      deaths: { bro: 0, raider: 0, zombie: 0 },
      dmgTaken: { bro: 0, raider: 0, zombie: 0 },
      alertsTotal: 0,
      series: []
    };
    this.setup();
  }

  private getWeaponStats(team: string, w: SimWeaponDef): WeaponStats {
    const k = `${team}:${w.name}`;
    if (!this.stats.w[k]) {
      this.stats.w[k] = {
        key: k,
        team,
        name: w.name,
        shots: 0,
        pellets: 0,
        hits: 0,
        pHit: 0,
        dmg: 0,
        kills: 0,
        over: 0,
        band: DIST_BANDS.map(() => ({ aim: 0, hits: 0, dmg: 0, ah: 0 })),
        ttk: [],
        killDist: 0,
        noise: 0,
        alerts: 0,
      };
    }
    return this.stats.w[k];
  }

  private setup(): void {
    const c = this.cfg;
    const broW = SIM_WEAPONS[c.broWeapon] ?? SIM_WEAPONS.nailgun_smg;

    // 1. Bros
    for (let i = 0; i < c.bros; i++) {
      const y = 20 + (i - (c.bros - 1) / 2) * 1.6;
      this.bros.push({
        id: this.nextId++,
        team: 'bro',
        kind: 'bro',
        x: 6,
        y,
        vx: 0,
        vy: 0,
        r: 0.4,
        hp: 100,
        maxHp: 100,
        spd: 3.4,
        sight: 20,
        dead: false,
        stunTimer: 0,
        firstDamageTime: -1,
        w: broW,
        cooldown: this.rng.range(0, 0.5),
        brain: c.broBrain,
        reactionTimer: 0,
        target: null,
        strafe: this.rng.chance(0.5) ? 1 : -1,
        strafeTimer: this.rng.range(0.5, 2),
        heardTimer: -9,
        heardPos: null,
        thinkTimer: this.rng.range(0, 0.15),
        angle: 0,
        moving: false,
        coverPos: null,
        coverTimer: 0,
      });
    }

    // 2. Raiders (PvPvE)
    if (c.mode === 'pvpve') {
      const rKinds: ('raider' | 'heavy' | 'marksman')[] = ['raider', 'raider', 'heavy', 'marksman'];
      for (let i = 0; i < c.raiders; i++) {
        const kind = rKinds[i % rKinds.length];
        const rDef = SIM_RAIDERS[kind];
        const w = SIM_WEAPONS[rDef.wp] ?? SIM_WEAPONS.nailgun_smg;
        const y = 20 + (i - (c.raiders - 1) / 2) * 1.4;
        this.raiders.push({
          id: this.nextId++,
          team: 'raider',
          kind,
          x: 58,
          y,
          vx: 0,
          vy: 0,
          r: rDef.r,
          hp: rDef.hp,
          maxHp: rDef.hp,
          spd: rDef.spd,
          sight: rDef.sight,
          dead: false,
          stunTimer: 0,
          firstDamageTime: -1,
          w,
          cooldown: this.rng.range(0, 0.5),
          brain: c.raiderBrain,
          reactionTimer: 0,
          target: null,
          strafe: this.rng.chance(0.5) ? 1 : -1,
          strafeTimer: this.rng.range(0.5, 2),
          heardTimer: -9,
          heardPos: null,
          thinkTimer: this.rng.range(0, 0.15),
          angle: Math.PI,
          moving: false,
          coverPos: null,
          coverTimer: 0,
        });
      }
    }

    // 3. Pre-spawn zombies if not wave-based
    if (c.pattern === 'sleepers') {
      for (let i = 0; i < c.zombies; i++) {
        let x = 0, y = 0, tries = 0;
        do {
          x = this.rng.range(4, ARENA_W - 4);
          y = this.rng.range(4, ARENA_H - 4);
          tries++;
        } while ((isBlocked(x, y, 0.4) || (x < 18 && Math.abs(y - 20) < 8) || (c.mode === 'pvpve' && x > 46 && Math.abs(y - 20) < 8)) && tries < 40);
        this.spawnZombie(x, y, 'sleep');
      }
    } else if (c.pattern === 'horde') {
      for (let i = 0; i < c.zombies; i++) {
        const x = 32 + this.rng.range(-7, 7);
        const y = 3 + this.rng.range(0, 5);
        const z = this.spawnZombie(x, y, 'alert');
        z.alertPos = { x: 8, y: 20 };
        z.alertTimer = 60;
      }
    }
  }

  private pickZombieKind(): string {
    const kinds = Object.keys(SIM_ZOMBIES);
    let tot = 0;
    for (const k of kinds) tot += SIM_ZOMBIES[k].w;
    let roll = this.rng.range(0, tot);
    for (const k of kinds) {
      roll -= SIM_ZOMBIES[k].w;
      if (roll <= 0) return k;
    }
    return 'walker';
  }

  private edgePoint(): [number, number] {
    const side = this.rng.int(0, 3);
    if (side === 0) return [this.rng.range(3, ARENA_W - 3), 2.2];
    if (side === 1) return [this.rng.range(3, ARENA_W - 3), ARENA_H - 2.2];
    if (side === 2) return [2.2, this.rng.range(3, ARENA_H - 3)];
    return [ARENA_W - 2.2, this.rng.range(3, ARENA_H - 3)];
  }

  private spawnZombie(x: number, y: number, state: 'sleep' | 'wander' | 'alert'): SimZombie {
    const kind = this.pickZombieKind();
    const d = SIM_ZOMBIES[kind] ?? SIM_ZOMBIES.walker;
    const z: SimZombie = {
      id: this.nextId++,
      team: 'zombie',
      kind,
      def: d,
      x,
      y,
      vx: 0,
      vy: 0,
      r: d.r,
      hp: d.hp,
      maxHp: d.hp,
      dead: false,
      stunTimer: 0,
      firstDamageTime: -1,
      state,
      target: null,
      alertPos: { x, y },
      alertTimer: 0,
      wanderTimer: this.rng.range(0, 3),
      wanderAngle: this.rng.range(0, Math.PI * 2),
      attackTimer: this.rng.range(0, d.cd),
      acqTimer: this.rng.range(0, 0.3),
      speedMul: this.rng.range(0.9, 1.1),
      screamCooldown: 0,
      drift: this.rng.chance(0.5) ? 1 : -1,
      angle: 0,
    };
    this.zombies.push(z);
    this.zSpawned++;
    return z;
  }

  private recordEvent(type: 'shot' | 'hit' | 'kill', actor: SimEntity | null, victim: SimEntity | null, w: SimWeaponDef | null, dist: number, dmg: number, hpAfter: number): void {
    if (this.events.length < 150000) {
      this.events.push({
        t: Math.round(this.time * 100) / 100,
        type,
        actorTeam: actor?.team ?? '',
        actorKind: actor ? (actor as SimShooter).kind : '',
        weapon: w ? w.name : '',
        victimTeam: victim?.team ?? '',
        victimKind: victim ? (victim as SimShooter).kind : '',
        distM: Math.round(dist * 10) / 10,
        dmg: Math.round(dmg * 10) / 10,
        hpAfter: Math.max(0, Math.round(hpAfter)),
      });
    }
  }

  emitNoise(x: number, y: number, rad: number, team: string, w: SimWeaponDef | null): void {
    let alerts = 0;
    for (const z of this.zombies) {
      if (z.dead) continue;
      const d = Math.hypot(z.x - x, z.y - y);
      if (d > rad) continue;
      if (z.state === 'sleep') {
        if (d < rad * 0.8) {
          z.state = 'alert';
          z.alertPos = { x, y };
          z.alertTimer = 8;
          alerts++;
        }
      } else if (z.state === 'wander' || z.state === 'alert') {
        if (z.state === 'wander') alerts++;
        z.state = 'alert';
        z.alertPos = { x, y };
        z.alertTimer = 6;
      }
    }

    if (w) {
      const s = this.getWeaponStats(team, w);
      s.alerts += alerts;
    }
    this.stats.alertsTotal += alerts;

    const foes = team === 'bro' ? this.raiders : this.bros;
    for (const f of foes) {
      if (!f.dead && Math.hypot(f.x - x, f.y - y) < rad) {
        f.heardTimer = this.time;
        f.heardPos = { x, y };
      }
    }
  }

  hurt(v: SimEntity, dmg: number, att: SimShooter | null, w: SimWeaponDef | null, dist: number): void {
    if (v.dead) return;
    if (v.firstDamageTime < 0) v.firstDamageTime = this.time;
    v.hp -= dmg;
    this.stats.dmgTaken[v.team] += dmg;

    if (att && w && att.team !== v.team) {
      const s = this.getWeaponStats(att.team, w);
      s.hits++;
      s.dmg += Math.max(0, Math.min(dmg, v.hp + dmg));
      const b = s.band[getBandIndex(dist || 0)];
      b.hits++;
      b.dmg += dmg;
    }

    this.recordEvent('hit', att, v, w, dist, dmg, Math.max(0, v.hp));

    if (v.team === 'zombie' && att && !att.dead && (v as SimZombie).state !== 'chase') {
      (v as SimZombie).state = 'chase';
      (v as SimZombie).target = att;
    }

    if (v.hp <= 0) {
      this.kill(v, att, w, dist, -v.hp);
    }
  }

  kill(v: SimEntity, att: SimShooter | null, w: SimWeaponDef | null, dist: number, over: number): void {
    v.dead = true;
    this.stats.deaths[v.team]++;
    const k = (v as SimShooter).kind;
    if (att) {
      const kt = this.stats.kills[att.team];
      kt[k] = (kt[k] || 0) + 1;
    }
    if (att && w && att.team !== v.team) {
      const s = this.getWeaponStats(att.team, w);
      s.kills++;
      s.over += Math.max(0, over);
      s.killDist += dist || 0;
      if (v.firstDamageTime >= 0) s.ttk.push(this.time - v.firstDamageTime);
    }
    this.recordEvent('kill', att, v, w, dist, over, 0);
  }

  fire(e: SimShooter, ang: number, aimDist: number): void {
    const w = e.w;
    const s = this.getWeaponStats(e.team, w);
    s.shots++;

    if (w.melee) {
      this.melee(e, w);
      return;
    }

    const n = w.pellets || 1;
    s.pellets += n;
    s.band[getBandIndex(aimDist || 0)].aim += n;

    for (let i = 0; i < n; i++) {
      const spread = (this.rng.next() - 0.5) * w.spread * (e.moving ? 1.7 : 0.85);
      const a = ang + spread;
      this.bullets.push({
        x: e.x + Math.cos(a) * 0.45,
        y: e.y + Math.sin(a) * 0.45,
        startX: e.x,
        startY: e.y,
        vx: Math.cos(a),
        vy: Math.sin(a),
        speed: w.speed,
        w,
        owner: e,
        team: e.team,
        distTraveled: 0,
        penetration: w.pen,
        hitSet: new Set(),
        aimBand: getBandIndex(aimDist || 0),
        counted: false,
      });
    }

    this.emitNoise(e.x, e.y, w.noise, e.team, w);
    s.noise += w.noise;
    e.angle = ang;
    this.recordEvent('shot', e, null, w, aimDist, 0, 0);
  }

  private melee(e: SimShooter, w: SimWeaponDef): void {
    this.emitNoise(e.x, e.y, w.noise, e.team, w);
    let hitCount = 0;
    const foes: SimEntity[] = e.team === 'bro' ? [...this.raiders, ...this.zombies] : [...this.bros, ...this.zombies];
    for (const f of foes) {
      if (f.dead) continue;
      const d = Math.hypot(f.x - e.x, f.y - e.y);
      if (d < (w.reach || 1.5) + f.r) {
        hitCount++;
        this.hurt(f, w.dmg, e, w, d);
        if (hitCount >= 3) break;
      }
    }
  }

  private moveEntity(e: SimEntity, dx: number, dy: number): void {
    const nx = e.x + dx;
    if (!isBlocked(nx, e.y, e.r)) e.x = nx;
    else if (e.team === 'zombie') e.vx *= 0.3;

    const ny = e.y + dy;
    if (!isBlocked(e.x, ny, e.r)) e.y = ny;
    else if (e.team === 'zombie') e.vy *= 0.3;
  }

  private updateBullets(dt: number): void {
    const active: SimBullet[] = [];
    for (const b of this.bullets) {
      let rem = b.speed * dt, alive = true;
      while (rem > 0 && alive) {
        const step = Math.min(0.5, rem);
        const nx = b.x + b.vx * step, ny = b.y + b.vy * step;

        for (const r of ARENA_RECTS) {
          if (segRectIntersect(b.x, b.y, nx, ny, r) >= 0) {
            alive = false;
            break;
          }
        }
        if (!alive) break;

        // Check collision against opponents
        const foes: SimEntity[] = b.team === 'bro' ? [...this.raiders, ...this.zombies] : [...this.bros, ...this.zombies];
        for (const c of foes) {
          if (!alive || c.dead || c.team === b.team) continue;
          if (b.hitSet.has(c.id)) continue;
          if (distToSegment(c.x, c.y, b.x, b.y, nx, ny) < c.r + 0.08) {
            const dist = b.distTraveled + step;
            const w = b.w;
            const falloff = dist <= w.fs ? 1 : clamp(1 - (dist - w.fs) / Math.max(1, w.range - w.fs), 0, 1);
            const dmg = Math.max(w.min, Math.round(w.dmg - (w.dmg - w.min) * (1 - falloff)));

            b.hitSet.add(c.id);
            if (!b.counted) {
              b.counted = true;
              const ws = this.getWeaponStats(b.team, w);
              ws.pHit++;
              ws.band[b.aimBand].ah++;
            }
            this.hurt(c, dmg, b.owner, w, dist);
            b.penetration--;
            if (b.penetration <= 0) alive = false;
          }
        }

        b.x = nx;
        b.y = ny;
        b.distTraveled += step;
        rem -= step;
        if (b.distTraveled >= b.w.range) alive = false;
      }
      if (alive) active.push(b);
    }
    this.bullets = active;
  }

  private thinkBot(e: SimShooter): void {
    const foesTeam = e.team === 'bro' ? 'raider' : 'bro';
    const br = e.brain;
    let best: SimEntity | null = null, bestScore = -1e9;
    let nearZ: SimZombie | null = null, nzd = 1e9, zc = 0, zx = 0, zy = 0;

    const candidates: SimEntity[] = e.team === 'bro' ? [...this.raiders, ...this.zombies] : [...this.bros, ...this.zombies];
    for (const c of candidates) {
      if (c.dead || c.team === e.team) continue;
      const d = Math.hypot(c.x - e.x, c.y - e.y);
      if (d > e.sight) continue;
      if (!lineOfSight(e.x, e.y, c.x, c.y)) continue;

      if (c.team === 'zombie') {
        if (d < nzd) { nzd = d; nearZ = c as SimZombie; }
        if (d < br.kite + 2) { zc++; zx += c.x; zy += c.y; }
      }

      let sc = -d;
      if (c.team === 'zombie' && d < 4.5) sc += 8;
      else if (c.team === foesTeam) sc += 3;
      sc += 2 * (1 - c.hp / c.maxHp);
      if (e.target === c) sc += 2;
      if (sc > bestScore) {
        bestScore = sc;
        best = c;
      }
    }

    if (best !== e.target) {
      e.target = best;
      if (best) e.reactionTimer = br.reaction;
    }
  }

  private updateBot(e: SimShooter, dt: number): void {
    if (e.stunTimer > 0) {
      e.stunTimer -= dt;
      return;
    }
    e.cooldown -= dt;
    e.reactionTimer -= dt;
    e.strafeTimer -= dt;
    e.thinkTimer -= dt;

    if (e.thinkTimer <= 0) {
      e.thinkTimer = 0.15;
      this.thinkBot(e);
    }

    const t = e.target && !e.target.dead ? e.target : null;
    const w = e.w, br = e.brain, melee = !!w.melee;
    let dx = 0, dy = 0, sp = e.spd;

    const ideal = melee ? (w.reach || 1.5) * 0.8 : clamp(w.range * (w.pellets > 1 ? 0.3 : 0.55), 3, 24);
    let mode: 'idle' | 'kite' | 'engage' | 'investigate' | 'advance' | 'regroup' | 'cover' = 'idle';

    if (!melee && t && t.team === 'zombie' && Math.hypot(t.x - e.x, t.y - e.y) < br.kite) {
      mode = 'kite';
    } else if (t) {
      mode = 'engage';
    } else if (e.heardPos && this.time - e.heardTimer < 8 && e.team === 'raider') {
      mode = 'investigate';
    } else if (e.team === 'raider') {
      mode = 'advance';
    } else {
      mode = 'regroup';
    }

    if (br.cover > 0 && e.hp < e.maxHp * 0.35 && t && mode !== 'kite') {
      mode = 'cover';
    }

    if (mode === 'kite' && t) {
      const ax = e.x - t.x, ay = e.y - t.y;
      const m = Math.hypot(ax, ay) || 1;
      dx = ax / m; dy = ay / m;
    } else if (mode === 'engage' && t) {
      const ddx = t.x - e.x, ddy = t.y - e.y, d = Math.hypot(ddx, ddy) || 0.001;
      const ux = ddx / d, uy = ddy / d;
      if (d > ideal * 1.25) { dx = ux; dy = uy; }
      else if (d < ideal * 0.6 && !melee) { dx = -ux; dy = -uy; }
      else {
        if (e.strafeTimer <= 0) {
          e.strafeTimer = this.rng.range(1, 2.6);
          e.strafe = this.rng.chance(0.5) ? 1 : -1;
        }
        dx = -uy * e.strafe * 0.8;
        dy = ux * e.strafe * 0.8;
      }
    } else if (mode === 'investigate' && e.heardPos) {
      const ddx = e.heardPos.x - e.x, ddy = e.heardPos.y - e.y, d = Math.hypot(ddx, ddy);
      if (d > 3) { dx = ddx / d; dy = ddy / d; sp = e.spd * 0.85; }
    } else if (mode === 'advance') {
      let cx = 0, cy = 0, n = 0;
      for (const b of this.bros) {
        if (!b.dead) { cx += b.x; cy += b.y; n++; }
      }
      if (n > 0) {
        cx /= n; cy /= n;
        const ddx = cx - e.x, ddy = cy - e.y, d = Math.hypot(ddx, ddy);
        if (d > 8) { dx = ddx / d; dy = ddy / d; sp = e.spd * 0.7 * br.aggr; }
      }
    } else if (mode === 'regroup') {
      const lead = this.bros.find((b) => !b.dead && b !== e);
      if (lead) {
        const ddx = lead.x - e.x, ddy = lead.y - e.y, d = Math.hypot(ddx, ddy);
        if (d > 5) { dx = ddx / d; dy = ddy / d; sp = e.spd * 0.8; }
      }
    }

    e.moving = (dx !== 0 || dy !== 0) && sp > 0;
    e.vx += (dx * sp - e.vx) * Math.min(1, dt * 9);
    e.vy += (dy * sp - e.vy) * Math.min(1, dt * 9);
    this.moveEntity(e, e.vx * dt, e.vy * dt);

    // Aiming and Shooting
    if (t) {
      const ddx = t.x - e.x, ddy = t.y - e.y, d = Math.hypot(ddx, ddy) || 0.001;
      const lead = melee ? 0 : br.lead * (d / (w.speed || 30));
      const ax = t.x + t.vx * lead, ay = t.y + t.vy * lead;
      const gaussian = (this.rng.next() + this.rng.next() - 1) * 0.5;
      const ang = Math.atan2(ay - e.y, ax - e.x) + gaussian * br.aimErr;
      e.angle = Math.atan2(ddy, ddx);

      const inRange = melee ? d < (w.reach || 1.5) + t.r + 0.3 : d < w.range * 0.95;
      if (inRange && lineOfSight(e.x, e.y, t.x, t.y) && e.reactionTimer <= 0 && e.cooldown <= 0) {
        e.cooldown = 1 / w.rate;
        this.fire(e, melee ? e.angle : ang, d);
      }
    }
  }

  private updateZombies(dt: number): void {
    for (const z of this.zombies) {
      if (z.dead) continue;
      const d = z.def;
      if (z.stunTimer > 0) {
        z.stunTimer -= dt;
        z.vx = z.vy = 0;
        continue;
      }

      z.attackTimer -= dt;
      z.acqTimer -= dt;
      z.screamCooldown -= dt;

      if (z.acqTimer <= 0) {
        z.acqTimer = 0.25 + this.rng.next() * 0.15;
        let best: SimShooter | null = null, bestDist = 1e9;
        const det = z.state === 'sleep' ? d.det * 0.5 : d.det;

        const candidates: SimShooter[] = [...this.bros, ...this.raiders];
        for (const c of candidates) {
          if (c.dead) continue;
          const dd = Math.hypot(c.x - z.x, c.y - z.y);
          if (dd < det && dd < bestDist) {
            bestDist = dd;
            best = c;
          }
        }
        if (best) {
          z.state = 'chase';
          z.target = best;
        } else if (z.state === 'chase' && (!z.target || z.target.dead || Math.hypot(z.target.x - z.x, z.target.y - z.y) > d.det * 1.7)) {
          z.state = 'alert';
          z.alertPos = z.target ? { x: z.target.x, y: z.target.y } : { x: z.x, y: z.y };
          z.alertTimer = 3;
          z.target = null;
        }
      }

      let dx = 0, dy = 0, sp = 0;
      if (z.state === 'chase' && z.target && !z.target.dead) {
        const ddx = z.target.x - z.x, ddy = z.target.y - z.y, dist = Math.hypot(ddx, ddy) || 0.001;
        z.angle = Math.atan2(ddy, ddx);
        if (dist <= d.reach + z.target.r) {
          if (z.attackTimer <= 0) {
            z.attackTimer = d.cd;
            this.hurt(z.target, d.dmg, null, null, dist);
          }
        } else {
          dx = ddx / dist; dy = ddy / dist;
          sp = d.run * z.speedMul;
          dx += -dy * 0.25 * z.drift * Math.sin(this.time * 1.7 + z.id);
          dy += dx * 0.25 * z.drift * Math.sin(this.time * 1.7 + z.id);
        }
      } else if (z.state === 'alert') {
        const ddx = z.alertPos.x - z.x, ddy = z.alertPos.y - z.y, dist = Math.hypot(ddx, ddy);
        z.alertTimer -= dt;
        if (dist < 1.2 || z.alertTimer <= 0) {
          z.state = 'wander';
        } else {
          dx = ddx / dist; dy = ddy / dist;
          sp = d.spd * 1.3 * z.speedMul;
          z.angle = Math.atan2(dy, dx);
        }
      } else if (z.state === 'wander') {
        z.wanderTimer -= dt;
        if (z.wanderTimer <= 0) {
          z.wanderTimer = this.rng.range(2, 5);
          z.wanderAngle = this.rng.range(0, Math.PI * 2);
        }
        dx = Math.cos(z.wanderAngle); dy = Math.sin(z.wanderAngle);
        sp = d.spd * 0.4 * z.speedMul;
      }

      z.vx = dx * sp;
      z.vy = dy * sp;
      if (z.vx !== 0 || z.vy !== 0) {
        this.moveEntity(z, z.vx * dt, z.vy * dt);
      }
    }
  }

  private spawnTick(dt: number): void {
    if (this.cfg.pattern !== 'waves' || this.zSpawned >= this.cfg.zombies) return;
    this.waveTimer -= dt;
    if (this.waveTimer > 0) return;
    this.waveTimer = this.cfg.waveEvery;

    const size = Math.max(1, Math.ceil(this.cfg.zombies / this.cfg.waves));
    for (let i = 0; i < size && this.zSpawned < this.cfg.zombies; i++) {
      const [x, y] = this.edgePoint();
      const z = this.spawnZombie(x, y, 'alert');
      z.alertPos = { x: 14, y: 20 };
      z.alertTimer = 30;
    }
  }

  step(dt = 1 / 30): void {
    if (this.over) return;
    this.time += dt;
    this.spawnTick(dt);

    for (const b of this.bros) {
      if (!b.dead) this.updateBot(b, dt);
    }
    for (const r of this.raiders) {
      if (!r.dead) this.updateBot(r, dt);
    }
    this.updateZombies(dt);
    this.updateBullets(dt);

    this.zombies = this.zombies.filter((z) => !z.dead);
    this.bros = this.bros.filter((b) => !b.dead);
    this.raiders = this.raiders.filter((r) => !r.dead);

    this.sampleTimer -= dt;
    if (this.sampleTimer <= 0) {
      this.sampleTimer = 0.5;
      this.stats.series.push([
        Math.round(this.time * 10) / 10,
        this.bros.length,
        this.raiders.length,
        this.zombies.length,
      ]);
    }

    const broAlive = this.bros.length > 0;
    const foesLeft = this.zombies.length + this.raiders.length > 0 || this.zSpawned < this.cfg.zombies;

    if (!broAlive) this.over = 'wiped';
    else if (!foesLeft) this.over = 'cleared';
    else if (this.time >= this.cfg.timeLimit) this.over = 'time';
  }

  summary(): SwarmSimSummary {
    const st = this.stats;
    const T = Math.max(0.1, this.time);
    const W = Object.values(st.w).filter((x) => x.team === 'bro');
    const sum = (k: keyof WeaponStats): number => W.reduce((a, x) => a + (typeof x[k] === 'number' ? (x[k] as number) : 0), 0);

    const shots = sum('shots');
    const pellets = sum('pellets');
    const hits = sum('pHit');
    const kills = sum('kills');
    const dmg = sum('dmg');
    const ttk = W.flatMap((x) => x.ttk);
    const kd = sum('killDist');
    const over = sum('over');
    const noise = sum('noise');
    const alerts = sum('alerts');
    const bk = Object.values(st.kills.bro).reduce((a, b) => a + b, 0);

    return {
      outcome: this.over || 'running',
      time: Math.round(this.time * 10) / 10,
      seed: this.seed,
      kills: bk,
      weaponKills: kills,
      killsPerMin: Math.round((bk / T) * 6000) / 100,
      shots,
      accuracy: pellets ? Math.round((hits / pellets) * 1000) / 1000 : 0,
      dmgDealt: Math.round(dmg),
      dmgPerShot: shots ? Math.round((dmg / shots) * 10) / 10 : 0,
      ammoPerKill: kills ? Math.round((shots / kills) * 100) / 100 : null,
      avgKillDist: kills ? Math.round((kd / kills) * 10) / 10 : null,
      avgTTK: ttk.length ? Math.round((ttk.reduce((a, b) => a + b, 0) / ttk.length) * 100) / 100 : null,
      overkillPct: dmg ? Math.round((100 * over / dmg) * 10) / 10 : 0,
      noisePerKill: kills ? Math.round((noise / kills) * 10) / 10 : null,
      alertsPerKill: kills ? Math.round((alerts / kills) * 100) / 100 : null,
      broDeaths: st.deaths.bro,
      broDmgTaken: Math.round(st.dmgTaken.bro),
      broAlive: this.bros.length,
      zombieKills: bk - (st.kills.bro.raider || 0) - (st.kills.bro.heavy || 0) - (st.kills.bro.marksman || 0),
      raiderKills: (st.kills.bro.raider || 0) + (st.kills.bro.heavy || 0) + (st.kills.bro.marksman || 0),
      raidersLost: st.deaths.raider,
    };
  }
}

export function defaultSwarmCfg(): SwarmSimCfg {
  return {
    mode: 'pvpve',
    pattern: 'waves',
    zombies: 120,
    raiders: 8,
    bros: 4,
    waves: 6,
    waveEvery: 8,
    timeLimit: 90,
    broWeapon: 'nailgun_smg',
    raiderWeapon: 'mixed',
    broBrain: { reaction: 0.25, aimErr: 0.02, kite: 3.5, lead: 0.6, cover: 0.5, aggr: 1 },
    raiderBrain: { reaction: 0.45, aimErr: 0.05, kite: 2.5, lead: 0.3, cover: 0.4, aggr: 1 },
  };
}

export async function runSwarmHeadless(cfg: SwarmSimCfg, seed: number): Promise<SwarmSimSummary> {
  const sim = new SwarmSim(cfg, seed);
  while (!sim.over) {
    sim.step(1 / 30);
  }
  return sim.summary();
}

export async function compareWeaponsPaired(
  weapons: string[],
  baseCfg: SwarmSimCfg,
  startSeed = 7,
  runs = 4
): Promise<Record<string, SwarmSimSummary[]>> {
  const results: Record<string, SwarmSimSummary[]> = {};
  for (const w of weapons) {
    results[w] = [];
    for (let r = 0; r < runs; r++) {
      const cfg: SwarmSimCfg = { ...baseCfg, broWeapon: w };
      const res = await runSwarmHeadless(cfg, startSeed + r);
      results[w].push(res);
    }
  }
  return results;
}
