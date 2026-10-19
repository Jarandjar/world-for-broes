/**
 * Weapons are a BASE (a GunDef or melee tool) plus ATTACHMENTS in SLOTS. Attachments are data: they change stats, add goofy
 * jank quirks, and some give INFO (HUD readouts that only exist if you fitted a part that provides them).
 * Nothing here knows about Game or rendering. See build.ts for the maths and docs/WEAPONS.md for the design.
 */
export type SlotId = 'muzzle' | 'barrel' | 'optic' | 'info' | 'underbarrel' | 'side' | 'feed' | 'stock' | 'grip' | 'charm' | 'head' | 'shaft' | 'pommel';
export const SLOT_ORDER: SlotId[] = ['head', 'muzzle', 'barrel', 'shaft', 'optic', 'info', 'underbarrel', 'side', 'feed', 'stock', 'grip', 'pommel', 'charm'];
export const SLOT_LABEL: Record<SlotId, string> = {
  muzzle: 'Muzzle', barrel: 'Barrel', optic: 'Sight', info: 'Info gizmo', underbarrel: 'Underbarrel', side: 'Side rail', feed: 'Feed',
  stock: 'Stock', grip: 'Grip', charm: 'Charm', head: 'Head', shaft: 'Shaft', pommel: 'Pommel',
};
/** How a part attaches. `tape` is the homemade mount: duct tape fits most things, badly. */
export type Mount = 'thread' | 'tube' | 'rail' | 'mag' | 'tape' | 'any';
export const MOUNT_LABEL: Record<Mount, string> = { thread: 'threaded', tube: 'barrel swap', rail: 'rail', mag: 'magazine well', tape: 'duct tape', any: 'any' };

/** Readouts. Only a fitted attachment that gives a mode switches it on. */
export type InfoMode = 'ammo' | 'range' | 'falloff' | 'noise' | 'spread' | 'trajectory' | 'enemy_hp' | 'threat';
export const INFO_ORDER: InfoMode[] = ['ammo', 'trajectory', 'spread', 'range', 'falloff', 'noise', 'enemy_hp', 'threat'];
export const INFO_LABEL: Record<InfoMode, string> = {
  ammo: 'exact ammo count', range: 'range rings', falloff: 'damage falloff', noise: 'noise ring', spread: 'spread cone',
  trajectory: 'aim line', enemy_hp: 'enemy health bars', threat: 'threat compass',
};

/** Every number an attachment can change. Guns use all of them; melee tools use damage, rate, range, noise and weight. */
export interface Stats {
  damage: number;   // per projectile (guns) or per hit (melee)
  pellets: number;
  rate: number;     // shots (or swings) per second
  spread: number;   // radians of random cone
  range: number;    // maximum range / reach in tiles
  noise: number;    // alert radius in tiles
  pen: number;      // targets pierced
  speed: number;    // projectile speed
  light: number;    // light radius while held
  weight: number;
  ammo: number;     // rounds used per shot
}
export const STAT_KEYS = ['damage', 'pellets', 'rate', 'spread', 'range', 'noise', 'pen', 'speed', 'light', 'weight', 'ammo'] as const satisfies readonly (keyof Stats)[];
export const STAT_META: Record<keyof Stats, { label: string; unit: string; better: 'up' | 'down' | 'neutral'; digits: number; scale?: number }> = {
  damage: { label: 'Damage', unit: '', better: 'up', digits: 0 },
  pellets: { label: 'Pellets', unit: '', better: 'up', digits: 0 },
  rate: { label: 'Fire rate', unit: '/s', better: 'up', digits: 2 },
  spread: { label: 'Spread', unit: '°', better: 'down', digits: 1, scale: 180 / Math.PI },
  range: { label: 'Range', unit: ' m', better: 'up', digits: 0 },
  noise: { label: 'Noise', unit: ' m', better: 'down', digits: 0 },
  pen: { label: 'Pierce', unit: '', better: 'up', digits: 0 },
  speed: { label: 'Bullet speed', unit: '', better: 'up', digits: 0 },
  light: { label: 'Light', unit: ' m', better: 'neutral', digits: 0 },
  weight: { label: 'Weight', unit: '', better: 'down', digits: 1 },
  ammo: { label: 'Ammo/shot', unit: '', better: 'down', digits: 0 },
};

/** Goofy side effects. Every one of these is implemented in the simulation (Game.shootGun and projectile hits). */
export type Quirk =
  | { t: 'noise_pulse'; extra: number }           // an extra racket every shot: bells, kazoos, rubber ducks (a lure, on purpose)
  | { t: 'wobble'; rad: number }                  // random extra spread
  | { t: 'backfire'; chance: number }             // soot-faced and a little hurt
  | { t: 'confetti' }                             // cosmetic burst at the muzzle
  | { t: 'stun'; chance: number; secs: number }   // target freezes for a moment on hit
  | { t: 'knockback'; power: number }             // target is shoved away
  | { t: 'ammo_leak'; chance: number }            // eats an extra round
  | { t: 'ammo_save'; chance: number }            // sometimes the shot is free
  | { t: 'sanity'; amount: number }               // each shot cheers you up
  | { t: 'recoil_boing'; strength: number };      // you get pushed backwards

export type Rarity = 'J' | 'C' | 'U' | 'R' | 'X';
export type GunClass = 'pistol' | 'revolver' | 'smg' | 'shotgun' | 'rifle' | 'tube' | 'toy' | 'melee';
export interface Profile { cls: GunClass; tags: string[]; slots: Partial<Record<SlotId, Mount[]>> }

export interface AttachmentDef {
  id: string; name: string; slot: SlotId; mount: Mount; rarity: Rarity; desc: string;
  add?: Partial<Stats>; mul?: Partial<Stats>; quirks?: Quirk[];
  /** Readouts this part switches on. */
  info?: InfoMode[];
  /** The weapon must have all of these tags / none of these tags. */
  requires?: string[]; forbid?: string[];
  /** Attachment ids that cannot be fitted at the same time. */
  conflicts?: string[];
  tags?: string[]; homemade?: boolean;
  /** Junk it is made from (item ids). */
  recipe?: Record<string, number>;
}
