import type { GunDef, ItemDef, RecipeDef } from '../data/schema';
import { ATTACHMENTS, ATTACHMENT_BY_ID } from './attachments';
import { SLOT_ORDER, STAT_KEYS, STAT_META, type AttachmentDef, type GunClass, type InfoMode, type Mount, type Profile, type Quirk, type SlotId, type Stats } from './schema';

export type Loadout = Partial<Record<SlotId, string>>;

// ------------------------------------------------------------------ which slots each weapon has
const rail: Mount[] = ['rail', 'tape'], bare: Mount[] = ['any', 'tape'];
const FIREARM: Profile['slots'] = { muzzle: ['thread', 'tape'], barrel: ['tube'], optic: rail, info: rail, underbarrel: rail, side: rail, feed: ['mag', 'tape'], stock: bare, grip: bare, charm: bare };
const profile = (cls: GunClass, tags: string[], slots: Profile['slots']): Profile => ({ cls, tags, slots });
export const GUN_PROFILES: Record<string, Profile> = {
  hunting_rifle: profile('rifle', ['firearm', 'long'], FIREARM),
  pump_shotgun: profile('shotgun', ['firearm', 'pellet'], FIREARM),
  nailgun_smg: profile('smg', ['firearm', 'auto'], FIREARM),
  plinker_pistol: profile('pistol', ['firearm', 'short'], { ...FIREARM, stock: ['tape'] }),
  bark_revolver: profile('revolver', ['firearm', 'short', 'revolver'], { ...FIREARM, stock: ['tape'] }),
  slingshot: profile('toy', ['toy', 'elastic'], { optic: ['tape'], info: ['tape'], side: ['tape'], grip: bare, charm: bare }),
  potato_cannon: profile('tube', ['tube'], { muzzle: ['thread', 'tape'], barrel: ['tube', 'tape'], optic: ['tape'], info: ['tape'], side: ['tape'], underbarrel: ['tape'], stock: bare, grip: bare, charm: bare }),
  foam_dart_blaster: profile('toy', ['toy', 'blaster', 'auto'], { muzzle: ['thread', 'tape'], optic: rail, info: rail, side: rail, underbarrel: rail, feed: ['mag', 'tape'], stock: bare, grip: bare, charm: bare }),
  cork_crossbow: profile('toy', ['toy', 'bow'], { optic: rail, info: rail, side: rail, underbarrel: rail, stock: bare, grip: bare, charm: bare }),
  chicken_cannon: profile('tube', ['tube', 'chicken'], { muzzle: ['tape'], barrel: ['tube', 'tape'], optic: ['tape'], info: ['tape'], side: ['tape'], stock: bare, grip: bare, charm: bare }),
  rubber_band_gatling: profile('toy', ['toy', 'auto'], { muzzle: ['tape'], optic: ['tape'], info: ['tape'], side: ['tape'], underbarrel: ['tape'], feed: ['tape'], stock: ['tape'], grip: bare, charm: bare }),
};
const MELEE: Profile = profile('melee', ['melee'], { head: ['tape', 'any'], shaft: ['tape'], grip: bare, pommel: ['tape'], side: ['tape'], info: ['tape'], charm: bare });

/** The slot layout for an item, or null if it cannot take attachments. Guns have a table; any melee weapon gets the melee layout. */
export function profileFor(id: string, def?: Pick<ItemDef, 'gun' | 'tool'> | null): Profile | null {
  const p = GUN_PROFILES[id];
  if (p) return p;
  if (def?.gun) return profile('toy', ['toy'], { optic: ['tape'], info: ['tape'], side: ['tape'], grip: bare, charm: bare });
  if (def?.tool?.kind === 'weapon') return MELEE;
  return null;
}
export const slotsOf = (p: Profile): SlotId[] => SLOT_ORDER.filter((s) => p.slots[s]);

// ------------------------------------------------------------------ compatibility
export type Fit = { ok: true; replaces?: string } | { ok: false; reason: string };
export function canFit(p: Profile, att: AttachmentDef, loadout: Loadout = {}): Fit {
  const mounts = p.slots[att.slot];
  if (!mounts) return { ok: false, reason: `This weapon has no ${att.slot} slot` };
  if (!mounts.includes(att.mount)) return { ok: false, reason: `Does not fit: needs a ${att.mount} mount here` };
  const missing = (att.requires ?? []).find((t) => !p.tags.includes(t));
  if (missing) return { ok: false, reason: `Needs a ${missing} weapon` };
  const bad = (att.forbid ?? []).find((t) => p.tags.includes(t));
  if (bad) return { ok: false, reason: `Will not work on ${bad} weapons` };
  const clash = (att.conflicts ?? []).find((c) => Object.values(loadout).includes(c));
  if (clash) return { ok: false, reason: `Clashes with ${ATTACHMENT_BY_ID[clash]?.name ?? clash}` };
  const cur = loadout[att.slot];
  return cur && cur !== att.id ? { ok: true, replaces: cur } : { ok: true };
}

// ------------------------------------------------------------------ stats
export function gunToStats(g: GunDef, weight = 1.5): Stats {
  return { damage: g.baseDamage, pellets: g.pelletCount ?? 1, rate: g.fireRate, spread: g.spreadAngle, range: g.maxRange, noise: g.noiseRadius, pen: g.penetration ?? 1, speed: g.projectileSpeed, light: 0, weight, ammo: g.ammoPerShot };
}
export function meleeToStats(t: { damage: number; speed: number; reach: number }, weight = 1): Stats {
  return { damage: t.damage, pellets: 1, rate: t.speed, spread: 0, range: t.reach, noise: 0, pen: 1, speed: 0, light: 0, weight, ammo: 0 };
}
const clampStats = (s: Stats): Stats => ({
  damage: Math.max(0, s.damage), pellets: Math.max(1, Math.round(s.pellets)), rate: Math.max(0.1, s.rate), spread: Math.max(0.002, s.spread), range: Math.max(1, s.range),
  noise: Math.max(0, s.noise), pen: Math.max(1, Math.round(s.pen)), speed: Math.max(3, s.speed), light: Math.max(0, s.light), weight: Math.max(0.1, s.weight), ammo: Math.max(1, Math.round(s.ammo)),
});
/** Additions first, then multipliers, so two scopes do not multiply a flat bonus twice. Spread is not clamped for melee (it is 0). */
export function applyAttachments(base: Stats, atts: AttachmentDef[]): Stats {
  const s: Stats = { ...base };
  for (const a of atts) for (const k of STAT_KEYS) s[k] += a.add?.[k] ?? 0;
  for (const a of atts) for (const k of STAT_KEYS) s[k] *= a.mul?.[k] ?? 1;
  const c = clampStats(s);
  if (base.spread === 0) c.spread = 0;
  if (base.ammo === 0) c.ammo = 0;
  if (base.speed === 0) c.speed = 0;
  return c;
}
export function applyToGun(g: GunDef, s: Stats): GunDef {
  const dmgK = g.baseDamage > 0 ? s.damage / g.baseDamage : 1, rngK = s.range / g.maxRange;
  return {
    ...g, baseDamage: Math.round(s.damage * 100) / 100, minDamage: Math.max(1, Math.round(g.minDamage * dmgK)), falloffStartDist: Math.max(1, g.falloffStartDist * rngK),
    maxRange: s.range, spreadAngle: s.spread, fireRate: s.rate, noiseRadius: s.noise, penetration: s.pen, projectileSpeed: s.speed, pelletCount: s.pellets > 1 ? s.pellets : g.pelletCount, ammoPerShot: s.ammo,
  };
}

export interface Derived {
  atts: AttachmentDef[]; stats: Stats; base: Stats;
  info: Set<InfoMode>; quirks: Quirk[]; light: number;
  /** Entries that no longer fit (for example after a rule change). They are ignored. */
  ignored: string[];
}
function collect(p: Profile, loadout: Loadout): { atts: AttachmentDef[]; ignored: string[] } {
  const atts: AttachmentDef[] = [], ignored: string[] = [];
  for (const slot of SLOT_ORDER) {
    const id = loadout[slot]; if (!id) continue;
    const a = ATTACHMENT_BY_ID[id];
    if (!a || a.slot !== slot) { ignored.push(id); continue; }
    const fit = canFit(p, a, { ...loadout, [slot]: undefined });
    if (!fit.ok) ignored.push(id); else atts.push(a);
  }
  return { atts, ignored };
}
function finish(p: Profile, base: Stats, intrinsic: Quirk[], loadout: Loadout): Derived {
  const { atts, ignored } = collect(p, loadout);
  const stats = applyAttachments(base, atts);
  const info = new Set<InfoMode>();
  for (const a of atts) for (const m of a.info ?? []) info.add(m);
  return { atts, stats, base, info, quirks: [...intrinsic, ...atts.flatMap((a) => a.quirks ?? [])], light: stats.light, ignored };
}
export function deriveGun(p: Profile, gun: GunDef, loadout: Loadout, weight = 1.5): Derived & { gun: GunDef } {
  const base = gunToStats(gun, weight), d = finish(p, base, gun.quirks ?? [], loadout);
  return { ...d, gun: { ...applyToGun(gun, d.stats), quirks: d.quirks } };
}
export function deriveMelee(p: Profile, tool: { damage: number; speed: number; reach: number; power: number; kind: string }, loadout: Loadout, weight = 1): Derived & { tool: typeof tool } {
  const base = meleeToStats(tool, weight), d = finish(p, base, [], loadout);
  return { ...d, tool: { ...tool, damage: d.stats.damage, speed: d.stats.rate, reach: d.stats.range } };
}
export const loadoutSignature = (l: Loadout): string => SLOT_ORDER.map((s) => l[s] ?? '').join('|');

// ------------------------------------------------------------------ menu helpers
export interface StatChange { key: keyof Stats; label: string; before: string; after: string; verdict: 'good' | 'bad' | 'neutral' | 'same' }
const show = (k: keyof Stats, v: number): string => { const m = STAT_META[k]; return `${(v * (m.scale ?? 1)).toFixed(m.digits)}${m.unit}`; };
/** What changes if `after` replaces `before`: used for the stat bars and chips in the Gunsmith. */
export function statChanges(before: Stats, after: Stats, only?: readonly (keyof Stats)[]): StatChange[] {
  const keys = only ?? STAT_KEYS;
  return keys.map((k) => {
    const m = STAT_META[k], b = before[k] * (m.scale ?? 1), a = after[k] * (m.scale ?? 1), same = Math.abs(a - b) < Math.pow(10, -m.digits) / 2;
    const up = a > b, verdict: StatChange['verdict'] = same ? 'same' : m.better === 'neutral' ? 'neutral' : (up === (m.better === 'up') ? 'good' : 'bad');
    return { key: k, label: m.label, before: show(k, before[k]), after: show(k, after[k]), verdict };
  });
}
export const dps = (s: Stats): number => s.damage * s.pellets * s.rate;
export function describeQuirk(q: Quirk): string {
  switch (q.t) {
    case 'noise_pulse': return `+${q.extra} m of racket each shot`;
    case 'wobble': return `wobbles (+${(q.rad * 180 / Math.PI).toFixed(1)}°)`;
    case 'backfire': return `${Math.round(q.chance * 100)}% backfire`;
    case 'confetti': return 'confetti';
    case 'stun': return `${Math.round(q.chance * 100)}% stun ${q.secs}s`;
    case 'knockback': return `knockback ${q.power}`;
    case 'ammo_leak': return `${Math.round(q.chance * 100)}% leaks a round`;
    case 'ammo_save': return `${Math.round(q.chance * 100)}% free shot`;
    case 'sanity': return `+${q.amount} sanity/shot`;
    case 'recoil_boing': return 'pushes you back';
  }
}

// ------------------------------------------------------------------ generated game content
/** Every attachment is an inventory item (so it can be found, crafted, carried, and fitted). */
export function attachmentItems(): ItemDef[] {
  return ATTACHMENTS.map((a) => ({ id: a.id, name: a.name, kind: 'material', stack: 1, weight: 0.3 + Math.max(0, a.add?.weight ?? 0) * 0.5, tags: ['attachment', a.slot, ...(a.homemade ? ['homemade'] : [])] }));
}
export function attachmentRecipes(): RecipeDef[] {
  return ATTACHMENTS.filter((a) => a.recipe).map((a) => ({ id: `craft_${a.id}`, name: a.name, inputs: a.recipe as Record<string, number>, output: { item: a.id, qty: 1 } }));
}
export { ATTACHMENTS, ATTACHMENT_BY_ID };
