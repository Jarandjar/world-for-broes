import type { Content, ColorSet, Id, Slot } from '../data/schema';
import { SLOTS } from '../data/schema';
import { HAIR_COLORS, HAIR_STYLES, SKIN_TONES, EYE_COLORS, BEARDS, OUTFIT_COLORS } from '../data/appearance';
import type { Rng } from '../engine/rng';

/** Everything that makes a character look different. All of it is plain data and serializable. */
export interface Appearance { skin: string; hairStyle: Id; hairColor: string; eyeColor: string; beard: Id; build: number; wheelchair: boolean }
export interface EquipInstance { item: Id; colors?: Partial<ColorSet> }
export type Outfit = Partial<Record<Slot, EquipInstance>>;
export interface Look { appearance: Appearance; outfit: Outfit }

const HEX = /^#[0-9a-fA-F]{6}$/;
const okHex = (v: unknown, fallback: string): string => (typeof v === 'string' && HEX.test(v) ? v : fallback);

/** The survivor from the reference art: hood, jacket, pack with bedroll, boots, an axe. */
export function defaultLook(): Look {
  return {
    appearance: { skin: SKIN_TONES[1] as string, hairStyle: 'messy', hairColor: HAIR_COLORS[1] as string, eyeColor: EYE_COLORS[0] as string, beard: 'none', build: 1, wheelchair: false },
    outfit: { head: { item: 'hood' }, torso: { item: 'hoodie' }, legs: { item: 'jeans', colors: { primary: '#33363b', secondary: '#25282c' } }, feet: { item: 'boots' }, hands: { item: 'leather_gloves' }, back: { item: 'hiker_pack' }, mainHand: { item: 'axe' } },
  };
}

export function randomLook(content: Content, rng: Rng): Look {
  const bySlot = (slot: Slot): Id[] => [...content.items.values()].filter((i) => i.equip?.slot === slot && i.kind === 'gear').map((i) => i.id);
  const outfit: Outfit = {};
  const pickColor = (): string => rng.pick(OUTFIT_COLORS);
  const add = (slot: Slot, chance: number): void => {
    const ids = bySlot(slot);
    if (ids.length && rng.chance(chance)) outfit[slot] = { item: rng.pick(ids), colors: rng.chance(0.7) ? { primary: pickColor(), secondary: pickColor() } : undefined };
  };
  add('torso', 1); add('legs', 1); add('feet', 1); add('head', 0.55); add('neck', 0.3); add('hands', 0.35); add('back', 0.6); add('face', 0.15); add('belt', 0.25);
  const tools = [...content.items.values()].filter((i) => i.equip?.slot === 'mainHand').map((i) => i.id);
  if (rng.chance(0.7)) outfit.mainHand = { item: rng.pick(tools) };
  return {
    appearance: { skin: rng.pick(SKIN_TONES), hairStyle: rng.pick(HAIR_STYLES).id, hairColor: rng.pick(HAIR_COLORS), eyeColor: rng.pick(EYE_COLORS), beard: rng.chance(0.25) ? rng.pick(BEARDS).id : 'none', build: rng.range(0.9, 1.1), wheelchair: false },
    outfit,
  };
}

/** Equip an item into a slot (or clear with null). Enforces slot match and two-handed rules. */
export function equip(look: Look, content: Content, slot: Slot, itemId: Id | null, colors?: Partial<ColorSet>): boolean {
  if (itemId === null) { delete look.outfit[slot]; return true; }
  const def = content.items.get(itemId);
  if (!def?.equip || def.equip.slot !== slot) return false;
  look.outfit[slot] = { item: itemId, colors };
  if (slot === 'mainHand' && def.equip.twoHanded) delete look.outfit.offHand;
  return true;
}

export function resolvedColors(content: Content, inst: EquipInstance): ColorSet {
  const base = content.items.get(inst.item)?.equip?.colors ?? { primary: '#777777', secondary: '#555555', accent: '#cccccc' };
  return { primary: inst.colors?.primary ?? base.primary, secondary: inst.colors?.secondary ?? base.secondary, accent: inst.colors?.accent ?? base.accent };
}

/** Stable string for caching and change detection. */
export function lookSignature(look: Look): string {
  const a = look.appearance;
  const o = SLOTS.map((s) => { const e = look.outfit[s]; return e ? `${s}:${e.item}:${e.colors?.primary ?? ''}:${e.colors?.secondary ?? ''}:${e.colors?.accent ?? ''}` : ''; }).join('|');
  return `${a.skin}|${a.hairStyle}|${a.hairColor}|${a.eyeColor}|${a.beard}|${a.build.toFixed(2)}|${a.wheelchair ? 1 : 0}|${o}`;
}

/** Validates untrusted JSON (a save file or a shared character) and falls back to defaults where needed. */
export function sanitizeLook(content: Content, raw: unknown): Look {
  const d = defaultLook();
  if (!raw || typeof raw !== 'object') return d;
  const r = raw as { appearance?: Partial<Appearance>; outfit?: Record<string, { item?: unknown; colors?: Partial<ColorSet> }> };
  const a = r.appearance ?? {};
  const appearance: Appearance = {
    skin: okHex(a.skin, d.appearance.skin), hairColor: okHex(a.hairColor, d.appearance.hairColor), eyeColor: okHex(a.eyeColor, d.appearance.eyeColor),
    hairStyle: content.hairStyles.some((h) => h.id === a.hairStyle) ? (a.hairStyle as Id) : d.appearance.hairStyle,
    beard: BEARDS.some((b) => b.id === a.beard) ? (a.beard as Id) : 'none',
    build: typeof a.build === 'number' ? Math.min(1.2, Math.max(0.8, a.build)) : 1,
    wheelchair: a.wheelchair === true,
  };
  const outfit: Outfit = {};
  for (const s of SLOTS) {
    const e = r.outfit?.[s];
    if (e && typeof e.item === 'string') {
      const def = content.items.get(e.item);
      if (def?.equip?.slot === s) outfit[s] = { item: e.item, colors: e.colors ? { primary: e.colors.primary && HEX.test(e.colors.primary) ? e.colors.primary : undefined, secondary: e.colors.secondary && HEX.test(e.colors.secondary) ? e.colors.secondary : undefined, accent: e.colors.accent && HEX.test(e.colors.accent) ? e.colors.accent : undefined } : undefined };
    }
  }
  return { appearance, outfit };
}
