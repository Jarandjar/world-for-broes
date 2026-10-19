import type { Content } from './schema';
import { ITEMS } from './items';
import { TERRAIN } from './terrain';
import { BIOMES } from './biomes';
import { OBJECTS } from './objects';
import { STRUCTURES } from './structures';
import { RECIPES } from './recipes';
import { HAIR_STYLES } from './appearance';
import { VEHICLES } from './vehicles';
import { ZOMBIES } from './zombies';
import { ANIMALS } from './animals';
import { BUILDINGS } from './buildings';
import { ATTACHMENTS } from './attachments';
import { EFFECTS, REACTIONS, STAGGER_LADDER, EFFECT_LIMITS } from './effects';

/** A content pack is just data. Add packs (for example a zombie pack) and merge them here. */
export interface ContentPack extends Partial<{
  items: typeof ITEMS; terrain: typeof TERRAIN; biomes: typeof BIOMES; objects: typeof OBJECTS;
  structures: typeof STRUCTURES; recipes: typeof RECIPES; hairStyles: typeof HAIR_STYLES;
  vehicles: typeof VEHICLES; zombies: typeof ZOMBIES; animals: typeof ANIMALS; buildings: typeof BUILDINGS;
  attachments: typeof ATTACHMENTS; effects: typeof EFFECTS; reactions: typeof REACTIONS;
  staggerLadder: typeof STAGGER_LADDER; effectLimits: typeof EFFECT_LIMITS;
}> {}

export const BASE_PACK: ContentPack = { items: ITEMS, terrain: TERRAIN, biomes: BIOMES, objects: OBJECTS, structures: STRUCTURES, recipes: RECIPES, hairStyles: HAIR_STYLES, vehicles: VEHICLES, zombies: ZOMBIES, animals: ANIMALS, buildings: BUILDINGS, attachments: ATTACHMENTS, effects: EFFECTS, reactions: REACTIONS, staggerLadder: STAGGER_LADDER, effectLimits: EFFECT_LIMITS };

export function buildContent(packs: ContentPack[] = [BASE_PACK]): Content {
  const c: Content = { items: new Map(), terrain: new Map(), biomes: [], objects: new Map(), structures: new Map(), recipes: [], hairStyles: [], vehicles: new Map(), zombies: new Map(), animals: new Map(), buildings: [], attachments: new Map(), effects: new Map(), reactions: [], staggerLadder: [...STAGGER_LADDER], effectLimits: { ...EFFECT_LIMITS } };
  for (const p of packs) {
    p.items?.forEach((i) => c.items.set(i.id, i));
    p.terrain?.forEach((t) => c.terrain.set(t.id, t));
    p.biomes?.forEach((b) => { const k = c.biomes.findIndex((x) => x.id === b.id); if (k >= 0) c.biomes[k] = b; else c.biomes.push(b); });
    p.objects?.forEach((o) => c.objects.set(o.kind, o));
    p.structures?.forEach((s) => c.structures.set(s.id, s));
    p.recipes?.forEach((r) => { const k = c.recipes.findIndex((x) => x.id === r.id); if (k >= 0) c.recipes[k] = r; else c.recipes.push(r); });
    p.hairStyles?.forEach((h) => { if (!c.hairStyles.some((x) => x.id === h.id)) c.hairStyles.push(h); });
    p.vehicles?.forEach((v) => c.vehicles.set(v.id, v));
    p.zombies?.forEach((z) => c.zombies.set(z.id, z));
    p.animals?.forEach((a) => c.animals.set(a.id, a));
    p.buildings?.forEach((b) => { const k = c.buildings.findIndex((x) => x.id === b.id); if (k >= 0) c.buildings[k] = b; else c.buildings.push(b); });
    p.attachments?.forEach((a) => c.attachments.set(a.id, a));
    p.effects?.forEach((e) => c.effects.set(e.id, e));
    p.reactions?.forEach((r) => { const k = c.reactions.findIndex((x) => x.effect === r.effect && x.tag === r.tag); if (k >= 0) c.reactions[k] = r; else c.reactions.push(r); });
    if (p.staggerLadder) c.staggerLadder = [...p.staggerLadder].sort((a, b) => a.share - b.share);
    if (p.effectLimits) c.effectLimits = { ...p.effectLimits };
  }
  return c;
}
