import { EFFECT_IDS, SURFACE_TAGS, type Content, type EffectId } from './schema';
/** Returns a list of problems. Empty list = content is consistent. Run in tests and at boot (dev). */
export function validateContent(c: Content): string[] {
  const errs: string[] = [];
  const hasItem = (id: string): boolean => c.items.has(id);
  for (const i of c.items.values()) {
    if (i.equip && !i.equip.colors.primary) errs.push(`item ${i.id}: missing colors`);
    if (i.stack < 1) errs.push(`item ${i.id}: stack < 1`);
  }
  for (const r of c.recipes) {
    if (!hasItem(r.output.item)) errs.push(`recipe ${r.id}: unknown output ${r.output.item}`);
    for (const k of Object.keys(r.inputs)) if (!hasItem(k)) errs.push(`recipe ${r.id}: unknown input ${k}`);
  }
  for (const o of c.objects.values()) {
    for (const d of o.drops) if (!hasItem(d.item)) errs.push(`object ${o.kind}: unknown drop ${d.item}`);
    if (o.becomes && !c.objects.has(o.becomes)) errs.push(`object ${o.kind}: unknown becomes ${o.becomes}`);
  }
  for (const b of c.biomes) {
    if (!c.terrain.has(b.terrain.base) || !c.terrain.has(b.terrain.patch)) errs.push(`biome ${b.id}: unknown terrain`);
    for (const r of b.objects) if (!c.objects.has(r.kind)) errs.push(`biome ${b.id}: unknown object ${r.kind}`);
  }
  for (const s of c.structures.values()) for (const k of Object.keys(s.cost)) if (!hasItem(k)) errs.push(`structure ${s.id}: unknown cost item ${k}`);
  for (const b of c.buildings) {
    if (b.weight <= 0) errs.push(`building ${b.id}: weight must be > 0`);
    if (b.minW < 4 || b.minH < 4 || b.minW > b.maxW || b.minH > b.maxH) errs.push(`building ${b.id}: bad size range`);
  }
  // status effects, reactions and stagger (Z26)
  const hasEffect = (id: string): boolean => c.effects.has(id as EffectId);
  const isTag = (t: string): boolean => (SURFACE_TAGS as readonly string[]).includes(t);
  for (const id of EFFECT_IDS) if (!c.effects.has(id)) errs.push(`effect ${id}: missing`);
  for (const e of c.effects.values()) {
    if (!/^#[0-9a-fA-F]{6}$/.test(e.color)) errs.push(`effect ${e.id}: colour must be #rrggbb`);
    if (!e.icon || !e.soundWord || !e.tooltip) errs.push(`effect ${e.id}: needs icon, sound word and tooltip`);
    if (e.duration < 0) errs.push(`effect ${e.id}: negative duration`);
    for (const k of e.clears ?? []) if (!hasEffect(k)) errs.push(`effect ${e.id}: clears unknown effect ${k}`);
    for (const t of [...(e.grantsTags ?? []), ...(e.atMax?.tags ?? [])]) if (!isTag(t)) errs.push(`effect ${e.id}: unknown tag ${t}`);
    if (e.atMax && !e.maxStacks) errs.push(`effect ${e.id}: atMax needs maxStacks`);
  }
  const seen = new Set<string>();
  for (const r of c.reactions) {
    const key = `${r.effect}:${r.tag}`;
    if (seen.has(key)) errs.push(`reaction ${key}: duplicate`);
    seen.add(key);
    if (!hasEffect(r.effect)) errs.push(`reaction ${key}: unknown effect`);
    if (!isTag(r.tag)) errs.push(`reaction ${key}: unknown tag`);
    for (const k of [...(r.apply ?? []), ...(r.remove ?? []), ...(r.spawnHazard ? [r.spawnHazard] : [])]) if (!hasEffect(k)) errs.push(`reaction ${key}: unknown effect ${k}`);
    if (r.spawnHazard && !c.effects.get(r.spawnHazard)?.hazard) errs.push(`reaction ${key}: ${r.spawnHazard} has no hazard`);
  }
  const ladder = c.staggerLadder;
  if (ladder.length === 0) errs.push('stagger ladder: empty');
  for (let i = 1; i < ladder.length; i++) if (ladder[i]!.share <= ladder[i - 1]!.share) errs.push('stagger ladder: shares must rise');
  const L = c.effectLimits;
  if (L.maxChainDepth < 1 || L.maxFires < 1 || L.maxPuddles < 1 || L.maxClouds < 1 || L.maxChainTargets < 1) errs.push('effect limits: caps must be >= 1');
  for (const z of c.zombies.values()) {
    for (const k of z.immune ?? []) if (!hasEffect(k)) errs.push(`zombie ${z.id}: unknown immunity ${k}`);
    for (const t of z.tags ?? []) if (!isTag(t)) errs.push(`zombie ${z.id}: unknown tag ${t}`);
    if (z.attackEffect && !hasEffect(z.attackEffect.effect)) errs.push(`zombie ${z.id}: unknown attack effect`);
  }
  for (const t of c.terrain.values()) for (const tag of t.tags ?? []) if (!isTag(tag)) errs.push(`terrain ${t.id}: unknown tag ${tag}`);
  for (const o of c.objects.values()) for (const tag of o.tags ?? []) if (!isTag(tag)) errs.push(`object ${o.kind}: unknown tag ${tag}`);
  return errs;
}
