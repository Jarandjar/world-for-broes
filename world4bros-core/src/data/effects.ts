import type { EffectDef, EffectLimits, ReactionDef, StaggerStepDef } from './schema';

/** The twelve status effects (World Bible §115). Colours are the §115 swatches; numbers are starting values. */
export const EFFECTS: EffectDef[] = [
  {
    id: 'acid', name: 'Acid', color: '#b6e64a', icon: 'drip', soundWord: 'SSSZZ',
    tooltip: 'Acid: hurts over time and eats metal gear. Wash it off with water.',
    from: 'Spitter zombies, acid pools, toxic rain, acid sprayers.', does: 'Damage over time. Wears down metal gear. Leaves a lingering puddle.', ends: 'Water, antidote gel, or about 6 s.',
    duration: 6, dps: 3, hazard: { kind: 'puddle', radius: 0.9, duration: 10 },
  },
  {
    id: 'burn', name: 'Burn', color: '#F08A32', icon: 'flame', soundWord: 'FWOOSH',
    tooltip: 'Burning: hurts over time and spreads to anything that burns. Water puts it out.',
    from: 'Fire, lava, flame weapons, explosions.', does: 'Damage over time. Spreads to anything flammable. Lights the area.', ends: 'Water, or about 5 s.',
    duration: 5, dps: 4, clears: ['cold'], hazard: { kind: 'fire', radius: 0.8, duration: 8 },
  },
  {
    id: 'explosion', name: 'Explosion', color: '#F9C45A', icon: 'starburst', soundWord: 'KA-BOOM',
    tooltip: 'Explosion: big hit that fades with distance and knocks you down.',
    from: 'Barrels, sappers, rockets, gas leaks.', does: 'Area damage that fades with distance, knockback, a crater, big noise.', ends: 'Instant.',
    duration: 0, burst: 30, radius: 2.6, impact: 100, blast: true, flashes: true,
  },
  {
    id: 'stagger', name: 'Stagger', color: '#EAE0C8', icon: 'impact', soundWord: 'WHUMP',
    tooltip: 'Stagger: a heavy hit knocks you off balance. Bigger hits, longer wobble.',
    from: 'Heavy hits, blasts, knockback guns.', does: 'A flinch, a stagger or a knockdown, depending on poise.', ends: '0.2 to 1.5 s.',
    duration: 0, impact: 20,
  },
  {
    id: 'electricity', name: 'Electricity', color: '#FDEB90', icon: 'bolt', soundWord: 'BZZZT',
    tooltip: 'Shocked: a jolt and a short stun. Water and metal pass it along.',
    from: 'Fences, tasers, storms, tesla coils.', does: 'Shock damage and a short stun. Chains through wet and metal. Stalls vehicles.', ends: 'Instant jolt, then the stun.',
    duration: 0.6, burst: 8, stun: true, flashes: true,
  },
  {
    id: 'cold', name: 'Cold', color: '#8fd8e8', icon: 'flake', soundWord: 'BRRR',
    tooltip: 'Cold: slows you down; too much and you freeze solid. Warm up by a fire.',
    from: 'Tundra, ice weapons, frozen water.', does: 'Slows, then freezes in place. Frozen things shatter when blasted.', ends: 'Warmth or fire.',
    duration: 6, slow: 0.8, maxStacks: 3, atMax: { stun: true, tags: ['frozen'], duration: 2.5 },
  },
  {
    id: 'spores', name: 'Spores', color: '#c084fc', icon: 'spores', soundWord: 'PFFT',
    tooltip: 'Spores: your head swims and your sanity drains. Get out of the cloud or into the light.',
    from: 'The Hollow, bogs, fungus.', does: 'Drains sanity and makes vision swim.', ends: 'Leaving the cloud, or light.',
    duration: 4, sanityDps: 4, sight: 0.8, endsInLight: true, hazard: { kind: 'cloud', radius: 2, duration: 10 },
  },
  {
    id: 'goo', name: 'Goo', color: '#f472b6', icon: 'blob', soundWord: 'SPLORCH',
    tooltip: 'Goo: sticky and slow, and easier to hit. Wash it off or wiggle free.',
    from: 'Slime zombies, sticky traps.', does: 'Slows and sticks. Easy to hit.', ends: 'Water, or struggling free.',
    duration: 4, slow: 0.45, damageTaken: 1.25, hazard: { kind: 'puddle', radius: 0.9, duration: 12 },
  },
  {
    id: 'wet', name: 'Wet', color: '#67e8f9', icon: 'droplet', soundWord: 'SPLOSH',
    tooltip: 'Wet: puts out fire, but electricity jumps to you and cold bites harder.',
    from: 'Rain, water, squirt guns.', does: 'Puts out fire, makes electricity chain, chills faster.', ends: 'Dries over time.',
    duration: 8, grantsTags: ['wet'], clears: ['burn', 'acid', 'goo'], hazard: { kind: 'puddle', radius: 1.1, duration: 15 },
  },
  {
    id: 'oil_slick', name: 'Oil Slick', color: '#9aa1ac', icon: 'slick', soundWord: 'SHLICK',
    tooltip: 'Oily: you slide around, and one spark sets you ablaze.',
    from: 'Spills, wrecks.', does: 'Slippery: you slide. Very flammable.', ends: 'Burns off or is wiped.',
    duration: 10, slide: true, grantsTags: ['oil', 'flammable'], hazard: { kind: 'puddle', radius: 1.2, duration: 30 },
  },
  {
    id: 'blind', name: 'Blind', color: '#f3e7b4', icon: 'eye', soundWord: 'SPLAT',
    tooltip: 'Blinded: you can barely see and your aim is ruined for a moment.',
    from: 'Cream pies, smoke, salt glare.', does: 'Blocks sight and ruins aim.', ends: 'About 2 s.',
    duration: 2, sight: 0.3, ruinsAim: true,
  },
  {
    id: 'smoke', name: 'Smoke', color: '#9aa1ac', icon: 'cloud', soundWord: 'KOFF',
    tooltip: 'Smoke: hides you from eyes, but makes you cough.',
    from: 'Fires, smoke pots.', does: 'Hides you from sight. A little coughing damage.', ends: 'Drifts away.',
    duration: 1, dps: 0.5, sight: 0.6, hidden: 0.4, hazard: { kind: 'cloud', radius: 2.2, duration: 8 },
  },
];

/**
 * The reaction table (§116), one row per effect × surface tag. Columns map to tags:
 * grass or wood = flammable, water = wet, oil = oil, metal = metal (+ conductive), ice = frozen, gas or barrel = explosive.
 */
export const REACTIONS: ReactionDef[] = [
  // Fire
  { effect: 'burn', tag: 'flammable', label: 'Spreads', amplifies: true, chainRadius: 1.6, spawnHazard: 'burn' },
  { effect: 'burn', tag: 'wet', label: 'Put out, steam', cancels: true, remove: ['wet'] },
  { effect: 'burn', tag: 'oil', label: 'Bigger blaze', amplifies: true, powerMult: 2, remove: ['oil_slick'], spawnHazard: 'burn' },
  { effect: 'burn', tag: 'metal', label: 'Heats up, nothing else' },
  { effect: 'burn', tag: 'frozen', label: 'Melts to water', cancels: true, remove: ['cold'], apply: ['wet'] },
  { effect: 'burn', tag: 'explosive', label: 'Explodes', amplifies: true, explode: true },
  // Electricity
  { effect: 'electricity', tag: 'flammable', label: 'Nothing' },
  { effect: 'electricity', tag: 'wet', label: 'Chains to every wet target', amplifies: true, chainRadius: 4 },
  { effect: 'electricity', tag: 'oil', label: 'Nothing' },
  { effect: 'electricity', tag: 'metal', label: 'Conducts: longer stun, vehicles stall', stunMult: 2, stallVehicle: 3 },
  { effect: 'electricity', tag: 'conductive', label: 'Conducts along metal', amplifies: true, chainRadius: 2.5 },
  { effect: 'electricity', tag: 'frozen', label: 'Nothing' },
  { effect: 'electricity', tag: 'explosive', label: 'Spark sets it off', explode: true },
  // Acid
  { effect: 'acid', tag: 'flammable', label: 'Weakens it', powerMult: 1.25 },
  { effect: 'acid', tag: 'wet', label: 'Dilutes the puddle', powerMult: 0.5 },
  { effect: 'acid', tag: 'oil', label: 'Floats on top' },
  { effect: 'acid', tag: 'metal', label: 'Corrodes gear', powerMult: 1.5 },
  { effect: 'acid', tag: 'frozen', label: 'Eats a pit', remove: ['cold'] },
  { effect: 'acid', tag: 'explosive', label: 'Nothing' },
  // Explosion
  { effect: 'explosion', tag: 'flammable', label: 'Scorches the ground' },
  { effect: 'explosion', tag: 'wet', label: 'Geyser and knockback', knockMult: 1.8 },
  { effect: 'explosion', tag: 'oil', label: 'Fireball', amplifies: true, apply: ['burn'], remove: ['oil_slick'], spawnHazard: 'burn' },
  { effect: 'explosion', tag: 'metal', label: 'Dents and breaks it', powerMult: 1.25 },
  { effect: 'explosion', tag: 'frozen', label: 'Shatters', powerMult: 3, remove: ['cold'] },
  { effect: 'explosion', tag: 'explosive', label: 'Chain explosion', amplifies: true, explode: true },
  // Cold
  { effect: 'cold', tag: 'flammable', label: 'Frosts it' },
  { effect: 'cold', tag: 'wet', label: 'Freezes the surface, slippery', powerMult: 2 },
  { effect: 'cold', tag: 'oil', label: 'Stiffens, no slip', remove: ['oil_slick'] },
  { effect: 'cold', tag: 'metal', label: 'Frosts it' },
];

/** The stagger ladder (§117). Share = how much of max poise the hits have used up. */
export const STAGGER_LADDER: StaggerStepDef[] = [
  { step: 'flinch', share: 0.2, duration: 0.15, speedMult: 0.5, aimKept: true, canAttack: true, pushback: 0, damageTaken: 1, ignoredFromPoise: 120 },
  { step: 'stagger', share: 0.5, duration: 0.5, speedMult: 0, aimKept: true, canAttack: false, pushback: 0.35, damageTaken: 1 },
  { step: 'stumble', share: 0.8, duration: 1.0, speedMult: 0.3, aimKept: false, canAttack: false, pushback: 0.2, damageTaken: 1.1 },
  { step: 'knockdown', share: 1.0, duration: 1.5, speedMult: 0, aimKept: false, canAttack: false, pushback: 0.5, damageTaken: 1.5 },
];

/** Caps (§118) so a big fight or a barrel field cannot overload the game. */
export const EFFECT_LIMITS: EffectLimits = {
  maxFires: 24, maxPuddles: 32, maxClouds: 12,
  maxChainDepth: 4, maxChainTargets: 24,
  poiseRegenDelay: 1.5, poiseRegenRate: 0.5,
  playerPoise: 60,
};
