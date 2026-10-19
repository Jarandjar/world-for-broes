import type { G, InkOpts } from './ink';
import { INK, TAU, ell, cap, rrect, poly, lumpy, stroke, shade, alpha, inked } from './ink';
import type { ZombieC } from '../sim/components';
import type { ZombieKind } from '../data/schema';
import { makeRng, hashStr, type Rng } from '../engine/rng';
import { SpriteCache, blit, type Sprite } from './sprites';
import { blitRim, blobShadow, ppuOf, silhouette } from './characterRenderer';

/**
 * Cartoon zombies (STYLE §2: lumpy potato bodies, mismatched googly eyes, dropped jaws, floppy limbs; no gore).
 * Every zombie is built from cached part sprites and animated with transforms only. Looks are data:
 *   ZOMBIE_KINDS  per-kind body plan (size, palettes, posture, extras)
 *   ZOMBIE_SHIRTS shirt painters, ZOMBIE_HATS hat drawers, ZOMBIE_EXTRAS kind add-ons (keg, cheeks, lumps)
 * `zombieVariant(kind, seed)` picks one look per zombie; see docs/EXTENDING.md "Add a zombie variant".
 */

type Box = [number, number, number, number];
/** Paints inside the torso clip. `w` is the torso half-width. */
export type ShirtPainter = (g: G, rng: Rng, v: ZombieVariant, w: number) => void;
/** Draws in head space (head centre at 0,0, facing +x). */
export type HatDrawer = (g: G, rng: Rng, col: string) => void;
export interface ZombieExtra { layer: 'torso' | 'head'; draw: (g: G, rng: Rng, v: ZombieVariant) => void }

export interface ZombieKindLook {
  scale: number;          // overall size
  skins: string[];        // skin tints to pick from
  shirts: string[];       // ZOMBIE_SHIRTS ids
  cloth: string[];        // shirt colours
  pants: string[];        // trouser colours
  hats: string[];         // ZOMBIE_HATS ids ('none' weights the odds)
  extras: string[];       // ZOMBIE_EXTRAS ids, always drawn for this kind
  hunch: number;          // forward head offset (posture)
  reach: number;          // arm reach multiplier
  gait: number;           // stride multiplier
  sway: number;           // body sway amount (radians)
}

export interface ZombieVariant {
  kind: ZombieKind; skin: string; shirt: string; cloth: string; pants: string; hat: string; hatColor: string;
  eyes: number;   // 0..3: which eye is the big googly one, and how big
  tear: number;   // 0..2 rip pattern
  lean: number;   // radians, permanent body lean
  droop: number;  // -1/1 the side whose arm hangs limp, 0 none
  limp: number;   // 0..0.6 drag on one leg
  wobble: number; // phase offset for head wobble
}

const SKINS = ['#93b37c', '#a2b78c', '#86ad97', '#aab57f', '#90a9ad', '#b4ad8f', '#a196b5'];
const CLOTH = ['#5b6f8f', '#9a5448', '#6f7f4a', '#8f7c58', '#4f6070', '#a8744a', '#6e5d80', '#3f6d6c', '#b59a4c'];
const PANTS = ['#3d4a5f', '#5a4a3a', '#4a4a52', '#2f4050', '#55603f'];
const GOO = '#8fd14f';

export const ZOMBIE_KINDS: Record<ZombieKind, ZombieKindLook> = {
  walker:   { scale: 1.0,  skins: SKINS, shirts: ['tee', 'plaid', 'stripes', 'tank', 'overalls'], cloth: CLOTH, pants: PANTS, hats: ['none', 'none', 'none', 'cap', 'beanie', 'bucket', 'party'], extras: [], hunch: 0.02, reach: 1, gait: 1, sway: 0.08 },
  runner:   { scale: 0.92, skins: SKINS, shirts: ['jersey', 'tee', 'tank'], cloth: ['#c0463c', '#3f74b0', '#e0a53a', '#4f9a5c'], pants: PANTS, hats: ['none', 'none', 'sweatband'], extras: [], hunch: 0.05, reach: 1.1, gait: 1.25, sway: 0.05 },
  brute:    { scale: 1.36, skins: ['#7e9a6c', '#8a9a74', '#76917f'], shirts: ['tank', 'overalls'], cloth: ['#4b4f57', '#6b5844', '#3f4c3f'], pants: PANTS, hats: ['none', 'hardhat'], extras: ['armor_plates', 'shoulder_lump'], hunch: 0.04, reach: 0.95, gait: 0.8, sway: 0.12 },
  sapper:   { scale: 1.0,  skins: SKINS, shirts: ['vest', 'overalls'], cloth: ['#7a6a4a', '#5f6a3f'], pants: PANTS, hats: ['none', 'hardhat', 'beanie'], extras: ['keg'], hunch: 0.03, reach: 1, gait: 1.05, sway: 0.07 },
  spitter:  { scale: 1.0,  skins: ['#9fc07a', '#a9c48a', '#8db885'], shirts: ['tee', 'stripes', 'hoodie'], cloth: CLOTH, pants: PANTS, hats: ['none', 'cap', 'beanie'], extras: ['cheeks'], hunch: 0.02, reach: 0.9, gait: 0.95, sway: 0.09 },
  burrower: { scale: 0.95, skins: ['#a39b78', '#9a9f7a', '#8f8a6e'], shirts: ['overalls', 'plaid'], cloth: ['#7a5f42', '#6a6a4a'], pants: ['#4f4033', '#3f3a30'], hats: ['miner', 'miner', 'none'], extras: ['dirt'], hunch: 0.07, reach: 1.05, gait: 0.9, sway: 0.1 },
  screamer: { scale: 0.95, skins: ['#b9b6a4', '#b3b9b0', '#c0b0b8'], shirts: ['hoodie', 'tee'], cloth: ['#3f3f4f', '#5a3f5f', '#2f4f5f'], pants: PANTS, hats: ['none'], extras: ['megamouth'], hunch: 0.0, reach: 0.8, gait: 1, sway: 0.06 },
  goliath:  { scale: 1.75, skins: ['#6f8a62', '#7a8a6a'], shirts: ['overalls', 'tank'], cloth: ['#4a3f38', '#3f4a52'], pants: ['#2f3540'], hats: ['none', 'hardhat'], extras: ['stitches', 'shoulder_lump', 'armor_plates'], hunch: 0.05, reach: 0.9, gait: 0.7, sway: 0.14 },
};

const ik = (g: G, rng: Rng, path: () => void, fill: string, o: InkOpts = {}): void => inked(g, path, fill, { rng, box: 0.3, grain: 0.5, wear: 0.3, ...o });
const clipped = (g: G, path: () => void, body: () => void): void => { g.save(); g.beginPath(); path(); g.clip(); body(); g.restore(); };

// ---------------------------------------------------------------- shirts (painted inside the torso shape)
export const ZOMBIE_SHIRTS: Record<string, ShirtPainter> = {
  tee: (g, _r, v, w) => { stroke(g, [[0.06, -w * 0.45], [0.13, 0], [0.06, w * 0.45]], 0.03, shade(v.cloth, -0.35), 0.8); },
  plaid: (g, _r, v, w) => {
    for (let y = -w; y <= w; y += 0.07) stroke(g, [[-0.3, y], [0.3, y]], 0.02, shade(v.cloth, -0.4), 0.45);
    for (let x = -0.3; x <= 0.3; x += 0.07) stroke(g, [[x, -w], [x, w]], 0.014, shade(v.cloth, 0.35), 0.4);
  },
  stripes: (g, _r, v, w) => { for (let x = -0.24; x <= 0.24; x += 0.06) stroke(g, [[x, -w], [x + 0.02, w]], 0.022, shade(v.cloth, 0.45), 0.55); },
  jersey: (g, _r, v, w) => {
    stroke(g, [[-0.25, -w * 0.8], [0.25, -w * 0.8]], 0.03, '#f4efe0', 0.8); stroke(g, [[-0.25, w * 0.8], [0.25, w * 0.8]], 0.03, '#f4efe0', 0.8);
    g.fillStyle = alpha('#f4efe0', 0.85); g.font = 'bold 0.16px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.save(); g.rotate(Math.PI / 2); g.fillText(String(v.tear + 7), 0, 0.03); g.restore();
  },
  tank: (g, _r, v, w) => { g.fillStyle = v.skin; for (const s of [-1, 1]) { g.beginPath(); ell(g, 0, s * w * 0.95, 0.12, w * 0.32); g.fill(); } stroke(g, [[0.12, -w * 0.5], [0.16, 0], [0.12, w * 0.5]], 0.02, shade(v.cloth, -0.4), 0.7); },
  vest: (g, _r, v, w) => {
    g.fillStyle = shade(v.cloth, 0.25); g.fillRect(-0.3, -w * 0.25, 0.6, w * 0.5);
    for (const s of [-1, 1]) for (let i = 0; i < 2; i++) { g.fillStyle = shade(v.cloth, -0.25); g.beginPath(); rrect(g, -0.02 + i * 0.1, s * w * 0.62 - 0.035, 0.07, 0.07, 0.015); g.fill(); }
  },
  overalls: (g, _r, v, w) => {
    g.fillStyle = v.pants; g.beginPath(); rrect(g, -0.06, -w * 0.5, 0.2, w, 0.04); g.fill();
    for (const s of [-1, 1]) { stroke(g, [[0.12, s * w * 0.42], [-0.25, s * w * 0.55]], 0.035, v.pants, 1); g.fillStyle = '#d9c27a'; g.beginPath(); g.arc(0.11, s * w * 0.42, 0.018, 0, TAU); g.fill(); }
  },
  hoodie: (g, rng, v, w) => {
    g.fillStyle = shade(v.cloth, -0.2); g.beginPath(); lumpy(g, -0.17, 0, 0.1, w * 0.5, rng, 8, 0.1); g.fill();
    stroke(g, [[0.14, -0.04], [0.02, -0.03]], 0.01, '#e8e0cc', 0.8); stroke(g, [[0.14, 0.04], [0.02, 0.03]], 0.01, '#e8e0cc', 0.8);
  },
};

// ---------------------------------------------------------------- hats (head space)
export const ZOMBIE_HATS: Record<string, HatDrawer> = {
  none: () => {},
  cap: (g, rng, c) => {
    ik(g, rng, () => smoothBrim(g), shade(c, -0.15), { hatch: 0.5, line: 0.016, box: 0.2 });
    ik(g, rng, () => ell(g, -0.03, 0, 0.15, 0.145), c, { hatch: 0.6, line: 0.018, box: 0.2 });
    g.fillStyle = INK; g.beginPath(); g.arc(-0.03, 0, 0.016, 0, TAU); g.fill();
  },
  beanie: (g, rng, c) => {
    ik(g, rng, () => ell(g, -0.04, 0, 0.15, 0.15), c, { hatch: 0.6, line: 0.018, box: 0.2 });
    for (let a = 0; a < TAU; a += 0.5) stroke(g, [[-0.04 + Math.cos(a) * 0.04, Math.sin(a) * 0.04], [-0.04 + Math.cos(a) * 0.13, Math.sin(a) * 0.13]], 0.009, INK, 0.3);
    ik(g, rng, () => ell(g, -0.06, 0.02, 0.04, 0.04), shade(c, 0.4), { hatch: 0, line: 0.014, box: 0.08 });
  },
  bucket: (g, rng, c) => {
    ik(g, rng, () => lumpy(g, -0.02, 0, 0.22, 0.21, rng, 12, 0.05), shade(c, -0.1), { hatch: 0.6, line: 0.018, box: 0.28 });
    ik(g, rng, () => ell(g, -0.02, 0, 0.13, 0.125), c, { hatch: 0.5, line: 0.016, box: 0.18 });
  },
  hardhat: (g, rng) => {
    ik(g, rng, () => ell(g, -0.01, 0, 0.19, 0.17), '#e8b43a', { hatch: 0.4, line: 0.018, box: 0.24, noWash: true });
    stroke(g, [[-0.17, 0], [0.15, 0]], 0.03, '#c9921f', 0.9);
    g.beginPath(); g.arc(-0.06, -0.06, 0.06, 3.4, 4.9); g.strokeStyle = alpha('#fff8de', 0.7); g.lineWidth = 0.014; g.stroke();
  },
  miner: (g, rng) => {
    ik(g, rng, () => ell(g, -0.01, 0, 0.17, 0.16), '#c7a548', { hatch: 0.5, line: 0.018, box: 0.22 });
    ik(g, rng, () => ell(g, 0.15, 0, 0.04, 0.05), '#fff2b0', { hatch: 0, line: 0.014, box: 0.08, noWash: true });
  },
  party: (g, rng, c) => {
    ik(g, rng, () => ell(g, -0.02, 0, 0.1, 0.1), c, { hatch: 0.3, line: 0.016, box: 0.14 });
    for (const r of [0.07, 0.04]) { g.beginPath(); g.arc(-0.02, 0, r, 0, TAU); g.strokeStyle = alpha('#f4efe0', 0.8); g.lineWidth = 0.014; g.stroke(); }
    g.fillStyle = '#f2c94c'; g.beginPath(); g.arc(-0.02, 0, 0.022, 0, TAU); g.fill();
  },
  sweatband: (g, _rng, c) => {
    g.beginPath(); ell(g, -0.01, 0, 0.16, 0.15); g.strokeStyle = INK; g.lineWidth = 0.05; g.stroke();
    g.beginPath(); ell(g, -0.01, 0, 0.16, 0.15); g.strokeStyle = c; g.lineWidth = 0.035; g.stroke();
  },
};
function smoothBrim(g: G): void { poly(g, [[0.04, -0.12], [0.24, -0.1], [0.27, 0], [0.24, 0.1], [0.04, 0.12]]); }

// ---------------------------------------------------------------- kind extras
export const ZOMBIE_EXTRAS: Record<string, ZombieExtra> = {
  shoulder_lump: { layer: 'torso', draw: (g, rng, v) => {
    ik(g, rng, () => lumpy(g, -0.06, -0.2, 0.14, 0.11, makeRng(31), 9, 0.12), shade(v.skin, -0.12), { hatch: 0.5, line: 0.02, box: 0.2 });
    for (let i = 0; i < 3; i++) { g.fillStyle = shade(v.skin, 0.3); g.beginPath(); g.arc(-0.1 + i * 0.05, -0.22 + rng.range(-0.03, 0.03), 0.02, 0, TAU); g.fill(); }   // mushroomy bumps
  } },
  armor_plates: { layer: 'torso', draw: (g, rng) => {
    for (const [x, y, r] of [[0.0, 0.17, 0.2], [-0.13, 0.05, -0.3]] as [number, number, number][]) {
      ik(g, rng, () => rrect(g, x - 0.06, y - 0.05, 0.13, 0.1, 0.02), '#8a8f96', { hatch: 0.4, line: 0.016, box: 0.12, angle: r });
      for (const dx of [-0.04, 0.05]) { g.fillStyle = '#d5d9de'; g.beginPath(); g.arc(x + dx, y, 0.01, 0, TAU); g.fill(); }
    }
    stroke(g, [[0.12, -0.25], [-0.12, 0.25]], 0.02, '#5a4632', 0.9);    // strap
  } },
  keg: { layer: 'torso', draw: (g, rng) => {
    ik(g, rng, () => ell(g, -0.17, 0, 0.12, 0.15), '#9a6236', { hatch: 0.7, line: 0.02, box: 0.2 });
    for (const y of [-0.08, 0.08]) stroke(g, [[-0.28, y], [-0.06, y]], 0.018, '#4a4f55', 0.9);
    stroke(g, [[-0.17, 0], [-0.26, -0.05], [-0.3, -0.02]], 0.012, '#3a2a1a', 1);                       // fuse
    g.fillStyle = '#ffd36a'; g.beginPath(); g.arc(-0.3, -0.02, 0.022, 0, TAU); g.fill();
    g.fillStyle = '#ff8a3a'; g.beginPath(); g.arc(-0.3, -0.02, 0.011, 0, TAU); g.fill();
  } },
  cheeks: { layer: 'head', draw: (g, rng, v) => {
    for (const s of [-1, 1]) ik(g, rng, () => ell(g, 0.04, s * 0.11, 0.06, 0.055), shade(v.skin, 0.12), { hatch: 0.2, line: 0.015, box: 0.08 });
    g.fillStyle = alpha(GOO, 0.9); g.beginPath(); ell(g, 0.2, 0.02, 0.025, 0.035); g.fill();          // cartoon green drool
  } },
  dirt: { layer: 'torso', draw: (g, rng) => {
    for (let i = 0; i < 7; i++) { g.fillStyle = alpha(rng.chance(0.5) ? '#6b5236' : '#8a6c48', 0.85); g.beginPath(); lumpy(g, rng.range(-0.18, 0.12), rng.range(-0.24, 0.24), 0.035, 0.03, rng, 6, 0.3); g.fill(); }
    stroke(g, [[-0.15, 0.1], [-0.05, 0.16], [0.0, 0.12]], 0.01, '#5a4a2a', 0.8);                       // root tendril
  } },
  megamouth: { layer: 'head', draw: (g) => {
    g.fillStyle = '#2a1820'; g.beginPath(); ell(g, 0.15, 0, 0.07, 0.085); g.fill();
    g.strokeStyle = INK; g.lineWidth = 0.016; g.stroke();
    g.fillStyle = '#d06a7a'; g.beginPath(); ell(g, 0.16, 0.02, 0.03, 0.035); g.fill();
    for (let i = 0; i < 3; i++) { const a = -0.6 + i * 0.6; stroke(g, [[0.25 + Math.cos(a) * 0.02, Math.sin(a) * 0.1], [0.31 + Math.cos(a) * 0.03, Math.sin(a) * 0.15]], 0.012, INK, 0.7); }   // scream lines
  } },
  stitches: { layer: 'torso', draw: (g) => {
    stroke(g, [[0.12, -0.18], [-0.02, 0.0], [0.1, 0.2]], 0.012, INK, 0.7);
    for (let i = 0; i < 6; i++) { const u = i / 5, x = 0.12 - 0.14 * Math.min(u * 2, 1) + (u > 0.5 ? (u - 0.5) * 0.24 : 0), y = -0.18 + u * 0.38; stroke(g, [[x - 0.025, y - 0.01], [x + 0.025, y + 0.01]], 0.01, '#e8dcc0', 0.9); }
  } },
};

/** Picks a stable look for one zombie. Pure: same kind + seed gives the same look. */
export function zombieVariant(kind: ZombieKind, seed: number): ZombieVariant {
  const k = ZOMBIE_KINDS[kind] ?? ZOMBIE_KINDS.walker, r = makeRng(seed);
  return {
    kind, skin: r.pick(k.skins), shirt: r.pick(k.shirts), cloth: r.pick(k.cloth), pants: r.pick(k.pants),
    hat: r.pick(k.hats), hatColor: r.pick(CLOTH), eyes: r.int(0, 3), tear: r.int(0, 2),
    lean: r.range(-0.12, 0.12), droop: r.chance(0.4) ? (r.chance(0.5) ? -1 : 1) : 0, limp: r.chance(0.35) ? r.range(0.25, 0.6) : 0, wobble: r.range(0, TAU),
  };
}

// ---------------------------------------------------------------- part drawers (cached)
const TORSO_BOX: Box = [-0.42, -0.42, 0.34, 0.42], HEAD_BOX: Box = [-0.26, -0.26, 0.34, 0.26], ARM_BOX: Box = [-0.08, -0.1, 0.44, 0.1], LEG_BOX: Box = [-0.1, -0.1, 0.34, 0.1];

function drawTorso(g: G, rng: Rng, v: ZombieVariant): void {
  const w = 0.29, path = (): void => lumpy(g, -0.02, 0, 0.2, w, makeRng(hashStr(v.kind)), 11, 0.07);
  ik(g, rng, () => lumpy(g, -0.12, 0, 0.13, w * 0.7, makeRng(7), 8, 0.08), shade(v.cloth, -0.25), { hatch: 0.8, line: 0.02, box: 0.3 });   // hunched back
  ik(g, rng, path, v.cloth, { hatch: 0.6, line: 0.024 });
  clipped(g, path, () => {
    (ZOMBIE_SHIRTS[v.shirt] ?? ZOMBIE_SHIRTS.tee!)(g, rng, v, w);
    // rips showing skin, a sewn patch, and cartoon goo stains (no gore)
    const tr = makeRng(hashStr(`${v.shirt}${v.tear}`));
    for (let i = 0; i < 2 + v.tear; i++) {
      const x = tr.range(-0.15, 0.12), y = tr.range(-w * 0.8, w * 0.8);
      g.fillStyle = v.skin; g.beginPath(); lumpy(g, x, y, 0.04, 0.03, tr, 7, 0.35); g.fill();
      g.strokeStyle = alpha(INK, 0.7); g.lineWidth = 0.009; g.stroke();
    }
    if (v.tear === 1) { const x = tr.range(-0.1, 0.05), y = tr.range(-0.12, 0.12); g.fillStyle = shade(v.cloth, 0.35); g.fillRect(x, y, 0.07, 0.06); for (let i = 0; i < 4; i++) stroke(g, [[x + i * 0.022, y - 0.01], [x + i * 0.022, y + 0.07]], 0.006, INK, 0.6); }
    g.fillStyle = alpha(GOO, 0.55); g.beginPath(); lumpy(g, tr.range(-0.12, 0.1), tr.range(-0.15, 0.15), 0.05, 0.04, tr, 7, 0.3); g.fill();
    g.fillStyle = alpha(INK, 0.18); g.beginPath(); g.ellipse(0.08, 0, 0.1, w * 0.4, 0, 0, TAU); g.fill();           // head's contact shadow
    stroke(g, [[-0.1, -w * 0.75], [-0.18, 0], [-0.1, w * 0.75]], 0.018, shade(v.cloth, 0.5), 0.3);              // rounded back highlight
  });
  // ragged hem: little cloth tatters hanging off the sides
  for (let i = 0; i < 4; i++) { const s = i % 2 ? 1 : -1, x = -0.12 + i * 0.06; ik(g, rng, () => poly(g, [[x - 0.03, s * w * 0.92], [x + 0.03, s * w * 0.92], [x + rng.range(-0.01, 0.02), s * (w + 0.05)]]), v.cloth, { hatch: 0, line: 0.012, box: 0.06 }); }
  for (const id of ZOMBIE_KINDS[v.kind].extras) { const e = ZOMBIE_EXTRAS[id]; if (e?.layer === 'torso') e.draw(g, rng, v); }
}

function drawHead(g: G, rng: Rng, v: ZombieVariant): void {
  const path = (): void => lumpy(g, 0, 0, 0.165, 0.155, makeRng(hashStr(v.skin + v.eyes)), 9, 0.08);
  // floppy ears
  for (const s of [-1, 1]) ik(g, rng, () => ell(g, -0.02, s * 0.155, 0.035, 0.045, s * 0.3), shade(v.skin, -0.05), { hatch: 0, line: 0.015, box: 0.08 });
  ik(g, rng, path, v.skin, { hatch: 0.5, line: 0.024, box: 0.22 });
  clipped(g, path, () => {
    g.fillStyle = alpha(shade(v.skin, -0.5), 0.35); for (let i = 0; i < 4; i++) { g.beginPath(); g.arc(rng.range(-0.12, 0.05), rng.range(-0.1, 0.1), rng.range(0.012, 0.025), 0, TAU); g.fill(); }  // blotches
    for (let i = 0; i < 5; i++) stroke(g, [[-0.12 + i * 0.02, -0.05 + rng.range(-0.04, 0.04)], [-0.15 + i * 0.02, rng.range(-0.08, 0.08)]], 0.008, shade(v.skin, -0.6), 0.5);       // stray hair
  });
  // dropped lower jaw, a couple of goofy teeth
  ik(g, rng, () => ell(g, 0.15, 0, 0.045, 0.075), '#3a2228', { hatch: 0, line: 0.016, box: 0.1, noWash: true });
  g.fillStyle = '#efe6cc'; g.fillRect(0.165, -0.05, 0.022, 0.024); g.fillRect(0.17, 0.022, 0.02, 0.022);
  // mismatched googly eyes
  const big = 0.048 + (v.eyes % 2) * 0.01, small = 0.026, bigSide = v.eyes < 2 ? -1 : 1;
  for (const s of [-1, 1]) {
    const r = s === bigSide ? big : small, ex = 0.085, ey = s * 0.07;
    ik(g, rng, () => ell(g, ex, ey, r, r), '#f4f1e2', { hatch: 0, line: 0.014, box: 0.06, noWash: true });
    g.fillStyle = INK; g.beginPath(); g.arc(ex + r * 0.35, ey + s * r * (v.eyes === 3 ? -0.3 : 0.25), r * 0.45, 0, TAU); g.fill();
  }
  for (const id of ZOMBIE_KINDS[v.kind].extras) { const e = ZOMBIE_EXTRAS[id]; if (e?.layer === 'head') e.draw(g, rng, v); }
}

function drawArm(g: G, rng: Rng, v: ZombieVariant): void {
  const sleeve = v.shirt === 'tank' ? 0.04 : v.tear === 2 ? 0.12 : 0.19;
  ik(g, rng, () => cap(g, 0, 0, 0.3, 0, 0.05, 0.036), v.skin, { hatch: 0.5, line: 0.018, box: 0.2 });
  ik(g, rng, () => poly(g, [[-0.03, -0.065], [sleeve, -0.06], [sleeve - 0.02, -0.02], [sleeve + 0.01, 0.015], [sleeve - 0.015, 0.06], [-0.03, 0.065]]), v.cloth, { hatch: 0.5, line: 0.018, box: 0.15 });
  ik(g, rng, () => ell(g, 0.33, 0, 0.048, 0.044), shade(v.skin, -0.06), { hatch: 0.3, line: 0.018, box: 0.08 });
  for (let i = -1; i <= 1; i++) ik(g, rng, () => cap(g, 0.35, i * 0.024, 0.395, i * 0.03, 0.014, 0.011), v.skin, { hatch: 0, line: 0.011, box: 0.05 });   // stubby fingers
}

function drawLeg(g: G, rng: Rng, v: ZombieVariant): void {
  ik(g, rng, () => cap(g, -0.04, 0, 0.2, 0, 0.065, 0.055), v.pants, { hatch: 0.6, line: 0.018, box: 0.2 });
  stroke(g, [[0.1, -0.05], [0.12, 0.05]], 0.008, shade(v.pants, -0.6), 0.5);
  ik(g, rng, () => ell(g, 0.25, 0, 0.075, 0.05), v.tear === 0 ? '#4a3b30' : v.skin, { hatch: 0.4, line: 0.016, box: 0.1 });   // shoe, or a lost shoe
}

// ---------------------------------------------------------------- renderer
/** Renders cartoon zombies from cached parts; per-frame work is transforms and blits only. */
export class ZombieRenderer {
  private cache = new SpriteCache();
  private looks = new WeakMap<ZombieC, ZombieVariant>();

  /** The look of one zombie (seeded from where it was first drawn, then stable). Exposed for previews and status hooks. */
  variantOf(z: ZombieC, x = 0, y = 0): ZombieVariant {
    let v = this.looks.get(z);
    if (!v) { v = zombieVariant(z.def.kind, hashStr(`${z.def.id}:${Math.round(x * 7)}:${Math.round(y * 7)}`)); this.looks.set(z, v); }
    return v;
  }

  private part(key: string, box: Box, v: ZombieVariant, ppu: number, draw: (g: G, rng: Rng, v: ZombieVariant) => void): Sprite {
    return this.cache.get(key, ppu, box[0], box[1], box[2], box[3], (gg) => draw(gg, makeRng(hashStr(key)), v));
  }

  draw(g: G, z: ZombieC, x: number, y: number, angle: number, dt: number): void {
    z.animPhase += (z.state === 'chase' ? 4.5 : z.state === 'wander' ? 2.0 : z.state === 'attack' ? 6 : 0.5) * dt;
    if (z.hitFlash > 0) z.hitFlash = Math.max(0, z.hitFlash - dt * 4);
    const v = this.variantOf(z, x, y), k = ZOMBIE_KINDS[z.def.kind] ?? ZOMBIE_KINDS.walker, sc = k.scale, ph = z.animPhase + v.wobble;
    const amt = (z.state === 'chase' ? 1.2 : z.state === 'attack' ? 0.6 : z.state === 'wander' ? 0.7 : 0.12) * k.gait;
    const bob = Math.sin(ph * 2) * amt;

    g.save();
    g.translate(x, y);
    blobShadow(g, 0.36 * sc, 0.29 * sc, Math.max(0, bob) * 0.02, 0.04, 0.05);
    g.rotate(angle); g.scale(sc, sc);
    const ppu = ppuOf(g);
    const torso = this.part(`zt|${v.kind}|${v.shirt}|${v.cloth}|${v.skin}|${v.pants}|${v.tear}`, TORSO_BOX, v, ppu, drawTorso);
    const head = this.part(`zh|${v.kind}|${v.skin}|${v.eyes}`, HEAD_BOX, v, ppu, drawHead);
    const arm = this.part(`za|${v.skin}|${v.cloth}|${v.shirt === 'tank' ? 't' : v.tear}`, ARM_BOX, v, ppu, drawArm);
    const leg = this.part(`zl|${v.pants}|${v.skin}|${v.tear === 0 ? 's' : 'b'}`, LEG_BOX, v, ppu, drawLeg);

    // legs: shuffling stride, one leg dragging when limping
    for (const side of [-1, 1]) {
      const drag = side === (v.droop || 1) ? 1 - v.limp : 1;
      const step = Math.sin(ph + (side > 0 ? 0 : Math.PI)) * 0.13 * amt * drag;
      blit(g, leg, -0.04 + step, side * 0.11, side * 0.06 * drag, 1, side);
    }

    // body: lean + sway, squash and stretch on each lurch
    const lunge = z.state === 'attack' ? Math.max(0, Math.sin(z.animPhase * 1.5)) * 0.09 : 0;
    g.save();
    g.translate(lunge, 0);
    g.rotate(v.lean + k.sway * Math.sin(ph) * Math.min(1, amt + 0.3));
    const sx = 1 + 0.05 * bob, sy = 1 - 0.04 * bob + 0.015 * Math.sin(z.animPhase * 1.3);
    g.scale(sx, sy);
    blit(g, torso); blitRim(g, torso);
    if (z.hitFlash > 0) blit(g, silhouette(torso, '#fff3dc'), 0, 0, 0, 1, 1, z.hitFlash * 0.85);

    // arms: classic zombie reach, a limp arm on the droop side, floppy wiggle
    const reach = (z.state === 'attack' ? 1.15 + 0.12 * Math.sin(z.animPhase * 3) : z.state === 'chase' ? 1.05 : 0.88) * k.reach;
    for (const side of [-1, 1]) {
      const limp = side === v.droop;
      const wig = Math.sin(ph * 1.6 + side) * (limp ? 0.18 : 0.08);
      const ang = limp ? side * (0.75 + wig) : side * (0.06 + wig * 0.6) + (z.state === 'attack' ? -side * 0.12 : 0);
      blit(g, arm, 0.02, side * 0.2, ang, limp ? 0.8 : reach, side);
      blitRim(g, arm, 0.02, side * 0.2, ang, limp ? 0.8 : reach, side, 0.6);
    }

    // head: hunched forward, wobbling on a floppy neck
    g.scale(1 / sx, 1 / sy);
    g.translate(0.11 + k.hunch + (z.state === 'attack' ? 0.03 : 0), 0);
    g.rotate(0.2 * Math.sin(ph * 0.7) + 0.06 * Math.sin(ph * 2.3));
    const tilt = 1 + 0.04 * Math.sin(ph * 0.9);
    blit(g, head, 0, 0, 0, tilt, 1 / tilt); blitRim(g, head, 0, 0, 0, tilt, 1 / tilt);
    if (v.hat !== 'none') {
      const hat = this.part(`zhat|${v.hat}|${v.hatColor}`, HEAD_BOX, v, ppu, (gg, rng) => (ZOMBIE_HATS[v.hat] ?? ZOMBIE_HATS.none!)(gg, rng, v.hatColor));
      blit(g, hat, -0.01, 0, 0.05 * Math.sin(ph * 1.1)); blitRim(g, hat, -0.01, 0, 0.05 * Math.sin(ph * 1.1));
    }
    if (z.hitFlash > 0) blit(g, silhouette(head, '#fff3dc'), 0, 0, 0, tilt, 1 / tilt, z.hitFlash * 0.85);
    g.restore();
    g.restore();

    // Overhead health bar if damaged
    if (z.hp < z.def.maxHp) {
      g.save(); g.translate(x, y);
      const barW = 0.44 * sc, barH = 0.06, barY = -0.42 * sc;
      g.fillStyle = 'rgba(14,16,20,0.85)'; g.strokeStyle = INK; g.lineWidth = 0.012;
      g.beginPath(); rrect(g, -barW / 2, barY, barW, barH, 0.02); g.fill(); g.stroke();
      const hpRatio = Math.max(0, z.hp / z.def.maxHp);
      g.fillStyle = hpRatio > 0.4 ? '#c4473d' : '#e03a2a';
      g.beginPath(); rrect(g, -barW / 2 + 0.01, barY + 0.01, (barW - 0.02) * hpRatio, barH - 0.02, 0.015); g.fill();
      g.restore();
    }
  }
}
