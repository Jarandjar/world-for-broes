import type { G } from './ink';
import { INK, TAU, ell, cap, poly, lumpy, smooth, stroke, shade, alpha, inked } from './ink';
import type { AnimalC } from '../sim/components';
import type { AnimalKind } from '../data/schema';
import { makeRng, hashStr, type Rng } from '../engine/rng';
import { SpriteCache, blit, type Sprite } from './sprites';
import { blitRim, blobShadow, ppuOf } from './characterRenderer';

/**
 * Creature rigs shared by animals and pets. A creature is a few cached part sprites (body, head, one leg,
 * tail, optional wing) placed on a rig and animated with transforms by an ANIM_PRESETS entry. Looks are data:
 *   COAT_PAINTERS  fur / wool / feather / bristle textures, keyed by id
 *   ANIM_PRESETS   gait and personality dials, keyed by id
 *   ANIMAL_LOOKS   one CreatureLook per AnimalKind (pets live in PET_LOOKS, petRenderer.ts)
 * Recipe: docs/EXTENDING.md "Add an animal or pet look".
 */

export type Box = [number, number, number, number];
/** A cached part. Drawn facing +x; pivot at (0,0): legs at the hip, head at the neck, tail at its root (tail points -x). */
export interface RigPart { key: string; box: Box; draw: (g: G, rng: Rng) => void }
/** Paints texture inside a clipped coat shape. */
export type CoatPainter = (g: G, rng: Rng, col: string, box: number) => void;

export interface AnimPreset {
  stride: number;    // leg swing (units)
  bob: number;       // body squash-and-stretch per step
  hop: number;       // whole-body hop (rabbits)
  waddle: number;    // side-to-side body roll (radians)
  headBob: number;   // head pump while moving
  graze: number;     // how far the head dips while grazing
  peck: number;      // quick forward pecks while grazing (birds)
  look: number;      // idle head turns (radians)
  tailWag: number;   // tail swing (radians)
  flap: number;      // wing flap when fleeing (radians)
}

export interface CreatureLook {
  anim: string;                                  // ANIM_PRESETS id
  shadow: [number, number];
  body: RigPart;
  head: { at: [number, number]; part: RigPart };
  legs?: { at: [number, number][]; part: RigPart };
  tail?: { at: [number, number]; part: RigPart };
  wings?: { at: [number, number]; part: RigPart };   // mirrored to both sides
}

export interface RigState { phase: number; tail: number; move: number; graze: boolean; alarm: boolean; sit?: boolean }

export const ANIM_PRESETS: Record<string, AnimPreset> = {
  plod:   { stride: 0.07, bob: 0.03, hop: 0, waddle: 0.03, headBob: 0.03, graze: 0.07, peck: 0, look: 0.3, tailWag: 0.45, flap: 0 },
  trot:   { stride: 0.08, bob: 0.045, hop: 0, waddle: 0.04, headBob: 0.025, graze: 0.05, peck: 0, look: 0.4, tailWag: 0.55, flap: 0 },
  prance: { stride: 0.11, bob: 0.04, hop: 0, waddle: 0.02, headBob: 0.02, graze: 0.08, peck: 0, look: 0.5, tailWag: 0.3, flap: 0 },
  hop:    { stride: 0.0, bob: 0.0, hop: 0.12, waddle: 0, headBob: 0.0, graze: 0.03, peck: 0, look: 0.5, tailWag: 0.2, flap: 0 },
  peck:   { stride: 0.05, bob: 0.03, hop: 0, waddle: 0.12, headBob: 0.04, graze: 0.0, peck: 0.06, look: 0.6, tailWag: 0.25, flap: 0.7 },
  waddle: { stride: 0.06, bob: 0.03, hop: 0, waddle: 0.16, headBob: 0.05, graze: 0.04, peck: 0.03, look: 0.45, tailWag: 0.3, flap: 0.9 },
  pup:    { stride: 0.09, bob: 0.05, hop: 0, waddle: 0.05, headBob: 0.025, graze: 0.03, peck: 0, look: 0.45, tailWag: 0.7, flap: 0 },
};

// ---------------------------------------------------------------- coat painters
const strokes = (g: G, rng: Rng, n: number, box: number, len: number, w: number, cols: string[], a: number, dir = Math.PI): void => {
  for (let i = 0; i < n; i++) {
    const x = rng.range(-box, box), y = rng.range(-box, box), d = dir + rng.range(-0.35, 0.35), l = len * rng.range(0.6, 1.2);
    stroke(g, [[x, y], [x + Math.cos(d) * l * 0.5, y + Math.sin(d) * l * 0.5 + l * 0.08], [x + Math.cos(d) * l, y + Math.sin(d) * l]], w, rng.pick(cols), a);
  }
};
export const COAT_PAINTERS: Record<string, CoatPainter> = {
  none: () => {},
  /** Short fur lying back along the body. */
  fur: (g, rng, col, box) => strokes(g, rng, Math.round(110 * box * box / 0.09), box, 0.05, 0.009, [shade(col, -0.45), shade(col, 0.4)], 0.42),
  /** Longer, softer fur in tufts (pets, rabbits). */
  soft: (g, rng, col, box) => { strokes(g, rng, Math.round(80 * box * box / 0.09), box, 0.065, 0.012, [shade(col, -0.35)], 0.35); strokes(g, rng, Math.round(50 * box * box / 0.09), box, 0.05, 0.01, [shade(col, 0.5)], 0.45); },
  /** Curly wool: many little arcs. */
  wool: (g, rng, col, box) => {
    for (let i = 0, n = Math.round(90 * box * box / 0.09); i < n; i++) {
      const x = rng.range(-box, box), y = rng.range(-box, box), r = rng.range(0.014, 0.026), a0 = rng.range(0, TAU);
      g.beginPath(); g.arc(x, y, r, a0, a0 + 3.6); g.strokeStyle = alpha(rng.chance(0.6) ? shade(col, -0.35) : shade(col, 0.6), 0.55); g.lineWidth = 0.009; g.stroke();
    }
  },
  /** Sparse bristles and freckles (pigs). */
  bristle: (g, rng, col, box) => {
    for (let i = 0, n = Math.round(40 * box * box / 0.09); i < n; i++) { g.fillStyle = alpha(shade(col, -0.4), 0.35); g.beginPath(); g.arc(rng.range(-box, box), rng.range(-box, box), rng.range(0.004, 0.009), 0, TAU); g.fill(); }
    strokes(g, rng, Math.round(30 * box * box / 0.09), box, 0.025, 0.006, [shade(col, 0.55)], 0.5);
  },
  /** Overlapping feather scallops in rows running back along the body. */
  feather: (g, rng, col, box) => {
    for (let x = box; x > -box; x -= 0.045) for (let y = -box; y < box; y += 0.05) {
      const yy = y + (Math.round(x / 0.045) % 2 ? 0.025 : 0) + rng.range(-0.005, 0.005);
      g.beginPath(); g.arc(x, yy, 0.03, Math.PI * 0.55, Math.PI * 1.45); g.strokeStyle = alpha(shade(col, -0.4), 0.45); g.lineWidth = 0.008; g.stroke();
      g.beginPath(); g.arc(x + 0.006, yy, 0.022, Math.PI * 0.65, Math.PI * 1.35); g.strokeStyle = alpha(shade(col, 0.6), 0.5); g.lineWidth = 0.006; g.stroke();
    }
  },
  /** Smooth hide with a soft sheen (cows). */
  hide: (g, rng, col, box) => strokes(g, rng, Math.round(45 * box * box / 0.09), box, 0.06, 0.008, [shade(col, -0.25), shade(col, 0.5)], 0.3),
};

/** Fill `path` with an inked, coat-textured, outlined shape. */
export function coat(g: G, rng: Rng, path: () => void, col: string, painter = 'fur', o: { line?: number; box?: number; hatch?: number } = {}): void {
  const box = o.box ?? 0.3;
  inked(g, path, col, { rng, box, hatch: o.hatch ?? 0.25, noOutline: true, grain: 0.35, wear: 0.1 });
  g.save(); g.beginPath(); path(); g.clip(); (COAT_PAINTERS[painter] ?? COAT_PAINTERS.none!)(g, rng, col, box); g.restore();
  g.save(); g.beginPath(); path(); g.lineJoin = 'round'; g.strokeStyle = INK; g.lineWidth = o.line ?? 0.02; g.stroke(); g.restore();
}
/** Shiny cartoon eye: ink dot + catchlight. */
export function eye(g: G, x: number, y: number, r: number): void {
  g.fillStyle = INK; g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill();
  g.fillStyle = '#fffaf0'; g.beginPath(); g.arc(x + r * 0.3, y - r * 0.35, r * 0.38, 0, TAU); g.fill();
}
/** Plain inked fill (no coat texture) for hooves, beaks, noses. */
export const plain = (g: G, rng: Rng, path: () => void, col: string, line = 0.014): void => inked(g, path, col, { rng, box: 0.12, hatch: 0, line, grain: 0.2, wear: 0 });

const part = (key: string, box: Box, draw: (g: G, rng: Rng) => void): RigPart => ({ key, box, draw });

// ---------------------------------------------------------------- rig driver
const sprite = (cache: SpriteCache, p: RigPart, ppu: number): Sprite => cache.get(p.key, ppu, p.box[0], p.box[1], p.box[2], p.box[3], (gg) => p.draw(gg, makeRng(hashStr(p.key))));

/** Draws a creature rig at the current origin facing +x. Only transforms and blits per frame. */
export function drawRig(g: G, cache: SpriteCache, look: CreatureLook, st: RigState): void {
  const a = ANIM_PRESETS[look.anim] ?? ANIM_PRESETS.trot!, ppu = ppuOf(g), mv = st.move, s = Math.sin(st.phase);
  const hop = a.hop * Math.abs(Math.sin(st.phase * 0.75)) * mv;
  blobShadow(g, look.shadow[0], look.shadow[1], hop);
  g.save();
  g.scale(1 + hop * 1.2, 1 + hop * 1.2);                       // hops read as "toward the camera"
  g.rotate(a.waddle * s * Math.max(mv, 0.15));

  if (look.legs) {
    const leg = sprite(cache, look.legs.part, ppu), two = look.legs.at.length <= 2;
    for (const [lx, ly] of look.legs.at) {
      const pair = two ? (ly > 0 ? 1 : -1) : (ly > 0) === (lx > 0) ? 1 : -1;      // diagonal gait for quadrupeds
      const step = st.sit ? -0.03 : pair * s * a.stride * mv - (a.hop ? Math.sin(st.phase * 0.75) * 0.06 * mv * (lx < 0 ? 1 : -1) : 0);
      blit(g, leg, lx + step, ly, 0, 1, ly < 0 ? -1 : 1);
    }
  }
  if (look.tail) {
    const tail = sprite(cache, look.tail.part, ppu), wag = Math.sin(st.tail) * a.tailWag * (st.sit ? 0.6 : 1);
    blit(g, tail, look.tail.at[0], look.tail.at[1], wag); blitRim(g, tail, look.tail.at[0], look.tail.at[1], wag, 1, 1, 0.7);
  }
  // body: squash and stretch per step, slow breathing at rest, compressed when sitting
  const b = Math.sin(st.phase * 2) * a.bob * mv, breath = 0.018 * Math.sin(st.phase * 1.7) * (1 - mv);
  const bx = (st.sit ? 0.86 : 1) + b, by = (st.sit ? 1.06 : 1) - b * 0.7 + breath;
  const body = sprite(cache, look.body, ppu);
  blit(g, body, 0, 0, 0, bx, by); blitRim(g, body, 0, 0, 0, bx, by);
  if (look.wings) {
    const wing = sprite(cache, look.wings.part, ppu), flap = st.alarm ? Math.sin(st.phase * 3) * a.flap : 0.04 * Math.sin(st.phase);
    for (const side of [-1, 1]) blit(g, wing, look.wings.at[0], side * look.wings.at[1], side * Math.max(0, flap), 1, side);
  }
  // head: pumps while moving, dips and pecks while grazing, looks around at rest
  const graze = st.graze ? -a.graze + Math.sin(st.phase * 2) * a.graze * 0.3 + a.peck * Math.max(0, Math.sin(st.phase * 3)) : 0;
  const pump = a.headBob * Math.sin(st.phase * 2) * mv;
  const turn = mv > 0.3 ? 0.06 * s : a.look * Math.sin(st.phase * 0.45) * Math.sin(st.phase * 0.21 + 1.1) * (st.graze ? 0.4 : 1);
  const head = sprite(cache, look.head.part, ppu), hx = look.head.at[0] * (st.sit ? 0.88 : 1) + graze + pump, hs = st.graze ? 0.94 : 1;
  blit(g, head, hx, look.head.at[1], turn, hs, hs); blitRim(g, head, hx, look.head.at[1], turn, hs, hs);
  g.restore();
}

// ---------------------------------------------------------------- animal looks
const quadLegs = (fx: number, bx: number, y: number): [number, number][] => [[bx, -y], [bx, y], [fx, -y], [fx, y]];

const COW = '#f3efe6', COW_PATCH = '#3a3640', PIG = '#f2a7b8', DEER = '#b8743a', SHEEP = '#f4f0e6', SHEEP_FACE = '#3a3540', RABBIT = '#a89c8c', HEN = '#fbf4e2', GOOSE = '#f7f7f2';

export const ANIMAL_LOOKS: Record<AnimalKind, CreatureLook> = {
  cow: {
    anim: 'plod', shadow: [0.5, 0.36],
    legs: { at: quadLegs(0.18, -0.22, 0.17), part: part('cow:leg', [-0.1, -0.08, 0.12, 0.08], (g, r) => { coat(g, r, () => cap(g, -0.04, 0, 0.05, 0, 0.045, 0.038), COW, 'hide', { box: 0.1, line: 0.016 }); plain(g, r, () => ell(g, 0.065, 0, 0.024, 0.034), '#2a2630'); }) },
    tail: { at: [-0.33, 0], part: part('cow:tail', [-0.3, -0.08, 0.04, 0.08], (g, r) => { coat(g, r, () => cap(g, 0, 0, -0.2, 0, 0.022, 0.012), COW, 'none', { box: 0.2, line: 0.012 }); coat(g, r, () => lumpy(g, -0.23, 0, 0.045, 0.035, makeRng(21), 7, 0.25), COW_PATCH, 'soft', { box: 0.06, line: 0.012 }); }) },
    body: part('cow:body', [-0.42, -0.3, 0.42, 0.3], (g, r) => {
      const body = (): void => lumpy(g, 0, 0, 0.36, 0.245, makeRng(3), 12, 0.035);
      coat(g, r, body, COW, 'hide', { box: 0.4, line: 0.022 });
      g.save(); g.beginPath(); body(); g.clip();
      for (const [x, y, rx, ry] of [[-0.14, -0.1, 0.13, 0.1], [0.12, 0.1, 0.11, 0.08], [-0.2, 0.12, 0.08, 0.07], [0.15, -0.16, 0.06, 0.05]]) { g.fillStyle = COW_PATCH; g.beginPath(); lumpy(g, x!, y!, rx!, ry!, r, 9, 0.2); g.fill(); }
      COAT_PAINTERS.hide!(g, r, COW_PATCH, 0.4);
      stroke(g, [[-0.3, 0], [0.28, 0]], 0.03, alpha('#ffffff', 0.25), 1);                         // spine sheen
      for (const s of [-1, 1]) stroke(g, [[-0.22, s * 0.12], [-0.17, s * 0.16]], 0.01, INK, 0.4);  // hip bones
      g.restore();
    }),
    head: { at: [0.3, 0], part: part('cow:head', [-0.08, -0.26, 0.32, 0.26], (g, r) => {
      plain(g, r, () => ell(g, 0.0, 0, 0.04, 0.16), '#7a4a26');                                      // leather collar
      for (const s of [-1, 1]) { coat(g, r, () => ell(g, 0.06, s * 0.15, 0.05, 0.03, s * 0.5), COW, 'hide', { box: 0.08, line: 0.014 }); plain(g, r, () => cap(g, 0.1, s * 0.09, 0.08, s * 0.19, 0.022, 0.012), '#e8dcc0'); }
      coat(g, r, () => lumpy(g, 0.1, 0, 0.13, 0.12, makeRng(22), 9, 0.05), COW, 'hide', { box: 0.16, line: 0.02 });
      g.fillStyle = COW_PATCH; g.beginPath(); lumpy(g, 0.07, -0.05, 0.05, 0.045, r, 7, 0.2); g.fill();
      coat(g, r, () => ell(g, 0.2, 0, 0.065, 0.09), '#f4a8b5', 'none', { box: 0.1, line: 0.016 });
      for (const s of [-1, 1]) { g.fillStyle = '#8a3048'; g.beginPath(); ell(g, 0.235, s * 0.035, 0.016, 0.012); g.fill(); eye(g, 0.1, s * 0.075, 0.02); }
      plain(g, r, () => ell(g, 0.01, 0, 0.04, 0.036), '#f2b630', 0.012);                             // cowbell
    }) },
  },
  sheep: {
    anim: 'trot', shadow: [0.4, 0.3],
    legs: { at: quadLegs(0.12, -0.16, 0.12), part: part('sheep:leg', [-0.06, -0.05, 0.1, 0.05], (g, r) => plain(g, r, () => cap(g, 0, 0, 0.06, 0, 0.026, 0.022), SHEEP_FACE)) },
    tail: { at: [-0.27, 0], part: part('sheep:tail', [-0.1, -0.07, 0.04, 0.07], (g, r) => coat(g, r, () => lumpy(g, -0.04, 0, 0.05, 0.045, makeRng(23), 7, 0.2), SHEEP, 'wool', { box: 0.07, line: 0.014 })) },
    body: part('sheep:body', [-0.36, -0.3, 0.36, 0.3], (g, r) => {
      const pts: [number, number][] = []; for (let i = 0; i < 14; i++) { const a = (i / 14) * TAU, k = i % 2 ? 1.08 : 0.94; pts.push([Math.cos(a) * 0.27 * k, Math.sin(a) * 0.205 * k]); }
      coat(g, r, () => smooth(g, pts), SHEEP, 'wool', { box: 0.32, line: 0.02 });
      g.fillStyle = alpha(INK, 0.12); g.beginPath(); g.ellipse(0.12, 0, 0.08, 0.12, 0, 0, TAU); g.fill();
    }),
    head: { at: [0.22, 0], part: part('sheep:head', [-0.06, -0.2, 0.2, 0.2], (g, r) => {
      for (const s of [-1, 1]) coat(g, r, () => cap(g, 0.03, s * 0.05, -0.0, s * 0.14, 0.026, 0.018), SHEEP_FACE, 'soft', { box: 0.08, line: 0.012 });
      coat(g, r, () => ell(g, 0.07, 0, 0.09, 0.07), SHEEP_FACE, 'soft', { box: 0.1, line: 0.018 });
      coat(g, r, () => lumpy(g, 0.03, 0, 0.055, 0.055, makeRng(24), 8, 0.15), SHEEP, 'wool', { box: 0.07, line: 0.012 });
      for (const s of [-1, 1]) { g.fillStyle = '#fffaf0'; g.beginPath(); g.arc(0.1, s * 0.04, 0.018, 0, TAU); g.fill(); eye(g, 0.105, s * 0.04, 0.01); }
    }) },
  },
  pig: {
    anim: 'trot', shadow: [0.4, 0.3],
    legs: { at: quadLegs(0.14, -0.18, 0.14), part: part('pig:leg', [-0.06, -0.06, 0.1, 0.06], (g, r) => { coat(g, r, () => cap(g, 0, 0, 0.05, 0, 0.036, 0.03), PIG, 'none', { box: 0.06, line: 0.014 }); plain(g, r, () => ell(g, 0.06, 0, 0.016, 0.024), '#b86a7a'); }) },
    tail: { at: [-0.29, 0], part: part('pig:tail', [-0.14, -0.08, 0.03, 0.08], (g) => {
      g.beginPath(); for (let i = 0; i <= 24; i++) { const u = i / 24, a = u * TAU * 1.6, rr = 0.04 * (1 - u * 0.5); const px = -0.07 + Math.cos(a) * rr, py = Math.sin(a) * rr; if (i) g.lineTo(px, py); else g.moveTo(0, 0); }
      g.strokeStyle = INK; g.lineWidth = 0.026; g.lineCap = 'round'; g.stroke(); g.strokeStyle = shade(PIG, -0.1); g.lineWidth = 0.014; g.stroke();
    }) },
    body: part('pig:body', [-0.36, -0.3, 0.36, 0.3], (g, r) => {
      coat(g, r, () => lumpy(g, 0, 0, 0.3, 0.225, makeRng(5), 12, 0.03), PIG, 'bristle', { box: 0.34, line: 0.022 });
      g.fillStyle = alpha('#7a4a3a', 0.35); g.beginPath(); lumpy(g, -0.12, 0.08, 0.07, 0.05, r, 7, 0.3); g.fill();   // a little mud
    }),
    head: { at: [0.24, 0], part: part('pig:head', [-0.08, -0.2, 0.24, 0.2], (g, r) => {
      coat(g, r, () => ell(g, 0.04, 0, 0.12, 0.11), PIG, 'bristle', { box: 0.14, line: 0.02 });
      for (const s of [-1, 1]) coat(g, r, () => poly(g, [[0.0, s * 0.06], [0.13, s * 0.15], [-0.02, s * 0.14]]), shade(PIG, -0.12), 'none', { box: 0.12, line: 0.014 });
      plain(g, r, () => ell(g, 0.16, 0, 0.045, 0.065), shade(PIG, -0.1), 0.016);
      for (const s of [-1, 1]) { g.fillStyle = '#8a3858'; g.beginPath(); ell(g, 0.17, s * 0.022, 0.011, 0.013); g.fill(); eye(g, 0.07, s * 0.06, 0.016); }
    }) },
  },
  chicken: {
    anim: 'peck', shadow: [0.2, 0.16],
    legs: { at: [[0.0, -0.05], [0.0, 0.05]], part: part('chicken:leg', [-0.03, -0.05, 0.12, 0.05], (g) => { for (const a of [-0.5, 0, 0.5]) stroke(g, [[0.03, 0], [0.03 + Math.cos(a) * 0.06, Math.sin(a) * 0.05]], 0.012, '#e8962a', 1); stroke(g, [[0, 0], [0.03, 0]], 0.014, '#e8962a', 1); }) },
    tail: { at: [-0.13, 0], part: part('chicken:tail', [-0.16, -0.08, 0.02, 0.08], (g, r) => { for (const a of [-0.45, 0, 0.45]) coat(g, r, () => cap(g, 0, 0, -0.12 * Math.cos(a), Math.sin(a) * 0.08, 0.025, 0.018), shade(HEN, a ? -0.05 : 0.0), 'feather', { box: 0.1, line: 0.012 }); }) },
    body: part('chicken:body', [-0.2, -0.17, 0.2, 0.17], (g, r) => coat(g, r, () => lumpy(g, 0, 0, 0.16, 0.13, makeRng(9), 9, 0.04), HEN, 'feather', { box: 0.18, line: 0.018 })),
    wings: { at: [-0.02, 0.05], part: part('chicken:wing', [-0.12, -0.04, 0.08, 0.1], (g, r) => coat(g, r, () => ell(g, -0.02, 0.04, 0.08, 0.04, -0.1), shade(HEN, -0.04), 'feather', { box: 0.1, line: 0.014 })) },
    head: { at: [0.12, 0], part: part('chicken:head', [-0.06, -0.1, 0.18, 0.1], (g, r) => {
      coat(g, r, () => ell(g, 0.03, 0, 0.065, 0.058), HEN, 'feather', { box: 0.08, line: 0.016 });
      plain(g, r, () => { ell(g, 0.03, 0, 0.04, 0.02); ell(g, 0.0, 0, 0.03, 0.017); ell(g, 0.06, 0, 0.025, 0.015); }, '#e0443a', 0.01);
      plain(g, r, () => poly(g, [[0.08, -0.022], [0.14, 0], [0.08, 0.022]]), '#f0a830', 0.01);
      plain(g, r, () => ell(g, 0.085, 0.0, 0.012, 0.012), '#e0443a', 0.008);
      for (const s of [-1, 1]) eye(g, 0.05, s * 0.035, 0.011);
    }) },
  },
  deer: {
    anim: 'prance', shadow: [0.44, 0.3],
    legs: { at: quadLegs(0.18, -0.2, 0.12), part: part('deer:leg', [-0.04, -0.04, 0.12, 0.04], (g, r) => { coat(g, r, () => cap(g, 0, 0, 0.07, 0, 0.024, 0.018), shade(DEER, -0.15), 'none', { box: 0.06, line: 0.012 }); plain(g, r, () => ell(g, 0.08, 0, 0.012, 0.016), '#2a2220', 0.008); }) },
    tail: { at: [-0.3, 0], part: part('deer:tail', [-0.1, -0.06, 0.03, 0.06], (g, r) => coat(g, r, () => ell(g, -0.035, 0, 0.045, 0.035), '#fbf7ee', 'soft', { box: 0.06, line: 0.012 })) },
    body: part('deer:body', [-0.38, -0.24, 0.38, 0.24], (g, r) => {
      const body = (): void => lumpy(g, 0, 0, 0.32, 0.175, makeRng(11), 12, 0.03);
      coat(g, r, body, DEER, 'fur', { box: 0.36, line: 0.02 });
      g.save(); g.beginPath(); body(); g.clip();
      g.fillStyle = alpha('#f6e3c0', 0.45); for (const s of [-1, 1]) { g.beginPath(); g.ellipse(0.0, s * 0.17, 0.26, 0.05, 0, 0, TAU); g.fill(); }   // pale flanks
      for (let i = 0; i < 9; i++) { g.fillStyle = alpha('#fbf2dc', 0.8); g.beginPath(); g.arc(r.range(-0.22, 0.15), r.range(-0.08, 0.08), r.range(0.01, 0.017), 0, TAU); g.fill(); }  // fawn spots
      g.restore();
    }),
    head: { at: [0.28, 0], part: part('deer:head', [-0.06, -0.26, 0.24, 0.26], (g, r) => {
      for (const s of [-1, 1]) {
        g.strokeStyle = '#5a3a1a'; g.lineWidth = 0.022; g.lineCap = 'round'; g.beginPath(); g.moveTo(0.02, s * 0.04); g.lineTo(0.08, s * 0.16); g.lineTo(0.14, s * 0.15); g.moveTo(0.06, s * 0.12); g.lineTo(0.06, s * 0.23); g.stroke();
        g.strokeStyle = '#c69a62'; g.lineWidth = 0.01; g.stroke();
        coat(g, r, () => ell(g, -0.01, s * 0.1, 0.05, 0.025, s * 0.9), DEER, 'soft', { box: 0.07, line: 0.012 });
      }
      coat(g, r, () => smooth(g, [[0.12, 0], [0.06, -0.06], [-0.04, -0.07], [-0.06, 0], [-0.04, 0.07], [0.06, 0.06]]), DEER, 'fur', { box: 0.12, line: 0.018 });
      plain(g, r, () => ell(g, 0.12, 0, 0.022, 0.025), '#2a2220', 0.01);
      for (const s of [-1, 1]) eye(g, 0.04, s * 0.045, 0.014);
    }) },
  },
  rabbit: {
    anim: 'hop', shadow: [0.2, 0.16],
    legs: { at: [[-0.08, -0.08], [-0.08, 0.08], [0.08, -0.05], [0.08, 0.05]], part: part('rabbit:leg', [-0.06, -0.04, 0.08, 0.04], (g, r) => coat(g, r, () => ell(g, 0.01, 0, 0.05, 0.026), shade(RABBIT, 0.1), 'soft', { box: 0.06, line: 0.012 })) },
    tail: { at: [-0.14, 0], part: part('rabbit:tail', [-0.08, -0.06, 0.03, 0.06], (g, r) => coat(g, r, () => lumpy(g, -0.025, 0, 0.042, 0.04, makeRng(25), 8, 0.15), '#fbf8f2', 'soft', { box: 0.06, line: 0.012 })) },
    body: part('rabbit:body', [-0.2, -0.16, 0.2, 0.16], (g, r) => coat(g, r, () => lumpy(g, 0, 0, 0.155, 0.12, makeRng(13), 9, 0.05), RABBIT, 'soft', { box: 0.18, line: 0.018 })),
    head: { at: [0.13, 0], part: part('rabbit:head', [-0.16, -0.2, 0.12, 0.2], (g, r) => {
      for (const s of [-1, 1]) { coat(g, r, () => cap(g, 0.0, s * 0.03, -0.12, s * 0.12, 0.03, 0.022), RABBIT, 'soft', { box: 0.12, line: 0.014 }); g.fillStyle = alpha('#f4a6c0', 0.85); g.beginPath(); cap(g, -0.01, s * 0.035, -0.105, s * 0.11, 0.015, 0.011); g.fill(); }
      coat(g, r, () => ell(g, 0.02, 0, 0.068, 0.062), RABBIT, 'soft', { box: 0.08, line: 0.016 });
      plain(g, r, () => ell(g, 0.085, 0, 0.014, 0.013), '#e86a9a', 0.008);
      for (const s of [-1, 1]) { eye(g, 0.04, s * 0.04, 0.012); stroke(g, [[0.08, s * 0.01], [0.12, s * 0.05]], 0.004, INK, 0.5); }   // whiskers
    }) },
  },
  goose: {
    anim: 'waddle', shadow: [0.26, 0.2],
    legs: { at: [[0.0, -0.07], [0.0, 0.07]], part: part('goose:leg', [-0.02, -0.05, 0.12, 0.05], (g, r) => plain(g, r, () => poly(g, [[0, -0.01], [0.09, -0.04], [0.1, 0], [0.09, 0.04], [0, 0.01]]), '#f08a2a', 0.01)) },
    tail: { at: [-0.19, 0], part: part('goose:tail', [-0.1, -0.06, 0.02, 0.06], (g, r) => coat(g, r, () => poly(g, [[0, -0.045], [-0.08, 0], [0, 0.045]]), shade(GOOSE, -0.08), 'feather', { box: 0.06, line: 0.012 })) },
    body: part('goose:body', [-0.25, -0.2, 0.25, 0.2], (g, r) => coat(g, r, () => lumpy(g, 0, 0, 0.21, 0.155, makeRng(17), 10, 0.03), GOOSE, 'feather', { box: 0.22, line: 0.018 })),
    wings: { at: [-0.03, 0.06], part: part('goose:wing', [-0.18, -0.04, 0.1, 0.13], (g, r) => coat(g, r, () => ell(g, -0.03, 0.05, 0.12, 0.05, -0.08), shade(GOOSE, -0.06), 'feather', { box: 0.14, line: 0.014 })) },
    head: { at: [0.1, 0], part: part('goose:head', [-0.04, -0.08, 0.3, 0.08], (g, r) => {
      coat(g, r, () => cap(g, 0, 0, 0.17, 0, 0.055, 0.045), GOOSE, 'feather', { box: 0.18, line: 0.016 });
      plain(g, r, () => poly(g, [[0.19, -0.028], [0.28, 0], [0.19, 0.028]]), '#ef7a24', 0.012);
      for (const s of [-1, 1]) eye(g, 0.16, s * 0.03, 0.012);
    }) },
  },
};

/** Ink Hollow renderer for livestock and wildlife: cached parts on a rig, coat texture, rim light and lively idles. */
export class AnimalRenderer {
  private cache = new SpriteCache();

  draw(g: G, animal: AnimalC, x: number, y: number, angle: number, dt: number): void {
    const isMoving = animal.state === 'flee' || animal.state === 'wander';
    const moveSpeed = animal.state === 'flee' ? 8.0 : 3.0;
    animal.animPhase += dt * (isMoving ? moveSpeed : 1.2);
    animal.tailWag = (animal.tailWag ?? 0) + dt * (animal.state === 'flee' ? 14.0 : 4.0);
    const look = ANIMAL_LOOKS[animal.kind];

    g.save();
    g.translate(x, y);
    g.rotate(angle);
    if (look) drawRig(g, this.cache, look, { phase: animal.animPhase, tail: animal.tailWag, move: animal.state === 'flee' ? 1 : animal.state === 'wander' ? 0.7 : 0, graze: animal.state === 'graze' || animal.eating, alarm: animal.state === 'flee' });
    g.restore();

    // Name badge for tamed livestock
    if (animal.tamed) {
      g.save();
      g.font = '0.22px "Crimson Pro", Georgia, serif';
      g.textAlign = 'center';
      g.fillStyle = 'rgba(240,240,220,0.95)';
      g.strokeStyle = 'rgba(10,14,18,0.85)';
      g.lineWidth = 0.04;
      const tagY = y - animal.def.radius - 0.22;
      g.strokeText(`♥ ${animal.name}`, x, tagY);
      g.fillText(`♥ ${animal.name}`, x, tagY);
      g.restore();
    }
  }
}
