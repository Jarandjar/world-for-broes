import { makeRng, hashStr } from '../engine/rng';
import { clamp, lerp, smoothstep } from '../engine/math';
import type { ColorSet, Content, Slot } from '../data/schema';
import type { Look } from '../sim/look';
import { resolvedColors } from '../sim/look';
import { PARTS, type PartCtx } from './parts';
import { type G, INK, TAU } from './ink';
import { SpriteCache, blit, type Sprite } from './sprites';

export interface Pose { phase: number; walk: number; swing: number; time: number }
export const IDLE_POSE: Pose = { phase: 0, walk: 0, swing: -1, time: 0 };

// ================================================================ shared sprite FX (used by zombies, animals and pets too)
/** Toon look dials. Tweak here; every figure renderer reads them. */
export const TOON = {
  rimColor: 'rgba(255,238,200,0.62)',   // warm paper-cutout edge light
  light: [-0.6, -0.8] as [number, number], // screen-space direction the key light comes from (upper left)
  rimInset: 0.02,                        // skip the ink outline band (world units)
  rimWidth: 0.022,                       // width of the lit edge (world units)
  heroOutline: 0.032,                    // extra outer contour around heroes (world units; ~3.5-4.5px at play zoom)
};

/** Sprite resolution bucket from the current transform (so renderers without a ppu argument can still cache). */
export function ppuOf(g: G): number {
  const m = g.getTransform(), s = Math.hypot(m.a, m.b);
  return s <= 56 ? 56 : s <= 96 ? 96 : s <= 150 ? 144 : 208;
}

const RIM_DIRS = 12;
const rimMaps = new WeakMap<HTMLCanvasElement, (HTMLCanvasElement | undefined)[]>();
const silMaps = new WeakMap<HTMLCanvasElement, Map<string, Sprite>>();

function rimCanvas(s: Sprite, k: number): HTMLCanvasElement {
  let arr = rimMaps.get(s.canvas);
  if (!arr) { arr = []; rimMaps.set(s.canvas, arr); }
  const hit = arr[k];
  if (hit) return hit;
  const c = document.createElement('canvas'); c.width = s.canvas.width; c.height = s.canvas.height;
  const x = c.getContext('2d') as G, a = (k / RIM_DIRS) * TAU, d0 = TOON.rimInset * s.ppu, d1 = d0 + Math.max(1.2, TOON.rimWidth * s.ppu);
  // band of pixels whose distance to the lit edge is between d0 and d1: S(p + L*d0) and not S(p + L*d1)
  x.drawImage(s.canvas, -Math.cos(a) * d0, -Math.sin(a) * d0);
  x.globalCompositeOperation = 'destination-out'; x.drawImage(s.canvas, -Math.cos(a) * d1, -Math.sin(a) * d1);
  x.globalCompositeOperation = 'destination-in'; x.drawImage(s.canvas, 0, 0);
  x.globalCompositeOperation = 'source-in'; x.fillStyle = TOON.rimColor; x.fillRect(0, 0, c.width, c.height);
  arr[k] = c;
  return c;
}

/** Toon rim light: a crisp lit edge on the side of the sprite facing the screen-space key light. Cached in 12 directions. */
export function blitRim(g: G, s: Sprite, x = 0, y = 0, rot = 0, sx = 1, sy = 1, a = 1): void {
  g.save(); g.translate(x, y); if (rot) g.rotate(rot); if (sx !== 1 || sy !== 1) g.scale(sx, sy);
  const m = g.getTransform(), det = m.a * m.d - m.b * m.c;
  if (det) {
    const [lx, ly] = TOON.light, ux = (m.d * lx - m.c * ly) / det, uy = (-m.b * lx + m.a * ly) / det;
    const k = ((Math.round((Math.atan2(uy, ux) / TAU) * RIM_DIRS) % RIM_DIRS) + RIM_DIRS) % RIM_DIRS;
    if (a !== 1) g.globalAlpha *= a;
    g.drawImage(rimCanvas(s, k), -s.ox / s.ppu, -s.oy / s.ppu, s.w / s.ppu, s.h / s.ppu);
  }
  g.restore();
}

/** Solid-colour silhouette of a sprite, optionally grown by `grow` world units (outer contours, hit flashes). Cached. */
export function silhouette(s: Sprite, color: string, grow = 0): Sprite {
  let map = silMaps.get(s.canvas);
  if (!map) { map = new Map(); silMaps.set(s.canvas, map); }
  const key = `${color}|${grow}`, hit = map.get(key);
  if (hit) return hit;
  const px = grow * s.ppu, pad = Math.ceil(px) + 1;
  const c = document.createElement('canvas'); c.width = s.w + pad * 2; c.height = s.h + pad * 2;
  const x = c.getContext('2d') as G;
  if (px > 0) for (let i = 0; i < 16; i++) { const a = (i / 16) * TAU; x.drawImage(s.canvas, pad + Math.cos(a) * px, pad + Math.sin(a) * px); }
  x.drawImage(s.canvas, pad, pad);
  x.globalCompositeOperation = 'source-in'; x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
  const out: Sprite = { canvas: c, ox: s.ox + pad, oy: s.oy + pad, ppu: s.ppu, w: c.width, h: c.height };
  map.set(key, out);
  return out;
}

/** Soft blob shadow, shrinking and fading as the figure lifts (STYLE §5.3). */
export function blobShadow(g: G, rx: number, ry: number, lift = 0, x = 0.03, y = 0.04): void {
  const k = 1 / (1 + 0.5 * lift * 8);
  g.save(); g.fillStyle = `rgba(21,18,30,${(0.4 / (1 + 0.8 * lift * 8)).toFixed(3)})`;
  g.beginPath(); g.ellipse(x, y, rx * k, ry * k, 0, 0, TAU); g.fill(); g.restore();
}

function drawWheelchair(g: G, phase: number, walk: number, build: number, frame = '#5d6672'): void {
  const seat = '#2f343d', tire = '#171a20', rim = '#aeb6bf', spin = phase * walk * 0.9, back = 0.1 * build;
  const wheel = (x: number, y: number, r: number, rot: number): void => {
    g.fillStyle = tire; g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill();
    g.fillStyle = rim; g.beginPath(); g.arc(x, y, r * 0.72, 0, TAU); g.fill();
    g.fillStyle = tire; g.beginPath(); g.arc(x, y, r * 0.54, 0, TAU); g.fill();
    g.strokeStyle = rim; g.lineWidth = 0.008;
    for (let i = 0; i < 6; i++) {
      const a = rot + (i / 6) * TAU;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * r * 0.68, y + Math.sin(a) * r * 0.68); g.stroke();
    }
  };

  g.save();
  wheel(-0.03, -0.2 * build, 0.11, spin);
  wheel(-0.03, 0.2 * build, 0.11, -spin);
  g.fillStyle = seat; g.fillRect(-0.05, -0.16 * build, 0.16, 0.32 * build);
  g.fillStyle = frame; g.fillRect(-0.13, -back, 0.08, back * 2); g.fillRect(0.05, -0.18 * build, 0.04, 0.36 * build);
  g.strokeStyle = frame; g.lineWidth = 0.018;
  g.beginPath(); g.moveTo(-0.04, -0.14 * build); g.lineTo(0.11, -0.14 * build); g.lineTo(0.16, -0.09 * build); g.stroke();
  g.beginPath(); g.moveTo(-0.04, 0.14 * build); g.lineTo(0.11, 0.14 * build); g.lineTo(0.16, 0.09 * build); g.stroke();
  g.fillStyle = rim; g.beginPath(); g.arc(0.2, -0.12 * build, 0.03, 0, TAU); g.arc(0.2, 0.12 * build, 0.03, 0, TAU); g.fill();
  g.restore();
}

// ================================================================ characters
const BOUNDS: Record<string, [number, number, number, number]> = {
  body: [-0.5, -0.45, 0.45, 0.45], back: [-0.75, -0.42, 0.15, 0.42], held: [-0.28, -0.42, 1.25, 0.45], arm: [-0.1, -0.16, 0.42, 0.16],
};
const boundsFor = (k: string): [number, number, number, number] => (k.startsWith('mainHand') || k.startsWith('offHand') ? BOUNDS.held : k.startsWith('back') ? BOUNDS.back : k === 'body:arm' ? BOUNDS.arm : BOUNDS.body) as [number, number, number, number];
const LONG_TOOLS = new Set(['axe', 'pickaxe', 'shovel', 'fishing_rod', 'crowbar', 'nail_bat']);
const DEFAULT_COLORS: ColorSet = { primary: '#777777', secondary: '#555555', accent: '#bbbbbb' };

/** Animation dials for characters (radians / fractions). Data, so other tools can tune them. */
export const CHAR_ANIM = {
  stepLift: 0.03,      // whole figure grows slightly mid-stride (bob toward the camera)
  stepSquash: 0.035,   // torso stretch along facing on each step
  breathe: 0.022,      // idle chest breathing
  headLook: 0.34,      // idle head look-around
  headWalk: 0.05,      // head counter-sway while walking
};

type Put = (s: Sprite | null, x?: number, y?: number, rot?: number, sx?: number, sy?: number, rim?: boolean) => void;

/** Composes a layered top-down character from cached part sprites and animates it procedurally. */
export class CharacterRenderer {
  constructor(private cache: SpriteCache, private content: Content) {}

  private sprite(key: string, ppu: number, ctx: Omit<PartCtx, 'g' | 'rng'>): Sprite | null {
    const fn = PARTS[key];
    if (!fn) return null;
    const full = `${key}|${ctx.c.primary}|${ctx.c.secondary}|${ctx.c.accent}|${ctx.skin}|${ctx.hair}|${ctx.hand}|${ctx.cuff}|${ctx.build.toFixed(2)}|${ctx.side}`;
    const [x0, y0, x1, y1] = boundsFor(key);
    return this.cache.get(full, ppu, x0, y0, x1, y1, (g) => fn({ ...ctx, g, rng: makeRng(hashStr(full)) }));
  }

  /** Draws the character at the current origin facing +x. Caller sets position, rotation and zoom. */
  draw(g: G, look: Look, pose: Pose, ppu: number): void {
    const w = pose.walk, lift = Math.abs(Math.sin(pose.phase)) * w;
    blobShadow(g, 0.36, 0.31 * look.appearance.build, lift * CHAR_ANIM.stepLift, 0.02, 0.035);
    const grow = TOON.heroOutline;
    // pass 1: one bold ink contour around the whole figure; pass 2: the parts with a toon rim on each
    this.compose(g, look, pose, ppu, (s, x = 0, y = 0, rot = 0, sx = 1, sy = 1) => { if (s) blit(g, silhouette(s, INK, grow), x, y, rot, sx, sy); });
    this.compose(g, look, pose, ppu, (s, x = 0, y = 0, rot = 0, sx = 1, sy = 1, rim = true) => { if (!s) return; blit(g, s, x, y, rot, sx, sy); if (rim) blitRim(g, s, x, y, rot, sx, sy); });
  }

  private compose(g: G, look: Look, pose: Pose, ppu: number, put: Put): void {
    const a = look.appearance, o = look.outfit, b = a.build;
    const colorOf = (slot: Slot): ColorSet | null => (o[slot] ? resolvedColors(this.content, o[slot]) : null);
    const torso = colorOf('torso'), legs = colorOf('legs'), feet = colorOf('feet'), hands = colorOf('hands');
    const base = { skin: a.skin, hair: a.hairColor, hand: hands?.primary ?? a.skin, cuff: hands?.secondary ?? torso?.secondary ?? torso?.primary ?? a.skin, build: b, side: 1 };
    const sp = (key: string, c: ColorSet | null, extra: Partial<typeof base> = {}): Sprite | null => this.sprite(key, ppu, { ...base, ...extra, c: c ?? DEFAULT_COLORS });
    const w = pose.walk, t = pose.time, idle = 1 - w;
    const lift = 1 + CHAR_ANIM.stepLift * Math.abs(Math.sin(pose.phase)) * w;
    const wheelchair = a.wheelchair;

    g.save(); g.scale(lift, lift);
    if (wheelchair) drawWheelchair(g, pose.phase, w, b, torso?.secondary ?? '#5d6672');
    // ---- legs and feet (stride)
    for (const side of [-1, 1]) {
      const step = wheelchair ? 0 : Math.sin(pose.phase + (side > 0 ? 0 : Math.PI)) * 0.15 * w;
      const fx = wheelchair ? 0.12 : 0.16 + step;
      const hy = side * (wheelchair ? 0.085 : 0.1) * b;
      const leg = sp(legs ? `legs:${o.legs?.item}` : 'legs:none', legs ?? { ...DEFAULT_COLORS, primary: a.skin }, { side });
      if (wheelchair) put(leg, -0.03, hy, side * 0.02, 0.45, 0.92, false);
      else put(leg, -0.02, hy, 0, clamp((fx + 0.02) / 0.21, 0.5, 1.8), 1, false);
      const foot = sp(feet ? `feet:${o.feet?.item}` : 'feet:none', feet, { side });
      put(foot, fx, hy, wheelchair ? side * 0.02 : side * 0.08 + step * 0.4, wheelchair ? 0.82 : 1 + Math.max(0, step) * 0.6, 1, false);   // toe lifts (stretches) on the forward foot
    }

    // ---- swing timeline for the main hand (top-down: a sweeping arc with foreshortening)
    const sw = pose.swing;
    let twist = 0, lunge = 0, swingArm = 0, itemScale = 1, smear = 0;
    if (sw >= 0) {
      if (sw < 0.35) { const u = smoothstep(0, 0.35, sw); swingArm = lerp(0, 1.45, u); twist = lerp(0, -0.22, u); smear = -0.05 * u; }
      else if (sw < 0.62) { const u = smoothstep(0.35, 0.62, sw); swingArm = lerp(1.45, -0.7, u); twist = lerp(-0.22, 0.3, u); lunge = Math.sin(u * Math.PI) * 0.07; itemScale = 1 + 0.12 * Math.sin(u * Math.PI); smear = 0.09 * Math.sin(u * Math.PI); }
      else { const u = smoothstep(0.62, 1, sw); swingArm = lerp(-0.7, 0, u); twist = lerp(0.3, 0, u); smear = -0.03 * (1 - u); }
    }
    const holdMain = !!o.mainHand, holdOff = !!o.offHand;
    const breathe = 1 + (0.008 + CHAR_ANIM.breathe * idle) * Math.sin(t * 2.2);
    const bounce = Math.sin(pose.phase * 2) * w;
    // squash and stretch: stretch along facing on each step and on the swing follow-through, keeping volume
    const sx = 1 + CHAR_ANIM.stepSquash * bounce + smear, sy = (1 - CHAR_ANIM.stepSquash * 0.7 * bounce - smear * 0.6) * breathe;

    // ---- upper body group
    g.save(); g.translate(lunge, 0); g.rotate(twist + 0.06 * Math.sin(pose.phase) * w); g.scale(sx, sy);
    const torsoSprite = sp(`torso:${o.torso?.item ?? 'none'}`, torso ?? { ...DEFAULT_COLORS, primary: a.skin });
    if (o.belt) put(sp(`belt:${o.belt.item}`, colorOf('belt')), 0, 0, 0, 1, 1, false);
    put(torsoSprite);
    if (o.back) put(sp(`back:${o.back.item}`, colorOf('back')));

    // arms with held items
    for (const side of [-1, 1]) {
      const holding = side > 0 ? holdMain : holdOff;
      let outward = holding ? -0.4 : 0.14 + 0.04 * idle * Math.sin(t * 2.2 + side);   // loose arms drift with the breath
      let reach = 1 + Math.sin(pose.phase + (side > 0 ? Math.PI : 0)) * 0.28 * w * (holding ? 0.35 : 1);
      if (side > 0 && sw >= 0 && holdMain) { outward = -0.4 + swingArm * 0.9; reach = 1.02; }
      const ang = side * outward;
      const sy2 = side * 0.29 * b;
      put(sp('body:arm', torso ?? { ...DEFAULT_COLORS, primary: a.skin }, { side }), 0, sy2, ang, reach, 1);
      const handX = Math.cos(ang) * 0.31 * reach, handY = sy2 + Math.sin(ang) * 0.31 * reach;
      const inst = side > 0 ? o.mainHand : o.offHand;
      if (inst) {
        const key = `${side > 0 ? 'mainHand' : 'offHand'}:${inst.item}`;
        const item = sp(PARTS[key] ? key : `mainHand:${inst.item}`, resolvedColors(this.content, inst), { side });
        put(item, handX, handY, ang + side * 0.05, itemScale * (LONG_TOOLS.has(inst.item) ? 0.78 : 1), 1);
      }
    }
    if (o.neck) put(sp(`neck:${o.neck.item}`, colorOf('neck')), 0, 0, 0, 1, 1, false);

    // ---- head group: idle look-around (two slow sines, desynced per entity by pose.time), counter-sway while walking
    const look2 = CHAR_ANIM.headLook * idle * Math.sin(t * 0.45) * Math.sin(t * 0.17 + 1.3) - CHAR_ANIM.headWalk * w * Math.sin(pose.phase);
    g.scale(1 / sx, 1 / sy);   // the head keeps its shape while the torso squashes
    g.translate(0.03, 0); g.rotate(-twist * 0.6 + look2);
    put(sp('body:head', null));
    put(sp(`hair:${a.hairStyle}`, null));
    if (o.head) put(sp(`head:${o.head.item}`, colorOf('head')));
    if (o.face) put(sp(`face:${o.face.item}`, colorOf('face')), 0, 0, 0, 1, 1, false);
    g.restore();
    g.restore();
  }
}
