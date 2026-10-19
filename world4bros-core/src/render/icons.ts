import type { Content } from '../data/schema';
import { makeRng, hashStr } from '../engine/rng';
import { PARTS } from './parts';
import { type G, INK, TAU, inked, ell, cap, rrect, poly, lumpy, stroke } from './ink';

const cache = new Map<string, HTMLCanvasElement>();
const VIEW: Record<string, [number, number, number, number]> = {   // [centerX, centerY, span, rotation]
  head: [0, 0, 0.8, 0], face: [0.1, 0, 0.55, 0], neck: [-0.05, 0.05, 0.6, 0], torso: [-0.02, 0, 0.78, -0.5], legs: [0.08, 0, 0.42, -0.7], feet: [0.03, 0, 0.3, -0.5],
  back: [-0.32, 0, 0.85, 0], belt: [0, 0, 0.7, 0], mainHand: [0.3, 0.02, 0.95, -0.78], offHand: [0.14, 0, 0.5, -0.4],
};
type IconFn = (g: G, rng: ReturnType<typeof makeRng>) => void;
const ik = (g: G, rng: ReturnType<typeof makeRng>, path: () => void, fill: string, o: { hatch?: number; line?: number } = {}): void => inked(g, path, fill, { rng, box: 0.35, ...o });
const ICONS: Record<string, IconFn> = {
  wood: (g, r) => { for (const [y, c] of [[-0.14, '#6a5645'], [0.14, '#5f4d3e']] as [number, string][]) { ik(g, r, () => cap(g, -0.34, y, 0.28, y, 0.13, 0.13), c); ik(g, r, () => ell(g, 0.28, y, 0.05, 0.13), '#a58d6c', { hatch: 0.2, line: 0.014 }); } },
  twig: (g) => { for (const p of [[[-0.4, 0.3], [0.35, -0.3]], [[0, 0], [0.3, 0.25]], [[-0.15, 0.1], [-0.35, -0.2]]] as [number, number][][]) { stroke(g, p, 0.075, INK); stroke(g, p, 0.05, '#7a6650'); } },
  stone: (g, r) => ik(g, r, () => poly(g, [[-0.34, 0.15], [-0.2, -0.28], [0.18, -0.32], [0.38, 0.05], [0.2, 0.3], [-0.2, 0.3]]), '#74787e'),
  flint: (g, r) => { ik(g, r, () => poly(g, [[-0.3, 0.2], [-0.1, -0.32], [0.3, -0.1], [0.28, 0.25]]), '#3d4348'); stroke(g, [[-0.05, -0.2], [0.12, 0.05]], 0.02, '#aab4bc', 0.8); },
  pine_cone: (g, r) => { ik(g, r, () => ell(g, 0, 0, 0.2, 0.34, 0.5), '#6a5645'); for (let y = -0.25; y < 0.28; y += 0.1) stroke(g, [[-0.15, y], [0.0, y + 0.07], [0.15, y]], 0.018, INK, 0.7); },
  acorn: (g, r) => { ik(g, r, () => ell(g, 0, 0.08, 0.17, 0.24), '#a8864f'); ik(g, r, () => ell(g, 0, -0.1, 0.2, 0.12), '#5a4838'); stroke(g, [[0, -0.22], [0.03, -0.36]], 0.03, INK); },
  fiber: (g) => { for (let i = -2; i <= 2; i++) { stroke(g, [[i * 0.07, 0.35], [i * 0.1, 0], [i * 0.04 + 0.05, -0.35]], 0.05, INK, 0.9); stroke(g, [[i * 0.07, 0.35], [i * 0.1, 0], [i * 0.04 + 0.05, -0.35]], 0.03, i % 2 ? '#9aa06a' : '#7d8456'); } },
  scrap_metal: (g, r) => { ik(g, r, () => poly(g, [[-0.35, -0.2], [0.2, -0.32], [0.38, 0.1], [-0.1, 0.3], [-0.3, 0.18]]), '#7a8088'); ik(g, r, () => ell(g, 0.05, -0.02, 0.04, 0.04), '#3d4348', { hatch: 0, line: 0.012 }); },
  cloth: (g, r) => { ik(g, r, () => rrect(g, -0.32, -0.24, 0.64, 0.48, 0.05), '#8a7f66'); stroke(g, [[-0.32, 0], [0.32, 0]], 0.02, INK, 0.5); stroke(g, [[-0.32, -0.12], [0.32, -0.12]], 0.05, '#8a3a32', 0.8); },
  berries: (g, r) => { for (const [x, y] of [[-0.14, 0.08], [0.12, 0.12], [0, -0.14]] as [number, number][]) ik(g, r, () => ell(g, x, y, 0.15, 0.15), '#9a3a36', { hatch: 0.5, line: 0.016 }); ik(g, r, () => poly(g, [[0, -0.3], [0.2, -0.34], [0.12, -0.2]]), '#566a4a', { hatch: 0, line: 0.012 }); },
  mushroom: (g, r) => { ik(g, r, () => cap(g, 0, 0.05, 0, 0.3, 0.07, 0.09), '#d8d0b8'); ik(g, r, () => ell(g, 0, -0.06, 0.3, 0.2), '#a8997f'); for (const [x, y] of [[-0.1, -0.08], [0.1, -0.12], [0.02, 0]] as [number, number][]) { g.fillStyle = '#efe8d6'; g.beginPath(); g.arc(x, y, 0.03, 0, TAU); g.fill(); } },
  cactus_pulp: (g, r) => { ik(g, r, () => lumpy(g, 0, 0, 0.3, 0.2, r, 8, 0.1), '#6a8a62'); for (let a = 0; a < TAU; a += 0.8) stroke(g, [[Math.cos(a) * 0.12, Math.sin(a) * 0.08], [Math.cos(a) * 0.28, Math.sin(a) * 0.18]], 0.012, '#d8d0b0'); },
  jerky: (g, r) => { for (const y of [-0.15, 0.05, 0.22]) ik(g, r, () => cap(g, -0.3, y, 0.28, y + 0.02, 0.07, 0.06), '#6a4a38'); },
  canned_beans: (g, r) => { ik(g, r, () => rrect(g, -0.2, -0.3, 0.4, 0.6, 0.04), '#8a8f95'); ik(g, r, () => rrect(g, -0.2, -0.12, 0.4, 0.26, 0.02), '#a8523a', { hatch: 0.3, line: 0.012 }); stroke(g, [[-0.2, -0.3], [0.2, -0.3]], 0.03, '#5f666d'); },
  leather_gloves: (g, r) => { ik(g, r, () => lumpy(g, 0, 0.05, 0.22, 0.28, r, 9, 0.08), '#4a3a2c'); for (let i = 0; i < 4; i++) ik(g, r, () => cap(g, -0.14 + i * 0.09, -0.12, -0.14 + i * 0.09, -0.34, 0.04, 0.04), '#4a3a2c', { hatch: 0.4, line: 0.014 }); ik(g, r, () => rrect(g, -0.2, 0.25, 0.4, 0.1, 0.03), '#8a7a62', { hatch: 0.2, line: 0.012 }); },
  mittens: (g, r) => { ik(g, r, () => lumpy(g, 0, 0.05, 0.22, 0.3, r, 9, 0.06), '#8a3a4a'); ik(g, r, () => ell(g, -0.22, 0.02, 0.07, 0.12, -0.4), '#8a3a4a', { hatch: 0.4, line: 0.014 }); ik(g, r, () => rrect(g, -0.2, 0.25, 0.4, 0.1, 0.03), '#c9c3b2', { hatch: 0.2, line: 0.012 }); },
  fuel_canister: (g, r) => {
    ik(g, r, () => rrect(g, -0.22, -0.2, 0.44, 0.5, 0.04), '#c44528', { hatch: 0.8, line: 0.02 });
    ik(g, r, () => cap(g, -0.12, -0.2, 0.12, -0.2, 0.04, 0.04), '#7a2818', { hatch: 0.2, line: 0.014 });
    stroke(g, [[-0.1, -0.05], [0.1, -0.05]], 0.02, INK, 0.7);
    ik(g, r, () => ell(g, 0.1, -0.28, 0.05, 0.04), '#f0ad4e', { hatch: 0, line: 0.012 });
  },
  car_battery: (g, r) => {
    ik(g, r, () => rrect(g, -0.28, -0.18, 0.56, 0.44, 0.03), '#32373d', { hatch: 0.6, line: 0.02 });
    ik(g, r, () => rrect(g, -0.2, -0.28, 0.1, 0.1, 0.01), '#b53828', { hatch: 0, line: 0.012 });
    ik(g, r, () => rrect(g, 0.1, -0.28, 0.1, 0.1, 0.01), '#3b6ea5', { hatch: 0, line: 0.012 });
  },
  scrap_tire: (g, r) => {
    ik(g, r, () => ell(g, 0, 0, 0.32, 0.32), '#25292e', { hatch: 0.7, line: 0.022 });
    ik(g, r, () => ell(g, 0, 0, 0.14, 0.14), '#747c87', { hatch: 0.3, line: 0.014 });
    ik(g, r, () => ell(g, 0, 0, 0.05, 0.05), '#15171b', { hatch: 0, line: 0.01 });
  },
  bandage: (g, r) => {
    ik(g, r, () => rrect(g, -0.22, -0.16, 0.44, 0.32, 0.06), '#e8e2d5', { hatch: 0.4, line: 0.016 });
    ik(g, r, () => rrect(g, -0.05, -0.14, 0.1, 0.28, 0.02), '#c4473d', { hatch: 0, line: 0.01 });
    ik(g, r, () => rrect(g, -0.18, -0.05, 0.36, 0.1, 0.02), '#c4473d', { hatch: 0, line: 0.01 });
  },
  rotten_meat: (g, r) => {
    ik(g, r, () => lumpy(g, 0, 0, 0.28, 0.22, r, 7, 0.08), '#5a6b57', { hatch: 0.7, line: 0.018 });
    ik(g, r, () => ell(g, -0.08, -0.04, 0.08, 0.06), '#7a2824', { hatch: 0.5, line: 0.012 });
    ik(g, r, () => cap(g, 0.14, -0.1, 0.26, -0.18, 0.03, 0.02), '#e4dfd0', { hatch: 0, line: 0.012 });
  },
  iron_ore: (g, r) => {
    ik(g, r, () => poly(g, [[-0.3, 0.15], [-0.18, -0.25], [0.15, -0.3], [0.34, 0.04], [0.22, 0.28], [-0.15, 0.3]]), '#5e656d');
    stroke(g, [[-0.1, -0.15], [0.15, 0.1]], 0.03, '#9ab0c4', 0.9);
    ik(g, r, () => ell(g, 0.1, -0.05, 0.06, 0.06), '#d4e3f0', { hatch: 0, line: 0.01 });
  },
  copper_ore: (g, r) => {
    ik(g, r, () => poly(g, [[-0.28, 0.18], [-0.15, -0.28], [0.18, -0.26], [0.32, 0.08], [0.18, 0.26], [-0.1, 0.28]]), '#6b5141');
    stroke(g, [[-0.12, -0.1], [0.12, 0.08]], 0.035, '#c97746', 0.95);
    ik(g, r, () => ell(g, 0.05, -0.08, 0.06, 0.05), '#4ecdc4', { hatch: 0, line: 0.01 });
  },
  coal_chunk: (g, r) => {
    ik(g, r, () => poly(g, [[-0.32, 0.1], [-0.15, -0.3], [0.22, -0.25], [0.3, 0.12], [0.1, 0.32], [-0.2, 0.25]]), '#24272c');
    stroke(g, [[-0.15, 0.0], [0.1, -0.15]], 0.025, '#495057', 0.8);
    ik(g, r, () => ell(g, -0.05, -0.1, 0.04, 0.04), '#6c757d', { hatch: 0, line: 0.01 });
  },
  crude_oil_canister: (g, r) => {
    ik(g, r, () => rrect(g, -0.22, -0.28, 0.44, 0.56, 0.06), '#2b2d42', { hatch: 0.7, line: 0.02 });
    for (const y of [-0.1, 0.1]) stroke(g, [[-0.22, y], [0.22, y]], 0.02, '#8d99ae', 0.8);
    ik(g, r, () => ell(g, 0, 0, 0.08, 0.12), '#111217', { hatch: 0, line: 0.01 });
  },
  energetic_crystal: (g, r) => {
    ik(g, r, () => poly(g, [[0, -0.35], [0.24, -0.1], [0.16, 0.32], [-0.16, 0.32], [-0.24, -0.1]]), '#8338ec');
    ik(g, r, () => poly(g, [[0, -0.35], [0.12, -0.05], [0.0, 0.32], [-0.12, -0.05]]), '#c77dff', { hatch: 0.2, line: 0.014 });
    stroke(g, [[-0.06, -0.1], [0.06, 0.1]], 0.02, '#ffffff', 0.9);
  },
  iron_plate: (g, r) => {
    ik(g, r, () => rrect(g, -0.3, -0.22, 0.6, 0.44, 0.04), '#6c757d', { hatch: 0.5, line: 0.02 });
    stroke(g, [[-0.28, 0], [0.28, 0]], 0.015, '#adb5bd', 0.7);
    for (const [x, y] of [[-0.22, -0.14], [0.22, -0.14], [-0.22, 0.14], [0.22, 0.14]] as [number, number][]) {
      ik(g, r, () => ell(g, x, y, 0.03, 0.03), '#343a40', { hatch: 0, line: 0.01 });
    }
  },
  copper_wire: (g, r) => {
    ik(g, r, () => ell(g, 0, 0, 0.28, 0.28), '#343a40', { hatch: 0.3, line: 0.018 });
    for (let radius = 0.08; radius <= 0.24; radius += 0.05) {
      g.beginPath(); g.arc(0, 0, radius, 0, TAU); g.strokeStyle = '#d97706'; g.lineWidth = 0.025; g.stroke();
    }
  },
  heavy_gear: (g, r) => {
    ik(g, r, () => ell(g, 0, 0, 0.3, 0.3), '#495057', { hatch: 0.8, line: 0.024 });
    for (let a = 0; a < TAU; a += TAU / 6) {
      stroke(g, [[Math.cos(a) * 0.2, Math.sin(a) * 0.2], [Math.cos(a) * 0.36, Math.sin(a) * 0.36]], 0.06, '#6c757d', 1);
    }
    ik(g, r, () => ell(g, 0, 0, 0.1, 0.1), '#212529', { hatch: 0, line: 0.014 });
  },
  conveyor_belt: (g, r) => {
    ik(g, r, () => rrect(g, -0.28, -0.28, 0.56, 0.56, 0.04), '#343a40', { hatch: 0.5, line: 0.02 });
    stroke(g, [[-0.15, 0.08], [0, -0.08], [0.15, 0.08]], 0.04, '#fbbf24', 0.9);
  },
  spring_inserter: (g, r) => {
    ik(g, r, () => ell(g, 0, 0.12, 0.18, 0.18), '#495057', { hatch: 0.5, line: 0.016 });
    stroke(g, [[0, 0.12], [0.2, -0.2]], 0.04, '#adb5bd', 1);
    stroke(g, [[-0.08, -0.05], [0.08, -0.01], [-0.08, 0.03], [0.08, 0.07]], 0.02, '#e09f3e', 0.9);
    ik(g, r, () => rrect(g, 0.16, -0.26, 0.1, 0.14, 0.02), '#ef233c', { hatch: 0, line: 0.012 });
  },
  burner_drill: (g, r) => {
    ik(g, r, () => rrect(g, -0.28, -0.28, 0.56, 0.56, 0.06), '#495057', { hatch: 0.8, line: 0.022 });
    ik(g, r, () => ell(g, 0, 0, 0.16, 0.16), '#6c757d', { hatch: 0.4, line: 0.016 });
    stroke(g, [[0, 0], [0, -0.25]], 0.05, '#212529', 1);
  },
  stone_furnace: (g, r) => {
    ik(g, r, () => rrect(g, -0.28, -0.28, 0.56, 0.56, 0.06), '#5c534b', { hatch: 0.8, line: 0.024 });
    ik(g, r, () => ell(g, 0, 0.06, 0.16, 0.12), '#f97316', { hatch: 0, line: 0.014 });
    ik(g, r, () => ell(g, 0, 0.06, 0.08, 0.06), '#fde047', { hatch: 0, line: 0.01 });
  },
};
const skin = '#d9a77f', hair = '#4a3022';

export function itemIcon(content: Content, id: string, px = 44): HTMLCanvasElement {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), key = `${id}@${px}@${dpr}`;
  const hit = cache.get(key); if (hit) return hit;
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = Math.round(px * dpr);
  const g = canvas.getContext('2d') as G; g.scale(canvas.width, canvas.height);
  const def = content.items.get(id), rng = makeRng(hashStr(id));
  g.translate(0.5, 0.5); g.scale(0.9, 0.9);
  const ic = ICONS[def?.icon ?? id];
  if (ic) { g.scale(1.0, 1.0); ic(g, rng); }
  else if (def?.equip) {
    const slot = def.equip.slot, part = PARTS[`${slot}:${id}`] ?? PARTS[`mainHand:${id}`];
    const [cx, cy, span, rot] = VIEW[slot] ?? [0, 0, 0.8, 0];
    if (part) { g.scale(1 / span, 1 / span); g.rotate(rot); g.translate(-cx, -cy); part({ g, rng, c: def.equip.colors, skin, hair, hand: skin, cuff: def.equip.colors.secondary, build: 1, side: 1 }); }
  } else { inked(g, () => ell(g, 0, 0, 0.3, 0.3), '#777', { rng, box: 0.35 }); }
  cache.set(key, canvas);
  return canvas;
}
export const clearIconCache = (): void => cache.clear();
