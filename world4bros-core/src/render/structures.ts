import type { Rng } from '../engine/rng';
import { type G, INK, TAU, inked, ell, cap, rrect, poly, lumpy, stroke, shade, alpha } from './ink';

/** Structure drawers. Origin = top-left corner of the footprint; units are tiles. */
export type StructDraw = (g: G, rng: Rng, w: number, h: number, open: boolean) => void;
const ik = (g: G, rng: Rng, path: () => void, fill: string, box: number, o: { hatch?: number; line?: number; light?: [number, number] } = {}): void => inked(g, path, fill, { rng, box, ...o });

export const STRUCTURE_DRAWERS: Record<string, StructDraw> = {
  campfire(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    // Radiant heat glow on ground
    const heat = g.createRadialGradient(0, 0, 0.1, 0, 0, 0.58);
    heat.addColorStop(0, 'rgba(249, 115, 22, 0.35)');
    heat.addColorStop(0.5, 'rgba(234, 88, 12, 0.15)');
    heat.addColorStop(1, 'rgba(234, 88, 12, 0)');
    g.fillStyle = heat;
    g.beginPath(); g.arc(0, 0, 0.58, 0, TAU); g.fill();

    // Deep ash bed
    ik(g, rng, () => ell(g, 0, 0, 0.44, 0.44), '#262322', 0.45, { hatch: 0.35, line: 0.016 });
    // Granite hearth ring stones with soot
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * TAU, col = i % 2 ? '#6b7280' : '#4b5563';
      ik(g, rng, () => lumpy(g, Math.cos(a) * 0.41, Math.sin(a) * 0.41, 0.11, 0.09, rng, 7, 0.18, a), col, 0.12, { hatch: 0.9, line: 0.018 });
    }
    // Charred split hardwood logs
    for (const a of [0.35, 1.85, 3.45, 4.95]) {
      ik(g, rng, () => cap(g, Math.cos(a) * 0.34, Math.sin(a) * 0.34, -Math.cos(a) * 0.14, -Math.sin(a) * 0.14, 0.07, 0.055), '#3e2e22', 0.32, { hatch: 0.9, line: 0.018 });
      // Glowing heat crack in log
      stroke(g, [[Math.cos(a) * 0.22, Math.sin(a) * 0.22], [-Math.cos(a) * 0.05, -Math.sin(a) * 0.05]], 0.015, '#f97316', 0.9);
    }
    // Incandescent glowing ember core
    ik(g, rng, () => ell(g, 0, 0, 0.18, 0.16), '#dc2626', 0.2, { hatch: 0, line: 0.01 });
    ik(g, rng, () => ell(g, 0, 0, 0.11, 0.09), '#f97316', 0.14, { hatch: 0, line: 0.008 });
    ik(g, rng, () => ell(g, 0, 0, 0.05, 0.04), '#fef08a', 0.08, { hatch: 0, line: 0 });
    g.restore();
  },
  claim_flag(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.2, 0.2), '#5f6368', 0.22, { hatch: 0.8, line: 0.018 });
    ik(g, rng, () => poly(g, [[0.02, -0.04], [0.5, -0.12], [0.44, 0.0], [0.52, 0.12], [0.02, 0.06]]), '#8a3a32', 0.3, { hatch: 0.9, line: 0.02 });
    stroke(g, [[0.1, -0.02], [0.4, -0.06]], 0.01, '#e8d8b0', 0.8);
    ik(g, rng, () => ell(g, 0, 0, 0.055, 0.055), '#3a2f28', 0.08, { hatch: 0.2, line: 0.014 });
    g.restore();
  },
  wall_wood(g, rng) {
    // Solid stacked timber wall with heavy mortise-and-tenon logs & iron corner brackets
    ik(g, rng, () => rrect(g, 0.04, 0.04, 0.92, 0.92, 0.04), '#5c4533', 0.5, { hatch: 0.8, line: 0.024 });

    // 3 Horizontal timber beam courses
    for (let j = 0; j < 3; j++) {
      const y = 0.06 + j * 0.29;
      // Top bevel highlight
      g.fillStyle = alpha('#ffffff', 0.14); g.fillRect(0.06, y, 0.88, 0.04);
      // Bottom shadow groove
      g.fillStyle = alpha(INK, 0.35); g.fillRect(0.06, y + 0.24, 0.88, 0.04);
      // Beam seam
      stroke(g, [[0.06, y + 0.28], [0.94, y + 0.28]], 0.02, INK, 0.6);
      // Wood grain striations
      stroke(g, [[0.12, y + 0.14], [0.88, y + 0.14]], 0.012, alpha('#854d0e', 0.5));
    }

    // Heavy round log ends with concentric growth rings on corners
    for (let j = 0; j < 3; j++) {
      for (const i of [0, 2]) {
        const cx = 0.15 + i * 0.35, cy = 0.18 + j * 0.31;
        ik(g, rng, () => ell(g, cx, cy, 0.12, 0.11), '#78553a', 0.14, { hatch: 0.6, line: 0.016 });
        g.strokeStyle = alpha('#451a03', 0.6); g.lineWidth = 0.01;
        g.beginPath(); g.arc(cx, cy, 0.06, 0, TAU); g.stroke();
        g.fillStyle = INK; g.beginPath(); g.arc(cx, cy, 0.02, 0, TAU); g.fill();
      }
    }

    // Studded black iron corner reinforcement brackets
    for (const [bx, by] of [[0.04, 0.04], [0.82, 0.04], [0.04, 0.82], [0.82, 0.82]]) {
      g.fillStyle = '#1e293b'; g.fillRect(bx, by, 0.14, 0.14);
      g.strokeStyle = INK; g.lineWidth = 0.02; g.strokeRect(bx, by, 0.14, 0.14);
      // Iron rivet
      g.fillStyle = '#94a3b8'; g.fillRect(bx + 0.045, by + 0.045, 0.04, 0.04);
    }
  },
  door_wood(g, rng, _w, _h, open) {
    if (open) {
      g.fillStyle = alpha('#3a2f28', 0.25); g.fillRect(0, 0, 1, 1);
      // Swung open heavy plank leaf
      ik(g, rng, () => rrect(g, 0.0, 0.0, 0.15, 0.95, 0.02), '#6a523d', 0.5, { hatch: 0.8, line: 0.02 });
      // Brass handle
      g.fillStyle = '#f59e0b'; g.fillRect(0.04, 0.48, 0.08, 0.08);
      // Door sweep guide arc
      g.beginPath(); g.arc(0.08, 0.05, 0.9, 0.05, 1.5);
      g.strokeStyle = alpha(INK, 0.35); g.lineWidth = 0.01;
      g.setLineDash([0.04, 0.04]); g.stroke(); g.setLineDash([]);
      return;
    }
    // Heavy wooden door frame
    ik(g, rng, () => rrect(g, 0.0, 0.06, 1.0, 0.88, 0.03), '#5a4330', 0.55, { hatch: 0.9, line: 0.024 });

    // Vertical tongue-and-groove planks with alternating oak tones
    for (let i = 0; i < 4; i++) {
      const px = 0.05 + i * 0.225;
      const col = i % 2 ? '#745942' : '#654c37';
      ik(g, rng, () => rrect(g, px, 0.1, 0.21, 0.8, 0.01), col, 0.4, { hatch: 0.5, line: 0.015 });
      // Plank seam
      stroke(g, [[px + 0.225, 0.1], [px + 0.225, 0.9]], 0.015, INK, 0.6);
    }

    // Heavy black iron strap hinges with pin knuckles
    for (const y of [0.22, 0.76]) {
      ik(g, rng, () => rrect(g, 0.02, y - 0.05, 0.96, 0.1, 0.02), '#334155', 0.5, { hatch: 0.6, line: 0.016 });
      // Iron hinge studs
      for (const x of [0.15, 0.45, 0.75]) {
        g.fillStyle = '#64748b'; g.fillRect(x, y - 0.02, 0.04, 0.04);
      }
    }

    // Antique forged iron ring latch & brass escutcheon plate
    ik(g, rng, () => rrect(g, 0.76, 0.44, 0.12, 0.14, 0.02), '#b45309', 0.15, { hatch: 0.3, line: 0.014 });
    g.fillStyle = '#1e293b';
    g.beginPath(); g.arc(0.82, 0.53, 0.045, 0, TAU); g.fill();
    g.strokeStyle = '#94a3b8'; g.lineWidth = 0.015; g.stroke();
  },
  floor_wood(g, rng) {
    // Interlocking parquet hardwood floorboards with varied warm tones & finishing nails
    for (let i = 0; i < 4; i++) {
      const y = i * 0.25 + 0.01;
      const col = i === 0 ? '#7d634c' : i === 1 ? '#6f543e' : i === 2 ? '#785b44' : '#674f3b';
      ik(g, rng, () => rrect(g, 0.01, y, 0.98, 0.23, 0.01), col, 0.5, { hatch: 0.4, line: 0.012 });
      // Wood grain striation
      stroke(g, [[0.05, y + 0.11], [0.95, y + 0.11]], 0.01, alpha('#a16207', 0.35));
      // Countersunk finishing nail heads at board ends
      for (const x of [0.08, 0.92]) {
        g.fillStyle = alpha(INK, 0.65);
        g.beginPath(); g.arc(x, y + 0.115, 0.015, 0, TAU); g.fill();
      }
    }
  },
  chest(g, rng) {
    // Solid oak treasure chest with curved banded lid & studded brass brackets
    ik(g, rng, () => rrect(g, 0.07, 0.12, 0.86, 0.76, 0.06), '#654a36', 0.55, { hatch: 0.9, line: 0.024 });
    // Chest lid highlight
    ik(g, rng, () => rrect(g, 0.09, 0.14, 0.82, 0.32, 0.05), '#7c5d44', 0.45, { hatch: 0.5, line: 0.014 });

    // Studded iron reinforcement bands
    for (const x of [0.2, 0.8]) {
      stroke(g, [[x, 0.12], [x, 0.88]], 0.045, '#334155', 1);
      // Stud rivets
      g.fillStyle = '#94a3b8';
      for (const y of [0.18, 0.38, 0.62, 0.82]) g.fillRect(x - 0.015, y - 0.015, 0.03, 0.03);
    }

    // Heavy brass central latch hasp with padlock keyhole
    ik(g, rng, () => rrect(g, 0.43, 0.36, 0.14, 0.2, 0.02), '#d97706', 0.15, { hatch: 0.3, line: 0.014 });
    g.fillStyle = '#1e1b18';
    g.beginPath(); g.arc(0.5, 0.46, 0.03, 0, TAU); g.fill();
    g.fillRect(0.485, 0.46, 0.03, 0.05);

    // Forged side drop handles
    for (const [hx, hy] of [[0.06, 0.5], [0.94, 0.5]]) {
      g.strokeStyle = '#475569'; g.lineWidth = 0.03;
      g.beginPath(); g.arc(hx, hy, 0.05, -Math.PI / 2, Math.PI / 2, hx > 0.5); g.stroke();
    }
  },
  bed(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.06, 0.8, 1.88, 0.1), '#6b5d4a', 0.9, { hatch: 1, line: 0.024 });
    ik(g, rng, () => rrect(g, 0.16, 0.5, 0.68, 1.38, 0.08), '#7a3f3a', 0.7, { hatch: 1, line: 0.018 });
    for (let y = 0.7; y < 1.85; y += 0.2) stroke(g, [[0.2, y], [0.8, y]], 0.012, alpha('#e8d8b0', 0.45), 1);
    ik(g, rng, () => ell(g, 0.5, 0.3, 0.3, 0.15), '#a8a090', 0.3, { hatch: 0.5, line: 0.016 });
  },
  workbench(g, rng) {
    // Heavy carpenter's workbench with butcher-block maple top, bench dogs, and steel vise
    ik(g, rng, () => rrect(g, 0.05, 0.08, 1.9, 0.84, 0.04), '#78553a', 0.95, { hatch: 0.9, line: 0.024 });
    // Butcher-block top slats
    for (let i = 1; i < 6; i++) {
      stroke(g, [[0.05 + i * 0.317, 0.08], [0.05 + i * 0.317, 0.92]], 0.012, INK, 0.45);
    }

    // Cast-iron front vise with steel crank screw
    ik(g, rng, () => rrect(g, 0.12, 0.18, 0.34, 0.24, 0.03), '#334155', 0.25, { hatch: 0.8, line: 0.016 });
    stroke(g, [[0.08, 0.3], [0.02, 0.3]], 0.03, '#94a3b8', 1);

    // Bench dog holes
    for (let x = 0.55; x < 1.7; x += 0.3) {
      g.fillStyle = '#1e1b18';
      g.beginPath(); g.arc(x, 0.22, 0.03, 0, TAU); g.fill();
    }

    // Hand tools neatly placed on tabletop:
    // 1. Steel claw hammer with wooden hickory handle
    ik(g, rng, () => cap(g, 1.0, 0.65, 1.48, 0.65, 0.028, 0.022), '#92400e', 0.4, { hatch: 0.5, line: 0.014 });
    ik(g, rng, () => rrect(g, 1.48, 0.58, 0.15, 0.14, 0.02), '#64748b', 0.12, { hatch: 0.6, line: 0.014 });

    // 2. Architect's draft blueprint scroll
    ik(g, rng, () => poly(g, [[0.85, 0.3], [1.65, 0.35], [1.63, 0.46], [0.83, 0.41]]), '#0284c7', 0.4, { hatch: 0.5, line: 0.012 });
    stroke(g, [[0.95, 0.35], [1.5, 0.38]], 0.01, '#f0f9ff', 0.8);

    // 3. Woodworking hand plane with curled wood shaving
    ik(g, rng, () => rrect(g, 1.3, 0.72, 0.32, 0.14, 0.02), '#713f12', 0.18, { hatch: 0.5, line: 0.012 });
    g.strokeStyle = '#fef08a'; g.lineWidth = 0.015;
    g.beginPath(); g.arc(1.68, 0.78, 0.04, 0, TAU * 0.8); g.stroke();
  },
  torch_post(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.12, 0.12), '#5a4838', 0.14, { hatch: 0.9, line: 0.02 });
    ik(g, rng, () => lumpy(g, 0, 0, 0.1, 0.09, rng, 8, 0.15), '#3a342f', 0.12, { hatch: 0.8, line: 0.016 });
    for (const a of [0.5, 2.1, 3.7, 5.3]) ik(g, rng, () => cap(g, Math.cos(a) * 0.1, Math.sin(a) * 0.1, Math.cos(a) * 0.28, Math.sin(a) * 0.28, 0.035, 0.02), '#6a5645', 0.3, { hatch: 0.6, line: 0.012 });
    g.restore();
  },
  conveyor_belt(g, rng) {
    ik(g, rng, () => rrect(g, 0.05, 0.05, 0.9, 0.9, 0.06), '#32363c', 0.5, { hatch: 0.4, line: 0.02 });
    for (let x = 0.16; x < 0.9; x += 0.22) {
      ik(g, rng, () => rrect(g, x, 0.12, 0.14, 0.76, 0.02), '#23262a', 0.4, { hatch: 0.2, line: 0.012 });
      stroke(g, [[x + 0.02, 0.38], [x + 0.09, 0.5], [x + 0.02, 0.62]], 0.022, '#f0b429', 0.9);
    }
    stroke(g, [[0.06, 0.06], [0.94, 0.06]], 0.04, '#6b7280', 1);
    stroke(g, [[0.06, 0.94], [0.94, 0.94]], 0.04, '#6b7280', 1);
  },
  spring_inserter(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.3, 0.3), '#3c424a', 0.35, { hatch: 0.8, line: 0.02 });
    ik(g, rng, () => ell(g, 0, 0, 0.18, 0.18), '#606975', 0.2, { hatch: 0.4, line: 0.016 });
    stroke(g, [[-0.08, -0.06], [0.08, -0.02], [-0.08, 0.04], [0.08, 0.08]], 0.02, '#e09f3e', 0.9);
    stroke(g, [[0, 0], [0.38, -0.15]], 0.045, '#8d99ae', 1);
    ik(g, rng, () => rrect(g, 0.34, -0.25, 0.12, 0.2, 0.02), '#d90429', 0.15, { hatch: 0.2, line: 0.012 });
    g.restore();
  },
  burner_drill(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.1, 1.8, 1.8, 0.12), '#4a5058', 1.0, { hatch: 0.9, line: 0.028 });
    ik(g, rng, () => rrect(g, 0.25, 0.25, 1.5, 1.5, 0.08), '#33373d', 0.9, { hatch: 0.5, line: 0.02 });
    ik(g, rng, () => ell(g, 1.0, 1.0, 0.42, 0.42), '#6c757d', 0.45, { hatch: 0.8, line: 0.022 });
    ik(g, rng, () => ell(g, 1.0, 1.0, 0.22, 0.22), '#212529', 0.25, { hatch: 0.2, line: 0.016 });
    ik(g, rng, () => ell(g, 0.45, 0.45, 0.18, 0.18), '#2b2d42', 0.2, { hatch: 0.4, line: 0.018 });
    ik(g, rng, () => ell(g, 0.45, 0.45, 0.1, 0.1), '#12131a', 0.12, { hatch: 0, line: 0.01 });
    for (let a = 0; a < TAU; a += 0.8) {
      stroke(g, [[1.0 + Math.cos(a) * 0.28, 1.0 + Math.sin(a) * 0.28], [1.0 + Math.cos(a) * 0.42, 1.0 + Math.sin(a) * 0.42]], 0.03, '#ced4da', 0.9);
    }
  },
  stone_furnace(g, rng) {
    // Heavy ashlar stone masonry hearth with mortar joints
    ik(g, rng, () => rrect(g, 0.1, 0.1, 1.8, 1.8, 0.14), '#58524d', 1.0, { hatch: 1.1, line: 0.028 });

    // Mortar lines between cut stone blocks
    for (let y = 0.38; y < 1.7; y += 0.36) {
      stroke(g, [[0.15, y], [1.85, y]], 0.02, alpha('#e2e8f0', 0.5), 0.8);
      // Staggered vertical joints
      const offset = (Math.round(y * 10) % 2) * 0.4;
      for (let x = 0.35 + offset; x < 1.7; x += 0.65) {
        stroke(g, [[x, y - 0.36], [x, y]], 0.018, INK, 0.6);
      }
    }

    // Cast iron chimney flue collar on top
    ik(g, rng, () => ell(g, 1.0, 0.32, 0.36, 0.22), '#33373d', 0.4, { hatch: 0.8, line: 0.02 });
    ik(g, rng, () => ell(g, 1.0, 0.32, 0.2, 0.12), '#121316', 0.2, { hatch: 0, line: 0.01 });

    // Arched firebox hearth opening
    ik(g, rng, () => rrect(g, 0.46, 0.72, 1.08, 0.94, 0.22), '#1c130d', 0.65, { hatch: 0.4, line: 0.024 });

    // Glowing forge embers & orange incandescent coals
    ik(g, rng, () => ell(g, 1.0, 1.25, 0.42, 0.32), '#ea580c', 0.45, { hatch: 0.1, line: 0.018 });
    ik(g, rng, () => ell(g, 1.0, 1.28, 0.28, 0.2), '#facc15', 0.3, { hatch: 0, line: 0.012 });
    ik(g, rng, () => ell(g, 1.0, 1.3, 0.14, 0.09), '#fef08a', 0.15, { hatch: 0, line: 0 });

    // Cast iron hearth grate bars
    for (let x = 0.62; x <= 1.38; x += 0.18) {
      stroke(g, [[x, 0.95], [x, 1.55]], 0.032, '#1e293b', 0.95);
    }
  },
  grand_banquet_table(g, rng) {
    ik(g, rng, () => rrect(g, 0.08, 0.15, 1.84, 0.7, 0.04), '#74543d', 0.9, { hatch: 0.9, line: 0.024 });
    ik(g, rng, () => rrect(g, 0.25, 0.2, 0.45, 0.6, 0.02), '#9c3d34', 0.4, { hatch: 0.3, line: 0.014 });
    ik(g, rng, () => rrect(g, 1.3, 0.2, 0.45, 0.6, 0.02), '#9c3d34', 0.4, { hatch: 0.3, line: 0.014 });
    ik(g, rng, () => ell(g, 1.0, 0.5, 0.16, 0.14), '#2b2d33', 0.2, { hatch: 0.4, line: 0.016 });
    ik(g, rng, () => ell(g, 1.0, 0.5, 0.1, 0.08), '#e07a34', 0.12, { hatch: 0, line: 0.01 });
  },
  medical_triage_cot(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.06, 0.8, 1.88, 0.08), '#adb5bd', 0.85, { hatch: 0.6, line: 0.022 });
    ik(g, rng, () => rrect(g, 0.16, 0.45, 0.68, 1.42, 0.06), '#edf2f4', 0.75, { hatch: 0.4, line: 0.016 });
    ik(g, rng, () => ell(g, 0.5, 0.28, 0.28, 0.16), '#f8f9fa', 0.3, { hatch: 0.2, line: 0.014 });
    stroke(g, [[0.45, 0.28], [0.55, 0.28]], 0.03, '#d90429', 1);
    stroke(g, [[0.5, 0.23], [0.5, 0.33]], 0.03, '#d90429', 1);
    ik(g, rng, () => ell(g, 0.18, 0.18, 0.06, 0.06), '#495057', 0.1, { hatch: 0.5, line: 0.014 });
    ik(g, rng, () => ell(g, 0.18, 0.18, 0.035, 0.05), '#e63946', 0.08, { hatch: 0, line: 0.01 });
  },
  horseshoe_pin_post(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.4, 0.36), '#b59b7a', 0.45, { hatch: 0.4, line: 0.016 });
    ik(g, rng, () => ell(g, 0, 0, 0.07, 0.07), '#212529', 0.1, { hatch: 0.5, line: 0.018 });
    for (const [x, y, a] of [[-0.18, -0.1, 0.4], [0.16, 0.12, 2.1], [-0.05, 0.18, -0.8]] as [number, number, number][]) {
      g.save(); g.translate(x, y); g.rotate(a);
      g.beginPath(); g.arc(0, 0, 0.08, 0.4, Math.PI - 0.4);
      g.strokeStyle = '#d4af37'; g.lineWidth = 0.024; g.stroke();
      g.restore();
    }
    g.restore();
  },
  colony_overseer_desk(g, rng) {
    ik(g, rng, () => rrect(g, 0.06, 0.1, 1.88, 0.8, 0.05), '#4a3728', 0.95, { hatch: 0.9, line: 0.026 });
    ik(g, rng, () => rrect(g, 0.45, 0.22, 1.1, 0.56, 0.02), '#25201c', 0.6, { hatch: 0.4, line: 0.016 });
    ik(g, rng, () => rrect(g, 0.65, 0.3, 0.4, 0.38, 0.02), '#6c757d', 0.25, { hatch: 0.6, line: 0.016 });
    ik(g, rng, () => rrect(g, 1.25, 0.32, 0.26, 0.34, 0.01), '#f8f9fa', 0.2, { hatch: 0.2, line: 0.012 });
    ik(g, rng, () => ell(g, 0.22, 0.32, 0.09, 0.09), '#d4af37', 0.12, { hatch: 0.3, line: 0.014 });
    ik(g, rng, () => ell(g, 0.22, 0.32, 0.05, 0.05), '#ffee88', 0.08, { hatch: 0, line: 0.01 });
  },
  belt_bolter(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.38, 0.38), '#343a40', 0.8, { hatch: 0.6, line: 0.02 });
    ik(g, rng, () => rrect(g, -0.08, -0.42, 0.16, 0.5, 0.02), '#495057', 0.6, { hatch: 0.4, line: 0.016 });
    stroke(g, [[-0.34, -0.25], [0.34, -0.25]], 0.034, '#6c757d', 0.9);
    ik(g, rng, () => rrect(g, 0.12, -0.12, 0.22, 0.24, 0.02), '#8a5a36', 0.4, { hatch: 0.3, line: 0.014 });
    g.restore();
  },
  flame_turret(g, rng) {
    g.save(); g.translate(0.5, 0.5);
    ik(g, rng, () => ell(g, 0, 0, 0.42, 0.42), '#b23a22', 0.85, { hatch: 0.7, line: 0.022 });
    ik(g, rng, () => rrect(g, -0.1, -0.46, 0.2, 0.46, 0.04), '#212529', 0.6, { hatch: 0.3, line: 0.016 });
    ik(g, rng, () => ell(g, 0, -0.48, 0.08, 0.08), '#f77f00', 0.25, { hatch: 0, line: 0.01 });
    ik(g, rng, () => ell(g, 0, -0.48, 0.04, 0.04), '#fcbf49', 0.15, { hatch: 0, line: 0.01 });
    g.restore();
  },
  zap_fence(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.15, 0.8, 0.7, 0.04), '#2b2d42', 0.7, { hatch: 0.4, line: 0.02 });
    stroke(g, [[0.15, 0.25], [0.85, 0.25]], 0.03, '#48cae4', 0.9);
    stroke(g, [[0.15, 0.5], [0.85, 0.5]], 0.03, '#48cae4', 0.9);
    stroke(g, [[0.15, 0.75], [0.85, 0.75]], 0.03, '#48cae4', 0.9);
    ik(g, rng, () => ell(g, 0.5, 0.5, 0.15, 0.15), '#90e0ef', 0.2, { hatch: 0, line: 0.014 });
  },
  mallet_catapult(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.1, 1.8, 1.8, 0.08), '#5a422d', 0.9, { hatch: 0.8, line: 0.028 });
    ik(g, rng, () => ell(g, 1.0, 1.0, 0.38, 0.38), '#2b2d33', 0.5, { hatch: 0.4, line: 0.02 });
    stroke(g, [[1.0, 1.0], [0.35, 0.35]], 0.06, '#74543d', 0.9);
    ik(g, rng, () => rrect(g, 0.15, 0.15, 0.45, 0.35, 0.04), '#3e2a1d', 0.4, { hatch: 0.5, line: 0.02 });
  },
  repair_drone_bay(g, rng) {
    ik(g, rng, () => rrect(g, 0.1, 0.1, 1.8, 1.8, 0.12), '#343a40', 0.95, { hatch: 0.8, line: 0.026 });
    ik(g, rng, () => ell(g, 1.0, 1.0, 0.65, 0.65), '#495057', 0.6, { hatch: 0.5, line: 0.02 });
    stroke(g, [[0.7, 1.0], [1.3, 1.0]], 0.05, '#06d6a0', 1);
    stroke(g, [[1.0, 0.7], [1.0, 1.3]], 0.05, '#06d6a0', 1);
    ik(g, rng, () => ell(g, 1.0, 1.0, 0.22, 0.22), '#adb5bd', 0.3, { hatch: 0.2, line: 0.014 });
  },
  clan_territory_totem(g, rng) {
    ik(g, rng, () => rrect(g, 0.12, 0.12, 1.76, 1.76, 0.14), '#3a322c', 1.0, { hatch: 1.1, line: 0.03 });
    ik(g, rng, () => poly(g, [[0.4, 1.6], [1.0, 0.35], [1.6, 1.6]]), '#26211d', 0.8, { hatch: 0.7, line: 0.022 });
    ik(g, rng, () => ell(g, 1.0, 0.32, 0.28, 0.28), '#f9c74f', 0.4, { hatch: 0.2, line: 0.018 });
    ik(g, rng, () => ell(g, 1.0, 0.32, 0.16, 0.16), '#90e0ef', 0.25, { hatch: 0, line: 0.012 });
    stroke(g, [[1.0, 0.8], [1.5, 0.8], [1.4, 1.1], [1.0, 1.1]], 0.03, '#d90429', 0.8);
  },
  gunsmith_bench(g, rng) {
    ik(g, rng, () => rrect(g, 0.05, 0.1, 1.9, 0.8, 0.04), '#334155', 0.9, { hatch: 1, line: 0.024 });
    for (let i = 1; i < 6; i++) stroke(g, [[0.05 + i * 0.317, 0.1], [0.05 + i * 0.317, 0.9]], 0.01, INK, 0.5);
    ik(g, rng, () => rrect(g, 0.15, 0.22, 0.35, 0.56, 0.03), '#1e293b', 0.4, { hatch: 0.8, line: 0.016 });
    stroke(g, [[0.7, 0.5], [1.7, 0.5]], 0.035, '#94a3b8', 0.9);
    stroke(g, [[1.1, 0.5], [1.1, 0.7]], 0.02, '#94a3b8', 0.9);
    ik(g, rng, () => ell(g, 1.5, 0.3, 0.1, 0.06), '#facc15', 0.1, { hatch: 0.5, line: 0.012 });
  },
};
void shade;
