import { smoothstep } from '../engine/math';
import { parse } from './ink';
export type Light = { x: number; y: number; radius: number; color: string; flicker: number };
export type LightQuality = 'off' | 'low' | 'high';

/** Darkness buffer scale per quality (smaller = cheaper, softer). */
const DARK_SCALE: Record<LightQuality, number> = { off: 0.33, low: 0.4, high: 0.5 };
/** Night tint (STYLE §3.2: obsidian violet, never pure black). */
const NIGHT_RGB = '9,8,22';

/** Night darkness with soft holes cut by light sources, plus a warm glow. Light falloff is a cached sprite, not a per-light gradient. */
export class Lighting {
  private dark = document.createElement('canvas');
  private dg = this.dark.getContext('2d') as CanvasRenderingContext2D;
  private hole: HTMLCanvasElement | null = null;
  private glows = new Map<string, HTMLCanvasElement>();

  private sprite(stops: [number, string][], size = 128): HTMLCanvasElement {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const g = c.getContext('2d') as CanvasRenderingContext2D, h = size / 2, gr = g.createRadialGradient(h, h, 0, h, h, h);
    for (const [o, col] of stops) gr.addColorStop(o, col);
    g.fillStyle = gr; g.fillRect(0, 0, size, size); return c;
  }
  private holeSprite(): HTMLCanvasElement {
    return this.hole ??= this.sprite([[0, 'rgba(0,0,0,1)'], [0.05, 'rgba(0,0,0,1)'], [0.55, 'rgba(0,0,0,0.8)'], [0.8, 'rgba(0,0,0,0.3)'], [1, 'rgba(0,0,0,0)']]);
  }
  private glowSprite(color: string): HTMLCanvasElement {
    let c = this.glows.get(color);
    if (!c) { const [r, g, b] = parse(color); c = this.sprite([[0, `rgba(${r},${g},${b},1)`], [0.35, `rgba(${r},${g},${b},0.45)`], [1, `rgba(${r},${g},${b},0)`]], 64); this.glows.set(color, c); }
    return c;
  }

  render(g: CanvasRenderingContext2D, W: number, H: number, camX: number, camY: number, z: number, lights: Light[], darkness: number, dayT: number, t: number, quality: LightQuality = 'high'): void {
    // dusk and dawn warm tint
    const dusk = smoothstep(0.5, 0.6, dayT) * (1 - smoothstep(0.62, 0.7, dayT)), dawn = 1 - smoothstep(0.0, 0.12, dayT);
    const warm = Math.max(dusk, dawn * 0.7);
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0);
    if (warm > 0.01) { g.globalCompositeOperation = 'multiply'; g.fillStyle = `rgba(255,${200 - warm * 40},${170 - warm * 50},${0.35 * warm})`; g.fillRect(0, 0, W, H); g.globalCompositeOperation = 'source-over'; }
    if (darkness > 0.01) {
      const s = DARK_SCALE[quality], dw = Math.max(2, Math.floor(W * s)), dh = Math.max(2, Math.floor(H * s));
      if (this.dark.width !== dw || this.dark.height !== dh) { this.dark.width = dw; this.dark.height = dh; }
      const d = this.dg, hole = this.holeSprite(); d.setTransform(1, 0, 0, 1, 0, 0); d.globalCompositeOperation = 'source-over';
      d.clearRect(0, 0, dw, dh); d.fillStyle = `rgba(${NIGHT_RGB},${Math.min(0.95, darkness * 1.05)})`; d.fillRect(0, 0, dw, dh);
      d.globalCompositeOperation = 'destination-out';
      const visible = (x: number, y: number, r: number, w: number, h: number): boolean => x + r > 0 && y + r > 0 && x - r < w && y - r < h;
      for (const l of lights) {
        const fl = 1 + Math.sin(t * 11 + l.x * 3) * l.flicker * 0.5 + Math.sin(t * 17.3 + l.y) * l.flicker * 0.3;
        const r = l.radius * z * s * fl, x = (W / 2 + (l.x - camX) * z) * s, y = (H / 2 + (l.y - camY) * z) * s;
        if (r < 0.5 || !visible(x, y, r, dw, dh)) continue;
        d.drawImage(hole, x - r, y - r, r * 2, r * 2);
      }
      g.imageSmoothingEnabled = true; g.drawImage(this.dark, 0, 0, W, H);
      if (quality !== 'off') {
        g.globalCompositeOperation = 'lighter';
        const glowA = (quality === 'high' ? 0.2 : 0.14) * Math.min(1, darkness + 0.2);
        for (const l of lights) {
          const fl = 1 + Math.sin(t * 11 + l.x * 3) * l.flicker * 0.5;
          const r = l.radius * z * 0.9 * fl, x = W / 2 + (l.x - camX) * z, y = H / 2 + (l.y - camY) * z;
          if (r < 1 || !visible(x, y, r, W, H)) continue;
          g.globalAlpha = glowA; g.drawImage(this.glowSprite(l.color), x - r, y - r, r * 2, r * 2);
        }
        g.globalAlpha = 1;
      }
    }
    g.restore();
  }
}
