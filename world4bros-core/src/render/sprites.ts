import type { G } from './ink';
export interface Sprite { canvas: HTMLCanvasElement; ox: number; oy: number; ppu: number; w: number; h: number }
export type SpriteQuality = 'off' | 'low' | 'high';

/** Pixel budget per quality (about 4 bytes each). Oldest unused sprites are dropped past it. */
const PIXEL_BUDGET: Record<SpriteQuality, number> = { off: 16e6, low: 24e6, high: 48e6 };

/** Renders procedural drawings once into small canvases, then reuses them every frame (LRU by pixel budget). */
export class SpriteCache {
  private map = new Map<string, Sprite>();
  private pixels = 0;
  get size(): number { return this.map.size; }
  clear(): void { this.map.clear(); this.pixels = 0; }
  /** Draw in world units inside the box (x0,y0)-(x1,y1). */
  get(key: string, ppu: number, x0: number, y0: number, x1: number, y1: number, draw: (g: G) => void): Sprite {
    const k = `${key}@${ppu}`;
    let s = this.map.get(k);
    if (s) { this.map.delete(k); this.map.set(k, s); return s; } // refresh recency
    const w = Math.max(2, Math.ceil((x1 - x0) * ppu)), h = Math.max(2, Math.ceil((y1 - y0) * ppu));
    const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
    const g = canvas.getContext('2d') as G;
    g.scale(ppu, ppu); g.translate(-x0, -y0);
    draw(g);
    s = { canvas, ox: -x0 * ppu, oy: -y0 * ppu, ppu, w, h };
    this.map.set(k, s); this.pixels += w * h;
    const budget = PIXEL_BUDGET[spriteQuality];
    if (this.pixels > budget) for (const [ok, os] of this.map) {
      if (this.pixels <= budget * 0.8 || ok === k) break;
      this.map.delete(ok); this.pixels -= os.w * os.h;
    }
    return s;
  }
}
/** Draw a sprite so its origin lands at (x,y) in the current unit space. */
export function blit(g: G, s: Sprite, x = 0, y = 0, rot = 0, sx = 1, sy = 1, a = 1): void {
  g.save(); g.translate(x, y); if (rot) g.rotate(rot); if (sx !== 1 || sy !== 1) g.scale(sx, sy); if (a !== 1) g.globalAlpha *= a;
  g.drawImage(s.canvas, -s.ox / s.ppu, -s.oy / s.ppu, s.w / s.ppu, s.h / s.ppu);
  g.restore();
}

let spriteQuality: SpriteQuality = 'high';
/** Set by the renderer from the graphics quality dial. */
export function setSpriteQuality(q: SpriteQuality): void { spriteQuality = q; }
/** Sprite resolution tiers (px per world unit). High adds a crisp 288 tier for close-ups on hi-dpi screens; low caps at 144. */
export const PPU_TIERS: Record<SpriteQuality, readonly number[]> = { off: [48, 96], low: [56, 96, 144], high: [56, 96, 144, 208, 288] };
export function ppuTier(px: number, q: SpriteQuality): number {
  const tiers = PPU_TIERS[q];
  for (const t of tiers) if (px <= t * 1.04) return t;
  return tiers[tiers.length - 1]!;
}
export const ppuFor = (pxPerUnit: number): number => ppuTier(pxPerUnit * (globalThis.devicePixelRatio || 1), spriteQuality);
