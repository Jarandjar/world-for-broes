import { hash2 } from './rng';
/** Deterministic 2D gradient noise in roughly [0,1]. */
export function makeNoise(seed: number): (x: number, y: number) => number {
  const grad = (ix: number, iy: number, dx: number, dy: number): number => {
    const h = hash2(seed, ix, iy) & 7;
    const gx = h < 4 ? (h & 1 ? 1 : -1) : (h & 1 ? 0.7071 : -0.7071);
    const gy = h < 4 ? (h & 2 ? 1 : -1) * (h < 2 ? 0 : 1) : (h & 2 ? 0.7071 : -0.7071);
    return gx * dx + gy * dy;
  };
  const fade = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10);
  return (x, y) => {
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    const u = fade(fx), v = fade(fy);
    const n00 = grad(x0, y0, fx, fy), n10 = grad(x0 + 1, y0, fx - 1, fy);
    const n01 = grad(x0, y0 + 1, fx, fy - 1), n11 = grad(x0 + 1, y0 + 1, fx - 1, fy - 1);
    const a = n00 + (n10 - n00) * u, b = n01 + (n11 - n01) * u;
    return 0.5 + (a + (b - a) * v) * 0.9;
  };
}
export function fbm(noise: (x: number, y: number) => number, x: number, y: number, octaves = 4, lac = 2, gain = 0.5): number {
  let amp = 1, freq = 1, sum = 0, norm = 0;
  for (let i = 0; i < octaves; i++) { sum += amp * noise(x * freq, y * freq); norm += amp; amp *= gain; freq *= lac; }
  return sum / norm;
}
