import { createNoise2D } from 'simplex-noise';

export function makeSeededRandom(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export class TerrainSampler {
  private readonly broad: ReturnType<typeof createNoise2D>;
  private readonly rolling: ReturnType<typeof createNoise2D>;
  private readonly detail: ReturnType<typeof createNoise2D>;

  constructor(seed: number) {
    this.broad = createNoise2D(makeSeededRandom(seed ^ 0x2f6e2b1));
    this.rolling = createNoise2D(makeSeededRandom(seed ^ 0x5a17c9e));
    this.detail = createNoise2D(makeSeededRandom(seed ^ 0x14d3a61));
  }

  sampleHeight(x: number, z: number): number {
    const broad = this.broad(x * 0.016, z * 0.016);
    const rolling = this.rolling(x * 0.045 + 31, z * 0.045 - 17);
    const detail = this.detail(x * 0.11 - 83, z * 0.11 + 59);
    return (broad * 0.68 + rolling * 0.24 + detail * 0.08) * 1.15;
  }
}

export function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function seededValue(seed: number, x: number, z: number, salt: number): number {
  let value = Math.imul((Math.floor(x * 1000) ^ seed) | 0, 0x45d9f3b);
  value = Math.imul((value ^ Math.floor(z * 1000) ^ salt) | 0, 0x45d9f3b);
  value ^= value >>> 16;
  value = Math.imul(value, 0x45d9f3b);
  value ^= value >>> 16;
  return (value >>> 0) / 4294967296;
}
