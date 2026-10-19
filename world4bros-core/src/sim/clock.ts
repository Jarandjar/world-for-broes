import { smoothstep } from '../engine/math';
export const DAY_SECONDS = 480;
/** 0 = full daylight, 0.9 = deep night. t is the fraction of the day in [0,1). */
export function darknessAt(t: number): number {
  const tp = t < 0.06 ? t + 1 : t;
  return 0.9 * smoothstep(0.52, 0.68, tp) * (1 - smoothstep(0.92, 1.06, tp));
}
export class Clock {
  /** Fraction of the current day. Starts mid-morning. */
  t = 0.22;
  day = 1;
  alwaysDay = false;

  advance(dt: number): void {
    if (this.alwaysDay) {
      this.t = 0.25;
      return;
    }
    this.t += dt / DAY_SECONDS;
    while (this.t >= 1) { this.t -= 1; this.day++; }
  }

  get darkness(): number {
    return this.alwaysDay ? 0 : darknessAt(this.t);
  }

  get phase(): 'dawn' | 'day' | 'dusk' | 'night' {
    if (this.alwaysDay) return 'day';
    if (this.t < 0.06) return 'dawn';
    if (this.t < 0.52) return 'day';
    if (this.t < 0.68) return 'dusk';
    return this.t < 0.92 ? 'night' : 'dawn';
  }
}
