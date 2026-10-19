/**
 * Tween & Spring Preset Library for World 4 Bro's Animation Engine (Phase Z0C).
 * Based on Section 62:
 * - TweenPreset: linear, quadEaseIn, quadEaseOut, quadEaseInOut, elasticOut, bounceOut, snapPunch
 * - SpringPreset: snappy (f=5, zeta=0.7), bouncy (f=3, zeta=0.4), heavy (f=2, zeta=0.9), loose (f=1.5, zeta=0.5)
 */

export type EasingFunction = (t: number) => number;

export const TweenPreset: Record<string, EasingFunction> = {
  linear: (t: number): number => t,

  quadEaseIn: (t: number): number => t * t,

  quadEaseOut: (t: number): number => t * (2 - t),

  quadEaseInOut: (t: number): number => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),

  elasticOut: (t: number): number => {
    if (t === 0 || t === 1) return t;
    const p = 0.3;
    const s = p / 4;
    return Math.pow(2, -10 * t) * Math.sin(((t - s) * (2 * Math.PI)) / p) + 1;
  },

  bounceOut: (t: number): number => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      return n1 * (t -= 1.5 / d1) * t + 0.75;
    } else if (t < 2.5 / d1) {
      return n1 * (t -= 2.25 / d1) * t + 0.9375;
    } else {
      return n1 * (t -= 2.625 / d1) * t + 0.984375;
    }
  },

  snapPunch: (t: number): number => {
    // Sharp anticipation inward (-0.15) then snappy overshoot (1.2) before resting at 1.0
    if (t < 0.25) return -0.15 * Math.sin((t / 0.25) * Math.PI);
    const sub = (t - 0.25) / 0.75;
    return 1 + Math.sin(sub * Math.PI) * 0.25 * (1 - sub);
  },
};

export interface SpringConfig {
  frequency: number; // Hz (oscillation cycles per second)
  dampingRatio: number; // zeta: 1.0 = critically damped, <1.0 = underdamped/bouncy, >1.0 = overdamped
}

export const SpringPreset: Record<'snappy' | 'bouncy' | 'heavy' | 'loose', SpringConfig> = {
  snappy: { frequency: 5.0, dampingRatio: 0.7 },
  bouncy: { frequency: 3.0, dampingRatio: 0.4 },
  heavy: { frequency: 2.0, dampingRatio: 0.9 },
  loose: { frequency: 1.5, dampingRatio: 0.5 },
};

/**
 * Second-Order Harmonic Dynamics Spring System.
 * Exact analytical solution for physics-driven secondary motion (hats, tails, ears, backpack bob).
 */
export class DampedSpring {
  public value: number = 0;
  public velocity: number = 0;
  public target: number = 0;

  private f: number;
  private z: number;

  constructor(preset: SpringConfig = SpringPreset.snappy, initialValue: number = 0) {
    this.f = preset.frequency;
    this.z = preset.dampingRatio;
    this.value = initialValue;
    this.target = initialValue;
  }

  public setConfig(config: SpringConfig) {
    this.f = config.frequency;
    this.z = config.dampingRatio;
  }

  public reset(value: number = 0) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
  }

  /**
   * Advances the second-order spring by dt seconds.
   */
  public update(dt: number, target: number = this.target): number {
    this.target = target;
    const clampedDt = Math.min(dt, 0.1); // Delta clamp to prevent simulation blowout

    // Semi-implicit Euler integration for stability
    const omega = 2 * Math.PI * this.f;
    const k1 = this.z / (Math.PI * this.f);
    const k2 = 1 / (omega * omega);

    const fSpring = (this.target - this.value) / k2;
    const fDamp = (this.velocity * k1) / k2;
    const accel = fSpring - fDamp;

    this.velocity += accel * clampedDt;
    this.value += this.velocity * clampedDt;

    return this.value;
  }
}
