import { AnimationClip } from './AnimationClip';
import { AnimationPose, lerpPose, createEmptyPose } from './AnimationPose';

/**
 * 1D Linear Blend Node.
 * Blends between two clips (e.g., walk and run) based on a scalar parameter in [0, 1].
 */
export class LinearBlend1D {
  public clipA: AnimationClip;
  public clipB: AnimationClip;
  public parameter: number = 0; // 0.0 = 100% Clip A, 1.0 = 100% Clip B

  constructor(clipA: AnimationClip, clipB: AnimationClip) {
    this.clipA = clipA;
    this.clipB = clipB;
  }

  public sample(time: number, param: number = this.parameter): AnimationPose {
    this.parameter = Math.max(0, Math.min(1, param));
    const poseA = this.clipA.sample(time);
    const poseB = this.clipB.sample(time);
    return lerpPose(poseA, poseB, this.parameter);
  }
}

export interface CrossfadeTransition {
  fromClip: AnimationClip;
  toClip: AnimationClip;
  fromTime: number;
  toTime: number;
  duration: number; // in seconds (e.g. 0.12 = 120ms)
  elapsed: number;
}

/**
 * Crossfade Controller.
 * Manages smooth crossfades between clips with zero visual joint pops.
 */
export class CrossfadeController {
  public currentClip: AnimationClip | null = null;
  public currentTime: number = 0;

  private activeTransition: CrossfadeTransition | null = null;

  public play(clip: AnimationClip, crossfadeDuration: number = 0.12) {
    if (this.currentClip === clip && !this.activeTransition) {
      return;
    }

    if (!this.currentClip || crossfadeDuration <= 0) {
      this.currentClip = clip;
      this.currentTime = 0;
      this.activeTransition = null;
      return;
    }

    // Start crossfade transition
    this.activeTransition = {
      fromClip: this.currentClip,
      toClip: clip,
      fromTime: this.currentTime,
      toTime: 0,
      duration: Math.max(0.01, crossfadeDuration),
      elapsed: 0,
    };

    this.currentClip = clip;
    this.currentTime = 0;
  }

  public update(dt: number): AnimationPose {
    if (!this.currentClip) {
      return createEmptyPose();
    }

    if (this.activeTransition) {
      this.activeTransition.elapsed += dt;
      this.activeTransition.fromTime += dt;
      this.activeTransition.toTime += dt;
      this.currentTime = this.activeTransition.toTime;

      const progress = Math.min(1.0, this.activeTransition.elapsed / this.activeTransition.duration);

      // Smoothstep crossfade easing: S(x) = 3x^2 - 2x^3
      const smoothAlpha = progress * progress * (3 - 2 * progress);

      const poseFrom = this.activeTransition.fromClip.sample(this.activeTransition.fromTime);
      const poseTo = this.activeTransition.toClip.sample(this.activeTransition.toTime);

      if (progress >= 1.0) {
        this.activeTransition = null;
      }

      return lerpPose(poseFrom, poseTo, smoothAlpha);
    } else {
      this.currentTime += dt;
      return this.currentClip.sample(this.currentTime);
    }
  }

  public isTransitioning(): boolean {
    return this.activeTransition !== null;
  }
}
