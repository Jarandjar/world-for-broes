import { AnimationClip } from './AnimationClip';
import { AnimationPose, clonePose, lerpPose, applyMaskedPose, createEmptyPose } from './AnimationPose';
import { CrossfadeController } from './BlendTree';
import { AnimationEventManager } from './AnimationEvent';

export const UPPER_BODY_MASK = new Set<string>([
  'torso',
  'head',
  'leftArm',
  'rightArm',
  'heldTool',
  'backpack',
]);

export interface LayerState {
  clip: AnimationClip | null;
  weight: number;
  time: number;
  speed: number;
}

/**
 * 4-Tier Layered State Machine:
 * - Layer 0 (Base): Full-body locomotion (idle, walk, run, sprint, sneak)
 * - Layer 1 (Upper Body): Decoupled action (chop, throw, eat, honk) masked onto upper body
 * - Layer 2 (Additive): Procedural secondary motion (breathing, wind sway)
 * - Layer 3 (Override): Full-body interrupts (hurt, down, get_up, dance)
 */
export class LayeredStateMachine {
  public baseController: CrossfadeController = new CrossfadeController();
  public upperController: CrossfadeController = new CrossfadeController();
  public overrideController: CrossfadeController = new CrossfadeController();
  public eventManager: AnimationEventManager = new AnimationEventManager();

  public upperWeight: number = 0; // 0 = inactive, 1 = fully active
  public overrideWeight: number = 0; // 0 = inactive, 1 = fully active

  public baseSpeedMultiplier: number = 1.0;
  public upperSpeedMultiplier: number = 1.0;

  // Stored current pose for rendering interpolation
  public currentPose: AnimationPose = createEmptyPose();
  public previousPose: AnimationPose = createEmptyPose();

  /**
   * Sets base locomotion clip with smooth crossfade.
   */
  public playBase(clip: AnimationClip, crossfadeDuration: number = 0.12) {
    this.baseController.play(clip, crossfadeDuration);
  }

  /**
   * Triggers an upper-body action (chop, eat, throw, honk).
   * Upper body executes while legs continue running.
   */
  public playUpper(clip: AnimationClip, crossfadeDuration: number = 0.08) {
    this.upperController.play(clip, crossfadeDuration);
    this.upperWeight = 1.0;
  }

  public stopUpper(fadeDuration: number = 0.10) {
    this.upperWeight = 0;
  }

  /**
   * Plays a high-priority full-body override interrupt (hurt, down, dance).
   */
  public playOverride(clip: AnimationClip, crossfadeDuration: number = 0.05) {
    this.overrideController.play(clip, crossfadeDuration);
    this.overrideWeight = 1.0;
  }

  public stopOverride() {
    this.overrideWeight = 0;
  }

  /**
   * Fixed Timestep Update (Runs on 60 Hz physics tick).
   */
  public update(dt: number): AnimationPose {
    this.previousPose = clonePose(this.currentPose);

    // 1. Evaluate Base Locomotion Layer
    const baseDt = dt * this.baseSpeedMultiplier;
    const prevBaseTime = this.baseController.currentTime;
    const basePose = this.baseController.update(baseDt);
    if (this.baseController.currentClip) {
      this.eventManager.update(this.baseController.currentClip, prevBaseTime, this.baseController.currentTime);
    }

    let combinedPose = basePose;

    // 2. Evaluate Upper Body Masked Layer
    if (this.upperWeight > 0 && this.upperController.currentClip) {
      const upperDt = dt * this.upperSpeedMultiplier;
      const prevUpperTime = this.upperController.currentTime;
      const upperPose = this.upperController.update(upperDt);

      this.eventManager.update(this.upperController.currentClip, prevUpperTime, this.upperController.currentTime);

      combinedPose = applyMaskedPose(combinedPose, upperPose, UPPER_BODY_MASK, this.upperWeight);

      // Auto-stop non-looping upper body clips when completed
      if (!this.upperController.currentClip.loop && this.upperController.currentTime >= this.upperController.currentClip.duration) {
        this.upperWeight = 0;
      }
    }

    // 3. Evaluate Override Full-Body Interrupt Layer
    if (this.overrideWeight > 0 && this.overrideController.currentClip) {
      const overridePose = this.overrideController.update(dt);
      combinedPose = lerpPose(combinedPose, overridePose, this.overrideWeight);

      if (!this.overrideController.currentClip.loop && this.overrideController.currentTime >= this.overrideController.currentClip.duration) {
        this.overrideWeight = 0;
      }
    }

    this.currentPose = combinedPose;
    return this.currentPose;
  }

  /**
   * Returns interpolated pose between previous fixed tick and current fixed tick by alpha in [0, 1].
   */
  public getInterpolatedPose(alpha: number): AnimationPose {
    return lerpPose(this.previousPose, this.currentPose, alpha);
  }
}
