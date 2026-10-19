import * as THREE from 'three';
import { DampedSpring, SpringPreset } from './TweenPreset';
import { AnimationPose } from './AnimationPose';

export interface ProceduralConfig {
  strideLength: number; // in meters (distance traveled per full walk/run cycle)
  maxLookAtAngleRad: number; // max head turn angle (e.g. Math.PI / 4 = 45 deg)
  leanFactor: number; // torso tilt multiplier on linear acceleration and turns
}

export const DEFAULT_PROCEDURAL_CONFIG: ProceduralConfig = {
  strideLength: 1.8, // 1.8 meters per 2-step cycle
  maxLookAtAngleRad: Math.PI / 4,
  leanFactor: 0.08,
};

/**
 * Procedural Layer.
 * Evaluates stride matching (zero foot sliding), head look-at IK, torso lean into turns,
 * and secondary harmonic springs on equipment (backpack, head bob).
 */
export class ProceduralLayer {
  public config: ProceduralConfig;

  // Secondary motion springs
  public headSpring: DampedSpring = new DampedSpring(SpringPreset.snappy);
  public backpackSpring: DampedSpring = new DampedSpring(SpringPreset.heavy);

  // Look-at state
  public currentLookAtAngle: number = 0;
  public targetLookAtAngle: number = 0;

  // Torso lean state
  public currentLean: number = 0;
  public targetLean: number = 0;

  constructor(config: Partial<ProceduralConfig> = {}) {
    this.config = { ...DEFAULT_PROCEDURAL_CONFIG, ...config };
  }

  /**
   * Stride Matching:
   * Computes the exact playback speed multiplier required for ground speed `velocityMagnitude`
   * such that feet never slide on the ground.
   * Speed multiplier = groundSpeed / (strideLength / nominalDuration)
   */
  public calculateStrideSpeed(velocityMagnitude: number, nominalDuration: number = 0.65): number {
    if (velocityMagnitude <= 0.05) return 1.0;
    const nominalSpeed = this.config.strideLength / nominalDuration;
    const multiplier = velocityMagnitude / nominalSpeed;
    // Clamp to reasonable ranges to prevent hyper-speed or complete freeze
    return Math.max(0.4, Math.min(2.2, multiplier));
  }

  /**
   * Updates procedural values on fixed tick.
   */
  public update(
    dt: number,
    velocity: THREE.Vector3,
    facingSign: number,
    turnAngularVelocity: number = 0,
    lookAtPoint?: THREE.Vector3,
    playerPos?: THREE.Vector3
  ) {
    // 1. Torso Lean into linear speed & angular turns
    const speed = velocity.length();
    const forwardLean = Math.min(speed * 0.025, 0.18);
    const turnLean = turnAngularVelocity * this.config.leanFactor;
    this.targetLean = (forwardLean + turnLean) * facingSign;
    this.currentLean += (this.targetLean - this.currentLean) * Math.min(1.0, dt * 10);

    // 2. Look-At IK Angle
    if (lookAtPoint && playerPos) {
      const dx = lookAtPoint.x - playerPos.x;
      const dz = lookAtPoint.z - playerPos.z;
      const angle = Math.atan2(dx, dz);
      this.targetLookAtAngle = Math.max(
        -this.config.maxLookAtAngleRad,
        Math.min(this.config.maxLookAtAngleRad, angle * 0.4)
      );
    } else {
      this.targetLookAtAngle = 0;
    }
    this.currentLookAtAngle += (this.targetLookAtAngle - this.currentLookAtAngle) * Math.min(1.0, dt * 8);

    // 3. Secondary Motion Harmonic Springs
    const verticalBob = Math.sin(speed * 2.5) * 0.05;
    this.headSpring.update(dt, verticalBob * 0.5);
    this.backpackSpring.update(dt, verticalBob * 0.8);
  }

  /**
   * Injects procedural offsets into an existing AnimationPose.
   */
  public applyToPose(pose: AnimationPose) {
    // Lean torso
    const torso = pose.get('torso');
    if (torso) {
      torso.rotZ += this.currentLean;
    }

    // Look-at head
    const head = pose.get('head');
    if (head) {
      head.rotZ += this.currentLookAtAngle + this.headSpring.value;
    }

    // Backpack lag
    const backpack = pose.get('backpack');
    if (backpack) {
      backpack.posY += this.backpackSpring.value;
      backpack.rotZ -= this.currentLean * 0.5;
    }
  }
}
