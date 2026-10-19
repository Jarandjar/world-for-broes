import * as THREE from 'three';
import { LayeredStateMachine } from './LayeredStateMachine';
import { ProceduralLayer } from './ProceduralLayer';
import { createHumanoidClips } from './HumanoidClips';
import { AnimationClip } from './AnimationClip';
import { applyTransformToNode, AnimationPose } from './AnimationPose';

export interface EntityAnimationState {
  isMoving: boolean;
  isSprinting: boolean;
  isSneaking: boolean;
  velocity: THREE.Vector3;
  isChopping: boolean;
  isEating: boolean;
  isDancing: boolean;
  isHonking: boolean;
  isHurt: boolean;
  isDown: boolean;
  facingSign: number; // 1 or -1
  turnRate?: number;
  lookAtTarget?: THREE.Vector3;
  worldPosition?: THREE.Vector3;
}

/**
 * High-Level Humanoid Animation Controller.
 * Implements Section 70 criteria 7:
 * "Every entity maps game state to animation state through a single deterministic function,
 * with zero scattered, ad-hoc play() calls in game logic."
 */
export class AnimationController {
  public clips: Record<string, AnimationClip>;
  public stateMachine: LayeredStateMachine;
  public procedural: ProceduralLayer;

  public nodes: Map<string, THREE.Object3D> = new Map();
  public basePositions: Map<string, THREE.Vector3> = new Map();

  // Metrics for profiling and Animation Lab (<2ms budget)
  public lastUpdateMicros: number = 0;
  public activeBoneCount: number = 0;

  constructor(clips?: Record<string, AnimationClip>) {
    this.clips = clips || createHumanoidClips();
    this.stateMachine = new LayeredStateMachine();
    this.procedural = new ProceduralLayer();

    // Start with idle clip on base layer
    if (this.clips['idle']) {
      this.stateMachine.playBase(this.clips['idle'], 0);
    }
  }

  /**
   * Registers a rig mesh/node to receive animated transforms.
   */
  public registerNode(name: string, node: THREE.Object3D) {
    this.nodes.set(name, node);
    this.basePositions.set(name, node.position.clone());
    this.activeBoneCount = this.nodes.size;
  }

  /**
   * Evaluates game state and maps it deterministically to animation layers.
   * Runs on the 60 Hz fixed timestep loop.
   */
  public updateAnimationState(dt: number, state: EntityAnimationState) {
    const t0 = performance.now();

    // 1. Map Overrides (Highest priority: hurt, knockdown, dance)
    if (state.isHurt) {
      if (this.clips['hurt']) this.stateMachine.playOverride(this.clips['hurt'], 0.04);
    } else if (state.isDown) {
      if (this.clips['down']) this.stateMachine.playOverride(this.clips['down'], 0.10);
    } else if (state.isDancing) {
      if (this.clips['dance']) this.stateMachine.playOverride(this.clips['dance'], 0.15);
    } else {
      this.stateMachine.stopOverride();
    }

    // 2. Map Upper-Body Masked Actions (chop, eat, honk)
    if (state.isChopping) {
      if (this.clips['chop']) this.stateMachine.playUpper(this.clips['chop'], 0.06);
    } else if (state.isEating) {
      if (this.clips['eat']) this.stateMachine.playUpper(this.clips['eat'], 0.10);
    } else if (state.isHonking) {
      if (this.clips['honk']) this.stateMachine.playUpper(this.clips['honk'], 0.08);
    } else {
      this.stateMachine.stopUpper(0.12);
    }

    // 3. Map Base Locomotion Layer (idle, sneak, walk, run, sprint)
    const speed = state.velocity.length();
    let currentNominal = 0.65;

    if (!state.isMoving || speed <= 0.05) {
      if (this.clips['idle']) this.stateMachine.playBase(this.clips['idle'], 0.15);
      this.stateMachine.baseSpeedMultiplier = 1.0;
    } else if (state.isSneaking) {
      if (this.clips['sneak']) this.stateMachine.playBase(this.clips['sneak'], 0.12);
      currentNominal = 1.0;
      this.stateMachine.baseSpeedMultiplier = this.procedural.calculateStrideSpeed(speed, currentNominal);
    } else if (state.isSprinting) {
      if (this.clips['sprint']) this.stateMachine.playBase(this.clips['sprint'], 0.10);
      currentNominal = 0.45;
      this.stateMachine.baseSpeedMultiplier = this.procedural.calculateStrideSpeed(speed, currentNominal);
    } else if (speed > 5.5) {
      if (this.clips['run']) this.stateMachine.playBase(this.clips['run'], 0.12);
      currentNominal = 0.60;
      this.stateMachine.baseSpeedMultiplier = this.procedural.calculateStrideSpeed(speed, currentNominal);
    } else {
      if (this.clips['walk']) this.stateMachine.playBase(this.clips['walk'], 0.12);
      currentNominal = 0.80;
      this.stateMachine.baseSpeedMultiplier = this.procedural.calculateStrideSpeed(speed, currentNominal);
    }

    // 4. Update Fixed Timestep State Machine
    this.stateMachine.update(dt);

    // 5. Update Procedural Layer
    this.procedural.update(
      dt,
      state.velocity,
      state.facingSign,
      state.turnRate ?? 0,
      state.lookAtTarget,
      state.worldPosition
    );

    const t1 = performance.now();
    this.lastUpdateMicros = (t1 - t0) * 1000;
  }

  /**
   * Interpolates pose between fixed physics states by alpha in [0, 1) and writes to rig nodes.
   * Runs at display refresh rate inside rendering loop.
   */
  public applyInterpolatedPose(alpha: number) {
    const pose = this.stateMachine.getInterpolatedPose(alpha);

    // Apply secondary procedural offsets onto final pose
    this.procedural.applyToPose(pose);

    for (const [nodeName, node] of this.nodes.entries()) {
      const transform = pose.get(nodeName);
      if (transform) {
        const basePos = this.basePositions.get(nodeName);
        applyTransformToNode(node, transform, basePos);
      }
    }
  }
}
