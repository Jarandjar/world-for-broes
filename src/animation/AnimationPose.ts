import * as THREE from 'three';

export interface TransformData {
  posX: number;
  posY: number;
  posZ: number;
  rotX: number;
  rotY: number;
  rotZ: number;
  scaleX: number;
  scaleY: number;
  scaleZ: number;
}

export function createDefaultTransform(): TransformData {
  return {
    posX: 0,
    posY: 0,
    posZ: 0,
    rotX: 0,
    rotY: 0,
    rotZ: 0,
    scaleX: 1,
    scaleY: 1,
    scaleZ: 1,
  };
}

export type AnimationPose = Map<string, TransformData>;

export function createEmptyPose(): AnimationPose {
  return new Map<string, TransformData>();
}

export function clonePose(source: AnimationPose): AnimationPose {
  const target: AnimationPose = new Map();
  for (const [key, val] of source.entries()) {
    target.set(key, { ...val });
  }
  return target;
}

/**
 * Linearly blends between pose A and pose B by weight alpha in [0, 1].
 */
export function lerpPose(poseA: AnimationPose, poseB: AnimationPose, alpha: number): AnimationPose {
  const result: AnimationPose = new Map();
  const clampedAlpha = Math.max(0, Math.min(1, alpha));
  const invAlpha = 1 - clampedAlpha;

  // Gather all unique bone/node keys
  const allKeys = new Set<string>([...poseA.keys(), ...poseB.keys()]);

  for (const key of allKeys) {
    const a = poseA.get(key) || createDefaultTransform();
    const b = poseB.get(key) || createDefaultTransform();

    result.set(key, {
      posX: a.posX * invAlpha + b.posX * clampedAlpha,
      posY: a.posY * invAlpha + b.posY * clampedAlpha,
      posZ: a.posZ * invAlpha + b.posZ * clampedAlpha,
      rotX: a.rotX * invAlpha + b.rotX * clampedAlpha,
      rotY: a.rotY * invAlpha + b.rotY * clampedAlpha,
      rotZ: a.rotZ * invAlpha + b.rotZ * clampedAlpha,
      scaleX: a.scaleX * invAlpha + b.scaleX * clampedAlpha,
      scaleY: a.scaleY * invAlpha + b.scaleY * clampedAlpha,
      scaleZ: a.scaleZ * invAlpha + b.scaleZ * clampedAlpha,
    });
  }

  return result;
}

/**
 * Applies a masked upper-body pose on top of a base pose.
 */
export function applyMaskedPose(
  basePose: AnimationPose,
  upperPose: AnimationPose,
  maskNodes: Set<string>,
  weight: number = 1.0
): AnimationPose {
  const result = clonePose(basePose);
  if (weight <= 0) return result;

  for (const node of maskNodes) {
    const up = upperPose.get(node);
    if (!up) continue;

    if (weight >= 1.0) {
      result.set(node, { ...up });
    } else {
      const base = result.get(node) || createDefaultTransform();
      const inv = 1 - weight;
      result.set(node, {
        posX: base.posX * inv + up.posX * weight,
        posY: base.posY * inv + up.posY * weight,
        posZ: base.posZ * inv + up.posZ * weight,
        rotX: base.rotX * inv + up.rotX * weight,
        rotY: base.rotY * inv + up.rotY * weight,
        rotZ: base.rotZ * inv + up.rotZ * weight,
        scaleX: base.scaleX * inv + up.scaleX * weight,
        scaleY: base.scaleY * inv + up.scaleY * weight,
        scaleZ: base.scaleZ * inv + up.scaleZ * weight,
      });
    }
  }

  return result;
}

/**
 * Applies transform to a THREE.Object3D node.
 */
export function applyTransformToNode(node: THREE.Object3D, t: TransformData, basePos?: THREE.Vector3) {
  if (basePos) {
    node.position.set(basePos.x + t.posX, basePos.y + t.posY, basePos.z + t.posZ);
  } else {
    node.position.set(t.posX, t.posY, t.posZ);
  }
  node.rotation.set(t.rotX, t.rotY, t.rotZ);
  node.scale.set(t.scaleX, t.scaleY, t.scaleZ);
}
