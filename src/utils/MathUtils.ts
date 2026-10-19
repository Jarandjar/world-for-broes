/**
 * Frame-rate-independent exponential damping and interpolation utilities.
 * Ensures consistent physics, camera follow, and motion across any refresh rate (30Hz, 60Hz, 144Hz+).
 */

export function damp(current: number, target: number, lambda: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}

export function dampAngle(current: number, target: number, lambda: number, dt: number): number {
  let diff = (target - current) % (Math.PI * 2);
  if (diff < -Math.PI) diff += Math.PI * 2;
  if (diff > Math.PI) diff -= Math.PI * 2;
  return current + diff * (1 - Math.exp(-lambda * dt));
}

export function dampVector3(
  current: { x: number; y: number; z: number },
  target: { x: number; y: number; z: number },
  lambda: number,
  dt: number
) {
  current.x = damp(current.x, target.x, lambda, dt);
  current.y = damp(current.y, target.y, lambda, dt);
  current.z = damp(current.z, target.z, lambda, dt);
}
