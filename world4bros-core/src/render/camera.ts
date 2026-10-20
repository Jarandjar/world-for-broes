import { clamp, damp } from '../engine/math';
/** Top-down camera. zoom = CSS pixels per world unit. */
export const MIN_CAMERA_ZOOM = 8;
export const MAX_CAMERA_ZOOM = 170;
export const DEFAULT_CAMERA_ZOOM = 54;
/** Zoom (CSS px per unit) where detail steps up: below the first is far, above the second is close-up. */
export const LOD_ZOOMS = [20, 90] as const;
export const lodFor = (zoom: number): number => (zoom < LOD_ZOOMS[0] ? 0 : zoom < LOD_ZOOMS[1] ? 1 : 2);

export class Camera {
  x = 0; y = 0; zoom = DEFAULT_CAMERA_ZOOM; targetZoom = DEFAULT_CAMERA_ZOOM; userZoom = DEFAULT_CAMERA_ZOOM; shake = 0;
  drivingSpeedOffset = 0;

  update(tx: number, ty: number, dt: number): void {
    // Keep player centered on screen at all times.
    this.x = tx; this.y = ty;
    const effectiveTarget = clamp(this.targetZoom - this.drivingSpeedOffset, MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
    this.zoom = damp(this.zoom, effectiveTarget, 8, dt);
    this.shake = Math.max(0, this.shake - dt * 3);
  }
  snap(tx: number, ty: number): void { this.x = tx; this.y = ty; }
  setZoom(z: number): void {
    this.userZoom = clamp(z, MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
    this.targetZoom = this.userZoom;
  }
  toggleMaxView(): void {
    if (this.targetZoom <= 12) {
      this.setZoom(DEFAULT_CAMERA_ZOOM);
    } else {
      this.setZoom(MIN_CAMERA_ZOOM);
    }
  }
  /** Level of detail: 0 = far (skip small details), 1 = normal, 2 = close-up. */
  get lod(): number { return lodFor(this.zoom); }
  screenToWorld(sx: number, sy: number, w: number, h: number): { x: number; y: number } { return { x: this.x + (sx - w / 2) / this.zoom, y: this.y + (sy - h / 2) / this.zoom }; }
}
