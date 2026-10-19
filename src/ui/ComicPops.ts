import * as THREE from 'three';
import { gameConfig } from '../config/GameConfig';

export interface ComicPopOptions {
  text: string;
  color?: string;
  durationMs?: number;
  rotationDeg?: number;
}

/**
 * Comic Onomatopoeia Word Pops ("BONK!", "POOF!", "SQUEAK!", "CLANG!").
 * Section 77: Spawns animated 2D comic sound words at impact coordinates.
 */
export class ComicPops {
  private static container: HTMLElement | null = null;

  private static getContainer(): HTMLElement | null {
    if (typeof document === 'undefined') return null;
    if (!this.container) {
      let el = document.getElementById('comic-pops-layer');
      if (!el) {
        el = document.createElement('div');
        el.id = 'comic-pops-layer';
        el.style.position = 'fixed';
        el.style.inset = '0';
        el.style.pointerEvents = 'none';
        el.style.zIndex = '90';
        document.body.appendChild(el);
      }
      this.container = el;
    }
    return this.container;
  }

  /**
   * Spawns a comic sound word at 3D world coordinates.
   */
  public static spawnWorld(
    worldPos: THREE.Vector3,
    camera: THREE.Camera,
    text: string,
    color: string = '#FFE600'
  ) {
    if (typeof document === 'undefined') return;
    const config = gameConfig.get();
    if (config.reducedMotion) return;

    // Project 3D pos to 2D screen space
    const projected = worldPos.clone().project(camera);
    if (projected.z > 1.0) return; // Behind camera

    const screenX = (projected.x * 0.5 + 0.5) * window.innerWidth;
    const screenY = (-(projected.y * 0.5) + 0.5) * window.innerHeight;

    this.spawnScreen(screenX, screenY, { text, color });
  }

  /**
   * Spawns a comic sound word at 2D screen pixel coordinates.
   */
  public static spawnScreen(x: number, y: number, options: ComicPopOptions) {
    const parent = this.getContainer();
    if (!parent) return;

    const config = gameConfig.get();
    const juiceScale = 0.5 + (config.juiceDial / 100) * 0.7; // 0.5x at 0% to 1.2x at 100%

    const wordEl = document.createElement('div');
    wordEl.className = 'comic-pop-word';
    wordEl.innerText = options.text;
    wordEl.style.left = `${x}px`;
    wordEl.style.top = `${y}px`;
    wordEl.style.color = options.color || '#FFE600';

    const rot = options.rotationDeg ?? (Math.random() * 24 - 12);
    wordEl.style.setProperty('--pop-rot', `${rot}deg`);
    wordEl.style.transformOrigin = 'center center';
    wordEl.style.fontSize = `${Math.round(32 * juiceScale)}px`;

    parent.appendChild(wordEl);

    const duration = options.durationMs ?? 480;
    setTimeout(() => {
      wordEl.remove();
    }, duration);
  }

  // Pre-configured sound word helpers
  public static show(text: string, worldPos: THREE.Vector3, camera: THREE.Camera, color?: string) {
    this.spawnWorld(worldPos, camera, text, color);
  }

  public static bonk(worldPos: THREE.Vector3, camera: THREE.Camera) {
    this.spawnWorld(worldPos, camera, 'BONK!', '#FFE500');
  }

  public static poof(worldPos: THREE.Vector3, camera: THREE.Camera) {
    this.spawnWorld(worldPos, camera, 'POOF!', '#70E000');
  }

  public static squeak(worldPos: THREE.Vector3, camera: THREE.Camera) {
    this.spawnWorld(worldPos, camera, 'SQUEAK!', '#FF007F');
  }

  public static clang(worldPos: THREE.Vector3, camera: THREE.Camera) {
    this.spawnWorld(worldPos, camera, 'CLANG!', '#FF9100');
  }

  public static honk(worldPos: THREE.Vector3, camera: THREE.Camera) {
    this.spawnWorld(worldPos, camera, 'HONK!', '#00F5D4');
  }
}
