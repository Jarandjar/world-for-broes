import * as THREE from 'three';
import { gameConfig, GameConfig } from '../config/GameConfig';

/**
 * Toon Diorama Rendering Stack (Phase Z0D - Section 73 to 76).
 * Manages:
 * 1. 3-Band Toon Ramp lighting & rim light.
 * 2. Dynamic light-tinted outlines with Juice Dial line-jitter.
 * 3. Biome & Time-of-Day Color Grading LUTs.
 * 4. Tilt-Shift Depth of Field (miniature lens blur).
 * 5. Warm campfire / lantern bloom diffusion.
 * 6. Subtle paper grain cardstock overlay.
 * 7. Squashed blob ambient occlusion under feet and wheels.
 */
export class ToonDioramaPipeline {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private domOverlay: HTMLElement | null = null;
  private jitterTimer: number = 0;
  private currentJitterIndex: number = 0;

  // Blob AO mesh pool
  private blobAOMeshes: Map<string, THREE.Mesh> = new Map();
  private blobAOGeometry: THREE.PlaneGeometry | null = null;
  private blobAOMaterial: THREE.MeshBasicMaterial | null = null;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;

    this.initDOMOverlays();
    this.initBlobAOResources();
  }

  private initDOMOverlays() {
    if (typeof document === 'undefined') return;

    let container = document.getElementById('diorama-post-stack');
    if (!container) {
      container = document.createElement('div');
      container.id = 'diorama-post-stack';
      container.className = 'diorama-post-stack';
      document.body.appendChild(container);
    }
    this.domOverlay = container;

    this.applyDOMStyles();
  }

  public applyDOMStyles() {
    if (!this.domOverlay) return;
    const config = gameConfig.get();

    // Tilt-shift and paper grain classes
    if (config.tiltShiftEnabled && config.animationQuality !== 'low') {
      this.domOverlay.classList.add('tilt-shift-active');
    } else {
      this.domOverlay.classList.remove('tilt-shift-active');
    }

    if (config.paperGrainEnabled && config.animationQuality === 'high') {
      this.domOverlay.classList.add('paper-grain-active');
    } else {
      this.domOverlay.classList.remove('paper-grain-active');
    }

    if (config.bloomEnabled && config.animationQuality !== 'low') {
      this.domOverlay.classList.add('warm-bloom-active');
    } else {
      this.domOverlay.classList.remove('warm-bloom-active');
    }
  }

  private initBlobAOResources() {
    // Squashed ellipse texture for blob ambient occlusion
    let canvas: any;
    if (typeof document !== 'undefined') {
      canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const grad = ctx.createRadialGradient(32, 32, 4, 32, 32, 30);
        grad.addColorStop(0, 'rgba(18, 14, 26, 0.55)');
        grad.addColorStop(0.5, 'rgba(20, 16, 28, 0.35)');
        grad.addColorStop(0.85, 'rgba(22, 18, 30, 0.12)');
        grad.addColorStop(1, 'rgba(22, 18, 30, 0.0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);
      }
    } else {
      canvas = { width: 64, height: 64 };
    }

    const texture = new THREE.CanvasTexture(canvas);
    this.blobAOGeometry = new THREE.PlaneGeometry(1, 1);
    this.blobAOGeometry.rotateX(-Math.PI * 0.5);

    this.blobAOMaterial = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
  }

  /**
   * Spawns or updates a blob AO shadow under an entity.
   */
  public updateBlobAO(id: string, x: number, y: number, z: number, radiusX: number = 1.0, radiusZ: number = 0.8) {
    const config = gameConfig.get();
    if (!config.blobAOEnabled) {
      const existing = this.blobAOMeshes.get(id);
      if (existing) existing.visible = false;
      return;
    }

    let mesh = this.blobAOMeshes.get(id);
    if (!mesh && this.blobAOGeometry && this.blobAOMaterial) {
      mesh = new THREE.Mesh(this.blobAOGeometry, this.blobAOMaterial.clone());
      this.scene.add(mesh);
      this.blobAOMeshes.set(id, mesh);
    }

    if (mesh) {
      mesh.visible = true;
      mesh.position.set(x, 0.015, z);

      // Height attenuation: scale and opacity diminish as entity jumps / elevates
      const elevation = Math.max(0, y);
      const scaleFactor = 1.0 / (1.0 + 0.5 * elevation);
      mesh.scale.set(radiusX * scaleFactor, 1, radiusZ * scaleFactor);

      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.opacity = (0.45 / (1.0 + 0.8 * elevation)) * (config.animationQuality === 'low' ? 0.3 : 1.0);
    }
  }

  /**
   * Removes a blob AO shadow when entity is despawned.
   */
  public removeBlobAO(id: string) {
    const mesh = this.blobAOMeshes.get(id);
    if (mesh) {
      this.scene.remove(mesh);
      this.blobAOMeshes.delete(id);
    }
  }

  /**
   * Returns dynamically tinted outline color based on time of day and biome.
   */
  public getOutlineColor(timeOfDay: number, isNight: boolean, biome: string = 'meadow'): string {
    const config = gameConfig.get();
    if (!config.outlinesEnabled) return 'transparent';

    // Time-of-day progression (0-360)
    // 0-60: Dawn, 60-200: Day, 200-260: Dusk, 260-360: Night
    if (isNight) {
      if (biome === 'hollow') return '#1A0C2B'; // Deep Void Purple
      if (biome === 'tundra') return '#101B2E'; // Arctic Midnight Slate
      return '#171B2B'; // Slate Indigo Night
    }

    if (timeOfDay >= 200 && timeOfDay <= 260) {
      return '#3A1624'; // Dusk warm plum-auburn
    }

    if (biome === 'desert') return '#4A2111'; // Warm adobe shadow
    if (biome === 'volcano') return '#30130D'; // Obsidian cinder
    if (biome === 'forest') return '#1A2922'; // Deep pine-ink

    return '#2E1810'; // Rich roasted chestnut daylight default
  }

  /**
   * Returns line jitter offset (in px) driven by Juice Dial.
   */
  public getLineJitterOffset(): { x: number; y: number } {
    const config = gameConfig.get();
    if (config.juiceDial <= 30 || config.reducedMotion) {
      return { x: 0, y: 0 };
    }

    const intensity = ((config.juiceDial - 30) / 70) * 1.5; // up to 1.5px
    const offsets = [
      { x: -intensity, y: intensity * 0.5 },
      { x: intensity * 0.8, y: -intensity * 0.6 },
      { x: -intensity * 0.4, y: -intensity * 0.8 },
      { x: intensity * 0.5, y: intensity * 0.9 },
    ];

    return offsets[this.currentJitterIndex % offsets.length];
  }

  /**
   * Returns Toon Ramp color adjustments for lighting shaders.
   */
  public getToonRampConfig(timeOfDay: number, isNight: boolean) {
    const isDusk = timeOfDay >= 200 && timeOfDay <= 260;

    return {
      highlightMultiplier: isNight ? 1.05 : 1.15,
      highlightWarmth: isDusk ? '#FFE0B2' : (isNight ? '#C5CAE9' : '#FFF3E0'),
      midtoneMultiplier: 1.0,
      shadowMultiplier: 0.75,
      shadowCoolness: isDusk ? '#311B92' : (isNight ? '#1A237E' : '#37474F'),
      rimIntensity: isNight ? 0.8 : 0.4,
      rimColor: isNight ? '#8C9EFF' : '#FFE082',
    };
  }

  /**
   * Updates pipeline per frame.
   */
  public update(delta: number) {
    const config = gameConfig.get();

    // Line Jitter timer (cycles at ~10 Hz)
    if (config.juiceDial > 30) {
      this.jitterTimer += delta;
      if (this.jitterTimer >= 0.1) {
        this.jitterTimer = 0;
        this.currentJitterIndex = (this.currentJitterIndex + 1) % 4;
      }
    }
  }

  /**
   * Destroys DOM overlays and cleans up scene meshes.
   */
  public destroy() {
    if (this.domOverlay) {
      this.domOverlay.remove();
      this.domOverlay = null;
    }
    for (const mesh of this.blobAOMeshes.values()) {
      this.scene.remove(mesh);
    }
    this.blobAOMeshes.clear();
  }
}
