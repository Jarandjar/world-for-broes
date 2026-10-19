import * as THREE from 'three';

/**
 * AtmosphericNoirPipeline - Gritty 2.5D Atmospheric Noir Visual Engine (Phase V1).
 * 
 * Features:
 * 1. Ground-Hugging Volumetric Fog:
 *    - Real-time animated multi-layered procedural fog hugging the terrain (y: 0.1m - 2.5m).
 *    - Simplex noise drift with wind velocity.
 *    - Dynamic player light-clearing radius (fog softly opens near campfires/torches and thickens into oppressive shadows).
 * 2. Heavy Vignette & Darkness Tunnel:
 *    - Deep multi-stop vignette creating claustrophobic focus on the player's immediate light radius.
 *    - Dynamically driven by Day/Night cycle (thicken to pitch-black at dusk & night).
 * 3. Film Grain & Color Grading:
 *    - High-contrast, desaturated gritty palette (charcoal, damp mud, rotting greens).
 *    - Animated 35mm film grain texture over deep shadows.
 * 4. Zero cardboard/papercraft or tilt-shift blur tricks.
 */
export class AtmosphericNoirPipeline {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private domOverlay: HTMLElement | null = null;
  private grainCanvas: HTMLCanvasElement | null = null;
  private grainCtx: CanvasRenderingContext2D | null = null;
  private grainImageData: ImageData | null = null;
  private grainAnimFrame: number = 0;

  // Volumetric ground fog mesh & custom shader
  private fogGroup: THREE.Group;
  private fogMaterial: THREE.ShaderMaterial;
  private fogPlanes: THREE.Mesh[] = [];

  // Lighting & Day/Night phase interpolation
  public currentDensity: number = 0.45;
  public targetDensity: number = 0.45;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;
    this.fogGroup = new THREE.Group();
    this.scene.add(this.fogGroup);

    this.fogMaterial = this.createVolumetricFogMaterial();
    this.initGroundFogPlanes();
    this.initDOMOverlays();
  }

  /**
   * Custom GLSL shader for ground-hugging volumetric fog with noise drift and player light opening.
   */
  private createVolumetricFogMaterial(): THREE.ShaderMaterial {
    const vertexShader = `
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      void main() {
        vUv = uv;
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `;

    const fragmentShader = `
      uniform float uTime;
      uniform float uDensity;
      uniform vec3 uColor;
      uniform vec3 uPlayerPos;
      uniform float uLightRadius;
      
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      // 2D Simplex-style procedural noise
      vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

      float snoise(vec2 v) {
        const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                            0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                           -0.577350269189626,  // -1.0 + 2.0 * C.x
                            0.024390243902439); // 1.0 / 41.0
        vec2 i  = floor(v + dot(v, C.yy) );
        vec2 x0 = v -   i + dot(i, C.xx);
        vec2 i1;
        i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz;
        x12.xy -= i1;
        i = mod289(i);
        vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
              + i.x + vec3(0.0, i1.x, 1.0 ));
        vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
        m = m*m ;
        m = m*m ;
        vec3 x = 2.0 * fract(p * C.www) - 1.0;
        vec3 h = abs(x) - 0.5;
        vec3 ox = floor(x + 0.5);
        vec3 a0 = x - ox;
        m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
        vec3 g;
        g.x  = a0.x  * x0.x  + h.x  * x0.y;
        g.yz = a0.yz * x12.xz + h.yz * x12.yw;
        return 130.0 * dot(m, g);
      }

      void main() {
        // Multi-octave drifting noise
        vec2 driftUv = (vWorldPosition.xz * 0.08) + vec2(uTime * 0.025, uTime * 0.015);
        float n1 = snoise(driftUv) * 0.5 + 0.5;
        float n2 = snoise(driftUv * 2.2 - vec2(uTime * 0.03, -uTime * 0.02)) * 0.25 + 0.25;
        float noise = clamp(n1 + n2 * 0.5, 0.0, 1.0);

        // Distance from player (clears fog inside campfire / flashlight radius)
        float distToPlayer = length(vWorldPosition.xz - uPlayerPos.xz);
        float clearFactor = smoothstep(uLightRadius * 0.4, uLightRadius * 1.5, distToPlayer);

        // Edge vignette fade on the plane boundaries
        vec2 planeCenterDist = abs(vUv - 0.5) * 2.0;
        float edgeFade = (1.0 - smoothstep(0.7, 1.0, planeCenterDist.x)) * 
                         (1.0 - smoothstep(0.7, 1.0, planeCenterDist.y));

        float alpha = noise * uDensity * clearFactor * edgeFade;

        // Oppressive dark charcoal green tint
        gl_FragColor = vec4(uColor, alpha);
      }
    `;

    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime: { value: 0.0 },
        uDensity: { value: 0.5 },
        uColor: { value: new THREE.Vector3(0.045, 0.055, 0.05) }, // Dark decayed green charcoal
        uPlayerPos: { value: new THREE.Vector3(0, 0, 0) },
        uLightRadius: { value: 5.5 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  /**
   * Multi-tiered ground fog planes centered around the player's view radius.
   */
  private initGroundFogPlanes() {
    const planeGeo = new THREE.PlaneGeometry(70, 70, 1, 1);
    planeGeo.rotateX(-Math.PI / 2);

    // 3 layered horizontal sheets hugging the ground
    const heights = [0.25, 0.85, 1.55];
    heights.forEach((h, idx) => {
      const mesh = new THREE.Mesh(planeGeo, this.fogMaterial);
      mesh.position.set(0, h, 0);
      mesh.rotation.y = (idx * Math.PI) / 3;
      this.fogGroup.add(mesh);
      this.fogPlanes.push(mesh);
    });
  }

  /**
   * Initializes DOM overlays for Heavy Vignette, Film Grain, and Gritty Color Grading.
   */
  private initDOMOverlays() {
    if (typeof document === 'undefined') return;

    let container = document.getElementById('atmospheric-noir-stack');
    if (!container) {
      container = document.createElement('div');
      container.id = 'atmospheric-noir-stack';
      container.className = 'atmospheric-noir-stack';
      document.body.appendChild(container);
    }
    this.domOverlay = container;

    // Remove legacy diorama classes
    const diorama = document.getElementById('diorama-post-stack');
    if (diorama) {
      diorama.className = '';
      diorama.style.display = 'none';
    }

    this.initFilmGrainCanvas();
  }

  /**
   * Fast procedural film grain buffer rendered to canvas.
   */
  private initFilmGrainCanvas() {
    if (typeof document === 'undefined' || !this.domOverlay) return;

    const canvas = document.createElement('canvas');
    canvas.id = 'noir-film-grain';
    canvas.className = 'noir-film-grain';
    canvas.width = 256;
    canvas.height = 256;
    this.domOverlay.appendChild(canvas);

    this.grainCanvas = canvas;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      this.grainCtx = ctx;
      this.grainImageData = ctx.createImageData(256, 256);
      this.generateGrainFrame();
    }
  }

  private generateGrainFrame() {
    if (!this.grainCtx || !this.grainImageData) return;
    const data = this.grainImageData.data;
    const len = data.length;

    // Generate monochrome high-frequency analog grain
    for (let i = 0; i < len; i += 4) {
      const noise = (Math.random() * 255) | 0;
      data[i] = noise;
      data[i + 1] = noise;
      data[i + 2] = noise;
      data[i + 3] = (Math.random() * 28 + 12) | 0; // ~5-15% grain opacity
    }

    this.grainCtx.putImageData(this.grainImageData, 0, 0);
  }

  /**
   * Updates Atmospheric Noir Pipeline per frame.
   * Call in GameApp.render() with raw delta, timeOfDay, and player position.
   */
  public update(
    delta: number,
    timeOfDay: number,
    cycleDuration: number,
    playerPos: THREE.Vector3,
    isNearFire: boolean = false
  ) {
    // 1. Update shader time and track player
    this.fogMaterial.uniforms.uTime.value += delta;
    this.fogMaterial.uniforms.uPlayerPos.value.copy(playerPos);

    // Follow player smoothly on ground plane
    this.fogGroup.position.set(playerPos.x, 0, playerPos.z);

    // 2. Day/Night Phase Density Curve
    // Day (0-210 deg): light atmospheric ground haze (0.32)
    // Dusk (210-280 deg): heavy rolling ground fog (0.75)
    // Night (280-360 deg): oppressive suffocating black fog (1.0)
    const progress = (timeOfDay % cycleDuration) / cycleDuration;
    const degrees = progress * 360;

    let targetDensity = 0.32;
    let targetLightRadius = isNearFire ? 8.5 : 5.0;
    let targetFogColor = new THREE.Vector3(0.07, 0.08, 0.075); // Desaturated swamp charcoal

    if (degrees >= 210 && degrees < 280) {
      // Dusk
      const t = (degrees - 210) / 70;
      targetDensity = THREE.MathUtils.lerp(0.32, 0.78, t);
      targetLightRadius = isNearFire ? 7.5 : 4.2;
      targetFogColor.set(0.06, 0.05, 0.045); // Rust charcoal twilight
    } else if (degrees >= 280) {
      // Night
      targetDensity = 0.98;
      targetLightRadius = isNearFire ? 7.0 : 3.5;
      targetFogColor.set(0.02, 0.025, 0.025); // Deep charcoal black
    }

    this.currentDensity = THREE.MathUtils.lerp(this.currentDensity, targetDensity, delta * 3.0);
    this.fogMaterial.uniforms.uDensity.value = this.currentDensity;
    this.fogMaterial.uniforms.uLightRadius.value = THREE.MathUtils.lerp(
      this.fogMaterial.uniforms.uLightRadius.value,
      targetLightRadius,
      delta * 4.0
    );
    this.fogMaterial.uniforms.uColor.value.lerp(targetFogColor, delta * 2.0);

    // 3. Update Heavy Vignette Intensity in DOM
    if (this.domOverlay) {
      const isNight = degrees >= 280;
      const isDusk = degrees >= 210 && degrees < 280;
      if (isNight) {
        this.domOverlay.setAttribute('data-phase', 'night');
      } else if (isDusk) {
        this.domOverlay.setAttribute('data-phase', 'dusk');
      } else {
        this.domOverlay.setAttribute('data-phase', 'day');
      }
    }

    // 4. Jitter film grain frame periodically
    this.grainAnimFrame++;
    if (this.grainAnimFrame % 3 === 0) {
      this.generateGrainFrame();
    }
  }

  public setVisible(visible: boolean) {
    this.fogGroup.visible = visible;
    if (this.domOverlay) {
      this.domOverlay.style.display = visible ? 'block' : 'none';
    }
  }
}
