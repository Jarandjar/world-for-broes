import * as THREE from 'three';

/**
 * NoirCharacterRenderer - Renders High-Fidelity 2.5D Normal-Mapped Characters (Phase V1/V2).
 * 
 * Features:
 * 1. Gritty Survivor Bro:
 *    - Diffuse + Normal Map wrapping with dynamic campfire & flashlight relief.
 *    - Weathered canvas field jacket, backpack, fire axe.
 * 2. Undead Shambler Zombie:
 *    - Decaying, mud-caked dark shadowy silhouette tint.
 *    - Piercing unlit glowing eyes (pale yellow-white emissive mask).
 *    - Jerky erratic posture and twitching idle.
 *    - Full 3D shadow casting.
 */
export class NoirCharacterRenderer {
  private static textureLoader = new THREE.TextureLoader();

  // Cached materials
  private static survivorMaterial: THREE.MeshStandardMaterial | null = null;
  private static shamblerMaterial: THREE.MeshStandardMaterial | null = null;

  public static getSurvivorMaterial(): THREE.MeshStandardMaterial {
    if (!this.survivorMaterial) {
      const diffuse = this.textureLoader.load('/textures/noir/survivor_diffuse.png');
      const normal = this.textureLoader.load('/textures/noir/survivor_normal.png');
      diffuse.colorSpace = THREE.SRGBColorSpace;

      this.survivorMaterial = new THREE.MeshStandardMaterial({
        map: diffuse,
        normalMap: normal,
        normalScale: new THREE.Vector2(1.8, 1.8),
        roughness: 0.72,
        metalness: 0.12,
        transparent: true,
        alphaTest: 0.12,
        side: THREE.DoubleSide,
      });
    }
    return this.survivorMaterial;
  }

  public static getShamblerMaterial(): THREE.MeshStandardMaterial {
    if (!this.shamblerMaterial) {
      const diffuse = this.textureLoader.load('/textures/noir/shambler_diffuse.png');
      const normal = this.textureLoader.load('/textures/noir/shambler_normal.png');
      const emissive = this.textureLoader.load('/textures/noir/shambler_emissive.png');
      diffuse.colorSpace = THREE.SRGBColorSpace;

      this.shamblerMaterial = new THREE.MeshStandardMaterial({
        map: diffuse,
        normalMap: normal,
        normalScale: new THREE.Vector2(2.4, 2.4),
        emissiveMap: emissive,
        emissive: new THREE.Color(0xffffcc),
        emissiveIntensity: 3.5, // Piercing glowing eyes that shine through pitch darkness
        color: new THREE.Color(0xffffff), // Clean diffuse base so rotting clothes & flesh textures pop
        roughness: 0.90,
        metalness: 0.05,
        transparent: true,
        alphaTest: 0.12,
        side: THREE.DoubleSide,
      });
    }
    return this.shamblerMaterial;
  }

  /**
   * Spawns a 2.5D Gritty Survivor Mesh.
   */
  public static createSurvivor(position: THREE.Vector3, height: number = 2.1): THREE.Group {
    const group = new THREE.Group();
    group.position.copy(position);

    const aspect = 1.0;
    const width = height * aspect;
    const geo = new THREE.PlaneGeometry(width, height);
    geo.translate(0, height * 0.5, 0); // Origin at feet

    const mat = this.getSurvivorMaterial();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = false;

    // Contact shadow beneath feet
    const shadowGeo = new THREE.PlaneGeometry(1.0, 0.5);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x050608,
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
    });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = 0.02;

    group.add(shadow);
    group.add(mesh);

    return group;
  }

  /**
   * Spawns an Undead Shambler Zombie with glowing piercing eyes and shadow casting.
   */
  public static createShambler(position: THREE.Vector3, height: number = 2.2): THREE.Group {
    const group = new THREE.Group();
    group.position.copy(position);

    const aspect = 1.0;
    const width = height * aspect;
    const geo = new THREE.PlaneGeometry(width, height);
    geo.translate(0, height * 0.5, 0);

    const mat = this.getShamblerMaterial();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = false;

    // Soft eye glow point light that illuminates surrounding fog
    const eyeLight = new THREE.PointLight(0xffffbb, 0.45, 2.5, 2.0);
    eyeLight.position.set(0, height * 0.85, 0.15);

    // Contact shadow
    const shadowGeo = new THREE.PlaneGeometry(0.9, 0.45);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x030405,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = 0.02;

    group.add(shadow);
    group.add(mesh);
    group.add(eyeLight);

    return group;
  }
}
