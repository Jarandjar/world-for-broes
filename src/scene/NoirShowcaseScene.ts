import * as THREE from 'three';
import { NoirCharacterRenderer } from '../rendering/NoirCharacterRenderer';

/**
 * NoirShowcaseScene - Live visual staging scene for Gritty 2.5D Atmospheric Noir.
 * Stages:
 * - Flickering campfire with harsh point light casting deep shadows.
 * - The Gritty Survivor Bro with dynamic normal-mapped lighting.
 * - The Undead Shamblers with shadowy silhouettes and piercing glowing unlit eyes.
 * - Rotting ancient pine trees and wet muddy ground.
 */
export class NoirShowcaseScene {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  public group: THREE.Group;
  public isActive: boolean = false;

  private campfireLight!: THREE.PointLight;
  private survivorMesh!: THREE.Group;
  private zombie1!: THREE.Group;
  private zombie2!: THREE.Group;
  private animTimer: number = 0;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.group.visible = false;

    this.buildScene();
  }

  private buildScene() {
    // 1. Center Campfire
    const firePos = new THREE.Vector3(0, 0, 0);

    // Stone ring around campfire
    const stonesGeo = new THREE.DodecahedronGeometry(0.18, 1);
    const stoneMat = new THREE.MeshStandardMaterial({
      color: 0x222625,
      roughness: 0.9,
    });
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const s = new THREE.Mesh(stonesGeo, stoneMat);
      s.position.set(Math.cos(angle) * 0.75, 0.1, Math.sin(angle) * 0.75);
      s.scale.set(1 + Math.random() * 0.3, 0.7, 1 + Math.random() * 0.3);
      s.castShadow = true;
      this.group.add(s);
    }

    // Burning ember logs
    const logGeo = new THREE.CylinderGeometry(0.08, 0.09, 0.85, 6);
    const logMat = new THREE.MeshStandardMaterial({
      color: 0x140e0a,
      roughness: 0.85,
    });
    const log1 = new THREE.Mesh(logGeo, logMat);
    log1.rotation.z = Math.PI * 0.35;
    log1.position.set(0, 0.15, 0);
    const log2 = new THREE.Mesh(logGeo, logMat);
    log2.rotation.z = -Math.PI * 0.35;
    log2.rotation.y = Math.PI * 0.5;
    log2.position.set(0, 0.15, 0);
    this.group.add(log1);
    this.group.add(log2);

    // Flame Core (Emissive glowing flame shape)
    const flameGeo = new THREE.ConeGeometry(0.25, 0.65, 5);
    const flameMat = new THREE.MeshBasicMaterial({
      color: 0xffaa22,
    });
    const flame = new THREE.Mesh(flameGeo, flameMat);
    flame.position.set(0, 0.38, 0);
    this.group.add(flame);

    // Harsh Dynamic Campfire Point Light casting deep shadows
    this.campfireLight = new THREE.PointLight(0xff7a18, 5.2, 25.0, 1.2);
    this.campfireLight.position.set(0, 0.75, 0);
    this.campfireLight.castShadow = true;
    this.campfireLight.shadow.mapSize.width = 1024;
    this.campfireLight.shadow.mapSize.height = 1024;
    this.campfireLight.shadow.bias = -0.002;
    this.group.add(this.campfireLight);

    // Warm firelight glow fill
    const fireFill = new THREE.PointLight(0xffaa33, 2.8, 14.0, 1.0);
    fireFill.position.set(0, 1.1, 0);
    this.group.add(fireFill);

    // Moody Darkwood ambient fill for forest silhouettes
    const forestAmbient = new THREE.AmbientLight(0x28322e, 0.95);
    this.group.add(forestAmbient);

    // 2. Gritty Survivor Bro (left of campfire, facing fire)
    const survivorPos = new THREE.Vector3(-1.8, 0, 0.4);
    this.survivorMesh = NoirCharacterRenderer.createSurvivor(survivorPos, 2.3);
    this.group.add(this.survivorMesh);

    // 3. Undead Shamblers emerging from fog and shadows (right of campfire)
    const z1Pos = new THREE.Vector3(1.7, 0, 0.2);
    this.zombie1 = NoirCharacterRenderer.createShambler(z1Pos, 2.3);
    this.group.add(this.zombie1);

    const z2Pos = new THREE.Vector3(3.4, 0, -1.2);
    this.zombie2 = NoirCharacterRenderer.createShambler(z2Pos, 2.1);
    this.group.add(this.zombie2);

    // 4. Decaying Rotting Trees & Logs
    this.spawnRottingFoliage();
  }

  private spawnRottingFoliage() {
    const treeGeo = new THREE.CylinderGeometry(0.25, 0.42, 5.5, 7);
    const barkMat = new THREE.MeshStandardMaterial({
      color: 0x161a18, // Weathered rotting dark bark
      roughness: 0.95,
      metalness: 0.05,
    });

    const positions = [
      [-4.5, -3.0],
      [-5.2, 2.8],
      [4.2, -4.0],
      [5.5, 3.2],
      [-2.0, -5.5],
    ];

    positions.forEach(([x, z]) => {
      const trunk = new THREE.Mesh(treeGeo, barkMat);
      trunk.position.set(x, 2.75, z);
      trunk.rotation.y = Math.random() * Math.PI;
      trunk.castShadow = true;
      this.group.add(trunk);
    });

    // Fallen decaying log near fire
    const logGeo = new THREE.CylinderGeometry(0.22, 0.28, 4.2, 6);
    const fallenLog = new THREE.Mesh(logGeo, barkMat);
    fallenLog.position.set(0.5, 0.22, -3.2);
    fallenLog.rotation.z = Math.PI * 0.5;
    fallenLog.rotation.y = 0.35;
    fallenLog.castShadow = true;
    this.group.add(fallenLog);
  }

  public activate(centerPos?: THREE.Vector3) {
    this.isActive = true;
    this.group.visible = true;
    if (centerPos) {
      this.group.position.set(centerPos.x, 0, centerPos.z);
    }
  }

  public deactivate() {
    this.isActive = false;
    this.group.visible = false;
  }

  public update(delta: number) {
    if (!this.isActive) return;

    this.animTimer += delta;

    // 1. Campfire light flicker (dynamic intensity & subtle jitter)
    const flicker = Math.sin(this.animTimer * 14) * 0.18 + Math.cos(this.animTimer * 23) * 0.12;
    this.campfireLight.intensity = 2.8 + flicker;
    this.campfireLight.position.x = Math.sin(this.animTimer * 8) * 0.04;
    this.campfireLight.position.z = Math.cos(this.animTimer * 11) * 0.04;

    // 2. Survivor gentle breathing stance
    const breath = Math.sin(this.animTimer * 2.5) * 0.015;
    this.survivorMesh.scale.y = 1.0 + breath;

    // 3. Zombie jerky erratic twitching in shadows
    const twitch1 = Math.sin(this.animTimer * 7.5 + Math.sin(this.animTimer * 15)) * 0.04;
    this.zombie1.rotation.y = twitch1;
    this.zombie1.position.y = Math.abs(Math.sin(this.animTimer * 1.8)) * 0.04;

    const twitch2 = Math.cos(this.animTimer * 6.2) * 0.05;
    this.zombie2.rotation.y = twitch2;

    // Billboards face camera yaw
    const camAngle = this.camera.rotation.y;
    this.survivorMesh.rotation.y = camAngle;
    this.zombie1.rotation.y = camAngle + twitch1;
    this.zombie2.rotation.y = camAngle + twitch2;
  }
}
