import * as THREE from 'three';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { sounds } from '../audio/SoundEffects';
import { InventorySystem, ITEM_REGISTRY } from '../systems/InventorySystem';
import { SurvivalStats } from '../systems/SurvivalStats';

export interface InteractiveEntity {
  mesh: THREE.Object3D;
  position: THREE.Vector3;
  interactRadius: number;
  collisionRadius?: number;
  setOccluded?(occluded: boolean): void;
  getPrompt(inventory: InventorySystem): string | null;
  interact(inventory: InventorySystem, stats: SurvivalStats, delta: number): boolean;
  update(delta: number, camera: THREE.Camera): void;
  isAlive(): boolean;
  getHitProgress?(): number;
}

// --- WOOD CHIP PARTICLE SYSTEM ---
// --- EXPANDED PARTICLE SYSTEM (CHIPS, SPARKS, DUST, SPARKLES) ---
export class ParticleManager {
  private particles: {
    mesh: THREE.Mesh;
    vel: THREE.Vector3;
    life: number;
    maxLife: number;
    rotSpeed: THREE.Vector3;
    isSparkle?: boolean;
  }[] = [];
  private scene: THREE.Scene;

  // Reusable Geometries
  private chipGeo: THREE.BoxGeometry;
  private sparkGeo: THREE.BoxGeometry;
  private dustGeo: THREE.PlaneGeometry;
  private starGeo: THREE.PlaneGeometry;

  // Reusable Materials
  private woodMat: THREE.MeshLambertMaterial;
  private leafMat: THREE.MeshLambertMaterial;
  private stoneMat: THREE.MeshLambertMaterial;
  private sparkMat: THREE.MeshBasicMaterial;
  private dustMat: THREE.MeshBasicMaterial;
  private goldStarMat: THREE.MeshBasicMaterial;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.chipGeo = new THREE.BoxGeometry(0.12, 0.12, 0.12);
    this.sparkGeo = new THREE.BoxGeometry(0.08, 0.08, 0.08);
    this.dustGeo = new THREE.PlaneGeometry(0.25, 0.25);
    this.dustGeo.rotateX(-Math.PI / 2);
    this.starGeo = new THREE.PlaneGeometry(0.3, 0.3);

    this.woodMat = new THREE.MeshLambertMaterial({ color: 0x8a6234 });
    this.leafMat = new THREE.MeshLambertMaterial({ color: 0x2e6629 });
    this.stoneMat = new THREE.MeshLambertMaterial({ color: 0x6e706e });
    this.sparkMat = new THREE.MeshBasicMaterial({ color: 0xffcc33 });
    this.dustMat = new THREE.MeshBasicMaterial({ color: 0xb5a089, transparent: true, opacity: 0.45 });
    this.goldStarMat = new THREE.MeshBasicMaterial({ color: 0xffe066, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
  }

  // Wood & Pine Leaf Explosion on Chop
  public spawnWoodChips(pos: THREE.Vector3, count: number = 10) {
    for (let i = 0; i < count; i++) {
      const isLeaf = Math.random() < 0.45;
      const mesh = new THREE.Mesh(this.chipGeo, isLeaf ? this.leafMat : this.woodMat);
      mesh.position.copy(pos);
      mesh.position.y += 1.2 + Math.random() * 0.8;
      mesh.position.x += (Math.random() - 0.5) * 0.4;
      mesh.position.z += (Math.random() - 0.5) * 0.4;
      this.scene.add(mesh);

      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * 4.5,
        2.8 + Math.random() * 2.8,
        (Math.random() - 0.5) * 4.5
      );
      const rotSpeed = new THREE.Vector3(
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12
      );

      this.particles.push({ mesh, vel, life: 0.85, maxLife: 0.85, rotSpeed });
    }
  }

  // Sparks & Stone Debris on Mining
  public spawnStoneSparks(pos: THREE.Vector3, count: number = 8) {
    for (let i = 0; i < count; i++) {
      const isSpark = Math.random() < 0.5;
      const mesh = new THREE.Mesh(this.sparkGeo, isSpark ? this.sparkMat : this.stoneMat);
      mesh.position.copy(pos);
      mesh.position.y += 0.8;
      this.scene.add(mesh);

      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * 5.0,
        3.0 + Math.random() * 3.0,
        (Math.random() - 0.5) * 5.0
      );
      const rotSpeed = new THREE.Vector3(8, 8, 8);
      this.particles.push({ mesh, vel, life: 0.6, maxLife: 0.6, rotSpeed });
    }
  }

  // Dust Puffs while Running
  public spawnDustPuffs(pos: THREE.Vector3) {
    const mesh = new THREE.Mesh(this.dustGeo, this.dustMat);
    mesh.position.copy(pos);
    mesh.position.y = 0.05;
    this.scene.add(mesh);

    const vel = new THREE.Vector3((Math.random() - 0.5) * 0.4, 0.2, (Math.random() - 0.5) * 0.4);
    const rotSpeed = new THREE.Vector3(0, 0, 0);
    this.particles.push({ mesh, vel, life: 0.4, maxLife: 0.4, rotSpeed });
  }

  // Golden Spawn / Co-Op Celebratory Sparkles
  public spawnSpawnSparkles(pos: THREE.Vector3, count: number = 14) {
    for (let i = 0; i < count; i++) {
      const mesh = new THREE.Mesh(this.starGeo, this.goldStarMat);
      mesh.position.copy(pos);
      mesh.position.y += 0.5 + Math.random() * 1.5;
      this.scene.add(mesh);

      const angle = (i / count) * Math.PI * 2;
      const speed = 2.0 + Math.random() * 1.5;
      const vel = new THREE.Vector3(
        Math.cos(angle) * speed,
        2.2 + Math.random() * 2.0,
        Math.sin(angle) * speed
      );
      const rotSpeed = new THREE.Vector3(0, 0, (Math.random() - 0.5) * 10);
      this.particles.push({ mesh, vel, life: 1.0, maxLife: 1.0, rotSpeed, isSparkle: true });
    }
  }

  // Drifting Campfire Embers
  public spawnCampfireEmbers(pos: THREE.Vector3) {
    if (Math.random() > 0.4) return;
    const mesh = new THREE.Mesh(this.sparkGeo, this.sparkMat);
    mesh.position.set(pos.x + (Math.random() - 0.5) * 0.4, 0.8, pos.z + (Math.random() - 0.5) * 0.4);
    mesh.scale.set(0.5, 0.5, 0.5);
    this.scene.add(mesh);

    const vel = new THREE.Vector3(
      (Math.random() - 0.5) * 0.6,
      1.2 + Math.random() * 1.0,
      (Math.random() - 0.5) * 0.6
    );
    const rotSpeed = new THREE.Vector3(0, 0, 0);
    this.particles.push({ mesh, vel, life: 1.2, maxLife: 1.2, rotSpeed });
  }

  // Honk expanding sound-wave ring
  public spawnSoundWave(pos: THREE.Vector3) {
    const geo = new THREE.RingGeometry(0.3, 0.55, 32);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffea66,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos.x, 0.15, pos.z);
    this.scene.add(mesh);

    const startTime = Date.now();
    const duration = 0.65;
    const animateWave = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed >= duration) {
        this.scene.remove(mesh);
        geo.dispose();
        mat.dispose();
        return;
      }
      const t = elapsed / duration;
      const s = 1.0 + t * 4.5;
      mesh.scale.set(s, s, s);
      mat.opacity = 0.85 * (1.0 - t);
      requestAnimationFrame(animateWave);
    };
    requestAnimationFrame(animateWave);
  }

  public spawnSmoke(pos: THREE.Vector3) {
    this.spawnDustPuffs(pos);
  }

  public spawnHitSparks(pos: THREE.Vector3) {
    this.spawnStoneSparks(pos, 6);
  }

  public spawnGreenPuff(pos: THREE.Vector3) {
    for (let i = 0; i < 8; i++) {
      const mesh = new THREE.Mesh(this.dustGeo, new THREE.MeshBasicMaterial({ color: 0x70e000, transparent: true, opacity: 0.7 }));
      mesh.position.copy(pos);
      mesh.position.y += 0.5 + Math.random() * 0.5;
      this.scene.add(mesh);
      const vel = new THREE.Vector3((Math.random() - 0.5) * 1.5, 1.2 + Math.random() * 1.2, (Math.random() - 0.5) * 1.5);
      this.particles.push({ mesh, vel, life: 0.5, maxLife: 0.5, rotSpeed: new THREE.Vector3() });
    }
  }

  // Floating "+1 Wood" billboard text
  public spawnFloatingText(text: string, pos: THREE.Vector3, color: string = '#ffd15c') {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    ctx.font = 'bold 28px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#140f0a';
    ctx.lineWidth = 4;
    ctx.strokeText(text, 128, 32);
    ctx.fillStyle = color;
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(pos.x, pos.y + 1.6, pos.z);
    sprite.scale.set(1.6, 0.4, 1);
    this.scene.add(sprite);

    const startTime = Date.now();
    const duration = 0.95;
    const animateText = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed >= duration) {
        this.scene.remove(sprite);
        texture.dispose();
        mat.dispose();
        return;
      }
      const t = elapsed / duration;
      sprite.position.y += 0.025 * (1.0 - t);
      mat.opacity = Math.max(0, 1.0 - t * t);
      requestAnimationFrame(animateText);
    };
    requestAnimationFrame(animateText);
  }

  public update(delta: number) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.life -= delta;

      if (!p.isSparkle) {
        p.vel.y -= 9.8 * delta; // Gravity
      } else {
        p.vel.y -= 4.0 * delta; // Lighter float for sparkles
      }

      p.mesh.position.addScaledVector(p.vel, delta);
      p.mesh.rotation.x += p.rotSpeed.x * delta;
      p.mesh.rotation.y += p.rotSpeed.y * delta;
      p.mesh.rotation.z += p.rotSpeed.z * delta;

      // Scale down near end of life
      const scale = Math.max(0.05, p.life / p.maxLife);
      p.mesh.scale.set(scale, scale, scale);

      if (p.life <= 0 || p.mesh.position.y < 0.02) {
        this.scene.remove(p.mesh);
        this.particles.splice(i, 1);
      }
    }
  }
}

// --- ITEM DROP WITH POP PARABOLIC BOUNCE & FLOATING TEXT ---
export class ItemDrop implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 2.0;
  public itemId: string;
  private alive: boolean = true;
  private hoverTime: number = Math.random() * 10;
  private sprite: THREE.Sprite;
  private arcVel: THREE.Vector3;
  private currentY: number = 0.3;
  private isBouncing: boolean = true;
  private bounceCount: number = 0;
  private particleManager?: ParticleManager;

  constructor(scene: THREE.Scene, itemId: string, pos: THREE.Vector3, particles?: ParticleManager) {
    this.itemId = itemId;
    this.particleManager = particles;
    this.position = pos.clone();
    this.currentY = 0.3;

    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);

    // Initial pop physics velocity
    this.arcVel = new THREE.Vector3(
      (Math.random() - 0.5) * 2.8,
      3.2 + Math.random() * 0.8,
      (Math.random() - 0.5) * 2.8
    );

    // Canvas sprite for the drop
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    ctx.font = '40px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ITEM_REGISTRY[itemId]?.icon || '📦', 32, 34);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    this.sprite = new THREE.Sprite(mat);
    this.sprite.scale.set(0.8, 0.8, 1);
    this.mesh.add(this.sprite);

    // Shadow circle
    const shadowGeo = new THREE.CircleGeometry(0.3, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = -0.28;
    this.mesh.add(shadow);

    scene.add(this.mesh);
  }

  public getPrompt(): string | null {
    return `Pick up ${ITEM_REGISTRY[this.itemId]?.name || 'Item'}`;
  }

  public interact(inventory: InventorySystem): boolean {
    if (inventory.addItem(this.itemId, 1)) {
      sounds.playPick();
      this.alive = false;
      if (this.particleManager) {
        const name = ITEM_REGISTRY[this.itemId]?.name || 'Item';
        this.particleManager.spawnFloatingText(`+1 ${name}`, this.mesh.position, '#ffd15c');
        this.particleManager.spawnSpawnSparkles(this.mesh.position, 6);
      }
      return true;
    }
    return false;
  }

  public magnetTo(playerPos: THREE.Vector3, delta: number, inventory: InventorySystem) {
    const dist = this.mesh.position.distanceTo(playerPos);
    if (dist < 2.5) {
      const dir = new THREE.Vector3().subVectors(playerPos, this.mesh.position).normalize();
      this.mesh.position.addScaledVector(dir, delta * 7);
      if (dist < 0.6) {
        this.interact(inventory);
      }
    }
  }

  public update(delta: number) {
    if (this.isBouncing) {
      this.arcVel.y -= 11.0 * delta; // Gravity
      this.mesh.position.x += this.arcVel.x * delta;
      this.mesh.position.z += this.arcVel.z * delta;
      this.currentY += this.arcVel.y * delta;

      if (this.currentY <= 0.3) {
        this.currentY = 0.3;
        this.bounceCount++;
        if (this.bounceCount < 2) {
          this.arcVel.y = 1.6; // Secondary bounce
          this.arcVel.x *= 0.5;
          this.arcVel.z *= 0.5;
          this.sprite.scale.set(1.05, 0.58, 1); // Ground squash on impact
        } else {
          this.isBouncing = false;
          this.sprite.scale.set(0.8, 0.8, 1);
        }
      }
      this.mesh.position.y = this.currentY;
    } else {
      this.hoverTime += delta;
      this.sprite.position.y = Math.sin(this.hoverTime * 3) * 0.08;
      this.sprite.scale.x = 0.8 + Math.sin(this.hoverTime * 2) * 0.04;
      this.sprite.scale.y = 0.8 - Math.sin(this.hoverTime * 2) * 0.04;
    }
  }

  public isAlive(): boolean {
    return this.alive;
  }
}

// --- PINE TREE (CROSSED-QUAD 3D VOLUME, NORMAL MAP, LIVING WIND SWAY & DYNAMIC FELLING) ---
export class PineTree implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 2.4;
  private maxHits: number = 4;
  private currentHits: number = 0;
  private isStump: boolean = false;
  private shakeTimer: number = 0;
  private fallTimer: number = 0;
  private isFalling: boolean = false;
  private fallBounce: number = 0;
  private regrowTimer: number = 0;
  private stumpScale: number = 1.0;
  
  // Crossed-Quad Hierarchy: swayGroup topples and sways
  private swayGroup: THREE.Group;
  private quad1: THREE.Mesh;
  private quad2: THREE.Mesh;
  private quad3: THREE.Mesh;
  private treeMat: THREE.MeshStandardMaterial;
  private stumpMesh: THREE.Mesh;
  private particleManager: ParticleManager;
  private onFelledCallback: (pos: THREE.Vector3) => void;

  // Dynamic Camera Occlusion Transparency
  private isOccluded: boolean = false;
  private currentOpacity: number = 1.0;

  public get collisionRadius(): number {
    return this.isStump ? 0.45 : (this.isFalling ? 0 : 0.85);
  }

  constructor(scene: THREE.Scene, pos: THREE.Vector3, particles: ParticleManager, onFelled: (p: THREE.Vector3) => void) {
    this.position = pos.clone();
    this.particleManager = particles;
    this.onFelledCallback = onFelled;

    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    this.swayGroup = new THREE.Group();
    this.mesh.add(this.swayGroup);

    // Crossed-Quad Leaves & Trunk Geometry with 3D Normal Map reaction to torch/campfire
    const geo = new THREE.PlaneGeometry(3.0, 6.0);
    geo.translate(0, 3.0, 0); // Origin at ground base

    this.treeMat = new THREE.MeshStandardMaterial({
      map: TextureGenerator.getPineTreeTexture(),
      normalMap: TextureGenerator.getTreeNormalMap(),
      normalScale: new THREE.Vector2(1.2, 1.2),
      roughness: 0.85,
      metalness: 0.05,
      transparent: true,
      alphaTest: 0.15,
      side: THREE.DoubleSide,
    });

    // Asterisk (*) 3-Plane Intersecting Crossed-Quads for true 360-degree volume
    this.quad1 = new THREE.Mesh(geo, this.treeMat);
    this.quad2 = new THREE.Mesh(geo, this.treeMat);
    this.quad2.rotation.y = Math.PI / 3;
    this.quad3 = new THREE.Mesh(geo, this.treeMat);
    this.quad3.rotation.y = (Math.PI * 2) / 3;

    this.swayGroup.add(this.quad1);
    this.swayGroup.add(this.quad2);
    this.swayGroup.add(this.quad3);

    // Tree Stump (Hidden until felled)
    const stumpGeo = new THREE.PlaneGeometry(1.4, 1.4);
    stumpGeo.translate(0, 0.7, 0);
    const stumpMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getTreeStumpTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.stumpMesh = new THREE.Mesh(stumpGeo, stumpMat);
    this.stumpMesh.visible = false;
    this.mesh.add(this.stumpMesh);

    scene.add(this.mesh);
  }

  public setOccluded(occluded: boolean) {
    this.isOccluded = occluded;
  }

  public getPrompt(inventory: InventorySystem): string | null {
    if (this.isStump || this.isFalling) return null;
    const hasAxe = inventory.getSelectedItemId() === 'axe' || inventory.getItemCount('axe') > 0;
    return hasAxe ? 'Chop Pine Tree (Axe equipped)' : 'Chop Pine Tree (Bare hands)';
  }

  public interact(inventory: InventorySystem, stats: SurvivalStats): boolean {
    if (this.isStump || this.isFalling) return false;

    const hasAxe = inventory.getSelectedItemId() === 'axe' || inventory.getItemCount('axe') > 0;
    const damage = hasAxe ? 2 : 1;
    this.currentHits += damage;

    this.shakeTimer = 0.4;
    sounds.playChop();
    this.particleManager.spawnWoodChips(this.mesh.position, 10);

    if (this.currentHits >= this.maxHits) {
      this.isFalling = true;
      this.fallTimer = 0;
      this.fallBounce = 0;
      sounds.playTreeFall();
      stats.treesFelled += 1;
    }
    return true;
  }

  public update(delta: number, camera: THREE.Camera) {
    // Dynamic Smooth Occlusion Transparency Transition
    const targetOpacity = this.isOccluded ? 0.25 : 1.0;
    this.currentOpacity = THREE.MathUtils.lerp(this.currentOpacity, targetOpacity, delta * 9);
    this.treeMat.opacity = this.currentOpacity;
    this.treeMat.alphaTest = this.currentOpacity < 0.95 ? 0.02 : 0.15;
    this.isOccluded = false; // Reset each frame, refreshed by camera raycaster

    if (this.isFalling) {
      this.fallTimer += delta;
      // Gravity accelerating topple curve
      const toppleProgress = Math.min(1.0, this.fallTimer / 0.55);
      const toppleAngle = Math.pow(toppleProgress, 2.4) * (Math.PI / 2);
      this.swayGroup.rotation.z = toppleAngle;
      this.swayGroup.position.y = Math.max(0, -Math.sin(toppleAngle) * 0.5);

      if (toppleProgress >= 1.0) {
        // Ground bounce impact
        this.fallBounce += delta * 20;
        const bounce = Math.sin(this.fallBounce) * 0.15 * Math.exp(-this.fallBounce * 2.5);
        this.swayGroup.rotation.z = Math.PI / 2 - bounce;
      }

      if (this.fallTimer >= 0.72) {
        this.isFalling = false;
        this.isStump = true;
        this.stumpScale = 0.3;
        this.swayGroup.visible = false;
        this.stumpMesh.visible = true;
        this.regrowTimer = 180; // 3 minutes to regrow
        this.particleManager.spawnWoodChips(this.mesh.position, 16);
        this.onFelledCallback(this.mesh.position);
      }
      return;
    }

    // Stump rotates Y-axis only towards camera
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.stumpMesh.rotation.set(0, camEuler.y, 0);

    if (this.isStump) {
      // Stump pop-in scale tween (Back.easeOut)
      if (this.stumpScale < 1.0) {
        this.stumpScale = Math.min(1.0, this.stumpScale + delta * 3.5);
        const s = 1.70158;
        const t = this.stumpScale - 1;
        const pop = (t * t * ((s + 1) * t + s) + 1);
        this.stumpMesh.scale.set(pop, pop, pop);
      }
      // Stump regrowth timer
      this.regrowTimer -= delta;
      if (this.regrowTimer <= 0) {
        this.isStump = false;
        this.currentHits = 0;
        this.stumpMesh.visible = false;
        this.swayGroup.visible = true;
        this.swayGroup.rotation.set(0, 0, 0);
        this.swayGroup.position.set(0, 0, 0);
        this.particleManager.spawnSpawnSparkles(this.mesh.position, 8);
      }
      return;
    }

    // 1. Organic Shared Living Wind Sway across all 3 quads
    const now = Date.now() * 0.0018;
    const windPhase = now + this.position.x * 0.12 + this.position.z * 0.08;
    const baseSway = Math.sin(windPhase) * 0.045 + Math.sin(windPhase * 2.2) * 0.015;

    // 2. Shake reaction when chopped (Decaying sine wobble)
    let chopSway = 0;
    if (this.shakeTimer > 0) {
      this.shakeTimer -= delta;
      const elapsed = 0.4 - this.shakeTimer;
      chopSway = Math.sin(elapsed * 45) * Math.exp(-elapsed * 7) * 0.28;
    }

    this.swayGroup.rotation.z = baseSway + chopSway;
    this.swayGroup.position.x = chopSway * 0.5;
  }

  public isAlive(): boolean {
    return true; // Stumps remain in world and can regrow
  }

  public getHitProgress(): number {
    if (this.isStump || this.isFalling) return 0;
    return Math.max(0, (this.maxHits - this.currentHits) / this.maxHits);
  }
}

// --- BERRY BUSH (SHARED WIND SWAY & HARVEST RUSTLE) ---
export class BerryBush implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 2.0;
  public collisionRadius: number = 0.6;
  private hasBerries: boolean = true;
  private regrowTimer: number = 0;
  private rustleTimer: number = 0;
  private billboardMesh: THREE.Mesh;
  private onHarvestCallback: (pos: THREE.Vector3) => void;

  constructor(scene: THREE.Scene, pos: THREE.Vector3, onHarvest: (p: THREE.Vector3) => void) {
    this.position = pos.clone();
    this.onHarvestCallback = onHarvest;
    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    const geo = new THREE.PlaneGeometry(1.8, 1.8);
    geo.translate(0, 0.9, 0);
    const mat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getBerryBushTexture(true),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.billboardMesh = new THREE.Mesh(geo, mat);
    this.mesh.add(this.billboardMesh);
    scene.add(this.mesh);
  }

  public getPrompt(): string | null {
    return this.hasBerries ? 'Pick Juicy Berries' : null;
  }

  public interact(): boolean {
    if (!this.hasBerries) return false;
    this.hasBerries = false;
    this.regrowTimer = 50; // 50 seconds to regrow
    this.rustleTimer = 0.4;
    this.billboardMesh.material = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getBerryBushTexture(false),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    sounds.playPick();
    this.onHarvestCallback(this.mesh.position);
    return true;
  }

  public update(delta: number, camera: THREE.Camera) {
    // Y-axis billboarding only - strictly planted on the ground
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.billboardMesh.rotation.y = camEuler.y;

    const now = Date.now() * 0.0018;
    const windPhase = now + this.position.x * 0.12 + this.position.z * 0.08;
    const sway = Math.sin(windPhase) * 0.04 + Math.sin(windPhase * 2.1) * 0.015;

    if (this.rustleTimer > 0) {
      this.rustleTimer -= delta;
      const elapsed = 0.4 - this.rustleTimer;
      const rustle = Math.sin(elapsed * 40) * Math.exp(-elapsed * 6) * 0.2;
      this.billboardMesh.scale.set(1.0 + rustle, 1.0 - rustle * 0.8, 1.0);
      this.billboardMesh.rotation.z = sway + rustle * 0.4;
    } else {
      this.billboardMesh.scale.set(1.0, 1.0, 1.0);
      this.billboardMesh.rotation.z = sway;
    }

    if (!this.hasBerries) {
      this.regrowTimer -= delta;
      if (this.regrowTimer <= 0) {
        this.hasBerries = true;
        this.billboardMesh.material = new THREE.MeshLambertMaterial({
          map: TextureGenerator.getBerryBushTexture(true),
          transparent: true,
          alphaTest: 0.2,
          side: THREE.DoubleSide,
        });
      }
    }
  }

  public isAlive(): boolean {
    return true;
  }
}

// --- GRASS TUFT (SHARED WIND SWAY) ---
export class GrassTuft implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 1.8;
  private isPicked: boolean = false;
  private regrowTimer: number = 0;
  private billboardMesh: THREE.Mesh;
  private onHarvestCallback: (pos: THREE.Vector3) => void;

  constructor(scene: THREE.Scene, pos: THREE.Vector3, onHarvest: (p: THREE.Vector3) => void) {
    this.position = pos.clone();
    this.onHarvestCallback = onHarvest;
    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    const geo = new THREE.PlaneGeometry(1.4, 1.4);
    geo.translate(0, 0.7, 0);
    const mat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getGrassTuftTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.billboardMesh = new THREE.Mesh(geo, mat);
    this.mesh.add(this.billboardMesh);
    scene.add(this.mesh);
  }

  public getPrompt(): string | null {
    return this.isPicked ? null : 'Pick Cut Grass';
  }

  public interact(): boolean {
    if (this.isPicked) return false;
    this.isPicked = true;
    this.regrowTimer = 35;
    this.billboardMesh.visible = false;
    sounds.playPick();
    this.onHarvestCallback(this.mesh.position);
    return true;
  }

  public update(delta: number, camera: THREE.Camera) {
    // Y-axis billboarding only
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.billboardMesh.rotation.y = camEuler.y;

    const now = Date.now() * 0.0018;
    const windPhase = now + this.position.x * 0.12 + this.position.z * 0.08;
    this.billboardMesh.rotation.z = Math.sin(windPhase * 1.2) * 0.06;

    if (this.isPicked) {
      this.regrowTimer -= delta;
      if (this.regrowTimer <= 0) {
        this.isPicked = false;
        this.billboardMesh.visible = true;
      }
    }
  }

  public isAlive(): boolean {
    return true;
  }
}

// --- BOULDER (NORMAL MAP 3D SHADING, MINING SPARKS & SPRING VIBRATION) ---
export class Boulder implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 2.2;
  private hitsRemaining: number = 3;
  private shakeTimer: number = 0;
  private billboardMesh: THREE.Mesh;
  private particleManager: ParticleManager;
  private onMinedCallback: (pos: THREE.Vector3) => void;

  public get collisionRadius(): number {
    return this.hitsRemaining > 0 ? 1.05 : 0;
  }

  constructor(scene: THREE.Scene, pos: THREE.Vector3, particles: ParticleManager, onMined: (p: THREE.Vector3) => void) {
    this.position = pos.clone();
    this.particleManager = particles;
    this.onMinedCallback = onMined;
    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    const geo = new THREE.PlaneGeometry(2.0, 2.0);
    geo.translate(0, 1.0, 0);
    const mat = new THREE.MeshStandardMaterial({
      map: TextureGenerator.getBoulderTexture(),
      normalMap: TextureGenerator.getBoulderNormalMap(),
      normalScale: new THREE.Vector2(1.3, 1.3),
      roughness: 0.9,
      metalness: 0.08,
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.billboardMesh = new THREE.Mesh(geo, mat);
    this.mesh.add(this.billboardMesh);
    scene.add(this.mesh);
  }

  public getPrompt(): string | null {
    return this.hitsRemaining > 0 ? 'Mine Boulder for Flint' : null;
  }

  public interact(): boolean {
    if (this.hitsRemaining <= 0) return false;
    this.hitsRemaining -= 1;
    this.shakeTimer = 0.35;
    sounds.playChop();
    this.particleManager.spawnStoneSparks(this.mesh.position, 10);

    if (this.hitsRemaining <= 0) {
      this.mesh.visible = false;
      this.onMinedCallback(this.mesh.position);
    }
    return true;
  }

  public update(delta: number, camera: THREE.Camera) {
    // Y-axis billboarding only - grounded in 2.5D space
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.billboardMesh.rotation.y = camEuler.y;

    if (this.shakeTimer > 0) {
      this.shakeTimer -= delta;
      const elapsed = 0.35 - this.shakeTimer;
      const shake = Math.sin(elapsed * 55) * Math.exp(-elapsed * 8) * 0.18;
      this.billboardMesh.position.x = shake;
    } else {
      this.billboardMesh.position.x = 0;
    }
  }

  public isAlive(): boolean {
    return this.hitsRemaining > 0;
  }

  public getHitProgress(): number {
    return Math.max(0, this.hitsRemaining / 3);
  }
}

// --- CAMPFIRE ENTITY (MULTI-FREQUENCY FLAME & AMBER GLOW) ---
export class CampfireEntity implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 3.2;
  public collisionRadius: number = 0.85;
  public fuelSeconds: number = 240; // 240 seconds (4 minutes) base fire
  public pointLight: THREE.PointLight;
  private flameMesh: THREE.Mesh;
  private flickerTime: number = 0;
  private particleManager?: ParticleManager;

  constructor(scene: THREE.Scene, pos: THREE.Vector3, particles?: ParticleManager) {
    this.particleManager = particles;
    this.position = pos.clone();
    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    // Stone base on ground plane
    const baseGeo = new THREE.PlaneGeometry(1.6, 1.6);
    baseGeo.rotateX(-Math.PI / 2);
    const baseMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getCampfireBaseTexture(),
      transparent: true,
    });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = 0.02;
    this.mesh.add(base);

    // Animated Comic Flame billboard
    const flameGeo = new THREE.PlaneGeometry(1.2, 2.0);
    flameGeo.translate(0, 1.0, 0);
    const flameMat = new THREE.MeshBasicMaterial({
      map: TextureGenerator.getCampfireFlameTexture(),
      transparent: true,
      side: THREE.DoubleSide,
    });
    this.flameMesh = new THREE.Mesh(flameGeo, flameMat);
    this.mesh.add(this.flameMesh);

    // Dynamic 3D Point Light for warmth & repelling darkness
    this.pointLight = new THREE.PointLight(0xff7722, 2.2, 16, 1.2);
    this.pointLight.position.set(0, 1.2, 0);
    this.mesh.add(this.pointLight);

    scene.add(this.mesh);
  }

  public getPrompt(inventory: InventorySystem): string | null {
    if (inventory.getItemCount('log') > 0) return 'Add Log to Fuel Campfire [SPACE]';
    if (inventory.getItemCount('twigs') > 0) return 'Add Twigs to Fuel Campfire [SPACE]';
    return 'Campfire (Warm & Safe)';
  }

  public interact(inventory: InventorySystem): boolean {
    if (inventory.removeItem('log', 1)) {
      this.fuelSeconds = Math.min(360, this.fuelSeconds + 90);
      sounds.playCraft();
      return true;
    } else if (inventory.removeItem('twigs', 1)) {
      this.fuelSeconds = Math.min(360, this.fuelSeconds + 30);
      sounds.playCraft();
      return true;
    }
    return false;
  }

  public update(delta: number, camera: THREE.Camera) {
    this.fuelSeconds -= delta;
    this.flickerTime += delta;

    if (this.fuelSeconds <= 0) {
      this.pointLight.intensity = 0;
      this.flameMesh.visible = false;
      return;
    }

    // Multi-frequency organic flame flicker
    const f1 = Math.sin(this.flickerTime * 14) * 0.12;
    const f2 = Math.sin(this.flickerTime * 27) * 0.07;
    const f3 = Math.cos(this.flickerTime * 41) * 0.04;
    const flicker = f1 + f2 + f3;
    const intensity = Math.max(0.5, (this.fuelSeconds / 240) * 2.2 + flicker);
    this.pointLight.intensity = intensity;

    // Shift color subtly between amber and bright gold
    const colorMix = 0.5 + flicker * 2;
    this.pointLight.color.setRGB(1.0, 0.45 + colorMix * 0.2, 0.12 + colorMix * 0.1);

    // Organic flame scale breathing
    this.flameMesh.scale.set(1.0 + flicker * 0.5, 1.0 + flicker * 0.8, 1.0);

    // Drifting ember particles
    if (this.particleManager) {
      this.particleManager.spawnCampfireEmbers(this.position);
    }

    // Billboard facing camera
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.flameMesh.rotation.y = camEuler.y;
  }

  public isAlive(): boolean {
    return this.fuelSeconds > -10; // Stays as cold pit when burnt out
  }

  public getHitProgress(): number {
    return Math.max(0, Math.min(1.0, this.fuelSeconds / 240));
  }
}

// --- AMBIENT CRITTER: WILDERNESS BUNNY ---
export class AmbientBunny implements InteractiveEntity {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public interactRadius: number = 1.5;
  private billboard: THREE.Mesh;
  private hopTimer: number = Math.random() * 3;
  private isHopping: boolean = false;
  private hopProgress: number = 0;
  private hopDir: THREE.Vector3 = new THREE.Vector3();
  private hopOrigin: THREE.Vector3 = new THREE.Vector3();
  private hopTarget: THREE.Vector3 = new THREE.Vector3();
  private isScared: boolean = false;

  constructor(scene: THREE.Scene, pos: THREE.Vector3) {
    this.position = pos.clone();
    this.mesh = new THREE.Group();
    this.mesh.position.copy(pos);

    const geo = new THREE.PlaneGeometry(1.0, 1.0);
    geo.translate(0, 0.5, 0);
    const mat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getBunnyTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.billboard = new THREE.Mesh(geo, mat);
    this.mesh.add(this.billboard);
    scene.add(this.mesh);
  }

  public getPrompt(): string | null {
    return 'Playful Bunny';
  }

  public interact(): boolean {
    // Scares bunny into a fast cartoon sprint
    this.triggerFlee(this.position);
    return false;
  }

  public triggerFlee(fromPos: THREE.Vector3) {
    this.isScared = true;
    const fleeAngle = Math.atan2(this.position.z - fromPos.z, this.position.x - fromPos.x);
    this.hopDir.set(Math.cos(fleeAngle), 0, Math.sin(fleeAngle));
    this.isHopping = true;
    this.hopProgress = 0;
    this.hopOrigin.copy(this.mesh.position);
    this.hopTarget.copy(this.mesh.position).addScaledVector(this.hopDir, 3.5);
  }

  public update(delta: number, camera: THREE.Camera, playerPos?: THREE.Vector3) {
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.billboard.rotation.y = camEuler.y;

    // Check player proximity - scurry away if within 3.2m
    if (playerPos) {
      const dist = this.mesh.position.distanceTo(playerPos);
      if (dist < 3.2 && !this.isHopping) {
        this.triggerFlee(playerPos);
      }
    }

    if (this.isHopping) {
      const speed = this.isScared ? 3.5 : 2.0;
      this.hopProgress += delta * speed;
      const t = Math.min(1.0, this.hopProgress);

      this.mesh.position.lerpVectors(this.hopOrigin, this.hopTarget, t);
      // Parabolic jump arc
      const jumpY = Math.sin(t * Math.PI) * (this.isScared ? 0.7 : 0.4);
      this.billboard.position.y = jumpY;

      // Squash-and-stretch during jump
      const stretch = Math.sin(t * Math.PI) * 0.35;
      this.billboard.scale.set(1.0 - stretch * 0.5, 1.0 + stretch, 1.0);

      if (t >= 1.0) {
        this.isHopping = false;
        this.isScared = false;
        this.billboard.position.y = 0;
        this.billboard.scale.set(1, 1, 1);
        this.hopTimer = 2.0 + Math.random() * 4.0;
      }
    } else {
      this.hopTimer -= delta;
      // Idle twitch / nose wiggle
      const twitch = Math.sin(Date.now() * 0.008) * 0.04;
      this.billboard.scale.set(1.0 + twitch, 1.0 - twitch, 1.0);

      if (this.hopTimer <= 0) {
        this.isHopping = true;
        this.hopProgress = 0;
        this.hopOrigin.copy(this.mesh.position);
        const randomAngle = Math.random() * Math.PI * 2;
        const hopDist = 0.8 + Math.random() * 1.2;
        this.hopTarget.set(
          this.mesh.position.x + Math.cos(randomAngle) * hopDist,
          0,
          this.mesh.position.z + Math.sin(randomAngle) * hopDist
        );
      }
    }
    this.position.copy(this.mesh.position);
  }

  public isAlive(): boolean {
    return true;
  }
}

// --- AMBIENT ECOSYSTEM (FIREFLIES, DRIFTING LEAVES, CREEPING TERROR EYES) ---
export class AmbientEcosystem {
  private scene: THREE.Scene;
  private fireflies: THREE.Sprite[] = [];
  private leaves: { mesh: THREE.Mesh; vel: THREE.Vector3; rotSpeed: THREE.Vector3 }[] = [];
  private terrorEyes: THREE.Sprite;
  private leafGeo: THREE.PlaneGeometry;
  private leafMat: THREE.MeshLambertMaterial;

  constructor(scene: THREE.Scene) {
    this.scene = scene;

    // 1. Fireflies (24 soft glowing orbs)
    const fireflyMat = new THREE.SpriteMaterial({
      map: TextureGenerator.getFireflyTexture(),
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    for (let i = 0; i < 24; i++) {
      const sprite = new THREE.Sprite(fireflyMat.clone());
      sprite.scale.set(0.65, 0.65, 1);
      sprite.position.set(
        (Math.random() - 0.5) * 45,
        0.5 + Math.random() * 2.2,
        (Math.random() - 0.5) * 45
      );
      this.scene.add(sprite);
      this.fireflies.push(sprite);
    }

    // 2. Drifting Wind Leaves (16 gentle fluttering leaves)
    this.leafGeo = new THREE.PlaneGeometry(0.35, 0.35);
    this.leafMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getDriftingLeafTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    for (let i = 0; i < 16; i++) {
      const mesh = new THREE.Mesh(this.leafGeo, this.leafMat);
      mesh.position.set(
        (Math.random() - 0.5) * 40,
        1.5 + Math.random() * 4.0,
        (Math.random() - 0.5) * 40
      );
      mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
      this.scene.add(mesh);
      this.leaves.push({
        mesh,
        vel: new THREE.Vector3(1.2 + Math.random() * 1.5, -0.4 - Math.random() * 0.4, 0.8 + Math.random() * 0.8),
        rotSpeed: new THREE.Vector3((Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4),
      });
    }

    // 3. Creeping Terror Eyes (peering during deadly pitch black)
    const eyesMat = new THREE.SpriteMaterial({
      map: TextureGenerator.getWarningEyesTexture(),
      transparent: true,
      opacity: 0,
      depthTest: false,
    });
    this.terrorEyes = new THREE.Sprite(eyesMat);
    this.terrorEyes.scale.set(1.4, 0.7, 1);
    this.terrorEyes.position.set(0, 0.6, -1000);
    this.scene.add(this.terrorEyes);
  }

  public update(delta: number, playerPos: THREE.Vector3, isNight: boolean, isPlayerInLight: boolean) {
    const time = Date.now() * 0.001;

    // Update Fireflies: active at Dusk & Night, asleep during bright Day
    const targetFireflyOpacity = isNight ? 0.9 : 0.0;
    for (let i = 0; i < this.fireflies.length; i++) {
      const ff = this.fireflies[i];
      // Keep fireflies clustered around player's active area
      if (ff.position.distanceTo(playerPos) > 28) {
        ff.position.set(
          playerPos.x + (Math.random() - 0.5) * 36,
          0.4 + Math.random() * 2.0,
          playerPos.z + (Math.random() - 0.5) * 36
        );
      }
      // Lazy wandering sine wave
      const phase = time * 1.5 + i * 2.1;
      ff.position.x += Math.cos(phase * 0.8) * delta * 0.6;
      ff.position.y = 0.5 + Math.sin(phase * 1.2) * 0.35;
      ff.position.z += Math.sin(phase * 0.7) * delta * 0.6;

      const mat = ff.material as THREE.SpriteMaterial;
      const pulse = 0.7 + Math.sin(time * 3 + i) * 0.3;
      mat.opacity += (targetFireflyOpacity * pulse - mat.opacity) * delta * 2.0;
    }

    // Update Drifting Leaves
    for (const leaf of this.leaves) {
      leaf.mesh.position.addScaledVector(leaf.vel, delta);
      leaf.mesh.rotation.x += leaf.rotSpeed.x * delta;
      leaf.mesh.rotation.y += leaf.rotSpeed.y * delta;
      leaf.mesh.rotation.z += leaf.rotSpeed.z * delta;

      // Respawn leaf upstream if drifted below ground or too far from player
      if (leaf.mesh.position.y <= 0.05 || leaf.mesh.position.distanceTo(playerPos) > 30) {
        leaf.mesh.position.set(
          playerPos.x - 15 + (Math.random() - 0.5) * 12,
          2.5 + Math.random() * 3.5,
          playerPos.z - 15 + (Math.random() - 0.5) * 12
        );
      }
    }

    // Update Creeping Terror Eyes in the dark
    const eyesMat = this.terrorEyes.material as THREE.SpriteMaterial;
    if (isNight && !isPlayerInLight) {
      // Place eyes at the perimeter of view
      const angle = time * 0.6;
      const dist = 6.5;
      this.terrorEyes.position.set(
        playerPos.x + Math.cos(angle) * dist,
        0.65 + Math.sin(time * 2) * 0.08,
        playerPos.z + Math.sin(angle) * dist
      );
      // Blinking eye animation
      const blink = Math.sin(time * 4) > 0.85 ? 0 : 1;
      eyesMat.opacity += (0.95 * blink - eyesMat.opacity) * delta * 3.5;
    } else {
      eyesMat.opacity = Math.max(0, eyesMat.opacity - delta * 4.0);
    }
  }
}

