import * as THREE from 'three';
import { ITEM_REGISTRY, ItemRarity } from '../registries/ItemDef';
import { InventorySystem } from '../systems/InventorySystem';
import { sounds } from '../audio/SoundEffects';

export interface ParabolicLootOptions {
  itemId: string;
  count: number;
  origin: THREE.Vector3;
  angleRad?: number;
  upwardSpeed?: number;
  radialSpeed?: number;
  rarity?: ItemRarity;
}

/**
 * ParabolicLoot (Section 15, Section 66 & Section 75).
 * Simulates ballistic 3D parabolic arcs for loot items ejected from opened containers,
 * with multi-bounce ground settle, squash-and-stretch elasticity, rarity glints, and magnet collection.
 */
export class ParabolicLoot {
  public mesh: THREE.Group;
  public itemId: string;
  public count: number;
  public rarity: ItemRarity;
  public position: THREE.Vector3;
  public velocity: THREE.Vector3;

  public isSettled: boolean = false;
  public isAlive: boolean = true;

  private gravity: number = -11.0;
  private bounces: number = 0;
  private maxBounces: number = 2;
  private restitution: number = 0.42;
  private age: number = 0;
  private squashTimer: number = 0;

  private spriteGroup: THREE.Group;
  private shadowMesh: THREE.Mesh;
  private rarityAura?: THREE.Mesh;

  constructor(scene: THREE.Scene, opts: ParabolicLootOptions) {
    this.itemId = opts.itemId;
    this.count = opts.count;
    this.position = opts.origin.clone();

    const def = ITEM_REGISTRY[this.itemId];
    this.rarity = opts.rarity || def?.rarity || 'C';

    // Ballistic Launch Velocity
    const angle = opts.angleRad !== undefined ? opts.angleRad : Math.random() * Math.PI * 2;
    const vY = opts.upwardSpeed !== undefined ? opts.upwardSpeed : 3.6 + Math.random() * 1.4;
    const vXZ = opts.radialSpeed !== undefined ? opts.radialSpeed : 1.6 + Math.random() * 1.5;

    this.velocity = new THREE.Vector3(
      Math.cos(angle) * vXZ,
      vY,
      Math.sin(angle) * vXZ
    );

    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);

    // 1. Sprite Billboard Group
    this.spriteGroup = new THREE.Group();
    this.mesh.add(this.spriteGroup);

    const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : ({} as any);
    if (typeof document !== 'undefined') {
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d')!;
      ctx.font = '38px serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(def?.icon || '📦', 32, 34);

      if (this.count > 1) {
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.font = 'bold 16px monospace';
        ctx.strokeText(`x${this.count}`, 46, 52);
        ctx.fillText(`x${this.count}`, 46, 52);
      }

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.scale.set(0.85, 0.85, 1);
      this.spriteGroup.add(sprite);
    }

    // 2. Rarity Glint Aura
    if (this.rarity === 'U' || this.rarity === 'R' || this.rarity === 'X') {
      const auraColor = this.rarity === 'X' ? 0xff0055 : this.rarity === 'R' ? 0xffd700 : 0x00f0ff;
      const auraGeo = new THREE.RingGeometry(0.2, 0.45, 16);
      auraGeo.rotateX(-Math.PI / 2);
      const auraMat = new THREE.MeshBasicMaterial({
        color: auraColor,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
      });
      this.rarityAura = new THREE.Mesh(auraGeo, auraMat);
      this.rarityAura.position.y = 0.05;
      this.mesh.add(this.rarityAura);
    }

    // 3. Ground Shadow Decal
    const shadowGeo = new THREE.CircleGeometry(0.3, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.35,
    });
    this.shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    this.shadowMesh.position.y = 0.02;
    this.mesh.add(this.shadowMesh);

    scene.add(this.mesh);
  }

  public update(delta: number, playerPos: THREE.Vector3, inventory: InventorySystem, particles?: any) {
    if (!this.isAlive) return;
    this.age += delta;

    if (!this.isSettled) {
      // 1. Ballistic Trajectory
      this.velocity.y += this.gravity * delta;
      this.position.addScaledVector(this.velocity, delta);

      // Rotate sprite in flight
      this.spriteGroup.rotation.z += delta * 6;

      // Ground Contact Check
      if (this.position.y <= 0.25) {
        if (this.bounces < this.maxBounces) {
          this.position.y = 0.25;
          this.velocity.y = -this.velocity.y * this.restitution;
          this.velocity.x *= 0.6;
          this.velocity.z *= 0.6;
          this.bounces++;
          this.squashTimer = 0.12;

          // Rubbery Squash on Impact
          this.spriteGroup.scale.set(1.35, 0.65, 1.35);

          if (particles && typeof particles.spawnDustPuff === 'function') {
            particles.spawnDustPuff(this.position);
          }
        } else {
          // Settled on ground
          this.position.y = 0.25;
          this.velocity.set(0, 0, 0);
          this.isSettled = true;
          this.spriteGroup.rotation.z = 0;
          this.spriteGroup.scale.set(1, 1, 1);
        }
      }

      // Smooth squash recovery
      if (this.squashTimer > 0) {
        this.squashTimer -= delta;
        if (this.squashTimer <= 0) {
          this.spriteGroup.scale.set(1, 1, 1);
        }
      }
    } else {
      // 2. Settled state: gentle floating bob
      this.position.y = 0.25 + Math.sin(this.age * 4.0) * 0.06;

      // Rotate rarity ring aura
      if (this.rarityAura) {
        this.rarityAura.rotation.z += delta * 2.0;
        const pulse = 0.5 + Math.sin(this.age * 5.0) * 0.2;
        (this.rarityAura.material as THREE.MeshBasicMaterial).opacity = pulse;
      }

      // 3. Magnetize to player when close
      const dist = this.position.distanceTo(playerPos);
      if (dist < 2.5) {
        const dir = new THREE.Vector3().subVectors(playerPos, this.position).normalize();
        this.position.addScaledVector(dir, delta * 7.5);

        if (dist < 0.6) {
          this.collect(inventory, particles);
        }
      }
    }

    this.mesh.position.copy(this.position);
  }

  public collect(inventory: InventorySystem, particles?: any) {
    if (!this.isAlive) return;

    if (inventory.addItem(this.itemId, this.count)) {
      this.isAlive = false;
      sounds.playPick();

      const def = ITEM_REGISTRY[this.itemId];
      const name = def?.name || 'Item';
      const color = this.rarity === 'X' ? '#FF0055' : this.rarity === 'R' ? '#FFD700' : '#4ADE80';

      if (particles && typeof particles.spawnFloatingText === 'function') {
        particles.spawnFloatingText(`+${this.count} ${name}`, this.position, color);
      }
      if (particles && typeof particles.spawnSpawnSparkles === 'function') {
        particles.spawnSpawnSparkles(this.position, 6);
      }
    }
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.mesh);
  }
}

/**
 * ParabolicLootManager:
 * Coordinates all active parabolic loot arcs in the world scene.
 */
export class ParabolicLootManager {
  private activeLoot: ParabolicLoot[] = [];
  private scene: THREE.Scene;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  public spawnArc(opts: ParabolicLootOptions): ParabolicLoot {
    const loot = new ParabolicLoot(this.scene, opts);
    this.activeLoot.push(loot);
    return loot;
  }

  public update(delta: number, playerPos: THREE.Vector3, inventory: InventorySystem, particles?: any) {
    for (let i = this.activeLoot.length - 1; i >= 0; i--) {
      const loot = this.activeLoot[i];
      loot.update(delta, playerPos, inventory, particles);

      if (!loot.isAlive) {
        loot.destroy(this.scene);
        this.activeLoot.splice(i, 1);
      }
    }
  }

  public getActiveCount(): number {
    return this.activeLoot.length;
  }

  public clear() {
    for (const loot of this.activeLoot) {
      loot.destroy(this.scene);
    }
    this.activeLoot = [];
  }
}
