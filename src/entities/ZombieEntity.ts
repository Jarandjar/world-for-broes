import * as THREE from 'three';
import { ZOMBIE_REGISTRY, ZombieDef, ZombieAIState } from '../registries/ZombieDef';
import { DampedSpring, SpringPreset } from '../animation/TweenPreset';
import { ComicPops } from '../ui/ComicPops';
import { sounds } from '../audio/SoundEffects';
import { gameConfig } from '../config/GameConfig';

export interface ZombieEntityOptions {
  type: string;
  position: THREE.Vector3;
  rotationY?: number;
}

/**
 * ZombieEntity (Section 14, Section 65 & Section 78-79).
 * Features:
 * - 2D/3D diorama cutout puppet with lumpy potato silhouette and mismatched googly eyes.
 * - Desynchronized unique gait (randomized stride frequency and phase offset phi).
 * - Full AI state machine: wander, investigate, chase, attack, stunned, flee_fire.
 * - Sensory perception: day/night sight range, hearing noise rings, hiding spot aggro drop.
 * - Armor hits: traffic cone absorbs first strike, flying off with a rubbery thwack.
 * - Comic green smoke puff defeat FX with zero gore.
 */
export class ZombieEntity {
  public def: ZombieDef;
  public position: THREE.Vector3;
  public velocity: THREE.Vector3 = new THREE.Vector3();
  public state: ZombieAIState = 'wander';
  public health: number;
  public armorHits: number = 0;
  public isAlive: boolean = true;

  public mesh: THREE.Group;
  public lodLevel: number = 0; // 0: <15m, 1: 15-35m, 2: >35m

  // Desynchronized animation offsets
  private walkFrequency: number;
  private phaseOffset: number;
  private animTimer: number = 0;
  private jawSpring: DampedSpring;

  // AI Timers & Targets
  private wanderTarget: THREE.Vector3 = new THREE.Vector3();
  private wanderTimer: number = 0;
  private attackCooldown: number = 0;
  private hitStopTimer: number = 0;
  private squashTimer: number = 0;

  // Visual sub-meshes
  private headGroup!: THREE.Group;
  private jawMesh!: THREE.Mesh;
  private armL!: THREE.Group;
  private armR!: THREE.Group;
  private eyeL!: THREE.Mesh;
  private eyeR!: THREE.Mesh;
  private coneHatMesh?: THREE.Mesh;

  // Throttled LOD update accumulator
  private lodAccumulator: number = 0;

  constructor(scene: THREE.Scene, opts: ZombieEntityOptions) {
    this.def = ZOMBIE_REGISTRY[opts.type] || ZOMBIE_REGISTRY['shambler'];
    this.position = opts.position.clone();
    this.health = this.def.maxHealth;
    this.armorHits = this.def.armorHits || 0;

    // Desynchronize walk cycle (0.90x to 1.15x speed, phase offset 0 to 2pi)
    this.walkFrequency = (3.2 + Math.random() * 0.8) * (0.9 + Math.random() * 0.25);
    this.phaseOffset = Math.random() * Math.PI * 2;
    this.jawSpring = new DampedSpring(SpringPreset.loose, 0.0);

    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = opts.rotationY || Math.random() * Math.PI * 2;

    this.buildZombieMesh();
    this.pickNewWanderTarget();

    scene.add(this.mesh);
  }

  private buildZombieMesh() {
    // 1. Lumpy Potato Torso (Dodecahedron with irregular scaling)
    const torsoColor = this.def.id === 'jogger' ? 0x2e6b8c : this.def.id === 'screamer' ? 0x6a3275 : 0x4a5d23;
    const torsoGeo = new THREE.DodecahedronGeometry(0.52, 1);
    const torsoMat = new THREE.MeshLambertMaterial({ color: torsoColor });
    const torso = new THREE.Mesh(torsoGeo, torsoMat);
    torso.position.y = 0.82;
    torso.scale.set(0.9, 1.25, 0.85);
    torso.castShadow = true;
    this.mesh.add(torso);

    // 2. Lumpy Asymmetrical Head
    this.headGroup = new THREE.Group();
    this.headGroup.position.set(0, 1.55, 0.08);

    const headColor = this.def.id === 'jogger' ? 0x4a856a : this.def.id === 'screamer' ? 0x8a458a : 0x5a7d2b;
    const headGeo = new THREE.DodecahedronGeometry(0.42, 1);
    const headMat = new THREE.MeshLambertMaterial({ color: headColor });
    const head = new THREE.Mesh(headGeo, headMat);
    this.headGroup.add(head);

    // 3. Mismatched Googly Eyes (one big, one small)
    const eyeWhiteGeoLarge = new THREE.CircleGeometry(0.13, 10);
    const eyeWhiteGeoSmall = new THREE.CircleGeometry(0.08, 8);
    const eyeWhiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const pupilGeo = new THREE.CircleGeometry(0.04, 6);
    const pupilMat = new THREE.MeshBasicMaterial({ color: 0x111111 });

    // Left Eye (Big)
    const eyeLWhite = new THREE.Mesh(eyeWhiteGeoLarge, eyeWhiteMat);
    eyeLWhite.position.set(-0.16, 0.06, 0.38);
    this.eyeL = new THREE.Mesh(pupilGeo, pupilMat);
    this.eyeL.position.set(0.02, -0.02, 0.01);
    eyeLWhite.add(this.eyeL);
    this.headGroup.add(eyeLWhite);

    // Right Eye (Small, droopy)
    const eyeRWhite = new THREE.Mesh(eyeWhiteGeoSmall, eyeWhiteMat);
    eyeRWhite.position.set(0.18, -0.04, 0.38);
    this.eyeR = new THREE.Mesh(pupilGeo, pupilMat);
    this.eyeR.position.set(-0.01, 0.01, 0.01);
    eyeRWhite.add(this.eyeR);
    this.headGroup.add(eyeRWhite);

    // 4. Dropped Wobbly Lower Jaw
    const jawGeo = new THREE.BoxGeometry(0.28, 0.16, 0.28);
    const jawMat = new THREE.MeshLambertMaterial({ color: 0x3d541c });
    this.jawMesh = new THREE.Mesh(jawGeo, jawMat);
    this.jawMesh.position.set(0, -0.28, 0.18);
    this.headGroup.add(this.jawMesh);

    // 5. Traffic Cone Armor Hat (for traffic_cone_zombie)
    if (this.def.id === 'traffic_cone_zombie') {
      const coneGeo = new THREE.ConeGeometry(0.28, 0.65, 8);
      const coneMat = new THREE.MeshLambertMaterial({ color: 0xff6600 });
      this.coneHatMesh = new THREE.Mesh(coneGeo, coneMat);
      this.coneHatMesh.position.set(0, 0.45, -0.05);
      this.coneHatMesh.rotation.z = 0.15; // Slanted goofy angle
      this.headGroup.add(this.coneHatMesh);
    }

    this.mesh.add(this.headGroup);

    // 6. Floppy Noodle Arms
    const armGeo = new THREE.CylinderGeometry(0.09, 0.07, 0.75, 6);
    armGeo.translate(0, -0.35, 0);
    const armMat = new THREE.MeshLambertMaterial({ color: headColor });

    this.armL = new THREE.Group();
    this.armL.position.set(-0.52, 1.25, 0);
    const armLMesh = new THREE.Mesh(armGeo, armMat);
    this.armL.add(armLMesh);
    this.mesh.add(this.armL);

    this.armR = new THREE.Group();
    this.armR.position.set(0.52, 1.25, 0);
    const armRMesh = new THREE.Mesh(armGeo, armMat);
    this.armR.add(armRMesh);

    // Highway Hulk Boss Road-Sign Club (Phase Z7)
    if (this.def.id === 'highway_hulk' || this.def.isBoss) {
      this.mesh.scale.set(2.2, 2.2, 2.2);

      const postGeo = new THREE.CylinderGeometry(0.04, 0.04, 1.4, 6);
      postGeo.translate(0, -0.4, 0);
      const postMat = new THREE.MeshLambertMaterial({ color: 0x888888 });
      const post = new THREE.Mesh(postGeo, postMat);

      const signGeo = new THREE.CircleGeometry(0.32, 8);
      const signMat = new THREE.MeshBasicMaterial({ color: 0xd63031, side: THREE.DoubleSide });
      const stopSign = new THREE.Mesh(signGeo, signMat);
      stopSign.position.set(0, -1.0, 0.04);
      post.add(stopSign);

      this.armR.add(post);
    }

    this.mesh.add(this.armR);

    // 7. Ground Shadow
    const shadowGeo = new THREE.CircleGeometry(0.42, 12);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = 0.02;
    this.mesh.add(shadow);
  }

  public update(
    delta: number,
    playerPos: THREE.Vector3,
    isPlayerHiding: boolean,
    isNight: boolean,
    onPlayerDamage?: (damage: number) => void,
    campfires?: { position: THREE.Vector3; radius: number }[]
  ) {
    if (!this.isAlive) return;

    // Hit-stop isolated freeze
    if (this.hitStopTimer > 0) {
      this.hitStopTimer -= delta;
      return;
    }

    const distToPlayer = this.position.distanceTo(playerPos);

    // Compute LOD level based on distance to player
    if (distToPlayer < 15.0) {
      this.lodLevel = 0; // Full 60Hz update
    } else if (distToPlayer < 35.0) {
      this.lodLevel = 1; // 20Hz update (skip 2 out of 3 frames)
    } else {
      this.lodLevel = 2; // 8Hz update
    }

    // LOD throttling
    this.lodAccumulator += delta;
    const targetInterval = this.lodLevel === 0 ? 0 : this.lodLevel === 1 ? 0.05 : 0.125;
    if (this.lodAccumulator < targetInterval) {
      return; // Skip throttled frame
    }
    const effectiveDt = this.lodAccumulator;
    this.lodAccumulator = 0;

    this.animTimer += effectiveDt;
    if (this.attackCooldown > 0) this.attackCooldown -= effectiveDt;

    // 1. AI Decision Loop
    this.updateAIState(distToPlayer, playerPos, isPlayerHiding, isNight, campfires);

    // 2. Movement & Locomotion
    this.updateMovement(effectiveDt, playerPos, onPlayerDamage);

    // 3. Procedural Gait & Squash Animation
    this.updateProceduralGait(effectiveDt);
  }

  private updateAIState(
    distToPlayer: number,
    playerPos: THREE.Vector3,
    isPlayerHiding: boolean,
    isNight: boolean,
    campfires?: { position: THREE.Vector3; radius: number }[]
  ) {
    if (this.state === 'stunned') return;

    // Check campfire warmth avoidance (fear of fire)
    if (campfires && campfires.length > 0) {
      for (const fire of campfires) {
        if (this.position.distanceTo(fire.position) < fire.radius + 1.2) {
          this.state = 'flee_fire';
          return;
        }
      }
    }

    // Sensory Sight Perception
    const sightRadius = isNight ? this.def.sightRadiusNight : this.def.sightRadius;

    // Hiding spot aggro drop: if player is inside a wardrobe/dumpster/bed, break sight!
    if (isPlayerHiding) {
      if (this.state === 'chase' || this.state === 'attack') {
        this.state = 'investigate'; // Drop chase to wandering investigation
        this.pickNewWanderTarget();
      }
      return;
    }

    if (distToPlayer <= sightRadius) {
      if (distToPlayer <= 1.4) {
        this.state = 'attack';
      } else {
        this.state = 'chase';
      }
    } else if (this.state === 'chase' && distToPlayer > sightRadius * 1.35) {
      // Lost sight of player
      this.state = 'wander';
      this.pickNewWanderTarget();
    }
  }

  private updateMovement(
    dt: number,
    playerPos: THREE.Vector3,
    onPlayerDamage?: (damage: number) => void
  ) {
    let moveDir = new THREE.Vector3();
    let speed = 0;

    switch (this.state) {
      case 'chase': {
        moveDir.subVectors(playerPos, this.position).setY(0).normalize();
        speed = this.def.chaseSpeed;
        break;
      }
      case 'wander':
      case 'investigate': {
        this.wanderTimer -= dt;
        if (this.wanderTimer <= 0) {
          this.pickNewWanderTarget();
        }
        moveDir.subVectors(this.wanderTarget, this.position).setY(0);
        if (moveDir.length() > 0.5) {
          moveDir.normalize();
          speed = this.def.walkSpeed;
        } else {
          speed = 0;
        }
        break;
      }
      case 'attack': {
        speed = 0;
        if (this.attackCooldown <= 0) {
          this.attackCooldown = this.def.attackIntervalSeconds;
          sounds.playChop();
          if (onPlayerDamage) {
            onPlayerDamage(this.def.damage);
          }
          // Arm swing forward
          this.armL.rotation.x = -1.2;
          this.armR.rotation.x = -1.2;
        }
        break;
      }
      case 'flee_fire': {
        // Run backward away from center/fire
        moveDir.copy(this.velocity).negate().setY(0).normalize();
        speed = this.def.walkSpeed * 1.2;
        break;
      }
    }

    if (speed > 0) {
      this.velocity.copy(moveDir).multiplyScalar(speed);
      this.position.addScaledVector(this.velocity, dt);

      // Rotate facing towards movement direction
      const targetAngle = Math.atan2(moveDir.x, moveDir.z);
      this.mesh.rotation.y = THREE.MathUtils.lerp(this.mesh.rotation.y, targetAngle, dt * 8);
    } else {
      this.velocity.set(0, 0, 0);
    }

    this.mesh.position.copy(this.position);
  }

  private updateProceduralGait(dt: number) {
    const isMoving = this.velocity.lengthSq() > 0.05;

    if (isMoving && this.lodLevel <= 1) {
      const cycle = this.animTimer * this.walkFrequency + this.phaseOffset;
      const bob = Math.sin(cycle) * 0.08;
      this.mesh.position.y = Math.max(0, bob);

      // Flapping noodle arms
      this.armL.rotation.x = Math.sin(cycle) * 0.75 - 0.4;
      this.armR.rotation.x = -Math.sin(cycle) * 0.75 - 0.4;

      // Asymmetrical wobbly head tilt
      this.headGroup.rotation.z = Math.sin(cycle * 0.5) * 0.12;
      this.headGroup.rotation.x = 0.15 + Math.sin(cycle) * 0.06;

      // Googly eyes wandering pupils
      if (this.lodLevel === 0) {
        this.eyeL.position.x = 0.02 + Math.sin(cycle * 0.7) * 0.03;
        this.eyeR.position.y = -0.02 + Math.cos(cycle * 0.9) * 0.03;
      }

      // Dropped wobbly jaw spring
      const jawTarget = Math.sin(cycle) * 0.12;
      this.jawMesh.position.y = -0.28 + this.jawSpring.update(dt, jawTarget);
    } else {
      this.armL.rotation.x = THREE.MathUtils.lerp(this.armL.rotation.x, -0.3, dt * 5);
      this.armR.rotation.x = THREE.MathUtils.lerp(this.armR.rotation.x, -0.3, dt * 5);
    }

    // Squash recovery
    if (this.squashTimer > 0) {
      this.squashTimer -= dt;
      if (this.squashTimer <= 0) {
        this.mesh.scale.set(1, 1, 1);
      }
    }
  }

  public takeDamage(
    amount: number,
    knockbackDir: THREE.Vector3,
    particles?: any,
    camera?: THREE.Camera
  ): { died: boolean; def: ZombieDef } {
    if (!this.isAlive) return { died: true, def: this.def };

    // Cone Armor absorption check
    if (this.armorHits > 0) {
      this.armorHits--;
      sounds.playRubberChicken();
      if (camera) {
        ComicPops.spawnWorld(this.position, camera, 'THWACK! 🪓', '#FF9900');
      }
      // Detach cone hat and send flying
      if (this.coneHatMesh) {
        this.headGroup.remove(this.coneHatMesh);
        this.coneHatMesh = undefined;
      }
      this.hitStopTimer = 0.045; // 45ms hit stop
      return { died: false, def: this.def };
    }

    this.health -= amount;
    this.hitStopTimer = 0.045; // 45ms hit-stop
    this.squashTimer = 0.14;
    this.mesh.scale.set(1.4, 0.6, 1.4); // Slapstick squash flinch

    // Knockback
    this.position.addScaledVector(knockbackDir.normalize(), 0.8);
    this.mesh.position.copy(this.position);

    if (this.health <= 0) {
      this.isAlive = false;
      this.die(particles, camera);
      return { died: true, def: this.def };
    } else {
      sounds.playHurt();
      this.state = 'chase';
      return { died: false, def: this.def };
    }
  }

  private die(particles?: any, camera?: THREE.Camera) {
    sounds.playBubblePop();

    // Spawns comic onomatopoeia POP!
    if (camera) {
      ComicPops.spawnWorld(this.position, camera, 'POP! 💨', '#84CC16');
    }

    // Comic Green Mushroom Smoke Puff (zero gore)
    if (particles && typeof particles.spawnWoodChips === 'function') {
      particles.spawnWoodChips(this.position, 18);
    }
  }

  public alertScreamerWave(screamerPos: THREE.Vector3) {
    if (this.state !== 'chase' && this.state !== 'attack') {
      this.state = 'chase';
      // Snap head turn towards screamer origin
      const dir = new THREE.Vector3().subVectors(screamerPos, this.position).setY(0).normalize();
      this.mesh.rotation.y = Math.atan2(dir.x, dir.z);
    }
  }

  private pickNewWanderTarget() {
    this.wanderTimer = 3.0 + Math.random() * 4.0;
    const angle = Math.random() * Math.PI * 2;
    const dist = 4.0 + Math.random() * 12.0;
    this.wanderTarget.set(
      this.position.x + Math.cos(angle) * dist,
      0,
      this.position.z + Math.sin(angle) * dist
    );
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.mesh);
  }
}
