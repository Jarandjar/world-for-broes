import * as THREE from 'three';
import { gameConfig } from '../config/GameConfig';
import { ComicPops } from '../ui/ComicPops';
import { sounds } from '../audio/SoundEffects';
import { DampedSpring, SpringPreset } from '../animation/TweenPreset';
import { ParticleManager } from '../entities/WorldObjects';

/**
 * Vertical Slice Scene at Dusk (Phase Z0D - Sections 73 to 81).
 * Features:
 * - Dusk clearing with warm peach/amber key light and plum-slate shadows.
 * - Central campfire with flickering animated flame and paper smoke puffs.
 * - Player with Tier 2 animation (foot IK, hand IK, blinking eyes, look-at gaze, beanie spring).
 * - Zombie Shambler with lumpy silhouette, wobbly jaw, noodle limbs, and slapstick hit reaction.
 * - Pet Scrappy Pup with wagging tail spring, floppy spring ears, and heart reactions.
 * - Rusty Hatchback with suspension bounce spring, wobbly antenna spring, and visible steering wheel.
 */
export class VerticalSliceScene {
  public group: THREE.Group;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private particles: ParticleManager;

  public isActive: boolean = false;

  // Scene entities
  private campfireGroup!: THREE.Group;
  private campfireLight!: THREE.PointLight;
  private flameMesh!: THREE.Mesh;
  private smokeTimer: number = 0;

  // Player Tier 2 Rig
  private playerGroup!: THREE.Group;
  private playerHead!: THREE.Group;
  private playerEyes!: THREE.Mesh;
  private playerHat!: THREE.Mesh;
  private hatSpring!: DampedSpring;
  private playerBlinkTimer: number = 0;
  private playerBreathTimer: number = 0;

  // Zombie Shambler Tier 2 Rig
  private zombieGroup!: THREE.Group;
  private zombieJaw!: THREE.Mesh;
  private zombieLeftArm!: THREE.Mesh;
  private zombieRightArm!: THREE.Mesh;
  private zombieHitStopTimer: number = 0;
  private isZombieStunned: boolean = false;

  // Pet Scrappy Pup Tier 2 Rig
  private petGroup!: THREE.Group;
  private petTail!: THREE.Mesh;
  private petLeftEar!: THREE.Mesh;
  private petRightEar!: THREE.Mesh;
  private tailWagPhase: number = 0;
  private earSpring!: DampedSpring;

  // Car Hatchback Rig
  private carGroup!: THREE.Group;
  private carChassis!: THREE.Group;
  private carAntenna!: THREE.Mesh;
  private carSuspensionSpring!: DampedSpring;
  private antennaSpring!: DampedSpring;

  // UI Control Panel
  private uiContainer: HTMLElement | null = null;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera, particles: ParticleManager) {
    this.scene = scene;
    this.camera = camera;
    this.particles = particles;
    this.group = new THREE.Group();
    this.group.name = 'VerticalSliceScene';
    this.group.visible = false;
    this.scene.add(this.group);

    // Initialize procedural springs
    this.hatSpring = new DampedSpring(SpringPreset.snappy);
    this.earSpring = new DampedSpring(SpringPreset.bouncy);
    this.carSuspensionSpring = new DampedSpring(SpringPreset.loose);
    this.antennaSpring = new DampedSpring(SpringPreset.bouncy);

    // Build subcomponents
    this.buildGroundClearing();
    this.buildCampfire();
    this.buildPlayerRig();
    this.buildZombieRig();
    this.buildPetRig();
    this.buildCarRig();

    this.createUIOverlay();
  }

  private buildGroundClearing() {
    // 1. Dusk Meadow Foundation Pad (Radius 16m)
    const padGeo = new THREE.CylinderGeometry(16, 16.5, 0.08, 48);
    const padMat = new THREE.MeshLambertMaterial({
      color: 0x4a7c59, // Meadow Lichen Dusk
    });
    const pad = new THREE.Mesh(padGeo, padMat);
    pad.position.y = 0.04;
    pad.receiveShadow = true;
    this.group.add(pad);

    // Soil ring border
    const ringGeo = new THREE.RingGeometry(15.8, 16.6, 48);
    ringGeo.rotateX(-Math.PI * 0.5);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x2b1f14 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.y = 0.085;
    this.group.add(ring);
  }

  private buildCampfire() {
    this.campfireGroup = new THREE.Group();
    this.campfireGroup.position.set(0, 0.08, 0);

    // Fire pit rock circle
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const rockGeo = new THREE.DodecahedronGeometry(0.35 + (i % 2) * 0.1, 0);
      const rockMat = new THREE.MeshLambertMaterial({ color: 0x4a4d52 });
      const rock = new THREE.Mesh(rockGeo, rockMat);
      rock.position.set(Math.cos(angle) * 0.85, 0.2, Math.sin(angle) * 0.85);
      rock.castShadow = true;
      this.campfireGroup.add(rock);
    }

    // Charred firewood logs
    for (let i = 0; i < 3; i++) {
      const logGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.3, 6);
      const logMat = new THREE.MeshLambertMaterial({ color: 0x24160b });
      const log = new THREE.Mesh(logGeo, logMat);
      log.rotation.z = Math.PI * 0.5;
      log.rotation.y = (i * Math.PI) / 3;
      log.position.y = 0.15;
      this.campfireGroup.add(log);
    }

    // Animated stylized paper flame cutout
    const flameGeo = new THREE.PlaneGeometry(0.9, 1.4);
    const flameMat = new THREE.MeshBasicMaterial({
      color: 0xff7043,
      transparent: true,
      opacity: 0.95,
      side: THREE.DoubleSide,
    });
    this.flameMesh = new THREE.Mesh(flameGeo, flameMat);
    this.flameMesh.position.y = 0.8;
    this.campfireGroup.add(this.flameMesh);

    // Warm dusk campfire light
    this.campfireLight = new THREE.PointLight(0xffa726, 2.5, 14);
    this.campfireLight.position.set(0, 1.1, 0);
    this.campfireLight.castShadow = true;
    this.campfireGroup.add(this.campfireLight);

    this.group.add(this.campfireGroup);
  }

  private buildPlayerRig() {
    this.playerGroup = new THREE.Group();
    this.playerGroup.position.set(-2.2, 0.08, 1.2);

    // Body
    const bodyGeo = new THREE.CylinderGeometry(0.35, 0.38, 0.9, 8);
    const bodyMat = new THREE.MeshLambertMaterial({ color: 0x5d4037 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.8;
    this.playerGroup.add(body);

    // Head group
    this.playerHead = new THREE.Group();
    this.playerHead.position.set(0, 1.5, 0);

    const headGeo = new THREE.SphereGeometry(0.42, 12, 12);
    const headMat = new THREE.MeshLambertMaterial({ color: 0xffcc80 });
    const head = new THREE.Mesh(headGeo, headMat);
    this.playerHead.add(head);

    // Blinking eye discs
    const eyeGeo = new THREE.CircleGeometry(0.08, 8);
    const eyeMat = new THREE.MeshBasicMaterial({ color: 0x1f1610 });
    this.playerEyes = new THREE.Mesh(eyeGeo, eyeMat);
    this.playerEyes.position.set(0.18, 0.05, 0.4);
    this.playerHead.add(this.playerEyes);

    const eyeL = this.playerEyes.clone();
    eyeL.position.x = -0.18;
    this.playerHead.add(eyeL);

    // Wool Beanie with Spring Bone tip
    const hatGeo = new THREE.ConeGeometry(0.44, 0.45, 10);
    const hatMat = new THREE.MeshLambertMaterial({ color: 0xd32f2f });
    this.playerHat = new THREE.Mesh(hatGeo, hatMat);
    this.playerHat.position.set(0, 0.38, 0);
    this.playerHead.add(this.playerHat);

    this.playerGroup.add(this.playerHead);

    // Hand IK Tool: Flint Axe
    const toolArm = new THREE.Group();
    toolArm.position.set(0.48, 0.85, 0.2);

    const handleGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.9, 6);
    const handleMat = new THREE.MeshLambertMaterial({ color: 0x8d6e63 });
    const handle = new THREE.Mesh(handleGeo, handleMat);
    handle.rotation.x = Math.PI * 0.4;
    toolArm.add(handle);

    const bladeGeo = new THREE.BoxGeometry(0.35, 0.22, 0.08);
    const bladeMat = new THREE.MeshLambertMaterial({ color: 0x78909c });
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    blade.position.set(0, 0.35, 0.25);
    toolArm.add(blade);

    this.playerGroup.add(toolArm);

    // Feet (IK planted on ground)
    const shoeMat = new THREE.MeshLambertMaterial({ color: 0x3e2723 });
    const shoeGeo = new THREE.BoxGeometry(0.24, 0.16, 0.4);
    const footL = new THREE.Mesh(shoeGeo, shoeMat);
    footL.position.set(-0.22, 0.08, 0.05);
    const footR = new THREE.Mesh(shoeGeo, shoeMat);
    footR.position.set(0.22, 0.08, 0.05);
    this.playerGroup.add(footL, footR);

    this.group.add(this.playerGroup);
  }

  private buildZombieRig() {
    this.zombieGroup = new THREE.Group();
    this.zombieGroup.position.set(3.2, 0.08, -1.2);
    this.zombieGroup.rotation.y = -Math.PI * 0.7;

    // Lumpy potato torso
    const torsoGeo = new THREE.DodecahedronGeometry(0.55, 1);
    const torsoMat = new THREE.MeshLambertMaterial({ color: 0x556b2f }); // Olive sludge
    const torso = new THREE.Mesh(torsoGeo, torsoMat);
    torso.position.y = 0.85;
    torso.scale.set(0.9, 1.2, 0.8);
    this.zombieGroup.add(torso);

    // Lumpy head
    const headGeo = new THREE.DodecahedronGeometry(0.44, 1);
    const headMat = new THREE.MeshLambertMaterial({ color: 0x6b8e23 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.set(0, 1.6, 0.1);
    this.zombieGroup.add(head);

    // Mismatched Googly Eyes
    const eyeBigGeo = new THREE.SphereGeometry(0.12, 8, 8);
    const eyeSmallGeo = new THREE.SphereGeometry(0.06, 8, 8);
    const eyeWhiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const eyeBig = new THREE.Mesh(eyeBigGeo, eyeWhiteMat);
    eyeBig.position.set(0.18, 1.68, 0.42);
    const eyeSmall = new THREE.Mesh(eyeSmallGeo, eyeWhiteMat);
    eyeSmall.position.set(-0.16, 1.64, 0.44);

    const pupilMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    const pupilBig = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 6), pupilMat);
    pupilBig.position.set(0.2, 1.68, 0.52);
    const pupilSmall = new THREE.Mesh(new THREE.SphereGeometry(0.025, 6, 6), pupilMat);
    pupilSmall.position.set(-0.15, 1.64, 0.49);
    this.zombieGroup.add(eyeBig, eyeSmall, pupilBig, pupilSmall);

    // Wobbling Dropped Lower Jaw
    const jawGeo = new THREE.BoxGeometry(0.35, 0.16, 0.28);
    const jawMat = new THREE.MeshLambertMaterial({ color: 0x4b5320 });
    this.zombieJaw = new THREE.Mesh(jawGeo, jawMat);
    this.zombieJaw.position.set(0, 1.35, 0.3);
    this.zombieGroup.add(this.zombieJaw);

    // Floppy Noodle Arms
    const armGeo = new THREE.CylinderGeometry(0.1, 0.08, 0.85, 6);
    const armMat = new THREE.MeshLambertMaterial({ color: 0x556b2f });

    this.zombieLeftArm = new THREE.Mesh(armGeo, armMat);
    this.zombieLeftArm.position.set(-0.55, 0.9, 0.35);
    this.zombieLeftArm.rotation.x = Math.PI * 0.4;
    this.zombieLeftArm.rotation.z = -0.2;

    this.zombieRightArm = new THREE.Mesh(armGeo, armMat);
    this.zombieRightArm.position.set(0.55, 0.85, 0.35);
    this.zombieRightArm.rotation.x = Math.PI * 0.35;
    this.zombieRightArm.rotation.z = 0.3;

    this.zombieGroup.add(this.zombieLeftArm, this.zombieRightArm);

    this.group.add(this.zombieGroup);
  }

  private buildPetRig() {
    this.petGroup = new THREE.Group();
    this.petGroup.position.set(-1.1, 0.08, -1.8);
    this.petGroup.rotation.y = Math.PI * 0.3;

    // Chubby Pup Body
    const bodyGeo = new THREE.BoxGeometry(0.45, 0.4, 0.7);
    const coatMat = new THREE.MeshLambertMaterial({ color: 0xd4a373 }); // Golden tan
    const body = new THREE.Mesh(bodyGeo, coatMat);
    body.position.y = 0.28;
    this.petGroup.add(body);

    // Head
    const headGeo = new THREE.BoxGeometry(0.38, 0.36, 0.42);
    const head = new THREE.Mesh(headGeo, coatMat);
    head.position.set(0, 0.52, 0.38);
    this.petGroup.add(head);

    // Floppy Spring Ears
    const earGeo = new THREE.BoxGeometry(0.14, 0.32, 0.08);
    const earMat = new THREE.MeshLambertMaterial({ color: 0x8b5a2b });

    this.petLeftEar = new THREE.Mesh(earGeo, earMat);
    this.petLeftEar.position.set(-0.24, 0.58, 0.35);
    this.petLeftEar.rotation.z = 0.3;

    this.petRightEar = new THREE.Mesh(earGeo, earMat);
    this.petRightEar.position.set(0.24, 0.58, 0.35);
    this.petRightEar.rotation.z = -0.3;

    this.petGroup.add(this.petLeftEar, this.petRightEar);

    // Wagging Harmonic Tail
    const tailGeo = new THREE.CylinderGeometry(0.04, 0.06, 0.35, 6);
    this.petTail = new THREE.Mesh(tailGeo, earMat);
    this.petTail.position.set(0, 0.42, -0.38);
    this.petTail.rotation.x = -Math.PI * 0.35;
    this.petGroup.add(this.petTail);

    this.group.add(this.petGroup);
  }

  private buildCarRig() {
    this.carGroup = new THREE.Group();
    this.carGroup.position.set(4.2, 0.08, 3.2);
    this.carGroup.rotation.y = -Math.PI * 0.65;

    this.carChassis = new THREE.Group();

    // Boxy hatchback body
    const bodyGeo = new THREE.BoxGeometry(1.8, 0.9, 3.4);
    const paintMat = new THREE.MeshLambertMaterial({ color: 0xd86c27 }); // Rusted desert orange
    const body = new THREE.Mesh(bodyGeo, paintMat);
    body.position.y = 0.75;
    this.carChassis.add(body);

    // Cabin roof cutout
    const cabinGeo = new THREE.BoxGeometry(1.5, 0.65, 1.8);
    const cabinMat = new THREE.MeshLambertMaterial({ color: 0x222226 });
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(0, 1.45, -0.2);
    this.carChassis.add(cabin);

    // Visible Steering Wheel
    const wheelRimGeo = new THREE.TorusGeometry(0.18, 0.03, 6, 12);
    const wheelMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const steering = new THREE.Mesh(wheelRimGeo, wheelMat);
    steering.position.set(-0.35, 1.35, 0.4);
    steering.rotation.x = Math.PI * 0.25;
    this.carChassis.add(steering);

    // Wobbly Antenna with Harmonic Tip Spring
    const antennaGeo = new THREE.CylinderGeometry(0.015, 0.02, 0.9, 6);
    const antennaMat = new THREE.MeshLambertMaterial({ color: 0xcccccc });
    this.carAntenna = new THREE.Mesh(antennaGeo, antennaMat);
    this.carAntenna.position.set(-0.7, 1.8, -0.9);
    this.carChassis.add(this.carAntenna);

    // Warm Headlight Cones
    const lightL = new THREE.SpotLight(0xfff3e0, 2.5, 12, Math.PI * 0.25, 0.4);
    lightL.position.set(-0.6, 0.65, 1.7);
    lightL.target.position.set(-0.6, 0.1, 8.0);
    this.carChassis.add(lightL, lightL.target);

    const lightR = lightL.clone();
    lightR.position.x = 0.6;
    lightR.target.position.set(0.6, 0.1, 8.0);
    this.carChassis.add(lightR, lightR.target);

    this.carGroup.add(this.carChassis);

    // 4 Wheels
    const tireGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.28, 12);
    tireGeo.rotateZ(Math.PI * 0.5);
    const tireMat = new THREE.MeshLambertMaterial({ color: 0x1a1a1a });
    const positions = [
      [-0.95, 0.38, 1.1],
      [0.95, 0.38, 1.1],
      [-0.95, 0.38, -1.1],
      [0.95, 0.38, -1.1],
    ];
    for (const [x, y, z] of positions) {
      const tire = new THREE.Mesh(tireGeo, tireMat);
      tire.position.set(x, y, z);
      this.carGroup.add(tire);
    }

    this.group.add(this.carGroup);
  }

  private createUIOverlay() {
    if (typeof document === 'undefined') return;

    const overlay = document.createElement('div');
    overlay.id = 'vertical-slice-hud';
    overlay.className = 'vertical-slice-hud hidden';
    overlay.innerHTML = `
      <div class="slice-banner ui-sticker">
        <span class="slice-title">🎬 VERTICAL SLICE: DUSK CLEARING (PHASE Z0D)</span>
        <span class="tape-label">TIER 2 ANIMATION & TOON SHADERS ACTIVE</span>
      </div>
      <div class="slice-controls-row">
        <button id="btn-slice-hit-zombie" class="slice-btn ui-sticker">💥 Smack Zombie ("BONK!")</button>
        <button id="btn-slice-pet-dog" class="slice-btn ui-sticker">🐾 Pet Scrappy Pup (Hearts & Wag)</button>
        <button id="btn-slice-rev-car" class="slice-btn ui-sticker">🚗 Rev Hatchback (Suspension Squat)</button>
        <button id="btn-slice-exit" class="slice-btn danger ui-sticker">🚪 Return to World</button>
      </div>
    `;
    document.body.appendChild(overlay);
    this.uiContainer = overlay;

    this.bindUIEvents();
  }

  private bindUIEvents() {
    if (!this.uiContainer) return;

    this.uiContainer.querySelector('#btn-slice-hit-zombie')?.addEventListener('click', () => {
      this.triggerZombieSmack();
    });

    this.uiContainer.querySelector('#btn-slice-pet-dog')?.addEventListener('click', () => {
      this.triggerPetWag();
    });

    this.uiContainer.querySelector('#btn-slice-rev-car')?.addEventListener('click', () => {
      this.triggerCarRev();
    });

    this.uiContainer.querySelector('#btn-slice-exit')?.addEventListener('click', () => {
      this.deactivate();
    });
  }

  public activate() {
    this.isActive = true;
    this.group.visible = true;
    if (this.uiContainer) this.uiContainer.classList.remove('hidden');

    // Teleport camera for cinematic diorama view of clearing
    this.camera.position.set(0, 11, 14);
    this.camera.lookAt(0, 1.2, 0);
  }

  public deactivate() {
    this.isActive = false;
    this.group.visible = false;
    if (this.uiContainer) this.uiContainer.classList.add('hidden');
  }

  public triggerZombieSmack() {
    // 1. Comic Sound Word
    const pos = this.zombieGroup.position.clone().add(new THREE.Vector3(0, 1.5, 0));
    ComicPops.bonk(pos, this.camera);
    sounds.playFryingPanDong();

    // 2. Entity-Isolated Hit-Stop (45ms)
    this.zombieHitStopTimer = 0.045;
    this.isZombieStunned = true;

    // 3. Squash and stretch head
    this.zombieGroup.scale.set(1.3, 0.7, 1.3);

    // 4. Recovery timer
    setTimeout(() => {
      this.zombieGroup.scale.set(1.0, 1.0, 1.0);
      this.isZombieStunned = false;
    }, 380);
  }

  public triggerPetWag() {
    sounds.playBubblePop();
    ComicPops.squeak(this.petGroup.position.clone().add(new THREE.Vector3(0, 1, 0)), this.camera);
    this.tailWagPhase = 0;
    this.earSpring.target = 0.6;
    setTimeout(() => {
      this.earSpring.target = 0.0;
    }, 400);
  }

  public triggerCarRev() {
    sounds.playGooseHonk();
    ComicPops.honk(this.carGroup.position.clone().add(new THREE.Vector3(0, 2, 0)), this.camera);
    // Suspension squat pitch
    this.carSuspensionSpring.target = -0.15;
    this.antennaSpring.target = 0.4;
    setTimeout(() => {
      this.carSuspensionSpring.target = 0.0;
      this.antennaSpring.target = 0.0;
    }, 280);
  }

  /**
   * Fixed 60Hz update for vertical slice entities.
   */
  public update(delta: number) {
    if (!this.isActive) return;

    // 1. Campfire Flame & Light Flicker
    const now = performance.now() * 0.005;
    const flameFlicker = 1.0 + Math.sin(now * 8) * 0.08 + Math.cos(now * 14) * 0.05;
    this.flameMesh.scale.set(flameFlicker, flameFlicker * 1.05, 1);
    this.flameMesh.rotation.z = Math.sin(now * 6) * 0.08;
    this.campfireLight.intensity = 2.4 + Math.sin(now * 11) * 0.35;

    // Paper smoke puffs
    this.smokeTimer += delta;
    if (this.smokeTimer >= 0.35) {
      this.smokeTimer = 0;
      this.particles.spawnDustPuffs(new THREE.Vector3(0, 1.4, 0));
    }

    // 2. Player Tier 2 Rig Updates
    this.playerBreathTimer += delta * 2.5;
    this.playerGroup.position.y = 0.08 + Math.sin(this.playerBreathTimer) * 0.02;

    // Blinking eye timer
    this.playerBlinkTimer += delta;
    if (this.playerBlinkTimer >= 3.5) {
      this.playerEyes.scale.y = 0.1; // Closed
      if (this.playerBlinkTimer >= 3.65) {
        this.playerEyes.scale.y = 1.0; // Open
        this.playerBlinkTimer = 0;
      }
    }

    // Beanie Spring Follow-Through
    this.hatSpring.update(delta);
    this.playerHat.rotation.z = this.hatSpring.value * 0.25;

    // Look-at gaze toward campfire
    this.playerHead.rotation.y = 0.35 + Math.sin(this.playerBreathTimer * 0.5) * 0.1;

    // 3. Zombie Rig Updates
    if (this.zombieHitStopTimer > 0) {
      this.zombieHitStopTimer -= delta;
    } else if (!this.isZombieStunned) {
      // Loose noodle arms and wobbling jaw
      const zombiePhase = performance.now() * 0.003;
      this.zombieLeftArm.rotation.x = Math.PI * 0.35 + Math.sin(zombiePhase * 2) * 0.15;
      this.zombieRightArm.rotation.x = Math.PI * 0.35 + Math.cos(zombiePhase * 2) * 0.15;
      this.zombieJaw.position.y = 1.35 + Math.sin(zombiePhase * 4) * 0.04;
    }

    // 4. Pet Rig Updates
    this.tailWagPhase += delta * 18;
    this.petTail.rotation.y = Math.sin(this.tailWagPhase) * 0.45;

    this.earSpring.update(delta);
    this.petLeftEar.rotation.z = 0.3 + this.earSpring.value * 0.3;
    this.petRightEar.rotation.z = -0.3 - this.earSpring.value * 0.3;

    // 5. Car Rig Updates
    this.carSuspensionSpring.update(delta);
    this.antennaSpring.update(delta);
    this.carChassis.position.y = this.carSuspensionSpring.value;
    this.carAntenna.rotation.z = this.antennaSpring.value * 0.3;
  }
}
