import * as THREE from 'three';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { sounds } from '../audio/SoundEffects';
import { InventorySystem } from '../systems/InventorySystem';
import { damp } from '../utils/MathUtils';
import { AnimationController } from '../animation/AnimationController';

export type PerspectiveView = 'FRONT' | 'SIDE' | 'BACK';

export class Player {
  public group: THREE.Group;
  public anim: AnimationController;
  public position: THREE.Vector3;
  public prevPosition: THREE.Vector3 = new THREE.Vector3();
  public renderPosition: THREE.Vector3 = new THREE.Vector3();
  public velocity: THREE.Vector3 = new THREE.Vector3();
  public targetVelocity: THREE.Vector3 = new THREE.Vector3();
  public moveSpeed: number = 6.5;
  public isSprinting: boolean = false;

  // Smooth facing direction squash flip (~100-120ms turn rate interpolation)
  public facingLeft: boolean = false;
  private targetFacingSign: number = 1;
  public facingScaleX: number = 1;

  // 2.5D Multi-Directional Perspective View & Hysteresis
  public currentView: PerspectiveView = 'FRONT';
  private viewHysteresisBuffer: number = 0.20; // 0.20 rad hysteresis prevents rapid angle flickering

  // Puppet Components
  private torso: THREE.Mesh;
  private head: THREE.Mesh;
  private backpack: THREE.Mesh;
  private leftArm: THREE.Mesh;
  private rightArm: THREE.Mesh;
  private leftLeg: THREE.Mesh;
  private rightLeg: THREE.Mesh;
  private heldTool: THREE.Mesh;
  private torchLight: THREE.PointLight;

  // Perspective Materials
  private headMatFront: THREE.MeshLambertMaterial;
  private headMatSide: THREE.MeshLambertMaterial;
  private headMatBack: THREE.MeshLambertMaterial;
  private torsoMatFront: THREE.MeshLambertMaterial;
  private torsoMatSide: THREE.MeshLambertMaterial;
  private torsoMatBack: THREE.MeshLambertMaterial;
  private backpackMat: THREE.MeshLambertMaterial;

  // Animation States
  public isMoving: boolean = false;
  public isChopping: boolean = false;
  public isSitting: boolean = false;
  public isEating: boolean = false;
  public isHurt: boolean = false;
  public isHighFiving: boolean = false;
  public isPointing: boolean = false;

  private walkPhase: number = 0;
  private chopTimer: number = 0;
  private chopDuration: number = 0.45;
  private eatTimer: number = 0;
  private hurtTimer: number = 0;
  private highFiveTimer: number = 0;
  private pointTimer: number = 0;

  // Tool textures
  private axeMat: THREE.MeshLambertMaterial;
  private torchMat: THREE.MeshBasicMaterial;
  private panMat: THREE.MeshLambertMaterial;
  private chickenMat: THREE.MeshLambertMaterial;
  private marshmallowMat: THREE.MeshBasicMaterial;

  // Speech bubble
  private speechBubble: THREE.Mesh;
  private speechTimer: number = 0;

  // Dance & Emote States
  public isDancing: boolean = false;
  private dancePhase: number = 0;
  public isHonking: boolean = false;
  private honkTimer: number = 0;

  // Chop Anticipation & Impact Sync
  private onChopImpactCallback?: () => void;
  private chopImpactFired: boolean = false;

  // Head Counter-bob Spring Lag
  private headSpringAngle: number = 0;
  private headSpringVel: number = 0;

  // Personality Idle Tracking
  public idleTimer: number = 0;

  // Footstep Ground Contact Callback
  private onFootstepCallback?: (pos: THREE.Vector3) => void;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.position = this.group.position;
    this.position.set(0, 0, 0);

    // Multi-Directional Materials
    this.headMatFront = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerHead(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.headMatSide = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerHeadSide(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.headMatBack = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerHeadBack(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });

    this.torsoMatFront = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerTorso(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.torsoMatSide = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerTorsoSide(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.torsoMatBack = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerTorsoBack(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });

    this.backpackMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerBackpack(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });

    // 1. Torso
    const torsoGeo = new THREE.PlaneGeometry(1.0, 1.0);
    torsoGeo.translate(0, 0.5, 0);
    this.torso = new THREE.Mesh(torsoGeo, this.torsoMatFront);
    this.torso.position.y = 0.65;
    this.group.add(this.torso);

    // Equipment Layer: Sturdy Traveler Backpack attached to Torso
    const backpackGeo = new THREE.PlaneGeometry(0.85, 0.85);
    backpackGeo.translate(0, 0.45, 0);
    this.backpack = new THREE.Mesh(backpackGeo, this.backpackMat);
    this.backpack.position.set(0, 0, -0.04);
    this.backpack.visible = false;
    this.torso.add(this.backpack);

    // 2. Head (Attached to top of Torso)
    const headGeo = new THREE.PlaneGeometry(1.2, 1.2);
    headGeo.translate(0, 0.6, 0);
    this.head = new THREE.Mesh(headGeo, this.headMatFront);
    this.head.position.set(0, 0.9, 0.02);
    this.torso.add(this.head);

    // 3. Legs
    const legGeo = new THREE.PlaneGeometry(0.5, 0.9);
    legGeo.translate(0, -0.45, 0);
    const legMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerLeg(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });

    this.leftLeg = new THREE.Mesh(legGeo, legMat);
    this.leftLeg.position.set(-0.2, 0.65, -0.01);
    this.group.add(this.leftLeg);

    this.rightLeg = new THREE.Mesh(legGeo, legMat);
    this.rightLeg.position.set(0.2, 0.65, 0.01);
    this.group.add(this.rightLeg);

    // 4. Arms
    const armGeo = new THREE.PlaneGeometry(0.5, 0.9);
    armGeo.translate(0, -0.45, 0);
    const armMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getPlayerArm(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });

    this.leftArm = new THREE.Mesh(armGeo, armMat);
    this.leftArm.position.set(-0.45, 0.85, -0.02);
    this.torso.add(this.leftArm);

    this.rightArm = new THREE.Mesh(armGeo, armMat);
    this.rightArm.position.set(0.45, 0.85, 0.05);
    this.torso.add(this.rightArm);

    // 5. Tool Slot
    this.axeMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getAxeTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.torchMat = new THREE.MeshBasicMaterial({
      map: TextureGenerator.getTorchTexture(),
      transparent: true,
      side: THREE.DoubleSide,
    });
    this.panMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getFryingPanTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.chickenMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getRubberChickenTexture(),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    this.marshmallowMat = new THREE.MeshBasicMaterial({
      map: TextureGenerator.getMarshmallowStickTexture(),
      transparent: true,
      side: THREE.DoubleSide,
    });

    const toolGeo = new THREE.PlaneGeometry(0.7, 1.2);
    toolGeo.translate(0.1, -0.2, 0);
    this.heldTool = new THREE.Mesh(toolGeo, this.axeMat);
    this.heldTool.position.set(0.0, -0.6, 0.05);
    this.heldTool.visible = false;
    this.rightArm.add(this.heldTool);

    // 6. Point Light
    this.torchLight = new THREE.PointLight(0xffaa33, 0, 10, 1.2);
    this.torchLight.position.set(0, 1.5, 0);
    this.group.add(this.torchLight);

    // 7. Speech Bubble
    const bubbleGeo = new THREE.PlaneGeometry(2.4, 0.9);
    bubbleGeo.translate(0, 0.45, 0);
    const bubbleMat = new THREE.MeshBasicMaterial({
      map: TextureGenerator.createSpeechBubbleTexture('Hello!'),
      transparent: true,
      depthTest: false,
    });
    this.speechBubble = new THREE.Mesh(bubbleGeo, bubbleMat);
    this.speechBubble.position.set(0, 2.3, 0);
    this.speechBubble.visible = false;
    this.group.add(this.speechBubble);

    // 8. Ground shadow
    const shadowGeo = new THREE.CircleGeometry(0.6, 20);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4 });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = 0.02;
    this.group.add(shadow);

    scene.add(this.group);

    // 9. Shared Humanoid Rig Animation Controller (Phase Z0C)
    this.anim = new AnimationController();
    this.anim.registerNode('torso', this.torso);
    this.anim.registerNode('head', this.head);
    this.anim.registerNode('backpack', this.backpack);
    this.anim.registerNode('leftArm', this.leftArm);
    this.anim.registerNode('rightArm', this.rightArm);
    this.anim.registerNode('leftLeg', this.leftLeg);
    this.anim.registerNode('rightLeg', this.rightLeg);
    this.anim.registerNode('heldTool', this.heldTool);

    // Event binding: Damage on chop is applied on hit_frame event
    this.anim.stateMachine.eventManager.addListener('hit_frame', () => {
      if (this.onChopImpactCallback && !this.chopImpactFired) {
        this.chopImpactFired = true;
        this.onChopImpactCallback();
      }
    });

    this.anim.stateMachine.eventManager.addListener('footstep', () => {
      this.onFootstepCallback?.(this.position);
    });
  }

  public setOnFootstep(cb: (pos: THREE.Vector3) => void) {
    this.onFootstepCallback = cb;
  }

  public showSpeechBubble(text: string) {
    this.speechBubble.material = new THREE.MeshBasicMaterial({
      map: TextureGenerator.createSpeechBubbleTexture(text),
      transparent: true,
      depthTest: false,
    });
    this.speechBubble.visible = true;
    this.speechTimer = 4.0;
    sounds.playBubblePop();
  }

  public triggerDance() {
    this.isDancing = !this.isDancing;
    if (this.isDancing) {
      this.isSitting = false;
      this.showSpeechBubble('🕺 PARTY TIME!');
    }
    this.idleTimer = 0;
  }

  public triggerHonk() {
    this.isHonking = true;
    this.honkTimer = 0;
    this.idleTimer = 0;
  }

  public triggerHighFive() {
    this.isHighFiving = true;
    this.highFiveTimer = 0;
    this.isSitting = false;
    this.showSpeechBubble('✋ HIGH FIVE, BRO!');
    this.idleTimer = 0;
  }

  public triggerPoint() {
    this.isPointing = true;
    this.pointTimer = 0;
    this.isSitting = false;
    this.showSpeechBubble('👉 OVER HERE!');
    this.idleTimer = 0;
  }

  public triggerSit() {
    this.isSitting = !this.isSitting;
    if (this.isSitting) {
      this.isDancing = false;
      this.showSpeechBubble('🏕️ Taking a breather...');
    }
    this.idleTimer = 0;
  }

  public triggerEat() {
    this.isEating = true;
    this.eatTimer = 0;
    this.idleTimer = 0;
  }

  public triggerHurt() {
    this.isHurt = true;
    this.hurtTimer = 0.35;
    sounds.playHurt();
    this.idleTimer = 0;
  }

  public triggerChop(onImpact?: () => void) {
    if (this.isChopping) return;
    this.isChopping = true;
    this.isSitting = false;
    this.chopTimer = 0;
    this.chopImpactFired = false;
    this.onChopImpactCallback = onImpact;
    this.idleTimer = 0;
  }

  public hasLitTorch(inventory: InventorySystem): boolean {
    const item = inventory.getSelectedItemId();
    return item === 'torch' || item === 'marshmallow';
  }

  /**
   * 2.5D Perspective Calculator: Evaluates relative angle between movement/facing and camera yaw.
   * Employs hysteresis buffer to avoid sprite flickering on boundaries.
   */
  private updatePerspectiveSprites(cameraYaw: number) {
    const speedSq = this.velocity.x * this.velocity.x + this.velocity.z * this.velocity.z;

    if (speedSq > 0.15) {
      const moveAngle = Math.atan2(this.velocity.x, this.velocity.z);
      let deltaAngle = moveAngle - cameraYaw;
      while (deltaAngle > Math.PI) deltaAngle -= Math.PI * 2;
      while (deltaAngle < -Math.PI) deltaAngle += Math.PI * 2;

      const absDelta = Math.abs(deltaAngle);
      const buf = this.viewHysteresisBuffer;

      let nextView: PerspectiveView = this.currentView;

      if (this.currentView === 'FRONT') {
        if (absDelta > 0.85 + buf) {
          nextView = absDelta > 2.25 ? 'BACK' : 'SIDE';
        }
      } else if (this.currentView === 'BACK') {
        if (absDelta < 2.25 - buf) {
          nextView = absDelta < 0.85 ? 'FRONT' : 'SIDE';
        }
      } else {
        // Current view is SIDE
        if (absDelta < 0.85 - buf) {
          nextView = 'FRONT';
        } else if (absDelta > 2.25 + buf) {
          nextView = 'BACK';
        }
      }

      this.currentView = nextView;

      // In SIDE view: set mirror flip based on deltaAngle sign
      if (nextView === 'SIDE') {
        this.setHorizontalFlip(deltaAngle < 0);
      }
    }

    // Apply materials based on view
    if (this.currentView === 'FRONT') {
      this.head.material = this.headMatFront;
      this.torso.material = this.torsoMatFront;
      this.backpack.visible = false;
    } else if (this.currentView === 'BACK') {
      this.head.material = this.headMatBack;
      this.torso.material = this.torsoMatBack;
      this.backpack.visible = true;
      this.backpack.position.set(0, 0, 0.05);
    } else {
      // SIDE
      this.head.material = this.headMatSide;
      this.torso.material = this.torsoMatSide;
      this.backpack.visible = true;
      this.backpack.position.set(-0.24, 0.45, -0.03);
    }
  }

  public update(
    delta: number,
    camera: THREE.Camera,
    inventory: InventorySystem,
    context?: { nearCampfire?: boolean; isColdOrLowSanity?: boolean }
  ) {
    // 1. Tool Slot Visibility
    const selectedItem = inventory.getSelectedItemId();
    if (selectedItem === 'axe') {
      this.heldTool.visible = true;
      this.heldTool.material = this.axeMat;
      this.torchLight.intensity = 0;
    } else if (selectedItem === 'torch') {
      this.heldTool.visible = true;
      this.heldTool.material = this.torchMat;
      this.torchLight.intensity = 1.8 + Math.sin(Date.now() * 0.02) * 0.2;
    } else if (selectedItem === 'pan') {
      this.heldTool.visible = true;
      this.heldTool.material = this.panMat;
      this.torchLight.intensity = 0;
    } else if (selectedItem === 'chicken') {
      this.heldTool.visible = true;
      this.heldTool.material = this.chickenMat;
      this.torchLight.intensity = 0;
    } else if (selectedItem === 'marshmallow') {
      this.heldTool.visible = true;
      this.heldTool.material = this.marshmallowMat;
      this.torchLight.intensity = 1.2 + Math.sin(Date.now() * 0.03) * 0.15;
    } else {
      this.heldTool.visible = false;
      this.torchLight.intensity = 0;
    }

    // Turn Rate Interpolation: Smooth facing direction squash flip (~120ms)
    this.facingScaleX = damp(this.facingScaleX, this.targetFacingSign, 18, delta);

    // Billboarding: Align character to camera yaw
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.group.rotation.y = camEuler.y;

    // Multi-Directional Perspective Swapping
    this.updatePerspectiveSprites(camEuler.y);

    // Speech bubble timer
    if (this.speechBubble.visible) {
      this.speechTimer -= delta;
      if (this.speechTimer <= 0) {
        this.speechBubble.visible = false;
      }
    }

    // Hurt feedback (Red tint flash & recovery)
    if (this.isHurt) {
      this.hurtTimer -= delta;
      const tint = Math.max(0, this.hurtTimer / 0.35);
      this.torsoMatFront.color.setRGB(1.0, 1.0 - tint * 0.6, 1.0 - tint * 0.6);
      this.torso.rotation.x = -0.35 * tint;
      if (this.hurtTimer <= 0) {
        this.isHurt = false;
        this.torsoMatFront.color.setRGB(1, 1, 1);
        this.torso.rotation.x = 0;
      }
    }

    // Eat animation
    if (this.isEating) {
      this.eatTimer += delta;
      const t = this.eatTimer / 0.5;
      const bite = Math.sin(t * Math.PI * 4) * 0.15;
      this.rightArm.rotation.x = -1.4 + bite;
      this.rightArm.rotation.z = -0.4;
      this.head.rotation.x = 0.2 + bite * 0.5;
      if (this.eatTimer >= 0.5) {
        this.isEating = false;
        this.rightArm.rotation.set(0, 0, 0);
      }
      return;
    }

    // High-Five Emote
    if (this.isHighFiving) {
      this.highFiveTimer += delta;
      this.rightArm.rotation.x = -2.5;
      this.rightArm.rotation.z = -0.35;
      this.torso.rotation.x = 0.15;
      this.torso.position.y = 0.65 + Math.sin(this.highFiveTimer * 12) * 0.08;
      if (this.highFiveTimer >= 1.4) {
        this.isHighFiving = false;
      }
      return;
    }

    // Point Emote
    if (this.isPointing) {
      this.pointTimer += delta;
      this.rightArm.rotation.x = -1.5;
      this.rightArm.rotation.z = 0;
      this.torso.rotation.x = 0.1;
      if (this.pointTimer >= 1.4) {
        this.isPointing = false;
      }
      return;
    }

    // 2. Procedural Walk & Run Cycle with Foot-Slip Prevention
    const actualSpeed = Math.sqrt(this.velocity.x * this.velocity.x + this.velocity.z * this.velocity.z);

    if (this.isMoving && !this.isChopping) {
      this.idleTimer = 0;
      this.isSitting = false;

      // Foot-Slip Prevention: Stride frequency tied directly to actual planar ground velocity
      const speedRatio = Math.max(0.2, Math.min(1.6, actualSpeed / 6.5));
      const prevWalkPhase = this.walkPhase;
      this.walkPhase += delta * 15 * speedRatio;

      const strideAmp = this.isSprinting ? 0.95 : 0.72;
      const legAngle = Math.sin(this.walkPhase) * strideAmp;
      this.leftLeg.rotation.x = legAngle;
      this.rightLeg.rotation.x = -legAngle;

      // Forward lean (more aggressive forward tilt in sprint)
      const targetLean = this.isSprinting ? 0.34 : 0.22;
      this.torso.rotation.x = THREE.MathUtils.lerp(this.torso.rotation.x, targetLean, 0.2);
      this.torso.rotation.z = Math.sin(this.walkPhase) * 0.07;

      // Arms swing
      this.leftArm.rotation.x = -legAngle * 0.85;
      if (!this.heldTool.visible) {
        this.rightArm.rotation.x = legAngle * 0.85;
      } else {
        this.rightArm.rotation.x = -0.35 + legAngle * 0.45;
      }

      // Torso vertical bounce & stride squash
      const bob = Math.abs(Math.sin(this.walkPhase)) * (this.isSprinting ? 0.15 : 0.11);
      this.torso.position.y = 0.65 + bob;

      const strideSquash = Math.sin(this.walkPhase * 2) * 0.06;
      this.torso.scale.y = 1.0 - strideSquash;
      this.torso.scale.x = this.facingScaleX * (1.0 + strideSquash * 0.6);

      // Head spring lag
      const targetHeadAngle = -0.12 - legAngle * 0.12;
      this.headSpringVel += (targetHeadAngle - this.headSpringAngle) * 55 * delta;
      this.headSpringVel *= Math.exp(-14 * delta);
      this.headSpringAngle += this.headSpringVel * delta;
      this.head.rotation.x = this.headSpringAngle;
      this.head.rotation.z = Math.sin(this.walkPhase * 0.5) * 0.05;

      // Footstep audio contact
      const prevSin = Math.sin(prevWalkPhase);
      const currSin = Math.sin(this.walkPhase);
      if ((prevSin < 0 && currSin >= 0) || (prevSin > 0 && currSin <= 0)) {
        sounds.playFootstep();
        this.onFootstepCallback?.(this.position);
      }
    } else if (this.isDancing) {
      this.idleTimer = 0;
      this.dancePhase += delta * 14;
      this.torso.rotation.z = Math.sin(this.dancePhase) * 0.32;
      this.torso.rotation.x = Math.sin(this.dancePhase * 0.5) * 0.15;
      this.leftArm.rotation.z = Math.sin(this.dancePhase) * 1.1 + 0.9;
      this.rightArm.rotation.z = -Math.sin(this.dancePhase) * 1.1 - 0.9;
      this.head.rotation.z = -Math.sin(this.dancePhase) * 0.25;
      this.torso.position.y = 0.65 + Math.abs(Math.sin(this.dancePhase * 2)) * 0.18;
      this.torso.scale.set(this.facingScaleX, 1.0 + Math.sin(this.dancePhase * 3) * 0.08, 1.0);
    } else if (this.isHonking) {
      this.idleTimer = 0;
      this.honkTimer += delta;
      const progress = this.honkTimer / 0.4;
      if (progress < 0.3) {
        const t = progress / 0.3;
        this.torso.scale.set(this.facingScaleX * 1.15, THREE.MathUtils.lerp(1.0, 0.76, t), 1.0);
        this.head.position.y = THREE.MathUtils.lerp(0.9, 0.72, t);
      } else if (progress < 0.7) {
        const t = (progress - 0.3) / 0.4;
        this.torso.scale.set(this.facingScaleX * 0.88, THREE.MathUtils.lerp(0.76, 1.28, t), 1.0);
        this.head.position.y = THREE.MathUtils.lerp(0.72, 1.05, t);
      } else {
        const t = (progress - 0.7) / 0.3;
        this.torso.scale.set(this.facingScaleX, THREE.MathUtils.lerp(1.28, 1.0, t), 1.0);
        this.head.position.y = THREE.MathUtils.lerp(1.05, 0.9, t);
      }
      if (this.honkTimer >= 0.4) {
        this.isHonking = false;
        this.honkTimer = 0;
      }
    } else if (!this.isChopping) {
      // 3. CONTEXTUAL IDLES
      this.idleTimer += delta;
      const idleTime = Date.now() * 0.0028;

      if (context?.nearCampfire) {
        // CONTEXTUAL IDLE 1: Warming Hands at Campfire
        const warmRub = Math.sin(idleTime * 4) * 0.08;
        this.torso.position.y = 0.65;
        this.torso.rotation.x = 0.12;
        this.leftArm.rotation.x = -1.15 + warmRub;
        this.rightArm.rotation.x = -1.15 - warmRub;
        this.leftArm.rotation.z = -0.2;
        this.rightArm.rotation.z = 0.2;
        this.head.rotation.x = 0.15;
      } else if (context?.isColdOrLowSanity) {
        // CONTEXTUAL IDLE 2: Shivering / Cold / Nervous Teeth Chattering
        const shiver = Math.sin(Date.now() * 0.045) * 0.07;
        this.torso.position.y = 0.65;
        this.torso.rotation.z = shiver;
        this.torso.rotation.x = -0.1;
        this.leftArm.rotation.z = 0.45;
        this.rightArm.rotation.z = -0.45;
        this.leftArm.rotation.x = -0.3;
        this.rightArm.rotation.x = -0.3;
        this.head.rotation.z = shiver * 0.8;
        this.head.rotation.x = Math.sin(idleTime * 2) * 0.1;
      } else if (this.isSitting || this.idleTimer > 15.0) {
        // CONTEXTUAL IDLE 3: Sitting down on grass
        this.torso.position.y = 0.36;
        this.leftLeg.rotation.x = -1.35;
        this.rightLeg.rotation.x = -1.35;
        this.leftArm.rotation.x = -0.3;
        this.rightArm.rotation.x = -0.3;
        this.torso.rotation.x = -0.05;
        this.head.rotation.x = Math.sin(idleTime * 0.4) * 0.05;
      } else {
        // Standard Organic Idle Breathing
        const breathe = Math.sin(idleTime) * 0.025;
        const weightShift = Math.sin(idleTime * 0.5) * 0.02;

        this.torso.position.y = 0.65 + breathe;
        this.torso.rotation.x = THREE.MathUtils.lerp(this.torso.rotation.x, 0, 0.15);
        this.torso.rotation.z = THREE.MathUtils.lerp(this.torso.rotation.z, weightShift, 0.15);
        this.torso.scale.set(this.facingScaleX * (1.0 - breathe * 0.3), 1.0 + breathe, 1.0);

        this.leftArm.rotation.z = Math.sin(idleTime) * 0.05;
        this.rightArm.rotation.z = -Math.sin(idleTime) * 0.05;

        this.leftLeg.rotation.x = THREE.MathUtils.lerp(this.leftLeg.rotation.x, 0, 0.2);
        this.rightLeg.rotation.x = THREE.MathUtils.lerp(this.rightLeg.rotation.x, 0, 0.2);
        this.leftArm.rotation.x = THREE.MathUtils.lerp(this.leftArm.rotation.x, 0, 0.2);
        this.rightArm.rotation.x = THREE.MathUtils.lerp(this.rightArm.rotation.x, 0, 0.2);
      }
    }

    // 4. Chop / Mine Attack Animation with Anticipation, Hit-Stop apex & Overshoot Recoil
    if (this.isChopping) {
      this.idleTimer = 0;
      this.chopTimer += delta;
      const progress = this.chopTimer / this.chopDuration;

      if (progress < 0.28) {
        const t = progress / 0.28;
        this.rightArm.rotation.x = THREE.MathUtils.lerp(0, -1.8, t);
        this.torso.rotation.x = THREE.MathUtils.lerp(0, -0.28, t);
        this.head.rotation.x = THREE.MathUtils.lerp(0, -0.22, t);
        this.torso.scale.y = THREE.MathUtils.lerp(1.0, 1.08, t);
      } else if (progress < 0.52) {
        const t = (progress - 0.28) / 0.24;
        this.rightArm.rotation.x = THREE.MathUtils.lerp(-1.8, 1.35, t);
        this.torso.rotation.x = THREE.MathUtils.lerp(-0.28, 0.35, t);
        this.head.rotation.x = THREE.MathUtils.lerp(-0.22, 0.24, t);
        this.torso.scale.y = THREE.MathUtils.lerp(1.08, 0.88, t);

        if (progress >= 0.38 && !this.chopImpactFired) {
          this.chopImpactFired = true;
          this.onChopImpactCallback?.();
        }
      } else {
        const t = (progress - 0.52) / 0.48;
        const settle = Math.sin(t * 20) * 0.08 * (1.0 - t);
        this.rightArm.rotation.x = THREE.MathUtils.lerp(1.35, 0, t) + settle;
        this.torso.rotation.x = THREE.MathUtils.lerp(0.35, 0, t);
        this.head.rotation.x = THREE.MathUtils.lerp(0.24, 0, t);
        this.torso.scale.y = THREE.MathUtils.lerp(0.88, 1.0, t);
      }

      if (this.chopTimer >= this.chopDuration) {
        this.isChopping = false;
        this.chopTimer = 0;
        this.chopImpactFired = false;
      }
    }

    // Unified Animation State Machine Evaluation (Phase Z0C)
    this.anim.updateAnimationState(delta, {
      isMoving: this.isMoving,
      isSprinting: this.isSprinting,
      isSneaking: this.moveSpeed < 4.5 && this.isMoving,
      velocity: this.velocity,
      isChopping: this.isChopping,
      isEating: this.isEating,
      isDancing: this.isDancing,
      isHonking: this.isHonking,
      isHurt: this.isHurt,
      isDown: false,
      facingSign: this.facingScaleX,
      worldPosition: this.position,
    });
  }

  public savePreviousState() {
    this.prevPosition.copy(this.position);
  }

  public interpolateRender(alpha: number) {
    this.renderPosition.lerpVectors(this.prevPosition, this.position, alpha);
    this.group.position.copy(this.renderPosition);
    this.anim.applyInterpolatedPose(alpha);
  }

  public setHorizontalFlip(isMovingLeft: boolean) {
    this.facingLeft = isMovingLeft;
    this.targetFacingSign = isMovingLeft ? -1 : 1;
  }
}
