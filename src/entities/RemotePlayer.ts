import * as THREE from 'three';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { damp, dampVector3 } from '../utils/MathUtils';

export class RemotePlayer {
  public id: string;
  public name: string;
  public group: THREE.Group;

  // Transform target for smooth exponential damping
  public targetPos: THREE.Vector3 = new THREE.Vector3();
  public prevPos: THREE.Vector3 = new THREE.Vector3();
  public isMoving: boolean = false;
  public isChopping: boolean = false;
  public isDancing: boolean = false;
  public facingLeft: boolean = false;
  public facingScaleX: number = 1; // Smooth facing flip squash
  public equippedTool: string | null = null;
  public isDead: boolean = false;
  private dancePhase: number = 0;
  private chopTimer: number = 0;

  // 2.5D Multi-Directional Perspective View & Hysteresis
  public currentView: 'FRONT' | 'SIDE' | 'BACK' = 'FRONT';
  private viewHysteresisBuffer: number = 0.20;

  // Perspective Materials
  private headMatFront: THREE.MeshLambertMaterial;
  private headMatSide: THREE.MeshLambertMaterial;
  private headMatBack: THREE.MeshLambertMaterial;
  private torsoMatFront: THREE.MeshLambertMaterial;
  private torsoMatSide: THREE.MeshLambertMaterial;
  private torsoMatBack: THREE.MeshLambertMaterial;
  private backpackMat: THREE.MeshLambertMaterial;

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
  private nameplateSprite: THREE.Sprite;
  private speechBubble: THREE.Mesh;
  private speechTimer: number = 0;

  private walkPhase: number = 0;
  private axeMat: THREE.MeshLambertMaterial;
  private torchMat: THREE.MeshBasicMaterial;
  private panMat: THREE.MeshLambertMaterial;
  private chickenMat: THREE.MeshLambertMaterial;
  private marshmallowMat: THREE.MeshBasicMaterial;

  constructor(scene: THREE.Scene, id: string, name: string, initialPos: { x: number; z: number }) {
    this.id = id;
    this.name = name;
    this.group = new THREE.Group();
    this.group.position.set(initialPos.x, 0, initialPos.z);
    this.targetPos.copy(this.group.position);
    this.prevPos.copy(this.group.position);

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

    // Torso
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

    // Head
    const headGeo = new THREE.PlaneGeometry(1.2, 1.2);
    headGeo.translate(0, 0.6, 0);
    this.head = new THREE.Mesh(headGeo, this.headMatFront);
    this.head.position.set(0, 0.9, 0.02);
    this.torso.add(this.head);

    // Legs
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

    // Arms
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

    // Tool
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

    // Shared Torch Light
    this.torchLight = new THREE.PointLight(0xff9922, 0, 10, 1.2);
    this.torchLight.position.set(0, 1.5, 0);
    this.group.add(this.torchLight);

    // Comic Speech Bubble
    const bubbleGeo = new THREE.PlaneGeometry(2.4, 0.9);
    bubbleGeo.translate(0, 0.45, 0);
    const bubbleMat = new THREE.MeshBasicMaterial({
      map: TextureGenerator.createSpeechBubbleTexture(''),
      transparent: true,
      depthTest: false,
    });
    this.speechBubble = new THREE.Mesh(bubbleGeo, bubbleMat);
    this.speechBubble.position.set(0, 2.7, 0);
    this.speechBubble.visible = false;
    this.group.add(this.speechBubble);

    // Floating Nameplate
    this.nameplateSprite = this.createNameplate(name);
    this.nameplateSprite.position.set(0, 2.2, 0);
    this.group.add(this.nameplateSprite);

    // Ground Shadow
    const shadowGeo = new THREE.CircleGeometry(0.6, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.position.y = 0.02;
    this.group.add(shadow);

    scene.add(this.group);
  }

  public showSpeechBubble(text: string) {
    this.speechBubble.material = new THREE.MeshBasicMaterial({
      map: TextureGenerator.createSpeechBubbleTexture(text),
      transparent: true,
      depthTest: false,
    });
    this.speechBubble.visible = true;
    this.speechTimer = 4.0;
  }

  private createNameplate(text: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(20, 14, 10, 0.75)';
    ctx.strokeStyle = '#c4a46a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(10, 10, 236, 44, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#fff4dc';
    ctx.font = 'bold 22px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 33);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.8, 0.45, 1);
    return sprite;
  }

  public updateData(data: { x: number; z: number; facingLeft: boolean; isMoving: boolean; isChopping: boolean; isDancing?: boolean; tool: string | null; isDead: boolean }) {
    this.prevPos.copy(this.targetPos);
    this.targetPos.set(data.x, 0, data.z);
    this.facingLeft = data.facingLeft;
    this.isMoving = data.isMoving;
    this.isChopping = data.isChopping;
    if (data.isDancing !== undefined) {
      this.isDancing = data.isDancing;
    }
    this.equippedTool = data.tool;
    this.isDead = data.isDead;
  }

  private updatePerspectiveSprites(cameraYaw: number) {
    const vx = this.targetPos.x - this.prevPos.x;
    const vz = this.targetPos.z - this.prevPos.z;
    const speedSq = vx * vx + vz * vz;

    if (speedSq > 0.001) {
      const moveAngle = Math.atan2(vx, vz);
      let deltaAngle = moveAngle - cameraYaw;
      while (deltaAngle > Math.PI) deltaAngle -= Math.PI * 2;
      while (deltaAngle < -Math.PI) deltaAngle += Math.PI * 2;

      const absDelta = Math.abs(deltaAngle);
      const buf = this.viewHysteresisBuffer;

      let nextView: 'FRONT' | 'SIDE' | 'BACK' = this.currentView;

      if (this.currentView === 'FRONT') {
        if (absDelta > 0.85 + buf) {
          nextView = absDelta > 2.25 ? 'BACK' : 'SIDE';
        }
      } else if (this.currentView === 'BACK') {
        if (absDelta < 2.25 - buf) {
          nextView = absDelta < 0.85 ? 'FRONT' : 'SIDE';
        }
      } else {
        if (absDelta < 0.85 - buf) {
          nextView = 'FRONT';
        } else if (absDelta > 2.25 + buf) {
          nextView = 'BACK';
        }
      }

      this.currentView = nextView;
    }

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
      this.head.material = this.headMatSide;
      this.torso.material = this.torsoMatSide;
      this.backpack.visible = true;
      this.backpack.position.set(-0.24, 0.45, -0.03);
    }
  }

  public update(delta: number, camera: THREE.Camera) {
    // Smooth frame-rate-independent exponential damping to network target position
    dampVector3(this.group.position, this.targetPos, 14, delta);

    // Smooth facing direction squash flip
    const targetSign = this.facingLeft ? -1 : 1;
    this.facingScaleX = damp(this.facingScaleX, targetSign, 24, delta);

    // Tool visibility
    if (this.equippedTool === 'axe') {
      this.heldTool.visible = true;
      this.heldTool.material = this.axeMat;
      this.torchLight.intensity = 0;
    } else if (this.equippedTool === 'torch') {
      this.heldTool.visible = true;
      this.heldTool.material = this.torchMat;
      this.torchLight.intensity = 1.8 + Math.sin(Date.now() * 0.02) * 0.2;
    } else if (this.equippedTool === 'pan') {
      this.heldTool.visible = true;
      this.heldTool.material = this.panMat;
      this.torchLight.intensity = 0;
    } else if (this.equippedTool === 'chicken') {
      this.heldTool.visible = true;
      this.heldTool.material = this.chickenMat;
      this.torchLight.intensity = 0;
    } else if (this.equippedTool === 'marshmallow') {
      this.heldTool.visible = true;
      this.heldTool.material = this.marshmallowMat;
      this.torchLight.intensity = 1.2 + Math.sin(Date.now() * 0.03) * 0.15;
    } else {
      this.heldTool.visible = false;
      this.torchLight.intensity = 0;
    }

    // Speech bubble decay
    if (this.speechTimer > 0) {
      this.speechTimer -= delta;
      const s = Math.min(1.0, (4.0 - this.speechTimer) * 4);
      this.speechBubble.scale.set(s, s, 1);
      if (this.speechTimer <= 0) {
        this.speechBubble.visible = false;
      }
    }

    // Billboarding face camera
    const camEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.group.rotation.y = camEuler.y;

    // Multi-Directional Perspective Swapping
    this.updatePerspectiveSprites(camEuler.y);

    // Movement & Procedural Walk Cycle
    if (this.isMoving && !this.isChopping) {
      this.walkPhase += delta * 15;
      const legAngle = Math.sin(this.walkPhase) * 0.72;
      this.leftLeg.rotation.x = legAngle;
      this.rightLeg.rotation.x = -legAngle;

      this.torso.rotation.x = THREE.MathUtils.lerp(this.torso.rotation.x, 0.22, 0.2);
      this.torso.rotation.z = Math.sin(this.walkPhase) * 0.07;
      this.leftArm.rotation.x = -legAngle * 0.85;
      this.rightArm.rotation.x = this.heldTool.visible ? -0.35 + legAngle * 0.45 : legAngle * 0.85;

      const bob = Math.abs(Math.sin(this.walkPhase)) * 0.11;
      this.torso.position.y = 0.65 + bob;

      const strideSquash = Math.sin(this.walkPhase * 2) * 0.06;
      this.torso.scale.y = 1.0 - strideSquash;
      this.torso.scale.x = this.facingScaleX * (1.0 + strideSquash * 0.6);

      this.head.rotation.x = -0.12 + Math.sin(this.walkPhase - 0.4) * 0.09;
      this.head.rotation.z = Math.sin(this.walkPhase * 0.5) * 0.05;
    } else if (this.isDancing) {
      this.dancePhase += delta * 14;
      this.torso.rotation.z = Math.sin(this.dancePhase) * 0.32;
      this.torso.rotation.x = Math.sin(this.dancePhase * 0.5) * 0.15;
      this.leftArm.rotation.z = Math.sin(this.dancePhase) * 1.1 + 0.9;
      this.rightArm.rotation.z = -Math.sin(this.dancePhase) * 1.1 - 0.9;
      this.head.rotation.z = -Math.sin(this.dancePhase) * 0.25;
      this.torso.position.y = 0.65 + Math.abs(Math.sin(this.dancePhase * 2)) * 0.18;
      this.torso.scale.set(this.facingScaleX, 1.0 + Math.sin(this.dancePhase * 3) * 0.08, 1.0);
    } else if (this.isChopping) {
      this.chopTimer += delta;
      const progress = Math.min(1.0, this.chopTimer / 0.45);
      if (progress < 0.35) {
        const t = progress / 0.35;
        this.rightArm.rotation.x = THREE.MathUtils.lerp(0, -1.6, t);
        this.torso.rotation.x = THREE.MathUtils.lerp(0, -0.22, t);
      } else if (progress < 0.65) {
        const t = (progress - 0.35) / 0.3;
        this.rightArm.rotation.x = THREE.MathUtils.lerp(-1.6, 1.15, t);
        this.torso.rotation.x = THREE.MathUtils.lerp(-0.22, 0.32, t);
      } else {
        const t = (progress - 0.65) / 0.35;
        this.rightArm.rotation.x = THREE.MathUtils.lerp(1.15, 0, t);
        this.torso.rotation.x = THREE.MathUtils.lerp(0.32, 0, t);
      }
      if (this.chopTimer >= 0.45) {
        this.chopTimer = 0;
      }
    } else {
      // Idle Breathing
      const idleTime = Date.now() * 0.0028;
      const breathe = Math.sin(idleTime) * 0.035;
      this.torso.position.y = 0.65 + breathe;
      this.torso.rotation.x = THREE.MathUtils.lerp(this.torso.rotation.x, 0, 0.15);
      this.torso.rotation.z = THREE.MathUtils.lerp(this.torso.rotation.z, 0, 0.15);
      this.torso.scale.set(this.facingScaleX * (1.0 - breathe * 0.4), 1.0 + breathe, 1.0);

      this.head.rotation.x = Math.sin(idleTime * 0.6) * 0.03;
      this.head.rotation.z = Math.sin(idleTime * 0.3) * 0.04;
      this.leftArm.rotation.z = Math.sin(idleTime) * 0.05;
      this.rightArm.rotation.z = -Math.sin(idleTime) * 0.05;

      this.leftLeg.rotation.x = THREE.MathUtils.lerp(this.leftLeg.rotation.x, 0, 0.2);
      this.rightLeg.rotation.x = THREE.MathUtils.lerp(this.rightLeg.rotation.x, 0, 0.2);
      this.leftArm.rotation.x = THREE.MathUtils.lerp(this.leftArm.rotation.x, 0, 0.2);
      this.rightArm.rotation.x = THREE.MathUtils.lerp(this.rightArm.rotation.x, 0, 0.2);
    }
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.group);
  }
}
