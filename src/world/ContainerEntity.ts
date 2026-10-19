import * as THREE from 'three';
import { ContainerDef } from '../registries/ContainerDef';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { DampedSpring } from '../animation/TweenPreset';
import { SpringPreset } from '../animation/TweenPreset';
import { InventorySystem } from '../systems/InventorySystem';
import { rollLootFromTable } from '../registries/LootTable';
import { ITEM_REGISTRY, ItemRarity } from '../registries/ItemDef';
import { ComicPops } from '../ui/ComicPops';
import { sounds } from '../audio/SoundEffects';
import { ParabolicLootManager } from './ParabolicLootArc';
import { JackpotFXManager } from './JackpotFX';
import { gameConfig } from '../config/GameConfig';

export interface ContainerEntityOptions {
  def: ContainerDef;
  position: THREE.Vector3;
  rotationY?: number;
}

/**
 * ContainerEntity (Section 15, Section 66 & Section 75).
 * 3D diorama containers with rubbery spring-hinged lid/door opening animations,
 * locked state verification, parabolic loot ejection arcs, and jackpot siren alerts.
 */
export class ContainerEntity {
  public def: ContainerDef;
  public position: THREE.Vector3;
  public rotationY: number;
  public mesh: THREE.Group;

  public isOpened: boolean = false;
  public isLocked: boolean = false;
  public restockTimer: number = 0;
  public interactionRadius: number = 2.4;

  private lidPivot: THREE.Group;
  private drawerMesh?: THREE.Mesh;
  private spring: DampedSpring;
  private lidAxis: 'x' | 'y' | 'z' | 'drawer' = 'x';
  private maxLidAngle: number = -1.85;

  private outlineColor: number = 0x2e1810;
  private sparkleMesh?: THREE.Mesh;

  constructor(scene: THREE.Scene, opts: ContainerEntityOptions) {
    this.def = opts.def;
    this.position = opts.position.clone();
    this.rotationY = opts.rotationY || 0;
    this.isLocked = !!this.def.isLocked;

    // Second-order spring with rubbery cartoon bounce
    this.spring = new DampedSpring(SpringPreset.bouncy, 0.0);

    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = this.rotationY;

    this.lidPivot = new THREE.Group();
    this.buildContainerModel();

    // Rarity glint sparkle on top of container
    if (this.def.sparkleTier && this.def.sparkleTier !== 'C') {
      const glintGeo = new THREE.PlaneGeometry(0.3, 0.3);
      const glintMat = new THREE.MeshBasicMaterial({
        color: this.def.sparkleTier === 'X' ? 0xff0077 : this.def.sparkleTier === 'R' ? 0xffd700 : 0x00f0ff,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide,
      });
      this.sparkleMesh = new THREE.Mesh(glintGeo, glintMat);
      this.sparkleMesh.position.y = 1.0;
      this.mesh.add(this.sparkleMesh);
    }

    scene.add(this.mesh);
  }

  private buildContainerModel() {
    const id = this.def.id;
    const tex = TextureGenerator.getContainerTexture(id);
    const mat = new THREE.MeshLambertMaterial({
      map: tex,
      color: 0xffffff,
    });

    const outlineMat = new THREE.MeshBasicMaterial({
      color: this.outlineColor,
      side: THREE.BackSide,
    });

    switch (id) {
      case 'toolbox': {
        // Red mechanic toolbox
        const w = 0.8, h = 0.42, d = 0.48;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Top lid hinged at back top edge
        const lidGeo = new THREE.BoxGeometry(w + 0.04, 0.1, d + 0.04);
        lidGeo.translate(0, 0.05, (d + 0.04) * 0.5);
        const lidMesh = new THREE.Mesh(lidGeo, mat);
        lidMesh.castShadow = true;

        this.lidPivot.position.set(0, h, -d * 0.5);
        this.lidPivot.add(lidMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'x';
        this.maxLidAngle = -1.9;
        break;
      }
      case 'cash_register': {
        // Antique cash register with sliding bottom drawer
        const w = 0.7, h = 0.55, d = 0.6;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Cash drawer slides out forward along Z
        const drawerGeo = new THREE.BoxGeometry(w - 0.08, 0.16, d - 0.1);
        this.drawerMesh = new THREE.Mesh(drawerGeo, mat);
        this.drawerMesh.position.set(0, 0.12, 0);
        this.drawerMesh.castShadow = true;
        this.mesh.add(this.drawerMesh);

        this.lidAxis = 'drawer';
        break;
      }
      case 'fridge': {
        // Refrigerator with side-hinged door
        const w = 0.9, h = 1.8, d = 0.8;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Door hinged at front-left edge
        const doorGeo = new THREE.BoxGeometry(w, h + 0.02, 0.1);
        doorGeo.translate(w * 0.5, h * 0.5, 0.05);
        const doorMesh = new THREE.Mesh(doorGeo, mat);
        doorMesh.castShadow = true;

        this.lidPivot.position.set(-w * 0.5, 0, d * 0.5);
        this.lidPivot.add(doorMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'y';
        this.maxLidAngle = -1.95;
        break;
      }
      case 'locker': {
        // Tall locker with front-left hinged door
        const w = 0.65, h = 2.0, d = 0.6;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        const doorGeo = new THREE.BoxGeometry(w, h, 0.06);
        doorGeo.translate(w * 0.5, h * 0.5, 0.03);
        const doorMesh = new THREE.Mesh(doorGeo, mat);
        doorMesh.castShadow = true;

        this.lidPivot.position.set(-w * 0.5, 0, d * 0.5);
        this.lidPivot.add(doorMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'y';
        this.maxLidAngle = -1.9;
        break;
      }
      case 'vault_safe': {
        // Heavy titanium vault safe
        const w = 1.2, h = 1.2, d = 1.0;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Heavy door hinged at front-left
        const doorGeo = new THREE.BoxGeometry(w, h, 0.16);
        doorGeo.translate(w * 0.5, h * 0.5, 0.08);
        const doorMesh = new THREE.Mesh(doorGeo, mat);
        doorMesh.castShadow = true;

        this.lidPivot.position.set(-w * 0.5, 0, d * 0.5);
        this.lidPivot.add(doorMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'y';
        this.maxLidAngle = -1.75;
        break;
      }
      case 'supply_drop_crate': {
        // Wooden military supply crate
        const w = 1.2, h = 0.85, d = 0.9;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Top lid hinged at back top
        const lidGeo = new THREE.BoxGeometry(w + 0.06, 0.12, d + 0.06);
        lidGeo.translate(0, 0.06, (d + 0.06) * 0.5);
        const lidMesh = new THREE.Mesh(lidGeo, mat);
        lidMesh.castShadow = true;

        this.lidPivot.position.set(0, h, -d * 0.5);
        this.lidPivot.add(lidMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'x';
        this.maxLidAngle = -1.8;
        break;
      }
      case 'vending_machine': {
        // Tall neon vending machine
        const w = 1.0, h = 2.1, d = 0.8;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Bottom push door hinged at top of dispense slot
        const flapGeo = new THREE.BoxGeometry(w * 0.75, 0.22, 0.04);
        flapGeo.translate(0, -0.11, 0.02);
        const flapMesh = new THREE.Mesh(flapGeo, mat);

        this.lidPivot.position.set(0, 0.45, d * 0.5);
        this.lidPivot.add(flapMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'x';
        this.maxLidAngle = 1.35; // Pushes inward
        break;
      }
      case 'pinata': {
        // Suspended rainbow piñata
        const w = 0.8, h = 0.7, d = 0.5;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = 1.4;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        // Hanging string
        const stringGeo = new THREE.CylinderGeometry(0.015, 0.015, 1.2);
        const stringMat = new THREE.MeshBasicMaterial({ color: 0xcccccc });
        const stringMesh = new THREE.Mesh(stringGeo, stringMat);
        stringMesh.position.y = 2.2;
        this.mesh.add(stringMesh);

        this.lidPivot.position.set(0, 1.4, 0);
        this.lidAxis = 'z';
        this.maxLidAngle = 0.6; // Swings on hit
        break;
      }
      case 'suitcase': {
        // Traveler leather suitcase
        const w = 0.85, h = 0.35, d = 0.6;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        const lidGeo = new THREE.BoxGeometry(w + 0.03, 0.08, d + 0.03);
        lidGeo.translate(0, 0.04, (d + 0.03) * 0.5);
        const lidMesh = new THREE.Mesh(lidGeo, mat);
        lidMesh.castShadow = true;

        this.lidPivot.position.set(0, h, -d * 0.5);
        this.lidPivot.add(lidMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'x';
        this.maxLidAngle = -1.95;
        break;
      }
      case 'medicine_cabinet': {
        // Wall-mounted medicine cabinet
        const w = 0.7, h = 0.9, d = 0.35;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = 1.3;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        const doorGeo = new THREE.BoxGeometry(w, h, 0.05);
        doorGeo.translate(w * 0.5, h * 0.5, 0.025);
        const doorMesh = new THREE.Mesh(doorGeo, mat);
        doorMesh.castShadow = true;

        this.lidPivot.position.set(-w * 0.5, 1.3 - h * 0.5, d * 0.5);
        this.lidPivot.add(doorMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'y';
        this.maxLidAngle = -1.85;
        break;
      }
      default: {
        // Generic box
        const w = 0.8, h = 0.6, d = 0.6;
        const baseGeo = new THREE.BoxGeometry(w, h, d);
        const baseMesh = new THREE.Mesh(baseGeo, mat);
        baseMesh.position.y = h * 0.5;
        baseMesh.castShadow = true;
        this.mesh.add(baseMesh);

        const lidGeo = new THREE.BoxGeometry(w + 0.04, 0.1, d + 0.04);
        lidGeo.translate(0, 0.05, (d + 0.04) * 0.5);
        const lidMesh = new THREE.Mesh(lidGeo, mat);

        this.lidPivot.position.set(0, h, -d * 0.5);
        this.lidPivot.add(lidMesh);
        this.mesh.add(this.lidPivot);

        this.lidAxis = 'x';
        this.maxLidAngle = -1.8;
        break;
      }
    }
  }

  public getPrompt(inventory: InventorySystem): string {
    if (this.isOpened) {
      return `[EMPTY / SEARCHED]`;
    }

    if (this.isLocked && this.def.lockType) {
      const toolName = this.getLockToolName(this.def.lockType);
      const hasKey = inventory.getItemCount(this.def.lockType) > 0 || (this.def.lockType === 'crowbar' && inventory.getItemCount('crowbar') > 0);
      if (hasKey) {
        return `Unlock & Open [E] (Uses ${toolName})`;
      } else {
        return `[LOCKED] Requires ${toolName}`;
      }
    }

    return `Open ${this.def.name} [E]`;
  }

  private getLockToolName(lockType: string): string {
    switch (lockType) {
      case 'crowbar': return 'Crowbar';
      case 'padlock': return 'Locker Key / Lockpick';
      case 'keycard_red': return 'Red Keycard';
      case 'keycard_blue': return 'Blue Keycard';
      case 'keycard_green': return 'Green Keycard';
      default: return 'Key';
    }
  }

  public canInteract(playerPos: THREE.Vector3): boolean {
    return this.position.distanceTo(playerPos) <= this.interactionRadius;
  }

  /**
   * Opens the container:
   * - Verifies locks and key items.
   * - Animates spring-hinged lid.
   * - Spawns 3D parabolic loot arcs.
   * - Triggers jackpot beam & siren if applicable.
   */
  public open(
    inventory: InventorySystem,
    lootManager: ParabolicLootManager,
    jackpotManager: JackpotFXManager,
    particles?: any,
    camera?: THREE.Camera
  ): { success: boolean; reason?: string } {
    if (this.isOpened) {
      return { success: false, reason: 'already_empty' };
    }

    // Verify lock
    if (this.isLocked && this.def.lockType) {
      const hasKey = inventory.getItemCount(this.def.lockType) > 0 || (this.def.lockType === 'crowbar' && inventory.getItemCount('crowbar') > 0);
      if (!hasKey) {
        // Locked rattle comic pop & sound
        sounds.playHurt();
        if (camera) {
          ComicPops.spawnWorld(
            new THREE.Vector3(this.position.x, this.position.y + 1.2, this.position.z),
            camera,
            'RATTLE! 🔒',
            '#FF4444'
          );
        }
        return { success: false, reason: 'locked' };
      } else {
        // Unlocked!
        this.isLocked = false;
        sounds.playCraft();
      }
    }

    this.isOpened = true;
    this.spring.target = 1.0; // Trigger rubbery spring lid opening

    // Sound & Comic Pops
    let comicWord = 'OPEN!';
    let comicColor = '#4ADE80';
    if (this.def.id === 'cash_register') {
      comicWord = 'DING! 💵';
      comicColor = '#FFD700';
      sounds.playPick();
    } else if (this.def.id === 'toolbox' || this.def.id === 'vault_safe') {
      comicWord = 'CLANG! ⚙️';
      comicColor = '#E0E0E0';
      sounds.playTreeFall();
    } else if (this.def.id === 'pinata') {
      comicWord = 'POOF! 🪅';
      comicColor = '#FF477E';
      sounds.playCraft();
      if (particles && typeof particles.spawnWoodChips === 'function') {
        particles.spawnWoodChips(this.position, 16);
      }
    } else {
      sounds.playChop();
    }

    if (camera) {
      ComicPops.spawnWorld(
        new THREE.Vector3(this.position.x, this.position.y + 1.5, this.position.z),
        camera,
        comicWord,
        comicColor
      );
    }

    // Roll Loot Items
    const density = gameConfig.get().lootDensity === 'high' ? 1.25 : gameConfig.get().lootDensity === 'low' ? 0.75 : 1.0;
    const rolled = rollLootFromTable(
      this.def.defaultLootTable,
      this.def.minItems,
      this.def.maxItems,
      density
    );

    let hasJackpotLoot = false;

    // Eject items along parabolic arcs
    const lootOrigin = new THREE.Vector3(this.position.x, this.position.y + 0.6, this.position.z);
    rolled.forEach((item, idx) => {
      const def = ITEM_REGISTRY[item.itemId];
      if (def?.rarity === 'R' || def?.rarity === 'X') {
        hasJackpotLoot = true;
      }

      // Radiate outwards facing player/forward
      const spreadAngle = this.rotationY + ((idx - (rolled.length - 1) * 0.5) * 0.6) + (Math.random() - 0.5) * 0.3;
      lootManager.spawnArc({
        itemId: item.itemId,
        count: item.count,
        origin: lootOrigin,
        angleRad: spreadAngle,
        upwardSpeed: 3.8 + Math.random() * 1.2,
        radialSpeed: 1.8 + Math.random() * 1.2,
        rarity: def?.rarity || 'C',
      });
    });

    // Jackpot Alarm & Golden Light Beam
    if (this.def.isJackpot || hasJackpotLoot) {
      jackpotManager.spawnJackpot({
        position: this.position,
        durationSeconds: 10.0,
        camera,
      });
    }

    // Initialize restock countdown (1 game day = 120s)
    this.restockTimer = this.def.restockDays * 120;

    return { success: true };
  }

  public update(delta: number, playerPos: THREE.Vector3) {
    // 1. Advance second-order spring dynamics
    const springVal = this.spring.update(delta);

    // Apply lid/door angle or drawer slide
    if (this.lidAxis === 'x') {
      this.lidPivot.rotation.x = this.maxLidAngle * springVal;
    } else if (this.lidAxis === 'y') {
      this.lidPivot.rotation.y = this.maxLidAngle * springVal;
    } else if (this.lidAxis === 'z') {
      this.lidPivot.rotation.z = Math.sin(springVal * Math.PI * 4) * 0.35 * (1 - springVal);
    } else if (this.lidAxis === 'drawer' && this.drawerMesh) {
      this.drawerMesh.position.z = 0.35 * springVal;
    }

    // 2. Rotate sparkle glint
    if (this.sparkleMesh && !this.isOpened) {
      this.sparkleMesh.rotation.z += delta * 2.0;
      this.sparkleMesh.visible = true;
    } else if (this.sparkleMesh) {
      this.sparkleMesh.visible = false;
    }

    // 3. Restock Timer
    if (this.isOpened && this.restockTimer > 0) {
      this.restockTimer -= delta;
      if (this.restockTimer <= 0) {
        // Restock container with fresh loot!
        this.isOpened = false;
        this.spring.target = 0.0; // Spring lid snaps closed
        if (this.def.isLocked) {
          this.isLocked = true;
        }
      }
    }
  }

  public setVisible(visible: boolean) {
    this.mesh.visible = visible;
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.mesh);
  }
}
