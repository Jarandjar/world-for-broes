import * as THREE from 'three';
import { CONTAINER_REGISTRY } from '../registries/ContainerDef';
import { ContainerEntity } from './ContainerEntity';
import { ParabolicLootManager } from './ParabolicLootArc';
import { JackpotFXManager } from './JackpotFX';
import { BuildingLot } from './SettlementDef';
import { InventorySystem } from '../systems/InventorySystem';

/**
 * ContainerManager (Section 15 & Section 66).
 * Coordinates all interactive searchables, spring-hinged lid containers,
 * parabolic loot ejection arcs, and jackpot light beams across settlements and the world.
 */
export class ContainerManager {
  public containers: ContainerEntity[] = [];
  public lootManager: ParabolicLootManager;
  public jackpotManager: JackpotFXManager;

  private scene: THREE.Scene;
  private isVisible: boolean = true;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.lootManager = new ParabolicLootManager(scene);
    this.jackpotManager = new JackpotFXManager(scene);
  }

  /**
   * Automatically populates containers inside settlement building lots
   * based on zone, building type, and spatial dimensions.
   */
  public populateFromSettlementLots(lots: BuildingLot[]) {
    lots.forEach(lot => {
      const halfW = lot.size.width * 0.5;
      const halfD = lot.size.depth * 0.5;
      const rot = lot.rotationY;

      // Local to world offset helper
      const getPos = (lx: number, lz: number): THREE.Vector3 => {
        const cos = Math.cos(rot);
        const sin = Math.sin(rot);
        const wx = lot.position.x + (lx * cos - lz * sin);
        const wz = lot.position.z + (lx * sin + lz * cos);
        return new THREE.Vector3(wx, 0, wz);
      };

      if (lot.zone === 'residential' || lot.buildingType === 'cabin' || lot.buildingType === 'motel') {
        // Suitcase in bedroom corner
        this.spawnContainer('suitcase', getPos(halfW - 1.5, -halfD + 1.6), rot);

        // Refrigerator in kitchen corner
        this.spawnContainer('fridge', getPos(-halfW + 1.4, halfD - 1.5), rot + Math.PI * 0.5);

        // Medicine cabinet on wall
        this.spawnContainer('medicine_cabinet', getPos(-halfW + 0.5, -halfD + 2.0), rot + Math.PI * 0.5);
      } else if (lot.zone === 'commercial' || lot.buildingType === 'diner' || lot.buildingType === 'supermarket') {
        // Cash Register on counter
        this.spawnContainer('cash_register', getPos(0, 0), rot);

        // Vending Machine in lobby
        this.spawnContainer('vending_machine', getPos(halfW - 1.4, halfD - 1.4), rot);

        // Commercial Refrigerator
        this.spawnContainer('fridge', getPos(-halfW + 1.4, -halfD + 1.6), rot);
      } else if (lot.zone === 'industrial' || lot.buildingType === 'auto_garage') {
        // Red Mechanic Toolbox on bench
        this.spawnContainer('toolbox', getPos(-halfW + 1.5, 0), rot);

        // Metal Locker
        this.spawnContainer('locker', getPos(halfW - 1.4, -halfD + 1.5), rot);

        // Piñata in workshop corner (comical surprise)
        if (lot.buildingType === 'auto_garage') {
          this.spawnContainer('pinata', getPos(0, -halfD + 1.8), rot);
        }
      } else if (lot.zone === 'civic' || lot.buildingType === 'town_hall' || lot.buildingType === 'outpost') {
        // Titanium Vault Safe in inner office
        this.spawnContainer('vault_safe', getPos(0, -halfD + 1.8), rot);

        // Supply Drop Crate in courtyard/rear
        this.spawnContainer('supply_drop_crate', getPos(halfW + 2.2, 0), rot + Math.PI * 0.25);
      }
    });
  }

  public spawnContainer(id: string, pos: THREE.Vector3, rotY: number = 0): ContainerEntity | null {
    const def = CONTAINER_REGISTRY[id];
    if (!def) {
      console.warn(`Container ID ${id} not found in CONTAINER_REGISTRY`);
      return null;
    }

    const container = new ContainerEntity(this.scene, {
      def,
      position: pos,
      rotationY: rotY,
    });

    container.setVisible(this.isVisible);
    this.containers.push(container);
    return container;
  }

  public getNearbyContainer(playerPos: THREE.Vector3): ContainerEntity | null {
    let nearest: ContainerEntity | null = null;
    let minDist = Infinity;

    for (const c of this.containers) {
      const dist = c.position.distanceTo(playerPos);
      if (dist <= c.interactionRadius && dist < minDist) {
        minDist = dist;
        nearest = c;
      }
    }

    return nearest;
  }

  public interactWithNearbyContainer(
    playerPos: THREE.Vector3,
    inventory: InventorySystem,
    particles?: any,
    camera?: THREE.Camera
  ): { success: boolean; container: ContainerEntity | null; reason?: string } {
    const nearby = this.getNearbyContainer(playerPos);
    if (!nearby) {
      return { success: false, container: null, reason: 'none_nearby' };
    }

    const res = nearby.open(inventory, this.lootManager, this.jackpotManager, particles, camera);
    return { success: res.success, container: nearby, reason: res.reason };
  }

  public update(
    delta: number,
    playerPos: THREE.Vector3,
    inventory: InventorySystem,
    particles?: any,
    camera?: THREE.Camera
  ) {
    // 1. Update container lids, drawer springs, and restock timers
    for (const c of this.containers) {
      c.update(delta, playerPos);
    }

    // 2. Update active parabolic loot arcs in ballistic flight / bouncing / settled
    this.lootManager.update(delta, playerPos, inventory, particles);

    // 3. Update active jackpot light beams and sirens
    this.jackpotManager.update(delta, camera);
  }

  public setVisible(visible: boolean) {
    this.isVisible = visible;
    for (const c of this.containers) {
      c.setVisible(visible);
    }
  }

  public clear() {
    for (const c of this.containers) {
      c.destroy(this.scene);
    }
    this.containers = [];
    this.lootManager.clear();
    this.jackpotManager.clear();
  }
}
