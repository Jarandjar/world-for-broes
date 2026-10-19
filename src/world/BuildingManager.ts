import * as THREE from 'three';
import { SettlementGenerator } from './SettlementGenerator';
import { BuildingEntity } from './BuildingEntity';
import { HidingSpot } from './HidingSpot';
import { gameConfig } from '../config/GameConfig';

/**
 * Building Manager Engine (Phase Z3).
 * Manages all physical building structures, pop-up roof flap animations,
 * interior visibility cutaways, and slapstick hiding spots across the world.
 */
export class BuildingManager {
  public buildings: Map<string, BuildingEntity> = new Map();
  public group: THREE.Group;
  public currentBuilding: BuildingEntity | null = null;
  public currentHidingSpot: HidingSpot | null = null;
  private scene?: THREE.Scene;

  constructor(settlements: SettlementGenerator, scene?: THREE.Scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'BuildingNetwork';

    this.initBuildings(settlements);

    if (scene) {
      scene.add(this.group);
      this.setVisible(gameConfig.isZombieMode());

      gameConfig.subscribe((cfg) => {
        this.setVisible(cfg.mode === 'zombie');
      });
    }
  }

  public setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  private initBuildings(settlements: SettlementGenerator) {
    const allLots = settlements.getAllLots();
    for (const lot of allLots) {
      const bldg = new BuildingEntity(lot);
      this.buildings.set(lot.id, bldg);
      this.group.add(bldg.group);
    }
  }

  /**
   * Updates buildings in proximity to player:
   * - Evaluates doorway entry and interior containment.
   * - Runs pop-up roof flap rotation tweens.
   * - Fades front walls for interior visibility.
   */
  public update(delta: number, playerPos: THREE.Vector3) {
    if (!this.group.visible) return;

    let insideAny: BuildingEntity | null = null;

    // Proximity update: only update buildings within 45m of player
    for (const bldg of this.buildings.values()) {
      const dx = playerPos.x - bldg.lot.position.x;
      const dz = playerPos.z - bldg.lot.position.z;
      const distSq = dx * dx + dz * dz;

      if (distSq <= 45 * 45) {
        bldg.update(delta, playerPos);
        if (bldg.isPlayerInside) {
          insideAny = bldg;
        }
      } else if (bldg.roofFoldProgress > 0.01) {
        // Smoothly close roof if player walked far away
        bldg.update(delta, playerPos);
      }
    }

    this.currentBuilding = insideAny;
  }

  /**
   * Returns nearby hiding spot within interaction reach of playerPos, if any.
   */
  public getNearbyHidingSpot(playerPos: THREE.Vector3): HidingSpot | null {
    for (const bldg of this.buildings.values()) {
      const spot = bldg.getNearbyHidingSpot(playerPos);
      if (spot) return spot;
    }
    return null;
  }

  /**
   * Interacts with nearby hiding spot.
   * Returns true if player successfully hid inside, false if player exited or none nearby.
   */
  public interactWithHidingSpot(playerPos: THREE.Vector3): { success: boolean; spot: HidingSpot | null; isHiding: boolean } {
    if (this.currentHidingSpot) {
      // Player is currently hiding -> exit!
      const spot = this.currentHidingSpot;
      spot.toggleHide();
      this.currentHidingSpot = null;
      return { success: true, spot, isHiding: false };
    }

    const nearby = this.getNearbyHidingSpot(playerPos);
    if (nearby) {
      const isNowHiding = nearby.toggleHide();
      this.currentHidingSpot = isNowHiding ? nearby : null;
      return { success: true, spot: nearby, isHiding: isNowHiding };
    }

    return { success: false, spot: null, isHiding: false };
  }

  public isPlayerHiding(): boolean {
    return this.currentHidingSpot !== null && this.currentHidingSpot.isOccupied;
  }
}
