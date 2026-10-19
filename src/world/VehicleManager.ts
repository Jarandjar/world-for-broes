import * as THREE from 'three';
import { VehicleEntity } from '../entities/VehicleEntity';
import { VEHICLE_REGISTRY } from '../registries/VehicleDef';
import { RoadNetwork } from './RoadNetwork';
import { ParticleManager } from '../entities/WorldObjects';
import { CrowdZombieEngine } from '../entities/CrowdZombieEngine';
import { gameConfig } from '../config/GameConfig';

import { BuildingLot } from './SettlementDef';

export class VehicleManager {
  public vehicles: VehicleEntity[] = [];
  private scene: THREE.Scene;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Spawn a vehicle at given world coordinates
   */
  public spawnVehicle(
    vehicleId: string = 'rusty_hatchback',
    position: THREE.Vector3,
    rotationY: number = 0
  ): VehicleEntity {
    const v = new VehicleEntity({
      scene: this.scene,
      vehicleId,
      position,
      rotationY,
    });
    this.vehicles.push(v);
    return v;
  }

  /**
   * Populate vehicles across settlements and roadsides
   */
  public populateVehicles(settlementLots: BuildingLot[] = []) {
    if (gameConfig.isClassicMode()) return;

    // 1. Starter Hatchback near the initial settlement road (at [8.0, 0, 4.0])
    this.spawnVehicle('rusty_hatchback', new THREE.Vector3(8.0, 0, 4.0), Math.PI * 0.25);

    // 2. Spawn a vehicle at settlements with garages, gas stations, or farmsteads
    settlementLots.forEach(lot => {
      const lower = lot.buildingType.toLowerCase();
      if (lower.includes('gas') || lower.includes('garage') || lower.includes('strip') || lower.includes('outpost') || lower.includes('farm')) {
        // Place vehicle in the driveway lot
        const offsetPos = new THREE.Vector3(lot.position.x + 3.5, 0, lot.position.z + 2.5);
        this.spawnVehicle('rusty_hatchback', offsetPos, Math.random() * Math.PI * 2);
      }
    });
  }

  /**
   * Find nearest vehicle to a position
   */
  public getNearbyVehicle(pos: THREE.Vector3, radius: number = 2.8): VehicleEntity | null {
    let nearest: VehicleEntity | null = null;
    let minDist = radius;

    for (const v of this.vehicles) {
      const d = v.position.distanceTo(pos);
      if (d < minDist) {
        minDist = d;
        nearest = v;
      }
    }

    return nearest;
  }

  /**
   * Get the vehicle currently being driven by the player
   */
  public getActiveVehicle(): VehicleEntity | null {
    return this.vehicles.find(v => v.isDriverSeated) || null;
  }

  /**
   * Update all vehicles, physics, collisions, and zombie impacts
   */
  public update(
    dt: number,
    roadNetwork?: RoadNetwork,
    particles?: ParticleManager,
    camera?: THREE.Camera,
    zombies?: CrowdZombieEngine,
    obstacles?: { position: THREE.Vector3; radius: number }[],
    isNight?: boolean,
    onNoiseBroadcast?: (pos: THREE.Vector3, radius: number) => void
  ) {
    if (gameConfig.isClassicMode()) return;

    for (const vehicle of this.vehicles) {
      vehicle.update(
        dt,
        roadNetwork,
        particles,
        camera,
        undefined,
        undefined,
        isNight
      );

      // Only check collisions for moving vehicles
      if (Math.abs(vehicle.speed) > 0.5) {
        // 1. Obstacle Collisions
        if (obstacles) {
          for (const obs of obstacles) {
            const dist = vehicle.position.distanceTo(obs.position);
            const minDist = vehicle.collisionRadius + obs.radius;
            if (dist < minDist) {
              vehicle.handleCollisionWithObstacle(obs.position, particles, camera);
              break;
            }
          }
        }

        // 2. Vehicular Zombie Ramming (Phase Z6 & Section 66)
        if (zombies && Math.abs(vehicle.speed) > 1.2) {
          const hitRadius = vehicle.collisionRadius + 0.6;
          // Query nearby zombies around front of vehicle
          const forward = new THREE.Vector3(0, 0, 1.2).applyAxisAngle(new THREE.Vector3(0, 1, 0), vehicle.group.rotation.y);
          const frontBumperPos = vehicle.position.clone().add(forward);

          const nearbyZombies = zombies.getZombiesInRadius(frontBumperPos, hitRadius);
          if (nearbyZombies.length > 0) {
            const dmg = vehicle.handleRamZombie(frontBumperPos, particles, camera);
            const knockback = vehicle.velocity.clone().normalize().multiplyScalar(Math.abs(vehicle.speed) * 1.5);
            
            zombies.hitZombiesInRadius(
              frontBumperPos,
              hitRadius,
              dmg,
              knockback,
              particles,
              camera
            );
          }
        }
      }
    }
  }

  public clear() {
    for (const v of this.vehicles) {
      v.destroy(this.scene);
    }
    this.vehicles = [];
  }
}
