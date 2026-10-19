import * as THREE from 'three';
import { VehicleDef, VEHICLE_REGISTRY } from '../registries/VehicleDef';
import { RoadNetwork } from '../world/RoadNetwork';
import { ParticleManager } from './WorldObjects';
import { DampedSpring, SpringPreset } from '../animation/TweenPreset';
import { ComicPops } from '../ui/ComicPops';
import { sounds } from '../audio/SoundEffects';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { Player } from './Player';

export interface VehicleEntityOptions {
  scene: THREE.Scene;
  def?: VehicleDef;
  vehicleId?: string;
  position: THREE.Vector3;
  rotationY?: number;
}

/**
 * VehicleEntity (Phase Z6 - Sections 15, 38, 66, 73-81).
 * Features:
 * - 3D Diorama Papercraft/Cardboard Hatchback chassis with rubber-hose spring suspension.
 * - Arcade Vehicle Physics (velocity, acceleration, deceleration, steering knuckle yaw, wheel roll ω = v/r).
 * - Second-order suspension roll (lean on turns) & pitch (squat on accel, dive on braking).
 * - Interactive boarding & smooth hinged driver door swing tween (0 -> 65°).
 * - Headlights with warm volumetric road cones.
 * - Exhaust pipe cartoon put-put puffs and handbrake drift tire smoke.
 * - Slapstick sad trombone / car horn broadcasting a 35m noise ring to zombies.
 * - Obstacle collision bounce and vehicular zombie ramming with comic sound pops!
 */
export class VehicleEntity {
  public group: THREE.Group;
  public chassis: THREE.Group;
  public def: VehicleDef;
  public position: THREE.Vector3;
  public velocity: THREE.Vector3 = new THREE.Vector3();

  // Physics & Navigation State
  public speed: number = 0; // m/s forward/backward
  public targetSteerAngle: number = 0; // -1 (left) to +1 (right)
  public steeringAngle: number = 0; // Radians
  public isDrifting: boolean = false;
  public isDriverSeated: boolean = false;
  public driverPlayer: Player | null = null;

  // Mechanics & Consumables
  public fuel: number;
  public durability: number = 100; // 0 to 100%
  public batteryCharge: number = 85; // 0 to 100%
  public headlightsOn: boolean = false;
  public collisionRadius: number = 1.9;
  public parts: Map<string, { slot: string; id: string; name: string; icon: string; condition: number; isRequired: boolean }> = new Map();

  // 3D Sub-components & Rigs
  public wheelMeshes: THREE.Mesh[] = [];
  public frontLeftKnuckle: THREE.Group;
  public frontRightKnuckle: THREE.Group;
  public driverDoor: THREE.Group;
  public driverDoorPivot: THREE.Group;
  public doorTargetAngle: number = 0;
  public doorCurrentAngle: number = 0;
  public antenna: THREE.Mesh;
  public headlights: THREE.SpotLight[] = [];
  public headlightMeshL: THREE.Mesh;
  public headlightMeshR: THREE.Mesh;
  public exhaustPipe: THREE.Mesh;

  // Procedural Suspension Springs
  public rollSpring: DampedSpring;
  public pitchSpring: DampedSpring;
  public bounceSpring: DampedSpring;
  public antennaSpring: DampedSpring;

  // Wheel geometry constants
  public readonly wheelRadius: number = 0.38;

  // Timers & Audio Loop
  private exhaustTimer: number = 0;
  private driftSmokeTimer: number = 0;
  private engineRevSoundTimer: number = 0;
  private tireScreechTimer: number = 0;

  constructor(opts: VehicleEntityOptions) {
    this.def = opts.def || VEHICLE_REGISTRY[opts.vehicleId || 'rusty_hatchback'] || VEHICLE_REGISTRY.rusty_hatchback;
    this.fuel = this.def.fuelCapacity;

    // Initialize Default Parts Inventory (Section 15 & 66)
    this.parts.set('engine_block', { slot: 'engine_block', id: 'engine_block', name: 'Engine Block', icon: '🏎️', condition: 85, isRequired: true });
    this.parts.set('battery', { slot: 'battery', id: 'battery', name: '12V Lead Battery', icon: '🔋', condition: 90, isRequired: true });
    this.parts.set('spark_plugs', { slot: 'spark_plugs', id: 'spark_plugs', name: 'Spark Plugs', icon: '⚡', condition: 75, isRequired: true });
    this.parts.set('radiator_coolant', { slot: 'radiator_coolant', id: 'radiator_coolant', name: 'Radiator Coolant', icon: '❄️', condition: 80, isRequired: true });
    this.parts.set('tires', { slot: 'tires', id: 'tire_standard', name: 'All-Terrain Tires', icon: '🛞', condition: 85, isRequired: true });
    this.recalculateDurability();

    this.group = new THREE.Group();
    this.group.name = `Vehicle_${this.def.id}`;
    this.position = this.group.position;
    this.position.copy(opts.position);
    this.position.y = 0.02; // Ground baseline

    if (opts.rotationY !== undefined) {
      this.group.rotation.y = opts.rotationY;
    }

    // Suspension Springs
    this.rollSpring = new DampedSpring(SpringPreset.snappy);
    this.pitchSpring = new DampedSpring(SpringPreset.snappy);
    this.bounceSpring = new DampedSpring(SpringPreset.bouncy);
    this.antennaSpring = new DampedSpring(SpringPreset.loose);

    // Build 3D Diorama Chassis & Wheels
    this.chassis = new THREE.Group();
    this.chassis.position.y = 0.38; // Suspension baseline resting above wheels
    this.group.add(this.chassis);

    // Front Knuckles (steerable)
    this.frontLeftKnuckle = new THREE.Group();
    this.frontLeftKnuckle.position.set(-0.95, 0.38, 1.15);
    this.frontRightKnuckle = new THREE.Group();
    this.frontRightKnuckle.position.set(0.95, 0.38, 1.15);
    this.group.add(this.frontLeftKnuckle, this.frontRightKnuckle);

    // Driver Door Pivot (hinged on front door pillar)
    this.driverDoorPivot = new THREE.Group();
    this.driverDoorPivot.position.set(-0.91, 0.45, 0.4);
    this.driverDoor = new THREE.Group();
    this.driverDoorPivot.add(this.driverDoor);
    this.chassis.add(this.driverDoorPivot);

    // Build Model
    const [hL, hR, ant, exh] = this.buildDioramaMesh();
    this.headlightMeshL = hL;
    this.headlightMeshR = hR;
    this.antenna = ant;
    this.exhaustPipe = exh;

    opts.scene.add(this.group);
  }

  private buildDioramaMesh(): [THREE.Mesh, THREE.Mesh, THREE.Mesh, THREE.Mesh] {
    const isClassic = false;

    // 1. Lower Body / Chassis (Boxy papercraft look with bevel edges)
    const bodyGeo = new THREE.BoxGeometry(1.8, 0.65, 3.3);
    const bodyMat = new THREE.MeshLambertMaterial({
      color: 0xd86c27, // Rusted desert orange
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    bodyMesh.position.y = 0.35;
    this.chassis.add(bodyMesh);

    // 2. Cabin (Cutout windows and roof)
    const cabinGeo = new THREE.BoxGeometry(1.5, 0.75, 1.75);
    const cabinMat = new THREE.MeshLambertMaterial({
      color: 0x222228, // Dark interior & tinted paper windows
    });
    const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
    cabinMesh.position.set(0, 0.95, -0.2);
    cabinMesh.castShadow = true;
    this.chassis.add(cabinMesh);

    // Cabin Windshield / Windows Decals (light blue cel-shaded glint)
    const glassMat = new THREE.MeshBasicMaterial({ color: 0x9be8ff });
    const frontWindshield = new THREE.Mesh(new THREE.PlaneGeometry(1.35, 0.55), glassMat);
    frontWindshield.position.set(0, 0.98, 0.69);
    frontWindshield.rotation.x = -Math.PI * 0.12;
    this.chassis.add(frontWindshield);

    const rearWindshield = new THREE.Mesh(new THREE.PlaneGeometry(1.35, 0.5), glassMat);
    rearWindshield.position.set(0, 0.98, -1.09);
    rearWindshield.rotation.y = Math.PI;
    rearWindshield.rotation.x = -Math.PI * 0.12;
    this.chassis.add(rearWindshield);

    // Side windows
    const sideWindowGeo = new THREE.PlaneGeometry(1.4, 0.45);
    const sideL = new THREE.Mesh(sideWindowGeo, glassMat);
    sideL.position.set(-0.76, 0.95, -0.2);
    sideL.rotation.y = -Math.PI / 2;
    const sideR = new THREE.Mesh(sideWindowGeo, glassMat);
    sideR.position.set(0.76, 0.95, -0.2);
    sideR.rotation.y = Math.PI / 2;
    this.chassis.add(sideL, sideR);

    // 3. Hinged Driver Door (Front-left side)
    const doorGeo = new THREE.BoxGeometry(0.08, 0.7, 0.95);
    const doorMat = new THREE.MeshLambertMaterial({ color: 0xca601e });
    const doorMesh = new THREE.Mesh(doorGeo, doorMat);
    doorMesh.position.set(0, 0.2, -0.45); // Centered relative to hinge
    this.driverDoor.add(doorMesh);

    // Door handle (silver nub)
    const handleGeo = new THREE.BoxGeometry(0.06, 0.04, 0.14);
    const handleMat = new THREE.MeshLambertMaterial({ color: 0xdddddd });
    const handle = new THREE.Mesh(handleGeo, handleMat);
    handle.position.set(-0.06, 0.25, -0.75);
    this.driverDoor.add(handle);

    // 4. Visible Steering Wheel inside cabin
    const wheelRimGeo = new THREE.TorusGeometry(0.18, 0.03, 6, 12);
    const wheelMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const steering = new THREE.Mesh(wheelRimGeo, wheelMat);
    steering.position.set(-0.35, 0.88, 0.35);
    steering.rotation.x = Math.PI * 0.3;
    this.chassis.add(steering);

    // 5. Front Bumper & Radiator Grill
    const bumperGeo = new THREE.BoxGeometry(1.9, 0.25, 0.3);
    const bumperMat = new THREE.MeshLambertMaterial({ color: 0x333333 });
    const frontBumper = new THREE.Mesh(bumperGeo, bumperMat);
    frontBumper.position.set(0, 0.15, 1.7);
    const rearBumper = new THREE.Mesh(bumperGeo, bumperMat);
    rearBumper.position.set(0, 0.15, -1.7);
    this.chassis.add(frontBumper, rearBumper);

    // Radiator Grill
    const grillGeo = new THREE.PlaneGeometry(1.2, 0.32);
    const grillMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    const grill = new THREE.Mesh(grillGeo, grillMat);
    grill.position.set(0, 0.35, 1.66);
    this.chassis.add(grill);

    // 6. Headlights (Two warm lamps on bumper)
    const lampGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.12, 12);
    lampGeo.rotateX(Math.PI / 2);
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xfff8d6 });
    const headL = new THREE.Mesh(lampGeo, lampMat);
    headL.position.set(-0.62, 0.45, 1.66);
    const headR = new THREE.Mesh(lampGeo, lampMat);
    headR.position.set(0.62, 0.45, 1.66);
    this.chassis.add(headL, headR);

    // Spotlights for Night Driving
    const spotL = new THREE.SpotLight(0xfff5d0, 3.2, 18, Math.PI * 0.28, 0.45);
    spotL.position.set(-0.62, 0.45, 1.7);
    spotL.target.position.set(-0.62, 0.1, 12);
    this.chassis.add(spotL, spotL.target);

    const spotR = new THREE.SpotLight(0xfff5d0, 3.2, 18, Math.PI * 0.28, 0.45);
    spotR.position.set(0.62, 0.45, 1.7);
    spotR.target.position.set(0.62, 0.1, 12);
    this.chassis.add(spotR, spotR.target);

    this.headlights.push(spotL, spotR);

    // 7. Wobbly Antenna with Harmonic Tip
    const antennaGeo = new THREE.CylinderGeometry(0.015, 0.02, 0.95, 6);
    const antennaMat = new THREE.MeshLambertMaterial({ color: 0xcccccc });
    const antenna = new THREE.Mesh(antennaGeo, antennaMat);
    antenna.position.set(-0.72, 1.4, -0.85);
    this.chassis.add(antenna);

    // 8. Exhaust Pipe (Back-right)
    const exhaustGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.4, 8);
    exhaustGeo.rotateX(Math.PI / 2);
    const exhaustMat = new THREE.MeshLambertMaterial({ color: 0x444444 });
    const exhaust = new THREE.Mesh(exhaustGeo, exhaustMat);
    exhaust.position.set(0.65, 0.12, -1.75);
    this.chassis.add(exhaust);

    // 9. Four Wheels with Hubcaps
    const tireGeo = new THREE.CylinderGeometry(this.wheelRadius, this.wheelRadius, 0.28, 12);
    tireGeo.rotateZ(Math.PI / 2);
    const tireMat = new THREE.MeshLambertMaterial({ color: 0x1f1f22 });
    const hubMat = new THREE.MeshBasicMaterial({ color: 0xdcdcdc });
    const hubGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.29, 8);
    hubGeo.rotateZ(Math.PI / 2);

    // Front Left (child of knuckle)
    const flWheel = new THREE.Mesh(tireGeo, tireMat);
    flWheel.add(new THREE.Mesh(hubGeo, hubMat));
    this.frontLeftKnuckle.add(flWheel);
    this.wheelMeshes.push(flWheel);

    // Front Right (child of knuckle)
    const frWheel = new THREE.Mesh(tireGeo, tireMat);
    frWheel.add(new THREE.Mesh(hubGeo, hubMat));
    this.frontRightKnuckle.add(frWheel);
    this.wheelMeshes.push(frWheel);

    // Rear Left
    const rlWheel = new THREE.Mesh(tireGeo, tireMat);
    rlWheel.add(new THREE.Mesh(hubGeo, hubMat));
    rlWheel.position.set(-0.95, 0.38, -1.15);
    this.group.add(rlWheel);
    this.wheelMeshes.push(rlWheel);

    // Rear Right
    const rrWheel = new THREE.Mesh(tireGeo, tireMat);
    rrWheel.add(new THREE.Mesh(hubGeo, hubMat));
    rrWheel.position.set(0.95, 0.38, -1.15);
    this.group.add(rrWheel);
    this.wheelMeshes.push(rrWheel);

    return [headL, headR, antenna, exhaust];
  }

  /**
   * Driver Entry & Boarding Logic
   */
  public canEnter(playerPos: THREE.Vector3): boolean {
    if (this.isDriverSeated) return false;
    const dist = this.position.distanceTo(playerPos);
    return dist <= 2.8;
  }

  public enterVehicle(player: Player) {
    if (this.isDriverSeated) return;
    this.isDriverSeated = true;
    this.driverPlayer = player;

    // Swing door open then shut with click
    this.doorTargetAngle = 1.1; // ~63 degrees open
    sounds.playCarDoorOpen();

    setTimeout(() => {
      this.doorTargetAngle = 0;
      sounds.playCarDoorShut();
    }, 280);

    // Position player inside chassis
    player.isMoving = false;
    player.group.visible = false; // Player tucked in driver cabin
    player.position.copy(this.position);

    // Play initial engine rev
    sounds.playEngineRev();
  }

  public exitVehicle(player: Player) {
    if (!this.isDriverSeated) return;
    this.isDriverSeated = false;
    this.driverPlayer = null;

    // Open door and step out to the left
    this.doorTargetAngle = 1.1;
    sounds.playCarDoorOpen();

    const exitOffset = new THREE.Vector3(-1.6, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.group.rotation.y);
    player.position.copy(this.position).add(exitOffset);
    player.group.visible = true;

    setTimeout(() => {
      this.doorTargetAngle = 0;
      sounds.playCarDoorShut();
    }, 320);

    // Reset throttle
    this.speed = 0;
    this.targetSteerAngle = 0;
  }

  /**
   * Apply Driving Inputs:
   * gas: 0 to 1
   * brake: 0 to 1 (also reverse if stopped)
   * steer: -1 (left) to +1 (right)
   * handbrake: boolean (initiates drift slip)
   */
  public applyInput(gas: number, brake: number, steer: number, handbrake: boolean) {
    if (!this.isDriverSeated || this.durability <= 0 || this.fuel <= 0) {
      this.targetSteerAngle = 0;
      return;
    }

    this.targetSteerAngle = steer;
    this.isDrifting = handbrake && Math.abs(this.speed) > 4.0;

    const maxSpeed = this.def.topSpeed;
    const accel = this.def.acceleration;
    const brakePower = 14.0;
    const reverseMax = 6.0;

    if (gas > 0) {
      if (this.speed < maxSpeed) {
        this.speed += gas * accel * 0.016;
      }
    } else if (brake > 0) {
      if (this.speed > 0.2) {
        this.speed -= brake * brakePower * 0.016;
      } else {
        // Reverse
        if (this.speed > -reverseMax) {
          this.speed -= brake * (accel * 0.6) * 0.016;
        }
      }
    } else {
      // Natural rolling drag / deceleration
      const rollFriction = this.isDrifting ? 6.0 : 3.5;
      if (this.speed > 0) {
        this.speed = Math.max(0, this.speed - rollFriction * 0.016);
      } else if (this.speed < 0) {
        this.speed = Math.min(0, this.speed + rollFriction * 0.016);
      }
    }

    // Fuel consumption: liters per frame based on speed
    if (Math.abs(this.speed) > 0.1) {
      const fuelCost = (this.def.fuelConsumptionRate / 100) * Math.abs(this.speed) * 0.016;
      this.fuel = Math.max(0, this.fuel - fuelCost);
    }
  }

  /**
   * Car Horn Action
   */
  public honk(particles?: ParticleManager, camera?: THREE.Camera, onNoiseBroadcast?: (pos: THREE.Vector3, radius: number) => void) {
    if (this.def.hornType === 'sad_trombone') {
      sounds.playSadTrombone();
    } else {
      sounds.playHonk();
    }

    if (particles) {
      particles.spawnSoundWave(this.position);
    }
    if (camera) {
      ComicPops.show('HONK HONK!! 📢', this.position, camera);
    }

    // Broadcast 35m noise ring to alert zombies
    if (onNoiseBroadcast) {
      onNoiseBroadcast(this.position, this.def.engineNoiseRadius || 35);
    }
  }

  /**
   * Frame-by-frame physics & visual update
   */
  public update(
    dt: number,
    roadNetwork?: RoadNetwork,
    particles?: ParticleManager,
    camera?: THREE.Camera,
    onZombieRam?: (carPos: THREE.Vector3, damage: number, knockback: THREE.Vector3) => void,
    onObstacleCrash?: (carPos: THREE.Vector3) => void,
    isNight?: boolean
  ) {
    const clampedDt = Math.min(dt, 0.1);

    // 1. Surface Grip Calculation
    let surfaceGrip = 0.85;
    if (roadNetwork) {
      const roadGrip = roadNetwork.getGripAt(this.position.x, this.position.z);
      surfaceGrip = roadGrip < 0.9 ? roadGrip : 1.0;
    }

    // 2. Smooth Steering Knuckle Yaw (Speed-sensitive steering: sharper at low speeds)
    const speedRatio = Math.min(1.0, Math.abs(this.speed) / (this.def.topSpeed || 14));
    const maxSteerRad = (32 - speedRatio * 14) * (Math.PI / 180); // 32 deg down to 18 deg at high speed
    const targetRad = this.targetSteerAngle * maxSteerRad;

    // Smooth damp towards target steer
    const steerResponsiveness = 12 * this.def.handling;
    this.steeringAngle += (targetRad - this.steeringAngle) * Math.min(1.0, clampedDt * steerResponsiveness);

    this.frontLeftKnuckle.rotation.y = this.steeringAngle;
    this.frontRightKnuckle.rotation.y = this.steeringAngle;

    // 3. Forward Translation & Angular Turn Yaw
    if (Math.abs(this.speed) > 0.05) {
      // Turn rate: dTheta = (speed / wheelbase) * tan(steeringAngle)
      const wheelbase = 2.3;
      const turnMultiplier = this.isDrifting ? 1.65 : 1.0;
      const dTheta = (this.speed / wheelbase) * Math.tan(this.steeringAngle) * turnMultiplier * clampedDt;
      this.group.rotation.y += dTheta;

      // Forward movement vector
      const forward = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.group.rotation.y);
      this.velocity.copy(forward).multiplyScalar(this.speed);

      // Apply drift lateral slip
      if (this.isDrifting) {
        const lateral = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.group.rotation.y);
        const slipDir = this.steeringAngle > 0 ? -1 : 1;
        this.velocity.addScaledVector(lateral, slipDir * 3.5);
      }

      this.position.addScaledVector(this.velocity, clampedDt);

      // Keep seated player synced with car
      if (this.isDriverSeated && this.driverPlayer) {
        this.driverPlayer.position.copy(this.position);
      }
    } else {
      this.velocity.set(0, 0, 0);
    }

    // 4. Wheel Roll (ω = v / r)
    if (Math.abs(this.speed) > 0.01) {
      const angularVelocity = (this.speed / this.wheelRadius) * clampedDt;
      for (const wheel of this.wheelMeshes) {
        wheel.rotation.x += angularVelocity;
      }
    }

    // 5. Suspension Second-Order Spring Dynamics (Body Roll & Pitch)
    // Roll: vehicle tilts AWAY from turn (centrifugal roll)
    const turnG = (this.speed * this.speed * Math.sin(this.steeringAngle)) / 2.3;
    const targetRoll = -Math.min(0.22, Math.max(-0.22, turnG * 0.035));
    this.rollSpring.update(targetRoll, clampedDt);
    this.chassis.rotation.z = this.rollSpring.value;

    // Pitch: vehicle dives on braking/deceleration, squats on acceleration
    const accelRate = this.speed > 0 ? (this.speed > 10 ? 0.04 : 0.08) : -0.06;
    const targetPitch = Math.min(0.12, Math.max(-0.12, accelRate));
    this.pitchSpring.update(targetPitch, clampedDt);
    this.chassis.rotation.x = this.pitchSpring.value;

    // Harmonic Antenna Spring
    const targetAntennaX = -this.speed * 0.04;
    this.antennaSpring.update(targetAntennaX, clampedDt);
    this.antenna.rotation.x = this.antennaSpring.value;

    // 6. Hinged Door Tween
    this.doorCurrentAngle += (this.doorTargetAngle - this.doorCurrentAngle) * Math.min(1.0, clampedDt * 14);
    this.driverDoorPivot.rotation.y = -this.doorCurrentAngle;

    // 7. Headlight illumination based on day/night or boarding
    this.headlightsOn = isNight || this.isDriverSeated;
    for (const spot of this.headlights) {
      spot.visible = this.headlightsOn;
    }

    // 8. Visual Particle FX (Exhaust Put-Puffs & Drift Tire Smoke)
    if (particles && this.isDriverSeated) {
      // Exhaust puffs
      this.exhaustTimer += clampedDt;
      if (this.exhaustTimer >= (Math.abs(this.speed) > 1.0 ? 0.12 : 0.35)) {
        this.exhaustTimer = 0;
        const exhWorld = new THREE.Vector3();
        this.exhaustPipe.getWorldPosition(exhWorld);
        particles.spawnDustPuffs(exhWorld);
      }

      // Drift / tire screech smoke
      if (this.isDrifting && Math.abs(this.speed) > 4.0) {
        this.driftSmokeTimer += clampedDt;
        if (this.driftSmokeTimer >= 0.08) {
          this.driftSmokeTimer = 0;
          particles.spawnDustPuffs(this.position);
        }

        this.tireScreechTimer += clampedDt;
        if (this.tireScreechTimer >= 0.4) {
          this.tireScreechTimer = 0;
          sounds.playTireScreech();
        }
      }
    }

    // 9. Low Durability Smoke (<30% health -> engine smoking from hood)
    if (particles && this.durability < 30) {
      const hoodWorld = this.position.clone().add(new THREE.Vector3(0, 0.9, 1.2).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.group.rotation.y));
      particles.spawnSmoke(hoodWorld);
    }
  }

  /**
   * Handle Obstacle Impact
   */
  public handleCollisionWithObstacle(obstaclePos: THREE.Vector3, particles?: ParticleManager, camera?: THREE.Camera) {
    if (Math.abs(this.speed) < 1.0) return;

    // Bounce back
    const impactDir = new THREE.Vector3().subVectors(this.position, obstaclePos).normalize().setY(0);
    this.position.addScaledVector(impactDir, 0.35);

    const dmg = Math.round(Math.abs(this.speed) * 2.8);
    this.durability = Math.max(0, this.durability - dmg);

    // Reverse velocity
    this.speed = -this.speed * 0.35;

    // Audio & Visual Crash
    sounds.playCarCrash();
    if (particles) {
      particles.spawnHitSparks(this.position);
      particles.spawnDustPuffs(this.position);
    }
    if (camera) {
      ComicPops.clang(this.position, camera);
    }
  }

  /**
   * Handle Ramming Zombie
   */
  public handleRamZombie(zombiePos: THREE.Vector3, particles?: ParticleManager, camera?: THREE.Camera): number {
    const impactSpeed = Math.abs(this.speed);
    if (impactSpeed < 1.2) return 0;

    const damage = Math.round(impactSpeed * 18);
    const knockback = new THREE.Vector3().subVectors(zombiePos, this.position).normalize().multiplyScalar(impactSpeed * 1.5);

    // Slight deceleration on impact
    this.speed *= 0.88;

    if (particles) {
      particles.spawnGreenPuff(zombiePos);
    }
    if (camera) {
      ComicPops.show('BONK! 🚗', zombiePos, camera);
    }

    return damage;
  }

  public getPart(slot: string) {
    return this.parts.get(slot);
  }

  public repairPart(slot: string, amount: number = 40) {
    const p = this.parts.get(slot);
    if (p) {
      p.condition = Math.min(100, p.condition + amount);
      this.recalculateDurability();
    }
  }

  public recalculateDurability() {
    let total = 0;
    let count = 0;
    for (const p of this.parts.values()) {
      total += p.condition;
      count++;
    }
    this.durability = count > 0 ? Math.round(total / count) : 100;
  }

  public siphonFuel(amount: number = 5): number {
    const siphoned = Math.min(this.fuel, amount);
    this.fuel = Math.max(0, this.fuel - siphoned);
    return siphoned;
  }

  public addFuel(amount: number = 5): number {
    const added = Math.min(this.def.fuelCapacity - this.fuel, amount);
    this.fuel = Math.min(this.def.fuelCapacity, this.fuel + added);
    return added;
  }

  public refuel(liters: number) {
    this.addFuel(liters);
  }

  public repair(amount: number) {
    this.durability = Math.min(100, this.durability + amount);
    for (const p of this.parts.values()) {
      p.condition = Math.min(100, p.condition + amount);
    }
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.group);
  }
}
