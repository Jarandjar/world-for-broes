import * as THREE from 'three';
import { createNoise2D } from 'simplex-noise';
import { makeSeededRandom, seededValue, smoothstep, TerrainSampler } from './TerrainSampler';
import { TextureGenerator } from '../rendering/TextureGenerator';
import {
  PineTree,
  BerryBush,
  GrassTuft,
  Boulder,
  ItemDrop,
  ParticleManager,
  InteractiveEntity,
  AmbientBunny,
  AmbientEcosystem,
} from '../entities/WorldObjects';
import { RoadNetwork } from './RoadNetwork';
import { SettlementGenerator } from './SettlementGenerator';
import { BuildingManager } from './BuildingManager';
import { ContainerManager } from './ContainerManager';
import { CrowdZombieEngine } from '../entities/CrowdZombieEngine';
import { VehicleManager } from './VehicleManager';

export class WorldGenerator {
  public entities: InteractiveEntity[] = [];
  public itemDrops: ItemDrop[] = [];
  public bunnies: AmbientBunny[] = [];
  public ecosystem: AmbientEcosystem;
  public roadNetwork: RoadNetwork;
  public settlements: SettlementGenerator;
  public buildings: BuildingManager;
  public containers: ContainerManager;
  public zombies: CrowdZombieEngine;
  public vehicles: VehicleManager;
  private scene: THREE.Scene;
  private particles: ParticleManager;
  private noise2D: ReturnType<typeof createNoise2D>;
  private terrainSampler: TerrainSampler;
  private seed: number;

  constructor(scene: THREE.Scene, particles: ParticleManager, seed: number = 4242) {
    this.scene = scene;
    this.particles = particles;
    this.seed = seed;
    this.terrainSampler = new TerrainSampler(seed);
    this.noise2D = createNoise2D(makeSeededRandom(seed ^ 0x6c8e9cf5));
    this.ecosystem = new AmbientEcosystem(scene);
    this.roadNetwork = new RoadNetwork(scene);
    this.settlements = new SettlementGenerator(scene);
    this.buildings = new BuildingManager(this.settlements, scene);
    this.containers = new ContainerManager(scene);
    this.containers.populateFromSettlementLots(this.settlements.getAllLots());
    this.zombies = new CrowdZombieEngine({ scene, lootManager: this.containers.lootManager });
    this.zombies.populateHorde();
    this.vehicles = new VehicleManager(scene);
    this.vehicles.populateVehicles(this.settlements.getAllLots());
    this.createGround();
    this.populateWorld();
  }

  private createGround() {
    const groundGeo = new THREE.PlaneGeometry(160, 160, 80, 80);
    groundGeo.rotateX(-Math.PI / 2);
    const groundPositions = groundGeo.getAttribute('position');
    for (let i = 0; i < groundPositions.count; i++) {
      const x = groundPositions.getX(i);
      const z = -groundPositions.getY(i);
      groundPositions.setY(i, this.getTerrainHeight(x, z));
    }
    groundPositions.needsUpdate = true;
    groundGeo.computeVertexNormals();
    const groundMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getGroundTexture(),
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.receiveShadow = true;
    ground.position.y = 0;
    this.scene.add(ground);

    const waterGeo = new THREE.PlaneGeometry(160, 160);
    waterGeo.rotateX(-Math.PI / 2);
    const water = new THREE.Mesh(waterGeo, new THREE.MeshBasicMaterial({
      color: 0x2b7890,
      transparent: true,
      opacity: 0.84,
      depthWrite: false,
    }));
    water.position.y = -0.62;
    water.renderOrder = 1;
    this.scene.add(water);

    // Scatter ground clutter decals (pebbles, dry pine needles, wilderness clovers)
    const clutterGeo = new THREE.PlaneGeometry(1.4, 1.4);
    clutterGeo.rotateX(-Math.PI / 2);
    const clutterMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getGroundClutterTexture(),
      transparent: true,
      alphaTest: 0.15,
      depthWrite: false,
    });

    const clutterCount = 90;
    for (let i = 0; i < clutterCount; i++) {
      const decal = new THREE.Mesh(clutterGeo, clutterMat);
      const angle = seededValue(this.seed, i, 0, 1) * Math.PI * 2;
      const dist = 3.0 + seededValue(this.seed, i, 0, 2) * 55;
      const x = Math.cos(angle) * dist, z = Math.sin(angle) * dist;
      decal.position.set(x, this.getTerrainHeight(x, z) + 0.015, z);
      decal.rotation.y = seededValue(this.seed, i, 0, 3) * Math.PI * 2;
      const s = 0.8 + seededValue(this.seed, i, 0, 4) * 0.5;
      decal.scale.set(s, s, s);
      this.scene.add(decal);
    }
  }

  public getObstacles(): { position: THREE.Vector3; radius: number }[] {
    const list: { position: THREE.Vector3; radius: number }[] = [];
    for (const ent of this.entities) {
      if (ent.collisionRadius && ent.collisionRadius > 0 && ent.isAlive()) {
        list.push({
          position: ent.position,
          radius: ent.collisionRadius,
        });
      }
    }
    return list;
  }

  private populateWorld() {
    const worldRadius = 60;
    const step = 4.0;

    // Guaranteed Starter pack near player (radius 3 to 10)
    // 3 Trees
    this.spawnPineTree(new THREE.Vector3(4, 0, 3));
    this.spawnPineTree(new THREE.Vector3(-5, 0, 4));
    this.spawnPineTree(new THREE.Vector3(2, 0, -6));

    // 4 Grass tufts
    this.spawnGrassTuft(new THREE.Vector3(-3, 0, -2));
    this.spawnGrassTuft(new THREE.Vector3(3, 0, -2));
    this.spawnGrassTuft(new THREE.Vector3(-2, 0, 3));
    this.spawnGrassTuft(new THREE.Vector3(1, 0, 4));

    // 2 Berry bushes
    this.spawnBerryBush(new THREE.Vector3(-5, 0, -4));
    this.spawnBerryBush(new THREE.Vector3(5, 0, -5));

    // 2 Boulders with Flint
    this.spawnBoulder(new THREE.Vector3(6, 0, 2));
    this.spawnBoulder(new THREE.Vector3(-6, 0, 0));

    // Starter floor item drops (Twigs & Flint)
    this.spawnItemDrop('twigs', new THREE.Vector3(1.5, 0, 1.5));
    this.spawnItemDrop('twigs', new THREE.Vector3(-1.5, 0, -1.5));
    this.spawnItemDrop('flint', new THREE.Vector3(2.5, 0, -1.0));

    // Starter harmless cute bunnies
    this.spawnBunny(new THREE.Vector3(-4.5, 0, -3.5));
    this.spawnBunny(new THREE.Vector3(4.5, 0, -4.5));
    this.spawnBunny(new THREE.Vector3(1.0, 0, 5.0));

    // Procedural terrain generation across world
    for (let x = -worldRadius; x <= worldRadius; x += step) {
      for (let z = -worldRadius; z <= worldRadius; z += step) {
        const distFromCenter = Math.hypot(x, z);
        if (distFromCenter < 9 || distFromCenter > worldRadius) continue;

        // Jitter position
        const jx = x + (seededValue(this.seed, x, z, 1) - 0.5) * 2.5;
        const jz = z + (seededValue(this.seed, x, z, 2) - 0.5) * 2.5;

        // Skip natural obstacle spawning along active road corridors & inside building lots
        if (this.roadNetwork.isOnRoad(jx, jz, 1.8)) continue;
        if (this.settlements.isInsideBuildingLot(jx, jz, 2.0)) continue;

        const pos = new THREE.Vector3(jx, this.getTerrainHeight(jx, jz), jz);
        const isInSettlementCore = this.settlements.isInSettlementClearance(jx, jz, -4.0);

        const forestNoise = this.noise2D(jx * 0.04, jz * 0.04);
        const resourceNoise = this.noise2D(jx * 0.09 + 100, jz * 0.09 + 100);

        if (forestNoise > 0.25) {
          // Forest Biome: Pine Trees & Twigs (cleared inside settlement core)
          if (!isInSettlementCore && seededValue(this.seed, x, z, 3) < 0.65) {
            this.spawnPineTree(pos);
          } else if (seededValue(this.seed, x, z, 4) < 0.35) {
            this.spawnItemDrop('twigs', pos);
          }
        } else if (forestNoise < -0.2) {
          // Rocky Biome: Boulders & Flint (cleared inside settlement core)
          if (!isInSettlementCore && seededValue(this.seed, x, z, 5) < 0.45) {
            this.spawnBoulder(pos);
          } else if (seededValue(this.seed, x, z, 6) < 0.3) {
            this.spawnItemDrop('flint', pos);
          }
        } else {
          // Grassland Biome: Savannah Grass & Berries & Bunnies
          if (resourceNoise > 0.3) {
            this.spawnBerryBush(pos);
            if (seededValue(this.seed, x, z, 7) < 0.35) {
              const bunnyX = pos.x + 1.2, bunnyZ = pos.z + 1.2;
              this.spawnBunny(new THREE.Vector3(bunnyX, this.getTerrainHeight(bunnyX, bunnyZ), bunnyZ));
            }
          } else if (resourceNoise > -0.2) {
            this.spawnGrassTuft(pos);
          }
        }
      }
    }
  }

  public spawnBunny(pos: THREE.Vector3) {
    const bunny = new AmbientBunny(this.scene, this.groundedPosition(pos));
    this.bunnies.push(bunny);
    this.entities.push(bunny);
  }

  public spawnPineTree(pos: THREE.Vector3) {
    const groundPos = this.groundedPosition(pos);
    const tree = new PineTree(this.scene, groundPos, this.particles, p => {
      // Tree felled: drop 2-3 logs and 1 twig
      this.spawnItemDrop('log', new THREE.Vector3(p.x + 0.8, 0, p.z));
      this.spawnItemDrop('log', new THREE.Vector3(p.x - 0.8, 0, p.z + 0.5));
      this.spawnItemDrop('twigs', new THREE.Vector3(p.x, 0, p.z - 0.8));
    });
    this.entities.push(tree);
  }

  public spawnBerryBush(pos: THREE.Vector3) {
    const bush = new BerryBush(this.scene, this.groundedPosition(pos), p => {
      this.spawnItemDrop('berries', new THREE.Vector3(p.x + 0.4, 0, p.z + 0.4));
      this.spawnItemDrop('berries', new THREE.Vector3(p.x - 0.4, 0, p.z - 0.4));
    });
    this.entities.push(bush);
  }

  public spawnGrassTuft(pos: THREE.Vector3) {
    const tuft = new GrassTuft(this.scene, this.groundedPosition(pos), p => {
      this.spawnItemDrop('grass', p.clone());
    });
    this.entities.push(tuft);
  }

  public spawnBoulder(pos: THREE.Vector3) {
    const boulder = new Boulder(this.scene, this.groundedPosition(pos), this.particles, p => {
      this.spawnItemDrop('flint', new THREE.Vector3(p.x + 0.5, 0, p.z));
      this.spawnItemDrop('flint', new THREE.Vector3(p.x - 0.5, 0, p.z));
    });
    this.entities.push(boulder);
  }

  public spawnItemDrop(itemId: string, pos: THREE.Vector3) {
    const drop = new ItemDrop(this.scene, itemId, this.groundedPosition(pos), this.particles);
    this.itemDrops.push(drop);
    this.entities.push(drop);
  }

  public getTerrainHeight(x: number, z: number): number {
    let height = this.terrainSampler.sampleHeight(x, z);
    const spawnBlend = smoothstep(9, 20, Math.hypot(x, z));
    height *= spawnBlend;

    let settlementBlend = 1;
    for (const settlement of this.settlements.settlements.values()) {
      const distance = Math.hypot(x - settlement.center.x, z - settlement.center.z);
      settlementBlend = Math.min(settlementBlend, smoothstep(settlement.clearanceRadius, settlement.clearanceRadius + 9, distance));
    }
    height *= settlementBlend;

    const roadDistance = this.roadNetwork.getDistanceToNearestRoad(x, z);
    height *= smoothstep(0, 7, roadDistance);
    return height;
  }

  private groundedPosition(position: THREE.Vector3): THREE.Vector3 {
    const grounded = position.clone();
    grounded.y = this.getTerrainHeight(grounded.x, grounded.z);
    return grounded;
  }

  public update(
    delta: number,
    camera: THREE.Camera,
    playerPos: THREE.Vector3,
    inventory: any,
    isNight: boolean = false,
    isPlayerInLight: boolean = true,
    onPlayerDamage?: (damage: number) => void
  ) {
    // Update all entities
    this.entities.forEach(e => {
      if (e instanceof AmbientBunny) {
        e.update(delta, camera, playerPos);
      } else {
        e.update(delta, camera);
      }
    });

    // Update ambient ecosystem (fireflies, wind leaves, creeping darkness eyes)
    this.ecosystem.update(delta, playerPos, isNight, isPlayerInLight);

    // Update road network billboard props orientation
    this.roadNetwork.update(delta, camera);

    // Update building roof fold tweens, interior visibility, and hiding spots
    this.buildings.update(delta, playerPos);

    // Update containers, spring lids, loot arcs, and jackpot effects
    this.containers.update(delta, playerPos, inventory, this.particles, camera);

    // Update crowd zombie engine (3-tier LOD, AI states, gait desync)
    this.zombies.update(
      delta,
      playerPos,
      this.buildings.isPlayerHiding(),
      isNight,
      onPlayerDamage
    );

    // Update arcade vehicles, physics, collisions, and zombie ramming
    this.vehicles.update(
      delta,
      this.roadNetwork,
      this.particles,
      camera,
      this.zombies,
      this.getObstacles(),
      isNight
    );

    // Handle item drop magnets and cleanup
    for (let i = this.itemDrops.length - 1; i >= 0; i--) {
      const drop = this.itemDrops[i];
      drop.magnetTo(playerPos, delta, inventory);

      if (!drop.isAlive()) {
        this.scene.remove(drop.mesh);
        this.itemDrops.splice(i, 1);
        const entityIdx = this.entities.indexOf(drop);
        if (entityIdx !== -1) this.entities.splice(entityIdx, 1);
      }
    }
  }
}
