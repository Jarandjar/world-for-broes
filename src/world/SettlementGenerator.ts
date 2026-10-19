import * as THREE from 'three';
import {
  SettlementInstance,
  BuildingLot,
  BuildingZone,
  SettlementLayoutType,
  SettlementCondition,
  SETTLEMENT_SPACING_RULES,
} from './SettlementDef';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { gameConfig } from '../config/GameConfig';

/**
 * Procedural Settlement & Town Layout Engine (Phase Z2).
 * Implements:
 * - 4 core procedural layout algorithms (Grid, Strip, Radial, Cluster) + Fortified Ring-Fence Outposts.
 * - Poisson-disc spacing and clearance enforcement across biomes.
 * - Non-overlapping oriented building lot subdivision with zone typing.
 * - Spatial queries (isInSettlementClearance, isInsideBuildingLot, getLotAt) to clear trees/rocks.
 * - Themed 3D ground pads, foundation slabs, curbs, and town entrance signboards.
 * - Classic mode preservation (cleanly hidden when mode === 'classic').
 */
export class SettlementGenerator {
  public settlements: Map<string, SettlementInstance> = new Map();
  public group: THREE.Group;
  private seed: number;

  constructor(scene?: THREE.Scene, seed: number = 4242) {
    this.seed = seed;
    this.group = new THREE.Group();
    this.group.name = 'SettlementNetwork';

    this.initSettlements();

    if (scene) {
      this.buildVisualMeshes(scene);
      scene.add(this.group);
      this.setVisible(gameConfig.isZombieMode());

      // Auto-toggle on mode switch
      gameConfig.subscribe((cfg) => {
        this.setVisible(cfg.mode === 'zombie');
      });
    }
  }

  public setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  /**
   * Initializes all settlements across the 10 biomes, building lots according to layout rules.
   */
  private initSettlements() {
    // 1. Bro-ville Township (Meadow, Grid Layout)
    const town = this.buildTownSettlement({
      id: 'town_bro_ville',
      name: 'Bro-ville Township',
      poiId: 'town',
      category: 'town',
      layoutType: 'grid',
      biome: 'meadow',
      center: { x: 0, z: 0 },
      clearanceRadius: 52,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.town,
      condition: 'worn',
      gimmick: 'Town hall alarm siren rings across the entire valley',
      zombieDensity: 'high',
      lootLevel: 'jackpot',
      connectedRoadNodeId: 'meadow_spawn',
      streetName: 'MAIN ST & BRO BROADWAY',
    });
    this.settlements.set(town.id, town);

    // 2. Desert Oasis Strip Stop (Desert Highway, Strip Layout)
    const stripStop = this.buildStripStopSettlement({
      id: 'strip_desert_oasis',
      name: 'Desert Oasis Strip Stop',
      poiId: 'strip_stop',
      category: 'strip_stop',
      layoutType: 'strip',
      biome: 'desert',
      center: { x: 42, z: -18 },
      clearanceRadius: 30,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.strip_stop,
      condition: 'pristine',
      gimmick: 'Roadside lifeline: 24/7 diner, petro pumps, and cozy motel beds',
      zombieDensity: 'medium',
      lootLevel: 'high',
      connectedRoadNodeId: 'desert_junction',
      streetName: 'ROUTE 66 ROADWAY',
    });
    this.settlements.set(stripStop.id, stripStop);

    // 3. Old McCready Farmstead (Meadow/Forest, Cluster Layout)
    const farmstead = this.buildFarmsteadSettlement({
      id: 'farm_mccready',
      name: 'Old McCready Farmstead',
      poiId: 'farmhouse',
      category: 'farmstead',
      layoutType: 'cluster',
      biome: 'meadow',
      center: { x: -22, z: -26 },
      clearanceRadius: 28,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.farmstead,
      condition: 'worn',
      gimmick: 'Abundant fresh crops, tool sheds, and a reliable lawn tractor',
      zombieDensity: 'low',
      lootLevel: 'medium',
      connectedRoadNodeId: 'meadow_farm',
    });
    this.settlements.set(farmstead.id, farmstead);

    // 4. Scrapyard Rust City (Gadget / Scrapyard, Industrial Cluster Layout)
    const junkyard = this.buildJunkyardSettlement({
      id: 'junkyard_rust_city',
      name: 'Scrapyard Rust City',
      poiId: 'junkyard_town',
      category: 'junkyard_town',
      layoutType: 'cluster',
      biome: 'gadget',
      center: { x: -35, z: 18 },
      clearanceRadius: 34,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.junkyard_town,
      condition: 'barricaded',
      gimmick: 'Rusted container homes, conveyor belts, car lift, and spare engine parts',
      zombieDensity: 'medium',
      lootLevel: 'jackpot',
      connectedRoadNodeId: 'scrapyard_garage',
    });
    this.settlements.set(junkyard.id, junkyard);

    // 5. Pine River Logging Camp (Forest, Cluster Layout)
    const loggingCamp = this.buildLoggingCampSettlement({
      id: 'logging_pine_river',
      name: 'Pine River Logging Camp',
      poiId: 'logging_camp',
      category: 'logging_camp',
      layoutType: 'cluster',
      biome: 'forest',
      center: { x: -42, z: -45 },
      clearanceRadius: 26,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.logging_camp,
      condition: 'overgrown',
      gimmick: 'Sawmill, timber stacks, and the coveted chainsaw of regret',
      zombieDensity: 'low',
      lootLevel: 'medium',
      connectedRoadNodeId: 'thicket_camp',
    });
    this.settlements.set(loggingCamp.id, loggingCamp);

    // 6. Splash Cove Evac Docks (Beach, Shoreline Cluster Layout)
    const harbor = this.buildHarborSettlement({
      id: 'harbor_splash_cove',
      name: 'Splash Cove Evacuation Dock',
      poiId: 'evac_dock',
      category: 'harbor_town',
      layoutType: 'cluster',
      biome: 'beach',
      center: { x: -16, z: 58 },
      clearanceRadius: 30,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.harbor_town,
      condition: 'barricaded',
      gimmick: 'Evac rescue dock: power the generator, load the boat, and escape!',
      zombieDensity: 'high',
      lootLevel: 'jackpot',
      connectedRoadNodeId: 'cove_dock',
    });
    this.settlements.set(harbor.id, harbor);

    // 7. Squelchy Bog Boardwalk Village (Swamp, Radial Stilt Layout)
    const stiltVillage = this.buildStiltVillageSettlement({
      id: 'village_squelchy_bog',
      name: 'Squelchy Bog Boardwalk Village',
      poiId: 'stilt_village',
      category: 'stilt_village',
      layoutType: 'radial',
      biome: 'swamp',
      center: { x: -32, z: -10 },
      clearanceRadius: 26,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.stilt_village,
      condition: 'flooded',
      gimmick: 'Boardwalk pathways above toxic mud; crawling zombies emerge from muck',
      zombieDensity: 'medium',
      lootLevel: 'medium',
      connectedRoadNodeId: 'bog_causeway',
    });
    this.settlements.set(stiltVillage.id, stiltVillage);

    // 8. Brrington Weather Station Outpost (Tundra, Ring-Fence Outpost Layout)
    const tundraOutpost = this.buildOutpostSettlement({
      id: 'outpost_brrington',
      name: 'Brrington Weather Station',
      poiId: 'radio_tower',
      category: 'outpost',
      layoutType: 'ring_fence',
      biome: 'tundra',
      center: { x: 48, z: 28 },
      clearanceRadius: 22,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.outpost,
      condition: 'worn',
      gimmick: 'Radar dome, storm warnings, and warm shelter from blizzard cold',
      zombieDensity: 'low',
      lootLevel: 'high',
      connectedRoadNodeId: 'tundra_outpost',
    });
    this.settlements.set(tundraOutpost.id, tundraOutpost);

    // 9. Rockbottom Ridge Pass Mining Camp (Mountain, Cluster Layout)
    const miningCamp = this.buildMiningCampSettlement({
      id: 'mining_rockbottom',
      name: 'Rockbottom Ridge Mining Camp',
      poiId: 'mining_camp',
      category: 'mining_camp',
      layoutType: 'cluster',
      biome: 'mountain',
      center: { x: 28, z: 44 },
      clearanceRadius: 24,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.mining_camp,
      condition: 'collapsed',
      gimmick: 'Mine shaft headframe, ore carts, and tool caches',
      zombieDensity: 'medium',
      lootLevel: 'high',
      connectedRoadNodeId: 'ridge_switchback',
    });
    this.settlements.set(miningCamp.id, miningCamp);

    // 10. Haunted Hospital Ruins (Hollow, Radial Layout)
    const ghostVillage = this.buildGhostVillageSettlement({
      id: 'ghost_hollow_ruins',
      name: 'Haunted Hospital Ruins',
      poiId: 'hospital_ruins',
      category: 'village',
      layoutType: 'radial',
      biome: 'hollow',
      center: { x: -46, z: 40 },
      clearanceRadius: 28,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.village,
      condition: 'burnt',
      gimmick: 'Sanity-draining hospital corridors, emergency triage, and antidote gel',
      zombieDensity: 'high',
      lootLevel: 'high',
      connectedRoadNodeId: 'hollow_crossroads',
    });
    this.settlements.set(ghostVillage.id, ghostVillage);

    // 11. Mount Grumbleguts Vault Outpost (Volcano, Ring-Fence Outpost Layout)
    const volcanoVault = this.buildOutpostSettlement({
      id: 'outpost_grumbleguts',
      name: 'Mount Grumbleguts Vault Outpost',
      poiId: 'volcano_scrap_vault',
      category: 'outpost',
      layoutType: 'ring_fence',
      biome: 'volcano',
      center: { x: 38, z: -48 },
      clearanceRadius: 24,
      minSeparationDistance: SETTLEMENT_SPACING_RULES.outpost,
      condition: 'barricaded',
      gimmick: 'High-security reinforced blast safe and geothermal generator',
      zombieDensity: 'boss',
      lootLevel: 'jackpot',
      connectedRoadNodeId: 'volcano_bridge',
    });
    this.settlements.set(volcanoVault.id, volcanoVault);
  }

  // --- PROCEDURAL LAYOUT BUILDERS ---

  /**
   * 1. GRID LAYOUT: Towns with street blocks, civic center, commercial rows, residential quarters.
   */
  private buildTownSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    // Block 1 (North-West): Civic & Medical
    lots.push({
      id: `${spec.id}_town_hall`,
      buildingType: 'town_hall',
      zone: 'civic',
      position: { x: cx - 14, z: cz - 14 },
      size: { width: 14, depth: 12 },
      rotationY: 0,
      doorPosition: { x: cx - 14, z: cz - 8 },
      condition: 'worn',
      containers: ['cash_register', 'vault_safe', 'locker'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 2,
    });
    lots.push({
      id: `${spec.id}_clinic`,
      buildingType: 'clinic',
      zone: 'civic',
      position: { x: cx - 14, z: cz + 14 },
      size: { width: 10, depth: 10 },
      rotationY: 0,
      doorPosition: { x: cx - 14, z: cz + 9 },
      condition: 'pristine',
      containers: ['medicine_cabinet', 'fridge'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    // Block 2 (North-East): Commercial Supermarket & Police
    lots.push({
      id: `${spec.id}_supermarket`,
      buildingType: 'supermarket',
      zone: 'commercial',
      position: { x: cx + 16, z: cz - 14 },
      size: { width: 15, depth: 12 },
      rotationY: 0,
      doorPosition: { x: cx + 16, z: cz - 8 },
      condition: 'worn',
      containers: ['fridge', 'cash_register', 'toolbox'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_police_station`,
      buildingType: 'police_station',
      zone: 'civic',
      position: { x: cx + 16, z: cz + 14 },
      size: { width: 11, depth: 10 },
      rotationY: 0,
      doorPosition: { x: cx + 16, z: cz + 9 },
      condition: 'barricaded',
      containers: ['locker', 'cash_register'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    // Block 3 (West): Residential Cabins & Small Houses
    lots.push({
      id: `${spec.id}_house_west_1`,
      buildingType: 'house_small',
      zone: 'residential',
      position: { x: cx - 32, z: cz - 12 },
      size: { width: 8, depth: 8 },
      rotationY: Math.PI / 2,
      doorPosition: { x: cx - 28, z: cz - 12 },
      condition: 'worn',
      containers: ['fridge', 'suitcase'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_house_west_2`,
      buildingType: 'house_two_story',
      zone: 'residential',
      position: { x: cx - 32, z: cz + 12 },
      size: { width: 9, depth: 9 },
      rotationY: Math.PI / 2,
      doorPosition: { x: cx - 27.5, z: cz + 12 },
      condition: 'worn',
      containers: ['medicine_cabinet', 'suitcase'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 2,
    });

    // Block 4 (East): Water Tower Landmark & Utility Shed
    lots.push({
      id: `${spec.id}_water_tower`,
      buildingType: 'water_tower',
      zone: 'outskirts',
      position: { x: cx + 34, z: cz - 14 },
      size: { width: 7, depth: 7 },
      rotationY: 0,
      doorPosition: { x: cx + 34, z: cz - 10.5 },
      condition: 'worn',
      containers: ['toolbox'],
      isSafeHouseCandidate: false,
      hasRoofFold: false,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_gas_station`,
      buildingType: 'gas_station',
      zone: 'commercial',
      position: { x: cx + 34, z: cz + 14 },
      size: { width: 11, depth: 9 },
      rotationY: -Math.PI / 2,
      doorPosition: { x: cx + 29.5, z: cz + 14 },
      condition: 'worn',
      containers: ['cash_register', 'toolbox', 'fridge'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'town',
      layoutType: 'grid',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 52,
      minSeparationDistance: spec.minSeparationDistance || 300,
      lots,
      condition: spec.condition || 'worn',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'high',
      lootLevel: spec.lootLevel || 'jackpot',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'meadow_spawn',
      streetName: spec.streetName,
    };
  }

  /**
   * 2. STRIP LAYOUT: Highway roadside stop aligned parallel to the road.
   */
  private buildStripStopSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    // Linear alignment along road corridor (west to east)
    // 1. Petro Gas Station & Canopy
    lots.push({
      id: `${spec.id}_gas_station`,
      buildingType: 'gas_station',
      zone: 'commercial',
      position: { x: cx - 16, z: cz - 5 },
      size: { width: 12, depth: 8 },
      rotationY: 0,
      doorPosition: { x: cx - 16, z: cz - 1 },
      condition: 'pristine',
      containers: ['cash_register', 'toolbox', 'fridge'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    // 2. Stumble-Inn 24/7 Roadside Diner
    lots.push({
      id: `${spec.id}_diner`,
      buildingType: 'diner',
      zone: 'commercial',
      position: { x: cx, z: cz - 6 },
      size: { width: 11, depth: 8 },
      rotationY: 0,
      doorPosition: { x: cx, z: cz - 2 },
      condition: 'worn',
      containers: ['fridge', 'cash_register'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    // 3. No-Vacancy Roadside Motel (with safe beds)
    lots.push({
      id: `${spec.id}_motel`,
      buildingType: 'motel',
      zone: 'commercial',
      position: { x: cx + 16, z: cz - 6 },
      size: { width: 14, depth: 9 },
      rotationY: 0,
      doorPosition: { x: cx + 16, z: cz - 1.5 },
      condition: 'worn',
      containers: ['suitcase', 'medicine_cabinet'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'strip_stop',
      layoutType: 'strip',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 30,
      minSeparationDistance: spec.minSeparationDistance || 45,
      lots,
      condition: spec.condition || 'pristine',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'medium',
      lootLevel: spec.lootLevel || 'high',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'desert_junction',
      streetName: spec.streetName,
    };
  }

  /**
   * 3. CLUSTER LAYOUT: Farmsteads, logging camps, junkyards. Organic clustering around main building.
   */
  private buildFarmsteadSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    // Farmhouse main residence
    lots.push({
      id: `${spec.id}_farmhouse`,
      buildingType: 'farmhouse',
      zone: 'residential',
      position: { x: cx - 6, z: cz - 6 },
      size: { width: 10, depth: 10 },
      rotationY: 0.15,
      doorPosition: { x: cx - 6, z: cz - 1 },
      condition: 'worn',
      containers: ['fridge', 'toolbox', 'suitcase'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 2,
    });

    // Big Timber Barn
    lots.push({
      id: `${spec.id}_barn`,
      buildingType: 'barn',
      zone: 'industrial',
      position: { x: cx + 11, z: cz - 5 },
      size: { width: 14, depth: 10 },
      rotationY: -0.1,
      doorPosition: { x: cx + 11, z: cz },
      condition: 'worn',
      containers: ['toolbox', 'fridge'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    // Grain Silo Tower
    lots.push({
      id: `${spec.id}_silo`,
      buildingType: 'silo',
      zone: 'industrial',
      position: { x: cx + 18, z: cz + 7 },
      size: { width: 6, depth: 6 },
      rotationY: 0,
      doorPosition: { x: cx + 18, z: cz + 4 },
      condition: 'worn',
      containers: ['fridge'],
      isSafeHouseCandidate: false,
      hasRoofFold: false,
      floorCount: 1,
    });

    // Tractor & Equipment Shed
    lots.push({
      id: `${spec.id}_shed`,
      buildingType: 'shed',
      zone: 'industrial',
      position: { x: cx - 7, z: cz + 10 },
      size: { width: 8, depth: 7 },
      rotationY: Math.PI / 4,
      doorPosition: { x: cx - 7, z: cz + 6.5 },
      condition: 'worn',
      containers: ['toolbox'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'farmstead',
      layoutType: 'cluster',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 28,
      minSeparationDistance: spec.minSeparationDistance || 45,
      lots,
      condition: spec.condition || 'worn',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'low',
      lootLevel: spec.lootLevel || 'medium',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'meadow_farm',
    };
  }

  private buildJunkyardSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    // Auto garage workshop with car lift
    lots.push({
      id: `${spec.id}_garage`,
      buildingType: 'auto_garage',
      zone: 'industrial',
      position: { x: cx - 7, z: cz - 7 },
      size: { width: 14, depth: 11 },
      rotationY: 0,
      doorPosition: { x: cx - 7, z: cz - 1.5 },
      condition: 'barricaded',
      containers: ['toolbox', 'car_trunk', 'glove_box'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    // Shipping Container Home
    lots.push({
      id: `${spec.id}_container_home`,
      buildingType: 'container_fort',
      zone: 'residential',
      position: { x: cx + 12, z: cz - 6 },
      size: { width: 10, depth: 6 },
      rotationY: Math.PI / 2,
      doorPosition: { x: cx + 9, z: cz - 6 },
      condition: 'worn',
      containers: ['toolbox', 'fridge'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    // Scrap sorting warehouse
    lots.push({
      id: `${spec.id}_scrap_shed`,
      buildingType: 'shed',
      zone: 'industrial',
      position: { x: cx + 7, z: cz + 10 },
      size: { width: 10, depth: 8 },
      rotationY: -0.2,
      doorPosition: { x: cx + 7, z: cz + 6 },
      condition: 'worn',
      containers: ['vault_safe', 'toolbox'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'junkyard_town',
      layoutType: 'cluster',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 34,
      minSeparationDistance: spec.minSeparationDistance || 70,
      lots,
      condition: spec.condition || 'barricaded',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'medium',
      lootLevel: spec.lootLevel || 'jackpot',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'scrapyard_garage',
    };
  }

  private buildLoggingCampSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    lots.push({
      id: `${spec.id}_sawmill`,
      buildingType: 'warehouse',
      zone: 'industrial',
      position: { x: cx - 6, z: cz - 5 },
      size: { width: 12, depth: 9 },
      rotationY: 0,
      doorPosition: { x: cx - 6, z: cz - 0.5 },
      condition: 'overgrown',
      containers: ['toolbox', 'chainsaw_cache'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_bunkhouse`,
      buildingType: 'cabin',
      zone: 'residential',
      position: { x: cx + 8, z: cz - 4 },
      size: { width: 9, depth: 8 },
      rotationY: -0.15,
      doorPosition: { x: cx + 8, z: cz },
      condition: 'overgrown',
      containers: ['suitcase', 'fridge'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'logging_camp',
      layoutType: 'cluster',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 26,
      minSeparationDistance: spec.minSeparationDistance || 45,
      lots,
      condition: spec.condition || 'overgrown',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'low',
      lootLevel: spec.lootLevel || 'medium',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'thicket_camp',
    };
  }

  private buildHarborSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    lots.push({
      id: `${spec.id}_pier_shack`,
      buildingType: 'shop',
      zone: 'commercial',
      position: { x: cx - 8, z: cz - 4 },
      size: { width: 10, depth: 8 },
      rotationY: 0,
      doorPosition: { x: cx - 8, z: cz },
      condition: 'barricaded',
      containers: ['supply_drop_crate', 'toolbox'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_boat_house`,
      buildingType: 'warehouse',
      zone: 'industrial',
      position: { x: cx + 9, z: cz - 4 },
      size: { width: 12, depth: 9 },
      rotationY: 0,
      doorPosition: { x: cx + 9, z: cz + 0.5 },
      condition: 'barricaded',
      containers: ['toolbox', 'fridge'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'harbor_town',
      layoutType: 'cluster',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 30,
      minSeparationDistance: spec.minSeparationDistance || 60,
      lots,
      condition: spec.condition || 'barricaded',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'high',
      lootLevel: spec.lootLevel || 'jackpot',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'cove_dock',
    };
  }

  /**
   * 4. RADIAL LAYOUT: Stilt villages & ghost settlements centered around central well/square.
   */
  private buildStiltVillageSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;
    const angles = [0, (Math.PI * 2) / 3, (Math.PI * 4) / 3];

    for (let i = 0; i < angles.length; i++) {
      const a = angles[i];
      const dist = 11;
      const lx = cx + Math.cos(a) * dist;
      const lz = cz + Math.sin(a) * dist;
      lots.push({
        id: `${spec.id}_stilt_hut_${i + 1}`,
        buildingType: 'cabin',
        zone: 'residential',
        position: { x: lx, z: lz },
        size: { width: 7, depth: 7 },
        rotationY: a + Math.PI,
        doorPosition: { x: cx + Math.cos(a) * (dist - 3.5), z: cz + Math.sin(a) * (dist - 3.5) },
        condition: 'flooded',
        containers: ['fridge', 'toolbox'],
        isSafeHouseCandidate: true,
        hasRoofFold: true,
        floorCount: 1,
      });
    }

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'stilt_village',
      layoutType: 'radial',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 26,
      minSeparationDistance: spec.minSeparationDistance || 55,
      lots,
      condition: spec.condition || 'flooded',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'medium',
      lootLevel: spec.lootLevel || 'medium',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'bog_causeway',
    };
  }

  private buildMiningCampSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    lots.push({
      id: `${spec.id}_bunkhouse`,
      buildingType: 'cabin',
      zone: 'residential',
      position: { x: cx - 6, z: cz - 4 },
      size: { width: 9, depth: 8 },
      rotationY: 0,
      doorPosition: { x: cx - 6, z: cz },
      condition: 'collapsed',
      containers: ['toolbox', 'locker'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_tool_shed`,
      buildingType: 'shed',
      zone: 'industrial',
      position: { x: cx + 7, z: cz - 3 },
      size: { width: 8, depth: 7 },
      rotationY: -0.2,
      doorPosition: { x: cx + 7, z: cz + 0.5 },
      condition: 'collapsed',
      containers: ['toolbox'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'mining_camp',
      layoutType: 'cluster',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 24,
      minSeparationDistance: spec.minSeparationDistance || 45,
      lots,
      condition: spec.condition || 'collapsed',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'medium',
      lootLevel: spec.lootLevel || 'high',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'ridge_switchback',
    };
  }

  private buildGhostVillageSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    lots.push({
      id: `${spec.id}_hospital_ward`,
      buildingType: 'clinic',
      zone: 'civic',
      position: { x: cx - 6, z: cz - 5 },
      size: { width: 13, depth: 10 },
      rotationY: 0,
      doorPosition: { x: cx - 6, z: cz },
      condition: 'burnt',
      containers: ['medicine_cabinet', 'locker'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });
    lots.push({
      id: `${spec.id}_morgue`,
      buildingType: 'shed',
      zone: 'civic',
      position: { x: cx + 8, z: cz - 4 },
      size: { width: 8, depth: 7 },
      rotationY: 0.1,
      doorPosition: { x: cx + 8, z: cz - 0.5 },
      condition: 'burnt',
      containers: ['medicine_cabinet'],
      isSafeHouseCandidate: false,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'village',
      layoutType: 'radial',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 28,
      minSeparationDistance: spec.minSeparationDistance || 60,
      lots,
      condition: spec.condition || 'burnt',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'high',
      lootLevel: spec.lootLevel || 'high',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'hollow_crossroads',
    };
  }

  /**
   * 5. RING-FENCE LAYOUT: Fortified research stations, bunkers, radar outposts.
   */
  private buildOutpostSettlement(spec: Partial<SettlementInstance> & { center: { x: number; z: number } }): SettlementInstance {
    const lots: BuildingLot[] = [];
    const cx = spec.center.x;
    const cz = spec.center.z;

    lots.push({
      id: `${spec.id}_bunker_core`,
      buildingType: 'cabin',
      zone: 'fortified',
      position: { x: cx, z: cz },
      size: { width: 10, depth: 10 },
      rotationY: 0,
      doorPosition: { x: cx, z: cz + 5 },
      condition: spec.condition || 'barricaded',
      containers: ['vault_safe', 'locker', 'supply_drop_crate'],
      isSafeHouseCandidate: true,
      hasRoofFold: true,
      floorCount: 1,
    });

    return {
      id: spec.id!,
      name: spec.name!,
      poiId: spec.poiId!,
      category: 'outpost',
      layoutType: 'ring_fence',
      biome: spec.biome!,
      center: spec.center,
      clearanceRadius: spec.clearanceRadius || 24,
      minSeparationDistance: spec.minSeparationDistance || 35,
      lots,
      condition: spec.condition || 'barricaded',
      gimmick: spec.gimmick || '',
      zombieDensity: spec.zombieDensity || 'low',
      lootLevel: spec.lootLevel || 'high',
      connectedRoadNodeId: spec.connectedRoadNodeId || 'tundra_outpost',
    };
  }

  // --- SPATIAL & CLEARANCE QUERIES ---

  /**
   * Checks if coordinate (x, z) falls within the clearance radius of any settlement.
   * Used by WorldGenerator to suppress procedural trees/boulders on settlement grounds.
   */
  public isInSettlementClearance(x: number, z: number, margin: number = 0): boolean {
    for (const s of this.settlements.values()) {
      const dx = x - s.center.x;
      const dz = z - s.center.z;
      const distSq = dx * dx + dz * dz;
      const rad = s.clearanceRadius + margin;
      if (distSq <= rad * rad) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks if coordinate (x, z) falls inside any building lot footprint.
   * Performs exact 2D point-in-rotated-rectangle test.
   */
  public isInsideBuildingLot(x: number, z: number, margin: number = 0): boolean {
    return this.getLotAt(x, z, margin) !== null;
  }

  /**
   * Retrieves the specific BuildingLot at (x, z) if any, or null.
   */
  public getLotAt(x: number, z: number, margin: number = 0): BuildingLot | null {
    for (const s of this.settlements.values()) {
      for (const lot of s.lots) {
        // Translate to lot local origin
        const dx = x - lot.position.x;
        const dz = z - lot.position.z;

        // Rotate into lot local coordinate space
        const cos = Math.cos(-lot.rotationY);
        const sin = Math.sin(-lot.rotationY);
        const localX = dx * cos - dz * sin;
        const localZ = dx * sin + dz * cos;

        const halfW = lot.size.width * 0.5 + margin;
        const halfD = lot.size.depth * 0.5 + margin;

        if (Math.abs(localX) <= halfW && Math.abs(localZ) <= halfD) {
          return lot;
        }
      }
    }
    return null;
  }

  /**
   * Returns the closest settlement to (x, z) and distance in meters.
   */
  public getNearestSettlement(x: number, z: number): { settlement: SettlementInstance; distance: number } | null {
    let nearest: SettlementInstance | null = null;
    let minDist = Infinity;

    for (const s of this.settlements.values()) {
      const d = Math.hypot(x - s.center.x, z - s.center.z);
      if (d < minDist) {
        minDist = d;
        nearest = s;
      }
    }

    return nearest ? { settlement: nearest, distance: minDist } : null;
  }

  /**
   * Returns flattened list of all building lots across all settlements.
   */
  public getAllLots(): BuildingLot[] {
    const list: BuildingLot[] = [];
    for (const s of this.settlements.values()) {
      list.push(...s.lots);
    }
    return list;
  }

  // --- 3D VISUAL RENDERING ---

  /**
   * Constructs 3D ground foundations, curbs, and town entrance signboards for all settlements.
   */
  public buildVisualMeshes(scene: THREE.Scene) {
    for (const settlement of this.settlements.values()) {
      // 1. Settlement Foundation Ground Pad (y = 0.015, just below road ribbons at 0.02)
      const theme = this.getThemeForSettlement(settlement);
      const padRadius = settlement.clearanceRadius * 0.92;
      const padGeo = new THREE.CircleGeometry(padRadius, 24);
      padGeo.rotateX(-Math.PI / 2);

      const padTexture = TextureGenerator.getSettlementPadTexture(theme);
      padTexture.repeat.set(padRadius / 8, padRadius / 8);

      const padMat = new THREE.MeshLambertMaterial({
        map: padTexture,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });

      const padMesh = new THREE.Mesh(padGeo, padMat);
      padMesh.position.set(settlement.center.x, 0.015, settlement.center.z);
      padMesh.receiveShadow = true;
      this.group.add(padMesh);

      // 2. Building Lot Foundation Slabs
      for (const lot of settlement.lots) {
        const lotGeo = new THREE.PlaneGeometry(lot.size.width, lot.size.depth);
        lotGeo.rotateX(-Math.PI / 2);

        const lotMat = new THREE.MeshLambertMaterial({
          map: TextureGenerator.getBuildingFoundationTexture(lot.buildingType, lot.condition),
          transparent: true,
          alphaTest: 0.1,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });

        const lotMesh = new THREE.Mesh(lotGeo, lotMat);
        lotMesh.position.set(lot.position.x, 0.018, lot.position.z);
        lotMesh.rotation.y = lot.rotationY;
        this.group.add(lotMesh);

        // Doorstep entrance marker
        const stepGeo = new THREE.PlaneGeometry(1.6, 0.8);
        stepGeo.rotateX(-Math.PI / 2);
        const stepMat = new THREE.MeshLambertMaterial({
          color: 0xd4aa67,
          polygonOffset: true,
          polygonOffsetFactor: -3,
          polygonOffsetUnits: -3,
        });
        const stepMesh = new THREE.Mesh(stepGeo, stepMat);
        stepMesh.position.set(lot.doorPosition.x, 0.022, lot.doorPosition.z);
        stepMesh.rotation.y = lot.rotationY;
        this.group.add(stepMesh);
      }

      // 3. Roadside Town Welcome Signboard
      const signGeo = new THREE.PlaneGeometry(2.4, 1.2);
      const signMat = new THREE.MeshLambertMaterial({
        map: TextureGenerator.getSettlementSignTexture(settlement.name, settlement.category.toUpperCase()),
        transparent: true,
        alphaTest: 0.15,
        side: THREE.DoubleSide,
      });
      const signMesh = new THREE.Mesh(signGeo, signMat);

      // Offset sign slightly outside the settlement perimeter near the road node
      const signAngle = Math.atan2(settlement.center.z, settlement.center.x) + 0.3;
      const signDist = settlement.clearanceRadius * 0.75;
      signMesh.position.set(
        settlement.center.x + Math.cos(signAngle) * signDist,
        1.1,
        settlement.center.z + Math.sin(signAngle) * signDist
      );
      signMesh.rotation.y = signAngle + Math.PI / 2;
      this.group.add(signMesh);

      // Wooden support post
      const postGeo = new THREE.BoxGeometry(0.12, 1.2, 0.12);
      const postMat = new THREE.MeshLambertMaterial({ color: 0x3a2717 });
      const postMesh = new THREE.Mesh(postGeo, postMat);
      postMesh.position.set(signMesh.position.x, 0.6, signMesh.position.z);
      this.group.add(postMesh);
    }
  }

  private getThemeForSettlement(s: SettlementInstance): string {
    if (s.category === 'town') return 'town_pavement';
    if (s.category === 'strip_stop') return 'strip_parking';
    if (s.category === 'farmstead') return 'farm_dirt';
    if (s.category === 'junkyard_town') return 'junkyard_scrap';
    if (s.category === 'outpost') return 'outpost_concrete';
    return 'village_plaza';
  }
}
