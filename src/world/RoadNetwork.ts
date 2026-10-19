import * as THREE from 'three';
import { ROAD_REGISTRY, RoadDef } from '../registries/RoadDef';
import { POI_REGISTRY, POIDef } from '../registries/POIDef';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { gameConfig } from '../config/GameConfig';

export interface RoadNode {
  id: string;
  name: string;
  position: THREE.Vector3;
  biome: string;
  poiId?: string;
}

export interface RoadPropInstance {
  type: 'traffic_cone' | 'road_sign' | 'billboard' | 'guardrail' | 'parked_wreck';
  position: THREE.Vector3;
  rotationY: number;
  label?: string;
  brandIndex?: number;
}

export interface RoadSegment {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  roadDefId: string;
  waypoints: THREE.Vector3[];
  curve: THREE.CatmullRomCurve3;
  length: number;
  width: number;
  props: RoadPropInstance[];
}

/**
 * Procedural Road Network & POI Placement System for World 4 Bro's (Phase Z1).
 * Builds a seeded highway network connecting all 10 biomes via Catmull-Rom splines,
 * renders textured 3D road ribbons, scatters roadside props, and provides spatial
 * query methods (isOnRoad, getSpeedBonusAt, getGripAt, findPath).
 */
export class RoadNetwork {
  public nodes: Map<string, RoadNode> = new Map();
  public segments: RoadSegment[] = [];
  public group: THREE.Group;
  private billboardMeshes: THREE.Mesh[] = [];
  private spatialSegments: { p1: THREE.Vector2; p2: THREE.Vector2; width: number; roadDef: RoadDef }[] = [];

  constructor(scene?: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'RoadNetwork';

    this.initNodes();
    this.initSegments();
    this.buildSpatialLookup();

    if (scene) {
      this.buildVisualMeshes(scene);
      scene.add(this.group);
      this.setVisible(gameConfig.isZombieMode());
    }
  }

  /**
   * Defines all major landmark & POI nodes across the 10 biomes.
   */
  private initNodes() {
    const rawNodes: RoadNode[] = [
      { id: 'meadow_spawn', name: 'Bro Meadow Crossroads', position: new THREE.Vector3(0, 0, 0), biome: 'meadow' },
      { id: 'meadow_farm', name: 'Old McCready Farmhouse', position: new THREE.Vector3(-22, 0, -26), biome: 'meadow', poiId: 'farmhouse' },
      { id: 'thicket_camp', name: 'Pinewood Logging Camp', position: new THREE.Vector3(-42, 0, -45), biome: 'forest' },
      { id: 'bog_causeway', name: 'Squelchy Bog Causeway', position: new THREE.Vector3(-32, 0, -10), biome: 'swamp' },
      { id: 'desert_junction', name: 'Dune Petro & Diner', position: new THREE.Vector3(42, 0, -18), biome: 'desert', poiId: 'gas_station' },
      { id: 'scrapyard_garage', name: 'Grease Monkey Auto Garage', position: new THREE.Vector3(-35, 0, 18), biome: 'gadget', poiId: 'auto_garage' },
      { id: 'tundra_outpost', name: 'Brrington Weather Station', position: new THREE.Vector3(48, 0, 28), biome: 'tundra' },
      { id: 'ridge_switchback', name: 'Rockbottom Ridge Pass', position: new THREE.Vector3(28, 0, 44), biome: 'mountain' },
      { id: 'hollow_crossroads', name: 'Haunted Hospital Ruins', position: new THREE.Vector3(-46, 0, 40), biome: 'hollow', poiId: 'hospital_ruins' },
      { id: 'volcano_bridge', name: 'Grumbleguts Obsidian Bridge', position: new THREE.Vector3(38, 0, -48), biome: 'volcano' },
      { id: 'cove_dock', name: 'Splash Cove Evac Docks', position: new THREE.Vector3(-16, 0, 58), biome: 'beach' },
    ];

    for (const node of rawNodes) {
      this.nodes.set(node.id, node);
    }
  }

  /**
   * Creates Catmull-Rom spline routes connecting all nodes into an interconnected highway loop graph.
   */
  private initSegments() {
    const rawSegments: {
      id: string;
      from: string;
      to: string;
      roadDefId: string;
      intermediatePoints?: [number, number][];
    }[] = [
      // 1. Meadow Spawn to McCready Farm
      { id: 'seg_spawn_farm', from: 'meadow_spawn', to: 'meadow_farm', roadDefId: 'country_lane', intermediatePoints: [[-10, -12]] },
      // 2. McCready Farm to Logging Camp
      { id: 'seg_farm_camp', from: 'meadow_farm', to: 'thicket_camp', roadDefId: 'logging_track', intermediatePoints: [[-32, -36]] },
      // 3. Logging Camp to Bog Causeway
      { id: 'seg_camp_bog', from: 'thicket_camp', to: 'bog_causeway', roadDefId: 'logging_track', intermediatePoints: [[-44, -26]] },
      // 4. Bog Causeway to Auto Garage (Scrapyard)
      { id: 'seg_bog_garage', from: 'bog_causeway', to: 'scrapyard_garage', roadDefId: 'boardwalk_causeway', intermediatePoints: [[-35, 4]] },
      // 5. Scrapyard Auto Garage back to Meadow Spawn (Closes Western Loop!)
      { id: 'seg_garage_spawn', from: 'scrapyard_garage', to: 'meadow_spawn', roadDefId: 'country_lane', intermediatePoints: [[-18, 10]] },
      // 6. Meadow Spawn to Desert Gas Station Junction
      { id: 'seg_spawn_desert', from: 'meadow_spawn', to: 'desert_junction', roadDefId: 'highway', intermediatePoints: [[20, -8]] },
      // 7. Desert Junction to Grumbleguts Volcano Obsidian Span
      { id: 'seg_desert_volcano', from: 'desert_junction', to: 'volcano_bridge', roadDefId: 'lava_bridge', intermediatePoints: [[44, -34]] },
      // 8. Desert Junction to Brrington Tundra Outpost
      { id: 'seg_desert_tundra', from: 'desert_junction', to: 'tundra_outpost', roadDefId: 'highway', intermediatePoints: [[48, 6]] },
      // 9. Tundra Outpost to Rockbottom Ridge Switchback Pass
      { id: 'seg_tundra_ridge', from: 'tundra_outpost', to: 'ridge_switchback', roadDefId: 'ice_road', intermediatePoints: [[40, 38]] },
      // 10. Ridge Switchback back to Meadow Spawn (Closes Eastern Loop!)
      { id: 'seg_ridge_spawn', from: 'ridge_switchback', to: 'meadow_spawn', roadDefId: 'switchback_pass', intermediatePoints: [[15, 22]] },
      // 11. Scrapyard to Haunted Hollow Hospital Crossroads
      { id: 'seg_garage_hollow', from: 'scrapyard_garage', to: 'hollow_crossroads', roadDefId: 'service_road', intermediatePoints: [[-43, 28]] },
      // 12. Hollow Crossroads to Splash Cove Evac Docks
      { id: 'seg_hollow_cove', from: 'hollow_crossroads', to: 'cove_dock', roadDefId: 'cursed_road', intermediatePoints: [[-34, 52]] },
      // 13. Rockbottom Ridge to Splash Cove Evac Docks (Scenic Coastal Route!)
      { id: 'seg_ridge_cove', from: 'ridge_switchback', to: 'cove_dock', roadDefId: 'coast_road', intermediatePoints: [[8, 54]] },
    ];

    for (const raw of rawSegments) {
      const fromNode = this.nodes.get(raw.from);
      const toNode = this.nodes.get(raw.to);
      if (!fromNode || !toNode) continue;

      const roadDef = ROAD_REGISTRY[raw.roadDefId] || ROAD_REGISTRY.country_lane;

      // Build waypoints array
      const points: THREE.Vector3[] = [fromNode.position.clone()];
      if (raw.intermediatePoints) {
        for (const [ix, iz] of raw.intermediatePoints) {
          points.push(new THREE.Vector3(ix, 0, iz));
        }
      }
      points.push(toNode.position.clone());

      // Create Catmull-Rom spline
      const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
      const length = curve.getLength();

      // Generate roadside props
      const props = this.generatePropsForSegment(curve, length, roadDef);

      this.segments.push({
        id: raw.id,
        fromNodeId: raw.from,
        toNodeId: raw.to,
        roadDefId: raw.roadDefId,
        waypoints: points,
        curve,
        length,
        width: roadDef.width,
        props,
      });
    }
  }

  /**
   * Deterministically generates roadside props along each spline segment.
   */
  private generatePropsForSegment(curve: THREE.CatmullRomCurve3, length: number, roadDef: RoadDef): RoadPropInstance[] {
    const props: RoadPropInstance[] = [];
    const interval = 16; // meters between prop opportunities
    const count = Math.floor(length / interval);

    for (let i = 1; i <= count; i++) {
      const u = (i * interval) / length;
      if (u > 0.88 || u < 0.12) continue; // Keep intersections clean

      const point = curve.getPointAt(u);
      const tangent = curve.getTangentAt(u);
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();
      const rotY = Math.atan2(tangent.x, tangent.z);

      // Alternate left / right shoulder
      const side = i % 2 === 0 ? 1 : -1;
      const offsetDist = roadDef.width * 0.5 + 0.9;
      const propPos = point.clone().add(normal.clone().multiplyScalar(side * offsetDist));
      propPos.y = 0;

      // Select prop based on ambient props of road def
      const pool = roadDef.ambientProps;
      if (pool.length === 0) continue;
      const choice = pool[(i * 3 + pool.length) % pool.length];

      if (choice === 'traffic_cone') {
        props.push({ type: 'traffic_cone', position: propPos, rotationY: rotY });
      } else if (choice === 'road_sign') {
        props.push({ type: 'road_sign', position: propPos, rotationY: rotY, label: roadDef.name });
      } else if (choice === 'billboard_ad') {
        const brandIdx = (i + Math.floor(length)) % 3;
        props.push({ type: 'billboard', position: propPos, rotationY: rotY, brandIndex: brandIdx });
      } else if (choice === 'guardrail') {
        props.push({ type: 'guardrail', position: propPos, rotationY: rotY });
      } else if (choice === 'parked_wreck') {
        // Offset wrecks slightly farther onto shoulder
        const wreckPos = point.clone().add(normal.clone().multiplyScalar(side * (offsetDist + 1.2)));
        props.push({ type: 'parked_wreck', position: wreckPos, rotationY: rotY + 0.3 });
      }
    }

    return props;
  }

  /**
   * Pre-calculates linear line segments for fast 2D distance checks.
   */
  private buildSpatialLookup() {
    this.spatialSegments = [];
    const samplesPerMeter = 0.5; // sample every 2 meters

    for (const seg of this.segments) {
      const roadDef = ROAD_REGISTRY[seg.roadDefId] || ROAD_REGISTRY.country_lane;
      const numSamples = Math.max(6, Math.ceil(seg.length * samplesPerMeter));
      const points = seg.curve.getPoints(numSamples);

      for (let i = 0; i < points.length - 1; i++) {
        this.spatialSegments.push({
          p1: new THREE.Vector2(points[i].x, points[i].z),
          p2: new THREE.Vector2(points[i + 1].x, points[i + 1].z),
          width: seg.width,
          roadDef,
        });
      }
    }
  }

  /**
   * Constructs Three.js ribbon meshes and billboard roadside props.
   */
  private buildVisualMeshes(scene: THREE.Scene) {
    if (typeof document === 'undefined') return; // Headless environment guard

    // 1. Build Road Ribbons
    for (const seg of this.segments) {
      const roadDef = ROAD_REGISTRY[seg.roadDefId] || ROAD_REGISTRY.country_lane;
      const texture = TextureGenerator.getRoadTexture(roadDef.surfaceTexture);

      const ribbonMesh = this.createRibbonMesh(seg.curve, seg.width, texture);
      this.group.add(ribbonMesh);
    }

    // 2. Build POI Aprons & Billboard Markers
    for (const [, node] of this.nodes) {
      this.buildPOIApron(node);
    }

    // 3. Build Roadside Props
    this.buildProps();
  }

  /**
   * Builds an extruded 3D ribbon geometry following the Catmull-Rom curve.
   */
  private createRibbonMesh(curve: THREE.CatmullRomCurve3, width: number, texture: THREE.CanvasTexture): THREE.Mesh {
    const segments = Math.max(16, Math.floor(curve.getLength() * 1.5));
    const points = curve.getPoints(segments);

    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    const halfWidth = width * 0.5;
    let accumulatedDistance = 0;

    for (let i = 0; i <= segments; i++) {
      const p = points[i];
      const u = i / segments;
      const tangent = curve.getTangentAt(u);
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

      if (i > 0) {
        accumulatedDistance += points[i].distanceTo(points[i - 1]);
      }

      // Height offset y = 0.02 avoids z-fighting with ground plane
      const leftX = p.x + normal.x * halfWidth;
      const leftZ = p.z + normal.z * halfWidth;
      const rightX = p.x - normal.x * halfWidth;
      const rightZ = p.z - normal.z * halfWidth;

      positions.push(leftX, 0.02, leftZ);
      positions.push(rightX, 0.02, rightZ);

      // Texture UV repetition: 1 unit per 6 meters
      const v = accumulatedDistance / 6.0;
      uvs.push(0, v);
      uvs.push(1, v);

      if (i < segments) {
        const base = i * 2;
        indices.push(base, base + 1, base + 2);
        indices.push(base + 1, base + 3, base + 2);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    return mesh;
  }

  /**
   * Creates a circular parking apron and roadside signboard at a POI node.
   */
  private buildPOIApron(node: RoadNode) {
    if (typeof document === 'undefined') return;

    const poi = node.poiId ? POI_REGISTRY[node.poiId] : null;
    const radius = poi ? Math.min(poi.sizeRadiusMeters * 0.5, 9) : 5.0;

    // Apron decal
    const apronGeo = new THREE.CircleGeometry(radius, 24);
    apronGeo.rotateX(-Math.PI / 2);
    const apronMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getRoadTexture('asphalt'),
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -0.8,
      polygonOffsetUnits: -0.8,
    });
    const apron = new THREE.Mesh(apronGeo, apronMat);
    apron.position.set(node.position.x, 0.015, node.position.z);
    apron.receiveShadow = true;
    this.group.add(apron);

    // POI Billboard Signboard
    const signGeo = new THREE.PlaneGeometry(1.6, 1.6);
    const signMat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getRoadSignTexture(node.name),
      transparent: true,
      alphaTest: 0.2,
      side: THREE.DoubleSide,
    });
    const signMesh = new THREE.Mesh(signGeo, signMat);
    signMesh.position.set(node.position.x + 3.0, 0.8, node.position.z + 3.0);
    this.group.add(signMesh);
    this.billboardMeshes.push(signMesh);
  }

  /**
   * Instantiates visual meshes for all roadside props across segments.
   */
  private buildProps() {
    if (typeof document === 'undefined') return;

    for (const seg of this.segments) {
      for (const prop of seg.props) {
        if (prop.type === 'traffic_cone') {
          const coneGeo = new THREE.PlaneGeometry(0.7, 0.7);
          const coneMat = new THREE.MeshLambertMaterial({
            map: TextureGenerator.getTrafficConeTexture(),
            transparent: true,
            alphaTest: 0.2,
            side: THREE.DoubleSide,
          });
          const cone = new THREE.Mesh(coneGeo, coneMat);
          cone.position.set(prop.position.x, 0.35, prop.position.z);
          this.group.add(cone);
          this.billboardMeshes.push(cone);

        } else if (prop.type === 'road_sign') {
          const signGeo = new THREE.PlaneGeometry(1.1, 1.1);
          const signMat = new THREE.MeshLambertMaterial({
            map: TextureGenerator.getRoadSignTexture(prop.label || 'BRO MEADOW'),
            transparent: true,
            alphaTest: 0.2,
            side: THREE.DoubleSide,
          });
          const sign = new THREE.Mesh(signGeo, signMat);
          sign.position.set(prop.position.x, 0.55, prop.position.z);
          this.group.add(sign);
          this.billboardMeshes.push(sign);

        } else if (prop.type === 'billboard') {
          const bbGeo = new THREE.PlaneGeometry(3.0, 1.6);
          const bbMat = new THREE.MeshLambertMaterial({
            map: TextureGenerator.getBillboardTexture(prop.brandIndex ?? 0),
            transparent: true,
            alphaTest: 0.15,
            side: THREE.DoubleSide,
          });
          const bb = new THREE.Mesh(bbGeo, bbMat);
          bb.position.set(prop.position.x, 1.0, prop.position.z);
          this.group.add(bb);
          this.billboardMeshes.push(bb);

        } else if (prop.type === 'guardrail') {
          const railGeo = new THREE.PlaneGeometry(1.6, 0.7);
          const railMat = new THREE.MeshLambertMaterial({
            map: TextureGenerator.getGuardrailTexture(),
            transparent: true,
            alphaTest: 0.15,
            side: THREE.DoubleSide,
          });
          const rail = new THREE.Mesh(railGeo, railMat);
          rail.position.set(prop.position.x, 0.35, prop.position.z);
          rail.rotation.y = prop.rotationY;
          this.group.add(rail);

        } else if (prop.type === 'parked_wreck') {
          const wreckGeo = new THREE.PlaneGeometry(3.2, 2.0);
          const wreckMat = new THREE.MeshLambertMaterial({
            map: TextureGenerator.getParkedWreckTexture(),
            transparent: true,
            alphaTest: 0.2,
            side: THREE.DoubleSide,
          });
          const wreck = new THREE.Mesh(wreckGeo, wreckMat);
          wreck.position.set(prop.position.x, 0.9, prop.position.z);
          this.group.add(wreck);
          this.billboardMeshes.push(wreck);
        }
      }
    }
  }

  /**
   * Sets visibility of all road visual elements.
   */
  public setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  /**
   * Orient billboard props toward camera.
   */
  public update(_delta: number, camera: THREE.Camera) {
    if (!this.group.visible) return;
    for (const mesh of this.billboardMeshes) {
      mesh.quaternion.copy(camera.quaternion);
    }
  }

  // ==========================================
  // --- SPATIAL QUERIES & GAMEPLAY HELPERS ---
  // ==========================================

  /**
   * Checks whether coordinates (x, z) are on or near any road surface ribbon.
   */
  public isOnRoad(x: number, z: number, margin: number = 0): boolean {
    if (gameConfig.isClassicMode()) return false;
    const p = new THREE.Vector2(x, z);

    for (const seg of this.spatialSegments) {
      const halfW = seg.width * 0.5 + margin;
      const d = this.distToSegment(p, seg.p1, seg.p2);
      if (d <= halfW) {
        return true;
      }
    }
    return false;
  }

  public getDistanceToNearestRoad(x: number, z: number): number {
    if (gameConfig.isClassicMode() || this.spatialSegments.length === 0) return Infinity;
    const point = new THREE.Vector2(x, z);
    let distance = Infinity;
    for (const segment of this.spatialSegments) {
      distance = Math.min(distance, this.distToSegment(point, segment.p1, segment.p2) - segment.width * 0.5);
    }
    return Math.max(0, distance);
  }

  /**
   * Returns the RoadDef for the road surface under coordinates (x, z), or null if off-road.
   */
  public getRoadAt(x: number, z: number): RoadDef | null {
    if (gameConfig.isClassicMode()) return null;
    const p = new THREE.Vector2(x, z);
    let bestDist = Infinity;
    let bestDef: RoadDef | null = null;

    for (const seg of this.spatialSegments) {
      const halfW = seg.width * 0.5;
      const d = this.distToSegment(p, seg.p1, seg.p2);
      if (d <= halfW && d < bestDist) {
        bestDist = d;
        bestDef = seg.roadDef;
      }
    }
    return bestDef;
  }

  /**
   * Returns player foot speed multiplier at (x, z). Default 1.0 (or +10% to +15% on road).
   */
  public getSpeedBonusAt(x: number, z: number): number {
    const road = this.getRoadAt(x, z);
    return road ? road.footSpeedMultiplier : 1.0;
  }

  /**
   * Returns surface grip multiplier at (x, z). Default 0.90 off-road, or 0.20 to 1.0 on road.
   */
  public getGripAt(x: number, z: number): number {
    const road = this.getRoadAt(x, z);
    return road ? road.baseGrip : 0.90;
  }

  /**
   * Finds the nearest POI to coordinates (x, z).
   */
  public getNearestPOI(x: number, z: number): { poi: POIDef; distance: number; node: RoadNode } | null {
    let nearestNode: RoadNode | null = null;
    let minDistance = Infinity;

    for (const [, node] of this.nodes) {
      if (!node.poiId) continue;
      const dist = Math.hypot(node.position.x - x, node.position.z - z);
      if (dist < minDistance) {
        minDistance = dist;
        nearestNode = node;
      }
    }

    if (nearestNode && nearestNode.poiId) {
      const poi = POI_REGISTRY[nearestNode.poiId];
      if (poi) {
        return { poi, distance: minDistance, node: nearestNode };
      }
    }
    return null;
  }

  /**
   * Shortest path search (A* graph search) between two nodes in the road network.
   */
  public findPath(fromId: string, toId: string): string[] | null {
    if (!this.nodes.has(fromId) || !this.nodes.has(toId)) return null;
    if (fromId === toId) return [fromId];

    // Build adjacency list
    const adj = new Map<string, { to: string; dist: number }[]>();
    for (const [id] of this.nodes) {
      adj.set(id, []);
    }

    for (const seg of this.segments) {
      adj.get(seg.fromNodeId)?.push({ to: seg.toNodeId, dist: seg.length });
      adj.get(seg.toNodeId)?.push({ to: seg.fromNodeId, dist: seg.length });
    }

    const startNode = this.nodes.get(fromId)!;
    const targetNode = this.nodes.get(toId)!;

    // A* data structures
    const openSet = new Set<string>([fromId]);
    const cameFrom = new Map<string, string>();
    const gScore = new Map<string, number>();
    const fScore = new Map<string, number>();

    for (const [id] of this.nodes) {
      gScore.set(id, Infinity);
      fScore.set(id, Infinity);
    }
    gScore.set(fromId, 0);
    fScore.set(fromId, startNode.position.distanceTo(targetNode.position));

    while (openSet.size > 0) {
      // Find node with lowest fScore
      let current = '';
      let lowestF = Infinity;
      for (const id of openSet) {
        const f = fScore.get(id) ?? Infinity;
        if (f < lowestF) {
          lowestF = f;
          current = id;
        }
      }

      if (current === toId) {
        // Reconstruct path
        const path = [current];
        while (cameFrom.has(current)) {
          current = cameFrom.get(current)!;
          path.unshift(current);
        }
        return path;
      }

      openSet.delete(current);
      const neighbors = adj.get(current) || [];

      for (const edge of neighbors) {
        const tentativeG = (gScore.get(current) ?? Infinity) + edge.dist;
        if (tentativeG < (gScore.get(edge.to) ?? Infinity)) {
          cameFrom.set(edge.to, current);
          gScore.set(edge.to, tentativeG);
          const neighborNode = this.nodes.get(edge.to)!;
          const h = neighborNode.position.distanceTo(targetNode.position);
          fScore.set(edge.to, tentativeG + h);
          openSet.add(edge.to);
        }
      }
    }

    return null; // No route found
  }

  /**
   * Automated verification test confirming every POI node in the graph is reachable from spawn.
   */
  public validateReachability(): { valid: boolean; unreachableNodes: string[]; checkedPairs: number } {
    const unreachable: string[] = [];
    let pairs = 0;

    for (const [targetId] of this.nodes) {
      if (targetId === 'meadow_spawn') continue;
      pairs++;
      const path = this.findPath('meadow_spawn', targetId);
      if (!path || path.length === 0) {
        unreachable.push(targetId);
      }
    }

    return {
      valid: unreachable.length === 0,
      unreachableNodes: unreachable,
      checkedPairs: pairs,
    };
  }

  /**
   * Helper: Euclidean distance from point P to line segment P1-P2.
   */
  private distToSegment(p: THREE.Vector2, p1: THREE.Vector2, p2: THREE.Vector2): number {
    const l2 = p1.distanceToSquared(p2);
    if (l2 === 0) return p.distanceTo(p1);
    const t = Math.max(0, Math.min(1, ((p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y)) / l2));
    const projX = p1.x + t * (p2.x - p1.x);
    const projY = p1.y + t * (p2.y - p1.y);
    return Math.hypot(p.x - projX, p.y - projY);
  }
}
