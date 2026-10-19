import * as THREE from 'three';
import { BuildingLot } from './SettlementDef';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { HidingSpot } from './HidingSpot';
import { damp } from '../utils/MathUtils';

/**
 * Modular Building Entity with Pop-up Roof Fold & Interior Cutaway (Section 26 & 66).
 * Features:
 * - Cardboard pop-up roof flap that folds upright like a dollhouse upon player entry (~0.25s tween).
 * - Front wall opacity fade/cutaway when player enters to prevent camera clipping.
 * - Interior furniture and slapstick hiding spots (wardrobe, under_bed, dumpster).
 * - Window peek-holes and open doorways with entry thresholds.
 */
export class BuildingEntity {
  public lot: BuildingLot;
  public group: THREE.Group;
  public roofGroup: THREE.Group;
  public frontWallGroup: THREE.Group;
  public hidingSpots: HidingSpot[] = [];

  // Pop-up Roof Fold State
  public isPlayerInside: boolean = false;
  public roofFoldProgress: number = 0; // 0.0 (closed) to 1.0 (fully folded upright)
  private targetRoofProgress: number = 0;
  private readonly ROOF_MAX_FOLD_ANGLE: number = -1.35; // ~77 degrees upright flap rotation

  // Wall Dimensions
  public wallHeight: number = 3.6;
  private frontWallMats: THREE.MeshLambertMaterial[] = [];

  constructor(lot: BuildingLot) {
    this.lot = lot;
    this.group = new THREE.Group();
    this.group.name = `Building_${lot.id}`;
    this.group.position.set(lot.position.x, 0, lot.position.z);
    this.group.rotation.y = lot.rotationY;

    this.roofGroup = new THREE.Group();
    this.roofGroup.name = 'PopUpRoofFlap';
    this.frontWallGroup = new THREE.Group();
    this.frontWallGroup.name = 'FrontWallCutaway';

    this.buildWalls();
    this.buildPopUpRoof();
    this.buildInteriorAndHidingSpots();

    this.group.add(this.frontWallGroup);
    this.group.add(this.roofGroup);
  }

  private buildWalls() {
    const w = this.lot.size.width;
    const d = this.lot.size.depth;
    const h = this.wallHeight;
    const halfW = w * 0.5;
    const halfD = d * 0.5;

    const wallTex = TextureGenerator.getBuildingWallTexture(this.lot.buildingType, this.lot.condition, true);
    const wallTexBlank = TextureGenerator.getBuildingWallTexture(this.lot.buildingType, this.lot.condition, false);

    const standardMat = new THREE.MeshLambertMaterial({
      map: wallTex,
      transparent: true,
      side: THREE.DoubleSide,
    });

    // 1. Back Wall (-Z)
    const backGeo = new THREE.PlaneGeometry(w, h);
    backGeo.translate(0, h * 0.5, 0);
    const backWall = new THREE.Mesh(backGeo, standardMat);
    backWall.position.set(0, 0, -halfD);
    this.group.add(backWall);

    // 2. Left Wall (-X)
    const leftGeo = new THREE.PlaneGeometry(d, h);
    leftGeo.translate(0, h * 0.5, 0);
    const leftWall = new THREE.Mesh(leftGeo, standardMat);
    leftWall.position.set(-halfW, 0, 0);
    leftWall.rotation.y = Math.PI * 0.5;
    this.group.add(leftWall);

    // 3. Right Wall (+X)
    const rightGeo = new THREE.PlaneGeometry(d, h);
    rightGeo.translate(0, h * 0.5, 0);
    const rightWall = new THREE.Mesh(rightGeo, standardMat);
    rightWall.position.set(halfW, 0, 0);
    rightWall.rotation.y = -Math.PI * 0.5;
    this.group.add(rightWall);

    // 4. Front Wall with Open Doorway (+Z)
    // Left panel + Right panel with 1.8m doorway gap in center
    const doorW = 2.0;
    const panelW = (w - doorW) * 0.5;

    const frontMat = new THREE.MeshLambertMaterial({
      map: wallTexBlank,
      transparent: true,
      opacity: 1.0,
      side: THREE.DoubleSide,
    });
    this.frontWallMats.push(frontMat);

    // Left Front Panel
    const leftFrontGeo = new THREE.PlaneGeometry(panelW, h);
    leftFrontGeo.translate(0, h * 0.5, 0);
    const leftFront = new THREE.Mesh(leftFrontGeo, frontMat);
    leftFront.position.set(-halfW + panelW * 0.5, 0, halfD);
    this.frontWallGroup.add(leftFront);

    // Right Front Panel
    const rightFrontGeo = new THREE.PlaneGeometry(panelW, h);
    rightFrontGeo.translate(0, h * 0.5, 0);
    const rightFront = new THREE.Mesh(rightFrontGeo, frontMat);
    rightFront.position.set(halfW - panelW * 0.5, 0, halfD);
    this.frontWallGroup.add(rightFront);

    // Doorway Lintel Header Bar
    const lintelH = 0.8;
    const lintelGeo = new THREE.PlaneGeometry(doorW, lintelH);
    lintelGeo.translate(0, h - lintelH * 0.5, 0);
    const lintel = new THREE.Mesh(lintelGeo, frontMat);
    lintel.position.set(0, 0, halfD);
    this.frontWallGroup.add(lintel);
  }

  private buildPopUpRoof() {
    if (!this.lot.hasRoofFold) return;

    const w = this.lot.size.width + 0.6; // Slight roof eaves overhang
    const d = this.lot.size.depth + 0.6;
    const h = this.wallHeight;
    const halfD = this.lot.size.depth * 0.5;

    // Hinge roof panel along the top of the back wall (-halfD, y = h)
    this.roofGroup.position.set(0, h, -halfD);

    const roofTex = TextureGenerator.getBuildingRoofTexture(this.lot.buildingType, this.lot.condition);
    const roofMat = new THREE.MeshLambertMaterial({
      map: roofTex,
      transparent: true,
      side: THREE.DoubleSide,
    });

    // Roof plane extends forward by d from the back hinge
    const roofGeo = new THREE.PlaneGeometry(w, d);
    roofGeo.rotateX(-Math.PI * 0.5);
    roofGeo.translate(0, 0, d * 0.5); // Pivot at hinge edge (z = 0 in roofGroup)

    const roofMesh = new THREE.Mesh(roofGeo, roofMat);
    roofMesh.castShadow = true;
    this.roofGroup.add(roofMesh);
  }

  private buildInteriorAndHidingSpots() {
    const halfW = this.lot.size.width * 0.5;
    const halfD = this.lot.size.depth * 0.5;

    // 1. Interior Hiding Spot: Wardrobe / Closet in Residential & Cabins
    if (this.lot.zone === 'residential' || this.lot.buildingType === 'cabin' || this.lot.buildingType === 'motel') {
      const wardrobeSpot = new HidingSpot(
        'wardrobe',
        new THREE.Vector3(this.lot.position.x - halfW + 1.8, 0, this.lot.position.z - halfD + 1.8),
        this.lot.rotationY
      );
      this.hidingSpots.push(wardrobeSpot);
      this.group.add(wardrobeSpot.mesh);

      // Under-bed spot for motel or houses
      if (this.lot.size.width >= 9) {
        const bedSpot = new HidingSpot(
          'under_bed',
          new THREE.Vector3(this.lot.position.x + halfW - 2.2, 0, this.lot.position.z - halfD + 2.0),
          this.lot.rotationY
        );
        this.hidingSpots.push(bedSpot);
        this.group.add(bedSpot.mesh);
      }
    }

    // 2. Exterior Alley Hiding Spot: Dumpster outside Diners, Garages, Supermarkets
    if (this.lot.buildingType === 'diner' || this.lot.buildingType === 'auto_garage' || this.lot.buildingType === 'supermarket' || this.lot.buildingType === 'town_hall') {
      const dumpsterSpot = new HidingSpot(
        'dumpster',
        new THREE.Vector3(this.lot.position.x + halfW + 1.2, 0, this.lot.position.z - halfD + 2.5),
        this.lot.rotationY + Math.PI * 0.5
      );
      this.hidingSpots.push(dumpsterSpot);
      this.group.add(dumpsterSpot.mesh);
    }
  }

  /**
   * Updates building state:
   * - Tests whether player is inside building footprint.
   * - Smoothly animates pop-up roof flap opening and closing.
   * - Smoothly fades front wall opacity for clear interior visibility.
   */
  public update(delta: number, playerPos: THREE.Vector3) {
    // 1. Point-in-box test in building local space
    const dx = playerPos.x - this.lot.position.x;
    const dz = playerPos.z - this.lot.position.z;

    const cos = Math.cos(-this.lot.rotationY);
    const sin = Math.sin(-this.lot.rotationY);
    const localX = dx * cos - dz * sin;
    const localZ = dx * sin + dz * cos;

    const halfW = this.lot.size.width * 0.5;
    const halfD = this.lot.size.depth * 0.5;

    // Small boundary margin so doorway entry triggers cleanly
    const inside = Math.abs(localX) <= halfW + 0.2 && Math.abs(localZ) <= halfD + 0.2;
    this.isPlayerInside = inside;
    this.targetRoofProgress = inside ? 1.0 : 0.0;

    // 2. Pop-up Roof Fold Tween (~0.25s response time via lambda=12)
    this.roofFoldProgress = damp(this.roofFoldProgress, this.targetRoofProgress, 12, delta);
    this.roofGroup.rotation.x = this.roofFoldProgress * this.ROOF_MAX_FOLD_ANGLE;

    // 3. Front Wall Opacity Fade
    const targetOpacity = inside ? 0.25 : 1.0;
    for (const mat of this.frontWallMats) {
      mat.opacity = damp(mat.opacity, targetOpacity, 10, delta);
    }
  }

  /**
   * Returns nearby hiding spot within interaction reach of playerPos, if any.
   */
  public getNearbyHidingSpot(playerPos: THREE.Vector3): HidingSpot | null {
    for (const spot of this.hidingSpots) {
      const dist = spot.position.distanceTo(playerPos);
      if (dist <= spot.interactionRadius) {
        return spot;
      }
    }
    return null;
  }
}
