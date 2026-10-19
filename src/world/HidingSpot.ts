import * as THREE from 'three';
import { TextureGenerator } from '../rendering/TextureGenerator';
import { sounds } from '../audio/SoundEffects';

export type HidingSpotType = 'wardrobe' | 'under_bed' | 'dumpster';

/**
 * HidingSpot Entity (Section 26 & Section 66).
 * Features:
 * - 3 slapstick hiding spots: wardrobe (closet), under_bed, and dumpster.
 * - Interactive entry/exit: pressing KeyE or action button toggles hiding.
 * - Comical visual peek state: cartoon eyes peer through door cracks / bed ruffles / dumpster lids.
 * - Aggro drop: breaks zombie line-of-sight and resets pursuer aggro.
 */
export class HidingSpot {
  public type: HidingSpotType;
  public position: THREE.Vector3;
  public mesh: THREE.Mesh;
  public isOccupied: boolean = false;
  public isPeeking: boolean = false;
  public promptText: string;
  public interactionRadius: number = 2.2;
  private mat: THREE.MeshLambertMaterial;

  constructor(type: HidingSpotType, position: THREE.Vector3, rotationY: number = 0) {
    this.type = type;
    this.position = position.clone();
    this.promptText = this.getPromptLabel(type);

    // Visual Dimensions
    let w = 1.4;
    let h = 1.8;
    if (type === 'under_bed') {
      w = 2.0;
      h = 1.2;
    } else if (type === 'dumpster') {
      w = 2.2;
      h = 1.5;
    }

    const geo = new THREE.PlaneGeometry(w, h);
    geo.translate(0, h * 0.5, 0);

    this.mat = new THREE.MeshLambertMaterial({
      map: TextureGenerator.getHidingSpotTexture(type, false),
      transparent: true,
      alphaTest: 0.15,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(geo, this.mat);
    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = rotationY;
    this.mesh.castShadow = true;
  }

  private getPromptLabel(type: HidingSpotType): string {
    switch (type) {
      case 'wardrobe':
        return 'Hide in Closet [E]';
      case 'under_bed':
        return 'Crawl Under Bed [E]';
      case 'dumpster':
        return 'Dive in Dumpster [E]';
    }
  }

  /**
   * Toggles player hiding state inside this spot.
   * Returns true if player successfully entered, false if player exited.
   */
  public toggleHide(): boolean {
    if (!this.isOccupied) {
      this.isOccupied = true;
      this.isPeeking = true;
      this.updateTexture();

      // Slapstick sound effects
      if (this.type === 'dumpster') {
        sounds.playRubberChicken();
      } else {
        sounds.playDoorOpen();
      }
      return true;
    } else {
      this.isOccupied = false;
      this.isPeeking = false;
      this.updateTexture();
      sounds.playDoorOpen();
      return false;
    }
  }

  public updateTexture() {
    this.mat.map = TextureGenerator.getHidingSpotTexture(this.type, this.isPeeking);
    this.mat.needsUpdate = true;
  }
}
