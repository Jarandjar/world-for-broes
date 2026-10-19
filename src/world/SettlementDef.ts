/**
 * Settlement & Town Layout Definitions for World 4 Bro's (Phase Z2).
 * Covers Section 24 (Towns, villages and small places),
 * Section 25 (Lost outposts and hidden places),
 * Section 26 (Building kit and interiors),
 * and Section 27 (Procedural layout algorithms & spacing rules).
 */

export type SettlementLayoutType = 'grid' | 'strip' | 'radial' | 'cluster' | 'ring_fence';

export type SettlementCondition =
  | 'pristine'
  | 'worn'
  | 'burnt'
  | 'flooded'
  | 'collapsed'
  | 'overgrown'
  | 'barricaded';

export type BuildingZone =
  | 'civic'
  | 'commercial'
  | 'residential'
  | 'industrial'
  | 'outskirts'
  | 'fortified';

export interface BuildingLot {
  id: string;
  buildingType: string;
  zone: BuildingZone;
  position: { x: number; z: number };
  size: { width: number; depth: number }; // meters
  rotationY: number; // radians
  doorPosition: { x: number; z: number };
  condition: SettlementCondition;
  containers: string[];
  isSafeHouseCandidate: boolean;
  hasRoofFold: boolean;
  floorCount: number;
}

export interface SettlementInstance {
  id: string;
  name: string;
  poiId: string;
  category:
    | 'town'
    | 'village'
    | 'hamlet'
    | 'farmstead'
    | 'strip_stop'
    | 'harbor_town'
    | 'stilt_village'
    | 'mining_camp'
    | 'logging_camp'
    | 'junkyard_town'
    | 'outpost';
  layoutType: SettlementLayoutType;
  biome: string;
  center: { x: number; z: number };
  clearanceRadius: number; // meters
  minSeparationDistance: number; // meters from other settlements of similar scale
  lots: BuildingLot[];
  condition: SettlementCondition;
  gimmick: string;
  zombieDensity: 'none' | 'low' | 'medium' | 'high' | 'boss';
  lootLevel: 'low' | 'medium' | 'high' | 'jackpot';
  connectedRoadNodeId: string;
  streetName?: string;
}

/**
 * Spacing rules derived from Section 27 (Poisson-disc site picking):
 * - Towns: >= 300m separation
 * - Large villages & Junkyard: >= 60m separation
 * - Strip stops & Farmsteads: >= 45m separation
 * - Hamlets & Outposts: >= 35m separation
 */
export const SETTLEMENT_SPACING_RULES: Record<string, number> = {
  town: 300,
  junkyard_town: 70,
  village: 60,
  harbor_town: 60,
  stilt_village: 55,
  farmstead: 45,
  strip_stop: 45,
  mining_camp: 45,
  logging_camp: 45,
  hamlet: 35,
  outpost: 35,
};
