export interface RoadDef {
  id: string;
  name: string;
  biome: string;
  width: number; // in meters (e.g. 7m highway, 4m lane)
  baseGrip: number; // 0.20 to 1.0
  footSpeedMultiplier: number; // default 1.10 (+10% on foot)
  surfaceTexture: 'asphalt' | 'dirt' | 'wood_plank' | 'gravel' | 'ice' | 'metal_plate' | 'cracked_asphalt' | 'cooled_lava';
  hazardTypes: ('oil_slick' | 'pothole' | 'spike_strip' | 'fallen_tree' | 'roadblock' | 'missing_planks')[];
  ambientProps: string[];
}

export const ROAD_REGISTRY: Record<string, RoadDef> = {
  highway: {
    id: 'highway',
    name: 'Desert Interstate Highway',
    biome: 'desert',
    width: 7.5,
    baseGrip: 1.0,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'asphalt',
    hazardTypes: ['oil_slick', 'pothole', 'roadblock'],
    ambientProps: ['billboard_ad', 'guardrail', 'traffic_cone', 'parked_wreck']
  },
  country_lane: {
    id: 'country_lane',
    name: 'Meadow Country Lane',
    biome: 'meadow',
    width: 5.0,
    baseGrip: 0.95,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'asphalt',
    hazardTypes: ['pothole', 'fallen_tree'],
    ambientProps: ['road_sign', 'traffic_cone']
  },
  logging_track: {
    id: 'logging_track',
    name: 'Pinewood Logging Track',
    biome: 'forest',
    width: 4.5,
    baseGrip: 0.70,
    footSpeedMultiplier: 1.08,
    surfaceTexture: 'dirt',
    hazardTypes: ['fallen_tree', 'pothole'],
    ambientProps: ['road_sign', 'parked_wreck']
  },
  boardwalk_causeway: {
    id: 'boardwalk_causeway',
    name: 'Squelchy Bog Boardwalk',
    biome: 'swamp',
    width: 4.0,
    baseGrip: 0.80,
    footSpeedMultiplier: 1.15, // saves from 40% mud slow!
    surfaceTexture: 'wood_plank',
    hazardTypes: ['missing_planks'],
    ambientProps: ['guardrail', 'road_sign']
  },
  switchback_pass: {
    id: 'switchback_pass',
    name: 'Rockbottom Ridge Switchback Pass',
    biome: 'mountain',
    width: 4.5,
    baseGrip: 0.85,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'gravel',
    hazardTypes: ['pothole', 'roadblock'],
    ambientProps: ['guardrail', 'road_sign', 'tunnel']
  },
  ice_road: {
    id: 'ice_road',
    name: 'Brrington Ice Road',
    biome: 'tundra',
    width: 6.0,
    baseGrip: 0.20,
    footSpeedMultiplier: 1.05,
    surfaceTexture: 'ice',
    hazardTypes: ['pothole', 'roadblock'],
    ambientProps: ['road_barrier', 'road_sign']
  },
  service_road: {
    id: 'service_road',
    name: 'Scrapyard Service Road',
    biome: 'gadget',
    width: 5.0,
    baseGrip: 0.75,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'metal_plate',
    hazardTypes: ['spike_strip', 'oil_slick', 'roadblock'],
    ambientProps: ['billboard_ad', 'parked_wreck', 'traffic_cone']
  },
  cursed_road: {
    id: 'cursed_road',
    name: 'The Hollow Cursed Road',
    biome: 'hollow',
    width: 5.5,
    baseGrip: 0.85,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'cracked_asphalt',
    hazardTypes: ['pothole', 'roadblock'],
    ambientProps: ['parked_wreck', 'road_sign']
  },
  coast_road: {
    id: 'coast_road',
    name: 'Splash Cove Coast Road',
    biome: 'beach',
    width: 5.5,
    baseGrip: 0.80,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'asphalt',
    hazardTypes: ['pothole', 'oil_slick'],
    ambientProps: ['guardrail', 'traffic_cone', 'billboard_ad']
  },
  lava_bridge: {
    id: 'lava_bridge',
    name: 'Grumbleguts Obsidian Lava Bridge',
    biome: 'volcano',
    width: 6.0,
    baseGrip: 0.90,
    footSpeedMultiplier: 1.10,
    surfaceTexture: 'cooled_lava',
    hazardTypes: ['roadblock'],
    ambientProps: ['guardrail', 'bridge']
  }
};
