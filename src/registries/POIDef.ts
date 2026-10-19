export interface POIDef {
  id: string;
  name: string;
  category: 'roadside_poi' | 'settlement' | 'outpost';
  preferredBiomes: string[];
  sizeRadiusMeters: number;
  zombieDensity: 'none' | 'low' | 'medium' | 'high' | 'boss';
  lootLevel: 'low' | 'medium' | 'high' | 'jackpot';
  containers: string[]; // container IDs from CONTAINER_REGISTRY
  gimmickDescription: string;
  hasShelterBed?: boolean;
  hasCarLift?: boolean;
  hasVehicleSpawn?: boolean;
  isEvacPoint?: boolean;
  isRadioQuestSite?: boolean;
}

export const POI_REGISTRY: Record<string, POIDef> = {
  gas_station: {
    id: 'gas_station',
    name: 'Roadside Petro Station',
    category: 'roadside_poi',
    preferredBiomes: ['desert', 'meadow'],
    sizeRadiusMeters: 16,
    zombieDensity: 'medium',
    lootLevel: 'high',
    containers: ['cash_register', 'toolbox', 'fridge'],
    gimmickDescription: 'Pumping fuel trips an emergency alarm that summons a mini horde',
    hasVehicleSpawn: true
  },
  motel_no_vacancy: {
    id: 'motel_no_vacancy',
    name: 'No Vacancy Roadside Motel',
    category: 'roadside_poi',
    preferredBiomes: ['desert', 'beach'],
    sizeRadiusMeters: 20,
    zombieDensity: 'low',
    lootLevel: 'medium',
    containers: ['suitcase', 'medicine_cabinet', 'cash_register'],
    gimmickDescription: 'Beds allow safe overnight sleeping to skip the night; keys behind front desk',
    hasShelterBed: true
  },
  auto_garage: {
    id: 'auto_garage',
    name: 'Grease Monkey Auto Garage',
    category: 'roadside_poi',
    preferredBiomes: ['gadget', 'meadow'],
    sizeRadiusMeters: 18,
    zombieDensity: 'medium',
    lootLevel: 'high',
    containers: ['toolbox', 'car_trunk', 'glove_box'],
    gimmickDescription: 'Best source of car parts and tools. Car lift speeds up repair by 50%',
    hasCarLift: true,
    hasVehicleSpawn: true
  },
  police_station: {
    id: 'police_station',
    name: 'County Sheriff Outpost',
    category: 'roadside_poi',
    preferredBiomes: ['meadow', 'mountain'],
    sizeRadiusMeters: 20,
    zombieDensity: 'high',
    lootLevel: 'high',
    containers: ['locker', 'cash_register'],
    gimmickDescription: 'Locked weapon lockers and armory. Breaching the gate trips an outdoor siren'
  },
  farmhouse: {
    id: 'farmhouse',
    name: 'Old McCready Farmhouse & Barn',
    category: 'roadside_poi',
    preferredBiomes: ['meadow', 'forest'],
    sizeRadiusMeters: 25,
    zombieDensity: 'low',
    lootLevel: 'medium',
    containers: ['fridge', 'toolbox'],
    gimmickDescription: 'Abundant food, jerrycans, tools, and a reliable lawn tractor',
    hasShelterBed: true,
    hasVehicleSpawn: true
  },
  supermarket: {
    id: 'supermarket',
    name: 'Mega-Mart Supermarket',
    category: 'roadside_poi',
    preferredBiomes: ['meadow'],
    sizeRadiusMeters: 24,
    zombieDensity: 'high',
    lootLevel: 'high',
    containers: ['fridge', 'cash_register', 'toolbox'],
    gimmickDescription: 'Aisle ambushes by zombie shoppers; shopping carts can be used as rolling barricades'
  },
  hospital_ruins: {
    id: 'hospital_ruins',
    name: 'Overgrown Hospital Ruins',
    category: 'roadside_poi',
    preferredBiomes: ['hollow', 'forest'],
    sizeRadiusMeters: 22,
    zombieDensity: 'high',
    lootLevel: 'high',
    containers: ['medicine_cabinet', 'locker'],
    gimmickDescription: 'Antidote gel and medical triage supplies. Dense sanity drain in ward corridors'
  },
  logging_camp: {
    id: 'logging_camp',
    name: 'Pine River Logging Camp',
    category: 'roadside_poi',
    preferredBiomes: ['forest'],
    sizeRadiusMeters: 20,
    zombieDensity: 'low',
    lootLevel: 'medium',
    containers: ['toolbox', 'suitcase'],
    gimmickDescription: 'Abundant timber, firewood, and the coveted chainsaw of regret'
  },
  radio_tower: {
    id: 'radio_tower',
    name: 'High Peak Broadcast Radio Tower',
    category: 'outpost',
    preferredBiomes: ['mountain'],
    sizeRadiusMeters: 14,
    zombieDensity: 'low',
    lootLevel: 'high',
    containers: ['toolbox', 'locker'],
    gimmickDescription: 'Key quest location: assemble the 3 radio parts to scan the map and signal the evac dock',
    isRadioQuestSite: true
  },
  drive_in_cinema: {
    id: 'drive_in_cinema',
    name: 'Starlight Drive-In Cinema',
    category: 'roadside_poi',
    preferredBiomes: ['desert'],
    sizeRadiusMeters: 28,
    zombieDensity: 'medium',
    lootLevel: 'medium',
    containers: ['car_trunk', 'glove_box', 'cash_register'],
    gimmickDescription: 'Projector still hums, sound lures passing zombies into rows of parked cars',
    hasVehicleSpawn: true
  },
  ice_cream_depot: {
    id: 'ice_cream_depot',
    name: 'Frozen Cones Ice Cream Depot',
    category: 'roadside_poi',
    preferredBiomes: ['tundra'],
    sizeRadiusMeters: 18,
    zombieDensity: 'low',
    lootLevel: 'medium',
    containers: ['fridge', 'cash_register'],
    gimmickDescription: 'Home of the Chilly Chimes Ice Cream Truck with freezer storage',
    hasVehicleSpawn: true
  },
  checkpoint_barricade: {
    id: 'checkpoint_barricade',
    name: 'Highway Military Barricade',
    category: 'roadside_poi',
    preferredBiomes: ['tundra', 'mountain'],
    sizeRadiusMeters: 16,
    zombieDensity: 'high',
    lootLevel: 'jackpot',
    containers: ['supply_drop_crate', 'locker'],
    gimmickDescription: 'Heavily fortified choke point guarding the road to the coast'
  },
  lighthouse_diner: {
    id: 'lighthouse_diner',
    name: 'Cove Lighthouse & Crab Shack',
    category: 'roadside_poi',
    preferredBiomes: ['beach'],
    sizeRadiusMeters: 20,
    zombieDensity: 'medium',
    lootLevel: 'high',
    containers: ['fridge', 'cash_register', 'toolbox'],
    gimmickDescription: 'Sweeping lighthouse beam clears zombies; holds the evac boat key'
  },
  volcano_scrap_vault: {
    id: 'volcano_scrap_vault',
    name: 'Mount Grumbleguts Scrap Vault',
    category: 'outpost',
    preferredBiomes: ['volcano'],
    sizeRadiusMeters: 22,
    zombieDensity: 'boss',
    lootLevel: 'jackpot',
    containers: ['vault_safe', 'supply_drop_crate'],
    gimmickDescription: 'Guarded by the Highway Hulk and lava vents. Holds golden toilet seats and rare engines'
  },
  evac_dock: {
    id: 'evac_dock',
    name: 'Splash Cove Evacuation Dock',
    category: 'roadside_poi',
    preferredBiomes: ['beach'],
    sizeRadiusMeters: 25,
    zombieDensity: 'high',
    lootLevel: 'jackpot',
    containers: ['supply_drop_crate'],
    gimmickDescription: 'The ultimate escape goal! Power the rescue boat engine and hold off the final zombie wave',
    isEvacPoint: true
  },

  // --- SETTLEMENTS ---
  town: {
    id: 'town',
    name: 'Bro-ville Township',
    category: 'settlement',
    preferredBiomes: ['meadow'],
    sizeRadiusMeters: 65,
    zombieDensity: 'high',
    lootLevel: 'jackpot',
    containers: ['cash_register', 'toolbox', 'fridge', 'locker', 'medicine_cabinet'],
    gimmickDescription: 'Grid layout with town hall siren that rings across the entire valley'
  },
  strip_stop: {
    id: 'strip_stop',
    name: 'Desert Oasis Strip Stop',
    category: 'settlement',
    preferredBiomes: ['desert', 'meadow'],
    sizeRadiusMeters: 35,
    zombieDensity: 'medium',
    lootLevel: 'high',
    containers: ['cash_register', 'fridge', 'toolbox', 'suitcase'],
    gimmickDescription: 'Lifeline cluster of gas station, 24/7 diner, and motel along the highway'
  },
  junkyard_town: {
    id: 'junkyard_town',
    name: 'Scrapyard Rust City',
    category: 'settlement',
    preferredBiomes: ['gadget'],
    sizeRadiusMeters: 45,
    zombieDensity: 'medium',
    lootLevel: 'jackpot',
    containers: ['toolbox', 'car_trunk', 'vault_safe'],
    gimmickDescription: 'Haphazard container homes, trampolines, and car parts strewn across conveyors',
    hasVehicleSpawn: true
  }
};
