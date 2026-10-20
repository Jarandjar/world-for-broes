# Asset Catalog

> **Generated** by `npm run art:catalog` (`tools/asset-catalog.ts`) from the game's content registries in `src/data` plus the planned snap kit, furniture, roads, characters, decals and FX. Do not edit the tables by hand — change the data or the script and re-run. Machine-readable copy: `tools/blender/jobs/asset_catalog.json`.

Style for every asset: `STYLE_BIBLE.md` §0 (isometric 2:1, 1 tile = 1 m, light from the north-west, muted gritty palette, painted not pixel art, mild blood). Workflow and sources: `ASSET_PIPELINE.md`.

Status: `placeholder` = exists in the game as code-drawn art, needs the HD sprite; `todo` = not in the game yet; `done` = final sprite in `public/assets/`.

## Summary

| Category | Count |
| :--- | ---: |
| Ground tiles | 11 |
| Vegetation | 10 |
| Rocks & resources | 6 |
| World props | 28 |
| Player buildables | 24 |
| Building types (assembled from the snap kit) | 11 |
| Vehicles | 7 |
| Zombies | 8 |
| Animals | 7 |
| Items: material | 52 |
| Items: food | 23 |
| Items: tool | 24 |
| Items: gear | 42 |
| Weapon attachments | 22 |
| Status effects | 12 |
| Character: hair | 10 |
| Snap kit: floors | 6 |
| Snap kit: walls | 36 |
| Snap kit: openings | 8 |
| Snap kit: roofs | 7 |
| Snap kit: structure | 9 |
| Furniture | 34 |
| Roads & streets | 19 |
| Characters: body & clothing | 13 |
| Decals | 11 |
| FX | 11 |
| Lights | 5 |
| **Total** | **456** |

## Ground tiles (11)

- **Make**: base diamond tile ×4 variants + 16/47 blob transitions + detail decals
- **Source**: CC0 texture (ambientCG / Poly Haven) → tileable 2:1 diamond in Blender, tinted to the palette

| id | name | status |
| :--- | :--- | :--- |
| `grass` | grass | done |
| `moss` | moss | placeholder |
| `dirt` | dirt | done |
| `needles` | needles | placeholder |
| `mud` | mud | placeholder |
| `rock` | rock | placeholder |
| `sand` | sand | placeholder |
| `snow` | snow | placeholder |
| `ash` | ash | placeholder |
| `shallow` | shallow | placeholder |
| `deep` | deep | placeholder |

## Vegetation (10)

- **Make**: trunk + canopy (canopy fades when behind), wind-sway frames, cut/stump state
- **Source**: Sapling Tree Gen or tree-gen in Blender (tool GPL, output ours) → trunk + canopy split

| id | name | status |
| :--- | :--- | :--- |
| `pine` | Pine Tree | placeholder |
| `snow_pine` | Snowy Pine | placeholder |
| `oak` | Oak Tree | placeholder |
| `dead_tree` | Dead Tree | placeholder |
| `bush` | Bush | done |
| `berry_bush` | Berry Bush | placeholder |
| `log` | Fallen Log | placeholder |
| `stump` | Stump | placeholder |
| `street_light` | Street Lamppost | done |
| `street_sign` | Street Name Sign | placeholder |

## Rocks & resources (6)

- **Make**: 1 view, states: intact / damaged / destroyed
- **Source**: Blender procedural rock (displace + decimate) or Poly Haven CC0 scan

| id | name | status |
| :--- | :--- | :--- |
| `rock` | Rock | placeholder |
| `boulder` | Boulder | placeholder |
| `iron_node` | Iron Mineral Node | placeholder |
| `copper_node` | Copper Mineral Node | placeholder |
| `coal_node` | Coal Seam | placeholder |
| `crystal_geode` | Glow Crystal Geode | placeholder |

## World props (28)

| id | name | make | source | status |
| :--- | :--- | :--- | :--- | :--- |
| `mushroom` | Mushrooms | 1 view | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `reeds` | Reeds | 1 view | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `cactus` | Cactus | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `crate` | Old Crate | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `tent` | Abandoned Tent | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `fuel_pump` | Fuel Pump | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `radio_tower` | Radio Tower | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `sandbags` | Sandbag Barrier | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `wrecked_car` | Wrecked Car | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `oil_seam` | Crude Oil Seam | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `town_fountain` | Town Square Fountain | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `guardrail` | Highway Guardrail | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `picket_fence` | Picket Fence | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `park_bench` | Park Bench | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `fire_hydrant` | Fire Hydrant | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `dumpster` | Alley Dumpster | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | done |
| `traffic_cone` | Traffic Warning Cone | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `target_dummy` | Ballistic Target Dummy | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `steel_target` | Steel Pop-Up Plate | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `slime_drum` | Bio-Slime Pressure Drum | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `scrap_wall` | Corrugated Scrap Barricade | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `tactical_shield` | Aegis Deploy Shield | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `stop_sign` | Stop Sign | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `bus_stop` | Bus Stop Shelter | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `trash_can` | Trash Can | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | done |
| `power_pole` | Power Pole | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `abandoned_car` | Abandoned Car | 1 view, states: intact / damaged / destroyed | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |
| `litter` | Street Litter | 1 view | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender | placeholder |

## Player buildables (24)

| id | name | make | source | status |
| :--- | :--- | :--- | :--- | :--- |
| `claim_flag` | Claim Flag | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `campfire` | Campfire | 4 rotations, states: intact / damaged / destroyed, light sprite | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `wall_wood` | Wooden Wall | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `door_wood` | Wooden Door | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `floor_wood` | Wooden Floor | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | done |
| `chest` | Storage Chest | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `bed` | Bedroll Bed | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | done |
| `workbench` | Workbench | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `torch_post` | Torch Post | 4 rotations, states: intact / damaged / destroyed, light sprite | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `conveyor_belt` | Conveyor Belt | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `spring_inserter` | Spring Inserter | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `burner_drill` | Burner Mining Drill | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `stone_furnace` | Smelting Furnace | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `grand_banquet_table` | Grand Dining Table | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `medical_triage_cot` | Medical Triage Cot | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `horseshoe_pin_post` | Horseshoe Pin Game | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `colony_overseer_desk` | Colony Overseer Desk | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `belt_bolter` | Belt-Fed Bolter Turret | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `flame_turret` | Pressurized Flame Turret | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `zap_fence` | Electrified Zap-Fence | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `mallet_catapult` | Spring Mallet Catapult | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `repair_drone_bay` | Autonomous Repair Drone Bay | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `clan_territory_totem` | Clan Territory Totem | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `gunsmith_bench` | Gunsmith Mod Workbench | 4 rotations, states: intact / damaged / destroyed | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |

## Building types (assembled from the snap kit) (11)

| id | name | make | source | status |
| :--- | :--- | :--- | :--- | :--- |
| `cottage` | Cozy Cottage | kit pieces in this style; roof gable, walls plank; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `log_cabin` | Log Cabin | kit pieces in this style; roof hip, walls log; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `townhouse` | Brick Townhouse | kit pieces in this style; roof hip, walls brick; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `corner_shop` | Corner Grocery Store | kit pieces in this style; roof gable, walls plaster; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `supermarket` | Supermarket Megastore | kit pieces in this style; roof flat, walls concrete; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `auto_garage` | Auto Repair Garage | kit pieces in this style; roof tin, walls concrete; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `clinic` | Emergency Clinic | kit pieces in this style; roof hip, walls plaster; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `warehouse` | Industrial Warehouse | kit pieces in this style; roof tin, walls scrap; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `tool_shed` | Tool & Storage Shed | kit pieces in this style; roof tin, walls plank; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `survivor_shack` | Survivor Barricaded Shack | kit pieces in this style; roof broken, walls scrap; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |
| `apartment_block` | Residential Apartments | kit pieces in this style; roof flat, walls brick; interior room templates | Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials | placeholder |

## Vehicles (7)

- **Make**: 16 directions × clean/dented/wrecked; part layers (doors, hood, wheels, glass, lights)
- **Source**: Kenney Car Kit (CC0) or our parametric car generator → parts as layers, grime

| id | name | status |
| :--- | :--- | :--- |
| `rusty_hatchback` | Rusty Hatchback | placeholder |
| `clown_car` | Clown Car of Wonders | placeholder |
| `ice_cream_truck` | Chilly Chimes Ice Cream Truck | placeholder |
| `golf_cart` | Golf Cart Deluxe | placeholder |
| `bathtub_buggy` | Bathtub Buggy | placeholder |
| `hot_dog_van` | Frankfurter Cruiser | placeholder |
| `tesla_roadster` | Cyber Roadster | placeholder |

## Zombies (8)

- **Make**: 8 directions × idle 8f, walk 8f, run 8f, attack 6f, hit 4f, die 8f @12 fps; mild blood variants
- **Source**: MPFB 2 body (CC0 assets) or the UE5 mannequin + CC0 Quaternius Universal Animation Library (Mixamo only privately) → render

| id | name | status |
| :--- | :--- | :--- |
| `walker` | Shambling Infected | placeholder |
| `runner` | Frenzied Sprinter | placeholder |
| `brute` | Armored Hulking Brute | placeholder |
| `sapper` | Powder-Keg Sapper | placeholder |
| `spitter` | Corrosive Acid Spitter | placeholder |
| `burrower` | Tunnel Burrower | placeholder |
| `screamer` | Siren Screamer Queen | placeholder |
| `goliath` | Highway Goliath Demolisher | placeholder |

## Animals (7)

- **Make**: 8 directions × idle, walk, run, eat, hit, die
- **Source**: Quaternius animated animals (CC0, check pack) → render

| id | name | status |
| :--- | :--- | :--- |
| `cow` | Spotted Meadow Cow | placeholder |
| `sheep` | Fluffy Cloud Sheep | placeholder |
| `pig` | Mud Snout Pig | placeholder |
| `chicken` | Clucking Farm Hen | placeholder |
| `deer` | Forest Antlered Deer | placeholder |
| `rabbit` | Hop-Bun Hare | placeholder |
| `goose` | Mischief Honk Goose | placeholder |

## Items: material (52)

- **Make**: inventory icon + ground drop sprite
- **Source**: Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon

| id | name | status |
| :--- | :--- | :--- |
| `wood` | Wood | placeholder |
| `twig` | Twig | placeholder |
| `stone` | Stone | placeholder |
| `flint` | Flint | placeholder |
| `pine_cone` | Pine Cone | placeholder |
| `acorn` | Acorn | placeholder |
| `fiber` | Plant Fiber | placeholder |
| `scrap_metal` | Scrap Metal | placeholder |
| `cloth` | Cloth | placeholder |
| `fuel_canister` | Fuel Canister | placeholder |
| `car_battery` | Car Battery | placeholder |
| `scrap_tire` | Scrap Tire | placeholder |
| `iron_ore` | Iron Ore | placeholder |
| `copper_ore` | Copper Ore | placeholder |
| `coal_chunk` | Coal Chunk | placeholder |
| `crude_oil_canister` | Crude Oil | placeholder |
| `energetic_crystal` | Energetic Crystal | placeholder |
| `iron_plate` | Iron Plate | placeholder |
| `copper_wire` | Copper Wire | placeholder |
| `heavy_gear` | Heavy Gear | placeholder |
| `conveyor_belt` | Conveyor Belt | placeholder |
| `spring_inserter` | Spring Inserter | placeholder |
| `burner_drill` | Burner Drill | placeholder |
| `stone_furnace` | Smelting Furnace | placeholder |
| `turret_ammo` | Turret Bolt Ammo | placeholder |
| `flame_fuel` | Pressurized Flame Fuel | placeholder |
| `c4_plunger_charge` | C4 Plunger Breaching Charge | placeholder |
| `repair_drone_kit` | Micro Repair Drone | placeholder |
| `clan_totem_crest` | Faction Clan Crest | placeholder |
| `raw_beef` | Raw Prime Beef | placeholder |
| `raw_mutton` | Raw Mutton | placeholder |
| `raw_pork` | Raw Pork Chop | placeholder |
| `raw_poultry` | Raw Poultry | placeholder |
| `raw_venison` | Raw Venison Steak | placeholder |
| `rabbit_meat` | Tender Rabbit Cut | placeholder |
| `leather` | Tanned Leather Hide | placeholder |
| `wool_bundle` | Fluffy Wool Bundle | placeholder |
| `feather` | Soft Down Feather | placeholder |
| `deer_antlers` | Branching Antlers | placeholder |
| `lucky_rabbit_foot` | Lucky Rabbit Foot Charm | placeholder |
| `animal_feed` | Golden Grain Feed | placeholder |
| `pebbles` | Smooth River Pebbles | placeholder |
| `rifle_ammo` | Precision Rifle Cartridges | placeholder |
| `shotgun_shells` | Heavy 12-Gauge Buckshot | placeholder |
| `scrap_nails` | Box of Scrap Nails | placeholder |
| `foam_darts` | Foam Suction Darts | placeholder |
| `cream_pie` | Custard Cream Pie | placeholder |
| `water_balloon` | Water Balloon | placeholder |
| `road_flare` | Handheld Road Flare | placeholder |
| `stink_sock_bomb` | Stink Sock Bomb | placeholder |
| `whoopee_cushion_trap` | Whoopee Cushion Trap | placeholder |
| `googly_eyes` | Self-Adhesive Googly Eyes | placeholder |

## Items: food (23)

- **Make**: inventory icon + ground drop sprite
- **Source**: Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon

| id | name | status |
| :--- | :--- | :--- |
| `bandage` | Bandage | placeholder |
| `rotten_meat` | Rotten Meat | placeholder |
| `berries` | Berries | placeholder |
| `mushroom` | Mushroom | placeholder |
| `cactus_pulp` | Cactus Pulp | placeholder |
| `jerky` | Mushroom Jerky | placeholder |
| `canned_beans` | Canned Beans | placeholder |
| `wild_carrot` | Wild Carrot | placeholder |
| `hiccup_berries` | Hiccup Berries | placeholder |
| `giggle_mushroom` | Giggle Mushroom | placeholder |
| `chili_of_regret` | Chili of Regret | placeholder |
| `pancake_stack` | Fluffy Pancake Stack | placeholder |
| `rainbow_soup` | Iridescent Rainbow Soup | placeholder |
| `crispy_sock_chips` | Crispy Sock Chips | placeholder |
| `bubble_tea_of_buoyancy` | Bubble Tea of Buoyancy | placeholder |
| `antidote_gel` | Bio-Antidote Gel | placeholder |
| `fresh_milk` | Fresh Creamy Milk | placeholder |
| `fresh_egg` | Fresh Farm Egg | placeholder |
| `cooked_steak` | Sizzling Beef Steak | placeholder |
| `cooked_bacon` | Crispy Country Bacon | placeholder |
| `roast_chicken` | Golden Roast Chicken | placeholder |
| `cooked_venison` | Campfire Roast Venison | placeholder |
| `fluffy_omelet` | Country Herb Omelet | placeholder |

## Items: tool (24)

| id | name | make | source | status |
| :--- | :--- | :--- | :--- | :--- |
| `axe` | Axe | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pickaxe` | Pickaxe | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `shovel` | Shovel | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `torch` | Torch | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `nail_bat` | Nail Bat | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `crowbar` | Crowbar | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `rubber_chicken` | Rubber Chicken | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `fishing_rod` | Fishing Rod | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pool_noodle_of_justice` | Pool Noodle of Justice | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `chainsaw_of_regret` | Chainsaw of Regret | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `traffic_sign_axe` | Traffic Sign Great-Axe | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `frying_pan` | Cast Iron Frying Pan | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `stale_baguette` | Stale Baguette | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `slingshot` | Forked Slingshot | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `potato_cannon` | Pressurized Potato Cannon | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hunting_rifle` | Precision Hunting Rifle | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pump_shotgun` | Heavy Pump Shotgun | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `nailgun_smg` | Scrap Pneumatic SMG | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `air_horn` | Canister Air Horn | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `lantern` | Lantern | inventory icon + ground drop sprite; worn/held layer on the character (offHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `torch_off` | Torch (off hand) | inventory icon + ground drop sprite; worn/held layer on the character (offHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pot_lid` | Pot Lid Shield | inventory icon + ground drop sprite; worn/held layer on the character (offHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `recon_rifle` | Recon Marksman DMR | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `scrap_nailgun` | Pneumatic Scrap Nail Driver | inventory icon + ground drop sprite; worn/held layer on the character (mainHand, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |

## Items: gear (42)

| id | name | make | source | status |
| :--- | :--- | :--- | :--- | :--- |
| `beanie` | Beanie | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hood` | Hood | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `cap` | Baseball Cap | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `bucket_hat` | Bucket Hat | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `cone_hat` | Traffic Cone Hat | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `straw_hat` | Straw Hat | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pot_helmet` | Pot Helmet | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `bandana` | Bandana | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `propeller_beanie` | Propeller Beanie | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `goggles` | Goggles | inventory icon + ground drop sprite; worn/held layer on the character (face, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `round_glasses` | Round Glasses | inventory icon + ground drop sprite; worn/held layer on the character (face, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `fake_mustache` | Fake Mustache | inventory icon + ground drop sprite; worn/held layer on the character (face, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `scarf` | Scarf | inventory icon + ground drop sprite; worn/held layer on the character (neck, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hoodie` | Hoodie Jacket | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `tshirt` | T-Shirt | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `trench_coat` | Trench Coat | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `poncho` | Rain Poncho | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hivis_vest` | Hi-Vis Vest | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `sweater` | Knit Sweater | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `pillow_vest` | Feather Pillow Vest | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `jeans` | Jeans | inventory icon + ground drop sprite; worn/held layer on the character (legs, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `cargo_pants` | Cargo Pants | inventory icon + ground drop sprite; worn/held layer on the character (legs, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `shorts` | Shorts | inventory icon + ground drop sprite; worn/held layer on the character (legs, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `overalls` | Overalls | inventory icon + ground drop sprite; worn/held layer on the character (legs, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `boots` | Work Boots | inventory icon + ground drop sprite; worn/held layer on the character (feet, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `sneakers` | Sneakers | inventory icon + ground drop sprite; worn/held layer on the character (feet, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `wellies` | Wellies | inventory icon + ground drop sprite; worn/held layer on the character (feet, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `boing_boots` | Spring Boing Boots | inventory icon + ground drop sprite; worn/held layer on the character (feet, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `leather_gloves` | Leather Gloves | inventory icon + ground drop sprite; worn/held layer on the character (hands, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `mittens` | Mittens | inventory icon + ground drop sprite; worn/held layer on the character (hands, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hiker_pack` | Hiker Pack | inventory icon + ground drop sprite; worn/held layer on the character (back, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `school_pack` | School Backpack | inventory icon + ground drop sprite; worn/held layer on the character (back, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `satchel` | Satchel | inventory icon + ground drop sprite; worn/held layer on the character (back, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `tool_belt` | Tool Belt | inventory icon + ground drop sprite; worn/held layer on the character (belt, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `fanny_pack_of_holding` | Fanny Pack of Holding | inventory icon + ground drop sprite; worn/held layer on the character (belt, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `breacher_helmet` | Tactical Breacher Helmet | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `breacher_vest` | Heavy Breacher Plate Carrier | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `welder_mask` | Industrial Welder Mask | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `welder_apron` | Heavy Leather Welder Apron | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hazmat_hood` | Duck Hazmat Filter Hood | inventory icon + ground drop sprite; worn/held layer on the character (head, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `hazmat_suit` | Sealed Duck Hazmat Suit | inventory icon + ground drop sprite; worn/held layer on the character (torso, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |
| `quack_pak` | Pressurized Quack-Pak | inventory icon + ground drop sprite; worn/held layer on the character (back, 8 directions) | Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender; Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon | placeholder |

## Weapon attachments (22)

- **Make**: icon + layer on the weapon sprite
- **Source**: Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon

| id | name | status |
| :--- | :--- | :--- |
| `optic_rangefinder` | Tactical Rangefinder HUD | placeholder |
| `optic_camcorder` | Taped Handycam CRT Night Sight | placeholder |
| `optic_googly` | Googly Eye Smart Reticle | placeholder |
| `optic_magnifier` | Taped Glass Magnifier | placeholder |
| `optic_toilet_roll` | Cardboard TP Tube Red-Dot | placeholder |
| `muzzle_potato_silencer` | Russet Potato Silencer | placeholder |
| `muzzle_soup_can` | Campbell Soup Can Choke | placeholder |
| `muzzle_car_horn` | Car Horn Blast Shroud | placeholder |
| `muzzle_copper_pipe` | Extended Copper Pipe Rifled Barrel | placeholder |
| `muzzle_kazoo` | Party Kazoo Muzzle Brake | placeholder |
| `under_cat_laser` | Kitty Cat Toy Laser Sight | placeholder |
| `under_plunger_bayonet` | Toilet Plunger Melee Bayonet | placeholder |
| `under_kitchen_fork` | Kitchen Meat Fork Bayonet | placeholder |
| `under_flashlight` | Duct-Taped Tactical Flashlight | placeholder |
| `under_salad_spoon` | Salad Tongs Ergonomic Foregrip | placeholder |
| `mag_twin_soda` | Twin Soda Bottles Mag Extension | placeholder |
| `mag_rubber_band` | Rubber Band Speedloader | placeholder |
| `mag_hopper_funnel` | Gravitational Hopper Funnel | placeholder |
| `goofy_rubber_duck` | Rubber Duck Trigger Squeaker | placeholder |
| `goofy_fuzzy_dice` | Lucky Fuzzy Dice Charm | placeholder |
| `goofy_whoopee` | Whoopee Cushion Recoil Buffer | placeholder |
| `goofy_windup_clock` | Wind-Up Clock Heartbeat Monitor | placeholder |

## Status effects (12)

- **Make**: HUD icon + world FX frames (loop) + sound word
- **Source**: Blender particles / simulation rendered to frames, or hand-painted sprite frames

| id | name | status |
| :--- | :--- | :--- |
| `acid` | Acid | placeholder |
| `burn` | Burn | placeholder |
| `explosion` | Explosion | placeholder |
| `stagger` | Stagger | placeholder |
| `electricity` | Electricity | placeholder |
| `cold` | Cold | placeholder |
| `spores` | Spores | placeholder |
| `goo` | Goo | placeholder |
| `wet` | Wet | placeholder |
| `oil_slick` | Oil Slick | placeholder |
| `blind` | Blind | placeholder |
| `smoke` | Smoke | placeholder |

## Character: hair (10)

- **Make**: 8 directions layer, recolourable
- **Source**: MPFB 2 body (CC0 assets) or the UE5 mannequin + CC0 Quaternius Universal Animation Library (Mixamo only privately) → render

| id | name | status |
| :--- | :--- | :--- |
| `bald` | Bald | placeholder |
| `buzz` | Buzz cut | placeholder |
| `messy` | Messy | placeholder |
| `side_part` | Side part | placeholder |
| `long` | Long | placeholder |
| `ponytail` | Ponytail | placeholder |
| `bun` | Top bun | placeholder |
| `mohawk` | Mohawk | placeholder |
| `afro` | Afro | placeholder |
| `braids` | Braids | placeholder |

## Snap kit: floors (6)

- **Make**: 1×1 diamond, clean/dirty/damaged
- **Source**: Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials

| id | name | status |
| :--- | :--- | :--- |
| `floor_wood` | floor wood | done |
| `floor_tile` | floor tile | todo |
| `floor_carpet` | floor carpet | todo |
| `floor_concrete` | floor concrete | done |
| `floor_linoleum` | floor linoleum | todo |
| `floor_dirt` | floor dirt | todo |

## Snap kit: walls (36)

- **Make**: 2 orientations × intact/cracked/breached/rubble; cut-away (knee-high) variant
- **Source**: Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials

| id | name | status |
| :--- | :--- | :--- |
| `wall_siding` | wall siding | done |
| `wall_siding_window` | wall siding window | done |
| `wall_siding_door` | wall siding door | done |
| `wall_siding_double_door` | wall siding double door | todo |
| `wall_siding_broken` | wall siding broken | todo |
| `wall_siding_boarded` | wall siding boarded | todo |
| `wall_brick` | wall brick | done |
| `wall_brick_window` | wall brick window | done |
| `wall_brick_door` | wall brick door | done |
| `wall_brick_double_door` | wall brick double door | todo |
| `wall_brick_broken` | wall brick broken | todo |
| `wall_brick_boarded` | wall brick boarded | todo |
| `wall_plaster` | wall plaster | todo |
| `wall_plaster_window` | wall plaster window | todo |
| `wall_plaster_door` | wall plaster door | todo |
| `wall_plaster_double_door` | wall plaster double door | todo |
| `wall_plaster_broken` | wall plaster broken | todo |
| `wall_plaster_boarded` | wall plaster boarded | todo |
| `wall_concrete` | wall concrete | todo |
| `wall_concrete_window` | wall concrete window | todo |
| `wall_concrete_door` | wall concrete door | todo |
| `wall_concrete_double_door` | wall concrete double door | todo |
| `wall_concrete_broken` | wall concrete broken | todo |
| `wall_concrete_boarded` | wall concrete boarded | todo |
| `wall_wood_log` | wall wood log | todo |
| `wall_wood_log_window` | wall wood log window | todo |
| `wall_wood_log_door` | wall wood log door | todo |
| `wall_wood_log_double_door` | wall wood log double door | todo |
| `wall_wood_log_broken` | wall wood log broken | todo |
| `wall_wood_log_boarded` | wall wood log boarded | todo |
| `wall_scrap` | wall scrap | todo |
| `wall_scrap_window` | wall scrap window | todo |
| `wall_scrap_door` | wall scrap door | todo |
| `wall_scrap_double_door` | wall scrap double door | todo |
| `wall_scrap_broken` | wall scrap broken | todo |
| `wall_scrap_boarded` | wall scrap boarded | todo |

## Snap kit: openings (8)

- **Make**: 2 orientations × open/closed/broken
- **Source**: Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials

| id | name | status |
| :--- | :--- | :--- |
| `door_wood` | door wood | todo |
| `door_metal` | door metal | todo |
| `door_glass` | door glass | todo |
| `garage_door` | garage door | todo |
| `window_glass` | window glass | todo |
| `window_broken` | window broken | todo |
| `window_boarded` | window boarded | todo |
| `curtains` | curtains | todo |

## Snap kit: roofs (7)

- **Make**: pieces per material (shingle, tin, tar), intact/sagging/collapsed
- **Source**: Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials

| id | name | status |
| :--- | :--- | :--- |
| `roof_gable_slope` | roof gable slope | done |
| `roof_ridge` | roof ridge | todo |
| `roof_hip_corner` | roof hip corner | todo |
| `roof_flat_parapet` | roof flat parapet | todo |
| `roof_edge` | roof edge | todo |
| `roof_tin` | roof tin | todo |
| `roof_broken` | roof broken | todo |

## Snap kit: structure (9)

- **Make**: 2 orientations, intact/damaged
- **Source**: Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials

| id | name | status |
| :--- | :--- | :--- |
| `stairs` | stairs | todo |
| `railing` | railing | todo |
| `porch_post` | porch post | todo |
| `column` | column | todo |
| `chimney` | chimney | todo |
| `fence_picket` | fence picket | todo |
| `fence_chainlink` | fence chainlink | done |
| `gate` | gate | todo |
| `hedge` | hedge | todo |

## Furniture (34)

- **Make**: 4 rotations, intact/looted/broken; container open/closed
- **Source**: Kenney Furniture Kit (CC0) as base → gritty re-texture, damaged variant

| id | name | status |
| :--- | :--- | :--- |
| `bed` | bed | done |
| `wardrobe` | wardrobe | done |
| `dresser` | dresser | todo |
| `nightstand` | nightstand | todo |
| `fridge` | fridge | done |
| `stove` | stove | done |
| `kitchen_counter` | kitchen counter | todo |
| `sink` | sink | done |
| `toilet` | toilet | done |
| `bathtub` | bathtub | todo |
| `shower` | shower | todo |
| `sofa` | sofa | done |
| `armchair` | armchair | todo |
| `tv` | tv | todo |
| `dining_table` | dining table | done |
| `chair` | chair | done |
| `desk` | desk | todo |
| `bookshelf` | bookshelf | todo |
| `cupboard` | cupboard | done |
| `washing_machine` | washing machine | todo |
| `workbench` | workbench | todo |
| `toolbox` | toolbox | todo |
| `safe` | safe | todo |
| `gun_cabinet` | gun cabinet | todo |
| `shop_shelf` | shop shelf | todo |
| `shop_counter` | shop counter | todo |
| `cash_register` | cash register | todo |
| `freezer_chest` | freezer chest | todo |
| `hospital_bed` | hospital bed | todo |
| `locker` | locker | todo |
| `filing_cabinet` | filing cabinet | todo |
| `crib` | crib | todo |
| `piano` | piano | todo |
| `fireplace` | fireplace | todo |

## Roads & streets (19)

- **Make**: tiles / decals / props following the road ribbons
- **Source**: CC0 texture (ambientCG / Poly Haven) → tileable 2:1 diamond in Blender, tinted to the palette

| id | name | status |
| :--- | :--- | :--- |
| `asphalt` | asphalt | done |
| `asphalt_cracked` | asphalt cracked | todo |
| `lane_line_white` | lane line white | done |
| `lane_line_yellow` | lane line yellow | todo |
| `kerb` | kerb | done |
| `sidewalk` | sidewalk | done |
| `crosswalk` | crosswalk | done |
| `driveway` | driveway | todo |
| `dirt_track` | dirt track | todo |
| `gravel_shoulder` | gravel shoulder | todo |
| `manhole` | manhole | todo |
| `storm_drain` | storm drain | todo |
| `pothole` | pothole | todo |
| `oil_stain` | oil stain | todo |
| `parking_bay` | parking bay | todo |
| `road_barrier` | road barrier | todo |
| `traffic_light` | traffic light | todo |
| `stop_sign` | stop sign | todo |
| `street_sign` | street sign | todo |

## Characters: body & clothing (13)

- **Make**: 8 directions × idle 8f, walk 8f, run 8f, attack 6f, hit 4f, die 8f @12 fps, recolourable layers
- **Source**: MPFB 2 body (CC0 assets) or the UE5 mannequin + CC0 Quaternius Universal Animation Library (Mixamo only privately) → render

| id | name | status |
| :--- | :--- | :--- |
| `body_base` | body base | done |
| `head_base` | head base | todo |
| `torso_shirt` | torso shirt | todo |
| `torso_jacket` | torso jacket | todo |
| `torso_hoodie` | torso hoodie | todo |
| `legs_jeans` | legs jeans | todo |
| `legs_cargo` | legs cargo | todo |
| `feet_boots` | feet boots | todo |
| `feet_sneakers` | feet sneakers | todo |
| `headwear_cap` | headwear cap | todo |
| `headwear_helmet` | headwear helmet | todo |
| `backpack` | backpack | todo |
| `face_mask` | face mask | todo |

## Decals (11)

- **Make**: 3–6 variants each, alpha, fade over time
- **Source**: Gemini 2D or Blender bake, alpha decal on the ground plane

| id | name | status |
| :--- | :--- | :--- |
| `blood_splat_small` | blood splat small | done |
| `blood_drip_wall` | blood drip wall | todo |
| `crack_ground` | crack ground | done |
| `leaves` | leaves | done |
| `litter` | litter | todo |
| `puddle` | puddle | todo |
| `graffiti` | graffiti | todo |
| `poster` | poster | todo |
| `scorch_mark` | scorch mark | todo |
| `tyre_tracks` | tyre tracks | todo |
| `footprints` | footprints | todo |

## FX (11)

- **Make**: frame sequences, additive where light
- **Source**: Blender particles / simulation rendered to frames, or hand-painted sprite frames

| id | name | status |
| :--- | :--- | :--- |
| `muzzle_flash` | muzzle flash | todo |
| `bullet_impact` | bullet impact | todo |
| `smoke_puff` | smoke puff | todo |
| `fire_loop` | fire loop | todo |
| `explosion` | explosion | todo |
| `dust_burst` | dust burst | todo |
| `glass_shatter` | glass shatter | todo |
| `sparks` | sparks | todo |
| `rain` | rain | todo |
| `snow` | snow | todo |
| `fog_layer` | fog layer | todo |

## Lights (5)

- **Make**: light sprites for the night pass
- **Source**: Blender particles / simulation rendered to frames, or hand-painted sprite frames

| id | name | status |
| :--- | :--- | :--- |
| `street_light_pool` | street light pool | todo |
| `window_glow` | window glow | todo |
| `headlight_cone` | headlight cone | todo |
| `torch_pool` | torch pool | todo |
| `fire_glow` | fire glow | todo |
