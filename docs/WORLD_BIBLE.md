# World 4 Bro's: World Bible and Build Plan
Draft 1 for the Antigravity build. Every name, sprite and sound must be original or CC0.

## How to use this document
- Each entry has a `snake_case` ID. Turn every ID into one data definition (a JSON or TypeScript object) so new content never needs engine changes.
- **Does** is the gameplay rule. **FX** is the animation, particles, sound and screen effects.
- Build every object with the same five states: `idle`, `hover`, `hit`, `destroy`, `respawn` (see section 9).
- Rarity: **C** common, **U** uncommon, **R** rare, **X** jackpot. Tags like `[night]` or `[multi]` mark special rules.
- Save this file as `docs/WORLD_BIBLE.md` in the repo so the coding agent can read it.

---

## 1. Maps and biomes
The world is one seeded map. Biomes blend using noise and a splat map. Each has a palette, a hazard and a landmark.
- **Bro Meadow (spawn)**. Warm greens, gentle hills. No hazard by day. Starter pine, berry bushes, a crooked welcome sign. Teaches chop, craft and campfire. Ambient: butterflies, pollen sparkle, birdsong.
- **Pinewood Thicket**. Dark teal and fog. Easy to get lost, glowing eyes at night. Wood-rich, hollow logs, hanging fireflies. Ambient: light shafts, falling needles, creaking trunks.
- **Squelchy Bog**. Olive and purple. Mud slows you 40%. Bouncy mushrooms, frog choir, gas bubbles that pop, glowing spores at night. Ambient: ripple rings, bubble pops.
- **Rockbottom Ridge**. Grey-blue cliffs. Rockfall after a loud honk. Flint, stone, gold veins, crystals that light the dark. Ambient: echoes, dust motes, skittering pebbles.
- **Sunburn Flats (desert)**. Orange with a heat-haze shader. Heat meter and thirst. Cacti, tumbleweeds, a fake oasis mirage that vanishes when you get close. Ambient: dust devils, shimmer.
- **Brrington Tundra**. White-blue. Cold meter and slippery ice. Snowdrifts, a frozen lake for skating, ice fishing holes, snowman watchers. Ambient: gusts, breath puffs, aurora at night.
- **Scrapyard Gulch (gadget biome)**. Rust and neon junk. Spring traps and conveyor belts. Source of gadget parts, vending machines, trampolines. Ambient: sparks, flickering signs, distant clanks.
- **The Hollow (nightmare)**. Purple-black. Sanity drain and whispers. Dead trees, hanging lanterns, ghost sheets, the best rare loot. Ambient: chromatic wobble, layered whispers, cold fog.
- **Splash Cove (beach and lake)**. Turquoise and sand. Deep water drowns you, goose gangs roam. Fishing, crabs, a rubber duck flotilla, a pier. Ambient: foam edges, gulls, waves.
- **Mount Grumbleguts (volcano, endgame)**. Black rock, orange glow. Lava and burning ash. Obsidian, ember fruit, the boss arena. Ambient: embers, rumble, pulsing glow.

### Proposed biomes (design of 2026-10-19, not built yet)
Six proposals that bring in acid, fire, electricity, glare, crystal and city hazards. Full rows, palettes and effects are in Part 18 (§119).
- **Toxic Marsh**. Dark olive and acid green. Acid pools and acid rain. Melted trees, leaking barrels. Ambient: green mist, bubbling, sizzles. Zombie: Melter (proposed).
- **Cinder Wastes**. Ash grey with ember cracks. Burning ground that spreads. Charred stumps, ash drifts. Ambient: falling ash, glowing cracks. Zombie: Ember zombie.
- **Storm Plains**. Slate sky over olive grass. Lightning strikes and live pylons. Pylons, fallen cables, tall grass. Ambient: thunder, rain, wind. Zombie: Static (proposed).
- **Salt Flats**. Bone white. Blinding glare and mirages. Salt crystals, dry boats. Ambient: heat shimmer, wind hiss. Zombie: Husk zombie.
- **Crystal Caverns**. Deep blue with violet crystals. Dark; loud noise shatters crystals. Glow crystals, echo rooms. Ambient: hum, drips, echoes. Zombie: Glimmer (proposed).
- **Ruined Downtown**. Charcoal streets. Gas leaks and live wires. Wrecks, barriers, cracked streets. Ambient: sirens, flickering lights. Zombies: Walker, sapper.

### Special maps and instances
- **Bro Basecamp**. A player-claimed zone for walls, chests, a tent and a firepit ring that creates a safe light zone shared by the party.
- **Echo Caves**. Underground, torch-only. Echoing footsteps, crystals, hidden loot, a gong room.
- **Paper Cutout Plateau**. A pop-up-book area: flat cardboard scenery folds upright as you approach. Home of secrets and the `ladder_to_nowhere`.
- **Bro-lympics Arena**. Party minigames: egg-and-spoon race, log toss, honk contest. Confetti and a trophy hat.
- **Moon Pond (secret, night only)**. Appears under a `cheese_moon`. Fishing here gives rare cosmetics. Lilies glow and the water mirrors the stars.

---

## 2. Natural world objects
- **pine_tree (C)**. Does: 3 chops, 2 logs, 30% pine cone for replanting. FX: shake per hit, wood chips, falls away from you with one bounce, leaves a stump that regrows with a pop-up scale. Thud and leaf burst.
- **oak_tree (U)**. Does: 5 chops, acorns. A squirrel may drop out and steal one item. FX: canopy rustle, acorn rain, squirrel scramble.
- **dead_tree (C)**. Does: 1 chop, twigs, 10% chance of a ghost sheet. FX: creaky sway, dry crack, dust puff.
- **berry_bush (C)**. Does: pick 3 berries, regrows in a day. FX: bush squish, berries pop and arc into the bag, bare twigs until regrowth.
- **grass_tuft and twig_sapling (C)**. Does: harvest for grass and twigs. FX: wind sway from shared noise, sparkle on pick, grow back with a tiny pop.
- **wild_carrot (C)**. Does: pull for food. FX: leaves wiggle before the pull, a squeak, flies in an arc.
- **boulder (C)**. Does: needs a pickaxe, 4 hits, stone and flint. FX: sparks, chips, a crack decal that grows each hit, rubble puff, low boom.
- **gold_vein (U)**. Does: 6 hits, gold nuggets. FX: glint sweep every few seconds, coin ping, faint glow at night.
- **crystal_cluster (U)**. Does: emits light, mine for glow shards (lamps). FX: slow hue shift, hum that rises as you approach, shatter into sparkles.
- **glow_mushroom (U)**. Does: eat for sanity, doubles as a night light. FX: pulsing glow, spore trail.
- **buzz_hive (U)**. Does: honey, but bees chase you unless a fire is near. FX: swarm particle cloud, rising hum, smoke calms them.
- **cactus (C)**. Does: hurts on touch, chop for water pulp. FX: wobble, spine particles, water droplets.
- **cattail_reeds (C)**. Does: fiber for rope. FX: sway, fluff puffs as you walk through.
- **fishing_pond (C)**. Does: fish with a rod. FX: ripple rings, bobber plop, tug animation, fish arcs into your hand. Lily pads and frogs that hop away.
- **ice_lake (U)**. Does: zero-friction skating, cracks under a loud honk. FX: scraping skate sound, crack lines, snow spray.
- **lava_vent (R)**. Does: erupts on a timer after a glow warning. FX: ember fountain, heat distortion, ground rumble.
- **campfire (C)**. Does: light, warmth and cooking. Burns down, needs logs. FX: layered flicker, sparks, smoke column, warm bloom, crackle. Steam hiss when it dies.

---

## 3. Goofy world objects
- **banana_peel_patch (C)**. Does: step on it and slide 3 metres. Also a placeable trap that stuns enemies. FX: spin with squash and stretch, cartoon whoosh, stars circling the head.
- **springy_stump (U)**. Does: launches you high to reach ledges. FX: squash, boing, camera zooms out in flight, dust ring on landing.
- **whoopee_boulder (U)**. Does: push it and it lets out a loud pbbbt that scares birds and the lurker for 5 s. FX: wobble, green puff, nearby critters freeze or flee.
- **giant_duck_statue (U)**. Does: squeaks when smacked and hands out a rubber duck. Answers your honk (H) with a honk chorus. FX: squash on hit, feather puff, sound ripple rings.
- **old_stuffins (C)**. Does: a talking scarecrow who gives terrible advice and occasionally a real hint. Swap his hat for a cosmetic reward. FX: gibberish speech blips, head wobble, straw puffs.
- **vending_machine_from_nowhere (R)**. Does: insert 3 gold nuggets for a random goofy item. It may eat the coins and spit out one sock. FX: shakes when smacked, blinking neon, clunk drop, a tiny fire if kicked too much.
- **suspicious_bush (C)**. Does: creeps toward you only while the camera is not facing it, so rotate with Q/E to catch it. Drops berries or a hat when caught. FX: eyes blink, leaves rustle, freeze pose when seen.
- **garden_gnome_squad (U)**. Does: gnomes turn to watch you. Find all ten for a reward (one is golden). FX: head-turn tween, hat jiggle, confetti pop on collect.
- **flush_teleporter (U)**. Does: enter a green outhouse, hear a flush, pop out of another one. FX: door slam, swirl particles, spin-out landing, cheeky jingle.
- **disco_stump (U) [multi]**. Does: dance (G) near it and every nearby player auto-dances and regains sanity. FX: disco ball, glowing floor tiles, beat-synced bounce, light beams.
- **gong_of_regret (U)**. Does: smack it to ping all teammates on the map, wake sleeping creatures and shake loose rocks. FX: shockwave ring, camera shake, long ring-out.
- **fake_campfire (C) [night]**. Does: looks like a campfire but gives no warmth. Smack it for confetti and a gotcha sting. FX: slightly wrong flicker colour, confetti pop, dramatic horn.
- **sock_tree (C)**. Does: drops mismatched socks, a craftable warmth item. FX: socks flutter down like leaves and wiggle on the branches.
- **leaning_log_tower (U)**. Does: chop the wrong log and the whole stack rolls over, stunning anyone below. FX: wobble warning, rumble, logs bounce and roll with dust, comedic crash.
- **wishing_well_of_lies (R)**. Does: throw an item in, get a joke fortune and a random item back (50% worse). FX: echoey plop, glowing ripples, a hammy narrator voice.
- **mystery_crate_pinata (U)**. Does: hit it for random loot. FX: candy colours, squash on each hit, confetti burst with items arcing out.
- **tumbleweed_buddy (C)**. Does: rolls past with googly eyes. Catch it for a random item. FX: bouncing roll, eyes track you, tiny squeak when caught.
- **ladder_to_nowhere (R)**. Does: climb it to reveal a tiny hidden platform with a chest. FX: the pop-up world folds open at the top, trumpet fanfare.
- **snowman_stalker (U)**. Does: turns its head to follow you. Look away and it creeps closer, then plops a bucket on your head (vision blocked for 4 s). FX: snow crunch, head swivel, clang and muffled screen.
- **clam_of_chance (U)**. Does: a giant beach clam that snaps. Time your grab for a pearl, mistime for a harmless pinch. FX: yawn animation, shine, comic bite with stars.
- **mailbox_with_attitude (C)**. Does: holds a random letter or hint and bites 1 time in 5. FX: flag flip, flap chatter, paper confetti.

---

## 4. Goofy gadgets and items
Core tools also needed: axe, pickaxe, shovel, fishing rod, spear, torch, lantern, each with its own swing animation and durability bar.
- **rubber_chicken (C)**. Does: weapon with low damage and huge knockback. Scares small enemies and never breaks. FX: squeak with randomized pitch, stretchy swing, feathers on hit.
- **mega_mallet (U)**. Does: slow heavy hit that dizzies enemies. FX: bonk text, spinning stars over the target, ground crack, hit-stop.
- **boing_boots (U)**. Does: jump higher and bounce on landing. FX: squeak per step, squash and stretch, spring recoil.
- **propeller_beanie (U)**. Does: glide for 5 s, then a cooldown. FX: spinning blades, whirr, leaves swirling below.
- **air_horn (U)**. Does: upgrade of honk (H). Stuns nearby creatures, wakes sleepers, calls teammates. FX: sound-wave rings, camera punch, birds scatter.
- **whoopee_cushion_trap (C)**. Does: placed trap. The enemy sits, is stunned and embarrassed. FX: the sound, red-face particles, sweat drops.
- **confetti_cannon (U)**. Does: scares night creatures for 6 s and restores sanity. FX: muzzle puff, confetti shower, party horn.
- **bean_burrito_booster (R)**. Does: short dash that costs hunger and leaves a green cloud that repels enemies. FX: speed lines, puff trail, comedic rocket sound.
- **umbrella_glider (U)**. Does: slow fall, blocks rain and the rain_of_frogs. FX: open pop, canopy flutter, raindrops bouncing off.
- **bucket_helmet (C)**. Does: absorbs one hit, then dents. FX: clang, dent decal, visible wobble.
- **disguise_kit (R)**. Does: a fake mustache. Creatures ignore you for 10 s, except the goose. FX: smoke puff on equip, twitching mustache.
- **googly_eyes (C)**. Does: stick on any object. Objects with eyes restore nearby sanity. FX: tracking eyes, wobbly pupils, sticker pop.
- **bubble_wand (U)**. Does: traps a small creature or floats an item. FX: iridescent bubble shader, sparkle pop.
- **dancing_lantern (U)**. Does: a light that bobs to the beat when you dance and restores sanity. FX: swaying light, musical notes, colour cycle.

---

## 5. Food and cooking with side effects
Cooked in the `bubblebro_cauldron`, whose bubbles change colour by recipe.
- **hiccup_berries**: restores hunger, and your character hiccups and hops slightly for 30 s.
- **giggle_mushroom**: laugh bubbles, wobbly steering, sanity up.
- **chili_of_regret**: big hunger and warmth, then 10 s of fire-breath puffs that scare enemies.
- **mystery_stew**: a random effect, sometimes a tiny voice or a giant head (cosmetic).
- **pancake_stack**: high hunger and sanity. The stack wobbles while carried.
- **rainbow_soup**: your sweat glows in colours and gives a light radius for 60 s.
- **bubble_tea_of_buoyancy**: float on water for 30 s.
- **crispy_sock_chips**: stamina boost, absurd crunch sound.
- **roast_sausage_stick**: simple fire food with a sizzle.

---

## 6. Cosmetics
Hats and wearables with their own animation: `traffic_cone_hat` (knocked off by hits), `cheese_hat` (attracts the goose), `flower_crown` (petals when dancing), `tiny_top_hat` (tips when you honk), `pot_helmet` (clangs), `shark_slippers` (flap), `cape_of_dramatic_wind` (flaps with speed). Also face stickers, name colours, emote packs and trail effects (sparkles, bubbles, hearts) so the four bros look distinct.

---

## 7. Creatures
- **lurker [night]**: the thing in the title. Invisible beyond the light, glowing eyes at the edge. States: stalk, lunge, retreat. Hates light, fears the whoopee_boulder.
- **honk_goose (U)**: steals an item and runs. Honk back to make it flee. Loves the cheese_hat.
- **hop_bun (C)**: rabbit-like, flees, drops food. Ears flick.
- **pudge_pigeon (C)**: too round to fly well, waddles, drops feathers, steals crumbs.
- **gloomblob [night]**: a puddle-like blob in The Hollow that slides toward you, shrinks in light, drains sanity.
- **pillow_bandit [night]**: hops onto your head and mumbles, draining sanity. Shake it off with Space.
- **waddle_crab (C)**: sideways walker on the beach that pinches.
- **dune_noodle (U)**: a long wiggly worm that pops out of the sand and trips you. Harmless splash.
- **pocket_sloth (pet)**: slow follower that carries a few items and hangs upside down from your pack.
- **quackzilla (R, boss)**: a giant rubber duck at the volcano. Squeaky stomps, three phases, drops a golden duck.

---

## 8. World events
- **rain_of_frogs**: frogs fall and hop. Umbrellas block them. Frog legs become food.
- **cheese_moon**: yellow moon, cheddar-coloured light, mice appear, the Moon Pond opens.
- **goose_invasion**: waves of honk_geese. Honk to defend and collect loot at the end.
- **traveling_salesbro**: a merchant on a wobbly unicycle who trades odd items for gold.
- **rubber_duck_meteor_shower**: squeaky meteors drop ducks. Craters become ponds.
- **dance_fever**: music starts and everyone must dance or lose sanity. Party reward.
- **fog_horn_morning**: thick fog and a low horn. Visibility is cut, good for ambushes.
- **static_storm (Scrapyard)**: gadgets go haywire, conveyors reverse, lights flicker.

---

## 9. Effect recipe for every object
Five states: `idle` (ambient sway or glow), `hover` (outline plus prompt), `hit` (shake, 40 to 60 ms hit-stop, particles, sound), `destroy` (fall, burst, drops that arc to the player), `respawn` (pop-up scale with `Back.easeOut`).
- **Particle presets**: `wood_chip`, `leaf_burst`, `dust_ring`, `spark`, `star_bonk`, `confetti`, `heart`, `sweat_drop`, `fart_cloud` (green puff), `bubble`, `feather`, `snow_puff`, `ember`, `spore`, `coin_glint`, `ripple_ring`, `sound_ring`.
- **Screen effects**: camera shake (scaled by a reduced-motion setting), zoom punch, chromatic pulse for sanity hits, dusk vignette, slow-mo on boss defeat, lightning flash.
- **Status effects**: acid, burn, explosion, stagger, electricity, cold, spores, goo, wet, oil slick, blind and smoke are data rows with their own reaction table. See Part 18 (§115–§118).
- **Audio rules**: 3 to 5 variants per sound, pitch randomised by about 8%, volume by distance, a rising hum for anything dangerous, a musical sting for rare finds. Layered music for day, dusk, night and danger with crossfades, plus short silly stingers.

---

## 10. What else you need
### Gameplay systems
- Inventory with a hotbar, stacks, drag-and-drop and equipment slots (hand, head, body, back)
- Crafting tech tree with stations (a brainstormer unlock machine, workbench, cauldron, drying rack)
- Chests and base building: walls, doors, a tent for sleeping, a claim radius
- Seasons (autumn, winter, spring, summer) and weather (rain, snow, fog, storms)
- Farming plots, fishing, hunting, drying food, and spoilage with an icon
- A hint system (the firefly guide), day goals and achievements
- Death rules: ghost mode that can haunt props, a revive item, a respawn penalty, a stats screen
- A boss ladder and difficulty settings
- Map, minimap, teammate pings, an emote wheel and quick chat
- Pets, trading and a friendly-fire toggle

### Multiplayer and engineering
- Data registries for items, objects, recipes, biomes, particles, sounds and events
- Seeded deterministic world generation with chunk streaming
- An authoritative realtime server (PartyKit, Colyseus or similar) with room codes. Vercel hosts the site, not the sockets.
- Save and load, autosave, world seeds
- Chat moderation: profanity filter, mute and report, name rules. Chat bubbles reach strangers.
- Reconnect handling and basic anti-cheat. The server validates harvests, crafts and damage.
- A debug panel: spawn any ID, set the time, jump to a biome, toggle night, show colliders and FPS
- Tests (unit tests for the rules, a Playwright smoke test), CI, Vercel deploys, error reporting
- An asset pipeline: glTF with Draco, KTX2 textures, atlas packing, audio sprites
- Performance budgets: instancing, culling, pooled particles, adaptive quality

---

## 11. Part 2: Zombie mode overview
Expansion pack. The maps and biomes in section 1 do not change. Roads, points of interest, cars and zombies are new layers on top. The same ID, Does and FX rules apply.
- **Premise**: the bros wake up after a goofy outbreak. Survive, scavenge, fix a car, drive across the biomes and escape. The tone is cartoon slapstick: zombies pop into green puffs with a silly sound. No blood or gore, so it stays playable by everyone.
- **Escape goal**: build a working car and reach the `evac_dock` at Splash Cove. An endless mode stays available for people who only want to survive.
- **Everything from Part 1 stays**: hunger, health, sanity, crafting, goofy objects, the lurker, day and night. Zombies add pressure on top of the old loop.
- **Mode switch**: one config flag (`mode: classic` or `zombie`) turns on zombie spawns, high loot, roads and vehicles. Classic mode must keep working.

---

## 12. Roads and the road network
Roads are generated from the world seed after the biomes exist. Connect points of interest with a minimum spanning tree plus a few extra loops, route with A* that avoids water and cliffs, smooth into splines, paint them into the terrain splat map, then place props along the edges. Biomes never move.

### Road types
- `highway` (Sunburn Flats): fast, wide, long straights, wrecks and roadblocks.
- `country_lane` (Bro Meadow): winding, hedges, farms.
- `logging_track` (Pinewood Thicket): dirt, slow, hides loot.
- `boardwalk_causeway` (Squelchy Bog): planks over mud that creak, with missing planks to patch using `plank_bundle`.
- `switchback_pass` (Rockbottom Ridge): mountain hairpins and a pitch-dark `rock_tunnel`.
- `ice_road` (Brrington Tundra): black ice, low grip, long slides.
- `service_road` (Scrapyard Gulch): conveyors cross the road and shove cars sideways.
- `cursed_road` (The Hollow): fog, cracked asphalt, ghostly headlight reflections.
- `coast_road` and `pier` (Splash Cove): tide splashes, sand drifts.
- `lava_bridge` (Mount Grumbleguts): cooled lava spans with glowing cracks.

### Road pieces and props
`junction`, `roundabout_of_doom` (zombies circle it), `toll_booth` (loot, and the barrier arm bonks you), `railway_crossing` (train event), `bridge` (a choke point that wrecks can block), `tunnel`, `road_sign` (goofy directions that sometimes lie), `guardrail`, `traffic_cone` (also a zombie hat and a noise maker), `road_barrier`, `billboard_ad` (original fake brands), `parked_wreck` (loot and part source).

### Road rules
- **Does**: roads give on-foot players +10% speed and full grip to cars. Off-road costs grip and fuel.
- **Hazards**: oil slicks, potholes, spike strips, fallen trees (clear with an axe), roadblocks (clear with a tow, a crowbar, or by driving through at the cost of car health).
- **Risk**: zombies cluster at wrecks and junctions, so roads are faster but riskier.
- **FX**: tyre-mark decals, dust trails on dirt, sparks on guardrail scrapes, puddle splashes in rain, headlight cones in fog.

---

## 13. Points of interest along the roads
POIs sit beside roads inside the existing biomes. Each has a loot table, a zombie density and one gimmick.
- `gas_station` (highway, lanes): fuel, snacks, tyre kits. Medium zombies. Touching a pump trips an alarm that pulls a mini horde.
- `motel_no_vacancy` (desert, coast): beds for safe sleep, vending machines, a key rack with car keys. Low zombies, surprises behind doors.
- `auto_garage` (Scrapyard, lanes): the best source of car parts and tools, plus a `car_lift` that speeds up repairs. Medium zombies.
- `police_station` (Meadow edge): flares, roadblock gear, locked cells with loot. High zombies. Tripping the alarm sets off a siren.
- `farmhouse` (Meadow, Thicket): food, jerrycans, tools, a tractor. Low zombies, barn hordes.
- `supermarket` (Meadow): lots of food and shopping carts. High zombies, aisle ambushes.
- `hospital_ruins` (Hollow edge): antidote gel and bandages. High zombies and sanity drain.
- `logging_camp` (Thicket): axes, planks, a loud chainsaw.
- `radio_tower` (Ridge): the rescue radio quest, signal flares. Climb it to reveal the map.
- `drive_in_cinema` (Flats): the film sound lures zombies, loot sits in the booths.
- `ice_cream_depot` (Tundra): an ice cream truck wreck and frozen loot.
- `checkpoint_barricade` (Tundra, Ridge): cartoon crates with high loot and high risk.
- `lighthouse_diner` (Cove): food and the boat key.
- `volcano_scrap_vault` (Grumbleguts): jackpot loot guarded by a boss.
- `evac_dock` (Cove): the escape point. Arriving starts a final defend-the-engine event.

---

## 14. Dumb cars: the vehicle roster
All cars are goofy, arcade-handled and original (no real brands). Each has seats, trunk slots, a top speed, a fuel use, a quirk and a list of required parts.
- `rusty_hatchback` (C): 4 seats, 8 trunk slots, average everything. Quirk: backfires a puff that startles zombies. Honk is a sad trombone.
- `clown_car` (U): 6 seats, 4 slots. Quirk: more passengers than seats appear, a bouncy ride, a bike-horn honk, and zombies it bumps burst into confetti. Needs a `big_red_nose` as its key.
- `ice_cream_truck` (U): 3 seats, 20 slots, slow. Quirk: its looping jingle pulls zombies in like a magnet, so you can lure hordes on purpose. The freezer stops food spoiling.
- `golf_cart_deluxe` (C): 2 seats, very quiet, low fuel use. Quirk: crawls up hills at a hilarious speed. Great for sneaking.
- `bathtub_buggy` (R): 1 seat, floats like a boat. Quirk: leaves a bubble trail and a soap slick.
- `lawn_tractor_racer` (C): 1 seat, tows a trailer, strong off-road grip. Quirk: lower the blades to clear grass and light zombies with cartoon puffs.
- `school_bus_of_chaos` (R): 12 seats, 30 slots, a rolling base. Slow turning. Quirk: the stop-sign arm swings out and bonks zombies.
- `hot_dog_van` (U): 3 seats, 16 slots, a griddle for cooking on the move. Quirk: the sizzle and smell cloud attract zombies and the goose.
- `shopping_cart_runner` (C): 1 seat, no engine, no fuel. Coast downhill. Quirk: a wobbly wheel that fights every turn.
- `pizza_scooter` (C): 2 seats, fast and fragile, a 6-slot delivery box. Quirk: pops a wheelie when you boost.
- `monster_truck_of_questionable_taste` (R): 2 seats, runs over wrecks and zombies, loud, thirsty. Quirk: googly eyes on the grille and a roar for a horn.
- `banana_van` (U): 4 seats. Quirk: drops banana peels behind it that slip zombies.

**Fuel**: `gas_can`, siphoned fuel (with a hose), and goofy emergency fuels listed under jank fixes. Fuel shows on the HUD, and running dry rolls the car to a stop with a cough.

---

## 15. Car repair: find and fix the parts
A car is a list of parts. Each part has a condition (0 to 100) and a repair rule. A car runs only when every required part is above 20%.

**Required parts**: `battery`, `spark_plugs`, `fuel`, `engine_block`, `ignition_key` (or hotwire with a mini-game), `tires` (4), `radiator_coolant`, `fan_belt`.
**Optional parts**: `headlights`, `horn`, `windshield`, `doors`, `bumper`, `roof_rack`, `radio`, `trunk_lock`, `seat_covers` (sanity bonus), `fuzzy_dice` (luck).

### How repair works (Does)
1. **Find a car**. Parked wrecks, garages, barns, roadsides. Each spawns with 30 to 60% of its parts missing, seeded so runs are repeatable.
2. **Inspect**. Hold on the car to open the repair panel. Every part is a card: missing, broken, damaged or fine. Missing parts appear as glowing ghost outlines in the engine bay.
3. **Find parts**. Garages, wrecks and shops. Parts come in junk, standard and shiny quality.
4. **Fit a part**. A hold-to-repair ring. Tools shorten it: `wrench`, `ratchet`, `jack`, and the garage `car_lift`.
5. **Fix damaged parts**. `duct_tape` (a patch that caps at 40%), `welding_kit`, `tire_patch`, `jump_leads` for the battery.
6. **Start the engine**. A short crank mini-game: hold, then release in the green zone. A failure floods the engine and costs time. Starting is loud and pulls zombies in.
7. **Fuel up**. Siphon or pour from a `gas_can`.
8. **Driving wears parts**. Zombie crowds dent the bumper, rough roads rattle the belt, hits crack the windshield. Smoke at 30% engine, fire at 10%. Put it out with a `fire_extinguisher` or lose the part.

**Co-op**: up to four bros repair in parallel, each on a different part, with name tags over the engine bay. Roles are free: driver, mechanic, lookout, gunner. A `mechanic_hat` makes repairs 20% faster.

### Jank fixes (the funny part)
With no proper part you can use junk. It works, but the car gains a quirk shown on its card.
- `bucket_headlight`: a narrow cone of light that swings loosely.
- `rubber_chicken_horn`: honks as a squeak and scares small enemies.
- `pancake_stack_tire`: a bouncy ride that boings over bumps, at lower top speed.
- `sock_fan_belt`: works at 50% and puffs sock-scented steam.
- `bean_burrito_fuel`: a speed boost that leaves green clouds and burns your food.
- `bucket_of_water_radiator`: works until the water runs out.
- `googly_eyes_dashboard`: sanity bonus for riders.
- `cheese_wheel_tire`: rolls oddly and attracts the goose.

**FX**: sparks, wrench and ratchet clicks, steam hiss, a ding when a part clicks in, the car shaking while it is fixed, a coughing engine and finally a triumphant roar with a camera zoom-out, a honk and a sparkle burst.

---

## 16. Driving model
- **Arcade physics, not a simulator**: acceleration, brake, handbrake drift. Grip per surface: asphalt 1.0, dirt 0.7, grass 0.65, sand 0.6, snow 0.45, mud 0.4, ice 0.2.
- **Camera**: follows behind and pulls out with speed. Q and E still rotate it, with a lock-behind option.
- **Collisions**: the car pushes light props, bounces off trees and boulders with part damage, and plows through zombies with a slowdown based on how many it hits.
- **Noise**: engine noise radius grows with speed and car type. The golf cart is quiet, the ice cream truck is loud.
- **Controls**: WASD, Space handbrake, H honk, F enter and exit. On mobile, a steering thumbstick, two pedals and a honk button.
- **Passengers**: seat swap, trunk access, and a gunner seat for window attacks.
- **Parking**: park at Basecamp to protect parts. A covered garage prevents weather damage.
- **FX**: wheel-spin dust, drift smoke, motion blur at speed, headlight cones in fog, engine hum pitched by speed, tyre squeal, suspension bounce.

---

## 17. Zombies
Zombies live in the existing biomes with tinted variants. All use the cartoon pop on defeat.
- `shambler` (C): slow, groans, the standard.
- `jogger` (U): fast, comes in packs, appears at dusk.
- `traffic_cone_zombie` (C): the cone blocks the first hit. Knock it off and the zombie topples.
- `clown_zombie` (U): squeaky honks that attract others.
- `bloater_balloon` (U): inflates and floats, then pops into a slowing cloud.
- `brute_in_a_bathrobe` (R): big and slow, stuns, headbutts cars.
- `screamer_karaoke` (R): holds a mic. Its scream calls a horde. Flares are the answer.
- `crawler_roomba` (C): low, hides in grass, trips you.
- `zombie_goose` (U): honks, bites, steals items.
- **Biome variants**: `frost_zombie` (Tundra), `husk_zombie` (Desert), `mud_zombie` (Bog), `ember_zombie` (Volcano), `ghost_sheet_zombie` (Hollow).
- `highway_hulk` (R, boss): a giant zombie stitched from wrecked cars, guarding the road to the evac dock. Drops a rare engine part.

### Zombie systems
- **Senses**: vision cone, hearing from noise, smell for food, attraction to light at night.
- **Noise events**: footsteps (sprinting is louder), engines, horns, loud tools, gongs, alarms. Show the noise radius as a ring in debug mode.
- **AI states**: wander, investigate, chase, attack, stunned, flee fire, follow horde.
- **Hordes**: size scales with the day and with noise. Every 7th night is a `horde_night` with a siren warning at dusk.
- **Infection**: bites add an infection meter. Cure it with `antidote_gel`. At 100% you turn into a `zombie_bro` that teammates can cure with a revive item.
- **Performance**: crowd instancing, a vertex-animation shader, LOD swaps, flow-field pathfinding on a grid, far zombies updating at 2 Hz, pooled spawns, and a separate active-zombie budget for phone (60) and desktop (150).

### Goofy objects meet zombies
- `banana_peel_patch`: zombies slip and slide.
- `disco_stump`: nearby zombies are forced to dance for 6 s.
- `gong_of_regret`: calls every zombie in a huge radius. A dangerous dinner bell.
- `whoopee_boulder`, `rubber_chicken` and `giant_duck_statue`: noise decoys.
- `fake_campfire`: pulls zombies in instead of warming you.
- `confetti_cannon`: stuns a crowd.

---

## 18. High loot system
The goal is that you find things all the time, so the fun is choosing what to carry. Danger and weight keep the economy fair.
- **Tiers**: Junk (grey), Common (white), Useful (green), Rare (blue), Jackpot (gold).
- **Containers**: `glove_box`, `car_trunk`, `suitcase`, `locker`, `toolbox`, `fridge`, `medicine_cabinet`, `cash_register`, `piggy_bank` (a mini jackpot), `supply_drop_crate`, plus the existing vending machine and pinata.
- **Loot density setting**: low, normal, high. High is the default: about 2x containers and +1 item per container.
- **Weight and slots**: 12 backpack slots to start, up to 24 with bags. Carrying over 100% weight slows you. Cars add trunk slots and bases add chests.
- **Risk and reward**: opening a Jackpot crate fires a light beam and a siren sting that attracts zombies for 20 s. Locked containers need a crowbar or lockpick, which makes noise.
- **Hot zones**: `police_station`, `checkpoint_barricade`, `hospital_ruins` and `volcano_scrap_vault` have 3x loot and dense zombies.
- **Restock**: containers refill every 3 days, only when no player is within 60 m. Wrecks refill parts more slowly.
- **Loot tables**: one weighted table per container and POI in data (example: garage is 50% parts, 25% tools, 15% junk, 10% rare).
- **Guarantees**: within 100 m of spawn there is always a campfire kit, a weapon and a snack. Car parts bias toward roads so the car quest stays winnable.
- **FX**: sparkle by tier, a pop sound with rising pitch, items arc into the loot panel, a confetti burst on Jackpot, container lid animations.
- **Multiplayer**: a setting for personal loot rolls (no arguments) or shared loot.

---

## 19. Weapons and defenses
Mostly melee and goofy, readable and fun. Everything has durability.
- **Melee**: crowbar, nail_bat, frying_pan, traffic_sign_axe, pool_noodle_of_justice (long reach, tiny damage, big knockback), plus the existing rubber_chicken and mega_mallet.
- **Ranged and throwables**: slingshot, water_balloon (slows), road_flare (lures or scares), firework_launcher (loud, area damage), foam_dart_blaster, cream_pie (blinds).
- **Defenses**: spike_strip, barricade_planks, sandbag, searchlight_tower, clam_trap.

---

## 20. New systems checklist for this expansion
- A mode and rules file: classic or zombie, loot density, zombie density, horde frequency, infection on or off, friendly fire, evac goal on or off.
- A seeded road generator with a debug overlay, and a POI placer with spacing rules and a reachability check. Every required part and the evac dock must be reachable.
- A vehicle entity (server-owned in multiplayer) with seats, parts, repair panel UI, fuel, damage, noise and saves.
- A zombie crowd system with instancing, flow fields and noise propagation, plus a performance budget.
- A loot system with tables, weights, tiers, restock and guarantees.
- A quest chain: the `rescue_radio` leads to parts, keys and the evac dock. The firefly guide gives hints.
- Balance: the parts for one car should be findable within about 3 days at normal density. Keep a playtest sheet for fuel, parts, zombie damage and loot rates.
- UI: repair panel, fuel and speed HUD, noise meter, infection meter, weight bar, minimap with roads and POI icons.
- Audio: engine sounds per car, horns, layered zombie groans, a horde siren, and a car radio with original goofy stations.
- Save data: part conditions, POI looted state, road seed.
- Content safety: cartoon tone by default, a gore-free setting, no real car brands, and a licence log for every asset.

---

## 21. Part 3: Master item catalog
Part 3 adds items, loot, settlements, shelters and the map-generation plan. Numbers are starting values to tune. **D** = damage, **S** = speed, **N** = noise from 0 (silent) to 5 (alarm), **Dur** = uses before it breaks. Tiers match section 18: **J** junk, **C** common, **U** useful, **R** rare, **X** jackpot.

### Melee weapons
- `stick` (J): D 5, S fast, N 1, Dur 25. The starter. Snaps with a cartoon crack.
- `stale_baguette` (J): D 6, S fast, N 0, Dur 20. Breaks into crumbs that pigeons chase.
- `nail_bat` (C): D 14, S medium, N 2, Dur 60. Nails glint, thwack sound.
- `crowbar` (C): D 12, S medium, N 2, Dur 80. Also opens locked containers and clears roadblocks.
- `frying_pan` (C): D 10, S medium, N 3, Dur 70. A bong sound and a 15% stun chance.
- `hockey_stick` (C): D 11, S medium, N 2, Dur 60. Long reach.
- `pool_noodle_of_justice` (C): D 2, S fast, N 0, Dur 999. Double reach, huge knockback, a boing on hit.
- `traffic_sign_axe` (U): D 20, S slow, N 3, Dur 50. The sign shows a random goofy symbol.
- `fire_axe` (R): D 28, S medium, N 3, Dur 80. Chops doors, trees and zombies.
- `chainsaw_of_regret` (R): D 35 over time, N 5, burns fuel. Wobbles in your hands, zombies pop in green puffs.
- Existing: `rubber_chicken`, `mega_mallet`.

### Ranged weapons
- `slingshot` (C): D 8, N 1, ammo pebbles.
- `bow_of_branches` (C): D 12, N 1, slow draw, arrows can be picked up again.
- `foam_dart_blaster` (U): D 6, rapid fire, N 1, ammo foam_darts (recoverable).
- `crossbow_of_corks` (U): D 18, N 1, slow reload, ammo corks (reusable).
- `squirt_cannon` (U): D 0, slows and soaks, puts out fires, refills at ponds.
- `plunger_launcher` (U): sticks a zombie to a wall for 4 s.
- `potato_cannon` (U): D 15, N 3, ammo potatoes (which are also food).
- `flare_gun` (U): D 5 plus burn, lights an area, lures or scares.
- `nail_gun_of_justice` (R): D 10, rapid fire, N 3, ammo nails.
- `firework_launcher` (R): D 40 in an area, N 5, ammo bottle_rockets.
- Existing: `confetti_cannon`, `bubble_wand`.

### Throwables
`water_balloon` (J, slows), `cream_pie` (C, blinds 4 s), `road_flare` (C, light, lures or scares), `marbles` (C, scatter and slip), `banana_peel` (C, slip), `alarm_clock_decoy` (C, rings after 5 s and pulls zombies), `stink_sock_bomb` (U, zombies stagger away), `smoke_pot` (U, cover), `boombox_decoy` (U, plays music and pulls a horde), `bowling_ball` (U, heavy, strikes zombies like pins), `glitter_bomb` (R, blinds and marks a crowd).

### Traps and defenses
`tripwire_cans` (C, noise alert), `spike_strip` (C), `barricade_planks` (C), `sandbag` (C), `shopping_cart_barrier` (C, blocks roads and doors), `whoopee_cushion_trap` (C), `barbed_wire_tangle` (U), `sprinkler_slip_n_slide` (U, soaks and slides zombies), `clam_trap` (U), `scarecrow_decoy` (U), `searchlight_tower` (R), `foam_turret` (R, auto-fires darts, needs power), `flame_pit` (R).

### Armor and wearables
- `bucket_helmet` (C): absorbs 1 hit, then dents.
- `bike_helmet` (C): less crash damage.
- `pot_helmet` (C): +10% head protection, clangs.
- `pillow_vest` (C): -20% bite chance, feathers burst out when hit.
- `cardboard_armor` (J): light protection, weakens when soaked by rain.
- `hockey_pads` (U): -20% melee damage taken.
- `rain_poncho` (C), `winter_coat` (U), `sun_hat` (C), `snow_boots` (C), `swim_floaties` (C, float): weather gear tied to the biome meters.
- `hi_vis_vest` (C): teammates show brighter on your minimap.
- `fanny_pack_of_holding` (R): +4 quick slots.
- Backpacks: `school_backpack` (+6 slots), `hiker_backpack` (+12 slots).

### Medical and survival
`bandage` (C), `painkillers` (C), `first_aid_kit` (U), `antidote_gel` (R, cures infection), `energy_drink` (C, stamina then jitters), `thermos_of_cocoa` (U), `water_bottle` (C), `water_purifier` (U), `hand_warmers` (C), `sunscreen` (C), `gas_mask` (R, blocks bloater clouds), `sleeping_bag` (C), `tent_kit` (C).

### Tools and utility
`axe`, `pickaxe`, `shovel`, `wrench`, `ratchet`, `jack`, `welding_kit`, `lockpick` (slow, noisy), `bolt_cutters` (U, cut fences and padlocks), `flashlight` (C, cone, uses batteries), `headlamp` (U), `walkie_talkie` (U, teammate range), `binoculars` (U), `map_scrap` (C, reveals a region), `compass` (C), `rope`, `duct_tape`, `hose`, `jerrycan`, `fishing_rod`, `multi_tool` (R), `grappling_hook_of_doubt` (R, reaches rooftops, sometimes misses).

### Movement toys
`skateboard` (U, fast and silent on roads), `roller_skates` (U), `pogo_stick` (U, bounces over fences), `zipline_kit` (R).

### Vehicle items (see section 15)
`battery` (U), `spark_plugs` (C), `engine_block` (R), `fan_belt` (C), `radiator` (U), `tire` (C), `fuel_pump` (U), `muffler` (U, less engine noise), `bull_bar` (U, plows zombies), `armor_plating` (R), `spoiler_of_speed` (U, +5% top speed), `turbo_can` (R), `horn_pack` (C, swaps your horn: trombone, bike horn, duck, rooster), `air_freshener_tree` (C, small sanity boost), `paint_can` (C, colours), `funny_license_plate` (C, cosmetic).

### Crafting materials
`wood_log`, `plank`, `stone`, `flint`, `scrap_metal`, `nails`, `cloth_scrap`, `rope`, `glue`, `wire`, `spring` (from mattresses), `gears`, `empty_bottle`, `battery_cell`, and `gunk_goo` (dropped by zombies, used for jank fuel and slip traps).

### Junk and goofy loot
`mismatched_sock`, `rubber_duck`, `whoopee_cushion`, `fake_mustache`, `lava_lamp` (light plus a calming effect), `garden_gnome`, `trophy_cup`, `broken_toaster` (sells or scraps for parts), `novelty_mug`, `inflatable_flamingo` (floats, works as a decoy), `kazoo` (decoy), `disco_ball_shard`, `giant_novelty_check` (X), `golden_toilet_seat` (X).

### Currency and ammo
- `shiny_buttons`: the common currency for traders, found everywhere in small amounts.
- **Ammo**: `pebbles`, `foam_darts`, `corks`, `nails`, `potatoes`, `bottle_rockets`, `flare_cartridges`, `batteries_aa`.

### Keys and quest items
`house_key`, `car_key`, `keycard_red`, `keycard_blue`, `keycard_green`, `padlock_code_note`, `rescue_radio_part_1`, `rescue_radio_part_2`, `rescue_radio_part_3`, `boat_key`, `map_fragment` (points to a treasure spot), `signal_flare`, `evac_beacon`.

---

## 22. Loot tables by place
Each table lists category weights out of 100. Every building rolls one theme, and each room type shifts the weights.
- `house_kitchen`: food 40, drinks 15, tools 10, junk 25, medical 5, rare 5.
- `house_bedroom`: wearables 35, junk 30, medical 15, melee 10, rare 10.
- `house_garage`: car parts 30, tools 30, fuel 15, melee 10, junk 10, rare 5.
- `shop_general`: food 30, drinks 15, junk 25, tools 15, medical 10, rare 5.
- `pharmacy`: medical 60, drinks 10, junk 15, rare 15 (antidote gel lives here).
- `diner`: food 55, drinks 25, junk 15, rare 5.
- `school`: junk 40, sports gear 20 (bats, hockey sticks, bowling balls), throwables 15, medical 10, keys 5, rare 10.
- `office`: junk 40, batteries 15, tools 15, maps 10, keycards 10, rare 10.
- `police_locker`: melee 20, throwables 20, armor 20, keys and keycards 15, ammo 15, rare 10.
- `farm_barn`: tools 30, food 25, fuel 20, melee 15, vehicle parts 10.
- `workshop`: car parts 50, tools 30, fuel 10, rare 10.
- `outpost_crate`: ammo 25, armor 20, medical 15, ranged 15, keycards 10, rare 15.
- `vault_jackpot`: rare 50, jackpot 30, keycards 10, medical 10.
- `zombie_drop`: gunk_goo 40, shiny_buttons 30, junk 20, rare 10.

---

## 23. Balance rules for items and weapons
- **Trade-offs**: every weapon gives up something: damage, noise, weight or durability. Nothing is best at everything.
- **Noise**: uses the same 0 to 5 scale as the zombie noise system, so weapons feed the stealth loop directly.
- **Weight scale**: light 0.2, medium 1, heavy 3, very heavy 8. Weight and slots are both limits.
- **Weapon mods**: a `duct_tape` grip (+durability), `nails` (+damage, more noise), `googly_eyes` (sanity), a `bell` (louder, but lets you pull zombies away from teammates).
- **Progression tiers**: 0 hands and sticks, 1 scavenged melee, 2 ranged toys and traps, 3 gadgets and turrets, 4 armored vehicles.
- **Scarcity**: the first night is guaranteed one melee weapon. Ammo is common but light, so you carry it deliberately.
- **Repairs**: melee weapons can be repaired at a workbench with `scrap_metal` and `glue`, to a lower max each time.

---

## 24. Settlements: towns, villages and small places
Settlements sit on top of the existing biomes. Nothing in section 1 moves. Each has a size, a biome fit, a zombie density, a loot level and a signature gimmick.
- `town` (any flat biome, 25 to 60 buildings): grid streets, town hall, school, clinic, supermarket, police station, gas station, apartments, a park and a water tower. Highest loot, highest zombies. Gimmick: a siren on the town hall that, once rung, draws every zombie in the valley.
- `village` (Meadow, Thicket, Tundra, 8 to 20 buildings): a main street, a well and a market square. Medium loot. Gimmick: a bell that wakes zombies but also calls helpful survivors.
- `hamlet` (any, 3 to 6 buildings): a few houses along a lane. Low zombies, good for the early game.
- `farmstead` (Meadow, Flats, 3 to 5): farmhouse, barn, silo, tractor. Food and fuel.
- `strip_stop` (highways, 3 to 5): gas station, diner and motel together. The roadside lifeline.
- `harbor_town` (Cove): pier, fish market, boats, lighthouse. Boats and the evac route.
- `stilt_village` (Bog): houses on stilts joined by boardwalks. Mud below, so zombies crawl.
- `desert_trading_post` (Flats): adobe shops, a water tower, the drive-in. Water is the loot.
- `ski_lodge_village` (Tundra): a lodge, chairlift and cabins. Warm shelter from the cold meter.
- `mining_camp` (Ridge): shafts, ore carts, a bunkhouse. Dark tunnels with tool loot.
- `logging_camp` (Thicket): sawmill, log stacks, a cookhouse.
- `junkyard_town` (Scrapyard): bus houses and container homes. Parts everywhere.
- `ghost_village` (Hollow): half-swallowed by fog, with sanity drain and rare loot.
- `volcano_observatory` (Grumbleguts): a research station with a jackpot vault.

---

## 25. Lost outposts and hidden places
- `ranger_tower` (Thicket, Ridge): climb it to reveal the map. Has a supply stash.
- `weather_station` (Tundra): radio parts and storm warnings.
- `survey_camp` (Flats): maps and binoculars.
- `abandoned_bunker` (Ridge): keycard doors, pitch dark, jackpot loot.
- `container_fort` (Scrapyard, Cove): stacked shipping containers that can become a base.
- `lighthouse_outpost` (Cove): with a generator its beam sweeps and clears zombies in a cone at night.
- `pancake_cult_compound` (Hollow): a goofy cult of pancake worshippers with silly relics.
- `abandoned_theme_park` (Meadow edge): a haunted carousel, prize booths full of loot and a goofy mid-boss.
- `circus_camp` (Flats): clown_zombies and a big top full of props.
- `crashed_cargo_plane` (any): looting the supply pile is very loud, and a ring of zombies arrives.
- `hermit_cabin` (Thicket): a hermit trader with odd tastes.
- `sunken_bus` (Bog): loot underwater, needs the `bubble_tea_of_buoyancy`.
- `treasure_x_spots`: found with a `map_fragment`, dig with a shovel.
- `bro_museum`: an easter-egg hall showing the four bros as statues.

---

## 26. Building kit and interiors
Building types: `house_small`, `house_two_story`, `cabin`, `bungalow`, `apartment_block`, `shop`, `supermarket`, `pharmacy`, `diner`, `garage`, `gas_station`, `school`, `chapel`, `town_hall`, `police_station`, `clinic`, `warehouse`, `barn`, `silo`, `water_tower`, `motel`, `shed`, `playground`.

### Interior rules
- Buildings use grid footprints (about 6x6 m up to 16x12 m) with 1 to 3 floors.
- Rooms come from templates (kitchen, bedroom, office, storage), and each picks a loot table from section 22.
- Doors are unlocked, locked, boarded or barricaded. Windows can be climbed through, which makes noise.
- **The pop-up look**: roofs and walls fade and fold away as you enter, so you can see inside without the camera clipping.
- Interiors generate lazily, only when a player is within about 40 m, using a hash of the seed and the building ID so they are always identical.
- Light: power can be on or off. A flashlight cone reveals dark rooms. Flickering bulbs hint at zombies nearby.
- **Hiding spots**: wardrobe, under_bed and dumpster, with funny cramped animations. Zombies sniff and wander off if you stay quiet.

### Town lots and yards (built 2026-10-19, phase Z30)
- A house sits at a fixed distance from the street edge of its lot, door toward the street, with the back of the lot left as a yard.
- Grounds: porch with railings and steps, stepping-stone path to the sidewalk, mailbox at the kerb, picket-fenced back yard, vegetable beds, porch lantern.
- Colour: each building picks one of its `palettes` from a hash of the seed and its position, so the same house always has the same colours and a town shows several roof colours.
- Code: `townBuildings` in `world4bros-core/src/sim/worldgen.ts` (optional `yard` and `front` on `Building`), drawing in `src/render/buildings.ts`, styles in `src/data/buildings.ts`.

**Settlement conditions** (set per settlement): pristine, worn, burnt, flooded, collapsed, overgrown, barricaded by survivors. Each changes props, light, loot and zombie behaviour.

---

## 27. How the map generates
1. **Seed and biomes**: the existing terrain and biome generation runs first and is never changed.
2. **Suitability map**: score every cell for settlements using flatness, distance to water, distance to other settlements and a biome weight.
3. **Site picking**: Poisson-disc sampling with per-type minimum distances picks sites. Starting targets for a normal map: 2 towns, 6 to 8 villages, 8 to 12 hamlets and farmsteads, 6 to 10 outposts, 10 to 12 strip stops. Keep towns far apart (400 m or more).
4. **Site types**: the biome and score decide each site's type from section 24.
5. **Roads**: connect every site with the road graph from section 12 (minimum spanning tree plus loops, pathfinding around water and cliffs, splines). Each settlement's main street aligns with the road passing through it.
6. **Layout algorithm by type**: grid (towns), strip along the road (stops), radial around a well or plaza (villages), cluster (farmsteads), shoreline (harbor), stilts and boardwalks (bog), ring fence (outposts).
7. **Lots**: split blocks into lots with binary space partition, assign a building type by zone (shops at the center, houses toward the edge, farms and warehouses at the outskirts), then check clearance so nothing overlaps.
8. **Fill**: props, fences, wrecks, street lamps, containers with loot and zombie spawn points.
9. **Condition**: pick the ruin level by distance from spawn and from the evac dock, so danger grows as you head toward the goal.
10. **Shelters**: mark claimable safe houses (section 28) in every settlement and outpost.
11. **Validate**: the three radio parts, the keys, a car, fuel and the evac dock must all be reachable. If not, adjust or reroll with seed plus one, up to 5 times, and save the result.

**Determinism**: every step is driven by a hash of the seed and the chunk coordinates, so the server and all clients get the same world. Chunks are about 64 m and are generated as the player approaches, spread across frames to avoid hitches.
**Navigation**: from far away, settlements read as silhouettes (water tower, smoke, a chapel spire, lit windows at night), and roads and signs lead to them.

---

## 28. Shelters and bases
### Shelter types
- **Temporary**: tent, lean_to, a sleeping_bag spot.
- **Found**: cabin, motel_room, garage, bus_base, container_fort, lighthouse_outpost, bunker, treehouse, silo, school_gym, water_tower_base.
- **Built**: the Bro Basecamp from section 1, now buildable anywhere.

### Base systems
- **Claim**: place a `claim_flag` (a cartoon pennant) to claim a building or area. The claim radius grows with base tier: 10, 20 or 35 m. Zombies do not spawn inside a claim. A claim lapses if every owner is away for 5 days.
- **Safe ring**: zombies that are chasing you can still enter, so defenses matter. Sleeping in a bed inside a claim is safe and skips the night.
- **Defenses**: wooden and metal doors, boarded windows, barricades, fences and gates, spike pits, sandbags, searchlights, foam_turret, alarm_cans.
- **Utilities**: generator (burns fuel, powers lights, fridge, workbench, radio and turrets), rain_barrel, solar_panel (R), stove, fridge (stops spoilage), bed, workbench, bubblebro_cauldron, drying_rack, chest (shared), garage_bay (car repair with a lift), garden_bed, chicken_coop (eggs, and a rooster alarm), radio_station (calls traders and events).
- **Trade-off**: generators are loud and lights draw zombies. Power makes you strong and visible.
- **Siege events**: some nights a horde targets a claimed base and attacks its walls. Holding it gives a reward crate. A siren warns at dusk.
- **Base tiers**: 0 camp, 1 shack, 2 fortified house, 3 fort, 4 compound. Each tier needs materials and unlocks more slots and defenses.
- **Co-op**: shared chests with permissions, roles, and ownership that transfers if the owner leaves. Up to 3 claims per party.
- **Goofy decor**: inflatable flamingos, a disco ball, a whoopee cushion trap on the couch, neon signs, gnome guards, and a `trophy_shelf` that shows off your finds.
- **FX**: hammer and nail sounds, boards snapping into place, a lights-on click, a generator rumble, a flag flap, dust when walls take hits.

---

## 29. Survivors and traders
- **Existing**: `old_stuffins`, `traveling_salesbro`.
- **Settlement NPCs**: `grumpy_shopkeeper` (barters for shiny_buttons), `jumpy_mechanic` (sells car parts and gives repair quests), `nervous_nurse` (heals and cures infection for a price), `retired_general_of_nothing` (gives terrible military advice and real quests), `hermit_hank` (the hermit cabin trader), `radio_dj_midnight` (a voice on the car radio announcing horde nights and supply drops), `lost_tourist` (an escort quest).
- **Rescue and settle**: rescued survivors can move into your base and take a role: guard, scavenger, cook or mechanic. They need food and a bed.
- **Presentation**: gibberish blip voices with subtitles, a trade window, and a tiny goofy animation per NPC.

---

## 30. Grab bag of extra ideas
- **Stealth and movement**: sneak, sprint with stamina, vault fences, cut fences with bolt_cutters, climb ladders to rooftops, ziplines.
- **Weather**: rain muffles noise, fog cuts vision both ways, snow leaves footprints that zombies follow. Floods, landslides and fallen bridges close roads and force detours.
- **Zombie day and night**: slower by day, faster at night, drawn to light and noise.
- **Radio**: the DJ announces horde nights and supply_drop crates falling by goofy parachute.
- **Fuel economy**: stations run dry, so siphoning and finding a gas_can matter. Tune it so you are never stranded with no way forward.
- **Perks as hats**: use a skill to earn it (lockpicking, mechanic, quiet steps, bargaining) and gain a matching hat.
- **Achievements**: Honk At 100 Zombies, Park A School Bus In A Tree, Survive Horde Night In A Bathtub, Find All The Gnomes.
- **Lore**: notes, graffiti and radio logs telling a silly outbreak story (original, no brand names).
- **Replay**: daily challenge seeds, a speedrun timer, seeded races, and modifiers such as one car only, no ranged weapons, or slapstick mode.
- **Photo mode with stickers and a postcard export**.
- **Pets**: the pocket_sloth and a dog that sniffs out loot.
- **Accessibility**: auto-drive assist, colourblind-safe loot tier colours, a reduced-motion option.

---

## 31. Plan additions
New phases for the build prompt, in order: items and weapons data, map layout with settlements and roads, settlement and building generation, loot placement, then zombies, vehicles and repair, then shelters and bases, then rosters, combat and NPCs, the quest, multiplayer and polish. Keep each as its own approval step. A progress log at `docs/PROGRESS.md` records each finished phase so a new session can resume.

---

## 32. Part 4: Controls and input layouts
Part 4 defines how the game is controlled on keyboard and mouse, gamepad and phone. Existing controls are kept: WASD, Q and E for the camera, Space for the action, Enter for chat, G for dance, H for honk. New controls fill the gaps.

### Principles
- **One set of game actions, many input providers**: Game logic reads actions (`move`, `action`, `interact`), never raw keys or touches.
- **Context-aware**: The same control does the right thing for the situation (on foot, driving, repairing, building), and the screen shows only the buttons that matter right now.
- **Nothing existing moves**: New bindings only fill gaps.
- **Thumb-friendly**: On a phone, everything used in a fight is reachable without moving your thumbs.
- **Device-matching prompts**: Key caps on desktop, icons on phone, button glyphs on gamepad.

---

## 33. Keyboard and mouse
### On foot
- **W A S D or arrow keys**: move, relative to the camera.
- **Shift (hold)**: sprint. Uses stamina and makes more noise.
- **Ctrl (hold)**: sneak. Slower and quieter.
- **Q / E**: rotate the camera (existing). Middle-mouse drag also orbits, mouse wheel zooms, `Home` recenters behind you.
- **Space or Left Click**: action: chop, attack, smack (hold to keep going, existing).
- **F**: interact: pick up, open, loot, enter or exit a car, claim, talk. Hold F for long actions such as looting a crate, repairing or claiming.
- **R**: quick use. Eats or heals with the best fitting item in one press.
- **T or Right Click**: throw the selected throwable. Right click aims, release throws.
- **1 to 8**: hotbar. `[` and `]`: previous and next slot.
- **Tab**: backpack (inventory, crafting and status tabs). `C`: crafting directly (the existing recipe drawer). `M`: map. `B`: build and base mode.
- **V**: ping a spot for teammates (middle click also works).
- **G**: dance (existing). Hold G for the emote wheel. `H`: honk (existing). `Enter`: chat bubble (existing).
- **Backspace**: drop the selected item.
- **Esc**: pause, or close the open panel.
- **F1**: help. `F2`: photo mode. `Backtick (~) `: debug panel (dev builds only).

### Driving
- **W / S**: gas, and brake or reverse. `A / D`: steer.
- **Space**: handbrake and drift. `Shift`: boost, if a booster is fitted.
- **H**: honk (hold for a long honk).
- **X**: start or stop the engine. Hold X to crank and release in the green zone.
- **L**: headlights. `R`: next radio station, hold R for off.
- **F**: exit the car. `1 to 4`: switch seats.
- **C**: car status (parts and fuel). `Tab`: trunk.
- **Q / E**: orbit the camera. `Home`: lock the camera behind the car.

### Passenger and gunner
- Mouse to aim, `Space` or Left Click to fire or swing from the window, `T` to throw.
- `1 to 4` to switch seats, `F` to exit, `V` to ping.

### Repair panel (parked, near the car)
- Hold `F` near the car to open it. Click a part card to select it. Hold Left Mouse, `Space` or `F` to repair while the ring fills. `1 to 8` picks a tool. `Esc` closes.
- Other players' name tags show which part they are working on.

### Build mode (B)
- Left Click places. Right Click or `Esc` cancels. `R` rotates the piece. Mouse wheel or `[` and `]` picks the next piece. Hold `Shift` for free placement without grid snap.

### Inventory
- Drag and drop. Right Click for a menu (use, drop, split). Shift-Click moves items quickly between backpack and trunk or chest.

---

## 34. Gamepad (optional)
- **Left stick**: move. **Right stick**: camera. Click left stick toggles sprint. Click right stick toggles sneak.
- **RT**: action. **LT**: aim and throw. **A**: interact. **X**: quick use. **Y**: backpack. **B**: back or cancel.
- **LB / RB**: previous and next hotbar slot.
- **D-pad**: Up for the map, hold Down for the emote wheel, Left for honk, Right for ping.
- **Start**: pause. **Select**: map and objectives.
- **Driving**: RT gas, LT brake and reverse, left stick steer, RB handbrake, Y exit, X engine, A honk, D-pad Up headlights, D-pad Right radio.
- Aim assist and larger prompts on gamepad.

---

## 35. Phone layout (landscape first)
The screen is split into zones by thumb reach (844 by 390 layout). Every touch target is at least 48 dp and spaced at least 12 dp from its neighbours. Everything respects safe areas around notches and rounded corners.

### On foot (default)
- **Bottom-left, floating thumbstick**: appears wherever the thumb lands in the left 40% of the screen. 12% dead zone, eased direction. Push past 85% of the radius to sprint. A small turtle button above it toggles sneak.
- **Bottom-right, a big ACTION button**: hold to chop or attack, with a haptic tick on each hit. An arc of smaller buttons surrounds it:
  - **INTERACT**: a hand that changes to a loot, door, car, wrench or flag icon depending on context.
  - **QUICK USE**: eats or heals with the best item in one tap.
  - **THROW**: drag from the button to aim, release to throw.
- **Bottom-center, hotbar of 5 slots**: tap to equip, swipe sideways for more, swipe up to open the full backpack.
- **Top-left**: round minimap with roads and teammates and compass. Under it: chat/emote wheel button, and honk button.
- **Top-center**: stat bars for hunger, health, and sanity. Dynamic meters for **Infection**, **Weight**, and **Noise** appear only when relevant!
- **Top-right**: small icons for pause, map, and backpack, with camera snap buttons (rotate left and right) under them.
- **Camera**: drag anywhere in the free area to orbit, pinch to zoom. Two-finger swipe still works.
- **Ping**: long-press on the world or the minimap.

### Driving (switches automatically when you enter a car)
- **Left**: steering. Default is two large steer buttons (Left and Right) that ramp smoothly the longer you hold. Options: a virtual wheel or tilt to steer.
- **Right**: large GAS pedal above a BRAKE and REVERSE pedal, with a smaller HANDBRAKE button beside them. HORN is a big button above the pedals.
- **Top-right row**: exit, headlights, engine key (hold to crank), seat switch.
- **Top-left**: radio (tap for next station, hold for off).
- **Top-center**: speed (km/h) and fuel gauge plus damage warnings ("Engine running hot").
- **Camera**: drag empty space to orbit.

### Repair panel
- A bottom sheet of part cards (missing, broken, damaged, fine). Tap a card, then press and hold the big FIX button (which is the ACTION button moved into place). The ring fills with haptic ticks and a success pulse.

### Build mode
- A ghost piece you drag to position. Buttons on the right: place, cancel, rotate, next piece.

### Inventory and trading
- A bottom sheet. Long-press to drag and drop. Tap an item for a menu (use, equip, drop). Double-tap to equip. A weight bar and a sort button sit on top.

### Haptics
- Light taps on chops, a strong pulse on damage, ticks during engine crank and repairs, and a short pattern for Jackpot or rare drops. Can be toggled in settings.

### Comfort and accessibility
- Left-handed mode that mirrors the layout.
- Button size slider (80% to 130%) and opacity slider.
- Layout editor: drag buttons where you want them and save.
- Hold vs. toggle options for sprint and sneak.
- Auto-aim assist for melee.
- Keep screen awake, go fullscreen on first tap, ask for landscape when driving starts.
- Portrait fallback with only stick, ACTION, and hotbar, plus rotate prompt for driving.

---

## 36. Input architecture and settings
- **Action map**: define actions: `move`, `sprint`, `sneak`, `action`, `interact`, `quick_use`, `throw`, `hotbar_n`, `camera_rotate`, `honk`, `dance`, `chat`, `backpack`, `craft`, `map`, `build`, `ping`, `steer`, `throttle`, `brake`, `handbrake`, `boost`, `engine`, `headlights`, `exit_vehicle`, `seat_swap`.
- **Contexts**: `on_foot`, `driving`, `passenger`, `repairing`, `building`, `ui`, `chat`, `boat`, `air` (whirlybird), `balloon`, `handcar`.
- **Providers**: keyboard and mouse, gamepad, and touch each translate raw input into actions. The game reads actions only.
- **Physical keys**: use `event.code` (physical position) so WASD works on AZERTY and other layouts.
- **Chat safety**: when chat box is focused, ignore game keys.
- **Browser hygiene**: `preventDefault` on Tab, Space, arrow scrolling, F-keys. On touch, use pointer events with `touch-action: none`, track every pointer ID, block double-tap zoom/pull-to-refresh, release all inputs when tab loses focus.
- **Buffering**: a 120 ms input buffer so action presses are never eaten.
- **Rebinding**: settings screen with conflict detection and presets (Default, Arrow keys, Left-handed, One-handed) in localStorage.

---

## 37. Plan additions for controls
A new phase **Z0B (input system and on-foot layouts)** goes right after Z0. Each later phase adds its own context: driving in Z6, repair in Z7, building in Z8, combat and throwing in Z10. Left-handed mode, the layout editor, gamepad support, and real-device tests go in Z13.

---

## 38. Part 5: Vehicle fleet expansion
Part 5 grows the fleet beyond section 14. Same IDs, tiers (J, C, U, R, X) and quirk rules. Every vehicle is original, goofy, and uses the repair system from section 15.

### Vehicle stat model (one `VehicleDef` per ID)
- Top speed, acceleration, handling, grip, toughness (HP plus part wear), fuel capacity and burn rate, noise (0 to 5), seats, cargo slots, weight class, height class, offroad rating, water rating, and one special ability.
- Classes: `scooter`, `compact`, `sedan`, `van`, `pickup`, `truck`, `bus`, `heavy`, `weird`, `boat`, `rail`, `air`.

### Fleet roles
- **Scout**: fast and fragile (`pizza_scooter`, `dune_buggy`, `whirlybird`).
- **Silent**: quiet for sneaking (`golf_cart_deluxe`, `bicycle_bell`, `rowboat_oars`, `scout_balloon`).
- **Hauler**: lots of cargo, slow (`box_truck_moving`, `semi_rig`, `school_bus_of_chaos`).
- **Base on wheels**: sleep and cook on the move (`camper_van_of_comfort`, `food_truck_fiesta`, `party_bus`).
- **Utility**: opens routes (`tow_truck_with_attitude`, `bulldozer_bro`, `fire_truck_of_fun`).
- **Fighter**: smashes zombies (`monster_truck_of_questionable_taste`, `garbage_truck_grumble`, `party_tank`).
- **Terrain specialist**: crosses what others cannot (`swamp_hovercraft`, `snowmobile_zippy`, `quad_bike_atv`, `jet_ski_zoom`).

---

## 39. More cars, trucks and big rigs
### Cars
- **`taxi_of_lies` (C)**: 4 seats, 6 slots. Rescued survivors pay fares in `shiny_buttons`. Meter ding counts as a noise event.
- **`tiny_bubble_car` (C)**: 2 seats, 3 slots, very low fuel use. Egg-shaped, tips onto side on hard turns, push it back upright.
- **`dune_buggy` (U)**: 2 seats, fast, best grip on sand and dirt. No roof, so zombies can reach the seats.
- **`vintage_roadster` (U)**: 2 seats, convertible. Sanity recovers while driving with top down, wind-blown hair particles.
- **`party_hearse` (U)**: 3 seats, 10 coffin slots. Spooky organ radio, Hollow creatures ignore you.
- **`patrol_wagon` (U)**: 4 seats, push bar and siren. Siren turns zombies within 40 m toward you (lure).
- **`ambulance_pal` (U)**: 3 seats, stretcher slot that slowly heals riders, siren.
- **`stretch_limo` (R)**: 8 seats, 12 slots, huge turning circle, mini fridge stops food spoiling, longest honk.
- **`camper_van_of_comfort` (R)**: 4 seats, 14 slots, bed, stove, fold-out tent. Mobile base with thirsty engine.
- **`hot_tub_hauler` (R)**: 4 seats, slow. Restores sanity and warmth while sitting in it. Bubble trail, jet-spray horn.
- **`couch_on_wheels` (J)**: 3 seats, lawnmower engine, very slow. Search cushions daily for 3 `shiny_buttons`.
- **`porta_potty_dragster` (R)**: 1 seat, rocket speed in a straight line, terrible steering. Blue puff trail when boosting.

### Trucks and big rigs
- **`pickup_with_bed` (C)**: 3 seats, 12-slot open bed. Two bed riders can swing weapons or throw. Cargo spills on big jumps.
- **`box_truck_moving` (U)**: 3 seats, 40 slots. A zombie sometimes hides in the back when spawned.
- **`food_truck_fiesta` (U)**: 3 seats, 16 slots, griddle and fryer with expanded recipes. Toss food through serving window. Smell lures zombies and the goose.
- **`mail_truck_mailbox` (C)**: 2 seats, 18 slots. Collect letters from roadside mailboxes for hints.
- **`tow_truck_with_attitude` (U)**: 2 seats. Winch pulls wrecks off roads and tows broken cars.
- **`fire_truck_of_fun` (R)**: 6 seats. Ladder reaches rooftops, hose sprays zombies and puts out fires. Loud siren.
- **`garbage_truck_grumble` (U)**: 3 seats, slow. Compactor crunches zombies into green puffs. Smell attracts zombies and goose.
- **`cement_mixer_maracas` (U)**: drum rattles loudly. Drops gunk puddle behind that slows zombies.
- **`tanker_truck_gassy` (R)**: 2 seats, carries 500 fuel. Works as a base fuel station. Cartoon fireball leak risk if damaged.
- **`armored_bank_truck` (R)**: 3 seats, very tough, slow. Vault in back needs `keycard_green` and holds jackpot loot.
- **`semi_rig` (R)**: 2 seats. Hitches trailers (section 44), cannot cross weak bridges.
- **`party_bus` (R)**: 10 seats with disco lights. Disco mode stuns zombies within 15 m for 6 s, but music is noise level 5.
- **`bulldozer_bro` (R)**: 1 seat, very slow. Blade pushes wrecks and barricades aside. Cannot use boardwalks.
- **`road_roller_stan` (R)**: 1 seat. Flattens zombies into cartoon pancakes and paves dirt tracks into smooth roads.
- **`forklift_fred` (U)**: 1 seat. Lifts crates up to rooftops and nudges roadblocks.
- **`party_tank` (X)**: Boss reward. Confetti cannon turret, crushes cars, extremely tough.

---

## 40. Bikes, scooters and tiny rides
- **`bicycle_bell` (J)**: No fuel, silent, 1 seat. Bell is a tiny decoy.
- **`bmx_trick_bike` (C)**: Jumps ramps and curbs. Landing a flip gives a small sanity boost.
- **`kick_scooter_electric` (C)**: Battery powered and silent. Folds into your backpack.
- **`quad_bike_atv` (C)**: 2 seats, strong off-road.
- **`chopper_motorbike` (U)**: Fast, 2 seats, wheelies and leans into turns.
- **`sidecar_motorbike` (U)**: 3 seats. Sidecar rider throws or aims.
- **`mobility_scooter_turbo` (U)**: Slow but heavy. Bumps zombies aside with cheerful beep. Flag and cup holder.
- **`unicycle_circus` (R)**: Balance mini-game. Hops over fences and juggles throwables.

---

## 41. Boats, snow and swamp rides
Zombies cannot swim, so deep water is a natural escape route.
- **`rowboat_oars` (J)**: Silent water crossing.
- **`inflatable_dinghy` (C)**: Pool-noodle oars. Can pop.
- **`rubber_duck_pedalboat` (U)**: Giant duck pedal boat that quacks.
- **`jet_ski_zoom` (U)**: Fast, 2 seats, loud, short fuel range.
- **`speedboat_splash` (U)**: 4 seats.
- **`pontoon_party_boat` (R)**: 8 seats, slow, with fire pit. Floating base.
- **`fishing_trawler` (R)**: Large. Nets bring in food, one of the three evac options. Needs `boat_key`.
- **`swamp_hovercraft` (R)**: Crosses mud, water and bog. Fan is loud and skirt blows dust and ripples.
- **`snowmobile_zippy` (U)**: Fast on ice and snow, poor on asphalt.
- **`sled_with_sloth` (C)**: Coasts downhill, pocket_sloth pulls it at leisurely pace on flat ground.

---

## 42. Rail and rideables
- **`rail_handcar` (U)**: Pump trolley for abandoned rail lines. Two players pump alternately. Fast, mostly zombie-free lane; broken gaps need `track_gap_plates`.
- **`locomotive_of_doom` (X)**: Late-game train engine on main rail loop. Needs `boiler_fuel` and a crew.
- **`goose_steed` (R)**: Tamed honk_goose you ride. Tame with `cheese_wheel`. Honk to dash.
- **`ostrich_runner` (U, Flats)**: Fast sprinter that kicks zombies. Panics when you honk.

---

## 43. Aircraft: the whirlybird and friends
Build the helicopter first. The rest are optional.
- **`whirlybird` (R)**: Small 2-seat helicopter. Fragile, thirsty, and very loud (noise 5).
- **`pedal_copter` (U)**: Human-powered. Low altitude only.
- **`scout_balloon` (U)**: Hot-air balloon. Silent, drifts with wind, burns propane.
- **`paraglider_kite` (U)**: Launches from cliffs in Rockbottom Ridge. Silent glide down.
- **`crop_duster_biplane` (R)**: Fast. Needs long straight road as runway.
- **`sky_blimp` (X)**: Endgame airship base. Very slow.

### Whirlybird flight model
- **Arcade flight with 3 altitude bands**: Ground, Low, High. Climb and descend buttons. Camera zooms out with altitude; ground shadow scales with height.
- **Skid grab**: Low over ground (<2 m) zombies jump up and grab skids. Spin helicopter to fling them off.
- **Hover assist**: Release controls and it holds position automatically.
- **Landing**: Needs flat ground, rooftop or helipad. Land assist lowers slowly.
- **Winch hook**: Lifts crates, survivors, or a small car on a sling.
- **Limits**: Noise 5 summons hordes. Burns `aviation_fuel` (gas causes engine damage). Weather wind gusts push it, fog grounds it, bird strikes from honk_goose flocks.
- **Spawn**: 1 `whirlybird_wreck` per map at rooftop helipad (`police_station`, `hospital_ruins`, `lighthouse_outpost`, or `volcano_observatory`). Parts chain: `rotor_blades`, `tail_rotor`, `turbine_engine`, `gearbox`, `landing_skids`, `cockpit_glass`, `flight_gizmo`.
- **Evac alternative**: 3 escape routes: car to dock boat, trawler, or flying whirlybird to rescue ship!
- **FX**: Spinning blade blur, grass/leaf downwash rings, doppler whop-whop with squeaky rubber-chicken twist, cartoon boing/spin on crash.

---

## 44. Trailers, towing and hitches
- **Hitch**: `trailer_hitch` lets vehicles pull trailers. Reversing is deliberately wobbly and funny.
- **Trailers**: `trailer_cargo` (30 slots), `trailer_camper` (a bed), `trailer_boat` (carries boat to water), `trailer_flatbed` (carries small car), `trailer_water_tank`, `trailer_hay_wagon` (soft ride, hides loot).
- **Towing**: `tow_rope` between two vehicles lets one pull broken car. Tow truck winch does same with wrecks.

---

## 45. Route rules: weight, height and width
- **Bridge load limits**: Light, medium, heavy. `weak_bridge` collapses under heavy rig in cartoon fashion.
- **Tunnel heights**: Tall vehicles (bus, fire truck with ladder, semi) cannot enter without detours.
- **Width**: Narrow boardwalks and mountain lanes scrape wide vehicles. Heavy vehicles break bog boardwalk planks.
- **Tires & runners**: `snow_chains`, `balloon_tires`, and `knobby_tires` change grip on snow, mud, and sand.
- **Validator**: Automated reachability test runs per vehicle class (light, medium, heavy, specialist) so valid routes exist for at least 3 classes and evac dock.

---

## 46. Parts and repair for the new vehicles
- **Cars & trucks**: Everything in Section 15, plus `trailer_hitch`, `winch_cable`, `hydraulic_pump`, `siren_box`.
- **Two-wheelers**: `chain`, `pedals`, `handlebars`, `battery_pack`, `kickstand`.
- **Boats**: `hull_patch_kit`, `outboard_motor`, `propeller`, `bilge_pump`, `oars`, `life_jacket`.
- **Snow & swamp**: `sled_runners`, `track_belt`, `fan_blade`, `skirt_fabric`.
- **Helicopter**: `rotor_blades`, `tail_rotor`, `turbine_engine`, `gearbox`, `landing_skids`, `cockpit_glass`, `flight_gizmo`, `aviation_fuel`.
- **Balloon**: `balloon_envelope`, `burner`, `basket`, `propane_tank`.
- **Rail**: `rail_wheels`, `pump_handle`, `track_gap_plates`.
- **New mini-games**: Rotor balancing (hold and release in green zone), hull patching (slap patches on leaks while water rises), chain re-seating, winch rigging.
- **Jank fixes**: `pool_noodle_oars`, `umbrella_sail`, `ceiling_fan_rotor` (works at 40%), `trash_bag_balloon`, `bedsheet_wings`.
- **Customization**: Paint, decals, bobbleheads, fuzzy dice, underglow neon, exhaust styles (flames, bubbles), horn songs, and mounted gear.

---

## 47. Controls for the new vehicles
- **Bikes and scooters**: Same as driving. Space bunny-hops, Shift pedals faster.
- **Boats, hovercraft, snowmobiles**: Space is brake or anchor instead of drift.
- **Handcar**: Alternate A and D to pump (or tap two buttons alternately on phone).
- **Whirlybird keyboard**: W/S forward/back, A/D turn, Space (hold) climb, Ctrl (hold) descend, Shift boost, X engine (4s rotor spool-up), F exit/rappel, Z winch, L searchlight, H horn. Release everything to hover.
- **Whirlybird phone**: Virtual stick on left (moves and auto-turns), stacked Climb/Descend buttons on right + Land button, top row winch/searchlight/horn/boost/exit. Lift thumbs to hover.
- **Balloon**: Space burner rise, Ctrl vent descend. Pick altitude bands to catch different wind.

---

## 48. Where vehicles spawn
- `auto_garage` & houses: cars and tow truck. Farmhouse: tractor and pickup. Supermarket: shopping carts and box trucks.
- `police_station`: patrol wagon + rooftop helipad. `hospital_ruins`: ambulance pal + helipad.
- `harbor_town`: boats and trawler. `junkyard_town`: weird rides (couch, hot tub, porta potty, road roller).
- `ski_lodge_village`: snowmobiles. `stilt_village`: hovercraft. `desert_trading_post`: dune buggy and ostrich runner. `mining_camp`: quad bikes.
- `abandoned_theme_park`: pedal copter. Rail depots: handcar. `volcano_observatory`: helipad.
- **Guarantees**: Each region has at least one common vehicle within reach; exactly one whirlybird wreck per map with its parts chain.

---

## 49. Plan additions for the fleet
After the rosters phase (Z9) add:
- **Phase Z9B**: Big rigs, trailers, and route rules (including per-class reachability validator).
- **Phase Z9C**: Bikes, boats, snow, swamp, and rail.
- **Phase Z9D**: Aircraft with the whirlybird first and the others optional.

---

## 50. Phone layouts for the whirlybird and boats
Matches the mockups on the control design canvas (`World 4 Bro's Control Layouts.pdf`):
- **Whirlybird**: Left 40% floating stick with hover assist; right stacked Climb and Descend buttons + 64px Land button; top row winch, searchlight, horn, boost, exit/rappel; top-center altitude band and fuel; warning chips (zombie on skids, low fuel, wind gust, bird strike); 4s rotor spool-up.
- **Boat**: Two steer buttons on left; Throttle and Reverse on right; Anchor button (replaces drift); top-right seat switch, engine key, lights, leave boat; top gauges for speed, fuel, hull; leak warning chip opening hull patching mini-game.

---

## 51. Plan additions for the new layouts
Implement the whirlybird layout in aircraft phase (Z9D) and boat layout in boats phase (Z9C). Reuse driving layout components. Add Playwright touch tests: lift thumbs to hover, confirm Land disabled over rough ground, 4s start engine hold, and leak warning hull patching sheet. Layout editor in Z13 covers every vehicle layout.

---

# Part 6: Pets, Companions & Wildlife

## 52. Pet philosophy and rules
- **Companions never die**: In keeping with the slapstick cartoon tone, pets are invulnerable to lethal zombie damage. When high danger or a horde attack occurs, pets panic, play comedic escape animations, and retreat safely to the nearest player basecamp, tent, or active campfire circle.
- **Personality over combat**: Pets provide passive utility, sanity preservation, comedic distractions, and emotional companionship rather than acting as combat damage sponges.

## 53. Pet roster
- **`pocket_sloth` (C)**: Clings lovingly to the player's backpack. Very slow when on ground. Slowly restores sanity while held, snoozes during tense combat, and helps pull the downhill snow sled.
- **`scrappy_pup` (U)**: Cheerful terrier mix. Barks to alert player of approaching zombies from behind, automatically fetches dropped twigs and flint, and sniffs out hidden roadside junk caches.
- **`camp_cat` (U)**: Independent calico. Catches nuisance mice around food caches, curls up on sleeping bags to accelerate overnight health recovery, and purrs near campfires to grant a warmth buff.
- **`honk_goose_pet` (R)**: Loud, chaotic guard goose. Tamed using a `cheese_wheel`. Honks furiously to scatter small critters, stuns sneak zombies for 1.5 s, and can be ridden using a cheese wheel on a stick.
- **`backpack_possum` (R)**: Rides in outer side-pocket. Plays dead with hilarious cartoon dramatic flair when a zombie grapples the player, providing a 2-second escape window. Grants 2 secret inventory slots.
- **`disco_parrot` (X)**: Perches on shoulder. Squeaks mimicry of player horn honks, performs rhythmic wing-flaps during dance emotes, and flashes iridescent feathers to dazzle zombies.

## 54. Pet taming and bonding
- **Taming Foods**: `marshmallow` for pocket sloth, `bone` for scrappy pup, `raw_fish` for camp cat, `cheese_wheel` for honk goose, `wild_berries` for possum, `cracker` for disco parrot.
- **Bonding Levels (1 to 5)**: Fed pets gain heart points. Higher bonding unlocks pet tricks, faster follow speeds, and unique cosmetics.

## 55. Pet commands and whistles
- Quick whistle radial (<kbd>V</kbd> or touch icon):
  - **Follow**: Stays close to player heel or rides backpack.
  - **Stay**: Guards basecamp or vehicle passenger seat.
  - **Fetch**: Runs to collect nearby loose floor items.
  - **Alert**: Warns of distant horde sounds.

## 56. Pet accessories and customization
- Tiny hats: mini top hat, propeller beanie, detective cap, party cone, sunglasses.
- Bandanas, bell collars, and saddlebag pouches.

## 57. Pet care and interactions
- **Petting Interaction**: Hold Action button near pet to play a warm petting animation with heart sparkles and contented sound effects.

## 58. Pet shelters and base beds
- Placeable base objects: `cozy_pet_basket`, `sloth_hammock`, `goose_straw_nest`.

## 59. Co-op multiplayer pet synchronization
- Pets are fully visible and synchronized across multiplayer peers; friends can pet each other's companions.

## 60. Plan additions for pets
- Implemented in Phase Z10B (Quadruped & Critter Rigs) and Phase Z10C (Avian & Exotic Pet Rigs).

---

# Part 7: Animation Engine & Procedural Rigging

## 61. Character pipeline: Option A vs Option B
The game demands expressive, readable slapstick animation in a 2.5D world with a rotating camera. There are two primary architectural routes for character rendering and skeletal animation:

### Option A: 3D Skinned Mesh with Toon Cel-Shading
- **Architecture**: 3D low-poly humanoid skeletal mesh bound to a standard humanoid bone hierarchy (`THREE.SkinnedMesh`, `THREE.Skeleton`). Shaded using a custom two-tone cel-shading material with inverted-hull black ink contour outlines.
- **Animation Mechanism**: Standard Three.js `AnimationClip` and `AnimationMixer` evaluating bone quaternions and translations.
- **Strengths**: Smooth full 360-degree rotation without billboard plane sorting; seamless blending and crossfades between arbitrary skeletal poses; IK look-at and procedural secondary bone physics (springs) are mathematically natural on 3D bone transforms; standard tooling compatibility (Blender rigs / mixamo / procedural bone hierarchies).
- **Trade-offs**: Heavier draw call and skinning matrix compute budget; higher texture/mesh memory; requires 3D modeling/rigging assets, drifting away from the pure procedural hand-drawn charcoal aesthetic.

### Option B: 2D Billboard Cutout Puppets (Multi-Directional Sprite Rig)
- **Architecture**: Segmented 2D planar hierarchy (head, torso, backpack, upper arms, forearms, legs, held tool) organized into a hierarchical node tree with pivot offsets. Billboards face camera yaw with multi-angle perspective sprite swapping (Front, Side, Back, 45-degree diagonals) using a hysteresis buffer ($\epsilon = 0.20$ rad) to eliminate edge flickering.
- **Animation Mechanism**: Hierarchical node transform tweening (interpolating rotations, local offsets, and squash-and-stretch scale factors) combined with sprite sheet/canvas texture state swaps.
- **Strengths**: 100% faithful to the Don't Starve / Paper Mario gothic ink aesthetic established in `TextureGenerator.ts`; extremely lightweight (draw quads, zero heavy skinned-mesh matrix computations, effortlessly runs hundreds of characters on mobile); instant hot-reloading with zero external binary asset dependencies; slapstick squash & stretch and cartoon snap effects look fantastic on paper cutout puppets.
- **Trade-offs**: Requires manual joint depth ordering (`renderOrder`) when rotating between front and back; turning animations require frame-synchronized squash flips.

---

## 62. Animation framework architecture
Build a unified, modular animation engine decoupled from rendering representations:
- **Clip Playback Engine**: Keyframe track evaluation supporting translation, rotation, scale, opacity, and sprite index over normalized time $t \in [0, 1]$.
- **State Machine with Layered Blending**:
  1. **Base Layer**: Core locomotion state machine (`idle`, `walk`, `run`, `sprint`, `sneak`) with automatic crossfades.
  2. **Upper Body Layer**: Decoupled actions (e.g. swinging axe, aiming ranged weapon, eating food, or honking horn while continuing to run).
  3. **Additive Layer**: Subtle procedural breathing, wind sway, and secondary motion overlays.
  4. **Override Layer**: Full-body prioritized interrupts (`hurt`, `knockdown`, `get_up`, `dance_emote`).
- **Blend Trees**: 1D speed blend (smooth interpolation between walk and run based on linear ground velocity) and 2D directional blend (strafing/turning).
- **Crossfades**: Smooth transition curves (`smoothstep`, `exponential`) with configurable blend durations (e.g. 120ms locomotion, 50ms combat snap).
- **Named Animation Event System**: Clips emit frame-accurate events (`hit_frame`, `footstep`, `sound`, `dust_puff`, `haptic_tick`). Subscribers trigger gameplay rules, audio cues, and particle bursts deterministically.
- **Procedural Layer**:
  - **Spring Dynamics**: Second-order damped spring system for head bob, backpack lag, antenna wobble, and tail bounce.
  - **Look-At IK**: Smooth head/eye gaze targeting toward nearby points of interest or mouse cursor within angle constraints.
  - **Procedural Lean**: Banking and torso leaning into movement direction and angular turn velocity.
  - **Stride Matching**: Dynamically scales clip playback speed to match actual ground movement velocity, eliminating foot sliding.
- **Preset Libraries**:
  - `TweenPreset`: `linear`, `quadEaseIn`, `quadEaseOut`, `elasticOut`, `bounceOut`, `snapPunch`.
  - `SpringPreset`: `snappy` ($f=5\text{Hz}, \zeta=0.7$), `bouncy` ($f=3\text{Hz}, \zeta=0.4$), `heavy` ($f=2\text{Hz}, \zeta=0.9$), `loose` ($f=1.5\text{Hz}, \zeta=0.5$).

---

## 63. Fixed-timestep loop & rendering interpolation
Animation state evaluates on the deterministic fixed physics tick:
- **Fixed Timestep**: `FIXED_TIMESTEP = 1 / 60` (60 Hz). All state machine transitions, clip advancement, and event dispatches occur strictly inside fixed ticks.
- **Alpha Interpolation**: Rendering computes an interpolation factor $\alpha = \text{accumulator} / \text{FIXED_TIMESTEP} \in [0, 1)$ between the previous frame's pose and current frame's pose:
  $$\text{pose}_{\text{render}} = \text{slerp}(\text{pose}_{\text{prev}}, \text{pose}_{\text{curr}}, \alpha)$$
- **Frame-Rate-Independent Damping**: Procedural springs and leans use analytical exponential damping:
  $$\text{damp}(a, b, \lambda, dt) = a + (b - a) \cdot (1 - e^{-\lambda \cdot dt})$$
- **Delta-Time Clamping**: Clamp incoming delta times ($\Delta t \le 0.1\text{ s}$) to prevent simulation spirals or spring blowouts when the browser tab loses focus.

---

## 64. Shared humanoid rig & core clip roster
A single standardized humanoid rig is shared across players, co-op survivors, NPCs, and humanoid zombies.
- **Roster of Core Clips**:
  1. `idle`: Breathing cycle with personality fidgets.
  2. `walk`: Low-intensity balanced step cycle (speed $3.0 - 5.0\text{ m/s}$).
  3. `run`: Forward drive with arm swing and torso lean (speed $6.5\text{ m/s}$).
  4. `sneak`: Crouched low-profile stalk with turtle-like head retraction.
  5. `sprint`: High-energy pump with exaggerated forward lean and dust puff triggers ($9.2\text{ m/s}$).
  6. `turn`: Rapid 180-degree pivot with horizontal squash flip.
  7. `chop`: 3-phase woodcutting strike (anticipation windup, aggressive apex, impact frame, recoil settle).
  8. `light_swing`: Quick horizontal tool/weapon swipe.
  9. `throw`: Two-handed overhead windup, forward release point, follow-through.
  10. `hurt`: Sharp backward flinch, head snap, squash distortion.
  11. `down`: Knockout collapse to the ground.
  12. `get_up`: Slapstick scramble back to standing posture.
  13. `eat`: Holding food to mouth with rapid chewing bob.
  14. `dance`: Rhythmic party bounce and arm wave.
  15. `wave`: Friendly overhead hand greeting.
  16. `honk`: Exaggerated chest expansion, head thrown back, air horn thrust.
- **Combat Event Sync**: Damage on `chop` and melee attacks is applied strictly upon firing the `hit_frame` event, perfectly synchronizing audio impact, particle sparks, and entity damage.

---

## 65. Crowd animation engine for zombies
Rendering up to 150 zombies on desktop and 60 on mobile requires aggressive hierarchical LOD and instancing:
- **LOD0 (< 15 m)**: Full humanoid rig updating at 60 Hz with procedural head look-at and dynamic limb springs.
- **LOD1 (15 m - 35 m)**: Throttled animation rate (15-20 Hz) with interpolated poses; secondary spring physics bypassed.
- **LOD2 (> 35 m)**: GPU-instanced impostors with baked pose-texture animation sampled at 5-10 Hz.
- **Desynchronization**: Each zombie instance spawns with randomized playback speed multiplier ($0.90\times - 1.15\times$) and time phase offset $\phi \in [0, 1)$ to eliminate unison robotic marching.
- **Defeat FX**: Defeated zombies instantly trigger a green-puff cartoon pop animation, releasing a mushroom-cloud puff particle while the entity mesh is culled. Zero blood or gore.
- **Performance Budget**: Crowd animation update and skinning computation must remain under **2 ms per frame** on a standard laptop (150 zombies) and under **4 ms** on mobile (60 zombies).

---

## 66. Vehicle, repair & environmental animation engine
- **Vehicles**: Procedural wheel spin mapped to linear speed ($\omega = v / r$), steering knuckle yaw, suspension bounce over bumps using vertical spring dampers, chassis body roll on steering turns, and chassis pitch on acceleration/braking.
- **Doors & Compartments**: Smooth spring tweens for opening/closing car doors, trunk hoods, and glove boxes.
- **Damage Transitions**: Step-wise visual mesh swaps and cartoon smoke puffs on passing damage thresholds ($75\%, 50\%, 25\%, 0\%$).
- **Part-Repair System**: Parts snap into place with a satisfying cartoon bounce; radial hold-to-repair ring smooth fill tween.
- **Environmental Buildings & Props**:
  - Pop-up roof fold: Roof panels fold upright like cardboard flaps when player enters interior doorways.
  - Containers: Hinged lid swings with rubbery spring stops.
  - Loot Spawns: Items arc out in parabolic trajectories with ground bounce settling.
  - Building Construction: Base walls and fences snap into socket positions with a dust puff.
- **Heavy Fleet**: Trailer pendulum hinge oscillation, boat hull water bobbing and wake ripples, helicopter rotor disc blur with ground downwash grass rings.
- **Built so far (Z30 vehicle HD pass, 2026-10-19)**: each car body is drawn once into a cached sprite with a clear-coat shine, chrome trim, tinted glass with sky reflections, door seams and a paint decal from data (`VEHICLE_LOOKS` in `src/data/vehicles.ts`). Dirt, rust, dents and scratches build up at the 75/50/25% hp steps, and the glass cracks below 25%. Lamps react to the car: tail lights at night, brake lights when it slows hard, reverse lights, blinking turn signals while steering, and hazards when it is wrecked. Exhaust smoke goes dark when hp is low, and the rear tyres kick up dust at speed.

---

## 67. Multiplayer animation synchronization
- **Protocol**: Authoritative network updates transmit animation state names and locomotion parameters only (e.g. `{ state: 'run', speed: 1.15, facing: 1, action: 'chop' }`), never raw joint matrices.
- **Client Reconstruction**: Remote entity states feed into a local playback controller that drives local blend trees.
- **Interpolation Buffer**: Remote entities render ~100 ms in the past using a sliding state history buffer, guaranteeing smooth transitions with zero visual snapping or teleportation.

---

## 68. Animation Lab debug scene, automated tests & accessibility
- **Animation Lab (<kbd>F2</kbd> or Dev Spawner Tab)**:
  - Interactive playback scrubber: scrub clip frame-by-frame, change speed ($0.1\times - 2.0\times$), toggle loop.
  - Blend Tree Tester: slider blending between two states (e.g. Walk $\to$ Run, Idle $\to$ Sneak).
  - Event Inspector: visual timeline showing trigger points for `hit_frame`, `footstep`, and custom tags.
  - LOD & Performance Inspector: toggle LOD tiers, display bone counts, and measure frame update time in microseconds.
- **Automated Test Suite**:
  - `testPoseContinuity`: Validates that quaternion and translation deltas across state crossfades never exceed maximum threshold $\Delta_{\max}$ (no visual pops).
  - `testRequiredClips`: Verifies all mandatory humanoid clips exist and bind to valid rig nodes.
  - `testEventOrder`: Confirms `hit_frame` consistently precedes recovery frames in all attack clips.
- **Accessibility & Settings**:
  - Reduced-Motion Mode: Dampens exaggerated squash/stretch, head counter-bob, and screen shakes.
  - Mobile Animation LOD: Automatically caps crowd updates and lowers blend layers on battery-saver mode.

---

# Part 8: Master Phase Plan, Acceptance Criteria & Workflow Contract

## 69. The Master Phase Plan
This master phase order supersedes every earlier build note, patch, and roadmap draft:

**Where to resume (updated 2026-10-19)**. Any agent (Claude, Antigravity/Gemini) picks up from here:
1. **Z30 Town Houses & HD Graphics** — in progress. Houses are done; five graphics passes (roads, props, vehicles, lighting, characters) were still running on 2026-10-19 and may have left work half-done. Run `npm run verify` in `world4bros-core` first and fix any type errors before starting new work (on 2026-10-19 new street props `stop_sign`, `street_sign`, `bus_stop`, `trash_can`, `power_pole`, `abandoned_car`, `litter` had data but no drawers in `src/render/objects.ts`).
2. **Z26 Status Effects** — data layer done; next steps in `world4bros-core/docs/Z26_PLAN.md`.
3. **Z28 Aim Arena** — rounds, score and match HUD next (`world4bros-core/docs/ARENA.md`).
4. Then **Z27 Biome Atlas** (needs Z26) and **Z29 Weapon Builder** (`world4bros-core/docs/WEAPONS.md`).
5. Later: Z20 multiplayer, Z21 audio, Z22 underground, Z23 endgame (§112).

**Phase numbers**: this table is the source of truth. The "Done" list in `world4bros-core/docs/ROADMAP.md` reused Z20–Z25 for already-finished work (proving grounds, gunsmith bench, army base arena, SwarmLab, DevLog, Gemini asset guide). Those are not the Z20–Z23 phases planned here.

| Phase | Title | Scope & Deliverables |
| :--- | :--- | :--- |
| **Z0** | **Mode Flag & Pure TS Registries** | Global `mode: 'classic' \| 'zombie'` flag. All 7 data registries in pure TypeScript (`ItemDef`, `VehicleDef`, `PartDef`, `ZombieDef`, `ContainerDef`, `LootTable`, `RoadDef`, `POIDef`). In-game dev console & spawner (<kbd>~</kbd> / <kbd>F1</kbd>). Headless automated test suite. *(COMPLETE)* |
| **Z1** | **Seeded Road Network & POI Placement** | 13 Catmull-Rom spline routes connecting 11 biome POI anchors. 10 textured road surface ribbons ($y=0.02$ depth offset). Roadside props (cones, signs, billboards, wrecks, guardrails). Road foot speed bonus (+10%), ice road friction physics. Automated A* reachability validation. *(COMPLETE)* |
| **Z0B** | **Input Architecture & On-Foot Controls** | Universal `ActionMap` mapping physical `event.code` (independent of keyboard layout). 120 ms action input buffer. Page 1 Phone Layout: floating thumbstick on left 40%, sprint ring, sneak toggle, contextual interact, quick-use, throw, 5-slot swipeable hotbar, dynamic infection, weight, and noise meters. *(COMPLETE)* |
| **Z0C** | **Animation Foundation** | Shared humanoid rig (Option B: 2D Billboard Cutout Puppets). 4-tier layered state machine (Base, Upper Body, Additive, Override). Blend trees, crossfades, named events (`hit_frame`). Procedural layer: springs, look-at IK, lean into turns, stride matching (zero foot sliding). Fixed 60 Hz loop with alpha pose interpolation and delta clamping ($\le 0.1$s). 16 core clips. Animation Lab (<kbd>F2</kbd>) and test suite (`testPoseContinuity`, `testRequiredClips`, `testEventOrder`). *(COMPLETE)* |
| **Z0D** | **Style Lock & Vertical Slice** | Propose style direction (A pop-up paper diorama, B rubber-hose cartoon, C vinyl toy, or recommended hybrid) for user approval. Author style bible in `docs/STYLE.md` (palettes, shapes, line weights, toon shader ramp, UI sticker kit, comic sound words). Build rendering stack (toon ramp shading, light-tinted outlines, time-of-day LUTs, tilt-shift DoF, bloom, paper grain, blob AO; all toggleable). Vertical slice scene at dusk (player, zombie, pet, car, campfire with Tier 2 IK/faces/springs). Animation Lab live sliders, onion skinning, pose library. Animation Quality & Juice Dial settings. *(COMPLETE — see `docs/PROGRESS.md`)* |
| **Z2** | **Settlements & Spacing Rules** | Procedural town, strip stop, village, farmstead, junkyard, and outpost layouts (grid, strip, radial, cluster, ring-fence) across biomes. Spacing rules and roadside POI clearings. *(COMPLETE)* |
| **Z3** | **Buildings & Pop-Up Roof Fold** | Cardboard pop-up roof flap fold-away upon doorway entry, interior wall transparency, reactive grass, footprints and tracks, cartoon destruction effects, window peek-holes, and hiding-spot animations (closets, dumpsters). *(COMPLETE)* |
| **Z4** | **Containers & High Loot System** | Hinged container lid animations with rubbery spring bounce. Parabolic loot ejection arcs with ground settle. High-loot gold light beams, jackpot emergency alarm sirens, cartoon destruction effects. |
| **Z5** | **Crowd Zombie Engine & Crowd Animation** | Shambler, Jogger, Traffic Cone zombies. Zombie gait variations, screamer head-turn waves, the pile-up. 3 LOD tiers (LOD0 full, LOD1 throttled 15-20 Hz, LOD2 instanced impostor). Baked pose textures, distant tick throttling, green-puff cartoon pops. Profiled against $\le 2\text{ ms}$ laptop (150 zombies) / $\le 4\text{ ms}$ phone (60 zombies) budget. |
| **Z6** | **Arcade Vehicle Physics & Driving Cockpit** | Single-player `rusty_hatchback`. Weight transfer, body roll in turns, nose dip on braking, visible steering wheel, procedural wheel spin ($\omega=v/r$), suspension pitch, door open/close tweens, damage state swaps, headlights, trunk access. Artboard 2 Phone Driving Layout. |
| **Z7** | **Car Part-Repair System & Horde Nights** | Cards UI bottom sheet, ghost outlines, hold-to-repair radial ring fill tween, part-fit bounce snap animations, hand crank mini-game, fuel siphoning, loose jank parts wobbling, hot tub slosh. Highway Hulk boss & 7th-day dusk siren horde event. |
| **Z8** | **Cartoon Weapons, Defenses & Base Construction** | Slapstick melee, ranged, and throwables with drag-aim. Barricades and building kit pieces snapping into socket positions with dust puff animations. |
| **Z9** | **Base Building & Shelters** | Claim pennant, safe rings, generator fuel burn vs lights trade-off, base siege horde defense. |
| **Z9B** | **Vehicle Fleet Expansion 1: Big Rigs, Trucks & Trailers** | Tow truck, semi rig, food truck, fire truck. Trailer pendulum hinge wobble physics, weak bridge load collapse limits, per-class reachability validator. |
| **Z9C** | **Vehicle Fleet Expansion 2: Bikes, Boats, Snow, Swamp & Rail** | Bicycle, quad bike, jet ski, trawler, hovercraft, rail handcar. Boat water bobbing and wake ripple waves, hull patching mini-game, Artboard 5 Phone Boat Layout. |
| **Z9D** | **Vehicle Fleet Expansion 3: Aircraft & The Whirlybird** | Whirlybird helicopter arcade flight model (3 altitude bands, hover assist, spinning rotor blur disc, winch rope as a verlet chain, prop wash on grass and water, landing squish, rooftop helipads, bird strikes, 4s rotor spool-up). Artboard 4 Whirlybird Phone Layout. |
| **Z10** | **The Great Escape Questline & Combat Hit Events** | Rescue radio chain, evac dock defense event, 3 escape routes. Attack lunge warping, smear frames, weapon trails, directional flinches, full combat feel with frame-accurate `hit_frame` damage application, hit reactions, entity-isolated hit-stop. |
| **Z10B** | **Quadruped & Critter Animation Rigs** | Scrappy pup, camp cat, pocket sloth, wildlife critters. Pet spring tails and ears, zoomies, facial expressions, reactions to music, 4-legged stride matching, pet interaction clips. |
| **Z10C** | **Avian & Exotic Pet Rigs** | Honk goose pet, disco parrot, backpack possum. Wing flaps, neck wobble, pet petting interaction with heart sparkles. |
| **Z11** | **Multiplayer Vehicle, Zombie & Co-op Sync** | Server-owned authoritative vehicles, client-side driver prediction, shared repair mini-games, interest management. |
| **Z12** | **Multiplayer Animation State Sync** | Network protocol sending animation state names and locomotion parameters only. Remote entities rendered ~100 ms in the past using client interpolation buffer; zero visual snapping. |
| **Z13** | **Accessibility, Layout Editor & Animation Polish** | Final animation and style tuning pass in Animation Lab. Reduced-motion settings (disables camera punch and squash exaggeration). Mobile animation LOD scaling (Tier 1 + reduced Tier 2). UI animation polish. Touch layout drag/scale editor (80–130%), gamepad support. |
| **Z26** | **Status Effects, Reactions & Stagger** | The 12 effects of §115 as `EffectDef` data rows (source, does, ends, icon, sound word, tooltip). Reaction table of §116 as data keyed by effect × surface tag, with a chain depth limit. Poise and the four-step stagger ladder of §117. Caps on fires, puddles and chains; reduced-flashing setting; server-owned effects (§118). Same rows for bros, zombies, raiders, vehicles and structures. *(IN PROGRESS — data layer done 2026-10-19 (`src/data/effects.ts`, schema, validation); sim, render and tests planned step by step in `world4bros-core/docs/Z26_PLAN.md`)* |
| **Z27** | **Biome Atlas Expansion** | Hazards, effect lists, props, ambient feel and spawn tables for the 10 existing biomes as data (§119). Add the 6 proposed biomes (Toxic Marsh, Cinder Wastes, Storm Plains, Salt Flats, Crystal Caverns, Ruined Downtown) as new `BiomeDef` entries plus a few drawn props each, with no engine change (§120). Needs Z26 first. *(PLANNED — design approved 2026-10-19, see `docs/design/biomes.html`)* |
| **Z28** | **Aim Arena & Test Bench** | Two arenas as data (Duel Pit 40×24 m, Free-for-all Yard 64×40 m) with cover that stops bullets; offline bot duel with four bot levels (Scarecrow, Regular, Sharp, Menace), rounds, score and results; a deterministic headless Test Bench (Watch and Data run, telemetry, weapon matchup matrix); then online 1v1 and free-for-all. Steps Z28a–e in `world4bros-core/docs/ARENA.md`. *(IN PROGRESS — Z28a arenas done, bot levels built; rounds, score and match HUD next)* |
| **Z29** | **Weapon Builder, Gunsmith & Info HUD** | Pure `src/weapons` module (part catalog, profiles, `canFit`, derived stats, quirks, six new guns), game wiring (loadouts, save, effective gun in `shootGun`, melee quirks, zombie stun), the Gunsmith panel (key `U`, desktop then phone), gated info-HUD overlays, and junk and part loot tables with a balance pass. Steps Z29a–e in `world4bros-core/docs/WEAPONS.md`. Builds on the existing attachment system and gunsmith bench. *(PLANNED)* |
| **Z30** | **Town Houses, Yards & HD Graphics Pass** | Houses face the street at a fixed setback with a porch, stepping-stone path, kerb mailbox, picket-fenced back yard, vegetable beds and porch lantern; each building picks a colour set from `BuildingDef.palettes` deterministically and roofs keep their colour (§26). Still to do: gable roofs with lit and shadow sides, four-face hip roofs, shingles, ink outlines, dormers; shutters, curtains, coloured doors, vines; yard props; a looser small-town layout; save `yard`/`front` in the static map (`staticmap.ts`); roof and feature registries so new styles are data only. Parallel HD passes: roads and ground, street props, vehicles, lighting, characters. *(IN PROGRESS — houses done 2026-10-19, `tests/buildings.test.ts`)* |

---

## 70. Combined Acceptance Criteria
1. **Map & Biome Preservation**: Base map noise, terrain heights, palettes, biomes, landmarks, and hazards must remain 100% untouched. All new features exist as purely additive layers verified by biome snapshot comparisons.
2. **Classic Mode Independence**: Classic mode must stay fully operational. When `mode === 'classic'`, all roads, zombies, vehicles, and expansion UI are cleanly hidden and inactive.
3. **Pure Data Registries**: All items, vehicles, parts, zombies, containers, roads, and POIs must be defined in pure TypeScript data files with `snake_case` IDs. Adding new content must require only data definitions, never engine modifications.
4. **Zero Foot Sliding**: Locomotion animation playback speed dynamically locks to actual ground velocity $\vec{v}$ via stride matching.
5. **Pose Continuity**: State machine transitions must crossfade smoothly with zero visual joint pops, verified by automated `testPoseContinuity` thresholds.
6. **Strict Animation & Crowd Performance Budget**:
   - Laptop / Desktop: Animation update and skinning must stay under **$\le 2$ ms per frame** with 150 active zombies. 60 fps steady.
   - Mobile Phone: Animation must stay under **$\le 4$ ms per frame** with 60 active zombies. 45 fps steady.
7. **Single-Source State Mapping**: Every entity maps game state to animation state through a single function (`entity.updateAnimationState()`), with zero scattered `play()` calls in game logic.
8. **Frame-Accurate Combat Synchronization**: Damage, impact audio, screen punch, and particle sparks are triggered strictly upon the `hit_frame` animation event.
9. **Slapstick Cartoon Tone**: Zero blood, gore, or real-world brands. Defeated zombies pop into green puffs. Pets never die.
10. **Headless & Network Decoupling**: Game rules must run in pure TypeScript with seeded RNG, fully testable without WebGL or DOM dependencies. Multiplayer transmits state names and parameters only, rendering remote entities ~100 ms in the past.
11. **Vertical Slice Sign-Off Gate**: The vertical slice in Phase Z0D must be approved by the user before visual assets or animation content are scaled up across all entities.
12. **Tiered Animation Budget**: Tier 2 features (IK, facial expressions, spring bones) automatically switch off by distance and on phones, staying strictly within the 2 ms (laptop) and 4 ms (phone) animation budgets.
13. **Combat Wind-up, Smear & Entity-Isolated Hit-Stop**: Every attack must feature clear wind-up, dynamic smear frames, impact, and follow-through. Hit-stop freeze frames (~45 ms) are isolated strictly to the attacker and victim entities, keeping the rest of the world and camera rendering smoothly.

---

## 71. Workflow Contract & Phase Sign-Off Rules
- Work strictly **ONE phase at a time** in the sequence specified in Section 69. Never start or preview the next phase prematurely.
- Phase kickoff requires: codebase inspection, reusable assets inventory, gap list, and short implementation plan, followed by `[AWAITING_APPROVAL]`.
- Phase completion requires: running the game, playing it, verifying console cleanliness, taking screenshots, updating `docs/PROGRESS.md`, listing tunable constants, and outputting `[AWAITING_APPROVAL]`.

---

# Part 9: Multi-Agent Swarm Architecture & Collaboration Rules

## 72. Working with Multiple Agents (Swarm Rules)
To safely scale expansion development across parallel agents without collisions, race conditions, or architecture divergence, follow these rules:

### 1. Sequential Foundation Rule
- **Phases Z0, Z0B, and Z0C MUST be completed sequentially by a single Lead Agent.**
- Parallel swarm tracks must NOT be spawned until Phase Z0C (Animation Foundation) is fully completed, verified, and approved.
- All downstream tracks depend directly on the schema contracts (Z0), the universal input buffer (Z0B), and the shared skeletal/puppet animation state machine (Z0C).

### 2. Swarm Roles & Track Allocation
Once Z0C is approved, work can split into distinct agent tracks:
- **Lead Agent (Architect & Reviewer)**:
  - Supervises task allocation, merges code, and conducts architecture audits.
  - Writes **no feature code**; sole owner of schema contract modifications in `src/registries/`.
- **Worker Agents (1 per domain, max 2–3 active concurrently)**:
  - **World Track**: Owns `src/world/` and settlement/building generation (`Z2`, `Z3`, `Z4`, `Z9`).
  - **Zombie Track**: Owns zombie crowd rendering, LOD impostors, and flow fields (`Z5`).
  - **Vehicle Track**: Owns vehicle physics, repair sheets, and fleet mechanics (`Z6`, `Z7`, `Z9B`, `Z9C`, `Z9D`).
  - **Pet Track**: Owns pet rigs and interaction behaviors (`Z10B`, `Z10C`).
- **Data Agent**:
  - Populates registry files in `src/registries/` without touching engine core or rendering.
- **Test Agent**:
  - Executes automated unit tests, biome preservation snapshots, A* graph reachability checks, and build validation.

### 3. Collision Prevention & Merge Gates
- **Contract Boundary**: Shared schema contracts in `src/registries/` can only be modified with Lead Agent approval.
- **Isolated Worktrees**: Every parallel agent works in its own dedicated branch or git worktree.
- **One-at-a-Time Merges**: Branches are merged sequentially, never in bulk.
- **Strict Quality Gates**: No branch merges unless:
  1. `npm run build` (`tsc && vite build`) passes with zero errors or warnings.
  2. Biome preservation snapshot comparison passes with zero regressions.
  3. A* graph reachability validator verifies 100% route connectivity to all POIs.
  4. Console logs remain completely clean with zero runtime exceptions.
- **Sign-Off Protocol**: Every worker updates `docs/PROGRESS.md` with commit hashes and tunable parameters, then outputs `[AWAITING_APPROVAL]` and halts until approved by the Lead/User.

---

# Part 10: Style and Animation Direction

## 73. The Pop-Up Paper & Rubber-Hose Hybrid Visual Style
- **The Core Aesthetic**: A handcrafted pop-up cartoon diorama combining cut-paper storybook layers with classic rubber-hose elasticity. Chunky, bouncy, warm, and goofy. The zombie apocalypse is framed as comedic chaos rather than grim survival horror.
- **Visual Materials**:
  - Cut-paper edges with subtle layered depth and cardboard thicknesses.
  - Light paper grain texture across flat shaded fills, avoiding sterile computer-flat colors.
  - Creased folding flaps on buildings, signposts, and props that visually stand up from the board.
- **Atmosphere**: Inviting miniature model world. The player feels like they are looking down into an animated mechanical pop-up book with living puppet characters.

## 74. Shape Language, Line Weight & Light-Tinted Outlines
- **Shape Language**:
  - **Friends & Companions (Player, Pets, Town NPCs)**: Soft, round, bulbous curves with warm silhouettes. Zero sharp aggressive spikes.
  - **Zombies**: Asymmetrical, lumpy, wobbly, and doughy. Oversized bobbing jaws, mismatched googly eyes, and floppy limbs. They look goofy and clumsy rather than menacing.
  - **Real Danger (Traps, Boss Attacks, Bosses, Explosives)**: Angular, jagged, and exaggerated geometric accents reserved strictly for active lethal threats.
- **Line Weight & Outlines**:
  - Variable-width brush outlines that taper at tips and thicken at corners to emulate hand-inked illustration.
  - Outlines are **never pure pitch black (`#000000`)**; they are dynamically tinted by environmental lighting (e.g. deep charcoal-indigo in shadow, rich warm auburn in sunlight, eerie plum at night).
  - Optional subtle hand-drawn jitter pulse (toggled in Juice Dial) that evokes 2s/3s hand-drawn animation while on-foot locomotion and physics stay silky 60 fps.

## 75. Biome Color Palettes, Shading Ramps & Color-Grading LUTs
- **Color Discipline**: A disciplined, curated 4-to-6 swatch palette per biome preventing visual noise.
- **Lighting Dynamics**:
  - Shading ramp: Toon-stepped Cel shader (3-band ramp) with rim light highlighting edges against dark backgrounds.
  - Warm/Cool Color Temperature Shift: Key lights shift warmer (golden amber, butter yellow, peachy daylight), while ambient shadows shift cooler (periwinkle, slate teal, deep indigo).
- **Universal Semantic Color Coding**:
  - **Gold Sparkle (`#FFD700`)**: Valuable loot, repairable part drops, and high-value rewards.
  - **Teal / Cyan (`#00E5FF`)**: Interactable items, friendly vehicles, door handles, and quest targets.
  - **Red-Orange (`#FF3D00`)**: Lethal hazards, trap triggers, boss warnings, and low-health emergencies.
- **Color-Grading LUTs**: Custom look-up color grades for Dawn, Noon, Dusk, Night, and Horde Siren states across all 10 biomes.
- **Built in the 2D core (Z30, 2026-10-19)**: sun drop shadows that swing and stretch with the time of day, violet night with warm porch and window pools, cosmetic weather per half day from the biome climate (clear, breezy, overcast, rain, storm, fog, snow, dust haze) with wet ground, puddles, rain, snow, leaves, mist and fireflies, plus vignette and paper grain, all behind an off/low/high quality dial. Weather is visual only for now; gameplay weather (noise, vision, footprints) will reuse the same data rows. Details: `docs/STYLE.md` §9.3, `world4bros-core/src/render/atmosphere.ts`.

## 76. Diorama & Pop-Up Camera Tricks
- **Tilt-Shift Depth of Field**: Subtle miniature-style top and bottom edge blur, enhancing the tactile illusion of a tiny diorama under a macro lens.
- **Foreground Cutout Framing**: Silhouetted paper cutout leaves, telephone wires, or canopy branches subtly floating in the extreme foreground to create theatrical diorama framing.
- **Pop-Up Building Unfolds**: Buildings flap open like architectural pop-up books when approached or entered, revealing dollhouse cross-sections.
- **Map Transitions**: Map changes and fast travels use a comic paper-page flip or cardboard slider wipe instead of generic black fade.

## 77. UI Sticker Kit & Comic Sound Words
- **Sticker & Tape UI Aesthetic**:
  - Interface panels, health cards, and buttons designed as physical stickers, paper labels, and masking tape scraps stuck directly to the diorama frame.
  - Slight randomized sticker tilt angles ($-1.5^\circ$ to $+1.5^\circ$) with peel shadows.
- **Comic Sound Word FX ("Onomatopoeia Pops")**:
  - Frame-accurate 2D comic sound words pop up at point of impact:
    - `"BONK!"` for blunt melee hits (frying pan, baseball bat).
    - `"CLANG!"` for metal wrecks, trash cans, and repair impacts.
    - `"SQUEAK!"` for rubber chicken swings and puppy steps.
    - `"POP!"` for defeated zombie green smoke clouds.
  - Scale tween: Spawns at scale $0.2$, punches to $1.3$ with elastic bounce, then floats upward and fades in $0.4$s.

## 78. Layered Character Locomotion & Expressive Body Language
- **Locomotion Mechanics**:
  - Characters lean dynamically into turns ($\theta = \text{clamp}(v \cdot \omega, -15^\circ, 15^\circ)$).
  - Skidding to a stop produces dust puffs, shoe squeaks, and backwards torso counter-balance.
  - Foot IK keeps shoes firmly planted on slopes and stairs without clipping into geometry.
- **Secondary Spring Bones**:
  - Hats, ponytails, floppy dog ears, cat tails, backpacks, and cape corners follow through using harmonic spring dampers ($f=5\text{ Hz}, \zeta=0.7$).
- **Expressive Facial Puppetry**:
  - Automatic eye blinks on randomized timers ($2.5 - 6.0$s interval).
  - Gaze tracking / look-at IK toward nearby points of interest (looted crates, barking dogs, charging screamers).
  - Mood morphs: Scared (wide eyes, shaking pupils), Sleepy (drooping lids), Dizzy (spiraling pupils), Happy (crescent smiles).
- **Physical Hands & Tool Grips**:
  - Real hand IK grips tool handles, axe hafts, and steering wheels.
- **Body Language of Survival Stats**:
  - High backpack weight ($\ge 35\text{ kg}$) causes hunched posture and heavy lagging bag sway.
  - Freezing cold triggers shivering teeth chatter and arms tucked to torso.
  - Severe hunger ($< 25\%$) triggers tummy rumbles, clutching stomach, and slowed gait.
  - Infection ($> 50\%$) displays animated creeping neon-green veins and an occasional comic stumble.

## 79. Slapstick Combat Feel
- **Four-Phase Combat Animation**:
  1. **Wind-Up**: Exaggerated anticipation pose with character stretching backward against motion direction.
  2. **Smear Frame**: 1-to-2 frames of comic motion blur / stretched geometry along swing trajectory.
  3. **Impact**: Frame-accurate `hit_frame` event:
     - Applies damage and triggers comic sound words.
     - Spawns directional impact sparks and comic stars.
     - Triggers **entity-isolated hit-stop**: Attacker and victim freeze in place for $45\text{ ms}$, while camera, particles, and surrounding world continue updating at full 60 fps.
  4. **Follow-Through & Recovery**: Springy overshoot and settled return to idle.
- **Zombie Horde Dynamics**:
  - Randomized gait offsets: No two shamblers walk with identical limb phases or cadences.
  - Screamer Wave: When a screamer unleashes a screech, a ripple wave of head-turns cascades outwards through the zombie crowd.
  - Cartoon Pile-Up: Zombies tumbling over barricades squash into comical ragdoll stacks.

## 80. Reactive World & Environmental Secondary Motion
- **Reactive Foliage**: Grass tufts, flower petals, and shrub branches dynamically bend away from player, zombie, and vehicle colliders using spring physics.
- **Deformable Footprints & Tire Tracks**: Snow, mud, and sand retain stamped footprint decals and continuous tire track ribbons that gradually fade over $30$s.
- **Tree Felling FX**: Trees fall with an energetic wobble, ground bounce, and explosive burst of paper leaves and pine needles.
- **Vehicle Secondary Physics**:
  - Car nose dips forward under heavy braking, rears up during acceleration.
  - Chassis rolls outward during high-speed drifts.
  - Jank parts (loose bumpers, open hoods, swinging dice) rattle and wobble.
  - Hot tub truck water sloshes dynamically with foam particles.
  - Whirlybird helicopter winch functions as a physical swinging Verlet rope chain.

## 81. Performance Tiering & Settings
- **Three-Tier Animation LOD Architecture**:
  - **Tier 1 (Base - All Entities)**: Basic blending, event dispatch, locomotion stride matching. Runs on all 150+ zombies and background critters.
  - **Tier 2 (Rich - Player, Pets, Active Vehicle, Nearby Enemies $< 15$m)**: Foot IK, hand grips, facial expressions, secondary spring bones, body language stat overlays.
  - **Tier 3 (Hero - Bosses, Whirlybird Helicopter, Player Close-ups)**: Full verlet physics, custom smear meshes, multi-layered particle trails, hero camera framing.
- **Mobile Phones & Low-End Tiering**:
  - Phones execute Tier 1 plus a reduced Tier 2 (springs simplified, facial morphs throttled, IK restricted to player).
  - Automatic distance culling: Tier 2 features step down to Tier 1 beyond $15$m distance.
  - Strict CPU Budgets: $\le 2\text{ ms}$ on laptop/desktop (150 zombies), $\le 4\text{ ms}$ on mobile (60 zombies).
- **Settings & User Sliders**:
  - **Animation Quality Setting**: Low (Tier 1 only, no DoF), Medium (Tier 1 + Tier 2, light DoF), High (Full Tier 1/2/3, tilt-shift, bloom, paper grain).
  - **Juice Dial (0% to 100%)**: Scales screen shake, cartoon squash/stretch exaggeration, onomatopoeia popup sizes, and outline jitter intensity.

---

# Part 11: Factory Automation & Logistics Engine (Factorio & Satisfactory Integration)

## 82. The Core Automation Pipeline & Discrete Resource Nodes
To bridge manual survival scavenging with mid-to-endgame mass production, the world generates discrete **Resource Geodes / Mineral Nodes** with purity ratings adapted from *Satisfactory*:
- **Purity Tiers**:
  - **Impure (0.5x yield)**: Slower extraction, found frequently near starter meadow and roadsides.
  - **Normal (1.0x yield)**: Standard output rate, balanced across mid-tier biomes.
  - **Pure (2.0x yield)**: Massive output velocity, located in dangerous hazard regions (The Hollow, Mount Grumbleguts, Scrapyard Gulch).
  - **Jackpot Geode (4.0x yield)**: Ultra-rare glowing formation, highly contested, guarded by screamer nests or boss hulks.
- **Node Resources**:
  - `iron_deposit`: Smelted into `iron_plate`, `iron_rod`, and `heavy_girder`.
  - `copper_deposit`: Drawn into `copper_wire`, `circuit_board`, and `spark_plug`.
  - `coal_seam`: Fuel for burner drills, steam boilers, and carbon steel alloy.
  - `limestone_quarry`: Crushed into `quick_cement` for reinforced factory slabs and pillbox walls.
  - `crude_oil_gusher`: Fluid node extracted with pumps, refined into `kerosene`, `rubber_sole`, and `napalm`.
  - `raw_bauxite`: Refined into lightweight `aluminum_sheet` for advanced vehicles and drones.
  - `glow_crystal_geode`: Mined for `energetic_quartz` powering laser turrets, overclocking modules, and Tesla battery banks.

## 83. Conveyor Belts, Splitters, Inserters & Vertical Lifts
Transporting thousands of raw chunks without inventory micro-management:
- **`wooden_chute_belt` (Tier 1)**: Gravity-assisted wooden trough with rolling logs. Speed: $6\text{ items/s}$. Cannot move uphill.
- **`rubber_conveyor_belt` (Tier 2)**: Motorized rubber belt with guide rails. Speed: $15\text{ items/s}$. Can slope up to $30^\circ$.
- **`mag_speed_belt` (Tier 3)**: High-speed electromagnetic track. Speed: $30\text{ items/s}$. Items click along with glowing neon guide lines.
- **`vertical_conveyor_lift`**: Telescoping cardboard/metal elevator tower allowing vertical multi-deck factory architecture (1 to 4 floors up).
- **`belt_splitter_merger`**: 3-in / 3-out modular block. Supports 50/50 division, priority lane output, and smart item filtering (e.g., send coal to boilers, iron ore to smelters).
- **`mechanical_spring_inserter`**: Spring-loaded cartoon grabber arm. Reaches into belt lane, grabs designated item, and drops it into a machine hopper or chest.
  - *Fast Inserter*: Upgraded gearbox for rapid throughput.
  - *Filter Inserter*: Only grabs items matching its chalked blackboard icon.
  - *Stack Inserter*: Grabs up to 5 items simultaneously with an oversized cartoon scoop.

## 84. Energy Grids, Fuel Cycles & Fluid Manifolds
Automation requires juice. If power cuts, factories grind to a halt and defense turrets power down:
- **Power Generation Hierarchy**:
  - `burner_wood_generator`: Starter engine running on logs and scrap twigs. Produces $75\text{ kW}$, billows comical black soot.
  - `steam_boiler_turbine`: Consumes water from pipes and coal/oil fuel. Produces $500\text{ kW}$ of steady electricity.
  - `wind_whirler`: Passive eco-power generating $50 - 150\text{ kW}$ depending on weather (windy storms provide huge surges).
  - `solar_pop_array`: Clean daytime power ($200\text{ kW}$), zero output at night. Requires battery banks.
  - `tesla_accumulator_bank`: Giant chemical capacitor storing surplus day/steam power to keep base defenses running during midnight sieges.
  - `geothermal_tap`: Placed on volcano lava vents. Produces $2.5\text{ MW}$ of infinite energy, but periodically vents superheated steam.
- **Power Grid Mechanics**:
  - `wooden_telegraph_pole`: Links power up to $16\text{ m}$ radius with visible sagging physics wires.
  - `high_voltage_pylon`: Heavy steel lattice carrying trunk power lines between remote mining outposts and base camp across $64\text{ m}$.
  - `substation_kiosk`: Area-of-effect power distributor covering a $32 \times 32\text{ m}$ factory floor without messy wire tangles.
- **Fluids & Pipe Manifolds**:
  - Liquids (water, crude oil, heavy fuel, chemical goo) flow through `galvanized_pipe_network`.
  - Driven by `inline_booster_pump` to maintain head pressure across slopes and long distances.
  - `fluid_storage_tank`: 25,000L cylindrical vat with an external float gauge that bobs with current volume.

## 85. Automated Industrial Machinery
Players construct self-contained automated production loops:
- **`burner_mining_drill` & `electric_quarry_extractor`**: Placed directly on mineral nodes. Churns out raw ore with chunky ground-thumping animation and onomatopoeia `"CHUNK-CHUNK!"`.
- **`smack_crusher`**: Heavy reciprocating piston that pulverizes boulders and slag into gravel, sand, and mineral concentrates.
- **`blast_furnace`**: Smelts raw ores into uniform metal ingots using fuel or electricity. Emits heat shimmer and sparks.
- **`auto_assembler_mk1 / mk2 / mk3`**: The heart of the factory. Takes up to 4 conveyor inputs, uses rotating robotic arms to assemble gears, motors, ammunition boxes, car parts, and gadgets.
- **`chemical_cracker_refinery`**: Fractionates crude oil into gasoline, plastic pellets, sulfur, and synthetic lubricant.
- **`modular_overclocking_socket`**: Every machine has 1 to 3 power shard slots. Inserting `glow_crystal_shards` overclocks production speed up to 250%, at the cost of quadrupled energy consumption and massive pollution spikes.

## 86. Macro Logistics: Freight Trucks, Automated Rail & Pneumatic Hypertubes
Scaling production across the continental map without running 5-kilometer conveyor belts:
- **`automated_freight_tractor`**: Small 4-wheeled tractor with a cargo trailer. Players record a driving loop between two `loading_dock_depots`; the tractor automatically follows the waypoint track, refueling and transporting 48 cargo slots back and forth.
- **`clockwork_narrow_gauge_rail`**: Placeable train tracks with snappy curves and switches.
  - `steam_locomotive`: High-speed bulk hauler towing up to 6 `ore_hopper_cars` or `fluid_tank_cars`. Features a working cartoon cowcatcher that flings roaming zombies into the air with a loud `"TOOT-TOOT!"`.
- **`pneumatic_hypertube`**: A goofy transparent cardboard tube system adapted from *Satisfactory*. Players jump into the intake suction ring, hear a vacuum `"SCHWOOP!"`, and are propelled through curves and over mountains at $90\text{ km/h}$, popping out with a comedy parachute landing.

## 87. Industrial Heatmaps: Pollution Cloud, Noise & Ground Vibration
Industrial automation is not free. Every running machine produces two environmental footprints:
1. **Pollution & Smog (Chemical)**:
   - Coal boilers, blast furnaces, and oil crackers emit a visible murky smudge that diffuses through the chunk splat map.
   - Cleansed naturally over time by nearby pine forests, bogs, and water bodies.
   - Infiltrates zombie hives, triggering evolutionary mutations (see Section 94).
2. **Noise & Ground Vibration (Acoustic)**:
   - Crushers, drills, and punch presses generate continuous acoustic decibels.
   - Decibel radius attracts nearby wandering hordes directly toward the loud bottleneck (e.g. miners or generators), creating emergent factory defense challenges.

---

# Part 12: Colony Management, Pawn Psychology & Storyteller System (RimWorld Integration)

## 88. Survivor Colonists, Trait Archetypes & Passions
Rescuing survivors from roadside wrecks, town attics, and bandit cages allows players to build a thriving communal colony. Every NPC bro possesses distinct personality traits and skill passions:
- **Core Skills**:
  - `construction`: Speed and HP of built structures and conveyor belts.
  - `mining`: Ore extraction efficiency and rare gem yields.
  - `crafting`: Crafting quality of weapons, armor, and precision machine parts.
  - `cooking`: Meal quality, hunger restoration, and zero food-poisoning risk.
  - `medicine`: Speed and success rate of treating injuries, infections, and surgeries.
  - `plants`: Crop harvest yields and wood chopping velocity.
  - `combat`: Accuracy, reload speed, and bravery in firefights.
- **Passions**: Zero flame (normal XP), single flame (passionate, +50% XP), double flame (burning passion, +100% XP and constant positive mood while working).
- **Goofy Character Traits**:
  - `grease_monkey`: Fixes vehicles and machines 50% faster, loves sleeping in the garage.
  - `pyromaniac`: Occasionally starts celebratory campfires or fireworks during mental breaks.
  - `slacker`: Walks 20% slower, takes frequent coffee breaks, but never suffers stress breaks.
  - `brawler`: Refuses ranged weapons; swings rubber chickens and bats with +40% attack speed.
  - `night_owl`: +20 mood when working between 10 PM and 6 AM, grumbles during the day.
  - `gourmand`: Demands hot cooked meals; eats double portions but cooks 5-star stews.
  - `nervous_wreck`: Breaks under mild stress, but has a 10% chance of sudden genius work sprees.
  - `clumsy`: Occasionally drops items off conveyor belts or trips over banana peels with a squeak.

## 89. Mood Dynamics, Room Impressiveness & Comedic Slapstick Mental Breaks
Colonists maintain an active **Mood Bar (0% to 100%)** driven by environment, comfort, and social dynamics:
- **Positive Mood Boosters**:
  - *Lavish Stew (+12)*: Ate a delicious hot meal cooked in a cast iron pot.
  - *Opulent Bedroom (+8)*: Slept in a decorated room with a feather bed, rug, and nightstand.
  - *Ate at a Dining Table (+5)*: Didn't have to eat sitting in the mud!
  - *High Recreation (+10)*: Played horseshoe toss, listened to radio DJ midnight, or danced at the disco stump.
  - *Base Victory (+15)*: Survived a horde night or defeated a boss hulk together.
- **Negative Debuffs**:
  - *Ate Without Table (-4)*: Classic complaint; forced to eat raw berries on the dirt.
  - *Slept in the Cold (-8)*: No heater or fireplace in the sleeping quarters.
  - *Ugly Environment (-6)*: Surrounded by rotting zombie smoke, spilled oil, or unhauled scrap.
  - *Exhausted & Starving (-15)*: Pushed past physical limits without rest.
- **Slapstick Mental Breaks (Minor, Major, Extreme)**:
  - *Binge Eating (Minor)*: Marches directly to the food chest and devours all the canned beans and strawberry jams.
  - *Hide in Outhouse (Minor)*: Locks themselves in the flush teleporter outhouse for 4 hours of quiet time.
  - *Tool Tantrum (Major)*: Grabs a mallet and smacks an empty oil barrel repeatedly with comedic `"BONK-BONK-BONK!"`.
  - *Conveyor Carousel (Major)*: Sits down on the main iron ore conveyor belt and rides in circles through the factory, refusing to get off.
  - *Berserk Pillow Fight (Extreme)*: Equips a feather pillow and attacks the nearest bro or dog with high knockback and zero damage.

## 90. The Priority Work Matrix (The 1-4 Role Grid)
Accessible via the Colony Overseer Tab (`Key: O`). Players assign priority numbers ($1 = \text{Top Emergency}, 4 = \text{Low Idle}$, or Blank $=$ Disabled) across work categories:
- **Priority Columns**:
  1. `Firefight`: Extinguish burning machines, campfires, and flame traps.
  2. `Patient`: Seek bed rest when wounded or infected.
  3. `Doctor`: Tend injured teammates, apply bandages, perform surgeries.
  4. `Basic`: Flip emergency switches, refill generator fuel hoppers.
  5. `Build`: Construct blueprints, repair damaged walls, assemble belts.
  6. `Farm`: Plant potato patches, prune berry bushes, collect honey.
  7. `Mine`: Extract mineral boulders and quarry nodes.
  8. `Haul`: Move loose items from ground/machines into designated stockpile zones and chests.
  9. `Craft`: Operate assembler benches, cook stoves, and repair stations.
  10. `Guard`: Man watchtowers, patrol perimeter walls, operate manual turrets.
  11. `Research`: Read blueprints at the chalkboard tech table to unlock new technologies.

## 91. Colony Architecture: Barracks, Mess Halls, Workshops & Medical Triage
Building functional rooms with walls, doors, and flooring transforms a rugged camp into a bustling town:
- **Room Types & Impressiveness Ratings**:
  - `communal_barracks`: Multiple cots in one room. Cheap and space-efficient, but gives a mild *"Disturbed Sleep"* debuff.
  - `private_bedroom`: 1 bed, 1 dresser, 1 lantern, wooden flooring. Grants a permanent positive mood boost.
  - `grand_mess_hall`: Long oak banquet table, benches, and a dart board. Bros gather here at noon and dusk to socialize.
  - `factory_workshop`: Concrete slabs, tool racks, and bright electric fluorescent lamps, speeding up crafting operations by +25%.
  - `medical_triage_bay`: Sterile linoleum floors, clean medical cots, and blood-transfusion IV stands (filled with cherry soda). Accelerates wound healing by 300%.
- **Slapstick Medical System**:
  - Injuries track specific body parts (Left Leg, Right Arm, Eyes, Torso).
  - Injuries cause humorous visual bandages (oversized head wrappings, arms in slings).
  - **Goofy Prosthetics**:
    - `wooden_peg_leg`: Restores 85% movement speed with a rhythmic wooden clack.
    - `turbo_spring_boot`: Replaces foot with a coiled pogo spring (+120% jump height, squeak on landing).
    - `wrench_prosthetic_arm`: Replaces hand with a chrome adjustable socket wrench (+30% repair speed).
    - `bionic_googly_eye`: Glows in the dark and detects stealth traps.

## 92. Dynamic Storyteller AI Engine
The game loop is orchestrated by a procedural incident generator tracking colony wealth, player level, and active defenses:
- **The Three Storytellers**:
  - **`Caspar Chill` (The Builder's Friend)**:
    - Focuses on peaceful progression, long intervals between events, frequent friendly wandering merchants, and gentle weather.
  - **`Randy Ruckus` (Pure Slapstick Chaos)**:
    - Entirely randomized, unweighted incident selection. May drop a jackpot crate from an exploding blimp, followed 3 minutes later by a stampede of 30 feral raccoons, a solar flare, and a traveling circus truck.
  - **`Hilda Hardcore` (The Relentless Tactician)**:
    - Calculates colony defense DPS and factory output. Deploys coordinated multi-pronged sieges, sapper squads, chemical toxic smog, and timed supply shortages to push base engineering to its absolute limits.
- **Dynamic Incidents Roster**:
  - *Supply Drop Parachute*: A crate floats down with rare machine parts.
  - *Wandering Scavenger*: A lost survivor arrives asking for shelter and food.
  - *Industrial Short-Circuit*: A overloaded power wire blows a fuse, causing a localized machine fire.
  - *Feral Beast Stampede*: A herd of mutated angry moose or geese charges toward the base.
  - *Toxic Acid Rain*: Forces bros indoors or under umbrellas; corrodes unprotected outdoor machinery over time.
  - *The Screamer Swarm*: A flock of screamer zombies descends upon the loudest sector of the factory.

## 93. Faction Relations, Guest Lodgings, Caravans & Captives
The continental map is populated by multiple procedural NPC factions:
- **Faction Archetypes**:
  - `the_grease_monkeys`: Friendly gearheads based at highway service plazas. Love car parts and refined fuel; offer vehicle blueprints and turbo engines.
  - `salvage_scavengers`: Neutral junk hoarders who barter tools for scrap metal and gold.
  - `hollow_cultists`: Hostile zealots wearing pumpkin masks who raid bases for glow crystals.
  - `road_pirates`: Aggressive motorized raiders patrolling the desert highways in spiked buggies.
- **Caravans & Guest Rooms**:
  - Visiting merchant caravans unpack their stalls in your town square, trading for 24 hours.
  - Building `guest_hotel_rooms` earns rental income (shiny buttons) from visiting travelers.
- **Prisoner & Rehabilitation Mechanics**:
  - Downed hostile raiders can be captured with handcuffs and placed in a `holding_cell`.
  - Bros with high social skill can talk to prisoners, gradually reducing their resistance. Once rehabilitated, they join your colony as full working survivors!

---

# Part 13: Evolved Zombie Threat & Industrial Siege Dynamics

## 94. Pollution-Induced Zombie Mutations
As your factory scales from a small campfire into an industrial powerhouse, smog and noise penetrate surrounding wilderness biomes, causing dormant zombies to mutate into terrifying specialized siege archetypes:
- **`sapper_zombie` (The Demolitionist)**:
  - Carries a rusty powder keg or unexploded bomb.
  - Ignores player combat and sprints directly toward high-value infrastructure (power generators, electrical substations, main gate hinges).
  - Explodes on death with a massive concussive blast that breaches reinforced walls.
- **`acid_spitter` (The Line-Breaker)**:
  - Bloated green zombie that shoots corrosive sludge balls from $22\text{ m}$ range.
  - Sludge melts conveyor belts, jams inserter gearboxes, and destroys electrical poles.
- **`tunnel_burrower` (The Infiltrator)**:
  - Mole-like zombie with oversized excavator claws.
  - Digs underground, bypassing surface spike traps and moat walls, popping up directly inside internal factory courtyards or sleeping barracks!
- **`screamer_queen` (The Catalyst)**:
  - Emits a high-pitched sonic wail that buffs nearby zombies with +50% movement speed and aggros every wandering shambler within a 200m radius.
  - Priority sniper target; must be eliminated before she alerts the entire sector.
- **`goliath_demolisher` (The Super-Hulk)**:
  - Massive 5-meter-tall zombie sporting armor plates made of flattened car doors.
  - Uses a telephone pole as a sweeping club, smashing through stone walls, vehicles, and turrets in single devastating arcs.

## 95. Siege Mechanics: Machine Targeting, Power Sabotage & Chain Alarms
Zombie hordes do not wander mindlessly during high-pollution events; they utilize dynamic tactical AI:
- **Target Value Hierarchy**:
  1. *Acoustic Emitters*: Sirens, running blast furnaces, diesel generators.
  2. *Power Infrastructure*: Vulnerable sagging wire poles and substations. Knocking out power disables automated defenses!
  3. *Turret Chokepoints*: Attempting to flank around turret arcs or overwhelm ammo belts.
  4. *Living Colonists & Players*.
- **The Sapper Wave**:
  - Sappers tunnel or sprint under cover of darkness, attempting to blow breaches in the outer perimeter wall to let hundreds of fast runner zombies pour through.
- **The Chain Screamer Cascade**:
  - If a player or turret fails to take down a screamer quickly, her screech alerts a second screamer in the next chunk, creating a rolling wave of endless reinforcements.

## 96. Automated Base Defense Grids: Belt-Fed Turrets, Catapults & Zap-Fences
To survive night 7 and pollution sieges without firing every bullet manually:
- **`automated_bolter_turret`**:
  - Rapid-fire twin machine gun mounted on a motorized swivel tripod.
  - Requires physical ammo input (`standard_9mm_box` or `heavy_rifle_box`).
  - Players route a conveyor belt of ammo directly behind the turrets; automated inserters load magazines into the turret hoppers!
- **`fire_spit_flame_turret`**:
  - Connected via pipe to a fuel or napalm tank. Shoots a continuous $15\text{ m}$ arc of cartoon liquid fire, turning charging swarms into popping green embers.
- **`rubber_mallet_catapult`**:
  - Heavy defensive catapult loaded with rocks or rubber bowling balls.
  - Launches heavy projectiles into tight zombie crowds, causing comedic mass knockdowns with stars spinning over their heads.
- **`electric_zap_fence`**:
  - Electrified chain-link fence connected to the base power grid.
  - Stuns and electrocutes zombies that touch it, illuminating the night with cartoon skeleton x-ray flashes.
- **`automated_repair_drone_bay`**:
  - Houses 4 flying mechanical repair drones.
  - When walls or turrets take damage, drones launch automatically with glue guns and duct tape, repairing structures in real-time from stockpiled materials.

---

# Part 14: PvPvE Ecosystem, Territorial Factions & Base Raiding

## 97. Tri-Zone World Topology (Green, Yellow, Red Zones)
To balance peaceful factory building and cooperative colony management with high-stakes competitive raiding, the world continent is partitioned into three distinct risk tiers:
- **Green Zones (Civilized Havens)**:
  - *Locations*: Starter Meadow, Abandoned Town Square, Gas Station Trading Plazas.
  - *Rules*: PvP damage is completely disabled. Structures cannot be damaged or raided. Ideal for starter camps, social trading, vehicle customizers, and casual co-op.
- **Yellow Zones (Contested Expansion Biomes)**:
  - *Locations*: Pinewood Thicket, Squelchy Bog, Sunburn Flats, Splash Cove.
  - *Rules*: Opt-in / Consensual PvP (Faction Warfare). Players can flag as Hostile. Resource nodes can be claimed and taxed. Raiding requires a formal Clan War declaration with a 24-hour warm-up countdown.
- **Red Zones (The Lawless Deep Wilderness)**:
  - *Locations*: Scrapyard Gulch, Mount Grumbleguts, The Deep Hollow.
  - *Rules*: Full open PvPvE. Friendly fire enabled. Death drops your complete backpack inventory (worn soulbound hats remain safe). Contains 80% of all Pure and Jackpot mineral nodes. Factories in Red Zones are fair game for spontaneous base assaults and territory conquest!

## 98. Territory Claim Totems, Shield Windows & Anti-Griefing Rules
Preventing "offline raiding" while maintaining thrilling tactical base sieges:
- **`clan_territory_totem`**:
  - Central terminal placed to claim a $96 \times 96\text{ m}$ territory sector.
  - Displays the clan's custom flag, colors, and name on the world map.
- **The Active Raid Window**:
  - Clan leaders set an **Active Defense Window** (e.g., 8:00 PM – 10:00 PM local server time).
  - Outside of this 2-hour window, the totem projects an impenetrable **Tesla Energy Shield**: All exterior walls, gates, and machines are 100% immune to player damage.
  - During the designated 2-hour window, shields power down into *Contested Vulnerability*: Rival clans can attack, breach walls, and attempt to capture the totem!
- **Anti-Griefing Protection**:
  - If a defending clan is completely wiped out, the totem enters a 48-hour *Peace Treaty Lockdown*, preventing repeat raids on the same base.
  - Production machines cannot be permanently deleted; raiders loot inventory chests and siphoned fuel tanks, leaving the core factory framework intact for recovery.

## 99. Base Raiding Mechanics: Breaching Charges, Siege Vehicles & Ambushes
Raiding an enemy fortified factory requires coordinated strategy, specialized tools, and heavy machinery:
- **`c4_plunger_breach_charge`**:
  - Heavy timed explosive placed on enemy gates or reinforced concrete walls.
  - Emits a beeping countdown timer, allowing defenders to shoot the charge to disarm it with a water bucket or bolt cutters!
  - Detonation blows a wide rubble hole through walls.
- **Siege Vehicles & Mobile Rams**:
  - Modifying vehicles with heavy reinforced front rams (e.g. `spiked_cowcatcher_hatchback`, `roadster_rammer`) allows players to smash through wooden barricades and light fences at top speed.
  - The `bathtub_buggy_mortar`: Vehicle-mounted catapult that fires explosive pumpkins over perimeter walls to disable enemy turret power poles.
- **Supply Convoy Ambushes**:
  - In Yellow and Red Zones, automated freight trucks and player supply caravans traveling between mining outposts and the central base can be intercepted.
  - Setting up road spike strips pops truck tires, causing cargo crates to spill across the asphalt for a high-speed heist!

## 100. Inter-Clan Diplomacy, Resource Monopolies & Black Market Trade
- **Alliances & Treaties**:
  - Clans can sign non-aggression pacts, share power grids across territory borders, and establish joint defense pacts against Hilda Hardcore mega-hordes.
- **Resource Monopolies**:
  - Controlling the only Pure Crude Oil Gusher in a region gives a clan immense leverage; they can sell refined aviation gasoline and plastic pellets to rival clans at high markups.
- **The Black Market Radio Station**:
  - Operated by `radio_dj_midnight`. Players post anonymous bounties on enemy clan leaders, order air-dropped contraband weapons, or purchase satellite scans revealing rival base layouts and defense weak spots.

---

# Part 15: Master Automation, Colony & PvP Item / Structure Catalog

## 101. Automation & Logistics Buildings
- **`wooden_conveyor_belt`**: Tier 1 transport. Cost: 2 wood, 1 twig.
- **`rubber_conveyor_belt`**: Tier 2 transport. Cost: 2 rubber, 1 iron_plate.
- **`mag_speed_belt`**: Tier 3 transport. Cost: 2 copper_wire, 1 circuit_board, 1 aluminum_sheet.
- **`conveyor_lift_tower`**: Vertical 1-to-4 floor elevator. Cost: 4 iron_rod, 2 rubber.
- **`splitter_merger_junction`**: 3-in / 3-out sorting block. Cost: 2 iron_plate, 2 copper_wire.
- **`spring_inserter_arm`**: Basic item grabber. Cost: 2 iron_rod, 1 spring, 1 gear.
- **`filter_smart_inserter`**: Chalk-selectable item sorter. Cost: 2 iron_rod, 1 circuit_board, 1 spring.
- **`pneumatic_hypertube_pipe`**: High-speed player vacuum transit. Cost: 4 plastic_pipe, 2 copper_wire.
- **`hypertube_intake_ring`**: Accelerator suction entry point. Cost: 6 iron_plate, 1 turbine_fan.
- **`loading_dock_depot`**: Automated vehicle and train cargo loading terminal. Cost: 12 heavy_girder, 8 conveyor_belt, 1 control_panel.

## 102. Industrial & Power Machinery
- **`burner_drill`**: Surface ore extractor running on solid fuel. Cost: 8 scrap_metal, 4 iron_rod.
- **`electric_quarry_miner`**: High-yield electric drill for pure nodes. Cost: 12 iron_plate, 6 copper_wire, 1 electric_motor.
- **`stone_blast_furnace`**: Smelts ores into ingots and plates. Cost: 16 stone, 4 iron_rod.
- **`auto_assembler_bench`**: Multi-input automated recipe crafter. Cost: 10 heavy_girder, 4 circuit_board, 2 electric_motor.
- **`sludge_cracker_refinery`**: Chemical distillation tower for oil and goo. Cost: 16 steel_plate, 8 copper_pipe, 4 valves.
- **`steam_boiler_engine`**: Water and coal electric generator ($500\text{ kW}$). Cost: 12 iron_plate, 6 copper_pipe.
- **`tesla_accumulator_vault`**: High-capacity industrial battery ($5\text{ MJ}$). Cost: 8 copper_wire, 4 glow_crystal_shards, 6 aluminum_sheet.
- **`overclock_power_shard`**: Crystal module slotted into machines for +50% speed. Cost: 3 energetic_quartz, 1 gold_nugget.

## 103. Colony, Medical & Fun Infrastructure
- **`survivor_cot`**: Basic wooden frame bed. Cost: 4 wood, 2 cloth.
- **`feather_luxury_bed`**: Comfortable bed granting great sleep buffs. Cost: 6 refined_plank, 4 goose_feathers, 2 wool_blanket.
- **`grand_banquet_table`**: Oak mess hall table preventing the "ate without table" tragedy. Cost: 10 refined_plank.
- **`horseshoe_pin_post`**: Outdoor colony recreation game. Cost: 2 wood, 4 scrap_horseshoe.
- **`medical_triage_cot`**: Fast-healing hospital bed with IV stand. Cost: 6 iron_rod, 4 clean_linen, 1 cherry_soda_iv.
- **`prosthetic_peg_leg`**: Wooden replacement leg. Cost: 2 refined_plank, 1 leather_strap.
- **`turbo_spring_boot`**: Bionic pogo limb. Cost: 2 heavy_spring, 1 steel_plate, 2 rubber_sole.
- **`colony_overseer_desk`**: Terminal used to configure the 1-4 Priority Work Matrix. Cost: 6 refined_plank, 1 typewriter.

## 104. Defense Turrets, Traps & Raiding Gear
- **`bolter_sentry_turret`**: Automated twin machine gun. Range: $24\text{ m}$. Belt-fed ammo. Cost: 12 iron_plate, 4 gears, 1 motor.
- **`fire_spit_flame_turret`**: Pipe-fed liquid napalm nozzle. Range: $15\text{ m}$. Cost: 10 steel_plate, 4 copper_pipe, 1 ignition_coil.
- **`mallet_catapult_turret`**: Launches bowling balls and boulders with huge knockback. Cost: 16 wood, 4 heavy_spring, 2 iron_rod.
- **`electric_zap_fence_wire`**: Electrified perimeter fence ($50\text{ kW}$ draw). Cost: 2 wooden_pole, 4 copper_wire, 1 insulator.
- **`automated_repair_drone_bay`**: Drone hangar housing 4 repair bots. Cost: 8 aluminum_sheet, 4 drone_rotors, 2 circuit_board.
- **`c4_plunger_breach_charge`**: Placeable siege explosive with a 15-second fuse. Cost: 4 tnt_stick, 1 alarm_clock, 2 duct_tape.
- **`road_spike_strip`**: Deployed on highways to shred raider or convoy tires. Cost: 6 steel_spikes, 2 rubber_mat.
- **`clan_territory_totem`**: Projects the 96m territory boundary and active defense shield. Cost: 20 heavy_girder, 8 energetic_quartz, 1 clan_flag.

## 105. Automation Tech Tree & Science Flasks
Unlocking advanced factory machines and defenses requires automated research production loops:
- **`wood_craft_flask` (Tier 1 Science)**: Crafted from 2 wood, 1 stone. Researched in the `chalkboard_lab` to unlock Burner Drills, Wooden Belts, and Campfire Stoves.
- **`steam_gear_flask` (Tier 2 Science)**: Crafted from 1 iron_gear, 1 copper_wire. Unlocks Steam Generators, Rubber Conveyors, Splitters, and Bolter Turrets.
- **`chemical_power_flask` (Tier 3 Science)**: Crafted from 1 plastic_pellet, 1 refined_fuel, 1 sulfur. Unlocks Sludge Refineries, Narrow-Gauge Trains, Flame Turrets, and Pneumatic Hypertubes.
- **`crystal_quantum_flask` (Tier 4 Science)**: Crafted from 1 energetic_quartz, 1 circuit_board, 1 aluminum_sheet. Unlocks Mag-Belts, Tesla Accumulators, Repair Drone Bays, and Clan Shield Generators.

---

# Part 16: Pastoral Agriculture, Livestock Husbandry & Ambient Wildlife Ecology

## 106. The 7 Living Animal Archetypes & Visual Rig Specifications
The world features a rich, living pastoral ecosystem that coexists alongside the factory automation and undead hazards. All animals are rendered with dedicated procedural Ink Hollow canvas routines featuring organic squash-and-stretch, dynamic ground drop shadows, trot bobbing, ear twitches, tail wags, and situational expressive eye states:

1. **`cow` (Farm Livestock)**:
   - *Visual & Rig*: Sturdy bovine torso with piebald ink-spotted hide, warm tan muzzle with pink nostrils, dual tapered horn nubs, dangling brass cowbell suspended on a collar ribbon, soft swishing brush tail, and an active rhythmic cud-chewing bob animation.
   - *Base Stats*: Max HP 60, speed 1.1 m/s (sprint 2.0 m/s), sound `"MOO!"` / `"LOWW-MOO!"`.
   - *Produce*: Produces `fresh_milk` on a 120-second cycle (1 game day). Interact with `[E]` when udder is full.
   - *Drops*: 2–3 `raw_beef`, 1–2 `leather`, 1 `bone`.
   - *Favorite Food*: `fiber`, `animal_feed`.

2. **`sheep` (Farm Livestock)**:
   - *Visual & Rig*: Voluminous cloud-like scalloped wool silhouette composed of concentric ink circles, dark chocolate face with horizontal slit pupils, delicate dark hooves, and a short bobbing tail. Visually transitions to a trimmed, slender bare silhouette when sheared until wool regrows.
   - *Base Stats*: Max HP 40, speed 1.3 m/s (sprint 2.4 m/s), sound `"BAA-AA!"` / `"MEHH!"`.
   - *Produce*: Produces `wool_bundle` on a 150-second cycle. Interact with `[E]` with shears or empty hands to shear.
   - *Drops*: 2 `raw_mutton`, 2 `wool_bundle`.
   - *Favorite Food*: `fiber`, `wild_carrot`, `animal_feed`.

3. **`pig` (Farm Livestock)**:
   - *Visual & Rig*: Round plump blush-pink body, prominent frontal snout disc with dual round nostrils, perky floppy triangular ears that jiggle on movement, and a tight corkscrew spring tail that wiggles when happy. Features a downward head-rooting foraging animation.
   - *Base Stats*: Max HP 45, speed 1.2 m/s (sprint 2.2 m/s), sound `"OINK!"` / `"SQUEAL!"`.
   - *Behavior*: Natural truffle and root forager. Roots in soil, occasionally unearthing wild tubers, acorns, or buried trinkets.
   - *Drops*: 2–3 `raw_pork`, 1 `leather`.
   - *Favorite Food*: `wild_carrot`, `berries`, `animal_feed`.

4. **`chicken` (Farm Livestock)**:
   - *Visual & Rig*: Plump feathered body, bright crimson serrated comb and wattle, sharp yellow triangular beak, rapid pecking cadence with alternating head thrusts, and a fan-feather tail. Periodically squats into an egg-laying posture with a gentle cluck.
   - *Base Stats*: Max HP 20, speed 1.4 m/s (sprint 2.6 m/s), sound `"CLUCK!"` / `"BOK-BOK!"`.
   - *Produce*: Produces `fresh_egg` on a 90-second cycle. Interact with `[E]` to gather fresh farm eggs.
   - *Drops*: 1–2 `raw_poultry`, 2–4 `feather`.
   - *Favorite Food*: `seeds`, `berries`, `animal_feed`.

5. **`deer` (Ambient Wildlife)**:
   - *Visual & Rig*: Graceful, slender quadruped silhouette in warm fawn brown, tall alert perked ears, branching chestnut-brown antlers (stags), white chest bib, and a distinct white rump flash that flares when alerted or sprinting.
   - *Base Stats*: Max HP 50, speed 2.2 m/s (sprint 4.5 m/s), sound `"SNORT!"` / `"WHISTLE!"`.
   - *Behavior*: High sensory vigilance. Browses meadow saplings and thicket bushes. Extremely alert to approaching predators, player footsteps, and loud machinery.
   - *Drops*: 2–3 `raw_venison`, 1–2 `leather`, 1 `deer_antlers`.
   - *Favorite Food*: `wild_carrot`, `berries`.

6. **`rabbit` (Ambient Wildlife)**:
   - *Visual & Rig*: Petite oval body in soft sandy tan, oversized upright ears with delicate pink inner shading, twitching nose whisker dots, and a fluffy white puff tail. Moves via an extreme cartoon squash-and-stretch bounding leap animation with dust puff particles.
   - *Base Stats*: Max HP 15, speed 2.5 m/s (sprint 5.2 m/s), sound `"SQUEAK!"` / `"THUMP!"`.
   - *Behavior*: Flees immediately at high velocity into tall grass, brush, or burrows when approached.
   - *Drops*: 1 `rabbit_meat`, 1 `lucky_rabbit_foot` (rare talisman granting +10% critical hits and bonus loot).
   - *Favorite Food*: `wild_carrot`, `berries`.

7. **`goose` (Ambient Wildlife)**:
   - *Visual & Rig*: Elegant S-curved arching neck, bright orange paddle bill and webbed feet, immaculate white plumage, and wide flappable wing arches. Emits a loud, echoing territorial honk and will aggressively pursue players wearing cheese hats or holding shiny trinkets.
   - *Base Stats*: Max HP 30, speed 1.8 m/s (sprint 3.6 m/s), sound `"HONK!"` / `"HISS!"`.
   - *Behavior*: Amphibious habitat near rivers, lakes, and Splash Cove. Glides gracefully across water bodies and struts insolently across highways.
   - *Drops*: 1 `raw_poultry`, 3–5 `feather`.
   - *Favorite Food*: `seeds`, `berries`, `animal_feed`.

---

## 107. Pastoral Husbandry, Feeding, Taming & Sanity Healing
Interacting with farm animals provides vital psychological grounding and nutritional sustainability for the player and colony:

1. **Direct Pastoral Harvesting (`[E]`)**:
   - Targeting a mature animal displays contextual action prompts matching its produce readiness:
     - `🥛 Milk <Name> ([E])`: Harvests **Fresh Milk** (`fresh_milk`), restoring +25 Hunger, +15 Sanity, and curing mild toxin debuffs.
     - `✂ Shear <Name> ([E])`: Harvests **Wool Bundles** (`wool_bundle`), used for warm outerwear, luxury cots, and acoustic sound dampeners.
     - `🥚 Collect Eggs ([E])`: Gathers **Fresh Eggs** (`fresh_egg`), the cornerstone ingredient for fluffy omelets and high-tier baking.

2. **Feeding, Taming & Sanity Healing**:
   - Holding an animal's favorite food or crafted `animal_feed` in the active hotbar slot transforms the interaction prompt into `♥ Feed <Name> (<Item>)`.
   - Consuming the food item triggers:
     - Permanently flags the animal as `tamed: true`, rendering a floating green heart and permanent nametag above its head.
     - Heals the animal for **+15 HP**.
     - Restores **+12 Player Sanity** as warm fuzzies calm the player's nerves.
   - Tamed animals stay near the player's basecamp fences and will follow players holding their preferred snack.

3. **Petting Interaction**:
   - Approaching any animal with empty hands prompts `Pet <Name> ([E])`.
   - Petting plays an affectionate petting hand animation, accompanied by contented animal sounds and a **+8 Sanity recovery** burst.

4. **Crafted Nutrition: `animal_feed`**:
   - Recipe: 2 `fiber` + 1 `berries` (or `seeds`) at any Workbench or Campfire.
   - Universally accepted by all 7 species, allowing efficient mass herd feeding and rapid taming.

---

## 108. Ambient Wildlife Ecology, Predator-Prey AI & Zombie Predation
The ecosystem is dynamically intertwined with the game's survival and zombie threats, creating emergent world events:

1. **Vigilance & Flight AI**:
   - Wildlife entities (deer, rabbits, geese) maintain continuous sensory scans:
     - Hearing radius: $14\text{ m}$ for gunshots, chainsaw revs, and car engines.
     - Vision cone: $10\text{ m}$ for fast-moving bipedal entities.
   - When an encroaching threat enters this perimeter, the animal enters an `alert` state (ears perk, white rump flash flares), followed by a high-speed sprint directly away from the danger vector.

2. **Emergent Zombie Predation**:
   - Undead hordes actively hunt living wildlife and farm livestock if the player is not in immediate range.
   - Zombies will prioritize charging grazing cows, sheep, and chickens, cornering them against fences and base walls.
   - This mechanics turns livestock pens into vital tactical defense points: players must fortify barnyards with zap fences and turrets or lose their valuable herds to nocturnal raids.

3. **Vehicular Collision Dynamics**:
   - Driving vehicles at speed into roaming wildlife or livestock triggers kinetic impact physics.
   - Vehicles deal $v_{\text{rel}} \times 15$ damage on impact, knocking animals sideways with cartoon screech sounds and dropping harvested meats upon roadkill. However, heavy collisions cause vehicular durability degradation, incentivizing players to slow down or install reinforced cattle guards (`spiked_cowcatcher`).

---

## 109. Farm-to-Table Gastronomy & Material Butchery
Hunting animals or slaughtering livestock yields raw cuts of meat and specialized biological crafting materials, expanding the game's culinary tech tree:

1. **Raw Harvest Matrix**:
   - `raw_beef`, `raw_mutton`, `raw_pork`, `raw_poultry`, `raw_venison`, `rabbit_meat`: Fresh uncooked cuts. Consuming raw risks mild food poisoning (-5 Sanity, nausea).
   - `leather`: Essential tough hide for armor vests, vehicle seat belts, industrial machinery gaskets, and bellows.
   - `wool_bundle`: Spun into yarn for thermal jackets, sleeping bags, and luxury sofa cushions.
   - `feather`: Used for silent hunting arrows, propeller beanie vanes, and feather-soft pillows.
   - `deer_antlers`: Harvested from stags; crafted into decorative wall mounts, bone pickaxes, or tribal shaman crowns.
   - `lucky_rabbit_foot`: Rare drop from wild hares; carried in fanny packs or toolbelts to grant +10% critical strike chance and boosted container jackpot odds.

2. **Campfire & Kitchen Culinary Recipes**:
   - **`cooked_steak`**: Seared thick-cut beef cutlet over hickory coals. Restores **+45 Hunger**, **+15 Health**.
   - **`cooked_bacon`**: Sizzling crispy cured pork strips. Restores **+35 Hunger**, **+10 Sanity**.
   - **`roast_chicken`**: Whole slow-roasted golden fowl with herb stuffing. Restores **+50 Hunger**, **+20 Sanity**, and grants a 60-second endurance sprint buff.
   - **`cooked_venison`**: Savory lean venison medallions. Restores **+40 Hunger**, **+10 Health**.
   - **`fluffy_omelet`**: Pan-folded triple-egg omelet topped with fresh herbs. Restores **+30 Hunger**, **+20 Sanity**, and sharpens visual awareness.

---

# Part 17: Comprehensive Project State, Engineering Architecture & Vision Realization

## 110. Implementation Status Matrix (Where We Are)
The project has evolved from a foundational top-down survival prototype into an expansive, hybrid simulation fusing the arcade open-world joy of *The Simpsons: Hit & Run*, the industrial logistics of *Factorio*, the psychological storytelling of *RimWorld*, the gritty undead survival of *DayZ*, and the charming pastoral husbandry of *Animal Crossing*.

| Phase | Milestone Name | Subsystem & Implemented Capabilities | Verification & Health |
| :--- | :--- | :--- | :--- |
| **Z0–Z0C** | **Core Engine & Architecture** | 60Hz fixed-timestep deterministic tick loop, ECS decoupled entity queries, universal input mapping with virtual touch and haptic support, procedural canvas bone animation rigs with stride matching. | 100% verified (`tests/sim.test.ts`) |
| **Z1–Z2** | **Chunked World & Biomes** | Deterministic seed-based procedural world generation, multi-stage noise splat map, 10 unique biomes with distinct hazard palettes, POI generation (Abandoned Towns, Gas Stations, Radio Towers, Outposts), infinite chunk paging. | 100% verified (`tests/world.test.ts`, `tests/poi-town.test.ts`) |
| **Z8–Z9** | **Base Building & Construction** | Base territory claim system, collision grids, placement rules, craftable structures (walls, gates, storage chests, campfires, illumination lamps), refund teardown economics. | 100% verified (`tests/sim.test.ts`) |
| **Z10** | **Vehicle Fleet & Driving Physics** | 7 fully simulated vehicles (`rusty_hatchback`, `clown_car`, `bathtub_buggy`, `golf_cart`, `ice_cream_truck`, `hot_dog_van`, `tesla_roadster`). Dynamic steering knuckles, handbrake drifts, fuel cycles, trunk inventories, and high-speed zombie ramming. | 100% verified (`tests/vehicle.test.ts`, `tests/fleet.test.ts`) |
| **Z11** | **Companion Pets** | Autonomous companion AI (`hunting_dog`, `wolf_pup`). Tethered follow logic, target barking, defensive zombie bite attacks, petting interaction with sanity recovery, full save/load roundtripping. | 100% verified (`tests/pet.test.ts`) |
| **Z12–Z13**| **Undead Survival & Environment** | 3 core zombie archetypes (`walker`, `runner`, `brute`), acoustic hearing propagation, day/night light sensitivity, dynamic darkness sanity damage, configurable lighting engine, always-day mode toggle. | 100% verified (`tests/zombie.test.ts`) |
| **Z14** | **Industrial Automation** | Conveyor belts with directional item physics and junction transfers, spring-loaded inserter arms with chest routing, mineral ore nodes, burner drills, automated smelting blast furnaces. | 100% verified (`tests/automation.test.ts`) |
| **Z15** | **Colony Simulation & Psychology** | Autonomous colonists with generative skills, passions, traits, 1–4 Priority Work Matrix (`Key: O`), 0–100% mood bars with environmental thoughts, slapstick mental breaks (conveyor carousel, noodle tantrum, dirt gourmand), and 3 AI Storytellers (Caspar Chill, Randy Ruckus, Hilda Hardcore). | 100% verified (`tests/colony.test.ts`) |
| **Z16** | **Evolved Sieges & Pollution** | Chunk-based machine pollution accumulation and natural forest dissipation, mutational siege hordes, specialized siege mutants (sapper bomb, spitter acid, burrower breach, screamer rally, goliath demolisher), automated turrets (belt-fed bolter, flame turret, zap fence, catapult), and belt-fed auto-reloading. | 100% verified (`tests/siege.test.ts`) |
| **Z17** | **PvPvE Factions & Base Raiding** | Tri-zone risk topology (Green Haven, Yellow Contested, Red Anarchy), Clan Territory Totem projecting automated Tesla shields, 2-hour active defense raid windows, 48-hour peace treaties, C4 plunger breaching demolition charges, radio black market bounties, and Faction dashboard (`Key: P`). | 100% verified (`tests/faction.test.ts`) |
| **Z18** | **Master Weapons & Goofy Food** | Slapstick weaponry (Pool Noodle of Justice knockback, Chainsaw of Regret massive noise/DPS, Traffic Sign Axe, Frying Pan stun bong, Slingshot with pebbles, Potato Cannon), wearable gear (Propeller Beanie, Boing Boots, Pillow Vest, Fanny Pack), throwables/traps (Cream Pie, Water Balloon, Whoopee Cushion, Road Flare), and side-effect foods (Hiccup Berries, Chili of Regret fire breath, Rainbow Soup glow, Crispy Sock Chips turbo, Bubble Tea buoyancy). | 100% verified (`tests/weapons-food.test.ts`) |
| **Z19** | **Livestock & Ambient Wildlife** | 7 animated animal archetypes (`cow`, `sheep`, `pig`, `chicken`, `deer`, `rabbit`, `goose`) rendered via procedural Ink Hollow canvas. Interactive milking, shearing, egg collecting, favorite food feeding & taming, sanity petting, alert fleeing AI, zombie predation, vehicle ramming, hunting drops, and 6 culinary recipes. | 100% verified (`tests/wildlife.test.ts`) |
| **Z19B** | **Continental Cartography & Master Catalog** | 680×440 Continental World Map with Tri-Zone risk rings, live entity tracking for player, vehicles, companion Barkley, farm livestock herds, and clan totems; dynamic zoom range expanded to 8–170 for ~240m horizon; toggleable Fog of War hotkey `[F]` and UI button with default un-fogged mode; comprehensive 82-item master catalog across weapons, ballistics, throwables, gear, foods, automation, vehicles, and base structures. | 100% verified (`tests/wildlife.test.ts`, live runtime) |
| **Z19C** | **Multi-Tier Road Network & Big Town Metropolis** | 3-tier seeded road network: Highways with deep asphalt `#232427`, golden double centerlines `#facc15` and $+35\%$ speed multiplier, rural Dirt Roads with packed earth and wheel ruts with $+18\%$ speed bonus, and small cobblestone town streets with mortar seams and $+10\%$ speed bonus; sprawling Big Towns including spawn metropolis "Metropolis Oakhaven" and 5 named regional hubs across the continent; full municipal and road infrastructure: `town_fountain` (+25 sanity drinking/resting), `street_light` (5.8m nocturnal illumination), `guardrail`, `picket_fence`, `park_bench` (+15 sanity resting), `fire_hydrant`, `dumpster` (scavenge comedy loot), and `traffic_cone`; minimap & continental world map colored road ribbons and civic crest `🏛️` icons. | 100% verified (`tests/roads-towns.test.ts`) |
| **Z19D** | **Terrain, Biomes & Streets Complete Revamp** | Vibrant & stylized indie palette overhaul: saturated emerald meadows, crystal turquoise-azure lagoons, warm terracotta desert dunes, deep alpine pines; multi-block urban street grids for all 6 metropolises featuring concrete sidewalks with expansion joints, raised curbs, bold white zebra pedestrian crosswalks, parking stalls, and sewer manholes; organic harmonic continental highways with double golden stripes and white fog lines; winding dirt byways with dual tire ruts and wild center trails; colorful meadow wildflowers and foamy coastal shorelines. | 100% verified (`tests/roads-towns.test.ts`, live runtime) |

### Core Engineering Quality Metrics
- **Automated Test Suite**: **85/85 unit and regression tests passing** across all modules with zero flaky tests.
- **Static Analysis**: Clean TypeScript compiler validation (`tsc --noEmit`) with strict typing enabled.
- **Production Distribution**: Bundled into a single, fully self-contained offline distribution file `dist/single.html` measuring **294 KB** (including complete runtime engine, registries, procedural art generators, and UI).
- **Runtime Performance**: Consistent 60 FPS execution on standard browser canvas with zero garbage collection spikes.

---

## 111. Architectural Tenets: How It Should Be
To maintain high velocity, stability, and aesthetic purity as the game scales toward full launch, all future engineering must strictly honor these five immutable architectural principles:

1. **Pure Data-Driven Declarations**:
   - Engine systems must never hardcode item names, craft recipes, vehicle speeds, or animal behaviors.
   - Everything must be declared in serializable data tables under `src/data/` (`items.ts`, `recipes.ts`, `vehicles.ts`, `animals.ts`, `zombies.ts`, `structures.ts`). Adding new content should be a simple data declaration without touching engine simulation loops.

2. **The Procedural Ink Hollow Aesthetic**:
   - The game relies on a hand-drawn, papercraft storybook visual signature (ink outlines, warm gouache color washes, hatched shadows, dynamic springs, squash-and-stretch).
   - **Zero External Bitmaps**: Everything is rendered procedurally onto the 2D canvas using vector paths, arcs, and gradients. This guarantees instant loading, zero missing asset errors, infinite resolution scalability, and an ultra-lean bundle size (<300 KB).

3. **Strict Timestep Decoupling via ECS**:
   - The simulation loop runs on a deterministic fixed 60Hz tick (`dt = 1/60`).
   - Gameplay rules read and write to discrete ECS components (`Transform`, `Body`, `Inventory`, `Needs`, `Vehicle`, `Animal`, `Zombie`, `Structure`, `Colony`, `Siege`).
   - The rendering layer is purely a reactive view: it reads ECS state and interpolates positions without mutating game logic.

4. **Emergent Systems Synergy**:
   - Game mechanics must not exist in isolated silos; they must interconnect to form dynamic player stories:
     - Factory machines produce pollution $\to$ Pollution enrages mutated zombie hives $\to$ Hordes launch base sieges $\to$ Automated turrets defend the perimeter $\to$ Downed mutants drop scrap and fuel $\to$ Fuel powers exploration vehicles $\to$ Vehicles explore distant POIs to rescue colonists $\to$ Colonists farm livestock and craft high-tech science flasks $\to$ Happy colonists maintain the production lines.

5. **Robust Backwards-Compatible Save Serialization**:
   - The save format (`toSave()` / `fromSave()`) must serialize the entire continental world state, player hotbars, discovered map sectors, colony work matrices, vehicle locations, and animal herds.
   - Every new subsystem must implement graceful schema migration fallbacks so existing player saves load seamlessly.

---

## 112. The Horizon Roadmap: Immediate Next Evolutionary Milestones
With single-player survival, vehicles, automation, colony psychology, territorial PvPvE, and pastoral wildlife fully operational, the project's next development horizons are:

1. **Phase Z20: Authoritative Multiplayer Networking & State Synchronization**:
   - Headless Node.js simulation server executing the identical 60Hz ECS loop.
   - Client-side prediction with server reconciliation for high-speed vehicle driving and combat.
   - Lightweight binary snapshot delta compression over WebSockets / WebRTC.
   - Room lobby matchmaking, team clan chat channels, and synchronized active raid window timers.

2. **Phase Z21: Dynamic Web Audio Synthesizer & Procedural Chiptune Soundscapes**:
   - Pure Web Audio API synthesis engine (zero external WAV/MP3 files, keeping the single-file distribution lean).
   - Real-time procedural audio effects: pitch-randomized rubber squeaks, chainsaw engine rumbles, metallic anvil clangs, cowbell chimes, animal vocalizations, and comedic cartoon pops (`"BONK!"`, `"POOF!"`).
   - Adaptive dynamic soundtrack that seamlessly shifts instruments and tempo based on time-of-day, biome, threat level, and territory risk zones (calm acoustic guitar in Green Havens $\to$ ominous industrial synth in Red Zones).

3. **Phase Z22: Subterranean Exploration, Dungeons & Mutant Hives**:
   - Procedural underground levels accessible via mine shafts, manholes, and cave entrances.
   - Pitch-black navigation requiring torches, road flares, or vehicle high-beams.
   - Concrete fallout shelters, abandoned military silos, and subterranean mutant queen nests containing Jackpot tier blueprint rewards and high-density quartz veins.

4. **Phase Z23: The Grand Escape Endgame & Boss Encounters**:
   - The ultimate objective: constructing the **Amphibious Escape Barge** at the Splash Cove Evac Dock.
   - Requires massing industrial resources, refined aviation fuel, and food provisions from all biomes.
   - Culminates in a legendary multi-phase endgame horde defense event featuring colossal boss monstrosities: **Highway Hulk**, **The Screamer Queen**, and **Quackzilla the Mountain Duck**.

5. **Phase Z26: Status Effects, Reactions & Stagger** (Part 18, §115–§118):
   - Twelve effects as data rows, a data-driven reaction table with a chain depth limit, poise and the stagger ladder.
   - Data layer is in code (`src/data/effects.ts`). Build plan for the rest: `world4bros-core/docs/Z26_PLAN.md`.

6. **Phase Z27: Biome Atlas Expansion** (Part 18, §119–§120):
   - Effect and hazard data for the 10 existing biomes, then the 6 proposed biomes as new definitions.

7. **Phase Z28: Aim Arena & Test Bench** (`world4bros-core/docs/ARENA.md`):
   - Arenas as data, offline bot duel, headless Test Bench, then online 1v1 and free-for-all.

8. **Phase Z29: Weapon Builder, Gunsmith & Info HUD** (`world4bros-core/docs/WEAPONS.md`):
   - Weapon parts as data with derived stats and quirks, the Gunsmith panel, the info HUD, then loot and balance.

9. **Phase Z30: Town Houses, Yards & HD Graphics Pass** (§26, §69):
   - Houses done; roof shapes, wall details, yard props, looser town layout and the HD passes remain.

---

## 113. Road Hierarchy & Sprawling Big Town Infrastructure (Phase Z19C)

### 1. Multi-Tier Road Network Specifications
The continental road network is generated deterministically across all chunks using a three-tier hierarchy:
1. **Highways (`highway`)**:
   - **Width**: $4.6\text{m}$ ribbon corridor.
   - **Visuals**: Deep obsidian asphalt (`#232427`) with vibrant golden double-dashed dividing centerlines (`#facc15`).
   - **Gameplay Multiplier**: $+35\%$ foot and vehicle velocity bonus ($1.35\times$).
   - **Layout**: Arterial grid aligned every $96\text{m}$ across the continent, intersecting major metropolitan hubs and linking all continental POIs.
2. **Dirt Roads (`dirt_road`)**:
   - **Width**: $3.6\text{m}$ ribbon corridor.
   - **Visuals**: Warm packed earth (`#544436`) with textured parallel dark wheel ruts (`#3a2e24`) and scattered gravel pebbles.
   - **Gameplay Multiplier**: $+18\%$ velocity bonus ($1.18\times$).
   - **Layout**: Secondary rural grid offset at $48\text{m}$ intervals connecting farms, radio outposts, and backcountry settlements.
3. **Small Roads & Alleys (`small_road`)**:
   - **Width**: $2.8\text{m}$ ribbon corridor.
   - **Visuals**: Weathered cobblestone flagstone paver blocks (`#4c4f57`) with recessed mortar seams (`#33353b`).
   - **Gameplay Multiplier**: $+10\%$ velocity bonus ($1.10\times$).
   - **Layout**: Town cross-alleys, residential avenues, and wilderness singletrack paths meandering between nature reserves.

### 2. Sprawling Big Town Metropolises
Towns are no longer single-chunk outposts; they span multi-chunk urban districts:
- **Metropolis Oakhaven**: The grand starting metropolis centered at world spawn (`0, 0`), featuring a grand civic promenade, central decorative fountain, park benches, and perimeter streetlights.
- **Regional Continental Hubs**:
  - `Rust Creek Junction`: Industrial rail and scrap junction at $(-72, 80)$.
  - `Port Seagull Harbor`: Coastal maritime commercial district at $(80, -96)$.
  - `Pine Ridge Outpost`: Northern logging community at $(96, 64)$.
  - `Deadwood Oasis Hub`: Arid desert trading bazaar at $(-88, -80)$.
  - `Copperfield Metropolis`: Eastern mining and smelting capital at $(120, -32)$.
- **Map & Minimap Visuals**: Marked with golden civic crests (`🏛️`) and bold gold typography, with color-coded road ribbons (Highways: `#fbbf24`, Dirt roads: `#b45309`, Cobblestones: `#94a3b8`).

### 3. Urban & Road Infrastructure Objects
| Object ID | Category | Solid | Interaction | Gameplay & Visual Effect |
| :--- | :--- | :--- | :--- | :--- |
| `town_fountain` | Civic / Comfort | Yes | Drink / Rest | Restores $+25$ Sanity on direct interaction; passive $+2.5$ Sanity/sec within $4.2\text{m}$ radius. Features tiered marble basins, bubbling water spout, and splash rings. |
| `street_light` | Illumination | Yes | Passive | Emits a warm $5.8\text{m}$ radial light pool during dusk and night, protecting players from darkness sanity drain. Cast-iron lantern post with ornamental bracket. |
| `park_bench` | Comfort | Yes | Sit / Rest | Restores $+15$ Sanity on direct interaction; passive $+2.0$ Sanity/sec proximity relaxation. Slat-wood bench with iron armrests. |
| `dumpster` | Scavenge | Yes | Scavenge Loot | Interactive green steel alley dumpster. Scavenging yields comedic loot (`rubber_chicken`, `stale_baguette`, `canned_beans`, `fanny_pack_of_holding`, `propeller_beanie`). |
| `guardrail` | Barrier | Yes | Collision / Chop | Galvanized corrugated steel barrier along highway curves and bridges. Prevents errant vehicle spinouts. |
| `picket_fence` | Barrier | Yes | Collision / Chop | Classic white-painted timber fence defining town property borders. |
| `fire_hydrant` | Municipal | Yes | Collision | Vibrant red cast-iron municipal water plug with brass outlet caps. |
| `traffic_cone` | Safety / Trap | No | Non-solid | Fluorescent orange road safety cone with reflective white collar. Pushable and placeable. |

---

## 114. Terrain, Biomes & Street Network Complete Overhaul (Phase Z19D)

### 1. Vibrant & Stylized Indie Art Direction
To transcend dull, washed-out prototypes, the entire palette has been upgraded to a rich, saturated, high-contrast aesthetic reminiscent of premium indie games (*Zelda: Link's Awakening*, *Cult of the Lamb*, *Don't Starve*):
- **Bro Meadow (Grass)**: Base emerald `#3b823e`, shadow deep forest `#285e2b`, highlight sunny clover `#5ec763`, populated by multi-colored wildflowers (golden dandelions `#facc15`, scarlet poppies `#f43f5e`, white daisies `#ffffff`, blue forget-me-nots `#38bdf8`, pink clovers `#f472b6`).
- **Pinewood Thicket (Needles / Moss)**: Base deep pine needle mulch `#624a35`, shadow `#453323`, highlight amber resin `#876749`, with cross-hatched fallen needles and twigs.
- **Squelchy Bog (Mud / Silt)**: Base dark peat muck `#443528`, shadow `#2c2219`, highlight slick silt `#5f4b39`.
- **Rockbottom Ridge (Rock / Granite)**: Base slate blue-grey `#5a6577`, shadow quarry `#3c4554`, highlight quartz shimmer `#7c8ba3`.
- **Sunburn Flats (Sand / Dunes)**: Base golden sun sand `#d99b45`, shadow terracotta `#ad7329`, highlight dune crest `#f5be6c`.
- **Brrington Tundra (Snow / Permafrost)**: Base crisp glacial white-blue `#e2effa`, shadow soft blue `#9ec0db`, highlight sparkling frost `#ffffff`.
- **Splash Cove & Oceans**:
  - Shallow Waters: Vibrant tropical cyan-azure lagoon `#06b6d4`, shadow `#0891b2`, highlight sunlit translucent aqua `#67e8f9`, with dynamic white wave foam contours.
  - Deep Waters: Oceanic deep navy `#0f3b4c`, abyssal shadow `#08232e`, wave swell highlight `#164e63`.
- **Mount Grumbleguts (Ash / Basalt)**: Base dark volcanic basalt `#34333b`, shadow pitch obsidian `#1e1e23`, highlight smoky pumice `#545260`.

### 2. Multi-Block Urban Street Grids & Highway Architecture
1. **Town Street Networks**:
   - **Grand Avenue & Main Street**: $4.6\text{m}$ arterial corridors crossing through the town center, classified as highways for maximum vehicular velocity.
   - **Paved Sidewalks**: Outer $0.9\text{m}$ of urban avenues feature segmented concrete sidewalk paving slabs (`#8f9aa8`) with expansion joint lines (`#555f6d`) and raised granite curbstones (`#3c4450`) with subtle drop shadows.
   - **Pedestrian Zebra Crosswalks**: Intersections are adorned with bold white reflective zebra crossing bars (`#f8fafc`) for safe foot transit across avenues.
   - **Municipal Street Utilities**: Cast-iron sewer manholes, storm drainage grates, and white parking stall demarcation lines.
   - **Town Civic Plaza**: Circular ornamental flagstone paver tiles (`#5e6573`) surrounding the central Town Fountain.
2. **Continental Harmonic Highways**:
   - Sweeping sinusoidal harmonic curves that contour naturally with topography, avoiding rigid artificial grids.
   - Deep obsidian asphalt (`#1b1c20`) with aggregate gravel flecks, white boundary fog lines (`rgba(240, 248, 255, 0.7)`), and vibrant golden double centerlines (`#facc15`).
3. **Rural Dirt Byways**:
   - Meandering unpaved country tracks with dual parallel dark compacted wheel ruts (`#453324`), raised grassy center ridges (`#4a7c3e`), and loose gravel pebbles.
4. **Cobblestone Alleys & Nature Trails**:
   - Interlocking stone flagstones with beveled 3D highlights and shadows, weaving through historic districts and wilderness lakeshores.

---

# Part 18: Status Effects, Reactions & Biome Atlas

Design approved 2026-10-19. The visual reference pages are `docs/design/index.html` (effects) and `docs/design/biomes.html` (biome atlas). Open them in a browser. Visuals are the target look; all numbers are starting values. Build order: Z26 (effects) then Z27 (biomes), see §69.

## 115. The twelve effects
Each effect is one data row (`EffectDef`). Every effect also has an icon, a hand-lettered sound word and a plain-words tooltip, so colour is never the only signal.

| ID | Colour | From | Does | Ends |
| :--- | :--- | :--- | :--- | :--- |
| `acid` | `#b6e64a` | Spitter zombies, acid pools, toxic rain, acid sprayers. | Damage over time. Wears down metal gear. Leaves a lingering puddle. | Water, antidote gel, or about 6 s. |
| `burn` | `#F08A32` | Fire, lava, flame weapons, explosions. | Damage over time. Spreads to anything flammable. Lights the area. | Water, or about 5 s. |
| `explosion` | `#F9C45A` | Barrels, sappers, rockets, gas leaks. | Area damage that fades with distance, knockback, a crater, big noise. | Instant. |
| `stagger` | `#EAE0C8` | Heavy hits, blasts, knockback guns. | A flinch, a stagger or a knockdown, depending on poise (§117). | 0.2 to 1.5 s. |
| `electricity` | `#FDEB90` | Fences, tasers, storms, tesla coils. | Shock damage and a short stun. Chains through wet and metal. Stalls vehicles. | Instant jolt, then the stun. |
| `cold` | `#8fd8e8` | Tundra, ice weapons, frozen water. | Slows, then freezes in place. Frozen things shatter when blasted. | Warmth or fire. |
| `spores` | `#c084fc` | The Hollow, bogs, fungus. | Drains sanity and makes vision swim. | Leaving the cloud, or light. |
| `goo` | `#f472b6` | Slime zombies, sticky traps. | Slows and sticks. Easy to hit. | Water, or struggling free. |
| `wet` | `#67e8f9` | Rain, water, squirt guns. | Puts out fire, makes electricity chain, chills faster. | Dries over time. |
| `oil_slick` | `#9aa1ac` | Spills, wrecks. | Slippery: you slide. Very flammable. | Burns off or is wiped. |
| `blind` | `#f3e7b4` | Cream pies, smoke, salt glare. | Blocks sight and ruins aim. | About 2 s. |
| `smoke` | `#9aa1ac` | Fires, smoke pots. | Hides you from sight. A little coughing damage. | Drifts away. |

## 116. Reaction table (what happens when they meet)
Rows are what hits; columns are what it hits. ▲ marks a reaction that makes things bigger or chains.

| Hits ↓ / Target → | Grass or wood | Water or wet | Oil | Metal | Ice | Gas or barrel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fire** | ▲ Spreads | Put out, steam | ▲ Bigger blaze | Heats up, nothing else | Melts to water | ▲ Explodes |
| **Electricity** | Nothing | ▲ Chains to every wet target | Nothing | Conducts: longer stun, vehicles stall | Nothing | Spark sets it off |
| **Acid** | Weakens it | Dilutes the puddle | Floats on top | Corrodes gear | Eats a pit | Nothing |
| **Explosion** | Scorches the ground | Geyser and knockback | ▲ Fireball | Dents and breaks it | Shatters | ▲ Chain explosion |
| **Cold** | Frosts it | Freezes the surface, slippery | Stiffens, no slip | Frosts it | Nothing | Nothing |

Every thing in the world carries tags (`flammable`, `wet`, `conductive`, `explosive`, `oil`, `frozen`, `metal`) and this table is data, so a new element or a new surface needs no engine change. Chains have a depth limit so a barrel field cannot freeze the game.

## 117. The stagger ladder
Hits should feel like something. Heavier things move you more. Every hit adds impact. Each creature has a `poise` value, and the share of poise a hit uses decides the step. Size, armour and a riot shield add poise; the biggest zombies shrug off a flinch. Poise refills when not hit.

| Step | Share of poise | Result |
| :--- | :--- | :--- |
| `flinch` | About 20% | A 0.15 s slowdown. Aim is kept. |
| `stagger` | About 50% | Cannot move for 0.5 s and steps back. |
| `stumble` | About 80% | Spins and loses aim for about 1 s. |
| `knockdown` | 100% or a blast | Falls for 1.5 s and is easy to hit. |

## 118. Rules for all effects
- **Cartoon outcomes**: effects show as comic visuals (green drips, flames, a starburst). No gore. Burnt things turn to soot and ash.
- **Readable at a glance**: each effect has its own icon and shape, not just a colour. A small row of icons shows what is on you.
- **Kind to eyes**: flashes (electricity, explosions) follow a reduced-flashing setting alongside reduced motion. Nothing flashes fast by default.
- **Caps**: a limit on active fires, puddles and chains, so a big fight cannot overload the game.
- **Same for players and zombies**: the same rows work on bros, zombies, raiders, vehicles and structures. Immunities are data.
- **Server decides**: in multiplayer the server applies and ends effects, so nobody can ignore a burn.

## 119. Biome atlas
Sixteen biomes. The first ten are in this bible (§1); the last six are proposals. Palette swatches are the terrain base, second terrain, accent and highlight colours from the atlas page.

| Biome | Status | Hazard | Effects | Props | Feel | Zombies | Palette |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bro Meadow | In the bible | None by day. A bee hive if you poke it. | (none) | Pine tree, berry bush, welcome sign | Butterflies, pollen, birdsong | Walker | `#3b823e` `#5ec763` `#78593d` `#f5e6a8` |
| Pinewood Thicket | In the bible | Fog and getting lost. Glowing eyes at night. | Blind (fog) | Hollow logs, fireflies | Light shafts, falling needles | Walker, lurker at night | `#1f4a2a` `#2d6a4f` `#624a35` `#6F8FA0` |
| Squelchy Bog | In the bible | Mud slows you. Gas bubbles pop and can ignite. | Goo, Explosion, Spores | Bouncy mushrooms, reeds, cattails | Ripple rings, frog choir | Mud zombie | `#443528` `#2d6a4f` `#5f4b39` `#7a5aa8` |
| Rockbottom Ridge | In the bible | Rockfall after a loud honk. | Stagger, Explosion | Flint, gold veins, crystals | Echoes, dust, skittering pebbles | Walker, brute | `#5a6577` `#3c4554` `#7c8ba3` `#c9a227` |
| Sunburn Flats | In the bible | Heat meter and thirst. A mirage that vanishes. | Burn (mild), Blind (glare) | Cactus, tumbleweed, dry weeds | Dust devils, shimmer | Husk zombie | `#d99b45` `#ad7329` `#f5be6c` `#C8501F` |
| Brrington Tundra | In the bible | Cold meter. Slippery ice, thin ice. | Cold, Wet | Snowdrifts, a frozen lake, ice holes | Gusts, breath puffs, aurora | Frost zombie | `#e2effa` `#9ec0db` `#ffffff` `#6F8FA0` |
| Scrapyard Gulch | In the bible | Spring traps, conveyors, live wires. | Electricity, Oil, Stagger | Tyre walls, scrap piles, barrels | Sparks, flickering signs, clanks | Walker, sapper | `#5a6577` `#8a4a2a` `#a8623a` `#F9C45A` |
| The Hollow | In the bible | Sanity drain and whispers. | Spores | Dead trees, lanterns, ghost sheets | Chromatic wobble, cold fog | Ghost-sheet zombie | `#090E14` `#2b1d3a` `#4a2d5a` `#C9D84A` |
| Splash Cove | In the bible | Deep water drowns. Goose gangs. | Wet | Pier, crabs, a duck flotilla | Foam edges, gulls, waves | Goose, crab | `#06b6d4` `#d99b45` `#67e8f9` `#f3e7b4` |
| Mount Grumbleguts | In the bible | Lava, burning ash, eruptions. | Burn, Explosion | Obsidian, ember fruit, a boss arena | Embers, rumble, a pulsing glow | Ember zombie | `#34333b` `#1e1e23` `#C8501F` `#F08A32` |
| Toxic Marsh | New proposal | Acid pools and acid rain. | Acid, Wet | Melted trees, leaking barrels | Green mist, bubbling, sizzles | Melter (proposed) | `#2a3a2a` `#445533` `#9be34a` `#b6e64a` |
| Cinder Wastes | New proposal | Burning ground that spreads. | Burn, Smoke | Charred stumps, ash drifts | Falling ash, glowing cracks | Ember zombie | `#34333b` `#545260` `#C8501F` `#FDEB90` |
| Storm Plains | New proposal | Lightning strikes and live pylons. | Electricity, Wet | Pylons, fallen cables, tall grass | Thunder, rain, wind | Static (proposed) | `#4a5a6a` `#8fa5b8` `#F9C45A` `#FDEB90` |
| Salt Flats | New proposal | Blinding glare and mirages. | Blind | Salt crystals, dry boats | Heat shimmer, wind hiss | Husk zombie | `#e8e4d8` `#cfc8b4` `#9ec0db` `#f5be6c` |
| Crystal Caverns | New proposal | Dark. Loud noise shatters crystals. | Stagger, Explosion-lite | Glow crystals, echo rooms | Hum, drips, echoes | Glimmer (proposed) | `#1b2a3a` `#6F8FA0` `#8fd8e8` `#c084fc` |
| Ruined Downtown | New proposal | Gas leaks and live wires. | Explosion, Electricity, Oil | Wrecks, barriers, cracked streets | Sirens, flickering lights | Walker, sapper | `#3a3f46` `#1c1e22` `#7c8ba3` `#c9b64a` |

## 120. What a biome is, as data
One biome definition holds: name and mood, terrain mix and palette, edge profiles, ground decals, props and set dressing, hazards (tiles and objects), weather and ambient effects, light and sky colour, the zombie and animal spawn table, loot flavour, and music and sound words.

Terrain colours, the seam recipe, decals, props, hazards, weather and spawns are all data, and effects are data too. A new biome is a new definition plus a few drawn props, with no engine change. The same effect rows (acid, burn, electricity) are reused across biomes, so Toxic Marsh and Scrapyard share their rules and only differ in where and how often. In code this extends `BiomeDef` in `world4bros-core/src/data/schema.ts`; today `src/data/biomes.ts` holds the 10 bible biomes with climate, terrain and object scatter only.




