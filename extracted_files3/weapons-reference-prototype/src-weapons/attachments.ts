import type { AttachmentDef, InfoMode, Mount, Quirk, Rarity, SlotId, Stats } from './schema';

type Opts = Partial<Pick<AttachmentDef, 'add' | 'mul' | 'quirks' | 'info' | 'requires' | 'forbid' | 'tags' | 'homemade' | 'recipe'>>;
const A = (id: string, name: string, slot: SlotId, mount: Mount, rarity: Rarity, desc: string, o: Opts = {}): AttachmentDef => ({ id, name, slot, mount, rarity, desc, ...o });
const q = (...x: Quirk[]): Quirk[] => x;
const tape = { duct_tape: 1 };
const noToy = ['toy', 'tube'];

/**
 * The attachment catalog. Everything here is data: add a row, add a drawer-free item (items are generated from this list),
 * and it appears in the Gunsmith. Homemade rows carry a `recipe` so the crafting panel offers them too.
 */
export const ATTACHMENTS: AttachmentDef[] = [
  // ---- muzzle
  A('suppressor_sock', 'Sock Suppressor', 'muzzle', 'tape', 'J', 'A mismatched sock taped over the end. Quiet-ish. Smells.', { mul: { noise: 0.45, range: 0.94, damage: 0.95 }, add: { weight: 0.2 }, tags: ['quiet'], homemade: true, recipe: { mismatched_sock: 1, ...tape } }),
  A('soda_can_baffle', 'Soda Can Baffle', 'muzzle', 'tape', 'J', 'A can with holes. Muffles, mostly.', { mul: { noise: 0.62, spread: 1.1 }, add: { weight: 0.15 }, tags: ['quiet'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('proper_suppressor', 'Proper Suppressor', 'muzzle', 'thread', 'R', 'Machined, threaded and quiet.', { mul: { noise: 0.35, range: 0.97 }, add: { weight: 0.6 }, forbid: noToy, tags: ['quiet'] }),
  A('compensator_spring', 'Spring Compensator', 'muzzle', 'thread', 'U', 'A spring that tames the muzzle. A bit louder.', { mul: { spread: 0.85, noise: 1.1 }, add: { weight: 0.2 }, forbid: noToy }),
  A('traffic_cone_flash', 'Mini Traffic Cone', 'muzzle', 'thread', 'C', 'Flash hider. Looks ridiculous, works fine.', { mul: { noise: 0.9, spread: 0.95 }, forbid: noToy }),
  A('bell_muzzle', 'Muzzle Bell', 'muzzle', 'tape', 'J', 'DING on every shot. On purpose: it is a lure.', { quirks: q({ t: 'noise_pulse', extra: 10 }), homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('kazoo_muzzle', 'Kazoo Muzzle', 'muzzle', 'tape', 'J', 'Every shot goes toot.', { quirks: q({ t: 'noise_pulse', extra: 5 }, { t: 'wobble', rad: 0.012 }), homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('party_horn_muzzle', 'Party Horn', 'muzzle', 'tape', 'C', 'Confetti, a honk, and a startled zombie.', { quirks: q({ t: 'confetti' }, { t: 'noise_pulse', extra: 6 }, { t: 'stun', chance: 0.06, secs: 1 }) }),
  A('pinwheel_muzzle', 'Pinwheel', 'muzzle', 'tape', 'J', 'Spins. Calming. Slightly stabilising.', { mul: { spread: 0.95 }, quirks: q({ t: 'sanity', amount: 0.3 }), homemade: true, recipe: { cloth: 1, twig: 1, ...tape } }),
  A('choke_ring', 'Choke Ring', 'muzzle', 'thread', 'U', 'Tightens a pellet pattern.', { mul: { spread: 0.7, range: 1.1 }, requires: ['pellet'] }),
  A('wide_funnel', 'Wide Funnel', 'muzzle', 'tape', 'J', 'More pellets, much more chaos.', { mul: { spread: 1.6 }, add: { pellets: 2 }, requires: ['pellet'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  // ---- barrel
  A('long_pipe_barrel', 'Long Pipe Barrel', 'barrel', 'tube', 'U', 'Longer reach, faster bullets, heavier.', { mul: { range: 1.25, spread: 0.85, speed: 1.1 }, add: { weight: 0.6 }, forbid: ['toy'] }),
  A('sawed_down_barrel', 'Sawed-Down Barrel', 'barrel', 'tube', 'C', 'Handy and wild.', { mul: { range: 0.7, spread: 1.3 }, add: { weight: -0.5 }, requires: ['firearm'] }),
  A('drinking_straw_barrel', 'Drinking-Straw Barrel', 'barrel', 'tube', 'J', 'It really does help your aim. It also reduces the bang.', { mul: { spread: 0.75, damage: 0.9 }, add: { weight: -0.1 }, homemade: true, recipe: { twig: 2, ...tape } }),
  A('garden_hose_barrel', 'Garden-Hose Barrel', 'barrel', 'tube', 'J', 'Floppy and fun. Gains range, loses dignity.', { mul: { range: 1.1 }, add: { weight: 0.3 }, quirks: q({ t: 'wobble', rad: 0.05 }), homemade: true, recipe: { hose: 1, ...tape } }),
  A('ported_swiss_barrel', 'Swiss-Cheese Barrel', 'barrel', 'tube', 'U', 'Holes everywhere. Steadier, louder.', { mul: { spread: 0.85, noise: 1.15, damage: 0.97 }, forbid: ['toy'] }),
  A('pipe_extension', 'Pipe Extension', 'barrel', 'tube', 'J', 'More tube for the tube-gun.', { mul: { range: 1.3, speed: 1.1 }, add: { weight: 0.8 }, requires: ['tube'], homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  // ---- sights
  A('iron_notch', 'Iron Notch Sight', 'optic', 'rail', 'J', 'A notch and a post.', { mul: { spread: 0.97 } }),
  A('red_dot_sticker', 'Red Dot Sticker', 'optic', 'tape', 'J', 'A sticker with a red dot. It helps, honest.', { mul: { spread: 0.88 }, homemade: true, recipe: { cloth: 1, ...tape } }),
  A('jam_jar_scope', 'Jam-Jar Scope', 'optic', 'tape', 'J', 'Magnifies, wobbles.', { mul: { spread: 0.9, range: 1.15 }, add: { weight: 0.3 }, quirks: q({ t: 'wobble', rad: 0.015 }), homemade: true, recipe: { empty_bottle: 1, ...tape } }),
  A('scope_4x', '4x Scope', 'optic', 'rail', 'U', 'Proper glass.', { mul: { spread: 0.78, range: 1.25 }, add: { weight: 0.5 } }),
  A('binocular_half', 'Half a Binocular', 'optic', 'tape', 'U', 'Taped on. Shows range rings.', { mul: { range: 1.3 }, add: { weight: 0.6 }, info: ['range'], homemade: true, recipe: { scrap_metal: 1, empty_bottle: 1, ...tape } }),
  A('periscope_corner', 'Periscope Sight', 'optic', 'rail', 'U', 'Aim line, a little less reach.', { mul: { range: 0.92 }, info: ['trajectory'] }),
  A('rearview_mirror', 'Rear-View Mirror', 'optic', 'tape', 'J', 'See who is behind you. Threat compass.', { mul: { spread: 1.05 }, info: ['threat'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('bottle_cap_peephole', 'Bottle-Cap Peephole', 'optic', 'tape', 'J', 'Sharper, but a narrow view.', { mul: { spread: 0.85, range: 0.92 }, homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('smart_scope_9000', 'Smart Scope 9000', 'optic', 'rail', 'R', 'Ammo count, aim line, spread cone, range rings, damage falloff and health bars. Heavy.', { mul: { spread: 0.75, range: 1.2 }, add: { weight: 0.9 }, info: ['ammo', 'trajectory', 'spread', 'range', 'falloff', 'enemy_hp'] }),
  A('zombie_hp_monocle', 'Zombie Health Monocle', 'optic', 'tape', 'U', 'Floating health bars over enemies.', { add: { weight: 0.1 }, info: ['enemy_hp'], homemade: true, recipe: { scrap_metal: 1, empty_bottle: 1, ...tape } }),
  // ---- info gizmos (the visual info givers)
  A('ammo_counter_clip', 'Ammo Counter Clip', 'info', 'rail', 'C', 'A tiny dial that shows your exact ammo.', { info: ['ammo'], add: { weight: 0.1 } }),
  A('rangefinder_gizmo', 'Rangefinder Gizmo', 'info', 'rail', 'U', 'Range rings and a damage-falloff readout.', { info: ['range', 'falloff'], add: { weight: 0.2 } }),
  A('decibel_doodad', 'Decibel Doodad', 'info', 'tape', 'J', 'A dial that draws your noise ring, so you see who can hear you.', { info: ['noise'], add: { weight: 0.1 }, homemade: true, recipe: { scrap_metal: 1, wire: 1, ...tape } }),
  A('trajectory_ruler', 'Trajectory Ruler', 'info', 'tape', 'J', 'A taped ruler. Shows the aim line and your spread cone.', { info: ['trajectory', 'spread'], homemade: true, recipe: { twig: 1, ...tape } }),
  A('threat_radar_lamp', 'Threat Radar Lamp', 'info', 'rail', 'R', 'A little lamp that points at nearby danger and shows health bars.', { info: ['threat', 'enemy_hp'], add: { weight: 0.3, light: 2 } }),
  A('laser_level_tool', 'Laser Level', 'info', 'rail', 'U', 'A builder\'s laser: an aim line and a steadier shot.', { info: ['trajectory'], mul: { spread: 0.95 } }),
  // ---- underbarrel
  A('vertical_grip', 'Vertical Grip', 'underbarrel', 'rail', 'C', 'Steadier hands.', { mul: { spread: 0.9 }, add: { weight: 0.2 } }),
  A('bipod_forks', 'Fork Bipod', 'underbarrel', 'rail', 'U', 'Two forks. Rock steady, slower to swing.', { mul: { spread: 0.75, rate: 0.95 }, add: { weight: 0.6 } }),
  A('pool_noodle_foregrip', 'Pool-Noodle Foregrip', 'underbarrel', 'tape', 'J', 'Comfortable and boingy.', { mul: { spread: 0.93 }, quirks: q({ t: 'knockback', power: 0.8 }, { t: 'wobble', rad: 0.01 }), homemade: true, recipe: { cloth: 1, ...tape } }),
  A('plunger_grip', 'Plunger Grip', 'underbarrel', 'tape', 'J', 'Hits stick a little. Zombies pause.', { quirks: q({ t: 'stun', chance: 0.1, secs: 1.2 }), add: { weight: 0.3 }, homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('squirt_tank', 'Squirt Tank', 'underbarrel', 'tape', 'J', 'Soaks the target. Occasionally startles.', { quirks: q({ t: 'stun', chance: 0.15, secs: 0.8 }), add: { weight: 0.5 }, homemade: true, recipe: { empty_bottle: 1, ...tape } }),
  A('confetti_tube', 'Confetti Tube', 'underbarrel', 'tape', 'U', 'Party.', { quirks: q({ t: 'confetti' }, { t: 'stun', chance: 0.08, secs: 1.5 }), add: { weight: 0.3 } }),
  A('laser_cat_toy', 'Cat-Toy Laser', 'underbarrel', 'tape', 'J', 'Steadies your aim. Cheers you up.', { mul: { spread: 0.95 }, quirks: q({ t: 'sanity', amount: 0.2 }) }),
  A('flashlight_taped', 'Taped Flashlight', 'underbarrel', 'tape', 'J', 'Lights the way. Lights you up too.', { add: { light: 6, weight: 0.3 }, homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  // ---- side rail
  A('flashlight_clip', 'Flashlight Clip', 'side', 'rail', 'C', 'A proper clip-on light.', { add: { light: 6, weight: 0.25 } }),
  A('laser_pointer', 'Laser Pointer', 'side', 'rail', 'C', 'A dot to aim by.', { mul: { spread: 0.94 }, add: { light: 0.5 } }),
  A('bike_bell_side', 'Bike Bell', 'side', 'tape', 'J', 'Ding on every shot. A lure with a bell on it.', { quirks: q({ t: 'noise_pulse', extra: 8 }), homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('googly_eyes_side', 'Googly Eyes', 'side', 'tape', 'J', 'Watching. Restores a little sanity every shot.', { quirks: q({ t: 'sanity', amount: 0.5 }), homemade: true, recipe: { cloth: 1, ...tape } }),
  A('tiny_fan', 'Tiny Desk Fan', 'side', 'tape', 'J', 'Faster everything. Do not ask.', { mul: { rate: 1.08 }, add: { weight: 0.3 }, homemade: true, recipe: { scrap_metal: 1, wire: 1, ...tape } }),
  A('lava_lamp_clip', 'Lava Lamp', 'side', 'tape', 'U', 'A calming glow.', { add: { light: 4, weight: 0.6 }, quirks: q({ t: 'sanity', amount: 0.6 }) }),
  A('ketchup_tank', 'Ketchup Bottle', 'side', 'tape', 'J', 'Splat. Sometimes a startle.', { quirks: q({ t: 'confetti' }, { t: 'stun', chance: 0.03, secs: 0.6 }), homemade: true, recipe: { empty_bottle: 1, ...tape } }),
  A('spare_tape_roll', 'Spare Tape Roll', 'side', 'tape', 'J', 'A little tape saves a round, sometimes.', { quirks: q({ t: 'ammo_save', chance: 0.05 }), add: { weight: 0.2 } }),
  // ---- feed
  A('extended_feed', 'Extended Feed', 'feed', 'mag', 'C', 'More rounds, so sometimes a shot is free.', { quirks: q({ t: 'ammo_save', chance: 0.1 }), add: { weight: 0.3 } }),
  A('coffee_can_drum', 'Coffee-Can Drum', 'feed', 'tape', 'J', 'Fast, heavy, leaky.', { mul: { rate: 1.2, spread: 1.1 }, add: { weight: 1 }, quirks: q({ t: 'ammo_leak', chance: 0.06 }), homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  A('banana_feed', 'Banana Magazine', 'feed', 'tape', 'J', 'Curved, yellow, and somehow it saves ammo.', { quirks: q({ t: 'ammo_save', chance: 0.12 }, { t: 'wobble', rad: 0.01 }), add: { weight: 0.3 }, homemade: true, recipe: { cloth: 1, ...tape } }),
  A('speed_loader_hoop', 'Speed-Loader Hoop', 'feed', 'mag', 'C', 'Revolvers cycle faster.', { mul: { rate: 1.15 }, requires: ['revolver'] }),
  A('taped_double_feed', 'Taped Double Feed', 'feed', 'tape', 'U', 'Two feeds taped together. Quicker.', { mul: { rate: 1.12 }, add: { weight: 0.4 }, homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('sausage_belt', 'Sausage Belt', 'feed', 'tape', 'J', 'A belt of sausages. Fast and greasy.', { mul: { rate: 1.25 }, add: { weight: 1.4 }, quirks: q({ t: 'ammo_leak', chance: 0.1 }), requires: ['auto'], homemade: true, recipe: { cloth: 2, ...tape } }),
  // ---- stock
  A('skeleton_stock', 'Skeleton Stock', 'stock', 'any', 'C', 'Light, steady.', { mul: { spread: 0.88 }, add: { weight: -0.1 }, forbid: ['short'] }),
  A('broom_handle_stock', 'Broom-Handle Stock', 'stock', 'tape', 'J', 'Wobbly but it does help.', { mul: { spread: 0.85 }, add: { weight: 0.3 }, quirks: q({ t: 'wobble', rad: 0.01 }), homemade: true, recipe: { twig: 2, ...tape } }),
  A('pillow_shoulder_pad', 'Shoulder Pillow', 'stock', 'any', 'J', 'Soft landings.', { mul: { spread: 0.9, noise: 0.97 }, add: { weight: 0.3 }, forbid: ['short'] }),
  A('pogo_stock', 'Pogo Stock', 'stock', 'tape', 'J', 'Kills recoil. Pushes you backwards instead.', { mul: { spread: 0.8 }, quirks: q({ t: 'recoil_boing', strength: 2 }), homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  A('lawn_chair_stock', 'Lawn-Chair Stock', 'stock', 'tape', 'J', 'Folding. Comfortable. Slow.', { mul: { spread: 0.85, rate: 0.95 }, add: { weight: 1.2 }, homemade: true, recipe: { scrap_metal: 2, cloth: 1, ...tape } }),
  // ---- grip
  A('rubber_grip', 'Rubber Grip', 'grip', 'any', 'C', 'A better hold.', { mul: { spread: 0.93 } }),
  A('tape_wrapped_grip', 'Tape-Wrapped Grip', 'grip', 'tape', 'J', 'Sticky.', { mul: { spread: 0.95 }, homemade: true, recipe: { ...tape } }),
  A('oven_mitt_grip', 'Oven-Mitt Grip', 'grip', 'tape', 'J', 'Hot? Steady? Both.', { mul: { spread: 0.92 }, quirks: q({ t: 'sanity', amount: 0.2 }), homemade: true, recipe: { cloth: 2, ...tape } }),
  // ---- charms
  A('lucky_rabbit_foot', 'Lucky Rabbit Foot', 'charm', 'any', 'U', 'A shot is free now and then.', { quirks: q({ t: 'ammo_save', chance: 0.08 }, { t: 'sanity', amount: 0.2 }) }),
  A('fuzzy_dice_charm', 'Fuzzy Dice', 'charm', 'any', 'J', 'Good vibes.', { quirks: q({ t: 'sanity', amount: 0.5 }) }),
  A('flame_paint_job', 'Flame Paint Job', 'charm', 'any', 'J', 'Looks faster. Slightly louder.', { mul: { noise: 1.05 }, quirks: q({ t: 'sanity', amount: 0.3 }) }),
  A('rubber_duck_topper', 'Rubber Duck Topper', 'charm', 'tape', 'J', 'Squeaks every shot (a lure!).', { quirks: q({ t: 'noise_pulse', extra: 3 }, { t: 'sanity', amount: 0.4 }), homemade: true, recipe: { ...tape } }),
  A('bobblehead_dash', 'Bobblehead', 'charm', 'tape', 'J', 'Nods along. Slight wobble.', { quirks: q({ t: 'wobble', rad: 0.01 }, { t: 'sanity', amount: 0.5 }) }),
  // ---- melee: heads
  A('nail_studs', 'Nail Studs', 'head', 'tape', 'J', 'Nails through it. A classic.', { add: { damage: 4, noise: 1 }, requires: ['melee'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('horseshoe_head', 'Horseshoe Head', 'head', 'tape', 'C', 'Heavy, lucky, stunning.', { add: { damage: 3 }, quirks: q({ t: 'stun', chance: 0.1, secs: 1 }), requires: ['melee'] }),
  A('cheese_grater_head', 'Cheese-Grater Head', 'head', 'tape', 'J', 'Grates. Slower to swing.', { add: { damage: 5 }, mul: { rate: 0.9 }, requires: ['melee'], homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  A('bell_head', 'Bell Head', 'head', 'tape', 'J', 'Ring-a-ding. A lure.', { quirks: q({ t: 'noise_pulse', extra: 10 }), requires: ['melee'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
  A('rubber_ball_head', 'Rubber Ball Head', 'head', 'any', 'J', 'Weak hits, enormous bounces.', { mul: { damage: 0.6 }, quirks: q({ t: 'knockback', power: 3 }), requires: ['melee'] }),
  A('spiked_cone_head', 'Spiked Cone Head', 'head', 'tape', 'U', 'Pointy and slow.', { add: { damage: 6 }, mul: { rate: 0.85 }, requires: ['melee'], homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  // ---- melee: shaft / pommel
  A('extension_pole', 'Extension Pole', 'shaft', 'tape', 'J', 'Reach, at the cost of speed.', { add: { range: 0.6, weight: 0.4 }, mul: { rate: 0.9 }, requires: ['melee'], homemade: true, recipe: { twig: 2, ...tape } }),
  A('spring_shaft', 'Spring Shaft', 'shaft', 'tape', 'U', 'BOING on every hit.', { mul: { rate: 1.05 }, quirks: q({ t: 'knockback', power: 2 }), requires: ['melee'], homemade: true, recipe: { scrap_metal: 2, ...tape } }),
  A('weighted_sock_pommel', 'Weighted Sock', 'pommel', 'tape', 'J', 'A sock with a rock in it.', { add: { damage: 2, weight: 0.3 }, requires: ['melee'], homemade: true, recipe: { mismatched_sock: 1, stone: 1, ...tape } }),
  A('pommel_flashlight', 'Pommel Flashlight', 'pommel', 'tape', 'J', 'Light on the end.', { add: { light: 4, weight: 0.2 }, requires: ['melee'], homemade: true, recipe: { scrap_metal: 1, ...tape } }),
];

export const ATTACHMENT_BY_ID: Record<string, AttachmentDef> = Object.fromEntries(ATTACHMENTS.map((a) => [a.id, a]));
export type { InfoMode, Stats };
