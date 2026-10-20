/**
 * Asset catalog generator: lists every piece of art the isometric game needs, built from the game's own
 * content registries (src/data) plus the planned snap kit, furniture, road and decal sets.
 * For each asset: what to make (views, directions, animations, states) and the best free source to start from.
 *
 * Run: npm run art:catalog   (writes docs/ASSET_CATALOG.md and tools/blender/jobs/asset_catalog.json)
 * Style rules: docs/STYLE_BIBLE.md §0. Sourcing: docs/OPEN_SOURCE_TOOLKIT.md.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { buildContent } from '../src/data/content';

interface Asset {
  category: string;
  id: string;
  name: string;
  /** What to render: views, directions, frames, layers, states. */
  make: string;
  /** Best free starting point, then how to make it ours. */
  source: string;
  status: 'todo' | 'placeholder' | 'done';
}

const SRC = {
  ground: 'CC0 texture (ambientCG / Poly Haven) → tileable 2:1 diamond in Blender, tinted to the palette',
  tree: 'Sapling Tree Gen or tree-gen in Blender (tool GPL, output ours) → trunk + canopy split',
  rock: 'Blender procedural rock (displace + decimate) or Poly Haven CC0 scan',
  prop: 'Kenney / Quaternius CC0 model as base → re-shape, grime and re-texture in Blender',
  furniture: 'Kenney Furniture Kit (CC0) as base → gritty re-texture, damaged variant',
  vehicle: 'Kenney Car Kit (CC0) or our parametric car generator → parts as layers, grime',
  human: 'MPFB 2 body (CC0 assets) or Unreal Lyra mannequin + Mixamo / Lyra animations → render',
  animal: 'Quaternius animated animals (CC0, check pack) → render',
  kit: 'Own Blender snap-kit piece (City Sample PCG rules as reference), CC0 materials',
  icon: 'Render the 3D item model top-down at 2× (or Gemini 2D concept, redrawn) → 64 px icon',
  decal: 'Gemini 2D or Blender bake, alpha decal on the ground plane',
  fx: 'Blender particles / simulation rendered to frames, or hand-painted sprite frames',
};

const DIRS8 = '8 directions', DIRS16 = '16 directions';
const ANIMS_HUMAN = 'idle 8f, walk 8f, run 8f, attack 6f, hit 4f, die 8f @12 fps';
const STATES = 'states: intact / damaged / destroyed';

function fromContent(): Asset[] {
  const c = buildContent();
  const out: Asset[] = [];
  const name = (x: unknown, fallback: string): string => ((x as { name?: string }).name ?? fallback);

  for (const t of c.terrain.values()) {
    out.push({ category: 'Ground tiles', id: t.id, name: t.id, make: 'base diamond tile ×4 variants + 16/47 blob transitions + detail decals', source: SRC.ground, status: 'placeholder' });
  }
  for (const o of c.objects.values()) {
    const id = String(o.kind);
    const isTree = /tree|pine|oak|stump|bush|log/.test(id);
    const isRock = /rock|boulder|stone|ore|node|geode/.test(id);
    out.push({
      category: isTree ? 'Vegetation' : isRock ? 'Rocks & resources' : 'World props',
      id, name: o.name,
      make: isTree ? 'trunk + canopy (canopy fades when behind), wind-sway frames, cut/stump state' : `1 view${o.hp > 1 ? `, ${STATES}` : ''}`,
      source: isTree ? SRC.tree : isRock ? SRC.rock : SRC.prop, status: 'placeholder',
    });
  }
  for (const s of c.structures.values()) {
    out.push({ category: 'Player buildables', id: s.id, name: s.name, make: `4 rotations, ${STATES}${s.light ? ', light sprite' : ''}`, source: SRC.kit, status: 'placeholder' });
  }
  for (const b of c.buildings) {
    out.push({ category: 'Building types (assembled from the snap kit)', id: b.id, name: b.name, make: `kit pieces in this style; roof ${String((b as { roof?: string }).roof ?? '?')}, walls ${String((b as { walls?: string }).walls ?? '?')}; interior room templates`, source: SRC.kit, status: 'placeholder' });
  }
  for (const v of c.vehicles.values()) {
    out.push({ category: 'Vehicles', id: v.id, name: v.name, make: `${DIRS16} × clean/dented/wrecked; part layers (doors, hood, wheels, glass, lights)`, source: SRC.vehicle, status: 'placeholder' });
  }
  for (const z of c.zombies.values()) {
    out.push({ category: 'Zombies', id: z.id, name: z.name, make: `${DIRS8} × ${ANIMS_HUMAN}; mild blood variants`, source: SRC.human, status: 'placeholder' });
  }
  for (const a of c.animals.values()) {
    out.push({ category: 'Animals', id: a.id, name: a.name, make: `${DIRS8} × idle, walk, run, eat, hit, die`, source: SRC.animal, status: 'placeholder' });
  }
  for (const i of c.items.values()) {
    const held = i.equip ? `; worn/held layer on the character (${i.equip.slot}, ${DIRS8})` : '';
    out.push({ category: `Items: ${i.kind}`, id: i.id, name: i.name, make: `inventory icon + ground drop sprite${held}`, source: i.equip ? `${SRC.prop}; ${SRC.icon}` : SRC.icon, status: 'placeholder' });
  }
  for (const a of c.attachments.values()) {
    out.push({ category: 'Weapon attachments', id: String(a.id), name: name(a, String(a.id)), make: 'icon + layer on the weapon sprite', source: SRC.icon, status: 'placeholder' });
  }
  for (const e of c.effects.values()) {
    out.push({ category: 'Status effects', id: e.id, name: e.name, make: 'HUD icon + world FX frames (loop) + sound word', source: SRC.fx, status: 'placeholder' });
  }
  for (const h of c.hairStyles) {
    out.push({ category: 'Character: hair', id: String((h as { id: string }).id), name: name(h, String((h as { id: string }).id)), make: `${DIRS8} layer, recolourable`, source: SRC.human, status: 'placeholder' });
  }
  return out;
}

/** Art the game needs that is not (yet) a registry entry: the snap kit, furniture, roads, characters, decals. */
function planned(): Asset[] {
  const out: Asset[] = [];
  const add = (category: string, ids: string[], make: string, source: string): void => {
    for (const id of ids) out.push({ category, id, name: id.replace(/_/g, ' '), make, source, status: 'todo' });
  };
  const wallMats = ['siding', 'brick', 'plaster', 'concrete', 'wood_log', 'scrap'];
  add('Snap kit: floors', ['floor_wood', 'floor_tile', 'floor_carpet', 'floor_concrete', 'floor_linoleum', 'floor_dirt'], '1×1 diamond, clean/dirty/damaged', SRC.kit);
  add('Snap kit: walls', wallMats.flatMap((m) => [`wall_${m}`, `wall_${m}_window`, `wall_${m}_door`, `wall_${m}_double_door`, `wall_${m}_broken`, `wall_${m}_boarded`]), '2 orientations × intact/cracked/breached/rubble; cut-away (knee-high) variant', SRC.kit);
  add('Snap kit: openings', ['door_wood', 'door_metal', 'door_glass', 'garage_door', 'window_glass', 'window_broken', 'window_boarded', 'curtains'], '2 orientations × open/closed/broken', SRC.kit);
  add('Snap kit: roofs', ['roof_gable_slope', 'roof_ridge', 'roof_hip_corner', 'roof_flat_parapet', 'roof_edge', 'roof_tin', 'roof_broken'], 'pieces per material (shingle, tin, tar), intact/sagging/collapsed', SRC.kit);
  add('Snap kit: structure', ['stairs', 'railing', 'porch_post', 'column', 'chimney', 'fence_picket', 'fence_chainlink', 'gate', 'hedge'], '2 orientations, intact/damaged', SRC.kit);
  add('Furniture', ['bed', 'wardrobe', 'dresser', 'nightstand', 'fridge', 'stove', 'kitchen_counter', 'sink', 'toilet', 'bathtub', 'shower', 'sofa', 'armchair', 'tv', 'dining_table', 'chair', 'desk', 'bookshelf', 'cupboard', 'washing_machine', 'workbench', 'toolbox', 'safe', 'gun_cabinet', 'shop_shelf', 'shop_counter', 'cash_register', 'freezer_chest', 'hospital_bed', 'locker', 'filing_cabinet', 'crib', 'piano', 'fireplace'], '4 rotations, intact/looted/broken; container open/closed', SRC.furniture);
  add('Roads & streets', ['asphalt', 'asphalt_cracked', 'lane_line_white', 'lane_line_yellow', 'kerb', 'sidewalk', 'crosswalk', 'driveway', 'dirt_track', 'gravel_shoulder', 'manhole', 'storm_drain', 'pothole', 'oil_stain', 'parking_bay', 'road_barrier', 'traffic_light', 'stop_sign', 'street_sign'], 'tiles / decals / props following the road ribbons', SRC.ground);
  add('Characters: body & clothing', ['body_base', 'head_base', 'torso_shirt', 'torso_jacket', 'torso_hoodie', 'legs_jeans', 'legs_cargo', 'feet_boots', 'feet_sneakers', 'headwear_cap', 'headwear_helmet', 'backpack', 'face_mask'], `${DIRS8} × ${ANIMS_HUMAN}, recolourable layers`, SRC.human);
  add('Decals', ['blood_splat_small', 'blood_drip_wall', 'crack_ground', 'leaves', 'litter', 'puddle', 'graffiti', 'poster', 'scorch_mark', 'tyre_tracks', 'footprints'], '3–6 variants each, alpha, fade over time', SRC.decal);
  add('FX', ['muzzle_flash', 'bullet_impact', 'smoke_puff', 'fire_loop', 'explosion', 'dust_burst', 'glass_shatter', 'sparks', 'rain', 'snow', 'fog_layer'], 'frame sequences, additive where light', SRC.fx);
  add('Lights', ['street_light_pool', 'window_glow', 'headlight_cone', 'torch_pool', 'fire_glow'], 'light sprites for the night pass', SRC.fx);
  return out;
}

const assets = [...fromContent(), ...planned()];
const cats = [...new Set(assets.map((a) => a.category))];
const root = join(dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');

const json = join(root, 'tools/blender/jobs/asset_catalog.json');
mkdirSync(dirname(json), { recursive: true });
writeFileSync(json, JSON.stringify({ version: 1, generated: 'npm run art:catalog', count: assets.length, assets }, null, 2));

const esc = (s: string): string => s.replace(/\|/g, '\\|');
const lines: string[] = [];
lines.push('# Asset Catalog');
lines.push('');
lines.push('> **Generated** by `npm run art:catalog` (`tools/asset-catalog.ts`) from the game\'s content registries in `src/data` plus the planned snap kit, furniture, roads, characters, decals and FX. Do not edit the tables by hand — change the data or the script and re-run. Machine-readable copy: `tools/blender/jobs/asset_catalog.json`.');
lines.push('');
lines.push('Style for every asset: `STYLE_BIBLE.md` §0 (isometric 2:1, 1 tile = 1 m, light from the north-west, muted gritty palette, painted not pixel art, mild blood). Workflow and sources: `ASSET_PIPELINE.md`.');
lines.push('');
lines.push('Status: `placeholder` = exists in the game as code-drawn art, needs the HD sprite; `todo` = not in the game yet; `done` = final sprite in `public/assets/`.');
lines.push('');
lines.push('## Summary');
lines.push('');
lines.push('| Category | Count |');
lines.push('| :--- | ---: |');
for (const cat of cats) lines.push(`| ${cat} | ${assets.filter((a) => a.category === cat).length} |`);
lines.push(`| **Total** | **${assets.length}** |`);
for (const cat of cats) {
  const rows = assets.filter((a) => a.category === cat);
  lines.push('');
  lines.push(`## ${cat} (${rows.length})`);
  const shared = rows.every((r) => r.make === rows[0]!.make && r.source === rows[0]!.source);
  if (shared) {
    lines.push('');
    lines.push(`- **Make**: ${rows[0]!.make}`);
    lines.push(`- **Source**: ${rows[0]!.source}`);
    lines.push('');
    lines.push('| id | name | status |');
    lines.push('| :--- | :--- | :--- |');
    for (const r of rows) lines.push(`| \`${r.id}\` | ${esc(r.name)} | ${r.status} |`);
  } else {
    lines.push('');
    lines.push('| id | name | make | source | status |');
    lines.push('| :--- | :--- | :--- | :--- | :--- |');
    for (const r of rows) lines.push(`| \`${r.id}\` | ${esc(r.name)} | ${esc(r.make)} | ${esc(r.source)} | ${r.status} |`);
  }
}
writeFileSync(join(root, 'docs/ASSET_CATALOG.md'), lines.join('\n') + '\n');
console.log(`asset catalog: ${assets.length} assets in ${cats.length} categories`);
