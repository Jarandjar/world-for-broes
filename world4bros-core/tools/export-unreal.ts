/**
 * Exports the web game's arena data to JSON for the Unreal arena (docs/UE_PIPELINE.md §4.1).
 *   tools/unreal/arenas.json     layouts for tools/unreal/pipeline/build_arena.py (levels are built from it)
 *   tools/unreal/game_data.json  guns, bot levels and match rules, loaded by the Unreal game at runtime
 * Units stay metres and seconds; Unreal code multiplies distances by 100.
 * `solids` are the exact collision shapes the web sim uses, so Unreal never re-derives offsets.
 * If W4B_UE_PROJECT points at the Unreal project (folder or .uproject), both files are also copied to
 * <project>/Content/W4B/Data/.
 *
 * Run: npm run ue:export   (npm run arena:export is the same command)
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { ARENAS, getArenaCircleSolids, getArenaRectSolids } from '../src/sim/arena';
import { ARENA_BOT_LEVELS } from '../src/data/arenaBots';
import { ARENA_MATCH_RULES } from '../src/data/arenaRules';
import { ITEMS } from '../src/data/items';

const root = join(dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const outDir = join(root, 'tools/unreal');
mkdirSync(outDir, { recursive: true });
const arenasPath = join(outDir, 'arenas.json');
const gameDataPath = join(outDir, 'game_data.json');

const arenas: Record<string, unknown> = Object.fromEntries(
  Object.entries(ARENAS).map(([id, a]) => [id, { ...a, solids: { rects: getArenaRectSolids(a), circles: getArenaCircleSolids(a) } }]),
);

// arenas.json is generated. Arenas typed straight into it are kept (with a warning) instead of silently lost;
// move them into src/sim/arena.ts so the web game and its tests know them too.
if (existsSync(arenasPath)) {
  const old = JSON.parse(readFileSync(arenasPath, 'utf8')) as { arenas?: Record<string, unknown> };
  for (const [id, a] of Object.entries(old.arenas ?? {})) {
    if (!(id in arenas)) {
      console.warn(`WARNING: arena "${id}" exists only in arenas.json; kept it, but add it to src/sim/arena.ts`);
      arenas[id] = { ...(a as object), handAdded: true };
    }
  }
}

writeFileSync(arenasPath, JSON.stringify({
  version: 2,
  units: 'metres. Game x = east, game y = south, origin at the top-left corner; rect x/y is the top-left corner, points and circles are centres. Unreal (left-handed, no flip): X = (x - width/2) * 100, Y = (y - height/2) * 100.',
  source: 'src/sim/arena.ts',
  arenas,
}, null, 2));

const guns = Object.fromEntries(ITEMS.filter((i) => i.gun).map((i) => [i.id, { name: i.name, damage: i.tool?.damage ?? i.gun!.baseDamage, ...i.gun }]));
writeFileSync(gameDataPath, JSON.stringify({
  version: 1,
  units: 'metres, seconds, radians (spreadAngle, aimSpread); projectileSpeed in m/s; fireRate in shots per second',
  sources: ['src/data/items.ts', 'src/data/arenaBots.ts', 'src/data/arenaRules.ts'],
  guns,
  botLevels: ARENA_BOT_LEVELS,
  matchRules: ARENA_MATCH_RULES,
}, null, 2));
console.log(`exported ${Object.keys(arenas).length} arenas, ${Object.keys(guns).length} guns, ${Object.keys(ARENA_BOT_LEVELS).length} bot levels → ${outDir}`);

const ue = process.env.W4B_UE_PROJECT;
if (ue) {
  const projectDir = existsSync(ue) && statSync(ue).isFile() ? dirname(ue) : ue;
  const dataDir = join(projectDir, 'Content', 'W4B', 'Data');
  mkdirSync(dataDir, { recursive: true });
  copyFileSync(arenasPath, join(dataDir, 'arenas.json'));
  copyFileSync(gameDataPath, join(dataDir, 'game_data.json'));
  console.log(`copied both files to ${dataDir}`);
}
