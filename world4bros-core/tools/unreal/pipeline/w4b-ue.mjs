#!/usr/bin/env node
/**
 * W4B Unreal pipeline commands (docs/UE_PIPELINE.md section 4). Every command was tested on UE 5.7.3 on 2026-10-20.
 * The Unreal project comes from the W4B_UE_PROJECT environment variable (path to the .uproject).
 *
 *   npm run ue:doctor                 check this PC (Visual Studio, disk, git, project settings)
 *   npm run ue:export                 web data -> tools/unreal/*.json (and into the project if W4B_UE_PROJECT is set)
 *   npm run ue:arena [-- duel_pit]    build arena levels headless, then verify them
 *   npm run ue:shots [-- duel_pit]    PNG of each arena from its preview camera -> <project>/Saved/W4B/shots/
 *   npm run ue:import -- <manifest>   import Blender/CC0 files with size checks
 *   npm run ue:remote -- --exec "..." run Python inside the open editor (bRemoteExecution=True)
 *   npm run ue:open                   open the editor
 */
import { spawnSync, spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '..', '..', '..');
const ueRoot = process.env.W4B_UE_ROOT || 'C:\\Program Files\\Epic Games\\UE_5.7';
const project = process.env.W4B_UE_PROJECT;
const [cmd = 'help', ...rest] = process.argv.slice(2);

const ps = (file, args) => {
  const r = spawnSync('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', join(here, file), ...args], { stdio: 'inherit' });
  return r.status ?? 1;
};
const needProject = () => {
  if (!project || !existsSync(project)) {
    console.error('W4B: set W4B_UE_PROJECT to the .uproject path first (docs/UE_PIPELINE.md section 3), e.g.');
    console.error('  setx W4B_UE_PROJECT C:\\W4B\\W4BArena\\W4BArena.uproject   (then open a new terminal)');
    process.exit(2);
  }
};
const arenaIds = () => {
  const ids = rest.filter((a) => !a.startsWith('-'));
  if (ids.length) return ids;
  return Object.keys(JSON.parse(readFileSync(join(repo, 'tools/unreal/arenas.json'), 'utf8')).arenas);
};
const mapName = (id) => 'L_W4B_' + id.split('_').map((p) => (['ffa', 'tdm', 'ctf'].includes(p) ? p.toUpperCase() : p[0].toUpperCase() + p.slice(1))).join('');

let code = 0;
switch (cmd) {
  case 'doctor':
    code = ps('doctor.ps1', rest);
    break;
  case 'arena': {
    needProject();
    const which = rest.filter((a) => !a.startsWith('-'));
    const arg = which.length === 1 ? which[0] : 'all';
    code = ps('ue.ps1', ['-Project', project, '-Script', join(here, 'build_arena.py'), '-ScriptArgs', `--arena ${arg}`]);
    if (code === 0) code = ps('ue.ps1', ['-Project', project, '-Script', join(here, 'verify_arena.py'), '-ScriptArgs', `--arena ${arg}`]);
    break;
  }
  case 'shots': {
    needProject();
    for (const id of arenaIds()) {
      const out = join(dirname(project), 'Saved', 'W4B', 'shots', `${id}.png`);
      code = ps('shoot.ps1', ['-Project', project, '-Map', `/Game/W4B/Maps/${mapName(id)}`, '-Out', out]) || code;
    }
    break;
  }
  case 'import':
    needProject();
    if (!rest[0]) { console.error('usage: npm run ue:import -- <manifest.json>'); process.exit(2); }
    code = ps('ue.ps1', ['-Project', project, '-Script', join(here, 'import_assets.py'), '-ScriptArgs', `--manifest ${resolve(rest[0]).replace(/\\/g, '/')}`]);
    break;
  case 'remote':
    code = spawnSync('python', [join(here, 'ue_remote.py'), ...rest], { stdio: 'inherit' }).status ?? 1;
    break;
  case 'open':
    needProject();
    spawn(join(ueRoot, 'Engine', 'Binaries', 'Win64', 'UnrealEditor.exe'), [project], { detached: true, stdio: 'ignore' }).unref();
    break;
  case 'sim-headless':
    console.error('W4B: there is no headless match simulation yet. The old run_headless_match.py only printed made-up');
    console.error('numbers. The real bot-vs-bot Test Bench is step U9 in docs/UE_ARENA_PLAN.md.');
    code = 1;
    break;
  default:
    console.log('commands: doctor | arena [ids] | shots [ids] | import <manifest> | remote --exec "..." | open');
}
process.exit(code);
