import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const UE_DEFAULT = 'C:\\Program Files\\Epic Games\\UE_5.7\\Engine\\Binaries\\Win64';
const UE_BIN_DIR = process.env.UE_BIN_DIR || UE_DEFAULT;
const UE_CMD = process.env.UE_CMD_PATH || resolve(UE_BIN_DIR, 'UnrealEditor-Cmd.exe');
const UE_EDITOR = process.env.UE_EDITOR_PATH || resolve(UE_BIN_DIR, 'UnrealEditor.exe');
const UPROJECT = process.env.UE_PROJECT_PATH || resolve('tools/unreal/World4Bros/World4Bros.uproject');

const argv = process.argv.slice(2);
const command = argv.find((a) => !a.startsWith('--')) || 'open';
const arenaArg = argv.find((a) => a.startsWith('--arena='));
const presetArg = argv.find((a) => a.startsWith('--preset='));
const preset = presetArg ? presetArg.split('=')[1] : '';
const dryRun = argv.includes('--dry-run');
const shot = argv.includes('--shot');

const presetIsometric = preset === 'isometric-square';
const arena = arenaArg ? arenaArg.split('=')[1] : (presetIsometric ? 'square_arena' : 'duel_pit');
const windowed = argv.includes('--window') || shot || presetIsometric;
const shotPath = resolve('tools/unreal/Saved_shots/arena.png').replace(/\\/g, '/');
const invalidPreset = preset && !presetIsometric;
// Windowed: real RHI, visible editor, stays open after the script (no QUIT_EDITOR).
// Shot: real RHI but off-screen; build_arena.py saves a PNG then quits the editor itself.
const headlessFlags = shot ? ['-RenderOffScreen', '-unattended'] : windowed ? [] : ['-unattended', '-nullrhi'];
const quitCmd = windowed ? '' : ',QUIT_EDITOR';
// UE resolves relative py paths against Engine/Binaries/Win64, so pass forward-slash absolute paths.
const scriptPath = (name) => resolve('tools/unreal', name).replace(/\\/g, '/');

if (invalidPreset) {
  console.error(`[UE CLI] Unknown preset: ${preset}`);
  console.error('[UE CLI] Presets: isometric-square');
  process.exit(1);
}

const cmdExe = windowed ? (existsSync(UE_EDITOR) ? UE_EDITOR : UE_CMD) : (existsSync(UE_CMD) ? UE_CMD : UE_EDITOR);
const editorExe = existsSync(UE_EDITOR) ? UE_EDITOR : UE_CMD;

function run(exe, args, envExtra = {}) {
  if (!existsSync(exe)) {
    console.error(`[UE CLI] Unreal executable not found: ${exe}`);
    console.error('[UE CLI] Set UE_BIN_DIR or UE_CMD_PATH/UE_EDITOR_PATH environment variables.');
    process.exit(1);
  }
  if (!existsSync(UPROJECT)) {
    console.error(`[UE CLI] Project file not found: ${UPROJECT}`);
    process.exit(1);
  }

  console.log(`[UE CLI] Executable: ${exe}`);
  console.log(`[UE CLI] Project: ${UPROJECT}`);
  console.log(`[UE CLI] Args: ${args.join(' ')}`);

  if (dryRun) {
    console.log('[UE CLI] Dry-run complete. No process started.');
    return;
  }

  const child = spawn(exe, args, {
    stdio: 'inherit',
    env: { ...process.env, ...envExtra }
  });

  child.on('exit', (code) => process.exit(code ?? 0));
}

if (command === 'open') {
  run(editorExe, [UPROJECT]);
} else if (command === 'test-window') {
  run(existsSync(UE_EDITOR) ? UE_EDITOR : UE_CMD, [
    UPROJECT,
    `-ExecCmds=py ${scriptPath('build_arena.py')}`,
    '-nosplash',
    '-nosound',
    '-stdout',
    '-FullStdOutLogOutput'
  ], { W4B_ARENA_ID: 'square_arena' });
} else if (command === 'build-arena') {
  run(cmdExe, [
    UPROJECT,
    // Full editor init (not bare commandlet) so an editor world exists.
    `-ExecCmds=py ${scriptPath('build_arena.py')}${quitCmd}`,
    ...headlessFlags,
    '-nosplash',
    '-nosound',
    '-stdout',
    '-FullStdOutLogOutput'
  ], { W4B_ARENA_ID: arena, ...(shot ? { W4B_SHOT: shotPath } : {}) });
} else if (command === 'sim-headless') {
  run(cmdExe, [
    UPROJECT,
    `-ExecCmds=py ${scriptPath('run_headless_match.py')}${quitCmd}`,
    ...headlessFlags,
    '-nosplash',
    '-nosound',
    '-stdout',
    '-FullStdOutLogOutput'
  ]);
} else {
  console.error(`[UE CLI] Unknown command: ${command}`);
  console.error('[UE CLI] Commands: open | test-window | build-arena [--arena=duel_pit|square_arena|ffa_yard] [--preset=isometric-square] [--window|--shot|--dry-run] | sim-headless [--window|--dry-run]');
  process.exit(1);
}
