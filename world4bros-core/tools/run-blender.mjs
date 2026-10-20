import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const BLENDER_DEFAULT_WIN = 'C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe';
const blenderBin = process.env.BLENDER_PATH || (existsSync(BLENDER_DEFAULT_WIN) ? BLENDER_DEFAULT_WIN : 'blender');

const args = ['--background', '--python', 'tools/blender/pipeline_runner.py', '--', ...process.argv.slice(2)];

console.log(`[Art Pipeline] Launching Blender: ${blenderBin}`);
console.log(`[Art Pipeline] Args: ${args.join(' ')}\n`);

const child = spawn(blenderBin, args, { stdio: 'inherit' });

child.on('exit', (code) => {
  process.exit(code ?? 0);
});
