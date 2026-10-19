/**
 * One-command release:  npm run release
 *   1. Generate any missing sounds via ElevenLabs
 *   2. Abort if ANY sound is missing (never ship a half-silent game)
 *   3. Type-check + production build
 *   4. Deploy to Vercel production (uses VITE_WS_URL from .env if set)
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { missingSounds } from './generate-sounds.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const envPath = path.join(ROOT, '.env');
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
}

function run(label, cmd, args) {
  console.log(`\n▶ ${label}`);
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: 'inherit', shell: true, env: process.env });
  if (r.status !== 0) {
    console.error(`\n❌ Release aborted at step: ${label}`);
    process.exit(r.status ?? 1);
  }
}

run('1/4 Generating sounds (ElevenLabs)', 'node', ['scripts/generate-sounds.mjs']);

const missing = missingSounds();
if (missing.length) {
  console.error(`❌ Not deploying — missing sounds: ${missing.join(', ')}`);
  process.exit(1);
}
console.log('\n✔ 2/4 All sounds verified');

run('3/4 Building production bundle', 'npm', ['run', 'build']);

const deployArgs = ['deploy', '--prod', '--yes'];
if (process.env.VITE_WS_URL) {
  deployArgs.push('--build-env', `VITE_WS_URL=${process.env.VITE_WS_URL}`);
} else {
  console.log('ℹ VITE_WS_URL not set — the deployed site will run in Solo/Offline mode until the multiplayer server is hosted.');
}
run('4/4 Deploying to Vercel', 'vercel', deployArgs);

console.log('\n🚀 Release complete!');
