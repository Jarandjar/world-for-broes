// Bundles the game into one self-contained HTML file (dist/single.html) for sharing or hosting anywhere.
// Usage: npm run bundle   (needs `npm install` first so esbuild is on PATH)
import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

mkdirSync('dist', { recursive: true });
await esbuild.build({
  entryPoints: ['src/main.ts'],
  bundle: true,
  format: 'iife',
  target: 'es2022',
  minify: true,
  outfile: 'dist/game.js'
});
const js = readFileSync('dist/game.js', 'utf8').replace(/<\/script>/g, '<\\/script>');
const html = readFileSync('index.html', 'utf8').replace('<script type="module" src="/src/main.ts"></script>', () => `<script>${js}</script>`);
writeFileSync('dist/single.html', html);
console.log(`dist/single.html written (${(html.length / 1024).toFixed(0)} KB)`);
