import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const ARTIFACT_DIR = 'C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622';

async function verify() {
  console.log('🚀 Launching headless browser for Lantern Hollow verification...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  page.on('pageerror', err => console.error('❌ Page error:', err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('❌ Console error:', msg.text());
  });

  console.log('Navigating to http://localhost:5175/lantern-hollow.html ...');
  await page.goto('http://localhost:5175/lantern-hollow.html');
  await page.waitForSelector('#gl');

  console.log('Waiting for canvas and WebGL frame...');
  await page.waitForTimeout(3000);

  // Check if any shader errors were recorded
  const shaderErr = await page.evaluate(() => window.__shaderError);
  if (shaderErr) {
    throw new Error('Shader compilation error: ' + shaderErr);
  }

  // Ensure local artifacts dir exists
  if (!fs.existsSync('artifacts')) fs.mkdirSync('artifacts', { recursive: true });

  // 1. Capture Don't Starve Gameplay Overview showing full survival HUD, settlement & goofy world objects
  console.log('📸 1. Capturing Don\'t Starve Gameplay Overview with new items & objects...');
  const topdownPath = 'artifacts/lantern_hollow_settlement.png';
  await page.screenshot({ path: topdownPath });
  fs.copyFileSync(topdownPath, path.join(ARTIFACT_DIR, 'lantern_hollow_settlement.png'));
  console.log('✅ Captured & copied lantern_hollow_settlement.png');

  // 2. Open Storage Chest Modal (Backpack + 16-slot Storage Chest)
  console.log('📸 2. Opening Storage Chest Modal...');
  await page.evaluate(() => {
    window.openChestModal();
  });
  await page.waitForTimeout(600);
  const chestPath = 'artifacts/lantern_hollow_chest.png';
  await page.screenshot({ path: chestPath });
  fs.copyFileSync(chestPath, path.join(ARTIFACT_DIR, 'lantern_hollow_chest.png'));
  console.log('✅ Captured & copied lantern_hollow_chest.png');

  // 3. Open Crafting Tech Tree Drawer with Category Tabs & Debug Spawner
  console.log('📸 3. Opening Crafting Tech Tree Drawer...');
  await page.evaluate(() => {
    document.querySelector('#chest-modal').classList.remove('open');
    window.toggleCraftDrawer(true);
  });
  await page.waitForTimeout(600);
  const craftPath = 'artifacts/lantern_hollow_crafting.png';
  await page.screenshot({ path: craftPath });
  fs.copyFileSync(craftPath, path.join(ARTIFACT_DIR, 'lantern_hollow_crafting.png'));
  console.log('✅ Captured & copied lantern_hollow_crafting.png');

  // 4. Trigger Old Stuffins Dialogue Speech Bubble & Goofy Objects
  console.log('📸 4. Triggering Old Stuffins Dialogue Speech Bubble...');
  await page.evaluate(() => {
    window.toggleCraftDrawer(false);
    const bubble = document.querySelector('#dialogue-bubble');
    document.querySelector('#dialogueText').textContent = '"A rubber chicken in the hand is worth two zombies in the bush! And NEVER honk at the Highway Hulk!"';
    bubble.classList.add('open');
  });
  await page.waitForTimeout(600);
  const goofyPath = 'artifacts/lantern_hollow_goofy_objects.png';
  await page.screenshot({ path: goofyPath });
  fs.copyFileSync(goofyPath, path.join(ARTIFACT_DIR, 'lantern_hollow_goofy_objects.png'));
  console.log('✅ Captured & copied lantern_hollow_goofy_objects.png');

  await browser.close();
  console.log('🎉 All Lantern Hollow verification screenshots captured and copied successfully!');
}

verify().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
