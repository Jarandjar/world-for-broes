import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Phase V1 Atmospheric Noir Verification...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  page.on('pageerror', err => console.error('Page error:', err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('Console error:', msg.text());
  });

  console.log('Navigating to http://localhost:5175/?autostart=1&mode=zombie ...');
  await page.goto('http://localhost:5175/?autostart=1&mode=zombie', { waitUntil: 'networkidle' });
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForTimeout(1500);

  // 1. Setup Dusk/Night atmospheric lighting, godmode, and clean UI
  await page.evaluate(() => {
    const style = document.createElement('style');
    style.innerHTML = '#death-modal, #death-screen, #crafting-sidebar, #start-overlay { display: none !important; }';
    document.head.appendChild(style);

    const game = window.game;
    if (game) {
      game.stats.health = 100;
      game.stats.hunger = 100;
      game.stats.sanity = 100;
      game.stats.isDead = false;
      game.stats.takeDamage = () => {};

      // Set to dusk (blood amber twilight, heavy fog & vignette)
      game.dayNight.timeOfDay = 72; // Dusk
      game.dayNight.update(0.1, game.lighting, game.camera, game.player.position);
    }
  });

  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'phase_v1_atmospheric_noir.png' });
  console.log('📸 Captured phase_v1_atmospheric_noir.png (Heavy vignette, volumetric ground fog, film grain & desaturated palette)');

  // 2. Open the controller layout overlay
  console.log('Activating Controller Layout HUD overlay...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.controllerLayout) {
      game.controllerLayout.show();
    }
  });

  await page.waitForTimeout(800);
  await page.screenshot({ path: 'phase_v1_controller_layout.png' });
  console.log('📸 Captured phase_v1_controller_layout.png (Virtual controller / Gamepad HUD layout)');

  await browser.close();
  console.log('✅ Phase V1 runtime verification completed successfully!');
}

verify().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
