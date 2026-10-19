import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Noir Characters & Lighting Verification...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  page.on('pageerror', err => console.error('Page error:', err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('Console error:', msg.text());
  });

  console.log('Navigating to http://localhost:5175/?autostart=1&mode=zombie&noir=1 ...');
  await page.goto('http://localhost:5175/?autostart=1&mode=zombie&noir=1', { waitUntil: 'networkidle' });
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForTimeout(2000);

  // Configure clean UI and nighttime lighting to emphasize the harsh campfire shadows and glowing eyes
  await page.evaluate(() => {
    const style = document.createElement('style');
    style.innerHTML = '#death-modal, #death-screen, #darkness-terror, #crafting-sidebar, #start-overlay, #dst-clock, #vitals-container, #inventory-bar, #emote-bar, #controls-hint { display: none !important; }';
    document.head.appendChild(style);

    const game = window.game;
    if (game) {
      game.stats.health = 100;
      game.stats.hunger = 100;
      game.stats.sanity = 100;
      game.stats.isDead = false;
      game.stats.takeDamage = () => {};

      // Relocate the legacy cartoon player far away so only the gritty noir survivor & shambler are in view
      game.player.position.set(-100, 0, -100);
      game.player.prevPosition.set(-100, 0, -100);
      game.player.renderPosition.set(-100, 0, -100);
      game.player.group.position.set(-100, 0, -100);
      game.player.group.visible = false;

      // Focus camera on campfire center
      game.cameraFocus.set(0, 0.9, 0);

      // Nighttime (290 deg) for pitch-black atmosphere and harsh campfire lighting
      game.dayNight.timeOfDay = 960; // 960/1200 = 0.80 = 288 deg (Night)
      game.dayNight.update(0.1, game.lighting, game.camera, game.cameraFocus);

      // Center the camera on the showcase scene
      if (game.noirShowcase) {
        game.noirShowcase.activate();
      }

      game.cameraDistance = 7.5;
      game.targetCameraDistance = 7.5;
      game.cameraHeight = 3.8;
      game.targetCameraAngle = 0;
      game.currentCameraAngle = 0;
      game.updateCameraPosition(0.1);
    }
  });

  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'noir_characters_showcase.png' });
  console.log('📸 Captured noir_characters_showcase.png (Survivor & Shamblers at Campfire with Volumetric Fog)');

  // 2. Zoomed in dynamic camera angle for closeup of normal maps and glowing eyes
  console.log('Framing closeup of Survivor and Shambler...');
  await page.evaluate(() => {
    const game = window.game;
    if (game) {
      game.player.position.set(-100, 0, -100);
      game.player.prevPosition.set(-100, 0, -100);
      game.player.renderPosition.set(-100, 0, -100);
      game.player.group.position.set(-100, 0, -100);
      game.player.group.visible = false;

      // Lower and zoom in camera directly onto the survivor & shambler
      game.cameraDistance = 5.0;
      game.targetCameraDistance = 5.0;
      game.cameraHeight = 2.4;
      game.targetCameraAngle = 0;
      game.currentCameraAngle = 0;
      game.cameraFocus.set(0, 0.8, 0);
      game.updateCameraPosition(0.1);
    }
  });

  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'noir_characters_closeup.png' });
  console.log('📸 Captured noir_characters_closeup.png (Closeup: Normal mapping, clothing relief & glowing eyes)');

  await browser.close();
  console.log('✅ Noir characters live runtime verification complete!');
}

verify().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
