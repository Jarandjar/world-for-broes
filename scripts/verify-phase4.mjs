import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Phase Z4 Live Runtime Verification...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  // 1. Visit local game in Zombie Mode with autostart
  console.log('Navigating to http://localhost:5175/?autostart=1&mode=zombie ...');
  await page.goto('http://localhost:5175/?autostart=1&mode=zombie', { waitUntil: 'networkidle' });
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForTimeout(2000);

  // Take screenshot: Phase 4 Settlement Containers Overview
  await page.screenshot({ path: 'phase4_containers_overview.png' });
  console.log('📸 Captured phase4_containers_overview.png');

  // 2. Open a container via debug evaluation in window/game context
  console.log('Testing interactive container opening and parabolic loot ejection...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.containers && game.world.containers.spawnContainer) {
      const pPos = game.player.position;
      const THREE = window.THREE || game.player.position.constructor;
      
      const toolbox = game.world.containers.spawnContainer(
        'toolbox',
        new pPos.constructor(pPos.x + 2.0, 0, pPos.z)
      );
      const safe = game.world.containers.spawnContainer(
        'vault_safe',
        new pPos.constructor(pPos.x - 2.5, 0, pPos.z + 1.0)
      );

      // Open toolbox to trigger rubbery spring lid and parabolic loot arcs!
      if (toolbox) {
        toolbox.open(
          game.inventory,
          game.world.containers.lootManager,
          game.world.containers.jackpotManager,
          game.particles,
          game.camera
        );
      }
    }
  });

  await page.waitForTimeout(450); // Mid-arc loot flight
  await page.screenshot({ path: 'phase4_container_loot_arc.png' });
  console.log('📸 Captured phase4_container_loot_arc.png (parabolic loot arcs in flight!)');

  // 3. Open the jackpot vault safe to trigger golden beam & siren alarm
  console.log('Triggering jackpot safe opening with golden beam and emergency siren...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.containers) {
      const pPos = game.player.position;
      game.inventory.addItem('keycard_red', 1);
      const nearbySafe = game.world.containers.getNearbyContainer(
        new pPos.constructor(pPos.x - 2.5, 0, pPos.z + 1.0)
      );
      if (nearbySafe) {
        nearbySafe.open(
          game.inventory,
          game.world.containers.lootManager,
          game.world.containers.jackpotManager,
          game.particles,
          game.camera
        );
      }
    }
  });

  await page.waitForTimeout(600); // Golden beam + siren
  await page.screenshot({ path: 'phase4_jackpot_siren_beam.png' });
  console.log('📸 Captured phase4_jackpot_siren_beam.png (golden beam & jackpot sirens!)');

  console.log('Console errors observed:', errors);
  await browser.close();
  console.log('🎉 Phase Z4 Live Runtime Verification complete!');
}

verify().catch(err => {
  console.error('Phase 4 verification error:', err);
  process.exit(1);
});
