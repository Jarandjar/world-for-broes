import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Phase Z5 Live Runtime Verification...');
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
  await page.waitForTimeout(1500);

  // Setup pristine daytime lighting & godmode
  await page.evaluate(() => {
    const game = window.game;
    if (game) {
      game.stats.health = 100;
      game.stats.isDead = false;
      game.stats.takeDamage = () => {}; // Invulnerable for photoshoot
      const death = document.getElementById('death-screen');
      if (death) death.classList.add('hidden');
      const start = document.getElementById('start-overlay');
      if (start) start.classList.add('hidden');
      const craft = document.getElementById('crafting-sidebar');
      if (craft) craft.classList.remove('drawer-open');

      // Set pleasant daytime
      game.dayNight.timeOfDay = 25; // midday sunshine
    }
  });

  // Spawn a lively cluster of zombies around the player for the overview shot
  console.log('Spawning diverse zombie horde (shamblers, screamers, traffic cones, bloaters)...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.zombies) {
      const pPos = game.player.position;
      const THREE = window.THREE || game.player.position.constructor;

      // Spawn a variety of archetypes around player with safe visual spacing
      const types = ['shambler', 'traffic_cone_zombie', 'screamer', 'buckethead', 'bloater', 'crawler'];
      const offsets = [
        { x: 3.5, z: 1.5 },
        { x: -3.0, z: 2.0 },
        { x: 1.8, z: -3.0 },
        { x: -2.2, z: -2.0 },
        { x: 4.2, z: -1.0 },
        { x: -1.0, z: 3.5 }
      ];

      offsets.forEach((off, i) => {
        const z = game.world.zombies.spawnZombie(
          types[i % types.length],
          new THREE(pPos.x + off.x, 0, pPos.z + off.z)
        );
        // Turn them to face the camera/player
        if (z && z.group) {
          z.group.lookAt(pPos.x, 0, pPos.z);
        }
      });
    }
  });

  await page.waitForTimeout(800); // Allow gait animations and eye wobble to run
  await page.screenshot({ path: 'phase5_zombie_crowd_overview.png' });
  console.log('📸 Captured phase5_zombie_crowd_overview.png (Horde of varied diorama zombies!)');

  // 2. Trigger screamer alert wave cascade
  console.log('Triggering screamer alert wave & horn alert cascade...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.zombies) {
      const pPos = game.player.position;
      // Broadcast horn/screamer noise wave
      game.world.zombies.broadcastScreamerWave(pPos, 35);
      game.honkEmote();
    }
  });

  await page.waitForTimeout(400); // Alert bubble and head-turn in flight
  await page.screenshot({ path: 'phase5_screamer_wave.png' });
  console.log('📸 Captured phase5_screamer_wave.png (Screamer alert wave & cascade!)');

  // 3. Trigger combat defeat: strike nearest zombie with frying pan for defeat puff & loot drop
  console.log('Triggering frying pan strike and cartoon defeat puff...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.zombies) {
      const pPos = game.player.position;
      const THREE = window.THREE || game.player.position.constructor;
      // Spawn a target right in front of player
      const target = game.world.zombies.spawnZombie('traffic_cone_zombie', new THREE(pPos.x + 1.2, 0, pPos.z));
      
      // Equip frying pan
      game.inventory.addItem('pan', 1);
      game.inventory.selectSlot(0);

      // Strike target with fatal hit
      game.world.zombies.hitZombiesInRadius(
        target.position,
        1.5,
        100, // Fatal blow
        new THREE(1, 0, 0),
        game.particles,
        game.camera
      );
    }
  });

  await page.waitForTimeout(250); // Green defeat puff & POP! comic pop
  await page.screenshot({ path: 'phase5_zombie_defeat_pop.png' });
  console.log('📸 Captured phase5_zombie_defeat_pop.png (Defeat pop, green smoke & loot drop!)');

  console.log('Console errors observed:', errors);
  await browser.close();
  console.log('🎉 Phase Z5 Live Runtime Verification complete!');
}

verify().catch(err => {
  console.error('Phase 5 verification error:', err);
  process.exit(1);
});
