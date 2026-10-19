import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Phase Z6 Live Runtime Verification...');
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

  // Setup daytime lighting, godmode, and clean UI
  await page.evaluate(() => {
    const game = window.game;
    if (game) {
      game.stats.health = 100;
      game.stats.isDead = false;
      game.stats.takeDamage = () => {};
      const death = document.getElementById('death-screen');
      if (death) death.classList.add('hidden');
      const start = document.getElementById('start-overlay');
      if (start) start.classList.add('hidden');
      const craft = document.getElementById('crafting-sidebar');
      if (craft) craft.classList.remove('drawer-open');

      // Set pleasant daytime
      game.dayNight.timeOfDay = 25;
    }
  });

  // 1. Board starter hatchback and show driving layout
  console.log('Boarding starter hatchback and activating Artboard 2 Phone Driving Cockpit...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.vehicles) {
      const pPos = game.player.position;
      const THREE = window.THREE || game.player.position.constructor;
      
      // Spawn hatchback right in front of player
      const car = game.world.vehicles.spawnVehicle(
        'rusty_hatchback',
        new THREE(pPos.x + 2.5, 0, pPos.z),
        Math.PI * 0.25
      );

      // Board the car
      game.enterVehicle(car);
      car.speed = 22.0 / 3.6; // ~22 km/h
    }
  });

  await page.waitForTimeout(800);
  await page.screenshot({ path: 'phase6_vehicle_cockpit_overview.png' });
  console.log('📸 Captured phase6_vehicle_cockpit_overview.png (Cockpit HUD, gauges, pedals & headlights!)');

  // 2. High-speed cornering & drift with tire smoke
  console.log('Triggering high-speed cornering drift with suspension body roll and steering knuckle yaw...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.vehicles) {
      const car = game.world.vehicles.getActiveVehicle();
      if (car) {
        car.speed = 12.0; // 43 km/h
        car.applyInput(1.0, 0, 1.0, true); // Hard steer right with handbrake drift
        game.drivingLayout.update(car);
      }
    }
  });

  await page.waitForTimeout(400); // Capture body roll and tire smoke
  await page.screenshot({ path: 'phase6_vehicle_driving_drift.png' });
  console.log('📸 Captured phase6_vehicle_driving_drift.png (Drift slide, body roll & knuckle yaw!)');

  // 3. Ramming a zombie with vehicle
  console.log('Spawning zombie in vehicle path and capturing vehicular impact...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.vehicles && game.world.zombies) {
      const car = game.world.vehicles.getActiveVehicle();
      if (car) {
        const THREE = window.THREE || game.player.position.constructor;
        // Position zombie right ahead of car
        const forward = new THREE(0, 0, 2.2).applyAxisAngle(new THREE(0, 1, 0), car.group.rotation.y);
        const zPos = car.position.clone().add(forward);
        const zombie = game.world.zombies.spawnZombie('traffic_cone_zombie', zPos);

        // Ram zombie with car
        car.speed = 11.0;
        const dmg = car.handleRamZombie(zombie.position, game.particles, game.camera);
        const knockback = new THREE(0, 0, 1).applyAxisAngle(new THREE(0, 1, 0), car.group.rotation.y).multiplyScalar(15);
        game.world.zombies.hitZombiesInRadius(zombie.position, 1.8, dmg, knockback, game.particles, game.camera);
      }
    }
  });

  await page.waitForTimeout(300); // BONK! comic pop & green smoke puff
  await page.screenshot({ path: 'phase6_vehicle_zombie_ram.png' });
  console.log('📸 Captured phase6_vehicle_zombie_ram.png (BONK pop, green puff & zombie ram impact!)');

  console.log('Console errors observed:', errors);
  await browser.close();
  console.log('🎉 Phase Z6 Live Runtime Verification complete!');
}

verify().catch(err => {
  console.error('Phase 6 verification error:', err);
  process.exit(1);
});
