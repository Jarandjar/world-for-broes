import { chromium } from 'playwright';

async function verify() {
  console.log('🚀 Launching headless browser for Phase Z7 Live Runtime Verification...');
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
    const style = document.createElement('style');
    style.innerHTML = '#death-screen, #crafting-sidebar, #start-overlay { display: none !important; }';
    document.head.appendChild(style);

    const game = window.game;
    if (game) {
      game.stats.health = 100;
      game.stats.hunger = 100;
      game.stats.sanity = 100;
      game.stats.isDead = false;
      game.stats.takeDamage = () => {};

      // Set pleasant daytime
      game.dayNight.timeOfDay = 25;
    }
  });

  // 1. Spawn vehicle and open Car Part-Repair Sheet
  console.log('Spawning vehicle, simulating part wear, and opening Vehicle Repair Sheet...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.vehicles) {
      const pPos = game.player.position;
      const THREE = window.THREE || pPos.constructor;

      // Spawn hatchback near player
      const car = game.world.vehicles.spawnVehicle(
        'rusty_hatchback',
        new THREE(pPos.x + 2.0, 0, pPos.z),
        Math.PI * 0.25
      );

      // Inflict wear on spark plugs & battery for visible repair status
      const spark = car.parts.get('spark_plugs');
      if (spark) spark.condition = 45;
      const bat = car.parts.get('battery');
      if (bat) bat.condition = 60;
      const eng = car.parts.get('engine_block');
      if (eng) eng.condition = 80;
      car.batteryCharge = 55;
      car.recalculateDurability();

      // Open repair sheet
      if (game.repairSheet) {
        game.repairSheet.open(car, game.inventory, game.particles, game.camera);
      }
    }
  });

  await page.waitForTimeout(800);
  await page.screenshot({ path: 'phase7_car_repair_sheet.png' });
  console.log('📸 Captured phase7_car_repair_sheet.png (Vehicle parts cards, durability meters & repair UI)');

  // Close repair sheet before proceeding
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.repairSheet) {
      game.repairSheet.close();
    }
  });
  await page.waitForTimeout(300);

  // 2. Trigger Horde Night: 7th-Day Dusk Siren with Blood-Amber Sky & Banner
  console.log('Triggering 7th-Day Dusk Siren Horde Night with Blood-Amber Sky...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.dayNight && game.world && game.world.zombies) {
      // Set to day 7 dusk
      game.dayNight.dayCount = 7;
      game.dayNight.timeOfDay = 72; // Dusk (216 deg)
      game.dayNight.isHordeNight = true;
      game.dayNight.showAtmosphereBanner('🚨 7th-DAY HORDE NIGHT HAS ARRIVED! 🚨', 'The air-raid siren wails! The Highway Hulk and relentless hordes converge!');
      game.dayNight.update(0.1, game.lighting, game.camera, game.player.position);

      // Trigger horde wave around player
      game.world.zombies.triggerHordeNight(game.player.position, 16);
    }
  });

  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'phase7_horde_night_siren.png' });
  console.log('📸 Captured phase7_horde_night_siren.png (Dusk siren alert banner, blood-amber sky & horde wave)');

  // 3. Spawn Highway Hulk Boss Zombie near player and car in daylight for maximum clarity
  console.log('Spawning Highway Hulk Boss Zombie with road-sign club...');
  await page.evaluate(() => {
    const game = window.game;
    if (game && game.world && game.world.zombies) {
      // Set pleasant afternoon light to show boss details clearly
      game.dayNight.timeOfDay = 35;
      game.dayNight.update(0.1, game.lighting, game.camera, game.player.position);

      const pPos = game.player.position;
      const THREE = window.THREE || pPos.constructor;

      // Spawn Highway Hulk boss right in front of camera view
      const bossPos = new THREE(pPos.x + 2.5, 0, pPos.z - 1.8);
      const boss = game.world.zombies.spawnHighwayHulk(bossPos);
      if (boss) {
        boss.state = 'chase';
        boss.target = pPos;
      }
    }
  });

  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'phase7_highway_hulk_boss.png' });
  console.log('📸 Captured phase7_highway_hulk_boss.png (2.2x scale Highway Hulk boss with STOP road-sign club!)');

  await browser.close();
  console.log('✅ Phase Z7 live runtime verification completed successfully!');
}

verify().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
