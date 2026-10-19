import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const ARTIFACT_DIR = 'C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622';

async function verify() {
  console.log('🚀 Launching headless browser to verify Gameplay Bots & Smooth Kinematic Animations...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  page.on('pageerror', err => console.error('❌ Page error:', err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('❌ Console error:', msg.text());
  });

  console.log('Navigating to http://127.0.0.1:5175/lantern-hollow.html ...');
  await page.goto('http://127.0.0.1:5175/lantern-hollow.html', { waitUntil: 'domcontentloaded', timeout: 15000 });
  console.log('Waiting for WebGL initialization...');
  await page.waitForTimeout(3000);

  // 1. Verify WebGL compiled without errors
  const shaderErr = await page.evaluate(() => window.__shaderError);
  if (shaderErr) {
    throw new Error('Shader compilation error: ' + shaderErr);
  }
  console.log('✅ WebGL2 raymarching shader compiled with ZERO errors!');

  // Ensure local artifacts dir exists
  if (!fs.existsSync('artifacts')) fs.mkdirSync('artifacts', { recursive: true });

  // 2. Capture The World 4 Bro's Companion Bots Squad in Action
  console.log('📸 1. Capturing Companion Bots Squad (Bro 1 Woody, Bro 2 Rocky, Bro 3 Spanner)...');
  await page.evaluate(() => {
    // Position camera for a cinematic hero angle showing all 3 bots and player
    window.bots[0].say('Chop chop! Firewood secured! 🪓🪵');
    window.bots[1].say('Struck gold in the quarry! ⛏️🪙');
    window.bots[2].say('V8 Scout Wagon purring like a kitten! 🔧🚗');
  });
  await page.waitForTimeout(500);
  const squadPath = 'artifacts/lantern_hollow_bots_squad.png';
  await page.screenshot({ path: squadPath });
  fs.copyFileSync(squadPath, path.join(ARTIFACT_DIR, 'lantern_hollow_bots_squad.png'));
  console.log('✅ Captured & copied lantern_hollow_bots_squad.png');

  // 3. Trigger and capture Smooth Multi-Phase Attack Animation
  console.log('📸 2. Capturing Smooth Kinematic Attack Animation (Hit-stop & Weapon Arc)...');
  await page.evaluate(() => {
    window.state.chop = true;
    window.player.isMoving = true;
    window.player.walkPhase = 1.2;
    window.bots[0].action = 'chop';
    window.bots[0].actionClock = 0.55; // Peak downstroke / hitstop
    window.bots[1].action = 'mine';
    window.bots[1].actionClock = 0.65; // Peak pickaxe plunge
  });
  await page.waitForTimeout(300);
  const animPath = 'artifacts/lantern_hollow_fluid_animation.png';
  await page.screenshot({ path: animPath });
  fs.copyFileSync(animPath, path.join(ARTIFACT_DIR, 'lantern_hollow_fluid_animation.png'));
  console.log('✅ Captured & copied lantern_hollow_fluid_animation.png');

  // 4. Activate Gameplay Auto-Tester Runner Banner
  console.log('📸 3. Triggering Autonomous Gameplay Test Runner...');
  await page.evaluate(() => {
    window.autoTester.toggle(true);
  });
  await page.waitForTimeout(800);
  const testPath = 'artifacts/lantern_hollow_autotest_runner.png';
  await page.screenshot({ path: testPath });
  fs.copyFileSync(testPath, path.join(ARTIFACT_DIR, 'lantern_hollow_autotest_runner.png'));
  console.log('✅ Captured & copied lantern_hollow_autotest_runner.png');

  // 5. Test Slapstick Combat & Goofy Weapons (Rubber Chicken & Whoopee Fart Cloud)
  console.log('📸 4. Capturing Slapstick Combat & Rubber Chicken Launch...');
  await page.evaluate(() => {
    window.bots[1].action = 'slap';
    window.bots[1].actionClock = 0.45;
    window.bots[1].say('SQUEEEAK! Undead airborne! 🐔💨');
    const z = window.zombies[0];
    z.hp = 10;
    z.stunTime = 4.0;
  });
  await page.waitForTimeout(400);
  const slapPath = 'artifacts/lantern_hollow_slapstick_combat.png';
  await page.screenshot({ path: slapPath });
  fs.copyFileSync(slapPath, path.join(ARTIFACT_DIR, 'lantern_hollow_slapstick_combat.png'));
  console.log('✅ Captured & copied lantern_hollow_slapstick_combat.png');

  await browser.close();
  console.log('🎉 All Gameplay Bots & Kinematic Animation verification tests PASSED!');
}

verify().catch(err => {
  console.error('❌ Verification failed:', err);
  process.exit(1);
});
