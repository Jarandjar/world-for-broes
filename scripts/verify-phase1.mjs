import { chromium } from 'playwright';

async function verify() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  console.log('Navigating to http://localhost:5175/?autostart=1...');
  await page.goto('http://localhost:5175/?autostart=1');
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForTimeout(1500);

  // Take screenshot: Phase 1 Resting Day
  await page.screenshot({ path: 'phase1_smooth_day.png' });
  console.log('Captured phase1_smooth_day.png');

  // Test acceleration, movement, and camera follow
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(600);
  await page.keyboard.up('KeyD');

  // Test reverse facing direction for ~80ms squash flip
  await page.keyboard.down('KeyA');
  await page.waitForTimeout(150); // Mid-turn
  await page.screenshot({ path: 'phase1_turn_squash.png' });
  await page.waitForTimeout(450);
  await page.keyboard.up('KeyA');

  // Test smooth camera orbit easing
  await page.keyboard.press('KeyE');
  await page.waitForTimeout(400); // Mid-orbit easing
  await page.screenshot({ path: 'phase1_camera_orbit.png' });

  console.log('Errors observed during session:', errors);

  await browser.close();
}

verify().catch(err => {
  console.error('Phase 1 test error:', err);
  process.exit(1);
});
