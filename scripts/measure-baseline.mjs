import { chromium } from 'playwright';

async function measure() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  const logs = [];
  page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));
  page.on('pageerror', err => logs.push(`[PAGE ERROR] ${err.message}`));

  console.log('Navigating to http://localhost:5175/?autostart=1...');
  await page.goto('http://localhost:5175/?autostart=1');

  // Wait for canvas to load
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForTimeout(2000);

  // Measure performance, draw calls, triangles
  const perfData = await page.evaluate(async () => {
    return new Promise((resolve) => {
      let frames = 0;
      let lastTime = performance.now();
      const frameTimes = [];

      function sample() {
        const now = performance.now();
        const dt = now - lastTime;
        lastTime = now;
        frameTimes.push(dt);
        frames++;

        if (frames < 120) {
          requestAnimationFrame(sample);
        } else {
          // Calculate stats
          const avgDt = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
          const maxDt = Math.max(...frameTimes);
          const minDt = Math.min(...frameTimes);
          const fps = 1000 / avgDt;

          // Check Three.js renderer stats if available
          const canvas = document.querySelector('canvas');
          let drawCalls = 'N/A';
          let triangles = 'N/A';
          if (window.getGameState) {
            // we have gameState
          }

          resolve({
            avgDt: avgDt.toFixed(2),
            maxDt: maxDt.toFixed(2),
            minDt: minDt.toFixed(2),
            fps: fps.toFixed(1),
            gameState: window.getGameState ? window.getGameState() : null,
          });
        }
      }
      requestAnimationFrame(sample);
    });
  });

  console.log('Performance Sample (120 frames):', JSON.stringify(perfData, null, 2));

  // Take baseline screenshot: Day
  await page.screenshot({ path: 'baseline_day.png' });
  console.log('Saved baseline_day.png');

  // Walk around to test camera & movement
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(1000);
  await page.keyboard.up('KeyD');
  await page.screenshot({ path: 'baseline_walk.png' });
  console.log('Saved baseline_walk.png');

  // Rotate camera
  await page.keyboard.press('KeyE');
  await page.waitForTimeout(600);
  await page.screenshot({ path: 'baseline_rotated.png' });
  console.log('Saved baseline_rotated.png');

  console.log('Console logs during test:', logs.slice(0, 15));

  await browser.close();
}

measure().catch(err => {
  console.error('Measurement failed:', err);
  process.exit(1);
});
