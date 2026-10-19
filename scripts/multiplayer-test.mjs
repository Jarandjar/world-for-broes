import { chromium } from 'playwright';

async function testMultiplayer() {
  console.log('Testing multiplayer connection...');
  // Launch Player 2
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  console.log('Player 2 navigating to http://localhost:5173/?autostart=1 ...');
  await page.goto('http://localhost:5173/?autostart=1', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1000);

  // Player 2 walks to (2, 2)
  console.log('Player 2 moving into view...');
  await page.keyboard.down('KeyW');
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(600);
  await page.keyboard.up('KeyW');
  await page.keyboard.up('KeyD');

  await page.waitForTimeout(1000);

  console.log('Capturing multiplayer screenshot from Player 2 perspective...');
  await page.screenshot({ path: 'C:/Users/Jalle/.gemini/antigravity-ide/brain/c0eb8496-fde1-432e-9e30-52902141b27f/multiplayer_screenshot.png' });

  await browser.close();
  console.log('Multiplayer test finished!');
}

testMultiplayer().catch(err => console.error('MP test error:', err));
