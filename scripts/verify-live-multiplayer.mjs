import { chromium } from 'playwright';

async function main() {
  console.log('🌐 Testing Live Vercel Multiplayer over public internet...');
  const browser = await chromium.launch({ headless: true });

  // Client 1 (PC on Vercel)
  const ctx1 = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const p1 = await ctx1.newPage();
  console.log('Connecting Client 1 to https://game-six-snowy.vercel.app ...');
  await p1.goto('https://game-six-snowy.vercel.app/?autostart=1');
  await p1.waitForTimeout(3000);

  // Client 2 (Phone on Vercel)
  const ctx2 = await browser.newContext({ viewport: { width: 844, height: 390 }, isMobile: true, hasTouch: true });
  const p2 = await ctx2.newPage();
  console.log('Connecting Client 2 (mobile) to https://game-six-snowy.vercel.app ...');
  await p2.goto('https://game-six-snowy.vercel.app/?autostart=1');
  await p2.waitForTimeout(3500);

  // Check online badge
  const badge1 = await p1.$eval('#mp-badge', el => el ? el.innerText : 'none').catch(() => 'none');
  const badge2 = await p2.$eval('#mp-badge', el => el ? el.innerText : 'none').catch(() => 'none');
  console.log('Client 1 Multiplayer Status:', badge1);
  console.log('Client 2 Multiplayer Status:', badge2);

  // Trigger emotes & speech bubbles
  await p1.keyboard.press('g'); // Dance
  await p2.waitForTimeout(500);
  const touchHonk = await p2.$('#btn-touch-honk');
  if (touchHonk) await touchHonk.click();
  await p1.waitForTimeout(1000);

  const artifactDir = 'C:/Users/Jalle/.gemini/antigravity-ide/brain/53496f9c-bc62-4e66-b6a9-6cbbd3275622';
  await p1.screenshot({ path: `${artifactDir}/live_vercel_multiplayer_client1.png` });
  await p2.screenshot({ path: `${artifactDir}/live_vercel_multiplayer_client2.png` });

  console.log('📸 Live public multiplayer screenshots saved!');
  await browser.close();
}

main().catch(e => {
  console.error('Test error:', e);
  process.exit(1);
});
