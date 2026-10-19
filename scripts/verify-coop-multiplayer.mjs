import { chromium, devices } from 'playwright';

async function main() {
  console.log('🚀 Launching Cross-Platform Co-Op Test (PC + Mobile Phone)...');
  const browser = await chromium.launch({ headless: true });

  // 1. Client 1: PC Desktop Host
  const pcContext = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const pcPage = await pcContext.newPage();
  console.log('PC Host connecting...');
  await pcPage.goto('http://localhost:5175/?autostart=1');
  await pcPage.waitForTimeout(2000);

  // 2. Client 2: Mobile Phone Survivor (iPhone 14 Landscape)
  const phoneContext = await browser.newContext({
    viewport: { width: 844, height: 390 },
    isMobile: true,
    hasTouch: true,
  });
  const phonePage = await phoneContext.newPage();
  console.log('Phone Survivor connecting from mobile browser...');
  await phonePage.goto('http://localhost:5175/?autostart=1');
  await phonePage.waitForTimeout(2000);

  // Phone player honks horn and dances
  console.log('Phone player triggering emotes via mobile touch buttons...');
  const phoneHonk = await phonePage.$('#btn-touch-honk');
  if (phoneHonk) await phoneHonk.click();
  await phonePage.waitForTimeout(400);

  const phoneDance = await phonePage.$('#btn-touch-dance');
  if (phoneDance) await phoneDance.click();
  await phonePage.waitForTimeout(800);

  const artifactDir = 'C:/Users/Jalle/.gemini/antigravity-ide/brain/53496f9c-bc62-4e66-b6a9-6cbbd3275622';
  await pcPage.screenshot({ path: `${artifactDir}/coop_pc_viewing_phone_player.png` });
  await phonePage.screenshot({ path: `${artifactDir}/coop_phone_screen_in_action.png` });

  console.log('✅ Co-Op PC & Phone screenshots saved!');
  await browser.close();
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
