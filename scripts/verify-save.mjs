import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
const page = await ctx.newPage();

await page.goto('http://localhost:5175/?autostart=1');
await page.waitForTimeout(1200);
// Walk right for a bit so position changes
await page.keyboard.down('d');
await page.waitForTimeout(1500);
await page.keyboard.up('d');
const before = await page.evaluate(() => window.getGameState());
await page.waitForTimeout(5500); // let autosave fire
const saved = await page.evaluate(() => localStorage.getItem('wild-survivors-save-v1'));

await page.reload();
await page.waitForTimeout(2000);
const after = await page.evaluate(() => window.getGameState());

console.log('before reload :', before.playerPos, before.inventory);
console.log('save present  :', !!saved);
console.log('after reload  :', after.playerPos, after.inventory);
const dist = Math.hypot(after.playerPos.x - before.playerPos.x, after.playerPos.z - before.playerPos.z);
console.log(dist < 0.5 ? '✅ position restored' : `❌ position drift ${dist.toFixed(2)}m`);
await browser.close();
