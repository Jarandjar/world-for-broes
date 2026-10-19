import { chromium } from 'playwright';

async function main() {
  console.log('====================================================');
  console.log('👁️  ANTIGRAVITY LIVE GAME OBSERVER LAUNCHING');
  console.log('====================================================');
  console.log('Launching visible desktop Chromium window...');

  const browser = await chromium.launch({
    headless: false, // Visible window on the user desktop!
    args: ['--window-size=1366,768'],
  });

  const context = await browser.newContext({ viewport: { width: 1366, height: 768 } });
  const page = await context.newPage();

  // Listen to browser console
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error(`[BROWSER ERROR] ${msg.text()}`);
    }
  });

  console.log('Navigating to http://localhost:5175/?autostart=1 ...');
  await page.goto('http://localhost:5175/?autostart=1', { waitUntil: 'domcontentloaded' });
  console.log('Game page loaded! Entering wilderness...');

  let lastState = null;
  let running = true;

  // Graceful shutdown on process exit
  process.on('SIGINT', async () => {
    running = false;
    await browser.close();
    process.exit(0);
  });

  // Watcher Loop
  const startTime = Date.now();
  console.log('\n👁️  OBSERVER ACTIVE: Streaming live telemetry. You can also play directly in the window!\n');

  while (running) {
    try {
      const state = await page.evaluate(() => {
        return (window).getGameState ? (window).getGameState() : null;
      });

      if (state) {
        const timeDiff = Math.round((Date.now() - startTime) / 1000);
        const phaseEmoji = state.phase === 'DAY' ? '☀️' : state.phase === 'DUSK' ? '🌆' : '🌙';
        const invSummary = state.inventory.length > 0 ? state.inventory.join(', ') : 'Empty';

        // Log regular telemetry every 2 seconds
        console.log(
          `[T+${timeDiff}s] ${phaseEmoji} ${state.phase} (Day ${state.day}) | ❤️ ${state.health}/100 | 🍗 ${state.hunger}/100 | 🧠 ${state.sanity}/100 | 📍 Pos: (${state.playerPos.x}, ${state.playerPos.z}) | 🎒 [${invSummary}]`
        );

        // Event Detection
        if (lastState) {
          if (state.phase !== lastState.phase) {
            console.log(`\n🔔 [PHASE SHIFT] Sun set to ${state.phase}! Beware of lethal darkness at night!\n`);
          }
          if (state.treesFelled > lastState.treesFelled) {
            console.log(`\n🪓 [ACTION EVENT] Tree felled! Total chopped: ${state.treesFelled}\n`);
          }
          if (state.health < lastState.health) {
            console.log(`\n⚠️  [DAMAGE TAKEN] Player took ${lastState.health - state.health} damage! Remaining: ${state.health}\n`);
          }
          if (state.isDead && !lastState.isDead) {
            console.log(`\n💀 [MORTALITY EVENT] Player perished! Game Over.\n`);
          }
        }
        lastState = state;
      }
    } catch {
      // If window closed
      break;
    }

    await new Promise(r => setTimeout(r, 2000));
  }
}

main().catch(err => {
  console.error('Observer error:', err);
});
