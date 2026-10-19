const { spawn } = require('child_process');
const fs = require('fs');

async function run() {
  console.log('Launching headless Chrome for Phase Z3 Building & Hiding Spot Verification...');
  const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', [
    '--headless=new',
    '--remote-debugging-port=9333',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1280,720',
    'http://localhost:5175/?mode=zombie&autostart=1'
  ]);

  await new Promise(r => setTimeout(r, 2000));

  let versionData;
  try {
    const res = await fetch('http://localhost:9333/json/list');
    versionData = await res.json();
  } catch (err) {
    console.error('Failed to connect to Chrome debug port:', err);
    chrome.kill();
    return;
  }

  const pageTarget = versionData.find(t => t.type === 'page');
  if (!pageTarget) {
    console.error('No page target found');
    chrome.kill();
    return;
  }

  const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

  let msgId = 1;
  const send = (method, params = {}) => {
    return new Promise((resolve) => {
      const id = msgId++;
      const handler = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.id === id) {
          ws.removeEventListener('message', handler);
          resolve(msg.result);
        }
      };
      ws.addEventListener('message', handler);
      ws.send(JSON.stringify({ id, method, params }));
    });
  };

  ws.onopen = async () => {
    await send('Runtime.enable');
    await send('Page.enable');

    console.log('Waiting 4s for assets and world generation...');
    await new Promise(r => setTimeout(r, 4000));

    // Teleport player near town hall doorway and walk inside
    console.log('Teleporting player inside Bro-ville Town Hall...');
    const insideRes = await send('Runtime.evaluate', {
      expression: `(() => {
        const app = window.app || (window as any).app;
        // Access GameApp via global or window
        // Let's find player position from getGameState and move player
        if (window.getGameState) {
          // Find game instance
          const canvas = document.querySelector('canvas');
          return window.getGameState();
        }
        return 'no-state';
      })()`,
      returnByValue: true
    });
    console.log('Initial Game State:', insideRes.result.value);

    // Now teleport player directly inside the town hall
    await send('Runtime.evaluate', {
      expression: `(() => {
        // We can dispatch keys or set player position
        // In GameApp, window.__gameApp or finding player
        // Let's check window objects
        for (const k of Object.keys(window)) {
          if (window[k] && window[k].player && window[k].world) {
            window.__gameApp = window[k];
            break;
          }
        }
      })()`
    });

    // Let's teleport player inside town hall (-14, 0, -14)
    const tpRes = await send('Runtime.evaluate', {
      expression: `(() => {
        if (window.__gameApp && window.__gameApp.player) {
          window.__gameApp.player.position.set(-14, 0, -14);
          return { success: true, pos: window.__gameApp.player.position, bldg: window.__gameApp.world.buildings.currentBuilding?.lot.buildingType };
        }
        return { success: false };
      })()`,
      returnByValue: true
    });
    console.log('Teleport result:', tpRes.result.value);

    // Wait 500ms for roof fold tween to smoothly fold upright
    await new Promise(r => setTimeout(r, 600));

    const stateInside = await send('Runtime.evaluate', {
      expression: `window.getGameState()`,
      returnByValue: true
    });
    console.log('Game State inside Town Hall:', stateInside.result.value);

    console.log('Capturing pop-up roof interior fold screenshot...');
    const interiorShot = await send('Page.captureScreenshot', { format: 'png' });
    const buffer1 = Buffer.from(interiorShot.data, 'base64');
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase3_popup_roof_interior.png', buffer1);
    console.log('Saved phase3_popup_roof_interior.png!');

    // Now teleport player near dumpster outside Town Hall or Diner
    console.log('Teleporting near Town Hall dumpster...');
    await send('Runtime.evaluate', {
      expression: `(() => {
        if (window.__gameApp && window.__gameApp.player) {
          // Town hall is at (-14, -14). Dumpster is at x = -14 + 14*0.5 + 1.2 = -5.8, z = -14 - 12*0.5 + 2.5 = -17.5
          window.__gameApp.player.position.set(-5.6, 0, -17.5);
          // Trigger hide action by pressing KeyR
          window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyR', bubbles: true }));
        }
      })()`
    });

    await new Promise(r => setTimeout(r, 400));

    const stateHiding = await send('Runtime.evaluate', {
      expression: `window.getGameState()`,
      returnByValue: true
    });
    console.log('Game State while Hiding:', stateHiding.result.value);

    console.log('Capturing dumpster hiding spot screenshot...');
    const hidingShot = await send('Page.captureScreenshot', { format: 'png' });
    const buffer2 = Buffer.from(hidingShot.data, 'base64');
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase3_hiding_spot_dumpster.png', buffer2);
    console.log('Saved phase3_hiding_spot_dumpster.png!');

    ws.close();
    chrome.kill();
    console.log('Done capturing live verification screenshots!');
  };
}

run();
