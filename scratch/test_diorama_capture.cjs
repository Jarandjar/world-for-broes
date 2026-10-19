const { spawn } = require('child_process');
const fs = require('fs');

async function run() {
  console.log('Launching headless Chrome for Phase Z0D Style & Vertical Slice Verification...');
  const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', [
    '--headless=new',
    '--remote-debugging-port=9333',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1280,720',
    'http://localhost:5175/?slice=1&autostart=1'
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

    console.log('Waiting 4s for assets and vertical slice scene initialization...');
    await new Promise(r => setTimeout(r, 4000));

    // Verify Vertical Slice is active
    const sliceCheck = await send('Runtime.evaluate', {
      expression: `(() => {
        if (window.__gameApp && window.__gameApp.verticalSlice) {
          if (!window.__gameApp.verticalSlice.isActive) {
            window.__gameApp.verticalSlice.activate();
          }
          return {
            active: window.__gameApp.verticalSlice.isActive,
            quality: window.__gameApp.verticalSlice.isActive
          };
        }
        return 'no-app';
      })()`,
      returnByValue: true
    });
    console.log('Vertical slice status:', sliceCheck.result.value);

    await new Promise(r => setTimeout(r, 500));

    // 1. Capture Dusk Vertical Slice Clearing
    console.log('Capturing Dusk Vertical Slice clearing...');
    const shot1 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase0d_vertical_slice_dusk.png', Buffer.from(shot1.data, 'base64'));
    console.log('Saved phase0d_vertical_slice_dusk.png!');

    // 2. Trigger Zombie Smack and capture "BONK!" comic word pop
    console.log('Triggering slapstick zombie smack ("BONK!")...');
    await send('Runtime.evaluate', {
      expression: `(() => {
        if (window.__gameApp && window.__gameApp.verticalSlice) {
          window.__gameApp.verticalSlice.triggerZombieSmack();
        }
      })()`
    });

    await new Promise(r => setTimeout(r, 150));

    console.log('Capturing "BONK!" comic onomatopoeia pop...');
    const shot2 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase0d_slapstick_bonk.png', Buffer.from(shot2.data, 'base64'));
    console.log('Saved phase0d_slapstick_bonk.png!');

    // 3. Open Animation Lab with Springs & Pose Library
    console.log('Opening Animation Lab on Springs & Pose tab...');
    await send('Runtime.evaluate', {
      expression: `(() => {
        if (window.__gameApp && window.__gameApp.animationLab) {
          window.__gameApp.animationLab.open();
          document.getElementById('tab-mode-springs')?.click();
        }
      })()`
    });

    await new Promise(r => setTimeout(r, 400));

    console.log('Capturing Animation Lab springs & pose library...');
    const shot3 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase0d_anim_lab_springs.png', Buffer.from(shot3.data, 'base64'));
    console.log('Saved phase0d_anim_lab_springs.png!');

    ws.close();
    chrome.kill();
    console.log('Done capturing Phase Z0D screenshots!');
  };
}

run();
