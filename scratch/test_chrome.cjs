const { spawn } = require('child_process');
const http = require('http');

async function run() {
  console.log('Launching headless Chrome...');
  const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', [
    '--headless=new',
    '--remote-debugging-port=9333',
    '--disable-gpu',
    '--no-sandbox',
    'http://localhost:5175/'
  ]);

  // Wait 2s for Chrome to bind port
  await new Promise(r => setTimeout(r, 2000));

  let versionData;
  try {
    const res = await fetch('http://localhost:9333/json/list');
    versionData = await res.json();
    console.log('Targets:', versionData);
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

  console.log('Connecting to WebSocket:', pageTarget.webSocketDebuggerUrl);
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
    console.log('Connected to CDP!');
    // Enable Console, Runtime, Page
    await send('Console.enable');
    await send('Runtime.enable');
    await send('Page.enable');

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.method === 'Runtime.consoleAPICalled') {
        console.log(`[BROWSER CONSOLE ${msg.params.type.toUpperCase()}]:`, msg.params.args.map(a => a.value || a.description || JSON.stringify(a)).join(' '));
      } else if (msg.method === 'Runtime.exceptionThrown') {
        console.error('[BROWSER UNCAUGHT EXCEPTION]:', JSON.stringify(msg.params.exceptionDetails, null, 2));
      } else if (msg.method === 'Log.entryAdded') {
        console.log('[BROWSER LOG]:', msg.params.entry);
      }
    };

    console.log('Navigating to http://localhost:5175/?mode=zombie&autostart=1 ...');
    await send('Page.navigate', { url: 'http://localhost:5175/?mode=zombie&autostart=1' });

    // Wait 4 seconds for page load, module execution, and world generation
    await new Promise(r => setTimeout(r, 4000));

    // Evaluate state
    const evalRes = await send('Runtime.evaluate', {
      expression: `({
        readyState: document.readyState,
        hasGameApp: typeof window.getGameState === 'function',
        isStarted: window.getGameState ? window.getGameState() : 'NO_STATE'
      })`,
      returnByValue: true
    });
    console.log('ZOMBIE MODE EVAL RESULT:', JSON.stringify(evalRes, null, 2));

    console.log('Capturing screenshot of settlements in zombie mode...');
    const fs = require('fs');
    const screenshot = await send('Page.captureScreenshot', { format: 'png' });
    const buffer = Buffer.from(screenshot.data, 'base64');
    fs.writeFileSync('C:\\Users\\Jalle\\.gemini\\antigravity-ide\\brain\\53496f9c-bc62-4e66-b6a9-6cbbd3275622\\phase2_settlements_town.png', buffer);
    console.log('Screenshot saved to phase2_settlements_town.png!');

    ws.close();
    chrome.kill();
  };
}

run().catch(console.error);
