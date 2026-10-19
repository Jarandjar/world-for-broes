import { chromium } from 'playwright';
import * as fs from 'fs';
import * as path from 'path';

const survivorSrc = 'C:/Users/Jalle/.gemini/antigravity-ide/brain/53496f9c-bc62-4e66-b6a9-6cbbd3275622/survivor_on_white_1792359416738.jpg';
const zombieSrc = 'C:/Users/Jalle/.gemini/antigravity-ide/brain/53496f9c-bc62-4e66-b6a9-6cbbd3275622/zombie_on_white_1792359433944.jpg';
const outDir = 'c:/Users/Jalle/OneDrive/Desktop/BUCKS/game/public/textures/noir';

async function processSprites() {
  console.log('🖼️ Processing Noir Sprites, Normal Maps & Emissive Masks...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Convert files to base64 data URLs
  const survivorBase64 = `data:image/jpeg;base64,${fs.readFileSync(survivorSrc).toString('base64')}`;
  const zombieBase64 = `data:image/jpeg;base64,${fs.readFileSync(zombieSrc).toString('base64')}`;

  const results = await page.evaluate(async ({ survivorBase64, zombieBase64 }) => {
    function loadImage(url) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
      });
    }

    // Helper: Remove pure white background cleanly starting from borders
    function removeDarkBackground(ctx, width, height, tolerance = 25) {
      const imgData = ctx.getImageData(0, 0, width, height);
      const data = imgData.data;

      const visited = new Uint8Array(width * height);
      const queue = new Int32Array(width * height);
      let head = 0, tail = 0;

      const isWhiteBg = (x, y) => {
        const idx = (y * width + x) * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        return r > 240 && g > 240 && b > 240;
      };

      // Seed all border pixels that are white
      for (let x = 0; x < width; x++) {
        if (isWhiteBg(x, 0)) { visited[x] = 1; queue[tail++] = x; }
        if (isWhiteBg(x, height - 1)) { visited[(height - 1) * width + x] = 1; queue[tail++] = (height - 1) * width + x; }
      }
      for (let y = 1; y < height - 1; y++) {
        if (isWhiteBg(0, y)) { visited[y * width] = 1; queue[tail++] = y * width; }
        if (isWhiteBg(width - 1, y)) { visited[y * width + (width - 1)] = 1; queue[tail++] = y * width + (width - 1); }
      }

      // Flood fill outer white region
      while (head < tail) {
        const curr = queue[head++];
        const cx = curr % width;
        const cy = (curr / width) | 0;

        // Make pixel completely transparent
        data[curr * 4 + 3] = 0;

        const neighbors = [
          [cx + 1, cy],
          [cx - 1, cy],
          [cx, cy + 1],
          [cx, cy - 1]
        ];

        for (let i = 0; i < 4; i++) {
          const nx = neighbors[i][0];
          const ny = neighbors[i][1];
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nIdx = ny * width + nx;
            if (!visited[nIdx] && isWhiteBg(nx, ny)) {
              visited[nIdx] = 1;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      // Feather semi-white halo fringe
      for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] > 0) {
          const r = data[i], g = data[i + 1], b = data[i + 2];
          if (r > 245 && g > 245 && b > 245) {
            data[i + 3] = 0;
          }
        }
      }

      ctx.putImageData(imgData, 0, 0);
      return imgData;
    }

    // Helper: Generate Sobel Normal Map
    function generateNormalMap(srcCtx, width, height, bumpStrength = 2.8) {
      const srcData = srcCtx.getImageData(0, 0, width, height).data;
      const normalCanvas = document.createElement('canvas');
      normalCanvas.width = width;
      normalCanvas.height = height;
      const nCtx = normalCanvas.getContext('2d');
      const nImgData = nCtx.createImageData(width, height);
      const nData = nImgData.data;

      const getLuma = (x, y) => {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        const idx = (y * width + x) * 4;
        if (srcData[idx + 3] < 30) return 0;
        return (0.299 * srcData[idx] + 0.587 * srcData[idx + 1] + 0.114 * srcData[idx + 2]) / 255;
      };

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          if (srcData[idx + 3] < 20) {
            nData[idx] = 128;
            nData[idx + 1] = 128;
            nData[idx + 2] = 255; // Neutral flat normal pointing Z
            nData[idx + 3] = 0;
            continue;
          }

          // Sobel operator
          const tl = getLuma(x - 1, y - 1);
          const t  = getLuma(x,     y - 1);
          const tr = getLuma(x + 1, y - 1);
          const l  = getLuma(x - 1, y);
          const r  = getLuma(x + 1, y);
          const bl = getLuma(x - 1, y + 1);
          const b  = getLuma(x,     y + 1);
          const br = getLuma(x + 1, y + 1);

          const dx = (tr + 2 * r + br) - (tl + 2 * l + bl);
          const dy = (bl + 2 * b + br) - (tl + 2 * t + tr);

          const nx = -dx * bumpStrength;
          const ny = dy * bumpStrength; // OpenGL / Three.js Y+
          const nz = 1.0;

          const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
          nData[idx]     = Math.round(((nx / len) * 0.5 + 0.5) * 255);
          nData[idx + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
          nData[idx + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
          nData[idx + 3] = srcData[idx + 3];
        }
      }
      nCtx.putImageData(nImgData, 0, 0);
      return normalCanvas.toDataURL('image/png');
    }

    // Helper: Extract Emissive Mask for Glowing Eyes
    function extractEmissiveEyes(srcCtx, width, height) {
      const srcData = srcCtx.getImageData(0, 0, width, height).data;
      const emissiveCanvas = document.createElement('canvas');
      emissiveCanvas.width = width;
      emissiveCanvas.height = height;
      const eCtx = emissiveCanvas.getContext('2d');
      const eImgData = eCtx.createImageData(width, height);
      const eData = eImgData.data;

      for (let i = 0; i < srcData.length; i += 4) {
        const r = srcData[i], g = srcData[i + 1], b = srcData[i + 2], a = srcData[i + 3];
        // Eyes are bright yellow/pale white (R > 180, G > 180, B > 120)
        if (a > 100 && r > 175 && g > 175 && b < 240) {
          eData[i]     = 255;
          eData[i + 1] = 245;
          eData[i + 2] = 160; // Bright pale yellow glow
          eData[i + 3] = 255;
        } else {
          eData[i] = 0; eData[i + 1] = 0; eData[i + 2] = 0; eData[i + 3] = 0;
        }
      }
      eCtx.putImageData(eImgData, 0, 0);
      return emissiveCanvas.toDataURL('image/png');
    }

    // 1. Process Survivor
    const sImg = await loadImage(survivorBase64);
    const sCanvas = document.createElement('canvas');
    sCanvas.width = sImg.width;
    sCanvas.height = sImg.height;
    const sCtx = sCanvas.getContext('2d');
    sCtx.drawImage(sImg, 0, 0);
    removeDarkBackground(sCtx, sImg.width, sImg.height, 42);
    const survivorDiffuse = sCanvas.toDataURL('image/png');
    const survivorNormal = generateNormalMap(sCtx, sImg.width, sImg.height, 2.5);

    // 2. Process Zombie Shambler
    const zImg = await loadImage(zombieBase64);
    const zCanvas = document.createElement('canvas');
    zCanvas.width = zImg.width;
    zCanvas.height = zImg.height;
    const zCtx = zCanvas.getContext('2d');
    zCtx.drawImage(zImg, 0, 0);
    removeDarkBackground(zCtx, zImg.width, zImg.height, 50);
    const zombieDiffuse = zCanvas.toDataURL('image/png');
    const zombieNormal = generateNormalMap(zCtx, zImg.width, zImg.height, 3.2);
    const zombieEmissive = extractEmissiveEyes(zCtx, zImg.width, zImg.height);

    return {
      survivorDiffuse,
      survivorNormal,
      zombieDiffuse,
      zombieNormal,
      zombieEmissive,
    };
  }, { survivorBase64, zombieBase64 });

  await browser.close();

  // Save to public/textures/noir
  function saveBase64(dataUrl, filename) {
    const base64Data = dataUrl.replace(/^data:image\/png;base64,/, '');
    const target = path.join(outDir, filename);
    fs.writeFileSync(target, Buffer.from(base64Data, 'base64'));
    console.log(`✅ Saved ${filename} (${(fs.statSync(target).size / 1024).toFixed(1)} KB)`);
  }

  saveBase64(results.survivorDiffuse, 'survivor_diffuse.png');
  saveBase64(results.survivorNormal, 'survivor_normal.png');
  saveBase64(results.zombieDiffuse, 'shambler_diffuse.png');
  saveBase64(results.zombieNormal, 'shambler_normal.png');
  saveBase64(results.zombieEmissive, 'shambler_emissive.png');

  console.log('🎉 All Noir sprite maps and textures processed successfully!');
}

processSprites().catch(err => {
  console.error('❌ Processing failed:', err);
  process.exit(1);
});
