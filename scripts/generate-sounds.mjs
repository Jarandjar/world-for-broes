/**
 * Generates every game sound into public/audio/:
 *   - SFX (13 effects): ElevenLabs Sound Generation API
 *   - Music (2 loops): Suno API (api.sunoapi.org)
 *
 * Usage:  npm run sounds            (only generates missing files)
 *         npm run sounds -- --force (regenerates everything)
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'public', 'audio');
const FORCE = process.argv.includes('--force');

// Load .env
const envPath = path.join(ROOT, '.env');
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
}

const ELEVEN_KEY = process.env.ELEVENLABS_API_KEY;
const SUNO_KEY = process.env.SUNO_API_KEY;

export const SFX = {
  chop: { text: 'Single heavy flint axe strike into a pine tree trunk, crisp wood thunk with small splinter crack, cartoon game sound', duration: 0.6 },
  treefall: { text: 'Pine tree creaking, wood snapping, then tree crashing down onto forest floor with a deep thud, cartoon game sound', duration: 2.5 },
  pick: { text: 'Quick soft cartoon pop and leafy rustle, picking up an item in a video game', duration: 0.5 },
  footstep: { text: 'Single soft footstep on dry dirt and grass, close up, short', duration: 0.5 },
  eat: { text: 'Two quick crunchy cartoon bites chewing berries, comedic', duration: 0.8 },
  craft: { text: 'Whimsical magical crafting success chime with wooden knock, video game item crafted jingle', duration: 1.2 },
  charlie_warning: { text: 'Eerie creepy whisper in the darkness with low dissonant horror drone, something lurking', duration: 2.5 },
  charlie_slash: { text: 'Fast vicious shadow monster claw slash swoosh with sharp impact', duration: 0.8 },
  hurt: { text: 'Short cartoon character grunt of pain, oof, comedic game hurt sound', duration: 0.5 },
  honk: { text: 'Loud silly goose honk mixed with a bicycle horn, comedic', duration: 0.7 },
  pan: { text: 'Cast iron frying pan hitting something, loud resonant metallic dong, cartoon', duration: 1.2 },
  chicken: { text: 'Rubber chicken squeeze toy squeak, high pitched comedic', duration: 0.6 },
  bubble: { text: 'Tiny soft bubble pop, UI speech bubble appear sound', duration: 0.5 },
};

export const SUNO_MUSIC = {
  bgm_day: {
    prompt: 'Don\'t Starve style whimsical eerie acoustic guitar folk instrumental, banjo, clarinet, upright bass, playful gothic ragtime survival, instrumental',
  },
  bgm_night: {
    prompt: 'Dark ambient horror survival instrumental, creeping low cello drones, eerie wind chime, tense suspense minimal bass, gothic mystery, instrumental',
  },
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

function needs(name) {
  const file = path.join(OUT_DIR, `${name}.mp3`);
  return FORCE || !fs.existsSync(file) || fs.statSync(file).size < 1000;
}

export function missingSounds() {
  return [...Object.keys(SFX), ...Object.keys(SUNO_MUSIC)].filter(n => {
    const f = path.join(OUT_DIR, `${n}.mp3`);
    return !fs.existsSync(f) || fs.statSync(f).size < 1000;
  });
}

// 1. ElevenLabs SFX Fetcher
async function fetchElevenSFX(endpoint, body, attempt = 1) {
  const res = await fetch(`https://api.elevenlabs.io${endpoint}`, {
    method: 'POST',
    headers: { 'xi-api-key': ELEVEN_KEY, 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
    body: JSON.stringify(body),
  });
  if (res.status === 429 && attempt <= 4) {
    const wait = 3000 * attempt;
    console.log(`   ⏳ rate limited, retrying in ${wait / 1000}s...`);
    await sleep(wait);
    return fetchElevenSFX(endpoint, body, attempt + 1);
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${detail.slice(0, 300)}`);
  }
  return Buffer.from(await res.arrayBuffer());
}

// 2. Suno Music Generator & Poller
async function generateSunoMusic(prompt) {
  const res = await fetch('https://api.sunoapi.org/api/v1/generate', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${SUNO_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt,
      customMode: false,
      instrumental: true,
      model: 'V4_5ALL',
      callBackUrl: 'https://httpbin.org/post'
    })
  });
  const data = await res.json();
  if (data.code !== 200 || !data.data?.taskId) {
    throw new Error(`Suno generate failed: ${JSON.stringify(data)}`);
  }

  const taskId = data.data.taskId;
  console.log(`task ${taskId} queued, polling...`);

  // Poll until ready (max 180s)
  for (let i = 0; i < 36; i++) {
    await sleep(5000);
    const pollRes = await fetch(`https://api.sunoapi.org/api/v1/generate/record-info?taskId=${taskId}`, {
      headers: { 'Authorization': `Bearer ${SUNO_KEY}` }
    });
    const pollData = await pollRes.json();
    if (pollData.code === 200 && pollData.data?.status === 'SUCCESS') {
      const sunoList = pollData.data.response?.sunoData || [];
      const audioUrl = sunoList[0]?.audio_url || sunoList[0]?.audioUrl;
      if (!audioUrl) throw new Error('Suno returned success but no audio_url');

      const audioResp = await fetch(audioUrl);
      return Buffer.from(await audioResp.arrayBuffer());
    } else if (pollData.data?.status === 'FAILED') {
      throw new Error(`Suno generation failed: ${pollData.data?.errorMessage}`);
    }
    process.stdout.write('.');
  }
  throw new Error('Suno generation timed out after 3 minutes');
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  console.log('--- Generating Sound Effects (ElevenLabs) ---');
  for (const [name, cfg] of Object.entries(SFX)) {
    if (!needs(name)) { console.log(`✔  ${name}.mp3 (ready)`); continue; }
    process.stdout.write(`🔊 ${name}.mp3 ... `);
    try {
      const buf = await fetchElevenSFX('/v1/sound-generation?output_format=mp3_44100_128', {
        text: cfg.text,
        duration_seconds: cfg.duration,
        prompt_influence: 0.6,
      });
      fs.writeFileSync(path.join(OUT_DIR, `${name}.mp3`), buf);
      console.log(`done (${(buf.length / 1024).toFixed(0)} KB)`);
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
    }
  }

  console.log('\n--- Generating Background Soundtracks (Suno) ---');
  for (const [name, cfg] of Object.entries(SUNO_MUSIC)) {
    if (!needs(name)) { console.log(`✔  ${name}.mp3 (ready)`); continue; }
    process.stdout.write(`🎼 ${name}.mp3 (Suno AI) ... `);
    try {
      const buf = await generateSunoMusic(cfg.prompt);
      fs.writeFileSync(path.join(OUT_DIR, `${name}.mp3`), buf);
      console.log(` done (${(buf.length / 1024).toFixed(0)} KB)`);
    } catch (e) {
      console.log(`\n❌ Suno generation failed for ${name}: ${e.message}`);
    }
  }

  const missing = missingSounds();
  if (missing.length) {
    console.error(`\n❌ ${missing.length} sound(s) still missing: ${missing.join(', ')}`);
    process.exit(1);
  }
  console.log(`\n🎉 All sounds & soundtracks generated and ready in public/audio/!`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(e => { console.error(e); process.exit(1); });
}
