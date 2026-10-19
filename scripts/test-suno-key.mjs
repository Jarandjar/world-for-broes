const KEY = 'f38b6d5170e82b1804feb8f34a86160e';

const targets = [
  { name: 'apiframe.ai', url: 'https://api.apiframe.ai/profile', headers: { 'Authorization': KEY } },
  { name: 'apiframe.ai Bearer', url: 'https://api.apiframe.ai/profile', headers: { 'Authorization': `Bearer ${KEY}` } },
  { name: 'sunoapi.org generate', url: 'https://api.sunoapi.org/api/v1/gateway/generate', headers: { 'Authorization': `Bearer ${KEY}` } },
  { name: 'sunor.cc', url: 'https://api.sunor.cc/api/v1/user', headers: { 'Authorization': `Bearer ${KEY}` } },
  { name: 'acedata.cloud', url: 'https://api.acedata.cloud/suno/audios', headers: { 'authorization': `bearer ${KEY}` } },
];

for (const t of targets) {
  try {
    const res = await fetch(t.url, { headers: t.headers, signal: AbortSignal.timeout(4000) });
    const text = await res.text().catch(() => '');
    console.log(`${t.name} -> HTTP ${res.status}: ${text.slice(0, 150)}`);
  } catch (e) {
    console.log(`${t.name} -> error: ${e.message}`);
  }
}
