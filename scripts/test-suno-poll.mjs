const KEY = 'f38b6d5170e82b1804feb8f34a86160e';
const taskId = 'b40977a381a2f78f0feaa55de9da5bcb';

const endpoints = [
  `https://api.sunoapi.org/api/v1/task?taskId=${taskId}`,
  `https://api.sunoapi.org/api/v1/query?taskId=${taskId}`,
  `https://api.sunoapi.org/api/v1/generate/record-info?taskId=${taskId}`,
  `https://api.sunoapi.org/api/v1/task/${taskId}`,
  `https://api.sunoapi.org/api/v1/feed/${taskId}`,
  `https://api.sunoapi.org/api/v1/audios?taskId=${taskId}`
];

for (const ep of endpoints) {
  try {
    const res = await fetch(ep, { headers: { 'Authorization': `Bearer ${KEY}` } });
    const text = await res.text();
    console.log(`${ep} -> HTTP ${res.status}: ${text.slice(0, 150)}`);
  } catch (e) {
    console.log(`${ep} -> Error: ${e.message}`);
  }
}
