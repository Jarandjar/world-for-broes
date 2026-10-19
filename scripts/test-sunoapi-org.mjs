const KEY = 'f38b6d5170e82b1804feb8f34a86160e';

try {
  const res = await fetch('https://api.sunoapi.org/api/v1/generate', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: 'A peaceful acoustic guitar melody, instrumental',
      customMode: false,
      instrumental: true,
      model: 'V4_5ALL'
    })
  });
  console.log('Status:', res.status);
  const data = await res.json();
  console.log('Response:', data);
} catch (e) {
  console.error('Error:', e);
}
