const KEY = 'f38b6d5170e82b1804feb8f34a86160e';

const res = await fetch('https://api.sunoapi.org/api/v1/generate', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    prompt: 'Whimsical eerie acoustic guitar folk instrumental',
    customMode: false,
    instrumental: true,
    model: 'V4_5ALL',
    callBackUrl: 'https://httpbin.org/post'
  })
});

console.log('Status:', res.status);
const data = await res.json();
console.log('Data:', JSON.stringify(data, null, 2));
