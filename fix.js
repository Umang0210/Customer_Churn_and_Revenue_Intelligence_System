const fs = require('fs');
let html = fs.readFileSync('public/index.html', 'utf8');

// Find all fetchWithTimeout functions and remove them.
const regex = /function fetchWithTimeout\([\s\S]*?\n\}\n/g;
html = html.replace(regex, '');

const correctFetch = `
function fetchWithTimeout(url, options = {}) {
  const { timeout = 15000, ...fetchOptions } = options;
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort();
      reject(new Error('Request Timeout'));
    }, timeout);
    fetchOptions.signal = controller.signal;
    fetch(url, fetchOptions)
      .then(response => { clearTimeout(timer); resolve(response); })
      .catch(err => { clearTimeout(timer); reject(err); });
  });
}
`;

html = html.replace('const API_BASE = window.location.origin;', 'const API_BASE = window.location.origin;\n' + correctFetch);

fs.writeFileSync('public/index.html', html);
console.log('Purged and fixed fetchWithTimeout');
