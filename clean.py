import codecs
import re

with codecs.open('public/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Completely remove all existing fetchWithTimeout declarations to clean the file
html = re.sub(r'function fetchWithTimeout\([\s\S]*?\n\}\n', '', html)
html = html.replace("reject(new Error(\\'Request Timeout\\'));", '')
html = html.replace("reject(new Error(\\'Request Timeout\'));", '')

# There might be some left over brackets if the regex missed the end
# Let's just use string slicing to cleanly extract the parts we know are good.
parts = html.split('const API_BASE = window.location.origin;')

if len(parts) == 2:
    good_top = parts[0]
    rest = parts[1]
    
    # find where the real boot script starts
    boot_idx = rest.find('if (!AbortSignal.timeout)')
    if boot_idx != -1:
        clean_rest = rest[boot_idx:]
    else:
        clean_rest = rest

    correctFetch = """
const API_BASE = window.location.origin;

function fetchWithTimeout(url, options = {}) {
  const { timeout = 15000, ...fetchOptions } = options;
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort();
      reject(new Error("Request Timeout"));
    }, timeout);
    fetchOptions.signal = controller.signal;
    fetch(url, fetchOptions)
      .then(response => { clearTimeout(timer); resolve(response); })
      .catch(err => { clearTimeout(timer); reject(err); });
  });
}

"""
    html = good_top + correctFetch + clean_rest

with codecs.open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Cleaned index.html')
