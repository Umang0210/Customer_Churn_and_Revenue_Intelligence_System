import codecs

with codecs.open('public/index_restored.html', 'r', encoding='utf-16le') as f:
    content = f.read()

# Replace AbortSignal.timeout polyfill with fetchWithTimeout function
polyfill = """if (!AbortSignal.timeout) {
      AbortSignal.timeout = function (ms) {
        const controller = new AbortController();
        setTimeout(() => controller.abort(), ms);
        return controller.signal;
      };
    }"""

fetch_with_timeout = """function fetchWithTimeout(url, options = {}) {
      const { timeout = 10000, ...fetchOptions } = options;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Request Timeout')), timeout);
        fetch(url, fetchOptions)
          .then(response => { clearTimeout(timer); resolve(response); })
          .catch(err => { clearTimeout(timer); reject(err); });
      });
    }"""

content = content.replace(polyfill, fetch_with_timeout)

# Add renderEmptyState function before DOMContentLoaded
empty_state = """
    function renderEmptyState() {
      const el = document.getElementById('generatedAt');
      if (el) el.textContent = 'No data';
      const grid = document.getElementById('kpiGrid');
      if (grid) {
        grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:48px;background:rgba(255,255,255,0.02);border-radius:12px;color:#8b949e;"><i data-lucide="database" style="width:48px;height:48px;opacity:0.5;margin-bottom:16px;display:inline-block"></i><h3 style="color:#fff;font-size:1.2rem;margin-bottom:8px">No dashboard data available.</h3><p style="margin-bottom:24px">Upload a dataset and run the pipeline to generate insights.</p><button onclick="navigateTo('upload')" style="background:var(--indigo);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-weight:600;cursor:pointer;font-family:inherit;"><i data-lucide="upload-cloud" class="icon-sm" style="display:inline-block;vertical-align:middle;margin-right:6px"></i> Upload Dataset</button></div>`;
      }
    }
"""

content = content.replace('// BOOT', empty_state + '\n    // BOOT')

# Fix fetch calls safely
content = content.replace("fetch(API_BASE + '/api/dashboard/data', { signal: AbortSignal.timeout(8000) })", "fetchWithTimeout(API_BASE + '/api/dashboard/data', { timeout: 8000 })")
content = content.replace("fetch(p, { signal: AbortSignal.timeout(4000) })", "fetchWithTimeout(p, { timeout: 4000 })")
content = content.replace("fetch(API_BASE + `/api/data/customers?page=${currentCustomerPage}&page_size=${CUSTOMER_PAGE_SIZE}`)", "fetchWithTimeout(API_BASE + `/api/data/customers?page=${currentCustomerPage}&page_size=${CUSTOMER_PAGE_SIZE}`, { timeout: 10000 })")

# Implement empty state logic in DOMContentLoaded
old_logic = """if (!DATA) throw new Error('Could not load dashboard data from API or static file. Please run the pipeline first.');
        renderOverview();
        hideLoader();
        initScroll();
        hydrateFromAPI();"""

new_logic = """if (!DATA) {
          renderEmptyState();
          hideLoader();
        } else {
          renderOverview();
          hideLoader();
          initScroll();
          hydrateFromAPI();
        }"""

content = content.replace(old_logic, new_logic)

with codecs.open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
