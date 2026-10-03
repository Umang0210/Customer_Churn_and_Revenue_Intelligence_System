const fs = require('fs');

// UPDATE index.html
let html = fs.readFileSync('public/index.html', 'utf8');

// Add View Logs button next to Pipeline Progress title
html = html.replace(
    /<div class="chart-title" style="margin-bottom:14px">.*?Pipeline Progress<\/div>/,
    `<div class="chart-title" style="margin-bottom:14px; display:flex; justify-content:space-between; align-items:center">
        <span><i data-lucide="rocket" class="icon-sm"></i> Pipeline Progress</span>
        <button class="btn btn-outline" style="padding:4px 10px; font-size:0.75rem; border-color:var(--border)" onclick="showLogs()">
            <i data-lucide="file-text" class="icon-sm"></i> View Logs
        </button>
    </div>`
);

// Add Logs Modal at the end of the body
if (!html.includes('logsModal')) {
    html = html.replace('</body>', `
<!-- LOGS MODAL -->
<div id="logsModal" class="modal" style="display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,0.7); backdrop-filter:blur(4px);">
  <div class="modal-content card" style="margin:5% auto; width:80%; max-width:800px; max-height:80vh; display:flex; flex-direction:column; padding:24px; position:relative;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
      <h3 style="margin:0; font-size:1.2rem; color:var(--tx1)"><i data-lucide="file-text" class="icon-sm"></i> Pipeline Logs</h3>
      <button onclick="document.getElementById('logsModal').style.display='none'" style="background:transparent; border:none; color:var(--tx2); cursor:pointer; font-size:1.5rem">&times;</button>
    </div>
    <div style="flex:1; overflow-y:auto; background:#1e1e1e; padding:16px; border-radius:8px; border:1px solid var(--border); text-align:left;">
      <pre id="logsContent" style="margin:0; font-family:monospace; font-size:0.85rem; color:#d4d4d4; white-space:pre-wrap; word-wrap:break-word;">Loading logs...</pre>
    </div>
    <div style="margin-top:16px; display:flex; justify-content:flex-end;">
      <button class="btn" onclick="fetchLogs()"><i data-lucide="refresh-cw" class="icon-sm"></i> Refresh Logs</button>
    </div>
  </div>
</div>
</body>`);
}

fs.writeFileSync('public/index.html', html, 'utf8');

// UPDATE dashboard.js
let js = fs.readFileSync('public/dashboard.js', 'utf8');

if (!js.includes('showLogs()')) {
    js += `

// --- LOGS MODAL FUNCTIONS ---
function showLogs() {
  document.getElementById("logsModal").style.display = "block";
  fetchLogs();
}

async function fetchLogs() {
  const content = document.getElementById("logsContent");
  content.textContent = "Loading logs...";
  try {
    const response = await fetch(API_BASE + "/api/upload/logs");
    if (!response.ok) {
        content.textContent = "Failed to fetch logs. Error " + response.status;
        return;
    }
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
        const data = await response.json();
        content.textContent = data.message || JSON.stringify(data);
    } else {
        const text = await response.text();
        content.textContent = text || "No logs available.";
    }
  } catch(e) {
    content.textContent = "Error fetching logs: " + e.message;
  }
}
`;
    fs.writeFileSync('public/dashboard.js', js, 'utf8');
}
console.log('Done modifying HTML/JS');
