const fs = require('fs');

// --- 1. UPDATE index.html ---
let html = fs.readFileSync('public/index.html', 'utf8');

// Insert Sidebar Link
if (!html.includes('data-page="logs"')) {
    html = html.replace(
        '<div class="nav-label">System</div>',
        `<div class="nav-label">System</div>
      <button class="nav-item" data-page="upload">
        <span class="nav-icon"><i data-lucide="folder-up" class="icon-sm"></i></span><span class="nav-text">Upload Data</span>
      </button>
      <button class="nav-item" data-page="logs">
        <span class="nav-icon"><i data-lucide="file-text" class="icon-sm"></i></span><span class="nav-text">Logs</span>
      </button>`
    );
    // Remove the old upload button to avoid duplicates
    html = html.replace(
        /<button class="nav-item" data-page="upload">[\s\S]*?<span class="nav-icon">.*?<\/span><span class="nav-text">Upload Data<\/span>\s*<\/button>/,
        ''
    );
}

// Insert View Logs Button in Pipeline Progress
if (!html.includes('onclick="viewCurrentLogs()"')) {
    html = html.replace(
        /<div class="chart-title" style="margin-bottom:14px">.*?Pipeline Progress<\/div>/,
        `<div class="chart-title" style="margin-bottom:14px; display:flex; justify-content:space-between; align-items:center">
        <span><i data-lucide="rocket" class="icon-sm"></i> Pipeline Progress</span>
        <button class="btn btn-outline" style="padding:4px 10px; font-size:0.75rem; border-color:var(--border)" onclick="viewCurrentLogs()">
            <i data-lucide="file-text" class="icon-sm"></i> View Logs
        </button>
    </div>`
    );
}

// Insert Logs Page
if (!html.includes('id="page-logs"')) {
    const logsPage = `
    <!-- 📄 LOGS PAGE 📄 -->
    <div class="page" id="page-logs">
      <div class="page-container">
        <div class="page-header">
          <div class="page-breadcrumb"><span>System</span> <span>/</span> Logs</div>
          <h1 class="page-title">Pipeline Logs</h1>
          <p class="page-subtitle">Execution details and step-by-step logs</p>
        </div>
        
        <div id="logsLoading" style="text-align:center; padding:40px; color:var(--tx2); display:none;">
            <div class="loading-spinner" style="margin: 0 auto 16px;"></div>
            Loading pipeline logs...
        </div>

        <div id="logsContentWrapper" style="display:none;">
            <!-- Summary Card -->
            <div class="card" style="margin-bottom: 24px;">
                <h3 style="margin-top:0; margin-bottom:16px; font-size:1.1rem; color:var(--tx1);">Execution Summary</h3>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:16px;">
                    <div><div style="color:var(--tx3); font-size:0.75rem; text-transform:uppercase;">Run ID</div><div id="logRunId" style="font-weight:600; color:var(--tx1); font-family:monospace;">-</div></div>
                    <div><div style="color:var(--tx3); font-size:0.75rem; text-transform:uppercase;">File</div><div id="logFile" style="font-weight:600; color:var(--tx1);">-_</div></div>
                    <div><div style="color:var(--tx3); font-size:0.75rem; text-transform:uppercase;">Started</div><div id="logStarted" style="font-weight:600; color:var(--tx1);">-_</div></div>
                    <div><div style="color:var(--tx3); font-size:0.75rem; text-transform:uppercase;">Duration</div><div id="logDuration" style="font-weight:600; color:var(--tx1);">-_</div></div>
                    <div><div style="color:var(--tx3); font-size:0.75rem; text-transform:uppercase;">Status</div><div id="logStatus" style="font-weight:600;">-_</div></div>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 2fr; gap: 24px;">
                <!-- Pipeline Steps -->
                <div class="card">
                    <h3 style="margin-top:0; margin-bottom:16px; font-size:1.1rem; color:var(--tx1);">Pipeline Steps</h3>
                    <div id="logStepsList" style="display:flex; flex-direction:column; gap:12px;"></div>
                </div>

                <!-- Raw Logs -->
                <div class="card" style="display:flex; flex-direction:column;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                        <h3 style="margin:0; font-size:1.1rem; color:var(--tx1);">Execution Details</h3>
                        <button class="btn btn-outline" style="padding:4px 10px; font-size:0.75rem;" onclick="loadLogsForRun(currentLogRunId)"><i data-lucide="refresh-cw" class="icon-sm"></i> Refresh</button>
                    </div>
                    <div style="flex:1; overflow-y:auto; background:#1e1e1e; padding:16px; border-radius:8px; border:1px solid var(--border); max-height:600px;">
                        <pre id="logRawText" style="margin:0; font-family:monospace; font-size:0.8rem; color:#d4d4d4; white-space:pre-wrap; word-wrap:break-word;">No logs available.</pre>
                    </div>
                </div>
            </div>
        </div>

        <div id="logsError" class="card" style="display:none; border-left:4px solid var(--rose);">
            <h3 style="margin-top:0; color:var(--rose);">Error</h3>
            <p id="logsErrorText" style="color:var(--tx2);">Unable to load pipeline logs.</p>
        </div>
      </div>
    </div>
    `;
    html = html.replace('</main>', logsPage + '\n  </main>');
}

fs.writeFileSync('public/index.html', html, 'utf8');

// --- 2. UPDATE dashboard.js ---
let js = fs.readFileSync('public/dashboard.js', 'utf8');

// Update routePage to handle /logs and /logs/RUN-xxx
js = js.replace(
`function routePage() {
    const path = window.location.pathname.split('/').pop() || 'index.html';

    if (path === 'index.html' || path === '' || path === '/') {
        renderOverviewPage();
    } else if (path === 'model.html') {
        renderModelPage();
    } else if (path === 'analytics.html') {
        renderAnalyticsPage();
    } else if (path === 'customers.html') {
        renderCustomersPage();
    }`,
`function routePage() {
    const pathname = window.location.pathname;
    const path = pathname.split('/').pop() || 'index.html';

    if (pathname.startsWith('/logs')) {
        const parts = pathname.split('/');
        const runId = parts.length > 2 ? parts[parts.length - 1] : null;
        navigateTo('logs');
        if (runId && runId !== 'logs') {
            loadLogsForRun(runId);
        } else if (window.currentPipelineRunId) {
            loadLogsForRun(window.currentPipelineRunId);
        }
        return;
    }

    if (path === 'index.html' || path === '' || path === '/') {
        renderOverviewPage();
    } else if (path === 'model.html') {
        renderModelPage();
    } else if (path === 'analytics.html') {
        renderAnalyticsPage();
    } else if (path === 'customers.html') {
        renderCustomersPage();
    }`
);

// Store the runId globally when upload returns it
js = js.replace(
    /const result = await res\.json\(\);\s+showSuccess\(result\.message\);/,
    `const result = await res.json();
        window.currentPipelineRunId = result.run_id;
        showSuccess(result.message);`
);

// Add the Logs JS Logic
if (!js.includes('function viewCurrentLogs()')) {
    js += `
// ========================
// LOGS PAGE LOGIC
// ========================
let currentLogRunId = null;

function viewCurrentLogs() {
    if (window.currentPipelineRunId) {
        // Change URL without reloading
        window.history.pushState({}, '', '/logs/' + window.currentPipelineRunId);
        navigateTo('logs');
        loadLogsForRun(window.currentPipelineRunId);
    } else {
        navigateTo('logs');
        document.getElementById('logsError').style.display = 'block';
        document.getElementById('logsContentWrapper').style.display = 'none';
        document.getElementById('logsErrorText').textContent = "No pipeline run is currently active in this session.";
    }
}

async function loadLogsForRun(runId) {
    if (!runId) return;
    currentLogRunId = runId;
    
    document.getElementById('logsLoading').style.display = 'block';
    document.getElementById('logsContentWrapper').style.display = 'none';
    document.getElementById('logsError').style.display = 'none';

    try {
        const res = await fetch(API_BASE + '/api/logs/' + runId);
        if (!res.ok) throw new Error('API returned ' + res.status);
        const data = await res.json();
        
        if (data.error) throw new Error(data.error);

        document.getElementById('logsLoading').style.display = 'none';
        document.getElementById('logsContentWrapper').style.display = 'block';

        // Render Summary
        document.getElementById('logRunId').textContent = data.run_id || runId;
        const meta = data.metadata || {};
        document.getElementById('logFile').textContent = meta.filename || 'Unknown';
        document.getElementById('logStarted').textContent = meta.uploaded_at ? new Date(meta.uploaded_at).toLocaleString() : 'Unknown';
        document.getElementById('logDuration').textContent = meta.duration ? meta.duration + 's' : '0s';
        
        const statusEl = document.getElementById('logStatus');
        statusEl.textContent = (meta.status || 'Unknown').toUpperCase();
        statusEl.style.color = meta.status === 'success' ? 'var(--emerald)' : (meta.status === 'failed' ? 'var(--rose)' : 'var(--amber)');

        // Render Steps
        const stepsEl = document.getElementById('logStepsList');
        if (meta.steps && meta.steps.length > 0) {
            stepsEl.innerHTML = meta.steps.map((s, i) => {
                let icon = '<i data-lucide="circle" class="icon-sm text-amber"></i>';
                let color = 'var(--tx2)';
                if (s.status === 'success') {
                    icon = '<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>';
                    color = 'var(--emerald)';
                } else if (s.status === 'failed') {
                    icon = '<i data-lucide="x-circle" class="icon-sm text-rose"></i>';
                    color = 'var(--rose)';
                }
                
                return \`
                <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex; align-items:center; gap:8px;">
                        \${icon}
                        <span style="font-size:0.9rem; color:var(--tx1);">Step \${i+1} — \${s.name || 'Unknown'}</span>
                    </div>
                    <div style="font-size:0.75rem; color:\${color};">\${s.status ? s.status.toUpperCase() : 'PENDING'}</div>
                </div>
                \`;
            }).join('');
        } else {
            stepsEl.innerHTML = '<p style="color:var(--tx3); font-size:0.85rem;">No step data recorded.</p>';
        }

        // Render Raw Logs
        document.getElementById('logRawText').textContent = data.logs || "No logs available for this run.";
        
        if (window.lucide) window.lucide.createIcons();

    } catch (e) {
        document.getElementById('logsLoading').style.display = 'none';
        document.getElementById('logsError').style.display = 'block';
        document.getElementById('logsErrorText').textContent = e.message;
    }
}
`;
}

// Add history.pushState to normal navigation so browser back button works and URLs look nice
js = js.replace(
    /function navigateTo\(pageId\) \{/,
    `function navigateTo(pageId) {
    if (pageId !== 'logs' && !window.location.pathname.startsWith('/logs')) {
        // window.history.pushState({}, '', '/' + (pageId === 'overview' ? '' : pageId + '.html'));
    }`
);

fs.writeFileSync('public/dashboard.js', js, 'utf8');
console.log('Logs page implemented!');
