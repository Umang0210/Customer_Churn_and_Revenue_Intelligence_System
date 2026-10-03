const fs = require('fs');

let html = fs.readFileSync('public/index.html', 'utf8');

// Inside fetch(API_BASE + '/api/upload/dataset')
html = html.replace(
    /const data = await response\.json\(\)\.catch\(\(\) => \(\{\}\)\);/,
    `const data = await response.json().catch(() => ({}));
    if (data && data.run_id) window.currentPipelineRunId = data.run_id;`
);

// Inside fetch(API_BASE + '/api/upload/status')
html = html.replace(
    /const data = await fetch\(API_BASE \+ '\/api\/upload\/status', \{signal:AbortSignal\.timeout\(3000\)\}\)\.then\(r=>r\.json\(\)\);/g,
    `const data = await fetch(API_BASE + '/api/upload/status', {signal:AbortSignal.timeout(3000)}).then(r=>r.json());
        if (data && data.run_id) window.currentPipelineRunId = data.run_id;`
);
html = html.replace(
    /const data = await fetch\(API_BASE \+ '\/api\/upload\/status', \{signal:AbortSignal\.timeout\(2000\)\}\)\.then\(r=>r\.json\(\)\);/g,
    `const data = await fetch(API_BASE + '/api/upload/status', {signal:AbortSignal.timeout(2000)}).then(r=>r.json());
      if (data && data.run_id) window.currentPipelineRunId = data.run_id;`
);

// When Pipeline finishes (Status Success/Failed), make sure Progress shows FAILED if there are failures
html = html.replace(
    /if \(data\.status === 'success'\) \{\s*showToast\('o. Pipeline complete! Refresh the page to see updated data.'\);/,
    `if (data.status === 'success') {
            showToast('✓ Pipeline complete! Refresh the page to see updated data.');`
);

// We should also display if the pipeline failed in the status bar
html = html.replace(
    /const msg = document\.getElementById\('statusMsg'\);\s*msg\.className = `status-msg \$\{data\.status==='running'\?'running':data\.status==='success'\?'success':'failed'\}`;/,
    `const msg = document.getElementById('statusMsg');
    msg.className = \`status-msg \${data.status==='running'?'running':data.status==='success'?'success':'failed'}\`;
    
    // If progress is 100% but status is failed
    if (data.status === 'failed' && pct === 100) {
        document.getElementById('progBar').style.background = 'var(--rose)';
    } else {
        document.getElementById('progBar').style.background = 'var(--gi)';
    }`
);

fs.writeFileSync('public/index.html', html, 'utf8');
console.log("Patched index.html run_id and status handling");
