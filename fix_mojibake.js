const fs = require('fs');

function fixFile(filePath) {
    let raw = fs.readFileSync(filePath, 'utf8');
    
    // First, let's try to fix double-encoding by reading as latin1 if it contains Ã
    if (raw.includes('Ã')) {
        try {
            const buf = Buffer.from(raw, 'latin1');
            const decoded = buf.toString('utf8');
            if (!decoded.includes('Ã')) {
                raw = decoded;
            }
        } catch(e) {}
    }

    // Now replace common emojis and mojibake with Lucide icons
    const replacements = {
        'ðŸ“Š': '<i data-lucide="bar-chart-2" class="icon-sm"></i>', // 📊
        '📊': '<i data-lucide="bar-chart-2" class="icon-sm"></i>',
        'ðŸ¤–': '<i data-lucide="bot" class="icon-sm"></i>', // 🤖
        '🤖': '<i data-lucide="bot" class="icon-sm"></i>',
        'ðŸ“ˆ': '<i data-lucide="line-chart" class="icon-sm"></i>', // 📈
        '📈': '<i data-lucide="line-chart" class="icon-sm"></i>',
        'ðŸ‘¥': '<i data-lucide="users" class="icon-sm"></i>', // 👥
        '👥': '<i data-lucide="users" class="icon-sm"></i>',
        'ðŸ“‚': '<i data-lucide="folder-up" class="icon-sm"></i>', // 📁
        '📁': '<i data-lucide="folder-up" class="icon-sm"></i>',
        'ðŸ“„': '<i data-lucide="file-text" class="icon-sm"></i>', // 📄
        '📄': '<i data-lucide="file-text" class="icon-sm"></i>',
        'âœ“': '<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>', // ✓
        '✓': '<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>',
        'âœ—': '<i data-lucide="x-circle" class="icon-sm text-rose"></i>', // ✗
        '✗': '<i data-lucide="x-circle" class="icon-sm text-rose"></i>',
        'â†“': '<i data-lucide="download" class="icon-sm"></i>', // ↓
        '↓': '<i data-lucide="download" class="icon-sm"></i>',
        'âš™': '<i data-lucide="settings" class="icon-sm"></i>', // ⚙
        '⚙': '<i data-lucide="settings" class="icon-sm"></i>',
        'â˜…': '<i data-lucide="star" class="icon-sm"></i>', // ★
        '★': '<i data-lucide="star" class="icon-sm"></i>',
        'â€œ': '"', // “
        'â€': '"', // ”
        'â€™': "'", // ’
        'â€”': '-', // —
        'â€¦': '...', // …
        'Ã¢â‚¬Â¦': '...',
        'Ã¢Å“â€œ': '<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>',
        'Ã¢â€¢Â': '═',
        'â• ': '═',
        'ðŸš€': '<i data-lucide="rocket" class="icon-sm"></i>', // 🚀
        '🚀': '<i data-lucide="rocket" class="icon-sm"></i>',
        'âš ï¸ ': '<i data-lucide="alert-triangle" class="icon-sm text-amber"></i>',
        '⚠️': '<i data-lucide="alert-triangle" class="icon-sm text-amber"></i>',
        'ðŸ”„': '<i data-lucide="refresh-cw" class="icon-sm"></i>',
        '🔄': '<i data-lucide="refresh-cw" class="icon-sm"></i>',
        'ðŸ“œ': '<i data-lucide="file-text" class="icon-sm"></i>',
        '📜': '<i data-lucide="file-text" class="icon-sm"></i>',
        'ðŸ”§': '<i data-lucide="wrench" class="icon-sm"></i>',
        '🔧': '<i data-lucide="wrench" class="icon-sm"></i>',
        'âš¡': '<i data-lucide="zap" class="icon-sm"></i>',
        '⚡': '<i data-lucide="zap" class="icon-sm"></i>',
        'ðŸ¤”': '<i data-lucide="help-circle" class="icon-sm"></i>',
        '🤔': '<i data-lucide="help-circle" class="icon-sm"></i>',
        'ðŸ’°': '<i data-lucide="dollar-sign" class="icon-sm"></i>',
        '💰': '<i data-lucide="dollar-sign" class="icon-sm"></i>'
    };

    for (const [bad, good] of Object.entries(replacements)) {
        raw = raw.split(bad).join(good);
    }

    // Add Lucide CDN and styling if not present in index.html
    if (filePath.endsWith('index.html')) {
        if (!raw.includes('lucide@latest')) {
            raw = raw.replace('</head>', '  <script src="https://unpkg.com/lucide@latest"></script>\n  <style>.icon-sm { width: 1.1em; height: 1.1em; vertical-align: -0.125em; display: inline-block; } .text-emerald { color: var(--emerald); } .text-rose { color: var(--rose); } .text-amber { color: var(--amber); }</style>\n</head>');
        }
        if (!raw.includes('lucide.createIcons()')) {
            raw = raw.replace('</body>', '  <script>lucide.createIcons();</script>\n</body>');
        }
    }

    fs.writeFileSync(filePath, raw, 'utf8');
    console.log('Fixed', filePath);
}

fixFile('public/index.html');
fixFile('public/dashboard.js');
