const fs = require('fs');
let html = fs.readFileSync('public/index.html', 'utf8');

let fixed = false;

// We're specifically targeting the weird A,A?A and dY", corruptions
if (html.includes('A,A?A')) {
    html = html.replace('A,A?A', '<i data-lucide="home" class="icon-sm"></i>');
    console.log('Fixed A,A?A');
    fixed = true;
}
if (html.includes('dY",')) {
    html = html.replace('dY",', '<i data-lucide="folder-up" class="icon-sm"></i>');
    console.log('Fixed dY",');
    fixed = true;
}
if (html.includes('ðŸ  ')) {
    html = html.replace(/ðŸ \ /g, '<i data-lucide="home" class="icon-sm"></i>');
    console.log('Fixed ðŸ  ');
    fixed = true;
}
if (html.includes('ðŸ')) {
    html = html.replace(/ðŸ/g, '');
    console.log('Fixed leftover ðŸ');
    fixed = true;
}

if (fixed) fs.writeFileSync('public/index.html', html, 'utf8');

const bad = ['ðŸ', 'âœ', 'â†', 'Ã', 'Â', 'â€', 'â€™', 'â€œ', 'â€¦', 'âš', 'â˜'];
let found = false;
for (const b of bad) {
    if (html.includes(b)) { console.log('Found in html:', b); found = true; }
}
if (!found) console.log('ALL MOJIBAKE GONE!');
