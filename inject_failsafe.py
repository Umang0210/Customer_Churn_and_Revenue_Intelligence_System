import codecs

with codecs.open('public/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

failsafe = """document.addEventListener('DOMContentLoaded', async () => {
  setTimeout(() => {
    const ov = document.getElementById('loadingOverlay');
    if (ov && ov.innerHTML.includes('spinner')) {
      ov.innerHTML = '<div style="text-align:center;color:#f43f5e;padding:32px;max-width:440px"><div style="font-size:2.5rem;margin-bottom:12px;color:var(--rose)"><i data-lucide="alert-triangle" style="width:48px;height:48px"></i></div><p style="font-size:1rem;font-weight:700;margin-bottom:8px">Timeout Error</p><p style="color:#8b949e;font-size:.83rem;margin-bottom:14px">The dashboard took too long to load. A network request may have hung indefinitely.</p><button onclick="window.location.reload()" style="background:var(--rose);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-weight:600;cursor:pointer;font-family:inherit;">Retry</button></div>';
      if (typeof lucide !== 'undefined') lucide.createIcons();
    }
  }, 20000);
"""

content = content.replace("document.addEventListener('DOMContentLoaded', async () => {", failsafe)

with codecs.open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Failsafe injected')
