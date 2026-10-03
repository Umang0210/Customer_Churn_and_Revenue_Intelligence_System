/**
 * Customer Churn Intelligence Dashboard
 * â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 * Multi-page JS â€” detects current page and renders only relevant sections
 */

// ========================
// Chart.js Global Config
// ========================
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.03)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 14;

const C = {
    indigo: '#6366f1', violet: '#8b5cf6', cyan: '#06b6d4',
    emerald: '#10b981', amber: '#f59e0b', rose: '#f43f5e',
    sky: '#0ea5e9', teal: '#14b8a6',
    indigoA: 'rgba(99,102,241,0.7)', violetA: 'rgba(139,92,246,0.7)',
    cyanA: 'rgba(6,182,212,0.7)', emeraldA: 'rgba(16,185,129,0.7)',
    amberA: 'rgba(245,158,11,0.7)', roseA: 'rgba(244,63,94,0.7)',
    skyA: 'rgba(14,165,233,0.7)',
};

let DATA = null;

// ========================
// BOOT
// ========================
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Use script-loaded data first (works with file://), fetch as fallback
        if (window.DASHBOARD_DATA) {
            DATA = window.DASHBOARD_DATA;
        } else {
            const r = await fetch('dashboard_data.json');
            DATA = await r.json();
        }
        routePage();
        hideLoader();
        initScrollReveal();
    } catch (e) {
        console.error('Load error:', e);
        const ov = document.getElementById('loadingOverlay');
        ov.innerHTML = `<div style="text-align:center;color:#f43f5e;">
      <p style="font-size:1.1rem;font-weight:600;">Failed to load data</p>
      <p style="color:#8b949e;margin-top:8px;">Ensure dashboard_data.js exists in the same folder.</p>
      <p style="color:#484f58;margin-top:4px;font-size:0.75rem;">Error: ${e.message}</p></div>`;
    }
});

function hideLoader() {
    const ov = document.getElementById('loadingOverlay');
    ov.classList.add('hidden');
    setTimeout(() => ov.remove(), 500);
}

// ========================
// ROUTER â€” detect page
// ========================
function routePage() {
    const path = window.location.pathname.split('/').pop() || 'index.html';

    if (path === 'index.html' || path === '' || path === '/') {
        renderOverviewPage();
    } else if (path === 'model.html') {
        renderModelPage();
    } else if (path === 'analytics.html') {
        renderAnalyticsPage();
    } else if (path === 'customers.html') {
        renderCustomersPage();
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PAGE: OVERVIEW (index.html)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function renderOverviewPage() {
    renderTimestamp();
    renderKPIs();
    renderChurnRiskPies();
    renderPipeline();
}

function renderTimestamp() {
    const el = document.getElementById('generatedAt');
    if (el) el.textContent = `Report generated: ${DATA.generated_at}`;
}

function renderKPIs() {
    const grid = document.getElementById('kpiGrid');
    if (!grid) return;
    const k = DATA.model.business_kpis;
    const ch = DATA.churn_distribution;
    const m = DATA.model.evaluation;

    const cards = [
<<<<<<< HEAD
        { label: 'Total Customers', value: fmt(k.total_customers), sub: `${DATA.dataset_overview.columns} features analyzed`, cls: 'neutral', color: 'indigo', icon: '<i data-lucide="users" class="icon-sm"></i>' },
        { label: 'Churn Rate', value: `${ch.churn_rate}%`, sub: `${fmt(ch.churned)} churned of ${fmt(ch.total)}`, cls: 'negative', color: 'rose', icon: '📉' },
        { label: 'Revenue at Risk', value: `$${compact(k.revenue_at_risk)}`, sub: `${k.revenue_at_risk_pct}% of total revenue`, cls: 'negative', color: 'amber', icon: '<i data-lucide="dollar-sign" class="icon-sm"></i>' },
        { label: 'Model ROC-AUC', value: m.roc_auc.toFixed(4), sub: `${DATA.model.selected_model.replace('_', ' ')} — ${DATA.model.model_version}`, cls: 'positive', color: 'emerald', icon: '🎯' },
        { label: 'High-Risk Customers', value: `${k.high_risk_pct}%`, sub: `${fmt(DATA.model.risk_distribution.HIGH || 0)} customers > 70% prob`, cls: 'negative', color: 'rose', icon: '🔥' },
        { label: 'Total Revenue', value: `$${compact(k.total_revenue)}`, sub: `Avg $${DATA.revenue_analysis.avg_monthly_charges}/mo per customer`, cls: 'neutral', color: 'cyan', icon: '💎' },
=======
        { label: 'Total Customers', value: fmt(k.total_customers), sub: `${DATA.dataset_overview.columns} features analyzed`, cls: 'neutral', color: 'indigo', icon: 'ðŸ‘¥' },
        { label: 'Churn Rate', value: `${ch.churn_rate}%`, sub: `${fmt(ch.churned)} churned of ${fmt(ch.total)}`, cls: 'negative', color: 'rose', icon: 'ðŸ“‰' },
        { label: 'Revenue at Risk', value: `$${compact(k.revenue_at_risk)}`, sub: `${k.revenue_at_risk_pct}% of total revenue`, cls: 'negative', color: 'amber', icon: 'ðŸ’°' },
        { label: 'Model ROC-AUC', value: m.roc_auc.toFixed(4), sub: `${DATA.model.selected_model.replace('_', ' ')} â€” ${DATA.model.model_version}`, cls: 'positive', color: 'emerald', icon: 'ðŸŽ¯' },
        { label: 'High-Risk Customers', value: `${k.high_risk_pct}%`, sub: `${fmt(DATA.model.risk_distribution.HIGH || 0)} customers > 70% prob`, cls: 'negative', color: 'rose', icon: 'ðŸ”¥' },
        { label: 'Total Revenue', value: `$${compact(k.total_revenue)}`, sub: `Avg $${DATA.revenue_analysis.avg_monthly_charges}/mo per customer`, cls: 'neutral', color: 'cyan', icon: 'ðŸ’Ž' },
>>>>>>> 93636510e0d10e6f707978fa8ef8e4985267ec67
    ];

    grid.innerHTML = cards.map(c => `
    <div class="kpi-card ${c.color} animate-in">
      <div class="kpi-icon">${c.icon}</div>
      <div class="kpi-label">${c.label}</div>
      <div class="kpi-value">${c.value}</div>
      <div class="kpi-change ${c.cls}">${c.sub}</div>
    </div>`).join('');
}

function renderChurnRiskPies() {
    const ch = DATA.churn_distribution;
    const risk = DATA.model.risk_distribution;

    const churnEl = document.getElementById('churnPieChart');
    if (churnEl) {
        new Chart(churnEl, {
            type: 'doughnut',
            data: {
                labels: ['Retained', 'Churned'],
                datasets: [{ data: [ch.retained, ch.churned], backgroundColor: [C.emeraldA, C.roseA], borderColor: ['rgba(16,185,129,0.25)', 'rgba(244,63,94,0.25)'], borderWidth: 2, hoverOffset: 8 }]
            },
            options: { responsive: true, maintainAspectRatio: true, cutout: '62%', plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: ctx => `${ctx.label}: ${fmt(ctx.raw)} (${((ctx.raw / ch.total) * 100).toFixed(1)}%)` } } } }
        });
    }

    const riskEl = document.getElementById('riskPieChart');
    if (riskEl) {
        const total = (risk.LOW || 0) + (risk.MEDIUM || 0) + (risk.HIGH || 0);
        new Chart(riskEl, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{ data: [risk.LOW || 0, risk.MEDIUM || 0, risk.HIGH || 0], backgroundColor: [C.emeraldA, C.amberA, C.roseA], borderColor: ['rgba(16,185,129,0.25)', 'rgba(245,158,11,0.25)', 'rgba(244,63,94,0.25)'], borderWidth: 2, hoverOffset: 8 }]
            },
            options: { responsive: true, maintainAspectRatio: true, cutout: '62%', plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: ctx => `${ctx.label}: ${fmt(ctx.raw)} (${((ctx.raw / total) * 100).toFixed(1)}%)` } } } }
        });
    }
}

function renderPipeline() {
    const flow = document.getElementById('pipelineFlow');
    if (!flow) return;
    const steps = [
<<<<<<< HEAD
        { icon: '<i data-lucide="folder-up" class="icon-sm"></i>', name: 'Raw Data', file: 'data/raw/', bg: 'rgba(139,92,246,0.12)' },
        { icon: '📥', name: 'Ingestion', file: 'ingestion.py', bg: 'rgba(6,182,212,0.12)' },
        { icon: '🧹', name: 'Cleaning', file: 'cleaning.py', bg: 'rgba(16,185,129,0.12)' },
        { icon: '<i data-lucide="settings" class="icon-sm"></i>️', name: 'Features', file: 'features.py', bg: 'rgba(245,158,11,0.12)' },
        { icon: '<i data-lucide="bot" class="icon-sm"></i>', name: 'Training', file: 'train.py', bg: 'rgba(99,102,241,0.12)' },
        { icon: '<i data-lucide="bar-chart-2" class="icon-sm"></i>', name: 'Evaluation', file: 'evaluate.py', bg: 'rgba(16,185,129,0.12)' },
        { icon: '💡', name: 'Insights', file: 'business_insights.py', bg: 'rgba(244,63,94,0.12)' },
        { icon: '🌐', name: 'Dashboard', file: 'index.html', bg: 'rgba(14,165,233,0.12)' },
=======
        { icon: 'ðŸ“', name: 'Raw Data', file: 'data/raw/', bg: 'rgba(139,92,246,0.12)' },
        { icon: 'ðŸ“¥', name: 'Ingestion', file: 'ingestion.py', bg: 'rgba(6,182,212,0.12)' },
        { icon: 'ðŸ§¹', name: 'Cleaning', file: 'cleaning.py', bg: 'rgba(16,185,129,0.12)' },
        { icon: 'âš™ï¸', name: 'Features', file: 'features.py', bg: 'rgba(245,158,11,0.12)' },
        { icon: 'ðŸ¤–', name: 'Training', file: 'train.py', bg: 'rgba(99,102,241,0.12)' },
        { icon: 'ðŸ“Š', name: 'Evaluation', file: 'evaluate.py', bg: 'rgba(16,185,129,0.12)' },
        { icon: 'ðŸ’¡', name: 'Insights', file: 'business_insights.py', bg: 'rgba(244,63,94,0.12)' },
        { icon: 'ðŸŒ', name: 'Dashboard', file: 'index.html', bg: 'rgba(14,165,233,0.12)' },
>>>>>>> 93636510e0d10e6f707978fa8ef8e4985267ec67
    ];
    flow.innerHTML = steps.map((s, i) => `
    ${i > 0 ? '<div class="pipeline-arrow">â†’</div>' : ''}
    <div class="pipeline-step">
      <div class="step-icon" style="background:${s.bg}">${s.icon}</div>
      <div class="step-name">${s.name}</div>
      <div class="step-file">${s.file}</div>
    </div>`).join('');
}


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PAGE: MODEL (model.html)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function renderModelPage() {
    renderModelCards();
    renderConfusionMatrix();
    renderRadarChart();
    renderFeatures();
    renderThresholds();
    renderFeatureImportance();
}

function renderModelCards() {
    const cardsEl = document.getElementById('modelCards');
    if (!cardsEl) return;
    const comp = DATA.model.model_comparison;
    const selected = DATA.model.selected_model;

    const models = Object.entries(comp).map(([name, metrics]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        key: name, isSelected: name === selected, metrics
    }));

    cardsEl.innerHTML = models.map(m => `
    <div class="model-card ${m.isSelected ? 'selected' : ''}">
      <div class="model-name">${m.name}</div>
      ${metricBar('ROC-AUC', m.metrics.roc_auc, 'indigo')}
      ${metricBar('Precision', m.metrics.precision, 'emerald')}
      ${metricBar('Recall', m.metrics.recall, 'amber')}
    </div>`).join('');

    requestAnimationFrame(() => {
        document.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.dataset.width + '%'; });
    });
}

function metricBar(label, val, color) {
    const pct = (val * 100).toFixed(1);
    return `<div class="model-metric">
    <span class="metric-label">${label}</span>
    <span class="metric-bar">
      <div class="bar-track"><div class="bar-fill ${color}" data-width="${pct}" style="width:0;"></div></div>
      <span class="metric-value">${pct}%</span>
    </span></div>`;
}

function renderConfusionMatrix() {
    const el = document.getElementById('confusionMatrix'); if (!el) return;
    const cm = DATA.model.evaluation.confusion_matrix;
    const tsEl = document.getElementById('testSamples');
    if (tsEl) tsEl.textContent = DATA.model.evaluation.test_samples;

    el.innerHTML = `
    <div class="cm-header"></div><div class="cm-header">Pred: No</div><div class="cm-header">Pred: Yes</div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg);">Actual: No</div>
    <div class="cm-cell tn">${cm[0][0]}<span class="cm-label">True Neg</span></div>
    <div class="cm-cell fp">${cm[0][1]}<span class="cm-label">False Pos</span></div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg);">Actual: Yes</div>
    <div class="cm-cell fn">${cm[1][0]}<span class="cm-label">False Neg</span></div>
    <div class="cm-cell tp">${cm[1][1]}<span class="cm-label">True Pos</span></div>`;
}

function renderRadarChart() {
    const el = document.getElementById('radarChart'); if (!el) return;
    const ev = DATA.model.evaluation;
    const comp = DATA.model.model_comparison;

    new Chart(el, {
        type: 'radar',
        data: {
            labels: ['ROC-AUC', 'Precision', 'Recall', 'F1 Score'],
            datasets: [
                { label: 'Random Forest', data: [comp.random_forest.roc_auc, comp.random_forest.precision, comp.random_forest.recall, ev.f1_score], borderColor: C.indigo, backgroundColor: 'rgba(99,102,241,0.08)', borderWidth: 2, pointBackgroundColor: C.indigo },
                { label: 'Logistic Regression', data: [comp.logistic_regression.roc_auc, comp.logistic_regression.precision, comp.logistic_regression.recall, null], borderColor: C.violet, backgroundColor: 'rgba(139,92,246,0.06)', borderWidth: 2, pointBackgroundColor: C.violet }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: true,
            scales: { r: { beginAtZero: true, max: 1, ticks: { stepSize: 0.2, display: false }, grid: { color: 'rgba(255,255,255,0.04)' }, angleLines: { color: 'rgba(255,255,255,0.04)' }, pointLabels: { font: { size: 11, weight: '500' } } } },
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function renderFeatures() {
    const countEl = document.getElementById('featureCount');
    const tagsEl = document.getElementById('featureTags');
    if (countEl) countEl.textContent = DATA.model.num_features;
    if (tagsEl) {
        tagsEl.innerHTML = DATA.model.feature_list.map((f, i) => `
      <div class="feature-tag"><span class="tag-number">${i + 1}</span>${f}</div>`).join('');
    }
}

function renderThresholds() {
    const thresholds = DATA.model.thresholds;

    const chartEl = document.getElementById('thresholdChart');
    if (chartEl) {
        new Chart(chartEl, {
            type: 'line',
            data: {
                labels: thresholds.map(t => t.threshold.toFixed(1)),
                datasets: [
                    { label: 'Precision', data: thresholds.map(t => t.precision), borderColor: C.emerald, backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 },
                    { label: 'Recall', data: thresholds.map(t => t.recall), borderColor: C.amber, backgroundColor: 'rgba(245,158,11,0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 },
                    { label: 'F1 Score', data: thresholds.map(t => t.f1_score), borderColor: C.indigo, backgroundColor: 'rgba(99,102,241,0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: true,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { title: { display: true, text: 'Threshold', font: { size: 11, weight: '500' } }, grid: { display: false } },
                    y: { beginAtZero: true, max: 1, title: { display: true, text: 'Score', font: { size: 11, weight: '500' } }, grid: { color: 'rgba(255,255,255,0.03)' } }
                },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    const tbody = document.querySelector('#thresholdTable tbody');
    if (tbody) {
        tbody.innerHTML = thresholds.map(t => {
            const hl = t.threshold === 0.5 ? ' style="background:rgba(99,102,241,0.06);"' : '';
<<<<<<< HEAD
            return `<tr${hl}><td>${t.threshold.toFixed(1)}${t.threshold === 0.5 ? ' <i data-lucide="star" class="icon-sm"></i>' : ''}</td><td>${(t.precision * 100).toFixed(2)}%</td><td>${(t.recall * 100).toFixed(2)}%</td><td>${(t.f1_score * 100).toFixed(2)}%</td></tr>`;
=======
            return `<tr${hl}><td>${t.threshold.toFixed(1)}${t.threshold === 0.5 ? ' â˜…' : ''}</td><td>${(t.precision * 100).toFixed(2)}%</td><td>${(t.recall * 100).toFixed(2)}%</td><td>${(t.f1_score * 100).toFixed(2)}%</td></tr>`;
>>>>>>> 93636510e0d10e6f707978fa8ef8e4985267ec67
        }).join('');
    }
}

async function renderFeatureImportance() {
    const chartEl = document.getElementById('featureImportanceChart');
    if (!chartEl) return;
    
    try {
        const res = await fetch(API_BASE + '/api/dashboard/feature_importances');
        if (!res.ok) return;
        const importances = await res.json();
        
        if (!importances || Object.keys(importances).length === 0) return;
        
        // Sort importances
        const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 10);
        
        new Chart(chartEl, {
            type: 'bar',
            data: {
                labels: sorted.map(d => d[0].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
                datasets: [{
                    label: 'Importance Score',
                    data: sorted.map(d => d[1]),
                    backgroundColor: 'rgba(99,102,241,0.8)',
                    borderColor: C.indigo,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { grid: { display: false } }
                }
            }
        });
    } catch (e) {
        console.error("Failed to load feature importances:", e);
    }
}



// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PAGE: ANALYTICS (analytics.html)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function renderAnalyticsPage() {
    renderRevenue();
    renderSegmentCharts();
    renderTenureTable();
}

function renderRevenue() {
    const rev = DATA.revenue_analysis;

    const barEl = document.getElementById('revenueBarChart');
    if (barEl) {
        new Chart(barEl, {
            type: 'bar',
            data: {
                labels: ['Retained Customers', 'Churned Customers'],
                datasets: [{ label: 'Total Revenue ($)', data: [rev.retained_total_revenue, rev.churned_total_revenue], backgroundColor: [C.emeraldA, C.roseA], borderColor: [C.emerald, C.rose], borderWidth: 1, borderRadius: 8, barPercentage: 0.5 }]
            },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => '$' + fmt(ctx.raw) } } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { callback: v => '$' + compact(v) } } } }
        });
    }

    const distEl = document.getElementById('chargeDistChart');
    if (distEl) {
        const dist = rev.charge_distribution;
        new Chart(distEl, {
            type: 'bar',
            data: {
                labels: Object.keys(dist),
                datasets: [{ label: 'Customers', data: Object.values(dist), backgroundColor: [C.emeraldA, C.cyanA, C.indigoA, C.violetA, C.amberA], borderRadius: 8, barPercentage: 0.6 }]
            },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' } } } }
        });
    }
}

function renderSegmentCharts() {
    const seg = DATA.segment_analysis;
    const tenure = DATA.tenure_analysis;
    const svc = DATA.services_analysis;

    // Contract
    const contractEl = document.getElementById('contractChart');
    if (contractEl) {
        new Chart(contractEl, {
            type: 'bar',
            data: {
                labels: seg.contract.map(c => c.contract),
                datasets: [{ label: 'Churn Rate (%)', data: seg.contract.map(c => c.churn_rate), backgroundColor: [C.roseA, C.amberA, C.emeraldA], borderRadius: 8, barPercentage: 0.6 }]
            },
            options: { responsive: true, maintainAspectRatio: true, indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.raw}% churn` } } }, scales: { x: { beginAtZero: true, max: 50, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { callback: v => v + '%' } }, y: { grid: { display: false } } } }
        });
    }

    // Tenure
    const tenureEl = document.getElementById('tenureChart');
    if (tenureEl) {
        const colors = [C.roseA, C.amberA, C.amberA, C.emeraldA, C.emeraldA];
        new Chart(tenureEl, {
            type: 'bar',
            data: {
                labels: tenure.groups.map(g => g.group),
                datasets: [{ label: 'Churn Rate (%)', data: tenure.groups.map(g => g.churn_rate), backgroundColor: colors, borderRadius: 6, barPercentage: 0.7 }]
            },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.raw}% churn` } } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { callback: v => v + '%' } } } }
        });
    }

    // Internet
    const internetEl = document.getElementById('internetChart');
    if (internetEl && svc.internet_service) {
        new Chart(internetEl, {
            type: 'bar',
            data: {
                labels: svc.internet_service.map(s => s.service),
                datasets: [{ label: 'Churn Rate (%)', data: svc.internet_service.map(s => s.churn_rate), backgroundColor: [C.cyanA, C.roseA, C.emeraldA], borderRadius: 6, barPercentage: 0.6 }]
            },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { callback: v => v + '%' } } } }
        });
    }

    // Payment
    const paymentEl = document.getElementById('paymentChart');
    if (paymentEl && svc.payment_method) {
        new Chart(paymentEl, {
            type: 'bar',
            data: {
                labels: svc.payment_method.map(s => s.method),
                datasets: [{ label: 'Churn Rate (%)', data: svc.payment_method.map(s => s.churn_rate), backgroundColor: svc.payment_method.map(s => s.churn_rate > 40 ? C.roseA : s.churn_rate > 20 ? C.amberA : C.emeraldA), borderRadius: 8, barPercentage: 0.6 }]
            },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { maxRotation: 15 } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { callback: v => v + '%' } } } }
        });
    }
}

function renderTenureTable() {
    const tbody = document.querySelector('#tenureTable tbody'); if (!tbody) return;
    const tenure = DATA.tenure_analysis;
    tbody.innerHTML = tenure.groups.map(g => `
    <tr>
      <td>${g.group}</td>
      <td>${fmt(g.count)}</td>
      <td><span class="risk-badge ${g.churn_rate > 40 ? 'high' : g.churn_rate > 20 ? 'medium' : 'low'}">${g.churn_rate}%</span></td>
      <td>$${g.avg_monthly.toFixed(2)}</td>
    </tr>`).join('');
}


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PAGE: CUSTOMERS (customers.html)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function renderCustomersPage() {
    renderTopCustomers();
    renderDatasetOverview();
}

function renderTopCustomers() {
    const tbody = document.querySelector('#customersTable tbody'); if (!tbody) return;
    const customers = DATA.model.top_customers;
    tbody.innerHTML = customers.map((c, i) => {
        const risk = c.churn_probability >= 0.7 ? 'high' : c.churn_probability >= 0.4 ? 'medium' : 'low';
        return `<tr>
      <td>${i + 1}</td>
      <td style="font-family:'Courier New',monospace;font-size:0.78rem;">${c.customer_id}</td>
      <td>${(c.churn_probability * 100).toFixed(1)}%</td>
      <td><span class="risk-badge ${risk}">${risk.toUpperCase()}</span></td>
      <td>$${c.monthly_charges.toFixed(2)}</td>
      <td>${c.tenure} mo</td>
      <td>$${fmt(c.revenue)}</td>
      <td style="color:#fb7185;font-weight:600;">$${fmt(c.expected_loss)}</td>
    </tr>`;
    }).join('');
}

function renderDatasetOverview() {
    const ds = DATA.dataset_overview;
    const statsEl = document.getElementById('datasetStats');
    const gridEl = document.getElementById('columnsGrid');

    if (statsEl) {
        const stats = [
            { value: fmt(ds.rows), label: 'Total Rows' },
            { value: ds.columns, label: 'Columns' },
            { value: ds.duplicate_rows, label: 'Duplicates' },
            { value: `${ds.missing_pct}%`, label: 'Missing Values' },
            { value: `${ds.memory_mb} MB`, label: 'Memory Usage' },
        ];
        statsEl.innerHTML = stats.map(s => `
      <div class="overview-stat">
        <div class="stat-value">${s.value}</div>
        <div class="stat-label">${s.label}</div>
      </div>`).join('');
    }

    if (gridEl) {
        gridEl.innerHTML = ds.columns_info.map(col => `
      <div class="column-item">
        <span class="column-dtype ${col.dtype}">${col.dtype}</span>
        <span class="column-name">${col.name}</span>
        <span class="column-unique">${col.unique} unique</span>
      </div>`).join('');
    }
}


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  SCROLL REVEAL
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function initScrollReveal() {
    const sections = document.querySelectorAll('.section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
    sections.forEach(s => observer.observe(s));
}


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  UTILITIES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function fmt(n) {
    if (n === null || n === undefined) return 'â€”';
    return Number(n).toLocaleString('en-US', { maximumFractionDigits: 2 });
}

function compact(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toFixed(2);
}


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

<<<<<<< HEAD
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
                
                return `
                <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex; align-items:center; gap:8px;">
                        ${icon}
                        <span style="font-size:0.9rem; color:var(--tx1);">Step ${i+1} — ${s.name || 'Unknown'}</span>
                    </div>
                    <div style="font-size:0.75rem; color:${color};">${s.status ? s.status.toUpperCase() : 'PENDING'}</div>
                </div>
                `;
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
=======
>>>>>>> 93636510e0d10e6f707978fa8ef8e4985267ec67
