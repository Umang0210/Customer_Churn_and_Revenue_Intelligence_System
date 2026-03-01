/**
 * Customer Churn Intelligence Dashboard
 * ──────────────────────────────────────
 * Fetches dashboard_data.json and renders all visualizations.
 */

// ========================
// Chart.js Global Config
// ========================
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.04)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;

// Color palette
const COLORS = {
    indigo: '#6366f1',
    violet: '#8b5cf6',
    cyan: '#06b6d4',
    emerald: '#10b981',
    amber: '#f59e0b',
    rose: '#f43f5e',
    sky: '#0ea5e9',
    slate: '#64748b',
    indigoAlpha: 'rgba(99, 102, 241, 0.7)',
    violetAlpha: 'rgba(139, 92, 246, 0.7)',
    cyanAlpha: 'rgba(6, 182, 212, 0.7)',
    emeraldAlpha: 'rgba(16, 185, 129, 0.7)',
    amberAlpha: 'rgba(245, 158, 11, 0.7)',
    roseAlpha: 'rgba(244, 63, 94, 0.7)',
};

let DATA = null;

// ========================
// INIT
// ========================
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('dashboard_data.json');
        DATA = await resp.json();
        render();
        hideLoader();
        initScrollReveal();
    } catch (err) {
        console.error('Failed to load data:', err);
        document.getElementById('loadingOverlay').innerHTML = `
      <div style="text-align:center; color:#f43f5e;">
        <p style="font-size:1.2rem; font-weight:600;">Failed to load dashboard data</p>
        <p style="color:#94a3b8; margin-top:8px;">Make sure dashboard_data.json exists in the same directory.</p>
        <p style="color:#64748b; margin-top:4px; font-size:0.8rem;">Error: ${err.message}</p>
      </div>`;
    }
});

function hideLoader() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 600);
}

// ========================
// MAIN RENDER
// ========================
function render() {
    renderTimestamp();
    renderKPIs();
    renderDatasetOverview();
    renderModelPerformance();
    renderThresholdAnalysis();
    renderChurnRevenue();
    renderSegments();
    renderTopCustomers();
    renderPipeline();
}

// ========================
// TIMESTAMP
// ========================
function renderTimestamp() {
    document.getElementById('generatedAt').textContent =
        `Report generated: ${DATA.generated_at}`;
}

// ========================
// KPI CARDS
// ========================
function renderKPIs() {
    const grid = document.getElementById('kpiGrid');
    const kpis = DATA.model.business_kpis;
    const churn = DATA.churn_distribution;
    const model = DATA.model.evaluation;

    const cards = [
        {
            label: 'Total Customers',
            value: formatNumber(kpis.total_customers),
            change: `${DATA.dataset_overview.columns} features analyzed`,
            changeClass: 'neutral',
            color: 'indigo',
            icon: '👥'
        },
        {
            label: 'Churn Rate',
            value: `${churn.churn_rate}%`,
            change: `${formatNumber(churn.churned)} churned out of ${formatNumber(churn.total)}`,
            changeClass: 'negative',
            color: 'rose',
            icon: '📉'
        },
        {
            label: 'Revenue at Risk',
            value: `$${formatCompact(kpis.revenue_at_risk)}`,
            change: `${kpis.revenue_at_risk_pct}% of total revenue`,
            changeClass: 'negative',
            color: 'amber',
            icon: '💰'
        },
        {
            label: 'Model ROC-AUC',
            value: model.roc_auc.toFixed(4),
            change: `${DATA.model.selected_model.replace('_', ' ')} — v${DATA.model.model_version.replace('v', '')}`,
            changeClass: 'positive',
            color: 'emerald',
            icon: '🎯'
        },
        {
            label: 'High-Risk Customers',
            value: `${kpis.high_risk_pct}%`,
            change: `${formatNumber(DATA.model.risk_distribution.HIGH || 0)} customers above 70% churn prob.`,
            changeClass: 'negative',
            color: 'rose',
            icon: '🔥'
        },
        {
            label: 'Total Revenue',
            value: `$${formatCompact(kpis.total_revenue)}`,
            change: `Avg $${DATA.revenue_analysis.avg_monthly_charges}/mo per customer`,
            changeClass: 'neutral',
            color: 'cyan',
            icon: '💎'
        }
    ];

    grid.innerHTML = cards.map(c => `
    <div class="kpi-card ${c.color} animate-in">
      <div class="kpi-icon">${c.icon}</div>
      <div class="kpi-label">${c.label}</div>
      <div class="kpi-value">${c.value}</div>
      <div class="kpi-change ${c.changeClass}">${c.change}</div>
    </div>
  `).join('');
}

// ========================
// DATASET OVERVIEW
// ========================
function renderDatasetOverview() {
    const ds = DATA.dataset_overview;
    const statsEl = document.getElementById('datasetStats');
    const gridEl = document.getElementById('columnsGrid');

    const stats = [
        { value: formatNumber(ds.rows), label: 'Total Rows' },
        { value: ds.columns, label: 'Columns' },
        { value: ds.duplicate_rows, label: 'Duplicates' },
        { value: `${ds.missing_pct}%`, label: 'Missing Values' },
        { value: `${ds.memory_mb} MB`, label: 'Memory Usage' },
    ];

    statsEl.innerHTML = stats.map(s => `
    <div class="overview-stat">
      <div class="stat-value">${s.value}</div>
      <div class="stat-label">${s.label}</div>
    </div>
  `).join('');

    gridEl.innerHTML = ds.columns_info.map(col => `
    <div class="column-item">
      <span class="column-dtype ${col.dtype}">${col.dtype}</span>
      <span class="column-name">${col.name}</span>
      <span class="column-unique">${col.unique} unique</span>
    </div>
  `).join('');
}

// ========================
// MODEL PERFORMANCE
// ========================
function renderModelPerformance() {
    const comp = DATA.model.model_comparison;
    const selected = DATA.model.selected_model;

    // Model comparison cards
    const cardsEl = document.getElementById('modelCards');
    const models = Object.entries(comp).map(([name, metrics]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        key: name,
        isSelected: name === selected,
        metrics
    }));

    cardsEl.innerHTML = models.map(m => `
    <div class="model-card ${m.isSelected ? 'selected' : ''}">
      <div class="model-name">${m.name}</div>
      ${renderMetricBar('ROC-AUC', m.metrics.roc_auc, 'indigo')}
      ${renderMetricBar('Precision', m.metrics.precision, 'emerald')}
      ${renderMetricBar('Recall', m.metrics.recall, 'amber')}
    </div>
  `).join('');

    // Animate bars after render
    requestAnimationFrame(() => {
        document.querySelectorAll('.bar-fill').forEach(bar => {
            const width = bar.dataset.width;
            bar.style.width = width + '%';
        });
    });

    // Feature count
    document.getElementById('featureCount').textContent = DATA.model.num_features;

    // Feature tags
    const tagsEl = document.getElementById('featureTags');
    tagsEl.innerHTML = DATA.model.feature_list.map((f, i) => `
    <div class="feature-tag">
      <span class="tag-number">${i + 1}</span>
      ${f}
    </div>
  `).join('');

    // Confusion Matrix
    renderConfusionMatrix();

    // Radar Chart
    renderRadarChart();
}

function renderMetricBar(label, value, color) {
    const pct = (value * 100).toFixed(1);
    return `
    <div class="model-metric">
      <span class="metric-label">${label}</span>
      <span class="metric-bar">
        <div class="bar-track">
          <div class="bar-fill ${color}" data-width="${pct}" style="width: 0;"></div>
        </div>
        <span class="metric-value">${pct}%</span>
      </span>
    </div>
  `;
}

function renderConfusionMatrix() {
    const cm = DATA.model.evaluation.confusion_matrix;
    const el = document.getElementById('confusionMatrix');

    el.innerHTML = `
    <div class="cm-header"></div>
    <div class="cm-header">Pred: No</div>
    <div class="cm-header">Pred: Yes</div>

    <div class="cm-header" style="writing-mode: vertical-lr; transform: rotate(180deg);">Actual: No</div>
    <div class="cm-cell tn">
      ${cm[0][0]}
      <span class="cm-label">True Neg</span>
    </div>
    <div class="cm-cell fp">
      ${cm[0][1]}
      <span class="cm-label">False Pos</span>
    </div>

    <div class="cm-header" style="writing-mode: vertical-lr; transform: rotate(180deg);">Actual: Yes</div>
    <div class="cm-cell fn">
      ${cm[1][0]}
      <span class="cm-label">False Neg</span>
    </div>
    <div class="cm-cell tp">
      ${cm[1][1]}
      <span class="cm-label">True Pos</span>
    </div>
  `;
}

function renderRadarChart() {
    const eval_ = DATA.model.evaluation;
    const comp = DATA.model.model_comparison;

    new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['ROC-AUC', 'Precision', 'Recall', 'F1 Score'],
            datasets: [
                {
                    label: 'Random Forest',
                    data: [comp.random_forest.roc_auc, comp.random_forest.precision, comp.random_forest.recall, eval_.f1_score],
                    borderColor: COLORS.indigo,
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: COLORS.indigo,
                },
                {
                    label: 'Logistic Regression',
                    data: [comp.logistic_regression.roc_auc, comp.logistic_regression.precision, comp.logistic_regression.recall, null],
                    borderColor: COLORS.violet,
                    backgroundColor: 'rgba(139, 92, 246, 0.08)',
                    borderWidth: 2,
                    pointBackgroundColor: COLORS.violet,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { stepSize: 0.2, display: false },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    angleLines: { color: 'rgba(255, 255, 255, 0.05)' },
                    pointLabels: { font: { size: 11, weight: '500' } }
                }
            },
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// ========================
// THRESHOLD ANALYSIS
// ========================
function renderThresholdAnalysis() {
    const thresholds = DATA.model.thresholds;

    // Line chart
    new Chart(document.getElementById('thresholdChart'), {
        type: 'line',
        data: {
            labels: thresholds.map(t => t.threshold.toFixed(1)),
            datasets: [
                {
                    label: 'Precision',
                    data: thresholds.map(t => t.precision),
                    borderColor: COLORS.emerald,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                },
                {
                    label: 'Recall',
                    data: thresholds.map(t => t.recall),
                    borderColor: COLORS.amber,
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                },
                {
                    label: 'F1 Score',
                    data: thresholds.map(t => t.f1_score),
                    borderColor: COLORS.indigo,
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    title: { display: true, text: 'Threshold', font: { size: 11, weight: '500' } },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: { display: true, text: 'Score', font: { size: 11, weight: '500' } },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' }
                }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });

    // Table
    const tbody = document.querySelector('#thresholdTable tbody');
    tbody.innerHTML = thresholds.map(t => {
        const best = t.threshold === 0.5 ? ' style="background: rgba(99,102,241,0.08);"' : '';
        return `
      <tr${best}>
        <td>${t.threshold.toFixed(1)}${t.threshold === 0.5 ? ' ★' : ''}</td>
        <td>${(t.precision * 100).toFixed(2)}%</td>
        <td>${(t.recall * 100).toFixed(2)}%</td>
        <td>${(t.f1_score * 100).toFixed(2)}%</td>
      </tr>
    `;
    }).join('');
}

// ========================
// CHURN & REVENUE
// ========================
function renderChurnRevenue() {
    const churn = DATA.churn_distribution;
    const risk = DATA.model.risk_distribution;
    const rev = DATA.revenue_analysis;

    // Churn Pie
    new Chart(document.getElementById('churnPieChart'), {
        type: 'doughnut',
        data: {
            labels: ['Retained', 'Churned'],
            datasets: [{
                data: [churn.retained, churn.churned],
                backgroundColor: [COLORS.emeraldAlpha, COLORS.roseAlpha],
                borderColor: ['rgba(16,185,129,0.3)', 'rgba(244,63,94,0.3)'],
                borderWidth: 2,
                hoverOffset: 10,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '60%',
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${formatNumber(ctx.raw)} (${((ctx.raw / churn.total) * 100).toFixed(1)}%)`
                    }
                }
            }
        }
    });

    // Risk Pie
    new Chart(document.getElementById('riskPieChart'), {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [risk.LOW || 0, risk.MEDIUM || 0, risk.HIGH || 0],
                backgroundColor: [COLORS.emeraldAlpha, COLORS.amberAlpha, COLORS.roseAlpha],
                borderColor: ['rgba(16,185,129,0.3)', 'rgba(245,158,11,0.3)', 'rgba(244,63,94,0.3)'],
                borderWidth: 2,
                hoverOffset: 10,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '60%',
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const total = (risk.LOW || 0) + (risk.MEDIUM || 0) + (risk.HIGH || 0);
                            return `${ctx.label}: ${formatNumber(ctx.raw)} (${((ctx.raw / total) * 100).toFixed(1)}%)`;
                        }
                    }
                }
            }
        }
    });

    // Revenue bar
    new Chart(document.getElementById('revenueBarChart'), {
        type: 'bar',
        data: {
            labels: ['Retained Customers', 'Churned Customers'],
            datasets: [{
                label: 'Total Revenue ($)',
                data: [rev.retained_total_revenue, rev.churned_total_revenue],
                backgroundColor: [COLORS.emeraldAlpha, COLORS.roseAlpha],
                borderColor: [COLORS.emerald, COLORS.rose],
                borderWidth: 1,
                borderRadius: 8,
                barPercentage: 0.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: ctx => `$${formatNumber(ctx.raw)}` }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => '$' + formatCompact(v) }
                }
            }
        }
    });

    // Charge distribution
    const chargeDist = rev.charge_distribution;
    new Chart(document.getElementById('chargeDistChart'), {
        type: 'bar',
        data: {
            labels: Object.keys(chargeDist),
            datasets: [{
                label: 'Customers',
                data: Object.values(chargeDist),
                backgroundColor: [
                    COLORS.emeraldAlpha, COLORS.cyanAlpha, COLORS.indigoAlpha,
                    COLORS.violetAlpha, COLORS.amberAlpha
                ],
                borderRadius: 8,
                barPercentage: 0.6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                }
            }
        }
    });
}

// ========================
// SEGMENTS
// ========================
function renderSegments() {
    const seg = DATA.segment_analysis;
    const tenure = DATA.tenure_analysis;
    const services = DATA.services_analysis;

    // Contract chart
    new Chart(document.getElementById('contractChart'), {
        type: 'bar',
        data: {
            labels: seg.contract.map(c => c.contract),
            datasets: [
                {
                    label: 'Churn Rate (%)',
                    data: seg.contract.map(c => c.churn_rate),
                    backgroundColor: [COLORS.roseAlpha, COLORS.amberAlpha, COLORS.emeraldAlpha],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `${ctx.raw}% churn rate` } }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 50,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' }
                },
                y: { grid: { display: false } }
            }
        }
    });

    // Tenure chart
    new Chart(document.getElementById('tenureChart'), {
        type: 'bar',
        data: {
            labels: tenure.groups.map(g => g.group),
            datasets: [{
                label: 'Churn Rate (%)',
                data: tenure.groups.map(g => g.churn_rate),
                backgroundColor: tenure.groups.map((_, i) => {
                    const colors = [COLORS.roseAlpha, COLORS.amberAlpha, COLORS.amberAlpha, COLORS.emeraldAlpha, COLORS.emeraldAlpha];
                    return colors[i];
                }),
                borderRadius: 6,
                barPercentage: 0.7,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `${ctx.raw}% churn rate` } }
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' }
                }
            }
        }
    });

    // Internet service
    if (services.internet_service) {
        new Chart(document.getElementById('internetChart'), {
            type: 'bar',
            data: {
                labels: services.internet_service.map(s => s.service),
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: services.internet_service.map(s => s.churn_rate),
                    backgroundColor: [COLORS.cyanAlpha, COLORS.roseAlpha, COLORS.emeraldAlpha],
                    borderRadius: 6,
                    barPercentage: 0.6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `${ctx.raw}% churn rate` } }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { callback: v => v + '%' }
                    }
                }
            }
        });
    }

    // Payment method
    if (services.payment_method) {
        new Chart(document.getElementById('paymentChart'), {
            type: 'bar',
            data: {
                labels: services.payment_method.map(s => s.method),
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: services.payment_method.map(s => s.churn_rate),
                    backgroundColor: services.payment_method.map(s =>
                        s.churn_rate > 40 ? COLORS.roseAlpha :
                            s.churn_rate > 20 ? COLORS.amberAlpha : COLORS.emeraldAlpha
                    ),
                    borderRadius: 8,
                    barPercentage: 0.6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { maxRotation: 20 } },
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { callback: v => v + '%' }
                    }
                }
            }
        });
    }

    // Tenure table
    const tbody = document.querySelector('#tenureTable tbody');
    tbody.innerHTML = tenure.groups.map(g => `
    <tr>
      <td>${g.group}</td>
      <td>${formatNumber(g.count)}</td>
      <td><span class="risk-badge ${g.churn_rate > 40 ? 'high' : g.churn_rate > 20 ? 'medium' : 'low'}">${g.churn_rate}%</span></td>
      <td>$${g.avg_monthly.toFixed(2)}</td>
    </tr>
  `).join('');
}

// ========================
// TOP CUSTOMERS TABLE
// ========================
function renderTopCustomers() {
    const customers = DATA.model.top_customers;
    const tbody = document.querySelector('#customersTable tbody');

    tbody.innerHTML = customers.map((c, i) => {
        const riskLevel = c.churn_probability >= 0.7 ? 'high' : c.churn_probability >= 0.4 ? 'medium' : 'low';
        const riskLabel = riskLevel.toUpperCase();

        return `
      <tr>
        <td>${i + 1}</td>
        <td style="font-family: 'Courier New', monospace; font-size: 0.8rem;">${c.customer_id}</td>
        <td>${(c.churn_probability * 100).toFixed(1)}%</td>
        <td><span class="risk-badge ${riskLevel}">${riskLabel}</span></td>
        <td>$${c.monthly_charges.toFixed(2)}</td>
        <td>${c.tenure} months</td>
        <td>$${formatNumber(c.revenue)}</td>
        <td style="color: #fb7185; font-weight: 600;">$${formatNumber(c.expected_loss)}</td>
      </tr>
    `;
    }).join('');
}

// ========================
// PIPELINE
// ========================
function renderPipeline() {
    const flow = document.getElementById('pipelineFlow');

    const steps = [
        { icon: '📁', name: 'Raw Data', file: 'data/raw/', bg: 'rgba(139,92,246,0.15)' },
        { icon: '📥', name: 'Ingestion', file: 'ingestion.py', bg: 'rgba(6,182,212,0.15)' },
        { icon: '🧹', name: 'Cleaning', file: 'cleaning.py', bg: 'rgba(16,185,129,0.15)' },
        { icon: '⚙️', name: 'Features', file: 'features.py', bg: 'rgba(245,158,11,0.15)' },
        { icon: '🤖', name: 'Training', file: 'train.py', bg: 'rgba(99,102,241,0.15)' },
        { icon: '📊', name: 'Evaluation', file: 'evaluate.py', bg: 'rgba(16,185,129,0.15)' },
        { icon: '💡', name: 'Insights', file: 'business_insights.py', bg: 'rgba(244,63,94,0.15)' },
        { icon: '🌐', name: 'Dashboard', file: 'dashboard.html', bg: 'rgba(14,165,233,0.15)' },
    ];

    flow.innerHTML = steps.map((s, i) => `
    ${i > 0 ? '<div class="pipeline-arrow">→</div>' : ''}
    <div class="pipeline-step">
      <div class="step-icon" style="background: ${s.bg}">${s.icon}</div>
      <div class="step-name">${s.name}</div>
      <div class="step-file">${s.file}</div>
    </div>
  `).join('');
}

// ========================
// SCROLL REVEAL
// ========================
function initScrollReveal() {
    const sections = document.querySelectorAll('.section');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    sections.forEach(s => observer.observe(s));
}

// ========================
// UTILITIES
// ========================
function formatNumber(num) {
    if (num === null || num === undefined) return '—';
    return Number(num).toLocaleString('en-US', { maximumFractionDigits: 2 });
}

function formatCompact(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toFixed(2);
}
