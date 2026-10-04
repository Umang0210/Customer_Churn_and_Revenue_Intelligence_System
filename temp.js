
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// CONFIG â€” API base URL
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const API_BASE = window.location.origin;

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// CHART DEFAULTS
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = 'rgba(255,255,255,0.03)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 14;

const C = {
  indigo:'#6366f1', violet:'#8b5cf6', cyan:'#06b6d4',
  emerald:'#10b981', amber:'#f59e0b', rose:'#f43f5e', sky:'#0ea5e9',
  indigoA:'rgba(99,102,241,.7)', violetA:'rgba(139,92,246,.7)',
  cyanA:'rgba(6,182,212,.7)', emeraldA:'rgba(16,185,129,.7)',
  amberA:'rgba(245,158,11,.7)', roseA:'rgba(244,63,94,.7)',
};

let DATA = null;
const charts = {};

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// BOOT
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.addEventListener('DOMContentLoaded', async () => {
  try {
    if (window.DASHBOARD_DATA) {
      DATA = window.DASHBOARD_DATA;
    } else {
      const paths = ['dashboard_data.json', '/dashboard_data.json', '/static/dashboard_data.json'];
      for (const p of paths) {
        try { const r = await fetch(p, {signal: AbortSignal.timeout(3000)}); if (r.ok) { DATA = await r.json(); break; } } catch(e) {}
      }
    }
    if (!DATA) throw new Error('dashboard_data.js not found in static/. Make sure the file exists.');
    renderOverview();
    hideLoader();
    initScroll();
    hydrateFromAPI();
  } catch(e) {
    document.getElementById('loadingOverlay').innerHTML =
      `<div style="text-align:center;color:#f43f5e;padding:32px;max-width:440px">
        <div style="font-size:2.5rem;margin-bottom:12px">âš </div>
        <p style="font-size:1rem;font-weight:700;margin-bottom:8px">Failed to load dashboard data</p>
        <p style="color:#8b949e;font-size:.83rem;margin-bottom:14px">${e.message}</p>
        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:14px;text-align:left;font-size:.77rem;color:#8b949e;line-height:1.8">
          <strong style="color:#e6edf3;display:block;margin-bottom:4px">Fix checklist:</strong>
          1. Is src/webapp/static/dashboard_data.js present?<br>
          2. Run python run_pipeline.py first to generate data<br>
          3. Both servers running? API :5000 and Dashboard :8000
        </div>
      </div>`;
  }

  // Nav
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = btn.dataset.page;
      navigateTo(page);
    });
  });

  // Upload init
  initUpload();
  loadHistory();
  checkPipelineStatus();
});

function hideLoader() {
  const ov = document.getElementById('loadingOverlay');
  ov.classList.add('hidden');
  setTimeout(() => ov.remove(), 500);
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// NAVIGATION
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function navigateTo(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + pageId)?.classList.add('active');
  document.querySelector(`[data-page="${pageId}"]`)?.classList.add('active');

  // Lazy render pages on first visit
  if (pageId === 'model'     && !charts.radar)     renderModel();
  if (pageId === 'analytics' && !charts.revenue)   renderAnalytics();
  if (pageId === 'customers' && !charts.custLoaded) renderCustomers();
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// LIVE API HYDRATION
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function hydrateFromAPI() {
  try {
    const summary = await fetch(API_BASE + '/api/dashboard/summary', {signal: AbortSignal.timeout(3000)}).then(r=>r.json());
    if (summary.total_predictions) {
      // Update KPI cards with live data
      updateKPILive(summary);
    }
    const version = await fetch(API_BASE + '/health', {signal: AbortSignal.timeout(2000)}).then(r=>r.json());
    if (version.model_name) {
      document.getElementById('versionBadge').textContent = `${version.model_version || 'v2.0'} â€” ${version.model_name}`;
    }
  } catch(e) { /* API not running - static data used */ }
}

function updateKPILive(summary) {
  // Subtly update values if API returns fresher data
  const atRiskEl = document.querySelector('[data-live="rev_at_risk"]');
  const hiRiskEl = document.querySelector('[data-live="high_risk"]');
  const churnRtEl = document.querySelector('[data-live="churn_rate"]');
  if (atRiskEl) atRiskEl.textContent = '$' + compact(summary.revenue_at_risk);
  if (hiRiskEl) hiRiskEl.textContent = fmt(summary.high_risk_count);
  if (churnRtEl) churnRtEl.textContent = (summary.avg_churn_probability * 100).toFixed(2) + '%';
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// OVERVIEW PAGE
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderOverview() {
  const el = document.getElementById('generatedAt');
  if (el) el.textContent = 'Report generated: ' + DATA.generated_at;

  // KPI Cards
  const k = DATA.model.business_kpis;
  const ch = DATA.churn_distribution;
  const m = DATA.model.evaluation;
  const grid = document.getElementById('kpiGrid');
  if (grid) {
    const cards = [
      { label:'Total Customers', value:fmt(k.total_customers), sub:`${DATA.dataset_overview.columns} features`, cls:'neutral', color:'indigo', icon:'<i data-lucide="users" class="icon-sm"></i>' },
      { label:'Churn Rate', value:`${ch.churn_rate}%`, sub:`${fmt(ch.churned)} churned`, cls:'negative', color:'rose', icon:'<i data-lucide="alert-triangle" class="icon-sm"></i>', live:'churn_rate' },
      { label:'Revenue at Risk', value:`$${compact(k.revenue_at_risk)}`, sub:`${k.revenue_at_risk_pct}% of total`, cls:'negative', color:'amber', icon:'<i data-lucide="dollar-sign" class="icon-sm"></i>', live:'rev_at_risk' },
      { label:'Model ROC-AUC', value:m.roc_auc.toFixed(4), sub:`${DATA.model.selected_model.replace(/_/g,' ')} - ${DATA.model.model_version}`, cls:'positive', color:'emerald', icon:'<i data-lucide="activity" class="icon-sm"></i>' },
      { label:'High-Risk Customers', value:`${k.high_risk_pct}%`, sub:`${fmt(DATA.model.risk_distribution.HIGH||0)} > 70% prob`, cls:'negative', color:'rose', icon:'<i data-lucide="alert-circle" class="icon-sm"></i>', live:'high_risk' },
      { label:'Total Revenue', value:`$${compact(k.total_revenue)}`, sub:`Avg $${DATA.revenue_analysis.avg_monthly_charges}/mo`, cls:'neutral', color:'cyan', icon:'<i data-lucide="credit-card" class="icon-sm"></i>' }
    ];
    grid.innerHTML = cards.map(c=>`
      <div class="kpi-card ${c.color} animate-in">
        <div class="kpi-icon">${c.icon}</div>
        <div class="kpi-label">${c.label}</div>
        <div class="kpi-value"${c.live ? ` data-live="${c.live}"`:``}>${c.value}</div>
        <div class="kpi-change ${c.cls}">${c.sub}</div>
      </div>`).join('');
  }

  // Churn pie
  const churnEl = document.getElementById('churnPieChart');
  if (churnEl) new Chart(churnEl, {
    type:'doughnut',
    data:{ labels:['Retained','Churned'], datasets:[{ data:[ch.retained, ch.churned], backgroundColor:[C.emeraldA, C.roseA], borderColor:['rgba(16,185,129,.25)','rgba(244,63,94,.25)'], borderWidth:2, hoverOffset:8 }] },
    options:{ responsive:true, maintainAspectRatio:true, cutout:'62%', plugins:{ legend:{position:'bottom'}, tooltip:{ callbacks:{ label:ctx=>`${ctx.label}: ${fmt(ctx.raw)} (${(ctx.raw/ch.total*100).toFixed(1)}%)` } } } }
  });

  // Risk pie
  const riskEl = document.getElementById('riskPieChart');
  if (riskEl) {
    const risk = DATA.model.risk_distribution;
    const total = (risk.LOW||0)+(risk.MEDIUM||0)+(risk.HIGH||0);
    new Chart(riskEl, {
      type:'doughnut',
      data:{ labels:['Low Risk','Medium Risk','High Risk'], datasets:[{ data:[risk.LOW||0, risk.MEDIUM||0, risk.HIGH||0], backgroundColor:[C.emeraldA, C.amberA, C.roseA], borderColor:['rgba(16,185,129,.25)','rgba(245,158,11,.25)','rgba(244,63,94,.25)'], borderWidth:2, hoverOffset:8 }] },
      options:{ responsive:true, maintainAspectRatio:true, cutout:'62%', plugins:{ legend:{position:'bottom'}, tooltip:{ callbacks:{ label:ctx=>`${ctx.label}: ${fmt(ctx.raw)} (${(ctx.raw/total*100).toFixed(1)}%)` } } } }
    });
  }

  // Pipeline
  const flow = document.getElementById('pipelineFlow');
  if (flow) {
    const steps = [
      {name:'Data Ingestion', file:'ingestion.py', bg:C.indigo, icon:'<i data-lucide="database" class="icon-sm"></i>'},
      {name:'Data Cleaning', file:'cleaning.py', bg:C.violet, icon:'<i data-lucide="brush" class="icon-sm"></i>'},
      {name:'Feature Engineering', file:'features.py', bg:C.cyan, icon:'<i data-lucide="settings" class="icon-sm"></i>'},
      {name:'EDA', file:'eda.py', bg:C.amber, icon:'<i data-lucide="pie-chart" class="icon-sm"></i>'},
      {name:'Model Training', file:'train.py', bg:C.rose, icon:'<i data-lucide="cpu" class="icon-sm"></i>'},
      {name:'Evaluation', file:'evaluate.py', bg:C.emerald, icon:'<i data-lucide="check-circle" class="icon-sm"></i>'},
      {name:'Predictions', file:'persist_insights.py', bg:C.sky, icon:'<i data-lucide="save" class="icon-sm"></i>'},
      {name:'Business Insights', file:'business_insights.py', bg:C.indigo, icon:'<i data-lucide="trending-up" class="icon-sm"></i>'}
    ];
    flow.innerHTML = steps.map((s,i)=>`
      ${i>0?'<div class="pipeline-arrow">â†’</div>':''}
      <div class="pipeline-step">
        <div class="step-icon" style="background:${s.bg}">${s.icon}</div>
        <div class="step-name">${s.name}</div>
        <div class="step-file">${s.file}</div>
      </div>`).join('');
  }
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// MODEL PAGE
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderModel() {
  const comp = DATA.model.model_comparison;
  const selected = DATA.model.selected_model;
  const cardsEl = document.getElementById('modelCards');
  if (cardsEl) {
    const models = Object.entries(comp).map(([name,m])=>({ name:name.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()), key:name, isSelected:name===selected, metrics:m }));
    cardsEl.innerHTML = models.map(m=>`
      <div class="model-card ${m.isSelected?'selected':''}">
        <div class="model-name">${m.name}</div>
        ${metricBar('ROC-AUC', m.metrics.roc_auc, 'indigo')}
        ${metricBar('Precision', m.metrics.precision, 'emerald')}
        ${metricBar('Recall', m.metrics.recall, 'amber')}
      </div>`).join('');
    requestAnimationFrame(()=>{
      document.querySelectorAll('.bar-fill').forEach(b=>{ b.style.width = b.dataset.w+'%'; });
    });
  }

  // Radar
  const ev = DATA.model.evaluation;
  const radarEl = document.getElementById('radarChart');
  if (radarEl) {
    charts.radar = new Chart(radarEl, {
      type:'radar',
      data:{ labels:['ROC-AUC','Precision','Recall','F1 Score'],
        datasets:[
          { label:'Random Forest', data:[comp.random_forest.roc_auc, comp.random_forest.precision, comp.random_forest.recall, ev.f1_score], borderColor:C.indigo, backgroundColor:'rgba(99,102,241,.08)', borderWidth:2, pointBackgroundColor:C.indigo },
          { label:'Logistic Regression', data:[comp.logistic_regression.roc_auc, comp.logistic_regression.precision, comp.logistic_regression.recall, null], borderColor:C.violet, backgroundColor:'rgba(139,92,246,.06)', borderWidth:2, pointBackgroundColor:C.violet }
        ] },
      options:{ responsive:true, maintainAspectRatio:true,
        scales:{ r:{ beginAtZero:true, max:1, ticks:{stepSize:.2,display:false}, grid:{color:'rgba(255,255,255,.04)'}, angleLines:{color:'rgba(255,255,255,.04)'}, pointLabels:{font:{size:11,weight:'500'}} } },
        plugins:{ legend:{position:'bottom'} } }
    });
  }

  // Confusion matrix
  const cm = DATA.model.evaluation.confusion_matrix;
  document.getElementById('testSamples').textContent = DATA.model.evaluation.test_samples;
  document.getElementById('confusionMatrix').innerHTML = `
    <div class="cm-header"></div><div class="cm-header">Pred: No</div><div class="cm-header">Pred: Yes</div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Actual: No</div>
    <div class="cm-cell tn">${cm[0][0]}<span class="cm-label">True Neg</span></div>
    <div class="cm-cell fp">${cm[0][1]}<span class="cm-label">False Pos</span></div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Actual: Yes</div>
    <div class="cm-cell fn">${cm[1][0]}<span class="cm-label">False Neg</span></div>
    <div class="cm-cell tp">${cm[1][1]}<span class="cm-label">True Pos</span></div>`;

  // Threshold chart
  const th = DATA.model.thresholds;
  const thEl = document.getElementById('thresholdChart');
  if (thEl) {
    charts.threshold = new Chart(thEl, {
      type:'line',
      data:{ labels:th.map(t=>t.threshold.toFixed(1)),
        datasets:[
          { label:'Precision', data:th.map(t=>t.precision), borderColor:C.emerald, backgroundColor:'rgba(16,185,129,.08)', fill:true, tension:.4, pointRadius:4 },
          { label:'Recall',    data:th.map(t=>t.recall),    borderColor:C.amber,   backgroundColor:'rgba(245,158,11,.08)',  fill:true, tension:.4, pointRadius:4 },
          { label:'F1 Score',  data:th.map(t=>t.f1_score),  borderColor:C.indigo,  backgroundColor:'rgba(99,102,241,.08)',  fill:true, tension:.4, pointRadius:4 }
        ] },
      options:{ responsive:true, maintainAspectRatio:true, interaction:{mode:'index',intersect:false},
        scales:{ x:{grid:{display:false}}, y:{beginAtZero:true, max:1, grid:{color:'rgba(255,255,255,.03)'}} },
        plugins:{ legend:{position:'bottom'} } }
    });
  }
  // Threshold table
  const tbody = document.querySelector('#thresholdTable tbody');
  if (tbody) tbody.innerHTML = th.map(t=>`<tr${t.threshold===.5?' style="background:rgba(99,102,241,.06)"':''}><td>${t.threshold.toFixed(1)}${t.threshold===.5?' *':''}</td><td>${(t.precision*100).toFixed(2)}%</td><td>${(t.recall*100).toFixed(2)}%</td><td>${(t.f1_score*100).toFixed(2)}%</td></tr>`).join('');

  // Features
  document.getElementById('featureCount').textContent = DATA.model.num_features;
  const tagsEl = document.getElementById('featureTags');
  if (tagsEl) tagsEl.innerHTML = DATA.model.feature_list.map((f,i)=>`<div class="feature-tag"><span class="tag-number">${i+1}</span>${f}</div>`).join('');
}

function metricBar(label, val, color) {
  const pct = (val*100).toFixed(1);
  return `<div class="model-metric">
    <span class="metric-label">${label}</span>
    <div class="metric-span" style="display:flex;align-items:center;gap:8px">
      <div class="bar-track"><div class="bar-fill ${color}" data-w="${pct}" style="width:0"></div></div>
      <span class="metric-value">${pct}%</span>
    </div></div>`;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// ANALYTICS PAGE
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderAnalytics() {
  const rev = DATA.revenue_analysis;
  const seg = DATA.segment_analysis;
  const tenure = DATA.tenure_analysis;
  const svc = DATA.services_analysis;

  charts.revenue = new Chart(document.getElementById('revenueBarChart'), {
    type:'bar',
    data:{ labels:['Retained Customers','Churned Customers'],
      datasets:[{ label:'Total Revenue ($)', data:[rev.retained_total_revenue, rev.churned_total_revenue], backgroundColor:[C.emeraldA, C.roseA], borderColor:[C.emerald, C.rose], borderWidth:1, borderRadius:8, barPercentage:.5 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}, tooltip:{callbacks:{label:ctx=>'$'+fmt(ctx.raw)}}}, scales:{ x:{grid:{display:false}}, y:{beginAtZero:true, grid:{color:'rgba(255,255,255,.03)'}, ticks:{callback:v=>'$'+compact(v)}} } }
  });

  const dist = rev.charge_distribution;
  new Chart(document.getElementById('chargeDistChart'), {
    type:'bar',
    data:{ labels:Object.keys(dist), datasets:[{ label:'Customers', data:Object.values(dist), backgroundColor:[C.emeraldA, C.cyanA, C.indigoA, C.violetA, C.amberA], borderRadius:8, barPercentage:.6 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false}}, y:{beginAtZero:true, grid:{color:'rgba(255,255,255,.03)'}} } }
  });

  new Chart(document.getElementById('contractChart'), {
    type:'bar',
    data:{ labels:seg.contract.map(c=>c.contract), datasets:[{ label:'Churn Rate (%)', data:seg.contract.map(c=>c.churn_rate), backgroundColor:[C.roseA, C.amberA, C.emeraldA], borderRadius:8, barPercentage:.6 }] },
    options:{ responsive:true, maintainAspectRatio:true, indexAxis:'y', plugins:{legend:{display:false}}, scales:{ x:{beginAtZero:true, max:50, grid:{color:'rgba(255,255,255,.03)'}, ticks:{callback:v=>v+'%'}}, y:{grid:{display:false}} } }
  });

  new Chart(document.getElementById('tenureChart'), {
    type:'bar',
    data:{ labels:tenure.groups.map(g=>g.group), datasets:[{ label:'Churn Rate (%)', data:tenure.groups.map(g=>g.churn_rate), backgroundColor:[C.roseA, C.amberA, C.amberA, C.emeraldA, C.emeraldA], borderRadius:6, barPercentage:.7 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false}}, y:{beginAtZero:true, grid:{color:'rgba(255,255,255,.03)'}, ticks:{callback:v=>v+'%'}} } }
  });

  if (svc.internet_service) new Chart(document.getElementById('internetChart'), {
    type:'bar',
    data:{ labels:svc.internet_service.map(s=>s.service), datasets:[{ label:'Churn Rate (%)', data:svc.internet_service.map(s=>s.churn_rate), backgroundColor:[C.cyanA, C.roseA, C.emeraldA], borderRadius:6, barPercentage:.6 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false}}, y:{beginAtZero:true, grid:{color:'rgba(255,255,255,.03)'}, ticks:{callback:v=>v+'%'}} } }
  });

  if (svc.payment_method) new Chart(document.getElementById('paymentChart'), {
    type:'bar',
    data:{ labels:svc.payment_method.map(s=>s.method), datasets:[{ label:'Churn Rate (%)', data:svc.payment_method.map(s=>s.churn_rate), backgroundColor:svc.payment_method.map(s=>s.churn_rate>40?C.roseA:s.churn_rate>20?C.amberA:C.emeraldA), borderRadius:8, barPercentage:.6 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false}, ticks:{maxRotation:15}}, y:{beginAtZero:true, grid:{color:'rgba(255,255,255,.03)'}, ticks:{callback:v=>v+'%'}} } }
  });

  const tbody = document.querySelector('#tenureTable tbody');
  if (tbody) tbody.innerHTML = tenure.groups.map(g=>`
    <tr><td>${g.group}</td><td>${fmt(g.count)}</td>
    <td><span class="risk-badge ${g.churn_rate>40?'high':g.churn_rate>20?'medium':'low'}">${g.churn_rate}%</span></td>
    <td>$${g.avg_monthly.toFixed(2)}</td></tr>`).join('');
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// CUSTOMERS PAGE
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderCustomers() {
  charts.custLoaded = true;

  // Try live API first
  fetch(API_BASE + '/api/dashboard/priority_customers?limit=20', {signal: AbortSignal.timeout(3000)})
    .then(r=>r.json())
    .then(data => renderCustomerTable(data, true))
    .catch(() => renderCustomerTable(DATA.model.top_customers, false));

  const ds = DATA.dataset_overview;
  const statsEl = document.getElementById('datasetStats');
  if (statsEl) {
    const stats = [
      {value:fmt(ds.rows), label:'Total Rows'},
      {value:ds.columns, label:'Columns'},
      {value:ds.duplicate_rows||0, label:'Duplicates'},
      {value:`${ds.missing_pct}%`, label:'Missing Values'},
      {value:`${ds.memory_mb||0} MB`, label:'Memory'},
    ];
    statsEl.innerHTML = stats.map(s=>`<div class="overview-stat"><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`).join('');
  }

  const gridEl = document.getElementById('columnsGrid');
  if (gridEl) gridEl.innerHTML = ds.columns_info.map(col=>`
    <div class="column-item">
      <span class="column-dtype">${col.dtype}</span>
      <span class="column-name">${col.name}</span>
      <span class="column-unique">${col.unique} unique</span>
    </div>`).join('');
}

function renderCustomerTable(customers, isLive) {
  const tbody = document.querySelector('#customersTable tbody');
  if (!tbody) return;

  if (isLive) {
    // Live API format
    tbody.innerHTML = customers.map((c,i) => {
      const risk = (c.risk_bucket||'low').toLowerCase();
      return `<tr>
        <td>${i+1}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:.76rem">${c.customer_id}</td>
        <td>${((c.churn_probability||0)*100).toFixed(1)}%</td>
        <td><span class="risk-badge ${risk}">${risk.toUpperCase()}</span></td>
        <td>$${(c.revenue||0).toFixed(2)}</td>
        <td>â€”</td>
        <td>$${fmt(c.revenue||0)}</td>
        <td style="color:#fb7185;font-weight:600">$${fmt(c.expected_revenue_loss||0)}</td>
      </tr>`;
    }).join('');
  } else {
    // Static data format
    tbody.innerHTML = customers.map((c,i) => {
      const risk = c.churn_probability>=.7?'high':c.churn_probability>=.4?'medium':'low';
      return `<tr>
        <td>${i+1}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:.76rem">${c.customer_id}</td>
        <td>${(c.churn_probability*100).toFixed(1)}%</td>
        <td><span class="risk-badge ${risk}">${risk.toUpperCase()}</span></td>
        <td>$${(c.monthly_charges||0).toFixed(2)}</td>
        <td>${c.tenure||'â€”'} mo</td>
        <td>$${fmt(c.revenue||0)}</td>
        <td style="color:#fb7185;font-weight:600">$${fmt(c.expected_loss||0)}</td>
      </tr>`;
    }).join('');
  }
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// UPLOAD
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const STEP_NAMES = ['','Data Ingestion','Data Cleaning','Feature Engineering','EDA','Model Training','Model Evaluation','Batch Predictions','Business Insights','Export Dashboard JSON'];
let pollInterval = null;
let selectedFile = null;

function initUpload() {
  const dz = document.getElementById('dropZone');
  const fi = document.getElementById('fileInput');

  dz.addEventListener('dragover', e=>{ e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', ()=>dz.classList.remove('dragover'));
  dz.addEventListener('drop', e=>{ e.preventDefault(); dz.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
  fi.addEventListener('change', ()=>{ if(fi.files[0]) handleFile(fi.files[0]); });
}

function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['csv','xlsx','xls'].includes(ext)) { showVal('error','Unsupported file type. Use CSV or Excel.'); return; }
  if (file.size > 50*1024*1024) { showVal('error','File too large. Max 50MB.'); return; }
  selectedFile = file;
  const preview = document.getElementById('filePreview');
  preview.classList.add('show');
  document.getElementById('previewIcon').innerHTML = ext==='csv'?'<i data-lucide="file-text" class="icon-sm"></i>':'<i data-lucide="bar-chart-2" class="icon-sm"></i>';
  document.getElementById('previewName').textContent = file.name;
  document.getElementById('previewMeta').textContent = `${(file.size/1024).toFixed(1)} KB Â· ${ext.toUpperCase()}`;
  document.getElementById('uploadBtn').disabled = false;
  hideVal();
}

function clearFile() {
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('filePreview').classList.remove('show');
  document.getElementById('uploadBtn').disabled = true;
  hideVal();
}

async function uploadFile() {
  if (!selectedFile) return;
  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin-sm"></span> Uploadingâ€¦';

  const formData = new FormData();
  formData.append('file', selectedFile);

  const url = API_BASE + '/api/upload/dataset';
  let response = null;
  let lastErr = null;

  try {
    response = await fetch(url, { method:'POST', body:formData });
  } catch(e) {
    lastErr = e;
  }

  if (!response) {
    showVal('error', 'Unable to connect to the API.');
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="rocket" class="icon-sm"></i> Upload & Run Pipeline';
    return;
  }

  const data = await response.json().catch(() => ({}));
    if (data && data.run_id) window.currentPipelineRunId = data.run_id;
  
  if (!response.ok) {
    if (response.status === 404) {
      showVal('error', 'Upload API endpoint not found. Please check the API configuration.');
    } else if (response.status === 500) {
      showVal('error', 'Server error while processing the file. Please try again.');
    } else if (response.status === 422 || response.status === 400) {
      const detail = data.detail;
      if (typeof detail === 'object' && detail.errors) {
        showVal('error', 'Invalid file format or required columns are missing.', detail.errors, detail.warnings||[]);
      } else {
        showVal('error', 'Invalid file format or required columns are missing.', [typeof detail === 'string' ? detail : JSON.stringify(detail)]);
      }
    } else {
      showVal('error', data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : 'Upload failed.');
    }
    btn.disabled = false;
    showVal('ok', `<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i> ${data.rows?.toLocaleString() || 'File'} uploaded. Pipeline starting.`);
    btn.innerHTML = '<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i> Uploaded - Pipeline Running';
  showToast('Pipeline started! Dashboard will update when complete.');
  document.getElementById('pipelineProgress').classList.add('show');
  startPolling();
}

async function checkPipelineStatus() {
  try {
    const data = await fetch(API_BASE + '/api/upload/status', {signal:AbortSignal.timeout(2000)}).then(r=>r.json());
      if (data && data.run_id) window.currentPipelineRunId = data.run_id;
    if (data.status === 'running') {
      document.getElementById('pipelineProgress').classList.add('show');
      updateProgress(data);
      startPolling();
    }
  } catch(e) {}
}

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const data = await fetch(API_BASE + '/api/upload/status', {signal:AbortSignal.timeout(3000)}).then(r=>r.json());
        if (data && data.run_id) window.currentPipelineRunId = data.run_id;
      updateProgress(data);
      if (data.status === 'success' || data.status === 'failed') {
        clearInterval(pollInterval); pollInterval = null;
        loadHistory();
        if (data.status === 'success') {
          showToast('âœ… Pipeline complete! Refresh the page to see updated data.');
          document.getElementById('uploadBtn').disabled = false;
          document.getElementById('uploadBtn').innerHTML = '<i data-lucide="rocket" class="icon-sm"></i> Upload & Run Pipeline';
        }
      }
    } catch(e) {}
  }, 2500);
}

function updateProgress(data) {
  const pct = data.progress_pct || 0;
  const step = data.step || 0;
  document.getElementById('progBar').style.width = pct + '%';
  document.getElementById('progPct').textContent = pct + '%';
  document.getElementById('progText').textContent = data.message || '';
  renderStepCards(step, data.status);
  const msg = document.getElementById('statusMsg');
    msg.className = `status-msg ${data.status==='running'?'running':data.status==='success'?'success':'failed'}`;
    
    // If progress is 100% but status is failed
    if (data.status === 'failed' && pct === 100) {
        document.getElementById('progBar').style.background = 'var(--rose)';
    } else {
        document.getElementById('progBar').style.background = 'var(--gi)';
    }
  const spin = data.status==='running'?'<span class="spin-sm"></span>':'';
  msg.innerHTML = spin + ' ' + (data.message||'');
}

function renderStepCards(currentStep, pipelineStatus) {
  const container = document.getElementById('stepCards');
  container.innerHTML = '';
  for (let i=1; i<=9; i++) {
    const div = document.createElement('div');
      let state = 'pending', icon = '';
      if (i < currentStep) { state='success'; icon='<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>'; }
      else if (i === currentStep) {
        if (pipelineStatus==='running') { state='running'; icon='<i data-lucide="loader-2" class="icon-sm spin"></i>'; }
        else if (pipelineStatus==='success') { state='success'; icon='<i data-lucide="check-circle-2" class="icon-sm text-emerald"></i>'; }
        else if (pipelineStatus==='failed') { state='failed'; icon='<i data-lucide="x-circle" class="icon-sm text-rose"></i>'; }
    }
    div.className = `step-card ${state}`;
    div.innerHTML = `<div class="snum">Step ${i}</div><div class="sname">${STEP_NAMES[i]}</div><span class="sico">${icon}</span>`;
    container.appendChild(div);
  }
}

async function loadHistory() {
  try {
    const data = await fetch(API_BASE + '/api/upload/history', {signal:AbortSignal.timeout(2000)}).then(r=>r.json());
    renderHistory(data);
  } catch(e) {
    console.error('Failed to load history:', e);
  }
}

function renderHistory(items) {
  const tbody = document.getElementById('historyBody');
  if (!items || items.length===0) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--tx3);padding:16px">No uploads yet</td></tr>';
    return;
  }
  tbody.innerHTML = items.map(item=>{
    const badge = item.status==='success'
      ? '<span class="risk-badge low">Success</span>'
      : '<span class="risk-badge high">Failed</span>';
    const dt = item.uploaded_at ? new Date(item.uploaded_at).toLocaleString() : 'â€”';
    return `<tr><td style="font-size:.78rem">${item.filename||'â€”'}</td><td>${badge}</td><td style="color:var(--tx2);font-size:.75rem">${dt}</td></tr>`;
  }).join('');
}

async function downloadTemplate() {
  const urls = [
    `http://${window.location.hostname}:5000/api/upload/template`,
    `${window.location.origin}/api/upload/template`,
  ];
  for (const url of urls) {
    try {
      const r = await fetch(url, {signal:AbortSignal.timeout(2000)});
      if (r.ok) { window.open(url, '_blank'); return; }
    } catch(e) {}
  }
  showToast('API not running â€” template unavailable.');
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// HELPERS
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function showVal(type, msg, errors=[], warnings=[]) {
  const box = document.getElementById('valBox');
  const cls = type==='error'?'error':type==='warn'?'warn':'ok';
  let html = `<strong>${msg}</strong>`;
  if (errors.length)   html += `<ul>${errors.map(e=>`<li>${e}</li>`).join('')}</ul>`;
  if (warnings.length) html += `<ul>${warnings.map(w=>`<li>âš  ${w}</li>`).join('')}</ul>`;
  box.className = `val-box show ${cls}`;
  box.innerHTML = html;
}
function hideVal() {
  const box = document.getElementById('valBox');
  box.className = 'val-box'; box.innerHTML = '';
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 5000);
}

function fmt(n) {
  if (n===null||n===undefined) return 'â€”';
  return Number(n).toLocaleString('en-US',{maximumFractionDigits:2});
}
function compact(n) {
  if (n>=1e6) return (n/1e6).toFixed(2)+'M';
  if (n>=1e3) return (n/1e3).toFixed(1)+'K';
  return Number(n).toFixed(2);
}

function initScroll() {
  const obs = new IntersectionObserver(entries=>{
    entries.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('visible'); obs.unobserve(e.target); } });
  }, {threshold:.08, rootMargin:'0px 0px -40px 0px'});
  document.querySelectorAll('.section').forEach(s=>obs.observe(s));
}
