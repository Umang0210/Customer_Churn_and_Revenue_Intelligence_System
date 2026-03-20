# Reports & Insights Documentation

This folder contains all generated outputs from the EDA, model evaluation, and batch prediction pipelines. Every file here is **auto-generated** — do not edit manually. Re-run the pipeline scripts to regenerate.

---

## Folder Structure

```
reports/
├── figures/                          ← All plots (PNG, 150 DPI)
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_tenure_vs_churn.png
│   ├── 04_monthly_charges_boxplot.png
│   ├── 05_complaints_payment_vs_churn.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_revenue_at_risk.png
│   ├── 08_high_risk_profiles.png
│   ├── 09_roc_curve.png
│   ├── 10_confusion_matrix.png
│   ├── 11_threshold_analysis.png
│   └── 12_feature_importance.png
│
├── business_insights.csv             ← KPI summary (metric, value)
├── segment_summary.csv               ← Risk bucket breakdown
├── top_priority_customers.csv        ← Top 20 customers to target
├── batch_predictions.csv             ← Full dataset predictions
├── evaluation_report.json            ← Model metrics + gate result
└── README.md                         ← This file
```

---

## EDA Figures (`src/eda.py`)

| File | What It Shows | Business Question Answered |
|------|---------------|---------------------------|
| `01_churn_distribution.png` | Pie + bar of churned vs retained | What is our overall churn rate? |
| `02_churn_by_contract.png` | Churn rate per contract type | Which contract segment loses the most customers? |
| `03_tenure_vs_churn.png` | Tenure distribution split by churn | Do new customers churn more than long-term ones? |
| `04_monthly_charges_boxplot.png` | Monthly charges: churned vs retained | Are churned customers paying more or less? |
| `05_complaints_payment_vs_churn.png` | Churn rate by complaint count & payment delays | How strongly do service issues predict churn? |
| `06_correlation_heatmap.png` | Feature-to-feature correlations (top 20 vs churn) | Which features are most correlated with churn? |
| `07_revenue_at_risk.png` | Total revenue split: at-risk vs retained | What is the dollar value of our churn problem? |
| `08_high_risk_profiles.png` | Top features separating churned from retained | What does a high-risk customer look like? |

**Regenerate:** `python src/eda.py`

---

## Model Evaluation Figures (`src/evaluate.py`)

| File | What It Shows |
|------|---------------|
| `09_roc_curve.png` | ROC curve with AUC score — model discrimination ability |
| `10_confusion_matrix.png` | TP / FP / TN / FN at threshold = 0.5 |
| `11_threshold_analysis.png` | Precision, Recall, F1 at thresholds 0.4 / 0.5 / 0.6 / 0.7 |
| `12_feature_importance.png` | Top 15 features by importance (Random Forest / XGBoost) |

**Regenerate:** `python src/evaluate.py`

---

## CSV Outputs

### `business_insights.csv`
Flat key-value table of 20+ business KPIs.

| Column | Description |
|--------|-------------|
| `metric` | KPI name (e.g. `churn_rate_%`, `revenue_at_risk_$`) |
| `value` | Computed value |

Key metrics included:
- Overall churn rate (%)
- Revenue at risk ($) and as a % of total revenue
- Churn rate by contract type (highest + lowest)
- Early vs late tenure churn ratio
- Churn rate segmented by complaints and payment delays
- Revenue uplift opportunity (15% retention scenario)

**Regenerate:** `python src/business_insights.py`

---

### `segment_summary.csv`
Risk bucket breakdown — the core of the prescriptive intelligence layer.

| Column | Description |
|--------|-------------|
| `risk_bucket` | Low / Medium / High |
| `customer_count` | Number of customers in bucket |
| `avg_churn_probability_%` | Average predicted churn probability |
| `total_revenue_at_risk_$` | Sum of expected revenue loss in bucket |
| `avg_priority_score` | Average priority score (if predictions available) |

**Regenerate:** `python src/business_insights.py`

---

### `top_priority_customers.csv`
Top 20 customers ranked by Priority Score for targeted retention action.

**Priority Score Formula (from project spec):**
```
Priority Score = churn_probability × expected_revenue_loss
expected_revenue_loss = churn_probability × customer_revenue
```

This answers: *"If we can't save everyone, who should we focus on first?"*

| Column | Description |
|--------|-------------|
| `rank` | Priority rank (1 = highest urgency) |
| `customer_id` | Customer identifier |
| `churn_probability` | Model predicted churn probability |
| `risk_bucket` | Low / Medium / High |
| `revenue` | Customer's total/monthly revenue |
| `expected_revenue_loss` | Revenue at risk ($) |
| `priority_score` | Composite retention priority score |

**Regenerate:** `python src/business_insights.py`

---

### `batch_predictions.csv`
Full dataset predictions from `src/persist_insights.py`. Contains all columns above for every customer. Also synced to MySQL table `customer_churn_analytics`.

**Regenerate:** `python src/persist_insights.py`

---

### `evaluation_report.json`
Machine-readable evaluation output consumed by the CI/CD pipeline.

```json
{
  "model_name": "XGBoost",
  "evaluated_at": "2025-...",
  "roc_auc": 0.8712,
  "passes_gate": true,
  "min_auc_threshold": 0.70,
  "threshold_analysis": [
    {"threshold": 0.4, "precision": ..., "recall": ..., "f1_score": ...},
    {"threshold": 0.5, "precision": ..., "recall": ..., "f1_score": ...},
    ...
  ]
}
```

**Regenerate:** `python src/evaluate.py`

---

## How to Regenerate Everything

```bash
# Full pipeline
python src/ingestion.py
python src/cleaning.py
python src/features.py
python src/eda.py                  # → figures/01–08 + business_insights.csv
python src/train.py
python src/evaluate.py             # → figures/09–12 + evaluation_report.json
python src/persist_insights.py    # → batch_predictions.csv + MySQL
python src/business_insights.py   # → segment_summary.csv + top_priority_customers.csv
```

---

## Notes

- All figures are saved at 150 DPI and use non-interactive `Agg` backend — safe to run on any server without a display.
- The `reports/` folder (except this README) is excluded from `.gitignore` — regenerate locally after pulling the repo and running the pipeline.
- The evaluation CI/CD gate requires `roc_auc ≥ 0.70`. If it falls below, `evaluate.py` exits with code 1 and blocks the pipeline.
