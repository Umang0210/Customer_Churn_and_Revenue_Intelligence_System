"""
EDA (Exploratory Data Analysis) - Customer Churn & Revenue Intelligence
========================================================================
Step 4 of the pipeline. Runs standalone (no Jupyter required).
Outputs: plots saved to reports/figures/, insights saved to reports/business_insights.csv

Run:
    python src/eda.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "processed" / "customer_features.csv"
CLEAN_PATH  = BASE_DIR / "data" / "processed" / "clean_customers.csv"
FIGURES_DIR   = BASE_DIR / "reports" / "figures"
REPORTS_DIR   = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE_RISK   = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
COLOR_CHURN    = "#e74c3c"
COLOR_RETAIN   = "#2ecc71"
COLOR_NEUTRAL  = "#3498db"
sns.set_theme(style="whitegrid", font_scale=1.1)


# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """Load feature-engineered dataset; fall back to clean dataset."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"[EDA] Loaded feature dataset: {df.shape}")
    elif CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
        print(f"[EDA] Loaded clean dataset (features not found): {df.shape}")
    else:
        raise FileNotFoundError(
            "No processed data found. Run ingestion.py → cleaning.py → features.py first."
        )

    # Normalise target column
    if "churn" in df.columns:
        churn_col = "churn"
    elif "Churn" in df.columns:
        churn_col = "Churn"
    else:
        raise KeyError("No 'churn' or 'Churn' column found in dataset.")

    df = df.rename(columns={churn_col: "churn_flag"})

    # Convert yes/no → 1/0 if needed
    if pd.api.types.is_string_dtype(df["churn_flag"]) or df["churn_flag"].dtype == object:
        df["churn_flag"] = df["churn_flag"].str.strip().str.lower().map({"yes": 1, "no": 0})

    df["churn_flag"] = pd.to_numeric(df["churn_flag"], errors="coerce").fillna(0).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
def _revenue_col(df: pd.DataFrame) -> str:
    """Return the best available revenue column name."""
    for col in ["total_spend", "TotalCharges", "totalcharges", "revenue",
                "monthly_charges", "MonthlyCharges", "monthlycharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return col
    return None


def _contract_col(df: pd.DataFrame) -> str:
    for col in ["contract_type", "Contract", "contract"]:
        if col in df.columns:
            return col
    return None


def _tenure_col(df: pd.DataFrame) -> str:
    for col in ["tenure", "Tenure", "tenure_group"]:
        if col in df.columns:
            return col
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Churn distribution (pie + bar side by side)
# ══════════════════════════════════════════════════════════════════════════════
def plot_churn_distribution(df: pd.DataFrame):
    counts  = df["churn_flag"].value_counts().sort_index()
    labels  = ["Retained", "Churned"]
    colors  = [COLOR_RETAIN, COLOR_CHURN]
    total   = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold", y=1.01)

    # Pie
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
    axes[0].set_title("Share (%)", fontsize=12)

    # Bar
    bars = axes[1].bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{cnt:,}\n({cnt/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )
    axes[1].set_ylabel("Number of Customers")
    axes[1].set_title("Count", fontsize=12)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out = FIGURES_DIR / "01_churn_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Churn rate by contract type
# ══════════════════════════════════════════════════════════════════════════════
def plot_churn_by_contract(df: pd.DataFrame):
    col = _contract_col(df)
    if col is None:
        print("[EDA] Skipping contract plot — no contract column found.")
        return

    grp = (
        df.groupby(col)["churn_flag"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "churned", "count": "total"})
    )
    grp["churn_rate"] = grp["churned"] / grp["total"] * 100
    grp = grp.sort_values("churn_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(grp.index, grp["churn_rate"], color=COLOR_CHURN, edgecolor="white", height=0.55)
    for bar, rate, churned, total in zip(
        bars, grp["churn_rate"], grp["churned"], grp["total"]
    ):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%  ({churned:,}/{total:,})",
            va="center", fontsize=10,
        )
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xlim(0, grp["churn_rate"].max() * 1.35)
    ax.invert_yaxis()
    plt.tight_layout()
    out = FIGURES_DIR / "02_churn_by_contract.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Tenure vs churn (histogram overlay)
# ══════════════════════════════════════════════════════════════════════════════
def plot_tenure_vs_churn(df: pd.DataFrame):
    col = _tenure_col(df)
    if col is None:
        print("[EDA] Skipping tenure plot — no tenure column found.")
        return

    # If it's already a group (string), use bar chart
    if df[col].dtype == object or df[col].nunique() <= 10:
        grp = (
            df.groupby(col)["churn_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        grp.columns = [col, "churn_rate"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(grp[col], grp["churn_rate"], color=COLOR_NEUTRAL, edgecolor="white")
        ax.set_xlabel("Tenure Group")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Tenure Group", fontsize=13, fontweight="bold")
        for i, row in grp.iterrows():
            ax.text(i, row["churn_rate"] + 0.5, f"{row['churn_rate']:.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        churned  = df[df["churn_flag"] == 1][col]
        retained = df[df["churn_flag"] == 0][col]
        bins     = min(30, df[col].nunique())
        ax.hist(retained, bins=bins, alpha=0.65, color=COLOR_RETAIN, label="Retained", edgecolor="white")
        ax.hist(churned,  bins=bins, alpha=0.65, color=COLOR_CHURN,  label="Churned",  edgecolor="white")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Tenure Distribution: Churned vs Retained", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()

    out = FIGURES_DIR / "03_tenure_vs_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Monthly charges: churn vs retain (boxplot)
# ══════════════════════════════════════════════════════════════════════════════
def plot_monthly_charges_box(df: pd.DataFrame):
    charge_col = None
    for c in ["monthly_charges", "MonthlyCharges", "monthlycharges"]:
        if c in df.columns:
            charge_col = c
            break
    if charge_col is None:
        print("[EDA] Skipping monthly charges plot — column not found.")
        return

    df["_churn_label"] = df["churn_flag"].map({0: "Retained", 1: "Churned"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="_churn_label", y=charge_col,
        palette={"Retained": COLOR_RETAIN, "Churned": COLOR_CHURN},
        width=0.5, ax=ax, linewidth=1.5,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Monthly Charges: Churned vs Retained", fontsize=13, fontweight="bold")

    for label, group in df.groupby("_churn_label"):
        median = group[charge_col].median()
        x_pos  = 0 if label == "Retained" else 1
        ax.text(x_pos, median + 1, f"Median: ${median:.0f}", ha="center", fontsize=10, color="black")

    df.drop(columns=["_churn_label"], inplace=True)
    plt.tight_layout()
    out = FIGURES_DIR / "04_monthly_charges_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Complaints & payment delays vs churn (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_complaints_payment(df: pd.DataFrame):
    has_complaints = "complaints_count" in df.columns
    has_payments   = "payment_delays"   in df.columns

    if not has_complaints and not has_payments:
        print("[EDA] Skipping complaints/payment plot — columns not found.")
        return

    cols_present = []
    if has_complaints:
        cols_present.append("complaints_count")
    if has_payments:
        cols_present.append("payment_delays")

    fig, axes = plt.subplots(1, len(cols_present), figsize=(6 * len(cols_present), 5))
    if len(cols_present) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_present):
        grp = (
            df.groupby(col)["churn_flag"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grp["churn_rate"] = grp["mean"] * 100
        ax.bar(grp[col].astype(str), grp["churn_rate"],
               color=COLOR_CHURN, edgecolor="white", alpha=0.85)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title(f"Churn Rate by {col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        for i, row in grp.iterrows():
            ax.text(i, row["churn_rate"] + 0.5, f"{row['churn_rate']:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Behavioural Signals vs Churn", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "05_complaints_payment_vs_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Feature correlation heatmap (numeric only)
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 3:
        print("[EDA] Skipping correlation heatmap — too few numeric columns.")
        return

    # Keep only columns with variance
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    # Limit to 20 most correlated with churn_flag
    if "churn_flag" in numeric_df.columns and numeric_df.shape[1] > 20:
        corr_with_target = numeric_df.corr()["churn_flag"].abs().sort_values(ascending=False)
        top_cols = corr_with_target.head(20).index.tolist()
        numeric_df = numeric_df[top_cols]

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, corr.shape[0] * 0.7), max(8, corr.shape[0] * 0.6)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = FIGURES_DIR / "06_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Revenue at risk (bar: churned vs retained total spend)
# ══════════════════════════════════════════════════════════════════════════════
def plot_revenue_at_risk(df: pd.DataFrame):
    rev_col = _revenue_col(df)
    if rev_col is None:
        print("[EDA] Skipping revenue plot — no revenue column found.")
        return

    total_revenue  = df[rev_col].sum()
    churned_rev    = df[df["churn_flag"] == 1][rev_col].sum()
    retained_rev   = df[df["churn_flag"] == 0][rev_col].sum()
    churn_rate_rev = churned_rev / total_revenue * 100 if total_revenue > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Retained Revenue", "Revenue at Risk (Churned)"],
        [retained_rev, churned_rev],
        color=[COLOR_RETAIN, COLOR_CHURN],
        edgecolor="white", linewidth=1.5, width=0.5,
    )
    for bar, val in zip(bars, [retained_rev, churned_rev]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_revenue * 0.005,
            f"${val:,.0f}\n({val/total_revenue*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Total Revenue ($)")
    ax.set_title(
        f"Revenue at Risk Due to Churn\n"
        f"${churned_rev:,.0f} ({churn_rate_rev:.1f}% of total revenue) is at risk",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    out = FIGURES_DIR / "07_revenue_at_risk.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — High-risk profile: top features separating churners
# ══════════════════════════════════════════════════════════════════════════════
def plot_high_risk_profiles(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "churn_flag" and df[c].std() > 0]

    if len(numeric_cols) == 0:
        print("[EDA] Skipping high-risk profile plot — no numeric feature columns.")
        return

    # Normalised mean difference per feature
    churned  = df[df["churn_flag"] == 1][numeric_cols].mean()
    retained = df[df["churn_flag"] == 0][numeric_cols].mean()
    overall  = df[numeric_cols].mean()

    diff = ((churned - retained) / (overall.replace(0, 1))).abs()
    top10 = diff.sort_values(ascending=False).head(10)

    direction = np.sign(churned[top10.index] - retained[top10.index])
    colors = [COLOR_CHURN if d > 0 else COLOR_RETAIN for d in direction]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Normalised Mean Difference (Churned − Retained)")
    ax.set_title("Top Features Differentiating Churned vs Retained Customers",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_CHURN,  label="Higher in Churned"),
        Patch(facecolor=COLOR_RETAIN, label="Higher in Retained"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    out = FIGURES_DIR / "08_high_risk_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS INSIGHTS CSV
# ══════════════════════════════════════════════════════════════════════════════
def generate_business_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a structured summary of key business metrics — 
    mirrors what the PDF calls 'Business Insights (Non-ML)'.
    """
    insights = {}

    # ── Basic churn metrics ──
    total        = len(df)
    n_churned    = df["churn_flag"].sum()
    churn_rate   = n_churned / total * 100

    insights["total_customers"]       = total
    insights["churned_customers"]     = int(n_churned)
    insights["retained_customers"]    = int(total - n_churned)
    insights["overall_churn_rate_%"]  = round(churn_rate, 2)

    # ── Revenue metrics ──
    rev_col = _revenue_col(df)
    if rev_col:
        total_rev   = df[rev_col].sum()
        churned_rev = df[df["churn_flag"] == 1][rev_col].sum()
        insights["total_revenue_$"]           = round(total_rev, 2)
        insights["revenue_at_risk_$"]         = round(churned_rev, 2)
        insights["revenue_at_risk_%"]         = round(churned_rev / total_rev * 100, 2) if total_rev > 0 else 0
        insights["avg_revenue_churned_$"]     = round(df[df["churn_flag"] == 1][rev_col].mean(), 2)
        insights["avg_revenue_retained_$"]    = round(df[df["churn_flag"] == 0][rev_col].mean(), 2)

    # ── Highest-churn contract type ──
    col = _contract_col(df)
    if col:
        worst = (
            df.groupby(col)["churn_flag"].mean()
            .sort_values(ascending=False)
        )
        insights["highest_churn_contract"]          = worst.index[0]
        insights["highest_churn_contract_rate_%"]   = round(worst.iloc[0] * 100, 2)

    # ── Tenure insight ──
    ten_col = _tenure_col(df)
    if ten_col and df[ten_col].dtype != object:
        low_tenure   = df[df[ten_col] <= df[ten_col].quantile(0.25)]["churn_flag"].mean() * 100
        high_tenure  = df[df[ten_col] >= df[ten_col].quantile(0.75)]["churn_flag"].mean() * 100
        insights["churn_rate_low_tenure_%"]  = round(low_tenure, 2)
        insights["churn_rate_high_tenure_%"] = round(high_tenure, 2)
        insights["early_churn_multiplier"]   = round(low_tenure / high_tenure, 2) if high_tenure > 0 else None

    # ── Complaints ──
    if "complaints_count" in df.columns:
        high_complaint = df[df["complaints_count"] >= 3]["churn_flag"].mean() * 100
        insights["churn_rate_3plus_complaints_%"] = round(high_complaint, 2)

    # ── Payment delays ──
    if "payment_delays" in df.columns:
        high_delay = df[df["payment_delays"] >= 2]["churn_flag"].mean() * 100
        insights["churn_rate_2plus_payment_delays_%"] = round(high_delay, 2)

    # ── Save as CSV ──
    insights_df = pd.DataFrame(
        list(insights.items()), columns=["metric", "value"]
    )
    out = PROCESSED_DIR / "business_insights.csv"
    insights_df.to_csv(out, index=False)
    print(f"[EDA] Saved → {out.name}")
    return insights_df


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TO CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(insights_df: pd.DataFrame):
    print("\n" + "="*60)
    print("  BUSINESS INSIGHTS SUMMARY")
    print("="*60)
    for _, row in insights_df.iterrows():
        label = str(row["metric"]).replace("_", " ").title()
        print(f"  {label:<45} {row['value']}")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n[EDA] Starting Exploratory Data Analysis...")
    df = load_data()

    print(f"\n[EDA] Dataset shape   : {df.shape}")
    print(f"[EDA] Churn rate      : {df['churn_flag'].mean()*100:.2f}%")
    print(f"[EDA] Columns         : {list(df.columns)}\n")

    # Run all plots
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_vs_churn(df)
    plot_monthly_charges_box(df)
    plot_complaints_payment(df)
    plot_correlation_heatmap(df)
    plot_revenue_at_risk(df)
    plot_high_risk_profiles(df)

    # Generate and display insights
    insights_df = generate_business_insights(df)
    print_summary(insights_df)

    print(f"[EDA] All outputs saved to: {FIGURES_DIR.parent}")
    print("[EDA] Done.\n")


if __name__ == "__main__":
    main()
