#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")


# In[2]:


#─────────────────────────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# In[3]:


#─────────────────────────────────────────────────────────────────
# 2. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd())          # Jupyter fallback

INPUT_FILE = BASE_DIR / "cleaned_sales_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR   : {BASE_DIR}")
print(f"INPUT_FILE : {INPUT_FILE}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"REPORT_DIR : {REPORT_DIR}")


# In[4]:


#─────────────────────────────────────────────────────────────────
# 3. THEME
# ─────────────────────────────────────────────────────────────────
DARK_BG  = "#0d0f14"
SURFACE  = "#13161e"


# In[5]:


#─────────────────────────────────────────────────────────────────
# 4. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """
    Load and validate cleaned sales data.

    Requires: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice.
    Recalculates TotalAmount if missing.

    Args:
        path (Path): Path to cleaned_sales_data.csv

    Returns:
        pd.DataFrame: Validated transaction dataframe

    Raises:
        FileNotFoundError: If CSV does not exist
        ValueError: If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            "Run 01_data_cleaning.py first."
        )

    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], encoding="latin-1")

    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    log.info(f"  Rows      : {len(df):,}")
    log.info(f"  Customers : {df['CustomerID'].nunique():,}")
    log.info(f"  Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    return df


def build_cohort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each transaction a cohort month and cohort index.

    WHY cohort_month = first purchase month per customer?
    A cohort must be defined by the acquisition event — the moment
    a customer was 'born' into your business. Using first purchase
    month groups customers by when they were acquired.

    WHY cohort_index = months between cohort and transaction?
    Index 0 = same month as acquisition (always 100% retention).
    Index 1 = returned next month, Index 2 = two months later, etc.
    This normalises time so cohorts of different sizes are comparable.

    Args:
        df (pd.DataFrame): Cleaned transaction dataframe

    Returns:
        pd.DataFrame: Transactions with CohortMonth and CohortIndex added
    """
    df = df.copy()

    # Transaction month (year + month only — day doesn't matter)
    df["TransactionMonth"] = df["InvoiceDate"].dt.to_period("M")

    # Each customer's first purchase month = their cohort
    cohort_map = (
        df.groupby("CustomerID")["TransactionMonth"]
        .min()
        .reset_index()
        .rename(columns={"TransactionMonth": "CohortMonth"})
    )

    df = df.merge(cohort_map, on="CustomerID", how="left")

    # Cohort index = how many months after first purchase
    # .n gives the integer difference between two Period objects
    df["CohortIndex"] = (
        df["TransactionMonth"] - df["CohortMonth"]
    ).apply(lambda x: x.n)

    n_cohorts = df["CohortMonth"].nunique()
    max_index  = df["CohortIndex"].max()
    log.info(f"  Cohorts   : {n_cohorts} monthly cohorts")
    log.info(f"  Max index : M{max_index} (months of history)")

    return df


def build_retention_matrix(cohort_df: pd.DataFrame) -> tuple:
    """
    Build raw customer count matrix and normalised retention % matrix.

    HOW it works:
    1. Group by CohortMonth + CohortIndex, count unique customers
    2. Pivot into a wide matrix (rows=cohorts, columns=M0,M1,M2...)
    3. Divide every row by its M0 value → retention percentages
    4. M0 is always 100% (every customer bought in their first month)

    Args:
        cohort_df (pd.DataFrame): Output of build_cohort_data()

    Returns:
        tuple: (cohort_counts, retention_pct)
               cohort_counts — raw customer counts per cell
               retention_pct — percentage retained vs M0
    """
    # Count unique customers per cohort × index cell
    cohort_counts = (
        cohort_df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .reset_index()
        .rename(columns={"CustomerID": "Customers"})
    )

    # Pivot: rows = cohort months, columns = cohort index (M0, M1...)
    cohort_pivot = cohort_counts.pivot_table(
        index="CohortMonth",
        columns="CohortIndex",
        values="Customers"
    )

    # Rename columns to M0, M1, M2... for clarity
    cohort_pivot.columns = [f"M{c}" for c in cohort_pivot.columns]
    cohort_pivot.index   = cohort_pivot.index.astype(str)

    # Normalise: divide every row by M0 (cohort size at acquisition)
    cohort_sizes  = cohort_pivot["M0"]
    retention_pct = cohort_pivot.divide(cohort_sizes, axis=0) * 100

    log.info(f"  Matrix shape : {cohort_pivot.shape[0]} cohorts × {cohort_pivot.shape[1]} periods")
    log.info(f"  Overall M1 retention : {retention_pct['M1'].mean():.1f}% avg across cohorts")
    if "M2" in retention_pct.columns:
        log.info(f"  Overall M2 retention : {retention_pct['M2'].mean():.1f}% avg across cohorts")

    return cohort_pivot, retention_pct


def build_revenue_matrix(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build average revenue per customer matrix across cohorts.

    WHY average revenue (not total)?
    Total revenue is biased by cohort size — larger cohorts always
    look better. Average revenue per customer reveals which cohorts
    contain genuinely higher-value buyers regardless of size.

    Args:
        cohort_df (pd.DataFrame): Output of build_cohort_data()

    Returns:
        pd.DataFrame: Pivot of avg revenue per customer per cohort×month
    """
    revenue_data = (
        cohort_df.groupby(["CohortMonth", "CohortIndex"])
        .agg(
            TotalRevenue=("TotalAmount", "sum"),
            UniqueCustomers=("CustomerID", "nunique")
        )
        .reset_index()
    )

    revenue_data["AvgRevenue"] = (
        revenue_data["TotalRevenue"] / revenue_data["UniqueCustomers"]
    ).round(2)

    revenue_pivot = revenue_data.pivot_table(
        index="CohortMonth",
        columns="CohortIndex",
        values="AvgRevenue"
    )

    revenue_pivot.columns = [f"M{c}" for c in revenue_pivot.columns]
    revenue_pivot.index   = revenue_pivot.index.astype(str)

    return revenue_pivot


def build_cohort_summary(cohort_pivot: pd.DataFrame,
                         retention_pct: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-cohort summary table for business reporting.

    Includes: cohort size, M1/M3/M6 retention rates, and a
    quality label (Strong / Average / Weak) based on M1 retention.

    Args:
        cohort_pivot (pd.DataFrame): Raw customer counts
        retention_pct (pd.DataFrame): Retention % matrix

    Returns:
        pd.DataFrame: One row per cohort with key KPIs
    """
    summary = pd.DataFrame()
    summary["CohortMonth"]    = cohort_pivot.index
    summary["CohortSize"]     = cohort_pivot["M0"].values
    summary["M1_Retention_%"] = retention_pct.get("M1", pd.Series()).values
    summary["M3_Retention_%"] = retention_pct.get("M3", pd.Series()).values
    summary["M6_Retention_%"] = retention_pct.get("M6", pd.Series()).values

    # Quality label based on M1 retention
    def quality(m1):
        if pd.isna(m1):
            return "Insufficient data"
        elif m1 >= 30:
            return "Strong"
        elif m1 >= 15:
            return "Average"
        else:
            return "Weak"

    summary["CohortQuality"] = summary["M1_Retention_%"].apply(quality)
    summary = summary.round(1)

    return summary


# In[6]:


#─────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

def plot_retention_heatmap(retention_pct: pd.DataFrame, save_path: Path) -> None:
    """
    Heatmap of retention % across cohorts and time periods.

    This is THE signature chart of cohort analysis. Each cell shows
    what % of the acquisition cohort (row) returned in period M (column).

    HOW TO READ IT:
    - Diagonal pattern from top-left = natural — recent cohorts have
      fewer months of history so cells are empty (NaN = grey)
    - Dark red cells = high retention = valuable cohorts
    - Pale/yellow cells = low retention = churn risk
    - Compare rows: which acquisition months produced best cohorts?
    - Compare columns: at which month does retention stabilise?

    Args:
        retention_pct (pd.DataFrame): Retention % matrix
        save_path (Path): Output PNG path
    """
    # Keep only first 13 periods (M0–M12) for readability
    cols_to_show = [c for c in retention_pct.columns
                    if int(c.replace("M", "")) <= 12]
    data = retention_pct[cols_to_show]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor=DARK_BG,
        mask=data.isnull(),          # grey out NaN cells (not enough history)
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Retention %", "shrink": 0.8},
        annot_kws={"size": 8},
    )

    ax.set_title(
        "Monthly Cohort Retention Heatmap\n"
        "% of customers who returned each month after first purchase",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Months Since First Purchase", color="#64748b", fontsize=11)
    ax.set_ylabel("Acquisition Cohort (First Purchase Month)", color="#64748b", fontsize=11)
    ax.tick_params(colors="#94a3b8", labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


def plot_revenue_heatmap(revenue_pivot: pd.DataFrame, save_path: Path) -> None:
    """
    Heatmap of average revenue per customer across cohorts and periods.

    Complements the retention heatmap — shows not just IF customers
    return but HOW MUCH they spend when they do.

    A cohort can have low retention but high avg revenue (premium buyers
    who buy infrequently) — this chart surfaces that nuance.

    Args:
        revenue_pivot (pd.DataFrame): Avg revenue matrix
        save_path (Path): Output PNG path
    """
    cols_to_show = [c for c in revenue_pivot.columns
                    if int(c.replace("M", "")) <= 12]
    data = revenue_pivot[cols_to_show]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.4,
        linecolor=DARK_BG,
        mask=data.isnull(),
        cbar_kws={"label": "Avg Revenue per Customer (GBP)", "shrink": 0.8},
        annot_kws={"size": 8},
    )

    ax.set_title(
        "Average Revenue per Customer by Cohort\n"
        "GBP spent per returning customer each month",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Months Since First Purchase", color="#64748b", fontsize=11)
    ax.set_ylabel("Acquisition Cohort", color="#64748b", fontsize=11)
    ax.tick_params(colors="#94a3b8", labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


def plot_retention_curves(retention_pct: pd.DataFrame, save_path: Path) -> None:
    """
    Line chart: retention curve per cohort over time.

    WHY a line chart in addition to the heatmap?
    The heatmap shows individual cells. The line chart shows SHAPE —
    how steeply each cohort drops off and where it stabilises.
    Cohorts that flatten early (instead of continuing to drop) have
    a loyal retained base worth investing in.

    A steep drop from M0→M1 followed by a flat line is HEALTHY.
    A continued downward slope with no flattening = ongoing churn problem.

    Args:
        retention_pct (pd.DataFrame): Retention % matrix
        save_path (Path): Output PNG path
    """
    cols_to_show = [c for c in retention_pct.columns
                    if int(c.replace("M", "")) <= 12]
    data = retention_pct[cols_to_show].copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(SURFACE)

    # Colour gradient across cohorts
    n = len(data)
    cmap = plt.cm.get_cmap("plasma", n)

    for i, (cohort, row) in enumerate(data.iterrows()):
        valid = row.dropna()
        if len(valid) < 2:          # skip cohorts with too little history
            continue
        x = [int(c.replace("M", "")) for c in valid.index]
        ax.plot(x, valid.values,
                color=cmap(i / n),
                linewidth=1.5,
                alpha=0.75,
                marker="o",
                markersize=3,
                label=str(cohort))

    # Average retention line across all cohorts
    avg_retention = data.mean()
    x_avg = [int(c.replace("M", "")) for c in avg_retention.index]
    ax.plot(x_avg, avg_retention.values,
            color="#f97316", linewidth=3,
            linestyle="--", label="Average (all cohorts)",
            zorder=5)

    ax.set_xlabel("Months Since First Purchase", color="#64748b", fontsize=11)
    ax.set_ylabel("Retention Rate (%)", color="#64748b", fontsize=11)
    ax.set_title(
        "Cohort Retention Curves\n"
        "Each line = one acquisition cohort  |  Orange dash = average",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#94a3b8")

    # Legend — only show a few cohorts to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    # Show first 3, last 3, and average
    keep_idx = list(range(min(3, len(handles)-1))) +                list(range(max(3, len(handles)-4), len(handles)))
    keep_idx = sorted(set(keep_idx))
    legend = ax.legend(
        [handles[i] for i in keep_idx],
        [labels[i]  for i in keep_idx],
        loc="upper right", framealpha=0.2,
        labelcolor="#e2e8f0", fontsize=8
    )
    legend.get_frame().set_facecolor("#1a1e2a")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


# In[9]:


#─────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_cohort_pipeline() -> dict:
    """
    Orchestrates the full cohort analysis pipeline.

    Steps:
        1. Load cleaned transaction data
        2. Assign cohort month + cohort index to every transaction
        3. Build retention matrix (counts + percentages)
        4. Build revenue matrix (avg spend per customer)
        5. Build cohort quality summary
        6. Export 3 CSVs
        7. Generate 3 visualisations

    Returns:
        dict: {
            'cohort_df'     : transactions with cohort columns,
            'cohort_counts' : raw count matrix,
            'retention_pct' : retention % matrix,
            'revenue_pivot' : avg revenue matrix,
            'summary'       : cohort KPI summary
        }
    """
    print("\n" + "=" * 60)
    print("  COHORT ANALYSIS PIPELINE")
    print("  E-Commerce CLV Optimization — uk-retail-analytics")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────
    print("\n[1/7] Loading cleaned data ...")
    df = load_data(INPUT_FILE)

    # ── Step 2: Assign cohort columns ────────────────────────────
    print("\n[2/7] Assigning cohort month and cohort index ...")
    cohort_df = build_cohort_data(df)

    # ── Step 3: Retention matrix ──────────────────────────────────
    print("\n[3/7] Building retention matrix ...")
    cohort_counts, retention_pct = build_retention_matrix(cohort_df)

    # ── Step 4: Revenue matrix ────────────────────────────────────
    print("\n[4/7] Building revenue matrix ...")
    revenue_pivot = build_revenue_matrix(cohort_df)

    # ── Step 5: Summary ───────────────────────────────────────────
    print("\n[5/7] Building cohort quality summary ...")
    summary = build_cohort_summary(cohort_counts, retention_pct)

    # ── Step 6: Save CSVs ─────────────────────────────────────────
    print("\n[6/7] Saving outputs ...")
    retention_pct.to_csv(OUTPUT_DIR / "cohort_retention_matrix.csv")
    revenue_pivot.to_csv(OUTPUT_DIR / "cohort_revenue_matrix.csv")
    summary.to_csv(OUTPUT_DIR / "cohort_summary.csv", index=False)
    log.info("  Saved: cohort_retention_matrix.csv")
    log.info("  Saved: cohort_revenue_matrix.csv")
    log.info("  Saved: cohort_summary.csv")

    # ── Step 7: Visualisations ────────────────────────────────────
    print("\n[7/7] Generating visualisations ...")
    plot_retention_heatmap(retention_pct, REPORT_DIR / "cohort_retention_heatmap.png")
    plot_revenue_heatmap(revenue_pivot,   REPORT_DIR / "cohort_revenue_heatmap.png")
    plot_retention_curves(retention_pct,  REPORT_DIR / "cohort_retention_curve.png")

    # ── Console Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COHORT QUALITY SUMMARY")
    print("=" * 60)
    print(summary[["CohortMonth", "CohortSize",
                   "M1_Retention_%", "M3_Retention_%",
                   "CohortQuality"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("  KEY BUSINESS INSIGHTS")
    print("=" * 60)

    best  = summary.dropna(subset=["M1_Retention_%"]).nlargest(1, "M1_Retention_%")
    worst = summary.dropna(subset=["M1_Retention_%"]).nsmallest(1, "M1_Retention_%")
    avg_m1 = summary["M1_Retention_%"].mean()

    if not best.empty:
        b = best.iloc[0]
        print(f"\n  BEST cohort  : {b.CohortMonth}  →  M1 retention {b['M1_Retention_%']:.1f}%")
        print(f"  Cohort size  : {int(b.CohortSize):,} customers")

    if not worst.empty:
        w = worst.iloc[0]
        print(f"\n  WORST cohort : {w.CohortMonth}  →  M1 retention {w['M1_Retention_%']:.1f}%")
        print(f"  Cohort size  : {int(w.CohortSize):,} customers")

    print(f"\n  AVG M1 retention across all cohorts : {avg_m1:.1f}%")
    print(f"  Interpretation: For every 100 new customers acquired,")
    print(f"  ~{avg_m1:.0f} return the following month on average.")

    strong = summary[summary["CohortQuality"] == "Strong"]
    weak   = summary[summary["CohortQuality"] == "Weak"]
    print(f"\n  Strong cohorts (>=30% M1) : {len(strong)}")
    print(f"  Weak cohorts   (<15% M1) : {len(weak)}")

    print("\n" + "=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    print("  data/processed/cohort_retention_matrix.csv")
    print("  data/processed/cohort_revenue_matrix.csv")
    print("  data/processed/cohort_summary.csv")
    print("  reports/cohort_retention_heatmap.png")
    print("  reports/cohort_revenue_heatmap.png")
    print("  reports/cohort_retention_curve.png")

    return {
        "cohort_df":      cohort_df,
        "cohort_counts":  cohort_counts,
        "retention_pct":  retention_pct,
        "revenue_pivot":  revenue_pivot,
        "summary":        summary,
    }


# In[10]:


# ─────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_cohort_pipeline()


# In[ ]:




