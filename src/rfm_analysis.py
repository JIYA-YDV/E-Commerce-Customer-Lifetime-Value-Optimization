#!/usr/bin/env python
# coding: utf-8

# In[28]:


import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")


# In[29]:


#─────────────────────────────────────────────────────────────────
# 1. LOGGING SETUP
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# In[30]:


# 2. PATHS & CONFIG
from pathlib import Path
import os

# Works in BOTH Jupyter notebooks AND .py scripts
try:
    BASE_DIR = Path(__file__).resolve().parent  
except NameError:
    BASE_DIR = Path(os.getcwd())                 

INPUT_FILE = BASE_DIR / "cleaned_sales_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RFM_CONFIG = {
    "n_quintiles": 5,
    "snapshot_offset_days": 1,
}

print(f"BASE_DIR   : {BASE_DIR}")
print(f"INPUT_FILE : {INPUT_FILE}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"REPORT_DIR : {REPORT_DIR}")


# In[31]:


#─────────────────────────────────────────────────────────────────
# 3. SEGMENT DEFINITIONS
#    Segments are driven primarily by R and F scores.
#    M is used for KPI reporting and refining edge cases
#    (e.g. high-spend customers who've gone quiet).
# ─────────────────────────────────────────────────────────────────
def assign_segment(r: int, f: int, m: int) -> str:
    """
    Assign a business segment label based on R, F, M integer scores (1-5).

    Segmentation logic:
    - Champions       : Recent + Frequent -> highest value, reward them
    - Loyal Customers : Regular buyers, solid frequency
    - Potential        : Recent but low frequency -> nurture into loyal
    - New Customers   : Very recent, only 1-2 orders so far
    - Promising       : Recent-ish, some frequency, room to grow
    - At Risk         : Were good customers, now going cold
    - Cannot Lose Them: High M/F historically, but disappeared
    - Hibernating     : Low recency + low frequency
    - Lost            : Haven't bought in a very long time

    Args:
        r (int): Recency score 1-5  (5 = most recent)
        f (int): Frequency score 1-5 (5 = most frequent)
        m (int): Monetary score 1-5  (5 = highest spend)

    Returns:
        str: Segment name
    """
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 2 and f >= 3:
        return "Loyal Customers"
    elif r >= 3 and f <= 2:
        return "Potential Loyalists"
    elif r == 5 and f == 1:
        return "New Customers"
    elif r >= 3 and f == 1:
        return "Promising"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    elif r == 1 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    elif r <= 2 and f <= 2:
        return "Hibernating"
    else:
        return "Lost"


# Segment colours and recommended marketing actions
SEGMENT_META = {
    "Champions":           {"color": "#10b981", "action": "Reward them. Early access. Ask for reviews."},
    "Loyal Customers":     {"color": "#3b82f6", "action": "Upsell. Loyalty programme. Referral asks."},
    "Potential Loyalists": {"color": "#6366f1", "action": "Membership offers. Personalised recommendations."},
    "New Customers":       {"color": "#8b5cf6", "action": "Onboarding flow. First-purchase follow-up."},
    "Promising":           {"color": "#a78bfa", "action": "Brand awareness. Free trials or samples."},
    "At Risk":             {"color": "#f59e0b", "action": "Win-back email. Renewal discount."},
    "Cannot Lose Them":    {"color": "#ef4444", "action": "Personal outreach. Special offer. ASAP."},
    "Hibernating":         {"color": "#94a3b8", "action": "Reactivation campaign. Low-cost channel."},
    "Lost":                {"color": "#475569", "action": "Ignore or very cheap re-engagement only."},
}


# In[32]:


#─────────────────────────────────────────────────────────────────
# 4. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """
    Load cleaned sales CSV produced by 01_data_cleaning.py.

    Validates required columns exist. Parses InvoiceDate as datetime.
    Recalculates TotalAmount if the column is absent.

    Args:
        path (Path): Path to cleaned_sales_data.csv

    Returns:
        pd.DataFrame: Loaded dataframe

    Raises:
        FileNotFoundError: If the CSV does not exist
        ValueError: If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run 01_data_cleaning.py first."
        )

    log.info(f"Loading data from {path.name} ...")
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], encoding="latin-1")

    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    log.info(f"  Loaded  : {len(df):>10,} rows | {df['CustomerID'].nunique():,} unique customers")
    return df


# In[33]:


def compute_rfm(df: pd.DataFrame, offset_days: int = 1) -> pd.DataFrame:
    """
    Compute raw Recency, Frequency, Monetary values per customer.

    WHY snapshot_date?
    We use (max InvoiceDate + offset_days) as our "analysis today".
    This makes analysis reproducible regardless of when you run it —
    unlike datetime.today() which changes every run.

    Args:
        df (pd.DataFrame): Cleaned transaction data
        offset_days (int): Days to add after the last transaction date

    Returns:
        pd.DataFrame: One row per customer with R, F, M columns
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=offset_days)
    log.info(f"  Snapshot date : {snapshot_date.date()}  (last tx + {offset_days}d)")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),   # unique orders, not row count
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"]).dt.days

    log.info(f"  Customers     : {len(rfm):,}")
    log.info(f"  Recency range : {rfm['Recency'].min()}d - {rfm['Recency'].max()}d")
    log.info(f"  Frequency     : 1 - {rfm['Frequency'].max()} orders")
    log.info(f"  Monetary      : GBP {rfm['Monetary'].min():.2f} - {rfm['Monetary'].max():,.2f}")

    return rfm


# In[34]:


def score_rfm(rfm: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Convert raw R, F, M values into 1-5 quintile scores.

    KEY DETAIL — Recency is INVERTED:
      Fewer days since last purchase = bought more recently = BETTER = score 5
      So labels for R go [5, 4, 3, 2, 1] (highest days = lowest score)

    Frequency uses rank(method='first') to break ties cleanly.
    Many customers in retail have the same order count (e.g. 1),
    so direct pd.qcut would fail. Ranking converts ties into unique values.

    Args:
        rfm (pd.DataFrame): Raw RFM dataframe
        n (int): Number of quantile tiers (default 5)

    Returns:
        pd.DataFrame: RFM dataframe with R_Score, F_Score, M_Score,
                      RFM_Score (string "543"), RFM_Total (integer sum)
    """
    labels_asc  = list(range(1, n + 1))   # [1, 2, 3, 4, 5]
    labels_desc = list(range(n, 0, -1))   # [5, 4, 3, 2, 1]

    # R: invert — fewer days = higher score
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], q=n, labels=labels_desc, duplicates="drop"
    ).astype(int)

    # F: rank to break ties (many customers buy only once)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=n, labels=labels_asc, duplicates="drop"
    ).astype(int)

    # M: straightforward ascending
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"], q=n, labels=labels_asc, duplicates="drop"
    ).astype(int)

    # Composite score string e.g. "543"
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) +
        rfm["F_Score"].astype(str) +
        rfm["M_Score"].astype(str)
    )

    # Numeric total for sorting (max = 15 = best customer)
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    log.info(f"  Score range   : {rfm['RFM_Total'].min()} - {rfm['RFM_Total'].max()} (max possible = {n*3})")
    return rfm


# In[35]:


def apply_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Map R/F/M scores to named business segments and attach action notes.

    Args:
        rfm (pd.DataFrame): Scored RFM dataframe

    Returns:
        pd.DataFrame: RFM with Segment and Action columns added
    """
    rfm["Segment"] = rfm.apply(
        lambda row: assign_segment(row["R_Score"], row["F_Score"], row["M_Score"]),
        axis=1,
    )
    rfm["Action"] = rfm["Segment"].map(lambda s: SEGMENT_META[s]["action"])

    seg_counts = rfm["Segment"].value_counts()
    log.info("  Segment distribution:")
    for seg, cnt in seg_counts.items():
        pct = cnt / len(rfm) * 100
        log.info(f"    {seg:<22} : {cnt:>5,} customers  ({pct:.1f}%)")

    return rfm


# In[36]:


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Build a segment-level KPI summary table for business reporting.

    Computes per segment: customer count, revenue share, recency, frequency.

    Args:
        rfm (pd.DataFrame): Fully scored and segmented RFM dataframe

    Returns:
        pd.DataFrame: Summary table sorted by total revenue descending
    """
    total_revenue   = rfm["Monetary"].sum()
    total_customers = len(rfm)

    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Revenue=("Monetary", "mean"),
            Median_Revenue=("Monetary", "median"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_RFM_Score=("RFM_Total", "mean"),
        )
        .reset_index()
    )

    summary["Customer_Share_%"] = (summary["Customers"] / total_customers * 100).round(1)
    summary["Revenue_Share_%"]  = (summary["Total_Revenue"] / total_revenue * 100).round(1)
    summary["Avg_Revenue"]      = summary["Avg_Revenue"].round(2)
    summary["Median_Revenue"]   = summary["Median_Revenue"].round(2)
    summary["Avg_Recency_Days"] = summary["Avg_Recency_Days"].round(1)
    summary["Avg_Frequency"]    = summary["Avg_Frequency"].round(1)
    summary["Avg_RFM_Score"]    = summary["Avg_RFM_Score"].round(1)

    summary = summary.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
    return summary


# In[42]:


def _apply_dark_style(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#94a3b8")


def plot_segment_distribution(rfm, save_path):
    seg_counts = rfm["Segment"].value_counts().sort_values()
    colors = [SEGMENT_META[s]["color"] for s in seg_counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    bars = ax.barh(seg_counts.index, seg_counts.values,
                   color=colors, edgecolor="none", height=0.65)

    for bar, val in zip(bars, seg_counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left",
                color="#e2e8f0", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Customers", color="#64748b", fontsize=10)
    ax.set_title("RFM Customer Segmentation — Customer Count per Segment",
                 color="#e2e8f0", fontsize=13, fontweight="bold", pad=15)
    for spine in ax.spines.values():           # ← fixed
        spine.set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for label in ax.get_yticklabels():
        label.set_color("#e2e8f0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_rfm_scatter(rfm, save_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    _apply_dark_style(fig, ax)

    mn, mx = rfm["Monetary"].min(), rfm["Monetary"].max()
    sizes = ((rfm["Monetary"] - mn) / (mx - mn)) * 200 + 10

    for seg, meta in SEGMENT_META.items():
        mask = rfm["Segment"] == seg
        if mask.sum() == 0:
            continue
        ax.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Frequency"],
                   s=sizes[mask], c=meta["color"], alpha=0.6, label=seg,
                   edgecolors="none")

    ax.set_xlabel("Recency (days)  Lower = Better", color="#64748b", fontsize=10)
    ax.set_ylabel("Frequency (unique orders)", color="#64748b", fontsize=10)
    ax.set_title("RFM Scatter: Recency vs Frequency\n(bubble size = Monetary value)",
                 color="#e2e8f0", fontsize=13, fontweight="bold", pad=15)
    legend = ax.legend(loc="upper right", framealpha=0.2,
                       labelcolor="#e2e8f0", fontsize=8)
    legend.get_frame().set_facecolor("#1a1e2a")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_rfm_heatmap(rfm, save_path):
    pivot = rfm.pivot_table(
        values="Monetary", index="R_Score", columns="F_Score", aggfunc="mean"
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.5, linecolor=DARK_BG,
                cbar_kws={"label": "Avg Revenue (GBP)"})

    ax.set_title("Avg Revenue by R x F Score",
                 color="#e2e8f0", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("F Score (Frequency)", color="#64748b", fontsize=10)
    ax.set_ylabel("R Score (Recency)", color="#64748b", fontsize=10)
    ax.tick_params(colors="#e2e8f0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path.name}")

print("✅ Plot functions redefined — now run: rfm_result = run_rfm_pipeline()")


# In[44]:


#─────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_rfm_pipeline() -> pd.DataFrame:
    """
    Orchestrates the full RFM analysis pipeline end-to-end.

    Steps:
        1. Load cleaned data (from 01_data_cleaning.py)
        2. Compute raw R, F, M metrics per customer
        3. Score each metric into quintiles (1-5)
        4. Assign named business segments
        5. Build segment-level KPI summary
        6. Export CSVs to data/processed/
        7. Generate 3 charts to reports/

    Returns:
        pd.DataFrame: Final RFM dataframe with scores and segments
    """
    print("\n" + "=" * 60)
    print("  RFM ANALYSIS PIPELINE")
    print("  E-Commerce CLV Optimization — uk-retail-analytics")
    print("=" * 60)

    print("\n[1/7] Loading cleaned data ...")
    df = load_data(INPUT_FILE)

    print("\n[2/7] Computing Recency / Frequency / Monetary ...")
    rfm = compute_rfm(df, offset_days=RFM_CONFIG["snapshot_offset_days"])

    print("\n[3/7] Scoring into quintiles (1-5) ...")
    rfm = score_rfm(rfm, n=RFM_CONFIG["n_quintiles"])

    print("\n[4/7] Assigning business segments ...")
    rfm = apply_segments(rfm)

    print("\n[5/7] Building segment KPI summary ...")
    summary = segment_summary(rfm)

    print("\n[6/7] Saving outputs ...")
    rfm_out = OUTPUT_DIR / "rfm_scores.csv"
    sum_out  = OUTPUT_DIR / "rfm_segments_summary.csv"
    rfm.drop(columns=["Action"]).to_csv(rfm_out, index=False)
    summary.to_csv(sum_out, index=False)
    log.info(f"  Saved: {rfm_out.name}  ({len(rfm):,} rows)")
    log.info(f"  Saved: {sum_out.name}  ({len(summary)} segments)")

    print("\n[7/7] Generating visualisations ...")
    plot_segment_distribution(rfm, REPORT_DIR / "rfm_segment_distribution.png")
    plot_rfm_scatter(rfm,          REPORT_DIR / "rfm_scatter.png")
    plot_rfm_heatmap(rfm,          REPORT_DIR / "rfm_heatmap.png")

    # ── Print summary table ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SEGMENT KPI SUMMARY")
    print("=" * 60)
    print(
        summary[[
            "Segment", "Customers", "Customer_Share_%",
            "Total_Revenue", "Revenue_Share_%",
            "Avg_Recency_Days", "Avg_Frequency"
        ]].to_string(index=False)
    )

    # ── Print top business insights ───────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY BUSINESS INSIGHTS")
    print("=" * 60)

    for seg_name, icon in [("Champions", "  BEST"),
                            ("At Risk",   " ALERT"),
                            ("Cannot Lose Them", " URGENT")]:
        row = summary[summary["Segment"] == seg_name]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  [{icon}] {seg_name}")
            print(f"    Customers  : {int(r.Customers):,}  ({r['Customer_Share_%']:.1f}% of base)")
            print(f"    Revenue    : GBP {r.Total_Revenue:,.0f}  ({r['Revenue_Share_%']:.1f}% of total)")
            print(f"    Action     : {SEGMENT_META[seg_name]['action']}")

    print("\n" + "=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    print(f"  data/processed/rfm_scores.csv")
    print(f"  data/processed/rfm_segments_summary.csv")
    print(f"  reports/rfm_segment_distribution.png")
    print(f"  reports/rfm_scatter.png")
    print(f"  reports/rfm_heatmap.png")

    return rfm


# In[45]:


# Run the full pipeline
rfm_result = run_rfm_pipeline()


# In[ ]:




