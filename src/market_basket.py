#!/usr/bin/env python
# coding: utf-8

# In[18]:


# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────
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

# mlxtend — install with: pip install mlxtend
try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("⚠️  mlxtend not installed. Run: pip install mlxtend")
    print("   Then restart your kernel and re-run this script.")


# In[19]:


import sys
get_ipython().system('{sys.executable} -m pip install mlxtend --user')


# In[20]:


from mlxtend.frequent_patterns import fpgrowth, association_rules
print("✅ mlxtend ready")


# In[21]:


#─────────────────────────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# In[22]:


# ─────────────────────────────────────────────────────────────────
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

# ── MBA Tuning Parameters ─────────────────────────────────────────
MBA_CONFIG = {
    # Only analyse UK transactions (90%+ of data, reduces noise)
    "uk_only": True,

    # Keep only top N products by transaction count before encoding.
    # WHY: Limits memory usage. Top 200 covers ~80% of volume.
    # Increase to 300–500 if you have 16GB+ RAM.
    "top_n_products": 200,

    # Minimum support: item combination must appear in this fraction
    # of all baskets. 0.01 = at least 1% of invoices. Lower = more
    # rules but slower and noisier.
    "min_support": 0.01,

    # Minimum confidence: given antecedent, how often is consequent
    # also purchased. 0.3 = 30%+ of the time.
    "min_confidence": 0.3,

    # Minimum lift: how much more likely than random chance.
    # 1.0 = no association. Filter to > 1.0 removes meaningless rules.
    "min_lift": 1.0,

    # Top N rules to show in the bar chart visual
    "top_n_rules": 20,
}

print(f"BASE_DIR   : {BASE_DIR}")
print(f"INPUT_FILE : {INPUT_FILE}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"REPORT_DIR : {REPORT_DIR}")


# In[23]:


#─────────────────────────────────────────────────────────────────
# 3. THEME
# ─────────────────────────────────────────────────────────────────
DARK_BG = "#0d0f14"
SURFACE = "#13161e"


# In[24]:


#─────────────────────────────────────────────────────────────────
# 4. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def load_data(path: Path, uk_only: bool = True) -> pd.DataFrame:
    """
    Load cleaned sales data and optionally filter to UK transactions.

    WHY filter to UK only?
    The UCI dataset is ~90% UK transactions. International orders
    often contain bulk/wholesale purchases (single-product invoices)
    that don't represent typical retail basket behaviour and create
    noise in association rules. UK-only gives cleaner, more
    actionable rules for a UK retail business.

    Args:
        path (Path): Path to cleaned_sales_data.csv
        uk_only (bool): If True, filter to Country == 'United Kingdom'

    Returns:
        pd.DataFrame: Filtered transaction dataframe

    Raises:
        FileNotFoundError: If CSV does not exist
        RuntimeError: If mlxtend is not installed
    """
    if not MLXTEND_AVAILABLE:
        raise RuntimeError("mlxtend is required. Run: pip install mlxtend")

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            "Run 01_data_cleaning.py first."
        )

    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    required = {"CustomerID", "InvoiceNo", "Description", "Quantity", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    log.info(f"  Total rows : {len(df):,}")

    if uk_only and "Country" in df.columns:
        df = df[df["Country"] == "United Kingdom"].copy()
        log.info(f"  UK rows    : {len(df):,}  (after UK filter)")

    # Drop rows with missing product description
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].str.strip().str.upper()

    log.info(f"  Invoices   : {df['InvoiceNo'].nunique():,} unique baskets")
    log.info(f"  Products   : {df['Description'].nunique():,} unique products")

    return df


def filter_top_products(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """
    Keep only transactions involving the top N most-purchased products.

    WHY this step is critical for memory:
    One-hot encoding all products creates a matrix of shape
    (n_invoices × n_products). With 20,000 invoices and 4,000 products,
    that's an 80 million cell boolean matrix — likely to OOM.

    Filtering to top 200 products:
    - Covers ~80% of actual transaction volume (Pareto principle)
    - Reduces matrix to 20,000 × 200 = 4 million cells (manageable)
    - Focuses rules on products customers actually buy regularly
      (rare products produce unreliable rules anyway)

    After filtering, drop invoices that become empty (had only
    rare products) and drop single-item invoices (can't form pairs).

    Args:
        df (pd.DataFrame): Transaction dataframe
        top_n (int): Number of top products to retain

    Returns:
        pd.DataFrame: Filtered dataframe with only top-N products
    """
    # Find top N products by number of invoices they appear in
    # (not by quantity — a product bought 1000 at once is less
    #  interesting than one bought across 500 separate orders)
    product_counts = (
        df.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    top_products = product_counts.head(top_n).index.tolist()

    df_filtered = df[df["Description"].isin(top_products)].copy()

    log.info(f"  Top {top_n} products selected")
    log.info(f"  Rows after product filter : {len(df_filtered):,}")

    # Remove invoices that now have fewer than 2 distinct products
    # (single-item baskets cannot produce association rules)
    basket_sizes = df_filtered.groupby("InvoiceNo")["Description"].nunique()
    valid_invoices = basket_sizes[basket_sizes >= 2].index
    df_filtered = df_filtered[df_filtered["InvoiceNo"].isin(valid_invoices)]

    log.info(f"  Valid multi-item invoices : {df_filtered['InvoiceNo'].nunique():,}")
    log.info(f"  Rows after basket filter  : {len(df_filtered):,}")

    return df_filtered


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the invoice × product binary matrix required by FP-Growth.

    Each row = one invoice (basket).
    Each column = one product.
    Each cell = True if the product was in that invoice, False otherwise.

    WHY binary (not quantity)?
    Association rules care about CO-OCCURRENCE, not quantity.
    Whether someone bought 1 or 10 of a product, what matters is
    that it appeared in the basket alongside another product.
    Binarising with applymap(bool) handles this.

    Args:
        df (pd.DataFrame): Filtered transaction dataframe

    Returns:
        pd.DataFrame: Boolean invoice × product matrix
    """
    log.info("  Building basket matrix ...")

    # Pivot: sum quantities per invoice × product, then binarise
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
        .astype(bool)      # True = product appeared, False = did not
    )

    log.info(f"  Basket matrix : {basket.shape[0]:,} invoices × {basket.shape[1]} products")
    log.info(f"  Matrix density: {basket.values.mean()*100:.2f}% non-zero")

    return basket


def run_fpgrowth(basket: pd.DataFrame,
                 min_support: float = 0.01) -> pd.DataFrame:
    """
    Run FP-Growth algorithm to find frequent itemsets.

    WHY FP-Growth over Apriori?

    Apriori generates ALL candidate itemsets explicitly, then prunes.
    For 200 products, that means checking 2^200 possible subsets —
    completely infeasible.

    FP-Growth builds a compressed "FP-tree" from the transaction data
    and mines it directly without generating candidates. It is
    typically 10–100× faster and uses far less memory. The results
    are mathematically identical to Apriori.

    use_colnames=True preserves product names in the output instead
    of returning column indices.

    Args:
        basket (pd.DataFrame): Boolean invoice × product matrix
        min_support (float): Minimum support threshold

    Returns:
        pd.DataFrame: Frequent itemsets with support values
    """
    log.info(f"  Running FP-Growth (min_support={min_support}) ...")

    itemsets = fpgrowth(
        basket,
        min_support=min_support,
        use_colnames=True,       # return product names not indices
        max_len=3,               # limit to pairs and triplets (faster)
        verbose=0,
    )

    # Add itemset length for filtering
    itemsets["length"] = itemsets["itemsets"].apply(len)

    log.info(f"  Frequent itemsets found : {len(itemsets):,}")
    log.info(f"  Pairs (length=2)        : {(itemsets['length']==2).sum():,}")
    log.info(f"  Triplets (length=3)     : {(itemsets['length']==3).sum():,}")

    return itemsets


def generate_rules(itemsets: pd.DataFrame,
                   min_confidence: float = 0.3,
                   min_lift: float = 1.0) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    From each frequent itemset, mlxtend generates all possible
    antecedent → consequent rules. For example, from itemset
    {A, B, C} it generates: A→BC, B→AC, C→AB, AB→C, AC→B, BC→A.

    Then filters by minimum confidence and lift thresholds.

    Post-processing adds human-readable columns:
    - antecedents_str / consequents_str : readable product names
    - rule_str : "Product A  →  Product B"
    - strength : "Strong" / "Moderate" / "Weak" based on lift

    Args:
        itemsets (pd.DataFrame): Frequent itemsets from FP-Growth
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold

    Returns:
        pd.DataFrame: Filtered and enriched association rules,
                      sorted by lift descending
    """
    log.info(f"  Generating rules (confidence≥{min_confidence}, lift≥{min_lift}) ...")

    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    # Apply lift filter
    rules = rules[rules["lift"] >= min_lift].copy()

    # Sort by lift descending (most meaningful associations first)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Human-readable columns
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["rule_str"] = rules["antecedents_str"] + "  →  " + rules["consequents_str"]

    # Strength label based on lift
    def lift_label(lift):
        if lift >= 5:
            return "Very Strong"
        elif lift >= 3:
            return "Strong"
        elif lift >= 2:
            return "Moderate"
        else:
            return "Weak"

    rules["strength"] = rules["lift"].apply(lift_label)

    # Round numeric columns for readability
    for col in ["support", "confidence", "lift",
                "leverage", "conviction"]:
        if col in rules.columns:
            rules[col] = rules[col].round(4)

    log.info(f"  Rules generated   : {len(rules):,}")
    log.info(f"  Very Strong (≥5x) : {(rules['strength']=='Very Strong').sum()}")
    log.info(f"  Strong (3–5x)     : {(rules['strength']=='Strong').sum()}")
    log.info(f"  Moderate (2–3x)   : {(rules['strength']=='Moderate').sum()}")

    return rules


def build_top_pairs(rules: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Extract the top product pairs (antecedent + consequent both single items).

    Filters to rules where both sides are single products — these are
    the cleanest cross-sell signals and easiest to action.

    Args:
        rules (pd.DataFrame): Full association rules dataframe
        top_n (int): Number of top pairs to return

    Returns:
        pd.DataFrame: Top N product pairs sorted by lift
    """
    # Filter to single-item antecedent AND single-item consequent
    pair_rules = rules[
        (rules["antecedents"].apply(len) == 1) &
        (rules["consequents"].apply(len) == 1)
    ].copy()

    top_pairs = pair_rules.nlargest(top_n, "lift")[
        ["antecedents_str", "consequents_str", "rule_str",
         "support", "confidence", "lift", "strength"]
    ].reset_index(drop=True)

    return top_pairs


# In[25]:


#─────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

def plot_top_rules(top_pairs: pd.DataFrame, save_path: Path) -> None:
    """
    Horizontal bar chart of top product pairs ranked by lift.

    WHY lift on the x-axis (not confidence)?
    Lift is the most informative single metric — it controls for
    product popularity. A rule with 80% confidence but lift=1.1
    is nearly useless; a rule with 40% confidence but lift=6.0
    is very actionable.

    Bars are coloured by strength category.

    Args:
        top_pairs (pd.DataFrame): Top product pairs dataframe
        save_path (Path): Output PNG path
    """
    if top_pairs.empty:
        log.warning("  No pairs to plot — skipping top_rules chart")
        return

    # Shorten long product names for readability
    def shorten(name, max_len=35):
        return name[:max_len] + "…" if len(name) > max_len else name

    top_pairs = top_pairs.head(15).copy()
    top_pairs["rule_short"] = top_pairs["rule_str"].apply(shorten)

    strength_colors = {
        "Very Strong": "#10b981",
        "Strong":      "#3b82f6",
        "Moderate":    "#f59e0b",
        "Weak":        "#94a3b8",
    }
    colors = top_pairs["strength"].map(strength_colors).fillna("#94a3b8")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(SURFACE)

    bars = ax.barh(
        top_pairs["rule_short"][::-1],   # reverse so highest lift is at top
        top_pairs["lift"][::-1],
        color=list(colors[::-1]),
        edgecolor="none",
        height=0.65,
    )

    # Value labels
    for bar, lift_val in zip(bars, top_pairs["lift"][::-1]):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"lift {lift_val:.2f}",
            va="center", ha="left",
            color="#e2e8f0", fontsize=8, fontweight="bold"
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=s)
        for s, c in strength_colors.items()
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower right",
        framealpha=0.2, labelcolor="#e2e8f0", fontsize=8
    )
    legend.get_frame().set_facecolor("#1a1e2a")

    ax.set_xlabel("Lift  (higher = stronger association)", color="#64748b", fontsize=10)
    ax.set_title(
        "Top Product Associations by Lift\n"
        "Products bought together more often than by random chance",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=15
    )

    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.tick_params(colors="#94a3b8")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#e2e8f0")
        label.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


def plot_support_confidence_scatter(rules: pd.DataFrame,
                                    save_path: Path) -> None:
    """
    Scatter plot: Support vs Confidence, coloured by Lift.

    WHY this chart?
    It shows the TRADE-OFF between the three metrics simultaneously.
    You want rules in the top-right (high support + high confidence)
    with warm colours (high lift). Rules in the bottom-left are
    technically valid but too rare to be actionable.

    This chart helps you tune your min_support and min_confidence
    thresholds — if all your rules cluster in one corner, adjust.

    Args:
        rules (pd.DataFrame): Full association rules dataframe
        save_path (Path): Output PNG path
    """
    if rules.empty:
        log.warning("  No rules to plot — skipping scatter chart")
        return

    # Sample if too many rules (for readability)
    plot_rules = rules if len(rules) <= 500 else rules.sample(500, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(SURFACE)

    sc = ax.scatter(
        plot_rules["support"],
        plot_rules["confidence"],
        c=plot_rules["lift"],
        cmap="YlOrRd",
        alpha=0.7,
        s=60,
        edgecolors="none",
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Lift", color="#e2e8f0", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")

    ax.set_xlabel("Support  (how common is this combination?)",
                  color="#64748b", fontsize=10)
    ax.set_ylabel("Confidence  (given A, how often is B bought?)",
                  color="#64748b", fontsize=10)
    ax.set_title(
        "Association Rules: Support vs Confidence\n"
        "Colour = Lift  |  Top-right + warm colour = best rules",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=15
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.tick_params(colors="#94a3b8")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#94a3b8")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


def plot_product_network(top_pairs: pd.DataFrame, save_path: Path) -> None:
    """
    Network graph of top product associations.

    Nodes = products, Edges = association rules.
    Edge thickness = confidence, Edge colour = lift strength.

    WHY a network graph?
    The bar chart shows individual rules. The network shows the
    ECOSYSTEM of associations — which products are hubs (connected
    to many others) vs islands (few connections). Hub products are
    ideal candidates for bundle promotions.

    Uses matplotlib only (no networkx dependency required).

    Args:
        top_pairs (pd.DataFrame): Top product pairs dataframe
        save_path (Path): Output PNG path
    """
    if top_pairs.empty:
        log.warning("  No pairs for network — skipping network chart")
        return

    # Use top 12 pairs for a clean network
    pairs = top_pairs.head(12).copy()

    # Shorten names
    def shorten(name, max_len=20):
        return name[:max_len] + "…" if len(name) > max_len else name

    pairs["ant_short"] = pairs["antecedents_str"].apply(shorten)
    pairs["con_short"] = pairs["consequents_str"].apply(shorten)

    # Collect unique nodes
    nodes = list(set(
        pairs["ant_short"].tolist() + pairs["con_short"].tolist()
    ))
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Place nodes in a circle
    angles = [2 * np.pi * i / n for i in range(n)]
    positions = {
        node: (np.cos(angles[i]), np.sin(angles[i]))
        for i, node in enumerate(nodes)
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_aspect("equal")
    ax.axis("off")

    strength_colors = {
        "Very Strong": "#10b981",
        "Strong":      "#3b82f6",
        "Moderate":    "#f59e0b",
        "Weak":        "#94a3b8",
    }

    # Draw edges
    for _, row in pairs.iterrows():
        x0, y0 = positions[row["ant_short"]]
        x1, y1 = positions[row["con_short"]]
        color = strength_colors.get(row["strength"], "#94a3b8")
        lw = max(1, row["confidence"] * 6)   # thickness = confidence

        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=0.7,
                mutation_scale=15,
            ),
        )

    # Draw nodes
    for node, (x, y) in positions.items():
        ax.scatter(x, y, s=800, color="#1a1e2a",
                   edgecolors="#f97316", linewidths=2, zorder=3)
        ax.text(x, y - 0.15, node,
                ha="center", va="top",
                color="#e2e8f0", fontsize=7, fontweight="bold",
                wrap=True)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, linewidth=3, label=s)
        for s, c in strength_colors.items()
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.2,
        labelcolor="#e2e8f0",
        fontsize=9,
    )
    legend.get_frame().set_facecolor("#1a1e2a")

    ax.set_title(
        "Product Association Network\n"
        "Arrows = buy direction  |  Thickness = Confidence  |  Colour = Lift strength",
        color="#e2e8f0", fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {save_path.name}")


# In[28]:


#─────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_mba_pipeline() -> dict:
    """
    Orchestrates the full Market Basket Analysis pipeline.

    Steps:
        1. Load and filter data (UK only)
        2. Filter to top-N products (memory management)
        3. Build binary basket matrix
        4. Run FP-Growth to find frequent itemsets
        5. Generate and filter association rules
        6. Extract top product pairs
        7. Export 3 CSVs
        8. Generate 3 visualisations

    Returns:
        dict: {
            'rules'      : full association rules dataframe,
            'itemsets'   : frequent itemsets dataframe,
            'top_pairs'  : top product pairs dataframe
        }
    """
    if not MLXTEND_AVAILABLE:
        print("\n❌ Cannot run pipeline — mlxtend not installed.")
        print("   Run: pip install mlxtend")
        print("   Then restart your kernel and re-run.\n")
        return {}

    print("\n" + "=" * 60)
    print("  MARKET BASKET ANALYSIS PIPELINE")
    print("  E-Commerce CLV Optimization — uk-retail-analytics")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────
    print("\n[1/8] Loading and filtering data ...")
    df = load_data(INPUT_FILE, uk_only=MBA_CONFIG["uk_only"])

    # ── Step 2: Filter products ───────────────────────────────────
    print(f"\n[2/8] Filtering to top {MBA_CONFIG['top_n_products']} products ...")
    df = filter_top_products(df, top_n=MBA_CONFIG["top_n_products"])

    # ── Step 3: Build basket matrix ───────────────────────────────
    print("\n[3/8] Building invoice × product basket matrix ...")
    basket = build_basket_matrix(df)

    # ── Step 4: FP-Growth ─────────────────────────────────────────
    print(f"\n[4/8] Running FP-Growth (min_support={MBA_CONFIG['min_support']}) ...")
    print("      This may take 1–3 minutes on 541K rows ...")
    itemsets = run_fpgrowth(basket, min_support=MBA_CONFIG["min_support"])

    if itemsets.empty:
        print("\n⚠️  No frequent itemsets found.")
        print("   Try lowering min_support in MBA_CONFIG (e.g. 0.005)")
        return {}

    # ── Step 5: Generate rules ────────────────────────────────────
    print(f"\n[5/8] Generating association rules ...")
    rules = generate_rules(
        itemsets,
        min_confidence=MBA_CONFIG["min_confidence"],
        min_lift=MBA_CONFIG["min_lift"],
    )

    if rules.empty:
        print("\n⚠️  No rules found above thresholds.")
        print("   Try lowering min_confidence in MBA_CONFIG (e.g. 0.2)")
        return {}

    # ── Step 6: Top pairs ─────────────────────────────────────────
    print("\n[6/8] Extracting top product pairs ...")
    top_pairs = build_top_pairs(rules, top_n=MBA_CONFIG["top_n_rules"])
    log.info(f"  Top pairs extracted : {len(top_pairs)}")

    # ── Step 7: Save CSVs ─────────────────────────────────────────
    print("\n[7/8] Saving outputs ...")

    # Save with string columns only (frozensets not CSV-friendly)
    rules_out = rules[[
        "antecedents_str", "consequents_str", "rule_str",
        "support", "confidence", "lift", "strength"
    ]].copy()
    rules_out.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    itemsets_out = itemsets.copy()
    itemsets_out["itemsets"] = itemsets_out["itemsets"].apply(
        lambda x: ", ".join(sorted(x))
    )
    itemsets_out.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)

    top_pairs.to_csv(OUTPUT_DIR / "top_product_pairs.csv", index=False)

    log.info(f"  Saved: association_rules.csv     ({len(rules):,} rules)")
    log.info(f"  Saved: frequent_itemsets.csv     ({len(itemsets):,} itemsets)")
    log.info(f"  Saved: top_product_pairs.csv     ({len(top_pairs)} pairs)")

    # ── Step 8: Visualisations ────────────────────────────────────
    print("\n[8/8] Generating visualisations ...")
    plot_top_rules(top_pairs,
                   REPORT_DIR / "mba_top_rules.png")
    plot_support_confidence_scatter(rules,
                                    REPORT_DIR / "mba_support_confidence_scatter.png")
    plot_product_network(top_pairs,
                         REPORT_DIR / "mba_product_network.png")

    # ── Console Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TOP 10 PRODUCT ASSOCIATIONS")
    print("=" * 60)
    if not top_pairs.empty:
        display_cols = ["rule_str", "confidence", "lift", "strength"]
        print(top_pairs[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("  BUSINESS INSIGHTS")
    print("=" * 60)
    print(f"\n  Total rules found       : {len(rules):,}")
    print(f"  Very Strong (lift ≥ 5)  : {(rules['strength']=='Very Strong').sum()}")
    print(f"  Strong (lift 3–5)       : {(rules['strength']=='Strong').sum()}")

    if not top_pairs.empty:
        best = top_pairs.iloc[0]
        print(f"\n  STRONGEST RULE:")
        print(f"  {best['rule_str']}")
        print(f"  Confidence : {best['confidence']:.1%}")
        print(f"  Lift       : {best['lift']:.2f}x")
        print(f"  Meaning    : {best['confidence']:.0%} of customers who bought")
        print(f"               [{best['antecedents_str']}]")
        print(f"               also bought [{best['consequents_str']}]")
        print(f"               — {best['lift']:.1f}x more likely than random chance")

    print("\n" + "=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    print("  data/processed/association_rules.csv")
    print("  data/processed/frequent_itemsets.csv")
    print("  data/processed/top_product_pairs.csv")
    print("  reports/mba_top_rules.png")
    print("  reports/mba_support_confidence_scatter.png")
    print("  reports/mba_product_network.png")

    return {
        "rules":     rules,
        "itemsets":  itemsets,
        "top_pairs": top_pairs,
    }


# In[29]:


#─────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_mba_pipeline()


# In[ ]:




