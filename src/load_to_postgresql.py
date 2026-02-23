#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('pip install psycopg2-binary sqlalchemy')


# In[16]:


import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, exc as sa_exc
from sqlalchemy.engine import Engine


# In[17]:


warnings.filterwarnings("ignore")


# In[18]:


# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl_pipeline.log")
    ]
)
log = logging.getLogger(__name__)


# In[19]:


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_db_config() -> dict:
    """
    Retrieve database configuration from environment variables.
    Falls back to defaults for local development only.
    """
    password = os.getenv("DB_PASSWORD")
    if not password:
        log.warning("DB_PASSWORD not set in environment, using fallback (INSECURE)")
        password = "Ln@jy"  # Only for development
    
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "ecommerce_clv_optimization"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": password,
    }


def get_file_paths() -> dict:
    """Resolve file paths relative to script location."""
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path(os.getcwd())
    
    return {
        "base_dir": base_dir,
        "input_sales": base_dir / "cleaned_sales_data.csv",
        "input_rfm": base_dir / "data" / "processed" / "rfm_scores.csv",
        "schema_file": base_dir / "sql" / "schema_creation_postgresql.sql",
    }


# In[20]:


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def create_connection_string(config: dict) -> str:
    """Create SQLAlchemy connection string with proper encoding."""
    password = quote_plus(config["password"])
    return (
        f"postgresql://{config['username']}:{password}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


def get_engine(config: Optional[dict] = None) -> Engine:
    """
    Create and test database engine with connection pooling.
    
    Args:
        config: Database configuration dict. If None, loads from environment.
    
    Returns:
        SQLAlchemy Engine instance
    
    Raises:
        ConnectionError: If database connection fails
    """
    if config is None:
        config = get_db_config()
    
    conn_str = create_connection_string(config)
    
    engine = create_engine(
        conn_str,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False,
    )
    
    # Test connection with timeout
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        log.info("PostgreSQL connection successful")
        return engine
    except sa_exc.OperationalError as e:
        raise ConnectionError(
            f"\n Could not connect to PostgreSQL.\n"
            f"   Error: {e}\n\n"
            f"   Checklist:\n"
            f"   1. PostgreSQL server is running\n"
            f"   2. Database '{config['database']}' exists\n"
            f"   3. Credentials are correct (host: {config['host']}, port: {config['port']})\n"
            f"   4. Network/firewall allows connection\n"
        ) from e
    except Exception as e:
        raise ConnectionError(f"Unexpected error connecting to database: {e}") from e


# In[21]:


# =============================================================================
# SCHEMA MANAGEMENT
# =============================================================================

def create_schema(engine: Engine, schema_file: Path) -> None:
    """
    Execute schema creation SQL file with proper error handling.
    
    Only ignores 'already exists' errors, fails on actual problems.
    """
    if not schema_file.exists():
        log.warning(f"Schema file not found at {schema_file} — assuming schema already exists")
        return
    
    log.info(f"Executing schema file: {schema_file}")
    
    with open(schema_file, "r", encoding="utf-8") as f:
        sql_content = f.read()
    
    # Split into individual statements
    statements = [
        s.strip() for s in sql_content.split(";") 
        if s.strip() and not s.strip().startswith("--")
    ]
    
    with engine.begin() as conn:  # Auto-commit/rollback transaction
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(text(stmt))
                log.debug(f"  Executed statement {i}/{len(statements)}")
            except sa_exc.ProgrammingError as e:
                # Check if it's an "already exists" error
                error_msg = str(e).lower()
                if any(x in error_msg for x in ["already exists", "duplicate", "42710"]):
                    log.debug(f"  Skipping statement {i}: object already exists")
                else:
                    log.error(f"  SQL Error in statement {i}: {stmt[:100]}...")
                    raise
            except Exception as e:
                log.error(f"  Unexpected error in statement {i}: {e}")
                raise
    
    log.info("Schema creation completed")


# In[22]:


def truncate_table(engine: Engine, table_name: str) -> None:
    """Truncate table using raw SQL to avoid FK issues."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
        log.info(f"  Truncated {table_name}")


def load_dim_customers(engine: Engine, rfm: pd.DataFrame, sales: pd.DataFrame) -> int:
    """Load customer dimension table with derived metrics."""
    log.info("Loading dim_customers ...")
    df = rfm.copy()
    
    # Handle last purchase date
    if "LastPurchaseDate" in df.columns:
        df["LastPurchaseDate"] = pd.to_datetime(df["LastPurchaseDate"]).dt.date
    else:
        last_dates = (
            sales.groupby("CustomerID")["InvoiceDate"]
            .max()
            .dt.date
            .reset_index()
            .rename(columns={"InvoiceDate": "LastPurchaseDate"})
        )
        df = df.merge(last_dates, on="CustomerID", how="left")
    
    # Calculate average order value safely
    df["AvgOrderValue"] = df.apply(
        lambda row: round(row["Monetary"] / row["Frequency"], 2) 
        if row["Frequency"] > 0 else 0.0, 
        axis=1
    )
    
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Select and reorder columns
    desired_cols = [
        "customerid", "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_total", "rfm_score",
        "segment", "lastpurchasedate", "avgordervalue"
    ]
    available_cols = [c for c in desired_cols if c in df.columns]
    df = df[available_cols]
    
    # Use append instead of replace to avoid FK constraint issues
    # First truncate if table exists
    try:
        truncate_table(engine, "dim_customers")
        if_exists = "append"
    except Exception:
        if_exists = "replace"
    
    df.to_sql(
        "dim_customers", 
        engine, 
        if_exists=if_exists,
        index=False, 
        method="multi", 
        chunksize=1000
    )
    
    log.info(f"  [OK] dim_customers: {len(df):,} rows loaded")
    return len(df)


def load_dim_products(engine: Engine, sales: pd.DataFrame) -> int:
    """Load product dimension table with aggregations."""
    log.info("Loading dim_products ...")
    
    products = (
        sales.groupby("StockCode")
        .agg(
            Description=("Description", "first"),
            AvgUnitPrice=("UnitPrice", "mean"),
            TotalQuantitySold=("Quantity", "sum"),
            TotalRevenue=("TotalAmount", "sum"),
            UniqueCustomers=("CustomerID", "nunique"),
            UniqueInvoices=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )
    
    max_cust = products["UniqueCustomers"].max()
    products["PopularityScore"] = (
        products["UniqueCustomers"] / max_cust if max_cust > 0 else 0
    ).round(4)
    
    products["AvgUnitPrice"] = products["AvgUnitPrice"].round(2)
    products["TotalRevenue"] = products["TotalRevenue"].round(2)
    products.columns = [c.lower() for c in products.columns]
    
    try:
        truncate_table(engine, "dim_products")
        if_exists = "append"
    except Exception:
        if_exists = "replace"
    
    products.to_sql(
        "dim_products", 
        engine, 
        if_exists=if_exists,
        index=False, 
        method="multi", 
        chunksize=1000
    )
    
    log.info(f"  [OK] dim_products: {len(products):,} rows loaded")
    return len(products)


def load_dim_date(engine: Engine, sales: pd.DataFrame) -> int:
    """Load date dimension table with calendar attributes."""
    log.info("Loading dim_date ...")
    
    unique_dates = pd.to_datetime(
        sales["InvoiceDate"].dt.date.unique()
    ).sort_values()
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                 "Friday", "Saturday", "Sunday"]
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    dim_date = pd.DataFrame({
        "datekey": unique_dates,
        "year": unique_dates.year,
        "month": unique_dates.month,
        "monthname": [month_names[m-1] for m in unique_dates.month],
        "quarter": unique_dates.quarter,
        "quartername": [f"Q{q}" for q in unique_dates.quarter],
        "day": unique_dates.day,
        "dayofweek": unique_dates.dayofweek,
        "dayname": [day_names[d] for d in unique_dates.dayofweek],
        "isweekend": (unique_dates.dayofweek >= 5).astype(int),
        "yearmonth": unique_dates.strftime("%Y-%m"),
        "yearquarter": unique_dates.strftime("%Y") + "-Q" + unique_dates.quarter.astype(str),
    })
    
    try:
        truncate_table(engine, "dim_date")
        if_exists = "append"
    except Exception:
        if_exists = "replace"
    
    dim_date.to_sql(
        "dim_date", 
        engine, 
        if_exists=if_exists,
        index=False, 
        method="multi", 
        chunksize=1000
    )
    
    log.info(f"  [OK] dim_date: {len(dim_date):,} rows loaded")
    return len(dim_date)


def load_dim_geography(engine: Engine, sales: pd.DataFrame) -> int:
    """Load geography dimension table with country metrics."""
    log.info("Loading dim_geography ...")
    
    geo = (
        sales.groupby("Country")
        .agg(
            TotalRevenue=("TotalAmount", "sum"),
            TotalOrders=("InvoiceNo", "nunique"),
            UniqueCustomers=("CustomerID", "nunique"),
        )
        .reset_index()
    )
    
    geo["IsUK"] = (geo["Country"] == "United Kingdom").astype(int)
    geo["AvgOrderValue"] = geo.apply(
        lambda row: round(row["TotalRevenue"] / row["TotalOrders"], 2) 
        if row["TotalOrders"] > 0 else 0.0,
        axis=1
    )
    geo["TotalRevenue"] = geo["TotalRevenue"].round(2)
    geo["RevenueRank"] = (
        geo["TotalRevenue"].rank(ascending=False, method="dense").astype(int)
    )
    geo.columns = [c.lower() for c in geo.columns]
    
    try:
        truncate_table(engine, "dim_geography")
        if_exists = "append"
    except Exception:
        if_exists = "replace"
    
    geo.to_sql(
        "dim_geography", 
        engine, 
        if_exists=if_exists,
        index=False, 
        method="multi",
        chunksize=1000
    )
    
    log.info(f"  [OK] dim_geography: {len(geo):,} rows loaded")
    return len(geo)


def load_fact_transactions(engine: Engine, sales: pd.DataFrame) -> int:
    """Load fact table with transaction data using memory-efficient chunking."""
    log.info(f"Loading fact_transactions ({len(sales):,} rows) ...")
    
    # Clear existing data first
    try:
        truncate_table(engine, "fact_transactions")
    except Exception as e:
        log.warning(f"Could not truncate fact_transactions: {e}")
    
    # Process in chunks to avoid memory issues
    chunk_size = 50000
    total_rows = len(sales)
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
    
    rows_loaded = 0
    
    for i, start in enumerate(range(0, total_rows, chunk_size)):
        end = min(start + chunk_size, total_rows)
        chunk = sales.iloc[start:end].copy()
        
        # Transformations
        chunk["InvoiceDate"] = pd.to_datetime(chunk["InvoiceDate"])
        chunk["DateKey"] = chunk["InvoiceDate"].dt.date
        chunk["Year"] = chunk["InvoiceDate"].dt.year
        chunk["Month"] = chunk["InvoiceDate"].dt.month
        chunk["Quarter"] = chunk["InvoiceDate"].dt.quarter
        chunk["DayOfWeek"] = chunk["InvoiceDate"].dt.dayofweek
        chunk["IsWeekend"] = (chunk["DayOfWeek"] >= 5).astype(int)
        chunk["YearMonth"] = chunk["InvoiceDate"].dt.strftime("%Y-%m")
        
        if "TotalAmount" not in chunk.columns:
            chunk["TotalAmount"] = chunk["Quantity"] * chunk["UnitPrice"]
        
        schema_cols = [
            "CustomerID", "StockCode", "DateKey", "Country",
            "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "TotalAmount",
            "Year", "Month", "Quarter", "DayOfWeek", "IsWeekend", "YearMonth",
        ]
        chunk = chunk[[c for c in schema_cols if c in chunk.columns]]
        chunk.columns = [c.lower() for c in chunk.columns]
        
        # Load chunk
        mode = "append" if i > 0 or rows_loaded > 0 else "replace"
        chunk.to_sql(
            "fact_transactions", 
            engine, 
            if_exists=mode,
            index=False, 
            method="multi", 
            chunksize=5000
        )
        
        rows_loaded += len(chunk)
        log.info(f"  Chunk {i+1}/{total_chunks}: {rows_loaded:,} / {total_rows:,} rows")
        
        del chunk
    
    log.info(f"  [OK] fact_transactions: {rows_loaded:,} rows loaded")
    return rows_loaded


# In[23]:


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_load(engine: Engine) -> dict:
    """
    Verify loaded data and return statistics.
    
    Returns:
        Dictionary with verification results
    """
    print("\n" + "=" * 60)
    print("  DATABASE LOAD VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Table row counts with minimum expectations
    tables = [
        ("fact_transactions", 390_000),
        ("dim_customers", 4_000),
        ("dim_products", 3_000),
        ("dim_date", 300),
        ("dim_geography", 30),
    ]
    
    for table, expected_min in tables:
        try:
            count = pd.read_sql(
                f"SELECT COUNT(*) AS n FROM {table}", 
                engine
            ).iloc[0, 0]
            status = "OK" if count >= expected_min else "⚠️"
            print(f"[ERROR]{status}  {table:<25} : {count:>8,} rows (min: {expected_min:,})")
            results[table] = {"count": count, "expected": expected_min, "ok": count >= expected_min}
        except Exception as e:
            print(f" {table:<25} : ERROR - {e}")
            results[table] = {"error": str(e)}
    
    # Top customers
    print("\n Top 5 customers by lifetime spend:")
    try:
        top_customers = pd.read_sql("""
            SELECT customerid, segment, monetary, frequency, r_score, f_score, m_score
            FROM dim_customers 
            ORDER BY monetary DESC 
            LIMIT 5
        """, engine)
        print(top_customers.to_string(index=False))
        results["top_customers"] = top_customers.to_dict()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Revenue by segment
    print("\n Revenue by RFM segment:")
    try:
        segment_revenue = pd.read_sql("""
            SELECT 
                c.segment,
                COUNT(DISTINCT f.customerid) AS customers,
                ROUND(SUM(f.totalamount)::NUMERIC, 0) AS total_revenue,
                ROUND(AVG(f.totalamount)::NUMERIC, 2) AS avg_line_value
            FROM fact_transactions f
            JOIN dim_customers c ON f.customerid = c.customerid
            GROUP BY c.segment 
            ORDER BY total_revenue DESC
        """, engine)
        print(segment_revenue.to_string(index=False))
        results["segment_revenue"] = segment_revenue.to_dict()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Top countries
    print("\n Top 5 countries by revenue:")
    try:
        top_countries = pd.read_sql("""
            SELECT country, totalrevenue, totalorders, uniquecustomers, revenuerank
            FROM dim_geography 
            ORDER BY revenuerank 
            LIMIT 5
        """, engine)
        print(top_countries.to_string(index=False))
        results["top_countries"] = top_countries.to_dict()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print(f"  Database: {engine.url.database} @ {engine.url.host}:{engine.url.port}")
    print("=" * 60)
    
    return results


# In[24]:


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def validate_inputs(paths: dict) -> None:
    """Validate that all required input files exist."""
    missing = []
    
    if not paths["input_sales"].exists():
        missing.append(f"Sales data: {paths['input_sales']} (Run 01_data_cleaning.py)")
    
    if not paths["input_rfm"].exists():
        missing.append(f"RFM data: {paths['input_rfm']} (Run 03_rfm_analysis.py)")
    
    if missing:
        raise FileNotFoundError("\n".join(["Missing required files:"] + missing))


def load_source_data(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate source CSV files."""
    log.info("Loading source CSV files ...")
    
    # Load sales data
    sales = pd.read_csv(
        paths["input_sales"], 
        parse_dates=["InvoiceDate"], 
        encoding="latin-1",
        low_memory=False
    )
    
    # Load RFM data
    rfm = pd.read_csv(paths["input_rfm"])
    
    # Calculate TotalAmount if missing
    if "TotalAmount" not in sales.columns:
        sales["TotalAmount"] = sales["Quantity"] * sales["UnitPrice"]
    
    log.info(f"  Sales: {len(sales):,} rows, {len(sales.columns)} columns")
    log.info(f"  RFM:   {len(rfm):,} rows, {len(rfm.columns)} columns")
    
    return sales, rfm


def run_pipeline(
    create_schema_flag: bool = False,
    skip_verification: bool = False
) -> Dict[str, Any]:
    """Execute full ETL pipeline."""
    print("\n" + "=" * 60)
    print("  POSTGRESQL ETL PIPELINE")
    print("  E-Commerce CLV Optimization")
    print("=" * 60)
    
    paths = get_file_paths()
    config = get_db_config()
    
    log.info(f"Base directory: {paths['base_dir']}")
    log.info(f"Target database: {config['database']} @ {config['host']}:{config['port']}")
    
    print("\n[1/7] Validating inputs ...")
    validate_inputs(paths)
    
    print("\n[2/7] Loading source data ...")
    sales, rfm = load_source_data(paths)
    
    print("\n[3/7] Connecting to PostgreSQL ...")
    engine = get_engine(config)
    
    if create_schema_flag:
        print("\n[3.5/7] Creating schema ...")
        create_schema(engine, paths["schema_file"])
    
    # IMPORTANT: Load dimensions FIRST, then fact table
    print("\n[4/7] Loading dimension tables ...")
    stats = {}
    stats["dim_date"] = load_dim_date(engine, sales)           # No dependencies
    stats["dim_geography"] = load_dim_geography(engine, sales) # No dependencies
    stats["dim_products"] = load_dim_products(engine, sales)   # No dependencies
    stats["dim_customers"] = load_dim_customers(engine, rfm, sales)  # Referenced by fact
    
    print("\n[5/7] Loading fact table ...")
    stats["fact_transactions"] = load_fact_transactions(engine, sales)
    
    del sales
    del rfm
    
    if not skip_verification:
        print("\n[6/7] Verifying load ...")
        stats["verification"] = verify_load(engine)
    
    print("\n[7/7] Cleaning up ...")
    engine.dispose()
    
    print("\n" + "=" * 60)
    print("  ETL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Total rows loaded: {sum(v for k, v in stats.items() if isinstance(v, int)):,}")
    print("=" * 60)
    
    return stats


# In[25]:


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Auto-detect Jupyter environment
    in_jupyter = any("ipykernel" in arg for arg in sys.argv)
    
    if in_jupyter:
        # Default settings for Jupyter
        create_schema = False
        skip_verify = False
        verbose = False
        print("Running in Jupyter mode (use run_pipeline() manually for custom args)")
    else:
        # Terminal mode with argparse
        import argparse
        parser = argparse.ArgumentParser(description="ETL Pipeline for E-Commerce Data")
        parser.add_argument("--create-schema", action="store_true")
        parser.add_argument("--skip-verify", action="store_true")
        parser.add_argument("-v", "--verbose", action="store_true")
        args = parser.parse_args()
        
        create_schema = args.create_schema
        skip_verify = args.skip_verify
        verbose = args.verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = run_pipeline(
            create_schema_flag=create_schema,
            skip_verification=skip_verify
        )
        if not in_jupyter:
            sys.exit(0)
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        if not in_jupyter:
            sys.exit(1)

