-- =============================================================
-- schema_creation.sql  (PostgreSQL version)
-- =============================================================
-- Project : E-Commerce Customer Lifetime Value Optimization
-- Dataset : UCI Online Retail — 541K+ UK transactions (2010-2011)
-- Database: PostgreSQL (tested on pgAdmin4)
--
-- HOW TO RUN IN pgAdmin4:
--   1. Open pgAdmin4
--   2. Create database named "uk_retail"
--      (right-click Databases → Create → Database → uk_retail)
--   3. Click on uk_retail → Tools → Query Tool
--   4. Open this file (folder icon) or paste contents
--   5. Press F5 or click the Run button
--   6. You should see "Query returned successfully"
--
-- SCHEMA DESIGN — STAR SCHEMA:
--
--         dim_customers
--               |
-- dim_products — fact_transactions — dim_date
--               |
--         dim_geography
-- =============================================================


-- =============================================================
-- SAFETY: Drop tables cleanly
-- CASCADE drops dependent foreign key constraints automatically
-- Order: fact table first, then dimensions
-- =============================================================

DROP TABLE IF EXISTS fact_transactions  CASCADE;
DROP TABLE IF EXISTS dim_customers      CASCADE;
DROP TABLE IF EXISTS dim_products       CASCADE;
DROP TABLE IF EXISTS dim_date           CASCADE;
DROP TABLE IF EXISTS dim_geography      CASCADE;
DROP TABLE IF EXISTS dim_rfm_segments   CASCADE;


-- =============================================================
-- DIMENSION TABLE 1: dim_customers
-- =============================================================
CREATE TABLE dim_customers (
    CustomerID          INTEGER         PRIMARY KEY,
    Recency             INTEGER,
    Frequency           INTEGER,
    Monetary            NUMERIC(12,2),
    R_Score             SMALLINT        CHECK (R_Score BETWEEN 1 AND 5),
    F_Score             SMALLINT        CHECK (F_Score BETWEEN 1 AND 5),
    M_Score             SMALLINT        CHECK (M_Score BETWEEN 1 AND 5),
    RFM_Total           SMALLINT,
    RFM_Score           VARCHAR(3),
    Segment             VARCHAR(50),
    LastPurchaseDate    DATE,
    AvgOrderValue       NUMERIC(12,2)
);

CREATE INDEX idx_customers_segment  ON dim_customers(Segment);
CREATE INDEX idx_customers_rfm      ON dim_customers(RFM_Total);


-- =============================================================
-- DIMENSION TABLE 2: dim_products
-- =============================================================
CREATE TABLE dim_products (
    StockCode               VARCHAR(20)     PRIMARY KEY,
    Description             TEXT,
    AvgUnitPrice            NUMERIC(10,2),
    TotalQuantitySold       INTEGER,
    TotalRevenue            NUMERIC(14,2),
    UniqueCustomers         INTEGER,
    UniqueInvoices          INTEGER,
    PopularityScore         NUMERIC(6,4)
);

CREATE INDEX idx_products_revenue     ON dim_products(TotalRevenue DESC);
CREATE INDEX idx_products_description ON dim_products(Description);


-- =============================================================
-- DIMENSION TABLE 3: dim_date
-- =============================================================
CREATE TABLE dim_date (
    DateKey             DATE            PRIMARY KEY,
    Year                SMALLINT,
    Month               SMALLINT,
    MonthName           VARCHAR(12),
    Quarter             SMALLINT,
    QuarterName         VARCHAR(2),
    Day                 SMALLINT,
    DayOfWeek           SMALLINT,
    DayName             VARCHAR(10),
    IsWeekend           SMALLINT,
    YearMonth           VARCHAR(7),
    YearQuarter         VARCHAR(7)
);

CREATE INDEX idx_date_yearmonth   ON dim_date(YearMonth);
CREATE INDEX idx_date_year        ON dim_date(Year);
CREATE INDEX idx_date_quarter     ON dim_date(Quarter);


-- =============================================================
-- DIMENSION TABLE 4: dim_geography
-- =============================================================
CREATE TABLE dim_geography (
    Country             VARCHAR(50)     PRIMARY KEY,
    IsUK                SMALLINT        DEFAULT 0,
    TotalRevenue        NUMERIC(14,2),
    TotalOrders         INTEGER,
    UniqueCustomers     INTEGER,
    AvgOrderValue       NUMERIC(10,2),
    RevenueRank         INTEGER
);


-- =============================================================
-- DIMENSION TABLE 5: dim_rfm_segments (reference table)
-- =============================================================
CREATE TABLE dim_rfm_segments (
    Segment             VARCHAR(50)     PRIMARY KEY,
    Description         TEXT,
    MarketingAction     TEXT,
    Priority            SMALLINT,
    ColorHex            VARCHAR(7)
);

INSERT INTO dim_rfm_segments VALUES
    ('Champions',
     'Bought recently, buy often, spend the most',
     'Reward them. Early access. Ask for reviews.',
     1, '#10b981'),
    ('Loyal Customers',
     'Buy regularly with solid frequency',
     'Upsell. Loyalty programme. Referral asks.',
     2, '#3b82f6'),
    ('Potential Loyalists',
     'Recent buyers with low frequency — room to grow',
     'Membership offers. Personalised recommendations.',
     3, '#6366f1'),
    ('At Risk',
     'Were good customers, now going cold',
     'Win-back email. Renewal discount.',
     4, '#f59e0b'),
    ('Hibernating',
     'Low recency and low frequency — gone quiet',
     'Reactivation campaign. Low-cost channel.',
     5, '#94a3b8'),
    ('Lost',
     'Have not bought in a very long time',
     'Ignore or very cheap re-engagement only.',
     6, '#475569'),
    ('New Customers',
     'Very recent, only 1-2 orders so far',
     'Onboarding flow. First-purchase follow-up.',
     7, '#8b5cf6'),
    ('Promising',
     'Recent-ish buyers with some frequency',
     'Brand awareness. Free trials or samples.',
     8, '#a78bfa'),
    ('Cannot Lose Them',
     'High historical value but have disappeared',
     'Personal outreach. Special offer. ASAP.',
     9, '#ef4444');


-- =============================================================
-- FACT TABLE: fact_transactions
-- PostgreSQL uses SERIAL instead of AUTOINCREMENT
-- =============================================================
CREATE TABLE fact_transactions (
    TransactionID       SERIAL          PRIMARY KEY,
    CustomerID          INTEGER         REFERENCES dim_customers(CustomerID),
    StockCode           VARCHAR(20)     REFERENCES dim_products(StockCode),
    DateKey             DATE            REFERENCES dim_date(DateKey),
    Country             VARCHAR(50)     REFERENCES dim_geography(Country),
    InvoiceNo           VARCHAR(20)     NOT NULL,
    InvoiceDate         TIMESTAMP       NOT NULL,
    Quantity            INTEGER         NOT NULL,
    UnitPrice           NUMERIC(10,2)   NOT NULL,
    TotalAmount         NUMERIC(12,2)   NOT NULL,
    Year                SMALLINT,
    Month               SMALLINT,
    Quarter             SMALLINT,
    DayOfWeek           SMALLINT,
    IsWeekend           SMALLINT,
    YearMonth           VARCHAR(7)
);

CREATE INDEX idx_fact_customer   ON fact_transactions(CustomerID);
CREATE INDEX idx_fact_date       ON fact_transactions(DateKey);
CREATE INDEX idx_fact_yearmonth  ON fact_transactions(YearMonth);
CREATE INDEX idx_fact_year_month ON fact_transactions(Year, Month);
CREATE INDEX idx_fact_quarter    ON fact_transactions(Quarter);
CREATE INDEX idx_fact_stockcode  ON fact_transactions(StockCode);
CREATE INDEX idx_fact_country    ON fact_transactions(Country);
CREATE INDEX idx_fact_invoice    ON fact_transactions(InvoiceNo);


-- =============================================================
-- VERIFY SCHEMA CREATED (run this after pressing F5)
-- =============================================================
SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns
     WHERE table_name = t.table_name
     AND table_schema = 'public') AS column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
ORDER BY table_name;
