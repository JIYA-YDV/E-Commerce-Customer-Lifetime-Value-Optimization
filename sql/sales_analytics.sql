-- =============================================================
-- sales_analytics.sql
-- =============================================================
-- QUERY 1: Monthly Revenue Trend
-- =============================================================
-- WHY: The most fundamental business metric. Shows whether
-- revenue is growing, declining, or seasonal. Month-over-month
-- growth % tells you if your business is accelerating.
-- This feeds directly into your Power BI line chart.
-- =============================================================

SELECT
    f.yearmonth                                         AS "Month",
    d.monthname                                         AS "Month_Name",
    d.year                                              AS "Year",
    d.quarter                                           AS "Quarter",
    COUNT(DISTINCT f.invoiceno)                         AS "Total_Orders",
    COUNT(DISTINCT f.customerid)                        AS "Unique_Customers",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Total_Revenue_GBP",
    ROUND(AVG(f.totalamount)::NUMERIC, 2)               AS "Avg_Line_Value",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP",
    -- Month-over-month revenue growth %
    ROUND(
        ((SUM(f.totalamount) - LAG(SUM(f.totalamount))
            OVER (ORDER BY f.yearmonth)) * 100.0 /
        NULLIF(LAG(SUM(f.totalamount))
            OVER (ORDER BY f.yearmonth), 0))::NUMERIC
    , 1)                                                AS "MoM_Growth_%",
    -- Running cumulative revenue
    ROUND(SUM(SUM(f.totalamount))
        OVER (ORDER BY f.yearmonth ROWS UNBOUNDED PRECEDING)::NUMERIC
    , 0)                                                AS "Cumulative_Revenue_GBP"
FROM fact_transactions f
JOIN dim_date d ON f.datekey = d.datekey
GROUP BY f.yearmonth, d.monthname, d.year, d.quarter
ORDER BY f.yearmonth;


-- =============================================================
-- QUERY 2: Quarterly Revenue Summary
-- =============================================================
-- WHY: Quarterly view smooths out weekly noise and shows
-- the macro trend. Q4 should show the Christmas spike.
-- Useful for board-level reporting and portfolio presentation.
-- =============================================================

SELECT
    d.yearquarter                                       AS "Quarter",
    d.year                                              AS "Year",
    d.quarter                                           AS "Q",
    COUNT(DISTINCT f.invoiceno)                         AS "Total_Orders",
    COUNT(DISTINCT f.customerid)                        AS "Unique_Customers",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Total_Revenue_GBP",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP",
    -- Quarter-over-quarter growth
    ROUND(
        ((SUM(f.totalamount) - LAG(SUM(f.totalamount))
            OVER (ORDER BY d.yearquarter)) * 100.0 /
        NULLIF(LAG(SUM(f.totalamount))
            OVER (ORDER BY d.yearquarter), 0))::NUMERIC
    , 1)                                                AS "QoQ_Growth_%"
FROM fact_transactions f
JOIN dim_date d ON f.datekey = d.datekey
GROUP BY d.yearquarter, d.year, d.quarter
ORDER BY d.yearquarter;


-- =============================================================
-- QUERY 3: Day of Week Sales Pattern
-- =============================================================
-- WHY: Identifies peak trading days. If Thursday generates
-- 40% more revenue than Monday, schedule email campaigns,
-- promotions, and stock replenishment accordingly.
-- Classic operational insight from transaction data.
-- =============================================================

SELECT
    d.dayofweek                                         AS "Day_Number",
    d.dayname                                           AS "Day",
    d.isweekend                                         AS "Is_Weekend",
    COUNT(DISTINCT f.invoiceno)                         AS "Total_Orders",
    COUNT(DISTINCT f.customerid)                        AS "Unique_Customers",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Total_Revenue_GBP",
    ROUND((AVG(SUM(f.totalamount)) OVER ())::NUMERIC, 0) AS "Daily_Avg_Revenue",
    ROUND(
        (SUM(f.totalamount) * 100.0 /
        SUM(SUM(f.totalamount)) OVER ())::NUMERIC
    , 1)                                                AS "% of Total Revenue",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP"
FROM fact_transactions f
JOIN dim_date d ON f.datekey = d.datekey
GROUP BY d.dayofweek, d.dayname, d.isweekend
ORDER BY d.dayofweek;


-- =============================================================
-- QUERY 4: Monthly Seasonality Pattern (averaged across years)
-- =============================================================
-- WHY: Shows which months are structurally stronger regardless
-- of year. November/December should be highest for a UK
-- retailer. Use this for inventory planning and budget
-- allocation — when to spend more on marketing.
-- =============================================================

SELECT
    d.month                                             AS "Month_Number",
    d.monthname                                         AS "Month",
    COUNT(DISTINCT f.invoiceno)                         AS "Total_Orders",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Total_Revenue_GBP",
    ROUND(AVG(f.totalamount)::NUMERIC, 2)               AS "Avg_Transaction_Value",
    ROUND(
        (SUM(f.totalamount) * 100.0 /
        SUM(SUM(f.totalamount)) OVER ())::NUMERIC
    , 1)                                                AS "% of Annual Revenue",
    -- Index vs average month (100 = average, >100 = above average)
    ROUND(
        ((SUM(f.totalamount) * 100.0) /
        (AVG(SUM(f.totalamount)) OVER ()))::NUMERIC
    , 0)                                                AS "Seasonality_Index"
FROM fact_transactions f
JOIN dim_date d ON f.datekey = d.datekey
GROUP BY d.month, d.monthname
ORDER BY d.month;


-- =============================================================
-- QUERY 5: Country Performance Ranking
-- =============================================================
-- WHY: Directly answers "which markets drive growth?"
-- Compares all 38 countries on revenue, orders, customers,
-- and average order value. High AOV + low customer count =
-- expansion opportunity (premium market not yet fully tapped).
-- =============================================================

SELECT
    g.country                                           AS "Country",
    g.isuk                                              AS "Is_UK",
    g.revenuerank                                       AS "Revenue_Rank",
    g.totalorders                                       AS "Total_Orders",
    g.uniquecustomers                                   AS "Unique_Customers",
    g.totalrevenue                                      AS "Total_Revenue_GBP",
    ROUND((g.totalrevenue * 100.0 /
        SUM(g.totalrevenue) OVER ())::NUMERIC
    , 2)                                                AS "% of Total Revenue",
    g.avgordervalue                                     AS "Avg_Order_Value_GBP",
    ROUND((g.totalrevenue /
        NULLIF(g.uniquecustomers, 0))::NUMERIC
    , 2)                                                AS "Revenue_per_Customer_GBP",
    -- Market classification
    CASE
        WHEN g.isuk = 1                           THEN 'Home Market'
        WHEN g.totalrevenue >= 50000              THEN 'Major Export Market'
        WHEN g.totalrevenue >= 10000              THEN 'Established Export'
        WHEN g.avgordervalue >
             AVG(g.avgordervalue) OVER ()         THEN 'Premium Small Market'
        ELSE                                           'Emerging Market'
    END                                                 AS "Market_Classification"
FROM dim_geography g
ORDER BY g.revenuerank;


-- =============================================================
-- QUERY 6: Top 20 Products by Revenue
-- =============================================================
-- WHY: Identifies your hero products — the ones generating
-- the most revenue. These should never be out of stock,
-- should be featured prominently, and are candidates for
-- bundle deals with complementary items from MBA analysis.
-- =============================================================

SELECT
    p.stockcode                                         AS "Stock_Code",
    p.description                                       AS "Product",
    p.totalrevenue                                      AS "Total_Revenue_GBP",
    ROUND((p.totalrevenue * 100.0 /
        SUM(p.totalrevenue) OVER ())::NUMERIC
    , 2)                                                AS "% of Total Revenue",
    p.totalquantitysold                                 AS "Units_Sold",
    p.avgunitprice                                      AS "Avg_Price_GBP",
    p.uniquecustomers                                   AS "Unique_Buyers",
    p.uniqueinvoices                                    AS "Invoices_Appeared_In",
    ROUND((p.popularityscore * 100)::NUMERIC, 1)        AS "Popularity_%",
    -- Product tier
    CASE
        WHEN p.totalrevenue >= 50000    THEN 'Hero Product'
        WHEN p.totalrevenue >= 20000    THEN 'Strong Performer'
        WHEN p.totalrevenue >= 10000    THEN 'Solid Contributor'
        ELSE                                 'Standard'
    END                                                 AS "Product_Tier"
FROM dim_products p
WHERE p.description IS NOT NULL
ORDER BY p.totalrevenue DESC
LIMIT 20;


-- =============================================================
-- QUERY 7: Average Order Value Trend Over Time
-- =============================================================
-- WHY: AOV is a key health metric. Rising AOV means customers
-- are buying more per visit (upsell success) or product
-- mix is shifting to higher-value items. Declining AOV
-- could mean discounting is hurting margins.
-- =============================================================

SELECT
    f.yearmonth                                         AS "Month",
    COUNT(DISTINCT f.invoiceno)                         AS "Orders",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Revenue_GBP",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP",
    -- 3-month rolling average AOV
    ROUND((
        AVG(
            SUM(f.totalamount) /
            NULLIF(COUNT(DISTINCT f.invoiceno), 0)
        ) OVER (
            ORDER BY f.yearmonth
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )
    )::NUMERIC, 2)                                      AS "3M_Rolling_Avg_AOV",
    -- AOV vs overall average
    ROUND((
        (SUM(f.totalamount) / NULLIF(COUNT(DISTINCT f.invoiceno), 0)) -
        AVG(SUM(f.totalamount) / NULLIF(COUNT(DISTINCT f.invoiceno), 0))
            OVER ()
    )::NUMERIC, 2)                                      AS "AOV_vs_Average"
FROM fact_transactions f
GROUP BY f.yearmonth
ORDER BY f.yearmonth;


-- =============================================================
-- QUERY 8: Revenue by UK vs International
-- =============================================================
-- WHY: Shows the UK vs international split — critical for
-- deciding where to focus marketing spend. If international
-- AOV is higher than UK, there's a strong case for
-- increasing international marketing budget.
-- =============================================================

SELECT
    CASE WHEN g.isuk = 1 THEN 'United Kingdom' ELSE 'International' END
                                                        AS "Market",
    COUNT(DISTINCT f.customerid)                        AS "Customers",
    COUNT(DISTINCT f.invoiceno)                         AS "Orders",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Revenue_GBP",
    ROUND((SUM(f.totalamount) * 100.0 /
        SUM(SUM(f.totalamount)) OVER ())::NUMERIC
    , 1)                                                AS "% of Revenue",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP",
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.customerid), 0))::NUMERIC
    , 2)                                                AS "Revenue_per_Customer_GBP"
FROM fact_transactions f
JOIN dim_geography g ON f.country = g.country
GROUP BY g.isuk
ORDER BY g.isuk DESC;


-- =============================================================
-- QUERY 9: New vs Returning Customer Revenue Split
-- =============================================================
-- WHY: One of the most important business metrics.
-- If >80% of revenue comes from returning customers,
-- retention is working. If acquisition dominates, the
-- business is expensive to run (high CAC dependency).
-- =============================================================

WITH customer_orders AS (
    SELECT
        customerid,
        invoiceno,
        yearmonth,
        totalamount,
        ROW_NUMBER() OVER (
            PARTITION BY customerid
            ORDER BY MIN(invoicedate)
        ) AS order_rank
    FROM fact_transactions
    GROUP BY customerid, invoiceno, yearmonth, totalamount, invoicedate
),
order_type AS (
    SELECT
        yearmonth,
        totalamount,
        CASE WHEN order_rank = 1 THEN 'New Customer' ELSE 'Returning Customer' END
            AS customer_type
    FROM customer_orders
)
SELECT
    yearmonth                                           AS "Month",
    customer_type                                       AS "Customer_Type",
    COUNT(*)                                            AS "Orders",
    ROUND(SUM(totalamount)::NUMERIC, 0)                 AS "Revenue_GBP",
    ROUND((SUM(totalamount) * 100.0 /
        SUM(SUM(totalamount)) OVER (PARTITION BY yearmonth))::NUMERIC
    , 1)                                                AS "% of Month Revenue"
FROM order_type
GROUP BY yearmonth, customer_type
ORDER BY yearmonth, customer_type;


-- =============================================================
-- QUERY 10: Executive Summary Dashboard Query
-- =============================================================
-- WHY: Single query that produces ALL key KPIs for the
-- executive summary page in Power BI. One source of truth
-- for headline numbers that appear at the top of your dashboard.
-- This is what you present first in your portfolio demo.
-- =============================================================

SELECT
    -- Revenue KPIs
    ROUND(SUM(f.totalamount)::NUMERIC, 0)               AS "Total_Revenue_GBP",
    COUNT(DISTINCT f.invoiceno)                         AS "Total_Orders",
    COUNT(DISTINCT f.customerid)                        AS "Total_Customers",
    COUNT(DISTINCT f.stockcode)                         AS "Total_Products",
    COUNT(DISTINCT f.country)                           AS "Countries_Served",

    -- Order KPIs
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.invoiceno), 0))::NUMERIC
    , 2)                                                AS "Avg_Order_Value_GBP",

    -- Customer KPIs
    ROUND(
        (SUM(f.totalamount) /
        NULLIF(COUNT(DISTINCT f.customerid), 0))::NUMERIC
    , 2)                                                AS "Avg_Revenue_per_Customer_GBP",

    -- Date range
    MIN(f.invoicedate)                                  AS "Data_From",
    MAX(f.invoicedate)                                  AS "Data_To",

    -- Segment highlights
    (SELECT COUNT(*) FROM dim_customers
     WHERE segment = 'Champions')                       AS "Champion_Customers",

    (SELECT ROUND((SUM(monetary) * 100.0 /
            (SELECT SUM(monetary) FROM dim_customers))::NUMERIC, 1)
     FROM dim_customers
     WHERE segment = 'Champions')                       AS "Champions_Revenue_%",

    (SELECT COUNT(*) FROM dim_customers
     WHERE segment IN ('At Risk', 'Cannot Lose Them'))  AS "At_Risk_Customers"

FROM fact_transactions f;