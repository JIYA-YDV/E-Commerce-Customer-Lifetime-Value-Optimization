SELECT
    'fact_transactions' AS tbl, COUNT(*) AS rows FROM fact_transactions
UNION ALL SELECT 'dim_customers', COUNT(*) FROM dim_customers
UNION ALL SELECT 'dim_products',  COUNT(*) FROM dim_products
UNION ALL SELECT 'dim_date',      COUNT(*) FROM dim_date
UNION ALL SELECT 'dim_geography', COUNT(*) FROM dim_geography;



-- =============================================================
-- QUERY 1: Full RFM Segment Overview
-- =============================================================
-- WHY: The first thing any stakeholder wants to see —
-- how many customers are in each segment and how much
-- revenue each segment generates. This is the executive
-- summary of your entire RFM analysis in one table.
-- =============================================================
SELECT
    c.segment                                               AS "Segment",
    COUNT(DISTINCT c.customerid)                            AS "Customers",
    ROUND((COUNT(DISTINCT c.customerid) * 100.0 /
          SUM(COUNT(DISTINCT c.customerid)) OVER())::NUMERIC, 1)
                                                            AS "Customer_%",
    ROUND(SUM(f.totalamount)::NUMERIC, 0)                   AS "Total_Revenue_GBP",
    ROUND((SUM(f.totalamount) * 100.0 /
          SUM(SUM(f.totalamount)) OVER())::NUMERIC, 1)
                                                            AS "Revenue_%",
    ROUND(AVG(c.monetary)::NUMERIC, 0)                      AS "Avg_Lifetime_Spend",
    ROUND(AVG(c.recency)::NUMERIC, 0)                       AS "Avg_Days_Since_Purchase",
    ROUND(AVG(c.frequency)::NUMERIC, 1)                     AS "Avg_Orders",
    s.marketingaction                                       AS "Recommended_Action"
FROM dim_customers c
JOIN fact_transactions f  ON c.customerid = f.customerid
JOIN dim_rfm_segments s   ON c.segment    = s.segment
GROUP BY c.segment, s.marketingaction, s.priority
ORDER BY s.priority;



-- =============================================================
-- QUERY 2: Top 20 Most Valuable Customers
-- =============================================================
-- WHY: Identifies the exact customers to prioritise for
-- VIP treatment, personal outreach, and loyalty rewards.
-- These customers drive disproportionate revenue.
-- Use this list to build a VIP customer programme.
-- =============================================================

SELECT
    c.customerid                                AS "CustomerID",
    c.segment                                   AS "Segment",
    ROUND(c.monetary::NUMERIC, 2)               AS "Lifetime_Spend_GBP",
    c.frequency                                 AS "Total_Orders",
    ROUND(c.avgordervalue::NUMERIC, 2)          AS "Avg_Order_Value_GBP",
    c.recency                                   AS "Days_Since_Last_Purchase",
    c.r_score                                   AS "R_Score",
    c.f_score                                   AS "F_Score",
    c.m_score                                   AS "M_Score",
    c.rfm_total                                 AS "RFM_Total",
    TO_CHAR(c.lastpurchasedate, 'YYYY-MM-DD')   AS "Last_Purchase_Date",
    CASE
        WHEN c.recency <= 30  THEN 'Active'
        WHEN c.recency <= 90  THEN 'Cooling'
        WHEN c.recency <= 180 THEN 'At Risk'
        ELSE 'Lapsed'
    END                                         AS "Activity_Status"
FROM dim_customers c
ORDER BY c.monetary DESC
LIMIT 20;


-- =============================================================
-- QUERY 3: Champions Deep Dive
-- =============================================================
-- WHY: Champions are your best customers — recent, frequent,
-- high spend. Understanding them reveals what your ideal
-- customer looks like so you can acquire more like them.
-- Compare their behaviour vs other segments.
-- =============================================================

SELECT
    c.customerid,
    ROUND(c.monetary::NUMERIC, 2)           AS lifetime_spend,
    c.frequency                             AS total_orders,
    ROUND(c.avgordervalue::NUMERIC, 2)      AS avg_order_value,
    c.recency                               AS days_since_purchase,
    COUNT(DISTINCT f.stockcode)             AS unique_products_bought,
    COUNT(DISTINCT f.yearmonth)             AS active_months,
    MIN(f.invoicedate)                      AS first_purchase,
    MAX(f.invoicedate)                      AS last_purchase
FROM dim_customers c
JOIN fact_transactions f ON c.customerid = f.customerid
WHERE c.segment = 'Champions'
GROUP BY
    c.customerid, c.monetary, c.frequency,
    c.avgordervalue, c.recency
ORDER BY c.monetary DESC;


-- =============================================================
-- QUERY 4: At Risk and Cannot Lose Them — Win-Back List
-- =============================================================
-- WHY: These customers used to be valuable but have gone
-- quiet. Every day without action increases churn probability.
-- This query produces the exact list for your win-back
-- email campaign — sorted by value lost (highest first).
-- =============================================================

SELECT
    c.customerid                                AS "CustomerID",
    c.segment                                   AS "Segment",
    ROUND(c.monetary::NUMERIC, 2)               AS "Lifetime_Spend_GBP",
    c.frequency                                 AS "Historical_Orders",
    c.recency                                   AS "Days_Silent",
    ROUND(c.avgordervalue::NUMERIC, 2)          AS "Avg_Order_Value_GBP",
    c.r_score                                   AS "R_Score",
    c.m_score                                   AS "M_Score",
    TO_CHAR(c.lastpurchasedate, 'YYYY-MM-DD')   AS "Last_Seen",
    -- Estimated revenue at risk (avg order value * expected orders per year)
    ROUND((c.avgordervalue * (c.frequency / 1.0))::NUMERIC, 0)
                                                AS "Est_Annual_Revenue_At_Risk",
    s.marketingaction                           AS "Action"
FROM dim_customers c
JOIN dim_rfm_segments s ON c.segment = s.segment
WHERE c.segment IN ('At Risk', 'Cannot Lose Them')
ORDER BY c.monetary DESC;


-- =============================================================
-- QUERY 5: Hibernating Customers — Reactivation Candidates
-- =============================================================
-- WHY: 1,057 customers (24.6% of base) are hibernating.
-- Even reactivating 10% of these would meaningfully
-- increase revenue. This query ranks them by historical
-- value to prioritise who to target first.
-- =============================================================

SELECT
    c.customerid,
    ROUND(c.monetary::NUMERIC, 2)           AS lifetime_spend,
    c.frequency                             AS past_orders,
    c.recency                               AS days_inactive,
    ROUND(c.avgordervalue::NUMERIC, 2)      AS avg_order_value,
    -- Reactivation value tier
    CASE
        WHEN c.monetary >= 1000 THEN 'High Value — Priority Reactivation'
        WHEN c.monetary >= 500  THEN 'Medium Value — Standard Campaign'
        ELSE                         'Low Value — Bulk Email Only'
    END                                     AS reactivation_tier,
    TO_CHAR(c.lastpurchasedate, 'DD Mon YYYY') AS last_purchase
FROM dim_customers c
WHERE c.segment = 'Hibernating'
ORDER BY c.monetary DESC;


-- =============================================================
-- QUERY 6: New and Promising Customers — Nurture Pipeline
-- =============================================================
-- WHY: These customers made their first purchase recently.
-- The next 30-60 days are critical — customers who buy
-- a second time are far more likely to become loyal.
-- This list drives your onboarding and follow-up campaigns.
-- =============================================================

SELECT
    c.customerid,
    c.segment,
    c.recency                               AS days_since_first_purchase,
    c.frequency                             AS orders_so_far,
    ROUND(c.monetary::NUMERIC, 2)           AS spend_so_far,
    ROUND(c.avgordervalue::NUMERIC, 2)      AS avg_order_value,
    TO_CHAR(c.lastpurchasedate, 'YYYY-MM-DD') AS last_purchase,
    -- Urgency score — newer customers need faster follow-up
    CASE
        WHEN c.recency <= 14 THEN 'Urgent — follow up this week'
        WHEN c.recency <= 30 THEN 'Soon — follow up this month'
        ELSE                      'Standard — include in next campaign'
    END                                     AS follow_up_urgency
FROM dim_customers c
WHERE c.segment IN ('New Customers', 'Promising', 'Potential Loyalists')
ORDER BY c.recency ASC;


-- =============================================================
-- QUERY 7: Month-over-Month Retention Rate
-- =============================================================
-- WHY: This directly answers your research question:
-- "What is our retention rate month-over-month?"
-- Calculates the % of customers who purchased in month N
-- who also purchased in month N+1.
-- A healthy retail business retains 20-40% month-over-month.
-- =============================================================

WITH monthly_customers AS (
    -- Get distinct customers per month
    SELECT
        yearmonth,
        customerid
    FROM fact_transactions
    GROUP BY yearmonth, customerid
),
retention AS (
    -- Self-join: find customers who appear in consecutive months
    SELECT
        curr.yearmonth                          AS current_month,
        COUNT(DISTINCT curr.customerid)         AS current_customers,
        COUNT(DISTINCT prev.customerid)         AS retained_customers
    FROM monthly_customers curr
    LEFT JOIN monthly_customers prev
        ON curr.customerid = prev.customerid
        AND TO_DATE(curr.yearmonth, 'YYYY-MM') =
            TO_DATE(prev.yearmonth, 'YYYY-MM') + INTERVAL '1 month'
    GROUP BY curr.yearmonth
)
SELECT
    current_month                               AS "Month",
    current_customers                           AS "Active_Customers",
    retained_customers                          AS "Retained_from_Prev_Month",
    CASE
        WHEN LAG(current_customers) OVER (ORDER BY current_month) > 0
        THEN ROUND(
            retained_customers * 100.0 /
            LAG(current_customers) OVER (ORDER BY current_month), 1
        )
        ELSE NULL
    END                                         AS "Retention_Rate_%",
    current_customers -
        LAG(current_customers) OVER (ORDER BY current_month)
                                                AS "Customer_Growth"
FROM retention
ORDER BY current_month;


-- =============================================================
-- QUERY 8: RFM Score Distribution
-- =============================================================
-- WHY: Shows how customers are distributed across the
-- 1-5 scoring grid. Validates that your quintile scoring
-- is producing a balanced spread (each score should have
-- roughly 20% of customers). Useful for portfolio presentation.
-- =============================================================

SELECT
    r_score                                     AS "R_Score",
    f_score                                     AS "F_Score",
    m_score                                     AS "M_Score",
    COUNT(*)                                    AS "Customers",
    ROUND(AVG(monetary)::NUMERIC, 0)            AS "Avg_Spend_GBP",
    ROUND(AVG(recency)::NUMERIC, 0)             AS "Avg_Recency_Days",
    ROUND(AVG(frequency)::NUMERIC, 1)           AS "Avg_Orders"
FROM dim_customers
GROUP BY r_score, f_score, m_score
ORDER BY r_score DESC, f_score DESC, m_score DESC;


-- =============================================================
-- QUERY 9: Customer Lifetime Value Tiers
-- =============================================================
-- WHY: Groups customers into CLV bands (Bronze/Silver/Gold/
-- Platinum) for simple stakeholder communication.
-- Non-technical audiences understand "Platinum customers"
-- better than "RFM score 555". Use this in your
-- business recommendations report.
-- =============================================================

SELECT
    CASE
        WHEN monetary >= 5000 THEN '💎 Platinum (£5,000+)'
        WHEN monetary >= 2000 THEN '🥇 Gold (£2,000–£4,999)'
        WHEN monetary >= 1000 THEN '🥈 Silver (£1,000–£1,999)'
        WHEN monetary >= 500  THEN '🥉 Bronze (£500–£999)'
        ELSE                       '⚪ Standard (< £500)'
    END                                         AS "CLV_Tier",
    COUNT(*)                                    AS "Customers",
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1)
                                                AS "% of Base",
    ROUND(SUM(monetary)::NUMERIC, 0)            AS "Total_Revenue_GBP",
    ROUND(SUM(monetary) * 100.0 /
          SUM(SUM(monetary)) OVER(), 1)         AS "% of Revenue",
    ROUND(AVG(monetary)::NUMERIC, 0)            AS "Avg_Spend_GBP",
    ROUND(AVG(frequency)::NUMERIC, 1)           AS "Avg_Orders"
FROM dim_customers
GROUP BY
    CASE
        WHEN monetary >= 5000 THEN '💎 Platinum (£5,000+)'
        WHEN monetary >= 2000 THEN '🥇 Gold (£2,000–£4,999)'
        WHEN monetary >= 1000 THEN '🥈 Silver (£1,000–£1,999)'
        WHEN monetary >= 500  THEN '🥉 Bronze (£500–£999)'
        ELSE                       '⚪ Standard (< £500)'
    END
ORDER BY MIN(monetary) DESC;


-- =============================================================
-- QUERY 10: Segment Migration Opportunity Analysis
-- =============================================================
-- WHY: Identifies customers who are one step away from
-- upgrading to a better segment. E.g. a Potential Loyalist
-- with high monetary score just needs one more order to
-- become Loyal. These are the easiest wins for marketing.
-- =============================================================

SELECT
    c.segment                               AS "Current_Segment",
    CASE
        WHEN c.segment = 'Potential Loyalists' THEN 'One more order → Loyal Customer'
        WHEN c.segment = 'Promising'           THEN 'Two more orders → Potential Loyalist'
        WHEN c.segment = 'At Risk'             THEN 'Recent purchase → Loyal Customer'
        WHEN c.segment = 'Hibernating'         THEN 'Any purchase → Promising/New Customer'
        WHEN c.segment = 'New Customers'       THEN 'Second purchase → Potential Loyalist'
        ELSE 'Already in top segment'
    END                                     AS "Path_to_Upgrade",
    COUNT(*)                                AS "Customers",
    ROUND(AVG(c.monetary)::NUMERIC, 0)      AS "Avg_Current_Spend",
    ROUND(AVG(c.avgordervalue)::NUMERIC, 0) AS "Avg_Order_Value",
    -- Expected revenue if they upgrade
    ROUND(AVG(c.avgordervalue) * 2::NUMERIC, 0)
                                            AS "Est_Revenue_If_They_Return"
FROM dim_customers c
WHERE c.segment NOT IN ('Champions', 'Loyal Customers')
GROUP BY c.segment,
    CASE
        WHEN c.segment = 'Potential Loyalists' THEN 'One more order → Loyal Customer'
        WHEN c.segment = 'Promising'           THEN 'Two more orders → Potential Loyalist'
        WHEN c.segment = 'At Risk'             THEN 'Recent purchase → Loyal Customer'
        WHEN c.segment = 'Hibernating'         THEN 'Any purchase → Promising/New Customer'
        WHEN c.segment = 'New Customers'       THEN 'Second purchase → Potential Loyalist'
        ELSE 'Already in top segment'
    END
ORDER BY AVG(c.monetary) DESC;















