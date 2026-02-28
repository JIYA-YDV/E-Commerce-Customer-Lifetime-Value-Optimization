🛒 E-Commerce Customer Lifetime Value (CLV) Optimization

End-to-end data analytics project analysing 393,994 transactions from a UK-based e-commerce retailer to identify high-value customers, optimise retention strategy, and uncover revenue growth opportunities across 37 countries.

📊 Dashboard Preview

Page 1 — Executive Summary

<img width="1863" height="1047" alt="Executive Summary" src="https://github.com/user-attachments/assets/3e3db42d-7bcb-4552-bbb7-73ca9659e1f4" />

Page 2 — Customer Segmentation

<img width="737" height="844" alt="Customer Segmentation " src="https://github.com/user-attachments/assets/b49c6be2-5e58-4813-a2f0-4ccb944fda16" />

Page 3 — Sales Trends

<img width="777" height="820" alt="Sales Trends " src="https://github.com/user-attachments/assets/7523d80c-7a91-4743-8051-e5c7a4873652" />

Page 4 — Geographic & Product Performance

<img width="888" height="832" alt="Geographic   Product Performance" src="https://github.com/user-attachments/assets/cbde2349-894e-45be-9342-fd4a13b7c9d4" />

🎯 Business Problem


A UK-based online retailer with 4,303 customers and £7.22M annual revenue needed to answer five critical business questions:


Who are our most valuable customers and how do we segment them for targeted marketing?

Which customers are at risk of churning and what is our month-over-month retention rate?

Which products are bought together and what cross-sell opportunities exist?

What are our peak sales periods and how do monthly and quarterly revenues trend?

Which international markets drive growth and where should we expand marketing spend?

🔑 Key Findings

Champions drive disproportionate revenue
1,133 customers (26.3% of base) generate 62% of total revenue — a classic Pareto distribution confirmed.

Hibernating customers represent a major opportunity
1,057 customers (24.6% of base) are inactive. Even reactivating 10% would add approximately £33K annual revenue.

November is peak month by a wide margin
Revenue of £1.02M in November 2011 — 85% above the monthly average of £555K.

Thursday is the strongest trading day
Thursday generates £1.57M vs Sunday £0.75M — a 2× difference driven by B2B wholesale buying behaviour.

Weekday revenue dominates completely
£6.5M weekday revenue vs £0.7M weekend confirms this is a B2B retailer, not a consumer business.

UK accounts for 84% of revenue across 37 countries
Germany, France and Netherlands are the highest-value international markets for expansion.

Top 3 products generate 18% of total revenue
Colouring Pencils Brown Tube, Bathroom Metal Sign and Paper Lantern are hero products — never allow out of stock.

Champions spend 14× more than Lost customers
Average Champion spend £4.2K vs Lost customer £0.3K — the CLV gap justifies aggressive retention investment.

🔄 Project Pipeline

Step 01 — Data Cleaning 01_data_cleaning.py

Input: online_retail.xlsx (541,909 rows)
Output: cleaned_sales_data.csv (393,994 rows)
Removes null customer IDs, cancelled orders, negative quantities, price outliers

Step 02 — Feature Engineering 02_feature_engineering.py

Input: cleaned_sales_data.csv
Output: engineered_sales_data.csv
Creates time features, customer-level aggregations, product-level metrics

Step 03 — RFM Analysis 03_rfm_analysis.py

Input: engineered_sales_data.csv
Output: rfm_scores.csv
Scores each customer on Recency, Frequency, Monetary using quintile scoring
Assigns 6 business segments with marketing actions

Step 04 — Cohort Analysis 04_cohort_analysis.py

Input: cleaned_sales_data.csv
Output: cohort_matrix.csv
Builds monthly cohort retention matrix
Reveals 22% average month-over-month retention rate

Step 05 — Market Basket Analysis 05_market_basket.py

Input: cleaned_sales_data.csv
Output: association_rules.csv
Uses FP-Growth algorithm to find products frequently bought together
Identifies cross-sell opportunities with confidence and lift scores

Step 06 — PostgreSQL ETL 06_load_to_postgresql.py

Input: All processed CSV files
Output: 6-table star schema in PostgreSQL
Loads dimension and fact tables with chunked processing and error handling

📈 RFM Segmentation Results

🏆 Champions — 1,133 customers (26.3%)
Average spend £6,952. Recent, frequent, high-value buyers.
Action: VIP loyalty programme, early product access, personal outreach.

💙 Loyal Customers — 1,247 customers (29.0%)
Average spend £2,841. Regular buyers with strong purchase history.
Action: Loyalty rewards, referral programme, upsell to Champion tier.

🌱 Potential Loyalists — 664 customers (15.4%)
Average spend £1,073. Recent buyers with growing order frequency.
Action: Nurture with targeted offers and second-purchase incentives.

😴 Hibernating — 1,057 customers (24.6%)
Average spend £317. Have not purchased in a long time.
Action: Win-back email campaign tiered by historical lifetime value.

⚠️ At Risk — 110 customers (2.6%)
Average spend £2,590. Were previously valuable, now going quiet.
Action: Urgent personal outreach — highest priority for win-back.

❌ Lost — 92 customers (2.1%)
Average spend £285. No purchase in over a year.
Action: Final re-engagement offer then remove from active marketing list.

🛠️ Tech Stack


Python 3.9+ — Core language for all analysis scripts

Pandas and NumPy — Data manipulation, cleaning, transformation

Scikit-learn — Quintile scoring for RFM segmentation

MLxtend — FP-Growth algorithm for market basket analysis

Matplotlib and Seaborn — Analysis charts and retention heatmaps

PostgreSQL 15 — Relational data warehouse with star schema design

SQLAlchemy and Psycopg2 — Python to PostgreSQL ETL pipeline

Power BI Desktop — 4-page interactive business intelligence dashboard

DAX — Calculated measures, KPIs, time intelligence, and dynamic filters

🚀 How to Run

1. Clone the repository
bashgit clone https://github.com/JIYA-YDV/E-Commerce-Customer-Lifetime-Value-Optimization.git
cd E-Commerce-Customer-Lifetime-Value-Optimization

3. Install Python dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn mlxtend sqlalchemy psycopg2-binary openpyxl

5. Download the dataset
Download from: https://archive.ics.uci.edu/ml/datasets/online+retail
Save to: data/raw/online_retail.xlsx

7. Run the Python pipeline in order
bashpython python/01_data_cleaning.py
python python/02_feature_engineering.py
python python/03_rfm_analysis.py
python python/04_cohort_analysis.py
python python/05_market_basket.py

9. Create the PostgreSQL database
Open pgAdmin4 and create a database named E-Commerce-Customer-Lifetime-Value-Optimization, then run:
bashpsql -U postgres -d ecommerce_clv_optimization -f sql/schema_creation_postgresql.sql

11. Set environment variables
bashexport DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=ecommerce_clv_optimization
export DB_USER=postgres
export DB_PASSWORD=yourpassword

13. Load data to PostgreSQL
bashpython python/06_load_to_postgresql.py

15. Open the Power BI dashboard
Open powerbi/retail_dashboard.pbix in Power BI Desktop, update the data source to point to your PostgreSQL instance, and refresh all tables.

📋 SQL Queries

customer_queries.sql

Q1 Full RFM Segment Overview — revenue and customer count per segment with marketing actions

Q2 Top 20 Most Valuable Customers — ranked by lifetime spend with activity status

Q3 Champions Deep Dive — orders, unique products, active months, first and last purchase

Q4 At Risk Win-Back List — 110 customers sorted by estimated annual revenue at risk

Q5 Hibernating Reactivation Candidates — 1,057 customers tiered by historical value

Q6 New Customer Nurture Pipeline — sorted by follow-up urgency score

Q7 Month-over-Month Retention Rate — rolling retention calculation using window functions

Q8 RFM Score Distribution — validates that quintile scoring produces balanced spread

Q9 Customer Lifetime Value Tiers — Bronze, Silver, Gold, Platinum classification

Q10 Segment Migration Opportunity — customers one order away from upgrading segment


sales_analytics.sql


Q1 Monthly Revenue Trend — MoM growth percentage and cumulative running total

Q2 Quarterly Revenue Summary — QoQ growth and quarter-over-quarter comparison

Q3 Day of Week Sales Pattern — identifies Thursday as peak, Sunday as weakest

Q4 Monthly Seasonality Index — 100 equals average month, November index 180+

Q5 Country Performance Ranking — all 37 markets with revenue, AOV, and classification

Q6 Top 20 Products by Revenue — hero products with popularity score

Q7 Average Order Value Trend — 3-month rolling average showing upward trajectory

Q8 UK vs International Revenue Split — confirms 84% UK concentration

Q9 New vs Returning Customer Revenue — 91% revenue from returning customers

Q10 Executive Summary KPIs — all headline numbers in a single query for reporting


💡 Business Recommendations

1. Launch a Champions VIP Programme
1,133 customers generate 62% of revenue. Treat them with exclusive benefits — early product access, free shipping, dedicated account contact. Cost of retention is far lower than cost of replacement for these customers.

3. Immediate At Risk Win-Back Campaign
110 customers averaging £2,590 lifetime spend have gone quiet. Estimated £285K annual revenue is at risk. Contact highest-spend customers personally within 30 days before they move to Lost status permanently.
   
5. Hibernating Customer Reactivation in Three Tiers
Segment 1,057 dormant customers by historical value. High value (£1,000+) receives personalised email with targeted discount. Medium value (£500–£999) receives standard campaign. Low value receives bulk re-engagement only. This focuses resource where recovery is most valuable.

7. Invest Heavily in October and November
Revenue peaks 85% above average in November. Begin inventory build-up in September. Double marketing spend in October to capture the full seasonal uplift window before competitors do.

9. International Expansion — Three Priority Markets
Germany, France and Netherlands consistently show high average order values among international customers. Targeted paid campaigns in these three countries offer the strongest international ROI based on existing order data.

📦 Dataset


UCI Online Retail Dataset

Source: https://archive.ics.uci.edu/ml/datasets/online+retail
Period: December 2010 to December 2011
Raw transactions: 541,909
Clean transactions after processing: 393,994
Unique customers: 4,303
Unique product SKUs: 3,684
Countries: 37

This is a real transactional dataset from a UK-based online retailer specialising in unique all-occasion gifts. Many customers are wholesale buyers purchasing for resale.
