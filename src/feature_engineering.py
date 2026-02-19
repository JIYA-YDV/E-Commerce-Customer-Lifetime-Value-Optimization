#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def create_datetime_features(df):
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=Monday
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Quarter'] = df['InvoiceDate'].dt.quarter

    return df


# In[3]:


def create_price_features(df):
    df = df.copy()

    # Calculate actual revenue 
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Price categories
    df['PricePerUnit_Binned'] = pd.cut(
        df['UnitPrice'], 
        bins=[0, 1, 5, 10, 50, 1000], 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Premium']
    )

    return df


# In[4]:


def create_customer_features(df):
    df = df.copy()

    # Customer-level aggregations
    customer_stats = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',     
        'Quantity': 'sum',             
        'Revenue': 'sum',              
        'InvoiceDate': ['min', 'max']  
    }).reset_index()

    customer_stats.columns = [
        'CustomerID', 'TotalOrders', 'TotalItems', 'TotalSpend', 
        'FirstPurchase', 'LastPurchase'
    ]

    # Customer lifetime in days
    customer_stats['CustomerLifetimeDays'] = (
        customer_stats['LastPurchase'] - customer_stats['FirstPurchase']
    ).dt.days

    # Average metrics
    customer_stats['AvgOrderValue'] = (
        customer_stats['TotalSpend'] / customer_stats['TotalOrders']
    )
    customer_stats['AvgItemsPerOrder'] = (
        customer_stats['TotalItems'] / customer_stats['TotalOrders']
    )

    # Merge back
    df = df.merge(
        customer_stats[['CustomerID', 'TotalOrders', 'TotalItems', 
                       'TotalSpend', 'AvgOrderValue', 'AvgItemsPerOrder',
                       'CustomerLifetimeDays', 'FirstPurchase', 'LastPurchase']], 
        on='CustomerID', 
        how='left'
    )

    return df


# In[5]:


def create_rfm_features(df):
    df = df.copy()

    # Reference date (day after last transaction)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics per customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency  
        'Revenue': 'sum'                                           # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Create RFM scores (quintiles 1-5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

    # Combined RFM score
    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) + 
        rfm['F_Score'].astype(str) + 
        rfm['M_Score'].astype(str)
    )

    # Customer segmentation based on RFM
    def segment_customers(row):
        rfm_score = str(row['RFM_Score'])
        if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif rfm_score in ['155', '254', '245', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '125', '124']:
            return 'Cannot Lose Them'
        elif rfm_score in ['331', '321', '231', '241', '251']:
            return 'Price Sensitive'
        else:
            return 'Others'

    rfm['CustomerSegment'] = rfm.apply(segment_customers, axis=1)

    # Merge RFM features
    df = df.merge(
        rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 
             'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'CustomerSegment']], 
        on='CustomerID', 
        how='left'
    )

    return df


# In[6]:


def create_product_features(df):
    df = df.copy()

    product_stats = df.groupby('StockCode').agg({
        'Quantity': 'sum',
        'Revenue': 'sum',
        'CustomerID': 'nunique',
        'InvoiceNo': 'nunique',
        'UnitPrice': 'mean'
    }).reset_index()

    product_stats.columns = [
        'StockCode', 'ProductTotalQty', 'ProductTotalRevenue',
        'ProductUniqueCustomers', 'ProductTransactions', 'AvgProductPrice'
    ]

    # Product popularity score
    product_stats['ProductPopularity'] = (
        product_stats['ProductUniqueCustomers'] / product_stats['ProductUniqueCustomers'].max()
    )

    df = df.merge(product_stats, on='StockCode', how='left')

    return df


# In[7]:


def create_order_features(df):
    
    df = df.copy()

    # Order-level aggregations
    order_stats = df.groupby('InvoiceNo').agg({
        'Quantity': 'sum',
        'Revenue': 'sum',
        'StockCode': 'nunique',
        'CustomerID': 'first',
        'InvoiceDate': 'first'
    }).reset_index()

    order_stats.columns = [
        'InvoiceNo', 'OrderTotalQty', 'OrderTotalRevenue', 
        'BasketSize', 'CustomerID', 'OrderDate'
    ]

    # Days since previous order per customer
    order_stats = order_stats.sort_values(['CustomerID', 'OrderDate'])
    order_stats['DaysSinceLastOrder'] = order_stats.groupby('CustomerID')['OrderDate'].diff().dt.days

    # Merge back
    df = df.merge(
        order_stats[['InvoiceNo', 'OrderTotalQty', 'OrderTotalRevenue', 
                    'BasketSize', 'DaysSinceLastOrder']], 
        on='InvoiceNo', 
        how='left'
    )

    return df


# In[8]:


def create_geographic_features(df):
    df = df.copy()

    # Binary encoding for UK vs International
    df['IsUK'] = (df['Country'] == 'United Kingdom').astype(int)

    # Country frequency encoding (optional - reduces dimensionality)
    country_freq = df['Country'].value_counts().to_dict()
    df['CountryFreq'] = df['Country'].map(country_freq)

    return df


# In[9]:


def full_feature_engineering_pipeline(input_path, output_path=None):
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)

    # Load data
    print(f"\n1. Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Original shape: {df.shape}")

    # Run all feature engineering steps
    print("\n2. Creating datetime features...")
    df = create_datetime_features(df)

    print("3. Creating price features...")
    df = create_price_features(df)

    print("4. Creating customer features...")
    df = create_customer_features(df)

    print("5. Creating RFM features (this may take a moment)...")
    df = create_rfm_features(df)

    print("6. Creating product features...")
    df = create_product_features(df)

    print("7. Creating order features...")
    df = create_order_features(df)

    print("8. Creating geographic features...")
    df = create_geographic_features(df)

    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final shape: {df.shape}")
    print(f"Total new features: {df.shape[1] - 10}")  # Assuming 10 original columns
    print(f"\nCustomer segments:")
    print(df['CustomerSegment'].value_counts())

    # Save if path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")

    return df


# In[10]:


if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "cleaned_sales_data.csv"
    OUTPUT_FILE = "engineered_sales_data.csv"

    df_result = full_feature_engineering_pipeline(INPUT_FILE, OUTPUT_FILE)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Check output file: engineered_sales_data.csv")
    print("2. Use features for: clustering, churn prediction, recommendation")
    print("3. Key features for ML: RFM_Score, CustomerSegment, Recency, Frequency, Monetary")


# In[ ]:




