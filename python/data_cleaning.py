#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# STEP 1: LOAD DATA
df = pd.read_csv('data.csv', encoding='latin-1')
print(f"✅ Loaded: {df.shape[0]:,} rows")


# In[3]:


# STEP 2: CLEAN DATA
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')


# In[4]:


# Remove returns & outliers
sales = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
sales = sales[(sales['Quantity'] > 0) & (sales['UnitPrice'] > 0)]
sales = sales[sales['Quantity'] <= sales['Quantity'].quantile(0.99)]


# In[5]:


# STEP 3: SAVE
sales.to_csv('cleaned_sales_data.csv', index=False)
print(f"✅ Cleaned: {len(sales):,} rows saved")
print(f"✅ Revenue: £{sales['TotalAmount'].sum():,.2f}")


# In[ ]:




