# %% Load required packages
# General Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# MBA Packages
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml

# %% Get file path location
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_path = os.path.join(__location__, 'online_retail_II.xlsx')

# %% read in sales data https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
sales_09 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
sales_10 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
sales = sales_09.append(sales_10).rename(columns={"Customer ID": "CID"})

# %% Check sales data loaded properly (should be 1M+ rows)
sales.shape

# %% view the first 5 rows
sales.head(5)

# %% Customer ID should be stored as a string
sales.CID = sales.CID.astype('object')

# %% Summarize Values in Columns
#How many orders are there total?
sales.Invoice.nunique()
#How many items (StockCode) are there total?
sales.StockCode.nunique()
sales.Description.nunique() #more descriptions than Stockcodes
#Number of different customers?
sales.CID.nunique()
#Number of different countries?
sales.Country.nunique()

# %% Missing Values
sales.isna().mean().round(4) * 100 #22% of records mising a CID
missing_cid = sales.loc[sales.CID.isna()] #Just under 1/4 million
missing_cid.Country.value_counts() / len(missing_cid) #98% are UK
#All invoices either have a CID or do not
sales['HasCID'] = sales.CID.notna()
sales[['Invoice', 'HasCID']].drop_duplicates().groupby('Invoice').count().sort_values('HasCID')
#Remove missing values, not sure what they represent
sales = sales.dropna()
sales.isna().mean().round(4) * 100 #no more missing values

# %% Plot Feature Distibutions
#Get the total number of each item sold (StockCode)
qnty_sold = sales.groupby(['StockCode', 'Description']).sum().nlargest(200, 'Quantity').reset_index()
#There are a handful of objects sold > 50k, then it tails off
qnty_sold.plot(kind='bar', x='StockCode', y='Quantity')
qnty_sold.loc[qnty_sold.Quantity >= 50000].plot(kind='bar', x='Description', y='Quantity')

# %% Remove stockcodes that seem to be place holders
items = sales.StockCode.unique().tolist()
drop_items = [x for x in items if not any(c.isdigit() for c in str(x))]
sales = sales[~sales.StockCode.isin(drop_items)]

# %% Remove Very Infrequent Items (running locally and need to preserve space)
item_trans = sales.StockCode.value_counts()
item_trans = item_trans[item_trans > 50]

# %% Market Basket Analysis
# Pivot so each row is a transaction and each column an item
sales = sales.assign(tmp = sales.Quantity / sales.Quantity)
trans = sales.pivot_table(index='Invoice', columns='StockCode', values='tmp', fill_value=0)

# %% Calculate Item Support
frequent_itemsets = apriori(trans, min_support=0.01, use_colnames=True)

#Total Transactions
sales.Invoice.nunique()
#No of Transactions Item is in
sales.StockCode.value_counts()
#Support for each item (aka popularity)
sales.StockCode.value_counts() / sales.Invoice.nunique()

# %% Calculate Pair Confidence


