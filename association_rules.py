# General Packages
import pandas as pd
import numpy as np
import os

# MBA Packages
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml

# %% Get file path location
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_path = os.path.join(__location__, 'online_retail_II.xlsx')
df_store = os.path.join(__location__, 'df_store')

# %% read in sales data https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
sales_09 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
sales_10 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
sales = sales_09.append(sales_10).rename(columns={"Customer ID": "CID"})

# %% Customer ID should be stored as a string
sales.CID = sales.CID.astype('object')

# %% Drop NA
sales = sales.dropna()

# %% Remove stockcodes that seem to be place holders
items = sales.StockCode.unique().tolist()
drop_items = [x for x in items if not any(c.isdigit() for c in str(x))]
sales = sales[~sales.StockCode.isin(drop_items)]

# %% Remove Very Infrequent Items (running locally and need to preserve space)
item_trans = sales.StockCode.value_counts()
item_trans_50 = item_trans[item_trans > 50].index.tolist()
sales = sales[sales.StockCode.isin(item_trans_50)]

# %% Market Basket Analysis
# Pivot so each row is a transaction and each column an item
sales = sales.assign(tmp = sales.Quantity / sales.Quantity)
transactions = sales.pivot_table(index='Invoice', columns='StockCode', values='tmp', fill_value=0)

# %% Calculate MBA Statistics
frequent_itemsets = apriori(transactions, min_support=0.015, use_colnames=True)
frequent_itemsets.to_excel(os.path.join(df_store, 'frequent_itemsets.xlsx'), index=False)

# %% Get Association Rules
rules = association_rules(frequent_itemsets, metric="lift")
rules.to_excel(os.path.join(df_store, 'rules.xlsx'), index=False)

rules.sort_values('confidence', ascending = False, inplace = True)
rules.head(10)