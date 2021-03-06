#%%Load required packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%%Get file path location
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_path = os.path.join(__location__, 'online_retail_II.xlsx')
df_store = os.path.join(__location__, 'df_store')

#%%read in sales data https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
sales_09 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
sales_10 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
sales = sales_09.append(sales_10).rename(columns={"Customer ID": "CID"})

#%%Check sales data loaded properly (should be 1M+ rows)
sales.shape

#%%view the first 5 rows
sales.head(5)

#%%Customer ID should be stored as a string
sales.CID = sales.CID.astype('object')

# %% Remove Cancelled Orders
#boolean array indicating which rows have a non integral number (i.e. starts with a letter)
invoice_non_numeric = pd.to_numeric(sales['Invoice'], errors='coerce').isnull()
#Some of these invoice_non_numeric records were for dept
adjust_debt = sales['Invoice'].str.startswith('A', na = False)
#Drop non_numeric_invoice records from sales data
sales = sales.loc[~invoice_non_numeric]

# %% Remove Non-5-digit integral number stock codes
#Description of dataset inlies any other type of code in this field
# is not associated with an actual stocked item and ofers no further explanation
sc_numeric = pd.to_numeric(sales['StockCode'], errors='coerce').notnull()
sales = sales.loc[sc_numeric]

#Summarize Values in Columns
#%%How many orders are there total?
sales.Invoice.nunique()
#%%How many items (StockCode) are there total?
sales.StockCode.nunique()
sales.Description.nunique() #more descriptions than Stockcodes
#%%Number of different customers?
sales.CID.nunique()
#%%Number of different countries?
sales.Country.nunique()

#%%Missing Values
sales.isna().mean().round(4) * 100 #22% of records mising a CID
missing_cid = sales.loc[sales.CID.isna()] #Just under 1/4 million
missing_cid.Country.value_counts() / len(missing_cid) #98% are UK
#All invoices either have a CID or do not
sales['HasCID'] = sales.CID.notna()
sales[['Invoice', 'HasCID']].drop_duplicates().groupby('Invoice').count().sort_values('HasCID')
#%%Remove missing values, not sure what they represent
sales = sales.dropna()
sales.isna().mean().round(4) * 100 #no more missing values

#Plot Feature Distibutions
#%%Quantity of Each Item Sold
#Get the total number of each item sold (StockCode)
qnty_sold = sales.groupby(['StockCode', 'Description']).sum().nlargest(200, 'Quantity').reset_index()
#There are a handful of objects sold > 50k, then it tails off
qnty_sold.plot(kind='bar', x='StockCode', y='Quantity')
qnty_sold.loc[qnty_sold.Quantity >= 50000].plot(kind='bar', x='Description', y='Quantity')

#%%Monthly Transactions
invoices = sales[['Invoice', 'InvoiceDate']].drop_duplicates()

monthly_transactions = pd.to_datetime(invoices['InvoiceDate']).dt.to_period('M').dt.to_timestamp().value_counts().reset_index().rename(columns={'index':'month', 'InvoiceDate':'transactions'})

monthly_transactions.set_index('month',inplace=True)

fig, ax = plt.subplots(figsize=(15,7))
monthly_transactions.plot(ax=ax)
ax.xaxis.set_major_locator(mdates.MonthLocator())

#Plot shows November is biggest month for sales
#Reasons: Black Friday, Holiday Shopping

# %% Price of items
#Get each items price
sales[['StockCode', 'Description', 'Price']].drop_duplicates().sort_values('Price')

numeric = pd.to_numeric(sales['StockCode'], errors='coerce').notnull()
non_numeric = pd.to_numeric(sales['StockCode'], errors='coerce').isnull()

sales.loc[non_numeric].sort_values('StockCode')

sales.loc[non_numeric].StockCode.unique()



# %% Price vs Quantity ordered on average

# %% What customers are ordering the most

# %% What countris are ordering the most


