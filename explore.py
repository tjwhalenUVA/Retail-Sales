#Load required packages
import pandas as pd
import numpy as np
import os

#Get file path location
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

file_path = os.path.join(__location__, 'online_retail_II.xlsx')

# read in sales data from 
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
sales_09 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
sales_10 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
sales = sales_09.append(sales_10).rename(columns={"Customer ID": "CID"})

# Check sales data loaded properly (should be 1M+ rows)
sales.shape
sales.head(5)
# Customer ID should be stored as a string
sales.CID = sales.CID.astype('object')

################ Summarize Values in Columns ################
#How many orders are there total?
sales.Invoice.nunique()
#How many items (StockCode) are there total?
sales.StockCode.nunique()
sales.Description.nunique() #more descriptions than Stockcodes
#Number of different customers?
sales.CID.nunique()
#Number of different countries?
sales.Country.nunique()

################ Missing Values ################
sales.isna().mean().round(4) * 100 #22% of records mising a CID
missing_cid = sales.loc[sales.CID.isna()] #Just under 1/4 million
missing_cid.Country.value_counts() / len(missing_cid) #98% are UK
#All invoices either have a CID or do not
sales['HasCID'] = sales.CID.notna()
sales[['Invoice', 'HasCID']].drop_duplicates().groupby('Invoice').count().sort_values('HasCID')
#Remove missing values, not sure what they represent
sales = sales.dropna()
sales.isna().mean().round(4) * 100 #no more missing values

################ Plot Feature Distibutions ################
sales.head()