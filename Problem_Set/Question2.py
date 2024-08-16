import pandas as pd

symbols = ['MSFT', 'PG', 'GSPC']
start_date = '2010-01-01'

#Read in data from csv to pandas dataframes
msft_data = pd.read_csv("MSFT.csv", index_col='Date')
pg_data = pd.read_csv("PG.csv", index_col='Date')
gspc_data = pd.read_csv("GSPC.csv", index_col='Date')

#Calculate Daily Returns
msft_returns = msft_data['Adj Close'].pct_change()
pg_returns = pg_data['Adj Close'].pct_change()
gspc_returns = gspc_data['Adj Close'].pct_change()

print(msft_returns.head())
print(pg_returns.head())
print(gspc_returns.head())

#Remove NaN values (eg. first row)
msft_returns.dropna()
pg_returns.dropna()
gspc_returns.dropna()

print(msft_returns.head())
print(pg_returns.head())
print(gspc_returns.head())