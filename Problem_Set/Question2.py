import pandas as pd
import statsmodels.api as sm

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
msft_returns = msft_returns.dropna()
pg_returns = pg_returns.dropna()
gspc_returns = gspc_returns.dropna()

print(msft_returns.head())
print(pg_returns.head())
print(gspc_returns.head())

combined_returns = pd.DataFrame({
    'MSFT': msft_returns,
    'PG': pg_returns,
    'GSPC': gspc_returns
})

print(combined_returns.head())

#Calculate Mean of Returns
mean_returns = combined_returns.mean()

#Standard Deviation
standard_deviation_returns = combined_returns.std()

#Create dataframe for output of data
summary_table = pd.DataFrame({
  "Mean Return" : mean_returns,
  "Standard Deviation" : standard_deviation_returns
})

#Print data
print(summary_table)

#Estimate Full Sample CAPM betas (with standard errors)

#1. Add constant to market (GSPC)
market_returns_with_const = sm.add_constant(gspc_returns)

#2. Individual Stock regressions
msft_model = sm.OLS(msft_returns, market_returns_with_const).fit()
pg_model = sm.OLS(pg_returns, market_returns_with_const).fit()

#Extract CAPM BETAS and Standard Errors
msft_beta = msft_model.params['Adj Close']
msft_se = msft_model.bse['Adj Close']
print(f"Microsoft CAPM Values:\n Beta: {msft_beta}\n Standard Error: {msft_se}")

pg_beta = pg_model.params['Adj Close']
pg_se = pg_model.bse['Adj Close']
print(f"PG CAPM Values:\n Beta: {pg_beta}\n Standard Error: {pg_se}")