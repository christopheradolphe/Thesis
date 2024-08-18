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

#Remove NaN values (eg. first row)
msft_returns = msft_returns.dropna()
pg_returns = pg_returns.dropna()
gspc_returns = gspc_returns.dropna()

#Combine the dataframes for further calculations
combined_returns = pd.DataFrame({
    'MSFT': msft_returns,
    'PG': pg_returns,
    'GSPC': gspc_returns
})

#Calculate Mean of Returns
mean_returns = combined_returns.mean()

#Standard Deviation
standard_deviation_returns = combined_returns.std()

#Create dataframe for output of data
summary_table = pd.DataFrame({
  "Mean Return" : mean_returns,
  "Standard Deviation" : standard_deviation_returns
})

#Print standard deviation and mean data
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
print(f"Microsoft CAPM Values:\n Beta: {msft_beta:.4f}\n Standard Error: {msft_se:.4f}")

pg_beta = pg_model.params['Adj Close']
pg_se = pg_model.bse['Adj Close']
print(f"PG CAPM Values:\n Beta: {pg_beta:.4f}\n Standard Error: {pg_se:.4f}")

#Part 2
# Portfolio starts at value of 1 and rebalances to 50% of total every year
weights = pd.Series([0.5, 0.5], index=['MSFT', 'PG'])


combined_returns.index = pd.to_datetime(combined_returns.index)

weights_df = pd.DataFrame(index=combined_returns.index, columns =['MSFT', 'PG'])
portfolio_value_df = pd.DataFrame(index=combined_returns.index, columns=['Portfolio Value'])

for year in combined_returns.index.year.unique():
  yearly_returns = combined_returns[combined_returns.index.year == year]

  for date in yearly_returns.index:
    #Find daily return
    # Update weights based on the returns
    weights = (weights * (1 + yearly_returns.loc[date]))
    portfolio_value = weights.sum()

    weights_df.loc[date] = weights
    portfolio_value_df.loc[date] = portfolio_value

  rebalanced_value = weights.sum() / 2
  weights['MSFT'] = rebalanced_value
  weights['PG'] = rebalanced_value

#Final Portfolio Value
final_value = portfolio_value_df.iloc[-1]['Portfolio Value']
print(f'Final Portfolio Value: {final_value * 100:.2f}% of Principal Investment')

#Calculate Daily Portfolio Returns
portfolio_returns = portfolio_value_df['Portfolio Value'].pct_change().dropna()

print(portfolio_returns.head())

#Mean Return
mean_return = portfolio_returns.mean()
print(f'Mean Portfolio Return: {mean_return:.6f}')

#Standard Deviation
std_return = portfolio_returns.std()
print(f'Standard Dev Portfolio Return: {std_return:.4f}')

#Find CAPM model values
market_returns_with_const.index = pd.to_datetime(market_returns_with_const.index)
rebalanced_portfolio_model = sm.OLS(portfolio_returns, market_returns_with_const.loc[portfolio_returns.index]).fit()
portfolio_beta = rebalanced_portfolio_model.params['Adj Close']
portfolio_se = rebalanced_portfolio_model.bse['Adj Close']
print(f"Rebalanced Portfolio CAPM Values:\n Beta: {portfolio_beta:.4f}\n Standard Error: {portfolio_se:.4f}")