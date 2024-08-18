import pandas as pd
import statsmodels.api as sm

# 1a) Calculate a series of returns (1+Return(t) = AdjClose(t) / AdjClose(t-1)). 
print("1a) Calculate a series of returns (1+Return(t) = AdjClose(t) / AdjClose(t-1)). \n")

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

# 1b) What are the mean and standard deviation of returns for each symbol?
print("1b) What are the mean and standard deviation of returns for each symbol?\n")

#Combine the dataframes for calculations
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

# 1c) What are the estimated full-sample CAPM betas (and their standard errors) of MSFT and PG?
print("1c) What are the estimated full-sample CAPM betas (and their standard errors) of MSFT and PG?\n")

# Add constant to market (GSPC)
market_returns_with_const = sm.add_constant(gspc_returns)

# Individual Stock regressions
msft_model = sm.OLS(msft_returns, market_returns_with_const).fit()
pg_model = sm.OLS(pg_returns, market_returns_with_const).fit()

# Extract CAPM BETAS and Standard Errors
msft_beta = msft_model.params['Adj Close']
msft_se = msft_model.bse['Adj Close']
print(f"Microsoft CAPM Values:\n Beta: {msft_beta:.4f}\n Standard Error: {msft_se:.4f}")

pg_beta = pg_model.params['Adj Close']
pg_se = pg_model.bse['Adj Close']
print(f"PG CAPM Values:\n Beta: {pg_beta:.4f}\n Standard Error: {pg_se:.4f}")



# 2a) Calculate the returns to a portfolio of MSFT and PG that rebalances to 50%/50% 
#     weights at the start of January. 
print("2a) Calculate the returns to a portfolio of MSFT and PG that rebalances to 50%/50% weights at the start of January. ")

# Portfolio starts at value of 1 and rebalances to 50% of total every year
weights = pd.Series([0.5, 0.5], index=['MSFT', 'PG'])

# Convert to datetime format
combined_returns.index = pd.to_datetime(combined_returns.index)

# Create dataframes for weights and portfolio value over time
weights_df = pd.DataFrame(index=combined_returns.index, columns =['MSFT', 'PG'])
portfolio_value_df = pd.DataFrame(index=combined_returns.index, columns=['Portfolio Value'])

# Calculate returns year by year and rebalance at end of year
for year in combined_returns.index.year.unique():
  yearly_returns = combined_returns[combined_returns.index.year == year]

  for date in yearly_returns.index:
    weights = (weights * (1 + yearly_returns.loc[date]))
    portfolio_value = weights.sum()

    # Store updated weights and portfolio value in dataframes
    weights_df.loc[date] = weights
    portfolio_value_df.loc[date] = portfolio_value

  # Rebalance weights at end of year
  rebalanced_value = weights.sum() / 2
  weights['MSFT'] = rebalanced_value
  weights['PG'] = rebalanced_value

#Final Portfolio Value
final_value = portfolio_value_df.iloc[-1]['Portfolio Value']
print(f'Final Portfolio Value: {final_value * 100:.2f}% of Principal Investment')

#Calculate Daily Portfolio Returns
portfolio_returns = portfolio_value_df['Portfolio Value'].pct_change().dropna()

# 2b) What are the mean and standard deviation of returns of this portfolio?
print("2b) What are the mean and standard deviation of returns of this portfolio?")

#Mean Return of Returns of Rebalanced Portfolio
mean_return = portfolio_returns.mean()
print(f'Mean Portfolio Return of Rebalanced Portfolio: {mean_return:.6f}')

#Standard Deviation of Returns of Rebalanced Portfolio
std_return = portfolio_returns.std()
print(f'Standard Deviation of Rebalanced Portfolio Portfolio Return: {std_return:.4f}')

# 2c) What is the estimated CAPM beta and its standard error?

market_returns_with_const.index = pd.to_datetime(market_returns_with_const.index)
rebalanced_portfolio_model = sm.OLS(portfolio_returns, market_returns_with_const.loc[portfolio_returns.index]).fit()
portfolio_beta = rebalanced_portfolio_model.params['Adj Close']
portfolio_se = rebalanced_portfolio_model.bse['Adj Close']
print(f"Rebalanced Portfolio CAPM Values:\n Beta: {portfolio_beta:.4f}\n Standard Error: {portfolio_se:.4f}")

print("""
  3. Write a paragraph summarizing your observations.  How does your results in (2) compare to (1)?

MSFT shows a higher mean return (0.0917%) and greater volatility (1.6192%) compared to PG, 
which has a mean return of 0.0451% and lower volatility (1.0803%). The CAPM analysis reflects 
this, with MSFT having a beta of 1.1222, indicating higher market sensitivity, while PG's beta 
is 0.5427.
A 50%/50% portfolio rebalanced annually yielded a mean return of 0.0678% and a lower standard 
deviation of 1.15%, demonstrating reduced volatility compared to MSFT alone. The portfolio's 
CAPM beta of 0.8394 provides balanced market exposure. The portfolio achieved a final value 
of 947.81% of the principal, showing that diversification offers a more stable and attractive 
risk-adjusted return compared to individual stocks.
"""
)