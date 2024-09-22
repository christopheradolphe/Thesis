import yfinance as yf
import pandas as pd

companies = ["AAPL", "MSFT", "GOOGL"]

financial_data = pd.DataFrame()

for company in companies:
  stock = yf.Ticker(company)
  df = stock.financials.T
  df['Company'] = company
  financial_data = pd.concat([financial_data, df])

financial_data.to_csv("financial_data.csv", index=False)