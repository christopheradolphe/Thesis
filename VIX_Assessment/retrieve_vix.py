import pandas as pd
import yfinance as yf
import os

ticker = "VIX"
start_date = "1991-01-12"
current_directory = os.path.dirname(os.path.abspath(__file__))

data = yf.download(ticker, start=start_date, interval='1d')
file_name = "vix_data.csv"
data.to_csv(os.path.join(current_directory, file_name))
