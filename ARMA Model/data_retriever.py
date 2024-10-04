import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
import time

def load_vix_data(start_date='1993-01-19', end_date='2023-12-31'):
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data = vix_data['Close'].dropna()  # Keep only the 'Close' column and drop NaN values
    return vix_data

def load_sp500_data(start_date='1993-01-19', end_date='2023-12-31'):
    adjusted_start_date = (pd.to_datetime(start_date) - pd.tseries.offsets.BDay(1))
    sp500_data = yf.download('^GSPC', start=adjusted_start_date, end=end_date)
    sp500_data = sp500_data[['Close', 'Volume']].dropna()
    sp500_data['S&P Returns'] = sp500_data['Close'].pct_change()
    return sp500_data.drop(columns = ['Close']).drop(sp500_data.index[0])

def load_term_spread(start_date='1993-01-19', end_date='2023-12-31'):
    dgs10 = web.DataReader('DGS10', 'fred', start_date, end_date)
    dgs2 = web.DataReader('DGS2', 'fred', start_date, end_date)
    term_spread = dgs10['DGS10'] - dgs2['DGS2']
    term_spread = term_spread.dropna()
    term_spread.name = 'TermSpread'
    return term_spread

def HAR_data_preparation(data):
    # Compute lagged VIX variables
    data['VIX_t-1'] = data['Close'].shift(1)
    data['VIX_t-5'] = data['Close'].rolling(window=5).mean().shift(1)
    data['VIX_t-22'] = data['Close'].rolling(window=22).mean().shift(1)
    
    # Compute lagged exogenous variables
    data['S&P Returns_t-1'] = data['S&P Returns'].shift(1)
    data['Volume_t-1'] = data['Volume'].shift(1)
    data['TermSpread_t-1'] = data['TermSpread'].shift(1)
    
    # Drop rows with NaN values
    data = data.dropna()
    return data

def get_latest_data(start_date='1993-01-19', end_date='2023-12-31'):
    # Record start time
    start_time = time.time()

    # Load data
    vix_data = load_vix_data(start_date, end_date)
    sp500_data = load_sp500_data(start_date, end_date)
    term_spread = load_term_spread(start_date, end_date)
    
    # Combine data into a single DataFrame
    data = pd.concat([vix_data, sp500_data, term_spread], axis = 1)
    data = HAR_data_preparation(data)
    
    # Drop missing values
    data = data.dropna()

    # Store data to CSV
    filename = f'Latest_VIX_Data.csv'
    data.to_csv(filename, index = True)
    print(f'Latest VIX Data Retrieved and Stored in {filename}.csv')

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time and filename
    print(f'Latest VIX Data Retrieved and Stored in {filename}.csv')
    print(f'Time taken to retrieve data: {elapsed_time:.2f} seconds')

    return data