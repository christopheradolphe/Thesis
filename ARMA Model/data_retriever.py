import pandas as pd
import yfinance as yf
import time
import numpy as np
from fredapi import Fred

# Set your FRED API key
fred = Fred(api_key='cdbebbf260397c58ad92162f048ac8eb')

def load_vix_data(start_date='1990-01-02', end_date='2023-12-31'):
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data = vix_data['Close'].dropna()  # Keep only the 'Close' column and drop NaN values
    log_vix = np.log(vix_data)
    log_vix.name = 'Log_VIX_Close'
    vix_data.name = 'VIX_Close'
    vix_data = pd.concat([vix_data, log_vix], axis=1)
    return vix_data

def load_sp500_data(start_date='1990-01-02', end_date='2023-12-31'):
    adjusted_start_date = (pd.to_datetime(start_date) - pd.tseries.offsets.BDay(1))
    sp500_data = yf.download('^GSPC', start=adjusted_start_date, end=end_date)

    sp500_close = sp500_data[['Close']].dropna()
    sp500_close = sp500_close.rename(columns={'Close': 'SP500_Close'})
    sp500_volume = sp500_data['Volume'].dropna()
    sp500_volume.name = "SP500_Volume"

    # Calculate k log returns for the periods specified
    k_values = [1, 5, 10, 22, 66]
    for k in k_values:
        sp500_close[f'SP500_Log_Return_{k}'] = np.log(sp500_close['SP500_Close'] / sp500_close['SP500_Close'].shift(k))
    
    # Calculate first difference of logorithm for S&P 500 Volume
    sp500_volume = sp500_volume.replace(0, np.nan)
    sp500_volume = sp500_volume.dropna()
    sp500_volume_log = np.log(sp500_volume)
    sp500_volume_log_diff = sp500_volume_log.diff()
    sp500_volume_log_diff.name = 'SP500_Volume_Change'

    # Combine Close and Volume data
    sp500_data_combined = pd.concat([sp500_close['SP500_Close'], sp500_close[[col for col in sp500_close.columns if 'SP500_Log_Return' in col]], sp500_volume, sp500_volume_log_diff], axis=1)
    return sp500_data_combined

def load_oil_data(start_date='1990-01-02', end_date='2023-12-31'):
    # Use the West Texas Intermediate (WTI) crude oil price as a proxy
    oil_data = fred.get_series('DCOILWTICO', observation_start=start_date, observation_end=end_date)
    oil_data = oil_data.replace('.', np.nan).astype(float).fillna(method='ffill')
    oil_data = oil_data.replace(0, np.nan).dropna()
    oil_data = np.log(oil_data)
    oil_data.name = 'Log_Oil_Price'

    # Calculate k-day log returns for k ∈ {1, 5, 10, 22, 66}
    k_values = [1, 5, 10, 22, 66]
    for k in k_values:
        oil_data[f'Oil_Log_Return_{k}'] = oil_data - oil_data.shift(k)
    return oil_data

def load_usd_data(start_date='1990-01-02', end_date='2013-01-15'):
    # Use the Trade Weighted U.S. Dollar Index: Broad
    usd_data = fred.get_series('DTWEXBGS', observation_start=start_date, observation_end=end_date)
    usd_data = usd_data.replace('.', np.nan).astype(float).fillna(method='ffill')
    usd_data = usd_data.replace(0, np.nan).dropna()
    usd_data = np.log(usd_data)
    usd_change = usd_data.diff()
    usd_change.name = 'USD_Change'
    return usd_change

def load_credit_spread(start_date='1990-01-02', end_date='2013-01-15'):
    baa_yield = fred.get_series('BAA', observation_start=start_date, observation_end=end_date)
    aaa_yield = fred.get_series('AAA', observation_start=start_date, observation_end=end_date)
    credit_spread = baa_yield - aaa_yield
    credit_spread.name = 'Credit_Spread'
    credit_spread = credit_spread.fillna(method='ffill')
    return credit_spread

def load_term_spread(start_date='1990-01-02', end_date='2013-01-15'):
    dgs10 = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
    dgs3mo = fred.get_series('DGS3MO', observation_start=start_date, observation_end=end_date)
    term_spread = dgs10 - dgs3mo
    term_spread.name = 'Term_Spread'
    term_spread = term_spread.fillna(method='ffill')
    return term_spread

def load_ff_deviation(start_date='1990-01-02', end_date='2023-12-31'):
    effective_ffr = fred.get_series('FEDFUNDS', observation_start=start_date, observation_end=end_date)
    target_ffr = fred.get_series('DFEDTAR', observation_start=start_date, observation_end=end_date)
    ff_deviation = effective_ffr - target_ffr
    ff_deviation.name = 'FF_Deviation'
    ff_deviation = ff_deviation.fillna(method='ffill')
    return ff_deviation

def get_latest_data(start_date='1993-01-19', end_date='2023-12-31'):
    # Record start time
    start_time = time.time()

    # Load data
    vix_data = load_vix_data(start_date, end_date)
    sp500_data = load_sp500_data(start_date, end_date)
    oil_data = load_oil_data(start_date, end_date)
    usd_change = load_usd_data(start_date, end_date)
    credit_spread = load_credit_spread(start_date, end_date)
    term_spread = load_term_spread(start_date, end_date)
    ff_deviation = load_ff_deviation(start_date, end_date)

    # Combine all data into a single DataFrame
    data = pd.concat([vix_data, sp500_data, oil_data, usd_change, credit_spread, term_spread, ff_deviation], axis=1)

    # Drop rows with missing values
    data = data.dropna()

    # Forward VIX values for HAR output
    for i in range(1, 35):
        data[f'VIX_t+{i}'] = data['VIX_t'].shift(-i)

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

get_latest_data()