import pandas as pd
import yfinance as yf
import time
import numpy as np
from fredapi import Fred

# Set your FRED API key
fred = Fred(api_key='cdbebbf260397c58ad92162f048ac8eb')

def load_vix_data(start_date='1990-01-02', end_date='2023-12-31'):
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data = vix_data[['Close']].dropna()  # Keep only the 'Close' column and drop NaN values
    vix_data = vix_data.rename(columns={'Close': 'VIX_Close'})

    log_vix = np.log(vix_data['VIX_Close']).to_frame(name='Log_VIX')

    # Calculate the averages over k days
    k_values = [1, 5, 10, 22, 66]
    for k in k_values:
        log_vix[f'Log_VIX_MA_{k}'] = log_vix['Log_VIX'].rolling(window=k).mean()
        vix_data[f'VIX_MA_{k}'] = vix_data['VIX_Close'].rolling(window=k).mean()
    
    log_vix.drop(['Log_VIX'], axis = 1, inplace = True)
    
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

        # Daily simple returns
        sp500_close['Daily_Return'] = sp500_close['SP500_Close'].pct_change()

        # Average daily simple return as a moving average over k days
        sp500_close[f'SP500_MA_{k}'] = sp500_close['Daily_Return'].rolling(window=k).mean()
    
    # Drop the intermediate Daily_Return column as it's no longer needed
    sp500_close = sp500_close.drop(columns=['Daily_Return'])

    # Calculate first difference of logorithm for S&P 500 Volume
    sp500_volume = sp500_volume.replace(0, np.nan)
    sp500_volume = sp500_volume.dropna()
    sp500_volume_log = np.log(sp500_volume)
    sp500_volume_log_diff = sp500_volume_log.diff()
    sp500_volume_log_diff.name = 'SP500_Volume_Change'


    # Combine the S&P 500 close, log returns, average daily returns, volume, and volume change data
    sp500_data_combined = pd.concat(
        [sp500_close, sp500_volume, sp500_volume_log_diff],
        axis=1
    )

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

def load_usd_data(start_date='1990-01-02', end_date='2023-12-31'):
    # Use the Trade Weighted U.S. Dollar Index: Major Currencies
    usd_data = fred.get_series('DTWEXM', observation_start=start_date, observation_end=end_date)
    usd_data = usd_data.fillna(method='ffill').dropna()
    usd_data = np.log(usd_data)
    usd_change = usd_data.diff()
    usd_change.name = 'USD_Change'
    return usd_change

def load_credit_spread(start_date='1990-01-02', end_date='2013-01-15'):
    # Retrieve monthly data
    baa_yield = fred.get_series('BAA', observation_start=start_date, observation_end=end_date)
    aaa_yield = fred.get_series('AAA', observation_start=start_date, observation_end=end_date)
    
    # Combine into a DataFrame
    credit_spread = pd.concat([baa_yield, aaa_yield], axis=1)
    credit_spread.columns = ['BAA_Yield', 'AAA_Yield']
    
    # Calculate the credit spread
    credit_spread['Credit_Spread'] = credit_spread['BAA_Yield'] - credit_spread['AAA_Yield']
    
    # Set index as datetime
    credit_spread.index = pd.to_datetime(credit_spread.index)
    
    # Resample to daily frequency
    credit_spread_daily = credit_spread.resample('B').interpolate(method='linear')
    
    # Select the 'Credit_Spread' column
    credit_spread_daily = credit_spread_daily['Credit_Spread']
    
    # Trim data to the desired date range
    credit_spread_daily = credit_spread_daily.loc[start_date:end_date]
    
    return credit_spread_daily


def load_term_spread(start_date='1990-01-02', end_date='2023-12-31'):
    dgs10 = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
    dgs3mo = fred.get_series('DGS3MO', observation_start=start_date, observation_end=end_date)
    term_spread = dgs10 - dgs3mo
    term_spread.name = 'Term_Spread'
    term_spread = term_spread.fillna(method='ffill')
    return term_spread

def load_ff_deviation(start_date='1990-01-02', end_date='2023-12-31'):
    # Effective Federal Funds Rate (Daily, only available after July 2000)
    effective_ffr_recent = fred.get_series('EFFR', observation_start='2000-07-01', observation_end=end_date)

    # Monthly Effective Federal Funds Rate (available from July 1954 to present)
    effective_ffr_old = fred.get_series('FEDFUNDS', observation_start=start_date, observation_end='2000-06-30')

    # Convert monthly data to daily frequency and interpolate
    effective_ffr_old = effective_ffr_old.resample('D').interpolate(method='linear')
    
    # Combine the two series
    effective_ffr = pd.concat([effective_ffr_old, effective_ffr_recent])
    
    # Federal Funds Target Rate (up to 2008-12-15)
    target_ffr_old = fred.get_series('DFEDTAR', observation_start=start_date, observation_end='2008-12-15')
    
    # Federal Funds Target Range Upper Limit (from 2008-12-16 onward)
    target_ffr_upper = fred.get_series('DFEDTARU', observation_start='2008-12-16', observation_end=end_date)
    
    # Federal Funds Target Range Lower Limit (from 2008-12-16 onward)
    target_ffr_lower = fred.get_series('DFEDTARL', observation_start='2008-12-16', observation_end=end_date)
    
    # Calculate the midpoint of the target range
    target_ffr_new = (target_ffr_upper + target_ffr_lower) / 2.0
    target_ffr_new.name = 'Target Rate'
    
    # Combine old and new target rates
    target_ffr = pd.concat([target_ffr_old, target_ffr_new])
    target_ffr.sort_index(inplace=True)
    
    # Forward-fill target rate to get daily data
    all_dates = pd.date_range(start_date, end_date, freq='D')
    target_ffr = target_ffr.reindex(all_dates)
    target_ffr = target_ffr.fillna(method='ffill')
    
    # Align effective_ffr with target_ffr
    effective_ffr = effective_ffr.reindex(all_dates)
    
    # Compute deviation
    ff_deviation = effective_ffr - target_ffr
    ff_deviation.name = 'FF_Deviation'
    
    # Drop missing values
    ff_deviation = ff_deviation.dropna()
    
    return ff_deviation

def get_latest_data(start_date='1993-01-19', end_date='2023-12-31'):
    # Record start time
    start_time = time.time()

    # Load data
    vix_data = load_vix_data(start_date, end_date)
    sp500_data = load_sp500_data(start_date, end_date)
    oil_data = load_oil_data(start_date, end_date)
    print(f"Oil Nans: {oil_data.isna().sum()}")
    usd_change = load_usd_data(start_date, end_date)
    print(f"USD Nans: {usd_change.isna().sum()}")
    credit_spread = load_credit_spread(start_date, end_date)
    print(f"Credit Spread Nans: {credit_spread.isna().sum()}")
    term_spread = load_term_spread(start_date, end_date)
    print(f"Term Spread Nans: {term_spread.isna().sum()}")

    ff_deviation = load_ff_deviation(start_date, end_date) # Starts out good and then ends only once a month

    # Combine all data into a single DataFrame
    data = pd.concat([vix_data, sp500_data, oil_data, usd_change, term_spread], axis=1)

    # Drop rows with missing values
    data = data.dropna()

    # Forward VIX values for HAR output
    for i in range(1, 35):
        data[f'VIX_t+{i}'] = data['VIX_Close'].shift(-i)

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