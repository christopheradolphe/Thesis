import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.regression.linear_model import OLS
import pickle

def load_vix_data(start_date='2004-01-01', end_date='2023-12-31'):
    """
    Load historical VIX data from Yahoo Finance.
    
    Parameters:
    - start_date: str, start date for the VIX data.
    - end_date: str, end date for the VIX data.
    
    Returns:
    - vix_data: Series, containing the VIX close prices.
    """
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data = vix_data['Close'].dropna()  # Keep only the 'Close' column and drop NaN values
    return vix_data

def load_sp500_data(start_date='2004-01-01', end_date='2023-12-31'):
    adjusted_start_date = (pd.to_datetime(start_date) - pd.tseries.offsets.BDay(1))
    sp500_data = yf.download('^GSPC', start=adjusted_start_date, end=end_date)
    sp500_data = sp500_data[['Close', 'Volume']].dropna()
    sp500_data['S&P Returns'] = sp500_data['Close'].pct_change()
    return sp500_data.drop(columns = ['Close']).drop(sp500_data.index[0])

def load_term_spread(start_date='2004-01-01', end_date='2023-12-31'):
    dgs10 = web.DataReader('DGS10', 'fred', start_date, end_date)
    dgs2 = web.DataReader('DGS2', 'fred', start_date, end_date)
    term_spread = dgs10['DGS10'] - dgs2['DGS2']
    term_spread = term_spread.dropna()
    term_spread.name = 'TermSpread'
    return term_spread


def prepare_data(start_date='2004-01-01', end_date='2023-12-31'):
    # Load data
    vix_data = load_vix_data(start_date, end_date)
    sp500_data = load_sp500_data(start_date, end_date)
    term_spread = load_term_spread(start_date, end_date)
    
    # Combine data into a single DataFrame
    data = pd.concat([vix_data, sp500_data, term_spread], axis = 1)
    
    # Drop missing values
    data = data.dropna()
    return data

def train_ARMA_model(train_data):
    model = ARIMA(train_data, order=(2,0,2))
    arma_model = model.fit()
    with open('arma_model.pkl', 'wb') as f:
        pickle.dump(arma_model, f)

    return arma_model

def load_ARMA_model():
    with open('arma_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def rolling_forecast_errors(vix_data, train_end_date, min_horizon=7, max_horizon=30):
    """
    Perform rolling forecast and compute errors for each forecast horizon.

    Parameters:
    - vix_data: pandas Series, VIX data.
    - train_end_date: str or datetime, the end date of initial training data.
    - min_horizon: int, minimum forecast horizon (e.g., 7 days).
    - max_horizon: int, maximum forecast horizon (e.g., 30 days).

    Returns:
    - mse_df: DataFrame, with MSE for each horizon from min_horizon to max_horizon.
    """
    train_end_date = pd.to_datetime(train_end_date)
    
    # Get the list of dates in the out-of-sample period
    out_of_sample_dates = vix_data.index[vix_data.index > train_end_date]
    
    horizons = range(min_horizon, max_horizon + 1)
    
    # Dictionary to store errors for each horizon
    errors = {h: [] for h in horizons}
    
    # For each date t in out-of-sample period
    for t in out_of_sample_dates:
        # Define the training data up to date t
        train_data = vix_data[:t]
        
        # Fit the ARMA(2,2) model
        try:
            model = ARIMA(train_data, order=(2, 0, 2))
            arma_result = model.fit()
        except Exception as e:
            # If the model fails to fit, skip this date
            print(f"Model failed to fit at date {t}: {e}")
            continue
        
        # Forecast up to max_horizon steps ahead
        forecast = arma_result.get_forecast(steps=max_horizon)
        forecast_values = forecast.predicted_mean
        
        # Forecast dates
        forecast_dates = pd.date_range(start=t, periods=max_horizon + 1, freq='B')[1:]
        
        # For each horizon h
        for h in horizons:
            if h - 1 >= len(forecast_values):
                continue  # Not enough forecasted values
            forecast_value = forecast_values.iloc[h - 1]
            
            # Actual value at t + h
            if h - 1 >= len(forecast_dates):
                continue  # Not enough forecast dates
            forecast_date = forecast_dates[h - 1]
            if forecast_date in vix_data.index:
                actual_value = vix_data.loc[forecast_date]
                # Compute squared error
                error = (forecast_value - actual_value) ** 2
                errors[h].append(error)
            else:
                # If we don't have actual data for forecast_date, skip
                continue
                
    # Compute MSE for each horizon
    mse = {h: np.mean(errors[h]) if len(errors[h]) > 0 else np.nan for h in horizons}
    
    # Convert to DataFrame
    mse_df = pd.DataFrame.from_dict(mse, orient='index', columns=['MSE'])
    
    return mse_df

# Run the main function
if __name__ == '__main__':
    # Getting data
    data = prepare_data(start_date='2004-01-01', end_date='2023-12-31')
    data.to_csv('December 31, 2023', index = True)

    # Refitting the model
    vix_data = pd.read_csv('December 31, 2023', index_col=0)
    model = train_ARMA_model(vix_data["Close"])

    # Load ARMA model
    model = load_ARMA_model()