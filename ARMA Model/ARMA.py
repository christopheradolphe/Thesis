import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from pandas_datareader import data as web
import pickle
import statsmodels.api as sm
import argparse

def load_vix_data(start_date='2004-01-01', end_date='2023-12-31'):
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

def get_latest_data(start_date='2004-01-01', end_date='2023-12-31'):
    # Load data
    vix_data = load_vix_data(start_date, end_date)
    sp500_data = load_sp500_data(start_date, end_date)
    term_spread = load_term_spread(start_date, end_date)
    
    # Combine data into a single DataFrame
    data = pd.concat([vix_data, sp500_data, term_spread], axis = 1)
    data = HAR_data_preparation(data)
    
    # Drop missing values
    data = data.dropna()
    filename = f'Latest_VIX_Data'
    data.to_csv(filename, index = True)
    print(f'Latest VIX Data Retrieved and Stored in {filename}.csv')
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

def generate_forecasts(vix_data, start_date, end_date, forecast_horizons=range(7, 34)):
    forecasts = []
    test_dates = vix_data[start_date:end_date].index

    for t in test_dates:
        # Get data up to date t
        available_data = vix_data[:t]

        # Apply the model parameters without refitting
        try:
            model = ARIMA(available_data, order=(2,0,2))
            arma_model = model.fit()
        except Exception as e:
            print(f"Error at date {t}: {e}")
            continue

        # Generate forecasts from t+1 to t+max_horizon
        steps_ahead = max(forecast_horizons)
        forecast = arma_model.get_forecast(steps=steps_ahead)

        # Get forecasted mean values
        forecast_mean = forecast.predicted_mean

        # Get forecast dates
        forecast_index = pd.date_range(start=t, periods=steps_ahead + 1, freq='B')[1:]

        daily_forecasts = {'Date' : t}
        # Collect forecasts for specified horizons
        for h in forecast_horizons:
            if h - 1 < len(forecast_mean):
                forecast_value = forecast_mean.iloc[h - 1]
                # Get actual value if available
                daily_forecasts[f'{h}'] = forecast_value
            else:
                break

        forecasts.append(daily_forecasts)
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.to_csv('ARMA Forecasts', index=True)
    return forecasts_df

def fit_har_model(train_data):
    y = train_data['Close']
    X = train_data[['VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def forecast_har_model(start_date, end_date, forecast_horizons=range(7, 31)):
    forecasts = []
    test_dates = vix_data[start_date:end_date].index
    for t in test_dates:
        available_data = vix_data[:t]
        model = fit_har_model(available_data)
                
        steps_ahead = max(forecast_horizons)
        forecast = model.predict(steps=steps_ahead)

        # Get forecasted mean values
        forecast_mean = forecast.predicted_mean

        # Get forecast dates
        forecast_index = pd.date_range(start=t, periods=steps_ahead + 1, freq='B')[1:]

        daily_forecasts = {'Date' : t}
        # Collect forecasts for specified horizons
        for h in forecast_horizons:
            if h - 1 < len(forecast_mean):
                forecast_value = forecast_mean.iloc[h - 1]
                # Get actual value if available
                daily_forecasts[f'{h}'] = forecast_value
            else:
                break

        forecasts.append(daily_forecasts)

    
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.to_csv('ARMA Forecasts', index=True)
    return forecasts_df



# Run the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--data', action='store_true', help='Retrieve Latest VIX Data')

    args = parser.parse_args()

    if args.data:
        get_latest_data()
        print(f'Data stored in file')
    
    vix_data = pd.read_csv('December 31, 2023', index_col=0)
    data = HAR_data_preparation(vix_data)

    # Refitting the model
    data.to_csv('HAR_Data_2004-2023', index=True)
    # model = train_ARMA_model(vix_data[vix_data.index < '2013-12-31'].Close)

    # # Load ARMA model
    # model = load_ARMA_model()

    # # Generate Forecasts
    # generate_forecasts(vix_data.Close, '2014-01-01', '2014-06-01')

    # Fit HAR model
    har_model = fit_har_model(data)