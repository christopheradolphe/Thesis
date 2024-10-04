import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from pandas_datareader import data as web
import pickle
import statsmodels.api as sm
import argparse
import data_retriever

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

def forecast_har_model(training_data, start_date, end_date, forecast_horizons=range(7, 31)):
    forecasts = []
    test_dates = training_data[start_date:end_date].index
    for t in test_dates:
        available_data = training_data[:t]
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

    # List of arguments
    parser.add_argument('-D', '--data', action='store_true', help='Retrieve Latest VIX Data')
    parser.add_argument('-A', '--artrain', action='store_true', help='Train Autoregressive model')
    parser.add_argument('-H', '--hartrain', action='store_true', help='Train HAR model')

    args = parser.parse_args()

    if args.data:
        data_retriever.get_latest_data()
        print(f'Data stored in file')
    
    if args.artrain:
        train_ARMA_model(pd.readcsv('Latest_VIX_Data'))
    



    # Refitting the model
    # model = train_ARMA_model(vix_data[vix_data.index < '2013-12-31'].Close)

    # # Load ARMA model
    # model = load_ARMA_model()

    # # Generate Forecasts
    # generate_forecasts(vix_data.Close, '2014-01-01', '2014-06-01')