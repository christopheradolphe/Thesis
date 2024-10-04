import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pickle
from statsmodels.tsa.ar_model import AutoReg

def train(data, train_start_date='1993-01-19', train_end_date='2004-12-31'):
    # Modify train set to only include certain dates
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    model = AutoReg(train_data.reset_index(drop=True), lags=2)
    # model = ARIMA(train_data, order=(2,0,2))
    arma_model = model.fit()
    with open('arma_model.pkl', 'wb') as f:
        pickle.dump(arma_model, f)
    return arma_model

def load():
    with open('arma_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_forecasts(vix_data, model, start_date, end_date, forecast_horizons=range(7, 35)):
    forecasts = []
    test_dates = vix_data[start_date:end_date].index

    # Extract the relevant model parameters
    params = model.params
    c = params['const']
    phi1 = params['Close.L1']
    phi2 = params['Close.L2']
    # These currently not used as residuals assumed to be 0
    # theta1 = params['ma.L1']
    # theta2 = params['ma.L2']

    for t in test_dates:
        # Get data up to date t
        available_data = vix_data[:t]

        # Generate forecasts from t+1 to t+max_horizon
        steps_ahead = max(forecast_horizons)

        # Initialize list to store predicted values 
        Y_values = [available_data[-2], available_data[-1]]

        daily_forecast = {'Date': t}

        # Get forecast dates
        forecast_index = pd.date_range(start=t, periods=steps_ahead + 1, freq='B')[1:]

        # Collect forecasts for specified horizons
        for h in range(1, steps_ahead+1):
          Y_pred = c + phi1 * Y_values[-1] + phi2 * Y_values[-2]
          Y_values.append(Y_pred)

          if h in forecast_horizons:
            daily_forecast[str(h)] = Y_pred

        forecasts.append(daily_forecast)
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)

    # Reorder the columns to ensure they are in numerical order
    column_order = [str(h) for h in forecast_horizons]
    forecasts_df = forecasts_df[column_order]

    forecasts_df.to_csv('ARMA_Forecasts.csv', index=True)
    return forecasts_df

data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/ARMA Model/Latest_VIX_Data.csv', index_col=0)
data = data['Close']
generate_forecasts(data, train(data), start_date='2004-01-01', end_date='2004-01-30')