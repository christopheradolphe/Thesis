import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pickle

def train(data, train_start_date='1993-01-19', train_end_date='2004-12-31'):
    # Modify train set to only include certain dates
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    model = ARIMA(train_data, order=(2,0,2))
    arma_model = model.fit()
    with open('arma_model.pkl', 'wb') as f:
        pickle.dump(arma_model, f)
    return arma_model

def load():
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

data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/ARMA Model/Latest_VIX_Data.csv', index_col=0)
data = data['Close']
train(data)