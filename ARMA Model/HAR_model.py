import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import pickle

def train(train_data):
    y = train_data['Close']
    X = train_data[['VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit()
    with open('har_model.pkl', 'wb') as f:
      pickle.dump(har_model, f)
    return har_model

def load():
    with open('har_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def forecast_har_model(training_data, start_date, end_date, forecast_horizons=range(7, 31)):
    forecasts = []
    test_dates = training_data[start_date:end_date].index
    for t in test_dates:
        available_data = training_data[:t]
        model = train(available_data)
                
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