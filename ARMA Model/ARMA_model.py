import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pickle
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

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

def performance_summary(forecasts_df, vix_data):
    """
    Calculate performance metrics for the 34th trading day forecast.

    Parameters:
    - forecasts_df: DataFrame containing forecasts with 'Date' as index and horizons as columns.
    - vix_data: Series or DataFrame containing the actual VIX values with dates as index.

    Returns:
    - metrics: Dictionary containing RMSE, MAE, MAPE, and R^2 for the 34th trading day forecast.
    - errors_df: DataFrame containing the errors and actual vs. forecasted values for each date.
    """
    # Ensure the index is datetime
    forecasts_df.index = pd.to_datetime(forecasts_df.index)
    vix_data.index = pd.to_datetime(vix_data.index)

    # Initialize lists to store actual and forecasted values
    actual_values = []
    forecast_values = []

    # Loop through each date in forecasts_df
    for t in forecasts_df.index:
        # Get the forecasted value for the 34th horizon
        forecast_value = forecasts_df.loc[t, '34']

        # Calculate the date 34 business days ahead
        t_plus_34 = t + pd.offsets.BusinessDay(34)

        # Check if the actual value exists in vix_data
        if t_plus_34 in vix_data.index:
            # Get the actual VIX value at t_plus_34
            actual_value = vix_data.loc[t_plus_34]

            # Store the values
            actual_values.append(actual_value)
            forecast_values.append(forecast_value)
        else:
            # Skip if the actual value is not available
            continue

    # Convert lists to numpy arrays for calculations
    actual_values = np.array(actual_values)
    forecast_values = np.array(forecast_values)

    # Calculate Errors
    errors = forecast_values - actual_values
    absolute_errors = np.abs(errors)
    percentage_errors = np.abs(errors / actual_values) * 100  # For MAPE

    # Calculate performance metrics
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_errors)
    mape = np.mean(percentage_errors)

    # Calculate out-of-sample R^2
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Create a dictionary of metrics
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R_squared': r_squared
    }

    print(f"Performance Metrics for the 34th Trading Day Forecast:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    print(f"Out-of-sample R^2: {r_squared}")

    # Create a DataFrame for errors and actual vs. forecasted values
    errors_df = pd.DataFrame({
        'Forecast_Date': forecasts_df.index[:len(errors)],
        'Actual_Date': forecasts_df.index[:len(errors)] + pd.offsets.BusinessDay(34),
        'Actual_Value': actual_values,
        'Forecast_Value': forecast_values,
        'Error': errors,
        'Absolute_Error': absolute_errors,
        'Percentage_Error': percentage_errors
    })
    errors_df.set_index('Forecast_Date', inplace=True)

    return metrics, errors_df

data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/ARMA Model/Latest_VIX_Data.csv', index_col=0)
data = data['Close']
forecasts_df = generate_forecasts(data, train(data), start_date='2004-01-01', end_date='2015-12-30')
performance_summary(forecasts_df, data)