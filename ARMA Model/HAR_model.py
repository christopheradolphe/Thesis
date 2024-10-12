import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import pickle
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os

def train(data, forecast_size, train_start_date='1993-01-19', train_end_date='2004-03-31'):
    # Train data only in range specified by arguments
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    if forecast_size == 1:
        y = train_data[f'VIX_t']
    else:
        y = train_data[f'VIX_t+{forecast_size - 1}']
    X = train_data[['VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    return har_model

def train_all(data, forecast_horizon, folder_name='har_models'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for day in range(1,forecast_horizon+1):
        har_model = train(data, day)
        model_path = os.path.join(folder_name, f'har_model_{day}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(har_model, f)       


def load(forecast_length, folder_name='har_models'):
    model_path = os.path.join(folder_name, f'har_model_{forecast_length}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model for horizon {forecast_length} not found at {model_path}')
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_har_forecasts(data, start_date, end_date, forecast_horizons=range(1, 35)):
    """
    Generate 34-day forecasts using the HAR model with forecasted exogenous variables.

    Parameters:
    - data: DataFrame containing the historical data.
    - start_date: Start date for forecasting.
    - end_date: End date for forecasting.
    - forecast_horizons: Range of forecast horizons (default is 1 to 34).
    - folder_name: Directory where the HAR models are stored.


    Returns:
    - forecasts_df: DataFrame containing forecasts for each date and horizon.
    """
    forecasts = []
    test_dates = data[start_date:end_date].index

    for t in test_dates:
        # Get data up to date t
        latest_data = data.loc[t]

        # Initialize forecast dictionary for date t
        daily_forecast = {'Date': t}


        # Now, forecast VIX using the forecasted exogenous variables
        for h in range(1, max(forecast_horizons) + 1):
            har_model = load(h)

            X_new = {
                    'const': 1,  # Add constant (intercept)
                    'VIX_t-1': latest_data['VIX_t-1'],  # Lag 1
                    'VIX_t-5': latest_data['VIX_t-5'],  # Lag 5
                    'VIX_t-22': latest_data['VIX_t-22'],  # Lag 22
                    'S&P Returns_t-1': latest_data['S&P Returns_t-1'],  # S&P Returns
                    'Volume_t-1': latest_data['Volume_t-1'],  # Volume
                    'TermSpread_t-1': latest_data['TermSpread_t-1']  # Term Spread
                }
            
            # Convert input data to DataFrame and add constant (if necessary)
            X_new_df = pd.DataFrame([X_new])
    
            # Make prediction using the trained HAR model
            prediction = har_model.predict(X_new_df).values[0]
            

            daily_forecast[str(h)] = prediction

        forecasts.append(daily_forecast)

    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)

    # Reorder columns
    column_order = [str(h) for h in forecast_horizons]
    forecasts_df = forecasts_df[column_order]

    forecasts_df.to_csv('HAR_Forecasts.csv', index=True)

    return forecasts_df

def output_model_coefficients(forecasts=34, folder_name='har_models'):
    har_coefficients_list = []

    for day in range(1, forecasts+1):
        model_path = os.path.join(folder_name, f'har_model_{day}.pkl')

        with open(model_path, 'rb') as f:
            har_model = pickle.load(f)
        
        coefficents = pd.DataFrame(har_model.params).T

        coefficents["Model"] = f'Model_{day}'

        har_coefficients_list.append(coefficents)
    
    har_coefficients_df = pd.concat(har_coefficients_list, ignore_index=True)
    har_coefficients_df.set_index("Model", inplace=True)

    har_coefficients_df.to_csv("HAR_Coefficients.csv")

    print("HAR Coefficnets CSV successfully created and saved to HAR_Coefficients.csv")

    return

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
train_all(data, 34)
output_model_coefficients()
# forecast = generate_har_forecasts(train(data), data, start_date='2004-01-01', end_date='2015-12-30')
# performance_summary(forecast, data['Close'])