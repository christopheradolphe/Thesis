import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pickle
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def train(data, train_start_date='1993-01-19', train_end_date='2004-03-31'):
    # Modify train set to only include certain dates
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    model = ARIMA(train_data, order=(2,0,2))
    arma_model = model.fit()
    print("New ARMA(2,2) trained")
    print(arma_model.summary())
    with open('arma_model.pkl', 'wb') as f:
        pickle.dump(arma_model, f)
    return arma_model

def load():
    with open('arma_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def calculate_resid(vix_data, start_date = "1993-02-19", end_date = "2015-12-31"):
    # Returns a Series with index as dates from start to end and 

    model = load()
    params = model.params
    c = params['const']
    phi1 = params['ar.L1']
    phi2 = params['ar.L2']
    theta1 = params['ma.L1']
    theta2 = params['ma.L2']

    # c = 20.083
    # phi1 = 1.651
    # phi2 = -0.654
    # theta1 = -0.714
    # theta2 = -0.064

    residuals = {}

    resid_dates = vix_data[start_date:end_date].index

    # First Date: No residual values and no previous values
    date1 = resid_dates[0]
    actual = vix_data.loc[date1]
    prediction = c
    residuals[date1] = actual - prediction

    # Second Date: Residual and lag for only one day previous
    date2 = resid_dates[1]
    actual = vix_data.loc[date2]
    prediction = c + phi1 * (vix_data.loc[date1] - c) + theta1 * residuals[date1]
    residuals[date2] = actual - prediction

    for date in resid_dates[2:]:
        dates = vix_data[:date].index
        date_before = dates[-2]
        date_2_before = dates[-3]
        # Residual is found to be 1 day ahead forecast minus actual data
        actual = vix_data.loc[date]
        prediction = c + phi1 * (vix_data.loc[date_before] - c) + phi2 * (vix_data.loc[date_2_before] - c) + theta1 * residuals[date_before] + theta2 * residuals[date_2_before]
        residuals[date] = actual - prediction

    residual_series = pd.Series(residuals)
    residual_df = pd.concat([residual_series, model.resid], axis=1)
    residual_df.columns = ["Projected Residuals", "Model Residuals"]
    residual_df['Error'] = residual_df['Projected Residuals'] - residual_df['Model Residuals']
    residual_df.index.name = "Date"
    residual_df.to_csv("residuals.csv", header=True)

    return 

def generate_forecasts(vix_data, model, start_date, end_date, forecast_horizons=range(7, 35)):
    forecasts = []
    test_dates = vix_data[start_date:end_date].index

    # Extract the relevant model parameters
    params = model.params
    c = params['const']
    phi1 = params['ar.L1']
    phi2 = params['ar.L2']
    theta1 = params['ma.L1']
    theta2 = params['ma.L2']

    # c = 20.083
    # phi1 = 1.651
    # phi2 = -0.654
    # theta1 = -0.714
    # theta2 = -0.064

    residuals_series = pd.read_csv("residuals.csv", usecols=["Projected Residuals", "Date"], index_col="Date")

    for t in test_dates:
        # Get data up to date t
        available_data = vix_data[:t]

        # Generate forecasts from t+1 to t+max_horizon
        steps_ahead = max(forecast_horizons)

        # Initialize list to store predicted values 
        Y_values = [available_data[-2], available_data[-1]]

        # Initialize set of residuals for predictions
        residuals = [residuals_series.loc[available_data.index[-2]].item(), residuals_series.loc[available_data.index[-1]].item()]

        daily_forecast = {'Date': t}

        # Get forecast dates
        forecast_index = pd.date_range(start=t, periods=steps_ahead + 1, freq='B')[1:]

        # Collect forecasts for specified horizons
        for h in range(1, steps_ahead+1):
          if h == 1: # Residuals available for t-1, t-2
              Y_pred = c + phi1 * (Y_values[-1] - c) + phi2 * (Y_values[-2] - c) + theta1 * residuals[-1] + theta2 * residuals[-2]
          elif h == 2: # Residual available for t-2
              Y_pred = c + phi1 * (Y_values[-1] - c) + phi2 * (Y_values[-2] - c) + theta2 * residuals[-1]
          else: # No residuals available -> residuals assumed to be 0
              Y_pred = c + phi1 * (Y_values[-1] - c) + phi2 * (Y_values[-2] - c)
        
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

# data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/ARMA Model/Latest_VIX_Data.csv', index_col=0)
# # model = load()
# data = data['Close']
# train(data)
# calculate_resid(data)
# forecasts_df = generate_forecasts(data, load(), start_date='2004-05-01', end_date='2015-11-30')
# performance_summary(forecasts_df, data)