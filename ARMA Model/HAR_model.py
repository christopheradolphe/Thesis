import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import pickle
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os
import time

def train_fernandes(data, forecast_size, train_start_date='1993-01-19', train_end_date='2004-03-31'):
    # Train data only in range specified by arguments
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    y = np.log(train_data[f'VIX_t+{forecast_size}'])
    X = train_data[['Log_VIX_MA_1', 'Log_VIX_MA_5', 'Log_VIX_MA_10', 'Log_VIX_MA_22', 'Log_VIX_MA_66', 'SP500_Log_Return_1', 'SP500_Log_Return_5',
     'SP500_Log_Return_10', 'SP500_Log_Return_22', 'SP500_Log_Return_66', 'SP500_Volume_Change', 'Log_Oil_Price', 'USD_Change', 'Term_Spread']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    return har_model

def train(data, forecast_size, train_start_date='1993-01-19', train_end_date='2004-03-31'):
    # Train data only in range specified by arguments
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    y = train_data[f'VIX_t+{forecast_size}']
    X = train_data[['Log_VIX_MA_1', 'Log_VIX_MA_5', 'Log_VIX_MA_10', 'Log_VIX_MA_22', 'Log_VIX_MA_66', 'SP500_Log_Return_1', 'SP500_Log_Return_5',
     'SP500_Log_Return_10', 'SP500_Log_Return_22', 'SP500_Log_Return_66', 'SP500_Volume_Change', 'Log_Oil_Price', 'USD_Change', 'Term_Spread']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    return har_model

def train_all(data, forecast_horizon, folder_name='har_models', fernandes=False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for day in range(1,forecast_horizon+1):
        if fernandes:
            har_model = train_fernandes(data, day)
            model_path = os.path.join(folder_name, f'har_fernandes_model_{day}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(har_model, f)       
        else:
            har_model = train(data, day)
            model_path = os.path.join(folder_name, f'har_model_{day}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(har_model, f) 


def load(forecast_length, folder_name='har_models', fernandes=False):
    if fernandes:
        model_path = os.path.join(folder_name, f'har_fernandes_model_{forecast_length}.pkl')
    else:
        model_path = os.path.join(folder_name, f'har_model_{forecast_length}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model for horizon {forecast_length} not found at {model_path}')
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_har_forecasts(data, start_date, end_date, forecast_horizons=range(1, 35), fernandes=False):
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

    start_time = time.time()
    forecasts = []
    test_dates = data[start_date:end_date].index

    for t in test_dates:
        # Get data up to date t
        latest_data = data.loc[t]

        # Initialize forecast dictionary for date t
        daily_forecast = {'Date': t}


        # Now, forecast VIX using the forecasted exogenous variables
        for h in range(1, max(forecast_horizons) + 1):
            har_model = load(h, fernandes=fernandes)

            X_new = {
                'const': 1,  # Add constant (intercept)
                'Log_VIX_MA_1': latest_data['Log_VIX_MA_1'],     # 1-day VIX moving average
                'Log_VIX_MA_5': latest_data['Log_VIX_MA_5'],     # 5-day VIX moving average
                'Log_VIX_MA_10': latest_data['Log_VIX_MA_10'],   # 10-day VIX moving average
                'Log_VIX_MA_22': latest_data['Log_VIX_MA_22'],   # 22-day VIX moving average
                'Log_VIX_MA_66': latest_data['Log_VIX_MA_66'],   # 66-day VIX moving average
                'SP500_Log_Return_1': latest_data['SP500_Log_Return_1'],   # 1-day S&P500 log return
                'SP500_Log_Return_5': latest_data['SP500_Log_Return_5'],   # 5-day S&P500 log return
                'SP500_Log_Return_10': latest_data['SP500_Log_Return_10'], # 10-day S&P500 log return
                'SP500_Log_Return_22': latest_data['SP500_Log_Return_22'], # 22-day S&P500 log return
                'SP500_Log_Return_66': latest_data['SP500_Log_Return_66'], # 66-day S&P500 log return
                'SP500_Volume_Change': latest_data['SP500_Volume_Change'], # Volume change
                'Log_Oil_Price': latest_data['Log_Oil_Price'],             # Log of oil price
                'USD_Change': latest_data['USD_Change'],                   # USD change
                'Term_Spread': latest_data['Term_Spread']                  # Term spread
            }
            
            # Convert input data to DataFrame and add constant (if necessary)
            X_new_df = pd.DataFrame([X_new])
    
            # Make prediction using the trained HAR model
            prediction = har_model.predict(X_new_df).values[0]
            

            daily_forecast[str(h)] = prediction

        forecasts.append(daily_forecast)

    end_time = time.time()
    print(f"Total time to Forecast HAR model for {max(forecast_horizons)} day forecasts: {end_time-start_time}")
    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)

    # Reorder columns
    column_order = [str(h) for h in forecast_horizons]
    forecasts_df = forecasts_df[column_order]

    forecasts_df.to_csv('HAR_Forecasts.csv', index=True)

    return forecasts_df

def output_model_coefficients(forecasts=34, folder_name='har_models', fernandes=False):
    har_coefficients_list = []

    for day in range(1, forecasts+1):
        if fernandes:
            model_path = os.path.join(folder_name, f'har_fernandes_model_{day}.pkl')
        else:
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

def performance_summary(vix_data, fernandes=False):
    """
    Calculate performance metrics for the 34th trading day forecast.

    Parameters:
    - vix_data: Series or DataFrame containing the actual VIX values with dates as index.

    Returns:
    - metrics: Dictionary containing MFE, SDFE, MSE, MAE, RMSE, MAPE, and R^2 for the 34th trading day forecast.
    - errors_df: DataFrame containing the errors and actual vs. forecasted values for each date.
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm

    # Ensure the index is datetime
    forecasts_df = pd.read_csv("HAR_Forecasts.csv", index_col=0)
    forecasts_df.index = pd.to_datetime(forecasts_df.index)
    vix_data.index = pd.to_datetime(vix_data.index)

    # Initialize lists to store actual and forecasted values
    actual_values = []
    forecast_values = []
    forecast_dates = []

    # Loop through each date in forecasts_df
    for t in forecasts_df.index:
        # Check if horizon '34' is in forecasts_df columns
        if '34' not in forecasts_df.columns:
            continue  # Skip if the horizon is not available

        # Get the forecasted value for the 34th horizon
        if fernandes:
            forecast_value = forecasts_df.loc[t, '22']
        else:
            forecast_value = forecasts_df.loc[t, '34']

        # Calculate the date 34 business days ahead
        t_plus_34 = t + pd.offsets.BusinessDay(22 if fernandes else 34)

        # Check if the actual value exists in vix_data
        if t_plus_34 in vix_data.index:
            # Get the actual VIX value at t_plus_34
            if isinstance(vix_data, pd.Series):
                actual_value = np.log(vix_data.loc[t_plus_34])
            elif isinstance(vix_data, pd.DataFrame):
                actual_value = np.log(vix_data.loc[t_plus_34, 'VIX_Close'])  # Adjust 'VIX' to the correct column name
            else:
                continue

            # Store the values
            actual_values.append(actual_value)
            forecast_values.append(forecast_value)
            forecast_dates.append(t)
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
    mfe = np.mean(errors)
    sdfe = np.std(errors, ddof=1)  # Sample standard deviation
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_errors)
    mape = np.mean(percentage_errors)

    # Mincer-Zarnowitz Regression for Out-of-Sample R²
    X = sm.add_constant(forecast_values)  # Add intercept
    model = sm.OLS(actual_values, X).fit()
    r_squared = model.rsquared

    # Calculate Out-of-Sample R² using total variance in actual values
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    out_of_sample_r_squared = 1 - (ss_res / ss_tot)

    # Create a dictionary of metrics
    metrics = {
        'MFE': mfe,
        'SDFE': sdfe,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R_squared': r_squared,
        'Out_of_Sample_R_squared': out_of_sample_r_squared
    }

    print(f"Performance Metrics for the {22 if fernandes else 34}th Trading Day Forecast:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # Create a DataFrame for errors and actual vs. forecasted values
    errors_df = pd.DataFrame({
        'Forecast_Date': forecast_dates,
        'Actual_Date': [t + pd.offsets.BusinessDay(22 if fernandes else 34) for t in forecast_dates],
        'Actual_Value': actual_values,
        'Forecast_Value': forecast_values,
        'Error': errors,
        'Absolute_Error': absolute_errors,
        'Percentage_Error': percentage_errors
    })
    errors_df.set_index('Forecast_Date', inplace=True)

    return metrics, errors_df


data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/Latest_VIX_Data.csv', index_col=0)
train_all(data, 34, fernandes=True)
output_model_coefficients()
generate_har_forecasts(data, start_date='2004-05-01', end_date='2013-10-30', fernandes=True)
performance_summary(data, fernandes=True)
