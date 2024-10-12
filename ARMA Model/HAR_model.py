import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import pickle
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def train(data, train_start_date='1993-01-19', train_end_date='2004-12-31'):
    # Train data only in range specified by arguments
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    y = train_data['Close']
    X = train_data[['VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    with open('har_model.pkl', 'wb') as f:
      pickle.dump(har_model, f)
    return har_model

def load():
    with open('har_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_har_forecasts(har_model, data, start_date, end_date, forecast_horizons=range(1, 35)):
    """
    Generate 34-day forecasts using the HAR model with forecasted exogenous variables.

    Parameters:
    - har_model: Trained HAR model (statsmodels OLSResults object).
    - data: DataFrame containing the historical data.
    - start_date: Start date for forecasting.
    - end_date: End date for forecasting.
    - forecast_horizons: Range of forecast horizons (default is 1 to 34).

    Returns:
    - forecasts_df: DataFrame containing forecasts for each date and horizon.
    """
    forecasts = []
    test_dates = data[start_date:end_date].index

    # Extract model coefficients
    params = har_model.params
    const = params.get('const', 0)
    beta_vix_t1 = params.get('VIX_t-1', 0)
    beta_vix_t5 = params.get('VIX_t-5', 0)
    beta_vix_t22 = params.get('VIX_t-22', 0)
    beta_sp_ret_t1 = params.get('S&P Returns_t-1', 0)
    beta_volume_t1 = params.get('Volume_t-1', 0)
    beta_termspread_t1 = params.get('TermSpread_t-1', 0)

    for t in test_dates:
        # Get data up to date t
        available_data = data.loc[:t]

        # Ensure there is enough data for lagged variables
        if len(available_data) < 22:
            continue  # Skip if not enough data

        # Initialize forecast dictionary for date t
        daily_forecast = {'Date': t}

        # Initialize lists to store forecasted VIX values
        forecasted_VIX = []
        VIX_history = available_data['Close'].tolist()

        # Forecast exogenous variables
        exog_forecasts = {}
        exog_variables = ['S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']

        for var in exog_variables:
            # Get the series up to date t
            exog_series = available_data[var]

            # Check if there is enough data to fit the model
            if len(exog_series) < 30:
                # If not enough data, assume constant value
                exog_forecasts[var] = [exog_series.iloc[-1]] * max(forecast_horizons)
                continue

            # Fit AR(1) model
            try:
                model = ARIMA(exog_series, order=(1, 0, 0))
                model_fit = model.fit()

                # Forecast for horizons 1 to max(forecast_horizons)
                exog_forecast = model_fit.forecast(steps=max(forecast_horizons))

                # Store forecasts
                exog_forecasts[var] = exog_forecast.tolist()
            except:
                # If model fitting fails, assume constant value
                exog_forecasts[var] = [exog_series.iloc[-1]] * max(forecast_horizons)

        # Now, forecast VIX using the forecasted exogenous variables
        for h in range(1, max(forecast_horizons) + 1):
            # Get lagged VIX values
            idx_t_h_minus_1 = -1 + h - 1
            idx_t_h_minus_5 = -1 + h - 5
            idx_t_h_minus_22 = -1 + h - 22

            # VIX_t-1
            if idx_t_h_minus_1 < 0:
                VIX_t1_h = VIX_history[idx_t_h_minus_1]
            else:
                VIX_t1_h = forecasted_VIX[idx_t_h_minus_1]

            # VIX_t-5
            if idx_t_h_minus_5 < 0:
                VIX_t5_h = VIX_history[idx_t_h_minus_5]
            else:
                VIX_t5_h = forecasted_VIX[idx_t_h_minus_5]

            # VIX_t-22
            if idx_t_h_minus_22 < 0:
                VIX_t22_h = VIX_history[idx_t_h_minus_22]
            else:
                VIX_t22_h = forecasted_VIX[idx_t_h_minus_22]

            # Get forecasted exogenous variables at horizon h
            SP_return_t1 = exog_forecasts['S&P Returns_t-1'][h - 1]
            Volume_t1 = exog_forecasts['Volume_t-1'][h - 1]
            TermSpread_t1 = exog_forecasts['TermSpread_t-1'][h - 1]

            # Compute forecast using the HAR model equation
            y_pred = (const +
                      beta_vix_t1 * VIX_t1_h +
                      beta_vix_t5 * VIX_t5_h +
                      beta_vix_t22 * VIX_t22_h +
                      beta_sp_ret_t1 * SP_return_t1 +
                      beta_volume_t1 * Volume_t1 +
                      beta_termspread_t1 * TermSpread_t1)

            # Append the forecasted value
            forecasted_VIX.append(y_pred)

            # Store forecasts for specified horizons
            if h in forecast_horizons:
                daily_forecast[str(h)] = y_pred

        forecasts.append(daily_forecast)

    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)

    # Reorder columns
    column_order = [str(h) for h in forecast_horizons]
    forecasts_df = forecasts_df[column_order]

    forecasts_df.to_csv('HAR_Forecasts.csv', index=True)

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
# forecast = generate_har_forecasts(train(data), data, start_date='2004-01-01', end_date='2015-12-30')
# performance_summary(forecast, data['Close'])