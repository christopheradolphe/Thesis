import pandas as pd
import numpy as np
import os
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import time

# Define a scaler to standardize features for SVR
scaler = StandardScaler()

# Directory to save SVR models
folder_name = 'svr_models'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)



def train_svr(data, forecast_horizon, train_start_date='1993-01-19', train_end_date='2004-03-31'):
    # Filter training data based on the specified date range
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    
    # Define target variable for the specified horizon
    y = train_data[f'VIX_t+{forecast_horizon}']
    
    # Select features and scale them
    X = train_data[['Log_VIX_MA_1', 'Log_VIX_MA_5', 'Log_VIX_MA_10', 'Log_VIX_MA_22', 'Log_VIX_MA_66', 
                    'SP500_Log_Return_1', 'SP500_Log_Return_5', 'SP500_Log_Return_10', 'SP500_Log_Return_22', 
                    'SP500_Log_Return_66', 'SP500_Volume_Change', 'Log_Oil_Price', 'USD_Change', 'Term_Spread']]
    
    X_scaled = scaler.fit_transform(X)  # Standardize features
    
    # Initialize SVR model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_scaled, y)
    
    # Save model to a pickle file
    model_path = os.path.join(folder_name, f'svr_model_{forecast_horizon}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(svr_model, f)
    
    return svr_model

def train_all_svr(data, max_forecast_horizon=34):
    for day in range(1, max_forecast_horizon + 1):
        train_svr(data, day)
    print(f"Trained and saved SVR models for horizons 1 to {max_forecast_horizon} days.")


def load_svr(forecast_horizon):
    model_path = os.path.join(folder_name, f'svr_model_{forecast_horizon}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model for horizon {forecast_horizon} not found at {model_path}')
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_svr_forecasts(data, start_date, end_date, forecast_horizons=range(1, 35)):
    start_time = time.time()
    forecasts = []
    test_dates = data[start_date:end_date].index
    
    for t in test_dates:
        # Get the latest data up to date t
        latest_data = data.loc[t]
        daily_forecast = {'Date': t}
        
        for h in forecast_horizons:
            svr_model = load_svr(h)
            
            # Prepare the input data for prediction
            X_new = latest_data[['Log_VIX_MA_1', 'Log_VIX_MA_5', 'Log_VIX_MA_10', 'Log_VIX_MA_22', 
                                 'Log_VIX_MA_66', 'SP500_Log_Return_1', 'SP500_Log_Return_5', 
                                 'SP500_Log_Return_10', 'SP500_Log_Return_22', 'SP500_Log_Return_66', 
                                 'SP500_Volume_Change', 'Log_Oil_Price', 'USD_Change', 'Term_Spread']].values.reshape(1, -1)
            X_new_scaled = scaler.transform(X_new)  # Scale the input
            
            # Make prediction
            prediction = svr_model.predict(X_new_scaled)[0]
            daily_forecast[str(h)] = prediction
        
        forecasts.append(daily_forecast)
    
    end_time = time.time()
    print(f"Total time for SVR forecasting up to {max(forecast_horizons)} days: {end_time - start_time} seconds")
    
    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)
    
    # Save forecasts to CSV
    forecasts_df.to_csv('SVR_Forecasts.csv', index=True)
    
    return forecasts_df

def svr_performance_summary(vix_data):
    """
    Calculate performance metrics for the 34th trading day forecast using SVR.

    Parameters:
    - vix_data: Series or DataFrame containing the actual VIX values with dates as index.

    Returns:
    - metrics: Dictionary containing MFE, SDFE, MSE, RMSE, MAE, MAPE, and R^2 for the 34th trading day forecast.
    - errors_df: DataFrame containing the errors and actual vs. forecasted values for each date.
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression

    # Load forecast data
    forecasts_df = pd.read_csv("SVR_Forecasts.csv", index_col=0)
    forecasts_df.index = pd.to_datetime(forecasts_df.index)
    vix_data.index = pd.to_datetime(vix_data.index)

    actual_values = []
    forecast_values = []
    forecast_dates = []

    # Evaluate performance for the 34th day forecast
    for t in forecasts_df.index:
        if '34' not in forecasts_df.columns:
            continue

        forecast_value = forecasts_df.loc[t, '34']
        t_plus_34 = t + pd.offsets.BusinessDay(34)

        if t_plus_34 in vix_data.index:
            actual_value = vix_data.loc[t_plus_34, 'VIX_Close']
            actual_values.append(actual_value)
            forecast_values.append(forecast_value)
            forecast_dates.append(t)

    actual_values = np.array(actual_values)
    forecast_values = np.array(forecast_values)
    errors = forecast_values - actual_values
    absolute_errors = np.abs(errors)
    percentage_errors = np.abs(errors / actual_values) * 100  # For MAPE

    # Calculate performance metrics
    mfe = np.mean(errors)
    sdfe = np.std(errors, ddof=1)
    mse = mean_squared_error(actual_values, forecast_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, forecast_values)
    mape = np.mean(percentage_errors)

    # In-Sample R² (Mincer-Zarnowitz R²)
    X = np.column_stack((np.ones(len(forecast_values)), forecast_values))  # Add constant
    model = LinearRegression().fit(X, actual_values)
    r_squared = model.score(X, actual_values)

    # Out-of-Sample R²
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

    print(f"Performance Metrics for the 34th Day Forecast:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # Create a DataFrame for errors and actual vs. forecasted values
    errors_df = pd.DataFrame({
        'Forecast_Date': forecast_dates,
        'Actual_Value': actual_values,
        'Forecast_Value': forecast_values,
        'Error': errors,
        'Absolute_Error': absolute_errors,
        'Percentage_Error': percentage_errors
    })
    errors_df.set_index('Forecast_Date', inplace=True)
    
    return metrics, errors_df

data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/Latest_VIX_Data.csv', index_col=0)
# train_all_svr(data, 34)
# generate_svr_forecasts(data, start_date='2004-05-01', end_date='2015-10-30')
svr_performance_summary(data)