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