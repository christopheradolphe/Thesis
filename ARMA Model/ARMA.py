import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

def load_vix_data(start_date='2004-01-01', end_date='2023-12-31'):
    """
    Load historical VIX data from Yahoo Finance.
    
    Parameters:
    - start_date: str, start date for the VIX data.
    - end_date: str, end date for the VIX data.
    
    Returns:
    - vix_data: DataFrame, containing the VIX close prices.
    """
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data = vix_data['Close'].dropna()  # Keep only the 'Close' column and drop NaN values
    return vix_data

def plot_vix_data(vix_data):
    """
    Plot VIX historical data.
    
    Parameters:
    - vix_data: DataFrame, VIX close prices over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(vix_data, label='VIX')
    plt.title('VIX Over Time')
    plt.xlabel('Date')
    plt.ylabel('VIX Value')
    plt.legend()
    plt.show()

def fit_arma_model(vix_data, order=(2, 0, 2)):
    """
    Fit an ARMA model to the VIX data.
    
    Parameters:
    - vix_data: DataFrame, VIX close prices.
    - order: tuple, ARMA model parameters (p, d, q).
    
    Returns:
    - arma_result: ARIMA model fit result.
    """
    model = ARIMA(vix_data, order=order) 
    arma_result = model.fit()
    return arma_result

def print_model_summary(model, coefficients = False):
  """
  Print out the Model Parameters

  Parameters:
  - model: ARIMA model that was fit to data
  - coefficients: True -> only return model coefficients for terms

  Returns:
  - Model Summary or Coefficients depending on coefficients input
  """
  if coefficients:
    print('Coefficients of ARMA model: {model.params}')
  else:
    print(model.summary())

def forecast_vix(arma_result, vix_data, steps=30):
  """
  Forecast future VIX values using the fitted ARMA model.
  
  Parameters:
  - arma_result: Fitted ARMA model.
  - vix_data: Historical VIX data.
  - steps: int, number of days to forecast.
  
  Returns:
  - forecast_series: A pandas Series containing the forecasted values for the given date range.
  """
  # Generate the forecast object
  forecast = arma_result.get_forecast(steps=steps)
  
  # Create a date range starting from the last date in the historical data
  forecast_index = pd.date_range(start=vix_data.index[-1], periods=steps+1, freq='B')[1:]
  
  # Extract the forecasted mean values and create a pandas Series with forecast_index as the date index
  forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
  
  return forecast_series


def plot_forecast(vix_data, forecast, forecast_index):
  """
  Plot the historical VIX data along with the forecasted values.
  
  Parameters:
  - vix_data: DataFrame, historical VIX data (close prices).
  - forecast: Forecast object, containing predicted mean and confidence intervals.
  - forecast_index: DateRange, representing the forecast dates.
  """
  plt.figure(figsize=(10, 6))
  plt.plot(vix_data[-100:], label='Historical VIX')  # Plot last 100 historical points
  plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted VIX', color='red')
  plt.fill_between(forecast_index, 
                    forecast.conf_int().iloc[:, 0], 
                    forecast.conf_int().iloc[:, 1], 
                    color='pink', alpha=0.3)
  plt.title('VIX Forecast (ARMA(2,2))')
  plt.xlabel('Date')
  plt.ylabel('VIX Value')
  plt.legend()
  plt.show()

# Run the main function
if __name__ == '__main__':
    # Step 1: Load the historical VIX data
  vix_data = load_vix_data(start_date='2004-01-01', end_date='2023-12-31')
  
  # Step 2: Plot the historical VIX data
  plot_vix_data(vix_data)
  
  # Step 3: Build and fit the ARMA(2,2) model
  arma_result = fit_arma_model(vix_data, order=(2, 0, 2))
  
  # Step 4: Print the model summary
  print(arma_result.summary())
  
  # Step 5: Forecast future VIX values (e.g., next 30 days)
  forecast, forecast_index = forecast_vix(arma_result, steps=30)
  
  # Step 6: Plot the forecasted VIX values
  plot_forecast(vix_data, forecast, forecast_index)