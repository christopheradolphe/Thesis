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

def forecast_vix(arma_result, steps=30):
    """
    Forecast future VIX values using the fitted ARMA model.
    
    Parameters:
    - arma_result: Fitted ARMA model.
    - steps: int, number of days to forecast.
    
    Returns:
    - forecast: Forecast object, containing predicted mean and confidence intervals.
    - forecast_index: DateRange, representing the forecast dates.
    """
    forecast = arma_result.get_forecast(steps=steps)
    forecast_index = pd.date_range(start=vix_data.index[-1], periods=steps+1, freq='B')[1:]
    return forecast, forecast_index

# Step 6: Forecast future VIX values (e.g., next 30 days)
forecast_steps = 30
forecast = arma_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=vix_data.index[-1], periods=forecast_steps+1, freq='B')[1:]

# Step 7: Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(vix_data[-100:], label='Historical VIX')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted VIX', color='red')
plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color='pink', alpha=0.3)
plt.title('VIX Forecast (ARMA(2,2))')
plt.xlabel('Date')
plt.ylabel('VIX Value')
plt.legend()
plt.show()