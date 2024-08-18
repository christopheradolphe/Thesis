import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

# 1. Estimate an AR(1) time series model of the VIX using data from 1990-2015.

#Load excel VIX Data into pandas dataframe
vix_data = pd.read_csv('vixdata.csv', index_col='dt', parse_dates=True)

#Remove empty vix data
vix_data.dropna(inplace=True)

#Stationarity Check
adf_test = adfuller(vix_data['vix'])
print("ADF Test Results")
print("p-value: ", adf_test[1])
print(list(adf_test[4].values())[0], type(list(adf_test[4].values())[0]))
if adf_test[1] > 0.05 or any(adf_test[0] > value for value in list(adf_test[4].values())):
  print("Data does not fulfill stationarity")
  exit(-1)
else:
  print("P-value is less than 0.05 so we can reject null hypothesis and data is stationary")

# Create indicies for in and out of sample dates
in_sample_start_date = '1990-01-02'
in_sample_end_date = '2015-12-31'
out_sample_start_date = '2016-01-04'
out_sample_end_date = '2024-02-16'
in_sample_data = vix_data[in_sample_start_date:in_sample_end_date]
out_sample_data = vix_data[out_sample_start_date:out_sample_end_date]

# Fit the AR model (with lag of 1)
model_params = AutoReg(in_sample_data['vix'].reset_index(drop=True), lags=1)
model_fit = model_params.fit()
print(model_fit.summary())

# Extract the model parameters
alpha = model_fit.params['const']
beta = model_fit.params['vix.L1']



#2. For each day in the data, use your model to calculate a 21-trading-day-ahead forecast of the VIX.  

# Using .predict() 
# forecast_21_day_in_sample = []

# for i in range(1, len(in_sample_data) - 21):
#   forecast = model_fit.predict(start=i,end=i+21, dynamic=True)[i+21]
#   forecast_21_day_in_sample.append(forecast)

# forecast_21_day_in_sample = pd.Series(forecast_21_day_in_sample, index=in_sample_data.index[22:])
# print(forecast_21_day_in_sample.head())
# in_sample_r2 = r2_score(in_sample_data.iloc[22:], forecast_21_day_in_sample)
# print("In Sample R-Squared: ", in_sample_r2)

#Forecasting In Sample Data (using alpha and beta values)
forecast_21_day_in_sample = []

for i in range(len(in_sample_data) - 22):
  value = vix_data['vix'].iloc[i]
  for _ in range(21):
    value = alpha + beta * value
  forecast_21_day_in_sample.append(value)

forecast_21_day_in_sample = pd.Series(forecast_21_day_in_sample, index=in_sample_data.index[22:])

#Forecasting Out-of-Sample Data 
forecast_21_day_out_sample = []

for i in range(len(in_sample_data), len(vix_data) - 22):
  value = vix_data['vix'].iloc[i]
  for _ in range(21):
    value = alpha + beta * value
  forecast_21_day_out_sample.append(value)

forecast_21_day_out_sample = pd.Series(forecast_21_day_out_sample, index=out_sample_data.index[22:])

# 2b) What is the R-squared of the realized VIX on your forecast value in sample (over 1990-2015)?  
#     What about out-of-sample (2016-most recent data)?

in_sample_r2 = r2_score(in_sample_data.iloc[22:], forecast_21_day_in_sample)
print("In Sample R-Squared: ", in_sample_r2)
out_sample_r2 = r2_score(out_sample_data.iloc[22:], forecast_21_day_out_sample)
print("Out Sample R-Squared: ", out_sample_r2)



#3) Plot forecasted compared to each other for observations
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot the in-sample data (actual and forecasted)
axes[0].plot(in_sample_data.index[22:], in_sample_data.iloc[22:], label='Actual VIX (In-Sample)', color='blue')
axes[0].plot(forecast_21_day_in_sample.index, forecast_21_day_in_sample, label='Forecasted VIX (In-Sample)', color='red', linestyle='--')
axes[0].set_title('In-Sample VIX: Actual vs Forecast')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('VIX')
axes[0].legend()

# Plot the out-of-sample data (actual and forecasted)
axes[1].plot(out_sample_data.index[22:], out_sample_data.iloc[22:], label='Actual VIX (Out-of-Sample)', color='blue')
axes[1].plot(forecast_21_day_out_sample.index, forecast_21_day_out_sample, label='Forecasted VIX (Out-of-Sample)', color='red', linestyle='--')
axes[1].set_title('Out-of-Sample VIX: Actual vs Forecast')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('VIX')
axes[1].legend()

plt.tight_layout()
plt.show()


"""
Summary of Findings:
The AR(1) model, estimated using VIX data from 1990-2015, achieved an in-sample R-squared of 
0.64, indicating a reasonable fit. However, the out-of-sample R-squared dropped to 0.345, 
showing the model’s limitations in predicting new data, particularly during periods of market 
stress. The model, relying only on the previous day's value, struggles with the VIX's rapid 
volatility changes.

Improvements to Model:
To improve accuracy, especially out-of-sample, a higher-order AR model could capture more 
historical patterns. Exploring ARIMA or GARCH models may also better handle the complex volatility 
dynamics. Additionally, incorporating exogenous variables like macroeconomic factors and market 
indices could enhance the model’s predictive power.

"""