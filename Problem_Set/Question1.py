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

#Preview Data
# print(vix_data.head())

#Plot the vix data
# vix_data.plot(title = "VIX data")
# plt.show()

#Stationarity Check
adf_test = adfuller(vix_data['vix'])
print("ADF Test Results")
print("Null Hypothesis Testing for a Unit Root")
print("ADF Statistic: ", adf_test[0])
print("p-value: ", adf_test[1])
print("Number of lags: ", adf_test[2])
print("Number of observations: ", adf_test[3])
print("Critical Values: ", adf_test[4].values())
print(list(adf_test[4].values()))
print(adf_test[0])
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


#2. a) For each day in the data, use your model to 
# calculate a 21-trading-day-ahead forecast of the VIX.  
forecast_21_day_in_sample = []

for i in range(1, len(in_sample_data) - 21):
  forecast = model_fit.predict(start=i,end=i+21, dynamic=True)[i+21]
  forecast_21_day_in_sample.append(forecast)

forecast_21_day_in_sample = pd.Series(forecast_21_day_in_sample, index=in_sample_data.index[22:])
print(forecast_21_day_in_sample.head())
in_sample_r2 = r2_score(in_sample_data.iloc[22:], forecast_21_day_in_sample)
print("In Sample R-Squared: ", in_sample_r2)

forecast_21_day_out_sample = []

for i in range(len(in_sample_data), len(vix_data)):
  forecast = model_fit.predict(start=i,end=i+21, dynamic=True)[i+21]
  forecast_21_day_out_sample.append(forecast)

forecast_21_day_out_sample = pd.Series(forecast_21_day_out_sample, index=out_sample_data.index[22:])
print(forecast_21_day_in_sample.head())
in_sample_r2 = r2_score(out_sample_data.iloc[22:], forecast_21_day_out_sample)
print("In Sample R-Squared: ", in_sample_r2)

#2b) R-squared for in sample predictions
forecast = model_fit.predict(start=0, end=len(vix_data))
forecast.plot(label='Forecasted In Sample VIX Data')
vix_data['vix'].plot(label='Actual VIX Data')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.title("Actual vs Forecasted VIX Data")
plt.legend()
plt.show()

# in_sample_r2 = r2_score(vix_data['vix'].iloc[1:in_sample_end_date+1], forecast_in_sample.iloc[1:])
# out_sample_r2 = r2_score(vix_data['vix'].iloc[out_sample_start_date:], forecast_out_sample)
# print("In Sample R-Squared: ", in_sample_r2)
# print("\nOut of Sample R-Squared: ", out_sample_r2)


#3. Suggestions to improve model
pacf = plot_pacf(vix_data['vix'], lags=10)