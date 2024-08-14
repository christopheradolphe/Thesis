import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score

#1. Estimate an AR(1) time series model of the VIX using data from 1990-2015.

#Loading excel VIX Data
vix_data = pd.read_csv('vixdata.csv', parse_dates=['dt'], index_col='dt')

#Plot the vix data
# vix_data.plot(title = "VIX data")
# plt.show()

#Remove empty vix data
if vix_data['vix'].isnull().sum() != 0:
  vix_data.dropna(subset= ['vix'], inplace=True)

#Stationarity Check
stationarity_check = adfuller(vix_data['vix'])
print("ADF Statistic: ", stationarity_check[0])
print("p-value: ", stationarity_check[1])
if stationarity_check[1] > 0.05:
  print("Data does not fulfill stationarity")
  exit(-1)

#Fit the AR model
start_date = vix_data.index.get_loc('1990-01-02')
end_date = vix_data.index.get_loc('2015-12-31')

model_params = AutoReg(vix_data['vix'].iloc[start_date:end_date+1], lags=1)
model_fit = model_params.fit()
print(model_fit.summary())


# #Check residuals
# residuals = model_fit.resid
# plt.plot(residuals)
# plt.show()

#2. a) For each day in the data, use your model to 
# calculate a 21-trading-day-ahead forecast of the VIX.  
forecast_days = 21
forecasts = []


# for i in range(1, len(vix_data)):
#   forecast = model_fit.predict(start=i, end = max(i+forecast_days - 1, len(vix_data)))[:21]
#   forecasts.append(forecast.values)

# forecast_df = pd.DataFrame(forecasts, index=vix_data.index[1:], columns=vix_data.index[1:1+forecast_days])

# print(forecast_df.head())

#2b) R-squared for in sample predictions
in_sample_forecast = model_fit.predict(start=0, end=len(vix_data))
out_sample_forecast = model_fit.predict(start=len(vix_data), end=)

in_sample_r2 = r2_score(vix_data['vix'].iloc[1:], in_sample_forecast[1:-1])
print("In Sample R-Squared: ", in_sample_r2)

