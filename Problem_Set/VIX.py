import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score

#1. Estimate an AR(1) time series model of the VIX using data from 1990-2015.

#Loading excel VIX Data
vix_data = pd.read_csv('vixdata.csv', index_col='dt')

#Remove empty vix data
vix_data.dropna(subset= ['vix'], inplace=True)

#Preview Data
print(vix_data.head())

#Plot the vix data
vix_data.plot(title = "VIX data")
plt.show()

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



#Fit the AR model
in_sample_start_date = vix_data.index.get_loc('1990-01-02')
in_sample_end_date = vix_data.index.get_loc('2015-12-31')
out_sample_start_date = vix_data.index.get_loc('2016-01-04')
out_sample_end_date = vix_data.index.get_loc('2024-02-16')

model_params = AutoReg(vix_data['vix'].iloc[in_sample_start_date:in_sample_end_date+1], lags=1)
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
forecast = model_fit.predict(start=in_sample_start_date, end=out_sample_end_date)
forecast.plot(label='Forecasted VIX Data')
vix_data['vix'].plot(label='Actual VIX Data')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.title("Actual vs Forecasted VIX Data")
plt.legend()
plt.show()

in_sample_r2 = r2_score(vix_data['vix'].iloc[1:in_sample_end_date+1], forecast[1:in_sample_end_date+1])
out_sample_r2 = r2_score(vix_data['vix'].iloc[out_sample_start_date:], forecast[out_sample_start_date:])
print("In Sample R-Squared: ", in_sample_r2)
print("\nOut of Sample R-Squared: ", out_sample_r2)

