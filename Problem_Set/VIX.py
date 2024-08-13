import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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

