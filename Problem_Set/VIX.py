import pandas as pd
import matplotlib.pyplot as plt

#Loading excel VIX Data
vix_data = pd.read_csv('vixdata.csv', parse_dates=['dt'], index_col='dt')

vix_data.plot(title = "VIX data")
plt.show()