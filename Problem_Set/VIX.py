import pandas as pd

#Loading excel VIX Data
vix_data = pd.read_csv('vixdata.csv', parse_dates=['dt'], index_col='dt')

