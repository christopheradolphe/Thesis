import matplotlib.pyplot as plt
import pandas as pd

vix_data = pd.read_csv('vix_data.csv', index_col='Date')

# plt.plot(vix_data.index, vix_data['Adj Close'], label='VIX Adj Close')
# plt.title('VIX Adjusted CLose Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Adjusted Close Price')
# plt.legend()
# plt.grid(True)
# plt.show()

election_years = [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]

vix_data.index = pd.to_datetime(vix_data.index)


for year in election_years:
  year_before = vix_data[(vix_data.index >= f'{year-1}-09-01') & (vix_data.index <= f'{year-1}-12-31')]
  election_year = vix_data[(vix_data.index >= f'{year}-09-01') & (vix_data.index <= f'{year}-12-31')]
  year_after = vix_data[(vix_data.index >= f'{year+1}-09-01') & (vix_data.index <= f'{year+1}-12-31')]

  min_y = min(year_before['Adj Close'].min(), election_year['Adj Close'].min(), year_after['Adj Close'].min())
  max_y = max(year_before['Adj Close'].max(), election_year['Adj Close'].max(), year_after['Adj Close'].max())

  fig, axes = plt.subplots(1, 3, figsize=(18,6))
  months = ['Sep', 'Oct', 'Nov', 'Dec']
  for ax in axes:
    ax.set_ylim(min_y, max_y)  # Set the same y-axis limits)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)

  axes[0].plot(year_before.index, year_before['Adj Close'], label=f'{year-1}')
  axes[0].set_title(f'Sep-Dec {year-1}')

  
  # Plot the election year
  axes[1].plot(election_year.index, election_year['Adj Close'], label=f'{year}')
  axes[1].set_title(f'Sep-Dec {year}')
  
  # Plot the year after
  axes[2].plot(year_after.index, year_after['Adj Close'], label=f'{year+1}')
  axes[2].set_title(f'Sep-Dec {year+1}')


  # Add a main title for the whole figure
  fig.suptitle(f'VIX Adjusted Close from September to December: {year-1}, {year}, {year+1}', fontsize=16)
  
  # Adjust layout to prevent overlap
  plt.tight_layout()
  plt.show()


# start_date = '2024-06-15'
# end_date = '2024-07-15'
# trump_election = vix_data[(vix_data.index >= start_date) & (vix_data.index <= end_date)]
# plt.plot(trump_election.index, trump_election['Adj Close'])
# print("Done")
