import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(data):
  correlation_matrix = data.corr()
  plt.figure(figsize=(12,10))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
  plt.title('Correlation Matrix Heatmap')
  plt.show()

def print_heavily_correlated_features(df, threshold=0.7):
  """
  For each feature in "df", this function counts the number of features that
  have a correlation coefficient with that is higher than "threshold".

  Parameters
  ----------
    df: pandas DataFrame type
        Contains the features for several data points
    threshold: float type
        he threshold for which "heavily correlated" is defined.
  """

  corr = df.corr(numeric_only=True).abs()  # calculate the correlation matrix
  ### FUNCTION APPLIED TO ENTIRE DATAFRAME****
  print(corr)
  corr = corr[corr > threshold]  # a mask of features that are heavily correlated (all non important ones become nan)
  print(corr)

  # Print out the "heavily correlated" counts
  print(corr.count().sort_values(ascending=False) - 1)

def remove_multicollinearity(df):
  # Execute the function
  print_heavily_correlated_features(df)

data = pd.read_csv('/Users/christopheradolphe/Desktop/Thesis/Latest_VIX_Data.csv', index_col=0)
vix_targets = [
    "VIX_t+1", "VIX_t+2", "VIX_t+3", "VIX_t+4", "VIX_t+5", "VIX_t+6", "VIX_t+7", 
    "VIX_t+8", "VIX_t+9", "VIX_t+10", "VIX_t+11", "VIX_t+12", "VIX_t+13", "VIX_t+14", 
    "VIX_t+15", "VIX_t+16", "VIX_t+17", "VIX_t+18", "VIX_t+19", "VIX_t+20", "VIX_t+21", 
    "VIX_t+22", "VIX_t+23", "VIX_t+24", "VIX_t+25", "VIX_t+26", "VIX_t+27", "VIX_t+28", 
    "VIX_t+29", "VIX_t+30", "VIX_t+31", "VIX_t+32", "VIX_t+33", "VIX_t+34"
]

correlation_matrix = data.corr()
corr_with_VIX_t_plus_5 = correlation_matrix['VIX_t+5'].drop(vix_targets)
corr_with_VIX_t_plus_34 = correlation_matrix['VIX_t+34'].drop(vix_targets)

# Calculate R² values
r_squared_VIX_t_plus_5 = corr_with_VIX_t_plus_5 ** 2
r_squared_VIX_t_plus_34 = corr_with_VIX_t_plus_34 ** 2

# Rank the variables based on R² values
top5_VIX_t_plus_5 = r_squared_VIX_t_plus_5.sort_values(ascending=False).head(25)
top5_VIX_t_plus_34 = r_squared_VIX_t_plus_34.sort_values(ascending=False).head(25)

# Display the top 5 variables
print("Top 5 variables correlated with VIX_t+5:")
print(top5_VIX_t_plus_5)

print("\nTop 5 variables correlated with VIX_t+34:")
print(top5_VIX_t_plus_34)

remove_multicollinearity(data)

# correlation_matrix(data)