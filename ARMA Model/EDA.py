import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(data):
  correlation_matrix = data.corr()
  plt.figure(figsize=(12,10))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
  plt.title('Correlation Matrix Heatmap')
  plt.show()