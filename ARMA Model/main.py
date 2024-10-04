import pandas as pd
import argparse
import os
from data_retriever import *
import ARMA_model
import HAR_model

# Run the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # List of arguments
    parser.add_argument('-D', '--data', action='store_true', help='Retrieve Latest VIX Data')
    parser.add_argument('-A', '--artrain', action='store_true', help='Train Autoregressive model')
    parser.add_argument('-H', '--hartrain', action='store_true', help='Train HAR model')

    args = parser.parse_args()

    if args.data:
        # Retrieve latest data
        get_latest_data()
        print(f'Data stored in file')
    
    if args.artrain:
        try:
            # Check if file exists
            if not os.path.exists('Latest_VIX_Data.csv'):
                raise FileNotFoundError("Latest_VIX_Data.csv not found. Please ensure the file exists.")
            
            # Load the data
            data = pd.read_csv('Latest_VIX_Data.csv')
            
            # Check if the data is usable (non-empty)
            if data.empty or 'Close' not in data.columns:
                raise ValueError("The file Latest_VIX_Data.csv is empty or contains no usable data.")

            # Only keep 'Close' Data column for AR model
            data = data['Close']
            
            # Train and save ARMA(2,2) model to pickle file
            ARMA_model.train(data)
            print("ARMA(2,2) model trained successfully.")

        except (FileNotFoundError, ValueError) as e:
            print(f"Error during ARMA model training: {e}")

    if args.hartrain:
        try:
            # Check if file exists
            if not os.path.exists('Latest_VIX_Data.csv'):
                raise FileNotFoundError("Latest_VIX_Data.csv not found. Please ensure the file exists.")
            
            # Load the data
            data = pd.read_csv('Latest_VIX_Data.csv')
            
            # Check if the data is usable (non-empty)
            required_columns = ['Close', 'VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']
            if data.empty or not set(required_columns).issubset(data.columns):
                raise ValueError("The file Latest_VIX_Data.csv is empty or contains no usable data.")

            # Only keep 'Close' Data column for AR model
            data = data['Close']
            
            # Train and save ARMA(2,2) model to pickle file
            ARMA_model.train(data)
            print("ARMA(2,2) model trained successfully.")

        except (FileNotFoundError, ValueError) as e:
            print(f"Error during ARMA model training: {e}")