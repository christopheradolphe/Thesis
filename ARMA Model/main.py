import pandas as pd
import argparse
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
        # Train and Save ARMA 2,2 model to pickle file
        ARMA_model.train(pd.readcsv('Latest_VIX_Data'))
    if args.hartrain:
        # Train and Save HAR model to pickle file
        HAR_model.train(pd.readcsv('Latest_VIX_Data'))