current_directory = os.path.dirname(os.path.abspath(__file__))

for ticker in tickers:
    # Downloading data for each ticker
    data = yf.download(ticker, start=start_date, interval='1mo')
    
    # Create a file name with the ticker symbol
    file_name = f"{ticker.replace('^', '')}_data.csv"  # Remove '^' from S&P 500 ticker for a cleaner file name
    
    # Save to CSV in the same directory
    data.to_csv(os.path.join(current_directory, file_name))
    
    print(f"Data for {ticker} saved to {file_name}")