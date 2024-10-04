import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import pickle

def train(data, train_start_date='1993-01-19', train_end_date='2004-12-31'):
    # Train data only in range specified by arguments
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    y = train_data['Close']
    X = train_data[['VIX_t-1', 'VIX_t-5', 'VIX_t-22', 'S&P Returns_t-1', 'Volume_t-1', 'TermSpread_t-1']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    har_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    with open('har_model.pkl', 'wb') as f:
      pickle.dump(har_model, f)
    return har_model

def load():
    with open('har_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def generate_har_forecasts(har_model, data, start_date, end_date, forecast_horizons=range(1, 35)):
    """
    Generate 34-day forecasts using the HAR model.

    Parameters:
    - har_model: Trained HAR model (statsmodels OLSResults object).
    - data: DataFrame containing the historical data.
    - start_date: Start date for forecasting.
    - end_date: End date for forecasting.
    - forecast_horizons: Range of forecast horizons (default is 1 to 34).

    Returns:
    - forecasts_df: DataFrame containing forecasts for each date and horizon.
    """
    forecasts = []
    test_dates = data[start_date:end_date].index

    # Extract model coefficients
    params = har_model.params
    const = params.get('const', 0)
    beta_vix_t1 = params.get('VIX_t-1', 0)
    beta_vix_t5 = params.get('VIX_t-5', 0)
    beta_vix_t22 = params.get('VIX_t-22', 0)
    beta_sp_ret_t1 = params.get('S&P Returns_t-1', 0)
    beta_volume_t1 = params.get('Volume_t-1', 0)
    beta_termspread_t1 = params.get('TermSpread_t-1', 0)

    for t in test_dates:
        # Get data up to date t
        available_data = data.loc[:t]

        # Ensure there is enough data for lagged variables
        if len(available_data) < 22:
            continue  # Skip if not enough data

        # Initialize forecast dictionary for date t
        daily_forecast = {'Date': t}

        # Initialize lists to store forecasted VIX values
        forecasted_VIX = []
        VIX_history = available_data['Close'].tolist()

        # Prepare initial lagged VIX values
        for h in range(1, max(forecast_horizons) + 1):
            # Get lagged VIX values
            idx_t_h_minus_1 = -1 + h - 1
            idx_t_h_minus_5 = -1 + h - 5
            idx_t_h_minus_22 = -1 + h - 22

            # VIX_t-1
            if idx_t_h_minus_1 < 0:
                VIX_t1_h = VIX_history[idx_t_h_minus_1]
            else:
                VIX_t1_h = forecasted_VIX[idx_t_h_minus_1]

            # VIX_t-5
            if idx_t_h_minus_5 < 0:
                VIX_t5_h = VIX_history[idx_t_h_minus_5]
            else:
                VIX_t5_h = forecasted_VIX[idx_t_h_minus_5]

            # VIX_t-22
            if idx_t_h_minus_22 < 0:
                VIX_t22_h = VIX_history[idx_t_h_minus_22]
            else:
                VIX_t22_h = forecasted_VIX[idx_t_h_minus_22]

            # Handle exogenous variables (assuming they remain constant)
            SP_return_t1 = available_data['S&P Returns_t-1'].iloc[-1]
            Volume_t1 = available_data['Volume_t-1'].iloc[-1]
            TermSpread_t1 = available_data['TermSpread_t-1'].iloc[-1]

            # Compute forecast using the HAR model equation
            y_pred = (const +
                      beta_vix_t1 * VIX_t1_h +
                      beta_vix_t5 * VIX_t5_h +
                      beta_vix_t22 * VIX_t22_h +
                      beta_sp_ret_t1 * SP_return_t1 +
                      beta_volume_t1 * Volume_t1 +
                      beta_termspread_t1 * TermSpread_t1)

            # Append the forecasted value
            forecasted_VIX.append(y_pred)

            # Store forecasts for specified horizons
            if h in forecast_horizons:
                daily_forecast[str(h)] = y_pred

        forecasts.append(daily_forecast)

    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.set_index('Date', inplace=True)

    # Reorder columns
    column_order = [str(h) for h in forecast_horizons]
    forecasts_df = forecasts_df[column_order]

    return forecasts_df