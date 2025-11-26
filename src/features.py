import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_returns(data: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the target variable (future returns) and the input data (current prices).
    
    Args:
        data (pd.DataFrame): DataFrame of historical adjusted closing prices.
        prediction_horizon (int): Number of trading days ahead to predict (e.g., 5 for next week).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - features_df (pd.DataFrame): Data ready for ML, with dates matching the targets.
            - targets_df (pd.DataFrame): DataFrame of future returns, indexed by the start date of the prediction.
    """
    targets_df = data.pct_change(prediction_horizon).shift(-prediction_horizon)
    features_df = data.copy() 
    
    valid_index = targets_df.dropna().index
    features_df = features_df.loc[valid_index]
    targets_df = targets_df.loc[valid_index]
    
    return features_df, targets_df

def add_technical_indicators(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds various technical indicators to the features DataFrame.
    
    Args:
        features_df (pd.DataFrame): DataFrame of adjusted prices.

    Returns:
        pd.DataFrame: DataFrame augmented with technical indicators.
    """
    # Create an empty DataFrame to store all new features
    # We use the original index for alignment
    new_features = pd.DataFrame(index=features_df.index)

    # Loop through each ticker to calculate indicators individually
    for ticker in features_df.columns:
        prices = features_df[ticker] # Select the price series for one ticker

        # 1. Calculate Simple Returns (1-day lagged returns)
        daily_return = prices.pct_change(1)
        new_features[f'{ticker}_daily_return'] = daily_return

        # 2. Calculate Volatility (Standard Deviation of daily returns over 20 days)
        # Apply rolling calculation on the daily_return series
        volatility = daily_return.rolling(window=20).std() * np.sqrt(252) # Annulized
        new_features[f'{ticker}_volatility'] = volatility

        # 3. Calculate Moving Averages (Simple Moving Averages - SMA)
        # Apply rolling calculation on the price series
        new_features[f'{ticker}_SMA_50'] = prices.rolling(window=50).mean()
        new_features[f'{ticker}_SMA_200'] = prices.rolling(window=200).mean()

        # 4. Calculate RSI (Relative Strength Index)
        delta = daily_return.dropna()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # We must use EWM directly on the series
        avg_gain = gains.ewm(com=13, adjust=False).mean()
        avg_loss = losses.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        new_features[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

        # 5. Add current price and lagged prices as features
        new_features[f'{ticker}_price'] = prices
        new_features[f'{ticker}_price_lag_1'] = prices.shift(1)


    # Merge new features with the original features (prices)
    features_df = pd.concat([features_df, new_features], axis=1)

    # Drop the first 200 rows because they contain NaNs from the largest rolling calculations (SMA_200)
    features_df.dropna(inplace=True)

    return features_df

if __name__ == "__main__":
    # We need enough data points (e.g., 250 rows) for the 200-day SMA to calculate
    dates = pd.to_datetime(pd.date_range('2024-01-01', periods=250, freq='D'))
    mock_data = pd.DataFrame({
        'TEST_A': np.linspace(100, 200, 250) + np.random.randn(250) * 5,
        'TEST_B': np.linspace(50, 150, 250) + np.random.randn(250) * 3
    }, index=dates)
    mock_data.index.name = 'Date'

    # 1. Calculate returns and align features/targets (Horizon = 5 days)
    features, targets = calculate_returns(mock_data, prediction_horizon=5)
    
    # 2. Add Indicators (The core of this step)
    features_with_indicators = add_technical_indicators(features)
    
    print("\n--- FEATURES WITH INDICATORS ---")
    print("Columns:", features_with_indicators.columns.tolist())
    print("\nShape after cleaning NaNs:", features_with_indicators.shape)
    print(features_with_indicators.tail(5)) # Check the end of the data
    
    # Check the column count: 2 (Original) + 6 (New Features per Ticker) * 2 Tickers = 14 columns
    print(f"\nTotal Columns: {len(features_with_indicators.columns)}. Expected columns (2 original + 12 new) = 14.")