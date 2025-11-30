import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_returns(data: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    targets_df = data.pct_change(prediction_horizon).shift(-prediction_horizon)
    features_df = data.copy() 
    
    valid_index = targets_df.dropna().index
    features_df = features_df.loc[valid_index]
    targets_df = targets_df.loc[valid_index]
    
    return features_df, targets_df

def add_technical_indicators(features_df: pd.DataFrame) -> pd.DataFrame:
    
    new_features = pd.DataFrame(index=features_df.index)

    for ticker in features_df.columns:
        prices = features_df[ticker] 

        # 1. Calculate Simple Returns (1-day lagged returns)
        daily_return = prices.pct_change(1)
        new_features[f'{ticker}_daily_return'] = daily_return

        # 2. Calculate Volatility (Standard Deviation of daily returns over 20 days)
        volatility = daily_return.rolling(window=20).std() * np.sqrt(252) # Annulized
        new_features[f'{ticker}_volatility'] = volatility

        # 3. Calculate Moving Averages (Simple Moving Averages - SMA)
        new_features[f'{ticker}_SMA_50'] = prices.rolling(window=50).mean()
        new_features[f'{ticker}_SMA_200'] = prices.rolling(window=200).mean()

        # 4. Calculate RSI (Relative Strength Index)
        delta = daily_return.dropna()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.ewm(com=13, adjust=False).mean()
        avg_loss = losses.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        new_features[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

        new_features[f'{ticker}_price'] = prices
        new_features[f'{ticker}_price_lag_1'] = prices.shift(1)


    features_df = pd.concat([features_df, new_features], axis=1)

    # Drop the first 200 rows because they contain NaNs from the largest rolling calculations (SMA_200)
    features_df.dropna(inplace=True)

    return features_df

if __name__ == "__main__":
    dates = pd.to_datetime(pd.date_range('2024-01-01', periods=250, freq='D'))
    mock_data = pd.DataFrame({
        'TEST_A': np.linspace(100, 200, 250) + np.random.randn(250) * 5,
        'TEST_B': np.linspace(50, 150, 250) + np.random.randn(250) * 3
    }, index=dates)
    mock_data.index.name = 'Date'

    # 1. Calculate returns and align features/targets (Horizon = 5 days)
    features, targets = calculate_returns(mock_data, prediction_horizon=5)
    
    features_with_indicators = add_technical_indicators(features)
    
    print("\n--- FEATURES WITH INDICATORS ---")
    print("Columns:", features_with_indicators.columns.tolist())
    print("\nShape after cleaning NaNs:", features_with_indicators.shape)
    print(features_with_indicators.tail(5)) 
    
    print(f"\nTotal Columns: {len(features_with_indicators.columns)}. Expected columns (2 original + 12 new) = 14.")