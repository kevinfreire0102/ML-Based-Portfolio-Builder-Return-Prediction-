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
    # 1. Calculate future returns (the target variable)
    # The .shift(-horizon) aligns the FUTURE return with the CURRENT date.
    targets_df = data.pct_change(prediction_horizon).shift(-prediction_horizon)

    # 2. Prepare features (current prices)
    features_df = data.copy() 
    
    # 3. Clean up by removing the last rows which contain NaNs (because we can't predict further)
    # We must remove NaNs from both features and targets at the same time
    valid_index = targets_df.dropna().index
    features_df = features_df.loc[valid_index]
    targets_df = targets_df.loc[valid_index]
    
    return features_df, targets_df

# --- Local Testing Block ---
if __name__ == "__main__":
    # Create a mock data structure similar to data_loader.py output
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'])
    mock_data = pd.DataFrame({
        'AAPL': [100, 101, 102, 103, 104, 105],
        'MSFT': [200, 202, 204, 206, 208, 210]
    }, index=dates)

    features, targets = calculate_returns(mock_data, prediction_horizon=3)
    
    print("\n--- TARGETS (Future 3-Day Returns) ---")
    print(targets)
    print("\n--- FEATURES (Input Data) ---")
    print(features)