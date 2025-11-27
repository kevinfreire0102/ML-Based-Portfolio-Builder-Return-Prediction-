import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Any, Dict

class BaseMLModel:
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        
        self.model = None
        self.model_type = model_type.lower()
        self.kwargs = kwargs

        if self.model_type == 'random_forest':
            # We use a Regressor because the target is a numerical return value
            self.model = RandomForestRegressor(random_state=42, n_estimators=100, **kwargs)
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(random_state=42, n_estimators=100, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Must be 'random_forest' or 'xgboost'.")

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        
        y_flat = y.values.flatten()
        
        X_stacked = pd.concat([X] * y.shape[1], ignore_index=True)
        
        valid_indices = ~np.isnan(y_flat)
        y_flat = y_flat[valid_indices]
        X_stacked = X_stacked.iloc[valid_indices]
        
        print(f"Training {self.model_type} model on {len(X_stacked)} samples...")
        self.model.fit(X_stacked, y_flat)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")
            
        # The prediction logic must match the training logic: 
        # we predict the return for all stocks based on the stacked features.
        
        # We assume X has features for multiple stocks, but the model predicts for all of them.
        # For prediction, we replicate the features (X) for the number of stocks
        X_stacked = pd.concat([X] * self.model.n_outputs_, ignore_index=True)
        
        predictions_flat = self.model.predict(X_stacked)
        
        num_stocks = self.model.n_outputs_
        num_rows = len(X)
        predictions_matrix = predictions_flat.reshape(num_rows, num_stocks)
        
        # Create DataFrame with the correct index (dates) and columns (tickers)
        # We use the original columns from the input data (X)
        predictions_df = pd.DataFrame(predictions_matrix, index=X.index, columns=X.columns)
        
        return predictions_df
        
if __name__ == '__main__':
    # 1. Setup Mock Data
    print("--- 1. Initial Mock Data ---")
    
    # Features (X)
    mock_X = pd.DataFrame({
        'price': [100, 101, 102],
        'volatility': [0.1, 0.2, 0.3]
    }, index=['2024-01-01', '2024-01-02', '2024-01-03'])
    mock_X.index.name = 'Date'
    
    # Targets (Y) - Future returns for 3 stocks (A, B, C)
    mock_y = pd.DataFrame({
        'A': [0.01, 0.02, 0.03],
        'B': [-0.01, -0.02, -0.03],
        'C': [0.00, 0.00, 0.00]
    }, index=['2024-01-01', '2024-01-02', '2024-01-03'])
    
    print("Mock Features (X - 3 rows, 2 columns):")
    print(mock_X)
    print("\nMock Targets (Y - 3 rows, 3 stocks):")
    print(mock_y)

    # 2. Replication of Stacking Logic (as done inside the train method)
    
    # Flatten Targets (y_flat)
    y_flat = mock_y.values.flatten()
    
    # Stack Features (X_stacked)
    # y.shape[1] is 3 (number of stocks A, B, C)
    X_stacked = pd.concat([mock_X] * mock_y.shape[1], ignore_index=True)
    
    print("\n--- 2. Formatting Result (Flattening & Stacking) ---")
    
    # Print y_flat
    print(f"Flattened Target (y_flat) - Shape: {y_flat.shape}. (3 rows * 3 stocks = 9 elements)")
    print("Content of y_flat (Returns for A, then B, then C):")
    print(y_flat)
    
    # Print X_stacked
    print(f"\nStacked Features (X_stacked) - Shape: {X_stacked.shape}. (3 rows * 3 stocks = 9 rows, 2 columns)")
    print("Content of X_stacked (The same features, stacked 3 times):")
    print(X_stacked)
    
    # 3. Conclusion
    print("\nCONCLUSION:")
    print(f"The goal is to have {len(X_stacked)} feature rows to match the {len(y_flat)} targets.")
    print("This allows one unified model to train on all stock examples.")