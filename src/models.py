import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Any, Dict

class BaseMLModel:
    """
    Base class for Machine Learning models used for return prediction.
    Encapsulates model training and prediction logic.
    """
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initializes the model instance based on the specified type.
        """
        self.model = None
        self.model_type = model_type.lower()
        self.kwargs = kwargs

        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=42, n_estimators=100, **kwargs)
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(random_state=42, n_estimators=100, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Must be 'random_forest' or 'xgboost'.")

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Trains the underlying model.
        """
        # Flatten the targets (future returns) into a 1D array
        y_flat = y.values.flatten()
        
        # Stack the features multiple times (once per stock) to match the target length
        X_stacked = pd.concat([X] * y.shape[1], ignore_index=True)
        
        # Clean up NaNs which might appear during stacking/flattening
        valid_indices = ~np.isnan(y_flat)
        y_flat = y_flat[valid_indices]
        X_stacked = X_stacked.iloc[valid_indices]
        
        # Fit the model
        print(f"Training {self.model_type} model on {len(X_stacked)} samples...")
        self.model.fit(X_stacked, y_flat)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the given features.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")
            
        num_stocks = 7 
        
        X_stacked = pd.concat([X] * num_stocks, ignore_index=True)
        
        predictions_flat = self.model.predict(X_stacked)
        
        num_rows = len(X) 
        
        predictions_matrix = predictions_flat.reshape(num_rows, num_stocks)
        
        ticker_columns = X.columns[:num_stocks] 
        
        predictions_df = pd.DataFrame(predictions_matrix, index=X.index, columns=ticker_columns)
        
        return predictions_df
        
if __name__ == '__main__':
    print("Base model training setup complete. We will integrate with feature data next.")