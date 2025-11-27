import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Dict
from src.models import BaseMLModel 
from src.lstm_model import LSTMPredictor 

class RollingWindowBacktester:
    
    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame, 
                 train_window: int = 500, predict_steps: int = 20):
        """
        Initializes the backtester with prepared data and window sizes.
        
        """
        self.features = features
        self.targets = targets
        self.train_window = train_window
        self.predict_steps = predict_steps
        self.all_predictions = {}
        
    def _split_data(self, start_index: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets based on the rolling window index.
        """
        train_end_index = start_index + self.train_window
        
        X_train = self.features.iloc[start_index : train_end_index]
        y_train = self.targets.iloc[start_index : train_end_index]
        
        X_test = self.features.iloc[train_end_index : train_end_index + self.predict_steps]
        y_test = self.targets.iloc[train_end_index : train_end_index + self.predict_steps]
        
        return X_train, y_train, X_test, y_test

    def run_backtest(self, model_instance: Any, model_name: str) -> pd.DataFrame:
        """
        Runs the full walk-forward backtest for a given model instance.
        """
        model_predictions = {}
        
        start_prediction_index = self.train_window
        
        for i in range(start_prediction_index, len(self.features) - self.predict_steps, self.predict_steps):
            
            window_start = i - self.train_window
            
            X_train, y_train, X_test, y_test = self._split_data(window_start)
            
            if X_test.empty:
                break
            
            print(f"Training {model_name} on data ending {X_train.index[-1].strftime('%Y-%m-%d')}")
            
            try:
                
                if isinstance(model_instance, LSTMPredictor):
                    model_instance.train(X_train, y_train, epochs=1) 
                else:
                    model_instance.train(X_train, y_train)
                    
            except Exception as e:
                print(f"Error during training {model_name} at index {i}: {e}")
                continue
            
            y_pred = model_instance.predict(X_test)
            
            model_predictions.update(y_pred.to_dict(orient='index'))

        predictions_df = pd.DataFrame.from_dict(model_predictions, orient='index')
        predictions_df.index = pd.to_datetime(predictions_df.index)
        predictions_df.index.name = 'Date'
        
        self.all_predictions[model_name] = predictions_df
        
        return predictions_df
        
if __name__ == '__main__':
    print("Backtester setup complete. Integration with full data and models will be tested later.")