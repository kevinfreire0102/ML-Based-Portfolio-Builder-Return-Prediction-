import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class LSTMPredictor:
    
    def __init__(self, look_back: int = 20, features_count: int = 1):
        
        self.look_back = look_back
        self.features_count = features_count
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _create_sequences(self, data: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms 2D data into 3D sequences required by the LSTM layer.
       
        """
        X, Y = [], []
        # We process the data sequentially to create overlapping sequences
        for i in range(len(data) - look_back - 1):
            a = data[i:(i + look_back), :] # Sequence of 'look_back' previous steps
            X.append(a)
            Y.append(data[i + look_back, 0]) # Target is the feature 0 (price/return) at the next step
        return np.array(X), np.array(Y)

    def _build_model(self) -> Sequential:
        """
        Defines the lightweight LSTM architecture.
        """
        model = Sequential()
        # Input layer requires the shape: (sequence_length, features_count)
        model.add(LSTM(50, input_shape=(self.look_back, self.features_count), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1)) # Output is a single value (the predicted return)
        
        # Compile the model with Adam optimizer and Mean Squared Error loss
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, X_features: pd.DataFrame, y_targets: pd.DataFrame, epochs: int = 25, batch_size: int = 1):
        """
        Scales data, creates sequences, and trains the LSTM model.
        """
        # --- Preprocessing Steps ---
        # 1. Scaling the data (essential for neural networks)
        # We simplify by using only the first stock's feature set for scaling
        stock_data = X_features.iloc[:, 0].values.reshape(-1, 1) # Select only the first column/stock for scaling
        scaled_data = self.scaler.fit_transform(stock_data)

        # 2. Creating sequences (2D -> 3D reshape)
        X_seq, y_seq = self._create_sequences(scaled_data, self.look_back)

        # --- Training ---
        print(f"Training LSTM model on {X_seq.shape[0]} sequences...")
        self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        print("LSTM model trained.")

    def predict(self, X_features: pd.DataFrame) -> pd.DataFrame:
        """
        Scales and reshapes current data, then generates a single step prediction.
        """
        # 1. Scale and Reshape for prediction
        stock_data = X_features.iloc[:, 0].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(stock_data)
        
        # Get the last sequence to predict the next step
        last_sequence = scaled_data[-self.look_back:].reshape(1, self.look_back, self.features_count)

        # 2. Predict and inverse transform
        scaled_prediction = self.model.predict(last_sequence, verbose=0)
        prediction = self.scaler.inverse_transform(scaled_prediction)
        
        # Return a mock DataFrame for integration
        return pd.DataFrame(prediction, index=[X_features.index[-1]], columns=['Predicted_Return'])

# --- Local Testing Block ---
if __name__ == '__main__':
    # Create mock feature data (200 rows of features for one stock)
    dates = pd.to_datetime(pd.date_range('2024-01-01', periods=200, freq='B'))
    mock_features = pd.DataFrame({'Feature_1': np.random.rand(200)}, index=dates)
    mock_targets = pd.DataFrame({'Target_A': np.random.rand(200)}, index=dates)
    
    # Initialize and Test
    lstm_model = LSTMPredictor(look_back=10, features_count=1)
    lstm_model.train(mock_features, mock_targets, epochs=1) # Train briefly
    
    # Simple prediction check
    prediction_df = lstm_model.predict(mock_features)
    print("\nLSTM Model Initialization and Training Check Passed.")
    print("Example Prediction:")
    print(prediction_df)