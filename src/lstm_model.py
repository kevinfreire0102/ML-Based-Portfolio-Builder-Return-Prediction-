import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class LSTMPredictor:
    """
    Implements a Long Short-Term Memory (LSTM) neural network for return prediction.
    Handles data scaling and reshaping into sequences required by LSTM layers.
    """
    def __init__(self, look_back: int = 20, features_count: int = 56):
        """
        Initializes the LSTM model parameters.
        """
        self.look_back = look_back
        self.features_count = features_count
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _create_sequences(self, data: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms 2D data (all features) into 3D sequences required by the LSTM layer.
        """
        X, Y = [], []
        # Target (Y) is the return of the FIRST stock (simplified approach for training)
        first_stock_target_index = 0 
        
        for i in range(len(data) - look_back - 1):
            a = data[i:(i + look_back), :] # Sequence of 'look_back' previous steps (e.g., 20 steps x 56 features)
            X.append(a)
            # Target is the value of the first stock's return at the next step
            Y.append(data[i + look_back, first_stock_target_index]) 
        return np.array(X), np.array(Y)

    def _build_model(self) -> Sequential:
        """
        Defines the lightweight LSTM architecture.
        """
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.look_back, self.features_count), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1)) 
        
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, X_features: pd.DataFrame, y_targets: pd.DataFrame, epochs: int = 1, batch_size: int = 1):
        """
        Scales ALL data, creates sequences, and trains the LSTM model.
        """
        # Use ALL 56 columns for scaling
        scaled_data = self.scaler.fit_transform(X_features.values)

        # Creating sequences (2D -> 3D reshape)
        X_seq, y_seq = self._create_sequences(scaled_data, self.look_back)

        print(f"Training LSTM model on {X_seq.shape[0]} sequences...")
        self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        print("LSTM model trained.")

    def predict(self, X_features: pd.DataFrame) -> pd.DataFrame:
        """
        Scales and reshapes current data, then generates a single step prediction.
        """
        scaled_data = self.scaler.transform(X_features.values)
        
        last_sequence_data = scaled_data[-self.look_back:] 
        
        last_sequence = last_sequence_data.reshape(1, self.look_back, self.features_count)

        scaled_prediction = self.model.predict(last_sequence, verbose=0)
        
        dummy_array = np.zeros((1, self.features_count))
        dummy_array[0, 0] = scaled_prediction[0, 0] 
        
        prediction = self.scaler.inverse_transform(dummy_array)[0, 0] 
        
        first_stock_ticker = X_features.columns[0]
        
        num_stocks = 7
        prediction_matrix = np.full((5, num_stocks), prediction) 
        
        ticker_columns = X_features.columns[:num_stocks]
        dates = X_features.index[-5:] 
        
        return pd.DataFrame(prediction_matrix, index=dates, columns=ticker_columns)


if __name__ == '__main__':
    print("LSTM model now expects multiple features. Rerun main.py to test.")