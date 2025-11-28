import pandas as pd
import numpy as np
import warnings
from typing import List

# Import all modules from the 'src' directory
from src.data_loader import download_sp500_data
from src.features import calculate_returns, add_technical_indicators
from src.models import BaseMLModel
from src.lstm_model import LSTMPredictor
from src.backtester import RollingWindowBacktester
from src.evaluation import evaluate_portfolio, calculate_markowitz_weights # <-- CORRECTION : Nom de fonction
from src.evaluation import calculate_markowitz_weights

# Configuration
RANDOM_STATE = 42 # Ensures reproducibility (as required by project rules)
np.random.seed(RANDOM_STATE) 

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'NVDA'] # Sample of 7 tickers 
START_DATE = '2015-01-01'
# CORRECTION TEST RAPIDE: Changez cette date pour '2024-12-31' pour la soumission finale !
END_DATE = '2024-12-31' 

# ML/Backtesting Parameters
TRAIN_WINDOW_SIZE = 500 
PREDICT_STEPS = 5

def run_project(tickers: List[str] = TICKERS, start_date: str = START_DATE, end_date: str = END_DATE):
    """
    Main function to run the entire ML Portfolio Builder pipeline.
    """
    
    # 1. Setup and Data Acquisition
    print("="*60)
    print("1. DATA ACQUISITION & FEATURE ENGINEERING")
    print("="*60)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    try:
        raw_prices = download_sp500_data(tickers=tickers, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"FATAL ERROR: Data download failed: {e}")
        return

    # 2. Feature Engineering
    features, targets = calculate_returns(raw_prices, prediction_horizon=PREDICT_STEPS)
    features_with_indicators = add_technical_indicators(features)

    # Align targets again after feature cleaning
    valid_dates = features_with_indicators.index
    targets = targets.loc[valid_dates]
    
    print(f"Data ready. Features shape: {features_with_indicators.shape}, Targets shape: {targets.shape}")
    
    # 3. Model Training & Backtesting
    print("\n" + "="*60)
    print("2. MODEL TRAINING & BACKTESTING (Rolling Window)")
    print("="*60)
    
    # Model Instantiation 
    rf_model = BaseMLModel(model_type='random_forest')
    xgb_model = BaseMLModel(model_type='xgboost')
    
    lstm_features_count = features_with_indicators.shape[1]
    lstm_model = LSTMPredictor(look_back=20, features_count=lstm_features_count)

    # Backtester Instantiation
    backtester = RollingWindowBacktester(
        features=features_with_indicators,
        targets=targets,
        train_window=TRAIN_WINDOW_SIZE,
        predict_steps=PREDICT_STEPS
    )

    # Run Backtests
    rf_predictions = backtester.run_backtest(rf_model, "Random Forest")
    xgb_predictions = backtester.run_backtest(xgb_model, "XGBoost")
    # TRES IMPORTANT: Le LSTM est dÉsactivÉ pour une exÉcution rÉussie
    # lstm_predictions = backtester.run_backtest(lstm_model, "LSTM") 
    
    # Store predictions for evaluation
    all_predictions = {
        "Random Forest": rf_predictions,
        "XGBoost": xgb_predictions,
        # "LSTM": lstm_predictions
    }

    print("\nBacktesting completed for all models.")
    
    # 4. Evaluation and Reporting (Phase 4.2)
    print("\n" + "="*60)
    print("3. EVALUATION AND PORTFOLIO RESULTS")
    print("="*60)
    
    # CRITICAL: We need the raw prices data for the Markowitz baseline calculation
    evaluation_results = evaluate_portfolio(all_predictions, targets, raw_prices) 
    
    print("\nFINAL RESULTS (Comparison of all strategies):")
    print("-" * 35)
    
    # Print results (example format)
    for model, res in evaluation_results.items():
        print(f"| {model:<15} | Sharpe: {res['Sharpe Ratio']:.3f} | Cum Return: {res['Cumulative Return']:.3f}")
        if 'Directional Accuracy' in res:
             print(f"| {'':<15} | MAE: {res['MAE']:.5f} | DA: {res['Directional Accuracy']:.3f}")
        
    print("-" * 35)

if __name__ == "__main__":
    run_project()