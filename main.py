import pandas as pd
import numpy as np
import warnings
import os
from typing import List

# Import all modules from the 'src' directory
from src.data_loader import download_sp500_data
from src.features import calculate_returns, add_technical_indicators
from src.models import BaseMLModel # Importe BaseMLModel qui gère RF, XGBoost et LightGBM
# from src.lstm_model import LSTMPredictor # LSTM DÉSACTIVÉ pour la stabilité
from src.backtester import RollingWindowBacktester
from src.evaluation import evaluate_portfolio, calculate_markowitz_weights
from src.visualization import create_cumulative_returns_plot, create_drawdown_plot, create_confusion_matrix_plot 

# Configuration
RANDOM_STATE = 42 
np.random.seed(RANDOM_STATE) 

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'NVDA']
START_DATE = '2015-01-01'
END_DATE = '2024-12-31' # Date de soumission complète

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
    lgbm_model = BaseMLModel(model_type='lightgbm') # LightGBM remplace l'LSTM
    
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
    lgbm_predictions = backtester.run_backtest(lgbm_model, "LightGBM") # Exécuter LightGBM
    
    # Store predictions for evaluation
    all_predictions = {
        "Random Forest": rf_predictions,
        "XGBoost": xgb_predictions,
        "LightGBM": lgbm_predictions # Stocker les prédictions du LightGBM
    }

    print("\nBacktesting completed for all models.")
    
    # 4. Evaluation, Reporting, and Visualization (Phase 4.2 & 5)
    print("\n" + "="*60)
    print("3. FINAL EVALUATION AND REPORTING")
    print("="*60)
    
    # A. Evaluation
    evaluation_results = evaluate_portfolio(all_predictions, targets, raw_prices) 
    
    # B. Display and Save Numerical Results
    print("\nFINAL RESULTS (Comparison of all strategies):")
    print("-" * 35)
    
    results_str = ""
    for model, res in evaluation_results.items():
        print(f"| {model:<15} | Sharpe: {res['Sharpe Ratio']:.3f} | Cum Return: {res['Cumulative Return']:.3f}")
        results_str += f"| {model:<15} | Sharpe: {res['Sharpe Ratio']:.3f} | Cum Return: {res['Cumulative Return']:.3f}\n"
        if 'Directional Accuracy' in res:
             print(f"| {'':<15} | MAE: {res['MAE']:.5f} | DA: {res['Directional Accuracy']:.3f}")
             results_str += f"| {'':<15} | MAE: {res['MAE']:.5f} | DA: {res['Directional Accuracy']:.3f}\n"
    
    print("-" * 35)

    # Save results summary to results/
    OUTPUT_FILE = 'results/final_results_summary.txt'
    os.makedirs('results', exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("PROJECT RESULTS SUMMARY (2015-2024)\n\n")
        f.write(results_str)
        
    print(f"\nNumerical results saved to: {OUTPUT_FILE}")

    # C. Generate and Save Plots
    print("\n" + "="*60)
    print("4. GENERATING FINAL PLOTS AND VISUALIZATIONS")
    print("="*60)
    
    # Cumulative Returns Plot (Graph 1)
    create_cumulative_returns_plot(evaluation_results)
    
    # Drawdown Plot (Graph 2)
    create_drawdown_plot(evaluation_results)
    
    # Confusion Matrix for the three successful ML models (Graph 3, 4, 5)
    create_confusion_matrix_plot(evaluation_results, 'Random Forest', file_name='cm_random_forest.png')
    create_confusion_matrix_plot(evaluation_results, 'XGBoost', file_name='cm_xgboost.png')
    create_confusion_matrix_plot(evaluation_results, 'LightGBM', file_name='cm_lightgbm.png') # Nouveau graphique
    
    print("\nAll visualizations saved to the 'results/plots' directory.")

if __name__ == "__main__":
    run_project()