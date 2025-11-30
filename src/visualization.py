import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import os

OUTPUT_DIR = 'results/plots'

def create_cumulative_returns_plot(results: Dict[str, Any], file_name: str = 'cumulative_returns.png'):
    """
    Generates and saves a plot comparing the cumulative returns of all strategies.
    (FIX: Ensures the Date Index is used for the X-axis)
    """
    plt.figure(figsize=(12, 7))
    
    for model_name, data in results.items():
        returns_series = pd.Series(data['Returns Series'])
        cumulative_returns = (1 + returns_series).cumprod()
        
        plt.plot(cumulative_returns.index, cumulative_returns.values, label=model_name)
    
    plt.title('Cumulative Portfolio Returns Comparison (2015-2024)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved: {save_path}")

def create_drawdown_plot(results: Dict[str, Any], file_name: str = 'max_drawdown.png'):
    """
    Generates and saves a plot comparing the Maximum Drawdown for all strategies.
    (FIX: Ensures the Date Index is used for the X-axis)
    """
    plt.figure(figsize=(12, 7))
    
    for model_name, data in results.items():
        returns_series = pd.Series(data['Returns Series'])
        
        # Calculate Drawdown: Max peak - current value
        cumulative_returns = (1 + returns_series).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        
        # Plotting - Use index and values explicitly to force date display on X-axis
        plt.plot(drawdown.index, drawdown.values, label=model_name) 
    
    plt.title('Maximum Drawdown Comparison (Risk Assessment)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved: {save_path}")

def create_confusion_matrix_plot(results: Dict[str, Any], model_name: str, file_name: str):
    """
    Generates and saves a plot of the Confusion Matrix for a single ML model.
    """
    if 'Confusion Matrix' not in results.get(model_name, {}):
        print(f"Error: Confusion Matrix data not found for {model_name}.")
        return

    cm_list = results[model_name]['Confusion Matrix']
    if not cm_list:
        print(f"Skipping Confusion Matrix for {model_name}: Data is empty.")
        return
        
    cm_data = np.array(cm_list)
    
    if cm_data.shape != (2, 2):
        print(f"Error: Confusion Matrix shape is {cm_data.shape}. Expected (2, 2). Skipping.")
        return

    cm_df = pd.DataFrame(cm_data, 
                         index=['Actual Down (-1)', 'Actual Up (+1)'], 
                         columns=['Predicted Down (-1)', 'Predicted Up (+1)'])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(f'Confusion Matrix for {model_name} (Directional Accuracy)', fontsize=14)
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == '__main__':
    print("Visualization functions are ready to be called from main.py.")