import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any 
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import warnings

# --- 1. FINANCIAL METRICS & PORTFOLIO ALLOCATION ---

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    """
    # Assuming 252 trading days per year
    annualized_returns = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    
    if annualized_volatility == 0:
        return 0.0 # Avoid division by zero
        
    return (annualized_returns - risk_free_rate) / annualized_volatility

def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculates the drawdown series from portfolio returns.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown

def evaluate_predictions(predictions: pd.DataFrame, targets: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Calculates ML-level metrics (MAE, Directional Accuracy, Confusion Matrix).
    """
    # Align predictions with targets for comparison
    common_index = predictions.index.intersection(targets.index)
    pred = predictions.loc[common_index]
    targ = targets.loc[common_index]

    # Flatten data for metric calculation
    pred_flat = pred.values.flatten()
    targ_flat = targ.values.flatten()

    # 1. Regression Metric: Mean Absolute Error (MAE)
    mae = mean_squared_error(targ_flat, pred_flat)
    
    # 2. Classification Metric: Directional Accuracy (DA)
    # Target and prediction are positive (Up) or negative (Down)
    targ_direction = np.sign(targ_flat)
    pred_direction = np.sign(pred_flat)
    
    # Ignore zero returns when calculating accuracy
    non_zero_mask = (targ_direction != 0)
    
    # Directional Accuracy (DA)
    da = accuracy_score(targ_direction[non_zero_mask], pred_direction[non_zero_mask])

    # 3. Confusion Matrix (For Visualization)
    cm = confusion_matrix(targ_direction[non_zero_mask], pred_direction[non_zero_mask])

    return {
        'model': model_name,
        'MAE': mae,
        'Directional Accuracy': da,
        'Confusion Matrix': cm.tolist() # Convert to list for easy storage/output
    }

# --- 2. MARKOWITZ BASELINE ---

def calculate_markowitz_weights(returns: pd.DataFrame, target_return: float = 0.0005) -> np.ndarray:
    """
    Calculates optimal weights using Markowitz Mean-Variance Optimization.
    """
    if returns.empty:
        return np.zeros(len(returns.columns))

    num_assets = len(returns.columns)
    
    # Objective function: Minimize volatility (standard deviation)
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
        return np.sqrt(portfolio_variance) * np.sqrt(252) # Annualized volatility

    # Constraint 1: Weights sum to 1 (full allocation)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Constraint 2: Weights are non-negative (no short-selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess for weights (equal allocation)
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    # Suppress warnings during optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Run the optimization to find the minimum variance portfolio
        optimized_results = minimize(
            objective, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

    # Return normalized weights
    if optimized_results.success:
        weights = optimized_results.x / np.sum(optimized_results.x)
        return weights
    else:
        # Fallback to equal weighting if optimization fails
        return initial_weights

# --- 3. MAIN EVALUATION ORCHESTRATOR ---

def evaluate_portfolio(predictions: Dict[str, pd.DataFrame], targets: pd.DataFrame, raw_prices: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs the portfolio construction and calculates all final metrics for all models and Markowitz.
    """
    results = {}
    
    # Ensure targets are aligned by only looking at the dates for which we have predictions
    common_index = predictions['Random Forest'].index.intersection(targets.index)
    targets = targets.loc[common_index]
    
    # --- Markowitz Baseline ---
    markowitz_returns = pd.Series(index=common_index, dtype=float)
    
    # The Markowitz portfolio needs the actual returns *before* the prediction date
    markowitz_lookback = 500
    
    # Markowitz calculation needs to be done iteratively over the prediction period
    for start_date in common_index:
        # Find the end of the Markowitz training period
        end_markowitz = start_date
        # Approximate 500 trading days for lookback
        start_markowitz = end_markowitz - pd.Timedelta(days=markowitz_lookback * 1.5) 
        
        # Get historical returns for Markowitz calculation
        historical_returns = raw_prices.loc[start_markowitz:end_markowitz].pct_change().dropna()
        
        # Calculate Markowitz weights
        weights = calculate_markowitz_weights(historical_returns)
        
        # Calculate the return for the prediction period (5 days) using these static weights
        target_period_returns = targets.loc[start_date]
        markowitz_returns[start_date] = np.sum(target_period_returns * weights)

    
    results['Markowitz'] = {
        'Sharpe Ratio': calculate_sharpe_ratio(markowitz_returns),
        'Max Drawdown': calculate_drawdown(markowitz_returns).min(),
        'Cumulative Return': (1 + markowitz_returns).prod() - 1,
        'Returns Series': markowitz_returns, # <-- FIX: Enregistre la SÉRIE Pandas (avec index)
    }
    
    # --- ML Models Evaluation ---
    for model_name, pred_df in predictions.items():
        
        # Portfolio Construction: Long Top 10% of stocks weighted by predicted return
        
        portfolio_returns = pd.Series(index=common_index, dtype=float)
        
        for date in common_index:
            # Get predictions for this date
            daily_pred = pred_df.loc[date].dropna()
            
            # Select top 10% of predictions (must be long positions, i.e., positive predictions)
            positive_pred = daily_pred[daily_pred > 0]
            
            # Select the top 10%
            num_to_select = max(1, int(len(positive_pred) * 0.10))
            
            # Sort and select (weights proportional to predicted values)
            sorted_pred = positive_pred.sort_values(ascending=False)
            selected_pred = sorted_pred.head(num_to_select)
            
            if selected_pred.empty:
                # If no positive predictions, assume zero return
                portfolio_returns[date] = 0.0
                continue
                
            # Weights: proportional to the predicted return
            weights = selected_pred / selected_pred.sum()
            
            # Get actual returns for the prediction period (5 days later)
            actual_returns = targets.loc[date][weights.index]
            
            # Calculate daily portfolio return for this period
            portfolio_returns[date] = np.sum(actual_returns * weights)
            
        
        # 1. Calculate Portfolio Metrics
        results[model_name] = {
            'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns),
            'Max Drawdown': calculate_drawdown(portfolio_returns).min(),
            'Cumulative Return': (1 + portfolio_returns).prod() - 1,
            'Returns Series': portfolio_returns, # <-- FIX: Enregistre la SÉRIE Pandas (avec index)
        }
        
        # 2. Calculate ML Metrics (MAE, DA, CM)
        ml_metrics = evaluate_predictions(pred_df, targets, model_name)
        results[model_name].update(ml_metrics)

    return results

# --- Local Testing Block ---
if __name__ == '__main__':
    print("Evaluation module requires full pipeline data to test.")