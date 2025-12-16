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
        return 0.0 
        
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
    targ_direction = np.sign(targ_flat)
    pred_direction = np.sign(pred_flat)
    
    non_zero_mask = (targ_direction != 0)
    
    da = accuracy_score(targ_direction[non_zero_mask], pred_direction[non_zero_mask])

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
    
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
        return np.sqrt(portfolio_variance) * np.sqrt(252) 

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        optimized_results = minimize(
            objective, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

    if optimized_results.success:
        weights = optimized_results.x / np.sum(optimized_results.x)
        return weights
    else:
        return initial_weights


def evaluate_portfolio(predictions: Dict[str, pd.DataFrame], targets: pd.DataFrame, raw_prices: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs the portfolio construction and calculates all final metrics for all models and Markowitz.
    """
    results = {}
    
    common_index = predictions['Random Forest'].index.intersection(targets.index)
    targets = targets.loc[common_index]
    
    markowitz_returns = pd.Series(index=common_index, dtype=float)
    
    markowitz_lookback = 500
    
    for start_date in common_index:
        
        end_markowitz = start_date
        
        start_markowitz = end_markowitz - pd.Timedelta(days=markowitz_lookback * 1.5) 
        
        historical_returns = raw_prices.loc[start_markowitz:end_markowitz].pct_change().dropna()
        
        weights = calculate_markowitz_weights(historical_returns)
        
        target_period_returns = targets.loc[start_date]
        markowitz_returns[start_date] = np.sum(target_period_returns * weights)

    
    results['Markowitz'] = {
        'Sharpe Ratio': calculate_sharpe_ratio(markowitz_returns),
        'Max Drawdown': calculate_drawdown(markowitz_returns).min(),
        'Cumulative Return': (1 + markowitz_returns).prod() - 1,
        'Returns Series': markowitz_returns, 
    }
    
    for model_name, pred_df in predictions.items():
        
        
        portfolio_returns = pd.Series(index=common_index, dtype=float)
        
        for date in common_index:
            daily_pred = pred_df.loc[date].dropna()
            
            positive_pred = daily_pred[daily_pred > 0]
            
            num_to_select = max(1, int(len(positive_pred) * 0.10))
            
            sorted_pred = positive_pred.sort_values(ascending=False)
            selected_pred = sorted_pred.head(num_to_select)
            
            if selected_pred.empty:
                portfolio_returns[date] = 0.0
                continue
                
            weights = selected_pred / selected_pred.sum()
            
            actual_returns = targets.loc[date][weights.index]
            
            portfolio_returns[date] = np.sum(actual_returns * weights)
            
        
        # 1. Calculate Portfolio Metrics
        results[model_name] = {
            'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns),
            'Max Drawdown': calculate_drawdown(portfolio_returns).min(),
            'Cumulative Return': (1 + portfolio_returns).prod() - 1,
            'Returns Series': portfolio_returns, 
        }
        
        # 2. Calculate ML Metrics (MAE, DA, CM)
        ml_metrics = evaluate_predictions(pred_df, targets, model_name)
        results[model_name].update(ml_metrics)

    return results

if __name__ == '__main__':
    print("Evaluation module requires full pipeline data to test.")