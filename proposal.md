# ML-Based Portfolio Builder — Predictive Allocation using Machine Learning

**Category:** Data Analysis & Visualization / Business & Finance Tools  

## Problem Statement
Financial markets generate massive amounts of data, yet most investors still rely on static portfolio models such as Markowitz’s mean-variance optimization.  
This project aims to explore whether **machine learning–based return prediction** can improve portfolio allocation decisions.  
The objective is to build a Python tool that predicts short-term stock returns, constructs an “ML-informed” portfolio, and compares its performance with a classical Markowitz baseline.  
The project combines financial data processing, feature engineering, model comparison, and quantitative evaluation — bridging finance and machine learning.

## Planned Approach and Technologies
- **Data Source & Universe:** Daily adjusted close prices for the **top 100 S&P 500 constituents by market capitalization**, retrieved via the **`yfinance` API**, covering **January 2015 to December 2024**.  
- **Feature Engineering:** Computation of technical indicators such as lagged returns, momentum, rolling volatility, Relative Strength Index (RSI), moving averages, and trading volume.  
- **Prediction Target:** Forecast **next-week (5-day ahead) returns** for each stock based on historical patterns.  
- **Machine Learning Models:** Train and compare multiple algorithms — **Random Forest**, **XGBoost**, and a **lightweight LSTM** to capture temporal dependencies. The LSTM will be a compact extension, ensuring feasibility while demonstrating the integration of deep learning.  
- **Temporal Validation:** Apply a **rolling window** scheme — e.g., train on 2015–2019, test on 2020; then train on 2016–2020, test on 2021, etc. — to preserve chronological order and prevent data leakage.  
- **Portfolio Construction:** At each test period, **long the top 10% of stocks ranked by predicted return**, weighting them **proportionally to predicted values**. Portfolios will be **rebalanced weekly**, consistent with the prediction horizon.  
- **Evaluation:**  
  - *Model-level metrics:* MAE, RMSE, and directional accuracy (percentage of correctly predicted return signs).  
  - *Portfolio-level metrics:* cumulative return, Sharpe ratio, and maximum drawdown.  
  This dual evaluation isolates predictive quality from investment performance.

## Expected Challenges
- Managing missing or inconsistent financial data from APIs.  
- Avoiding overfitting while ensuring temporal robustness.  
- Efficiently computing and validating models across multiple stocks.  
- Maintaining readability and modularity of the codebase.  

## Success Criteria
- Functional Python pipeline capable of training, predicting, and backtesting over 10 years of data.  
- Clear visualizations of prediction performance and portfolio results.  
- Documentation and testing following professional standards (PEP8, ≥70% test coverage).  

## Stretch Goals
- Introduce transaction cost simulation and turnover analysis.  
- Extend the LSTM module with multi-feature inputs (returns + technical indicators).  
- Build a small Streamlit dashboard to visualize portfolio evolution interactively.  
