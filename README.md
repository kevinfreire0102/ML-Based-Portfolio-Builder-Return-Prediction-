# ML-Based Portfolio Builder: Predictive Allocation using Rolling Window

## 1. Project Overview

This project aims to demonstrate the effectiveness of Machine Learning (ML) models in financial portfolio construction. We predict next-week (5-day) stock returns for selected S\&P 500 constituents based on historical technical indicators.

The final results show that the **Random Forest Regressor** provides the most optimal performance, generating over 4600% cumulative return with the best Sharpe Ratio (2.335) over the 2015-2024 backtesting period.

## 2. Methodology & Key Features

* **Prediction Target:** Next-week (5-day ahead) return for each stock.
* **Data Source:** Daily Adjusted Close Prices for 7 sample S\&P 500 tickers (2015-2024) via the `yfinance` API.
* **Models:** Comparison of **Random Forest**, **XGBoost**, and **LSTM** (Deep Learning) Regressors.
* **Validation:** **Rolling Window / Walk-Forward Analysis**  (500-day training window, 5-day prediction step) to ensure chronological consistency and avoid data leakage.
* **Portfolio Strategy:** Long the **top 10%** of stocks ranked by predicted positive return, weighted proportionally to the prediction value.
* **Baseline:** **Markowitz Mean-Variance Optimization** is used as a non-ML baseline for comparison.

## 3. Setup and Reproducibility

The project requires Python 3.10+ and uses a modular structure.

### A. Environment Setup

1.  **Install Dependencies:** Ensure your environment is active, then run the installation command:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install Project Module:** Install the local package (`src/`) to ensure all imports work correctly:
    ```bash
    pip install -e .
    ```

### B. Execution

To run the entire pipeline (data acquisition, feature engineering, model training, backtesting, and result generation), use the main script:

```bash
python main.py