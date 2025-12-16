# ML-Based Portfolio Builder: Predictive Allocation

## Research Question
Which machine learning model performs best for predicting 5-day stock returns and building an optimal portfolio: Random Forest, XGBoost, or LightGBM?

## Setup

# Create environment
pip install -r requirements.txt

# Install local source package
pip install -e .

## Usage

python main.py

Expected output: Sharpe Ratio and Cumulative Return comparison between three ML models and a Markowitz baseline.

## Project Structure
ml-portfolio-project/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── data_loader.py   # Data loading/preprocessing
│   ├── features.py       # Technical indicators engineering
│   ├── models.py         # Model definitions (RF, XGB, LGBM)
│   ├── backtester.py     # Rolling window logic
│   └── evaluation.py     # Portfolio metrics calculation
├── data/
│   └── raw/             # Original stock price data
├── results/             # Output plots and metrics
└── requirements.txt     # Pip dependencies

## Results
- Random Forest: 2.294 Sharpe Ratio
- XGBoost: 2.077 Sharpe Ratio
- LightGBM: 2.139 Sharpe Ratio
- Markowitz (Baseline): 1.927 Sharpe Ratio

Winner: Random Forest (3844.248% Cumulative Return)

## Requirements
- Python 3.11
- scikit-learn, pandas, xgboost, lightgbm, yfinance, matplotlib