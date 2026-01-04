# ML-Based Portfolio Builder: Predictive Allocation

## 1. Research Question
**"Which machine learning model performs best for predicting 5-day stock returns and building an optimal portfolio: Random Forest, XGBoost, or LightGBM?"**

## 2. Setup & Installation

### Create environment
It is recommended to use a virtual environment.

    pip install -r requirements.txt

### Install local source package
This installs the project in editable mode, which is required for internal imports to work correctly.

    pip install -e .

## 3. Usage

Run the main pipeline to download data, train models, and generate results:

    python main.py

**Expected output:** Sharpe Ratio and Cumulative Return comparison between three ML models and a Markowitz baseline.

## 4. Project Structure
The project follows a modular data science structure:

    ml-portfolio-project/
    ├── main.py              # Main entry point
    ├── src/                 # Source code
    │   ├── data_loader.py   # Data loading/preprocessing
    │   ├── features.py      # Technical indicators engineering
    │   ├── models.py        # Model definitions (RF, XGB, LGBM)
    │   ├── backtester.py    # Rolling window logic
    │   └── evaluation.py    # Portfolio metrics calculation
    ├── tests/               # Unit tests
    │   ├── test_data_loader.py
    │   ├── test_features.py
    ├── data/
    │   └── raw/             # Original stock price data
    ├── results/             # Output plots and metrics
    │   └── plots/           # Generated graphs (PNG)
    ├── proposal.md
    ├── README.md
    ├── setup.py
    ├── AI_usage.md
    └── requirements.txt     # Pip dependencies

## 5. Results & Findings

I compared four approaches over a 10-year rolling window (2015-2024), focusing on the **Sharpe Ratio** (risk-adjusted return) and **Cumulative Return**.

* **Random Forest** was the absolute best performer, achieving a Sharpe Ratio of **2.294** and generating a massive cumulative return of **3844%**. It demonstrated exceptional robustness in various market conditions.
* **LightGBM** and **XGBoost** also significantly outperformed the baseline, with Sharpe Ratios above 2.0, proving that gradient boosting methods are effective at capturing non-linear market patterns.
* **Markowitz (Baseline)** provided a respectable return (345%) but failed to match the predictive power of machine learning models during volatile periods.

**Key Takeaway:** Contrary to traditional financial theory which relies on Mean-Variance optimization (Markowitz), ensemble Machine Learning models—specifically Random Forest—proved capable of identifying complex signals in technical and macroeconomic data, delivering 10x the returns of the standard approach.

### Summary Table
* **Random Forest:** 2.294 Sharpe Ratio (🏆 Winner)
* **LightGBM:** 2.139 Sharpe Ratio
* **XGBoost:** 2.077 Sharpe Ratio
* **Markowitz (Baseline):** 1.927 Sharpe Ratio

## 6. Requirements
* **Python 3.10+**
* **Data & Calc:** `pandas`, `numpy`, `yfinance`, `scikit-learn`
* **Models:** `xgboost`, `lightgbm`
* **Visualization:** `matplotlib`, `seaborn`
* **Utils:** `requests`, `multitasking`, `frozendict`, `beautifulsoup4`, `lxml`