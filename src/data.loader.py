import pandas as pd
import yfinance as yf
from typing import List, Dict

# Sample list of tickers (will be replaced by the 100 real ones later)
TICKERS_SAMPLE: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'] 
DEFAULT_START_DATE: str = '2015-01-01'
DEFAULT_END_DATE: str = '2024-12-31'

def download_sp500_data(tickers: List[str] = TICKERS_SAMPLE, 
                        start_date: str = DEFAULT_START_DATE, 
                        end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for a list of S&P 500 tickers.
    
    Args:
        tickers (List[str]): List of stock tickers to download.
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame of adjusted prices, indexed by date, 
                      with one column per ticker.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Use yfinance.download() to fetch data, targeting only 'Adj Close' prices
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        interval="1d"
    )['Adj Close']
    
    # Drop dates where ALL prices are missing
    data.dropna(how='all', inplace=True)
    
    print("Download completed.")
    return data

# --- Local Testing Block (will be removed later) ---
if __name__ == "__main__":
    historical_data = download_sp500_data()
    print("\nShape of downloaded data:", historical_data.shape)
    print("\nData preview:")
    print(historical_data.head())
