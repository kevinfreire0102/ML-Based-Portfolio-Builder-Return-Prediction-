import pandas as pd
import yfinance as yf
import os
from typing import List

TICKERS_SAMPLE: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'] 
DEFAULT_START_DATE: str = '2015-01-01'
DEFAULT_END_DATE: str = '2024-12-31'
DATA_PATH = 'data/raw/stock_prices.csv' 

def download_sp500_data(tickers: List[str] = TICKERS_SAMPLE, 
                        start_date: str = DEFAULT_START_DATE, 
                        end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:

    
    if os.path.exists(DATA_PATH):
        print(f"Data found at {DATA_PATH}. Loading existing file...")
        try:
            return pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
        except Exception as e:
            print(f"Error reading saved data: {e}. Downloading new data...")
            

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        interval="1d",
        auto_adjust=False, 
        actions=True
    )['Adj Close']
    
    data = data.copy()
    
    # Drop dates where all prices are missing
    data.dropna(how='all', inplace=True)
    
    print("Download completed.")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True) 
    data.to_csv(DATA_PATH)
    print(f"Data saved to {DATA_PATH}.") 
    
    return data

if __name__ == "__main__":
    historical_data = download_sp500_data()
    print("\nShape of downloaded data:", historical_data.shape)
    print("\nData preview:")
    print(historical_data.head())