import pandas as pd
import yfinance as yf
from typing import List

TICKERS_SAMPLE: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'] 
DEFAULT_START_DATE: str = '2015-01-01'
DEFAULT_END_DATE: str = '2024-12-31'

def download_sp500_data(tickers: List[str] = TICKERS_SAMPLE, 
                        start_date: str = DEFAULT_START_DATE, 
                        end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
   
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
    
    # Drop dates where ALL prices are missing
    data.dropna(how='all', inplace=True)
    
    print("Download completed.")
    return data

if __name__ == "__main__":
    historical_data = download_sp500_data()
    print("\nShape of downloaded data:", historical_data.shape)
    print("\nData preview:")
    print(historical_data.head())