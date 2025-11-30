import pandas as pd
import yfinance as yf
import os # <-- NOUVEL IMPORT POUR LA GESTION DES FICHIERS
from typing import List

# --- Configuration des chemins et des paramètres par défaut ---
TICKERS_SAMPLE: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'] 
DEFAULT_START_DATE: str = '2015-01-01'
DEFAULT_END_DATE: str = '2024-12-31'
DATA_PATH = 'data/raw/stock_prices.csv' # <-- NOUVEAU CHEMIN DE SAUVEGARDE

def download_sp500_data(tickers: List[str] = TICKERS_SAMPLE, 
                        start_date: str = DEFAULT_START_DATE, 
                        end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for a list of S&P 500 tickers,
    or loads from DATA_PATH if the file exists (for faster execution).
    """
    
    # 1. Check if data already exists (for fast loading)
    if os.path.exists(DATA_PATH):
        print(f"Data found at {DATA_PATH}. Loading existing file...")
        try:
            # Assurez-vous que la colonne 'Date' est bien l'index
            return pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
        except Exception as e:
            print(f"Error reading saved data: {e}. Downloading new data...")
            # Fallback to downloading if file is corrupted

    # 2. If not found, download and save
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

    # 3. Save the raw data before returning
    # os.makedirs crée le dossier data/raw s'il n'existe pas
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True) 
    data.to_csv(DATA_PATH)
    print(f"Data saved to {DATA_PATH}.") # <-- CONFIRMATION DE SAUVEGARDE
    
    return data

if __name__ == "__main__":
    historical_data = download_sp500_data()
    print("\nShape of downloaded data:", historical_data.shape)
    print("\nData preview:")
    print(historical_data.head())