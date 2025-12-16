import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.data_loader import download_sp500_data


def create_mock_data():
    """Creates a mock DataFrame mimicking yfinance output structure."""
    dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
    data = {
        'AAPL': [150.0, 151.0, 152.0, np.nan],
        'MSFT': [200.0, 201.0, np.nan, np.nan],
        'GOOGL': [1000.0, 1001.0, 1002.0, np.nan] 
    }
    # yfinance returns a DataFrame with MultiIndex columns (e.g., ('Adj Close', 'AAPL'))
    mock_df_adj_close = pd.DataFrame(data, index=dates)
    
    # Build the mock return structure
    mock_return = pd.DataFrame(index=dates)
    mock_return['Adj Close', 'AAPL'] = mock_df_adj_close['AAPL']
    mock_return['Adj Close', 'MSFT'] = mock_df_adj_close['MSFT']
    mock_return['Adj Close', 'GOOGL'] = mock_df_adj_close['GOOGL']
    # Set the column names to be a MultiIndex
    mock_return.columns = pd.MultiIndex.from_tuples(mock_return.columns)
    
    return mock_return

class TestDataLoader(unittest.TestCase):
    """
    Unit tests for functions in src/data_loader.py using mocking.
    """

    @patch('yfinance.download')
    def test_download_sp500_data_structure(self, mock_download):
        """Test if the downloaded DataFrame has the correct shape and type after cleaning."""
        
        # 1. Configure the mock to return our simulated data
        mock_download.return_value = create_mock_data()
        
        tickers_subset = ['AAPL', 'MSFT', 'GOOGL']
        data = download_sp500_data(tickers=tickers_subset, start_date='2020-01-01', end_date='2020-01-04')
        
        # 2. Assertions (The 4th row, all NaN, should be dropped by the function)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (3, 3)) 
        self.assertListEqual(data.columns.tolist(), tickers_subset)
        self.assertNotIn(pd.to_datetime('2020-01-04'), data.index)

    @patch('yfinance.download')
    def test_download_sp500_data_call(self, mock_download):
        """Test if yfinance.download is called with the correct arguments."""
        
        mock_download.return_value = create_mock_data()
        test_tickers = ['T1', 'T2']
        test_start = '2022-01-01'
        test_end = '2023-01-01'
        
        download_sp500_data(tickers=test_tickers, start_date=test_start, end_date=test_end)
        
        mock_download.assert_called_once_with(
            test_tickers, 
            start=test_start, 
            end=test_end, 
            interval="1d",
            auto_adjust=False, 
            actions=True        
        )

if __name__ == '__main__':
    unittest.main()