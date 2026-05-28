import pandas as pd
import numpy as np
import logging
import yfinance as yf
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataHandler:
    @staticmethod
    def fetch_yahoo_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        logging.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        if data.empty:
            raise ValueError("No data fetched. Check tickers and date range.")

        data = data.ffill().dropna()
        return data

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(prices / prices.shift(1)).dropna()

        for col in log_returns.columns:
            if log_returns[col].var() == 0:
                raise ValueError(f"Zero variance detected in returns for {col}")
                
        return log_returns

    @staticmethod
    def train_test_split(data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        logging.info(f"Data split: {len(train_data)} train samples, {len(test_data)} test samples.")
        return train_data, test_data