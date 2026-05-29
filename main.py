import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from src.copula_engine import CopulaStatArb

def fetch_data(ticker_x: str, ticker_y: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download([ticker_x, ticker_y], start=start_date, end=end_date)['Adj Close']
    data = data.dropna()
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def plot_copula_density(copula_trader, returns_x, returns_y):
    pseudo = copula_trader._compute_pseudo_observations(returns_x, returns_y)
    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=pseudo['U'], y=pseudo['V'], cmap="Blues", fill=True, bw_adjust=0.5)
    plt.title(f"Empirical Copula Density ({copula_trader.copula_name} Selected)")
    plt.xlabel('ECDF of Asset X')
    plt.ylabel('ECDF of Asset Y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    asset_x = 'XOM' 
    asset_y = 'CVX'
    
    print("Downloading Market Data...")
    returns = fetch_data(asset_x, asset_y, "2020-01-01", "2023-12-31")

    train_size = int(len(returns) * 0.7)
    train_data = returns.iloc[:train_size]
    test_data = returns.iloc[train_size:]

    trader = CopulaStatArb(asset_x, asset_y)
    trader.fit(train_data[asset_x], train_data[asset_y])

    print("Generating Trading Signals...")
    signals = trader.generate_signals(test_data[asset_x], test_data[asset_y])

    trade_days = signals[(signals['Position_X'] != 0)].shape[0]
    print(f"\nTotal Out-Of-Sample Days: {len(test_data)}")
    print(f"Days with active trade signals: {trade_days}")

    plot_copula_density(trader, train_data[asset_x], train_data[asset_y])