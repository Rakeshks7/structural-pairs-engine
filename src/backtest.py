import pandas as pd
import numpy as np
import logging

class VectorizedBacktester:
    
    def __init__(self, initial_capital: float = 100000.0, transaction_cost_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.tcost = transaction_cost_bps / 10000.0

    def run(self, prices_x: pd.Series, prices_y: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing Vectorized Backtest...")
        
        df = pd.DataFrame(index=prices_x.index)
        df['Price_X'] = prices_x
        df['Price_Y'] = prices_y

        df['Ret_X'] = np.log(df['Price_X'] / df['Price_X'].shift(1))
        df['Ret_Y'] = np.log(df['Price_Y'] / df['Price_Y'].shift(1))

        df['Pos_X'] = signals['Position_X'].shift(1).fillna(0)
        df['Pos_Y'] = signals['Position_Y'].shift(1).fillna(0)

        df['Strategy_Ret'] = (df['Pos_X'] * df['Ret_X'] * 0.5) + (df['Pos_Y'] * df['Ret_Y'] * 0.5)

        df['Trade_X'] = df['Pos_X'].diff().abs()
        df['Trade_Y'] = df['Pos_Y'].diff().abs()
        
        df['Costs'] = (df['Trade_X'] * self.tcost * 0.5) + (df['Trade_Y'] * self.tcost * 0.5)
        df['Strategy_Ret_Net'] = df['Strategy_Ret'] - df['Costs'].fillna(0)

        df['Cumulative_Return'] = df['Strategy_Ret_Net'].cumsum()
        df['Equity'] = self.initial_capital * np.exp(df['Cumulative_Return'])
        
        return df

    def calculate_metrics(self, backtest_results: pd.DataFrame) -> dict:
        returns = backtest_results['Strategy_Ret_Net'].dropna()
        
        total_return = np.exp(returns.sum()) - 1
        annualized_return = np.exp(returns.mean() * 252) - 1

        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0

        cumulative = backtest_results['Equity']
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            "Total Return (%)": round(total_return * 100, 2),
            "Annualized Return (%)": round(annualized_return * 100, 2),
            "Annualized Volatility (%)": round(annualized_vol * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown * 100, 2)
        }
        
        for k, v in metrics.items():
            logging.info(f"{k}: {v}")
            
        return metrics