import numpy as np
from typing import Tuple

class VectorQuantEngine:
    def __init__(self, filepath: str):
        print("--- Initializing Engine ---")
        self.raw_data = np.load(filepath, mmap_mode='r') 
        self.data = np.array(self.raw_data) 
        print(f"Loaded data shape: {self.data.shape} (Days, Assets)")

    def clean_data(self) -> np.ndarray:
        nan_mask = np.isnan(self.data)
        col_means = np.nanmean(self.data, axis=0)
        
        cleaned_data = self.data.copy()
        for i in range(cleaned_data.shape[1]):
            col_mask = np.isnan(cleaned_data[:, i])
            cleaned_data[col_mask, i] = col_means[i]
            
        return cleaned_data

    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        delta = np.diff(prices, axis=0)
        returns = delta / prices[:-1, :] 
        return returns

    def get_rolling_stats(self, prices: np.ndarray, window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        # Fast Moving Average via Cumsum
        ret = np.cumsum(prices, axis=0)
        ret[window:] = ret[window:] - ret[:-window]
        sma = ret[window - 1:] / window
        
        # Fast Rolling Volatility via Stride Tricks
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(prices, window_shape=window, axis=0)
        rolling_vol = np.std(windows, axis=-1)
        
        return sma, rolling_vol

    def portfolio_simulation(self, returns: np.ndarray) -> dict:
        num_assets = returns.shape[1]
        
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) 
        
        # Vectorized portfolio returns using Matrix Multiplication
        portfolio_daily_returns = returns @ weights
        
        total_return = np.sum(portfolio_daily_returns)
        sharpe_ratio = np.mean(portfolio_daily_returns) / np.std(portfolio_daily_returns)
        
        return {
            "weights": weights,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio * np.sqrt(252)
        }

if __name__ == "__main__":
    engine = VectorQuantEngine("data/market_prices.npy")
    
    clean_prices = engine.clean_data()
    print(f"Data Cleaned. Any NaNs left? {np.isnan(clean_prices).any()}")
    
    returns = engine.calculate_returns(clean_prices)
    print(f"Daily Returns Calculated. Shape: {returns.shape}")
    
    sma, vol = engine.get_rolling_stats(clean_prices, window=30)
    print(f"Rolling SMA generated. Shape: {sma.shape}")
    
    metrics = engine.portfolio_simulation(returns)
    print("\n--- Portfolio Report ---")
    print(f"Projected Annual Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")

