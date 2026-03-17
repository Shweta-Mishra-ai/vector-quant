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

    def calculate_risk_metrics(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        PHASE 3: LINEAR ALGEBRA
        Task: Calculate Covariance and Correlation matrices to see how assets move together.
        """
        # np.cov expects assets as rows, observations as columns. So we transpose (.T)
        cov_matrix = np.cov(returns.T)
        
        # Correlation matrix scales covariance between -1 and 1
        corr_matrix = np.corrcoef(returns.T)
        
        return cov_matrix, corr_matrix

    def monte_carlo_var(self, returns: np.ndarray, initial_value: float = 100000.0, days: int = 252, simulations: int = 10000) -> float:
        """
        PHASE 3: PROBABILITY & SIMULATION
        Task: Simulate 10,000 future portfolio paths to find the 95% Value at Risk (VaR).
        """
        num_assets = returns.shape[1]
        mu = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # 1. Cholesky Decomposition (The Senior Data Scientist Secret)
        # We need correlated random numbers. L @ L.T = Covariance Matrix
        L = np.linalg.cholesky(cov_matrix)
        
        # 2. Generate pure random noise: Shape (10000 sims, 252 days, 10 assets)
        Z = np.random.normal(0, 1, size=(simulations, days, num_assets))
        
        # 3. Tensor Multiplication (Advanced NumPy)
        # Apply the correlation matrix (L) to our random noise (Z) using Einstein Summation
        daily_shocks = np.einsum('ij, stj -> sti', L, Z)
        
        # 4. Add the historical mean (drift)
        simulated_returns = mu + daily_shocks
        
        # 5. Calculate cumulative portfolio value paths (assuming equal weights)
        weights = np.ones(num_assets) / num_assets
        port_sim_returns = np.sum(simulated_returns * weights, axis=-1)
        
        # Exponential cumulative sum to simulate compounding growth
        cumulative_returns = np.exp(np.cumsum(port_sim_returns, axis=1))
        final_values = initial_value * cumulative_returns[:, -1]
        
        # 6. Calculate 95% Value at Risk (VaR)
        # What is the worst-case loss in the bottom 5% of our 10,000 alternate realities?
        var_95 = initial_value - np.percentile(final_values, 5)
        
        return var_95
    if __name__ == "__main__":
    # --- PHASE 1 & 2 ---
    engine = VectorQuantEngine("data/market_prices.npy")
    clean_prices = engine.clean_data()
    returns = engine.calculate_returns(clean_prices)
    
    # --- PHASE 3 ---
    print("\n--- Phase 3: Risk & Simulation ---")
    cov, corr = engine.calculate_risk_metrics(returns)
    print(f"Covariance Matrix Shape: {cov.shape}")
    
    # Run Monte Carlo Simulation
    print("Running 10,000 Monte Carlo Simulations... (Watch how fast NumPy does this)")
    var_95 = engine.monte_carlo_var(returns, initial_value=100000.0)
    
    print("\n--- Risk Report ---")
    print(f"Initial Portfolio Value: $100,000")
    print(f"95% Value at Risk (1 Year): ${var_95:,.2f}")
    print("Interpretation: We are 95% confident our portfolio will NOT lose more than this amount in the next year.")

