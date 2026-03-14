import numpy as np
import os

def generate_market_data(n_assets: int = 10, n_years: int = 5, seed: int = 42) -> None:
    """
    Generates synthetic OHLC data using Geometric Brownian Motion.
    """
    np.random.seed(seed)
    
    TRADING_DAYS = 252
    TOTAL_DAYS = n_years * TRADING_DAYS
    DT = 1 / TRADING_DAYS
    
    mu = 0.0005 
    sigma = 0.02 
    start_price = 100.0

    print(f"Generating {n_assets} assets over {TOTAL_DAYS} days...")

    returns = np.random.normal(loc=mu * DT, scale=sigma * np.sqrt(DT), size=(TOTAL_DAYS, n_assets))
    price_paths = start_price * np.exp(np.cumsum(returns, axis=0))
    
    mask = np.random.choice([True, False], size=price_paths.shape, p=[0.005, 0.995])
    price_paths[mask] = np.nan

    # Save to data folder at the root of the project
    os.makedirs("data", exist_ok=True)
    np.save("data/market_prices.npy", price_paths)
    print("✅ Data saved to data/market_prices.npy")

if __name__ == "__main__":
    generate_market_data()


