
import numpy as np
import pandas as pd


def load_data(ticker: str = "AAPL", start: str = "2015-01-01",
              end: str | None = None, csv_path: str | None = None) -> pd.DataFrame:
    
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty dataframe returned by yfinance")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"[data_loader] Could not fetch live data for '{ticker}' ({e}).")
        print("[data_loader] Falling back to synthetic demo data so you can still run the pipeline.")
        return generate_synthetic_data(start=start, end=end)


def generate_synthetic_data(start: str = "2015-01-01", end: str | None = None,
                             seed: int = 42) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(seed)

    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=len(idx))
    close = 100 * np.exp(np.cumsum(daily_returns))

    open_ = close * (1 + rng.normal(0, 0.002, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    volume = rng.integers(1_000_000, 10_000_000, len(idx))

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=idx)
    df.index.name = "Date"
    return df
