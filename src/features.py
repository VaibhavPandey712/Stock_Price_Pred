

import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility_10"] = df["Close"].rolling(10).std()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)

    df["Close_to_MA20"] = df["Close"] / df["MA_20"] - 1

    for lag in (1, 2, 3, 5, 10):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    return df


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
   
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["Target_Price"] = future_close
    df["Target"] = future_close / df["Close"] - 1 
    return df


def prepare_dataset(df: pd.DataFrame, horizon: int = 1):
    df = add_technical_indicators(df)
    df = create_target(df, horizon=horizon)
    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ("Target", "Target_Price")]
    return df, feature_cols
