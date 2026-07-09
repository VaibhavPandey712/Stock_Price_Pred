"""
predict.py
Loads a model trained by train.py and forecasts prices N trading days
into the future, using an iterative (recursive) forecast: each day's
predicted price becomes part of the input for forecasting the next day.

Usage:
    python src/train.py --ticker AAPL           # train + save a model first
    python src/predict.py --ticker AAPL --days 5
"""

import argparse
import os

import joblib
import pandas as pd

from data_loader import load_data
from features import add_technical_indicators


def forecast(ticker: str, start: str, days: int, model_dir: str, csv_path: str | None = None):
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))

    raw = load_data(ticker=ticker, start=start, csv_path=csv_path)
    history = raw.copy()

    forecasts = []
    last_date = history.index[-1]

    for step in range(1, days + 1):
        feat_df = add_technical_indicators(history).dropna()
        latest_features = feat_df[feature_cols].iloc[[-1]]
        latest_scaled = scaler.transform(latest_features)

        predicted_return = model.predict(latest_scaled)[0]
        last_close = history["Close"].iloc[-1]
        predicted_price = last_close * (1 + predicted_return)

        next_date = last_date + pd.tseries.offsets.BDay(step)
        forecasts.append({"Date": next_date, "Predicted_Close": predicted_price})

        # Append a synthetic row so the next iteration's rolling features
        # (moving averages, lags, etc.) include this forecasted point.
        new_row = history.iloc[[-1]].copy()
        new_row.index = [next_date]
        new_row["Close"] = predicted_price
        new_row["Open"] = predicted_price
        new_row["High"] = predicted_price
        new_row["Low"] = predicted_price
        history = pd.concat([history, new_row])

    return pd.DataFrame(forecasts).set_index("Date")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast future stock prices with a trained model.")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01", help="History start date, needed to rebuild features")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--days", type=int, default=5, help="Number of future trading days to forecast")
    parser.add_argument("--model_dir", type=str, default="outputs")
    args = parser.parse_args()

    result = forecast(args.ticker, args.start, args.days, args.model_dir, args.csv)
    print(f"\n{args.days}-day forecast for {args.ticker}:\n")
    print(result.to_string(float_format=lambda x: f"${x:,.2f}"))
