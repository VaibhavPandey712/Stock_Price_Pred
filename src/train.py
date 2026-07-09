"""
train.py
End-to-end pipeline: load data -> engineer features -> chronological
train/test split -> train multiple models -> evaluate -> plot -> save
the best model + scaler to outputs/.

Usage:
    python src/train.py --ticker AAPL --start 2015-01-01 --horizon 1
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # safe for headless / CI environments
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from features import prepare_dataset
from models import evaluate, get_models


def time_series_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split -- never shuffle time series data."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/5] Loading data for {args.ticker}...")
    raw = load_data(ticker=args.ticker, start=args.start, end=args.end, csv_path=args.csv)

    print("[2/5] Engineering features (moving averages, RSI, MACD, lags)...")
    data, feature_cols = prepare_dataset(raw, horizon=args.horizon)
    feature_cols = [c for c in feature_cols if c != "Target"]

    train_df, test_df = time_series_split(data, test_size=args.test_size)
    # Models are trained on returns (stationary target) -- see features.py for why.
    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_test, y_test = test_df[feature_cols], test_df["Target"]
    # But we evaluate/report in actual price terms, since that's what's meaningful.
    y_test_price = test_df["Target_Price"]
    test_current_close = test_df["Close"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[3/5] Training {len(get_models())} models on {len(X_train)} samples "
          f"(testing on {len(X_test)} held-out samples)...")
    models = get_models()
    results, predictions = {}, {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred_returns = model.predict(X_test_scaled)
        # Reconstruct price forecast: today's close * (1 + predicted return)
        pred_price = test_current_close.values * (1 + pred_returns)
        metrics = evaluate(y_test_price.values, pred_price)
        results[name] = metrics
        predictions[name] = pred_price
        print(f"    {name:<16} RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  "
              f"R2={metrics['R2']:.3f}  MAPE={metrics['MAPE']:.2f}%")

    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    results_df.to_csv(os.path.join(args.output_dir, "model_comparison.csv"))

    best_name = results_df.index[0]
    best_model = models[best_name]
    print(f"[4/5] Best model: {best_name} (lowest RMSE)")

    joblib.dump(best_model, os.path.join(args.output_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(args.output_dir, "feature_cols.pkl"))
    with open(os.path.join(args.output_dir, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    print("[5/5] Saving plots...")
    plt.figure(figsize=(14, 6))
    plt.plot(test_df.index, y_test_price.values, label="Actual", linewidth=2, color="black")
    for name, preds in predictions.items():
        plt.plot(test_df.index, preds, label=f"{name}", alpha=0.75)
    plt.title(f"{args.ticker} — Actual vs Predicted Close Price ({args.horizon}-day horizon)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "predictions.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results_df.index, results_df["RMSE"], color="#4C72B0")
    plt.title("Model Comparison — RMSE (lower is better)")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rmse_comparison.png"), dpi=150)
    plt.close()

    print(f"\nDone. Results, plots, and the trained model are in '{args.output_dir}/'.")
    print("\n=== Final comparison (sorted by RMSE) ===")
    print(results_df.to_string())
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock price prediction models.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--csv", type=str, default=None, help="Path to local OHLCV CSV instead of downloading")
    parser.add_argument("--horizon", type=int, default=1, help="Trading days ahead to predict")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save results")
    args = parser.parse_args()
    main(args)
