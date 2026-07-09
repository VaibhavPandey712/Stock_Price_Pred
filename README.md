# 📈 Stock Price Predictor

Predicts short-term stock price movement from historical OHLCV data using
technical indicators and multiple regression models (Linear Regression,
Ridge, Random Forest, and SVR), with an interactive Streamlit dashboard.



## Demo

Run `streamlit run app.py` for an interactive dashboard where you can enter a
ticker, pick a prediction horizon, and compare models live.

## How it works

1. **Data** — historical OHLCV data is pulled via [`yfinance`](https://pypi.org/project/yfinance/).
   If no internet connection is available (or a ticker fails to resolve),
   the pipeline automatically falls back to a synthetic-but-realistic price
   series, so the project always runs out of the box.
2. **Feature engineering** (`src/features.py`) — builds technical indicators
   commonly used in quantitative finance:
   - Moving averages (5/10/20-day) and exponential moving averages
   - MACD and signal line
   - RSI (14-day)
   - Rolling volatility and momentum
   - Lagged closing prices (1/2/3/5/10 days)
3. **Target** — instead of predicting the raw future price, the model
   predicts the **future percentage return**. Raw prices are non-stationary
   (they drift over time), and tree/kernel models cannot extrapolate beyond
   the price range seen in training — they silently cap predictions near the
   training min/max. Returns are stationary, so every model generalizes
   properly to price levels never seen during training. The predicted return
   is converted back to a price forecast (`price × (1 + predicted_return)`)
   for reporting.
4. **Train/test split** — chronological (never shuffled), since shuffling
   time series data leaks future information into training.
5. **Models** — Linear Regression, Ridge, Random Forest, and SVR are trained
   and compared on held-out data using RMSE, MAE, R², and MAPE.
6. **Output** — the best model (lowest RMSE) is saved, along with comparison
   plots, to `outputs/`.

## Project structure

```
stock-price-predictor/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── data_loader.py      # Fetch data (yfinance / CSV / synthetic fallback)
│   ├── features.py         # Technical indicators + target construction
│   ├── models.py           # Model registry + evaluation metrics
│   ├── train.py            # End-to-end training pipeline (CLI)
│   └── predict.py          # Forecast N days into the future with a saved model
├── outputs/                # Saved model, scaler, plots, comparison table
├── data/                   # Optional: put your own CSVs here
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/stock-price-predictor.git
cd stock-price-predictor
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train models on a ticker

```bash
python src/train.py --ticker AAPL --start 2015-01-01 --horizon 1
```

Arguments:

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `AAPL` | Stock ticker symbol |
| `--start` | `2015-01-01` | History start date |
| `--end` | today | History end date |
| `--csv` | — | Use a local OHLCV CSV instead of downloading |
| `--horizon` | `1` | Trading days ahead to predict |
| `--test_size` | `0.2` | Fraction of data held out (chronologically) for testing |
| `--output_dir` | `outputs` | Where to save the model, scaler, and plots |

This saves `best_model.pkl`, `scaler.pkl`, `feature_cols.pkl`,
`model_comparison.csv`, `predictions.png`, and `rmse_comparison.png` to the
output directory.

### 2. Forecast future prices with the trained model

```bash
python src/predict.py --ticker AAPL --start 2015-01-01 --days 5
```

Produces a recursive N-day-ahead forecast (each predicted day feeds into the
next day's features).

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

### Using your own data

If you don't have live internet access to Yahoo Finance (or want to use a
different data source), pass a CSV with `Date, Open, High, Low, Close,
Volume` columns:

```bash
python src/train.py --csv data/my_stock.csv
```

## Example results

On Apple (AAPL) daily data, 1-day-ahead prediction, 80/20 chronological
split, models trained on returns:

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| SVR | ~0.68 | ~0.53 | ~0.994 | ~1.4% |
| Ridge | ~0.70 | ~0.54 | ~0.994 | ~1.4% |
| Linear Regression | ~0.71 | ~0.54 | ~0.994 | ~1.4% |
| Random Forest | ~0.96 | ~0.76 | ~0.989 | ~2.0% |

*(Exact numbers vary by ticker, date range, and market conditions — a 1-day
horizon on liquid large-cap stocks is close to a random walk, so a naive
"tomorrow = today" baseline is already strong. Metrics above are on a
synthetic demo series generated automatically when live data isn't
reachable; run the pipeline yourself on a real ticker to reproduce
comparable numbers.)*

![Predictions](outputs/predictions.png)
![RMSE comparison](outputs/rmse_comparison.png)

## Limitations & honest caveats

- **Short horizons on liquid stocks are close to a random walk.** A 1-day
  prediction that closely tracks "yesterday's price" is not necessarily
  informative — always compare against a naive baseline.
- **No fundamental or sentiment data** (news, earnings, macro indicators)
  is used — only price/volume history.
- **Random Forest and SVR** are more prone to overfitting on noisy financial
  data than the linear baselines; cross-validate before trusting them on a
  new ticker.
- This is a **regression/technical-analysis demo**, not a trading system.
  Transaction costs, slippage, and risk management are out of scope.

## Ideas for extending this project

- Add an LSTM/GRU (TensorFlow or PyTorch) baseline for sequence modeling
- Walk-forward (rolling-origin) cross-validation instead of a single split
- Multi-ticker portfolio-level prediction
- Backtest a simple trading strategy against a buy-and-hold baseline
- Add sentiment features from news/social data

## License

MIT — see [LICENSE](LICENSE).
