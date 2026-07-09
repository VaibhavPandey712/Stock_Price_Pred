"""
app.py
Interactive Streamlit dashboard for the Stock Price Predictor.

Run with:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from features import prepare_dataset
from models import evaluate, get_models

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor")
st.caption(
    "Trains Linear Regression, Ridge, Random Forest, and SVR on technical "
    "indicators to predict short-term stock price movement. "
    "Educational project — not financial advice."
)

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL")
    start = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    horizon = st.slider("Prediction horizon (trading days ahead)", 1, 10, 1)
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    run = st.button("Train & Predict", type="primary")

if run:
    with st.spinner(f"Loading data for {ticker}..."):
        raw = load_data(ticker=ticker, start=str(start))

    data, feature_cols = prepare_dataset(raw, horizon=horizon)
    feature_cols = [c for c in feature_cols if c not in ("Target", "Target_Price")]

    split_idx = int(len(data) * (1 - test_size))
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_test = test_df[feature_cols]
    y_test_price = test_df["Target_Price"]
    test_current_close = test_df["Close"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results, predictions = {}, {}
    progress = st.progress(0.0, text="Training models...")
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        pred_returns = model.predict(X_test_scaled)
        pred_price = test_current_close.values * (1 + pred_returns)
        results[name] = evaluate(y_test_price.values, pred_price)
        predictions[name] = pred_price
        progress.progress((i + 1) / len(models), text=f"Trained {name}")
    progress.empty()

    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    best_name = results_df.index[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Actual vs Predicted — {ticker}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_df.index, y_test_price.values, label="Actual", color="black", linewidth=2)
        for name, preds in predictions.items():
            ax.plot(test_df.index, preds, label=name, alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Model comparison")
        st.dataframe(results_df.style.format("{:.3f}"))
        st.success(f"Best model: **{best_name}**")

        last_close = data["Close"].iloc[-1]
        last_pred_return = models[best_name].predict(
            scaler.transform(data[feature_cols].iloc[[-1]])
        )[0]
        next_price = last_close * (1 + last_pred_return)
        st.metric(
            label=f"Predicted price in {horizon} trading day(s)",
            value=f"${next_price:,.2f}",
            delta=f"{last_pred_return * 100:+.2f}%",
        )

    st.caption(
        "⚠️ This is a technical/educational demo. Stock markets are influenced by "
        "far more than historical price patterns — do not use this for real trading decisions."
    )
else:
    st.info("Set your parameters in the sidebar and click **Train & Predict** to get started.")
