#!/usr/bin/env python3
# Simple Streamlit app to run the Time-Series Forecasting pipeline and show interactive graphs
# Usage:
# 1) Install requirements: pip install -r requirements.txt
# 2) From repo root: streamlit run app/streamlit_app.py

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.utils.ts_utils import (
    download_yfinance,
    preprocess_df,
    create_lag_rolling_features,
    train_lgbm,
    forecast_lgbm_iterative,
    train_prophet,
    forecast_prophet,
    evaluate_metrics,
)

st.set_page_config(page_title="Time-Series Forecasting Demo", layout="wide")

st.title("Time-Series Forecasting (Interactive)")

with st.sidebar:
    st.header("Input")
    ticker = st.text_input("Ticker / Series name", "AAPL")
    start_date = st.date_input("Start date", datetime(2015, 1, 1))
    end_date = st.date_input("End date", datetime.today())
    horizon_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=30)
    model_choice = st.selectbox("Model", ["LightGBM (lag features)", "Prophet"])
    run_button = st.button("Run Forecast")

st.info("Notes: This demo downloads historical 'Close' price via yfinance and runs a quick baseline forecast.")

if run_button:
    st.session_state.run_time = datetime.utcnow().isoformat()
    st.spinner("Downloading data...")
    try:
        raw_path = download_yfinance(ticker, start_date.isoformat(), end_date.isoformat(), out_dir=None)
        df_raw = pd.read_csv(raw_path, parse_dates=True, index_col=0)
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

    st.success("Data downloaded.")
    st.write("Raw data sample:", df_raw.tail(5))

    st.info("Preprocessing...")
    try:
        df = preprocess_df(df_raw)  # returns DataFrame with column 'y' and DateTimeIndex
    except Exception as e:
        st.error(f"Preprocess failed: {e}")
        st.stop()

    st.write("Processed series (last 10 rows):")
    st.line_chart(df["y"].tail(200))

    # Prepare features for ML
    st.info("Preparing features...")
    df_feat = create_lag_rolling_features(df.copy(), lags=[1,2,3,7,14,30], windows=[3,7,14])
    df_feat = df_feat.dropna()
    st.write(f"Features shape: {df_feat.shape}")

    # Split into train (all) for model fit, we'll forecast horizon iteratively
    if model_choice.startswith("LightGBM"):
        st.info("Training LightGBM (quick baseline)...")
        try:
            model = train_lgbm(df_feat)
            st.success("LightGBM trained.")
            preds = forecast_lgbm_iterative(model, df.copy(), steps=int(horizon_days))
            # preds: DataFrame with index of future dates and column 'pred'
            forecast_df = preds
        except Exception as e:
            st.error(f"LightGBM training/forecast failed: {e}")
            st.stop()

    else:  # Prophet
        st.info("Training Prophet (quick)...")
        try:
            prop_model = train_prophet(df)
            st.success("Prophet trained.")
            forecast_df = forecast_prophet(prop_model, df, periods=int(horizon_days))
        except Exception as e:
            st.error(f"Prophet training/forecast failed: {e}")
            st.stop()

    # Combine actual + forecast
    combined = pd.concat([df[["y"]], forecast_df.rename(columns={"pred":"y_pred"})], axis=0)
    # Metrics: if overlap (holdout) exists, compute metrics on the forecast horizon where actual known
    metrics = {}
    try:
        # evaluate where both actual and pred exist
        if "y" in forecast_df.columns:
            y_true = forecast_df["y"].dropna()
            y_pred = forecast_df["pred"]
            metrics = evaluate_metrics(y_true, y_pred)
        else:
            # If forecast only future, compute no metrics (display horizon only)
            metrics = {}
    except Exception:
        metrics = {}

    # Plot interactive result
    st.header("Forecast result")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["y"], name="Actual", line=dict(color="black")))
    # plot forecast (pred)
    if "pred" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["pred"], name="Forecast", line=dict(color="red")))
    if "y_pred" in combined.columns:
        fig.add_trace(go.Scatter(x=combined.index, y=combined["y_pred"], name="Forecast (combined)", line=dict(color="red", dash="dot")))
    fig.update_layout(title=f"{ticker} - Actual vs Forecast ({model_choice})", xaxis_title="Date", yaxis_title="Value", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Show metrics if any
    if metrics:
        st.subheader("Metrics on available holdout")
        st.write(metrics)

    # Download forecast CSV
    csv = forecast_df.reset_index().to_csv(index=False)
    st.download_button("Download forecast CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")

    st.success("Done. You can refine the pipeline (hyperparams, CV, feature engineering) in notebooks/ for production use.")