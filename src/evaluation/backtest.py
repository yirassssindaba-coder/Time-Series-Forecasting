#!/usr/bin/env python3
"""
Simple rolling-origin backtest for ML models.
Usage:
  python src/evaluation/backtest.py --input data/processed/AAPL_parsed.csv
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.features.feature_engineering import prepare_ml_df
from src.models.lgbm_model import train_lgbm
import joblib
import os

def rolling_backtest(df, initial=1000, horizon=30, step=30):
    # df: series index sorted, column y
    results = []
    n = len(df)
    cursor = initial
    while cursor + horizon <= n:
        train = df.iloc[:cursor]
        test = df.iloc[cursor:cursor+horizon]
        X_train, y_train = prepare_ml_df(train.copy())
        X_test, y_test = prepare_ml_df(pd.concat([train.tail( max(30, 1) ), test]).copy())
        # align test to first horizon rows
        X_test = X_test.iloc[-horizon:]
        y_test = y_test.iloc[-horizon:]
        model = train_lgbm(X_train, y_train, model_path=f"models/lgbm_bt_{cursor}.pkl")
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Window end {cursor}: MAE={mae:.4f}")
        results.append({"end": cursor, "mae": mae})
        cursor += step
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=True, index_col=0)
    df = df.rename(columns={"y":"y"})[["y"]]
    os.makedirs("models", exist_ok=True)
    res = rolling_backtest(df, initial=1000, horizon=30, step=90)
    res.to_csv("backtest_results.csv", index=False)
    print("Backtest finished. Results saved to backtest_results.csv")