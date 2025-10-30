"""
Create lag and rolling features for ML models (LightGBM/LSTM input).
Example usage from notebook or script.
"""
import pandas as pd

def create_lag_features(df, lags=[1,2,3,7,14,30]):
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)
    return df

def create_rolling_features(df, windows=[3,7,14,30]):
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std()
    return df

def prepare_ml_df(df):
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y