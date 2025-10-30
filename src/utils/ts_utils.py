# Utilities for a simple forecasting demo used by the Streamlit app.
# Keep functions lightweight and safe for quick interactive runs.
import os
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# lightgbm may not be available in minimal env; import lazily
def _import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise ImportError("lightgbm not installed. Install via `pip install lightgbm` or use Prophet.") from e

def download_yfinance(ticker, start, end, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker/date range.")
    path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(path)
    return path

def preprocess_df(df):
    # df: DataFrame from yfinance with DateTimeIndex or index col 0 as Date
    if not isinstance(df.index, pd.DatetimeIndex):
        # try parse first column as date
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "Close" not in df.columns:
        raise RuntimeError("Expected 'Close' column")
    out = pd.DataFrame({"y": df["Close"]}, index=df.index)
    out = out.asfreq("D")  # daily freq
    out["y"] = out["y"].ffill()
    return out

def create_lag_rolling_features(df, lags=[1,2,3,7], windows=[3,7]):
    # df: DataFrame with column 'y' and DateTimeIndex
    df_feat = df.copy()
    for l in lags:
        df_feat[f"lag_{l}"] = df_feat["y"].shift(l)
    for w in windows:
        df_feat[f"roll_mean_{w}"] = df_feat["y"].shift(1).rolling(w).mean()
        df_feat[f"roll_std_{w}"] = df_feat["y"].shift(1).rolling(w).std()
    return df_feat

def train_lgbm(df_feat, target_col="y"):
    # df_feat: with features and target; dropna expected
    lgb = _import_lgb()
    data = df_feat.dropna()
    X = data.drop(columns=[target_col])
    y = data[target_col]
    # use last 20% as validation chronologically
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    return model

def forecast_lgbm_iterative(model, df_series, steps=30, lags=[1,2,3,7,14,30], windows=[3,7,14]):
    # df_series: DataFrame with column 'y' indexed by date (full history)
    # iterative forecasting: use last available values to produce next steps
    history = df_series.copy()
    history = history.asfreq("D")
    preds = []
    current = history.copy()
    for i in range(steps):
        # create features from current
        feat = create_lag_rolling_features(current, lags=lags, windows=windows).iloc[[-1]]
        feat = feat.drop(columns=["y"])
        # some features may be NaN (if short history) -> fill with last value
        feat = feat.fillna(method="ffill").fillna(method="bfill").fillna(0)
        pred = model.predict(feat)[0]
        next_date = current.index[-1] + pd.Timedelta(days=1)
        preds.append({"ds": next_date, "pred": float(pred)})
        # append to current to allow next iteration
        current.loc[next_date] = [pred]
    df_pred = pd.DataFrame(preds).set_index("ds")
    return df_pred

def train_prophet(df):
    # df: DataFrame with column 'y' and DateTimeIndex
    df_p = df.reset_index().rename(columns={df.index.name or "index":"ds", "y":"y"})
    df_p["ds"] = pd.to_datetime(df_p["ds"])
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_p)
    return m

def forecast_prophet(model, df, periods=30):
    last_date = df.index.max()
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    # keep only future beyond last_date
    fc = forecast.set_index("ds")[["yhat"]]
    fc = fc[fc.index > last_date].rename(columns={"yhat":"pred"})
    return fc

def evaluate_metrics(y_true, y_pred):
    # y_true, y_pred: pandas Series aligned by index
    y_true = pd.to_numeric(y_true)
    y_pred = pd.to_numeric(y_pred)
    mask = y_true.dropna().index.intersection(y_pred.dropna().index)
    if len(mask) == 0:
        return {}
    yt = y_true.loc[mask]
    yp = y_pred.loc[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp, squared=False)
    mape = np.mean(np.abs((yt - yp) / (yt + 1e-9))) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE(%)": float(mape)}
