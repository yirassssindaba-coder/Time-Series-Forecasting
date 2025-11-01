#!/usr/bin/env python
"""
pipeline_run.py (v42)

Robust local pipeline:
- download_data (yfinance primary, local CSV fallback, Stooq fallback)
- preprocess, feature creation
- train model (LightGBM fallback RandomForest)
- iterative forecast
- save forecast CSV into results/, save Plotly HTML + PNG into notebooks_html/

Fixes in v42:
- generate_report_html_png uses the function parameter "ticker" (lowercase) instead of an undefined global TICKER.
- Minor robustness around CSV writing and path creation.
"""
import sys
import traceback
import re
import warnings
import time
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "notebooks_html"
RESULTS = ROOT / "results"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

for d in (DATA_RAW, DATA_PROC, MODELS, REPORTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

def flatten_columns_if_multiindex(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for col in df.columns:
            parts = [str(x) for x in col if (x is not None and str(x) != "")]
            cols.append("_".join(parts) if parts else str(col))
        df.columns = cols
    return df

def download_data(ticker, start, end, out_dir=DATA_RAW, max_retries=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{ticker}.csv"
    last_exc = None
    if yf is not None:
        attempt = 0
        backoff = 1.0
        while attempt < max_retries:
            attempt += 1
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df is None or df.empty:
                    last_exc = RuntimeError("yfinance returned empty")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                df = flatten_columns_if_multiindex(df)
                df.to_csv(p)
                return df, p
            except Exception as e:
                last_exc = e
                time.sleep(backoff)
                backoff *= 2
    # local fallback
    for cand in [p, out_dir / f"{ticker.upper()}.csv", out_dir / f"{ticker.lower()}.csv"]:
        if cand.exists():
            try:
                df = pd.read_csv(cand, index_col=0, parse_dates=True, engine="python")
                if not df.empty:
                    return df, cand
            except Exception:
                pass
    # stooq fallback
    if requests is not None:
        try:
            sym = ticker
            if re.match(r"^[A-Z0-9]+$", ticker):
                sym = f"{ticker.lower()}.us"
            d1 = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
            d2 = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")
            url = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text.strip():
                p2 = out_dir / f"{ticker}_stooq.csv"
                p2.write_text(r.text, encoding="utf-8")
                df = pd.read_csv(p2, parse_dates=["Date"], index_col=0, engine="python")
                if not df.empty:
                    return df, p2
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to download data (last exception: {last_exc})")

def preprocess(df_or_path, out_path):
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        df = pd.read_csv(df_or_path, engine="python")
    df = flatten_columns_if_multiindex(df)
    numeric = []
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().sum() > 0:
                numeric.append(c)
        except Exception:
            pass
    pref = None
    for cand in ("Close","close","Adj_Close","Adj Close","AdjClose"):
        if cand in df.columns:
            pref = cand; break
    if not pref and numeric:
        pref = numeric[0]
    if not pref:
        raise RuntimeError("No numeric column found")
    s = df[[pref]].rename(columns={pref:"y"}).sort_index()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].copy()
    try:
        s = s.asfreq("D")
    except Exception:
        dr = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
        s = s.reindex(dr)
        s.index.name = None
    s["y"] = s["y"].ffill()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_path)
    return s, out_path

def create_features(df):
    out = df.copy()
    out["lag_1"] = out["y"].shift(1)
    out["lag_7"] = out["y"].shift(7)
    out["roll_7"] = out["y"].shift(1).rolling(7).mean()
    return out

def train_simple(X, y):
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=100)
        model.fit(X, y)
        return model, "lightgbm"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, "random_forest"

def iterative_forecast(model, history_df, steps=30):
    cur = history_df.copy().asfreq("D")
    preds = []
    for _ in range(steps):
        feat = create_features(cur).iloc[[-1]].drop(columns=["y"], errors="ignore").fillna(0)
        try:
            p = float(model.predict(feat)[0])
        except Exception:
            p = float(model.predict(feat.values)[0])
        nxt = cur.index[-1] + pd.Timedelta(days=1)
        preds.append((nxt, p))
        cur.loc[nxt] = [p]
    return pd.DataFrame({"forecast":[v for (_,v) in preds]}, index=[d for (d,_) in preds])

def generate_report_html_png(series, fc, ticker):
    """
    Generate a Plotly HTML report and a PNG snapshot.
    IMPORTANT: use 'ticker' parameter (lowercase) to avoid NameError.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["y"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color="red")))
    fig.update_layout(title=f"{ticker} Actual vs Forecast", xaxis_title="Date", yaxis_title="Price", height=600)
    out_html = REPORTS / f"report_{ticker}_{TIMESTAMP}.html"
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)
    out_png = out_html.with_suffix(".png")
    try:
        plt.figure(figsize=(10,6))
        plt.plot(series.index, series["y"], color="black", label="Actual")
        plt.plot(fc.index, fc["forecast"], color="red", label="Forecast")
        plt.legend()
        plt.title(f"{ticker} Actual vs Forecast")
        plt.tight_layout()
        plt.savefig(str(out_png))
        plt.close()
    except Exception:
        pass
    return out_html, out_png

def main():
    try:
        TICKER="AAPL"; START="2015-01-01"; END="2024-01-01"; HORIZON=30
        df_raw, raw_path = download_data(TICKER, START, END)
        processed_df, processed_path = preprocess(df_raw, DATA_PROC / f"{TICKER}_parsed.csv")
        series = processed_df.copy()
        df_feat = create_features(series).dropna()
        X = df_feat.drop(columns=["y"]); y = df_feat["y"]
        model, model_name = train_simple(X, y)
        joblib.dump({"model":model, "meta":{"model_name":model_name}}, MODELS / f"{model_name}_{TICKER}_{TIMESTAMP}.pkl")
        fc = iterative_forecast(model, series, steps=HORIZON)

        RESULTS.mkdir(parents=True, exist_ok=True)
        forecast_csv_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.csv"
        combined = pd.concat([series.rename(columns={"y":"actual"}), fc], axis=0)
        forecast_csv_path.write_text(combined.to_csv())
        print(f"[main] Forecast saved: {forecast_csv_path}")

        html, png = generate_report_html_png(series, fc, TICKER)
        print("Generated report:", html, png)
    except Exception as e:
        print("Pipeline failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
