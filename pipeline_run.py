# pipeline_run.py (robust preprocess)
import sys, subprocess, json, warnings, traceback
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORTS = RESULTS / "notebooks_html"

for d in (DATA_RAW, DATA_PROC, MODELS, RESULTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

def pip_install(packages):
    if isinstance(packages, str):
        packages = [packages]
    for pkg in packages:
        name = pkg.split("==")[0].split(">=")[0].split("<=")[0].split()[0]
        try:
            import importlib
            if importlib.util.find_spec(name) is not None:
                print(f"[pip] {name} already installed")
                continue
        except Exception:
            pass
        print(f"[pip] Installing {pkg} ...")
        try:
            proc = subprocess.run([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout[:2000])
        except Exception as e:
            print(f"[pip] install failed for {pkg}: {e}")

# Install lightweight essentials if missing (best-effort)
pip_install(["pandas>=1.5", "numpy", "yfinance", "scikit-learn", "plotly", "joblib"])

# Try heavier libs but not fatal if fail
pip_install(["lightgbm", "prophet", "statsmodels"])

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import plotly.graph_objects as go
import plotly.io as pio
import traceback

# Config -- adjust as needed
TICKER = "AAPL"
START = "2015-01-01"
END = "2024-01-01"
HORIZON = 30
BACKTEST_HORIZON = 30
BACKTEST_STEP = 30
INITIAL_TRAIN_YEARS = 3
TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def download_data(ticker, start, end, out_dir=DATA_RAW):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.csv"
    print(f"[download] {ticker} {start}..{end}")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")
    if df is None or df.empty:
        raise RuntimeError("Downloaded data empty. Check ticker/date range or internet.")
    df.to_csv(path)
    return path

def _identify_date_column(df):
    # Try common names or detect column with most parseable datetimes
    candidates = []
    for col in df.columns:
        if col.lower() in ("date", "day", "timestamp", "ds"):
            return col
    # Try index if index looks like dates
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().sum() > 0 and idx.notna().mean() > 0.5:
            return None  # indicates use index
    except Exception:
        pass
    # Try to find a column with many parseable datetimes
    best_col = None
    best_count = 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            cnt = parsed.notna().sum()
            if cnt > best_count:
                best_count = cnt
                best_col = col
        except Exception:
            continue
    if best_col and best_count > max(1, len(df) * 0.1):
        return best_col
    return None

def preprocess(path_in, path_out):
    # Robust CSV loading: handle files with extra top rows or non-date header lines
    # Read without parsing first
    df_raw = pd.read_csv(path_in, header=0, dtype=str, engine="python")
    if df_raw.empty:
        raise RuntimeError("Raw CSV empty")
    # If first column header is something like "Ticker" or file has meta rows, attempt to drop non-date rows
    # Try detect date column
    date_col = _identify_date_column(df_raw)
    if date_col is None:
        # try using index as date: if first column name looks like date or is 'Unnamed: 0'
        first_col = df_raw.columns[0]
        # Attempt to parse first column to datetime
        parsed = pd.to_datetime(df_raw[first_col], errors="coerce")
        if parsed.notna().sum() >= 1:
            df_raw[first_col] = parsed
            df_raw = df_raw.loc[parsed.notna()].copy()
            df_raw.set_index(first_col, inplace=True)
        else:
            # If still no date, try to read csv again with infer_datetime_format on index
            df_try = pd.read_csv(path_in, index_col=0, parse_dates=True, infer_datetime_format=True, engine="python")
            # coerce index to datetime, drop invalid
            try:
                idx = pd.to_datetime(df_try.index, errors="coerce")
                df_try = df_try.loc[idx.notna()].copy()
                df_try.index = idx[idx.notna()]
                df_try.to_csv(path_out)  # fallback save
                print("[preprocess] Warning: used fallback read and saved.")
                return path_out
            except Exception as e:
                raise RuntimeError(f"Could not identify date column/index automatically: {e}")
    else:
        # If date column detected, set as index parsed
        # If date_col is None returned previously indicating index is date, we handle above.
        print(f"[preprocess] Identified date column: {date_col}")
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.loc[df_raw[date_col].notna()].copy()
        df_raw.set_index(date_col, inplace=True)

    # Now we have df_raw indexed by dates (strings coerced to datetime)
    # Try to convert numeric columns to floats and look for Close column
    # If Close not present, try 'Adj Close', or pick first numeric column
    numeric_cols = []
    for c in df_raw.columns:
        # try convert
        try:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
            if df_raw[c].notna().sum() > 0:
                numeric_cols.append(c)
        except Exception:
            continue
    preferred_close = None
    for cand in ("Close", "close", "Adj Close", "Adj_Close", "adjclose"):
        if cand in df_raw.columns:
            preferred_close = cand
            break
    if not preferred_close:
        if numeric_cols:
            preferred_close = numeric_cols[0]
            print(f"[preprocess] Warning: 'Close' not found. Using first numeric column: {preferred_close}")
        else:
            raise RuntimeError("No numeric price column found in raw CSV")

    s = df_raw[[preferred_close]].rename(columns={preferred_close: "y"}).sort_index()
    # Ensure index is DatetimeIndex
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].copy()
    if s.empty:
        raise RuntimeError("After parsing dates, no valid rows remain.")
    # Resample to daily frequency and forward-fill
    s = s.asfreq("D")
    s["y"] = s["y"].ffill()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(path_out)
    return path_out

def create_features(df, lags=(1,2,3,7,14,30), windows=(3,7,14)):
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = out["y"].shift(l)
    for w in windows:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["y"].shift(1).rolling(w).std()
    return out

def choose_model():
    try:
        import lightgbm as lgb  # noqa
        return "lightgbm"
    except Exception:
        return "random_forest"

def train_model(name, X_train, y_train, X_val=None, y_val=None):
    if name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
        try:
            if X_val is not None and len(X_val) > 0:
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            else:
                m.fit(X_train, y_train)
        except Exception:
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
            m.fit(X_train, y_train)
        return m
    else:
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=200, random_state=42)
        m.fit(X_train, y_train)
        return m

def iterative_forecast(model, history_df, steps=30):
    cur = history_df.copy().asfreq("D")
    preds = []
    for i in range(steps):
        feat = create_features(cur).iloc[[-1]].drop(columns=["y"], errors="ignore")
        feat = feat.fillna(method="ffill").fillna(method="bfill").fillna(0)
        try:
            p = float(model.predict(feat)[0])
        except Exception:
            p = float(model.predict(feat.values)[0])
        next_date = cur.index[-1] + pd.Timedelta(days=1)
        preds.append((next_date, p))
        cur.loc[next_date] = [p]
    return pd.DataFrame({"pred": [v for (_,v) in preds]}, index=[d for (d,_) in preds])

def rolling_backtest(series_df, model_name, horizon=BACKTEST_HORIZON, step=BACKTEST_STEP, min_train_days=INITIAL_TRAIN_YEARS*365):
    df = series_df.copy().asfreq("D")
    n = len(df)
    results = []
    models_saved = []
    if n <= min_train_days + horizon:
        raise RuntimeError("Series too short for requested backtest configuration.")
    train_end_idx = min_train_days
    while train_end_idx + horizon <= n:
        train = df.iloc[:train_end_idx]
        test = df.iloc[train_end_idx: train_end_idx + horizon]
        train_feat = create_features(train).dropna()
        if len(train_feat) < 5:
            train_end_idx += step
            continue
        X_train = train_feat.drop(columns=["y"])
        y_train = train_feat["y"]
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
        model = train_model(model_name, X_tr, y_tr, X_val, y_val)
        fc = iterative_forecast(model, train, steps=len(test))
        y_true = test["y"].iloc[:len(fc)]
        y_pred = fc["pred"].values[:len(y_true)]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        model_file = MODELS / f"{model_name}_window_{train_end_idx}_{TIMESTAMP}.pkl"
        joblib.dump(model, model_file)
        models_saved.append(str(model_file))
        results.append({"window_end_index": int(train_end_idx), "window_end_date": str(train.index[train_end_idx-1].date()),
                        "mae": float(mae), "rmse": float(rmse), "n_train": len(train), "n_test": len(test),
                        "model_file": str(model_file)})
        train_end_idx += step
    return pd.DataFrame(results), models_saved

# RUN pipeline
try:
    print("1) Download")
    raw_path = download_data(TICKER, START, END)
    print("2) Preprocess")
    processed_path = DATA_PROC / f"{TICKER}_parsed.csv"
    preprocess(raw_path, processed_path)
    series = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    series = series.sort_index().asfreq("D")
    series["y"] = series["y"].ffill()
    model_choice = choose_model()
    metrics_df = pd.DataFrame()
    try:
        print("3) Backtest")
        metrics_df, saved_models = rolling_backtest(series, model_choice)
        metrics_path = RESULTS / f"{TICKER}_backtest_metrics_{TIMESTAMP}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print("Backtest metrics saved:", metrics_path)
    except Exception as e:
        print("Backtest skipped or failed:", e)
    print("4) Final training on full data")
    df_feat = create_features(series).dropna()
    X = df_feat.drop(columns=["y"])
    y = df_feat["y"]
    final_model = train_model(model_choice, X, y)
    final_model_file = MODELS / f"{model_choice}_{TICKER}_final_{TIMESTAMP}.pkl"
    joblib.dump(final_model, final_model_file)
    print("Final model saved:", final_model_file)
    print("5) Forecast")
    fc = iterative_forecast(final_model, series, steps=HORIZON)
    forecast_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.csv"
    combined = pd.concat([series.rename(columns={"y":"actual"}), fc.rename(columns={"pred":"forecast"})], axis=0)
    combined.to_csv(forecast_path)
    print("Forecast saved:", forecast_path)
    run_info = {"ticker":TICKER, "start":START, "end":END, "horizon":HORIZON, "model_used":model_choice, "timestamp":TIMESTAMP}
    run_info_path = RESULTS / f"run_info_{TIMESTAMP}.json"
    run_info_path.write_text(json.dumps(run_info, indent=2))
    print("Run info saved:", run_info_path)
    # Report
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["y"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["pred"], name="Forecast", line=dict(color="red")))
    fig.update_layout(title=f"{TICKER} Actual vs Forecast", height=600)
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    html = "<html><head><meta charset=\'utf-8\'></head><body>"
    html += f"<h2>Report {TICKER}</h2><p>Generated: {datetime.utcnow().isoformat()}</p>"
    html += plot_html
    if 'metrics_df' in locals() and not metrics_df.empty:
        html += "<h3>Backtest metrics</h3>" + metrics_df.to_html(index=False)
    report_path = REPORTS / f"report_{TICKER}_{TIMESTAMP}.html"
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print("Report saved:", report_path)
except Exception as e:
    print("Pipeline failed:", e)
    traceback.print_exc()
    sys.exit(1)

print("=== Done ===")
