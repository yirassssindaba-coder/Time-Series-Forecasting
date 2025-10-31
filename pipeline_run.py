# pipeline_run.py
# Save location: project root (C:\Users\ASUS\Desktop\time-series-forecasting\pipeline_run.py)
# Use: run with python from appropriate environment: python pipeline_run.py
import sys, subprocess, json, warnings, traceback
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\ASUS\Desktop\time-series-forecasting")
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
        print(f"[pip] Installing {pkg} ... (this may take some time)")
        try:
            proc = subprocess.run([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout[:2000])
        except Exception as e:
            print(f"[pip] install failed for {pkg}: {e}")

# best-effort installs (will not stop pipeline if heavy packages fail)
pip_install(["pandas>=1.5", "numpy", "yfinance", "scikit-learn", "plotly", "joblib"])
pip_install(["lightgbm", "prophet", "statsmodels"])

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import plotly.graph_objects as go
import plotly.io as pio

# CONFIG
TICKER = "AAPL"
START = "2015-01-01"
END = "2024-01-01"
HORIZON = 30
TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def download_data(ticker, start, end, out_dir=DATA_RAW):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.csv"
    print(f"[download] {ticker} {start}..{end}")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Downloaded data empty. Check ticker/date range or internet.")
    df.to_csv(path)
    return path

def preprocess(path_in, path_out):
    # robust loading: handle CSV with metadata rows or unexpected headers
    df_raw = pd.read_csv(path_in, header=0, dtype=str, engine="python")
    if df_raw.empty:
        raise RuntimeError("Raw CSV empty")
    # detect date column or use first column if parseable
    def detect_date_col(df):
        for col in df.columns:
            if col.lower() in ("date","day","timestamp","ds","index"):
                return col
        # try index detection
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            if idx.notna().sum() > 0 and idx.notna().mean() > 0.5:
                return None
        except Exception:
            pass
        # find column with many parseable datetimes
        best_col, best_count = None, 0
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            cnt = parsed.notna().sum()
            if cnt > best_count:
                best_count = cnt
                best_col = col
        if best_col and best_count > max(1, len(df)*0.1):
            return best_col
        return None

    date_col = detect_date_col(df_raw)
    if date_col is None:
        first_col = df_raw.columns[0]
        parsed = pd.to_datetime(df_raw[first_col], errors="coerce")
        if parsed.notna().sum() >= 1:
            df_raw[first_col] = parsed
            df_raw = df_raw.loc[parsed.notna()].copy()
            df_raw.set_index(first_col, inplace=True)
        else:
            # fallback: try reading CSV with index_col=0 parse_dates
            df_try = pd.read_csv(path_in, index_col=0, parse_dates=True, infer_datetime_format=True, engine="python")
            idx = pd.to_datetime(df_try.index, errors="coerce")
            df_try = df_try.loc[idx.notna()].copy()
            df_try.index = idx[idx.notna()]
            df_try.to_csv(path_out)
            print("[preprocess] Fallback used and saved.")
            return path_out
    else:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.loc[df_raw[date_col].notna()].copy()
        df_raw.set_index(date_col, inplace=True)

    # numeric column detection and choose Close if present
    numeric_cols = []
    for c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        if df_raw[c].notna().sum() > 0:
            numeric_cols.append(c)
    preferred = None
    for cand in ("Close","close","Adj Close","Adj_Close","adjclose"):
        if cand in df_raw.columns:
            preferred = cand
            break
    if not preferred:
        if numeric_cols:
            preferred = numeric_cols[0]
            print(f"[preprocess] 'Close' not found. Using: {preferred}")
        else:
            raise RuntimeError("No numeric price column found")
    s = df_raw[[preferred]].rename(columns={preferred:"y"}).sort_index()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].copy()
    if s.empty:
        raise RuntimeError("No valid rows after parsing dates")
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
        import lightgbm as lgb
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
    return pd.DataFrame({"pred":[v for (_,v) in preds]}, index=[d for (d,_) in preds])

# RUN
try:
    print("1) Download")
    raw = download_data(TICKER, START, END)
    print("2) Preprocess")
    processed = DATA_PROC / f"{TICKER}_parsed.csv"
    preprocess(raw, processed)
    series = pd.read_csv(processed, index_col=0, parse_dates=True)
    series = series.sort_index().asfreq("D")
    series["y"] = series["y"].ffill()
    model_choice = choose_model()
    print("Model chosen:", model_choice)
    print("3) Train final model")
    df_feat = create_features(series).dropna()
    X = df_feat.drop(columns=["y"])
    y = df_feat["y"]
    final_model = train_model(model_choice, X, y)
    final_model_file = MODELS / f"{model_choice}_{TICKER}_final_{TIMESTAMP}.pkl"
    joblib.dump(final_model, final_model_file)
    print("Final model saved:", final_model_file)
    print("4) Forecast")
    fc = iterative_forecast(final_model, series, steps=HORIZON)
    forecast_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.csv"
    combined = pd.concat([series.rename(columns={"y":"actual"}), fc.rename(columns={"pred":"forecast"})], axis=0)
    combined.to_csv(forecast_path)
    print("Forecast saved:", forecast_path)
    run_info = {"ticker":TICKER,"start":START,"end":END,"horizon":HORIZON,"model_used":model_choice,"timestamp":TIMESTAMP}
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
    report_path = REPORTS / f"report_{TICKER}_{TIMESTAMP}.html"
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print("Report saved:", report_path)
except Exception as e:
    print("Pipeline failed:", e)
    traceback.print_exc()
    sys.exit(1)

print("=== Done ===")
