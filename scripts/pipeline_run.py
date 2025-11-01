# pipeline_run.py
# Robust pipeline that writes all outputs into the current working directory.
# Run this from the folder where you want results saved (e.g. results/).
# Usage:
#   python pipeline_run.py
#
# Requirements (install if missing):
#   pip install pandas numpy yfinance scikit-learn joblib plotly jinja2

import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# Use timezone-aware UTC timestamp (avoids DeprecationWarning)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ROOT is the directory where this script runs (so run it from results folder)
ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "notebooks_html"
RESULTS = ROOT

for d in (DATA_RAW, DATA_PROC, MODELS, REPORTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

def download_data(ticker, start, end, out_dir=DATA_RAW):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.csv"
    print(f"[download] {ticker} {start}..{end}")
    try:
        # specify auto_adjust to silence future warning and get adjusted prices if desired
        df = __import__("yfinance").download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")
    if df is None or df.empty:
        raise RuntimeError("Downloaded data empty. Check ticker/date range or internet.")
    # write CSV (use index_label so rebuild is consistent)
    df.to_csv(path, index=True)
    return df, path

def _identify_date_column_df(df):
    # returns None -> use index; or column name string to use as date column
    # common names first
    for col in df.columns:
        if str(col).lower() in ("date", "day", "timestamp", "ds"):
            return col
    # check index
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().sum() / max(1, len(idx)) > 0.5:
            return None
    except Exception:
        pass
    # try any column with many parseable datetimes
    best_col, best_count = None, 0
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

def preprocess_df_or_path(df_or_path, out_path):
    import pandas as pd
    # accepts either a DataFrame or a path to CSV
    if isinstance(df_or_path, pd.DataFrame):
        df_raw = df_or_path.copy()
    else:
        df_raw = pd.read_csv(df_or_path, header=0, dtype=str, engine="python")
    if df_raw.empty:
        raise RuntimeError("Raw CSV empty")
    # If DataFrame with index name 'Date' etc, keep index; otherwise try detect date column
    date_col = _identify_date_column_df(df_raw)
    if date_col is None:
        # use index as datetime (coerce)
        try:
            idx = pd.to_datetime(df_raw.index, errors="coerce")
            valid = idx.notna().sum()
            if valid / max(1, len(idx)) > 0.1:
                df_raw.index = idx
            else:
                # try first column
                first_col = df_raw.columns[0]
                parsed = pd.to_datetime(df_raw[first_col], errors="coerce")
                if parsed.notna().sum() >= 1:
                    df_raw[first_col] = parsed
                    df_raw = df_raw.loc[parsed.notna()].copy()
                    df_raw.set_index(first_col, inplace=True)
                else:
                    # as last resort, try reading again with index_col=0 parse_dates
                    df_try = pd.read_csv(df_or_path, index_col=0, parse_dates=True, infer_datetime_format=True, engine="python")
                    idx2 = pd.to_datetime(df_try.index, errors="coerce")
                    df_try = df_try.loc[idx2.notna()].copy()
                    df_try.index = idx2[idx2.notna()]
                    df_try.to_csv(out_path)
                    print("[preprocess] Fallback used and saved.")
                    return out_path
        except Exception:
            # fallback
            df_try = pd.read_csv(df_or_path, index_col=0, parse_dates=True, infer_datetime_format=True, engine="python")
            idx2 = pd.to_datetime(df_try.index, errors="coerce")
            df_try = df_try.loc[idx2.notna()].copy()
            df_try.index = idx2[idx2.notna()]
            df_try.to_csv(out_path)
            print("[preprocess] Fallback used and saved.")
            return out_path
    else:
        # date_col is a column name -> set as index
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.loc[df_raw[date_col].notna()].copy()
        df_raw.set_index(date_col, inplace=True)

    # convert numeric columns and find price column
    numeric_cols = []
    for c in df_raw.columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
            if df_raw[c].notna().sum() > 0:
                numeric_cols.append(c)
        except Exception:
            continue
    preferred = None
    for cand in ("Close", "close", "Adj Close", "Adj_Close", "adjclose"):
        if cand in df_raw.columns:
            preferred = cand
            break
    if not preferred:
        if numeric_cols:
            preferred = numeric_cols[0]
            print(f"[preprocess] 'Close' not found. Using: {preferred}")
        else:
            raise RuntimeError("No numeric price column found")
    s = df_raw[[preferred]].rename(columns={preferred: "y"}).sort_index()
    # ensure datetime index
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].copy()
    if s.empty:
        raise RuntimeError("After parsing dates, no valid rows remain.")
    # resample to daily frequency and forward-fill
    try:
        s = s.asfreq("D")
    except Exception:
        # if asfreq fails, ensure index is sorted and create date_range
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    s["y"] = s["y"].ffill()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_path)
    return out_path

def create_features(df, lags=(1,2,3,7,14,30), windows=(3,7,14)):
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = out["y"].shift(l)
    for w in windows:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["y"].shift(1).rolling(w).std()
    return out

def train_final_model(series_df):
    df_feat = create_features(series_df).dropna()
    X = df_feat.drop(columns=["y"])
    y = df_feat["y"]
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        m.fit(X, y)
        name = "lightgbm"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=200, random_state=42)
        m.fit(X, y)
        name = "random_forest"
    return m, name

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
    return pd.DataFrame({"forecast": [v for (_, v) in preds]}, index=[d for (d, _) in preds])

# RUN pipeline
if __name__ == "__main__":
    try:
        import pandas as pd
        TICKER = "AAPL"
        START = "2015-01-01"
        END = "2024-01-01"
        HORIZON = 30

        print("1) Download")
        df_raw, raw_path = download_data(TICKER, START, END)

        print("2) Preprocess")
        processed_path = DATA_PROC / f"{TICKER}_parsed.csv"
        preprocess_df_or_path(df_raw, processed_path)

        series = pd.read_csv(processed_path, index_col=0, parse_dates=True).sort_index().asfreq("D")
        series["y"] = series["y"].ffill()

        print("3) Train final model")
        model, model_name = train_final_model(series)
        model_file = MODELS / f"{model_name}_{TICKER}_final_{TIMESTAMP}.pkl"
        joblib.dump(model, model_file)
        print("Model saved:", model_file)

        print("4) Forecast")
        fc = iterative_forecast(model, series, steps=HORIZON)
        forecast_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.csv"
        combined = pd.concat([series.rename(columns={"y": "actual"}), fc.rename(columns={"forecast": "forecast"})], axis=0)
        combined.to_csv(forecast_path)
        print("Forecast saved:", forecast_path)

        run_info = {"ticker": TICKER, "start": START, "end": END, "horizon": HORIZON, "model_used": model_name, "timestamp": TIMESTAMP}
        run_info_path = RESULTS / f"run_info_{TIMESTAMP}.json"
        run_info_path.write_text(json.dumps(run_info, indent=2))
        print("Run info saved:", run_info_path)

        # Build Plotly figure and report
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series["y"], name="Actual", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color="red")))
        fig.update_layout(title=f"{TICKER} Actual vs Forecast", xaxis_title="Date", yaxis_title="Price", height=600)

        # Try save PNG (kaleido optional)
        try:
            img_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.png"
            fig.write_image(str(img_path))
            print("Saved plot PNG:", img_path)
        except Exception as e:
            print("PNG save skipped (kaleido missing or other issue):", e)

        report_path = REPORTS / f"report_{TICKER}_{TIMESTAMP}.html"
        metrics_df = pd.DataFrame()
        generate_report(fig, metrics_df, run_info, report_path)

        print("=== Done ===")
    except Exception as e:
        print("Pipeline failed:", e)
        traceback.print_exc()
        sys.exit(1)