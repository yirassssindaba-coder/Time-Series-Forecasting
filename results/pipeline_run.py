# pipeline_run.py
# Pipeline runs with ROOT = current working directory (so outputs are inside the results folder)
import sys, subprocess, json, warnings, traceback
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

ROOT = Path.cwd()   # important: pipeline will write all outputs under the folder where this script is run
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT
REPORTS = RESULTS / "notebooks_html"
TEMPLATE = ROOT / "report_template.html"

for d in (DATA_RAW, DATA_PROC, MODELS, RESULTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

def pip_install(pkg):
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return True
    except Exception:
        return False

# Best-effort: try install kaleido for PNG export and jinja2 for template rendering
pip_install("kaleido")
pip_install("jinja2")

# Imports (fail clearly if not present)
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
    import plotly.graph_objects as go
    import plotly.io as pio
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except Exception as e:
    print("Missing Python packages. Install: pandas, numpy, yfinance, scikit-learn, joblib, plotly, jinja2 (kaleido optional).")
    print("Error:", e)
    sys.exit(1)

# Config (ubah jika perlu)
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

def _identify_date_column(df):
    for col in df.columns:
        if col.lower() in ("date","day","timestamp","ds","index"):
            return col
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().sum() > 0 and idx.notna().mean() > 0.5:
            return None
    except Exception:
        pass
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
    if best_col and best_count > max(1, len(df)*0.1):
        return best_col
    return None

def preprocess(path_in, path_out):
    df_raw = pd.read_csv(path_in, header=0, dtype=str, engine="python")
    if df_raw.empty:
        raise RuntimeError("Raw CSV empty")
    date_col = _identify_date_column(df_raw)
    if date_col is None:
        first_col = df_raw.columns[0]
        parsed = pd.to_datetime(df_raw[first_col], errors="coerce")
        if parsed.notna().sum() >= 1:
            df_raw[first_col] = parsed
            df_raw = df_raw.loc[parsed.notna()].copy()
            df_raw.set_index(first_col, inplace=True)
        else:
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

def train_final_model(series_df):
    df_feat = create_features(series_df).dropna()
    X = df_feat.drop(columns=["y"])
    y = df_feat["y"]
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
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
    return pd.DataFrame({"forecast":[v for (_,v) in preds]}, index=[d for (d,_) in preds])

def generate_report(fig, metrics_df, run_info, out_path):
    env = Environment(loader=FileSystemLoader(searchpath=str(TEMPLATE.parent)), autoescape=select_autoescape())
    tmpl = env.get_template(TEMPLATE.name) if TEMPLATE.exists() else None
    fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    metrics_html = metrics_df.to_html(index=False, float_format="%.4f") if metrics_df is not None and not metrics_df.empty else ""
    explanation = ("Grafik menampilkan nilai actual (historis) pada garis hitam dan perkiraan (forecast) pada garis merah. "
                   "Forecast dibuat secara iteratif berdasarkan fitur lag dan rolling mean/std.")
    if tmpl:
        html = tmpl.render(title=f"Forecast report - {run_info.get('ticker')}", run_info=run_info, fig_html=fig_html, metrics_html=metrics_html, explanation=explanation)
    else:
        html = f"<html><head><meta charset='utf-8'></head><body><h1>Forecast report - {run_info.get('ticker')}</h1><p>Generated: {run_info.get('timestamp')}</p>"
        html += fig_html
        if metrics_html:
            html += "<h2>Backtest metrics</h2>" + metrics_html
        html += f"<h2>Explanation</h2><p>{explanation}</p></body></html>"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("[report] Written:", out_path)

# Run pipeline
try:
    print("1) Download")
    raw_path = download_data(TICKER, START, END)

    print("2) Preprocess")
    processed_path = DATA_PROC / f"{TICKER}_parsed.csv"
    preprocess(raw_path, processed_path)

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
    combined = pd.concat([series.rename(columns={"y":"actual"}), fc], axis=0)
    combined.to_csv(forecast_path)
    print("Forecast saved:", forecast_path)

    run_info = {"ticker":TICKER,"start":START,"end":END,"horizon":HORIZON,"model_used":model_name,"timestamp":TIMESTAMP}
    run_info_path = RESULTS / f"run_info_{TIMESTAMP}.json"
    run_info_path.write_text(json.dumps(run_info, indent=2))
    print("Run info saved:", run_info_path)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["y"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color="red")))
    fig.update_layout(title=f"{TICKER} Actual vs Forecast", xaxis_title="Date", yaxis_title="Price", height=600)

    try:
        img_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.png"
        fig.write_image(str(img_path))
        print("Saved plot PNG:", img_path)
    except Exception as e:
        print("PNG save skipped (kaleido may be missing). To enable PNG export run: pip install kaleido")
        print("PNG error:", e)

    report_path = REPORTS / f"report_{TICKER}_{TIMESTAMP}.html"
    # create minimal template if not exists
    if not TEMPLATE.exists():
        TEMPLATE.write_text("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{ title }}</title>
<style>body{font-family:Segoe UI, Roboto, Arial, sans-serif;margin:20px;color:#222;} .card{background:#fff;border:1px solid #e6e6e6;padding:16px;border-radius:8px;margin-bottom:16px;} h1,h2{color:#0b2545;} .footer{font-size:0.9rem;color:#555;margin-top:20px;}</style>
</head><body>
<div class=\"card\"><h1>{{ title }}</h1><p>Generated: {{ run_info.timestamp }}</p></div>
<div class=\"card\"><h2>Chart</h2>{{ fig_html | safe }}</div>
<div class=\"card\"><h2>Notes</h2><p>{{ explanation }}</p></div>
<div class=\"footer\">Report generated by pipeline_run.py</div>
</body></html>""", encoding='utf-8')

    metrics_df = pd.DataFrame()
    generate_report(fig, metrics_df, run_info, report_path)

    print("=== Done ===")
except Exception as e:
    print("Pipeline failed:", e)
    traceback.print_exc()
    sys.exit(1)
