```markdown
# Rencana Pengerjaan — Pipeline & Single-dashboard

Ringkasan (tujuan)
- Membuat sebuah skrip PowerShell (run_project_and_run_fixed.ps1) yang:
  - Membuat struktur proyek bila belum ada.
  - Menulis file Python `hasil/pipeline_run.py` yang lengkap.
  - Menjalankan pipeline untuk memproses semua CSV di folder data/, menghasilkan:
    - Satu file HTML interaktif (single dashboard) di hasil/notebooks_html/report_dashboard_{TIMESTAMP}.html
    - Satu file JSON ringkasan di hasil/notebooks_html/summaries_{TIMESTAMP}.json
  - Opsional: inisialisasi Git dan commit scaffold.

Fitur utama pipeline (hasil/pipeline_run.py)
- Deteksi tipe file (time_series, ratings, items, generic table).
- Untuk time-series:
  - Canonicalize tanggal, resample daily, isi missing dengan ffill.
  - Buat fitur lag/rolling.
  - Latih model RandomForest (atau LightGBM jika tersedia).
  - Lakukan forecast iteratif untuk horizon (default 30).
  - Hitung pseudo-evaluation (MAE, RMSE) menggunakan pseudo-holdout.
  - Simpan fig Plotly (serialized JSON), table preview (editable).
  - Sertakan penjelasan otomatis (MAE/RMSE, trend vs recent_mean).
- Untuk ratings/items:
  - Jika modul `src.recommender.models.cf.UserKNN` atau `src.recommender.models.content.ContentRecommender` tersedia, latih dan buat artifact sample.
  - Jika tidak tersedia, tetap tampilkan table preview.
- Dashboard:
  - Single-page HTML yang memuat semua series/entitas.
  - Per-series interactive Plotly plots (actual vs forecast), editable table (contenteditable), tombol Add/Delete row, Download CSV/JSON.
  - Overview dengan ringkasan metrik dan link.
- Persistence:
  - Saat ini edits di UI disimpan di browser memori; gunakan tombol Download untuk menyimpan lokal.
  - Opsi lanjutan: tambahkan Flask server untuk POST edits sehingga bisa menulis ke disk.

Alur penggunaan (user)
1. Pastikan Python environment: pip install pandas numpy scikit-learn joblib plotly (opsional lightgbm).
2. Jalankan PowerShell:
   pwsh -ExecutionPolicy Bypass -File .\run_project_and_run_fixed.ps1 -ProjectDir "C:\path\to\repo" -RunNow:$true
3. Buka hasil/notebooks_html/report_dashboard_{TIMESTAMP}.html di browser.
4. Interaksi:
   - Klik file di sidebar → lihat plot + table.
   - Edit table di browser → Download CSV/JSON → replace data source → rerun pipeline.
   - Untuk menyimpan langsung ke disk otomatis, minta tambahan Flask server.

Interpretasi hasil dan visualisasi penting
- Pie chart untuk distribusi tipe file (time_series/ratings/items/table).
- Histogram MAE untuk menilai kualitas model across series.
- Scatter actual vs forecast untuk memeriksa bias/lag.
- Bar chart top recommended/similarity frequency (ratings/items).
- Heatmap sparsity untuk matrix ratings (menentukan strategi CF).

Catatan keamanan dan izin
- Menyimpan perubahan dari UI langsung ke filesystem memerlukan server (Flask) dan izin write.
- Hati-hati menulis/commit file otomatis — pastikan backup data asli.

Jika Anda ingin, saya akan:
- Tambahkan Flask endpoint agar tombol Save di UI bisa mengirim POST sehingga file ditulis ke hasil/results secara otomatis dan (opsional) commit ke Git.
- Buat skrip analisis tambahan (analyze_summaries.py) untuk menghasilkan semua visual tambahan secara otomatis dari summaries JSON.

```
```

```powershell name=run_project_and_run_fixed.ps1
<#
run_project_and_run_fixed.ps1

Creates a scaffold (if missing), writes a robust Python pipeline into hasil/pipeline_run.py (single-dashboard generator),
then optionally runs the pipeline. The pipeline produces:
 - One interactive HTML dashboard: hasil/notebooks_html/report_dashboard_{TIMESTAMP}.html
 - One summary JSON: hasil/notebooks_html/summaries_{TIMESTAMP}.json

Usage:
  pwsh -ExecutionPolicy Bypass -File .\run_project_and_run_fixed.ps1
  pwsh -ExecutionPolicy Bypass -File .\run_project_and_run_fixed.ps1 -ProjectDir "C:\path\to\repo" -RunNow:$true -InitializeGit

Parameters:
  -ProjectDir   : project root (defaults to current directory)
  -RunNow       : run pipeline after writing files (default = $true)
  -InitializeGit: initialize git and commit scaffold (optional)

What this script does (narrative):
- I will create common folders (src, data, hasil, hasil/notebooks_html, results, models).
- I will write a complete Python pipeline file into hasil/pipeline_run.py that handles all processing and writes a single dashboard + JSON summary.
- If requested, I will run the pipeline and show logs. The pipeline code includes defensive handling for optional project models (UserKNN and ContentRecommender), so it works even if those modules are absent.

Now writing the pipeline file and (optionally) running it.
#>

param(
  [string] $ProjectDir = (Get-Location).ProviderPath,
  [switch] $RunNow = $true,
  [switch] $InitializeGit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    Write-Host "Created directory: $Path"
  }
}

function Write-FileSafely {
  param(
    [Parameter(Mandatory=$true)][string] $Path,
    [Parameter(Mandatory=$true)][string] $Content,
    [ValidateSet("utf8","utf8bom","unicode")][string] $EncodingName = "utf8"
  )
  $dir = Split-Path -Path $Path -Parent
  if ($dir -and -not (Test-Path -LiteralPath $dir)) { Ensure-Dir -Path $dir }
  try {
    switch ($EncodingName.ToLower()) {
      "utf8bom" { $enc = New-Object System.Text.UTF8Encoding $true }
      "unicode" { $enc = [System.Text.Encoding]::Unicode }
      default  { $enc = [System.Text.Encoding]::UTF8 }
    }
    [System.IO.File]::WriteAllText($Path, $Content, $enc)
    Write-Host "Wrote: $Path"
  } catch { Write-Warning "Failed writing $Path : $_" }
}

function Find-Python {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { return $py.Path }
  $py3 = Get-Command python3 -ErrorAction SilentlyContinue
  if ($py3) { return $py3.Path }
  return "python"
}

# Resolve project root and create folders
try {
  $proj = Resolve-Path -LiteralPath $ProjectDir -ErrorAction Stop
  $root = $proj.Path.TrimEnd('\','/')
} catch {
  Write-Error "Invalid ProjectDir: $ProjectDir"
  throw
}

$dirs = @(
  $root,
  (Join-Path $root "src"),
  (Join-Path $root "data"),
  (Join-Path $root "data\sample"),
  (Join-Path $root "hasil"),
  (Join-Path $root "hasil\notebooks_html"),
  (Join-Path $root "hasil\results"),
  (Join-Path $root "hasil\models")
)
foreach ($d in $dirs) { Ensure-Dir -Path $d }

$hasil = Join-Path $root "hasil"
$notebooks = Join-Path $hasil "notebooks_html"

# Embedded pipeline_run.py content (complete)
$pipeline_py = @'
#!/usr/bin/env python3
"""
hasil/pipeline_run.py

Single-dashboard generator (fixed). See Rencana_Pipeline.md for full plan.

Generates:
 - One interactive HTML dashboard: hasil/notebooks_html/report_dashboard_{TIMESTAMP}.html
 - One summary JSON: hasil/notebooks_html/summaries_{TIMESTAMP}.json

Requirements:
  pip install pandas numpy scikit-learn joblib plotly
"""
from pathlib import Path
from datetime import datetime, timezone
import json, traceback, sys, math, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

# Plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Ensure optional classes exist in global scope
UserKNN = None
ContentRecommender = None
try:
    from src.recommender.models.cf import UserKNN as _UserKNN
    UserKNN = _UserKNN
    print("Imported UserKNN from src.recommender.models.cf")
except Exception as e:
    UserKNN = None
    print("UserKNN not available (optional):", e)

try:
    from src.recommender.models.content import ContentRecommender as _ContentRecommender
    ContentRecommender = _ContentRecommender
    print("Imported ContentRecommender from src.recommender.models.content")
except Exception as e:
    ContentRecommender = None
    print("ContentRecommender not available (optional):", e)

# Paths
HASIL = Path(__file__).resolve().parent
ROOT = HASIL.parent
DATA_FOLDERS = [ROOT / "data", ROOT / "data" / "raw", ROOT / "data" / "processed"]
REPORTS_DIR = HASIL / "notebooks_html"
RESULTS_DIR = HASIL / "results"
MODELS_DIR = HASIL / "models"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

for d in (REPORTS_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Utilities
def find_csv_files():
    files = []
    for base in DATA_FOLDERS:
        if base.exists():
            files.extend(list(base.rglob("*.csv")))
    return sorted(set(files))

def detect_csv_type(df):
    cols = [c.lower() for c in df.columns]
    date_candidates = [c for c in df.columns if c.lower() in ("date","ds","timestamp","time")]
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    if date_candidates and numeric_cols:
        try:
            parsed = pd.to_datetime(df[date_candidates[0]], errors="coerce")
            if parsed.notna().sum() >= 8:
                # preserve original case column names
                date_col = [c for c in df.columns if c.lower() == date_candidates[0].lower()][0]
                return ("time_series", date_col, numeric_cols[0])
        except Exception:
            pass
    if {"user_id","item_id","rating"}.issubset(set(cols)):
        return ("ratings", None, None)
    if "item_id" in cols and any(x in cols for x in ("text","title","description")):
        for t in ("text","description","title"):
            if t in cols:
                real_text = [c for c in df.columns if c.lower() == t][0]
                return ("items", None, real_text)
    return ("table", None, None)

def canonicalize_series(df, date_col, value_col):
    df2 = df[[date_col, value_col]].copy()
    df2.columns = ["date","value"]
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.loc[df2["date"].notna()]
    if df2.empty:
        return df2
    df2 = df2.set_index("date").sort_index()
    try:
        s = df2.resample("D").mean()
    except Exception:
        idx = pd.date_range(start=df2.index.min(), end=df2.index.max(), freq="D")
        s = df2.reindex(idx)
    s["value"] = s["value"].ffill()
    return s

def create_features(series):
    df = series.copy()
    df["lag_1"] = df["value"].shift(1)
    df["lag_7"] = df["value"].shift(7)
    df["roll_7"] = df["value"].shift(1).rolling(7).mean()
    df = df.dropna()
    return df

def train_model(X, y):
    if HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(n_estimators=200)
        model.fit(X, y)
        return model, "lightgbm"
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, "random_forest"

def iterative_forecast(model, history_df, steps=30):
    cur = history_df.copy().asfreq("D")
    preds = []
    for _ in range(steps):
        feat = create_features(cur).iloc[[-1]].drop(columns=["value"], errors="ignore").fillna(0)
        try:
            p = float(model.predict(feat)[0])
        except Exception:
            p = float(model.predict(feat.values)[0])
        nxt = cur.index[-1] + pd.Timedelta(days=1)
        preds.append((nxt, p))
        cur.loc[nxt] = [p]
    return pd.DataFrame({"forecast":[v for (_,v) in preds]}, index=[d for (d,_) in preds])

def auto_explanation(summary):
    if not summary or summary.get("type") != "time_series":
        return ""
    mae = summary.get("mae")
    rmse = summary.get("rmse")
    recent = summary.get("recent_mean", float("nan"))
    forecast_mean = summary.get("forecast_mean", float("nan"))
    n = summary.get("n_points", 0)
    parts = []
    if mae is not None:
        parts.append(f"MAE = {mae:.3f}")
    if rmse is not None:
        parts.append(f"RMSE = {rmse:.3f}")
    parts.append(f"Data points: {n}")
    try:
        if not math.isnan(recent) and not math.isnan(forecast_mean):
            if forecast_mean > recent * 1.02:
                parts.append("Model predicts an increase vs recent average (>2%).")
            elif forecast_mean < recent * 0.98:
                parts.append("Model predicts a decrease vs recent average (>2%).")
            else:
                parts.append("Model predicts values similar to recent average (within ±2%).")
    except Exception:
        pass
    parts.append("Model is a simple tree regressor — inspect plots and table; consider more advanced models for production.")
    return " ".join(parts)

# Processors
def process_time_series(path: Path, date_col, value_col, horizon=30, min_points=30):
    df = pd.read_csv(path, engine="python")
    s = canonicalize_series(df, date_col, value_col)
    if s.empty or s.shape[0] < min_points:
        return {"type":"time_series","name":path.stem,"skipped":True,"n_points":int(s.shape[0] if not s.empty else 0)}
    horizon_eval = min(horizon, max(1, int(s.shape[0]*0.1)))
    train_series = s.iloc[:-horizon_eval] if horizon_eval < s.shape[0] else s.iloc[:-1]
    test_series = s.iloc[-horizon_eval:] if horizon_eval < s.shape[0] else s.iloc[-horizon_eval:]
    df_feat = create_features(train_series)
    if df_feat.shape[0] < 8:
        return {"type":"time_series","name":path.stem,"skipped":True,"reason":"insufficient_features"}
    X = df_feat.drop(columns=["value"], errors="ignore"); y = df_feat["value"]
    model, model_name = train_model(X, y)
    model_file = MODELS_DIR / f"{path.stem}_{model_name}_{TIMESTAMP}.pkl"
    joblib.dump({"model": model, "meta": {"source": str(path), "trained_at": TIMESTAMP}}, str(model_file))
    # fit on full series and forecast
    full_feat = create_features(s)
    try:
        model_full, _ = train_model(full_feat.drop(columns=["value"], errors="ignore"), full_feat["value"])
    except Exception:
        model_full = model
    fc = iterative_forecast(model_full, s, steps=horizon)
    # pseudo-eval
    mae = rmse = None
    try:
        if not test_series.empty:
            pseudo_feat = create_features(train_series)
            if pseudo_feat.shape[0] >= 1:
                pseudo_model, _ = train_model(pseudo_feat.drop(columns=["value"], errors="ignore"), pseudo_feat["value"])
                pseudo_fc = iterative_forecast(pseudo_model, train_series, steps=horizon_eval)
                if not pseudo_fc.empty:
                    actual = test_series["value"].iloc[:len(pseudo_fc)]
                    forecast_vals = pseudo_fc["forecast"].iloc[:len(actual)]
                    mae = float(mean_absolute_error(actual, forecast_vals))
                    rmse = float(math.sqrt(mean_squared_error(actual, forecast_vals)))
    except Exception:
        mae = rmse = None
    recent_mean = float(s["value"].iloc[-min(30, len(s)):].mean())
    forecast_mean = float(np.mean(fc["forecast"])) if not fc.empty else float("nan")
    fig_json = None
    if HAS_PLOTLY:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s["value"], name="Actual", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color="red")))
            fig.update_layout(title=path.stem + " Actual vs Forecast", xaxis_title="Date", yaxis_title="Value", height=520)
            fig_json = fig.to_json()
        except Exception:
            fig_json = None
    table_preview = s.reset_index().rename(columns={"index":"date","value":"value"}).tail(500).reset_index(drop=True).to_dict(orient="records")
    summary = {
        "type":"time_series",
        "name": path.stem,
        "n_points": int(s.shape[0]),
        "model_file": str(model_file.relative_to(HASIL)),
        "forecast": fc.reset_index().rename(columns={"index":"date","forecast":"forecast"}).to_dict(orient="records"),
        "fig_json": fig_json,
        "table": table_preview,
        "mae": mae,
        "rmse": rmse,
        "recent_mean": recent_mean,
        "forecast_mean": forecast_mean,
        "explanation": None
    }
    summary["explanation"] = auto_explanation(summary)
    return summary

def process_ratings(path: Path):
    df = pd.read_csv(path, engine="python")
    cols_map = {c.lower(): c for c in df.columns}
    if not {"user_id","item_id","rating"}.issubset(set(cols_map.keys())):
        return {"type":"ratings","name":path.stem,"error":"missing required columns","table": df.head(200).to_dict(orient="records")}
    df2 = df[[cols_map["user_id"], cols_map["item_id"], cols_map["rating"]]].copy()
    df2.columns = ["user_id","item_id","rating"]
    try:
        df2["user_id"] = df2["user_id"].astype(int)
        df2["item_id"] = df2["item_id"].astype(int)
    except Exception:
        pass
    df2["rating"] = pd.to_numeric(df2["rating"], errors="coerce").fillna(0.0)
    n_users = int(df2["user_id"].nunique())
    n_items = int(df2["item_id"].nunique())
    model_file = None
    sample_recs = []
    global UserKNN
    if (UserKNN is not None) and (n_users > 1) and (n_items > 0):
        try:
            cf = UserKNN(n_neighbors=min(10, max(1, n_users-1)))
            cf.fit(df2)
            model_file = MODELS_DIR / f"cf_userknn_{path.stem}_{TIMESTAMP}.pkl"
            joblib.dump({"model": cf, "meta": {"source": str(path), "trained_at": TIMESTAMP}}, str(model_file))
            users = df2["user_id"].unique()[:8]
            for u in users:
                recs = cf.recommend(int(u), k=10)
                sample_recs.append({"user_id": int(u), "recommendations": recs})
        except Exception as e:
            print(f"UserKNN training failed for {path}: {e}")
    return {"type":"ratings","name":path.stem,"n_users":n_users,"n_items":n_items,"model_file": str(model_file.relative_to(HASIL)) if model_file else None,"sample_recs": sample_recs,"table": df2.head(500).to_dict(orient="records")}

def process_items(path: Path, text_col):
    df = pd.read_csv(path, engine="python")
    item_col_candidates = [c for c in df.columns if c.lower() == "item_id"]
    item_col = item_col_candidates[0] if item_col_candidates else df.columns[0]
    if text_col is None or text_col not in df.columns:
        return {"type":"items","name":path.stem,"n_items":int(df.shape[0]),"error":"no text column detected","table": df.head(500).to_dict(orient="records")}
    items_df = df[[item_col, text_col]].copy()
    items_df.columns = ["item_id","text"]
    try:
        items_df["item_id"] = items_df["item_id"].astype(int)
    except Exception:
        pass
    items_df["text"] = items_df["text"].fillna("").astype(str)
    model_file = None
    similar = []
    global ContentRecommender
    if ContentRecommender is not None:
        try:
            content = ContentRecommender(max_features=5000)
            content.fit(items_df, text_col="text")
            model_file = MODELS_DIR / f"content_tfidf_{path.stem}_{TIMESTAMP}.pkl"
            joblib.dump({"model": content, "meta": {"source": str(path), "trained_at": TIMESTAMP}}, str(model_file))
            for iid in items_df["item_id"].tolist():
                try:
                    recs = content.recommend(int(iid), k=5)
                    for rid, sc in recs:
                        similar.append({"item_id": int(iid), "similar_item_id": int(rid), "score": float(sc)})
                except Exception:
                    continue
        except Exception as e:
            print(f"ContentRecommender training failed for {path}: {e}")
    return {"type":"items","name":path.stem,"n_items":int(items_df.shape[0]),"model_file": str(model_file.relative_to(HASIL)) if model_file else None,"similar": similar,"table": items_df.head(500).to_dict(orient="records")}

def process_table(path: Path):
    df = pd.read_csv(path, engine="python")
    return {"type":"table","name":path.stem,"n_rows":int(df.shape[0]),"table": df.head(500).to_dict(orient="records")}

# Build dashboard and JSON
def build_single_dashboard(summaries, html_path: Path, json_path: Path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)
    plots = [ {"name": s["name"], "json": s["fig_json"]} for s in summaries if s and s.get("type")=="time_series" and s.get("fig_json") ]
    overview = [ {"type": s.get("type"), "name": s.get("name"), "count": s.get("n_points", s.get("n_rows", s.get("n_items",""))), "model": s.get("model_file",""), "explanation": s.get("explanation","")} for s in summaries if s ]
    html = "<!doctype html><html><head><meta charset=\'utf-8\'/><meta name=\'viewport\' content=\'width=device-width,initial-scale=1\'/>"
    html += "<title>Unified Dashboard</title>"
    html += "<style>body{font-family:Arial,Helvetica,sans-serif;margin:12px;background:#f6f8fb} .grid{display:grid;grid-template-columns:320px 1fr;gap:12px}.card{background:#fff;padding:12px;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.06)} .sidebar{height:calc(100vh - 40px);overflow:auto}.file-item{padding:8px;border-radius:6px;margin-bottom:6px;cursor:pointer;border:1px solid #eee;background:#fff}.file-item:hover{background:#f0f8ff}.btn{padding:6px 8px;margin:4px;border-radius:6px;border:1px solid #ddd;background:#fff;cursor:pointer}.table-edit td{padding:6px;border:1px solid #ddd}</style>"
    html += "<script src=\'https://cdn.plot.ly/plotly-latest.min.js\'></script></head><body>"
    html += f"<h1>Unified Dashboard</h1><p>Generated: {TIMESTAMP}</p>"
    html += "<div class=\'grid\'><div class=\'card sidebar\'><h3>Files</h3><div id=\'fileList\'></div><hr/><div><button class=\'btn\' onclick=\'showOverview()\'>Overview</button><button class=\'btn\' onclick=\'showAllPlots()\'>All Plots</button></div></div>"
    html += "<div class=\'card\'><div id=\'mainArea\'></div></div></div>"
    html += "<script>\n"
    html += "const SUMMARIES = " + json.dumps(summaries) + ";\n"
    html += "const PLOTS = " + json.dumps(plots) + ";\n"
    html += "const OVERVIEW = " + json.dumps(overview) + ";\n"
    html += "</script>\n"
    html += """
<script>
function downloadText(filename, text) {
  const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click();
  setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 1500);
}
function renderFileList(){
  const fileList = document.getElementById('fileList');
  fileList.innerHTML = '';
  for(let i=0;i<SUMMARIES.length;i++){
    const s = SUMMARIES[i];
    if(!s) continue;
    const div = document.createElement('div'); div.className='file-item';
    div.innerText = s.type + ' — ' + s.name;
    div.onclick = ()=>{ renderDetail(i); };
    fileList.appendChild(div);
  }
}
function renderOverview(){
  let html = '<h3>Overview</h3><table border=1 width=100%><tr><th>Type</th><th>Name</th><th>Count</th><th>Model</th><th>Explanation</th></tr>';
  for(let i=0;i<OVERVIEW.length;i++){
    const r = OVERVIEW[i];
    html += `<tr><td>${r.type}</td><td>${r.name}</td><td>${r.count||''}</td><td>${r.model||''}</td><td>${r.explanation||''}</td></tr>`;
  }
  html += '</table>';
  document.getElementById('mainArea').innerHTML = html;
}
function showAllPlots(){
  let html = '';
  if(PLOTS.length===0){ document.getElementById('mainArea').innerHTML='<p>No plots available</p>'; return; }
  for(let i=0;i<PLOTS.length;i++){ const id='plot_'+i; html += `<h4>${PLOTS[i].name}</h4><div id="${id}" style="height:420px"></div>`; }
  document.getElementById('mainArea').innerHTML = html;
  for(let i=0;i<PLOTS.length;i++){
    try{ const fig = JSON.parse(PLOTS[i].json); Plotly.newPlot('plot_'+i, fig.data, fig.layout||{}, {responsive:true}); } catch(e){ const el = document.getElementById('plot_'+i); if(el) el.innerText='Plot error: '+e; }
  }
}
function renderDetail(idx){
  const s = SUMMARIES[idx];
  if(!s){ document.getElementById('mainArea').innerHTML = '<p>Empty</p>'; return; }
  let html = `<h3>${s.name} (${s.type})</h3>`;
  if(s.type==='time_series'){
    if(s.fig_json){ html += `<div id="plot_detail" style="height:420px"></div>`; }
    html += `<h4>Explanation</h4><div id="explain">${s.explanation||''}</div>`;
  }
  html += '<h4>Table (editable)</h4>';
  html += `<div><button class="btn" onclick="addRow(${idx})">Add row</button> <button class="btn" onclick="downloadTableCSV(${idx})">Download CSV</button> <button class="btn" onclick="downloadTableJSON(${idx})">Download JSON</button></div>`;
  html += `<div id="table_container_${idx}" style="max-height:480px;overflow:auto"></div>`;
  document.getElementById('mainArea').innerHTML = html;
  if(s.type==='time_series' && s.fig_json){
    try{ const fig = JSON.parse(s.fig_json); Plotly.newPlot('plot_detail', fig.data, fig.layout||{}, {responsive:true}); } catch(e){ document.getElementById('plot_detail').innerText = 'Plot error: '+e; }
  }
  renderEditableTable(idx);
}
function renderEditableTable(idx){
  const s = SUMMARIES[idx];
  const rows = s.table || [];
  const container = document.getElementById('table_container_'+idx);
  let html = '<table class="table-edit" border="0" style="border-collapse:collapse"><thead><tr>';
  if(rows.length===0){
    html += '<th>Empty</th></tr></thead><tbody></tbody></table>';
    container.innerHTML = html; return;
  }
  const keys = Object.keys(rows[0]);
  for(let k=0;k<keys.length;k++){ html += `<th>${keys[k]}</th>`; }
  html += '<th>Actions</th></tr></thead><tbody>';
  for(let r=0;r<rows.length;r++){
    html += '<tr>';
    for(let c=0;c<keys.length;c++){
      const val = rows[r][keys[c]]===null?'':rows[r][keys[c]];
      html += `<td contenteditable="true" data-row="${r}" data-col="${keys[c]}">${val}</td>`;
    }
    html += `<td><button class="btn" onclick="deleteRow(${idx},${r})">Delete</button></td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
  const tds = container.querySelectorAll('td[contenteditable="true"]');
  tds.forEach(td=>{
    td.addEventListener('input', function(e){
      const row = parseInt(this.getAttribute('data-row'));
      const col = this.getAttribute('data-col');
      SUMMARIES[idx].table[row][col] = this.innerText;
    });
  });
}
function addRow(idx){
  const s = SUMMARIES[idx];
  const table = s.table || [];
  let newRow = {};
  if(table.length>0){ const keys = Object.keys(table[0]); keys.forEach(k=> newRow[k]=''); } else { newRow = {"value":""}; }
  table.push(newRow); s.table = table; renderEditableTable(idx);
}
function deleteRow(idx, rowIdx){ const s = SUMMARIES[idx]; s.table.splice(rowIdx,1); renderEditableTable(idx); }
function downloadTableCSV(idx){ const s = SUMMARIES[idx]; const rows = s.table || []; if(rows.length===0){ alert('No rows to download'); return; } const keys = Object.keys(rows[0]); let csv = keys.join(',') + '\n'; for(let r=0;r<rows.length;r++){ const vals = keys.map(k=> { const v = rows[r][k]; if(v===null||v===undefined) return ''; const s = String(v).replace(/"/g,'""'); return '"' + s + '"'; }); csv += vals.join(',') + '\n'; } downloadText(s.name + '_table.csv', csv); }
function downloadTableJSON(idx){ const s = SUMMARIES[idx]; const jsonText = JSON.stringify(s.table || [], null, 2); downloadText(s.name + '_table.json', jsonText); }
document.addEventListener('DOMContentLoaded', function(){ renderFileList(); renderOverview(); });
</script>
"""
    # write html file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

# Main runner
def main(horizon=30, min_points=30):
    csvs = find_csv_files()
    if not csvs:
        print("No CSVs found under data/ - nothing to do.")
        return 0
    summaries = []
    for p in csvs:
        print("Processing:", p)
        try:
            df = pd.read_csv(p, engine="python")
        except Exception as e:
            print(f"[{p.stem}] read error: {e}")
            summaries.append(None)
            continue
        try:
            ctype, dc, vc = detect_csv_type(df)
            if ctype == "time_series":
                out = process_time_series(p, dc, vc, horizon=horizon, min_points=min_points)
            elif ctype == "ratings":
                out = process_ratings(p)
            elif ctype == "items":
                out = process_items(p, vc)
            else:
                out = process_table(p)
            summaries.append(out)
        except Exception as e:
            print(f"[{p.stem}] processing error: {e}")
            traceback.print_exc()
            summaries.append(None)
    # remove older dashboards (keep only new)
    for f in REPORTS_DIR.glob("report_dashboard_*.html"):
        try:
            f.unlink()
        except Exception:
            pass
    out_html = REPORTS_DIR / f"report_dashboard_{TIMESTAMP}.html"
    out_json = REPORTS_DIR / f"summaries_{TIMESTAMP}.json"
    build_single_dashboard(summaries, out_html, out_json)
    print("Dashboard written to:", out_html)
    print("Summary JSON written to:", out_json)
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--min_points", type=int, default=30)
    args = parser.parse_args()
    sys.exit(main(horizon=args.horizon, min_points=args.min_points))
'@

# Write pipeline file
$pyPath = Join-Path $hasil "pipeline_run.py"
Write-FileSafely -Path $pyPath -Content $pipeline_py -EncodingName "utf8"

# Optionally initialize git
if ($InitializeGit) {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    Push-Location $root
    if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) {
      git init
      git add --all
      git commit -m "Add pipeline_run.py: unified interactive dashboard (fixed imports)"
      Write-Host "Initialized git repository and committed files."
    } else {
      Write-Host "Git repo already exists."
    }
    Pop-Location
  } else {
    Write-Warning "git not found; skipping initialization."
  }
}

# Run pipeline
if ($RunNow) {
  $python = Find-Python
  Write-Host "Running pipeline: $python $pyPath"
  try {
    $outLog = Join-Path $hasil "proc_out_pipeline.log"
    $errLog = Join-Path $hasil "proc_err_pipeline.log"
    $proc = Start-Process -FilePath $python -ArgumentList @($pyPath) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Write-Host "Exit code:" $proc.ExitCode
    if (Test-Path $outLog) { Get-Content -Raw $outLog | Write-Host }
    if (Test-Path $errLog) { $err = Get-Content -Raw $errLog; if ($err) { Write-Host "Errors:"; Write-Host $err } }
  } catch {
    Write-Warning "Failed to run pipeline: $_"
  }
} else {
  Write-Host "Pipeline written to $pyPath (not executed because -RunNow not specified)."
}

Write-Host ""
Write-Host "Single dashboard will be available at: $notebooks\report_dashboard_<TIMESTAMP>.html"
Write-Host "Summary JSON at: $notebooks\summaries_<TIMESTAMP>.json"
```
